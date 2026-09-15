"""Transport base classes — how an external-SCF job reaches a CPU.

A *transport* takes a pre-populated work directory + a command and
runs it somewhere, then brings the outputs back. Backends
(``backends/crystal.py`` etc.) sit above transports: a backend
knows how to *emit* an input file and *parse* an output file; it
hands the transport a ``(work_dir, command)`` pair and doesn't
care whether the SCF ran in a local subprocess or on a remote
``vq`` daemon.

The interface is deliberately small:

* ``submit(work_dir, command, ...) -> JobHandle``
* ``poll(handle) -> str``           (current state, lowercased)
* ``wait(handle, ...) -> str``      (concrete; polls until terminal)
* ``fetch(handle, dest) -> Path``   (download outputs; return dir)

Concrete implementations:

* ``transports.vq.VqTransport``    — submit through the vibe-queue
  CLI to a remote daemon (compute-host).
* ``transports.local.LocalTransport`` — synchronous subprocess on
  the local machine; for laptop smoke tests.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence


# vq reports job state on the ``state:`` line of ``vq status <id>``.
# Anything not in TERMINAL_STATES is treated as still-in-flight.
TERMINAL_STATES = frozenset({"completed", "failed", "killed", "interrupted"})
SUCCESS_STATES = frozenset({"completed"})


class TransportError(RuntimeError):
    """A transport CLI call failed (non-zero exit, unparseable output)."""


class JobSubmitError(TransportError):
    """``submit`` could not extract a job id from the transport's output."""


class JobTimeoutError(TransportError):
    """``wait`` exceeded its timeout before the job reached a terminal state."""


@dataclass(frozen=True)
class JobHandle:
    """Opaque handle to a submitted job.

    ``command`` is stored as a tuple so the handle stays hashable /
    frozen; it's the exact argv that was shipped (minus the
    transport-specific submit wrapper).
    """
    job_id: str
    work_dir: Path
    command: tuple[str, ...]
    label: Optional[str] = None


@dataclass(frozen=True)
class JobResult:
    """Terminal result of a job after ``wait`` + ``fetch``."""
    handle: JobHandle
    state: str                 # lowercased terminal state
    output_dir: Path           # directory the outputs were fetched into

    @property
    def ok(self) -> bool:
        """True iff the job reached a success state (``completed``)."""
        return self.state in SUCCESS_STATES


class Transport(abc.ABC):
    """Abstract transport: ship a command to a CPU, wait, fetch outputs."""

    @abc.abstractmethod
    def submit(
        self,
        work_dir: Path | str,
        command: Sequence[str],
        *,
        cpus: Optional[int] = None,
        wall_time_s: Optional[int] = None,
        label: Optional[str] = None,
    ) -> JobHandle:
        """Submit ``command`` to run inside ``work_dir``.

        ``work_dir`` must already contain every input file the
        command references (the backend/recipe populates it before
        calling). Returns a :class:`JobHandle`.
        """

    @abc.abstractmethod
    def poll(self, handle: JobHandle) -> str:
        """Return the job's current state, lowercased.

        Terminal states are in :data:`TERMINAL_STATES`; everything
        else means still-in-flight.
        """

    @abc.abstractmethod
    def fetch(self, handle: JobHandle, dest: Path | str) -> Path:
        """Download the job's outputs under ``dest``; return the dir."""

    # ------------------------------------------------------------------
    # Concrete: wait is just a poll-loop, identical for every transport.
    # sleep / clock are injectable so tests don't actually sleep.
    # ------------------------------------------------------------------
    def wait(
        self,
        handle: JobHandle,
        *,
        timeout_s: float = 86_400.0,
        poll_interval_s: float = 30.0,
        sleep: Optional[Callable[[float], None]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> str:
        """Poll ``handle`` until it reaches a terminal state.

        Returns the terminal state string. Raises
        :class:`JobTimeoutError` if ``timeout_s`` elapses first.

        ``sleep`` and ``clock`` are injectable for testing — pass a
        no-op ``sleep`` and a fake ``clock`` to exercise the loop
        without real wall-time.
        """
        _sleep = sleep or time.sleep
        _clock = clock or time.monotonic
        start = _clock()
        while True:
            state = self.poll(handle)
            if state in TERMINAL_STATES:
                return state
            if _clock() - start > timeout_s:
                raise JobTimeoutError(
                    f"job {handle.job_id!r} still in state {state!r} "
                    f"after {timeout_s:.0f}s"
                )
            _sleep(poll_interval_s)

    def run(
        self,
        work_dir: Path | str,
        command: Sequence[str],
        dest: Path | str,
        *,
        cpus: Optional[int] = None,
        wall_time_s: Optional[int] = None,
        label: Optional[str] = None,
        timeout_s: float = 86_400.0,
        poll_interval_s: float = 30.0,
        sleep: Optional[Callable[[float], None]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> JobResult:
        """Convenience: submit → wait → fetch in one call.

        Returns a :class:`JobResult`. The caller still has to parse
        the fetched output file with the relevant backend — the
        transport is engine-agnostic and doesn't know what a
        "converged SCF" looks like.
        """
        handle = self.submit(
            work_dir, command,
            cpus=cpus, wall_time_s=wall_time_s, label=label,
        )
        state = self.wait(
            handle,
            timeout_s=timeout_s, poll_interval_s=poll_interval_s,
            sleep=sleep, clock=clock,
        )
        output_dir = self.fetch(handle, dest)
        return JobResult(handle=handle, state=state, output_dir=output_dir)
