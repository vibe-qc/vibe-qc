"""local transport — run an external-SCF job in a local subprocess.

For laptop smoke tests and developer iteration: runs the command
*synchronously* in the work directory, so by the time ``submit``
returns the job is already finished. ``poll`` then just reports
the cached terminal state, and ``fetch`` is a no-op (the outputs
are already in ``work_dir`` — it's local).

This is intentionally the simple sibling of ``VqTransport``. The
production Goal 8 path is CRYSTAL14-on-compute-host via ``vq``; this
transport exists so the ``Transport`` abstraction is exercised by
≥ 2 implementations and so a one-off "does this .d12 even run"
check doesn't need the queue.

Per CLAUDE.md § 10 / vibe-basis discipline, the external program
is run as a subprocess — we never import it.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Callable, Optional, Sequence

from .base import JobHandle, Transport, TransportError

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _default_runner(
    argv: Sequence[str], cwd: Path
) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(  # noqa: S603 — argv built from known tokens
        list(argv),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


class LocalTransport(Transport):
    """Synchronous local-subprocess transport.

    Parameters
    ----------
    runner
        Injected ``runner(argv, cwd) -> CompletedProcess`` callable,
        for testing. Defaults to a wrapper over ``subprocess.run``.

    Notes
    -----
    Because ``submit`` runs the command synchronously, a ``submit``
    immediately followed by ``poll`` always returns a terminal
    state. The ``wait`` inherited from :class:`Transport` still
    works (it polls once, sees a terminal state, returns).
    """

    def __init__(self, *, runner: Optional[Runner] = None) -> None:
        self._runner: Runner = runner or _default_runner
        # job_id -> terminal state ("completed" / "failed")
        self._states: dict[str, str] = {}

    def submit(
        self,
        work_dir: Path | str,
        command: Sequence[str],
        *,
        cpus: Optional[int] = None,   # accepted for interface parity; unused
        wall_time_s: Optional[int] = None,  # ditto
        label: Optional[str] = None,
    ) -> JobHandle:
        work_dir = Path(work_dir)
        if not work_dir.is_dir():
            raise TransportError(f"work_dir does not exist: {work_dir}")
        job_id = f"local-{uuid.uuid4().hex[:12]}"
        proc = self._runner(command, work_dir)
        # Persist stdout/stderr next to the outputs so they're
        # fetchable the same way a remote job's would be.
        (work_dir / f"{job_id}.stdout").write_text(proc.stdout or "")
        (work_dir / f"{job_id}.stderr").write_text(proc.stderr or "")
        self._states[job_id] = "completed" if proc.returncode == 0 else "failed"
        return JobHandle(
            job_id=job_id,
            work_dir=work_dir,
            command=tuple(command),
            label=label,
        )

    def poll(self, handle: JobHandle) -> str:
        state = self._states.get(handle.job_id)
        if state is None:
            raise TransportError(
                f"unknown job id {handle.job_id!r} — not submitted by "
                f"this LocalTransport instance"
            )
        return state

    def fetch(self, handle: JobHandle, dest: Path | str) -> Path:
        # Local jobs leave their outputs in work_dir already; there's
        # nothing to download. Return work_dir so callers get a
        # consistent "where are the outputs" answer across transports.
        del dest  # unused — local outputs aren't copied anywhere
        return handle.work_dir
