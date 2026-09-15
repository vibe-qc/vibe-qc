"""Opt-in live QVF checkpointing for :func:`vibeqc.run_job` /
:func:`vibeqc.run_periodic_job`.

For live job visualization, vibe-view wants to hot-reload a QVF *as a
calculation progresses* -- SCF convergence climbing, an optimization
trajectory growing, the geometry morphing toward the relaxed structure.
The runners normally write the QVF once, at the end. This module adds a
small handle that a runner drives to also refresh a *checkpoint* QVF on a
cadence, each snapshot carrying:

* ``provenance.run_status = "running"`` while the job is in flight, then
  ``"converged"`` (success) or ``"failed"`` (crash) at the end;
* a monotonic ``provenance.checkpoint.seq`` so a reader can order snapshots
  and tell a fresh one from a stale one without diffing bytes;
* ``partial: true`` on any still-growing section (e.g. an optimization
  ``trajectory``).

Every checkpoint write goes through :func:`vibeqc.output.formats.qvf.write_qvf`
with ``atomic=True`` (temp file + ``os.replace``) so a concurrent reader
never observes a half-written zip.

These provenance/section fields validate against the *current* QVF v1
manifest schema (``provenance`` is ``additionalProperties: true`` and
sections carry no ``additionalProperties: false``), so checkpointing does
not depend on the QVF v3 schema landing first.

Design contract
---------------

A checkpoint write must **never** take down the calculation it is
observing. Every write is wrapped so a filesystem hiccup or a transient
writer error is surfaced as a warning and swallowed -- the SCF keeps
running and the authoritative end-of-run output QVF is unaffected.
"""

from __future__ import annotations

import datetime as _dt
import os
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Optional, Union


def _now_iso() -> str:
    """UTC timestamp, ISO-8601 with a trailing ``Z``."""
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class QvfCheckpointer:
    """Drive periodic + terminal QVF snapshots to a checkpoint path.

    Parameters
    ----------
    path
        Where to write the checkpoint archive (the ``.qvf`` suffix is
        appended if absent, matching :func:`write_qvf`). ``None`` disables
        checkpointing entirely -- every method becomes a cheap no-op.
    every
        Refresh the checkpoint every ``every`` SCF / optimization
        iterations. ``0`` (or negative) disables the *cadence* snapshots;
        :meth:`snapshot` (initial frame) and :meth:`finalize` (terminal
        frame) still fire when a non-``None`` ``path`` is given, so a job
        with ``every=0`` but a ``path`` gets a start + end frame only.
    plan
        The :class:`~vibeqc.output.plan.OutputPlan` for the job, forwarded
        to :func:`write_qvf`.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike, None],
        every: int,
        *,
        plan: Any,
    ) -> None:
        self._path: Optional[Path] = Path(os.fspath(path)) if path else None
        self._every = int(every) if every else 0
        self._plan = plan
        self._seq = 0
        self._t0 = time.monotonic()

    # ----- state ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True when a checkpoint ``path`` was configured."""
        return self._path is not None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def seq(self) -> int:
        """The last sequence number written (0 before the first write)."""
        return self._seq

    def due(self, iteration: int) -> bool:
        """Whether a cadence snapshot is due at ``iteration``.

        False when checkpointing is disabled, ``every`` is 0, or the
        iteration is not a positive multiple of ``every``.
        """
        if not self.enabled or self._every <= 0:
            return False
        return iteration > 0 and iteration % self._every == 0

    # ----- writes --------------------------------------------------------

    def snapshot(
        self,
        *,
        scf_iteration: Optional[int] = None,
        energy_eh: Optional[float] = None,
        partial_sections: Any = None,
        **context: Any,
    ) -> None:
        """Write a running ("in flight") snapshot.

        Use for the initial geometry frame and for each cadence hit. The
        caller supplies whatever section context is available so far
        (``molecule=`` / ``system=`` at minimum; ``result=``,
        ``trajectory_frames=``, ``scf_history_data=`` when they exist).
        ``partial_sections`` flags still-growing sections (see
        :func:`write_qvf`).
        """
        if not self.enabled:
            return
        self._write(
            "running",
            scf_iteration=scf_iteration,
            energy_eh=energy_eh,
            partial_sections=partial_sections,
            **context,
        )

    def maybe_snapshot(
        self,
        iteration: int,
        *,
        energy_eh: Optional[float] = None,
        partial_sections: Any = None,
        **context: Any,
    ) -> bool:
        """Write a running snapshot only if :meth:`due` at ``iteration``.

        Returns ``True`` if a snapshot was written. Convenience wrapper for
        the SCF / optimization loop hooks.
        """
        if not self.due(iteration):
            return False
        self.snapshot(
            scf_iteration=iteration,
            energy_eh=energy_eh,
            partial_sections=partial_sections,
            **context,
        )
        return True

    def finalize(self, status: str, **context: Any) -> None:
        """Write the terminal snapshot with ``run_status=status``.

        ``status`` is ``"converged"`` (success) or ``"failed"`` (the job
        raised -- mirrors :meth:`ManifestUpdater.crash`). Always writes when
        checkpointing is enabled, regardless of ``every``, so the final
        checkpoint archive is always labeled and readers can stop watching.
        """
        if not self.enabled:
            return
        if status not in ("converged", "failed"):
            raise ValueError(
                "QvfCheckpointer.finalize: status must be 'converged' or "
                f"'failed', got {status!r}."
            )
        self._write(status, partial_sections=None, **context)

    # ----- internals -----------------------------------------------------

    def _write(
        self,
        status: str,
        *,
        scf_iteration: Optional[int] = None,
        energy_eh: Optional[float] = None,
        partial_sections: Any = None,
        **context: Any,
    ) -> None:
        # Import lazily so importing this module doesn't pull the (heavy)
        # QVF writer + its optional native deps before they're needed.
        from .formats.qvf import write_qvf

        self._seq += 1
        checkpoint: dict[str, Any] = {
            "seq": self._seq,
            "wall_time_s": time.monotonic() - self._t0,
            "written_at": _now_iso(),
        }
        if scf_iteration is not None:
            checkpoint["scf_iteration"] = int(scf_iteration)
        if energy_eh is not None:
            checkpoint["energy_eh"] = float(energy_eh)

        try:
            write_qvf(
                self._path,
                self._plan,
                atomic=True,
                run_status=status,
                checkpoint=checkpoint,
                partial_sections=partial_sections,
                **context,
            )
        except Exception as exc:  # noqa: BLE001 -- never tank the job
            # A checkpoint is a best-effort visualization aid. Surface the
            # failure but keep the calculation (and its authoritative
            # end-of-run output QVF) going.
            warnings.warn(
                f"QVF checkpoint write to {self._path} failed "
                f"(seq={self._seq}, status={status}): {exc!r}. The "
                f"calculation continues; the final output QVF is "
                f"unaffected.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Roll the seq back so the next successful write is still
            # monotonic-without-gaps from a reader's point of view.
            self._seq -= 1


class _CheckpointingProgressProxy:
    """Transparent proxy over a :class:`ProgressLogger`.

    Forwards every attribute to the wrapped logger, but on each
    ``iteration(n, **fields)`` call also invokes ``on_iteration(n, fields)``
    after the real log line is emitted. Used by the periodic runner so a
    checkpoint snapshot can fire as each Python-driven SCF cycle lands,
    without any driver having to know about checkpointing. The callback is
    fully guarded -- a checkpoint error can never disturb the SCF's own
    progress logging.
    """

    def __init__(self, inner: Any, on_iteration: Any) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_on_iteration", on_iteration)

    def iteration(self, n: int, **fields: Any) -> None:
        self._inner.iteration(n, **fields)
        try:
            self._on_iteration(n, fields)
        except Exception:  # noqa: BLE001 -- never disturb the SCF
            pass

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found on the proxy itself.
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)


def wrap_progress_for_checkpoints(plog: Any, on_iteration: Any) -> Any:
    """Return ``plog`` wrapped so per-iteration checkpoints fire.

    ``on_iteration`` is called as ``on_iteration(n, fields)`` after each
    ``plog.iteration(n, **fields)``; ``fields`` is the dict the driver
    passed (typically containing ``energy``). Returns the original ``plog``
    unchanged when ``on_iteration`` is falsy.
    """
    if not on_iteration:
        return plog
    return _CheckpointingProgressProxy(plog, on_iteration)


def normalize_partial_iterable(spec: Any) -> Optional[Union[bool, set]]:
    """Coerce a ``partial_sections`` argument to the shape write_qvf wants.

    ``None`` / falsy -> ``None``; ``True`` -> ``True``; an iterable of
    strings -> a ``set`` of them. Kept here so both runners share one
    interpretation.
    """
    if spec is None or spec is False:
        return None
    if spec is True:
        return True
    if isinstance(spec, str):
        return {spec}
    if isinstance(spec, Iterable):
        return {str(x) for x in spec}
    return None
