"""``OutputWriter`` -- the per-job coordinator.

A single object instantiated by the molecular or periodic runner at job start. It

* owns the path stem and the :class:`~vibeqc.output.OutputPlan`;
* owns the :class:`~vibeqc.output.manifest.ManifestUpdater` for
  ``{stem}.system``;
* dispatches plan roles to single-shot format writers and records successful
  paths immediately;
* exposes ``record(path, *, wall_time_s=None)`` for stream-style and
  runtime-owned outputs that cannot use single-shot role dispatch.

Public API
----------

``OutputWriter(plan, *, record_hostname=True)``
    Construct from a built :class:`OutputPlan`. The constructor writes
    the initial ``{stem}.system`` (status ``"running"``, plan declared,
    empty outputs).

``OutputWriter.record(path, *, wall_time_s=None)``
    Mark ``path`` as written; rewrites ``.system`` atomically.

``OutputWriter.dispatch_role(role, **context)``
    Run the matching registered adapters and record each successful file.

``OutputWriter.finish(*, wall_seconds=None)``
    End-of-job success path. Flips status to ``"complete"`` only after every
    guaranteed plan row was written; otherwise records a crashed manifest and
    raises :class:`~vibeqc.output.IncompleteOutputError`.

``OutputWriter.crash(*, wall_seconds=None)``
    End-of-job failure path. Flips status to ``"crashed"``. Called by
    the crash-dump path in :mod:`vibeqc.runner` before the exception
    re-raises.

``OutputWriter.context()``
    Context-manager helper that calls :meth:`finish` on clean exit and
    :meth:`crash` on exception. Use when wrapping a single job.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from .dispatch import Dispatcher, default_dispatcher
from .manifest import FileOutcome, ManifestUpdater
from .plan import OutputPlan, PlannedFile


__all__ = ["OutputWriter"]


class OutputWriter:
    """Per-job output coordinator.

    It owns the plan, manifest updater, and job-start wall clock. Single-shot
    format writers are reached through :meth:`dispatch_role`; stream-style
    log/performance writers and the crash path retain explicit lifecycle
    integration.
    """

    def __init__(
        self,
        plan: OutputPlan,
        *,
        record_hostname: bool = True,
        wall_seconds: float = 0.0,
        extra_run_fields: dict[str, Any] | None = None,
    ) -> None:
        self._plan = plan
        self._manifest = ManifestUpdater(
            plan,
            record_hostname=record_hostname,
            wall_seconds=wall_seconds,
            extra_run_fields=extra_run_fields,
        )
        self._t0 = time.monotonic()

    # ------------------------------------------------------------------ #
    # Public accessors                                                   #
    # ------------------------------------------------------------------ #

    @property
    def plan(self) -> OutputPlan:
        return self._plan

    @property
    def stem(self) -> Path:
        return self._plan.stem

    @property
    def manifest_path(self) -> Path:
        return self._manifest.path

    @property
    def status(self) -> str:
        return self._manifest.status

    def outcomes(self) -> tuple[FileOutcome, ...]:
        return self._manifest.outcomes()

    def wall_seconds(self) -> float:
        return time.monotonic() - self._t0

    # ------------------------------------------------------------------ #
    # Mutating operations                                                #
    # ------------------------------------------------------------------ #

    def record(
        self,
        path: os.PathLike | str,
        *,
        wall_time_s: float | None = None,
    ) -> None:
        """Record that ``path`` was just written. Atomically rewrites
        the manifest with the new ``[[outputs.files]]`` row.

        ``wall_time_s`` defaults to "now minus job-start" so callers
        don't have to thread their own clocks unless they care about a
        sub-phase timing.
        """
        if wall_time_s is None:
            wall_time_s = self.wall_seconds()
        # Always also update the running wall clock -- cheap and lets vq
        # see liveness even between recorded files.
        self._manifest.update_wall_seconds(self.wall_seconds())
        self._manifest.mark_written(path, wall_time_s=wall_time_s)

    def set_citations(self, entries: list[dict[str, Any]]) -> None:
        """Record the job's assembled citation provenance into the
        ``[citations]`` section of ``{stem}.system`` (CLAUDE.md
        Sec.8.3/Sec.8.5 -- the full ``.citations`` view, every entry regardless
        of its ``print`` flag). ``entries`` is a list of flat scalar
        dicts; see :func:`vibeqc.runner._citation_manifest_rows`."""
        self._manifest.set_citations(entries)

    def set_basis_library(
        self,
        *,
        resolved_root: str = "",
        fetched: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    ) -> None:
        """Record basis-data provenance into ``[basis_library]`` of
        ``{stem}.system``: the library root libint actually read, and one
        row per basis rendered on demand from the Basis Set Exchange."""
        self._manifest.set_basis_library(
            resolved_root=resolved_root, fetched=fetched
        )

    def update_run_fields(self, fields: dict[str, Any]) -> None:
        """Merge late-bound scalar reproducibility fields into ``[run]``."""
        self._manifest.update_run_fields(fields)

    def update_progress(self, **fields: object) -> None:
        """Stamp live calculation progress into the ``[progress]``
        section of ``{stem}.system``.  See
        :meth:`vibeqc.output.manifest.ManifestUpdater.update_progress`."""
        self._manifest.update_progress(**fields)

    def finish(self, *, wall_seconds: float | None = None) -> None:
        """End-of-job success path. Stamps the finish timestamp and
        flips the status flag to ``"complete"`` only when every guaranteed
        plan row has been written."""
        if wall_seconds is None:
            wall_seconds = self.wall_seconds()
        self._manifest.finish(wall_seconds=wall_seconds)

    def crash(self, *, wall_seconds: float | None = None) -> None:
        """End-of-job failure path. Flips status to ``"crashed"``.

        The crash-dump writer still produces ``{stem}.dump`` separately;
        this is the manifest-side bookkeeping.
        """
        if wall_seconds is None:
            wall_seconds = self.wall_seconds()
        self._manifest.crash(wall_seconds=wall_seconds)

    # ------------------------------------------------------------------ #
    # Role-driven dispatch                                                #
    # ------------------------------------------------------------------ #

    def dispatch_role(
        self,
        role: str,
        *,
        dispatcher: Optional[Dispatcher] = None,
        skip_conditional: bool = True,
        only_format: str | None = None,
        only_path: os.PathLike | str | None = None,
        runtime_path: os.PathLike | str | None = None,
        runtime_format: str | None = None,
        runtime_description: str | None = None,
        raise_on_error: bool = False,
        **context: Any,
    ) -> list[Path]:
        """Dispatch every :class:`PlannedFile` of the given role
        through the registered writer in ``dispatcher`` (or the
        default dispatcher when ``None``). Records each successful
        write in the manifest's ``[[outputs.files]]``. Returns the
        list of paths actually written (None entries from the
        dispatcher are filtered out).

        ``only_format`` and ``only_path`` target one subset of a role's
        declared rows.  ``runtime_path`` + ``runtime_format`` dispatch one
        runtime-conditional artefact that could not be declared before the
        calculation (for example an NTO molden file or ``.opt.xyz``); its
        successful path is appended to the manifest by :meth:`record`.

        ``raise_on_error`` preserves the dispatcher's fail-soft default when
        false.  Runners pass true inside their existing per-writer
        ``try/except`` blocks so the established user warning and manifest
        failure policies still see the original exception.

        ``**context`` is forwarded verbatim to every adapter, so a
        caller passes the union of all adapters' needs and each
        adapter pulls what it cares about (via ``**_`` for the
        rest). Typical call::

            writer.dispatch_role(
                "orbitals",
                result=result, basis=basis_obj, molecule=molecule,
                title=output_stem.name,
            )

        With ``skip_conditional=True`` (the default) only
        ``always=True`` plan rows are dispatched -- conditional ones
        like the crash dump are caller-driven (the exception path
        fires them explicitly). Pass ``skip_conditional=False`` to
        sweep everything.
        """
        if runtime_path is not None and runtime_format is None:
            raise ValueError("runtime_format is required with runtime_path")
        if runtime_path is not None and only_path is not None:
            raise ValueError("runtime_path and only_path are mutually exclusive")

        d = dispatcher or default_dispatcher()
        if runtime_path is not None:
            candidates = (
                PlannedFile(
                    role=role,  # type: ignore[arg-type]
                    path=Path(os.fspath(runtime_path)),
                    format=runtime_format,  # type: ignore[arg-type]
                    always=True,
                    description=(
                        runtime_description
                        or f"Runtime-conditional {role} artefact."
                    ),
                ),
            )
        else:
            candidates = self._plan.files

        target_path = (
            Path(os.fspath(only_path)) if only_path is not None else None
        )
        dispatch_context = dict(context)
        dispatch_context.setdefault("plan", self._plan)
        dispatch_context.setdefault("_dispatch_cache", {})
        written: list[Path] = []
        for pf in candidates:
            if pf.role != role:
                continue
            if skip_conditional and not pf.always:
                continue
            if only_format is not None and pf.format != only_format:
                continue
            if target_path is not None and pf.path != target_path:
                continue
            path = d.dispatch_planned_file(
                pf,
                stem=self.stem,
                raise_on_error=raise_on_error,
                **dispatch_context,
            )
            if path is not None:
                self.record(path)
                written.append(path)
        return written

    def downgrade_planned(self, role: str) -> None:
        """Mark all planned files of the given role as no longer
        guaranteed.  ``finish()`` will not raise
        :class:`IncompleteOutputError` for unwritten rows of this role.

        Use when a runtime physics condition (non-\u0393 periodic
        orbitals, unavailable population for a basis-free route) makes
        a declared-always artefact legitimately unavailable after the
        SCF succeeded."""
        for planned in self._plan.files:
            if planned.role == role and planned.always:
                self._manifest.downgrade_planned(planned.path)

    def mark_self_excluded(self, path: os.PathLike | str) -> None:
        """Mark a written artefact with a self-excluded checksum.

        Forwarded to ``self._manifest.mark_self_excluded(path)``."""
        self._manifest.mark_self_excluded(path)

    # ------------------------------------------------------------------ #
    # Context manager                                                    #
    # ------------------------------------------------------------------ #

    @contextlib.contextmanager
    def context(self) -> Iterator["OutputWriter"]:
        """Use as ``with writer.context() as w: ...``. Calls
        :meth:`finish` on clean exit, :meth:`crash` on exception. The
        exception is re-raised -- this is bookkeeping only."""
        try:
            yield self
        except Exception:
            self.crash()
            raise
        else:
            self.finish()
