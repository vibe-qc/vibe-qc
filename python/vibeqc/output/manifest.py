"""Extended ``{stem}.system`` manifest -- adds ``[plan]`` and ``[outputs]``.

The pre-Phase-O1 manifest (see :mod:`vibeqc.system_info`) carries the
runtime environment: ``[vibeqc]``, ``[host]``, ``[cpu]``, ``[memory]``,
``[python]``, ``[libraries]``, ``[validation]``, ``[run]``. The output
module layers two additional sections on top:

* ``[plan]`` -- written exactly once, at job start, before any compute.
  Mirrors the contents of :class:`vibeqc.output.OutputPlan` and includes
  an array-of-tables ``[[plan.files]]`` with one row per declared
  artefact.

* ``[outputs]`` -- rewritten in place every time a writer reports
  completion of one of the declared files. Carries the run-time status
  flag (``"running"`` / ``"complete"`` / ``"crashed"``), the finish
  timestamp, and one ``[[outputs.files]]`` row per declared artefact with
  its write/checksum state and wall-time-since-job-start. Ordinary written
  artefacts carry size + SHA-256; the manifest's own row is explicitly
  self-excluded because a file cannot truthfully contain its final digest.

The manifest is rewritten atomically (write-to-tmp + ``os.replace``)
on every update so a job killed mid-update never leaves a half-written
``.system`` for ``vq`` to misread. The fixed-shape rule from
:mod:`vibeqc.system_info` is preserved -- sections never disappear and
keys are never renamed; values fall back to documented sentinel strings
when unknown.

Public API
----------

``ManifestUpdater``
    A small mutable handle that owns the path to ``{stem}.system`` and
    knows how to write it. Instantiated by
    :class:`vibeqc.output.OutputWriter` at job start; not usually
    constructed directly.

``FileOutcome``
    Per-file runtime record (was the file written? how big? what checksum
    state applies?). Stored on the ``ManifestUpdater``; emitted into
    ``[[outputs.files]]``.

``write_initial_manifest(...)``
    One-shot helper: write a complete manifest with ``[plan]``,
    ``[outputs].status = "running"``, and the runtime-env sections, at
    job start. Convenience over building a ``ManifestUpdater`` for the
    very-first write.

The on-disk TOML is produced by a small hand-rolled emitter that
supports the few extra types we need beyond what :mod:`vibeqc.system_info`
emits (arrays of inline tables for ``[[plan.files]]`` and
``[[outputs.files]]``, nested-dot section headers, ``""`` strings for
optional-functional fields).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .formats.system_info import system_info
from .plan import OutputPlan, PlannedFile
from ._text_safety import toml_escape_str
from ._stem_paths import stem_sibling

__all__ = [
    "FileOutcome",
    "IncompleteOutputError",
    "ManifestUpdater",
    "ManifestStatus",
    "write_initial_manifest",
]


ManifestStatus = str  # "running" | "complete" | "crashed" | "dry_run"


class IncompleteOutputError(RuntimeError):
    """Finalization was refused because guaranteed files are missing.

    The manifest is durably rewritten with ``status = "crashed"`` and a
    failed outcome for every still-pending guaranteed artefact before this
    exception is raised.  Conditional plan rows are not part of this gate.
    """

    def __init__(self, files: tuple[PlannedFile, ...]) -> None:
        self.files = files
        self.paths = tuple(planned.path for planned in files)
        rendered = ", ".join(str(path) for path in self.paths)
        super().__init__(
            "output finalization refused: guaranteed artefact(s) were not "
            f"written: {rendered}"
        )


# ---------------------------------------------------------------------- #
# Per-file runtime record                                                #
# ---------------------------------------------------------------------- #


@dataclass
class FileOutcome:
    """Post-hoc record of one written artefact.

    Mutable on purpose -- the ``ManifestUpdater`` mutates the ``written`` /
    ``bytes`` / ``sha256`` / ``checksum_status`` / ``wall_time_s`` fields as
    writers report completion. The matching pre-flight declaration is the
    immutable :class:`vibeqc.output.PlannedFile`.
    """

    path: Path
    written: bool = False
    bytes: int | None = None
    sha256: str | None = None
    checksum_status: str = "pending"
    wall_time_s: float | None = None
    error: str | None = None

    def to_toml_table(self) -> dict[str, Any]:
        """Return a plain mapping suitable for the
        ``[[outputs.files]]`` array-of-tables emitter."""
        out: dict[str, Any] = {
            "path": str(self.path),
            "written": bool(self.written),
        }
        # Optional fields: emit "" / 0 sentinels rather than omit, so
        # the fixed-shape rule holds for downstream parsers.
        out["bytes"] = int(self.bytes) if self.bytes is not None else 0
        out["sha256"] = str(self.sha256) if self.sha256 is not None else ""
        out["checksum_status"] = str(self.checksum_status)
        out["wall_time_s"] = (
            float(self.wall_time_s) if self.wall_time_s is not None else 0.0
        )
        # When a file was declared but not written, surface the error.
        out["error"] = str(self.error) if self.error is not None else ""
        return out


# ---------------------------------------------------------------------- #
# Updater                                                                #
# ---------------------------------------------------------------------- #


class ManifestUpdater:
    """Owns ``{stem}.system`` for the lifetime of a job.

    Lifecycle:

    1. Construct with the :class:`OutputPlan` at job start; the
       constructor writes the initial manifest (``status = "running"``,
       all declared files in ``[[plan.files]]``, and pending rows in
       ``[[outputs.files]]``).
    2. Call :meth:`mark_written` each time a writer finishes producing
       one of the declared files; the manifest is rewritten atomically.
    3. Call :meth:`finish` (or :meth:`crash`) at job end to flip the
       status flag and stamp ``finished_at_iso``.

    Thread-safe -- every rewrite acquires an internal lock so concurrent
    writers (e.g. progress logger + perf writer + structured logger)
    don't race on the file.
    """

    def __init__(
        self,
        plan: OutputPlan,
        *,
        record_hostname: bool = True,
        wall_seconds: float = 0.0,
        estimate_bytes: int | None = None,
        extra_run_fields: dict[str, Any] | None = None,
    ) -> None:
        self._plan = plan
        self._record_hostname = record_hostname
        # Optional scalar fields merged into the [run] section (e.g. the periodic
        # SCF's exchange-q=0 finite-size convention, the §0.5 reproducibility
        # gate). Invariant across a run; None leaves [run] as the fixed shape.
        self._extra_run_fields: dict[str, Any] = dict(extra_run_fields or {})
        self._lock = threading.Lock()
        self._status: ManifestStatus = "running"
        self._finished_at_iso: str = ""
        self._wall_seconds = float(wall_seconds)
        # Optional peak-memory estimate (bytes, configured headroom already
        # applied) appended to the [memory] section. Set only by the
        # VIBEQC_DRY_RUN_ESTIMATE dry-run path (see
        # vibeqc.output.dry_run_manifest); None for normal runs, which
        # leaves the [memory] section unchanged. Read by `vq submit
        # auto` as [memory].estimate_bytes for RAM-fit placement.
        self._estimate_bytes: int | None = (
            int(estimate_bytes) if estimate_bytes is not None else None
        )
        # Live [progress] section — updated per SCF iteration.  When
        # None (the default) the section is omitted; set to a non-empty
        # dict to make it appear.  Fields: phase, iteration, energy_eh,
        # gradient_norm, diis_subspace.
        self._progress: dict[str, object] | None = None
        # Optional [hessian] section -- which potential-energy surface a
        # requested Hessian was built on, and whether that surface is the
        # method the caller asked for. Set by run_job through
        # ``set_hessian`` when hessian=True; None omits the section, so a
        # job that ran no Hessian gains no empty stub.
        self._hessian: dict[str, object] | None = None
        # One outcome per declared file, keyed by path for fast update.
        self._outcomes: dict[Path, FileOutcome] = {
            f.path: FileOutcome(path=f.path) for f in plan.files
        }
        # Full assembled-citation provenance (CLAUDE.md Sec.8.3/Sec.8.5: the
        # *full* .citations view -- every entry regardless of its `print`
        # flag -- feeds the .system manifest). Seeded empty; run_job calls
        # ``set_citations`` once the end-of-run assemble() has fired its
        # routes. Each row is a flat dict of scalar fields (see
        # ``runner._citation_manifest_rows``).
        self._citations: list[dict[str, Any]] = []
        # Which basis library actually answered, and anything rendered on
        # demand into it. Seeded empty; run_job calls ``set_basis_library``
        # once the resolution for the job is settled. The resolved root is
        # worth recording even when nothing was fetched: vibe-qc prefers a
        # build overlay over the committed tree, and a run that silently
        # read a stale overlay is otherwise indistinguishable from one that
        # did not (see ``vibeqc._warn_if_basis_overlay_is_stale``).
        self._basis_library: dict[str, Any] = {}
        self._declared_at_iso = _now_iso()
        # Two spellings of the manifest path, deliberately kept apart:
        #
        # * ``_declared_path`` is what the plan declares and what every
        #   ``[[plan.files]]`` / ``[[outputs.files]]`` row renders -- a
        #   bare ``pilot.system`` for ``run_job(output="pilot")``.
        # * ``_path`` is where the bytes go. A relative stem is anchored
        #   to the directory the job *started* in, once, here.
        #
        # Without the anchor every rewrite re-resolved the relative path
        # against the cwd current *at that moment*, so any rewrite that
        # fired after the process had changed directory (a progress
        # heartbeat, ``finish()``, a stale handler) dropped a second
        # copy of the manifest into whatever directory was current --
        # the checkout root, in vibe-qc#127 -- while the ``.out``
        # written synchronously stayed put. Every on-disk probe and
        # write below goes through :meth:`_on_disk` for the same reason.
        self._declared_path = stem_sibling(plan.stem, ".system")
        try:
            self._job_cwd: Path | None = Path.cwd()
        except OSError:
            # The process cwd is gone. Nothing relative can be written
            # from here anyway (the first write fails, as it always did);
            # an absolute stem never needed the anchor and must keep
            # working, so record "no anchor" instead of failing here.
            self._job_cwd = None
        self._path = self._on_disk(self._declared_path)
        # Snapshot the runtime-env probe ONCE. The coordinator
        # rewrites the manifest after every artefact lands (N times
        # per job), the [vibeqc] / [host] / [cpu] / [memory] /
        # [python] / [libraries] / [validation] sections are
        # invariant across a single run, so probing them once
        # (system_info() does subprocess sysctl / platform calls)
        # rather than per-rewrite keeps the lifecycle cheap.
        self._system_info: dict[str, Any] = system_info(
            record_hostname=record_hostname,
        )
        # Paths whose planned rows were downgraded from guaranteed to
        # conditional during the run (e.g. Molden for non-Γ periodic
        # orbitals).  finish() skips them in the completeness check.
        self._downgraded_paths: set[Path] = set()
        self._write()

    # -- public state mutators ----------------------------------------- #

    def mark_written(
        self,
        path: os.PathLike | str,
        *,
        wall_time_s: float | None = None,
    ) -> None:
        """Record that ``path`` has finished being written. Reads its
        size + sha256 from disk and rewrites the manifest atomically, except
        that the manifest itself is explicitly marked ``self-excluded``.

        ``path`` does not need to have been declared in the plan -- undeclared
        files are appended to ``[[outputs.files]]`` with ``written=true``
        so they are still surfaced to ``vq``. The CI gate that asserts
        "no undeclared outputs" runs against the *plan*, not here.
        """
        p = Path(os.fspath(path))
        on_disk = self._on_disk(p)
        is_self = _same_path(on_disk, self._path)
        size: int | None = None
        digest: str | None = None
        if not is_self:
            size, digest = _stat_and_hash(on_disk)
        with self._lock:
            # The self row is always the declared one, however the caller
            # spelled the manifest path (``updater.path`` is absolute for a
            # relative stem); a plan without a manifest row must not gain
            # an absolute one.
            outcome = self._outcome_for_path_unlocked(
                self._declared_path if is_self else p
            )
            outcome.written = True
            outcome.error = None
            if is_self:
                outcome.bytes = None
                outcome.sha256 = None
                outcome.checksum_status = "self-excluded"
            else:
                outcome.bytes = size
                outcome.sha256 = digest
                outcome.checksum_status = (
                    "sha256" if digest is not None else "unavailable"
                )
            if wall_time_s is not None:
                outcome.wall_time_s = float(wall_time_s)
            self._write_unlocked()

    def mark_failed(
        self,
        path: os.PathLike | str,
        error: str,
    ) -> None:
        """Record that ``path`` was declared but failed to write. The
        row's ``written`` stays ``False`` and the ``error`` field
        carries a short ``ExceptionType: message`` string so downstream
        ``.system`` readers (``vq fetch``, ``vibeqc outputs``) can see
        the artefact was *declared* but did not land -- instead of the
        manifest silently leaving it as never-written-because-not-yet-
        produced.

        Undeclared paths are appended just like :meth:`mark_written`
        does, so a writer that crashed before its row was seeded still
        gets a manifest entry.
        """
        p = Path(os.fspath(path))
        with self._lock:
            outcome = self._outcome_for_path_unlocked(p)
            outcome.written = False
            outcome.bytes = None
            outcome.sha256 = None
            outcome.checksum_status = "failed"
            outcome.error = str(error)
            self._write_unlocked()

    def downgrade_planned(self, path: os.PathLike | str) -> None:
        """Mark a planned file as no longer guaranteed.

        After this call, :meth:`finish` will not raise
        :class:`IncompleteOutputError` if the file is unwritten.
        Use when a runtime condition (e.g. non-\u0393 periodic
        orbital export) makes a declared-always artefact legitimately
        unavailable after the SCF has already succeeded.
        """
        p = Path(os.fspath(path))
        with self._lock:
            self._downgraded_paths.add(p)
            # Also update the outcome to show the downgrade.
            outcome = self._outcome_for_path_unlocked(p)
            if outcome.error is None and not outcome.written:
                outcome.error = "downgraded: not required for this route"
            self._write_unlocked()

    def mark_self_excluded(
        self,
        path: os.PathLike | str,
    ) -> None:
        """Mark a written artefact with ``checksum_status = "self-excluded"``.

        Use when a file is rewritten after the initial manifest snapshot
        (e.g. a QVF container that embeds the ``.system`` manifest after
        the writer has already stamped it). The row records ``written=True``
        but omits the ``sha256`` / ``bytes`` claim to avoid shipping a
        stale checksum.
        """
        p = Path(os.fspath(path))
        with self._lock:
            outcome = self._outcome_for_path_unlocked(p)
            outcome.written = True
            outcome.bytes = None
            outcome.sha256 = None
            outcome.checksum_status = "self-excluded"
            self._write_unlocked()

    def set_citations(self, entries: list[dict[str, Any]]) -> None:
        """Record the assembled citation provenance into the manifest's
        ``[citations]`` section and rewrite atomically.

        ``entries`` is the *full* assembled list (both ``print = true``
        and ``print = false`` rows) as flat scalar dicts -- see
        :func:`vibeqc.runner._citation_manifest_rows`. Called once near
        end-of-run, after ``CitationDatabase.assemble(...)`` has walked
        the job's routes (CLAUDE.md Sec.8.3/Sec.8.5)."""
        with self._lock:
            self._citations = [dict(e) for e in entries]
            self._write_unlocked()

    def set_basis_library(
        self,
        *,
        resolved_root: str = "",
        fetched: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    ) -> None:
        """Record basis-data provenance into ``[basis_library]``.

        ``resolved_root`` is the directory libint actually read for this
        job. ``fetched`` carries one flat scalar dict per basis rendered on
        demand from the Basis Set Exchange (see
        :mod:`vibeqc.basis_fetch`); it is empty for the ordinary case where
        every basis came from the bundle.

        Branch on ``fetched_count``, not on ``fetched``: the emitter drops an
        empty array-of-tables, so on an ordinary run the ``fetched`` key is
        absent rather than ``[]``. Same shape as ``[citations]`` with no
        entries. The section itself is always present.
        """
        with self._lock:
            self._basis_library = {
                "resolved_root": str(resolved_root),
                "fetched_count": len(fetched),
                "fetched": [dict(e) for e in fetched],
            }
            self._write_unlocked()

    def set_hessian(self, fields: dict[str, Any]) -> None:
        """Record the Hessian's surface into ``[hessian]`` and rewrite.

        ``fields`` comes from
        :func:`vibeqc.output.hessian_surface_manifest_fields` -- flat
        scalars naming the surface the Hessian was (or would have been)
        built on, the method the caller requested, and whether the two
        agree. run_job resolves a correlated request down to its
        mean-field reference before the Hessian is built, so without
        this section a consumer reading frequencies out of a manifest
        cannot tell an MP2 job's RHF frequencies from an RHF job's.
        """
        with self._lock:
            self._hessian = dict(fields)
            self._write_unlocked()

    def update_wall_seconds(self, wall_seconds: float) -> None:
        """Stamp the running wall-time figure into the ``[run]``
        section. Called periodically by the writer / on completion."""
        with self._lock:
            self._wall_seconds = float(wall_seconds)
            self._write_unlocked()

    def update_progress(self, **fields: object) -> None:
        """Stamp live calculation progress into the ``[progress]``
        section.

        Typical call after each SCF iteration::

            manifest.update_progress(
                phase="scf", iteration=6, energy_eh=-73.496,
                gradient_norm=3.3e-08, diis_subspace=6,
            )

        Fields are written verbatim into the ``[progress]`` TOML
        section; unknown keys are ignored by consumers per the
        fixed-shape rule.  Passing no fields clears the section.
        """
        with self._lock:
            if fields:
                self._progress = dict(fields)
            else:
                self._progress = None
            self._write_unlocked()

    def update_run_fields(self, fields: dict[str, Any]) -> None:
        """Merge scalar reproducibility fields into ``[run]``.

        Periodic routing resolves some values only after the calculation has
        selected an executable backend. The manifest still starts before
        compute; this mutator fills those late-bound values without replacing
        the coordinator or rebuilding its output outcomes.
        """
        with self._lock:
            self._extra_run_fields.update(fields)
            self._write_unlocked()

    def finish(self, *, wall_seconds: float | None = None) -> None:
        """Finalize only when every ``always=True`` plan row was written.

        A missing guaranteed artefact is first recorded truthfully in the
        manifest, whose status becomes ``"crashed"``, then raises
        :class:`IncompleteOutputError`. Conditional rows may remain pending.
        """
        incomplete_error: IncompleteOutputError | None = None
        with self._lock:
            self._finished_at_iso = _now_iso()
            if wall_seconds is not None:
                self._wall_seconds = float(wall_seconds)
            self._mark_manifest_written_unlocked()
            incomplete = tuple(
                planned
                for planned in self._plan.guaranteed_files()
                if not self._outcome_for_path_unlocked(planned.path).written
                and planned.path not in self._downgraded_paths
            )
            if incomplete:
                self._status = "crashed"
                for planned in incomplete:
                    outcome = self._outcome_for_path_unlocked(planned.path)
                    outcome.bytes = None
                    outcome.sha256 = None
                    outcome.checksum_status = "failed"
                    if outcome.error is None:
                        outcome.error = (
                            "IncompleteOutputError: guaranteed artefact was "
                            "not written before finalization"
                        )
                incomplete_error = IncompleteOutputError(incomplete)
            else:
                self._status = "complete"
            self._write_unlocked()
        if incomplete_error is not None:
            raise incomplete_error

    def crash(self, *, wall_seconds: float | None = None) -> None:
        """Flip status to ``"crashed"`` (SCF raised / job aborted)."""
        with self._lock:
            self._status = "crashed"
            self._finished_at_iso = _now_iso()
            if wall_seconds is not None:
                self._wall_seconds = float(wall_seconds)
            self._mark_manifest_written_unlocked()
            self._write_unlocked()

    def mark_dry_run(self) -> None:
        """Flip status to ``"dry_run"`` and stamp the finish time --
        the manifest was produced by a pre-flight pass that won't run
        any SCF. Used by :func:`vibeqc.output.dry_run_manifest` and by
        the ``VIBEQC_DRY_RUN=1`` short-circuit in
        :func:`vibeqc.runner.run_job`."""
        with self._lock:
            self._status = "dry_run"
            self._finished_at_iso = _now_iso()
            self._mark_manifest_written_unlocked()
            self._write_unlocked()

    # -- public read accessors ----------------------------------------- #

    @property
    def path(self) -> Path:
        """The on-disk manifest path.

        Absolute for a relative stem: anchored to the cwd at job start so
        the location is fixed for the lifetime of the updater, however the
        process moves afterwards. The manifest *text* keeps the declared
        (relative) spelling.
        """
        return self._path

    @property
    def status(self) -> ManifestStatus:
        return self._status

    def outcomes(self) -> tuple[FileOutcome, ...]:
        """Snapshot copy of the current per-file outcome records."""
        with self._lock:
            return tuple(
                FileOutcome(
                    path=o.path,
                    written=o.written,
                    bytes=o.bytes,
                    sha256=o.sha256,
                    checksum_status=o.checksum_status,
                    wall_time_s=o.wall_time_s,
                    error=o.error,
                )
                for o in self._outcomes.values()
            )

    # -- internals ----------------------------------------------------- #

    def _on_disk(self, path: Path) -> Path:
        """Where ``path`` lives on disk: a relative artefact path is
        anchored to the job-start cwd, an absolute one is returned as is.

        Every stat, hash and write in this class goes through here so a
        cwd change after job start cannot redirect them (vibe-qc#127).
        """
        if path.is_absolute() or self._job_cwd is None:
            return path
        return self._job_cwd / path

    def _write(self) -> None:
        """Public-friendly first write; acquires the lock."""
        with self._lock:
            self._write_unlocked()

    def _outcome_for_path_unlocked(self, path: Path) -> FileOutcome:
        """Return the matching row, accepting equivalent path spellings."""
        outcome = self._outcomes.get(path)
        if outcome is not None:
            return outcome
        for existing_path, existing in self._outcomes.items():
            if _same_path(self._on_disk(existing_path), self._on_disk(path)):
                return existing
        outcome = FileOutcome(path=path)
        self._outcomes[path] = outcome
        return outcome

    def _mark_manifest_written_unlocked(self) -> None:
        """Record the truthful self-manifest checksum exclusion contract."""
        outcome = self._outcome_for_path_unlocked(self._declared_path)
        outcome.written = True
        outcome.bytes = None
        outcome.sha256 = None
        outcome.checksum_status = "self-excluded"
        outcome.wall_time_s = float(self._wall_seconds)
        outcome.error = None

    def _write_unlocked(self) -> None:
        body = _render_manifest(
            plan=self._plan,
            outcomes=tuple(self._outcomes.values()),
            status=self._status,
            declared_at_iso=self._declared_at_iso,
            finished_at_iso=self._finished_at_iso,
            wall_seconds=self._wall_seconds,
            system_info_snapshot=self._system_info,
            citations=self._citations,
            basis_library=self._basis_library,
            estimate_bytes=self._estimate_bytes,
            extra_run_fields=self._extra_run_fields,
            progress=self._progress,
            hessian=self._hessian,
        )
        _atomic_write(self._path, body)


# ---------------------------------------------------------------------- #
# Convenience entry point                                                #
# ---------------------------------------------------------------------- #


def write_initial_manifest(
    plan: OutputPlan,
    *,
    record_hostname: bool = True,
    wall_seconds: float = 0.0,
) -> ManifestUpdater:
    """One-shot helper -- construct + return a :class:`ManifestUpdater`.

    The constructor already writes the initial manifest, so the caller
    can immediately start calling ``mark_written`` / ``finish`` on the
    returned handle.
    """
    return ManifestUpdater(
        plan,
        record_hostname=record_hostname,
        wall_seconds=wall_seconds,
    )


# ---------------------------------------------------------------------- #
# TOML emitter -- extends what system_info.py knows                       #
# ---------------------------------------------------------------------- #

# Section ordering for the emitted file -- fixed so diffs are stable
# across runs. ``[plan]`` and ``[outputs]`` land after the runtime-env
# block so the v0.5.1 manifest shape is a strict prefix of the v0.8.x
# shape.
_SECTION_ORDER = (
    "vibeqc",
    "host",
    "cpu",
    "memory",
    "python",
    "libraries",
    "validation",
    "run",
    "progress",
    "hessian",
    "plan",
    "outputs",
    "basis_library",
    "citations",
)


def _render_manifest(
    *,
    plan: OutputPlan,
    outcomes: tuple[FileOutcome, ...],
    status: ManifestStatus,
    declared_at_iso: str,
    finished_at_iso: str,
    wall_seconds: float,
    system_info_snapshot: dict[str, Any],
    citations: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    basis_library: dict[str, Any] | None = None,
    estimate_bytes: int | None = None,
    extra_run_fields: dict[str, Any] | None = None,
    progress: dict[str, object] | None = None,
    hessian: dict[str, object] | None = None,
) -> str:
    """Produce the full manifest TOML body.

    ``system_info_snapshot`` is the runtime-env probe taken once at
    :class:`ManifestUpdater` construction (see
    :func:`vibeqc.system_info.system_info`) -- it is invariant across
    a run, so the coordinator passes the cached dict here rather than
    re-probing on every rewrite. The runtime-env sections are
    rendered with the same hand-rolled emitter used by
    :func:`vibeqc.system_info.write_system_manifest`; the new
    ``[plan]`` + ``[outputs]`` sections support arrays of inline tables
    which the original emitter did not need.

    ``estimate_bytes`` (when not ``None``) is appended to the
    ``[memory]`` section as ``estimate_bytes``: vibe-qc's peak-memory
    estimate (``estimate_memory(...).total_bytes``, configured headroom
    already applied) for memory-aware ``vq submit auto`` placement. It
    is set only on the ``VIBEQC_DRY_RUN_ESTIMATE`` dry-run path; ``None``
    leaves ``[memory]`` exactly as the runtime probe produced it.
    """
    # Shallow-copy so the per-rewrite run / plan / outputs keys don't
    # accumulate on the cached snapshot (they're overwritten each
    # call, but a fresh top-level dict keeps the snapshot pristine).
    info: dict[str, Any] = dict(system_info_snapshot)
    # Append the opt-in peak-memory estimate to [memory] when computed.
    # Copy the sub-dict (a shallow top-level copy still shares it) so the
    # cached snapshot held by the updater stays pristine across rewrites.
    if estimate_bytes is not None:
        mem = dict(info.get("memory") or {})
        mem["estimate_bytes"] = int(estimate_bytes)
        info["memory"] = mem
    info["run"] = {
        "timestamp_iso": declared_at_iso,
        "wall_seconds": float(wall_seconds),
        "basename": str(plan.stem.name),
        "pid": int(os.getpid()),
    }
    # Optional per-run scalar provenance (e.g. the periodic exchange-q=0
    # convention). Merged after the fixed keys so the v0.5.1 prefix shape holds.
    for _k, _v in (extra_run_fields or {}).items():
        info["run"][str(_k)] = _v
    if progress is not None:
        info["progress"] = dict(progress)
    # [hessian] -- the potential-energy surface a requested Hessian was
    # built on. Conditional like [progress]: present only when the job
    # asked for a Hessian, absent otherwise.
    if hessian is not None:
        info["hessian"] = dict(hessian)
    info["plan"] = plan.to_toml_section()
    info["outputs"] = {
        "status": status,
        "finished_at_iso": finished_at_iso,
        "files": [o.to_toml_table() for o in outcomes],
    }
    # [citations] -- full assembled provenance for this job (CLAUDE.md
    # Sec.8.3/Sec.8.5). Always emitted (count = 0 with no entries before the
    # end-of-run assemble fires, or for citation-less paths) so the
    # fixed-shape rule holds: the section never disappears. Distinct from
    # [libraries], which lists what is *linked* into the binary; this
    # lists what this job actually *cites*.
    info["citations"] = {
        "count": len(citations),
        "entries": list(citations),
    }
    # [basis_library] -- where this job's basis data came from. Always
    # emitted, like [citations] above, so the fixed-shape rule holds and a
    # consumer never has to branch on the section's absence: an ordinary run
    # shows fetched_count = 0 and names the resolved root.
    info["basis_library"] = dict(basis_library or {}) or {
        "resolved_root": "",
        "fetched_count": 0,
        "fetched": [],
    }

    lines: list[str] = [
        "# vibe-qc system manifest -- written alongside output-<job>.out by run_job(...).",
        "# Captures the runtime environment so generated calculation outputs are",
        "# reproducible and wall-time numbers are interpretable. The [plan]",
        "# section is the declared output contract (written once, at job start);",
        "# the [outputs] section is the running status (rewritten as each file",
        "# lands). External QC programs are validation references only: run",
        "# them out-of-process and parse their outputs.",
        "",
    ]
    for section in _SECTION_ORDER:
        if section not in info:
            continue
        if section in ("plan", "outputs", "citations", "basis_library"):
            _emit_complex_section(lines, section, info[section])
        else:
            _emit_flat_section(lines, section, info[section])
    return "\n".join(lines)


def _emit_flat_section(lines: list[str], name: str, body: dict[str, Any]) -> None:
    """Emit a section whose body is a flat key/value mapping (no
    sub-tables, no arrays-of-tables) -- i.e. the original
    ``system_info``-style sections."""
    lines.append(f"[{name}]")
    for key, value in body.items():
        lines.append(f"{key:<14s} = {_toml_value(value)}")
    lines.append("")


def _emit_complex_section(lines: list[str], name: str, body: dict[str, Any]) -> None:
    """Emit a section that may contain a ``files`` array-of-tables
    (the ``[plan]`` and ``[outputs]`` sections).

    Layout:

    .. code-block:: text

        [name]
        scalar_key1 = ...
        scalar_key2 = ...

        [[name.files]]
        path = ...
        ...

        [[name.files]]
        ...
    """
    lines.append(f"[{name}]")
    array_key: str | None = None
    array_items: list[dict[str, Any]] = []
    for key, value in body.items():
        if isinstance(value, list):
            # We only support one array-of-tables per section; the
            # caller controls which.
            array_key = key
            array_items = value
            continue
        lines.append(f"{key:<16s} = {_toml_value(value)}")
    lines.append("")

    if array_key is not None:
        for item in array_items:
            lines.append(f"[[{name}.{array_key}]]")
            for k, v in item.items():
                lines.append(f"{k:<13s} = {_toml_value(v)}")
            lines.append("")


# ---------------------------------------------------------------------- #
# Helpers -- atomic write, sha256, ISO timestamps                         #
# ---------------------------------------------------------------------- #

# ---------------------------------------------------------------------- #
# Tiny TOML value emitter -- kept local so this module has no coupling    #
# to private helpers in vibeqc.system_info. The pre-v1.0 rewrite will    #
# consolidate the two emitters under vibeqc.output.                      #
# ---------------------------------------------------------------------- #


def _toml_str(s: str) -> str:
    """Quote a string as a TOML basic string. Escapes ``\\``, ``"``, the
    whitespace + C0 controls, and the bidi / zero-width / BOM format
    controls (U+202E etc.) per the TOML 1.0 spec. Delegates to the shared
    output text-safety helper so the dangerous-character set is defined in
    exactly one place -- see :mod:`vibeqc.output._text_safety`."""
    return toml_escape_str(s)


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = repr(v)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s
    return _toml_str(str(v))


def _atomic_write(path: Path, body: str) -> None:
    """Write ``body`` to ``path`` atomically (tmp + rename).

    Crashes mid-write leave the previous valid manifest in place rather
    than a partial file -- important because ``vq`` polls this file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


def _stat_and_hash(path: Path) -> tuple[int | None, str | None]:
    """Best-effort ``(bytes, sha256)`` of an on-disk file. Returns
    ``(None, None)`` when the file is missing or unreadable -- never
    raises (a stat probe shouldn't crash a job at completion)."""
    try:
        size = path.stat().st_size
    except OSError:
        return (None, None)
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return (size, h.hexdigest())
    except OSError:
        return (size, None)


def _same_path(left: Path, right: Path) -> bool:
    """Compare paths without requiring either target to already exist."""
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return os.path.abspath(os.fspath(left)) == os.path.abspath(os.fspath(right))


def _now_iso() -> str:
    """ISO-8601 timestamp with timezone, second precision. Matches the
    convention used by :func:`vibeqc.system_info.write_system_manifest`
    and the structured-log emitter."""
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")
