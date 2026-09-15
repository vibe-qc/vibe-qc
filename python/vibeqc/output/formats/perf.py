"""Post-mortem performance / debug log for vibe-qc.

The v0.5.1 :mod:`vibeqc.progress` module shows progress *during* a
run; this module shows where the time went *afterwards*. The two
pair: live for "is the SCF stuck?", perf for "why is my LiH /
pob-TZVP run taking 20 minutes -- is it J, K, XC quadrature, or
Bloch sums?"

Three ways to enable, all of which feed the same accumulator under
the hood:

* **Environment variable** -- common case for one-off jobs::

      VIBEQC_PERFLOG=output.perf python my-calc.py

  When set, every ``run_job`` call (and every entry point that
  threads ``perf_log=``) routes through the tracker.

* **Programmatic context manager** -- wrap a region::

      import vibeqc as vq
      with vq.perf_log("output.perf"):
          result = vq.run_rhf(mol, basis, opts)
          hess = vq.compute_hessian_rhf_analytic(...)

  All work inside the block accumulates into the same tracker;
  the report is written when the block exits.

* **``run_job`` kwarg** -- one-shot::

      vq.run_job(mol, basis="cc-pVDZ", method="rks",
                 functional="pbe", perf_log="output.perf")

The output file is plain text with section headers, sortable by
wall-time -- **unless the target is named ``*.json``**, in which case
the same tracker is written as a JSON object (schema
``vibeqc.perf/1``) for automated performance harvests::

      vq.run_job(..., perf_log="output.perf")       # text report
      vq.run_job(..., perf_log="output.perf.json")  # JSON object

v0.5.2 instruments the Python-driven periodic SCF
pipeline (the EWALD_3D paths) and the high-level :func:`run_job`
phases. C++ kernel-level scopes (``compute_eri``, ``build_coulomb``,
``xc_eval``) are gated behind a compile-time ``#ifdef
VIBEQC_PERFLOG`` and arrive in a v0.5.2.x patch -- release builds pay
zero runtime cost when the feature is off.

Public surface
--------------

.. autoclass:: PerfTracker
.. autoclass:: PerfScope
.. autofunction:: perf_log
.. autofunction:: format_perf_report
.. autofunction:: format_perf_json
.. autofunction:: perf_artefact_format
.. autofunction:: active_tracker
"""

from __future__ import annotations

import contextvars
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union

from .._errors import OutputFailureKind, warn_output_failure
from ..document import (
    FormatPolicy,
    HeaderlessBlock,
    HeaderlessColumn,
    Quantity,
    active_policy,
)


__all__ = [
    "PERF_JSON_SCHEMA",
    "PerfScope",
    "PerfTracker",
    "active_tracker",
    "format_perf_json",
    "format_perf_report",
    "perf_artefact_format",
    "perf_log",
]


# Schema tag stamped into every JSON perf artefact. Bump the integer
# when a field is removed or its meaning changes; adding a field is
# backwards-compatible and does not bump it.
PERF_JSON_SCHEMA = "vibeqc.perf/1"


_DURATION_UNIT_NAMES = {
    "s": "seconds",
    "ms": "milliseconds",
    "min": "minutes",
}


def _format_timing_summary(
    rows: Sequence[tuple[str, float, Optional[str]]],
    *,
    label_width: int,
    body_indent: int = 0,
    policy: Optional[FormatPolicy] = None,
) -> str:
    """Render the runners' compact wall-clock summary.

    ``annotation`` values such as ``"(12 iters)"`` remain ragged metadata:
    they do not extend the content-sized rule. The active duration unit drives
    both the numeric conversion and the title, so a millisecond policy cannot
    leave a stale ``seconds`` label behind.
    """
    render_policy = active_policy() if policy is None else policy
    unit = render_policy.unit_of("duration")
    unit_name = _DURATION_UNIT_NAMES.get(unit, unit)
    block = HeaderlessBlock(
        f"Timings (wall clock, {unit_name})",
        [
            HeaderlessColumn("<", min_width=label_width),
            HeaderlessColumn(">", min_width=12),
        ],
        body_indent=body_indent,
        gutter=1,
        annotation_gutter=2,
    )
    for label, duration_s, annotation in rows:
        block.add_row(
            label,
            Quantity(float(duration_s), "duration"),
            annotation=annotation,
        )
    return block.render(render_policy)


# ---------------------------------------------------------------------------
# Memory probe -- best-effort, never raises.
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Current resident set size in MiB. ``0.0`` when unavailable."""
    try:
        import resource  # POSIX-only
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # Linux reports ru_maxrss in KiB, macOS in bytes. Detect by magnitude.
        if sys.platform == "darwin":
            return ru.ru_maxrss / (1024.0 * 1024.0)
        return ru.ru_maxrss / 1024.0
    except Exception:
        try:
            import psutil  # type: ignore[import-not-found]
            return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# PerfTracker -- accumulator for the duration of a perf_log() block.
# ---------------------------------------------------------------------------

@dataclass
class _PhaseStat:
    """Cumulative timing record for a single named scope."""
    name: str
    n_calls: int = 0
    wall_s: float = 0.0
    cpu_s: float = 0.0

    def __iadd__(self, other: tuple[float, float]) -> "_PhaseStat":
        self.n_calls += 1
        self.wall_s += other[0]
        self.cpu_s += other[1]
        return self


@dataclass
class PerfTracker:
    """Accumulator for per-phase wall/CPU timings, per-iteration SCF
    rows, and memory snapshots over the course of a calculation.

    Constructed inside :func:`perf_log` and passed implicitly to every
    :class:`PerfScope` opened in the same async-context (uses
    :class:`contextvars.ContextVar`, so perf trackers do NOT leak
    across threads or asyncio tasks). Call sites that want to read
    the live tracker -- to add a custom snapshot, for instance --
    use :func:`active_tracker`.

    Public attributes:

    * ``phases`` -- map ``phase_name -> _PhaseStat``; ``wall_s`` on
      each entry is the cumulative wall time across all calls.
    * ``scf_iters`` -- list of per-iteration dicts (energy / dE /
      grad / DIIS / wall_s); populated by SCF entry points wired
      into the progress logger.
    * ``memory_snapshots`` -- list of ``{label, rss_mb, t_s}`` taken
      at SCF transitions.
    * ``threads`` -- snapshot of the OpenMP thread count at tracker
      construction (used to compute parallelism = cpu_s /
      (wall_s x threads) per phase).
    * ``t_start`` -- perf_counter timestamp; phase rows report
      wall-time deltas relative to this anchor.
    """

    phases: dict[str, _PhaseStat] = field(default_factory=dict)
    scf_iters: list[dict[str, Any]] = field(default_factory=list)
    memory_snapshots: list[dict[str, Any]] = field(default_factory=list)
    threads: int = 1
    t_start: float = field(default_factory=time.perf_counter)

    def add_scope(self, name: str, wall_s: float, cpu_s: float) -> None:
        stat = self.phases.setdefault(name, _PhaseStat(name=name))
        stat += (wall_s, cpu_s)

    def add_scf_iter(self, **fields: Any) -> None:
        """Record one SCF iteration. Recognized keys: ``iter``,
        ``energy``, ``dE``, ``grad``, ``diis``. Extra keys are kept
        verbatim in the row."""
        self.scf_iters.append(fields)

    def snapshot_memory(self, label: str) -> None:
        """Take a labeled RSS snapshot. ``label`` is what shows up
        in the report (typically ``"start_of_scf"``,
        ``"after_iter_5"``, ``"end_of_scf"``)."""
        self.memory_snapshots.append({
            "label":  label,
            "rss_mb": _rss_mb(),
            "t_s":    time.perf_counter() - self.t_start,
        })

    @property
    def total_wall_s(self) -> float:
        return time.perf_counter() - self.t_start


# ---------------------------------------------------------------------------
# Active-tracker plumbing via ContextVar (thread / task safe).
# ---------------------------------------------------------------------------

_active: contextvars.ContextVar[Optional[PerfTracker]] = contextvars.ContextVar(
    "vibeqc_perf_tracker", default=None,
)


def active_tracker() -> Optional[PerfTracker]:
    """The currently active :class:`PerfTracker`, or ``None`` if no
    :func:`perf_log` block is open. Used by lower-level instrumented
    code paths (e.g. periodic SCF iteration loops) to report
    per-iter rows without taking an explicit tracker argument."""
    return _active.get()


def _native_thread_count() -> int | None:
    """Return the native OpenMP runtime's current maximum, if available."""
    try:
        from ..._vibeqc_core import get_num_threads

        value = int(get_num_threads())
    except Exception:
        return None
    return value if value > 0 else None


def _affinity_thread_count() -> int | None:
    """Return the process CPU-affinity allocation when the OS exposes it."""
    try:
        affinity = os.sched_getaffinity(0)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None
    count = len(affinity)
    return count if count > 0 else None


def _positive_env_int(name: str) -> int | None:
    """Parse one positive scheduler/OpenMP thread-count environment value."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    # OMP_NUM_THREADS may be a nested-region list (for example ``8,2``).
    token = raw.split(",", 1)[0].strip()
    try:
        value = int(token)
    except ValueError:
        return None
    return value if value > 0 else None


_ALLOCATION_THREAD_ENVS = (
    "OMP_NUM_THREADS",
    "VQ_CPUS",
    "SLURM_CPUS_PER_TASK",
    "PBS_NP",
    "NSLOTS",
    "LSB_DJOB_NUMPROC",
)


def _detect_threads() -> int:
    """Snapshot the OpenMP threads available to this calculation.

    Runtime and scheduler allocation signals cap one another. Host-wide core
    counts are deliberately excluded: a one-CPU batch allocation on a
    96-core node is still a one-thread job, and using ``os.cpu_count()`` here
    would fabricate severe under-parallelisation in the post-mortem report.
    When no allocation/runtime signal is available, report the conservative
    single-thread value rather than guessing from host capacity.
    """
    runtime = _native_thread_count()
    allocated = [
        value
        for name in _ALLOCATION_THREAD_ENVS
        if (value := _positive_env_int(name)) is not None
    ]
    # Affinity is useful as a cap once another signal establishes that the
    # calculation has a multi-thread runtime/allocation.  It is not sufficient
    # by itself: an unrestricted process commonly sees every CPU on the host.
    established = ([runtime] if runtime is not None else []) + allocated
    if not established:
        return 1
    affinity = _affinity_thread_count()
    if affinity is not None:
        established.append(affinity)
    return min(int(value) for value in established)


# ---------------------------------------------------------------------------
# PerfScope -- RAII timer.
# ---------------------------------------------------------------------------

class PerfScope:
    """Context manager that times its body and pushes the wall + CPU
    delta into the currently active :class:`PerfTracker`. No-op when
    no tracker is active, so call sites can sprinkle ``with
    PerfScope("phase_name"): ...`` unconditionally -- the cost when
    perf logging is off is one ``contextvars.ContextVar.get()`` and
    one ``time.perf_counter()`` call.

    Re-entrant: nesting two scopes with the same name accumulates
    correctly (each scope contributes its own wall/CPU).
    """

    __slots__ = ("name", "_t_wall", "_t_cpu")

    def __init__(self, name: str) -> None:
        self.name = name
        self._t_wall = 0.0
        self._t_cpu = 0.0

    def __enter__(self) -> "PerfScope":
        self._t_wall = time.perf_counter()
        self._t_cpu = time.process_time()
        return self

    def __exit__(self, *exc: Any) -> None:
        wall = time.perf_counter() - self._t_wall
        cpu = time.process_time() - self._t_cpu
        tracker = _active.get()
        if tracker is not None:
            tracker.add_scope(self.name, wall, cpu)


# ---------------------------------------------------------------------------
# perf_log() context manager -- install/uninstall a tracker.
# ---------------------------------------------------------------------------

@contextmanager
def perf_log(
    path: Union[str, os.PathLike, None] = None,
) -> Iterator[PerfTracker]:
    """Activate a fresh :class:`PerfTracker` for the duration of the
    block; write the formatted report to ``path`` on exit.

    Resolution order for ``path``:

      1. Explicit argument when not ``None``.
      2. ``$VIBEQC_PERFLOG`` env var.
      3. No file written (the tracker still accumulates and the user
         can read it from the yielded value).

    Nesting two ``perf_log()`` blocks is allowed but discouraged --
    the inner block's tracker is independent and only inner-block
    work routes there. Most callers want a single top-level block.

    Example::

        import vibeqc as vq
        with vq.perf_log("h2o.perf") as tracker:
            vq.run_rhf(mol, basis, opts)
        # h2o.perf now exists; tracker.phases is also accessible
        # post-block for programmatic inspection.
    """
    target: Optional[Path] = None
    if path is not None:
        target = Path(os.fspath(path))
    else:
        env = os.environ.get("VIBEQC_PERFLOG", "").strip()
        if env:
            target = Path(env)

    tracker = PerfTracker(threads=_detect_threads())
    token = _active.set(tracker)
    try:
        yield tracker
    finally:
        _active.reset(token)
        if target is not None:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Honour the caller's declared extension. A path named
                # ``.json`` gets JSON; everything else gets the text
                # report. See perf_artefact_format().
                render = (
                    format_perf_json
                    if perf_artefact_format(target) == "json"
                    else format_perf_report
                )
                target.write_text(render(tracker), encoding="utf-8")
            except OSError as exc:
                # A perf-log write failure must never propagate up
                # and crash a finished calculation. Surface the shared
                # structured warning so the user notices, but keep going.
                warn_output_failure(
                    exc,
                    target,
                    "performance_log",
                    category=OutputFailureKind.optional_artifact,
                )


# ---------------------------------------------------------------------------
# Artefact-format selection.
# ---------------------------------------------------------------------------

def perf_artefact_format(path: Union[str, os.PathLike]) -> str:
    """``"json"`` when ``path`` is named ``*.json``, else ``"text"``.

    The perf artefact has two renderings of the same tracker: the
    human-readable report (:func:`format_perf_report`, the default
    ``.perf`` sibling) and a machine-readable object
    (:func:`format_perf_json`). The caller selects between them by
    naming the target -- ``perf_log="run.perf"`` gets prose,
    ``perf_log="run.perf.json"`` gets JSON.

    Before this dispatch existed the writer emitted prose regardless of
    the name, so campaign payloads passing ``perf_log="<stem>.perf.json"``
    produced 545 artefacts that ``json.loads()`` refused. The suffix is
    the only signal the caller gives, so it is the one the writer obeys;
    :mod:`vibeqc.output.plan` calls this same helper so the ``.system``
    manifest declares the format that is actually on disk.
    """
    return "json" if Path(os.fspath(path)).suffix.lower() == ".json" else "text"


# ---------------------------------------------------------------------------
# Report formatter.
# ---------------------------------------------------------------------------

_BAR = "=" * 72


def _fmt_seconds(s: float) -> str:
    if s < 1.0:
        return f"{s * 1e3:8.2f} ms"
    if s < 60.0:
        return f"{s:8.3f} s "
    m, sec = divmod(s, 60.0)
    return f"{int(m):>4d}m{sec:05.2f}s"


def format_perf_report(tracker: PerfTracker) -> str:
    """Render ``tracker`` as the on-disk perf report text.

    Sections, in order:

      1. **Header** -- total wall-time, thread count, timestamp.
      2. **Phase summary** -- one row per phase, sorted by total wall
         time descending. Columns: phase name, n calls, total wall,
         total CPU, % of total wall, parallelism (cpu / (wall x n
         threads)).
      3. **Top-3 hot phases + parallelism flags** -- phases consuming
         > 5% of wall AND with parallelism < 0.7x the available
         thread count are flagged (under-parallelised hot paths).
      4. **Memory snapshots** -- labeled RSS samples taken at SCF
         transitions.
      5. **Per-iteration SCF table** -- one row per SCF iteration,
         when populated.

    The output is plain ASCII, no color codes -- matches the ``.out``
    file convention.
    """
    total_wall = tracker.total_wall_s
    threads = max(1, tracker.threads)

    lines: list[str] = []
    lines.append(_BAR)
    lines.append("vibe-qc performance / debug log")
    lines.append(_BAR)
    lines.append(f"  PID:             {os.getpid()}")
    lines.append(f"  Total wall time: {_fmt_seconds(total_wall)}")
    lines.append(f"  OMP threads:     {threads}")
    lines.append(f"  Phases tracked:  {len(tracker.phases)}")
    lines.append("")

    # -- Phase summary --
    lines.append("Phase summary  (sorted by total wall time, descending)")
    lines.append("-" * 72)
    header = (
        f"  {'phase':<32s}  {'n':>5s}  "
        f"{'wall':>11s}  {'cpu':>11s}  {'% wall':>6s}  {'par':>5s}"
    )
    lines.append(header)
    lines.append("  " + "-" * 70)
    sorted_phases = sorted(
        tracker.phases.values(),
        key=lambda p: p.wall_s,
        reverse=True,
    )
    flags: list[str] = []
    for stat in sorted_phases:
        pct = (100.0 * stat.wall_s / total_wall) if total_wall > 0 else 0.0
        # parallelism: cpu time / (wall x threads). 1.0 = perfect
        # OMP scaling, < 0.7 = under-parallelised hot path.
        par = (stat.cpu_s / (stat.wall_s * threads)) if stat.wall_s > 0 else 0.0
        lines.append(
            f"  {stat.name:<32s}  {stat.n_calls:>5d}  "
            f"{_fmt_seconds(stat.wall_s)}  {_fmt_seconds(stat.cpu_s)}  "
            f"{pct:5.1f}%  {par:5.2f}"
        )
        if pct > 5.0 and par < 0.7 and threads > 1:
            flags.append(
                f"  - {stat.name} consumed {pct:.1f}% of wall but ran at "
                f"parallelism {par:.2f}x ({threads} threads available)"
            )
    lines.append("")

    # -- Hot-path flags --
    if flags:
        lines.append("Under-parallelised hot paths  (>5% wall, par < 0.7x)")
        lines.append("-" * 72)
        lines.extend(flags)
        lines.append("")

    # -- Memory snapshots --
    if tracker.memory_snapshots:
        lines.append("Memory snapshots  (RSS in MiB)")
        lines.append("-" * 72)
        lines.append(f"  {'label':<32s}  {'t (s)':>10s}  {'RSS (MiB)':>12s}")
        lines.append("  " + "-" * 60)
        for snap in tracker.memory_snapshots:
            lines.append(
                f"  {snap['label']:<32s}  {snap['t_s']:10.3f}  "
                f"{snap['rss_mb']:12.1f}"
            )
        lines.append("")

    # -- Per-iteration SCF table --
    if tracker.scf_iters:
        lines.append("SCF iterations")
        lines.append("-" * 72)
        lines.append(
            f"  {'iter':>4s}  {'E (Ha)':>18s}  {'dE':>11s}  "
            f"{'||[F,DS]||':>11s}  {'DIIS':>4s}  {'wall (s)':>9s}"
        )
        lines.append("  " + "-" * 70)
        for row in tracker.scf_iters:
            n = int(row.get("iter", 0))
            e = float(row.get("energy", float("nan")))
            de = row.get("dE")
            de_str = f"{float(de):+.3e}" if de is not None and n > 1 else "      --   "
            grad = row.get("grad")
            grad_str = f"{float(grad):.3e}" if grad is not None else "       --"
            diis = int(row.get("diis", 0))
            diis_str = f"{diis:4d}" if diis > 0 else "   -"
            # An absent/None wall is UNMEASURED: render "--", never a
            # fabricated "0.000" (BUG87-A: the shipped .perf printed
            # 0.000 on every iteration of a 2258 s SCF because the
            # molecular C++ trace carried no walls and this default
            # invented one).
            wall = row.get("wall_s")
            wall_str = (
                f"{float(wall):9.3f}" if wall is not None else f"{'--':>9s}"
            )
            lines.append(
                f"  {n:>4d}  {e:>18.10f}  {de_str}  {grad_str}  "
                f"{diis_str}  {wall_str}"
            )
        lines.append("")

    lines.append(_BAR)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# JSON formatter.
# ---------------------------------------------------------------------------

def format_perf_json(tracker: PerfTracker) -> str:
    """Render ``tracker`` as the on-disk perf artefact in JSON.

    Machine-readable sibling of :func:`format_perf_report`: same
    tracker, same quantities, no formatting decisions baked in. Every
    duration is a raw float in **seconds** and every memory figure a
    raw float in **MiB** -- a harvest reading this file does its own
    rounding rather than re-parsing ``"  12.345 ms"``.

    Top-level object, schema ``vibeqc.perf/1``::

        {
          "schema":        "vibeqc.perf/1",
          "pid":           12345,
          "threads":       8,             # OMP threads at tracker start
          "total_wall_s":  61.15,
          "n_phases":      6,
          "phases": [                     # sorted by wall_s descending
            {"name": "scf.rhf", "n_calls": 1, "wall_s": 41.2,
             "cpu_s": 302.4, "pct_wall": 67.4, "parallelism": 0.92}
          ],
          "memory_snapshots": [
            {"label": "end_of_scf", "t_s": 41.9, "rss_mb": 1830.5}
          ],
          "scf_iters": [                  # rows verbatim from the tracker
            {"iter": 1, "energy": -76.02, "dE": null, "grad": 1.2e-3,
             "diis": 0, "wall_s": 1.8}
          ]
        }

    ``parallelism`` is ``cpu_s / (wall_s * threads)`` -- 1.0 is perfect
    OpenMP scaling -- and is ``null`` when ``wall_s`` is zero rather
    than a fabricated 0.0. ``scf_iters`` rows are passed through as the
    tracker recorded them, so an absent ``wall_s`` stays absent instead
    of becoming a fake ``0.000`` (BUG87-A).
    """
    total_wall = tracker.total_wall_s
    threads = max(1, tracker.threads)

    phases: list[dict[str, Any]] = []
    for stat in sorted(
        tracker.phases.values(), key=lambda p: p.wall_s, reverse=True
    ):
        phases.append({
            "name": stat.name,
            "n_calls": stat.n_calls,
            "wall_s": stat.wall_s,
            "cpu_s": stat.cpu_s,
            "pct_wall": (
                100.0 * stat.wall_s / total_wall if total_wall > 0 else 0.0
            ),
            "parallelism": (
                stat.cpu_s / (stat.wall_s * threads)
                if stat.wall_s > 0
                else None
            ),
        })

    payload = {
        "schema": PERF_JSON_SCHEMA,
        "pid": os.getpid(),
        "threads": threads,
        "total_wall_s": total_wall,
        "n_phases": len(tracker.phases),
        "phases": phases,
        "memory_snapshots": [
            {
                "label": snap.get("label"),
                "t_s": snap.get("t_s"),
                "rss_mb": snap.get("rss_mb"),
            }
            for snap in tracker.memory_snapshots
        ],
        # Verbatim rows: the SCF drivers own these keys, and a harvest
        # that wants a column the text table never had should not have
        # to wait for this writer to learn about it.
        "scf_iters": [dict(row) for row in tracker.scf_iters],
    }
    # default=str keeps a stray non-serialisable value in an scf_iters
    # row from turning a finished calculation's perf artefact into a
    # write failure.
    return json.dumps(payload, indent=2, default=str, sort_keys=False) + "\n"
