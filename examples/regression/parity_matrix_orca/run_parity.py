"""On-demand driver for the ORCA axis of the HF/DFT parity matrix.

Per cell: run ORCA on compute-host-d (or reuse a cached result), decompose
vibe-qc locally, compare each energy piece, emit a markdown report.

ORCA round-trips are slow and serialised (the vq daemon is in a
``--max-jobs 1`` phase), so historical ORCA fixtures are kept as
committed JSON under ``cache/`` — keyed on the cell id. Normal reports,
vq workspaces, fetched outputs, and regenerated cache copies go under
``~/vibeqc-runs/<run-id>/`` (or ``--output-root`` / ``VIBEQC_RUNS_DIR``).
Use ``--update-fixture-cache`` only when intentionally refreshing the
committed fixtures.

Usage::

    # compare every cell that already has a cached ORCA result
    python -m examples.regression.parity_matrix_orca.run_parity

    # run ONE cell end-to-end on compute-host-d first (the handover's
    # "nail the loop before parametrising" step), refresh its cache
    python -m examples.regression.parity_matrix_orca.run_parity --first --run

    # fill in every missing cache entry by running ORCA on compute-host-d
    python -m examples.regression.parity_matrix_orca.run_parity --run \
        --output-root ~/vibeqc-runs

    # just the cells whose id contains a substring
    python -m examples.regression.parity_matrix_orca.run_parity --cells H2O__def2-svp

CLAUDE.md § 10: ORCA runs out-of-process via vq; vibe-qc never imports
it. compute-host-d is a shared family machine — ORCA cells are submitted
serial (``--cpus 1``); be sparing with ``--run``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from examples.regression.core.output_paths import (
    RUNS_DIR_ENV,
    make_run_id,
    resolve_output_root,
)

from .cases import FIRST_CELL, PARITY_CELLS, Cell
from .orca_input import orca_simple_input
from .orca_vq import run_orca_cell

_PKG_DIR = Path(__file__).resolve().parent
FIXTURE_CACHE_DIR = _PKG_DIR / "cache"


@dataclass(frozen=True)
class OutputPaths:
    run_dir: Path
    report: Path
    cache_dir: Path
    work_dir: Path
    fetch_dir: Path


def build_output_paths(
    *,
    output_root: Optional[str],
    run_id: Optional[str],
    report: Optional[Path],
) -> OutputPaths:
    root = resolve_output_root(output_root, create=True)
    rid = run_id or make_run_id("orca-parity")
    run_dir = root / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        report.expanduser().resolve(strict=False)
        if report is not None
        else run_dir / "PARITY_REPORT.md"
    )
    return OutputPaths(
        run_dir=run_dir,
        report=report_path,
        cache_dir=run_dir / "cache",
        work_dir=run_dir / "vq_work",
        fetch_dir=run_dir / "vq_fetch",
    )


# --- cache io -----------------------------------------------------------

def cache_path(cell: Cell, *, cache_dir: Path = FIXTURE_CACHE_DIR) -> Path:
    return cache_dir / f"{cell.cell_id}.json"


def load_cache(
    cell: Cell,
    *,
    cache_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Return the cached ORCA decomposition for `cell`, or None on a miss."""
    candidates = []
    if cache_dir is not None:
        candidates.append(cache_path(cell, cache_dir=cache_dir))
    candidates.append(cache_path(cell, cache_dir=FIXTURE_CACHE_DIR))
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def write_cache(
    cell: Cell,
    orca: Dict[str, Any],
    *,
    cache_dir: Path,
) -> Path:
    """Persist a parsed ORCA decomposition plus provenance."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "cell": {
            "system": cell.system, "basis": cell.basis,
            "method": cell.method, "charge": cell.charge,
            "spin": cell.spin, "df": cell.df,
        },
        "orca_simple_input": orca_simple_input(cell),
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        # parse_orca_output payload + orca_vq provenance, verbatim.
        **orca,
    }
    path = cache_path(cell, cache_dir=cache_dir)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


# --- per-cell drive -----------------------------------------------------

def _obtain_orca(cell: Cell, *, run: bool, vq_kw: Dict[str, Any],
                 cache_dir: Path, workspace_root: Path, fetch_root: Path,
                 update_fixture_cache: bool = False,
                 ) -> Optional[Dict[str, Any]]:
    """Cached ORCA decomposition for `cell`, running it on compute-host-d if asked."""
    cached = load_cache(cell, cache_dir=cache_dir)
    if cached is not None and not run:
        return cached
    if cached is not None and run:
        print(f"[{cell.cell_id}] cache hit — re-running on compute-host-d (--run)")
    if cached is None and not run:
        return None
    # run on compute-host-d, refresh the cache
    parsed = run_orca_cell(
        cell, workspace_root=workspace_root, fetch_root=fetch_root, **vq_kw)
    path = write_cache(cell, parsed, cache_dir=cache_dir)
    print(f"[{cell.cell_id}] cache written -> {path}")
    if update_fixture_cache:
        fixture_path = write_cache(cell, parsed, cache_dir=FIXTURE_CACHE_DIR)
        print(f"[{cell.cell_id}] fixture cache updated -> {fixture_path}")
    return load_cache(cell, cache_dir=cache_dir)


def evaluate_cell(
    cell: Cell,
    *,
    run: bool,
    vq_kw: Dict[str, Any],
    cache_dir: Path,
    workspace_root: Path,
    fetch_root: Path,
    update_fixture_cache: bool = False,
) -> Dict[str, Any]:
    """Run/load ORCA, decompose vibe-qc, compare. Returns a verdict record."""
    record: Dict[str, Any] = {"cell": cell, "status": "pending"}
    try:
        orca = _obtain_orca(
            cell,
            run=run,
            vq_kw=vq_kw,
            cache_dir=cache_dir,
            workspace_root=workspace_root,
            fetch_root=fetch_root,
            update_fixture_cache=update_fixture_cache,
        )
    except Exception as exc:  # noqa: BLE001 - report per cell, don't crash the sweep
        record["status"] = "orca-error"
        record["note"] = f"{type(exc).__name__}: {exc}"
        return record
    if orca is None:
        record["status"] = "no-cache"
        record["note"] = "no cached ORCA result; pass --run to generate it"
        return record

    # Imported lazily: cache misses and ORCA submit failures do not need the
    # local native extension just to produce a report row.
    from .vibeqc_compare import compare, vibeqc_decompose

    try:
        vq = vibeqc_decompose(cell)
    except Exception as exc:  # noqa: BLE001
        record["status"] = "vibeqc-error"
        record["note"] = f"{type(exc).__name__}: {exc}"
        record["orca"] = orca
        return record

    verdict = compare(vq, orca)
    record["status"] = "pass" if verdict["ok"] else "FAIL"
    record["verdict"] = verdict
    record["orca"] = orca
    record["vibeqc"] = vq
    return record


# --- report -------------------------------------------------------------

def _fmt_report(records: List[Dict[str, Any]]) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")
    n_pass = sum(r["status"] == "pass" for r in records)
    n_fail = sum(r["status"] == "FAIL" for r in records)
    n_other = len(records) - n_pass - n_fail

    lines = [
        "# ORCA parity matrix — vibe-qc vs ORCA 6.1.1",
        "",
        f"_Generated {now} by "
        "`examples/regression/parity_matrix_orca/run_parity.py`._",
        "",
        "Per-intermediate energy decomposition, vibe-qc vs ORCA. ORCA's "
        "standard output does not split the Coulomb and exchange "
        "energies, so the Fock-build bucket is the combined "
        "`e_coulomb_plus_exchange` (= ORCA `Two Electron Energy - E(XC)`; "
        "vibe-qc `e_coulomb + e_exchange`).",
        "",
        f"**{n_pass} pass, {n_fail} fail, {n_other} not evaluated** "
        f"of {len(records)} cells.",
        "",
    ]

    for r in records:
        cell: Cell = r["cell"]
        lines.append(f"## {cell.cell_id} — {r['status']}")
        lines.append("")
        if r["status"] in ("no-cache", "orca-error", "vibeqc-error"):
            lines.append(f"_{r.get('note', '')}_")
            lines.append("")
            continue
        orca = r["orca"]
        lines.append(
            f"ORCA `{r['orca'].get('orca_simple_input', '?')}` "
            f"(v{orca.get('code_version', '?')}, "
            f"jobid `{orca.get('vq_jobid', 'cached')}`)"
        )
        lines.append("")
        verdict = r["verdict"]
        lines.append("| piece | vibe-qc | ORCA | |Δ| | tol | ok |")
        lines.append("|---|---:|---:|---:|---:|:--:|")
        for p in verdict["pieces"]:
            lines.append(
                f"| {p['piece']} | {p['vibeqc']:.8f} | {p['orca']:.8f} | "
                f"{p['delta']:.2e} | {p['tol']:.0e} | "
                f"{'✓' if p['ok'] else '✗'} |"
            )
        mo = verdict["mo"]
        lines.append(
            f"| mo (max, n={mo['n_compared']}) | | | "
            f"{mo['max_delta']:.2e} | {mo['tol']:.0e} | "
            f"{'✓' if mo['ok'] else '✗'} |"
        )
        lines.append("")

    return "\n".join(lines) + "\n"


# --- cli ----------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="ORCA axis of the HF/DFT parity matrix")
    ap.add_argument("--run", action="store_true",
                    help="run ORCA on compute-host-d for cache misses (and "
                         "refresh hits); default: compare cached only")
    ap.add_argument("--first", action="store_true",
                    help=f"only the handover's first cell "
                         f"({FIRST_CELL.cell_id})")
    ap.add_argument("--cells", default="",
                    help="substring filter on cell id")
    ap.add_argument(
        "--output-root",
        default=None,
        help=(
            "Directory that owns parity-matrix run artifacts. Precedence: "
            f"this option, then ${RUNS_DIR_ENV}, then ~/vibeqc-runs."
        ),
    )
    ap.add_argument("--run-id", default=None,
                    help="Override the auto-generated run id.")
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Markdown report path (default: <output-root>/<run-id>/PARITY_REPORT.md)",
    )
    ap.add_argument(
        "--update-fixture-cache",
        action="store_true",
        help=(
            "Also refresh examples/regression/parity_matrix_orca/cache. "
            "Normal --run cache artifacts stay in the output root."
        ),
    )
    ap.add_argument("--cpus", type=int, default=1,
                    help="cpus per ORCA job (compute-host-d courtesy: default 1)")
    ap.add_argument("--wall-time-seconds", type=int, default=1800,
                    help="ORCA job wall-time limit (mandatory for vq)")
    ap.add_argument("--poll-seconds", type=int, default=20)
    ap.add_argument("--timeout-seconds", type=int, default=7200,
                    help="give up waiting on a queued/running job after this")
    args = ap.parse_args(argv)

    if args.first:
        cells = [FIRST_CELL]
    else:
        cells = list(PARITY_CELLS)
    if args.cells:
        cells = [c for c in cells if args.cells in c.cell_id]
    if not cells:
        print("no cells selected", file=sys.stderr)
        return 2

    paths = build_output_paths(
        output_root=args.output_root,
        run_id=args.run_id,
        report=args.report,
    )

    vq_kw = {
        "cpus": args.cpus,
        "nprocs": args.cpus,
        "wall_time_s": args.wall_time_seconds,
        "poll_s": args.poll_seconds,
        "timeout_s": args.timeout_seconds,
    }

    print(f"evaluating {len(cells)} cell(s); "
          f"{'RUNNING ORCA on compute-host-d' if args.run else 'cached-only'}")
    print(f"artifacts -> {paths.run_dir}")
    records = []
    for cell in cells:
        print(f"--- {cell.cell_id} ---")
        rec = evaluate_cell(
            cell,
            run=args.run,
            vq_kw=vq_kw,
            cache_dir=paths.cache_dir,
            workspace_root=paths.work_dir,
            fetch_root=paths.fetch_dir,
            update_fixture_cache=args.update_fixture_cache,
        )
        print(f"    {rec['status']}"
              + (f": {rec['note']}" if rec.get("note") else ""))
        records.append(rec)

    report = _fmt_report(records)
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    paths.report.write_text(report, encoding="utf-8")
    print(f"\nreport -> {paths.report}")

    n_fail = sum(r["status"] == "FAIL" for r in records)
    n_eval = sum(r["status"] in ("pass", "FAIL") for r in records)
    print(f"{n_eval - n_fail}/{n_eval} evaluated cells pass")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
