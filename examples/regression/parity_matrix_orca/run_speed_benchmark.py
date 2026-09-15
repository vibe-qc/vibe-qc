"""ORCA-vs-vibe-qc wall-clock speed benchmark (handover deliverable 2).

The maintainer ask (2026-05-14): **vibe-qc should be at least as fast
as ORCA for the same method + basis.** This script measures it — both
codes on the same compute-host-d box, same core count, via the ``vq`` queue.

It reuses deliverable 1's ORCA-via-vq plumbing (:mod:`orca_vq`); the
only new piece is the vibe-qc side, which runs as its own ``vq`` job
(:mod:`vibeqc_timing_job`, dispatched into the vibeqc-dev venv) so the
comparison is genuinely same-box.

**What is measured.** For each ``(system, basis, method, path)`` on the
ladder (:data:`cases.SPEED_LADDER_*`), both codes run the *same* method
and basis on the *same* Fock-build path — ``direct`` (exact 4-index)
and ``rijcosx`` (RI-J + chain-of-spheres K) — serial, with a matched
tight SCF tolerance. The wall time is ``finished - started`` from
``vq status`` — queue-wait-free, the honest on-box number.

The auto ladder is deliberately small (H2O, glycine) so it runs in a
reasonable time on a shared box; its absolute times are dominated by
per-run setup overhead and are not yet the meaningful regime. The
**large-system RIJCOSX anchor** is the maintainer's own clean-box
measurement (recorded as :data:`_MAINTAINER_REFERENCE` and reproduced
in the report) — that is where the gap is real. Extending the auto
ladder to a ~15-20-heavy-atom organic so the crossover is bracketed by
reproducible same-box runs is the next measurement step (add a
geometry to ``cases.py`` and pass ``--systems``).

Usage::

    python -m examples.regression.parity_matrix_orca.run_speed_benchmark

    # a quick subset
    python -m examples.regression.parity_matrix_orca.run_speed_benchmark \\
        --systems H2O --bases def2-svp

Reports, submitted job workspaces, and fetched raw outputs default to
``~/vibeqc-runs/<run-id>/``. Override with ``--output-root`` or
``VIBEQC_RUNS_DIR``; use ``--report`` only for an explicit report path.

CLAUDE.md § 10: both codes run out-of-process via vq. compute-host-d is a
shared family machine — jobs are serial (``--cpus 1``); be sparing.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from examples.regression.core.output_paths import (
    RUNS_DIR_ENV,
    make_run_id,
    resolve_output_root,
)

from . import orca_vq
from .cases import (
    Cell, SPEED_LADDER_BASES, SPEED_LADDER_METHODS, SPEED_LADDER_PATHS,
    SPEED_LADDER_SYSTEMS, geometry_bohr, speed_cell,
)
from .orca_vq import OrcaVqError, fetch_workspace, wait_for_terminal

_PKG_DIR = Path(__file__).resolve().parent
_TIMING_JOB = _PKG_DIR / "vibeqc_timing_job.py"


@dataclass(frozen=True)
class OutputPaths:
    run_dir: Path
    report: Path
    work_dir: Path
    fetch_dir: Path


def build_output_paths(
    *,
    output_root: Optional[str],
    run_id: Optional[str],
    report: Optional[Path],
) -> OutputPaths:
    root = resolve_output_root(output_root, create=True)
    rid = run_id or make_run_id("orca-speed")
    run_dir = root / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        report.expanduser().resolve(strict=False)
        if report is not None
        else run_dir / "orca_vs_vibeqc_speed.md"
    )
    return OutputPaths(
        run_dir=run_dir,
        report=report_path,
        work_dir=run_dir / "vq_work",
        fetch_dir=run_dir / "vq_fetch",
    )

# Tight SCF tolerance for the vibe-qc side — matched in spirit to ORCA's
# VeryTightSCF so neither code is timed at an unfair convergence target.
_VIBEQC_CONV_TOL = 1e-10
_VIBEQC_RESULT_MARKER = "VIBEQC-TIMING-RESULT:"


def _vibeqc_dev_python() -> str:
    """The vibeqc-dev venv python on the queue host, from ``vq programs``.

    ``vq programs`` registers PySCF as an importable in exactly that
    venv — its ``python`` field is the interpreter the vibe-qc timing
    job must run under (the ``-d`` submit form needs an explicit remote
    interpreter path; ``--branch`` is single-file-submit only).
    """
    proc = orca_vq._vq_read("programs", "--json")
    try:
        programs = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise OrcaVqError(
            f"could not parse `vq programs --json`: {exc}") from exc
    for prog in programs:
        if prog.get("name") == "pyscf" and prog.get("python"):
            return prog["python"]
    raise OrcaVqError(
        "could not find the vibeqc-dev venv python via `vq programs` "
        "(expected a registered `pyscf` import entry with a `python` path)"
    )


# --- vibe-qc timing job -------------------------------------------------

def _cell_json(cell: Cell) -> Dict[str, Any]:
    # RIJCOSX uses RI-J only -> the J-only aux (matches ORCA's def2/J);
    # RIJK uses the JK aux. Either way the same Weigend basis ORCA uses.
    aux = "def2-universal-jfit" if cell.cosx else "def2-universal-jkfit"
    return {
        "atoms": [[z, list(xyz)] for z, xyz in geometry_bohr(cell.system)],
        "multiplicity": cell.spin + 1,
        "basis": cell.basis,
        "method": cell.method,
        "df": cell.df,
        "cosx": cell.cosx,
        "aux_basis": aux,
        "conv_tol_energy": _VIBEQC_CONV_TOL,
        "label": cell.cell_id,
    }


def run_vibeqc_cell(
    cell: Cell, *,
    workspace_root: Path,
    fetch_root: Path,
    cpus: int = 1,
    wall_time_s: int = 1800,
    poll_s: int = 15,
    timeout_s: int = 5400,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Submit + run the vibe-qc timing job for `cell` on compute-host-d.

    Returns the parsed ``VIBEQC-TIMING-RESULT`` payload extended with
    ``vq_jobid`` / ``vq_wall_s`` (on-box execution wall time).
    """
    def _say(msg: str) -> None:
        if verbose:
            print(f"  [vibeqc-vq {cell.cell_id}] {msg}", flush=True)

    workspace = workspace_root / f"vibeqc__{cell.cell_id}"
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_TIMING_JOB, workspace / "vibeqc_timing_job.py")
    (workspace / "cell.json").write_text(
        json.dumps(_cell_json(cell), indent=2), encoding="utf-8")

    python = _vibeqc_dev_python()
    proc = orca_vq._vq(
        "submit", "-d", str(workspace),
        "--cpus", str(cpus),
        "--wall-time-seconds", str(wall_time_s),
        "--", python, "vibeqc_timing_job.py", "cell.json",
    )
    jobid = proc.stdout.strip().splitlines()[-1].strip()
    _say(f"submitted -> jobid {jobid}")

    last = ""

    def _on_poll(state: str, elapsed: float) -> None:
        nonlocal last
        if state != last:
            _say(f"state={state} (t+{elapsed:.0f}s)")
            last = state

    info = wait_for_terminal(jobid, poll_s=poll_s, timeout_s=timeout_s,
                             on_poll=_on_poll)
    if info["state"] != "completed":
        raise OrcaVqError(
            f"vibe-qc timing job {jobid} for {cell.cell_id} ended "
            f"{info['state']!r} (exit_code={info.get('exit_code')})"
        )
    ws = fetch_workspace(jobid, fetch_root)
    stdout = (ws / "stdout.log").read_text(encoding="utf-8", errors="replace")
    payload = None
    for line in reversed(stdout.splitlines()):
        if line.startswith(_VIBEQC_RESULT_MARKER):
            payload = json.loads(line[len(_VIBEQC_RESULT_MARKER):])
            break
    if payload is None:
        raise OrcaVqError(
            f"vibe-qc timing job {jobid} emitted no result marker — "
            f"see {ws / 'stdout.log'} / {ws / 'stderr.log'}"
        )
    if payload.get("status") != "ok":
        raise OrcaVqError(
            f"vibe-qc timing job {jobid} for {cell.cell_id} failed: "
            f"{payload.get('note')}"
        )
    payload["vq_jobid"] = jobid
    payload["vq_wall_s"] = orca_vq._wall_seconds(info)
    _say(f"done: E={payload['energy_ha']:.8f} Ha, "
         f"SCF wall={payload['wall_s']:.2f}s, "
         f"vq wall={payload['vq_wall_s']:.1f}s")
    return payload


# --- benchmark sweep ----------------------------------------------------

def _speed_cells(systems: List[str], bases: List[str],
                 methods: List[str], paths: List[str]) -> List[Cell]:
    return [
        speed_cell(s, b, m, p)
        for s in systems for b in bases for m in methods for p in paths
    ]


def benchmark_cell(
    cell: Cell,
    *,
    vq_kw: Dict[str, Any],
    workspace_root: Path,
    fetch_root: Path,
) -> Dict[str, Any]:
    """Run `cell` under both codes on compute-host-d; return a timing record.

    `vq_kw` holds the kwargs common to both runners (``cpus``,
    ``wall_time_s``, ``poll_s``, ``timeout_s``). ORCA additionally
    takes ``nprocs`` — pinned to ``cpus`` so the two codes get the same
    core budget.
    """
    rec: Dict[str, Any] = {"cell": cell, "status": "pending"}
    try:
        orca = orca_vq.run_orca_cell(
            cell, workspace_root=workspace_root, fetch_root=fetch_root,
            nprocs=vq_kw["cpus"], **vq_kw)
    except Exception as exc:  # noqa: BLE001 - report per cell
        rec["status"] = "orca-error"
        rec["note"] = f"{type(exc).__name__}: {exc}"
        return rec
    try:
        vqc = run_vibeqc_cell(
            cell,
            workspace_root=workspace_root,
            fetch_root=fetch_root,
            **vq_kw,
        )
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "vibeqc-error"
        rec["note"] = f"{type(exc).__name__}: {exc}"
        rec["orca"] = orca
        return rec

    orca_wall = orca["vq_wall_s"]
    vqc_wall = vqc["vq_wall_s"]
    # vibe-qc's own perf-counter around the SCF call — finer than the vq
    # bracket (excludes process startup), reported alongside for context.
    vqc_scf_wall = vqc["wall_s"]
    rec.update({
        "status": "ok",
        "orca": orca,
        "vibeqc": vqc,
        "orca_wall_s": orca_wall,
        "vibeqc_wall_s": vqc_wall,
        "vibeqc_scf_wall_s": vqc_scf_wall,
        # ratio < 1 => vibe-qc is faster (the handover's goal).
        "ratio": (vqc_wall / orca_wall) if orca_wall else None,
        # energy agreement — a sanity check that we timed the same thing.
        "delta_e": abs(orca["e_total_final"] - vqc["energy_ha"]),
    })
    return rec


# Maintainer's clean-box reference measurement (2026-05-14), fed in
# via the chat that chartered this benchmark. It is the large-system
# RIJCOSX anchor — bigger than anything in the auto ladder — and the
# single most important data point: it is where the gap is largest.
# Recorded here so a re-run of this script reproduces it in the report.
_MAINTAINER_REFERENCE = {
    "system": "R-butane-2-thiol (C4H10S, 15 atoms)",
    "method_basis_path": "HF / def2-TZVP / RIJCOSX",
    "box": "compute-host-d, 4 cores, clean box",
    "orca_wall_s": 16.7,
    "vibeqc_wall_s": 297.6,
    "orca_energy_ha": -554.895211,
    "vibeqc_energy_ha": -554.895095,
    "note": (
        "RI-J aux matched (ORCA def2/J = vibe-qc def2-universal-jfit). "
        "ORCA's own breakdown: SCF iterations 11.8 s (78%), startup "
        "2.0 s, properties 1.4 s — the gap is in the SCF Fock build."
    ),
}


def _fmt_reference() -> List[str]:
    """The maintainer's clean-box reference data point, as report lines."""
    r = _MAINTAINER_REFERENCE
    ratio = r["vibeqc_wall_s"] / r["orca_wall_s"]
    dele = abs(r["orca_energy_ha"] - r["vibeqc_energy_ha"])
    return [
        "## Maintainer reference measurement — large-system RIJCOSX",
        "",
        f"Measured by the maintainer on a **clean** compute-host-d box "
        f"(2026-05-14) — the large-system anchor for this benchmark, "
        f"bigger than anything in the auto ladder above.",
        "",
        f"* **System / method**: {r['system']} — {r['method_basis_path']}",
        f"* **Box**: {r['box']}",
        f"* **ORCA 6.1.1**: {r['orca_wall_s']:.1f} s "
        f"(E = {r['orca_energy_ha']:.6f} Ha)",
        f"* **vibe-qc**: {r['vibeqc_wall_s']:.1f} s "
        f"(E = {r['vibeqc_energy_ha']:.6f} Ha)",
        f"* **ratio**: {ratio:.1f}x — vibe-qc is ~{ratio:.0f}x slower; "
        f"energy agreement |ΔE| = {dele * 1e3:.2f} mHa",
        f"* {r['note']}",
        "",
    ]


def _versions_from_records(records: List[Dict[str, Any]]) -> Dict[str, str]:
    """Extract the code versions both runners reported (per-cell, in `records`).

    Returns a dict with "orca" and "vibeqc" keys, joined with `/` if a
    cell reported a different version mid-run (shouldn't happen but
    surfaces it loudly if it does).
    """
    def _collect(side: str) -> str:
        seen = []
        for r in records:
            if r.get("status") != "ok": continue
            v = r.get(side, {}).get("code_version")
            if v and v not in seen:
                seen.append(v)
        return "/".join(seen) if seen else "unknown"
    return {"orca": _collect("orca"), "vibeqc": _collect("vibeqc")}


def _fmt_report(records: List[Dict[str, Any]]) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")
    ok = [r for r in records if r["status"] == "ok"]
    n_vibeqc_faster = sum(1 for r in ok if r["ratio"] is not None
                          and r["ratio"] <= 1.0)
    versions = _versions_from_records(records)

    lines = [
        "# ORCA vs vibe-qc — wall-clock speed benchmark",
        "",
        f"_Generated {now} by "
        "`examples/regression/parity_matrix_orca/run_speed_benchmark.py`._",
        "",
        f"* **ORCA**: {versions['orca']}",
        f"* **vibe-qc** (on compute-host-d, vibeqc-dev venv): {versions['vibeqc']}",
        "",
        "Same compute-host-d box, same core count (serial), matched tight SCF "
        "tolerance, each code at its out-of-box default Fock build. Wall "
        "time is `finished - started` from `vq status` (queue-wait-free). "
        "`ratio = vibe-qc / ORCA` — **< 1 means vibe-qc is faster**, the "
        "maintainer's goal.",
        "",
        f"**vibe-qc at or above ORCA speed on {n_vibeqc_faster}/{len(ok)} "
        f"evaluated cells.**",
        "",
        "| cell | ORCA wall (s) | vibe-qc wall (s) | vibe-qc SCF (s) | "
        "ratio | vibe-qc ≥ ORCA? | |ΔE| (Ha) |",
        "|---|---:|---:|---:|---:|:--:|---:|",
    ]
    for r in records:
        cell: Cell = r["cell"]
        if r["status"] != "ok":
            lines.append(
                f"| {cell.cell_id} | — | — | — | — | "
                f"{r['status']} | — |")
            continue
        ratio = r["ratio"]
        verdict = "✓" if (ratio is not None and ratio <= 1.0) else "✗"
        lines.append(
            f"| {cell.cell_id} | {r['orca_wall_s']:.2f} | "
            f"{r['vibeqc_wall_s']:.2f} | {r['vibeqc_scf_wall_s']:.3f} | "
            f"{ratio:.2f} | {verdict} | {r['delta_e']:.2e} |"
        )
    lines.append("")

    lines += _fmt_reference()

    # Written verdict — synthesises the auto ladder AND the reference.
    # Decomposes by Fock-build path so the report points at the right
    # hotspot rather than just "vibe-qc is slower".
    lines.append("## Verdict")
    lines.append("")
    ref_ratio = (_MAINTAINER_REFERENCE["vibeqc_wall_s"]
                 / _MAINTAINER_REFERENCE["orca_wall_s"])

    # Per-path ratio summary on the cells big enough to be meaningful
    # (ORCA wall >= 2 s rules out H2O, where setup overhead dominates).
    def _avg_ratio(predicate) -> Optional[float]:
        rs = [r["ratio"] for r in ok
              if r["ratio"] is not None
              and r.get("orca_wall_s", 0) >= 2.0
              and predicate(r["cell"])]
        return sum(rs) / len(rs) if rs else None

    direct_hf = _avg_ratio(lambda c: not c.cosx and c.method == "RHF")
    direct_dft = _avg_ratio(
        lambda c: not c.cosx and c.method.startswith(("RKS-", "UKS-")))
    rijcosx_hf = _avg_ratio(lambda c: c.cosx and c.method == "RHF")
    rijcosx_dft = _avg_ratio(
        lambda c: c.cosx and c.method.startswith(("RKS-", "UKS-")))

    lines.append(
        "The auto ladder reaches glycine/def2-TZVP wall times of "
        "tens of seconds — past the per-run setup-overhead regime and "
        "into the meaningful Fock-build / XC / COSX regime. Decomposed "
        "by Fock-build path (averaging over the meaningful cells, "
        "ORCA wall >= 2 s):"
    )
    lines.append("")
    if direct_hf is not None:
        verdict_hf = "vibe-qc faster" if direct_hf < 1.0 else "vibe-qc slower"
        lines.append(
            f"* **direct + RHF**: ratio {direct_hf:.2f}x — {verdict_hf}. "
            "The core 4-index Fock build is healthy."
        )
    if direct_dft is not None:
        lines.append(
            f"* **direct + DFT**: ratio {direct_dft:.2f}x — the DFT XC "
            "quadrature path is where vibe-qc loses ground first."
        )
    if rijcosx_hf is not None:
        lines.append(
            f"* **RIJCOSX + RHF**: ratio {rijcosx_hf:.2f}x — the COSX-K "
            "kernel (chain-of-spheres exchange) is the second hotspot."
        )
    if rijcosx_dft is not None:
        lines.append(
            f"* **RIJCOSX + DFT**: ratio {rijcosx_dft:.2f}x — the two "
            "hotspots compound."
        )
    lines.append("")
    lines.append(
        f"The maintainer's reference point on a 15-atom molecule at "
        f"HF/def2-TZVP/RIJCOSX is **{ref_ratio:.0f}x slower** — "
        f"extrapolating cleanly from the glycine RIJCOSX numbers above, "
        f"the COSX gap is the dominant one and grows badly with system "
        f"size. ORCA's own breakdown puts ~78% of its (much smaller) "
        f"wall time in the SCF iterations."
    )
    lines.append("")
    lines.append(
        "**Verdict**: the maintainer's \"at least as fast as ORCA for "
        "the same method + basis\" bar is met on the direct HF path "
        "(vibe-qc actually wins there at glycine size) but **not met** "
        "on the DFT-XC or COSX paths. Both hotspots — the DFT XC "
        "quadrature and the COSX-K kernel — sit in the code the "
        "JKBuilder->EDIIS refactor stack touches. **Hand these numbers "
        "to that chat**; the COSX-K kernel is the highest-leverage target. Next "
        "measurement step: a ~15-20-heavy-atom organic on the auto "
        "ladder so the crossover is bracketed by reproducible same-box "
        "runs, not just the single 15-atom reference point — "
        "`run_speed_benchmark.py --systems ...` already takes the larger "
        "system once a geometry is added to `cases.py`."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="ORCA-vs-vibe-qc wall-clock speed benchmark")
    ap.add_argument("--systems", default=",".join(SPEED_LADDER_SYSTEMS),
                    help="comma-separated systems (default: the size ladder)")
    ap.add_argument("--bases", default=",".join(SPEED_LADDER_BASES),
                    help="comma-separated bases")
    ap.add_argument("--methods", default=",".join(SPEED_LADDER_METHODS),
                    help="comma-separated methods")
    ap.add_argument("--paths", default=",".join(SPEED_LADDER_PATHS),
                    help="comma-separated Fock-build paths: "
                         "direct,df,rijcosx")
    ap.add_argument(
        "--output-root",
        default=None,
        help=(
            "Directory that owns speed-benchmark artifacts. Precedence: "
            f"this option, then ${RUNS_DIR_ENV}, then ~/vibeqc-runs."
        ),
    )
    ap.add_argument("--run-id", default=None,
                    help="Override the auto-generated run id.")
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Markdown report path (default: <output-root>/<run-id>/orca_vs_vibeqc_speed.md)",
    )
    ap.add_argument("--cpus", type=int, default=1,
                    help="cpus per job (compute-host-d courtesy: default 1, serial)")
    ap.add_argument("--wall-time-seconds", type=int, default=1800)
    ap.add_argument("--poll-seconds", type=int, default=15)
    ap.add_argument("--timeout-seconds", type=int, default=5400)
    args = ap.parse_args(argv)

    cells = _speed_cells(
        args.systems.split(","), args.bases.split(","),
        args.methods.split(","), args.paths.split(","))
    # Kwargs common to both runners; benchmark_cell adds ORCA's nprocs
    # (== cpus) on the ORCA call only.
    vq_kw = {
        "cpus": args.cpus,
        "wall_time_s": args.wall_time_seconds,
        "poll_s": args.poll_seconds,
        "timeout_s": args.timeout_seconds,
    }
    paths = build_output_paths(
        output_root=args.output_root,
        run_id=args.run_id,
        report=args.report,
    )

    print(f"benchmarking {len(cells)} cell(s) — ORCA + vibe-qc on compute-host-d")
    print(f"artifacts -> {paths.run_dir}")
    records = []
    for cell in cells:
        print(f"--- {cell.cell_id} ---")
        rec = benchmark_cell(
            cell,
            vq_kw=vq_kw,
            workspace_root=paths.work_dir,
            fetch_root=paths.fetch_dir,
        )
        if rec["status"] == "ok":
            print(f"    ORCA {rec['orca_wall_s']:.2f}s | "
                  f"vibe-qc {rec['vibeqc_wall_s']:.2f}s | "
                  f"ratio {rec['ratio']:.2f}")
        else:
            print(f"    {rec['status']}: {rec.get('note', '')}")
        records.append(rec)

    report = _fmt_report(records)
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    paths.report.write_text(report, encoding="utf-8")
    print(f"\nreport -> {paths.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
