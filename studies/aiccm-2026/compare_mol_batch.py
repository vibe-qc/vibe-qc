#!/usr/bin/env python
"""Aggregate results from the molecular batch and produce comparison tables.

Parses ``<stem>.out`` files (one per job) written by the ``batch_full_matrix.py``
workflow, extracts energies, convergence flags, iteration counts, and wall times,
then produces tables grouped by dimension.

Usage:
    # Parse a results directory and print text tables
    python compare_mol_batch.py results/

    # CSV output for spreadsheet import
    python compare_mol_batch.py results/ --csv > batch_results.csv

    # JSON output for further scripting
    python compare_mol_batch.py results/ --json > batch_results.json

    # Filter to a subset of the matrix
    python compare_mol_batch.py results/ --method rhf rks --basis def2-svp

    # Flag any failed or non-converged jobs
    python compare_mol_batch.py results/ --failures-only

Output tables (text mode):
  1. Per-molecule × per-method energy summary (all convergers averaged)
  2. Per-converger convergence rate + mean iterations
  3. Per-COSX level convergence rate + mean iterations
  4. COSX vs no-COSX energy differences (µHa) for identical method/basis/converger
  5. Full pivot table: method × basis × COSX × converger
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# ── Result model ────────────────────────────────────────────────────────────

@dataclass
class JobResult:
    molecule: str
    method: str
    basis: str
    cosx: str
    converger: str
    stem: str
    energy: float | None = None        # total energy (Ha)
    e_corr: float | None = None        # correlation energy (MP2/CCSD)
    e_t: float | None = None           # (T) correction
    converged: bool = False
    n_iter: int = 0                    # SCF iterations
    wall_s: float | None = None        # wall time (seconds)
    error: str | None = None           # parse error message

    @property
    def label(self) -> str:
        return f"{self.molecule}/{self.method}/{self.basis}/{self.cosx}/{self.converger}"


# ── Output file parser ──────────────────────────────────────────────────────

# vibe-qc .out banner lines we recognize
_RE_ENERGY = re.compile(
    r"(?:Total|Final)\s+(?:RHF|UHF|RKS|UKS|ROHF|ROKS)\s+energy\s*[:=]\s*"
    r"([+-]?\d+\.\d+)"
)
_RE_SCF_CONV = re.compile(r"SCF\s+converged", re.IGNORECASE)
_RE_SCF_ITERS = re.compile(
    r"(?:Converged|Stopped)\s+(?:in|after)\s+(\d+)\s+iterations?", re.IGNORECASE
)
_RE_WALL = re.compile(
    r"(?:Wall\s+time|Elapsed|Total\s+wall)\s*[:=]\s*([\d.]+)\s*s", re.IGNORECASE
)
# Backups: vibe-qc structured output lines
_RE_E_TOTAL = re.compile(r"E\(CCSD\(T\)\s+total\)\s*[:=]\s*([+-]?\d+\.\d+)")
_RE_E_CCSD_CORR = re.compile(r"E_corr\(CCSD\)\s*[:=]\s*([+-]?\d+\.\d+)")
_RE_E_T_INCR = re.compile(r"E\(\(T\)\s+increment\)\s*[:=]\s*([+-]?\d+\.\d+)")
_RE_MP2_CORR = re.compile(r"MP2\s+correlation\s+energy\s*[:=]\s*([+-]?\d+\.\d+)")
_RE_CC_CONV = re.compile(r"CCSD\s+converged\s*[:=]\s*(?:1|True|yes)", re.IGNORECASE)
_RE_CC_ITERS = re.compile(r"CCSD\s+n_iter\s*[:=]\s*(\d+)", re.IGNORECASE)


def _parse_stem(filename: str) -> dict[str, str] | None:
    """Parse molecule/method/basis/cosx/converger from a stem like
    ``h2o__rhf__def2-svp__legacy__kdiis``.
    
    Reverses the safe-name transformation applied by batch_full_matrix.py:
    ``ccsdt`` → ``ccsd(t)``, ``6-31gs`` → ``6-31g*``."""
    parts = filename.replace(".out", "").split("__")
    if len(parts) != 5:
        return None
    mol, method, basis, cosx, converger = parts
    # Reverse safe-stem transforms
    method = method.replace("ccsdt", "ccsd(t)")
    basis = basis.replace("6-31gs", "6-31g*")
    return dict(molecule=mol, method=method, basis=basis, cosx=cosx,
                converger=converger)


def parse_out_file(path: Path) -> JobResult | None:
    """Parse one .out file and return a JobResult."""
    stem = path.stem
    meta = _parse_stem(stem)
    if meta is None:
        return None

    try:
        text = path.read_text()
    except Exception as e:
        r = JobResult(stem=stem, error=str(e), **meta)
        return r

    result = JobResult(stem=stem, **meta)

    # Scan for energy
    for pattern in [_RE_E_TOTAL, _RE_ENERGY]:
        m = pattern.search(text)
        if m:
            result.energy = float(m.group(1))
            break

    # CCSD correlation
    m = _RE_E_CCSD_CORR.search(text)
    if m:
        result.e_corr = float(m.group(1))
    m = _RE_E_T_INCR.search(text)
    if m:
        result.e_t = float(m.group(1))

    # MP2 correlation
    m = _RE_MP2_CORR.search(text)
    if m:
        result.e_corr = float(m.group(1))

    # Convergence
    result.converged = bool(_RE_SCF_CONV.search(text))
    m = _RE_SCF_ITERS.search(text)
    if m:
        result.n_iter = int(m.group(1))

    # CCSD n_iter for post-SCF methods
    m_cc = _RE_CC_ITERS.search(text)
    if m_cc:
        result.n_iter = max(result.n_iter, int(m_cc.group(1)))

    # Wall time
    m = _RE_WALL.search(text)
    if m:
        result.wall_s = float(m.group(1))

    return result


# ── Analysis ────────────────────────────────────────────────────────────────

def collect_results(results_dir: str | Path,
                    methods: Sequence[str] | None = None,
                    molecules: Sequence[str] | None = None,
                    bases: Sequence[str] | None = None,
                    cosxes: Sequence[str] | None = None,
                    convergers: Sequence[str] | None = None,
                    ) -> list[JobResult]:
    """Collect and parse all .out files in a results directory."""
    results: list[JobResult] = []
    for path in sorted(Path(results_dir).glob("*.out")):
        r = parse_out_file(path)
        if r is None:
            continue
        if methods and r.method not in methods:
            continue
        if molecules and r.molecule not in molecules:
            continue
        if bases and r.basis not in bases:
            continue
        if cosxes and r.cosx not in cosxes:
            continue
        if convergers and r.converger not in convergers:
            continue
        results.append(r)
    return results


def filter_converged(results: list[JobResult]) -> list[JobResult]:
    return [r for r in results if r.converged and r.energy is not None]


def filter_failed(results: list[JobResult]) -> list[JobResult]:
    return [r for r in results if not r.converged or r.error is not None]


# ── Table formatters ────────────────────────────────────────────────────────

def _row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"

def _sep(n: int) -> str:
    return "|" + "|".join(["---"] * n) + "|"

def _fmt_e(e: float | None) -> str:
    if e is None:
        return "—"
    return f"{e:.6f}"

def _fmt_de(de: float | None) -> str:
    if de is None:
        return "—"
    return f"{de*1e6:+.1f}"


def table_energy_by_molecule_method(results: list[JobResult]) -> str:
    """Energy summary: rows=molecules, columns=methods (averaged over bases/COSX/convergers)."""
    mols = sorted({r.molecule for r in results})
    meths = sorted({r.method for r in results}, key=lambda m: ["rhf","rks","mp2","ccsd","ccsdt"].index(m) if m in ["rhf","rks","mp2","ccsd","ccsdt"] else 99)
    grid: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in filter_converged(results):
        grid[(r.molecule, r.method)].append(r.energy)

    lines = ["## Energy per atom by molecule × method (Ha)\n"]
    lines.append(_row("Molecule", *meths))
    lines.append(_sep(len(meths) + 1))
    for mol in mols:
        cells = [mol]
        for meth in meths:
            vals = grid.get((mol, meth), [])
            if vals:
                cells.append(f"{sum(vals)/len(vals):.6f}")
            else:
                cells.append("—")
        lines.append(_row(*cells))
    return "\n".join(lines)


def table_converger_stats(results: list[JobResult]) -> str:
    """Convergence rate + mean iterations per converger."""
    convs = sorted({r.converger for r in results})
    lines = ["## Converger performance\n"]
    lines.append(_row("Converger", "N jobs", "Converged", "Rate %", "Mean iters"))
    lines.append(_sep(5))
    for c in convs:
        subset = [r for r in results if r.converger == c]
        n = len(subset)
        conv = sum(1 for r in subset if r.converged)
        rate = 100.0 * conv / n if n else 0.0
        mean_iter = sum(r.n_iter for r in subset if r.converged) / max(conv, 1)
        lines.append(_row(c, str(n), str(conv), f"{rate:.1f}", f"{mean_iter:.1f}"))
    return "\n".join(lines)


def table_cosx_stats(results: list[JobResult]) -> str:
    """Convergence rate + mean iterations per COSX level."""
    cosxes = sorted({r.cosx for r in results})
    lines = ["## COSX / GridX performance\n"]
    lines.append(_row("COSX level", "N jobs", "Converged", "Rate %", "Mean iters"))
    lines.append(_sep(5))
    for c in cosxes:
        subset = [r for r in results if r.cosx == c]
        n = len(subset)
        conv = sum(1 for r in subset if r.converged)
        rate = 100.0 * conv / n if n else 0.0
        mean_iter = sum(r.n_iter for r in subset if r.converged) / max(conv, 1)
        lines.append(_row(c, str(n), str(conv), f"{rate:.1f}", f"{mean_iter:.1f}"))
    return "\n".join(lines)


def table_cosx_delta(results: list[JobResult]) -> str:
    """COSX vs no-COSX energy differences for identical (mol, method, basis, converger)."""
    off = {(r.molecule, r.method, r.basis, r.converger): r
           for r in filter_converged(results) if r.cosx == "off"}
    cosxes = sorted({r.cosx for r in results if r.cosx != "off"})
    lines = ["## COSX minus no-COSX energy difference (µHa)\n"]
    lines.append(_row("COSX level", "N pairs", "Mean ΔE (µHa)", "Max |ΔE| (µHa)"))
    lines.append(_sep(4))
    for c in cosxes:
        subset = [r for r in filter_converged(results) if r.cosx == c]
        deltas = []
        for r in subset:
            key = (r.molecule, r.method, r.basis, r.converger)
            ref = off.get(key)
            if ref and ref.energy is not None and r.energy is not None:
                deltas.append((r.energy - ref.energy) * 1e6)
        if deltas:
            lines.append(_row(c, str(len(deltas)),
                              f"{sum(deltas)/len(deltas):+.2f}",
                              f"{max(abs(d) for d in deltas):.2f}"))
        else:
            lines.append(_row(c, "0", "—", "—"))
    return "\n".join(lines)


def table_failures(results: list[JobResult]) -> str:
    """List all failed/non-converged jobs."""
    failed = filter_failed(results)
    if not failed:
        return "## Failures\n\nNone — all jobs converged.\n"
    lines = ["## Failures\n"]
    lines.append(_row("Job", "Energy", "Converged", "Iters", "Error"))
    lines.append(_sep(5))
    for r in failed:
        lines.append(_row(
            r.label, _fmt_e(r.energy),
            "yes" if r.converged else "NO",
            str(r.n_iter), r.error or "—",
        ))
    return "\n".join(lines)


# ── CSV / JSON export ───────────────────────────────────────────────────────

def export_csv(results: list[JobResult], stream=sys.stdout) -> None:
    writer = csv.writer(stream)
    writer.writerow(["molecule", "method", "basis", "cosx", "converger",
                     "energy_Ha", "e_corr_Ha", "e_t_Ha",
                     "converged", "n_iter", "wall_s", "error"])
    for r in results:
        writer.writerow([
            r.molecule, r.method, r.basis, r.cosx, r.converger,
            f"{r.energy:.10f}" if r.energy is not None else "",
            f"{r.e_corr:.10f}" if r.e_corr is not None else "",
            f"{r.e_t:.10f}" if r.e_t is not None else "",
            int(r.converged), r.n_iter,
            f"{r.wall_s:.1f}" if r.wall_s is not None else "",
            r.error or "",
        ])


def export_json(results: list[JobResult], stream=sys.stdout) -> None:
    data = []
    for r in results:
        data.append({
            "molecule": r.molecule, "method": r.method, "basis": r.basis,
            "cosx": r.cosx, "converger": r.converger,
            "energy_Ha": r.energy, "e_corr_Ha": r.e_corr, "e_t_Ha": r.e_t,
            "converged": r.converged, "n_iter": r.n_iter,
            "wall_s": r.wall_s, "error": r.error,
        })
    json.dump(data, stream, indent=2)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate and analyze molecular batch results")
    ap.add_argument("results_dir", help="directory containing .out files")
    ap.add_argument("--method", nargs="*", default=None)
    ap.add_argument("--molecule", nargs="*", default=None)
    ap.add_argument("--basis", nargs="*", default=None)
    ap.add_argument("--cosx", nargs="*", default=None)
    ap.add_argument("--converger", nargs="*", default=None)
    ap.add_argument("--csv", action="store_true", help="CSV export")
    ap.add_argument("--json", action="store_true", help="JSON export")
    ap.add_argument("--failures-only", action="store_true",
                    help="only show failed jobs")
    args = ap.parse_args()

    results = collect_results(
        args.results_dir,
        methods=args.method,
        molecules=args.molecule,
        bases=args.basis,
        cosxes=args.cosx,
        convergers=args.converger,
    )

    if not results:
        print(f"No .out files found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    if args.csv:
        export_csv(results)
        return

    if args.json:
        export_json(results)
        return

    if args.failures_only:
        print(table_failures(results))
        return

    # Text report
    print(f"# Molecular batch results — {len(results)} jobs parsed "
          f"({len(filter_converged(results))} converged)\n")
    print(table_energy_by_molecule_method(results))
    print()
    print(table_converger_stats(results))
    print()
    print(table_cosx_stats(results))
    print()
    print(table_cosx_delta(results))
    print()
    print(table_failures(results))


if __name__ == "__main__":
    main()
