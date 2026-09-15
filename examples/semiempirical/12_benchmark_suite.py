"""Semiempirical benchmark suite — systematic validation across methods.

Run:  .venv/bin/python examples/semiempirical/12_benchmark_suite.py

Produces a markdown report at examples/semiempirical/12_benchmark_suite.md
comparing energies, iteration counts, and convergence across all molecular
semiempirical methods (DFTB0, SCC-DFTB, GFN2-xTB, PM6) for the standard
validation set (H2, H2O, CH4, NH3, CO2, C2H4, H2O dimer).

Historical semiempirical validation sweep; current status is documented in
docs/user_guide/semiempirical.md.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from vibeqc import Atom, Molecule
from vibeqc.semiempirical.dftb0 import DFTB0Model
from vibeqc.semiempirical.methods.gfn2 import GFN2Model
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params
from vibeqc.semiempirical.methods.pm6 import PM6Model
from vibeqc.semiempirical.dftb0 import SCCDFTBModel

HERE = Path(__file__).resolve().parent

# ── Validation set ──────────────────────────────────────────────────────

THETA = np.deg2rad(104.5 / 2)
R_OH = 1.81  # bohr


def _h2():
    return Molecule([Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])])


def _h2o():
    return Molecule(
        [
            Atom(8, [0, 0, 0]),
            Atom(1, [R_OH * np.sin(THETA), R_OH * np.cos(THETA), 0]),
            Atom(1, [-R_OH * np.sin(THETA), R_OH * np.cos(THETA), 0]),
        ]
    )


def _ch4():
    d = 1.186  # bohr
    return Molecule(
        [
            Atom(6, [0, 0, 0]),
            Atom(1, [d, d, d]),
            Atom(1, [-d, -d, d]),
            Atom(1, [-d, d, -d]),
            Atom(1, [d, -d, -d]),
        ]
    )


def _nh3():
    r = 1.91  # bohr
    h = 0.72  # bohr
    return Molecule(
        [
            Atom(7, [0, 0, 0]),
            Atom(1, [0, 0, r]),
            Atom(1, [r * np.sin(np.deg2rad(112)), 0, -h]),
            Atom(1, [-r * np.sin(np.deg2rad(112)), 0, -h]),
        ]
    )


def _co2():
    r_co = 2.20
    return Molecule(
        [
            Atom(6, [0, 0, 0]),
            Atom(8, [r_co, 0, 0]),
            Atom(8, [-r_co, 0, 0]),
        ]
    )


def _c2h4():
    r_cc = 2.53
    r_ch = 2.05
    a = np.deg2rad(121.3)
    return Molecule(
        [
            Atom(6, [0, 0, 0]),
            Atom(6, [r_cc, 0, 0]),
            Atom(1, [-r_ch * np.cos(a / 2), r_ch * np.sin(a / 2), 0]),
            Atom(1, [-r_ch * np.cos(a / 2), -r_ch * np.sin(a / 2), 0]),
            Atom(1, [r_cc + r_ch * np.cos(a / 2), r_ch * np.sin(a / 2), 0]),
            Atom(1, [r_cc + r_ch * np.cos(a / 2), -r_ch * np.sin(a / 2), 0]),
        ]
    )


def _h2o_dimer():
    r_oo = 5.6
    return Molecule(
        [
            Atom(8, [0, 0, 0]),
            Atom(1, [R_OH * np.sin(THETA), R_OH * np.cos(THETA), 0]),
            Atom(1, [-R_OH * np.sin(THETA), R_OH * np.cos(THETA), 0]),
            Atom(8, [r_oo, 0, 0]),
            Atom(1, [r_oo + R_OH * np.sin(THETA), R_OH * np.cos(THETA), 0]),
            Atom(1, [r_oo - R_OH * np.sin(THETA), R_OH * np.cos(THETA), 0]),
        ]
    )


MOLECULES = {
    "H2": _h2,
    "H2O": _h2o,
    "CH4": _ch4,
    "NH3": _nh3,
    "CO2": _co2,
    "C2H4": _c2h4,
}

# ── Benchmark runner ────────────────────────────────────────────────────


def run_benchmarks() -> list[dict]:
    """Run all methods on all molecules, return list of result dicts."""
    gfn2_params = load_gfn2_params()
    rows = []

    for name, mol_fn in MOLECULES.items():
        mol = mol_fn()

        # DFTB0 (non-iterative, always converges)
        t0 = time.perf_counter()
        try:
            m = DFTB0Model(mol)
            e = m.energy()
            dt = time.perf_counter() - t0
            rows.append({
                "molecule": name, "method": "DFTB0",
                "energy": e, "n_iter": 1, "converged": True,
                "time_s": dt, "error": None,
            })
        except Exception as exc:
            rows.append({
                "molecule": name, "method": "DFTB0",
                "energy": None, "n_iter": 0, "converged": False,
                "time_s": 0, "error": str(exc),
            })

        # SCC-DFTB
        t0 = time.perf_counter()
        try:
            m = SCCDFTBModel(mol)
            e = m.energy()
            dt = time.perf_counter() - t0
            n_iter = getattr(m, 'n_iter', 1)
            converged = getattr(m, 'converged', True)
            rows.append({
                "molecule": name, "method": "SCC-DFTB",
                "energy": e, "n_iter": n_iter, "converged": converged,
                "time_s": dt, "error": None,
            })
        except Exception as exc:
            rows.append(
                {
                    "molecule": name,
                    "method": "SCC-DFTB",
                    "energy": None,
                    "n_iter": 0,
                    "converged": False,
                    "time_s": 0,
                    "error": str(exc),
                }
            )

        # GFN2-xTB (Broyden default; fall back to Simple if needed)
        t0 = time.perf_counter()
        try:
            m = GFN2Model(mol, gfn2_params, warn=False)
            e = m.energy()

            dt = time.perf_counter() - t0
            rows.append(
                {
                    "molecule": name,
                    "method": "GFN2-xTB",
                    "energy": e,
                    "n_iter": m.n_iter,
                    "converged": m.converged,
                    "time_s": dt,
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "molecule": name,
                    "method": "GFN2-xTB",
                    "energy": None,
                    "n_iter": 0,
                    "converged": False,
                    "time_s": 0,
                    "error": str(exc),
                }
            )

        # PM6
        t0 = time.perf_counter()
        try:
            m = PM6Model(mol)
            e = m.energy()
            dt = time.perf_counter() - t0
            rows.append(
                {
                    "molecule": name,
                    "method": "PM6",
                    "energy": e,
                    "n_iter": m.n_iter,
                    "converged": m.converged,
                    "time_s": dt,
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "molecule": name,
                    "method": "PM6",
                    "energy": None,
                    "n_iter": 0,
                    "converged": False,
                    "time_s": 0,
                    "error": str(exc),
                }
            )

    return rows


# ── Report generation ───────────────────────────────────────────────────


def _format_row(r: dict) -> str:
    if r["error"]:
        return (
            f"| {r['molecule']} | {r['method']} | ❌ FAIL | — | — | {r['error'][:60]} |"
        )
    conv = "✅" if r["converged"] else "❌"
    return (
        f"| {r['molecule']:8s} | {r['method']:10s} | {conv:4s} | "
        f"{r['n_iter']:3d} | {r['energy']:12.6f} | {r['time_s']:.4f}s |"
    )


def generate_report(rows: list[dict]) -> str:
    lines = [
        "# Semiempirical benchmark suite",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        f"**Molecules:** {', '.join(MOLECULES)}",
        f"**Methods:** DFTB0, SCC-DFTB, GFN2-xTB, PM6",
        "",
        "## Results",
        "",
        "| Molecule | Method | Conv | Iters | Energy (Ha) | Wall time |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["molecule"], x["method"])):
        lines.append(_format_row(r))

    # Summary
    lines.append("")
    lines.append("## Summary")
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r["converged"] and not r["error"])
    n_fail = n_total - n_ok
    lines.append(f"- **{n_ok}/{n_total}** calculations converged successfully")
    if n_fail:
        lines.append(f"- **{n_fail}** failures (see table above)")
    lines.append(f"- Total wall time: {sum(r['time_s'] for r in rows):.2f}s")

    # Per-method summary
    lines.append("")
    lines.append("### Per-method convergence")
    lines.append("| Method | Converged | Avg iters | Avg time |")
    lines.append("|---|---:|---:|---:|")
    for method in ["DFTB0", "SCC-DFTB", "GFN2-xTB", "PM6"]:
        method_rows = [r for r in rows if r["method"] == method]
        n_conv = sum(1 for r in method_rows if r["converged"] and not r["error"])
        avg_iter = (
            np.mean([r["n_iter"] for r in method_rows if r["converged"]])
            if n_conv
            else 0
        )
        avg_time = np.mean([r["time_s"] for r in method_rows])
        lines.append(
            f"| {method:10s} | {n_conv}/{len(method_rows)} | "
            f"{avg_iter:.1f} | {avg_time:.3f}s |"
        )

    return "\n".join(lines) + "\n"


# ── Main ────────────────────────────────────────────────────────────────


def main():
    print("Running semiempirical benchmark suite...")
    print(f"  Molecules: {len(MOLECULES)}")
    print(f"  Methods: DFTB0, SCC-DFTB, GFN2-xTB, PM6")
    print()

    rows = run_benchmarks()
    report = generate_report(rows)

    out_path = HERE / "12_benchmark_suite.md"
    out_path.write_text(report)
    print(report)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
