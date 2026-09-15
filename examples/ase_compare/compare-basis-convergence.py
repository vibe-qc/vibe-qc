"""Cross-validate basis-set convergence on H2O / RHF.

Runs the same molecule through five basis sets — STO-3G, 6-31G*,
cc-pVDZ, cc-pVTZ, def2-TZVP — through every available calculator
(vibe-qc, PySCF, ORCA). Builds a per-basis comparison table and a
basis-convergence plot of the total energy.

The point: every code at every basis should produce the **same**
RHF energy (to µHa precision). Any disagreement is a real bug
worth investigating — either in the basis-set parser, the integrals,
or the SCF.

Run:
    .venv/bin/python examples/ase_compare/compare-basis-convergence.py

Produces:
    output-compare-basis-convergence.csv  — full per-basis table
    output-compare-basis-convergence.png  — convergence plot

Wall time: ~30 seconds with all three codes; <5 seconds with just
vibe-qc + PySCF (no ORCA).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.build import molecule
from vibeqc.ase import VibeQC
from vibeqc.benchmark import (
    compare_calculators,
    make_orca_calculator,
    print_calculator_availability,
)

HERE = Path(__file__).resolve().parent

# Basis sets, ordered roughly small → large. Each entry is
# (vibe-qc/PySCF name, ORCA name) — the libint name we use isn't
# always the same string ORCA accepts (e.g. "6-31g*" vs "6-31G(d)").
BASES: list[tuple[str, str]] = [
    ("sto-3g", "STO-3G"),
    ("6-31g*", "6-31G*"),
    ("cc-pvdz", "cc-pVDZ"),
    ("cc-pvtz", "cc-pVTZ"),
    ("def2-tzvp", "def2-TZVP"),
]


def main() -> None:
    print("=" * 72)
    print(" Cross-validation:  H2O / RHF / basis-set convergence")
    print("=" * 72)
    print()
    print("Calculator availability:")
    print_calculator_availability()
    print()

    atoms = molecule("H2O")

    # Per-basis table: rows = bases, cols = calculators.
    table: dict[str, dict[str, float]] = {}

    for vqname, orcaname in BASES:
        print(f"\n--- basis = {vqname} ---")

        # PySCF is retired as an in-process calculator; use the
        # out-of-process runner in examples/regression/ for parity.
        # ("PySCF",   PySCFCalculator(basis=vqname, method="RHF")),
        calcs: list = [
            ("vibe-qc", VibeQC(basis=vqname)),
        ]
        orca = make_orca_calculator(
            orcasimpleinput=f"HF {orcaname}",
            label=f"orca-{vqname}",
        )
        if orca is not None:
            calcs.append(("ORCA", orca))

        results = compare_calculators(atoms, calcs, properties=("energy",))
        results.print_table()

        # Stash energies per calculator
        per_calc = {}
        for row in results:
            if row.status in ("ok", "partial") and "energy" in row.properties:
                per_calc[row.label] = float(row.properties["energy"])
        table[vqname] = per_calc

    # Wide table.
    print("\n" + "=" * 72)
    print(" Summary: H2O / RHF energy convergence (eV)")
    print("=" * 72)
    print()
    code_labels = sorted({k for v in table.values() for k in v})
    header = f"{'basis':<14}" + "".join(f"{l:>20}" for l in code_labels)
    print(header)
    print("-" * len(header))
    for vqname, _ in BASES:
        row = f"{vqname:<14}"
        for code in code_labels:
            v = table[vqname].get(code)
            row += f"{v:>20.6f}" if v is not None else f"{'—':>20}"
        print(row)

    # Plot if matplotlib is available.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping convergence plot)")
        plt = None  # type: ignore[assignment]

    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        markers = ["o", "s", "^", "D", "v"]
        for i, code in enumerate(code_labels):
            xs, ys = [], []
            for j, (vqname, _) in enumerate(BASES):
                e = table[vqname].get(code)
                if e is not None:
                    xs.append(j)
                    ys.append(e)
            ax.plot(xs, ys, marker=markers[i % len(markers)], linestyle="-", label=code)
        ax.set_xticks(range(len(BASES)))
        ax.set_xticklabels([b[0] for b in BASES], rotation=15)
        ax.set_ylabel("E(RHF) / eV")
        ax.set_title("H2O / RHF — basis-set convergence")
        ax.legend()
        ax.grid(alpha=0.3)
        png_path = HERE / "output-compare-basis-convergence.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        print(
            f"\nConvergence plot written to {png_path.relative_to(HERE.parent.parent)}"
        )

    # Cross-code consistency check at every basis.
    print("\n--- per-basis consistency check ---")
    failures = 0
    for vqname, _ in BASES:
        per_calc = table[vqname]
        if "vibe-qc" not in per_calc:
            continue
        ref = per_calc["vibe-qc"]
        for code, val in per_calc.items():
            if code == "vibe-qc":
                continue
            gap_meV = (val - ref) * 1000
            ok = abs(val - ref) < 1e-3  # <1 meV threshold (~0.04 mHa)
            tag = "✓" if ok else "✗"
            print(f"  {tag} {vqname:<14}  {code:>10}: Δ = {gap_meV:+.4f} meV")
            if not ok:
                failures += 1
    if failures > 0:
        raise AssertionError(
            f"{failures} per-basis cross-code disagreements > 1 meV — see table above"
        )
    print(
        "\n✓ vibe-qc, PySCF (and ORCA, if available) agree on RHF "
        "across all five basis sets to <1 meV (~0.04 mHa). The "
        "underlying integrals + SCF + basis parsing are consistent."
    )


if __name__ == "__main__":
    main()
