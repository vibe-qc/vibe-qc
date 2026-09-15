"""H2O / RHF / 5 bases: vibe-qc vs ORCA, head-to-head.

Designed for the dev chat's bug-fixing handover. Same molecule,
same geometry, same bases across both codes; tabulates total
energy + |F|max + per-code wall time so any divergence is
immediately visible.

Bases (small to big):
    sto-3g          minimal, smoke test
    6-31G*          small Pople, popular workhorse
    cc-pVDZ         Dunning DZ, well-defined contraction
    cc-pVTZ         Dunning TZ, fewer integrals shared with DZ
    def2-TZVP       Ahlrichs TZ, different family

Each row shows:
    label                  status     wall_time_s    energy        |F|max

Expected outcome: both columns agree to ~uHa on each row.
ANY disagreement larger than ~10 uHa is a real issue worth
investigating. The CSV captures the full table for further
analysis / regression-test fixture generation.

PySCF was previously a third column via the in-process
``PySCFCalculator`` ASE shim, which was retired (CLAUDE.md sec 10).
For a PySCF cross-check, use the out-of-process runner at
``examples/regression/core/runner_pyscf.py``.

Run:
    .venv/bin/python examples/ase_compare/cross_code_regression/compare_h2o_hf_basis_scan.py

CSV: output-h2o-hf-basis-scan.csv  (next to script)
"""

from __future__ import annotations

import csv
from pathlib import Path

from ase.build import molecule

from vibeqc.ase import VibeQC
from vibeqc.benchmark import (
    compare_calculators,
    make_orca_calculator,
    print_calculator_availability,
)

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "output-h2o-hf-basis-scan.csv"

BASES = ["sto-3g", "6-31g*", "cc-pvdz", "cc-pvtz", "def2-tzvp"]


def main() -> None:
    print("=" * 72)
    print(" H2O / RHF / 5-basis scan  —  vibe-qc vs ORCA")
    print("=" * 72)
    print()
    print_calculator_availability()
    print()

    atoms = molecule("H2O")

    all_rows = []
    for basis in BASES:
        print(f"--- basis = {basis} ---")
        # PySCF row dropped: the in-process PySCFCalculator shim was
        # retired (CLAUDE.md sec 10). For a PySCF cross-check, use the
        # out-of-process runner at
        # examples/regression/core/runner_pyscf.py.
        calculators: list = [
            (f"vibe-qc/RHF/{basis}",
             VibeQC(basis=basis)),
        ]
        orca = make_orca_calculator(
            orcasimpleinput=f"HF {basis} EnGrad",
            label=f"orca-h2o-hf-{basis.replace('*', 's')}",
        )
        if orca is not None:
            calculators.append((f"ORCA/RHF/{basis}", orca))

        results = compare_calculators(atoms, calculators)
        results.print_pretty()
        all_rows.append((basis, results))
        print()

    # Single combined CSV
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["basis", "label", "status", "wall_time_s",
                    "energy_eV", "fmax_eV_per_A"])
        for basis, results in all_rows:
            for row in results.rows:
                w.writerow([basis, row.label, row.status,
                            row.wall_time_s, row.energy, row.fmax])

    # Pivoted summary
    print("=" * 72)
    print(" Pivoted summary — energies (eV) by basis × code")
    print("=" * 72)
    headers = ["basis", "vibe-qc", "ORCA"]
    print(f"  {headers[0]:14s}  {headers[1]:>14s}  {headers[2]:>14s}")
    for basis, results in all_rows:
        cells = [basis]
        for code in ("vibe-qc", "ORCA"):
            row = next(
                (r for r in results.rows if r.label.startswith(code)),
                None,
            )
            cells.append(
                f"{row.energy:14.6f}" if (row and row.energy is not None)
                else " " * 14
            )
        print(f"  {cells[0]:14s}  {cells[1]}  {cells[2]}")
    print()
    print(f"  CSV: {CSV_PATH.relative_to(HERE.parent.parent.parent)}")


if __name__ == "__main__":
    main()
