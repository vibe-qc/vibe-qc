"""H2O / RKS-PBE / 5 bases — vibe-qc vs ORCA.

Same shape as compare_h2o_hf_basis_scan.py but with RKS-PBE
instead of RHF. PBE is a pure GGA — no HF exchange, exercises the
XC-grid integration. Numerical differences vs ORCA can come from:

  - XC-grid resolution + partition scheme (vibe-qc uses a 75 ×
    17×36 Gauss-Legendre grid; ORCA varies by ``! GRIDx`` keyword)
  - libxc version (vibe-qc uses libxc; ORCA has its own XC library)

Expect agreement to ~1 mHa on small bases, ~5 mHa on triple-zeta
because the grids start to matter. ANY disagreement of order 10 mHa
or larger is suspicious.

PySCF was previously a third column via the in-process
``PySCFCalculator`` ASE shim, which was retired (CLAUDE.md sec 10).
For a PySCF cross-check, use the out-of-process runner at
``examples/regression/core/runner_pyscf.py``.

Run:
    .venv/bin/python examples/ase_compare/cross_code_regression/compare_h2o_dft_basis_scan.py

CSV: output-h2o-dft-basis-scan.csv
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
CSV_PATH = HERE / "output-h2o-dft-basis-scan.csv"

BASES = ["sto-3g", "6-31g*", "cc-pvdz", "cc-pvtz", "def2-tzvp"]


def main() -> None:
    print("=" * 72)
    print(" H2O / RKS-PBE / 5-basis scan  —  vibe-qc vs ORCA")
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
        calculators = [
            (f"vibe-qc/PBE/{basis}",
             VibeQC(basis=basis, functional="PBE", method="rks")),
        ]
        orca = make_orca_calculator(
            orcasimpleinput=f"PBE {basis} EnGrad",
            label=f"orca-h2o-pbe-{basis.replace('*', 's')}",
        )
        if orca is not None:
            calculators.append((f"ORCA/PBE/{basis}", orca))

        results = compare_calculators(atoms, calculators)
        results.print_pretty()
        all_rows.append((basis, results))
        print()

    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["basis", "label", "status", "wall_time_s",
                    "energy_eV", "fmax_eV_per_A"])
        for basis, results in all_rows:
            for row in results.rows:
                w.writerow([basis, row.label, row.status,
                            row.wall_time_s, row.energy, row.fmax])

    print("=" * 72)
    print(" Pivoted summary — energies (eV)")
    print("=" * 72)
    print(f"  {'basis':14s}  {'vibe-qc':>14s}  {'ORCA':>14s}")
    for basis, results in all_rows:
        cells = [basis]
        for code in ("vibe-qc", "ORCA"):
            row = next((r for r in results.rows if r.label.startswith(code)), None)
            cells.append(f"{row.energy:14.6f}" if (row and row.energy is not None) else " " * 14)
        print(f"  {cells[0]:14s}  {cells[1]}  {cells[2]}")
    print()
    print(f"  CSV: {CSV_PATH.relative_to(HERE.parent.parent.parent)}")


if __name__ == "__main__":
    main()
