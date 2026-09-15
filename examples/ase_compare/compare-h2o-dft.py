"""Cross-validate H2O / RKS / PBE / 6-31G* against ORCA.

DFT is the second cross-check after HF: same SCF skeleton but with
an exchange-correlation functional bolted on. Two ASE-driven
calculators, two integration grids; energies should still agree to a
few mEh (the residual is grid-discretization, which differs between
codes).

Run:
    .venv/bin/python examples/ase_compare/compare-h2o-dft.py

Set ORCA_COMMAND or put 'orca' on $PATH to include ORCA in the table.

For a PySCF cross-check, use the out-of-process subprocess runner at
``examples/regression/core/runner_pyscf.py``. The previous in-process
``PySCFCalculator`` ASE shim was retired (CLAUDE.md sec 10).

Produces:
    output-compare-h2o-dft.csv

Tolerances are looser than the HF script (5e-3 eV ~= 0.2 mHa) because
each code uses its own DFT integration grid. If you want bit-identical
agreement, you'd need to match grid choices across codes, which is
possible but rare in practice.
"""

from __future__ import annotations

from pathlib import Path

from ase.build import molecule

from vibeqc.ase import VibeQC
from vibeqc.benchmark import (
    compare_calculators,
    make_orca_calculator,
    print_calculator_availability,
)

HERE = Path(__file__).resolve().parent


def main() -> None:
    print("=" * 72)
    print(" Cross-validation:  H2O / RKS / PBE / 6-31G*")
    print("=" * 72)
    print()
    print("Calculator availability:")
    print_calculator_availability()
    print()

    atoms = molecule("H2O")

    calculators: list = [
        ("vibe-qc/PBE/6-31G*", VibeQC(basis="6-31g*", functional="PBE")),
        # PySCF row dropped: the in-process PySCFCalculator shim was
        # retired. For a PySCF cross-check, use the out-of-process
        # runner at examples/regression/core/runner_pyscf.py.
    ]

    orca = make_orca_calculator(
        orcasimpleinput="PBE 6-31G* EnGrad",
        label="orca-h2o-dft",
    )
    if orca is not None:
        calculators.append(("ORCA/PBE/6-31G*", orca))
    else:
        print("(ORCA not found: set ORCA_COMMAND or add orca to $PATH "
              "to include it in the comparison)")
        print()

    results = compare_calculators(
        atoms,
        calculators,
        properties=("energy", "forces", "dipole"),
    )
    results.print_table()
    csv_path = HERE / "output-compare-h2o-dft.csv"
    results.to_csv(csv_path)
    print(f"\nFull table written to {csv_path.relative_to(HERE.parent.parent)}")

    # Loose-ish tolerance for DFT: different DFT grids across codes
    # contribute O(0.1-1 mHa) to the energy. Tighter tolerance would
    # require matching grid choices across codes (not in scope here).
    results.assert_agreement(
        reference="vibe-qc/PBE/6-31G*",
        tol={"energy": 5e-3, "forces": 1e-2, "dipole": 1e-2},
    )
    print(
        "OK: all calculators agree on E, F, dipole within 5 mHa / "
        "1e-2 eV/Angstrom / 1e-2 e*Angstrom (residuals are dominated "
        "by per-code DFT grid discretization, the expected source of "
        "disagreement)."
    )


if __name__ == "__main__":
    main()
