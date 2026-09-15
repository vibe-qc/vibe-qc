"""Cross-validate H2O / RHF / 6-31G* against (optionally) ORCA.

The "Hello, validation" of vibe-qc cross-checking. Same molecule, same
basis, two ASE-driven calculators, same number to ~uHa.

Run:
    .venv/bin/python examples/ase_compare/compare-h2o-hf.py

ORCA is optional. Provide its path via the ``ORCA_COMMAND`` (or
``ASE_ORCA_COMMAND``) environment variable, or have ``orca`` on
``$PATH``. If unavailable, the ORCA row is reported as "unavailable"
and the comparison continues with just the vibe-qc row.

For a PySCF cross-check, use the out-of-process subprocess runner at
``examples/regression/core/runner_pyscf.py``. The previous in-process
``PySCFCalculator`` ASE shim was retired (CLAUDE.md sec 10: no
in-process imports of external QC programs under ``python/vibeqc/``).

Produces:
    output-compare-h2o-hf.csv   (full comparison table)
    (stdout)                    (pretty-printed table)

Expected outcome: both rows agree on E and |F|max to ~uHa precision
(the basis is identical and the SCF threshold is tight enough). If
they disagree, that's a real bug worth investigating, either in
vibe-qc or in the calling convention.
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
    print(" Cross-validation:  H2O / RHF / 6-31G*")
    print("=" * 72)
    print()
    print("Calculator availability:")
    print_calculator_availability()
    print()

    atoms = molecule("H2O")  # ASE's stock H2O geometry (Angstrom)

    calculators: list = [
        ("vibe-qc/RHF/6-31G*", VibeQC(basis="6-31g*")),
        # PySCF row dropped: the in-process PySCFCalculator shim was
        # retired. For a PySCF cross-check, use the out-of-process
        # runner at examples/regression/core/runner_pyscf.py.
    ]

    orca = make_orca_calculator(
        orcasimpleinput="HF 6-31G* EnGrad",
        label="orca-h2o-hf",
    )
    if orca is not None:
        calculators.append(("ORCA/RHF/6-31G*", orca))
    else:
        print("(ORCA not found: set ORCA_COMMAND or add orca to $PATH "
              "to include it in the comparison)")
        print()

    results = compare_calculators(
        atoms,
        calculators,
        properties=("energy", "forces"),
    )
    results.print_table()
    csv_path = HERE / "output-compare-h2o-hf.csv"
    results.to_csv(csv_path)
    print(f"\nFull table written to {csv_path.relative_to(HERE.parent.parent)}")

    # Tight tolerance: these are the same SCF on the same basis. Any
    # disagreement past 1e-5 eV (~0.4 uHa) is a real problem.
    results.assert_agreement(
        reference="vibe-qc/RHF/6-31G*",
        tol={"energy": 1e-5, "forces": 1e-4},
    )
    print("OK: all calculators agree to within 1e-5 eV / 1e-4 eV/Angstrom "
          "(identical SCF on identical basis, as expected).")


if __name__ == "__main__":
    main()
