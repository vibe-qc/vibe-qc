"""Closed-shell RHF single-point energy on H2O / 6-31G*.

Run:
    .venv/bin/python input-h2o-rhf.py

Produces:
    output-h2o-rhf.out     — banner, SCF trace, energies, MO table
    output-h2o-rhf.molden  — molecular orbitals for Jmol / Avogadro / Molden
"""

from pathlib import Path

from vibeqc import Molecule, run_job

HERE = Path(__file__).parent

mol = Molecule.from_xyz(HERE.parent / "h2o.xyz")

run_job(
    mol,
    basis="6-31g*",
    method="rhf",
    output=HERE / "output-h2o-rhf",
)
