"""Kohn-Sham DFT single-point on H2O with the PBE functional.

Run:
    .venv/bin/python input-h2o-dft.py

Produces:
    output-h2o-dft.out     — SCF trace + energy components (J, K, XC) +
                              OpenMP thread count + wall-clock timings
    output-h2o-dft.molden  — Kohn-Sham orbitals for visualization

Switch the functional by editing ``functional=`` — anything libxc
recognises works (e.g. "LDA", "BLYP", "B3LYP"). Adjust parallelism
with ``num_threads=`` (or export ``OMP_NUM_THREADS`` before running).
"""

from pathlib import Path

from vibeqc import Molecule, run_job

HERE = Path(__file__).parent

mol = Molecule.from_xyz(HERE.parent / "h2o.xyz")

run_job(
    mol,
    basis="6-31g*",
    method="rks",
    functional="PBE",
    output=HERE / "output-h2o-dft",
    num_threads=4,     # pin OpenMP thread count; omit to use the default
                       # (OMP_NUM_THREADS or the hardware core count)
)
