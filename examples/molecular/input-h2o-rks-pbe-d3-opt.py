"""H2O — RKS-PBE / 6-31G* + D3(BJ) dispersion + geometry optimization.

The "your first calculation" example from the documentation homepage
(docs/index.md). Demonstrates the full ``run_job`` high-level API in
one call:

  * RKS-DFT with the PBE functional
  * D3(BJ) dispersion correction wired into the SCF and the gradient
  * BFGS geometry optimization (via ASE) before the final SCF
  * .out / .molden / .traj written automatically; .system manifest
    when ``record_hostname=True`` (default).

Run:
    .venv/bin/python examples/molecular/input-h2o-rks-pbe-d3-opt.py

Outputs (next to this script):
    input-h2o-rks-pbe-d3-opt.out      — banner, SCF trace, energies
    input-h2o-rks-pbe-d3-opt.molden   — orbitals at the optimised geometry
    input-h2o-rks-pbe-d3-opt.traj     — ASE trajectory
    input-h2o-rks-pbe-d3-opt.system   — host / build / runtime manifest
"""

from pathlib import Path

from vibeqc import Atom, Molecule, run_job

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem  # → "input-h2o-rks-pbe-d3-opt"

mol = Molecule([
    Atom(8, [0.0,  0.00,  0.00]),
    Atom(1, [0.0,  1.43, -0.98]),
    Atom(1, [0.0, -1.43, -0.98]),
])

run_job(
    mol,
    basis="6-31g*",
    method="rks",
    functional="PBE",
    dispersion="d3bj",
    optimize=True,
    output=HERE / STEM,
)
