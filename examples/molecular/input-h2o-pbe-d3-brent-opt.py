"""H2O — RKS-PBE / 6-31G* + D3(BJ) + Brent geometry optimization + QVF.

Full-featured example: DFT with the PBE functional, D3(BJ) dispersion
correction wired into SCF and gradient, Brent steepest-descent optimizer
with QVF archive output.  Open the .qvf with vibe-view to see the
trajectory animation.

Compare with examples/molecular/input-h2o-rks-pbe-d3-opt.py (same but ASE
BFGS backend).

Run:
    .venv/bin/python examples/molecular/input-h2o-pbe-d3-brent-opt.py

Produces:
    output-h2o-pbe-d3-brent-opt.out     — banner, SCF trace, energies
    output-h2o-pbe-d3-brent-opt.molden  — MOs at the optimised geometry
    output-h2o-pbe-d3-brent-opt.qvf     — QVF archive (trajectory + density)
    output-h2o-pbe-d3-brent-opt.system  — host / build / runtime manifest
"""

from pathlib import Path

from vibeqc import Atom, Molecule, run_job

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem.replace("input-", "output-")

mol = Molecule(
    [
        Atom(8, [0.0, 0.00, 0.00]),
        Atom(1, [0.0, 1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ]
)

run_job(
    mol,
    basis="6-31g*",
    method="rks",
    functional="PBE",
    dispersion="d3bj",  # D3(BJ) dispersion on energy + gradient
    optimize=True,
    optimizer_backend="brent",  # steepest-descent + Brent line search
    output=HERE / STEM,
    output_qvf=True,
    fmax=0.05,
    max_opt_steps=50,
)
