"""Geometry optimization of H2O using BFGS (via ASE) at HF/6-31G*.

Success criterion: the optimiser stops below fmax = 0.01 eV/A after four
steps (five frames in the .traj) and the final SCF gives E = -76.009341 Ha
(vibe-qc 0.17.10 development build; agree to 1e-5 Ha), 5.7 mHa below the
starting geometry, with R(OH) = 1.791 bohr (0.948 A) in the optimised
structure.

Run:
    .venv/bin/python input-h2o-opt.py

Produces:
    output-h2o-opt.out     — initial geometry, optimizer summary,
                              optimized geometry, final SCF trace
    output-h2o-opt.molden  — MOs at the optimized geometry
    output-h2o-opt.traj    — per-step trajectory (ASE binary format)

The trajectory can be viewed as an animation by any ASE-aware tool, e.g.

    ase gui output-h2o-opt.traj

or converted to XYZ-frame format for Molden / VMD with

    ase convert output-h2o-opt.traj output-h2o-opt.xyz
"""

from pathlib import Path

from vibeqc import Molecule, run_job

HERE = Path(__file__).parent

# Deliberately start from a slightly distorted geometry so there's
# something for the optimizer to do.
mol = Molecule.from_xyz(HERE.parent / "h2o.xyz")

run_job(
    mol,
    basis="6-31g*",
    method="rhf",
    output=HERE / "output-h2o-opt",
    optimize=True,
    fmax=0.01,           # eV / A — tight enough to converge to ~10⁻⁴ bohr
    max_opt_steps=50,
)
