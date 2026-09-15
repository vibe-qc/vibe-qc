"""Formaldehyde geometry optimisation in gas vs CPCM water.

Demonstrates the v0.9.0 analytic CPCM gradient driving an ASE
BFGSLineSearch optimisation. Reports the change in the C=O bond
length on going from gas → water (expected: ~0.7 pm elongation as
the solvent stabilises the polarised C^δ⁺=O^δ⁻ resonance form).

Matches Section 2 of docs/tutorial/solvation_water.md.

Run:
    .venv/bin/python examples/molecular/solvation/input-ch2o-water-opt.py

Outputs (next to this script):
    input-ch2o-water-opt.gas.traj
    input-ch2o-water-opt.water.traj
"""

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.optimize import BFGSLineSearch

from vibeqc.ase import VibeQC

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem


def build_atoms() -> Atoms:
    """Formaldehyde at near-equilibrium gas-phase geometry (Å)."""
    return Atoms(
        numbers=[6, 8, 1, 1],
        positions=[
            [0.0,  0.000,  0.0000],
            [0.0,  0.000,  1.2055],
            [0.0,  0.940, -0.5870],
            [0.0, -0.940, -0.5870],
        ],
    )


def r_co(atoms: Atoms) -> float:
    return float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))


def optimise(label: str, calc: VibeQC) -> Atoms:
    atoms = build_atoms()
    atoms.calc = calc
    traj_path = HERE / f"{STEM}.{label}.traj"
    with Trajectory(str(traj_path), "w", atoms) as traj:
        opt = BFGSLineSearch(atoms, logfile=None)
        opt.attach(traj)
        opt.run(fmax=0.01)
    return atoms


# Gas-phase reference.
calc_gas = VibeQC(basis="def2-svp", functional="pbe0")
atoms_gas = optimise("gas", calc_gas)

# CPCM-water optimisation. Forces routed through
# vibeqc.solvation.cpcm_gradient (analytic semi-numerical hybrid).
calc_water = VibeQC(basis="def2-svp", functional="pbe0", solvent="water")
atoms_aq = optimise("water", calc_water)

print(f"\n=== Formaldehyde geometry: gas vs water ===")
print(f"r(C=O) gas    : {r_co(atoms_gas):.4f} Å")
print(f"r(C=O) water  : {r_co(atoms_aq):.4f} Å")
print(f"Δr(C=O)       : {r_co(atoms_aq) - r_co(atoms_gas):+.4f} Å")
print(f"(experiment   : ~+0.002 Å gas → aqueous IR/Raman)")
