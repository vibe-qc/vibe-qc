#!/usr/bin/env python3
"""GFN2-xTB geometry optimization.

Mirrors the xtb workshop § "GeoOpt". Demonstrates:
  1. run_job with optimize=True
  2. ASE-based optimization with SemiempiricalCalculator
  3. Gradient validation (analytic vs finite-difference)

Run:
    .venv/bin/python examples/semiempirical/32_gfn2_geometry_optimization.py
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from vibeqc import Molecule, Atom, run_job
from vibeqc.semiempirical.methods.gfn2 import GFN2Model
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

HERE = Path(__file__).resolve().parent
params = load_gfn2_params()

# ── Build a distorted water ─────────────────────────────────────────────
print("=" * 72)
print("1. run_job geometry optimization")
theta = np.deg2rad(104.5 / 2)
r_oh = 2.20  # stretched from equilibrium ~1.81 bohr
mol = Molecule([
    Atom(8, [0.0, 0.0, 0.0]),
    Atom(1, [r_oh * np.sin(theta), r_oh * np.cos(theta), 0.0]),
    Atom(1, [-r_oh * np.sin(theta), r_oh * np.cos(theta), 0.0]),
])

model_start = GFN2Model(mol, params=params, warn=False)
print(f"   Starting energy: {model_start.energy():.8f} Ha")
print(f"   Starting O-H distance: {r_oh:.3f} bohr (equilibrium ~1.81)")

run_job(
    mol, method="gfn2_xtb",
    output=str(HERE / "output-h2o-gfn2-opt"),
    optimize=True, fmax=0.02,
)

print(f"   -> see {HERE / 'output-h2o-gfn2-opt.out'}")

# ── ASE optimization ────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("2. ASE geometry optimization")
try:
    from ase import Atoms
    from ase.optimize import BFGS
    from ase.units import Bohr
    from vibeqc.semiempirical.ase import SemiempiricalCalculator

    # Build ASE Atoms (positions in Å)
    atoms = Atoms('OH2', positions=np.array([
        [0.0, 0.0, 0.0],
        [r_oh * np.sin(theta), r_oh * np.cos(theta), 0.0],
        [-r_oh * np.sin(theta), r_oh * np.cos(theta), 0.0],
    ]) / Bohr * Bohr)  # already in bohr, convert to Å for ASE
    # Actually, our Molecule uses bohr. Convert to Å for ASE.
    atoms_ase = Atoms('OH2', positions=np.array([
        [0.0, 0.0, 0.0],
        [r_oh * np.sin(theta) * Bohr, r_oh * np.cos(theta) * Bohr, 0.0],
        [-r_oh * np.sin(theta) * Bohr, r_oh * np.cos(theta) * Bohr, 0.0],
    ]))

    # Build GFN2Model and wrap as ASE Calculator
    calc = SemiempiricalCalculator(model_start)
    atoms_ase.calc = calc

    print(f"   Starting ASE energy: {atoms_ase.get_potential_energy():.6f} eV")
    print(f"   Starting forces |F|_max: {np.abs(atoms_ase.get_forces()).max():.4f} eV/Å")

    opt = BFGS(atoms_ase)
    opt.run(fmax=0.02)

    final_pos_ang = atoms_ase.get_positions()
    print(f"\n   Optimised positions (Å):")
    for i, pos in enumerate(final_pos_ang):
        print(f"      {atoms_ase.get_chemical_symbols()[i]}: {pos}")
    r_oh_opt = np.linalg.norm(final_pos_ang[0] - final_pos_ang[1])
    print(f"   Optimised O-H: {r_oh_opt:.3f} Å = {r_oh_opt / Bohr:.3f} bohr")
    print(f"   Optimised energy: {atoms_ase.get_potential_energy():.6f} eV")

except ImportError:
    print("   ASE not installed — skipping")

# ── Gradient validation ─────────────────────────────────────────────────
print("\n" + "=" * 72)
print("3. Analytic vs finite-difference gradient")

model_val = GFN2Model(mol, params=params, warn=False)
grad_analytic = np.asarray(model_val.gradient())

# Finite-difference gradient
h = 0.001
grad_fd = np.zeros_like(grad_analytic)
for a in range(len(mol.atoms)):
    for c in range(3):
        xyz_p = list(mol.atoms[a].xyz)
        xyz_p[c] += h
        mol_p = Molecule(
            [Atom(at.Z, xyz_p if i == a else list(at.xyz))
             for i, at in enumerate(mol.atoms)],
            mol.charge, mol.multiplicity,
        )
        ep = GFN2Model(mol_p, params=params, warn=False).energy()
        xyz_m = list(mol.atoms[a].xyz)
        xyz_m[c] -= h
        mol_m = Molecule(
            [Atom(at.Z, xyz_m if i == a else list(at.xyz))
             for i, at in enumerate(mol.atoms)],
            mol.charge, mol.multiplicity,
        )
        em = GFN2Model(mol_m, params=params, warn=False).energy()
        grad_fd[a, c] = (ep - em) / (2.0 * h)

diff = grad_analytic - grad_fd
print(f"   |grad_analytic|_max:  {np.abs(grad_analytic).max():.6e} Ha/bohr")
print(f"   |grad_fd|_max:        {np.abs(grad_fd).max():.6e} Ha/bohr")
print(f"   |diff|_max:           {np.abs(diff).max():.6e} Ha/bohr")
print(f"   |diff|_rms:           {np.sqrt(np.mean(diff**2)):.6e} Ha/bohr")
print(f"   Analytic and FD agree to ~{np.abs(diff).max() / max(np.abs(grad_analytic).max(), 1e-12):.1e}")

SYMBOLS = {8: "O", 1: "H"}
for a in range(len(mol.atoms)):
    for c, xyz_label in enumerate(["x", "y", "z"]):
        print(f"   {SYMBOLS[mol.atoms[a].Z]}{a} {xyz_label}:  "
              f"ana={grad_analytic[a,c]:10.6f}  fd={grad_fd[a,c]:10.6f}  "
              f"diff={diff[a,c]:10.3e}")
