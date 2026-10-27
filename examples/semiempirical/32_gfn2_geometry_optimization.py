#!/usr/bin/env python3
"""GFN2-xTB geometry optimization.

Mirrors the xtb workshop § "GeoOpt". Demonstrates:
  1. run_job(method="gfn2_xtb", optimize=True): the high-level route, which
     drives the optimisation through ASE and writes the usual output family
  2. The native geomopt framework: ``run_geomopt`` with
     ``SemiempiricalProvider("gfn2-xtb")``, no ASE in the loop
  3. Gradient validation (analytic vs finite-difference)

Success criterion: both optimisers shorten the stretched O-H bond from
2.20 bohr to 1.83 bohr and agree on the minimum energy, E = -5.070026 Ha,
to 1e-6 Ha (vibe-qc 0.17.10 development build); the analytic gradient
agrees with the central finite difference to better than 1e-6 Ha/bohr
(observed 3e-8).

GFN2-xTB is an experimental route in vibe-qc (``GFN2ExperimentalWarning``).

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

# ── Native geometry optimiser through the semiempirical provider ─────
print("\n" + "=" * 72)
print("2. Native geomopt framework with SemiempiricalProvider")
from vibeqc.geomopt import run_geomopt
from vibeqc.geomopt.semiempirical_provider import SemiempiricalProvider

# SemiempiricalProvider wraps the unified semiempirical dispatcher as an
# EnergyGradientProvider, so every optimiser in vibeqc.geomopt (sd, cg,
# bfgs, lbfgs, rfo, ...) can relax a GFN2-xTB geometry without ASE.
provider = SemiempiricalProvider("gfn2-xtb")
opt = run_geomopt(mol, provider, geom_opt="bfgs", geom_max_iter=100)

final = opt.system
print(f"   Converged: {opt.converged} after {opt.n_iter} BFGS steps "
      f"({opt.n_energy_evals} energy+gradient evaluations)")
print("   Optimised geometry (bohr):")
for at in final.atoms:
    x, y, z = at.xyz
    print(f"      Z={at.Z:2d}: {x:10.5f} {y:10.5f} {z:10.5f}")
r_oh_opt = float(np.linalg.norm(
    np.asarray(final.atoms[0].xyz) - np.asarray(final.atoms[1].xyz)))
print(f"   Optimised O-H: {r_oh_opt:.3f} bohr (started at {r_oh:.3f})")
print(f"   Optimised energy: {opt.energy:.8f} Ha")

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
