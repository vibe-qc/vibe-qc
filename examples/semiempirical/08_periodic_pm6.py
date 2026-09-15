#!/usr/bin/env python3
"""Example 8: Periodic PM6 for 1D chains.

Demonstrates:
  1. Building a PeriodicSystem for 1D chains
  2. Gamma-point PM6 energy
  3. Finite-difference gradient and stress
  4. Cutoff convergence check

Run: python examples/semiempirical/08_periodic_pm6.py
"""

from __future__ import annotations

import numpy as np
from vibeqc._vibeqc_core import Atom, PeriodicSystem
from vibeqc.semiempirical.methods.pm6_params import load_pm6_params

# ── Load PM6 parameters (Stewart 2007, H,C,N,O,F) ──
params = load_pm6_params()

print("=" * 60)
print("Periodic PM6 — 1D Chain Examples")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# Example 1: He chain
# ─────────────────────────────────────────────────────────────
print("\n─── 1D He Chain ───")
system_he = PeriodicSystem(
    1,
    np.diag([3.0, 20.0, 20.0]),
    [Atom(2, [0.0, 0.0, 0.0])],
    0,
    1,
)

from vibeqc.semiempirical.methods.periodic_pm6 import (
    PeriodicPM6Model,
    compute_pm6_gamma_gradient_fd,
    compute_pm6_gamma_stress_fd,
    run_pm6_gamma,
)

# Energy
model_he = PeriodicPM6Model(system_he, params, cutoff_bohr=15.0)
e_he = model_he.energy()
print(f"  Energy/cell:     {e_he:12.6f} Ha")
print(f"  n_cells:         {run_pm6_gamma(system_he, params).n_cells}")

# Gradient — should be near zero for symmetric atom at origin
grad_he = model_he.gradient()
print(
    f"  Gradient:        [{grad_he[0, 0]:.6f}, {grad_he[0, 1]:.6f}, {grad_he[0, 2]:.6f}] Ha/bohr"
)

# Stress
stress_he = model_he.stress()
print(f"  Stress σ_xx:     {stress_he[0, 0]:.6f} Ha/bohr³")
print(f"  Stress σ_yy:     {stress_he[1, 1]:.6f} Ha/bohr³ (should be ~0)")
print(f"  Stress σ_zz:     {stress_he[2, 2]:.6f} Ha/bohr³ (should be ~0)")

# ─────────────────────────────────────────────────────────────
# Example 2: C chain
# ─────────────────────────────────────────────────────────────
print("\n─── 1D Carbon Chain ───")
system_c = PeriodicSystem(
    1,
    np.diag([2.5, 20.0, 20.0]),
    [Atom(6, [0.0, 0.0, 0.0])],
    0,
    1,
)

model_c = PeriodicPM6Model(system_c, params, cutoff_bohr=15.0)
e_c = model_c.energy()
print(f"  Energy/cell:     {e_c:12.6f} Ha")

# ─────────────────────────────────────────────────────────────
# Example 3: H₂O chain
# ─────────────────────────────────────────────────────────────
print("\n─── 1D H₂O Chain ───")
system_h2o = PeriodicSystem(
    1,
    np.diag([5.0, 20.0, 20.0]),
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [1.8, 0.0, 0.0]),
        Atom(1, [-0.5, 1.5, 0.0]),
    ],
    0,
    1,
)

model_h2o = PeriodicPM6Model(system_h2o, params, cutoff_bohr=15.0, max_iter=200)
e_h2o = model_h2o.energy()
grad_h2o = model_h2o.gradient()
print(f"  Energy/cell:     {e_h2o:12.6f} Ha")
print(f"  |grad|_max:      {np.max(np.abs(grad_h2o)):.6f} Ha/bohr")

# ─────────────────────────────────────────────────────────────
# Example 4: Cutoff convergence (He chain)
# ─────────────────────────────────────────────────────────────
print("\n─── Cutoff Convergence (He chain) ───")
print(f"  {'cutoff/bohr':>14s}  {'energy/Ha':>14s}  {'n_cells':>8s}")
for cutoff in [3.0, 5.0, 10.0, 15.0]:
    r = run_pm6_gamma(system_he, params, cutoff_bohr=cutoff)
    print(f"  {cutoff:14.1f}  {r.energy:14.6f}  {r.n_cells:8d}")

# ─────────────────────────────────────────────────────────────
# Example 5: Molecular-limit check
# ─────────────────────────────────────────────────────────────
print("\n─── Molecular-Limit Check (H₂) ───")
from vibeqc._vibeqc_core import Molecule
from vibeqc._vibeqc_core.semiempirical.nddo import run_pm6 as run_pm6_mol

atoms_h2 = [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.4, 0.0, 0.0])]
mol_h2 = Molecule(atoms_h2)
e_mol = run_pm6_mol(mol_h2, params).energy

cell_big = np.eye(3) * 10.0
sys_h2 = PeriodicSystem(3, cell_big, atoms_h2, 0, 1)
r_g0 = run_pm6_gamma(sys_h2, params, cutoff_bohr=15.0)
# Force gamma_only_0 for molecular limit
from vibeqc._vibeqc_core.semiempirical import nddo

opts_g0 = nddo.PeriodicPM6Options()
opts_g0.gamma_only_0 = True
r_g0_exact = nddo.run_pm6_gamma(sys_h2, params, opts_g0)

print(f"  Molecular PM6:   {e_mol:14.10f} Ha")
print(f"  Periodic (g=0):  {r_g0_exact.energy:14.10f} Ha")
print(f"  Match:           {abs(e_mol - r_g0_exact.energy) < 1e-10}")

print("\nDone.")
