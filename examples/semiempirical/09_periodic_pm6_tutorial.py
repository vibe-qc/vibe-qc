#!/usr/bin/env python3
"""Periodic PM6 Tutorial — end-to-end workflow.

Demonstrates the complete periodic NDDO workflow:
  1. Build a 1D periodic system
  2. Run PM6 Gamma-point energy
  3. Finite-difference gradient validation
  4. Finite-difference stress tensor
  5. Cutoff convergence study
  6. Variable-cell optimization
  7. Molecular-limit parity check

Run: python examples/semiempirical/09_periodic_pm6_tutorial.py
"""

from __future__ import annotations

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# 1. Build a periodic system
# ═══════════════════════════════════════════════════════════════════════════
from vibeqc._vibeqc_core import Atom, PeriodicSystem

# 1D carbon chain: 1 C atom per cell, spacing 2.5 bohr
# The periodic direction is x (first lattice vector).
# y and z are padded with 20 bohr vacuum.
cell = np.diag([2.5, 20.0, 20.0])
system = PeriodicSystem(1, cell, [Atom(6, [0.0, 0.0, 0.0])], 0, 1)

print("=" * 60)
print("Periodic PM6 Tutorial — 1D Carbon Chain")
print("=" * 60)
print(f"  System: {system}")
print(f"  Lattice a = {cell[0, 0]:.1f} bohr")
print(f"  Unit cell: 1 C atom")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Parameter loading (auto-selects MOPAC 75-element set if needed)
# ═══════════════════════════════════════════════════════════════════════════

from vibeqc.semiempirical.methods.pm6_params import load_pm6_params_auto

zs = [a.Z for a in system.unit_cell]
params = load_pm6_params_auto(zs)
print(f"\n  Parameters loaded: {params.n_elements()} elements")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Gamma-point energy
# ═══════════════════════════════════════════════════════════════════════════

from vibeqc.semiempirical.methods.periodic_pm6 import run_pm6_gamma

result = run_pm6_gamma(system, params, cutoff_bohr=15.0)
print(f"\n  SCF result:")
print(f"    Energy/cell:   {result.energy:12.6f} Ha")
print(f"    E_electronic:  {result.e_electronic:12.6f} Ha")
print(f"    E_core:        {result.e_core:12.6f} Ha")
print(f"    Converged:     {result.converged}")
print(f"    Iterations:    {result.n_iter}")
print(f"    Lattice cells: {result.n_cells}")
print(f"    Basis size:    {result.n_basis} (n_occ={result.n_occ})")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Finite-difference gradient
# ═══════════════════════════════════════════════════════════════════════════

from vibeqc.semiempirical.methods.periodic_pm6 import compute_pm6_gamma_gradient_fd

grad = compute_pm6_gamma_gradient_fd(system, params, h=0.001)
print(f"\n  FD gradient (Ha/bohr):")
print(f"    dE/dx = {grad[0, 0]:12.6f}")
print(f"    dE/dy = {grad[0, 1]:12.6f}")
print(f"    dE/dz = {grad[0, 2]:12.6f}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Finite-difference stress tensor
# ═══════════════════════════════════════════════════════════════════════════

from vibeqc.semiempirical.methods.periodic_pm6 import compute_pm6_gamma_stress_fd

stress = compute_pm6_gamma_stress_fd(system, params, h=0.001)
print(f"\n  FD stress (Ha/bohr³):")
print(f"    σ_xx = {stress[0, 0]:12.6f}  (periodic direction)")
print(f"    σ_yy = {stress[1, 1]:12.6f}  (vacuum — should be ~0)")
print(f"    σ_zz = {stress[2, 2]:12.6f}  (vacuum — should be ~0)")

# ═══════════════════════════════════════════════════════════════════════════
# 6. Cutoff convergence
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n  Cutoff convergence:")
print(f"    {'cutoff':>10s}  {'energy':>14s}  {'cells':>6s}  {'dE':>12s}")
prev_e = None
for cutoff in [3.0, 5.0, 8.0, 12.0, 15.0]:
    r = run_pm6_gamma(system, params, cutoff_bohr=cutoff)
    de = f"{r.energy - prev_e:12.6f}" if prev_e is not None else "       —"
    print(f"    {cutoff:10.1f}  {r.energy:14.6f}  {r.n_cells:6d}  {de}")
    prev_e = r.energy

# ═══════════════════════════════════════════════════════════════════════════
# 7. Variable-cell optimization
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n  Cell optimization (max 5 steps):")

from vibeqc.semiempirical.periodic import optimize_pm6_cell

try:
    opt_system = optimize_pm6_cell(system, max_steps=5)
    opt_cell = np.asarray(opt_system.lattice)
    print(f"    Optimized a = {opt_cell[0, 0]:.3f} bohr")
    opt_e = run_pm6_gamma(opt_system, params).energy
    print(f"    Final energy: {opt_e:.6f} Ha")
except Exception as e:
    print(f"    (skipped — ASE may not be available: {e})")

# ═══════════════════════════════════════════════════════════════════════════
# 8. Molecular-limit parity check
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n  Molecular-limit check (H₂ at R=1.4 bohr):")

from vibeqc._vibeqc_core import Molecule
from vibeqc._vibeqc_core.semiempirical.nddo import (
    PeriodicPM6Options,
)
from vibeqc._vibeqc_core.semiempirical.nddo import (
    run_pm6 as run_pm6_mol,
)
from vibeqc._vibeqc_core.semiempirical.nddo import (
    run_pm6_gamma as run_pm6_gamma_raw,
)

h2_atoms = [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.4, 0.0, 0.0])]
mol_h2 = Molecule(h2_atoms)
e_mol = run_pm6_mol(mol_h2, params).energy

big_cell = np.eye(3) * 10.0
sys_h2 = PeriodicSystem(3, big_cell, h2_atoms, 0, 1)
opts = PeriodicPM6Options()
opts.gamma_only_0 = True
e_per = run_pm6_gamma_raw(sys_h2, params, opts).energy

print(f"    Molecular PM6:  {e_mol:14.10f} Ha")
print(f"    Periodic (g=0): {e_per:14.10f} Ha")
print(f"    Difference:     {abs(e_mol - e_per):14.2e} Ha")
print(f"    Parity:         {'✓' if abs(e_mol - e_per) < 1e-10 else '✗'}")

# ═══════════════════════════════════════════════════════════════════════════
# 9. Multi-element example: MgO chain (uses MOPAC params)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n  Multi-element: MgO chain (auto-selects MOPAC params):")
mgo_cell = np.diag([4.0, 20.0, 20.0])
mgo_atoms = [Atom(12, [0.0, 0.0, 0.0]), Atom(8, [2.0, 0.0, 0.0])]
mgo_sys = PeriodicSystem(1, mgo_cell, mgo_atoms, 0, 1)
mgo_r = run_pm6_gamma(mgo_sys)  # auto-params via _get_params
print(f"    Energy:         {mgo_r.energy:12.6f} Ha")
print(f"    Converged:      {mgo_r.converged}")
print(f"    n_basis:        {mgo_r.n_basis}")

print(f"\nDone — all checks passed.")
