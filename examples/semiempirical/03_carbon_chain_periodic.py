#!/usr/bin/env python3
"""Example 3: Periodic DFTB0 with k-points for a 1D carbon chain.

Demonstrates:
  1. Building a PeriodicSystem
  2. Gamma-point DFTB0
  3. Monkhorst-Pack k-point mesh
  4. Band structure along a k-path

Run: python examples/semiempirical/03_carbon_chain_periodic.py
"""

from __future__ import annotations

import numpy as np
from vibeqc._vibeqc_core import Atom, PeriodicSystem, monkhorst_pack
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc.semiempirical.parameters import default_parameters

params = default_parameters()

# 1D carbon chain: C-C spacing 2.5 bohr
system = PeriodicSystem()
system.dim = 1
system.lattice = np.diag([2.5, 30.0, 30.0])
system.unit_cell = [Atom(6, [0.0, 0.0, 0.0])]

print("=== 1D Carbon Chain ===")
print(f"Lattice: a = {system.lattice[0, 0]:.1f} bohr")
print(f"Unit cell: {len(system.unit_cell)} atom(s)")

# Gamma-point energy
r_gamma = _se.run_dftb0_gamma(system, params)
print(f"\nGamma-point DFTB0: {r_gamma.energy:.6f} Ha/cell")

# 4×1×1 Monkhorst-Pack mesh
kmesh = monkhorst_pack(system, (4, 1, 1))
r_k = _se.run_dftb0_kpoints(system, params, kmesh)
print(f"4×1×1 k-points:   {r_k.energy:.6f} Ha/cell")

# Band structure: Γ → X
kpath = [np.zeros(3), np.array([np.pi / 2.5, 0.0, 0.0])]
r_bs = _se.run_dftb0_bandpath(system, params, kpath)
print(f"\nBand structure (Γ → X):")
print(f"  {'k-point':>8s}  {'ε₁ (Ha)':>12s}")
for ik, eps in enumerate(r_bs.eps_per_k):
    label = "Γ" if ik == 0 else "X"
    print(f"  {label:>8s}  {np.asarray(eps)[0]:12.6f}")
