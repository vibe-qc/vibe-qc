#!/usr/bin/env python3
"""Example 1: Fast DFTB0 energy and gradient for water.

Demonstrates the simplest semiempirical workflow:
  1. Build a molecule from scratch
  2. Compute DFTB0 energy
  3. Compute analytic gradient
  4. Verify translational invariance

Run: python examples/semiempirical/01_water_dftb0.py
"""

from __future__ import annotations

import numpy as np
from vibeqc import Atom, Molecule
from vibeqc.semiempirical import DFTB0Model

# Build H2O at approximate equilibrium geometry (bohr)
theta = np.deg2rad(104.5 / 2)
r_oh = 1.81
mol = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [r_oh * np.sin(theta), r_oh * np.cos(theta), 0.0]),
        Atom(1, [-r_oh * np.sin(theta), r_oh * np.cos(theta), 0.0]),
    ],
    charge=0,
    multiplicity=1,
)

# Run DFTB0
model = DFTB0Model(mol)
energy = model.energy()
gradient = model.gradient()

print(f"H₂O DFTB0 energy:     {energy:12.6f} Ha")
print(f"H₂O DFTB0 energy:     {energy * 27.2114:12.6f} eV")
print()
print("Analytic gradient (Ha/bohr):")
for i, atom in enumerate(mol.atoms):
    print(f"  Atom {i} (Z={atom.Z}): {gradient[i]}")

# Translational invariance check
net_force = np.sum(gradient, axis=0)
print(f"\nNet force: {net_force}")
print(
    f"Translational invariance: {'PASS' if np.allclose(net_force, 0, atol=1e-12) else 'FAIL'}"
)
