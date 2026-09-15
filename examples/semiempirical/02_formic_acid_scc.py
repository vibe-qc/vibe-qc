#!/usr/bin/env python3
"""Example 2: SCC-DFTB with charge self-consistency for formic acid.

Demonstrates:
  1. Building a molecule with multiple elements
  2. SCC-DFTB with charge mixing
  3. Mulliken charge analysis
  4. Energy decomposition (electronic + repulsive + SCC)

Run: python examples/semiempirical/02_formic_acid_scc.py
"""

from __future__ import annotations

import numpy as np
from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc.semiempirical.parameters import default_parameters

# HCOOH at approximate geometry (bohr)
# C=O ≈ 2.3, C-O ≈ 2.5, C-H ≈ 2.0, O-H ≈ 1.8
mol = Molecule(
    [
        Atom(6, [0.000, 0.000, 0.000]),  # C
        Atom(8, [0.000, 0.000, 2.300]),  # O (carbonyl)
        Atom(8, [0.000, 2.500, -1.000]),  # O (hydroxyl)
        Atom(1, [0.000, -2.000, 0.000]),  # H (formyl)
        Atom(1, [0.000, 3.200, 0.500]),  # H (hydroxyl)
    ],
    charge=0,
    multiplicity=1,
)

params = default_parameters()
opts = _se.SCCOptions()
opts.charge_mixing = 0.3
opts.max_iter = 200

# Run SCC-DFTB
result = _se.run_scc_dftb(mol, params, opts)

print("SCC-DFTB: Formic Acid (HCOOH)")
print(f"  Total energy:      {result.energy:12.6f} Ha")
print(f"  Electronic (band): {result.e_electronic:12.6f} Ha")
print(f"  Repulsive:         {result.e_repulsive:12.6f} Ha")
print(f"  SCC correction:    {result.e_scc:12.6f} Ha")
print(f"  Converged: {result.converged} in {result.n_iter} iterations")
print()
print("Mulliken charge fluctuations (Δq):")
charges = np.asarray(result.charges)
for i, atom in enumerate(mol.atoms):
    sym = {6: "C", 8: "O", 1: "H"}[atom.Z]
    print(f"  {sym}: Δq = {charges[i]:+.4f} e")
print(f"  Sum Δq = {charges.sum():.6f} e")

# Compare with DFTB0
result0 = _se.run_dftb0(mol, params)
print(f"\nDFTB0 energy: {result0.energy:12.6f} Ha")
print(f"SCC stabilization: {result.energy - result0.energy:+.6f} Ha")
