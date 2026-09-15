#!/usr/bin/env python3
"""Example 4: Geometry optimization and preoptimization workflow.

Demonstrates:
  1. DFTB0 geometry optimization via ASE
  2. SCC-DFTB energy comparison
  3. Dispersion correction with D3(BJ)
  4. Preoptimization → ab initio handoff workflow

Run: python examples/semiempirical/04_optimize_and_preoptimize.py
Requires: pip install ase
"""

from __future__ import annotations

import numpy as np
from vibeqc import Atom, Molecule
from vibeqc.semiempirical import (
    DFTB_D3_DEFAULTS,
    DFTB0Model,
    DispersionCorrectedModel,
    SCCDFTBModel,
    d3bj_energy,
)

# Build a distorted H2O geometry
theta = np.deg2rad(104.5 / 2)
r_oh = 2.2  # stretched from equilibrium ~1.81 bohr
mol = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [r_oh * np.sin(theta), r_oh * np.cos(theta), 0.0]),
        Atom(1, [-r_oh * np.sin(theta), r_oh * np.cos(theta), 0.0]),
    ],
    charge=0,
    multiplicity=1,
)

# ---- Part 1: Single-point comparison ----
print("=== Single-Point Energies ===")
model_dftb0 = DFTB0Model(mol)
model_scc = SCCDFTBModel(mol)

e_dftb0 = model_dftb0.energy()
e_scc = model_scc.energy()
e_d3 = d3bj_energy(mol)

print(f"DFTB0:              {e_dftb0:12.6f} Ha")
print(f"SCC-DFTB:           {e_scc:12.6f} Ha")
print(f"D3(BJ) dispersion:  {e_d3:12.6f} Ha")
print(f"SCC + D3:           {e_scc + e_d3:12.6f} Ha")

# ---- Part 2: Geometry optimization ----
print("\n=== Geometry Optimization ===")
try:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.optimize import BFGSLineSearch
    from ase.units import Bohr, Hartree

    class _DFTB0Calc(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms, properties, system_changes):
            pos_bohr = atoms.positions / Bohr
            m = Molecule(
                [Atom(int(z), list(xyz)) for z, xyz in zip(atoms.numbers, pos_bohr)],
                charge=0,
                multiplicity=1,
            )
            model = DFTB0Model(m)
            self.results["energy"] = model.energy() * Hartree
            self.results["forces"] = -model.gradient() * (Hartree / Bohr)

    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=np.array([np.array(a.xyz) for a in mol.atoms]) * Bohr,
    )
    atoms.calc = _DFTB0Calc()

    opt = BFGSLineSearch(atoms, logfile=None)
    opt.run(fmax=0.05, steps=50)

    # Measure final bond lengths
    final_pos = atoms.positions / Bohr
    for i in range(1, 3):
        r = np.linalg.norm(final_pos[i] - final_pos[0])
        print(f"  O-H{i} bond: {r:.3f} bohr = {r * 0.529:.3f} Å")

    print(f"  Steps: {opt.nsteps}")
except ImportError:
    print("  (ASE not installed — skipping optimization)")

# ---- Part 3: Dispersion-corrected model ----
print("\n=== Dispersion-Corrected Model ===")
disp_model = DispersionCorrectedModel(model_scc, DFTB_D3_DEFAULTS)
e_disp_total = disp_model.energy()
print(f"SCC + D3 via wrapper: {e_disp_total:12.6f} Ha")
print(f"Dispersion contribution: {e_disp_total - e_scc:+.6f} Ha")
