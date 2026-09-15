#!/usr/bin/env python3
"""Example 5: Production DFTB workflow — from parameter fitting to optimization.

Demonstrates the complete production DFTB pipeline:
  1. Load production parameters with spline-fitted repulsives
  2. Save/load parameter files (TOML I/O)
  3. Compare DFTB0 vs SCC-DFTB vs SCC+D3
  4. Geometry optimization with production parameters
  5. Preoptimization → ready for ab initio refinement

Run: python examples/semiempirical/05_production_workflow.py
Requires: pip install ase
"""

from __future__ import annotations

import numpy as np
import tempfile
from pathlib import Path

from vibeqc import Atom, Molecule
from vibeqc.semiempirical import (
    DFTB0Model,
    SCCDFTBModel,
    DispersionCorrectedModel,
    DFTB_D3_DEFAULTS,
    d3bj_energy,
)
from vibeqc.semiempirical.io import (
    load_production_parameters,
    save_parameters,
    load_parameters,
)

# ---- Step 1: Load production parameters ----
print("=" * 60)
print("Step 1: Load production DFTB parameters")
print("=" * 60)
params = load_production_parameters()
print(f"  kappa = {params.kappa}")
print(f"  Elements: {sum(1 for Z in range(1,87) if params.has_element(Z))}")
for Z in [1, 6, 7, 8]:
    print(f"  Z={Z}: ε_s={params.on_site_energy(Z,0):.4f}, U={params.hubbard_u(Z):.4f}")

# ---- Step 2: Save and reload parameters ----
print("\n" + "=" * 60)
print("Step 2: Save and reload parameter file (TOML I/O)")
print("=" * 60)
with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
    tmp = f.name
save_parameters(params, tmp)
params2 = load_parameters(tmp)
Path(tmp).unlink()
print(f"  Round-trip OK: kappa={params2.kappa}, has O={params2.has_element(8)}")

# ---- Step 3: Build molecule and compute energies ----
print("\n" + "=" * 60)
print("Step 3: Single-point energies for H2O")
print("=" * 60)
theta = np.deg2rad(104.5 / 2)
mol = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [1.81 * np.sin(theta), 1.81 * np.cos(theta), 0.0]),
        Atom(1, [-1.81 * np.sin(theta), 1.81 * np.cos(theta), 0.0]),
    ],
    charge=0,
    multiplicity=1,
)

# Use production params (spline repulsives)
dftb0_prod = DFTB0Model(mol, params=params)
e_dftb0 = dftb0_prod.energy()
e_d3 = d3bj_energy(mol)
print(f"  DFTB0 (spline repulsives): {e_dftb0:12.6f} Ha")
print(f"  + D3(BJ) dispersion:       {e_dftb0 + e_d3:12.6f} Ha")

# Compare with default R⁻¹² params
dftb0_default = DFTB0Model(mol)  # uses dftb0_default()
e_def = dftb0_default.energy()
print(f"  DFTB0 (R⁻¹² default):      {e_def:12.6f} Ha")

# SCC-DFTB with production params
e_scc = SCCDFTBModel(mol, params=params).energy()
print(f"  SCC-DFTB (production):     {e_scc:12.6f} Ha")
print(f"  SCC + D3:                  {e_scc + e_d3:12.6f} Ha")

# ---- Step 4: Geometry optimization ----
print("\n" + "=" * 60)
print("Step 4: Geometry optimization with production DFTB")
print("=" * 60)
try:
    from ase import Atoms
    from ase.optimize import BFGSLineSearch
    from ase.units import Bohr, Hartree
    from ase.calculators.calculator import Calculator

    class _DFTB0Calc(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms, properties, system_changes):
            pos_bohr = atoms.positions / Bohr
            m = Molecule(
                [Atom(int(z), list(xyz)) for z, xyz in zip(atoms.numbers, pos_bohr)],
                charge=0, multiplicity=1,
            )
            model = DFTB0Model(m, params=params)
            self.results["energy"] = model.energy() * Hartree
            self.results["forces"] = -model.gradient() * (Hartree / Bohr)

    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=np.array([np.array(a.xyz) for a in mol.atoms]) * Bohr,
    )
    atoms.calc = _DFTB0Calc()
    opt = BFGSLineSearch(atoms, logfile=None)
    opt.run(fmax=0.05, steps=50)

    final_pos = atoms.positions / Bohr
    for i in range(1, 3):
        r = np.linalg.norm(final_pos[i] - final_pos[0])
        print(f"  O-H{i} bond: {r:.3f} bohr ({r * 0.529:.3f} Å)")
    print(f"  Optimization steps: {opt.nsteps}")
except ImportError:
    print("  (ASE not installed — skipping)")

# ---- Step 5: Dispersion-corrected energy decomposition ----
print("\n" + "=" * 60)
print("Step 5: Dispersion-corrected energy decomposition")
print("=" * 60)
disp_model = DispersionCorrectedModel(SCCDFTBModel(mol, params=params), DFTB_D3_DEFAULTS)
e_total = disp_model.energy()
e_base = SCCDFTBModel(mol, params=params).energy()
print(f"  SCC-DFTB base:     {e_base:12.6f} Ha")
print(f"  D3(BJ) correction: {e_total - e_base:+.6f} Ha")
print(f"  SCC + D3 total:    {e_total:12.6f} Ha")

print("\n✅ Production DFTB workflow complete!")
