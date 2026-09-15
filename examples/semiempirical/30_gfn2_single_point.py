#!/usr/bin/env python3
"""GFN2-xTB single-point energies — water, methane, ammonia, CO₂.

Mirrors the xtb workshop § "SP". Demonstrates:
  1. Parameter-set auto-fetch and inspection
  2. Single-point energy via run_job and GFN2Model
  3. Energy component decomposition (electronic, repulsive, SCC, D4)
  4. Mulliken charge analysis
  5. Element-coverage check

Run:
    .venv/bin/python examples/semiempirical/30_gfn2_single_point.py
"""

from __future__ import annotations

import numpy as np
import warnings
from vibeqc import Molecule, Atom, run_job
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc.semiempirical.methods.gfn2 import GFN2Model, GFN2ExperimentalWarning
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

# ── 1. Fetch parameters ─────────────────────────────────────────────────
print("=" * 72)
print("1. Parameter set")
params = load_gfn2_params()
print(f"   Elements: {params.n_elements()}")
print(f"   Has C: {params.has_element(6)}   Has Fe: {params.has_element(26)}")
ed = params.element(6)
print(f"   C: zeta={list(ed.zeta)}  Hubbard U={ed.hubbard_u:.4f}")
print(f"      on_site={[f'{v:.4f}' for v in ed.on_site]}  valence e={ed.valence_electrons}")
print(f"   D4 params: s8={params.d4_s8:.3f}  a1={params.d4_a1:.3f}  a2={params.d4_a2:.3f}")

# ── 2. Water single-point ───────────────────────────────────────────────
print("\n" + "=" * 72)
print("2. Water (H2O)")

mol_h2o = Molecule([
    Atom(8, [ 0.00,  0.00,  0.00]),
    Atom(1, [ 1.43,  0.98,  0.00]),
    Atom(1, [-1.43,  0.98,  0.00]),
])

# Direct C++ call — shows SCC result before D4
opts = _xtb.XTBSccOptions()
result = _xtb.run_gfn2_xtb(mol_h2o, params, opts)
print(f"   SCC result:  E_electronic = {result.e_electronic:12.6f} Ha")
print(f"                E_repulsive  = {result.e_repulsive:12.6f} Ha")
print(f"                E_scc        = {result.e_scc:12.6f} Ha")
print(f"                SCC total    = {result.energy:12.6f} Ha")
print(f"   charges: {[f'{q:+.4f}' for q in result.charges]}")
print(f"   converged={result.converged}  n_iter={result.n_iter}")

# GFN2Model — includes post-SCF D4
model = GFN2Model(mol_h2o, params=params, warn=False)
print(f"   + D4:       total        = {model.energy():12.6f} Ha")
print(f"   MO energies (first 8): {[f'{e:.3f}' for e in result.mo_energies[:8]]} Ha")
print(f"   n_basis={result.n_basis}  n_occ={result.n_occ}")

# ── 3. Methane ──────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("3. Methane (CH4)")
# Tetrahedral, C-H ≈ 2.05 bohr
d = 2.05 / np.sqrt(3)
mol_ch4 = Molecule([
    Atom(6, [ 0.00,  0.00,  0.00]),
    Atom(1, [    d,     d,     d]),
    Atom(1, [   -d,    -d,     d]),
    Atom(1, [   -d,     d,    -d]),
    Atom(1, [    d,    -d,    -d]),
])
r_ch4 = _xtb.run_gfn2_xtb(mol_ch4, params, opts)
print(f"   Energy:   {r_ch4.energy:12.6f} Ha  (+D4: {GFN2Model(mol_ch4, params, warn=False).energy():.6f})")
print(f"   Charges:  {[f'{q:+.4f}' for q in r_ch4.charges]}")
print(f"   converged={r_ch4.converged}  n_iter={r_ch4.n_iter}")

# ── 4. Ammonia ──────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("4. Ammonia (NH3)")
# Pyramidal, N-H ≈ 1.91 bohr, H-N-H ≈ 107°
theta_nh3 = np.deg2rad(107.0 / 2)
r_nh = 1.91
mol_nh3 = Molecule([
    Atom(7, [0.0, 0.0, 0.0]),
    Atom(1, [0.0, r_nh * np.cos(theta_nh3), r_nh * np.sin(theta_nh3)]),
    Atom(1, [r_nh * np.sin(theta_nh3) * np.cos(np.deg2rad(30)),
             r_nh * np.cos(theta_nh3),
             -r_nh * np.sin(theta_nh3) * np.sin(np.deg2rad(30))]),
    Atom(1, [-r_nh * np.sin(theta_nh3) * np.cos(np.deg2rad(30)),
             r_nh * np.cos(theta_nh3),
             -r_nh * np.sin(theta_nh3) * np.sin(np.deg2rad(30))]),
])
r_nh3 = _xtb.run_gfn2_xtb(mol_nh3, params, opts)
print(f"   Energy:   {r_nh3.energy:12.6f} Ha")
print(f"   Charges:  {[f'{q:+.4f}' for q in r_nh3.charges]}")
print(f"   converged={r_nh3.converged}  n_iter={r_nh3.n_iter}")

# ── 5. CO₂ ──────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("5. Carbon dioxide (CO2)")
mol_co2 = Molecule([
    Atom(8, [-2.20, 0.0, 0.0]),
    Atom(6, [ 0.00, 0.0, 0.0]),
    Atom(8, [ 2.20, 0.0, 0.0]),
])
r_co2 = _xtb.run_gfn2_xtb(mol_co2, params, opts)
print(f"   Energy:   {r_co2.energy:12.6f} Ha")
print(f"   Charges:  {[f'{q:+.4f}' for q in r_co2.charges]}")
print(f"   converged={r_co2.converged}  n_iter={r_co2.n_iter}")

# ── 6. run_job (text output) ────────────────────────────────────────────
print("\n" + "=" * 72)
print("6. run_job integration")
print("   -> see h2o_gfn2.out")
run_job(mol_h2o, method="gfn2_xtb", output="h2o_gfn2")

print("\nDone. Output files: h2o_gfn2.out")
