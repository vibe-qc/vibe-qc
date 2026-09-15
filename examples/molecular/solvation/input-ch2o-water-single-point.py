"""Formaldehyde in water — single-point CPCM hydration energy.

The "your first solvation calculation" example. Computes the gas-phase
SCF and the in-water SCF (CPCM), and prints the polarisation energy
and total ΔG_solv. Matches Section 1 of the v0.9.0 solvation tutorial
(docs/tutorial/solvation_water.md).

Run:
    .venv/bin/python examples/molecular/solvation/input-ch2o-water-single-point.py

Outputs (next to this script):
    input-ch2o-water-single-point.gas.out
    input-ch2o-water-single-point.water.out
    input-ch2o-water-single-point.gas.molden
    input-ch2o-water-single-point.water.molden
    input-ch2o-water-single-point.gas.system
    input-ch2o-water-single-point.water.system
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem

ANG = 1.8897261339213   # Å → bohr

mol = vq.Molecule([
    vq.Atom(6, [0.0,  0.000 * ANG,  0.0000 * ANG]),
    vq.Atom(8, [0.0,  0.000 * ANG,  1.2055 * ANG]),
    vq.Atom(1, [0.0,  0.940 * ANG, -0.5870 * ANG]),
    vq.Atom(1, [0.0, -0.940 * ANG, -0.5870 * ANG]),
], 0, 1)

# Gas-phase reference.
gas = vq.run_job(
    mol, basis="def2-svp", method="rks", functional="pbe0",
    output=HERE / f"{STEM}.gas",
)

# In water — CPCM (Cossi-Scalmani 2003) with PySCF / ORCA conventions:
# scaled-Bondi cavity (× 1.20), 302 Lebedev points per atomic sphere,
# Scalmani-Frisch continuous switching, ε = 78.39.
aqueous = vq.run_job(
    mol, basis="def2-svp", method="rks", functional="pbe0",
    solvent="water",
    output=HERE / f"{STEM}.water",
)

sol = aqueous.solvent_result
kcal_per_Ha = 627.5094740631

delta_g_total = sol.energy - gas.energy
print(f"\n=== Formaldehyde hydration summary ===")
print(f"E(gas)            = {gas.energy:14.6f} Ha")
print(f"E(water, total)   = {sol.energy:14.6f} Ha")
print(f"E_solv (½ q · V)  = {sol.e_solv:+12.6f} Ha = "
      f"{sol.e_solv * kcal_per_Ha:+8.3f} kcal/mol")
print(f"ΔG_solv (total)   = {delta_g_total:+12.6f} Ha = "
      f"{delta_g_total * kcal_per_Ha:+8.3f} kcal/mol")
print(f"CPCM macro-iters  = {sol.n_macro_iter}")
print(f"Cavity points     = {sol.cavity.n_points} (after switching)")
print(f"Solvent           = {sol.solvent_name}  (ε = {sol.epsilon:.3f})")
