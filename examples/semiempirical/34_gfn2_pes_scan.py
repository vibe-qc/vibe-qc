#!/usr/bin/env python3
"""GFN2-xTB PES scans — bond-length and angle scans.

Mirrors the xtb workshop § "PES Scans". Demonstrates:
  1. Bond-length scan for water OH bond
  2. Angle scan for water H-O-H angle
  3. Combined 2D scan
  4. Identifying the minimum and computing curvature

Run:
    .venv/bin/python examples/semiempirical/34_gfn2_pes_scan.py
"""

from __future__ import annotations

import numpy as np
from vibeqc import Molecule, Atom
from vibeqc.semiempirical.methods.gfn2 import GFN2Model
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

params = load_gfn2_params()

SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def energy_at(mol, params):
    """Quick GFN2-xTB energy at a geometry."""
    return GFN2Model(mol, params=params, warn=False).energy()


# ── 1. OH bond scan in water ────────────────────────────────────────────
print("=" * 72)
print("1. OH bond-length scan (water)")

r_vals = np.linspace(1.2, 3.5, 24)
theta = np.deg2rad(104.5 / 2)

energies = []
print(f"   {'r(OH)':>6s}  {'E(Ha)':>14s}  {'ΔE(kcal/mol)':>14s}")
e_ref = None
for r in r_vals:
    mol_r = Molecule([
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [r * np.sin(theta), r * np.cos(theta), 0.0]),
        Atom(1, [-r * np.sin(theta), r * np.cos(theta), 0.0]),
    ])
    e = energy_at(mol_r, params)
    energies.append(e)
    if e_ref is None:
        e_ref = e
    print(f"   {r:6.3f}  {e:14.8f}  {(e - e_ref) * 627.509:+14.2f}")

energies = np.array(energies)
idx_min = np.argmin(energies)
print(f"\n   Minimum at r(OH) = {r_vals[idx_min]:.3f} bohr")
print(f"   E_min = {energies[idx_min]:.8f} Ha")

# Harmonic force constant from quadratic fit near minimum
window = slice(max(0, idx_min - 2), min(len(r_vals), idx_min + 3))
r_fit = r_vals[window]
e_fit = energies[window]
coeffs = np.polyfit(r_fit, e_fit, 2)
k = 2.0 * coeffs[0]  # E = ½ k (r - r0)²
print(f"   Force constant k ≈ {k:.4f} Ha/bohr²")

# ── 2. HOH angle scan ──────────────────────────────────────────────────
print("\n" + "=" * 72)
print("2. H-O-H angle scan (water)")

r_oh_fixed = 1.81
angles = np.linspace(70, 140, 15)

e_angles = []
print(f"   {'∠HOH':>6s}  {'E(Ha)':>14s}  {'ΔE(kcal/mol)':>14s}")
e_ang_ref = None
for ang in angles:
    theta_a = np.deg2rad(ang / 2)
    mol_a = Molecule([
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [r_oh_fixed * np.sin(theta_a), r_oh_fixed * np.cos(theta_a), 0.0]),
        Atom(1, [-r_oh_fixed * np.sin(theta_a), r_oh_fixed * np.cos(theta_a), 0.0]),
    ])
    e = energy_at(mol_a, params)
    e_angles.append(e)
    if e_ang_ref is None:
        e_ang_ref = e
    print(f"   {ang:6.1f}  {e:14.8f}  {(e - e_ang_ref) * 627.509:+14.2f}")

e_angles = np.array(e_angles)
idx_amin = np.argmin(e_angles)
print(f"\n   Minimum at ∠HOH = {angles[idx_amin]:.1f}°")
print(f"   E_min = {e_angles[idx_amin]:.8f} Ha")

# ── 3. H₂ bond scan ────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("3. H₂ bond-length scan")

r_h2 = np.linspace(0.8, 4.0, 33)
e_h2 = []
print(f"   {'r(HH)':>6s}  {'E(Ha)':>14s}")
for r in r_h2:
    mol_r = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, r])])
    e = energy_at(mol_r, params)
    e_h2.append(e)

e_h2 = np.array(e_h2)
idx_h2_min = np.argmin(e_h2)
print(f"\n   Minimum at r(HH) = {r_h2[idx_h2_min]:.3f} bohr")
print(f"   E_min = {e_h2[idx_h2_min]:.8f} Ha")
print(f"   Dissociation energy (r=4.0 vs min):")
print(f"      Dₑ ≈ {(e_h2[-1] - e_h2[idx_h2_min]) * 627.509:.1f} kcal/mol")

# ── 4. 2D scan (grid) ──────────────────────────────────────────────────
print("\n" + "=" * 72)
print("4. 2D scan — H₂O r(OH) × ∠HOH (coarse grid)")

r_grid = np.array([1.4, 1.6, 1.81, 2.0, 2.3])
ang_grid = np.array([80.0, 95.0, 104.5, 115.0, 130.0])

print(f"   {'r\\∠':>6s}", end="")
for ang in ang_grid:
    print(f"  {ang:>8.1f}°", end="")
print()

for r in r_grid:
    print(f"   {r:6.3f}", end="")
    for ang in ang_grid:
        theta_a = np.deg2rad(ang / 2)
        mol_2d = Molecule([
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [r * np.sin(theta_a), r * np.cos(theta_a), 0.0]),
            Atom(1, [-r * np.sin(theta_a), r * np.cos(theta_a), 0.0]),
        ])
        e = energy_at(mol_2d, params)
        print(f"  {e:10.6f}", end="")
    print()

print("\nDone.")
