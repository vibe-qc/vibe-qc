"""Smoke-test heterocycle / fused-ring / nucleobase / benzene-derivative naming.

Run from vibe-qc root:
    PYTHONPATH=python python3 python/vibeqc_naming/_test_heterocycle.py
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, "python")

from vibeqc_naming import from_atoms_list, name_from_atoms, name_ring_system


def _regular_polygon(n, radius, z=0.0):
    """n atoms equally spaced on a circle of given radius."""
    a = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        a.append((radius * math.cos(angle), radius * math.sin(angle), z))
    return a


def _build_ring(atoms_with_types, ring_positions):
    """Given [(Z, idx)] and [(x,y,z)], produce full atom list."""
    return [
        (
            atoms_with_types[i][0],
            ring_positions[i][0],
            ring_positions[i][1],
            ring_positions[i][2],
        )
        for i in range(len(atoms_with_types))
    ]


def _add_ring_hydrogens(core_atoms, center=(0, 0, 0), bond_len=1.09):
    """Add H atoms to ring carbons (radially outward from center)."""
    result = list(core_atoms)
    for i, (z, x, y, z0) in enumerate(core_atoms):
        if z != 6:
            continue  # only add H to carbons
        dx, dy = x - center[0], y - center[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.01:
            continue
        dx, dy = dx / dist, dy / dist
        result.append((1, x + dx * bond_len, y + dy * bond_len, z0))
    return result


tests = 0
failed = 0


def check(label, atoms, expected, exact=False):
    global tests, failed
    tests += 1
    result = name_from_atoms(atoms)
    ok = (result == expected) if exact else (expected.lower() in result.lower())
    if ok:
        print(f"  {chr(0x2713)} {label}: '{result}'")
    else:
        print(f"  {chr(0x2717)} {label}: got '{result}', expected '{expected}'")
        failed += 1


def graph_info(atoms):
    g = from_atoms_list(atoms)
    return f"n={g.n_atoms} bonds={len(g.bonds)} rings={len(g.rings)}"


print("=" * 60)
print("TEST: Heterocycle / fused-ring / nucleobase naming")
print("=" * 60)

# ── Pyridine: C5H5N, 6-membered ring with 1 N ──────────────────────────

print("\n-- Pyridine --")
r = 1.40
verts = _regular_polygon(6, r)
# Replace vertex 0 with N
pyridine_core = [(7, verts[0][0], verts[0][1], 0)]
for i in range(1, 6):
    pyridine_core.append((6, verts[i][0], verts[i][1], 0))
# Add H to carbons only
center = (sum(v[0] for v in verts) / 6, sum(v[1] for v in verts) / 6, 0)
pyridine = _add_ring_hydrogens(pyridine_core, center)
print(f"  {graph_info(pyridine)}")
check("pyridine", pyridine, "pyridine")

# ── Phenol: C6H5OH = benzene + OH ──────────────────────────────────────

print("\n-- Phenol --")
verts = _regular_polygon(6, r)
benzene_core = [(6, v[0], v[1], 0) for v in verts]
center = (0, 0, 0)
# Add 5 ring H (one C gets OH instead)
phenol = [(6, verts[0][0], verts[0][1], 0)]  # C with OH
for i in range(1, 6):
    phenol.append((6, verts[i][0], verts[i][1], 0))
# OH substituent on atom 0
phenol.append((8, verts[0][0] + verts[0][0] * 0.7, verts[0][1] + verts[0][1] * 0.7, 0))
phenol.append((1, verts[0][0] + verts[0][0] * 1.2, verts[0][1] + verts[0][1] * 1.2, 0))
# H on remaining carbons (indices 1-5)
for i in range(1, 6):
    dx, dy = verts[i][0] - center[0], verts[i][1] - center[1]
    d = math.sqrt(dx * dx + dy * dy)
    dx, dy = dx / d, dy / d
    phenol.append((1, verts[i][0] + dx * 1.09, verts[i][1] + dy * 1.09, 0))
print(f"  {graph_info(phenol)}")
check("phenol", phenol, "phenol")

# ── Toluene: methyl-benzene ────────────────────────────────────────────

print("\n-- Toluene --")
toluene = [(6, verts[0][0], verts[0][1], 0)]
for i in range(1, 6):
    toluene.append((6, verts[i][0], verts[i][1], 0))
# Methyl on atom 0
toluene.append((6, verts[0][0] * 1.7, verts[0][1] * 1.7, 0))
# 3 H on methyl
for j in range(3):
    ang = j * 2 * math.pi / 3
    toluene.append(
        (
            1,
            verts[0][0] * 2.2 + 0.5 * math.cos(ang),
            verts[0][1] * 2.2 + 0.5 * math.sin(ang),
            0,
        )
    )
# 5 ring H
for i in range(1, 6):
    dx, dy = verts[i][0] - center[0], verts[i][1] - center[1]
    d = math.sqrt(dx * dx + dy * dy)
    dx, dy = dx / d, dy / d
    toluene.append((1, verts[i][0] + dx * 1.09, verts[i][1] + dy * 1.09, 0))
print(f"  {graph_info(toluene)}")
check("toluene", toluene, "toluene")

# ── Aniline: NH2-benzene ───────────────────────────────────────────────

print("\n-- Aniline --")
r2 = 1.40
aniline = [
    (6, r2, 0, 0),
    (6, 0.70, 1.212, 0),
    (6, -0.70, 1.212, 0),
    (6, -r2, 0, 0),
    (6, -0.70, -1.212, 0),
    (6, 0.70, -1.212, 0),
    (7, r2 + 1.49, 0, 0),      # N
    (1, r2 + 2.20, 0.50, 0),   # H on N
    (1, r2 + 2.20, -0.50, 0),  # H on N
    (1, 0.70, 2.30, 0),        # ring H
    (1, -1.15, 2.00, 0),
    (1, -r2 - 1.09, 0, 0),
    (1, -0.70, -2.30, 0),
    (1, 1.15, -2.00, 0),
]
print(f"  {graph_info(aniline)}")
check("aniline", aniline, "aniline")

# ── Naphthalene: two fused benzenes ────────────────────────────────────

print("\n-- Naphthalene (C10H8) --")
# Standard naphthalene: two hexagons sharing a bond
naph_core = [
    (6, 0.0, 0.0, 0),
    (6, 1.40, 0.0, 0),
    (6, 2.10, 1.212, 0),
    (6, 0.70, 2.424, 0),
    (6, -0.70, 2.424, 0),
    (6, -2.10, 1.212, 0),
    (6, -2.10, -1.212, 0),
    (6, -0.70, -2.424, 0),
    (6, 0.70, -2.424, 0),
    (6, 2.10, -1.212, 0),
]
naphthalene = list(naph_core)
# 8 hydrogens (H at positions 1, 3, 4, 5, 6, 8 have H)
h_positions = [
    (0.0, -0.7, 0),
    (2.10, -0.7, 0),
    (3.0, 1.732, 0),
    (1.0, 3.0, 0),
    (-2.0, 3.0, 0),
    (-3.0, 1.732, 0),
    (-3.0, -1.732, 0),
    (-1.0, -3.0, 0),
]
for hx, hy, hz in h_positions:
    naphthalene.append((1, hx, hy, hz))
print(f"  {graph_info(naphthalene)}")
check("naphthalene", naphthalene, "naphthalene")

# ── Regression: existing molecules still work ──────────────────────────

print("\n-- Regression --")
benzene_core = [(6, v[0], v[1], 0) for v in _regular_polygon(6, 1.40)]
benzene = _add_ring_hydrogens(benzene_core)
check("benzene", benzene, "benzene")

water = [(8, 0, 0, 0.1173), (1, 0, 0.7572, -0.4692), (1, 0, -0.7572, -0.4692)]
check("water", water, "water")

methane = [
    (6, 0, 0, 0),
    (1, 0.6276, 0.6276, 0.6276),
    (1, 0.6276, -0.6276, -0.6276),
    (1, -0.6276, 0.6276, -0.6276),
    (1, -0.6276, -0.6276, 0.6276),
]
check("methane", methane, "methane")

# ── Summary ────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"Results: {tests - failed}/{tests} passed")
if failed:
    print(f"FAILED: {failed} tests")
    sys.exit(1)
print("All tests passed")
