"""Smoke-test the surface/slab naming module.

Run from vibe-qc root:
    PYTHONPATH=python python3 python/vibeqc_naming/_test_surface.py
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, "python")

from vibeqc_naming import (
    Confidence,
    detect_slab,
    from_atoms_list,
    name_from_atoms_with_lattice,
)


def _build_mgo_slab_001(
    nx: int, ny: int, nz: int
) -> list[tuple[int, float, float, float]]:
    """Build a rocksalt MgO(001) slab.

    Mg at corner + face-center positions, O at edge-center + body-center.
    Lattice constant a = 4.21 A, nearest-neighbor Mg-O distance = a/2 = 2.105 A.
    Covalent radii: Mg 1.41 A, O 0.66 A -> bond cutoff 1.25*(1.41+0.66)=2.5875 A.
    Mg-O bond at 2.105 A will be detected.
    """
    a = 4.21
    atoms = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x0, y0, z = ix * a, iy * a, iz * a / 2
                # Mg at (0,0,z) and (a/2, a/2, z)
                atoms.append((12, x0, y0, z))
                atoms.append((12, x0 + a / 2, y0 + a / 2, z))
                # O at (a/2, 0, z) and (0, a/2, z)
                atoms.append((8, x0 + a / 2, y0, z))
                atoms.append((8, x0, y0 + a / 2, z))
    return atoms


def _make_benzene(
    cx: float, cy: float, cz: float
) -> list[tuple[int, float, float, float]]:
    """Benzene ring at (cx, cy, cz). C-C = 1.40 A, C-H = 1.09 A."""
    r = 1.40
    atoms = []
    for i in range(6):
        angle = i * math.pi / 3
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        z = cz
        atoms.append((6, x, y, z))
    for i in range(6):
        angle = i * math.pi / 3
        x = cx + 2.49 * math.cos(angle)
        y = cy + 2.49 * math.sin(angle)
        z = cz
        atoms.append((1, x, y, z))
    return atoms


a = 4.21
lattice = [
    (2 * a, 0.0, 0.0),  # 2 by 2 in-plane supercell below
    (0.0, 2 * a, 0.0),
    (0.0, 0.0, 30.0),  # vacuum
]

print("=" * 60)
print("TEST: Surface / slab naming")
print("=" * 60)

# ── Test 1: Bare MgO slab ────────────────────────────────────────────────

print("\n-- Test 1: Bare MgO(001) slab --")
slab_atoms = _build_mgo_slab_001(2, 2, 2)
graph = from_atoms_list(slab_atoms)
print(f"  n_atoms: {graph.n_atoms}, formula: {graph.formula}")
print(f"  n_bonds: {len(graph.bonds)}")
print(f"  n_components: {len(graph.connected_components())}")
assert len(graph.connected_components()) == 1, "Should be one connected slab"
print("  slab is fully connected ✓")

slab = detect_slab(graph, lattice)
print(f"  is_slab: {slab.is_slab}, material: {slab.slab_material!r}")
assert slab.is_slab
assert "magnesia" in slab.slab_material or "Mg" in slab.slab_material
print("  slab detected ✓")

result = name_from_atoms_with_lattice(slab_atoms, lattice, pbc=(True, True, False))
print(f"  name: {result.name!r}")
print(f"  confidence: {result.confidence.value}")
assert "magnesia" in result.name or "Mg" in result.name
print("  slab name ✓")

# ── Test 2: Benzene on MgO(001) ──────────────────────────────────────────

print("\n-- Test 2: Benzene on MgO(001) --")
# Place benzene 4 A above the top surface
nz = 2
top_z = (nz - 0.5) * a / 2 + 4.0
benzene = _make_benzene(cx=a, cy=a, cz=top_z)
combined = slab_atoms + benzene

graph2 = from_atoms_list(combined)
comps = graph2.connected_components()
print(f"  n_components: {len(comps)}")
for i, comp in enumerate(comps[:4]):
    syms = {graph2.atoms[j].symbol for j in comp}
    print(f"    comp {i}: {len(comp)} atoms, types={syms}")

assert len(comps) >= 2, f"Expected >=2 components, got {len(comps)}"
# Find the benzene component
benzene_comps = [c for c in comps if any(graph2.atoms[j].z == 6 for j in c)]
assert len(benzene_comps) == 1
assert len(benzene_comps[0]) == 12  # C6H6
print("  benzene as separate component ✓")

result2 = name_from_atoms_with_lattice(combined, lattice, pbc=(True, True, False),
    adsorbate_groups=[list(range(len(slab_atoms), len(combined)))])
print(f"  name: {result2.name!r}")
print(f"  confidence: {result2.confidence.value}")
assert "on" in result2.name
print("  combined name ✓")

# ── Test 3: CO on MgO(001) ───────────────────────────────────────────────

print("\n-- Test 3: CO on MgO(001) --")
co_z = top_z + 3.0
co = [(6, a, a, co_z), (8, a, a, co_z + 1.13)]
combined3 = slab_atoms + co

result3 = name_from_atoms_with_lattice(combined3, lattice, pbc=(True, True, False),
    adsorbate_groups=[list(range(len(slab_atoms), len(combined3)))])
print(f"  name: {result3.name!r}")
assert result3.name == "carbon monoxide on magnesia"
assert "Mg" in result3.name or "magnesia" in result3.name or "MgO" in result3.name
print("  CO-on-MgO ✓")

# ── Test 4: Multiple adsorbates on MgO ───────────────────────────────────

print("\n-- Test 4: H2O and CO on MgO(001) --")
h2o_top = [
    (8, a, a + 2.0, top_z),
    (1, a, a + 2.5, top_z + 0.757),
    (1, a, a + 1.5, top_z + 0.757),
]
co_top = [(6, a + 3, a, top_z), (8, a + 3, a, top_z + 1.13)]
combined4 = slab_atoms + h2o_top + co_top

result4 = name_from_atoms_with_lattice(combined4, lattice, pbc=(True, True, False),
    adsorbate_groups=[list(range(len(slab_atoms), len(slab_atoms) + 3)),
                      list(range(len(slab_atoms) + 3, len(combined4)))])
print(f"  name: {result4.name!r}")
assert "on" in result4.name
assert "carbon monoxide" in result4.name and "water" in result4.name
print("  multiple adsorbates ✓")

# ── Test 5: Components on graph ──────────────────────────────────────────

print("\n-- Test 5: connected_components API --")
graph5 = from_atoms_list(combined)
comps5 = graph5.connected_components()
assert len(comps5) == 2  # slab + benzene
sub = graph5.subgraph(comps5[1])  # benzene component
assert sub.n_atoms == 12
assert "C" in sub.formula
print(f"  subgraph formula: {sub.formula} ✓")

# ── Summary ──────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("All surface/slab tests passed ✓")
print("=" * 60)
