"""Smoke-test the IUPAC naming module (standalone vibeqc_naming package).

Run from vibeqc-zed root:
    PYTHONPATH=python python3 python/vibeqc_naming/_smoke_test.py
    # or
    cd python && python3 -m vibeqc_naming._smoke_test
"""

from __future__ import annotations

import sys

sys.path.insert(0, "python")

from collections import Counter

# ── Imports ───────────────────────────────────────────────────────────────────
from vibeqc_naming import (
    Confidence,
    NamedResult,
    NamingSource,
    build_graph,
    formula_to_name,
    formula_to_name_detailed,
    from_atoms_list,
    name_from_atoms,
    name_from_atoms_detailed,
    name_from_graph,
    name_from_graph_detailed,
)
from vibeqc_naming.molecular_graph import ATOMIC_NUMBER_TO_SYMBOL

print("All imports OK ✓")

# ── Test data ─────────────────────────────────────────────────────────────────

_WATER_ATOMS = [
    (8, 0.0, 0.0, 0.1173),
    (1, 0.0, 0.7572, -0.4692),
    (1, 0.0, -0.7572, -0.4692),
]

_METHANE_ATOMS = [
    (6, 0.0, 0.0, 0.0),
    (1, 0.6276, 0.6276, 0.6276),
    (1, 0.6276, -0.6276, -0.6276),
    (1, -0.6276, 0.6276, -0.6276),
    (1, -0.6276, -0.6276, 0.6276),
]

_BENZENE_ATOMS = [
    (6, 1.40, 0.0, 0.0),
    (6, 0.70, 1.212, 0.0),
    (6, -0.70, 1.212, 0.0),
    (6, -1.40, 0.0, 0.0),
    (6, -0.70, -1.212, 0.0),
    (6, 0.70, -1.212, 0.0),
    (1, 2.16, 0.0, 0.0),
    (1, 1.08, 1.87, 0.0),
    (1, -1.08, 1.87, 0.0),
    (1, -2.16, 0.0, 0.0),
    (1, -1.08, -1.87, 0.0),
    (1, 1.08, -1.87, 0.0),
]

_CO2_ATOMS = [
    (6, 0.0, 0.0, 0.0),
    (8, 1.16, 0.0, 0.0),
    (8, -1.16, 0.0, 0.0),
]

_ETHANOL_ATOMS = [
    (6, -0.95, 0.0, 0.0),
    (6, 0.53, 0.0, 0.0),
    (8, 1.15, 1.05, 0.0),
    (1, -1.40, 0.70, 0.0),
    (1, -0.95, -0.68, 0.52),
    (1, -0.95, -0.68, -0.52),
    (1, 0.97, -0.54, 0.52),
    (1, 0.97, -0.54, -0.52),
    (1, 1.65, 1.15, 0.0),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

tests_passed = 0
tests_failed = 0


def check(name, got, expected):
    global tests_passed, tests_failed
    ok = False
    if callable(expected):
        ok = expected(got)
    elif isinstance(expected, (list, tuple)):
        ok = got in expected
    else:
        ok = got == expected

    if ok:
        print(f"  \u2713 {name}: {got}")
        tests_passed += 1
    else:
        print(f"  \u2717 {name}: got {got!r}, expected {expected!r}")
        tests_failed += 1


def section(label):
    print(f"\n\u2500\u2500 {label} \u2500\u2500")


# ── Test 1: formula_to_name ──────────────────────────────────────────────────

section("formula_to_name (no geometry needed)")
check("H2O", formula_to_name("H2O"), "water")
check("CH4", formula_to_name("CH4"), "methane")
check("CO2", formula_to_name("CO2"), "carbon dioxide")
check("C6H6", formula_to_name("C6H6"), "benzene")
comp = formula_to_name("NaCl")
check(
    "NaCl has 'sodium' or 'chloride'",
    "sodium" in comp or "chloride" in comp.lower(),
    True,
)

# ── Test 2: name_from_atoms ──────────────────────────────────────────────────

section("name_from_atoms (with geometry)")
check("water", name_from_atoms(_WATER_ATOMS), "water")
check("methane", name_from_atoms(_METHANE_ATOMS), "methane")
check("benzene", name_from_atoms(_BENZENE_ATOMS), "benzene")
co2_name = name_from_atoms(_CO2_ATOMS)
check("CO2 contains 'carbon dioxide'", "carbon dioxide" in co2_name, True)

# ── Test 3: Detailed naming ──────────────────────────────────────────────────

section("Detailed naming — NamedResult")
r = name_from_atoms_detailed(_WATER_ATOMS)
check("water name", r.name, "water")
check("water source", r.source, NamingSource.TRIVIAL_IUPAC)
check("water confidence", r.confidence, Confidence.HIGH)
check("water is_trivial", r.is_trivial, True)

r = name_from_atoms_detailed(_ETHANOL_ATOMS)
print(f"  ethanol: name={r.name!r} source={r.source.value} conf={r.confidence.value}")

# ── Test 4: Graph construction ───────────────────────────────────────────────

section("Graph construction")
graph = from_atoms_list(_WATER_ATOMS)
check("water formula", graph.formula, "H2O")
check("water n_atoms", graph.n_atoms, 3)
check("water is_organic", graph.is_organic, False)
check("water has 2 bonds", len(graph.bonds), 2)

graph = from_atoms_list(_BENZENE_ATOMS)
check("benzene formula", graph.formula, "C6H6")
check("benzene has bonds", len(graph.bonds) > 0, True)

# ── Test 5: efiname_from_graph ───────────────────────────────────────────────

section("name_from_graph")
graph = from_atoms_list(_METHANE_ATOMS)
check("methane via graph", name_from_graph(graph), "methane")

# ── Test 6: Ethylene ─────────────────────────────────────────────────────────

section("Ethylene C2H4")
ethylene_atoms = [
    (6, -0.67, 0.0, 0.0),
    (6, 0.67, 0.0, 0.0),
    (1, -0.85, 0.92, 0.0),
    (1, -0.85, -0.92, 0.0),
    (1, 0.85, 0.92, 0.0),
    (1, 0.85, -0.92, 0.0),
]
c2h4_name = name_from_atoms(ethylene_atoms)
check("C2H4 -> ethylene", c2h4_name, ["ethylene", "ethene"])

# ── Test 7: Acetylene ────────────────────────────────────────────────────────

section("Acetylene C2H2")
acetylene_atoms = [
    (6, -0.60, 0.0, 0.0),
    (6, 0.60, 0.0, 0.0),
    (1, -1.06, 0.0, 0.0),
    (1, 1.06, 0.0, 0.0),
]
c2h2_name = name_from_atoms(acetylene_atoms)
check("C2H2 -> acetylene", c2h2_name, ["acetylene", "ethyne"])

# ── Test 8: Ethane ───────────────────────────────────────────────────────────

section("Ethane C2H6")
ethane_atoms = [
    (6, -0.77, 0.0, 0.0),
    (6, 0.77, 0.0, 0.0),
    (1, -1.20, 0.58, 0.0),
    (1, -1.20, -0.29, 0.50),
    (1, -1.20, -0.29, -0.50),
    (1, 1.20, 0.58, 0.0),
    (1, 1.20, -0.29, 0.50),
    (1, 1.20, -0.29, -0.50),
]
c2h6_name = name_from_atoms(ethane_atoms)
check("C2H6 -> ethane", c2h6_name, "ethane")

# ── Test 9: NH3 ──────────────────────────────────────────────────────────────

section("Ammonia NH3")
nh3_atoms = [
    (7, 0.0, 0.0, 0.1163),
    (1, 0.0, 0.9389, -0.2721),
    (1, 0.8131, -0.4694, -0.2721),
    (1, -0.8131, -0.4694, -0.2721),
]
nh3_name = name_from_atoms(nh3_atoms)
check("NH3 -> ammonia", nh3_name, "ammonia")

# ── Summary ──────────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"Results: {tests_passed} passed, {tests_failed} failed")
if tests_failed > 0:
    sys.exit(1)
print("All smoke tests passed ✓")
