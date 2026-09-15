#!/usr/bin/env python3
"""Smoke-test the IUPAC naming module through the vibeqc.naming shim.

Run from vibeqc-zed root:
    python3 python/vibeqc/naming/_smoke_test.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "python")

# Import through the shim — this exercises the re-export path.
# If vibeqc C++ core is not built, this will fail with ImportError;
# use ``vibeqc_naming._smoke_test`` instead for standalone testing.
from vibeqc.naming import (
    Confidence,
    NamingSource,
    from_atoms_list,
    name_from_atoms,
    name_from_atoms_detailed,
    name_from_graph,
)

print("Imports OK ✓")

_WATER = [
    (8, 0.0, 0.0, 0.1173),
    (1, 0.0, 0.7572, -0.4692),
    (1, 0.0, -0.7572, -0.4692),
]

_METHANE = [
    (6, 0.0, 0.0, 0.0),
    (1, 0.6276, 0.6276, 0.6276),
    (1, 0.6276, -0.6276, -0.6276),
    (1, -0.6276, 0.6276, -0.6276),
    (1, -0.6276, -0.6276, 0.6276),
]

r = name_from_atoms_detailed(_WATER)
assert r.name == "water", r.name
assert r.source == NamingSource.TRIVIAL_IUPAC
assert r.confidence == Confidence.HIGH
assert r.is_trivial
print(f"  water: {r} ✓")

m = name_from_atoms(_METHANE)
assert m == "methane", m
print(f"  methane: {m} ✓")

print("\nAll shim tests passed ✓")
