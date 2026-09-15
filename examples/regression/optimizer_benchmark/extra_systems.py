"""Periodic and scaling test systems for the optimizer benchmark.

Adds a size-scaling family (naphthalene → coronene) to the test
system registry.  All geometries are approximate equilibrium structures
in bohr, perturbed so every optimizer must descend.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .test_systems import TestSystem

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_mol(zs, coords, scale=1.06, noise=0.05):
    """Build a perturbed Molecule from Zs and coords (bohr)."""
    import random

    from vibeqc import Atom, Molecule

    random.seed(42)
    atoms = []
    for z, (x, y, zz) in zip(zs, coords):
        nx = x * scale + (random.uniform(-noise, noise) if noise else 0)
        ny = y * scale + (random.uniform(-noise, noise) if noise else 0)
        nz = zz * scale + (random.uniform(-noise, noise) if noise else 0)
        atoms.append(Atom(z, [nx, ny, nz]))
    random.seed()
    return Molecule(atoms)


# ---------------------------------------------------------------------------
# Naphthalene C10H8 — two fused benzene rings
# ---------------------------------------------------------------------------

_NAPHTHALENE_ZS = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]
_NAPHTHALENE_COORDS = [
    # Ring 1 (left)
    (0.000, 2.300, 0.0),  # C1
    (1.350, 1.500, 0.0),  # C2
    (1.350, 0.000, 0.0),  # C3
    (0.000, -0.800, 0.0),  # C4 (shared)
    (-1.350, 0.000, 0.0),  # C5
    (-1.350, 1.500, 0.0),  # C6
    # Ring 2 (right)
    (2.700, -0.800, 0.0),  # C7
    (2.700, -2.300, 0.0),  # C8
    (1.350, -3.100, 0.0),  # C9
    (0.000, -2.300, 0.0),  # C10 (shared)
    # Hydrogens
    (-0.050, 3.380, 0.0),
    (2.050, 2.080, 0.0),
    (2.370, 0.280, 0.0),
    (-2.370, 0.280, 0.0),
    (-2.050, 2.080, 0.0),
    (3.730, -0.520, 0.0),
    (3.680, -3.060, 0.0),
    (1.350, -4.190, 0.0),
]


def _naphthalene():
    return TestSystem(
        "naphthalene",
        "scaling",
        _make_mol(_NAPHTHALENE_ZS, _NAPHTHALENE_COORDS),
        "C\u2081\u2080H\u2088 \u2014 naphthalene (2 fused rings)",
    )


# ---------------------------------------------------------------------------
# Anthracene C14H10 — three fused rings
# ---------------------------------------------------------------------------

_ANTHRACENE_ZS = [6] * 14 + [1] * 10
_ANTHRACENE_COORDS = [
    # Left ring
    (0.000, 2.300, 0.0),
    (1.350, 1.500, 0.0),
    (1.350, 0.000, 0.0),
    (0.000, -0.800, 0.0),
    (-1.350, 0.000, 0.0),
    (-1.350, 1.500, 0.0),
    # Middle ring
    (2.700, -0.800, 0.0),
    (2.700, -2.300, 0.0),
    (1.350, -3.100, 0.0),
    (0.000, -2.300, 0.0),
    # Right ring
    (4.050, -3.100, 0.0),
    (4.050, -4.600, 0.0),
    (2.700, -5.400, 0.0),
    (1.350, -4.600, 0.0),
    # Hydrogens
    (-0.050, 3.380, 0.0),
    (2.050, 2.080, 0.0),
    (2.370, 0.280, 0.0),
    (-2.370, 0.280, 0.0),
    (-2.050, 2.080, 0.0),
    (3.730, -0.520, 0.0),
    (3.680, -3.060, 0.0),
    (5.120, -2.750, 0.0),
    (5.030, -5.370, 0.0),
    (2.700, -6.490, 0.0),
]


def _anthracene():
    return TestSystem(
        "anthracene",
        "scaling",
        _make_mol(_ANTHRACENE_ZS, _ANTHRACENE_COORDS),
        "C\u2081\u2084H\u2081\u2080 \u2014 anthracene (3 fused rings)",
    )


# ---------------------------------------------------------------------------
# Tetracene C18H12 — four fused rings
# ---------------------------------------------------------------------------

_TETRACENE_ZS = [6] * 18 + [1] * 12
_TETRACENE_COORDS = [
    # Rings 1-3 (same as anthracene)
    (0.000, 2.300, 0.0),
    (1.350, 1.500, 0.0),
    (1.350, 0.000, 0.0),
    (0.000, -0.800, 0.0),
    (-1.350, 0.000, 0.0),
    (-1.350, 1.500, 0.0),
    (2.700, -0.800, 0.0),
    (2.700, -2.300, 0.0),
    (1.350, -3.100, 0.0),
    (0.000, -2.300, 0.0),
    (4.050, -3.100, 0.0),
    (4.050, -4.600, 0.0),
    (2.700, -5.400, 0.0),
    (1.350, -4.600, 0.0),
    # Ring 4
    (5.400, -5.400, 0.0),
    (5.400, -6.900, 0.0),
    (4.050, -7.700, 0.0),
    (2.700, -6.900, 0.0),
    # Hydrogens
    (-0.050, 3.380, 0.0),
    (2.050, 2.080, 0.0),
    (2.370, 0.280, 0.0),
    (-2.370, 0.280, 0.0),
    (-2.050, 2.080, 0.0),
    (3.730, -0.520, 0.0),
    (3.680, -3.060, 0.0),
    (5.120, -2.750, 0.0),
    (5.030, -5.370, 0.0),
    (6.480, -5.050, 0.0),
    (6.380, -7.660, 0.0),
    (4.050, -8.790, 0.0),
]


def _tetracene():
    return TestSystem(
        "tetracene",
        "scaling",
        _make_mol(_TETRACENE_ZS, _TETRACENE_COORDS),
        "C\u2081\u2088H\u2081\u2082 \u2014 tetracene (4 fused rings)",
    )


# ---------------------------------------------------------------------------
# Coronene C24H12 — circumacene, 7 fused rings in hexagonal arrangement
# ---------------------------------------------------------------------------


def _coronene():
    """Coronene — hexagonal superbenzene.  Approximate planar geometry."""
    r = 2.639  # C-C bond in bohr
    r_ch = 2.048
    zs = []
    coords = []
    # Central hexagon (6 C)
    for i in range(6):
        a = i * np.pi / 3
        zs.append(6)
        coords.append((r * np.cos(a), r * np.sin(a), 0.0))
    # Inner bridge C (6 atoms at radius 2r)
    for i in range(6):
        a = i * np.pi / 3
        zs.append(6)
        coords.append((2.0 * r * np.cos(a), 2.0 * r * np.sin(a), 0.0))
    # Outer edge: 12 C + 12 H
    for i in range(6):
        a1 = (i + 0.5) * np.pi / 3
        zs += [6, 1]
        coords.append((2.5 * r * np.cos(a1), 2.5 * r * np.sin(a1), 0.0))
        coords.append(
            ((2.5 * r + r_ch) * np.cos(a1), (2.5 * r + r_ch) * np.sin(a1), 0.0)
        )
        a2 = i * np.pi / 3 + np.pi / 6
        zs += [6, 1]
        coords.append((2.5 * r * np.cos(a2), 2.5 * r * np.sin(a2), 0.0))
        coords.append(
            ((2.5 * r + r_ch) * np.cos(a2), (2.5 * r + r_ch) * np.sin(a2), 0.0)
        )

    return TestSystem(
        "coronene",
        "scaling",
        _make_mol(zs, coords, scale=1.06, noise=0.04),
        "C\u2082\u2084H\u2081\u2082 \u2014 coronene (7 fused rings)",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_EXTRA_SYSTEMS: dict[str, callable] = {}


def _register_extra():
    global _EXTRA_SYSTEMS
    if _EXTRA_SYSTEMS:
        return
    _EXTRA_SYSTEMS = {
        "naphthalene": _naphthalene,
        "anthracene": _anthracene,
        "tetracene": _tetracene,
        "coronene": _coronene,
    }


def get_extra_system(name: str) -> TestSystem:
    _register_extra()
    if name not in _EXTRA_SYSTEMS:
        available = ", ".join(sorted(_EXTRA_SYSTEMS))
        raise ValueError(f"Unknown extra system {name!r}. Available: {available}")
    return _EXTRA_SYSTEMS[name]()


def list_extra_systems() -> list[str]:
    _register_extra()
    return sorted(_EXTRA_SYSTEMS)
