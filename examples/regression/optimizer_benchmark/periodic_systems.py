"""Periodic test systems for the optimizer benchmark.

Adds periodic solids (bulk Si, MgO, NaCl) with perturbed atom
positions to the benchmark suite.  Uses :class:`vibeqc.PeriodicSystem`
with primitive unit cells in bohr.

Each system has its lattice parameters and atom positions perturbed
from equilibrium so the optimizer must descend.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class PeriodicTestSystem:
    """A periodic benchmark test system.

    Attributes
    ----------
    name : str
    category : str
        ``"periodic_covalent"``, ``"periodic_ionic"``.
    description : str
    system : PeriodicSystem
        The perturbed starting geometry.
    kmesh_density : float
        k-point mesh density (Å), e.g. 3.0 → ~3 Å spacing.
    basis : str
        Suggested basis set.
    n_atoms : int
    charge : int
    multiplicity : int
    """

    name: str
    category: str
    description: str
    system: "PeriodicSystem"
    kmesh_density: float = 3.0
    basis: str = "sto-3g"
    charge: int = 0
    multiplicity: int = 1

    @property
    def n_atoms(self) -> int:
        return len(self.system.unit_cell)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_periodic(
    dim: int,
    lattice_vecs: list[list[float]],  # columns in bohr
    zs: list[int],
    frac_coords: list[tuple[float, float, float]],
    charge: int = 0,
    multiplicity: int = 1,
    noise_frac: float = 0.0,
) -> "PeriodicSystem":
    """Build a PeriodicSystem with optional fractional-coordinate noise.

    ``lattice_vecs`` are columns of the 3×3 lattice matrix in bohr.
    ``frac_coords`` are fractional coordinates (0..1).
    ``noise_frac`` adds random perturbation in fractional coords.
    """
    from vibeqc._vibeqc_core import Atom, PeriodicSystem

    random.seed(42)
    lattice = np.array(lattice_vecs, dtype=float)
    atoms = []
    for z, (fx, fy, fz) in zip(zs, frac_coords):
        if noise_frac > 0:
            fx += random.uniform(-noise_frac, noise_frac)
            fy += random.uniform(-noise_frac, noise_frac)
            fz += random.uniform(-noise_frac, noise_frac)
        # Convert fractional → Cartesian
        cart = lattice @ np.array([fx, fy, fz])
        atoms.append(Atom(z, list(cart)))
    random.seed()
    return PeriodicSystem(dim, lattice.tolist(), atoms, charge, multiplicity)


# ---------------------------------------------------------------------------
# Si bulk — diamond structure, FCC with 2-atom basis
# ---------------------------------------------------------------------------


def _si_bulk() -> PeriodicTestSystem:
    """Si bulk — diamond, 2 atoms/primitive cell.

    Lattice constant 10.263 bohr (5.431 Å).  Atoms at (0,0,0)
    and (1/4,1/4,1/4).  Perturb: noise ±0.005 in fractional coords.
    """
    a = 10.263  # bohr
    lattice = [[a / 2, a / 2, 0.0], [a / 2, 0.0, a / 2], [0.0, a / 2, a / 2]]
    zs = [14, 14]
    frac = [(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)]
    system = _make_periodic(3, lattice, zs, frac, noise_frac=0.005)
    return PeriodicTestSystem(
        "si_bulk",
        "periodic_covalent",
        "Si — diamond, 2 atoms/cell, a=5.431 Å",
        system,
        kmesh_density=4.0,
        basis="sto-3g",
    )


# ---------------------------------------------------------------------------
# MgO bulk — rocksalt structure
# ---------------------------------------------------------------------------


def _mgo_bulk() -> PeriodicTestSystem:
    """MgO bulk — rocksalt, 2 atoms/primitive cell.

    Lattice constant 7.960 bohr (4.212 Å).  Mg at (0,0,0),
    O at (1/2,1/2,1/2).
    """
    a = 7.960  # bohr
    lattice = [[a / 2, a / 2, 0.0], [a / 2, 0.0, a / 2], [0.0, a / 2, a / 2]]
    zs = [12, 8]  # Mg, O
    frac = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
    system = _make_periodic(3, lattice, zs, frac, noise_frac=0.003)
    return PeriodicTestSystem(
        "mgo_bulk",
        "periodic_ionic",
        "MgO — rocksalt, 2 atoms/cell, a=4.212 Å",
        system,
        kmesh_density=4.0,
        basis="sto-3g",
    )


# ---------------------------------------------------------------------------
# NaCl bulk — rocksalt structure
# ---------------------------------------------------------------------------


def _nacl_bulk() -> PeriodicTestSystem:
    """NaCl bulk — rocksalt, 2 atoms/primitive cell.

    Lattice constant 10.630 bohr (5.626 Å).
    """
    a = 10.630  # bohr
    lattice = [[a / 2, a / 2, 0.0], [a / 2, 0.0, a / 2], [0.0, a / 2, a / 2]]
    zs = [11, 17]  # Na, Cl
    frac = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
    system = _make_periodic(3, lattice, zs, frac, noise_frac=0.003)
    return PeriodicTestSystem(
        "nacl_bulk",
        "periodic_ionic",
        "NaCl — rocksalt, 2 atoms/cell, a=5.626 Å",
        system,
        kmesh_density=4.0,
        basis="sto-3g",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PERIODIC_SYSTEMS: dict[str, callable] = {}


def _register_periodic():
    global _PERIODIC_SYSTEMS
    if _PERIODIC_SYSTEMS:
        return
    _PERIODIC_SYSTEMS = {
        "si_bulk": _si_bulk,
        "mgo_bulk": _mgo_bulk,
        "nacl_bulk": _nacl_bulk,
    }


def get_periodic_system(name: str) -> PeriodicTestSystem:
    _register_periodic()
    if name not in _PERIODIC_SYSTEMS:
        available = ", ".join(sorted(_PERIODIC_SYSTEMS))
        raise ValueError(f"Unknown periodic system {name!r}. Available: {available}")
    return _PERIODIC_SYSTEMS[name]()


def list_periodic_systems() -> list[str]:
    _register_periodic()
    return sorted(_PERIODIC_SYSTEMS)
