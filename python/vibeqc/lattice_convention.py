"""Lattice orientation helpers -- vibe-qc stores lattice vectors as COLUMNS.

``PeriodicSystem.lattice[:, i]`` is the i-th Cartesian lattice vector
``a_i`` in bohr (``cpp/include/vibeqc/periodic.hpp``: "Columns = Cartesian
lattice vectors"; the real-space image sum is ``r_cart = lattice @ index``).
Passing the vectors as ROWS is silently accepted -- a transposed matrix is
still a full-rank lattice -- and produces a sheared, physically different
crystal that converges and reports plausible numbers (GitLab #445: three
independent instances, #319 / #364 / #108, plus a fourth live measurement of
779 Ha/prim on graphene). Nothing geometric can refuse the transpose in
general, because ``L`` and ``L.T`` are both valid cells; only a *declared*
orientation can. This module is the declared-orientation surface:

* :func:`lattice_from_vectors` / the ``lattice_vectors=`` keyword of
  :class:`vibeqc.PeriodicSystem` -- build the matrix from the three vectors
  themselves, so the caller never states a matrix orientation at all.
* :func:`cell_parameters` -- ``|a|, |b|, |c|`` and the enclosed angles as
  the engine consumes them. These ARE orientation-sensitive on a
  non-symmetric lattice, unlike the volume: ``det(L) == det(L.T)``, so a
  volume check is blind to exactly this defect.
* :func:`nearest_neighbour_distance` -- the shortest interatomic distance
  including lattice images, the literature-comparable number a reviewer
  should assert (h-BN: 1.446 Angstrom B-N; a row-fed lattice gave 0.857).

The periodic ``.out`` prints the cell parameters and the nearest-neighbour
distance in its "Cell parameters" block so every job carries the check.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "CellParameters",
    "NearestNeighbour",
    "cell_parameters",
    "lattice_from_vectors",
    "lattice_vectors",
    "nearest_neighbour_distance",
]

_BOHR_TO_ANGSTROM = 0.529177210903


def lattice_from_vectors(
    a1: Sequence[float],
    a2: Sequence[float],
    a3: Sequence[float],
) -> np.ndarray:
    """Return the 3x3 lattice matrix whose COLUMNS are ``a1, a2, a3``.

    This is the unambiguous way to build a ``PeriodicSystem`` lattice from
    three vectors: the caller names the vectors, never a matrix orientation.
    Equivalent to ``np.column_stack([a1, a2, a3])`` with shape validation.
    """
    vectors = []
    for name, vec in (("a1", a1), ("a2", a2), ("a3", a3)):
        arr = np.asarray(vec, dtype=float).reshape(-1)
        if arr.shape != (3,):
            raise ValueError(
                f"lattice_from_vectors: {name} must have 3 Cartesian "
                f"components (bohr); got shape {np.asarray(vec).shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"lattice_from_vectors: {name} has non-finite components: "
                f"{arr.tolist()}"
            )
        vectors.append(arr)
    return np.column_stack(vectors)


def lattice_vectors(lattice: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(a1, a2, a3)`` -- the COLUMNS of a lattice matrix."""
    L = _as_lattice(lattice)
    return (L[:, 0].copy(), L[:, 1].copy(), L[:, 2].copy())


@dataclass(frozen=True)
class CellParameters:
    """Cell lengths (bohr), angles (degrees) and volume (bohr^3).

    ``alpha`` is the angle between ``a2`` and ``a3``, ``beta`` between ``a1``
    and ``a3``, ``gamma`` between ``a1`` and ``a2`` (the crystallographic
    convention). Lengths and angles are orientation-sensitive on a
    non-symmetric lattice; ``volume`` is not (``det(L) == det(L.T)``), which
    is why a volume check cannot catch a transposed lattice (#445).
    """

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    volume: float

    @property
    def lengths(self) -> tuple[float, float, float]:
        return (self.a, self.b, self.c)

    @property
    def angles(self) -> tuple[float, float, float]:
        return (self.alpha, self.beta, self.gamma)


def cell_parameters(lattice_or_system: Any) -> CellParameters:
    """Cell parameters of a lattice matrix or ``PeriodicSystem``.

    The lattice is read in the engine's convention: COLUMNS are the lattice
    vectors. Feed ``lattice.T`` to see what a row-oriented reading would have
    produced; on any non-symmetric cell the two disagree in lengths and
    angles while agreeing on the volume.
    """
    L = _as_lattice(lattice_or_system)
    a1, a2, a3 = L[:, 0], L[:, 1], L[:, 2]
    la, lb, lc = (float(np.linalg.norm(v)) for v in (a1, a2, a3))
    return CellParameters(
        a=la,
        b=lb,
        c=lc,
        alpha=_angle_deg(a2, a3),
        beta=_angle_deg(a1, a3),
        gamma=_angle_deg(a1, a2),
        volume=float(abs(np.linalg.det(L))),
    )


@dataclass(frozen=True)
class NearestNeighbour:
    """Shortest interatomic distance including lattice images (bohr)."""

    distance_bohr: float
    atom_i: int
    atom_j: int
    image: tuple[int, int, int]

    @property
    def distance_angstrom(self) -> float:
        return self.distance_bohr * _BOHR_TO_ANGSTROM


def nearest_neighbour_distance(
    system: Any,
    *,
    image_range: int = 1,
) -> NearestNeighbour:
    """Shortest distance between any two atoms, lattice images included.

    Scans images ``n_i in [-image_range, image_range]`` along the periodic
    axes (``system.dim`` of them; the synthesized axes of a 1-D / 2-D cell
    carry no images). A single atom per cell reports its distance to its own
    nearest image. This is the number to assert against literature when a
    lattice-related defect is suspected: unlike the volume, it moves when
    the lattice is transposed.
    """
    L = _as_lattice(system)
    dim = int(getattr(system, "dim", 3))
    atoms = list(system.unit_cell)
    if not atoms:
        raise ValueError("nearest_neighbour_distance: the cell has no atoms")
    positions = np.asarray([np.asarray(a.xyz, dtype=float) for a in atoms])
    rng = range(-int(image_range), int(image_range) + 1)
    axes = [rng if k < dim else (0,) for k in range(3)]
    best = None
    for n in itertools.product(*axes):
        shift = L @ np.asarray(n, dtype=float)
        for i, j in itertools.product(range(len(atoms)), repeat=2):
            if i == j and n == (0, 0, 0):
                continue
            if i > j and n == (0, 0, 0):
                continue
            d = float(np.linalg.norm(positions[j] + shift - positions[i]))
            if best is None or d < best.distance_bohr:
                best = NearestNeighbour(d, i, j, (int(n[0]), int(n[1]), int(n[2])))
    assert best is not None
    return best


def _as_lattice(lattice_or_system: Any) -> np.ndarray:
    lattice = getattr(lattice_or_system, "lattice", lattice_or_system)
    L = np.asarray(lattice, dtype=float)
    if L.shape != (3, 3):
        raise ValueError(
            f"lattice must be a 3x3 matrix whose columns are the lattice "
            f"vectors (bohr); got shape {L.shape}"
        )
    return L


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return float("nan")
    cosine = float(np.dot(u, v) / (nu * nv))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
