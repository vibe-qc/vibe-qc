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

    Exact for any full-rank cell. A single atom per cell reports its distance
    to its own nearest image; only the ``system.dim`` periodic axes carry
    images, so the synthesized axes of a 1-D / 2-D cell never translate. This
    is the number to assert against literature when a lattice-related defect
    is suspected: unlike the volume, it moves when the lattice is transposed.

    GitLab #128: this used to scan a fixed ``+-image_range`` shell over the
    basis exactly as supplied, which is not sufficient for an arbitrary
    basis. The shortest translation of a sheared cell needs coefficients
    larger than one -- for the 2-D cell ``a1 = (4,0,0)``, ``a2 = (7,1,0)`` it
    is ``-2*a1 + a2`` -- so the default search returned 3.162278 bohr where
    the true shortest image distance is 1.414214, and the periodic ``.out``
    printed that wrong number as its nearest-pair line. The verifier that
    rejected the original fix asked for a provably sufficient search rather
    than a wider fixed shell, so the basis is now Minkowski-reduced first
    (Nguyen and Stehle, "Low-dimensional lattice basis reduction revisited",
    ACM Trans. Algorithms 5, 46 (2009); via ``ase.geometry.minkowski_reduce``,
    the same routine :mod:`vibeqc.periodic.ccm.wigner_seitz` uses). In a
    reduced basis the closest image lies within one cell of the rounded
    fractional coordinate for dimension <= 3, so a small fixed shell is
    exact, and a hit on the rim of that shell raises instead of being
    returned.

    ``image_range`` is retained for compatibility and is now advisory: it can
    only widen the guard margin, never narrow the search below what
    exactness requires. Honouring a narrowing request is what produced the
    wrong answer.
    """
    L = _as_lattice(system)
    dim = int(getattr(system, "dim", 3))
    atoms = list(system.unit_cell)
    if not atoms:
        raise ValueError("nearest_neighbour_distance: the cell has no atoms")
    positions = np.asarray([np.asarray(a.xyz, dtype=float) for a in atoms])
    n_atoms = len(atoms)

    if dim <= 0:
        return _closest_pair_without_images(positions, n_atoms)

    basis, to_original = _reduced_periodic_basis(L, dim)
    # +-1 is provably sufficient after reduction; the extra ring is margin so
    # a legitimate winner never sits on the rim the guard below checks.
    search = max(_MIN_IMAGE_SEARCH, int(image_range))
    offsets = np.asarray(
        list(itertools.product(range(-search, search + 1), repeat=dim)),
        dtype=float,
    )
    pinv_basis = np.linalg.pinv(basis)

    best: NearestNeighbour | None = None
    best_offset = None
    for i in range(n_atoms):
        for j in range(i, n_atoms):
            d0 = positions[j] - positions[i]
            # Coefficients that most nearly cancel d0, then the shell around
            # them.  ``pinv`` ignores any component outside the periodic
            # span, which no translation can reach anyway.
            centre = np.rint(-(d0 @ pinv_basis))
            coeffs = centre + offsets
            if i == j:
                # An atom is not its own neighbour: drop the identity image.
                coeffs = coeffs[np.any(coeffs != 0.0, axis=1)]
            distances = np.linalg.norm(d0 + coeffs @ basis, axis=1)
            k = int(np.argmin(distances))
            d = float(distances[k])
            if best is None or d < best.distance_bohr:
                image = _to_original_image(coeffs[k], dim, to_original)
                best = NearestNeighbour(d, i, j, image)
                best_offset = coeffs[k] - centre

    assert best is not None and best_offset is not None
    if int(np.max(np.abs(best_offset))) >= search:
        raise RuntimeError(
            "nearest_neighbour_distance: the closest image landed on the rim "
            f"of the {search}-cell search shell, so the search cannot be "
            "shown to be exhaustive. This should be unreachable after "
            "Minkowski reduction; please report the lattice on GitLab #128."
        )
    return best


# Half-width of the reduced-basis image shell.  Minkowski reduction bounds the
# closest image to one cell from the rounded fractional coordinate in
# dimension <= 3; the second ring is margin so the rim guard above fires only
# on a genuine reduction failure.
_MIN_IMAGE_SEARCH = 2


def _reduced_periodic_basis(
    L: np.ndarray, dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Minkowski-reduced periodic lattice vectors and the map back.

    Returns ``(basis, to_original)`` where ``basis`` holds the ``dim``
    reduced lattice vectors as ROWS and ``to_original`` is the integer
    unimodular matrix with ``basis_row_k = sum_m to_original[k, m] a_m``, so
    a reduced-basis coefficient vector maps to the caller's basis by
    ``n @ to_original``. Reduction touches only the periodic axes.
    """
    try:
        from ase.geometry import minkowski_reduce
    except ImportError as exc:  # pragma: no cover - ASE is a base dependency
        raise ImportError(
            "nearest_neighbour_distance needs ASE (a base runtime dependency) "
            "for the exact minimum-image search"
        ) from exc

    cell_rows = np.ascontiguousarray(L.T, dtype=float)
    pbc = np.asarray([k < dim for k in range(3)], dtype=bool)
    reduced, op = minkowski_reduce(cell_rows, pbc=pbc)
    return np.asarray(reduced, dtype=float)[:dim], np.asarray(op, dtype=int)


def _to_original_image(
    coeffs: np.ndarray, dim: int, to_original: np.ndarray
) -> tuple[int, int, int]:
    """Reduced-basis coefficients as an image in the caller's own basis.

    ``NearestNeighbour.image`` is documented in the caller's coefficients and
    the periodic ``.out`` prints it, so it must not leak the reduced basis.
    """
    padded = np.zeros(3, dtype=int)
    padded[:dim] = np.rint(coeffs).astype(int)
    image = padded @ to_original
    if np.any(image[dim:] != 0):  # pragma: no cover - unimodular block form
        raise RuntimeError(
            "nearest_neighbour_distance: the reduction mixed a periodic axis "
            "into a non-periodic one"
        )
    return (int(image[0]), int(image[1]), int(image[2]))


def _closest_pair_without_images(
    positions: np.ndarray, n_atoms: int
) -> NearestNeighbour:
    """Closest pair of a cell with no periodic axes."""
    if n_atoms < 2:
        raise ValueError(
            "nearest_neighbour_distance: a non-periodic cell needs at least "
            "two atoms"
        )
    best: NearestNeighbour | None = None
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d = float(np.linalg.norm(positions[j] - positions[i]))
            if best is None or d < best.distance_bohr:
                best = NearestNeighbour(d, i, j, (0, 0, 0))
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
