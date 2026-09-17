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
    "CRYSTAL_SYSTEMS",
    "PLANE_LATTICE_SYSTEMS",
    "CellParameters",
    "LatticeDeclarationError",
    "NearestNeighbour",
    "cell_parameters",
    "check_crystal_system",
    "check_space_group",
    "lattice_from_vectors",
    "lattice_vectors",
    "nearest_neighbour_distance",
]

_BOHR_TO_ANGSTROM = 0.529177210903


def lattice_from_vectors(
    a1: Sequence[float],
    a2: Sequence[float],
    a3: Sequence[float],
    *,
    crystal_system: str | None = None,
    dim: int = 3,
) -> np.ndarray:
    """Return the 3x3 lattice matrix whose COLUMNS are ``a1, a2, a3``.

    This is the unambiguous way to build a ``PeriodicSystem`` lattice from
    three vectors: the caller names the vectors, never a matrix orientation.
    Equivalent to ``np.column_stack([a1, a2, a3])`` with shape validation.

    Pass ``crystal_system=`` to declare what the cell is meant to be and have
    a contradicting lattice refused (GitLab #128); ``dim`` says how many axes
    are physical, so a sheet declares a plane lattice. See
    :func:`check_crystal_system` for what a declaration does and does not
    catch.
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
    lattice = np.column_stack(vectors)
    if crystal_system is not None:
        check_crystal_system(lattice, crystal_system, dim=dim)
    return lattice


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


# ---------------------------------------------------------------------------
# Declared crystal system (GitLab #128, ask 1)
#
# ``cell_parameters`` reports what a lattice IS; this section refuses a lattice
# that contradicts what the caller SAID it is. Nothing geometric can reject a
# transpose on its own -- ``L`` and ``L.T`` are both valid lattices -- so a
# declaration is the only thing that can.
#
# What is checked is the LATTICE (Bravais) system, because a lattice matrix
# cannot determine more than that: the crystal system is fixed by the point
# group, which the basis lowers and the metric does not see. An exactly
# hexagonal cell whose basis breaks the symmetry is still a hexagonal
# *lattice*, and spglib will still call its space group monoclinic. The
# keyword is spelled ``crystal_system`` because that is what a caller reaches
# for, and "trigonal" is accepted as an alias for "hexagonal or rhombohedral",
# which is exactly the ambiguity the trigonal crystal system carries: it has no
# lattice of its own (International Tables for Crystallography Vol. A,
# section 3.1; Hahn ed., 5th ed. 2002, Table 2.1.2.1).
#
# Conditions are EQUALITIES ONLY. The familiar "a != c" of a tetragonal cell
# and "beta != 90" of a monoclinic one are conventions for *choosing* a cell,
# not requirements on one, so they are not enforced: a cubic cell declared
# orthorhombic is not an error, and refusing it would be. The conditions are
# therefore necessary and not sufficient, and a higher-symmetry cell passes a
# lower declaration.
#
# HOW MUCH OF THE TRANSPOSE HAZARD THIS ACTUALLY CATCHES -- stated here
# because overselling it would be worse than not having it:
#
#   * hexagonal and rhombohedral: caught. The four instances in #128 were all
#     hexagonal sheets, and the transposed h-BN cell misses a = b by 22.5% and
#     the nearer of {60, 120} degrees by 3.43.
#   * cubic, tetragonal, orthorhombic: NOT an orientation check. Written in the
#     usual Cartesian setting their matrices are symmetric, so the transpose is
#     a no-op; and a cubic metric is transpose-invariant in any frame at all
#     (G = a^2 I makes L/a orthogonal, hence normal). Declaring them still
#     catches a mistyped angle or a wrong c/a, which is worth having, but it is
#     not orientation validation.
#   * monoclinic and triclinic: the transpose changes the crystal but stays
#     inside the same system, so the declaration passes. Compare the cell
#     parameters against known values for those.
#
# The volume cannot help, and neither can the trace: G = L^T L and G' = L L^T
# are similar, so every metric eigenvalue -- hence det G and tr G -- survives a
# transpose.
# ---------------------------------------------------------------------------

#: Declarations accepted for a 3-D cell. "trigonal" is an alias admitting
#: either lattice a trigonal space group can sit on.
CRYSTAL_SYSTEMS: tuple[str, ...] = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "rhombohedral",
    "trigonal",
    "hexagonal",
    "cubic",
)

#: Declarations accepted for a 2-D cell: the five plane lattices. Only
#: ``|a1|``, ``|a2|`` and the angle between them are physical for ``dim=2``;
#: the third axis is synthesized bookkeeping.
PLANE_LATTICE_SYSTEMS: tuple[str, ...] = (
    "oblique",
    "rectangular",
    "centred-rectangular",
    "square",
    "hexagonal",
)

# A 3-D name a 2-D caller may reasonably use for the same plane lattice.
_PLANE_ALIASES = {
    "monoclinic": "oblique",
    "orthorhombic": "rectangular",
    "tetragonal": "square",
    "centered-rectangular": "centred-rectangular",
}

# Lengths: rounding in the input is absolute, so a purely relative tolerance
# is too tight on a small cell. Three-decimal-Angstrom CIF input can put
# |a - b| / a at 4.4e-4 on a cell that is exactly hexagonal by construction.
_LENGTH_RTOL = 1.0e-4
_LENGTH_ATOL_BOHR = 5.0e-3
# Angles: an absolute tolerance in degrees. The transposed h-BN cell misses by
# 3.43 degrees, so this is not a close call in the case that matters.
_ANGLE_ATOL_DEG = 1.0e-2

# The only angles a cubic lattice's cell can present: the conventional cube,
# the face-centred primitive rhombohedron, and the body-centred one.
_CUBIC_SETTING_ANGLES_DEG = (90.0, 60.0, math.degrees(math.acos(-1.0 / 3.0)))


class LatticeDeclarationError(ValueError):
    """A lattice contradicts the crystal system its caller declared."""


def check_crystal_system(
    lattice_or_system: Any,
    crystal_system: str,
    *,
    dim: int | None = None,
    length_rtol: float = _LENGTH_RTOL,
    length_atol_bohr: float = _LENGTH_ATOL_BOHR,
    angle_atol_deg: float = _ANGLE_ATOL_DEG,
) -> CellParameters:
    """Refuse a lattice whose metric contradicts ``crystal_system``.

    Returns the cell parameters when the declaration holds, and raises
    :class:`LatticeDeclarationError` naming the constraint that failed when it
    does not. This is the declared-intent half of GitLab #128: the engine
    reads lattice vectors as COLUMNS, a row-oriented matrix is silently
    accepted as a different sheared crystal, and no geometric test can refuse
    that in general because both readings are valid lattices. A declaration
    can.

    ``dim`` defaults to ``system.dim`` when a ``PeriodicSystem`` is passed and
    to 3 for a bare matrix. Only the ``dim`` periodic axes are constrained,
    mirroring what the periodic ``.out`` reports: a 2-D sheet's third axis is
    synthesized vacuum and carries no physical length or angle, so declaring a
    plane lattice constrains ``|a1|``, ``|a2|`` and the angle between them and
    nothing else.

    Read the module section above before relying on this as an orientation
    check: it catches a transposed hexagonal or rhombohedral cell, which is
    where every instance in #128 arose, but a cubic, tetragonal or
    orthorhombic declaration is not an orientation test at all, and a
    transposed monoclinic or triclinic cell stays inside its own system.

    Conditions follow the lattice (Bravais) systems of International Tables
    for Crystallography Vol. A section 3.1, as equalities only.
    """
    params = cell_parameters(lattice_or_system)
    resolved_dim = _resolve_dim(lattice_or_system, dim)
    declared = _normalise_declaration(crystal_system, resolved_dim)

    failures = _declaration_failures(
        declared,
        params,
        resolved_dim,
        length_rtol,
        length_atol_bohr,
        angle_atol_deg,
    )
    if not failures:
        return params

    raise LatticeDeclarationError(
        _declaration_message(
            lattice_or_system,
            crystal_system,
            declared,
            params,
            resolved_dim,
            failures,
            length_rtol,
            length_atol_bohr,
            angle_atol_deg,
        )
    )


def _resolve_dim(lattice_or_system: Any, dim: int | None) -> int:
    if dim is None:
        dim = int(getattr(lattice_or_system, "dim", 3))
    if dim not in (1, 2, 3):
        raise ValueError(f"dim must be 1, 2 or 3; got {dim}")
    return dim


def _normalise_declaration(crystal_system: str, dim: int) -> str:
    if not isinstance(crystal_system, str):
        raise TypeError(
            "crystal_system must be a string naming a lattice system; got "
            f"{type(crystal_system).__name__}"
        )
    name = crystal_system.strip().lower().replace("_", "-")

    if dim == 1:
        # Only |a1| exists; every system condition is a statement about axes
        # this cell does not have, so any declaration would validate nothing.
        raise ValueError(
            "check_crystal_system: a crystal system is not meaningful for a "
            "1-D cell -- only |a1| is physical, so no metric condition can be "
            "tested. Compare |a1| against its expected value instead."
        )

    if dim == 2:
        name = _PLANE_ALIASES.get(name, name)
        if name not in PLANE_LATTICE_SYSTEMS:
            if name in CRYSTAL_SYSTEMS:
                raise ValueError(
                    f"check_crystal_system: {crystal_system!r} is a 3-D "
                    "lattice system and cannot describe a 2-D cell. The plane "
                    "lattices are " + ", ".join(PLANE_LATTICE_SYSTEMS) + "."
                )
            raise ValueError(
                f"check_crystal_system: unknown plane lattice "
                f"{crystal_system!r}; expected one of "
                + ", ".join(PLANE_LATTICE_SYSTEMS)
            )
        return name

    if name == "anorthic":
        name = "triclinic"
    if name not in CRYSTAL_SYSTEMS:
        raise ValueError(
            f"check_crystal_system: unknown crystal system "
            f"{crystal_system!r}; expected one of " + ", ".join(CRYSTAL_SYSTEMS)
        )
    return name


def _declaration_failures(
    declared: str,
    params: CellParameters,
    dim: int,
    length_rtol: float,
    length_atol_bohr: float,
    angle_atol_deg: float,
) -> list[str]:
    """Unmet equalities for ``declared``; empty means the declaration holds."""

    def leq(x: float, y: float) -> bool:
        return abs(x - y) <= length_rtol * max(abs(x), abs(y)) + length_atol_bohr

    def aeq(t: float, u: float) -> bool:
        return abs(t - u) <= angle_atol_deg

    a, b, c = params.a, params.b, params.c
    alpha, beta, gamma = params.alpha, params.beta, params.gamma
    out: list[str] = []

    if dim == 2:
        # Only a, b and gamma are physical.
        if declared in ("centred-rectangular", "square", "hexagonal"):
            if not leq(a, b):
                out.append(f"|a1| = |a2|  (got {a:.6f} and {b:.6f} bohr)")
        if declared in ("rectangular", "square"):
            if not aeq(gamma, 90.0):
                out.append(f"gamma = 90 deg  (got {gamma:.4f})")
        if declared == "hexagonal":
            if not (aeq(gamma, 120.0) or aeq(gamma, 60.0)):
                out.append(f"gamma = 120 or 60 deg  (got {gamma:.4f})")
        return out

    if declared == "triclinic":
        return out

    if declared == "monoclinic":
        # ITA's standard setting is unique-axis b, but unique-axis c is
        # tabulated alongside it and is common in older literature, so a bare
        # declaration admits any single unique axis.
        unique_b = aeq(alpha, 90.0) and aeq(gamma, 90.0)
        unique_c = aeq(alpha, 90.0) and aeq(beta, 90.0)
        unique_a = aeq(beta, 90.0) and aeq(gamma, 90.0)
        if not (unique_b or unique_c or unique_a):
            out.append(
                "two of alpha, beta, gamma = 90 deg (one unique axis)  "
                f"(got {alpha:.4f}, {beta:.4f}, {gamma:.4f})"
            )
        return out

    if declared == "trigonal":
        # Trigonal has no lattice of its own: a trigonal space group sits on
        # either a hexagonal or a rhombohedral lattice.
        hexagonal = not _declaration_failures(
            "hexagonal", params, dim, length_rtol, length_atol_bohr,
            angle_atol_deg,
        )
        rhombohedral = not _declaration_failures(
            "rhombohedral", params, dim, length_rtol, length_atol_bohr,
            angle_atol_deg,
        )
        if not (hexagonal or rhombohedral):
            out.append(
                "the hexagonal metric (|a1| = |a2|, alpha = beta = 90 deg, "
                "gamma = 120 or 60 deg) or the rhombohedral one "
                "(|a1| = |a2| = |a3|, alpha = beta = gamma)  "
                f"(got lengths {a:.6f}, {b:.6f}, {c:.6f} bohr and angles "
                f"{alpha:.4f}, {beta:.4f}, {gamma:.4f} deg)"
            )
        return out

    if declared in ("tetragonal", "hexagonal"):
        if not leq(a, b):
            out.append(f"|a1| = |a2|  (got {a:.6f} and {b:.6f} bohr)")
    if declared in ("rhombohedral", "cubic"):
        if not (leq(a, b) and leq(b, c)):
            out.append(
                f"|a1| = |a2| = |a3|  (got {a:.6f}, {b:.6f}, {c:.6f} bohr)"
            )

    if declared in ("orthorhombic", "tetragonal"):
        for name, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if not aeq(value, 90.0):
                out.append(f"{name} = 90 deg  (got {value:.4f})")
    elif declared == "cubic":
        # A cubic lattice does not only come as the conventional 90-degree
        # cell: the primitive cell of cF is a 60-degree rhombohedron and that
        # of cI a 109.4712-degree one (arccos(-1/3)). Refusing those would
        # reject the most common real input -- a primitive cell straight from
        # Materials Project -- so all three settings are admitted. Requiring
        # the common angle to be one of exactly those three still rejects a
        # general rhombohedral cell, which equal angles alone would not.
        if not (aeq(alpha, beta) and aeq(beta, gamma)):
            out.append(
                "alpha = beta = gamma  (got "
                f"{alpha:.4f}, {beta:.4f}, {gamma:.4f} deg)"
            )
        elif not any(aeq(alpha, s) for s in _CUBIC_SETTING_ANGLES_DEG):
            out.append(
                "alpha = beta = gamma = 90 deg (conventional), 60 deg "
                "(face-centred primitive) or 109.4712 deg (body-centred "
                f"primitive)  (got {alpha:.4f})"
            )
    elif declared == "hexagonal":
        for name, value in (("alpha", alpha), ("beta", beta)):
            if not aeq(value, 90.0):
                out.append(f"{name} = 90 deg  (got {value:.4f})")
        # gamma = 60 and gamma = 120 with |a1| = |a2| generate the SAME
        # lattice (the change of basis is unimodular), and this repository's
        # own h-BN and graphene references use 60, so both are accepted.
        if not (aeq(gamma, 120.0) or aeq(gamma, 60.0)):
            out.append(f"gamma = 120 or 60 deg  (got {gamma:.4f})")
    elif declared == "rhombohedral":
        if not (aeq(alpha, beta) and aeq(beta, gamma)):
            out.append(
                "alpha = beta = gamma  (got "
                f"{alpha:.4f}, {beta:.4f}, {gamma:.4f} deg)"
            )

    return out


def _declaration_message(
    lattice_or_system: Any,
    spelled: str,
    declared: str,
    params: CellParameters,
    dim: int,
    failures: list[str],
    length_rtol: float,
    length_atol_bohr: float,
    angle_atol_deg: float,
) -> str:
    ang = _BOHR_TO_ANGSTROM
    lines = [
        f"lattice contradicts the declared crystal system {spelled!r}"
        + (f" (read as {declared!r})" if declared != spelled.strip().lower() else "")
        + f" for a {dim}-D cell.",
        "",
        "Unmet condition(s):",
    ]
    lines += [f"  - {f}" for f in failures]
    lines += [
        "",
        "Measured, reading the lattice vectors as COLUMNS:",
        f"  |a1| = {params.a:.6f}  |a2| = {params.b:.6f}  "
        f"|a3| = {params.c:.6f} bohr",
        f"       = {params.a * ang:.6f}       = {params.b * ang:.6f}       "
        f"= {params.c * ang:.6f} Angstrom",
        f"  alpha = {params.alpha:.4f}  beta = {params.beta:.4f}  "
        f"gamma = {params.gamma:.4f} deg",
    ]
    if dim < 3:
        lines.append(
            f"  (only the first {dim} axis/axes are physical for dim={dim}; "
            "the rest are synthesized bookkeeping and are not constrained)"
        )

    # The single most useful thing this message can say: whether the OTHER
    # reading of the same matrix would have satisfied the declaration.
    try:
        transposed = _as_lattice(lattice_or_system).T
        if not _declaration_failures(
            declared,
            cell_parameters(transposed),
            dim,
            length_rtol,
            length_atol_bohr,
            angle_atol_deg,
        ):
            lines += [
                "",
                "The TRANSPOSE of this matrix does satisfy the declaration, so "
                "the vectors were almost certainly supplied as ROWS. vibe-qc "
                "reads them as COLUMNS (cpp/include/vibeqc/periodic.hpp), "
                "while ASE cells, PySCF, POSCAR and CIF listings are all "
                "row-oriented. Build the cell from the vectors themselves --",
                "    PeriodicSystem(dim, lattice_vectors=[a1, a2, a3], "
                "unit_cell=...)",
                "or vq.lattice_from_vectors(a1, a2, a3) -- which never asks "
                "you for a matrix orientation. The lattice is NOT transposed "
                "for you: both readings are valid cells, and silently picking "
                "one would rotate the lattice relative to the unchanged "
                "Cartesian atom positions (GitLab #128).",
            ]
    except Exception:  # pragma: no cover - diagnostics must never mask the error
        pass

    lines += [
        "",
        f"Tolerances: lengths {length_rtol:g} relative + "
        f"{length_atol_bohr:g} bohr absolute, angles {angle_atol_deg:g} deg. "
        "Loosen with length_rtol=, length_atol_bohr=, angle_atol_deg=.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Declared space group (GitLab #128, ask 1 -- "a crystal system OR space group")
#
# This is a different check from check_crystal_system, not a stronger one, and
# the difference is worth being precise about.
#
# A crystal-system declaration is a statement about the LATTICE, tested against
# the metric alone. A space-group declaration is a statement about the WHOLE
# STRUCTURE -- lattice and basis together -- and is tested by handing both to
# spglib. So it catches things no metric can see: an atom on the wrong site, a
# fractional coordinate transcribed wrongly, a basis that silently broke the
# symmetry the caller believed the structure had.
#
# What it does NOT do is supersede the crystal-system check as a transpose
# guard. Measured on the standard cell constructions, with the Cartesian atom
# positions held fixed and only the lattice matrix transposed -- the actual
# shape of every instance in #128:
#
#   hexagonal    P-6m2 (187) -> Pm (6)      caught
#   monoclinic   P2/m  (10)  -> P2/m (10)   NOT caught
#   triclinic    P-1   (2)   -> P-1  (2)    NOT caught
#
# Monoclinic and triclinic transposes stay inside their own system and keep
# their symmetry operations, so neither check sees them. Compare the cell
# parameters against known values for those.
#
# Note what this means for the obvious shortcut: do NOT derive a crystal system
# from a detected space group and then check the metric against it. The space
# group is lowered by the basis while the metric is not, so an exactly
# hexagonal lattice carrying a symmetry-breaking basis is reported monoclinic
# and a metric check driven off that number would refuse a perfectly good cell.
# The two declarations are kept separate here for that reason.
# ---------------------------------------------------------------------------


def check_space_group(
    system: Any,
    space_group: int | str,
    *,
    symprec: float = 1.0e-4,
) -> Any:
    """Refuse a structure whose space group contradicts the declaration.

    ``space_group`` is either an International Tables number (1 to 230) or a
    Hermann-Mauguin symbol such as ``"Fd-3m"``; spaces and case are ignored in
    a symbol, so ``"P 6/m m m"`` and ``"p6/mmm"`` both work. Returns the
    detected :class:`SpaceGroup` when the declaration holds and raises
    :class:`LatticeDeclarationError` when it does not.

    The declaration is about the structure, lattice and basis together, so
    unlike :func:`check_crystal_system` this needs the atoms and reports what
    spglib makes of them. The comparison is exact: declare the group the
    structure actually has, not a parent group it was derived from. A
    symmetry-broken supercell of a P6/mmm crystal is not P6/mmm, and saying so
    is the point of the check.

    ``symprec`` is spglib's tolerance and genuinely changes the answer -- a
    cell 0.1% off cubic reads as P4/mmm at the 1e-4 default and Pm-3m at 1e-2
    -- so it is exposed rather than hidden. The default matches the
    ``symmetry_precision`` used elsewhere in the runner.

    Only 3-D cells are accepted. For a slab spglib would treat the synthesized
    vacuum axis as periodic and return a group that depends on how much vacuum
    was chosen, which is not a fact about the material.

    Read the module section above for how this relates to a crystal-system
    declaration: it is complementary, and neither catches a transposed
    monoclinic or triclinic cell.
    """
    dim = int(getattr(system, "dim", 3))
    if dim != 3:
        raise ValueError(
            "check_space_group: only 3-D cells have a space group here; got "
            f"dim={dim}. For a slab spglib would treat the synthesized vacuum "
            "axis as periodic and return a group that changes with the vacuum "
            "thickness. Use check_crystal_system for the plane lattice."
        )
    if not hasattr(system, "unit_cell"):
        raise TypeError(
            "check_space_group needs a PeriodicSystem (lattice and atoms), "
            f"not a bare lattice matrix; got {type(system).__name__}. A space "
            "group is a property of the structure, not of the lattice alone."
        )

    from .periodic_symmetrize import detect_spacegroup

    detected = detect_spacegroup(system, symprec=symprec)
    if _space_group_matches(space_group, detected):
        return detected

    raise LatticeDeclarationError(
        _space_group_message(system, space_group, detected, symprec)
    )


def _normalise_hm(symbol: str) -> str:
    """Hermann-Mauguin symbol with spacing and case made irrelevant."""
    return "".join(str(symbol).split()).lower()


def _space_group_matches(declared: int | str, detected: Any) -> bool:
    if isinstance(declared, bool):  # bool is an int; nobody means that here
        raise TypeError("check_space_group: space_group must be a number or a symbol")
    if isinstance(declared, (int, np.integer)):
        number = int(declared)
        if not 1 <= number <= 230:
            raise ValueError(
                "check_space_group: an International Tables space-group "
                f"number runs from 1 to 230; got {number}"
            )
        return int(detected.number) == number
    if isinstance(declared, str):
        want = _normalise_hm(declared)
        if not want:
            raise ValueError("check_space_group: space_group symbol is empty")
        return _normalise_hm(detected.international_symbol) == want
    raise TypeError(
        "check_space_group: space_group must be an International Tables "
        f"number or a Hermann-Mauguin symbol; got {type(declared).__name__}"
    )


def _space_group_message(
    system: Any, declared: int | str, detected: Any, symprec: float
) -> str:
    lines = [
        f"structure contradicts the declared space group {declared!r}.",
        "",
        f"  declared : {declared!r}",
        f"  detected : {detected.number} ({detected.international_symbol}), "
        f"point group {detected.point_group}",
        f"  symprec  : {symprec:g}",
    ]
    try:
        params = cell_parameters(system)
        ang = _BOHR_TO_ANGSTROM
        lines += [
            "",
            "Reading the lattice vectors as COLUMNS:",
            f"  |a1| = {params.a:.6f}  |a2| = {params.b:.6f}  "
            f"|a3| = {params.c:.6f} bohr",
            f"       = {params.a * ang:.6f}       = {params.b * ang:.6f}       "
            f"= {params.c * ang:.6f} Angstrom",
            f"  alpha = {params.alpha:.4f}  beta = {params.beta:.4f}  "
            f"gamma = {params.gamma:.4f} deg",
        ]
    except Exception:  # pragma: no cover - diagnostics must not mask the error
        pass

    # As for the metric check: the single most useful thing to say is whether
    # the other reading of the same matrix would have satisfied the
    # declaration.
    try:
        from .periodic_symmetrize import detect_spacegroup

        # Build a fresh system rather than copying: PeriodicSystem is a
        # pybind11 type with no __copy__, and mutating the caller's object
        # here would be worse than having no diagnostic.
        transposed = type(system)(
            int(system.dim),
            _as_lattice(system).T,
            list(system.unit_cell),
            int(getattr(system, "charge", 0)),
            int(getattr(system, "multiplicity", 1)),
        )
        if _space_group_matches(declared, detect_spacegroup(transposed, symprec=symprec)):
            lines += [
                "",
                "The TRANSPOSE of this lattice does satisfy the declaration, "
                "so the vectors were almost certainly supplied as ROWS. "
                "vibe-qc reads them as COLUMNS, while ASE cells, PySCF, "
                "POSCAR and CIF listings are all row-oriented. Build the cell "
                "from the vectors themselves --",
                "    PeriodicSystem(dim, lattice_vectors=[a1, a2, a3], "
                "unit_cell=...)",
                "-- which never asks you for a matrix orientation. The "
                "lattice is NOT transposed for you (GitLab #128).",
            ]
    except Exception:  # pragma: no cover
        pass

    lines += [
        "",
        "If the structure is right and the declaration was the parent group "
        "of a symmetry-broken cell, declare the group this structure actually "
        "has. If the symmetry is only approximate, raise symprec=.",
    ]
    return "\n".join(lines)
