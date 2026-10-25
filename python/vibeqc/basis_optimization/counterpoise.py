"""Counterpoise free atoms for solids: an atom in the ghost basis of its crystal.

This is vibe-qc's counterpart of CRYSTAL's ``ATOMBSSE``, the counterpoise
correction the published pob cohesive energies use for every free atom
(Vilela Oliveira, Laun, Peintinger & Bredow, *J. Comput. Chem.* **40**, 2364
(2019), doi:10.1002/jcc.26013). It is roadmap dependency BOC-3 of
``vibe-basis/ROADMAP.md``.

Method
------
The counterpoise correction of Boys and Bernardi, *Mol. Phys.* **19**, 553
(1970), doi:10.1080/00268977000101561, computes each fragment in the full
basis of the complex, so the basis-set superposition error cancels in the
difference. In a crystal the fragment is one atom and the complex is
infinite, so the ghost basis is truncated to a finite cluster. The
truncation follows the CRYSTAL23 manual's definition of ``ATOMBSSE IAT NSTAR
RMAX`` (geometry input, "ATOMBSSE - counterpoise for closed shell atoms and
ions"):

    A cluster is defined including the selected atom and the basis
    functions belonging to the NSTAR sets of neighbors, when their distance
    R from the central atom is smaller than RMAX.

and the manual's ``CLUSTER`` keyword, which applies the same two criteria
("one of the first N stars of neighbours" *and* "a distance ... smaller then
RMAX") and reads RMAX in angstrom. CRYSTAL then computes the atom as a 0D
system. This module does the same: it builds the cluster from a
:class:`~vibeqc.PeriodicSystem` and returns a molecule whose only real atom
is the selected one, with every neighbour in the cluster carrying its basis
functions on a ``Z = 0`` ghost centre.

A **star** is a set of neighbours related by the site symmetry of the
central atom, not merely a distance shell. In rocksalt the neighbours at
``(a/2)(3, 0, 0)`` and ``(a/2)(2, 2, 1)`` lie at the same distance but form
two stars of 6 and 24 atoms, and CRYSTAL counts them as two. The stars here
are the orbits of the neighbour vectors under the site-symmetry group that
spglib reports for the central atom, so ``nstar`` counts what CRYSTAL counts.
Where a star boundary falls inside a set of stars at one distance, the order
among them is by atomic number, then star size, then the smallest Cartesian
member. That is deterministic and independent of the cell, but it is not
guaranteed to be CRYSTAL's order.

Cost
----
The cluster calculation is a molecular SCF over every ghost function in the
cluster. The reference set's converged settings range from ``10 / 5.0`` for
MgO, about 56 neighbours, to ``30 / 10.0`` for LiF, about 500. A counterpoise
atom is also specific to its host crystal, so unlike a bare atom it cannot be
shared between compounds. Use it for reference comparisons, not for every
point of an optimisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Bohr per angstrom, the value :mod:`vibeqc.molecule` uses.
from vibeqc.molecule import ANGSTROM_TO_BOHR

__all__ = [
    "GhostCluster",
    "ghost_cluster",
    "ghost_cluster_basis",
    "ghost_cluster_molecule",
]

# A neighbour closer than this to the central atom is the central atom itself.
_SELF_TOL_BOHR = 1.0e-6
# Neighbours at RMAX within this margin are excluded: CRYSTAL keeps only those
# "smaller than RMAX", and floating point cannot settle an exact tie.
_RMAX_MARGIN_ANGSTROM = 1.0e-6
# Decimals kept when canonicalising a fractional neighbour vector.
_KEY_DECIMALS = 6


@dataclass(frozen=True)
class GhostCluster:
    """The finite cluster of an ``ATOMBSSE``-style counterpoise atom.

    Positions are Cartesian, in bohr, relative to the central atom, which sits
    at the origin.

    Attributes
    ----------
    center_index
        Index of the real atom in ``system.unit_cell``.
    center_Z
        Its atomic number.
    nstar, rmax_angstrom
        The two truncation criteria the cluster was built with.
    ghost_Z
        Atomic number of each ghost's element, which selects its basis
        functions. The ghosts themselves carry no charge and no electrons.
    ghost_positions_bohr
        ``(n_ghost, 3)`` array of ghost positions.
    ghost_star
        Zero-based star index of each ghost.
    star_distances_angstrom, star_sizes, star_Z
        Distance, membership count and element of each kept star, in order.
    """

    center_index: int
    center_Z: int
    nstar: int
    rmax_angstrom: float
    ghost_Z: tuple[int, ...]
    ghost_positions_bohr: np.ndarray
    ghost_star: tuple[int, ...]
    star_distances_angstrom: tuple[float, ...]
    star_sizes: tuple[int, ...]
    star_Z: tuple[int, ...]

    @property
    def n_ghost(self) -> int:
        """Number of ghost centres in the cluster."""
        return len(self.ghost_Z)

    @property
    def n_star(self) -> int:
        """Number of neighbour stars the cluster keeps."""
        return len(self.star_sizes)

    def summary(self) -> dict[str, Any]:
        """A JSON-friendly record of the cluster, for provenance."""
        return {
            "center_index": self.center_index,
            "center_Z": self.center_Z,
            "nstar": self.nstar,
            "rmax_angstrom": self.rmax_angstrom,
            "n_ghost": self.n_ghost,
            "n_star": self.n_star,
            "star_distances_angstrom": [
                round(d, 6) for d in self.star_distances_angstrom
            ],
            "star_sizes": list(self.star_sizes),
            "star_Z": list(self.star_Z),
        }


def _site_rotations(
    lattice_rows: np.ndarray,
    fractional: np.ndarray,
    numbers: list[int],
    center_index: int,
    symprec: float,
) -> list[np.ndarray]:
    """Fractional rotation parts of the operations that fix the central atom."""
    import spglib

    symmetry = spglib.get_symmetry(
        (lattice_rows, fractional, numbers), symprec=symprec
    )
    if symmetry is None:  # pragma: no cover - spglib returns identity at least
        return [np.eye(3, dtype=int)]
    f0 = fractional[center_index]
    # A tolerance in fractional units that matches spglib's Cartesian one.
    tol = symprec / max(min(np.linalg.norm(lattice_rows, axis=1)), 1e-12) * 10
    rotations = []
    for rot, trans in zip(symmetry["rotations"], symmetry["translations"]):
        shift = rot @ f0 + trans - f0
        if np.all(np.abs(shift - np.round(shift)) < max(tol, 1e-6)):
            rotations.append(np.asarray(rot, dtype=int))
    return rotations or [np.eye(3, dtype=int)]


def _canonical_key(d_frac: np.ndarray, rotations: list[np.ndarray]) -> tuple:
    """The smallest image of a neighbour vector under the site rotations."""
    images = (
        tuple(np.round(rot @ d_frac, _KEY_DECIMALS) + 0.0) for rot in rotations
    )
    return min(images)


def ghost_cluster(
    system: Any,
    center_index: int,
    *,
    nstar: int = 30,
    rmax_angstrom: float = 10.0,
    symprec: float = 1.0e-5,
) -> GhostCluster:
    """Build the ``ATOMBSSE`` cluster around one atom of a 3-D crystal.

    Parameters
    ----------
    system
        A :class:`~vibeqc.PeriodicSystem` with ``dim == 3``.
    center_index
        Index of the real atom in ``system.unit_cell``.
    nstar
        ``NSTAR``: keep at most this many stars of neighbours.
    rmax_angstrom
        ``RMAX``: keep only neighbours closer than this, in angstrom.
    symprec
        spglib's tolerance for the site symmetry that defines the stars.

    Returns
    -------
    GhostCluster
    """
    if int(getattr(system, "dim", 3)) != 3:
        raise ValueError(
            "ghost_cluster needs a 3-D crystal; ATOMBSSE carves a 0D cluster "
            f"out of a 3-D lattice, got dim={getattr(system, 'dim', None)}"
        )
    if nstar < 0:
        raise ValueError(f"nstar must be >= 0, got {nstar}")
    if not rmax_angstrom >= 0.0:
        raise ValueError(f"rmax_angstrom must be >= 0, got {rmax_angstrom}")

    atoms = list(system.unit_cell)
    if not 0 <= center_index < len(atoms):
        raise ValueError(
            f"center_index {center_index} is outside the {len(atoms)}-atom "
            "unit cell"
        )

    columns = np.asarray(system.lattice, dtype=float)  # columns a, b, c (bohr)
    rows = columns.T
    inverse = np.linalg.inv(columns)
    cart = np.array([np.asarray(a.xyz, dtype=float) for a in atoms])
    frac = (inverse @ cart.T).T
    numbers = [int(a.Z) for a in atoms]

    rmax_bohr = max(rmax_angstrom - _RMAX_MARGIN_ANGSTROM, 0.0) * ANGSTROM_TO_BOHR
    # Every atom within rmax of the centre lies within ceil(rmax / d_i) + 1
    # cells along axis i, where d_i = 1 / |row i of the inverse| is the
    # spacing of the lattice planes normal to that axis.
    reach = [
        int(math.ceil(rmax_bohr * np.linalg.norm(inverse[i]))) + 1
        for i in range(3)
    ]
    grid = np.array(
        np.meshgrid(*(np.arange(-n, n + 1) for n in reach), indexing="ij")
    ).reshape(3, -1).T

    rotations = _site_rotations(rows, frac, numbers, center_index, symprec)

    f0 = frac[center_index]
    neighbours = []  # (distance, Z, key, cartesian vector)
    for j, fj in enumerate(frac):
        d_frac = (fj - f0)[None, :] + grid
        d_cart = d_frac @ rows
        dist = np.linalg.norm(d_cart, axis=1)
        keep = (dist > _SELF_TOL_BOHR) & (dist < rmax_bohr)
        for k in np.nonzero(keep)[0]:
            neighbours.append(
                (
                    float(dist[k]),
                    numbers[j],
                    _canonical_key(d_frac[k], rotations),
                    d_cart[k],
                )
            )

    stars: dict[tuple, list] = {}
    for dist, Z, key, vec in neighbours:
        stars.setdefault((Z, key), []).append((dist, vec))
    def star_order(item):
        (Z, _key), members = item
        # Distance first; ties (two stars at one distance) by element, then
        # size, then the smallest Cartesian member. Every part is a property
        # of the crystal, not of the cell that describes it, so the same
        # NSTAR keeps the same atoms from a primitive or a conventional cell.
        corner = min(tuple(np.round(vec, _KEY_DECIMALS) + 0.0) for _d, vec in members)
        return (round(members[0][0], 8), Z, len(members), corner)

    ordered = sorted(stars.items(), key=star_order)[:nstar]

    ghost_Z: list[int] = []
    ghost_pos: list[np.ndarray] = []
    ghost_star: list[int] = []
    star_dist, star_size, star_Z = [], [], []
    for index, ((Z, _key), members) in enumerate(ordered):
        star_dist.append(members[0][0] / ANGSTROM_TO_BOHR)
        star_size.append(len(members))
        star_Z.append(Z)
        for _dist, vec in members:
            ghost_Z.append(Z)
            ghost_pos.append(vec)
            ghost_star.append(index)

    return GhostCluster(
        center_index=center_index,
        center_Z=numbers[center_index],
        nstar=int(nstar),
        rmax_angstrom=float(rmax_angstrom),
        ghost_Z=tuple(ghost_Z),
        ghost_positions_bohr=(
            np.array(ghost_pos, dtype=float).reshape(-1, 3)
        ),
        ghost_star=tuple(ghost_star),
        star_distances_angstrom=tuple(star_dist),
        star_sizes=tuple(star_size),
        star_Z=tuple(star_Z),
    )


def ghost_cluster_molecule(
    cluster: GhostCluster, *, charge: int = 0, multiplicity: int = 1
):
    """The cluster as a molecule: the real atom plus ``Z = 0`` ghost centres."""
    import vibeqc as vq

    atoms = [vq.Atom(cluster.center_Z, [0.0, 0.0, 0.0])]
    atoms += [
        vq.Atom(0, [float(c) for c in pos]) for pos in cluster.ghost_positions_bohr
    ]
    return vq.Molecule(atoms, charge, multiplicity)


def ghost_cluster_basis(cluster: GhostCluster, molecule: Any, basis_name: str):
    """The basis of every cluster atom, placed on ``molecule``'s centres.

    ``molecule`` is the output of :func:`ghost_cluster_molecule`. The shells
    are taken from the same cluster with every atom real, so each ghost
    carries exactly the functions its element has in ``basis_name``, and the
    real atom's shells come first and are the ones a bare atom would get.
    """
    import vibeqc as vq

    real_atoms = [vq.Atom(cluster.center_Z, [0.0, 0.0, 0.0])]
    real_atoms += [
        vq.Atom(int(Z), [float(c) for c in pos])
        for Z, pos in zip(cluster.ghost_Z, cluster.ghost_positions_bohr)
    ]
    electrons = sum(int(a.Z) for a in real_atoms)
    full = vq.Molecule(real_atoms, 0, 1 if electrons % 2 == 0 else 2)
    shells = vq.BasisSet(full, basis_name).shells()
    return vq.BasisSet(molecule, shells, name=basis_name)
