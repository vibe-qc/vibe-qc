"""Phase SYM2b: orbit identification and LatticeMatrixSet compression
under the space group.

Given a real-space ``LatticeMatrixSet`` (blocks of an integral matrix
``O(h)`` indexed by lattice cells ``h``) and the crystal's
crystallographic point group, partition the cell list into orbits
under the group action ``h ↦ R . h`` and compress the matrix by
storing only one block per orbit. Reconstruction on demand uses the
AO-rotation machinery from SYM1 + SYM2a.

Orbit reconstruction
--------------------

If ``h'' = R . h`` and both ``h`` and ``h'`` are in the cell list,
then for any operator-invariant integral matrix ``O``:

    O(R . h) = P(R) . O(h) . P(R)^T

where ``P(R)`` is the AO permutation matrix built by SYM2a. This
identity is the compression: storing ``O(h)`` for one representative
per orbit plus the operator index suffices to rebuild every other
member of the orbit.

Scope and assumptions
---------------------

- **Symmorphic space groups only.** The underlying space group must
  have zero fractional translation for every operator (``t = 0`` for
  every ``(R, t)``). Non-symmorphic groups introduce a per-operator
  lattice shift on the cell index that this implementation doesn't
  thread through yet.

- **All atoms at operator-fixed positions.** Every operator must
  leave every atom fixed *within the reference cell* -- meaning
  ``atom_permutation_under_op(system, R, 0).lattice_shift`` is all
  zero for every operator. This is the case when every atom sits at
  a Wyckoff position fixed by the whole point group (e.g., atoms at
  the origin of a primitive cell). Structures like NaCl (Cl at
  ``(a/2, a/2, a/2)``) have an origin-fixed atom plus one that picks
  up a lattice shift under most cubic operators; those need the
  AO-pair-dependent cell-shift machinery, which is a planned SYM2c
  follow-up.

  The routines here raise ``ValueError`` when an operator has a
  non-zero atom lattice shift, rather than silently giving wrong
  reconstructions.

- **Symmetry-closed cell list.** If ``h`` is in ``lms.cells``, then
  ``R . h`` must also be there for every operator ``R``. Real-space
  sums truncated by a spherical cutoff satisfy this automatically;
  other cutoff shapes might not. The routines here raise
  ``ValueError`` when the cell set isn't closed, rather than silently
  reconstructing wrong blocks.

- **Operator matrix elements in Cartesian.** The AO rotation goes
  through Wigner D-matrices in Cartesian coordinates. The cell-index
  action uses the integer lattice-basis form of the same operator.
  We convert via ``R_cart = L . R_lat . L⁻¹`` where ``L`` is the
  crystal's lattice matrix (columns = a₁, a₂, a₃).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    LatticeCell,
    LatticeMatrixSet,
    PeriodicSystem,
    SymmetryOp,
)
from .symmetry_ao import atom_permutation_under_op, build_ao_permutation_matrix

__all__ = [
    "LatticeOrbit",
    "LatticeOrbits",
    "lattice_to_cartesian_rotation",
    "identify_lattice_orbits",
    "compress_lattice_matrix_set",
    "reconstruct_lattice_matrix_set",
]


# ---------------------------------------------------------------------------
# Coordinate-frame conversion
# ---------------------------------------------------------------------------


def lattice_to_cartesian_rotation(
    R_lat: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """Convert an integer lattice-basis operator matrix to Cartesian.

    ``R_lat`` is an integer 3x3 matrix in the basis of lattice vectors
    (the form returned by spglib); ``lattice`` is the 3x3 lattice
    matrix with columns ``a₁, a₂, a₃`` (bohr). The Cartesian form

        R_cart  =  L . R_lat . L⁻¹

    is real orthogonal -- a rotation or reflection of 3D space --
    because point-group operators preserve the Gram matrix of the
    lattice.
    """
    L = np.asarray(lattice, dtype=float)
    R_l = np.asarray(R_lat, dtype=float)
    return L @ R_l @ np.linalg.inv(L)


# ---------------------------------------------------------------------------
# Orbit structures
# ---------------------------------------------------------------------------


@dataclass
class LatticeOrbit:
    """A single orbit of the space group acting on the cell index list.

    Attributes
    ----------
    representative
        Index of the orbit's representative cell in the original
        ``lms.cells`` list. By convention the representative is the
        orbit member with the lowest cell-list index.
    members
        Indices (in ``lms.cells``) of all orbit members. Always
        contains ``representative`` as ``members[0]``.
    ops
        Operator indices (into the ``operations`` list passed to
        :func:`identify_lattice_orbits`) such that
        ``cells[members[i]].index == ops[i] . cells[representative].index``.
        ``ops[0]`` is always the identity-operator index.
    """

    representative: int
    members: List[int] = field(default_factory=list)
    ops: List[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class LatticeOrbits:
    """Partition of a ``LatticeMatrixSet``'s cell list into orbits."""

    orbits: List[LatticeOrbit]
    n_cells: int

    @property
    def n_orbits(self) -> int:
        return len(self.orbits)

    @property
    def compression_ratio(self) -> float:
        """Average orbit size -- the factor by which storing only
        representatives reduces memory for the block list."""
        if self.n_orbits == 0:
            return 1.0
        return self.n_cells / self.n_orbits

    def representative_indices(self) -> List[int]:
        """Cell-list indices of the orbit representatives, one per
        orbit, in orbit order."""
        return [o.representative for o in self.orbits]


# ---------------------------------------------------------------------------
# Orbit identification
# ---------------------------------------------------------------------------


def identify_lattice_orbits(
    cells: Sequence[LatticeCell],
    operations: Sequence[SymmetryOp],
    *,
    require_closed: bool = True,
) -> LatticeOrbits:
    """Partition ``cells`` into orbits under the space-group action
    ``h ↦ R . h`` on lattice-vector indices.

    Parameters
    ----------
    cells
        Cell list, typically ``LatticeMatrixSet.cells`` or equivalent.
    operations
        Sequence of :class:`SymmetryOp` (from
        :attr:`PeriodicSystem.symmetry.operations`). Must have
        ``translation = 0`` (symmorphic group); non-zero translations
        are rejected.
    require_closed
        If True (default), raise ``ValueError`` when some cell has an
        orbit partner that isn't in the cell list -- the cell set
        isn't symmetry-closed and compression would silently drop
        blocks. Set False to partition whatever orbits are fully
        present and silently ignore cells with partners outside the
        list.

    Returns
    -------
    :class:`LatticeOrbits`.
    """
    n = len(cells)
    # Map from integer cell index tuple to position in the cell list.
    cell_indices: List[Tuple[int, int, int]] = [
        tuple(int(x) for x in c.index) for c in cells
    ]
    index_to_slot = {idx: i for i, idx in enumerate(cell_indices)}

    # Preflight on operators: store rotations for cell-index action.
    op_rotations: List[np.ndarray] = []
    for op_idx, op in enumerate(operations):
        op_rotations.append(np.asarray(op.rotation, dtype=int))

    # Partition via a visited array; for each unvisited cell, trace
    # out its orbit by applying every operator and chasing the union.
    visited = [False] * n
    orbits: List[LatticeOrbit] = []
    # Identity operator index -- we assert the operator list contains
    # one rather than search, since the cell itself is always its own
    # representative under identity.
    identity_op_idx: Optional[int] = None
    for op_idx, R in enumerate(op_rotations):
        if np.array_equal(R, np.eye(3, dtype=int)):
            identity_op_idx = op_idx
            break
    if identity_op_idx is None:
        raise ValueError(
            "identify_lattice_orbits: operator list must contain "
            "the identity (I) -- this is a required group axiom and "
            "every spglib-returned operator set satisfies it."
        )

    for start in range(n):
        if visited[start]:
            continue
        rep = start
        orb = LatticeOrbit(representative=rep, members=[rep], ops=[identity_op_idx])
        visited[rep] = True
        for op_idx, R in enumerate(op_rotations):
            image = tuple(int(x) for x in (R @ np.array(cell_indices[rep])))
            if image not in index_to_slot:
                if require_closed:
                    raise ValueError(
                        f"identify_lattice_orbits: cell "
                        f"{cell_indices[rep]} has an orbit partner "
                        f"{image} under operator {op_idx} that isn't "
                        f"in the cell list -- the cell set is not "
                        f"symmetry-closed. Use a spherical cutoff or "
                        f"pass require_closed=False."
                    )
                continue
            j = index_to_slot[image]
            if j == rep:
                # Stabiliser member; already have (rep, identity).
                continue
            if not visited[j]:
                visited[j] = True
                orb.members.append(j)
                orb.ops.append(op_idx)
        orbits.append(orb)

    return LatticeOrbits(orbits=orbits, n_cells=n)


# ---------------------------------------------------------------------------
# Compression and reconstruction
# ---------------------------------------------------------------------------


def _ao_permutation_cache(
    basis: BasisSet,
    system: PeriodicSystem,
    operations: Sequence[SymmetryOp],
) -> List[np.ndarray]:
    """Build an AO permutation matrix P(R) for every symmetry
    operator once. Cached so caller loops don't re-diagonalize for
    every block."""
    L = np.asarray(system.lattice, dtype=float)
    cache: List[np.ndarray] = []
    for op_idx, op in enumerate(operations):
        R_cart = lattice_to_cartesian_rotation(op.rotation, L)
        t_cart = L @ np.asarray(op.translation, dtype=float)
        ap = atom_permutation_under_op(system, R_cart, t_cart)
        # Guard against the case that defeats the simple reconstruction
        # formula: any atom picking up a non-zero lattice shift under
        # this operator means the effective cell-index shift depends on
        # which AO-pair block we're in. See the SYM2c note in the
        # module docstring.
        if np.any(ap.lattice_shift != 0):
            raise ValueError(
                f"symmetry_lattice: operator {op_idx} moves at least "
                f"one atom with a non-zero lattice shift "
                f"(lattice_shift = {ap.lattice_shift.tolist()}). The "
                f"current SYM2b implementation supports only structures "
                f"where every atom is fixed (up to permutation within "
                f"the reference cell) by every operator. Multi-atom "
                f"structures with atoms at non-origin-fixed Wyckoff "
                f"positions need the AO-pair-dependent cell-shift "
                f"machinery planned as Phase SYM2c."
            )
        P = build_ao_permutation_matrix(basis, R_cart, ap)
        cache.append(P)
    return cache


def compress_lattice_matrix_set(
    lms: LatticeMatrixSet,
    orbits: LatticeOrbits,
) -> List[np.ndarray]:
    """Return the list of orbit-representative blocks, one per
    orbit, in orbit order.

    This is a data-only reduction -- it doesn't check whether the
    operator actually leaves the matrix invariant (that's a physics
    property, not a data-structure property). Use
    :func:`reconstruct_lattice_matrix_set` to round-trip and verify.
    """
    return [np.asarray(lms.blocks[orb.representative]).copy() for orb in orbits.orbits]


def reconstruct_lattice_matrix_set(
    representatives: Sequence[np.ndarray],
    orbits: LatticeOrbits,
    basis: BasisSet,
    system: PeriodicSystem,
    operations: Sequence[SymmetryOp],
) -> List[np.ndarray]:
    """Rebuild the full list of blocks from orbit representatives.

    For each orbit member related to its representative by operator
    ``R``, the member block is ``P(R) . O(rep) . P(R)^T`` where
    ``P(R)`` is the AO permutation matrix.

    Returns a ``List[np.ndarray]`` of length ``orbits.n_cells``, one
    block per cell, in the cell-list order of the original LMS.
    """
    if len(representatives) != orbits.n_orbits:
        raise ValueError(
            f"reconstruct_lattice_matrix_set: expected "
            f"{orbits.n_orbits} representative blocks, got "
            f"{len(representatives)}"
        )

    # Build AO permutation matrices once, then place blocks.
    P_cache = _ao_permutation_cache(basis, system, operations)

    n_bf = basis.nbasis
    blocks: List[Optional[np.ndarray]] = [None] * orbits.n_cells

    for orb_idx, orb in enumerate(orbits.orbits):
        rep_block = np.asarray(representatives[orb_idx], dtype=float)
        if rep_block.shape != (n_bf, n_bf):
            raise ValueError(
                f"reconstruct_lattice_matrix_set: representative "
                f"block for orbit {orb_idx} has shape "
                f"{rep_block.shape}; expected ({n_bf}, {n_bf})"
            )
        for member_cell, op_idx in zip(orb.members, orb.ops):
            P = P_cache[op_idx]
            blocks[member_cell] = P @ rep_block @ P.T

    # Every cell should have been visited.
    missing = [i for i, b in enumerate(blocks) if b is None]
    if missing:
        raise RuntimeError(
            f"reconstruct_lattice_matrix_set: {len(missing)} cells "
            f"were not covered by any orbit -- orbit partition is "
            f"incomplete. First missing index: {missing[0]}"
        )
    return [b for b in blocks if b is not None]
