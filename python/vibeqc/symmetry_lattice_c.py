"""Phase SYM2c: atom-pair-resolved orbit identification and
LatticeMatrixSet compression for structures where atoms aren't all
at operator-fixed positions.

SYM2b restricts to structures where every atom is fixed (up to
within-cell permutation) by every operator -- this holds for
single-atom primitive cells but fails for common ionic crystals
like NaCl where Cl at (1/2, 1/2, 1/2) picks up a lattice shift under
most cubic operators. Under that restriction the simple formula

    F(R . h)  =  P(R) . F(h) . P(R)ᵀ

relates the full ``(n_bf x n_bf)`` block at cell ``h`` to the block
at cell ``R.h``.

SYM2c lifts the restriction. The more general identity relates
*atom-pair sub-blocks*:

    F^{(pi(a), pi(b))}(R.h + s_b - s_a)  =  D_a(R) . F^{(a, b)}(h) . D_b(R)ᵀ

where ``D_a(R)`` / ``D_b(R)`` are the shell-wise Wigner D-matrices
on atoms a and b (block-diagonal in shell-ℓ), and ``s_a(R)``,
``s_b(R)`` are the integer lattice-basis shifts that take atom
a / b to its image under ``R``.

The corresponding orbit structure is **per-(atom-pair, cell)** --
the natural labels are triples ``(source_atom, dest_atom, h)`` under
the group action

    (a, b, h)  ↦  (pi(a), pi(b), R . h + s_b - s_a).

This module provides:

- :class:`AtomPairOrbit` / :class:`AtomPairOrbits` data structures.
- :func:`identify_atom_pair_orbits` -- partition the full triple
  space into orbits under the group.
- :func:`compress_lattice_matrix_set_c` / :func:`reconstruct_lattice_matrix_set_c`
  -- compression and round-trip that work on NaCl-class structures.

SYM2b's simpler API (:func:`identify_lattice_orbits`,
:func:`compress_lattice_matrix_set`, :func:`reconstruct_lattice_matrix_set`)
remains the preferred path when all atoms are origin-fixed --
it has less bookkeeping overhead and the same accuracy. Use SYM2c
when SYM2b raises "non-origin-fixed atom" errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    LatticeCell,
    LatticeMatrixSet,
    PeriodicSystem,
    SymmetryOp,
)
from .symmetry_ao import atom_permutation_under_op, build_ao_permutation_matrix
from .symmetry_lattice import lattice_to_cartesian_rotation

__all__ = [
    "AtomPairOrbit",
    "AtomPairOrbits",
    "identify_atom_pair_orbits",
    "operator_triple_actions",
    "compress_lattice_matrix_set_c",
    "reconstruct_lattice_matrix_set_c",
]


# A cell index is stored as a tuple[int, int, int] for hashability.
CellIndex = Tuple[int, int, int]
Triple = Tuple[int, int, CellIndex]  # (source_atom, dest_atom, cell_index)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AtomPairOrbit:
    """A single orbit of the space group acting on
    (source_atom, dest_atom, cell_index) triples.

    ``representative`` is the triple picked as the orbit canonical
    member; ``members`` lists every triple in the orbit; ``ops``
    gives the operator index that carries the representative to each
    member (``ops[0]`` is always the identity-operator index).
    """

    representative: Triple
    members: List[Triple] = field(default_factory=list)
    ops: List[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class AtomPairOrbits:
    """Partition of the (atom-pair, cell-index) triple space into
    orbits under the space group."""

    orbits: List[AtomPairOrbit]
    n_triples: int
    n_atoms: int
    n_cells: int

    @property
    def n_orbits(self) -> int:
        return len(self.orbits)

    @property
    def compression_ratio(self) -> float:
        """Average orbit size -- the memory-reduction factor when
        storing only representative sub-blocks."""
        if self.n_orbits == 0:
            return 1.0
        return self.n_triples / self.n_orbits


# ---------------------------------------------------------------------------
# Orbit identification
# ---------------------------------------------------------------------------


def _cell_tuple(idx) -> CellIndex:
    arr = np.asarray(idx, dtype=int)
    return (int(arr[0]), int(arr[1]), int(arr[2]))


def operator_triple_actions(
    operations: Sequence[SymmetryOp],
    system: PeriodicSystem,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
    """Per-operator ingredients of the triple action
    ``(a, b, h) -> (perm[a], perm[b], R.h + shifts[a] - shifts[b])``.

    Returns ``(op_R_lat, op_atom_perm, op_atom_shifts, identity_op_idx)``.
    ``shifts`` follow the ``atom_permutation_under_op`` storage
    convention (``shifts[a] = -s_a`` of the SYM2c math -- see the
    sign-convention note in :func:`identify_atom_pair_orbits`).
    Raises ``ValueError`` when the operator list lacks the identity.
    """
    L = np.asarray(system.lattice, dtype=float)
    op_R_lat: List[np.ndarray] = []
    op_atom_perm: List[np.ndarray] = []
    op_atom_shifts: List[np.ndarray] = []
    identity_op_idx: Optional[int] = None
    for op_idx, op in enumerate(operations):
        t_frac = np.asarray(op.translation, dtype=float)
        R_lat = np.asarray(op.rotation, dtype=int)
        op_R_lat.append(R_lat)
        R_cart = lattice_to_cartesian_rotation(R_lat, L)
        # Pass the full fractional translation for correct atom mapping
        t_cart = L @ t_frac
        ap = atom_permutation_under_op(system, R_cart, t_cart)
        op_atom_perm.append(ap.perm)
        op_atom_shifts.append(ap.lattice_shift.astype(int))
        if np.array_equal(R_lat, np.eye(3, dtype=int)) and np.all(
            np.abs(t_frac) < 1e-8
        ):
            identity_op_idx = op_idx
    if identity_op_idx is None:
        raise ValueError(
            "operator_triple_actions: operator list must contain the identity."
        )
    return op_R_lat, op_atom_perm, op_atom_shifts, identity_op_idx


def identify_atom_pair_orbits(
    cells: Sequence[LatticeCell],
    operations: Sequence[SymmetryOp],
    system: PeriodicSystem,
    *,
    require_closed: bool = True,
    triples: Optional[set] = None,
) -> AtomPairOrbits:
    """Partition the (source_atom, dest_atom, cell_index) triple
    space into orbits under the group action.

    Parameters
    ----------
    cells
        Cell list -- typically ``LatticeMatrixSet.cells``.
    operations
        Sequence of :class:`SymmetryOp`. Must have ``translation=0``
        (symmorphic groups); non-symmorphic are rejected.
    system
        :class:`PeriodicSystem` -- used to compute per-atom lattice
        shifts under each operator.
    require_closed
        If True (default), raise ``ValueError`` when some triple has
        an orbit partner that isn't in the triple space (cell outside
        the cutoff). Spherical-cutoff LMSs satisfy closure
        automatically for single-atom-at-origin cells; the general
        atom-pair action ``h -> R.h + s_a - s_b`` is NOT closed on any
        radial cell list once atoms sit off the rotation centres (the
        2026-07-12 SYM3b root cause) -- pass a group-invariant
        ``triples`` set (pair-resolved truncation) to restore closure.
    triples
        Optional restriction of the triple space: a set of
        ``(a, b, cell_tuple)`` triples. When given, only these triples
        are enumerated and partnership is checked against this set
        (partners outside it raise / are skipped per
        ``require_closed``); every triple's cell must be in ``cells``.
        When ``None`` (default), the triple space is the full
        ``n_atoms x n_atoms x cells`` product.

    Returns
    -------
    :class:`AtomPairOrbits`.
    """
    n_atoms = len(system.unit_cell)
    cell_indices: List[CellIndex] = [_cell_tuple(c.index) for c in cells]
    cell_to_slot: Dict[CellIndex, int] = {c: i for i, c in enumerate(cell_indices)}
    n_cells = len(cell_indices)
    if triples is not None:
        for t in triples:
            if t[2] not in cell_to_slot:
                raise ValueError(
                    f"identify_atom_pair_orbits: triple {t} names a cell "
                    f"that is not in the cell list."
                )

    # Preflight: build per-operator arrays for rotation, atom permutation,
    # and lattice shifts.
    op_R_lat, op_atom_perm, op_atom_shifts, identity_op_idx = (
        operator_triple_actions(operations, system)
    )

    def _in_space(triple: Triple) -> bool:
        if triples is not None:
            return triple in triples
        return triple[2] in cell_to_slot

    # Enumerate all triples (a, b, cell_idx). Store "visited" by
    # tuple lookup.
    visited: set = set()
    orbits: List[AtomPairOrbit] = []

    for a in range(n_atoms):
        for b in range(n_atoms):
            for cell_idx in cell_indices:
                rep = (a, b, cell_idx)
                if rep in visited:
                    continue
                if triples is not None and rep not in triples:
                    continue
                orb = AtomPairOrbit(
                    representative=rep,
                    members=[rep],
                    ops=[identity_op_idx],
                )
                visited.add(rep)
                for op_idx in range(len(operations)):
                    R_lat = op_R_lat[op_idx]
                    perm = op_atom_perm[op_idx]
                    shifts = op_atom_shifts[op_idx]
                    a_img = int(perm[a])
                    b_img = int(perm[b])
                    # Effective cell-index transformation for this
                    # atom pair under this operator. Sign-convention
                    # note: ``atom_permutation_under_op`` stores the
                    # shift as ``L^{-1} . (r_{pi(a)} - R . r_a)`` which
                    # is the *opposite* of the "s_a" convention used
                    # in the SYM2c math (where ``R . r_a = r_{pi(a)}
                    # + L . s_a``). So ``shifts[a] = -s_a`` in code,
                    # and the derived formula h' = R.h + s_b - s_a
                    # becomes, in code's shifts:
                    #     h' = R.h + shifts[a] - shifts[b]
                    h_img = tuple(
                        int(x)
                        for x in (R_lat @ np.array(cell_idx) + shifts[a] - shifts[b])
                    )
                    member = (a_img, b_img, h_img)
                    if not _in_space(member):
                        if require_closed:
                            raise ValueError(
                                f"identify_atom_pair_orbits: triple "
                                f"(a={a}, b={b}, h={cell_idx}) has an "
                                f"orbit partner "
                                f"(a'={a_img}, b'={b_img}, h'={h_img}) "
                                f"under operator {op_idx} that is not "
                                f"in the triple space -- the space "
                                f"is not symmetry-closed. Use a "
                                f"group-invariant (pair-resolved) "
                                f"triple set, expand the cell cutoff, "
                                f"or pass require_closed=False."
                            )
                        continue
                    if member == rep or member in visited:
                        continue
                    visited.add(member)
                    orb.members.append(member)
                    orb.ops.append(op_idx)
                orbits.append(orb)

    n_triples = (
        len(triples) if triples is not None else n_atoms * n_atoms * n_cells
    )
    return AtomPairOrbits(
        orbits=orbits,
        n_triples=n_triples,
        n_atoms=n_atoms,
        n_cells=n_cells,
    )


# ---------------------------------------------------------------------------
# Per-atom sub-block extraction
# ---------------------------------------------------------------------------


def _atom_ao_slices(basis: BasisSet) -> List[slice]:
    """Return one slice per atom naming the rows/columns of the full
    AO basis that belong to that atom. Assumes libint's convention
    that shells are ordered by atom, contiguously per atom."""
    shells = list(basis.shells())
    # Map atom index -> list of shell indices.
    shells_by_atom: Dict[int, List[int]] = {}
    for s_idx, sh in enumerate(shells):
        shells_by_atom.setdefault(int(sh.atom_index), []).append(s_idx)
    # Build offsets.
    offsets = np.empty(len(shells) + 1, dtype=int)
    offsets[0] = 0
    for s_idx, sh in enumerate(shells):
        offsets[s_idx + 1] = offsets[s_idx] + (2 * int(sh.l) + 1)
    # Aggregate per-atom range.
    atom_slices: Dict[int, slice] = {}
    for atom_idx, shell_list in shells_by_atom.items():
        start = offsets[shell_list[0]]
        end = offsets[shell_list[-1] + 1]
        atom_slices[atom_idx] = slice(start, end)
    # Order by atom index (which must be 0..n_atoms-1).
    return [atom_slices[a] for a in sorted(atom_slices)]


# ---------------------------------------------------------------------------
# Compression + reconstruction
# ---------------------------------------------------------------------------


def compress_lattice_matrix_set_c(
    lms: LatticeMatrixSet,
    orbits: AtomPairOrbits,
    basis: BasisSet,
) -> List[np.ndarray]:
    """Extract one representative sub-block per atom-pair orbit.

    Returns a list (one entry per orbit, in orbit order) of numpy
    arrays of shape ``(n_AOs_on_source_atom, n_AOs_on_dest_atom)``.
    The orbit representative triple ``(a, b, h)`` indexes the
    (atom_a, atom_b) sub-block at cell ``h``.
    """
    slices = _atom_ao_slices(basis)
    cell_to_slot: Dict[CellIndex, int] = {
        _cell_tuple(c.index): i for i, c in enumerate(lms.cells)
    }

    reps: List[np.ndarray] = []
    for orb in orbits.orbits:
        a, b, h = orb.representative
        cell_slot = cell_to_slot[h]
        full_block = np.asarray(lms.blocks[cell_slot], dtype=float)
        sub = full_block[slices[a], :][:, slices[b]].copy()
        reps.append(sub)
    return reps


def reconstruct_lattice_matrix_set_c(
    reps: Sequence[np.ndarray],
    orbits: AtomPairOrbits,
    basis: BasisSet,
    system: PeriodicSystem,
    operations: Sequence[SymmetryOp],
    *,
    cells: Sequence[LatticeCell],
    into_blocks: Optional[List[np.ndarray]] = None,
) -> List[np.ndarray]:
    """Rebuild the full per-cell block list from orbit representative
    sub-blocks.

    For each orbit member related to its representative by operator
    ``R``, the member sub-block is ``D_a(R) . F^{(a,b)}(rep) . D_b(R)^T``
    where ``D_a(R)`` is the on-atom-a block of the full AO
    permutation matrix. The member sub-block is scattered into the
    right slot of the right full block.

    Parameters
    ----------
    reps
        Output of :func:`compress_lattice_matrix_set_c` -- one
        sub-block per orbit.
    orbits
        The :class:`AtomPairOrbits` partition.
    basis, system, operations
        As for :func:`compress_lattice_matrix_set_c`.
    cells
        The original cell list (``lms.cells``) -- needed to rebuild
        blocks in the same order and shape. Passed explicitly because
        the orbit partition doesn't carry the LatticeCell objects
        themselves.
    into_blocks
        When ``None`` (default), fresh zero blocks are allocated, so
        every sub-block NOT covered by an orbit comes back zero --
        correct when the orbit partition covers the whole triple
        space of ``cells``. When the orbits cover only a SUBSET of
        the caller's triples (a pair-resolved restriction, or a
        caller template wider than the orbit space), pass the
        caller's live blocks (parallel to ``cells``): orbit members
        are scattered in place and uncovered sub-blocks keep their
        existing content.

    Returns
    -------
    List of full ``(n_bf x n_bf)`` blocks, one per cell in ``cells``
    (``into_blocks`` itself when given).
    """
    n_bf = basis.nbasis
    slices = _atom_ao_slices(basis)
    cell_to_slot: Dict[CellIndex, int] = {
        _cell_tuple(c.index): i for i, c in enumerate(cells)
    }

    # Cache full AO permutation matrices once per operator.
    L = np.asarray(system.lattice, dtype=float)
    P_cache: Dict[int, np.ndarray] = {}
    ap_cache: Dict[int, object] = {}

    def _get_P(op_idx: int) -> np.ndarray:
        if op_idx not in P_cache:
            op = operations[op_idx]
            R_cart = lattice_to_cartesian_rotation(op.rotation, L)
            t_cart = L @ np.asarray(op.translation, dtype=float)
            ap = atom_permutation_under_op(system, R_cart, t_cart)
            ap_cache[op_idx] = ap
            P_cache[op_idx] = build_ao_permutation_matrix(basis, R_cart, ap)
        return P_cache[op_idx]

    # Initialise full blocks to zero (or scatter into the caller's).
    if into_blocks is not None:
        if len(into_blocks) != len(cells):
            raise ValueError(
                f"reconstruct_lattice_matrix_set_c: into_blocks has "
                f"{len(into_blocks)} entries for {len(cells)} cells."
            )
        blocks = into_blocks
    else:
        blocks = [np.zeros((n_bf, n_bf), dtype=float) for _ in range(len(cells))]

    for orb_idx, orb in enumerate(orbits.orbits):
        rep_sub = np.asarray(reps[orb_idx], dtype=float)
        a_rep, b_rep, _ = orb.representative
        for member, op_idx in zip(orb.members, orb.ops):
            a_mem, b_mem, h_mem = member
            P = _get_P(op_idx)
            # Extract D_a = P[slices[a_mem], slices[a_rep]], and
            # D_b = P[slices[b_mem], slices[b_rep]]. These are the
            # on-atom Wigner-D blocks that rotate the sub-block from
            # (a_rep -> a_mem, b_rep -> b_mem).
            D_a = P[slices[a_mem], :][:, slices[a_rep]]
            D_b = P[slices[b_mem], :][:, slices[b_rep]]
            member_sub = D_a @ rep_sub @ D_b.T
            # Place into the correct cell block at (a_mem, b_mem) slot.
            cell_slot = cell_to_slot[h_mem]
            blocks[cell_slot][slices[a_mem], slices[b_mem]] = member_sub

    return blocks
