"""Pair-resolved (group-invariant, CRYSTAL-style) truncation domains.

The BIPOLE direct-space builders historically truncate every lattice
sum with one radial ball on CELL ORIGINS (``direct_lattice_cells``).
That ball is closed under the bare point-group action ``g -> R.g``,
but the SYM3b symmetry machinery acts on atom-pair triples
``(a, b, h) -> (perm[a], perm[b], R.h + s_a - s_b)`` whose per-atom
lattice shifts (up to 13.78 bohr on MgO) push orbit partners of EVERY
cell outside the radial domain -- 100% of MgO cells are orbit-open at
production cutoffs, cell-list expansion provably never closes them,
and the resulting truncation asymmetry is the ~4e-2 Ha SYM3b
enforcement shift that keeps ``use_fock_symmetry*`` opt-in
(HANDOVER_BIPOLE_PRODUCTION.md Sec. 0a, 2026-07-12).

The fix is to truncate by the PAIR criterion CRYSTAL uses:

    (a, b, h) is kept  <=>  |r_b + L.h - r_a| <= cutoff

which the group action preserves exactly (it is a distance between the
two atoms of the pair), so the triple space is group-invariant by
construction and orbit closure holds with ``require_closed=True``.

This module builds that domain:

* :class:`PairResolvedDomain` -- the kept cells and the qualifying
  atom pairs per cell.
* :func:`pair_resolved_domain` -- construct it, optionally
  orbit-closing against a symmorphic operator set (protects the
  boundary from floating-point ties: an orbit is kept iff ANY member
  qualifies, so membership never depends on which member's distance
  was rounded across the cutoff).
* :func:`domain_triples` -- the set of triples, in the
  ``identify_atom_pair_orbits(triples=...)`` format.
* :func:`atom_pair_shell_masks` -- per-cell nshells x nshells uint8
  masks selecting the qualifying pairs, in the
  ``build_jk_2e_real_space_domains`` / ``_output_subset_masked``
  format.

Geometry note: ``|L.h| <= cutoff + |r_b - r_a| <= cutoff + pair_span``
bounds every kept cell, so candidates are enumerated on the radial
ball at ``cutoff + pair_span`` and filtered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    LatticeCell,
    PeriodicSystem,
    SymmetryOp,
    direct_lattice_cells,
)
from .symmetry_lattice_c import operator_triple_actions

__all__ = [
    "PairResolvedDomain",
    "pair_resolved_domain",
    "domain_triples",
    "atom_pair_shell_masks",
    "atom_pair_span",
    "mask_lattice_blocks_to_domain",
]

CellIndex = Tuple[int, int, int]
Triple = Tuple[int, int, CellIndex]


def _cell_tuple(idx) -> CellIndex:
    arr = np.asarray(idx, dtype=int)
    return (int(arr[0]), int(arr[1]), int(arr[2]))


@dataclass(frozen=True)
class PairResolvedDomain:
    """The pair-resolved truncation domain of one periodic system.

    ``cells`` is ordered by |r_cart| ascending (home cell first --
    the ``direct_lattice_cells`` convention), restricted to cells
    hosting at least one qualifying pair. ``pairs_by_cell`` is
    parallel to ``cells``; entry ``i`` is the frozenset of qualifying
    ``(source_atom, dest_atom)`` pairs of ``cells[i]``.
    """

    cutoff_bohr: float
    cells: Tuple[LatticeCell, ...]
    pairs_by_cell: Tuple[FrozenSet[Tuple[int, int]], ...]

    @property
    def n_triples(self) -> int:
        return sum(len(p) for p in self.pairs_by_cell)

    def cell_keys(self) -> List[CellIndex]:
        return [_cell_tuple(c.index) for c in self.cells]


def atom_pair_span(system: PeriodicSystem) -> float:
    """``max_{a,b} |r_b - r_a|`` over the home-cell atoms (bohr).

    The kept-cell bound: every pair-qualifying cell origin satisfies
    ``|L.h| <= cutoff + atom_pair_span(system)``.
    """
    pos = [
        np.asarray(atom.xyz, dtype=float)
        for atom in system.unit_cell_molecule().atoms
    ]
    span = 0.0
    for i in range(len(pos)):
        for j in range(len(pos)):
            span = max(span, float(np.linalg.norm(pos[j] - pos[i])))
    return span


def pair_resolved_domain(
    system: PeriodicSystem,
    cutoff_bohr: float,
    *,
    operations: Optional[Sequence[SymmetryOp]] = None,
) -> PairResolvedDomain:
    """Build the pair-resolved domain ``{(a,b,h): |r_b + L.h - r_a| <= cutoff}``.

    With ``operations`` (a symmorphic operator set, e.g.
    ``symmorphic_operations(system.symmetry.operations)``), the set is
    additionally closed under the triple action by UNION: an orbit is
    kept iff any member's distance qualifies. Mathematically the
    action preserves the pair distance, so this only matters for
    floating-point ties exactly at the boundary -- but it makes the
    group-invariance of the returned domain exact by construction
    rather than up to rounding.
    """
    cutoff = float(cutoff_bohr)
    if not (cutoff > 0.0):
        raise ValueError(
            f"pair_resolved_domain: cutoff_bohr must be positive; got "
            f"{cutoff_bohr!r}"
        )
    pos = [
        np.asarray(atom.xyz, dtype=float)
        for atom in system.unit_cell_molecule().atoms
    ]
    n_atoms = len(pos)
    span = atom_pair_span(system)
    # Tiny slack so a candidate whose |L.h| rounds a hair past the
    # bound is still enumerated (the pair criterion decides).
    candidates = list(direct_lattice_cells(system, cutoff + span + 1e-9))

    cutoff_sq = cutoff * cutoff
    pairs_by_key: Dict[CellIndex, Set[Tuple[int, int]]] = {}
    for cell in candidates:
        key = _cell_tuple(cell.index)
        r_cell = np.asarray(cell.r_cart, dtype=float)
        kept: Set[Tuple[int, int]] = set()
        for a in range(n_atoms):
            for b in range(n_atoms):
                d = pos[b] + r_cell - pos[a]
                if float(d @ d) <= cutoff_sq:
                    kept.add((a, b))
        if kept:
            pairs_by_key[key] = kept

    if operations:
        op_R_lat, op_perm, op_shifts, _ = operator_triple_actions(
            operations, system
        )
        # Union closure: image triples join the domain. One sweep over
        # the current triples per operator suffices for a group (the
        # image of a kept triple under any op has the same pair
        # distance, so it is already kept up to boundary rounding;
        # iterate to a fixed point anyway -- the set is finite and
        # grows monotonically, and real inputs converge immediately).
        changed = True
        while changed:
            changed = False
            current = [
                (a, b, key)
                for key, pairs in pairs_by_key.items()
                for (a, b) in pairs
            ]
            for a, b, key in current:
                h = np.array(key, dtype=int)
                for op_idx in range(len(op_R_lat)):
                    a_img = int(op_perm[op_idx][a])
                    b_img = int(op_perm[op_idx][b])
                    h_img = _cell_tuple(
                        op_R_lat[op_idx] @ h
                        + op_shifts[op_idx][a]
                        - op_shifts[op_idx][b]
                    )
                    kept = pairs_by_key.setdefault(h_img, set())
                    if (a_img, b_img) not in kept:
                        kept.add((a_img, b_img))
                        changed = True

    # Order kept cells by the direct_lattice_cells convention
    # (|r| ascending, home first). Cells that joined only through
    # orbit closure still lie inside the candidate ball (the action
    # preserves pair distances), so `candidates` covers every key.
    candidate_keys = {_cell_tuple(c.index) for c in candidates}
    missing = set(pairs_by_key) - candidate_keys
    if missing:
        raise RuntimeError(
            f"pair_resolved_domain: orbit closure produced cells outside "
            f"the candidate ball ({sorted(missing)[:4]} ...) -- the "
            f"operator set does not preserve pair distances. This is an "
            f"internal invariant violation, not a user error."
        )
    cells_kept: List[LatticeCell] = []
    pairs_kept: List[FrozenSet[Tuple[int, int]]] = []
    for cell in candidates:
        key = _cell_tuple(cell.index)
        if key in pairs_by_key:
            cells_kept.append(cell)
            pairs_kept.append(frozenset(pairs_by_key[key]))
    return PairResolvedDomain(
        cutoff_bohr=cutoff,
        cells=tuple(cells_kept),
        pairs_by_cell=tuple(pairs_kept),
    )


def domain_triples(domain: PairResolvedDomain) -> Set[Triple]:
    """The domain as a set of ``(a, b, cell_tuple)`` triples -- the
    ``identify_atom_pair_orbits(triples=...)`` restriction format."""
    out: Set[Triple] = set()
    for cell, pairs in zip(domain.cells, domain.pairs_by_cell):
        key = _cell_tuple(cell.index)
        for a, b in pairs:
            out.add((a, b, key))
    return out


def mask_lattice_blocks_to_domain(
    basis: BasisSet,
    domain: PairResolvedDomain,
    lms,
) -> None:
    """Zero the non-qualifying (a, b) sub-blocks of a LatticeMatrixSet.

    M3 (pair-resolved density support): the direct SR tensor density's
    triple set must be group-invariant for J/K blocks to be orbit-symmetric.
    The Fock builder therefore applies this operation to a private copy of
    the physical SCF density, using the same criterion as CRYSTAL's ITOL-style
    pair screening. Mutates ``lms`` in place via ``set_block``. ``lms.cells``
    must be parallel to ``domain.cells`` (checked by key; raises on mismatch).
    """
    from .symmetry_lattice_c import _atom_ao_slices

    if len(lms.cells) != len(domain.cells):
        raise ValueError(
            f"mask_lattice_blocks_to_domain: {len(lms.cells)} blocks for "
            f"{len(domain.cells)} domain cells."
        )
    for got, want in zip(lms.cells, domain.cells):
        if _cell_tuple(got.index) != _cell_tuple(want.index):
            raise ValueError(
                f"mask_lattice_blocks_to_domain: cell order mismatch at "
                f"{_cell_tuple(got.index)} vs {_cell_tuple(want.index)} -- "
                f"the density must be built on domain.cells."
            )
    slices = _atom_ao_slices(basis)
    n_atoms = len(slices)
    all_pairs = n_atoms * n_atoms
    for i, pairs in enumerate(domain.pairs_by_cell):
        if len(pairs) == all_pairs:
            continue
        block = np.array(lms.blocks[i], dtype=float, copy=True)
        keep = np.zeros_like(block, dtype=bool)
        for a, b in pairs:
            keep[slices[a], slices[b]] = True
        block[~keep] = 0.0
        lms.set_block(i, block)


def atom_pair_shell_masks(
    basis: BasisSet,
    pairs_by_cell: Sequence[FrozenSet[Tuple[int, int]]],
) -> List[np.ndarray]:
    """Per-cell shell-pair masks selecting the given atom pairs.

    One ``(n_shells * n_shells,)`` uint8 row-major array per entry of
    ``pairs_by_cell`` -- the ``output_shell_masks`` format of
    ``build_jk_2e_real_space_domains`` /
    ``build_jk_2e_real_space_output_subset_masked``.
    """
    shells_by_atom: Dict[int, List[int]] = {}
    n_shells = 0
    for s_idx, sh in enumerate(basis.shells()):
        shells_by_atom.setdefault(int(sh.atom_index), []).append(s_idx)
        n_shells += 1
    masks: List[np.ndarray] = []
    for pairs in pairs_by_cell:
        mask = np.zeros(n_shells * n_shells, dtype=np.uint8)
        for a, b in pairs:
            for s1 in shells_by_atom.get(a, []):
                base = s1 * n_shells
                for s2 in shells_by_atom.get(b, []):
                    mask[base + s2] = 1
        masks.append(mask)
    return masks
