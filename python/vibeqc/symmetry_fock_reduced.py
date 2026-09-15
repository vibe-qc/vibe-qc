"""Phase M3b: symmetry-reduced two-electron Fock build (Gamma-only).

Uses the per-cell-pair J/K kernel to evaluate ERIs only at
orbit-representative (c_g, c_p) pairs, then scatters via Wigner-D
to all orbit members and accumulates into J(Γ) and K(Γ).

Activation is explicit -- the standard build_jk_gamma_molecular_limit
path is unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    LatticeCell,
    LatticeSumOptions,
    PeriodicSystem,
    SymmetryOp,
    build_jk_pair_contributions,
    direct_lattice_cells,
)
from .symmetry_integrals import symmorphic_operations
from .symmetry_scf import build_ao_permutation_cache

__all__ = [
    "compute_jk_gamma_reduced",
]


def _cell_index_to_slot(cells: Sequence[LatticeCell]) -> dict:
    """Map (i,j,k) tuple -> position in cell list."""
    return {
        (int(c.index[0]), int(c.index[1]), int(c.index[2])): slot
        for slot, c in enumerate(cells)
    }


def compute_jk_gamma_reduced(
    basis: BasisSet,
    system: PeriodicSystem,
    opts: LatticeSumOptions,
    D_gamma: np.ndarray,
    operations: Sequence[SymmetryOp],
    *,
    omega: float = 0.0,
    transform_density: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build J(Γ) and K(Γ) with symmetry-reduced cell pairs.

    Identifies orbit representatives of (c_g, c_p) cell pairs under
    the space group, computes J/K contributions only at those pairs
    via ``build_jk_pair_contributions``, then scatters each member's
    contribution via Wigner-D rotation to accumulate full J and K.

    Parameters
    ----------
    basis, system, opts
        As for build_jk_gamma_molecular_limit.
    D_gamma
        Γ-only density matrix (nbf, nbf).
    operations
        Symmetry operators. Non-symmorphic ops are filtered out.
    omega
        Ewald screening parameter (0 = full Coulomb).
    transform_density
        If True, symmetrize the density before the Fock build
        by group-averaging. This improves accuracy of the
        Wigner-D reconstruction.

    Returns
    -------
    (J_gamma, K_gamma) as (nbf, nbf) numpy arrays.
    """
    from .symmetry_scf import symmetrize_matrix as _sym

    nbf = basis.nbasis
    D = np.asarray(D_gamma, dtype=float)
    if transform_density:
        P_cache_d = build_ao_permutation_cache(system, basis, operations)
        D = _sym(D, P_cache_d)

    sym_ops = symmorphic_operations(operations)
    full_cells = list(direct_lattice_cells(system, opts.cutoff_bohr))
    n_cells = len(full_cells)
    cell_to_slot = _cell_index_to_slot(full_cells)
    P_cache = build_ao_permutation_cache(system, basis, sym_ops)

    # Build all (c_g, c_p) pairs
    all_pairs = [(g, p) for g in range(n_cells) for p in range(n_cells)]

    # Identify orbit representatives: two pairs are equivalent if there
    # exists an operator R such that R.cells[g] = cells[g'] and
    # R.cells[p] = cells[p'].
    visited: set = set()
    rep_pairs: List[Tuple[int, int]] = []
    pair_to_rep: dict = {}  # (g, p) -> (g_rep, p_rep)
    pair_to_op: dict = {}  # (g, p) -> op_idx

    # Identity operator index
    id_op = next(
        i
        for i, op in enumerate(sym_ops)
        if np.array_equal(np.asarray(op.rotation, dtype=int), np.eye(3, dtype=int))
    )

    for g, p in all_pairs:
        if (g, p) in visited:
            continue
        rep_pairs.append((g, p))
        pair_to_rep[(g, p)] = (g, p)
        pair_to_op[(g, p)] = id_op
        visited.add((g, p))

        for op_idx, op in enumerate(sym_ops):
            if op_idx == id_op:
                continue
            R = np.asarray(op.rotation, dtype=int)
            g_idx = tuple(int(x) for x in (R @ full_cells[g].index))
            p_idx = tuple(int(x) for x in (R @ full_cells[p].index))
            if g_idx in cell_to_slot and p_idx in cell_to_slot:
                g_img = cell_to_slot[g_idx]
                p_img = cell_to_slot[p_idx]
                if (g_img, p_img) not in visited:
                    visited.add((g_img, p_img))
                    pair_to_rep[(g_img, p_img)] = (g, p)
                    pair_to_op[(g_img, p_img)] = op_idx

    # Compute J/K contributions at representative pairs only
    contribs = build_jk_pair_contributions(
        basis,
        system,
        full_cells,
        rep_pairs,
        opts,
        D,
        omega,
    )

    # Build rep lookup: (g, p) -> J_contrib, K_contrib
    rep_data: dict = {}
    for entry in contribs:
        rep_data[(entry.c_g, entry.c_p)] = (
            np.asarray(entry.J_contrib, dtype=float),
            np.asarray(entry.K_contrib, dtype=float),
        )

    # Scatter via Wigner-D and accumulate
    J_gamma = np.zeros((nbf, nbf), dtype=float)
    K_gamma = np.zeros((nbf, nbf), dtype=float)

    for g, p in all_pairs:
        g_rep, p_rep = pair_to_rep[(g, p)]
        op_idx = pair_to_op[(g, p)]
        J_rep, K_rep = rep_data[(g_rep, p_rep)]
        P = P_cache[op_idx]
        J_gamma += P @ J_rep @ P.T
        K_gamma += P @ K_rep @ P.T

    return J_gamma, K_gamma
