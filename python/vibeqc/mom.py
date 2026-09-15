"""Maximum-Overlap Method (MOM) for orbital tracking across SCF iters.

Standard SCF picks the occupied subspace at iter k by Aufbau (lowest
``n_occ`` MO energies). On systems with near-degenerate band edges,
ionic SCFs starting from a poor initial guess, and other regimes
where the energy ordering at one iter doesn't match the converged
ordering, plain Aufbau can carry the SCF into a non-physical stable
basin (per ``examples/regression/crystal_parity/diag_lih_multik_iter3_*``
on this branch -- LiH primitive at kmesh=(2,2,2) over-binds to
E_total = -228 Ha because iter-2 diagonalisation rotates the
occupied subspace away from the SAD-derived one even though the
iter-1 gradient is small).

MOM instead picks the occupied subspace at iter k by maximum
overlap with the previous iter's occupied subspace:

    O_{ij}(k) = <C_prev_occ[:, i] | S(k) | C_new[:, j]>
    proj_j    = S_i |O_{ij}(k)|^2
    sel       = argsort(-proj_j)[:n_occ]    # top-n_occ overlaps
    C_occ_k   = C_new[:, sel]                # re-sorted within by energy

Greedy max-overlap selection (not Hungarian assignment); fine in
practice because the occupied subspace's projection onto C_new
collapses cleanly onto n_occ "good" columns once the SCF settles.

Reference:
  - Gilbert, Besley, Gill, *J. Phys. Chem. A* 112, 13164 (2008)
    ("Self-consistent field calculations of excited states using
    the maximum overlap method").
  - Barca, Gilbert, Gill, *J. Chem. Theory Comput.* 14, 1501 (2018).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def select_occupied_by_max_overlap(
    C_new: np.ndarray,
    S_k: np.ndarray,
    C_prev_occ: np.ndarray,
    n_occ: int,
    *,
    eps_new: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pick the ``n_occ`` columns of ``C_new`` that have the largest
    summed S(k)-overlap with ``C_prev_occ``'s columns.

    Parameters
    ----------
    C_new
        Per-k MO coefficients freshly diagonalised at this iter.
        Shape ``(n_bf, n_kept)`` with ``n_kept >= n_occ``.
    S_k
        Per-k overlap matrix, ``(n_bf, n_bf)`` Hermitian.
    C_prev_occ
        Previous iter's occupied MOs, ``(n_bf, n_occ)``.
    n_occ
        Number of occupied orbitals to select.
    eps_new
        Optional per-k MO energies; when supplied, the selected
        ``n_occ`` indices are sorted within by ``eps_new`` so the
        canonical "occupied-first, sorted by energy" convention
        is preserved for callers that depend on it.

    Returns
    -------
    Integer array of shape ``(n_occ,)``: column indices into
    ``C_new`` selecting the new occupied subspace.
    """
    if C_prev_occ.shape[1] != n_occ:
        raise ValueError(
            f"MOM: C_prev_occ has {C_prev_occ.shape[1]} columns but "
            f"n_occ={n_occ}"
        )
    if C_new.shape[1] < n_occ:
        raise ValueError(
            f"MOM: C_new has only {C_new.shape[1]} columns but "
            f"n_occ={n_occ}"
        )
    # O has shape (n_occ, n_kept); column j gives the projection of
    # C_new[:, j] onto each previous occupied orbital.
    O = C_prev_occ.conj().T @ S_k @ C_new
    proj = np.sum(np.abs(O) ** 2, axis=0)  # shape (n_kept,)
    # Top-n_occ by overlap, descending.
    sel = np.argsort(-proj)[:n_occ]
    if eps_new is not None:
        order = np.argsort(np.real(eps_new[sel]))
        sel = sel[order]
    else:
        sel = np.sort(sel)
    return sel.astype(int)


def reorder_occupied_by_max_overlap(
    C_new: np.ndarray,
    eps_new: np.ndarray,
    S_k: np.ndarray,
    C_prev_occ: np.ndarray,
    n_occ: int,
) -> tuple:
    """Permute ``C_new`` / ``eps_new`` so the ``n_occ`` MOM-selected occupied
    columns (max overlap with ``C_prev_occ``) come first -- then a column-order
    aufbau fill (``occ[:n_occ] = 1``) picks up the held pattern unchanged.

    Python mirror of the C++ ``mom_reorder_occupied`` (cpp/include/vibeqc/
    mom.hpp), used by the SPINLOCK PATTERN_HOLD: for the first
    ``spinlock_iterations`` cycles the occupied subspace is held by maximum
    overlap with the previous cycle rather than pure aufbau, protecting a
    broken-symmetry (ATOMSPIN) seed from collapsing to the symmetric solution,
    then released. Returns ``(C_perm, eps_perm)``; a no-op on degenerate inputs.
    """
    n_kept = int(C_new.shape[1])
    if n_occ <= 0 or C_prev_occ.shape[1] != n_occ or n_kept < n_occ:
        return C_new, eps_new
    sel = select_occupied_by_max_overlap(
        C_new, S_k, C_prev_occ, n_occ, eps_new=eps_new)
    occ_set = set(int(j) for j in sel)
    order = list(int(j) for j in sel) + [
        j for j in range(n_kept) if j not in occ_set
    ]
    idx = np.asarray(order, dtype=int)
    return C_new[:, idx], eps_new[idx]


__all__ = [
    "select_occupied_by_max_overlap",
    "reorder_occupied_by_max_overlap",
]
