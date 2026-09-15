"""Tests for the MOM occupied-subspace selector.

Pure-numpy; no vibeqc-core import needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from vibeqc.mom import select_occupied_by_max_overlap


def _build_random_S(n_bf: int, seed: int = 0) -> np.ndarray:
    """Hermitian PSD overlap matrix (small noise added so it's not I)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_bf, n_bf))
    S = A @ A.T + np.eye(n_bf)
    return 0.5 * (S + S.T)


def _S_orthonormal_columns(S: np.ndarray, n: int, seed: int = 1) -> np.ndarray:
    """Generate n S-orthonormal columns: solve via S^{1/2} Q."""
    rng = np.random.default_rng(seed)
    n_bf = S.shape[0]
    # S^{1/2}
    eigvals, eigvecs = np.linalg.eigh(S)
    S_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    # Random orthonormal Q (Euclidean) — n_bf x n.
    A = rng.standard_normal((n_bf, n))
    Q, _ = np.linalg.qr(A)
    # C = S^{-1/2} Q has columns S-orthonormal: C^T S C = Q^T Q = I.
    return S_inv_sqrt @ Q


def test_aufbau_match_when_orbitals_unchanged():
    """If C_new == C_prev (no rotation), MOM should pick [0, 1, ..., n_occ-1]
    just like Aufbau."""
    n_bf, n_occ = 6, 2
    S = _build_random_S(n_bf, seed=42)
    C = _S_orthonormal_columns(S, n_bf, seed=43)
    eps = np.array([-1.0, -0.5, 0.1, 0.3, 0.4, 0.5])
    sel = select_occupied_by_max_overlap(
        C_new=C, S_k=S, C_prev_occ=C[:, :n_occ], n_occ=n_occ, eps_new=eps,
    )
    assert sorted(sel.tolist()) == [0, 1]


def test_picks_overlap_partner_over_energy_when_subspace_rotates():
    """Construct C_new so the LOWEST-energy MOs do NOT overlap with
    C_prev_occ — MOM must pick the higher-energy MOs that do."""
    n_bf, n_occ = 4, 2
    S = np.eye(n_bf)
    # C_prev_occ = [e0, e1]; C_new[:, 0:2] are virtual-like (e2, e3)
    # but C_new[:, 2:4] is the rotated occupied pair.
    C_new = np.eye(n_bf)[:, [2, 3, 0, 1]]
    C_prev_occ = np.eye(n_bf)[:, :2]
    # Energies: lowest are C_new[:, 0:2], but they don't overlap.
    eps = np.array([-1.0, -0.5, 0.1, 0.2])
    sel = select_occupied_by_max_overlap(
        C_new=C_new, S_k=S, C_prev_occ=C_prev_occ, n_occ=n_occ, eps_new=eps,
    )
    # Aufbau would pick [0, 1]; MOM picks [2, 3] because those overlap.
    assert sorted(sel.tolist()) == [2, 3]


def test_handles_complex_overlap_matrix():
    """Per-k S(k) is complex Hermitian (Bloch-summed); confirm MOM
    still works with complex C_new / S_k. Uses S-orthonormal C_new
    so the diagonal of C_prev_occ^† S C_new dominates and the
    overlap-max picks the matching column unambiguously."""
    n_bf, n_occ = 4, 1
    # Hermitian S with imaginary off-diagonal entries.
    S = np.eye(n_bf, dtype=complex)
    S[0, 1] = 0.1j
    S[1, 0] = -0.1j
    # S-orthonormalise random complex orbitals: C^† S C = I.
    rng = np.random.default_rng(7)
    A = (rng.standard_normal((n_bf, n_bf))
         + 1j * rng.standard_normal((n_bf, n_bf)))
    # Cholesky of S, then Q = qr(L^{-1} A) gives S-orth C = L^{-1} Q.
    eigvals, eigvecs = np.linalg.eigh(S)
    S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T
    # Euclidean-orthonormalise then map back to S-orthonormal.
    Q, _ = np.linalg.qr(A)
    C_new = S_inv_sqrt @ Q
    # Confirm S-orthonormality (sanity check the setup, not MOM).
    I_check = C_new.conj().T @ S @ C_new
    assert np.allclose(I_check, np.eye(n_bf), atol=1e-10)
    C_prev_occ = C_new[:, 1:2].copy()
    eps = np.array([1.0, 0.0, 2.0, 3.0])  # column 1 is NOT the lowest
    sel = select_occupied_by_max_overlap(
        C_new=C_new, S_k=S, C_prev_occ=C_prev_occ, n_occ=n_occ, eps_new=eps,
    )
    # Best-overlap column with C_prev_occ (=C_new[:, 1]) is column 1
    # (S-orthonormal C_new ⇒ off-diagonal projections are zero).
    assert sel.tolist() == [1]


def test_within_selection_sorted_by_energy_when_eps_provided():
    """Selected indices should be returned in *energy* order when
    eps_new is supplied — preserves the canonical occupied-MO layout."""
    n_bf, n_occ = 4, 2
    S = np.eye(n_bf)
    # Pick columns 3 and 0 by overlap; expected return [0, 3] (energy
    # order) since eps[0] < eps[3].
    C_new = np.eye(n_bf)
    C_prev_occ = np.column_stack([C_new[:, 3], C_new[:, 0]])
    eps = np.array([-1.0, 5.0, 6.0, -0.5])
    sel = select_occupied_by_max_overlap(
        C_new=C_new, S_k=S, C_prev_occ=C_prev_occ, n_occ=n_occ, eps_new=eps,
    )
    assert sel.tolist() == [0, 3]


def test_rejects_mismatched_n_occ():
    n_bf = 4
    S = np.eye(n_bf)
    C = np.eye(n_bf)
    with pytest.raises(ValueError, match="C_prev_occ has 2 columns but n_occ=3"):
        select_occupied_by_max_overlap(
            C_new=C, S_k=S, C_prev_occ=C[:, :2], n_occ=3,
        )


def test_rejects_too_few_C_new_columns():
    n_bf = 2
    S = np.eye(n_bf)
    C_new = np.eye(n_bf)
    C_prev_occ = np.eye(4)[:, :3]  # mismatched on purpose
    # Trigger the n_occ check via deliberate inconsistency (C_prev has 3
    # cols, but C_new has only 2). The first check (C_prev mismatch) won't
    # fire if n_occ matches; here we make n_occ=3 to hit C_new check.
    C_prev_occ_2 = np.zeros((n_bf, 3))
    with pytest.raises(ValueError, match="C_new has only"):
        select_occupied_by_max_overlap(
            C_new=C_new, S_k=S, C_prev_occ=C_prev_occ_2, n_occ=3,
        )
