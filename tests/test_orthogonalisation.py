"""Tests for ``vibeqc.orthogonalisation`` (item 5).

Three methods exposed as a public API for users who explicitly want
to override the default critical-severity abort and run SCF on a
near-singular overlap with a curated truncation:

1. Canonical orthogonalisation (Löwdin) — eigenvalue-truncating.
2. Pivoted Cholesky (Lehtola) — pivoted Cholesky decomposition,
   numerically robust at extreme overcompleteness.
3. Symmetric (S^{-1/2}) — non-truncating, for well-conditioned S.

The dispatch entry point :func:`vq.orthogonalise_overlap` lets the
user pick a method (or use ``method='auto'`` to graceful-degrade
from canonical to Cholesky based on cond(S)).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Method 1: canonical
# ---------------------------------------------------------------------------

def test_canonical_orth_satisfies_xtsx_identity():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    S = A @ A.T + 0.1 * np.eye(6)
    X, info = vq.canonical_orth(S, threshold=1e-7)
    assert info.method == "canonical"
    assert info.n_basis == 6
    assert info.n_kept == 6
    assert np.allclose(X.T @ S @ X, np.eye(6), atol=1e-10)


def test_canonical_orth_drops_near_singular():
    """A genuinely near-singular direction (off-diagonal coupling)
    is dropped at default threshold."""
    S = np.array([
        [1.0,        1.0 - 1e-10, 0.0],
        [1.0 - 1e-10, 1.0,        0.0],
        [0.0,        0.0,        1.0],
    ])
    X, info = vq.canonical_orth(S, threshold=1e-7)
    assert info.n_kept == 2
    assert np.allclose(X.T @ S @ X, np.eye(2), atol=1e-8)


def test_canonical_orth_records_normalize_diag_in_info():
    S = np.diag([1.0, 1.0, 1.0])
    _, info_default = vq.canonical_orth(S)
    _, info_off = vq.canonical_orth(S, normalize_diag_first=False)
    assert info_default.normalize_diag_first is True
    assert info_off.normalize_diag_first is False


# ---------------------------------------------------------------------------
# Method 2: pivoted Cholesky
# ---------------------------------------------------------------------------

def test_pivoted_cholesky_orth_full_rank_case():
    """Well-conditioned S → Cholesky keeps every basis function."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 5))
    S = A @ A.T + 0.5 * np.eye(5)
    X, info = vq.pivoted_cholesky_orth(S)
    assert info.method == "cholesky"
    assert info.n_kept == 5
    assert info.cholesky_indices is not None
    assert sorted(info.cholesky_indices.tolist()) == [0, 1, 2, 3, 4]
    assert np.allclose(X.T @ S @ X, np.eye(5), atol=1e-10)


def test_pivoted_cholesky_drops_redundant_basis_function():
    """Two near-identical basis functions (rows 0 and 1 of S are
    nearly equal) → Cholesky should keep one of them and drop the
    other. The kept indices are a strict subset."""
    # Build S where AO 0 and AO 1 overlap almost perfectly.
    S = np.array([
        [1.0, 0.99999999, 0.0, 0.0],
        [0.99999999, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.5, 1.0],
    ])
    X, info = vq.pivoted_cholesky_orth(S, cholesky_threshold=1e-7)
    assert info.n_kept == 3
    # Exactly one of {0, 1} is kept.
    kept = set(info.cholesky_indices.tolist())
    assert (0 in kept) ^ (1 in kept)   # XOR
    assert 2 in kept and 3 in kept


def test_pivoted_cholesky_xtsx_identity():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((4, 4))
    S = A @ A.T + 0.1 * np.eye(4)
    X, _ = vq.pivoted_cholesky_orth(S)
    assert np.allclose(X.T @ S @ X, np.eye(X.shape[1]), atol=1e-9)


# ---------------------------------------------------------------------------
# Method 3: symmetric
# ---------------------------------------------------------------------------

def test_symmetric_orth_psd_case():
    """S^{-1/2} of a well-conditioned PSD matrix gives X^T S X = I."""
    rng = np.random.default_rng(3)
    A = rng.standard_normal((4, 4))
    S = A @ A.T + 0.5 * np.eye(4)
    X, info = vq.symmetric_orth(S)
    assert info.method == "symmetric"
    assert info.n_kept == 4
    assert np.allclose(X.T @ S @ X, np.eye(4), atol=1e-10)


def test_symmetric_orth_rejects_non_psd():
    """Negative eigenvalues → reject (refuse to compute S^{-1/2})."""
    S = np.diag([-0.1, 1.0, 1.0])
    with pytest.raises(ValueError, match="positive-definite"):
        vq.symmetric_orth(S)


# ---------------------------------------------------------------------------
# Dispatch: orthogonalise_overlap
# ---------------------------------------------------------------------------

def test_orthogonalise_overlap_method_dispatch():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((4, 4))
    S = A @ A.T + 0.5 * np.eye(4)

    _, info_can = vq.orthogonalise_overlap(S, method="canonical")
    _, info_chol = vq.orthogonalise_overlap(S, method="cholesky")
    _, info_sym = vq.orthogonalise_overlap(S, method="symmetric")

    assert info_can.method == "canonical"
    assert info_chol.method == "cholesky"
    assert info_sym.method == "symmetric"


def test_orthogonalise_overlap_auto_picks_canonical_when_well_conditioned():
    rng = np.random.default_rng(5)
    A = rng.standard_normal((4, 4))
    S = A @ A.T + 0.5 * np.eye(4)
    _, info = vq.orthogonalise_overlap(S, method="auto")
    assert info.method == "canonical"


def test_orthogonalise_overlap_auto_picks_cholesky_when_extreme_cond():
    """At cond(S) ≫ 1e15, auto switches to pivoted Cholesky."""
    # Force extreme cond by including a 1e-20 eigenvalue.
    eigs = np.array([1e-20, 1.0, 1.0])
    Q, _ = np.linalg.qr(np.random.default_rng(6).standard_normal((3, 3)))
    S = Q @ np.diag(eigs) @ Q.T
    _, info = vq.orthogonalise_overlap(
        S, method="auto", cholesky_trigger_cond=1e15,
    )
    assert info.method == "cholesky"


def test_orthogonalise_overlap_unknown_method_raises():
    S = np.eye(2)
    with pytest.raises(ValueError, match="unknown method"):
        vq.orthogonalise_overlap(S, method="hocus-pocus")


# ---------------------------------------------------------------------------
# Cross-method consistency
# ---------------------------------------------------------------------------

def test_canonical_and_cholesky_agree_on_well_conditioned_basis():
    """Both methods should produce X with X^T S X = I, with the same
    n_kept on a well-conditioned PSD S (no truncation needed for either)."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((5, 5))
    S = A @ A.T + 0.5 * np.eye(5)

    X_can, info_can = vq.canonical_orth(S)
    X_chol, info_chol = vq.pivoted_cholesky_orth(S)

    assert info_can.n_kept == info_chol.n_kept == 5
    assert np.allclose(X_can.T @ S @ X_can, np.eye(5), atol=1e-9)
    assert np.allclose(X_chol.T @ S @ X_chol, np.eye(5), atol=1e-9)


# ---------------------------------------------------------------------------
# OrthogonalisationInfo content
# ---------------------------------------------------------------------------

def test_orthogonalisation_info_records_relevant_fields():
    rng = np.random.default_rng(8)
    A = rng.standard_normal((4, 4))
    S = A @ A.T + 0.5 * np.eye(4)
    _, info = vq.canonical_orth(S, threshold=1e-7)
    assert info.n_basis == 4
    assert info.n_kept == 4
    assert info.threshold == pytest.approx(1e-7)
    assert info.condition_number > 0
    assert np.isfinite(info.condition_number)
    assert info.cholesky_indices is None   # canonical doesn't set this


def test_pivoted_cholesky_info_records_kept_indices():
    rng = np.random.default_rng(9)
    A = rng.standard_normal((4, 4))
    S = A @ A.T + 0.5 * np.eye(4)
    _, info = vq.pivoted_cholesky_orth(S)
    assert info.cholesky_indices is not None
    assert len(info.cholesky_indices) == info.n_kept
