"""Tests for the v0.7 sqrt(diag(S)) normalisation in canonical orth.

PySCF's ``canonical_orth_`` pre-conditions the overlap matrix by
dividing rows / columns by ``sqrt(diag(S))`` before the eigh +
truncation step. The threshold then operates on a unit-diagonal
(scale-free) matrix where it has its intended meaning, instead of on
the bare S whose diagonal entries can vary by orders of magnitude
(symmetry-adapted bases, ECPs, mixed bases on different elements).

Vibe-qc's pre-v0.7 ``_canonical_orthogonalizer`` skipped the
normalisation step. Item 3 of the linear-dep roadmap restores
parity with PySCF and exposes ``normalize_diag_first: bool = True``
on the helper so the legacy raw-S path remains available for
cross-checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.periodic_rhf_ewald import _canonical_orthogonalizer
from vibeqc.periodic_rhf_multi_k_ewald import _canonical_orthogonalizer_complex


# ---------------------------------------------------------------------------
# Real-symmetric (Γ-only) path
# ---------------------------------------------------------------------------

def test_orthogonaliser_satisfies_xtsx_identity_default():
    """Default normalize_diag_first=True: X^T S X = I_{n_kept}."""
    # Random PSD matrix.
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 8))
    S = A @ A.T + 0.1 * np.eye(8)   # ensure well-conditioned
    X, n_kept = _canonical_orthogonalizer(S, threshold=1e-7)
    assert n_kept == 8
    assert np.allclose(X.T @ S @ X, np.eye(n_kept), atol=1e-10)


def test_orthogonaliser_satisfies_xtsx_identity_legacy():
    """Legacy normalize_diag_first=False also works (just less robust
    on poorly-scaled bases)."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 8))
    S = A @ A.T + 0.1 * np.eye(8)
    X, n_kept = _canonical_orthogonalizer(
        S, threshold=1e-7, normalize_diag_first=False,
    )
    assert n_kept == 8
    assert np.allclose(X.T @ S @ X, np.eye(n_kept), atol=1e-10)


def test_normalize_diag_handles_disparate_diagonal_scales():
    """The point of the diagonal-normalisation step. Construct a
    matrix whose diagonal varies wildly: with raw S the threshold
    on eigenvalues is misleading; with sqrt(diag(S))-normalisation
    we get the same threshold semantics regardless of scale."""
    # Two orthogonal AO blocks, one scaled 1e6 larger than the other.
    base = np.array([[1.0, 0.5], [0.5, 1.0]])
    scale = np.diag([1.0, 1e6])
    S = scale @ base @ scale
    # Unit-diagonal version is `base` again, with eigenvalues 0.5, 1.5.
    # On the bare S, eigenvalues are roughly 1e12 and 0.5 — the
    # condition number is 2e12 even though physically the basis is
    # well-conditioned.
    X_norm, n_norm = _canonical_orthogonalizer(
        S, threshold=1e-7, normalize_diag_first=True,
    )
    X_raw, n_raw = _canonical_orthogonalizer(
        S, threshold=1e-7, normalize_diag_first=False,
    )
    # Both pass at threshold 1e-7 (basis is fine), but the X^T S X
    # identity should hold for both.
    assert np.allclose(X_norm.T @ S @ X_norm, np.eye(n_norm), atol=1e-6)
    assert np.allclose(X_raw.T @ S @ X_raw, np.eye(n_raw), atol=1e-6)


def test_normalize_diag_drops_genuinely_singular_direction():
    """Genuinely linearly-dependent direction (off-diagonal coupling
    makes one eigenvalue near zero). After diag-normalisation S has
    unit diagonal and the same null direction; threshold drops it."""
    # Two nearly-parallel basis functions plus one independent one.
    # Off-diagonal coupling 1 - 1e-10 makes the first 2x2 block
    # near-singular even though the diagonal is unit-scale.
    S = np.array([
        [1.0,        1.0 - 1e-10, 0.0],
        [1.0 - 1e-10, 1.0,        0.0],
        [0.0,        0.0,        1.0],
    ])
    X, n_kept = _canonical_orthogonalizer(
        S, threshold=1e-7, normalize_diag_first=True,
    )
    assert n_kept == 2
    # X^T S X = I_2.
    assert np.allclose(X.T @ S @ X, np.eye(2), atol=1e-8)


def test_normalize_diag_rescues_diagonal_scaled_eigenvalue():
    """Counter-test: a *diagonal-only* near-zero eigenvalue is
    actually a SCALING artefact, not a linear dependence. PySCF's
    sqrt-diag normalisation rescues it (correctly), whereas the
    legacy raw-S path drops it."""
    # Single AO with tiny diagonal (numerical scaling, not collapse).
    S = np.diag([1.0, 1.0, 1e-10, 1.0])
    X_norm, n_norm = _canonical_orthogonalizer(
        S, threshold=1e-7, normalize_diag_first=True,
    )
    X_raw, n_raw = _canonical_orthogonalizer(
        S, threshold=1e-7, normalize_diag_first=False,
    )
    # The two paths legitimately disagree here.
    assert n_norm == 4   # rescued by the diag-normalisation
    assert n_raw == 3    # legacy drops it


def test_normalize_diag_falls_back_when_diagonal_nonpositive():
    """If the input has a non-positive diagonal entry (rare but
    possible after a corrupted Bloch sum), the normalisation step
    silently falls back to the raw-S path."""
    S = np.diag([-0.001, 1.0, 1.0])
    # The negative entry triggers severity=critical at the preflight,
    # but if the user opts past it, canonical orth shouldn't crash.
    # The negative direction will be filtered (eigvals >= threshold).
    X, n_kept = _canonical_orthogonalizer(
        S, threshold=1e-7, normalize_diag_first=True,
    )
    # With the legacy fallback path, we still drop the negative
    # eigenvalue and keep the two positive ones.
    assert n_kept == 2


# ---------------------------------------------------------------------------
# Complex-Hermitian (multi-k) path
# ---------------------------------------------------------------------------

def test_complex_orthogonaliser_satisfies_xhsx_identity_default():
    """Default normalize_diag_first=True for the complex path."""
    # Build a Hermitian PSD complex matrix.
    rng = np.random.default_rng(1)
    A = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    S = A @ A.conj().T + 0.1 * np.eye(6)
    # Diagonal of a Hermitian matrix is real.
    assert np.allclose(np.imag(np.diag(S)), 0)
    X, n_kept = _canonical_orthogonalizer_complex(S, threshold=1e-7)
    assert n_kept == 6
    XSX = X.conj().T @ S @ X
    assert np.allclose(XSX, np.eye(n_kept), atol=1e-9)


def test_complex_orthogonaliser_satisfies_xhsx_identity_legacy():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    S = A @ A.conj().T + 0.1 * np.eye(6)
    X, n_kept = _canonical_orthogonalizer_complex(
        S, threshold=1e-7, normalize_diag_first=False,
    )
    assert n_kept == 6
    XSX = X.conj().T @ S @ X
    assert np.allclose(XSX, np.eye(n_kept), atol=1e-9)


# ---------------------------------------------------------------------------
# Both paths agree on a well-conditioned matrix (cross-check)
# ---------------------------------------------------------------------------

def test_both_paths_agree_on_well_conditioned_matrix():
    """For a basis where diag(S) ~ 1 already (e.g. a normalised AO
    set from libint), the two paths should produce X matrices that
    span the same subspace (up to the gauge freedom in the
    rectangular orthogonaliser)."""
    rng = np.random.default_rng(2)
    A = rng.standard_normal((5, 5))
    S = A @ A.T + 0.1 * np.eye(5)
    # Force unit diagonal, simulating a normalised basis.
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)

    X_norm, n_norm = _canonical_orthogonalizer(
        S, threshold=1e-7, normalize_diag_first=True,
    )
    X_raw, n_raw = _canonical_orthogonalizer(
        S, threshold=1e-7, normalize_diag_first=False,
    )
    assert n_norm == n_raw
    # Both span the full space → P_norm = X_norm @ X_norm.T (in the
    # S-metric) equals P_raw.
    P_norm = X_norm @ X_norm.T
    P_raw = X_raw @ X_raw.T
    assert np.allclose(P_norm, P_raw, atol=1e-9)
