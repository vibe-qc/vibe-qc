# Copyright (c) vibe-qc contributors.
# SPDX-License-Identifier: MPL-2.0
"""``DIIS::extrapolate_blocks``: the canonical Pulay kernel on block vectors.

The multi-k periodic accelerators used to carry their own numpy Pulay DIIS.
They are now thin adapters over the same C++ ``DIIS`` the molecular drivers
use, reached through ``extrapolate_blocks``. Two properties make that
substitution sound, and both are pinned here:

* the block kernel is the scalar kernel on a longer vector, and
* the ``√w_k`` block bridge turns its Euclidean inner product into the
  k-weighted Pulay inner product ``Σ_k w_k Re Tr[e_i(k)† e_j(k)]``.

Also guards the two behaviours the numpy copy had that the C++ kernel
previously lacked: the coefficient blow-up fallback, and raw-Fock passthrough
at zero-weight k-points.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc._vibeqc_core import DIIS, DIISDepthPolicy
from vibeqc.periodic_scf_accelerators import (
    _MultiKKDIIS,
    _MultiKKDIISOpenShell,
    _MultiKPulayDIIS,
)


def _sym(n, rng):
    a = rng.normal(size=(n, n))
    return a + a.T


def _herm(n, rng):
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return a + a.conj().T


# ---------------------------------------------------------------------------
# The block kernel is the scalar kernel on a concatenated vector
# ---------------------------------------------------------------------------

def test_single_block_matches_scalar_extrapolate():
    rng = np.random.default_rng(3)
    scalar, blocked = DIIS(8), DIIS(8)
    for _ in range(6):
        F, e = _sym(5, rng), _sym(5, rng)
        got_scalar = scalar.extrapolate(F, e)
        got_block = blocked.extrapolate_blocks([F], [e])
        assert len(got_block) == 1
        np.testing.assert_allclose(got_block[0], got_scalar, atol=1e-12)


def test_two_blocks_match_the_stacked_scalar_history():
    """Splitting one vector into two blocks must not change the answer."""
    rng = np.random.default_rng(4)
    scalar, blocked = DIIS(8), DIIS(8)
    for _ in range(6):
        F1, F2 = _sym(4, rng), _sym(4, rng)
        e1, e2 = _sym(4, rng), _sym(4, rng)
        # Vertical stack: the Frobenius product of the stack is the sum of
        # the per-block Frobenius products, which is what the kernel forms.
        got_scalar = scalar.extrapolate(np.vstack([F1, F2]), np.vstack([e1, e2]))
        got_block = blocked.extrapolate_blocks([F1, F2], [e1, e2])
        np.testing.assert_allclose(got_block[0], got_scalar[:4], atol=1e-12)
        np.testing.assert_allclose(got_block[1], got_scalar[4:], atol=1e-12)


def test_block_shapes_may_differ_between_fock_and_error():
    """KDIIS pairs nbf×nbf Fock blocks with n_vir×n_occ gradient blocks."""
    rng = np.random.default_rng(5)
    diis = DIIS(8)
    for _ in range(4):
        F = _sym(6, rng)
        g = rng.normal(size=(4, 2))       # rectangular error, deliberately
        out = diis.extrapolate_blocks([F], [g])
        assert out[0].shape == (6, 6)


def test_layout_change_between_iterates_is_rejected():
    diis = DIIS(8)
    rng = np.random.default_rng(6)
    diis.extrapolate_blocks([_sym(4, rng)], [_sym(4, rng)])
    with pytest.raises(ValueError, match="block layout must be stable"):
        diis.extrapolate_blocks([_sym(4, rng)], [_sym(5, rng)])


# ---------------------------------------------------------------------------
# Pulay constraint and the degenerate-history fallback
# ---------------------------------------------------------------------------

def test_coefficients_sum_to_one():
    """Σ c_i = 1, so a history of identical Focks extrapolates to itself."""
    rng = np.random.default_rng(7)
    diis = DIIS(8)
    F = _sym(5, rng)
    for _ in range(5):
        out = diis.extrapolate_blocks([F], [_sym(5, rng)])
    np.testing.assert_allclose(out[0], F, atol=1e-9)


def test_blown_up_coefficients_fall_back_to_the_raw_fock():
    """Near-parallel errors drive the Pulay coefficients to ~1/ε.

    With e_2 = (1+ε)·e_1 the constrained minimiser is c = (1+1/ε, -1/ε): for
    ε = 1e-13 that is ~1e13, and Σ c_i F_i is then pure cancellation
    round-off. The kernel must shrink the window and, at a two-iterate
    window, hand back the raw Fock. A plain SCF step is a slow move, never a
    wrong one.
    """
    rng = np.random.default_rng(8)
    diis = DIIS(8)
    e1 = _sym(4, rng)
    F1, F2 = _sym(4, rng), _sym(4, rng)
    diis.extrapolate_blocks([F1], [e1])
    out = diis.extrapolate_blocks([F2], [e1 * (1.0 + 1.0e-13)])
    assert np.all(np.isfinite(out[0]))
    np.testing.assert_allclose(out[0], F2, atol=1e-12)


def test_pulay_coefficients_are_invariant_to_residual_scale():
    """Multiplying every error by one scalar must not change the fit.

    The Pulay Gram matrix then gains the square of that scalar, which changes
    only the bordered system's Lagrange multiplier in exact arithmetic. A raw
    ``B ~ 1e-16`` block beside the unit constraint used to change FullPivLU's
    rank decision and therefore the extrapolated Fock.
    """
    rng = np.random.default_rng(87)
    unit_errors = DIIS(8)
    small_errors = DIIS(8)

    for _ in range(5):
        fock = _sym(4, rng)
        error = _sym(4, rng)
        expected = unit_errors.extrapolate(fock, error)
        actual = small_errors.extrapolate(fock, 1.0e-8 * error)
        np.testing.assert_allclose(actual, expected, atol=5e-13, rtol=0.0)


# ---------------------------------------------------------------------------
# The √w_k bridge reproduces the k-weighted Pulay inner product
# ---------------------------------------------------------------------------

def _reference_multik_pulay(F_hist, E_hist, weights):
    """The pre-port numpy implementation, kept here as the oracle."""
    n = len(F_hist)
    if n < 2:
        return [np.asarray(F).copy() for F in F_hist[-1]]
    n_k = len(F_hist[0])
    B = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(i, n):
            bij = sum(
                float(weights[k]) * float(np.real(np.vdot(
                    E_hist[i][k].ravel(), E_hist[j][k].ravel())))
                for k in range(n_k)
            )
            B[i, j] = B[j, i] = bij
    B[n, :n] = -1.0
    B[:n, n] = -1.0
    rhs = np.zeros(n + 1)
    rhs[n] = -1.0
    c = np.linalg.solve(B, rhs)
    return [
        sum(c[i] * F_hist[i][k] for i in range(n))
        for k in range(n_k)
    ]


def test_multik_pulay_matches_the_numpy_oracle():
    rng = np.random.default_rng(9)
    weights = [0.5, 0.3, 0.2]
    accel = _MultiKPulayDIIS(max_subspace=8)
    F_hist, E_hist = [], []
    for it in range(7):
        Fk = [_herm(5, rng) for _ in weights]
        Ek = [_herm(5, rng) * (0.5 ** it) for _ in weights]   # converging
        F_hist.append(Fk)
        E_hist.append(Ek)
        got = accel.extrapolate(Fk, Ek, weights)
        want = _reference_multik_pulay(F_hist, E_hist, weights)
        for g, w in zip(got, want):
            np.testing.assert_allclose(g, w, atol=1e-10)


@pytest.mark.parametrize("policy", [DIISDepthPolicy.RESTART,
                                    DIISDepthPolicy.ADAPTIVE])
def test_multik_depth_policy_matches_the_scalar_kernel(policy):
    """The √w_k-flattened multi-k history must retain the same depth as the
    scalar C++ path fed the identical flattened vector."""
    weights = [0.5, 0.3, 0.2]
    rng_mk = np.random.default_rng(10)
    rng_sc = np.random.default_rng(10)      # same draws
    multik = _MultiKPulayDIIS(8, policy, 1.0e-4)
    scalar = DIIS(8, policy, 1.0e-4)

    def flatten(mats):
        return np.concatenate([
            np.sqrt(weights[k]) * np.concatenate(
                [m.real.ravel(order="F"), m.imag.ravel(order="F")])
            for k, m in enumerate(mats)
        ]).reshape(-1, 1)

    for it in range(10):
        Fk = [_herm(4, rng_mk) for _ in weights]
        Ek = [_herm(4, rng_mk) * (0.6 ** it) for _ in weights]
        multik.extrapolate(Fk, Ek, weights)
        Fs = [_herm(4, rng_sc) for _ in weights]
        Es = [_herm(4, rng_sc) * (0.6 ** it) for _ in weights]
        scalar.extrapolate(flatten(Fs), flatten(Es))
        assert multik.subspace_size == scalar.subspace_size


def test_zero_weight_k_returns_the_raw_fock():
    """A zero-weight k never enters the B-matrix, so DIIS knows nothing about
    it; handing back a zeroed Fock would poison its bands."""
    rng = np.random.default_rng(11)
    weights = [1.0, 0.0]
    accel = _MultiKPulayDIIS(max_subspace=8)
    for _ in range(3):
        Fk = [_herm(4, rng) for _ in weights]
        Ek = [_herm(4, rng) for _ in weights]
        out = accel.extrapolate(Fk, Ek, weights)
    np.testing.assert_allclose(out[1], Fk[1], atol=1e-12)


# ---------------------------------------------------------------------------
# KDIIS adapters
# ---------------------------------------------------------------------------

def test_multik_kdiis_extrapolates_and_preserves_shapes():
    rng = np.random.default_rng(12)
    weights = [0.6, 0.4]
    accel = _MultiKKDIIS(max_subspace=8)
    for _ in range(4):
        Fk = [_herm(6, rng) for _ in weights]
        Ck = [np.linalg.qr(_herm(6, rng))[0] for _ in weights]
        out = accel.extrapolate(Fk, Ck, 2, weights)
    assert len(out) == 2
    assert all(o.shape == (6, 6) and np.all(np.isfinite(o)) for o in out)
    assert accel.subspace_size == 4


def test_multik_kdiis_open_shell_is_spin_coupled():
    """One B-matrix, one coefficient set for both spins: feeding identical α
    and β inputs must return identical α and β extrapolations."""
    rng = np.random.default_rng(13)
    weights = [0.6, 0.4]
    accel = _MultiKKDIISOpenShell(max_subspace=8)
    for _ in range(4):
        Fk = [_herm(6, rng) for _ in weights]
        Ck = [np.linalg.qr(_herm(6, rng))[0] for _ in weights]
        Fa_ex, Fb_ex = accel.extrapolate(Fk, Fk, Ck, Ck, 2, 2, weights)
    for a, b in zip(Fa_ex, Fb_ex):
        np.testing.assert_allclose(a, b, atol=1e-12)
    # And the coefficient set actually did something: the extrapolate is not
    # simply the last Fock handed back.
    assert not np.allclose(Fa_ex[0], Fk[0], atol=1e-8)
