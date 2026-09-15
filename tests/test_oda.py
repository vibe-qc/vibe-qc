"""Tests for the ODA helper.

The Bloch-sum-to-k path needs a LatticeMatrixSet, so the tests use
a minimal fake (a Python list of blocks + a cells iterable) where
needed, plus pure-numpy tests for the λ-selection math.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from vibeqc.oda import (
    ODAStep,
    _bloch_sum_to_k,
    compute_oda_lambda,
    oda_mix_densities,
)


# ---- Pure λ-math tests (synthetic g0, g1 via mocked F & D) -------------------


@dataclass
class _FakeCell:
    index: np.ndarray
    r_cart: np.ndarray


class _FakeLMS:
    """Minimal LatticeMatrixSet stand-in: just `cells` + `blocks` +
    `set_block`. Sufficient for ODA's read/write needs."""

    def __init__(self, cells, blocks):
        self.cells = cells
        self.blocks = [b.copy() for b in blocks]

    def set_block(self, g_idx, block):
        self.blocks[g_idx] = block


def _gamma_only_lms(D: np.ndarray) -> _FakeLMS:
    """Build a Γ-only LMS: single cell at origin with the given block."""
    return _FakeLMS(
        cells=[_FakeCell(index=np.array([0, 0, 0]), r_cart=np.zeros(3))],
        blocks=[D],
    )


def test_lambda_one_when_curvature_zero_and_descent():
    """If g0 < 0 and g1 == g0 (zero curvature), full step is optimal."""
    n_bf = 4
    D_n = np.eye(n_bf, dtype=complex)
    D_naive = 2 * np.eye(n_bf, dtype=complex)
    # F_n chosen so tr((D_naive - D_n) F_n) = tr(I * F_n) < 0
    F_n = -1.0 * np.eye(n_bf, dtype=complex)
    F_naive = F_n.copy()  # same → g0 == g1
    step = compute_oda_lambda(
        _gamma_only_lms(D_n), _gamma_only_lms(D_naive),
        [F_n], [F_naive],
        [np.zeros(3)], [1.0],
    )
    assert step.lam == 1.0
    assert not step.quadratic_minimum  # clamped, not interior


def test_lambda_one_when_concave_and_both_endpoints_descent():
    """Concave model (g1 < g0 < 0): both endpoints downhill → λ=1
    is optimal. This is the case that was buggy in the first impl
    (the buggy clamp gave λ=0, freezing the SCF at the initial
    guess basin)."""
    n_bf = 2
    D_n = np.eye(n_bf, dtype=complex)
    D_naive = 2 * np.eye(n_bf, dtype=complex)
    # g0 = tr(I·(-1·I)) = -2; g1 = tr(I·(-5·I)) = -10
    # curvature = g1-g0 = -8 (concave)
    # E(0) = 0, E(1) = -2 + 0.5·(-8) = -6 (lower) → λ=1 wins
    F_n = -1.0 * np.eye(n_bf, dtype=complex)
    F_naive = -5.0 * np.eye(n_bf, dtype=complex)
    step = compute_oda_lambda(
        _gamma_only_lms(D_n), _gamma_only_lms(D_naive),
        [F_n], [F_naive],
        [np.zeros(3)], [1.0],
    )
    assert step.lam == 1.0
    assert not step.quadratic_minimum  # endpoint, not interior


def test_trust_region_caps_full_step():
    """When unconstrained λ ≥ trust_lambda_max, clamp to that."""
    n_bf = 2
    D_n = np.eye(n_bf, dtype=complex)
    D_naive = 2 * np.eye(n_bf, dtype=complex)
    # Same as the "concave both descent" case above: unconstrained
    # picks λ=1, but trust_lambda_max=0.3 caps it.
    F_n = -1.0 * np.eye(n_bf, dtype=complex)
    F_naive = -5.0 * np.eye(n_bf, dtype=complex)
    step = compute_oda_lambda(
        _gamma_only_lms(D_n), _gamma_only_lms(D_naive),
        [F_n], [F_naive],
        [np.zeros(3)], [1.0],
        trust_lambda_max=0.3,
    )
    assert step.lam == pytest.approx(0.3)


def test_trust_region_caps_convex_minimum():
    """Convex case with interior min above the trust cap should
    clamp to trust_lambda_max."""
    n_bf = 2
    D_n = np.eye(n_bf, dtype=complex)
    D_naive = 2 * np.eye(n_bf, dtype=complex)
    # g0 = -1, g1 = +1. Convex (g1-g0=2 > 0). λ_raw = -1/-2 = 0.5
    # With trust_lambda_max=0.1: clamp to 0.1.
    F_n = -0.5 * np.eye(n_bf, dtype=complex)
    F_naive = +0.5 * np.eye(n_bf, dtype=complex)
    step = compute_oda_lambda(
        _gamma_only_lms(D_n), _gamma_only_lms(D_naive),
        [F_n], [F_naive],
        [np.zeros(3)], [1.0],
        trust_lambda_max=0.1,
    )
    assert step.lam == pytest.approx(0.1)


def test_trust_region_rejects_invalid_max():
    n_bf = 2
    D_n = _gamma_only_lms(np.eye(n_bf, dtype=complex))
    D_naive = _gamma_only_lms(2 * np.eye(n_bf, dtype=complex))
    F = [np.eye(n_bf, dtype=complex)]
    with pytest.raises(ValueError, match="trust_lambda_max"):
        compute_oda_lambda(
            D_n, D_naive, F, F, [np.zeros(3)], [1.0],
            trust_lambda_max=0.0,
        )
    with pytest.raises(ValueError, match="trust_lambda_max"):
        compute_oda_lambda(
            D_n, D_naive, F, F, [np.zeros(3)], [1.0],
            trust_lambda_max=1.5,
        )


def test_lambda_clamped_to_zero_when_uphill():
    """If g0 > 0 and curvature ≤ 0, ODA refuses the step."""
    n_bf = 2
    D_n = np.eye(n_bf, dtype=complex)
    D_naive = 2 * np.eye(n_bf, dtype=complex)
    F_n = +1.0 * np.eye(n_bf, dtype=complex)   # g0 = tr(I·I) = 2 > 0
    F_naive = F_n.copy()                        # g1 = g0
    step = compute_oda_lambda(
        _gamma_only_lms(D_n), _gamma_only_lms(D_naive),
        [F_n], [F_naive],
        [np.zeros(3)], [1.0],
    )
    assert step.lam == 0.0


def test_lambda_picks_interior_minimum_when_curvature_positive():
    """The textbook case: g0 < 0, g1 > 0 ⇒ λ_opt = g0/(g0-g1) ∈ (0, 1)."""
    n_bf = 2
    D_n = np.eye(n_bf, dtype=complex)
    D_naive = 2 * np.eye(n_bf, dtype=complex)
    # tr((D_naive-D_n) F_n) = tr(I · (-2I)) = -4 = g0
    # tr((D_naive-D_n) F_naive) = tr(I · 2I) = +4 = g1
    # λ_opt = -4 / (-4 - 4) = 0.5
    F_n = -2.0 * np.eye(n_bf, dtype=complex)
    F_naive = +2.0 * np.eye(n_bf, dtype=complex)
    step = compute_oda_lambda(
        _gamma_only_lms(D_n), _gamma_only_lms(D_naive),
        [F_n], [F_naive],
        [np.zeros(3)], [1.0],
    )
    assert step.quadratic_minimum
    assert step.lam == pytest.approx(0.5, abs=1e-12)


def test_lambda_clamped_to_one_when_curvature_too_small():
    """If unconstrained λ ≥ 1, ODA takes the full step (capped)."""
    n_bf = 2
    D_n = np.eye(n_bf, dtype=complex)
    D_naive = 2 * np.eye(n_bf, dtype=complex)
    # g0 = -10 (steep descent), g1 = -5 (still descent at D_naive)
    # λ_opt = -10/(-10 - -5) = -10/-5 = 2 → clamped to 1.
    F_n = -5.0 * np.eye(n_bf, dtype=complex)   # tr(I · (-5I)) = -10
    F_naive = -2.5 * np.eye(n_bf, dtype=complex)  # tr(I · (-2.5I)) = -5
    step = compute_oda_lambda(
        _gamma_only_lms(D_n), _gamma_only_lms(D_naive),
        [F_n], [F_naive],
        [np.zeros(3)], [1.0],
    )
    assert step.lam == 1.0
    assert not step.quadratic_minimum


def test_multi_k_weights_combine_properly():
    """Two k-points with equal weight should average g0 / g1."""
    n_bf = 2
    D_n = np.eye(n_bf, dtype=complex)
    D_naive = 2 * np.eye(n_bf, dtype=complex)
    F_n0 = -2.0 * np.eye(n_bf, dtype=complex)   # g0_0 = -4
    F_n1 = -4.0 * np.eye(n_bf, dtype=complex)   # g0_1 = -8
    F_naive0 = +2.0 * np.eye(n_bf, dtype=complex)  # g1_0 = +4
    F_naive1 = 0.0 * np.eye(n_bf, dtype=complex)   # g1_1 = 0
    # weighted: g0 = 0.5*(-4) + 0.5*(-8) = -6; g1 = 0.5*4 + 0.5*0 = 2
    # λ = -6/(-6 - 2) = -6/-8 = 0.75
    step = compute_oda_lambda(
        _gamma_only_lms(D_n), _gamma_only_lms(D_naive),
        [F_n0, F_n1], [F_naive0, F_naive1],
        [np.zeros(3), np.zeros(3)], [0.5, 0.5],
    )
    assert step.g0 == pytest.approx(-6.0)
    assert step.g1 == pytest.approx(2.0)
    assert step.lam == pytest.approx(0.75)


def test_bloch_sum_at_gamma_recovers_g0_block():
    """At k = (0, 0, 0), Bloch sum = Σ_g block(g)."""
    n_bf = 2
    blk0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    blk1 = np.array([[0.0, 0.5], [0.5, 0.0]])
    lms = _FakeLMS(
        cells=[
            _FakeCell(index=np.array([0, 0, 0]), r_cart=np.array([0., 0., 0.])),
            _FakeCell(index=np.array([1, 0, 0]), r_cart=np.array([3., 0., 0.])),
        ],
        blocks=[blk0, blk1],
    )
    D_k = _bloch_sum_to_k(lms, np.zeros(3))
    np.testing.assert_allclose(D_k, blk0 + blk1, atol=1e-12)


def test_oda_mix_densities_full_step():
    """λ = 1: result = D_naive exactly."""
    D_n = _gamma_only_lms(np.ones((2, 2), dtype=float))
    D_naive = _gamma_only_lms(7 * np.ones((2, 2), dtype=float))
    oda_mix_densities(D_n, D_naive, 1.0)
    np.testing.assert_allclose(D_n.blocks[0], 7 * np.ones((2, 2)))


def test_oda_mix_densities_no_step():
    """λ = 0: D_n unchanged."""
    D_n_blk = np.ones((2, 2), dtype=float)
    D_n = _gamma_only_lms(D_n_blk.copy())
    D_naive = _gamma_only_lms(7 * np.ones((2, 2), dtype=float))
    oda_mix_densities(D_n, D_naive, 0.0)
    np.testing.assert_allclose(D_n.blocks[0], D_n_blk)


def test_oda_mix_densities_interior_mix():
    """λ = 0.25: result = 0.75·D_n + 0.25·D_naive."""
    D_n = _gamma_only_lms(np.zeros((2, 2)))
    D_naive = _gamma_only_lms(4 * np.ones((2, 2)))
    oda_mix_densities(D_n, D_naive, 0.25)
    np.testing.assert_allclose(D_n.blocks[0], np.ones((2, 2)))


def test_oda_mix_rejects_out_of_range_lam():
    D_n = _gamma_only_lms(np.zeros((2, 2)))
    D_naive = _gamma_only_lms(np.ones((2, 2)))
    with pytest.raises(ValueError, match="lam must be in"):
        oda_mix_densities(D_n, D_naive, -0.1)
    with pytest.raises(ValueError, match="lam must be in"):
        oda_mix_densities(D_n, D_naive, 1.1)


def test_compute_oda_lambda_rejects_mismatched_inputs():
    D_n = _gamma_only_lms(np.eye(2, dtype=complex))
    D_naive = _gamma_only_lms(2 * np.eye(2, dtype=complex))
    F_n = [np.eye(2, dtype=complex)]
    F_naive = [np.eye(2, dtype=complex), np.eye(2, dtype=complex)]
    with pytest.raises(ValueError, match="F_naive_k_list has 2"):
        compute_oda_lambda(D_n, D_naive, F_n, F_naive, [np.zeros(3)], [1.0])
