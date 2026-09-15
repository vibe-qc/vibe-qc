"""Tests for the periodic density mixers (v0.10.x metal-mixing program, D4).

The decisive property for metals: Anderson / Broyden converge a fixed-point map
that *oscillates/diverges* under plain linear iteration (the limit cycle that
defeats linear density damping on a Gilat-Raubenheimer metal SCF).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.periodic_density_mixing import (
    AndersonMixer,
    BroydenMixer,
    per_k_density_to_vector,
    vector_to_per_k_density,
)


def test_depth1_reduces_to_linear_mixing():
    mix = AndersonMixer(depth=1, beta=0.3)
    x = np.array([1.0, 2.0, 3.0])
    g = np.array([4.0, 0.0, -1.0])
    nxt = mix.update(x, g)
    assert np.allclose(nxt, 0.7 * x + 0.3 * g)


def test_converges_oscillatory_map_where_linear_diverges():
    # g(x) = M x + b with spectral radius 1.1 (eigenvalues 1.1, -0.7): plain
    # linear iteration diverges; Anderson must still converge to the fixed point.
    M = np.array([[0.2, 0.9], [0.9, 0.2]])
    b = np.array([1.0, -2.0])
    x_star = np.linalg.solve(np.eye(2) - M, b)

    x_lin = np.zeros(2)
    for _ in range(60):
        x_lin = M @ x_lin + b
    assert np.linalg.norm(x_lin - x_star) > 1.0  # linear iteration diverged

    mix = AndersonMixer(depth=5, beta=1.0)
    x = np.zeros(2)
    for _ in range(40):
        x = mix.update(x, M @ x + b)
    assert np.linalg.norm(x - x_star) < 1e-8


def test_converges_contractive_map_to_fixed_point():
    rng = np.random.default_rng(0)
    n = 6
    A = rng.standard_normal((n, n))
    A = 0.3 * A / np.linalg.norm(A, 2)  # spectral radius < 1
    b = rng.standard_normal(n)
    x_star = np.linalg.solve(np.eye(n) - A, b)

    mix = AndersonMixer(depth=6, beta=0.8)
    x = np.zeros(n)
    for _ in range(60):
        x = mix.update(x, A @ x + b)
    assert np.linalg.norm(x - x_star) < 1e-9


def test_reset_clears_history():
    mix = AndersonMixer(depth=4, beta=0.5)
    x = np.zeros(3)
    for _ in range(3):
        x = mix.update(x, x + 1.0)
    assert mix.history_size > 1
    mix.reset()
    assert mix.history_size == 0
    # after reset the first update is linear mixing again
    nxt = mix.update(np.array([1.0, 1.0, 1.0]), np.array([3.0, 3.0, 3.0]))
    assert np.allclose(nxt, 0.5 * np.array([1.0, 1.0, 1.0]) + 0.5 * np.array([3.0, 3.0, 3.0]))


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        AndersonMixer(depth=0)
    with pytest.raises(ValueError):
        AndersonMixer(beta=0.0)
    with pytest.raises(ValueError):
        AndersonMixer(beta=1.5)


# ---------------------------------------------------------------------------
# BroydenMixer — same contract as AndersonMixer (limited-memory Type-II Broyden)
# ---------------------------------------------------------------------------


def test_broyden_depth1_reduces_to_linear_mixing():
    mix = BroydenMixer(depth=1, beta=0.3)
    x = np.array([1.0, 2.0, 3.0])
    g = np.array([4.0, 0.0, -1.0])
    nxt = mix.update(x, g)
    assert np.allclose(nxt, 0.7 * x + 0.3 * g)


def test_broyden_converges_oscillatory_map_where_linear_diverges():
    M = np.array([[0.2, 0.9], [0.9, 0.2]])  # spectral radius 1.1
    b = np.array([1.0, -2.0])
    x_star = np.linalg.solve(np.eye(2) - M, b)

    x_lin = np.zeros(2)
    for _ in range(60):
        x_lin = M @ x_lin + b
    assert np.linalg.norm(x_lin - x_star) > 1.0  # linear iteration diverged

    mix = BroydenMixer(depth=5, beta=1.0)
    x = np.zeros(2)
    for _ in range(40):
        x = mix.update(x, M @ x + b)
    assert np.linalg.norm(x - x_star) < 1e-8


def test_broyden_converges_contractive_map_to_fixed_point():
    rng = np.random.default_rng(0)
    n = 6
    A = rng.standard_normal((n, n))
    A = 0.3 * A / np.linalg.norm(A, 2)  # spectral radius < 1
    b = rng.standard_normal(n)
    x_star = np.linalg.solve(np.eye(n) - A, b)

    mix = BroydenMixer(depth=6, beta=0.8)
    x = np.zeros(n)
    for _ in range(60):
        x = mix.update(x, A @ x + b)
    assert np.linalg.norm(x - x_star) < 1e-9


def test_broyden_reset_clears_history():
    # A contractive (non-constant-residual) map so the secant history actually
    # builds before the reset.
    rng = np.random.default_rng(1)
    A = 0.3 * rng.standard_normal((4, 4))
    A = 0.3 * A / np.linalg.norm(A, 2)
    b = rng.standard_normal(4)
    mix = BroydenMixer(depth=4, beta=0.5)
    x = np.zeros(4)
    for _ in range(4):
        x = mix.update(x, A @ x + b)
    assert mix.history_size > 0
    mix.reset()
    assert mix.history_size == 0
    nxt = mix.update(np.array([1.0, 1.0, 1.0]), np.array([3.0, 3.0, 3.0]))
    assert np.allclose(nxt, 0.5 * np.array([1.0, 1.0, 1.0]) + 0.5 * np.array([3.0, 3.0, 3.0]))


def test_broyden_rejects_bad_params():
    with pytest.raises(ValueError):
        BroydenMixer(depth=0)
    with pytest.raises(ValueError):
        BroydenMixer(beta=0.0)
    with pytest.raises(ValueError):
        BroydenMixer(beta=1.5)


# ---------------------------------------------------------------------------
# Per-k density-matrix <-> flat-vector bridge
# ---------------------------------------------------------------------------


def _random_hermitian(n, rng):
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return 0.5 * (A + A.conj().T)


def test_per_k_vector_roundtrip_preserves_hermitian_density():
    rng = np.random.default_rng(3)
    D_per_k = [_random_hermitian(4, rng) for _ in range(3)]
    vec = per_k_density_to_vector(D_per_k)
    assert vec.dtype == np.float64
    back = vector_to_per_k_density(vec, D_per_k)
    for D, D2 in zip(D_per_k, back):
        assert np.allclose(D, D2)
        assert np.allclose(D2, D2.conj().T)  # Hermitian


def test_per_k_vector_mix_preserves_total_trace():
    # A real linear combination (sum of coeffs = 1) of per-k densities must
    # preserve the per-k trace (the electron-count functional) exactly — the
    # invariant the multi-k driver relies on instead of the aliased bloch-sum.
    rng = np.random.default_rng(4)
    Da = [_random_hermitian(3, rng) for _ in range(2)]
    Db = [_random_hermitian(3, rng) for _ in range(2)]
    va = per_k_density_to_vector(Da)
    vb = per_k_density_to_vector(Db)
    theta = 0.3
    mixed = vector_to_per_k_density(theta * va + (1 - theta) * vb, Da)
    for k in range(2):
        expect = theta * np.trace(Da[k]) + (1 - theta) * np.trace(Db[k])
        assert np.isclose(np.trace(mixed[k]).real, expect.real)
