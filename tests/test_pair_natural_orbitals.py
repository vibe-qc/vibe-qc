"""Tiny independent algebra oracles for the native restricted pair-PNO leaf.

No SCF, molecular integrals, or periodic calculation is needed. The density
oracle uses Riplinger and Neese, JCP 138, 034106 (2013), Eq. (23),
doi:10.1063/1.4773581, directly, independently of the implementation's
symmetric/antisymmetric Gram decomposition.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


_TOL = 32.0 * np.finfo(float).eps
_AMPLITUDES = np.array(
    [[0.10, 0.04, -0.02], [-0.03, 0.08, 0.05], [0.07, -0.01, 0.02]],
    dtype=np.float64,
)


def _mp2_pair(g, eps, fii=-0.7, fjj=-0.4, floor=1e-8, cap=None):
    if cap is None:
        cap = 8 * g.shape[0] ** 2
    return core._restricted_pair_semicanonical_mp2(g, eps, fii, fjj, floor, cap)


@pytest.mark.parametrize("n", [1, 2, 4, 9])
def test_initial_mp2_pair_against_defining_amplitudes_and_spin_energy(n):
    rng = np.random.default_rng(8103 + n)
    g = rng.normal(size=(n, n)) * 0.2
    eps = np.linspace(0.2, 1.1, n)
    result = _mp2_pair(g, eps)
    delta = eps[:, None] + eps[None, :] + 1.1
    expected = -g / delta
    np.testing.assert_allclose(result.amplitudes_copy(), expected, atol=1e-16, rtol=1e-15)
    assert result.ordered_pair_energy == pytest.approx(np.sum((2 * g - g.T) * expected), abs=1e-15)
    assert result.ordered_pair_energy <= 0
    assert result.minimum_denominator == pytest.approx(delta.min(), abs=1e-15)
    assert result.maximum_denominator == pytest.approx(delta.max(), abs=1e-15)
    assert result.maximum_absolute_amplitude == pytest.approx(np.max(np.abs(expected)), abs=1e-16)
    assert result.maximum_residual < 1e-16
    assert result.residual_frobenius_norm < 4e-16
    assert result.energy_term_underflow_count == 0
    assert result.memory.input_bytes == 8 * (n * n + n)
    assert result.memory.peak_owned_numerical_bytes == 8 * n * n
    # Exchange of the occupied labels transposes G and T, and does not add
    # the ordered-pair multiplicity. Translation/cell weights belong above.
    reversed_pair = _mp2_pair(g.T.copy(), eps, fii=-0.4, fjj=-0.7)
    np.testing.assert_array_equal(reversed_pair.amplitudes_copy(), result.amplitudes_copy().T)
    assert reversed_pair.ordered_pair_energy == result.ordered_pair_energy


def test_initial_mp2_pair_connects_to_full_domain_pnos():
    g = np.array([[0.31, 0.12], [-0.04, 0.17]])
    eps = np.array([0.4, 0.9])
    initial = _mp2_pair(g, eps)
    t = initial.amplitudes_copy()
    pnos = _build(t, cutoff=0.0)
    c = pnos.coefficients_copy()
    expected_density = (4 * t - 2 * t.T) @ t.T + (4 * t.T - 2 * t) @ t
    np.testing.assert_allclose(pnos.density_copy(), expected_density, atol=1e-16)
    np.testing.assert_allclose(c @ (c.T @ t @ c) @ c.T, t, atol=2e-16)


def test_initial_mp2_ordered_diagonal_pair_has_no_extra_factor():
    g = np.array([[0.2, -0.13], [-0.13, 0.3]])
    eps = np.array([0.3, 0.8])
    result = _mp2_pair(g, eps, fii=-0.5, fjj=-0.5)
    expected = -g / (eps[:, None] + eps[None, :] + 1.0)
    assert result.ordered_pair_energy == pytest.approx(np.sum(g * expected), abs=1e-16)
    np.testing.assert_array_equal(result.amplitudes_copy(), result.amplitudes_copy().T)


def test_initial_mp2_zero_integrals_are_valid_and_input_is_borrowed():
    g = np.zeros((3, 3))
    eps = np.array([0.3, 0.5, 0.7])
    g.flags.writeable = eps.flags.writeable = False
    result = _mp2_pair(g, eps)
    assert result.ordered_pair_energy == result.maximum_residual == 0.0
    copied = result.amplitudes_copy()
    copied[:] = 9.0
    np.testing.assert_array_equal(result.amplitudes_copy(), np.zeros((3, 3)))
    np.testing.assert_array_equal(g, np.zeros((3, 3)))


@pytest.mark.parametrize("cap", [0, 31])
def test_initial_mp2_cap_precedes_nonfinite_input_read(cap):
    with pytest.raises(ValueError, match="byte cap"):
        _mp2_pair(np.full((2, 2), np.nan), np.ones(2), cap=cap)


@pytest.mark.parametrize("floor", [0.0, -0.1, np.nan, np.inf])
def test_initial_mp2_requires_explicit_positive_denominator_floor(floor):
    with pytest.raises(ValueError, match="positive denominator floor"):
        _mp2_pair(np.ones((1, 1)), np.ones(1), floor=floor)


@pytest.mark.parametrize("eps", [-0.1, 0.0, 0.25])
def test_initial_mp2_small_zero_negative_or_equal_floor_denominators_fail(eps):
    with pytest.raises(ValueError, match="strictly exceed"):
        _mp2_pair(np.ones((1, 1)), np.array([eps]), fii=0.0, fjj=0.0, floor=0.5)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_initial_mp2_rejects_nonfinite_numbers(bad):
    with pytest.raises(ValueError, match="exchange integrals"):
        _mp2_pair(np.array([[bad]]), np.ones(1))
    with pytest.raises(ValueError, match="virtual energies"):
        _mp2_pair(np.ones((1, 1)), np.array([bad]))
    with pytest.raises(ValueError, match="occupied Fock"):
        _mp2_pair(np.ones((1, 1)), np.ones(1), fii=bad)


def test_initial_mp2_float_range_and_energy_underflow_are_explicit():
    # Avoid g*g overflow when the final energy remains representable.
    result = _mp2_pair(np.array([[1e200]]), np.array([1e150]), fii=0.0, fjj=0.0)
    assert result.ordered_pair_energy == pytest.approx(-5e249, rel=3e-15)
    result = _mp2_pair(np.array([[1e-200]]), np.array([1.0]), fii=0.0, fjj=0.0)
    assert result.energy_term_underflow_count == 1
    assert result.ordered_pair_energy == 0.0
    assert result.amplitudes_copy()[0, 0] == -5e-201
    with pytest.raises(OverflowError, match="amplitude overflow or underflow"):
        _mp2_pair(np.array([[np.nextafter(0.0, 1.0)]]), np.array([2.0]), fii=0.0, fjj=0.0)
    with pytest.raises(OverflowError, match="energy term overflow"):
        _mp2_pair(np.array([[1e200]]), np.ones(1), fii=0.0, fjj=0.0)


def test_initial_mp2_subnormal_denominators_and_weighted_energies_survive():
    d = np.nextafter(0.0, 1.0)
    result = _mp2_pair(np.array([[6 * d]]), np.array([3 * d]), fii=0.0, fjj=0.0, floor=d)
    assert result.minimum_denominator == 6 * d
    assert result.amplitudes_copy()[0, 0] == -1.0
    assert result.ordered_pair_energy == -6 * d
    g = np.array([[0.0, d], [d, 0.0]])
    result = _mp2_pair(g, np.zeros(2), fii=-2 * d, fjj=0.0, floor=d)
    assert result.amplitudes_copy()[0, 1] == -0.5
    assert result.ordered_pair_energy == -d
    assert result.energy_term_underflow_count == 0
    g = np.array([[0.0, 2.0**-538], [-2.0**-538, 0.0]])
    result = _mp2_pair(g, np.array([0.5, 0.5]), fii=0.0, fjj=0.0)
    assert result.ordered_pair_energy == -2 * d
    assert result.energy_term_underflow_count == 0


def test_initial_mp2_denominator_rejects_partial_scaling_loss_before_cancellation():
    # A nonzero scaled subnormal can still have lost bits, even when a
    # scaled-to-zero-only check would accept it. The enormous occupied
    # entries cancel, exposing exactly that lost low-order energy scale.
    with pytest.raises(OverflowError, match="denominator input scaling loses range"):
        _mp2_pair(np.ones((1, 1)), np.array([3 * 2.0**-51]),
                  fii=2.0**1023, fjj=-2.0**1023, floor=2.0**-54)


@pytest.mark.parametrize("g", [np.ones((2, 2), dtype=np.float32),
                               np.ones((2, 2), dtype=complex),
                               np.asfortranarray(np.ones((2, 2))),
                               np.ones((2, 3))])
def test_initial_mp2_never_casts_or_copies_invalid_integrals(g):
    with pytest.raises(ValueError, match="existing C-contiguous float64"):
        _mp2_pair(g, np.ones(2))


def test_initial_mp2_rejects_noncontiguous_energies_empty_and_excess_work():
    with pytest.raises(ValueError, match="existing C-contiguous float64"):
        _mp2_pair(np.ones((2, 2)), np.ones(4)[::2])
    with pytest.raises(ValueError, match="dimension must be positive"):
        _mp2_pair(np.zeros((0, 0)), np.zeros(0))
    n = core.PAIR_PNO_DIAGNOSTIC_MAXIMUM_DIMENSION + 1
    with pytest.raises(ValueError, match="tiny-matrix limit"):
        _mp2_pair(np.zeros((n, n)), np.ones(n))
    with pytest.raises(OverflowError):
        core._plan_restricted_pair_semicanonical_mp2(2**63)


def _semicanonicalize(
    coefficients, fock, *, cap=None, orthogonality_tolerance=1e-12,
    sweeps=80, tolerance=_TOL,
):
    if cap is None:
        cap = core.plan_restricted_pair_semicanonicalization(
            *coefficients.shape
        ).peak_owned_numerical_bytes
    return core.restricted_pair_semicanonicalize(
        coefficients, fock, orthogonality_tolerance, cap, sweeps, tolerance
    )


def _assert_semicanonicalization(coefficients, fock, result):
    rotated = result.coefficients_copy()
    energies = result.energies_copy()
    r = coefficients.shape[1]
    np.testing.assert_allclose(rotated.T @ rotated, np.eye(r), atol=3e-13)
    np.testing.assert_allclose(
        rotated @ rotated.T, coefficients @ coefficients.T, atol=3e-13,
    )
    np.testing.assert_allclose(
        rotated.T @ fock @ rotated, np.diag(energies), atol=3e-13,
    )
    np.testing.assert_allclose(
        energies, np.linalg.eigvalsh(coefficients.T @ fock @ coefficients), atol=3e-13,
    )
    assert np.all(energies[:-1] <= energies[1:])
    assert result.input_orthonormality_error < 3e-13
    assert result.output_orthonormality_error < 3e-13
    assert result.reduced_eigensystem_relative_residual < 3e-13
    assert result.projected_fock_relative_residual < 3e-13
    assert result.subspace_projector_frobenius_error < 3e-13
    denominator = np.linalg.norm(fock)
    full_residual = np.linalg.norm(fock @ rotated - rotated * energies)
    full_relative = full_residual / (denominator if denominator else 1)
    assert result.full_space_relative_residual == pytest.approx(full_relative, abs=3e-14)
    for column in rotated.T:
        assert column[np.argmax(np.abs(column))] >= 0


@pytest.mark.parametrize(("n", "r"), [(1, 1), (5, 1), (5, 3), (6, 6)])
def test_semicanonicalization_matches_independent_selected_fock(n, r):
    rng = np.random.default_rng(1928 + 10 * n + r)
    rotation, _ = np.linalg.qr(rng.normal(size=(n, n)))
    coefficients = np.ascontiguousarray(rotation[:, :r])
    raw_fock = rng.normal(size=(n, n))
    fock = np.ascontiguousarray((raw_fock + raw_fock.T) / 2)
    original_c, original_f = coefficients.copy(), fock.copy()
    result = _semicanonicalize(coefficients, fock)
    _assert_semicanonicalization(coefficients, fock, result)
    assert result.eigensolver_performed
    np.testing.assert_array_equal(coefficients, original_c)
    np.testing.assert_array_equal(fock, original_f)


def test_semicanonicalization_does_not_require_full_fock_invariance():
    coefficients = np.eye(3)[:, :2].copy()
    fock = np.array([[1.0, 0.3, 2.0], [0.3, 2.0, -1.0], [2.0, -1.0, 4.0]])
    result = _semicanonicalize(coefficients, fock)
    _assert_semicanonicalization(coefficients, fock, result)
    assert result.full_space_relative_residual > 0.1
    assert result.projected_fock_relative_residual < 1e-14
    np.testing.assert_array_equal(result.coefficients_copy()[2], [0, 0])


def test_semicanonicalization_zero_selected_fock_can_couple_outside():
    coefficients = np.eye(2)[:, :1].copy()
    fock = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = _semicanonicalize(coefficients, fock)
    _assert_semicanonicalization(coefficients, fock, result)
    np.testing.assert_array_equal(result.energies_copy(), [0.0])
    assert result.full_space_relative_residual > 0.5


def test_semicanonicalization_empty_selected_space_does_not_force_an_orbital():
    coefficients = np.empty((4, 0), dtype=np.float64)
    result = _semicanonicalize(coefficients, np.eye(4), cap=0)
    assert result.selected_dimension == 0
    assert not result.eigensolver_performed
    assert result.memory.peak_owned_numerical_bytes == 0
    assert result.coefficients_copy().shape == (4, 0)
    assert result.energies_copy().shape == (0,)
    _assert_semicanonicalization(coefficients, np.eye(4), result)


def test_semicanonicalization_degenerate_fock_preserves_projector():
    rng = np.random.default_rng(54)
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    coefficients = np.ascontiguousarray(rotation[:, :3])
    fock = 3 * np.eye(6)
    result = _semicanonicalize(coefficients, fock)
    _assert_semicanonicalization(coefficients, fock, result)
    np.testing.assert_allclose(result.energies_copy(), [3, 3, 3], atol=2e-14)


def test_semicanonicalization_accepts_selected_pno_coefficients_without_density():
    pnos = _build(_AMPLITUDES, cutoff=0.05)
    coefficients = pnos.coefficients_copy()
    del pnos
    gc.collect()
    fock = np.array([[1.0, 0.1, -0.3], [0.1, 1.8, 0.2], [-0.3, 0.2, 3.0]])
    result = _semicanonicalize(coefficients, fock)
    _assert_semicanonicalization(coefficients, fock, result)


def test_semicanonicalization_memory_phases_and_preallocation_gate():
    for n, r in [(1, 0), (8, 0), (8, 1), (8, 4), (3, 3), (1_000_000, 1)]:
        plan = core.plan_restricted_pair_semicanonicalization(n, r)
        assert plan.input_bytes == 8 * (n * n + n * r)
        assert plan.output_bytes == 8 * (n * r + r)
        expected_phases = (
            (8 * r * r + 8 * n, 40 * r * r + 8 * r,
             8 * n * r + 16 * r * r + 8 * r, 8 * n * r + 8 * r + 8 * n)
            if r else (0, 0, 0, 0)
        )
        assert (plan.projection_phase_bytes, plan.factorization_phase_bytes,
                plan.rotation_phase_bytes, plan.validation_phase_bytes) == expected_phases
        assert plan.peak_owned_numerical_bytes == max(expected_phases)
    n, r = 5, 2
    cap = core.plan_restricted_pair_semicanonicalization(n, r).peak_owned_numerical_bytes
    for insufficient in [0, cap - 1]:
        with pytest.raises(ValueError, match="byte cap is insufficient"):
            _semicanonicalize(np.full((n, r), np.nan), np.full((n, n), np.nan),
                             cap=insufficient)
    result = _semicanonicalize(np.eye(n)[:, :r].copy(), np.eye(n), cap=cap)
    assert result.memory.peak_owned_numerical_bytes == cap


@pytest.mark.parametrize(("n", "r"), [(0, 0), (2, 3), (2**32, 1)])
def test_semicanonicalization_count_only_shape_and_overflow_rejection(n, r):
    with pytest.raises((ValueError, OverflowError)):
        core.plan_restricted_pair_semicanonicalization(n, r)


@pytest.mark.parametrize("which", ["coefficients", "fock"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_semicanonicalization_rejects_nonfinite_input(which, bad):
    coefficients, fock = np.eye(3)[:, :2].copy(), np.eye(3)
    target = coefficients if which == "coefficients" else fock
    target[0, 0] = bad
    with pytest.raises(ValueError, match="must be finite"):
        _semicanonicalize(coefficients, fock)


def test_semicanonicalization_exact_fock_symmetry_and_orthonormality_gate():
    coefficients = np.eye(3)[:, :2].copy()
    fock = np.eye(3)
    fock[0, 1] = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="exactly symmetric"):
        _semicanonicalize(coefficients, fock)
    with pytest.raises(ValueError, match="not orthonormal"):
        _semicanonicalize(2 * coefficients, np.eye(3))
    coefficients[0, 0] *= 1 + 5e-11
    with pytest.raises(ValueError, match="not orthonormal"):
        _semicanonicalize(coefficients, np.eye(3), orthogonality_tolerance=1e-12)
    result = _semicanonicalize(coefficients, np.eye(3), orthogonality_tolerance=2e-10)
    assert 5e-11 < result.input_orthonormality_error < 2e-10
    assert result.output_orthonormality_error < 2e-10


@pytest.mark.parametrize("bad", [0.0, -1.0, 1.0, np.nan, np.inf])
def test_semicanonicalization_requires_explicit_valid_orthonormality_tolerance(bad):
    with pytest.raises(ValueError, match="numerical tolerances"):
        _semicanonicalize(np.eye(2), np.eye(2), orthogonality_tolerance=bad)


def test_semicanonicalization_rejects_nonconvergence_and_sweep_controls():
    rng = np.random.default_rng(412)
    raw = rng.normal(size=(5, 5))
    fock = (raw + raw.T) / 2
    with pytest.raises(RuntimeError, match="eigensolver failed"):
        _semicanonicalize(np.eye(5), fock, sweeps=1)
    with pytest.raises(ValueError, match="numerical tolerances"):
        _semicanonicalize(np.eye(5), fock, sweeps=0)
    with pytest.raises(ValueError, match="200 eigensolver sweeps"):
        _semicanonicalize(np.eye(5), fock, sweeps=201)


def test_semicanonicalization_readonly_inputs_and_owned_result_lifetime():
    coefficients = np.eye(3)[:, :2].copy()
    fock = np.array([[1.0, 0.4, 0.2], [0.4, 2.0, 0.1], [0.2, 0.1, 4.0]])
    coefficients.setflags(write=False)
    fock.setflags(write=False)
    result = _semicanonicalize(coefficients, fock)
    _assert_semicanonicalization(coefficients, fock, result)
    rotated, energies = result.coefficients_copy(), result.energies_copy()
    del result, coefficients, fock
    gc.collect()
    np.testing.assert_allclose(rotated.T @ rotated, np.eye(2), atol=1e-14)
    assert np.all(np.isfinite(energies))


def test_semicanonicalization_subnormal_fock_and_representable_extreme_energy():
    tiny = np.nextafter(0.0, 1.0)
    result = _semicanonicalize(np.eye(2), np.diag([tiny, 2 * tiny]))
    np.testing.assert_array_equal(result.energies_copy(), [tiny, 2 * tiny])
    assert result.fock_scaling_underflow_entries == 0
    largest = np.finfo(float).max
    result = _semicanonicalize(np.ones((1, 1)), np.array([[largest]]))
    np.testing.assert_array_equal(result.energies_copy(), [largest])
    with pytest.raises(OverflowError, match="semicanonical energy"):
        _semicanonicalize(np.eye(2), np.full((2, 2), largest))


def test_semicanonicalization_scaling_loss_is_reported():
    tiny = np.nextafter(0.0, 1.0)
    result = _semicanonicalize(np.eye(2), np.diag([np.finfo(float).max, tiny]))
    assert result.fock_scaling_underflow_entries == 1


@pytest.mark.parametrize(("coefficients", "fock"), [
    (np.ones((3, 2), dtype=np.float32), np.eye(3)),
    (np.eye(3), np.eye(3, dtype=np.float32)),
    (np.asfortranarray(np.eye(3)), np.eye(3)),
    (np.eye(3), np.asfortranarray(np.eye(3))),
    (np.ones((3, 4)), np.eye(3)),
    (np.eye(3), np.eye(2)),
    (np.ones((0, 0)), np.ones((0, 0))),
])
def test_semicanonicalization_binding_does_not_copy_invalid_arrays(coefficients, fock):
    with pytest.raises(ValueError, match=r"C-contiguous float64 C\(n,r\)"):
        core.restricted_pair_semicanonicalize(coefficients, fock, 1e-12, 10**6, 80, _TOL)


def test_semicanonicalization_diagnostic_dimension_cap_precedes_numerics():
    with pytest.raises(ValueError, match="limited to 128"):
        core.restricted_pair_semicanonicalize(
            np.full((129, 1), np.nan), np.full((129, 129), np.nan), 1e-12, 10**8, 80, _TOL,
        )


def _build(
    amplitudes: np.ndarray,
    *,
    diagonal: bool = False,
    cutoff: float = 0.0,
    cap: int | None = None,
    sweeps: int = 80,
    tolerance: float = _TOL,
):
    if cap is None:
        cap = core.plan_restricted_pair_pnos(
            amplitudes.shape[0]
        ).peak_owned_numerical_bytes
    kind = (
        core.RestrictedPairKind.Diagonal
        if diagonal
        else core.RestrictedPairKind.OffDiagonal
    )
    return core.restricted_pair_pnos(
        amplitudes, kind, cutoff, cap, sweeps, tolerance
    )


def _paper_density(amplitudes: np.ndarray, *, diagonal: bool = False):
    # Eq. (23): the plus sign on a matrix denotes the adjoint. The domain
    # and amplitudes here are real, so it is an ordinary transpose.
    t_tilde = (4.0 * amplitudes - 2.0 * amplitudes.T) / (2 if diagonal else 1)
    return t_tilde @ amplitudes.T + t_tilde.T @ amplitudes


def _assert_eigensystem(result):
    density = result.density_copy()
    coefficients = result.coefficients_copy()
    occupations = result.occupations_copy()
    retained = result.retained_dimension
    np.testing.assert_allclose(density, density.T, rtol=0, atol=0)
    np.testing.assert_allclose(
        coefficients.T @ coefficients, np.eye(retained), rtol=0, atol=2e-13
    )
    np.testing.assert_allclose(
        density @ coefficients,
        coefficients * occupations[:retained],
        rtol=2e-12,
        atol=2e-14,
    )
    assert np.all(occupations[:-1] >= occupations[1:])
    for column in coefficients.T:
        assert column[np.argmax(np.abs(column))] >= 0
    assert result.negative_occupation_count == np.count_nonzero(occupations < 0)
    assert result.minimum_occupation == occupations[-1]
    assert result.eigensystem_relative_residual < 2e-12
    assert result.eigenvector_orthogonality_error < 2e-12


def test_spin_adapted_density_matches_independent_paper_equation():
    result = _build(_AMPLITUDES)
    expected = _paper_density(_AMPLITUDES)
    np.testing.assert_allclose(result.density_copy(), expected, rtol=2e-15, atol=1e-17)
    np.testing.assert_allclose(
        result.occupations_copy(), np.linalg.eigvalsh(expected)[::-1],
        rtol=3e-14, atol=1e-16,
    )
    assert result.retained_dimension == 3
    assert result.density_trace == pytest.approx(np.trace(expected), rel=2e-15)
    assert result.discarded_occupation_sum == 0
    # The unadapted TT^T+T^TT convention does not reproduce Eq. (23).
    unadapted = _AMPLITUDES @ _AMPLITUDES.T + _AMPLITUDES.T @ _AMPLITUDES
    assert np.linalg.norm(expected - unadapted) > 0.01
    _assert_eigensystem(result)


def test_diagonal_pair_normalization_is_explicit_and_not_energy_multiplicity():
    amplitudes = np.array([[0.2, -0.03], [-0.03, 0.1]])
    diagonal = _build(amplitudes, diagonal=True)
    off_diagonal = _build(amplitudes)
    np.testing.assert_allclose(
        diagonal.density_copy(), _paper_density(amplitudes, diagonal=True),
        rtol=2e-15, atol=1e-17,
    )
    np.testing.assert_allclose(
        off_diagonal.density_copy(), 2.0 * diagonal.density_copy(), rtol=0, atol=0
    )
    assert diagonal.discarded_antisymmetric_norm == 0
    assert diagonal.input_relative_antisymmetric_norm == 0
    _assert_eigensystem(diagonal)


def test_pair_reversal_has_identical_density_and_retained_projector():
    first = _build(_AMPLITUDES, cutoff=0.05)
    reversed_pair = _build(np.ascontiguousarray(_AMPLITUDES.T), cutoff=0.05)
    np.testing.assert_array_equal(first.density_copy(), reversed_pair.density_copy())
    np.testing.assert_array_equal(first.occupations_copy(), reversed_pair.occupations_copy())
    np.testing.assert_array_equal(first.coefficients_copy(), reversed_pair.coefficients_copy())


def test_virtual_rotation_covariance_of_density_and_selected_subspace():
    rotation, _ = np.linalg.qr(
        np.array([[1.0, 2.0, -1.0], [2.0, -1.0, 3.0], [1.0, 1.0, 2.0]])
    )
    eigenvalues = np.linalg.eigvalsh(_paper_density(_AMPLITUDES))
    cutoff = float((eigenvalues[0] + eigenvalues[1]) / 2)
    original = _build(_AMPLITUDES, cutoff=cutoff)
    rotated = _build(
        np.ascontiguousarray(rotation.T @ _AMPLITUDES @ rotation), cutoff=cutoff
    )
    assert original.retained_dimension == rotated.retained_dimension == 2
    np.testing.assert_allclose(
        rotated.density_copy(), rotation.T @ original.density_copy() @ rotation,
        rtol=4e-14, atol=2e-16,
    )
    v_original = original.coefficients_copy()
    v_rotated = rotation @ rotated.coefficients_copy()
    np.testing.assert_allclose(
        v_original @ v_original.T, v_rotated @ v_rotated.T, rtol=2e-13, atol=2e-13
    )
    _assert_eigensystem(rotated)


@pytest.mark.parametrize(
    ("cutoff", "retained"),
    [(0.0, 3), (1.0, 1), (np.nextafter(1.0, 0.0), 2),
     (np.nextafter(1.0, np.inf), 1), (4.0, 0)],
)
def test_positive_cutoff_is_strict_and_zero_explicitly_keeps_null_space(cutoff, retained):
    result = _build(np.diag([1.0, 0.5, 0.0]), cutoff=cutoff)
    np.testing.assert_array_equal(result.occupations_copy(), [4.0, 1.0, 0.0])
    assert result.retained_dimension == retained
    assert result.coefficients_copy().shape == (3, retained)
    assert result.discarded_occupation_sum == sum([4.0, 1.0, 0.0][retained:])
    assert result.output_numerical_bytes == 8 * (9 + 3 + 3 * retained)
    _assert_eigensystem(result)


def test_zero_density_is_valid_for_full_and_empty_retained_spaces():
    amplitudes = np.zeros((4, 4))
    full = _build(amplitudes, diagonal=True)
    empty = _build(amplitudes, diagonal=True, cutoff=1e-12)
    assert full.retained_dimension == 4
    assert empty.retained_dimension == 0
    np.testing.assert_array_equal(full.density_copy(), amplitudes)
    np.testing.assert_array_equal(full.occupations_copy(), np.zeros(4))
    _assert_eigensystem(full)
    _assert_eigensystem(empty)


def test_rank_deficient_zero_cutoff_keeps_complete_orthonormal_domain():
    vector = np.array([0.1, 0.03, -0.06])
    amplitudes = np.outer(vector, vector)
    result = _build(amplitudes, diagonal=True)
    assert result.retained_dimension == 3
    coefficients = result.coefficients_copy()
    np.testing.assert_allclose(coefficients @ coefficients.T, np.eye(3), atol=2e-14)
    assert result.minimum_occupation >= -1e-16
    _assert_eigensystem(result)


def test_diagonal_symmetry_error_is_rejected_at_any_amplitude_scale():
    for scale in (1e-100, 1.0, 1e100):
        with pytest.raises(ValueError, match="not symmetric"):
            _build(_AMPLITUDES * scale, diagonal=True)


def test_accepted_diagonal_roundoff_is_symmetrized_and_reported():
    amplitudes = np.array([[0.2, 0.03], [0.03 + 4e-16, 0.1]])
    result = _build(amplitudes, diagonal=True)
    symmetric = (amplitudes + amplitudes.T) / 2
    antisymmetric = (amplitudes - amplitudes.T) / 2
    assert result.discarded_antisymmetric_norm == pytest.approx(
        np.linalg.norm(antisymmetric), rel=2e-15
    )
    assert 0 < result.input_relative_antisymmetric_norm < 64 * np.finfo(float).eps
    np.testing.assert_allclose(
        result.density_copy(), _paper_density(symmetric, diagonal=True),
        rtol=2e-15, atol=1e-17,
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_amplitudes_are_rejected(bad):
    amplitudes = _AMPLITUDES.copy()
    amplitudes[0, 1] = bad
    with pytest.raises(ValueError, match="amplitudes must be finite"):
        _build(amplitudes)


@pytest.mark.parametrize("cutoff", [-1e-9, np.nan, np.inf, -np.inf])
def test_invalid_occupation_cutoffs_are_rejected(cutoff):
    with pytest.raises(ValueError, match="cutoff must be finite and nonnegative"):
        _build(_AMPLITUDES, cutoff=cutoff)


def test_large_representable_density_and_subnormal_density():
    large = _build(np.array([[1e150]]))
    assert large.density_copy()[0, 0] == pytest.approx(4e300, rel=3e-15)
    small = _build(np.array([[1e-160]]))
    density = small.density_copy()[0, 0]
    assert density > 0
    assert abs(density - 4e-160 * 1e-160) <= 2 * np.nextafter(0.0, 1.0)
    assert small.occupations_copy()[0] == density


def test_unrepresentable_density_is_rejected_and_underflow_is_reported():
    with pytest.raises(OverflowError, match="density element is not finite"):
        _build(np.array([[1e308]]))
    underflow = _build(np.array([[1e-200]]))
    assert underflow.density_copy()[0, 0] == 0
    assert underflow.density_underflow_entry_count == 1
    assert underflow.retained_dimension == 1
    mixed = _build(np.diag([1e150, 1e-200]))
    assert mixed.amplitude_scaling_underflow_count == 1


def test_memory_plan_is_exact_and_cap_precedes_numerical_input_inspection():
    for dimension in (1, 3, 128, 1_000_000):
        plan = core.plan_restricted_pair_pnos(dimension)
        assert plan.input_bytes == 8 * dimension**2
        assert plan.output_bytes_upper_bound == 16 * dimension**2 + 8 * dimension
        assert plan.eigensolver_workspace_bytes == 32 * dimension**2
        assert plan.peak_owned_numerical_bytes == 40 * dimension**2 + 8 * dimension
    plan = core.plan_restricted_pair_pnos(3)
    result = _build(_AMPLITUDES, cap=plan.peak_owned_numerical_bytes)
    assert result.memory.peak_owned_numerical_bytes == plan.peak_owned_numerical_bytes
    for cap in (0, plan.peak_owned_numerical_bytes - 1):
        with pytest.raises(ValueError, match="byte cap is insufficient"):
            _build(np.full((3, 3), np.nan), cap=cap)


def test_count_only_plan_rejects_zero_and_overflow_without_allocation():
    with pytest.raises(ValueError, match="dimension must be positive"):
        core.plan_restricted_pair_pnos(0)
    for dimension in (2**32, 2**64 - 1):
        with pytest.raises(OverflowError, match="overflows uint64"):
            core.plan_restricted_pair_pnos(dimension)
    with pytest.raises(ValueError, match="dimension must be positive"):
        _build(np.empty((0, 0)), cap=1)


@pytest.mark.parametrize(
    "amplitudes",
    [np.ones((2, 3)), np.ones(3), np.eye(3, dtype=np.float32),
     np.eye(3, dtype=complex), np.asfortranarray(_AMPLITUDES),
     np.ones((6, 6))[::2, ::2]],
)
def test_binding_refuses_implicit_dtype_and_layout_copies(amplitudes):
    with pytest.raises(ValueError, match="square C-contiguous float64"):
        _build(amplitudes, cap=4096)


def test_binding_caps_domain_and_iteration_work():
    dimension = core.PAIR_PNO_DIAGNOSTIC_MAXIMUM_DIMENSION + 1
    with pytest.raises(ValueError, match="dimension is limited"):
        _build(np.zeros((dimension, dimension)))
    with pytest.raises(ValueError, match="limited to 200"):
        _build(_AMPLITUDES, sweeps=core.PAIR_PNO_DIAGNOSTIC_MAXIMUM_SWEEPS + 1)


@pytest.mark.parametrize(
    ("sweeps", "tolerance"),
    [(0, _TOL), (80, 0.0), (80, -1.0), (80, 1.0), (80, np.nan)],
)
def test_invalid_eigensolver_controls_are_rejected(sweeps, tolerance):
    with pytest.raises(ValueError, match="eigensolver options are invalid"):
        _build(_AMPLITUDES, sweeps=sweeps, tolerance=tolerance)


def test_unconverged_eigensystem_does_not_return_pnos():
    with pytest.raises(RuntimeError, match="eigensolver failed"):
        _build(_AMPLITUDES, sweeps=1, tolerance=1e-16)


def test_input_is_unmodified_and_result_copies_have_independent_lifetimes():
    amplitudes = _AMPLITUDES.copy()
    original = amplitudes.copy()
    amplitudes.flags.writeable = False
    result = _build(amplitudes)
    density = result.density_copy()
    coefficients = result.coefficients_copy()
    occupations = result.occupations_copy()
    density[:] = 0
    assert np.linalg.norm(result.density_copy()) > 0
    np.testing.assert_array_equal(amplitudes, original)
    del result
    gc.collect()
    assert np.all(np.isfinite(coefficients))
    assert np.all(np.isfinite(occupations))
