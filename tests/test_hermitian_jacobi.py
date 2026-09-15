"""Independent small-matrix residual tests for the native scalar eigensolver.

Successful cases use dimension <= 16; one dimension-32 diagonal checks the
diagnostic boundary. No basis, integral, SCF, or chemistry calculation runs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


_STATUS = core._HermitianJacobiStatus
_TOLERANCE = 8.0e-15


def _solve(matrix, *, sweeps=100, tolerance=_TOLERANCE):
    return core._hermitian_jacobi_diagnostic(matrix, sweeps, tolerance)


def _random_hermitian(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return np.ascontiguousarray(0.5 * (raw + raw.conj().T))


def _scaled_complex(matrix: np.ndarray, exponent: int) -> np.ndarray:
    # Separate component scaling avoids first creating an overflowing or
    # underflowing reciprocal of the scale itself.
    return np.ldexp(matrix.real, -exponent) + 1j * np.ldexp(matrix.imag, -exponent)


def _check_eigensystem(original: np.ndarray, result: dict) -> None:
    assert result["status"] == _STATUS.SUCCESS
    values = result["eigenvalues"]
    vectors = result["eigenvectors"]
    exponent = result["input_scale_exponent"]
    scaled = _scaled_complex(original, exponent)
    scaled_values = np.ldexp(values, -exponent)
    denominator = np.linalg.norm(scaled)
    residual = scaled @ vectors - vectors * scaled_values[None, :]
    assert np.linalg.norm(residual) <= 3.0e-13 * max(denominator, 1.0)
    orthogonality = np.linalg.norm(vectors.conj().T @ vectors - np.eye(len(values)))
    assert orthogonality < 3.0e-13
    assert result["orthogonality_frobenius_error"] < 3.0e-13
    assert abs(result["orthogonality_frobenius_error"] - orthogonality) < 1.0e-14
    assert np.all(np.diff(values) >= 0.0)
    np.testing.assert_allclose(
        scaled_values, np.linalg.eigvalsh(scaled), rtol=2.0e-13,
        atol=2.0e-13 * max(denominator, 1.0),
    )
    transformed = result["matrix_scaled"]
    offdiagonal = transformed.copy()
    np.fill_diagonal(offdiagonal, 0.0)
    assert np.linalg.norm(offdiagonal) <= _TOLERANCE * max(denominator, 1.0)
    assert result["scaled_offdiagonal_frobenius_norm"] == pytest.approx(
        np.linalg.norm(offdiagonal), rel=1.0e-13, abs=1.0e-300,
    )
    np.testing.assert_allclose(
        vectors.conj().T @ scaled @ vectors, transformed,
        rtol=3.0e-13, atol=3.0e-13 * max(denominator, 1.0),
    )


@pytest.mark.parametrize("n", range(1, 17))
def test_random_hermitian_independent_original_residual(n):
    matrix = _random_hermitian(n, 2026090400 + n)
    before = matrix.copy()
    result = _solve(matrix)
    _check_eigensystem(before, result)
    np.testing.assert_array_equal(matrix, before)


@pytest.mark.parametrize("pivot", [1j, 0.6 + 0.8j, -0.3 + 0.4j, 0.0 + 1e-300j])
def test_complex_two_by_two_pivot(pivot):
    matrix = np.array([[1.0, pivot], [np.conj(pivot), 2.0]], dtype=np.complex128)
    result = _solve(matrix)
    _check_eigensystem(matrix, result)
    gap = math.hypot(0.5, abs(pivot))
    np.testing.assert_allclose(result["eigenvalues"], [1.5 - gap, 1.5 + gap])


def test_zero_matrix_is_explicit_success_without_sweeps():
    result = _solve(np.zeros((4, 4), dtype=np.complex128))
    assert result["status"] == _STATUS.SUCCESS
    assert result["sweeps"] == 0
    assert result["rotations"] == 0
    assert result["scaled_input_frobenius_norm"] == 0.0
    assert result["scaled_offdiagonal_frobenius_norm"] == 0.0
    np.testing.assert_array_equal(result["eigenvalues"], np.zeros(4))
    np.testing.assert_array_equal(result["eigenvectors"], np.eye(4))


def test_indefinite_spectrum_is_not_a_solver_failure():
    matrix = np.array([[0, 1j], [-1j, 0]], dtype=np.complex128)
    result = _solve(matrix)
    _check_eigensystem(matrix, result)
    np.testing.assert_allclose(result["eigenvalues"], [-1, 1], atol=2e-15)


def test_repeated_spectrum_checks_subspace_not_eigenvector_gauge():
    rng = np.random.default_rng(414)
    unitary, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    values = np.array([-2.0, -2.0, 3.0, 3.0])
    raw = (unitary * values) @ unitary.conj().T
    matrix = np.ascontiguousarray(0.5 * (raw + raw.conj().T))
    result = _solve(matrix)
    _check_eigensystem(matrix, result)
    np.testing.assert_allclose(result["eigenvalues"], values, atol=5e-14)
    expected_projector = unitary[:, :2] @ unitary[:, :2].conj().T
    actual = result["eigenvectors"][:, :2]
    np.testing.assert_allclose(actual @ actual.conj().T, expected_projector, atol=5e-14)


def test_equal_diagonal_eigenvalues_use_stable_sort():
    matrix = np.diag([3.0, -1.0, 3.0, -1.0]).astype(np.complex128)
    result = _solve(matrix)
    np.testing.assert_array_equal(result["eigenvalues"], [-1.0, -1.0, 3.0, 3.0])
    np.testing.assert_array_equal(result["eigenvectors"], np.eye(4)[:, [1, 3, 0, 2]])


def test_conjugation_covariance_and_repeat_determinism():
    matrix = _random_hermitian(9, 98761)
    result = _solve(matrix)
    repeated = _solve(matrix)
    conjugated = _solve(np.ascontiguousarray(matrix.conj()))
    np.testing.assert_array_equal(result["eigenvalues"], repeated["eigenvalues"])
    np.testing.assert_array_equal(result["eigenvectors"], repeated["eigenvectors"])
    np.testing.assert_allclose(result["eigenvalues"], conjugated["eigenvalues"], atol=1e-14)
    np.testing.assert_allclose(
        result["eigenvectors"].conj(), conjugated["eigenvectors"], atol=1e-14,
    )


@pytest.mark.parametrize("exponent", [-1000, 1000])
def test_power_of_two_scaling_preserves_eigensystem(exponent):
    base = _random_hermitian(6, 119)
    matrix = np.ldexp(base.real, exponent) + 1j * np.ldexp(base.imag, exponent)
    _check_eigensystem(matrix, _solve(matrix))


def test_subnormal_complex_phase_during_an_ordinary_sweep_stays_unitary():
    tiny = np.nextafter(0.0, 1.0)
    matrix = np.zeros((4, 4), dtype=np.complex128)
    matrix[0, 0], matrix[1, 1] = 0.5, 0.75
    matrix[0, 1], matrix[1, 0] = 0.1 + 0.07j, 0.1 - 0.07j
    matrix[2, 3], matrix[3, 2] = complex(tiny, tiny), complex(tiny, -tiny)
    result = _solve(matrix)
    assert result["status"] == _STATUS.SUCCESS
    assert result["rotations"] >= 2
    assert result["scaling_underflow_components"] == 0
    assert result["orthogonality_frobenius_error"] < 2e-14
    _check_eigensystem(matrix, result)


def test_entirely_subnormal_input_validates_vectors_in_the_scaled_problem():
    # Physical eigenvalues must round to the subnormal grid. Their relative
    # residual need not be small; the scaled work matrix retains the resolved
    # values and lets us test the rotation separately from output rounding.
    tiny = np.nextafter(0.0, 1.0)
    matrix = np.array([[0, complex(tiny, tiny)], [complex(tiny, -tiny), 0]])
    result = _solve(matrix)
    assert result["status"] == _STATUS.SUCCESS
    assert result["input_scale_exponent"] == -1074
    assert result["scaling_underflow_components"] == 0
    vectors = result["eigenvectors"]
    scaled = _scaled_complex(matrix, result["input_scale_exponent"])
    scaled_values = result["matrix_scaled"].diagonal().real
    np.testing.assert_allclose(scaled @ vectors, vectors * scaled_values, atol=2e-15)
    np.testing.assert_allclose(scaled_values, [-math.sqrt(2), math.sqrt(2)], atol=2e-15)
    np.testing.assert_array_equal(result["eigenvalues"], [-tiny, tiny])


def test_scaling_loss_is_reported_instead_of_claiming_original_residual():
    tiny = np.nextafter(0.0, 1.0)
    matrix = np.diag([np.finfo(float).max, tiny]).astype(np.complex128)
    result = _solve(matrix)
    assert result["status"] == _STATUS.SUCCESS
    assert result["scaling_underflow_components"] == 1


def test_largest_finite_diagonal_remains_representable():
    largest = np.finfo(float).max
    matrix = np.diag([largest, -largest]).astype(np.complex128)
    result = _solve(matrix)
    assert result["status"] == _STATUS.SUCCESS
    np.testing.assert_array_equal(result["eigenvalues"], [-largest, largest])
    assert math.isfinite(result["input_scale"])


@pytest.mark.parametrize("complex_pivot", [False, True])
def test_unrepresentable_eigenvalue_is_explicit_failure(complex_pivot):
    largest = np.finfo(float).max
    matrix = np.full((2, 2), largest, dtype=np.complex128)
    if complex_pivot:
        matrix[0, 0] = matrix[1, 1] = 0
        matrix[0, 1], matrix[1, 0] = complex(largest, largest), complex(largest, -largest)
    result = _solve(matrix)
    assert result["status"] == _STATUS.NUMERICAL_FAILURE


def test_exhausted_sweep_budget_is_explicit_failure():
    result = _solve(_random_hermitian(8, 472), sweeps=1)
    assert result["status"] == _STATUS.NO_CONVERGENCE
    assert result["sweeps"] == 1
    assert result["scaled_offdiagonal_frobenius_norm"] > (
        _TOLERANCE * result["scaled_input_frobenius_norm"]
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_matrix_rejected_before_mutation(bad):
    matrix = np.eye(2, dtype=np.complex128)
    matrix[0, 1] = complex(bad, 0)
    result = _solve(matrix)
    assert result["status"] == _STATUS.NON_FINITE_INPUT
    np.testing.assert_array_equal(result["matrix_scaled"], matrix)
    np.testing.assert_array_equal(result["eigenvectors"], np.zeros((2, 2)))


@pytest.mark.parametrize("diagonal", [False, True])
def test_nonhermitian_matrix_rejected_exactly_before_mutation(diagonal):
    matrix = np.eye(2, dtype=np.complex128)
    tiny = np.nextafter(0.0, 1.0)
    if diagonal:
        matrix[0, 0] += complex(0, tiny)
    else:
        matrix[0, 1] = tiny
    result = _solve(matrix)
    assert result["status"] == _STATUS.NOT_HERMITIAN
    np.testing.assert_array_equal(result["matrix_scaled"], matrix)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, 1.0, float("nan"), float("inf")])
def test_invalid_numerical_tolerance_returns_status(tolerance):
    matrix = np.eye(2, dtype=np.complex128)
    result = _solve(matrix, tolerance=tolerance)
    assert result["status"] == _STATUS.INVALID_INPUT
    np.testing.assert_array_equal(result["matrix_scaled"], matrix)


def test_zero_sweep_budget_is_invalid_even_for_diagonal_input():
    result = _solve(np.eye(2, dtype=np.complex128), sweeps=0)
    assert result["status"] == _STATUS.INVALID_INPUT


def test_non_nearest_rounding_is_rejected_and_environment_is_restored():
    assert core._hermitian_jacobi_non_nearest_probe() == _STATUS.INVALID_INPUT
    result = _solve(np.eye(2, dtype=np.complex128))
    assert result["status"] == _STATUS.SUCCESS


@pytest.mark.parametrize("shape", [(0, 0), (33, 33), (2, 3), (4,)])
def test_shape_and_dimension_cap_rejected_before_copy(shape):
    with pytest.raises(ValueError, match="dimension 1..32"):
        _solve(np.zeros(shape, dtype=np.complex128))


@pytest.mark.parametrize("matrix", [
    np.eye(2, dtype=np.float64),
    np.eye(2, dtype=np.complex64),
    np.asfortranarray(np.eye(2, dtype=np.complex128)),
    np.eye(4, dtype=np.complex128)[::2, ::2],
    np.eye(2, dtype=np.dtype(">c16")),
])
def test_non_native_or_non_contiguous_input_is_not_implicitly_copied(matrix):
    with pytest.raises(ValueError, match="C-contiguous native complex128"):
        _solve(matrix)


def test_diagnostic_sweep_cap_and_required_controls():
    matrix = np.eye(2, dtype=np.complex128)
    with pytest.raises(ValueError, match="1024 sweeps"):
        _solve(matrix, sweeps=1025)
    with pytest.raises(TypeError):
        core._hermitian_jacobi_diagnostic(matrix)


def test_dimension_cap_positive_control():
    assert core._HERMITIAN_JACOBI_DIAGNOSTIC_MAXIMUM_DIMENSION == 32
    result = _solve(np.eye(32, dtype=np.complex128))
    assert result["status"] == _STATUS.SUCCESS
    assert result["sweeps"] == 0
