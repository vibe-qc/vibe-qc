"""Tiny numerical tests of the native original-AO principal metric whitener."""

from __future__ import annotations

import gc
import hashlib
import math
import struct

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_reciprocal_metric import (
    _raw_metric_bundle,
    _auxiliary_basis,
    _mixed_s_p_metric_basis,
    _two_s_metric_basis,
)


def _sqrt(matrix, rank=1e-10, negative=1e-12, self=False):
    return core._metric_principal_square_root_diagnostic(
        np.asarray(matrix, dtype=np.complex128, order="C"), rank, negative, self
    )


def _oracle(matrix, cutoff):
    values, vectors = np.linalg.eigh(matrix)
    keep = values > cutoff
    return ((vectors[:, keep] / np.sqrt(values[keep]))
            @ vectors[:, keep].conj().T), vectors[:, keep] @ vectors[:, keep].conj().T


def _physical_bundle(*, q=1, block=2, columns=None, basis=None, backend=None,
                     mesh=(3, 1, 1)):
    bundle = _raw_metric_bundle(reciprocal_block=block, basis=basis, mesh=mesh)
    config = bundle.source_config
    config.backend_identity_sha256 = backend or (
        core._periodic_correlation_metric_factorization_backend_identity_sha256()
    )
    if columns is not None:
        config.whitener_column_block = columns
    census = core._make_periodic_correlation_factor_build_census(
        bundle.reference, bundle.schedule, config,
        [source.factor_build_q_record() for source in bundle.sources],
    )
    raw = core._build_periodic_correlation_reciprocal_metric(
        bundle.reference, bundle.schedule, census, bundle.sources[q], bundle.basis
    )
    return bundle, census, raw


def _factor(bundle, census, raw, negative=1e-12):
    return core._factorize_periodic_correlation_metric(
        bundle.reference, bundle.schedule, census, raw, negative
    )


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 16])
@pytest.mark.parametrize("seed", [3, 17, 81])
def test_complex_psd_numpy_oracle_and_principal_identities(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    matrix = x @ x.conj().T + np.eye(n) * 0.2
    matrix = np.triu(matrix) + np.triu(matrix, 1).conj().T
    matrix[np.diag_indices(n)] = matrix.diagonal().real
    result = _sqrt(matrix)
    w = result["matrix"]
    expected, projector = _oracle(matrix, 1e-10)
    np.testing.assert_allclose(w, expected, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(w @ matrix @ w, projector, atol=4e-12)
    np.testing.assert_allclose(w @ w, np.linalg.inv(matrix), atol=4e-12)
    assert np.array_equal(w, w.conj().T)
    assert result["diagnostics"].retained_rank == n


def test_rank_boundary_tolerated_negative_and_zero_subspace():
    cutoff = 1e-6
    above = np.nextafter(cutoff, np.inf)
    matrix = np.diag([-1e-12, 0, cutoff, above, 4])
    result = _sqrt(matrix, rank=cutoff, negative=1e-12, self=True)
    d = result["diagnostics"]
    assert d.retained_rank == 2
    assert d.largest_discarded_eigenvalue == cutoff
    np.testing.assert_array_equal(
        result["matrix"], np.diag([0, 0, 0, 1 / math.sqrt(above), 0.5])
    )
    with pytest.raises(RuntimeError, match="indefinite"):
        _sqrt(np.diag([np.nextafter(-1e-12, -np.inf), 4]))
    with pytest.raises(RuntimeError, match="rank is zero"):
        _sqrt(np.zeros((2, 2)))


def test_complex_rank_deficient_degenerate_subspace():
    q = np.array([[1, 1j], [1j, 1]], dtype=complex) / math.sqrt(2)
    matrix = (q * [0.0, 2.0]) @ q.conj().T
    result = _sqrt(matrix)
    expected, projector = _oracle(matrix, 1e-10)
    np.testing.assert_allclose(result["matrix"], expected, atol=2e-15)
    np.testing.assert_allclose(result["matrix"] @ matrix @ result["matrix"], projector, atol=2e-15)
    assert result["diagnostics"].retained_rank == 1


def test_nondiagonal_spectrum_straddles_absolute_cutoff_at_large_scale():
    # Rank classification is deliberately away from the roundoff uncertainty
    # of the cutoff, while still spanning ten orders of magnitude.
    q = np.array([[1, 1j], [1j, 1]], dtype=complex) / math.sqrt(2)
    unitary = np.zeros((4, 4), dtype=complex)
    unitary[:2, :2] = q
    unitary[2:, 2:] = q
    matrix = (unitary * [1e-8, 1e-4, 1, 1e2]) @ unitary.conj().T
    matrix = np.triu(matrix) + np.triu(matrix, 1).conj().T
    matrix[np.diag_indices(4)] = matrix.diagonal().real
    result = _sqrt(matrix, rank=1e-6, negative=1e-9)
    expected, projector = _oracle(matrix, 1e-6)
    assert result["diagnostics"].retained_rank == 3
    np.testing.assert_allclose(result["matrix"], expected, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(result["matrix"] @ matrix @ result["matrix"], projector, atol=2e-11)


@pytest.mark.parametrize("rank,negative", [
    (0, 1e-12), (1e-10, 0), (-1, 1e-12), (1e-10, 1e-9),
    (np.inf, 1e-12), (1e-10, np.nan),
])
def test_cutoff_controls_are_explicit_and_separate(rank, negative):
    with pytest.raises(ValueError, match="negative_tolerance"):
        _sqrt(np.eye(2), rank, negative)


def test_nonhermitian_nonfinite_and_nonreal_self_fail_closed():
    with pytest.raises(ValueError, match="Hermitian"):
        _sqrt([[1, 1j], [1j, 2]])
    with pytest.raises(ValueError, match="non-finite"):
        _sqrt([[np.nan]])
    with pytest.raises(ValueError, match="real AO gauge"):
        _sqrt([[2, 1j], [-1j, 2]], self=True)


def test_diagnostic_caps_and_no_implicit_array_cast():
    with pytest.raises((ValueError, RuntimeError), match="1..16"):
        _sqrt(np.eye(17))
    with pytest.raises(ValueError, match="complex128"):
        core._metric_principal_square_root_diagnostic(np.eye(2), 1e-10, 1e-12)


@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("basis_factory", [_two_s_metric_basis, _mixed_s_p_metric_basis])
def test_actual_periodic_metric_composes_with_principal_factorization(q, basis_factory):
    bundle, census, raw = _physical_bundle(q=q, basis=basis_factory())
    matrix = raw.matrix_copy()
    input_hash = raw.payload_identity_sha256
    result = _factor(bundle, census, raw)
    expected, projector = _oracle(matrix, 1e-10)
    np.testing.assert_allclose(result.matrix, expected, atol=4e-11, rtol=3e-11)
    np.testing.assert_allclose(result.matrix @ matrix @ result.matrix, projector, atol=3e-11)
    assert result.input_payload_identity_sha256 == input_hash
    assert result.q_index == q
    assert result.state.state_identity_sha256 == bundle.state.state_identity_sha256
    assert result.numerical_peak_bytes == 32 * len(matrix) ** 2 + 8 * len(matrix)
    with pytest.raises((ValueError, RuntimeError), match="provenance"):
        _factor(bundle, census, raw)
    if q == 0:
        assert np.count_nonzero(result.matrix.imag) == 0


def test_block_invariance_q_conjugation_and_owned_lifetime():
    matrices = []
    hashes = []
    for block, columns in [(1, 1), (2, 2), (5, 1)]:
        bundle, census, raw = _physical_bundle(block=block, columns=columns)
        result = _factor(bundle, census, raw)
        matrices.append(result.matrix)
        hashes.append(result.payload_identity_sha256)
    for matrix in matrices[1:]:
        np.testing.assert_array_equal(matrix, matrices[0])
    assert len(set(hashes)) == 1
    bundle, census, raw = _physical_bundle(q=2)
    partner = _factor(bundle, census, raw)
    np.testing.assert_allclose(partner.matrix, matrices[0].conj(), atol=2e-14)
    state_hash = partner.state.state_identity_sha256
    del bundle, census, raw
    gc.collect()
    assert partner.state.state_identity_sha256 == state_hash
    copied = partner.matrix
    copied[:] = -5
    assert not np.array_equal(partner.matrix, copied)


def test_wrong_backend_rejected_before_raw_consumption():
    bundle, census, raw = _physical_bundle(backend="e" * 64)
    original = raw.matrix_copy()
    with pytest.raises(ValueError, match="compiled scalar-Jacobi"):
        _factor(bundle, census, raw)
    np.testing.assert_array_equal(raw.matrix_copy(), original)


def test_foreign_identical_state_and_changed_census_do_not_consume_raw():
    bundle, census, raw = _physical_bundle()
    original = raw.matrix_copy()
    foreign, foreign_census, _ = _physical_bundle()
    assert foreign.state.state_identity_sha256 == bundle.state.state_identity_sha256
    with pytest.raises(ValueError, match="provenance"):
        _factor(foreign, foreign_census, raw)
    config = census.config
    config.whitener_column_block = 1
    changed = core._make_periodic_correlation_factor_build_census(
        bundle.reference, bundle.schedule, config,
        [s.factor_build_q_record() for s in bundle.sources],
    )
    with pytest.raises(ValueError, match="provenance"):
        _factor(bundle, changed, raw)
    np.testing.assert_array_equal(raw.matrix_copy(), original)


def test_rank_failure_after_eigensolve_consumes_sealed_raw():
    bundle, census, raw = _physical_bundle(basis=_auxiliary_basis(exponent=1e-12))
    assert not np.any(raw.matrix_copy())
    with pytest.raises(RuntimeError, match="rank is zero"):
        _factor(bundle, census, raw)
    with pytest.raises((ValueError, RuntimeError), match="provenance"):
        _factor(bundle, census, raw)


def test_negative_nyquist_factorization_preserves_exact_real_gauge():
    bundle, census, raw = _physical_bundle(mesh=(2, 1, 1))
    assert raw.self_conjugate_transfer
    matrix = raw.matrix_copy()
    result = _factor(bundle, census, raw)
    assert result.q_index == result.conjugate_q_index == 1
    assert not np.any(result.matrix.imag)
    expected, _ = _oracle(matrix, 1e-10)
    np.testing.assert_allclose(result.matrix, expected, atol=2e-12)


def test_payload_wire_oracle_and_negative_tolerance_provenance():
    bundle, census, raw = _physical_bundle()
    result = _factor(bundle, census, raw)
    digest = hashlib.sha256()
    def u32(x):
        digest.update(struct.pack(">I", x))
    def u64(x):
        digest.update(struct.pack(">Q", x))
    def string(x):
        data = x.encode()
        u64(len(data))
        digest.update(data)
    def real(x):
        digest.update(struct.pack(">d", 0.0 if x == 0 else x))
    string("vibeqc.periodic.correlation.metric-factorization.payload")
    u32(1)
    string(result.input_payload_identity_sha256)
    u64(result.q_index)
    u64(result.conjugate_q_index)
    d = result.diagnostics
    for name in ("n_auxiliary", "retained_rank", "sweeps", "rotations"):
        u64(getattr(d, name))
    for name in (
        "rank_cutoff", "negative_tolerance", "minimum_eigenvalue", "maximum_eigenvalue",
        "smallest_retained_eigenvalue", "largest_discarded_eigenvalue", "input_scale",
        "scaled_initial_frobenius_norm", "scaled_final_offdiagonal_norm",
        "orthogonality_frobenius_error", "maximum_self_conjugate_imaginary_residual",
    ):
        real(getattr(d, name))
    u64(result.matrix.size)
    for z in result.matrix.flat:
        real(z.real)
        real(z.imag)
    assert digest.hexdigest() == result.payload_identity_sha256
    bundle, census, raw = _physical_bundle()
    other = _factor(bundle, census, raw, negative=2e-12)
    np.testing.assert_array_equal(other.matrix, result.matrix)
    assert other.payload_identity_sha256 != result.payload_identity_sha256
