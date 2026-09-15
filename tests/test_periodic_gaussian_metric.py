"""Bounded authenticated pre-SCF metric and principal root; tiny oracles."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import _basis, _system
from tests.test_periodic_auxiliary_fourier import _identity_fixture
from tests.test_periodic_gaussian_source_context import (
    _make as _context, _caps as _basis_caps, _options, _CAP_FIELDS as _BASIS_CAP_FIELDS,
)
from tests.test_periodic_gaussian_reciprocal_source import _source
from tests.test_periodic_correlation_reciprocal_metric import (
    _CanonicalDigest, _analytic_raw_metric, _two_s_metric_basis, _mixed_s_p_metric_basis,
    _leaf_composed_raw_metric, _raw_metric_bundle,
)


_CAP_FIELDS = ("maximum_owned_numeric_bytes", "maximum_per_replica_inventoried_bytes",
               "maximum_node_inventoried_bytes", "maximum_candidate_evaluations", "maximum_work_units")
_PLAN_COUNTS = (
    "n_auxiliary", "accepted_vector_count", "candidate_count", "reciprocal_panel_capacity",
    "auxiliary_validation_passes", "matrix_bytes", "reciprocal_panel_bytes", "double_fourier_panel_bytes",
    "metric_owned_numeric_peak_bytes", "factorization_owned_numeric_peak_bytes", "owned_numeric_peak_bytes",
    "returned_numeric_bytes", "borrowed_basis_active_numeric_bytes", "fixed_inventoried_object_bytes",
    "per_replica_inventoried_bytes", "node_inventoried_bytes", "metric_product_count", "fourier_radial_term_count",
    "harmonic_term_count_upper_bound", "jacobi_rotation_count_upper_bound", "whitener_term_count_upper_bound",
    "candidate_evaluations", "work_units_upper_bound",
)
_LIVE_FIELDS = (
    "replicas_per_node", "other_retained_bytes_per_replica", "other_transient_bytes_per_replica",
    "fixed_backend_margin_bytes_per_replica", "external_node_bytes",
)


def _live():
    live = core._PeriodicGaussianMetricLiveInventory()
    live.replicas_per_node = 1
    live.fixed_backend_margin_bytes_per_replica = 65536
    return live


def _caps():
    caps = core._PeriodicGaussianMetricCaps()
    for field in _CAP_FIELDS:
        setattr(caps, field, 10**12)
    return caps


def _bundle(*, q=1, mesh=(3, 1, 1), auxiliary=None, block=2, rank=1e-9):
    ao = _identity_fixture()
    auxiliary = _two_s_metric_basis() if auxiliary is None else auxiliary
    options = _options()
    options.metric_absolute_eigenvalue_threshold = rank
    context = _context(system=_system(2 * np.pi * np.diag([1.0, 0.5, 1 / 3])),
                       ao=ao, auxiliary=auxiliary, mesh=core._RegularKMesh(list(mesh)), options=options)
    source = _source(context, q)
    config = core._PeriodicGaussianMetricConfig()
    config.reciprocal_block = block
    config.whitener_column_block = auxiliary.nbasis
    config.basis_verification_caps = _basis_caps()
    return SimpleNamespace(ao=ao, auxiliary=auxiliary, context=context, source=source,
                           config=config, live=_live(), caps=_caps())


def _raw(b, *, ao=None, auxiliary=None, live=None, caps=None):
    return core._build_periodic_gaussian_reciprocal_metric(
        b.source, b.ao if ao is None else ao, b.auxiliary if auxiliary is None else auxiliary,
        b.config, b.live if live is None else live, b.caps if caps is None else caps,
    )


def _factor(raw, b, *, columns=None, live=None, caps=None):
    return core._factorize_periodic_gaussian_metric(raw,
        b.auxiliary.nbasis if columns is None else columns,
        b.live if live is None else live, b.caps if caps is None else caps)


def _plan(b, phase="RAW_METRIC", *, live=None, caps=None):
    return core._plan_periodic_gaussian_metric(b.source, b.config,
        b.live if live is None else live, b.caps if caps is None else caps,
        getattr(core._PeriodicGaussianMetricPhase, phase))


def _raw_wire(raw):
    w = _CanonicalDigest("vibeqc.periodic.gaussian-metric.payload", 1)
    for value in (raw.context.source_context_identity_sha256, raw.source_identity_sha256,
                  raw.conjugate_source_identity_sha256):
        w.string(value)
    m = raw.matrix
    for value in (raw.q_index, raw.conjugate_q_index, len(m), m.size, m.nbytes):
        w.u64(value)
    for value in m.ravel():
        w.complex128(value)
    return w.finish()


def _plan_wire(b, plan):
    w = _CanonicalDigest("vibeqc.periodic.gaussian-metric.plan", 1)
    for value in (b.context.source_context_identity_sha256, b.source.source_identity_sha256,
                  b.source.conjugate_source_identity_sha256):
        w.string(value)
    w.u64(b.source.q_index)
    w.u64(b.source.conjugate_q_index)
    w.u32(0 if plan.phase == core._PeriodicGaussianMetricPhase.RAW_METRIC else 1)
    for name in _PLAN_COUNTS:
        w.u64(getattr(plan, name))
    w.u64(plan.config.reciprocal_block)
    w.u64(plan.config.whitener_column_block)
    for name in _BASIS_CAP_FIELDS:
        w.u64(getattr(plan.config.basis_verification_caps, name))
    for name in _LIVE_FIELDS:
        w.u64(getattr(plan.live, name))
    for name in _CAP_FIELDS:
        w.u64(getattr(plan.caps, name))
    return w.finish()


def _white_wire(white):
    w = _CanonicalDigest("vibeqc.periodic.gaussian-whitener.payload", 1)
    w.string(white.input_payload_identity_sha256)
    w.u64(white.q_index)
    w.u64(white.conjugate_q_index)
    d = white.diagnostics
    for name in ("n_auxiliary", "retained_rank", "sweeps", "rotations"):
        w.u64(getattr(d, name))
    for name in ("rank_cutoff", "negative_tolerance", "minimum_eigenvalue", "maximum_eigenvalue",
                 "smallest_retained_eigenvalue", "largest_discarded_eigenvalue", "input_scale",
                 "scaled_initial_frobenius_norm", "scaled_final_offdiagonal_norm", "orthogonality_frobenius_error",
                 "maximum_self_conjugate_imaginary_residual"):
        w.binary64(getattr(d, name))
    w.u64(white.matrix.size)
    for value in white.matrix.ravel():
        w.complex128(value)
    return w.finish()


@pytest.mark.parametrize("mesh,q", [((1, 1, 1), 0), ((3, 1, 1), 0), ((3, 1, 1), 1), ((3, 1, 1), 2), ((2, 1, 1), 1)])
def test_physical_s_metrics_and_roots_match_independent_gaussian_spectral_oracles(mesh, q):
    b = _bundle(mesh=mesh, q=q)
    raw = _raw(b)
    matrix = raw.matrix
    np.testing.assert_allclose(matrix, _analytic_raw_metric(b.source, b.auxiliary), atol=2e-12, rtol=2e-12)
    assert raw.payload_identity_sha256 == _raw_wire(raw)
    assert raw.source_identity_sha256 == b.source.source_identity_sha256
    assert raw.context is b.context
    values, vectors = np.linalg.eigh(matrix)
    keep = values > b.context.options.metric_absolute_eigenvalue_threshold
    expected = (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].conj().T
    raw_id = raw.payload_identity_sha256
    white = _factor(raw, b)
    assert raw.consumed
    with pytest.raises(ValueError, match="consumed"):
        _ = raw.matrix
    assert white.input_payload_identity_sha256 == raw_id
    assert white.payload_identity_sha256 == _white_wire(white)
    np.testing.assert_allclose(white.matrix, expected, atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(white.matrix @ matrix @ white.matrix,
                               vectors[:, keep] @ vectors[:, keep].conj().T, atol=3e-11)
    assert white.diagnostics.retained_rank == np.count_nonzero(keep)
    if white.self_conjugate_transfer:
        assert np.all(white.matrix.imag == 0)
        assert np.all(matrix.imag == 0)


def test_mixed_sp_metric_has_native_fourier_normalization():
    b = _bundle(auxiliary=_mixed_s_p_metric_basis())
    raw = _raw(b)
    np.testing.assert_allclose(raw.matrix, _leaf_composed_raw_metric(b.source, b.auxiliary), atol=2e-12, rtol=2e-12)


def test_shared_numeric_kernels_remain_bit_identical_to_legacy_state_origin():
    b = _bundle(rank=1e-10)
    old = _raw_metric_bundle(mesh=(3, 1, 1), reciprocal_lattice=b.context.reciprocal_lattice,
                            basis=b.auxiliary, reciprocal_block=b.config.reciprocal_block)
    config = old.source_config
    config.backend_identity_sha256 = core._periodic_correlation_metric_factorization_backend_identity_sha256()
    config.whitener_column_block = b.auxiliary.nbasis
    census = core._make_periodic_correlation_factor_build_census(old.reference, old.schedule, config,
        [s.factor_build_q_record() for s in old.sources])
    legacy = core._build_periodic_correlation_reciprocal_metric(old.reference, old.schedule, census,
        old.sources[1], b.auxiliary)
    raw = _raw(b)
    np.testing.assert_array_equal(raw.matrix.view(np.uint64), legacy.matrix_copy().view(np.uint64))
    old_white = core._factorize_periodic_correlation_metric(old.reference, old.schedule, census, legacy, 1e-12)
    white = _factor(raw, b)
    np.testing.assert_array_equal(white.matrix.view(np.uint64), old_white.matrix.view(np.uint64))
    assert white.payload_identity_sha256 != old_white.payload_identity_sha256


def test_reciprocal_and_whitener_blocks_change_plan_not_numeric_payload():
    rows = []
    for block, columns in ((1, 1), (2, 2), (5, 1)):
        b = _bundle(block=block)
        raw = _raw(b)
        raw_array, raw_id, plan_id = raw.matrix, raw.payload_identity_sha256, raw.plan.plan_identity_sha256
        white = _factor(raw, b, columns=columns)
        rows.append((raw_array, raw_id, plan_id, white.matrix, white.payload_identity_sha256, white.plan.plan_identity_sha256))
    for row in rows[1:]:
        np.testing.assert_array_equal(row[0].view(np.uint64), rows[0][0].view(np.uint64))
        np.testing.assert_array_equal(row[3].view(np.uint64), rows[0][3].view(np.uint64))
        assert row[1] == rows[0][1] and row[4] == rows[0][4]
        assert row[2] != rows[0][2] and row[5] != rows[0][5]


@pytest.mark.parametrize("phase", ["RAW_METRIC", "PRINCIPAL_WHITENING"])
def test_exact_one_q_memory_work_and_node_replication_inventory(phase):
    b = _bundle()
    live = _live()
    live.replicas_per_node = 3
    live.other_retained_bytes_per_replica = 17
    live.other_transient_bytes_per_replica = 23
    live.external_node_bytes = 101
    p = _plan(b, phase, live=live)
    a, n, g = b.auxiliary.nbasis, b.source.accepted_vector_count, min(2, b.source.accepted_vector_count)
    assert p.matrix_bytes == 16 * a*a
    assert p.metric_owned_numeric_peak_bytes == 16*a*a + 40*g + 32*a*g
    assert p.factorization_owned_numeric_peak_bytes == 32*a*a + 8*a
    assert p.returned_numeric_bytes == 16*a*a
    assert p.per_replica_inventoried_bytes == (p.owned_numeric_peak_bytes + p.borrowed_basis_active_numeric_bytes
        + p.fixed_inventoried_object_bytes + 17 + 23 + 65536)
    assert p.node_inventoried_bytes == 101 + 3 * p.per_replica_inventoried_bytes
    assert p.metric_product_count == n*a*(a+1)//2
    assert p.harmonic_term_count_upper_bound == 28*n*a
    assert p.jacobi_rotation_count_upper_bound == 64*a*(a-1)//2
    assert p.whitener_term_count_upper_bound == a*a*(a+1)//2
    assert p.candidate_evaluations == (2*b.source.candidate_count if phase == "RAW_METRIC" else 0)
    assert p.plan_identity_sha256 == _plan_wire(b, p)
    assert p.auxiliary_validation_passes == (n + g - 1)//g + 1
    aux = b.context.inventory.auxiliary
    assert p.fourier_radial_term_count == n*aux.coefficient_count
    if phase == "RAW_METRIC":
        scan = aux.borrowed_active_numeric_bytes//8 + aux.shell_count + aux.contraction_count
        expected_work = (b.context.inventory.work_units_upper_bound
            + 128*p.auxiliary_validation_passes*scan + 128*p.fourier_radial_term_count
            + 512*p.harmonic_term_count_upper_bound + 64*p.metric_product_count
            + 64*n*a + 2048*b.source.candidate_count + 128*a*a + 4096)
    else:
        expected_work = 32768*a*a*a + 1024*a*a + 4096
    assert p.work_units_upper_bound == expected_work


def _exact_caps(p):
    caps = _caps()
    for name, value in zip(_CAP_FIELDS, (p.owned_numeric_peak_bytes, p.per_replica_inventoried_bytes,
            p.node_inventoried_bytes, max(1, p.candidate_evaluations), p.work_units_upper_bound)):
        setattr(caps, name, value)
    return caps


@pytest.mark.parametrize("field", _CAP_FIELDS)
def test_raw_exact_caps_and_cap_minus_one_precede_basis_mismatch(field):
    b = _bundle()
    caps = _exact_caps(_plan(b))
    raw = _raw(b, caps=caps)
    assert not raw.consumed
    setattr(caps, field, getattr(caps, field) - 1)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _raw(b, ao=_identity_fixture(first_coefficient=0.9), caps=caps)


@pytest.mark.parametrize("field", [name for name in _CAP_FIELDS if name != "maximum_candidate_evaluations"])
def test_whitening_readmits_current_caps_before_raw_consumption(field):
    b = _bundle()
    raw = _raw(b)
    original = raw.matrix
    caps = _exact_caps(_plan(b, "PRINCIPAL_WHITENING"))
    setattr(caps, field, getattr(caps, field) - 1)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _factor(raw, b, caps=caps)
    assert not raw.consumed
    np.testing.assert_array_equal(raw.matrix, original)
    white = _factor(raw, b, caps=_exact_caps(_plan(b, "PRINCIPAL_WHITENING")))
    assert raw.consumed and white.matrix.shape == original.shape


def test_changed_current_live_inventory_cannot_use_stale_raw_admission():
    b = _bundle()
    raw = _raw(b)
    caps = _exact_caps(_plan(b, "PRINCIPAL_WHITENING"))
    live = _live()
    live.other_retained_bytes_per_replica = 1
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _factor(raw, b, caps=caps, live=live)
    assert not raw.consumed


def test_both_actual_basis_roles_are_revalidated_before_metric_allocation():
    b = _bundle()
    with pytest.raises(ValueError, match="mismatch"):
        _raw(b, ao=_identity_fixture(first_coefficient=0.9))
    wrong = _basis([(0, (0, 0, 0), [0.7], [0.5], True), (0, (0.3, 0, 0), [0.8], [0.5], True)])
    with pytest.raises(ValueError, match="mismatch"):
        _raw(b, auxiliary=wrong)
    with pytest.raises(ValueError, match="finite"):
        _raw(b, ao=_identity_fixture(first_coefficient=np.nan))


def test_larger_actual_basis_is_census_capped_before_nonfinite_payload_scan():
    b = _bundle()
    larger = _basis([(0, (0, 0, 0), [0.5]*20, [np.nan]*20, True),
                     (1, (1.25, -0.5, 0.75), [0.8], [-0.25], True)])
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _raw(b, ao=larger)
    # Same aggregate metadata/work but one primitive lane-pair shifts roles.
    # This remains bounded and must fail on content, not an invented per-role cap.
    shifted_ao = _basis([(0, (0, 0, 0), [1.25], [0.75], False),
                         (1, (1.25, -0.5, 0.75), [0.8], [-0.25], True)])
    shifted_aux = _basis([(0, (0, 0, 0), [0.5, 0.8], [1.0, 0.1], True),
                          (0, (0.4, 0, 0), [1.0], [0.75], True)])
    with pytest.raises(ValueError, match="content mismatch"):
        _raw(b, ao=shifted_ao, auxiliary=shifted_aux)


def test_rank_deficiency_and_zero_rank_failure_have_explicit_consumption_semantics():
    duplicate = _basis([(0, (0, 0, 0), [0.7], [0.5], True)] * 2)
    b = _bundle(auxiliary=duplicate)
    raw = _raw(b)
    matrix = raw.matrix
    white = _factor(raw, b)
    assert white.diagnostics.retained_rank == 1
    np.testing.assert_allclose(white.matrix @ matrix @ white.matrix, np.full((2, 2), 0.5), atol=1e-12)
    b = _bundle(rank=1e100)
    raw = _raw(b)
    with pytest.raises(RuntimeError, match="rank is zero"):
        _factor(raw, b)
    assert raw.consumed
    with pytest.raises(ValueError, match="consumed"):
        _factor(raw, b)


@pytest.mark.parametrize("field", _CAP_FIELDS)
def test_all_admission_caps_are_explicit_positive(field):
    b = _bundle()
    caps = _caps()
    setattr(caps, field, 0)
    with pytest.raises(ValueError, match="positive"):
        _raw(b, caps=caps)


@pytest.mark.parametrize("field", ["replicas_per_node", "fixed_backend_margin_bytes_per_replica"])
def test_replica_count_and_unverified_backend_allowance_are_explicit(field):
    b = _bundle()
    live = _live()
    setattr(live, field, 0)
    with pytest.raises(ValueError, match="positive"):
        _raw(b, live=live)


def test_invalid_blocks_basis_cap_and_node_overflow_fail_without_consumption():
    for field in ("reciprocal_block", "whitener_column_block"):
        b = _bundle()
        setattr(b.config, field, 0)
        with pytest.raises(ValueError, match="blocks"):
            _raw(b)
    b = _bundle()
    basis_cap = _basis_caps()
    basis_cap.maximum_work_units = 1
    b.config.basis_verification_caps = basis_cap
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _raw(b)
    b = _bundle()
    live = _live()
    live.replicas_per_node = 2**64 - 1
    with pytest.raises(OverflowError):
        _raw(b, live=live)
    raw = _raw(b)
    with pytest.raises(ValueError, match="blocks"):
        _factor(raw, b, columns=b.auxiliary.nbasis + 1)
    assert not raw.consumed


def test_result_ownership_and_diagnostic_copies_cannot_mutate_sealed_payload():
    b = _bundle()
    raw = _raw(b)
    identity = raw.payload_identity_sha256
    m = raw.matrix
    m[:] = 100
    plan = raw.plan
    config = plan.config
    config.reciprocal_block = 100
    assert raw.payload_identity_sha256 == identity
    assert raw.plan.config.reciprocal_block == 2
    white = _factor(raw, b)
    context_id = white.context.source_context_identity_sha256
    del b, raw
    gc.collect()
    assert white.context.source_context_identity_sha256 == context_id
    assert not np.all(white.matrix == 100)
    assert not hasattr(white, "state")
