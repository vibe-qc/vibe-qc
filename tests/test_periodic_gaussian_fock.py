"""Streamed native G[D] against dense contractions of actual tiny Gaussian tiles."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import _basis, _system
from tests.test_periodic_gaussian_source_context import _make as _context, _options, _caps as _basis_caps
from tests.test_periodic_gaussian_source_context import _CAP_FIELDS as _BASIS_CAP_FIELDS
from tests.test_periodic_gaussian_reciprocal_source import _caps as _source_caps
from tests.test_periodic_gaussian_reciprocal_source import _CAP_FIELDS as _SOURCE_CAP_FIELDS
from tests.test_periodic_gaussian_metric import _live, _CAP_FIELDS
from tests.test_periodic_gaussian_metric import _LIVE_FIELDS
from tests.test_periodic_correlation_reciprocal_metric import _two_s_metric_basis, _CanonicalDigest

_PLAN_FIELDS = (
    "n_kpoints",
    "n_basis",
    "n_auxiliary",
    "auxiliary_block",
    "ao_column_block",
    "auxiliary_block_count",
    "ao_column_block_count",
    "density_element_count",
    "borrowed_density_bytes",
    "response_bytes",
    "response_compensation_bytes",
    "hartree_vector_bytes",
    "exchange_double_panel_bytes",
    "resident_whitener_bytes",
    "metric_phase_owned_numeric_upper_bound",
    "tile_phase_owned_numeric_upper_bound",
    "owned_numeric_upper_bound",
    "borrowed_basis_active_numeric_bytes",
    "macro_fixed_object_bytes",
    "maximum_leaf_fixed_inventory_bytes",
    "per_replica_inventoried_bytes",
    "node_inventoried_bytes",
    "hartree_tile_calls",
    "exchange_tile_calls",
    "total_tile_calls",
    "progress_callback_upper_bound",
    "driver_contraction_terms",
    "reciprocal_candidate_evaluations_upper_bound",
    "image_candidate_evaluations_upper_bound",
    "driver_work_units_upper_bound",
    "work_units_upper_bound",
)
_DIAG_FIELDS = (
    "maximum_magnitude", "maximum_hermiticity_residual", "relative_hermiticity_residual",
    "maximum_time_reversal_residual", "relative_time_reversal_residual",
    "maximum_diagonal_imaginary",
)
_FOCK_CAP_FIELDS = ("maximum_tile_calls", "maximum_progress_callbacks", "maximum_image_candidate_evaluations")


def _leaf_caps():
    c = core._PeriodicGaussianMetricCaps()
    for field, value in zip(_CAP_FIELDS, (131072, 2**24, 2**24, 100000, 10**11)):
        setattr(c, field, value)
    return c


def _caps():
    c = core._PeriodicGaussianFockCaps()
    r = core._PeriodicGaussianMetricCaps()
    for field, value in zip(_CAP_FIELDS, (2**23, 2**25, 2**27, 10**10, 10**16)):
        setattr(r, field, value)
    c.resources = r
    c.maximum_tile_calls = 4096
    c.maximum_progress_callbacks = 4096
    c.maximum_image_candidate_evaluations = 10**13
    return c


def _density(nk, n, *, seed=197):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(nk, n, n)) + 1j*rng.normal(size=(nk, n, n))
    d = np.empty_like(x)
    for k in range(nk):
        d[k] = 0.2*(x[k]+x[k].conj().T)
        d[k].flat[::n+1] = np.arange(1, n+1)*0.3
    return np.ascontiguousarray(d)


def _bundle(*, mesh=(2, 1, 1), aux_block=2, columns=2, reciprocal_block=2,
            skew=False, ragged=False):
    ao_shells = [(0, (0.1, -0.2, 0.15), [0.55], [0.7], True),
                 (0, (0.6, 0.3, -0.25), [0.7], [0.8], True)]
    if ragged:
        ao_shells.append((0, (-0.3, 0.2, 0.4), [0.8], [0.65], True))
    ao = _basis(ao_shells)
    auxiliary = (_basis([(0, (0.0, 0.0, 0.0), [0.5], [1.0], True),
                         (0, (0.4, 0.0, 0.0), [1.0], [0.75], True),
                         (0, (0.2, 0.3, 0.0), [0.65], [0.8], True)])
                 if ragged else _two_s_metric_basis())
    lattice = (np.array([[6.0, 0.4, 0.1], [0.0, 4.7, 0.2], [0.0, 0.0, 4.1]]) if skew
               else 2*np.pi*np.diag([1, 0.5, 1/3]))
    options = _options()
    options.ao_pair_image_cutoff_bohr = 3.0
    context = _context(system=_system(lattice), ao=ao, auxiliary=auxiliary,
                       mesh=core._RegularKMesh(list(mesh)), options=options)
    config = core._PeriodicGaussianFockConfig()
    config.auxiliary_block, config.ao_column_block = aux_block, columns
    sc = _source_caps()
    sc.maximum_fixed_storage_bytes = 65536
    sc.maximum_candidates_per_source = 4096
    sc.maximum_candidate_evaluations = 25000
    config.source_caps = sc
    mc = core._PeriodicGaussianMetricConfig()
    mc.reciprocal_block, mc.whitener_column_block = reciprocal_block, auxiliary.nbasis
    mc.basis_verification_caps = _basis_caps()
    config.metric, config.metric_caps = mc, _leaf_caps()
    tc = core._PeriodicGaussianThreeCenterConfig()
    tc.reciprocal_block = reciprocal_block
    tc.basis_verification_caps = _basis_caps()
    config.tile = tc
    tcap = core._PeriodicGaussianThreeCenterCaps()
    tcap.maximum_image_candidates = 20000
    tcap.resources = _leaf_caps()
    config.tile_caps = tcap
    return SimpleNamespace(ao=ao, auxiliary=auxiliary, context=context, config=config,
                           live=_live(), caps=_caps(), density=_density(context.n_kpoints, ao.nbasis))


def _plan(b, *, caps=None, live=None):
    return core._plan_periodic_gaussian_fock(b.context, b.config,
        b.live if live is None else live, b.caps if caps is None else caps)


def _call(b, *, density=None, progress=None, caps=None, live=None, ao=None):
    return core._build_periodic_gaussian_fock(b.context, b.ao if ao is None else ao, b.auxiliary,
        b.density if density is None else density, b.config, b.live if live is None else live,
        b.caps if caps is None else caps, progress)


def _q_data(b, q):
    source = core._make_periodic_gaussian_reciprocal_source(b.context, q, b.config.source_caps)
    raw = core._build_periodic_gaussian_reciprocal_metric(source, b.ao, b.auxiliary,
        b.config.metric, _live(), b.config.metric_caps)
    raw_id = raw.payload_identity_sha256
    w = core._factorize_periodic_gaussian_metric(raw, b.auxiliary.nbasis, _live(), b.config.metric_caps)
    return source, raw_id, w


def _physical_tile(b, source, w, bra, mu=0, nu=0, columns=None, aux=0, rows=None):
    s = core._PeriodicGaussianThreeCenterSelection()
    s.k_bra_index = bra
    s.ao_pair_begin = mu*b.ao.nbasis+nu
    s.ao_pair_count = b.ao.nbasis**2 if columns is None else columns
    s.auxiliary_begin = aux
    s.auxiliary_count = b.auxiliary.nbasis if rows is None else rows
    return core._build_periodic_gaussian_three_center_tile(source, w, b.ao, b.auxiliary,
        s, b.config.tile, _live(), b.config.tile_caps)


def _dense_oracle(b, density=None):
    d = b.density if density is None else density
    nk, n, _ = d.shape
    j, exchange = np.zeros_like(d), np.zeros_like(d)
    for q in range(nk):
        source, _, w = _q_data(b, q)
        matrices = [_physical_tile(b, source, w, bra).matrix.reshape(-1, n, n) for bra in range(nk)]
        if q == 0:
            z = sum(np.einsum("pij,ji->p", m, d[k]) for k, m in enumerate(matrices))/nk
            for k, m in enumerate(matrices):
                j[k] = np.einsum("p,pji->ij", z, m.conj())
        for bra, m in enumerate(matrices):
            ket = b.context.ket_index(bra, q)
            for mp in m:
                exchange[ket] += mp.conj().T @ d[bra] @ mp / nk
    return j-0.5*exchange


def _density_wire(context, d):
    w = _CanonicalDigest("vibeqc.periodic.gaussian-fock.density", 1)
    w.string(context.source_context_identity_sha256)
    for value in (len(d), d.shape[1], d.size, d.nbytes):
        w.u64(value)
    assert w.wire_bytes == 116+len("vibeqc.periodic.gaussian-fock.density")
    for value in d.ravel():
        w.complex128(value)
    return w.finish()


def _payload_wire(r):
    w = _CanonicalDigest("vibeqc.periodic.gaussian-fock.payload", 1)
    for value in (r.context.source_context_identity_sha256, r.density_identity_sha256,
                  r.consumed_factor_identity_sha256):
        w.string(value)
    for value in (r.plan.n_kpoints, r.plan.n_basis, r.matrix.size, r.matrix.nbytes):
        w.u64(value)
    for d in (r.density_diagnostics, r.response_diagnostics):
        for field in _DIAG_FIELDS:
            w.binary64(getattr(d, field))
    assert w.wire_bytes == 356+len("vibeqc.periodic.gaussian-fock.payload")
    for value in r.matrix.ravel():
        w.complex128(value)
    return w.finish()


def _plan_wire(context, p):
    w = _CanonicalDigest("vibeqc.periodic.gaussian-fock.plan", 1)
    w.string(context.source_context_identity_sha256)
    for field in _PLAN_FIELDS:
        w.u64(getattr(p, field))
    config = p.config
    w.u64(config.auxiliary_block)
    w.u64(config.ao_column_block)
    for field in _SOURCE_CAP_FIELDS:
        w.u64(getattr(config.source_caps, field))
    w.u64(config.metric.reciprocal_block)
    w.u64(config.metric.whitener_column_block)
    for field in _BASIS_CAP_FIELDS:
        w.u64(getattr(config.metric.basis_verification_caps, field))
    for field in _CAP_FIELDS:
        w.u64(getattr(config.metric_caps, field))
    w.u64(config.tile.reciprocal_block)
    for field in _BASIS_CAP_FIELDS:
        w.u64(getattr(config.tile.basis_verification_caps, field))
    w.u64(config.tile_caps.maximum_image_candidates)
    for field in _CAP_FIELDS:
        w.u64(getattr(config.tile_caps.resources, field))
    for field in _LIVE_FIELDS:
        w.u64(getattr(p.live, field))
    for field in _CAP_FIELDS:
        w.u64(getattr(p.caps.resources, field))
    for field in _FOCK_CAP_FIELDS:
        w.u64(getattr(p.caps, field))
    return w.finish()


def _consumed_wire_oracle(b):
    plan = _plan(b)
    h = _CanonicalDigest("vibeqc.periodic.gaussian-fock.consumed-factors", 1)
    h.string(b.context.source_context_identity_sha256)
    h.string(_density_wire(b.context, b.density))
    nk, n, a = b.context.n_kpoints, b.ao.nbasis, b.auxiliary.nbasis
    for value in (nk, n, a, plan.total_tile_calls):
        h.u64(value)
    assert h.wire_bytes == 188+len("vibeqc.periodic.gaussian-fock.consumed-factors")
    sequence = 0
    ab, block = b.config.auxiliary_block, b.config.ao_column_block
    for q in range(nk):
        source, raw_id, w = _q_data(b, q)
        h.u64(q)
        for value in (source.source_identity_sha256, source.conjugate_source_identity_sha256,
                      raw_id, w.payload_identity_sha256):
            h.string(value)
        def emit(bra, aux, rows, mu, nu, columns):
            nonlocal sequence
            tile = _physical_tile(b, source, w, bra, mu, nu, columns, aux, rows)
            h.u64(sequence)
            h.string(tile.payload_identity_sha256)
            sequence += 1
        if q == 0:
            for aux in range(0, a, ab):
                rows = min(ab, a-aux)
                for _ in range(2):
                    for bra in range(nk):
                        for mu in range(n):
                            for nu in range(0, n, block):
                                emit(bra, aux, rows, mu, nu, min(block, n-nu))
        for bra in range(nk):
            for aux in range(0, a, ab):
                rows = min(ab, a-aux)
                for sigma in range(0, n, block):
                    for mu in range(n):
                        emit(bra, aux, rows, mu, sigma, min(block, n-sigma))
                    for nu in range(0, n, block):
                        for mu in range(n):
                            emit(bra, aux, rows, mu, nu, min(block, n-nu))
    assert sequence == plan.total_tile_calls
    assert h.wire_bytes == (188+len("vibeqc.periodic.gaussian-fock.consumed-factors")
                           +296*nk+80*sequence)
    return h.finish()


@pytest.mark.parametrize("mesh,skew", [((1, 1, 1), False), ((2, 1, 1), False),
    ((3, 1, 1), False), ((2, 2, 1), False), ((2, 1, 1), True)])
def test_complex_density_response_matches_independent_dense_physical_factor_contraction(mesh, skew):
    b = _bundle(mesh=mesh, skew=skew)
    r = _call(b)
    expected = _dense_oracle(b)
    np.testing.assert_allclose(r.matrix, expected, atol=3e-12, rtol=3e-11)
    assert r.density_identity_sha256 == _density_wire(b.context, b.density)
    assert r.payload_identity_sha256 == _payload_wire(r)
    assert r.receipt.completed_q_count == b.context.n_kpoints
    assert r.receipt.completed_tile_count == r.plan.total_tile_calls
    assert r.receipt.progress_callback_count == 0
    assert r.receipt.charged_work_units_upper_bound <= r.plan.work_units_upper_bound
    assert r.receipt.reciprocal_candidate_evaluations <= r.plan.reciprocal_candidate_evaluations_upper_bound
    assert r.receipt.image_candidate_evaluations <= r.plan.image_candidate_evaluations_upper_bound
    assert r.receipt.maximum_observed_owned_numeric_bytes <= r.plan.owned_numeric_upper_bound
    assert r.receipt.maximum_observed_per_replica_inventoried_bytes <= r.plan.per_replica_inventoried_bytes
    assert r.density_diagnostics.maximum_hermiticity_residual == 0
    assert r.response_diagnostics.maximum_hermiticity_residual == pytest.approx(
        np.max(np.abs(r.matrix-r.matrix.conj().transpose(0, 2, 1))), abs=1e-30)
    addressing = core._RegularKMesh(list(mesh))
    opposite = [addressing.negate_index(k) for k in range(len(expected))]
    assert r.response_diagnostics.maximum_time_reversal_residual == pytest.approx(
        np.max(np.abs(r.matrix-r.matrix[opposite].conj())), rel=1e-13, abs=1e-30)
    if len(expected)>1:
        assert np.max(np.abs(r.matrix-len(expected)*expected)) > 1e-5


def test_hermitian_density_differences_are_linear_and_no_tr_projection_occurs():
    b = _bundle()
    d, e = b.density, _density(2, 2, seed=251)
    first, second = _call(b, density=d), _call(b, density=e)
    combination = _call(b, density=np.ascontiguousarray(0.7*d-0.3*e))
    np.testing.assert_allclose(combination.matrix, 0.7*first.matrix-0.3*second.matrix, atol=3e-12, rtol=3e-11)
    assert first.density_diagnostics.maximum_time_reversal_residual > 1e-3
    assert np.max(np.abs(first.matrix.imag)) > 1e-5


def test_column_auxiliary_and_reciprocal_blocks_preserve_operator():
    rows = []
    for columns, aux, reciprocal in ((2, 2, 2), (1, 1, 1), (1, 2, 5)):
        rows.append(_call(_bundle(columns=columns, aux_block=aux, reciprocal_block=reciprocal)))
    for r in rows[1:]:
        np.testing.assert_array_equal(r.matrix.view(np.uint64), rows[0].matrix.view(np.uint64))
        assert r.plan.plan_identity_sha256 != rows[0].plan.plan_identity_sha256
        assert r.consumed_factor_identity_sha256 != rows[0].consumed_factor_identity_sha256


def test_partial_auxiliary_and_ao_column_blocks_match_dense_operator():
    b = _bundle(mesh=(1, 1, 1), ragged=True)
    r = _call(b)
    assert r.plan.n_basis == r.plan.n_auxiliary == 3
    assert r.plan.auxiliary_block_count == r.plan.ao_column_block_count == 2
    np.testing.assert_allclose(r.matrix, _dense_oracle(b), atol=3e-12, rtol=3e-11)
    assert r.receipt.completed_tile_count == r.plan.total_tile_calls


def test_zero_density_has_exact_zero_response_without_skipping_physical_factors():
    b = _bundle(mesh=(1, 1, 1))
    r = _call(b, density=np.zeros_like(b.density))
    np.testing.assert_array_equal(r.matrix, np.zeros_like(b.density))
    assert r.response_diagnostics.maximum_magnitude == 0
    assert r.receipt.completed_tile_count == r.plan.total_tile_calls > 0


def test_explicit_phase_inventory_call_counts_and_wrapper_copy_are_exact():
    b = _bundle(columns=1, aux_block=1)
    p = _plan(b)
    nk, n, a, ab, columns = p.n_kpoints, p.n_basis, p.n_auxiliary, p.auxiliary_block, p.ao_column_block
    u, l = (a+ab-1)//ab, (n+columns-1)//columns
    assert p.borrowed_density_bytes == p.response_bytes == p.response_compensation_bytes == 16*nk*n*n
    assert p.hartree_vector_bytes == 32*ab
    assert p.exchange_double_panel_bytes == 32*ab*n*columns
    assert p.resident_whitener_bytes == 16*a*a
    assert p.metric_phase_owned_numeric_upper_bound == 32*nk*n*n+b.config.metric_caps.maximum_owned_numeric_bytes
    assert p.tile_phase_owned_numeric_upper_bound == (32*nk*n*n+16*a*a
        +max(32*ab, 32*ab*n*columns)+b.config.tile_caps.resources.maximum_owned_numeric_bytes)
    assert p.hartree_tile_calls == 2*u*nk*n*l
    assert p.exchange_tile_calls == nk*nk*u*l*n*(1+l)
    assert p.progress_callback_upper_bound == 2+4*nk+2*u*nk+nk*nk*u*l
    assert p.driver_contraction_terms == 2*nk*a*n*n+2*nk*nk*a*n*n*n
    assert p.live.other_retained_bytes_per_replica == b.live.other_retained_bytes_per_replica+p.borrowed_density_bytes
    assert p.per_replica_inventoried_bytes == (p.owned_numeric_upper_bound+p.borrowed_density_bytes
        +p.borrowed_basis_active_numeric_bytes+p.macro_fixed_object_bytes+p.maximum_leaf_fixed_inventory_bytes
        +p.live.other_retained_bytes_per_replica+p.live.other_transient_bytes_per_replica
        +p.live.fixed_backend_margin_bytes_per_replica)
    assert p.plan_identity_sha256 == _plan_wire(b.context, p)


def test_consumption_trace_authenticates_every_actual_q_metric_whitener_and_tile():
    b = _bundle(mesh=(1, 1, 1), aux_block=1)
    r = _call(b)
    assert r.consumed_factor_identity_sha256 == _consumed_wire_oracle(b)
    assert r.plan.plan_identity_sha256 == _plan_wire(b.context, r.plan)


def _exact_caps(p):
    c = _caps()
    r = c.resources
    for field, value in zip(_CAP_FIELDS, (p.owned_numeric_upper_bound, p.per_replica_inventoried_bytes,
        p.node_inventoried_bytes, p.reciprocal_candidate_evaluations_upper_bound, p.work_units_upper_bound)):
        setattr(r, field, value)
    c.resources = r
    c.maximum_tile_calls, c.maximum_progress_callbacks = p.total_tile_calls, p.progress_callback_upper_bound
    c.maximum_image_candidate_evaluations = p.image_candidate_evaluations_upper_bound
    return c


@pytest.mark.parametrize("field", (*_CAP_FIELDS, *_FOCK_CAP_FIELDS))
def test_macro_cap_minus_one_rejects_before_density_copy_or_progress(field):
    b = _bundle()
    p = _plan(b)
    caps = _exact_caps(p)
    if field in _CAP_FIELDS:
        r = caps.resources
        setattr(r, field, getattr(r, field)-1)
        caps.resources = r
    else:
        setattr(caps, field, getattr(caps, field)-1)
    bad = b.density.copy()
    bad[0, 0, 0] = np.nan
    events = []
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _call(b, density=bad, caps=caps, progress=events.append)
    assert not events


def test_exact_caps_admit_and_current_node_replication_is_rechecked():
    b = _bundle()
    p = _plan(b)
    r = _call(b, caps=_exact_caps(p))
    assert r.receipt.completed_q_count == p.n_kpoints
    live = _live()
    live.replicas_per_node = 2
    with pytest.raises((ValueError, RuntimeError), match="node"):
        _call(b, caps=_exact_caps(p), live=live)


def test_progress_is_scalar_cancelable_and_controls_and_density_are_snapshotted():
    b = _bundle(mesh=(1, 1, 1))
    expected = _dense_oracle(b)
    original = b.density.copy()
    events = []
    def callback(e):
        events.append(e)
        if e.stage == core._PeriodicGaussianFockStage.BEGIN:
            b.density[:] = 0
            b.config.ao_column_block = 1
    r = _call(b, progress=callback)
    np.testing.assert_allclose(r.matrix, expected, atol=3e-12, rtol=3e-11)
    assert r.density_identity_sha256 == _density_wire(b.context, original)
    assert r.plan.config.ao_column_block == 2
    assert r.receipt.progress_callback_count == r.plan.progress_callback_upper_bound == len(events)
    assert events[-1].stage == core._PeriodicGaussianFockStage.COMPLETE
    assert events[-1].completed_tile_count == r.plan.total_tile_calls
    def cancel(_):
        raise RuntimeError("cancel-fock")
    with pytest.raises(RuntimeError, match="cancel-fock"):
        _call(b, progress=cancel)
    def cancel_after_tiles(e):
        if e.stage == core._PeriodicGaussianFockStage.QCOMPLETE:
            assert e.completed_tile_count > 0
            raise RuntimeError("cancel-fock-after-tiles")
    with pytest.raises(RuntimeError, match="cancel-fock-after-tiles"):
        _call(b, progress=cancel_after_tiles)


def test_native_density_mutation_guard_aborts_before_source_work():
    b = _bundle(mesh=(1, 1, 1))
    with pytest.raises(ValueError, match="density changed"):
        core._periodic_gaussian_fock_density_mutation_diagnostic(
            b.context, b.ao, b.auxiliary, b.density, b.config, b.live, b.caps)


@pytest.mark.parametrize("kind", ["nonhermitian", "nonfinite", "shape", "dtype", "strided"])
def test_invalid_density_rejected_without_silent_repair(kind):
    b = _bundle()
    d = b.density.copy()
    if kind == "nonhermitian":
        d[0, 0, 1] += 1e-15
    elif kind == "nonfinite":
        d[0, 0, 0] = np.inf
    elif kind == "shape":
        d = d[0]
    elif kind == "dtype":
        d = d.real.copy()
    else:
        d = d[:, :, ::-1]
    with pytest.raises((ValueError, OverflowError), match="Hermitian|nonfinite|contiguous"):
        _call(b, density=d)


def test_actual_basis_provenance_and_result_ownership_are_not_caller_labels():
    b = _bundle(mesh=(1, 1, 1))
    wrong = _basis([(0, (0.1, -0.2, 0.15), [0.55], [0.8], True),
                    (0, (0.6, 0.3, -0.25), [0.7], [0.8], True)])
    with pytest.raises(ValueError, match="content mismatch"):
        _call(b, ao=wrong)
    with pytest.raises(TypeError):
        core._PeriodicGaussianFockResult()
    r = _call(b)
    identity = r.payload_identity_sha256
    copy = r.matrix
    copy[:] = 100
    del b
    gc.collect()
    assert identity == r.payload_identity_sha256 == _payload_wire(r)
    assert not np.all(r.matrix == 100)
    assert not hasattr(r, "state") and not hasattr(r, "calculation_id")


@pytest.mark.parametrize("field", ["auxiliary_block", "ao_column_block"])
def test_zero_or_out_of_range_blocks_do_not_fall_back(field):
    b = _bundle()
    for value in (0, 3):
        setattr(b.config, field, value)
        with pytest.raises(ValueError, match="blocks"):
            _call(b)


@pytest.mark.parametrize("field", _CAP_FIELDS)
def test_zero_cumulative_resource_caps_are_rejected(field):
    b = _bundle()
    resources = b.caps.resources
    setattr(resources, field, 0)
    b.caps.resources = resources
    with pytest.raises(ValueError, match="positive"):
        _call(b)
