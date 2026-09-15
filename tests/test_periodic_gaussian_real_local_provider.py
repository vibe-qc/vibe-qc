"""Actual Gaussian HF to real RI rows, with independent tiny contractions.

Finite onsite-image/reciprocal Hamiltonians only; these are not bulk targets
or a claim of infinite-source, auxiliary-basis or periodic DLPNO accuracy.
"""

from __future__ import annotations

import gc
from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_local_orbital_factors import _bundle, _dense_panel
from tests.test_periodic_gaussian_fock import _q_data
from tests.test_periodic_gaussian_rhf import _bundle as _hf_bundle, _run as _hf_run
from tests.test_periodic_correlation_real_local_provider import (
    _options, _expected_rows, _make as _legacy_make, _real_gauge,
)
from tests.test_periodic_correlation_real_local_basis import _make as _real_basis
from tests.test_periodic_correlation_density_factors import _virtual
from tests.test_periodic_correlation_local_factors import _native_store
from tests.test_periodic_correlation_reciprocal_metric import _CanonicalDigest


def _controls(b, *, auxiliary_block=1):
    config = core._PeriodicGaussianRealLocalProviderConfig()
    config.auxiliary_block, config.panel = auxiliary_block, b.panel_config
    live = core._PeriodicGaussianRealLocalProviderLiveInventory()
    live.fixed_backend_margin_bytes_per_worker = 65536
    caps = core._PeriodicGaussianRealLocalProviderCaps()
    resources = core._PeriodicGaussianMetricCaps()
    resources.maximum_owned_numeric_bytes = 2**23
    resources.maximum_per_replica_inventoried_bytes = 2**26
    resources.maximum_node_inventoried_bytes = 2**28
    resources.maximum_candidate_evaluations = 10**16
    resources.maximum_work_units = 10**18
    caps.resources, caps.metric, caps.panel = resources, b.config.metric_caps, b.panel_caps
    caps.maximum_factor_panels, caps.maximum_tile_calls = 64, 4096
    caps.maximum_image_candidate_evaluations = 10**18
    caps.maximum_progress_callbacks, caps.maximum_scalar_work_units = 512, 1000000
    return config, live, caps


def _plan(b, controls=None):
    config, live, caps = _controls(b) if controls is None else controls
    return core._plan_periodic_gaussian_real_local_provider(
        b.hf, b.reference, b.wannier, b.domain, b.space, b.basis, config, live, caps)


def _make(b, *, controls=None, gauge=None, ao=None, auxiliary=None, hf=None, progress=None, options=None):
    config, live, caps = _controls(b) if controls is None else controls
    return core._make_periodic_gaussian_real_local_provider(
        b.hf if hf is None else hf, b.reference, b.ao if ao is None else ao,
        b.auxiliary if auxiliary is None else auxiliary, b.wannier,
        b.gauge if gauge is None else gauge, b.domain, b.space, b.basis, config,
        _options() if options is None else options, live, caps, progress)


def _all_dense_panels(b):
    panels = []
    for q in range(b.context.n_kpoints):
        source, _, whitener = _q_data(b, q)
        panels.append(_dense_panel(b, source, whitener))
    return np.asarray(panels)


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 2, 2)])
def test_actual_hf_gaussian_rows_and_all_oriented_integrals(mesh):
    nk = int(np.prod(mesh))
    b = _bundle(mesh, rows=[[0, min(1, nk-1)]], translation=min(1, nk-1))
    events = []
    wrapped = _make(b, progress=events.append)
    result, plan, receipt = wrapped.provider, wrapped.memory, wrapped.receipt
    panels = _all_dense_panels(b)
    expected, _ = _expected_rows(panels, mesh)
    np.testing.assert_allclose(result.rows_copy(), expected, atol=3e-13, rtol=3e-12)
    n = plan.orbital_count
    for p, q, r, s in product(range(n), repeat=4):
        actual = result.integral(p, q, r, s)
        wanted = np.vdot(panels[:, :, q, p], panels[:, :, r, s])
        np.testing.assert_allclose(actual.value, wanted, atol=3e-12, rtol=3e-12)
        assert actual.roundoff_error_bound <= result.options.maximum_scalar_roundoff_error
    assert receipt.completed_source_count == receipt.completed_metric_count == receipt.completed_whitening_count == nk
    assert receipt.completed_factor_panels == plan.factor_panels
    assert receipt.completed_tile_calls == plan.tile_calls
    assert receipt.charged_work_units_upper_bound <= plan.work_units
    assert receipt.reciprocal_candidate_evaluations <= plan.reciprocal_candidate_evaluations_upper_bound
    assert receipt.image_candidate_evaluations <= plan.image_candidate_evaluations_upper_bound
    assert receipt.maximum_observed_owned_numerical_bytes <= plan.peak_owned_numerical_bytes
    assert receipt.maximum_observed_per_replica_inventoried_bytes <= plan.per_replica_inventoried_bytes
    assert receipt.progress_callback_count == len(events) == plan.progress_callback_upper_bound
    assert events[0].stage == core._PeriodicGaussianRealLocalProviderStage.Begin
    assert events[-1].stage == core._PeriodicGaussianRealLocalProviderStage.Complete
    assert result.memory.tile_visits == result.memory.maximum_reader_tile_bytes == 0
    assert result.memory.live_reader_numeric_bytes == result.memory.live_reader_control_bytes == 0
    assert result.diagnostics.tile_visits == 0
    assert result.identity_sha256 == wrapped.identity_sha256
    assert wrapped.source_context_identity_sha256 == b.context.source_context_identity_sha256
    assert result.source_context_identity_sha256 == wrapped.source_context_identity_sha256
    assert wrapped.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256
    assert result.hf_reference_source_identity_sha256 == wrapped.hf_reference_source_identity_sha256
    assert wrapped.matched_finite_gaussian_hf_recipe and result.matched_finite_gaussian_hf_recipe
    assert not wrapped.bitwise_hf_factor_consumption_verified
    assert not result.hf_hamiltonian_match_certified and not result.ao_image_source_certified
    assert not wrapped.infinite_source_accuracy_certified
    with pytest.raises(RuntimeError, match="no factor store"):
        _ = result.store_identity_sha256
    if nk == 3:
        assert np.max(np.abs(panels[1].imag)) > 1e-5
        assert np.linalg.norm(expected[2*b.auxiliary.nbasis:]) > 1e-5


@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1)])
def test_exact_variable_phase_inventory_and_no_owner_alias_double_charge(mesh):
    b = _bundle(mesh)
    p = _plan(b)
    controls = _controls(b)
    config, live, caps = controls
    k, a, n, m, o, h = p.n_cells, p.n_auxiliary, p.n_basis, p.orbital_count, p.occupied_count, p.density_count
    nonself = p.self_inverse_q_count != k
    assert p.retained_row_bytes == 8*k*a*h
    assert p.norm_workspace_bytes == 24*h
    assert p.whitener_bytes == 16*a*a
    assert p.maximum_live_whitener_bytes == (2 if nonself else 1)*16*a*a
    assert p.retained_partner_panel_bytes == (16*config.auxiliary_block*m*m+16*o if nonself else 0)
    assert p.panel_driver_owned_bytes == 32*config.auxiliary_block*m*m+32*n*m+32*n+16*o
    retained = p.retained_row_bytes+p.norm_workspace_bytes
    assert p.metric_phase_owned_upper_bound == retained+(p.whitener_bytes if nonself else 0)+caps.metric.maximum_owned_numeric_bytes
    assert p.panel_phase_owned_upper_bound == (retained+p.maximum_live_whitener_bytes
        +p.retained_partner_panel_bytes+p.panel_driver_owned_bytes+caps.panel.tile.resources.maximum_owned_numeric_bytes)
    assert p.peak_owned_numerical_bytes == max(p.metric_phase_owned_upper_bound, p.panel_phase_owned_upper_bound)
    assert p.basis_index_alias_bytes == b.basis.memory.retained_index_bytes
    known = (p.borrowed_basis_active_numeric_bytes+p.caller_gauge_bytes+p.live_wannier_bytes
        +p.live_domain_bytes+p.live_space_bytes+p.live_basis_bytes)
    assert p.per_replica_inventoried_bytes == (p.peak_owned_numerical_bytes+known
        +p.macro_fixed_object_bytes+p.maximum_leaf_fixed_object_bytes+live.fixed_backend_margin_bytes_per_worker)
    live.other_retained_bytes_per_worker, live.other_transient_bytes_per_worker = 123, 456
    updated = _plan(b, controls)
    assert updated.peak_owned_numerical_bytes == p.peak_owned_numerical_bytes
    assert updated.required_node_memory_bytes-p.required_node_memory_bytes == p.replicas_per_node*579
    assert updated.plan_identity_sha256 != p.plan_identity_sha256


_RESOURCE_CAPS = {
    "maximum_owned_numeric_bytes": "peak_owned_numerical_bytes",
    "maximum_per_replica_inventoried_bytes": "per_replica_inventoried_bytes",
    "maximum_node_inventoried_bytes": "required_node_memory_bytes",
    "maximum_work_units": "work_units",
    "maximum_candidate_evaluations": "reciprocal_candidate_evaluations_upper_bound",
}
_DIRECT_CAPS = {
    "maximum_factor_panels": "factor_panels", "maximum_tile_calls": "tile_calls",
    "maximum_image_candidate_evaluations": "image_candidate_evaluations_upper_bound",
    "maximum_progress_callbacks": "progress_callback_upper_bound", "maximum_scalar_work_units": "scalar_work_units",
}


@pytest.mark.parametrize("field,plan_field", list(_RESOURCE_CAPS.items())+list(_DIRECT_CAPS.items()))
def test_one_below_each_macro_cap_precedes_input_scan_and_callback(field, plan_field):
    b = _bundle((1, 1, 1))
    controls = _controls(b)
    p = _plan(b, controls)
    caps = controls[2]
    if field in _RESOURCE_CAPS:
        resources = caps.resources
        setattr(resources, field, getattr(p, plan_field)-1)
        caps.resources = resources
    else:
        setattr(caps, field, getattr(p, plan_field)-1)
    gauge = np.full_like(b.gauge, np.nan)
    events = []
    with pytest.raises((ValueError, OverflowError), match="cap|exceed"):
        _make(b, controls=controls, gauge=gauge, progress=events.append)
    assert not events


def test_exact_macro_caps_admit_and_auxiliary_block_change_preserves_rows():
    b = _bundle((3, 1, 1), rows=[[0, 1]], translation=1)
    controls = _controls(b)
    p, caps = _plan(b, controls), controls[2]
    resources = caps.resources
    for field, plan_field in _RESOURCE_CAPS.items():
        setattr(resources, field, getattr(p, plan_field))
    caps.resources = resources
    for field, plan_field in _DIRECT_CAPS.items():
        setattr(caps, field, getattr(p, plan_field))
    tight = _make(b, controls=controls)
    blocked = _make(b, controls=_controls(b, auxiliary_block=b.auxiliary.nbasis))
    np.testing.assert_array_equal(tight.provider.rows_copy(), blocked.provider.rows_copy())
    assert tight.provider.payload_sha256 == blocked.provider.payload_sha256
    assert tight.identity_sha256 != blocked.identity_sha256  # Different consumed panel partition.


def test_genuine_hf_owner_and_basis_content_are_not_caller_labels():
    b, other = _bundle((1, 1, 1)), _bundle((1, 1, 1))
    assert b.hf.reference_source_identity_sha256 == other.hf.reference_source_identity_sha256
    with pytest.raises(ValueError, match="identical.*owner"):
        _make(b, hf=other.hf)
    with pytest.raises(ValueError, match="basis|content|identity"):
        _make(b, ao=b.auxiliary)
    unconverged = _hf_run(_hf_bundle(iterations=1))
    assert not unconverged.converged
    with pytest.raises(ValueError, match="converged"):
        _make(b, hf=unconverged)


def test_progress_exception_or_input_mutation_cannot_return_a_provider():
    b = _bundle((1, 1, 1))
    def cancel(event):
        raise RuntimeError("intentional provider cancellation")
    with pytest.raises(RuntimeError, match="intentional provider cancellation"):
        _make(b, progress=cancel)
    gauge = b.gauge.copy()
    def mutate(event):
        gauge.flat[0] *= -1
    with pytest.raises(ValueError, match="gauge.*(changed|differs)"):
        _make(b, gauge=gauge, progress=mutate)


def test_borrowed_provider_python_view_pins_wrapper_and_numeric_owner():
    b = _bundle((1, 1, 1))
    wrapper = _make(b)
    provider = wrapper.provider
    value, identity = provider.integral(0, 0, 1, 1).value, provider.identity_sha256
    del wrapper, b
    gc.collect()
    assert provider.integral(0, 0, 1, 1).value == value
    assert provider.identity_sha256 == identity
    assert provider.matched_finite_gaussian_hf_recipe


def test_store_backed_identity_wire_and_flags_remain_v1(tmp_path):
    # Independent legacy SHA assembly pins the unchanged wire through the
    # shared converter extraction. Existing Decimal/orientation tests pin
    # its arithmetic, including subnormal scalar dot error bounds.
    b = _native_store(tmp_path, mesh=(3, 1, 1), cutoff=6.5, gauge=_real_gauge((3, 1, 1)))
    basis = _real_basis(b, [[0, 0]], _virtual())
    result = _legacy_make(b, basis)
    payload = _CanonicalDigest("vibeqc.periodic.correlation.real-local-provider.rows", 1)
    payload.u64(result.memory.row_count)
    payload.u64(result.memory.density_count)
    for value in result.rows_copy().ravel():
        payload.binary64(value)
    assert payload.finish() == result.payload_sha256
    identity = _CanonicalDigest("vibeqc.periodic.correlation.real-local-provider.identity", 1)
    for value in (b.reference.state.state_identity_sha256, b.reference.dimensions.allocation_identity,
                  result.local_basis_identity_sha256, result.basis_certificate_identity_sha256,
                  result.store_identity_sha256, result.consumed_panels_identity_sha256, result.payload_sha256):
        identity.string(value)
    identity.string("binary64;round-to-nearest;gradual-underflow;no-fast-math;correctly-rounded-basic-and-sqrt;outward-nextafter-v1")
    identity.string("finite-source-global-auxiliary-ri;hf-operator-match-and-image-tail-uncertified")
    for field in ("reversal_absolute_tolerance", "reversal_relative_tolerance", "conjugacy_absolute_tolerance",
                  "conjugacy_relative_tolerance", "self_q_absolute_tolerance", "self_q_relative_tolerance",
                  "maximum_eri_projection_error", "maximum_scalar_roundoff_error"):
        identity.binary64(getattr(result.options, field))
    identity.u64(result.diagnostics.factor_panels_built)
    identity.u64(result.diagnostics.tile_visits)
    for field in ("maximum_reversal_error", "maximum_conjugacy_error", "maximum_self_q_imaginary_magnitude",
                  "maximum_original_density_norm", "maximum_reversal_norm", "maximum_covariance_projection_norm",
                  "maximum_real_row_conversion_norm", "orientation_error_bound", "covariance_error_bound",
                  "conversion_error_bound", "maximum_eri_projection_error_bound"):
        identity.binary64(getattr(result.diagnostics, field))
    assert identity.finish() == result.identity_sha256
    assert not result.matched_finite_gaussian_hf_recipe
    assert not result.hf_hamiltonian_match_certified
    with pytest.raises(RuntimeError, match="no live Gaussian"):
        _ = result.source_context_identity_sha256
