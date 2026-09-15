"""Bounded rectangular actual-source factors, not a scalar replay provider."""

from __future__ import annotations

import gc

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_selected_local_ccsd_t import _bundle, _prepare_leaves
from tests.test_periodic_gaussian_local_orbital_factors import (
    _bundle as _small_bundle, _build as _full, _plan as _full_plan,
    _dense_panel, _CAP_TO_PLAN, _PLAN_FIELDS, _LIVE_FIELDS, _CAP_FIELDS, _BASIS_CAP_FIELDS,
    _payload_wire as _full_payload_wire, _plan_wire as _full_plan_wire,
)
from tests.test_periodic_gaussian_fock import _q_data
from tests.test_periodic_correlation_reciprocal_metric import _CanonicalDigest


def _case(nk=3):
    b = _bundle((nk, 1, 1))
    _prepare_leaves(b)
    b.panel_live = core._PeriodicGaussianLocalOrbitalFactorLiveInventory()
    b.panel_live.fixed_backend_margin_bytes_per_worker = 65536
    return b


def _selection(left, right):
    s = core._PeriodicGaussianLocalOrbitalFactorSelection()
    s.left_begin, s.left_count = left
    s.right_begin, s.right_count = right
    return s


def _plan(b, source, whitener, selection, *, begin=0, count=None, caps=None, **changes):
    args = dict(hf=b.hf, reference=b.reference, source=source, whitener=whitener,
        wannier=b.wannier, domain=b.domain, space=b.space, basis=b.basis,
        auxiliary_begin=begin, auxiliary_count=b.auxiliary.nbasis-begin if count is None else count,
        selection=selection, config=b.panel_config, live=b.panel_live,
        caps=b.panel_caps if caps is None else caps)
    args.update(changes)
    return core._plan_periodic_gaussian_local_orbital_factor_block(**args)


def _build(b, source, whitener, selection, *, begin=0, count=None, caps=None, **changes):
    args = dict(hf=b.hf, reference=b.reference, source=source, whitener=whitener,
        ao_basis=b.ao, auxiliary_basis=b.auxiliary, wannier=b.wannier, gauges=b.gauge,
        domain=b.domain, space=b.space, basis=b.basis,
        auxiliary_begin=begin, auxiliary_count=b.auxiliary.nbasis-begin if count is None else count,
        selection=selection, config=b.panel_config, live=b.panel_live,
        caps=b.panel_caps if caps is None else caps)
    args.update(changes)
    return core._build_periodic_gaussian_local_orbital_factor_block(**args)


def _range_wire(h, selection):
    h.string("rectangular-common-orbital-ranges-v1")
    for field in ("left_begin", "left_count", "right_begin", "right_count"):
        h.u64(getattr(selection, field))


def _payload_wire(result):
    h = _CanonicalDigest("vibeqc.periodic.gaussian-local-orbital-factors.payload", 1)
    for x in (result.q_index, result.auxiliary_begin, result.auxiliary_count, result.orbital_count):
        h.u64(x)
    _range_wire(h, result.selection)
    for x in result.tensor_copy().ravel():
        h.complex128(x)
    return h.finish()


def _plan_wire(b, source, whitener, p):
    h = _CanonicalDigest("vibeqc.periodic.gaussian-local-orbital-factors.plan", 1)
    for x in (b.hf.reference_source_identity_sha256, source.source_identity_sha256,
        source.conjugate_source_identity_sha256, whitener.payload_identity_sha256,
        b.basis.identity_sha256, b.reference.dimensions.allocation_identity,
        b.wannier.wannier_identity_sha256, b.domain.pao_domain_identity_sha256,
        b.space.pao_space_identity_sha256):
        h.string(x)
    for field in _PLAN_FIELDS:
        h.u64(getattr(p, field))
    h.u64(p.config.ao_pair_block)
    for field in _LIVE_FIELDS:
        h.u64(getattr(p.live, field))
    for caps in (p.caps.resources, p.caps.tile.resources):
        for field in _CAP_FIELDS:
            h.u64(getattr(caps, field))
    for x in (p.caps.tile.maximum_image_candidates, p.caps.maximum_tile_calls,
              p.caps.maximum_image_candidate_evaluations, p.tile_config.reciprocal_block):
        h.u64(x)
    for field in _BASIS_CAP_FIELDS:
        h.u64(getattr(p.tile_config.basis_verification_caps, field))
    _range_wire(h, p.selection)
    return h.finish()


@pytest.mark.parametrize("nk,q", [(1,0), (2,0), (2,1), (3,0), (3,1), (3,2)])
def test_rectangular_assembly_is_bitwise_full_and_matches_independent_gaussian_transform(nk, q):
    b = _case(nk)
    source, _, w = _q_data(b, q)
    full = _full(b, source, w)
    reference, expected = full.tensor_copy(), _dense_panel(b, source, w)
    m = full.orbital_count
    output = np.empty_like(reference)
    # Irregular partitions cross occupied/virtual boundaries for K=2,3.
    left_blocks = [(0,1), (1,m-1)]
    right_blocks = [(0,m-1), (m-1,1)]
    for left in left_blocks:
        for right in right_blocks:
            s = _selection(left, right)
            block = _build(b, source, w, s)
            data = block.tensor_copy()
            l, L, r, R = s.left_begin, s.left_count, s.right_begin, s.right_count
            assert data.shape == (b.auxiliary.nbasis, L, R)
            np.testing.assert_array_equal(data, reference[:, l:l+L, r:r+R])
            np.testing.assert_allclose(data, expected[:, l:l+L, r:r+R], atol=4e-12, rtol=3e-11)
            output[:, l:l+L, r:r+R] = data
            assert block.orbital_count == m
            assert block.payload_sha256 == _payload_wire(block)
            assert block.memory.plan_identity_sha256 == _plan_wire(b, source, w, block.memory)
            assert block.consumed_tiles_identity_sha256 == full.consumed_tiles_identity_sha256
            assert block.basis_certificate_identity_sha256 == full.basis_certificate_identity_sha256
            assert block.source_identity_sha256 == full.source_identity_sha256
            assert block.whitener_payload_identity_sha256 == full.whitener_payload_identity_sha256
            assert block.element(0, L-1, R-1) == data[0, L-1, R-1]
            with pytest.raises(IndexError):
                block.element(0, L, 0)
    np.testing.assert_array_equal(output, reference)


def test_full_range_wrapper_preserves_original_wires_and_numerical_inventory():
    b = _case(2)
    source, _, w = _q_data(b, 1)
    m = b.basis.memory.orbital_count
    block = _build(b, source, w, _selection((0,m), (0,m)))
    old = _full(b, source, w)
    np.testing.assert_array_equal(block.tensor_copy(), old.tensor_copy())
    assert block.identity_sha256 == old.identity_sha256
    assert block.payload_sha256 == _full_payload_wire(block)
    assert block.memory.plan_identity_sha256 == _full_plan_wire(b, source, w, block.memory)
    for field in _PLAN_FIELDS:
        assert getattr(block.memory, field) == getattr(old.memory, field)


def test_block_owns_no_full_orbital_square_output_or_coefficient_panel():
    b = _case(3)
    source, _, w = _q_data(b, 1)
    s = _selection((1,2), (4,1))
    p = _plan(b, source, w, s, begin=1, count=1)
    full = _full_plan(b, source, w, begin=1, count=1)
    assert p.retained_factor_bytes == p.compensation_bytes == 16*2
    assert p.coefficient_panel_bytes == 16*b.ao.nbasis*(2+1)
    assert p.driver_owned_numerical_bytes == 32*2+16*b.ao.nbasis*3+32*b.ao.nbasis+16*len(b.rows)
    assert p.driver_owned_numerical_bytes < full.driver_owned_numerical_bytes
    assert p.contraction_term_count == b.context.n_kpoints*b.ao.nbasis**2*2
    assert p.tile_calls == full.tile_calls
    assert p.resident_whitener_bytes == full.resident_whitener_bytes
    assert p.live_basis_bytes == full.live_basis_bytes  # Borrowed common F still exists upstream.
    result = _build(b, source, w, s, begin=1, count=1)
    assert result.diagnostics.completed_tile_calls == p.tile_calls
    assert result.diagnostics.maximum_observed_owned_numerical_bytes <= p.peak_owned_numerical_bytes
    assert result.diagnostics.maximum_observed_per_replica_inventoried_bytes <= p.per_replica_inventoried_bytes
    np.testing.assert_array_equal(result.tensor_copy(), _full(b, source, w).tensor_copy()[1:2,1:3,4:5])


def test_complex_sine_lane_and_opposite_density_orientation_are_not_conjugated_at_same_q():
    b = _small_bundle((3,1,1), rows=[[0,1]], translation=1)
    source, _, w = _q_data(b, 1)
    lr = _build(b, source, w, _selection((0,1), (1,1))).tensor_copy()[:,0,0]
    rl = _build(b, source, w, _selection((1,1), (0,1))).tensor_copy()[:,0,0]
    expected = _dense_panel(b, source, w)
    np.testing.assert_allclose(lr, expected[:,0,1], atol=4e-12, rtol=3e-11)
    np.testing.assert_allclose(rl, expected[:,1,0], atol=4e-12, rtol=3e-11)
    assert np.linalg.norm(lr.imag) > 1e-7
    assert np.linalg.norm(lr-rl.conj()) > 1e-7


@pytest.mark.parametrize("field,reported", list(_CAP_TO_PLAN.items()))
def test_resource_caps_exact_then_minus_one_precede_gauge_scan(field, reported):
    b = _case(2)
    source, _, w = _q_data(b, 1)
    s = _selection((1,2), (0,1))
    p = _plan(b, source, w, s)
    caps = b.panel_caps
    resource = caps.resources
    setattr(resource, field, getattr(p, reported))
    caps.resources = resource
    _build(b, source, w, s, caps=caps)
    setattr(resource, field, getattr(p, reported)-1)
    caps.resources = resource
    corrupt = b.gauge.copy()
    corrupt[:] = np.nan
    with pytest.raises(ValueError, match="cap"):
        _build(b, source, w, s, caps=caps, gauges=corrupt)


@pytest.mark.parametrize("field,value", [("left_count",0), ("right_count",0),
    ("left_begin",4), ("right_begin",4), ("left_count",5), ("right_count",5),
    ("left_begin",2**64-1), ("right_count",2**64-1)])
def test_invalid_ranges_fail_closed(field, value):
    b = _case(2)
    source, _, w = _q_data(b, 1)
    s = _selection((0,1), (0,1))
    setattr(s, field, value)
    with pytest.raises((ValueError, OverflowError, IndexError)):
        _plan(b, source, w, s)


def test_native_range_and_output_copies_are_immutable_and_owner_survives_inputs():
    b = _case(2)
    source, _, w = _q_data(b, 1)
    s = _selection((0,1), (2,2))
    block = _build(b, source, w, s)
    expected, identity = block.tensor_copy(), block.identity_sha256
    s.left_count = 999
    copy = block.selection
    copy.right_count = 999
    plan = block.memory
    copy = plan.selection
    copy.left_begin = 999
    block.tensor_copy()[:] = np.nan
    assert block.selection.left_count == 1 and block.selection.right_count == 2
    assert plan.selection.left_begin == 0
    del b, source, w
    gc.collect()
    np.testing.assert_array_equal(block.tensor_copy(), expected)
    assert block.identity_sha256 == identity


def test_virtual_slice_offsets_are_relative_to_the_certified_common_selection():
    from tests.test_periodic_correlation_real_local_basis import _make as make_basis
    from tests.test_periodic_correlation_density_factors import _virtual
    b = _case(3)
    b.selected = _virtual(1, 2, 1)
    b.basis = make_basis(b, b.rows, b.selected)
    source, _, w = _q_data(b, 1)
    block = _build(b, source, w, _selection((2,2), (4,1)))
    expected = _dense_panel(b, source, w)
    np.testing.assert_allclose(block.tensor_copy(), expected[:,2:4,4:5], atol=4e-12, rtol=3e-11)
    np.testing.assert_array_equal(block.tensor_copy(), _full(b, source, w).tensor_copy()[:,2:4,4:5])


def test_rectangles_keep_frozen_core_excluded_from_the_active_occupied_columns():
    from tests.test_periodic_gaussian_frozen_core_correlation import _frozen_bundle
    from tests.test_periodic_gaussian_metric import _live
    b = _frozen_bundle((2,1,1))
    _prepare_leaves(b)
    b.panel_live = core._PeriodicGaussianLocalOrbitalFactorLiveInventory()
    b.panel_live.fixed_backend_margin_bytes_per_worker = 65536
    # The dense-oracle helper uses all auxiliary columns as its whitening
    # block. The actual-HF factory requires the captured resource recipe,
    # whose block is2 for this four-auxiliary He2 fixture.
    source = core._make_periodic_gaussian_reciprocal_source(b.context, 1, b.config.source_caps)
    raw = core._build_periodic_gaussian_reciprocal_metric(source, b.ao, b.auxiliary,
        b.config.metric, _live(), b.config.metric_caps)
    w = core._factorize_periodic_gaussian_metric(raw, b.config.metric.whitener_column_block,
        _live(), b.config.metric_caps)
    assert b.hf.state.n_frozen_core == 1 and len(b.rows) == 2
    block = _build(b, source, w, _selection((1,2), (2,1)))
    np.testing.assert_allclose(block.tensor_copy(), _dense_panel(b, source, w)[:,1:3,2:3], atol=4e-12, rtol=3e-11)


@pytest.mark.parametrize("nk", [1, 2, 3])
def test_provider_panel_control_reservation_does_not_depend_on_generous_source_caps(nk):
    from tests.test_periodic_gaussian_rhf import _bundle as hf_bundle, _run as run_hf
    from tests.test_periodic_gaussian_selected_local_ccsd_t import _admit
    from tests.test_periodic_gaussian_one_electron import _normalized_basis
    from tests.test_periodic_gaussian_real_local_provider import _plan as provider_plan, _make as make_provider
    b = hf_bundle((nk,1,1))
    source = core._make_periodic_gaussian_reciprocal_source(b.context, 0, b.config.source_caps)
    sc = b.config.source_caps
    sc.maximum_fixed_storage_bytes = source.inventory.inventoried_fixed_storage_bytes
    config = b.config
    config.source_caps = sc
    b.config = config
    b.hf = run_hf(b)
    assert b.hf.converged
    b.minimal = _normalized_basis([(0,(0,0,0),[1.8],[1.0],True)])
    _admit(b, (nk,1,1))
    _prepare_leaves(b)
    b.panel_live = core._PeriodicGaussianLocalOrbitalFactorLiveInventory()
    b.panel_live.fixed_backend_margin_bytes_per_worker = 65536
    source, _, w = _q_data(b, 0)
    parent, leaf = provider_plan(b), _full_plan(b, source, w, count=1)
    # This lower bound intentionally omits the sibling W and panel objects:
    # even it must fit independently of the source's unrelated fixed cap.
    minimum = leaf.macro_fixed_object_bytes + leaf.tile_fixed_object_bytes + source.inventory.fixed_source_storage_bytes
    assert parent.maximum_leaf_fixed_object_bytes >= minimum
    result = make_provider(b)
    assert result.receipt.maximum_observed_per_replica_inventoried_bytes <= parent.per_replica_inventoried_bytes
