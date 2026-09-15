"""Actual mixed K/J/O and an independent Riplinger2013 Eq.26 oracle.

The physical factory accepts no T amplitudes. The separate numerical Eq.26
leaf is checked with clearly identified source amplitudes in its own PNO
frame. Neither leaf is a complete CCSD residual, energy, or DLPNO method.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_mixed_pair_factors import (
    _mixed_case, _equal_owner_cases, _controls as _panel_controls,
    local_geometry_case, _geometry_view, _geometry_payload_bytes,
)
from tests.test_periodic_gaussian_embedded_pair_pnos import (
    _options as _embedded_options, _make as _embedded, _physical, _frame,
)
from tests.test_periodic_gaussian_pair_pnos import _make as _legacy
from tests.test_periodic_gaussian_pair_space import _make as _pair_space
from tests.test_periodic_correlation_real_local_provider import _options as _real_options
from tests.test_periodic_correlation_real_pao_embedding import _torus
from tests.test_periodic_correlation_density_factors import _virtual_columns
from tests.test_periodic_aopair_fourier_panel import _basis as _raw_basis


_RESOURCE_CAPS = dict(maximum_owned_numeric_bytes="peak_owned_numerical_bytes",
    maximum_per_replica_inventoried_bytes="per_worker_inventoried_bytes",
    maximum_node_inventoried_bytes="required_node_memory_bytes",
    maximum_candidate_evaluations="reciprocal_candidate_evaluations_upper_bound",
    maximum_work_units="work_units")
_DIRECT_CAPS = dict(maximum_factor_panels="factor_panels", maximum_tile_calls="tile_calls",
    maximum_image_candidate_evaluations="image_candidate_evaluations_upper_bound")


@pytest.fixture(scope="module", params=[2, 3])
def mixed(request):
    return _mixed_case(request.param)


@pytest.fixture(scope="module")
def k2():
    return _mixed_case(2)


def _controls(case, *, auxiliary_block=3, ao_pair_block=3):
    panel, live, panel_caps = _panel_controls(case.b, pair_block=ao_pair_block)
    config = core._PeriodicGaussianPairInteractionConfig()
    config.auxiliary_block, config.panel = auxiliary_block, panel
    options = core._PeriodicGaussianPairInteractionOptions()
    options.real_projection = _real_options()
    options.maximum_overlap_imaginary_norm = 1e-10
    caps = core._PeriodicGaussianPairInteractionCaps()
    resources = core._PeriodicGaussianMetricCaps()
    for field, value in dict(maximum_owned_numeric_bytes=2**21,
        maximum_per_replica_inventoried_bytes=2**24, maximum_node_inventoried_bytes=2**26,
        maximum_candidate_evaluations=10**16, maximum_work_units=10**18).items():
        setattr(resources, field, value)
    caps.resources, caps.metric, caps.panel = resources, case.b.config.metric_caps, panel_caps
    caps.maximum_factor_panels, caps.maximum_tile_calls = 64, 4096
    caps.maximum_image_candidate_evaluations = 10**16
    caps.maximum_leaf_control_bytes = 2**20
    return config, options, live, caps


def _arguments(case, controls=None, **changes):
    config, options, live, caps = _controls(case) if controls is None else controls
    b = case.b
    result = dict(hf=b.hf, reference=b.reference, wannier=b.wannier, domain=b.domain,
        space=b.space, basis=b.basis, target_frame=case.a, source_frame=case.source,
        occupied_slot_i=case.i, occupied_slot_k=case.k,
        config=config, options=options, live=live, caps=caps)
    result.update(changes)
    return result


def _plan(case, controls=None, **changes):
    return core._plan_periodic_gaussian_pair_interaction_blocks(**_arguments(case, controls, **changes))


def _build(case, controls=None, **changes):
    args = _arguments(case, controls)
    args.update(ao_basis=case.b.ao, auxiliary_basis=case.b.auxiliary, gauges=case.b.gauge)
    args.update(changes)
    return core._build_periodic_gaussian_pair_interaction_blocks(**args)


def _oracle(case, *, a=None, source=None, i=None, k=None):
    b, dense = case.b, case.dense
    a, source = case.a if a is None else a, case.source if source is None else source
    i, k = case.i if i is None else i, case.k if k is None else k
    u, v, o = a.coefficients_copy(), source.coefficients_copy(), dense["o"]
    # K(i,a|k,c) and J(i,k|c,a) have different occupied index positions.
    common_k, common_j = dense["eri"][i, o:, k, o:], dense["eri"][i, k, o:, o:]
    kab, jba = u.T@common_k@v, v.T@common_j@u
    # Independent finite-torus ORIGINAL-S overlap. Do not assume that C^T C
    # alone proves the physical metric of a supplied generation frame.
    geometry = SimpleNamespace(**vars(b))
    geometry.pair_domain, geometry.pair_real_space, geometry.pair_selected = b.domain, b.real_space, b.selected
    torus = _torus(geometry)
    overlap = (torus.common@v).conj().T@torus.s@(torus.common@u)
    direct_k = np.zeros_like(overlap)
    for q in range(b.context.n_kpoints):
        common = _virtual_columns(b, q, b.selected)
        direct_k += (common@v).conj().T@b.reference.state.overlap(q)@(common@u)/b.context.n_kpoints
    np.testing.assert_allclose(overlap, direct_k, atol=4e-12, rtol=3e-11)
    assert np.linalg.norm(overlap.imag) < 1e-11
    return SimpleNamespace(k=kab, j=jba, overlap=overlap.real,
        common_k=common_k, common_j=common_j, u=u, v=v)


def _check(result, expected):
    np.testing.assert_allclose(result.k_ab_copy(), expected.k, atol=5e-12, rtol=3e-11)
    np.testing.assert_allclose(result.j_ba_copy(), expected.j, atol=5e-12, rtol=3e-11)
    np.testing.assert_allclose(result.overlap_ba_copy(), expected.overlap, atol=5e-12, rtol=3e-11)
    assert not result.production_dlpno and not result.infinite_source_accuracy_certified
    assert not result.original_provider_projection_reproduced
    d, p = result.diagnostics, result.memory
    assert d.completed_source_count == d.completed_metric_count == d.completed_whitening_count == p.n_cells
    assert d.completed_factor_panels == p.factor_panels
    assert d.completed_tile_calls == p.tile_calls
    assert d.charged_work_units_upper_bound <= p.work_units
    assert d.reciprocal_candidate_evaluations <= p.reciprocal_candidate_evaluations_upper_bound
    assert d.image_candidate_evaluations <= p.image_candidate_evaluations_upper_bound
    assert d.maximum_observed_owned_numerical_bytes <= p.peak_owned_numerical_bytes
    assert d.maximum_observed_per_worker_inventoried_bytes <= p.per_worker_inventoried_bytes
    assert d.maximum_integral_roundoff_error <= 1e-10
    for name in ("identity_sha256", "payload_sha256", "consumed_sources_identity_sha256"):
        value = getattr(result, name)
        assert len(value) == 64 and int(value, 16) >= 0


def _eq26(blocks, t, *, reverse=False):
    k, j, overlap = (blocks.k_ab_copy(), blocks.j_ba_copy(), blocks.overlap_ba_copy())
    inventory = core._BoundedRestrictedPairCCSDInteractionInventory()
    inventory.numerical_replicas = 1
    inventory.backend_margin_bytes_per_replica = 65536
    inventory.other_live_numerical_bytes_per_replica = 2**20
    inventory.other_live_control_bytes_per_replica = 65536
    p = core._plan_bounded_restricted_pair_ccsd_interaction(k.shape[0], k.shape[1], inventory)
    caps = core._BoundedRestrictedPairCCSDInteractionCaps()
    for field, value in dict(maximum_target_dimension=k.shape[0], maximum_source_dimension=k.shape[1],
        maximum_borrowed_numerical_bytes=p.borrowed_numerical_bytes,
        maximum_owned_numerical_bytes=p.peak_owned_numerical_bytes,
        maximum_control_storage_bytes=p.total_control_storage_bytes,
        maximum_per_replica_inventoried_bytes=p.per_replica_inventoried_bytes,
        maximum_node_inventoried_bytes=p.required_node_inventoried_bytes,
        maximum_scalar_products=p.scalar_products, maximum_work_units=p.work_units_upper_bound).items():
        setattr(caps, field, value)
    return core._bounded_restricted_pair_ccsd_interaction_diagnostic(k, j, overlap,
        np.ascontiguousarray(t, dtype=np.float64), inventory, caps, source_transposed=reverse)


def test_all_q_mixed_integrals_and_original_overlap_match_independent_oracle(mixed):
    result = _build(mixed)
    expected = _oracle(mixed)
    _check(result, expected)
    assert result.k_ab_copy().shape == (2, 3)
    assert result.j_ba_copy().shape == result.overlap_ba_copy().shape == (3, 2)
    assert np.linalg.norm(expected.k) > 1e-5
    # The occupied(0,translated0) density is suppressed by this explicit
    # finite AO-image source. Preserve its genuinely near-zero J channel;
    # the same-source nonzero operator channel is tested separately below.
    assert np.linalg.norm(expected.j) < 1e-12
    assert result.target_frame_identity_sha256 == mixed.a.identity_sha256
    assert result.source_frame_identity_sha256 == mixed.source.identity_sha256
    assert result.hf_reference_source_identity_sha256 == mixed.b.hf.reference_source_identity_sha256


def test_same_cell_operator_channel_has_nonzero_j_and_correct_half_weight(mixed):
    # Operator slots are independent of frame-generation pair labels.
    # Keeping the same authentic U/V frames and HF/source while using(i,i)
    # tests the nonzero Coulomb channel, not a claimed full pair residual.
    blocks = _build(mixed, occupied_slot_k=mixed.i)
    expected = _oracle(mixed, k=mixed.i)
    _check(blocks, expected)
    assert np.linalg.norm(expected.j) > 1e-5
    r = expected.v.shape[1]
    t = np.eye(r)+.2*np.triu(np.ones((r, r)), 1)
    # Explicit diagnostic amplitudes, not a caller-certified CCSD snapshot.
    z = 2*t-t.T
    wanted = (expected.k-.5*expected.j.T)@z@expected.overlap
    common_t = expected.v@t@expected.v.T
    independent = expected.u.T@(expected.common_k-.5*expected.common_j.T)@(
        2*common_t-common_t.T)@expected.u
    np.testing.assert_allclose(wanted, independent, atol=5e-13, rtol=3e-11)
    np.testing.assert_allclose(_eq26(blocks, t).residual_copy(), independent,
        atol=5e-12, rtol=3e-11)
    without_coulomb = expected.k@z@expected.overlap
    assert np.linalg.norm(wanted-without_coulomb) > 1e-8


def test_source_pair_reversal_and_premature_target_projection_are_distinguishable(mixed):
    blocks, expected = _build(mixed), _oracle(mixed)
    u, v, n = expected.u, expected.v, mixed.dense["v"]
    assert np.linalg.norm((np.eye(n)-u@u.T)@v) > .1
    assert np.linalg.norm((np.eye(n)-v@v.T)@u) > .1
    assert np.linalg.norm(v.T@u) > .1
    # Actual scalar-derived semicanonical starting amplitudes of canonical
    # source pair (j,k)=(0,2). They are not a converged CCSD claim.
    o, j, k = mixed.dense["o"], 0, mixed.k
    eps = mixed.source.energies_copy()
    foo = mixed.b.basis.fock_copy()
    g = v.T@mixed.dense["eri"][j, o:, k, o:]@v
    delta = eps[:, None]+eps[None, :]-foo[j, j]-foo[k, k]
    assert delta.min() > 1e-8
    stored = -g/delta
    assert np.linalg.norm(stored-stored.T) > 1e-8
    # The requested (k,j) source is reversed, so only T is transposed.
    oriented = stored.T
    z = 2*oriented-oriented.T
    wanted = (expected.k-.5*expected.j.T)@z@expected.overlap
    common_t = v@oriented@v.T
    d = expected.common_k-.5*expected.common_j.T
    common_residual = u.T@d@(2*common_t-common_t.T)@u
    np.testing.assert_allclose(wanted, common_residual, atol=5e-13, rtol=3e-11)
    actual = _eq26(blocks, stored, reverse=True)
    np.testing.assert_allclose(actual.residual_copy(), common_residual, atol=5e-12, rtol=3e-11)
    assert actual.source_transposed and not actual.physical_source_certified
    wrong = (u.T@d@u)@(u.T@v)@z@(v.T@u)
    assert np.linalg.norm(wanted-wrong) > 1e-10
    unreversed = _eq26(blocks, stored).residual_copy()
    assert np.linalg.norm(actual.residual_copy()-unreversed) > 1e-10
    # This is one diagram, not a complete residual to be symmetrized.
    np.testing.assert_allclose(unreversed,
        (expected.k-.5*expected.j.T)@(2*stored-stored.T)@expected.overlap,
        atol=5e-12, rtol=3e-11)


@pytest.mark.parametrize("auxiliary_block,ao_pair_block", [(1, 1), (3, 5), (4, 16)])
def test_ragged_stream_partitions_preserve_k_j_and_overlap(k2, auxiliary_block, ao_pair_block):
    result = _build(k2, _controls(k2, auxiliary_block=auxiliary_block, ao_pair_block=ao_pair_block))
    _check(result, _oracle(k2))
    p = result.memory
    assert p.factor_panels == 2*((4+auxiliary_block-1)//auxiliary_block)
    assert p.tile_calls == p.factor_panels*2*((16+ao_pair_block-1)//ao_pair_block)


def test_swapped_frames_and_occupied_labels_obey_mixed_tensor_orientation(k2):
    direct = _build(k2)
    swapped = _build(k2, target_frame=k2.source, source_frame=k2.a,
        occupied_slot_i=k2.k, occupied_slot_k=k2.i)
    _check(swapped, _oracle(k2, a=k2.source, source=k2.a, i=k2.k, k=k2.i))
    np.testing.assert_allclose(swapped.k_ab_copy(), direct.k_ab_copy().T, atol=5e-12)
    np.testing.assert_allclose(swapped.j_ba_copy(), direct.j_ba_copy().T, atol=5e-12)
    np.testing.assert_allclose(swapped.overlap_ba_copy(), direct.overlap_ba_copy().T, atol=5e-12)


def test_same_complete_legacy_frame_is_the_common_virtual_limit(k2):
    a = _legacy(k2.b, k2.provider)
    result = _build(k2, target_frame=a, source_frame=a)
    _check(result, _oracle(k2, a=a, source=a))
    np.testing.assert_allclose(result.overlap_ba_copy(), np.eye(4), atol=5e-11)
    assert result.memory.borrowed_frame_numerical_bytes == 8*(4*4+4+4)
    source, target = _legacy(k2.b, k2.provider, 0, 2), _embedded(k2.b, k2.provider, k2.ea)
    direct = _build(k2, target_frame=target, source_frame=source)
    target_space, source_space = _pair_space(k2.b, k2.provider, target), _pair_space(k2.b, k2.provider, source)
    wrapped = _build(k2, target_frame=target_space, source_frame=source_space)
    _check(wrapped, _oracle(k2, a=target_space, source=source_space))
    np.testing.assert_array_equal(wrapped.k_ab_copy(), direct.k_ab_copy())
    np.testing.assert_array_equal(wrapped.j_ba_copy(), direct.j_ba_copy())
    assert wrapped.memory.per_worker_inventoried_bytes > direct.memory.per_worker_inventoried_bytes


def test_retained_output_is_only_k_j_o_and_phase_inventory_is_complete(mixed):
    controls = _controls(mixed)
    config, _, live, caps = controls
    p = _plan(mixed, controls)
    ab, h, m, nk, a = 6, 28, 7, mixed.b.context.n_kpoints, 4
    nonself = nk == 3
    assert p.retained_output_bytes == 24*ab
    assert p.integral_accumulator_bytes == 48*ab
    assert p.norm_workspace_bytes == 24*h
    assert p.row_slab_bytes == (16 if nonself else 8)*config.auxiliary_block*h
    assert p.factor_phase_retained_bytes == p.retained_output_bytes+p.integral_accumulator_bytes+p.norm_workspace_bytes+p.row_slab_bytes
    assert p.whitener_bytes == 16*a*a
    assert p.maximum_live_whitener_bytes == (2 if nonself else 1)*16*a*a
    assert p.retained_partner_panel_bytes == (16*config.auxiliary_block*m*m if nonself else 0)
    assert p.peak_owned_numerical_bytes == max(p.overlap_phase_bytes, p.metric_phase_upper_bytes, p.panel_phase_upper_bytes)
    assert p.required_node_memory_bytes == p.reference_base_node_bytes+p.replicas_per_node*p.per_worker_inventoried_bytes
    assert p.per_worker_inventoried_bytes == (p.peak_owned_numerical_bytes+p.borrowed_local_numerical_bytes
        +p.borrowed_frame_numerical_bytes+p.borrowed_basis_active_numeric_bytes+p.control_storage_reservation_bytes
        +live.other_retained_bytes_per_worker+live.other_transient_bytes_per_worker+live.fixed_backend_margin_bytes_per_worker)
    assert p.row_slab_bytes < 8*nk*a*h  # No retained all-q real rows.
    live.other_retained_bytes_per_worker += 123
    live.other_transient_bytes_per_worker += 456
    changed = _plan(mixed, controls)
    assert changed.required_node_memory_bytes-p.required_node_memory_bytes == 579*p.replicas_per_node
    result = _build(mixed)
    assert not hasattr(result, "rows_copy") and not hasattr(result, "provider")


@pytest.mark.parametrize("field,plan_field", list(_RESOURCE_CAPS.items())+list(_DIRECT_CAPS.items()))
def test_cap_minus_one_precedes_expensive_input_scan(k2, field, plan_field):
    controls = _controls(k2)
    p, caps = _plan(k2, controls), controls[-1]
    if field in _RESOURCE_CAPS:
        resource = caps.resources
        setattr(resource, field, getattr(p, plan_field)-1)
        caps.resources = resource
    else:
        setattr(caps, field, getattr(p, plan_field)-1)
    with pytest.raises((ValueError, OverflowError), match="cap|exceed"):
        _build(k2, controls, gauges=np.full_like(k2.b.gauge, np.nan))


def test_exact_macro_caps_are_accepted(k2):
    controls = _controls(k2)
    p, caps = _plan(k2, controls), controls[-1]
    resources = caps.resources
    for field, plan_field in _RESOURCE_CAPS.items():
        setattr(resources, field, getattr(p, plan_field))
    caps.resources = resources
    for field, plan_field in _DIRECT_CAPS.items():
        setattr(caps, field, getattr(p, plan_field))
    _check(_build(k2, controls), _oracle(k2))


@pytest.mark.parametrize("which", ["target", "source", "both"])
def test_zero_rank_is_explicit_empty_shape_not_an_invented_direction(k2, which):
    cutoff = 1+max(k2.a.original_pno_occupations_copy().max(), k2.source.original_pno_occupations_copy().max())
    a = (_embedded(k2.b, k2.provider, k2.ea, options=_embedded_options(cutoff=cutoff))
        if which in ("target", "both") else k2.a)
    source = (_embedded(k2.b, k2.provider, k2.eb, 0, 2, options=_embedded_options(cutoff=cutoff))
        if which in ("source", "both") else k2.source)
    result = _build(k2, target_frame=a, source_frame=source)
    _check(result, _oracle(k2, a=a, source=source))
    ra, rb = a.coefficients_copy().shape[1], source.coefficients_copy().shape[1]
    assert result.k_ab_copy().shape == (ra, rb) and result.overlap_ba_copy().shape == (rb, ra)
    assert result.memory.retained_output_bytes == 0
    numerical = _eq26(result, np.zeros((rb, rb)))
    np.testing.assert_array_equal(numerical.residual_copy(), np.zeros((ra, ra)))


@pytest.mark.parametrize("field", ["maximum_overlap_imaginary_norm", "maximum_eri_projection_error",
    "maximum_scalar_roundoff_error"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_explicit_projection_budgets_are_required(k2, field, value):
    controls = _controls(k2)
    options = controls[1]
    if field == "maximum_overlap_imaginary_norm":
        setattr(options, field, value)
    else:
        projection = options.real_projection
        setattr(projection, field, value)
        options.real_projection = projection
    with pytest.raises((ValueError, OverflowError), match="positive|finite|budget|tolerance"):
        _plan(k2, controls)


def test_real_owner_content_and_runtime_input_validation_are_not_labels():
    k2, other = _equal_owner_cases()
    assert other.source.identity_sha256 == k2.source.identity_sha256
    with pytest.raises((ValueError, RuntimeError), match="owner|frame|source|state|context"):
        _plan(k2, source_frame=other.source)
    with pytest.raises((ValueError, RuntimeError), match="owner|state|context"):
        _plan(k2, hf=other.b.hf)
    with pytest.raises((ValueError, RuntimeError), match="basis|content|identity"):
        _build(k2, ao_basis=k2.b.auxiliary)
    with pytest.raises((ValueError, RuntimeError), match="finite|gauge|payload"):
        _build(k2, gauges=np.full_like(k2.b.gauge, np.nan))
    with pytest.raises((ValueError, RuntimeError), match="slot|frame"):
        _plan(k2, occupied_slot_i=len(k2.b.rows))


@pytest.mark.parametrize("role", ["ao_basis", "auxiliary_basis"])
def test_larger_actual_basis_is_rejected_before_nonfinite_payload_scan(k2, role):
    grown = _raw_basis([(0, p, [exponent]*20, [np.nan]*20, True)
        for p in ((0, 0, 0), (3, 0, 0)) for exponent in (.6, 1.3)])
    with pytest.raises((ValueError, OverflowError), match="cap|census|exceed"):
        _build(k2, **{role: grown})


def test_exact_minimum_private_leaf_control_allowance_and_minus_one(k2):
    controls = _controls(k2)
    p = _plan(k2, controls)
    assert 1 < p.minimum_leaf_control_bytes <= controls[-1].maximum_leaf_control_bytes
    controls[-1].maximum_leaf_control_bytes = p.minimum_leaf_control_bytes
    _check(_build(k2, controls), _oracle(k2))
    controls[-1].maximum_leaf_control_bytes -= 1
    with pytest.raises((ValueError, OverflowError), match="control.*cap|cap.*control"):
        _build(k2, controls, gauges=np.full_like(k2.b.gauge, np.nan))


def test_owned_result_copies_remain_valid_without_original_frames_or_hf():
    case = _mixed_case(2)
    result = _build(case)
    expected = (result.k_ab_copy(), result.j_ba_copy(), result.overlap_ba_copy())
    identity = result.identity_sha256
    del case
    gc.collect()
    for actual, wanted in zip((result.k_ab_copy(), result.j_ba_copy(), result.overlap_ba_copy()), expected):
        np.testing.assert_array_equal(actual, wanted)
        assert not actual.flags.writeable
    assert result.identity_sha256 == identity
    assert not hasattr(result, "target_frame") and not hasattr(result, "source_frame")


def test_local_pao_geometry_k_j_original_overlap_and_eq26_match_actual_oracles(local_geometry_case):
    case = local_geometry_case
    geometry = dict(geometry_a=_geometry_view(case.geometry_a, case.ea),
        geometry_b=_geometry_view(case.geometry_b, case.eb))
    local, common = _build(case, **geometry), _build(case)
    expected = _oracle(case)
    _check(local, expected)
    for left, right in ((local.k_ab_copy(), common.k_ab_copy()),
        (local.j_ba_copy(), common.j_ba_copy()), (local.overlap_ba_copy(), common.overlap_ba_copy())):
        np.testing.assert_allclose(left, right, atol=5e-12, rtol=3e-11)
    r = expected.v.shape[1]
    t = np.eye(r)+.2*np.triu(np.ones((r, r)), 1)
    tc = expected.v@t.T@expected.v.T
    wanted = expected.u.T@(expected.common_k-.5*expected.common_j.T)@(2*tc-tc.T)@expected.u
    np.testing.assert_allclose(_eq26(local, t, reverse=True).residual_copy(), wanted,
        atol=5e-12, rtol=3e-11)
    assert local.target_frame_identity_sha256 == common.target_frame_identity_sha256
    assert local.source_frame_identity_sha256 == common.source_frame_identity_sha256
    assert local.hf_reference_source_identity_sha256 == common.hf_reference_source_identity_sha256
    assert local.identity_sha256 != common.identity_sha256
    assert local.memory.local_geometry_a and local.memory.local_geometry_b
    assert not common.memory.local_geometry_a and not common.memory.local_geometry_b
    assert local.memory.borrowed_geometry_numerical_bytes == (
        _geometry_payload_bytes(case.geometry_a, case.ea)+_geometry_payload_bytes(case.geometry_b, case.eb))
    assert local.memory.geometry_validation_work_units > 0


def test_local_geometry_all_q_minimum_leaf_control_is_exact_and_precedes_scan(k2):
    geometry = dict(geometry_a=_geometry_view(k2.geometry_a, k2.ea),
        geometry_b=_geometry_view(k2.geometry_b, k2.eb))
    controls = _controls(k2)
    p = _plan(k2, controls, **geometry)
    controls[-1].maximum_leaf_control_bytes = p.minimum_leaf_control_bytes
    _check(_build(k2, controls, **geometry), _oracle(k2))
    controls[-1].maximum_leaf_control_bytes -= 1
    with pytest.raises((ValueError, OverflowError), match="control.*cap|cap.*control"):
        _build(k2, controls, gauges=np.full_like(k2.b.gauge, np.nan), **geometry)


def test_local_geometry_all_q_keeps_zero_rank_and_source_validation(k2):
    high = 1+max(k2.a.original_pno_occupations_copy().max(), k2.source.original_pno_occupations_copy().max())
    a = _embedded(k2.b, k2.provider, k2.ea, options=_embedded_options(cutoff=high))
    source = _embedded(k2.b, k2.provider, k2.eb, 0, 2, options=_embedded_options(cutoff=high))
    geometry = dict(geometry_a=_geometry_view(k2.geometry_a, k2.ea),
        geometry_b=_geometry_view(k2.geometry_b, k2.eb))
    result = _build(k2, target_frame=a, source_frame=source, **geometry)
    _check(result, _oracle(k2, a=a, source=source))
    assert result.k_ab_copy().shape == result.j_ba_copy().shape == (0, 0)
    wrong = core._PeriodicGaussianPairPNOGeometryView(
        k2.geometry_b.pair_domain, k2.geometry_b.pair_real_space, k2.ea)
    with pytest.raises((ValueError, RuntimeError), match="geometry|authenticated.*frame"):
        _build(k2, target_frame=a, source_frame=source,
            geometry_a=wrong, geometry_b=geometry["geometry_b"])


def test_frozen_occupied_bands_are_absent_from_mixed_factors_and_interactions():
    from tests.test_periodic_gaussian_frozen_core_correlation import _core_columns
    from tests.test_periodic_gaussian_mixed_pair_factors import _coefficients, _dense_mixed_panel, _q_data, _build as _panel

    b, provider, dense = _physical("he2", 2, frozen=True)
    ea, _ = _frame(b, columns=[[0, 0], [1, 2]])
    eb, _ = _frame(b, columns=[[0, 0], [0, 2], [1, 0]])
    a, source = _embedded(b, provider, ea), _embedded(b, provider, eb, 0, 1)
    case = SimpleNamespace(b=b, provider=provider, dense=dense, a=a, source=source, i=0, k=1)
    assert b.reference.state.n_frozen_core == b.reference.state.n_correlated_occupied == 1
    assert dense["o"] == 2 and dense["v"] == 4
    # Original occupied-space metric, not orbital-index relabeling. Both
    # active occupied columns and both PNO frames exclude the frozen band.
    cross = sum(_core_columns(b, k).conj().T@b.reference.state.overlap(k)@_coefficients(case, k)
                for k in range(2))/2
    np.testing.assert_allclose(cross, 0, atol=1e-11)
    for q in range(2):
        q_source, _, w = _q_data(b, q)
        panel = _panel(case, q_source, w)
        np.testing.assert_allclose(panel.tensor_copy(), _dense_mixed_panel(case, q_source, w), atol=5e-12, rtol=3e-11)
    _check(_build(case), _oracle(case))


def _gram_controls(case, *, auxiliary_block=None, ao_pair_block=None):
    return _controls(case,
        auxiliary_block=min(3, case.b.auxiliary.nbasis) if auxiliary_block is None else auxiliary_block,
        ao_pair_block=min(3, case.b.ao.nbasis**2) if ao_pair_block is None else ao_pair_block)


def _gram_geometry(case):
    return dict(geometry_a=case.ga.geometry_view(), geometry_b=case.gb.geometry_view())


def _gram_oracle(case):
    """Original PAOs times authentic local D; no exported C or provider."""
    from tests.test_periodic_gaussian_mixed_pair_factors import (
        _direct_gram_columns, _direct_gram_panel_oracle, _q_data,
    )
    from tests.test_periodic_correlation_real_local_provider import _expected_rows

    b, nk = case.b, case.b.context.n_kpoints
    panels = np.asarray([_direct_gram_panel_oracle(case, source, w)
        for source, _, w in (_q_data(b, q) for q in range(nk))])
    rows, labels = _expected_rows(panels, b.reference.state.mesh)
    ra, rb = case.a.coefficients_copy().shape[1], case.source.coefficients_copy().shape[1]
    kab, jba = np.zeros((ra, rb)), np.zeros((rb, ra))
    raw_k, raw_j = np.zeros((ra, rb), complex), np.zeros((rb, ra), complex)
    for a in range(ra):
        for c in range(rb):
            aa, cc = 2+a, 2+ra+c
            kab[a, c] = rows[:, labels.index((0, aa))]@rows[:, labels.index((1, cc))]
            jba[c, a] = rows[:, labels.index((0, 1))]@rows[:, labels.index((aa, cc))]
            # Chemists' density orientation reverses the first density
            # before conjugation. A same-oriented complex Gram is wrong.
            raw_k[a, c] = np.vdot(panels[:, :, aa, 0], panels[:, :, 1, cc])
            raw_j[c, a] = np.vdot(panels[:, :, 1, 0], panels[:, :, cc, aa])
    overlap = np.zeros((rb, ra), complex)
    for k in range(nk):
        columns = _direct_gram_columns(case, k)
        overlap += columns[:, 2+ra:].conj().T@b.reference.state.overlap(k)@columns[:, 2:2+ra]/nk
    np.testing.assert_allclose(overlap.imag, 0, atol=1e-11)
    np.testing.assert_allclose(raw_k, kab, atol=5e-12, rtol=3e-11)
    np.testing.assert_allclose(raw_j, jba, atol=5e-12, rtol=3e-11)
    return SimpleNamespace(k=kab, j=jba, overlap=overlap.real,
        raw_k=raw_k, raw_j=raw_j, panels=panels, rows=rows)


@pytest.fixture(scope="module", params=[
    (1, "he", False), (2, "he", False), (3, "he", False),
    (1, "he2", False), (2, "he2", False), (2, "he2", True),
])
def direct_gram_interaction(request):
    from tests.test_periodic_gaussian_mixed_pair_factors import _direct_gram_case

    nk, kind, frozen = request.param
    case = _direct_gram_case(nk, kind=kind, frozen=frozen)
    controls, geometry = _gram_controls(case), _gram_geometry(case)
    # New native source -> K/J/O precedes ALL independent dense or old-row
    # consumers. The fixture contains no hidden common-factor provider.
    result = _build(case, controls, **geometry)
    expected = _gram_oracle(case)
    assert not hasattr(case.b, "provider")
    return SimpleNamespace(case=case, controls=controls, geometry=geometry,
        result=result, expected=expected, frozen=frozen)


def test_direct_gram_all_q_k_j_and_original_overlap_without_common_provider(direct_gram_interaction):
    test = direct_gram_interaction
    case, result, expected = test.case, test.result, test.expected
    _check(result, expected)
    assert result.direct_gram_frames and result.memory.direct_gram_frames
    assert result.memory.source_receipt_control_bytes == 2*65
    assert result.target_gram_identity_sha256 == case.a.gram_identity_sha256
    assert result.source_gram_identity_sha256 == case.source.gram_identity_sha256
    assert result.target_frame_identity_sha256 == case.a.identity_sha256
    assert result.source_frame_identity_sha256 == case.source.identity_sha256
    assert result.memory.borrowed_frame_numerical_bytes == (
        case.a.retained_numerical_bytes+case.source.retained_numerical_bytes)
    assert result.memory.borrowed_geometry_numerical_bytes > 0
    nk = case.b.context.n_kpoints
    assert result.memory.self_inverse_q_count == (1 if nk == 3 else nk)
    if nk == 3:
        assert np.linalg.norm(expected.panels[1].imag) > 1e-12
    if test.frozen:
        from tests.test_periodic_gaussian_frozen_core_correlation import _core_columns
        from tests.test_periodic_gaussian_mixed_pair_factors import _direct_gram_columns
        b = case.b
        assert b.reference.state.n_frozen_core == 1
        cross = sum(_core_columns(b, k).conj().T@b.reference.state.overlap(k)@
            _direct_gram_columns(case, k)/nk for k in range(nk))
        np.testing.assert_allclose(cross, 0, atol=1e-11)
    # Raw source-frame amplitudes are diagnostic numbers, not a CCSD claim.
    rb = expected.overlap.shape[0]
    t = np.eye(rb)+.2*np.triu(np.ones((rb, rb)), 1)
    for reverse in (False, True):
        oriented = t.T if reverse else t
        wanted = (expected.k-.5*expected.j.T)@(2*oriented-oriented.T)@expected.overlap
        np.testing.assert_allclose(_eq26(result, t, reverse=reverse).residual_copy(), wanted,
            atol=5e-12, rtol=3e-11)


@pytest.fixture(scope="module")
def direct_gram_k2():
    from tests.test_periodic_gaussian_mixed_pair_factors import _direct_gram_case
    return _direct_gram_case(2)


@pytest.mark.parametrize("field,plan_field", list(_RESOURCE_CAPS.items())+list(_DIRECT_CAPS.items()))
def test_direct_gram_interaction_caps_precede_gauge_payload(direct_gram_k2, field, plan_field):
    case = direct_gram_k2
    controls, geometry = _gram_controls(case), _gram_geometry(case)
    p, caps = _plan(case, controls, **geometry), controls[-1]
    if field in _RESOURCE_CAPS:
        resources = caps.resources
        setattr(resources, field, getattr(p, plan_field)-1)
        caps.resources = resources
    else:
        setattr(caps, field, getattr(p, plan_field)-1)
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap|exceed"):
        _build(case, controls, gauges=np.full_like(case.b.gauge, np.nan), **geometry)


def test_direct_gram_interaction_exact_caps_and_leaf_control(direct_gram_k2):
    case = direct_gram_k2
    controls, geometry = _gram_controls(case), _gram_geometry(case)
    caps = controls[-1]
    p = _plan(case, controls, **geometry)
    caps.maximum_leaf_control_bytes = p.minimum_leaf_control_bytes
    p = _plan(case, controls, **geometry)
    resources = caps.resources
    for field, plan_field in _RESOURCE_CAPS.items():
        setattr(resources, field, getattr(p, plan_field))
    caps.resources = resources
    for field, plan_field in _DIRECT_CAPS.items():
        setattr(caps, field, getattr(p, plan_field))
    _check(_build(case, controls, **geometry), _gram_oracle(case))
    caps.maximum_leaf_control_bytes -= 1
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="control.*cap|cap.*control"):
        _build(case, controls, gauges=np.full_like(case.b.gauge, np.nan), **geometry)


@pytest.mark.parametrize("auxiliary_block,ao_pair_block", [(1, 1), (3, 5), (4, 16)])
def test_direct_gram_interaction_ragged_slab_splits(direct_gram_k2, auxiliary_block, ao_pair_block):
    case = direct_gram_k2
    controls = _gram_controls(case, auxiliary_block=auxiliary_block, ao_pair_block=ao_pair_block)
    _check(_build(case, controls, **_gram_geometry(case)), _gram_oracle(case))


def test_direct_gram_interaction_swapped_frames_and_nonzero_j_operator(direct_gram_k2):
    case = direct_gram_k2
    direct = _build(case, _gram_controls(case), **_gram_geometry(case))
    swapped = SimpleNamespace(**vars(case))
    swapped.a, swapped.source, swapped.ga, swapped.gb = case.source, case.a, case.gb, case.ga
    swapped.i, swapped.k = case.k, case.i
    result = _build(swapped, _gram_controls(swapped), **_gram_geometry(swapped))
    _check(result, _gram_oracle(swapped))
    np.testing.assert_allclose(result.k_ab_copy(), direct.k_ab_copy().T, atol=5e-12)
    np.testing.assert_allclose(result.j_ba_copy(), direct.j_ba_copy().T, atol=5e-12)
    np.testing.assert_allclose(result.overlap_ba_copy(), direct.overlap_ba_copy().T, atol=5e-12)
    same = SimpleNamespace(**vars(case))
    same.k = same.i
    result = _build(same, _gram_controls(same), **_gram_geometry(same))
    expected = _gram_oracle(same)
    _check(result, expected)
    assert np.linalg.norm(expected.j) > 1e-5


def test_direct_gram_interaction_requires_original_geometry_and_homogeneous_sources(direct_gram_k2):
    case = direct_gram_k2
    controls, geometry = _gram_controls(case), _gram_geometry(case)
    for kwargs in ({}, dict(geometry_a=geometry["geometry_a"]), dict(geometry_b=geometry["geometry_b"])):
        with pytest.raises((ValueError, RuntimeError), match="Gram frames require both original pair geometries"):
            _plan(case, controls, **kwargs)
    with pytest.raises((ValueError, RuntimeError), match="geometry|generation"):
        _plan(case, controls, geometry_a=geometry["geometry_b"], geometry_b=geometry["geometry_a"])
    # Old source construction is deliberate and confined to this rejection
    # witness, after a valid direct-Gram interaction has already completed.
    _build(case, controls, **geometry)
    from tests.test_periodic_gaussian_gram_pair_pnos import _oracle_inputs, _old_embedding, _old_pnos
    old = _old_pnos(case.b, _oracle_inputs(case.b).provider, _old_embedding(case.b))
    with pytest.raises((ValueError, RuntimeError), match="source kind|direct.*legacy|Gram.*legacy"):
        _plan(case, controls, source_frame=old, **geometry)


def test_direct_gram_interaction_rejects_independent_actual_hf_owner(direct_gram_k2):
    from tests.test_periodic_gaussian_mixed_pair_factors import _direct_gram_case
    case, other = direct_gram_k2, _direct_gram_case(2)
    geometry, controls = _gram_geometry(case), _gram_controls(case)
    # Both are genuine calculations of the same fixture. Exact physical
    # state/context ownership is required regardless of floating SHA equality.
    with pytest.raises((ValueError, RuntimeError), match="owner|state|source|frame|context"):
        _plan(case, controls, source_frame=other.source,
            geometry_a=geometry["geometry_a"], geometry_b=other.gb.geometry_view())
    with pytest.raises((ValueError, RuntimeError), match="owner|state|source|context"):
        _plan(case, controls, hf=other.b.hf, **geometry)
    assert not hasattr(case.b, "provider") and not hasattr(other.b, "provider")


@pytest.mark.parametrize("which", ["target", "source", "both"])
def test_direct_gram_interaction_empty_ranks_keep_geometry_validation(direct_gram_k2, which):
    from tests.test_periodic_gaussian_gram_pair_pnos import _make, _controls as seed_controls
    case = SimpleNamespace(**vars(direct_gram_k2))
    high = 1+max(case.a.original_pno_occupations_copy().max(), case.source.original_pno_occupations_copy().max())
    if which in ("target", "both"):
        case.a = _make(case.b, case.ga, seed_controls(case.b, cutoff=high))
    if which in ("source", "both"):
        case.source = _make(case.b, case.gb, seed_controls(case.b, cutoff=high))
    result = _build(case, _gram_controls(case), **_gram_geometry(case))
    expected = _gram_oracle(case)
    _check(result, expected)
    assert result.memory.retained_output_bytes == 0
    assert result.memory.borrowed_geometry_numerical_bytes > 0
    rb = expected.overlap.shape[0]
    np.testing.assert_array_equal(_eq26(result, np.zeros((rb, rb))).residual_copy(),
        np.zeros((expected.k.shape[0], expected.k.shape[0])))
    with pytest.raises((ValueError, RuntimeError), match="geometry|generation"):
        _plan(case, _gram_controls(case), geometry_a=case.gb.geometry_view(), geometry_b=case.ga.geometry_view())


def test_direct_gram_interaction_same_owner_and_wrapper_g_not_duplicated():
    from tests.test_periodic_gaussian_mixed_pair_factors import _direct_gram_case
    from tests.test_periodic_gaussian_gram_pair_pnos import _space
    deltas = []
    for cutoff in (0.0, 1.0):
        case = _direct_gram_case(2, cutoff=cutoff)
        case.source, case.gb = case.a, case.ga
        controls, geometry = _gram_controls(case), _gram_geometry(case)
        plain = _plan(case, controls, **geometry)
        expected = _gram_oracle(case)
        assert plain.borrowed_frame_numerical_bytes == case.a.retained_numerical_bytes
        owned = _space(case.b, case.a)
        result = _build(case, controls, target_frame=owned, source_frame=owned, **geometry)
        _check(result, expected)
        assert result.memory.borrowed_frame_numerical_bytes == plain.borrowed_frame_numerical_bytes
        deltas.append(result.memory.per_worker_inventoried_bytes-plain.per_worker_inventoried_bytes)
        with pytest.raises((ValueError, RuntimeError), match="mov|consum|live|empty"):
            _plan(case, controls, **geometry)
    # The exact same retained G belongs to the seed, not an extra PairSpace
    # allocation. Only one rank-independent immediate wrapper is charged.
    assert deltas[0] == deltas[1] > 0


def test_direct_gram_interaction_positive_cut_source_lineage_and_detached_lifetime():
    from tests.test_periodic_gaussian_mixed_pair_factors import _direct_gram_case
    from tests.test_periodic_gaussian_gram_pair_pnos import _make, _controls as seed_controls
    from tests.test_periodic_correlation_wannier import _make as make_wannier
    case = _direct_gram_case(2)
    occupations = case.a.original_pno_occupations_copy()
    assert len(occupations) == 2 and occupations[0] > occupations[1] >= 0
    case.a = _make(case.b, case.ga, seed_controls(case.b, cutoff=float(np.mean(occupations))))
    result = _build(case, _gram_controls(case), **_gram_geometry(case))
    assert result.memory.target_dimension == 1
    _check(result, _gram_oracle(case))
    snapshots = result.k_ab_copy(), result.j_ba_copy(), result.overlap_ba_copy()
    # Valid unitary, real/TR sign change preserves Foo but is not the
    # localization sealed by the common basis and original pair geometry.
    changed = np.ascontiguousarray(-case.b.gauge)
    wrong_wannier = make_wannier(case.b.reference, changed)
    with pytest.raises((ValueError, RuntimeError), match="basis|localiz|gauge|coefficient"):
        _build(case, _gram_controls(case), wannier=wrong_wannier, gauges=changed, **_gram_geometry(case))
    del case, wrong_wannier, changed
    gc.collect()
    for values, expected in zip((result.k_ab_copy(), result.j_ba_copy(), result.overlap_ba_copy()), snapshots):
        np.testing.assert_array_equal(values, expected)
        assert not values.flags.writeable


def test_legacy_interaction_has_no_direct_gram_receipt(k2):
    result = _build(k2)
    assert not result.direct_gram_frames and not result.memory.direct_gram_frames
    assert result.memory.source_receipt_control_bytes == 0
    for field in ("target_gram_identity_sha256", "source_gram_identity_sha256"):
        with pytest.raises((ValueError, RuntimeError), match="Gram|legacy|source"):
            getattr(result, field)
