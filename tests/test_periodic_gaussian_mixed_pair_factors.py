"""Actual finite-Gaussian factors in two independent, overlapping PNO frames.

Riplinger/Neese doi:10.1063/1.4773581 II.C requires mixed pair spaces:
their concatenation is not an orthonormal basis. The dense AO transforms
here are tiny independent oracles, not a production common-factor cache.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_embedded_pair_pnos import (
    _physical as _existing_physical, _frame, _make as _embedded, _options as _embedded_options,
)
from tests.test_periodic_gaussian_selected_local_ccsd_t import (
    _he2_bundle, _he2_controls, _prepare_leaves, _dense_case,
)
from tests.test_periodic_gaussian_real_local_provider import _make as _provider
from tests.test_periodic_gaussian_pair_pnos import _make as _legacy
from tests.test_periodic_gaussian_pair_space import _make as _pair_space
from tests.test_periodic_gaussian_fock import _physical_tile
from tests.test_periodic_gaussian_metric import _live as _metric_live
from tests.test_periodic_gaussian_local_orbital_factors import _caps as _panel_caps
from tests.test_periodic_correlation_density_factors import _occupied_columns, _virtual_columns
from tests.test_periodic_aopair_fourier_panel import _basis as _raw_basis


_RESOURCE_CAPS = dict(maximum_owned_numeric_bytes="peak_owned_numerical_bytes",
    maximum_per_replica_inventoried_bytes="per_replica_inventoried_bytes",
    maximum_node_inventoried_bytes="required_node_memory_bytes",
    maximum_candidate_evaluations="reciprocal_candidate_evaluations",
    maximum_work_units="work_units")
_DIRECT_CAPS = dict(maximum_tile_calls="tile_calls",
    maximum_image_candidate_evaluations="image_candidate_evaluations_upper_bound")


def _q_data(b, q):
    """Reproduce the actual HF metric recipe, including its W block size."""
    source = core._make_periodic_gaussian_reciprocal_source(b.context, q, b.config.source_caps)
    raw = core._build_periodic_gaussian_reciprocal_metric(source, b.ao, b.auxiliary,
        b.config.metric, _metric_live(), b.config.metric_caps)
    raw_identity = raw.payload_identity_sha256
    whitener = core._factorize_periodic_gaussian_metric(raw,
        b.config.metric.whitener_column_block, _metric_live(), b.config.metric_caps)
    return source, raw_identity, whitener


def _mixed_case(nk=2, *, frozen=False):
    """Different physical PAO domains, with no modified HF/source controls."""
    if frozen:
        b, provider, dense = _existing_physical("he2", nk, frozen=True)
    elif nk == 3:
        b = _he2_bundle((nk, 1, 1))
        local = _he2_controls(b)[0].localization
        optimizer = local.optimizer
        # The unchanged He2K3 objective exhausted32 accepted steps with
        # gradient1.9961975204108934e-5. Allow more iterations, retaining
        # the existing explicit2e-6 PM tolerance and every physical gate.
        optimizer.maximum_iterations = 128
        local.optimizer = optimizer
        _prepare_leaves(b, localization_options=local)
        assert b.localization.optimizer.diagnostics.riemannian_gradient_norm <= 2e-6
        provider, dense = _provider(b), _dense_case(b)
    else:
        b, provider, dense = _existing_physical("he2", nk)
    ea, geometry_a = _frame(b, columns=[[0, 0], [1, 2]])
    eb, geometry_b = _frame(b, columns=[[0, 0], [0, 2], [1, 0]])
    a = _embedded(b, provider, ea, 0, 0)
    # The translated source(k,0) uses canonical storage(0,k), where one
    # active orbital per cell remains after the explicit frozen-core mask.
    k = 1 if frozen else 2
    source = _embedded(b, provider, eb, 0, k)
    assert a.coefficients_copy().shape == (dense["v"], 2)
    assert source.coefficients_copy().shape == (dense["v"], 3)
    return SimpleNamespace(b=b, provider=provider, dense=dense, a=a, source=source,
        ea=ea, eb=eb, geometry_a=geometry_a, geometry_b=geometry_b, i=0, k=k)


def _equal_owner_cases():
    # Bitwise-equal SHA payloads require the same reduction order. Rebuild
    # BOTH genuine HF chains serially, not one cached parallel fixture.
    threads = core.get_num_threads()
    try:
        core.set_num_threads(1)
        first, second = _mixed_case(2), _mixed_case(2)
    finally:
        core.set_num_threads(threads)
    assert first.b.hf.state is not second.b.hf.state
    assert first.b.context is not second.b.context
    assert first.b.hf.state.state_identity_sha256 == second.b.hf.state.state_identity_sha256
    assert first.b.context.source_context_identity_sha256 == second.b.context.source_context_identity_sha256
    assert first.b.hf.reference_source_identity_sha256 == second.b.hf.reference_source_identity_sha256
    assert first.a.identity_sha256 == second.a.identity_sha256
    assert first.source.identity_sha256 == second.source.identity_sha256
    return first, second


@pytest.fixture(scope="module", params=[2, 3])
def mixed(request):
    return _mixed_case(request.param)


@pytest.fixture(scope="module")
def k2():
    return _mixed_case(2)


def _controls(b, *, pair_block=3):
    config = core._PeriodicGaussianLocalOrbitalFactorConfig()
    config.ao_pair_block = pair_block
    live = core._PeriodicGaussianLocalOrbitalFactorLiveInventory()
    # Explicit remainder: generation provider, other embeddings and tiny
    # dense test oracles. Shared HF/common geometry and both frames are
    # counted separately by the native leaf, not hidden in this allowance.
    live.other_retained_bytes_per_worker = 2**20
    live.fixed_backend_margin_bytes_per_worker = 65536
    return config, live, _panel_caps(b)


def _plan(case, source, whitener, *, frame_a=None, frame_b=None, i=None, k=None,
          begin=0, count=None, controls=None, geometry_a=None, geometry_b=None):
    b = case.b
    geometry = {key: value for key, value in dict(geometry_a=geometry_a, geometry_b=geometry_b).items()
        if value is not None}
    return core._plan_periodic_gaussian_mixed_pair_factor_panel(b.hf, b.reference,
        source, whitener, b.wannier, b.domain, b.space, b.basis,
        case.a if frame_a is None else frame_a, case.source if frame_b is None else frame_b,
        case.i if i is None else i, case.k if k is None else k, begin,
        b.auxiliary.nbasis-begin if count is None else count,
        *(_controls(b) if controls is None else controls), **geometry)


def _build(case, source, whitener, *, frame_a=None, frame_b=None, i=None, k=None,
           begin=0, count=None, controls=None, gauge=None, ao=None, auxiliary=None, hf=None,
           geometry_a=None, geometry_b=None):
    b = case.b
    geometry = {key: value for key, value in dict(geometry_a=geometry_a, geometry_b=geometry_b).items()
        if value is not None}
    return core._build_periodic_gaussian_mixed_pair_factor_panel(
        b.hf if hf is None else hf, b.reference, source, whitener,
        b.ao if ao is None else ao, b.auxiliary if auxiliary is None else auxiliary,
        b.wannier, b.gauge if gauge is None else gauge, b.domain, b.space, b.basis,
        case.a if frame_a is None else frame_a, case.source if frame_b is None else frame_b,
        case.i if i is None else i, case.k if k is None else k, begin,
        b.auxiliary.nbasis-begin if count is None else count,
        *(_controls(b) if controls is None else controls), **geometry)


def _coefficients(case, kpoint, a=None, source=None, *, i=None, k=None):
    b = case.b
    a, source = case.a if a is None else a, case.source if source is None else source
    occupied = _occupied_columns(b, kpoint, b.rows)
    virtuals = _virtual_columns(b, kpoint, b.selected)
    return np.column_stack((occupied[:, case.i if i is None else i],
        occupied[:, case.k if k is None else k],
        virtuals@a.coefficients_copy(), virtuals@source.coefficients_copy()))


def _dense_mixed_panel(case, source, whitener, *, a=None, other=None,
                       i=None, k=None, begin=0, count=None):
    """Direct AO-to-mixed transform; no native common/mixed factor leaf."""
    b, nk, n = case.b, case.b.context.n_kpoints, case.b.ao.nbasis
    count = b.auxiliary.nbasis-begin if count is None else count
    m = _coefficients(case, 0, a, other, i=i, k=k).shape[1]
    output = np.zeros((count, m, m), complex)
    for bra in range(nk):
        ket = b.context.ket_index(bra, source.q_index)
        left = _coefficients(case, bra, a, other, i=i, k=k)
        right = _coefficients(case, ket, a, other, i=i, k=k)
        ao = _physical_tile(b, source, whitener, bra, aux=begin, rows=count).matrix.reshape(count, n, n)
        output += np.einsum("ml,nr,pmn->plr", left.conj(), right, ao)
    return output/(nk*np.sqrt(nk))


def _common_transform(case, a=None, source=None, *, i=None, k=None):
    a, source = case.a if a is None else a, case.source if source is None else source
    u, v = a.coefficients_copy(), source.coefficients_copy()
    o, n = case.dense["o"], case.dense["v"]
    transform = np.zeros((o+n, 2+u.shape[1]+v.shape[1]))
    transform[case.i if i is None else i, 0] = 1
    transform[case.k if k is None else k, 1] = 1
    transform[o:, 2:2+u.shape[1]], transform[o:, 2+u.shape[1]:] = u, v
    return transform


def test_all_q_and_all_density_orientations_match_direct_actual_ao_transform(mixed):
    case, b = mixed, mixed.b
    transform = _common_transform(case)
    assert np.linalg.norm(transform.T@transform-np.eye(transform.shape[1])) > .1
    for q in range(b.context.n_kpoints):
        source, _, w = _q_data(b, q)
        result = _build(case, source, w)
        expected = _dense_mixed_panel(case, source, w)
        indirect = np.einsum("il,pij,jr->plr", transform, case.dense["panels"][q], transform)
        np.testing.assert_allclose(expected, indirect, atol=4e-12, rtol=3e-11)
        np.testing.assert_allclose(result.tensor_copy(), expected, atol=4e-12, rtol=3e-11)
        assert result.tensor_copy().shape == (4, 7, 7)
        assert result.rank_a == 2 and result.rank_b == 3
        assert result.occupied_i == tuple(b.rows[case.i]) and result.occupied_k == tuple(b.rows[case.k])
        assert result.state is b.reference.state and result.context is b.context
        assert result.frame_a_identity_sha256 == case.a.identity_sha256
        assert result.frame_b_identity_sha256 == case.source.identity_sha256
        assert result.frame_a_provider_identity_sha256 == case.provider.identity_sha256
        assert result.frame_b_provider_identity_sha256 == case.provider.identity_sha256
        assert result.source_identity_sha256 == source.source_identity_sha256
        assert result.conjugate_source_identity_sha256 == source.conjugate_source_identity_sha256
        assert result.whitener_payload_identity_sha256 == w.payload_identity_sha256
        assert result.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256
        assert result.finite_image_reference and result.matched_finite_gaussian_hf_recipe
        assert not result.jointly_orthonormal_basis_certified and not result.density_symmetry_certified
        assert not result.bitwise_origin_provider_factors_verified and not result.production_dlpno
        if b.context.n_kpoints == 3 and q:
            assert np.linalg.norm(expected.imag) > 1e-6


def test_swapping_mixed_frames_and_occupied_columns_is_only_an_axis_permutation(k2):
    source, _, w = _q_data(k2.b, 1)
    original = _build(k2, source, w).tensor_copy()
    swapped = _build(k2, source, w, frame_a=k2.source, frame_b=k2.a, i=k2.k, k=k2.i)
    order = [1, 0, 4, 5, 6, 2, 3]
    np.testing.assert_array_equal(swapped.tensor_copy(), original[:, order][:, :, order])
    assert (swapped.rank_a, swapped.rank_b) == (3, 2)


@pytest.mark.parametrize("pair_block", [1, 3, 5, 16])
def test_ragged_ao_and_auxiliary_blocks_preserve_factors(k2, pair_block):
    source, _, w = _q_data(k2.b, 1)
    controls = _controls(k2.b, pair_block=pair_block)
    result = _build(k2, source, w, controls=controls)
    slices = [_build(k2, source, w, begin=p, count=min(3, 4-p), controls=controls).tensor_copy()
        for p in range(0, 4, 3)]
    np.testing.assert_allclose(np.concatenate(slices), result.tensor_copy(), atol=5e-13, rtol=3e-12)
    np.testing.assert_allclose(result.tensor_copy(), _dense_mixed_panel(k2, source, w), atol=4e-12, rtol=3e-11)
    assert result.memory.tile_calls == 2*((16+pair_block-1)//pair_block)
    assert result.diagnostics.completed_tile_calls == result.memory.tile_calls
    assert result.diagnostics.charged_work_units_upper_bound <= result.memory.work_units


def test_exact_owned_and_borrowed_ragged_inventory_and_same_owner_dedup(k2):
    source, _, w = _q_data(k2.b, 0)
    p = _plan(k2, source, w, count=3)
    n, m, r, ab = 4, 7, 5, 3
    assert p.retained_factor_bytes == p.retained_output_bytes == 16*ab*m*m
    assert p.compensation_bytes == 16*ab*m*m
    assert p.coefficient_panel_bytes == 32*n*m
    assert p.coefficient_helper_workspace_bytes == 16*n*(r+3)
    assert p.driver_owned_numerical_bytes == 32*ab*m*m+32*n*m+16*n*(r+3)
    assert p.peak_owned_numerical_bytes == p.driver_owned_numerical_bytes+p.maximum_tile_owned_numerical_bytes
    assert p.resident_whitener_bytes == 16*4*4
    assert p.borrowed_frame_numerical_bytes == 8*(4*2+2*2+2+2)+8*(4*3+3*3+3+3)
    assert not p.same_frame_owner
    same = _plan(k2, source, w, frame_b=k2.a)
    assert same.same_frame_owner and same.borrowed_frame_numerical_bytes == 8*(4*2+2*2+2+2)
    assert p.required_node_memory_bytes == p.reference_base_node_bytes+p.replicas_per_node*p.per_replica_inventoried_bytes
    config, live, caps = _controls(k2.b)
    baseline = _plan(k2, source, w, controls=(config, live, caps))
    live.other_retained_bytes_per_worker += 123
    live.other_transient_bytes_per_worker += 456
    extra = _plan(k2, source, w, controls=(config, live, caps))
    assert extra.required_node_memory_bytes-baseline.required_node_memory_bytes == 579*p.replicas_per_node


@pytest.mark.parametrize("field,plan_field", list(_RESOURCE_CAPS.items())+list(_DIRECT_CAPS.items()))
def test_one_below_every_macro_cap_fails_before_nonfinite_gauge_scan(k2, field, plan_field):
    source, _, w = _q_data(k2.b, 0)
    config, live, caps = _controls(k2.b)
    p = _plan(k2, source, w, controls=(config, live, caps))
    if field in _RESOURCE_CAPS:
        resources = caps.resources
        setattr(resources, field, getattr(p, plan_field)-1)
        caps.resources = resources
    else:
        setattr(caps, field, getattr(p, plan_field)-1)
    with pytest.raises((ValueError, OverflowError), match="cap|exceed"):
        _build(k2, source, w, controls=(config, live, caps), gauge=np.full_like(k2.b.gauge, np.nan))


def test_explicit_zero_pno_ranks_keep_occupied_factors_and_source_checks(k2):
    cutoff = 1+max(k2.a.original_pno_occupations_copy().max(),
        k2.source.original_pno_occupations_copy().max())
    a = _embedded(k2.b, k2.provider, k2.ea, options=_embedded_options(cutoff=cutoff))
    other = _embedded(k2.b, k2.provider, k2.eb, 0, 2, options=_embedded_options(cutoff=cutoff))
    source, _, w = _q_data(k2.b, 1)
    result = _build(k2, source, w, frame_a=a, frame_b=other)
    assert result.tensor_copy().shape == (4, 2, 2)
    assert result.rank_a == result.rank_b == 0
    assert result.memory.coefficient_helper_workspace_bytes == 0
    assert result.memory.borrowed_frame_numerical_bytes == 8*(2+3)
    np.testing.assert_allclose(result.tensor_copy(), _dense_mixed_panel(k2, source, w, a=a, other=other), atol=4e-12)


def test_legacy_and_safe_pairspace_dispatch_preserve_the_actual_frame(k2):
    b = k2.b
    a = _legacy(b, k2.provider, 0, 0)
    other = _embedded(b, k2.provider, k2.eb, 0, 2)
    source, _, w = _q_data(b, 1)
    direct = _build(k2, source, w, frame_a=a, frame_b=other)
    a_space = _pair_space(b, k2.provider, a)
    b_space = _pair_space(b, k2.provider, other)
    wrapped = _build(k2, source, w, frame_a=a_space, frame_b=b_space)
    np.testing.assert_array_equal(wrapped.tensor_copy(), direct.tensor_copy())
    assert wrapped.frame_a_identity_sha256 == direct.frame_a_identity_sha256
    assert wrapped.frame_b_identity_sha256 == direct.frame_b_identity_sha256
    # PairSpace additionally retains G; binding must charge that owner.
    assert wrapped.memory.per_replica_inventoried_bytes > direct.memory.per_replica_inventoried_bytes


def test_equal_receipts_do_not_replace_exact_physical_owner_and_basis():
    k2, other = _equal_owner_cases()
    b = k2.b
    assert other.a.identity_sha256 == k2.a.identity_sha256
    source, _, w = _q_data(b, 1)
    with pytest.raises((ValueError, RuntimeError), match="owner|state|context|source|frame"):
        _plan(k2, source, w, frame_a=other.a)
    with pytest.raises((ValueError, RuntimeError), match="owner|state|context|source"):
        _build(k2, source, w, hf=other.b.hf)
    with pytest.raises((ValueError, RuntimeError), match="basis|content|identity"):
        _build(k2, source, w, ao=b.auxiliary)
    wrong_source, _, wrong_w = _q_data(b, 0)
    with pytest.raises((ValueError, RuntimeError), match="source|whitener|identity|q"):
        _plan(k2, source, wrong_w)
    assert wrong_source.q_index != source.q_index


@pytest.mark.parametrize("role", ["ao", "auxiliary"])
def test_grown_basis_is_census_capped_before_nan_primitive_payload(k2, role):
    # Same AO/aux function count, but more primitives than the immutable HF
    # source census. Counting must fail before the NaN coefficient scan.
    grown = _raw_basis([(0, p, [exponent]*20, [np.nan]*20, True)
        for p in ((0, 0, 0), (3, 0, 0)) for exponent in (.6, 1.3)])
    source, _, w = _q_data(k2.b, 1)
    with pytest.raises((ValueError, OverflowError), match="cap|census|exceed"):
        _build(k2, source, w, **{role: grown})


def test_result_is_immutable_owned_data_without_retaining_mutable_frame_views():
    case = _mixed_case(2)
    source, _, w = _q_data(case.b, 1)
    result = _build(case, source, w)
    expected, identity = result.tensor_copy(), result.identity_sha256
    del case, source, w
    gc.collect()
    np.testing.assert_array_equal(result.tensor_copy(), expected)
    assert not result.tensor_copy().flags.writeable
    assert result.identity_sha256 == identity
    assert not hasattr(result, "frame_a") and not hasattr(result, "frame_b")


@pytest.fixture(scope="module", params=[(2, False), (3, False), (2, True)])
def local_geometry_case(request):
    nk, frozen = request.param
    return _mixed_case(nk, frozen=frozen)


def _geometry_view(frame, embedding):
    return core._PeriodicGaussianPairPNOGeometryView(frame.pair_domain, frame.pair_real_space, embedding)


def _local_pno_columns(b, frame, pno, kpoint):
    # The direct pair-space route uses the TRUE retained generation D,
    # never a reconstructed X.T@C or the exported common coefficients.
    pair = SimpleNamespace(reference=b.reference, domain=frame.pair_domain,
        space=frame.pair_real_space.space)
    d = pno.generation_coefficients_copy()
    assert not d.flags.writeable
    return _virtual_columns(pair, kpoint, frame.pair_selected)@d


def _local_coefficients(case, kpoint):
    occupied = _occupied_columns(case.b, kpoint, case.b.rows)
    return np.column_stack((occupied[:, case.i], occupied[:, case.k],
        _local_pno_columns(case.b, case.geometry_a, case.a, kpoint),
        _local_pno_columns(case.b, case.geometry_b, case.source, kpoint)))


def _dense_local_panel(case, source, whitener):
    b, nk, n = case.b, case.b.context.n_kpoints, case.b.ao.nbasis
    m = 2+case.a.generation_coefficients_copy().shape[1]+case.source.generation_coefficients_copy().shape[1]
    result = np.zeros((b.auxiliary.nbasis, m, m), complex)
    for bra in range(nk):
        ket = b.context.ket_index(bra, source.q_index)
        left, right = _local_coefficients(case, bra), _local_coefficients(case, ket)
        ao = _physical_tile(b, source, whitener, bra).matrix.reshape(-1, n, n)
        result += np.einsum("ml,nr,pmn->plr", left.conj(), right, ao)
    return result/(nk*np.sqrt(nk))


def _geometry_payload_bytes(frame, embedding):
    domain, space = frame.pair_domain, frame.pair_real_space.space
    d, s = domain.domain_dimension, space.retained_dimension
    return 16*d+32*d*d+16*d*s+8*(d+s)+embedding.memory.output_numerical_bytes


def test_local_generation_geometry_matches_common_export_and_actual_ao_panels(local_geometry_case):
    case, b = local_geometry_case, local_geometry_case.b
    ga, gb = _geometry_view(case.geometry_a, case.ea), _geometry_view(case.geometry_b, case.eb)
    for kpoint in range(b.context.n_kpoints):
        columns = _local_coefficients(case, kpoint)
        np.testing.assert_allclose(columns, _coefficients(case, kpoint), atol=5e-11, rtol=3e-11)
        occupied_mask = np.asarray(b.reference.state.frozen_core_mask(kpoint), bool)
        occupied_mask |= np.asarray(b.reference.state.correlated_occupied_mask(kpoint), bool)
        occupied = b.reference.state.coefficients(kpoint)[:, occupied_mask]
        # Both frozen and correlated occupied orbitals are excluded by Q.
        np.testing.assert_allclose(occupied.conj().T@b.reference.state.overlap(kpoint)@columns[:, 2:],
            0, atol=5e-11, rtol=3e-11)
    for q in range(b.context.n_kpoints):
        source, _, w = _q_data(b, q)
        local = _build(case, source, w, geometry_a=ga, geometry_b=gb)
        common = _build(case, source, w)
        np.testing.assert_allclose(local.tensor_copy(), _dense_local_panel(case, source, w), atol=4e-12, rtol=3e-11)
        np.testing.assert_allclose(local.tensor_copy(), _dense_mixed_panel(case, source, w), atol=4e-12, rtol=3e-11)
        np.testing.assert_allclose(local.tensor_copy(), common.tensor_copy(), atol=4e-12, rtol=3e-11)
        assert local.memory.local_geometry_a and local.memory.local_geometry_b
        assert not common.memory.local_geometry_a and not common.memory.local_geometry_b
        # Current admission conservatively reserves two full common-sized
        # traversals. It is a work ceiling, not a speedup/timing claim.
        assert local.memory.coefficient_work_units == 2*common.memory.coefficient_work_units
        assert local.memory.borrowed_geometry_numerical_bytes == (
            _geometry_payload_bytes(case.geometry_a, case.ea)+_geometry_payload_bytes(case.geometry_b, case.eb))
        assert local.memory.geometry_validation_work_units > 0
        assert local.identity_sha256 != common.identity_sha256
        assert local.consumed_tiles_identity_sha256 == common.consumed_tiles_identity_sha256
        assert local.frame_a_identity_sha256 == common.frame_a_identity_sha256
        assert local.source_identity_sha256 == common.source_identity_sha256


@pytest.mark.parametrize("local_side", ["a", "b"])
def test_optional_local_geometry_can_mix_with_an_unchanged_legacy_frame(k2, local_side):
    source, _, w = _q_data(k2.b, 1)
    legacy = _legacy(k2.b, k2.provider)
    if local_side == "a":
        frames = dict(frame_a=k2.a, frame_b=legacy)
        geometry = dict(geometry_a=_geometry_view(k2.geometry_a, k2.ea))
    else:
        frames = dict(frame_a=legacy, frame_b=k2.source)
        geometry = dict(geometry_b=_geometry_view(k2.geometry_b, k2.eb))
    result = _build(k2, source, w, **frames, **geometry)
    np.testing.assert_allclose(result.tensor_copy(), _dense_mixed_panel(k2, source, w,
        a=frames["frame_a"], other=frames["frame_b"]), atol=4e-12, rtol=3e-11)
    assert result.memory.local_geometry_a == (local_side == "a")
    assert result.memory.local_geometry_b == (local_side == "b")


def test_local_geometry_payload_is_counted_once_by_owner_not_view_or_digest(k2):
    source, _, w = _q_data(k2.b, 0)
    first = _geometry_view(k2.geometry_a, k2.ea)
    alias = _geometry_view(k2.geometry_a, k2.ea)
    one = _plan(k2, source, w, frame_b=k2.a, geometry_a=first)
    both = _plan(k2, source, w, frame_b=k2.a, geometry_a=first, geometry_b=alias)
    assert one.borrowed_geometry_numerical_bytes == both.borrowed_geometry_numerical_bytes
    assert both.borrowed_geometry_numerical_bytes == _geometry_payload_bytes(k2.geometry_a, k2.ea)
    # Independently owned arrays with equal native receipts must not be
    # treated as aliases. Scalar local generation is unchanged here.
    embedding, frame = _frame(k2.b, columns=[[0, 0], [1, 2]])
    assert embedding.identity_sha256 == k2.ea.identity_sha256
    distinct = _plan(k2, source, w, frame_b=k2.a, geometry_a=first,
        geometry_b=_geometry_view(frame, embedding))
    assert distinct.borrowed_geometry_numerical_bytes == 2*both.borrowed_geometry_numerical_bytes
    full_embedding, full_frame = _frame(k2.b, full=True)
    full_pno = _embedded(k2.b, k2.provider, full_embedding)
    full = _geometry_view(full_frame, full_embedding)
    shared_common = _plan(k2, source, w, frame_a=full_pno, frame_b=full_pno,
        geometry_a=full, geometry_b=full)
    # These exact domain/compact-space allocations already occur in the
    # common baseline. Only the additionally retained embedding is new.
    assert shared_common.borrowed_geometry_numerical_bytes == full_embedding.memory.output_numerical_bytes


@pytest.mark.parametrize("field,plan_field", list(_RESOURCE_CAPS.items())+list(_DIRECT_CAPS.items()))
def test_local_geometry_exact_cap_minus_one_precedes_gauge_payload_scan(k2, field, plan_field):
    source, _, w = _q_data(k2.b, 0)
    geometry = dict(geometry_a=_geometry_view(k2.geometry_a, k2.ea),
        geometry_b=_geometry_view(k2.geometry_b, k2.eb))
    config, live, caps = _controls(k2.b)
    p = _plan(k2, source, w, controls=(config, live, caps), **geometry)
    if field in _RESOURCE_CAPS:
        resources = caps.resources
        setattr(resources, field, getattr(p, plan_field)-1)
        caps.resources = resources
    else:
        setattr(caps, field, getattr(p, plan_field)-1)
    with pytest.raises((ValueError, OverflowError), match="cap|exceed"):
        _build(k2, source, w, controls=(config, live, caps),
            gauge=np.full_like(k2.b.gauge, np.nan), **geometry)


def test_local_geometry_exact_outer_caps_admit(k2):
    source, _, w = _q_data(k2.b, 1)
    geometry = dict(geometry_a=_geometry_view(k2.geometry_a, k2.ea),
        geometry_b=_geometry_view(k2.geometry_b, k2.eb))
    config, live, caps = _controls(k2.b)
    p = _plan(k2, source, w, controls=(config, live, caps), **geometry)
    resources = caps.resources
    for field, plan_field in _RESOURCE_CAPS.items():
        setattr(resources, field, getattr(p, plan_field))
    caps.resources = resources
    for field, plan_field in _DIRECT_CAPS.items():
        setattr(caps, field, getattr(p, plan_field))
    result = _build(k2, source, w, controls=(config, live, caps), **geometry)
    np.testing.assert_allclose(result.tensor_copy(), _dense_local_panel(k2, source, w), atol=4e-12, rtol=3e-11)


def test_zero_rank_local_geometry_preserves_validation_without_forcing_columns(k2):
    high = 1+max(k2.a.original_pno_occupations_copy().max(), k2.source.original_pno_occupations_copy().max())
    a = _embedded(k2.b, k2.provider, k2.ea, options=_embedded_options(cutoff=high))
    b = _embedded(k2.b, k2.provider, k2.eb, 0, 2, options=_embedded_options(cutoff=high))
    source, _, w = _q_data(k2.b, 1)
    result = _build(k2, source, w, frame_a=a, frame_b=b,
        geometry_a=_geometry_view(k2.geometry_a, k2.ea), geometry_b=_geometry_view(k2.geometry_b, k2.eb))
    assert a.generation_coefficients_copy().shape == (2, 0)
    assert b.generation_coefficients_copy().shape == (3, 0)
    assert result.memory.coefficient_helper_workspace_bytes == 0
    assert result.memory.borrowed_geometry_numerical_bytes > 0
    np.testing.assert_allclose(result.tensor_copy(), _dense_mixed_panel(k2, source, w, a=a, other=b), atol=4e-12)


def test_geometry_view_is_not_a_certificate_for_mismatched_or_legacy_frames(k2):
    source, _, w = _q_data(k2.b, 1)
    wrong = _geometry_view(k2.geometry_b, k2.eb)
    with pytest.raises((ValueError, RuntimeError), match="geometry|embedding|frame|generation"):
        _plan(k2, source, w, geometry_a=wrong)
    translated, frame = _frame(k2.b, columns=[[0, 0], [1, 2]], translation=1)
    with pytest.raises((ValueError, RuntimeError), match="geometry|embedding|frame|generation"):
        _plan(k2, source, w, geometry_a=_geometry_view(frame, translated))
    # Correct embedding label with unrelated actual pair geometry passes
    # the cheap shape census but must fail the admitted native frame replay.
    false_geometry = core._PeriodicGaussianPairPNOGeometryView(
        k2.geometry_b.pair_domain, k2.geometry_b.pair_real_space, k2.ea)
    with pytest.raises((ValueError, RuntimeError), match="geometry|authenticated.*frame"):
        _build(k2, source, w, geometry_a=false_geometry)
    legacy = _legacy(k2.b, k2.provider)
    with pytest.raises((ValueError, RuntimeError), match="geometry|embedded|legacy|frame"):
        _plan(k2, source, w, frame_a=legacy, geometry_a=_geometry_view(k2.geometry_a, k2.ea))


def test_view_rejects_consumed_geometry_while_original_common_owners_remain_live():
    from tests.test_periodic_gaussian_pair_domain_builder import _build as build_domain_owner
    from tests.test_periodic_gaussian_occupied_pao_domain import _controls as occupied_controls
    case = _mixed_case(2)
    embedding, frame = _frame(case.b, columns=case.b.columns.tolist())
    pno = _embedded(case.b, case.provider, embedding)
    view = _geometry_view(frame, embedding)
    source, _, w = _q_data(case.b, 0)
    _plan(case, source, w, frame_a=pno, geometry_a=view)
    moving = SimpleNamespace(**vars(case.b))
    moving.domain, moving.real_space = frame.pair_domain, frame.pair_real_space
    moving.space = moving.real_space.space
    moving.domain_options = occupied_controls(moving, full=True)[0]
    owner = build_domain_owner(moving)  # Existing authenticated move seam.
    assert owner.memory.n_home_occupied > 0 and case.b.domain.domain_dimension > 0
    with pytest.raises((ValueError, RuntimeError), match="moved|consumed|owner|state|space|domain"):
        _plan(case, source, w, frame_a=pno, geometry_a=view)


def test_geometry_python_view_pins_its_three_owners_and_result_is_detached():
    case = _mixed_case(2)
    embedding, frame = _frame(case.b, columns=case.b.columns.tolist())
    pno = _embedded(case.b, case.provider, embedding)
    view = _geometry_view(frame, embedding)
    del embedding, frame
    gc.collect()
    source, _, w = _q_data(case.b, 1)
    result = _build(case, source, w, frame_a=pno, geometry_a=view)
    expected = _dense_mixed_panel(case, source, w, a=pno)
    del case, pno, view, source, w
    gc.collect()
    np.testing.assert_allclose(result.tensor_copy(), expected, atol=4e-12, rtol=3e-11)


def _direct_gram_case(nk=2, *, kind="he2", frozen=False, cutoff=0.0):
    # Lazy imports keep the existing Gram/PNO oracle helper dependency acyclic.
    from tests.test_periodic_gaussian_gram_pair_pnos import (
        _case, _geometry, _make, _controls as gram_controls,
    )
    b = _case(nk, kind=kind, frozen=frozen, full=kind == "he")
    k = (1 if frozen or kind == "he" else 2) if nk > 1 else 1
    k = min(k, b.basis.memory.occupied_count-1)
    ga, gb = _geometry(b), _geometry(b, 0, k)
    a = _make(b, ga, gram_controls(b, cutoff=cutoff))
    other = _make(b, gb, gram_controls(b, cutoff=cutoff))
    assert not hasattr(b, "provider")
    return SimpleNamespace(b=b, a=a, source=other, i=0, k=k, ga=ga, gb=gb)


def _direct_gram_columns(case, kpoint):
    from tests.test_periodic_gaussian_gram_pair_pnos import _local_columns
    b = case.b
    occupied = _occupied_columns(b, kpoint, b.rows)
    virtuals = []
    for geometry, frame in ((case.ga, case.a), (case.gb, case.source)):
        qocc = 1 if geometry.occupied_slot_i == geometry.occupied_slot_j else 2
        original = _local_columns(b, geometry, kpoint)[:, qocc:]
        virtuals.append(original @ frame.generation_coefficients_copy())
    return np.column_stack((occupied[:, case.i], occupied[:, case.k], *virtuals))


def _direct_gram_panel_oracle(case, source, whitener):
    b, nk, n = case.b, case.b.context.n_kpoints, case.b.ao.nbasis
    columns = [_direct_gram_columns(case, k) for k in range(nk)]
    m = columns[0].shape[1]
    out = np.zeros((b.auxiliary.nbasis, m, m), complex)
    for bra in range(nk):
        ket = b.context.ket_index(bra, source.q_index)
        ao = _physical_tile(b, source, whitener, bra).matrix.reshape(-1, n, n)
        out += np.einsum("ml,nr,pmn->plr", columns[bra].conj(), columns[ket], ao)
    return out/(nk*np.sqrt(nk))


@pytest.mark.parametrize("nk,kind,frozen", [
    (1, "he", False), (2, "he", False), (3, "he", False),
    (1, "he2", False), (2, "he2", False), (2, "he2", True),
])
def test_direct_gram_frames_feed_actual_mixed_factors_without_prior_provider(nk, kind, frozen):
    case = _direct_gram_case(nk, kind=kind, frozen=frozen)
    geometry = dict(geometry_a=case.ga.geometry_view(), geometry_b=case.gb.geometry_view())
    for q in range(nk):
        source, _, w = _q_data(case.b, q)
        result = _build(case, source, w, **geometry)
        np.testing.assert_allclose(result.tensor_copy(), _direct_gram_panel_oracle(case, source, w),
            atol=4e-12, rtol=3e-11)
        # Independent compatibility export check, not a factor-provider input.
        np.testing.assert_allclose(result.tensor_copy(), _dense_mixed_panel(case, source, w),
            atol=4e-12, rtol=3e-11)
        assert result.direct_gram_frames and result.memory.direct_gram_frames
        assert result.frame_a_gram_identity_sha256 == case.a.gram_identity_sha256
        assert result.frame_b_gram_identity_sha256 == case.source.gram_identity_sha256
        assert result.frame_a_identity_sha256 == case.a.identity_sha256
        assert result.frame_b_identity_sha256 == case.source.identity_sha256
        assert result.memory.borrowed_frame_numerical_bytes == (
            case.a.retained_numerical_bytes+case.source.retained_numerical_bytes)
        assert not result.jointly_orthonormal_basis_certified
        assert not result.density_symmetry_certified and not result.production_dlpno
        for name in ("frame_a_provider_identity_sha256", "frame_b_provider_identity_sha256"):
            with pytest.raises((ValueError, RuntimeError), match="Gram.*provider|provider.*receipt"):
                getattr(result, name)
    assert not hasattr(case.b, "provider")


@pytest.mark.parametrize("field,plan_field", [
    ("maximum_owned_numeric_bytes", "peak_owned_numerical_bytes"),
    ("maximum_per_replica_inventoried_bytes", "per_replica_inventoried_bytes"),
    ("maximum_node_inventoried_bytes", "required_node_memory_bytes"),
    ("maximum_work_units", "work_units"),
])
def test_direct_gram_mixed_caps_precede_nonfinite_gauges(field, plan_field):
    case = _direct_gram_case(1)
    source, _, w = _q_data(case.b, 0)
    geometry = dict(geometry_a=case.ga.geometry_view(), geometry_b=case.gb.geometry_view())
    config, live, caps = _controls(case.b)
    plan = _plan(case, source, w, controls=(config, live, caps), **geometry)
    resources = caps.resources
    setattr(resources, field, getattr(plan, plan_field)-1)
    caps.resources = resources
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap"):
        _build(case, source, w, controls=(config, live, caps),
            gauge=np.full_like(case.b.gauge, np.nan), **geometry)


def test_direct_gram_mixed_requires_both_original_geometries_and_source_kind():
    case = _direct_gram_case()
    source, _, w = _q_data(case.b, 0)
    ga, gb = case.ga.geometry_view(), case.gb.geometry_view()
    for geometry in ({}, dict(geometry_a=ga), dict(geometry_b=gb)):
        with pytest.raises((ValueError, RuntimeError), match="Gram.*geometry"):
            _plan(case, source, w, **geometry)
    with pytest.raises((ValueError, RuntimeError), match="geometry|generation"):
        _plan(case, source, w, geometry_a=gb, geometry_b=ga)
    # Old-source owners exist only in this explicit incompatibility witness.
    from tests.test_periodic_gaussian_gram_pair_pnos import _oracle_inputs, _old_embedding, _old_pnos
    old = _old_pnos(case.b, _oracle_inputs(case.b).provider, _old_embedding(case.b))
    with pytest.raises((ValueError, RuntimeError), match="source kinds"):
        _plan(case, source, w, frame_b=old, geometry_a=ga, geometry_b=gb)


def test_direct_gram_mixed_wrapper_does_not_duplicate_seed_exchange_array():
    from tests.test_periodic_gaussian_gram_pair_pnos import _space
    wrapper_deltas = []
    for cutoff in (0.0, 1.0):
        case = _direct_gram_case(cutoff=cutoff)
        source, _, w = _q_data(case.b, 1)
        geometry = dict(geometry_a=case.ga.geometry_view(), geometry_b=case.gb.geometry_view())
        seed_plan = _plan(case, source, w, **geometry)
        expected = _direct_gram_panel_oracle(case, source, w)
        a, b = _space(case.b, case.a), _space(case.b, case.source)
        result = _build(case, source, w, frame_a=a, frame_b=b, **geometry)
        np.testing.assert_allclose(result.tensor_copy(), expected, atol=4e-12, rtol=3e-11)
        assert result.memory.borrowed_frame_numerical_bytes == seed_plan.borrowed_frame_numerical_bytes
        wrapper_deltas.append(result.memory.live.other_retained_bytes_per_worker-
            seed_plan.live.other_retained_bytes_per_worker)
        assert not hasattr(case.b, "provider")
        if cutoff:
            assert result.rank_a == result.rank_b == 0
            assert result.memory.borrowed_geometry_numerical_bytes > 0
    # Wrapper controls are rank-independent. Counting G again would add a
    # rank-squared term in the first case, despite G already belonging to seed.
    assert wrapper_deltas[0] == wrapper_deltas[1] > 0


def test_direct_gram_mixed_counts_same_owner_once_and_supports_slab_splits():
    case = _direct_gram_case()
    source, _, w = _q_data(case.b, 1)
    geometry = case.ga.geometry_view()
    plan = _plan(case, source, w, frame_b=case.a, geometry_a=geometry, geometry_b=geometry)
    assert plan.same_frame_owner
    assert plan.borrowed_frame_numerical_bytes == case.a.retained_numerical_bytes
    whole = _build(case, source, w, frame_b=case.a, geometry_a=geometry, geometry_b=geometry)
    split = [_build(case, source, w, frame_b=case.a, geometry_a=geometry, geometry_b=geometry,
        begin=a, count=1, controls=_controls(case.b, pair_block=1)).tensor_copy()
        for a in range(case.b.auxiliary.nbasis)]
    np.testing.assert_allclose(np.concatenate(split), whole.tensor_copy(), atol=4e-12, rtol=3e-11)


def test_direct_gram_mixed_positive_pno_cut_and_localization_lineage():
    from tests.test_periodic_correlation_wannier import _make as make_wannier
    probe = _direct_gram_case()
    occupations = probe.a.original_pno_occupations_copy()
    assert len(occupations) == 2 and occupations[0] > occupations[1] >= 0
    case = _direct_gram_case(cutoff=float(np.mean(occupations)))
    source, _, w = _q_data(case.b, 1)
    geometry = dict(geometry_a=case.ga.geometry_view(), geometry_b=case.gb.geometry_view())
    result = _build(case, source, w, **geometry)
    assert result.rank_a == 1
    expected = _direct_gram_panel_oracle(case, source, w)
    np.testing.assert_allclose(result.tensor_copy(), expected, atol=4e-12, rtol=3e-11)
    snapshot = result.tensor_copy()
    # A sign flip remains unitary, real/TR and preserves occupied F. It is
    # nevertheless not the actual localization used by the common basis.
    changed = np.ascontiguousarray(-case.b.gauge)
    case.b.wannier = make_wannier(case.b.reference, changed)
    with pytest.raises((ValueError, RuntimeError), match="basis|coefficients|gauge"):
        _build(case, source, w, gauge=changed, **geometry)
    del case, probe, source, w, geometry
    gc.collect()
    np.testing.assert_array_equal(result.tensor_copy(), snapshot)
