"""Actual-HF-origin finite-source local factors; tiny independent contractions.

The normalized two-s He fixture is an explicit finite Hamiltonian, not a
bulk benchmark. No factor store, caller-labelled SCF state or external QC
runtime enters this route. Dense AO tensors exist only in these tiny oracles.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_rhf import _bundle as _hf_bundle, _run as _run_hf
from tests.test_periodic_gaussian_fock import _q_data, _physical_tile
from tests.test_periodic_gaussian_metric import _live as _metric_live, _CAP_FIELDS
from tests.test_periodic_gaussian_source_context import _CAP_FIELDS as _BASIS_CAP_FIELDS
from tests.test_periodic_correlation_pair_topology import _make_dimensions, _make_budget
from tests.test_periodic_correlation_wannier import _make as _make_wannier
from tests.test_periodic_correlation_pao_domain import _make as _make_domain
from tests.test_periodic_correlation_pao_space import _make as _make_space
from tests.test_periodic_correlation_real_local_basis import _make as _make_real_basis
from tests.test_periodic_correlation_density_factors import _occupied_columns, _virtual_columns, _virtual
from tests.test_periodic_correlation_reciprocal_metric import _CanonicalDigest


_PLAN_FIELDS = (
    "n_cells", "n_basis", "n_auxiliary", "occupied_count", "virtual_count", "orbital_count",
    "q_index", "auxiliary_begin", "auxiliary_count", "ao_pair_block", "tile_calls",
    "retained_factor_bytes", "retained_index_bytes", "retained_output_bytes", "compensation_bytes",
    "coefficient_panel_bytes", "coefficient_scratch_bytes", "driver_owned_numerical_bytes",
    "maximum_tile_owned_numerical_bytes", "peak_owned_numerical_bytes", "resident_whitener_bytes",
    "borrowed_basis_active_numeric_bytes", "caller_gauge_bytes", "live_wannier_bytes", "live_domain_bytes",
    "live_space_bytes", "live_basis_bytes", "basis_index_alias_bytes", "state_resident_bytes",
    "macro_fixed_object_bytes", "tile_fixed_object_bytes", "replicas_per_node", "reference_base_node_bytes",
    "per_replica_inventoried_bytes", "required_node_memory_bytes", "reciprocal_candidate_evaluations",
    "image_candidate_evaluations_upper_bound", "coefficient_work_units", "contraction_term_count",
    "driver_work_units", "work_units",
)
_LIVE_FIELDS = ("other_retained_bytes_per_worker", "other_transient_bytes_per_worker",
                "fixed_backend_margin_bytes_per_worker")
_CAP_TO_PLAN = dict(zip(_CAP_FIELDS, (
    "peak_owned_numerical_bytes", "per_replica_inventoried_bytes", "required_node_memory_bytes",
    "reciprocal_candidate_evaluations", "work_units")))


def _caps(b):
    c = core._PeriodicGaussianLocalOrbitalFactorCaps()
    r = core._PeriodicGaussianMetricCaps()
    for field, value in zip(_CAP_FIELDS, (2**20, 2**24, 2**26, 10**7, 10**14)):
        setattr(r, field, value)
    c.resources = r
    c.tile = b.config.tile_caps
    c.maximum_tile_calls = 128
    c.maximum_image_candidate_evaluations = 10**11
    return c


def _bundle(mesh=(3, 1, 1), *, rows=None, translation=0, pair_block=3):
    b = _hf_bundle(mesh)
    b.hf = _run_hf(b)
    assert b.hf.converged
    dims = _make_dimensions(mesh, 1, calculation_identity=b.hf.state.calculation_identity)
    dims.n_auxiliary = dims.domain_local_auxiliary_upper_bound = b.auxiliary.nbasis
    budget = _make_budget()
    budget.memory_limit_bytes = 2**26
    b.reference = core._make_periodic_correlation_admitted_reference(b.hf.state, dims, budget)
    b.gauge = np.ones((b.context.n_kpoints, 1, 1), dtype=np.complex128)
    b.wannier = _make_wannier(b.reference, b.gauge)
    b.domain = _make_domain(b.reference, np.array([[0, 0], [0, 1]], dtype=np.uint64))
    b.space = _make_space(b.reference, b.domain)
    assert b.space.retained_dimension == 1
    b.rows = np.array([[0, 0]] if rows is None else rows, dtype=np.uint64)
    b.selected = _virtual(0, 1, translation)
    b.basis = _make_real_basis(b, b.rows, b.selected)
    b.panel_config = core._PeriodicGaussianLocalOrbitalFactorConfig()
    b.panel_config.ao_pair_block = pair_block
    b.panel_live = core._PeriodicGaussianLocalOrbitalFactorLiveInventory()
    b.panel_live.fixed_backend_margin_bytes_per_worker = 65536
    b.panel_caps = _caps(b)
    return b


def _plan(b, source, whitener, *, begin=0, count=None, caps=None, live=None, config=None):
    return core._plan_periodic_gaussian_local_orbital_factor_panel(b.hf, b.reference, source, whitener,
        b.wannier, b.domain, b.space, b.basis, begin,
        b.auxiliary.nbasis-begin if count is None else count,
        b.panel_config if config is None else config, b.panel_live if live is None else live,
        b.panel_caps if caps is None else caps)


def _build(b, source, whitener, *, begin=0, count=None, caps=None, live=None, config=None,
           gauge=None, ao=None, auxiliary=None):
    return core._build_periodic_gaussian_local_orbital_factor_panel(b.hf, b.reference, source, whitener,
        b.ao if ao is None else ao, b.auxiliary if auxiliary is None else auxiliary,
        b.wannier, b.gauge if gauge is None else gauge, b.domain, b.space, b.basis, begin,
        b.auxiliary.nbasis-begin if count is None else count,
        b.panel_config if config is None else config, b.panel_live if live is None else live,
        b.panel_caps if caps is None else caps)


def _dense_panel(b, source, whitener, *, begin=0, count=None):
    """NumPy AO-to-selected-local contraction, independent of the new leaf."""
    count = b.auxiliary.nbasis-begin if count is None else count
    nk, n, m = b.context.n_kpoints, b.ao.nbasis, len(b.rows)+b.selected.count
    expected = np.zeros((count, m, m), dtype=complex)
    for bra in range(nk):
        ket = b.context.ket_index(bra, source.q_index)
        left = np.column_stack((_occupied_columns(b, bra, b.rows), _virtual_columns(b, bra, b.selected)))
        right = np.column_stack((_occupied_columns(b, ket, b.rows), _virtual_columns(b, ket, b.selected)))
        tensor = _physical_tile(b, source, whitener, bra, aux=begin, rows=count).matrix.reshape(count, n, n)
        expected += np.einsum("ml,nr,pmn->plr", left.conj(), right, tensor)
    return expected/(nk*np.sqrt(nk))


def _plan_wire(b, source, whitener, p):
    h = _CanonicalDigest("vibeqc.periodic.gaussian-local-orbital-factors.plan", 1)
    for value in (b.hf.reference_source_identity_sha256, source.source_identity_sha256,
                  source.conjugate_source_identity_sha256, whitener.payload_identity_sha256,
                  b.basis.identity_sha256, b.reference.dimensions.allocation_identity,
                  b.wannier.wannier_identity_sha256, b.domain.pao_domain_identity_sha256,
                  b.space.pao_space_identity_sha256):
        h.string(value)
    for field in _PLAN_FIELDS:
        h.u64(getattr(p, field))
    h.u64(p.config.ao_pair_block)
    for field in _LIVE_FIELDS:
        h.u64(getattr(p.live, field))
    for caps in (p.caps.resources, p.caps.tile.resources):
        for field in _CAP_FIELDS:
            h.u64(getattr(caps, field))
    for value in (p.caps.tile.maximum_image_candidates, p.caps.maximum_tile_calls,
                  p.caps.maximum_image_candidate_evaluations, p.tile_config.reciprocal_block):
        h.u64(value)
    for field in _BASIS_CAP_FIELDS:
        h.u64(getattr(p.tile_config.basis_verification_caps, field))
    return h.finish()


def _payload_wire(result):
    h = _CanonicalDigest("vibeqc.periodic.gaussian-local-orbital-factors.payload", 1)
    for value in (result.q_index, result.auxiliary_begin, result.auxiliary_count, result.orbital_count):
        h.u64(value)
    for value in result.tensor_copy().ravel():
        h.complex128(value)
    return h.finish()


def _consumed_wire(b, source, whitener, result):
    h = _CanonicalDigest("vibeqc.periodic.gaussian-local-orbital-factors.consumed-tiles", 1)
    for value in (b.hf.reference_source_identity_sha256, source.source_identity_sha256,
                  source.conjugate_source_identity_sha256, whitener.payload_identity_sha256):
        h.string(value)
    p, n = result.memory, b.ao.nbasis
    h.u64(p.tile_calls)
    visit = 0
    for k in range(b.context.n_kpoints):
        for pair in range(0, n*n, p.ao_pair_block):
            tile = _physical_tile(b, source, whitener, k, pair//n, pair % n,
                min(p.ao_pair_block, n*n-pair), p.auxiliary_begin, p.auxiliary_count)
            h.u64(visit)
            h.string(tile.payload_identity_sha256)
            visit += 1
    return h.finish()


@pytest.mark.parametrize("mesh,q", [((1, 1, 1), 0), ((2, 1, 1), 0), ((2, 1, 1), 1),
                                   ((3, 1, 1), 0), ((3, 1, 1), 1), ((3, 1, 1), 2)])
def test_all_ordered_oo_ov_vo_vv_match_actual_gaussian_dense_oracle(mesh, q):
    b = _bundle(mesh)
    source, _, w = _q_data(b, q)
    result = _build(b, source, w)
    expected = _dense_panel(b, source, w)
    np.testing.assert_allclose(result.tensor_copy(), expected, atol=4e-12, rtol=3e-11)
    assert np.linalg.norm(expected[:, 0, 0]) > 1e-4
    assert np.linalg.norm(expected[:, 0, 1]) > 1e-4
    assert np.linalg.norm(expected[:, 1, 0]) > 1e-4
    assert np.linalg.norm(expected[:, 1, 1]) > 1e-4
    assert result.context is b.context and result.state is b.hf.state
    assert result.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256
    assert result.local_basis_identity_sha256 == b.basis.local_basis_identity_sha256
    assert result.basis_certificate_identity_sha256 == b.basis.identity_sha256
    assert result.source_identity_sha256 == source.source_identity_sha256
    assert result.conjugate_source_identity_sha256 == source.conjugate_source_identity_sha256
    assert result.whitener_payload_identity_sha256 == w.payload_identity_sha256
    assert result.occupied(0) == (0, 0)
    assert result.finite_image_reference and result.matched_finite_gaussian_hf_recipe
    assert not result.ao_image_source_certified and not result.density_symmetry_certified
    assert not result.bitwise_hf_factor_consumption_verified
    assert not hasattr(result, "store_identity_sha256")
    assert result.payload_sha256 == _payload_wire(result)
    assert result.memory.plan_identity_sha256 == _plan_wire(b, source, w, result.memory)


def test_translated_real_orbitals_keep_complex_sine_lane_and_nk_normalization():
    b = _bundle((3, 1, 1), rows=[[0, 1]], translation=1)
    source, _, w = _q_data(b, 1)
    translated = _build(b, source, w)
    origin = SimpleNamespace(**vars(b))
    origin.rows, origin.selected = np.array([[0, 0]], np.uint64), _virtual()
    origin.basis = _make_real_basis(origin, origin.rows, origin.selected)
    original = _build(origin, source, w)
    phase = np.exp(-2j*np.pi/3)
    np.testing.assert_allclose(translated.tensor_copy(), phase*original.tensor_copy(), atol=5e-12, rtol=3e-11)
    np.testing.assert_allclose(translated.tensor_copy(), _dense_panel(b, source, w), atol=5e-12, rtol=3e-11)
    assert np.max(abs(translated.tensor_copy().imag)) > 1e-3
    assert translated.local_basis_identity_sha256 != original.local_basis_identity_sha256


@pytest.mark.parametrize("pair_block", [1, 2, 3, 4])
def test_ragged_ao_blocks_auxiliary_slices_and_exact_consumed_tile_order(pair_block):
    b = _bundle((2, 1, 1), rows=[[0, 0], [0, 1]], pair_block=pair_block)
    source, _, w = _q_data(b, 1)
    whole = _build(b, source, w)
    slices = [_build(b, source, w, begin=p, count=1).tensor_copy() for p in range(2)]
    np.testing.assert_allclose(np.concatenate(slices), whole.tensor_copy(), atol=4e-13, rtol=3e-12)
    np.testing.assert_allclose(whole.tensor_copy(), _dense_panel(b, source, w), atol=5e-12, rtol=3e-11)
    assert whole.consumed_tiles_identity_sha256 == _consumed_wire(b, source, w, whole)
    reordered = SimpleNamespace(**vars(b))
    reordered.rows = b.rows[::-1].copy()
    reordered.basis = _make_real_basis(reordered, reordered.rows, reordered.selected)
    permutation = [1, 0, 2]
    np.testing.assert_array_equal(_build(reordered, source, w).tensor_copy(),
        whole.tensor_copy()[:, permutation][:, :, permutation])


def test_inventory_counts_one_w_one_state_and_alias_indices_only_once():
    b = _bundle((3, 1, 1), rows=[[0, 0], [0, 1]])
    source, _, w = _q_data(b, 1)
    p = _plan(b, source, w)
    n, m, o, v, a, ab, nk = 2, 3, 2, 1, 2, 2, 3
    assert p.retained_factor_bytes == 16*ab*m*m
    assert p.retained_output_bytes == 16*ab*m*m+16*o
    assert p.compensation_bytes == p.retained_factor_bytes
    assert p.coefficient_panel_bytes == 32*n*m
    assert p.coefficient_scratch_bytes == 32*n
    assert p.driver_owned_numerical_bytes == 32*ab*m*m+32*n*m+32*n+16*o
    assert p.maximum_tile_owned_numerical_bytes == b.panel_caps.tile.resources.maximum_owned_numeric_bytes
    assert p.peak_owned_numerical_bytes == p.driver_owned_numerical_bytes+p.maximum_tile_owned_numerical_bytes
    assert p.resident_whitener_bytes == 16*a*a
    assert p.live_basis_bytes == 16*o+8*(o*o+v*v+o*v)
    assert p.basis_index_alias_bytes == 16*o
    assert p.state_resident_bytes == b.reference.state_resident_bytes
    assert p.tile_calls == nk*2
    assert p.contraction_term_count == nk*n*n*ab*m*m
    assert p.reciprocal_candidate_evaluations == 2*source.candidate_count*p.tile_calls
    assert p.required_node_memory_bytes == p.reference_base_node_bytes+p.replicas_per_node*p.per_replica_inventoried_bytes
    expected = sum(getattr(p, field) for field in (
        "peak_owned_numerical_bytes", "resident_whitener_bytes", "borrowed_basis_active_numeric_bytes",
        "caller_gauge_bytes", "live_wannier_bytes", "live_domain_bytes", "live_space_bytes", "live_basis_bytes",
        "macro_fixed_object_bytes", "tile_fixed_object_bytes"))
    expected += sum(getattr(p.live, field) for field in _LIVE_FIELDS)
    assert p.per_replica_inventoried_bytes == expected
    result = _build(b, source, w)
    d = result.diagnostics
    assert d.completed_tile_calls == p.tile_calls
    assert d.reciprocal_candidate_evaluations == p.reciprocal_candidate_evaluations
    assert 0 < d.image_candidate_evaluations <= p.image_candidate_evaluations_upper_bound
    assert 0 < d.charged_work_units_upper_bound <= p.work_units
    assert 0 < d.maximum_observed_owned_numerical_bytes <= p.peak_owned_numerical_bytes
    assert 0 < d.maximum_observed_per_replica_inventoried_bytes <= p.per_replica_inventoried_bytes
    extra = core._PeriodicGaussianLocalOrbitalFactorLiveInventory()
    extra.other_retained_bytes_per_worker = 8192
    extra.other_transient_bytes_per_worker = 4096
    extra.fixed_backend_margin_bytes_per_worker = b.panel_live.fixed_backend_margin_bytes_per_worker
    expanded = _plan(b, source, w, live=extra)
    assert expanded.per_replica_inventoried_bytes == p.per_replica_inventoried_bytes+12288
    assert expanded.required_node_memory_bytes == p.required_node_memory_bytes+12288*p.replicas_per_node
    assert expanded.plan_identity_sha256 != p.plan_identity_sha256


def _exact_caps(b, p):
    caps = _caps(b)
    resources = caps.resources
    for field, target in _CAP_TO_PLAN.items():
        setattr(resources, field, getattr(p, target))
    caps.resources = resources
    caps.maximum_tile_calls = p.tile_calls
    caps.maximum_image_candidate_evaluations = p.image_candidate_evaluations_upper_bound
    return caps


@pytest.mark.parametrize("field", [*_CAP_TO_PLAN, "maximum_tile_calls", "maximum_image_candidate_evaluations"])
def test_exact_caps_and_one_less_precede_gauge_scan(field):
    b = _bundle((1, 1, 1))
    source, _, w = _q_data(b, 0)
    p = _plan(b, source, w)
    caps = _exact_caps(b, p)
    _build(b, source, w, caps=caps)
    if field in _CAP_TO_PLAN:
        resources = caps.resources
        setattr(resources, field, getattr(resources, field)-1)
        caps.resources = resources
    else:
        setattr(caps, field, getattr(caps, field)-1)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _build(b, source, w, caps=caps, gauge=np.full_like(b.gauge, np.nan))


@pytest.mark.parametrize("field", _CAP_FIELDS)
def test_zero_resource_caps_fail_closed(field):
    b = _bundle((1, 1, 1))
    source, _, w = _q_data(b, 0)
    caps = _caps(b)
    resources = caps.resources
    setattr(resources, field, 0)
    caps.resources = resources
    with pytest.raises(ValueError, match="positive explicit caps"):
        _plan(b, source, w, caps=caps)


def test_actual_source_context_and_hf_state_owners_cannot_be_forged_by_equal_labels():
    b, other = _bundle((1, 1, 1)), _bundle((1, 1, 1))
    source, _, w = _q_data(b, 0)
    foreign_source, _, foreign_w = _q_data(other, 0)
    assert b.context.source_context_identity_sha256 == other.context.source_context_identity_sha256
    assert b.hf.state.calculation_identity == other.hf.state.calculation_identity
    with pytest.raises(ValueError, match="context differs from actual HF"):
        _plan(b, foreign_source, foreign_w)
    with pytest.raises(ValueError, match="context differs from actual HF"):
        _plan(b, source, foreign_w)
    different_hf = SimpleNamespace(**vars(b))
    different_hf.hf = other.hf
    with pytest.raises(ValueError, match="owners disagree"):
        _plan(different_hf, source, w)
    with pytest.raises(TypeError):
        core._PeriodicGaussianLocalOrbitalFactorPanel()


def test_q_and_whitening_recipe_mismatch_reject_before_actual_basis_work():
    b = _bundle((2, 1, 1))
    source, _, w = _q_data(b, 0)
    _, _, foreign_q_w = _q_data(b, 1)
    with pytest.raises(ValueError, match="recipe or shape"):
        _build(b, source, foreign_q_w, ao=b.auxiliary)
    config = b.config.metric
    config.reciprocal_block = 1
    raw = core._build_periodic_gaussian_reciprocal_metric(source, b.ao, b.auxiliary,
        config, _metric_live(), b.config.metric_caps)
    changed = core._factorize_periodic_gaussian_metric(raw, b.auxiliary.nbasis, _metric_live(), b.config.metric_caps)
    with pytest.raises(ValueError, match="recipe or shape"):
        _plan(b, source, changed)


def test_orbital_basis_certificate_and_actual_basis_payloads_are_verified():
    b = _bundle((1, 1, 1))
    source, _, w = _q_data(b, 0)
    with pytest.raises((ValueError, RuntimeError), match="basis.*(identity|content|digest)"):
        _build(b, source, w, ao=b.auxiliary)
    with pytest.raises((ValueError, RuntimeError), match="gauge"):
        _build(b, source, w, gauge=-b.gauge)
    changed = SimpleNamespace(**vars(b))
    changed.gauge = -b.gauge
    changed.wannier = _make_wannier(b.reference, changed.gauge)
    with pytest.raises(ValueError, match="basis certificate"):
        _build(changed, source, w)


@pytest.mark.parametrize("begin,count,block", [(2, 1, 3), (0, 0, 3), (1, 2, 3), (0, 2, 0), (0, 2, 5)])
def test_invalid_auxiliary_slices_and_ao_block_fail_closed(begin, count, block):
    b = _bundle((1, 1, 1))
    source, _, w = _q_data(b, 0)
    config = core._PeriodicGaussianLocalOrbitalFactorConfig()
    config.ao_pair_block = block
    with pytest.raises(ValueError, match="slice or AO block"):
        _plan(b, source, w, begin=begin, count=count, config=config)


def test_no_implicit_gauge_conversion_and_detached_panel_lifetime():
    b = _bundle((3, 1, 1), rows=[[0, 1]], translation=1)
    source, _, w = _q_data(b, 1)
    with pytest.raises(ValueError, match="contiguous complex128"):
        _build(b, source, w, gauge=b.gauge.real)
    with pytest.raises(ValueError, match="contiguous complex128"):
        _build(b, source, w, gauge=b.gauge[::-1])
    result = _build(b, source, w)
    expected, identity = result.tensor_copy(), result.identity_sha256
    result.tensor_copy()[:] = 17
    copied_selection = result.virtual_selection
    copied_selection.translation_cell = 0
    copied_plan = result.memory
    copied_config = copied_plan.config
    copied_config.ao_pair_block = 1
    del b, source, w
    gc.collect()
    np.testing.assert_array_equal(result.tensor_copy(), expected)
    assert result.identity_sha256 == identity
    assert result.virtual_selection.translation_cell == 1
    assert result.memory.config.ao_pair_block == 3
    assert result.element(0, 0, 1) == expected[0, 0, 1]
    with pytest.raises(IndexError):
        result.element(2, 0, 0)
    with pytest.raises(IndexError):
        result.occupied(1)
