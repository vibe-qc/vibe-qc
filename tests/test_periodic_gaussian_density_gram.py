"""Actual-source selected Gram blocks without a prior full-row provider.

Independent NumPy AO transforms check both the projected real-row Gram and
the unprojected complex Coulomb orientation. Finite-source projection and
summation bounds do not certify an infinite lattice or a production method.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_selected_local_ccsd_t import (
    _bundle, _prepare_leaves, _he2_controls,
)
from tests.test_periodic_gaussian_frozen_core_correlation import _frozen_bundle
from tests.test_periodic_gaussian_local_orbital_factors import _dense_panel
from tests.test_periodic_gaussian_mixed_pair_factors import _q_data
from tests.test_periodic_correlation_real_local_provider import _expected_rows, _options
from tests.test_periodic_gaussian_real_local_provider import _make as _full_provider
from tests.test_periodic_aopair_fourier_panel import _basis as _raw_basis


_RESOURCE_CAPS = dict(maximum_owned_numeric_bytes="peak_owned_numerical_bytes",
    maximum_per_replica_inventoried_bytes="per_worker_inventoried_bytes",
    maximum_node_inventoried_bytes="required_node_memory_bytes",
    maximum_candidate_evaluations="reciprocal_candidate_evaluations_upper_bound",
    maximum_work_units="work_units")
_DIRECT_CAPS = dict(maximum_left_density_count="left", maximum_right_density_count="right",
    maximum_output_elements="output_elements", maximum_factor_panels="factor_panels",
    maximum_tile_calls="tile_calls", maximum_image_candidate_evaluations="image_candidate_evaluations_upper_bound")


def _case(nk=2, *, frozen=False):
    b = _frozen_bundle((nk, 1, 1)) if frozen else _bundle((nk, 1, 1))
    _prepare_leaves(b, localization_options=_he2_controls(b)[0].localization if frozen else None)
    assert not hasattr(b, "provider")
    return b


def _selection(left, right):
    s = core._PeriodicGaussianDensityGramSelection()
    for name, (begin, count) in (("left", left), ("right", right)):
        r = core._PeriodicGaussianDensityRange()
        r.begin, r.count = begin, count
        setattr(s, name, r)
    return s


def _controls(b, *, auxiliary_block=1, pair_block=3):
    config = core._PeriodicGaussianDensityGramConfig()
    config.auxiliary_block = auxiliary_block
    panel = core._PeriodicGaussianLocalOrbitalFactorConfig()
    panel.ao_pair_block = pair_block
    config.panel = panel
    live = core._PeriodicGaussianLocalOrbitalFactorLiveInventory()
    live.other_retained_bytes_per_worker = 2**20  # Other tiny test owners, never factor rows input.
    live.fixed_backend_margin_bytes_per_worker = 65536
    caps = core._PeriodicGaussianDensityGramCaps()
    resources = core._PeriodicGaussianMetricCaps()
    resources.maximum_owned_numeric_bytes = 2**23
    resources.maximum_per_replica_inventoried_bytes = 2**26
    resources.maximum_node_inventoried_bytes = 2**27
    resources.maximum_candidate_evaluations = 10**16
    resources.maximum_work_units = 10**19
    caps.resources, caps.metric, caps.panel = resources, b.config.metric_caps, b.panel_caps
    caps.maximum_left_density_count = caps.maximum_right_density_count = 36
    caps.maximum_output_elements = 1296
    caps.maximum_factor_panels, caps.maximum_tile_calls = 4096, 65536
    caps.maximum_image_candidate_evaluations = 10**16
    caps.maximum_leaf_control_bytes = 2**20
    return config, _options(), live, caps


def _arguments(b, selection, controls=None, **changes):
    config, options, live, caps = _controls(b) if controls is None else controls
    args = dict(hf=b.hf, reference=b.reference, wannier=b.wannier, domain=b.domain,
        space=b.space, basis=b.basis, selection=selection, config=config, options=options, live=live, caps=caps)
    args.update(changes)
    return args


def _plan(b, selection, controls=None, **changes):
    return core._plan_periodic_gaussian_density_gram_diagnostic(**_arguments(b, selection, controls, **changes))


def _build(b, selection, controls=None, **changes):
    args = _arguments(b, selection, controls)
    args.update(ao_basis=b.ao, auxiliary_basis=b.auxiliary, gauges=b.gauge)
    args.update(changes)
    return core._build_periodic_gaussian_density_gram_diagnostic(**args)


def _oracle(b):
    # Reproduce the original HF whitening block size, including He2/frozen
    # fixtures where that recipe differs from the full auxiliary dimension.
    panels = np.asarray([_dense_panel(b, source, w)
        for source, _, w in (_q_data(b, q) for q in range(b.context.n_kpoints))])
    rows, labels = _expected_rows(panels, b.reference.state.mesh)
    return SimpleNamespace(panels=panels, rows=rows, labels=labels)


@pytest.fixture(scope="module", params=[(1, False), (2, False), (3, False), (2, True)])
def actual(request):
    nk, frozen = request.param
    b = _case(nk, frozen=frozen)
    m = b.basis.memory.orbital_count
    h = m*(m+1)//2
    selection = _selection((1, h-1), (0, min(h, m+1)))
    controls = _controls(b)
    # This is the FIRST integral consumer. No old provider/rows object has
    # been constructed, supplied, or relabelled as this new source.
    result = _build(b, selection, controls)
    oracle = _oracle(b)
    provider = _full_provider(b)  # Independent legacy path only afterward.
    return SimpleNamespace(b=b, nk=nk, frozen=frozen, selection=selection,
        controls=controls, result=result, oracle=oracle, provider=provider)


@pytest.fixture(scope="module")
def k2():
    return _case(2)


def _check(result, oracle):
    p, d = result.memory, result.diagnostics
    s = p.selection
    left = slice(s.left.begin, s.left.begin+s.left.count)
    right = slice(s.right.begin, s.right.begin+s.right.count)
    expected = oracle.rows[:, left].T@oracle.rows[:, right]
    values = result.matrix_copy()
    np.testing.assert_allclose(values, expected, atol=5e-12, rtol=3e-11)
    assert np.all(np.isfinite(values))
    for i, (a, b) in enumerate(oracle.labels[left]):
        assert tuple(result.left_density(i)) == (a, b)
        for j, (c, e) in enumerate(oracle.labels[right]):
            assert tuple(result.right_density(j)) == (c, e)
            assert result.element(i, j) == values[i, j]
            # The unprojected Coulomb orientation reverses the LEFT density
            # before conjugation. Check all symmetric density orientations,
            # not only a same-oriented complex Gram product.
            for x, y in ((a, b), (b, a)):
                for z, t in ((c, e), (e, c)):
                    raw = np.vdot(oracle.panels[:, :, y, x], oracle.panels[:, :, z, t])
                    bound = (d.real_projection.maximum_eri_projection_error_bound
                        + d.maximum_integral_roundoff_error)
                    assert abs(raw-values[i, j]) <= bound+5e-12
    assert d.completed_source_count == d.completed_metric_count == d.completed_whitening_count == p.n_cells
    assert d.completed_factor_panels == p.factor_panels
    assert d.completed_tile_calls == p.tile_calls
    assert d.charged_work_units_upper_bound <= p.work_units
    assert d.reciprocal_candidate_evaluations <= p.reciprocal_candidate_evaluations_upper_bound
    assert d.image_candidate_evaluations <= p.image_candidate_evaluations_upper_bound
    assert d.maximum_observed_owned_numerical_bytes <= p.peak_owned_numerical_bytes
    assert d.maximum_observed_per_worker_inventoried_bytes <= p.per_worker_inventoried_bytes
    assert 0 <= d.maximum_integral_roundoff_error <= 1e-10
    assert d.real_projection.maximum_eri_projection_error_bound <= 1e-10


def test_actual_block_without_prior_provider_matches_projected_and_complex_oracles(actual):
    result = actual.result
    _check(result, actual.oracle)
    s = actual.selection
    rows = actual.provider.provider.rows_copy()
    expected = rows[:, s.left.begin:s.left.begin+s.left.count].T@rows[:, s.right.begin:s.right.begin+s.right.count]
    np.testing.assert_allclose(result.matrix_copy(), expected, atol=5e-12, rtol=3e-11)
    assert result.hf_reference_source_identity_sha256 == actual.b.hf.reference_source_identity_sha256
    assert result.basis_identity_sha256 == actual.b.basis.identity_sha256
    assert result.source_context_identity_sha256 == actual.b.context.source_context_identity_sha256
    assert result.matched_finite_gaussian_hf_recipe
    assert not result.original_provider_projection_reproduced_bitwise
    assert not result.all_common_densities_audited
    assert not result.infinite_source_accuracy_certified and not result.production_dlpno
    for name in ("identity_sha256", "payload_sha256", "consumed_sources_identity_sha256"):
        assert len(getattr(result, name)) == 64 and int(getattr(result, name), 16) >= 0


def test_q_pairing_preserves_sine_rows_and_has_no_extra_cell_energy_normalization(actual):
    p = actual.result.memory
    assert p.row_count == actual.nk*actual.b.auxiliary.nbasis
    assert p.self_inverse_q_count == (1 if actual.nk == 3 else actual.nk)
    s = actual.selection
    h = actual.oracle.rows.shape[1]
    if actual.nk == 3:
        # Occupied index1 is the home orbital translated into cell1. Its
        # diagonal density supplies a genuine q1 sine lane in this source.
        translated_density = p.orbital_count
        assert actual.oracle.labels[translated_density] == (1, 1)
        sine = actual.oracle.rows[2*actual.b.auxiliary.nbasis:3*actual.b.auxiliary.nbasis, translated_density]
        assert np.linalg.norm(sine) > 1e-12
        i, j = translated_density-s.left.begin, translated_density-s.right.begin
        without_sine = actual.oracle.rows[:, translated_density].copy()
        without_sine[2*actual.b.auxiliary.nbasis:] = 0
        assert abs(actual.result.element(i, j)-without_sine@without_sine) > 1e-12
    assert h == p.common_density_count
    if actual.nk > 1:
        expected = actual.oracle.rows[:, s.left.begin:s.left.begin+s.left.count].T@actual.oracle.rows[:, s.right.begin:s.right.begin+s.right.count]
        assert np.linalg.norm(expected) > 1e-8
        assert np.linalg.norm(actual.result.matrix_copy()-expected/actual.nk) > 1e-8
    if actual.frozen:
        assert p.occupied_count == actual.nk and p.virtual_count == 2*actual.nk


def _runs(labels):
    return [sum(a == p for a, _ in labels) for p in sorted({a for a, _ in labels})]


def test_exact_bounded_inventory_counts_overlapping_ranges_by_role(actual):
    p, config, caps = actual.result.memory, actual.controls[0], actual.controls[3]
    s = p.selection
    l, r, d, b, a, o = s.left.count, s.right.count, p.selected_density_count, config.auxiliary_block, p.n_basis, p.occupied_count
    nonself = p.self_inverse_q_count != p.n_cells
    lruns = _runs(actual.oracle.labels[s.left.begin:s.left.begin+l])
    rruns = _runs(actual.oracle.labels[s.right.begin:s.right.begin+r])
    t = max(lruns+rruns)
    assert d == l+r  # Not the size of the union when ranges overlap.
    assert (p.left_run_count, p.right_run_count, p.maximum_run_length) == (len(lruns), len(rruns), t)
    assert p.retained_output_bytes == 8*l*r == actual.result.matrix_copy().nbytes
    assert p.retained_output_bytes+p.integral_accumulator_bytes == 32*l*r
    assert p.norm_workspace_bytes == 24*d
    assert p.row_slab_bytes == (16 if nonself else 8)*b*d
    base = 32*l*r+24*d+p.row_slab_bytes
    assert p.factor_phase_retained_bytes == base
    assert p.whitener_bytes == 16*p.n_auxiliary**2
    assert p.maximum_live_whitener_bytes == (2 if nonself else 1)*p.whitener_bytes
    assert p.retained_panel_upper_bytes == 16*b*t+16*o
    assert p.panel_driver_owned_upper_bytes == 32*b*t+16*a*(t+1)+32*a+16*o
    assert p.metric_phase_upper_bytes == base+(p.whitener_bytes if nonself else 0)+caps.metric.maximum_owned_numeric_bytes
    assert p.panel_phase_upper_bytes == (base+p.maximum_live_whitener_bytes
        +(2 if nonself else 1)*p.retained_panel_upper_bytes+p.panel_driver_owned_upper_bytes
        +caps.panel.tile.resources.maximum_owned_numeric_bytes)
    assert p.peak_owned_numerical_bytes == max(p.metric_phase_upper_bytes, p.panel_phase_upper_bytes)
    assert p.factor_panels == 2*p.n_cells*p.auxiliary_block_count*(p.left_run_count+p.right_run_count)
    assert p.required_node_memory_bytes == p.reference_base_node_bytes+p.replicas_per_node*p.per_worker_inventoried_bytes
    assert not hasattr(actual.result, "provider") and not hasattr(actual.result, "rows_copy")


def test_split_blocks_and_auxiliary_partitions_reconstruct_the_same_selected_gram(k2):
    m = k2.basis.memory.orbital_count
    h = m*(m+1)//2
    full = _build(k2, _selection((0, h), (0, h)))
    assembled = np.empty((h, h))
    # Packed-range boundaries cut both a same-p run and occupied/virtual
    # groups. Overlap between left/right is intentional, not deduplicated.
    for lb, ln in ((0, 2), (2, h-2)):
        for rb, rn in ((0, m+1), (m+1, h-m-1)):
            block = _build(k2, _selection((lb, ln), (rb, rn)), _controls(k2, auxiliary_block=2, pair_block=1))
            assembled[lb:lb+ln, rb:rb+rn] = block.matrix_copy()
    np.testing.assert_allclose(assembled, full.matrix_copy(), atol=5e-12, rtol=3e-11)
    np.testing.assert_allclose(assembled, assembled.T, atol=5e-12, rtol=3e-11)
    assert np.linalg.eigvalsh((assembled+assembled.T)/2).min() >= -5e-12
    _check(full, _oracle(k2))


def test_swapping_density_ranges_transposes_values_without_reversed_label_inputs(k2):
    forward = _build(k2, _selection((1, 4), (5, 3)))
    reverse = _build(k2, _selection((5, 3), (1, 4)))
    np.testing.assert_allclose(forward.matrix_copy(), reverse.matrix_copy().T, atol=5e-12, rtol=3e-11)
    assert forward.identity_sha256 != reverse.identity_sha256
    assert all(p <= q for p, q in (forward.left_density(i) for i in range(4)))
    repeated = _build(k2, _selection((4, 1), (4, 1)))
    assert repeated.memory.selected_density_count == 2  # Same density, two roles.
    assert tuple(repeated.left_density(0)) == tuple(repeated.right_density(0)) == (1, 1)
    assert repeated.element(0, 0) >= 0


def test_count_only_interior_range_plans_match_explicit_packed_labels(k2):
    m = k2.basis.memory.orbital_count
    labels = [(p, q) for p in range(m) for q in range(p, m)]
    rng = np.random.default_rng(571902)
    for _ in range(100):
        left, right = [], []
        for output in (left, right):
            begin = int(rng.integers(len(labels)))
            count = int(rng.integers(1, len(labels)-begin+1))
            output.extend((begin, count))
        selection = _selection(left, right)
        plan = _plan(k2, selection)
        lrun = _runs(labels[left[0]:left[0]+left[1]])
        rrun = _runs(labels[right[0]:right[0]+right[1]])
        assert plan.left_run_count == len(lrun)
        assert plan.right_run_count == len(rrun)
        assert plan.maximum_run_length == max(lrun+rrun)
        assert plan.output_elements == left[1]*right[1]


def test_finite_roundoff_budget_is_enforced_on_actual_streamed_products(k2):
    controls = _controls(k2)
    controls[1].maximum_scalar_roundoff_error = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="roundoff"):
        _build(k2, _selection((0, 1), (0, 1)), controls)


@pytest.mark.parametrize("field,reported", list(_RESOURCE_CAPS.items()))
def test_each_resource_cap_minus_one_precedes_nan_gauge_payload(k2, field, reported):
    s, controls = _selection((1, 4), (2, 3)), _controls(k2)
    p = _plan(k2, s, controls)
    caps, resources = controls[3], controls[3].resources
    setattr(resources, field, getattr(p, reported)-1)
    caps.resources = resources
    bad = np.full_like(k2.gauge, np.nan)
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap|budget|limit"):
        _build(k2, s, controls, gauges=bad)


@pytest.mark.parametrize("field,reported", list(_DIRECT_CAPS.items()))
def test_each_count_cap_minus_one_precedes_nan_gauge_payload(k2, field, reported):
    s, controls = _selection((1, 4), (2, 3)), _controls(k2)
    p = _plan(k2, s, controls)
    wanted = getattr(p.selection, reported).count if reported in ("left", "right") else getattr(p, reported)
    setattr(controls[3], field, wanted-1)
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap|budget|limit"):
        _build(k2, s, controls, gauges=np.full_like(k2.gauge, np.nan))


def test_exact_caps_and_minimum_leaf_control_admit_then_minus_one_rejects(k2):
    s, controls = _selection((1, 4), (2, 3)), _controls(k2)
    p = _plan(k2, s, controls)
    controls[3].maximum_leaf_control_bytes = p.minimum_leaf_control_bytes
    p = _plan(k2, s, controls)
    resources = controls[3].resources
    for field, reported in _RESOURCE_CAPS.items():
        setattr(resources, field, getattr(p, reported))
    controls[3].resources = resources
    for field, reported in _DIRECT_CAPS.items():
        setattr(controls[3], field, getattr(p.selection, reported).count
            if reported in ("left", "right") else getattr(p, reported))
    _check(_build(k2, s, controls), _oracle(k2))
    controls[3].maximum_leaf_control_bytes = p.minimum_leaf_control_bytes-1
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="control|cap"):
        _build(k2, s, controls, gauges=np.full_like(k2.gauge, np.nan))


@pytest.mark.parametrize("side,field,value", [(side, field, value) for side in ("left", "right")
    for field, value in (("count", 0), ("begin", 10), ("count", 11), ("begin", 2**64-1), ("count", 2**64-1))])
def test_invalid_packed_ranges_cannot_produce_labels_or_allocate(k2, side, field, value):
    s = _selection((0, 1), (0, 1))
    part = getattr(s, side)
    setattr(part, field, value)
    setattr(s, side, part)
    with pytest.raises((ValueError, RuntimeError, IndexError, OverflowError)):
        _build(k2, s, gauges=np.full_like(k2.gauge, np.nan))


@pytest.mark.parametrize("field", ["reversal_absolute_tolerance", "reversal_relative_tolerance",
    "conjugacy_absolute_tolerance", "conjugacy_relative_tolerance", "self_q_absolute_tolerance",
    "self_q_relative_tolerance", "maximum_eri_projection_error", "maximum_scalar_roundoff_error"])
def test_nonfinite_scientific_controls_are_not_defaults(k2, field):
    controls = _controls(k2)
    setattr(controls[1], field, float("nan"))
    with pytest.raises((ValueError, RuntimeError), match="finite|tolerance|control|bound"):
        _plan(k2, _selection((0, 1), (0, 1)), controls)


@pytest.mark.parametrize("field", ["auxiliary_block", "ao_pair_block", "margin"])
def test_zero_block_and_backend_controls_fail_before_source_work(k2, field):
    controls = _controls(k2)
    if field == "auxiliary_block":
        controls[0].auxiliary_block = 0
    elif field == "ao_pair_block":
        panel = controls[0].panel
        panel.ao_pair_block = 0
        controls[0].panel = panel
    else:
        controls[2].fixed_backend_margin_bytes_per_worker = 0
    with pytest.raises((ValueError, RuntimeError), match="positive|block|margin|control"):
        _build(k2, _selection((0, 1), (0, 1)), controls, gauges=np.full_like(k2.gauge, np.nan))


def test_binding_accepts_neither_caller_factor_rows_nor_a_callback(k2):
    selection = _selection((0, 1), (0, 1))
    with pytest.raises(TypeError):
        _build(k2, selection, rows=np.ones((1, 1)))
    with pytest.raises(TypeError):
        _build(k2, selection, progress=lambda event: None)


@pytest.mark.parametrize("kind", ["dtype", "shape", "strides", "nonfinite", "changed"])
def test_gauge_descriptors_and_actual_contents_are_not_relabelled(k2, kind):
    gauge = k2.gauge.copy()
    if kind == "dtype":
        gauge = gauge.real.copy()
    elif kind == "shape":
        gauge = gauge.reshape(-1)
    elif kind == "strides":
        gauge = np.repeat(gauge, 2, axis=0)[::2]
    elif kind == "nonfinite":
        gauge.flat[0] = complex(float("nan"), 0)
    else:
        gauge.flat[0] *= -1
    with pytest.raises((TypeError, ValueError, RuntimeError, OverflowError)):
        _build(k2, _selection((0, 1), (0, 1)), gauges=gauge)


def test_actual_source_and_basis_owners_cannot_be_substituted(k2):
    other = _case(2)
    s = _selection((0, 1), (0, 1))
    for changes in (dict(hf=other.hf), dict(reference=other.reference), dict(basis=other.basis)):
        with pytest.raises((ValueError, RuntimeError), match="owner|state|source|identity|reference|basis"):
            _build(k2, s, **changes)
    with pytest.raises((ValueError, RuntimeError), match="basis|content|identity"):
        _build(k2, s, ao_basis=k2.auxiliary)
    # Larger actual primitive census must fail before accessing its NaN
    # lanes, despite generous caller metadata verification ceilings.
    grown = _raw_basis([(0, (0, 0, 0), [float("nan")], [1.0], True)]*3)
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="census|cap|count|basis"):
        _build(k2, s, auxiliary_basis=grown)


def test_explicit_caller_live_bytes_are_added_once_to_worker_and_node(k2):
    s, controls = _selection((0, 2), (2, 2)), _controls(k2)
    original = _plan(k2, s, controls)
    controls[2].other_retained_bytes_per_worker += 1232
    controls[2].other_transient_bytes_per_worker += 456
    extended = _plan(k2, s, controls)
    assert extended.peak_owned_numerical_bytes == original.peak_owned_numerical_bytes
    assert extended.per_worker_inventoried_bytes-original.per_worker_inventoried_bytes == 1688
    assert extended.required_node_memory_bytes-original.required_node_memory_bytes == 1688*original.replicas_per_node


def test_detached_controls_and_outputs_do_not_mutate_block_and_owners_may_die():
    b = _case(1)
    selection, controls = _selection((0, 2), (1, 2)), _controls(b)
    result = _build(b, selection, controls)
    expected, identity = result.matrix_copy(), result.identity_sha256
    r = selection.left
    r.begin = 2**64-1
    selection.left = r
    local = result.memory.selection
    r = local.right
    r.count = 999
    local.right = r
    controls[0].auxiliary_block = 0
    copy = result.matrix_copy()
    assert not copy.flags.writeable
    copy.setflags(write=True)
    copy[:] = np.nan
    for method in (result.left_density, result.right_density):
        with pytest.raises(IndexError):
            method(2)
    with pytest.raises(IndexError):
        result.element(2, 0)
    del b, selection, controls
    gc.collect()
    np.testing.assert_array_equal(result.matrix_copy(), expected)
    assert result.identity_sha256 == identity
    assert result.state.n_kpoints == result.context.n_kpoints == 1
