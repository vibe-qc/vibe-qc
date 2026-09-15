"""Connected native Gaussian/localization references, never chemical SCF jobs."""

from __future__ import annotations

import gc
import hashlib
import struct

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_bloch_iao import (
    _case, _caps as _source_caps, _options as _source_options,
    _make as _source, _plan as _source_plan, _basis,
)
from tests.test_periodic_correlation_bloch_iao import _options as _iao_options, _ao_oracle
from tests.test_periodic_correlation_diabatic_seed import (
    _make as _seed, _options as _seed_options, _plan as _seed_plan,
)
from tests.test_periodic_correlation_iao_optimizer import _options as _optimizer_options
from tests.test_periodic_correlation_iao_pm import _options as _pm_options, _oracle as _pm_oracle
from tests.test_periodic_correlation_wannier import _options as _wannier_options, _home


def _options(case, **optimizer_changes):
    result = core._PeriodicGaussianLocalizationOptions()
    result.source = _source_options(case["cutoff"])
    result.iao = _iao_options()
    result.seed = _seed_options(case["reference"])
    result.pm = _pm_options()
    result.optimizer = _optimizer_options(**optimizer_changes)
    result.wannier = _wannier_options()
    return result


def _caps(**changes):
    result = core._PeriodicGaussianLocalizationCaps()
    result.maximum_owned_numerical_bytes = 1 << 20
    result.maximum_control_storage_bytes = 1 << 20
    result.maximum_point_owners = 8
    result.maximum_work_units = 20_000_000_000_000
    result.maximum_total_image_candidates = 2_000_000
    result.maximum_optimizer_work_units = 1_000_000_000
    for name, value in changes.items():
        setattr(result, name, value)
    return result


def _run(case, *, options=None, caps=None, source_caps=None, other=0, callback=None):
    return core._localize_periodic_gaussian_occupied(
        case["reference"], case["ao"], case["minimal"], case["system"], other,
        _options(case) if options is None else options,
        _source_caps() if source_caps is None else source_caps,
        _caps() if caps is None else caps, callback,
    )


def _direct(case, options, *, optimizer_work=1_000_000_000, other=0):
    """Existing independent native calls, including their real live owners."""
    seed = _seed(case["reference"], controls=options.seed)
    points, previous = [], 0
    for point in range(len(case["cells"])):
        owner = _source(case, point, other=other + seed.memory.output_numerical_bytes + previous)
        previous += owner.memory.output_numerical_bytes
        points.append(owner)
    caps = core._PeriodicCorrelationIAOOptimizerCaps()
    caps.maximum_owned_numerical_bytes = 1 << 20
    caps.maximum_work_units = optimizer_work
    optimized = core._optimize_periodic_correlation_iao_pm(case["reference"], seed,
        [point.iao for point in points], options.pm, options.optimizer, caps)
    return seed, points, optimized


def _ordered_digest(domain, values):
    payload = bytearray()

    def integer(n):
        payload.extend(struct.pack(">Q", n))

    def text(value):
        encoded = value.encode()
        integer(len(encoded)); payload.extend(encoded)

    text(domain); integer(1); integer(len(values))
    for point, value in enumerate(values):
        integer(point); text(value)
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1), (2, 2, 2)])
def test_connected_gaussian_native_chain_matches_direct_calls_and_original_ao_charge_oracle(mesh):
    case = _case(mesh)
    options = _options(case)
    result = _run(case, options=options)
    _, points, optimized = _direct(case, options)
    assert result.optimizer.status == optimized.status
    np.testing.assert_array_equal(result.optimizer.gauges_copy(), optimized.gauges_copy())
    assert result.optimizer.diagnostics.final_objective == optimized.diagnostics.final_objective
    assert result.ordered_source_identity_sha256 == _ordered_digest(
        "vibeqc.periodic.gaussian-localization.ordered-sources", [p.source_identity_sha256 for p in points])
    assert result.ordered_reference_match_identity_sha256 == _ordered_digest(
        "vibeqc.periodic.gaussian-localization.ordered-reference-matches",
        [p.reference_match_identity_sha256 for p in points])
    bs, ds = [], []
    for point in range(len(case["cells"])):
        _, _, b, d = _ao_oracle(case["c"][point], case["s"][point],
                              case["cross"][point], case["small"][point], [0, 1, 2])
        bs.append(b[:, [0, 2]]); ds.append(d[:, [0, 2]])
    labels = np.array([0, 1, 2], dtype=np.uint64)
    score, _, _ = _pm_oracle(mesh, result.optimizer.gauges_copy(), np.array(bs), np.array(ds), labels)
    assert result.optimizer.diagnostics.final_objective == pytest.approx(score, abs=3e-10)
    assert not result.hf_basis_source_authenticated
    assert not result.infinite_image_tail_certified
    assert result.optimizer.state is case["reference"].state
    assert result.optimizer.allocation_identity == case["reference"].dimensions.allocation_identity
    if result.converged:
        direct = core._make_periodic_correlation_wannier_from_iao_optimizer(
            case["reference"], optimized, 1 << 20, options.wannier)
        np.testing.assert_array_equal(_home(result.wannier), _home(direct))
    else:
        with pytest.raises(RuntimeError, match="converged"):
            _ = result.wannier


def test_gamma_converges_and_home_coefficients_match_independent_fourier_metric():
    case = _case((1, 1, 1))
    result = _run(case)
    assert result.converged
    u = result.optimizer.gauges_copy()
    expected = case["c"][:, :, [0, 2]] @ u
    np.testing.assert_allclose(_home(result.wannier), expected, atol=3e-12)
    home = _home(result.wannier)[0]
    np.testing.assert_allclose(home.conj().T @ case["s"][0] @ home, np.eye(2), atol=3e-12)
    assert result.wannier.diagnostics.maximum_home_imaginary_magnitude < 3e-12
    assert result.optimizer.diagnostics.riemannian_gradient_norm <= _options(case).optimizer.riemannian_gradient_tolerance


@pytest.mark.parametrize("workers", [1, 2])
def test_full_enclosing_inventory_counts_borrowed_gaussians_and_each_live_owner_once(workers):
    case = _case((3, 1, 1))
    ref = case["reference"]
    dims, budget = ref.dimensions, ref.budget
    dims.external_bytes -= ref.state.resident_bytes
    budget.workers_per_rank = workers
    case["reference"] = core._make_periodic_correlation_admitted_reference(ref.state, dims, budget)
    other = 1234
    result = _run(case, other=other, options=_options(case, maximum_iterations=1))
    m = result.memory
    source = _source_plan(case)
    seed = _seed_plan(case["reference"])
    seed_owner = _seed(case["reference"])
    opt = core._plan_periodic_correlation_iao_optimizer(case["reference"], seed_owner,
        case["minimal"].nbasis, _options(case, maximum_iterations=1).optimizer)
    w = core._plan_periodic_correlation_wannier(case["mesh"], m.n_basis, m.n_active)
    k = len(case["cells"])
    assert m.borrowed_gaussian_numerical_bytes == source.borrowed_basis_numeric_bytes + source.borrowed_geometry_numeric_bytes
    assert m.other_live_numerical_bytes == other
    assert m.seed_phase_owned_bytes == seed.peak_owned_numerical_bytes
    assert m.source_phase_owned_bytes == seed.output_numerical_bytes + (k - 1) * source.output_numerical_bytes + source.peak_owned_numerical_bytes
    assert m.optimizer_phase_owned_bytes == seed.output_numerical_bytes + k * source.output_numerical_bytes + 8 * k + opt.peak_owned_numerical_bytes
    assert m.wannier_phase_owned_bytes == opt.output_numerical_bytes + w.peak_owned_numerical_bytes
    assert m.source_identity_character_bytes == k * 11 * 64
    assert m.fixed_control_reservation_bytes == 65536
    assert m.peak_control_storage_bytes == (m.point_plan_record_bytes + m.point_owner_record_bytes
        + m.source_identity_character_bytes + m.fixed_control_reservation_bytes)
    assert m.peak_owned_numerical_bytes == max(m.seed_phase_owned_bytes, m.source_phase_owned_bytes,
                                             m.optimizer_phase_owned_bytes, m.wannier_phase_owned_bytes)
    dims = case["reference"].dimensions
    baseline = dims.external_bytes + dims.shared_bytes + dims.per_rank_bytes + dims.localization_window_bytes_per_rank
    extra = m.borrowed_gaussian_numerical_bytes + other + m.peak_control_storage_bytes
    for phase in ("seed", "source", "optimizer", "wannier"):
        assert getattr(m, phase + "_required_node_bytes") == baseline + workers * (extra + getattr(m, phase + "_phase_owned_bytes"))
    assert m.required_node_memory_bytes == baseline + workers * (extra + m.peak_owned_numerical_bytes)
    assert m.required_node_memory_bytes > result.optimizer.memory.required_node_memory_bytes
    assert result.diagnostics.image_candidate_visits == m.planned_total_image_visits
    assert m.planned_total_image_visits == source.image_candidate_count + 4 * m.planned_source_image_candidates


@pytest.mark.parametrize("field,reported", [
    ("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_control_storage_bytes", "peak_control_storage_bytes"),
    ("maximum_total_image_candidates", "planned_total_image_visits"),
])
def test_exact_enclosing_caps_and_minus_one(field, reported):
    case = _case((1, 1, 1))
    options = _options(case, maximum_iterations=1)
    first = _run(case, options=options)
    required = getattr(first.memory, reported)
    repeated = _run(case, options=options, caps=_caps(**{field: required}))
    np.testing.assert_array_equal(repeated.optimizer.gauges_copy(), first.optimizer.gauges_copy())
    with pytest.raises((ValueError, RuntimeError), match="cap|reservation"):
        _run(case, options=options, caps=_caps(**{field: required - 1}))


def test_original_reference_enclosing_node_exact_boundary_and_other_live_increment():
    case = _case((1, 1, 1))
    options = _options(case, maximum_iterations=1)
    first = _run(case, options=options)
    ref = case["reference"]
    dims, budget = ref.dimensions, ref.budget
    dims.external_bytes -= ref.state.resident_bytes
    budget.memory_limit_bytes = first.memory.required_node_memory_bytes
    case["reference"] = core._make_periodic_correlation_admitted_reference(ref.state, dims, budget)
    repeated = _run(case, options=options)
    assert repeated.optimizer.state is ref.state
    assert repeated.memory.required_node_memory_bytes == budget.memory_limit_bytes
    with pytest.raises((ValueError, RuntimeError), match="enclosing node"):
        _run(case, options=options, other=1)


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_control_storage_bytes",
    "maximum_point_owners", "maximum_work_units", "maximum_total_image_candidates", "maximum_optimizer_work_units"])
def test_zero_caps_are_never_unlimited(field):
    with pytest.raises(ValueError, match="positive"):
        _run(_case((1, 1, 1)), caps=_caps(**{field: 0}))


def test_early_work_gate_precedes_bad_s11_and_tiny_owner_gate_precedes_any_points():
    case = _case((3, 1, 1), reference_scale=1.01)
    with pytest.raises((ValueError, RuntimeError), match="total work"):
        _run(case, caps=_caps(maximum_work_units=1))
    with pytest.raises((ValueError, RuntimeError), match="point-owner"):
        _run(case, caps=_caps(maximum_point_owners=1))
    with pytest.raises(ValueError, match="S11.*admitted overlap"):
        _run(case)


@pytest.mark.parametrize("kind", ["iteration", "work", "line"])
def test_nonconvergence_preserves_accepted_optimizer_and_never_returns_wannier(kind):
    case = _case((1, 1, 1))
    options, caps = _options(case), _caps()
    seed = _seed(case["reference"])
    if kind == "iteration":
        options.optimizer.maximum_iterations = 1
        expected = core._PeriodicCorrelationIAOOptimizerStatus.ITERATION_LIMIT
    elif kind == "work":
        plan = core._plan_periodic_correlation_iao_optimizer(case["reference"], seed,
            case["minimal"].nbasis, options.optimizer)
        caps.maximum_optimizer_work_units = plan.initial_work_units
        expected = core._PeriodicCorrelationIAOOptimizerStatus.WORK_LIMIT
    else:
        options.optimizer.initial_step = options.optimizer.minimum_step = 1e-300
        expected = core._PeriodicCorrelationIAOOptimizerStatus.LINE_SEARCH_FAILED
    result = _run(case, options=options, caps=caps)
    assert not result.converged
    assert result.optimizer.status == expected
    assert result.memory.output_numerical_bytes == result.optimizer.memory.output_numerical_bytes
    with pytest.raises(RuntimeError, match="converged"):
        _ = result.wannier
    if kind != "iteration":
        np.testing.assert_array_equal(result.optimizer.gauges_copy(), np.array([
            [[seed.gauge(k, a, b) for b in range(2)] for a in range(2)] for k in range(len(case["cells"]))]))


def test_live_progress_is_scalar_monotonic_and_final_receipt_retains_no_source_owner():
    case = _case((1, 1, 1))
    events = []
    result = _run(case, callback=events.append)
    stage = core._PeriodicGaussianLocalizationStage
    assert [e.stage for e in events[:2]] == [stage.SEED_READY, stage.POINT_READY]
    assert events[-1].stage == stage.FINISHED
    assert any(e.stage == stage.OPTIMIZATION for e in events)
    assert events[-1].charged_work_units == result.diagnostics.charged_work_units
    assert result.diagnostics.callback_input_checks == len(events)
    assert all(a.charged_work_units <= b.charged_work_units for a, b in zip(events, events[1:]))
    optimizer, wannier = result.optimizer, result.wannier
    original = _home(wannier)
    del result, case
    gc.collect()
    assert optimizer.converged
    np.testing.assert_array_equal(_home(wannier), original)


@pytest.mark.parametrize("stage", ["SEED_READY", "POINT_READY", "OPTIMIZATION", "FINISHED"])
def test_callback_cell_mutation_is_detected_even_inside_overlap_tolerance(stage):
    case = _case((1, 1, 1))
    selected = getattr(core._PeriodicGaussianLocalizationStage, stage)

    def callback(event):
        if event.stage == selected:
            lattice = case["system"].lattice.copy()
            lattice[0, 0] += 1e-14
            case["system"].lattice = lattice

    with pytest.raises(ValueError, match="callback changed"):
        _run(case, callback=callback)


def test_callback_oversized_atom_list_rejected_before_new_numeric_scans():
    case = _case((1, 1, 1))

    def callback(event):
        case["system"].unit_cell = case["system"].unit_cell * 5

    with pytest.raises(ValueError, match="atom count changed"):
        _run(case, callback=callback)


def test_callback_can_drop_python_inputs_and_mutate_original_option_wrapper_without_freeing_native_arguments():
    case = _case((1, 1, 1))
    options = _options(case)

    def callback(event):
        case.clear()
        options.optimizer.maximum_iterations = 0
        options.source.image_cutoff_bohr = 0
        gc.collect()

    result = _run(case, options=options, callback=callback)
    assert result.converged
    assert result.wannier.n_home_occupied == 2


def test_callback_exception_aborts_without_partial_result():
    def callback(event):
        raise RuntimeError("caller stopped localization")

    with pytest.raises(RuntimeError, match="caller stopped localization"):
        _run(_case((1, 1, 1)), callback=callback)


def test_finite_gaussian_identity_is_content_sensitive_and_panel_partition_invariant():
    case = _case((1, 1, 1))
    first = _run(case)
    partial = _run(case, source_caps=_source_caps(maximum_panel_pairs=1))
    assert first.gaussian_input_identity_sha256 == partial.gaussian_input_identity_sha256
    assert first.ordered_source_identity_sha256 == partial.ordered_source_identity_sha256
    np.testing.assert_array_equal(first.optimizer.gauges_copy(), partial.optimizer.gauges_copy())
    case["minimal"] = _basis(case["molecule"], case["specs"], name="new display name", scale=1.25)
    renamed = _run(case)
    assert renamed.gaussian_input_identity_sha256 == first.gaussian_input_identity_sha256
    case["minimal"] = _basis(case["molecule"], case["specs"], scale=1.4)
    changed = _run(case)
    assert changed.gaussian_input_identity_sha256 != first.gaussian_input_identity_sha256
    assert changed.ordered_source_identity_sha256 != first.ordered_source_identity_sha256


@pytest.mark.parametrize("field", ["require_time_reversal", "require_real_home_coefficients"])
def test_driver_never_disables_physical_wannier_audits(field):
    case = _case((1, 1, 1))
    options = _options(case)
    setattr(options.wannier, field, False)
    with pytest.raises(ValueError, match="TR and real-home"):
        _run(case, options=options)


def _verify_inputs(result, case, maximum_work_units=None):
    return core._verify_periodic_gaussian_localization_inputs(
        result, case["reference"], case["ao"], case["minimal"], case["system"],
        result.memory.input_verification_work_units if maximum_work_units is None else maximum_work_units,
    )


@pytest.mark.parametrize("iterations", [1, 64])
def test_downstream_original_input_check_accepts_same_content_not_display_names(iterations):
    case = _case((1, 1, 1))
    result = _run(case, options=_options(case, maximum_iterations=iterations))
    assert result.source_image_cutoff_bohr == case["cutoff"]
    _verify_inputs(result, case)
    case["ao"] = core.BasisSet(case["molecule"], case["ao"].shells(), "renamed AO", True)
    case["minimal"] = core.BasisSet(case["molecule"], case["minimal"].shells(), "renamed minimal", True)
    _verify_inputs(result, case)
    assert not result.hf_basis_source_authenticated


@pytest.mark.parametrize("kind", ["ao", "minimal", "cell", "atoms", "ao_map"])
def test_downstream_original_input_check_rejects_changed_physical_content(kind):
    case = _case((1, 1, 1))
    result = _run(case)
    if kind in ("ao", "minimal"):
        specs = [*case["specs"], (0, 0, 1.9)] if kind == "ao" else case["specs"]
        case[kind] = _basis(case["molecule"], specs, scale=1.01)
    elif kind == "cell":
        lattice = case["system"].lattice.copy()
        lattice[0, 0] += 1e-14
        case["system"].lattice = lattice
    elif kind == "atoms":
        case["system"].unit_cell = list(reversed(case["system"].unit_cell))
    else:
        remapped = core.Molecule(list(reversed(case["molecule"].atoms)))
        case["ao"] = core.BasisSet(remapped, case["ao"].shells(), "same AO, different atom map", True)
    with pytest.raises(ValueError, match="original Gaussian inputs changed"):
        _verify_inputs(result, case)


@pytest.mark.parametrize("kind", ["state", "allocation"])
def test_downstream_original_input_check_requires_exact_state_and_allocation(kind):
    case = _case((1, 1, 1))
    result = _run(case)
    if kind == "state":
        case["reference"] = _case((1, 1, 1))["reference"]
    else:
        ref = case["reference"]
        dims = ref.dimensions
        dims.external_bytes -= ref.state.resident_bytes
        dims.allocation_identity = "a" * 64
        case["reference"] = core._make_periodic_correlation_admitted_reference(ref.state, dims, ref.budget)
    with pytest.raises(ValueError, match="exact admitted state/allocation"):
        _verify_inputs(result, case)


def test_downstream_original_input_work_admission_precedes_content_scan():
    case = _case((1, 1, 1))
    result = _run(case)
    required = result.memory.input_verification_work_units
    _verify_inputs(result, case, required)
    lattice = case["system"].lattice.copy()
    lattice[0, 0] = np.nan
    case["system"].lattice = lattice
    with pytest.raises((ValueError, RuntimeError), match="verification work cap"):
        _verify_inputs(result, case, required - 1)
    with pytest.raises(ValueError, match="not finite"):
        _verify_inputs(result, case, required)
