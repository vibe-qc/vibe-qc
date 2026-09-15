"""Tiny TR-safe native optimizer references; no SCF or chemical benchmarks."""

from __future__ import annotations

import gc
import hashlib
import struct

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_bloch_iao import _ao_oracle, _make as _iao
from tests.test_periodic_correlation_diabatic_seed import _gauges, _make as _seed
from tests.test_periodic_correlation_iao_pm import _options as _pm_options, _oracle
from tests.test_periodic_correlation_wannier import _reference


def _options(**changes):
    out = core._PeriodicCorrelationIAOOptimizerOptions()
    out.maximum_iterations = 192
    out.maximum_line_search_trials = 16
    out.initial_step = 0.1
    out.minimum_step = 1e-14
    out.backtracking_factor = 0.5
    out.armijo_fraction = 1e-4
    out.riemannian_gradient_tolerance = 2e-6
    out.absolute_tolerance = 3e-11
    out.relative_tolerance = 3e-11
    out.jacobi_max_sweeps = 32
    out.jacobi_relative_tolerance = 2e-15
    for key, value in changes.items():
        setattr(out, key, value)
    return out


def _case(mesh=(1, 1, 1), *, frozen=(1,), complex_gauge=False, nactive=2):
    p = nactive + len(frozen)
    active_bands = [i for i in range(p) if i not in frozen]
    nk = int(np.prod(mesh))
    rotations = np.tile(np.eye(p, dtype=complex), (nk, 1, 1))
    if complex_gauge:
        rng = np.random.default_rng(784)
        for k in range(nk):
            q = np.linalg.qr(rng.normal(size=(nactive, nactive))
                             + 1j * rng.normal(size=(nactive, nactive)))[0]
            rotations[k][np.ix_(active_bands, active_bands)] = q
            for band in frozen:
                rotations[k, band, band] = np.exp(1j * (0.31 + 0.13 * k))
    reference, cells, c, s, _ = _reference(mesh, nactive=nactive, frozen_indices=frozen,
                                         canonical_rotations=rotations)
    seed = _seed(reference)
    points, bs, ds, inputs = [], [], [], []
    labels = np.arange(p, dtype=np.uint64) + 17
    scales = np.linspace(1.15, 1.55, p)
    for k, cell in enumerate(cells):
        original = c[k, :, :p] @ rotations[k].conj().T
        atomic_rotation = np.eye(p)
        if nactive >= 2:
            angle = 0.23 + 0.035 * (1 - np.cos(2 * np.pi * np.sum(cell / mesh)))
            a, b = active_bands[:2]
            atomic_rotation[np.ix_([a, b], [a, b])] = [[np.cos(angle), -np.sin(angle)],
                                                      [np.sin(angle), np.cos(angle)]]
        # An unorthogonalized atomic frame with known charge-invariant column
        # scales. Its exact localized optimum is nactive, not a molecular SCF.
        minimal_ao = original @ (atomic_rotation * scales)
        cross = np.ascontiguousarray(s[k] @ minimal_ao)
        minimal = np.ascontiguousarray(np.diag(scales**2), dtype=complex)
        points.append(_iao(reference, cross, minimal, labels, point=k))
        _, _, bb, dd = _ao_oracle(c[k], s[k], cross, minimal, list(range(p)))
        bs.append(bb[:, active_bands]); ds.append(dd[:, active_bands])
        inputs.append((cross, minimal, labels))
    return dict(reference=reference, seed=seed, points=points, u=_gauges(seed),
                c=c, b=np.array(bs), d=np.array(ds), labels=labels, mesh=mesh,
                cells=cells, active=active_bands, inputs=inputs)


def _plan(case, options=None):
    return core._plan_periodic_correlation_iao_optimizer(case["reference"], case["seed"],
        len(case["labels"]), _options() if options is None else options)


def _caps(plan, **changes):
    out = core._PeriodicCorrelationIAOOptimizerCaps()
    out.maximum_owned_numerical_bytes = plan.peak_owned_numerical_bytes
    out.maximum_work_units = plan.maximum_work_units
    for name, value in changes.items():
        setattr(out, name, value)
    return out


def _run(case, *, options=None, caps=None, callback=None, points=None, seed=None):
    options = _options() if options is None else options
    caps = _caps(_plan(case, options)) if caps is None else caps
    return core._optimize_periodic_correlation_iao_pm(case["reference"],
        case["seed"] if seed is None else seed, case["points"] if points is None else points,
        _pm_options(), options, caps, callback)


def _score(case, u):
    return _oracle(case["mesh"], u, case["b"], case["d"], case["labels"])


def _projected_tangent(case, u):
    _, e, _ = _score(case, u)
    m = u.conj().swapaxes(1, 2) @ e
    h = (m - m.conj().swapaxes(1, 2)) / 2
    lookup = {tuple(cell): i for i, cell in enumerate(case["cells"])}
    for k, cell in enumerate(case["cells"]):
        l = lookup[tuple((-cell) % case["mesh"])]
        if l < k:
            continue
        if l == k:
            h[k] = h[k].real
        else:
            h[k] = (h[k] + h[l].conj()) / 2
            h[l] = h[k].conj()
    return h, e


def _exponentials(h, step):
    result = []
    for point in h:
        values, vectors = np.linalg.eigh(1j * point)
        result.append((vectors * np.exp(-1j * step * values)) @ vectors.conj().T)
    return np.array(result)


def _assert_physical(case, result):
    u = result.gauges_copy()
    np.testing.assert_allclose(u.conj().swapaxes(1, 2) @ u, np.tile(np.eye(u.shape[-1]), (len(u), 1, 1)), atol=3e-12)
    physical = np.array([case["c"][k][:, case["active"]] @ u[k] for k in range(len(u))])
    lookup = {tuple(cell): i for i, cell in enumerate(case["cells"])}
    for k, cell in enumerate(case["cells"]):
        l = lookup[tuple((-cell) % case["mesh"])]
        np.testing.assert_allclose(physical[k], physical[l].conj(), atol=3e-12)
        if l == k:
            np.testing.assert_allclose(physical[k].imag, 0.0, atol=2e-12)
        assert [result.active_band(k, i) for i in range(u.shape[-1])] == case["active"]


def test_gamma_real_optimizer_converges_to_known_unorthogonalized_atomic_optimum():
    case = _case()
    events = []
    result = _run(case, callback=events.append)
    assert result.converged
    assert result.status == core._PeriodicCorrelationIAOOptimizerStatus.CONVERGED
    assert result.diagnostics.final_objective == pytest.approx(2.0, abs=2e-11)
    assert result.diagnostics.final_objective > result.diagnostics.initial_objective + 0.1
    assert result.diagnostics.riemannian_gradient_norm <= _options().riemannian_gradient_tolerance
    assert np.linalg.norm(case["points"][0].metric_copy() - np.eye(3)) > 0.1
    accepted = [e for e in events if e.event == core._PeriodicCorrelationIAOOptimizerEvent.ACCEPTED]
    assert len(accepted) == result.diagnostics.accepted_steps > 0
    assert all(a.objective < b.objective for a, b in zip(accepted, accepted[1:]))
    assert events[0].event == core._PeriodicCorrelationIAOOptimizerEvent.INITIAL
    assert events[-1].event == core._PeriodicCorrelationIAOOptimizerEvent.FINISHED
    assert events[-1].objective == result.diagnostics.final_objective
    _assert_physical(case, result)


@pytest.mark.parametrize("mesh", [(3, 1, 1), (2, 2, 2)])
def test_nonself_pairs_and_all_eight_trim_with_arbitrary_complex_canonical_gauges(mesh):
    case = _case(mesh, complex_gauge=True)
    result = _run(case)
    assert result.converged
    assert result.diagnostics.final_objective == pytest.approx(2.0, abs=2e-10)
    assert result.diagnostics.maximum_physical_time_reversal_residual < 2e-12
    assert result.diagnostics.maximum_trim_relative_imaginary_correction < 1e-12
    assert np.max(np.abs(result.gauges_copy().imag)) > 0.05  # raw gauges must not be forced real
    _assert_physical(case, result)


def test_projected_full_mesh_gradient_and_first_step_match_independent_tr_directional_derivative():
    case = _case((3, 1, 1), complex_gauge=True)
    h, e = _projected_tangent(case, case["u"])
    slope = np.vdot(e, case["u"] @ h).real
    assert slope == pytest.approx(np.linalg.norm(h)**2, abs=3e-12)
    assert slope > 0.01
    delta = 2e-6
    plus = _score(case, case["u"] @ _exponentials(h, delta))[0]
    minus = _score(case, case["u"] @ _exponentials(h, -delta))[0]
    assert (plus - minus) / (2 * delta) == pytest.approx(slope, abs=3e-9, rel=3e-7)
    controls = _options(maximum_iterations=1, initial_step=1e-4, riemannian_gradient_tolerance=1e-14)
    result = _run(case, options=controls)
    assert result.status == core._PeriodicCorrelationIAOOptimizerStatus.ITERATION_LIMIT
    assert result.diagnostics.accepted_steps == 1
    expected = case["u"] @ _exponentials(h, controls.initial_step)
    np.testing.assert_allclose(result.gauges_copy(), expected, atol=5e-13)
    assert result.diagnostics.final_objective == pytest.approx(_score(case, expected)[0], abs=2e-12)
    next_h, _ = _projected_tangent(case, result.gauges_copy())
    assert result.diagnostics.riemannian_gradient_norm == pytest.approx(np.linalg.norm(next_h), abs=3e-12)
    _assert_physical(case, result)


def test_line_failure_keeps_last_accepted_seed_objective_gauge_and_gradient():
    case = _case()
    controls = _options(initial_step=10.0, maximum_line_search_trials=1, armijo_fraction=0.9)
    result = _run(case, options=controls)
    assert result.status == core._PeriodicCorrelationIAOOptimizerStatus.LINE_SEARCH_FAILED
    assert not result.converged and result.diagnostics.accepted_steps == 0
    assert result.diagnostics.rejected_trials == 1
    np.testing.assert_array_equal(result.gauges_copy(), case["u"])
    h, _ = _projected_tangent(case, case["u"])
    assert result.diagnostics.final_objective == pytest.approx(_score(case, case["u"])[0], abs=2e-12)
    assert result.diagnostics.riemannian_gradient_norm == pytest.approx(np.linalg.norm(h), abs=3e-12)


def test_precision_stagnation_is_line_failure_not_false_convergence():
    case = _case()
    controls = _options(initial_step=1e-300, minimum_step=1e-310, maximum_line_search_trials=1)
    result = _run(case, options=controls)
    assert result.status == core._PeriodicCorrelationIAOOptimizerStatus.LINE_SEARCH_FAILED
    assert result.diagnostics.riemannian_gradient_norm > controls.riemannian_gradient_tolerance
    assert not result.converged
    np.testing.assert_array_equal(result.gauges_copy(), case["u"])


def test_representable_improvement_is_not_rejected_when_scaled_armijo_bound_rounds_away():
    case = _case()
    controls = _options()
    events = []
    result = _run(case, options=controls, callback=events.append)
    assert result.converged
    states = [event for event in events if event.event in (
        core._PeriodicCorrelationIAOOptimizerEvent.INITIAL,
        core._PeriodicCorrelationIAOOptimizerEvent.ACCEPTED,
    )]
    witnesses = []
    for before, after in zip(states, states[1:]):
        model = after.step * before.riemannian_gradient_norm**2
        required = controls.armijo_fraction * model
        if before.objective + required == before.objective:
            assert before.riemannian_gradient_norm > controls.riemannian_gradient_tolerance
            assert before.objective + model > before.objective
            assert after.objective - before.objective >= required > 0
            witnesses.append((before, after))
    assert witnesses


@pytest.mark.parametrize("extra", [0, -1, 1])
def test_work_boundaries_preserve_exact_last_accepted_state(extra):
    case = _case()
    controls = _options(initial_step=0.05, riemannian_gradient_tolerance=1e-14)
    plan = _plan(case, controls)
    limit = plan.initial_work_units if extra == 0 else plan.initial_work_units + plan.work_units_per_trial + min(extra, 0)
    result = _run(case, options=controls, caps=_caps(plan, maximum_work_units=limit))
    assert result.status == core._PeriodicCorrelationIAOOptimizerStatus.WORK_LIMIT
    assert result.diagnostics.accepted_steps == (1 if extra == 1 else 0)
    assert result.diagnostics.charged_work_units <= limit
    if extra != 1:
        np.testing.assert_array_equal(result.gauges_copy(), case["u"])
    objective, _, _ = _score(case, result.gauges_copy())
    h, _ = _projected_tangent(case, result.gauges_copy())
    assert result.diagnostics.final_objective == pytest.approx(objective, abs=3e-12)
    assert result.diagnostics.riemannian_gradient_norm == pytest.approx(np.linalg.norm(h), abs=3e-12)


def test_stationary_single_orbital_stops_despite_nonzero_ambient_euclidean_gradient():
    case = _case(frozen=(), nactive=1, complex_gauge=True)
    result = _run(case)
    assert result.converged and result.diagnostics.accepted_steps == 0
    _, ambient, _ = _score(case, result.gauges_copy())
    assert np.linalg.norm(ambient) == pytest.approx(8.0, abs=2e-12)
    assert result.diagnostics.riemannian_gradient_norm == 0.0
    assert result.diagnostics.final_objective == pytest.approx(1.0, abs=3e-13)


def test_exact_simultaneous_peak_borrowed_owners_and_node_inventory():
    case = _case((3, 1, 1))
    plan = _plan(case)
    k, o, r = plan.n_points, plan.n_active, plan.n_minimal
    assert plan.fixed_optimizer_bytes == 80 * k * o**2 + 8 * k * o
    assert plan.exponential_workspace_bytes == 32 * o**2 + 24 * o
    assert plan.peak_owned_numerical_bytes == 112 * k * o**2 + 16 * k * o + 32 * o**2 + 32 * o + 72 * r * o
    assert plan.output_numerical_bytes == 16 * k * o**2 + 8 * k * o
    assert plan.borrowed_seed_numerical_bytes == case["seed"].memory.output_numerical_bytes
    assert plan.live_iao_numerical_bytes == sum(point.memory.output_numerical_bytes for point in case["points"])
    dims = case["reference"].dimensions
    assert plan.required_node_memory_bytes == (dims.external_bytes + dims.shared_bytes + dims.per_rank_bytes
        + dims.localization_window_bytes_per_rank + plan.peak_owned_numerical_bytes
        + plan.borrowed_seed_numerical_bytes + plan.live_iao_numerical_bytes + plan.borrowed_owner_pointer_bytes)
    assert plan.maximum_work_units == (plan.initial_work_units
        + _options().maximum_iterations * _options().maximum_line_search_trials * plan.work_units_per_trial)


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_work_units"])
def test_caps_precede_invalid_later_point_owner(field):
    case = _case((3, 1, 1))
    plan = _plan(case)
    for value in (0, (plan.peak_owned_numerical_bytes if field == "maximum_owned_numerical_bytes" else plan.initial_work_units) - 1):
        caps = _caps(plan, **{field: value})
        with pytest.raises((ValueError, RuntimeError), match="cap"):
            _run(case, caps=caps, points=case["points"][::-1])


def test_charge_frame_time_reversal_failure_is_not_silently_real_projected():
    case = _case((3, 1, 1))
    cross, minimal, labels = case["inputs"][1]
    broken = cross.copy()
    broken[:, 0] *= np.exp(0.37j)
    points = list(case["points"])
    points[1] = _iao(case["reference"], broken, minimal, labels, point=1)
    with pytest.raises(ValueError, match="charge frames.*time reversal"):
        _run(case, points=points)


def test_same_content_different_state_owner_and_wrong_point_order_are_rejected():
    case = _case((3, 1, 1))
    other = _case((3, 1, 1))
    assert other["reference"].state.state_identity_sha256 == case["reference"].state.state_identity_sha256
    with pytest.raises(ValueError, match="exact admitted state owner"):
        _run(case, seed=other["seed"])
    with pytest.raises(ValueError, match="IAO owners"):
        _run(case, points=other["points"])
    with pytest.raises(ValueError, match="IAO owners"):
        _run(case, points=case["points"][::-1])


def test_progress_exception_aborts_and_control_mutation_cannot_change_active_options():
    case = _case()
    def abort(event):
        raise RuntimeError("consumer progress failure")
    with pytest.raises(RuntimeError, match="consumer progress failure"):
        _run(case, callback=abort)
    np.testing.assert_array_equal(_gauges(case["seed"]), case["u"])
    controls = _options(maximum_iterations=1)
    def change_original_options(event):
        controls.maximum_iterations = 192
    result = _run(case, options=controls, callback=change_original_options)
    assert result.status == core._PeriodicCorrelationIAOOptimizerStatus.ITERATION_LIMIT
    assert result.diagnostics.accepted_steps == 1


@pytest.mark.parametrize("replace", [False, True])
def test_progress_mutation_of_original_point_list_cannot_release_native_owners(replace):
    case = _case((3, 1, 1))
    options = _options(maximum_iterations=1)
    expected = _run(case, options=options)
    caller_points = case.pop("points")
    events = []
    def mutate_original_list(event):
        events.append(event)
        caller_points.clear()
        if replace:
            caller_points.extend([None] * 3)
        gc.collect()
    actual = _run(case, options=options, points=caller_points, callback=mutate_original_list)
    assert events[0].event == core._PeriodicCorrelationIAOOptimizerEvent.INITIAL
    assert events[-1].event == core._PeriodicCorrelationIAOOptimizerEvent.FINISHED
    assert actual.status == expected.status
    assert actual.diagnostics.accepted_steps == expected.diagnostics.accepted_steps == 1
    assert actual.diagnostics.final_objective == expected.diagnostics.final_objective
    np.testing.assert_array_equal(actual.gauges_copy(), expected.gauges_copy())


def test_result_owns_only_accepted_gauges_map_and_state_with_independent_digest_wire():
    case = _case()
    controls = _options(maximum_iterations=1)
    result = _run(case, options=controls)
    gauges = result.gauges_copy()
    domain = b"vibeqc.periodic.correlation.iao-optimizer.gauges"
    wire = struct.pack(">Q", len(domain)) + domain + struct.pack(">IQQ", 1, len(gauges), gauges.shape[1])
    lanes = gauges.reshape(-1).view(float).copy()
    lanes[lanes == 0] = 0.0
    wire += b"".join(struct.pack(">d", lane) for lane in lanes)
    wire += b"".join(struct.pack(">Q", int(band)) for _ in gauges for band in case["active"])
    assert result.gauge_payload_sha256 == hashlib.sha256(wire).hexdigest()
    assert len(result.optimizer_identity_sha256) == 64
    assert len(result.source_identity_sha256) == 64
    del case
    gc.collect()
    np.testing.assert_array_equal(result.gauges_copy(), gauges)
    changed = result.gauges_copy()
    changed[:] = 0
    np.testing.assert_array_equal(result.gauges_copy(), gauges)
    assert result.state.n_frozen_core == 1
    with pytest.raises(IndexError):
        result.gauge(1, 0, 0)


@pytest.mark.parametrize("changes", [
    {"maximum_iterations": 0}, {"maximum_line_search_trials": 0}, {"backtracking_factor": 1.0},
    {"initial_step": -1.0}, {"minimum_step": 0.2}, {"jacobi_max_sweeps": 0},
    {"riemannian_gradient_tolerance": np.nan}, {"absolute_tolerance": 0.0, "relative_tolerance": 0.0},
])
def test_invalid_controls_fail_closed_without_iteration(changes):
    case = _case()
    with pytest.raises(ValueError):
        _plan(case, _options(**changes))


def test_iteration_work_count_overflow_is_count_only():
    case = _case()
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        _plan(case, _options(maximum_iterations=2**64 - 1))
