"""Bounded local-domain Jacobi reference; tiny synthetic Hamiltonians only."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_bounded_restricted_ccsd_target import _legacy


def _problem(o=2, v=2, seed=734):
    rng = np.random.default_rng(seed)
    n = o + v
    raw = 0.018 * rng.normal(size=(3, n, n))
    factors = np.ascontiguousarray(raw + raw.transpose(0, 2, 1))
    raw_f = 0.008 * rng.normal(size=(n, n))
    fock = raw_f + raw_f.T
    np.fill_diagonal(fock, np.r_[-np.linspace(1.1, 1.7, o), np.linspace(0.5, 1.2, v)])
    return dict(o=o, v=v, fock=np.ascontiguousarray(fock), factors=factors,
                eri=np.ascontiguousarray(np.einsum("Ppq,Prs->pqrs", factors, factors)),
                t1=np.zeros((o, v)), t2=np.zeros((o, o, v, v)))


def _options(**changes):
    options = core._BoundedRestrictedCCSDSolverOptions()
    options.maximum_iterations = 80
    options.denominator_floor = 1e-8
    options.singles_residual_tolerance = 1e-12
    options.doubles_residual_tolerance = 1e-12
    options.energy_tolerance = 1e-13
    options.input_symmetry_tolerance = 1e-13
    for name, value in changes.items():
        setattr(options, name, value)
    return options


def _inventory(replicas=1, external=0):
    inventory = core._BoundedRestrictedCCSDSolverInventory()
    inventory.numerical_replicas = replicas
    inventory.external_node_numerical_bytes = external
    return inventory


def _plan(problem, options=None, inventory=None, supplied=False, transient=0):
    options = _options() if options is None else options
    inventory = _inventory() if inventory is None else inventory
    return core._plan_bounded_restricted_ccsd_solver(
        problem["o"], problem["v"], options.maximum_iterations, supplied,
        inventory, problem["eri"].nbytes, transient)


def _caps(plan):
    caps = core._BoundedRestrictedCCSDSolverCaps()
    caps.maximum_owned_numerical_bytes = plan.peak_owned_numerical_bytes
    caps.maximum_total_numerical_bytes = plan.total_live_numerical_bytes
    caps.maximum_node_numerical_bytes = plan.required_node_numerical_bytes
    caps.maximum_integral_calls = plan.integral_calls_upper_bound
    caps.maximum_kernel_work_units = plan.kernel_work_units_upper_bound
    return caps


def _run(problem, *, options=None, inventory=None, caps=None, supplied=False, **kwargs):
    options = _options() if options is None else options
    inventory = _inventory() if inventory is None else inventory
    if caps is None:
        caps = _caps(_plan(problem, options, inventory, supplied,
                           kwargs.get("provider_transient_bytes", 0)))
    o, f = problem["o"], problem["fock"]
    if supplied:
        kwargs.setdefault("initial_t1", problem["t1"])
        kwargs.setdefault("initial_t2", problem["t2"])
    return core._bounded_restricted_ccsd_solve_diagnostic(
        np.ascontiguousarray(f[:o, :o]), np.ascontiguousarray(f[o:, o:]),
        np.ascontiguousarray(f[:o, o:]), problem["eri"], options, inventory, caps, **kwargs)


def _energy(problem, t1, t2):
    o = problem["o"]
    g = problem["eri"][:o, o:, :o, o:].transpose(0, 2, 1, 3)
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    return 2 * np.einsum("ia,ia->", problem["fock"][:o, o:], t1) + np.einsum(
        "ijab,ijab->", 2 * g - g.transpose(0, 1, 3, 2), tau)


def _legacy_jacobi(problem):
    p = dict(problem)
    p["t1"], p["t2"] = np.zeros_like(p["t1"]), np.zeros_like(p["t2"])
    o = p["o"]
    eo, ev = np.diag(p["fock"])[:o], np.diag(p["fock"])[o:]
    d1 = eo[:, None] - ev[None, :]
    d2 = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
    previous = None
    for _ in range(80):
        r1, r2 = _legacy(p)
        energy = _energy(p, p["t1"], p["t2"])
        if previous is not None and abs(energy - previous) < 1e-14 and max(np.max(abs(r1)), np.max(abs(r2))) < 1e-13:
            return p["t1"], p["t2"], energy
        p["t1"] = np.ascontiguousarray(p["t1"] + r1 / d1)
        p["t2"] = np.ascontiguousarray(p["t2"] + r2 / d2)
        previous = energy
    raise AssertionError("tiny independent legacy Jacobi oracle did not converge")


@pytest.mark.parametrize("o,v", [(1, 1), (1, 3), (2, 2), (3, 2)])
def test_weak_reference_converges_against_legacy_and_verifies_returned_residual(o, v):
    problem = _problem(o, v)
    result = _run(problem)
    assert result.final_snapshot.converged
    assert 2 <= result.final_snapshot.iteration < 80
    t1, t2, energy = _legacy_jacobi(problem)
    np.testing.assert_allclose(result.t1, t1, atol=2e-12, rtol=1e-10)
    np.testing.assert_allclose(result.t2, t2, atol=2e-12, rtol=1e-10)
    assert result.final_snapshot.correlation_energy == pytest.approx(energy, abs=2e-13)
    assert np.linalg.norm(result.t1) > 1e-4  # Exercise the nonzero singles energy.
    r1, r2 = _legacy(dict(problem, t1=result.t1, t2=result.t2))
    record = result.final_snapshot
    assert record.singles_max_residual == pytest.approx(np.max(abs(r1)), abs=2e-14)
    assert record.doubles_max_residual == pytest.approx(np.max(abs(r2)), abs=2e-14)
    assert record.singles_residual_norm == pytest.approx(np.linalg.norm(r1), abs=3e-14)
    assert record.doubles_residual_norm == pytest.approx(np.linalg.norm(r2), abs=3e-14)
    assert record.correlation_energy == pytest.approx(_energy(problem, result.t1, result.t2), abs=2e-15)
    assert record.target_evaluations == record.iteration * o**2
    leaf_calls = result.memory.target.integral_calls_upper_bound - 2 * o * v**3 + v**2
    assert record.integral_calls == record.iteration * (o**2 * leaf_calls + 2 * o**2 * v**2)


@pytest.mark.parametrize("o,v", [(1, 2), (2, 2)])
def test_independent_spin_orbital_converged_fixed_point(o, v):
    from vibeqc.dlpno._ccsd_ref import _spin_orbital_eri, so_energy, so_residuals

    problem = _problem(o, v, seed=881)
    result = _run(problem)
    n = o + v
    f = np.zeros((2 * n, 2 * n))
    f[0::2, 0::2] = problem["fock"]
    f[1::2, 1::2] = problem["fock"]
    off = f.copy()
    np.fill_diagonal(off, 0)
    eri = _spin_orbital_eri(problem["factors"], n)
    occupied, virtual = slice(0, 2 * o), slice(2 * o, 2 * n)
    eo, ev = np.diag(f)[occupied], np.diag(f)[virtual]
    d1 = eo[:, None] - ev[None, :]
    d2 = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
    t1, t2 = np.zeros((2 * o, 2 * v)), np.zeros((2 * o, 2 * o, 2 * v, 2 * v))
    for _ in range(80):
        r1, r2 = so_residuals(f, off, eri, t1, t2, occupied, virtual, d1, d2)
        if max(np.max(abs(r1)), np.max(abs(r2))) < 1e-13:
            break
        t1, t2 = t1 + r1 / d1, t2 + r2 / d2
    else:
        raise AssertionError("tiny spin-orbital CCSD oracle did not converge")
    np.testing.assert_allclose(result.t1, t1[0::2, 0::2], atol=2e-12)
    np.testing.assert_allclose(result.t2, t2[0::2, 1::2, 0::2, 1::2], atol=2e-12)
    assert result.final_snapshot.correlation_energy == pytest.approx(
        so_energy(f, eri, t1, t2, occupied, virtual), abs=2e-13)


def test_pure_one_body_nonzero_singles_matches_exact_two_electron_energy():
    problem = _problem(1, 1)
    problem["factors"][:] = 0
    problem["eri"][:] = 0
    result = _run(problem)
    expected = 2 * (np.linalg.eigvalsh(problem["fock"])[0] - problem["fock"][0, 0])
    assert result.final_snapshot.converged
    assert result.final_snapshot.correlation_energy == pytest.approx(expected, abs=2e-14)
    np.testing.assert_allclose(result.t2, 0, atol=1e-14)


@pytest.mark.parametrize("supplied", [False, True])
def test_exact_current_candidate_leaf_and_replica_inventory(supplied):
    problem = _problem(2, 3)
    inventory = _inventory(3, 257)
    p = _plan(problem, inventory=inventory, supplied=supplied, transient=113)
    o, v = 2, 3
    amplitudes = 8 * (o * v + o**2 * v**2)
    assert p.amplitude_snapshot_bytes == p.candidate_snapshot_bytes == amplitudes
    assert p.borrowed_initial_amplitude_bytes == (amplitudes if supplied else 0)
    assert p.borrowed_fock_bytes == 8 * (o**2 + v**2 + o * v)
    assert p.peak_owned_numerical_bytes == 2 * amplitudes + p.target.peak_owned_numerical_bytes
    assert p.total_live_numerical_bytes == p.peak_owned_numerical_bytes + p.borrowed_fock_bytes + p.borrowed_initial_amplitude_bytes + problem["eri"].nbytes + 113
    assert p.required_node_numerical_bytes == 257 + 3 * p.total_live_numerical_bytes
    assert p.target_evaluations_upper_bound == 80 * o**2
    assert p.integral_calls_upper_bound == 80 * (o**2 * p.target.integral_calls_upper_bound + 2 * o**2 * v**2)
    assert p.kernel_work_units_upper_bound > 80 * o**2 * p.target.kernel_work_units_upper_bound
    result = _run(problem, inventory=inventory, supplied=supplied, provider_transient_bytes=113)
    assert result.final_snapshot.converged


@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_exhaustion_returns_evaluated_snapshot_not_the_next_candidate(iterations):
    problem = _problem()
    options = _options(maximum_iterations=iterations, singles_residual_tolerance=1e-30,
                       doubles_residual_tolerance=1e-30, energy_tolerance=1e-30)
    result = _run(problem, options=options)
    assert not result.final_snapshot.converged
    assert result.final_snapshot.iteration == iterations
    p = dict(problem)
    o = p["o"]
    eo, ev = np.diag(p["fock"])[:o], np.diag(p["fock"])[o:]
    d1 = eo[:, None] - ev[None, :]
    d2 = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
    for _ in range(iterations - 1):
        r1, r2 = _legacy(p)
        p = dict(p, t1=np.ascontiguousarray(p["t1"] + r1 / d1), t2=np.ascontiguousarray(p["t2"] + r2 / d2))
    np.testing.assert_allclose(result.t1, p["t1"], atol=2e-14)
    np.testing.assert_allclose(result.t2, p["t2"], atol=2e-14)
    assert result.final_snapshot.correlation_energy == pytest.approx(_energy(p, p["t1"], p["t2"]), abs=2e-15)


def test_supplied_initial_snapshot_and_final_copy_ownership():
    problem = _problem()
    rng = np.random.default_rng(224)
    problem["t1"][:] = rng.normal(size=problem["t1"].shape) * 0.01
    raw = rng.normal(size=problem["t2"].shape) * 0.005
    problem["t2"][:] = raw + raw.transpose(1, 0, 3, 2)
    originals = {key: problem[key].copy() for key in ("t1", "t2", "fock", "eri")}
    initial = _run(problem, supplied=True, options=_options(maximum_iterations=1))
    np.testing.assert_array_equal(initial.t1, problem["t1"])
    np.testing.assert_array_equal(initial.t2, problem["t2"])
    result = _run(problem, supplied=True)
    zero = _run(problem)
    np.testing.assert_allclose(result.t1, zero.t1, atol=2e-12)
    np.testing.assert_allclose(result.t2, zero.t2, atol=2e-12)
    for key, original in originals.items():
        np.testing.assert_array_equal(problem[key], original)
    saved = result.t1.copy()
    copy = result.t1
    copy[:] = 123
    del problem
    np.testing.assert_array_equal(result.t1, saved)


def test_live_progress_is_scalar_same_snapshot_and_can_cancel():
    problem = _problem()
    events = []
    result = _run(problem, inventory=_inventory(external=32768), progress=events.append)
    assert len(events) == result.final_snapshot.iteration
    assert not events[0].has_previous_energy and events[0].energy_change == 0
    assert events[0].correlation_energy == 0
    assert [p.iteration for p in events] == list(range(1, len(events) + 1))
    assert all(left.integral_calls < right.integral_calls for left, right in zip(events, events[1:]))
    assert events[-1].converged
    assert events[-1].correlation_energy == result.final_snapshot.correlation_energy

    def cancel(record):
        if record.iteration == 2:
            raise RuntimeError("intentional scalar-progress cancellation")

    with pytest.raises(RuntimeError, match="intentional scalar-progress cancellation"):
        _run(problem, progress=cancel)
    assert _run(problem).final_snapshot.converged


def test_initial_exact_solution_requires_two_energy_snapshots():
    problem = _problem(1, 1)
    problem["eri"][:] = 0
    problem["fock"][0, 1] = problem["fock"][1, 0] = 0
    one = _run(problem, options=_options(maximum_iterations=1))
    assert not one.final_snapshot.converged
    two = _run(problem, options=_options(maximum_iterations=2))
    assert two.final_snapshot.converged
    assert two.final_snapshot.correlation_energy == 0
    assert two.final_snapshot.singles_max_residual == two.final_snapshot.doubles_max_residual == 0


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_total_numerical_bytes",
                                  "maximum_node_numerical_bytes", "maximum_integral_calls", "maximum_kernel_work_units"])
@pytest.mark.parametrize("zero", [False, True])
def test_caps_reject_before_input_scan_or_callbacks(field, zero):
    problem = _problem()
    inventory = _inventory(2, 37)
    caps = _caps(_plan(problem, inventory=inventory))
    setattr(caps, field, 0 if zero else getattr(caps, field) - 1)
    problem["fock"][0, 0] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(problem, inventory=inventory, caps=caps, fail_before_call=0)


@pytest.mark.parametrize("fail_before", [0, 5, 1000])
def test_integral_failure_aborts_without_mutating_inputs(fail_before):
    problem = _problem()
    original = problem["t2"].copy()
    with pytest.raises(RuntimeError, match="injected integral callback failure"):
        _run(problem, supplied=True, fail_before_call=fail_before)
    np.testing.assert_array_equal(problem["t2"], original)


def test_nonfinite_integral_is_rejected():
    problem = _problem()
    problem["eri"][:] = np.nan
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="finite"):
        _run(problem)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
@pytest.mark.parametrize("field", ["denominator_floor", "singles_residual_tolerance",
                                  "doubles_residual_tolerance", "energy_tolerance"])
def test_explicit_positive_finite_options(field, bad):
    with pytest.raises(ValueError, match="positive finite"):
        _run(_problem(), options=_options(**{field: bad}), fail_before_call=0)


@pytest.mark.parametrize("field", ["fock", "t1", "t2"])
def test_finite_input_preflight(field):
    problem = _problem()
    problem[field].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _run(problem, supplied=True, fail_before_call=0)


@pytest.mark.parametrize("which", ["occupied", "virtual", "amplitude"])
def test_asymmetric_inputs_rejected_not_repaired(which):
    problem = _problem()
    if which == "occupied":
        problem["fock"][0, 1] += 0.1
    elif which == "virtual":
        problem["fock"][2, 3] += 0.1
    else:
        problem["t2"][0, 1, 0, 1] += 0.1
    with pytest.raises(ValueError, match="symmetry audit"):
        _run(problem, supplied=True, fail_before_call=0)


@pytest.mark.parametrize("gap", [-1.0, 0.0, 1e-8])
def test_small_nonpositive_or_floor_equal_gaps_rejected_before_provider(gap):
    problem = _problem(1, 1)
    problem["fock"][0, 0] = 0.0
    problem["fock"][1, 1] = gap
    with pytest.raises(ValueError, match="strictly exceed"):
        _run(problem, fail_before_call=0)


def test_double_gap_overflow_is_not_a_shift_or_false_convergence():
    problem = _problem(1, 1)
    problem["fock"][0, 0] = 0.0
    problem["fock"][1, 1] = np.finfo(float).max
    with pytest.raises(ValueError, match="denominator"):
        _run(problem, fail_before_call=0)


@pytest.mark.parametrize("o,v,iterations,replicas", [(0, 1, 1, 1), (1, 0, 1, 1),
    (1, 1, 0, 1), (1, 1, 1, 0), (2**63, 2, 2, 1), (2, 2, 2**63, 1)])
def test_plan_invalid_or_overflow_counts(o, v, iterations, replicas):
    with pytest.raises((ValueError, OverflowError, RuntimeError)):
        core._plan_bounded_restricted_ccsd_solver(o, v, iterations, False, _inventory(replicas))


def test_diagnostic_rejects_half_initial_pair_and_bad_layout():
    problem = _problem()
    with pytest.raises(ValueError, match="both"):
        _run(problem, initial_t1=problem["t1"])
    with pytest.raises(ValueError, match="contiguous"):
        _run(problem, supplied=True, initial_t1=np.asfortranarray(problem["t1"]))
