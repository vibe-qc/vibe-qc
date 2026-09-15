"""Coupled common-space (T) against independent linear algebra and spin sums."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_bounded_restricted_triples_moments import _case, _einsum_oracle


def _problem(o=2, v=3):
    case = _case(o, v, seed=983)
    rng = np.random.default_rng(742)
    raw = 0.035 * rng.normal(size=(o, o))
    case["foo"] = np.ascontiguousarray(raw + raw.T - np.diag(np.linspace(0.9, 1.7, o)))
    case["fvv"] = np.diag(np.linspace(0.3, 1.2, v))
    return case


def _options(**changes):
    options = core._BoundedRestrictedTriplesSolverOptions()
    options.maximum_iterations = 80
    options.denominator_floor = 1e-8
    options.residual_tolerance = 1e-12
    options.energy_tolerance = 1e-13
    options.input_symmetry_tolerance = 1e-13
    for name, value in changes.items():
        setattr(options, name, value)
    return options


def _inventory(replicas=1, external=0):
    inventory = core._BoundedRestrictedTriplesSolverInventory()
    inventory.numerical_replicas = replicas
    inventory.external_node_numerical_bytes = external
    return inventory


def _plan(case, options=None, inventory=None, transient=0):
    return core._plan_bounded_restricted_triples_solver(
        case["o"], case["v"], (_options() if options is None else options).maximum_iterations,
        _inventory() if inventory is None else inventory, case["eri"].nbytes, transient)


def _caps(plan):
    caps = core._BoundedRestrictedTriplesSolverCaps()
    caps.maximum_owned_numerical_bytes = plan.peak_owned_numerical_bytes
    caps.maximum_total_numerical_bytes = plan.total_live_numerical_bytes
    caps.maximum_node_numerical_bytes = plan.required_node_numerical_bytes
    caps.maximum_integral_calls = plan.integral_calls_upper_bound
    caps.maximum_kernel_work_units = plan.kernel_work_units_upper_bound
    return caps


def _run(case, options=None, inventory=None, caps=None, snapshot=81, **kwargs):
    options = _options() if options is None else options
    inventory = _inventory() if inventory is None else inventory
    if caps is None:
        caps = _caps(_plan(case, options, inventory, kwargs.get("provider_transient_bytes", 0)))
    return core._bounded_restricted_triples_solve_diagnostic(
        case["t1"], case["t2"], case["foo"], case["fvv"], case["fov"], case["eri"],
        snapshot, options, inventory, caps, **kwargs)


def _moments(case):
    o, v = case["o"], case["v"]
    w, u = np.empty((o, o, o, v, v, v)), np.empty((o, o, o, v, v, v))
    for indices in itertools.product(range(o), repeat=3):
        w[indices], u[indices] = _einsum_oracle(case, indices)
    return w, u


def _operator(case, t):
    f, ev = case["foo"], np.diag(case["fvv"])
    vsum = ev[:, None, None] + ev[None, :, None] + ev[None, None, :]
    return (vsum * t - np.einsum("il,ljkabc->ijkabc", f, t)
            - np.einsum("jl,ilkabc->ijkabc", f, t)
            - np.einsum("kl,ijlabc->ijkabc", f, t))


def _energy(t, w, u):
    adapted = (4 * t - 2 * t.transpose(0, 1, 2, 3, 5, 4)
               - 2 * t.transpose(0, 1, 2, 5, 4, 3) - 2 * t.transpose(0, 1, 2, 4, 3, 5)
               + t.transpose(0, 1, 2, 5, 3, 4) + t.transpose(0, 1, 2, 4, 5, 3))
    return sum((2 - int(i == j) - int(j == k)) * np.sum(adapted[i, j, k] * (w[i, j, k] + u[i, j, k]))
               for i, j, k in itertools.combinations_with_replacement(range(len(t)), 3))


def _linear_oracle(case, w):
    o, v, f = case["o"], case["v"], case["foo"]
    identity = np.eye(o)
    foo3 = (np.kron(np.kron(f, identity), identity) + np.kron(np.kron(identity, f), identity)
            + np.kron(np.kron(identity, identity), f))
    ev = np.diag(case["fvv"])
    t = np.empty_like(w)
    for a, b, c in itertools.product(range(v), repeat=3):
        t[:, :, :, a, b, c] = np.linalg.solve(
            (ev[a] + ev[b] + ev[c]) * np.eye(o**3) - foo3, -w[:, :, :, a, b, c].ravel()).reshape(o, o, o)
    return t


@pytest.mark.parametrize("o,v", [(1, 1), (1, 3), (2, 2), (2, 3), (3, 2)])
def test_coupled_fixed_point_matches_independent_linear_system_and_returned_residual(o, v):
    case = _problem(o, v)
    result = _run(case)
    record = result.final_snapshot
    assert record.converged
    w, u = _moments(case)
    t = _linear_oracle(case, w)
    np.testing.assert_allclose(result.amplitudes, t, atol=5e-13, rtol=1e-10)
    residual = w + _operator(case, result.amplitudes)
    assert record.maximum_absolute_residual == pytest.approx(np.max(abs(residual)), abs=2e-16)
    assert record.residual_frobenius_norm == pytest.approx(np.linalg.norm(residual), abs=5e-16)
    assert record.triples_energy == pytest.approx(_energy(t, w, u), abs=2e-13)
    assert record.target_evaluations == record.iteration * o**3
    assert record.integral_calls == record.iteration * o**3 * v**3 * (6 * (o + v) + 3)
    assert record.neighbour_visits == record.iteration * o**3 * 3 * (o - 1)
    assert result.ccsd_snapshot_id == 81


def _canonical_spin_energy(case):
    from vibeqc.dlpno._ccsd_ref import _spin_orbital_eri, so_triples_correction

    o, v = case["o"], case["v"]
    eo, rotation = np.linalg.eigh(case["foo"])
    full = np.eye(o + v)
    full[:o, :o] = rotation
    factors = np.einsum("pi,Ppq,qj->Pij", full, case["factors"], full)
    t1 = rotation.T @ case["t1"]
    t2 = np.einsum("pi,qj,pqab->ijab", rotation, rotation, case["t2"])
    t1so = np.zeros((2 * o, 2 * v))
    t1so[0::2, 0::2] = t1so[1::2, 1::2] = t1
    t2so = np.zeros((2 * o, 2 * o, 2 * v, 2 * v))
    for si, sj in itertools.product((0, 1), repeat=2):
        t2so[si::2, sj::2, si::2, sj::2] += t2
        t2so[si::2, sj::2, sj::2, si::2] -= t2.transpose(0, 1, 3, 2)
    return so_triples_correction(np.repeat(np.r_[eo, np.diag(case["fvv"])], 2),
        _spin_orbital_eri(factors, o + v), t1so, t2so, slice(0, 2 * o), slice(2 * o, 2 * (o + v)))


def test_occupied_rotation_recovers_canonical_spin_orbital_triples_not_t0():
    case = _problem(2, 3)
    result = _run(case)
    energy = _canonical_spin_energy(case)
    assert result.final_snapshot.triples_energy == pytest.approx(energy, abs=3e-13)
    diagonal = dict(case, foo=np.diag(np.diag(case["foo"])))
    assert abs(_run(diagonal).final_snapshot.triples_energy - energy) > 1e-7


def test_native_ccsd_then_coupled_triples_equation_chain():
    from tests.test_bounded_restricted_ccsd_solver import _problem as ccsd_problem, _run as ccsd_run

    problem = ccsd_problem(2, 3, seed=647)
    o = problem["o"]
    problem["fock"][:o, o:] = 0
    problem["fock"][o:, :o] = 0
    problem["fock"][o:, o:] = np.diag(np.diag(problem["fock"])[o:])
    ccsd = ccsd_run(problem)
    assert ccsd.final_snapshot.converged
    case = dict(problem, t1=ccsd.t1, t2=ccsd.t2,
        foo=np.ascontiguousarray(problem["fock"][:o, :o]),
        fvv=np.ascontiguousarray(problem["fock"][o:, o:]), fov=np.zeros((o, problem["v"])))
    triples = _run(case, snapshot=ccsd.final_snapshot.iteration)
    assert triples.final_snapshot.converged
    assert triples.ccsd_snapshot_id == ccsd.final_snapshot.iteration
    assert np.linalg.norm(ccsd.t1) > 1e-8
    assert triples.final_snapshot.triples_energy == pytest.approx(_canonical_spin_energy(case), abs=1e-17, rel=1e-7)
    # This exercises the native equation chain on a tiny supplied Hamiltonian,
    # not physical Gaussian-source matching or a periodic per-cell driver.


@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_exhaustion_returns_evaluated_snapshot_and_scalar_progress(iterations):
    case = _problem()
    events = []
    options = _options(maximum_iterations=iterations, residual_tolerance=1e-15, energy_tolerance=1e-15)
    result = _run(case, options=options, progress=events.append)
    assert not result.final_snapshot.converged
    w, u = _moments(case)
    t = np.zeros_like(w)
    o, v = case["o"], case["v"]
    eo, ev = np.diag(case["foo"]), np.diag(case["fvv"])
    gap = (ev[None, None, None, :, None, None] + ev[None, None, None, None, :, None]
           + ev[None, None, None, None, None, :] - eo[:, None, None, None, None, None]
           - eo[None, :, None, None, None, None] - eo[None, None, :, None, None, None])
    previous = 0.0
    assert len(events) == iterations
    for index, event in enumerate(events):
        energy = _energy(t, w, u)
        assert event.iteration == index + 1
        assert event.has_previous_energy == (index > 0)
        assert event.triples_energy == pytest.approx(energy, abs=2e-15)
        assert event.energy_change == pytest.approx(energy - previous if index else 0.0, abs=2e-15)
        if index + 1 != iterations:
            t -= (w + _operator(case, t)) / gap
        previous = energy
    np.testing.assert_allclose(result.amplitudes, t, atol=2e-16)
    assert result.amplitudes.shape == (o, o, o, v, v, v)


def test_exact_peak_and_replica_inventory():
    case = _problem()
    p = _plan(case, inventory=_inventory(3, 127), transient=31)
    o, v = case["o"], case["v"]
    assert p.amplitude_snapshot_bytes == p.candidate_snapshot_bytes == 8 * o**3 * v**3
    assert p.orbital_workspace_bytes == 8 * (v**2 + v + 3 * o)
    assert p.peak_owned_numerical_bytes == 16 * o**3 * v**3 + 32 * v**3 + 24 * v**2 + 8 * v + 24 * o
    assert p.borrowed_input_bytes == 8 * (o**2 + v**2 + 2 * o * v + o**2 * v**2)
    assert p.total_live_numerical_bytes == p.peak_owned_numerical_bytes + p.borrowed_input_bytes + case["eri"].nbytes + 31
    assert p.required_node_numerical_bytes == 127 + 3 * p.total_live_numerical_bytes


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_total_numerical_bytes",
    "maximum_node_numerical_bytes", "maximum_integral_calls", "maximum_kernel_work_units"])
@pytest.mark.parametrize("zero", [False, True])
def test_caps_precede_nonfinite_input_scans_and_callbacks(field, zero):
    case = _problem()
    caps = _caps(_plan(case))
    setattr(caps, field, 0 if zero else getattr(caps, field) - 1)
    case["t1"][0, 0] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(case, caps=caps, fail_before_call=0)


def test_transient_provider_storage_is_charged_once_per_replica():
    case = _problem()
    with pytest.raises((ValueError, RuntimeError), match="byte cap"):
        _run(case, caps=_caps(_plan(case)), provider_transient_bytes=1)
    assert _run(case, provider_transient_bytes=1).final_snapshot.converged


@pytest.mark.parametrize("field", ["t1", "t2", "foo", "fvv", "fov"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_all_input_lanes_are_scanned_before_provider(field, bad):
    case = _problem()
    case[field].flat[-1] = bad
    with pytest.raises(ValueError, match="finite"):
        _run(case, fail_before_call=0)


@pytest.mark.parametrize("bad", [0.1, -0.1, np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0)])
@pytest.mark.parametrize("field", ["fov", "fvv"])
def test_exact_brillouin_and_quasi_canonical_scope(field, bad):
    case = _problem()
    case[field][0, 1] = bad
    with pytest.raises(ValueError, match="exactly"):
        _run(case, fail_before_call=0)


@pytest.mark.parametrize("field", ["foo", "t2"])
def test_asymmetry_is_not_silently_repaired(field):
    case = _problem()
    case[field][0, 1] += 0.1
    with pytest.raises(ValueError, match="symmetry audit"):
        _run(case, fail_before_call=0)


@pytest.mark.parametrize("failure", [0, 50, 1500, 7000])
def test_provider_failure_does_not_mutate_inputs(failure):
    case = _problem()
    saved = {key: value.copy() for key, value in case.items() if isinstance(value, np.ndarray)}
    with pytest.raises(RuntimeError, match="injected triples integral"):
        _run(case, fail_before_call=failure)
    for key, value in saved.items():
        np.testing.assert_array_equal(case[key], value)


def test_progress_exception_aborts_and_outputs_do_not_alias():
    case = _problem()

    def stop(event):
        assert event.iteration == 1
        raise RuntimeError("stop triples now")

    with pytest.raises(RuntimeError, match="stop triples now"):
        _run(case, progress=stop)
    result = _run(case)
    amplitudes = result.amplitudes.copy()
    result.amplitudes[:] = 0
    del case
    np.testing.assert_array_equal(result.amplitudes, amplitudes)


def test_nonpositive_gap_and_nonfinite_product_are_explicit_failures():
    case = _problem()
    case["fvv"][:] = 0
    case["foo"][:] = np.eye(case["o"])
    with pytest.raises(ValueError, match="denominator"):
        _run(case, fail_before_call=0)
    case = _problem()
    case["t2"][:] = np.finfo(float).max
    case["eri"][:] = 2.0
    with pytest.raises(OverflowError, match="finite"):
        _run(case)


@pytest.mark.parametrize("changes", [dict(maximum_iterations=0), dict(denominator_floor=0),
    dict(residual_tolerance=0), dict(energy_tolerance=np.nan), dict(input_symmetry_tolerance=-1)])
def test_invalid_controls(changes):
    with pytest.raises(ValueError):
        _run(_problem(), options=_options(**changes), fail_before_call=0)


@pytest.mark.parametrize("o,v", [(0, 1), (1, 0), (2**40, 2**20), (1, 2**63)])
def test_impossible_plan_refuses_without_allocating(o, v):
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        core._plan_bounded_restricted_triples_solver(o, v, 2, _inventory())
