"""Actual finite-source pair-space CCSD, not production DLPNO or triples.

The native constructor receives no caller numerical arrays. Dense factors,
expanded amplitudes and spin-orbital residuals below are tiny independent
test oracles only. Singles and doubles use separate explicit PNO cutoffs.
"""

from __future__ import annotations

import gc
from itertools import combinations_with_replacement

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_pair_mp2 import (
    _physical, _run as _mp2_run, _controls as _mp2_controls,
)
from tests.test_periodic_gaussian_pair_pnos import (
    _options as _pno_options, _caps as _pno_caps,
)
from tests.test_bounded_restricted_pair_ccsd_solver import (
    _options as _solver_options, _expand, _project, _projected_norms,
    _jacobi_oracle, _spin_residual,
)
from tests.test_bounded_restricted_ccsd_target import _legacy
from tests.test_bounded_restricted_ccsd_solver import _run as _dense_run, _energy
from tests.test_periodic_gaussian_selected_local_ccsd_t import (
    _dense_case, _prepare_leaves, _he2_controls,
)
from tests.test_periodic_gaussian_frozen_core_correlation import (
    _frozen_bundle, _full_core_hamiltonian, _two_electron_fci,
)
from tests.test_periodic_gaussian_real_local_provider import _make as _provider


def _controls(provider=None, *, singles_cutoff=0.0, iterations=80):
    options = core._PeriodicGaussianPairCCSDOptions()
    options.singles = _pno_options(occupation_cutoff=singles_cutoff)
    options.solver = _solver_options(maximum_iterations=iterations,
        coefficient_orthogonality_tolerance=1e-10,
        maximum_diagonal_update_antisymmetry_norm=1e-10,
        # The outer physical driver derives this cost from its future
        # provider. Direct leaf tests pass the actual provider explicitly.
        maximum_integral_work_units_per_call=0 if provider is None else provider.memory.scalar_work_units)
    live = core._PeriodicGaussianPairCCSDLiveInventory()
    live.other_live_bytes_per_worker = 2**20  # Other tiny fixture/oracle owners.
    live.fixed_backend_margin_bytes_per_worker = 65536
    caps = core._PeriodicGaussianPairCCSDCaps()
    caps.singles = _pno_caps()
    solver = core._BoundedRestrictedPairCCSDSolverCaps()
    solver.maximum_occupied_count = solver.maximum_common_virtual_dimension = 4
    solver.maximum_pair_count = 10
    solver.maximum_singles_rank = solver.maximum_pair_rank = 4
    solver.maximum_owned_numerical_bytes = 2**20
    solver.maximum_per_replica_inventoried_bytes, solver.maximum_node_inventoried_bytes = 2**24, 2**26
    solver.maximum_integral_calls = solver.maximum_singles_calls = solver.maximum_doubles_calls = 10**12
    solver.maximum_work_units = 10**15
    caps.solver = solver
    caps.maximum_owned_numerical_bytes = 2**22
    caps.maximum_per_worker_inventoried_bytes, caps.maximum_node_inventoried_bytes = 2**24, 2**26
    caps.maximum_pair_count, caps.maximum_integral_calls = 10, 10**12
    caps.maximum_progress_callbacks, caps.maximum_work_units = 1024, 10**15
    return options, live, caps


def _run(b, provider, warmstart, controls=None, callback=None):
    options, live, caps = _controls(provider) if controls is None else controls
    return core._run_periodic_gaussian_pair_ccsd(b.reference, b.basis, provider, warmstart,
        options, live, caps, callback)


def _plan(b, provider, warmstart, controls=None):
    options, live, caps = _controls(provider) if controls is None else controls
    return core._plan_periodic_gaussian_pair_ccsd(b.reference, b.basis, provider, warmstart,
        options, live, caps)


def _oracle_problem(b, warmstart, result):
    """Independent actual AO-tile integrals, original F and retained frames.

    The legacy/spin residual helpers take real factors. Only in this tiny
    oracle, eigendecompose the independently assembled Coulomb Gram, with
    measured negative roundoff and reconstruction checks. No native provider
    rows or scalar ERI implementation supplies the test integrals.
    """
    p = _dense_case(b)
    o, n = p["o"], p["v"]
    total = o+n
    gram = p["eri"].reshape(total**2, total**2)
    np.testing.assert_allclose(gram, gram.T, atol=4e-12, rtol=4e-12)
    values, vectors = np.linalg.eigh((gram+gram.T)/2)
    assert values.min() > -4e-12
    factors = (np.sqrt(np.maximum(values, 0))[:, None]*vectors.T).reshape(-1, total, total)
    np.testing.assert_allclose(np.einsum("Ppq,Prs->pqrs", factors, factors), p["eri"], atol=5e-12, rtol=5e-12)
    # Original Fock, not the semicanonical approximation used to generate
    # PNOs. The dense selected-(T) helper audits the tiny projected difference.
    fock = b.basis.fock_copy()
    np.testing.assert_allclose(fock, p["fock"], atol=1e-10, rtol=0)
    labels = list(combinations_with_replacement(range(o), 2))
    sc = [result.singles_coefficients_copy(i) for i in range(o)]
    pc = [warmstart.pair(i, j).coefficients_copy() for i, j in labels]
    # _dense_case returns the strided real view of a complex AO contraction.
    # The native dense diagnostic deliberately accepts no implicit casts or
    # copies, so explicitly materialize this tiny oracle's binary64 tensor.
    p.update(n=n, eri=np.ascontiguousarray(p["eri"]),
        factors=np.ascontiguousarray(factors), fock=fock, labels=labels,
        foo=np.ascontiguousarray(fock[:o, :o]), fvv=np.ascontiguousarray(fock[o:, o:]),
        fov=np.ascontiguousarray(fock[:o, o:]), sc=sc, pc=pc,
        sr=[c.shape[1] for c in sc], pr=[c.shape[1] for c in pc],
        st=[np.zeros(c.shape[1]) for c in sc],
        pt=[warmstart.solver.pair_copy(i, j) for i, j in labels])
    return p


def _verify_returned_snapshot(b, warmstart, result, *, spin=True):
    p = _oracle_problem(b, warmstart, result)
    st = [result.solver.singles_copy(i) for i in range(p["o"])]
    pt = [result.solver.pair_copy(i, j) for i, j in p["labels"]]
    common = _expand(p, st, pt)
    r1, r2 = _legacy(common)
    norms = _projected_norms(p, *_project(p, r1, r2))
    energy = _energy(common, common["t1"], common["t2"])
    snapshot = result.solver.final_snapshot
    for field, expected in zip(("singles_max_residual", "doubles_max_residual",
            "singles_residual_norm", "doubles_residual_norm"), norms, strict=True):
        assert getattr(snapshot, field) == pytest.approx(expected, abs=1e-11)
    assert snapshot.correlation_energy == pytest.approx(energy, abs=1e-11)
    if result.converged:
        assert max(norms[:2]) < 2e-11
    if spin:
        sr1, sr2, se = _spin_residual(common)
        np.testing.assert_allclose(sr1, r1, atol=2e-11, rtol=2e-10)
        np.testing.assert_allclose(sr2, r2, atol=2e-11, rtol=2e-10)
        assert energy == pytest.approx(se, abs=1e-11)
    for i, j in p["labels"]:
        np.testing.assert_array_equal(result.solver.pair_copy(i, j), result.solver.pair_copy(j, i).T)
    return p, common, energy, (r1, r2)


@pytest.mark.parametrize("system,nk", [("he", 1), ("he", 2), ("he", 3), ("he2", 1), ("he2", 2)])
def test_full_rank_actual_pair_ccsd_matches_independent_common_solution(system, nk):
    b, provider = _physical(system, nk)
    warmstart = _mp2_run(b, provider)
    events = []
    result = _run(b, provider, warmstart, callback=events.append)
    assert result.converged and result.periodic_energy_per_cell
    assert result.diagnostics.complete_common_finite_torus_basis
    assert result.diagnostics.all_singles_full_rank and result.diagnostics.all_pairs_full_rank
    p, common, energy, _ = _verify_returned_snapshot(b, warmstart, result)
    expected_s, expected_p, expected_e, _, _ = _jacobi_oracle(p)
    expected = _expand(p, expected_s, expected_p)
    np.testing.assert_allclose(common["t1"], expected["t1"], atol=2e-11, rtol=2e-8)
    np.testing.assert_allclose(common["t2"], expected["t2"], atol=2e-11, rtol=2e-8)
    assert energy == pytest.approx(expected_e, abs=1e-11)
    if p["o"] <= 3:
        dense_input = _expand(p, p["st"], p["pt"])
        for key in ("eri", "fock", "t1", "t2"):
            assert dense_input[key].dtype == np.float64 and dense_input[key].flags.c_contiguous
        dense = _dense_run(dense_input, supplied=True)
        assert dense.final_snapshot.converged
        np.testing.assert_allclose(common["t1"], dense.t1, atol=2e-11, rtol=2e-8)
        np.testing.assert_allclose(common["t2"], dense.t2, atol=2e-11, rtol=2e-8)
    assert result.correlation_energy_per_cell == pytest.approx(energy/nk, abs=1e-11)
    assert result.total_energy_per_cell == pytest.approx(b.hf.state.reference_energy_per_cell+energy/nk, abs=1e-11)
    assert result.matched_finite_gaussian_hf_recipe
    assert not result.production_dlpno and not result.includes_triples and not result.infinite_source_accuracy_certified
    assert result.warmstart_identity_sha256 == warmstart.identity_sha256
    assert result.pair_spaces_identity_sha256 == warmstart.pair_spaces_identity_sha256
    assert result.basis_identity_sha256 == b.basis.identity_sha256
    assert result.provider_identity_sha256 == provider.identity_sha256
    assert result.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256
    assert len(events) == result.diagnostics.completed_progress_callbacks <= result.memory.progress_callback_upper_bound
    stages = core._PeriodicGaussianPairCCSDStage
    assert events[0].stage == stages.BEGIN and events[-1].stage == stages.FINISHED
    solver_events = [e.solver for e in events if e.stage == stages.SOLVER]
    assert [e.iteration for e in solver_events] == list(range(1, result.solver.final_snapshot.iteration+1))
    assert solver_events[-1].correlation_energy == result.solver.final_snapshot.correlation_energy
    assert result.diagnostics.completed_integral_calls <= result.memory.integral_calls_upper_bound
    if system == "he2":
        assert abs(p["foo"][0, 1]) > 1e-3  # Occupied-Fock coupling remains present.
        assert np.linalg.norm(common["t1"]) > 1e-10


def test_truncated_actual_spaces_match_independent_projected_fixed_point():
    b, provider = _physical("he2", 2)
    full_mp2 = _mp2_run(b, provider)
    occupations = np.concatenate([full_mp2.pair(i, j).original_pno_occupations_copy()
        for i, j in combinations_with_replacement(range(4), 2)])
    warmstart = _mp2_run(b, provider, _mp2_controls(cutoff=float(np.median(occupations))))
    full_singles = _run(b, provider, warmstart)
    singles_occupations = np.concatenate([full_singles.singles_original_pno_occupations_copy(i) for i in range(4)])
    result = _run(b, provider, warmstart, _controls(provider, singles_cutoff=float(np.median(singles_occupations))))
    assert result.converged and not result.diagnostics.all_pairs_full_rank
    assert not result.diagnostics.all_singles_full_rank
    p, common, energy, raw = _verify_returned_snapshot(b, warmstart, result)
    es, ep, ee, _, _ = _jacobi_oracle(p)
    independent = _expand(p, es, ep)
    np.testing.assert_allclose(common["t1"], independent["t1"], atol=2e-11, rtol=2e-8)
    np.testing.assert_allclose(common["t2"], independent["t2"], atol=2e-11, rtol=2e-8)
    assert energy == pytest.approx(ee, abs=1e-11)
    assert max(np.max(abs(raw[0])), np.max(abs(raw[1]))) > 1e-8
    assert result.identity_sha256 != full_singles.identity_sha256


def test_singles_cutoff_is_independent_of_diagonal_double_space_and_mp2_owner():
    b, provider = _physical("he2", 1)
    warmstart = _mp2_run(b, provider)
    full = _run(b, provider, warmstart)
    cutoff = 2*max(float(full.singles_original_pno_occupations_copy(i).max()) for i in range(2))
    pair_payloads = [warmstart.pair(i, j).identity_sha256 for i, j in combinations_with_replacement(range(2), 2)]
    zero_singles = _run(b, provider, warmstart, _controls(provider, singles_cutoff=cutoff))
    assert zero_singles.converged and zero_singles.diagnostics.zero_rank_singles == 2
    assert zero_singles.diagnostics.all_pairs_full_rank
    assert zero_singles.pair_spaces_identity_sha256 == full.pair_spaces_identity_sha256
    assert zero_singles.singles_spaces_identity_sha256 != full.singles_spaces_identity_sha256
    assert [warmstart.pair(i, j).identity_sha256 for i, j in combinations_with_replacement(range(2), 2)] == pair_payloads
    for i in range(2):
        assert zero_singles.solver.singles_rank(i) == 0
        assert zero_singles.solver.pair_rank(i, i) == 2
        assert zero_singles.singles_options(i).occupation_cutoff == cutoff
        assert full.singles_options(i).occupation_cutoff == 0
    _verify_returned_snapshot(b, warmstart, zero_singles)


def test_actual_truncated_gamma_fixed_point_with_spin_orbital_jacobi_only():
    b, provider = _physical("he2", 1)
    full = _mp2_run(b, provider)
    occupations = np.concatenate([full.pair(i, j).original_pno_occupations_copy()
        for i, j in combinations_with_replacement(range(2), 2)])
    warmstart = _mp2_run(b, provider, _mp2_controls(cutoff=float(np.median(occupations))))
    result = _run(b, provider, warmstart)
    assert result.converged and not result.diagnostics.all_pairs_full_rank
    p, common, energy, _ = _verify_returned_snapshot(b, warmstart, result)
    es, ep, ee, _, _ = _jacobi_oracle(p, spin=True)
    expected = _expand(p, es, ep)
    np.testing.assert_allclose(common["t1"], expected["t1"], atol=2e-11, rtol=2e-8)
    np.testing.assert_allclose(common["t2"], expected["t2"], atol=2e-11, rtol=2e-8)
    assert energy == pytest.approx(ee, abs=1e-11)


def test_all_zero_spaces_and_no_doubles_with_full_singles_are_valid_limits():
    b, provider = _physical("he2", 1)
    full = _mp2_run(b, provider)
    cutoff = 2*max(float(full.pair(i, j).original_pno_occupations_copy().max())
        for i, j in combinations_with_replacement(range(2), 2))
    warmstart = _mp2_run(b, provider, _mp2_controls(cutoff=cutoff))
    zero = _run(b, provider, warmstart, _controls(provider, singles_cutoff=cutoff))
    assert zero.converged and zero.solver.final_snapshot.iteration == 2
    assert zero.solver.final_snapshot.correlation_energy == 0
    assert zero.diagnostics.zero_rank_singles == 2
    assert zero.solver.memory.total_singles_elements == zero.solver.memory.total_doubles_elements == 0
    assert zero.total_energy_per_cell == b.hf.state.reference_energy_per_cell
    assert zero.solver.final_snapshot.integral_calls > 0  # Full residual, not silently skipped targets.
    singles = _run(b, provider, warmstart)
    assert singles.converged and singles.diagnostics.all_singles_full_rank
    assert singles.solver.memory.total_doubles_elements == 0
    assert singles.solver.memory.total_singles_elements == 4
    _verify_returned_snapshot(b, warmstart, singles)


@pytest.mark.parametrize("nk", [1, 2])
def test_actual_frozen_core_pair_ccsd_keeps_full_hf_core_constant(nk):
    b = _frozen_bundle((nk, 1, 1))
    _prepare_leaves(b, localization_options=_he2_controls(b)[0].localization)
    provider = _provider(b)
    warmstart = _mp2_run(b, provider)
    result = _run(b, provider, warmstart)
    assert (result.memory.occupied_count, result.memory.common_virtual_dimension) == (nk, 2*nk)
    assert result.converged and result.periodic_energy_per_cell
    _, common, energy, _ = _verify_returned_snapshot(b, warmstart, result)
    physical = _full_core_hamiltonian(b)
    expected = (physical["inactive"]+physical["active_reference"]+energy)/nk+physical["nuclear"]
    assert result.total_energy_per_cell == pytest.approx(expected, abs=2e-11)
    assert abs(physical["inactive"]) > 1
    if nk == 1:
        ground, reference = _two_electron_fci(physical["h"], physical["eri"])
        assert energy == pytest.approx(ground-reference, abs=2e-11)
        assert result.total_energy_per_cell == pytest.approx(physical["inactive"]+ground+physical["nuclear"], abs=2e-11)
    assert common["t1"].shape == (nk, 2*nk)


def test_borrowed_warmstart_inventory_and_explicit_other_live_bytes():
    b, provider = _physical("he2", 1)
    warmstart = _mp2_run(b, provider)
    controls = _controls(provider)
    plan = _plan(b, provider, warmstart, controls)
    result = _run(b, provider, warmstart, controls)
    o, n = plan.occupied_count, plan.common_virtual_dimension
    pair_ct = sum(8*(pair.coefficients_copy().size+warmstart.solver.pair_copy(i, j).size)
        for i, j in combinations_with_replacement(range(o), 2) for pair in [warmstart.pair(i, j)])
    assert plan.borrowed_mp2_pair_ct_bytes == pair_ct
    assert plan.borrowed_mp2_numerical_bytes >= pair_ct
    assert plan.borrowed_mp2_control_bytes > 0
    assert plan.singles_output_upper_bytes == o*(8*n*n+16*n)
    assert plan.zero_singles_upper_bytes == 8*o*n
    assert result.diagnostics.retained_singles_bytes == plan.singles_output_upper_bytes
    assert result.diagnostics.zero_initial_singles_bytes == plan.zero_singles_upper_bytes
    assert plan.peak_owned_numerical_bytes == max(plan.singles_generation_phase_upper_bytes, plan.solver_phase_owned_upper_bytes)
    assert result.solver.memory.peak_owned_numerical_bytes <= plan.solver_upper.peak_owned_numerical_bytes
    assert plan.required_node_memory_bytes == plan.reference_base_node_bytes+plan.replicas_per_node*plan.per_worker_inventoried_bytes
    controls[1].other_live_bytes_per_worker += 777
    extra = _plan(b, provider, warmstart, controls)
    assert extra.per_worker_inventoried_bytes-plan.per_worker_inventoried_bytes == 777
    assert extra.required_node_memory_bytes-plan.required_node_memory_bytes == 777*plan.replicas_per_node


@pytest.mark.parametrize("cap,field", [
    ("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_per_worker_inventoried_bytes", "per_worker_inventoried_bytes"),
    ("maximum_node_inventoried_bytes", "required_node_memory_bytes"),
    ("maximum_pair_count", "pair_count"), ("maximum_integral_calls", "integral_calls_upper_bound"),
    ("maximum_progress_callbacks", "progress_callback_upper_bound"), ("maximum_work_units", "work_units_upper_bound"),
])
def test_outer_cap_exact_and_minus_one_fail_before_progress(cap, field):
    b, provider = _physical()
    warmstart = _mp2_run(b, provider)
    controls = _controls(provider)
    value = getattr(_plan(b, provider, warmstart, controls), field)
    setattr(controls[2], cap, value)
    assert _run(b, provider, warmstart, controls).converged
    setattr(controls[2], cap, value-1)
    events = []
    with pytest.raises((ValueError, OverflowError), match="cap"):
        _run(b, provider, warmstart, controls, events.append)
    assert not events


@pytest.mark.parametrize("component,cap", [("singles", "maximum_owned_numerical_bytes"),
    ("singles", "maximum_integral_calls"), ("solver", "maximum_work_units"),
    ("solver", "maximum_owned_numerical_bytes"), ("solver", "maximum_singles_calls"),
    ("solver", "maximum_doubles_calls")])
def test_inner_caps_reject_before_singles_callback(component, cap):
    b, provider = _physical()
    warmstart = _mp2_run(b, provider)
    controls = _controls(provider)
    setattr(getattr(controls[2], component), cap, 0)
    events = []
    with pytest.raises((ValueError, OverflowError), match="cap"):
        _run(b, provider, warmstart, controls, events.append)
    assert not events


def test_unconverged_warmstart_and_unconverged_ccsd_are_distinct_failures():
    b, provider = _physical("he2", 1)
    incomplete = _mp2_run(b, provider, _mp2_controls(iterations=1))
    assert not incomplete.converged
    events = []
    with pytest.raises((ValueError, RuntimeError), match="converged|warm"):
        _run(b, provider, incomplete, callback=events.append)
    assert not events
    warmstart = _mp2_run(b, provider)
    result = _run(b, provider, warmstart, _controls(provider, iterations=1))
    assert not result.converged and not result.periodic_energy_per_cell
    assert result.solver.final_snapshot.iteration == 1
    _, common, _, _ = _verify_returned_snapshot(b, warmstart, result)
    np.testing.assert_array_equal(common["t1"], 0)
    for i, j in combinations_with_replacement(range(2), 2):
        np.testing.assert_array_equal(result.solver.pair_copy(i, j), warmstart.solver.pair_copy(i, j))
    with pytest.raises(RuntimeError, match="converged|per.cell"):
        _ = result.total_energy_per_cell


def test_selected_incomplete_common_basis_never_claims_per_cell_energy():
    b, provider = _physical("he", 2, full=False)
    warmstart = _mp2_run(b, provider)
    result = _run(b, provider, warmstart)
    assert result.converged and not result.diagnostics.complete_common_finite_torus_basis
    assert not result.periodic_energy_per_cell
    with pytest.raises(RuntimeError, match="complete|per.cell"):
        _ = result.correlation_energy_per_cell


@pytest.mark.parametrize("field", ["occupation_cutoff", "denominator_floor", "maximum_initial_fvv_offdiagonal_norm"])
def test_no_implicit_singles_controls_or_provider_work_declaration(field):
    b, provider = _physical()
    warmstart = _mp2_run(b, provider)
    controls = _controls(provider)
    setattr(controls[0].singles, field, np.nan)
    with pytest.raises((ValueError, OverflowError)):
        _plan(b, provider, warmstart, controls)
    controls = _controls(provider)
    controls[0].solver.maximum_integral_work_units_per_call = provider.memory.scalar_work_units+1
    with pytest.raises(ValueError, match="work|provider"):
        _plan(b, provider, warmstart, controls)
    controls = _controls(provider)
    controls[0].singles.denominator_floor *= 2
    with pytest.raises(ValueError, match="denominator|floor"):
        _plan(b, provider, warmstart, controls)


def test_exact_owners_callbacks_control_snapshots_and_nested_output_lifetime():
    b, provider = _physical()
    warmstart = _mp2_run(b, provider)
    other, foreign = _physical()
    foreign_warmstart = _mp2_run(other, foreign)
    with pytest.raises(ValueError, match="owner|receipt|source|warm"):
        _plan(b, provider, foreign_warmstart)
    with pytest.raises(ValueError, match="owner|receipt|source|provider"):
        _plan(b, foreign, warmstart)
    controls = _controls(provider)
    baseline = _run(b, provider, warmstart, controls)
    def change_controls(event):
        controls[0].singles.occupation_cutoff = np.nan
        controls[0].solver.maximum_iterations = 0
        controls[2].maximum_pair_count = 0
    copied = _run(b, provider, warmstart, controls, change_controls)
    assert copied.identity_sha256 == baseline.identity_sha256
    for stage in (core._PeriodicGaussianPairCCSDStage.BEGIN, core._PeriodicGaussianPairCCSDStage.SOLVER,
                  core._PeriodicGaussianPairCCSDStage.FINISHED):
        def cancel(event):
            if event.stage == stage:
                raise RuntimeError("intentional actual pair CCSD cancellation")
        with pytest.raises(RuntimeError, match="intentional actual"):
            _run(b, provider, warmstart, callback=cancel)
    with pytest.raises(ValueError, match="callable"):
        _run(b, provider, warmstart, callback=7)
    assert not hasattr(baseline, "singles")  # No PNO-consuming API can steal a nested owner.
    original = baseline.singles_coefficients_copy(0)
    baseline.singles_coefficients_copy(0)[:] = 7
    options = baseline.singles_options(0)
    options.occupation_cutoff = 999
    assert baseline.singles_options(0).occupation_cutoff == 0
    solver = baseline.solver
    del b, provider, warmstart, other, foreign, foreign_warmstart
    gc.collect()
    np.testing.assert_array_equal(baseline.singles_coefficients_copy(0), original)
    assert baseline.total_energy_per_cell < 0
    del baseline, copied
    gc.collect()
    assert solver.final_snapshot.converged
    assert np.isfinite(solver.pair_copy(0, 0)).all()
