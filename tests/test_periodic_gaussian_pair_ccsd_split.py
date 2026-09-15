"""Actual HF/warm-start wrapper with a generic native scalar PH producer.

This tests replacement inside the iteration, not addition to a completed
residual. The selected-common scalar diagnostic deliberately cannot certify
the forthcoming direct all-q source or publish physical per-cell totals.
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
from tests.test_periodic_gaussian_pair_ccsd import (
    _controls as _full_controls, _run as _full_run, _plan as _full_plan,
    _verify_returned_snapshot,
)
from tests.test_periodic_gaussian_domain_pair_mp2 import (
    _physical as _domain_physical, _run as _domain_mp2,
    _controls as _domain_mp2_controls,
)
from tests.test_periodic_gaussian_domain_pair_ccsd_t import (
    _controls as _domain_controls, _check_ccsd,
)
from tests.test_periodic_gaussian_triple_spaces import _run as _triple_spaces


def _controls(provider, *, singles_cutoff=0.0, iterations=80):
    controls = _full_controls(provider, singles_cutoff=singles_cutoff, iterations=iterations)
    solver = controls[2].solver
    solver.maximum_particle_hole_calls = 1280
    controls[2].solver = solver
    return controls


def _embedded_controls(b, *, singles_cutoff=0.0, iterations=80):
    controls = _domain_controls(b, singles_cutoff=singles_cutoff, iterations=iterations)
    solver = controls[2].solver
    solver.maximum_particle_hole_calls = 1280
    controls[2].solver = solver
    return controls


def _plan(b, provider, warm, controls=None, extra=0):
    o, live, caps = _controls(provider) if controls is None else controls
    return core._plan_periodic_gaussian_pair_ccsd_split_scalar_reference(
        b.reference, b.basis, provider, warm, o, live, caps, extra)


def _run(b, provider, warm, controls=None, callback=None, *, extra=0, fault=0):
    o, live, caps = _controls(provider) if controls is None else controls
    return core._run_periodic_gaussian_pair_ccsd_split_scalar_reference(
        b.reference, b.basis, provider, warm, o, live, caps, callback, extra, fault)


def _compare_amplitudes(full, split, *, tolerance=2e-11):
    assert full.converged == split.converged
    assert split.solver.final_snapshot.correlation_energy == pytest.approx(
        full.solver.final_snapshot.correlation_energy, abs=tolerance)
    o = split.memory.occupied_count
    for i in range(o):
        # Independent singles generation remains exactly the same owner path.
        assert split.singles_identity_sha256(i) == full.singles_identity_sha256(i)
        np.testing.assert_array_equal(split.singles_coefficients_copy(i), full.singles_coefficients_copy(i))
        np.testing.assert_allclose(split.solver.singles_copy(i), full.solver.singles_copy(i),
                                   atol=tolerance, rtol=2e-8)
        for j in range(i, o):
            np.testing.assert_allclose(split.solver.pair_copy(i, j), full.solver.pair_copy(i, j),
                                       atol=tolerance, rtol=2e-8)
            np.testing.assert_array_equal(split.solver.pair_copy(i, j), split.solver.pair_copy(j, i).T)


def _check_receipts_and_scope(full, split):
    assert split.memory.split_bare_particle_hole and split.diagnostics.split_bare_particle_hole
    assert split.split_bare_particle_hole and split.solver.memory.split_bare_particle_hole
    assert not full.split_bare_particle_hole
    assert full.split_execution_identity_sha256 == full.consumed_particle_hole_identity_sha256 == ""
    assert len(split.split_execution_identity_sha256) == 64
    assert len(split.consumed_particle_hole_identity_sha256) == 64
    assert split.split_execution_identity_sha256 == split.solver.split_execution_identity_sha256
    assert split.consumed_particle_hole_identity_sha256 == split.solver.consumed_particle_hole_identity_sha256
    assert split.identity_sha256 != full.identity_sha256
    assert split.warmstart_identity_sha256 == full.warmstart_identity_sha256
    assert split.pair_spaces_identity_sha256 == full.pair_spaces_identity_sha256
    assert split.singles_spaces_identity_sha256 == full.singles_spaces_identity_sha256
    assert split.provider_identity_sha256 == full.provider_identity_sha256
    assert split.matched_finite_gaussian_hf_recipe  # Original-source lineage only.
    assert not split.particle_hole_physical_source_certified
    assert not split.physical_particle_hole_source_identity_sha256
    assert not split.periodic_energy_per_cell
    assert not split.production_dlpno and not split.includes_triples
    assert not split.infinite_source_accuracy_certified
    for name in ("correlation_energy_per_cell", "total_energy_per_cell"):
        with pytest.raises(RuntimeError, match="certified particle-hole source"):
            getattr(split, name)


@pytest.mark.parametrize("kind,nk", [("he", 1), ("he", 2), ("he", 3), ("he2", 1), ("he2", 2)])
def test_actual_full_rank_current_snapshot_split_matches_independent_ccsd(kind, nk):
    b, provider = _physical(kind, nk)
    warm = _mp2_run(b, provider)
    full = _full_run(b, provider, warm)
    events = []
    split = _run(b, provider, warm, callback=events.append)
    assert split.converged and split.diagnostics.complete_common_finite_torus_basis
    _compare_amplitudes(full, split)
    _verify_returned_snapshot(b, warm, split)
    _check_receipts_and_scope(full, split)
    final = split.solver.final_snapshot
    pairs, occupied = split.memory.pair_count, split.memory.occupied_count
    assert final.particle_hole_calls == final.particle_hole_visits == final.iteration*pairs
    assert final.particle_hole_source_slots == 2*occupied*final.particle_hole_visits
    assert split.diagnostics.completed_particle_hole_calls == final.particle_hole_calls
    assert split.diagnostics.completed_particle_hole_visits == final.particle_hole_visits
    assert split.diagnostics.charged_particle_hole_work_units == final.charged_particle_hole_work_units > 0
    solver_events = [e.solver for e in events if e.stage == core._PeriodicGaussianPairCCSDStage.SOLVER]
    assert [e.iteration for e in solver_events] == list(range(1, final.iteration+1))
    assert [e.particle_hole_calls for e in solver_events] == [pairs*i for i in range(1, final.iteration+1)]
    assert split.diagnostics.completed_progress_callbacks == len(events)


def test_actual_truncated_nonnested_spaces_preserve_orthogonal_common_residual():
    b, provider = _physical("he2", 2)
    all_mp2 = _mp2_run(b, provider)
    occupations = np.concatenate([all_mp2.pair(i, j).original_pno_occupations_copy()
        for i, j in combinations_with_replacement(range(4), 2)])
    warm = _mp2_run(b, provider, _mp2_controls(cutoff=float(np.median(occupations))))
    full_singles = _full_run(b, provider, warm)
    single_occupations = np.concatenate([full_singles.singles_original_pno_occupations_copy(i) for i in range(4)])
    controls = _controls(provider, singles_cutoff=float(np.median(single_occupations)))
    full = _full_run(b, provider, warm, controls)
    split = _run(b, provider, warm, controls)
    assert split.converged and not split.diagnostics.all_pairs_full_rank
    assert not split.diagnostics.all_singles_full_rank
    _compare_amplitudes(full, split)
    _, _, _, raw = _verify_returned_snapshot(b, warm, split)
    assert max(np.max(np.abs(raw[0])), np.max(np.abs(raw[1]))) > 1e-8


@pytest.mark.parametrize("kind,nk,full_domain", [("he", 1, False), ("he", 3, True),
                                               ("he2", 1, False), ("he2", 2, False)])
def test_actual_embedded_generation_overload_uses_original_diagonal_spaces(kind, nk, full_domain):
    b = _domain_physical(nk, kind=kind, full=full_domain)
    warm = _domain_mp2(b)
    controls = _embedded_controls(b)
    full = _full_run(b, b.provider, warm, controls)
    split = _run(b, b.provider, warm, controls)
    assert split.memory.domain_generated and split.converged
    _compare_amplitudes(full, split)
    _check_ccsd(b, warm, split)
    _check_receipts_and_scope(full, split)
    for i in range(split.memory.occupied_count):
        assert split.singles_domain_generated(i)
        assert split.singles_embedding_identity_sha256(i) == warm.diagonal_generation_embedding(i).identity_sha256
    assert split.diagnostics.singles_generation_dimension_sum == warm.diagnostics.diagonal_generation_dimension_sum


@pytest.mark.parametrize("embedded", [False, True])
def test_count_plans_include_split_work_without_summing_sequential_numeric_phases(embedded):
    if embedded:
        b = _domain_physical(1, kind="he2")
        provider, warm, controls = b.provider, _domain_mp2(b), _embedded_controls(b)
    else:
        b, provider = _physical("he2", 1)
        warm, controls = _mp2_run(b, provider), _controls(provider)
    old, split = _full_plan(b, provider, warm, controls), _plan(b, provider, warm, controls)
    solver = split.solver_upper
    o, n, pairs = split.occupied_count, split.common_virtual_dimension, split.pair_count
    saved = controls[0].solver.maximum_iterations*pairs*6*o*n*n
    assert old.integral_calls_upper_bound-split.integral_calls_upper_bound == saved
    assert solver.omitted_bare_integral_calls_upper_bound == saved
    assert old.solver_upper.singles_calls_upper_bound == solver.singles_calls_upper_bound
    assert old.solver_upper.doubles_calls_upper_bound == solver.doubles_calls_upper_bound
    assert split.solver_phase_owned_upper_bytes == (
        split.singles_output_upper_bytes+split.zero_singles_upper_bytes+solver.peak_owned_numerical_bytes)
    assert split.peak_owned_numerical_bytes == max(
        split.singles_generation_phase_upper_bytes, split.solver_phase_owned_upper_bytes)
    assert solver.peak_owned_numerical_bytes == (
        2*solver.amplitude_snapshot_bytes+solver.retained_record_bytes+solver.projected_fock_diagonal_bytes
        + max(solver.target_owned_peak_bytes, solver.particle_hole_transient_numerical_bytes))
    padded = _plan(b, provider, warm, controls, extra=913)
    assert padded.borrowed_particle_hole_additional_numerical_bytes == 913
    assert padded.solver_upper.particle_hole_retained_numerical_bytes == 913
    assert padded.per_worker_inventoried_bytes-split.per_worker_inventoried_bytes == 913
    assert padded.required_node_memory_bytes-split.required_node_memory_bytes == 913*split.replicas_per_node
    assert padded.solver_upper.per_replica_inventoried_bytes-solver.per_replica_inventoried_bytes == 913
    assert padded.peak_owned_numerical_bytes == split.peak_owned_numerical_bytes
    if not embedded:
        assert padded.singles_upper.per_worker_inventoried_bytes-split.singles_upper.per_worker_inventoried_bytes == 913
    result = _run(b, provider, warm, controls, extra=913)
    assert result.converged


@pytest.mark.parametrize("embedded", [False, True])
def test_zero_pair_and_singles_ranks_keep_all_source_slots(embedded):
    if embedded:
        b = _domain_physical(1, kind="he2")
        provider = b.provider
        warm = _domain_mp2(b, _domain_mp2_controls(b, cutoff=1.0))
        controls = _embedded_controls(b, singles_cutoff=1.0)
    else:
        b, provider = _physical("he2", 1)
        warm = _mp2_run(b, provider, _mp2_controls(cutoff=1.0))
        controls = _controls(provider, singles_cutoff=1.0)
    result = _run(b, provider, warm, controls)
    assert result.converged and result.solver.final_snapshot.iteration == 2
    assert result.diagnostics.zero_rank_singles == result.memory.occupied_count
    assert warm.diagnostics.zero_rank_pairs == warm.memory.pair_count
    s = result.solver.final_snapshot
    assert s.correlation_energy == 0 and s.singles_residual_norm == s.doubles_residual_norm == 0
    assert s.particle_hole_source_slots == 2*result.memory.occupied_count*s.particle_hole_visits
    assert s.particle_hole_visits == 2*result.memory.pair_count


@pytest.mark.parametrize("field,plan_field", [
    ("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_per_worker_inventoried_bytes", "per_worker_inventoried_bytes"),
    ("maximum_node_inventoried_bytes", "required_node_memory_bytes"),
    ("maximum_work_units", "work_units_upper_bound"),
    ("maximum_integral_calls", "integral_calls_upper_bound"),
])
def test_outer_cap_minus_one_precedes_progress_and_native_producer(field, plan_field):
    b, provider = _physical("he", 1)
    warm, controls = _mp2_run(b, provider), _controls(provider)
    plan = _plan(b, provider, warm, controls)
    setattr(controls[2], field, getattr(plan, plan_field)-1)
    events = []
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(b, provider, warm, controls, events.append, fault=1)
    assert events == []


def test_particle_hole_exact_call_cap_and_exhaustion_keep_last_evaluated_snapshot():
    b, provider = _physical("he2", 1)
    warm, controls = _mp2_run(b, provider), _controls(provider, iterations=1)
    plan = _plan(b, provider, warm, controls)
    cap = controls[2].solver
    cap.maximum_particle_hole_calls = plan.solver_upper.particle_hole_calls_upper_bound
    controls[2].solver = cap
    split = _run(b, provider, warm, controls)
    full = _full_run(b, provider, warm, controls)
    assert not split.converged and split.solver.final_snapshot.iteration == 1
    _compare_amplitudes(full, split, tolerance=2e-14)
    _verify_returned_snapshot(b, warm, split)
    o, n = split.memory.occupied_count, split.memory.common_virtual_dimension
    assert full.diagnostics.completed_integral_calls-split.diagnostics.completed_integral_calls == split.memory.pair_count*6*o*n*n
    cap.maximum_particle_hole_calls -= 1
    controls[2].solver = cap
    events = []
    with pytest.raises((ValueError, RuntimeError), match="particle-hole call cap"):
        _run(b, provider, warm, controls, events.append, fault=1)
    assert not events


@pytest.mark.parametrize("fault,match", [(1, "injected native Gaussian"), (2, "visit"),
                                        (3, "visit"), (4, "snapshot")])
def test_native_producer_failure_protocol_publishes_no_partial_result(fault, match):
    b, provider = _physical("he", 1)
    warm = _mp2_run(b, provider)
    with pytest.raises((RuntimeError, ValueError), match=match):
        _run(b, provider, warm, fault=fault)
    assert warm.converged


def test_control_copies_exact_owner_checks_and_nested_result_lifetime():
    b, provider = _physical("he", 1)
    warm = _mp2_run(b, provider)
    controls = _controls(provider)
    baseline = _run(b, provider, warm, controls)
    def mutate(event):
        controls[0].singles.occupation_cutoff = np.nan
        controls[0].solver.maximum_iterations = 0
        controls[2].maximum_pair_count = 0
    copied = _run(b, provider, warm, controls, mutate)
    assert copied.identity_sha256 == baseline.identity_sha256
    other, foreign = _physical("he", 1)
    foreign_warm = _mp2_run(other, foreign)
    with pytest.raises(ValueError, match="owner|receipt|source|warm"):
        _plan(b, provider, foreign_warm)
    with pytest.raises(ValueError, match="owner|receipt|source|provider"):
        _plan(b, foreign, warm)
    for stage in (core._PeriodicGaussianPairCCSDStage.BEGIN, core._PeriodicGaussianPairCCSDStage.SOLVER,
                  core._PeriodicGaussianPairCCSDStage.FINISHED):
        def cancel(event):
            if event.stage == stage:
                raise RuntimeError("intentional generic split callback cancellation")
        with pytest.raises(RuntimeError, match="intentional generic split"):
            _run(b, provider, warm, callback=cancel)
    original = baseline.singles_coefficients_copy(0)
    baseline.singles_coefficients_copy(0)[:] = 99
    solver = baseline.solver
    del b, provider, warm, other, foreign, foreign_warm, baseline, copied
    gc.collect()
    assert solver.final_snapshot.converged
    assert solver.pair_copy(0, 0).shape == (original.shape[1],)*2


def test_uncertified_split_source_cannot_enter_physical_triple_space_factory():
    b, provider = _physical("he", 1)
    warm = _mp2_run(b, provider)
    split = _run(b, provider, warm)
    assert split.converged and split.matched_finite_gaussian_hf_recipe
    with pytest.raises(ValueError, match="matching converged actual CCSD"):
        _triple_spaces(b, provider, warm, split)
