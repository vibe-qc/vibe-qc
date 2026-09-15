"""Tiny iterative all-q Gaussian PH replacement through physical CCSD/(T).

The reference contracts the complete common-frame residual independently.
Neither this finite-source oracle nor convergence certifies production DLPNO
scaling, correlation-auxiliary quality, or an infinite-lattice result.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_pair_mp2 import (
    _physical, _run as _mp2, _controls as _mp2_controls,
)
from tests.test_periodic_gaussian_pair_ccsd import (
    _run as _full_run, _verify_returned_snapshot,
)
from tests.test_periodic_gaussian_domain_pair_mp2 import (
    _physical as _domain_physical, _run as _domain_mp2,
    _controls as _domain_mp2_controls,
)
from tests.test_periodic_gaussian_domain_pair_ccsd_t import _check_ccsd
from tests.test_periodic_gaussian_pair_ccsd_split import (
    _controls as _ccsd_controls, _embedded_controls, _compare_amplitudes,
    _run as _uncertified_run,
)
from tests.test_periodic_gaussian_pair_particle_hole import _controls as _ph_controls
from tests.test_periodic_gaussian_triple_spaces import _run as _spaces
from tests.test_periodic_gaussian_local_triples import _run as _triples


def _case(kind="he", nk=2, *, domain=False, cutoff=0.0):
    if domain:
        b = _domain_physical(nk, kind=kind, full=(kind == "he"))
        warm = _domain_mp2(b, _domain_mp2_controls(b, cutoff=cutoff))
    else:
        b, b_provider = _physical(kind, nk)
        b.provider = b_provider
        warm = _mp2(b, b.provider, _mp2_controls(cutoff=cutoff))
    return SimpleNamespace(b=b, warm=warm, domain=domain)


def _controls(case, *, iterations=80, singles_cutoff=0.0):
    config, interaction_options, _, ph_caps = _ph_controls(case)
    # Per-target operation bounds include the real all-q factories. An
    # unbounded sentinel multiplied by the iteration/pair census overflows.
    ph_caps.maximum_work_units = 10**16
    ph_caps.maximum_reciprocal_candidates = ph_caps.maximum_image_candidates = 10**15
    base = (_embedded_controls(case.b, iterations=iterations, singles_cutoff=singles_cutoff)
        if case.domain else _ccsd_controls(case.b.provider,
            iterations=iterations, singles_cutoff=singles_cutoff))
    options, live, caps = base
    solver = caps.solver
    solver.maximum_owned_numerical_bytes = 2**24
    solver.maximum_per_replica_inventoried_bytes = 2**26
    solver.maximum_node_inventoried_bytes = 2**27
    solver.maximum_work_units = 10**19
    caps.solver = solver
    caps.maximum_owned_numerical_bytes = 2**24
    caps.maximum_per_worker_inventoried_bytes = 2**26
    caps.maximum_node_inventoried_bytes = 2**27
    caps.maximum_work_units = 10**19
    singles = caps.singles
    singles.maximum_per_worker_inventoried_bytes = 2**26
    singles.maximum_node_inventoried_bytes = 2**27
    caps.singles = singles
    if case.domain:
        embedded = caps.embedded_singles
        embedded.maximum_per_worker_inventoried_bytes = 2**26
        embedded.maximum_node_inventoried_bytes = 2**27
        embedded.maximum_control_storage_bytes_per_worker = 2**25
        caps.embedded_singles = embedded
    return config, interaction_options, ph_caps, options, live, caps


def _arguments(case, controls=None, **changes):
    config, interaction_options, ph_caps, options, live, caps = (
        _controls(case) if controls is None else controls)
    b = case.b
    args = dict(hf=b.hf, reference=b.reference, wannier=b.wannier,
        common_geometry=b.builder if case.domain else b.domain,
        space=None if case.domain else b.space, basis=b.basis,
        provider=b.provider, warmstart=case.warm, config=config,
        interaction_options=interaction_options, particle_hole_caps=ph_caps,
        options=options, live=live, caps=caps)
    args.update(changes)
    return args


def _plan(case, controls=None, **changes):
    return core._plan_periodic_gaussian_pair_ccsd_physical_particle_hole_diagnostic(
        **_arguments(case, controls, **changes))


def _run(case, controls=None, **changes):
    args = _arguments(case, controls)
    args.update(ao_basis=case.b.ao, auxiliary_basis=case.b.auxiliary, gauges=case.b.gauge)
    args.update(changes)
    return core._run_periodic_gaussian_pair_ccsd_physical_particle_hole_diagnostic(**args)


@pytest.fixture(scope="module", params=[("he", 2, False), ("he", 3, False),
    ("he2", 2, True), ("frozen", 2, True)])
def actual(request):
    kind, nk, domain = request.param
    case = _case(kind, nk, domain=domain)
    case.controls = _controls(case)
    case.full = _full_run(case.b, case.b.provider, case.warm, case.controls[3:])
    case.events = []
    case.result = _run(case, case.controls, progress=case.events.append)
    return case


@pytest.fixture(scope="module")
def small():
    return _case("he", 1)


def test_actual_all_q_replacement_matches_the_complete_independent_CCSD_snapshot(actual):
    result, full = actual.result, actual.full
    assert result.converged and result.periodic_energy_per_cell
    assert result.split_bare_particle_hole and result.particle_hole_physical_source_certified
    assert len(result.physical_particle_hole_source_identity_sha256) == 64
    assert result.identity_sha256 != full.identity_sha256
    assert result.physical_particle_hole_source_identity_sha256 != result.consumed_particle_hole_identity_sha256
    assert result.warmstart_identity_sha256 == full.warmstart_identity_sha256
    assert result.provider_identity_sha256 == full.provider_identity_sha256
    assert result.matched_finite_gaussian_hf_recipe
    assert not result.production_dlpno and not result.includes_triples
    assert not result.infinite_source_accuracy_certified
    _compare_amplitudes(full, result)
    if actual.domain:
        _check_ccsd(actual.b, actual.warm, result)
    else:
        _verify_returned_snapshot(actual.b, actual.warm, result)
    assert result.total_energy_per_cell == pytest.approx(full.total_energy_per_cell, abs=2e-11)


def test_current_snapshot_progress_counts_every_pair_and_both_occupied_legs(actual):
    r = actual.result
    f = r.solver.final_snapshot
    pairs, occupied = r.memory.pair_count, r.memory.occupied_count
    assert f.particle_hole_calls == f.particle_hole_visits == pairs*f.iteration
    assert f.particle_hole_source_slots == 2*occupied*f.particle_hole_visits
    assert r.diagnostics.completed_particle_hole_calls == f.particle_hole_calls
    events = [e.solver for e in actual.events if e.stage == core._PeriodicGaussianPairCCSDStage.SOLVER]
    assert [e.iteration for e in events] == list(range(1, f.iteration+1))
    assert [e.particle_hole_calls for e in events] == [pairs*i for i in range(1, f.iteration+1)]
    assert f.charged_particle_hole_work_units > 0
    assert not r.solver.particle_hole_physical_source_certified  # Only enclosing actual-source factory qualifies.


def test_qualified_multik_CCSD_result_continues_through_TNO_and_coupled_triples(actual):
    b, warm, result = actual.b, actual.warm, actual.result
    spaces = _spaces(b, b.provider, warm, result)
    triples = _triples(b, b.provider, warm, result, spaces)
    full_spaces = _spaces(b, b.provider, warm, actual.full)
    full_triples = _triples(b, b.provider, warm, actual.full, full_spaces)
    assert triples.converged and triples.periodic_energy_per_cell
    assert not triples.production_dlpno and not triples.infinite_source_accuracy_certified
    assert triples.triples_energy_per_cell == pytest.approx(full_triples.triples_energy_per_cell, abs=8e-13)
    assert triples.total_energy_per_cell == pytest.approx(full_triples.total_energy_per_cell, abs=2e-11)


@pytest.mark.parametrize("iterations", [1, 2])
def test_limit_preserves_last_evaluated_snapshot_and_never_publishes_unconverged_total(small, iterations):
    controls = _controls(small, iterations=iterations)
    full = _full_run(small.b, small.b.provider, small.warm, controls[3:])
    result = _run(small, controls)
    assert not result.converged and result.solver.final_snapshot.iteration == iterations
    assert result.particle_hole_physical_source_certified
    assert not result.periodic_energy_per_cell
    _compare_amplitudes(full, result, tolerance=2e-13)
    _verify_returned_snapshot(small.b, small.warm, result)
    with pytest.raises(RuntimeError, match="convergence"):
        _ = result.total_energy_per_cell


@pytest.mark.parametrize("domain", [False, True])
def test_zero_rank_still_consumes_all_physical_source_slots(domain):
    case = _case("he", 2, domain=domain, cutoff=1.0)
    result = _run(case, _controls(case, singles_cutoff=1.0))
    f = result.solver.final_snapshot
    assert result.converged and result.particle_hole_physical_source_certified
    assert f.correlation_energy == f.singles_residual_norm == f.doubles_residual_norm == 0
    assert f.particle_hole_visits == f.iteration*result.memory.pair_count
    assert f.particle_hole_source_slots == 2*result.memory.occupied_count*f.particle_hole_visits


def test_generic_scalar_producer_still_cannot_obtain_a_physical_source_certificate(small):
    controls = _controls(small)
    result = _uncertified_run(small.b, small.b.provider, small.warm, controls[3:])
    assert result.converged and not result.particle_hole_physical_source_certified
    assert not result.periodic_energy_per_cell
    assert result.physical_particle_hole_source_identity_sha256 == ""


def test_count_plan_preserves_source_owners_and_labels_duplicate_roles(actual):
    p = _plan(actual, actual.controls)
    inner = p.ccsd
    warm = actual.warm
    assert p.domain_generated == actual.domain
    assert p.pair_count == inner.pair_count == warm.memory.pair_count
    assert p.pair_coefficient_bytes == sum(warm.pair(i, j).coefficients_copy().nbytes
        for i in range(p.occupied_count) for j in range(i, p.occupied_count))
    assert p.borrowed_warmstart_numerical_bytes == (warm.diagnostics.retained_pair_bytes
        + warm.diagnostics.retained_pair_geometry_bytes + warm.solver.memory.output_numerical_bytes)
    assert p.additional_retained_numerical_bytes > 0
    assert inner.borrowed_particle_hole_additional_numerical_bytes == p.additional_retained_numerical_bytes
    assert inner.solver_upper.particle_hole_retained_numerical_bytes == p.additional_retained_numerical_bytes
    assert p.producer_transient_upper_bytes >= actual.controls[2].maximum_owned_numerical_bytes+p.pair_coefficient_bytes
    assert p.producer_additional_control_storage_bytes == (
        actual.controls[2].maximum_control_storage_bytes+p.wrapper_fixed_control_storage_bytes)
    assert p.work_units_upper_bound == inner.work_units_upper_bound+p.wrapper_work_units_upper_bound
    assert p.required_node_memory_bytes == (inner.reference_base_node_bytes
        + inner.replicas_per_node*p.per_worker_inventoried_bytes)
    controls = _controls(actual)
    controls[-2].other_live_bytes_per_worker += 1234
    extra = _plan(actual, controls)
    assert extra.per_worker_inventoried_bytes-p.per_worker_inventoried_bytes == 1234
    assert extra.required_node_memory_bytes-p.required_node_memory_bytes == 1234*inner.replicas_per_node
    assert extra.peak_owned_numerical_bytes == p.peak_owned_numerical_bytes


@pytest.mark.parametrize("field,plan_field", [
    ("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_per_worker_inventoried_bytes", "per_worker_inventoried_bytes"),
    ("maximum_node_inventoried_bytes", "required_node_memory_bytes"),
    ("maximum_work_units", "work_units_upper_bound"),
])
def test_outer_cap_minus_one_rejects_before_gauge_scan_or_progress(small, field, plan_field):
    controls = _controls(small)
    p = _plan(small, controls)
    setattr(controls[-1], field, getattr(p, plan_field)-1)
    events = []
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap|limit"):
        _run(small, controls, gauges=np.full_like(small.b.gauge, np.nan), progress=events.append)
    assert not events


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes",
    "maximum_per_worker_inventoried_bytes", "maximum_node_inventoried_bytes",
    "maximum_pair_count", "maximum_work_units"])
def test_zero_outer_caps_reject_before_gauge_scan_or_progress(small, field):
    controls = _controls(small)
    setattr(controls[-1], field, 0)
    events = []
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap|limit"):
        _run(small, controls, gauges=np.full_like(small.b.gauge, np.nan), progress=events.append)
    assert not events


def test_physical_leaf_caps_and_source_identity_cannot_be_bypassed(small):
    controls = _controls(small)
    controls[2].maximum_source_slots = 0
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap|source|controls"):
        _run(small, controls, gauges=np.full_like(small.b.gauge, np.nan))
    with pytest.raises((ValueError, RuntimeError), match="finite|gauge"):
        _run(small, gauges=np.full_like(small.b.gauge, np.nan))
    other = _case("he", 1)
    with pytest.raises((ValueError, RuntimeError), match="state|owner|source|context|warm"):
        _plan(small, warmstart=other.warm)


@pytest.mark.parametrize("stage", [core._PeriodicGaussianPairCCSDStage.BEGIN,
    core._PeriodicGaussianPairCCSDStage.SOLVER, core._PeriodicGaussianPairCCSDStage.FINISHED])
def test_progress_gauge_mutation_is_rejected_even_at_finished(small, stage):
    gauge = small.b.gauge.copy()
    def mutate(event):
        if event.stage == stage:
            gauge.flat[0] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="finite|gauge"):
        _run(small, gauges=gauge, progress=mutate)


def test_progress_gauge_descriptor_change_is_rejected_before_native_payload_replay(small):
    gauge = small.b.gauge.copy()
    def mutate(event):
        if event.stage == core._PeriodicGaussianPairCCSDStage.BEGIN:
            gauge.resize((gauge.size,), refcheck=False)
    with pytest.raises((ValueError, RuntimeError), match="gauge descriptors"):
        _run(small, gauges=gauge, progress=mutate)


def test_control_copies_and_result_ownership_are_independent_of_python_ancestors():
    case = _case("he", 1)
    controls = _controls(case)
    baseline = _run(case, controls)
    def mutate(event):
        controls[0].auxiliary_block = 0
        controls[1].maximum_overlap_imaginary_norm = np.nan
        controls[2].maximum_source_slots = 0
        controls[3].solver.maximum_iterations = 0
        controls[-1].maximum_pair_count = 0
    result = _run(case, controls, progress=mutate)
    assert result.identity_sha256 == baseline.identity_sha256
    energy, identity = result.total_energy_per_cell, result.identity_sha256
    solver = result.solver
    del case, baseline, controls
    gc.collect()
    assert result.total_energy_per_cell == energy and result.identity_sha256 == identity
    assert solver.final_snapshot.converged
