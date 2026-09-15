"""One actual-HF entry with all-q bare-PH replacement inside pair CCSD.

The remaining CCSD diagrams still use the complete selected common frame.
These tiny finite-source tests are not production DLPNO or solid benchmarks.
All localization, domains and post-HF owners are produced inside the first
native call; separately prepared leaves and dense oracles come afterward.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_selected_local_ccsd_t import (
    _run, _plan, _prepare_leaves, _dense_case,
    _controls as _common_controls,
)
from tests.test_periodic_gaussian_selected_local_pair_ccsd_t import (
    _fixture, _controls as _pair_controls, _assert_stages, _check_receipts,
)
from tests.test_periodic_gaussian_selected_domain_pair_ccsd_t import (
    _controls as _domain_controls,
)
from tests.test_periodic_gaussian_frozen_core_correlation import _frozen_bundle
from tests.test_periodic_gaussian_pair_particle_hole import _controls as _ph_controls
from tests.test_periodic_gaussian_real_local_provider import (
    _make as _provider, _controls as _provider_controls,
)
from tests.test_periodic_gaussian_pair_mp2 import (
    _run as _mp2, _controls as _mp2_controls,
)
from tests.test_periodic_gaussian_domain_pair_mp2 import (
    _run as _domain_mp2, _controls as _domain_mp2_controls,
)
from tests.test_periodic_gaussian_pair_domain_builder import _build as _builder
from tests.test_periodic_gaussian_pair_ccsd_physical_particle_hole import (
    _controls as _physical_controls, _run as _physical_ccsd,
)
from tests.test_periodic_gaussian_pair_ccsd import _verify_returned_snapshot
from tests.test_periodic_gaussian_domain_pair_ccsd_t import _check_ccsd
from tests.test_periodic_gaussian_triple_spaces import (
    _run as _spaces, _controls as _spaces_controls,
)
from tests.test_periodic_gaussian_local_triples import (
    _run as _triples, _controls as _triples_controls,
    _oracle_problem, _check_snapshot, _full_rank_linear_oracle,
)


_CHILD_CAPS = ("resources", "metric", "panel", "tile", "singles", "embedded_singles",
    "projection", "solver", "pnos", "domain", "pair_union", "embedding", "real_space",
    "occupied", "selector", "geometry", "moments", "kernel")


def _worker_caps(caps, *, control=None):
    """Raise only inventory ceilings for the enclosing tiny live-owner union.

    Owned payload, work and all scientific choices stay unchanged. Native
    caps are value properties, so every modified child is assigned back.
    """
    for name in ("maximum_per_worker_inventoried_bytes", "maximum_per_replica_inventoried_bytes",
                 "maximum_worker_bytes"):
        if hasattr(caps, name):
            setattr(caps, name, 2**26)
    for name in ("maximum_node_inventoried_bytes", "maximum_node_bytes"):
        if hasattr(caps, name):
            setattr(caps, name, 2**27)
    if control is not None:
        for name in ("maximum_control_storage_bytes", "maximum_control_storage_bytes_per_worker",
                     "maximum_control_storage_bytes_per_replica"):
            if hasattr(caps, name):
                setattr(caps, name, control)
    for name in _CHILD_CAPS:
        if hasattr(caps, name):
            child = getattr(caps, name)
            _worker_caps(child, control=control)
            setattr(caps, name, child)


def _controls(b, *, kind="he", domain=False, pair_cutoff=0.0,
              singles_cutoff=0.0, tno_cutoff=0.0):
    if domain:
        controls = _domain_controls(b, he2=kind != "he", full=kind == "he", pair_cutoff=pair_cutoff)
    else:
        controls = _pair_controls(b, he2=kind != "he", pair_cutoff=pair_cutoff)
    options, live, caps = controls
    options.physical_particle_hole_ccsd = True
    config, interaction_options, _, ph = _ph_controls(SimpleNamespace(b=b))
    ph.maximum_work_units = 10**16
    ph.maximum_reciprocal_candidates = ph.maximum_image_candidates = 10**15
    _worker_caps(ph)
    options.particle_hole, options.particle_hole_options, caps.particle_hole = config, interaction_options, ph
    cc = options.pair_ccsd
    singles = cc.singles
    singles.occupation_cutoff = singles_cutoff
    cc.singles = singles
    options.pair_ccsd = cc
    tno = options.triple_spaces
    tno.occupation_cutoff = tno_cutoff
    options.triple_spaces = tno
    ccc = caps.pair_ccsd
    solver = ccc.solver
    solver.maximum_particle_hole_calls = 1280
    solver.maximum_owned_numerical_bytes = ccc.maximum_owned_numerical_bytes = 2**23
    solver.maximum_work_units = ccc.maximum_work_units = 9*10**18
    ccc.solver, caps.pair_ccsd = solver, ccc
    caps.maximum_owned_numerical_bytes = 2**24
    caps.maximum_per_worker_inventoried_bytes = 2**26
    caps.maximum_node_inventoried_bytes = 2**27
    caps.maximum_work_units = 10**19
    caps.maximum_pair_control_storage_bytes = 12*2**20
    factors, resources = caps.factors, caps.factors.resources
    # A new-test-only admission ceiling, not a scientific approximation.
    # The former 1MiB factor ceiling plus the four domain-stage ceilings
    # exceeded the 16MiB outer cap by the small common Fock owner.
    resources.maximum_owned_numeric_bytes = 2**19
    factors.resources, caps.factors = resources, factors
    for name in ("factors", "pair_mp2", "domain_pair_mp2", "pair_ccsd", "triple_spaces",
                 "local_triples", "pair_domain_builder"):
        child = getattr(caps, name)
        _worker_caps(child)
        setattr(caps, name, child)
    outer = _plan(b, controls)
    # Fixed outer reservations remain fixed. Only subordinate control
    # ceilings cover the already-admitted whole-call control union.
    for name in ("pair_mp2", "domain_pair_mp2", "pair_ccsd", "triple_spaces",
                 "local_triples", "pair_domain_builder"):
        child = getattr(caps, name)
        _worker_caps(child, control=outer.control_storage_reservation_bytes)
        setattr(caps, name, child)
    return controls


@pytest.fixture(scope="module", params=[("he", 2, False), ("he", 3, False),
    ("he2", 2, True), ("frozen", 2, True)])
def actual(request):
    kind, nk, domain = request.param
    b = _frozen_bundle((nk, 1, 1)) if kind == "frozen" else _fixture(kind, nk)
    assert not hasattr(b, "basis") and not hasattr(b, "localization") and not hasattr(b, "builder")
    controls = _controls(b, kind=kind, domain=domain)
    events = []
    result = _run(b, controls=controls, callback=events.append)
    return SimpleNamespace(b=b, kind=kind, nk=nk, domain=domain, controls=controls,
        result=result, events=events)


@pytest.fixture(scope="module")
def small():
    return _fixture(nk=1)


@pytest.fixture(scope="module")
def limited():
    # Two occupied orbitals give a nontrivial triples exhaustion witness.
    return _fixture("he2", 1)


def test_one_actual_hf_call_qualifies_current_ph_and_finishes_all_four_stages(actual):
    b, r = actual.b, actual.result
    _assert_stages(r)
    _check_receipts(b, r)
    assert r.plan.physical_particle_hole_ccsd
    assert r.diagnostics.physical_particle_hole_evaluated
    assert r.converged and r.periodic_energy_per_cell
    cc = r.pair_ccsd
    assert cc.split_bare_particle_hole and cc.particle_hole_physical_source_certified
    assert len(cc.physical_particle_hole_source_identity_sha256) == 64
    assert not cc.solver.particle_hole_physical_source_certified
    assert r.total_energy_per_cell == r.local_triples.total_energy_per_cell
    assert r.total_energy_per_cell == pytest.approx(b.hf.state.reference_energy_per_cell
        + cc.correlation_energy_per_cell + r.local_triples.triples_energy_per_cell, abs=2e-15)
    assert not r.production_dlpno and not r.infinite_source_accuracy_certified
    assert actual.controls[0].pair_ccsd.solver.maximum_integral_work_units_per_call == 0
    assert actual.controls[0].local_triples.moments.maximum_integral_work_units_per_call == 0


def test_all_current_snapshot_callbacks_and_whole_owner_phases_are_admitted(actual):
    r, events = actual.result, actual.events
    p, d = r.plan, r.diagnostics
    assert p.owned_numerical_upper_bound <= actual.controls[2].maximum_owned_numerical_bytes == 2**24
    assert p.work_units_upper_bound < actual.controls[2].maximum_work_units == 10**19
    assert p.physical_pair_ccsd_metadata_work_units_upper_bound > 0
    c = actual.controls[2]
    assert p.physical_pair_ccsd_phase_owned_upper_bound == (
        p.pair_mp2_phase_owned_upper_bound+c.pair_ccsd.maximum_owned_numerical_bytes)
    assert p.pair_mp2_phase_owned_upper_bound > 0
    assert p.physical_pair_ccsd_phase_owned_upper_bound <= p.owned_numerical_upper_bound
    mp2_cap = c.domain_pair_mp2 if actual.domain else c.pair_mp2
    correlation_work = (c.pair_ccsd.maximum_work_units+2*p.physical_pair_ccsd_metadata_work_units_upper_bound
        + 4*(mp2_cap.maximum_work_units+c.triple_spaces.maximum_work_units+c.local_triples.maximum_work_units
             + (c.pair_domain_builder.maximum_work_units if actual.domain else 0)))
    assert p.work_units_upper_bound == ((p.progress_callback_upper_bound+2)*p.input_check_work_units
        + c.localization.maximum_work_units+p.domain_work_units_upper_bound+c.space.maximum_work_units
        + c.basis.maximum_work_units+c.factors.resources.maximum_work_units+correlation_work)
    assert p.correlation_phase_owned_upper_bound == (c.basis.maximum_owned_numerical_bytes
        + c.factors.resources.maximum_owned_numeric_bytes+mp2_cap.maximum_owned_numerical_bytes
        + c.pair_ccsd.maximum_owned_numerical_bytes+c.triple_spaces.maximum_owned_numerical_bytes
        + c.local_triples.maximum_owned_numerical_bytes)
    assert d.factors_memory.peak_owned_numerical_bytes <= 2**19
    physical = d.physical_particle_hole_memory
    assert physical.domain_generated == actual.domain
    assert physical.work_units_upper_bound <= actual.controls[2].pair_ccsd.maximum_work_units
    assert physical.required_node_memory_bytes <= p.node_inventoried_bytes_upper_bound
    assert physical.ccsd.per_worker_inventoried_bytes <= p.per_worker_inventoried_bytes_upper_bound
    stage = core._PeriodicGaussianSelectedLocalCCSDTStage
    sequence = [e.stage for e in events]
    assert sequence[0] == stage.BEGIN and sequence[-1] == stage.FINISHED
    ordered = [stage.PAIR_MP2, stage.PAIR_CCSD, stage.TRIPLE_SPACES, stage.LOCAL_TRIPLES]
    assert [sequence.index(s) for s in ordered] == sorted(sequence.index(s) for s in ordered)
    assert stage.CORRELATION not in sequence
    assert [e.callback_count for e in events] == list(range(1, len(events)+1))
    assert len(events) == d.completed_progress_callbacks <= p.progress_callback_upper_bound
    f, pairs, o = r.pair_ccsd.solver.final_snapshot, r.pair_ccsd.memory.pair_count, r.pair_ccsd.memory.occupied_count
    assert f.particle_hole_calls == f.particle_hole_visits == f.iteration*pairs
    assert f.particle_hole_source_slots == 2*o*f.particle_hole_visits
    solver_events = [e.pair_ccsd.solver for e in events if e.stage == stage.PAIR_CCSD
        and e.pair_ccsd.stage == core._PeriodicGaussianPairCCSDStage.SOLVER]
    assert [e.iteration for e in solver_events] == list(range(1, f.iteration+1))
    assert [e.particle_hole_calls for e in solver_events] == [pairs*i for i in range(1, f.iteration+1)]


def _manual_chain(actual):
    """Prepare separate physical owners only AFTER the native outer result."""
    b, options = actual.b, actual.controls[0]
    _prepare_leaves(b, localization_options=options.localization)
    _, live, caps = _provider_controls(b)
    b.provider = _provider(b, controls=(options.factors, live, caps), options=options.real_factors)
    b.expected_dense = _dense_case(b)  # Original geometry is still alive here.
    if actual.domain:
        b.domain_options = options.occupied_domains
        b.builder = _builder(b)
        _, live, caps = _domain_mp2_controls(b)
        warm = _domain_mp2(b, (options.domain_pair_mp2, live, caps))
    else:
        _, live, caps = _mp2_controls()
        warm = _mp2(b, b.provider, (options.pair_mp2, live, caps))
    case = SimpleNamespace(b=b, warm=warm, domain=actual.domain)
    _, _, phcaps, _, cc_live, cc_caps = _physical_controls(case)
    direct = options.pair_ccsd
    solver = direct.solver
    solver.maximum_integral_work_units_per_call = b.provider.memory.scalar_work_units
    direct.solver = solver
    cc = _physical_ccsd(case, (options.particle_hole, options.particle_hole_options,
        phcaps, direct, cc_live, cc_caps))
    _, live, caps = _spaces_controls()
    spaces = _spaces(b, b.provider, warm, cc, (options.triple_spaces, live, caps))
    _, live, caps = _triples_controls(b.provider)
    direct_t = options.local_triples
    moments = direct_t.moments
    moments.maximum_integral_work_units_per_call = b.provider.memory.scalar_work_units
    direct_t.moments = moments
    triples = _triples(b, b.provider, warm, cc, spaces, (direct_t, live, caps))
    return warm, cc, spaces, triples


def test_one_call_matches_manual_physical_chain_and_independent_projected_residual(actual):
    warm, cc, spaces, triples = _manual_chain(actual)
    r = actual.result
    assert warm.converged and cc.converged and triples.converged
    assert r.total_energy_per_cell == pytest.approx(triples.total_energy_per_cell, abs=2e-11)
    assert r.local_triples.triples_energy_per_cell == pytest.approx(triples.triples_energy_per_cell, abs=8e-13)
    for i in range(cc.memory.occupied_count):
        np.testing.assert_allclose(r.pair_ccsd.solver.singles_copy(i), cc.solver.singles_copy(i), atol=2e-11, rtol=2e-8)
        for j in range(i, cc.memory.occupied_count):
            np.testing.assert_allclose(r.pair_mp2.pair(i, j).coefficients_copy(), warm.pair(i, j).coefficients_copy(),
                atol=2e-12, rtol=2e-10)
            np.testing.assert_allclose(r.pair_ccsd.solver.pair_copy(i, j), cc.solver.pair_copy(i, j), atol=2e-11, rtol=2e-8)
    # This is the original complete residual with actual AO Gaussian-factor
    # integrals, evaluated independently at the returned projected snapshot.
    if actual.domain:
        _check_ccsd(actual.b, warm, cc)
    else:
        _verify_returned_snapshot(actual.b, warm, cc)
        p, common = _oracle_problem(actual.b, warm, cc, spaces)
        _check_snapshot(p, triples, _full_rank_linear_oracle(p, common))


def test_full_common_limit_matches_the_independent_common_ccsd_t_connector(actual):
    if actual.domain:
        pytest.skip("A genuinely truncated PAO generation is not the complete common calculation")
    baseline = _run(actual.b, controls=_common_controls(actual.b))
    assert baseline.converged and not baseline.plan.physical_particle_hole_ccsd
    assert actual.result.total_energy_per_cell == pytest.approx(baseline.total_energy_per_cell, abs=8e-12)
    assert actual.result.local_triples.triples_energy_per_cell == pytest.approx(
        baseline.correlation.triples.final_snapshot.triples_energy/actual.nk, abs=8e-13)


def test_distinct_original_pair_geometry_and_frozen_core_survive_the_one_call(actual):
    r = actual.result
    if actual.domain:
        warm, cc = r.pair_mp2, r.pair_ccsd
        assert warm.domain_generated and warm.diagnostics.retained_pair_geometry_bytes > 0
        assert warm.diagnostics.retained_pair_geometry_control_bytes > 0
        for i in range(cc.memory.occupied_count):
            diagonal = warm.diagonal_generation_embedding(i)
            assert cc.singles_embedding_identity_sha256(i) == diagonal.identity_sha256
            assert cc.singles_generation_dimension(i) == diagonal.memory.pair_dimension
        if actual.kind == "he2":
            assert warm.diagnostics.generation_dimension_sum < warm.memory.pair_count*warm.memory.common_virtual_dimension
            assert not cc.diagnostics.all_singles_full_rank
    if actual.kind == "frozen":
        assert r.pair_ccsd.memory.occupied_count == actual.nk
        assert r.pair_ccsd.memory.common_virtual_dimension == 2*actual.nk
        assert r.diagnostics.hf_energy_per_cell == actual.b.unfrozen_hf.state.reference_energy_per_cell


@pytest.mark.parametrize("stop,expected", [("pair_mp2", (True, False, False, False)),
    ("pair_ccsd", (True, True, False, False)), ("local_triples", (True, True, True, True))])
def test_exhaustion_keeps_last_evaluated_snapshot_and_stops_later_stages(limited, stop, expected):
    controls = _controls(limited, kind="he2")
    nested = getattr(controls[0], stop)
    solver = nested.solver
    solver.maximum_iterations = 1
    nested.solver = solver
    setattr(controls[0], stop, nested)
    events = []
    result = _run(limited, controls=controls, callback=events.append)
    _assert_stages(result, expected)
    assert not result.converged and not result.periodic_energy_per_cell
    assert getattr(result, stop).solver.final_snapshot.iteration == 1
    assert result.diagnostics.physical_particle_hole_evaluated == expected[1]
    if expected[1]:
        assert result.pair_ccsd.particle_hole_physical_source_certified
    assert events[-1].stage == core._PeriodicGaussianSelectedLocalCCSDTStage.FINISHED
    for name in ("correlation_energy_per_cell", "total_energy_per_cell"):
        with pytest.raises((ValueError, RuntimeError), match="converg|complete|energy|cell"):
            getattr(result, name)


def test_explicit_zero_spaces_still_consume_every_ph_source_slot(small):
    controls = _controls(small, pair_cutoff=1.0, singles_cutoff=1.0, tno_cutoff=1.0)
    result = _run(small, controls=controls)
    _assert_stages(result)
    assert result.converged and result.pair_ccsd.particle_hole_physical_source_certified
    f = result.pair_ccsd.solver.final_snapshot
    assert f.correlation_energy == f.singles_residual_norm == f.doubles_residual_norm == 0
    assert f.particle_hole_source_slots == 2*result.pair_ccsd.memory.occupied_count*f.particle_hole_visits
    assert result.triple_spaces.diagnostics.maximum_retained_rank == 0
    assert result.local_triples.triples_energy_per_cell == 0


@pytest.mark.parametrize("field,plan_field", [
    ("maximum_owned_numerical_bytes", "owned_numerical_upper_bound"),
    ("maximum_per_worker_inventoried_bytes", "per_worker_inventoried_bytes_upper_bound"),
    ("maximum_node_inventoried_bytes", "node_inventoried_bytes_upper_bound"),
    ("maximum_progress_callbacks", "progress_callback_upper_bound"),
    ("maximum_work_units", "work_units_upper_bound"),
])
def test_whole_call_cap_minus_one_precedes_hf_basis_payload_and_first_callback(small, field, plan_field):
    controls = _controls(small)
    p = _plan(small, controls)
    setattr(controls[2], field, getattr(p, plan_field)-1)
    events = []
    # Wrong actual AO content would fail source verification if reached.
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap|budget|limit"):
        _run(small, controls=controls, callback=events.append, ao=small.auxiliary)
    assert events == []


def test_opt_in_requires_pair_branch_before_any_progress(small):
    controls = _controls(small)
    controls[0].pair_local_correlation = False
    events = []
    with pytest.raises((ValueError, RuntimeError), match="pair-local"):
        _run(small, controls=controls, callback=events.append)
    assert events == []


@pytest.mark.parametrize("field", ["maximum_source_slots", "maximum_work_units"])
def test_ph_controls_fail_closed_before_localization_and_source_payload(small, field):
    controls = _controls(small)
    caps = controls[2].particle_hole
    setattr(caps, field, 0)
    controls[2].particle_hole = caps
    events = []
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="positive|cap|control"):
        _run(small, controls=controls, callback=events.append, ao=small.auxiliary)
    assert events == []


def test_partial_common_selection_never_acquires_a_per_cell_total():
    b = _fixture(nk=2, full=False)
    result = _run(b, controls=_controls(b))
    _assert_stages(result)
    assert result.converged and result.pair_ccsd.particle_hole_physical_source_certified
    assert not result.diagnostics.complete_finite_torus_basis
    assert not result.periodic_energy_per_cell
    with pytest.raises((ValueError, RuntimeError), match="complete|cell|energy"):
        _ = result.total_energy_per_cell


@pytest.mark.parametrize("stage_name", ["PAIR_CCSD", "FINISHED"])
def test_cancellation_propagates_without_publishing_a_result(small, stage_name):
    wanted = getattr(core._PeriodicGaussianSelectedLocalCCSDTStage, stage_name)
    def cancel(event):
        if event.stage == wanted:
            raise RuntimeError("cancel physical selected chain")
    with pytest.raises(RuntimeError, match="cancel physical selected chain"):
        _run(small, controls=_controls(small), callback=cancel)


def test_new_controls_are_snapshotted_before_begin_callback(small):
    controls = _controls(small)
    baseline = _run(small, controls=controls)
    changed = []
    def mutate(event):
        if event.stage == core._PeriodicGaussianSelectedLocalCCSDTStage.BEGIN:
            controls[0].physical_particle_hole_ccsd = False
            config = controls[0].particle_hole
            config.auxiliary_block = 0
            controls[0].particle_hole = config
            nested = controls[0].particle_hole_options
            nested.maximum_overlap_imaginary_norm = float("nan")
            controls[0].particle_hole_options = nested
            cap = controls[2].particle_hole
            cap.maximum_source_slots = 0
            controls[2].particle_hole = cap
            changed.append(True)
    result = _run(small, controls=controls, callback=mutate)
    assert changed == [True]
    assert result.plan.physical_particle_hole_ccsd and result.pair_ccsd.particle_hole_physical_source_certified
    assert result.identity_sha256 == baseline.identity_sha256
    assert result.total_energy_per_cell == baseline.total_energy_per_cell


def test_unused_ph_options_preserve_default_pair_path_receipts(small):
    controls = _pair_controls(small)
    baseline = _run(small, controls=controls)
    config = core._PeriodicGaussianPairInteractionConfig()
    config.auxiliary_block = 0
    controls[0].particle_hole = config
    options = core._PeriodicGaussianPairInteractionOptions()
    options.maximum_overlap_imaginary_norm = float("nan")
    controls[0].particle_hole_options = options
    result = _run(small, controls=controls)
    assert not result.plan.physical_particle_hole_ccsd
    assert not result.diagnostics.physical_particle_hole_evaluated
    assert not result.pair_ccsd.split_bare_particle_hole
    assert result.identity_sha256 == baseline.identity_sha256
    assert result.pair_ccsd.identity_sha256 == baseline.pair_ccsd.identity_sha256
    assert result.total_energy_per_cell == baseline.total_energy_per_cell


def test_owned_result_survives_temporary_source_and_controls():
    b = _fixture(nk=1)
    controls = _controls(b)
    result = _run(b, controls=controls)
    cc, triples, energy = result.pair_ccsd, result.local_triples, result.total_energy_per_cell
    del b, controls, result
    gc.collect()
    assert cc.particle_hole_physical_source_certified and cc.converged
    assert triples.total_energy_per_cell == energy
