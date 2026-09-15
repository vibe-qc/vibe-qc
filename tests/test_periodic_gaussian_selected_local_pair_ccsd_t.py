"""One actual-HF native call through pair MP2, CCSD, TNOs and coupled (T).

Only tiny finite-source He/He2 cells are exercised. No selected local
orbitals or post-HF leaves are prepared before the native entry point.
Independent dense oracles may construct their own leaves afterward.
"""

from __future__ import annotations

import gc
from itertools import combinations_with_replacement

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_selected_local_ccsd_t import (
    _bundle, _he2_bundle, _controls as _common_controls, _he2_controls,
    _run, _plan, _prepare_leaves,
)
from tests.test_periodic_gaussian_pair_mp2 import _controls as _mp2_controls
from tests.test_periodic_gaussian_pair_ccsd import _controls as _ccsd_controls
from tests.test_periodic_gaussian_triple_spaces import _controls as _spaces_controls
from tests.test_periodic_gaussian_local_triples import (
    _controls as _triples_controls, _oracle_problem, _check_snapshot, _full_rank_linear_oracle,
)
from tests.test_periodic_gaussian_frozen_core_correlation import _frozen_bundle


def _fixture(system="he", nk=1, *, full=True):
    return _he2_bundle((nk, 1, 1)) if system == "he2" else _bundle((nk, 1, 1), full=full)


def _controls(b, *, he2=False, pair_cutoff=0.0, singles_cutoff=0.0, tno_cutoff=0.0):
    options, live, caps = _he2_controls(b) if he2 else _common_controls(b)
    options.pair_local_correlation = True
    mp2, _, mc = _mp2_controls(cutoff=pair_cutoff)
    cc, _, ccc = _ccsd_controls(singles_cutoff=singles_cutoff)
    spaces, _, sc = _spaces_controls(cutoff=tno_cutoff)
    triples, _, tc = _triples_controls()
    # Each physical stage's explicit admission ceiling also bounds the
    # retained owner later stages borrow. Tight tiny ceilings keep their
    # conservative sum plus Fock/factors inside the existing 8MiB envelope.
    mc.maximum_owned_numerical_bytes = 2**20
    ccc.maximum_owned_numerical_bytes = 2**20
    sc.maximum_owned_numerical_bytes = 2**21
    tc.maximum_owned_numerical_bytes = 2**20
    options.pair_mp2, options.pair_ccsd = mp2, cc
    options.triple_spaces, options.local_triples = spaces, triples
    caps.pair_mp2, caps.pair_ccsd, caps.triple_spaces, caps.local_triples = mc, ccc, sc, tc
    caps.maximum_pair_control_storage_bytes = 2**21
    caps.maximum_pair_rank_padding_bytes = 65536
    # A nested stage borrows the enclosing driver's control reservations as
    # well as its own. The standalone 4MiB control ceiling is therefore a
    # few fixed objects short of this composed budget. Use the native outer
    # count-only plan; keep its 8MiB owned / 16MiB worker ceilings unchanged.
    outer = _plan(b, (options, live, caps))
    sc.maximum_control_storage_bytes_per_worker = outer.control_storage_reservation_bytes
    geometry = sc.geometry
    geometry.maximum_control_storage_bytes_per_replica = outer.control_storage_reservation_bytes
    sc.geometry = geometry
    tc.maximum_control_storage_bytes_per_worker = outer.control_storage_reservation_bytes
    caps.triple_spaces, caps.local_triples = sc, tc
    return options, live, caps


def _assert_stages(result, expected=(True, True, True, True)):
    names = ("pair_mp2", "pair_ccsd", "triple_spaces", "local_triples")
    assert result.plan.pair_local_correlation
    for name, present in zip(names, expected, strict=True):
        assert getattr(result, name+"_evaluated") == present
        if present:
            assert len(getattr(result, name).identity_sha256) == 64
        else:
            with pytest.raises((ValueError, RuntimeError), match="evaluated"):
                getattr(result, name)
    with pytest.raises((ValueError, RuntimeError), match="evaluated"):
        _ = result.correlation  # No hidden fallback to the old common branch.


def _check_receipts(b, result):
    assert result.matched_finite_gaussian_hf_recipe
    assert result.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256
    assert result.pair_mp2.provider_identity_sha256 == result.provider_identity_sha256
    assert result.pair_ccsd.provider_identity_sha256 == result.provider_identity_sha256
    assert result.triple_spaces.provider_identity_sha256 == result.provider_identity_sha256
    assert result.pair_ccsd.warmstart_identity_sha256 == result.pair_mp2.identity_sha256
    assert result.triple_spaces.ccsd_identity_sha256 == result.pair_ccsd.identity_sha256
    assert result.local_triples.ccsd_identity_sha256 == result.pair_ccsd.identity_sha256
    assert result.local_triples.triple_spaces_identity_sha256 == result.triple_spaces.identity_sha256
    assert not result.production_dlpno and not result.infinite_source_accuracy_certified
    assert not result.bitwise_hf_factor_consumption_verified


@pytest.mark.parametrize("system,nk", [("he", 1), ("he", 2), ("he", 3), ("he2", 1), ("he2", 2)])
def test_one_actual_hf_entry_matches_complete_common_ccsd_t_and_independent_coupled_oracle(system, nk):
    b = _fixture(system, nk)
    assert not hasattr(b, "basis") and not hasattr(b, "localization")
    controls = _controls(b, he2=system == "he2")
    assert controls[0].pair_ccsd.solver.maximum_integral_work_units_per_call == 0
    assert controls[0].local_triples.moments.maximum_integral_work_units_per_call == 0
    events = []
    result = _run(b, controls=controls, callback=events.append)
    _assert_stages(result)
    _check_receipts(b, result)
    assert result.converged and result.correlation_evaluated and result.periodic_energy_per_cell
    assert result.diagnostics.complete_finite_torus_basis
    assert result.pair_mp2.diagnostics.all_pairs_full_rank
    assert result.pair_ccsd.diagnostics.all_pairs_full_rank and result.pair_ccsd.diagnostics.all_singles_full_rank
    assert result.triple_spaces.diagnostics.all_retained_full_common_rank
    assert result.total_energy_per_cell == result.local_triples.total_energy_per_cell
    assert result.correlation_energy_per_cell == result.local_triples.correlation_energy_per_cell
    assert result.diagnostics.hf_energy_per_cell == b.hf.state.reference_energy_per_cell
    # The old actual-source connector is a separate full-common-space CCSD
    # solve. It does not reuse the new pair solver's returned amplitudes.
    baseline = _run(b, controls=_he2_controls(b) if system == "he2" else _common_controls(b))
    assert baseline.converged
    assert result.total_energy_per_cell == pytest.approx(baseline.total_energy_per_cell, abs=8e-12)
    assert result.local_triples.triples_energy_per_cell == pytest.approx(
        baseline.correlation.triples.final_snapshot.triples_energy/nk, abs=8e-13)
    _prepare_leaves(b, localization_options=controls[0].localization)
    p, common = _oracle_problem(b, result.pair_mp2, result.pair_ccsd, result.triple_spaces)
    _check_snapshot(p, result.local_triples, _full_rank_linear_oracle(p, common))
    stage = core._PeriodicGaussianSelectedLocalCCSDTStage
    assert events[0].stage == stage.BEGIN and events[-1].stage == stage.FINISHED
    sequence = [e.stage for e in events]
    stages = [stage.PAIR_MP2, stage.PAIR_CCSD, stage.TRIPLE_SPACES, stage.LOCAL_TRIPLES]
    assert [sequence.index(s) for s in stages] == sorted(sequence.index(s) for s in stages)
    assert stage.CORRELATION not in sequence
    assert [e.callback_count for e in events] == list(range(1, len(events)+1))
    assert len(events) == result.diagnostics.completed_progress_callbacks <= result.plan.progress_callback_upper_bound
    # The caller's pre-provider placeholder remains untouched.
    assert controls[0].pair_ccsd.solver.maximum_integral_work_units_per_call == 0
    assert controls[0].local_triples.moments.maximum_integral_work_units_per_call == 0


@pytest.mark.parametrize("stop,expected", [("pair_mp2", (True, False, False, False)),
    ("pair_ccsd", (True, True, False, False)), ("local_triples", (True, True, True, True))])
def test_each_exhausted_iteration_stops_before_later_stage_and_refuses_cell_energy(stop, expected):
    b = _fixture("he2")
    controls = _controls(b, he2=True)
    nested = getattr(controls[0], stop)
    solver = nested.solver
    solver.maximum_iterations = 1
    nested.solver = solver
    setattr(controls[0], stop, nested)
    events = []
    result = _run(b, controls=controls, callback=events.append)
    _assert_stages(result, expected)
    assert not result.converged and not result.periodic_energy_per_cell
    stopped = getattr(result, stop)
    assert not stopped.converged and stopped.solver.final_snapshot.iteration == 1
    stage = core._PeriodicGaussianSelectedLocalCCSDTStage
    for name, evaluated in zip(("PAIR_MP2", "PAIR_CCSD", "TRIPLE_SPACES", "LOCAL_TRIPLES"), expected, strict=True):
        assert (getattr(stage, name) in [e.stage for e in events]) == evaluated
    assert events[-1].stage == stage.FINISHED
    for name in ("correlation_energy_per_cell", "total_energy_per_cell"):
        with pytest.raises((ValueError, RuntimeError), match="converged|complete|energy|cell"):
            getattr(result, name)


def test_partial_torus_selected_common_space_is_not_a_per_cell_energy():
    b = _fixture(nk=2, full=False)
    result = _run(b, controls=_controls(b))
    _assert_stages(result)
    assert result.converged and not result.periodic_energy_per_cell
    assert not result.diagnostics.complete_finite_torus_basis
    assert not result.local_triples.periodic_energy_per_cell
    for name in ("correlation_energy_per_cell", "total_energy_per_cell"):
        with pytest.raises((ValueError, RuntimeError), match="complete|cell|energy"):
            getattr(result, name)


def test_explicit_pair_and_singles_cutoffs_then_positive_tno_selection_and_zero_tno():
    b = _fixture("he2")
    full = _run(b, controls=_controls(b, he2=True))
    pair_occ = np.concatenate([full.pair_mp2.pair(i, j).original_pno_occupations_copy()
        for i, j in combinations_with_replacement(range(2), 2)])
    singles_occ = np.concatenate([full.pair_ccsd.singles_original_pno_occupations_copy(i) for i in range(2)])
    pair_cutoff, singles_cutoff = float(np.median(pair_occ)), .5*float(singles_occ.max())
    assert pair_cutoff > 0 and singles_cutoff > 0
    controls = _controls(b, he2=True, pair_cutoff=pair_cutoff, singles_cutoff=singles_cutoff)
    selected = _run(b, controls=controls)
    assert selected.converged and selected.periodic_energy_per_cell
    assert not selected.pair_mp2.diagnostics.all_pairs_full_rank
    assert not selected.pair_ccsd.diagnostics.all_singles_full_rank
    assert selected.pair_mp2.pair(0, 0).pno_options.occupation_cutoff == pair_cutoff
    assert selected.pair_ccsd.singles_options(0).occupation_cutoff == singles_cutoff
    tno_occ = np.concatenate([selected.triple_spaces.triple_occupations_copy(*label)
        for label in combinations_with_replacement(range(2), 3)])
    cutoff = .5*float(tno_occ.max())
    # Explicitly assign the modified nested value so the test does not
    # depend on whether pybind returns a reference or a copy.
    options = controls[0].triple_spaces
    options.occupation_cutoff = cutoff
    controls[0].triple_spaces = options
    truncated = _run(b, controls=controls)
    assert truncated.converged and truncated.periodic_energy_per_cell
    assert truncated.triple_spaces.options.occupation_cutoff == cutoff
    assert truncated.pair_mp2.identity_sha256 == selected.pair_mp2.identity_sha256
    assert truncated.pair_ccsd.identity_sha256 == selected.pair_ccsd.identity_sha256
    assert truncated.triple_spaces.identity_sha256 != selected.triple_spaces.identity_sha256
    options.occupation_cutoff = 2*float(tno_occ.max())
    controls[0].triple_spaces = options
    zero = _run(b, controls=controls)
    _assert_stages(zero)
    assert zero.converged and zero.periodic_energy_per_cell
    assert zero.triple_spaces.diagnostics.maximum_retained_rank == 0
    assert not zero.local_triples.memory.moments_required
    assert zero.local_triples.diagnostics.completed_common_integral_calls == 0
    assert zero.local_triples.triples_energy_per_cell == 0
    assert zero.total_energy_per_cell == zero.pair_ccsd.total_energy_per_cell


@pytest.mark.parametrize("nk", [1, 2])
def test_frozen_core_outer_call_preserves_full_hf_constant_and_active_dimensions(nk):
    b = _frozen_bundle((nk, 1, 1))
    assert not hasattr(b, "basis")
    controls = _controls(b, he2=True)
    result = _run(b, controls=controls)
    _assert_stages(result)
    assert result.converged and result.periodic_energy_per_cell
    assert result.pair_ccsd.memory.occupied_count == result.local_triples.memory.occupied_count == nk
    assert result.local_triples.memory.common_virtual_dimension == 2*nk
    assert b.hf.state.reference_energy_per_cell == b.unfrozen_hf.state.reference_energy_per_cell
    assert result.total_energy_per_cell == pytest.approx(b.unfrozen_hf.state.reference_energy_per_cell
        +(result.pair_ccsd.solver.final_snapshot.correlation_energy
          +result.local_triples.solver.final_snapshot.triples_energy)/nk, abs=3e-15)
    _prepare_leaves(b, localization_options=controls[0].localization)
    p, common = _oracle_problem(b, result.pair_mp2, result.pair_ccsd, result.triple_spaces)
    _check_snapshot(p, result.local_triples, _full_rank_linear_oracle(p, common))


_OUTER_CAPS = dict(maximum_owned_numerical_bytes="owned_numerical_upper_bound",
    maximum_per_worker_inventoried_bytes="per_worker_inventoried_bytes_upper_bound",
    maximum_node_inventoried_bytes="node_inventoried_bytes_upper_bound",
    maximum_progress_callbacks="progress_callback_upper_bound", maximum_work_units="work_units_upper_bound")


@pytest.mark.parametrize("field", list(_OUTER_CAPS))
def test_complete_outer_cap_minus_one_precedes_progress_and_any_localization(field):
    b = _fixture()
    controls = _controls(b)
    p = _plan(b, controls)
    for cap, report in _OUTER_CAPS.items():
        setattr(controls[2], cap, getattr(p, report))
    assert _run(b, controls=controls).converged
    setattr(controls[2], field, getattr(controls[2], field)-1)
    events = []
    with pytest.raises((ValueError, RuntimeError), match="cap|budget|inventory"):
        _run(b, controls=controls, callback=events.append)
    assert events == []


def test_outer_phase_sum_control_and_rank_padding_are_explicit_inventory():
    b = _fixture()
    options, live, caps = controls = _controls(b)
    p = _plan(b, controls)
    stage_caps = sum(getattr(caps, name).maximum_owned_numerical_bytes
        for name in ("pair_mp2", "pair_ccsd", "triple_spaces", "local_triples"))
    assert stage_caps == 5*2**20
    assert p.correlation_phase_owned_upper_bound == (caps.basis.maximum_owned_numerical_bytes
        +caps.factors.resources.maximum_owned_numeric_bytes+stage_caps)
    assert p.owned_numerical_upper_bound <= caps.maximum_owned_numerical_bytes == 2**23
    assert p.pair_rank_padding_reservation_bytes == 65536
    assert p.per_worker_inventoried_bytes_upper_bound == (p.owned_numerical_upper_bound
        +p.pair_rank_padding_reservation_bytes+p.borrowed_input_bytes_upper_bound
        +p.control_storage_reservation_bytes+live.other_live_bytes_per_worker+live.fixed_backend_margin_bytes_per_worker)
    caps.maximum_pair_control_storage_bytes += 117
    q = _plan(b, controls)
    assert q.control_storage_reservation_bytes-p.control_storage_reservation_bytes == 117
    assert q.per_worker_inventoried_bytes_upper_bound-p.per_worker_inventoried_bytes_upper_bound == 117
    caps.maximum_pair_control_storage_bytes = 0
    events = []
    with pytest.raises(ValueError, match="positive|control|cap"):
        _run(b, controls=controls, callback=events.append)
    assert events == []


@pytest.mark.parametrize("stage_name", ["PAIR_MP2", "PAIR_CCSD", "TRIPLE_SPACES", "LOCAL_TRIPLES", "FINISHED"])
def test_cancellation_at_each_new_stage_publishes_no_result(stage_name):
    b = _fixture()
    target = getattr(core._PeriodicGaussianSelectedLocalCCSDTStage, stage_name)
    def cancel(event):
        if event.stage == target:
            raise RuntimeError("outer pair-chain cancellation witness")
    with pytest.raises(RuntimeError, match="pair-chain cancellation"):
        _run(b, controls=_controls(b), callback=cancel)
    assert _run(b, controls=_controls(b)).converged


def test_all_nested_controls_are_copied_and_owned_stage_views_survive_inputs():
    b = _fixture("he2")
    expected = _run(b, controls=_controls(b, he2=True))
    controls = _controls(b, he2=True)
    def mutate(_):
        controls[0].pair_local_correlation = False
        for name in ("pair_mp2", "pair_ccsd", "local_triples"):
            option = getattr(controls[0], name)
            solver = option.solver
            solver.maximum_iterations = 0
            option.solver = solver
            setattr(controls[0], name, option)
            cap = getattr(controls[2], name)
            cap.maximum_owned_numerical_bytes = 0
            setattr(controls[2], name, cap)
        tno = controls[0].triple_spaces
        tno.occupation_cutoff = np.nan
        controls[0].triple_spaces = tno
        controls[2].maximum_pair_control_storage_bytes = 0
        controls[2].maximum_pair_rank_padding_bytes = 0
    result = _run(b, controls=controls, callback=mutate)
    assert result.identity_sha256 == expected.identity_sha256
    _assert_stages(result)
    mp2, cc, spaces, triples = result.pair_mp2, result.pair_ccsd, result.triple_spaces, result.local_triples
    c, t = spaces.triple_coefficients_copy(0, 0, 1), triples.solver.amplitudes_copy(0, 0, 1)
    spaces.triple_coefficients_copy(0, 0, 1)[:] = 17
    triples.solver.amplitudes_copy(0, 0, 1)[:] = 19
    total = result.total_energy_per_cell
    del b, controls, expected, result
    gc.collect()
    assert mp2.converged and cc.converged and triples.converged
    assert triples.total_energy_per_cell == total
    np.testing.assert_array_equal(spaces.triple_coefficients_copy(0, 0, 1), c)
    np.testing.assert_array_equal(triples.solver.amplitudes_copy(0, 0, 1), t)
    assert not hasattr(spaces, "triple") and not hasattr(spaces, "semicanonical")


def test_actual_physical_input_mutation_at_new_stage_is_rejected():
    b = _fixture()
    changed = b.rows.copy()
    def mutate(event):
        if event.stage == core._PeriodicGaussianSelectedLocalCCSDTStage.PAIR_CCSD:
            changed[0, 0] = 99
    with pytest.raises((ValueError, RuntimeError, IndexError), match="selection|occupied|identity|changed|range|input"):
        _run(b, controls=_controls(b), occupied=changed, callback=mutate)


@pytest.mark.parametrize("field", ["domain", "occupied"])
@pytest.mark.parametrize("stage_name", ["BEGIN", "FINISHED"])
def test_pinned_selection_resize_is_rejected_before_native_pointer_reuse(field, stage_name):
    b = _fixture()
    target = (b.columns if field == "domain" else b.rows).copy()
    stage = getattr(core._PeriodicGaussianSelectedLocalCCSDTStage, stage_name)
    def resize(event):
        if event.stage == stage:
            target.resize((target.size+2,), refcheck=False)
    with pytest.raises(ValueError, match="selection descriptors"):
        _run(b, controls=_controls(b), callback=resize, **{field: target})


def test_strict_he2_localization_failure_does_not_construct_any_pair_stage():
    b = _fixture("he2")
    result = _run(b, controls=_controls(b))  # Existing strict PM1e-11, not the explicit He2 tolerance.
    _assert_stages(result, (False, False, False, False))
    assert not result.converged and not result.correlation_evaluated
    assert not result.periodic_energy_per_cell
    assert result.unfinished_localization.optimizer.status == core._PeriodicCorrelationIAOOptimizerStatus.LINE_SEARCH_FAILED
