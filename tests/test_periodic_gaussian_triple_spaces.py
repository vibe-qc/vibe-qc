"""Actual-source CCSD doubles -> per-triple natural-orbital spaces.

No caller numerical arrays enter the actual factory. NumPy densities below
are independent tiny oracles; this batch produces neither amplitudes nor (T).
"""

from __future__ import annotations

import gc
from itertools import combinations_with_replacement

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_pair_mp2 import _physical, _run as _mp2_run, _controls as _mp2_controls
from tests.test_periodic_gaussian_pair_ccsd import _run as _ccsd_run, _controls as _ccsd_controls
from tests.test_bounded_restricted_triple_natural_orbitals import (
    _options as _geometry_options, _oracle as _geometry_oracle, _plan as _geometry_plan,
)
from tests.test_periodic_gaussian_frozen_core_correlation import _frozen_bundle
from tests.test_periodic_gaussian_selected_local_ccsd_t import _prepare_leaves, _he2_controls
from tests.test_periodic_gaussian_real_local_provider import _make as _provider


def _controls(*, cutoff=0.0):
    options = _geometry_options(occupation_cutoff=cutoff)
    live = core._PeriodicGaussianTripleSpacesLiveInventory()
    live.other_live_numerical_bytes_per_worker = 2**20
    live.other_live_control_bytes_per_worker = 65536
    live.fixed_backend_margin_bytes_per_worker = 65536
    caps = core._PeriodicGaussianTripleSpacesCaps()
    geometry = core._BoundedRestrictedTNOCaps()
    geometry.maximum_common_dimension = 4
    geometry.maximum_union_columns = 12
    geometry.maximum_owned_numerical_bytes = 2**20
    geometry.maximum_control_storage_bytes_per_replica = 2**22
    geometry.maximum_node_bytes, geometry.maximum_work_units = 2**26, 10**13
    caps.geometry = geometry
    caps.maximum_pair_count, caps.maximum_triple_count = 10, 20
    caps.maximum_owned_numerical_bytes = caps.maximum_control_storage_bytes_per_worker = 2**22
    caps.maximum_per_worker_inventoried_bytes, caps.maximum_node_inventoried_bytes = 2**24, 2**26
    caps.maximum_progress_callbacks, caps.maximum_work_units = 64, 10**15
    return options, live, caps


def _run(b, provider, mp2, ccsd, controls=None, callback=None):
    options, live, caps = _controls() if controls is None else controls
    return core._make_periodic_gaussian_triple_spaces(b.reference, b.basis, provider, mp2, ccsd,
        options, live, caps, callback)


def _plan(b, provider, mp2, ccsd, controls=None):
    options, live, caps = _controls() if controls is None else controls
    return core._plan_periodic_gaussian_triple_spaces(b.reference, b.basis, provider, mp2, ccsd,
        options, live, caps)


def _chain(system="he", nk=1, *, full=True):
    b, provider = _physical(system, nk, full=full)
    mp2 = _mp2_run(b, provider)
    ccsd = _ccsd_run(b, provider, mp2)
    assert mp2.converged and ccsd.converged
    return b, provider, mp2, ccsd


def _oracle_problem(b, mp2, ccsd, occupied, *, use_mp2_amplitudes=False):
    i, j, k = occupied
    labels = [(i, j), (i, k), (j, k)]
    solver = mp2.solver if use_mp2_amplitudes else ccsd.solver
    pairs = {label: (mp2.pair(*label).coefficients_copy(), solver.pair_copy(*label)) for label in set(labels)}
    o, n = ccsd.memory.occupied_count, ccsd.memory.common_virtual_dimension
    return dict(o=o, n=n, occupied=occupied, labels=labels,
        c=[pairs[label][0] for label in labels], t=[pairs[label][1] for label in labels],
        fvv=np.ascontiguousarray(b.basis.fock_copy()[o:, o:]))


def _check_batch(b, mp2, ccsd, result, cutoff=0.0):
    owned = 0
    total_rank = total_union = maximum_rank = 0
    for label in combinations_with_replacement(range(result.memory.occupied_count), 3):
        p = _oracle_problem(b, mp2, ccsd, label)
        _, occupations, c, eps, union = _geometry_oracle(p, cutoff)
        actual = result.triple_coefficients_copy(*label)
        diagnostics = result.triple_diagnostics(*label)
        assert diagnostics["union_rank"] == len(occupations)
        assert diagnostics["retained_rank"] == c.shape[1]
        assert diagnostics["usable"] == bool(c.shape[1])
        np.testing.assert_allclose(result.triple_occupations_copy(*label), occupations, atol=5e-12, rtol=3e-9)
        np.testing.assert_allclose(actual@actual.T, c@c.T, atol=5e-10, rtol=3e-9)
        np.testing.assert_allclose(actual.T@actual, np.eye(actual.shape[1]), atol=2e-10)
        np.testing.assert_allclose(result.triple_energies_copy(*label), eps, atol=5e-10, rtol=2e-9)
        np.testing.assert_allclose(actual.T@p["fvv"]@actual, np.diag(result.triple_energies_copy(*label)), atol=5e-10)
        assert diagnostics["semicanonical_projected_fock_relative_residual"] < 1e-10
        assert diagnostics["union_column_reconstruction_frobenius_error"] < 1e-9
        count = 8*(p["n"]*actual.shape[1]+actual.shape[1]+len(occupations))
        assert diagnostics["output_numerical_bytes"] == count
        owned += count
        total_rank += actual.shape[1]
        total_union += len(occupations)
        maximum_rank = max(maximum_rank, actual.shape[1])
        assert len(diagnostics["input_identity_sha256"]) == len(diagnostics["result_identity_sha256"]) == 64
        if cutoff == 0:
            np.testing.assert_allclose(actual@actual.T, union@union.T, atol=5e-10)
    assert result.diagnostics.retained_numerical_bytes == owned
    assert result.diagnostics.total_retained_rank == total_rank
    assert result.diagnostics.total_union_rank == total_union
    assert result.diagnostics.maximum_retained_rank == maximum_rank
    return owned


@pytest.mark.parametrize("system,nk", [("he", 1), ("he", 2), ("he", 3), ("he2", 1), ("he2", 2)])
def test_actual_converged_ccsd_density_all_canonical_multisets_and_original_fock(system, nk):
    b, provider, mp2, ccsd = _chain(system, nk)
    events = []
    result = _run(b, provider, mp2, ccsd, callback=events.append)
    _check_batch(b, mp2, ccsd, result)
    o, n = result.memory.occupied_count, result.memory.common_virtual_dimension
    labels = list(combinations_with_replacement(range(o), 3))
    assert result.diagnostics.completed_triples == len(labels) == o*(o+1)*(o+2)//6
    assert result.diagnostics.all_equal_tuple_count == o
    assert result.diagnostics.complete_common_finite_torus_basis
    assert result.diagnostics.all_unions_full_common_rank and result.diagnostics.all_retained_full_common_rank
    assert result.diagnostics.empty_union_count == result.diagnostics.empty_retained_space_count == 0
    assert result.ccsd_identity_sha256 == ccsd.identity_sha256
    assert result.ccsd_amplitude_payload_sha256 == ccsd.solver.payload_sha256
    assert result.warmstart_identity_sha256 == mp2.identity_sha256
    assert result.pair_spaces_identity_sha256 == mp2.pair_spaces_identity_sha256
    assert result.basis_identity_sha256 == b.basis.identity_sha256
    assert result.provider_identity_sha256 == provider.identity_sha256
    assert result.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256
    assert result.state is b.reference.state and result.context is b.context
    assert result.matched_finite_gaussian_hf_recipe
    assert not result.includes_triples_energy and not result.production_dlpno and not result.infinite_source_accuracy_certified
    assert not hasattr(result, "total_energy_per_cell")
    stage = core._PeriodicGaussianTripleSpacesStage
    assert events[0].stage == stage.BEGIN and events[-1].stage == stage.FINISHED
    completed = [e for e in events if e.stage == stage.TRIPLE_COMPLETE]
    assert [tuple(e.occupied) for e in completed] == labels
    assert [e.completed_triples for e in completed] == list(range(1, len(labels)+1))
    assert [e.callback_count for e in events] == list(range(1, len(events)+1))
    assert len(events) == result.diagnostics.completed_progress_callbacks <= result.memory.progress_callback_upper_bound
    assert all(e.union_rank == e.retained_rank == n for e in completed)


def test_converged_ccsd_not_mp2_amplitudes_define_tno_density():
    b, provider, mp2, ccsd = _chain("he2", 1)
    result = _run(b, provider, mp2, ccsd)
    differences = []
    for label in combinations_with_replacement(range(2), 3):
        _, correct, _, _, _ = _geometry_oracle(_oracle_problem(b, mp2, ccsd, label))
        _, initial, _, _, _ = _geometry_oracle(_oracle_problem(b, mp2, ccsd, label, use_mp2_amplitudes=True))
        actual = result.triple_occupations_copy(*label)
        np.testing.assert_allclose(actual, correct, atol=5e-12, rtol=3e-9)
        differences.append(np.max(abs(actual-initial)))
    assert max(differences) > 1e-10  # A concrete scientific wrong-input witness.


def test_pair_truncation_union_and_tno_occupation_truncation_are_separate():
    b, provider, full_mp2, _ = _chain("he2", 2)
    occupations = np.concatenate([full_mp2.pair(i, j).original_pno_occupations_copy()
        for i, j in combinations_with_replacement(range(4), 2)])
    mp2 = _mp2_run(b, provider, _mp2_controls(cutoff=float(np.median(occupations))))
    ccsd = _ccsd_run(b, provider, mp2)
    union = _run(b, provider, mp2, ccsd)
    _check_batch(b, mp2, ccsd, union)
    assert not mp2.diagnostics.all_pairs_full_rank
    assert any(union.triple_diagnostics(i, i, i)["union_rank"] < 4 for i in range(4))
    assert not union.diagnostics.all_unions_full_common_rank
    all_occupations = np.concatenate([union.triple_occupations_copy(*label)
        for label in combinations_with_replacement(range(4), 3)])
    cutoff = .5*float(all_occupations.max())
    assert cutoff > 0
    selected = _run(b, provider, mp2, ccsd, _controls(cutoff=cutoff))
    _check_batch(b, mp2, ccsd, selected, cutoff)
    assert selected.pair_spaces_identity_sha256 == union.pair_spaces_identity_sha256
    assert selected.ccsd_amplitude_payload_sha256 == union.ccsd_amplitude_payload_sha256
    assert selected.identity_sha256 != union.identity_sha256
    assert any(selected.triple_diagnostics(*label)["retained_rank"] < selected.triple_diagnostics(*label)["union_rank"]
        for label in combinations_with_replacement(range(4), 3))
    assert selected.diagnostics.empty_retained_space_count < selected.memory.triple_count
    zero = _run(b, provider, mp2, ccsd, _controls(cutoff=2*float(all_occupations.max())))
    assert zero.diagnostics.empty_retained_space_count == zero.memory.triple_count
    assert zero.diagnostics.total_retained_rank == zero.diagnostics.maximum_retained_rank == 0
    assert zero.diagnostics.total_union_rank > 0
    assert zero.diagnostics.retained_numerical_bytes == sum(8*zero.triple_diagnostics(*label)["union_rank"]
        for label in combinations_with_replacement(range(4), 3))


def test_all_empty_pair_union_stays_empty_with_no_diagonal_edge_invention():
    b, provider, full, _ = _chain("he2", 1)
    cutoff = 2*max(float(full.pair(i, j).original_pno_occupations_copy().max())
        for i, j in combinations_with_replacement(range(2), 2))
    mp2 = _mp2_run(b, provider, _mp2_controls(cutoff=cutoff))
    ccsd = _ccsd_run(b, provider, mp2)
    assert ccsd.diagnostics.all_singles_full_rank  # These spaces must NOT supplement the triples union.
    result = _run(b, provider, mp2, ccsd)
    assert result.diagnostics.empty_union_count == result.diagnostics.empty_retained_space_count == 4
    assert result.diagnostics.retained_numerical_bytes == 0
    assert result.diagnostics.total_retained_rank == result.diagnostics.total_union_rank == result.diagnostics.maximum_retained_rank == 0
    assert result.diagnostics.all_equal_tuple_count == 2
    for label in combinations_with_replacement(range(2), 3):
        assert result.triple_coefficients_copy(*label).shape == (2, 0)
        assert result.triple_occupations_copy(*label).size == result.triple_energies_copy(*label).size == 0


@pytest.mark.parametrize("nk", [1, 2])
def test_frozen_core_actual_owners_keep_only_active_occupied_triples(nk):
    b = _frozen_bundle((nk, 1, 1))
    _prepare_leaves(b, localization_options=_he2_controls(b)[0].localization)
    provider = _provider(b)
    mp2 = _mp2_run(b, provider)
    ccsd = _ccsd_run(b, provider, mp2)
    result = _run(b, provider, mp2, ccsd)
    assert (result.memory.occupied_count, result.memory.common_virtual_dimension) == (nk, 2*nk)
    assert result.memory.triple_count == nk*(nk+1)*(nk+2)//6
    assert result.state.n_frozen_core == result.state.n_correlated_occupied == 1
    assert result.state.reference_energy_per_cell == b.unfrozen_hf.state.reference_energy_per_cell
    _check_batch(b, mp2, ccsd, result)
    with pytest.raises((ValueError, IndexError), match="range|canonical|triple|occupied"):
        result.triple_diagnostics(0, 0, nk)


def test_complete_input_owner_inventory_and_serialized_leaf_upper():
    b, provider, mp2, ccsd = _chain("he2", 1)
    controls = _controls()
    plan = _plan(b, provider, mp2, ccsd, controls)
    result = _run(b, provider, mp2, ccsd, controls)
    n, q = plan.common_virtual_dimension, plan.triple_count
    assert plan.borrowed_mp2_numerical_bytes == mp2.diagnostics.retained_pair_bytes+mp2.solver.memory.output_numerical_bytes
    assert plan.borrowed_ccsd_numerical_bytes == ccsd.diagnostics.retained_singles_bytes+ccsd.solver.memory.output_numerical_bytes
    assert plan.retained_triple_output_upper_bytes == q*(8*n*n+16*n)
    assert result.diagnostics.retained_numerical_bytes == plan.retained_triple_output_upper_bytes
    assert plan.retained_triple_seal_bytes == q*2*65
    assert plan.borrowed_mp2_control_bytes > 0 and plan.borrowed_ccsd_control_bytes > 0
    assert plan.exact_rank_plan_output_upper_bytes <= plan.retained_triple_output_upper_bytes
    assert plan.exact_rank_plan_peak_owned_bytes <= plan.peak_owned_numerical_bytes
    assert plan.exact_rank_plan_geometry_work_units <= plan.work_units_upper_bound
    geometry_work = sum(_geometry_plan(_oracle_problem(b, mp2, ccsd, label), controls[0]).work_units_upper_bound
        for label in combinations_with_replacement(range(plan.occupied_count), 3))
    assert result.diagnostics.geometry_work_units == plan.exact_rank_plan_geometry_work_units == geometry_work
    assert plan.required_node_memory_bytes == plan.reference_base_node_bytes+plan.replicas_per_node*plan.per_worker_inventoried_bytes
    controls[1].other_live_numerical_bytes_per_worker += 777
    enlarged = _plan(b, provider, mp2, ccsd, controls)
    assert enlarged.per_worker_inventoried_bytes-plan.per_worker_inventoried_bytes == 777
    assert enlarged.required_node_memory_bytes-plan.required_node_memory_bytes == 777*plan.replicas_per_node


_OUTER_CAPS = dict(maximum_pair_count="pair_count", maximum_triple_count="triple_count",
    maximum_owned_numerical_bytes="peak_owned_numerical_bytes",
    maximum_control_storage_bytes_per_worker="control_storage_reservation_bytes",
    maximum_per_worker_inventoried_bytes="per_worker_inventoried_bytes",
    maximum_node_inventoried_bytes="required_node_memory_bytes",
    maximum_progress_callbacks="progress_callback_upper_bound", maximum_work_units="work_units_upper_bound")


@pytest.mark.parametrize("field", list(_OUTER_CAPS))
def test_exact_outer_caps_and_one_less_reject_before_progress(field):
    b, provider, mp2, ccsd = _chain()
    controls = _controls()
    value = getattr(_plan(b, provider, mp2, ccsd, controls), _OUTER_CAPS[field])
    setattr(controls[2], field, value)
    assert _run(b, provider, mp2, ccsd, controls).diagnostics.completed_triples == 1
    setattr(controls[2], field, value-1)
    events = []
    with pytest.raises((ValueError, OverflowError), match="cap"):
        _run(b, provider, mp2, ccsd, controls, events.append)
    assert not events


@pytest.mark.parametrize("field", ["maximum_common_dimension", "maximum_union_columns", "maximum_owned_numerical_bytes",
    "maximum_control_storage_bytes_per_replica", "maximum_node_bytes", "maximum_work_units"])
def test_nested_geometry_caps_precede_first_callback(field):
    b, provider, mp2, ccsd = _chain()
    controls = _controls()
    setattr(controls[2].geometry, field, 0)
    events = []
    with pytest.raises((ValueError, OverflowError), match="cap|ceiling|geometry"):
        _run(b, provider, mp2, ccsd, controls, events.append)
    assert not events


def test_exact_foreign_owner_and_unconverged_ccsd_rejections():
    b, provider, mp2, ccsd = _chain()
    foreign_b, foreign_provider, foreign_mp2, foreign_ccsd = _chain()
    for p, warm, state in ((foreign_provider, mp2, ccsd), (provider, foreign_mp2, ccsd),
                           (provider, mp2, foreign_ccsd)):
        with pytest.raises((ValueError, RuntimeError), match="owner|source|frame|receipt|match|CCSD|ccsd"):
            _plan(b, p, warm, state)
    incomplete = _ccsd_run(b, provider, mp2, _ccsd_controls(provider, iterations=1))
    assert not incomplete.converged
    with pytest.raises((ValueError, RuntimeError), match="converged|convergence|CCSD|ccsd"):
        _plan(b, provider, mp2, incomplete)
    unfinished_mp2 = _mp2_run(b, provider, _mp2_controls(iterations=1))
    assert not unfinished_mp2.converged
    with pytest.raises((ValueError, RuntimeError), match="converged|convergence|MP2|warm|match"):
        _plan(b, provider, unfinished_mp2, ccsd)


def test_partial_common_basis_is_not_promoted_to_periodic_completion():
    b, provider, mp2, ccsd = _chain("he", 2, full=False)
    result = _run(b, provider, mp2, ccsd)
    assert not result.diagnostics.complete_common_finite_torus_basis
    assert not result.includes_triples_energy and not result.production_dlpno
    _check_batch(b, mp2, ccsd, result)


def test_callback_cancellation_controls_snapshot_and_no_mutable_child_escape():
    b, provider, mp2, ccsd = _chain("he2", 1)
    controls = _controls()
    baseline = _run(b, provider, mp2, ccsd, controls)
    def mutate_controls(event):
        controls[0].occupation_cutoff = np.nan
        controls[1].fixed_backend_margin_bytes_per_worker = 0
        controls[2].maximum_triple_count = 0
    copied = _run(b, provider, mp2, ccsd, controls, mutate_controls)
    assert copied.identity_sha256 == baseline.identity_sha256
    for stage in (core._PeriodicGaussianTripleSpacesStage.BEGIN,
                  core._PeriodicGaussianTripleSpacesStage.TRIPLE_COMPLETE,
                  core._PeriodicGaussianTripleSpacesStage.FINISHED):
        def cancel(event):
            if event.stage == stage:
                raise RuntimeError("intentional actual TNO cancellation")
        with pytest.raises(RuntimeError, match="intentional actual TNO"):
            _run(b, provider, mp2, ccsd, callback=cancel)
    with pytest.raises(ValueError, match="callable"):
        _run(b, provider, mp2, ccsd, callback=17)
    assert not hasattr(baseline, "triple") and not hasattr(baseline, "semicanonical")
    original = baseline.triple_coefficients_copy(0, 0, 1)
    baseline.triple_coefficients_copy(0, 0, 1)[:] = 999
    diagnostic = baseline.triple_diagnostics(0, 0, 1)
    diagnostic["retained_rank"] = 999
    assert baseline.triple_diagnostics(0, 0, 1)["retained_rank"] == original.shape[1]
    options = baseline.options
    options.occupation_cutoff = 999
    assert baseline.options.occupation_cutoff == 0
    state = baseline.state
    del b, provider, mp2, ccsd, copied
    gc.collect()
    np.testing.assert_array_equal(baseline.triple_coefficients_copy(0, 0, 1), original)
    assert state.n_correlated_occupied == 2
    with pytest.raises((ValueError, IndexError), match="canonical|sorted|triple|occupied"):
        baseline.triple_diagnostics(1, 0, 0)
