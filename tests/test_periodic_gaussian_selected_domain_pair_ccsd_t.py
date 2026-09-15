"""One actual-HF native call through selected generation domains and (T).

These tiny finite-source diagnostics are not production solid benchmarks.
All domains, embeddings and correlation leaves are made inside C++.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_selected_local_ccsd_t import _run, _plan
from tests.test_periodic_gaussian_selected_local_pair_ccsd_t import (
    _fixture, _controls as _common_controls, _assert_stages, _check_receipts,
)
from tests.test_periodic_gaussian_domain_pair_mp2 import _controls as _domain_mp2_controls
from tests.test_periodic_gaussian_occupied_pao_domain import _controls as _occupied_controls
from tests.test_periodic_gaussian_pair_domain_builder import _builder_caps
from tests.test_periodic_gaussian_domain_pair_ccsd_t import _EMBEDDING_FIELDS
from tests.test_periodic_gaussian_embedded_pair_pnos import _options as _embedding_options, _caps as _embedding_caps
from tests.test_periodic_gaussian_frozen_core_correlation import _frozen_bundle


def _controls(b, *, he2=False, full=False, cut=.2, tail=.1, pair_cutoff=0.0):
    options, live, caps = _common_controls(b, he2=he2)
    options.selected_pair_domains = True
    options.occupied_domains = _occupied_controls(b, full=full, cut=cut, tail=tail)[0]
    mp2, _, mc = _domain_mp2_controls(b, cutoff=pair_cutoff)
    options.domain_pair_mp2, caps.domain_pair_mp2 = mp2, mc
    builder = _builder_caps(b)
    builder.maximum_owned_numerical_bytes = 2**21
    builder.maximum_control_storage_bytes = 2**23
    occupied = builder.occupied
    selector = occupied.selector
    selector.maximum_control_storage_bytes = 2**23
    occupied.selector, builder.occupied = selector, occupied
    caps.pair_domain_builder = builder
    cc = options.pair_ccsd
    audits = core._PeriodicGaussianPairCCSDEmbeddingOptions()
    source = _embedding_options()
    for name in _EMBEDDING_FIELDS:
        setattr(audits, name, getattr(source, name))
    cc.singles_embedding = audits
    options.pair_ccsd = cc
    ccc = caps.pair_ccsd
    embedded = _embedding_caps()
    embedded.maximum_control_storage_bytes_per_worker = 2**23
    embedded.maximum_per_worker_inventoried_bytes = 2**25
    ccc.embedded_singles = embedded
    caps.pair_ccsd = ccc
    # Admission ceilings, not allocated slabs. Keep the complete tiny
    # one-call plan within the binding's 16MiB owned / 128MiB node guards.
    caps.maximum_owned_numerical_bytes = 2**24
    caps.maximum_per_worker_inventoried_bytes = 2**25
    outer = _plan(b, (options, live, caps))
    sc, tc = caps.triple_spaces, caps.local_triples
    sc.maximum_control_storage_bytes_per_worker = outer.control_storage_reservation_bytes
    geometry = sc.geometry
    geometry.maximum_control_storage_bytes_per_replica = outer.control_storage_reservation_bytes
    sc.geometry = geometry
    tc.maximum_control_storage_bytes_per_worker = outer.control_storage_reservation_bytes
    caps.triple_spaces, caps.local_triples = sc, tc
    return options, live, caps


@pytest.mark.parametrize("kind,nk", [("he", 2), ("he2", 1), ("frozen", 1)])
def test_full_domain_one_call_matches_existing_common_frame_chain(kind, nk):
    b = _frozen_bundle((nk, 1, 1)) if kind == "frozen" else _fixture(kind, nk)
    assert not hasattr(b, "basis") and not hasattr(b, "localization")
    controls = _controls(b, he2=kind != "he", full=True)
    events = []
    result = _run(b, controls=controls, callback=events.append)
    _assert_stages(result)
    _check_receipts(b, result)
    assert result.converged and result.periodic_energy_per_cell
    assert result.plan.selected_pair_domains and result.pair_mp2.domain_generated
    assert result.pair_ccsd.memory.domain_generated
    assert result.pair_mp2.diagnostics.all_pairs_full_rank
    assert result.pair_ccsd.diagnostics.all_singles_full_rank
    baseline = _run(b, controls=_common_controls(b, he2=kind != "he"))
    assert baseline.converged and not baseline.plan.selected_pair_domains
    assert result.total_energy_per_cell == pytest.approx(baseline.total_energy_per_cell, abs=1e-11)
    assert result.local_triples.triples_energy_per_cell == pytest.approx(
        baseline.local_triples.triples_energy_per_cell, abs=1e-12)
    assert result.diagnostics.pair_domain_builder.constructed_home_domains == b.reference.state.n_correlated_occupied
    assert result.diagnostics.pair_domain_builder_memory.worker_bytes <= result.plan.per_worker_inventoried_bytes_upper_bound
    stage = core._PeriodicGaussianSelectedLocalCCSDTStage
    sequence = [e.stage for e in events]
    assert sequence.index(stage.FACTORS_READY) < sequence.index(stage.PAIR_DOMAINS_READY) < sequence.index(stage.PAIR_MP2)
    assert sequence.count(stage.PAIR_DOMAINS_READY) == 1
    assert sequence[-1] == stage.FINISHED
    assert [e.callback_count for e in events] == list(range(1, len(events)+1))
    assert len(events) == result.diagnostics.completed_progress_callbacks <= result.plan.progress_callback_upper_bound
    assert controls[0].pair_ccsd.solver.maximum_integral_work_units_per_call == 0
    total = result.total_energy_per_cell
    del b, baseline
    gc.collect()
    assert result.total_energy_per_cell == total


def test_rectangular_pair_generations_run_all_stages_without_common_generation_fallback():
    b = _fixture("he2", 2)
    result = _run(b, controls=_controls(b, he2=True))
    _assert_stages(result)
    _check_receipts(b, result)
    assert result.converged and result.periodic_energy_per_cell
    mp2, cc = result.pair_mp2, result.pair_ccsd
    n, o = mp2.memory.common_virtual_dimension, mp2.memory.occupied_count
    assert any(mp2.diagonal_generation_embedding(i).memory.pair_dimension < n for i in range(o))
    assert not cc.diagnostics.all_singles_full_rank
    assert mp2.diagnostics.generation_dimension_sum < mp2.memory.pair_count*n
    for i in range(o):
        diagonal = mp2.diagonal_generation_embedding(i)
        assert cc.singles_embedding_identity_sha256(i) == diagonal.identity_sha256
        assert cc.singles_generation_dimension(i) == diagonal.memory.pair_dimension
        assert cc.singles_domain_generated(i)
    assert result.total_energy_per_cell == pytest.approx(b.hf.state.reference_energy_per_cell
        + cc.correlation_energy_per_cell + result.local_triples.triples_energy_per_cell, abs=2e-15)


def test_zero_mp2_pnos_do_not_destroy_independent_diagonal_generation_spaces():
    b = _fixture("he2", 1)
    result = _run(b, controls=_controls(b, he2=True, pair_cutoff=1.0))
    _assert_stages(result)
    assert result.converged and result.periodic_energy_per_cell
    assert result.pair_mp2.diagnostics.maximum_pair_rank == 0
    assert all(result.pair_ccsd.singles_coefficients_copy(i).shape[1] > 0 for i in range(2))
    assert result.triple_spaces.diagnostics.maximum_retained_rank == 0
    assert result.local_triples.triples_energy_per_cell == 0


@pytest.mark.parametrize("field,plan_field", [
    ("maximum_owned_numerical_bytes", "owned_numerical_upper_bound"),
    ("maximum_per_worker_inventoried_bytes", "per_worker_inventoried_bytes_upper_bound"),
    ("maximum_node_inventoried_bytes", "node_inventoried_bytes_upper_bound"),
    ("maximum_progress_callbacks", "progress_callback_upper_bound"),
    ("maximum_work_units", "work_units_upper_bound"),
])
def test_whole_call_admission_rejects_before_first_progress(field, plan_field):
    b = _fixture(nk=2)
    controls = _controls(b)
    plan = _plan(b, controls)
    setattr(controls[2], field, getattr(plan, plan_field)-1)
    events = []
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        _run(b, controls=controls, callback=events.append)
    assert events == []


def test_selected_domains_require_pair_branch_and_explicit_population_audits():
    b = _fixture(nk=2)
    controls = _controls(b)
    controls[0].pair_local_correlation = False
    with pytest.raises(ValueError, match="pair-local"):
        _plan(b, controls)
    controls[0].pair_local_correlation = True
    domains = controls[0].occupied_domains
    domains.normalization_tolerance = np.nan
    controls[0].occupied_domains = domains
    with pytest.raises(ValueError):
        _run(b, controls=controls)


def test_exhausted_domain_mp2_stops_before_ccsd_and_refuses_cell_energy():
    b = _fixture(nk=2)
    controls = _controls(b)
    mp2 = controls[0].domain_pair_mp2
    solver = mp2.solver
    solver.maximum_iterations = 1
    mp2.solver, controls[0].domain_pair_mp2 = solver, mp2
    result = _run(b, controls=controls)
    _assert_stages(result, (True, False, False, False))
    assert not result.converged and not result.periodic_energy_per_cell
    assert result.pair_mp2.domain_generated
    with pytest.raises((ValueError, RuntimeError)):
        _ = result.total_energy_per_cell


def test_cancel_after_builder_publication_and_original_input_mutation():
    b = _fixture(nk=2)
    controls = _controls(b)
    stage = core._PeriodicGaussianSelectedLocalCCSDTStage
    events = []
    def cancel(event):
        events.append(event.stage)
        if event.stage == stage.PAIR_DOMAINS_READY:
            raise LookupError("cancel at native domain publication")
    with pytest.raises(LookupError, match="native domain"):
        _run(b, controls=controls, callback=cancel)
    assert events[-1] == stage.PAIR_DOMAINS_READY and stage.PAIR_MP2 not in events
    def mutate(event):
        if event.stage == stage.PAIR_DOMAINS_READY:
            b.rows[0, 0] = 999
    with pytest.raises((ValueError, RuntimeError, IndexError), match="changed|range|label|selection"):
        _run(b, controls=controls, callback=mutate)
