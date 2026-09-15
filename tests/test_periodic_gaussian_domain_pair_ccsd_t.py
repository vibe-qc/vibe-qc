"""Actual distinct PAO generation frames through CCSD, TNO and coupled (T).

Dense arrays below are independent tiny oracles captured BEFORE the common
geometry moves into its native builder. None enter the physical consumers.
These tests do not certify production DLPNO scaling or infinite-source error.
"""

from __future__ import annotations

import gc
from itertools import combinations_with_replacement

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_domain_pair_mp2 import (
    _physical, _controls as _mp2_controls, _run as _mp2_run,
)
from tests.test_periodic_gaussian_pair_domain_builder import _case
from tests.test_periodic_gaussian_pair_mp2 import _run as _legacy_mp2
from tests.test_periodic_gaussian_pair_ccsd import (
    _controls as _common_ccsd_controls, _run as _ccsd_run, _plan as _ccsd_plan,
)
from tests.test_periodic_gaussian_embedded_pair_pnos import (
    _options as _embedded_options, _caps as _embedded_caps, _oracle as _singles_oracle,
)
from tests.test_bounded_restricted_pair_ccsd_solver import (
    _expand, _project, _projected_norms, _spin_residual,
)
from tests.test_bounded_restricted_ccsd_target import _legacy
from tests.test_bounded_restricted_ccsd_solver import _energy as _ccsd_energy
from tests.test_periodic_gaussian_triple_spaces import (
    _run as _spaces_run, _plan as _spaces_plan, _controls as _spaces_controls,
    _check_batch, _oracle_problem as _tno_problem,
)
from tests.test_bounded_restricted_triple_natural_orbitals import _oracle as _tno_oracle
from tests.test_periodic_gaussian_local_triples import (
    _run as _triples_run,
    _projected_common, _check_snapshot as _check_triples_snapshot,
    _full_rank_linear_oracle,
)
from tests.test_bounded_restricted_triples_moments import _einsum_oracle


_EMBEDDING_FIELDS = (
    "maximum_diagonal_exchange_projection_norm", "maximum_fock_symmetry_projection_norm",
    "maximum_exported_gram_error", "maximum_exported_fock_error", "maximum_exported_subspace_error",
)


def _controls(b, *, singles_cutoff=0.0, iterations=80):
    options, live, caps = _common_ccsd_controls(b.provider,
        singles_cutoff=singles_cutoff, iterations=iterations)
    source = _embedded_options()
    audits = core._PeriodicGaussianPairCCSDEmbeddingOptions()
    for field in _EMBEDDING_FIELDS:
        setattr(audits, field, getattr(source, field))
    options.singles_embedding = audits
    caps.embedded_singles = _embedded_caps()
    return options, live, caps


def _run(b, mp2, controls=None, callback=None):
    return _ccsd_run(b, b.provider, mp2,
        _controls(b) if controls is None else controls, callback)


def _plan(b, mp2, controls=None):
    return _ccsd_plan(b, b.provider, mp2, _controls(b) if controls is None else controls)


def _problem(b, mp2, ccsd):
    """Independent dense AO-factor Hamiltonian, not native provider rows."""
    p = dict(b.expected_dense)
    o, n = p["o"], p["v"]
    gram = p["eri"].reshape((o+n)**2, (o+n)**2)
    np.testing.assert_allclose(gram, gram.T, atol=4e-12, rtol=4e-12)
    values, vectors = np.linalg.eigh((gram+gram.T)/2)
    assert values.min() > -4e-12
    factors = (np.sqrt(np.maximum(values, 0))[:, None]*vectors.T).reshape(-1, o+n, o+n)
    np.testing.assert_allclose(np.einsum("Ppq,Prs->pqrs", factors, factors), p["eri"], atol=5e-12, rtol=5e-12)
    fock = b.basis.fock_copy()
    np.testing.assert_allclose(fock, p["fock"], atol=1e-10, rtol=0)
    labels = list(combinations_with_replacement(range(o), 2))
    sc = [ccsd.singles_coefficients_copy(i) for i in range(o)]
    pc = [mp2.pair(*label).coefficients_copy() for label in labels]
    p.update(n=n, labels=labels, eri=np.ascontiguousarray(p["eri"]), factors=np.ascontiguousarray(factors),
        foo=np.ascontiguousarray(fock[:o, :o]), fvv=np.ascontiguousarray(fock[o:, o:]),
        fov=np.ascontiguousarray(fock[:o, o:]), sc=sc, pc=pc,
        sr=[c.shape[1] for c in sc], pr=[c.shape[1] for c in pc])
    return p


def _check_ccsd(b, mp2, result):
    p = _problem(b, mp2, result)
    common = _expand(p, [result.solver.singles_copy(i) for i in range(p["o"])],
        [result.solver.pair_copy(*label) for label in p["labels"]])
    r1, r2 = _legacy(common)
    norms = _projected_norms(p, *_project(p, r1, r2))
    snapshot = result.solver.final_snapshot
    for field, value in zip(("singles_max_residual", "doubles_max_residual",
            "singles_residual_norm", "doubles_residual_norm"), norms, strict=True):
        assert getattr(snapshot, field) == pytest.approx(value, abs=1e-11)
    energy = _ccsd_energy(common, common["t1"], common["t2"])
    assert snapshot.correlation_energy == pytest.approx(energy, abs=1e-11)
    sr1, sr2, se = _spin_residual(common)
    np.testing.assert_allclose(sr1, r1, atol=2e-11, rtol=2e-10)
    np.testing.assert_allclose(sr2, r2, atol=2e-11, rtol=2e-10)
    assert se == pytest.approx(energy, abs=1e-11)
    if result.converged:
        assert max(norms[:2]) < 2e-11
    return common


def _check_singles(b, mp2, ccsd, cutoff=0.0):
    n = ccsd.memory.common_virtual_dimension
    generation_sum = retained = local_coefficients = 0
    for i in range(ccsd.memory.occupied_count):
        embedding = mp2.diagonal_generation_embedding(i)
        expected = _singles_oracle(b, b.expected_dense, embedding, i, i)
        values = expected["occupations"]
        chosen = np.ones(len(values), bool) if cutoff == 0 else values > cutoff
        local = expected["pnos"][:, chosen]
        wanted = expected["x"]@local
        c = ccsd.singles_coefficients_copy(i)
        eps = ccsd.singles_energies_copy(i)
        m, r = embedding.memory.pair_dimension, int(chosen.sum())
        assert c.shape == (n, r) and eps.shape == (r,)
        assert ccsd.singles_generation_dimension(i) == m
        assert ccsd.singles_domain_generated(i)
        assert ccsd.singles_embedding_identity_sha256(i) == embedding.identity_sha256
        assert ccsd.singles_options(i).occupation_cutoff == cutoff
        np.testing.assert_allclose(ccsd.singles_original_pno_occupations_copy(i), values, atol=4e-12, rtol=3e-9)
        np.testing.assert_allclose(c@c.T, wanted@wanted.T, atol=3e-9, rtol=3e-9)
        np.testing.assert_allclose(c.T@c, np.eye(r), atol=5e-11)
        np.testing.assert_allclose(c.T@expected["common_f"]@c, np.diag(eps), atol=5e-11, rtol=5e-11)
        np.testing.assert_allclose(eps, np.linalg.eigvalsh(local.T@expected["f"]@local), atol=5e-11, rtol=5e-11)
        assert ccsd.singles_retained_numerical_bytes(i) == 8*(n*r+m*r+r+m)
        # Nested generation diagnostics count LOCAL D[m,r], not both D and C.
        assert ccsd.singles_diagnostics(i).retained_output_bytes == 8*(m*r+r+m)
        generation_sum += m
        retained += 8*(n*r+m*r+r+m)
        local_coefficients += 8*m*r
    assert generation_sum == ccsd.memory.singles_generation_dimension_sum
    assert generation_sum == ccsd.diagnostics.singles_generation_dimension_sum
    assert retained == ccsd.diagnostics.retained_singles_bytes
    assert local_coefficients == ccsd.diagnostics.retained_singles_generation_coefficient_bytes


def _triples_problem(common, spaces, snapshot):
    assert np.linalg.norm(common["fov"]) < 1e-10
    common = dict(common, fov=np.zeros_like(common["fov"]))
    labels = list(combinations_with_replacement(range(common["o"]), 3))
    cs = [spaces.triple_coefficients_copy(*label) for label in labels]
    moments = [_einsum_oracle(_projected_common(common, c), label)
        for label, c in zip(labels, cs, strict=True)]
    return dict(o=common["o"], n=common["v"], labels=labels, ranks=[c.shape[1] for c in cs], c=cs,
        eps=[spaces.triple_energies_copy(*label) for label in labels], foo=common["foo"], fvv=common["fvv"],
        w=[m[0] for m in moments], u=[m[1] for m in moments], snapshot=snapshot), common


@pytest.mark.parametrize("kind,nk", [("he", 2), ("he2", 1), ("frozen", 1)])
def test_full_generation_limit_matches_common_ccsd_and_independent_coupled_triples(kind, nk):
    b = _physical(nk, kind=kind, full=True)
    mp2 = _mp2_run(b)
    result = _run(b, mp2)
    assert mp2.converged and result.converged and result.memory.domain_generated
    assert result.diagnostics.all_singles_full_rank and result.diagnostics.all_pairs_full_rank
    _check_singles(b, mp2, result)
    common = _check_ccsd(b, mp2, result)
    old_mp2 = _legacy_mp2(b, b.provider)
    old = _ccsd_run(b, b.provider, old_mp2)
    assert old.converged
    old_common = _check_ccsd(b, old_mp2, old)
    np.testing.assert_allclose(common["t1"], old_common["t1"], atol=2e-11, rtol=2e-8)
    np.testing.assert_allclose(common["t2"], old_common["t2"], atol=2e-11, rtol=2e-8)
    assert result.correlation_energy_per_cell == pytest.approx(old.correlation_energy_per_cell, abs=1e-11)
    spaces = _spaces_run(b, b.provider, mp2, result)
    _check_batch(b, mp2, result, spaces)
    triples = _triples_run(b, b.provider, mp2, result, spaces)
    assert triples.converged and triples.periodic_energy_per_cell
    p, common = _triples_problem(common, spaces, result.solver.final_snapshot.iteration)
    _check_triples_snapshot(p, triples, _full_rank_linear_oracle(p, common))
    assert triples.total_energy_per_cell == pytest.approx(b.hf.state.reference_energy_per_cell
        + result.correlation_energy_per_cell + triples.triples_energy_per_cell, abs=2e-15)
    if kind == "frozen":
        assert b.reference.state.n_frozen_core == b.reference.state.n_correlated_occupied == 1
        assert result.memory.occupied_count == nk
        assert b.hf.state.reference_energy_per_cell == b.unfrozen_hf.state.reference_energy_per_cell
    if kind == "he2":
        differences = []
        for label in p["labels"]:
            correct = _tno_oracle(_tno_problem(b, mp2, result, label))[1]
            wrong = _tno_oracle(_tno_problem(b, mp2, result, label, use_mp2_amplitudes=True))[1]
            differences.append(np.max(abs(correct-wrong), initial=0))
        assert max(differences) > 1e-10  # Connected CCSD T2, never MP2 T2, defines TNO density.
    total = triples.total_energy_per_cell
    del b, mp2, result, spaces, old, old_mp2
    gc.collect()
    assert triples.total_energy_per_cell == total


@pytest.fixture(scope="module")
def rectangular():
    # One four-occupied/four-virtual CCSD solve, shared by the rectangular
    # tests. No repeated expensive full He2K2 common-space solve is needed.
    b = _physical(2, kind="he2", full=False, cut=.2, tail=.1)
    mp2 = _mp2_run(b)
    ccsd = _run(b, mp2)
    assert mp2.converged and ccsd.converged
    return b, mp2, ccsd


def test_rectangular_generations_preserve_exact_payloads_and_dense_projected_residual(rectangular):
    b, mp2, result = rectangular
    n, o = result.memory.common_virtual_dimension, result.memory.occupied_count
    assert any(mp2.diagonal_generation_embedding(i).memory.pair_dimension < n for i in range(o))
    assert not result.diagnostics.all_singles_full_rank
    _check_singles(b, mp2, result)
    _check_ccsd(b, mp2, result)
    expected_mp2 = (mp2.diagnostics.retained_pair_bytes + mp2.solver.memory.output_numerical_bytes
        + mp2.diagnostics.retained_pair_geometry_bytes)
    assert result.memory.borrowed_mp2_numerical_bytes == expected_mp2
    assert result.memory.borrowed_generation_embedding_bytes == 8*(n+1)*result.memory.singles_generation_dimension_sum
    assert result.memory.solver_upper.per_replica_inventoried_bytes <= result.memory.per_worker_inventoried_bytes
    assert result.solver.memory.per_replica_inventoried_bytes <= result.memory.per_worker_inventoried_bytes
    plan = _spaces_plan(b, b.provider, mp2, result)
    assert plan.borrowed_mp2_numerical_bytes == expected_mp2
    assert plan.borrowed_ccsd_numerical_bytes == result.diagnostics.retained_singles_bytes+result.solver.memory.output_numerical_bytes
    assert result.warmstart_identity_sha256 == mp2.identity_sha256
    assert result.pair_spaces_identity_sha256 == mp2.pair_spaces_identity_sha256
    assert not result.production_dlpno and not result.infinite_source_accuracy_certified


def test_rectangular_tno_density_and_zero_tno_moments_preserve_original_ccsd_energy(rectangular):
    b, mp2, ccsd = rectangular
    spaces = _spaces_run(b, b.provider, mp2, ccsd)
    _check_batch(b, mp2, ccsd, spaces)
    occupations = np.concatenate([spaces.triple_occupations_copy(*label)
        for label in combinations_with_replacement(range(ccsd.memory.occupied_count), 3)])
    cutoff = 2*float(occupations.max())
    assert cutoff > 0
    empty = _spaces_run(b, b.provider, mp2, ccsd, _spaces_controls(cutoff=cutoff))
    _check_batch(b, mp2, ccsd, empty, cutoff)
    result = _triples_run(b, b.provider, mp2, ccsd, empty)
    assert result.converged and not result.memory.moments_required
    assert result.diagnostics.completed_common_integral_calls == result.memory.common_integral_calls_upper_bound == 0
    assert result.solver.final_snapshot.moment_visits == 0
    assert result.triples_energy_per_cell == 0 and result.total_energy_per_cell == ccsd.total_energy_per_cell
    assert result.memory.complete_borrowed_owner_numerical_bytes == (
        b.basis.memory.retained_output_bytes + b.provider.memory.retained_row_bytes
        + ccsd.memory.borrowed_mp2_numerical_bytes + ccsd.diagnostics.retained_singles_bytes
        + ccsd.solver.memory.output_numerical_bytes + empty.diagnostics.retained_numerical_bytes)


def test_independent_singles_use_original_diagonal_embedding_even_when_double_rank_is_zero():
    b = _physical(1, kind="he2", full=True)
    mp2 = _mp2_run(b, _mp2_controls(b, cutoff=1.0))
    assert mp2.converged and mp2.diagnostics.zero_rank_pairs == mp2.memory.pair_count
    pairs_sha = mp2.pair_spaces_identity_sha256
    singles = _run(b, mp2)
    assert singles.converged and singles.solver.memory.total_doubles_elements == 0
    assert singles.solver.memory.total_singles_elements > 0
    _check_singles(b, mp2, singles)
    _check_ccsd(b, mp2, singles)
    for i in range(singles.memory.occupied_count):
        assert singles.solver.singles_rank(i) == mp2.diagonal_generation_embedding(i).memory.pair_dimension > 0
        assert mp2.solver.pair_rank(i, i) == 0
    zero = _run(b, mp2, _controls(b, singles_cutoff=1.0))
    assert zero.converged and zero.diagnostics.zero_rank_singles == zero.memory.occupied_count
    assert zero.solver.final_snapshot.correlation_energy == 0
    _check_singles(b, mp2, zero, 1.0)
    assert zero.singles_spaces_identity_sha256 != singles.singles_spaces_identity_sha256
    assert mp2.pair_spaces_identity_sha256 == pairs_sha == zero.pair_spaces_identity_sha256
    spaces = _spaces_run(b, b.provider, mp2, singles)
    assert spaces.diagnostics.total_union_rank == spaces.diagnostics.maximum_retained_rank == 0
    triples = _triples_run(b, b.provider, mp2, singles, spaces)
    assert triples.converged and triples.diagnostics.completed_common_integral_calls == 0
    assert triples.total_energy_per_cell == singles.total_energy_per_cell


@pytest.mark.parametrize("field,reported", [
    ("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_per_worker_inventoried_bytes", "per_worker_inventoried_bytes"),
    ("maximum_node_inventoried_bytes", "required_node_memory_bytes"),
    ("maximum_work_units", "work_units_upper_bound"),
])
def test_domain_ccsd_outer_exact_caps_and_minus_one_before_progress(field, reported):
    b = _case(1, kind="he", full=True)
    mp2 = _mp2_run(b)
    controls = _controls(b)
    required = getattr(_plan(b, mp2, controls), reported)
    setattr(controls[2], field, required)
    assert _run(b, mp2, controls).converged
    setattr(controls[2], field, required-1)
    events = []
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(b, mp2, controls, events.append)
    assert not events


@pytest.mark.parametrize("field,report", [
    ("maximum_generation_dimension", "common_virtual_dimension"),
    ("maximum_control_storage_bytes_per_worker", "control_storage_reservation_bytes"),
])
def test_embedded_singles_nested_caps_are_preflighted(field, report):
    b = _case(1, kind="he", full=True)
    mp2, events = _mp2_run(b), []
    controls = _controls(b)
    required = getattr(_plan(b, mp2, controls), report)
    nested = controls[2].embedded_singles
    setattr(nested, field, required-1)
    controls[2].embedded_singles = nested
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(b, mp2, controls, events.append)
    assert not events


@pytest.mark.parametrize("field", _EMBEDDING_FIELDS)
def test_embedded_singles_audit_budgets_are_explicit_before_progress(field):
    b = _case(1, kind="he", full=True)
    mp2, events = _mp2_run(b), []
    controls = _controls(b)
    audits = controls[0].singles_embedding
    setattr(audits, field, float("nan"))
    controls[0].singles_embedding = audits
    with pytest.raises((ValueError, RuntimeError), match="explicit|finite|positive|budget|projection"):
        _run(b, mp2, controls, events.append)
    assert not events


def test_foreign_and_unconverged_domain_warmstart_cannot_forge_consumer_provenance():
    b, foreign = _case(1, kind="he", full=True), _case(1, kind="he", full=True)
    mp2 = _mp2_run(b)
    events = []
    with pytest.raises((ValueError, RuntimeError), match="source|owner|state|context|matching|frame"):
        _run(foreign, mp2, callback=events.append)
    limited = _mp2_run(b, _mp2_controls(b, maximum_iterations=1))
    assert not limited.converged
    with pytest.raises((ValueError, RuntimeError), match="converged"):
        _run(b, limited, callback=events.append)
    assert not events


@pytest.mark.parametrize("stage", ["BEGIN", "FINISHED"])
def test_scalar_callback_cancellation_publishes_no_result(stage):
    b = _case(1, kind="he", full=True)
    mp2 = _mp2_run(b)
    def cancel(event):
        if event.stage == getattr(core._PeriodicGaussianPairCCSDStage, stage):
            raise RuntimeError("cancel domain singles chain")
    with pytest.raises(RuntimeError, match="cancel domain singles chain"):
        _run(b, mp2, callback=cancel)


def test_mutable_frontend_controls_are_copied_before_the_first_callback():
    b = _case(1, kind="he", full=True)
    mp2 = _mp2_run(b)
    baseline = _run(b, mp2)
    controls, events = _controls(b), []
    def mutate(event):
        events.append(event)
        if event.stage == core._PeriodicGaussianPairCCSDStage.BEGIN:
            pno = controls[0].singles
            pno.occupation_cutoff = 1.0
            controls[0].singles = pno
            audits = controls[0].singles_embedding
            audits.maximum_exported_gram_error = float("nan")
            controls[0].singles_embedding = audits
            controls[2].maximum_work_units = 0
    result = _run(b, mp2, controls, mutate)
    assert result.converged and result.identity_sha256 == baseline.identity_sha256
    assert result.singles_options(0).occupation_cutoff == 0
    assert events[0].stage == core._PeriodicGaussianPairCCSDStage.BEGIN
    assert events[-1].stage == core._PeriodicGaussianPairCCSDStage.FINISHED
    child = result.solver
    expected = child.singles_copy(0)
    del b, mp2, result, baseline
    gc.collect()
    np.testing.assert_array_equal(child.singles_copy(0), expected)
