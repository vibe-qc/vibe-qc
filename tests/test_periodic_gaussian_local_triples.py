"""Actual finite-source pair-CCSD/TNO coupled triples, tiny independent oracles.

No caller numerical matrices enter the physical factory. Dense common
amplitudes, AO-tile integrals and linear solves below are test-only. These
fixtures do not certify infinite source convergence or production DLPNO.
"""

from __future__ import annotations

import gc
from itertools import combinations_with_replacement, permutations

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_triple_spaces import (
    _chain, _run as _spaces_run, _controls as _spaces_controls,
)
from tests.test_periodic_gaussian_pair_mp2 import _run as _mp2_run, _controls as _mp2_controls
from tests.test_periodic_gaussian_pair_ccsd import (
    _run as _ccsd_run, _controls as _ccsd_controls, _oracle_problem as _ccsd_problem,
)
from tests.test_bounded_restricted_pair_ccsd_solver import _expand
from tests.test_bounded_restricted_local_triples_solver import (
    _options as _solver_options, _action, _energy, _norms, _linear_oracle, _ordered,
)
from tests.test_bounded_restricted_local_triples_moments import _options as _moments_options
from tests.test_bounded_restricted_triples_moments import _einsum_oracle
from tests.test_bounded_restricted_triples_solver import (
    _moments, _run as _common_run, _canonical_spin_energy,
)
from tests.test_periodic_gaussian_selected_local_ccsd_t import (
    _run as _selected_run, _prepare_leaves, _he2_controls,
)
from tests.test_periodic_gaussian_frozen_core_correlation import _frozen_bundle
from tests.test_periodic_gaussian_real_local_provider import _make as _provider


def _controls(provider=None, *, iterations=80):
    options = core._PeriodicGaussianLocalTriplesOptions()
    moments = _moments_options()
    moments.coefficient_orthogonality_tolerance = 1e-10
    moments.amplitude_symmetry_tolerance = 1e-12
    # The outer native driver derives this cost from its future actual
    # provider. Direct leaf calls always supply the already certified owner.
    moments.maximum_integral_work_units_per_call = 0 if provider is None else provider.memory.scalar_work_units
    options.moments = moments
    options.solver = _solver_options(maximum_iterations=iterations,
        coefficient_orthogonality_tolerance=1e-10, maximum_projected_fock_error=1e-10,
        maximum_repeated_moment_defect_norm=1e-10, maximum_repeated_update_defect_norm=1e-10)
    options.maximum_brillouin_projection_norm = 1e-10
    live = core._PeriodicGaussianLocalTriplesLiveInventory()
    live.other_live_numerical_bytes_per_worker = 2**20  # Other bounded test owners.
    live.other_live_control_bytes_per_worker = 65536
    live.fixed_backend_margin_bytes_per_worker = 65536
    caps = core._PeriodicGaussianLocalTriplesCaps()
    moments_cap = core._BoundedRestrictedLocalTriplesMomentsCaps()
    moments_cap.maximum_occupied_count = moments_cap.maximum_common_virtual_dimension = moments_cap.maximum_rank = 4
    moments_cap.maximum_owned_numerical_bytes = 2**20
    moments_cap.maximum_per_replica_inventoried_bytes = 2**24
    moments_cap.maximum_node_inventoried_bytes = 2**26
    moments_cap.maximum_common_singles_calls = moments_cap.maximum_common_doubles_calls = 10**12
    moments_cap.maximum_transformed_integral_calls = moments_cap.maximum_common_integral_calls = 10**12
    moments_cap.maximum_work_units = 10**15
    caps.moments = moments_cap
    solver_cap = core._BoundedRestrictedLocalTriplesSolverCaps()
    solver_cap.maximum_occupied_count = solver_cap.maximum_common_virtual_dimension = solver_cap.maximum_rank = 4
    solver_cap.maximum_triple_count = 20
    solver_cap.maximum_owned_numerical_bytes = 2**20
    solver_cap.maximum_per_replica_inventoried_bytes = 2**24
    solver_cap.maximum_node_inventoried_bytes = 2**26
    solver_cap.maximum_moment_visits = 10000
    solver_cap.maximum_work_units = 10**15
    caps.solver = solver_cap
    caps.maximum_pair_count, caps.maximum_triple_count = 10, 20
    caps.maximum_owned_numerical_bytes = 2**22
    caps.maximum_control_storage_bytes_per_worker = 2**22
    caps.maximum_per_worker_inventoried_bytes = 2**24
    caps.maximum_node_inventoried_bytes = 2**26
    caps.maximum_common_integral_calls = 10**12
    caps.maximum_progress_callbacks, caps.maximum_work_units = 1024, 10**15
    return options, live, caps


def _run(b, provider, mp2, ccsd, spaces, controls=None, callback=None):
    return core._run_periodic_gaussian_local_triples(b.reference, b.basis, provider, mp2, ccsd, spaces,
        *(_controls(provider) if controls is None else controls), callback)


def _plan(b, provider, mp2, ccsd, spaces, controls=None):
    return core._plan_periodic_gaussian_local_triples(b.reference, b.basis, provider, mp2, ccsd, spaces,
        *(_controls(provider) if controls is None else controls))


def _common(b, mp2, ccsd):
    p = _ccsd_problem(b, mp2, ccsd)
    st = [ccsd.solver.singles_copy(i) for i in range(p["o"])]
    pt = [ccsd.solver.pair_copy(*label) for label in p["labels"]]
    common = _expand(p, st, pt)
    # This is the explicit Brillouin approximation also audited by the
    # physical factory; the original CCSD/HF energies are never altered.
    assert np.linalg.norm(common["fov"]) < 1e-10
    common["fov"] = np.zeros_like(common["fov"])
    return common


def _projected_common(common, c):
    """Project amplitudes before W's internal virtual contraction."""
    o, n, rank = common["o"], common["v"], c.shape[1]
    transform = np.zeros((o+n, o+rank))
    transform[:o, :o], transform[o:, o:] = np.eye(o), c
    return dict(o=o, v=rank, foo=common["foo"], fov=np.zeros((o, rank)),
        fvv=np.ascontiguousarray(c.T@common["fvv"]@c),
        eri=np.ascontiguousarray(np.einsum("pP,qQ,rR,sS,pqrs->PQRS", transform, transform,
            transform, transform, common["eri"], optimize=True)),
        factors=np.ascontiguousarray(np.einsum("pi,Ppq,qj->Pij", transform, common["factors"], transform)),
        t1=np.ascontiguousarray(common["t1"]@c),
        t2=np.ascontiguousarray(np.einsum("xa,ijxy,yb->ijab", c, common["t2"], c)))


def _oracle_problem(b, mp2, ccsd, spaces):
    common = _common(b, mp2, ccsd)
    o, n = common["o"], common["v"]
    labels = list(combinations_with_replacement(range(o), 3))
    cs = [spaces.triple_coefficients_copy(*label) for label in labels]
    eps = [spaces.triple_energies_copy(*label) for label in labels]
    w, u = [], []
    for label, c in zip(labels, cs, strict=True):
        if not c.shape[1]:
            wi, ui = np.empty((0,)*3), np.empty((0,)*3)
        else:
            wi, ui = _einsum_oracle(_projected_common(common, c), label)
        w.append(wi)
        u.append(ui)
    return dict(o=o, n=n, labels=labels, ranks=[c.shape[1] for c in cs], c=cs, eps=eps,
        foo=common["foo"], fvv=common["fvv"], w=w, u=u, snapshot=ccsd.solver.final_snapshot.iteration), common


def _full_rank_linear_oracle(p, common):
    """Diagonalize the independent complete Kronecker-sum operator.

    At most 4^6 doubles are retained; no 1280-square ragged matrix for He2K2.
    Both original occupied and virtual Fock blocks enter this exact solve.
    """
    w, _ = _moments(common)
    eo, co = np.linalg.eigh(common["foo"])
    ev, cv = np.linalg.eigh(common["fvv"])
    canonical = np.einsum("iI,jJ,kK,aA,bB,cC,ijkabc->IJKABC", co, co, co, cv, cv, cv, w, optimize=True)
    gap = (ev[None, None, None, :, None, None]+ev[None, None, None, None, :, None]
        + ev[None, None, None, None, None, :]-eo[:, None, None, None, None, None]
        - eo[None, :, None, None, None, None]-eo[None, None, :, None, None, None])
    assert gap.min() > 1e-8 and canonical.size <= 4096
    t = np.einsum("iI,jJ,kK,aA,bB,cC,IJKABC->ijkabc", co, co, co, cv, cv, cv,
        -canonical/gap, optimize=True)
    return [np.einsum("ax,by,cz,abc->xyz", c, c, c, t[label])
        for label, c in zip(p["labels"], p["c"], strict=True)]


def _check_snapshot(p, result, expected=None):
    actual = [result.solver.amplitudes_copy(*label) for label in p["labels"]]
    residuals = [w+r for w, r in zip(p["w"], _action(p, actual), strict=True)]
    maximum, norm = _norms(p, residuals)
    snapshot = result.solver.final_snapshot
    assert snapshot.maximum_absolute_residual == pytest.approx(maximum, abs=2e-13)
    assert snapshot.residual_frobenius_norm == pytest.approx(norm, abs=5e-13)
    assert snapshot.triples_energy == pytest.approx(_energy(p, actual), abs=5e-13)
    if expected is not None:
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, e, atol=3e-12, rtol=3e-9)
        assert snapshot.triples_energy == pytest.approx(_energy(p, expected), abs=5e-13)
    for label, a in zip(p["labels"], actual, strict=True):
        for permuted in set(permutations(label)):
            np.testing.assert_array_equal(result.solver.amplitudes_copy(*permuted), _ordered(a, permuted))
    return actual


@pytest.mark.parametrize("system,nk", [("he", 1), ("he", 2), ("he", 3), ("he2", 1), ("he2", 2)])
def test_full_actual_pair_ccsd_tno_coupled_triples_matches_independent_linear_and_common_solver(system, nk):
    b, provider, mp2, ccsd = _chain(system, nk)
    spaces = _spaces_run(b, provider, mp2, ccsd)
    events = []
    result = _run(b, provider, mp2, ccsd, spaces, callback=events.append)
    assert result.converged and result.periodic_energy_per_cell
    assert result.diagnostics.complete_common_finite_torus_basis
    assert result.diagnostics.all_retained_triples_full_common_rank
    assert not result.production_dlpno and not result.infinite_source_accuracy_certified
    p, common = _oracle_problem(b, mp2, ccsd, spaces)
    _check_snapshot(p, result, _full_rank_linear_oracle(p, common))
    # An independent full-space virtual rotation connects the original
    # non-diagonal Fvv to the existing common-space triples solver.
    ev, cv = np.linalg.eigh(common["fvv"])
    canonical = _projected_common(common, cv)
    canonical["fvv"] = np.diag(ev)
    if p["o"] <= 3:  # The generic supplied-array diagnostic has an occupied<=3 cap.
        old = _common_run(canonical)
        assert old.final_snapshot.converged
        assert result.solver.final_snapshot.triples_energy == pytest.approx(old.final_snapshot.triples_energy, abs=6e-13)
    assert result.solver.final_snapshot.triples_energy == pytest.approx(_canonical_spin_energy(canonical), abs=6e-13)
    assert result.triples_energy_per_cell == pytest.approx(result.solver.final_snapshot.triples_energy/nk, abs=2e-15)
    assert result.correlation_energy_per_cell == pytest.approx(ccsd.correlation_energy_per_cell+result.triples_energy_per_cell, abs=2e-15)
    assert result.total_energy_per_cell == pytest.approx(b.hf.state.reference_energy_per_cell+result.correlation_energy_per_cell, abs=2e-15)
    assert result.ccsd_identity_sha256 == ccsd.identity_sha256
    assert result.triple_spaces_identity_sha256 == spaces.identity_sha256
    for value in (result.identity_sha256, result.consumed_moments_receipt_sha256):
        assert len(value) == 64 and int(value, 16) >= 0
    stage = core._PeriodicGaussianLocalTriplesStage
    assert events[0].stage == stage.BEGIN and events[-1].stage == stage.FINISHED
    solver_events = [e for e in events if e.stage == stage.SOLVER]
    assert [e.solver.iteration for e in solver_events] == list(range(1, result.solver.final_snapshot.iteration+1))
    assert [e.callback_count for e in events] == list(range(1, len(events)+1))
    assert len(events) == result.diagnostics.completed_progress_callbacks <= result.memory.progress_callback_upper_bound
    assert events[-1].common_integral_calls == result.diagnostics.completed_common_integral_calls
    assert result.diagnostics.completed_common_integral_calls <= result.memory.common_integral_calls_upper_bound
    if system == "he2":
        assert abs(common["foo"][0, 1]) > 1e-3
        assert result.solver.final_snapshot.neighbour_visits > 0
        t0 = [-w/(eps[:, None, None]+eps[None, :, None]+eps[None, None, :]
            -sum(common["foo"][i, i] for i in label))
            for label, eps, w in zip(p["labels"], p["eps"], p["w"], strict=True)]
        assert abs(_energy(p, t0)-result.solver.final_snapshot.triples_energy) > 1e-10


@pytest.mark.parametrize("system,nk", [("he", 1), ("he2", 1), ("he2", 2)])
def test_full_rank_matches_existing_actual_selected_ccsd_t_connector(system, nk):
    b, provider, mp2, ccsd = _chain(system, nk)
    spaces = _spaces_run(b, provider, mp2, ccsd)
    result = _run(b, provider, mp2, ccsd, spaces)
    old = _selected_run(b, controls=_he2_controls(b) if system == "he2" else None)
    assert old.converged and old.periodic_energy_per_cell
    assert result.total_energy_per_cell == pytest.approx(old.total_energy_per_cell, abs=8e-12)
    assert result.triples_energy_per_cell == pytest.approx(old.correlation.triples.final_snapshot.triples_energy/nk, abs=8e-13)


@pytest.mark.parametrize("nk,tno_fraction", [(1, .2), (2, .001)])
def test_truncated_pairs_and_tnos_use_preprojected_moments_not_external_projection(nk, tno_fraction):
    b, provider, full, _ = _chain("he2", nk)
    o = 2*nk
    occupations = np.concatenate([full.pair(i, j).original_pno_occupations_copy()
        for i, j in combinations_with_replacement(range(o), 2)])
    mp2 = _mp2_run(b, provider, _mp2_controls(cutoff=float(np.median(occupations))))
    ccsd = _ccsd_run(b, provider, mp2)
    assert ccsd.converged and not ccsd.diagnostics.all_pairs_full_rank
    initial = _spaces_run(b, provider, mp2, ccsd)
    occ = np.concatenate([initial.triple_occupations_copy(*label)
        for label in combinations_with_replacement(range(o), 3)])
    # Keeping only all-equal rank-one triples made the previous Gamma
    # fixture lossless for internal-d projection. These explicit cuts retain
    # mixed occupied tuples; the two-cell case also has differing ranks1–3.
    spaces = _spaces_run(b, provider, mp2, ccsd, _spaces_controls(cutoff=tno_fraction*float(occ.max())))
    assert spaces.diagnostics.maximum_retained_rank > 0
    assert not spaces.diagnostics.all_retained_full_common_rank
    result = _run(b, provider, mp2, ccsd, spaces)
    assert result.converged and result.periodic_energy_per_cell
    p, common = _oracle_problem(b, mp2, ccsd, spaces)
    _check_snapshot(p, result, _linear_oracle(p))
    common_w, _ = _moments(common)
    external = [np.einsum("ax,by,cz,abc->xyz", c, c, c, common_w[label])
        for label, c in zip(p["labels"], p["c"], strict=True)]
    # Physical witness: truncating W's internal d differs from projecting
    # only the three external virtual axes of an untruncated moment.
    assert max(np.max(abs(a-b), initial=0) for a, b in zip(external, p["w"], strict=True)) > 1e-10
    assert result.memory.rank_padding_reservation_bytes == max(result.memory.reader_borrowed_rank_padding_bytes,
        result.memory.solver_borrowed_rank_padding_bytes)


@pytest.mark.parametrize("empty_pairs", [False, True])
def test_zero_retained_rank_skips_all_moments_and_common_integral_calls(empty_pairs):
    b, provider, mp2, ccsd = _chain("he2", 1)
    if empty_pairs:
        occupation = max(float(mp2.pair(i, j).original_pno_occupations_copy().max())
            for i, j in combinations_with_replacement(range(2), 2))
        mp2 = _mp2_run(b, provider, _mp2_controls(cutoff=2*occupation))
        ccsd = _ccsd_run(b, provider, mp2)
        spaces = _spaces_run(b, provider, mp2, ccsd)
    else:
        full = _spaces_run(b, provider, mp2, ccsd)
        occupation = max(float(full.triple_occupations_copy(*label).max())
            for label in combinations_with_replacement(range(2), 3))
        spaces = _spaces_run(b, provider, mp2, ccsd, _spaces_controls(cutoff=2*occupation))
    assert spaces.diagnostics.maximum_retained_rank == 0
    result = _run(b, provider, mp2, ccsd, spaces)
    assert result.converged and result.periodic_energy_per_cell
    assert not result.memory.moments_required
    assert result.memory.maximum_moment_transient_numerical_bytes == 0
    assert result.memory.moments_upper.peak_owned_numerical_bytes == 0
    assert result.memory.common_integral_calls_upper_bound == result.diagnostics.completed_common_integral_calls == 0
    assert result.solver.final_snapshot.moment_visits == 0
    assert result.solver.final_snapshot.triples_energy == result.triples_energy_per_cell == 0
    assert result.total_energy_per_cell == ccsd.total_energy_per_cell
    assert result.memory.reader_table_bytes > 0  # Rank zero is not an owner-check bypass.
    assert result.memory.maximum_retained_rank == result.memory.total_retained_rank == result.memory.total_amplitude_elements == 0
    for label in combinations_with_replacement(range(2), 3):
        assert result.solver.amplitudes_copy(*label).shape == (0,)*3


@pytest.mark.parametrize("nk", [1, 2])
def test_actual_frozen_core_reference_constant_and_active_only_coupled_triples(nk):
    b = _frozen_bundle((nk, 1, 1))
    _prepare_leaves(b, localization_options=_he2_controls(b)[0].localization)
    provider = _provider(b)
    mp2 = _mp2_run(b, provider)
    ccsd = _ccsd_run(b, provider, mp2)
    spaces = _spaces_run(b, provider, mp2, ccsd)
    result = _run(b, provider, mp2, ccsd, spaces)
    assert result.converged and result.periodic_energy_per_cell
    assert (result.memory.occupied_count, result.memory.common_virtual_dimension) == (nk, 2*nk)
    p, common = _oracle_problem(b, mp2, ccsd, spaces)
    _check_snapshot(p, result, _full_rank_linear_oracle(p, common))
    assert b.hf.state.reference_energy_per_cell == b.unfrozen_hf.state.reference_energy_per_cell
    assert result.total_energy_per_cell == pytest.approx(b.unfrozen_hf.state.reference_energy_per_cell
        +(ccsd.solver.final_snapshot.correlation_energy+result.solver.final_snapshot.triples_energy)/nk, abs=2e-15)
    if nk == 1:
        assert abs(result.triples_energy_per_cell) < 1e-15  # Two active electrons; raw iii energy weight zero.


def test_last_evaluated_snapshot_on_exhaustion_has_no_publishable_cell_energy():
    b, provider, mp2, ccsd = _chain("he2", 1)
    spaces = _spaces_run(b, provider, mp2, ccsd)
    result = _run(b, provider, mp2, ccsd, spaces, _controls(provider, iterations=1))
    assert not result.converged and not result.periodic_energy_per_cell
    assert result.solver.final_snapshot.iteration == 1
    p, _ = _oracle_problem(b, mp2, ccsd, spaces)
    actual = _check_snapshot(p, result)
    assert all(np.count_nonzero(t) == 0 for t in actual)
    assert result.solver.final_snapshot.maximum_absolute_residual > 1e-12
    for name in ("triples_energy_per_cell", "correlation_energy_per_cell", "total_energy_per_cell"):
        with pytest.raises((ValueError, RuntimeError), match="converged|complete|cell|energy"):
            getattr(result, name)


def test_complete_owner_inventory_rank_bounds_and_external_live_delta():
    b, provider, mp2, ccsd = _chain("he2", 1)
    spaces = _spaces_run(b, provider, mp2, ccsd)
    controls = _controls(provider)
    p = _plan(b, provider, mp2, ccsd, spaces, controls)
    assert p.maximum_retained_rank == spaces.diagnostics.maximum_retained_rank
    assert p.total_retained_rank == spaces.diagnostics.total_retained_rank
    assert p.total_amplitude_elements == sum(spaces.triple_diagnostics(*label)["retained_rank"]**3
        for label in combinations_with_replacement(range(p.occupied_count), 3))
    assert p.projected_zero_fov_bytes == 8*p.occupied_count*p.common_virtual_dimension
    assert p.maximum_moment_transient_numerical_bytes == p.moments_upper.peak_owned_numerical_bytes
    assert p.maximum_moment_transient_numerical_bytes == 8*p.occupied_count*p.maximum_retained_rank+16*p.maximum_retained_rank**3
    assert p.reader_borrowed_rank_padding_bytes == p.solver_borrowed_rank_padding_bytes == p.rank_padding_reservation_bytes == 0
    assert p.required_node_memory_bytes == p.reference_base_node_bytes+p.replicas_per_node*p.per_worker_inventoried_bytes
    assert p.complete_borrowed_owner_numerical_bytes >= (mp2.diagnostics.retained_pair_bytes
        +mp2.solver.memory.output_numerical_bytes+ccsd.diagnostics.retained_singles_bytes
        +ccsd.solver.memory.output_numerical_bytes+spaces.diagnostics.retained_numerical_bytes)
    controls[1].other_live_numerical_bytes_per_worker += 777
    q = _plan(b, provider, mp2, ccsd, spaces, controls)
    assert q.per_worker_inventoried_bytes-p.per_worker_inventoried_bytes == 777
    assert q.required_node_memory_bytes-p.required_node_memory_bytes == 777*p.replicas_per_node


_OUTER_CAPS = dict(maximum_pair_count="pair_count", maximum_triple_count="triple_count",
    maximum_owned_numerical_bytes="peak_owned_numerical_bytes",
    maximum_control_storage_bytes_per_worker="control_storage_reservation_bytes",
    maximum_per_worker_inventoried_bytes="per_worker_inventoried_bytes",
    maximum_node_inventoried_bytes="required_node_memory_bytes",
    maximum_common_integral_calls="common_integral_calls_upper_bound",
    maximum_progress_callbacks="progress_callback_upper_bound", maximum_work_units="work_units_upper_bound")


@pytest.mark.parametrize("field", list(_OUTER_CAPS))
def test_outer_exact_caps_and_minus_one_fail_before_any_progress(field):
    b, provider, mp2, ccsd = _chain()
    spaces = _spaces_run(b, provider, mp2, ccsd)
    controls = _controls(provider)
    p = _plan(b, provider, mp2, ccsd, spaces, controls)
    for cap, report in _OUTER_CAPS.items():
        setattr(controls[2], cap, getattr(p, report))
    assert _run(b, provider, mp2, ccsd, spaces, controls).converged
    setattr(controls[2], field, getattr(controls[2], field)-1)
    events = []
    with pytest.raises((ValueError, RuntimeError), match="cap|budget|admission"):
        _run(b, provider, mp2, ccsd, spaces, controls, events.append)
    assert events == []


@pytest.mark.parametrize("leaf,field", [("moments", "maximum_owned_numerical_bytes"),
    ("moments", "maximum_common_integral_calls"), ("moments", "maximum_work_units"),
    ("solver", "maximum_owned_numerical_bytes"), ("solver", "maximum_moment_visits"),
    ("solver", "maximum_work_units")])
def test_nested_caps_fail_before_progress(leaf, field):
    b, provider, mp2, ccsd = _chain()
    spaces = _spaces_run(b, provider, mp2, ccsd)
    controls = _controls(provider)
    nested = getattr(controls[2], leaf)
    setattr(nested, field, 0)
    setattr(controls[2], leaf, nested)
    events = []
    with pytest.raises((ValueError, RuntimeError), match="cap|budget|positive"):
        _run(b, provider, mp2, ccsd, spaces, controls, events.append)
    assert events == []


def test_foreign_equal_content_owners_and_unconverged_inputs_are_not_certificates():
    b, provider, mp2, ccsd = _chain()
    spaces = _spaces_run(b, provider, mp2, ccsd)
    other, foreign_provider, foreign_mp2, foreign_ccsd = _chain()
    foreign_spaces = _spaces_run(other, foreign_provider, foreign_mp2, foreign_ccsd)
    cases = [(b, foreign_provider, mp2, ccsd, spaces), (b, provider, foreign_mp2, ccsd, spaces),
        (b, provider, mp2, foreign_ccsd, spaces), (b, provider, mp2, ccsd, foreign_spaces)]
    unconverged = _ccsd_run(b, provider, mp2, _ccsd_controls(provider, iterations=1))
    assert not unconverged.converged
    cases.append((b, provider, mp2, unconverged, spaces))
    unfinished_mp2 = _mp2_run(b, provider, _mp2_controls(iterations=1))
    assert not unfinished_mp2.converged
    cases.append((b, provider, unfinished_mp2, ccsd, spaces))
    for args in cases:
        events = []
        with pytest.raises((ValueError, RuntimeError), match="owner|state|source|identity|converged|match|warmstart"):
            _run(*args, callback=events.append)
        assert events == []


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
def test_brillouin_projection_budget_requires_explicit_finite_nonnegative_choice(bad):
    b, provider, mp2, ccsd = _chain()
    spaces = _spaces_run(b, provider, mp2, ccsd)
    controls = _controls(provider)
    controls[0].maximum_brillouin_projection_norm = bad
    with pytest.raises(ValueError, match="Brillouin|brillouin|projection|finite"):
        _run(b, provider, mp2, ccsd, spaces, controls)


def test_actual_original_fov_norm_and_provider_work_are_audited_not_declared():
    b, provider, mp2, ccsd = _chain("he2", 1)
    spaces = _spaces_run(b, provider, mp2, ccsd)
    result = _run(b, provider, mp2, ccsd, spaces)
    o = result.memory.occupied_count
    norm = np.linalg.norm(b.basis.fock_copy()[:o, o:])
    assert result.diagnostics.brillouin_projection_frobenius_norm == pytest.approx(norm, abs=1e-25)
    assert norm > 0  # Explicit zero budget must reject the actual nonzero residual.
    controls = _controls(provider)
    controls[0].maximum_brillouin_projection_norm = norm/2
    with pytest.raises(ValueError, match="Brillouin|brillouin|projection"):
        _run(b, provider, mp2, ccsd, spaces, controls)
    controls = _controls(provider)
    moments = controls[0].moments
    moments.maximum_integral_work_units_per_call += 1
    controls[0].moments = moments
    with pytest.raises(ValueError, match="work|provider|exact|source"):
        _run(b, provider, mp2, ccsd, spaces, controls)


def test_partial_common_basis_never_exposes_per_cell_energy():
    b, provider, mp2, ccsd = _chain(nk=2, full=False)
    spaces = _spaces_run(b, provider, mp2, ccsd)
    result = _run(b, provider, mp2, ccsd, spaces)
    assert result.converged and not result.periodic_energy_per_cell
    assert not result.diagnostics.complete_common_finite_torus_basis
    for name in ("triples_energy_per_cell", "correlation_energy_per_cell", "total_energy_per_cell"):
        with pytest.raises((ValueError, RuntimeError), match="complete|cell|energy"):
            getattr(result, name)


@pytest.mark.parametrize("cancel_stage", ["BEGIN", "SOLVER", "FINISHED"])
def test_progress_cancellation_publishes_no_result(cancel_stage):
    b, provider, mp2, ccsd = _chain()
    spaces = _spaces_run(b, provider, mp2, ccsd)
    target = getattr(core._PeriodicGaussianLocalTriplesStage, cancel_stage)
    def cancel(event):
        if event.stage == target:
            raise RuntimeError("physical coupled triples cancellation")
    with pytest.raises(RuntimeError, match="triples cancellation"):
        _run(b, provider, mp2, ccsd, spaces, callback=cancel)
    assert _run(b, provider, mp2, ccsd, spaces).converged


def test_controls_are_copied_and_detached_results_outlive_all_input_owners():
    b, provider, mp2, ccsd = _chain("he2", 1)
    spaces = _spaces_run(b, provider, mp2, ccsd)
    expected = _run(b, provider, mp2, ccsd, spaces)
    controls = _controls(provider)
    def mutate(_):
        controls[0].maximum_brillouin_projection_norm = np.nan
        solver = controls[0].solver
        solver.maximum_iterations = 0
        controls[0].solver = solver
        moments = controls[0].moments
        moments.maximum_integral_work_units_per_call = 0
        controls[0].moments = moments
        controls[1].fixed_backend_margin_bytes_per_worker = 0
        controls[2].maximum_work_units = 0
    result = _run(b, provider, mp2, ccsd, spaces, controls, mutate)
    assert result.identity_sha256 == expected.identity_sha256
    assert result.consumed_moments_receipt_sha256 == expected.consumed_moments_receipt_sha256
    solver = result.solver
    original = solver.amplitudes_copy(0, 0, 1)
    result.solver.amplitudes_copy(0, 0, 1)[:] = 42
    np.testing.assert_array_equal(solver.amplitudes_copy(0, 0, 1), original)
    total = result.total_energy_per_cell
    del b, provider, mp2, ccsd, spaces, controls, expected
    gc.collect()
    assert result.total_energy_per_cell == total
    del result
    gc.collect()
    np.testing.assert_array_equal(solver.amplitudes_copy(0, 0, 1), original)
    with pytest.raises((ValueError, IndexError)):
        solver.amplitudes_copy(0, 0, 2)
