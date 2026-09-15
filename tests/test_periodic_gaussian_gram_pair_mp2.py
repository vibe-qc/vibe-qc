"""Direct Gaussian Gram PNOs consumed by the bounded native MP2 equations.

The original numerical-leaf coverage is retained alongside the native
physical driver. Only the former accepts detached arrays. The physical
driver receives actual HF and domain owners, never a provider or G/F/C/T.
Independent Python linear solves are oracles, never runtime paths.
"""

from __future__ import annotations

import gc
from itertools import combinations_with_replacement

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_bounded_restricted_pair_mp2_solver import (
    _blocks, _energy, _inventory, _operator, _projected_linear_oracle,
    _run as _solve,
)
from tests.test_periodic_gaussian_gram_pair_pnos import (
    _case, _controls, _geometry, _make, _oracle_inputs,
)
from tests.test_periodic_gaussian_domain_pair_mp2 import (
    _controls as _domain_controls, _run as _domain_run,
)
from tests.test_periodic_gaussian_pair_mp2 import _controls as _mp2_controls
from tests.test_periodic_gaussian_real_local_provider import _make as _provider


def _space(b, pno):
    options, _, caps = _mp2_controls()
    live = core._PeriodicGaussianPairSpaceLiveInventory()
    live.other_live_bytes_per_worker = 2**23
    live.fixed_backend_margin_bytes_per_worker = 65536
    return core._make_periodic_gaussian_gram_pair_space_diagnostic(
        b.reference, b.basis, pno, options.projection, live, caps.projection)


def _problem(b, *, cutoff=0.0):
    """Finish every new-source pair before constructing any old provider."""
    assert not hasattr(b, "provider")
    occupied = b.basis.memory.occupied_count
    labels = list(combinations_with_replacement(range(occupied), 2))
    geometries, spaces = [], []
    for i, j in labels:
        geometry = _geometry(b, i, j)
        seed = _make(b, geometry, _controls(b, cutoff=cutoff))
        spaces.append(_space(b, seed))
        geometries.append(geometry)
    assert not hasattr(b, "provider")
    problem = dict(
        o=occupied, n=b.basis.memory.virtual_count, labels=labels,
        ranks=[s.memory.retained_dimension for s in spaces],
        f=np.ascontiguousarray(b.basis.fock_copy()[:occupied, :occupied]),
        g=[s.exchange_integrals_copy() for s in spaces],
        eps=[s.energies_copy() for s in spaces],
        c=[s.coefficients_copy() for s in spaces],
    )
    return problem, geometries, spaces


def _check_native_equations(problem, result):
    expected = _projected_linear_oracle(problem)
    actual = _blocks(result, problem)
    assert result.final_snapshot.converged
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=3e-11, rtol=3e-9)
    assert result.final_snapshot.correlation_energy == pytest.approx(
        _energy(problem, expected), abs=3e-11)
    residuals = [g + r for g, r in zip(
        problem["g"], _operator(problem, actual), strict=True)]
    assert max((np.max(np.abs(r), initial=0.0) for r in residuals), default=0.0) < 3e-11
    return actual


@pytest.mark.parametrize("kind,nk,frozen,full", [
    ("he", 1, False, True), ("he", 2, False, True),
    ("he", 3, False, True), ("he2", 1, False, True),
    ("he2", 2, False, False), ("he2", 2, True, False),
])
def test_actual_gram_pnos_feed_complete_coupled_mp2_without_prior_rows(kind, nk, frozen, full):
    b = _case(nk, kind=kind, frozen=frozen, full=full)
    problem, geometries, spaces = _problem(b)
    events = []
    options = _mp2_controls()[0].solver
    result = _solve(problem, options=options,
        inventory=_inventory(other_live_bytes_per_replica=2**23), callback=events.append)
    _check_native_equations(problem, result)
    assert not hasattr(b, "provider")
    assert len(events) == result.final_snapshot.iteration >= 2
    assert all(s.memory.integral_calls == 0 for s in spaces)
    assert all(s.memory.borrowed_provider_row_bytes == 0 for s in spaces)
    assert all(s.direct_gram_generation and not s.embedded_generation for s in spaces)
    assert all(not s.production_dlpno for s in spaces)
    assert len(geometries) == len(problem["labels"])

    # The old implementation is an additional independent projection path,
    # instantiated only AFTER every new Gram/PNO/MP2 numerical stage finishes.
    b.provider = _provider(b)
    legacy = _domain_run(b, _domain_controls(b))
    assert legacy.converged
    assert result.final_snapshot.correlation_energy / nk == pytest.approx(
        legacy.correlation_energy_per_cell, abs=3e-11)
    # A native MP2 energy is a finite-torus sum. Normalize exactly once here;
    # do not attach a production-energy certificate to the array diagnostic.
    assert b.hf.state.reference_energy_per_cell + result.final_snapshot.correlation_energy / nk == pytest.approx(
        legacy.total_energy_per_cell, abs=3e-11)


def test_positive_pno_cut_preserves_coupled_equations_in_ragged_actual_frames():
    b = _case(2, kind="he2", full=False)
    probe = _make(b, _geometry(b, 0, 0))
    occupations = probe.original_pno_occupations_copy()
    assert len(occupations) == 2 and occupations[0] > occupations[1] >= 0.0
    cutoff = float((occupations[0] + occupations[1]) / 2)
    problem, geometries, spaces = _problem(b, cutoff=cutoff)
    assert 0 < spaces[0].memory.retained_dimension < spaces[0].memory.generation_dimension
    result = _solve(problem, options=_mp2_controls()[0].solver,
        inventory=_inventory(other_live_bytes_per_replica=2**23))
    _check_native_equations(problem, result)
    assert not hasattr(b, "provider") and geometries


def test_empty_gram_pno_frames_give_zero_coupled_mp2_without_erasing_domains():
    b = _case(2, kind="he2", full=False)
    problem, geometries, spaces = _problem(b, cutoff=1.0)
    assert problem["ranks"] == [0] * len(problem["labels"])
    result = _solve(problem, options=_mp2_controls()[0].solver,
        inventory=_inventory(other_live_bytes_per_replica=2**23))
    _check_native_equations(problem, result)
    assert result.final_snapshot.correlation_energy == 0.0
    assert result.final_snapshot.iteration == 2
    assert all(s.memory.generation_dimension > 0 for s in spaces)
    assert len(geometries) == len(spaces)


def _native_controls(b, *, cutoff=0.0, maximum_iterations=None):
    domain_options, live, domain_caps = _domain_controls(b, cutoff=cutoff,
        maximum_iterations=maximum_iterations)
    seed_config, seed_options, _, seed_caps = _controls(b, cutoff=cutoff)
    config = core._PeriodicGaussianGramDomainPairMP2Config()
    config.pnos = seed_config
    options = core._PeriodicGaussianGramDomainPairMP2Options()
    options.domain, options.pnos = domain_options.domain, seed_options
    options.projection, options.solver = domain_options.projection, domain_options.solver
    options.maximum_occupied_virtual_fock_norm = domain_options.maximum_occupied_virtual_fock_norm
    caps = core._PeriodicGaussianGramDomainPairMP2Caps()
    caps.domain, caps.projection, caps.solver = domain_caps.domain, domain_caps.projection, domain_caps.solver
    seed_caps.maximum_owned_numerical_bytes = 2**21
    seed_caps.maximum_work_units = 5*10**16
    seed_caps.maximum_control_storage_bytes = 2**24
    seed_caps.maximum_worker_bytes = 2**26
    # Complete enclosing controls flow into every child, unlike standalone
    # leaf fixtures. These are byte ceilings only, not numerical tolerances.
    caps.domain.maximum_control_storage_bytes = 2**24
    caps.domain.pair_union.maximum_control_storage_bytes = 2**24
    caps.domain.embedding.maximum_control_storage_bytes_per_worker = 2**24
    caps.domain.maximum_worker_bytes = 2**26
    caps.domain.maximum_node_bytes = 2**27
    caps.domain.pair_union.maximum_worker_bytes = 2**26
    caps.domain.embedding.maximum_per_worker_inventoried_bytes = 2**26
    caps.projection.maximum_per_worker_inventoried_bytes = 2**26
    caps.projection.maximum_node_inventoried_bytes = 2**27
    caps.solver.maximum_per_replica_inventoried_bytes = 2**26
    caps.solver.maximum_node_inventoried_bytes = 2**27
    for resource in (seed_caps.gram.resources, seed_caps.gram.metric,
                     seed_caps.gram.panel.resources, seed_caps.gram.panel.tile.resources):
        resource.maximum_per_replica_inventoried_bytes = 2**26
        resource.maximum_node_inventoried_bytes = 2**27
    caps.pnos = seed_caps
    caps.maximum_pno_leaf_control_storage_bytes = 2**23
    caps.maximum_owned_numerical_bytes = 2**24
    caps.maximum_control_storage_bytes = 2**24
    caps.maximum_per_worker_inventoried_bytes = 2**26
    caps.maximum_node_inventoried_bytes = 2**27
    pair_count = b.basis.memory.occupied_count*(b.basis.memory.occupied_count+1)//2
    caps.maximum_pair_count = caps.maximum_gram_builds = 16
    caps.maximum_factor_panels = pair_count*seed_caps.gram.maximum_factor_panels
    caps.maximum_tile_calls = pair_count*seed_caps.gram.maximum_tile_calls
    caps.maximum_progress_callbacks, caps.maximum_work_units = 1024, 10**18
    return config, options, live, caps


def _native_arguments(b, controls=None, **changes):
    config, options, live, caps = _native_controls(b) if controls is None else controls
    args = dict(hf=b.hf, reference=b.reference, wannier=b.wannier, basis=b.basis,
        builder=b.builder, config=config, options=options, live=live, caps=caps)
    args.update(changes)
    return args


def _native_plan(b, controls=None, **changes):
    return core._plan_periodic_gaussian_gram_domain_pair_mp2_diagnostic(**_native_arguments(b, controls, **changes))


def _native_run(b, controls=None, progress=None, **changes):
    args = _native_arguments(b, controls)
    args.update(ao_basis=b.ao, auxiliary_basis=b.auxiliary, gauges=b.gauge, progress=progress)
    args.update(changes)
    return core._run_periodic_gaussian_gram_domain_pair_mp2_diagnostic(**args)


def _native_problem(b, result, *, independent_eri=None):
    o, n = result.memory.occupied_count, result.memory.common_virtual_dimension
    labels = list(combinations_with_replacement(range(o), 2))
    spaces = [result.pair(i, j) for i, j in labels]
    coefficients = [s.coefficients_copy() for s in spaces]
    f = b.basis.fock_copy()
    g = [s.exchange_integrals_copy() for s in spaces]
    eps = [s.energies_copy() for s in spaces]
    if independent_eri is not None:
        for slot, ((i, j), c) in enumerate(zip(labels, coefficients, strict=True)):
            expected = c.T @ independent_eri[i, o:, j, o:] @ c
            np.testing.assert_allclose(g[slot], expected, atol=7e-12, rtol=3e-9)
            g[slot] = expected
            projected = c.T @ f[o:, o:] @ c
            np.testing.assert_allclose(projected, np.diag(eps[slot]), atol=8e-11)
            eps[slot] = np.diag(projected).copy()
    return dict(o=o, n=n, labels=labels, ranks=[s.memory.retained_dimension for s in spaces],
        f=np.ascontiguousarray(f[:o, :o]), g=g, eps=eps, c=coefficients)


@pytest.fixture(scope="module", params=[("he", 1, False, True), ("he", 2, False, True),
    ("he", 3, False, True), ("he2", 1, False, True), ("he2", 2, False, False), ("he2", 2, True, False)])
def native_actual(request):
    kind, nk, frozen, full = request.param
    b = _case(nk, kind=kind, frozen=frozen, full=full)
    events = []
    result = _native_run(b, progress=events.append)
    assert not hasattr(b, "provider")
    # Both independent AO contraction and the legacy row provider are
    # constructed only after the authentic new native solver finishes.
    oracle = _oracle_inputs(b)
    b.provider = oracle.provider
    legacy = _domain_run(b, _domain_controls(b))
    return b, result, events, oracle, legacy, nk, frozen


def test_native_direct_gram_driver_matches_independent_equations_and_legacy_afterward(native_actual):
    b, result, events, oracle, legacy, nk, frozen = native_actual
    assert result.converged and result.domain_generated and result.direct_gram_source
    assert result.source_kind == core._PeriodicGaussianPairMP2SourceKind.DIRECT_PAO_GRAM
    assert result.matched_finite_gaussian_hf_recipe and result.periodic_energy_per_cell
    problem = _native_problem(b, result, independent_eri=oracle.eri)
    _check_native_equations(problem, result.solver)
    expected_energy = _energy(problem, _projected_linear_oracle(problem))
    assert result.correlation_energy_per_cell == pytest.approx(expected_energy/nk, abs=3e-11)
    assert result.total_energy_per_cell == pytest.approx(b.hf.state.reference_energy_per_cell+expected_energy/nk, abs=3e-11)
    assert legacy.converged and result.correlation_energy_per_cell == pytest.approx(legacy.correlation_energy_per_cell, abs=3e-11)
    assert not result.production_dlpno and not result.infinite_source_accuracy_certified
    assert result.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256
    assert result.domain_builder_identity_sha256 == b.builder.identity_sha256
    for field in ("identity_sha256", "pair_spaces_identity_sha256", "gram_sources_identity_sha256"):
        assert len(getattr(result, field)) == 64 and int(getattr(result, field), 16) >= 0
    with pytest.raises((ValueError, RuntimeError), match="source|direct|Gram|provider"):
        _ = result.provider_identity_sha256
    if nk > 1:
        assert abs(expected_energy) > 1e-8
        assert abs(result.correlation_energy_per_cell-expected_energy/nk**2) > 1e-8
    if frozen:
        assert problem["o"] == nk and problem["n"] == 2*nk
    stage = core._PeriodicGaussianPairMP2Stage
    assert events[0].stage == stage.BEGIN and events[-1].stage == stage.FINISHED
    assert sum(e.stage == stage.PAIR_DOMAIN_READY for e in events) == result.memory.pair_count
    assert sum(e.stage == stage.PAIR_COMPLETE for e in events) == result.memory.pair_count
    assert len(events) == result.diagnostics.completed_progress_callbacks <= result.memory.progress_callback_upper_bound
    assert [e.callback_count for e in events] == list(range(1, len(events)+1))


def test_native_direct_result_retains_original_generation_geometry_for_future_singles(native_actual):
    b, result, _, _, _, _, _ = native_actual
    p, d = result.memory, result.diagnostics
    spaces = [result.pair(i, j) for i, j in combinations_with_replacement(range(p.occupied_count), 2)]
    assert all(s.direct_gram_generation and not s.embedded_generation for s in spaces)
    assert p.borrowed_provider_row_bytes == 0
    assert p.integral_calls_upper_bound == d.completed_integral_calls == 0
    assert d.completed_gram_builds == p.pair_count == p.gram_builds_upper_bound
    assert d.completed_factor_panels <= p.factor_panels_upper_bound
    assert d.completed_tile_calls <= p.tile_calls_upper_bound
    assert d.retained_pair_bytes == sum(s.memory.retained_output_bytes for s in spaces)
    assert d.generation_dimension_sum == sum(s.memory.generation_dimension for s in spaces)
    assert d.retained_generation_coefficient_bytes == sum(8*s.memory.generation_dimension*s.memory.retained_dimension for s in spaces)
    geometry = [result.pair_generation_geometry_metadata(i, j)
        for i, j in combinations_with_replacement(range(p.occupied_count), 2)]
    assert d.retained_pair_geometry_bytes == sum(g["retained_numerical_bytes"] for g in geometry)
    assert d.retained_pair_geometry_control_bytes == sum(g["retained_control_storage_bytes"] for g in geometry)
    diagonal = [result.diagonal_generation_embedding(i) for i in range(p.occupied_count)]
    assert d.diagonal_generation_dimension_sum == sum(e.memory.pair_dimension for e in diagonal)
    assert d.retained_generation_embedding_bytes == sum(e.memory.output_numerical_bytes for e in diagonal)
    assert d.retained_generation_embedding_bytes <= d.retained_pair_geometry_bytes <= p.retained_pair_geometry_upper_bytes
    assert p.required_node_memory_bytes == p.reference_base_node_bytes+p.replicas_per_node*p.per_worker_inventoried_bytes
    assert result.memory.borrowed_domain_builder_bytes == b.builder.retained_numerical_bytes


def test_native_direct_result_is_rejected_by_legacy_ccsd_before_iterations(native_actual):
    from tests.test_periodic_gaussian_pair_ccsd import _controls as ccsd_controls, _run as ccsd_run
    b, result, _, oracle, _, _, _ = native_actual
    events = []
    with pytest.raises((ValueError, RuntimeError), match="source|provider|Gram|direct"):
        ccsd_run(b, oracle.provider, result, ccsd_controls(oracle.provider, iterations=1), events.append)
    assert not events


@pytest.fixture(scope="module")
def native_small():
    return _case(2)


def test_native_positive_and_empty_ranks_preserve_geometry_and_coupled_equations():
    b = _case(2, kind="he2", full=False)
    full = _native_run(b)
    occ = full.pair(0, 0).original_pno_occupations_copy()
    assert len(occ) == 2 and occ[0] > occ[1] >= 0
    partial = _native_run(b, _native_controls(b, cutoff=float((occ[0]+occ[1])/2)))
    empty = _native_run(b, _native_controls(b, cutoff=1.0))
    assert partial.pair(0, 0).memory.retained_dimension == 1
    for result in (partial, empty):
        _check_native_equations(_native_problem(b, result), result.solver)
        assert result.diagnostics.generation_dimension_sum == full.diagnostics.generation_dimension_sum
        for i in range(result.memory.occupied_count):
            assert result.diagonal_generation_embedding(i).identity_sha256 == full.diagonal_generation_embedding(i).identity_sha256
    assert empty.diagnostics.zero_rank_pairs == empty.memory.pair_count
    assert empty.correlation_energy_per_cell == 0 and empty.total_energy_per_cell == b.hf.state.reference_energy_per_cell
    assert not hasattr(b, "provider")


def test_native_partial_common_occupied_selection_cannot_publish_per_cell_energy():
    from types import SimpleNamespace
    from tests.test_periodic_correlation_real_local_basis import _make as make_basis
    from tests.test_periodic_gaussian_pair_domain_builder import _build as make_builder
    from tests.test_periodic_correlation_real_pao_embedding import _domain, _real_space
    b = _case(2)
    b.rows = np.ascontiguousarray(b.rows[:1])
    b.basis = make_basis(b, b.rows, b.selected, options=b.basis.options)
    owner = SimpleNamespace(**vars(b))
    owner.domain = _domain(b.reference, b.columns)
    owner.real_space = _real_space(b.reference, owner.domain)
    owner.space = owner.real_space.space
    b.builder = make_builder(owner)
    result = _native_run(b)
    assert result.converged and result.memory.occupied_count == 1
    assert not result.diagnostics.complete_common_finite_torus_basis
    assert not result.periodic_energy_per_cell
    for name in ("correlation_energy_per_cell", "total_energy_per_cell"):
        with pytest.raises(RuntimeError, match="complete|finite.torus|per.cell"):
            getattr(result, name)
    assert not hasattr(b, "provider")


@pytest.mark.parametrize("field,reported", [
    ("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_control_storage_bytes", "control_storage_reservation_bytes"),
    ("maximum_per_worker_inventoried_bytes", "per_worker_inventoried_bytes"),
    ("maximum_node_inventoried_bytes", "required_node_memory_bytes"),
    ("maximum_pair_count", "pair_count"), ("maximum_progress_callbacks", "progress_callback_upper_bound"),
    ("maximum_gram_builds", "gram_builds_upper_bound"),
    ("maximum_factor_panels", "factor_panels_upper_bound"), ("maximum_tile_calls", "tile_calls_upper_bound"),
    ("maximum_work_units", "work_units_upper_bound"),
])
def test_native_outer_caps_precede_payload_and_progress(native_small, field, reported):
    b = native_small
    controls = _native_controls(b)
    p = _native_plan(b, controls)
    setattr(controls[3], field, getattr(p, reported))
    _native_plan(b, controls)
    setattr(controls[3], field, getattr(p, reported)-1)
    gauges = b.gauge.copy()
    gauges.flat[0] = np.nan
    events = []
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap|budget|bound|work|pair"):
        _native_run(b, controls, progress=events.append, gauges=gauges)
    assert not events


def test_native_complete_borrowed_source_and_extra_live_inventory(native_small):
    b = native_small
    controls = _native_controls(b)
    p = _native_plan(b, controls)
    assert p.borrowed_gauge_bytes == b.gauge.nbytes
    assert p.borrowed_gaussian_bytes == b.context.inventory.combined_borrowed_active_numeric_bytes
    assert p.borrowed_wannier_bytes == b.wannier.memory.retained_coefficient_bytes
    assert p.borrowed_basis_bytes == b.basis.memory.retained_output_bytes
    controls[2].other_live_bytes_per_worker += 12345
    q = _native_plan(b, controls)
    assert q.per_worker_inventoried_bytes-p.per_worker_inventoried_bytes == 12345
    assert q.required_node_memory_bytes-p.required_node_memory_bytes == 12345*p.replicas_per_node
    controls[3].maximum_pno_leaf_control_storage_bytes = 0
    bad = b.gauge.copy()
    bad.flat[0] = np.nan
    events = []
    with pytest.raises((ValueError, RuntimeError), match="control|positive|cap"):
        _native_run(b, controls, progress=events.append, gauges=bad)
    assert not events


@pytest.mark.parametrize("role", ["ao_basis", "auxiliary_basis"])
def test_native_same_size_changed_gaussian_basis_rejects_before_progress(native_small, role):
    from tests.test_periodic_aopair_fourier_panel import _basis as raw_basis
    changed = raw_basis([(0, (0., 0., 0.), [.9], [1.0], False),
        (0, (0., 0., 0.), [.3], [1.0], False)])
    events = []
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="basis|source|content|digest|census"):
        _native_run(native_small, progress=events.append, **{role: changed})
    assert not events


def test_native_gauge_descriptors_and_callback_type_are_bounded(native_small):
    for gauges in (native_small.gauge.astype(np.complex64), native_small.gauge.real,
                   native_small.gauge.reshape(-1)):
        with pytest.raises(TypeError, match="complex128"):
            _native_run(native_small, gauges=gauges)
    with pytest.raises((ValueError, TypeError), match="callable"):
        _native_run(native_small, progress=123)


def test_native_valid_but_foreign_localization_and_equal_shape_context_are_not_labels(native_small):
    from tests.test_periodic_correlation_wannier import _make as make_wannier
    b = native_small
    gauges = np.ascontiguousarray(-b.gauge)
    wannier = make_wannier(b.reference, gauges)
    with pytest.raises((ValueError, RuntimeError), match="gauge|localization|basis|lineage|source"):
        _native_run(b, gauges=gauges, wannier=wannier)
    other = _case(2)
    with pytest.raises((ValueError, RuntimeError), match="owner|source|HF|state|context|builder"):
        _native_plan(b, hf=other.hf)
    with pytest.raises((ValueError, RuntimeError), match="owner|source|HF|state|context|builder"):
        _native_plan(b, builder=other.builder)


@pytest.mark.parametrize("stage_name", ["BEGIN", "PAIR_DOMAIN_READY", "PAIR_COMPLETE", "SOLVER", "FINISHED"])
def test_native_callback_cancellation_publishes_no_result(native_small, stage_name):
    stage = getattr(core._PeriodicGaussianPairMP2Stage, stage_name)
    seen = []
    def stop(event):
        seen.append(event.stage)
        if event.stage == stage:
            raise RuntimeError("cancel direct Gram MP2")
    with pytest.raises(RuntimeError, match="cancel direct Gram MP2"):
        _native_run(native_small, progress=stop)
    assert stage in seen


def test_native_copied_controls_and_last_evaluated_nonconvergence(native_small):
    b = native_small
    controls = _native_controls(b)
    def mutate(event):
        if event.stage == core._PeriodicGaussianPairMP2Stage.BEGIN:
            controls[1].solver.maximum_iterations = 1
            controls[1].pnos.pno.pno.occupation_cutoff = 1.0
            controls[3].maximum_work_units = 0
    result = _native_run(b, controls, progress=mutate)
    assert result.converged and result.solver.final_snapshot.iteration >= 2
    assert result.diagnostics.zero_rank_pairs < result.memory.pair_count
    exhausted = _native_run(b, _native_controls(b, maximum_iterations=1))
    assert not exhausted.converged and not exhausted.periodic_energy_per_cell
    assert exhausted.solver.final_snapshot.iteration == 1
    with pytest.raises(RuntimeError, match="converged"):
        _ = exhausted.total_energy_per_cell


@pytest.mark.parametrize("stage_name", ["BEGIN", "FINISHED"])
def test_native_callback_gauge_payload_mutation_is_detected(stage_name):
    b = _case(1)
    gauges = b.gauge.copy()
    stage = getattr(core._PeriodicGaussianPairMP2Stage, stage_name)
    def mutate(event):
        if event.stage == stage:
            gauges.flat[0] *= -1
    with pytest.raises((ValueError, RuntimeError), match="gauge|payload|source|lineage|localization"):
        _native_run(b, progress=mutate, gauges=gauges)


@pytest.mark.parametrize("stage_name", ["BEGIN", "FINISHED"])
def test_native_callback_gauge_resize_rejects_before_stale_pointer_scan(stage_name):
    b = _case(1)
    gauges = b.gauge.copy()
    stage = getattr(core._PeriodicGaussianPairMP2Stage, stage_name)
    def resize(event):
        if event.stage == stage:
            gauges.resize((2, 2, 2), refcheck=False)
    with pytest.raises((ValueError, RuntimeError), match="gauge descriptors"):
        _native_run(b, progress=resize, gauges=gauges)


@pytest.mark.parametrize("stage_name", ["BEGIN", "FINISHED"])
def test_native_callback_real_basis_move_is_detected_before_stale_fock_scan(stage_name):
    from tests.test_periodic_gaussian_pair_mp2 import _run as old_mp2
    from tests.test_periodic_gaussian_pair_ccsd import _controls as old_ccsd_controls
    from tests.test_periodic_correlation_real_local_basis import _make as make_basis
    b = _case(1)
    # Deliberate exception to factory-first in THIS adversarial fixture only:
    # the existing native move seam requires a real legacy provider/warmstart.
    provider = _provider(b)
    warm = old_mp2(b, provider)
    replacement = make_basis(b, b.rows, b.selected, options=b.basis.options)
    receipt, fock = b.basis.identity_sha256, b.basis.fock_copy()
    assert replacement.identity_sha256 == receipt
    stage = getattr(core._PeriodicGaussianPairMP2Stage, stage_name)
    def consume(event):
        if event.stage == stage:
            options, live, caps = old_ccsd_controls(provider, iterations=1)
            with pytest.raises(ValueError, match="original basis storage changed across progress"):
                core._run_periodic_gaussian_pair_ccsd_basis_replacement_diagnostic(
                    b.reference, b.basis, replacement, provider, warm, options, live, caps,
                    core._PeriodicGaussianPairCCSDStage.BEGIN, False)
    with pytest.raises((ValueError, RuntimeError), match="source storage changed during progress"):
        _native_run(b, progress=consume)
    assert b.basis.identity_sha256 == receipt
    np.testing.assert_array_equal(b.basis.fock_copy(), fock)
    with pytest.raises(RuntimeError, match="consumed"):
        replacement.fock_copy()


def test_native_pair_and_diagonal_geometry_children_keep_complete_owner_alive():
    b = _case(1)
    result = _native_run(b)
    pair, embedding, solver = result.pair(0, 0), result.diagonal_generation_embedding(0), result.solver
    c, x, t = pair.coefficients_copy(), embedding.coefficients_copy(), solver.pair_copy(0, 0)
    del result, b
    gc.collect()
    np.testing.assert_array_equal(pair.coefficients_copy(), c)
    np.testing.assert_array_equal(embedding.coefficients_copy(), x)
    np.testing.assert_array_equal(solver.pair_copy(0, 0), t)


@pytest.mark.parametrize("kind,nk,frozen,cutoff", [
    ("he", 2, False, 0.0), ("he", 3, False, 0.0),
    ("he2", 2, False, 0.0), ("he2", 2, True, 0.0),
    ("he2", 2, False, 1.0),
])
def test_native_mp2_owners_feed_actual_k_j_overlap_without_a_factor_provider(kind, nk, frozen, cutoff):
    from types import SimpleNamespace
    from tests.test_periodic_gaussian_pair_interaction import (
        _build as interaction, _controls as interaction_controls,
        _check as check_interaction, _oracle as interaction_oracle,
    )
    b = _case(nk, kind=kind, frozen=frozen, full=kind == "he")
    warm = _native_run(b, _native_controls(b, cutoff=cutoff))
    k = 2 if kind == "he2" and not frozen else 1
    case = SimpleNamespace(b=b, a=warm.pair(0, 0), source=warm.pair(0, k), i=0, k=k)
    geometry = dict(geometry_a=warm.pair_generation_geometry_view(0, 0),
        geometry_b=warm.pair_generation_geometry_view(0, k))
    controls = interaction_controls(case, auxiliary_block=min(2, b.auxiliary.nbasis))
    # Explicit conservative whole-parent reservation, including the domain
    # builder's separate equal-content common geometry. It is not silently
    # dropped because the leaf can only see two borrowed PairSpace views.
    controls[2].other_retained_bytes_per_worker += (
        warm.memory.peak_owned_numerical_bytes+warm.memory.control_storage_reservation_bytes+
        b.builder.retained_numerical_bytes+b.builder.retained_control_storage_bytes)
    blocks = interaction(case, controls, **geometry)
    assert not hasattr(b, "provider")
    assert blocks.direct_gram_frames
    assert blocks.target_frame_identity_sha256 == case.a.pno_identity_sha256
    assert blocks.source_frame_identity_sha256 == case.source.pno_identity_sha256
    # Old-provider and independent common-AO reference creation follow the
    # complete new native MP2 -> actual Gaussian interaction chain.
    oracle = _oracle_inputs(b)
    case.dense = dict(o=warm.memory.occupied_count, eri=oracle.eri)
    check_interaction(blocks, interaction_oracle(case))
    if cutoff:
        assert blocks.k_ab_copy().shape == blocks.j_ba_copy().shape == (0, 0)
        assert blocks.memory.borrowed_geometry_numerical_bytes > 0
    saved = blocks.k_ab_copy(), blocks.j_ba_copy(), blocks.overlap_ba_copy()
    del case, b, warm, geometry, controls, oracle
    gc.collect()
    for actual, expected in zip((blocks.k_ab_copy(), blocks.j_ba_copy(), blocks.overlap_ba_copy()), saved, strict=True):
        np.testing.assert_array_equal(actual, expected)
