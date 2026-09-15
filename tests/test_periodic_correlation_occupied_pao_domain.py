"""Tiny typed-owner signed Mulliken/one-step PAO domains; no chemistry jobs."""

from __future__ import annotations

import gc
import hashlib
from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_bloch_iao import _make as _make_iao, _provenance
from tests.test_periodic_correlation_diabatic_seed import _make as _make_seed
from tests.test_periodic_correlation_iao_optimizer import (
    _case as _optimizer_case,
    _run as _run_optimizer,
)
from tests.test_periodic_correlation_pao_domain import _reference as _pao_reference
from tests.test_periodic_correlation_pair_topology import _make_budget, _make_dimensions
from tests.test_periodic_correlation_wannier import _options as _wannier_options


def _options(**changes):
    out = core._PeriodicCorrelationOccupiedPAODomainOptions()
    out.full_domain = False
    out.mulliken_population_cutoff = 0.01
    out.pao_tail_cutoff = 0.1
    out.normalization_tolerance = 2e-10
    out.maximum_population_imaginary_magnitude = 2e-10
    out.maximum_pao_imaginary_magnitude = 2e-10
    out.maximum_negative_absolute_population = 4.0
    out.maximum_seed_omitted_absolute_population = 4.0
    out.maximum_expanded_omitted_absolute_population = 4.0
    for name, value in changes.items():
        setattr(out, name, value)
    return out


def _inventory(**changes):
    out = core._PeriodicCorrelationOccupiedPAODomainInventory()
    out.backend_allowance_bytes = 4096
    for name, value in changes.items():
        setattr(out, name, value)
    return out


def _caps(**changes):
    out = core._PeriodicCorrelationOccupiedPAODomainCaps()
    out.maximum_atom_cells = 64
    out.maximum_seed_atom_cells = 64
    out.maximum_expanded_atom_cells = 64
    out.maximum_owned_numerical_bytes = 1 << 20
    out.maximum_control_storage_bytes = 1 << 20
    out.maximum_worker_bytes = 8 << 20
    out.maximum_work_units = 10**12
    for name, value in changes.items():
        setattr(out, name, value)
    return out


def _wannier(reference, optimizer):
    s = reference.state
    plan = core._plan_periodic_correlation_wannier(s.mesh, s.n_basis, s.n_correlated_occupied)
    return core._make_periodic_correlation_wannier_from_iao_optimizer(
        reference, optimizer, plan.peak_owned_numerical_bytes, _wannier_options())


def _from_optimizer_case(case):
    optimizer = _run_optimizer(case)
    assert optimizer.converged
    reference = case["reference"]
    return dict(reference=reference, optimizer=optimizer, wannier=_wannier(reference, optimizer),
                mapping=np.arange(reference.state.n_basis, dtype=np.uint64),
                atom_count=reference.state.n_basis)


def _bundle(mesh=(1, 1, 1), *, frozen=(), nactive=1, complex_gauge=False):
    return _from_optimizer_case(_optimizer_case(mesh, frozen=frozen, nactive=nactive,
                                                complex_gauge=complex_gauge))


def _reduced_bundle(mesh=(3, 1, 1), *, reduced=True, broken_projector_tr=False):
    ref, cells, c, s, _ = _pao_reference(mesh, reduced=reduced,
                                        broken_projector_tr=broken_projector_tr)
    labels = np.arange(2, dtype=np.uint64)
    points = [_make_iao(ref, np.ascontiguousarray(s[k] @ c[k, :, :2]),
                        np.eye(2, dtype=complex), labels, point=k) for k in range(len(cells))]
    case = dict(reference=ref, seed=_make_seed(ref), points=points, labels=labels)
    return _from_optimizer_case(case)


def _analytic_bundle(three_atoms=False):
    """Actual native owner chain for exact signed/oblique-projector witnesses.

    These are algebraic states, not chemically authenticated HF results.
    Minimal AO=c spans the occupied exactly, so its IAO source is a genuine
    positive Gram block and Gamma O(1) optimization has no tangent direction.
    """
    if three_atoms:
        occupied = np.array([1.0, 2.0, -4.0])
        dual = np.array([0.75, 0.25, 0.0625])
        s = np.outer(dual, dual) + np.eye(3) - np.outer(occupied, occupied)/21
        columns = [occupied]
        for candidate in np.eye(3)[:2]:
            vector = candidate.copy()
            for previous in columns:
                vector -= previous * (previous @ s @ vector)
            columns.append(vector / np.sqrt(vector @ s @ vector))
        c = np.column_stack(columns)
    else:
        s = np.array([[1.0, 0.75], [0.75, 1.0]])
        occupied = np.array([1.0, -2.0])/np.sqrt(2.0)
        c = np.column_stack((occupied, np.array([5.0, -2.0])/np.sqrt(14.0)))
    n = len(s)
    np.testing.assert_allclose(c.T @ s @ c, np.eye(n), atol=2e-14)
    eps = np.r_[-1.0, np.full(n-1, 0.5)]
    f = s @ c @ np.diag(eps) @ c.T @ s
    data = core._PeriodicRestrictedMeanFieldInput()
    data.calculation_identity = "1"*64
    data.reference_kind = core._PeriodicMeanFieldReferenceKind.RESTRICTED_HARTREE_FOCK
    data.normalization = (core._PeriodicMeanFieldNormalizationConvention
                          .UNNORMALIZED_AO_BLOCH_SUMS_UNIFORM_FULL_BZ_WEIGHTS)
    data.periodic_dimension = 3
    data.mesh = (1, 1, 1)
    data.is_shift = (0, 0, 0)
    data.reciprocal_lattice = np.eye(3)
    data.converged = True
    data.n_basis = data.n_effective_orbitals = n
    data.electrons_per_cell = 2
    data.reference_energy_per_cell = -1.5
    data.minimum_band_gap_hartree = 0.1
    data.validation_tolerance_version = core._PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION
    data.add_kpoint(np.zeros(3), 1.0, s.astype(complex), f.astype(complex), c.astype(complex),
                    eps, np.r_[2.0, np.zeros(n-1)], [0]*n, [1]+[0]*(n-1), [0]+[1]*(n-1))
    state = core._make_periodic_restricted_mean_field_state(data)
    dims = _make_dimensions((1, 1, 1), 1)
    dims.n_basis = dims.n_effective_orbitals = n
    dims.n_home_total_occupied = 1
    dims.n_home_virtual = n-1
    dims.n_auxiliary = 2*n
    dims.domain_ao_support_upper_bound = dims.domain_pao_upper_bound = n
    dims.domain_pno_upper_bound = n
    dims.domain_local_auxiliary_upper_bound = 2*n
    reference = core._make_periodic_correlation_admitted_reference(state, dims, _make_budget())
    cross = np.ascontiguousarray(s @ occupied[:, None], dtype=complex)
    minimal = np.ones((1, 1), complex)
    provenance = _provenance(reference, 1,
        cross_overlap_identity=hashlib.sha256(cross.tobytes()).hexdigest(),
        minimal_basis_identity=hashlib.sha256(minimal.tobytes()).hexdigest())
    labels = np.array([17], np.uint64)
    point = _make_iao(reference, cross, minimal, labels, point=0, provenance=provenance)
    bundle = _from_optimizer_case(dict(reference=reference, seed=_make_seed(reference),
                                       points=[point], labels=labels))
    return bundle


def _plan(bundle, *, options=None, inventory=None, caps=None):
    return core._plan_periodic_correlation_occupied_pao_domain(
        bundle["reference"], bundle["optimizer"], bundle["wannier"], bundle["atom_count"],
        _options() if options is None else options, _inventory() if inventory is None else inventory,
        _caps() if caps is None else caps)


def _run(bundle, *, options=None, inventory=None, caps=None, occupied=0, mapping=None):
    return core._select_periodic_correlation_occupied_pao_domain(
        bundle["reference"], bundle["optimizer"], bundle["wannier"],
        bundle["mapping"] if mapping is None else mapping, bundle["atom_count"], occupied,
        _options() if options is None else options, _inventory() if inventory is None else inventory,
        _caps() if caps is None else caps)


def _dense_torus(bundle, occupied=0):
    """Independent full real-space S and retained-virtual metric projector.

    This deliberately builds the tiny whole torus using a unitary Fourier
    matrix, not the native modulo lookup, one-column projector or DFT leaf.
    """
    state = bundle["reference"].state
    mesh = np.array(state.mesh)
    cells = np.array(list(product(*(range(n) for n in mesh))))
    nk, n, nv = len(cells), state.n_basis, state.n_virtual
    phase = np.exp(2j * np.pi * cells @ (cells / mesh).T)
    fourier = np.kron(phase / np.sqrt(nk), np.eye(n))
    sblock = np.zeros((nk*n, nk*n), complex)
    cvblock = np.zeros((nk*n, nk*nv), complex)
    occupied_k = []
    gauges = bundle["optimizer"].gauges_copy()
    for k in range(nk):
        s = np.asarray(state.overlap(k))
        c = np.asarray(state.coefficients(k))
        active = np.asarray(state.correlated_occupied_mask(k), bool)
        virtual = np.asarray(state.virtual_mask(k), bool)
        sblock[k*n:(k+1)*n, k*n:(k+1)*n] = s
        cvblock[k*n:(k+1)*n, k*nv:(k+1)*nv] = c[:, virtual]
        occupied_k.append(c[:, active] @ gauges[k, :, occupied])
    metric = fourier @ sblock @ fourier.conj().T
    virtuals = fourier @ cvblock
    projector = virtuals @ virtuals.conj().T @ metric
    beta = fourier @ np.array(occupied_k).reshape(-1) / np.sqrt(nk)
    gross = (beta.conj() * (metric @ beta)).reshape(nk, n)
    populations = np.column_stack([gross[:, bundle["mapping"] == a].sum(axis=1)
                                    for a in range(bundle["atom_count"])])
    return populations, projector, beta, metric


def _oracle(bundle, options, occupied=0):
    populations, projector, beta, metric = _dense_torus(bundle, occupied)
    nk, atoms = populations.shape
    n = bundle["reference"].state.n_basis
    seeds = [(r, a) for r in range(nk) for a in range(atoms)
             if options.full_domain or populations[r, a].real > options.mulliken_population_cutoff]
    expanded = set(seeds)
    if not options.full_domain:
        for ra, atom_a in seeds:
            rows = ra*n + np.flatnonzero(bundle["mapping"] == atom_a)
            for rb in range(nk):
                for atom_b in range(atoms):
                    cols = rb*n + np.flatnonzero(bundle["mapping"] == atom_b)
                    if np.abs(projector[np.ix_(rows, cols)]).sum() > options.pao_tail_cutoff:
                        expanded.add((rb, atom_b))
    return (populations, np.array(seeds, np.uint64).reshape(-1, 2),
            np.array(sorted(expanded), np.uint64).reshape(-1, 2), beta, metric)


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 2, 2)])
@pytest.mark.parametrize("full", [False, True])
def test_actual_typed_localizer_signed_population_and_one_step_match_dense_torus(mesh, full):
    bundle = _bundle(mesh)
    options = _options(full_domain=full)
    result = _run(bundle, options=options)
    populations, seeds, expanded, beta, metric = _oracle(bundle, options)
    np.testing.assert_allclose(result.populations_copy(), populations.real, atol=3e-12, rtol=2e-12)
    np.testing.assert_array_equal(result.seeds_copy(), seeds)
    np.testing.assert_array_equal(result.expanded_copy(), expanded)
    assert np.max(np.abs(populations.imag)) < 3e-12
    assert beta.conj() @ metric @ beta == pytest.approx(1.0, abs=3e-12)
    assert result.diagnostics.population_sum == pytest.approx(1.0, abs=3e-12)
    assert result.diagnostics.actual_peak_owned_numerical_bytes <= result.memory.peak_owned_numerical_bytes
    assert not result.hf_basis_source_authenticated and not result.extended_ccsd_domain
    for cell, atom in product(range(len(populations)), range(bundle["atom_count"])):
        assert result.population(cell, atom) == result.populations_copy()[cell, atom]
    if full:
        assert result.diagnostics.projected_pao_columns == 0
        assert result.memory.tail_score_bytes_upper == 0


@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (3, 1, 1)])
def test_frozen_and_discarded_scf_directions_do_not_reenter_the_pao_projector(mesh, reduced):
    bundle = _reduced_bundle(mesh, reduced=reduced)
    options = _options(mulliken_population_cutoff=0.15, pao_tail_cutoff=0.15)
    result = _run(bundle, options=options)
    populations, seeds, expanded, _, _ = _oracle(bundle, options)
    np.testing.assert_allclose(result.populations_copy(), populations.real, atol=3e-11)
    np.testing.assert_array_equal(result.seeds_copy(), seeds)
    np.testing.assert_array_equal(result.expanded_copy(), expanded)
    assert result.state.n_frozen_core == 1
    if reduced:
        assert result.state.n_effective_orbitals < result.state.n_basis


def test_broken_retained_virtual_time_reversal_fails_the_actual_home_pao_gate():
    bundle = _reduced_bundle(broken_projector_tr=True)
    with pytest.raises(ValueError, match="PAO.*imaginary"):
        _run(bundle, options=_options(mulliken_population_cutoff=0.0, pao_tail_cutoff=0.0))


def test_cutoffs_are_strict_full_mode_is_explicit_and_thresholded_empty_is_valid():
    bundle = _bundle((3, 1, 1))
    all_domain = _run(bundle, options=_options(full_domain=True))
    highest = float(np.max(all_domain.populations_copy()))
    equal = _run(bundle, options=_options(mulliken_population_cutoff=highest))
    assert equal.seeds_copy().shape == equal.expanded_copy().shape == (0, 2)
    assert equal.diagnostics.projected_pao_columns == 0
    below = _run(bundle, options=_options(mulliken_population_cutoff=np.nextafter(highest, 0.0),
                                          pao_tail_cutoff=10.0))
    assert len(below.seeds_copy()) > 0
    np.testing.assert_array_equal(below.expanded_copy(), below.seeds_copy())
    full = _run(bundle, options=_options(full_domain=True, mulliken_population_cutoff=10.0,
                                         pao_tail_cutoff=10.0))
    assert len(full.seeds_copy()) == 3*bundle["atom_count"]
    np.testing.assert_array_equal(full.seeds_copy(), full.expanded_copy())


def test_exact_signed_negative_population_and_nonsymmetric_pao_tail_direction():
    bundle = _analytic_bundle()
    options = _options(mulliken_population_cutoff=1/8, pao_tail_cutoff=9/16)
    result = _run(bundle, options=options)
    np.testing.assert_allclose(result.populations_copy(), [[-1/4, 5/4]], atol=3e-13)
    np.testing.assert_array_equal(result.seeds_copy(), [[0, 1]])
    np.testing.assert_array_equal(result.expanded_copy(), [[0, 1]])
    # Correct row B -> candidate PAO-column A is 1/2; the transpose is 5/8.
    # A transposed projector or |population| replacement changes the domain.
    _, q, _, _ = _dense_torus(bundle)
    np.testing.assert_allclose(q, [[5/4, 5/8], [-1/2, -1/4]], atol=3e-13)
    assert abs(q[1, 0]) < 9/16 < abs(q[0, 1])
    assert result.diagnostics.negative_absolute_population == pytest.approx(1/4, abs=3e-13)
    assert result.diagnostics.seed_omitted_absolute_population == pytest.approx(1/4, abs=3e-13)
    for field in ("maximum_negative_absolute_population", "maximum_seed_omitted_absolute_population",
                  "maximum_expanded_omitted_absolute_population"):
        with pytest.raises(ValueError, match="population.*budget"):
            _run(bundle, options=_options(mulliken_population_cutoff=1/8, pao_tail_cutoff=9/16,
                                          **{field: 0.24}))
    assert len(_run(bundle, options=_options(full_domain=True)).expanded_copy()) == 2


def test_exact_one_step_expansion_never_recurses_through_a_newly_added_atom():
    bundle = _analytic_bundle(three_atoms=True)
    options = _options(mulliken_population_cutoff=5/8, pao_tail_cutoff=3/32)
    result = _run(bundle, options=options)
    np.testing.assert_allclose(result.populations_copy(), [[3/4, 1/2, -1/4]], atol=5e-13)
    np.testing.assert_array_equal(result.seeds_copy(), [[0, 0]])
    np.testing.assert_array_equal(result.expanded_copy(), [[0, 0], [0, 1]])
    _, q, _, _ = _dense_torus(bundle)
    assert abs(q[0, 2]) < 3/32 < abs(q[1, 2])


def test_tail_cutoff_exactly_equal_to_actual_strength_is_excluded():
    bundle = _analytic_bundle()
    first = _run(bundle, options=_options(mulliken_population_cutoff=1/8, pao_tail_cutoff=9/16))
    strength = first.diagnostics.maximum_pao_tail_strength
    assert strength == pytest.approx(0.5, abs=3e-13)
    at = _run(bundle, options=_options(mulliken_population_cutoff=1/8, pao_tail_cutoff=strength))
    np.testing.assert_array_equal(at.expanded_copy(), [[0, 1]])
    below = _run(bundle, options=_options(mulliken_population_cutoff=1/8,
                                          pao_tail_cutoff=np.nextafter(strength, 0.0)))
    np.testing.assert_array_equal(below.expanded_copy(), [[0, 0], [0, 1]])


def test_all_occupied_owner_gauges_and_active_indices_are_charged_once():
    bundle = _bundle((3, 1, 1), frozen=(1,), nactive=2, complex_gauge=True)
    options = _options(full_domain=True)
    result = _run(bundle, options=options, occupied=1)
    expected = _oracle(bundle, options, occupied=1)
    np.testing.assert_allclose(result.populations_copy(), expected[0].real, atol=3e-11)
    p = result.memory
    assert p.borrowed_optimizer_numerical_bytes == bundle["optimizer"].memory.output_numerical_bytes
    assert p.borrowed_wannier_numerical_bytes == bundle["wannier"].memory.retained_coefficient_bytes
    assert p.caller_mapping_bytes == 8*bundle["reference"].state.n_basis
    assert result.home_occupied_index == 1
    # Native schemas differ: equality of these strings would be a false gate.
    assert bundle["optimizer"].gauge_payload_sha256 != bundle["wannier"].gauge_payload_sha256


@pytest.mark.parametrize("field,plan_field", [
    ("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_worker_bytes", "worker_bytes"),
    ("maximum_work_units", "planned_work_units"),
])
def test_exact_resource_boundary_then_one_byte_or_work_unit_less_before_bad_mapping(field, plan_field):
    bundle = _bundle((2, 1, 1))
    p = _plan(bundle)
    _run(bundle, caps=_caps(**{field: getattr(p, plan_field)}))
    bad = np.full_like(bundle["mapping"], 2**64-1)
    with pytest.raises((ValueError, RuntimeError), match="cap|memory|work"):
        _run(bundle, caps=_caps(**{field: getattr(p, plan_field)-1}), mapping=bad)


def test_other_live_controls_are_in_the_total_control_cap_and_worker_inventory():
    bundle = _bundle()
    inventory = _inventory(other_live_control_bytes=71, other_live_numerical_bytes=97)
    p = _plan(bundle, inventory=inventory)
    total = p.fixed_control_storage_bytes + p.borrowed_owner_control_bytes + 71
    _run(bundle, inventory=inventory, caps=_caps(maximum_control_storage_bytes=total))
    with pytest.raises((ValueError, RuntimeError), match="control"):
        _run(bundle, inventory=inventory, caps=_caps(maximum_control_storage_bytes=total-1))
    base = _plan(bundle)
    assert p.worker_bytes - base.worker_bytes == 71 + 97


@pytest.mark.parametrize("mapping", [np.array([0, 1], np.int64), np.array([[0, 1]], np.uint64),
                                     np.array([0], np.uint64), np.array([0, 2], np.uint64),
                                     np.array([0, 0], np.uint64), np.arange(4, dtype=np.uint64)[::2]])
def test_mapping_dtype_shape_range_empty_atom_and_strides_are_fail_closed(mapping):
    with pytest.raises((TypeError, ValueError, IndexError), match="mapping|atom|label|uint64|AO"):
        _run(_bundle(), mapping=mapping)


@pytest.mark.parametrize("field", ["mulliken_population_cutoff", "pao_tail_cutoff", "normalization_tolerance",
    "maximum_population_imaginary_magnitude", "maximum_pao_imaginary_magnitude",
    "maximum_negative_absolute_population", "maximum_seed_omitted_absolute_population",
    "maximum_expanded_omitted_absolute_population"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -0.01])
def test_scientific_controls_require_explicit_finite_nonnegative_values(field, value):
    with pytest.raises(ValueError, match="budget|cut|tolerance"):
        _run(_bundle(), options=_options(**{field: value}))


def test_genuine_but_wrong_native_state_owner_and_unlocalized_gauge_wannier_rejected():
    bundle, replacement = _bundle(), _bundle()
    bundle["optimizer"] = replacement["optimizer"]
    with pytest.raises(ValueError, match="optimizer|owner"):
        _run(bundle)
    bundle = _bundle()
    state = bundle["reference"].state
    p = core._plan_periodic_correlation_wannier(state.mesh, state.n_basis, state.n_correlated_occupied)
    bundle["wannier"] = core._make_periodic_correlation_wannier(bundle["reference"],
        bundle["optimizer"].gauges_copy(), p.peak_owned_numerical_bytes, _wannier_options())
    with pytest.raises(ValueError, match="optimizer|Wannier"):
        _run(bundle)


def test_result_owns_compact_outputs_independently_and_seals_mapping_options_payload():
    bundle = _bundle((2, 1, 1))
    result = _run(bundle)
    expected = result.populations_copy()
    payload = result.payload_identity_sha256
    swapped = dict(bundle, mapping=bundle["mapping"][::-1].copy())
    assert _run(swapped).mapping_identity_sha256 != result.mapping_identity_sha256
    assert _run(bundle, options=_options(pao_tail_cutoff=0.11)).occupied_pao_domain_identity_sha256 != result.occupied_pao_domain_identity_sha256
    del bundle
    gc.collect()
    copy = result.populations_copy()
    copy[:] = 99
    np.testing.assert_array_equal(result.populations_copy(), expected)
    assert len(payload) == len(result.occupied_pao_domain_identity_sha256) == 64
    assert result.diagnostics.retained_numerical_bytes == 8*expected.size + 16*(
        result.diagnostics.seed_count + result.diagnostics.expanded_count)
    with pytest.raises(IndexError):
        result.population(len(expected), 0)
    with pytest.raises(IndexError):
        result.seed(result.diagnostics.seed_count)
    with pytest.raises(IndexError):
        result.expanded(result.diagnostics.expanded_count)
