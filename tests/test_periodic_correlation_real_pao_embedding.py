"""Native common-to-pair PAO maps, independent tiny finite-torus oracles.

Synthetic generalized eigenproblems test geometry and failure gates. The
separate actual He2K2 fixture starts from native finite-source HF; neither
kind claims converged infinite-source chemistry or distinct-domain scaling.
"""

from __future__ import annotations

import gc
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_pao_space import _geometry
from tests.test_periodic_correlation_pao_domain import _make as _domain
from tests.test_periodic_correlation_real_pao_space import _make as _real_space
from tests.test_periodic_correlation_real_local_basis import _make as _basis
from tests.test_periodic_correlation_wannier import _make as _wannier
from tests.test_periodic_correlation_density_factors import _virtual
from tests.test_periodic_gaussian_selected_local_ccsd_t import _he2_bundle, _he2_controls, _prepare_leaves


_BUDGETS = dict(maximum_cross_overlap_imaginary_norm="cross_overlap_imaginary_frobenius_upper_bound",
    maximum_embedding_gram_error="embedding_gram_frobenius_upper_bound",
    maximum_pair_metric_error="pair_metric_frobenius_upper_bound",
    maximum_containment_norm="containment_s_absolute_upper_bound",
    maximum_pair_fock_error="pair_original_fock_frobenius_upper_bound",
    maximum_common_fock_error="common_projected_fock_frobenius_upper_bound",
    maximum_embedded_fock_error="embedded_original_fock_frobenius_upper_bound")
_CAPS = dict(maximum_common_dimension="common_dimension", maximum_pair_dimension="pair_dimension",
    maximum_owned_numerical_bytes="peak_owned_numerical_bytes",
    maximum_control_storage_bytes_per_worker="control_storage_reservation_bytes",
    maximum_per_worker_inventoried_bytes="per_worker_inventoried_bytes",
    maximum_node_inventoried_bytes="required_node_memory_bytes", maximum_work_units="work_units")


def _options(**changes):
    options = core._PeriodicCorrelationRealPAOEmbeddingOptions()
    for name in _BUDGETS:
        setattr(options, name, 1e-8)
    for name, value in changes.items():
        setattr(options, name, value)
    return options


def _live(**changes):
    live = core._PeriodicCorrelationRealPAOEmbeddingLiveInventory()
    live.other_live_numerical_bytes_per_worker = 2**20  # Other tiny fixture and oracle owners.
    live.other_live_control_bytes_per_worker = live.fixed_backend_margin_bytes_per_worker = 65536
    for name, value in changes.items():
        setattr(live, name, value)
    return live


def _caps(plan=None):
    caps = core._PeriodicCorrelationRealPAOEmbeddingCaps()
    values = dict(maximum_common_dimension=8, maximum_pair_dimension=8,
        maximum_owned_numerical_bytes=2**23, maximum_control_storage_bytes_per_worker=2**22,
        maximum_per_worker_inventoried_bytes=2**24, maximum_node_inventoried_bytes=2**26,
        maximum_work_units=10**12)
    for name, value in values.items():
        setattr(caps, name, value if plan is None else getattr(plan, _CAPS[name]))
    return caps


def _arguments(b):
    return (b.reference, b.basis, b.domain, b.real_space, b.selected,
        b.pair_domain, b.pair_real_space, b.pair_selected)


def _plan(b, *, options=None, live=None, caps=None):
    return core._plan_periodic_correlation_real_pao_embedding(*_arguments(b),
        _options() if options is None else options, _live() if live is None else live,
        _caps() if caps is None else caps)


def _make(b, *, options=None, live=None, caps=None):
    return core._make_periodic_correlation_real_pao_embedding(*_arguments(b),
        _options() if options is None else options, _live() if live is None else live,
        _caps() if caps is None else caps)


def _skew_reference(reference):
    """Same synthetic S/F/C, but independently declared skew Cartesian k."""
    old = reference.state
    data = core._PeriodicRestrictedMeanFieldInput()
    for name in ("calculation_identity", "reference_kind", "normalization", "periodic_dimension",
        "mesh", "is_shift", "converged", "n_basis", "n_effective_orbitals", "electrons_per_cell", "reference_energy_per_cell"):
        setattr(data, name, getattr(old, name))
    reciprocal = np.array([[2., .45, -.13], [.12, 3., .26], [0., .18, 5.]])
    data.reciprocal_lattice = reciprocal
    data.minimum_band_gap_hartree = .1
    data.validation_tolerance_version = core._PERIODIC_MEAN_FIELD_VALIDATION_TOLERANCE_VERSION
    cells = list(product(*(range(n) for n in old.mesh)))
    for k, address in enumerate(cells):
        data.add_kpoint(reciprocal@(np.asarray(address)/old.mesh), old.uniform_weight,
            old.overlap(k), old.fock(k), old.coefficients(k), old.orbital_energies(k), old.occupations(k),
            old.frozen_core_mask(k), old.correlated_occupied_mask(k), old.virtual_mask(k))
    state = core._make_periodic_restricted_mean_field_state(data)
    dims = reference.dimensions
    dims.external_bytes -= reference.state_resident_bytes
    return core._make_periodic_correlation_admitted_reference(state, dims, reference.budget)


def _bundle(mesh=(3, 1, 1), *, reduced=False, skew=False, pair_translation=None):
    reference, _ = _geometry(mesh, reduced=reduced)
    if skew:
        reference = _skew_reference(reference)
    nk = reference.state.n_kpoints
    # A spanning set of virtual-support AO columns needs no all-AO domain.
    # At mixed-even six-k, reduced=True gives just six columns/rank six.
    columns = np.array(list(product(range(nk), (2,) if reduced else (2, 3))), np.uint64)
    domain = _domain(reference, columns)
    real_space = _real_space(reference, domain)
    cells = np.asarray(list(product(*(range(n) for n in mesh))))
    gauge = np.ascontiguousarray(np.exp(.21j*np.sin(2*np.pi*(cells/mesh).sum(axis=1)))[:, None, None])
    b = SimpleNamespace(reference=reference, domain=domain, real_space=real_space, space=real_space.space,
        gauge=gauge, wannier=_wannier(reference, gauge), rows=np.array([[0, 0]], np.uint64))
    b.selected = _virtual(0, b.space.retained_dimension)
    b.basis = _basis(b, b.rows, b.selected)
    pair_columns = np.array([[0, 0], [nk-1, 2 if reduced else 3]], np.uint64)
    b.pair_domain = _domain(reference, pair_columns)
    b.pair_real_space = _real_space(reference, b.pair_domain)
    b.pair_selected = _virtual(0, b.pair_real_space.space.retained_dimension,
        nk-1 if pair_translation is None else pair_translation)
    return b


def _torus(b):
    """Build a unitary Fourier basis, not the native coefficient/phase leaf."""
    state = b.reference.state
    mesh = np.asarray(state.mesh)
    cells = np.asarray(list(product(*(range(n) for n in mesh))))
    nk, nao = len(cells), state.n_basis
    phase = np.exp(2j*np.pi*cells@(cells/mesh).T)
    fourier = np.kron(phase/np.sqrt(nk), np.eye(nao))
    sk = np.zeros((nk*nao, nk*nao), complex)
    fk, qk = np.zeros_like(sk), np.zeros_like(sk)
    for k in range(nk):
        sl = slice(k*nao, (k+1)*nao)
        s, c = state.overlap(k), state.coefficients(k)
        virtuals = c[:, np.asarray(state.virtual_mask(k), bool)]
        sk[sl, sl], fk[sl, sl] = s, state.fock(k)
        qk[sl, sl] = virtuals@virtuals.conj().T@s
    sr, fr, qr = [fourier@x@fourier.conj().T for x in (sk, fk, qk)]
    def frame(domain, space, selected):
        indices = []
        for a in range(domain.domain_dimension):
            cell, ao = domain.column(a)
            address = tuple((cells[cell]+cells[selected.translation_cell]) % mesh)
            indices.append(np.ravel_multi_index(address, tuple(mesh))*nao+ao)
        return qr[:, indices]@space.coefficients_copy()[:, selected.begin:selected.begin+selected.count]
    common = frame(b.domain, b.space, b.selected)
    pair = frame(b.pair_domain, b.pair_real_space.space, b.pair_selected)
    return SimpleNamespace(s=sr, f=fr, common=common, pair=pair, fourier=fourier, sk=sk)


def _oracle_check(b, result):
    p = _torus(b)
    expected = p.common.conj().T@p.s@p.pair
    x, eps = result.coefficients_copy(), result.energies_copy()
    np.testing.assert_allclose(expected.imag, 0, atol=3e-11)
    np.testing.assert_allclose(x, expected.real, atol=3e-11, rtol=2e-10)
    assert x.shape == (b.selected.count, b.pair_selected.count)
    np.testing.assert_allclose(x.T@x, np.eye(len(eps)), atol=6e-10)
    np.testing.assert_allclose(p.pair.conj().T@p.s@p.pair, np.eye(len(eps)), atol=5e-10)
    error = p.pair-p.common@x
    physical = np.trace(error.conj().T@p.s@error)
    assert abs(physical.imag) < 1e-23
    assert physical.real < 1e-18
    assert abs(result.diagnostics.containment_metric_quadratic_imaginary) < 1e-23
    assert result.diagnostics.containment_metric_quadratic_real < 1e-18
    assert result.diagnostics.containment_s_absolute_upper_bound < result.options.maximum_containment_norm
    # Both original F contractions are checked independently; copying eps
    # or testing only X^T X would not establish these physical relations.
    fpair = p.pair.conj().T@p.f@p.pair
    o = b.basis.memory.occupied_count
    fcommon = b.basis.fock_copy()[o:, o:]
    np.testing.assert_allclose(fpair, np.diag(eps), atol=6e-10)
    np.testing.assert_allclose(x.T@fcommon@x, fpair, atol=6e-10)
    embedded = (p.common@x).conj().T@p.f@(p.common@x)
    np.testing.assert_allclose(embedded, x.T@fcommon@x, atol=6e-10)
    np.testing.assert_array_equal(eps, b.pair_real_space.space.energies_copy()[
        b.pair_selected.begin:b.pair_selected.begin+b.pair_selected.count])
    assert result.common_basis_identity_sha256 == b.basis.identity_sha256
    assert result.state is b.reference.state
    assert not result.production_distinct_domain_scaling
    for name in ("common_frame_identity_sha256", "pair_frame_identity_sha256", "overlap_identity_sha256",
        "source_payload_receipt_sha256", "payload_sha256", "identity_sha256"):
        value = getattr(result, name)
        assert len(value) == 64 and int(value, 16) >= 0
    return p


@pytest.mark.parametrize("mesh,reduced,skew", [((1, 1, 1), False, False), ((2, 1, 1), False, False),
    ((3, 1, 1), False, False), ((2, 2, 1), False, True), ((2, 3, 1), True, True)])
def test_rectangular_native_cross_overlap_matches_independent_finite_torus(mesh, reduced, skew):
    b = _bundle(mesh, reduced=reduced, skew=skew)
    result = _make(b)
    p = _oracle_check(b, result)
    if np.prod(mesh) > 1:
        assert result.memory.common_dimension > result.memory.pair_dimension
    if np.prod(mesh) == 3:
        assert np.max(abs(b.reference.state.overlap(1).imag)) > 1e-3
    # The full retained-virtual projector excludes active AND frozen bands.
    assert b.reference.state.n_frozen_core == 1
    assert np.linalg.matrix_rank(p.common, tol=1e-9) == np.prod(mesh)*(1 if reduced else 2)


def test_actual_he2_two_k_hf_localization_and_distinct_pair_pao_domain():
    b = _he2_bundle((2, 1, 1))
    _prepare_leaves(b, localization_options=_he2_controls(b)[0].localization)
    assert b.hf.converged and b.localization.converged
    b.pair_domain = _domain(b.reference, np.array([[0, 0], [1, 2]], np.uint64))
    b.pair_real_space = _real_space(b.reference, b.pair_domain)
    b.pair_selected = _virtual(0, b.pair_real_space.space.retained_dimension, 1)
    result = _make(b)
    assert result.memory.common_dimension == 4 and result.memory.pair_dimension == 2
    _oracle_check(b, result)
    assert result.state is b.hf.state


@pytest.mark.parametrize("translation", [0, 1, 2])
def test_identical_full_frame_has_identity_embedding_and_single_owner_inventory(translation):
    b = _bundle()
    b.selected = _virtual(0, b.space.retained_dimension, translation)
    b.basis = _basis(b, b.rows, b.selected)
    b.pair_domain, b.pair_real_space, b.pair_selected = b.domain, b.real_space, _virtual(0, b.selected.count, translation)
    result = _make(b)
    np.testing.assert_allclose(result.coefficients_copy(), np.eye(b.selected.count), atol=4e-10)
    assert result.memory.unique_domain_owners == result.memory.unique_space_owners == result.memory.unique_real_wrapper_owners == 1
    _oracle_check(b, result)


def test_pair_subselection_and_translation_are_not_caller_labels():
    b = _bundle(pair_translation=0)
    first = _make(b)
    b.pair_selected = _virtual(0, 1, 0)
    sliced = _make(b)
    np.testing.assert_allclose(sliced.coefficients_copy(), first.coefficients_copy()[:, :1], atol=4e-12)
    b.pair_selected = _virtual(0, 1, 1)
    translated = _make(b)
    _oracle_check(b, translated)
    assert np.linalg.norm(translated.coefficients_copy()-sliced.coefficients_copy()) > 1e-3
    assert translated.pair_frame_identity_sha256 != sliced.pair_frame_identity_sha256
    assert translated.common_frame_identity_sha256 == sliced.common_frame_identity_sha256


def test_missing_pair_direction_fails_physical_containment_without_renormalization():
    b = _bundle((2, 1, 1))
    b.selected = _virtual(0, 1)
    b.basis = _basis(b, b.rows, b.selected)
    b.pair_domain, b.pair_real_space, b.pair_selected = b.domain, b.real_space, _virtual(1, 1)
    p = _torus(b)
    overlap = p.common.conj().T@p.s@p.pair
    assert np.linalg.norm(overlap) < 1e-8
    error = p.pair-p.common@overlap
    assert np.trace(error.conj().T@p.s@error).real > .99
    with pytest.raises(ValueError, match=r"X\^T X"):
        _make(b)
    # Relax ONLY the preceding Gram diagnostic so the original-S
    # containment gate itself is reached; no pair vector is renormalized.
    with pytest.raises(ValueError, match="not contained"):
        _make(b, options=_options(maximum_embedding_gram_error=2.))


def test_equal_content_distinct_owners_are_not_deduplicated_by_hash():
    b = _bundle()
    b.pair_domain, b.pair_real_space, b.pair_selected = b.domain, b.real_space, _virtual(0, b.selected.count)
    shared = _make(b)
    columns = np.array([b.domain.column(i) for i in range(b.domain.domain_dimension)], np.uint64)
    b.pair_domain = _domain(b.reference, columns)
    b.pair_real_space = _real_space(b.reference, b.pair_domain)
    assert b.pair_domain.pao_domain_identity_sha256 == b.domain.pao_domain_identity_sha256
    assert b.pair_real_space.identity_sha256 == b.real_space.identity_sha256
    separate = _make(b)
    assert separate.memory.unique_domain_owners == separate.memory.unique_space_owners == separate.memory.unique_real_wrapper_owners == 2
    assert separate.memory.live_domain_bytes == 2*shared.memory.live_domain_bytes
    assert separate.memory.live_space_bytes == 2*shared.memory.live_space_bytes
    np.testing.assert_array_equal(separate.coefficients_copy(), shared.coefficients_copy())


def test_exact_peak_phase_and_whole_live_inventory():
    b = _bundle()
    live = _live()
    p = _plan(b, live=live)
    n, r, nao = p.common_dimension, p.pair_dimension, p.n_basis
    assert p.output_numerical_bytes == 8*(n*r+r)
    assert p.overlap_phase_bytes == p.overlap.peak_owned_numerical_bytes
    assert p.conversion_phase_bytes == p.overlap.output_bytes+p.output_numerical_bytes
    assert p.physical_audit_phase_bytes == p.output_numerical_bytes+144*nao
    assert p.peak_owned_numerical_bytes == max(p.overlap_phase_bytes, p.conversion_phase_bytes, p.physical_audit_phase_bytes)
    assert p.live_basis_bytes == b.basis.memory.retained_output_bytes
    assert p.complete_borrowed_numerical_bytes == p.live_domain_bytes+p.live_space_bytes+p.live_basis_bytes
    assert p.per_worker_inventoried_bytes == (p.complete_borrowed_numerical_bytes+p.peak_owned_numerical_bytes
        +p.control_storage_reservation_bytes+live.other_live_numerical_bytes_per_worker+live.fixed_backend_margin_bytes_per_worker)
    assert p.required_node_memory_bytes == p.reference_base_node_bytes+p.replicas_per_node*p.per_worker_inventoried_bytes
    live.other_live_numerical_bytes_per_worker += 117
    live.other_live_control_bytes_per_worker += 229
    q = _plan(b, live=live)
    assert q.control_storage_reservation_bytes-p.control_storage_reservation_bytes == 229
    assert q.per_worker_inventoried_bytes-p.per_worker_inventoried_bytes == 346
    assert q.required_node_memory_bytes-p.required_node_memory_bytes == 346*p.replicas_per_node


@pytest.mark.parametrize("field", list(_CAPS))
def test_exact_caps_succeed_and_one_less_fails(field):
    b = _bundle((1, 1, 1))
    caps = _caps(_plan(b))
    assert _make(b, caps=caps).memory.pair_dimension > 0
    setattr(caps, field, getattr(caps, field)-1)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _make(b, caps=caps)


@pytest.mark.parametrize("field,diagnostic", list(_BUDGETS.items()))
def test_each_measured_frobenius_budget_is_enforced_without_operator_repair(field, diagnostic):
    b = _bundle()
    baseline = _make(b)
    value = getattr(baseline.diagnostics, diagnostic)
    assert value > 0  # Nontrivial odd-k fixture must expose a represented error.
    with pytest.raises(ValueError, match=r"imaginary|X\^T X|metric|contained|Fock"):
        _make(b, options=_options(**{field: np.nextafter(value, 0)}))


@pytest.mark.parametrize("field", list(_BUDGETS))
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_every_scientific_budget_is_explicit_positive_finite(field, value):
    b = _bundle((1, 1, 1))
    with pytest.raises(ValueError, match="strictly positive"):
        _plan(b, options=_options(**{field: value}))


def test_backend_margin_and_positive_nonempty_dimensions_are_required():
    b = _bundle((1, 1, 1))
    with pytest.raises(ValueError, match="backend margin"):
        _plan(b, live=_live(fixed_backend_margin_bytes_per_worker=0))
    b.pair_selected = _virtual(0, 0)
    with pytest.raises(ValueError, match="positive"):
        _plan(b)


@pytest.mark.parametrize("field", ["reference", "basis", "domain", "real_space", "pair_domain", "pair_real_space"])
def test_equal_content_foreign_state_and_mismatched_frame_owners_fail(field):
    b, foreign = _bundle((1, 1, 1)), _bundle((1, 1, 1))
    setattr(b, field, getattr(foreign, field))
    with pytest.raises((ValueError, RuntimeError), match="state|owner|frame|source|allocation|identity"):
        _make(b)


@pytest.mark.parametrize("field,value", [("begin", 2**64-1), ("count", 0), ("translation_cell", 99)])
def test_common_selection_must_match_certified_basis_and_pair_selection_must_fit(field, value):
    b = _bundle((2, 1, 1))
    setattr(b.selected, field, value)
    with pytest.raises((ValueError, IndexError, RuntimeError, OverflowError), match="selection|frame|positive|count|dimensions|overflow"):
        _plan(b)
    b = _bundle((2, 1, 1))
    setattr(b.pair_selected, field, value)
    with pytest.raises((ValueError, IndexError, RuntimeError, OverflowError), match="selection|frame|positive|count|dimensions|overflow"):
        _plan(b)


def test_detached_output_metadata_and_state_lifetime_no_mutable_native_children():
    b = _bundle()
    result = _make(b)
    coefficients, energies = result.coefficients_copy(), result.energies_copy()
    result.coefficients_copy()[:] = 42
    result.energies_copy()[:] = 43
    selection = result.pair_selection
    selection.begin = 99
    options = result.options
    options.maximum_containment_norm = 0
    assert result.pair_selection.begin == b.pair_selected.begin
    assert result.options.maximum_containment_norm > 0
    for i, j in np.ndindex(coefficients.shape):
        assert result.coefficient(i, j) == coefficients[i, j]
    for i, value in enumerate(energies):
        assert result.energy(i) == value
    state = result.state
    del b
    gc.collect()
    assert state.n_kpoints == result.memory.n_cells
    np.testing.assert_array_equal(result.coefficients_copy(), coefficients)
    np.testing.assert_array_equal(result.energies_copy(), energies)
    with pytest.raises(IndexError):
        result.coefficient(coefficients.shape[0], 0)
    with pytest.raises(IndexError):
        result.energy(len(energies))
    assert not hasattr(result, "common_space") and not hasattr(result, "pair_space")
