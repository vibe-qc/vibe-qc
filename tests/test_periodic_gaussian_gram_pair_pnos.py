"""Actual local PAO Gram -> SC-MP2 density -> PNO, without prior rows.

Nejad2025 Eqs.37-40 are evaluated independently in the genuine PAO pair
space. Sun2017 Eqs.13/16/21 fix the finite-source Fourier/metric oracle.
Dense NumPy contractions here are tiny validation, never runtime providers.
"""

from __future__ import annotations

import gc
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_selected_local_ccsd_t import (
    _bundle, _he2_bundle, _he2_controls, _prepare_leaves,
)
from tests.test_periodic_gaussian_frozen_core_correlation import _frozen_bundle
from tests.test_periodic_gaussian_pair_domain_builder import (
    _build as _builder, _pair_options, _pair_caps, _live as _geometry_live,
    _embedding as _old_embedding,
)
from tests.test_periodic_gaussian_occupied_pao_domain import _controls as _occupied_controls
from tests.test_periodic_correlation_real_pao_embedding import _domain, _real_space
from tests.test_periodic_correlation_real_local_basis import _options as _basis_options
from tests.test_periodic_gaussian_density_gram import _controls as _gram_controls, _oracle as _common_panels
from tests.test_periodic_gaussian_embedded_pair_pnos import _options as _pno_options, _make as _old_pnos
from tests.test_periodic_gaussian_real_local_provider import _make as _provider
from tests.test_periodic_correlation_density_factors import _occupied_columns
from tests.test_periodic_correlation_wannier import _make as _wannier
from tests.test_periodic_correlation_real_local_provider import _expected_rows
from tests.test_periodic_gaussian_mixed_pair_factors import _q_data
from tests.test_periodic_gaussian_fock import _physical_tile
from tests.test_periodic_gaussian_pair_space import (
    _options as _space_options, _live as _space_live, _caps as _space_caps,
    _overlap as _space_overlap, _overlap_caps,
)
from tests.test_periodic_aopair_fourier_panel import _basis as _raw_basis


_CAPS = dict(maximum_common_dimension="common_virtual_dimension",
    maximum_generation_dimension="generation_dimension", maximum_owned_numerical_bytes="peak_owned_numerical_bytes",
    maximum_control_storage_bytes="control_storage_reservation_bytes", maximum_worker_bytes="worker_bytes",
    maximum_node_bytes="required_node_memory_bytes", maximum_work_units="work_units")


def _case(nk=2, *, kind="he", frozen=False, full=True):
    """Authentic sources only; old provider creation is explicitly deferred."""
    b = (_frozen_bundle if frozen else _he2_bundle if kind == "he2" else _bundle)((nk, 1, 1))
    _prepare_leaves(b, localization_options=_he2_controls(b)[0].localization if frozen or kind == "he2" else None)
    b.domain_options = _occupied_controls(b, full=full, cut=.2, tail=.1)[0]
    # Keep the ORIGINAL common geometry alive for later independent oracles.
    # The builder consumes a genuinely regenerated, equal-content native
    # domain/space; no source labels or numerical arrays are substituted.
    owner = SimpleNamespace(**vars(b))
    owner.domain = _domain(b.reference, b.columns)
    owner.real_space = _real_space(b.reference, owner.domain)
    owner.space = owner.real_space.space
    assert owner.domain.pao_domain_identity_sha256 == b.domain.pao_domain_identity_sha256
    assert owner.space.pao_space_identity_sha256 == b.space.pao_space_identity_sha256
    b.builder = _builder(owner)
    assert not hasattr(b, "provider")
    return b


def _geometry(b, i=0, j=0):
    return core._make_periodic_gaussian_pair_domain_geometry(b.reference, b.basis, b.builder,
        i, j, _pair_options(), _geometry_live(), _pair_caps(b))


def _controls(b, *, cutoff=0.0, auxiliary_block=1, pair_block=3):
    gram_config, gram_options, _, gram_caps = _gram_controls(b,
        auxiliary_block=auxiliary_block, pair_block=pair_block)
    # Explicit small nested ceilings: the enclosing plan charges these
    # ceilings, not only observed cost. No tolerance or source change.
    gram_caps.resources.maximum_owned_numeric_bytes = 2**20
    gram_caps.resources.maximum_work_units = 10**16
    config = core._PeriodicGaussianGramPairPNOConfig()
    config.gram = gram_config
    options = core._PeriodicGaussianGramPairPNOOptions()
    options.local_basis, options.gram, options.pno = _basis_options(), gram_options, _pno_options(cutoff=cutoff)
    options.maximum_occupied_fock_difference = 1e-10
    options.maximum_retained_diagonal_projection_norm = 1e-10
    live = core._PeriodicGaussianGramPairPNOLiveInventory()
    live.other_live_numerical_bytes_per_worker = 2**20
    live.other_live_control_bytes_per_worker = live.fixed_backend_margin_bytes_per_worker = 65536
    caps = core._PeriodicGaussianGramPairPNOCaps()
    local = core._PeriodicCorrelationRealLocalBasisCaps()
    local.maximum_owned_numerical_bytes, local.maximum_work_units = 65536, 10**12
    caps.local_basis, caps.gram = local, gram_caps
    caps.maximum_gram_control_storage_bytes = 2**22
    caps.maximum_common_dimension = caps.maximum_generation_dimension = 4
    caps.maximum_owned_numerical_bytes = 2**24
    caps.maximum_control_storage_bytes = 2**23
    caps.maximum_worker_bytes = 2**26
    caps.maximum_node_bytes = 2**27
    caps.maximum_work_units = 10**18
    return config, options, live, caps


def _arguments(b, geometry, controls=None, **changes):
    config, options, live, caps = _controls(b) if controls is None else controls
    args = dict(hf=b.hf, reference=b.reference, wannier=b.wannier, basis=b.basis,
        geometry=geometry, config=config, options=options, live=live, caps=caps)
    args.update(changes)
    return args


def _plan(b, geometry, controls=None, **changes):
    return core._plan_periodic_gaussian_gram_pair_pnos_diagnostic(**_arguments(b, geometry, controls, **changes))


def _make(b, geometry, controls=None, **changes):
    args = _arguments(b, geometry, controls)
    args.update(ao_basis=b.ao, auxiliary_basis=b.auxiliary, gauges=b.gauge)
    args.update(changes)
    return core._make_periodic_gaussian_gram_pair_pnos_diagnostic(**args)


def _oracle_inputs(b):
    """Call ONLY after all direct-source outputs of interest are built."""
    actual = _common_panels(b)
    raw = np.einsum("xAqp,xArs->pqrs", actual.panels.conj(), actual.panels)
    assert np.max(np.abs(raw.imag), initial=0.0) < 1e-10
    return SimpleNamespace(panels=actual.panels, eri=raw.real, provider=_provider(b))


def _local_columns(b, geometry, k):
    state = b.reference.state
    cells = np.asarray(list(product(*(range(n) for n in state.mesh))))
    frac = cells[k] / np.asarray(state.mesh)
    canonical = state.coefficients(k)[:, np.asarray(state.virtual_mask(k), bool)]
    projector = canonical @ canonical.conj().T @ state.overlap(k)
    columns = geometry.domain_columns_copy()
    shift = cells[geometry.pair_translation]
    phases = np.exp(-2j*np.pi*((cells[columns[:, 0]]+shift) @ frac))
    virtual = (projector[:, columns[:, 1]]*phases) @ geometry.real_coefficients_copy()
    slots = [geometry.occupied_slot_i]
    if geometry.occupied_slot_j != geometry.occupied_slot_i:
        slots.append(geometry.occupied_slot_j)
    occupied = _occupied_columns(b, k, b.rows[slots])
    return np.column_stack((occupied, virtual))


def _local_oracle(b, geometry):
    """No common exported C: expand original PAOs and contract AO tiles."""
    nk, nao, aux = b.context.n_kpoints, b.ao.nbasis, b.auxiliary.nbasis
    columns = [_local_columns(b, geometry, k) for k in range(nk)]
    size = columns[0].shape[1]
    panels = np.zeros((nk, aux, size, size), complex)
    fock = sum(c.conj().T@b.reference.state.fock(k)@c/nk for k, c in enumerate(columns))
    overlap = sum(c.conj().T@b.reference.state.overlap(k)@c/nk for k, c in enumerate(columns))
    np.testing.assert_allclose(overlap, np.eye(size), atol=2e-10)
    np.testing.assert_allclose(fock.imag, 0, atol=1e-10)
    for q in range(nk):
        source, _, w = _q_data(b, q)
        for bra in range(nk):
            ket = b.context.ket_index(bra, q)
            tile = _physical_tile(b, source, w, bra).matrix.reshape(aux, nao, nao)
            panels[q] += np.einsum("ml,nr,pmn->plr", columns[bra].conj(), columns[ket], tile)
    panels /= nk*np.sqrt(nk)
    rows, labels = _expected_rows(panels, b.reference.state.mesh)
    qocc = 1 if geometry.occupied_slot_i == geometry.occupied_slot_j else 2
    j = qocc-1
    left = [labels.index((0, qocc+a)) for a in range(geometry.generation_dimension)]
    right = [labels.index((j, qocc+a)) for a in range(geometry.generation_dimension)]
    g = rows[:, left].T @ rows[:, right]
    raw_g = np.einsum("xAa,xAb->ab", panels[:, :, qocc:, 0].conj(), panels[:, :, j, qocc:])
    np.testing.assert_allclose(g, raw_g, atol=5e-12, rtol=3e-10)
    if qocc == 1:
        g = (g+g.T)/2
    f = (fock.real+fock.real.T)/2
    eps = np.diag(f)[qocc:]
    np.testing.assert_allclose(f[qocc:, qocc:], np.diag(eps), atol=1e-10)
    delta = eps[:, None]+eps[None, :]-f[0, 0]-f[j, j]
    t = -g/delta
    u = 2*t-t.T
    density = 2*(t@u.T+t.T@u)
    occupations, vectors = np.linalg.eigh(density)
    return SimpleNamespace(g=g, raw_g=raw_g, f=f, qocc=qocc, t=t, density=density,
        occupations=occupations[::-1], vectors=vectors[:, ::-1], panels=panels, rows=rows, labels=labels)


def _check(seed, b, geometry, expected):
    p, d = seed.memory, seed.diagnostics
    n, m, r = p.common_virtual_dimension, p.generation_dimension, d.retained_dimension
    cutoff = seed.options.pno.pno.occupation_cutoff
    keep = np.ones(m, bool) if cutoff == 0 else expected.occupations > cutoff
    desired = expected.vectors[:, keep]
    c, local, eps, occ, g = (seed.coefficients_copy(), seed.generation_coefficients_copy(),
        seed.energies_copy(), seed.original_pno_occupations_copy(), seed.exchange_integrals_copy())
    assert c.shape == (n, r) and local.shape == (m, r) and g.shape == (r, r)
    np.testing.assert_allclose(occ, expected.occupations, atol=4e-12, rtol=3e-9)
    np.testing.assert_allclose(local@local.T, desired@desired.T, atol=3e-9)
    x = geometry.embedding_coefficients_copy()
    np.testing.assert_allclose(c, x@local, atol=3e-11, rtol=3e-10)
    np.testing.assert_allclose(c@c.T, x@desired@desired.T@x.T, atol=3e-9)
    np.testing.assert_allclose(c.T@c, np.eye(r), atol=3e-10)
    fvv = expected.f[expected.qocc:, expected.qocc:]
    np.testing.assert_allclose(local.T@fvv@local, np.diag(eps), atol=5e-11)
    np.testing.assert_allclose(eps, np.linalg.eigvalsh(desired.T@fvv@desired), atol=5e-11)
    np.testing.assert_allclose(g, local.T@expected.g@local, atol=5e-12, rtol=3e-9)
    assert seed.complete_generation_pair_space == (r == m and r > 0)
    assert seed.complete_common_virtual_space == (r == n and r > 0)
    if seed.diagonal_pair:
        np.testing.assert_array_equal(g, g.T)
        # Periodic Eq.39 has no molecular /(1+delta_ij).
        np.testing.assert_allclose(expected.density, 4*expected.t@expected.t.T, atol=2e-14)
    assert d.retained_output_bytes == 8*(n*r+m*r+r+m+r*r)
    assert d.retained_frame_bytes == 8*(n*r+m*r+r+m)
    assert d.retained_generation_coefficient_bytes == local.nbytes
    assert seed.retained_numerical_bytes == d.retained_output_bytes
    assert seed.geometry_identity_sha256 == geometry.identity_sha256
    assert seed.embedding_identity_sha256 == geometry.embedding_identity_sha256
    assert seed.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256
    assert seed.basis_identity_sha256 == b.basis.identity_sha256
    assert seed.source_context_identity_sha256 == b.context.source_context_identity_sha256
    assert seed.matched_finite_gaussian_hf_recipe
    assert not seed.original_provider_projection_reproduced_bitwise
    assert not seed.coupled_mp2_solution and not seed.production_dlpno and not seed.infinite_source_accuracy_certified


@pytest.fixture(scope="module", params=[("he", 1, False, True), ("he", 2, False, True),
    ("he", 3, False, True), ("he2", 1, False, True), ("he2", 2, False, False), ("he2", 2, True, False)])
def actual(request):
    kind, nk, frozen, full = request.param
    b = _case(nk, kind=kind, frozen=frozen, full=full)
    geometry = _geometry(b)
    controls = _controls(b)
    seed = _make(b, geometry, controls)  # First two-electron consumer, not old rows.
    assert not hasattr(b, "provider")
    local = _local_oracle(b, geometry)
    original = _oracle_inputs(b)
    embedding = _old_embedding(b)
    old = _old_pnos(b, original.provider, embedding, options=controls[1].pno)
    return SimpleNamespace(b=b, geometry=geometry, controls=controls, seed=seed,
        local=local, original=original, old=old, nk=nk, frozen=frozen)


def test_direct_actual_pair_pnos_precede_rows_and_match_original_pao_equations(actual):
    a = actual
    _check(a.seed, a.b, a.geometry, a.local)
    x, o = a.geometry.embedding_coefficients_copy(), a.b.basis.memory.occupied_count
    expected_common = x.T@a.original.eri[0, o:, 0, o:]@x
    np.testing.assert_allclose(a.local.g, expected_common, atol=5e-12, rtol=3e-9)
    np.testing.assert_allclose(a.seed.original_pno_occupations_copy(), a.old.original_pno_occupations_copy(), atol=4e-12, rtol=3e-9)
    c, old_c = a.seed.coefficients_copy(), a.old.coefficients_copy()
    np.testing.assert_allclose(c@c.T, old_c@old_c.T, atol=3e-9)
    np.testing.assert_allclose(a.seed.energies_copy(), a.old.energies_copy(), atol=5e-11)
    assert not hasattr(a.seed, "provider_identity_sha256") and not hasattr(a.seed, "provider")
    if a.frozen:
        assert o == a.nk and a.b.basis.memory.virtual_count == 2*a.nk


def test_exact_live_phase_and_count_inventory(actual):
    a = actual
    p, d = a.seed.memory, a.seed.diagnostics
    assert p.borrowed_geometry_bytes == a.geometry.retained_numerical_bytes
    numeric = max(p.copy_phase_bytes, p.amplitude_phase_bytes, p.density_phase_bytes,
        p.semicanonical_phase_upper_bytes, p.retained_integral_phase_upper_bytes, p.export_phase_upper_bytes)
    assert p.peak_owned_numerical_bytes == max(p.local_basis.peak_owned_numerical_bytes,
        p.gram_phase_upper_bytes, p.local_basis.retained_output_bytes+numeric)
    assert p.required_node_memory_bytes == p.reference_base_node_bytes+p.replicas_per_node*p.worker_bytes
    assert d.retained_output_bytes <= p.retained_output_upper_bytes <= p.peak_owned_numerical_bytes
    assert p.local_occupied_count == 1
    assert d.gram_memory.selection.left.count == d.gram_memory.selection.right.count == p.generation_dimension
    assert d.gram.completed_source_count == p.n_cells
    assert d.gram_memory.work_units <= p.gram_work_units_upper_bound
    assert d.occupied_fock_frobenius_upper_bound <= a.controls[1].maximum_occupied_fock_difference
    for name in ("identity_sha256", "payload_sha256", "local_basis_identity_sha256", "gram_identity_sha256",
                 "gram_payload_sha256", "gram_consumed_sources_identity_sha256", "source_payload_receipt_sha256"):
        assert len(getattr(a.seed, name)) == 64 and int(getattr(a.seed, name), 16) >= 0
    validation = core._plan_periodic_gaussian_gram_pair_pno_payload_validation(a.seed)
    assert validation.numerical_lanes*8 == a.seed.retained_numerical_bytes
    core._verify_periodic_gaussian_gram_pair_pno_payload(a.seed, validation.work_units)
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="work|cap|budget"):
        core._verify_periodic_gaussian_gram_pair_pno_payload(a.seed, validation.work_units-1)


@pytest.fixture(scope="module")
def small():
    b = _case(2)
    return b, _geometry(b)


@pytest.mark.parametrize("field", list(_CAPS))
def test_exact_outer_cap_minus_one_rejects_before_nonfinite_gauges(small, field):
    b, geometry = small
    controls = _controls(b)
    p = _plan(b, geometry, controls)
    setattr(controls[3], field, getattr(p, _CAPS[field]))
    _plan(b, geometry, controls)
    setattr(controls[3], field, getattr(p, _CAPS[field])-1)
    bad = b.gauge.copy()
    bad.flat[0] = complex(np.nan, 0)
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap|budget|bound|dimension|work"):
        _make(b, geometry, controls, gauges=bad)


def test_caller_live_numerical_and_control_roles_are_charged(small):
    b, geometry = small
    controls = _controls(b)
    p = _plan(b, geometry, controls)
    controls[2].other_live_numerical_bytes_per_worker += 1234
    q = _plan(b, geometry, controls)
    assert q.worker_bytes-p.worker_bytes == 1234
    assert q.required_node_memory_bytes-p.required_node_memory_bytes == 1234*p.replicas_per_node
    controls[2].other_live_control_bytes_per_worker += 5678
    r = _plan(b, geometry, controls)
    assert r.control_storage_reservation_bytes-q.control_storage_reservation_bytes == 5678
    assert r.worker_bytes-q.worker_bytes == 5678


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_explicit_occupied_fock_gate_is_required(small, value):
    b, geometry = small
    controls = _controls(b)
    controls[1].maximum_occupied_fock_difference = value
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        _plan(b, geometry, controls)


def test_missing_diagonal_budget_and_zero_backend_fail_closed(small):
    b, geometry = small
    controls = _controls(b)
    controls[1].maximum_retained_diagonal_projection_norm = np.nan
    with pytest.raises((ValueError, RuntimeError)):
        _plan(b, geometry, controls)
    controls = _controls(b)
    controls[2].fixed_backend_margin_bytes_per_worker = 0
    with pytest.raises((ValueError, RuntimeError)):
        _plan(b, geometry, controls)


def test_gauge_descriptor_and_payload_are_real_inputs_not_labels(small):
    b, geometry = small
    for invalid in (b.gauge.astype(np.complex64), b.gauge.reshape(-1), b.gauge.real):
        with pytest.raises(TypeError, match="complex128"):
            _make(b, geometry, gauges=invalid)
    bad = b.gauge.copy()
    bad.flat[0] = np.nan
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="finite|gauge|payload"):
        _make(b, geometry, gauges=bad)
    bad = b.gauge.copy()
    bad.flat[0] = 0
    with pytest.raises((ValueError, RuntimeError), match="gauge|payload|orthonormal|identity"):
        _make(b, geometry, gauges=bad)


def test_valid_sign_flipped_native_wannier_cannot_replace_common_localization(small):
    b, geometry = small
    flipped = np.ascontiguousarray(-b.gauge)
    other_wannier = _wannier(b.reference, flipped)
    assert other_wannier.wannier_identity_sha256 != b.wannier.wannier_identity_sha256
    # Both gauges remain exactly unitary/TR and have identical Foo. The
    # common basis and pair geometry still belong to the original gauge.
    for k in range(b.context.n_kpoints):
        np.testing.assert_array_equal(flipped[k].conj().T@flipped[k], b.gauge[k].conj().T@b.gauge[k])
    with pytest.raises((ValueError, RuntimeError), match="localization|basis|gauge|lineage"):
        _make(b, geometry, wannier=other_wannier, gauges=flipped)


@pytest.mark.parametrize("role", ["ao_basis", "auxiliary_basis"])
def test_changed_gaussian_payload_and_same_size_foreign_geometry_reject(small, role):
    b, geometry = small
    changed = _raw_basis([(0, (0., 0., 0.), [.9], [1.0], False),
        (0, (0., 0., 0.), [.3], [1.0], False)])
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="basis|content|digest|census|source"):
        _make(b, geometry, **{role: changed})


def test_exact_physical_owners_are_required(small):
    b, geometry = small
    foreign = _case(2)
    alien = _geometry(foreign)
    with pytest.raises((ValueError, RuntimeError), match="owner|source|state|geometry|basis|context"):
        _plan(b, alien)
    with pytest.raises((ValueError, RuntimeError), match="owner|source|state|geometry|basis|context"):
        _plan(b, geometry, hf=foreign.hf)


def test_reduced_pair_density_and_positive_cutoff_match_original_pao_equations():
    b = _case(2, kind="he2", full=False)
    geometry = _geometry(b)
    full = _make(b, geometry)
    occ = full.original_pno_occupations_copy()
    assert len(occ) == 2 and occ[0] > occ[1] >= 0
    cutoff = float((occ[0]+occ[1])/2)
    result = _make(b, geometry, _controls(b, cutoff=cutoff))
    oracle = _local_oracle(b, geometry)
    _check(result, b, geometry, oracle)
    assert result.memory.generation_dimension < result.memory.common_virtual_dimension
    assert result.diagnostics.retained_dimension == 1
    assert full.complete_generation_pair_space and not full.complete_common_virtual_space
    # This onsite-image, atom-complete generation space is a decoupled
    # block of the common space. It cannot distinguish late common-density
    # projection from genuine pair generation. The noncommuting witness
    # lives in test_periodic_gaussian_embedded_pair_pnos.py; here the
    # independent original-PAO equations pin both original occupations and
    # the positive-cutoff subspace without constructing an old provider.
    assert not hasattr(b, "provider")


def test_translated_reversed_pair_uses_actual_local_geometry_and_not_forced_symmetry():
    b = _case(2, kind="he2", full=False)
    a, reverse = _geometry(b, 0, 2), _geometry(b, 2, 0)
    seed, back = _make(b, a), _make(b, reverse)
    expected, expected_back = _local_oracle(b, a), _local_oracle(b, reverse)
    _check(seed, b, a, expected)
    _check(back, b, reverse, expected_back)
    assert seed.memory.local_occupied_count == back.memory.local_occupied_count == 2
    assert not seed.diagonal_pair and not back.diagonal_pair
    c, cr = seed.coefficients_copy(), back.coefficients_copy()
    np.testing.assert_allclose(c@seed.exchange_integrals_copy()@c.T,
        (cr@back.exchange_integrals_copy()@cr.T).T, atol=8e-12, rtol=3e-9)
    assert seed.geometry_identity_sha256 != back.geometry_identity_sha256


def test_zero_rank_keeps_original_occupations_and_no_fake_complete_space(small):
    b, geometry = small
    result = _make(b, geometry, _controls(b, cutoff=1.0))
    assert result.diagnostics.retained_dimension == 0
    assert result.exchange_integrals_copy().shape == (0, 0)
    assert result.coefficients_copy().shape == (2, 0)
    assert result.retained_numerical_bytes == 8*geometry.generation_dimension
    assert not result.complete_generation_pair_space and not result.complete_common_virtual_space
    assert np.all(np.isfinite(result.original_pno_occupations_copy()))
    space = _space(b, result)
    caps = _overlap_caps()
    caps.maximum_owned_numerical_bytes = 0  # Truly empty overlap, not empty source occupations.
    overlap = _space_overlap(b, space, space, caps=caps)
    assert overlap.overlaps_copy().shape == (0, 0)
    assert overlap.memory.peak_owned_numerical_bytes == 0
    assert overlap.memory.same_object_borrower
    assert overlap.memory.borrowed_pair_space_bytes == space.memory.retained_output_bytes


def test_auxiliary_and_ao_block_splits_preserve_numerical_subspaces(small):
    b, geometry = small
    a = _make(b, geometry, _controls(b, auxiliary_block=1, pair_block=1))
    c = _make(b, geometry, _controls(b, auxiliary_block=b.auxiliary.nbasis, pair_block=3))
    np.testing.assert_allclose(a.original_pno_occupations_copy(), c.original_pno_occupations_copy(), atol=4e-12, rtol=3e-9)
    ca, cc = a.coefficients_copy(), c.coefficients_copy()
    np.testing.assert_allclose(ca@ca.T, cc@cc.T, atol=3e-9)
    np.testing.assert_allclose(ca@a.exchange_integrals_copy()@ca.T, cc@c.exchange_integrals_copy()@cc.T, atol=5e-12)


def _space(b, seed, *, plan=False, caps=None, options=None):
    fn = core._plan_periodic_gaussian_gram_pair_space_diagnostic if plan else core._make_periodic_gaussian_gram_pair_space_diagnostic
    bounds = _space_caps() if caps is None else caps
    bounds.maximum_integral_calls = 0
    return fn(b.reference, b.basis, seed, _space_options() if options is None else options, _space_live(), bounds)


def test_space_transfers_all_arrays_without_replay_and_moved_seed_rejects(small):
    b, geometry = small
    seed = _make(b, geometry)
    expected = [seed.coefficients_copy(), seed.generation_coefficients_copy(), seed.energies_copy(),
        seed.original_pno_occupations_copy(), seed.exchange_integrals_copy()]
    identity, payload = seed.identity_sha256, seed.payload_sha256
    plan = _space(b, seed, plan=True)
    assert plan.construction_owned_bytes == plan.integral_calls == plan.provider_work_units == plan.borrowed_provider_row_bytes == 0
    assert plan.borrowed_pno_bytes == plan.retained_output_bytes == plan.peak_owned_numerical_bytes == sum(x.nbytes for x in expected)
    result = _space(b, seed)
    assert result.direct_gram_generation and not result.embedded_generation
    assert result.pno_identity_sha256 == identity and result.pno_payload_sha256 == payload
    actual = [result.coefficients_copy(), core._periodic_gaussian_pair_space_direct_gram_generation_coefficients_copy(result),
        result.energies_copy(), result.original_pno_occupations_copy(), result.exchange_integrals_copy()]
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, e)
    assert not result.diagnostics.source_integrals_replayed and result.diagnostics.completed_integral_calls == 0
    for name in ("coefficients_copy", "generation_coefficients_copy", "exchange_integrals_copy", "energies_copy"):
        with pytest.raises((ValueError, RuntimeError), match="moved|live|consumed|owner"):
            getattr(seed, name)()
    with pytest.raises((ValueError, RuntimeError), match="moved|live|consumed|owner"):
        _space(b, seed)
    assert not hasattr(result, "direct_gram_pnos")  # No stealable typed child.


@pytest.mark.parametrize("cap,field", [("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_per_worker_inventoried_bytes", "per_worker_inventoried_bytes"),
    ("maximum_node_inventoried_bytes", "required_node_memory_bytes"), ("maximum_work_units", "work_units")])
def test_space_failed_admission_preserves_seed(small, cap, field):
    b, geometry = small
    seed = _make(b, geometry)
    p = _space(b, seed, plan=True)
    bounds = _space_caps()
    setattr(bounds, cap, getattr(p, field)-1)
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="cap|budget|work|bound"):
        _space(b, seed, caps=bounds)
    assert seed.coefficients_copy().shape[1] == seed.diagnostics.retained_dimension
    result = _space(b, seed)
    assert result.memory.integral_calls == 0


def test_direct_spaces_overlap_with_different_gram_receipts_and_reject_old_source_kinds(small):
    b, geometry = small
    a = _space(b, _make(b, geometry))
    c = _space(b, _make(b, _geometry(b, 0, 1)))
    overlap = _space_overlap(b, a, c)
    np.testing.assert_allclose(overlap.overlaps_copy(), a.coefficients_copy().T@c.coefficients_copy(), atol=3e-12)
    for name in ("provider_identity_sha256", "common_source_exchange_integral_identity_sha256", "embedded_pno_diagnostics"):
        with pytest.raises((ValueError, RuntimeError), match="source|direct|Gram|legacy|embedded|kind"):
            getattr(a, name)
    original = _oracle_inputs(b)
    embedding = _old_embedding(b)
    old = _old_pnos(b, original.provider, embedding)
    from tests.test_periodic_gaussian_pair_space import _make as old_space
    legacy = old_space(b, original.provider, old)
    with pytest.raises((ValueError, RuntimeError), match="source|kind|provider|Gram"):
        _space_overlap(b, a, legacy)
    # The old mixed-factor factory must refuse DirectPAOGram frames before
    # interpreting absent common-provider/common-G receipts.
    from tests.test_periodic_gaussian_mixed_pair_factors import _plan as mixed_plan, _controls as mixed_controls
    source, _, w = _q_data(b, 0)
    mixed_case = SimpleNamespace(b=b, a=a, source=c, i=0, k=1)
    with pytest.raises((ValueError, RuntimeError), match="source|kind|provider|Gram|direct"):
        mixed_plan(mixed_case, source, w, controls=mixed_controls(b))


def test_owned_output_and_copied_controls_outlive_all_input_geometry():
    b = _case(1)
    geometry, controls = _geometry(b), _controls(b)
    result = _make(b, geometry, controls)
    c, g, e, occ = result.coefficients_copy(), result.exchange_integrals_copy(), result.energies_copy(), result.original_pno_occupations_copy()
    original_cutoff = result.options.pno.pno.occupation_cutoff
    controls[1].pno.pno.occupation_cutoff = 1
    assert result.options.pno.pno.occupation_cutoff == original_cutoff
    for value in (c, g, e, occ):
        assert not value.flags.writeable
    del geometry, b, controls
    gc.collect()
    np.testing.assert_array_equal(result.coefficients_copy(), c)
    np.testing.assert_array_equal(result.exchange_integrals_copy(), g)
    assert result.state.n_kpoints == 1 and result.context.n_kpoints == 1
