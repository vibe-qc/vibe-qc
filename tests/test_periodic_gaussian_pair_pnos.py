"""Actual-source periodic Eq.39 PNOs, not a coupled-MP2 energy driver.

Tiny normalized He finite Hamiltonians only. The independent density uses
Nejad2025 Eqs.37-40 directly, not the implementation's S/A decomposition.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_selected_local_ccsd_t import (
    _bundle, _prepare_leaves, _dense_case, _he2_bundle, _he2_controls,
)
from tests.test_periodic_gaussian_real_local_provider import _make as _provider
from tests.test_pair_natural_orbitals import _build as _legacy_pnos, _mp2_pair
from tests.test_periodic_correlation_reciprocal_metric import _CanonicalDigest


_CAP_TO_PLAN = {
    "maximum_owned_numerical_bytes": "peak_owned_numerical_bytes",
    "maximum_per_worker_inventoried_bytes": "per_worker_inventoried_bytes",
    "maximum_node_inventoried_bytes": "required_node_memory_bytes",
    "maximum_integral_calls": "integral_calls", "maximum_work_units": "work_units",
}
_PLAN_FIELDS = (
    "occupied_count", "virtual_count", "occupied_slot_i", "occupied_slot_j", "integral_calls",
    "provider_work_units", "numerical_work_units", "work_units", "integral_amplitude_phase_bytes",
    "density_phase_bytes", "semicanonical_phase_upper_bytes", "peak_owned_numerical_bytes",
    "retained_output_upper_bytes", "borrowed_basis_bytes", "borrowed_provider_row_bytes", "state_resident_bytes",
    "fixed_control_storage_bytes", "replicas_per_node", "reference_base_node_bytes",
    "per_worker_inventoried_bytes", "required_node_memory_bytes",
)
_POLICY = ("Nejad2025-Eq39;D=4SS^T+12AA^T;all-pairs;no-delta-denominator;initial-SC-only;"
           "original-Fvv-recanonicalization;no-energy-multiplicity")


def _options(**changes):
    o = core._PeriodicGaussianPairPNOOptions()
    o.occupation_cutoff = 0.0
    o.denominator_floor = 1e-8
    o.maximum_initial_fvv_offdiagonal_norm = 1e-10
    o.semicanonical_orthonormality_tolerance = 1e-10
    o.pno_max_sweeps = o.semicanonical_max_sweeps = 80
    o.pno_relative_eigensolver_tolerance = o.semicanonical_relative_eigensolver_tolerance = 1e-14
    for field, value in changes.items():
        setattr(o, field, value)
    return o


def _live():
    live = core._PeriodicGaussianPairPNOLiveInventory()
    # Conservatively cover the still-live tiny Gaussian/localization/PAO
    # fixture owners, gauge copy and independent diagnostic buffers. Exact
    # basis F/labels, provider rows and shared HF state are counted natively.
    live.other_live_bytes_per_worker = 2**20
    live.fixed_backend_margin_bytes_per_worker = 65536
    return live


def _caps():
    c = core._PeriodicGaussianPairPNOCaps()
    for field, value in zip(_CAP_TO_PLAN, (65536, 2**24, 2**26, 1024, 10**12)):
        setattr(c, field, value)
    return c


def _plan(b, provider, i=0, j=0, *, options=None, live=None, caps=None):
    return core._plan_periodic_gaussian_pair_pnos(b.reference, b.basis, provider, i, j,
        _options() if options is None else options, _live() if live is None else live,
        _caps() if caps is None else caps)


def _make(b, provider, i=0, j=0, *, options=None, live=None, caps=None):
    return core._make_periodic_gaussian_pair_pnos(b.reference, b.basis, provider, i, j,
        _options() if options is None else options, _live() if live is None else live,
        _caps() if caps is None else caps)


@pytest.fixture(scope="module", params=[("he", 1), ("he", 2), ("he", 3), ("he2", 1), ("he2", 2)])
def physical_case(request):
    system, nk = request.param
    mesh = (nk, 1, 1)
    if system == "he2":
        b = _he2_bundle(mesh)
        _prepare_leaves(b, localization_options=_he2_controls(b)[0].localization)
    else:
        b = _prepare_leaves(_bundle(mesh))
    return b, _provider(b), _dense_case(b)


def _oracle(b, dense, i, j):
    o, v = dense["o"], dense["v"]
    fock = b.basis.fock_copy()
    fvv = fock[o:, o:]
    eps = np.diag(fvv)
    delta = eps[:, None]+eps[None, :]-fock[i, i]-fock[j, j]
    g = dense["eri"][i, o:, j, o:]
    t = -g/delta
    u = 2*t-t.T
    density = 2*(t@u.T+t.T@u)
    values, coefficients = np.linalg.eigh(density)
    return dict(g=g, t=t, density=density, occupations=values[::-1], coefficients=coefficients[:, ::-1],
                delta=delta, fvv=fvv, fock=fock, o=o, v=v)


def _check_selected_subspace(result, oracle, cutoff):
    occupations = oracle["occupations"]
    selected = np.ones(len(occupations), bool) if cutoff == 0 else occupations > cutoff
    c = oracle["coefficients"][:, selected]
    actual = result.coefficients_copy()
    assert actual.shape == c.shape
    np.testing.assert_allclose(actual.T@actual, np.eye(actual.shape[1]), atol=3e-11)
    np.testing.assert_allclose(actual@actual.T, c@c.T, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(actual.T@oracle["fvv"]@actual,
        np.diag(result.energies_copy()), atol=3e-11, rtol=3e-11)
    expected_energy = np.linalg.eigvalsh(c.T@oracle["fvv"]@c)
    np.testing.assert_allclose(result.energies_copy(), expected_energy, atol=3e-11, rtol=3e-11)


def test_periodic_diagonal_density_matches_eq34_and_eq39_not_molecular_half(physical_case):
    b, provider, dense = physical_case
    result = _make(b, provider)
    oracle = _oracle(b, dense, 0, 0)
    np.testing.assert_allclose(oracle["t"], oracle["t"].T, atol=3e-13)
    np.testing.assert_allclose(oracle["density"], 4*oracle["t"]@oracle["t"].T, atol=3e-13)
    np.testing.assert_allclose(result.original_pno_occupations_copy(), oracle["occupations"], atol=3e-12, rtol=2e-9)
    molecular = _legacy_pnos(np.ascontiguousarray((oracle["t"]+oracle["t"].T)/2), diagonal=True, cutoff=0.0)
    np.testing.assert_allclose(result.original_pno_occupations_copy(),
        2*molecular.occupations_copy(), atol=3e-12, rtol=2e-9)
    assert result.diagonal_pair and result.occupied_i == result.occupied_j == (0, 0)
    assert result.diagnostics.maximum_diagonal_integral_asymmetry == 0
    assert result.diagnostics.maximum_diagonal_amplitude_asymmetry == 0
    assert result.diagnostics.density_trace == pytest.approx(np.trace(oracle["density"]), abs=3e-12)
    assert result.diagnostics.minimum_denominator == pytest.approx(oracle["delta"].min(), abs=3e-12)
    assert result.diagnostics.maximum_denominator == pytest.approx(oracle["delta"].max(), abs=3e-12)
    _check_selected_subspace(result, oracle, 0.0)


def test_actual_offdiagonal_pair_uses_unweighted_periodic_density_and_transpose_covariance(physical_case):
    b, provider, dense = physical_case
    if dense["o"] == 1:
        return
    direct, reversed_pair = _make(b, provider, 0, 1), _make(b, provider, 1, 0)
    oracle = _oracle(b, dense, 0, 1)
    np.testing.assert_allclose(direct.original_pno_occupations_copy(), oracle["occupations"], atol=4e-12, rtol=2e-9)
    np.testing.assert_allclose(direct.original_pno_occupations_copy(), reversed_pair.original_pno_occupations_copy(),
        atol=4e-13, rtol=3e-11)
    np.testing.assert_allclose(direct.coefficients_copy()@direct.coefficients_copy().T,
        reversed_pair.coefficients_copy()@reversed_pair.coefficients_copy().T, atol=4e-12)
    assert not direct.diagonal_pair
    assert direct.occupied_i == tuple(b.rows[0]) and direct.occupied_j == tuple(b.rows[1])
    assert direct.identity_sha256 != reversed_pair.identity_sha256
    _check_selected_subspace(direct, oracle, 0.0)


def test_explicit_positive_truncation_and_empty_rank_have_no_forced_orbitals(physical_case):
    b, provider, dense = physical_case
    oracle = _oracle(b, dense, 0, 0)
    # A data-chosen explicit cutoff tests scientific truncation, not a preset.
    cutoff = 0.5*oracle["occupations"][0]
    assert cutoff > 0
    result = _make(b, provider, options=_options(occupation_cutoff=cutoff))
    _check_selected_subspace(result, oracle, cutoff)
    assert result.diagnostics.discarded_occupation_sum == pytest.approx(
        oracle["occupations"][oracle["occupations"] <= cutoff].sum(), abs=4e-12)
    empty = _make(b, provider, options=_options(occupation_cutoff=2*oracle["occupations"][0]))
    assert empty.diagnostics.retained_dimension == 0
    assert empty.coefficients_copy().shape == (dense["v"], 0)
    assert empty.energies_copy().shape == (0,)
    assert empty.diagnostics.retained_output_bytes == 8*dense["v"]
    assert empty.semicanonical_generation_approximation and not empty.coupled_mp2_solution
    assert not empty.production_dlpno and not empty.infinite_source_accuracy_certified
    assert not hasattr(empty, "correlation_energy") and not hasattr(empty, "ordered_pair_energy")


def _option_wire(h, options):
    for field in ("occupation_cutoff", "denominator_floor", "maximum_initial_fvv_offdiagonal_norm",
                  "semicanonical_orthonormality_tolerance"):
        h.binary64(getattr(options, field))
    for prefix in ("pno", "semicanonical"):
        h.u64(getattr(options, prefix+"_max_sweeps"))
        h.binary64(getattr(options, prefix+"_relative_eigensolver_tolerance"))


def _plan_wire(b, provider, p):
    h = _CanonicalDigest("vibeqc.periodic.gaussian-pair-pnos.plan", 1)
    for value in (b.reference.state.state_identity_sha256, b.reference.dimensions.allocation_identity,
                  b.basis.identity_sha256, provider.identity_sha256, provider.hf_reference_source_identity_sha256):
        h.string(value)
    for field in _PLAN_FIELDS:
        h.u64(getattr(p, field))
    _option_wire(h, p.options)
    h.u64(p.live.other_live_bytes_per_worker)
    h.u64(p.live.fixed_backend_margin_bytes_per_worker)
    for field in _CAP_TO_PLAN:
        h.u64(getattr(p.caps, field))
    h.string(_POLICY)
    return h.finish()


def _array_wire(role, array, n):
    h = _CanonicalDigest("vibeqc.periodic.gaussian-pair-pnos."+role, 1)
    h.u64(n)
    h.u64(array.size)
    for x in array.ravel():
        h.binary64(x)
    return h.finish()


def test_exact_phase_inventory_seals_consumed_leaf_payloads_and_state_once(physical_case):
    b, provider, dense = physical_case
    p, result = _plan(b, provider), _make(b, provider)
    n, o = dense["v"], dense["o"]
    assert p.integral_calls == n*n == result.diagnostics.completed_integral_calls
    assert p.provider_work_units == n*n*provider.memory.scalar_work_units
    assert p.integral_amplitude_phase_bytes == 16*n*n+8*n
    assert p.density_phase_bytes == p.semicanonical_phase_upper_bytes == p.peak_owned_numerical_bytes == 48*n*n+16*n
    assert p.retained_output_upper_bytes == 8*n*n+16*n
    assert p.borrowed_basis_bytes == b.basis.memory.retained_output_bytes
    assert p.borrowed_provider_row_bytes == provider.memory.retained_row_bytes
    assert p.state_resident_bytes == b.reference.state_resident_bytes
    assert p.per_worker_inventoried_bytes == (p.peak_owned_numerical_bytes+p.borrowed_basis_bytes
        +p.borrowed_provider_row_bytes+p.fixed_control_storage_bytes+p.live.other_live_bytes_per_worker
        +p.live.fixed_backend_margin_bytes_per_worker)
    assert p.required_node_memory_bytes == p.reference_base_node_bytes+p.replicas_per_node*p.per_worker_inventoried_bytes
    assert result.diagnostics.actual_semicanonical_phase_bytes <= p.semicanonical_phase_upper_bytes
    assert p.plan_identity_sha256 == _plan_wire(b, provider, p)
    actual_g = np.array([[provider.provider.integral(0, o+a, 0, o+c).value for c in range(n)] for a in range(n)])
    f = b.basis.fock_copy()
    t = _mp2_pair(actual_g, np.diag(f[o:, o:]).copy(), f[0, 0], f[0, 0]).amplitudes_copy()
    density = _legacy_pnos(t, diagonal=False, cutoff=0.0, sweeps=80, tolerance=1e-14).density_copy()
    assert result.exchange_integral_identity_sha256 == _array_wire("integrals", actual_g, n)
    assert result.initial_amplitude_identity_sha256 == _array_wire("amplitudes", t, n)
    assert result.density_identity_sha256 == _array_wire("density", density, n)
    h = _CanonicalDigest("vibeqc.periodic.gaussian-pair-pnos.payload", 1)
    h.u64(n)
    h.u64(result.diagnostics.retained_dimension)
    for array in (result.original_pno_occupations_copy(), result.coefficients_copy(), result.energies_copy()):
        for value in array.ravel():
            h.binary64(value)
    assert result.payload_sha256 == h.finish()
    assert result.basis_identity_sha256 == b.basis.identity_sha256
    assert result.provider_identity_sha256 == provider.identity_sha256
    assert result.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256


@pytest.mark.parametrize("field", _CAP_TO_PLAN)
def test_exact_caps_and_one_below_every_admission(physical_case, field):
    b, provider, _ = physical_case
    p = _plan(b, provider)
    caps = _caps()
    for cap, name in _CAP_TO_PLAN.items():
        setattr(caps, cap, getattr(p, name))
    _make(b, provider, caps=caps)
    setattr(caps, field, getattr(caps, field)-1)
    with pytest.raises(ValueError, match="cap"):
        _make(b, provider, caps=caps)


@pytest.mark.parametrize("field,value", [
    ("occupation_cutoff", np.nan), ("occupation_cutoff", -1.0), ("occupation_cutoff", np.inf),
    ("denominator_floor", 0.0), ("denominator_floor", np.nan),
    ("maximum_initial_fvv_offdiagonal_norm", np.nan), ("maximum_initial_fvv_offdiagonal_norm", -1.0),
    ("semicanonical_orthonormality_tolerance", 0.0), ("semicanonical_orthonormality_tolerance", 1.0),
    ("pno_max_sweeps", 0), ("pno_relative_eigensolver_tolerance", 0.0),
    ("semicanonical_max_sweeps", 0), ("semicanonical_relative_eigensolver_tolerance", np.inf),
])
def test_missing_or_invalid_scientific_controls_fail_closed(field, value):
    b = _prepare_leaves(_bundle())
    provider = _provider(b)
    with pytest.raises(ValueError, match="controls"):
        _plan(b, provider, options=_options(**{field: value}))
    assert np.isnan(core._PeriodicGaussianPairPNOOptions().occupation_cutoff)
    assert np.isnan(core._PeriodicGaussianPairPNOOptions().maximum_initial_fvv_offdiagonal_norm)


def test_denominator_floor_and_explicit_fvv_projection_budget_are_not_shifts(physical_case):
    b, provider, dense = physical_case
    result = _make(b, provider)
    with pytest.raises(ValueError, match="strictly exceed"):
        _make(b, provider, options=_options(denominator_floor=result.diagnostics.maximum_denominator))
    norm = result.diagnostics.initial_fvv_offdiagonal_norm_upper_bound
    if norm > 0:
        with pytest.raises(ValueError, match="projection.*budget"):
            _make(b, provider, options=_options(maximum_initial_fvv_offdiagonal_norm=np.nextafter(norm, 0)))
    assert norm <= 1e-10
    assert result.diagnostics.maximum_initial_residual < 1e-12


def test_foreign_same_payload_owner_wrong_slots_and_no_array_constructor():
    b, other = _prepare_leaves(_bundle()), _prepare_leaves(_bundle())
    provider, foreign = _provider(b), _provider(other)
    assert b.reference.state.state_identity_sha256 == other.reference.state.state_identity_sha256
    with pytest.raises(ValueError, match="owners|receipts"):
        _plan(b, foreign)
    with pytest.raises(IndexError, match="occupied slot"):
        _plan(b, provider, 1, 0)
    with pytest.raises(TypeError):
        core._PeriodicGaussianPairPNOResult()


def test_result_copies_lifetime_and_explicit_other_live_storage():
    b = _prepare_leaves(_bundle())
    provider = _provider(b)
    p = _plan(b, provider)
    live = _live()
    live.other_live_bytes_per_worker += 777
    expanded = _plan(b, provider, live=live)
    assert expanded.per_worker_inventoried_bytes-p.per_worker_inventoried_bytes == 777
    assert expanded.required_node_memory_bytes-p.required_node_memory_bytes == p.replicas_per_node*777
    result = _make(b, provider)
    c, eps, occupations = result.coefficients_copy(), result.energies_copy(), result.original_pno_occupations_copy()
    result.coefficients_copy()[:] = 9
    result.energies_copy()[:] = 9
    result.original_pno_occupations_copy()[:] = 9
    options = result.memory.options
    options.occupation_cutoff = 9
    del b, provider
    gc.collect()
    np.testing.assert_array_equal(result.coefficients_copy(), c)
    np.testing.assert_array_equal(result.energies_copy(), eps)
    np.testing.assert_array_equal(result.original_pno_occupations_copy(), occupations)
    assert result.memory.options.occupation_cutoff == 0
    assert result.state.n_kpoints == 1 and result.matched_finite_gaussian_hf_recipe
