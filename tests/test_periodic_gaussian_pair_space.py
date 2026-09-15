"""Actual Gaussian pair projection and certified common-basis overlaps.

Native finite-source He/He2 fixtures only. Independent dense contractions
start from actual AO factors, not supplied G/C arrays accepted by the API.
No coupled-MP2 or production DLPNO energy claim.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_pair_pnos import (
    physical_case, _make as _pnos, _options as _pno_options,
    _bundle, _prepare_leaves, _provider, _CanonicalDigest,
)


_CAP_TO_PLAN = {
    "maximum_owned_numerical_bytes": "peak_owned_numerical_bytes",
    "maximum_per_worker_inventoried_bytes": "per_worker_inventoried_bytes",
    "maximum_node_inventoried_bytes": "required_node_memory_bytes",
    "maximum_integral_calls": "integral_calls", "maximum_work_units": "work_units",
}
_STORAGE_FIELDS = (
    "virtual_count", "retained_dimension", "integral_calls", "construction_owned_bytes", "borrowed_pno_bytes",
    "retained_output_bytes", "peak_owned_numerical_bytes", "construction_live_numerical_bytes",
    "numerical_work_units", "fixed_control_storage_bytes",
)
_PLAN_FIELDS = (
    "occupied_count", "occupied_slot_i", "occupied_slot_j", "provider_work_units", "work_units",
    "borrowed_basis_bytes", "borrowed_provider_row_bytes", "state_resident_bytes", "replicas_per_node",
    "reference_base_node_bytes", "per_worker_inventoried_bytes", "required_node_memory_bytes",
)
_POLICY = ("Nejad2025-Eqs37-44;common-PAO;Gprime=C^TGC;row-streamed;PNO-owner-transfer;"
           "explicit-diagonal-roundoff-projection;no-energy-multiplicity")


def _options(budget=1e-10):
    o = core._PeriodicGaussianPairSpaceOptions()
    o.maximum_diagonal_symmetry_projection_error = budget
    return o


def _live():
    live = core._PeriodicGaussianPairSpaceLiveInventory()
    live.other_live_bytes_per_worker = 2**20
    live.fixed_backend_margin_bytes_per_worker = 65536
    return live


def _caps():
    c = core._PeriodicGaussianPairSpaceCaps()
    for field, value in zip(_CAP_TO_PLAN, (65536, 2**24, 2**26, 1024, 10**12)):
        setattr(c, field, value)
    return c


def _plan(b, provider, pno, *, options=None, live=None, caps=None):
    return core._plan_periodic_gaussian_pair_space(b.reference, b.basis, provider, pno,
        _options() if options is None else options, _live() if live is None else live,
        _caps() if caps is None else caps)


def _make(b, provider, pno, *, options=None, live=None, caps=None):
    return core._make_periodic_gaussian_pair_space(b.reference, b.basis, provider, pno,
        _options() if options is None else options, _live() if live is None else live,
        _caps() if caps is None else caps)


def _overlap_caps():
    c = core._PeriodicGaussianPairOverlapCaps()
    for field, value in zip((f for f in _CAP_TO_PLAN if f != "maximum_integral_calls"),
                           (65536, 2**24, 2**26, 10**12)):
        setattr(c, field, value)
    return c


def _overlap(b, target, source, *, caps=None, plan=False):
    function = core._plan_periodic_gaussian_pair_overlap if plan else core._make_periodic_gaussian_pair_overlap
    return function(b.reference, target, source, _live(), _overlap_caps() if caps is None else caps)


def _integrals_wire(values, n, r):
    h = _CanonicalDigest("vibeqc.periodic.gaussian-pair-space.integrals", 1)
    for value in (n, r, values.size):
        h.u64(value)
    for value in values.ravel():
        h.binary64(value)
    return h.finish()


def _accumulate(value, total, correction):
    following = total+value
    tail = (total-following)+value if abs(total) >= abs(value) else (value-following)+total
    return following, correction+tail


def _row_order_oracle(provider, coefficients, o, i, j):
    # Separately spell out the bounded two-stage summation for raw-wire
    # reproducibility; the numerical oracle below also uses dense NumPy.
    n, r = coefficients.shape
    raw, compensation = np.zeros((r, r)), np.zeros((r, r))
    for a in range(n):
        row, correction = [0.0]*r, [0.0]*r
        for b in range(n):
            value = provider.provider.integral(i, o+a, j, o+b).value
            for beta in range(r):
                row[beta], correction[beta] = _accumulate(value*float(coefficients[b, beta]), row[beta], correction[beta])
        row = [x+y for x, y in zip(row, correction)]
        for alpha in range(r):
            for beta in range(r):
                raw[alpha, beta], compensation[alpha, beta] = _accumulate(
                    float(coefficients[a, alpha])*row[beta], float(raw[alpha, beta]), float(compensation[alpha, beta]))
    return raw+compensation


def test_actual_projected_integrals_and_strong_owner_transfer(physical_case):
    b, provider, dense = physical_case
    o, n = dense["o"], dense["v"]
    for i, j in [(0, 0)]+([(0, 1), (1, 0)] if o > 1 else []):
        pno = _pnos(b, provider, i, j)
        c, eps, occupations = pno.coefficients_copy(), pno.energies_copy(), pno.original_pno_occupations_copy()
        pno_identity, pno_payload = pno.identity_sha256, pno.payload_sha256
        expected = c.T@dense["eri"][i, o:, j, o:]@c
        raw = _row_order_oracle(provider, c, o, i, j)
        result = _make(b, provider, pno)
        np.testing.assert_allclose(result.exchange_integrals_copy(), expected, atol=6e-12, rtol=3e-11)
        np.testing.assert_array_equal(result.coefficients_copy(), c)
        np.testing.assert_array_equal(result.energies_copy(), eps)
        np.testing.assert_array_equal(result.original_pno_occupations_copy(), occupations)
        assert result.raw_exchange_integral_identity_sha256 == _integrals_wire(raw, n, c.shape[1])
        assert result.exchange_integral_identity_sha256 == _integrals_wire(result.exchange_integrals_copy(), n, c.shape[1])
        assert result.pno_identity_sha256 == pno_identity and result.pno_payload_sha256 == pno_payload
        assert result.occupied_i == tuple(b.rows[i]) and result.occupied_j == tuple(b.rows[j])
        assert result.diagnostics.completed_integral_calls == n*n
        assert result.diagnostics.source_integrals_replayed
        assert not result.coupled_mp2_solution and not result.production_dlpno
        assert not result.infinite_source_accuracy_certified
        assert not hasattr(result, "pnos")  # No mutable borrowed PNO escape through pybind const erasure.
        with pytest.raises(RuntimeError, match="moved|malformed"):
            pno.coefficients_copy()
        with pytest.raises(RuntimeError, match="moved|malformed"):
            _make(b, provider, pno)
        if i == j:
            np.testing.assert_array_equal(result.exchange_integrals_copy(), result.exchange_integrals_copy().T)
            defect = np.linalg.norm(raw-result.exchange_integrals_copy())
            assert defect <= result.diagnostics.diagonal_symmetry_projection_frobenius_bound+1e-30
        else:
            np.testing.assert_array_equal(result.exchange_integrals_copy(), raw)
            assert not result.diagnostics.diagonal_symmetry_projection_applied
            assert result.diagnostics.diagonal_symmetry_projection_frobenius_bound == 0


def test_projection_budget_failure_does_not_consume_pno(physical_case):
    b, provider, _ = physical_case
    accepted = _make(b, provider, _pnos(b, provider))
    bound = accepted.diagnostics.diagonal_symmetry_projection_frobenius_bound
    pno = _pnos(b, provider)
    c = pno.coefficients_copy()
    if bound > 0:
        with pytest.raises(ValueError, match="projection.*budget"):
            _make(b, provider, pno, options=_options(np.nextafter(bound, 0)))
        np.testing.assert_array_equal(pno.coefficients_copy(), c)
    exact = _make(b, provider, pno, options=_options(bound))
    assert exact.diagnostics.diagonal_symmetry_projection_frobenius_bound == bound


def test_truncated_and_empty_pair_spaces_have_exact_variable_storage(physical_case):
    b, provider, dense = physical_case
    original = _pnos(b, provider)
    largest = original.original_pno_occupations_copy()[0]
    for cutoff in (0.5*largest, 2*largest):
        pno = _pnos(b, provider, options=_pno_options(occupation_cutoff=cutoff))
        c = pno.coefficients_copy()
        result = _make(b, provider, pno)
        n, r = c.shape
        np.testing.assert_allclose(result.exchange_integrals_copy(),
            c.T@dense["eri"][0, dense["o"]:, 0, dense["o"]:]@c, atol=6e-12, rtol=3e-11)
        assert result.memory.borrowed_pno_bytes == 8*n*r+8*r+8*n
        assert result.memory.retained_output_bytes == 8*n*r+8*r+8*n+8*r*r
        assert result.memory.construction_owned_bytes == 16*r*r+16*r
        if r == 0:
            assert result.exchange_integrals_copy().shape == (0, 0)
            assert result.memory.integral_calls == result.diagnostics.completed_integral_calls == 0
            assert not result.diagnostics.source_integrals_replayed
            assert result.diagnostics.diagonal_symmetry_projection_frobenius_bound == 0
            assert result.memory.peak_owned_numerical_bytes == 8*n
    assert r == 0


def test_storage_formula_and_plan_wire(physical_case):
    b, provider, dense = physical_case
    pno = _pnos(b, provider)
    p = _plan(b, provider, pno)
    n, r = p.virtual_count, p.retained_dimension
    upper = core._periodic_gaussian_pair_space_storage(n, n)
    for field in _STORAGE_FIELDS:
        assert getattr(p, field) <= getattr(upper, field)
    assert p.construction_owned_bytes == 16*r*r+16*r
    assert p.borrowed_pno_bytes == 8*n*r+8*r+8*n
    assert p.construction_live_numerical_bytes == p.borrowed_pno_bytes+p.construction_owned_bytes
    assert p.peak_owned_numerical_bytes == max(p.construction_owned_bytes, p.retained_output_bytes)
    assert p.provider_work_units == p.integral_calls*provider.memory.scalar_work_units
    assert p.per_worker_inventoried_bytes == (p.construction_live_numerical_bytes+p.borrowed_basis_bytes
        +p.borrowed_provider_row_bytes+p.fixed_control_storage_bytes+p.live.other_live_bytes_per_worker
        +p.live.fixed_backend_margin_bytes_per_worker)
    assert p.required_node_memory_bytes == p.reference_base_node_bytes+p.replicas_per_node*p.per_worker_inventoried_bytes
    h = _CanonicalDigest("vibeqc.periodic.gaussian-pair-space.plan", 1)
    for value in (b.reference.state.state_identity_sha256, b.reference.dimensions.allocation_identity,
                  b.basis.identity_sha256, provider.identity_sha256, provider.hf_reference_source_identity_sha256,
                  pno.identity_sha256):
        h.string(value)
    for field in _STORAGE_FIELDS+_PLAN_FIELDS:
        h.u64(getattr(p, field))
    h.binary64(p.options.maximum_diagonal_symmetry_projection_error)
    h.u64(p.live.other_live_bytes_per_worker)
    h.u64(p.live.fixed_backend_margin_bytes_per_worker)
    for field in _CAP_TO_PLAN:
        h.u64(getattr(p.caps, field))
    h.string(_POLICY)
    assert p.plan_identity_sha256 == h.finish()
    result = _make(b, provider, pno)
    payload = _CanonicalDigest("vibeqc.periodic.gaussian-pair-space.payload", 1)
    payload.string(result.pno_payload_sha256)
    payload.string(result.exchange_integral_identity_sha256)
    assert result.payload_sha256 == payload.finish()


@pytest.mark.parametrize("field", _CAP_TO_PLAN)
def test_exact_caps_and_rejected_build_preserve_input(physical_case, field):
    b, provider, _ = physical_case
    pno = _pnos(b, provider)
    before = pno.coefficients_copy()
    p = _plan(b, provider, pno)
    c = _caps()
    for cap, name in _CAP_TO_PLAN.items():
        setattr(c, cap, getattr(p, name))
    setattr(c, field, getattr(c, field)-1)
    with pytest.raises(ValueError, match="cap"):
        _make(b, provider, pno, caps=c)
    np.testing.assert_array_equal(pno.coefficients_copy(), before)
    setattr(c, field, getattr(p, _CAP_TO_PLAN[field]))
    _make(b, provider, pno, caps=c)


@pytest.mark.parametrize("budget", [np.nan, np.inf, -1.0])
def test_projection_choice_is_explicit_and_invalid_controls_preserve_input(physical_case, budget):
    b, provider, _ = physical_case
    pno = _pnos(b, provider)
    with pytest.raises(ValueError, match="explicit.*finite"):
        _plan(b, provider, pno, options=_options(budget))
    assert pno.coefficients_copy().size > 0
    assert np.isnan(core._PeriodicGaussianPairSpaceOptions().maximum_diagonal_symmetry_projection_error)


def test_overlap_direct_numpy_ragged_ranks_transpose_and_same_object_inventory(physical_case):
    b, provider, dense = physical_case
    first = _make(b, provider, _pnos(b, provider))
    largest = first.original_pno_occupations_copy()[0]
    pno = _pnos(b, provider, 0, min(1, dense["o"]-1), options=_pno_options(occupation_cutoff=0.5*largest))
    second = _make(b, provider, pno)
    for a, other in [(first, first), (first, second), (second, first)]:
        overlap = _overlap(b, a, other)
        np.testing.assert_allclose(overlap.overlaps_copy(), a.coefficients_copy().T@other.coefficients_copy(), atol=5e-14)
        p = overlap.memory
        assert p.peak_owned_numerical_bytes == 8*p.target_dimension*p.source_dimension
        assert p.same_object_borrower == (a is other)
        assert p.borrowed_pair_space_bytes == a.memory.retained_output_bytes+(0 if a is other else other.memory.retained_output_bytes)
        assert overlap.target_identity_sha256 == a.identity_sha256
        assert overlap.source_identity_sha256 == other.identity_sha256
        h = _CanonicalDigest("vibeqc.periodic.gaussian-pair-overlap.payload", 1)
        for value in (p.virtual_count, p.target_dimension, p.source_dimension):
            h.u64(value)
        for value in overlap.overlaps_copy().ravel():
            h.binary64(value)
        assert overlap.payload_sha256 == h.finish()
    np.testing.assert_array_equal(_overlap(b, first, second).overlaps_copy(),
                                  _overlap(b, second, first).overlaps_copy().T)
    empty = _make(b, provider, _pnos(b, provider, options=_pno_options(occupation_cutoff=2*largest)))
    assert _overlap(b, first, empty).overlaps_copy().shape == (first.memory.retained_dimension, 0)


@pytest.mark.parametrize("field", [f for f in _CAP_TO_PLAN if f != "maximum_integral_calls"])
def test_overlap_exact_caps_and_one_below(physical_case, field):
    b, provider, _ = physical_case
    pair = _make(b, provider, _pnos(b, provider))
    p = _overlap(b, pair, pair, plan=True)
    c = _overlap_caps()
    for cap in (f for f in _CAP_TO_PLAN if f != "maximum_integral_calls"):
        setattr(c, cap, getattr(p, _CAP_TO_PLAN[cap]))
    _overlap(b, pair, pair, caps=c)
    setattr(c, field, getattr(c, field)-1)
    with pytest.raises(ValueError, match="cap"):
        _overlap(b, pair, pair, caps=c)


def test_foreign_equal_payload_state_fails_and_no_array_or_mutable_owner_escape():
    b, foreign = _prepare_leaves(_bundle()), _prepare_leaves(_bundle())
    provider, other = _provider(b), _provider(foreign)
    pno = _pnos(b, provider)
    assert b.reference.state.state_identity_sha256 == foreign.reference.state.state_identity_sha256
    with pytest.raises(ValueError, match="owners|receipts"):
        _plan(b, other, pno)
    first, second = _make(b, provider, pno), _make(foreign, other, _pnos(foreign, other))
    with pytest.raises(ValueError, match="same certified"):
        _overlap(b, first, second)
    for cls in (core._PeriodicGaussianPairSpace, core._PeriodicGaussianPairOverlap):
        with pytest.raises(TypeError):
            cls()
    identity = first.identity_sha256
    values = first.exchange_integrals_copy()
    first.coefficients_copy()[:] = np.nan
    first.exchange_integrals_copy()[:] = np.nan
    first.pno_options.occupation_cutoff = 1.0
    np.testing.assert_array_equal(first.exchange_integrals_copy(), values)
    assert first.identity_sha256 == identity
    assert first.pno_options.occupation_cutoff == 0.0
    overlap = _overlap(b, first, first)
    expected = overlap.overlaps_copy()
    del b, provider, foreign, other, first, second, pno
    gc.collect()
    np.testing.assert_array_equal(overlap.overlaps_copy(), expected)


@pytest.mark.parametrize("n,r", [(0, 0), (2, 3), (2**63, 1), (2**32, 2**32)])
def test_count_only_inventory_rejects_invalid_shapes_and_overflow(n, r):
    with pytest.raises((ValueError, OverflowError)):
        core._periodic_gaussian_pair_space_storage(n, r)
