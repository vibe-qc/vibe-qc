"""Streamed common integrals in genuine native embedded pair-PNO frames.

The generation density has already been formed in each distinct PAO frame.
This consumer neither reconstructs it nor pads its original occupations.
Only tiny actual finite-Gaussian sources are used; no production DLPNO claim.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_embedded_pair_pnos import (
    physical, rectangular, _physical, _frame, _make as _pnos,
    _options as _pno_options,
)
from tests.test_periodic_gaussian_pair_space import (
    _make, _plan, _caps, _options, _overlap,
    _row_order_oracle, _integrals_wire, _CAP_TO_PLAN,
    _STORAGE_FIELDS, _PLAN_FIELDS, _POLICY, _CanonicalDigest,
)
from tests.test_periodic_gaussian_pair_pnos import _make as _legacy_pnos


def _source_integral_digest(provider, o, n, i, j):
    h = _CanonicalDigest("vibeqc.periodic.gaussian-pair-pnos.integrals", 1)
    h.u64(n)
    h.u64(n*n)
    for a in range(n):
        for b in range(n):
            h.binary64(provider.provider.integral(i, o+a, j, o+b).value)
    return h.finish()


def _check(b, provider, dense, pno, i=0, j=0):
    c = pno.coefficients_copy()
    eps = pno.energies_copy()
    occupations = pno.original_pno_occupations_copy()
    n, r = c.shape
    m = len(occupations)
    o = dense["o"]
    source_identity, source_payload = pno.identity_sha256, pno.payload_sha256
    embedding_identity = pno.embedding_identity_sha256
    source_g = _source_integral_digest(provider, o, n, i, j)
    assert pno.common_exchange_integral_identity_sha256 == source_g
    raw = _row_order_oracle(provider, c, o, i, j)
    result = _make(b, provider, pno)
    np.testing.assert_array_equal(result.coefficients_copy(), c)
    np.testing.assert_array_equal(result.energies_copy(), eps)
    np.testing.assert_array_equal(result.original_pno_occupations_copy(), occupations)
    np.testing.assert_allclose(result.exchange_integrals_copy(),
        c.T@dense["eri"][i, o:, j, o:]@c, atol=6e-12, rtol=3e-11)
    assert result.raw_exchange_integral_identity_sha256 == _integrals_wire(raw, n, r)
    assert result.exchange_integral_identity_sha256 == _integrals_wire(result.exchange_integrals_copy(), n, r)
    assert result.common_source_exchange_integral_identity_sha256 == source_g
    assert result.embedded_generation
    assert result.memory.generation_dimension == m
    assert result.memory.borrowed_pno_bytes == 8*(n*r+m*r+r+m)
    assert result.memory.retained_output_bytes == 8*(n*r+m*r+r+m+r*r)
    assert result.pno_identity_sha256 == source_identity
    assert result.pno_payload_sha256 == source_payload
    assert result.embedding_identity_sha256 == embedding_identity
    assert result.complete_generation_pair_space == (r == m)
    assert result.complete_common_virtual_space == (r == n)
    assert result.occupied_i == tuple(b.rows[i]) and result.occupied_j == tuple(b.rows[j])
    assert result.diagnostics.completed_integral_calls == (n*n if r else 0)
    assert result.diagnostics.source_integrals_replayed == bool(r)
    assert result.pno_diagnostics.retained_output_bytes == 8*(m*r+r+m)
    assert result.embedded_pno_diagnostics.retained_output_bytes == 8*(n*r+m*r+r+m)
    assert not result.production_dlpno and not result.coupled_mp2_solution
    for name in ("pnos", "embedded_pnos", "frame"):
        assert not hasattr(result, name)  # No consumable borrowed owner escape.
    with pytest.raises(RuntimeError, match="moved|malformed|consumed"):
        pno.coefficients_copy()
    with pytest.raises(RuntimeError, match="moved|malformed|consumed"):
        pno.generation_coefficients_copy()
    if i == j:
        np.testing.assert_array_equal(result.exchange_integrals_copy(), result.exchange_integrals_copy().T)
    else:
        np.testing.assert_array_equal(result.exchange_integrals_copy(), raw)
        assert not result.diagnostics.diagonal_symmetry_projection_applied
    return result


def test_actual_gamma_even_and_nonself_meshes_preserve_source_and_payload(physical):
    b, provider, dense = physical
    embedding, _ = _frame(b)
    _check(b, provider, dense, _pnos(b, provider, embedding))


@pytest.mark.parametrize("i,j", [(0, 0), (0, 2), (2, 0)])
def test_rectangular_generation_and_ordered_pair_integrals(rectangular, i, j):
    b, provider, dense, embedding, _ = rectangular
    result = _check(b, provider, dense, _pnos(b, provider, embedding, i, j), i, j)
    assert result.memory.virtual_count == 4 and result.memory.generation_dimension == 2
    assert result.complete_generation_pair_space and not result.complete_common_virtual_space
    assert result.original_pno_occupations_copy().shape == (2,)
    if i != j:
        assert np.linalg.norm(result.exchange_integrals_copy()-result.exchange_integrals_copy().T) > 1e-10


def test_positive_cutoff_and_zero_rank_keep_unpadded_generation_occupations(rectangular):
    b, provider, dense, embedding, _ = rectangular
    values = _pnos(b, provider, embedding).original_pno_occupations_copy()
    assert values[0] > values[1] and values[0] > 0
    for cutoff, rank in (((values[0]+values[1])/2, 1), (2*values[0], 0)):
        pno = _pnos(b, provider, embedding, options=_pno_options(cutoff=cutoff))
        result = _check(b, provider, dense, pno)
        assert result.memory.retained_dimension == rank
        assert result.original_pno_occupations_copy().shape == (2,)
        if not rank:
            assert result.memory.peak_owned_numerical_bytes == 16
            assert result.exchange_integrals_copy().shape == (0, 0)
            assert result.diagnostics.diagonal_symmetry_projection_frobenius_bound == 0


def test_count_only_storage_and_embedded_plan_wire(rectangular):
    b, provider, _, embedding, _ = rectangular
    pno = _pnos(b, provider, embedding)
    p = _plan(b, provider, pno)
    n, m, r = p.virtual_count, p.generation_dimension, p.retained_dimension
    storage = core._periodic_gaussian_pair_space_storage(n, m, r)
    for field in _STORAGE_FIELDS:
        assert getattr(storage, field) == getattr(p, field)
    assert p.construction_owned_bytes == 16*r*r+16*r
    assert p.construction_live_numerical_bytes == p.borrowed_pno_bytes+p.construction_owned_bytes
    assert p.peak_owned_numerical_bytes == max(p.construction_owned_bytes, p.retained_output_bytes)
    assert p.borrowed_pno_bytes == pno.diagnostics.retained_output_bytes
    assert p.per_worker_inventoried_bytes == (p.construction_live_numerical_bytes+p.borrowed_basis_bytes
        +p.borrowed_provider_row_bytes+p.fixed_control_storage_bytes+p.live.other_live_bytes_per_worker
        +p.live.fixed_backend_margin_bytes_per_worker)
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
    h.string("embedded-pair-generation")
    h.u64(m)
    assert p.plan_identity_sha256 == h.finish()
    legacy = core._periodic_gaussian_pair_space_storage(n, r)
    full_embedded = core._periodic_gaussian_pair_space_storage(n, n, r)
    # Equal dimensions do not alias the two distinct retained coefficients.
    extra = 8*n*r
    assert full_embedded.borrowed_pno_bytes == legacy.borrowed_pno_bytes+extra
    assert full_embedded.retained_output_bytes == legacy.retained_output_bytes+extra
    assert full_embedded.construction_live_numerical_bytes == legacy.construction_live_numerical_bytes+extra
    assert full_embedded.construction_owned_bytes == legacy.construction_owned_bytes
    assert full_embedded.numerical_work_units == legacy.numerical_work_units+512*n*r


@pytest.mark.parametrize("dimensions", [(0, 1, 0), (2, 0, 0), (2, 3, 0), (3, 2, 3)])
def test_invalid_generation_dimensions_fail_closed(dimensions):
    with pytest.raises(ValueError, match="dimensions"):
        core._periodic_gaussian_pair_space_storage(*dimensions)


def test_count_only_extent_overflow_fails_before_allocation():
    with pytest.raises((ValueError, OverflowError), match="overflow|extent|cap"):
        core._periodic_gaussian_pair_space_storage(2**63, 1, 1)


@pytest.mark.parametrize("field", _CAP_TO_PLAN)
def test_exact_caps_and_failures_preserve_embedded_owner(rectangular, field):
    b, provider, _, embedding, _ = rectangular
    pno = _pnos(b, provider, embedding)
    c, occupations = pno.coefficients_copy(), pno.original_pno_occupations_copy()
    p = _plan(b, provider, pno)
    caps = _caps()
    for cap, name in _CAP_TO_PLAN.items():
        setattr(caps, cap, getattr(p, name))
    setattr(caps, field, getattr(caps, field)-1)
    with pytest.raises(ValueError, match="cap"):
        _make(b, provider, pno, caps=caps)
    np.testing.assert_array_equal(pno.coefficients_copy(), c)
    np.testing.assert_array_equal(pno.original_pno_occupations_copy(), occupations)
    setattr(caps, field, getattr(p, _CAP_TO_PLAN[field]))
    result = _make(b, provider, pno, caps=caps)
    np.testing.assert_array_equal(result.coefficients_copy(), c)


def test_projection_failure_before_final_transfer_and_exact_boundary(rectangular):
    b, provider, _, embedding, _ = rectangular
    accepted = _make(b, provider, _pnos(b, provider, embedding))
    bound = accepted.diagnostics.diagonal_symmetry_projection_frobenius_bound
    pno = _pnos(b, provider, embedding)
    c = pno.coefficients_copy()
    if bound > 0:
        with pytest.raises(ValueError, match="projection.*budget"):
            _make(b, provider, pno, options=_options(np.nextafter(bound, 0)))
        np.testing.assert_array_equal(pno.coefficients_copy(), c)
    result = _make(b, provider, pno, options=_options(bound))
    assert result.diagnostics.diagonal_symmetry_projection_frobenius_bound == bound


def test_mixed_generation_frames_overlap_without_owner_or_occupation_padding(rectangular):
    b, provider, _, embedding, _ = rectangular
    common = _make(b, provider, _legacy_pnos(b, provider))
    pair = _make(b, provider, _pnos(b, provider, embedding, 0, 2))
    values = _pnos(b, provider, embedding).original_pno_occupations_copy()
    truncated = _make(b, provider, _pnos(b, provider, embedding,
        options=_pno_options(cutoff=(values[0]+values[1])/2)))
    empty = _make(b, provider, _pnos(b, provider, embedding,
        options=_pno_options(cutoff=2*values[0])))
    assert not common.embedded_generation
    with pytest.raises(RuntimeError, match="no embedded"):
        _ = common.embedded_pno_diagnostics
    for left, right in ((common, pair), (pair, common), (pair, pair), (pair, truncated), (empty, pair)):
        result = _overlap(b, left, right)
        np.testing.assert_allclose(result.overlaps_copy(), left.coefficients_copy().T@right.coefficients_copy(), atol=5e-14)
        expected_bytes = left.memory.retained_output_bytes
        if left is not right:
            expected_bytes += right.memory.retained_output_bytes
        assert result.memory.borrowed_pair_space_bytes == expected_bytes
    equal = _make(b, provider, _pnos(b, provider, embedding, 0, 2))
    assert equal.identity_sha256 == pair.identity_sha256
    double = _overlap(b, pair, equal, plan=True)
    single = _overlap(b, pair, pair, plan=True)
    assert double.borrowed_pair_space_bytes == 2*single.borrowed_pair_space_bytes
    assert double.work_units > single.work_units


def test_wrong_exact_owner_rejected_without_consuming_source(rectangular):
    b, provider, _, embedding, _ = rectangular
    other, other_provider, _ = _physical()
    pno = _pnos(b, provider, embedding)
    before = pno.coefficients_copy()
    with pytest.raises(ValueError, match="identical|receipts|owner"):
        _make(other, other_provider, pno)
    np.testing.assert_array_equal(pno.coefficients_copy(), before)
    result = _make(b, provider, pno)
    with pytest.raises(ValueError, match="same certified|origin"):
        _overlap(other, result, result)


def test_retained_frame_lifetime_and_copies_cannot_consume_child():
    b, provider, _ = _physical("he", 2)
    embedding, generated = _frame(b)
    result = _make(b, provider, _pnos(b, provider, embedding))
    c, eps, occupations = result.coefficients_copy(), result.energies_copy(), result.original_pno_occupations_copy()
    state, context = result.state, result.context
    del b, provider, embedding, generated
    gc.collect()
    np.testing.assert_array_equal(result.coefficients_copy(), c)
    np.testing.assert_array_equal(result.energies_copy(), eps)
    np.testing.assert_array_equal(result.original_pno_occupations_copy(), occupations)
    assert result.state is state and result.context is context
    c[:] = np.nan
    assert np.all(np.isfinite(result.coefficients_copy()))
    options = result.embedded_pno_options
    options.maximum_exported_gram_error = 0
    assert result.embedded_pno_options.maximum_exported_gram_error > 0
