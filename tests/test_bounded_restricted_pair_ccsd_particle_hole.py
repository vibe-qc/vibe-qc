from __future__ import annotations

import gc
import hashlib
import struct

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


def _inventory(**changes):
    result = core._BoundedRestrictedPairCCSDParticleHoleInventory()
    result.numerical_replicas = 1
    result.backend_margin_bytes_per_replica = 1024
    for key, value in changes.items():
        setattr(result, key, value)
    return result


def _plan(a, o, maximum_rank, inventory=None):
    return core._plan_bounded_restricted_pair_ccsd_particle_hole(
        a, o, maximum_rank, _inventory() if inventory is None else inventory
    )


_CAP_FIELDS = {
    "maximum_target_dimension": "target_dimension",
    "maximum_source_dimension": "maximum_source_dimension",
    "maximum_occupied_count": "occupied_count",
    "maximum_borrowed_numerical_bytes": "maximum_borrowed_numerical_bytes",
    "maximum_owned_numerical_bytes": "peak_owned_numerical_bytes",
    "maximum_control_storage_bytes": "total_control_storage_bytes",
    "maximum_per_replica_inventoried_bytes": "per_replica_inventoried_bytes",
    "maximum_node_inventoried_bytes": "required_node_inventoried_bytes",
    "maximum_scalar_products": "scalar_products_upper_bound",
    "maximum_work_units": "work_units_upper_bound",
}


def _caps(plan, **changes):
    result = core._BoundedRestrictedPairCCSDParticleHoleCaps()
    for cap, count in _CAP_FIELDS.items():
        setattr(result, cap, getattr(plan, count))
    for key, value in changes.items():
        setattr(result, key, value)
    return result


def _accumulator(a, o, i, j, maximum_rank, *, inventory=None, caps=None):
    inventory = _inventory() if inventory is None else inventory
    caps = _caps(_plan(a, o, maximum_rank, inventory)) if caps is None else caps
    return core._BoundedRestrictedPairCCSDParticleHoleAccumulator(
        a, o, i, j, maximum_rank, inventory, caps
    )


def _source(a=2, b=3, seed=81):
    rng = np.random.default_rng(seed)
    return tuple(np.ascontiguousarray(rng.normal(scale=0.1, size=shape))
                 for shape in [(b, a), (b, a), (b, a), (b, b)])


def _system(o=3, n=6, ranks=None, seed=9041):
    """Independent dense COMMON-tensor oracle fixture, not a physical owner."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(scale=0.2, size=(5, o + n, o + n))
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    eri = np.einsum("Ppq,Prs->pqrs", factors, factors)
    frames, amplitudes = {}, {}
    for i in range(o):
        for j in range(i, o):
            rank = 1 + (i + 2 * j) % 3 if ranks is None else ranks[i, j]
            frames[i, j] = np.linalg.qr(rng.normal(size=(n, max(rank, 1))))[0][:, :rank]
            t = rng.normal(scale=0.05, size=(rank, rank))
            amplitudes[i, j] = (t + t.T) / 2 if i == j else t
    return o, n, eri, frames, amplitudes


def _common_amplitudes(system):
    o, n, _, frames, amplitudes = system
    t2 = np.zeros((o, o, n, n))
    for (i, j), c in frames.items():
        block = c @ amplitudes[i, j] @ c.T
        t2[i, j] = block
        t2[j, i] = block.T
    return t2


def _original_three_seed_oracle(system, i, j):
    # This is the ORIGINAL target's W1/W2/WX contraction, NOT the new F
    # factorization. Build all common T only in this tiny independent test,
    # evaluate all global e terms, then project the FINAL target. See
    # bounded_restricted_ccsd_target.cpp's two particle-hole passes and
    # Stanton1991 Eq.(2)/(8), Riplinger2013 restricted Eq.(7).
    o, n, eri, frames, _ = system
    t2 = _common_amplitudes(system)
    result = np.zeros((n, n))
    for leg, (left, right) in enumerate([(i, j), (j, i)]):
        for a in range(n):
            for b in range(n):
                value = 0.0
                for m in range(o):
                    for e in range(n):
                        w1 = eri[m, o + e, right, o + b]
                        w2 = eri[m, o + e, right, o + b] - eri[m, right, o + b, o + e]
                        wx = -eri[m, left, o + b, o + e]
                        value += (t2[left, m, a, e] - t2[m, left, a, e]) * w1
                        value += t2[left, m, a, e] * w2
                        value += t2[m, right, a, e] * wx
                if leg == 0:
                    result[a, b] += value
                else:
                    result[b, a] += value
    target = frames[min(i, j), max(i, j)]
    return target.T @ result @ target


def _system_sources(system, i, j):
    o, _, eri, frames, amplitudes = system
    target = frames[min(i, j), max(i, j)]
    sources = []
    for leg, (left, right) in enumerate([(i, j), (j, i)]):
        for m in range(o):
            key = min(left, m), max(left, m)
            source = frames[key]
            k = source.T @ eri[m, o:, right, o:] @ target
            exchange = source.T @ eri[m, right, o:, o:] @ target
            overlap = source.T @ target
            arrays = tuple(np.ascontiguousarray(x) for x in (k, exchange, overlap, amplitudes[key]))
            sources.append((leg, m, arrays, left > m))
    return sources


def _evaluate_system(system, i, j):
    o, _, _, frames, _ = system
    a = frames[min(i, j), max(i, j)].shape[1]
    sources = _system_sources(system, i, j)
    maximum_rank = max(arrays[3].shape[0] for _, _, arrays, _ in sources)
    accumulator = _accumulator(a, o, i, j, maximum_rank)
    for leg, m, arrays, transposed in sources:
        accumulator.accumulate(leg, m, *arrays, source_transposed=transposed)
    return accumulator.finish()


@pytest.mark.parametrize("o", [1, 2, 3, 4])
@pytest.mark.parametrize("equal", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_complete_three_bare_terms_both_legs_match_independent_common_target(o, equal, reverse):
    system = _system(o=o)
    i, j = (0, 0) if equal or o == 1 else (0, o - 1)
    if reverse:
        i, j = j, i
    result = _evaluate_system(system, i, j)
    np.testing.assert_allclose(result.residual_copy(), _original_three_seed_oracle(system, i, j), atol=2e-16, rtol=3e-13)
    assert (result.target_i, result.target_j) == (i, j)
    assert not result.physical_source_certified
    assert not result.entire_ccsd_residual
    if i == j:
        np.testing.assert_allclose(result.residual_copy(), result.residual_copy().T, atol=2e-17, rtol=3e-14)


@pytest.mark.parametrize("a", range(4))
@pytest.mark.parametrize("b", range(4))
@pytest.mark.parametrize("c", range(4))
def test_independent_unequal_nonnested_a_b_c_frames_and_zero_rank_slots(a, b, c):
    ranks = {(0, 0): b, (0, 1): a, (0, 2): c,
             (1, 1): c, (1, 2): b, (2, 2): 1}
    system = _system(ranks=ranks)
    result = _evaluate_system(system, 0, 1)
    np.testing.assert_allclose(result.residual_copy(), _original_three_seed_oracle(system, 0, 1), atol=2e-16, rtol=3e-13)
    p, d = result.memory, result.diagnostics
    sources = _system_sources(system, 0, 1)
    source_ranks = [arrays[3].shape[0] for _, _, arrays, _ in sources]
    maximum = max(source_ranks)
    assert p.source_slots == d.visited_sources == 6
    assert d.zero_rank_sources == source_ranks.count(0)
    assert p.output_bytes == p.compensation_bytes == 8 * a * a
    assert p.intermediate_bytes == 8 * a * maximum
    assert p.peak_owned_numerical_bytes == 16 * a * a + 8 * a * maximum
    assert p.maximum_borrowed_numerical_bytes == 8 * (maximum**2 + 3 * a * maximum)
    assert d.scalar_products == sum(4 * a * r**2 + 2 * a**2 * r for r in source_ranks)
    assert d.input_scan_elements == sum(2 * (r**2 + 3 * a * r) for r in source_ranks)
    assert d.scalar_products <= p.scalar_products_upper_bound
    expected_work = p.finish_work_units + sum(
        core._plan_bounded_restricted_pair_ccsd_particle_hole_source(a, r).work_units_upper_bound
        for r in source_ranks
    )
    assert d.charged_work_units == expected_work <= p.work_units_upper_bound


def test_independent_frame_rotations_reversal_and_exchange_terms():
    original = _system()
    o, n, eri, frames, amplitudes = original
    rng = np.random.default_rng(762)
    rotations = {key: np.linalg.qr(rng.normal(size=(c.shape[1], c.shape[1])))[0]
                 for key, c in frames.items()}
    rotated = (o, n, eri,
               {key: c @ rotations[key] for key, c in frames.items()},
               {key: rotations[key].T @ t @ rotations[key] for key, t in amplitudes.items()})
    first = _evaluate_system(original, 0, 2).residual_copy()
    second = _evaluate_system(rotated, 0, 2).residual_copy()
    rotation = rotations[0, 2]
    np.testing.assert_allclose(second, rotation.T @ first @ rotation, atol=3e-16, rtol=3e-13)
    np.testing.assert_allclose(second, _original_three_seed_oracle(rotated, 0, 2), atol=3e-16, rtol=3e-13)
    reverse = _evaluate_system(original, 2, 0).residual_copy()
    np.testing.assert_allclose(reverse, first.T, atol=3e-17, rtol=3e-14)
    # Exchange is numerically substantial; this must not collapse to twice
    # the Eq.26 semi-joint contribution or a direct-only ring.
    sources = _system_sources(original, 0, 2)
    a = frames[0, 2].shape[1]
    stripped = _accumulator(a, o, 0, 2, max(x[2][3].shape[0] for x in sources))
    for leg, m, arrays, transposed in sources:
        k, j, overlap, t = arrays
        stripped.accumulate(leg, m, k, np.zeros_like(j), overlap, t, transposed)
    assert np.linalg.norm(stripped.finish().residual_copy() - first) > 1e-5


def test_projecting_internal_source_to_target_is_a_detected_scientific_error():
    system = _system(o=2, n=5, ranks={(0, 0): 3, (0, 1): 2, (1, 1): 3})
    o, n, eri, frames, amplitudes = system
    correct = _evaluate_system(system, 0, 1).residual_copy()
    target = frames[0, 1]
    projector = target @ target.T
    # Deliberately wrong target-first restriction of each SOURCE frame loses
    # components needed in K/J's contracted internal index.
    wrong = (o, n, eri, {key: projector @ c for key, c in frames.items()}, amplitudes)
    wrong_result = _evaluate_system(wrong, 0, 1).residual_copy()
    assert np.linalg.norm(correct - wrong_result) > 1e-4
    np.testing.assert_allclose(correct, _original_three_seed_oracle(system, 0, 1), atol=2e-16, rtol=3e-13)


@pytest.mark.parametrize("transposed", [False, True])
def test_source_transpose_flag_only_changes_t_and_matches_explicit_transpose_bitwise(transposed):
    sources = [_source(seed=121), _source(seed=913)]
    results = []
    for explicit in [False, True]:
        accumulator = _accumulator(2, 1, 0, 0, 3)
        for leg, arrays in enumerate(sources):
            k, j, overlap, t = arrays
            if explicit and transposed:
                t = np.ascontiguousarray(t.T)
            accumulator.accumulate(leg, 0, k, j, overlap, t, transposed and not explicit)
        results.append(accumulator.finish())
    np.testing.assert_array_equal(results[0].residual_copy(), results[1].residual_copy())
    assert results[0].payload_identity_sha256 == results[1].payload_identity_sha256
    if transposed:
        assert results[0].input_stream_identity_sha256 != results[1].input_stream_identity_sha256


@pytest.mark.parametrize("field", list(_CAP_FIELDS))
def test_every_admission_cap_minus_one_rejects_before_a_source_can_be_scanned(field):
    p = _plan(2, 2, 3)
    cap = _caps(p, **{field: getattr(p, _CAP_FIELDS[field]) - 1})
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _accumulator(2, 2, 0, 1, 3, caps=cap)


def test_exact_owner_inventory_controls_and_replica_arithmetic():
    inv = _inventory(numerical_replicas=3, external_node_bytes=771,
                     other_live_numerical_bytes_per_replica=514,
                     other_live_control_bytes_per_replica=127,
                     backend_margin_bytes_per_replica=4096)
    p = _plan(2, 3, 4, inv)
    assert p.total_control_storage_bytes == p.fixed_control_storage_bytes + 127
    assert p.per_replica_inventoried_bytes == p.peak_owned_numerical_bytes + p.maximum_borrowed_numerical_bytes + 514 + p.total_control_storage_bytes + 4096
    assert p.required_node_inventoried_bytes == 771 + 3 * p.per_replica_inventoried_bytes
    source = core._plan_bounded_restricted_pair_ccsd_particle_hole_source(2, 4)
    assert source.borrowed_input_elements == 4**2 + 3 * 2 * 4
    assert source.scalar_products == 4 * 2 * 4**2 + 2 * 2**2 * 4
    assert p.scalar_products_upper_bound == 6 * source.scalar_products


@pytest.mark.parametrize("field", ["numerical_replicas", "backend_margin_bytes_per_replica"])
def test_required_inventory_is_explicit(field):
    with pytest.raises(ValueError, match="positive"):
        _plan(1, 1, 1, _inventory(**{field: 0}))


@pytest.mark.parametrize("args", [(2**63, 1, 1), (1, 2**63, 1), (1, 1, 2**63)])
def test_count_only_planner_rejects_overflow_without_arrays(args):
    with pytest.raises(OverflowError):
        _plan(*args)


def test_required_occupied_count_and_target_indices():
    with pytest.raises(ValueError, match="positive"):
        _plan(1, 0, 1)
    with pytest.raises(ValueError, match="occupied index"):
        _accumulator(1, 1, 1, 0, 1)


@pytest.mark.parametrize("leg,m", [(1, 0), (0, 1), (2, 0)])
def test_source_slot_order_is_strict_and_native_errors_poison_stream(leg, m):
    accumulator = _accumulator(2, 2, 0, 1, 3)
    with pytest.raises(ValueError, match="next leg/occupied"):
        accumulator.accumulate(leg, m, *_source())
    assert accumulator.failed
    with pytest.raises(ValueError, match="failed"):
        accumulator.finish()


def test_omitted_even_zero_rank_slots_extra_slots_and_finish_state():
    empty = _source(a=2, b=0)
    accumulator = _accumulator(2, 2, 0, 1, 0)
    accumulator.accumulate(0, 0, *empty)
    with pytest.raises(ValueError, match="every"):
        accumulator.finish()
    assert accumulator.failed
    good = _accumulator(2, 2, 0, 1, 0)
    for leg in range(2):
        for m in range(2):
            good.accumulate(leg, m, *empty)
    result = good.finish()
    assert good.finished and not good.failed
    assert result.diagnostics.visited_sources == result.diagnostics.zero_rank_sources == 4
    np.testing.assert_array_equal(result.residual_copy(), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finished"):
        good.finish()
    np.testing.assert_array_equal(result.residual_copy(), np.zeros((2, 2)))
    extra = _accumulator(2, 1, 0, 0, 0)
    extra.accumulate(0, 0, *empty)
    extra.accumulate(1, 0, *empty)
    with pytest.raises(ValueError, match="next leg/occupied"):
        extra.accumulate(1, 0, *empty)


def test_source_rank_gate_precedes_nonfinite_payload_scan():
    accumulator = _accumulator(2, 1, 0, 0, 2)
    arrays = _source(a=2, b=3)
    arrays[0][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="rank exceeds"):
        accumulator.accumulate(0, 0, *arrays)
    assert accumulator.failed


@pytest.mark.parametrize("which", range(4))
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_all_source_roles_are_finite_scanned_before_contraction(which, bad):
    accumulator = _accumulator(2, 1, 0, 0, 3)
    arrays = _source()
    arrays[which].flat[0] = bad
    with pytest.raises(ValueError, match="nonfinite"):
        accumulator.accumulate(0, 0, *arrays)
    assert accumulator.failed


def test_zero_target_still_audits_nonempty_source_amplitudes():
    accumulator = _accumulator(0, 1, 0, 0, 2)
    arrays = _source(a=0, b=2)
    arrays[3][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        accumulator.accumulate(0, 0, *arrays)


@pytest.mark.parametrize("fault", [1, 2, 3, 4])
def test_native_exact_extent_null_and_alignment_protocol(fault):
    accumulator = _accumulator(2, 1, 0, 0, 3)
    with pytest.raises(ValueError, match="extent|pointer/alignment"):
        accumulator.accumulate(0, 0, *_source(), view_fault=fault)
    assert accumulator.failed


@pytest.mark.parametrize("fault", [1, 4])
def test_empty_views_require_exact_zero_extents(fault):
    accumulator = _accumulator(2, 1, 0, 0, 0)
    with pytest.raises(ValueError, match="extent"):
        accumulator.accumulate(0, 0, *_source(a=2, b=0), view_fault=fault)


@pytest.mark.parametrize("kind", ["dtype", "strided", "shape", "unaligned"])
def test_binding_does_not_forcecast_or_accept_wrong_shapes(kind):
    arrays = list(_source())
    if kind == "dtype":
        arrays[0] = arrays[0].astype(np.float32)
    elif kind == "strided":
        arrays[0] = np.ones((3, 4))[:, ::2]
    elif kind == "shape":
        arrays[1] = np.ones((2, 3))
    else:
        arrays[0] = np.ndarray((3, 2), dtype=np.float64, buffer=bytearray(49), offset=1)
    accumulator = _accumulator(2, 1, 0, 0, 3)
    with pytest.raises(ValueError, match="aligned|shape"):
        accumulator.accumulate(0, 0, *arrays)


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_rounding_mode_gate_poison_and_binding_restore(mode):
    accumulator = _accumulator(2, 1, 0, 0, 3)
    with pytest.raises(ValueError, match="nearest rounding"):
        accumulator.accumulate(0, 0, *_source(), rounding_fault=mode)
    assert accumulator.failed
    # The diagnostic seam restores the caller environment even on exception.
    _evaluate_system(_system(o=1), 0, 0)


def test_compensated_cancellation_subnormal_and_finite_overflow_gates():
    k = np.array([[1e16], [1.0], [-1e16]])
    exchange = np.zeros((3, 1))
    overlap = np.ones((3, 1))
    t = np.eye(3)
    accumulator = _accumulator(1, 1, 0, 0, 3)
    for leg in range(2):
        accumulator.accumulate(leg, 0, k, exchange, overlap, t)
    np.testing.assert_array_equal(accumulator.finish().residual_copy(), [[2.0]])
    tiny = np.nextafter(0.0, 1.0)
    subnormal = _accumulator(1, 1, 0, 0, 1)
    for leg in range(2):
        subnormal.accumulate(leg, 0, np.array([[tiny]]), np.zeros((1, 1)), np.ones((1, 1)), np.ones((1, 1)))
    assert subnormal.finish().residual(0, 0) == 2 * tiny
    underflow = _accumulator(1, 1, 0, 0, 1)
    for leg in range(2):
        underflow.accumulate(leg, 0, np.array([[tiny]]), np.zeros((1, 1)), np.ones((1, 1)), np.array([[0.5]]))
    result = underflow.finish()
    assert result.residual(0, 0) == 0
    assert result.diagnostics.product_underflow_count == 2
    overflow = _accumulator(1, 1, 0, 0, 1)
    with pytest.raises(OverflowError, match="nonfinite"):
        overflow.accumulate(0, 0, np.ones((1, 1)), np.zeros((1, 1)), np.ones((1, 1)), np.array([[np.finfo(float).max]]))
    assert overflow.failed


def test_controls_are_snapshotted_source_owners_not_retained_and_result_copy_is_readonly():
    inventory = _inventory()
    p = _plan(2, 1, 3, inventory)
    caps = _caps(p)
    accumulator = _accumulator(2, 1, 0, 0, 3, inventory=inventory, caps=caps)
    inventory.numerical_replicas = 0
    for field in _CAP_FIELDS:
        setattr(caps, field, 0)
    arrays = _source()
    for leg in range(2):
        accumulator.accumulate(leg, 0, *arrays)
    result = accumulator.finish()
    expected = result.residual_copy()
    for x in arrays:
        x[:] = np.nan
    del arrays, accumulator
    gc.collect()
    np.testing.assert_array_equal(result.residual_copy(), expected)
    assert not expected.flags.writeable
    with pytest.raises(ValueError):
        expected[0, 0] = 1
    with pytest.raises(IndexError):
        result.residual(2, 0)


def _digest(domain, *values):
    h = hashlib.sha256()
    def string(value):
        raw = value.encode()
        h.update(struct.pack(">Q", len(raw)))
        h.update(raw)
    string(domain)
    h.update(struct.pack(">Q", 1))
    for value in values:
        if isinstance(value, str):
            string(value)
        elif isinstance(value, int):
            h.update(struct.pack(">Q", value))
        else:
            for x in np.asarray(value).flat:
                h.update(struct.pack(">d", 0.0 if x == 0 else float(x)))
    return h.hexdigest()


def test_actual_stream_hash_binds_slots_orientations_arrays_and_not_resource_caps():
    sources = [_source(seed=110), _source(seed=130)]
    accumulator = _accumulator(2, 1, 0, 0, 3)
    stream = _digest("vibeqc.bounded.pair-ccsd-particle-hole.stream", 2, 1, 0, 0)
    for leg, arrays in enumerate(sources):
        accumulator.accumulate(leg, 0, *arrays, source_transposed=bool(leg))
        receipt = _digest("vibeqc.bounded.pair-ccsd-particle-hole.source", 2, 3, leg, 0, leg, *arrays)
        stream = _digest("vibeqc.bounded.pair-ccsd-particle-hole.chain", stream, receipt)
    result = accumulator.finish()
    assert result.input_stream_identity_sha256 == stream
    assert result.payload_identity_sha256 == _digest(
        "vibeqc.bounded.pair-ccsd-particle-hole.payload", 2, 1, 0, 0, result.residual_copy()
    )
    assert len(result.identity_sha256) == 64


def test_aliases_are_charged_per_role_and_signed_zero_receipts_are_canonical():
    p = _plan(2, 1, 2)
    assert p.maximum_borrowed_numerical_bytes == 4 * 2 * 2 * 8
    results = []
    for negative in [False, True]:
        common = np.full((2, 2), -0.0 if negative else 0.0)
        accumulator = _accumulator(2, 1, 0, 0, 2)
        for leg in range(2):
            accumulator.accumulate(leg, 0, common, common, common, common)
        results.append(accumulator.finish())
    assert results[0].identity_sha256 == results[1].identity_sha256
