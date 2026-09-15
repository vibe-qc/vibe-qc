from __future__ import annotations

import gc
import hashlib
import struct
from decimal import Decimal, localcontext

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


def _inventory(**changes):
    result = core._BoundedRestrictedPairCCSDInteractionInventory()
    result.numerical_replicas = 1
    result.backend_margin_bytes_per_replica = 1024
    for key, value in changes.items():
        setattr(result, key, value)
    return result


def _plan(a, b, inventory=None):
    return core._plan_bounded_restricted_pair_ccsd_interaction(
        a, b, _inventory() if inventory is None else inventory
    )


_CAP_FIELDS = {
    "maximum_target_dimension": "target_dimension",
    "maximum_source_dimension": "source_dimension",
    "maximum_borrowed_numerical_bytes": "borrowed_numerical_bytes",
    "maximum_owned_numerical_bytes": "peak_owned_numerical_bytes",
    "maximum_control_storage_bytes": "total_control_storage_bytes",
    "maximum_per_replica_inventoried_bytes": "per_replica_inventoried_bytes",
    "maximum_node_inventoried_bytes": "required_node_inventoried_bytes",
    "maximum_scalar_products": "scalar_products",
    "maximum_work_units": "work_units_upper_bound",
}


def _caps(plan, **changes):
    result = core._BoundedRestrictedPairCCSDInteractionCaps()
    for cap, count in _CAP_FIELDS.items():
        setattr(result, cap, getattr(plan, count))
    for key, value in changes.items():
        setattr(result, key, value)
    return result


def _case(a=2, b=3, seed=771):
    rng = np.random.default_rng(seed)
    return tuple(np.ascontiguousarray(rng.normal(scale=0.2, size=shape))
                 for shape in [(a, b), (b, a), (b, a), (b, b)])


def _run(case, *, inventory=None, caps=None, **kwargs):
    inventory = _inventory() if inventory is None else inventory
    a, b = case[0].shape
    caps = _caps(_plan(a, b, inventory)) if caps is None else caps
    return core._bounded_restricted_pair_ccsd_interaction_diagnostic(
        *case, inventory, caps, **kwargs
    )


def _decimal_oracle(case, source_transposed=False):
    # Independent FOUR-index Eq.26 sum, not the implementation's two GEMMs.
    # Riplinger/Neese doi:10.1063/1.4773581, II.C Eq.(26): explicit J[c,a]
    # and O[d,b] axes, no occupied-pair energy weights or disconnected terms.
    k, j, overlap, t = case
    if source_transposed:
        t = t.T
    a, b = k.shape
    output = np.zeros((a, a))
    with localcontext() as context:
        context.prec = 1600
        d = lambda x: Decimal.from_float(float(x))
        for left in range(a):
            for right in range(a):
                total = Decimal(0)
                for c in range(b):
                    for e in range(b):
                        total += (d(k[left, c]) - d(j[c, left]) / 2) * (
                            2 * d(t[c, e]) - d(t[e, c])
                        ) * d(overlap[e, right])
                output[left, right] = float(total)
    return output


@pytest.mark.parametrize("a", range(6))
@pytest.mark.parametrize("b", range(6))
@pytest.mark.parametrize("source_transposed", [False, True])
def test_exhaustive_tiny_rectangular_ranks_exact_inventory_and_decimal_equation(a, b, source_transposed):
    case = _case(a, b)
    result = _run(case, source_transposed=source_transposed)
    np.testing.assert_allclose(result.residual_copy(), _decimal_oracle(case, source_transposed), atol=3e-16, rtol=3e-14)
    p, d = result.memory, result.diagnostics
    assert (p.target_dimension, p.source_dimension) == (a, b)
    assert p.borrowed_input_elements == 3 * a * b + b * b
    assert p.borrowed_numerical_bytes == 8 * (3 * a * b + b * b)
    assert p.output_bytes == 8 * a * a
    assert p.intermediate_bytes == 8 * a * b
    assert p.peak_owned_numerical_bytes == 8 * (a * a + a * b)
    assert p.intermediate_contraction_terms == d.intermediate_contraction_terms == a * b * b
    assert p.residual_contraction_terms == d.residual_contraction_terms == a * a * b
    assert p.scalar_products == d.scalar_products == 2 * (a * b * b + a * a * b)
    assert p.input_scan_elements == 2 * (3 * a * b + b * b)
    assert p.output_scan_elements == a * a
    assert d.charged_work_units == p.work_units_upper_bound
    assert result.source_transposed is source_transposed
    assert not result.physical_source_certified
    if not a or not b:
        assert d.scalar_products == 0
        assert d.maximum_absolute_residual == 0


def test_nonnested_source_component_survives_but_target_first_projection_loses_it():
    u = np.eye(4)[:, :2]
    v = np.eye(4)[:, 1:3]
    k_common = np.zeros((4, 4))
    k_common[0, 2] = 1
    j_common = np.zeros((4, 4))
    t = np.array([[0.0, 0.0], [1.0, 0.0]])
    case = tuple(np.ascontiguousarray(x) for x in
                 (u.T @ k_common @ v, v.T @ j_common @ u, v.T @ u, t))
    expected = np.array([[0.0, 2.0], [0.0, 0.0]])
    np.testing.assert_array_equal(_run(case).residual_copy(), expected)
    common_t = v @ t @ v.T
    independent = u.T @ (k_common - 0.5 * j_common) @ (2 * common_t - common_t.T) @ u
    np.testing.assert_array_equal(independent, expected)
    bad_k = u.T @ k_common @ (u @ u.T @ v)
    bad = (bad_k - 0.5 * case[1].T) @ (2 * t - t.T) @ case[2]
    np.testing.assert_array_equal(bad, np.zeros((2, 2)))


def test_independent_frame_rotations_and_common_projection_oracle():
    rng = np.random.default_rng(3454)
    u = np.linalg.qr(rng.normal(size=(7, 2)))[0]
    v = np.linalg.qr(rng.normal(size=(7, 3)))[0]
    ka = rng.normal(size=(7, 7))
    ja = rng.normal(size=(7, 7))
    ja = (ja + ja.T) / 2
    t = rng.normal(scale=0.1, size=(3, 3))
    case = tuple(np.ascontiguousarray(x) for x in (u.T @ ka @ v, v.T @ ja @ u, v.T @ u, t))
    expected = u.T @ (ka - 0.5 * ja) @ (v @ (2 * t - t.T) @ v.T) @ u
    np.testing.assert_allclose(_run(case).residual_copy(), expected, atol=8e-16, rtol=3e-14)
    ra = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    rb = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    rotated = tuple(np.ascontiguousarray(x) for x in
                    (ra.T @ case[0] @ rb, rb.T @ case[1] @ ra, rb.T @ case[2] @ ra, rb.T @ t @ rb))
    np.testing.assert_allclose(_run(rotated).residual_copy(), ra.T @ expected @ ra, atol=2e-15, rtol=3e-14)


def test_reversed_source_transposes_only_amplitudes_bitwise():
    case = _case()
    reversed_case = (*case[:3], case[3].T.copy())
    normal = _run(case)
    reversed_result = _run(reversed_case, source_transposed=True)
    np.testing.assert_array_equal(normal.residual_copy(), reversed_result.residual_copy())
    assert normal.payload_identity_sha256 == reversed_result.payload_identity_sha256
    assert normal.input_payload_identity_sha256 != reversed_result.input_payload_identity_sha256
    assert np.max(np.abs(normal.residual_copy() - _run(case, source_transposed=True).residual_copy())) > 1e-4


@pytest.mark.parametrize("cap_field", list(_CAP_FIELDS))
def test_every_exact_cap_boundary_and_one_below_precedes_nonfinite_scan(cap_field):
    case = _case()
    plan = _plan(2, 3)
    _run(case, caps=_caps(plan))
    lowered = _caps(plan, **{cap_field: getattr(plan, _CAP_FIELDS[cap_field]) - 1})
    broken = tuple(x.copy() for x in case)
    broken[0][0, 0] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="cap exceeded"):
        _run(broken, caps=lowered)


def test_exact_other_owner_control_replica_inventory_and_alias_roles():
    live = _inventory(numerical_replicas=3, external_node_bytes=71,
                      other_live_numerical_bytes_per_replica=53,
                      other_live_control_bytes_per_replica=29,
                      backend_margin_bytes_per_replica=83)
    p = _plan(2, 2, live)
    assert p.total_control_storage_bytes == p.fixed_control_storage_bytes + 29
    assert p.per_replica_inventoried_bytes == p.peak_owned_numerical_bytes + p.borrowed_numerical_bytes + 53 + 83 + p.total_control_storage_bytes
    assert p.required_node_inventoried_bytes == 71 + 3 * p.per_replica_inventoried_bytes
    array = np.array([[0.2, -0.1], [0.3, 0.4]])
    result = _run((array, array, array, array), inventory=live)
    assert result.memory.borrowed_numerical_bytes == 4 * array.nbytes
    np.testing.assert_allclose(result.residual_copy(), _decimal_oracle((array,) * 4), atol=3e-17)


@pytest.mark.parametrize("field", ["numerical_replicas", "backend_margin_bytes_per_replica"])
def test_explicit_positive_inventory_required(field):
    with pytest.raises(ValueError, match="must be positive"):
        _plan(0, 0, _inventory(**{field: 0}))


@pytest.mark.parametrize("a,b", [(2**64 - 1, 0), (0, 2**64 - 1), (2**32, 2**32), (2**28, 1)])
def test_count_only_overflow_is_rejected_without_arrays(a, b):
    with pytest.raises(OverflowError, match="overflows|extent"):
        _plan(a, b)


@pytest.mark.parametrize("field", ["other_live_numerical_bytes_per_replica", "other_live_control_bytes_per_replica", "external_node_bytes"])
def test_inventory_overflow(field):
    with pytest.raises(OverflowError):
        _plan(1, 1, _inventory(**{field: 2**64 - 1}))


@pytest.mark.parametrize("slot", range(4))
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_all_input_roles_require_finite_values(slot, value):
    case = _case()
    case[slot][0, 0] = value
    with pytest.raises(ValueError, match="nonfinite"):
        _run(case)


def test_zero_target_still_validates_nonempty_source_amplitudes():
    case = _case(0, 2)
    case[3][0, 1] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        _run(case)


@pytest.mark.parametrize("fault", [1, 2, 3, 4])
def test_exact_native_view_extent_pointer_and_alignment_gates(fault):
    with pytest.raises(ValueError, match="view"):
        _run(_case(), view_fault=fault)


@pytest.mark.parametrize("fault", [1, 4])
def test_zero_views_require_exact_zero_extent(fault):
    with pytest.raises(ValueError, match="exact element extent"):
        _run(_case(0, 0), view_fault=fault)


@pytest.mark.parametrize("slot", range(4))
def test_diagnostic_rejects_conversion_and_shapes(slot):
    for kind in ("float32", "complex", "strided", "shape", "misaligned"):
        case = list(_case())
        original = case[slot]
        if kind == "float32":
            case[slot] = original.astype(np.float32)
        elif kind == "complex":
            case[slot] = original.astype(np.complex128)
        elif kind == "strided":
            case[slot] = np.ascontiguousarray(original.T).T
        elif kind == "shape":
            case[slot] = np.zeros((1, 1))
        else:
            backing = bytearray(original.nbytes + 1)
            case[slot] = np.ndarray(original.shape, dtype=np.float64, buffer=backing, offset=1)
        with pytest.raises(ValueError, match="binary64|shape"):
            _run(case, caps=_caps(_plan(2, 3)))


@pytest.mark.parametrize("fault", [1, 2, 3])
def test_non_nearest_environment_rejects_and_diagnostic_restores_it(fault):
    case = _case()
    before = _run(case).residual_copy()
    with pytest.raises(ValueError, match="nearest rounding"):
        _run(case, rounding_fault=fault)
    np.testing.assert_array_equal(_run(case).residual_copy(), before)


def test_compensated_cancellation_and_subnormal_decimal_witnesses():
    # K dot [1,1,1] with a cancellation that naive left-to-right summation loses.
    case = (np.array([[1e16, 1.0, -1e16]]), np.zeros((3, 1)), np.ones((3, 1)), np.eye(3))
    np.testing.assert_array_equal(_run(case).residual_copy(), _decimal_oracle(case))
    assert _run(case).residual(0, 0) == 1.0
    tiny = np.nextafter(0.0, 1.0)
    subnormal = (np.ones((1, 1)), np.zeros((1, 1)), np.ones((1, 1)), np.array([[tiny]]))
    np.testing.assert_array_equal(_run(subnormal).residual_copy(), _decimal_oracle(subnormal))
    underflow = (np.ones((1, 1)), np.zeros((1, 1)), np.array([[0.25]]), np.array([[tiny]]))
    result = _run(underflow)
    np.testing.assert_array_equal(result.residual_copy(), _decimal_oracle(underflow))
    assert result.diagnostics.product_underflow_count == 1


@pytest.mark.parametrize("slot", [0, 3])
def test_finite_inputs_with_overflowing_arithmetic_fail_without_result(slot):
    case = tuple(np.ones((1, 1)) for _ in range(4))
    case[1][:] = 0
    case[slot][:] = np.finfo(float).max
    if slot == 0:
        case[3][:] = 2
    with pytest.raises(OverflowError, match="arithmetic is nonfinite"):
        _run(case)


def _digest(domain, integers, arrays):
    h = hashlib.sha256()
    text = domain.encode()
    h.update(struct.pack(">Q", len(text)))
    h.update(text)
    h.update(struct.pack(">Q", 1))
    for x in integers:
        h.update(struct.pack(">Q", x))
    for array in arrays:
        for x in array.flat:
            h.update(struct.pack(">d", 0.0 if x == 0 else x))
    return h.hexdigest()


def test_receipt_wire_immutable_output_and_detached_owner_lifetime():
    case = _case()
    for x in case:
        x.setflags(write=False)
    result = _run(case)
    expected = result.residual_copy()
    assert result.input_payload_identity_sha256 == _digest(
        "vibeqc.bounded.pair-ccsd-interaction.input", [2, 3, 0], case
    )
    assert result.payload_identity_sha256 == _digest(
        "vibeqc.bounded.pair-ccsd-interaction.payload", [2, 3], [expected]
    )
    assert len(result.identity_sha256) == 64
    assert not expected.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        expected[0, 0] = 10
    copy = result.residual_copy()
    assert not np.shares_memory(copy, expected)
    del case
    gc.collect()
    np.testing.assert_array_equal(result.residual_copy(), expected)
    with pytest.raises(IndexError):
        result.residual(2, 0)


def test_signed_zero_source_payload_is_normalized():
    positive = tuple(np.zeros((1, 1)) for _ in range(4))
    negative = tuple(-x for x in positive)
    a, b = _run(positive), _run(negative)
    assert a.input_payload_identity_sha256 == b.input_payload_identity_sha256
    assert a.payload_identity_sha256 == b.payload_identity_sha256
    assert a.identity_sha256 == b.identity_sha256
