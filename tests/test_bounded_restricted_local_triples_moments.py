"""Published preprojected TNO moments, not external projection of full W.

All dense expansions and tensor transformations are tiny independent test
oracles. The numerical adapter never materializes common T1/T2 or ERIs.
"""

from __future__ import annotations

import hashlib
import itertools
import struct

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_bounded_restricted_pair_ccsd_amplitudes import _ragged, _dense, _arguments
from tests.test_bounded_restricted_triples_moments import _einsum_oracle, _run as _old_moments


def _case(o=2, n=3, r=2, *, uniform=False, seed=812):
    rng = np.random.default_rng(seed)
    pairs = o * (o + 1) // 2
    sr = (n,) * o if uniform else tuple((i + 1) % (n + 1) for i in range(o))
    pr = (n,) * pairs if uniform else tuple((i + 2) % (n + 1) for i in range(pairs))
    b = _ragged(n, sr, pr, seed=seed)
    c = np.linalg.qr(rng.normal(size=(n, n)))[0][:, :r]
    factors = .07 * rng.normal(size=(4, o + n, o + n))
    factors += factors.transpose(0, 2, 1)
    b.update(c=np.ascontiguousarray(c), fov=np.zeros((o, n)),
             eri=np.ascontiguousarray(np.einsum("Ppq,Prs->pqrs", factors, factors)))
    return b


def _options():
    options = core._BoundedRestrictedLocalTriplesMomentsOptions()
    options.coefficient_orthogonality_tolerance = 1e-12
    options.amplitude_symmetry_tolerance = 0
    options.maximum_integral_work_units_per_call = 1
    return options


def _inventory():
    inventory = core._BoundedRestrictedLocalTriplesMomentsInventory()
    inventory.numerical_replicas = 1
    inventory.fixed_backend_margin_bytes_per_replica = 65536
    return inventory


def _upper(case, *, options=None, inventory=None, transient=0):
    return core._plan_bounded_restricted_local_triples_moments_upper(
        case["o"], case["n"], case["c"].shape[1], max(t.size for t in case["st"]),
        max(t.shape[0] for t in case["pt"]), case["eri"].nbytes, transient,
        _options() if options is None else options, _inventory() if inventory is None else inventory)


def _caps(plan):
    caps = core._BoundedRestrictedLocalTriplesMomentsCaps()
    for field, value in {
        "maximum_occupied_count": plan.n_occupied,
        "maximum_common_virtual_dimension": plan.common_virtual_dimension,
        "maximum_rank": plan.rank,
        "maximum_owned_numerical_bytes": plan.peak_owned_numerical_bytes,
        "maximum_per_replica_inventoried_bytes": plan.per_replica_inventoried_bytes,
        "maximum_node_inventoried_bytes": plan.required_node_inventoried_bytes,
        "maximum_common_singles_calls": plan.common_singles_calls,
        "maximum_common_doubles_calls": plan.common_doubles_calls,
        "maximum_transformed_integral_calls": plan.transformed_integral_calls,
        "maximum_common_integral_calls": plan.common_integral_calls,
        "maximum_work_units": plan.work_units_upper_bound,
    }.items():
        setattr(caps, field, value)
    return caps


def _run(case, occupied=None, *, options=None, inventory=None, caps=None, snapshot=59, **kwargs):
    options = _options() if options is None else options
    inventory = _inventory() if inventory is None else inventory
    if occupied is None:
        occupied = (0, min(1, case["o"] - 1), 0)
    if caps is None:
        caps = _caps(_upper(case, options=options, inventory=inventory,
                           transient=kwargs.get("provider_transient_bytes", 0)))
    return core._bounded_restricted_local_triples_moments_diagnostic(
        *_arguments(case), case["c"], case["fov"], case["eri"], occupied, snapshot,
        options, inventory, caps, **kwargs)


def _projected(case):
    o, n, c = case["o"], case["n"], case["c"]
    r = c.shape[1]
    t1, t2 = _dense(case)
    transform = np.zeros((o + n, o + r))
    transform[:o, :o] = np.eye(o)
    transform[o:, o:] = c
    eri = np.einsum("pP,qQ,rR,sS,pqrs->PQRS", transform, transform, transform, transform,
                    case["eri"], optimize=True)
    return dict(o=o, v=r, t1=np.ascontiguousarray(t1 @ c),
        t2=np.ascontiguousarray(np.einsum("xa,ijxy,yb->ijab", c, t2, c)),
        eri=np.ascontiguousarray(eri), fov=np.zeros((o, r)))


@pytest.mark.parametrize("o,n,r", [(1, 1, 1), (2, 3, 1), (2, 3, 2), (2, 3, 3),
                                  (3, 3, 2), (2, 4, 3), (4, 2, 2)])
def test_native_scalar_preprojection_matches_independent_dense_and_existing_moment_leaf(o, n, r):
    case = _case(o, n, r)
    occupied = (o - 1, 0, min(1, o - 1))
    result = _run(case, occupied)
    projected = _projected(case)
    w, u = _einsum_oracle(projected, occupied)
    old = _old_moments(projected, occupied, symmetry=1e-12)
    np.testing.assert_allclose(result.connected, w, rtol=5e-12, atol=3e-16)
    np.testing.assert_allclose(result.singles, u, rtol=5e-12, atol=2e-16)
    np.testing.assert_allclose(result.connected, old.connected, rtol=5e-12, atol=3e-16)
    np.testing.assert_allclose(result.singles, old.singles, rtol=5e-12, atol=2e-16)
    assert result.moments.maximum_consumed_amplitude_symmetry_error == 0
    assert result.moments.moments.amplitude_snapshot_id == 59
    assert tuple(result.moments.moments.occupied) == occupied
    assert result.coefficient_orthogonality_error < 1e-12
    for identity in (result.reader_snapshot_identity_sha256, result.target_coefficients_identity_sha256,
                     result.input_identity_sha256, result.consumed_integral_receipt_sha256,
                     result.payload_identity_sha256):
        assert len(identity) == 64 and int(identity, 16) >= 0


def test_exact_truncated_internal_d_witness_is_not_external_projection_of_common_w():
    case = dict(o=1, n=2, sc=[np.eye(2)], st=[np.zeros(2)], pc=[np.eye(2)],
        pt=[np.array([[0., 1.], [1., 0.]])], c=np.array([[1.], [0.]]), fov=np.zeros((1, 2)),
        eri=np.zeros((3, 3, 3, 3)))
    for p, q in ((0, 1), (1, 0)):
        for r, s in ((1, 2), (2, 1)):
            case["eri"][p, q, r, s] = case["eri"][r, s, p, q] = 1
    result = _run(case, (0, 0, 0))
    np.testing.assert_array_equal(result.connected, 0)
    projected = _projected(case)
    np.testing.assert_array_equal(_einsum_oracle(projected, (0, 0, 0))[0], 0)
    t1, t2 = _dense(case)
    common = dict(o=1, v=2, t1=t1, t2=t2, eri=case["eri"], fov=case["fov"])
    w, _ = _einsum_oracle(common, (0, 0, 0))
    external = np.einsum("xa,yb,zc,xyz->abc", case["c"], case["c"], case["c"], w)
    np.testing.assert_array_equal(external, [[[6.0]]])
    case["c"] = np.eye(2)
    np.testing.assert_array_equal(_run(case, (0, 0, 0)).connected, w)


def test_query_census_and_owned_inventory_are_exact_without_hidden_common_tensors():
    case = _case(uniform=True)
    inventory = _inventory()
    inventory.numerical_replicas = 2
    inventory.other_live_bytes_per_replica = 113
    inventory.external_node_bytes = 29
    result = _run(case, inventory=inventory, provider_transient_bytes=17)
    p = result.memory
    o, n, r = case["o"], case["n"], case["c"].shape[1]
    assert p.projected_zero_f_ov_bytes == 8 * o * r
    assert p.output_numerical_bytes == 16 * r**3
    assert p.peak_owned_numerical_bytes == 8 * o * r + 16 * r**3
    assert p.common_singles_calls == result.common_singles_calls == 3 * r**3 * n
    assert p.common_doubles_calls == result.common_doubles_calls == 12 * r**3 * (o + r) * n**2
    assert p.transformed_integral_calls == result.transformed_integral_calls == 6 * r**3 * (o + r) + 3 * r**3
    assert p.one_virtual_integral_calls == 6 * o * r**3
    assert p.two_virtual_integral_calls == 3 * r**3
    assert p.three_virtual_integral_calls == 6 * r**4
    assert p.common_integral_calls == result.common_integral_calls == 6*r**4*n**3 + 6*o*r**3*n + 3*r**3*n**2
    assert p.maximum_common_integral_calls_per_transformed_query == n**4
    assert p.per_replica_inventoried_bytes == (p.peak_owned_numerical_bytes
        + p.reader_borrowed_numerical_bytes + case["c"].nbytes + case["fov"].nbytes + case["eri"].nbytes
        + 17 + p.control_storage_reservation_bytes + 113 + inventory.fixed_backend_margin_bytes_per_replica)
    assert p.required_node_inventoried_bytes == 29 + 2 * p.per_replica_inventoried_bytes
    upper = _upper(case, inventory=inventory, transient=17)
    assert upper.uniform_rank_upper_bound and not p.uniform_rank_upper_bound
    for name in ("peak_owned_numerical_bytes", "per_replica_inventoried_bytes", "required_node_inventoried_bytes",
                 "work_units_upper_bound", "reader_borrowed_numerical_bytes"):
        assert getattr(upper, name) == getattr(p, name)


def test_ragged_upper_bound_dominates_exact_plan_and_plan_has_no_integral_calls():
    case = _case()
    upper = _upper(case)
    actual = _run(case, planning=True, fail_before_call=0)
    assert actual.reader_borrowed_numerical_bytes < upper.reader_borrowed_numerical_bytes
    assert actual.work_units_upper_bound <= upper.work_units_upper_bound
    assert actual.required_node_inventoried_bytes <= upper.required_node_inventoried_bytes


@pytest.mark.parametrize("field", ["maximum_occupied_count", "maximum_common_virtual_dimension", "maximum_rank",
    "maximum_owned_numerical_bytes", "maximum_per_replica_inventoried_bytes", "maximum_node_inventoried_bytes",
    "maximum_common_singles_calls", "maximum_common_doubles_calls", "maximum_transformed_integral_calls",
    "maximum_common_integral_calls", "maximum_work_units"])
@pytest.mark.parametrize("zero", [False, True])
def test_caps_precede_reader_floating_scan_and_any_integral_callback(field, zero):
    case = _case(n=2, r=2, uniform=True)
    caps = _caps(_upper(case))
    setattr(caps, field, 0 if zero else getattr(caps, field) - 1)
    case["sc"][0][0, 0] = np.nan
    case["c"][0, 0] = case["fov"][0, 0] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(case, caps=caps, fail_before_call=0)


def test_original_eri_transient_inventory_cannot_hide():
    case = _case(uniform=True)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(case, caps=_caps(_upper(case)), provider_transient_bytes=1)
    assert _run(case, provider_transient_bytes=1).common_integral_calls > 0


@pytest.mark.parametrize("value", [1e-8, -1e-8, np.nextafter(0., 1.), -np.nextafter(0., 1.)])
def test_nonzero_original_fov_is_rejected_even_when_its_tno_projection_is_zero(value):
    case = _case(n=2, r=1)
    case["c"] = np.array([[1.], [0.]])
    case["fov"][0, 1] = value
    np.testing.assert_array_equal(case["fov"] @ case["c"], 0)
    with pytest.raises(ValueError, match="exactly zero ORIGINAL Fov"):
        _run(case, fail_before_call=0)


@pytest.mark.parametrize("field", ["c", "fov", "eri"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_inputs_and_consumed_eris_abort(field, value):
    case = _case()
    case[field][:] = value
    with pytest.raises((ValueError, OverflowError, RuntimeError), match="finite"):
        _run(case)


@pytest.mark.parametrize("last", [False, True])
def test_original_integral_callback_exceptions_publish_no_result(last):
    case = _case()
    saved = case["c"].copy()
    count = _upper(case).common_integral_calls
    with pytest.raises(RuntimeError, match="injected integral failure"):
        _run(case, fail_before_call=count - 1 if last else 0)
    np.testing.assert_array_equal(case["c"], saved)
    assert _run(case).common_integral_calls == count


def test_python_diagnostic_callback_exception_is_not_swallowed():
    def fail():
        raise RuntimeError("projected-moment callback witness")
    with pytest.raises(RuntimeError, match="callback witness"):
        _run(_case(), callback=fail)


@pytest.mark.parametrize("field", ["c", "fov", "st", "pt"])
def test_complete_adapter_revalidates_reader_and_original_frame_after_external_callbacks(field):
    case = _case(uniform=True)
    def mutate():
        if field == "c":
            case["c"][:, 0] *= -1 # Gram-preserving change must still fail identity.
        elif field == "fov":
            case["fov"][0, 0] = .1
        elif field == "st":
            case["st"][0][0] += .1
        else:
            case["pt"][0][0, 0] += .1
    with pytest.raises(ValueError, match="immutable|ORIGINAL Fov"):
        _run(case, callback=mutate)


def test_pinned_array_owners_and_copied_native_controls_survive_diagnostic_callback():
    case = _case()
    expected = _run(case)
    def replace_lists():
        for field in ("sc", "st", "pc", "pt"):
            case[field][:] = [np.full_like(value, np.nan) for value in case[field]]
    result = _run(case, callback=replace_lists, mutate_control_objects=True)
    np.testing.assert_array_equal(result.connected.view(np.uint64), expected.connected.view(np.uint64))
    np.testing.assert_array_equal(result.singles.view(np.uint64), expected.singles.view(np.uint64))
    assert result.payload_identity_sha256 == expected.payload_identity_sha256
    w = result.connected
    result.connected[:] = 42
    nested = result.moments
    del result, case
    np.testing.assert_array_equal(nested.moments.connected, w)


@pytest.mark.parametrize("fault", [1, 2])
def test_floating_environment_gates_entry_and_original_eri_callback(fault):
    case = _case()
    expected = _run(case)
    with pytest.raises(RuntimeError, match="nearest binary64"):
        _run(case, floating_environment_fault=fault)
    np.testing.assert_array_equal(_run(case).connected.view(np.uint64), expected.connected.view(np.uint64))


def test_within_retained_tno_rotation_transforms_all_external_moment_axes():
    case = _case()
    original = _run(case)
    q = np.array([[.6, -.8], [.8, .6]])
    case["c"] = np.ascontiguousarray(case["c"] @ q)
    rotated = _run(case)
    for name in ("connected", "singles"):
        expected = np.einsum("xa,yb,zc,xyz->abc", q, q, q, getattr(original, name))
        np.testing.assert_allclose(getattr(rotated, name), expected, atol=4e-16, rtol=5e-12)
    assert rotated.moments.maximum_consumed_amplitude_symmetry_error == 0


def test_reversed_occupied_triples_permute_the_same_three_virtual_axes():
    case = _case(o=3, n=2, r=2)
    occupied = (0, 1, 2)
    original = _run(case, occupied)
    for order in itertools.permutations(range(3)):
        result = _run(case, tuple(occupied[i] for i in order))
        np.testing.assert_allclose(result.connected, original.connected.transpose(order), atol=4e-16, rtol=5e-12)
        np.testing.assert_allclose(result.singles, original.singles.transpose(order), atol=3e-16, rtol=5e-12)
        assert result.moments.maximum_consumed_amplitude_symmetry_error == 0


def test_common_virtual_basis_rotation_preserves_physical_selected_tno_moments():
    case = _case()
    original = _run(case)
    q = np.linalg.qr(np.random.default_rng(962).normal(size=(case["n"], case["n"])))[0]
    for key in ("sc", "pc"):
        case[key] = [np.ascontiguousarray(q.T @ c) for c in case[key]]
    case["c"] = np.ascontiguousarray(q.T @ case["c"])
    x = np.eye(case["o"] + case["n"])
    x[case["o"]:, case["o"]:] = q
    case["eri"] = np.ascontiguousarray(np.einsum("pP,qQ,rR,sS,pqrs->PQRS", x, x, x, x, case["eri"], optimize=True))
    rotated = _run(case)
    np.testing.assert_allclose(rotated.connected, original.connected, atol=4e-16, rtol=5e-12)
    np.testing.assert_allclose(rotated.singles, original.singles, atol=3e-16, rtol=5e-12)


def test_no_integral_symmetry_averaging_is_inferred_from_the_numerical_provider():
    case = _case()
    case["eri"] = np.random.default_rng(402).normal(size=case["eri"].shape)
    result = _run(case)
    w, u = _einsum_oracle(_projected(case), (0, 1, 0))
    np.testing.assert_allclose(result.connected, w, atol=4e-15, rtol=5e-12)
    np.testing.assert_allclose(result.singles, u, atol=3e-15, rtol=5e-12)


def test_actual_original_integral_receipt_has_independent_sequence_digest():
    case = _case(1, 1, 1, uniform=True)
    result = _run(case, (0, 0, 0))
    h = hashlib.sha256(b"vibeqc.local-tno-moments.consumed-common-integrals" + struct.pack(">Q", 1))
    sequence = [(0, 1, 1, 1), (0, 1, 0, 0)] * 6 + [(0, 1, 0, 1)] * 3
    for labels in sequence:
        for label in labels:
            h.update(struct.pack(">Q", label))
        value = case["eri"][labels]
        h.update(struct.pack(">d", 0.0 if value == 0 else value))
    assert result.consumed_integral_receipt_sha256 == h.hexdigest()
    assert result.common_integral_calls == 15


@pytest.mark.parametrize("field,value", [("coefficient_orthogonality_tolerance", 0),
    ("coefficient_orthogonality_tolerance", 1), ("coefficient_orthogonality_tolerance", np.nan),
    ("amplitude_symmetry_tolerance", -1), ("amplitude_symmetry_tolerance", np.nan),
    ("maximum_integral_work_units_per_call", 0)])
def test_invalid_explicit_controls_reject(field, value):
    case = _case()
    options = _options()
    setattr(options, field, value)
    with pytest.raises(ValueError, match="tolerance|positive|declaration"):
        _run(case, options=options, caps=_caps(_upper(case)), fail_before_call=0)


@pytest.mark.parametrize("snapshot,occupied", [(0, (0, 1, 0)), (59, (2, 0, 0))])
def test_snapshot_and_occupied_labels_are_validated(snapshot, occupied):
    with pytest.raises((ValueError, IndexError), match="snapshot|occupied label"):
        _run(_case(), snapshot=snapshot, occupied=occupied, fail_before_call=0)


def test_nonorthonormal_frame_is_not_repaired():
    case = _case()
    case["c"][:, 0] *= 1.1
    with pytest.raises(ValueError, match="not orthonormal"):
        _run(case, fail_before_call=0)


@pytest.mark.parametrize("o,n,r,s,p", [(0, 2, 1, 1, 1), (1, 0, 1, 0, 0), (1, 2, 0, 1, 1),
    (1, 2, 3, 1, 1), (1, 2, 1, 3, 1), (2**63, 2, 1, 1, 1), (1, 2**32, 1, 0, 0)])
def test_upper_planner_rejects_bad_or_overflowing_counts_without_fake_reader(o, n, r, s, p):
    with pytest.raises((ValueError, OverflowError)):
        core._plan_bounded_restricted_local_triples_moments_upper(o, n, r, s, p, 0, 0, _options(), _inventory())


def test_zero_rank_amplitude_sources_do_not_skip_projected_or_integral_queries():
    case = _case()
    case.update(_ragged(case["n"], (0,) * case["o"], (0,) * len(case["pt"])))
    result = _run(case)
    np.testing.assert_array_equal(result.connected, 0)
    np.testing.assert_array_equal(result.singles, 0)
    assert result.common_integral_calls == result.memory.common_integral_calls > 0
    assert result.common_doubles_calls == result.memory.common_doubles_calls > 0
