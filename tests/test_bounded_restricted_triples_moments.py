"""Native local W/U moments with independent tensor and spin-orbital oracles."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_bounded_restricted_triples_target import _run as _target


def _case(o=3, v=3, seed=731):
    rng = np.random.default_rng(seed)
    raw = 0.1 * rng.normal(size=(4, o + v, o + v))
    factors = np.ascontiguousarray(raw + raw.transpose(0, 2, 1))
    raw_t2 = 0.07 * rng.normal(size=(o, o, v, v))
    return dict(o=o, v=v, factors=factors,
                eri=np.ascontiguousarray(np.einsum("Ppq,Prs->pqrs", factors, factors)),
                t1=np.ascontiguousarray(0.04 * rng.normal(size=(o, v))),
                t2=np.ascontiguousarray(raw_t2 + raw_t2.transpose(1, 0, 3, 2)),
                fov=np.zeros((o, v)))


def _plan(case, transient=0):
    return core._plan_bounded_restricted_triples_moments(
        case["o"], case["v"], case["eri"].nbytes, transient)


def _caps(plan):
    caps = core._BoundedRestrictedTriplesMomentsCaps()
    caps.maximum_owned_numerical_bytes = plan.peak_owned_numerical_bytes
    caps.maximum_total_numerical_bytes = plan.total_live_numerical_bytes
    caps.maximum_integral_calls = plan.integral_calls
    caps.maximum_kernel_work_units = plan.kernel_work_units_upper_bound
    return caps


def _run(case, occupied=(0, 1, 2), caps=None, snapshot=73, symmetry=1e-13, **kwargs):
    caps = _caps(_plan(case, kwargs.get("provider_transient_bytes", 0))) if caps is None else caps
    return core._bounded_restricted_triples_moments_diagnostic(
        case["t1"], case["t2"], case["fov"], case["eri"], occupied,
        snapshot, symmetry, caps, **kwargs)


def _einsum_oracle(case, occupied):
    o, v, t1, t2, eri = (case[k] for k in ("o", "v", "t1", "t2", "eri"))
    w = np.zeros((v, v, v))
    for order in itertools.permutations(range(3)):
        i, j, k = (occupied[axis] for axis in order)
        base = (np.einsum("cd,abd->abc", t2[k, j], eri[i, o:, o:, o:])
                - np.einsum("lab,cl->abc", t2[i], eri[k, o:, j, :o]))
        w += base.transpose(np.argsort(order))
    i, j, k = occupied
    u = (np.einsum("a,bc->abc", t1[i], eri[j, o:, k, o:])
         + np.einsum("b,ac->abc", t1[j], eri[i, o:, k, o:])
         + np.einsum("c,ba->abc", t1[k], eri[j, o:, i, o:]))
    return w, u


@pytest.mark.parametrize("o,v,occupied", [(1, 1, (0, 0, 0)), (1, 3, (0, 0, 0)),
    (2, 2, (0, 0, 1)), (3, 3, (0, 1, 2)), (3, 2, (2, 0, 1)), (4, 3, (1, 3, 1))])
def test_moments_match_independent_tensor_contractions(o, v, occupied):
    case = _case(o, v)
    result = _run(case, occupied)
    w, u = _einsum_oracle(case, occupied)
    np.testing.assert_allclose(result.connected, w, atol=3e-16, rtol=3e-13)
    np.testing.assert_allclose(result.singles, u, atol=1e-16, rtol=3e-13)
    assert tuple(result.occupied) == occupied
    assert result.amplitude_snapshot_id == 73
    assert result.integral_calls == result.memory.integral_calls == v**3 * (6 * (o + v) + 3)


@pytest.mark.parametrize("occupied", [(0, 1, 2), (1, 1, 2), (1, 1, 1)])
def test_all_six_simultaneous_axis_permutations(occupied):
    case = _case()
    original = _run(case, occupied)
    for order in itertools.permutations(range(3)):
        permuted = _run(case, tuple(occupied[axis] for axis in order))
        np.testing.assert_allclose(permuted.connected, original.connected.transpose(order), atol=3e-16)
        np.testing.assert_allclose(permuted.singles, original.singles.transpose(order), atol=1e-16)


def test_repeated_labels_keep_six_terms_and_no_energy_multiplicity():
    case = _case(1, 1)
    case["t1"][:] = 2
    case["t2"][:] = 3
    case["eri"][:] = 0

    def set_integral(p, q, r, s, value):
        for a, b in ((p, q), (q, p)):
            for c, d in ((r, s), (s, r)):
                case["eri"][a, b, c, d] = case["eri"][c, d, a, b] = value

    set_integral(0, 1, 1, 1, 5)
    set_integral(0, 1, 0, 0, 7)
    set_integral(0, 1, 0, 1, 11)
    result = _run(case, (0, 0, 0))
    np.testing.assert_array_equal(result.connected, [[[-36.0]]])
    np.testing.assert_array_equal(result.singles, [[[66.0]]])
    assert result.integral_calls == 15


def test_native_moments_and_operator_reproduce_spin_orbital_and_canonical_triples_energy():
    from vibeqc.dlpno._ccsd_ref import _spin_orbital_eri, so_triples_correction

    case = _case(2, 3, seed=819)
    o, v = case["o"], case["v"]
    eo, ev = np.array([-1.3, -0.9]), np.array([0.3, 0.7, 1.2])
    energy = 0.0
    for occupied in itertools.combinations_with_replacement(range(o), 3):
        moments = _run(case, occupied)
        w, u = moments.connected, moments.singles
        gap = ev[:, None, None] + ev[None, :, None] + ev[None, None, :] - sum(eo[index] for index in occupied)
        leaf = _target(dict(o=o, v=v, dmax=v, occupied=occupied,
            amplitudes=np.ascontiguousarray(-w / gap), connected=w, singles=u,
            energies=ev, fock_rows=np.ascontiguousarray(np.diag(eo)[list(occupied)]),
            sources=[None] * (3 * o), overlaps=[None] * (3 * o)))
        np.testing.assert_allclose(leaf.residual, 0, atol=2e-16)
        i, j, k = occupied
        energy += (2 - int(i == j) - int(j == k)) * leaf.raw_energy_contraction
    t1so = np.zeros((2 * o, 2 * v))
    t1so[0::2, 0::2] = t1so[1::2, 1::2] = case["t1"]
    t2so = np.zeros((2 * o, 2 * o, 2 * v, 2 * v))
    for si, sj in itertools.product((0, 1), repeat=2):
        t2so[si::2, sj::2, si::2, sj::2] += case["t2"]
        t2so[si::2, sj::2, sj::2, si::2] -= case["t2"].transpose(0, 1, 3, 2)
    spin = so_triples_correction(np.repeat(np.r_[eo, ev], 2), _spin_orbital_eri(case["factors"], o + v),
                                t1so, t2so, slice(0, 2 * o), slice(2 * o, 2 * (o + v)))
    assert energy == pytest.approx(spin, rel=3e-12, abs=2e-14)
    p, factors = len(case["factors"]), case["factors"]
    canonical = core.dlpno_spatial_triples_correction(
        case["t1"], case["t2"].reshape(o * o, v * v),
        np.ascontiguousarray(factors[:, :o, o:].reshape(p, o * v)),
        np.ascontiguousarray(factors[:, :o, :o].reshape(p, o * o)),
        np.ascontiguousarray(factors[:, o:, o:].reshape(p, v * v)), eo, ev)
    assert energy == pytest.approx(canonical, rel=3e-12, abs=2e-14)


def test_exact_output_only_heap_and_borrowed_provider_inventory():
    o, v = 3, 4
    p = core._plan_bounded_restricted_triples_moments(o, v, 123, 456)
    assert p.connected_output_bytes == p.singles_output_bytes == 8 * v**3
    assert p.peak_owned_numerical_bytes == 16 * v**3
    assert p.borrowed_input_bytes == 8 * (2 * o * v + o**2 * v**2)
    assert p.total_live_numerical_bytes == p.peak_owned_numerical_bytes + p.borrowed_input_bytes + 123 + 456
    assert p.integral_calls == v**3 * (6 * (o + v) + 3)
    assert p.kernel_work_units_upper_bound == 2 * o * v + 2 * o**2 * v**2 + 128 * p.integral_calls + 64 * v**3 + 256


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_total_numerical_bytes",
                                  "maximum_integral_calls", "maximum_kernel_work_units"])
@pytest.mark.parametrize("zero", [False, True])
def test_all_caps_precede_scans_and_provider(field, zero):
    case = _case()
    caps = _caps(_plan(case))
    setattr(caps, field, 0 if zero else getattr(caps, field) - 1)
    case["t1"][0, 0] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(case, caps=caps, fail_before_call=0)


def test_provider_transient_bytes_cannot_hide():
    case = _case()
    with pytest.raises((ValueError, RuntimeError), match="byte cap"):
        _run(case, caps=_caps(_plan(case)), provider_transient_bytes=1)
    assert _run(case, provider_transient_bytes=1).integral_calls > 0


@pytest.mark.parametrize("fov", [0.1, -0.1, np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0)])
def test_even_subnormal_nonzero_fov_rejects_general_non_hf_scope(fov):
    case = _case()
    case["fov"][0, 0] = fov
    with pytest.raises(ValueError, match="exactly zero Fov"):
        _run(case, fail_before_call=0)


@pytest.mark.parametrize("field", ["t1", "t2", "fov"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_all_input_lanes_are_finite_scanned(field, bad):
    case = _case()
    case[field].flat[-1] = bad
    with pytest.raises(ValueError, match="finite"):
        _run(case, fail_before_call=0)


def test_asymmetric_t2_rejects_instead_of_averaging():
    case = _case()
    case["t2"][0, 1, 0, 1] += 0.2
    with pytest.raises(ValueError, match="symmetry audit"):
        _run(case, fail_before_call=0)


@pytest.mark.parametrize("failure", [0, 1, 50, 700])
def test_provider_failure_no_partial_output_or_input_mutation(failure):
    case = _case()
    saved = {key: case[key].copy() for key in ("t1", "t2", "fov", "eri")}
    with pytest.raises(RuntimeError, match="injected integral callback failure"):
        _run(case, fail_before_call=failure)
    for key, data in saved.items():
        np.testing.assert_array_equal(case[key], data)
    result = _run(case)
    w, u = result.connected.copy(), result.singles.copy()
    result.connected[:] = 0
    result.singles[:] = 0
    del case
    np.testing.assert_array_equal(result.connected, w)
    np.testing.assert_array_equal(result.singles, u)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_provider_values_abort(bad):
    case = _case()
    case["eri"][:] = bad
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="finite"):
        _run(case)


def test_finite_inputs_whose_product_overflows_abort():
    case = _case()
    case["t2"][:] = np.finfo(float).max
    case["eri"][:] = 2
    with pytest.raises(OverflowError, match="finite"):
        _run(case)


def test_zero_amplitudes_do_not_skip_integral_calls():
    case = _case()
    case["t1"][:] = case["t2"][:] = 0
    result = _run(case)
    np.testing.assert_array_equal(result.connected, 0)
    np.testing.assert_array_equal(result.singles, 0)
    assert result.integral_calls == result.memory.integral_calls


@pytest.mark.parametrize("occupied", [(3, 0, 0), (0, 3, 0), (0, 0, 3)])
def test_ordered_labels_must_be_local_and_active(occupied):
    with pytest.raises(IndexError, match="out of range"):
        _run(_case(), occupied=occupied, fail_before_call=0)


@pytest.mark.parametrize("kwargs", [dict(snapshot=0), dict(symmetry=-1), dict(symmetry=np.nan)])
def test_sequence_and_symmetry_options_precede_provider(kwargs):
    with pytest.raises(ValueError, match="snapshot|symmetry"):
        _run(_case(), fail_before_call=0, **kwargs)


@pytest.mark.parametrize("o,v", [(0, 1), (1, 0), (2**63, 2), (1, 2**63)])
def test_checked_plan_rejects_bad_dimensions(o, v):
    with pytest.raises((ValueError, OverflowError, RuntimeError)):
        core._plan_bounded_restricted_triples_moments(o, v)


def _accessor_plan(case, amplitude_transient_bytes=0, integral_transient_bytes=0,
                   singles_work_units=1, doubles_work_units=1):
    return core._plan_bounded_restricted_triples_moments_accessor(
        case["o"], case["v"], case["t1"].nbytes + case["t2"].nbytes,
        amplitude_transient_bytes, singles_work_units, doubles_work_units,
        case["eri"].nbytes, integral_transient_bytes)


def _accessor_caps(plan):
    caps = core._BoundedRestrictedTriplesMomentsAccessorCaps()
    caps.kernel = _caps(plan.kernel)
    caps.maximum_singles_calls = plan.singles_calls
    caps.maximum_doubles_calls = plan.doubles_calls
    caps.maximum_amplitude_work_units = plan.amplitude_work_units_upper_bound
    caps.maximum_control_storage_bytes = plan.fixed_control_storage_bytes
    return caps


def _accessor_run(case, occupied=(0, 1, 2), caps=None, snapshot=73, symmetry=1e-13, **kwargs):
    if caps is None:
        plan_keys = ("amplitude_transient_bytes", "integral_transient_bytes",
                     "singles_work_units", "doubles_work_units")
        caps = _accessor_caps(_accessor_plan(case, **{k: kwargs[k] for k in plan_keys if k in kwargs}))
    return core._bounded_restricted_triples_moments_accessor_diagnostic(
        case["t1"], case["t2"], case["fov"], case["eri"], occupied,
        snapshot, symmetry, caps, **kwargs)


@pytest.mark.parametrize("o,v", list(itertools.product(range(1, 5), repeat=2)) + [(3, 5)])
def test_accessor_all_ordered_tiny_triples_are_bitwise_dense_with_exact_query_counts(o, v):
    case = _case(o, v)
    for occupied in itertools.product(range(o), repeat=3):
        dense = _run(case, occupied)
        result = _accessor_run(case, occupied)
        for field in ("connected", "singles"):
            np.testing.assert_array_equal(getattr(result.moments, field).view(np.uint64),
                                          getattr(dense, field).view(np.uint64))
        d, s = 6 * v**3 * (o + v), 3 * v**3
        assert result.singles_calls == result.memory.singles_calls == s
        assert result.doubles_calls == result.memory.doubles_calls == 2 * d
        assert result.reverse_symmetry_audits == result.memory.contraction_doubles_calls == d
        assert result.memory.reverse_audit_doubles_calls == d
        assert result.charged_amplitude_work_units == s + 2 * d
        assert result.moments.integral_calls == dense.integral_calls == d + s
        assert result.maximum_consumed_amplitude_symmetry_error == 0
        assert result.moments.amplitude_snapshot_id == 73
        assert tuple(result.moments.occupied) == occupied


def test_accessor_independent_tensor_oracle_and_declared_work_inventory():
    case = _case(3, 4)
    result = _accessor_run(case, (2, 0, 1), amplitude_transient_bytes=17,
                           integral_transient_bytes=29, singles_work_units=7, doubles_work_units=11)
    p, k = result.memory, result.memory.kernel
    w, u = _einsum_oracle(case, (2, 0, 1))
    np.testing.assert_allclose(result.moments.connected, w, atol=3e-16, rtol=3e-13)
    np.testing.assert_allclose(result.moments.singles, u, atol=1e-16, rtol=3e-13)
    assert k.borrowed_input_bytes == 8 * 3 * 4
    assert k.peak_owned_numerical_bytes == 16 * 4**3
    assert k.connected_output_bytes == k.singles_output_bytes == 8 * 4**3
    assert p.amplitude_retained_numerical_bytes == case["t1"].nbytes + case["t2"].nbytes
    assert p.amplitude_maximum_transient_numerical_bytes == 17
    assert k.provider_retained_numerical_bytes == case["eri"].nbytes
    assert k.provider_maximum_transient_numerical_bytes == 29
    assert k.total_live_numerical_bytes == (k.peak_owned_numerical_bytes + k.borrowed_input_bytes
        + p.amplitude_retained_numerical_bytes + case["eri"].nbytes + 17 + 29)
    assert result.charged_amplitude_work_units == p.amplitude_work_units_upper_bound == 7 * p.singles_calls + 11 * p.doubles_calls
    assert k.kernel_work_units_upper_bound == 12 + 128 * k.integral_calls + 64 * 4**3 + 256 + 64 * (p.singles_calls + p.doubles_calls)
    assert 4096 < p.fixed_control_storage_bytes <= 65536


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_total_numerical_bytes",
    "maximum_integral_calls", "maximum_kernel_work_units", "maximum_singles_calls",
    "maximum_doubles_calls", "maximum_amplitude_work_units", "maximum_control_storage_bytes"])
@pytest.mark.parametrize("zero", [False, True])
def test_accessor_caps_precede_all_scans_and_callbacks(field, zero):
    case = _case()
    caps = _accessor_caps(_accessor_plan(case))
    owner = caps.kernel if hasattr(caps.kernel, field) else caps
    setattr(owner, field, 0 if zero else getattr(owner, field) - 1)
    case["fov"][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _accessor_run(case, caps=caps, fail_before_singles=0, fail_before_doubles=0, fail_before_integral=0)


@pytest.mark.parametrize("role", ["amplitude_transient_bytes", "integral_transient_bytes"])
def test_accessor_transient_roles_cannot_hide(role):
    case = _case()
    with pytest.raises((ValueError, RuntimeError), match="byte cap"):
        _accessor_run(case, caps=_accessor_caps(_accessor_plan(case)), **{role: 1})
    assert _accessor_run(case, **{role: 1}).moments.integral_calls > 0


@pytest.mark.parametrize("lane", ["singles", "doubles", "integral"])
@pytest.mark.parametrize("last", [False, True])
def test_accessor_callback_exceptions_abort_without_mutating_input_or_publishing(lane, last):
    case = _case()
    plan = _accessor_plan(case)
    calls = plan.kernel.integral_calls if lane == "integral" else getattr(plan, f"{lane}_calls")
    saved = {key: case[key].copy() for key in ("t1", "t2", "fov", "eri")}
    with pytest.raises(RuntimeError, match=f"injected {lane} callback failure"):
        _accessor_run(case, **{f"fail_before_{lane}": calls - 1 if last else 0})
    for key, array in saved.items():
        np.testing.assert_array_equal(case[key], array)
    result = _accessor_run(case)
    w = result.moments.connected.copy()
    result.moments.connected[:] = 0
    del case
    np.testing.assert_array_equal(result.moments.connected, w)


@pytest.mark.parametrize("field", ["t1", "t2", "eri"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_accessor_consumed_nonfinite_amplitude_or_integral_aborts(field, value):
    case = _case()
    case[field][:] = value
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="finite"):
        _accessor_run(case)


def test_accessor_scope_is_consumed_plus_reverse_queries_not_global_amplitude_certification():
    case = _case(4, 2)
    clean = _accessor_run(case, (0, 0, 0))
    # W with occupied 000 reads T00 and T0l, then explicitly audits Tl0.
    # T33 and singles row 3 are not consumed: the accessor makes NO claim
    # about them. The separate dense entry point retains its full audit.
    case["t2"][3, 3] = np.nan
    case["t1"][3] = np.nan
    partial = _accessor_run(case, (0, 0, 0))
    np.testing.assert_array_equal(partial.moments.connected, clean.moments.connected)
    np.testing.assert_array_equal(partial.moments.singles, clean.moments.singles)
    with pytest.raises(ValueError, match="finite"):
        _run(case, (0, 0, 0))
    # T30 is reverse-only for this triple. Its mismatch must still reject.
    case["t2"][3, 0, 0, 1] += 0.2
    with pytest.raises(ValueError, match="consumed amplitude symmetry audit"):
        _accessor_run(case, (0, 0, 0))
    case["t2"][3, 0, 0, 1] = np.nan
    with pytest.raises(OverflowError, match="finite"):
        _accessor_run(case, (0, 0, 0))


def test_accessor_symmetry_audit_preserves_raw_amplitudes_within_explicit_budget():
    case = _case(2, 2)
    case["t2"][0, 1, 0, 1] += 1e-10
    tolerance = 2e-10
    expected = _run(case, (0, 0, 1), symmetry=tolerance)
    result = _accessor_run(case, (0, 0, 1), symmetry=tolerance)
    assert 0 < result.maximum_consumed_amplitude_symmetry_error < tolerance
    for field in ("connected", "singles"):
        np.testing.assert_array_equal(getattr(result.moments, field).view(np.uint64),
                                      getattr(expected, field).view(np.uint64))
    boundary = result.maximum_consumed_amplitude_symmetry_error
    _accessor_run(case, (0, 0, 1), symmetry=boundary)
    with pytest.raises(ValueError, match="symmetry audit"):
        _accessor_run(case, (0, 0, 1), symmetry=np.nextafter(boundary, 0.0))


@pytest.mark.parametrize("value", [0.1, -0.1, np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0)])
def test_accessor_does_not_silently_project_nonzero_brillouin_block(value):
    case = _case()
    case["fov"][0, 0] = value
    with pytest.raises(ValueError, match="exactly zero Fov"):
        _accessor_run(case, fail_before_singles=0, fail_before_doubles=0, fail_before_integral=0)


def test_accessor_zero_values_keep_every_call_and_control_snapshot_is_isolated():
    case = _case()
    before = _accessor_run(case)
    after = _accessor_run(case, mutate_control_objects=True)
    assert after.moments.integral_calls == before.moments.integral_calls
    assert after.singles_calls == before.singles_calls
    assert after.doubles_calls == before.doubles_calls
    assert after.moments.amplitude_snapshot_id == 73
    assert tuple(after.moments.occupied) == (0, 1, 2)
    np.testing.assert_array_equal(after.moments.connected.view(np.uint64), before.moments.connected.view(np.uint64))
    np.testing.assert_array_equal(after.moments.singles.view(np.uint64), before.moments.singles.view(np.uint64))
    case["t1"][:] = case["t2"][:] = 0
    zero = _accessor_run(case, mutate_control_objects=True)
    assert zero.singles_calls == before.singles_calls and zero.doubles_calls == before.doubles_calls
    assert zero.moments.integral_calls == before.moments.integral_calls
    np.testing.assert_array_equal(zero.moments.connected, 0)
    np.testing.assert_array_equal(zero.moments.singles, 0)


@pytest.mark.parametrize("fault", [1, 2, 3, 4])
def test_accessor_floating_environment_is_checked_at_entry_and_after_each_provider(fault):
    case = _case()
    clean = _accessor_run(case)
    with pytest.raises(RuntimeError, match="nearest IEEE"):
        _accessor_run(case, floating_environment_fault=fault)
    # Diagnostic restores the caller's environment even on native rejection.
    recovered = _accessor_run(case)
    np.testing.assert_array_equal(recovered.moments.connected.view(np.uint64), clean.moments.connected.view(np.uint64))


@pytest.mark.parametrize("disabled", range(1, 8))
def test_accessor_requires_all_callbacks(disabled):
    with pytest.raises(ValueError, match="callbacks"):
        _accessor_run(_case(), disabled_callbacks=disabled)


@pytest.mark.parametrize("field", ["singles_work_units", "doubles_work_units"])
def test_accessor_positive_query_work_is_mandatory(field):
    case = _case()
    with pytest.raises(ValueError, match="positive amplitude query work"):
        _accessor_run(case, caps=_accessor_caps(_accessor_plan(case)), **{field: 0})


@pytest.mark.parametrize("kwargs", [dict(snapshot=0), dict(symmetry=-1), dict(symmetry=np.nan),
    dict(symmetry=np.inf),
    dict(occupied=(3, 0, 0)), dict(occupied=(0, 3, 0)), dict(occupied=(0, 0, 3))])
def test_accessor_invalid_operator_options_precede_callbacks(kwargs):
    with pytest.raises((ValueError, IndexError), match="snapshot|symmetry|out of range"):
        _accessor_run(_case(), fail_before_singles=0, fail_before_doubles=0,
                      fail_before_integral=0, **kwargs)


@pytest.mark.parametrize("field", ["t1", "t2", "fov", "eri"])
@pytest.mark.parametrize("bad_kind", ["dtype", "shape", "strides", "alignment"])
def test_accessor_diagnostic_rejects_bad_views(field, bad_kind):
    case = _case()
    caps = _accessor_caps(_accessor_plan(case))
    original = case[field]
    if bad_kind == "dtype":
        case[field] = original.astype(np.float32)
    elif bad_kind == "shape":
        case[field] = original.reshape(-1)
    elif bad_kind == "strides":
        case[field] = original[..., ::-1]
    else:
        raw = np.zeros(original.nbytes + 1, dtype=np.uint8)
        case[field] = np.ndarray(original.shape, dtype=np.float64, buffer=raw, offset=1)
    with pytest.raises(ValueError, match="aligned C-contiguous binary64|shape mismatch"):
        _accessor_run(case, caps=caps)


@pytest.mark.parametrize("o,v", [(0, 1), (1, 0), (2**63, 2), (1, 2**63)])
def test_accessor_count_planner_rejects_dimensions_before_views_exist(o, v):
    with pytest.raises((ValueError, OverflowError, RuntimeError)):
        core._plan_bounded_restricted_triples_moments_accessor(o, v, 0, 0, 1, 1)


def test_accessor_count_planner_checks_query_work_and_role_addition_overflow():
    maximum = 2**64 - 1
    for args in [(1, 1, 0, 0, maximum, 1), (1, 1, 0, 0, 1, maximum),
                 (1, 1, maximum, 1, 1, 1), (1, 1, 0, 0, 1, 1, maximum, 1)]:
        with pytest.raises(OverflowError, match="overflow"):
            core._plan_bounded_restricted_triples_moments_accessor(*args)


def test_accessor_count_only_plan_has_no_phantom_dense_amplitude_extent():
    # Pure integer planning, no allocation or amplitude/ERI callback. This
    # admitted Fov extent exists even though a hypothetical dense T2 cannot.
    o = 2**33
    plan = core._plan_bounded_restricted_triples_moments_accessor(o, 1, 0, 0, 1, 1)
    assert plan.kernel.borrowed_input_bytes == 8 * o
    assert plan.kernel.peak_owned_numerical_bytes == 16
    assert plan.contraction_doubles_calls == 6 * (o + 1)
    with pytest.raises(OverflowError, match="overflow"):
        core._plan_bounded_restricted_triples_moments(o, 1)


def test_accessor_finite_amplitude_products_overflow_fail_closed():
    case = _case()
    case["t2"][:] = np.finfo(float).max
    case["eri"][:] = 2
    with pytest.raises(OverflowError, match="finite"):
        _accessor_run(case)
