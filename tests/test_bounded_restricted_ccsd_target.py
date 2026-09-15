"""Tiny arbitrary-amplitude CCSD leaf oracles; no chemistry or method claim."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core


def _problem(o=2, v=3, seed=178):
    rng = np.random.default_rng(seed)
    n = o + v
    factors = 0.15 * rng.normal(size=(5, n, n))
    factors = np.ascontiguousarray(factors + factors.transpose(0, 2, 1))
    eri = np.ascontiguousarray(np.einsum("Ppq,Prs->pqrs", factors, factors))
    fock = 0.2 * rng.normal(size=(n, n))
    fock = np.ascontiguousarray(fock + fock.T + np.diag(np.linspace(-1, 1, n)))
    t1 = np.ascontiguousarray(0.07 * rng.normal(size=(o, v)))
    raw = 0.06 * rng.normal(size=(o, o, v, v))
    t2 = np.ascontiguousarray(raw + raw.transpose(1, 0, 3, 2))
    return dict(o=o, v=v, factors=factors, eri=eri, fock=fock, t1=t1, t2=t2)


def _plan(problem, transient=0):
    return core._plan_bounded_restricted_ccsd_target(
        problem["o"], problem["v"], problem["eri"].nbytes, transient
    )


def _caps(plan):
    caps = core._BoundedRestrictedCCSDTargetCaps()
    caps.maximum_owned_numerical_bytes = plan.peak_owned_numerical_bytes
    caps.maximum_total_numerical_bytes = plan.total_live_numerical_bytes
    caps.maximum_integral_calls = plan.integral_calls_upper_bound
    caps.maximum_kernel_work_units = plan.kernel_work_units_upper_bound
    return caps


def _run(problem, i=0, j=1, caps=None, **kwargs):
    o, fock = problem["o"], problem["fock"]
    if caps is None:
        caps = _caps(_plan(problem, kwargs.get("provider_transient_bytes", 0)))
    return core._bounded_restricted_ccsd_target_residual_diagnostic(
        problem["t1"], problem["t2"],
        np.ascontiguousarray(fock[:o, :o]),
        np.ascontiguousarray(fock[o:, o:]),
        np.ascontiguousarray(fock[:o, o:]),
        problem["eri"], i, j, caps, **kwargs,
    )


def _legacy(problem):
    o, v, factors, fock = (problem[k] for k in ("o", "v", "factors", "fock"))
    p = factors.shape[0]
    r1, r2 = core.dlpno_pair_residual(
        problem["t1"], problem["t2"].reshape(o * o, v * v),
        np.ascontiguousarray(factors[:, :o, o:].reshape(p, o * v)),
        np.ascontiguousarray(factors[:, :o, :o].reshape(p, o * o)),
        np.ascontiguousarray(factors[:, o:, o:].reshape(p, v * v)),
        np.ascontiguousarray(fock[:o, :o]),
        np.ascontiguousarray(fock[o:, o:]),
        np.ascontiguousarray(fock[:o, o:]),
    )
    return np.asarray(r1), np.asarray(r2).reshape(o, o, v, v)


@pytest.mark.parametrize("o,v", [(1, 1), (1, 3), (2, 2), (3, 4)])
@pytest.mark.parametrize("blocked", [False, True])
def test_arbitrary_amplitude_target_matches_validated_spatial_kernel(o, v, blocked, monkeypatch):
    monkeypatch.setenv("VIBEQC_CCSD_VVVV_INCORE_BYTES", "0" if blocked else "1000000")
    monkeypatch.setenv("VIBEQC_CCSD_VVVV_TILE_ROWS", "1")
    problem = _problem(o, v, seed=178 + 10 * o + v)
    r1, r2 = _legacy(problem)
    for i, j in {(0, 0), (0, o - 1), (o - 1, 0), (o - 1, o - 1)}:
        result = _run(problem, i, j)
        np.testing.assert_allclose(result.singles, r1[i], rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(result.doubles, r2[i, j], rtol=2e-13, atol=2e-13)
        # Exact traversal count differs from the conservative plan by the
        # shared two-virtual ring integrals, and includes the bare pair block.
        expected = result.memory.integral_calls_upper_bound - 2 * o * v**3 + v**2
        assert result.integral_calls == expected


def test_independent_spin_orbital_stanton_anchor_at_nonzero_amplitudes():
    from vibeqc.dlpno._ccsd_ref import _spin_orbital_eri, so_residuals

    problem = _problem(2, 3, seed=999)
    o, v, n = 2, 3, 5
    fock = problem["fock"]
    fso = np.zeros((2 * n, 2 * n))
    fso[0::2, 0::2] = fock
    fso[1::2, 1::2] = fock
    offdiagonal = fso.copy()
    np.fill_diagonal(offdiagonal, 0)
    eo, ev = np.diag(fso)[:2 * o], np.diag(fso)[2 * o:]
    d1 = eo[:, None] - ev[None, :]
    d2 = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
    t1so = np.zeros((2 * o, 2 * v))
    t1so[0::2, 0::2] = problem["t1"]
    t1so[1::2, 1::2] = problem["t1"]
    t2so = np.zeros((2 * o, 2 * o, 2 * v, 2 * v))
    for si in (0, 1):
        for sj in (0, 1):
            t2so[si::2, sj::2, si::2, sj::2] += problem["t2"]
            t2so[si::2, sj::2, sj::2, si::2] -= problem["t2"].transpose(0, 1, 3, 2)
    r1, r2 = so_residuals(
        fso, offdiagonal, _spin_orbital_eri(problem["factors"], n),
        t1so, t2so, slice(0, 2 * o), slice(2 * o, 2 * n), d1, d2,
    )
    for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)):
        result = _run(problem, i, j)
        np.testing.assert_allclose(result.singles, r1[2 * i, 0::2], rtol=3e-13, atol=3e-13)
        np.testing.assert_allclose(result.doubles, r2[2 * i, 2 * j + 1, 0::2, 1::2], rtol=3e-13, atol=3e-13)


def test_extended_domain_contraction_precedes_target_pno_projection():
    """A coupled amplitude outside target PNOs still affects its residual."""
    problem = _problem(2, 3, seed=333)
    problem["t1"][:] = 0
    problem["t2"][:] = 0
    problem["t2"][1, 1, 0, 2] = 0.3
    problem["t2"][1, 1, 2, 0] = 0.3
    result = _run(problem, 0, 1)
    _, full_r2 = _legacy(problem)
    target = np.eye(3)[:, :1]
    np.testing.assert_allclose(target.T @ result.doubles @ target,
                               target.T @ full_r2[0, 1] @ target, atol=1e-13)
    truncated = dict(problem)
    truncated["t2"] = problem["t2"].copy()
    truncated["t2"][1, 1, 0, 2] = 0
    truncated["t2"][1, 1, 2, 0] = 0
    wrong = _run(truncated, 0, 1)
    assert abs((target.T @ (result.doubles - wrong.doubles) @ target)[0, 0]) > 1e-6


def test_zero_amplitudes_return_bare_fock_and_pair_integrals():
    problem = _problem()
    problem["t1"][:] = 0
    problem["t2"][:] = 0
    result = _run(problem, 0, 1)
    np.testing.assert_array_equal(result.singles, problem["fock"][0, 2:])
    np.testing.assert_array_equal(result.doubles, problem["eri"][0, 2:, 1, 2:])


def test_exact_owned_peak_and_borrowed_provider_inventory():
    o, v = 3, 4
    plan = core._plan_bounded_restricted_ccsd_target(o, v, 123, 456)
    assert plan.output_bytes == 8 * (v + v * v)
    assert plan.workspace_bytes == 8 * (v * v + 2 * o + 3 * o * v)
    assert plan.peak_owned_numerical_bytes == 8 * (2 * v * v + v + 3 * o * v + 2 * o)
    assert plan.borrowed_input_bytes == 8 * (o * o * v * v + 2 * o * v + o * o + v * v)
    assert plan.total_live_numerical_bytes == plan.peak_owned_numerical_bytes + plan.borrowed_input_bytes + 123 + 456


@pytest.mark.parametrize("field", [
    "maximum_owned_numerical_bytes", "maximum_total_numerical_bytes",
    "maximum_integral_calls", "maximum_kernel_work_units",
])
@pytest.mark.parametrize("zero", [False, True])
def test_every_cap_fails_before_integral_callback(field, zero):
    problem = _problem()
    caps = _caps(_plan(problem))
    setattr(caps, field, 0 if zero else getattr(caps, field) - 1)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(problem, caps=caps, fail_before_call=0)


def test_provider_transient_inventory_cannot_hide_outside_total_cap():
    problem = _problem()
    with pytest.raises((ValueError, RuntimeError), match="byte cap"):
        _run(problem, caps=_caps(_plan(problem)), provider_transient_bytes=1)
    _run(problem, caps=_caps(_plan(problem, 1)), provider_transient_bytes=1)


@pytest.mark.parametrize("fail_before", [0, 4, 300])
def test_callback_exception_discards_partial_result_and_inputs_stay_immutable(fail_before):
    problem = _problem()
    copies = {key: problem[key].copy() for key in ("t1", "t2", "fock", "eri")}
    with pytest.raises(RuntimeError, match="injected integral callback failure"):
        _run(problem, fail_before_call=fail_before)
    for key, value in copies.items():
        np.testing.assert_array_equal(problem[key], value)
    assert _run(problem).integral_calls > fail_before


@pytest.mark.parametrize("field", ["t1", "t2", "fock"])
def test_nonfinite_input_fails_before_callback(field):
    problem = _problem()
    problem[field].flat[0] = np.nan
    with pytest.raises(ValueError, match="input must be finite"):
        _run(problem, fail_before_call=0)


def test_nonfinite_integral_and_arithmetic_overflow_fail_closed():
    problem = _problem()
    problem["eri"][:] = np.inf
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="not finite"):
        _run(problem)
    problem = _problem()
    problem["t1"][:] = 1e308
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="not finite"):
        _run(problem)


def test_output_copies_are_independent_and_no_raw_view_casts():
    problem = _problem()
    result = _run(problem)
    before = result.doubles
    changed = result.doubles
    changed[:] = 123
    np.testing.assert_array_equal(result.doubles, before)
    problem["t1"] = np.asfortranarray(problem["t1"])
    with pytest.raises(ValueError, match="C-contiguous"):
        _run(problem)
    problem["t1"] = np.ascontiguousarray(problem["t1"], dtype=np.float32)
    with pytest.raises(ValueError, match="binary64"):
        _run(problem)


@pytest.mark.parametrize("o,v", [(0, 1), (1, 0), (2**63, 2), (2**32, 2**32)])
def test_invalid_extent_plan_is_allocation_free(o, v):
    with pytest.raises((ValueError, OverflowError)):
        core._plan_bounded_restricted_ccsd_target(o, v)


def test_target_and_array_shape_rejected():
    problem = _problem()
    with pytest.raises((ValueError, IndexError), match="occupied index"):
        _run(problem, i=problem["o"])
    problem["t2"] = problem["t2"][:, :, :, :1].copy()
    with pytest.raises(ValueError, match="shape mismatch"):
        _run(problem)


def _accessor_plan(problem, *, singles_work_units=1, doubles_work_units=1,
                   amplitude_transient_bytes=0, integral_transient_bytes=0):
    return core._plan_bounded_restricted_ccsd_target_accessor(
        problem["o"], problem["v"], problem["t1"].nbytes + problem["t2"].nbytes,
        amplitude_transient_bytes, singles_work_units, doubles_work_units,
        problem["eri"].nbytes, integral_transient_bytes,
    )


def _accessor_caps(plan):
    caps = core._BoundedRestrictedCCSDTargetAccessorCaps()
    caps.kernel = _caps(plan.kernel)
    caps.maximum_singles_calls = plan.singles_calls_upper_bound
    caps.maximum_doubles_calls = plan.doubles_calls_upper_bound
    caps.maximum_amplitude_work_units = plan.amplitude_work_units_upper_bound
    return caps


def _run_accessor(problem, i=0, j=1, caps=None, **kwargs):
    if caps is None:
        plan_keys = {key: value for key, value in kwargs.items() if key in {
            "singles_work_units", "doubles_work_units", "amplitude_transient_bytes",
            "integral_transient_bytes",
        }}
        caps = _accessor_caps(_accessor_plan(problem, **plan_keys))
    o, fock = problem["o"], problem["fock"]
    return core._bounded_restricted_ccsd_target_residual_accessor_diagnostic(
        problem["t1"], problem["t2"],
        np.ascontiguousarray(fock[:o, :o]), np.ascontiguousarray(fock[o:, o:]),
        np.ascontiguousarray(fock[:o, o:]), problem["eri"], i, j, caps, **kwargs,
    )


def _amplitude_query_census(o, v):
    """Independent stage-by-stage traversal inventory, not a polynomial copy."""
    singles = doubles = 0
    # Fme: each (m,e,n,f) reads one single.
    for _m in range(o):
        for _e in range(v):
            for _n in range(o):
                for _f in range(v):
                    singles += 1
    # Fae: half-Fov, one t1-integral contraction, then half-tau.
    for _a in range(v):
        for _e in range(v):
            for _m in range(o):
                singles += 1 + v
            for _m in range(o):
                for _n in range(o):
                    for _f in range(v):
                        singles += 2
                        doubles += 1
    # Two Fmi columns, including i==j.
    for _column in range(2):
        for _m in range(o):
            singles += v + o*v + 2*o*v*v
            doubles += o*v*v
    # One complete singles residual row.
    for _a in range(v):
        singles += v + o + o*v
        doubles += 2*o*v + 2*o*v*v + 2*o*o*v
    # Doubles forms of Fae and the two Fmi columns.
    for _b in range(v):
        for _e in range(v):
            singles += o
    for _column in range(2):
        for _m in range(o):
            singles += v
    # Merged occupied and virtual half-tau ladders.
    for _m in range(o):
        for _n in range(o):
            singles += 2*v + 4*v*v
            doubles += 2*v*v
    for _a in range(v):
        for _b in range(v):
            for _e in range(v):
                for _f in range(v):
                    singles += 2*o + 2
                    doubles += 1
    # Two shared ring passes. Each n,f cell reuses three doubles and one
    # singles product; the different-left WX contribution reads a fourth T2.
    for _pass in range(2):
        for _b in range(v):
            for _m in range(o):
                for _e in range(v):
                    singles += 3*v + 3*o + 4*o*v
                    doubles += 4*o*v
            for _a in range(v):
                doubles += v + o + 3*o*v
                singles += 4*o*v + v + o
    return singles, doubles


@pytest.mark.parametrize("o", range(1, 5))
@pytest.mark.parametrize("v", range(1, 5))
def test_accessor_bitwise_dense_identity_and_exhaustive_tiny_query_bounds(o, v):
    problem = _problem(o, v, seed=621 + 10*o + v)
    expected_singles, expected_doubles = _amplitude_query_census(o, v)
    for i in range(o):
        for j in range(o):
            dense = _run(problem, i, j)
            result = _run_accessor(problem, i, j, singles_work_units=3, doubles_work_units=7)
            np.testing.assert_array_equal(result.target.singles.view(np.uint64), dense.singles.view(np.uint64))
            np.testing.assert_array_equal(result.target.doubles.view(np.uint64), dense.doubles.view(np.uint64))
            assert result.target.integral_calls == dense.integral_calls
            assert result.singles_calls == result.memory.singles_calls_upper_bound == expected_singles
            assert result.doubles_calls == result.memory.doubles_calls_upper_bound == expected_doubles
            assert result.charged_amplitude_work_units == 3*expected_singles + 7*expected_doubles
            assert result.charged_amplitude_work_units == result.memory.amplitude_work_units_upper_bound


def test_accessor_zero_amplitudes_keep_all_queries_and_extended_domain_terms():
    problem = _problem(2, 3, seed=333)
    problem["t1"][:] = 0
    problem["t2"][:] = 0
    zero = _run_accessor(problem)
    assert (zero.singles_calls, zero.doubles_calls) == _amplitude_query_census(2, 3)
    np.testing.assert_array_equal(zero.target.singles, problem["fock"][0, 2:])
    np.testing.assert_array_equal(zero.target.doubles, problem["eri"][0, 2:, 1, 2:])
    problem["t2"][1, 1, 0, 2] = problem["t2"][1, 1, 2, 0] = .3
    extended = _run_accessor(problem)
    assert abs(extended.target.doubles[0, 0] - zero.target.doubles[0, 0]) > 1e-6
    np.testing.assert_array_equal(extended.target.doubles, _run(problem).doubles)


def test_accessor_plan_excludes_dense_amplitude_views_and_counts_both_providers():
    o, v = 3, 4
    problem = _problem(o, v)
    plan = _accessor_plan(problem, amplitude_transient_bytes=23, integral_transient_bytes=47)
    dense = _plan(problem, transient=47)
    kernel = plan.kernel
    assert kernel.borrowed_input_bytes == 8*(o*o + v*v + o*v)
    assert plan.amplitude_retained_numerical_bytes == 8*(o*v + o*o*v*v)
    assert plan.amplitude_maximum_transient_numerical_bytes == 23
    assert kernel.provider_retained_numerical_bytes == problem["eri"].nbytes
    assert kernel.provider_maximum_transient_numerical_bytes == 47
    assert kernel.peak_owned_numerical_bytes == dense.peak_owned_numerical_bytes
    assert kernel.workspace_bytes == dense.workspace_bytes
    assert kernel.total_live_numerical_bytes == dense.total_live_numerical_bytes + 23
    assert kernel.integral_calls_upper_bound == dense.integral_calls_upper_bound
    assert kernel.kernel_work_units_upper_bound == (
        dense.kernel_work_units_upper_bound - (o*v + o*o*v*v)
        + 16*(plan.singles_calls_upper_bound + plan.doubles_calls_upper_bound)
    )
    # Count-only native planning does not require any amplitude array or owner.
    count_only = core._plan_bounded_restricted_ccsd_target_accessor(o, v, 0, 0, 1, 1)
    assert count_only.kernel.total_live_numerical_bytes == (
        count_only.kernel.peak_owned_numerical_bytes + count_only.kernel.borrowed_input_bytes
    )


@pytest.mark.parametrize("field", [
    "maximum_owned_numerical_bytes", "maximum_total_numerical_bytes",
    "maximum_integral_calls", "maximum_kernel_work_units",
    "maximum_singles_calls", "maximum_doubles_calls", "maximum_amplitude_work_units",
])
@pytest.mark.parametrize("zero", [False, True])
def test_accessor_all_caps_preadmit_before_scans_or_callbacks(field, zero):
    problem = _problem()
    caps = _accessor_caps(_accessor_plan(problem))
    owner = caps if field in {"maximum_singles_calls", "maximum_doubles_calls",
                             "maximum_amplitude_work_units"} else caps.kernel
    setattr(owner, field, 0 if zero else getattr(owner, field) - 1)
    problem["fock"][:] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run_accessor(problem, caps=caps, fail_before_singles=0,
                      fail_before_doubles=0, fail_before_integral=0)


@pytest.mark.parametrize("role", ["amplitude_transient_bytes", "integral_transient_bytes"])
def test_accessor_provider_transient_roles_cannot_hide_from_total_cap(role):
    problem = _problem()
    with pytest.raises((ValueError, RuntimeError), match="byte cap"):
        _run_accessor(problem, caps=_accessor_caps(_accessor_plan(problem)), **{role: 1})
    _run_accessor(problem, **{role: 1})


@pytest.mark.parametrize("role", ["singles", "doubles", "integral"])
@pytest.mark.parametrize("fail_before", [0, 1, 20])
def test_accessor_exception_aborts_without_touching_input_arrays(role, fail_before):
    problem = _problem()
    before = {key: problem[key].copy() for key in ("t1", "t2", "fock", "eri")}
    with pytest.raises(RuntimeError, match=f"injected {role} callback failure"):
        _run_accessor(problem, **{f"fail_before_{role}": fail_before})
    for key, original in before.items():
        np.testing.assert_array_equal(problem[key], original)


@pytest.mark.parametrize("field", ["t1", "t2", "eri"])
def test_accessor_nonfinite_callback_value_is_not_published(field):
    problem = _problem()
    problem[field][:] = np.nan
    with pytest.raises((ValueError, RuntimeError, OverflowError), match="not finite"):
        _run_accessor(problem)


def test_accessor_does_not_prescan_dense_amplitudes_but_does_scan_operator():
    problem = _problem()
    problem["t2"][:] = np.nan
    with pytest.raises(RuntimeError, match="injected singles callback failure"):
        _run_accessor(problem, fail_before_singles=0)
    problem["fock"][:] = np.nan
    with pytest.raises(ValueError, match="input must be finite"):
        _run_accessor(problem, fail_before_singles=0, fail_before_doubles=0, fail_before_integral=0)


def test_accessor_copies_control_descriptors_without_claiming_context_immutability():
    problem = _problem()
    expected = _run_accessor(problem)
    actual = _run_accessor(problem, mutate_control_objects=True)
    np.testing.assert_array_equal(actual.target.singles, expected.target.singles)
    np.testing.assert_array_equal(actual.target.doubles, expected.target.doubles)
    assert actual.singles_calls == expected.singles_calls
    assert actual.doubles_calls == expected.doubles_calls
    assert actual.target.integral_calls == expected.target.integral_calls


@pytest.mark.parametrize("mask", [1, 2, 3])
def test_accessor_requires_both_amplitude_callbacks(mask):
    with pytest.raises(ValueError, match="requires singles and doubles"):
        _run_accessor(_problem(), disabled_amplitude_callbacks=mask)


@pytest.mark.parametrize("singles_work,doubles_work", [(0, 1), (1, 0), (0, 0)])
def test_accessor_per_query_work_is_an_explicit_positive_declaration(singles_work, doubles_work):
    with pytest.raises(ValueError, match="work ceilings must be positive"):
        core._plan_bounded_restricted_ccsd_target_accessor(2, 3, 0, 0, singles_work, doubles_work)


def test_accessor_shape_layout_targets_and_overflow_fail_closed():
    problem = _problem()
    with pytest.raises(IndexError, match="occupied index"):
        _run_accessor(problem, i=2)
    problem["t2"] = problem["t2"][:, :, :, :1].copy()
    with pytest.raises(ValueError, match="shape mismatch"):
        _run_accessor(problem)
    problem = _problem()
    problem["t1"] = np.asfortranarray(problem["t1"])
    with pytest.raises(ValueError, match="C-contiguous"):
        _run_accessor(problem)
    for args in [(0, 1, 0, 0, 1, 1), (1, 0, 0, 0, 1, 1),
                 (2**32, 2**32, 0, 0, 1, 1), (1, 1, 2**64-1, 1, 1, 1),
                 (1, 1, 0, 0, 2**64-1, 1)]:
        with pytest.raises((ValueError, OverflowError)):
            core._plan_bounded_restricted_ccsd_target_accessor(*args)
