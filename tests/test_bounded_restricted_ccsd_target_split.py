from __future__ import annotations

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_bounded_restricted_ccsd_target import (
    _problem, _plan, _caps, _run, _accessor_plan, _accessor_caps,
    _run_accessor, _amplitude_query_census,
)
from tests.test_bounded_restricted_pair_ccsd_particle_hole import (
    _system, _common_amplitudes, _evaluate_system,
)


def _remainder_plan(p, transient=0):
    return core._plan_bounded_restricted_ccsd_target_without_bare_particle_hole(
        p["o"], p["v"], p["eri"].nbytes, transient
    )


def _remainder_accessor_plan(p, *, singles_work_units=1, doubles_work_units=1,
                             amplitude_transient_bytes=0, integral_transient_bytes=0):
    return core._plan_bounded_restricted_ccsd_target_accessor_without_bare_particle_hole(
        p["o"], p["v"], p["t1"].nbytes + p["t2"].nbytes,
        amplitude_transient_bytes, singles_work_units, doubles_work_units,
        p["eri"].nbytes, integral_transient_bytes,
    )


def _arguments(p, i, j):
    o, f = p["o"], p["fock"]
    return (p["t1"], p["t2"], np.ascontiguousarray(f[:o, :o]),
            np.ascontiguousarray(f[o:, o:]), np.ascontiguousarray(f[:o, o:]), p["eri"], i, j)


def _remainder(p, i=0, j=1, caps=None, **kwargs):
    caps = _caps(_remainder_plan(p, kwargs.get("provider_transient_bytes", 0))) if caps is None else caps
    return core._bounded_restricted_ccsd_target_residual_without_bare_particle_hole_diagnostic(
        *_arguments(p, i, j), caps, **kwargs
    )


def _remainder_accessor(p, i=0, j=1, caps=None, **kwargs):
    if caps is None:
        keys = {key: value for key, value in kwargs.items() if key in {
            "singles_work_units", "doubles_work_units", "amplitude_transient_bytes", "integral_transient_bytes",
        }}
        caps = _accessor_caps(_remainder_accessor_plan(p, **keys))
    return core._bounded_restricted_ccsd_target_residual_accessor_without_bare_particle_hole_diagnostic(
        *_arguments(p, i, j), caps, **kwargs
    )


def _bare_original_terms(p, i, j):
    # Independent common-space original restricted target: THREE bare seeds,
    # including WX's different occupied index and BOTH output permutations.
    # Not full-minus-remainder, not the staged local F factorization, and no
    # amplitude-dependent W dressing. Use the platform long-double type;
    # it supplies extra precision where available, but is binary64 on macOS ARM.
    o, v = p["o"], p["v"]
    t = p["t2"].astype(np.longdouble)
    g = p["eri"].astype(np.longdouble)
    output = np.zeros((v, v), dtype=np.longdouble)
    for leg, (left, right) in enumerate([(i, j), (j, i)]):
        for a in range(v):
            for b in range(v):
                value = np.longdouble(0)
                for m in range(o):
                    for e in range(v):
                        w1 = g[m, o+e, right, o+b]
                        w2 = g[m, o+e, right, o+b] - g[m, right, o+b, o+e]
                        wx = -g[m, left, o+b, o+e]
                        value += (t[left, m, a, e] - t[m, left, a, e]) * w1
                        value += t[left, m, a, e] * w2
                        value += t[m, right, a, e] * wx
                if leg == 0:
                    output[a, b] += value
                else:
                    output[b, a] += value
    return np.asarray(output, dtype=np.float64)


@pytest.mark.parametrize("o,v", [(1, 1), (1, 3), (2, 2), (3, 4), (4, 2)])
@pytest.mark.parametrize("accessor", [False, True])
def test_full_target_equals_remainder_plus_independent_complete_bare_group(o, v, accessor):
    p = _problem(o, v, seed=733 + 10*o + v)
    assert np.linalg.norm(p["t1"]) > 0
    for i, j in sorted({(0, 0), (0, o-1), (o-1, 0), (o-1, o-1)}):
        full = _run(p, i, j)
        remainder = _remainder_accessor(p, i, j).target if accessor else _remainder(p, i, j)
        bare = _bare_original_terms(p, i, j)
        np.testing.assert_allclose(remainder.doubles + bare, full.doubles, atol=4e-14, rtol=3e-13)
        np.testing.assert_array_equal(remainder.singles.view(np.uint64), full.singles.view(np.uint64))
        assert full.complete_target_residual
        assert not remainder.complete_target_residual
        assert remainder.memory.bare_particle_hole_excluded
        assert not full.memory.bare_particle_hole_excluded
        assert full.integral_calls - remainder.integral_calls == 6*o*v*v


@pytest.mark.parametrize("o", range(1, 5))
@pytest.mark.parametrize("v", range(1, 5))
def test_exhaustive_exact_callback_reduction_same_amplitude_census_and_allocation(o, v):
    p = _problem(o, v, seed=124 + 10*o + v)
    full_plan, split_plan = _plan(p), _remainder_plan(p)
    full = _run_accessor(p, 0, o-1, singles_work_units=3, doubles_work_units=7)
    split = _remainder_accessor(p, 0, o-1, singles_work_units=3, doubles_work_units=7)
    dense = _remainder(p, 0, o-1)
    saved = 6*o*v*v
    old_bound = (o*o*(1+6*v+14*v*v+6*v**3)
                 + o*(10*v*v+13*v**3) + 2*v**3 + v**4*(1+2*o))
    old_loops = o*o*v*v + o*v*v + v**3 + o*v + v*v + o + v
    old_input_lanes = o*o*v*v + 2*o*v + o*o + v*v
    assert full_plan.integral_calls_upper_bound == old_bound
    assert full_plan.kernel_work_units_upper_bound == old_input_lanes + 128*old_bound + 64*old_loops
    assert full_plan.omitted_bare_integral_calls == full_plan.omitted_bare_kernel_work_units == 0
    assert split_plan.integral_calls_upper_bound == old_bound-saved
    assert split_plan.omitted_bare_integral_calls == saved
    assert split_plan.omitted_bare_kernel_work_units == 128*saved
    assert split_plan.kernel_work_units_upper_bound == full_plan.kernel_work_units_upper_bound-128*saved
    assert dense.integral_calls == split.target.integral_calls == old_bound-2*o*v**3+v*v-saved
    assert full.target.integral_calls - split.target.integral_calls == saved
    expected = _amplitude_query_census(o, v)
    assert (split.singles_calls, split.doubles_calls) == expected == (full.singles_calls, full.doubles_calls)
    assert split.charged_amplitude_work_units == full.charged_amplitude_work_units == 3*expected[0]+7*expected[1]
    assert split.memory.kernel.kernel_work_units_upper_bound == full.memory.kernel.kernel_work_units_upper_bound-128*saved
    for name in ("borrowed_input_bytes", "output_bytes", "workspace_bytes", "peak_owned_numerical_bytes", "total_live_numerical_bytes"):
        assert getattr(split_plan, name) == getattr(full_plan, name)
        assert getattr(split.memory.kernel, name) == getattr(full.memory.kernel, name)
    np.testing.assert_array_equal(split.target.singles.view(np.uint64), dense.singles.view(np.uint64))
    np.testing.assert_array_equal(split.target.doubles.view(np.uint64), dense.doubles.view(np.uint64))


@pytest.mark.parametrize("a,b,c", [(1, 2, 3), (2, 3, 1), (3, 1, 2), (2, 0, 3), (0, 2, 1)])
@pytest.mark.parametrize("target", [(0, 1), (1, 0), (0, 0), (1, 1)])
def test_ragged_nonnested_projection_remainder_plus_native_local_bare_group(a, b, c, target):
    ranks = {(0, 0): b, (0, 1): a, (0, 2): c,
             (1, 1): c, (1, 2): b, (2, 2): 1}
    system = _system(o=3, n=6, ranks=ranks)
    o, n, eri, frames, _ = system
    rng = np.random.default_rng(643)
    f = rng.normal(scale=0.1, size=(o+n, o+n))
    p = dict(o=o, v=n, eri=np.ascontiguousarray(eri), fock=np.ascontiguousarray(f+f.T),
             t1=np.ascontiguousarray(rng.normal(scale=0.05, size=(o, n))),
             t2=np.ascontiguousarray(_common_amplitudes(system)))
    i, j = target
    frame = frames[min(i, j), max(i, j)]
    full = _run(p, i, j)
    remainder = _remainder_accessor(p, i, j).target
    local = _evaluate_system(system, i, j).residual_copy()
    np.testing.assert_allclose(local, frame.T@_bare_original_terms(p, i, j)@frame, atol=3e-16, rtol=3e-13)
    np.testing.assert_allclose(frame.T@remainder.doubles@frame + local,
                               frame.T@full.doubles@frame, atol=4e-14, rtol=3e-13)


@pytest.mark.parametrize("t1_zero", [False, True])
def test_zero_t2_keeps_all_dressed_and_explicit_t1_squared_contributions(t1_zero):
    p = _problem(2, 3, seed=901)
    p["t2"][:] = 0
    if t1_zero:
        p["t1"][:] = 0
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        full = _run(p, i, j)
        remainder = _remainder(p, i, j)
        np.testing.assert_array_equal(full.singles.view(np.uint64), remainder.singles.view(np.uint64))
        np.testing.assert_array_equal(full.doubles.view(np.uint64), remainder.doubles.view(np.uint64))
        assert full.integral_calls-remainder.integral_calls == 6*2*3*3
    if not t1_zero:
        assert np.linalg.norm(_remainder(p).doubles-p["eri"][0, 2:, 1, 2:]) > 1e-4


def test_removed_group_is_independent_of_t1_while_the_remainder_is_not():
    p = _problem(2, 3, seed=849)
    no_singles = dict(p, t1=np.zeros_like(p["t1"]))
    bare = _bare_original_terms(p, 0, 1)
    np.testing.assert_array_equal(bare, _bare_original_terms(no_singles, 0, 1))
    for problem in (p, no_singles):
        full, remainder = _run(problem), _remainder(problem)
        np.testing.assert_allclose(full.doubles, remainder.doubles+bare, atol=3e-14, rtol=3e-13)
    assert np.linalg.norm(_remainder(p).doubles-_remainder(no_singles).doubles) > 1e-4


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_total_numerical_bytes",
                                   "maximum_integral_calls", "maximum_kernel_work_units"])
def test_remainder_dense_cap_minus_one_before_floating_scan_or_integral_callback(field):
    p = _problem()
    cap = _caps(_remainder_plan(p))
    setattr(cap, field, getattr(cap, field)-1)
    p["t1"][0, 0] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _remainder(p, caps=cap, fail_before_call=0)


@pytest.mark.parametrize("field", ["maximum_owned_numerical_bytes", "maximum_total_numerical_bytes",
                                   "maximum_integral_calls", "maximum_kernel_work_units",
                                   "maximum_singles_calls", "maximum_doubles_calls", "maximum_amplitude_work_units"])
def test_remainder_accessor_cap_minus_one_before_any_callback(field):
    p = _problem()
    cap = _accessor_caps(_remainder_accessor_plan(p))
    if hasattr(cap, field):
        setattr(cap, field, getattr(cap, field)-1)
    else:
        kernel = cap.kernel
        setattr(kernel, field, getattr(kernel, field)-1)
        cap.kernel = kernel
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _remainder_accessor(p, caps=cap, fail_before_singles=0, fail_before_doubles=0, fail_before_integral=0)


def test_split_exact_callback_boundary_and_unchanged_control_snapshot_contract():
    p = _problem()
    split = _remainder(p)
    # Failure at the first invocation after the COMPLETE split census must
    # never fire. The full target needs more callbacks and must hit it.
    _remainder(p, fail_before_call=split.integral_calls)
    with pytest.raises(RuntimeError, match="injected integral"):
        _remainder(p, fail_before_call=split.integral_calls-1)
    with pytest.raises(RuntimeError, match="injected integral"):
        _run(p, fail_before_call=split.integral_calls)
    reference = _remainder_accessor(p)
    mutated = _remainder_accessor(p, mutate_control_objects=True)
    np.testing.assert_array_equal(reference.target.doubles.view(np.uint64), mutated.target.doubles.view(np.uint64))
    assert reference.target.integral_calls == mutated.target.integral_calls
    assert (reference.singles_calls, reference.doubles_calls) == (mutated.singles_calls, mutated.doubles_calls)


@pytest.mark.parametrize("field", ["t1", "t2", "fock"])
@pytest.mark.parametrize("accessor", [False, True])
def test_remainder_nonfinite_original_inputs_or_accessor_values_abort(field, accessor):
    p = _problem()
    p[field].flat[0] = np.nan
    with pytest.raises((ValueError, OverflowError), match="finite"):
        (_remainder_accessor if accessor else _remainder)(p)


@pytest.mark.parametrize("failure", ["singles", "doubles", "integral"])
def test_excluded_accessor_exception_aborts_without_partial_result(failure):
    with pytest.raises(RuntimeError, match=f"injected {failure}"):
        _remainder_accessor(_problem(), **{f"fail_before_{failure}": 0})


def test_excluded_inventory_keeps_both_provider_transient_roles_and_explicit_query_work():
    p = _problem()
    options = dict(singles_work_units=3, doubles_work_units=9,
                   amplitude_transient_bytes=193, integral_transient_bytes=317)
    full, split = _accessor_plan(p, **options), _remainder_accessor_plan(p, **options)
    assert split.kernel.total_live_numerical_bytes == full.kernel.total_live_numerical_bytes
    assert split.amplitude_retained_numerical_bytes == p["t1"].nbytes+p["t2"].nbytes
    assert split.amplitude_maximum_transient_numerical_bytes == 193
    assert split.kernel.provider_maximum_transient_numerical_bytes == 317
    _remainder_accessor(p, **options)
    with pytest.raises(ValueError, match="positive"):
        _remainder_accessor_plan(p, singles_work_units=0)


@pytest.mark.parametrize("o,v", [(0, 1), (1, 0), (2**63, 2), (1, 2**63)])
def test_count_only_excluded_plans_fail_closed_on_zero_or_overflow(o, v):
    with pytest.raises((ValueError, OverflowError)):
        core._plan_bounded_restricted_ccsd_target_without_bare_particle_hole(o, v)
