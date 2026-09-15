"""Native-only bare-PH replacement, independent full-residual reference.

The producer below is a compiled tiny-array fixture, not a Python callback
or a physical Gaussian source. All energy/residual comparisons use the same
original Fock, connected T1/T2 and real eightfold-symmetric common ERIs.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_bounded_restricted_pair_ccsd_solver import (
    _CAP_FIELDS, _arguments, _caps as _legacy_caps, _energy, _expand,
    _inventory, _jacobi_oracle, _legacy, _local_result, _options,
    _plan as _full_plan, _problem, _project, _projected_norms, _run as _full,
)


def _caps(plan):
    caps = _legacy_caps(plan)
    caps.maximum_particle_hole_calls = plan.particle_hole_calls_upper_bound
    return caps


def _upper(p, options=None, inventory=None, transient=0, extra=0):
    return core._plan_bounded_restricted_pair_ccsd_solver_split_upper(
        p["o"], p["n"], max(p["sr"]), max(p["pr"]),
        _options() if options is None else options,
        _inventory() if inventory is None else inventory, p["eri"].nbytes,
        transient, extra,
    )


def _plan(p, options=None, inventory=None, caps=None, transient=0, extra=0):
    options = _options() if options is None else options
    inventory = _inventory() if inventory is None else inventory
    if caps is None:
        caps = _caps(_upper(p, options, inventory, transient, extra))
    return core._plan_bounded_restricted_pair_ccsd_solver_split(
        *_arguments(p), options, inventory, caps, transient, extra,
    )


def _run(p, *, options=None, inventory=None, caps=None, callback=None,
         transient=0, extra=0, fault=0, fail_before_call=2**64-1):
    options = _options() if options is None else options
    inventory = _inventory() if inventory is None else inventory
    if caps is None:
        caps = _caps(_plan(p, options, inventory, transient=transient, extra=extra))
    return core._bounded_restricted_pair_ccsd_solve_split_diagnostic(
        *_arguments(p), options, inventory, caps, callback, fail_before_call,
        transient, fault, extra,
    )


_NORMS = ("singles_max_residual", "doubles_max_residual",
          "singles_residual_norm", "doubles_residual_norm")


def _payload_oracle(p, result):
    """Original amplitude wire, unchanged for both execution routes."""
    digest = hashlib.sha256()
    def u64(value):
        digest.update(struct.pack(">Q", value))
    def text(value):
        data = value.encode()
        u64(len(data))
        digest.update(data)
    text("vibeqc.bounded-restricted-pair-ccsd.amplitudes")
    u64(1)
    text(result.input_identity_sha256)
    for value in np.concatenate([x.ravel() for group in _local_result(p, result) for x in group]):
        digest.update(struct.pack(">d", 0.0 if value == 0 else value))
    return digest.hexdigest()


@pytest.mark.parametrize("sr,pr", [
    ([3, 3], [3, 3, 3]), ([1, 2], [2, 1, 3]),
    ([0, 2], [0, 2, 1]), ([2, 1], [0, 0, 0]),
])
def test_nonzero_singles_mixed_frames_same_snapshot_full_residual_and_energy(sr, pr):
    p = _problem(2, 3, singles_ranks=sr, pair_ranks=pr,
                 rotated=True, nonzero_initial=True)
    options = _options(maximum_iterations=1)
    full, split = _full(p, options=options), _run(p, options=options)
    case = _expand(p, p["st"], p["pt"])
    residual = _legacy(case)
    expected = _projected_norms(p, *_project(p, *residual))
    assert not split.final_snapshot.converged
    for name, value in zip(_NORMS, expected, strict=True):
        assert getattr(split.final_snapshot, name) == pytest.approx(value, abs=3e-14)
        assert getattr(split.final_snapshot, name) == pytest.approx(getattr(full.final_snapshot, name), abs=3e-14)
    energy = _energy(case, case["t1"], case["t2"])
    assert split.final_snapshot.correlation_energy == pytest.approx(energy, abs=3e-14)
    assert split.final_snapshot.correlation_energy == full.final_snapshot.correlation_energy
    # An exhausted solve returns the evaluated initial snapshot, not candidate R.
    for actual, expected in zip(sum(_local_result(p, split), []), p["st"]+p["pt"], strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert split.input_identity_sha256 == full.input_identity_sha256
    assert split.payload_sha256 == full.payload_sha256 == _payload_oracle(p, split)
    assert full.split_execution_identity_sha256 == ""
    assert full.consumed_particle_hole_identity_sha256 == ""
    assert len(split.split_execution_identity_sha256) == 64
    assert len(split.consumed_particle_hole_identity_sha256) == 64
    assert not split.particle_hole_physical_source_certified


@pytest.mark.parametrize("o,n,sr,pr", [
    (1, 2, [2], [2]), (2, 3, [1, 2], [2, 1, 3]),
    (2, 3, [0, 2], [0, 2, 1]), (3, 2, [1, 0, 2], [2, 1, 0, 1, 2, 1]),
])
def test_iterated_split_fixedpoint_matches_independent_projected_oracle(o, n, sr, pr):
    p = _problem(o, n, singles_ranks=sr, pair_ranks=pr,
                 rotated=True, nonzero_initial=True)
    split, full = _run(p), _full(p)
    assert split.final_snapshot.converged and full.final_snapshot.converged
    expected_s, expected_p, energy, _, _ = _jacobi_oracle(p)
    for actual, expected in zip(sum(_local_result(p, split), []), expected_s+expected_p, strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=1e-10)
    for actual, expected in zip(sum(_local_result(p, split), []), sum(_local_result(p, full), []), strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=1e-10)
    assert split.final_snapshot.correlation_energy == pytest.approx(energy, abs=2e-13)
    for i, j in p["labels"]:
        np.testing.assert_array_equal(split.pair_copy(j, i), split.pair_copy(i, j).T)
    assert split.payload_sha256 == _payload_oracle(p, split)
    validation = core._plan_bounded_restricted_pair_ccsd_solver_payload_validation(split)
    core._verify_bounded_restricted_pair_ccsd_solver_payload(split, validation.work_units)


def test_two_fixed_iterations_pin_candidate_update_and_bare_only_call_difference():
    p = _problem(2, 3, singles_ranks=[1, 2], pair_ranks=[2, 1, 3],
                 rotated=True, nonzero_initial=True)
    options = _options(maximum_iterations=2)
    events = []
    split, full = _run(p, options=options, callback=events.append), _full(p, options=options)
    expected_s, expected_p, energy, _, norms = _jacobi_oracle(p, iterations=2, fixed_iterations=True)
    for actual, expected in zip(sum(_local_result(p, split), []), expected_s+expected_p, strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=1e-12)
    assert split.final_snapshot.correlation_energy == pytest.approx(energy, abs=3e-14)
    for name, expected in zip(_NORMS, norms, strict=True):
        assert getattr(split.final_snapshot, name) == pytest.approx(expected, abs=3e-14)
    a, b = full.final_snapshot, split.final_snapshot
    evaluations = 2*len(p["labels"])
    omitted = evaluations*6*p["o"]*p["n"]**2
    assert a.integral_calls-b.integral_calls == omitted
    assert split.memory.omitted_bare_integral_calls_upper_bound == omitted
    assert a.singles_calls == b.singles_calls
    assert a.doubles_calls == b.doubles_calls
    assert b.particle_hole_calls == b.particle_hole_visits == evaluations
    assert b.particle_hole_source_slots == 2*p["o"]*evaluations
    assert b.charged_particle_hole_work_units == evaluations*(
        split.memory.particle_hole_work_units_per_target+split.memory.particle_hole_guard_work_units_per_target)
    assert b.charged_work_units == split.memory.work_units_upper_bound
    assert [e.particle_hole_calls for e in events] == [len(p["labels"]), evaluations]
    assert b.input_checks == b.iteration+1+2*evaluations


def test_zero_pair_ranks_are_visited_and_singles_product_energy_is_not_projected_away():
    p = _problem(2, 3, singles_ranks=[2, 1], pair_ranks=[0, 0, 0],
                 rotated=True, nonzero_initial=True)
    options = _options(maximum_iterations=1)
    result = _run(p, options=options)
    case = _expand(p, p["st"], p["pt"])
    product_energy = np.einsum("ijab,ia,jb", 2*p["eri"][:2, 2:, :2, 2:].transpose(0, 2, 1, 3)
        -p["eri"][:2, 2:, :2, 2:].transpose(0, 2, 3, 1), case["t1"], case["t1"])
    assert abs(product_energy) > 1e-15
    assert result.final_snapshot.correlation_energy == pytest.approx(_energy(case, case["t1"], case["t2"]), abs=3e-14)
    assert result.final_snapshot.doubles_residual_norm == 0
    assert result.final_snapshot.particle_hole_calls == 3
    assert result.final_snapshot.particle_hole_source_slots == 12


def test_all_zero_spaces_still_require_one_complete_visit_per_target_and_snapshot():
    p = _problem(2, 2, singles_ranks=[0, 0], pair_ranks=[0, 0, 0])
    result = _run(p)
    assert result.final_snapshot.converged
    assert result.final_snapshot.iteration == 2
    assert result.final_snapshot.correlation_energy == 0
    assert result.final_snapshot.particle_hole_calls == 6
    assert result.final_snapshot.particle_hole_visits == 6
    assert result.final_snapshot.particle_hole_source_slots == 24
    assert result.memory.local_particle_hole_phase_bytes == 0
    with pytest.raises(ValueError, match="exactly one"):
        _run(p, fault=1)


@pytest.mark.parametrize("transient,extra", [(0, 0), (32768, 0), (0, 32768), (32768, 32768)])
def test_peak_is_maximum_of_serialized_common_and_local_ph_phases(transient, extra):
    p = _problem(2, 3, singles_ranks=[1, 2], pair_ranks=[2, 1, 3])
    options = _options(maximum_iterations=1)
    plan = _plan(p, options, transient=transient, extra=extra)
    resident = (plan.amplitude_snapshot_bytes+plan.candidate_snapshot_bytes
                +plan.projected_fock_diagonal_bytes+plan.retained_record_bytes)
    assert plan.remaining_target_phase_bytes == plan.target_owned_peak_bytes+transient
    assert plan.local_particle_hole_phase_bytes == 48*max(p["pr"])**2+extra
    assert plan.peak_owned_numerical_bytes == resident+max(
        plan.target_owned_peak_bytes, plan.local_particle_hole_phase_bytes)
    expected = resident+max(plan.remaining_target_phase_bytes, plan.local_particle_hole_phase_bytes)
    expected += (plan.borrowed_initial_numerical_bytes+plan.borrowed_fock_bytes
                 +plan.provider_retained_numerical_bytes+plan.particle_hole_retained_numerical_bytes
                 +plan.owned_snapshot_table_bytes+plan.borrowed_input_table_bytes
                 +plan.control_storage_reservation_bytes+65536)
    assert plan.per_replica_inventoried_bytes == expected
    # The diagnostic reserved extra scratch need not allocate it; no science changes.
    result = _run(p, options=options, transient=transient, extra=extra, caps=_caps(plan))
    assert result.final_snapshot.iteration == 1


@pytest.mark.parametrize("field", list(_CAP_FIELDS)+["maximum_particle_hole_calls"])
def test_every_exact_cap_minus_one_rejects_before_nan_or_any_source_call(field):
    p = _problem(2, 2, nonzero_initial=True)
    options = _options(maximum_iterations=2)
    caps = _caps(_plan(p, options))
    setattr(caps, field, getattr(caps, field)-1)
    p["fov"][0, 0] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="cap|ceiling|bound|limit"):
        _run(p, options=options, caps=caps, fail_before_call=0, fault=6)


@pytest.mark.parametrize("fault,match", [
    (1, "exactly one"), (2, "exactly one"), (3, "snapshot"), (4, "target"),
    (5, "non-finite"), (6, "producer exception"), (7, "snapshot|changed"),
    (8, "changed"), (9, "non-finite"),
])
def test_untrusted_native_producer_failure_never_publishes_a_partial_result(fault, match):
    p = _problem(2, 2, nonzero_initial=True)
    events = []
    with pytest.raises((ValueError, RuntimeError, OverflowError), match=match):
        _run(p, callback=events.append, fault=fault)
    assert events == []


def test_ordinary_integral_failure_still_aborts_before_particle_hole_production():
    with pytest.raises(RuntimeError, match="integral callback"):
        _run(_problem(2, 2), fail_before_call=0, fault=6)


def test_progress_mutation_and_callback_exception_are_not_hidden_by_split_mode():
    p = _problem(2, 2)
    def mutate(_):
        p["st"][0][0] += .125
    with pytest.raises(ValueError, match="changed|snapshot"):
        _run(p, callback=mutate)
    def abort(_):
        raise LookupError("injected split progress stop")
    with pytest.raises(LookupError, match="split progress"):
        _run(_problem(2, 2), callback=abort)


def test_plan_has_no_floating_reads_and_legacy_route_ignores_split_only_cap():
    p = _problem(2, 2)
    p["fov"][0, 0] = np.nan
    assert _plan(p).split_bare_particle_hole
    with pytest.raises(ValueError, match="finite"):
        _run(p)
    p = _problem(2, 2)
    full_plan = _full_plan(p)
    caps = _legacy_caps(full_plan)
    assert caps.maximum_particle_hole_calls == 0
    result = _full(p, caps=caps)
    assert result.final_snapshot.converged
    assert result.final_snapshot.particle_hole_calls == 0
