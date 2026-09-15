from __future__ import annotations

import gc
import hashlib
import struct

import pytest

from vibeqc import _vibeqc_core as core
from tests.test_bounded_restricted_pair_ccsd_solver import _options, _problem, _run


def _validation(result):
    return core._plan_bounded_restricted_pair_ccsd_solver_payload_validation(result)


def _verify(result, cap=None):
    cap = _validation(result).work_units if cap is None else cap
    core._verify_bounded_restricted_pair_ccsd_solver_payload(result, cap)


def _independent_codec(result):
    h = hashlib.sha256()
    def string(value):
        raw = value.encode()
        h.update(struct.pack(">Q", len(raw)))
        h.update(raw)
    string("vibeqc.bounded-restricted-pair-ccsd.amplitudes")
    h.update(struct.pack(">Q", 1))
    string(result.input_identity_sha256)
    o = result.memory.n_occupied
    # Stored singles first, then lexicographic unordered pairs, WITHOUT
    # reverse duplicates, zero-rank padding, occupied weights or metadata.
    for i in range(o):
        for value in result.singles_copy(i):
            h.update(struct.pack(">d", 0.0 if value == 0 else float(value)))
    for i in range(o):
        for j in range(i, o):
            for value in result.pair_copy(i, j).flat:
                h.update(struct.pack(">d", 0.0 if value == 0 else float(value)))
    return h.hexdigest()


@pytest.mark.parametrize("sr,pr", [([3, 3], [3, 3, 3]), ([1, 2], [0, 2, 1]),
                                  ([0, 0], [0, 0, 0]), ([0, 1], [2, 0, 0])])
@pytest.mark.parametrize("iterations", [1, 2])
def test_exact_final_snapshot_codec_and_admitted_ragged_inventory(sr, pr, iterations):
    problem = _problem(singles_ranks=sr, pair_ranks=pr, rotated=True, nonzero_initial=True)
    result = _run(problem, options=_options(maximum_iterations=iterations))
    p = _validation(result)
    assert p.n_occupied == 2
    assert p.pair_count == 3
    assert p.record_count == 5
    assert p.singles_elements == sum(sr)
    assert p.doubles_elements == sum(r*r for r in pr)
    assert p.numerical_lanes == p.singles_elements + p.doubles_elements
    assert p.retained_numerical_bytes == 8 * p.numerical_lanes
    assert p.retained_record_bytes == 16 * p.record_count
    assert p.payload_message_bytes == 8 + len("vibeqc.bounded-restricted-pair-ccsd.amplitudes") + 8 + 8 + 64 + 8*p.numerical_lanes
    assert p.fixed_codec_payload_bytes == 65
    assert p.fixed_control_storage_bytes >= p.fixed_codec_payload_bytes
    assert p.work_units == p.metadata_work_units + p.payload_work_units
    assert result.payload_sha256 == _independent_codec(result)
    _verify(result)


def test_converged_and_unfinished_owners_verify_without_claiming_convergence():
    problem = _problem(o=1, n=1)
    converged = _run(problem)
    unfinished = _run(problem, options=_options(maximum_iterations=1))
    assert converged.final_snapshot.converged
    assert not unfinished.final_snapshot.converged
    _verify(converged)
    _verify(unfinished)
    assert converged.payload_sha256 == _independent_codec(converged)
    assert unfinished.payload_sha256 == _independent_codec(unfinished)


def test_zero_and_cap_minus_one_are_rejected_and_exact_cap_replays():
    result = _run(_problem(o=1, n=2), options=_options(maximum_iterations=1))
    p = _validation(result)
    with pytest.raises(ValueError, match="positive"):
        _verify(result, 0)
    with pytest.raises((ValueError, RuntimeError), match="work cap exceeded"):
        _verify(result, p.work_units - 1)
    _verify(result, p.work_units)
    _verify(result, p.work_units + 1)


def test_result_owns_snapshot_after_initial_input_arrays_are_destroyed():
    problem = _problem(nonzero_initial=True)
    result = _run(problem, options=_options(maximum_iterations=1))
    expected = result.payload_sha256
    for array in problem["st"] + problem["pt"]:
        array[:] = 9.0
    del problem
    gc.collect()
    _verify(result)
    assert result.payload_sha256 == expected == _independent_codec(result)
