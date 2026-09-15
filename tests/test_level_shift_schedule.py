"""Tests for the iter-indexed level-shift schedule helper.

Pure-Python module; no vibeqc-core import needed.
"""
from __future__ import annotations

import pytest

from vibeqc.level_shift_schedule import LevelShiftSchedule


def test_at_returns_per_iter_value():
    s = LevelShiftSchedule([5.0, 3.0, 1.0])
    assert s.at(1) == 5.0
    assert s.at(2) == 3.0
    assert s.at(3) == 1.0


def test_at_clamps_past_end_to_last_entry():
    """Iterations past the schedule length keep using the last entry —
    set the last entry to 0.0 to fully release the shift."""
    s = LevelShiftSchedule([5.0, 3.0, 1.0, 0.0])
    assert s.at(4) == 0.0
    assert s.at(10) == 0.0
    assert s.at(100) == 0.0


def test_at_rejects_iter_below_one():
    s = LevelShiftSchedule([1.0])
    with pytest.raises(ValueError, match="iter_idx must be >= 1"):
        s.at(0)


def test_empty_schedule_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        LevelShiftSchedule([])


def test_negative_shift_rejected():
    """Saunders-Hillier shifts must be non-negative — a negative shift
    *lowers* virtual eigvals into the occupied space and breaks
    orbital ordering. Catch that at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        LevelShiftSchedule([1.0, -0.5, 0.0])


def test_crystal_default_matches_documented_schedule():
    """CRYSTAL14 `LEVSHIFT 5 1`: 0.5,0.4,0.3,0.2,0.1,0.05,0."""
    s = LevelShiftSchedule.crystal_default()
    assert s.as_list() == [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]


def test_crystal_aggressive_starts_higher_and_decays_slower():
    s = LevelShiftSchedule.crystal_aggressive()
    seq = s.as_list()
    assert seq[0] == 1.0
    assert seq[-1] == 0.0
    # Monotone non-increasing.
    for a, b in zip(seq, seq[1:]):
        assert b <= a


def test_constant_replicates_legacy_static_shift():
    """`LevelShiftSchedule.constant(b)` is a drop-in for the legacy
    `opts.level_shift = b` convention (shift held forever)."""
    s = LevelShiftSchedule.constant(0.3)
    assert s.at(1) == 0.3
    assert s.at(100) == 0.3


def test_dataclass_is_frozen():
    """Schedule should be immutable so passing one to multiple SCF
    drivers doesn't risk accidental in-place modification."""
    s = LevelShiftSchedule([1.0])
    with pytest.raises((AttributeError, TypeError)):
        s.shifts = [2.0]  # type: ignore[misc]
