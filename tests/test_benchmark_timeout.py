"""Tests for vibeqc.benchmark.compare_calculators timeout + isolation.

Covers the contract:

* ``timeout_s=None`` keeps the historical in-process serial path.
* ``timeout_s`` set spawns a child process per calculator and reports
  ``status='timeout'`` past the deadline (without leaking a runaway
  worker into the parent test runner).
* Unpicklable calculator instances raise ``ValueError`` with migration
  guidance, rather than failing silently or hanging.
* Zero-arg factory callables work both with and without a timeout.
* Subprocess and in-process paths produce numerically-equal results
  for a well-behaved calculator.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from vibeqc.benchmark import compare_calculators


# ---------------------------------------------------------------------------
# Test calculators — defined at module level so multiprocessing.spawn
# can re-import them in the child process.
# ---------------------------------------------------------------------------


class _ConstantEnergyCalc(Calculator):
    """Trivial calculator: returns a fixed energy and zero forces.

    Picklable, no external dependencies, instant to evaluate. Used as
    the baseline for "subprocess path returns the same numbers as the
    in-process path".
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, energy: float = -1.234, **kwargs):
        Calculator.__init__(self, **kwargs)
        self._energy = float(energy)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results["energy"] = self._energy
        self.results["forces"] = np.zeros((len(self.atoms), 3))


class _HangingCalc(Calculator):
    """Calculator whose calculate() never returns in any reasonable test run.

    Used to verify that ``timeout_s`` actually kills the child process
    rather than letting the benchmark hang.
    """

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        # 1 hour — well past any reasonable test deadline; if the
        # timeout machinery is broken the test runner will hit its
        # own per-test timeout and surface a clear failure.
        time.sleep(3600.0)
        self.results["energy"] = 0.0


def _factory_constant() -> Calculator:
    """Module-level factory — picklable by multiprocessing.spawn."""
    return _ConstantEnergyCalc(energy=-2.5)


def _factory_hanging() -> Calculator:
    return _HangingCalc()


def _atoms() -> Atoms:
    return Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


# ---------------------------------------------------------------------------
# In-process (timeout_s=None) preserves historical behaviour.
# ---------------------------------------------------------------------------


def test_in_process_calculator_instance_returns_energy():
    res = compare_calculators(
        _atoms(),
        [("const", _ConstantEnergyCalc(energy=-1.5))],
        properties=("energy",),
    )
    assert len(res) == 1
    (row,) = res.rows
    assert row.status == "ok"
    assert row.properties["energy"] == pytest.approx(-1.5)


def test_in_process_factory_callable_works_without_timeout():
    # Factories are accepted on the in-process path too, for API
    # symmetry with the timeout path. Constructed once and run inline.
    res = compare_calculators(
        _atoms(),
        [("factory", _factory_constant)],
        properties=("energy",),
    )
    (row,) = res.rows
    assert row.status == "ok"
    assert row.properties["energy"] == pytest.approx(-2.5)


# ---------------------------------------------------------------------------
# timeout_s validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0, -0.001])
def test_nonpositive_timeout_rejected(bad):
    with pytest.raises(ValueError, match="timeout_s"):
        compare_calculators(
            _atoms(),
            [("const", _ConstantEnergyCalc())],
            timeout_s=bad,
        )


# ---------------------------------------------------------------------------
# Subprocess timeout path — well-behaved calculator should match
# in-process numbers and report status='ok'.
# ---------------------------------------------------------------------------


def test_subprocess_path_returns_same_energy_as_in_process():
    expected = -2.5
    res = compare_calculators(
        _atoms(),
        [("subproc", _ConstantEnergyCalc(energy=expected))],
        properties=("energy", "forces"),
        timeout_s=60.0,
    )
    (row,) = res.rows
    assert row.status == "ok", f"unexpected status={row.status!r} err={row.error!r}"
    assert row.properties["energy"] == pytest.approx(expected)
    np.testing.assert_allclose(row.properties["forces"], np.zeros((2, 3)))
    assert row.wall_time_s is not None and row.wall_time_s >= 0.0


def test_subprocess_path_accepts_factory():
    res = compare_calculators(
        _atoms(),
        [("via-factory", _factory_constant)],
        properties=("energy",),
        timeout_s=60.0,
    )
    (row,) = res.rows
    assert row.status == "ok"
    assert row.properties["energy"] == pytest.approx(-2.5)


# ---------------------------------------------------------------------------
# Hanging calculator → status='timeout' within a bounded wall time.
# ---------------------------------------------------------------------------


def test_hanging_calculator_is_timed_out():
    timeout = 2.0
    t0 = time.perf_counter()
    res = compare_calculators(
        _atoms(),
        [("hang", _factory_hanging)],
        properties=("energy",),
        timeout_s=timeout,
    )
    elapsed = time.perf_counter() - t0

    (row,) = res.rows
    assert row.status == "timeout", (
        f"expected status='timeout', got {row.status!r} (error={row.error!r})"
    )
    assert row.error is not None and "timeout_s" in row.error
    # Generous upper bound: cold spawn + import vibeqc + terminate
    # handshake. If we ever exceed this, something is wrong with the
    # kill path rather than slow CI hardware.
    assert elapsed < timeout + 30.0, (
        f"timeout enforcement took {elapsed:.1f}s for timeout_s={timeout}s — "
        f"child process was not killed promptly"
    )


def test_hang_does_not_block_subsequent_calculators():
    """A timed-out calculator should not poison the rest of the batch."""
    res = compare_calculators(
        _atoms(),
        [
            ("hang", _factory_hanging),
            ("good", _factory_constant),
        ],
        properties=("energy",),
        timeout_s=2.0,
    )
    assert len(res) == 2
    hang_row, good_row = res.rows
    assert hang_row.status == "timeout"
    assert good_row.status == "ok"
    assert good_row.properties["energy"] == pytest.approx(-2.5)


# ---------------------------------------------------------------------------
# Unpicklable instance → ValueError with migration guidance.
# ---------------------------------------------------------------------------


def test_unpicklable_calculator_under_timeout_raises_value_error():
    # Closures over local state aren't picklable by multiprocessing.spawn.
    # Wrapping one in a class makes the resulting *instance* unpicklable.
    class _LocalCalc(Calculator):
        implemented_properties = ["energy"]

        def calculate(
            self, atoms=None, properties=("energy",), system_changes=all_changes
        ):
            Calculator.calculate(self, atoms, properties, system_changes)
            self.results["energy"] = 0.0

    with pytest.raises(ValueError) as exc_info:
        compare_calculators(
            _atoms(),
            [("local", _LocalCalc())],
            properties=("energy",),
            timeout_s=5.0,
        )
    msg = str(exc_info.value)
    # Migration guidance must name the factory pattern so the user
    # knows what to do next.
    assert "factory" in msg.lower()
    assert "local" in msg  # the offending label, for context


# ---------------------------------------------------------------------------
# print_table renders the new statuses without crashing.
# ---------------------------------------------------------------------------


def test_print_table_handles_timeout_row(capsys):
    res = compare_calculators(
        _atoms(),
        [("hang", _factory_hanging)],
        properties=("energy",),
        timeout_s=2.0,
    )
    res.print_table()
    out = capsys.readouterr().out
    assert "hang" in out
    assert "timeout" in out
    assert "error:" in out  # the └─ error: ... line under the row
