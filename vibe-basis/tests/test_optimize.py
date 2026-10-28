"""Tests for vibe_basis.optimize — NLopt BOBYQA, scipy, iminuit wrappers.

All tests use MockEnergy (a pure Python quadratic) — no CRYSTAL14,
no vibe-qc, no external optimizer binary needed if scipy is available.
NLopt and iminuit tests are skipped if the packages aren't installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibe_basis.optimize import (
    OptResult,
    optimize_minuit,
    optimize_nlopt,
    optimize_scipy,
)

# ---------------------------------------------------------------------------
# Mock objective — simple quadratic
# ---------------------------------------------------------------------------


def _quadratic(x: np.ndarray, center: np.ndarray = np.zeros(1)) -> float:
    d = x - center
    return float(0.5 * (d @ d))


# ---------------------------------------------------------------------------
# NLopt BOBYQA
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("nlopt"),
    reason="nlopt not installed",
)
class TestOptimizeNlopt:
    def test_finds_center_of_1d_quadratic(self):
        result = optimize_nlopt(
            lambda x: _quadratic(x, np.array([3.0])),
            x0=np.array([0.0]),
            bounds=[(-10.0, 10.0)],
            max_eval=30,
        )
        assert result.success
        assert result.x[0] == pytest.approx(3.0, abs=0.05)
        assert result.fun == pytest.approx(0.0, abs=0.01)

    def test_respects_lower_bound(self):
        result = optimize_nlopt(
            lambda x: _quadratic(x, np.array([-5.0])),
            x0=np.array([0.0]),
            bounds=[(0.0, 10.0)],  # optimum at -5 is outside
            max_eval=30,
        )
        # Should converge to or near the bound (0.0)
        assert result.x[0] >= -0.01
        # Energy at bound (0) should be > energy at true min (-5)
        assert result.fun > 0.0

    def test_2d_quadratic(self):
        result = optimize_nlopt(
            lambda x: _quadratic(x, np.array([2.0, -1.0])),
            x0=np.array([0.0, 0.0]),
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            max_eval=50,
        )
        assert result.success
        assert result.x[0] == pytest.approx(2.0, abs=0.1)
        assert result.x[1] == pytest.approx(-1.0, abs=0.1)

    def test_a_failed_evaluation_does_not_stall_bobyqa(self):
        """One infinite value stalled raw BOBYQA at f = 5.66 on this problem
        (a quadratic with an infeasible disc); the finite stand-in reaches
        the minimum. History keeps the value the objective returned."""
        target, centre = np.array([1.485, -0.555]), np.array([0.393, -1.763])

        def obj(x: np.ndarray) -> float:
            if np.linalg.norm(x - centre) < 0.433:
                return float("inf")
            return float(np.sum((x - target) ** 2))

        result = optimize_nlopt(
            obj, x0=np.array([-1.062, -2.099]), bounds=[(-5.0, 5.0)] * 2,
            tol=0.0, xtol_abs=1e-6, max_eval=80, initial_step=1.5,
        )
        assert result.fun < 1e-10
        assert result.x == pytest.approx(target, abs=1e-4)
        assert "1 non-finite evaluation(s) replaced by a finite stand-in" in result.message
        assert sum(not np.isfinite(v) for _, v in result.history) == 1

    def test_a_start_point_that_fails_stops_unsuccessfully(self):
        result = optimize_nlopt(lambda x: float("nan"), x0=np.zeros(2), bounds=[(-1.0, 1.0)] * 2)
        assert not result.success and np.isnan(result.fun)
        assert "not finite at the start point" in result.message
        assert result.n_evaluations == 1

    def test_absolute_tolerances_hold_on_a_total_energy_scale(self):
        """ftol_rel is relative to |f|: on a -7e5 kJ/mol cell energy it stops
        long before the minimum. The absolute tolerances do not."""
        offset = -723_000.0

        def energy(x: np.ndarray) -> float:
            return offset + 50.0 * float((x[0] - 0.4) ** 2)

        loose = optimize_nlopt(energy, x0=np.array([-1.0]), bounds=[(-3.0, 3.0)],
                               initial_step=0.3)
        tight = optimize_nlopt(energy, x0=np.array([-1.0]), bounds=[(-3.0, 3.0)],
                               initial_step=0.3, tol=0.0, ftol_abs=1e-6,
                               xtol_abs=1e-4)
        assert abs(loose.x[0] - 0.4) > 0.1
        assert tight.x[0] == pytest.approx(0.4, abs=1e-3)


# ---------------------------------------------------------------------------
# scipy L-BFGS-B
# ---------------------------------------------------------------------------


class TestOptimizeScipy:
    def test_finds_center_of_1d_quadratic(self):
        result = optimize_scipy(
            lambda x: _quadratic(x, np.array([3.0])),
            x0=np.array([0.0]),
            bounds=[(None, None)],
            max_iter=30,
        )
        assert result.success
        assert result.x[0] == pytest.approx(3.0, abs=0.01)

    def test_without_bounds(self):
        result = optimize_scipy(
            lambda x: _quadratic(x, np.array([-1.0])),
            x0=np.array([5.0]),
            max_iter=30,
        )
        assert result.success
        assert result.x[0] == pytest.approx(-1.0, abs=0.01)

    def test_history_is_recorded(self):
        result = optimize_scipy(
            lambda x: _quadratic(x),
            x0=np.array([10.0]),
            bounds=[(-20.0, 20.0)],
            max_iter=20,
            record_history=True,
        )
        assert len(result.history) > 0
        # First entry should be at x0
        assert result.history[0][0][0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# iminuit MIGRAD
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("iminuit"),
    reason="iminuit not installed",
)
class TestOptimizeMinuit:
    def test_finds_center_of_1d_quadratic(self):
        result = optimize_minuit(
            lambda x: _quadratic(x, np.array([3.0])),
            x0=np.array([0.0]),
            bounds=[(None, None)],
            max_iter=30,
        )
        assert result.success
        assert result.x[0] == pytest.approx(3.0, abs=0.1)

    def test_respects_bounds(self):
        result = optimize_minuit(
            lambda x: _quadratic(x, np.array([-5.0])),
            x0=np.array([0.0]),
            bounds=[(0.0, 10.0)],
            max_iter=30,
        )
        assert result.x[0] >= -0.01


# ---------------------------------------------------------------------------
# OptResult dataclass
# ---------------------------------------------------------------------------


class TestOptResult:
    def test_fields(self):
        r = OptResult(
            x=np.array([1.0, 2.0]),
            fun=0.5,
            success=True,
            message="ok",
            n_evaluations=25,
            wall_seconds=1.2,
            driver="nlopt",
        )
        assert r.success
        assert r.driver == "nlopt"
        assert len(r.x) == 2


def test_the_stand_in_is_the_worst_finite_value_plus_a_margin():
    from vibe_basis.optimize import _FiniteStandIn, _StartNotFinite

    with pytest.raises(_StartNotFinite):
        _FiniteStandIn()(float("inf"))
    s = _FiniteStandIn()
    assert [s(v) for v in (-10.0, -12.0)] == [-10.0, -12.0]
    assert s(float("nan")) == pytest.approx(-10.0 + 2.0)        # the spread
    one = _FiniteStandIn()
    one(-275.0)
    assert one(float("inf")) == pytest.approx(-275.0 + 0.275)   # 1e-3 of |worst|
    fixed = _FiniteStandIn(margin=100.0)
    fixed(5.0)
    assert fixed(float("inf")) == 105.0 and fixed.replaced == 1


def test_scipy_carries_on_past_a_failed_evaluation():
    """Raw L-BFGS-B stops at the start with ABNORMAL on this objective."""
    pytest.importorskip("scipy")

    def obj(x: np.ndarray) -> float:
        return float("nan") if 3.0 < x[0] < 9.0 else float((x[0] - 2.0) ** 2)

    result = optimize_scipy(obj, x0=np.array([0.0]), bounds=[(-10.0, 10.0)])
    assert result.success and result.x[0] == pytest.approx(2.0, abs=1e-6)
    assert "replaced by a finite stand-in" in result.message
    bad = optimize_scipy(lambda x: float("inf"), x0=np.array([0.0]), bounds=[(-1.0, 1.0)])
    assert not bad.success and "not finite at the start point" in bad.message
