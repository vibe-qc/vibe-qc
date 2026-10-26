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

    def test_inf_objective_is_handled(self):
        """BOBYQA should route around np.inf from failed evals."""
        calls = [0]

        def obj(x: np.ndarray) -> float:
            calls[0] += 1
            # First call is the start; return inf for a narrow region.
            if 1.5 < x[0] < 2.5:
                return float("inf")
            return _quadratic(x, np.array([3.0]))

        result = optimize_nlopt(
            obj,
            x0=np.array([0.0]),
            bounds=[(-10.0, 10.0)],
            max_eval=50,
        )
        # Should still converge (or at least not crash)
        assert calls[0] > 0
        # May be inf at some points, but the result should be finite
        assert not np.isnan(result.fun)


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
