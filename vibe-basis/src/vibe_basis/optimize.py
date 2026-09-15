"""Optimizer driver wrappers for vibe-basis recipes.

Standalone module — no vibe-qc dependency.  Provides:

* ``optimize_nlopt`` — NLopt BOBYQA (derivative-free, Powell-family).
  The primary driver for basis-set optimization because each SCF
  evaluation costs minutes of wall time on a remote host.  BOBYQA
  is the modern successor to the MINUIT2 Powell driver that the
  2013 pob paper used.
* ``optimize_scipy`` — scipy L-BFGS-B (gradient-friendly, uses FD).
  Best for cheap evaluations (e.g., when the backend is in-process).
* ``optimize_minuit`` — iminuit MIGRAD + HESSE.
  Publication-grade parameter uncertainties.  Use at the converged
  BOBYQA point.

All three return a common :class:`OptResult`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np


@dataclass
class OptResult:
    """Outcome of one optimizer run.

    Attributes
    ----------
    x : np.ndarray
        Best-found parameter vector (in optimizer space).
    fun : float
        Objective value at ``x``.
    success : bool
        Convergence flag from the underlying driver.
    message : str
        Driver-specific status message.
    n_evaluations : int
        Number of objective evaluations performed.
    wall_seconds : float
        Wall-clock time of the optimization.
    driver : str
        ``"nlopt"``, ``"scipy"``, or ``"minuit"``.
    history : list[tuple[np.ndarray, float]]
        ``(x_k, f_k)`` pairs in evaluation order.  Useful for
        convergence plots and debugging.
    raw : Any
        The driver's native result object.
    """

    x: np.ndarray
    fun: float
    success: bool
    message: str
    n_evaluations: int
    wall_seconds: float
    driver: str
    history: list[tuple[np.ndarray, float]] = field(default_factory=list)
    raw: Any = None


# ---------------------------------------------------------------------------
# NLopt BOBYQA — derivative-free, Powell-family
# ---------------------------------------------------------------------------


def optimize_nlopt(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    bounds: Sequence[tuple[float, float]],
    tol: float = 1e-4,
    max_eval: int = 200,
    initial_step: Optional[float] = None,
    rho_begin: Optional[float] = None,
    record_history: bool = True,
) -> OptResult:
    """Minimize *objective* with NLopt BOBYQA.

    BOBYQA (Bound Optimization BY Quadratic Approximation) is a
    derivative-free trust-region method in the Powell lineage.
    It's the best wall-time choice when each evaluation costs
    minutes of external-SCF + queue latency.

    Parameters
    ----------
    objective
        ``Callable[[np.ndarray], float]`` — evaluated inside the
        loop.  Must return ``np.inf`` for failed evaluations
        (non-converged SCF, transport error, …).  BOBYQA will
        route around infeasible points.
    x0
        Starting guess in optimizer space.
    bounds
        List of (lower, upper) pairs per parameter.  Must be finite.
    tol
        Convergence tolerance on the objective gradient norm.
    max_eval
        Maximum number of objective evaluations.
    initial_step
        Initial trust-region radius.  Defaults to 0.5 × min(bounds width).
    rho_begin
        Alias for *initial_step* (BOBYQA parameter name).
    record_history
        If True, record every (x_k, f_k) pair.

    Returns
    -------
    OptResult

    Raises
    ------
    RuntimeError
        If ``nlopt`` is not installed.  ``pip install nlopt`` or
        ``pip install vibe-basis[nlopt]``.
    """
    try:
        import nlopt  # type: ignore[import-not-found]
    except ImportError:
        raise RuntimeError(
            "optimize_nlopt needs nlopt installed; pip install nlopt"
        ) from None

    n = len(x0)
    if len(bounds) != n:
        raise ValueError(f"bounds length {len(bounds)} != n_params {n}")

    _history: list[tuple[np.ndarray, float]] = []
    _call_count = [0]

    def _wrap_objective(
        f: Callable[[np.ndarray], float], record: bool
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        def wrapped(x_in: np.ndarray, _grad: np.ndarray) -> float:
            x = np.asarray(x_in, dtype=float)
            v = float(f(x))
            _call_count[0] += 1
            if record:
                _history.append((x.copy(), v))
            return v

        return wrapped

    opt = nlopt.opt(nlopt.LN_BOBYQA, n)
    opt.set_min_objective(_wrap_objective(objective, record_history))
    opt.set_lower_bounds([float(lo) for lo, _ in bounds])
    opt.set_upper_bounds([float(hi) for _, hi in bounds])
    opt.set_ftol_rel(tol)
    opt.set_maxeval(max_eval)

    step = initial_step or rho_begin
    if step is not None:
        opt.set_initial_step(step)

    t0 = time.perf_counter()
    x_out = np.asarray(x0, dtype=float).copy()
    try:
        x_out = opt.optimize(x_out)
        f_best = opt.last_optimum_value()
        # BOBYQA considers itself converged when f reaches ftol;
        # check return code.
        rc = opt.last_optimize_result()
        success = rc > 0  # NLOPT_SUCCESS = 1, NLOPT_XTOL_REACHED = 4, etc.
        if rc < 0:
            msg = f"NLopt error {rc}: {_nlopt_message(rc)}"
        else:
            msg = f"NLopt converged (code {rc})"
    except nlopt.RoundoffLimited:
        success = True
        msg = "NLopt: roundoff-limited"
        f_best = opt.last_optimum_value()
        x_out = np.asarray(x0, dtype=float)
    except RuntimeError as e:
        success = False
        msg = f"NLopt error: {e}"
        f_best = float("nan")
    dt = time.perf_counter() - t0

    return OptResult(
        x=np.asarray(x_out, dtype=float),
        fun=float(f_best),
        success=success,
        message=msg,
        n_evaluations=_call_count[0],
        wall_seconds=dt,
        driver="nlopt",
        history=list(_history),
        raw=opt,
    )


def _nlopt_message(code: int) -> str:
    messages = {
        1: "SUCCESS",
        2: "STOPVAL_REACHED",
        3: "FTOL_REACHED",
        4: "XTOL_REACHED",
        5: "MAXEVAL_REACHED",
        6: "MAXTIME_REACHED",
        -1: "GENERIC FAILURE",
        -2: "INVALID_ARGS",
        -3: "OUT_OF_MEMORY",
        -4: "ROUNDOFF_LIMITED",
        -5: "FORCED_STOP",
    }
    return messages.get(code, f"unknown ({code})")


# ---------------------------------------------------------------------------
# scipy L-BFGS-B — gradient-friendly, for cheap evaluations
# ---------------------------------------------------------------------------


def optimize_scipy(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    bounds: Optional[Sequence[tuple[Optional[float], Optional[float]]]] = None,
    method: str = "L-BFGS-B",
    tol: float = 1e-6,
    max_iter: int = 200,
    record_history: bool = True,
) -> OptResult:
    """Minimize *objective* with ``scipy.optimize.minimize``.

    Uses numerical finite-difference gradients.  Best when
    evaluations are cheap (< 1 s) — e.g., in-process SCF.
    """
    try:
        from scipy.optimize import minimize  # type: ignore[import-not-found]
    except ImportError:
        raise RuntimeError(
            "optimize_scipy needs scipy installed; pip install scipy"
        ) from None

    history: list[tuple[np.ndarray, float]] = []

    def f(x: np.ndarray) -> float:
        v = objective(x)
        if record_history:
            history.append((np.asarray(x, dtype=float).copy(), float(v)))
        return v

    t0 = time.perf_counter()
    bnds = list(bounds) if bounds is not None else None
    res = minimize(
        f,
        np.asarray(x0, dtype=float),
        method=method,
        bounds=bnds,
        tol=tol,
        options={"maxiter": max_iter},
    )
    dt = time.perf_counter() - t0
    return OptResult(
        x=np.asarray(res.x, dtype=float),
        fun=float(res.fun),
        success=bool(res.success),
        message=str(res.message),
        n_evaluations=int(getattr(res, "nfev", 0)),
        wall_seconds=dt,
        driver="scipy",
        history=history,
        raw=res,
    )


# ---------------------------------------------------------------------------
# iminuit MIGRAD — publication-grade uncertainties
# ---------------------------------------------------------------------------


def optimize_minuit(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    bounds: Optional[Sequence[tuple[Optional[float], Optional[float]]]] = None,
    labels: Optional[list[str]] = None,
    tol: float = 1e-6,
    max_iter: int = 200,
    record_history: bool = True,
) -> OptResult:
    """Minimize *objective* with iminuit MIGRAD + HESSE.

    Same Minuit2 / ROOT library used by the original 2013 pob recipe.
    HESSE error analysis at the converged point gives publication-
    grade parameter uncertainties.
    """
    try:
        from iminuit import Minuit  # type: ignore[import-not-found]
    except ImportError:
        raise RuntimeError(
            "optimize_minuit needs iminuit installed; pip install iminuit"
        ) from None

    history: list[tuple[np.ndarray, float]] = []
    n = len(x0)
    names = labels or [f"p{i}" for i in range(n)]

    def f(*args: float) -> float:
        x = np.asarray(args, dtype=float)
        v = objective(x)
        if record_history:
            history.append((x.copy(), float(v)))
        return v

    f._parameters = {nm: tuple([None, None]) for nm in names}

    m = Minuit(f, *x0, name=names)
    m.errordef = 1.0
    if bounds is not None:
        for nm, (lo, hi) in zip(names, bounds, strict=True):
            m.limits[nm] = (lo, hi)

    t0 = time.perf_counter()
    m.migrad(ncall=max_iter, iterate=5)
    dt = time.perf_counter() - t0

    x_best = np.asarray(m.values, dtype=float)
    return OptResult(
        x=x_best,
        fun=float(m.fval if m.fval is not None else float("nan")),
        success=bool(m.valid),
        message=("converged" if m.valid else f"MIGRAD: {m.fmin}"),
        n_evaluations=len(history),
        wall_seconds=dt,
        driver="minuit",
        history=history,
        raw=m,
    )
