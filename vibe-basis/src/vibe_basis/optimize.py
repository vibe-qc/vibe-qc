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

**Failed evaluations.** An objective returns a non-finite value (``np.inf``
or ``nan``) for a point it could not evaluate: a non-converged SCF, a lost
job. No driver here passes that value on. BOBYQA's quadratic model, and the
finite differences of the other two, do not survive one: across 43 random
two-parameter problems in which BOBYQA evaluated a point in an infeasible
hole, returning ``inf`` left the median best value at 1.83 against 0.19 for
a finite stand-in, which did better in 40 of the 43. Each driver therefore
hands its optimiser the worst finite value seen so far plus a margin
(:class:`_FiniteStandIn`). ``history`` keeps the value the objective
returned, and the result's message counts the stand-ins. A start point that
is itself not finite stops the run unsuccessfully: there is nothing to
stand in from.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np


class _StartNotFinite(Exception):
    """The objective is not finite at the first point evaluated."""


class _FiniteStandIn:
    """Replace a non-finite objective value with a finite one.

    The stand-in is the worst finite value seen plus ``margin``; by default
    the spread of the finite values seen, and at least ``1e-3 * |worst|``,
    so it is unmistakably worse than every real point on the objective's own
    scale.
    """

    def __init__(self, margin: Optional[float] = None) -> None:
        self.margin = margin
        self.worst = -np.inf
        self.best = np.inf
        self.replaced = 0

    def __call__(self, value: float) -> float:
        if np.isfinite(value):
            self.worst = max(self.worst, value)
            self.best = min(self.best, value)
            return value
        if not np.isfinite(self.worst):
            raise _StartNotFinite()
        self.replaced += 1
        margin = self.margin
        if margin is None:
            margin = max(self.worst - self.best, 1e-3 * abs(self.worst), 1e-8)
        return self.worst + margin

    def note(self, message: str) -> str:
        if not self.replaced:
            return message
        return f"{message}; {self.replaced} non-finite evaluation(s) replaced by a finite stand-in"


_START_NOT_FINITE = "the objective is not finite at the start point; nothing to optimise from"


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
    ftol_abs: Optional[float] = None,
    xtol_abs: Optional[float] = None,
    failure_margin: Optional[float] = None,
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
        loop.  Returns a non-finite value for a failed evaluation
        (non-converged SCF, …); BOBYQA sees a finite stand-in instead
        (see the module docstring).
    x0
        Starting guess in optimizer space.
    bounds
        List of (lower, upper) pairs per parameter.  Must be finite.
    tol
        Relative tolerance on the objective (NLopt ``ftol_rel``); 0
        disables it. Relative to the objective's size, so it suits a
        normalised objective but not a total energy: on a -275 Ha cell
        the default 1e-4 stops at changes of 0.03 Ha.
    ftol_abs
        Absolute tolerance on the objective, in its own units. ``None``
        leaves it off.
    xtol_abs
        Absolute tolerance on every parameter, in optimizer space (for a
        log exponent, 0.005 is a 0.5 % change). ``None`` leaves it off.
    max_eval
        Maximum number of objective evaluations.
    initial_step
        Initial trust-region radius.  Defaults to 0.5 × min(bounds width).
    rho_begin
        Alias for *initial_step* (BOBYQA parameter name).
    record_history
        If True, record every (x_k, f_k) pair.
    failure_margin
        How far above the worst finite value a failed evaluation's
        stand-in lies, in the objective's units. ``None``: the spread of
        the finite values seen, at least 1e-3 of the worst one.

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
    stand_in = _FiniteStandIn(failure_margin)

    def _wrap_objective(
        f: Callable[[np.ndarray], float], record: bool
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        def wrapped(x_in: np.ndarray, _grad: np.ndarray) -> float:
            x = np.asarray(x_in, dtype=float)
            v = float(f(x))
            _call_count[0] += 1
            if record:
                _history.append((x.copy(), v))
            try:
                return stand_in(v)
            except _StartNotFinite:
                raise nlopt.ForcedStop() from None

        return wrapped

    opt = nlopt.opt(nlopt.LN_BOBYQA, n)
    opt.set_min_objective(_wrap_objective(objective, record_history))
    opt.set_lower_bounds([float(lo) for lo, _ in bounds])
    opt.set_upper_bounds([float(hi) for _, hi in bounds])
    opt.set_ftol_rel(tol)
    if ftol_abs is not None:
        opt.set_ftol_abs(ftol_abs)
    if xtol_abs is not None:
        opt.set_xtol_abs(xtol_abs)
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
    except nlopt.ForcedStop:
        success = False
        msg = _START_NOT_FINITE
        f_best = float("nan")
        x_out = np.asarray(x0, dtype=float)
    except nlopt.RoundoffLimited:
        success = True
        msg = "NLopt: roundoff-limited"
        # The exception carries no point, and last_optimum_value() belongs
        # to the best point, not to x0: report the best evaluation seen.
        finite = [(v, x) for x, v in _history if np.isfinite(v)]
        if finite:
            f_best, x_out = min(finite, key=lambda row: row[0])
        else:
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
        message=stand_in.note(msg),
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
    failure_margin: Optional[float] = None,
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
    stand_in = _FiniteStandIn(failure_margin)

    def f(x: np.ndarray) -> float:
        v = float(objective(x))
        if record_history:
            history.append((np.asarray(x, dtype=float).copy(), v))
        return stand_in(v)

    t0 = time.perf_counter()
    bnds = list(bounds) if bounds is not None else None
    try:
        res = minimize(
            f,
            np.asarray(x0, dtype=float),
            method=method,
            bounds=bnds,
            tol=tol,
            options={"maxiter": max_iter},
        )
    except _StartNotFinite:
        return OptResult(
            x=np.asarray(x0, dtype=float), fun=float("nan"), success=False,
            message=_START_NOT_FINITE, n_evaluations=len(history),
            wall_seconds=time.perf_counter() - t0, driver="scipy", history=history,
        )
    dt = time.perf_counter() - t0
    return OptResult(
        x=np.asarray(res.x, dtype=float),
        fun=float(res.fun),
        success=bool(res.success),
        message=stand_in.note(str(res.message)),
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
    failure_margin: Optional[float] = None,
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

    stand_in = _FiniteStandIn(failure_margin)

    def f(*args: float) -> float:
        x = np.asarray(args, dtype=float)
        v = float(objective(x))
        if record_history:
            history.append((x.copy(), v))
        return stand_in(v)

    f._parameters = {nm: tuple([None, None]) for nm in names}

    m = Minuit(f, *x0, name=names)
    m.errordef = 1.0
    if bounds is not None:
        for nm, (lo, hi) in zip(names, bounds, strict=True):
            m.limits[nm] = (lo, hi)

    t0 = time.perf_counter()
    try:
        m.migrad(ncall=max_iter, iterate=5)
    except _StartNotFinite:
        return OptResult(
            x=np.asarray(x0, dtype=float), fun=float("nan"), success=False,
            message=_START_NOT_FINITE, n_evaluations=len(history),
            wall_seconds=time.perf_counter() - t0, driver="minuit", history=history,
        )
    dt = time.perf_counter() - t0

    x_best = np.asarray(m.values, dtype=float)
    return OptResult(
        x=x_best,
        fun=float(m.fval if m.fval is not None else float("nan")),
        success=bool(m.valid),
        message=stand_in.note("converged" if m.valid else f"MIGRAD: {m.fmin}"),
        n_evaluations=len(history),
        wall_seconds=dt,
        driver="minuit",
        history=history,
        raw=m,
    )
