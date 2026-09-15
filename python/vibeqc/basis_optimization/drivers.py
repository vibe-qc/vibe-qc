"""Optimiser driver shims with a common :class:`OptResult` return.

Two drivers are wrapped:

* ``optimize_scipy`` -- ``scipy.optimize.minimize``. Default method
  ``L-BFGS-B`` (numerical gradient via FD). Cheap to set up; well
  suited to one-parameter smoke tests and small problems.
* ``optimize_minuit`` -- ``iminuit.Minuit.migrad`` (the original
  ROOT/Minuit2 driver, which is what M. F. Peintinger's 2013
  python wrapper around CRYSTAL09 used). HESSE error analysis at
  the minimum, robust on correlated parameters, but a heavier
  dependency. Optional.

The two return a common :class:`OptResult` so recipe code can swap
drivers without conditional branching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from .objective import Objective


@dataclass
class OptResult:
    """Outcome of one optimiser run.

    Attributes
    ----------
    x : np.ndarray
        Best-found parameter vector (in optimiser space).
    fun : float
        Objective value at ``x``.
    success : bool
        Convergence flag from the underlying driver.
    message : str
        Driver-specific status message.
    n_evaluations : int
        Number of objective evaluations performed.
    wall_seconds : float
        Wall-clock time of the optimisation.
    driver : str
        ``"scipy"`` or ``"minuit"``.
    history : list[tuple[np.ndarray, float]]
        ``(x_k, f_k)`` pairs in the order the driver evaluated them.
        Optional; populated when the driver supports it.
    raw : Any
        The driver's native result object (``OptimizeResult`` /
        ``Minuit``). Use sparingly -- opaque cross-driver.
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


def optimize_scipy(
    objective: "Objective",
    x0: np.ndarray,
    *,
    bounds: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
    method: str = "L-BFGS-B",
    tol: float = 1e-6,
    max_iter: int = 200,
    record_history: bool = True,
) -> OptResult:
    """Wrap ``scipy.optimize.minimize`` and return an :class:`OptResult`."""
    from scipy.optimize import minimize  # type: ignore[import-not-found]

    history: list[tuple[np.ndarray, float]] = []

    def f(x: np.ndarray) -> float:
        v = objective(x)
        if record_history:
            history.append((np.asarray(x, dtype=float).copy(), float(v)))
        return v

    t0 = time.perf_counter()
    res = minimize(
        f,
        np.asarray(x0, dtype=float),
        method=method,
        bounds=bounds,
        tol=tol,
        options={"maxiter": max_iter},
    )
    dt = time.perf_counter() - t0
    return OptResult(
        x=np.asarray(res.x, dtype=float),
        fun=float(res.fun),
        success=bool(res.success),
        message=str(res.message),
        n_evaluations=int(getattr(res, "nfev", objective.n_calls)),
        wall_seconds=dt,
        driver="scipy",
        history=history,
        raw=res,
    )


def optimize_minuit(
    objective: "Objective",
    x0: np.ndarray,
    *,
    bounds: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
    labels: Optional[list[str]] = None,
    error_init: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_iter: int = 200,
    record_history: bool = True,
    errordef: float = 1.0,
) -> OptResult:
    """Wrap ``iminuit.Minuit.migrad`` (Minuit2 underneath) and return :class:`OptResult`.

    ``errordef`` is Minuit's UP parameter -- 1.0 for chi^2-style objectives
    (default; appropriate for a sum of energies treated as a
    least-squares-like target), 0.5 for log-likelihoods. Energy
    minimisation is neither cleanly, but 1.0 gives reasonable HESSE
    error scales for our use.
    """
    try:
        from iminuit import Minuit  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "optimize_minuit needs iminuit installed; pip install iminuit"
        ) from exc

    history: list[tuple[np.ndarray, float]] = []
    n = len(x0)
    names = labels or [f"p{i}" for i in range(n)]

    def f(*args: float) -> float:
        x = np.asarray(args, dtype=float)
        v = objective(x)
        if record_history:
            history.append((x.copy(), float(v)))
        return v

    f._parameters = {nm: tuple([None, None]) for nm in names}  # iminuit signature hint

    m = Minuit(f, *x0, name=names)
    m.errordef = errordef
    if error_init is not None:
        if len(error_init) != n:
            raise ValueError(f"error_init length {len(error_init)} != n_params {n}")
        for nm, e in zip(names, error_init, strict=True):
            m.errors[nm] = float(e)
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
        message=("converged" if m.valid else "MIGRAD did not converge: "
                 f"{m.fmin}"),
        n_evaluations=objective.n_calls,
        wall_seconds=dt,
        driver="minuit",
        history=history,
        raw=m,
    )
