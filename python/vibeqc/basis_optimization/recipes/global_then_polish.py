"""Global pre-pass with CMA-ES, then a BDIIS polish: one call, one OptResult.

The three quasi-Newton / model-based drivers (``optimize_bdiis``,
``optimize_scipy``, ``optimize_bobyqa``) are local: they converge to the
minimum of the basin they start in. On the raw-exponent landscape (non-convex,
ill-conditioned, multimodal) a poor starting basis can leave them in the wrong
basin. CMA-ES is population-based and derivative-free, so it can leave a
basin, but it is an inefficient final polisher: near the minimum a
quasi-Newton step with a BFGS inverse-Hessian beats a population update on
evaluation count by a wide margin.

This recipe composes the two the way they are meant to be used:

1. :func:`~vibeqc.basis_optimization.drivers.optimize_cmaes` for at most
   ``cmaes_max_evaluations`` evaluations from ``x0`` with an initial step
   derived from the *optimiser-space* (log-exponent) box, then
2. :func:`~vibeqc.basis_optimization.bdiis.optimize_bdiis` from the best
   CMA-ES sample, with the analytic gradient when the caller has one.

Use::

    from vibeqc.basis_optimization.recipes.global_then_polish import (
        optimize_global_then_polish)
    res = optimize_global_then_polish(
        objective, x0, bounds=par.optim_bounds(), grad=gradient,
        cmaes_max_evaluations=300, cmaes_seed=1)
    res.driver          # "cmaes+bdiis"
    res.raw["cmaes"]    # the CMA-ES stage's OptResult
    res.raw["bdiis"]    # the BDIIS stage's OptResult

The objective is whatever scalar the caller hands over, normally
``E + gamma ln kappa(S)`` from
:func:`vibeqc.basis_optimization.recipes.objective.make_native_objective_gradient`;
the recipe does not add or change penalty terms.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np

from ..bdiis import BDIIS_CITATION_KEYS, optimize_bdiis
from ..drivers import (
    CMAES_CITATION_KEYS,
    Bounds,
    OptResult,
    optimize_cmaes,
)

GLOBAL_THEN_POLISH_CITATION_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(CMAES_CITATION_KEYS + BDIIS_CITATION_KEYS)
)
"""Citation keys for both stages of the recipe, in stage order, deduplicated."""


def method_citations() -> tuple[str, ...]:
    """Citation-database keys for the CMA-ES stage and the BDIIS stage."""
    return GLOBAL_THEN_POLISH_CITATION_KEYS


def optimize_global_then_polish(
    objective: Any,
    x0: np.ndarray,
    *,
    bounds: Bounds = None,
    grad: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    cmaes_max_evaluations: int = 400,
    cmaes_sigma0: Optional[float] = None,
    cmaes_popsize: Optional[int] = None,
    cmaes_seed: Optional[int] = None,
    cmaes_tol_fun: float = 1e-6,
    cmaes_options: Optional[dict[str, Any]] = None,
    bdiis_max_iter: int = 200,
    bdiis_tol_energy: float = 1e-7,
    bdiis_tol_grad: float = 1e-4,
    bdiis_kwargs: Optional[dict[str, Any]] = None,
    record_history: bool = True,
) -> OptResult:
    """Run CMA-ES from ``x0``, then polish its best sample with BDIIS.

    Parameters
    ----------
    objective
        Scalar objective ``x -> float`` (``+inf`` for a forbidden iterate),
        shared by both stages.
    x0
        Starting point, the initial CMA-ES distribution mean (optimiser space).
    bounds
        Per-parameter ``(lo, hi)`` box, ``None`` = unbounded on that side.
        Both stages respect it; CMA-ES also derives its initial step from it.
    grad
        Optional analytic gradient for the BDIIS stage. ``None`` lets BDIIS
        finite-difference the objective.
    cmaes_max_evaluations, cmaes_sigma0, cmaes_popsize, cmaes_seed,
    cmaes_tol_fun, cmaes_options
        Passed to :func:`optimize_cmaes`. The pre-pass only has to land in
        the right basin, so ``cmaes_tol_fun`` is loose by default (1e-6):
        stopping CMA-ES early and letting BDIIS finish is what saves
        evaluations.
    bdiis_max_iter, bdiis_tol_energy, bdiis_tol_grad, bdiis_kwargs
        Passed to :func:`optimize_bdiis` (``bdiis_kwargs`` for the rest of
        its keyword arguments: ``diis_size``, ``trust_radius``, ...).
    record_history
        Record the evaluation history of both stages, concatenated.

    Returns
    -------
    OptResult
        ``x`` / ``fun`` are the better of the two stages' best points
        (normally BDIIS's, which never accepts an increase from its start).
        ``success`` is the BDIIS stage's convergence flag,
        ``n_evaluations`` and ``wall_seconds`` sum both stages,
        ``driver == "cmaes+bdiis"``, and ``raw`` is
        ``{"cmaes": OptResult, "bdiis": OptResult}``.
    """
    t0 = time.perf_counter()
    stage1 = optimize_cmaes(
        objective,
        np.asarray(x0, dtype=float),
        bounds=bounds,
        sigma0=cmaes_sigma0,
        max_evaluations=cmaes_max_evaluations,
        popsize=cmaes_popsize,
        seed=cmaes_seed,
        tol_fun=cmaes_tol_fun,
        options=cmaes_options,
        record_history=record_history,
    )
    if not np.isfinite(stage1.fun):
        # Nothing feasible found: polishing from x0 would repeat the local
        # driver's own behaviour; report the failure instead of hiding it.
        return OptResult(
            x=stage1.x,
            fun=stage1.fun,
            success=False,
            message=f"cmaes: {stage1.message}; bdiis: skipped (no feasible start)",
            n_evaluations=stage1.n_evaluations,
            wall_seconds=time.perf_counter() - t0,
            driver="cmaes+bdiis",
            history=list(stage1.history),
            raw={"cmaes": stage1, "bdiis": None},
        )

    extra = dict(bdiis_kwargs or {})
    stage2 = optimize_bdiis(
        objective,
        stage1.x,
        bounds=bounds,
        grad=grad,
        max_iter=bdiis_max_iter,
        tol_energy=bdiis_tol_energy,
        tol_grad=bdiis_tol_grad,
        record_history=record_history,
        **extra,
    )
    dt = time.perf_counter() - t0

    best = stage2 if stage2.fun <= stage1.fun else stage1
    return OptResult(
        x=np.asarray(best.x, dtype=float),
        fun=float(best.fun),
        success=bool(stage2.success),
        message=f"cmaes: {stage1.message}; bdiis: {stage2.message}",
        n_evaluations=int(stage1.n_evaluations) + int(stage2.n_evaluations),
        wall_seconds=dt,
        driver="cmaes+bdiis",
        history=list(stage1.history) + list(stage2.history),
        raw={"cmaes": stage1, "bdiis": stage2},
    )
