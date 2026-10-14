"""Optimiser driver shims with a common :class:`OptResult` return.

Five drivers are wrapped here (a sixth, ``optimize_bdiis``, lives in
:mod:`vibeqc.basis_optimization.bdiis`):

* ``optimize_scipy`` -- ``scipy.optimize.minimize``. Default method
  ``L-BFGS-B`` (numerical gradient via FD). Cheap to set up; well
  suited to one-parameter smoke tests and small problems.
* ``optimize_minuit`` -- ``iminuit.Minuit.migrad`` (the original
  ROOT/Minuit2 driver, which is what M. F. Peintinger's 2013
  python wrapper around CRYSTAL09 used). HESSE error analysis at
  the minimum, robust on correlated parameters, but a heavier
  dependency. Optional.
* ``optimize_bobyqa`` -- Py-BOBYQA (Cartis, Fiala, Marteau, Roberts
  2019; Powell 2009). Model-based derivative-free trust region for a
  *scalar* objective. Builds a quadratic interpolation model from the
  points it has already evaluated, so one step costs a handful of SCF
  runs instead of the ``2 n_params`` extra runs a finite-difference
  gradient needs. The default choice while an objective has no analytic
  energy gradient (external-program energies, SP shells, ...). Optional.
* ``optimize_dfols`` -- DFO-LS (same authors). The least-squares sibling
  of Py-BOBYQA: it needs the *residual vector* r(x) and minimises
  ``sum r_i^2``, exploiting that structure for much faster convergence.
  For a fit of energies against reference values (the cohesive-energy
  objective vibe-basis targets). A plain energy is *not* a residual:
  see the function's docstring for why it refuses one. Optional.
* ``optimize_cmaes`` -- CMA-ES (Hansen, Ostermeier 2001) through pycma.
  Population-based, derivative-free, global. The right tool for escaping
  a poor starting guess on the multimodal raw-exponent landscape, the
  wrong tool for the final high-precision polish; the recipe
  :func:`vibeqc.basis_optimization.recipes.global_then_polish.optimize_global_then_polish`
  chains it with BDIIS. Optional.

All return the common :class:`OptResult` so recipe code can swap drivers
without conditional branching. The optional backends are imported lazily
inside the driver that needs them; vibe-qc's ``[basisopt-drivers]`` extra
installs the whole set.
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
        ``"scipy"``, ``"minuit"``, ``"bdiis"``, ``"pybobyqa"``, ``"dfols"``,
        ``"cmaes"`` or a composite such as ``"cmaes+bdiis"``.
    history : list[tuple[np.ndarray, float]]
        ``(x_k, f_k)`` pairs in the order the driver evaluated them.
        Optional; populated when the driver supports it.
    raw : Any
        The driver's native result object (``OptimizeResult`` /
        ``Minuit`` / Py-BOBYQA and DFO-LS ``OptimResults`` /
        ``CMAEvolutionStrategy``). Use sparingly -- opaque cross-driver.
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


# ---------------------------------------------------------------------------
# Citation keys (resolved against python/vibeqc/output/citations/database.toml)
# ---------------------------------------------------------------------------
#
# Like BDIIS_CITATION_KEYS in bdiis.py: none of these drivers is an SCF-job
# feature with a ``[routes.*]`` entry, so the optimiser surfaces its own
# provenance. tests/basisset_dev/test_basis_opt_dfo_drivers.py pins that every
# key resolves; tests/test_citation_no_dead_entries.py finds them here.

DFO_CITATION_KEYS: tuple[str, ...] = (
    "cartis_dfols_2019",   # Py-BOBYQA and DFO-LS
    "powell_bobyqa_2009",  # the BOBYQA method itself
)
"""Citation keys for :func:`optimize_bobyqa` and :func:`optimize_dfols`."""

CMAES_CITATION_KEYS: tuple[str, ...] = (
    "hansen_cmaes_2001",
    "hansen_cma_tutorial_2016",
    "hansen_pycma_software",
)
"""Citation keys for :func:`optimize_cmaes`."""


def dfo_method_citations() -> tuple[str, ...]:
    """Citation-database keys for the Py-BOBYQA / DFO-LS drivers."""
    return DFO_CITATION_KEYS


def cmaes_method_citations() -> tuple[str, ...]:
    """Citation-database keys for the CMA-ES driver."""
    return CMAES_CITATION_KEYS


# ---------------------------------------------------------------------------
# Shared bound handling for the derivative-free drivers
# ---------------------------------------------------------------------------

Bounds = Optional[list[tuple[Optional[float], Optional[float]]]]

_UNBOUNDED = 1e20
"""Py-BOBYQA / DFO-LS spell "no bound on this side" as +-1e20."""


def _check_bounds_length(bounds: Bounds, n: int) -> None:
    if bounds is not None and len(bounds) != n:
        raise ValueError(f"bounds length {len(bounds)} != n_params {n}")


def _clip_to_box(x: np.ndarray, bounds: Bounds) -> np.ndarray:
    """Project ``x`` into the box (``None`` = unbounded on that side)."""
    if bounds is None:
        return x
    out = np.array(x, dtype=float)
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None and out[i] < lo:
            out[i] = lo
        if hi is not None and out[i] > hi:
            out[i] = hi
    return out


def _dfo_box(bounds: Bounds, n: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """``(lower, upper)`` arrays in the Py-BOBYQA / DFO-LS convention."""
    _check_bounds_length(bounds, n)
    if bounds is None:
        return None
    lo = np.array([-_UNBOUNDED if b[0] is None else float(b[0]) for b in bounds])
    hi = np.array([_UNBOUNDED if b[1] is None else float(b[1]) for b in bounds])
    if np.any(lo > hi):
        raise ValueError(f"lower bound above upper bound in {bounds!r}")
    return lo, hi


def _box_is_finite(box: Optional[tuple[np.ndarray, np.ndarray]]) -> bool:
    if box is None:
        return False
    lo, hi = box
    return bool(np.all(np.abs(lo) < _UNBOUNDED) and np.all(np.abs(hi) < _UNBOUNDED))


# ---------------------------------------------------------------------------
# Py-BOBYQA -- model-based derivative-free, scalar objective
# ---------------------------------------------------------------------------


def optimize_bobyqa(
    objective: "Objective",
    x0: np.ndarray,
    *,
    bounds: Bounds = None,
    max_evaluations: int = 500,
    rhobeg: Optional[float] = None,
    rhoend: float = 1e-8,
    noisy: bool = False,
    seek_global_minimum: bool = False,
    scaling_within_bounds: bool = False,
    record_history: bool = True,
) -> OptResult:
    """Minimise a scalar objective with Py-BOBYQA and return an :class:`OptResult`.

    BOBYQA (Bound Optimization BY Quadratic Approximation, Powell 2009)
    maintains a quadratic interpolation model of the objective on a set of
    ``2 n + 1`` (default) already-evaluated points and takes trust-region
    steps on that model. A new step therefore costs about one objective
    evaluation, against the ``2 n_params`` re-SCF evaluations a
    central-difference gradient costs ``optimize_bdiis`` / ``optimize_scipy``
    per step. Py-BOBYQA (Cartis, Fiala, Marteau, Roberts 2019) adds the
    noise-robust and multiple-restart machinery exposed below.

    Use it when the objective has no analytic gradient: external-program
    energies, SP-shell coefficients, or a penalty term the analytic path does
    not cover. With an analytic gradient available, ``optimize_bdiis`` still
    converges in fewer evaluations.

    Parameters
    ----------
    objective
        Callable ``x -> float``. May return ``+inf`` for a forbidden iterate
        (failed SCF, degenerate overlap); Py-BOBYQA treats it as a very
        poor point and steps away from it.
    x0
        Starting point (optimiser space). Projected into ``bounds`` first.
    bounds
        Per-parameter ``(lo, hi)`` box, ``None`` = unbounded on that side;
        the convention every driver in this module shares.
    max_evaluations
        Objective-evaluation budget (Py-BOBYQA ``maxfun``).
    rhobeg, rhoend
        Initial and final trust-region radius. ``rhobeg=None`` lets
        Py-BOBYQA pick ``0.1 * min(box width)`` (bounded) or
        ``0.1 * max(1, |x0|_inf)`` (unbounded). For log-space exponents a
        radius of 0.1 is a 10 % change of the exponent.
    noisy
        Py-BOBYQA ``objfun_has_noise``: sample-averaging and restart
        heuristics for an objective that is only accurate to a few digits
        (a loosely converged SCF, a queue-run external program).
    seek_global_minimum
        Py-BOBYQA's multiple-restart mode for a multimodal objective. For a
        genuinely global pre-pass prefer :func:`optimize_cmaes`.
    scaling_within_bounds
        Rescale every parameter to [0, 1] over its box before modelling.
        Needs a finite box on every parameter; silently off otherwise.
    record_history
        Record ``(x_k, f_k)`` for every evaluation.

    Returns
    -------
    OptResult
        ``driver == "pybobyqa"``; ``raw`` is Py-BOBYQA's ``OptimResults``.
    """
    try:
        import pybobyqa  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "optimize_bobyqa needs Py-BOBYQA installed; pip install Py-BOBYQA "
            "(or vibe-qc's [basisopt-drivers] extra)"
        ) from exc

    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    box = _dfo_box(bounds, n)
    x0 = _clip_to_box(x0, bounds)

    history: list[tuple[np.ndarray, float]] = []

    def f(x_in: np.ndarray) -> float:
        x = np.asarray(x_in, dtype=float)
        v = float(objective(x))
        if record_history:
            history.append((x.copy(), v))
        return v

    t0 = time.perf_counter()
    soln = pybobyqa.solve(
        f,
        x0,
        bounds=box,
        maxfun=int(max_evaluations),
        rhobeg=rhobeg,
        rhoend=rhoend,
        objfun_has_noise=noisy,
        seek_global_minimum=seek_global_minimum,
        scaling_within_bounds=bool(scaling_within_bounds and _box_is_finite(box)),
        do_logging=False,
    )
    dt = time.perf_counter() - t0
    return OptResult(
        x=np.asarray(soln.x, dtype=float),
        fun=float(soln.f),
        success=bool(soln.flag == soln.EXIT_SUCCESS),
        message=str(soln.msg),
        n_evaluations=int(soln.nf),
        wall_seconds=dt,
        driver="pybobyqa",
        history=history,
        raw=soln,
    )


# ---------------------------------------------------------------------------
# DFO-LS -- model-based derivative-free, least-squares residuals
# ---------------------------------------------------------------------------


def optimize_dfols(
    objective: Any,
    x0: np.ndarray,
    *,
    residuals: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    bounds: Bounds = None,
    max_evaluations: int = 500,
    rhobeg: Optional[float] = None,
    rhoend: float = 1e-8,
    noisy: bool = False,
    scaling_within_bounds: bool = False,
    record_history: bool = True,
) -> OptResult:
    """Minimise ``sum_i r_i(x)^2`` with DFO-LS and return an :class:`OptResult`.

    DFO-LS (Cartis, Fiala, Marteau, Roberts 2019) is Py-BOBYQA's
    least-squares sibling: it builds a *linear* model of every residual
    ``r_i`` and minimises the resulting Gauss-Newton quadratic in a trust
    region. When the objective really is a sum of squares that is far
    cheaper than modelling the scalar, so this is the driver for fitting a
    basis to reference data, ``r_i = sqrt(w_i) (E_i(x) - E_i^ref)`` (the
    cohesive-energy objective of ``vibe-basis/ROADMAP.md``).

    Why it refuses a plain scalar objective
    ---------------------------------------
    A single residual ``r = Omega`` would make DFO-LS minimise ``Omega^2``,
    which for a negative total energy drives ``Omega`` toward *zero*, i.e.
    upward, and any shifted or square-rooted encoding leaves the model
    rank-one and the solver crawling. A scalar energy therefore goes to
    :func:`optimize_bobyqa`; this driver raises ``TypeError`` unless a
    residual vector is available.

    Parameters
    ----------
    objective
        The objective being minimised, ``x -> sum r_i^2``. Consulted for a
        ``residuals(x)`` method when ``residuals`` is not passed
        (see :class:`vibeqc.basis_optimization.objective.ResidualObjective`).
    x0
        Starting point (optimiser space). Projected into ``bounds`` first.
    residuals
        ``x -> ndarray`` residual vector; overrides ``objective.residuals``.
        Entries may be ``+inf`` for a forbidden iterate.
    bounds, max_evaluations, rhobeg, rhoend, noisy, scaling_within_bounds,
    record_history
        As in :func:`optimize_bobyqa`.

    Returns
    -------
    OptResult
        ``driver == "dfols"``; ``fun`` is DFO-LS's ``sum r_i^2`` at ``x`` and
        ``raw`` its ``OptimResults`` (with ``resid`` and ``jacobian``).
    """
    resid_fn = residuals if residuals is not None else getattr(objective, "residuals", None)
    if resid_fn is None:
        raise TypeError(
            "optimize_dfols minimises a sum of squares and needs the residual "
            "vector: pass residuals=... or give the objective a .residuals(x) "
            "method. A scalar energy is not a least-squares residual (a single "
            "residual r = Omega would minimise Omega^2, which for a negative "
            "energy drives it upward); use optimize_bobyqa for scalar objectives."
        )
    try:
        import dfols  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "optimize_dfols needs DFO-LS installed; pip install DFO-LS "
            "(or vibe-qc's [basisopt-drivers] extra)"
        ) from exc

    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    box = _dfo_box(bounds, n)
    x0 = _clip_to_box(x0, bounds)

    history: list[tuple[np.ndarray, float]] = []

    def r(x_in: np.ndarray) -> np.ndarray:
        x = np.asarray(x_in, dtype=float)
        v = np.asarray(resid_fn(x), dtype=float).ravel()
        if record_history:
            history.append((x.copy(), float(v @ v)))
        return v

    t0 = time.perf_counter()
    soln = dfols.solve(
        r,
        x0,
        bounds=box,
        maxfun=int(max_evaluations),
        rhobeg=rhobeg,
        rhoend=rhoend,
        objfun_has_noise=noisy,
        scaling_within_bounds=bool(scaling_within_bounds and _box_is_finite(box)),
        do_logging=False,
    )
    dt = time.perf_counter() - t0
    return OptResult(
        x=np.asarray(soln.x, dtype=float),
        fun=float(soln.obj),
        success=bool(soln.flag == soln.EXIT_SUCCESS),
        message=str(soln.msg),
        n_evaluations=int(soln.nf),
        wall_seconds=dt,
        driver="dfols",
        history=history,
        raw=soln,
    )


# ---------------------------------------------------------------------------
# CMA-ES -- population-based, derivative-free, global
# ---------------------------------------------------------------------------


def cmaes_default_sigma(
    bounds: Bounds, n: int, *, fraction: float = 0.25, fallback: float = 0.5
) -> tuple[float, Optional[list[float]]]:
    """Initial CMA-ES step size and per-coordinate scaling from the box.

    Returns ``(sigma0, stds)``: ``sigma0 = fraction * min(finite box width)``
    and ``stds[i] = width_i / min width`` (1.0 for a parameter without a
    finite box), so that every bounded parameter starts with a step of the
    same *fraction of its own range*. With no finite width anywhere,
    ``sigma0 = fallback`` and ``stds = None``.

    The box is the *optimiser-space* box (log-exponents for a ``LOG``
    parameter), so the step is a fraction of the log-range, never of the
    physical exponent range; that is what keeps the first population inside
    a sensible exponent window instead of sampling wildly out of range.
    """
    _check_bounds_length(bounds, n)
    widths: list[Optional[float]] = []
    if bounds is not None:
        for lo, hi in bounds:
            widths.append(float(hi) - float(lo) if lo is not None and hi is not None else None)
    else:
        widths = [None] * n
    finite = [w for w in widths if w is not None and w > 0.0]
    if not finite:
        return float(fallback), None
    w_min = min(finite)
    stds = [w / w_min if (w is not None and w > 0.0) else 1.0 for w in widths]
    if all(s == 1.0 for s in stds):
        stds = None
    return float(fraction * w_min), stds


def optimize_cmaes(
    objective: "Objective",
    x0: np.ndarray,
    *,
    bounds: Bounds = None,
    sigma0: Optional[float] = None,
    max_evaluations: int = 2000,
    popsize: Optional[int] = None,
    seed: Optional[int] = None,
    tol_fun: float = 1e-9,
    tol_x: float = 1e-8,
    options: Optional[dict[str, Any]] = None,
    record_history: bool = True,
) -> OptResult:
    """Minimise a scalar objective with CMA-ES (pycma) and return an :class:`OptResult`.

    CMA-ES (Covariance Matrix Adaptation Evolution Strategy, Hansen and
    Ostermeier 2001; Hansen 2016) samples a population from a Gaussian whose
    mean, step size and covariance adapt to the ranking of the sampled
    objective values. It uses no gradient, tolerates ``+inf`` for forbidden
    iterates, and is not trapped by the local basin it starts in, which is
    what a raw-exponent landscape (non-convex, ill-conditioned, multimodal)
    needs when the starting basis is poor. It is *not* an efficient final
    polisher: hand ``res.x`` to :func:`~vibeqc.basis_optimization.optimize_bdiis`
    (the ``recipes.global_then_polish`` recipe does exactly that).

    Parameters
    ----------
    objective
        Callable ``x -> float``; ``+inf`` marks a forbidden iterate.
    x0
        Initial distribution mean (optimiser space). Projected into ``bounds``.
    bounds
        Per-parameter ``(lo, hi)`` box, ``None`` = unbounded on that side.
        Handled by pycma's bound transformation, so every sampled point the
        objective sees lies inside the box.
    sigma0
        Initial step size in optimiser space. Default: a quarter of the
        narrowest finite box width, with per-coordinate scaling so wider
        parameters get proportionally larger steps
        (:func:`cmaes_default_sigma`). Set it explicitly for an unbounded
        problem; the fallback there is 0.5 (a factor ``e^0.5`` on an
        exponent in log space).
    max_evaluations
        Objective-evaluation budget (``maxfevals``). CMA-ES evaluates a whole
        population per generation, so the last generation may overshoot it
        by up to ``popsize - 1``.
    popsize
        Population size; pycma's default is ``4 + floor(3 ln n)``.
    seed
        pycma random seed, for a reproducible run.
    tol_fun, tol_x
        pycma ``tolfun`` / ``tolx`` stop criteria (objective range over the
        recent generations, and step size times the largest axis).
    options
        Extra pycma options, merged last (``{"tolstagnation": 50}``, ...).
    record_history
        Record ``(x_k, f_k)`` for every evaluation.

    Returns
    -------
    OptResult
        ``x`` / ``fun`` are the best point ever sampled. ``success`` is
        ``True`` when a convergence criterion (not the evaluation budget)
        stopped the run and ``fun`` is finite; ``message`` lists the
        triggered criteria. ``driver == "cmaes"``; ``raw`` is the
        ``CMAEvolutionStrategy`` (``raw.result.xfavorite`` is the final
        distribution mean, often a slightly better estimate of the optimum
        than the best sample).
    """
    try:
        import cma  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "optimize_cmaes needs pycma installed; pip install cma "
            "(or vibe-qc's [basisopt-drivers] extra)"
        ) from exc

    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    _check_bounds_length(bounds, n)
    x0 = _clip_to_box(x0, bounds)

    sigma_default, stds = cmaes_default_sigma(bounds, n)
    if sigma0 is None:
        sigma0 = sigma_default
    if sigma0 <= 0.0:
        raise ValueError(f"sigma0 must be positive, got {sigma0}")

    opts: dict[str, Any] = {
        "maxfevals": int(max_evaluations),
        "tolfun": float(tol_fun),
        "tolx": float(tol_x),
        "verbose": -9,   # no console output
        "verb_log": 0,   # no outcmaes/ log directory in the cwd
    }
    if bounds is not None:
        lo = [None if b[0] is None else float(b[0]) for b in bounds]
        hi = [None if b[1] is None else float(b[1]) for b in bounds]
        opts["bounds"] = [lo, hi]
        if n == 1:
            # pycma 4.x caps each coordinate's std at a third of its bound
            # range ("maxstd_boundrange") through a per-index setter that
            # cannot address the scalar step-size vector of a one-parameter
            # problem: the first generation whose std exceeds the cap raises
            # "not yet initialized (dimension needed)". Lift the cap for
            # n == 1; the bound transform still keeps every sample inside
            # the box, and the default sigma0 is a quarter of the range.
            opts["maxstd"] = float("inf")
    if stds is not None:
        opts["CMA_stds"] = stds
    if popsize is not None:
        opts["popsize"] = int(popsize)
    if seed is not None:
        opts["seed"] = int(seed)
    if options:
        opts.update(options)

    history: list[tuple[np.ndarray, float]] = []
    n_evals = 0

    t0 = time.perf_counter()
    es = cma.CMAEvolutionStrategy(x0, float(sigma0), opts)
    while not es.stop():
        candidates = es.ask()
        values: list[float] = []
        for cand in candidates:
            # pycma's bound transform keeps candidates inside the box; the
            # projection is a no-op safeguard for a user-supplied handler.
            x = _clip_to_box(np.asarray(cand, dtype=float), bounds)
            v = float(objective(x))
            n_evals += 1
            if not np.isfinite(v):
                v = float("inf")
            if record_history:
                history.append((x.copy(), v))
            values.append(v)
        es.tell(candidates, values)
    dt = time.perf_counter() - t0

    result = es.result
    stop = dict(es.stop())
    budget_only = bool(stop) and set(stop) <= {"maxfevals", "maxiter"}
    x_best = _clip_to_box(np.asarray(result.xbest, dtype=float), bounds)
    f_best = float(result.fbest)
    success = bool(stop and not budget_only and np.isfinite(f_best))
    if success:
        message = "converged (" + ", ".join(f"{k}={v}" for k, v in stop.items()) + ")"
    elif budget_only:
        message = "evaluation budget exhausted before a CMA-ES stop criterion"
    elif not np.isfinite(f_best):
        message = "no feasible point sampled (objective non-finite everywhere)"
    else:  # pragma: no cover - pycma always reports a criterion on stop
        message = f"stopped: {stop}"
    return OptResult(
        x=x_best,
        fun=f_best,
        success=success,
        message=message,
        n_evaluations=n_evals,
        wall_seconds=dt,
        driver="cmaes",
        history=history,
        raw=es,
    )
