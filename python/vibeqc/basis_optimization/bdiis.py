"""BDIIS -- DIIS-accelerated basis-set parameter optimisation.

``optimize_bdiis`` is a third optimiser driver alongside
:func:`~vibeqc.basis_optimization.drivers.optimize_scipy` and
:func:`~vibeqc.basis_optimization.drivers.optimize_minuit`, returning the
same :class:`~vibeqc.basis_optimization.drivers.OptResult` so recipe code
can swap drivers without branching.

Background
----------
BDIIS (*Basis-set Direct Inversion in the Iterative Subspace*) applies
Pulay's DIIS extrapolation to the optimisation of Gaussian basis-set
parameters (exponents and contraction coefficients). It is the optimiser
behind CRYSTAL23's ``OPTBASIS`` keyword, introduced for periodic CRYSTAL
calculations by

    R. Daga, B. Civalleri, L. Maschio, "Gaussian Basis Sets for
    Crystalline Solids: All-Purpose Basis Set Libraries vs System-Specific
    Optimizations", J. Chem. Theory Comput. 16, 2192 (2020),
    doi:10.1021/acs.jctc.9b01004.

BDIIS is the basis-parameter analogue of *geometry* DIIS (GDIIS):

    P. Császár, P. Pulay, J. Mol. Struct. 114, 31 (1984)         [GDIIS]
    Ö. Farkas, H. B. Schlegel, J. Chem. Phys. 111, 10806 (1999)  [robust GDIIS]
    P. Pulay, Chem. Phys. Lett. 73, 393 (1980);
    P. Pulay, J. Comput. Chem. 3, 556 (1982)                     [DIIS]

The objective is expected to already include the condition-number penalty
g.ln κ(S) -- see
:func:`vibeqc.basis_optimization.ld_diagnostics.condition_number_penalty`
and the ``use_cond_penalty`` flag on the objective factories in
:mod:`vibeqc.basis_optimization.recipes.objective`. BDIIS minimises
whatever scalar it is handed.

Where this implementation goes beyond CRYSTAL's "very basic" OPTBASIS
optimiser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* A **BFGS inverse-Hessian** is accumulated and used both for the DIIS
  error vectors (eᵢ = -H⁻¹.gᵢ) and for the quasi-Newton fallback step,
  rather than a fixed diagonal step -- curvature information accelerates
  the late-stage convergence GDIIS-on-raw-gradients cannot.
* A **trust radius + Armijo backtracking** safeguard (Farkas-Schlegel)
  guarantees the objective never increases on an accepted step. This is
  what actually tames the diffuse-exponent collapse that the
  condition-number penalty only *discourages*.
* The DIIS subspace is **conditioned**: an ill-posed extrapolation
  (runaway coefficients or a singular ``B`` matrix) falls back to the
  quasi-Newton step instead of taking a wild step into a region where
  the SCF will not converge.

Citation keys (resolved against the citation database) are exported as
:data:`BDIIS_CITATION_KEYS` and via :func:`method_citations`, so an
optimisation run can surface its provenance the same way a job's SCF log
surfaces its references.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from .drivers import OptResult

if TYPE_CHECKING:
    from .objective import Objective


# Citation keys for the method, resolved against
# python/vibeqc/output/citations/database.toml. BDIIS is not an SCF-job
# feature, so it has no `[routes.*]` entry; instead the optimiser surfaces
# these explicitly (a run that produces a basis should cite the optimiser
# the same way a paper's Methods section does). The resolve-in-database
# guard lives in tests/basisset_dev/test_bdiis_driver.py.
BDIIS_CITATION_KEYS: tuple[str, ...] = (
    "daga_optbasis_2020",
    "vandevondele_basisopt_2007",
    "pulay_diis_1980",
    "pulay_diis_1982",
)


def method_citations() -> tuple[str, ...]:
    """Citation-database keys for the BDIIS optimiser and its penalty."""
    return BDIIS_CITATION_KEYS


def _central_fd_gradient(
    f: Callable[[np.ndarray], float], x: np.ndarray, step: float
) -> np.ndarray:
    """Central finite-difference gradient df/dxᵢ ≈ (f(x+h)-f(x-h))/2h.

    Central differences (O(h^2) error) rather than scipy's forward
    differences (O(h)) -- the extra evaluation per dimension buys an order
    of accuracy, which matters near a shallow minimum in log-exponent
    space where the forward-difference bias can stall convergence.
    """
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[i] += step
        xm[i] -= step
        fp = f(xp)
        fm = f(xm)
        if not (np.isfinite(fp) and np.isfinite(fm)):
            # One-sided fallback when a probe lands in a forbidden region
            # (e.g. an exponent collapse that fails the SCF / overlap).
            f0 = f(x)
            if np.isfinite(fp):
                g[i] = (fp - f0) / step
            elif np.isfinite(fm):
                g[i] = (f0 - fm) / step
            else:
                g[i] = 0.0
        else:
            g[i] = (fp - fm) / (2.0 * step)
    return g


def _clip_to_bounds(
    x: np.ndarray,
    bounds: Optional[list[tuple[Optional[float], Optional[float]]]],
) -> np.ndarray:
    """Project ``x`` into the box defined by ``bounds`` (None = unbounded)."""
    if bounds is None:
        return x
    out = x.copy()
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None and out[i] < lo:
            out[i] = lo
        if hi is not None and out[i] > hi:
            out[i] = hi
    return out


def _projected_grad_inf_norm(
    x: np.ndarray,
    g: np.ndarray,
    bounds: Optional[list[tuple[Optional[float], Optional[float]]]],
) -> float:
    """inf-norm of the box-projected gradient (the KKT stationarity measure).

    For a variable pinned at a bound whose (anti-)gradient points *out* of
    the feasible box, the descent direction is blocked, so that component
    is already stationary and is projected out. At an interior point this
    reduces to ``max|g|``. Using this -- rather than the raw gradient --
    lets a constrained minimum on a bound register as converged.
    """
    if len(g) == 0:
        return 0.0
    if bounds is None:
        return float(np.max(np.abs(g)))
    pg = g.copy()
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None and x[i] <= lo + 1e-12 and g[i] > 0.0:
            pg[i] = 0.0
        elif hi is not None and x[i] >= hi - 1e-12 and g[i] < 0.0:
            pg[i] = 0.0
    return float(np.max(np.abs(pg)))


def _gdiis_coefficients(errs: list[np.ndarray]) -> Optional[np.ndarray]:
    """Solve the DIIS least-squares for the extrapolation coefficients.

    Minimise ‖Sᵢ cᵢ.eᵢ‖^2 subject to Sᵢ cᵢ = 1 via the bordered linear
    system (Pulay 1980)::

        ⎡ B   -1 ⎤ ⎡ c ⎤   ⎡ 0 ⎤
        ⎣ -1ᵀ  0 ⎦ ⎣ l ⎦ = ⎣ -1⎦      Bᵢⱼ = eᵢ.eⱼ

    Returns the coefficient vector ``c`` (length ``len(errs)``), or
    ``None`` if the system is too ill-conditioned to trust (the caller
    then falls back to a plain quasi-Newton step).
    """
    m = len(errs)
    if m == 1:
        return np.array([1.0])
    b = np.empty((m, m))
    for i in range(m):
        for j in range(i, m):
            v = float(errs[i] @ errs[j])
            b[i, j] = v
            b[j, i] = v
    # Scale B to keep the bordered system well-posed; DIIS B matrices span
    # many orders of magnitude as the gradient shrinks near convergence.
    scale = np.trace(b) / m
    if scale <= 0.0 or not np.isfinite(scale):
        return None
    b = b / scale
    a = np.zeros((m + 1, m + 1))
    a[:m, :m] = b
    a[:m, m] = -1.0
    a[m, :m] = -1.0
    rhs = np.zeros(m + 1)
    rhs[m] = -1.0
    try:
        sol = np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError:
        return None
    c = sol[:m]
    if not np.all(np.isfinite(c)):
        return None
    return c


def _bfgs_inverse_update(
    h_inv: np.ndarray, s: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """BFGS update of the inverse-Hessian approximation.

    Standard Sherman-Morrison form (Nocedal & Wright, *Numerical
    Optimization*, Eq. 6.17). Skips the update when the curvature
    condition sᵀy > 0 fails, which keeps ``h_inv`` positive definite.
    """
    sy = float(s @ y)
    if sy <= 1e-12:
        return h_inv
    n = len(s)
    rho = 1.0 / sy
    ident = np.eye(n)
    left = ident - rho * np.outer(s, y)
    right = ident - rho * np.outer(y, s)
    return left @ h_inv @ right + rho * np.outer(s, s)


def optimize_bdiis(
    objective: "Objective",
    x0: np.ndarray,
    *,
    bounds: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
    grad: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_iter: int = 200,
    tol_energy: float = 1e-7,
    tol_grad: float = 1e-4,
    diis_size: int = 8,
    fd_step: float = 1e-5,
    trust_radius: float = 0.5,
    init_hessian_scale: float = 1.0,
    max_backtracks: int = 12,
    record_history: bool = True,
) -> OptResult:
    """Optimise a basis-parameter objective with robust BDIIS / GDIIS.

    Parameters
    ----------
    objective
        Callable ``x -> float`` (the composite objective, energy +
        condition-number penalty). May return ``+inf`` for a forbidden
        iterate; the driver treats such points as non-improving and
        backtracks rather than crashing.
    x0
        Initial parameter vector (optimiser space -- typically log-exponents).
    bounds
        Per-parameter ``(lo, hi)`` box (``None`` = unbounded on that side),
        matching the scipy driver's convention. Steps are projected back
        into the box.
    grad
        Optional analytic gradient ``x -> ndarray``. Defaults to a central
        finite-difference gradient with spacing ``fd_step``.
    max_iter
        Maximum number of outer BDIIS iterations.
    tol_energy, tol_grad
        Convergence thresholds: |ΔΩ| < ``tol_energy`` **and**
        ‖g‖inf < ``tol_grad``. (CRYSTAL's ``TOLOPTE`` / ``TOLOPTG``.)
    diis_size
        Maximum number of (point, gradient) pairs kept in the DIIS
        subspace; the oldest is dropped when the history exceeds it.
    fd_step
        Finite-difference spacing for the numerical gradient.
    trust_radius
        Maximum 2-norm of a single accepted step (in optimiser space).
    init_hessian_scale
        ``H₀ = init_hessian_scale . I``; the inverse used for the first
        steps is ``I / init_hessian_scale``.
    max_backtracks
        Maximum Armijo step halvings before declaring the iterate stuck.
    record_history
        Record ``(x_k, f_k)`` per accepted iterate.

    Returns
    -------
    OptResult
        ``driver == "bdiis"``.
    """
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)

    grad_fn: Callable[[np.ndarray], np.ndarray]
    if grad is not None:
        grad_fn = grad
    else:
        grad_fn = lambda xx: _central_fd_gradient(objective, xx, fd_step)  # noqa: E731

    history: list[tuple[np.ndarray, float]] = []

    def record(x: np.ndarray, f: float) -> None:
        if record_history:
            history.append((x.copy(), float(f)))

    t0 = time.perf_counter()

    x = _clip_to_bounds(x0, bounds)
    f_cur = float(objective(x))
    if not np.isfinite(f_cur):
        return OptResult(
            x=x, fun=f_cur, success=False,
            message="initial point is infeasible (objective non-finite)",
            n_evaluations=getattr(objective, "n_calls", 0),
            wall_seconds=time.perf_counter() - t0, driver="bdiis",
            history=history,
        )
    g_cur = np.asarray(grad_fn(x), dtype=float)
    record(x, f_cur)

    h_inv = np.eye(n) / init_hessian_scale
    xs: list[np.ndarray] = [x.copy()]
    gs: list[np.ndarray] = [g_cur.copy()]
    trust = trust_radius
    best_x, best_f = x.copy(), f_cur

    success = False
    message = "max_iter reached"

    for _ in range(max_iter):
        # --- propose a step: GDIIS extrapolation, else quasi-Newton ------
        errs = [-(h_inv @ gi) for gi in gs]
        coeffs = _gdiis_coefficients(errs)
        proposal: Optional[np.ndarray] = None
        if coeffs is not None and np.max(np.abs(coeffs)) < 1e8:
            x_star = sum(c * xi for c, xi in zip(coeffs, xs))
            g_star = sum(c * gi for c, gi in zip(coeffs, gs))
            proposal = x_star - h_inv @ g_star
        if proposal is None:
            # Quasi-Newton fallback from the current point.
            proposal = x - h_inv @ g_cur

        step = proposal - x
        step_norm = float(np.linalg.norm(step))
        if step_norm > trust and step_norm > 0.0:
            step = step * (trust / step_norm)

        # --- Armijo backtracking: never accept a non-improving step ------
        accepted = False
        trial_step = step.copy()
        for _bt in range(max_backtracks):
            x_new = _clip_to_bounds(x + trial_step, bounds)
            f_new = float(objective(x_new))
            if np.isfinite(f_new) and f_new < f_cur - 1e-12 * abs(f_cur):
                accepted = True
                break
            trial_step = trial_step * 0.5
        if not accepted:
            # GDIIS/quasi-Newton both stuck -- try a damped steepest descent
            # along -g before giving up (covers the rare case where h_inv
            # has drifted to a poor estimate).
            sd = -g_cur
            sd_norm = float(np.linalg.norm(sd))
            if sd_norm > 0.0:
                sd = sd * (min(trust, sd_norm) / sd_norm)
            for _bt in range(max_backtracks):
                x_new = _clip_to_bounds(x + sd, bounds)
                f_new = float(objective(x_new))
                if np.isfinite(f_new) and f_new < f_cur - 1e-12 * abs(f_cur):
                    accepted = True
                    break
                sd = sd * 0.5
        if not accepted:
            # No downhill direction found within the trust region: either a
            # true (projected-)stationary point or a stall. Converged iff
            # the box-projected gradient is below tolerance.
            success = _projected_grad_inf_norm(x, g_cur, bounds) < tol_grad
            message = (
                "converged (no further decrease; projected gradient below tol)"
                if success
                else "stuck: no decreasing step within trust region"
            )
            break

        g_new = np.asarray(grad_fn(x_new), dtype=float)

        # --- BFGS inverse-Hessian update + trust-radius adaptation -------
        s = x_new - x
        y = g_new - g_cur
        h_inv = _bfgs_inverse_update(h_inv, s, y)
        if float(np.linalg.norm(s)) > 0.5 * trust:
            trust = min(trust * 1.5, 10.0 * trust_radius)  # confident: grow
        else:
            trust = max(trust * 0.8, 1e-4)  # cautious: shrink

        de = abs(f_new - f_cur)
        x, f_cur, g_cur = x_new, f_new, g_new
        record(x, f_cur)
        if f_cur < best_f:
            best_x, best_f = x.copy(), f_cur

        # --- DIIS subspace bookkeeping -----------------------------------
        xs.append(x.copy())
        gs.append(g_cur.copy())
        if len(xs) > diis_size:
            xs.pop(0)
            gs.pop(0)

        # --- convergence -------------------------------------------------
        if de < tol_energy and _projected_grad_inf_norm(x, g_cur, bounds) < tol_grad:
            success = True
            message = "converged (ΔΩ and projected ‖g‖inf below tolerance)"
            break

    dt = time.perf_counter() - t0
    return OptResult(
        x=best_x,
        fun=float(best_f),
        success=success,
        message=message,
        n_evaluations=getattr(objective, "n_calls", len(history)),
        wall_seconds=dt,
        driver="bdiis",
        history=history,
        raw=None,
    )
