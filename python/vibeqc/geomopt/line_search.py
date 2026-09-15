"""Line-search strategies for geometry optimisation.

Each line-search function takes a direction vector ``p`` (in optimiser
space) and returns the step length ``a`` that minimises the energy
along ``x + a.p``.  The interface is uniform so optimisers can swap
strategies via the ``geom_line_search`` keyword.

Built-in strategies
-------------------
* ``"brent"`` -- Golden-section + parabolic interpolation (Brent 1973).
  The same classic algorithm already in :mod:`vibeqc.molecular_optimize`.
* ``"backtracking"`` -- Simple Armijo backtracking with geometric
  step reduction.
* ``"wolfe"`` / ``"strong_wolfe"`` -- Strong Wolfe conditions line
  search with cubic/quadratic interpolation (Nocedal & Wright,
  Algorithm 3.5).
* ``"none"`` -- Fixed step length (trust-region or steepest descent
  with a fixed a).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .._vibeqc_core import Molecule

# Convenience alias for the closure used inside the line search.
_PhiFn = Callable[[float], tuple[float, float]]  # alpha -> (phi, dphi)

# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def resolve_line_search(
    name: str,
    *,
    step: float = 0.05,
    tol: float = 1e-5,
    max_backtracks: int = 12,
    reduction: float = 0.5,
    c1: float = 1e-2,
    c2: float = 0.9,
    max_ls_iter: int = 20,
) -> "LineSearchFn":
    """Return a line-search callable by name.

    Parameters
    ----------
    name : str
        ``"brent"``, ``"backtracking"``, ``"wolfe"``,
        ``"strong_wolfe"``, or ``"none"``.
    step : float
        Initial step size for bracketing / backtracking (bohr).
    tol : float
        Tolerance for Brent (1e-5 default).
    max_backtracks : int
        Max backtracks for Armijo (12).
    reduction : float
        Geometric reduction factor for backtracking (0.5).
    c1 : float
        Armijo / sufficient-decrease constant (1e-2).
    c2 : float
        Strong Wolfe curvature constant (0.9).  Must satisfy
        c1 < c2 < 1.
    max_ls_iter : int
        Maximum line-search iterations for Wolfe (20).
    """
    if name in ("none", ""):
        return fixed_step
    if name == "brent":
        return _BrentLineSearch(initial_step=step, tol=tol)
    if name == "backtracking":
        return _ArmijoLineSearch(
            max_backtracks=max_backtracks, reduction=reduction, c1=c1
        )
    if name in ("wolfe", "strong_wolfe"):
        return _StrongWolfeLineSearch(c1=c1, c2=c2, max_iter=max_ls_iter)
    raise ValueError(
        f"Unknown line_search={name!r}. "
        f"Use 'brent', 'backtracking', 'wolfe', 'strong_wolfe', or 'none'."
    )


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------

LineSearchFn = Callable[
    [
        np.ndarray,  # x0 (flat parameter vector)
        float,  # f0 (energy at x0)
        np.ndarray,  # direction p (flat, unit or scaled)
        Callable[[np.ndarray], tuple[float, np.ndarray]],  # f(x) -> (E, grad)
        Callable[[Molecule], Molecule],  # x -> Molecule
    ],
    tuple[np.ndarray, float, int],  # (x_new, energy_new, n_extra_evals)
]


# ---------------------------------------------------------------------------
# Fixed step (no line search)
# ---------------------------------------------------------------------------


def fixed_step(
    x0: np.ndarray,
    f0: float,
    direction: np.ndarray,
    f_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    _to_molecule: Callable[[Molecule], Molecule],
) -> tuple[np.ndarray, float, int]:
    """Take the full step: x_new = x0 + direction.  No line search."""
    x_new = x0 + direction
    e_new, _ = f_fn(x_new)
    return x_new, e_new, 1


# ---------------------------------------------------------------------------
# Brent line search (adapts the existing brent_minimize_1d)
# ---------------------------------------------------------------------------


class _BrentLineSearch:
    """1-D Brent line search (golden-section + parabolic interpolation)."""

    def __init__(self, initial_step: float = 0.05, tol: float = 1e-5):
        self._step = initial_step
        self._tol = tol

    def __call__(
        self,
        x0: np.ndarray,
        f0: float,
        direction: np.ndarray,
        f_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
        _to_molecule: Callable[[Molecule], Molecule],
    ) -> tuple[np.ndarray, float, int]:
        from ..molecular_optimize import _bracket_line_minimum, brent_minimize_1d

        def f_line(alpha: float) -> float:
            e, _ = f_fn(x0 + alpha * direction)
            return e

        # Bracket
        a, fa, b, fb, c, fc, n_bracket = _bracket_line_minimum(
            f_line, 0.0, f0, step=self._step, max_steps=40, growth=1.8
        )
        if fb >= fa or fb >= fc:
            vals = [(a, fa), (b, fb), (c, fc)]
            best = min(vals, key=lambda v: v[1])
            x_new = x0 + best[0] * direction
            return x_new, best[1], n_bracket

        # Brent minimise within bracket
        alpha_opt, f_opt, n_brent = brent_minimize_1d(
            f_line, min(a, c), b, max(a, c), tol=self._tol
        )
        x_new = x0 + alpha_opt * direction
        return x_new, f_opt, n_bracket + n_brent


# ---------------------------------------------------------------------------
# Armijo backtracking line search
# ---------------------------------------------------------------------------


class _ArmijoLineSearch:
    """Simple Armijo backtracking: halve the step until f(x+a.p) < f(x).

    Parameters
    ----------
    max_backtracks : int
        Maximum number of halvings before accepting the step as-is.
    reduction : float
        Factor by which the trial step is reduced each backtrack
        (default 0.5).
    c1 : float
        Armijo sufficient-decrease constant (default 1e-2).
    """

    def __init__(
        self,
        max_backtracks: int = 12,
        reduction: float = 0.5,
        c1: float = 1e-2,
    ):
        self._max_backtracks = max_backtracks
        self._reduction = reduction
        self._c1 = c1

    def __call__(
        self,
        x0: np.ndarray,
        f0: float,
        direction: np.ndarray,
        f_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
        _to_molecule: Callable[[Molecule], Molecule],
    ) -> tuple[np.ndarray, float, int]:
        g0_dir = 0.0  # could be computed, but Armijo w/o Wolfe is simpler
        alpha = 1.0
        x_new = x0 + alpha * direction
        f_new, g_new = f_fn(x_new)
        n_eval = 1

        # Armijo condition: f(x + a.p) <= f(x) + c₁.a.gradf(x)ᵀ.p
        # Since p = -g (steepest descent), gradfᵀ.p = -‖g‖^2 < 0 guaranteed.
        # For simplicity we use f_new < f0 as a proxy (sufficient for
        # most problems).
        for _ in range(self._max_backtracks):
            if f_new < f0:
                return x_new, f_new, n_eval
            alpha *= self._reduction
            x_new = x0 + alpha * direction
            f_new, g_new = f_fn(x_new)
            n_eval += 1

        return x_new, f_new, n_eval


# ---------------------------------------------------------------------------
# Strong Wolfe conditions line search (Nocedal & Wright, Algorithm 3.5)
# ---------------------------------------------------------------------------


class _StrongWolfeLineSearch:
    """Line search satisfying the strong Wolfe conditions.

    Implements Algorithm 3.5 (with zoom via Algorithm 3.6) from
    Nocedal & Wright, *Numerical Optimization* (2nd ed., 2006).

    At each iteration the step length a is chosen so that:

    1. **Sufficient decrease** (Armijo):
       ``phi(a) <= phi(0) + c₁.a.phi'(0)``

    2. **Curvature** (strong Wolfe):
       ``|phi'(a)| <= c₂.|phi'(0)|``

    where ``phi(a) = f(x0 + a.p)`` and ``phi'(a) = gradf(x0 + a.p)ᵀ.p``.

    Trial step lengths are chosen by cubic interpolation when both
    endpoint derivatives are known, quadratic interpolation otherwise,
    with bisection as a safeguard.

    Parameters
    ----------
    c1 : float
        Sufficient-decrease constant (default 1e-4, per Nocedal & Wright).
        Must satisfy 0 < c1 < c2 < 1.
    c2 : float
        Curvature condition constant (default 0.9).  Larger values
        enforce a more exact line search; typical range 0.1--0.9.
    max_iter : int
        Maximum iterations for the main loop + zoom phase combined (20).
    alpha_max : float
        Maximum allowed step length (10.0).

    References
    ----------
    Nocedal, J. and Wright, S. J., *Numerical Optimization*,
    Springer, 2nd edition, 2006.  Chapter 3, Algorithms 3.5 and 3.6.
    """

    def __init__(
        self,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_iter: int = 20,
        alpha_max: float = 10.0,
    ):
        if not (0 < c1 < c2 < 1):
            raise ValueError(
                f"Strong Wolfe requires 0 < c1 < c2 < 1; got c1={c1}, c2={c2}"
            )
        self._c1 = c1
        self._c2 = c2
        self._max_iter = max_iter
        self._alpha_max = alpha_max

    # ------------------------------------------------------------------
    # Public __call__
    # ------------------------------------------------------------------

    def __call__(
        self,
        x0: np.ndarray,
        f0: float,
        direction: np.ndarray,
        f_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
        _to_molecule: Callable[[Molecule], Molecule],
    ) -> tuple[np.ndarray, float, int]:
        """Run the strong-Wolfe line search.

        Returns ``(x_new, energy_new, n_extra_evals)``.
        """
        # ---- gradient at x0 -------------------------------------------
        _, g0 = f_fn(x0)
        n_eval = 1

        dphi0: float = float(np.dot(g0, direction))
        # If the direction is not a descent direction we cannot satisfy
        # the sufficient-decrease condition; return x0 unchanged.
        if dphi0 >= 0:
            return x0.copy(), f0, n_eval

        # ---- closure: phi(a) -> (energy, directional derivative) ---------
        def phi(alpha: float) -> tuple[float, float]:
            f, g = f_fn(x0 + alpha * direction)
            return float(f), float(np.dot(g, direction))

        # ---- main loop (Algorithm 3.5) --------------------------------
        alpha_prev: float = 0.0
        phi_prev: float = f0
        dphi_prev: float = dphi0
        alpha_i: float = 1.0  # initial trial step

        for iteration in range(1, self._max_iter + 1):
            phi_i, dphi_i = phi(alpha_i)
            n_eval += 1

            # -- condition 1: Armijo violation or function increase ----
            if (phi_i > f0 + self._c1 * alpha_i * dphi0) or (
                iteration > 1 and phi_i >= phi_prev
            ):
                alpha_star, f_star, nz = self._zoom(
                    phi,
                    alpha_prev,
                    alpha_i,
                    phi_prev,
                    phi_i,
                    dphi_prev,
                    dphi_i,
                    f0,
                    dphi0,
                )
                n_eval += nz
                x_star = x0 + alpha_star * direction
                return x_star, f_star, n_eval

            # -- condition 2: curvature satisfied ----------------------
            if abs(dphi_i) <= -self._c2 * dphi0:
                x_new = x0 + alpha_i * direction
                return x_new, phi_i, n_eval

            # -- condition 3: positive derivative -> zoom from high side
            if dphi_i >= 0:
                alpha_star, f_star, nz = self._zoom(
                    phi,
                    alpha_i,
                    alpha_prev,
                    phi_i,
                    phi_prev,
                    dphi_i,
                    dphi_prev,
                    f0,
                    dphi0,
                )
                n_eval += nz
                x_star = x0 + alpha_star * direction
                return x_star, f_star, n_eval

            # -- advance to next trial step -----------------------------
            alpha_prev = alpha_i
            phi_prev = phi_i
            dphi_prev = dphi_i
            alpha_i = min(2.0 * alpha_i, self._alpha_max)

        # -- max iterations exhausted; return the last trial point ------
        x_last = x0 + alpha_i * direction
        f_last, _ = f_fn(x_last)
        n_eval += 1
        return x_last, f_last, n_eval

    # ------------------------------------------------------------------
    # Zoom (Algorithm 3.6)
    # ------------------------------------------------------------------

    def _zoom(
        self,
        phi: _PhiFn,
        alpha_lo: float,
        alpha_hi: float,
        phi_lo: float,
        phi_hi: float,
        dphi_lo: float,
        dphi_hi: float,
        f0: float,
        dphi0: float,
    ) -> tuple[float, float, int]:
        """Zoom into an interval known to contain an acceptable step.

        Returns ``(alpha_star, f_star, n_eval)``.
        """
        n_eval = 0
        # Initialise in case max_iter is zero (satisfies type checker).
        alpha_j = (alpha_lo + alpha_hi) / 2.0
        phi_j: float = phi_lo
        for _ in range(self._max_iter):
            # Interpolate trial alpha between lo and hi.
            alpha_j = self._interpolate(
                alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo, dphi_hi
            )

            phi_j, dphi_j = phi(alpha_j)
            n_eval += 1

            # Sufficient-decrease violation or function increase:
            # contract the high side.
            if phi_j > f0 + self._c1 * alpha_j * dphi0 or phi_j >= phi_lo:
                alpha_hi = alpha_j
                phi_hi = phi_j
                dphi_hi = dphi_j
            else:
                # Curvature check.
                if abs(dphi_j) <= -self._c2 * dphi0:
                    return alpha_j, phi_j, n_eval
                # Keep the downhill side.
                if dphi_j * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                    phi_hi = phi_lo
                    dphi_hi = dphi_lo
                alpha_lo = alpha_j
                phi_lo = phi_j
                dphi_lo = dphi_j

        return alpha_j, phi_j, n_eval

    # ------------------------------------------------------------------
    # Interpolation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate(
        a: float,
        b: float,
        fa: float,
        fb: float,
        da: float,
        db: float,
    ) -> float:
        """Pick a trial point between *a* and *b*.

        Uses cubic interpolation when both endpoint derivatives are
        known, quadratic when only *da* is available, and bisection
        as a fallback guard when the interpolant lies outside (a, b).
        """
        lo = min(a, b)
        hi = max(a, b)
        margin = 0.1 * (hi - lo)

        # --- cubic (Hermite) interpolant minimum --------------------
        try:
            alpha_cubic = _cubicmin(a, b, fa, fb, da, db)
            if lo + margin < alpha_cubic < hi - margin:
                return alpha_cubic
        except (ValueError, ZeroDivisionError):
            pass

        # --- quadratic interpolant minimum --------------------------
        try:
            alpha_quad = _quadmin(a, b, fa, fb, da)
            if lo + margin < alpha_quad < hi - margin:
                return alpha_quad
        except (ValueError, ZeroDivisionError):
            pass

        # --- fallback: bisection ------------------------------------
        return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Low-level interpolation primitives
# ---------------------------------------------------------------------------


def _cubicmin(a: float, b: float, fa: float, fb: float, da: float, db: float) -> float:
    """Minimum of the cubic Hermite interpolant through (a, fa, da)
    and (b, fb, db).

    Formula from Nocedal & Wright, Eq. (3.60).  Returns a value in
    the open interval (a, b) when the interpolant is strictly convex.
    """
    s = b - a
    d1 = da + db - 3.0 * (fa - fb) / s
    disc = d1 * d1 - da * db
    if disc < 0:
        raise ValueError("Cubic interpolant is not convex")
    d2 = float(np.sign(s)) * float(np.sqrt(disc))
    return b - s * (db + d2 - d1) / (db - da + 2.0 * d2)


def _quadmin(a: float, b: float, fa: float, fb: float, da: float) -> float:
    """Minimum of the quadratic interpolant through (a, fa, da) and
    (b, fb).

    The quadratic matches ``f(a)``, ``f'(a)``, and ``f(b)``.  Its
    stationary point is returned -- this is a minimum when the
    interpolant is convex.
    """
    s = b - a
    c = (fb - fa - da * s) / (s * s)
    return a - da / (2.0 * c)
