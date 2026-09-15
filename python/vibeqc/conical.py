"""Penalty-function conical-intersection (MECI) optimizer -- method-agnostic.

A minimum-energy conical intersection (MECI) is the lowest-energy geometry at
which two electronic states are degenerate.  This module locates one using the
**Levine-Coe-Martínez penalty-constrained** scheme, which needs *only the two
state energies and their nuclear gradients* -- no nonadiabatic derivative
coupling vector.  That makes it the right partner for the finite-difference CIS
gradients in :mod:`vibeqc.excited_gradient` (which deliver exactly energies +
gradients, no coupling vector), and it keeps the optimizer reference-agnostic:
anything that can hand back two tracked states' ``(E, gradE)`` at a geometry -- HF
or MSINDO CIS -- can be driven to its MECI.

**The penalty function** (Levine, Coe & Martínez, *J. Phys. Chem. B* **112**,
405 (2008), Eq. 7).  With the upper/lower state energies ``E_J >= E_I`` and gap
``ΔE = E_J - E_I``,

    F_s(R) = 1/2(E_I + E_J) + s . G(ΔE),     G(ΔE) = ΔE^2 / (ΔE + a)

The averaged-energy term pulls the geometry downhill on the mean surface; the
penalty term ``s.G`` (with fixed smoothing ``a``) drives the gap to zero, and
``s`` is escalated until the gap is below threshold.  The gradient is

    gradF_s = 1/2(gradE_I + gradE_J) + s . G'(ΔE) . (gradE_J - gradE_I),
    G'(ΔE) = (ΔE^2 + 2aΔE) / (ΔE + a)^2

-- the average gradient plus a gap-difference term whose prefactor ``s.G'(ΔE)``
vanishes as ``ΔE -> 0``.  That structure -- ``1/2.gradsum + f(ΔE).graddiff`` -- is the
coupling-vector-free penalty gradient (the same combination MSINDO assembles in
``conicalintersect.f:126-137``; the functional form + constants here are the
published Levine-Coe-Martínez ones, not MSINDO's empirical log-penalty).

Because the upper/lower assignment is made by *energy ordering at each
evaluation*, the optimizer is robust to the two states swapping along the path:
``1/2(gradE_I+gradE_J)`` is assignment-independent, and the gap-difference term's
prefactor ``-> 0`` at the crossing, so ``F_s`` and ``gradF_s`` stay continuous through
it even though the individual state gradients do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .excited_gradient import ANGSTROM_TO_BOHR
from .progress import resolve_progress

__all__ = [
    "MECIResult",
    "lcm_penalty",
    "penalty_objective",
    "optimize_conical_intersection",
    "make_cis_meci_fn",
]

# A two-state energy+gradient function: coords (Angstrom, (natom, 3)) -> the two
# tracked states' energies (2,) in Hartree and gradients (2, natom, 3) in
# Hartree/bohr.  The optimizer assigns upper/lower by energy itself.
StateEGFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class MECIResult:
    """Result of a penalty-function MECI optimization."""

    coords_angstrom: np.ndarray   # optimized geometry (natom, 3)
    e_lower: float                # lower state energy at the MECI (Ha)
    e_upper: float                # upper state energy at the MECI (Ha)
    gap: float                    # E_upper - E_lower (Ha) -- ~0 at a true MECI
    sigma: float                  # final penalty weight reached
    alpha: float                  # penalty smoothing parameter used
    converged: bool               # gap < gap_tol and ‖gradF_s‖inf < fmax
    n_macro: int                  # s-escalation cycles run
    n_energy_evals: int           # state_eg_fn calls (energy+gradient)
    history: List[dict] = field(default_factory=list)   # per-macro diagnostics

    @property
    def gap_ev(self) -> float:
        return float(self.gap) * 27.211386245988


def lcm_penalty(delta_e: float, sigma: float, alpha: float) -> Tuple[float, float]:
    """Levine-Coe-Martínez penalty ``s.G(ΔE)`` and its derivative ``s.G'(ΔE)``.

    ``G(ΔE) = ΔE^2/(ΔE+a)``, ``G'(ΔE) = (ΔE^2+2aΔE)/(ΔE+a)^2``.  ``delta_e`` is the
    (non-negative) gap ``E_upper - E_lower``.  ``alpha > 0`` keeps the
    denominator positive, so the penalty is smooth through the degeneracy.
    Returns ``(penalty_value, d_penalty_d_deltaE)`` (both scaled by ``s``).
    """
    dE = float(delta_e)
    denom = dE + alpha
    g = dE * dE / denom
    g_prime = (dE * dE + 2.0 * alpha * dE) / (denom * denom)
    return sigma * g, sigma * g_prime


def penalty_objective(energies: np.ndarray, gradients: np.ndarray,
                      sigma: float, alpha: float
                      ) -> Tuple[float, np.ndarray, float]:
    """Penalty objective ``F_s`` and gradient ``gradF_s`` for two states.

    ``energies`` is ``(2,)`` (Hartree) and ``gradients`` is ``(2, natom, 3)``
    (Hartree/bohr) for the two tracked states, in any order -- the upper/lower
    roles are assigned here by energy so the result is continuous through a state
    crossing.  Returns ``(F_sigma, grad_F_sigma (natom, 3), gap)``.
    """
    e = np.asarray(energies, float)
    g = np.asarray(gradients, float)
    lo, hi = (0, 1) if e[0] <= e[1] else (1, 0)
    e_lo, e_hi = e[lo], e[hi]
    g_lo, g_hi = g[lo], g[hi]
    gap = float(e_hi - e_lo)
    pen, dpen = lcm_penalty(gap, sigma, alpha)
    f_val = 0.5 * (e_lo + e_hi) + pen
    grad = 0.5 * (g_lo + g_hi) + dpen * (g_hi - g_lo)
    return float(f_val), grad, gap


def optimize_conical_intersection(state_eg_fn: StateEGFn, coords_angstrom, *,
                                  alpha: float = 0.02, sigma0: float = 3.5,
                                  sigma_growth: float = 2.0,
                                  max_sigma: float = 1.0e4,
                                  gap_tol: float = 1.0e-4, fmax: float = 1.0e-3,
                                  max_macro: int = 12, max_micro: int = 60,
                                  verbose: bool = False) -> MECIResult:
    """Locate a minimum-energy conical intersection by the penalty method.

    ``state_eg_fn(coords_angstrom) -> (energies (2,), gradients (2, natom, 3))``
    returns the two tracked states' energies (Ha) and nuclear gradients
    (Ha/bohr).  Build one with :func:`make_cis_meci_fn` from a CIS energy
    function (HF or MSINDO).

    The algorithm minimizes the Levine-Coe-Martínez penalty ``F_s`` (see the
    module docstring) at a fixed ``s`` with L-BFGS-B, then escalates
    ``s <- s.sigma_growth`` and re-minimizes until the state gap falls below
    ``gap_tol`` (with the penalty-objective gradient below ``fmax``) or
    ``max_macro`` escalations / ``max_sigma`` is reached.  ``alpha`` is the fixed
    penalty smoothing (Hartree).  Energies in Ha, lengths in Angstrom, forces in
    Ha/bohr.
    """
    from scipy.optimize import minimize

    plog = resolve_progress(bool(verbose), verbose=2)
    C0 = np.asarray(coords_angstrom, float)
    if C0.ndim != 2 or C0.shape[1] != 3:
        raise ValueError("coords_angstrom must have shape (natom, 3)")
    natom = C0.shape[0]

    counter = {"n": 0}

    def eval_states(C):
        counter["n"] += 1
        e, g = state_eg_fn(np.asarray(C, float).reshape(natom, 3))
        return np.asarray(e, float), np.asarray(g, float)

    # The start geometry must be evaluable (a failed reference is a usage error,
    # not something to optimize around).
    e0, g0 = eval_states(C0)
    f0, _g0, gap0 = penalty_objective(e0, g0, float(sigma0), alpha)

    # `best` tracks the last successfully evaluated geometry + its states, so a
    # trial step into a non-convergent region (a real hazard of MECI searches --
    # the average-energy pull can wander toward dissociation, where a
    # semiempirical/HF SCF stops converging) can be turned into a smooth barrier
    # that retreats toward known-good ground instead of crashing the run.
    best = {"x": C0.ravel().copy(), "e": e0, "g": g0, "f": f0}

    sigma = float(sigma0)
    x = C0.ravel().copy()
    history: List[dict] = []
    converged = False
    gap = gap0
    fmax_now = float("inf")

    for macro in range(int(max_macro)):
        def fun(xv):
            try:
                e, g = eval_states(xv)
            except Exception:
                # SCF (or its FD displacements) failed here: return a convex
                # barrier centred on the last good geometry so L-BFGS-B steps
                # back into the convergent basin.
                dx = np.asarray(xv, float) - best["x"]
                return best["f"] + 1.0 + 0.5 * float(dx @ dx), dx
            f_val, grad, _gap = penalty_objective(e, g, sigma, alpha)
            best.update(x=np.asarray(xv, float).copy(), e=e, g=g, f=f_val)
            # scipy works in Angstrom: dF/dÅ = (Ha/bohr) . (bohr/Å).
            return f_val, (grad * ANGSTROM_TO_BOHR).ravel()

        res = minimize(fun, x, jac=True, method="L-BFGS-B",
                       options={"gtol": fmax * ANGSTROM_TO_BOHR,
                                "maxiter": int(max_micro)})
        x = res.x
        # Re-evaluate at the inner-converged geometry for the true gap + force;
        # if that geometry is non-convergent, fall back to the last good one.
        try:
            e, g = eval_states(x)
        except Exception:
            x = best["x"].copy()
            e, g = best["e"], best["g"]
        _f, grad, gap = penalty_objective(e, g, sigma, alpha)
        fmax_now = float(np.max(np.abs(grad)))
        history.append({"macro": macro, "sigma": sigma, "gap": gap,
                        "fmax": fmax_now})
        plog.write_raw(
            f"[meci] macro {macro:2d}  s={sigma:9.3f}  "
            f"gap={gap:.3e} Ha  fmax={fmax_now:.3e}"
        )
        if gap < gap_tol and fmax_now < fmax:
            converged = True
            break
        if sigma >= max_sigma:
            break
        sigma *= float(sigma_growth)

    C = x.reshape(natom, 3)
    e_sorted = np.sort(e)
    return MECIResult(
        coords_angstrom=C, e_lower=float(e_sorted[0]), e_upper=float(e_sorted[1]),
        gap=float(e_sorted[1] - e_sorted[0]), sigma=sigma, alpha=alpha,
        converged=bool(converged), n_macro=len(history),
        n_energy_evals=counter["n"], history=history)


def make_cis_meci_fn(energy_fn, lower_state: int, upper_state: int, *,
                     step: float = 1e-3, atoms=None) -> StateEGFn:
    """Build a two-state energy+gradient function for the MECI optimizer.

    ``energy_fn(coords_angstrom) -> CISStateSet`` is the reference-agnostic CIS
    seam (:func:`vibeqc.excited_gradient.make_hf_cis_energy_fn` for HF,
    ``make_msindo_cis_energy_fn`` for MSINDO).  ``lower_state`` / ``upper_state``
    are state indices (0 = ground, k >= 1 = k-th CIS root) -- e.g. ``0, 1`` for an
    S1/S0 conical intersection, ``1, 2`` for S2/S1.  Each excited state is
    tracked across the finite-difference displacements (root-flip guard); see
    :func:`vibeqc.excited_gradient.cis_states_energy_and_gradients_fd`.
    """
    from .excited_gradient import cis_states_energy_and_gradients_fd

    lo, hi = int(lower_state), int(upper_state)

    def state_eg_fn(coords_angstrom):
        energies, grads = cis_states_energy_and_gradients_fd(
            energy_fn, coords_angstrom, [lo, hi], step=step, atoms=atoms)
        E = np.array([energies[lo], energies[hi]], float)
        G = np.array([grads[lo], grads[hi]], float)
        return E, G

    return state_eg_fn
