"""GDIIS and FIRE optimisers (Phase 6).

Geometry Direct Inversion in the Iterative Subspace (Császár-Pulay
1984, Farkas-Schlegel 1999) and the Fast Inertial Relaxation Engine
(Bitzek et al., PRL 97, 170201, 2006).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .._vibeqc_core import Molecule
from .convergence import ConvergencePolicy
from .coordinates import CartesianCoordinates, CoordinateRepresentation
from .line_search import resolve_line_search
from .optimizers import (
    GeomOptResult,
    OptimizerState,
    _make_objective,
    _make_x_fn,
    _print_header,
    _print_initial_evaluation,
    _print_step,
)
from .providers import EnergyGradientProvider, HessianProvider
from .registry import register

# ---------------------------------------------------------------------------
# GDIIS -- Geometry Direct Inversion in the Iterative Subspace
# ---------------------------------------------------------------------------


@register("gdiis")
def gdiis(
    molecule: Molecule,
    provider: EnergyGradientProvider,
    *,
    coords: Optional[CoordinateRepresentation] = None,
    max_iter: int = 100,
    conv: Optional[ConvergencePolicy] = None,
    record_trajectory: bool = True,
    progress: bool = False,
    hessian_provider: Optional[HessianProvider] = None,
    freeze_indices: Optional[Sequence[int]] = None,
    diis_size: int = 8,
    trust_radius: float = 0.3,
    hessian_init_scale: float = 1.0,
    line_search: str = "none",
    **kwargs: Any,
) -> GeomOptResult:
    r"""Geometry DIIS (Császár-Pulay 1984, Farkas-Schlegel 1999).

    Maintains a history of (geometry, error) pairs where the error
    vector for step k is

        e_k = -H^{-1} g_k

    using an approximate inverse Hessian.  A DIIS extrapolation
    produces the next geometry:

        x_{k+1} = S c_i (x_i - H^{-1} g_i)

    where the coefficients c_i minimise ‖S c_i e_i‖ subject to
    S c_i = 1.  A trust radius and steepest-descent fallback
    safeguard against DIIS overshoot.
    """
    n_atoms = len(list(molecule.atoms))
    _frozen_set: set[int] = (
        {int(i) for i in freeze_indices} if freeze_indices else set()
    )

    if coords is None:
        coords = CartesianCoordinates(n_atoms, freeze_indices=freeze_indices)

    _f_fn = _make_objective(provider, coords, molecule)
    _to_mol = _make_x_fn(coords, molecule)
    _conv = conv or ConvergencePolicy.default()

    x0 = coords.x0(molecule)
    n = len(x0)

    H_inv = np.eye(n) / hessian_init_scale

    if progress:
        _print_header("GDIIS", provider, molecule, max_iter, _conv, progress)
        _print_initial_evaluation(progress)
    e_cur, g_cur = _f_fn(x0)
    g_cur = coords.apply_frozen(x0, g_cur, _frozen_set)
    n_eval = 1

    traj_frames: list[Molecule] = []
    traj_energies: list[float] = []
    if record_trajectory:
        traj_frames.append(molecule)
        traj_energies.append(e_cur)

    if progress:
        _print_step(
            0,
            e_cur,
            g_cur,
            report=_conv.check(gradient=g_cur),
            progress=progress,
        )

    xs: list[np.ndarray] = [x0.copy()]
    errs: list[np.ndarray] = [-(H_inv @ g_cur)]
    gs: list[np.ndarray] = [g_cur.copy()]
    best_x, best_e = x0.copy(), e_cur
    delta = trust_radius

    state = OptimizerState(x=x0, energy=e_cur, gradient=g_cur, n_energy_evals=n_eval)
    converged = False

    for iteration in range(1, max_iter + 1):
        report = _conv.check(
            gradient=g_cur,
            x_old=state.x_prev,
            x_new=state.x,
            e_old=state.energy_prev,
            e_new=e_cur,
        )
        if report:
            converged = True
            break

        # DIIS extrapolation
        m = len(errs)
        if m >= 2:
            coeffs = _diis_coefficients(errs)
            if coeffs is not None:
                x_star = sum(c * xi for c, xi in zip(coeffs, xs))
                e_star = sum(c * ei for c, ei in zip(coeffs, errs))
                p = x_star + e_star - state.x
            else:
                p = -H_inv @ g_cur
        else:
            p = -H_inv @ g_cur

        p_norm = float(np.linalg.norm(p))
        if p_norm > delta and p_norm > 1e-15:
            p = p * (delta / p_norm)
        if p_norm < 1e-15:
            converged = True
            break

        x_cur = state.x
        x_new = x_cur + p
        e_new, g_new = _f_fn(x_new)
        g_new = coords.apply_frozen(x_new, g_new, _frozen_set)
        n_eval += 1

        # Safeguard: reject uphill steps
        if e_new >= e_cur:
            # Fall back to steepest descent within trust radius
            sd = -g_cur
            sd_norm = float(np.linalg.norm(sd))
            if sd_norm > 1e-15:
                sd = sd * (min(delta, sd_norm) / sd_norm)
            x_new = x_cur + sd
            e_new, g_new = _f_fn(x_new)
            g_new = coords.apply_frozen(x_new, g_new, _frozen_set)
            n_eval += 1
            if e_new >= e_cur:
                delta = max(delta * 0.5, 1e-4)
                continue

        # Update approximate inverse Hessian (BFGS)
        s = x_new - x_cur
        y = g_new - g_cur
        sy = float(np.dot(s, y))
        if sy > 1e-12:
            rho = 1.0 / sy
            I_ = np.eye(n)
            H_inv = (I_ - rho * np.outer(s, y)) @ H_inv @ (
                I_ - rho * np.outer(y, s)
            ) + rho * np.outer(s, s)

        # Trust radius adaptation
        if e_new < e_cur:
            delta = min(delta * 1.2, trust_radius * 5.0)
        else:
            delta = max(delta * 0.5, 1e-4)

        # DIIS bookkeeping
        err = -(H_inv @ g_new)
        xs.append(x_new.copy())
        gs.append(g_new.copy())
        errs.append(err)
        if len(xs) > diis_size:
            xs.pop(0)
            gs.pop(0)
            errs.pop(0)

        if e_new < best_e:
            best_x, best_e = x_new.copy(), e_new

        state.x_prev = x_cur
        state.energy_prev = e_cur
        state.x = x_new
        state.energy = e_new
        state.gradient = g_new
        state.iteration = iteration
        state.n_energy_evals = n_eval
        x_cur, e_cur, g_cur = x_new, e_new, g_new

        if record_trajectory:
            traj_frames.append(_to_mol(x_cur))
            traj_energies.append(e_cur)
        if progress:
            _print_step(
                iteration,
                e_cur,
                g_cur,
                prev_energy=state.energy_prev,
                step=state.x - state.x_prev,
                report=_conv.check(
                    gradient=g_cur,
                    x_old=state.x_prev,
                    x_new=state.x,
                    e_old=state.energy_prev,
                    e_new=e_cur,
                ),
                progress=progress,
            )

    if not converged:
        report = _conv.check(gradient=g_cur, e_old=state.energy_prev, e_new=e_cur)
        converged = bool(report)

    mol_final = _to_mol(best_x)
    return GeomOptResult(
        system=mol_final,
        energy=best_e,
        gradient=state.gradient,
        converged=converged,
        n_iter=state.iteration,
        n_energy_evals=state.n_energy_evals,
        optimizer="gdiis",
        trajectory_frames=traj_frames if record_trajectory else [],
        trajectory_energies=traj_energies if record_trajectory else [],
        convergence_report=report,
        message="converged" if converged else "max_iter reached",
    )


# ---------------------------------------------------------------------------
# FIRE -- Fast Inertial Relaxation Engine (Bitzek et al. 2006)
# ---------------------------------------------------------------------------


@register("fire")
def fire(
    molecule: Molecule,
    provider: EnergyGradientProvider,
    *,
    coords: Optional[CoordinateRepresentation] = None,
    max_iter: int = 500,
    conv: Optional[ConvergencePolicy] = None,
    record_trajectory: bool = True,
    progress: bool = False,
    hessian_provider: Optional[HessianProvider] = None,
    freeze_indices: Optional[Sequence[int]] = None,
    dt: float = 0.1,
    dt_max: float = 1.0,
    dt_min: float = 0.01,
    alpha_init: float = 0.1,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    f_alpha: float = 0.99,
    n_min: int = 5,
    line_search: str = "none",
    **kwargs: Any,
) -> GeomOptResult:
    r"""Fast Inertial Relaxation Engine (FIRE).

    A damped molecular-dynamics optimiser from Bitzek et al.
    (PRL 97, 170201, 2006).  Well-suited for large systems where
    Hessian construction is prohibitive.

    The velocity is mixed with the gradient direction; when the
    velocity and force are aligned (P > 0), the time step is increased
    and the velocity is only lightly damped.  When they oppose
    (P < 0), the system is frozen (v = 0) and restarted with a
    smaller time step.

    Parameters
    ----------
    dt : float
        Initial MD time step (dimensionless, default 0.1).
    dt_max, dt_min : float
        Bounds on the adaptive time step.
    alpha_init : float
        Initial velocity mixing parameter.
    f_inc, f_dec : float
        Time-step increase/decrease factors.
    f_alpha : float
        Velocity mixing decay factor.
    n_min : int
        Steps of positive power before increasing dt.
    """
    n_atoms = len(list(molecule.atoms))
    _frozen_set: set[int] = (
        {int(i) for i in freeze_indices} if freeze_indices else set()
    )

    if coords is None:
        coords = CartesianCoordinates(n_atoms, freeze_indices=freeze_indices)

    _f_fn = _make_objective(provider, coords, molecule)
    _to_mol = _make_x_fn(coords, molecule)
    _conv = conv or ConvergencePolicy.default()

    x0 = coords.x0(molecule)
    if progress:
        _print_header("FIRE", provider, molecule, max_iter, _conv, progress)
        _print_initial_evaluation(progress)
    e_cur, g_cur = _f_fn(x0)
    g_cur = coords.apply_frozen(x0, g_cur, _frozen_set)
    n_eval = 1

    traj_frames: list[Molecule] = []
    traj_energies: list[float] = []
    if record_trajectory:
        traj_frames.append(molecule)
        traj_energies.append(e_cur)

    if progress:
        _print_step(
            0,
            e_cur,
            g_cur,
            report=_conv.check(gradient=g_cur),
            progress=progress,
        )

    v = np.zeros(len(x0))
    alpha = alpha_init
    dt_cur = dt
    n_pos = 0

    state = OptimizerState(x=x0, energy=e_cur, gradient=g_cur, n_energy_evals=n_eval)
    converged = False

    for iteration in range(1, max_iter + 1):
        report = _conv.check(
            gradient=g_cur,
            x_old=state.x_prev,
            x_new=state.x,
            e_old=state.energy_prev,
            e_new=e_cur,
        )
        if report:
            converged = True
            break

        # FIRE operates on forces. Providers return gradients, so convert
        # gradE to force = -gradE before the inertial update.
        force = -g_cur
        force_norm = float(np.linalg.norm(force))
        if force_norm < 1e-15:
            converged = True
            break
        force_hat = force / force_norm

        # Mix velocity with the force direction.
        v = (1.0 - alpha) * v + alpha * force_norm * force_hat

        # Power: P = F . v
        P = float(np.dot(force, v))

        if P > 0.0:
            # Moving downhill -- increase time step
            n_pos += 1
            if n_pos > n_min:
                dt_cur = min(dt_cur * f_inc, dt_max)
                alpha = alpha * f_alpha
        else:
            # Moving uphill -- freeze and restart
            n_pos = 0
            dt_cur = max(dt_cur * f_dec, dt_min)
            alpha = alpha_init
            v = np.zeros_like(v)

        # MD step: x_new = x + v.dt
        x_cur = state.x
        x_new = x_cur + v * dt_cur
        e_new, g_new = _f_fn(x_new)
        g_new = coords.apply_frozen(x_new, g_new, _frozen_set)
        n_eval += 1

        state.x_prev = x_cur
        state.energy_prev = e_cur
        state.x = x_new
        state.energy = e_new
        state.gradient = g_new
        state.iteration = iteration
        state.n_energy_evals = n_eval
        x_cur, e_cur, g_cur = x_new, e_new, g_new

        if record_trajectory:
            traj_frames.append(_to_mol(x_cur))
            traj_energies.append(e_cur)
        if progress:
            _print_step(
                iteration,
                e_cur,
                g_cur,
                prev_energy=state.energy_prev,
                step=state.x - state.x_prev,
                report=_conv.check(
                    gradient=g_cur,
                    x_old=state.x_prev,
                    x_new=state.x,
                    e_old=state.energy_prev,
                    e_new=e_cur,
                ),
                progress=progress,
            )

    if not converged:
        report = _conv.check(gradient=g_cur, e_old=state.energy_prev, e_new=e_cur)
        converged = bool(report)

    mol_final = _to_mol(state.x)
    return GeomOptResult(
        system=mol_final,
        energy=state.energy,
        gradient=state.gradient,
        converged=converged,
        n_iter=state.iteration,
        n_energy_evals=state.n_energy_evals,
        optimizer="fire",
        trajectory_frames=traj_frames if record_trajectory else [],
        trajectory_energies=traj_energies if record_trajectory else [],
        convergence_report=report,
        message="converged" if converged else "max_iter reached",
    )


# ---------------------------------------------------------------------------
# DIIS coefficient solver (same as BDIIS)
# ---------------------------------------------------------------------------


def _diis_coefficients(errs: list[np.ndarray]) -> Optional[np.ndarray]:
    """Solve the DIIS least-squares for the extrapolation coefficients.

    Minimise ‖S c_i.e_i‖^2 subject to S c_i = 1 via the bordered
    linear system (Pulay 1980).
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
