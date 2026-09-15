"""Trust-region and rational-function optimisation (Phase 3).

Geometry optimisers that solve the trust-region subproblem

    min_p  m(p) = f + g^T p + 1/2 p^T B p    subject to  ‖p‖ <= Δ

using the More-Sorensen algorithm (secular equation for the boundary
level-shift l).  Powell's r test adapts the trust radius across steps.

Two variants:
* :func:`trust_region` -- standard trust-region minimisation (B >= 0).
* :func:`rfo` -- rational-function optimisation using an augmented
  Hessian (Banerjee, Adams, Simons & Shepard, J. Phys. Chem. 89, 52,
  1985; doi:10.1021/j100247a015).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .._vibeqc_core import Molecule
from .convergence import ConvergencePolicy
from .coordinates import CartesianCoordinates, CoordinateRepresentation
from .hessian_update import resolve_hessian_update
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
# Trust-region subproblem solver (More-Sorensen)
# ---------------------------------------------------------------------------


def _trust_region_step(
    g: np.ndarray,
    B: np.ndarray,
    delta: float,
    max_ms_iter: int = 30,
    ms_tol: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Solve the trust-region subproblem via More-Sorensen.

    Find p minimising m(p) = 1/2 p^T B p + g^T p subject to ‖p‖ <= Δ.

    If the unconstrained Newton step p = -B⁻¹g satisfies ‖p‖ <= Δ,
    return it directly.  Otherwise solve the secular equation

        ‖p(l)‖ = Δ    where    p(l) = -(B + lI)⁻¹g

    for l > 0 via safeguarded Newton iteration on phi(l) = 1/‖p(l)‖ - 1/Δ.

    Returns (p, l) where l is the level shift (0 if interior step).
    """
    n = len(g)

    # Try the Newton step first (l = 0).
    try:
        # Use a stable solve -- Cholesky if B is positive-definite.
        L = np.linalg.cholesky(B)
        p_newton = -np.linalg.solve(L.T, np.linalg.solve(L, g))
    except np.linalg.LinAlgError:
        # B is indefinite or singular -- fall back to LU.
        try:
            p_newton = -np.linalg.solve(B, g)
        except np.linalg.LinAlgError:
            # B is singular -- go directly to boundary solve.
            p_newton = None

    if p_newton is not None:
        pn_norm = float(np.linalg.norm(p_newton))
        if pn_norm <= delta * (1.0 + 1e-12):
            return p_newton, 0.0

    # Boundary solve: find l > 0 such that ‖p(l)‖ = Δ.
    # Use the secular equation phi(l) = 1/‖p(l)‖ - 1/Δ = 0.
    # Newton: l_{k+1} = l_k - phi(l_k) / phi'(l_k).

    # Eigenvalue bounds for l: l > max(0, -l_min(B))
    try:
        eigvals = np.linalg.eigvalsh(B)
        lambda_l = max(0.0, -eigvals[0] + 1e-6)
    except np.linalg.LinAlgError:
        lambda_l = 1e-6

    lam = max(lambda_l + 0.1, 0.1)

    for _ in range(max_ms_iter):
        B_shifted = B + lam * np.eye(n)
        try:
            L = np.linalg.cholesky(B_shifted)
            p = -np.linalg.solve(L.T, np.linalg.solve(L, g))
        except np.linalg.LinAlgError:
            lam = max(lam * 2.0, lambda_l + 1e-4)
            continue

        p_norm = float(np.linalg.norm(p))
        if p_norm < 1e-15:
            return np.zeros(n), lam

        phi = 1.0 / p_norm - 1.0 / delta
        if abs(phi * delta) < ms_tol:
            return p, lam

        # phi'(l) = (p^T (B + lI)⁻¹ p) / ‖p‖^3
        try:
            q = np.linalg.solve(L.T, np.linalg.solve(L, p))
        except np.linalg.LinAlgError:
            q = np.linalg.solve(B_shifted, p)
        phi_prime = float(p @ q) / (p_norm**3)
        if abs(phi_prime) < 1e-15:
            return p, lam

        lam_new = lam - phi / phi_prime
        if lam_new <= lambda_l:
            lam = (lam + lambda_l) / 2.0
        else:
            lam = lam_new

    # Fallback: Cauchy point (steepest descent within trust region).
    g_norm = float(np.linalg.norm(g))
    if g_norm < 1e-15:
        return np.zeros(n), lam
    gBg = float(g @ B @ g)
    if gBg <= 0.0:
        tau = 1.0
    else:
        tau = min(1.0, g_norm**3 / (delta * gBg))
    return -tau * delta * g / g_norm, lam


# ---------------------------------------------------------------------------
# Powell's r test for trust-radius adaptation
# ---------------------------------------------------------------------------


def _powell_rho(
    f_old: float,
    f_new: float,
    g: np.ndarray,
    B: np.ndarray,
    p: np.ndarray,
) -> float:
    """Powell's r = actual_reduction / predicted_reduction.

    r ≈ 1 -> model is accurate -> grow trust radius.
    r < 0.25 -> model is poor -> shrink trust radius.
    r < 0 -> step was uphill -> reject step.
    """
    actual = f_old - f_new
    predicted = -(float(g @ p) + 0.5 * float(p @ B @ p))
    if abs(predicted) < 1e-15:
        return 0.0
    return actual / predicted


# ---------------------------------------------------------------------------
# Trust-region optimiser
# ---------------------------------------------------------------------------


@register("trust")
def trust_region(
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
    hessian_init: str = "diagonal",
    hessian_init_scale: float = 1.0,
    hessian_update: str = "bfgs",
    initial_trust_radius: float = 0.3,
    max_trust_radius: float = 1.0,
    min_trust_radius: float = 1e-4,
    rho_shrink: float = 0.25,
    rho_expand: float = 0.75,
    trust_shrink_factor: float = 0.5,
    trust_expand_factor: float = 2.0,
    # Not used but accepted for uniform interface
    line_search: str = "none",
    **kwargs: Any,
) -> GeomOptResult:
    r"""Trust-region geometry optimisation with Powell's r adaptation.

    At each step, solves the trust-region subproblem

        min_p  m(p) = g^T p + 1/2 p^T B p    s.t.  ‖p‖ <= Δ

    using a More-Sorensen secular-equation solver.  The Hessian model B
    is initialised as a scaled identity and updated via the chosen
    ``hessian_update`` formula after each accepted step.

    The trust radius Δ is adapted by Powell's r test:

    * r > r_expand -> Δ <- min(Δ x t_expand, Δ_max)
    * r < r_shrink -> Δ <- max(Δ x t_shrink, Δ_min)
    * r < 0 -> step rejected, Δ <- Δ x t_shrink, recompute with smaller radius.

    Parameters
    ----------
    initial_trust_radius : float
        Starting trust radius in bohr (default 0.3).
    max_trust_radius : float
        Maximum trust radius (default 1.0).
    min_trust_radius : float
        Minimum trust radius before declaring a stall (default 1e-4).
    rho_shrink, rho_expand : float
        Powell r thresholds for shrinking/expanding Δ.
    trust_shrink_factor, trust_expand_factor : float
        Multiplicative factors for adapting Δ.
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
    _hess_updater = resolve_hessian_update(hessian_update)

    x0 = coords.x0(molecule)
    n = len(x0)

    # Initial Hessian model
    if hessian_init == "exact" and hessian_provider is not None:
        B = hessian_provider(molecule)
    else:
        B = np.eye(n) * hessian_init_scale

    if progress:
        _print_header("Trust-region", provider, molecule, max_iter, _conv, progress)
        _print_initial_evaluation(progress)
    e_cur, g_cur = _f_fn(x0)
    g_cur = coords.apply_frozen(x0, g_cur, _frozen_set)
    n_eval = 1
    delta = initial_trust_radius

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

        # Solve trust-region subproblem
        p, _lam = _trust_region_step(g_cur, B, delta)
        p_norm = float(np.linalg.norm(p))
        if p_norm < 1e-15:
            converged = True
            break

        # Trial step
        x_cur = state.x
        x_new = x_cur + p
        e_new, g_new = _f_fn(x_new)
        g_new = coords.apply_frozen(x_new, g_new, _frozen_set)
        n_eval += 1

        # Powell's r test
        rho = _powell_rho(e_cur, e_new, g_cur, B, p)

        if rho < 0.0:
            # Uphill step -- reject, shrink trust radius.
            delta = max(delta * trust_shrink_factor, min_trust_radius)
            if progress:
                _print_step(
                    iteration,
                    e_cur,
                    g_cur,
                    step=p,
                    status="rejected",
                    progress=progress,
                )
            continue

        # Accept step
        s = x_new - x_cur
        y = g_new - g_cur
        B = _hess_updater(B, s, y)

        # Adapt trust radius
        if rho < rho_shrink:
            delta = max(delta * trust_shrink_factor, min_trust_radius)
        elif rho > rho_expand and p_norm > 0.8 * delta:
            delta = min(delta * trust_expand_factor, max_trust_radius)

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
        optimizer="trust",
        trajectory_frames=traj_frames if record_trajectory else [],
        trajectory_energies=traj_energies if record_trajectory else [],
        convergence_report=report,
        message="converged" if converged else "max_iter reached",
    )


# ---------------------------------------------------------------------------
# Rational function optimisation (RFO -- Banerjee et al. 1985)
# ---------------------------------------------------------------------------


@register("rfo")
def rfo(
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
    hessian_init: str = "diagonal",
    hessian_init_scale: float = 1.0,
    hessian_update: str = "bfgs",
    initial_trust_radius: float = 0.3,
    max_trust_radius: float = 1.0,
    min_trust_radius: float = 1e-4,
    rho_shrink: float = 0.25,
    rho_expand: float = 0.75,
    trust_shrink_factor: float = 0.5,
    trust_expand_factor: float = 2.0,
    line_search: str = "none",
    **kwargs: Any,
) -> GeomOptResult:
    r"""Rational function optimisation (Banerjee et al., 1985).

    RFO augments the quadratic model with a rational term controlled
    by a step-length parameter a:

        m(p) = 1/2 p^T H p + g^T p / (1 + a p^T S p)

    where S is a scaling matrix (identity by default).  The augmented
    Hessian eigenproblem

        ⎡ H   g ⎤ ⎡ p ⎤       ⎡ p ⎤
        ⎣ g^T 0 ⎦ ⎣ 1 ⎦ = l   ⎣ 1 ⎦

    gives the step direction from the lowest eigenvector.  For minimum
    search the (n+1)-dimensional eigenvector with the lowest eigenvalue
    is chosen; for TS search, the one whose n-dimensional part overlaps
    most with the mode-following vector.

    Parameters are identical to :func:`trust_region`.
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
    _hess_updater = resolve_hessian_update(hessian_update)

    x0 = coords.x0(molecule)
    n = len(x0)

    # Initial Hessian model
    if hessian_init == "exact" and hessian_provider is not None:
        B = hessian_provider(molecule)
    else:
        B = np.eye(n) * hessian_init_scale

    if progress:
        _print_header("RFO", provider, molecule, max_iter, _conv, progress)
        _print_initial_evaluation(progress)
    e_cur, g_cur = _f_fn(x0)
    g_cur = coords.apply_frozen(x0, g_cur, _frozen_set)
    n_eval = 1
    delta = initial_trust_radius

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

        # Build augmented Hessian: [B, g; g^T, 0]
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = B
        A[:n, n] = g_cur
        A[n, :n] = g_cur

        eigvals, eigvecs = np.linalg.eigh(A)

        # For a minimum search, take the eigenvector with the lowest
        # eigenvalue (most downhill).
        idx_min = int(np.argmin(eigvals))
        v = eigvecs[:, idx_min]
        p = v[:n] / v[n] if abs(v[n]) > 1e-15 else v[:n]

        # Trust-radius cap
        p_norm = float(np.linalg.norm(p))
        if p_norm > delta and p_norm > 1e-15:
            p = p * (delta / p_norm)

        if p_norm < 1e-15:
            converged = True
            break

        # Trial step
        x_cur = state.x
        x_new = x_cur + p
        e_new, g_new = _f_fn(x_new)
        g_new = coords.apply_frozen(x_new, g_new, _frozen_set)
        n_eval += 1

        # Powell's r test
        rho_actual = _powell_rho(e_cur, e_new, g_cur, B, p)

        if rho_actual < 0.0:
            delta = max(delta * trust_shrink_factor, min_trust_radius)
            if progress:
                _print_step(
                    iteration,
                    e_cur,
                    g_cur,
                    step=p,
                    status="rejected",
                    progress=progress,
                )
            continue

        # Accept
        s = x_new - x_cur
        y = g_new - g_cur
        B = _hess_updater(B, s, y)

        if rho_actual < rho_shrink:
            delta = max(delta * trust_shrink_factor, min_trust_radius)
        elif rho_actual > rho_expand and p_norm > 0.8 * delta:
            delta = min(delta * trust_expand_factor, max_trust_radius)

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
        optimizer="rfo",
        trajectory_frames=traj_frames if record_trajectory else [],
        trajectory_energies=traj_energies if record_trajectory else [],
        convergence_report=report,
        message="converged" if converged else "max_iter reached",
    )


# ``_print_step`` is imported from ``.optimizers`` (see the import block at
# the top of this module). An earlier revision carried a byte-identical
# local copy that shadowed that import; it was deleted so the row format
# lives in exactly one place and the two cannot drift (the DIIS-width class
# of drift the output audit flagged). Verified byte-identical before removal.
