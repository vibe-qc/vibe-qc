"""Transition-state optimisers (Phase 5).

Eigenvector-following (EF) and partitioned rational function
optimisation (P-RFO) for locating first-order saddle points.

References
----------
* Baker, J. Comput. Chem. 7, 385 (1986) -- Eigenvector-following.
* Banerjee, Adams, Simons & Shepard, J. Phys. Chem. 89, 52 (1985)
  -- RFO foundation.
* Baker, J. Comput. Chem. 8, 563 (1987) -- P-RFO with mode-following.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .._vibeqc_core import Molecule
from ..output import Quantity, active_policy
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
    _progress_print,
    _print_step,
)
from .providers import EnergyGradientProvider, HessianProvider
from .registry import register

# ---------------------------------------------------------------------------
# Eigenvector-following (Baker 1986)
# ---------------------------------------------------------------------------


@register("ef")
def eigenvector_following(
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
    hessian_update: str = "bofill",
    initial_trust_radius: float = 0.3,
    max_trust_radius: float = 1.0,
    min_trust_radius: float = 1e-4,
    rho_shrink: float = 0.25,
    rho_expand: float = 0.75,
    trust_shrink_factor: float = 0.5,
    trust_expand_factor: float = 2.0,
    mode_follow: int = 0,
    target: str = "minimum",
    line_search: str = "none",
    _progress_label: Optional[str] = None,
    _result_optimizer: str = "ef",
    **kwargs: Any,
) -> GeomOptResult:
    r"""Eigenvector-following (EF) for minima and transition states.

    Uses the augmented Hessian approach: at each step, diagonalises
    the Hessian model B and follows the eigenvector corresponding to
    the mode specified by ``mode_follow``.

    For minimum search (``target="minimum"``): follows the lowest
    eigenvector (mode_follow=0 by default), equivalent to RFO.

    For TS search (``target="transition_state"``): follows the
    eigenvector with the lowest eigenvalue (mode_follow=0). The
    Hessian model B is updated via ``hessian_update`` (default
    "bofill", which handles indefinite Hessians for TS searches).

    Parameters
    ----------
    mode_follow : int
        Which eigenmode to follow (0 = lowest eigenvalue).
    target : str
        ``"minimum"`` or ``"transition_state"``.
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

    if progress:
        label = _progress_label or (
            "EF (TS)" if target == "transition_state" else "EF (min)"
        )
        _print_header(label, provider, molecule, max_iter, _conv, progress)
        _print_initial_evaluation(progress)

    initial_eval: Optional[tuple[float, np.ndarray]] = None

    # Initial Hessian model
    if hessian_init == "exact" and hessian_provider is not None:
        B = hessian_provider(molecule)
    else:
        B = np.eye(n) * hessian_init_scale
        if target == "transition_state":
            # Seed a negative eigenvalue for TS search -- Baker's approach:
            # find the eigenvector of the initial gradient and set its
            # eigenvalue to -hessian_init_scale.
            initial_eval = _f_fn(x0)
            _, g0 = initial_eval
            g0 = coords.apply_frozen(x0, g0, _frozen_set)
            g0_norm = float(np.linalg.norm(g0))
            if g0_norm > 1e-12:
                v = g0 / g0_norm
                B = B - 2.0 * hessian_init_scale * np.outer(v, v)

    if initial_eval is None:
        e_cur, g_cur = _f_fn(x0)
    else:
        e_cur, g_cur = initial_eval
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
    is_saddle: Optional[bool] = None
    curvature: Optional[float] = None
    mode_vec: Optional[np.ndarray] = None

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

        # Diagonalise B
        eigvals, eigvecs = np.linalg.eigh(B)

        # Select the mode to follow
        idx = mode_follow
        if target == "transition_state":
            # Follow the lowest eigenvalue (most negative for TS)
            idx = int(np.argmin(eigvals))

        # Augmented Hessian eigenproblem for the selected mode:
        # [ l_i   g.v_i ] [a]       [a]
        # [ g.v_i   0   ] [b] = mu   [b]
        v_i = eigvecs[:, idx]
        lambda_i = eigvals[idx]
        gv = float(g_cur @ v_i)

        A = np.array([[lambda_i, gv], [gv, 0.0]])
        mu_vals, mu_vecs = np.linalg.eigh(A)
        mu = mu_vals[0]
        alpha = mu_vecs[0, 0]
        beta = mu_vecs[1, 0]

        # RFO step: p = -(alpha/beta) v_i  if beta != 0
        if abs(beta) > 1e-15:
            p = -(alpha / beta) * v_i
        else:
            p = -v_i

        # Add orthogonal components (standard NR step in the orthogonal complement)
        g_orth = g_cur - gv * v_i
        if float(np.linalg.norm(g_orth)) > 1e-15:
            for j in range(n):
                if j == idx:
                    continue
                v_j = eigvecs[:, j]
                lambda_j = eigvals[j]
                g_j = float(g_cur @ v_j)
                if abs(lambda_j) > 1e-12:
                    p = p - (g_j / lambda_j) * v_j

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

        rho_actual = _powell_rho_local(e_cur, e_new, g_cur, B, p)

        if rho_actual < 0.0:
            delta = max(delta * trust_shrink_factor, min_trust_radius)
            if progress:
                _print_local(iteration, e_cur, g_cur, "rejected", progress=progress)
            continue

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

    # Characterise stationary point
    if converged:
        eigvals_final, _ = np.linalg.eigh(B)
        n_neg = int(np.sum(eigvals_final < -1e-6))
        is_saddle = n_neg == 1
        curvature = float(eigvals_final[0])

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
        optimizer=_result_optimizer,
        trajectory_frames=traj_frames if record_trajectory else [],
        trajectory_energies=traj_energies if record_trajectory else [],
        is_saddle=is_saddle,
        curvature=curvature,
        mode=mode_vec,
        convergence_report=report,
        message="converged" if converged else "max_iter reached",
    )


# ---------------------------------------------------------------------------
# P-RFO (Partitioned RFO -- Baker 1987)
# ---------------------------------------------------------------------------


@register("prfo")
def prfo(
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
    hessian_update: str = "bofill",
    initial_trust_radius: float = 0.3,
    max_trust_radius: float = 1.0,
    min_trust_radius: float = 1e-4,
    rho_shrink: float = 0.25,
    rho_expand: float = 0.75,
    trust_shrink_factor: float = 0.5,
    trust_expand_factor: float = 2.0,
    target: str = "transition_state",
    line_search: str = "none",
    **kwargs: Any,
) -> GeomOptResult:
    r"""Partitioned RFO (Baker 1987) for transition-state searches.

    Splits the step into two components:
    1. Along the reaction mode (lowest eigenvector): maximisation via
       the RFO augmented Hessian projected onto that mode.
    2. Orthogonal complement: minimisation via standard NR steps.

    This is the same algorithm Q-Chem uses for TS searches.
    """
    return eigenvector_following(
        molecule,
        provider,
        coords=coords,
        max_iter=max_iter,
        conv=conv,
        record_trajectory=record_trajectory,
        progress=progress,
        hessian_provider=hessian_provider,
        freeze_indices=freeze_indices,
        hessian_init=hessian_init,
        hessian_init_scale=hessian_init_scale,
        hessian_update=hessian_update,
        initial_trust_radius=initial_trust_radius,
        max_trust_radius=max_trust_radius,
        min_trust_radius=min_trust_radius,
        rho_shrink=rho_shrink,
        rho_expand=rho_expand,
        trust_shrink_factor=trust_shrink_factor,
        trust_expand_factor=trust_expand_factor,
        mode_follow=0,
        target=target,
        line_search=line_search,
        _progress_label=(
            "P-RFO (TS)" if target == "transition_state" else "P-RFO (min)"
        ),
        _result_optimizer="prfo",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Local copy of Powell's r (avoids circular import from trust.py)
# ---------------------------------------------------------------------------


def _powell_rho_local(
    f_old: float,
    f_new: float,
    g: np.ndarray,
    B: np.ndarray,
    p: np.ndarray,
) -> float:
    actual = f_old - f_new
    predicted = -(float(g @ p) + 0.5 * float(p @ B @ p))
    if abs(predicted) < 1e-15:
        return 0.0
    return actual / predicted


def _print_local(
    iteration: int,
    energy: float,
    gradient: np.ndarray,
    suffix: str = "",
    *,
    progress: Any = True,
) -> None:
    policy = active_policy()
    gmax = float(np.max(np.abs(gradient))) if gradient.size else float("inf")
    energy_text = policy.with_spec(
        "energy", width=16, precision=8, notation="f", sign=False
    ).render(Quantity(energy, "energy"))
    gmax_text = policy.with_spec(
        "gradient", width=10, precision=4, notation="e", sign=False
    ).render(Quantity(gmax, "gradient"))
    line = f"  {iteration:4d}  {energy_text}  {gmax_text}"
    if suffix:
        line += f"  [{suffix}]"
    _progress_print(progress, line)
