"""Geometry optimiser implementations -- uniform interface.

Each optimiser is a function that accepts standardised inputs and
returns a :class:`~vibeqc.geomopt.state.GeomOptResult`.  The factory
:func:`resolve_optimizer` maps the ``geom_opt`` keyword to the right
function.

Phase 1 implementers (Cartesian minimum-search):
* :func:`steepest_descent` -- -gradE direction + line search
* :func:`conjugate_gradient` -- Polak-Ribière nonlinear CG
* :func:`bfgs` -- Quasi-Newton with BFGS Hessian update
* :func:`lbfgs` -- Limited-memory BFGS (two-loop recursion)

Phase 3: trust-region and RFO in :mod:`vibeqc.geomopt.trust`.
Phase 5: EF, P-RFO, dimer.
Phase 6: FIRE, GDIIS.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

from .._vibeqc_core import Molecule
from ..output import Quantity, active_channel, active_policy, flush, write
from ..progress import resolve_progress
from .convergence import ConvergencePolicy
from .coordinates import CartesianCoordinates, CoordinateRepresentation
from .line_search import LineSearchFn, resolve_line_search
from .providers import EnergyGradientProvider, HessianProvider, _positions_native
from .registry import register

# Re-export the registry resolve function
from .registry import resolve as resolve_optimizer  # noqa: F401
from .state import GeomOptResult, OptimizerState

# C++ acceleration kernels
_cpp_kernels = None


def _get_cpp():
    """Return C++ kernel object or None if unavailable."""
    global _cpp_kernels
    if _cpp_kernels is None:
        try:
            from .._vibeqc_core import bfgs_inverse_update as _bi
            from .._vibeqc_core import lbfgs_two_loop as _lb

            _cpp_kernels = type("Cpp", (), {"lbfgs": _lb, "bfgs_inv": _bi})()
        except ImportError:
            _cpp_kernels = False
    return _cpp_kernels if _cpp_kernels is not False else None


def _n_atoms(obj: Any) -> int:
    """Return number of atoms, supporting both Molecule and PeriodicSystem."""
    if hasattr(obj, "atoms"):
        return len(list(obj.atoms))
    if hasattr(obj, "unit_cell"):
        return len(list(obj.unit_cell))
    raise TypeError(f"Cannot determine n_atoms from {type(obj).__name__}")


# ---------------------------------------------------------------------------
# Shared helpers (also used by trust.py, ts.py)
# ---------------------------------------------------------------------------


def _make_x_fn(
    coords: CoordinateRepresentation,
    template: Molecule,
) -> Callable[[np.ndarray], Molecule]:
    """Return a closure ``x -> Molecule``."""

    def _to_mol(x: np.ndarray) -> Molecule:
        return coords.to_cartesian(template, x)

    return _to_mol


def _make_objective(
    provider: EnergyGradientProvider,
    coords: CoordinateRepresentation,
    template: Molecule,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """Return a closure ``x -> (energy, gradient)`` in optimiser space."""

    def _eval(x: np.ndarray) -> tuple[float, np.ndarray]:
        mol = coords.to_cartesian(template, x)
        e, g_cart = provider(mol)
        g = coords.project_gradient(mol, g_cart)
        return e, g

    return _eval


# ---------------------------------------------------------------------------
# Progress helpers (also used by trust.py)
# ---------------------------------------------------------------------------


def _print_header(
    name: str,
    provider: EnergyGradientProvider,
    molecule: Molecule,
    max_iter: int,
    conv: ConvergencePolicy,
    progress: Any = True,
) -> None:
    policy = active_policy()
    n_atoms = _n_atoms(molecule)
    parts = [f"\n  Geometry optimization ({name})"]
    if hasattr(provider, "_method"):
        parts[-1] += f" -- {provider._method.upper()}"
    parts.append(f"  n_atoms={n_atoms}, max_iter={max_iter}")
    if conv.gmax:
        threshold_policy = policy.with_spec(
            "gradient", width=0, precision=1, notation="e", sign=False
        )
        threshold = threshold_policy.render(Quantity(conv.gmax, "gradient"))
        parts.append(f"  gmax={threshold} {policy.unit_of('gradient')}")
    energy_header = f"E ({policy.unit_of('energy')})"
    _progress_print(progress, "  ".join(parts) + "\n")
    _progress_print(
        progress,
        f"  {'step':>4s}  {energy_header:>16s}  {'dE':>11s}  "
        f"{'max|g|':>10s}  {'|step|':>10s}  conv",
    )


def _progress_print(progress: Any, text: str = "") -> None:
    """Emit one optimizer line to the owned text surfaces.

    The ambient channel owns persistent ``.out`` text. ``ProgressLogger``
    independently owns optional live terminal chatter; a caller never hands
    this module a file object.
    """
    payload = text + "\n"
    if active_channel() is not None:
        write(payload)
        flush()
    resolve_progress(progress).write_raw(payload)


def _print_initial_evaluation(progress: Any) -> None:
    _progress_print(progress, "  evaluating initial energy and gradient ...")


def _print_step(
    iteration: int,
    energy: float,
    gradient: np.ndarray,
    *,
    prev_energy: Optional[float] = None,
    step: Optional[np.ndarray] = None,
    report: Any = None,
    status: str = "",
    progress: Any = True,
) -> None:
    policy = active_policy()
    gmax = float(np.max(np.abs(gradient))) if gradient.size else float("inf")
    energy_text = policy.with_spec(
        "energy", width=16, precision=8, notation="f", sign=False
    ).render(Quantity(energy, "energy"))
    de = (
        "     --    "
        if prev_energy is None
        else policy.render(Quantity(energy - prev_energy, "energy_delta"))
    )
    gmax_text = policy.with_spec(
        "gradient", width=10, precision=4, notation="e", sign=False
    ).render(Quantity(gmax, "gradient"))
    step_norm = (
        "     --   "
        if step is None
        else policy.with_spec(
            "length", width=10, precision=3, notation="e", sign=False
        ).render(
            Quantity(
                float(np.linalg.norm(np.asarray(step, dtype=float))),
                "length",
            )
        )
    )
    flags = report.summary() if hasattr(report, "summary") else ""
    if status:
        flags = f"{status}" if not flags else f"{status}; {flags}"
    _progress_print(
        progress,
        f"  {iteration:4d}  {energy_text}  {de}  "
        f"{gmax_text}  {step_norm}  {flags}",
    )


# ---------------------------------------------------------------------------
# Steepest descent
# ---------------------------------------------------------------------------


@register("sd")
def steepest_descent(
    molecule: Molecule,
    provider: EnergyGradientProvider,
    *,
    coords: Optional[CoordinateRepresentation] = None,
    max_iter: int = 100,
    conv: Optional[ConvergencePolicy] = None,
    line_search: str = "brent",
    line_search_step: float = 0.05,
    record_trajectory: bool = True,
    progress: bool = False,
    hessian_provider: Optional[HessianProvider] = None,
    freeze_indices: Optional[Sequence[int]] = None,
    x0: Optional[np.ndarray] = None,
    iteration_start: int = 0,
    extra: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> GeomOptResult:
    r"""Steepest-descent geometry optimisation.

    At each step the search direction is :math:`p_k = -\nabla E(x_k)`.
    A line search along *p_k* finds the optimal step length.
    """
    n_atoms = _n_atoms(molecule)
    _frozen_set: set[int] = (
        {int(i) for i in freeze_indices} if freeze_indices else set()
    )

    if coords is None:
        coords = CartesianCoordinates(n_atoms, freeze_indices=freeze_indices)

    _f_fn = _make_objective(provider, coords, molecule)
    _to_mol = _make_x_fn(coords, molecule)
    _ls = resolve_line_search(line_search, step=line_search_step)
    _conv = conv or ConvergencePolicy.default()

    if x0 is not None:
        x0 = np.asarray(x0, dtype=float).ravel()
    else:
        x0 = coords.x0(molecule)
    if progress:
        _print_header("Steepest descent", provider, molecule, max_iter, _conv, progress)
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
            iteration_start,
            e_cur,
            g_cur,
            report=_conv.check(gradient=g_cur),
            progress=progress,
        )

    state = OptimizerState(
        x=x0,
        energy=e_cur,
        gradient=g_cur,
        n_energy_evals=n_eval,
        iteration=iteration_start,
    )
    converged = False

    for iteration in range(iteration_start + 1, iteration_start + max_iter + 1):
        report = _conv.check(
            gradient=g_cur,
            x_old=state.x_prev,
            x_new=x0 if iteration == 1 else x_cur,
            e_old=state.energy_prev,
            e_new=e_cur,
        )
        if report:
            converged = True
            break

        p = -g_cur
        p_norm = float(np.linalg.norm(p))
        if p_norm < 1e-15:
            converged = True
            break
        p = p / p_norm

        x_cur = state.x
        x_new, e_new, n_ls = _ls(x_cur, e_cur, p, _f_fn, _to_mol)
        n_eval += n_ls

        _, g_new = _f_fn(x_new)
        g_new = coords.apply_frozen(x_new, g_new, _frozen_set)

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
        optimizer="sd",
        trajectory_frames=traj_frames if record_trajectory else [],
        trajectory_energies=traj_energies if record_trajectory else [],
        convergence_report=report,
        message="converged" if converged else "max_iter reached",
    )


# ---------------------------------------------------------------------------
# Nonlinear conjugate gradient (Polak-Ribière)
# ---------------------------------------------------------------------------


@register("cg")
def conjugate_gradient(
    molecule: Molecule,
    provider: EnergyGradientProvider,
    *,
    coords: Optional[CoordinateRepresentation] = None,
    max_iter: int = 100,
    conv: Optional[ConvergencePolicy] = None,
    line_search: str = "brent",
    line_search_step: float = 0.05,
    record_trajectory: bool = True,
    progress: bool = False,
    hessian_provider: Optional[HessianProvider] = None,
    freeze_indices: Optional[Sequence[int]] = None,
    x0: Optional[np.ndarray] = None,
    iteration_start: int = 0,
    extra: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> GeomOptResult:
    r"""Nonlinear conjugate gradient (Polak-Ribière)."""
    n_atoms = _n_atoms(molecule)
    _frozen_set: set[int] = (
        {int(i) for i in freeze_indices} if freeze_indices else set()
    )

    if coords is None:
        coords = CartesianCoordinates(n_atoms, freeze_indices=freeze_indices)

    _f_fn = _make_objective(provider, coords, molecule)
    _to_mol = _make_x_fn(coords, molecule)
    _ls = resolve_line_search(line_search, step=line_search_step)
    _conv = conv or ConvergencePolicy.default()

    if x0 is not None:
        x0 = np.asarray(x0, dtype=float).ravel()
    else:
        x0 = coords.x0(molecule)
    if progress:
        _print_header("Conjugate gradient", provider, molecule, max_iter, _conv, progress)
        _print_initial_evaluation(progress)
    e_cur, g_cur = _f_fn(x0)
    g_cur = coords.apply_frozen(x0, g_cur, _frozen_set)
    n_eval = 1
    n_params = len(x0)

    traj_frames: list[Molecule] = []
    traj_energies: list[float] = []
    if record_trajectory:
        traj_frames.append(molecule)
        traj_energies.append(e_cur)

    if progress:
        _print_step(
            iteration_start,
            e_cur,
            g_cur,
            report=_conv.check(gradient=g_cur),
            progress=progress,
        )

    p_prev: Optional[np.ndarray] = None
    g_prev: Optional[np.ndarray] = None
    state = OptimizerState(
        x=x0,
        energy=e_cur,
        gradient=g_cur,
        n_energy_evals=n_eval,
        iteration=iteration_start,
    )
    converged = False

    for iteration in range(iteration_start + 1, iteration_start + max_iter + 1):
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

        if p_prev is None or g_prev is None or iteration % n_params == 0:
            beta = 0.0
        else:
            beta = float(np.dot(g_cur, g_cur - g_prev) / np.dot(g_prev, g_prev))
            if beta < 0.0:
                beta = 0.0

        p = -g_cur + beta * (p_prev if p_prev is not None else np.zeros_like(g_cur))
        p_norm = float(np.linalg.norm(p))
        if p_norm < 1e-15:
            converged = True
            break
        p = p / p_norm

        x_cur = state.x
        x_new, e_new, n_ls = _ls(x_cur, e_cur, p, _f_fn, _to_mol)
        n_eval += n_ls

        _, g_new = _f_fn(x_new)
        g_new = coords.apply_frozen(x_new, g_new, _frozen_set)

        g_prev = g_cur
        p_prev = p
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
        optimizer="cg",
        trajectory_frames=traj_frames if record_trajectory else [],
        trajectory_energies=traj_energies if record_trajectory else [],
        convergence_report=report,
        message="converged" if converged else "max_iter reached",
    )


# ---------------------------------------------------------------------------
# BFGS -- full quasi-Newton
# ---------------------------------------------------------------------------


@register("bfgs")
def bfgs(
    molecule: Molecule,
    provider: EnergyGradientProvider,
    *,
    coords: Optional[CoordinateRepresentation] = None,
    max_iter: int = 100,
    conv: Optional[ConvergencePolicy] = None,
    line_search: str = "backtracking",
    line_search_step: float = 0.05,
    record_trajectory: bool = True,
    progress: bool = False,
    hessian_provider: Optional[HessianProvider] = None,
    freeze_indices: Optional[Sequence[int]] = None,
    hessian_init: str = "diagonal",
    hessian_init_scale: float = 1.0,
    x0: Optional[np.ndarray] = None,
    iteration_start: int = 0,
    extra: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> GeomOptResult:
    r"""BFGS quasi-Newton geometry optimisation."""
    n_atoms = _n_atoms(molecule)
    _frozen_set: set[int] = (
        {int(i) for i in freeze_indices} if freeze_indices else set()
    )

    if coords is None:
        coords = CartesianCoordinates(n_atoms, freeze_indices=freeze_indices)

    _f_fn = _make_objective(provider, coords, molecule)
    _to_mol = _make_x_fn(coords, molecule)
    _ls = resolve_line_search(line_search, step=line_search_step)
    _conv = conv or ConvergencePolicy.default()

    if x0 is not None:
        x0 = np.asarray(x0, dtype=float).ravel()
    else:
        x0 = coords.x0(molecule)
    n = len(x0)

    if extra is not None and "h_inv" in extra:
        H_inv = np.asarray(extra["h_inv"], dtype=float)
    elif hessian_init == "exact" and hessian_provider is not None:
        H0 = hessian_provider(molecule)
        H_inv = np.linalg.inv(H0)
    else:
        H_inv = np.eye(n) / hessian_init_scale

    if progress:
        _print_header("BFGS", provider, molecule, max_iter, _conv, progress)
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
            iteration_start,
            e_cur,
            g_cur,
            report=_conv.check(gradient=g_cur),
            progress=progress,
        )

    state = OptimizerState(
        x=x0,
        energy=e_cur,
        gradient=g_cur,
        n_energy_evals=n_eval,
        iteration=iteration_start,
    )
    converged = False

    for iteration in range(iteration_start + 1, iteration_start + max_iter + 1):
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

        p = -H_inv @ g_cur
        p_norm = float(np.linalg.norm(p))
        if p_norm < 1e-15:
            converged = True
            break

        x_cur = state.x
        x_new, e_new, n_ls = _ls(x_cur, e_cur, p, _f_fn, _to_mol)
        n_eval += n_ls

        _, g_new = _f_fn(x_new)
        g_new = coords.apply_frozen(x_new, g_new, _frozen_set)

        # BFGS inverse-Hessian update -- C++ accelerated when available
        s = x_new - x_cur
        y = g_new - g_cur
        sy = float(np.dot(s, y))
        if sy > 1e-12:
            cpp = _get_cpp()
            if cpp is not None:
                H_inv = np.asarray(cpp.bfgs_inv(H_inv, s, y), dtype=float)
            else:
                rho = 1.0 / sy
                I = np.eye(n)
                left = I - rho * np.outer(s, y)
                right = I - rho * np.outer(y, s)
                H_inv = left @ H_inv @ right + rho * np.outer(s, s)

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
        optimizer="bfgs",
        trajectory_frames=traj_frames if record_trajectory else [],
        trajectory_energies=traj_energies if record_trajectory else [],
        convergence_report=report,
        message="converged" if converged else "max_iter reached",
    )


# ---------------------------------------------------------------------------
# L-BFGS -- Limited-memory BFGS (two-loop recursion)
# ---------------------------------------------------------------------------


@register("lbfgs")
def lbfgs(
    molecule: Molecule,
    provider: EnergyGradientProvider,
    *,
    coords: Optional[CoordinateRepresentation] = None,
    max_iter: int = 100,
    conv: Optional[ConvergencePolicy] = None,
    line_search: str = "backtracking",
    line_search_step: float = 0.05,
    record_trajectory: bool = True,
    progress: bool = False,
    hessian_provider: Optional[HessianProvider] = None,
    freeze_indices: Optional[Sequence[int]] = None,
    memory: int = 10,
    hessian_init_scale: float = 1.0,
    x0: Optional[np.ndarray] = None,
    iteration_start: int = 0,
    extra: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> GeomOptResult:
    r"""Limited-memory BFGS (L-BFGS) using the two-loop recursion."""
    n_atoms = _n_atoms(molecule)
    _frozen_set: set[int] = (
        {int(i) for i in freeze_indices} if freeze_indices else set()
    )

    if coords is None:
        coords = CartesianCoordinates(n_atoms, freeze_indices=freeze_indices)

    _f_fn = _make_objective(provider, coords, molecule)
    _to_mol = _make_x_fn(coords, molecule)
    _ls = resolve_line_search(line_search, step=line_search_step)
    _conv = conv or ConvergencePolicy.default()

    if x0 is not None:
        x0 = np.asarray(x0, dtype=float).ravel()
    else:
        x0 = coords.x0(molecule)
    if progress:
        _print_header("L-BFGS", provider, molecule, max_iter, _conv, progress)
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
            iteration_start,
            e_cur,
            g_cur,
            report=_conv.check(gradient=g_cur),
            progress=progress,
        )

    # Restore L-BFGS history from restart extra if provided
    s_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    rho_list: list[float] = []
    if extra is not None:
        s_raw = extra.get("s_list", [])
        y_raw = extra.get("y_list", [])
        if s_raw and y_raw:
            s_list = [np.asarray(s, dtype=float) for s in s_raw]
            y_list = [np.asarray(y, dtype=float) for y in y_raw]
            rho_list = [
                float(1.0 / np.dot(s_list[i], y_list[i])) for i in range(len(s_list))
            ]
            # Trim to memory
            if len(s_list) > memory:
                s_list = s_list[:memory]
                y_list = y_list[:memory]
                rho_list = rho_list[:memory]

    state = OptimizerState(
        x=x0,
        energy=e_cur,
        gradient=g_cur,
        n_energy_evals=n_eval,
        iteration=iteration_start,
    )
    converged = False

    for iteration in range(iteration_start + 1, iteration_start + max_iter + 1):
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

        # Two-loop recursion -- C++ accelerated when available
        cpp = _get_cpp()
        if cpp is not None and s_list:
            p, ok = cpp.lbfgs(
                s_list, y_list, g_cur, hessian_init_scale if not s_list else 1.0
            )
            if ok:
                p = np.asarray(p, dtype=float).ravel()
            else:
                p = -g_cur  # fallback
        else:
            q = g_cur.copy()
            alpha: list[float] = []
            for s_i, y_i, rho_i in zip(s_list, y_list, rho_list):
                a = rho_i * float(np.dot(s_i, q))
                alpha.append(a)
                q = q - a * y_i
            if s_list:
                gamma = float(
                    np.dot(s_list[0], y_list[0]) / np.dot(y_list[0], y_list[0])
                )
            else:
                gamma = 1.0 / hessian_init_scale
            r = gamma * q
            for s_i, y_i, rho_i, a in zip(
                reversed(s_list), reversed(y_list), reversed(rho_list), reversed(alpha)
            ):
                beta = rho_i * float(np.dot(y_i, r))
                r = r + s_i * (a - beta)
            p = -r
        p_norm = float(np.linalg.norm(p))
        if p_norm < 1e-15:
            converged = True
            break

        x_cur = state.x
        x_new, e_new, n_ls = _ls(x_cur, e_cur, p, _f_fn, _to_mol)
        n_eval += n_ls

        _, g_new = _f_fn(x_new)
        g_new = coords.apply_frozen(x_new, g_new, _frozen_set)

        s = x_new - x_cur
        y = g_new - g_cur
        sy = float(np.dot(s, y))
        if sy > 1e-12:
            s_list.insert(0, s.copy())
            y_list.insert(0, y.copy())
            rho_list.insert(0, 1.0 / sy)
            if len(s_list) > memory:
                s_list.pop()
                y_list.pop()
                rho_list.pop()

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
        optimizer="lbfgs",
        trajectory_frames=traj_frames if record_trajectory else [],
        trajectory_energies=traj_energies if record_trajectory else [],
        convergence_report=report,
        message="converged" if converged else "max_iter reached",
    )
