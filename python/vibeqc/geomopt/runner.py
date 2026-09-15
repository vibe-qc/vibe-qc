"""Single entry point for geometry optimisation -- :func:`run_geomopt`.

This is the keyword-driven facade that picks the optimiser, builds
the coordinate representation, resolves the line search, and runs to
completion.  Every theory backend calls this same function.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .._vibeqc_core import Molecule, PeriodicSystem
from ..progress import resolve_progress
from .convergence import ConvergencePolicy
from .coordinates import CoordinateRepresentation
from .history import RestartSerializer, build_restart_state, restore_optimizer_state
from .providers import EnergyGradientProvider, HessianProvider
from .registry import resolve as resolve_optimizer
from .state import GeomOptResult, OptimizerState

# ---------------------------------------------------------------------------
# Keyword mapping: user-facing geom_opt -> internal optimiser name
# ---------------------------------------------------------------------------

_KEYWORD_MAP: dict[str, str] = {
    "sd": "sd",
    "steepest_descent": "sd",
    "steepest-descent": "sd",
    "cg": "cg",
    "conjugate_gradient": "cg",
    "conjugate-gradient": "cg",
    "bfgs": "bfgs",
    "lbfgs": "lbfgs",
    "l-bfgs": "lbfgs",
    "trust": "trust",  # Phase 3
    "trust_region": "trust",
    "trust-region": "trust",
    "rfo": "rfo",  # Phase 3
    "ef": "ef",  # Phase 5
    "eigenvector_following": "ef",
    "eigenvector-following": "ef",
    "prfo": "prfo",  # Phase 5
    "dimer": "dimer",  # Phase 5
    "fire": "fire",  # Phase 6
    "gdiis": "gdiis",  # Phase 6
}


def _resolve_geom_opt(requested: str) -> str:
    """Normalise the user-facing keyword to an internal optimiser name."""
    key = requested.lower().replace("_", "-")
    try:
        return _KEYWORD_MAP[key]
    except KeyError:
        available = sorted(set(_KEYWORD_MAP.values()))
        raise ValueError(
            f"Unknown geom_opt={requested!r}.  Available: {available}"
        ) from None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_geomopt(
    molecule: Molecule,
    provider: EnergyGradientProvider,
    *,
    geom_opt: str = "bfgs",
    geom_coords: str = "cartesian",
    geom_target: str = "minimum",
    geom_hessian_init: str = "diagonal",
    geom_hessian_update: str = "none",
    geom_line_search: str = "backtracking",
    geom_conv: Optional[ConvergencePolicy] = None,
    geom_max_iter: int = 100,
    geom_freeze: Optional[Sequence[int]] = None,
    geom_coords_options: Optional[dict[str, Any]] = None,
    geom_opt_options: Optional[dict[str, Any]] = None,
    hessian_provider: Optional[HessianProvider] = None,
    record_trajectory: bool = True,
    progress: Any = False,
    # Restart / checkpoint
    geom_restart: Optional[str] = None,
    geom_checkpoint: Optional[str] = None,
    # Backward-compat shim
    optimizer_backend: Optional[str] = None,
    **extra_kwargs: Any,
) -> GeomOptResult:
    """Run a geometry optimisation with keyword-selectable method.

    Parameters
    ----------
    molecule : Molecule
        Starting geometry (Cartesian coordinates in bohr).
    provider : EnergyGradientProvider
        Supplies (energy, gradient) at each geometry.  Build via
        :class:`~vibeqc.geomopt.providers.MolecularSCFProvider` or
        a custom callable.
    geom_opt : str
        Optimiser to use: ``"sd"``, ``"cg"``, ``"bfgs"``, ``"lbfgs"``.
        Phase 3+: ``"trust"``, ``"rfo"``.
        Phase 5+: ``"ef"``, ``"prfo"``, ``"dimer"``.
    geom_coords : str
        Coordinate system: ``"cartesian"`` only in Phase 1.
        Phase 4+: ``"internal"``, ``"dlc"``.
    geom_target : str
        ``"minimum"`` or ``"transition_state"``.
    geom_hessian_init : str
        How to initialise the Hessian: ``"diagonal"``, ``"exact"``,
        ``"read"``, ``"model"``.
    geom_hessian_update : str
        Hessian update formula: ``"none"``, ``"bfgs"``, ``"powell"``,
        ``"bofill"``, ``"murtagh_sargent"`` (Phase 6).
    geom_line_search : str
        Line search strategy: ``"brent"``, ``"backtracking"``,
        ``"none"``.
    geom_conv : ConvergencePolicy or None
        Convergence criteria.  ``None`` = default (gmax=4.5e-4).
    geom_max_iter : int
        Maximum optimisation steps (default 100).
    geom_freeze : sequence of int or None
        Atom indices to hold fixed.
    geom_coords_options : dict or None
        Extra options forwarded to the coordinate representation.
    geom_opt_options : dict or None
        Extra options forwarded to the optimiser.
    hessian_provider : HessianProvider or None
        For optimisers that need an explicit Hessian.
    record_trajectory : bool
        Collect per-step geometries for QVF output.
    progress : bool or ProgressLogger
        ``True`` streams per-step energy and gradient to stdout; a
        :class:`~vibeqc.progress.ProgressLogger` gives the caller control of
        the live sink. Persistent ``.out`` text uses the ambient output
        channel independently.
    optimizer_backend : str or None
        Backward-compat: ``"brent"`` -> ``geom_opt="sd"``,
        ``"native"`` -> ``geom_opt="lbfgs"``.

    Returns
    -------
    GeomOptResult
    """
    progress = resolve_progress(progress)

    # ---- periodic auto-detection ---------------------------------------------
    if isinstance(molecule, PeriodicSystem):
        from .periodic_providers import run_periodic_geomopt

        return run_periodic_geomopt(
            molecule,
            provider,
            geom_opt=geom_opt,
            geom_line_search=geom_line_search,
            geom_max_iter_atoms=geom_max_iter,
            record_trajectory=record_trajectory,
            progress=progress,
            relax_cell=True,
        )

    # ---- backward-compat shim -----------------------------------------------
    if optimizer_backend is not None and geom_opt == "bfgs":
        if optimizer_backend == "brent":
            geom_opt = "sd"
        elif optimizer_backend == "native":
            geom_opt = "lbfgs"
        # "ase" -> caller should still use the ASE path; don't override

    # ---- validate -----------------------------------------------------------
    if geom_target not in ("minimum", "transition_state"):
        raise ValueError(
            f"geom_target={geom_target!r} -- expected 'minimum' or 'transition_state'."
        )
    opt_name = _resolve_geom_opt(geom_opt)
    if geom_target == "transition_state":
        if opt_name not in ("ef", "prfo", "trust", "dimer"):
            opt_name = "ef"  # default TS optimiser
    else:
        if opt_name == "dimer":
            raise ValueError(
                "geom_opt='dimer' is a transition-state/saddle-search method; "
                "set geom_target='transition_state'."
            )

    opt_fn = resolve_optimizer(opt_name)

    # ---- build coordinate representation ------------------------------------
    if geom_coords == "cartesian":
        from .coordinates import CartesianCoordinates

        n_atoms = len(list(molecule.atoms))
        coords = CartesianCoordinates(n_atoms, freeze_indices=geom_freeze)
    elif geom_coords in ("dlc", "internal", "delocalized"):
        from .internal_coords import DelocalizedInternalCoordinates

        n_atoms = len(list(molecule.atoms))
        atomic_numbers = [int(a.Z) for a in molecule.atoms]
        positions = np.array([list(a.xyz) for a in molecule.atoms], dtype=float)
        coords = DelocalizedInternalCoordinates(
            n_atoms, atomic_numbers, positions, freeze_indices=geom_freeze
        )
    else:
        raise NotImplementedError(
            f"geom_coords={geom_coords!r} -- supported: 'cartesian', 'dlc'."
        )
    if geom_coords_options:
        for k, v in geom_coords_options.items():
            if hasattr(coords, k):
                setattr(coords, k, v)

    # ---- build convergence policy -------------------------------------------
    conv = geom_conv or ConvergencePolicy.default()

    # ---- restart from checkpoint ------------------------------------------
    restart_x0: Optional[np.ndarray] = None
    restart_iteration: int = 0
    restart_extra: dict[str, Any] = {}
    if geom_restart is not None:
        serializer = RestartSerializer(geom_restart)
        state_dict = serializer.load()
        restart_x0 = np.asarray(state_dict["x"], dtype=float).ravel()
        restart_iteration = int(state_dict.get("iteration", 0))
        restart_extra = state_dict.get("extra", {})

    # ---- run ----------------------------------------------------------------
    opt_kwargs: dict[str, Any] = {
        "coords": coords,
        "max_iter": geom_max_iter,
        "conv": conv,
        "line_search": geom_line_search,
        "record_trajectory": record_trajectory,
        "progress": progress,
        "hessian_provider": hessian_provider,
        "freeze_indices": geom_freeze,
        "hessian_init": geom_hessian_init,
        "hessian_update": geom_hessian_update,
        "target": geom_target,
        "x0": restart_x0,
        "iteration_start": restart_iteration,
        "extra": restart_extra,
    }
    if geom_opt_options:
        opt_kwargs.update(geom_opt_options)
    opt_kwargs.update(extra_kwargs)

    result = opt_fn(molecule, provider, **opt_kwargs)

    # ---- checkpoint on completion --------------------------------------------
    if geom_checkpoint is not None:
        _save_checkpoint(
            result,
            geom_checkpoint,
            optimizer=opt_name,
            method=getattr(provider, "_method", ""),
            conv=conv,
        )

    # ---- stamp metadata on result -------------------------------------------
    if isinstance(provider, object) and hasattr(provider, "_method"):
        result.method = getattr(provider, "_method", None)
    if isinstance(provider, object) and hasattr(provider, "_basis_name"):
        result.basis = getattr(provider, "_basis_name", None)
    if isinstance(provider, object) and hasattr(provider, "_functional"):
        result.functional = getattr(provider, "_functional", None)

    return result


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------


def _save_checkpoint(
    result: GeomOptResult,
    path: str,
    *,
    optimizer: str = "",
    method: str = "",
    conv: Any = None,
) -> None:
    """Save the final optimisation result as a restart checkpoint."""
    x_flat = np.array([c for a in result.system.atoms for c in a.xyz], dtype=float)
    state = OptimizerState(
        x=x_flat,
        energy=result.energy,
        gradient=result.gradient,
        iteration=result.n_iter,
        n_energy_evals=result.n_energy_evals,
        converged=result.converged,
    )
    restart = build_restart_state(
        state,
        optimizer=optimizer,
        method=method,
        convergence_policy=conv,
    )
    restart["converged"] = result.converged
    serializer = RestartSerializer(path)
    serializer.save(restart)
