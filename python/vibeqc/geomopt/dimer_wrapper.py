"""Dimer-method optimiser registered in the ``vibeqc.geomopt`` framework.

This module wraps :func:`vibeqc.dimer._dimer_search` (the core
algorithm) so it can be selected via ``geom_opt="dimer"`` alongside
the other optimisers (EF, P-RFO, BFGS, ...).  The provider already
supplies (energy, gradient) at each geometry, so the wrapper only
needs to convert ``gradient -> force = -gradient`` and map
``GeomOptResult`` fields from the dimer outcome.

Use
---

.. code-block:: python

    from vibeqc.geomopt import run_geomopt, MolecularSCFProvider, ConvergencePolicy

    provider = MolecularSCFProvider("def2-svp", method="rks", functional="pbe")
    result = run_geomopt(
        ts_guess,
        provider,
        geom_opt="dimer",
        geom_target="transition_state",
        geom_max_iter=100,
        geom_opt_options={"initial_direction": guess_direction},
    )
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .._vibeqc_core import Atom, Molecule
from ..dimer import _dimer_search
from .convergence import ConvergencePolicy
from .providers import EnergyGradientProvider
from .registry import register
from .state import GeomOptResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _positions_of(molecule: Molecule) -> np.ndarray:
    """(n_atoms, 3) Cartesian positions in bohr."""
    return np.array([list(a.xyz) for a in molecule.atoms], dtype=float)


def _rebuild_molecule(template: Molecule, positions: np.ndarray) -> Molecule:
    """Return a copy of *template* with *positions* (bohr)."""
    new_atoms = [Atom(int(a.Z), list(p)) for a, p in zip(template.atoms, positions)]
    return Molecule(new_atoms, template.charge, template.multiplicity)


# ---------------------------------------------------------------------------
# Registered optimiser
# ---------------------------------------------------------------------------


@register("dimer")
def dimer_optimize(
    molecule: Molecule,
    provider: EnergyGradientProvider,
    *,
    coords: Any = None,  # ignored -- dimer always operates in Cartesian space
    max_iter: int = 100,
    conv: Optional[ConvergencePolicy] = None,
    record_trajectory: bool = True,
    progress: bool = False,
    freeze_indices: Optional[Sequence[int]] = None,
    target: str = "transition_state",
    **kwargs: Any,
) -> GeomOptResult:
    """Find a first-order saddle point with the Henkelman-Jonsson dimer method.

    Parameters
    ----------
    molecule : Molecule
        Starting geometry (Cartesian coordinates in bohr).
    provider : EnergyGradientProvider
        Supplies ``(energy, gradient)`` at each geometry.
    coords
        Ignored -- the dimer always works in Cartesian space.
    max_iter : int
        Maximum outer-loop dimer iterations (default 100).
    conv : ConvergencePolicy or None
        ``conv.gmax`` sets the convergence threshold on max-norm
        true force (Ha/bohr).  Default: 1e-3.
    record_trajectory : bool
        Collect all force-evaluation geometries for QVF output.
    progress : bool
        Print per-iteration diagnostics to stdout.
    freeze_indices : sequence of int or None
        Atom indices to hold fixed.
    target : str
        Should be ``"transition_state"`` (default) -- the dimer is a
        TS-only method; minimum searches are handled by other optimisers.
    **kwargs
        Additional dimer-specific options passed through to the core:

        * ``initial_direction`` -- ``(n_atoms, 3)`` guess for the reaction
          mode.  ``None`` (default) => negative initial gradient.
        * ``dimer_separation`` -- half-distance between images (bohr,
          default 0.01).
        * ``max_step`` -- max translation step per iteration (bohr,
          default 0.1).
        * ``translation_step`` / ``initial_step`` -- initial quick-min step
          (bohr, default 0.05).
        * ``rotation_max_iter`` / ``n_rotations`` -- max dimer-rotation
          steps (default 4).
        * ``rotation_force_tol`` -- rotational-force convergence tolerance
          (default 1e-3).
        * ``lbfgs_acceleration`` -- bool (default False).  When True, the
          dimer translation step uses an L-BFGS two-loop recursion to
          build a local quadratic model of the PES, converging
          superlinearly (Kästner & Sherwood, J. Chem. Phys. 128, 014106,
          2008).  Only affects the translation step; rotation is
          unchanged.
        * ``lbfgs_memory`` -- int (default 10).  Number of saved (s, y)
          pairs for L-BFGS acceleration.

    Returns
    -------
    GeomOptResult
    """
    if target != "transition_state":
        raise ValueError(
            "geom_opt='dimer' is a transition-state/saddle-search method; "
            "set geom_target='transition_state'."
        )

    n_atoms = len(list(molecule.atoms))

    # ---- frozen mask ---------------------------------------------------------
    frozen_mask: Optional[np.ndarray] = None
    if freeze_indices:
        fi = {int(i) for i in freeze_indices}
        bad = sorted(i for i in fi if i < 0 or i >= n_atoms)
        if bad:
            raise ValueError(
                f"dimer_optimize: freeze_indices {bad} out of range [0, {n_atoms})"
            )
        frozen_mask = np.zeros(n_atoms, dtype=bool)
        for i in fi:
            frozen_mask[i] = True

    # ---- convergence tolerance -----------------------------------------------
    _conv = conv or ConvergencePolicy.default()
    conv_tol_force = float(_conv.gmax if _conv.gmax is not None else 1e-3)

    # ---- trajectory capture --------------------------------------------------
    _traj_positions: list[np.ndarray] = []
    _traj_energies: list[float] = []

    # ---- force function that wraps the provider ------------------------------
    def force_fn(positions: np.ndarray) -> tuple[float, np.ndarray]:
        """(n_atoms, 3) bohr -> (energy Ha, force Ha/bohr); force = -gradient."""
        mol = _rebuild_molecule(molecule, positions)
        energy, gradient = provider(mol)
        force = -np.asarray(gradient, dtype=float).ravel()
        if record_trajectory:
            _traj_positions.append(positions.copy())
            _traj_energies.append(energy)
        return energy, force

    # ---- initial direction ---------------------------------------------------
    x0 = _positions_of(molecule)
    initial_direction = kwargs.get("initial_direction")
    if initial_direction is None:
        # Evaluate gradient at the start geometry and use the negative
        # gradient as the default direction (points uphill in energy).
        _e0, grad0 = provider(molecule)
        direction = -np.asarray(grad0, dtype=float).reshape(n_atoms, 3)
    else:
        direction = np.asarray(initial_direction, dtype=float).reshape(n_atoms, 3)

    # ---- dimer-specific options from kwargs ----------------------------------
    dimer_separation = float(kwargs.get("dimer_separation", 0.01))
    max_step = float(kwargs.get("max_step", 0.1))
    initial_step = float(
        kwargs.get("translation_step", kwargs.get("initial_step", 0.05))
    )
    n_rotations = int(kwargs.get("rotation_max_iter", kwargs.get("n_rotations", 4)))
    rotation_force_tol = float(kwargs.get("rotation_force_tol", 1e-3))
    lbfgs_acceleration = bool(kwargs.get("lbfgs_acceleration", False))
    lbfgs_memory = int(kwargs.get("lbfgs_memory", 10))

    # ---- run the dimer search ------------------------------------------------
    (R, N, curvature, _energy, _max_force, converged, n_iter, n_evals) = _dimer_search(
        x0,
        force_fn,
        initial_direction=direction,
        dimer_separation=dimer_separation,
        max_iter=max_iter,
        conv_tol_force=conv_tol_force,
        max_step=max_step,
        initial_step=initial_step,
        n_rotations=n_rotations,
        rotation_force_tol=rotation_force_tol,
        frozen_mask=frozen_mask,
        progress=progress,
        lbfgs_acceleration=lbfgs_acceleration,
        lbfgs_memory=lbfgs_memory,
    )

    # ---- build GeomOptResult -------------------------------------------------
    saddle_mol = _rebuild_molecule(molecule, R)
    is_saddle = bool(converged and curvature < 0.0)

    # Final gradient (at the converged/saddle geometry)
    energy_final, grad_final = provider(saddle_mol)
    gradient_flat = np.asarray(grad_final, dtype=float).ravel()

    traj_frames: list[Molecule] = []
    if record_trajectory:
        traj_frames = [_rebuild_molecule(molecule, p) for p in _traj_positions]

    # Build a convergence report from the final gradient
    report = _conv.check(gradient=gradient_flat)

    message = (
        "converged (saddle)"
        if is_saddle
        else "converged (minimum)"
        if converged
        else "max_iter reached"
    )

    return GeomOptResult(
        system=saddle_mol,
        energy=energy_final,
        gradient=gradient_flat,
        converged=converged,
        n_iter=n_iter,
        n_energy_evals=n_evals,
        optimizer="dimer",
        trajectory_frames=traj_frames,
        trajectory_energies=list(_traj_energies),
        is_saddle=is_saddle,
        curvature=curvature,
        mode=N.ravel(),
        convergence_report=report,
        message=message,
    )
