"""State model for the uniform geometry optimization framework.

:class:`GeomOptResult` is the single return type every optimizer
produces.  It supersedes the three currently-scattered result types
(:class:`~vibeqc.molecular_optimize.MolecularOptimizeResult`,
:class:`~vibeqc.bipole_optimize.OptimizeResult`, and
:class:`~vibeqc.dimer.DimerResult`) and carries the fields all three
currently track, plus the new convergence report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .._vibeqc_core import Molecule, PeriodicSystem

System = Molecule  # alias for type annotations


@dataclass
class GeomOptResult:
    """Standard result from any geometry optimisation in vibe-qc.

    Every optimiser (steepest descent, CG, BFGS, L-BFGS, trust-region,
    RFO, EF, P-RFO, dimer, FIRE, GDIIS) returns this struct.  Fields
    that don't apply to a particular optimiser are ``None`` or empty.

    Attributes
    ----------
    system
        Converged (or final) geometry as a Molecule or PeriodicSystem.
    energy
        Total energy at *system* (Hartree).
    gradient
        Flat ``(3N,)`` gradient gradE at *system* (Ha/bohr), or an empty
        array when unavailable.
    converged
        ``True`` iff the convergence policy gates all passed.
    n_iter
        Number of optimisation steps taken.
    n_energy_evals
        Total energy+gradient evaluations consumed.
    optimizer
        Short name of the optimiser used (``"sd"``, ``"bfgs"``, ...).
    method
        Electronic-structure method string (``"rhf"``, ``"rks"``, ...).
    basis
        Basis-set name, or ``None`` for semiempirical.
    functional
        XC functional name, or ``None``.
    trajectory_frames
        Per-step Molecule snapshots (first = start, last = final).
    trajectory_energies
        Per-step total energies (same order as *trajectory_frames*).
    # TS-specific
    is_saddle
        ``True`` iff this is a converged first-order saddle.
    curvature
        PES curvature along the reaction mode (Ha/bohr^2), for TS runs.
    mode
        Lowest-curvature eigenvector (flat ``(3N,)``, unit norm),
        for TS runs.
    convergence_report
        Full per-gate convergence diagnostic.
    message
        Human-readable outcome summary.
    """

    system: System
    energy: float
    gradient: np.ndarray = field(default_factory=lambda: np.array([]))
    converged: bool = False
    n_iter: int = 0
    n_energy_evals: int = 0
    optimizer: str = ""
    method: Optional[str] = None
    basis: Optional[str] = None
    functional: Optional[str] = None
    trajectory_frames: list[System] = field(default_factory=list)
    trajectory_energies: list[float] = field(default_factory=list)
    # TS-specific
    is_saddle: Optional[bool] = None
    curvature: Optional[float] = None
    mode: Optional[np.ndarray] = None
    convergence_report: Optional[object] = None  # ConvergenceReport
    message: str = ""

    def __repr__(self) -> str:
        g = np.abs(np.asarray(self.gradient, dtype=float))
        grad_max = float(np.max(g)) if g.size else float("inf")
        lines = [
            f"GeomOptResult(",
            f"  optimizer={self.optimizer!r}, method={self.method!r},",
            f"  energy={self.energy:.8f}, max|grad|={grad_max:.4e},",
            f"  n_iter={self.n_iter}, converged={self.converged})",
        ]
        if self.is_saddle is not None:
            lines.insert(-1, f"  is_saddle={self.is_saddle},")
        return "\n".join(lines)


@dataclass
class OptimizerState:
    """Mutable working state carried across steps by every optimiser.

    This is not the result -- it's the scratchpad the optimiser updates
    in-place each iteration.  Subclasses may add optimiser-specific
    fields (e.g. ``H_inv`` for quasi-Newton, ``trust_radius`` for
    trust-region methods).
    """

    x: np.ndarray  # current flat parameter vector
    energy: float  # current total energy (Ha)
    gradient: np.ndarray  # current flat gradient (Ha/bohr)
    iteration: int = 0
    converged: bool = False
    n_energy_evals: int = 0
    x_prev: Optional[np.ndarray] = None
    energy_prev: Optional[float] = None
