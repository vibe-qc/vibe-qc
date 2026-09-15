"""Uniform geometry optimisation framework -- ``vibeqc.geomopt``.

This package replaces the three disjoint optimisation backends
(ASE/BFGS, scipy/L-BFGS-B, Brent/steepest descent) with a single
architecture and keyword-selectable interface.

Quickstart
----------
::

    from vibeqc.geomopt import (
        run_geomopt,
        MolecularSCFProvider,
        CartesianCoordinates,
        ConvergencePolicy,
    )

    provider = MolecularSCFProvider("def2-svp", method="rks", functional="PBE")
    result = run_geomopt(
        molecule,
        provider,
        geom_opt="bfgs",
        geom_coords="cartesian",
        geom_target="minimum",
        geom_line_search="backtracking",
        geom_conv=ConvergencePolicy(gmax=4.5e-4),
    )

Package structure
-----------------
* :mod:`vibeqc.geomopt.providers` -- ``EnergyGradientProvider``,
  ``HessianProvider``, ``MolecularSCFProvider``
* :mod:`vibeqc.geomopt.coordinates` -- ``CoordinateRepresentation``,
  ``CartesianCoordinates``
* :mod:`vibeqc.geomopt.state` -- ``GeomOptResult``, ``OptimizerState``
* :mod:`vibeqc.geomopt.convergence` -- ``ConvergencePolicy``,
  ``ConvergenceReport``
* :mod:`vibeqc.geomopt.line_search` -- ``resolve_line_search``
* :mod:`vibeqc.geomopt.optimizers` -- ``resolve_optimizer``,
  ``steepest_descent``, ``conjugate_gradient``, ``bfgs``, ``lbfgs``
"""

from __future__ import annotations

# Phase 3: trust-region + RFO -- trigger registration
# Phase 5: EF + P-RFO -- trigger registration
# Phase 6: GDIIS + FIRE -- trigger registration
from . import (
    dimer_wrapper,  # noqa: F401
    gdiis_fire,  # noqa: F401
    trust,  # noqa: F401
    ts,  # noqa: F401
)
from .convergence import ConvergencePolicy, ConvergenceReport
from .coordinates import CartesianCoordinates, CoordinateRepresentation
from .history import (
    HistoryManager,
    HistoryRecord,
    RestartSerializer,
    build_restart_state,
    restore_optimizer_state,
)
from .internal_coords import DelocalizedInternalCoordinates
from .optimizers import (
    bfgs,
    conjugate_gradient,
    lbfgs,
    resolve_optimizer,
    steepest_descent,
)  # noqa: F401
from .periodic_providers import (
    CellStrainCoordinates,
    FractionalCoordinates,
    PeriodicSCFProvider,
    run_periodic_geomopt,
)
from .providers import (
    EnergyGradientProvider,
    HessianProvider,
    HessianVectorProductProvider,
    MolecularHessianFDProvider,
    MolecularSCFProvider,
)
from .registry import resolve as resolve_optimizer  # alias for convenience
from .runner import run_geomopt
from .state import GeomOptResult, OptimizerState

__all__ = [
    # Providers
    "EnergyGradientProvider",
    "HessianProvider",
    "HessianVectorProductProvider",
    "MolecularSCFProvider",
    "MolecularHessianFDProvider",
    # Coordinates
    "CoordinateRepresentation",
    "CartesianCoordinates",
    "DelocalizedInternalCoordinates",
    "FractionalCoordinates",
    "CellStrainCoordinates",
    # Periodic
    # State
    "GeomOptResult",
    "OptimizerState",
    # Convergence
    "ConvergencePolicy",
    "ConvergenceReport",
    # History / restart
    "HistoryManager",
    "HistoryRecord",
    "RestartSerializer",
    "build_restart_state",
    "restore_optimizer_state",
    # Optimizer functions
    "steepest_descent",
    "conjugate_gradient",
    "bfgs",
    "lbfgs",
    "resolve_optimizer",
    # Runner
    "run_geomopt",
    "run_periodic_geomopt",
    "PeriodicSCFProvider",
]
