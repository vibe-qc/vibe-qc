"""Stage-1 recipe: single-atom HF energy minimisation.

The simplest non-trivial test of the basis-optimisation pipeline.
We run an isolated atom (closed-shell or open-shell as appropriate)
under RHF / UHF, declare ONE exponent free, and verify the optimiser
converges to a sensible value.

This recipe does not depend on periodic features -- it runs entirely
through the molecular SCF path (``vq.run_rhf`` / ``run_uhf``). It is
therefore the only stage that is laptop-runnable today; stages 2-4
need REQUIREMENTS-PERIODIC.md R1-R5 to land first.

The recipe is intentionally single-system. The pob recipe is
multi-system; this stage is about validating the architecture, not
reproducing pob numbers. Stage 2 onwards adds the multi-system
energy reducer.

Use::

    from vibeqc.basis_optimization.recipes.single_atom import run_h_smoke
    result = run_h_smoke()
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# These imports are deferred at usage time so the module can be
# inspected (e.g. by docs builders) without a working vibeqc install.


@dataclass
class SmokeResult:
    """Outcome of a single-atom smoke run."""

    element: str
    free_field: str
    starting_value: float
    optimal_value: float
    starting_energy: float
    optimal_energy: float
    n_evaluations: int
    wall_seconds: float
    converged: bool

    def summary(self) -> str:
        delta_e_mha = (self.starting_energy - self.optimal_energy) * 1000.0
        delta_x_pct = (
            100.0 * (self.optimal_value - self.starting_value) / self.starting_value
            if self.starting_value
            else float("nan")
        )
        return (
            f"single-atom {self.element} smoke ({self.free_field})\n"
            f"  start:    x={self.starting_value:.6f}  E={self.starting_energy:.8f} Ha\n"
            f"  optimum:  x={self.optimal_value:.6f}  E={self.optimal_energy:.8f} Ha\n"
            f"  Δx        {delta_x_pct:+7.2f} %\n"
            f"  ΔE        {delta_e_mha:+8.4f} mHa  (lower is better)\n"
            f"  evals:    {self.n_evaluations}   wall: {self.wall_seconds:.1f} s\n"
            f"  converged: {self.converged}"
        )


def run_h_smoke(
    *,
    perturbation: float = 1.5,
    bounds: tuple[float, float] = (0.05, 1.0),
    max_iter: int = 30,
) -> SmokeResult:
    """Vary pob-TZVP H's most-diffuse s exponent and watch the optimiser.

    Reference state: pob-TZVP H has shells

      * S 3-prim (contracted core, 34.06, 5.12, 1.16)
      * S 1-prim (0.4157)
      * S 1-prim (0.1795)   <- the one we vary
      * P 1-prim (0.8000)

    The H atom is a single-electron system, so RHF energy is exact at
    the basis limit. Varying the diffuse-most exponent does change
    the energy weakly. The optimiser should slide toward a value
    that minimises the H-atom energy in this fixed basis (which,
    for the diffuse-most s with all other shells fixed, is the
    value that best represents the trailing 1s tail).

    The "perturbation" lets us start from a *wrong* exponent
    (default 1.5x the published value) and verify the optimiser
    pulls back to a sensible region. The exact landing point is
    not the published 0.1795 because we vary only one parameter
    against a different objective (atomic energy, not the multi-
    compound pob calibration set).

    Returns
    -------
    SmokeResult
    """
    import vibeqc as vq  # type: ignore[import-not-found]
    from ..parametrise import BasisParametrisation, FreeSpec, Transform
    from ..objective import SinglePointEnergy
    from ..io import TempBasisLibrary
    from ..drivers import optimize_scipy
    from ...basis_crystal import parse_crystal_atom_basis_file

    pkg_root = Path(vq.__file__).parent
    src_h = pkg_root / "basis_library" / "sources" / "pob-TZVP" / "01_H"
    if not src_h.exists():
        # Fall back to the in-tree copy when running from a worktree.
        from .. import __file__ as bo_file
        worktree_root = Path(bo_file).resolve().parents[3]
        src_h = (
            worktree_root
            / "python" / "vibeqc" / "basis_library" / "sources" / "pob-TZVP" / "01_H"
        )
    atoms = {"H": parse_crystal_atom_basis_file(src_h)}

    parametrisation = BasisParametrisation(
        atoms=atoms,
        free=[FreeSpec(symbol="H", shell_idx=2, prim_idx=0,
                       field="exponent", transform=Transform.LOG,
                       bounds=bounds, label="H_diffuse_s")],
    )
    starting = parametrisation.atoms["H"].shells[2].exponents[0]

    # Hydrogen atom: doublet ground state, single electron -> UHF.
    def molecule_factory():
        return vq.Molecule(
            [vq.Atom(1, [0.0, 0.0, 0.0])],
            multiplicity=2,
        )

    with TempBasisLibrary() as lib:
        # Energy at the perturbed start.
        x_start = parametrisation.pack() + np.log(perturbation)
        objective = SinglePointEnergy(
            parametrisation=parametrisation,
            molecule_factory=molecule_factory,
            library=lib,
            scf_runner=vq.run_uhf,
        )
        e_start = objective(x_start)
        # Optimisation.
        res = optimize_scipy(
            objective, x0=x_start,
            bounds=parametrisation.optim_bounds(),
            method="L-BFGS-B",
            tol=1e-8,
            max_iter=max_iter,
        )

    optimal = parametrisation.unpack(res.x)["H"].shells[2].exponents[0]
    return SmokeResult(
        element="H",
        free_field="diffuse-most s exponent",
        starting_value=starting * perturbation,
        optimal_value=optimal,
        starting_energy=e_start,
        optimal_energy=res.fun,
        n_evaluations=res.n_evaluations,
        wall_seconds=res.wall_seconds,
        converged=res.success,
    )


if __name__ == "__main__":
    print(run_h_smoke().summary())
