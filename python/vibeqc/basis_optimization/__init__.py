"""Basis-set optimisation engine.

Originally the Goal-4 deliverable on the ``basissetdev`` branch; this
package **ships on ``main``** and is not basissetdev-conditional. Do not
read CLAUDE.md § 4 as gating it -- that rule governs the basissetdev
branch and its BSE-fetched basis sets, and names neither this package
nor the co-located ``vibe-basis/`` driver. (Stating this because the
stale "on the basissetdev branch" wording here was read, on 2026-07-28,
as meaning basis optimisation is outside the release/deploy pipeline
for that reason. It is outside it, but for an unrelated one -- see
``handovers/HANDOVER_VIBE_BASIS_DEPLOYMENT.md``.)

Reproduces the historical pob optimisation recipe (Peintinger 2013,
Vilela Oliveira 2019): single-point energies driven by an outer
numerical optimiser across a set of reference systems.

The CRYSTAL recipes here (:mod:`~vibeqc.basis_optimization.calculators`
and ``recipes.crystal_stage1..3`` / ``crystal_objective`` /
``production``) import ``vibe_basis`` at module top level and are
unimportable without it. Install it with vibe-qc's ``[basisopt]``
extra; everything else in this package works without.

Original architecture: CRYSTAL09/17 + MINUIT2 (ROOT) + python wrapper.
This re-implementation: vibe-qc + iminuit (libminuit2) / scipy + python.

Module layout
-------------

* :mod:`vibeqc.basis_optimization.parametrise` -- turn a basis dict
  into a flat parameter vector and back, with log-space transforms,
  per-parameter bounds, and a fixed-vs-free designation.
* :mod:`vibeqc.basis_optimization.objective` -- protocol + base
  classes for objective functions. The default
  :class:`MultiSystemEnergy` sums per-system SCF energies with
  weights and adds the linear-dependence penalty when active.
* :mod:`vibeqc.basis_optimization.io` -- write a basis as a temp
  ``.g94`` file libint can pick up via ``LIBINT_DATA_PATH``.
* :mod:`vibeqc.basis_optimization.ld_diagnostics` -- overlap-matrix
  diagnostics plus two penalty terms: ``ld_penalty`` (a hinge on the
  smallest overlap eigenvalue, the linear-dependence guard the pob
  papers handled by raising the lower exponent threshold) and
  ``condition_number_penalty`` (the smooth g.ln κ(S) term of the
  CRYSTAL OPTBASIS / VandeVondele objective).
* :mod:`vibeqc.basis_optimization.drivers` -- scipy and iminuit
  driver shims with a common :class:`OptResult` return type.
* :mod:`vibeqc.basis_optimization.bdiis` -- ``optimize_bdiis``, a
  robust BDIIS / GDIIS driver (DIIS extrapolation + BFGS curvature +
  trust-region backtracking) returning the same :class:`OptResult`.
  This is the basis-parameter analogue of CRYSTAL23's OPTBASIS.
* :mod:`vibeqc.basis_optimization.gradients` -- analytic gradients of
  the overlap-based penalty terms
  (``condition_number_penalty_gradient``, ``ld_penalty_gradient``),
  built on the closed-form same-center overlap derivative dS/da, so a
  BDIIS ``grad=`` need not finite-difference the (stiff) penalty.
* :mod:`vibeqc.basis_optimization.recipes` -- composed pipelines.
  Stage 1 (``recipes.single_atom``) is the architecture smoke
  test: optimise one parameter on a single-atom RHF and verify
  the optimiser lands at the expected minimum.

Stages
------

Each stage is a publishable validation milestone:

1. **Single-atom HF energy minimisation.** One free parameter on a
   one-system objective. Architecture smoke test. Laptop-runnable.
2. **One-compound bulk solid optimisation.** Drop H's diffuse-most
   primitive in pob-TZVP, refit the valence on LiH at multi-k. Needs
   periodic-features R1-R4 from REQUIREMENTS-PERIODIC.md.
3. **Multi-compound simultaneous optimisation.** The actual pob /
   pob-rev2 recipe. Needs the test-set inputs from Goal 3 +
   periodic-feature stack.
4. **LD-aware multi-compound optimisation.** Adds the
   :mod:`ld_penalty` term with a sensible l, defends against the
   "exponent collapses to zero, energy diverges, no SCF convergence"
   failure mode that the pob papers handled by hand.

References
----------

* Peintinger, Vilela Oliveira, Bredow, J. Comput. Chem. 34, 451 (2013).
* Vilela Oliveira, Laun, Peintinger, Bredow, J. Comput. Chem. 40, 2364 (2019).
* Daga, Civalleri, Maschio, J. Chem. Theory Comput. 16, 2192 (2020)
  [BDIIS].
* VandeVondele, Hutter, J. Chem. Phys. 127, 114105 (2007)
  [condition-number penalty].
* iminuit: https://iminuit.readthedocs.io
"""

from .parametrise import (
    BasisParametrisation,
    FreeSpec,
    Transform,
)
from .objective import (
    Objective,
    MockEnergy,
)
from .drivers import (
    OptResult,
    optimize_scipy,
    optimize_minuit,
)
from .bdiis import (
    optimize_bdiis,
    method_citations,
    BDIIS_CITATION_KEYS,
)
from .ld_diagnostics import (
    condition_number_penalty,
    cond_penalty_from_atom,
)
from .gradients import (
    condition_number_penalty_gradient,
    ld_penalty_gradient,
    penalty_gradient,
)
from .energy_gradient import (
    energy_gradient_fd,
    energy_gradient_analytic,
    energy_gradient_analytic_uhf,
    energy_gradient_analytic_rks,
    energy_gradient_analytic_uks,
    build_g,
    build_j,
    build_k,
    electronic_energy,
    energy_weighted_density,
    VibeqcIntegralProvider,
)
from .periodic_energy_gradient import (
    periodic_energy_gradient_fd,
    periodic_electronic_energy,
    energy_weighted_density_k,
    PeriodicIntegralProvider,
    bloch_summed_one_electron_exponent_derivatives,
    periodic_energy_gradient_analytic,
    PeriodicCoulombNuclearDerivativeProvider,
    periodic_xc_param_gradient_term,
    build_periodic_xc_gradient_grid,
    periodic_xc_param_gradient_term_uks,
    build_periodic_xc_gradient_grid_uks,
)

__all__ = [
    "BasisParametrisation",
    "FreeSpec",
    "Transform",
    "Objective",
    "MockEnergy",
    "OptResult",
    "optimize_scipy",
    "optimize_minuit",
    "optimize_bdiis",
    "method_citations",
    "BDIIS_CITATION_KEYS",
    "condition_number_penalty",
    "cond_penalty_from_atom",
    "condition_number_penalty_gradient",
    "ld_penalty_gradient",
    "penalty_gradient",
    "energy_gradient_fd",
    "energy_gradient_analytic",
    "energy_gradient_analytic_uhf",
    "energy_gradient_analytic_rks",
    "energy_gradient_analytic_uks",
    "build_g",
    "build_j",
    "build_k",
    "electronic_energy",
    "energy_weighted_density",
    "VibeqcIntegralProvider",
    "periodic_energy_gradient_fd",
    "periodic_electronic_energy",
    "energy_weighted_density_k",
    "PeriodicIntegralProvider",
    "bloch_summed_one_electron_exponent_derivatives",
    "periodic_energy_gradient_analytic",
    "PeriodicCoulombNuclearDerivativeProvider",
    "periodic_xc_param_gradient_term",
    "build_periodic_xc_gradient_grid",
    "periodic_xc_param_gradient_term_uks",
    "build_periodic_xc_gradient_grid_uks",
]
