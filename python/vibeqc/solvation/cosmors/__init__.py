"""COSMO-RS / COSMOSPACE: statistical thermodynamics on conductor surfaces.

Method-independent by construction. The layer's only input is a
:class:`~vibeqc.solvation.surface.ConductorSurface` -- segments with areas and
screening charge densities -- plus a named
:class:`~vibeqc.solvation.cosmors.parameters.Parameterization`. It never sees a
Hamiltonian, a basis or a density, so any electronic method that can produce a
conductor surface gets COSMO-RS for free.

Pipeline::

    ConductorSurface
      -> segment_descriptors      sigma averaging        (Klamt 1998 eq. 11, 14)
      -> sigma_profile            p(sigma)               (eq. 16)
      -> sigma_potential_*        mu(sigma), iterated    (eq. 18 / 23-25)
      -> chemical_potential       mu*_S^X                (eq. 19, 20)
      -> dG_solv, activity and partition coefficients

COSMOSPACE (:mod:`.cosmospace`) solves the same pairing problem exactly
rather than under the independent-patch approximation, and is exposed
alongside rather than in place of it -- the shipped parameterization was
fitted with the sigma potential, so the two are not interchangeable at
fixed parameters.

References
----------
* Klamt, *J. Phys. Chem.* **99**, 2224 (1995), doi:10.1021/j100007a062 --
  the COSMO-RS derivation.
* Klamt, Jonas, Buerger & Lohrenz, *J. Phys. Chem. A* **102**, 5074 (1998),
  doi:10.1021/jp980017s -- refinement and the parameterization used here.
* Klamt & Schuurmann, *J. Chem. Soc. Perkin Trans. 2* 799 (1993),
  doi:10.1039/P29930000799 -- the underlying COSMO model.
* Klamt, Krooshof & Taylor, *AIChE J.* **48**, 2332 (2002),
  doi:10.1002/aic.690481023 -- COSMOSPACE.
"""

from __future__ import annotations

from .cosmospace import (
    SG_COORDINATION_NUMBER,
    SegmentActivity,
    binary_segment_activity_coefficient,
    exchange_energy_matrix,
    residual_ln_activity_coefficient,
    segment_activity_coefficients,
    staverman_guggenheim_ln_gamma,
    tau_matrix,
)
from .parameters import (
    DEFAULT_PARAMETERIZATION,
    KLAMT_1998,
    PARAMETERIZATIONS,
    Parameterization,
    get_parameterization,
)
from .potential import (
    SigmaPotential,
    hydrogen_bond_energy,
    interaction_energy,
    interaction_energy_dsigma,
    misfit_energy,
    sigma_potential_profile,
    sigma_potential_segments,
)
from .sigma import (
    SegmentDescriptors,
    SigmaProfile,
    average_sigma,
    default_sigma_grid,
    mixture_sigma_profile,
    segment_descriptors,
    sigma_profile,
)
from .thermo import (
    ChemicalPotential,
    activity_coefficient,
    chemical_potential,
    gas_phase_chemical_potential,
    ln_activity_coefficient,
    log_partition_coefficient,
    per_element_area,
    solvation_free_energy,
)

__all__ = [
    "tau_matrix",
    "staverman_guggenheim_ln_gamma",
    "segment_activity_coefficients",
    "residual_ln_activity_coefficient",
    "exchange_energy_matrix",
    "binary_segment_activity_coefficient",
    "SegmentActivity",
    "SG_COORDINATION_NUMBER",
    "DEFAULT_PARAMETERIZATION",
    "KLAMT_1998",
    "PARAMETERIZATIONS",
    "ChemicalPotential",
    "Parameterization",
    "SegmentDescriptors",
    "SigmaPotential",
    "SigmaProfile",
    "activity_coefficient",
    "average_sigma",
    "chemical_potential",
    "default_sigma_grid",
    "gas_phase_chemical_potential",
    "get_parameterization",
    "hydrogen_bond_energy",
    "interaction_energy",
    "interaction_energy_dsigma",
    "ln_activity_coefficient",
    "log_partition_coefficient",
    "misfit_energy",
    "mixture_sigma_profile",
    "per_element_area",
    "segment_descriptors",
    "sigma_potential_profile",
    "sigma_potential_segments",
    "sigma_profile",
    "solvation_free_energy",
]
