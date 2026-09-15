"""Chemical potentials and derived properties -- Klamt 1998 eq. 19-21, 30.

Once the solvent's sigma potential is known, a solute's chemical potential in
that solvent is a single integral of the solute's sigma profile against it
(eq. 19/20). Everything users actually ask for -- activity coefficients,
partition coefficients, solvation free energies, solubilities -- is a
difference of two such chemical potentials.

Reference states are where this gets silently wrong, so they are named
explicitly at every entry point rather than left implicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .parameters import R_KCAL, Parameterization, T_ROOM_K
from .potential import SigmaPotential
from .sigma import SegmentDescriptors, SigmaProfile

HARTREE_TO_KCAL = 627.5094740631


@dataclass(frozen=True)
class ChemicalPotential:
    """A solute's chemical potential in one ensemble, kcal/mol.

    Attributes
    ----------
    total : float
        ``mu*_S^X`` -- the standard chemical potential relative to the ideally
        screened state, at unimolar concentration (eq. 19).
    restoring : float
        ``beta mu~_S^X`` -- the electrostatic/hydrogen-bond part, i.e. the
        integral of the solute profile against the solvent sigma potential.
    combinatorial : float
        ``-lambda k T ln A^S`` -- the size-dependent term of eq. 19.
    solute_area, solvent_area : float
        angstrom^2, carried because every derived property needs them.
    parameterization, temperature_k : str, float
        Provenance. Reported so a number is never quotable without the
        parameter set and temperature that produced it.
    """

    total: float
    restoring: float
    combinatorial: float
    solute_area: float
    solvent_area: float
    parameterization: str
    temperature_k: float


def per_element_area(
    surface, descriptors: SegmentDescriptors
) -> dict[int, float]:
    """Exposed surface area per element, angstrom^2 -- ``A^X_k`` of eq. 21.

    Sums segment areas by the element of the owning atom. Uses the same
    filtered segment set the descriptors were built from, so the areas add up
    to the descriptors' total rather than the raw cavity's.
    """
    from ..surface import MIN_SEGMENT_AREA_BOHR2

    areas_b = np.asarray(surface.areas, dtype=np.float64)
    keep = areas_b > MIN_SEGMENT_AREA_BOHR2
    owner = np.asarray(surface.segment_atom, dtype=int)[keep]
    z = np.asarray(surface.atomic_numbers, dtype=int)
    areas = np.asarray(descriptors.areas, dtype=np.float64)

    out: dict[int, float] = {}
    for element in np.unique(z[owner]):
        out[int(element)] = float(np.sum(areas[z[owner] == element]))
    return out


def chemical_potential(
    solute_profile: SigmaProfile,
    solvent_potential: SigmaPotential,
    params: Parameterization,
    *,
    solvent_area: float,
    temperature_k: float = T_ROOM_K,
) -> ChemicalPotential:
    """``mu*_S^X`` for solute ``X`` in solvent ``S``, kcal/mol (eq. 19/20).

        mu~_S^X = int dsigma p^X(sigma) mu~_S(sigma)                 (20)
        mu*_S^X = beta mu~_S^X - lambda k T ln A^S                   (19)

    ``p^X`` is the solute's *unnormalised* profile, integrating to ``A^X``, so
    the first term scales with the solute's surface as it must.

    The result is relative to the ideally screened state and at unimolar
    concentration. Converting to a different concentration reference is the
    caller's business and is what the ``-kT ln x`` of the paper's discussion
    handles; doing it silently here is how reference states get lost.
    """
    grid = solute_profile.sigma_grid
    mu_tilde = solvent_potential(grid)
    restoring = params.beta(temperature_k) * float(
        np.sum(solute_profile.p * mu_tilde) * solute_profile.dsigma
    )
    kT = R_KCAL * float(temperature_k)
    if solvent_area <= 0.0:
        raise ValueError("chemical_potential: solvent area must be positive.")
    combinatorial = -params.lambda_comb * kT * math.log(solvent_area)
    return ChemicalPotential(
        total=restoring + combinatorial,
        restoring=restoring,
        combinatorial=combinatorial,
        solute_area=solute_profile.total_area,
        solvent_area=float(solvent_area),
        parameterization=params.name,
        temperature_k=float(temperature_k),
    )


def gas_phase_chemical_potential(
    surface,
    descriptors: SegmentDescriptors,
    params: Parameterization,
    *,
    ideal_screening_energy_hartree: Optional[float] = None,
    n_ring_atoms: int = 0,
    temperature_k: float = T_ROOM_K,
) -> float:
    """``mu'^X_gas`` of eq. 21, kcal/mol.

        mu'^X_gas = -Delta'^X - sum_k gamma_k A^X_k - omega n^X_ra - eta R T

    The first term is the negative of the ideal screening energy: the cost of
    stripping a molecule of its perfect conductor screening. The second is
    dispersion, taken as proportional to exposed area per element. The third
    is Klamt's empirical ring correction, and the fourth the gas-to-liquid
    entropy constant of the paper's chosen reference state.

    Raises when an element has no dispersion constant rather than treating it
    as zero: a silent zero would read as "this element does not disperse",
    which is a physical claim the parameterization is not making. It simply
    was not fitted for that element.
    """
    if ideal_screening_energy_hartree is None:
        delta = surface.ideal_screening_energy
        if delta is None:
            raise ValueError(
                "gas_phase_chemical_potential: the surface carries no ideal "
                "screening energy (needs both the gas-phase and conductor "
                "energies). Supply it explicitly rather than assuming zero."
            )
    else:
        delta = float(ideal_screening_energy_hartree)

    areas = per_element_area(surface, descriptors)
    ok, missing = params.supports_elements(areas)
    if not ok:
        raise ValueError(
            f"gas_phase_chemical_potential: parameterization "
            f"{params.name!r} has no dispersion constant for element(s) "
            f"{missing}; it was fitted for "
            f"{sorted(params.dispersion)}. Refusing to substitute zero."
        )

    dispersion = sum(params.dispersion[z] * a for z, a in areas.items())
    return (
        -delta * HARTREE_TO_KCAL
        - dispersion
        - params.omega_ring * float(n_ring_atoms)
        - params.eta_entropy * R_KCAL * float(temperature_k)
    )


def solvation_free_energy(
    mu_solvent: ChemicalPotential, mu_gas_kcal: float
) -> float:
    """``dG_solv = mu*_S^X - mu'^X_gas``, kcal/mol.

    Both sides are referenced to the same ideally screened state, which is why
    the difference is meaningful; mixing a solvent chemical potential from this
    module with a gas reference computed another way is not.
    """
    return float(mu_solvent.total) - float(mu_gas_kcal)


def ln_activity_coefficient(
    mu_in_solvent: ChemicalPotential,
    mu_in_pure_solute: ChemicalPotential,
    *,
    temperature_k: float = T_ROOM_K,
) -> float:
    """``ln gamma^X_S`` from the two chemical potentials.

        ln gamma = (mu*_S^X - mu*_X^X) / RT

    The reference is the pure solute, so ``gamma -> 1`` as the solvent becomes
    the solute. That limit is the cheapest available check on a mixture
    calculation and is worth asserting in any workflow built on this.
    """
    kT = R_KCAL * float(temperature_k)
    return (float(mu_in_solvent.total) - float(mu_in_pure_solute.total)) / kT


def activity_coefficient(
    mu_in_solvent: ChemicalPotential,
    mu_in_pure_solute: ChemicalPotential,
    *,
    temperature_k: float = T_ROOM_K,
) -> float:
    return math.exp(
        ln_activity_coefficient(
            mu_in_solvent, mu_in_pure_solute, temperature_k=temperature_k
        )
    )


def log_partition_coefficient(
    mu_phase_a: ChemicalPotential,
    mu_phase_b: ChemicalPotential,
    *,
    temperature_k: float = T_ROOM_K,
    molar_volume_ratio: float = 1.0,
) -> float:
    """``log10 K_{B/A}`` for a solute distributed between two phases.

        log K = (mu*_A^X - mu*_B^X) / (RT ln 10) + log10(V_A / V_B)

    A positive value means the solute prefers phase ``B``. The molar-volume
    ratio converts between mole-fraction and molarity references; it defaults
    to 1, which is the mole-fraction convention. Partition coefficients in the
    literature are usually molarity-based (log P for octanol/water is), so
    supplying the ratio is normally required rather than optional.
    """
    kT_ln10 = R_KCAL * float(temperature_k) * math.log(10.0)
    if molar_volume_ratio <= 0.0:
        raise ValueError(
            "log_partition_coefficient: molar volume ratio must be positive."
        )
    return (
        float(mu_phase_a.total) - float(mu_phase_b.total)
    ) / kT_ln10 + math.log10(molar_volume_ratio)


__all__ = [
    "HARTREE_TO_KCAL",
    "ChemicalPotential",
    "activity_coefficient",
    "chemical_potential",
    "gas_phase_chemical_potential",
    "ln_activity_coefficient",
    "log_partition_coefficient",
    "per_element_area",
    "solvation_free_energy",
]
