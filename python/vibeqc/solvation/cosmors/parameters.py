"""Named, versioned COSMO-RS parameterizations.

A COSMO-RS parameter set is inseparable from the electronic method, basis,
cavity construction, radii and post-processing it was fitted against. The
constants below are *published values*, read from the papers; what makes them
usable or not is whether the surface they are applied to was built the same
way. Each set therefore records :attr:`Parameterization.fitted_protocol`, and
:class:`~vibeqc.solvation.surface.SurfaceProvenance` records the protocol a
surface actually has, so the pair can be reported and compared instead of
assumed.

Units follow the papers, which is also what keeps the published numbers
recognisable: lengths in angstrom, areas in angstrom^2, screening charge
density in e/angstrom^2, energies in kcal/mol. The conversion from vibe-qc's
atomic units happens once, at the surface boundary
(:mod:`vibeqc.solvation.cosmors.sigma`), not scattered through the equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Gas constant in kcal/(mol K), and room temperature used by the 1998 fit.
R_KCAL = 1.987204258640832e-3
T_ROOM_K = 298.15


@dataclass(frozen=True)
class Parameterization:
    """One named COSMO-RS parameter set.

    Attributes
    ----------
    name, version
        Identity. Reported by every calculation that uses the set.
    fitted_protocol
        The QC/basis/cavity protocol the constants were fitted against, as a
        human-readable token. Compared against a surface's provenance.
    a_eff
        Effective contact-segment area, angstrom^2. Sets ``beta = kT/a_eff``,
        the energy scale of the sigma-potential integral equation.
    alpha_prime
        Misfit energy constant ``alpha' = alpha f_pol``,
        kcal angstrom^2/(mol e^2). Klamt 1998 eq. 7/26.
    f_corr
        Correlation-term coefficient in the misfit energy, eq. 26. Not a free
        parameter: Klamt derives it from the regression against the dielectric
        energy.
    r_av
        Sigma-averaging radius, angstrom. Klamt 1998 eq. 11.
    c_hb, sigma_hb
        Hydrogen-bond strength kcal angstrom^2/(mol e^2) and threshold
        e/angstrom^2. Klamt 1998 eq. 22.
    lambda_comb
        Combinatorial exponent, eq. 30.
    omega_ring, eta_entropy
        Ring correction (kcal/mol per ring atom) and the gas-to-liquid entropy
        constant (dimensionless; enters as ``eta R T``). Eq. 21.
    dispersion
        Element-specific dispersion constants ``gamma_k``,
        kcal/(mol angstrom^2), keyed by atomic number. Eq. 21.
    cavity_radii
        Element cavity radii in angstrom that the set was fitted with. Applying
        the set on a cavity built with different radii is a protocol change.
    """

    name: str
    version: str
    fitted_protocol: str
    a_eff: float
    alpha_prime: float
    f_corr: float
    r_av: float
    c_hb: float
    sigma_hb: float
    lambda_comb: float
    omega_ring: float
    eta_entropy: float
    dispersion: dict[int, float] = field(default_factory=dict)
    cavity_radii: dict[int, float] = field(default_factory=dict)
    reference: str = ""
    notes: str = ""

    def beta(self, temperature_k: float = T_ROOM_K) -> float:
        """``beta = kT / a_eff`` in kcal/(mol angstrom^2).

        The scale that turns the per-area interaction energy into the
        dimensionless exponent of the sigma-potential equation (Klamt 1998
        eq. 18).
        """
        return R_KCAL * float(temperature_k) / self.a_eff

    def supports_elements(self, atomic_numbers) -> tuple[bool, list[int]]:
        """Which requested elements have no dispersion constant.

        Returned rather than raised: a caller computing a pure electrostatic
        sigma profile does not need the dispersion term, and only the
        gas-phase reference (eq. 21) does.
        """
        missing = sorted({int(z) for z in atomic_numbers} - set(self.dispersion))
        return (not missing), missing


# Klamt, Jonas, Buerger & Lohrenz, J. Phys. Chem. A 102, 5074 (1998),
# doi:10.1021/jp980017s, section 5.1 -- the classic COSMO-RS parameterization.
#
# Fitted against DMol / BPW91 / DNP with COSMO at f(eps) = 1 and NSPA = 92
# segments, on the optimized radii below, with the Klamt 1996 outlying-charge
# correction (doi:10.1063/1.472829). That protocol is what `fitted_protocol`
# records; a surface built another way is a different protocol, and the size
# of the resulting deviation has to be measured, not assumed.
#
# The fitting data set (about 1300 experimental values) is the paper's
# Supporting Information, is publisher-restricted, and is deliberately NOT
# reproduced here. Nothing below depends on it: these are the fitted results,
# published in the article text.
KLAMT_1998 = Parameterization(
    name="klamt1998",
    version="1998.1",
    fitted_protocol="dmol/bpw91/dnp/cosmo-inf/nspa92",
    a_eff=7.1,                 # section 5.1, from beta = 0.0832 at kT = 0.592
    alpha_prime=1288.0,        # section 5.1, alpha' = alpha f_pol
    f_corr=2.4,                # eq. 26; derived, not free
    r_av=0.5,                  # section 5.1
    c_hb=7400.0,               # section 5.1
    sigma_hb=0.0082,           # section 5.1
    lambda_comb=0.14,          # eq. 30
    omega_ring=-0.21,          # section 5.1, kcal/mol per ring atom
    eta_entropy=-9.15,         # section 5.1; eta k T = -5.4 kcal/mol at 298 K
    dispersion={
        1: -0.041,   # H
        6: -0.037,   # C
        7: -0.027,   # N
        8: -0.042,   # O
        17: -0.052,  # Cl
    },
    cavity_radii={
        1: 1.30,     # H
        6: 2.00,     # C
        7: 1.83,     # N
        8: 1.72,     # O
        17: 2.05,    # Cl
    },
    reference=(
        "Klamt, Jonas, Buerger & Lohrenz, J. Phys. Chem. A 102, 5074 (1998), "
        "doi:10.1021/jp980017s, section 5.1"
    ),
    notes=(
        "Covers H, C, N, O, Cl only -- the elements the 1998 fit spanned. "
        "The radii here are the fitted cavity radii and are 13-18 percent "
        "larger than the corresponding van der Waals radii."
    ),
)

PARAMETERIZATIONS: dict[str, Parameterization] = {
    KLAMT_1998.name: KLAMT_1998,
}

DEFAULT_PARAMETERIZATION = KLAMT_1998.name


def get_parameterization(name: Optional[str] = None) -> Parameterization:
    """Look up a named set. ``None`` gives the default."""
    key = (name or DEFAULT_PARAMETERIZATION).strip().lower()
    try:
        return PARAMETERIZATIONS[key]
    except KeyError:
        raise ValueError(
            f"unknown COSMO-RS parameterization {name!r} "
            f"(known: {', '.join(sorted(PARAMETERIZATIONS))})."
        ) from None


__all__ = [
    "DEFAULT_PARAMETERIZATION",
    "KLAMT_1998",
    "PARAMETERIZATIONS",
    "Parameterization",
    "R_KCAL",
    "T_ROOM_K",
    "get_parameterization",
]
