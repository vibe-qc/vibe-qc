"""A solvent sigma potential from a real COSMO calculation (#558).

Direct COSMO-RS needs the *solvent's* sigma potential, and until now the only
ones in this package were hand-built toys. The chain from a converged solvent
run to a usable potential is short --

    run_cpcm_scf(conductor) -> build_conductor_surface -> segment_descriptors
                            -> sigma_potential_segments

-- but three things about it are easy to get wrong and none of them announce
themselves in the result, so they are checked here instead of documented.

**The run must be a conductor.** :func:`~vibeqc.solvation.surface.build_conductor_surface`
takes the charges as stored, and they are the ideal ``q*`` only at
``epsilon = inf``; at finite dielectric vibe-qc stores ``q = f q*``, and a sigma
profile built from those is screened rather than ideal. Nothing downstream can
tell the difference: the profile looks perfectly reasonable, just wrong by a
factor that depends on the dielectric constant.

**The cavity radii must be the parameterization's own.** A COSMO-RS parameter
set is fitted against a specific radius set -- ``KLAMT_1998`` against
H 1.30, C 2.00, N 1.83, O 1.72, Cl 2.05 angstrom -- and the default scaled-Bondi
cavity is a different surface. Since ``sigma = q/a``, changing the radii changes
every sigma the constants were fitted to reproduce.

**The QC protocol usually will not match, and that cannot be fixed here.**
``KLAMT_1998`` was fitted on ``dmol/bpw91/dnp/cosmo-inf/nspa92``; anything this
package computes carries its own method and basis. That is a real accuracy
limit and not a bug, so it is recorded on the potential rather than raised:
:attr:`~vibeqc.solvation.cosmors.potential.SigmaPotential.protocol` says what
the surface was, and ``params.fitted_protocol`` says what the constants expect.
Comparing two free-form tokens is a judgement, so the judgement is left to
whoever reads the number, with both halves in front of them.
"""

from __future__ import annotations

import numpy as np

from ..cavity import ANG_TO_BOHR
from ..surface import build_conductor_surface
from .parameters import Parameterization
from .potential import sigma_potential_segments
from .sigma import segment_descriptors

__all__ = ["solvent_sigma_potential"]


def _check_conductor(result) -> None:
    screening = getattr(result, "screening", None)
    if screening is None or not getattr(screening, "is_conductor", False):
        eps = getattr(screening, "epsilon", getattr(result, "epsilon", "?"))
        raise ValueError(
            "solvent_sigma_potential: the solvent run must be in the conductor "
            f"limit (epsilon=inf); this one has epsilon={eps}. vibe-qc stores "
            "the *screened* charges q = f q* at finite dielectric, so a sigma "
            "profile built from them is scaled by f and nothing downstream can "
            "detect it: the profile looks entirely reasonable. Re-run the "
            "solvent with SolventModel(epsilon=math.inf, variant='cosmo')."
        )


def _check_radii(result, params: Parameterization) -> None:
    cavity = result.cavity
    z = np.asarray(cavity.atom_numbers, dtype=int)
    used = np.asarray(cavity.atom_radii, dtype=np.float64) / ANG_TO_BOHR
    wanted = np.array(
        [params.cavity_radii.get(int(zi), np.nan) for zi in z], dtype=np.float64
    )
    missing = np.isnan(wanted)
    if np.any(missing):
        raise ValueError(
            f"solvent_sigma_potential: {params.name} has no cavity radius for "
            f"element(s) {sorted({int(v) for v in z[missing]})}, so the surface "
            "cannot be built at the radii it was fitted with. Extending a "
            "parameterization to a new element is a fitting decision, not a "
            "default."
        )
    off = ~np.isclose(used, wanted, rtol=0.0, atol=1e-6)
    if np.any(off):
        first = int(np.flatnonzero(off)[0])
        raise ValueError(
            f"solvent_sigma_potential: the solvent cavity was built with radii "
            f"the parameterization was not fitted with (atom {first}, Z="
            f"{int(z[first])}: {used[first]:.4f} A used against "
            f"{wanted[first]:.4f} A fitted). sigma = q/a, so different radii "
            f"mean different sigma for the same physics and the constants no "
            f"longer describe it. Pass radii={params.name}.cavity_radii with "
            f"radii_scale=1.0 and solvent_probe_radius_ang=0.0."
        )


def solvent_sigma_potential(
    conductor_result,
    params: Parameterization,
    *,
    method: str,
    basis: str | None = None,
    cavity_kind: str = "fine-cfc",
    notes: str | None = None,
    **potential_kwargs,
):
    """The sigma potential of a pure solvent, from its own COSMO run.

    Parameters
    ----------
    conductor_result
        A converged :class:`~vibeqc.solvation.driver.SolventResult` for the
        *solvent* molecule at ``epsilon = inf``, built on the fine cavity with
        the parameterization's own radii.
    params
        The COSMO-RS parameter set the potential is for.
    method, basis, cavity_kind, notes
        Recorded in the surface provenance, and ``method``/``basis`` become the
        potential's :attr:`protocol`.
    **potential_kwargs
        Forwarded to
        :func:`~vibeqc.solvation.cosmors.potential.sigma_potential_segments`
        (``temperature_k``, ``max_iter``, ``tol``, ``mixing``).

    Returns the potential for the pure component; a mixture needs
    ``sigma_potential_segments`` with several descriptor sets and their mole
    fractions, which this deliberately does not wrap -- picking mole fractions
    is the caller's.
    """
    _check_conductor(conductor_result)
    _check_radii(conductor_result, params)

    surface = build_conductor_surface(
        conductor_result, method=method, basis=basis, cavity_kind=cavity_kind,
        energy_conductor=float(conductor_result.energy), notes=notes,
    )
    descriptors = segment_descriptors(surface, params.r_av)
    potential = sigma_potential_segments(
        [descriptors], np.array([1.0]), params, **potential_kwargs
    )
    potential.protocol = surface.provenance.descriptor()
    return potential
