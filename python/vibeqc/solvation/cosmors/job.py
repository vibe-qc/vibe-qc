"""COSMO-RS thermodynamics as a ``run_job`` request (M0; #558).

``run_job(..., cosmors=CosmoRSOptions(sigma_potential=...))`` runs the solute
as an ideally screened conductor on the fine cavity at the parameterization's
radii, and then evaluates the library chain of this package on that one
surface:

    ConductorSurface -> segment_descriptors -> sigma_profile
      -> chemical_potential(solute in the solvent)       Klamt 1998 eq. 19/20
      -> gas_phase_chemical_potential                    eq. 21 (see #435)
      -> solvation_free_energy = mu_S - mu_gas

This module is the orchestration only: it adds no physics of its own. Its job
is to refuse, *before* the SCF starts, every combination whose number would be
silently wrong, and to report what the number was computed from. The design
and the maintainer's decisions behind it are in
``docs/design_cosmors_run_job.md``:

* the solute's method and basis must be the solvent potential's (Q3). The two
  sigma distributions are otherwise not comparable and nothing downstream
  can tell; a pair that agrees with itself but not with the
  parameterization's fitted protocol is recorded and printed, not refused;
* ``n_ring_atoms`` comes from the caller and defaults to 0 (Q4), or is
  perceived from covalent-radius bonds on request (``"auto"``,
  :mod:`vibeqc.solvation.cosmors.rings`), in which case the output says so;
* a solvent given by name (M1) is computed at the solute's own protocol, on
  the same fine cavity, from the geometry library of
  :mod:`vibeqc.solvation.cosmors.solvents`, and its surface is cached per
  user, so the protocol agreement of Q3 holds by construction;
* the reference states are Klamt's: ideal gas at 1 bar, and 1 mol/mol in the
  solvent. Converting to 1 mol/L is the caller's business, and doing it
  silently is how reference states get lost.

``CosmoRSOptions(density="direct")`` (M2a) runs the solute in Direct COSMO-RS
with the same solvent potential instead of in the bare conductor, and takes
the free energy from the self-consistent density: Klamt & Diedenhofen 2015
(doi:10.1021/jp511158y) eq. 9,

    dG_solv = E_DCRS - (E_gas + omega n_ring + eta_gas)

with E_DCRS their eq. 4, the conductor COSMO energy plus ``sum_t a_t
mu_S(sigma_t)`` plus the combinatorial term, at f_pol = 1. vibe-qc's Direct
COSMO-RS energy is that expression without the combinatorial term, which
does not depend on the density and is added here. What changes against the
conductor chain is one substitution: ``E_COSMO + restoring`` becomes the
self-consistent energy. So the gas reference keeps KLAMT_1998's form, its
per-element dispersion included, and is fed the conductor functional
``E_gas[D] + (1/2) q.V`` *at the Direct COSMO-RS density* as its "E_COSMO";
the restoring term is the self-consistent ``sum_t a_t mu_S(sigma_t)``. Each
field then reduces to the conductor chain's for a solvent with
``mu_S == 0``. Three departures from the 2015 paper are deliberate and
recorded in the design note: eta is KLAMT_1998's, not refitted for Direct
COSMO-RS (Klamt refitted his own by -0.2 kcal/mol and found the unrefitted
value costs 0.01 kcal/mol MUE); one sigma potential per solvent rather than
one per element, since KLAMT_1998 keeps dispersion in the gas reference; and
no 1 mol/L conversion, as for the conductor chain. The sigma potential is
evaluated on each segment's own ``q/a``, as Sinnecker 2006 eq. 17 and
Klamt 2015 eq. 4 write it, not on the r_av-averaged profile the conductor
chain integrates, so the two densities differ by that as well as by the
relaxation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .parameters import R_KCAL, Parameterization, T_ROOM_K, get_parameterization
from .potential import SigmaPotential
from .sigma import segment_descriptors, sigma_profile
from .thermo import (
    HARTREE_TO_KCAL,
    ChemicalPotential,
    chemical_potential,
    combinatorial_chemical_potential,
    gas_phase_chemical_potential,
    solvation_free_energy,
)

__all__ = ["CosmoRSOptions", "CosmoRSResult", "CosmoRSPlan", "plan_cosmors_job"]

#: The cavity kind the conductor surface of a fine-cavity run is recorded as,
#: the same token :func:`~vibeqc.solvation.cosmors.solvent_sigma_potential`
#: writes into a potential's protocol.
FINE_CAVITY_KIND = "fine-cfc"

#: Which density the thermodynamics are evaluated on: the ideal conductor's
#: (Klamt 1998, M0/M1) or the self-consistent Direct COSMO-RS one (Klamt &
#: Diedenhofen 2015, M2a).
DENSITIES = ("conductor", "direct")

#: The cavity grid spacing (angstrom) at and below which a Direct COSMO-RS
#: free energy is converged with the grid. The direct density evaluates the
#: potential on each segment's raw q/a, which depends on the segment size;
#: the conductor chain's r_av-averaged profile does not. Measured 2026-10-22
#: in a water potential (RHF/STO-3G, KLAMT_1998 radii), dG_solv on the direct
#: density at 0.40 / 0.35 / 0.30 / 0.25 / 0.20 A: benzene +0.69 / +0.93 /
#: +1.19 / +1.19 / +1.18, water -0.63 / -0.53 / -0.46 / -0.41 / -0.41
#: kcal/mol, against the conductor density's 0.01 kcal/mol over the same range.
DIRECT_CONVERGED_SPACING_ANG = 0.25

_HF_TOKENS = frozenset({"hf", "rhf", "uhf", "rohf"})
_KS_WITHOUT_FUNCTIONAL = frozenset({"rks", "uks", "roks", "dft", "ks"})


@dataclass
class CosmoRSOptions:
    """What ``run_job`` should compute COSMO-RS thermodynamics against.

    Exactly one of ``sigma_potential`` and ``solvent`` is given.

    Parameters
    ----------
    sigma_potential
        The pure solvent's sigma potential, from
        :func:`~vibeqc.solvation.cosmors.solvent_sigma_potential`. It fixes
        the solvent, the parameterization, the temperature and the protocol
        the solute has to match.
    solvent
        A solvent name from the library
        (:func:`~vibeqc.solvation.cosmors.solvents.available_solvents`). Its
        conductor surface is computed at the solute's own method and basis
        on the same fine cavity, once, and cached under the user's cache
        directory; the potential is built from it at ``temperature_k`` with
        ``parameter_set``.
    n_ring_atoms
        Ring atoms of the solute, for the ring correction of Klamt 1998
        eq. 21: an integer the caller counted, or ``"auto"`` to perceive them
        from covalent-radius bonds (:mod:`vibeqc.solvation.cosmors.rings`).
        The default is 0, and the output states the value used and whether it
        was perceived.
    solvent_area
        Surface area of one solvent molecule, angstrom^2, for the
        combinatorial term of eq. 19. ``None`` takes the area the potential
        recorded when it was built from a solvent run; a hand-built potential
        records none and needs it given. A library solvent's area is its
        surface's and cannot be overridden.
    temperature_k
        Temperature of the sigma potential built for a library solvent;
        default 298.15 K. A given ``sigma_potential`` carries its own
        temperature, which this must then equal or leave unset.
    parameter_set
        The COSMO-RS parameter set for a library solvent, by name or as a
        :class:`~vibeqc.solvation.cosmors.parameters.Parameterization`;
        default ``klamt1998``. A given ``sigma_potential`` carries its own.
    cache_dir
        Overrides the solvent-surface cache root
        (:func:`~vibeqc.solvation.cosmors.solvents.cosmors_cache_dir`).
    density
        ``"conductor"`` (default) evaluates the solute on its ideally screened
        conductor density (Klamt 1998). ``"direct"`` runs it in Direct
        COSMO-RS with the same sigma potential and evaluates the free energy
        on that self-consistent density (Klamt & Diedenhofen 2015 eq. 9).
    activity_coefficient
        Also report the solute's infinite-dilution activity coefficient in
        the solvent, ``ln gamma^inf = (mu_S^X - mu_X^X) / RT`` (Klamt 1998
        eq. 19 in both phases; pure-solute reference), from the same chain as
        ``dG_solv``. Conductor density only. COSMOSPACE (Klamt 2002) solves
        the same pair statistics; on the same segments its residual part is
        identical, and the tests use it as an independent oracle for this
        profile integral.
    """

    sigma_potential: Optional[SigmaPotential] = None
    n_ring_atoms: Any = 0
    solvent_area: Optional[float] = None
    solvent: Optional[str] = None
    temperature_k: Optional[float] = None
    parameter_set: Any = None
    cache_dir: Any = None
    density: str = "conductor"
    activity_coefficient: bool = False

    def __post_init__(self):
        density = str(self.density).strip().lower() if isinstance(self.density, str) else None
        if density not in DENSITIES:
            raise ValueError(
                f"CosmoRSOptions.density must be one of {DENSITIES}; got "
                f"{self.density!r}."
            )
        self.density = density
        if not isinstance(self.activity_coefficient, bool):
            raise TypeError(
                "CosmoRSOptions.activity_coefficient must be True or False; got "
                f"{self.activity_coefficient!r}."
            )
        if self.activity_coefficient and density == "direct":
            raise ValueError(
                "CosmoRSOptions: activity_coefficient=True is computed on the "
                "conductor density only. Its reference is the pure solute, "
                "whose own Direct COSMO-RS state is a different calculation; "
                "use density='conductor' for the activity coefficient."
            )
        pot = self.sigma_potential
        if (pot is None) == (self.solvent is None):
            raise ValueError(
                "CosmoRSOptions: give exactly one of sigma_potential= (a "
                "SigmaPotential from solvent_sigma_potential) and solvent= (a "
                "library solvent name)."
            )
        if pot is not None and not isinstance(pot, SigmaPotential):
            raise TypeError(
                "CosmoRSOptions.sigma_potential must be a SigmaPotential, from "
                "vibeqc.solvation.cosmors.solvent_sigma_potential; got "
                f"{type(pot).__name__}."
            )
        if self.solvent is not None and not isinstance(self.solvent, str):
            raise TypeError(
                "CosmoRSOptions.solvent must be a solvent name (str); got "
                f"{type(self.solvent).__name__}. A SigmaPotential goes in "
                "sigma_potential=."
            )

        n_ring = self.n_ring_atoms
        if isinstance(n_ring, str):
            if n_ring.strip().lower() != "auto":
                raise ValueError(
                    "CosmoRSOptions.n_ring_atoms must be an integer or 'auto'; "
                    f"got {n_ring!r}."
                )
            self.n_ring_atoms = "auto"
        else:
            if isinstance(n_ring, bool) or int(n_ring) != n_ring:
                raise TypeError(
                    f"CosmoRSOptions.n_ring_atoms must be an integer or 'auto'; "
                    f"got {n_ring!r}."
                )
            self.n_ring_atoms = int(n_ring)
            if self.n_ring_atoms < 0:
                raise ValueError(
                    f"CosmoRSOptions.n_ring_atoms must be >= 0; got {self.n_ring_atoms}."
                )

        if pot is not None:
            self._init_from_potential(pot)
        else:
            self._init_from_solvent_name()

    def _init_from_potential(self, pot) -> None:
        if self.parameter_set is not None:
            given = (
                self.parameter_set if isinstance(self.parameter_set, Parameterization)
                else get_parameterization(str(self.parameter_set))
            )
            if given != pot.params:
                raise ValueError(
                    "CosmoRSOptions: parameter_set= differs from the parameter "
                    f"set the sigma potential was built with ({pot.params.name}); "
                    "the potential fixes it. Rebuild the potential with the set "
                    "you want, or leave parameter_set unset."
                )
        if self.temperature_k is not None and not math.isclose(
            float(self.temperature_k), float(pot.temperature_k), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"CosmoRSOptions: temperature_k={self.temperature_k} differs "
                f"from the sigma potential's {pot.temperature_k} K; the potential "
                "fixes the temperature. Rebuild it at the temperature you want "
                "(solvent_sigma_potential(..., temperature_k=...)), or leave "
                "temperature_k unset."
            )
        self.temperature_k = float(pot.temperature_k)
        area = self.solvent_area
        if area is None:
            area = getattr(pot, "solvent_area", None)
        if area is None:
            raise ValueError(
                "CosmoRSOptions: the sigma potential records no solvent surface "
                "area, which the combinatorial term of the chemical potential "
                "needs (Klamt 1998 eq. 19). Build the potential with "
                "solvent_sigma_potential, which records it, or pass "
                "solvent_area= in angstrom^2."
            )
        area = float(area)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError(
                f"CosmoRSOptions.solvent_area must be a positive area in "
                f"angstrom^2; got {area!r}."
            )
        self.solvent_area = area

    def _init_from_solvent_name(self) -> None:
        from .solvents import _resolve_solvent_name

        self.solvent = _resolve_solvent_name(self.solvent)
        if self.solvent_area is not None:
            raise ValueError(
                "CosmoRSOptions: solvent_area= cannot be given with a library "
                f"solvent ({self.solvent!r}); its area is its computed surface's."
            )
        if self.parameter_set is None:
            self.parameter_set = get_parameterization(None)
        elif not isinstance(self.parameter_set, Parameterization):
            self.parameter_set = get_parameterization(str(self.parameter_set))
        if self.temperature_k is None:
            self.temperature_k = T_ROOM_K
        temperature = float(self.temperature_k)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                f"CosmoRSOptions.temperature_k must be a positive temperature in "
                f"kelvin; got {self.temperature_k!r}."
            )
        self.temperature_k = temperature

    @property
    def parameterization(self) -> Parameterization:
        if self.sigma_potential is not None:
            return self.sigma_potential.params
        return self.parameter_set


@dataclass(frozen=True)
class CosmoRSResult:
    """A COSMO-RS solvation free energy and what it was computed from.

    Energies are kcal/mol. ``dg_solv = mu_solvent.total - mu_gas`` is the free
    energy of transfer from the ideal gas at 1 bar to the solvent at 1 mol/mol
    (Klamt 1998 reference states).

    With ``density == "direct"`` the same fields hold the Direct COSMO-RS
    decomposition: ``mu_solvent.restoring`` is the self-consistent
    ``sum_t a_t mu_S(sigma_t)`` on raw segment sigma, and
    ``ideal_screening_energy`` is ``E_gas`` minus the conductor functional
    ``E_gas[D] + (1/2) q.V`` at the Direct COSMO-RS density ``D``, so that
    ``dg_solv`` is Klamt & Diedenhofen 2015 eq. 9 and every field equals the
    conductor chain's when ``mu_S == 0``.
    """

    dg_solv: float
    mu_solvent: ChemicalPotential
    mu_gas: float
    ideal_screening_energy: float  # E_gas - E_COSMO, kcal/mol
    n_ring_atoms: int
    temperature_k: float
    parameterization: str
    fitted_protocol: str
    solute_protocol: str
    solvent_protocol: str
    fraction_outside_grid: float
    solute_area: float
    solvent_area: float
    # Where the solvent potential came from: "given" (a SigmaPotential passed
    # by the caller), "cache" (a library solvent's surface read from the
    # per-user cache) or "computed" (computed now, and cached).
    solvent_source: str = "given"
    solvent_name: Optional[str] = None
    # Whether ``n_ring_atoms`` was perceived from covalent-radius bonds rather
    # than given by the caller.
    ring_atoms_perceived: bool = False
    # "conductor" or "direct": the density the free energy was evaluated on.
    density: str = "conductor"
    # Direct COSMO-RS only: E_DCRS - E_gas without the combinatorial term,
    # kcal/mol, and the macro-iterations the feedback took.
    direct_minus_gas: Optional[float] = None
    n_macro_iter: Optional[int] = None
    # CosmoRSOptions(activity_coefficient=True) only: ln gamma^inf of the
    # solute in the solvent and its residual and combinatorial parts
    # (dimensionless, pure-solute reference), and mu_X^X, kcal/mol.
    # The fine-cavity grid spacing the solute ran on, angstrom.
    fine_grid_spacing_ang: Optional[float] = None
    ln_gamma_inf: Optional[float] = None
    ln_gamma_inf_residual: Optional[float] = None
    ln_gamma_inf_combinatorial: Optional[float] = None
    mu_pure_solute: Optional[float] = None

    @property
    def protocol_matches_fit(self) -> bool:
        """Whether the solute's protocol is the parameterization's own.

        Almost never true for vibe-qc (DMol's numerical DNP basis has no
        Gaussian equivalent); reported so the transfer is visible where the
        number is read.
        """
        return _norm(self.solute_protocol) == _norm(self.fitted_protocol)


def _norm(token: str) -> str:
    return str(token or "").strip().lower()


def _method_token(token: str) -> str:
    """The level-of-theory part of a protocol, spelled for comparison.

    Hartree-Fock is one level whether restricted or not, so rhf/uhf/rohf
    compare equal. Density functionals are compared by the name given, which
    is what the user guide's recipe records (``method="bpw91"``).
    """
    t = _norm(token)
    return "hf" if t in _HF_TOKENS else t


def _split_protocol(protocol: str) -> tuple[str, str, str]:
    # "method/basis/cavity"; a functional may itself contain a slash
    # (b3lyp/g), a basis name does not.
    parts = str(protocol).rsplit("/", 2)
    if len(parts) != 3:
        raise ValueError(
            f"COSMO-RS: cannot read the solvent potential's protocol "
            f"{protocol!r}; expected 'method/basis/cavity' as "
            f"solvent_sigma_potential records it."
        )
    return parts[0], parts[1], parts[2]


@dataclass
class CosmoRSPlan:
    """A validated request: the solvent model to run, and how to finish.

    ``potential`` is the solvent's sigma potential the thermodynamics use,
    whether the caller gave it or the plan computed (or read back) a library
    solvent's; ``n_ring_atoms`` is the resolved count.
    """

    options: CosmoRSOptions
    solvent_model: Any
    solute_protocol: str
    method_token: str
    basis: str
    charge: int
    multiplicity: int
    potential: SigmaPotential
    solvent_area: float
    n_ring_atoms: int
    ring_atoms_perceived: bool = False
    solvent_name: Optional[str] = None
    solvent_source: str = "given"

    def compute(self, solvent_result) -> CosmoRSResult:
        """Evaluate the chain on the solute's converged conductor run."""
        from ..surface import build_conductor_surface
        from .solvent import _check_conductor, _check_radii

        opts = self.options
        pot = self.potential
        params = opts.parameterization
        # The same two refusals the solvent potential passed, now on the
        # solute's own run: the plan already checked the model, this checks
        # what actually ran.
        _check_conductor(solvent_result)
        _check_radii(solvent_result, params)
        if opts.density == "direct":
            return self._compute_direct(solvent_result)

        surface = build_conductor_surface(
            solvent_result,
            method=self.method_token,
            basis=self.basis,
            cavity_kind=FINE_CAVITY_KIND,
            energy_conductor=float(solvent_result.energy),
            charge=self.charge,
            multiplicity=self.multiplicity,
        )
        descriptors = segment_descriptors(surface, params.r_av)
        profile = sigma_profile(descriptors)
        temperature = float(pot.temperature_k)
        mu_s = chemical_potential(
            profile, pot, params,
            solvent_area=float(self.solvent_area), temperature_k=temperature,
        )
        mu_gas = gas_phase_chemical_potential(
            surface, descriptors, params,
            n_ring_atoms=self.n_ring_atoms, temperature_k=temperature,
        )
        sigma = np.asarray(descriptors.sigma, dtype=np.float64)
        areas = np.asarray(descriptors.areas, dtype=np.float64)
        grid = np.asarray(pot.sigma_grid, dtype=np.float64)
        outside = (sigma < grid[0]) | (sigma > grid[-1])
        total = float(areas.sum())
        activity = {}
        if opts.activity_coefficient:
            # The pure solute as its own solvent, built exactly as a solvent
            # potential is, so a solute in itself gives ln gamma = 0.
            from .solvents import solvent_sigma_potential_from_surface

            self_pot = solvent_sigma_potential_from_surface(
                surface, params, temperature_k=temperature
            )
            mu_xx = chemical_potential(
                profile, self_pot, params,
                solvent_area=float(self_pot.solvent_area), temperature_k=temperature,
            )
            rt = R_KCAL * temperature
            activity = dict(
                ln_gamma_inf=(mu_s.total - mu_xx.total) / rt,
                ln_gamma_inf_residual=(mu_s.restoring - mu_xx.restoring) / rt,
                ln_gamma_inf_combinatorial=(
                    mu_s.combinatorial - mu_xx.combinatorial
                ) / rt,
                mu_pure_solute=float(mu_xx.total),
            )
        return CosmoRSResult(
            dg_solv=solvation_free_energy(mu_s, mu_gas),
            mu_solvent=mu_s,
            mu_gas=float(mu_gas),
            ideal_screening_energy=float(surface.ideal_screening_energy) * HARTREE_TO_KCAL,
            n_ring_atoms=self.n_ring_atoms,
            temperature_k=temperature,
            parameterization=params.name,
            fitted_protocol=params.fitted_protocol,
            solute_protocol=self.solute_protocol,
            solvent_protocol=str(pot.protocol),
            fraction_outside_grid=float(areas[outside].sum() / total) if total > 0 else 0.0,
            solute_area=total,
            solvent_area=float(self.solvent_area),
            solvent_source=self.solvent_source,
            solvent_name=self.solvent_name,
            ring_atoms_perceived=self.ring_atoms_perceived,
            fine_grid_spacing_ang=float(self.solvent_model.fine_grid_spacing_ang),
            **activity,
        )

    def _compute_direct(self, solvent_result) -> CosmoRSResult:
        """Klamt & Diedenhofen 2015 eq. 9 on the Direct COSMO-RS run."""
        from ..surface import build_conductor_surface

        opts = self.options
        pot = self.potential
        params = opts.parameterization
        fb = getattr(solvent_result, "direct_feedback", None)
        if fb is None or getattr(solvent_result, "solvent_variant", None) != "dcosmo-rs":
            raise RuntimeError(
                "run_job(cosmors=CosmoRSOptions(density='direct')): the solvent "
                "run is not a Direct COSMO-RS run; the plan built one, so this "
                "is a defect in the job wiring."
            )
        e_total = float(solvent_result.energy)
        e_gas = float(solvent_result.e_gas)
        e_rs = float(fb.energy)                      # sum_t a_t mu_S, Hartree
        # The conductor functional E_gas[D] + (1/2) q.V at the Direct COSMO-RS
        # density: the "E_COSMO" slot of eq. 21, so that the gas reference is
        # the conductor chain's own function, fed one different number.
        e_conductor_at_d = e_total - e_rs
        surface = build_conductor_surface(
            solvent_result,
            method=self.method_token,
            basis=self.basis,
            cavity_kind=FINE_CAVITY_KIND,
            energy_conductor=e_conductor_at_d,
            charge=self.charge,
            multiplicity=self.multiplicity,
        )
        # Descriptors only for the per-element areas of the dispersion term;
        # the cavity, and so the areas, are the conductor run's.
        descriptors = segment_descriptors(surface, params.r_av)
        temperature = float(pot.temperature_k)
        restoring = e_rs * HARTREE_TO_KCAL
        combinatorial = combinatorial_chemical_potential(
            params, solvent_area=float(self.solvent_area), temperature_k=temperature
        )
        mu_s = ChemicalPotential(
            total=restoring + combinatorial,
            restoring=restoring,
            combinatorial=combinatorial,
            solute_area=descriptors.total_area,
            solvent_area=float(self.solvent_area),
            parameterization=params.name,
            temperature_k=temperature,
        )
        mu_gas = gas_phase_chemical_potential(
            surface, descriptors, params,
            ideal_screening_energy_hartree=e_gas - e_conductor_at_d,
            n_ring_atoms=self.n_ring_atoms, temperature_k=temperature,
        )
        return CosmoRSResult(
            dg_solv=solvation_free_energy(mu_s, mu_gas),
            mu_solvent=mu_s,
            mu_gas=float(mu_gas),
            ideal_screening_energy=(e_gas - e_conductor_at_d) * HARTREE_TO_KCAL,
            n_ring_atoms=self.n_ring_atoms,
            temperature_k=temperature,
            parameterization=params.name,
            fitted_protocol=params.fitted_protocol,
            solute_protocol=self.solute_protocol,
            solvent_protocol=str(pot.protocol),
            # On the raw segment sigma the potential was actually evaluated at.
            fraction_outside_grid=float(fb.fraction_outside_grid),
            solute_area=descriptors.total_area,
            solvent_area=float(self.solvent_area),
            solvent_source=self.solvent_source,
            solvent_name=self.solvent_name,
            ring_atoms_perceived=self.ring_atoms_perceived,
            density="direct",
            fine_grid_spacing_ang=float(self.solvent_model.fine_grid_spacing_ang),
            direct_minus_gas=(e_total - e_gas) * HARTREE_TO_KCAL,
            n_macro_iter=int(getattr(solvent_result, "n_macro_iter", 0)),
        )


def direct_solvent_model(conductor_model, potential):
    """The Direct COSMO-RS model on exactly the conductor model's cavity.

    ``dataclasses.replace`` re-runs ``SolventModel.__post_init__``, so every
    Direct COSMO-RS refusal (ensemble, cavity, epsilon) still applies.
    """
    import dataclasses

    return dataclasses.replace(
        conductor_model,
        variant="dcosmo-rs",
        sigma_potential=potential,
        name=f"Direct COSMO-RS ({potential.params.name})",
    )


def conductor_solvent_model(params: Parameterization):
    """The conductor run COSMO-RS needs: epsilon = inf on the fine cavity at
    the parameterization's own radii, unscaled, without a probe."""
    from ..driver import SolventModel

    return SolventModel(
        epsilon=math.inf,
        name=f"COSMO-RS conductor ({params.name})",
        variant="cosmo",
        cavity="fine",
        radii=dict(params.cavity_radii),
        radii_scale=1.0,
        solvent_probe_radius_ang=0.0,
    )


def _check_solvent_model(sm, params: Parameterization, elements) -> None:
    problems = []
    if not math.isinf(float(sm.epsilon)):
        problems.append(
            f"epsilon={sm.epsilon} (needs epsilon=inf: the sigma profile is "
            "built from the ideal conductor charges)"
        )
    if sm.variant == "dcosmo-rs":
        problems.append(
            "variant='dcosmo-rs' (pass the conductor model, or none, and ask "
            "for the Direct COSMO-RS density with CosmoRSOptions("
            "density='direct'): the job builds that model from the options' "
            "own sigma potential)"
        )
    if sm.cavity != "fine":
        problems.append(f"cavity={sm.cavity!r} (needs cavity='fine')")
    if abs(float(sm.radii_scale) - 1.0) > 1e-12:
        problems.append(f"radii_scale={sm.radii_scale} (needs 1.0)")
    if abs(float(sm.solvent_probe_radius_ang)) > 1e-12:
        problems.append(
            f"solvent_probe_radius_ang={sm.solvent_probe_radius_ang} (needs 0.0)"
        )
    radii = sm.radii or {}
    off = sorted(
        z for z in elements
        if z not in radii or abs(float(radii[z]) - float(params.cavity_radii[z])) > 1e-6
    )
    if off:
        problems.append(
            f"radii for Z={off} are not {params.name}'s own "
            f"({ {z: params.cavity_radii[z] for z in off} } angstrom)"
        )
    if problems:
        raise ValueError(
            "run_job(cosmors=...): the solvent model cannot produce the "
            "surface COSMO-RS is parameterized on: " + "; ".join(problems)
            + ". Omit solvent= and run_job builds the right conductor model, "
            "or pass one with epsilon=math.inf, variant='cosmo', cavity='fine', "
            f"radii=dict({params.name}.cavity_radii), radii_scale=1.0, "
            "solvent_probe_radius_ang=0.0."
        )


def plan_cosmors_job(
    options,
    *,
    molecule,
    method: str,
    resolved_method: str,
    functional: Optional[str],
    basis: Optional[str],
    solvent,
    optimize: bool,
) -> CosmoRSPlan:
    """Validate a ``run_job(cosmors=...)`` request before any SCF runs.

    Returns the plan whose ``solvent_model`` the job must run with. Raises
    ``NotImplementedError`` for a combination M0 does not cover and
    ``ValueError`` for inputs whose number would be wrong.
    """
    if not isinstance(options, CosmoRSOptions):
        raise TypeError(
            "run_job(cosmors=...) takes a vibeqc.CosmoRSOptions; got "
            f"{type(options).__name__}."
        )
    params = options.parameterization
    if resolved_method not in ("rhf", "uhf", "rks", "uks") or method not in (
        "rhf", "uhf", "rks", "uks", "auto",
    ):
        raise NotImplementedError(
            "run_job(cosmors=...) computes COSMO-RS thermodynamics on a "
            "mean-field conductor surface: method must be rhf, uhf, rks or "
            f"uks (got method={method!r}, resolved to {resolved_method!r})."
        )
    if optimize:
        raise NotImplementedError(
            "run_job(cosmors=...) with a geometry optimization is not "
            "implemented: optimizing on a COSMO-RS surface needs its nuclear "
            "gradient, which does not exist yet. Optimize first, then run the "
            "COSMO-RS single point at the optimized geometry."
        )
    if not isinstance(basis, str) or not basis.strip():
        raise ValueError(
            "run_job(cosmors=...) needs a named basis set, which it compares "
            "against the solvent potential's protocol."
        )

    elements = sorted({int(a.Z) for a in molecule.atoms})
    no_radius = [z for z in elements if z not in params.cavity_radii]
    ok, no_dispersion = params.supports_elements(elements)
    if no_radius or not ok:
        raise ValueError(
            f"run_job(cosmors=...): {params.name} was not fitted for "
            f"element(s) {sorted(set(no_radius) | set(no_dispersion))} (it "
            f"covers {sorted(params.cavity_radii)}). Extending a "
            "parameterization to a new element is a fitting decision, not a "
            "default."
        )

    if resolved_method in ("rks", "uks"):
        method_token = str(functional or "").strip().lower()
        if not method_token:
            raise ValueError(
                "run_job(cosmors=...): a KS solute needs a named functional, "
                "which the protocol records."
            )
    else:
        method_token = resolved_method

    if solvent is None:
        solvent_model = conductor_solvent_model(params)
    else:
        from ..driver import resolve_solvent

        solvent_model = resolve_solvent(solvent)
        if solvent_model is None or getattr(solvent_model, "is_gas_phase", False):
            raise ValueError(
                "run_job(cosmors=...): solvent= resolves to the gas phase. "
                "Omit it and run_job builds the conductor model COSMO-RS needs."
            )
        _check_solvent_model(solvent_model, params, elements)

    solvent_name = None
    solvent_source = "given"
    if options.solvent is not None:
        # A library solvent: its surface at the solute's protocol, on the same
        # cavity settings the solute will run with, from the cache when it is
        # there. This is the one calculation the chain needs besides the
        # solute's, and it runs before the solute's SCF so that a solvent the
        # library cannot provide fails before any work.
        from .solvents import solvent_sigma_potential_from_surface, solvent_surface

        solvent_name = options.solvent
        surface, solvent_source = solvent_surface(
            solvent_name,
            method=method_token,
            basis=basis,
            params=params,
            fine_grid_spacing_ang=solvent_model.fine_grid_spacing_ang,
            fine_segments_h=solvent_model.fine_segments_h,
            fine_segments_other=solvent_model.fine_segments_other,
            fine_frame=solvent_model.fine_frame,
            cache_dir=options.cache_dir,
        )
        pot = solvent_sigma_potential_from_surface(
            surface, params, temperature_k=float(options.temperature_k)
        )
        solvent_area = float(pot.solvent_area)
    else:
        pot = options.sigma_potential
        solvent_area = float(options.solvent_area)
    if not pot.protocol:
        raise ValueError(
            "run_job(cosmors=...): the solvent potential records no protocol, "
            "so the solute cannot be checked against it. Build it from a "
            "solvent run with solvent_sigma_potential(..., method=..., "
            "basis=...)."
        )
    pot_method, pot_basis, pot_cavity = _split_protocol(pot.protocol)
    if _norm(pot_method) in _KS_WITHOUT_FUNCTIONAL:
        raise ValueError(
            f"run_job(cosmors=...): the solvent potential's protocol "
            f"{pot.protocol!r} names no functional, so a DFT solute cannot be "
            "checked against it. Rebuild it with solvent_sigma_potential("
            "..., method='<functional name>')."
        )
    mismatch = []
    if _method_token(pot_method) != _method_token(method_token):
        mismatch.append(f"method {method_token!r} against {pot_method!r}")
    if _norm(pot_basis) != _norm(basis):
        mismatch.append(f"basis {basis!r} against {pot_basis!r}")
    if _norm(pot_cavity) != FINE_CAVITY_KIND:
        mismatch.append(f"cavity {FINE_CAVITY_KIND!r} against {pot_cavity!r}")
    if mismatch:
        raise ValueError(
            "run_job(cosmors=...): the solute would be computed at a different "
            "protocol from the solvent potential (" + "; ".join(mismatch)
            + f"; potential: {pot.protocol!r}). Two sigma distributions from "
            "different levels of theory are not comparable, and nothing "
            "downstream can detect it. Run the solute at the solvent's level, "
            "or rebuild the potential at the solute's. Functionals are "
            "compared by name: 'bpw91' and '106,134' are the same functional "
            "but not the same name."
        )

    if options.n_ring_atoms == "auto":
        from .rings import ring_atom_count

        n_ring_atoms = int(ring_atom_count(molecule))
        ring_atoms_perceived = True
    else:
        n_ring_atoms = int(options.n_ring_atoms)
        ring_atoms_perceived = False

    if options.density == "direct":
        if not getattr(pot, "has_ensemble", False):
            raise ValueError(
                "run_job(cosmors=CosmoRSOptions(density='direct')): the sigma "
                "potential does not carry its solvent ensemble, which Direct "
                "COSMO-RS evaluates and differentiates. Build it with "
                "solvent_sigma_potential, or give the solvent by name."
            )
        solvent_model = direct_solvent_model(solvent_model, pot)

    solute_protocol = f"{method_token}/{basis.strip().lower()}/{FINE_CAVITY_KIND}"
    # Every refusal has passed: the route will run, so it says what it is.
    from .experimental import _warn_experimental

    _warn_experimental("run_job(cosmors=...) (COSMO-RS thermodynamics)")
    return CosmoRSPlan(
        options=options,
        solvent_model=solvent_model,
        solute_protocol=solute_protocol,
        method_token=method_token,
        basis=basis.strip().lower(),
        charge=int(molecule.charge),
        multiplicity=int(molecule.multiplicity),
        potential=pot,
        solvent_area=solvent_area,
        n_ring_atoms=n_ring_atoms,
        ring_atoms_perceived=ring_atoms_perceived,
        solvent_name=solvent_name,
        solvent_source=solvent_source,
    )


def _solvent_source_line(res: CosmoRSResult) -> str:
    name = res.solvent_name or "(sigma potential given by the caller)"
    if res.solvent_source == "cache":
        return f"{name}, surface from the per-user cache"
    if res.solvent_source == "computed":
        return f"{name}, surface computed now and cached"
    return name


def render_cosmors_block(res: CosmoRSResult) -> str:
    """The ``## COSMO-RS`` block of the ``.out`` (design note Q5)."""
    mu = res.mu_solvent
    direct = res.density == "direct"
    lines = [
        "  Density                   = "
        + (
            "Direct COSMO-RS, self-consistent (Klamt & Diedenhofen 2015 eq. 9)"
            if direct else "ideal conductor (Klamt 1998)"
        ),
        f"  Parameterization          = {res.parameterization} "
        f"(fitted on {res.fitted_protocol})",
        f"  Solute surface            = {res.solute_protocol}  "
        f"(area {res.solute_area:.2f} A^2)",
        f"  Solvent potential         = {res.solvent_protocol}  "
        f"(solvent area {res.solvent_area:.2f} A^2)",
        "  Solvent                   = " + _solvent_source_line(res),
        f"  Temperature               = {res.temperature_k:.2f} K",
        "  Reference states          = ideal gas at 1 bar -> 1 mol/mol in solvent",
        (
            f"  E_gas - E_COSMO[D_DCRS]   = {res.ideal_screening_energy:12.4f} kcal/mol"
            if direct else
            f"  E_gas - E_COSMO           = {res.ideal_screening_energy:12.4f} kcal/mol"
        ),
        f"  mu_S (solute in solvent)  = {mu.total:12.4f} kcal/mol  "
        f"(restoring {mu.restoring:.4f}, combinatorial {mu.combinatorial:.4f})",
        f"  mu_gas (gas reference)    = {res.mu_gas:12.4f} kcal/mol  "
        + (
            f"(ring atoms: {res.n_ring_atoms}, perceived from covalent-radius bonds)"
            if res.ring_atoms_perceived
            else f"(ring atoms given: {res.n_ring_atoms})"
        ),
        f"  dG_solv = mu_S - mu_gas   = {res.dg_solv:12.4f} kcal/mol",
    ]
    if direct:
        lines += [
            f"  E_DCRS - E_gas            = {res.direct_minus_gas:12.4f} kcal/mol  "
            f"(without the combinatorial term; {res.n_macro_iter} macro-iterations)",
            "  eta                       = KLAMT_1998's, not refitted for "
            "Direct COSMO-RS",
        ]
    if res.ln_gamma_inf is not None:
        lines += [
            f"  mu_X (solute in itself)   = {res.mu_pure_solute:12.4f} kcal/mol",
            f"  ln gamma_inf (in solvent) = {res.ln_gamma_inf:12.4f}  "
            f"(residual {res.ln_gamma_inf_residual:.4f}, combinatorial "
            f"{res.ln_gamma_inf_combinatorial:.4f}; reference: pure solute)",
        ]
    lines += [
        f"  Solute area beyond the fitted sigma grid = "
        f"{100.0 * res.fraction_outside_grid:.1f} %",
    ]
    if not res.protocol_matches_fit:
        lines.append(
            "  Note: the surfaces are not at the parameterization's fitted "
            "protocol; the constants are transferred, not refitted."
        )
    if (
        direct
        and res.fine_grid_spacing_ang is not None
        and res.fine_grid_spacing_ang > DIRECT_CONVERGED_SPACING_ANG + 1e-12
    ):
        lines.append(
            f"  Note: the Direct COSMO-RS density evaluates the sigma potential "
            f"on raw segment sigma, which converges with the cavity grid only "
            f"near {DIRECT_CONVERGED_SPACING_ANG:.2f} A; this run used "
            f"{res.fine_grid_spacing_ang:.2f} A. Pass a SolventModel with "
            f"fine_grid_spacing_ang={DIRECT_CONVERGED_SPACING_ANG:.2f} for a "
            "converged number."
        )
    if res.fraction_outside_grid > 0.005:
        lines.append(
            "  Note: part of the solute surface lies beyond the sigma range "
            "the parameterization covers; there the sigma potential is "
            "extrapolated."
        )
    return "\n".join(lines) + "\n"
