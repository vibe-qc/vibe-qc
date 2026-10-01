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
* ``n_ring_atoms`` comes from the caller and defaults to 0 (Q4); vibe-qc has
  no bond perception to count ring atoms from;
* the reference states are Klamt's: ideal gas at 1 bar, and 1 mol/mol in the
  solvent. Converting to 1 mol/L is the caller's business, and doing it
  silently is how reference states get lost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .parameters import Parameterization
from .potential import SigmaPotential
from .sigma import segment_descriptors, sigma_profile
from .thermo import (
    HARTREE_TO_KCAL,
    ChemicalPotential,
    chemical_potential,
    gas_phase_chemical_potential,
    solvation_free_energy,
)

__all__ = ["CosmoRSOptions", "CosmoRSResult", "CosmoRSPlan", "plan_cosmors_job"]

#: The cavity kind the conductor surface of a fine-cavity run is recorded as,
#: the same token :func:`~vibeqc.solvation.cosmors.solvent_sigma_potential`
#: writes into a potential's protocol.
FINE_CAVITY_KIND = "fine-cfc"

_HF_TOKENS = frozenset({"hf", "rhf", "uhf", "rohf"})
_KS_WITHOUT_FUNCTIONAL = frozenset({"rks", "uks", "roks", "dft", "ks"})


@dataclass
class CosmoRSOptions:
    """What ``run_job`` should compute COSMO-RS thermodynamics against.

    Parameters
    ----------
    sigma_potential
        The pure solvent's sigma potential, from
        :func:`~vibeqc.solvation.cosmors.solvent_sigma_potential`. It fixes
        the solvent, the parameterization, the temperature and the protocol
        the solute has to match.
    n_ring_atoms
        Ring atoms of the solute, for the ring correction of Klamt 1998
        eq. 21. The caller counts them: vibe-qc does not perceive rings, and
        the output states the value used.
    solvent_area
        Surface area of one solvent molecule, angstrom^2, for the
        combinatorial term of eq. 19. ``None`` takes the area the potential
        recorded when it was built from a solvent run; a hand-built potential
        records none and needs it given.
    """

    sigma_potential: SigmaPotential
    n_ring_atoms: int = 0
    solvent_area: Optional[float] = None

    def __post_init__(self):
        pot = self.sigma_potential
        if not isinstance(pot, SigmaPotential):
            raise TypeError(
                "CosmoRSOptions.sigma_potential must be a SigmaPotential, from "
                "vibeqc.solvation.cosmors.solvent_sigma_potential; got "
                f"{type(pot).__name__}."
            )
        if isinstance(self.n_ring_atoms, bool) or int(self.n_ring_atoms) != self.n_ring_atoms:
            raise TypeError(
                f"CosmoRSOptions.n_ring_atoms must be an integer; got {self.n_ring_atoms!r}."
            )
        self.n_ring_atoms = int(self.n_ring_atoms)
        if self.n_ring_atoms < 0:
            raise ValueError(
                f"CosmoRSOptions.n_ring_atoms must be >= 0; got {self.n_ring_atoms}."
            )
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

    @property
    def parameterization(self) -> Parameterization:
        return self.sigma_potential.params


@dataclass(frozen=True)
class CosmoRSResult:
    """A COSMO-RS solvation free energy and what it was computed from.

    Energies are kcal/mol. ``dg_solv = mu_solvent.total - mu_gas`` is the free
    energy of transfer from the ideal gas at 1 bar to the solvent at 1 mol/mol
    (Klamt 1998 reference states).
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
    """A validated request: the solvent model to run, and how to finish."""

    options: CosmoRSOptions
    solvent_model: Any
    solute_protocol: str
    method_token: str
    basis: str
    charge: int
    multiplicity: int

    def compute(self, solvent_result) -> CosmoRSResult:
        """Evaluate the chain on the solute's converged conductor run."""
        from ..surface import build_conductor_surface
        from .solvent import _check_conductor, _check_radii

        opts = self.options
        pot = opts.sigma_potential
        params = opts.parameterization
        # The same two refusals the solvent potential passed, now on the
        # solute's own run: the plan already checked the model, this checks
        # what actually ran.
        _check_conductor(solvent_result)
        _check_radii(solvent_result, params)

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
            solvent_area=float(opts.solvent_area), temperature_k=temperature,
        )
        mu_gas = gas_phase_chemical_potential(
            surface, descriptors, params,
            n_ring_atoms=opts.n_ring_atoms, temperature_k=temperature,
        )
        sigma = np.asarray(descriptors.sigma, dtype=np.float64)
        areas = np.asarray(descriptors.areas, dtype=np.float64)
        grid = np.asarray(pot.sigma_grid, dtype=np.float64)
        outside = (sigma < grid[0]) | (sigma > grid[-1])
        total = float(areas.sum())
        return CosmoRSResult(
            dg_solv=solvation_free_energy(mu_s, mu_gas),
            mu_solvent=mu_s,
            mu_gas=float(mu_gas),
            ideal_screening_energy=float(surface.ideal_screening_energy) * HARTREE_TO_KCAL,
            n_ring_atoms=opts.n_ring_atoms,
            temperature_k=temperature,
            parameterization=params.name,
            fitted_protocol=params.fitted_protocol,
            solute_protocol=self.solute_protocol,
            solvent_protocol=str(pot.protocol),
            fraction_outside_grid=float(areas[outside].sum() / total) if total > 0 else 0.0,
            solute_area=total,
            solvent_area=float(opts.solvent_area),
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
            "variant='dcosmo-rs' (COSMO-RS thermodynamics on the Direct "
            "COSMO-RS density is milestone M2, not implemented)"
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

    pot = options.sigma_potential
    if not pot.protocol:
        raise ValueError(
            "run_job(cosmors=...): the solvent potential records no protocol, "
            "so the solute cannot be checked against it. Build it from a "
            "solvent run with solvent_sigma_potential(..., method=..., "
            "basis=...)."
        )
    if resolved_method in ("rks", "uks"):
        method_token = str(functional or "").strip().lower()
    else:
        method_token = resolved_method
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

    solute_protocol = f"{method_token}/{basis.strip().lower()}/{FINE_CAVITY_KIND}"
    return CosmoRSPlan(
        options=options,
        solvent_model=solvent_model,
        solute_protocol=solute_protocol,
        method_token=method_token,
        basis=basis.strip().lower(),
        charge=int(molecule.charge),
        multiplicity=int(molecule.multiplicity),
    )


def render_cosmors_block(res: CosmoRSResult) -> str:
    """The ``## COSMO-RS`` block of the ``.out`` (design note Q5)."""
    mu = res.mu_solvent
    lines = [
        f"  Parameterization          = {res.parameterization} "
        f"(fitted on {res.fitted_protocol})",
        f"  Solute surface            = {res.solute_protocol}  "
        f"(area {res.solute_area:.2f} A^2)",
        f"  Solvent potential         = {res.solvent_protocol}  "
        f"(solvent area {res.solvent_area:.2f} A^2)",
        f"  Temperature               = {res.temperature_k:.2f} K",
        "  Reference states          = ideal gas at 1 bar -> 1 mol/mol in solvent",
        f"  E_gas - E_COSMO           = {res.ideal_screening_energy:12.4f} kcal/mol",
        f"  mu_S (solute in solvent)  = {mu.total:12.4f} kcal/mol  "
        f"(restoring {mu.restoring:.4f}, combinatorial {mu.combinatorial:.4f})",
        f"  mu_gas (gas reference)    = {res.mu_gas:12.4f} kcal/mol  "
        f"(ring atoms given: {res.n_ring_atoms})",
        f"  dG_solv = mu_S - mu_gas   = {res.dg_solv:12.4f} kcal/mol",
        f"  Solute area beyond the fitted sigma grid = "
        f"{100.0 * res.fraction_outside_grid:.1f} %",
    ]
    if not res.protocol_matches_fit:
        lines.append(
            "  Note: the surfaces are not at the parameterization's fitted "
            "protocol; the constants are transferred, not refitted."
        )
    if res.fraction_outside_grid > 0.005:
        lines.append(
            "  Note: part of the solute surface lies beyond the sigma range "
            "the parameterization covers; there the sigma potential is "
            "extrapolated."
        )
    return "\n".join(lines) + "\n"
