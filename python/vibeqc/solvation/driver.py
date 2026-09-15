"""CPCM SCF driver -- macro-iteration coupling of density and ASC (S1b, S1d).

Wraps the molecular SCF entry points (RHF / UHF / RKS / UKS) with a
self-consistent CPCM apparent-surface-charge loop:

* Macro-iter 0: solve the gas-phase SCF for ``D_0``.
* Macro-iter k >= 1:
    1. Evaluate the molecular electrostatic potential V_i (from
       ``D_{k-1}`` + nuclei) at every cavity tessellation point.
    2. Solve ``A q = -f(e) V`` for the surface charge.
    3. Build the one-electron operator V_q^{muν} =
       S_i q_i <mu|1/|r - s_i||ν> and add it to ``H_core``.
    4. Re-run the inner SCF with the modified ``H_core``; obtain
       ``D_k``.
* Outer convergence on |E_solv(k) - E_solv(k-1)| <= ``tol_e_solv``.

The macro-iteration is the canonical CPCM coupling pattern (see
PySCF's ``solvent.PCM``, ORCA's ``CPCM`` keyword, and Cossi-
Scalmani-Barone 2003 Sec. II.B). Three to five outer cycles typically
converge ΔE_solv to 1e-6 Hartree for neutral closed-shell solutes;
charged or strongly-polar solutes may want 8-10.

This driver builds ``H_core`` from Python via the bundled molecular
integrals and uses the ``run_*_scf_with_jk`` low-level entry points
so the inner SCF -- DIIS / level shift / linear-dependence projection
/ canonical orthogonalisation -- runs unchanged. ``CavityTessellation``
is geometry-tied and recomputed once per ``run_cpcm_scf`` call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np

from .cavity import CavityTessellation, atom_radii_bohr, build_cavity
from .fine_cavity import (
    GRID_SPACING_ANG as FINE_GRID_SPACING_ANG,
    NSPA_FINE,
    NSPH_FINE,
    FineCavity,
    build_fine_cavity,
)
from .cpcm import (
    CPCMResult,
    build_cavity_A_matrix,
)
from .presets import SOLVENT_PRESETS, canonicalise_solvent_name
from .engine import ReactionField, lu_cavity_solve, reaction_field_step
from .screening import ScreeningModel


def _build_cavity_for(sm: "SolventModel", pos: np.ndarray, Zs: np.ndarray):
    """The cavity ``sm`` asks for. One call site for both constructions.

    Both return objects with the same field names (``points``, ``weights``,
    ``point_atom``, ``atom_positions``, ``atom_radii``, ``switching``), so
    everything downstream -- the A matrix, the provider, the reaction-field
    step, the surface record -- is construction-agnostic.
    """
    if sm.cavity == "fine":
        return build_fine_cavity(
            pos,
            Zs,
            atom_radii_bohr(
                Zs, sm.radii, sm.radii_scale, sm.solvent_probe_radius_ang
            ),
            grid_spacing_ang=sm.fine_grid_spacing_ang,
            n_segments_h=sm.fine_segments_h,
            n_segments_other=sm.fine_segments_other,
            frame=sm.fine_frame,
        )
    return build_cavity(
        atom_positions_bohr=pos,
        atom_numbers=Zs,
        radii=sm.radii,
        radii_scale=sm.radii_scale,
        solvent_probe_radius_ang=sm.solvent_probe_radius_ang,
        n_points_per_sphere=sm.n_points_per_sphere,
        switching_sigma_bohr=sm.switching_sigma_bohr,
        drop_threshold=sm.switching_drop_threshold,
    )


@dataclass
class SolventModel:
    """Top-level user-facing solvation spec.

    Parameters
    ----------
    epsilon : float
        Static dielectric constant of the solvent (dimensionless,
        e_rel). Must be > 1; ``epsilon <= 1`` is treated as a gas-phase
        no-op by :func:`run_cpcm_scf`.
    name : str, optional
        Human-readable name for logging (e.g. ``"water"``). Defaults
        to ``"custom (e = ...)"`` if omitted.
    variant : {"cpcm", "cosmo"}, default "cpcm"
        Dielectric factor formula. ``"cpcm"`` uses
        ``f = (e - 1)/e``; ``"cosmo"`` uses ``f = (e - 1)/(e + 0.5)``.
        See :func:`vibeqc.solvation.cpcm.dielectric_factor`.
    radii : dict[int, float], optional
        Per-element vdW-radii overrides (Å). Falls back to the bundled
        Bondi table; see :data:`vibeqc.solvation.cavity.BONDI_RADII_ANG`.
    radii_scale : float, default 1.20
        Multiplier on each Bondi radius (PCM / GEPOL convention).
    solvent_probe_radius_ang : float, default 0.0
        Adds a probe-radius offset to each atomic sphere. ``0.0``
        builds the scaled-vdW (SES) cavity; set to e.g. 1.385 Å for
        the water-probe SAS.
    n_points_per_sphere : int, default 302
        Lebedev order per atomic sphere. See :func:`build_cavity`.
    fine_frame : {"molecular", "lab"}, default "molecular"
        For ``cavity="fine"``, the coordinate system the CFC workflow is
        carried out in. The default performs the 2018 workflow's step 1, so the
        cavity rotates rigidly with the solute rather than depending on its
        orientation (#769). ``"lab"`` is the pre-#769 behaviour and the right
        choice for a linear solute. See
        :func:`vibeqc.solvation.fine_cavity.molecular_frame`.
    switching_sigma_bohr : float, default 0.5
        Scalmani-Frisch switching width (bohr).
    switching_drop_threshold : float, default 1e-8
        Points whose switched weight falls below this fraction of their raw
        Lebedev weight are dropped from the discrete cavity.

        Exposed because it is the documented remedy for a discontinuity that
        was otherwise unreachable from here. The switching function is smooth
        across the cutoff but the *point set* is not, so a point crossing it
        steps the energy. At the default width the cutoff bites nothing --
        832 points, and the energy's second difference is constant to five
        digits -- but a wider switch pushes many more points near it: at
        ``switching_sigma_bohr = 0.8`` a 2.5e-04 bohr displacement moves the
        count from 904 to 905 and the local slope reads -0.027 against a trend
        of +0.016. Set this to 0.0 to keep every point, which costs zeros in
        the ``A`` matrix and buys a smooth energy. A wide switch is exactly
        what ``switching_sigma_bohr`` exists for (close atom pairs), so the two
        belong together.
    max_macro_iter : int, default 30
        Outer CPCM macro-iteration cap. Three to five iters is typical;
        the cap protects against pathological non-convergence.
    tol_e_solv : float, default 1e-6
        Outer convergence threshold on |ΔE_solv| (Hartree).
    """

    epsilon: float
    name: str = ""
    variant: str = "cpcm"
    radii: Optional[dict[int, float]] = None
    radii_scale: float = 1.20
    solvent_probe_radius_ang: float = 0.0
    n_points_per_sphere: int = 302
    switching_sigma_bohr: float = 0.5
    switching_drop_threshold: float = 1e-8
    # Cavity construction. "lebedev" (default) is the switched union of
    # Lebedev-paved atomic spheres; "fine" is the Klamt & Diedenhofen 2018
    # COSMO FINE Cavity, a pseudo-density iso-surface that also paves the
    # concave regions between atoms (vibeqc.solvation.fine_cavity).
    #
    # The default is unchanged deliberately: the two constructions give
    # different segment sets, so switching would move every solvated energy.
    cavity: str = "lebedev"
    fine_grid_spacing_ang: float = FINE_GRID_SPACING_ANG
    fine_segments_h: int = NSPH_FINE
    fine_segments_other: int = NSPA_FINE
    # Which coordinate system the CFC workflow runs in: "molecular" (2018
    # workflow step 1; see fine_cavity.molecular_frame) or "lab". The default
    # makes the cavity -- and so the solvated energy, the gradient and the sigma
    # profile -- rotate rigidly with the solute instead of depending on its
    # orientation (#769). "lab" is the pre-#769 behaviour, and is the right
    # choice for a linear solute, whose molecule-fixed frame has no nuclear
    # derivative.
    fine_frame: str = "molecular"
    max_macro_iter: int = 30
    tol_e_solv: float = 1e-6
    q_diis: bool = False
    # When True, use DIIS extrapolation on the apparent surface charges
    # q between macro-iterations to accelerate CPCM convergence.  Adds
    # ~1 ms per macro-iteration for the DIIS solve; useful for charged
    # solutes or systems that need > 8 macro-iterations.  Default False.

    def __post_init__(self):
        if not self.name:
            self.name = f"custom (e = {self.epsilon:.3f})"
        kind = self.cavity.strip().lower()
        if kind not in {"lebedev", "fine"}:
            raise ValueError(
                "SolventModel.cavity must be 'lebedev' or 'fine' "
                f"(got {self.cavity!r})."
            )
        self.cavity = kind
        frame = self.fine_frame.strip().lower()
        if frame not in {"lab", "molecular"}:
            raise ValueError(
                "SolventModel.fine_frame must be 'lab' or 'molecular' "
                f"(got {self.fine_frame!r})."
            )
        self.fine_frame = frame
        variant = self.variant.strip().lower()
        if variant not in {"cpcm", "cosmo"}:
            raise ValueError(
                "SolventModel.variant must be 'cpcm' or 'cosmo' "
                f"(got {self.variant!r})."
            )
        self.variant = variant

    @property
    def is_gas_phase(self) -> bool:
        return self.epsilon <= 1.0


# Accepted user input types for ``solvent=``.
SolventSpec = Union[None, str, float, int, Mapping[str, Any], SolventModel]


def resolve_solvent(spec: SolventSpec) -> Optional[SolventModel]:
    """Coerce a user-supplied ``solvent`` argument to a SolventModel.

    Accepts:

    * ``None`` -- gas phase (returns ``None``).
    * ``"water"`` (or any other key/alias in
      :data:`vibeqc.solvation.SOLVENT_PRESETS`) -- looks up the
      preset dielectric.
    * a numeric e -- treated as a custom-dielectric solvent.
    * a dict -- passed to :class:`SolventModel` as kwargs (must include
      ``epsilon``).
    * a :class:`SolventModel` -- returned as-is.

    The case ``"vacuum"`` / ``"none"`` / ``"gas"`` resolves to e = 1
    and triggers the gas-phase no-op branch in :func:`run_cpcm_scf`.
    """
    if spec is None:
        return None
    if isinstance(spec, SolventModel):
        return spec
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        eps = float(spec)
        return SolventModel(epsilon=eps)
    if isinstance(spec, str):
        canon = canonicalise_solvent_name(spec)
        eps = SOLVENT_PRESETS[canon]
        return SolventModel(epsilon=eps, name=canon)
    if isinstance(spec, Mapping):
        params = dict(spec)
        if "epsilon" not in params:
            raise ValueError(
                "solvent dict must contain 'epsilon' "
                "(see vibeqc.solvation.SolventModel for other keys)."
            )
        return SolventModel(**params)
    raise TypeError(
        f"Unrecognised solvent spec: {type(spec).__name__}. "
        f"Expected None, str, float, dict, or SolventModel."
    )


def _build_point_charge_operators(
    basis,  # vibeqc.BasisSet
    cavity_points: np.ndarray,
) -> np.ndarray:
    """Stack of one-electron operators <mu|1/|r - s_i||ν> per cavity point.

    Built once per geometry by calling :func:`vibeqc.compute_nuclear`
    on a single-atom fake molecule (Z = 1) at each cavity point. The
    libint nuclear-attraction kernel returns
    ``M_i = -<mu|1/r_i|ν>``; we negate to expose the positive Coulomb
    kernel so callers can write ``F += q[i] * stack[i]`` directly
    (matches the standard CPCM Fock-contribution sign convention --
    derivation in module docstring of :mod:`vibeqc.solvation.cpcm`).

    Shape (n_pts, n_bf, n_bf), dtype float64. Memory is
    n_pts . n_bf^2 doubles -- ~5 MB for 240 points x 50 basis fns.

    Performance: n_pts independent libint calls; each is microseconds
    for a single-atom Z=1 nuclear operator on a typical AO basis.
    Total setup cost is comparable to a single full Fock build.
    """
    # Local import to avoid a top-level circular import via
    # ``vibeqc/__init__.py`` (which imports vibeqc.solvation at
    # bottom-of-file scope).
    from vibeqc import Atom, Molecule, compute_nuclear

    n_pts = int(cavity_points.shape[0])
    n_bf = int(basis.nbasis)
    stack = np.empty((n_pts, n_bf, n_bf), dtype=np.float64)
    for i in range(n_pts):
        # Fake single-atom molecule with Z = 1 at the cavity point.
        # libint returns V_muν = -Z . <mu|1/|r - s_i||ν> = -<mu|1/r_i|ν>.
        # Negate to surface the positive Coulomb kernel.
        # charge=1 + Z=1 -> 0 electrons, mult=1 -> valid empty system.
        # The integral builder only reads Z + position from the atom,
        # so this gives M_i = -<mu|1/|r - s_i||ν> at unit charge without
        # tripping Molecule's electron/multiplicity consistency check.
        fake = Molecule([Atom(1, tuple(cavity_points[i]))], 1, 1)
        M_i = compute_nuclear(basis, fake)
        stack[i] = -np.asarray(M_i)
    return stack


def _nuclear_potential_at_cavity(
    atom_positions_bohr: np.ndarray,
    atomic_numbers: np.ndarray,
    cavity_points: np.ndarray,
) -> np.ndarray:
    """V_nuc(s_i) = S_A Z_A / |R_A - s_i| at every cavity point.

    Vectorised; O(n_atoms . n_pts) flops. Closed-form -- no integral
    machinery needed since the nuclear charge density is a sum of
    delta functions.
    """
    R = np.asarray(atom_positions_bohr, dtype=np.float64)
    Z = np.asarray(atomic_numbers, dtype=np.float64)
    S = np.asarray(cavity_points, dtype=np.float64)
    # Pairwise distances (n_atoms, n_pts):
    dist = np.sqrt(((R[:, None, :] - S[None, :, :]) ** 2).sum(axis=2))
    if np.any(dist == 0.0):
        raise ValueError(
            "Cavity point lies exactly on a nucleus -- this should not "
            "happen with the GEPOL switching, but did. Check "
            "atom_positions_bohr."
        )
    return (Z[:, None] / dist).sum(axis=0)


def _density_potential_at_cavity(
    density: np.ndarray,
    point_charge_operators: np.ndarray,
) -> np.ndarray:
    """V_elec(s_i) = -∫ r/|r - s_i| dr = +tr(D . M_i) for each i.

    Sign convention: ``point_charge_operators[i]`` is the positive
    Coulomb kernel <mu|1/r_i|ν>. The electron density carries an
    implicit negative charge, so
    V_elec(s_i) = -S_muν D_muν . <mu|1/r_i|ν>
                = -tensordot(D, M_i, axes=2).
    Returns shape (n_pts,) values typically negative for normal
    electron distributions (negative density -> negative potential).
    """
    return -np.einsum("ab,iab->i", density, point_charge_operators)


def _fock_solvent_contribution(
    apparent_charges: np.ndarray,
    point_charge_operators: np.ndarray,
) -> np.ndarray:
    """V_q^{muν} = - S_i q_i . <mu|1/r_i|ν> -- the Fock matrix correction.

    Derivation: the one-electron operator an electron (charge -1) at
    position **r** feels from external point charges ``q_i`` at
    cavity positions ``s_i`` is

        v(**r**) = S_i (-1) . q_i / |**r** - **s_i**|.

    In AO basis ``V_q^{muν} = -S_i q_i <mu|1/|r - s_i||ν>``, which is
    what gets *added* to ``H_core`` when the SCF runs in solvent.
    ``point_charge_operators[i]`` already exposes the positive
    Coulomb kernel ``+<mu|1/|r-s_i||ν>`` (see
    :func:`_build_point_charge_operators`), so the minus sign appears
    here.

    Equivalent identity: at convergence, ``Tr(D V_q) = +q.V_elec``
    (with the sign convention ``V_elec_i = -S_{muν} D_{muν} <mu|...|ν>``
    used in :func:`_density_potential_at_cavity`). That identity is
    what the total-energy decomposition in :func:`run_cpcm_scf` relies
    on -- getting this sign wrong tips the in-solvent SCF energy by
    ``≈ -2 q.V_elec`` (tens of mHa on polar solutes), so the test
    suite asserts ``sol.energy < sol.e_gas`` as a hard guard.
    """
    return -np.einsum("i,iab->ab", apparent_charges, point_charge_operators)


class GaussianESPProvider:
    """ESP-on-grid solute<->cavity coupling for a Gaussian-basis reference.

    Implements :class:`vibeqc.solvation.provider.SolutePotentialProvider` for
    HF / DFT: it caches the per-cavity-point one-electron Coulomb operators
    ``<mu|1/|r-s_i||ν>`` once per geometry and uses them for both arrows -- the
    electronic ESP at the surface (from the density) and the Fock operator
    (from the apparent charges).  This is the coupling :func:`run_cpcm_scf`
    has always used, now behind the provider seam so a non-Gaussian reference
    (MSINDO) can supply its own coupling to the same reaction-field engine.
    """

    __slots__ = ("_operators",)

    def __init__(self, basis, cavity_points: np.ndarray):
        self._operators = _build_point_charge_operators(basis, cavity_points)

    @property
    def operators(self) -> np.ndarray:
        """The cached ``(n_pts, n_bf, n_bf)`` Coulomb-operator stack."""
        return self._operators

    def esp_at_cavity(self, density: np.ndarray) -> np.ndarray:
        return _density_potential_at_cavity(density, self._operators)

    def fock_contribution(self, charges: np.ndarray) -> np.ndarray:
        return _fock_solvent_contribution(charges, self._operators)


@dataclass
class SolventResult:
    """Outcome of a CPCM-coupled SCF run.

    Carries the original SCF result type (RHFResult / UHFResult /
    RKSResult / UKSResult) on ``.scf`` plus the converged solvation
    diagnostics.
    """

    scf: Any  # underlying SCF result
    energy: float  # total energy in solvent
    e_solv: float  # solvation energy
    e_gas: Optional[float]  # gas-phase reference
    epsilon: float
    solvent_name: str
    solvent_variant: str
    cavity: CavityTessellation
    cpcm: CPCMResult
    n_macro_iter: int
    macro_history: list[dict[str, float]] = field(default_factory=list)
    converged: bool = False
    # The screening model that actually built ``cpcm.q``. ``None`` means no
    # reaction field (gas phase): absent screening, not an out-of-range
    # epsilon. Consumers read ``screening.f`` rather than re-deriving it from
    # ``(epsilon, solvent_variant)`` -- that re-derivation is what made COSMO
    # gradients differentiate an energy the run never reported (#546, #548).
    screening: Optional[ScreeningModel] = None
    z_eff: Optional[np.ndarray] = (
        None  # (n_atoms,) -- ECP-effective Z; None = all-electron
    )
    all_scf_traces: list[Any] = field(default_factory=list)
    # Aggregated per-iteration records across gas-phase + all inner
    # SCF phases, suitable for replay into PerfTracker (BUG 100).

    def __repr__(self) -> str:
        return (
            f"SolventResult(solvent={self.solvent_name!r}, "
            f"variant={self.solvent_variant!r}, "
            f"e={self.epsilon:.3f}, E_solv={self.e_solv:+.6f} Ha, "
            f"E_total={self.energy:.6f} Ha, "
            f"macro_iter={self.n_macro_iter}, "
            f"converged={self.converged})"
        )


@dataclass(frozen=True)
class _RecordedECPCenter:
    """Immutable copy of one XML ECP center used by the CPCM Hcore."""

    Z: int
    xyz: tuple[float, float, float]


class _VerifiedMolecularSCFResult:
    """Read-only provenance view of a CPCM low-level SCF result.

    The native ``run_*_scf_with_jk`` functions correctly leave ECP
    provenance unverified because their public API accepts an arbitrary
    caller-built Hcore. ``run_cpcm_scf`` is a stronger boundary: it validates
    the XML input and constructs that Hcore itself. This wrapper records only
    that high-level fact without mutating or self-certifying the native
    low-level result. Gradient wrappers can recover the pybind object through
    ``_vibeqc_native_result`` after validating this exact provenance.
    """

    __slots__ = (
        "_vibeqc_native_result",
        "ecp_operator_applied",
        "ecp_provenance_verified",
        "ecp_xml_centers",
        "ecp_xml_library",
        "ecp_primitive_blocks",
        "ecp_primitive_centers",
        "ecp_effective_charges",
        "ecp_total_ncore",
    )

    def __init__(
        self,
        native_result,
        *,
        xml_centers,
        xml_library: str,
        total_ncore: int,
        primitive_blocks=(),
        primitive_centers=(),
        effective_charges=(),
    ) -> None:
        recorded_centers = tuple(
            _RecordedECPCenter(
                int(center.Z),
                tuple(float(value) for value in center.xyz),
            )
            for center in xml_centers
        )
        primitive_blocks = tuple(primitive_blocks)
        primitive_centers = tuple(
            tuple(float(v) for v in center) for center in primitive_centers
        )
        effective_charges = tuple(float(q) for q in effective_charges)
        if recorded_centers and primitive_blocks:
            raise ValueError(
                "run_cpcm_scf: XML and inline ECP routes recorded together"
            )
        if primitive_blocks and len(primitive_blocks) != len(primitive_centers):
            raise ValueError(
                "run_cpcm_scf: inline ECP blocks and centres differ in length"
            )
        operator_applied = bool(recorded_centers or primitive_blocks)
        total_ncore = int(total_ncore)
        if total_ncore < 0 or (not operator_applied and total_ncore != 0):
            raise ValueError(
                "run_cpcm_scf: inconsistent verified ECP core-count "
                "provenance"
            )
        object.__setattr__(self, "_vibeqc_native_result", native_result)
        object.__setattr__(self, "ecp_operator_applied", operator_applied)
        object.__setattr__(self, "ecp_provenance_verified", True)
        object.__setattr__(self, "ecp_xml_centers", recorded_centers)
        object.__setattr__(
            self,
            "ecp_xml_library",
            str(xml_library or "ecp10mdf") if recorded_centers else "",
        )
        object.__setattr__(self, "ecp_primitive_blocks", primitive_blocks)
        object.__setattr__(self, "ecp_primitive_centers", primitive_centers)
        object.__setattr__(
            self,
            "ecp_effective_charges",
            effective_charges if primitive_blocks else (),
        )
        object.__setattr__(self, "ecp_total_ncore", total_ncore)

    def __getattr__(self, name):
        return getattr(self._vibeqc_native_result, name)

    def __setattr__(self, name, value):
        raise AttributeError(
            f"_VerifiedMolecularSCFResult is read-only; cannot set {name!r}"
        )


def _resolve_method_runners(
    method: str,
):
    """Map the high-level method to (gas_runner, jk_inner_runner, needs_grid).

    Each runner pair returns ``(SCFResult, density)`` tuples so the
    macro-iteration can compose them generically.
    """
    from vibeqc import (
        run_rhf,
        run_rhf_scf_with_jk,
        run_rks,
        run_rks_scf_with_jk,
        run_uhf,
        run_uhf_scf_with_jk,
        run_uks,
        run_uks_scf_with_jk,
    )

    m = method.lower()
    if m == "rhf":
        return ("rhf", run_rhf, run_rhf_scf_with_jk, False)
    if m == "uhf":
        return ("uhf", run_uhf, run_uhf_scf_with_jk, False)
    if m == "rks":
        return ("rks", run_rks, run_rks_scf_with_jk, True)
    if m == "uks":
        return ("uks", run_uks, run_uks_scf_with_jk, True)
    raise ValueError(
        f"run_cpcm_scf: unsupported method {method!r} "
        f"(expected one of: rhf, uhf, rks, uks)."
    )


def _density_of(result, *, open_shell: bool) -> np.ndarray:
    """Extract the SCF density (closed-shell or a+b total)."""
    if open_shell:
        return np.asarray(result.density_alpha) + np.asarray(result.density_beta)
    return np.asarray(result.density)


def _ensure_cpcm_aux_basis(options, basis) -> None:
    """If ``options`` asks for density fitting / RIJCOSX but leaves
    ``aux_basis`` empty, autodetect it from the orbital basis name and
    fill it in (in place).

    The high-level ``run_rhf`` / ``run_rks`` / ``run_uhf`` / ``run_uks``
    drivers -- used for the CPCM gas-phase bootstrap SCF -- refuse
    ``density_fit=True`` with an empty ``aux_basis``. Resolving the aux
    here keeps the gas SCF, the macro-iteration inner SCFs, and the
    JKBuilder all on one consistent auxiliary basis. In-place mutation
    matches the established options-handling pattern in
    ``vibeqc.runner._run_single_point`` (which sets ``opts.functional``
    the same way).
    """
    density_fit = bool(getattr(options, "density_fit", False))
    cosx = bool(getattr(options, "cosx", False))
    if not density_fit and not cosx:
        return
    if (getattr(options, "aux_basis", "") or "").strip():
        return
    from vibeqc import default_aux_basis_for

    orbital_name = (getattr(basis, "name", "") or "").strip()
    if not orbital_name:
        raise ValueError(
            "run_cpcm_scf: density_fit/cosx requested but aux_basis is "
            "empty and the orbital BasisSet has no name to autodetect "
            "from. Set options.aux_basis explicitly (e.g. "
            "'def2-svp-jk')."
        )
    options.aux_basis = default_aux_basis_for(orbital_name, kind="jk")


def _resolve_cpcm_jk_builder(basis, molecule, options):
    """Pick the JKBuilder the CPCM macro-iteration drives the inner SCF
    with, off the method options.

    * ``options.cosx`` (with or without ``density_fit``) -> RIJCOSX:
      RI-J Coulomb + chain-of-spheres exchange
      (:func:`vibeqc.make_cosx_jk_builder`).
    * ``options.density_fit`` -> RI-JK
      (:func:`vibeqc.make_df_jk_builder`).
    * neither -> direct, integral-driven
      (:func:`vibeqc.make_direct_jk_builder`) -- the v0.9.0 default.

    DF / RIJCOSX both need an auxiliary basis. ``options.aux_basis``
    is used when set; otherwise it is autodetected from the orbital
    basis name via :func:`vibeqc.default_aux_basis_for`. The builder
    is constructed once and reused across every CPCM macro-iteration
    (the basis doesn't move).
    """
    from vibeqc import (
        build_grid,
        default_aux_basis_for,
        make_cosx_jk_builder,
        make_df_jk_builder,
        make_direct_jk_builder,
    )

    density_fit = bool(getattr(options, "density_fit", False))
    cosx = bool(getattr(options, "cosx", False))

    if not density_fit and not cosx:
        # Read the incremental-Fock settings from the SCF options so the
        # CPCM macro-iteration benefits from the same ΔD screening the
        # plain SCF uses.  incremental=true cuts per-iteration Fock-build
        # cost ~3-10× after the first iteration (BUG 100).
        incremental = bool(getattr(options, "incremental_fock", False))
        reset_freq = int(getattr(options, "incremental_fock_reset_freq", 8))
        schwarz_thr = float(getattr(options, "schwarz_threshold", 1e-10))
        return make_direct_jk_builder(
            basis, schwarz_thr, incremental=incremental, reset_freq=reset_freq,
        )

    # DF / RIJCOSX path -- resolve the auxiliary basis.
    aux_name = (getattr(options, "aux_basis", "") or "").strip()
    if not aux_name:
        orbital_name = getattr(basis, "name", "") or ""
        if not orbital_name:
            raise ValueError(
                "run_cpcm_scf: density_fit/cosx requested but the "
                "auxiliary basis could not be autodetected (orbital "
                "BasisSet has no name). Set options.aux_basis "
                "explicitly (e.g. 'def2-svp-jk')."
            )
        aux_name = default_aux_basis_for(orbital_name, kind="jk")
    # #480: this auxiliary basis goes straight to the C++ JK builders, not
    # through DensityFitting, so it needs its own coverage check. libint2
    # returns zero shells on an element a fitting file omits instead of
    # raising, which would fit the CPCM inner SCF with no auxiliary
    # functions on that centre and converge cleanly to a wrong energy.
    from vibeqc.density_fitting import make_checked_aux_basis

    aux = make_checked_aux_basis(
        molecule,
        aux_name,
        orbital_basis=basis,
        orbital_basis_name=getattr(basis, "name", "") or "",
        route="run_cpcm_scf",
    )

    if cosx:
        # RIJCOSX -- RI-J + seminumerical COSX-K on a quadrature grid.
        cosx_grid_opts = getattr(options, "cosx_grid", None)
        cosx_grid = (
            build_grid(molecule, cosx_grid_opts)
            if cosx_grid_opts is not None
            else build_grid(molecule)
        )
        return make_cosx_jk_builder(basis, aux, cosx_grid)

    return make_df_jk_builder(basis, aux)


def run_cpcm_scf(
    molecule,  # vibeqc.Molecule
    basis,  # vibeqc.BasisSet
    *,
    method: str = "rhf",
    solvent: SolventSpec = "water",
    options: Any = None,
    xc_grid_options: Any = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    output=None,
) -> SolventResult:
    """Self-consistent CPCM SCF for a molecular system.

    Parameters
    ----------
    molecule
        :class:`vibeqc.Molecule` -- atomic structure + charge + spin.
    basis
        :class:`vibeqc.BasisSet`.
    method : {"rhf", "uhf", "rks", "uks"}
        Underlying SCF method. KS methods (``rks`` / ``uks``) require
        the corresponding ``options`` to carry a ``functional``.
    solvent
        Anything accepted by :func:`resolve_solvent` -- preset name,
        e value, dict, or :class:`SolventModel`. ``None`` /
        ``"vacuum"`` short-circuits to a plain gas-phase SCF
        (still returned as a :class:`SolventResult` with
        ``e_solv = 0`` for uniform downstream handling).
    options
        Method-specific options struct (``RHFOptions`` / ``UHFOptions`` /
        ``RKSOptions`` / ``UKSOptions``). Default uses each method's
        bundled default.
    xc_grid_options
        Optional :class:`GridOptions` for the KS numerical integration
        grid. Defaults to the options object's own ``.grid``.
    progress_callback
        Optional callable invoked once per macro-iteration with a
        diagnostic dict (``iter``, ``e_solv``, ``delta_e_solv``,
        ``total_q``, ``scf_iters``). Useful for live-logging in
        :func:`vibeqc.run_job` and notebook progress bars.

    Returns
    -------
    SolventResult
        Final SCF result + solvation diagnostics. ``result.energy``
        is the *total* in-solvent energy (gas-phase electronic +
        nuclear + solvation); ``result.e_solv`` is just the
        polarisation contribution.
    """
    # ---- Gas-phase short-circuit ----------------------------------------
    sm = resolve_solvent(solvent)
    if sm is None or sm.is_gas_phase:
        return _gas_phase_solvent_result(
            molecule,
            basis,
            method=method,
            options=options,
            xc_grid_options=xc_grid_options,
            solvent_model=sm,
        )

    method_label, _gas_runner, jk_runner, needs_grid = _resolve_method_runners(method)
    open_shell = method_label in ("uhf", "uks")

    # Work on a value copy for every route. CPCM fills derived ECP metadata and
    # may disable an unsupported coupled-solvent stability check; those
    # internal decisions should not mutate a caller's reusable options object.
    # Auxiliary-basis autodetection is the established exception: callers have
    # historically received the resolved ``aux_basis`` value in place.
    from vibeqc import RHFOptions, RKSOptions, UHFOptions, UKSOptions

    _copy_cls = {
        "rhf": RHFOptions,
        "uhf": UHFOptions,
        "rks": RKSOptions,
        "uks": UKSOptions,
    }[method_label]
    opts_for_jk = _copy_cls(
        options if options is not None else _default_options(method_label)
    )

    # Direct run_cpcm_scf callers need the same authoritative basis-sidecar
    # attachment and BUG-99 validation as run_job. The current CPCM Hcore
    # builder consumes the selected XML or inline operator and preserves its
    # effective charges and core count through the inner SCF.
    from vibeqc.ecp_metadata import (
        attach_inline_ecp_options_from_basis_sidecar,
        validate_ecp_required,
    )

    attach_inline_ecp_options_from_basis_sidecar(opts_for_jk, molecule, basis)
    _basis_name = str(getattr(basis, "name", "") or "").strip()
    if _basis_name:
        validate_ecp_required(opts_for_jk, molecule, _basis_name)

    # ---- Resolve ECP effective charges (v0.11.0) ------------------------
    # When ECP centres are present, look up n_core per atom via libecpint
    # and replace the bare atomic numbers with Z_eff = Z - n_core in every
    # place that feeds the CPCM cavity electrostatic potential and nuclear
    # repulsion.  Without this correction the cavity sees a spuriously
    # large net solute charge (e.g. Zn^2⁺ with ecp10mdf: Z=30, n_core=10 ->
    # bare Z gives q_net=+12 instead of physical +2) and E_solv is wrong
    # by (Z_bare / Z_eff)^2.
    z_eff: Optional[np.ndarray] = None
    total_ncore = 0
    ecp_centers = list(getattr(opts_for_jk, "ecp_centers", []) or [])
    inline_blocks = list(getattr(opts_for_jk, "ecp_primitive_blocks", []) or [])
    inline_centers = [
        [float(v) for v in center]
        for center in (getattr(opts_for_jk, "ecp_primitive_centers", []) or [])
    ]
    if inline_blocks and ecp_centers:
        raise ValueError(
            "run_cpcm_scf: ecp_centers (XML library) and ecp_primitive_blocks "
            "(inline primitives) are mutually exclusive"
        )
    if inline_blocks:
        # Inline route (the bundled sidecars, vDZP, CRYSTAL/pob data): the
        # options already carry the per-atom effective charges and the
        # aggregate core count. Every centre must sit on an atom of this
        # molecule, exactly as the native SCF drivers demand.
        from vibeqc.ecp_metadata import (
            ecp_centre_atom_indices,
            effective_nuclear_charges,
        )

        ecp_centre_atom_indices(opts_for_jk, molecule)
        z_eff = effective_nuclear_charges(molecule, opts_for_jk)
        total_ncore = int(getattr(opts_for_jk, "ecp_total_ncore", 0) or 0)
        _core_from_charges = round(
            sum(float(a.Z) for a in molecule.atoms) - float(z_eff.sum())
        )
        if total_ncore != _core_from_charges:
            raise ValueError(
                "run_cpcm_scf: ecp_total_ncore disagrees with the effective "
                f"charges ({total_ncore} vs {_core_from_charges})"
            )
    if ecp_centers:
        from vibeqc import ecp_core_electrons

        ecp_lib = getattr(opts_for_jk, "ecp_library", "") or "ecp10mdf"
        charges = [c.Z for c in ecp_centers]
        core_map = ecp_core_electrons(charges, ecp_lib, "")

        atoms = list(molecule.atoms)
        atom_cores = [0] * len(atoms)
        matched_atoms: set[int] = set()
        for center in ecp_centers:
            matches = []
            for atom_index, atom in enumerate(atoms):
                dx = atom.xyz[0] - center.xyz[0]
                dy = atom.xyz[1] - center.xyz[1]
                dz = atom.xyz[2] - center.xyz[2]
                if (
                    int(atom.Z) == int(center.Z)
                    and dx * dx + dy * dy + dz * dz < 1e-12
                ):
                    matches.append(atom_index)
            if len(matches) != 1:
                raise ValueError(
                    "run_cpcm_scf: each ECP center must match exactly one "
                    "molecule atom by atomic number and position"
                )
            atom_index = matches[0]
            if atom_index in matched_atoms:
                raise ValueError(
                    "run_cpcm_scf: duplicate ECP centers match the same "
                    "molecule atom"
                )
            if int(center.Z) not in core_map:
                raise ValueError(
                    "run_cpcm_scf: the selected ECP library has no "
                    f"core-electron entry for atomic number {int(center.Z)}"
                )
            ncore = int(core_map[int(center.Z)])
            if ncore < 0 or ncore > int(center.Z):
                raise ValueError(
                    "run_cpcm_scf: invalid ECP core-electron count for "
                    f"atomic number {int(center.Z)}"
                )
            matched_atoms.add(atom_index)
            atom_cores[atom_index] = ncore

        # Build per-atom Z_eff only after every requested centre has passed
        # the one-to-one molecular placement and library-coverage checks.
        z_eff = np.array([float(a.Z) for a in atoms], dtype=np.float64)
        total_ncore = int(sum(atom_cores))
        for atom_index, ncore in enumerate(atom_cores):
            z_eff[atom_index] -= float(ncore)

    # ---- Validate ECP library coverage ----------------------------------
    # If the user requested ECPs but libecpint didn't return any n_core
    # data (e.g. mis-specified library name), raise early.
    if ecp_centers and total_ncore == 0:
        raise ValueError(
            "ECP centres were specified but ecp_core_electrons returned "
            "no n_core data for any centre. Check that ecp_library matches "
            "a bundled libecpint XML library (ecp10mdf, ecp28mdf, ecp46mdf, "
            "ecp60mdf, ecp78mdf, lanl2dz)."
        )

    # If density-fitting / RIJCOSX was requested, make sure the
    # auxiliary basis is resolved *before* the gas-phase SCF -- the
    # high-level run_rhf/run_rks/... refuse density_fit=True with an
    # empty aux_basis. Autodetect from the orbital basis name and fill
    # it in so the gas SCF, the macro-iteration inner SCFs, and the
    # JKBuilder all see one consistent aux basis.
    _ensure_cpcm_aux_basis(opts_for_jk, basis)
    if options is not None and hasattr(options, "aux_basis"):
        options.aux_basis = opts_for_jk.aux_basis

    from vibeqc.guess import prepare_molecular_guess_source

    prepare_molecular_guess_source(method_label, opts_for_jk, molecule, basis)

    # ---- Step 0: precompute Hcore + JKBuilder (shared by all SCFs) ------
    # Moved before the gas-phase SCF so every phase (gas + inner) shares
    # one JKBuilder.  Without this the gas-phase boot SCF routes through
    # the high-level runner which constructs its own JKBuilder — two
    # builders with potentially different Fock modes, doubling memory
    # and making the SCF trace single-phased (BUG 100).
    atoms = list(molecule.atoms)
    from vibeqc import (
        compute_ecp_matrix,
        compute_kinetic,
        compute_nuclear,
        compute_nuclear_with_charges,
        compute_overlap,
    )

    S = np.asarray(compute_overlap(basis))
    T_op = np.asarray(compute_kinetic(basis))
    if inline_blocks:
        from vibeqc import compute_ecp_matrix_from_primitives

        V_nuc_op = np.asarray(
            compute_nuclear_with_charges(
                basis,
                [list(a.xyz) for a in atoms],
                z_eff.tolist(),
            )
        )
        _flat_centers = [v for center in inline_centers for v in center]
        V_ecp_op = np.asarray(
            compute_ecp_matrix_from_primitives(basis, _flat_centers, inline_blocks)
        )
        Hcore_gas = T_op + V_nuc_op + V_ecp_op
        E_nuc = _point_charge_repulsion(atoms, z_eff)
        n_electrons = int(molecule.n_electrons()) - total_ncore
    elif ecp_centers:
        ecp_lib = getattr(opts_for_jk, "ecp_library", "") or "ecp10mdf"
        V_nuc_op = np.asarray(
            compute_nuclear_with_charges(
                basis,
                [list(a.xyz) for a in atoms],
                z_eff.tolist(),
            )
        )
        V_ecp_op = np.asarray(compute_ecp_matrix(basis, ecp_centers, ecp_lib))
        Hcore_gas = T_op + V_nuc_op + V_ecp_op
        E_nuc = _point_charge_repulsion(atoms, z_eff)
        n_electrons = int(molecule.n_electrons()) - total_ncore
    else:
        V_nuc_op = np.asarray(compute_nuclear(basis, molecule))
        Hcore_gas = T_op + V_nuc_op
        E_nuc = float(molecule.nuclear_repulsion())
        n_electrons = int(molecule.n_electrons())

    if method_label in ("uks", "rhf", "rks"):
        # A gas-phase orbital Hessian omits the density response of the
        # self-consistent reaction field, so it cannot classify or follow a
        # solvated stationary point. Work on a value copy: callers retain
        # their option object, including whether True was an implicit default
        # or an explicit request. RHF/RKS carry the issue-#144
        # restricted-stability VERDICT surface (detect, never follow); the
        # same incomplete-response argument applies to its Hessian.
        if bool(opts_for_jk.stability_check):
            if bool(opts_for_jk._stability_check_explicit):
                raise RuntimeError(
                    f"CPCM {method_label.upper()} internal stability analysis "
                    "was explicitly requested, but the self-consistent "
                    "reaction-field density response is not yet plumbed. Set "
                    "stability_check=False to run the supported first-order "
                    "solvated SCF without a stability verdict."
                )
            opts_for_jk.stability_check = False
    # The CPCM path builds the ECP Hamiltonian itself, bypassing the
    # high-level SCF wrapper that normally stamps this authoritative count.
    # Carry it through the low-level result so every downstream consumer can
    # distinguish an effective-electron reference from an all-electron one.
    opts_for_jk.ecp_total_ncore = int(total_ncore)
    jk_builder = _resolve_cpcm_jk_builder(basis, molecule, opts_for_jk)

    xc_grid = None
    if needs_grid:
        from vibeqc import build_grid

        grid_opts = xc_grid_options
        if grid_opts is None and hasattr(opts_for_jk, "grid"):
            grid_opts = opts_for_jk.grid
        xc_grid = (
            build_grid(molecule, grid_opts)
            if grid_opts is not None
            else build_grid(molecule)
        )

    # ---- Step 1: gas-phase SCF (shared JKBuilder, no second ERI tensor) --
    if method_label == "rhf":
        gas_result = jk_runner(
            basis, n_electrons, S, Hcore_gas, E_nuc,
            jk_builder, opts_for_jk, molecule=molecule,
        )
    elif method_label == "uhf":
        n_alpha, n_beta = _alpha_beta_counts(molecule, n_electrons)
        gas_result = jk_runner(
            basis, n_alpha, n_beta, S, Hcore_gas, E_nuc,
            jk_builder, opts_for_jk, molecule=molecule,
        )
    elif method_label == "rks":
        gas_result = jk_runner(
            basis, n_electrons, S, Hcore_gas, E_nuc,
            jk_builder, xc_grid, opts_for_jk, molecule=molecule,
        )
    else:  # uks
        n_alpha, n_beta = _alpha_beta_counts(molecule, n_electrons)
        gas_result = jk_runner(
            basis, n_alpha, n_beta, S, Hcore_gas, E_nuc,
            jk_builder, xc_grid, opts_for_jk, molecule=molecule,
        )
    if not gas_result.converged:
        raise RuntimeError(
            "run_cpcm_scf: gas-phase SCF did not converge -- CPCM "
            "macro-iteration is meaningless from a non-stationary "
            "density. Address the gas-phase convergence problem first "
            "(level_shift / damping / better guess)."
        )
    from vibeqc.guess import GuessSelection
    from vibeqc import InitialGuess

    warm_selection = GuessSelection(
        gas_result.guess_selection.requested,
        gas_result.guess_selection.effective, InitialGuess.READ,
    )
    e_gas = float(gas_result.energy)
    D = _density_of(gas_result, open_shell=open_shell)

    # Collect SCF traces across all phases for truthful per-iteration
    # wall times in the .perf report (BUG 100 / BUG87-A).
    _all_scf_traces: list[Any] = list(
        getattr(gas_result, "scf_trace", []) or []
    )

    # ---- Step 2: build cavity + cached operators ------------------------
    pos = np.array([list(a.xyz) for a in atoms], dtype=np.float64)
    Zs = np.array([int(a.Z) for a in atoms], dtype=int)
    # Use effective Z for the cavity electrostatic potential when ECPs
    # are present, so the apparent surface charges see the physical net
    # solute charge rather than the bare-nucleus charge.
    if z_eff is not None:
        Zs_cav = z_eff
    else:
        Zs_cav = Zs.astype(float)
    cavity = _build_cavity_for(sm, pos, Zs)
    # The A matrix is construction-independent. Its diagonal
    # 1.0694*sqrt(4pi)/sqrt(w) = 3.7909/sqrt(w) is Klamt 1993 eq. 7b's
    # a_uu ~ 3.8 |S_u|^(-1/2) -- the self-energy of a patch of area w, which
    # depends on the segment's area and not on how the tessellation produced
    # it. So the same builder serves the Lebedev and FINE cavities.
    A = build_cavity_A_matrix(cavity)
    # Solute<->cavity coupling behind the SolutePotentialProvider seam: the
    # Gaussian ESP-on-grid coupling here, an MSINDO multipole provider later
    # (vibeqc.solvation.provider). The reaction-field engine below (ESP ->
    # screened ASC solve -> Fock correction -> macro-iteration) is
    # reference-independent.
    provider = GaussianESPProvider(basis, cavity.points)
    V_nuc_cav = _nuclear_potential_at_cavity(pos, Zs_cav, cavity.points)

    # Pre-factor A once — the cavity is fixed across macro-iterations, so
    # each cycle reuses one LU rather than repeating an O(N³) decomposition
    # (BUG 100). The solve is injected into the shared reaction-field step.
    _cavity_solve = lu_cavity_solve(A)

    # One screening model for the whole run: built here, carried on the
    # result, read by every consumer. Nothing downstream re-derives ``f``.
    screening = ScreeningModel.from_variant(sm.epsilon, sm.variant)

    def _as_cpcm(field: ReactionField) -> CPCMResult:
        """The legacy per-solve record, projected from the shared field."""
        return CPCMResult(q=field.q, V=field.V_total,
                          e_solv=field.e_pol, epsilon=float(sm.epsilon))

    history: list[dict[str, float]] = []
    prev_e_solv = 0.0
    converged = False
    last_inner_result = gas_result
    last_cpcm: Optional[CPCMResult] = None
    _prev_V_q_solvent: np.ndarray | None = None  # for shifted DIIS warm-start

    # CPCM q-DIIS accelerator.  After each macro-iteration we store the
    # (q, error) pair where error = V_new − V_old is the change in the
    # total electrostatic potential at the cavity after the inner SCF.
    # On the next cycle we DIIS-extrapolate q to accelerate convergence.
    # This is the same technique PySCF / ORCA use (BUG 100 — q-DIIS).
    _q_diis_max = 5
    _q_diis_hist: list[np.ndarray] = []  # recent q vectors
    _e_diis_hist: list[np.ndarray] = []  # error = V_new - V_old
    _V_prev: np.ndarray | None = None     # V from previous iteration

    def _q_diis_extrapolate() -> np.ndarray | None:
        """DIIS-extrapolated q, or None if too few iterates."""
        n = len(_q_diis_hist)
        if n < 2:
            return None
        # B_{ij} = <e_i, e_j>
        B = np.zeros((n + 1, n + 1), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                v = float(np.dot(_e_diis_hist[i], _e_diis_hist[j]))
                B[i, j] = v
                B[j, i] = v
        B[n, :n] = -1.0
        B[:n, n] = -1.0
        # B[n, n] = 0.0
        rhs = np.zeros(n + 1, dtype=np.float64)
        rhs[n] = -1.0
        try:
            c = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            return None
        c_q = c[:n]
        if not np.all(np.isfinite(c_q)):
            return None
        # q_diis = Σ_i c_i · q_i  (Σ c_i = 1)
        q_ext = np.zeros_like(_q_diis_hist[0])
        for i in range(n):
            q_ext += c_q[i] * _q_diis_hist[i]
        return q_ext

    for macro_iter in range(1, sm.max_macro_iter + 1):
        # The shared reaction-field step: ESP -> screened ASC solve -> Fock
        # contribution -> energy split. One implementation, every method
        # (vibeqc.solvation.engine; #554).
        field = reaction_field_step(provider, _cavity_solve, V_nuc_cav, D, screening)
        V_tot = field.V_total

        # q-DIIS (opt-in): extrapolate apparent surface charges.
        if sm.q_diis:
            if _V_prev is not None:
                _q_diis_hist.append(np.asarray(field.q, dtype=np.float64))
                _e_diis_hist.append(
                    np.asarray(V_tot, dtype=np.float64)
                    - np.asarray(_V_prev, dtype=np.float64)
                )
                if len(_q_diis_hist) > _q_diis_max:
                    _q_diis_hist.pop(0)
                    _e_diis_hist.pop(0)
            _q_use = _q_diis_extrapolate()
            if _q_use is not None:
                # Re-express through the same step rather than rebuilding the
                # downstream quantities by hand; the potentials are reused, so
                # this costs one Fock scatter and no ESP pass.
                field = field.with_charges(_q_use, provider)
        cpcm = _as_cpcm(field)
        last_cpcm = cpcm
        _V_prev = V_tot  # save for next iteration's error computation

        # V_q matrix and corrected Hcore (built by the shared step).
        V_q = field.fock
        Hcore = Hcore_gas + V_q

        # Shifted DIIS warm-start (BUG 100): carry one (F, e) pair from
        # the previous converged SCF, with F shifted by ΔV_q so it
        # approximates the new Hamiltonian's Fock.  The error vector is
        # recomputed as e = F_shifted·D_old·S − S·D_old·F_shifted.
        # This gives DIIS a head-start entry before the first real
        # iteration — unlike naive history transfer, the shifted Fock
        # points to the right region of the new convergence path.
        _warm_fock: list = []
        _warm_error: list = []
        _prev_fock = getattr(last_inner_result, "fock", None)
        if _prev_fock is not None and _prev_V_q_solvent is not None and not open_shell:
            _dv = np.asarray(V_q, dtype=np.float64) - _prev_V_q_solvent
            _f_shifted = np.asarray(_prev_fock, dtype=np.float64) + _dv
            _d_prev = np.asarray(last_inner_result.density, dtype=np.float64)
            _e_shifted = _f_shifted @ _d_prev @ S - S @ _d_prev @ _f_shifted
            _warm_fock = [_f_shifted]
            _warm_error = [_e_shifted]

        # Inner SCF on the corrected Hcore. Seed with the current
        # density so DIIS bootstraps from where we left off -- converges
        # in ~2-4 iters typically once the macro-loop is close.
        # Reuse the same resolved option object as the gas bootstrap and JK
        # builder. For UKS this is the value copy with coupled-solvent
        # stability disabled; reverting to ``options`` here would re-enable
        # the incomplete fixed-reaction-field verdict on every macro step.
        opts = opts_for_jk
        if method_label == "rhf":
            inner = jk_runner(
                basis,
                n_electrons,
                S,
                Hcore,
                E_nuc,
                jk_builder,
                opts,
                D,
                molecule=molecule, guess_selection=warm_selection,
            )
        elif method_label == "uhf":
            n_alpha, n_beta = _alpha_beta_counts(molecule, n_electrons)
            D_a = np.asarray(last_inner_result.density_alpha)
            D_b = np.asarray(last_inner_result.density_beta)
            inner = jk_runner(
                basis,
                n_alpha,
                n_beta,
                S,
                Hcore,
                E_nuc,
                jk_builder,
                opts,
                D_a,
                D_b,
                molecule=molecule, guess_selection=warm_selection,
            )
        elif method_label == "rks":
            inner = jk_runner(
                basis,
                n_electrons,
                S,
                Hcore,
                E_nuc,
                jk_builder,
                xc_grid,
                opts,
                D,
                molecule=molecule, guess_selection=warm_selection,
                warm_fock_history=_warm_fock,
                warm_error_history=_warm_error,
            )
        else:  # uks
            n_alpha, n_beta = _alpha_beta_counts(molecule, n_electrons)
            D_a = np.asarray(last_inner_result.density_alpha)
            D_b = np.asarray(last_inner_result.density_beta)
            inner = jk_runner(
                basis,
                n_alpha,
                n_beta,
                S,
                Hcore,
                E_nuc,
                jk_builder,
                xc_grid,
                opts,
                D_a,
                D_b,
                molecule=molecule, guess_selection=warm_selection,
            )

        # Pull fresh density for the next macro-iteration.
        D = _density_of(inner, open_shell=open_shell)
        last_inner_result = inner
        _prev_V_q_solvent = np.asarray(V_q, dtype=np.float64).copy()

        # Aggregate per-iteration wall times so the .perf report covers
        # every SCF phase (gas + all macro-iterations), not just the
        # last inner SCF (BUG 100).
        _all_scf_traces.extend(
            getattr(inner, "scf_trace", []) or []
        )

        delta_e = cpcm.e_solv - prev_e_solv
        record = {
            "iter": macro_iter,
            "e_solv": cpcm.e_solv,
            "delta_e_solv": delta_e,
            "total_q": float(cpcm.q.sum()),
            "scf_iters": int(inner.n_iter),
        }
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)

        prev_e_solv = cpcm.e_solv
        if abs(delta_e) < sm.tol_e_solv:
            converged = True
            break

    assert last_cpcm is not None  # macro_iter loop runs at least once

    # ---- Standard CPCM total-energy decomposition -----------------------
    # The SCF returned ``last_inner_result.energy`` is the SCF energy that
    # includes ``V_q`` in the Fock matrix:
    #     E_SCF^{w/V_q} = E_HF^{gas}[D^{solv}] + Tr(D^{solv} V_q)
    #                   = E_HF^{gas}[D^{solv}] + q^T V_elec
    # The standard CPCM total energy (Cossi-Scalmani-Mennucci-Tomasi 2003
    # eq. 3, matches PySCF / Gaussian / ORCA convention) is
    #     E_tot^{PCM} = E_HF^{gas}[D^{solv}] + (1/2) q^T V_tot
    # so the "added" piece beyond the in-solvent SCF return value is
    #     ΔE = (1/2) q^T V_tot - q^T V_elec
    #        = (1/2) q^T V_nuc - (1/2) q^T V_elec.
    # This guards against the common subtle bug where the SCF's already-
    # included Tr(D V_q) term is double-counted by adding the full
    # 1/2 q.V on top.
    q_dot_Velec = float(np.dot(last_cpcm.q, last_cpcm.V - V_nuc_cav))
    e_gas_at_D_solv = float(last_inner_result.energy) - q_dot_Velec
    total_energy = e_gas_at_D_solv + last_cpcm.e_solv

    # Optional citation siblings -- fire the CPCM solvation papers
    # (Klamt-Schüürmann 1993 + Cossi-Rega-Scalmani-Barone 2003 +
    # Scalmani-Frisch 2010) when the standalone driver was invoked
    # with a non-None ``output=`` stem. ``run_job`` already handles
    # this for its own pipeline via runner.py.
    if output is not None:
        try:
            from vibeqc.output.citations import emit_citations

            _func = (
                getattr(options, "functional", None) if options is not None else None
            )
            emit_citations(
                output,
                method=method,
                basis=basis.name,
                functional=_func,
                uses_cpcm=True,
                scf_guess=last_inner_result.guess_selection.effective.name,
                solvent_variant="cpcm",
            )
        except Exception:
            # Best-effort -- never let citation writer failure tank a
            # converged CPCM run.
            pass

    verified_scf_result = _VerifiedMolecularSCFResult(
        last_inner_result,
        xml_centers=ecp_centers,
        xml_library=(
            str(getattr(opts_for_jk, "ecp_library", "") or "ecp10mdf")
            if ecp_centers else ""
        ),
        total_ncore=total_ncore,
        primitive_blocks=inline_blocks,
        primitive_centers=inline_centers,
        effective_charges=(z_eff.tolist() if inline_blocks else ()),
    )
    return SolventResult(
        scf=verified_scf_result,
        energy=total_energy,
        e_solv=last_cpcm.e_solv,
        e_gas=e_gas,
        epsilon=sm.epsilon,
        solvent_name=sm.name,
        solvent_variant=sm.variant,
        screening=screening,
        cavity=cavity,
        cpcm=last_cpcm,
        n_macro_iter=len(history),
        macro_history=history,
        converged=converged,
        z_eff=z_eff,
        all_scf_traces=_all_scf_traces,
    )


class _SolventAwareSCFResult:
    """Attribute-forwarding wrapper around an SCF result view.

    The underlying result is normally a pybind11 ``RHFResult`` / ``UHFResult``
    / ``RKSResult`` / ``UKSResult``; a CPCM run may first wrap it in the
    read-only exact-provenance view above. To keep
    ``run_job(..., solvent=...)`` returning something that downstream
    code can use exactly like the underlying SCF result *and* exposes
    the new solvation diagnostics, wrap the inner result and forward
    attribute access via ``__getattr__``.

    Exposed extra attributes:

    * ``solvent_result`` -- the full :class:`SolventResult`.
    * ``e_solv`` -- electrostatic polarisation energy ``1/2 q.V_tot``
      (Hartree).
    * ``e_gas`` -- gas-phase reference energy used by the solvent run.
    * ``energy_in_solvent`` -- standard Cossi-Scalmani total
      ``E_HF^gas[D^solv] + 1/2 q.V_tot``.

    The ``.energy`` attribute resolves to the **in-solvent total**
    (``energy_in_solvent``), so every consumer that reads ``.energy`` --
    the .out banner, the .xyz comment energy, the progress / structured
    log, ASE -- harvests the same headline the solvation block reports
    (IID 148: the pre-fix value was the inner SCF's ``E_SCF^{w/V_q}``,
    which is neither the gas-phase nor the in-solvent total).  The inner
    value stays reachable as ``.solvent_result.scf.energy``.
    """

    __slots__ = (
        "_inner",
        "solvent_result",
        "e_solv",
        "e_gas",
        "energy_in_solvent",
        "solvent_variant",
        "_all_scf_traces",
    )

    def __init__(self, scf_result, solvent_result):
        # Use object.__setattr__ to bypass our own __setattr__ guard.
        object.__setattr__(self, "_inner", scf_result)
        object.__setattr__(self, "solvent_result", solvent_result)
        object.__setattr__(self, "e_solv", solvent_result.e_solv)
        object.__setattr__(self, "e_gas", solvent_result.e_gas)
        object.__setattr__(self, "energy_in_solvent", solvent_result.energy)
        object.__setattr__(self, "solvent_variant", solvent_result.solvent_variant)
        # Aggregated SCF traces (gas + all inner phases) for
        # truthful per-iteration wall times in .perf (BUG 100).
        _traces = getattr(solvent_result, "all_scf_traces", None)
        object.__setattr__(
            self, "_all_scf_traces",
            list(_traces) if _traces else list(
                getattr(scf_result, "scf_trace", []) or []
            ),
        )

    def __getattr__(self, name):
        # scf_trace: return the aggregated multi-phase trace when the
        # SolventResult carries one; fall back to the wrapped inner
        # result's trace (single-phase, pre-BUG-100 behaviour).
        if name == "scf_trace":
            return self._all_scf_traces
        # energy: the headline total of a solvated run is the in-solvent
        # total (IID 148) -- banner, .xyz comment, progress log, and ASE
        # all read .energy, and they must agree with the solvation block.
        if name == "energy":
            return self.energy_in_solvent
        # __getattr__ is only called when the attribute isn't found
        # on this object (or its class); forward to the wrapped result.
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        # Disallow mutating the wrapped result via the proxy -- keeps
        # the C++ object's immutability semantics intact.
        raise AttributeError(
            f"_SolventAwareSCFResult is read-only; cannot set {name!r}"
        )

    def __repr__(self):
        return (
            f"_SolventAwareSCFResult("
            f"inner={type(self._inner).__name__}, "
            f"energy={self.energy:.6f}, "
            f"e_solv={self.e_solv:+.6f}, "
            f"energy_in_solvent={self.energy_in_solvent:.6f})"
        )


def _solvent_aware_scf_result(solvent_result):
    """Public constructor for :class:`_SolventAwareSCFResult`."""
    return _SolventAwareSCFResult(solvent_result.scf, solvent_result)


def _point_charge_repulsion(atoms, charges) -> float:
    """Pairwise ``q_i q_j / r_ij`` over the atoms with the given charges."""
    e = 0.0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dx = atoms[i].xyz[0] - atoms[j].xyz[0]
            dy = atoms[i].xyz[1] - atoms[j].xyz[1]
            dz = atoms[i].xyz[2] - atoms[j].xyz[2]
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            e += float(charges[i]) * float(charges[j]) / r
    return e


def _alpha_beta_counts(molecule, n_total: Optional[int] = None) -> tuple[int, int]:
    """Per-spin counts from a molecule and an optional effective total.

    For multiplicity m (= 2S + 1) and total n_electrons n:
        n_alpha - n_beta = m - 1
        n_alpha + n_beta = n
    Solving: n_alpha = (n + m - 1) / 2; n_beta = (n - m + 1) / 2.

    ``n_total`` is the ECP-reduced electron count on a pseudopotential
    route. Omitting it preserves the all-electron convenience behaviour.
    """
    if n_total is None:
        n_total = int(molecule.n_electrons())
    else:
        n_total = int(n_total)
    mult = int(molecule.multiplicity)
    spin_excess = mult - 1
    if mult < 1 or n_total < spin_excess or (n_total - spin_excess) % 2:
        raise ValueError(
            "Molecule multiplicity is incompatible with the effective "
            f"electron count ({n_total} electrons, multiplicity {mult})"
        )
    n_alpha = (n_total + mult - 1) // 2
    n_beta = n_total - n_alpha
    return n_alpha, n_beta


def _default_options(method: str):
    """Construct each method's bundled default options struct."""
    from vibeqc import RHFOptions, RKSOptions, UHFOptions, UKSOptions

    m = method.lower()
    if m == "rhf":
        return RHFOptions()
    if m == "uhf":
        return UHFOptions()
    if m == "rks":
        return RKSOptions()
    if m == "uks":
        return UKSOptions()
    raise ValueError(f"_default_options: unknown method {method!r}")


def _gas_phase_solvent_result(
    molecule,
    basis,
    *,
    method,
    options,
    xc_grid_options,
    solvent_model,
) -> SolventResult:
    """Gas-phase short-circuit -- uniform return type for downstream code.

    Called from :func:`run_cpcm_scf` when ``solvent`` resolves to
    ``None`` or ``e <= 1`` ("vacuum"). Builds an empty cavity (so
    ``result.cavity`` is still a valid :class:`CavityTessellation`
    that callers can introspect) and a zero-charge :class:`CPCMResult`.
    """
    method_label, gas_runner, _, _ = _resolve_method_runners(method)
    open_shell = method_label in ("uhf", "uks")
    if options is None:
        gas_result = gas_runner(molecule, basis)
    else:
        gas_result = gas_runner(molecule, basis, options)

    atoms = list(molecule.atoms)
    pos = np.array([list(a.xyz) for a in atoms], dtype=np.float64)
    Zs = np.array([int(a.Z) for a in atoms], dtype=int)
    # Build a minimal (but valid) cavity with the same defaults that
    # the in-solvent path would use, so downstream geometry inspection
    # is consistent across the gas / solvent branch.
    cavity = build_cavity(
        atom_positions_bohr=pos,
        atom_numbers=Zs,
        radii_scale=1.20,
        n_points_per_sphere=110,
    )
    empty_cpcm = CPCMResult(
        q=np.zeros(cavity.n_points),
        V=np.zeros(cavity.n_points),
        e_solv=0.0,
        epsilon=(solvent_model.epsilon if solvent_model is not None else 1.0),
    )
    name = solvent_model.name if solvent_model is not None else "vacuum"
    return SolventResult(
        scf=gas_result,
        energy=float(gas_result.energy),
        e_solv=0.0,
        e_gas=float(gas_result.energy),
        epsilon=(solvent_model.epsilon if solvent_model is not None else 1.0),
        solvent_name=name,
        solvent_variant=(
            solvent_model.variant if solvent_model is not None else "cpcm"
        ),
        cavity=cavity,
        cpcm=empty_cpcm,
        n_macro_iter=0,
        macro_history=[],
        converged=True,
    )
