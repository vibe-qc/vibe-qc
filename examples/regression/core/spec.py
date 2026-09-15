"""Per-system / per-method / per-reference dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Union

Frac = Tuple[float, float, float]


@dataclass(frozen=True)
class AtomFrac:
    symbol: str
    z: int
    frac: Frac


@dataclass(frozen=True)
class Provenance:
    """Where a structure came from, for citation in benchmark reports.

    Set by the external structure fetcher (``vibeqc.fetch``). Hand-curated
    SPECs leave ``provenance=None``; emitted SPECs always carry a
    populated Provenance.

    Empty string means "not applicable / unknown"; never None for the
    individual fields — the benchmark report writer treats None as
    "field missing from schema" and warns.
    """
    source_db: str
    """``"MaterialsProject"`` | ``"COD"`` | ``"OPTIMADE/<provider>"`` | ``"NOMAD"`` | ``"manual"``."""
    source_id: str
    """e.g. ``"mp-1265"``, ``"1011027"``, ``"nomad/abc123"``."""
    source_url: str
    """Canonical permalink for citation."""
    original_reference: str
    """DOI of the publication, when known."""
    license: str
    """``"CC-BY-4.0"`` | ``"CC0"`` | ``"non-commercial"`` | …"""
    fetched_at: str
    """ISO-8601 UTC timestamp."""
    fetcher_version: str
    """Version of the vibe-qc fetcher that produced this record."""
    notes: str = ""
    """Human-readable extras (e.g. which polymorph, MP property fields)."""


@dataclass(frozen=True)
class PeriodicSpec:
    """Geometry + metadata for one periodic test system."""
    id: str                                       # filesystem-safe slug
    family: str                                   # "rocksalt", "corundum", ...
    lattice_ang: Tuple[Tuple[float, float, float], ...]   # 3x3 row-vectors
    space_group: str
    atoms: Tuple[AtomFrac, ...]
    default_kmesh: Tuple[int, int, int] = (1, 1, 1)
    default_spacing_bohr: float = 0.6
    default_cutoff_bohr: float = 12.0
    default_nuclear_cutoff_bohr: float = 25.0
    default_omega: float = 0.5
    default_conv_tol_energy: float = 1e-7
    default_max_iter: int = 40
    # SCF-stability knobs. Defaults match the rare-gas / wide-gap path;
    # ionic / deep-core systems (Na, Mg, Cl, Li, Al, transition metals)
    # need SAD guess + damping ≥ 0.85 to dodge the HCORE-divergence
    # diagnostic the v0.7 SCF driver raises.
    default_initial_guess: str = "HCORE"          # "HCORE" | "SAD"
    default_damping: float = 0.5
    default_use_periodic_becke: bool = False      # True → expensive but accurate for tight bulk
    notes: str = ""
    citation: str = ""
    # ----- Optional fetcher-provided fields. Backward-compatible: every
    # hand-curated SPEC predates these and is unaffected.
    recommended_basis: Optional[str] = None
    """Heuristic-chosen basis (``"pob-tzvp"`` for ionic crystals,
    ``"sto-3g"`` for fast smoke tests). ``None`` for hand-curated SPECs."""
    magnetic_moments: Optional[Tuple[float, ...]] = None
    """Initial atomic magnetic moments (μ_B) for UHF/UKS guess. Length
    must equal ``len(atoms)`` when set; ``None`` for diamagnetic / unset."""
    is_open_shell: bool = False
    """When ``True``, downstream picks UHF/UKS over RHF/RKS."""
    provenance: Optional[Provenance] = None
    """Populated by the fetcher; ``None`` for hand-curated SPECs."""


@dataclass(frozen=True)
class AtomCart:
    symbol: str
    z: int
    xyz_ang: Tuple[float, float, float]


@dataclass(frozen=True)
class MoleculeSpec:
    """Geometry + metadata for one molecular test system."""
    id: str
    family: str = "molecule"
    atoms: Tuple[AtomCart, ...] = ()
    charge: int = 0
    multiplicity: int = 1
    default_conv_tol_energy: float = 1e-9
    default_max_iter: int = 80
    notes: str = ""
    citation: str = ""
    # ----- Optional fetcher-provided fields (mirrors PeriodicSpec).
    # Molecules already carry ``charge`` + ``multiplicity``; no extra
    # spin field needed.
    recommended_basis: Optional[str] = None
    """Heuristic-chosen basis (``"def2-svp"`` typical, ``"sto-3g"``
    smoke). ``None`` for hand-curated SPECs."""
    provenance: Optional[Provenance] = None
    """Populated by the fetcher; ``None`` for hand-curated SPECs."""


# Discriminated by isinstance at the call site. The fetcher returns
# ``TestSystem`` so consumers see a single point of polymorphism without
# a wrapper class to keep in sync with two underlying schemas.
TestSystem = Union[PeriodicSpec, MoleculeSpec]


@dataclass(frozen=True)
class MethodSpec:
    """One SCF / post-SCF method recipe."""
    id: str                                       # "rks-lda", "rhf", "uks-pbe", "mp2", "rhf-df"
    scf: str                                      # "rhf", "uhf", "rks", "uks"
    xc: Optional[str] = None                      # "lda", "pbe", "b3lyp", ...
    post: Optional[str] = None                    # None | "mp2"
    spin: str = "closed"
    periodic: bool = True
    molecular: bool = True
    # Density-fitting flag. When True the runners enable DF on PySCF
    # (mf.density_fit) and ORCA. ORCA uses RI + def2/J for pure GGAs
    # and RIJK + def2/JK for HF/hybrids; RIJCOSX/GridX is a separate
    # COSX exchange approximation route, not the default DF route.
    df: bool = False
    aux_basis: str = "def2-universal-jkfit"


@dataclass(frozen=True)
class ExpectedRef:
    """Reference values for one (system, basis, method, kmesh) cell."""
    system_id: str
    basis: str
    method_id: str
    kmesh: Tuple[int, int, int]
    primary_source: str = "pyscf_at_runtime"      # or "orca_at_runtime" / "published"
    energy_ha: Optional[float] = None             # null when recomputed at runtime
    tolerance_ha: float = 5e-3
    tolerance_rationale: str = ""
    published_energy_ha: Optional[float] = None
    published_source: str = ""


# ----------------------------------------------------------------------------
# Phase 2 (reference-data fetcher): experimental / computed reference records
# ----------------------------------------------------------------------------
#
# Pulled from NIST CCCBDB (SRD 101, doi:10.18434/T47C7Z) and NIST Chemistry
# WebBook (SRD 69) by ``vibeqc.fetch.references``. See
# ``docs/tutorial/external_data_fetcher.md``.

ReferenceKind = Literal["experimental", "computed", "evaluated"]
"""Discriminator for an :class:`ExperimentalReference`.

* ``"experimental"`` — direct measurement, with NIST-quoted uncertainty.
  This is the calibration anchor we want for QC method validation.
* ``"computed"`` — high-level theory result reported by the source
  (e.g. CCCBDB's CCSD(T)/aug-cc-pVTZ tabulations). Useful as a
  cross-check, NOT as method-validation ground truth — see
  ``docs/tutorial/external_data_fetcher.md`` § 3.
* ``"evaluated"`` — recommended value from a literature compilation
  where the source could be either or both (e.g. ATcT).
"""


@dataclass(frozen=True)
class ExperimentalReference:
    """Experimental / reference property record for one molecule.

    Maps 1-to-1 to a CCCBDB CAS-keyed entry. Every populated value
    carries an optional uncertainty (NIST publishes them; we record
    them). Each property field may be ``None`` when the source doesn't
    report it for this molecule — never assume coverage is complete.

    ``provenance`` is **required** (no default). The
    ``field(default_factory=lambda: ... .throw(...))`` idiom in the
    handover-doc draft was unsound (default_factory fires on every
    construction, not just when ``provenance`` is omitted); the
    cleaner contract is "schema-required, dataclass enforces at
    construction".
    """
    # ---- Identity --------------------------------------------------------
    cas: str
    """CAS registry number, hyphenated, e.g. ``"7732-18-5"`` for water.

    String, not int — leading zeros must be preserved
    (``"00057-13-6"`` for urea is not 57136).
    """
    formula: str
    """Hill-system molecular formula, e.g. ``"H2O"``, ``"CH4"``."""
    name: str
    """IUPAC / common name, e.g. ``"Water"``, ``"Methane"``."""
    system_id: Optional[str] = None
    """Cross-link to a :class:`MoleculeSpec`'s ``id`` field, or ``None``
    when the reference isn't yet attached to a regression-suite system."""
    kind: ReferenceKind = "experimental"

    # ---- Energetics ------------------------------------------------------
    # Atomization energy reported in kcal/mol per CCCBDB convention;
    # convert to Hartree at the use site (``unit_convert.kcal_per_mol_to_hartree``).
    atomization_energy_kcal_per_mol: Optional[float] = None
    atomization_energy_uncertainty_kcal_per_mol: Optional[float] = None

    # Enthalpy of formation at 298 K and 0 K (kJ/mol per CCCBDB / WebBook).
    enthalpy_of_formation_298_kj_per_mol: Optional[float] = None
    enthalpy_of_formation_298_uncertainty_kj_per_mol: Optional[float] = None
    enthalpy_of_formation_0_kj_per_mol: Optional[float] = None
    enthalpy_of_formation_0_uncertainty_kj_per_mol: Optional[float] = None

    # Ionization energy (eV — CCCBDB / WebBook convention).
    ionization_energy_ev: Optional[float] = None
    ionization_energy_uncertainty_ev: Optional[float] = None

    # Electron affinity (eV).
    electron_affinity_ev: Optional[float] = None
    electron_affinity_uncertainty_ev: Optional[float] = None

    # Proton affinity (kJ/mol).
    proton_affinity_kj_per_mol: Optional[float] = None

    # ---- Thermochemistry at 298.15 K ------------------------------------
    entropy_298_j_per_mol_per_k: Optional[float] = None
    heat_capacity_298_j_per_mol_per_k: Optional[float] = None

    # ---- Vibrational ----------------------------------------------------
    # Both fundamental (anharmonic) and harmonic when available; CCCBDB
    # tabulates them separately (don't conflate — see § 12 of the
    # handover doc). Tuple-of-floats keeps ordering meaningful
    # (lowest-to-highest frequency).
    vibrational_fundamentals_cm_inv: Tuple[float, ...] = ()
    vibrational_harmonics_cm_inv: Tuple[float, ...] = ()
    ir_intensities_km_per_mol: Tuple[float, ...] = ()

    # ---- Electrostatic --------------------------------------------------
    dipole_moment_debye: Optional[float] = None
    dipole_moment_uncertainty_debye: Optional[float] = None
    polarizability_au: Optional[float] = None
    """Isotropic mean polarizability in atomic units (a₀³)."""

    # ---- Geometry (bond list) -------------------------------------------
    # Nullable; fetch lazily because parsing is bond-specific and not
    # always needed by the caller. Atom labels follow CCCBDB's per-page
    # numbering convention (e.g. "O1", "H2", "H3").
    bond_lengths_ang: Tuple[Tuple[str, str, float], ...] = ()
    """Tuple of ``(atom_label_1, atom_label_2, length_angstrom)``."""
    bond_angles_deg: Tuple[Tuple[str, str, str, float], ...] = ()
    """Tuple of ``(atom_label_1, atom_label_2, atom_label_3, angle_deg)``."""
    cartesian_geometry_ang: Tuple[Tuple[str, int, float, float, float], ...] = ()
    """Per-atom Cartesian coordinates as ``(label, z, x, y, z_coord)`` —
    ``label`` is CCCBDB's per-page numbering (``"O1"``, ``"H2"``);
    ``z`` is the atomic number (0 if the symbol couldn't be inferred);
    ``x``/``y``/``z_coord`` are in Ångström.

    Lazy: ``()`` when the source didn't expose Cartesians (e.g. a
    bond-list-only entry). When populated, this is the input to
    :func:`vibeqc.fetch.references.experimental_geometry_to_molecule_spec`."""

    # ---- Provenance — REQUIRED, no default ------------------------------
    # Must be the last positional / keyword: dataclass requires that
    # fields-without-defaults precede fields-with-defaults. Swap order at
    # the call site (always pass ``provenance=...`` as keyword) and the
    # rule is harmless.
    provenance: "Provenance" = field(  # forward ref; defined above
        default=None,                  # type: ignore[assignment]
    )
    """Where the reference came from — set by the fetcher. Validated
    non-None at construction via ``__post_init__``."""

    def __post_init__(self) -> None:
        # We use a default of None on the field to satisfy dataclass's
        # "no defaultless field after a defaulted field" rule (every
        # property field above has a default), then enforce
        # required-ness here. Cleaner than the default_factory-throw
        # idiom in the handover-doc draft, which fires on EVERY
        # construction including legitimate ``provenance=...`` calls.
        if self.provenance is None:
            raise ValueError(
                "ExperimentalReference.provenance is required — pass "
                "Provenance(source_db='CCCBDB', source_id=<CAS>, ...). "
                "Hand-curated references may use Provenance(source_db="
                "'manual', ...). See "
                "docs/tutorial/external_data_fetcher.md § 5."
            )
