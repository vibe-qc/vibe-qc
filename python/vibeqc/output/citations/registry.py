"""Citation database registry.

Parses ``database.toml`` (and, on the ``basissetdev`` branch, the
sibling ``database_basissetdev.toml``) into an in-memory
:class:`CitationDatabase`. Exposes :meth:`assemble` which walks the
``[routes.*]`` tables for a job and returns an ordered, deduplicated
list of :class:`Citation` entries.

Routing semantics
-----------------

The routes table is hierarchical:

1. ``routes.software.always`` -- always cited, first.
2. ``routes.integrals.always`` -- always cited (libint).
3. ``routes.basis_sets[basis.lower()]`` -- keyed lookup; the basis-set
   name is normalised to lowercase (``"6-31g*"`` not ``"6-31G*"``).
   A miss is a routing gap -- reported via the ``warnings`` channel of
   the assembled result, and CI's ``test_citations_complete_coverage``
   fails if any bundled basis is unrouted.
4. ``routes.functionals._libxc_always`` (if a libxc-backed functional is
   set) + ``routes.functionals[functional.lower()]``. Names present in
   ``routes.external_functionals`` skip only the libxc citation.
5. ``routes.methods[<method-key>]`` for DIIS / EDIIS / D3 / D3BJ / D4
   / CCM / etc. -- triggered by the job options the caller passes in.
6. ``routes.dispersion_params["<disp>:<param-key>"]`` -- the damping-
   parameter *fit* paper, fired alongside the dispersion method route
   when the caller passes ``dispersion_params`` (the normalized
   functional / composite key whose damping set the run used). A miss
   is silent: parametrizations fit in the dispersion method paper
   itself carry no extra row.
7. ``routes.libraries[<lib>]`` for spglib / libecpint / FFTW3 / ASE,
   triggered when the corresponding subsystem ran.

Deduplication: every entry appears once in the assembled list, in
first-fire order, even if multiple routes pull it in.

Public API
----------

* :func:`load_default_database` -- bundled DB(s).
* :func:`load_database` -- load from an explicit path.
* :class:`CitationDatabase` -- parsed in-memory form.
* :class:`Citation` -- one reference, with BibTeX / plain-text helpers.
* :func:`assemble` -- convenience over ``db.assemble(...)``.
* :class:`DatabaseError` -- raised on malformed input.
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ...banner import VIBEQC_VERSION
from ..plan import OutputPlan

__all__ = [
    "Citation",
    "CitationDatabase",
    "DatabaseError",
    "AssembledCitations",
    "assemble",
    "citation_manifest_rows",
    "load_database",
    "load_default_database",
]


# Names registered through vibeqc.define_external_functional() in this Python
# process.  The citation database cannot invent a defining paper for an
# application-supplied callback, but it must not falsely cite libxc as that
# callback's numerical backend.  Shipped external functionals additionally
# carry durable markers and paper routes in database.toml.
_RUNTIME_EXTERNAL_FUNCTIONALS: set[str] = set()


def _register_runtime_external_functional(name: str) -> None:
    """Mark a successfully registered application-supplied XC backend."""

    _RUNTIME_EXTERNAL_FUNCTIONALS.add(name.strip().lower())


def _is_runtime_external_functional(name: str) -> bool:
    """Return whether the native process-lifetime registry owns ``name``.

    The Python marker is the fast path.  The native query also survives the
    package's supported ``sys.modules`` purge/re-import pattern, where Python
    module state is rebuilt while registered providers remain alive.
    """

    if name in _RUNTIME_EXTERNAL_FUNCTIONALS:
        return True
    try:
        from ... import _vibeqc_core

        lookup = getattr(
            _vibeqc_core, "_external_functional_registration", None
        )
        return callable(lookup) and lookup(name) is not None
    except (ImportError, RuntimeError, TypeError):
        return False


class DatabaseError(ValueError):
    """Raised when the citation database is malformed (missing
    required field, dangling route reference, schema-version
    mismatch)."""


# The DF-CCSD integral-assembly paper (DePrince-Sherrill 2013). Carried by
# every coupled-cluster method route (the DF route is the default); dropped
# by assemble(cc_density_fit=False) when the run used the canonical exact
# four-index integral route instead.
_DF_CC_ASSEMBLY_ENTRY = "deprince_sherrill_df_ccsd_2013"


# ---------------------------------------------------------------------- #
# Data model                                                             #
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class Citation:
    """One citable reference.

    Mirrors a ``[entries.<key>]`` block in ``database.toml`` after the
    template fields (``{{VIBEQC_VERSION}}``, ``{{VIBEQC_YEAR}}``) have
    been resolved against the running version.
    """

    key: str
    kind: str
    bibtex_key: str
    authors: tuple[str, ...]
    title: str
    journal: str | None = None
    volume: int | str | None = None
    issue: int | str | None = None
    pages: str | None = None
    publisher: str | None = None
    year: int | str | None = None
    doi: str | None = None
    url: str | None = None
    version: str | None = None
    license: str | None = None
    notes: str | None = None
    #: What the entry is cited *for* on the routes that fire it, when a
    #: route mixes provenance kinds: ``"parameter-source"`` or ``"method"``
    #: (see the database header). Provenance bookkeeping only: it is not
    #: rendered and not written to the manifest, so adding it changes no
    #: output byte.
    role: str | None = None
    print: bool = True

    def to_jsonable(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": self.key,
            "kind": self.kind,
            "bibtex_key": self.bibtex_key,
            "authors": list(self.authors),
            "title": self.title,
        }
        for fld in (
            "journal",
            "volume",
            "issue",
            "pages",
            "publisher",
            "year",
            "doi",
            "url",
            "version",
            "license",
            "notes",
        ):
            v = getattr(self, fld)
            if v is not None:
                out[fld] = v
        out["print"] = self.print
        return out


@dataclass(frozen=True)
class AssembledCitations:
    """Ordered, deduplicated list of :class:`Citation` entries plus
    routing warnings (e.g. ``"no route for basis 'foo'"``)."""

    citations: tuple[Citation, ...]
    warnings: tuple[str, ...] = ()

    def __iter__(self):
        return iter(self.citations)

    def __len__(self) -> int:
        return len(self.citations)

    def __bool__(self) -> bool:
        return bool(self.citations)

    @property
    def printable(self) -> tuple[Citation, ...]:
        """Citations marked for user-visible output (print=True)."""
        return tuple(c for c in self.citations if c.print)


def citation_manifest_rows(refs: Any) -> list[dict[str, Any]]:
    """Flatten an :class:`AssembledCitations` into the scalar-dict rows
    the ``.system`` manifest's ``[citations]`` section expects.

    Uses the *full* ``.citations`` view -- every entry regardless of its
    ``print`` flag (CLAUDE.md Sec.8.3/Sec.8.5: the ``.system`` manifest is the
    internal-provenance destination, so link-time-only entries like
    pybind11 belong here even though they are filtered out of the
    user-facing references block). Missing optional fields become ``""``
    sentinels so the array-of-tables keeps a fixed shape across rows.

    Shared by the molecular (``run_job``) and periodic
    (``run_periodic_job``) runners to feed
    :meth:`vibeqc.output.manifest.ManifestUpdater.set_citations`.
    """
    rows: list[dict[str, Any]] = []
    for c in getattr(refs, "citations", ()) or ():
        year = getattr(c, "year", None)
        rows.append(
            {
                "key": str(getattr(c, "key", "")),
                "kind": str(getattr(c, "kind", "")),
                "authors": "; ".join(getattr(c, "authors", ()) or ()),
                "title": str(getattr(c, "title", "")),
                "journal": str(getattr(c, "journal", None) or ""),
                "year": year if isinstance(year, int) else str(year or ""),
                "doi": str(getattr(c, "doi", None) or ""),
                "license": str(getattr(c, "license", None) or ""),
                "print": bool(getattr(c, "print", True)),
            }
        )
    return rows


# ---------------------------------------------------------------------- #
# Database                                                               #
# ---------------------------------------------------------------------- #


class CitationDatabase:
    """In-memory citation database."""

    SCHEMA_VERSION = "1"

    def __init__(
        self,
        *,
        entries: Mapping[str, Citation],
        routes: Mapping[str, Mapping[str, Sequence[str]]],
        source_paths: Sequence[Path] = (),
    ) -> None:
        self._entries: dict[str, Citation] = dict(entries)
        self._routes: dict[str, dict[str, tuple[str, ...]]] = {
            cat: {k: tuple(v) for k, v in body.items()} for cat, body in routes.items()
        }
        self._source_paths: tuple[Path, ...] = tuple(source_paths)
        self._validate_routes()

    # -- inspection --------------------------------------------------- #

    @property
    def source_paths(self) -> tuple[Path, ...]:
        return self._source_paths

    def entries(self) -> Mapping[str, Citation]:
        return self._entries

    def routes(self) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
        return self._routes

    # -- assembly ----------------------------------------------------- #

    def assemble(
        self,
        *,
        method: str | None = None,
        cc_density_fit: bool = True,
        basis: str | None = None,
        functional: str | None = None,
        dispersion: str | None = None,
        dispersion_params: str | None = None,
        scf_accelerator: str | None = None,
        uses_integrals: bool = True,
        uses_scf: bool = True,
        periodic: bool = False,
        uses_ecp: bool = False,
        uses_fftw_poisson: bool = False,
        uses_ase: bool = False,
        direct_scf: bool = False,
        uses_cpcm: bool = False,
        solvent_variant: str | None = None,
        pno_norm: str | None = None,
        extra_libraries: Iterable[str] = (),
        dft_plus_u: bool = False,
        uses_neb: bool = False,
        uses_ci_neb: bool = False,
        uses_dimer: bool = False,
        uses_irc: bool = False,
        uses_md: bool = False,
        md_thermostat: str | None = None,
        uses_metadynamics: bool = False,
        well_tempered: bool = False,
        uses_smearing: bool = False,
        uses_level_shift: bool = False,
        uses_gpw: bool = False,
        uses_gapw: bool = False,
        uses_gdf: bool = False,
        uses_gdf_2d: bool = False,
        uses_rsgdf: bool = False,
        uses_bipole: bool = False,
        uses_bipole_sr_range: bool = False,
        uses_ewald_ao_ft: bool = False,
        uses_slab_ewald_2d: bool = False,
        uses_soscf: bool = False,
        uses_trah: bool = False,
        uses_scf_stability: bool = False,
        uses_ml_kpredictor: bool = False,
        uses_tddft: bool = False,
        tddft_variant: str | None = None,
        tddft_hybrid_kernel: bool = False,
        uses_cis: bool = False,
        uses_conical_intersection: bool = False,
        uses_gradient: bool = False,
        uses_hessian: bool = False,
        geom_optimizer: str | None = None,
        geom_coords: str | None = None,
        smearing_method: str | None = None,
        electronic_temperature: float | None = None,
        seccm_dimension: int | None = None,
        seccm_electrostatics_kernel: str | None = None,
        scf_guess: str | None = None,
        properties: Iterable[str] = (),
        acceleration: Iterable[str] = (),
        numerics: Iterable[str] = (),
        extra_entries: Iterable[str] = (),
    ) -> AssembledCitations:
        """Walk the routes table for a job and return the assembled
        list of citations.

        Parameters
        ----------
        method
            ``"RHF"`` / ``"UHF"`` / ``"RKS"`` / ``"UKS"`` / ``"CCSD"``
            / ``"CCSD(T)"`` / ``"FCI"`` / a composite-3c keyword
            (``"hf-3c"``, ``"pbeh-3c"``, ...) / etc. Case-insensitive;
            looked up against ``routes.methods``. A miss is *silent*
            -- mean-field methods (RHF / UHF / RKS / UKS) correctly
            have no method-specific route (they are covered by the
            integral library + the functional). Only post-SCF and
            composite methods carry their own defining-paper routes.
        cc_density_fit
            True (default) when a coupled-cluster method ran on the
            density-fitted integral route. Set False when the run used
            the canonical exact four-index route
            (``CCSDOptions(density_fit=False)``): the DF-CCSD
            integral-assembly citation (DePrince-Sherrill 2013) is then
            dropped from the method bundle, since no density fitting of
            the CC integrals took place. Ignored for non-CC methods
            (their routes do not carry the entry).
        basis
            Basis-set name as user-facing string (case-insensitive
            lookup).
        functional
            XC functional for KS-DFT (case-insensitive). When set,
            ``routes.functionals._libxc_always`` fires automatically unless
            the normalized name is present in
            ``routes.external_functionals``.
        dispersion
            ``"d3"`` / ``"d3bj"`` / ``"d4"`` -- keyed lookup into
            ``routes.methods``.
        dispersion_params
            Normalized key of the damping-parameter set the run
            actually used (the functional or composite name, e.g.
            ``"r2scan"`` / ``"revdsdpbep86"`` / ``"r2scan3c"`` -- for
            D4, normalize with
            :func:`vibeqc.dispersion_d4_parameters.normalize_d4_key`).
            Fires ``routes.dispersion_params["<dispersion>:<key>"]``
            *in addition* to the dispersion method route, so a
            parametrization whose fit was published separately (e.g.
            revDSD-PBEP86-D4 -> Santra-Sylvetsky-Martin 2019) cites
            its fit paper. Ignored unless ``dispersion`` is set; a
            lookup miss is silent (most parameter sets come from the
            dispersion method paper itself).
        scf_accelerator
            ``"diis"`` / ``"ediis"`` -- keyed lookup. None => default
            (DIIS) cite.
        uses_integrals
            True (default) => the always-on libint integral-library
            citation fires. Set False for engines that evaluate no
            Gaussian integrals -- e.g. an MLIP (``method="mace"``),
            whose energy comes from a pre-trained model, not from
            integrals over a Gaussian basis.
        uses_scf
            True (default) => the SCF-accelerator citation fires
            (DIIS by default). Set False for engines that run no SCF
            (again, the MLIP path) so DIIS is not credited spuriously.
        periodic
            True => spglib is cited.
        uses_ecp
            True => libecpint is cited.
        uses_fftw_poisson
            True => FFTW3 is cited (the FFT-Poisson backend was used).
        uses_ewald_ao_ft
            True => the analytical AO-pair Fourier-transform references
            for the EWALD_3D Hartree J backend fire.
        uses_bipole
            True => the BIPOLE periodic-Coulomb methodology references
            fire.
        uses_ase
            True => ASE is cited (BFGS optimisation or Calculator).
        uses_cpcm
            True => the CPCM solvation routes fire (Klamt-Schüürmann
            1993 + Cossi-Rega-Scalmani-Barone 2003 + Scalmani-Frisch
            2010). Set by ``run_job`` when ``solvent`` is given, and
            by ``run_cpcm_scf`` directly.
        solvent_variant
            ``"cpcm"`` (default when uses_cpcm=True) or ``"cosmo"``;
            routes through ``routes.solvation[<variant>]``.
        extra_libraries
            Additional library keys to route -- useful for tutorials
            that want to surface, e.g., Eigen by name.
        dft_plus_u
            True => fire ``routes.methods.dft_plus_u`` (Dudarev 1998 +
            Cococcioni-Gironcoli 2005). Set by callers that ran an SCF
            with a non-empty ``RHFOptions.dft_plus_u_sites`` (or the
            UHF/RKS/UKS equivalents once those land in Increment 2c+).
        uses_neb
            True => fire ``routes.drivers.neb`` (Henkelman+Jónsson 2000
            improved-tangent NEB + Smidstrup 2014 IDPP). Set by
            callers that drove a NEB run end-to-end via
            :func:`vibeqc.run_neb`. The per-image SCF method /
            functional / basis citations fire through their normal
            routes.
        uses_ci_neb
            True => fire ``routes.drivers.ci_neb`` (Henkelman+Uberuaga
            +Jónsson 2000 climbing-image NEB), in *addition* to the
            ``uses_neb`` route. Set when
            :func:`vibeqc.run_neb` ran with ``climbing_image=True``.
        uses_md
            True => fire ``routes.drivers.md`` (the Swope 1982 velocity-
            Verlet integrator) for any :func:`vibeqc.md.run_md` run.
        md_thermostat
            ``"berendsen"`` / ``"nose_hoover"`` => additionally fire the
            matching NVT-thermostat route (Berendsen 1984, or Nosé 1984 +
            Hoover 1985). ``None`` => NVE (no thermostat citation).
        uses_metadynamics
            True => fire ``routes.drivers.metadynamics`` (Laio-Parrinello
            2002) for any :func:`vibeqc.metadynamics.run_metadynamics` run.
        well_tempered
            True => additionally fire
            ``routes.drivers.well_tempered_metadynamics``
            (Barducci-Bussi-Parrinello 2008).
        uses_smearing
            True => fire the finite-temperature smearing route.
        electronic_temperature
            Runtime electronic temperature in Hartree. A positive value
            fires the Mermin finite-temperature route even when the caller
            did not also set the legacy ``uses_smearing`` flag.
        seccm_dimension
            Number of active SECCM translation dimensions (1, 2, or 3).
            Used together with ``seccm_electrostatics_kernel`` so slab
            references fire only for the concrete 2-D kernel.
        seccm_electrostatics_kernel
            Concrete SECCM kernel label recorded by the route plan, such as
            ``"madelung_parry_2d"`` or ``"ewald_gamma_ewald_3d"``.
        """
        warnings: list[str] = []
        seen_keys: list[str] = []  # preserves first-fire order
        seen_set: set[str] = set()

        def _add(keys: Sequence[str], origin: str) -> None:
            for k in keys:
                if k in seen_set:
                    continue
                if k not in self._entries:
                    warnings.append(f"route {origin!r} references missing entry {k!r}")
                    continue
                seen_set.add(k)
                seen_keys.append(k)

        # 1. Software (always).
        _add(self._routes.get("software", {}).get("always", ()), "software.always")

        # 2. Integrals (always -- libint). Suppressed for engines that
        # evaluate no Gaussian integrals (e.g. an MLIP, method="mace").
        if uses_integrals:
            _add(
                self._routes.get("integrals", {}).get("always", ()), "integrals.always"
            )

        # 3. Basis set.
        if basis:
            b_key = basis.strip().lower()
            row = self._routes.get("basis_sets", {}).get(b_key)
            filtered_match = re.fullmatch(
                r"_vibeqc_filtered_(.+)_[0-9a-f]{8}", b_key
            )
            if row is None and filtered_match is not None:
                # Default basis filtering preserves the original name inside
                # a deterministic synthetic name.  Cite that source basis;
                # filtering primitives does not create a new basis family.
                b_key = filtered_match.group(1)
                row = self._routes.get("basis_sets", {}).get(b_key)
            if row is None:
                warnings.append(f"no citation route for basis set {basis!r}")
            else:
                _add(row, f"basis_sets[{b_key!r}]")
                # The bundled basis library is sourced from the Basis Set
                # Exchange (offline, at build time). Any job that resolves a
                # bundled basis -- i.e. the name matched a routed entry --
                # therefore cites BSE for the library it drew from. A custom /
                # external basis is a route miss above and does not fire this.
                _add(
                    self._routes.get("basis_sets", {}).get(
                        "_bse_bundled_provenance", ()
                    ),
                    "basis_sets._bse_bundled_provenance",
                )

        # 4. Functional (+ its backend + the numerical integration grid). Any
        # KS-DFT run evaluates the XC energy on a Becke-partitioned atomic grid
        # (Becke 1988 + Treutler-Ahlrichs 1995), so ``_grid_always`` fires
        # whenever a functional is set. libxc fires only for names absent from
        # the external-functional marker table.
        if functional:
            f_key = functional.strip().lower()
            if (
                f_key not in self._routes.get("external_functionals", {})
                and not _is_runtime_external_functional(f_key)
            ):
                _add(
                    self._routes.get("functionals", {}).get("_libxc_always", ()),
                    "functionals._libxc_always",
                )
            _add(
                self._routes.get("functionals", {}).get("_grid_always", ()),
                "functionals._grid_always",
            )
            row = self._routes.get("functionals", {}).get(f_key)
            if row is None:
                warnings.append(f"no citation route for functional {functional!r}")
            else:
                _add(row, f"functionals[{f_key!r}]")

        # 5. SCF accelerator (default DIIS). Suppressed for engines that
        # run no SCF (e.g. an MLIP, method="mace"); ``accel`` stays None
        # so the step-5b method route still fires.
        accel = (scf_accelerator or "diis").strip().lower() if uses_scf else None
        if accel is not None:
            row = self._routes.get("methods", {}).get(accel)
            if row is not None:
                _add(row, f"methods[{accel!r}]")

        # 5b. Post-SCF / composite method (CCSD, CCSD(T), FCI,
        # hf-3c, pbeh-3c, ...). Keyed lookup into routes.methods by
        # the lowercased method name. A miss is silent on purpose:
        # the mean-field methods (rhf / uhf / rks / uks) have no
        # method-specific route -- their citations come from the
        # integral library + the functional. Only post-SCF and
        # composite methods carry a defining-paper route, and the
        # test_citations.py _REQUIRED_METHODS gate enforces that
        # every such method that ships actually has one.
        if method:
            m_key = method.strip().lower()
            # Don't double-fire the SCF-accelerator / dispersion
            # keys if a method happened to collide with one.
            if m_key not in (accel,):
                row = self._routes.get("methods", {}).get(m_key)
                if row is not None:
                    if not cc_density_fit:
                        # Canonical (non-DF) coupled cluster: the run
                        # assembled exact four-index integrals, so the
                        # DF-CCSD assembly paper is not part of what it
                        # used. Truthful-citation gate (CLAUDE.md sec. 8).
                        row = [
                            k for k in row
                            if k != _DF_CC_ASSEMBLY_ENTRY
                        ]
                    _add(row, f"methods[{m_key!r}]")

        # 6. Dispersion.
        if dispersion:
            d_key = dispersion.strip().lower()
            row = self._routes.get("methods", {}).get(d_key)
            if row is None:
                warnings.append(f"no citation route for dispersion {dispersion!r}")
            else:
                _add(row, f"methods[{d_key!r}]")
            # 6a. Damping-parameter fit paper. Fires when the parameter
            # set the run used was fit in a publication separate from
            # the dispersion method paper (routes.dispersion_params,
            # keyed "<disp>:<parametrization>"). A miss is silent by
            # design -- parametrizations from the method paper itself
            # have no extra row.
            if dispersion_params:
                p_key = f"{d_key}:{str(dispersion_params).strip().lower()}"
                p_row = self._routes.get("dispersion_params", {}).get(p_key)
                if p_row is not None:
                    _add(p_row, f"dispersion_params[{p_key!r}]")

        # 6b. DFT+U (Dudarev). Fired when the SCF was run with a
        # non-empty Hubbard-site list. The route key is "dft_plus_u";
        # the entries are Dudarev 1998 (rotationally-invariant
        # formalism) + Cococcioni-Gironcoli 2005 (linear-response U).
        if dft_plus_u:
            row = self._routes.get("methods", {}).get("dft_plus_u")
            if row is not None:
                _add(row, "methods.dft_plus_u")

        # 7. Direct SCF (integral-driven Fock build).
        if direct_scf:
            row = self._routes.get("methods", {}).get("direct_scf")
            if row is not None:
                _add(row, "methods.direct_scf")

        # 7b. Finite-temperature occupations / Mermin free energy. The
        # Mermin functional fires for any smeared run; a non-Fermi-Dirac
        # broadening (Methfessel-Paxton, Marzari-Vanderbilt, Gaussian)
        # additionally cites its defining paper via smearing_method.
        if uses_smearing or (
            electronic_temperature is not None
            and float(electronic_temperature) > 0.0
        ):
            row = self._routes.get("methods", {}).get("fermi_dirac_smearing")
            if row is not None:
                _add(row, "methods.fermi_dirac_smearing")
            if smearing_method:
                s_key = smearing_method.strip().lower()
                if s_key not in ("fermi_dirac", "fermi-dirac", "fd"):
                    s_row = self._routes.get("methods", {}).get(s_key)
                    if s_row is not None:
                        _add(s_row, f"methods[{s_key!r}]")

        # 7b'. Saunders-Hillier level shift. Fires for any job run with a
        # non-zero level_shift or an explicit level_shift_schedule
        # (molecular or periodic). The shift is inert at the converged
        # density but is a cited convergence technique.
        if uses_level_shift:
            row = self._routes.get("methods", {}).get("level_shift")
            if row is not None:
                _add(row, "methods.level_shift")

        # 7c. GPW / GAPW Gaussian-plane-wave route (Lippert-Hutter 1997
        # + Quickstep / VandeVondele-Hutter 2005 for the CP2K-style
        # M3b convention + GAPW augmentation). Fired by the periodic
        # runner when ``jk_method='gpw'`` (or 'gapw' once the augmentation
        # SCF lands). The Mura-Knowles entry also lands on the GAPW
        # path because the per-atom radial grids that the augmentation
        # consumes use that transform.
        if uses_gpw:
            row = self._routes.get("methods", {}).get("gpw")
            if row is not None:
                _add(row, "methods.gpw")
        if uses_gapw:
            row = self._routes.get("methods", {}).get("gapw")
            if row is not None:
                _add(row, "methods.gapw")

        # 7d. GDF / RSGDF Gaussian-density-fitting periodic route. Sun-
        # Berkelbach 2017 (the periodic-AO-pair-FT recursion) + the
        # McMurchie-Davidson 1978 and Helgaker-Jorgensen-Olsen 2000
        # molecular foundations fire whenever ``jk_method='gdf'`` or
        # ``run_pbc_gdf_rhf`` is exercised. RSGDF additionally routes
        # to its own row so the GDF chat can extend it with the
        # Ye-Berkelbach 2021 range-separation citation when it lands.
        if uses_gdf:
            row = self._routes.get("methods", {}).get("gdf")
            if row is not None:
                _add(row, "methods.gdf")
        if uses_gdf_2d:
            row = self._routes.get("methods", {}).get("gdf_2d")
            if row is not None:
                _add(row, "methods.gdf_2d")
        if uses_rsgdf:
            row = self._routes.get("methods", {}).get("rsgdf")
            if row is not None:
                _add(row, "methods.rsgdf")
        if uses_bipole:
            row = self._routes.get("methods", {}).get("bipole")
            if row is not None:
                _add(row, "methods.bipole")
        # M4b charge-pair separation-aware SR screening (opt-in
        # LatticeSumOptions.sr_range_screening on the BIPOLE direct
        # erfc J/K build).
        if uses_bipole_sr_range:
            row = self._routes.get("methods", {}).get("bipole_sr_range")
            if row is not None:
                _add(row, "methods.bipole_sr_range")
        if uses_ewald_ao_ft:
            row = self._routes.get("methods", {}).get("ewald_ao_ft")
            if row is not None:
                _add(row, "methods.ewald_ao_ft")
        # Vacuum-free 2D (slab) Coulomb: the rigorous Parry / de Leeuw-Perram
        # gauge. Fires whenever a dim=2 slab runs (AUTO resolves there) or
        # ``jk_method='slab_ewald_2d'`` is requested explicitly.
        seccm_uses_slab_ewald_2d = (
            seccm_dimension == 2
            and (seccm_electrostatics_kernel or "").endswith("parry_2d")
        )
        if uses_slab_ewald_2d or seccm_uses_slab_ewald_2d:
            row = self._routes.get("methods", {}).get("slab_ewald_2d")
            if row is not None:
                _add(row, "methods.slab_ewald_2d")
        # 1-D background-corrected Parry-type wire Ewald (cpp/src/
        # semiempirical/seccm/ewald_1d.h). Fires for SECCM 1-D runs
        # whose electrostatics kernel resolves to *_wire_1d - the
        # madelung embedding and the self-consistent ewald_gamma shell
        # kernel alike. The MSINDO engine is excluded by construction:
        # its frozen 1-D kernel is the truncated bare lattice sum the
        # wire Ewald replaced, so the adapter names it *_truncated_1d
        # and it never reaches this route (GitLab #442).
        seccm_uses_wire_ewald_1d = (
            seccm_dimension == 1
            and (seccm_electrostatics_kernel or "").endswith("wire_1d")
        )
        if seccm_uses_wire_ewald_1d:
            row = self._routes.get("methods", {}).get("wire_ewald_1d")
            if row is not None:
                _add(row, "methods.wire_ewald_1d")
        # CCM Madelung embedding: fires whenever the resolved SECCM
        # electrostatics kernel is a madelung_* one, for any adapter.
        # Kernel-driven like its siblings above, so a route that does not
        # run the embedding never cites it. Two rows, by what the kernel
        # actually computes (GitLab vibe-qc#64): the MSINDO engine's 1-D
        # kernel is the frozen truncated +/-2-shell bare point-charge sum
        # (indo::_madelung_potential_1d), which owes the CCM construction
        # reference but performs no Ewald split, so it must not be credited
        # with the Janetzko 2002 exact Madelung matrices or Ewald 1921.
        # Every other madelung_* kernel (parry_2d, ewald_3d, wire_1d) runs
        # an exact operator and keeps the full embedding row.
        _madelung_kernel = seccm_electrostatics_kernel or ""
        if _madelung_kernel.startswith("madelung_"):
            _madelung_route = (
                "ccm_madelung_truncated_1d"
                if _madelung_kernel.endswith("truncated_1d")
                else "ccm_madelung_embedding"
            )
            row = self._routes.get("methods", {}).get(_madelung_route)
            if row is not None:
                _add(row, f"methods.{_madelung_route}")

        # 7e. Second-order SCF convergence (SOSCF / TRAH). Engaged by the
        # RHFOptions.soscf_threshold / trah_threshold paths (cpp/src/
        # soscf.hpp + trah.hpp). Routed through routes.methods.
        if uses_soscf:
            row = self._routes.get("methods", {}).get("soscf")
            if row is not None:
                _add(row, "methods.soscf")
        if uses_trah:
            row = self._routes.get("methods", {}).get("trah")
            if row is not None:
                _add(row, "methods.trah")
        # 7e''. Internal SCF stability analysis (molecular UHF/UKS where the
        # response is complete; cpp/src/newton.cpp plus uhf.cpp/uks.cpp). The
        # runner sets the flag from the returned stability_checked verdict.
        if uses_scf_stability:
            row = self._routes.get("methods", {}).get("scf_stability")
            if row is not None:
                _add(row, "methods.scf_stability")
            if (method or "").strip().lower() == "uks":
                row = self._routes.get("methods", {}).get(
                    "uks_scf_stability"
                )
                if row is not None:
                    _add(row, "methods.uks_scf_stability")

        # 7e'. Machine-learned k-spacing predictor (K8-F).  Fired when
        # KPoints.recommend(predictor="ml") uses the bundled random-forest
        # model (Choudhary & Tavazza 2020).  The periodic runner sets this
        # flag by reading .uses_ml_predictor from the KPoints object.
        if uses_ml_kpredictor:
            row = self._routes.get("methods", {}).get("ml_kpoints")
            if row is not None:
                _add(row, "methods.ml_kpoints")

        # 7f. Linear-response TDDFT (python/vibeqc/tddft.py). The base
        # ``tddft`` row (Runge-Gross + Casida) fires for any run; the
        # ``tddft_tda`` row fires *in addition* when the Tamm-Dancoff
        # approximation is used (tddft_variant='tda'), and the
        # ``tddft_hybrid`` row (Bauernschmitt-Ahlrichs 1996) fires *in
        # addition* when the response kernel carries a hybrid
        # functional's exact-exchange admixture (tddft_hybrid_kernel).
        if uses_tddft:
            row = self._routes.get("drivers", {}).get("tddft")
            if row is None:
                warnings.append("no citation route for driver 'tddft'")
            else:
                _add(row, "drivers.tddft")
            if (tddft_variant or "").strip().lower() == "tda":
                tda = self._routes.get("drivers", {}).get("tddft_tda")
                if tda is not None:
                    _add(tda, "drivers.tddft_tda")
            if tddft_hybrid_kernel:
                hyb = self._routes.get("drivers", {}).get("tddft_hybrid")
                if hyb is not None:
                    _add(hyb, "drivers.tddft_hybrid")

        # 7f'. Wavefunction CIS excited states + CIS excited-state nuclear
        # gradients (vibeqc.excited / msindo_cis / vibeqc.excited_gradient).
        # Foresman-Head-Gordon-Pople-Frisch 1992. Separate from the TDDFT rows:
        # CIS is the variational wavefunction excited state, not linear response.
        if uses_cis:
            row = self._routes.get("drivers", {}).get("cis")
            if row is None:
                warnings.append("no citation route for driver 'cis'")
            else:
                _add(row, "drivers.cis")

        # 7f''. Penalty-function minimum-energy conical-intersection optimizer
        # (vibeqc.conical). Levine-Coe-Martínez 2008 -- locates a MECI from two
        # state gradients without a derivative coupling vector.
        if uses_conical_intersection:
            row = self._routes.get("drivers", {}).get("conical_intersection")
            if row is None:
                warnings.append("no citation route for driver 'conical_intersection'")
            else:
                _add(row, "drivers.conical_intersection")

        # 7g. SCF initial guess (SAD / SAP / GWH / extended-Hückel / ...).
        # Routed through routes.scf_guess keyed by the guess name. Hcore is
        # the traditional default and carries no defining-paper route.
        if scf_guess:
            g_key = scf_guess.strip().lower()
            row = self._routes.get("scf_guess", {}).get(g_key)
            if row is not None:
                _add(row, f"scf_guess[{g_key!r}]")

        # 7h. Properties / analysis (population analysis, NMR, response,
        # topological analysis, ...). Routed through routes.properties; the
        # caller passes the list of property keys a job actually computed.
        for prop in properties:
            p_key = str(prop).strip().lower()
            if not p_key:
                continue
            row = self._routes.get("properties", {}).get(p_key)
            if row is not None:
                _add(row, f"properties[{p_key!r}]")

        # 7i. Integral / exchange acceleration techniques (RI-J, RIJCOSX,
        # LinK/sn-LinK, FMM, ADMM, ACE, PAW, ...). Routed through
        # routes.acceleration; the caller passes the techniques a job used.
        for acc in acceleration:
            a_key = str(acc).strip().lower()
            if not a_key:
                continue
            row = self._routes.get("acceleration", {}).get(a_key)
            if row is not None:
                _add(row, f"acceleration[{a_key!r}]")

        # 7j. Numerical-stability machinery (canonical / pivoted-Cholesky
        # orthogonalization for overcomplete bases). Routed through
        # routes.numerics.
        for num in numerics:
            n_key = str(num).strip().lower()
            if not n_key:
                continue
            row = self._routes.get("numerics", {}).get(n_key)
            if row is not None:
                _add(row, f"numerics[{n_key!r}]")

        # 7k. Analytic gradient (Pulay forces + Hellmann-Feynman) and
        # Hessian / vibrational analysis drivers.
        if uses_gradient:
            row = self._routes.get("drivers", {}).get("gradient")
            if row is not None:
                _add(row, "drivers.gradient")
        if uses_hessian:
            row = self._routes.get("drivers", {}).get("hessian")
            if row is not None:
                _add(row, "drivers.hessian")

        # 7l. Geometry-optimization algorithm + coordinate system. The
        # keyword-selected optimizer (vibeqc.geomopt, run_job(geom_opt=...))
        # fires its defining-paper route via routes.optimizers; the
        # coordinate system (run_job(geom_coords=...)) via routes.coordinates.
        # Both misses are silent on purpose: Cartesian coordinates and the
        # textbook sd / cg / (L-)BFGS optimizers carry no defining-paper row
        # (BFGS is covered by the ASE citation + the Pulay-forces gradient
        # route). RFO / P-RFO / GDIIS / FIRE / trust-region and the
        # delocalized-internal coordinates each carry their own.
        if geom_optimizer:
            o_key = geom_optimizer.strip().lower()
            row = self._routes.get("optimizers", {}).get(o_key)
            if row is not None:
                _add(row, f"optimizers[{o_key!r}]")
        if geom_coords:
            c_key = geom_coords.strip().lower()
            row = self._routes.get("coordinates", {}).get(c_key)
            if row is not None:
                _add(row, f"coordinates[{c_key!r}]")

        # 8. Libraries (conditional, except Eigen).
        libs = self._routes.get("libraries", {})
        # Eigen is linked into the C++ core for every job, so it is
        # unconditional. Its entry sets print = false: link-time
        # infrastructure belongs in the .system manifest as provenance, not in
        # the references block a paper's Methods section copies from.
        _add(libs.get("eigen", ()), "libraries.eigen")
        if periodic:
            _add(libs.get("spglib", ()), "libraries.spglib")
            # Foundational monograph for the LCAO crystalline-orbital periodic
            # HF treatment vibe-qc implements. Fires for any periodic job (the
            # crystalline-orbital reference underlies the periodic post-HF
            # methods too), the periodic analogue of an always-on method paper.
            _add(
                self._routes.get("methods", {}).get("_periodic_lcao", ()),
                "methods._periodic_lcao",
            )
        if uses_ecp:
            _add(libs.get("libecpint", ()), "libraries.libecpint")
        if uses_fftw_poisson:
            _add(libs.get("fftw3", ()), "libraries.fftw3")
        if uses_ase:
            _add(libs.get("ase", ()), "libraries.ase")
        for lib in extra_libraries:
            row = libs.get(lib.strip().lower())
            if row is not None:
                _add(row, f"libraries[{lib!r}]")

        # 8b. NEB driver. Fires the two-paper NEB bundle when the
        # caller drove a NEB run (run_neb sets this at the end of a
        # converged or maxed-out run). Per-image SCF citations are
        # handled separately by their normal method / basis routes.
        # CI-NEB fires *in addition* -- the climbing-image paper covers
        # only that variant; the base improved-tangent + IDPP papers
        # still apply.
        if uses_neb:
            row = self._routes.get("drivers", {}).get("neb")
            if row is None:
                warnings.append("no citation route for driver 'neb'")
            else:
                _add(row, "drivers.neb")
        if uses_ci_neb:
            row = self._routes.get("drivers", {}).get("ci_neb")
            if row is None:
                warnings.append("no citation route for driver 'ci_neb'")
            else:
                _add(row, "drivers.ci_neb")
        if uses_dimer:
            row = self._routes.get("drivers", {}).get("dimer")
            if row is None:
                warnings.append("no citation route for driver 'dimer'")
            else:
                _add(row, "drivers.dimer")
        if uses_irc:
            row = self._routes.get("drivers", {}).get("irc")
            if row is None:
                warnings.append("no citation route for driver 'irc'")
            else:
                _add(row, "drivers.irc")
        # Molecular dynamics (vibeqc.md). The velocity-Verlet ``md`` row
        # always fires; the thermostat-specific row fires *in addition*
        # when md_thermostat names a Berendsen / Nosé-Hoover NVT run.
        if uses_md:
            row = self._routes.get("drivers", {}).get("md")
            if row is None:
                warnings.append("no citation route for driver 'md'")
            else:
                _add(row, "drivers.md")
            if md_thermostat:
                tkey = {
                    "berendsen": "md_berendsen",
                    "nose_hoover": "md_nose_hoover",
                    "nosehoover": "md_nose_hoover",
                    "nose": "md_nose_hoover",
                }.get(str(md_thermostat).strip().lower().replace("-", "_"))
                if tkey:
                    trow = self._routes.get("drivers", {}).get(tkey)
                    if trow is None:
                        warnings.append(f"no citation route for driver '{tkey}'")
                    else:
                        _add(trow, f"drivers.{tkey}")
        # Metadynamics (vibeqc.metadynamics). The base ``metadynamics``
        # row fires for any run; ``well_tempered_metadynamics`` fires *in
        # addition* for the well-tempered variant (the only one shipped).
        if uses_metadynamics:
            row = self._routes.get("drivers", {}).get("metadynamics")
            if row is None:
                warnings.append("no citation route for driver 'metadynamics'")
            else:
                _add(row, "drivers.metadynamics")
            if well_tempered:
                wt = self._routes.get("drivers", {}).get("well_tempered_metadynamics")
                if wt is None:
                    warnings.append(
                        "no citation route for driver 'well_tempered_metadynamics'"
                    )
                else:
                    _add(wt, "drivers.well_tempered_metadynamics")

        # 9. Solvation (CPCM / COSMO). The runtime sets uses_cpcm=True
        # when the molecular runner observes a non-None ``solvent`` kwarg
        # (or when ``run_cpcm_scf`` emits its own citations). The variant
        # defaults to "cpcm" -- vibe-qc's only shipped solvent model -- but
        # ``solvent_variant="cosmo"`` is accepted for future drivers.
        if uses_cpcm:
            variant = (solvent_variant or "cpcm").strip().lower()
            row = self._routes.get("solvation", {}).get(variant)
            if row is None:
                warnings.append(f"no citation route for solvent variant {variant!r}")
            else:
                _add(row, f"solvation[{variant!r}]")

        # 9b. DLPNO pair-density convention. ``tcut_pno`` is a cut on the
        # eigenvalues of this density, so the convention is part of the method
        # actually run. "legacy" is vibe-qc's own density and deliberately has
        # no route row, so it contributes nothing and warns nothing.
        if pno_norm:
            key = str(pno_norm).strip().lower()
            row = self._routes.get("pno_norm", {}).get(key)
            if row is not None:
                _add(row, f"pno_norm[{key!r}]")

        # 10. Extra entry keys supplied directly by the caller (not via a
        # route) -- e.g. the specific MLIP foundation-model paper for the
        # model actually selected at runtime (vibeqc.mlip), which depends
        # on a runtime choice rather than a static method route. Empty
        # strings are skipped (an unregistered MLIP model has no pinned
        # foundation citation).
        if extra_entries:
            _add([str(k) for k in extra_entries if k], "extra_entries")

        citations = tuple(self._entries[k] for k in seen_keys)
        return AssembledCitations(
            citations=citations,
            warnings=tuple(warnings),
        )

    def assemble_from_plan(
        self, plan: OutputPlan, **overrides: Any
    ) -> AssembledCitations:
        """Convenience: pull ``method`` / ``basis`` / ``functional``
        from the plan, with optional per-call overrides."""
        return self.assemble(
            method=overrides.pop("method", plan.method),
            basis=overrides.pop("basis", plan.basis),
            functional=overrides.pop("functional", plan.functional),
            **overrides,
        )

    # -- internals ---------------------------------------------------- #

    def _validate_routes(self) -> None:
        """Sanity-check that every route key references an existing
        entry. Catches typos in ``database.toml`` at load time rather
        than at assembly time."""
        for category, body in self._routes.items():
            for route_key, entry_keys in body.items():
                for entry_key in entry_keys:
                    if entry_key not in self._entries:
                        raise DatabaseError(
                            f"route {category}.{route_key!r} references "
                            f"missing entry {entry_key!r}"
                        )


# ---------------------------------------------------------------------- #
# Loaders                                                                #
# ---------------------------------------------------------------------- #


def load_database(*paths: os.PathLike | str) -> CitationDatabase:
    """Load and merge one or more TOML database files.

    When multiple paths are given, entries / routes from later files
    layer on top of earlier ones (an entry key collision is an error;
    a route key collision replaces). This is the mechanism by which
    ``basissetdev`` adds its 87 basis-set citations on top of the
    main-branch database without forking the schema.
    """
    if not paths:
        raise DatabaseError("load_database() called with no paths")

    entries: dict[str, Citation] = {}
    routes: dict[str, dict[str, list[str]]] = {}
    resolved: list[Path] = []

    for raw in paths:
        path = Path(os.fspath(raw))
        if not path.is_file():
            raise DatabaseError(f"database file not found: {path}")
        resolved.append(path)
        with path.open("rb") as f:
            data = tomllib.load(f)

        schema = str(data.get("schema_version", ""))
        if schema and schema != CitationDatabase.SCHEMA_VERSION:
            raise DatabaseError(
                f"database {path} has schema_version={schema!r}, "
                f"this build expects "
                f"{CitationDatabase.SCHEMA_VERSION!r}"
            )

        raw_entries = data.get("entries", {})
        if not isinstance(raw_entries, dict):
            raise DatabaseError(f"database {path}: [entries] must be a table")
        for key, blob in raw_entries.items():
            if key in entries:
                raise DatabaseError(
                    f"duplicate entry {key!r} (already loaded from "
                    f"a previous database file)"
                )
            entries[key] = _entry_from_toml(key, blob)

        raw_routes = data.get("routes", {})
        if not isinstance(raw_routes, dict):
            raise DatabaseError(f"database {path}: [routes] must be a table")
        for category, body in raw_routes.items():
            cat = routes.setdefault(category, {})
            if not isinstance(body, dict):
                raise DatabaseError(
                    f"database {path}: routes.{category} must be a table"
                )
            for route_key, entry_keys in body.items():
                if not isinstance(entry_keys, list):
                    raise DatabaseError(
                        f"database {path}: "
                        f"routes.{category}.{route_key} must be a "
                        f"list of entry keys"
                    )
                cat[route_key.strip().lower()] = [str(k) for k in entry_keys]

    return CitationDatabase(
        entries=entries,
        routes=routes,
        source_paths=resolved,
    )


def load_default_database() -> CitationDatabase:
    """Load the bundled ``database.toml``.

    On the ``basissetdev`` branch, also loads
    ``database_basissetdev.toml`` next to it (if present) so the
    extra 87 basis-set routes overlay the main-branch coverage.
    """
    here = Path(__file__).resolve().parent
    paths: list[Path] = [here / "database.toml"]
    extra = here / "database_basissetdev.toml"
    if extra.is_file():
        paths.append(extra)
    return load_database(*paths)


def assemble(plan: OutputPlan, **overrides: Any) -> AssembledCitations:
    """Convenience wrapper around
    :meth:`CitationDatabase.assemble_from_plan` that uses the bundled
    default database. The full-featured entry point is
    :meth:`CitationDatabase.assemble`."""
    db = load_default_database()
    return db.assemble_from_plan(plan, **overrides)


# ---------------------------------------------------------------------- #
# TOML -> Citation conversion                                             #
# ---------------------------------------------------------------------- #

_REQUIRED_FIELDS = ("kind", "bibtex_key", "authors", "title")


def _entry_from_toml(key: str, blob: Mapping[str, Any]) -> Citation:
    """Turn a ``[entries.<key>]`` blob into a :class:`Citation`.

    Resolves the ``{{VIBEQC_VERSION}}`` / ``{{VIBEQC_YEAR}}`` templates
    against the running vibe-qc version.
    """
    if not isinstance(blob, Mapping):
        raise DatabaseError(f"entry {key!r} must be a table")
    for fld in _REQUIRED_FIELDS:
        if fld not in blob:
            raise DatabaseError(f"entry {key!r} missing required field {fld!r}")

    authors = blob.get("authors") or ()
    if isinstance(authors, str):
        authors = (authors,)
    elif isinstance(authors, list):
        authors = tuple(str(a) for a in authors)
    else:
        raise DatabaseError(f"entry {key!r}: authors must be a list of strings")

    def _opt(name: str) -> Any:
        return blob.get(name)

    def _opt_template(field_name: str, template_name: str) -> Any:
        v = blob.get(template_name)
        if v is None:
            return blob.get(field_name)
        return _resolve_template(str(v))

    def _opt_bool(name: str, default: bool) -> bool:
        v = blob.get(name)
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        raise DatabaseError(f"'{name}' must be a boolean, got {type(v).__name__}")

    return Citation(
        key=key,
        kind=str(blob["kind"]),
        bibtex_key=str(blob["bibtex_key"]),
        authors=authors,
        title=str(blob["title"]),
        journal=_opt("journal"),
        volume=_opt("volume"),
        issue=_opt("issue"),
        pages=_opt("pages"),
        publisher=_opt("publisher"),
        year=_opt_template("year", "year_template"),
        doi=_opt("doi"),
        url=_opt("url"),
        version=_opt_template("version", "version_template"),
        license=_opt("license"),
        notes=_opt("notes"),
        role=_opt("role"),
        print=_opt_bool("print", True),
    )


def _resolve_template(value: str) -> str:
    """Substitute the small set of template tokens vibeqc supports
    inside database.toml string fields. Currently:

    * ``{{VIBEQC_VERSION}}`` -> current package version.
    * ``{{VIBEQC_YEAR}}`` -> year extracted from the package version's
      release metadata, or the current year as a fallback.
    """
    out = value
    out = out.replace("{{VIBEQC_VERSION}}", str(VIBEQC_VERSION))
    out = out.replace("{{VIBEQC_YEAR}}", _vibeqc_year())
    return out


def _vibeqc_year() -> str:
    """Best-effort current vibe-qc release year.

    Falls back to the system clock -- manifest's
    :func:`vibeqc.system_info.system_info` already records the
    timestamp, so this is the relevant year for the citation.
    """
    import datetime as _dt

    return str(_dt.date.today().year)
