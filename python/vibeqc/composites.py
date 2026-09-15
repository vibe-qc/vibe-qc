"""Composite 3c methods -- keyword-shortcut registry.

The "3c" methods of the Grimme school bundle a tuned small basis, a
modern dispersion correction (D3-BJ or D4), a geometric counterpoise
correction (:mod:`vibeqc.gcp`), and (sometimes) a modified short-range
correction (3c-SRB) into a single user-facing keyword. The result is a
single recipe that, on the GMTKN55 / S22 / X23 benchmarks, lands close
to a much-more-expensive parent functional + triple-zeta calculation at
small-basis cost.

This module is the **registry**: a single static dict mapping each
composite name (``"hf-3c"``, ``"pbeh-3c"``, ``"b97-3c"``,
``"b3lyp-3c"``, ``"r2scan-3c"``, ``"wb97x-3c"``, ``"hse-3c"``) to a
:class:`CompositeRecipe` describing the four ingredients plus an
availability marker. The keyword-shortcut dispatcher lives in
:mod:`vibeqc.runner` (the ``method=`` argument of :func:`run_job`
accepts composite names that resolve here).

Per CLAUDE.md Sec. 9 (no surprise dependencies), every composite carries
an :class:`Availability` flag the dispatcher checks before running:

* ``RUNNABLE`` -- every ingredient (basis + functional + dispersion +
  gCP + sr_mod) is fully wired today; the keyword shortcut works
  end-to-end. The user can ``run_job(method="hf-3c", ...)`` and get a
  total energy.
* ``NEEDS_BASIS`` -- the parent functional + dispersion + gCP framework
  are wired, but the target basis is not yet bundled (or is gated
  behind the basissetdev BSE-fetcher per CLAUDE.md Sec. 4). The user can
  still run the composite by passing ``basis_path=`` to point at a
  local file with the published basis-set definition.
* ``PENDING_F1`` -- the recipe depends on range-separated-hybrid (RSH)
  Coulomb machinery (F1 in the v0.8.0 functional sweep) that has not
  yet landed in vibe-qc's K-build. The dispatcher raises
  :class:`CompositeUnavailable` with the pointer.
* ``PENDING_F3`` -- same shape for meta-GGA t-density support (F3 in
  the v0.8.0 functional sweep). Required for r^2SCAN-3c.
* ``PENDING_ECP`` -- historical guard for composite bases carrying
  custom effective-core potentials. Main now wires those sidecars through
  libecpint's inline-primitive feed; this flag remains for future recipes
  whose ECP data are present but not yet consumable.
* ``PENDING_GCP_DATA`` -- the basis-fit (s, a, b) constants are
  registered in :mod:`vibeqc.gcp`, but the per-element e_mis / n_virt
  / zeta tables are not yet bundled. Calling the composite raises
  :class:`vibeqc.gcp.GCPDataMissing` with the contribute-via-PR
  pointer.

The availability matrix is **transparent**: ``vibeqc.list_composites()``
returns the catalogue with status flags so a user can see at a glance
which recipes work today and which are gated.

The recipe specifications are pinned to the originating publications:

* **HF-3c** -- Sure & Grimme, *J. Comput. Chem.* **34**, 1672 (2013).
* **PBEh-3c** -- Grimme, Brandenburg, Bannwarth, Hansen, *J. Chem.
  Phys.* **143**, 054107 (2015).
* **B97-3c** -- Brandenburg, Bannwarth, Hansen, Grimme, *J. Chem. Phys.*
  **148**, 064104 (2018).
* **r^2SCAN-3c** -- Grimme, Hansen, Ehlert, Mewes, *J. Chem. Phys.*
  **154**, 064103 (2021).
* **wB97X-3c** -- Müller, Hansen, Grimme, *J. Chem. Phys.* **158**,
  014103 (2023).
* **B3LYP-3c** -- community-extension recipe; B3LYP-VWN5 + D3(BJ) +
  gCP on def2-mSVP. Not a published "official" 3c composite but
  widely used in spectroscopy workflows for IR / vibrational
  frequencies (B3LYP's strength).
* **HSE-3c** -- Brandenburg, Caldeweyher, Grimme, *Phys. Chem. Chem.
  Phys.* **18**, 15519 (2016). HSE06 + D3(BJ,ATM) + damped gCP on
  def2-mSVP, the screened-exchange analogue of PBEh-3c.

Adding a new composite is a single :class:`CompositeRecipe` entry plus
the matching :class:`vibeqc.gcp.GCPParams` registration if a new basis
is involved -- the dispatcher needs no changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


__all__ = [
    "Availability",
    "CompositeRecipe",
    "CompositeUnavailable",
    "GCPDamping",
    "ShortRangeCorrection",
    "list_composites",
    "resolve_composite",
]


class CompositeUnavailable(RuntimeError):
    """Raised by :func:`vibeqc.run_job` when the requested composite
    keyword names a recipe gated on infrastructure (RSH, mGGA t, an
    unbundled basis) that hasn't landed yet. The message points at the
    blocking roadmap item.
    """


class Availability(str, Enum):
    """Per-recipe availability flag.

    See module docstring for the full list of statuses and their
    meaning. Values are strings (not bare enum constants) so they
    survive JSON serialisation cleanly when the recipe table is
    dumped into a structured log.
    """

    RUNNABLE = "runnable"
    NEEDS_BASIS = "needs_basis"
    PENDING_F1 = "pending_f1_rsh"
    PENDING_F3 = "pending_f3_mgga"
    PENDING_ECP = "pending_ecp_inline"
    PENDING_GCP_DATA = "pending_gcp_data"


@dataclass(frozen=True)
class ShortRangeCorrection:
    """Canonical short-range bond (SRB) correction of the 3c stack.

    HF-3c and B97-3c use genuinely DIFFERENT SRB functional forms, both
    ported from the authoritative Grimme-lab source (``grimme-lab/gcp``
    ``src/gcp.f90``, LGPL-3.0). They are discriminated by :attr:`kind`:

    * ``kind="hf3c_base"`` -- HF-3c (``gcp.f90::basegrad``), applied for
      element pairs with Z = 1..18 only::

          E_SRB = -qscal . S_{A<B} (Z_A Z_B)^{3/2}
                          . exp(-rscal . r0ab_AB^{0.75} . R_AB)

      with (qscal, rscal) = (0.03, 0.7).

    * ``kind="b973c_mod"`` -- B97-3c (``gcp.f90::srb_egrad2``, "modified
      form derived from the HF-3c SRB potential", SG Nov-2016), applied
      for ALL elements::

          E_SRB = -qscal . S_{A<B} (Z_A Z_B)^{1/2}
                          . exp(-(rscal / r0ab_AB) . R_AB)

      with (qscal, rscal) = (0.08, 10.0).

    ``r0ab_AB`` is the DFT-D3 ab-initio cut-off radius (bohr) from
    :mod:`vibeqc._d3_r0ab` (Grimme, Antony, Ehrlich, Krieg, *J. Chem.
    Phys.* 132, 154104 (2010)).

    The pre-v0.10.4 single form ``-s.S(Z_aZ_b)^0.5.exp(-g.R^{1/3})``
    matched NEITHER recipe (audit F1.4): it over-bound HF-3c/H2O SRB
    roughly 2x (gave ~-52 vs the canonical -22.25 kcal/mol).

    Attributes
    ----------
    name : str
        Human-readable label for the SCF log.
    kind : str
        ``"hf3c_base"`` or ``"b973c_mod"`` -- selects the functional form.
    qscal, rscal : float
        Per-recipe prefactor and radii-scaling parameter.
    citation : str
        Originating publication.
    """

    name: str
    kind: str
    qscal: float
    rscal: float
    citation: str

    def evaluate(self, atomic_numbers, positions) -> float:
        """Evaluate E_SRB (Hartree) for (atomic_numbers, positions in
        bohr). Standalone -- no SCF result needed. SRB is *binding*, so
        the value is negative for a bound geometry.

        Validated against grimme-lab/gcp on the H2O test geometry:
        HF-3c SRB = -0.035456180 Ha (-22.249 kcal/mol); B97-3c SRB =
        -0.005713243 Ha (-3.585 kcal/mol).
        """
        import numpy as np

        from ._d3_r0ab import r0ab

        positions = np.asarray(positions, dtype=float)
        n = len(atomic_numbers)
        energy = 0.0
        for a in range(n):
            za = int(atomic_numbers[a])
            for b in range(a + 1, n):
                zb = int(atomic_numbers[b])
                R = float(np.linalg.norm(positions[a] - positions[b]))
                if R < 1e-10:
                    continue
                if self.kind == "hf3c_base":
                    # gcp.f90::basegrad -- HF-3c, Z=1..18 only.
                    if not (1 <= za <= 18 and 1 <= zb <= 18):
                        continue
                    r0 = self.rscal * r0ab(za, zb) ** 0.75
                    energy -= (za * zb) ** 1.5 * np.exp(-r0 * R)
                elif self.kind == "b973c_mod":
                    # gcp.f90::srb_egrad2 -- B97-3c, all elements.
                    r0 = self.rscal / r0ab(za, zb)
                    energy -= (za * zb) ** 0.5 * np.exp(-r0 * R)
                else:
                    raise ValueError(
                        f"ShortRangeCorrection: unknown kind {self.kind!r} "
                        f"(expected 'hf3c_base' or 'b973c_mod')."
                    )
        return self.qscal * energy


@dataclass(frozen=True)
class CompositeDampingD3BJ:
    """D3-BJ damping parameters specific to a composite 3c recipe.

    The 3c composites use *re-fit* D3-BJ parameters that differ from
    the published damping for their parent functional. For example:

    * PBEh-3c uses (s6, s8, a1, a2) = (1.0, 0.0, 0.4860, 4.5000),
      whereas the standard PBE0-D3BJ damping is
      (1.0, 1.2177, 0.4145, 4.8593).
    * B97-3c uses (1.0, 0.4416, 0.3719, 3.9748).
    * HF-3c uses (1.0, 0.8777, 0.4171, 2.9149).
    * HSE-3c uses (1.0, 0.0, 0.4411, 4.5182) -- its own fit, *not*
      PBEh-3c's, even though the two share the def2-mSVP basis and the
      short-range potential.

    Two composites sharing a basis, a parent functional family, or a
    short-range potential is not evidence that they share a damping fit.
    Copy constants only from the recipe's own publication.

    Carrying the damping in the recipe (instead of letting the
    dispersion module's per-functional lookup decide) makes the
    composite recipe self-contained and avoids the wrong-default
    failure mode where a name lookup returns the published damping
    for a *different* method.

    ``s9`` scales the Axilrod-Teller-Muto three-body term. It is ``0.0``
    for a plain two-body D3-BJ recipe and ``1.0`` for the composites
    whose published definition includes the three-body C9 contribution.
    PBEh-3c's paper is explicit: "the three-body dispersion is always
    included in the PBEh-3c method" (Grimme, Brandenburg, Bannwarth &
    Hansen, J. Chem. Phys. 143, 054107 (2015), Sec. IV C; the scale
    factor is called a3 in their Eq. (3) and "is unity by default").
    HSE-3c inherits that convention -- see its recipe below. HF-3c's
    paper (Sure & Grimme 2013) never mentions ATM, so it stays 0.0.
    """
    s6: float
    s8: float
    a1: float
    a2: float
    citation: str
    s9: float = 0.0


@dataclass(frozen=True)
class GCPDamping:
    """Short-range damping parameters for the *damped* gCP form.

    The PBEh-3c / HSE-3c / r^2SCAN-3c composites use the damped gCP
    (``gcp.f90`` ``damp=.true.`` path): each pairwise gCP contribution
    is scaled by ``1 - 1/(1 + dmp_scal.(R/r0ab)^dmp_exp)``, suppressing
    the bonded-pair term so only genuine long-range (intermolecular)
    BSSE survives. Carrying this on the recipe (not the basis TOML) is
    deliberate -- the same def2-mSVP gCP parameters are used *damped* by
    PBEh-3c / HSE-3c but *undamped* by B3LYP-3c.

    Grimme's group default is ``(dmp_scal, dmp_exp) = (4.0, 6.0)``;
    passed to :func:`vibeqc.gcp.compute_gcp` as ``damping=(...)``.
    Without it an isolated molecule's gCP is grossly over-counted
    (audit F1.5: r^2SCAN-3c/H2O +28.8 undamped vs +1.12 damped).
    """
    dmp_scal: float = 4.0
    dmp_exp: float = 6.0
    citation: str = ""


@dataclass(frozen=True)
class CompositeRecipe:
    """Specification of a single composite 3c method.

    Attributes
    ----------
    name : str
        Lowercase canonical keyword (``"hf-3c"``, ``"pbeh-3c"``, ...).
    functional : str | None
        Parent XC functional name passed to :class:`Functional`
        (libxc alias). ``None`` for pure HF recipes (HF-3c).
    basis : str
        Target basis-set name. Passed to :class:`BasisSet` lookup.
    dispersion : str
        Dispersion correction: ``"d3bj"``, ``"d4"``, or ``"none"``.
    d3bj_damping : CompositeDampingD3BJ | None
        Composite-specific D3-BJ damping parameters. Set when
        ``dispersion == "d3bj"`` so the recipe is self-contained;
        the dispatcher uses these directly rather than asking
        :func:`vibeqc.d3bj_params_for` to look the functional up.
        ``None`` for ``dispersion == "d4"`` (where dftd4 has its own
        per-composite damping table keyed on the composite name).
    gcp_basis : str | None
        Basis-set name keyed into the gCP parameter registry
        (:mod:`vibeqc.gcp`). Usually equals ``basis``. ``None`` when the
        recipe applies NO gCP at all -- e.g. B97-3c, whose 3c stack uses
        the SRB term instead of gCP (mctc-gcp computes SRB-only for
        b973c; audit F1.5).
    gcp_variant : str | None
        Name of the per-method ``[variants.*]`` block to select from the
        ``gcp_basis`` parameter file. gCP's four fit constants are per
        ``(method, basis)`` -- ``gcp.f90`` keys them on the method -- and
        pbeh-3c / b3lyp-3c / hse-3c all sit on def2-mSVP, so the basis
        alone does not identify them. ``None`` uses the file's basis-level
        default. Only HSE-3c sets this today.
    sr_mod : ShortRangeCorrection | None
        Optional short-range bond (SRB) correction. Only HF-3c and
        B97-3c carry one; None for the others.
    gcp_damping : GCPDamping | None
        When set, the gCP for this recipe uses the damped form (PBEh-3c,
        HSE-3c, r^2SCAN-3c). ``None`` = plain (undamped) Kruse-Grimme gCP.
    availability : Availability
        Status flag the dispatcher checks before running.
    citation : str
        Originating publication.
    notes : str
        Free-text per-recipe annotation surfaced in
        ``vibeqc.list_composites()`` and in the SCF log when the
        composite is invoked.
    """

    name: str
    functional: Optional[str]
    basis: str
    dispersion: str
    gcp_basis: Optional[str]
    sr_mod: Optional[ShortRangeCorrection]
    availability: Availability
    citation: str
    notes: str = ""
    d3bj_damping: Optional[CompositeDampingD3BJ] = None
    gcp_damping: Optional[GCPDamping] = None
    gcp_variant: Optional[str] = None


# ---------------------------------------------------------------------------
# The seven composite recipes catalogued for the 3c surface. Availability
# reflects the actual state of ``main``; the dispatcher's behaviour is
# governed by the flag, not by code-path branches scattered across the runner.
# ---------------------------------------------------------------------------

_RECIPES: dict[str, CompositeRecipe] = {}


def _register(r: CompositeRecipe) -> None:
    _RECIPES[r.name.lower()] = r


_SRB_HF3C = ShortRangeCorrection(
    name="HF-3c short-range bond correction (SRB)",
    kind="hf3c_base",
    qscal=0.03,
    rscal=0.7,
    citation="Sure & Grimme, J. Comput. Chem. 34, 1672 (2013); "
             "form: grimme-lab/gcp gcp.f90::basegrad",
)
_SRB_B973C = ShortRangeCorrection(
    name="B97-3c short-range bond correction (SRB)",
    kind="b973c_mod",
    qscal=0.08,
    rscal=10.0,
    citation="Brandenburg, Bannwarth, Hansen, Grimme, J. Chem. Phys. 148, "
             "064104 (2018); form: grimme-lab/gcp gcp.f90::srb_egrad2",
)


_register(CompositeRecipe(
    name="hf-3c",
    functional=None,                    # pure HF
    basis="minix",
    dispersion="d3bj",
    gcp_basis="minix",
    sr_mod=_SRB_HF3C,
    availability=Availability.RUNNABLE,
    citation="Sure & Grimme, J. Comput. Chem. 34, 1672 (2013)",
    notes=(
        "Mean-field qualitative method for very large systems; "
        "MINIX basis = MINIS + d-functions on Na-Ar + d/p polarisation. "
        "Basis + functional alias landed in v0.9.0; gCP per-element "
        "tables for MINIX are the remaining bundling gate "
        "(PENDING_GCP_DATA)."
    ),
    d3bj_damping=CompositeDampingD3BJ(
        s6=1.0, s8=0.8777, a1=0.4171, a2=2.9149,
        citation="Sure & Grimme, J. Comput. Chem. 34, 1672 (2013)",
    ),
))

_register(CompositeRecipe(
    name="pbeh-3c",
    functional="pbeh-3c",               # modified-PBE0 alias landed in v0.9.0 (xc.cpp)
    basis="def2-msvp",
    dispersion="d3bj",
    gcp_basis="def2-msvp",
    sr_mod=None,
    availability=Availability.RUNNABLE,
    citation="Grimme, Brandenburg, Bannwarth, Hansen, J. Chem. Phys. 143, 054107 (2015)",
    notes=(
        "Modified PBE0 with re-tuned HF exchange (42% vs PBE0's 25%) "
        "on def2-mSVP. Strong noncovalent geometries; widely used as "
        "the cheap geometry-prep step before a higher-level single "
        "point. Full turnkey: SCF + D3-BJ(ATM) + damped gCP on def2-mSVP. "
        "gCP uses the damped form (dmp_scal=4, dmp_exp=6). The D3 energy "
        "includes the Axilrod-Teller-Muto three-body term (s9=1), which "
        "the defining paper makes part of the method definition."
    ),
    d3bj_damping=CompositeDampingD3BJ(
        # s9=1.0: "the three-body dispersion is always included in the
        # PBEh-3c method" (Grimme 2015, Sec. IV C). Their Eq. (3) is
        # E_disp = E_disp^(2) + a3 E_disp^(3) with a3 = 1 by default.
        s6=1.0, s8=0.0, a1=0.4860, a2=4.5000, s9=1.0,
        citation="Grimme, Brandenburg, Bannwarth, Hansen, J. Chem. Phys. 143, 054107 (2015)",
    ),
    gcp_damping=GCPDamping(
        dmp_scal=4.0, dmp_exp=6.0,
        citation="Grimme, Brandenburg, Bannwarth, Hansen 2015; gcp.f90 "
                 "damp=.true. (pbeh3c)",
    ),
))

_register(CompositeRecipe(
    name="b97-3c",
    functional="b97-3c",                # libxc XC_GGA_XC_B97_3C (id 327, pure GGA)
    basis="def2-mtzvp",
    dispersion="d3bj",
    gcp_basis=None,
    sr_mod=_SRB_B973C,
    availability=Availability.RUNNABLE,
    citation="Brandenburg, Bannwarth, Hansen, Grimme, J. Chem. Phys. 148, 064104 (2018)",
    notes=(
        "Workhorse GGA composite; especially good for transition metals. "
        "Becke 1997 GGA reparameterised by Brandenburg + the standard "
        "3c stack. B97-3c uses NO gCP: the def2-mTZVP basis is large "
        "enough that the residual short-range basis error is handled by "
        "the SRB term instead (mctc-gcp computes SRB-only for b973c -- "
        "audit F1.5). Total = SCF + D3-BJ + SRB."
    ),
    d3bj_damping=CompositeDampingD3BJ(
        s6=1.0, s8=0.4416, a1=0.3719, a2=3.9748,
        citation="Brandenburg, Bannwarth, Hansen, Grimme, J. Chem. Phys. 148, 064104 (2018)",
    ),
))

_register(CompositeRecipe(
    name="b3lyp-3c",
    functional="b3lyp",                 # vibe-qc's b3lyp is VWN5 (ORCA
                                        # definition) -- exactly the flavor the
                                        # TURBOMOLE-culture 3c lineage means
    basis="def2-msvp",
    dispersion="d3bj",
    gcp_basis="def2-msvp",
    sr_mod=None,
    availability=Availability.RUNNABLE,
    citation="Community extension (Grimme group lineage); see Najibi-Goerigk 2018 for benchmarks",
    notes=(
        "B3LYP-VWN5 + D3(BJ) + gCP on def2-mSVP. Particularly accurate "
        "for IR / vibrational frequencies (B3LYP's traditional "
        "strength). Not in the original Grimme 3c paper series; "
        "registered here for the spectroscopy workflows that ask for "
        "B3LYP -- there is NO canonical mctc-gcp reference for it. It "
        "reuses the def2-mSVP gCP parameters (sigma=1.0), which are "
        "tuned for the DAMPED form, so b3lyp-3c uses the same damped "
        "gCP as pbeh-3c/hse-3c (undamped those params give a "
        "nonphysical ~+30 kcal/mol on H2O; damped ~+0.8). D3-BJ damping "
        "is the standard B3LYP-D3BJ set (Grimme-Ehrlich-Goerigk 2011)."
    ),
    d3bj_damping=CompositeDampingD3BJ(
        s6=1.0, s8=1.9889, a1=0.3981, a2=4.4211,
        citation="Grimme, Ehrlich, Goerigk, J. Comput. Chem. 32, 1456 (2011) -- standard B3LYP-D3BJ damping",
    ),
    gcp_damping=GCPDamping(
        dmp_scal=4.0, dmp_exp=6.0,
        citation="Community recipe: reuses the damped def2-mSVP gCP "
                 "convention of PBEh-3c (no canonical b3lyp-3c source).",
    ),
))

_register(CompositeRecipe(
    name="r2scan-3c",
    functional="r2scan",                # mGGA -- F3 landed in v0.9.0
    basis="def2-mtzvpp",
    dispersion="d4",
    gcp_basis="def2-mtzvpp",
    sr_mod=None,
    availability=Availability.RUNNABLE,
    citation="Grimme, Hansen, Ehlert, Mewes, J. Chem. Phys. 154, 064103 (2021)",
    notes=(
        "'Swiss army knife' of the modern 3c stack -- best general-"
        "purpose composite per the published GMTKN55 / TM-track "
        "benchmarks. F3 (meta-GGA t-density support) + def2-mTZVPP "
        "basis landed in v0.9.0. Full turnkey: SCF + D4 + damped gCP. "
        "gCP uses the r2scan3c-specific virtual counts and the damped "
        "form (dmp_scal=4, dmp_exp=6) -- NOT zero; the earlier "
        "'gCP=0 by design' note was wrong (audit F1.5). gCP(H2O) = "
        "+1.12 kcal/mol, matching mctc-gcp."
    ),
    gcp_damping=GCPDamping(
        dmp_scal=4.0, dmp_exp=6.0,
        citation="Grimme, Hansen, Ehlert, Mewes 2021; gcp.f90 damp=.true. (r2scan3c)",
    ),
))

_register(CompositeRecipe(
    name="wb97x-3c",
    functional="wb97x-v",               # range-separated hybrid (F1 wired v0.9.0)
    basis="vdzp",
    dispersion="d4",
    gcp_basis="vdzp",
    sr_mod=None,
    availability=Availability.RUNNABLE,
    citation="Müller, Hansen, Grimme, J. Chem. Phys. 158, 014103 (2023)",
    notes=(
        "Range-separated hybrid 3c composite -- vDZP is a valence-only "
        "basis that REQUIRES its Stuttgart small-core ECPs to be "
        "physically meaningful. The F1 K-build (erf-Coulomb K_LR) and "
        "D4 dispersion are wired, but vDZP's per-element ECPs use "
        "non-standard ncores (e.g. O: ncore=2, lmax=3) that match no "
        "bundled libecpint XML library, so the molecular SCF wrappers "
        "auto-attach the parsed sidecar through libecpint's inline-"
        "primitive feed instead of running all-electron on the valence "
        "contraction."
    ),
))

_register(CompositeRecipe(
    name="hse-3c",
    functional="hse06",                 # screened RSH (F1 wired v0.9.0)
    basis="def2-msvp",
    dispersion="d3bj",
    gcp_basis="def2-msvp",
    gcp_variant="hse-3c",               # NOT PBEh-3c's gCP constants; see below
    sr_mod=None,
    availability=Availability.RUNNABLE,
    citation="Brandenburg, Caldeweyher, Grimme, Phys. Chem. Chem. Phys. 18, "
             "15519 (2016)",
    notes=(
        "Solid-state range-separated variant -- HSE06 (screened RSH "
        "with a=0, b=0.25, w=0.11) + def2-mSVP + D3(BJ,ATM) + gCP. "
        "A published composite (Brandenburg 2016), not a community "
        "extension: both its D3(BJ) damping and its gCP fit constants "
        "are HSE-3c's own (ESI Table S1), re-fit on S66x8 rather than "
        "inherited from PBEh-3c. "
        "F1 K-build landed in v0.9.0; both molecular and (when the "
        "periodic RSH path lands) cluster-embedded variants run via "
        "the same recipe. Provides the molecular-cluster analogue of "
        "HSE06's solid-state hybrid behaviour. Like PBEh-3c, the D3 "
        "energy includes the three-body C9 term (s9=1)."
    ),
    # Table S1 ("Empirical parameters of the HSE-3c method"), E_disp row,
    # of the ESI to Brandenburg, Caldeweyher & Grimme, Phys. Chem. Chem.
    # Phys. 18, 15519 (2016), doi:10.1039/c6cp01697a:
    #     s6 = 1.00000 (constrained)    s8 = 0.00000 (constrained)
    #     a1 = 0.44110                  a2 = 4.51820
    # a1 / a2 are HSE-3c's own, re-fit on S66x8 (paper, p. 15520), and are
    # NOT PBEh-3c's (0.4860 / 4.5000): screening the long-range Fock
    # exchange changes the medium-range correlation the damping absorbs.
    # vibe-qc shipped the inherited PBEh-3c pair here through v0.15.x.
    d3bj_damping=CompositeDampingD3BJ(
        # a1/a2 are HSE-3c's own (see the Table S1 comment above; the
        # inherited PBEh-3c 0.4860/4.5000 was corrected in v0.15.34).
        # s9=1.0 adds the three-body ATM term: the ESI's Sec. A states the
        # D3 energy includes "the three-body C9 (dipole-dipole-dipole)
        # contributions", and the CRYSTAL23 manual (Sec. 5.3.3) has the ATM
        # term on by default for HSE-3c. simple-dftd3 reproduces the ESI
        # Table S2 gas-phase benzene D3 energy (-0.0068628129 a.u.) to
        # 1e-10 Ha with these (s6, s8, a1, a2, s9).
        s6=1.0, s8=0.0, a1=0.4411, a2=4.5182, s9=1.0,
        citation="Brandenburg, Caldeweyher, Grimme, Phys. Chem. Chem. "
                 "Phys. 18, 15519 (2016), ESI Sec. A + Table S1/S2.",
    ),
    # The gCP fit constants (sigma, eta, alpha, beta) = (1.00000, 1.40858,
    # 0.29083, 1.95260) live in data_library/gcp/def2-msvp.toml under
    # [variants.hse-3c], selected by gcp_variant above. They are Table S1's
    # E_gCP row, and are NOT PBEh-3c's (eta = 1.32492). Grimme's own gcp
    # program disagrees with the paper here (gcp.f90 case 'hse3c' has
    # eta = 1.32378); the paper wins because only Table S1's constants
    # reproduce the ESI's own published benzene value, E_gCP = 0.0148400663
    # a.u. (ESI section B, Table S2). See the TOML's [variants.hse-3c].source
    # for the arbitration and grimme-lab/gcp issue #17 for the upstream bug.
    # vibe-qc shipped the inherited PBEh-3c constants here through v0.15.x.
    gcp_damping=GCPDamping(
        dmp_scal=4.0, dmp_exp=6.0,
        citation="Shares PBEh-3c's damped-gCP convention; gcp.f90 "
                 "damp=.true. (hse3c). The fit constants themselves are "
                 "HSE-3c's own -- see [variants.hse-3c] in def2-msvp.toml.",
    ),
))


def list_composites() -> list[CompositeRecipe]:
    """Return all registered composite recipes, sorted by keyword name.

    Useful for surface introspection -- e.g. for documentation
    generation, for CLI help, and for the v0.9.0 release-prep
    availability matrix in :doc:`docs/user_guide/composites.md`.
    """
    return [_RECIPES[k] for k in sorted(_RECIPES.keys())]


def resolve_composite(name: str) -> Optional[CompositeRecipe]:
    """Look up a composite recipe by keyword name. Case-insensitive,
    accepts both ``"wb97x-3c"`` and ``"wb97x-3c"`` (the unicode-omega
    form gets ASCII-normalised). Returns ``None`` when the name is
    not a registered composite -- caller decides whether that's an
    error.
    """
    key = name.lower().strip()
    # Accept unicode spellings while keeping this source file ASCII (chr() avoids
    # literal Greek/superscript glyphs). ".lower()" above folds the capital
    # forms, so normalise lower-case omega U+03C9 (also the .lower() target of
    # U+03A9 and U+2126) in the wb97 family and superscript-two U+00B2 in r2scan.
    key = key.replace(chr(0x3C9), "w").replace(chr(0x2126), "w")
    key = key.replace(chr(0xB2), "2")
    return _RECIPES.get(key)
