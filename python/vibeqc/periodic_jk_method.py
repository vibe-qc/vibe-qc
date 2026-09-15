"""User-selectable J/K (Coulomb + exchange) method for periodic SCF,
plus an intelligent AUTO picker.

vibe-qc supports several alternatives to FFT-Poisson for the periodic
two-electron Hartree-Fock / KS build. Each has a different sweet spot
in (basis quality, lattice shape, system size, accuracy target). This
module provides:

* :class:`PeriodicJKMethod` -- enum naming each method.
* :func:`pick_jk_method` -- heuristic that turns ``AUTO`` into a concrete
  choice given the system + basis + cell shape.
* :func:`validate_jk_method` -- hard error / warning on illegal or
  risky combinations.
* :func:`describe_jk_method` -- short human-readable label for logs.

Status by method:

  Method                  Status
  ---------------------   ---------------------------------------------
  GDF                     Native Gamma and multi-k RHF/RKS/UHF/UKS
                            implementation. The earlier PySCF-backed
                            spike is retired:
                            PySCF and CRYSTAL are external reference
                            programs, not in-process vibe-qc backends.
  AICCM (four members)    Selected through the front door
                            run_periodic_job(method="aiccm", variant=...)
                            (handovers/HANDOVER_AICCM_STANDARD_METHOD.md);
                            every member stamps [run].method_status =
                            "experimental" and keeps its experimental warning.
    AICCM2026DEV_A          variant="four-center": union-and-weight/
                            Wigner--Seitz four-center construction (the 2014
                            lineage) through the adapter
                            periodic.ccm.four_center_runner. RHF/RKS/UHF/UKS;
                            HF on the scalable builder. A different
                            construction from the two neutral producers, not
                            a representation of theirs.
    AICCM2026DEV_A_REAL_GAMMA
                            variant="real-gamma": neutral Γ-CCM (ruling R1),
                            real-Γ supercell representation of the finite
                            BvK-torus Hamiltonian (periodic.ccm.direct via
                            periodic.ccm.real_gamma_runner). RHF/UHF/RKS/UKS.
    NEUTRAL_BLOCH           variant="neutral-bloch": the same neutral torus
                            Hamiltonian in its Bloch representation, produced
                            by periodic.ccm.ri (run_ccm_*_gdf) through the
                            adapter periodic.ccm.neutral_bloch_runner, NOT the
                            plain unit-cell GDF route. RHF/RKS/UHF/UKS;
                            Fourier-equivalent to real-gamma (Theorem 1).
    AICCM2026DEV_B          variant="chi": independent finite
                            translation-group-character (χ-CCM) torus
                            RHF/RKS/UHF/UKS (periodic.chi); Fourier-
                            equivalent to a real-Γ form only for the same
                            specified χ Hamiltonian.
                            The legacy jk_method spellings aiccm2026dev-a,
                            aiccm2026dev-b, real-gamma, real_gamma, chi and
                            chi-ccm still resolve, each with a
                            DeprecationWarning; gamma, gamma-ccm, gamma_ccm
                            and the former GDF aliases neutral-bloch,
                            bloch-control, gdf-control, aiccm-ri fail closed
                            (ruling R1).
  BIPOLE                  ✓ multi-k RHF/UHF/RKS/UKS via CRYSTAL-gauge
                            Ewald J-split: shared Ewald a, analytic
                            J^LR, direct J^SR + K (HF) or V_xc (DFT),
                            with the incomplete quartet multipole far field
                            unavailable and fail-closed. ROHF (Gamma + full
                            Monkhorst-Pack meshes) via the corrected-
                            Ewald-exchange EWALD_3D engine; maintained
                            preview, integer 2/1/0 occupations.
  DIRECT                  Limited scope. Plumbed against the existing
                            C++ ``build_fock_2e_real_space`` which does
                            the proper triple-cell-sum + Schwarz
                            screening on each pair displacement. BUT --
                            with omega=0 (full Coulomb), the truncated
                            real-space sum DIVERGES for tight ionic
                            crystals (verified on MgO sto-3g: max|J-J_py|
                            ≈ 363 Ha at cutoff 18 bohr). DIRECT is only
                            valid for vacuum-padded (molecular-limit)
                            cells where the cutoff actually clips the
                            density. AUTO will refuse to pick DIRECT for
                            non-vacuum-padded crystals. Native GDF/FFTDF
                            is the production target.
  FFT_POISSON             ✓ native EWALD_3D path. The public name is
                            legacy; the default Hartree J backend now
                            uses the analytical AO-pair FT with G=0
                            omitted. The historical FFT-Poisson grid
                            backend is env-gated for parity tests.
  GPW                     ✓ Γ-only RHF/ROHF/ROKS/RKS/UHF/UKS via FFT-Poisson
                            on a smooth real-space grid; multi-k pure-DFT
                            RKS/ROKS/UKS; forces, Hessian,
                            DFT+U, smearing, D3-BJ dispersion on the
                            non-restricted-open-shell routes. ROHF/ROKS are
                            maintained preview.
                            Experimental -- emits GAPWExperimentalWarning.
                            See docs/design_periodic_gapw.md.
  GAPW                    ✓ All-electron Gaussian-augmented plane-wave
                            (Lippert & Hutter 1999, doi:10.1007/978-3-
                            540-48272-4_3; Krack & Parrinello 2000,
                            doi:10.1039/B001167N). Adds per-atom
                            radial augmentation on top of GPW.
                            Experimental -- guarded by
                            GAPWExperimentalWarning.
  RSGDF                   not yet implemented (range-separated GDF;
                            Ye & Berkelbach, J. Chem. Phys. 154,
                            131104 (2021), DOI 10.1063/5.0046617).
  RIJCOSX                 Γ-only RHF on the dedicated periodic COSX
                            driver; true multi-k RHF/RKS/UHF/UKS via the
                            native GDF loop with k_exchange='cosx'.
  CFMM                    not yet implemented (continuous fast-multipole;
                            White et al. CPL 230, 8).

The AUTO picker chooses based on:

  - **Lattice shape**: all implemented methods accept general 3D
    lattice vectors at the API level; method quality is still basis-
    and system-dependent.
  - **Basis compactness**: a "compact" basis (sto-3g, minimal) lets
    DIRECT converge; diffuse bases (cc-pVTZ, def2-TZVP) need GDF or
    RSGDF.
  - **System size**: huge supercells favor CFMM (when implemented).

The default policy does not silently fall back to a method that is known
to be scientifically unsafe for tight ionic crystals. Native GDF is the
maintained AUTO route where its dimensional and method envelope applies;
unsupported combinations raise with an explicit explanation.
"""

from __future__ import annotations

import enum
import os
import warnings
from typing import Optional

import numpy as np

__all__ = [
    "PeriodicJKMethod",
    "pick_jk_method",
    "validate_jk_method",
    "describe_jk_method",
    "is_orthorhombic",
    "resolve_jk_method_string",
    # AICCM front door (handovers/HANDOVER_AICCM_STANDARD_METHOD.md, M1):
    # method="aiccm", variant=... is resolved here so the runner, the docs
    # and the T2 selector tests share one table without importing
    # vibeqc.periodic.ccm. Module-level names only; vibeqc.__all__ is
    # unchanged.
    "AICCM_VARIANTS",
    "AICCM_VARIANT_ROUTES",
    "AICCM_VARIANT_OF",
    "resolve_aiccm_variant",
    "jk_method_agrees_with_variant",
    # M4b (#778): the post-HF selector that rides beside ``variant``.
    "AICCM_CORRELATIONS",
    "resolve_aiccm_correlation",
]


class PeriodicJKMethod(enum.Enum):
    """User-selectable periodic J/K builder.

    Use ``AUTO`` to let vibe-qc choose; pass any other value to force
    a specific method. The resolved method is logged in the output
    file's banner so reproducibility doesn't depend on the AUTO
    heuristic version.
    """

    AUTO = "auto"
    GDF = "gdf"  # native Gamma and multi-k Gaussian density fitting
    # --- AICCM: four formulations behind one front door --------------------
    # run_periodic_job(method="aiccm", variant=...) maps a variant onto one
    # of these members (AICCM_VARIANT_ROUTES below); the member is what the
    # runner dispatches on and what the .system manifest records. Ruling R1
    # (2026-08-21, periodic/ccm/route.py): Paper-1 Γ-CCM is the NEUTRAL
    # finite-torus construction with two producers (real-gamma,
    # neutral-bloch); the union-and-weight four-centre line is the 2014
    # lineage; chi is a separate construction.
    AICCM2026DEV_A = "aiccm2026dev-a"  # variant="four-center" (2014 lineage)
    # Value is the front-door variant name (M0 of
    # handovers/HANDOVER_AICCM_STANDARD_METHOD.md): what the .system manifest
    # records. The A-prefixed spelling is a rejected input (D89/D90).
    AICCM2026DEV_A_REAL_GAMMA = "real-gamma"  # variant="real-gamma"
    # The neutral Γ-CCM Bloch producer (periodic.ccm.ri), wired into the
    # runner by M1b through periodic.ccm.neutral_bloch_runner. The value is
    # ALSO a retired jk_method string (_RETIRED_GDF_CONTROL_STRINGS): the
    # member is reachable only as variant="neutral-bloch", never as
    # jk_method="neutral-bloch", so the word cannot mean plain unit-cell GDF
    # anywhere (D-2b).
    NEUTRAL_BLOCH = "neutral-bloch"  # variant="neutral-bloch"
    AICCM2026DEV_B = "aiccm2026dev-b"  # variant="chi" (χ-CCM torus)
    BIPOLE = "bipole"  # CRYSTAL-gauge Ewald J-split RHF/ROHF/UHF/RKS/UKS
    # (multi-k); shared Ewald a; hybrids OK; ROHF = corrected-Ewald-exchange
    # EWALD_3D engine (maintained preview, integer occupations)
    DIRECT = "direct"  # partial -- see module docstring
    FFT_POISSON = "fft_poisson"  # implemented; native FFT metric
    RIJCOSX = "rijcosx"  # RI-J + periodic COSX-K; Gamma RHF + multi-k all spin cases
    # GPW / GAPW: v0.11.x periodic route (Lippert & Hutter 1997,
    # doi:10.1080/00268979709482119; Krack & Parrinello 2000,
    # doi:10.1039/B001167N). GPW = FFT-Poisson Hartree-J on a smooth
    # real-space grid; GAPW = all-electron via per-atom radial
    # augmentation on top of GPW. Both are experimental (guarded by
    # GAPWExperimentalWarning). See docs/design_periodic_gapw.md.
    GPW = "gpw"  # Gamma all six methods; multi-k pure-DFT RKS/ROKS/UKS
    GAPW = "gapw"  # all-electron via per-atom radial augmentation
    # Vacuum-free 2D (slab) Coulomb -- the rigorous Parry / de Leeuw-Perram
    # gauge routed via lattice_opts.coulomb_method=SLAB_EWALD_2D. AUTO for
    # dim=2 closed-/open-shell (RHF/RKS/UKS). Not a bulk (dim=3) method.
    SLAB_EWALD_2D = "slab_ewald_2d"
    RSGDF = "rsgdf"  # not yet implemented
    CFMM = "cfmm"  # not yet implemented


# Selectors that name a recognized AICCM formulation with no runner arm yet.
# EMPTY since M3a (2026-09-04): the four-centre construction was the last one,
# and it now dispatches through ``vibeqc.periodic.ccm.four_center_runner``.
# The set is kept rather than deleted because it is the mechanism a future
# formulation uses to fail closed with a library pointer instead of an
# AttributeError, and ``validate_jk_method`` still consults it.
_UNWIRED_CCM_ROUTES: frozenset[PeriodicJKMethod] = frozenset()

# Every selector that dispatches an AICCM formulation through
# ``run_periodic_job``. Membership drives the ``[run].method_status =
# "experimental"`` stamp the periodic runner writes into the ``.system``
# manifest at job start (handovers/HANDOVER_AICCM_STANDARD_METHOD.md, M0).
# This set equals set(AICCM_VARIANT_ROUTES.values()); pinned by
# tests/test_periodic_feature_input_guards.py.
_AICCM_JK_METHODS: frozenset[PeriodicJKMethod] = frozenset(
    {
        PeriodicJKMethod.AICCM2026DEV_A,
        PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA,
        PeriodicJKMethod.NEUTRAL_BLOCH,
        PeriodicJKMethod.AICCM2026DEV_B,
    }
)

# Members reachable only through method="aiccm", variant=...: they never had
# a jk_method spelling of their own, so the unknown-name error must not list
# their value (as a jk_method *string* it is refused, see
# _RETIRED_GDF_CONTROL_STRINGS) and a member passed directly as jk_method
# fails closed in the runner.
_FRONT_DOOR_ONLY_JK_METHODS: frozenset[PeriodicJKMethod] = frozenset(
    {PeriodicJKMethod.NEUTRAL_BLOCH}
)

# Short / spoken aliases accepted anywhere a jk_method *string* is resolved,
# normalised to the canonical enum value before lookup. Keys are lower-cased.
# Since M1 this table carries the slab spellings only; the AICCM spellings
# live in _LEGACY_AICCM_JK_SPELLINGS (resolve, DeprecationWarning) and in the
# two retired tables below (fail closed).
_JK_STRING_ALIASES: dict[str, str] = {
    "slab": PeriodicJKMethod.SLAB_EWALD_2D.value,
    "ewald_2d": PeriodicJKMethod.SLAB_EWALD_2D.value,
    "ewald-2d": PeriodicJKMethod.SLAB_EWALD_2D.value,
    "slab-ewald": PeriodicJKMethod.SLAB_EWALD_2D.value,
}
_REJECTED_A_PREFIXED_CONTROL_STRINGS = frozenset(
    {
        # Literal on purpose: since M0 no enum member carries this value.
        "aiccm2026dev-a-real-gamma",
        "aiccm2026dev-a-direct",
    }
)

# --- AICCM front door: variants -> members ---------------------------------
#: The four formulations selectable as ``run_periodic_job(method="aiccm",
#: variant=...)``, in the order the docs list them. Variants name
#: formulations, never lines: there is no ``variant="gamma"`` (ruling R1).
AICCM_VARIANTS: tuple[str, ...] = (
    "real-gamma",
    "neutral-bloch",
    "four-center",
    "chi",
)
#: variant -> the :class:`PeriodicJKMethod` member the runner dispatches on
#: (and records as ``jk_method_requested`` / ``_resolved`` / ``_executed``).
AICCM_VARIANT_ROUTES: dict[str, PeriodicJKMethod] = {
    "real-gamma": PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA,
    "neutral-bloch": PeriodicJKMethod.NEUTRAL_BLOCH,
    "four-center": PeriodicJKMethod.AICCM2026DEV_A,
    "chi": PeriodicJKMethod.AICCM2026DEV_B,
}
#: Inverse of :data:`AICCM_VARIANT_ROUTES`: what :func:`describe_jk_method`,
#: the ``.out`` "J/K method" line and the ``[run].aiccm_variant`` manifest
#: field print for each AICCM member.
AICCM_VARIANT_OF: dict[PeriodicJKMethod, str] = {
    member: variant for variant, member in AICCM_VARIANT_ROUTES.items()
}

#: The post-HF methods selectable as ``run_periodic_job(method="aiccm",
#: variant=..., correlation=...)`` (M4b, #778). ``correlation`` names the
#: correlation treatment only; the SCF reference underneath it still comes
#: from ``variant`` and the D-3 inference, so the two selectors are
#: orthogonal and neither implies the other.
AICCM_CORRELATIONS: tuple[str, ...] = (
    "mp2",
    "ccsd",
    "dlpno-mp2",
    "dlpno-ccsd",
)

# Legacy dev-era jk_method spellings (dash-normalised) -> variant. They keep
# resolving, each with a DeprecationWarning naming the method="aiccm",
# variant=... form (plan section 3.2). The warning is emitted by the string
# resolver, so run_periodic_job(jk_method=<spelling>) and
# resolve_jk_method_string(<spelling>) both warn; the front door passes the
# member and never enters the resolver, so it never warns.
_LEGACY_AICCM_JK_SPELLINGS: dict[str, str] = {
    "aiccm2026dev-a": "four-center",
    "aiccm2026dev-b": "chi",
    "chi": "chi",
    "chi-ccm": "chi",
    "real-gamma": "real-gamma",  # real_gamma arrives dash-normalised
}

# Ruling R1 (2026-08-21, periodic/ccm/route.py): Paper-1 Γ-CCM denotes the
# NEUTRAL finite-BvK-torus construction, which has two admissible producers,
# so a bare 'gamma' can neither pick a producer nor silently select the 2014
# union-and-weight four-centre lineage (D-2). Dash-normalised (gamma_ccm
# arrives as gamma-ccm); mirrors the library's _R1_PAPER_FACING_GAMMA_ALIASES
# so both surfaces refuse the same words.
_R1_PAPER_FACING_GAMMA_STRINGS: frozenset[str] = frozenset({"gamma", "gamma-ccm"})

# D-2b: in the CCM library these words select the neutral Γ-CCM Bloch
# producer (run_ccm_scf(ccm, route='neutral-bloch'), periodic/ccm/ri.py);
# here they selected PLAIN unit-cell GDF. One word must not mean two
# Hamiltonians, so they fail closed with a pointer. Checked BEFORE the enum
# lookup because "neutral-bloch" is also the value of
# PeriodicJKMethod.NEUTRAL_BLOCH. Dash-normalised.
_RETIRED_GDF_CONTROL_STRINGS: frozenset[str] = frozenset(
    {"neutral-bloch", "bloch-control", "gdf-control", "aiccm-ri"}
)


def _r1_bare_gamma_message(name: object, *, surface: str = "jk_method") -> str:
    return (
        f"{surface}={name!r} is retired (ruling R1, 2026-08-21): Paper-1 "
        "Γ-CCM denotes the NEUTRAL finite-BvK-torus construction, which has "
        "two admissible producers, so one word can neither pick a producer "
        "nor silently select the 2014 union-and-weight four-centre lineage. "
        "Select the formulation with method='aiccm' and variant='real-gamma' "
        "(real-Γ supercell representation of the neutral torus), "
        "variant='neutral-bloch' (Bloch representation of the same torus "
        "Hamiltonian), or variant='four-center' (the union-and-weight "
        "lineage)."
    )


def _retired_gdf_control_message(
    name: object, *, surface: str = "jk_method"
) -> str:
    return (
        f"{surface}={name!r} is retired (ruling R1, 2026-08-21; D-2b): in the "
        "CCM library this word names the neutral Γ-CCM Bloch producer, "
        "run_ccm_scf(ccm, route='neutral-bloch') in vibeqc.periodic.ccm, "
        "while here it selected plain unit-cell GDF, and one word must not "
        "mean two Hamiltonians. Use jk_method='gdf' for plain unit-cell GDF, "
        "or method='aiccm', variant='neutral-bloch' for the neutral producer "
        "(its runner arm is milestone M1b of "
        "handovers/HANDOVER_AICCM_STANDARD_METHOD.md; until then the library "
        "entry above runs it)."
    )


#: Directory of the installed ``vibeqc`` package. Warnings raised for a user
#: mistake are attributed to the first frame OUTSIDE it, so a deprecation
#: points at the line the user wrote rather than at a vibe-qc frame.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _warn_legacy_aiccm_spelling(
    spelling: object, member: PeriodicJKMethod, *, stacklevel: int = 3
) -> None:
    """Emit the DeprecationWarning for a legacy AICCM jk_method spelling.

    Attribution matters more than it looks: Python's default filters show a
    :class:`DeprecationWarning` only when it is attributed to ``__main__``
    (everything else is swallowed by the catch-all ``ignore`` filter), so a
    warning blamed on a vibe-qc frame is invisible to every user. The
    entry points differ in depth -- :func:`resolve_jk_method_string` is three
    frames away, ``run_periodic_job`` is five, and its output-writer
    decorator adds a sixth -- so no fixed ``stacklevel`` is right for all of
    them. ``skip_file_prefixes`` walks out of the package instead and lands
    on the caller's own line whatever the path; ``stacklevel`` is the
    fallback for a runtime without it (added in Python 3.12).

    Deliberately no once-per-process memory: the standard ``warnings``
    registry already deduplicates per call site under the default filters,
    and a module-level flag would make ``pytest.warns`` order-dependent.
    """
    variant = AICCM_VARIANT_OF[member]
    message = (
        f"jk_method={spelling!r} is a deprecated AICCM selector: select the "
        f"formulation with method='aiccm', variant={variant!r} "
        "(docs/user_guide/aiccm.md). The spelling keeps resolving for now."
    )
    try:
        warnings.warn(
            message,
            DeprecationWarning,
            skip_file_prefixes=(_PACKAGE_DIR,),
        )
    except TypeError:  # pragma: no cover - Python < 3.12
        warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def resolve_aiccm_variant(variant: object) -> str:
    """Normalise the ``variant`` of ``method="aiccm"``; fail closed otherwise.

    ``None`` and unknown values raise :class:`ValueError` listing the four
    variants; the bare gamma spellings and the retired GDF-control words
    raise with the ruling-R1 message on this surface too (D-2 / D-2b).
    Case-insensitive; ``_`` and ``-`` are equivalent.
    """
    listing = ", ".join(repr(v) for v in AICCM_VARIANTS)
    if variant is None:
        raise ValueError(
            "method='aiccm' requires variant=...: the AICCM method name does "
            f"not imply a formulation. Valid: {listing}."
        )
    key = str(variant).strip().lower().replace("_", "-")
    if key in AICCM_VARIANT_ROUTES:
        return key
    if key in _R1_PAPER_FACING_GAMMA_STRINGS:
        raise ValueError(_r1_bare_gamma_message(variant, surface="variant"))
    if key in _RETIRED_GDF_CONTROL_STRINGS:
        raise ValueError(_retired_gdf_control_message(variant, surface="variant"))
    raise ValueError(
        f"unknown AICCM variant {variant!r} for method='aiccm'. "
        f"Valid: {listing}."
    )


def resolve_aiccm_correlation(correlation: object) -> "str | None":
    """Normalise ``correlation`` of ``method="aiccm"``; fail closed otherwise.

    ``None`` passes through unchanged and means "SCF only" -- unlike
    ``variant``, this selector is optional, because an AICCM job without a
    correlation treatment is the ordinary and complete thing to ask for.
    Anything else must name a member of :data:`AICCM_CORRELATIONS`.
    Case-insensitive; ``_`` and ``-`` are equivalent, matching
    :func:`resolve_aiccm_variant`.
    """
    if correlation is None:
        return None
    listing = ", ".join(repr(c) for c in AICCM_CORRELATIONS)
    key = str(correlation).strip().lower().replace("_", "-")
    if key in AICCM_CORRELATIONS:
        return key
    raise ValueError(
        f"unknown AICCM correlation {correlation!r} for method='aiccm'. "
        f"Valid: {listing}, or None for an SCF-only run."
    )


def jk_method_agrees_with_variant(
    jk_method: "PeriodicJKMethod | str", variant: str
) -> bool:
    """True iff ``jk_method`` is AUTO or names the route of ``variant``.

    The front door's conflict rule: ``jk_method`` stays ``"auto"`` or equals
    the variant's own underlying route (its enum member, the member's value,
    or a legacy spelling of that same member). A retired spelling of another
    route never agrees. Compares strings without resolving, so it never
    warns.
    """
    member = AICCM_VARIANT_ROUTES[resolve_aiccm_variant(variant)]
    if isinstance(jk_method, PeriodicJKMethod):
        return jk_method in (PeriodicJKMethod.AUTO, member)
    key = str(jk_method).strip().lower()
    if key in ("auto", ""):
        return True
    key = key.replace("_", "-")
    if key == member.value:
        return True
    return _LEGACY_AICCM_JK_SPELLINGS.get(key) == AICCM_VARIANT_OF[member]


def resolve_jk_method_string(name: str) -> "PeriodicJKMethod":
    """Public alias of the jk_method string resolver (aliases applied).

    The legacy AICCM spellings ``"aiccm2026dev-a"``, ``"aiccm2026dev-b"``,
    ``"real-gamma"`` / ``"real_gamma"``, ``"chi"`` / ``"chi-ccm"`` resolve to
    their :class:`PeriodicJKMethod` member and warn with
    :class:`DeprecationWarning` naming the ``method="aiccm", variant=...``
    form. ``"gamma"`` / ``"gamma-ccm"`` / ``"gamma_ccm"`` and the former GDF
    aliases ``"neutral-bloch"`` / ``"bloch-control"`` / ``"gdf-control"`` /
    ``"aiccm-ri"`` fail closed with the ruling-R1 message; the A-prefixed
    control spellings stay rejected (D89/D90).
    """
    return _resolve_jk_string(name, stacklevel=4)


def _resolve_jk_string(
    name: str, *, stacklevel: int = 4
) -> "PeriodicJKMethod":
    """Resolve a jk_method *string* to a :class:`PeriodicJKMethod`.

    Order: the A-prefixed control spellings are rejected (D89/D90); the
    R1-retired spellings fail closed pointing at ``method="aiccm"``; the
    legacy AICCM spellings resolve with a :class:`DeprecationWarning`
    attributed to the first frame outside the package (``stacklevel`` is
    only the fallback for a runtime without ``skip_file_prefixes``, see
    :func:`_warn_legacy_aiccm_spelling`); the slab aliases are
    applied; then the enum lookup. Raises :class:`ValueError` on an unknown
    name listing the ``jk_method`` values still selected here (the AICCM
    formulations are selected with ``method="aiccm", variant=...``; the
    front-door-only members are not listed).
    """
    key = str(name).strip().lower()
    dashed = key.replace("_", "-")
    if key in _REJECTED_A_PREFIXED_CONTROL_STRINGS:
        raise ValueError(
            f"Periodic JK method {name!r} is rejected because the "
            "'aiccm2026dev-a' prefix denotes the union-and-weight four-centre "
            "lineage, while real-Gamma is a producer of the neutral Γ-CCM "
            "construction (ruling R1). Use method='aiccm', "
            "variant='four-center' for the lineage or method='aiccm', "
            "variant='real-gamma' for the neutral producer."
        )
    if dashed in _R1_PAPER_FACING_GAMMA_STRINGS:
        raise ValueError(_r1_bare_gamma_message(name))
    if dashed in _RETIRED_GDF_CONTROL_STRINGS:
        raise ValueError(_retired_gdf_control_message(name))
    if dashed in _LEGACY_AICCM_JK_SPELLINGS:
        member = AICCM_VARIANT_ROUTES[_LEGACY_AICCM_JK_SPELLINGS[dashed]]
        _warn_legacy_aiccm_spelling(name, member, stacklevel=stacklevel)
        return member
    key = _JK_STRING_ALIASES.get(key, key)
    try:
        member = PeriodicJKMethod(key)
    except ValueError:
        valid = ", ".join(
            m.value
            for m in PeriodicJKMethod
            if m not in _FRONT_DOOR_ONLY_JK_METHODS
        )
        aliases = ", ".join(sorted(_JK_STRING_ALIASES))
        raise ValueError(
            f"Unknown periodic JK method: {name!r}. Valid: {valid}. "
            f"Accepted aliases: {aliases}. The AICCM formulations are "
            "selected with method='aiccm', variant="
            + "|".join(AICCM_VARIANTS)
            + "."
        ) from None
    if member in _FRONT_DOOR_ONLY_JK_METHODS:
        # Every front-door-only value is also a retired string above, so this
        # is unreachable today; it keeps a future member fail-closed.
        raise ValueError(_retired_gdf_control_message(name))
    return member


# Methods that are actually wired up natively.
_IMPLEMENTED: frozenset[PeriodicJKMethod] = frozenset(
    {
        PeriodicJKMethod.GDF,  # Gamma and multi-k in periodic_runner
        PeriodicJKMethod.AICCM2026DEV_B,  # finite BvK torus RHF/RKS/UHF/UKS
        PeriodicJKMethod.BIPOLE,  # multi-k RHF/UHF/RKS/UKS via pbc_bipole
        PeriodicJKMethod.DIRECT,  # only valid for vacuum-padded cells
        # PeriodicJKMethod.FFT_POISSON -- RETIRED v0.13.0 as a user route
        # (validate_jk_method raises). Internal Γ-only EWALD_3D drivers
        # remain for dilute periodic + mechanics, fail-closed on dense cells.
        PeriodicJKMethod.GPW,  # M2-full / M3a / M3b -- Γ-only RHF
        PeriodicJKMethod.GAPW,  # M3 -- adds Gaussian augmentation
        PeriodicJKMethod.RIJCOSX,  # Gamma RHF + multi-k GDF/COSX
        PeriodicJKMethod.SLAB_EWALD_2D,  # dim=2 vacuum-free slab (Γ + multi-k)
        # EXPERIMENTAL (2026-07-26): neutral Γ-CCM, real-Γ supercell producer
        # via the per-unit-cell adapter (periodic.ccm.real_gamma_runner).
        # RHF/UHF/RKS/UKS incl. screened hybrids; dim=3 only (driver-enforced).
        PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA,
        # EXPERIMENTAL (M3a): the union-and-weight/Wigner--Seitz four-centre
        # Γ-CCM construction (the 2014 lineage) via the per-unit-cell adapter
        # periodic.ccm.four_center_runner. RHF/RKS/UHF/UKS; dim=3 only. HF
        # runs on the scalable C++ lattice-sum builder (D-6b).
        PeriodicJKMethod.AICCM2026DEV_A,
        # EXPERIMENTAL (M1b): the Bloch producer of the same neutral torus
        # (periodic.ccm.ri run_ccm_*_gdf) via the per-unit-cell adapter
        # periodic.ccm.neutral_bloch_runner. RHF/RKS/UHF/UKS; dim=3 only.
        PeriodicJKMethod.NEUTRAL_BLOCH,
        # PeriodicJKMethod.RSGDF,
        # PeriodicJKMethod.CFMM,
    }
)


# ============================================================
# Helpers
# ============================================================


def is_orthorhombic(lattice: np.ndarray, tol: float = 1e-10) -> bool:
    """True iff ``lattice`` is diagonal (axis-aligned orthorhombic).

    Compatibility helper for older routing code and docs. ``lattice``
    is the (3, 3) Cartesian matrix whose columns are a₁, a₂, a₃.
    """
    L = np.asarray(lattice, dtype=float)
    off_diag = L - np.diag(np.diag(L))
    diag_norm = max(np.linalg.norm(np.diag(L)), 1.0)
    return float(np.max(np.abs(off_diag))) < tol * diag_norm


def _basis_is_compact(basis_name: str) -> bool:
    """Is this a compact (minimal / sto-3g-like) basis where DIRECT
    can converge with manageable cell-cutoff?

    Heuristic: name starts with "sto" (any zeta count) or contains
    "minimal". Could be extended once DIRECT is back online.
    """
    n = (basis_name or "").lower()
    return n.startswith("sto") or "minimal" in n


# ============================================================
# AUTO picker
# ============================================================


def pick_jk_method(
    method: PeriodicJKMethod | str,
    *,
    lattice: np.ndarray,
    basis_name: str,
    n_atoms: int,
    scf_method: str = "RHF",
    dim: int = 3,
) -> PeriodicJKMethod:
    """Resolve an ``AUTO`` choice into a concrete :class:`PeriodicJKMethod`.

    Returns ``method`` unchanged if it is already concrete (and
    implemented). Raises if the resolved method is not implemented or
    is incompatible with the lattice / basis / dimensionality.

    ``dim`` is the system's periodic dimensionality (``system.dim``). For
    ``dim == 2`` (a slab), AUTO resolves to the vacuum-free
    :attr:`~PeriodicJKMethod.SLAB_EWALD_2D` gauge for RHF/RKS/UKS; every bulk
    unsupported bulk J/K builders fail closed on a slab because running them
    would treat the slab as a 3-D crystal of sheets ``a3`` apart, giving
    ``a3``-/k-mesh-dependent energies (CLAUDE.md Sec. 7). The bounded explicit
    ``jk_method='gdf'`` route is available for closed-shell RHF/RKS on a full
    Gamma-centered slab mesh. For ``dim == 1``, AUTO selects GDF for
    RHF/RKS/UHF/UKS and fails closed for ROHF. For ``dim == 3``, AUTO selects
    GDF for closed-shell RHF/RKS and BIPOLE for UHF/UKS/ROHF; explicit
    open-shell GDF remains available for UHF/UKS.
    """
    if isinstance(method, str):
        # The deprecation is attributed to the first frame outside the
        # package, so this depth is only the pre-3.12 fallback: warn() ->
        # _warn_legacy_aiccm_spelling -> _resolve_jk_string -> here ->
        # run_periodic_job -> the user's call.
        method = _resolve_jk_string(method, stacklevel=5)
    dim = int(dim)
    spin = scf_method.upper()

    # --- dim=2 (slab) routing: the vacuum-free SLAB_EWALD_2D gauge only ------
    if dim == 2:
        if method in (PeriodicJKMethod.AUTO, PeriodicJKMethod.SLAB_EWALD_2D):
            if spin in ("RHF", "RKS", "UKS"):
                return PeriodicJKMethod.SLAB_EWALD_2D
            raise NotImplementedError(
                "dim=2 (slab) SCF is available for RHF/RKS/UKS via the "
                "vacuum-free SLAB_EWALD_2D gauge; "
                f"scf_method={scf_method!r} is not yet supported for slabs "
                "(open-shell HF / UHF on slabs is a follow-on)."
            )
        if method == PeriodicJKMethod.GDF:
            if spin in ("RHF", "RKS"):
                return PeriodicJKMethod.GDF
            raise NotImplementedError(
                "dim=2 slab GDF is currently available for closed-shell "
                f"RHF/RKS only; got scf_method={scf_method!r}. Use "
                "jk_method='auto' for the direct UKS slab route."
            )
        if method == PeriodicJKMethod.AICCM2026DEV_B:
            raise NotImplementedError(
                "aiccm2026dev-b does not define a dim=2 neutral finite-torus "
                "Hamiltonian. Generic slab_ewald_2d or slab GDF routes are "
                "not substitutes for the χ-CCM construction; derive and "
                "validate the shared wire/slab Coulomb convention first."
            )
        # An explicit bulk (dim=3) builder on a slab would silently run the
        # slab as a 3-D crystal of sheets a3 apart -> a3-/k-mesh-dependent
        # garbage. Fail closed with a pointer (CLAUDE.md Sec. 7).
        raise NotImplementedError(
            f"jk_method={method.value!r} is a bulk (dim=3) Coulomb builder and "
            "cannot run a dim=2 slab correctly: it would treat the slab as a "
            "3-D crystal of sheets a3 apart, giving a3-/k-mesh-dependent "
            "energies. Use jk_method='auto' (or 'slab_ewald_2d') for the "
            "rigorous vacuum-free 2D Coulomb."
        )

    # SLAB_EWALD_2D is a dim=2-only gauge; reject it on bulk / polymer cells.
    if method == PeriodicJKMethod.SLAB_EWALD_2D and dim != 2:
        raise NotImplementedError(
            f"jk_method='slab_ewald_2d' requires a dim=2 slab; got dim={dim}."
        )

    if method != PeriodicJKMethod.AUTO:
        return method

    # The public BIPOLE route is 3D-only. Open-shell GDF is maintained for
    # dim=1, so AUTO must not select a backend that the runner immediately
    # rejects. There is no maintained dim=1 ROHF backend yet.
    if dim == 1:
        if spin == "ROHF":
            raise NotImplementedError(
                "AUTO has no maintained dim=1 ROHF route. Use UHF with "
                "jk_method='gdf', or provide a validated 3D periodic model."
            )
        return PeriodicJKMethod.GDF

    # In 3D, ROHF rides the open-shell arm: its public multi-k-capable
    # backend is the BIPOLE-route corrected-Ewald-exchange engine (ROKS stays
    # on the closed-shell arm and fails closed downstream unless GPW is
    # selected explicitly).
    if spin in ("UHF", "UKS", "ROHF"):
        return PeriodicJKMethod.BIPOLE
    return PeriodicJKMethod.GDF


# ============================================================
# Validation
# ============================================================


def validate_jk_method(
    method: PeriodicJKMethod,
    *,
    lattice: np.ndarray,
    basis_name: str,
) -> None:
    """Raise / warn on illegal or risky combinations."""
    if method == PeriodicJKMethod.FFT_POISSON:
        # Retired as a user-facing route (v0.13.0). The Γ-only EWALD_3D
        # path returns a ~2 Ha-wrong energy on dense ionic crystals (its
        # molecular-limit density convention D(g!=0)=0 is invalid once
        # periodic images overlap) and is superseded everywhere by GDF /
        # BIPOLE / GPW. The internal drivers (run_r{h,k}f_periodic_gamma_
        # ewald3d) remain for dilute periodic + mechanics tests, fail-closed
        # on dense cells (CLAUDE.md Sec.7).
        raise ValueError(
            "jk_method='fft_poisson' (Γ-only EWALD_3D) is retired as of "
            "v0.13.0: it returns a ~2 Ha-wrong energy on dense ionic "
            "crystals and is superseded everywhere by GDF / BIPOLE / GPW. "
            "Use jk_method='gdf' (default), 'bipole', or 'gpw'."
        )
    if method in _UNWIRED_CCM_ROUTES:
        # Fail closed with a library pointer rather than an AttributeError.
        # Empty since M3a; kept as the mechanism for the next formulation.
        raise NotImplementedError(
            f"jk_method={method.value!r} names a recognized AICCM formulation "
            "with no runner arm yet. Call its library entry in "
            "vibeqc.periodic.ccm (run_ccm_scf(ccm, route=...))."
        )
    # GPW / GAPW are wired through `run_periodic_job` as of M3b/M3c:
    # the GAPW dispatch uses the per-atom augmentation correction
    # for all-electron accuracy. Both are Γ-only RHF/UHF/RKS/UKS.
    if method == PeriodicJKMethod.GAPW:
        # GAPW is implemented -- see periodic_gapw_augment.py
        pass
    if method not in _IMPLEMENTED:
        raise NotImplementedError(
            f"Periodic JK method {method.value!r} is not yet wired up "
            f"natively. Currently available: "
            f"{ {m.value for m in _IMPLEMENTED} }"
        )

    if method == PeriodicJKMethod.GDF:
        return

    if method == PeriodicJKMethod.AICCM2026DEV_B:
        return

    if method == PeriodicJKMethod.AICCM2026DEV_A:
        # The union-and-weight four-centre construction: the runner arm
        # enforces its own envelope (RHF/RKS/UHF/UKS, dim=3, no DFT+U, no
        # gradients), and the front door always executes the symmetric
        # BvK-torus weighting rather than the library's union12 default.
        return

    if method == PeriodicJKMethod.NEUTRAL_BLOCH:
        # The neutral Γ-CCM Bloch producer: the runner arm enforces its own
        # envelope (RHF/RKS/UHF/UKS, dim=3, no DFT+U, no gradients), and the
        # producer refuses a vacuum-padded cluster and a positive converged
        # neutral energy (IID 291).
        return

    if method == PeriodicJKMethod.BIPOLE:
        # BIPOLE requires 3D (the Ewald gauge is 3D-only).
        # The public runner rejects lower-dimensional BIPOLE requests.
        L = np.asarray(lattice, dtype=float)
        if np.linalg.matrix_rank(L) < 3:
            raise ValueError(
                "PeriodicJKMethod.BIPOLE requires a 3D lattice. "
                "For 1D/2D systems use GDF or DIRECT."
            )
        return

    if method == PeriodicJKMethod.DIRECT:
        # build_fock_2e_real_space at omega=0 is real-space-truncated,
        # NOT Ewald-split. The truncated 1/r lattice sum diverges for
        # tight ionic crystals. Verified on MgO sto-3g cutoff=18:
        # max|J-J_pyscf| ≈ 363 Ha. DIRECT is only valid for cells
        # padded enough that the Coulomb tail outside cutoff is small.
        warnings.warn(
            "PeriodicJKMethod.DIRECT (build_fock_2e_real_space, omega=0) "
            "is only valid for VACUUM-PADDED (molecular-limit) cells. "
            "On tight ionic crystals the truncated real-space Coulomb "
            "lattice sum diverges (e.g., MgO sto-3g shows a historical "
            "hundreds-of-Hartree J error at cutoff 18 bohr). Native "
            "GDF/FFTDF is the target for tight crystals; DIRECT is "
            "appropriate for vacuum-padded molecules in PBC "
            "boxes (Makov-Payne regime). Future Method 2 augmentation "
            "would add a reciprocal-space LR contribution to make it "
            "valid for tight crystals (= Ewald composition).",
            stacklevel=2,
        )


# ============================================================
# Description
# ============================================================

_DESCRIPTIONS = {
    PeriodicJKMethod.AUTO: "AUTO -- pick at runtime",
    PeriodicJKMethod.GDF: "GDF -- Gaussian density fitting (native Gamma and "
    "multi-k RHF/RKS/UHF/UKS on the unit cell; PySCF/CRYSTAL are external "
    "references only). Not an AICCM route: the neutral fitted-torus Bloch "
    "producer is method='aiccm', variant='neutral-bloch'",
    # AICCM members: "aiccm (<variant>) -- ...; experimental". The prefix
    # follows AICCM_VARIANT_OF and the trailing token is the M0 stamp
    # contract (tests/test_periodic_feature_input_guards.py).
    PeriodicJKMethod.AICCM2026DEV_A: "aiccm (four-center) -- "
    "union-and-weight/Wigner--Seitz four-center construction (the 2014 "
    "lineage, not the neutral Γ-CCM torus): bare 1/r minimum image with the "
    "WSSC four-centre weights, RHF/RKS/UHF/UKS, HF on the scalable "
    "lattice-sum builder. Library entry: run_ccm_rhf(method='aiccm2026dev-a') "
    "/ run_ccm_scf(route='four-center'); experimental",
    PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA: "aiccm (real-gamma) -- "
    "neutral Γ-CCM (ruling R1), real-Γ supercell representation of the "
    "neutral fitted-torus Hamiltonian: one real generalized eigenproblem "
    "with the derived exchange-q0 seam; Fourier-equivalent to variant "
    "'neutral-bloch', not the four-center construction. Library entry: "
    "run_ccm_rhf_direct / run_ccm_scf(route='real-gamma'); experimental",
    PeriodicJKMethod.NEUTRAL_BLOCH: "aiccm (neutral-bloch) -- "
    "neutral Γ-CCM (ruling R1), Bloch representation of the neutral "
    "fitted-torus Hamiltonian on the torus's own Gamma-centred mesh through "
    "the fitted producer (periodic.ccm.ri); RHF/RKS/UHF/UKS, "
    "Fourier-equivalent to variant 'real-gamma', not the plain unit-cell GDF "
    "route. Library entry: run_ccm_rhf_gdf / "
    "run_ccm_scf(route='neutral-bloch'); experimental",
    PeriodicJKMethod.AICCM2026DEV_B: "aiccm (chi) -- χ-CCM, independent "
    "finite translation-group-character BvK-torus RHF/RKS/UHF/UKS "
    "(4-center, RI, or RIJCOSX; exact Gamma-supercell / Gamma-centred "
    "character-mesh equivalence only for the same specified χ Hamiltonian); "
    "experimental",
    PeriodicJKMethod.BIPOLE: "BIPOLE -- CRYSTAL-gauge Ewald J-split (multi-k; "
    "shared Ewald a across V_ne/E_nn/J^LR; "
    "RHF/UHF/RKS/UKS, hybrids OK; ROHF via the corrected-Ewald-exchange "
    "EWALD_3D engine, maintained preview)",
    PeriodicJKMethod.DIRECT: "DIRECT -- 4-center periodic lattice sum + Schwarz screening "
    "(partial: missing-s_q bug)",
    PeriodicJKMethod.FFT_POISSON: "FFT_POISSON -- RETIRED v0.13.0 (Γ-only EWALD_3D; "
    "wrong on dense ionic crystals). Use GDF (default) / BIPOLE / GPW",
    PeriodicJKMethod.RIJCOSX: "RIJCOSX -- RI-J (density fitting) + periodic COSX-K "
    "(Gamma RHF plus true multi-k RHF/RKS/UHF/UKS; experimental -- periodic "
    "COSX is novel)",
    PeriodicJKMethod.GPW: "GPW -- Gaussian + plane-wave J via FFT-Poisson on a smooth "
    "real-space grid (Γ-only RHF/ROHF/RKS/UHF/UKS; multi-k RKS; "
    "ROHF maintained preview with integer occupations; other routes add "
    "forces, Hessian, DFT+U, smearing, D3-BJ; experimental)",
    PeriodicJKMethod.GAPW: "GAPW -- Gaussian-augmented plane-wave J (Lippert-Hutter 1997 + "
    "Krack-Parrinello 2000); smooth plane-wave grid + per-atom "
    "radial augmentation for all-electron accuracy (experimental)",
    PeriodicJKMethod.RSGDF: "RSGDF -- range-separated GDF (not yet implemented)",
    PeriodicJKMethod.CFMM: "CFMM -- continuous fast multipole (not yet implemented)",
}


def describe_jk_method(method: PeriodicJKMethod) -> str:
    """Short human-readable description for output logs."""
    return _DESCRIPTIONS.get(method, str(method))
