"""Semiempirical route implementation status.

This module is a compact, queryable companion to the user-guide status table
and the semiempirical handover.  It deliberately labels Python-heavy
performance routes as reference-only until they have native kernels or a
maintainer decision that Python orchestration is the intended final shape.

``backend`` names **where a route's kernel lives**; it is not a maturity
claim, and it is not a second maturity vocabulary.  Route maturity has exactly
one source, :data:`vibeqc.semiempirical.routes.SEMIEMPIRICAL_MATURITIES`,
reached through a plan's ``maturity`` (#150).  The one place the two tables
overlap is :attr:`SemiempiricalRouteStatus.production`, which must equal
``plan.maturity == MATURITY_PRODUCTION`` for every reachable plan;
``tests/test_semiempirical_route_plan.py`` pins that invariant so the tables
cannot drift apart again.
"""

from __future__ import annotations

from dataclasses import dataclass

from .routes import SEMIEMPIRICAL_METHOD_ALIASES, normalise_semiempirical_method


BACKEND_NATIVE = "native"
BACKEND_NATIVE_FD = "native-fd"
BACKEND_MIXED_NATIVE = "mixed-native"
BACKEND_PYTHON_REFERENCE = "python-reference"
BACKEND_GATED_EXPERIMENTAL = "gated-experimental"


@dataclass(frozen=True)
class SemiempiricalRouteStatus:
    """Implementation status for one public semiempirical route.

    ``backend`` is the implementation axis (where the kernel lives), not a
    maturity label.  ``production`` is the single field shared with the
    canonical maturity vocabulary and tracks
    ``plan.maturity == MATURITY_PRODUCTION``; see the module docstring.
    """

    route: str
    backend: str
    production: bool
    performance_critical: bool
    summary: str


_ROUTES = (
    SemiempiricalRouteStatus(
        "dftb",
        BACKEND_NATIVE,
        True,
        True,
        "DFTB0/SCC-DFTB production calls use native kernels; parameters remain "
        "in-house screening/preoptimization quality.",
    ),
    SemiempiricalRouteStatus(
        "periodic-dftb0-gradient-analytic",
        BACKEND_NATIVE,
        True,
        True,
        "Periodic Gamma DFTB0 gradients use the native analytic overlap and "
        "repulsive derivative at the validated 15-bohr image domain.",
    ),
    SemiempiricalRouteStatus(
        "periodic-dftb0-kpoint",
        BACKEND_GATED_EXPERIMENTAL,
        False,
        True,
        "Full k-point periodic DFTB0 single-point energy/band routes use the "
        "native complex Bloch prototype with weighted occupations. Public "
        "analytic derivatives and external DFTB+ parity remain gated.",
    ),
    SemiempiricalRouteStatus(
        "periodic-dftb0-kpoint-gradient-fd",
        BACKEND_NATIVE_FD,
        False,
        True,
        "Full k-point periodic DFTB0 gradients and stress use one native "
        "batched finite-difference pass over the complex Bloch prototype. The "
        "route remains experimental pending analytic derivatives and external "
        "DFTB+ parity.",
    ),
    SemiempiricalRouteStatus(
        "dftb0-seccm-energy",
        BACKEND_GATED_EXPERIMENTAL,
        False,
        True,
        "DFTB0-SECCM uses the native frozen-record finite-cluster energy "
        "adapter for neutral closed-shell insulating cyclic clusters in "
        "one, two, or three dimensions, scoped to the explicit "
        "repulsive-pair table (H, C, N, O, F, P, S, Cl); stress, SCC, "
        "and broader parameter scopes remain gated.",
    ),
    SemiempiricalRouteStatus(
        "dftb0-seccm-gradient-analytic",
        BACKEND_GATED_EXPERIMENTAL,
        False,
        True,
        "DFTB0-SECCM analytic gradients differentiate the complete frozen "
        "record set with exact fractional ownership for the gated neutral "
        "closed-shell insulating cyclic route in one, two, or three "
        "dimensions.",
    ),
    SemiempiricalRouteStatus(
        "scc-dftb-seccm-energy",
        BACKEND_GATED_EXPERIMENTAL,
        False,
        True,
        "SCC-DFTB-SECCM uses the native frozen-record finite-cluster "
        "adapter with atomic charge self-consistency and an opt-in "
        "Madelung/Ewald embedding for charged cells; closed-shell "
        "insulating clusters only, per-cell energy closure.",
    ),
    SemiempiricalRouteStatus(
        "scc-dftb-seccm-gradient-analytic",
        BACKEND_GATED_EXPERIMENTAL,
        False,
        True,
        "SCC-DFTB-SECCM analytic gradients differentiate the converged "
        "fixed-charge energy on the frozen record set (M = -W + "
        "D . 1/2 (kappa hbar - V_A - V_B) overlap contraction, WS-weighted "
        "repulsive and gamma derivatives). Embedded cells add the analytic "
        "fixed-charge Madelung derivative; the charge interaction is "
        "variational, so no coupled-perturbed SCC charge term remains.",
    ),
    SemiempiricalRouteStatus(
        "pm6-seccm-energy",
        BACKEND_GATED_EXPERIMENTAL,
        False,
        True,
        "PM6-SECCM uses the native frozen-record supercell NDDO Fock with "
        "WS-weighted two-center terms (Bredow-Geudtner-Jug); neutral "
        "closed-shell insulating clusters, energy only.",
    ),
    SemiempiricalRouteStatus(
        "pm6-seccm-gradient-fd",
        BACKEND_PYTHON_REFERENCE,
        False,
        True,
        "PM6-SECCM gradients use central differences of the converged "
        "per-cell energy over the frozen topology displacements. No analytic "
        "derivative of the current PM6 Hamiltonian is exposed.",
    ),
    SemiempiricalRouteStatus(
        "omx-seccm-energy",
        BACKEND_GATED_EXPERIMENTAL,
        False,
        True,
        "OM2/OM3-SECCM evaluates neutral closed-shell insulating finite groups "
        "over frozen Wigner-Seitz topologies. Any cyclic-image execution "
        "requires an explicit truncated-electrostatics acknowledgement and "
        "is an algorithm-mechanics probe, not a quantitative solid-state "
        "result. "
        "The published eq-13 three-center image weighting is the default; "
        "eq-10 remains available for explicit comparison. OM1-SECCM fails "
        "closed until its defining analytic core-valence ECP is implemented.",
    ),
    SemiempiricalRouteStatus(
        "gfn2-seccm-energy",
        BACKEND_GATED_EXPERIMENTAL,
        False,
        True,
        "GFN2-SECCM evaluates molecular-limit and multi-replica frozen "
        "Wigner-Seitz topologies with the native shell-resolved SCC engine; "
        "optional Madelung and Ewald-gamma embeddings are recorded by the "
        "runtime route plan.",
    ),
    SemiempiricalRouteStatus(
        "periodic-scc-dftb-gradient-fd",
        BACKEND_PYTHON_REFERENCE,
        False,
        True,
        "Periodic Gamma SCC-DFTB gradients use central differences of the "
        "converged total energy; the fixed-charge analytic prototype is not "
        "promoted for reaction-path forces.",
    ),
    SemiempiricalRouteStatus(
        "gfn2-xtb",
        BACKEND_GATED_EXPERIMENTAL,
        False,
        True,
        "GFN2-xTB is gated experimental until external xtb parity is closed; "
        "the periodic Gamma driver carries the lattice-summed Ewald-split "
        "shell gamma and the Bannwarth 2019 AES with image-resolved moments.",
    ),
    SemiempiricalRouteStatus(
        "periodic-gfn2-gradient",
        BACKEND_NATIVE,
        False,
        True,
        "Periodic Gamma GFN2-xTB gradients and stress are the native analytic "
        "derivatives of the lattice-summed energy: the reduced SCC state "
        "(shell charges and CAMM moments) is variational, and both match "
        "central finite differences on gapped 3-D and 2-D fixtures. They "
        "remain experimental pending external parity.",
    ),
    SemiempiricalRouteStatus(
        "periodic-scc-dftb-kpoint",
        BACKEND_GATED_EXPERIMENTAL,
        False,
        True,
        "Full k-point periodic SCC-DFTB single-point energy/band routes use "
        "the native complex Bloch SCC prototype with weighted Mulliken "
        "populations and optional Fermi occupations. Public analytic "
        "derivatives and external DFTB+ parity remain gated.",
    ),
    SemiempiricalRouteStatus(
        "periodic-scc-dftb-kpoint-gradient-fd",
        BACKEND_NATIVE_FD,
        False,
        True,
        "Full k-point periodic SCC-DFTB gradients and stress use one native "
        "batched finite-difference pass over converged complex Bloch SCC "
        "energies. The route remains experimental pending analytic "
        "derivatives and external DFTB+ parity.",
    ),
    SemiempiricalRouteStatus(
        "pm6",
        BACKEND_NATIVE,
        False,
        True,
        "Molecular PM6/UPM6 calls route through native NDDO kernels. H-only "
        "and H-heavy s/p interactions are source-correct, but full heavy-heavy "
        "two-center tensor parity remains open; periodic PM6 is experimental.",
    ),
    SemiempiricalRouteStatus(
        "pm6-gradient-fd",
        BACKEND_NATIVE_FD,
        False,
        True,
        "Molecular PM6/UPM6 gradients use native finite-difference wrappers "
        "over native NDDO energy calls. This is C++-backed but remains a "
        "finite-difference stopgap until analytic gradients land.",
    ),
    SemiempiricalRouteStatus(
        "periodic-pm6",
        BACKEND_MIXED_NATIVE,
        False,
        True,
        "Periodic PM6 Gamma energy uses native NDDO kernels; gradient and "
        "stress use one native finite-difference batch. The route remains "
        "experimental pending broader periodic validation.",
    ),
    SemiempiricalRouteStatus(
        "pm7",
        BACKEND_GATED_EXPERIMENTAL,
        False,
        False,
        "Molecular PM7/UPM7 is gated because the shared prototype omits "
        "Stewart's feathered electron-electron, electron-core, and core-core "
        "electrostatics and therefore is not the published PM7 Hamiltonian.",
    ),
    SemiempiricalRouteStatus(
        "pm7-gradient-fd",
        BACKEND_GATED_EXPERIMENTAL,
        False,
        False,
        "Molecular PM7/UPM7 derivatives are gated with the incomplete energy "
        "Hamiltonian.",
    ),
    SemiempiricalRouteStatus(
        "periodic-pm7",
        BACKEND_GATED_EXPERIMENTAL,
        False,
        False,
        "Bloch-periodic PM7 is gated because no public dispatcher implements "
        "the PM7 core-core Hamiltonian. PM6 and topology-bound SECCM routes "
        "remain separate implementations.",
    ),
    SemiempiricalRouteStatus(
        "om1",
        BACKEND_NATIVE,
        False,
        True,
        "Molecular OM1 routes through the native NDDO kernel but remains "
        "experimental and warns because its analytic core-valence ECP is "
        "not implemented.",
    ),
    SemiempiricalRouteStatus(
        "om1-gradient-fd",
        BACKEND_NATIVE_FD,
        False,
        True,
        "Molecular OM1 gradients use native finite differences over the "
        "experimental incomplete-ECP energy route.",
    ),
    SemiempiricalRouteStatus(
        "omx",
        BACKEND_NATIVE,
        False,
        True,
        "Molecular OM2/OM3 calls route through native NDDO kernels with "
        "published relative-energy fixtures. External implementation parity "
        "and geometry accuracy remain open, so this is a validated "
        "development/prescreening surface rather than a production claim.",
    ),
    SemiempiricalRouteStatus(
        "omx-gradient-fd",
        BACKEND_NATIVE_FD,
        False,
        True,
        "Molecular OM2/OM3 gradients use native finite-difference wrappers "
        "over the validated development/prescreening energy calls.",
    ),
    SemiempiricalRouteStatus(
        "periodic-omx",
        BACKEND_GATED_EXPERIMENTAL,
        False,
        False,
        "Bloch-periodic OMx is gated because the retired prototype did not "
        "implement the published ORT, ECP, and penetration Hamiltonian. "
        "Topology-bound OM2/OM3-SECCM remains a separate experimental route; "
        "OM1-SECCM is gated with the missing analytic ECP.",
    ),
    SemiempiricalRouteStatus(
        "msindo-energy",
        BACKEND_NATIVE,
        True,
        True,
        "MSINDO INDO RHF/UHF and validated closed-shell NDDO production energy "
        "calls route through native C++ with Python parity tests retained. The "
        "NDDO scope covers H, Li-F, and Na-Cl, including source-parity SPDD "
        "terms for Al through Cl.",
    ),
    SemiempiricalRouteStatus(
        "msindo-gradient",
        BACKEND_NATIVE,
        True,
        True,
        "Closed-shell MSINDO analytic gradients route through native C++ within "
        "the validated scope.",
    ),
    SemiempiricalRouteStatus(
        "msindo-nddo-gradient-analytic",
        BACKEND_MIXED_NATIVE,
        True,
        True,
        "Closed-shell MSINDO-NDDO results for H, Li-F, and Na-Cl use the native "
        "energy kernel and a Python analytic multipole-derivative contraction, "
        "including SPDD terms for Al through Cl. Open-shell NDDO gradients fail "
        "closed.",
    ),
    SemiempiricalRouteStatus(
        "msindo-ccm-energy",
        BACKEND_MIXED_NATIVE,
        True,
        True,
        "MSINDO CCM closed-shell energy calls use native C++ for H-Kr and the "
        "deterministic Python reference for Rb-Xe or when the binding is absent. "
        "A selected native calculation never falls back after failure.",
    ),
    SemiempiricalRouteStatus(
        "msindo-ccm-gradient-fd",
        BACKEND_MIXED_NATIVE,
        True,
        True,
        "MSINDO CCM finite-difference gradients use native C++ for valid H-Kr "
        "closed-shell cells and the Python reference for Rb-Xe or a missing "
        "binding. A selected native calculation never falls back after failure.",
    ),
    SemiempiricalRouteStatus(
        "msindo-ccm-gradient-analytic",
        BACKEND_MIXED_NATIVE,
        True,
        True,
        "MSINDO CCM analytic gradients use native C++ for valid H-Kr "
        "closed-shell cells and the Python reference for Rb-Xe or a missing "
        "binding. A selected native calculation never falls back after failure.",
    ),
    SemiempiricalRouteStatus(
        "msindo-cosmo",
        BACKEND_MIXED_NATIVE,
        True,
        True,
        "COSMO/GEPOL matrix construction, ASC solves, and reaction-field steps "
        "use native helpers; the outer COSMO SCF loop remains Python "
        "orchestration.",
    ),
    SemiempiricalRouteStatus(
        "msindo-cis",
        BACKEND_PYTHON_REFERENCE,
        False,
        True,
        "CIS/TDA matrix construction and diagonalization are validated Python "
        "reference code until a native excited-state kernel lands.",
    ),
    SemiempiricalRouteStatus(
        "msindo-cis-gradient",
        BACKEND_PYTHON_REFERENCE,
        False,
        True,
        "Analytic CIS gradient contractions are validated Python reference code "
        "until the CPHF/Z-vector and pair contractions move native.",
    ),
    SemiempiricalRouteStatus(
        "msindo-cis-gradient-fd",
        BACKEND_PYTHON_REFERENCE,
        False,
        True,
        "Finite-difference excited-state gradients and root tracking are robust "
        "Python reference code; each step intentionally runs repeated SCFs.",
    ),
    SemiempiricalRouteStatus(
        "msindo-cisd",
        BACKEND_PYTHON_REFERENCE,
        False,
        True,
        "The semiempirical CISD route uses the generic Python CI solver stack as "
        "a correctness reference.",
    ),
    SemiempiricalRouteStatus(
        "msindo-ovgf",
        BACKEND_PYTHON_REFERENCE,
        False,
        True,
        "OVGF/GF2 MO transforms and diagonal self-energy loops remain Python "
        "reference code with memory preflight coverage.",
    ),
    SemiempiricalRouteStatus(
        "msindo-md",
        BACKEND_PYTHON_REFERENCE,
        False,
        True,
        "MSINDO MD is Python orchestration over repeated energy/gradient calls "
        "and is scoped to small reference trajectories.",
    ),
    SemiempiricalRouteStatus(
        "msindo-metadynamics",
        BACKEND_PYTHON_REFERENCE,
        False,
        True,
        "MSINDO metadynamics is Python orchestration over FD-gradient MD and is "
        "scoped to small reference demonstrations.",
    ),
)


SEMIEMPIRICAL_ROUTE_STATUS = {entry.route: entry for entry in _ROUTES}
_METHOD_STATUS_ROUTES = {
    "ccm": "msindo-ccm-energy",
    "dftb0": "dftb",
    "scc_dftb": "dftb",
    "gfn2_xtb": "gfn2-xtb",
    "pm6": "pm6",
    "pm7": "pm7",
    "om1": "om1",
    "om2": "omx",
    "om3": "omx",
    "msindo": "msindo-energy",
}
_ROUTE_ONLY_ALIASES = {
    "ccm": "msindo-ccm-energy",
    "msindo-ccm": "msindo-ccm-energy",
    "nddo": "msindo-energy",
    "om1-gradient": "om1-gradient-fd",
    "om1-gradient-fd": "om1-gradient-fd",
    "om2-gradient": "omx-gradient-fd",
    "om2-gradient-fd": "omx-gradient-fd",
    "om3-gradient": "omx-gradient-fd",
    "om3-gradient-fd": "omx-gradient-fd",
    "omx-gradient": "omx-gradient-fd",
    "pm6-gradient": "pm6-gradient-fd",
    "upm6-gradient": "pm6-gradient-fd",
    "upm6-gradient-fd": "pm6-gradient-fd",
    "upm6": "pm6",
    "pm7-gradient": "pm7-gradient-fd",
    "upm7-gradient": "pm7-gradient-fd",
    "upm7-gradient-fd": "pm7-gradient-fd",
    "upm7": "pm7",
}


def _route_alias_key(route: str) -> str:
    return str(route).strip().lower().replace("_", "-")


def _method_status_aliases() -> dict[str, str]:
    aliases = {
        _route_alias_key(method): status_route
        for method, status_route in _METHOD_STATUS_ROUTES.items()
    }
    for alias, method in SEMIEMPIRICAL_METHOD_ALIASES.items():
        status_route = _METHOD_STATUS_ROUTES.get(method)
        if status_route is not None:
            aliases[_route_alias_key(alias)] = status_route
    return aliases


SEMIEMPIRICAL_ROUTE_ALIASES = {
    **_method_status_aliases(),
    **_ROUTE_ONLY_ALIASES,
}
REFERENCE_ONLY_SEMIEMPIRICAL_ROUTES = frozenset(
    entry.route
    for entry in _ROUTES
    if entry.backend == BACKEND_PYTHON_REFERENCE
)


def semiempirical_route_status(route: str) -> SemiempiricalRouteStatus:
    """Return the implementation status for ``route``.

    Route names are normalized by lower-casing and replacing underscores with
    hyphens, so ``"msindo_cis"`` and ``"msindo-cis"`` resolve identically.
    Public method aliases such as ``"dftb0"``, ``"scc-dftb"``,
    ``"gfn2xtb"``, ``"om2"``, ``"msindo"``, and ``"seccm"`` resolve to
    their canonical route labels.  A ``SemiempiricalRoutePlan``-like object is
    also accepted if it exposes a ``status_route`` attribute.
    """

    if hasattr(route, "status_route"):
        key = getattr(route, "status_route")
    else:
        method_key = normalise_semiempirical_method(route)
        key = _METHOD_STATUS_ROUTES.get(method_key)
        if key is None:
            key = _route_alias_key(route)
            key = SEMIEMPIRICAL_ROUTE_ALIASES.get(key, key)
    try:
        return SEMIEMPIRICAL_ROUTE_STATUS[key]
    except KeyError as exc:
        known = ", ".join(sorted(
            set(SEMIEMPIRICAL_ROUTE_STATUS) | set(SEMIEMPIRICAL_ROUTE_ALIASES)
        ))
        raise KeyError(
            f"unknown semiempirical route {route!r}; known: {known}"
        ) from exc


def semiempirical_route_runtime_available(route: str) -> bool:
    """Return whether route-specific runtime assets are locally available.

    This is separate from implementation maturity: experimental routes can be
    runtime-available, and production routes can still fail later for chemistry
    or convergence reasons. The probe is intentionally cache/local only.
    """
    status = semiempirical_route_status(route)
    if status.route in {
        "pm7",
        "pm7-gradient-fd",
        "periodic-pm7",
        "periodic-omx",
    }:
        return False
    if status.route in {"gfn2-xtb", "periodic-gfn2-gradient"}:
        from .methods.gfn2_params import gfn2_parameter_cache_available

        return gfn2_parameter_cache_available()
    return True


__all__ = [
    "BACKEND_GATED_EXPERIMENTAL",
    "BACKEND_MIXED_NATIVE",
    "BACKEND_NATIVE",
    "BACKEND_NATIVE_FD",
    "BACKEND_PYTHON_REFERENCE",
    "REFERENCE_ONLY_SEMIEMPIRICAL_ROUTES",
    "SEMIEMPIRICAL_ROUTE_ALIASES",
    "SEMIEMPIRICAL_ROUTE_STATUS",
    "SemiempiricalRouteStatus",
    "semiempirical_route_runtime_available",
    "semiempirical_route_status",
]
