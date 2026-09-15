"""Route matrix for the χ-CCM / aiccm2026dev-b benchmark stream."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, prod

import probe_host


TWO_ELECTRON_SUPPORT_SCHEMA = (
    "vibeqc.aiccm2026dev-b.two-electron-support/v1"
)
TWO_ELECTRON_SUPPORT_KEYS = {
    "schema",
    "qualification",
    "reason",
    "backend",
    "runtime_backend",
    "direct",
    "fitted",
}
TWO_ELECTRON_DIRECT_KEYS = {
    "domain_policy",
    "sr_image_precision",
    "electronic_cutoff_bohr",
    "nuclear_cutoff_bohr",
    "sr_image_extent_bohr",
}
TWO_ELECTRON_FITTED_KEYS = {
    "gdf_method",
    "rsgdf_base_ke_cutoff_ha",
    "rsgdf_tail_ke_cutoff_ha",
    "mdf_ke_cutoff_ha",
    "cosx_exchange_support",
    "correlation_support",
}
DIRECT_DOMAIN_POLICY = "m5-qqr-padded-erfc/v1"
DIRECT_SR_IMAGE_PRECISION = 1.0e-6
OVERLAP_FOLD_SUPPORT_SCHEMA = (
    "vibeqc.aiccm2026dev-b.overlap-fold-support/v1"
)
OVERLAP_FOLD_SUPPORT_KEYS = {
    "schema",
    "qualification",
    "reason",
    "backend",
    "applicability",
    "max_k_drift",
    "cutoff_bohr",
    "diagnostic",
    "reference_cutoff_factor",
    "character_mesh_shape",
    "quantitative_target",
    "stop_threshold",
}
OVERLAP_FOLD_DIAGNOSTIC = "max-abs-element-overlap-bloch-fold-drift/v1"
OVERLAP_FOLD_REFERENCE_CUTOFF_FACTOR = 1.5
OVERLAP_FOLD_QUANTITATIVE_TARGET = 1.0e-4
OVERLAP_FOLD_STOP_THRESHOLD = 1.0e-2
EWALD_SHIFTED_PAIR_SUPPORT_SCHEMA = (
    "vibeqc.aiccm2026dev-b.ewald-shifted-pair-support/v2"
)
EWALD_SHIFTED_PAIR_SUPPORT_KEYS = {
    "schema",
    "qualification",
    "reason",
    "implementation",
    "repair_commit",
    "producer_commit",
    "core_build_id",
    "probe_attestation_id",
    "repair_is_ancestor",
    "canary",
}
EWALD_SHIFTED_PAIR_IMPLEMENTATION = (
    "centered-displacement-interplanar-pair-bounds"
)
EWALD_SHIFTED_PAIR_REPAIR_COMMIT = (
    "e578b86c00268a172b651cae29c7845836e31e17"
)
EWALD_SHIFTED_PAIR_CANARY_SCHEMA = (
    "vibeqc.ewald.shifted-pair-canary/v1"
)
EWALD_SHIFTED_PAIR_CANARY_KEYS = {
    "schema",
    "fixture",
    "real_cutoff_bohr",
    "lattice_shift",
    "energy_ha",
    "shifted_energy_ha",
    "abs_delta_ha",
    "tolerance_ha",
    "passed",
}
EWALD_SHIFTED_PAIR_CANARY_FIXTURE = (
    "mgo-point-charges-basis-shift-37a1/c18"
)
EWALD_SHIFTED_PAIR_CANARY_TOLERANCE = 1.0e-10
EXACT_EXCHANGE_ASSEMBLY_SCHEMA = (
    "vibeqc.aiccm2026dev-b.exact-exchange-assembly/v1"
)
EXACT_EXCHANGE_ASSEMBLY_KEYS = {
    "schema",
    "c_full",
    "c_sr",
    "omega_screen_bohr_inv",
    "resolver",
    "screened_exchange_applicability",
    "screened_exchange_assembly",
}
EXACT_EXCHANGE_RESOLVER = (
    "vibeqc.periodic_screened_exchange.resolve_periodic_exchange"
)
SCREENED_EXCHANGE_ASSEMBLY_BY_ROUTE: dict[str, tuple[str, str]] = {
    "rhf-4c": ("inactive", "not-applicable"),
    "rhf-ri": ("inactive", "not-applicable"),
    "rhf-rijcosx": ("inactive", "not-applicable"),
    "rks-pbe-4c": ("inactive", "not-applicable"),
    "rks-pbe-ri": ("inactive", "not-applicable"),
    "rks-pbe-rijcosx": ("inactive", "not-applicable"),
    "rks-pbe0-4c": ("inactive", "not-applicable"),
    "rks-pbe0-ri": ("inactive", "not-applicable"),
    "rks-pbe0-rijcosx": ("inactive", "not-applicable"),
    "ri-mp2": ("inactive", "not-applicable"),
    "dlpno-mp2": ("inactive", "not-applicable"),
    "dlpno-ccsd": ("inactive", "not-applicable"),
    "dlpno-ccsd-t": ("inactive", "not-applicable"),
}
_MISSING = object()


@dataclass(frozen=True)
class Route:
    """One independently selectable χ-CCM calculation route."""

    method: str
    backend: str
    functional: str | None = None
    post_hf: bool = False
    exchange_q0_applicability: str = "active"


ROUTES: dict[str, Route] = {
    "rhf-4c": Route("RHF", "four_center"),
    "rhf-ri": Route("RHF", "ri"),
    "rhf-rijcosx": Route("RHF", "rijcosx"),
    "rks-pbe-4c": Route(
        "RKS",
        "four_center",
        "pbe",
        exchange_q0_applicability="inactive",
    ),
    "rks-pbe-ri": Route(
        "RKS",
        "ri",
        "pbe",
        exchange_q0_applicability="inactive",
    ),
    "rks-pbe-rijcosx": Route(
        "RKS",
        "rijcosx",
        "pbe",
        exchange_q0_applicability="inactive",
    ),
    "rks-pbe0-4c": Route("RKS", "four_center", "pbe0"),
    "rks-pbe0-ri": Route("RKS", "ri", "pbe0"),
    "rks-pbe0-rijcosx": Route("RKS", "rijcosx", "pbe0"),
    "ri-mp2": Route("RI-MP2", "ri", post_hf=True),
    "dlpno-mp2": Route("DLPNO-MP2", "ri", post_hf=True),
    "dlpno-ccsd": Route("DLPNO-CCSD", "ri", post_hf=True),
    "dlpno-ccsd-t": Route("DLPNO-CCSD(T)", "ri", post_hf=True),
}

SCF_ROUTES = tuple(name for name, route in ROUTES.items() if not route.post_hf)
POST_HF_ROUTES = tuple(name for name, route in ROUTES.items() if route.post_hf)

EXACT_EXCHANGE_ASSEMBLY_BY_ROUTE: dict[
    str,
    tuple[float, float, float],
] = {
    "rhf-4c": (1.0, 0.0, 0.0),
    "rhf-ri": (1.0, 0.0, 0.0),
    "rhf-rijcosx": (1.0, 0.0, 0.0),
    "rks-pbe-4c": (0.0, 0.0, 0.0),
    "rks-pbe-ri": (0.0, 0.0, 0.0),
    "rks-pbe-rijcosx": (0.0, 0.0, 0.0),
    "rks-pbe0-4c": (0.25, 0.0, 0.0),
    "rks-pbe0-ri": (0.25, 0.0, 0.0),
    "rks-pbe0-rijcosx": (0.25, 0.0, 0.0),
    "ri-mp2": (1.0, 0.0, 0.0),
    "dlpno-mp2": (1.0, 0.0, 0.0),
    "dlpno-ccsd": (1.0, 0.0, 0.0),
    "dlpno-ccsd-t": (1.0, 0.0, 0.0),
}
if set(EXACT_EXCHANGE_ASSEMBLY_BY_ROUTE) != set(ROUTES):
    raise RuntimeError(
        "EXACT_EXCHANGE_ASSEMBLY_BY_ROUTE must cover b_routes.ROUTES exactly"
    )
if set(SCREENED_EXCHANGE_ASSEMBLY_BY_ROUTE) != set(ROUTES):
    raise RuntimeError(
        "SCREENED_EXCHANGE_ASSEMBLY_BY_ROUTE must cover b_routes.ROUTES exactly"
    )


def _finite_positive(value: object) -> float | None:
    """Return one finite positive JSON number, excluding booleans."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if isfinite(number) and number > 0.0 else None


def _finite_number(value: object) -> float | None:
    """Return one finite JSON number, excluding booleans."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _full_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def exact_exchange_assembly_failure(
    value: object,
    route_name: object,
) -> str | None:
    """Validate one exact route-resolved exchange-assembly descriptor.

    This is an independent fleet route oracle for the versioned serialized
    coefficient and screened-assembly provenance.
    """

    route = ROUTES.get(route_name) if isinstance(route_name, str) else None
    if route is None:
        return "the result route is not registered in b_routes.ROUTES"
    expected = EXACT_EXCHANGE_ASSEMBLY_BY_ROUTE[route_name]

    expected_applicability = "active" if expected[0] != 0.0 else "inactive"
    if route.exchange_q0_applicability != expected_applicability:
        return (
            "the route exchange_q0_applicability contradicts its expected "
            "full-range exact-exchange coefficient"
        )

    if not isinstance(value, dict):
        return "exact_exchange_assembly is missing or is not an object"
    if set(value) != EXACT_EXCHANGE_ASSEMBLY_KEYS:
        return "exact_exchange_assembly does not have the exact keys for v1"
    if value.get("schema") != EXACT_EXCHANGE_ASSEMBLY_SCHEMA:
        return "exact_exchange_assembly has the wrong schema identifier"
    if value.get("resolver") != EXACT_EXCHANGE_RESOLVER:
        return "exact_exchange_assembly has the wrong resolver"

    actual = tuple(
        _finite_number(value.get(name))
        for name in ("c_full", "c_sr", "omega_screen_bohr_inv")
    )
    if any(number is None for number in actual):
        return (
            "exact_exchange_assembly coefficients must be finite JSON "
            "numbers excluding booleans"
        )
    if actual != expected:
        return "exact_exchange_assembly coefficients contradict the route selector"
    screened = (
        value.get("screened_exchange_applicability"),
        value.get("screened_exchange_assembly"),
    )
    if screened != SCREENED_EXCHANGE_ASSEMBLY_BY_ROUTE[route_name]:
        return (
            "exact_exchange_assembly screened branch contradicts the "
            "route selector"
        )
    expected_screened_applicability = (
        "active" if actual[1] != 0.0 else "inactive"
    )
    if screened[0] != expected_screened_applicability:
        return (
            "exact_exchange_assembly screened_exchange_applicability "
            "contradicts c_sr"
        )
    if screened[0] == "inactive" and screened[1] != "not-applicable":
        return (
            "inactive screened exchange must use assembly='not-applicable'"
        )
    if screened[0] == "active" and screened[1] not in {
        "short-range-direct",
        "full-range-minus-long-range",
    }:
        return "active screened exchange has an invalid assembly convention"
    return None


def _allowed_runtime_backends(route: Route) -> set[str]:
    """Return the low-level backends that can execute one fleet route."""

    if route.backend == "four_center":
        return {"pbc-bipole"}

    reference_method = "rhf" if route.post_hf else route.method.lower()
    exchange_backend = "cosx" if route.backend == "rijcosx" else "gdf"
    values = {
        f"native-multi-k-gdf-{exchange_backend}-{reference_method}",
    }
    if route.backend == "ri":
        values.add("native-gamma-gdf-via-k-gdf")
    return values | {f"{value}+PARITY_HELD" for value in values}


def two_electron_support_schema_failure(
    value: object,
    route_name: object,
) -> str | None:
    """Validate the exact route-specific two-electron support v1 payload.

    A valid fitted payload is deliberately still ``not-qualified``.  Call
    :func:`two_electron_support_reportability_failure` when deciding whether
    an energy may enter a quantitative audit or comparison.
    """

    route = ROUTES.get(route_name) if isinstance(route_name, str) else None
    if route is None:
        return "the result route is not registered in b_routes.ROUTES"
    if not isinstance(value, dict):
        return "two_electron_support is missing or is not an object"
    if set(value) != TWO_ELECTRON_SUPPORT_KEYS:
        return "two_electron_support does not have the exact v1 top-level keys"
    if value.get("schema") != TWO_ELECTRON_SUPPORT_SCHEMA:
        return "two_electron_support has the wrong schema identifier"
    qualification = value.get("qualification")
    if qualification not in {"qualified", "not-qualified"}:
        return "two_electron_support has an invalid qualification"
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return "two_electron_support reason must be nonempty text"
    if value.get("backend") != route.backend:
        return "two_electron_support backend contradicts the route selector"
    runtime_backend = value.get("runtime_backend")
    if not isinstance(runtime_backend, str) or not runtime_backend.strip():
        return "two_electron_support runtime_backend must be nonempty text"
    if runtime_backend not in _allowed_runtime_backends(route):
        return "two_electron_support runtime_backend contradicts the route selector"

    direct = value.get("direct")
    fitted = value.get("fitted")
    if route.backend == "four_center":
        if fitted is not None:
            return "four-center support must set fitted=null"
        if not isinstance(direct, dict) or set(direct) != TWO_ELECTRON_DIRECT_KEYS:
            return "four-center support does not have the exact v1 direct keys"
        if direct.get("domain_policy") != DIRECT_DOMAIN_POLICY:
            return "four-center support has the wrong direct domain policy"
        precision = _finite_positive(direct.get("sr_image_precision"))
        if precision != DIRECT_SR_IMAGE_PRECISION:
            return "four-center support has the wrong M5 SR image precision"
        electronic = _finite_positive(direct.get("electronic_cutoff_bohr"))
        nuclear = _finite_positive(direct.get("nuclear_cutoff_bohr"))
        extent = _finite_positive(direct.get("sr_image_extent_bohr"))
        if electronic is None or nuclear is None or extent is None:
            return (
                "four-center support cutoffs and image extent must be finite "
                "and positive"
            )
        if nuclear > electronic:
            return "four-center nuclear cutoff exceeds the electronic cutoff"
        if extent <= electronic:
            return "four-center SR image extent does not exceed the electronic cutoff"
        if qualification != "qualified":
            return "complete four-center M5 support must be qualification=qualified"
        return None

    if direct is not None:
        return "fitted support must set direct=null"
    if not isinstance(fitted, dict) or set(fitted) != TWO_ELECTRON_FITTED_KEYS:
        return "fitted support does not have the exact v1 fitted keys"
    gdf_method = fitted.get("gdf_method")
    if gdf_method not in {"rsgdf", "mdf"}:
        return "fitted support has an invalid gdf_method"
    if gdf_method == "rsgdf":
        if _finite_positive(fitted.get("rsgdf_base_ke_cutoff_ha")) is None:
            return "RSGDF support needs a finite positive base KE cutoff"
        if fitted.get("rsgdf_tail_ke_cutoff_ha") is not None:
            return "RSGDF support must record the unattested tail cutoff as null"
        if fitted.get("mdf_ke_cutoff_ha") is not None:
            return "RSGDF support must set the inactive MDF cutoff to null"
    else:
        if fitted.get("rsgdf_base_ke_cutoff_ha") is not None:
            return "MDF support must set the inactive RSGDF base cutoff to null"
        if fitted.get("rsgdf_tail_ke_cutoff_ha") is not None:
            return "MDF support must set the inactive RSGDF tail cutoff to null"
        if _finite_positive(fitted.get("mdf_ke_cutoff_ha")) is None:
            return "MDF support needs a finite positive KE cutoff"
    expected_cosx = "not-attested" if route.backend == "rijcosx" else "not-applicable"
    if fitted.get("cosx_exchange_support") != expected_cosx:
        return "fitted COSX support contradicts the route selector"
    expected_correlation = "not-attested" if route.post_hf else "not-applicable"
    if fitted.get("correlation_support") != expected_correlation:
        return "fitted correlation support contradicts the route selector"
    if qualification != "not-qualified":
        return "fitted and post-HF support cannot claim qualification=qualified"
    return None


def two_electron_support_reportability_failure(
    value: object,
    route_name: object,
    direct_lattice_cutoffs: object = _MISSING,
) -> str | None:
    """Return why one support payload cannot qualify a quantitative B row.

    When the containing record's legacy ``direct_lattice_cutoffs`` value is
    supplied, the v1 payload must agree with that independent serialization.
    """

    failure = two_electron_support_schema_failure(value, route_name)
    if failure is not None:
        return failure
    assert isinstance(value, dict)
    route = ROUTES[route_name]
    if direct_lattice_cutoffs is not _MISSING:
        if route.backend == "four_center":
            if (
                not isinstance(direct_lattice_cutoffs, dict)
                or set(direct_lattice_cutoffs)
                != {"electronic_cutoff_bohr", "nuclear_cutoff_bohr"}
            ):
                return (
                    "four-center direct_lattice_cutoffs is missing or does not "
                    "have the exact keys"
                )
            direct = value["direct"]
            assert isinstance(direct, dict)
            for name in (
                "electronic_cutoff_bohr",
                "nuclear_cutoff_bohr",
            ):
                legacy_number = _finite_positive(direct_lattice_cutoffs.get(name))
                if legacy_number is None or legacy_number != float(direct[name]):
                    return (
                        "two_electron_support contradicts direct_lattice_cutoffs "
                        f"for {name}"
                    )
        elif direct_lattice_cutoffs is not None:
            return "fitted support requires direct_lattice_cutoffs=null"
    if value["qualification"] != "qualified":
        return f"two-electron support is not-qualified: {value['reason']}"
    return None


def overlap_fold_support_schema_failure(
    value: object,
    route_name: object,
) -> str | None:
    """Validate exact direct overlap-fold support provenance for one route."""

    route = ROUTES.get(route_name) if isinstance(route_name, str) else None
    if route is None:
        return "the result route is not registered in b_routes.ROUTES"
    if not isinstance(value, dict):
        return "overlap_fold_support is missing or is not an object"
    if set(value) != OVERLAP_FOLD_SUPPORT_KEYS:
        return "overlap_fold_support does not have the exact v1 keys"
    if value.get("schema") != OVERLAP_FOLD_SUPPORT_SCHEMA:
        return "overlap_fold_support has the wrong schema identifier"
    if value.get("backend") != route.backend:
        return "overlap_fold_support backend contradicts the route selector"
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return "overlap_fold_support reason must be nonempty text"
    if _finite_number(value.get("quantitative_target")) != (
        OVERLAP_FOLD_QUANTITATIVE_TARGET
    ):
        return "overlap_fold_support has the wrong quantitative target"
    if _finite_number(value.get("stop_threshold")) != (
        OVERLAP_FOLD_STOP_THRESHOLD
    ):
        return "overlap_fold_support has the wrong stop threshold"

    qualification = value.get("qualification")
    applicability = value.get("applicability")
    drift = value.get("max_k_drift")
    cutoff = value.get("cutoff_bohr")
    if route.backend != "four_center":
        if applicability != "not-applicable":
            return "fitted overlap-fold support must be not-applicable"
        if qualification != "not-applicable":
            return "fitted overlap-fold qualification must be not-applicable"
        if drift is not None or cutoff is not None:
            return "fitted overlap-fold support must set drift and cutoff to null"
        if value.get("diagnostic") != "not-applicable":
            return "fitted overlap-fold diagnostic must be not-applicable"
        if value.get("reference_cutoff_factor") is not None:
            return "fitted overlap-fold reference factor must be null"
        if value.get("character_mesh_shape") is not None:
            return "fitted overlap-fold character mesh must be null"
        return None

    if applicability != "active":
        return "four-center overlap-fold support must be active"
    if qualification not in {"qualified", "not-qualified"}:
        return "four-center overlap-fold support has an invalid qualification"
    cutoff_number = _finite_positive(cutoff)
    if cutoff_number is None:
        return "four-center overlap-fold cutoff must be finite and positive"
    if value.get("diagnostic") != OVERLAP_FOLD_DIAGNOSTIC:
        return "four-center overlap-fold support has the wrong diagnostic"
    if _finite_number(value.get("reference_cutoff_factor")) != (
        OVERLAP_FOLD_REFERENCE_CUTOFF_FACTOR
    ):
        return "four-center overlap-fold support has the wrong reference factor"
    mesh = value.get("character_mesh_shape")
    if (
        not isinstance(mesh, list)
        or len(mesh) != 3
        or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item <= 0
            for item in mesh
        )
    ):
        return "four-center overlap-fold character mesh must be three positive integers"
    if drift is None:
        if qualification != "not-qualified":
            return "missing overlap-fold drift cannot be qualification=qualified"
        return None
    drift_number = _finite_number(drift)
    if drift_number is None or drift_number < 0.0:
        return "four-center overlap-fold drift must be finite and nonnegative"
    if drift_number > OVERLAP_FOLD_STOP_THRESHOLD:
        return "four-center overlap-fold drift exceeds the fail-closed threshold"
    expected = (
        "qualified"
        if drift_number <= OVERLAP_FOLD_QUANTITATIVE_TARGET
        else "not-qualified"
    )
    if qualification != expected:
        return "four-center overlap-fold qualification contradicts its drift"
    return None


def overlap_fold_support_reportability_failure(
    value: object,
    route_name: object,
    direct_lattice_cutoffs: object = _MISSING,
    character_mesh: object = _MISSING,
) -> str | None:
    """Return why one row lacks quantitative overlap-fold support."""

    failure = overlap_fold_support_schema_failure(value, route_name)
    if failure is not None:
        return failure
    assert isinstance(value, dict)
    route = ROUTES[route_name]
    if route.backend != "four_center":
        return None
    if direct_lattice_cutoffs is not _MISSING:
        if (
            not isinstance(direct_lattice_cutoffs, dict)
            or set(direct_lattice_cutoffs)
            != {"electronic_cutoff_bohr", "nuclear_cutoff_bohr"}
        ):
            return (
                "four-center direct_lattice_cutoffs is missing or does not "
                "have the exact keys"
            )
        electronic = _finite_positive(
            direct_lattice_cutoffs.get("electronic_cutoff_bohr")
        )
        if electronic is None or electronic != float(value["cutoff_bohr"]):
            return "overlap_fold_support contradicts the electronic cutoff"
    if character_mesh is not _MISSING:
        if (
            not isinstance(character_mesh, (list, tuple))
            or len(character_mesh) != 3
            or any(
                not isinstance(item, int)
                or isinstance(item, bool)
                or item <= 0
                for item in character_mesh
            )
        ):
            return "the row character mesh must be three positive integers"
        if list(character_mesh) != value["character_mesh_shape"]:
            return "overlap_fold_support contradicts the character mesh"
    if value["qualification"] != "qualified":
        return f"overlap-fold support is not-qualified: {value['reason']}"
    return None


def ewald_shifted_pair_support_schema_failure(value: object) -> str | None:
    """Validate exact post-repair D104 support evidence."""

    if not isinstance(value, dict):
        return "ewald_shifted_pair_support is missing or is not an object"
    if set(value) != EWALD_SHIFTED_PAIR_SUPPORT_KEYS:
        return "ewald_shifted_pair_support does not have the exact v2 keys"
    if value.get("schema") != EWALD_SHIFTED_PAIR_SUPPORT_SCHEMA:
        return "ewald_shifted_pair_support has the wrong schema identifier"
    qualification = value.get("qualification")
    if not isinstance(qualification, str) or qualification not in {
        "qualified",
        "not-qualified",
    }:
        return "ewald_shifted_pair_support has an invalid qualification"
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return "ewald_shifted_pair_support reason must be nonempty text"
    if value.get("implementation") != EWALD_SHIFTED_PAIR_IMPLEMENTATION:
        return "ewald_shifted_pair_support has the wrong implementation state"
    if value.get("repair_commit") != EWALD_SHIFTED_PAIR_REPAIR_COMMIT:
        return "ewald_shifted_pair_support has the wrong repair commit"
    producer_commit = value.get("producer_commit")
    if producer_commit is not None and not _full_commit(producer_commit):
        return "ewald_shifted_pair_support producer commit is malformed"
    for key in ("core_build_id", "probe_attestation_id"):
        identity = value.get(key)
        if identity is not None and not _sha256(identity):
            return f"ewald_shifted_pair_support {key} is malformed"
    repair_is_ancestor = value.get("repair_is_ancestor")
    if repair_is_ancestor is not None and type(repair_is_ancestor) is not bool:
        return "ewald_shifted_pair_support repair ancestry must be boolean or null"
    canary = value.get("canary")
    if not isinstance(canary, dict):
        return "ewald_shifted_pair_support canary is missing or is not an object"
    if set(canary) != EWALD_SHIFTED_PAIR_CANARY_KEYS:
        return "ewald_shifted_pair_support canary does not have the exact v1 keys"
    if canary.get("schema") != EWALD_SHIFTED_PAIR_CANARY_SCHEMA:
        return "ewald_shifted_pair_support canary has the wrong schema"
    if canary.get("fixture") != EWALD_SHIFTED_PAIR_CANARY_FIXTURE:
        return "ewald_shifted_pair_support canary has the wrong fixture"
    if _finite_number(canary.get("real_cutoff_bohr")) != 18.0:
        return "ewald_shifted_pair_support canary has the wrong cutoff"
    lattice_shift = canary.get("lattice_shift")
    if (
        not isinstance(lattice_shift, list)
        or len(lattice_shift) != 3
        or any(type(item) is not int for item in lattice_shift)
        or lattice_shift != [37, 0, 0]
    ):
        return "ewald_shifted_pair_support canary has the wrong lattice shift"
    if _finite_number(canary.get("tolerance_ha")) != (
        EWALD_SHIFTED_PAIR_CANARY_TOLERANCE
    ):
        return "ewald_shifted_pair_support canary has the wrong tolerance"
    if type(canary.get("passed")) is not bool:
        return "ewald_shifted_pair_support canary passed flag must be boolean"
    canary_energy = _finite_number(canary.get("energy_ha"))
    shifted_energy = _finite_number(canary.get("shifted_energy_ha"))
    canary_delta = _finite_number(canary.get("abs_delta_ha"))
    raw_results = (
        canary.get("energy_ha"),
        canary.get("shifted_energy_ha"),
        canary.get("abs_delta_ha"),
    )
    results_missing = all(item is None for item in raw_results)
    results_finite = all(
        item is not None
        for item in (canary_energy, shifted_energy, canary_delta)
    )
    if not results_missing and not results_finite:
        return "ewald_shifted_pair_support canary has incomplete results"
    if results_finite:
        assert canary_energy is not None
        assert shifted_energy is not None
        assert canary_delta is not None
        expected_passed = (
            canary_delta >= 0.0
            and abs(canary_delta - abs(shifted_energy - canary_energy))
            <= 1.0e-15
            and canary_delta <= EWALD_SHIFTED_PAIR_CANARY_TOLERANCE
        )
        if canary.get("passed") is not expected_passed:
            return "ewald_shifted_pair_support canary contradicts its result"
    elif canary.get("passed") is True:
        return "ewald_shifted_pair_support canary contradicts its result"
    if value.get("qualification") == "qualified" and (
        not _full_commit(producer_commit)
        or not _sha256(value.get("core_build_id"))
        or not _sha256(value.get("probe_attestation_id"))
        or value.get("repair_is_ancestor") is not True
        or canary.get("passed") is not True
    ):
        return (
            "qualified ewald_shifted_pair_support requires a complete "
            "post-repair producer identity"
        )
    return None


def ewald_shifted_pair_support_reportability_failure(
    value: object,
    provenance: object = _MISSING,
) -> str | None:
    """Return why one row lacks post-repair shared-Ewald support."""

    failure = ewald_shifted_pair_support_schema_failure(value)
    if failure is not None:
        return failure
    assert isinstance(value, dict)
    if value["qualification"] != "qualified":
        return f"ewald_shifted_pair_support is not-qualified: {value['reason']}"
    if provenance is _MISSING:
        return None
    if not isinstance(provenance, dict):
        return "qualified ewald_shifted_pair_support lacks producer provenance"
    producer_commit = provenance.get(
        "vibeqc_commit",
        provenance.get("source_commit"),
    )
    aliases = {
        "producer_commit": producer_commit,
        "core_build_id": provenance.get("core_build_id"),
        "probe_attestation_id": provenance.get("probe_attestation_id"),
    }
    if any(value.get(key) != expected for key, expected in aliases.items()):
        return "ewald_shifted_pair_support contradicts producer provenance"
    if (
        provenance.get("source_clean") is not True
        or provenance.get("probe_passed") is not True
    ):
        return "ewald_shifted_pair_support producer provenance is not D77-clean"
    attestation = provenance.get("probe_attestation")
    if not isinstance(attestation, dict):
        return "ewald_shifted_pair_support lacks a finalized probe attestation"
    normalized_provenance = dict(provenance)
    normalized_provenance.setdefault("vibeqc_commit", producer_commit)
    normalized_provenance.setdefault(
        "probe_version",
        provenance.get("attestation_mode"),
    )
    expected_payload = attestation.get("producer_payload")
    attestation_failure = probe_host.serialized_producer_attestation_failure(
        normalized_provenance,
        expected_payload,
        expected_source_commit=producer_commit,
    )
    if attestation_failure is not None:
        return (
            "ewald_shifted_pair_support lacks finalized D77 provenance: "
            f"{attestation_failure}"
        )
    return None


def d102_absolute_energy_revision_failure(
    value: object,
    provenance: object = _MISSING,
) -> str | None:
    """Compatibility spelling for the D102/D104 reportability gate."""

    return ewald_shifted_pair_support_reportability_failure(value, provenance)


def unsupported_reason(meta: dict, route_name: str, mesh: tuple[int, int, int]) -> str | None:
    """Return why a registered input cannot run in B, or ``None``."""

    route = ROUTES[route_name]
    if bool(meta.get("open_shell", False)):
        return "this paper-1 B route matrix is restricted to closed-shell references"
    if int(meta["dim"]) != 3:
        if route.post_hf:
            return "aiccm2026dev-b post-HF is currently restricted to 3D"
        return (
            "aiccm2026dev-b SCF is restricted to 3D until one neutral "
            "wire/slab Coulomb kernel is shared by all backends"
        )
    if prod(mesh) == 1 and route.backend == "rijcosx":
        return "the native finite-torus COSX bridge requires at least two cells"
    if prod(mesh) == 1 and route.method == "RKS" and route.backend == "ri":
        return "the pair-resolved finite-torus RI-RKS path requires at least two cells"
    return None
