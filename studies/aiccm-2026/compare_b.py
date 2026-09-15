#!/usr/bin/env python3
"""Compare χ-CCM records with real-Gamma controls and CRYSTAL values."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path

from b_routes import (
    ROUTES,
    d102_absolute_energy_revision_failure,
    exact_exchange_assembly_failure,
    overlap_fold_support_reportability_failure,
    two_electron_support_reportability_failure,
)

HA_TO_MHA = 1000.0
EXPECTED_EXCHANGE_Q0 = "bvk-ewald"
EXPECTED_LATTICE_VECTOR_CONVENTION = "columns"
EXPECTED_CCM_APPROACH = "chi-ccm"
EXPECTED_CCM_CONSTRUCTION = "finite-translation-group-character"
EXPECTED_EVALUATION_REPRESENTATION = "gamma-centred-character-mesh"
KNOWN_B_STATUSES = {"ok", "not_converged", "unsupported", "error"}
EXPECTED_REAL_GAMMA_CONTROL_IDENTITY = {
    "ccm_approach": "not-applicable",
    "ccm_construction": "neutral-fitted-torus",
    "evaluation_representation": "real-gamma-supercell",
    "route_role": "representation-control",
}
EXPECTED_PROBE_VERSION = "aiccm-host-probe/v2"
EXPECTED_PRODUCER_ATTESTATION_VERSION = "aiccm-producer-attestation/v1"
EXPECTED_PRODUCER_PAYLOAD_VERSION = "aiccm-producer-payload/v1"
EXPECTED_LOADED_CORE_CHECK_VERSION = "aiccm-loaded-core-mirror/v1"
# Frozen legacy literals retained solely to validate existing synthetic-control
# provenance.  Despite their historical spelling, they do not attest a
# Gamma-CCM union-and-weight construction or a Gamma-CCM/chi-CCM comparison.
REAL_GAMMA_CONTROL_CONTRACT_VERSION = "aiccm2026-gamma-chi/v2"
REAL_GAMMA_CONTROL_INPUT_VERSION = "vibeqc.aiccm2026.gamma-chi-input/v2"
_PREFLIGHT_ATTESTATION_FIELDS = {
    "probe_version",
    "probe_script_id",
    "probe_passed",
    "host",
    "vibeqc_version",
    "core_build_id",
    "source_commit",
    "source_clean",
}
_PRODUCER_ATTESTATION_FIELDS = {
    "attestation_version",
    "attestor_script_id",
    "probe_passed",
    "host",
    "vibeqc_version",
    "core_build_id",
    "source_commit",
    "source_clean",
    "native_library_versions",
    "producer_payload",
    "producer_payload_id",
    "preflight_attestation",
    "preflight_attestation_id",
    "current_core_attestation",
    "current_core_attestation_id",
    "core_changed_since_preflight",
    "loaded_core_check",
    "result_identity_stable",
}
_NATIVE_LIBRARY_KEYS = {
    "libint",
    "libxc",
    "spglib",
    "libecpint",
    "fftw3",
    "blas",
}
_LOADED_CORE_CHECK_FIELDS = {
    "version",
    "passed",
    "fixture",
    "n_ao",
    "max_l",
    "n_cells",
    "n_frequencies",
    "atol",
    "rtol",
    "max_abs_cxx_python",
    "max_scaled_error",
    "reference_transpose_asymmetry",
}
_PRODUCER_STREAM_FILES = {
    "aiccm2026dev-a": {"producer", "probe_host", "testset", "launcher"},
    "aiccm2026dev-b": {
        "producer",
        "probe_host",
        "testset",
        "launcher",
        "b_routes",
    },
}
GAMMA_CHI_APPROACH_ROUTE_FOR_B: dict[str, str] = {}

REAL_GAMMA_CONTROL_ROUTE_FOR_B = {
    # This is a neutral fitted-torus character-vs-real-Gamma representation
    # control.  The real-Gamma route independently assembles and minimizes the
    # shared fitted Hamiltonian while reusing its numerical primitives.  It is
    # not evidence for the Gamma-CCM union-and-weight construction.
    "rhf-ri": "aiccm-hf-direct",
}
REAL_GAMMA_CONTROL_METHOD_FOR_ROUTE = {
    "aiccm-hf-direct": "RHF",
}


def _is_retracted(record: dict) -> bool:
    """Conservatively treat every truthy retraction marker as withdrawn."""

    return bool(record.get("retracted", False))


def _is_converged_success(
    record: dict | None,
    *,
    allow_missing_status: bool = False,
) -> bool:
    """Return whether a record is independently reportable as converged."""

    if not record or record.get("converged") is not True:
        return False
    return record.get("status") == "ok" or (
        allow_missing_status and "status" not in record
    )


def _finite_float(value: object) -> float | None:
    """Return a finite JSON number without accepting booleans or strings."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    result = float(value)
    return result if isfinite(result) else None


def _candidate_json_paths(directory: str | Path) -> list[Path]:
    """Return raw run outputs and curated nested result records."""

    root = Path(directory)
    paths = set(root.rglob("*__*.json"))
    paths.update(root.rglob("result.json"))
    return sorted(paths)


def _functional_key(record: dict) -> str | None:
    """Return the normalized functional component of a result identity."""

    route = ROUTES.get(record.get("route"))
    value = route.functional if route is not None else record.get("functional")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().casefold()


def _run_meta(path: Path) -> dict[str, str] | None:
    """Read curated producer metadata; raw run JSON has no sidecar contract."""

    if path.name != "result.json":
        return None
    sidecar = path.with_name("RUN.meta")
    if not sidecar.is_file():
        return {}
    values: dict[str, str] = {}
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator:
            key = key.strip()
            value = value.strip()
            if key in values and values[key] != value:
                values["_invalid"] = f"conflicting {key} declarations"
            values[key] = value
    return values


def load(directory: str | Path) -> dict[tuple[str, str, str | None], dict]:
    records: dict[tuple[str, str, str | None], dict] = {}
    sources: dict[tuple[str, str, str | None], Path] = {}
    for path in _candidate_json_paths(directory):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed JSON result candidate {path}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"result candidate {path} is not a JSON object")
        record = dict(record)
        record.pop("_run_meta", None)
        system = record.get("system")
        route = record.get("route")
        if not isinstance(system, str) or not system.strip():
            raise ValueError(
                f"result candidate {path} has a missing or invalid system"
            )
        if not isinstance(route, str) or not route.strip():
            raise ValueError(
                f"result candidate {path} has a missing or invalid route"
            )
        run_meta = _run_meta(path)
        if run_meta is not None:
            record["_run_meta"] = run_meta
        key = (system, route, _functional_key(record))
        if key in records:
            previous = records[key]
            raise ValueError(
                "duplicate result records for "
                f"{key[0]}/{key[1]}/{key[2] or 'hf'}; select exactly one "
                "basis and mesh for each functional "
                "before comparing: "
                f"{sources[key]} (basis={previous.get('basis')}, "
                f"mesh={previous.get('mesh')}) and {path} "
                f"(basis={record.get('basis')}, mesh={record.get('mesh')})"
            )
        records[key] = record
        sources[key] = path
    return records


def _sorted_records(records: dict) -> list[tuple[tuple, dict]]:
    """Sort functional-aware keys without comparing ``None`` with strings."""

    return sorted(
        records.items(),
        key=lambda item: tuple("" if value is None else value for value in item[0]),
    )


def _energy_per_atom(
    record: dict | None,
    *,
    allow_missing_status: bool = False,
) -> float | None:
    if not _is_converged_success(
        record,
        allow_missing_status=allow_missing_status,
    ) or _is_retracted(record):
        return None
    assert record is not None
    value = record.get("energy_per_atom_ha", record.get("energy_per_atom"))
    return _finite_float(value)


def _has_unreportable_b_dimension(record: dict) -> bool:
    """Return whether a B record carries a reportable-status low-D energy."""

    return (
        record.get("status") in ("ok", "not_converged")
        and record.get("dim") != 3
    )


def _b_two_electron_support_failure(record: dict) -> str | None:
    """Independently gate quantitative B rows on D82 support evidence."""

    return two_electron_support_reportability_failure(
        record.get("two_electron_support"),
        record.get("route"),
        record.get("direct_lattice_cutoffs"),
    )


def _b_exact_exchange_assembly_failure(record: dict) -> str | None:
    """Independently gate B rows on route-resolved exchange provenance."""

    return exact_exchange_assembly_failure(
        record.get("exact_exchange_assembly"),
        record.get("route"),
    )


def _b_overlap_fold_support_failure(record: dict) -> str | None:
    """Independently gate direct B rows on D103 fold evidence."""

    return overlap_fold_support_reportability_failure(
        record.get("overlap_fold_support"),
        record.get("route"),
        record.get("direct_lattice_cutoffs"),
        record.get("mesh"),
    )


def _normalized_text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().casefold()


def _normalized_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    label = _normalized_text(value)
    if label in {"true", "1", "yes"}:
        return True
    if label in {"false", "0", "no"}:
        return False
    return None


def _b_route_identity_matches(route_name: str, record: dict) -> bool:
    """Require a B payload to agree with its public route selector."""

    route = ROUTES.get(route_name)
    if route is None or record.get("route") != route_name or any(
        field not in record for field in ("method", "backend", "functional")
    ):
        return False
    return (
        _normalized_text(record.get("method")) == _normalized_text(route.method)
        and _normalized_text(record.get("backend"))
        == _normalized_text(route.backend)
        and _normalized_text(record.get("functional"))
        == _normalized_text(route.functional)
    )


def _b_construction_identity_matches(record: dict) -> bool:
    """Require the exact χ construction and route-level seam applicability."""

    convention = record.get("finite_torus_convention")
    route_name = record.get("route")
    route = ROUTES.get(route_name) if isinstance(route_name, str) else None
    return isinstance(convention, dict) and route is not None and all(
        convention.get(key) == expected
        for key, expected in {
            "ccm_approach": EXPECTED_CCM_APPROACH,
            "ccm_construction": EXPECTED_CCM_CONSTRUCTION,
            "evaluation_representation": EXPECTED_EVALUATION_REPRESENTATION,
            "exchange_q0_applicability": route.exchange_q0_applicability,
        }.items()
    )


def _real_gamma_control_identity_matches(record: dict) -> bool:
    """Require the exact role emitted for the neutral real-Gamma control."""

    return all(
        record.get(key) == expected
        for key, expected in EXPECTED_REAL_GAMMA_CONTROL_IDENTITY.items()
    )


def _exchange_q0(record: dict) -> str | None:
    convention = record.get("finite_torus_convention")
    declarations = []
    if isinstance(convention, dict) and "exchange_q0" in convention:
        declarations.append(convention["exchange_q0"])
    if "exchange_q0" in record:
        declarations.append(record["exchange_q0"])
    labels = [_normalized_text(value) for value in declarations]
    if not labels or any(label is None for label in labels):
        return None
    normalized = [label.replace("_", "-") for label in labels if label]
    if any(label != normalized[0] for label in normalized):
        return None
    return normalized[0]


def _mesh_shape(record: dict, key: str) -> tuple[int, int, int] | None:
    value = record.get(key)
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item <= 0
            for item in value
        )
    ):
        return None
    return tuple(value)


def _b_finite_torus_lattice_matches_input(record: dict) -> bool:
    """Bind a χ record to the column-vector BvK lattice it declares."""

    convention = record.get("finite_torus_convention")
    comparison_input = record.get("comparison_input")
    if not isinstance(convention, dict) or not isinstance(comparison_input, dict):
        return False
    if (
        convention.get("lattice_vector_convention")
        != EXPECTED_LATTICE_VECTOR_CONVENTION
    ):
        return False
    mesh = _mesh_shape(record, "mesh")
    character_mesh = _mesh_shape(convention, "character_mesh_shape")
    repetitions = _mesh_shape(
        convention,
        "bvk_madelung_supercell_repetitions",
    )
    if (
        mesh is None
        or character_mesh != mesh
        or repetitions != mesh
        or type(convention.get("periodic_dimension")) is not int
        or convention.get("periodic_dimension") != 3
    ):
        return False
    physical = comparison_input.get("system")
    if not isinstance(physical, dict):
        return False
    lattice = physical.get("lattice_bohr")
    primitive_lattice = record.get("primitive_lattice_bohr")
    bvk_lattice = convention.get("bvk_madelung_supercell_lattice_bohr")
    if not (
        isinstance(lattice, list)
        and isinstance(primitive_lattice, list)
        and isinstance(bvk_lattice, list)
        and len(lattice) == 3
        and len(primitive_lattice) == 3
        and len(bvk_lattice) == 3
        and all(_finite_vector(row, 3) for row in lattice)
        and all(_finite_vector(row, 3) for row in primitive_lattice)
        and all(_finite_vector(row, 3) for row in bvk_lattice)
    ):
        return False
    if any(
        abs(float(primitive_lattice[i][j]) - float(lattice[i][j])) > 1.0e-12
        for i in range(3)
        for j in range(3)
    ):
        return False
    return all(
        abs(
            float(bvk_lattice[i][j])
            - float(primitive_lattice[i][j]) * mesh[j]
        )
        <= 1.0e-12
        for i in range(3)
        for j in range(3)
    )


def _text_declarations_match(
    record: dict,
    key: str,
    expected: object,
    *,
    required: bool = False,
) -> bool:
    """Reconcile top-level and finite-convention aliases with a contract."""

    convention = record.get("finite_torus_convention")
    sources = [record]
    if isinstance(convention, dict):
        sources.append(convention)
    values = [
        _normalized_text(source[key])
        for source in sources
        if key in source
    ]
    normalized_expected = _normalized_text(expected)
    if normalized_expected is None or any(value is None for value in values):
        return False
    if not values:
        return not required
    return all(value == normalized_expected for value in values)


def _numeric_declaration_matches(
    record: dict,
    key: str,
    expected: object,
    *,
    required: bool = False,
) -> bool:
    """Check an optional finite numeric record field against the contract."""

    if key not in record:
        return not required
    actual = _finite_float(record[key])
    expected_value = _finite_float(expected)
    return actual is not None and actual == expected_value


def _smearing_declaration_matches(
    record: dict,
    expected: object,
    *,
    required: bool = False,
) -> bool:
    """Reconcile requested and reported SCF smearing with the contract."""

    expected_value = _finite_float(expected)
    if expected_value is None:
        return False
    options = record.get("scf_options")
    if options is None:
        return not required
    if not isinstance(options, dict) or "smearing_temperature" not in options:
        return False
    if _finite_float(options["smearing_temperature"]) != expected_value:
        return False

    diagnostics = record.get("convergence_diagnostics")
    if diagnostics is None:
        return not required
    return (
        isinstance(diagnostics, dict)
        and "smearing_temperature" in diagnostics
        and _finite_float(diagnostics["smearing_temperature"])
        == expected_value
    )


def _comparison_input_sha256(value: object) -> str | None:
    """Hash one JSON-native comparison input with deterministic encoding."""

    def is_json_native(item: object) -> bool:
        if item is None or isinstance(item, (str, bool, int)):
            return True
        if isinstance(item, float):
            return isfinite(item)
        if isinstance(item, list):
            return all(is_json_native(child) for child in item)
        if isinstance(item, dict):
            return all(
                isinstance(key, str) and is_json_native(child)
                for key, child in item.items()
            )
        return False

    if not isinstance(value, dict) or not value or not is_json_native(value):
        return None
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        return None
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _finite_vector(value: object, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(_finite_float(item) is not None for item in value)
    )


def _comparison_input_matches_record(
    record: dict,
    contract: dict,
) -> bool:
    """Validate and reconcile the complete v2 canonical-input payload."""

    value = record.get("comparison_input")
    required = {
        "schema",
        "system_name",
        "system",
        "electronic_method",
        "functional",
        "orbital_basis",
        "auxiliary_basis",
        "finite_torus_mesh",
        "boundary_model",
        "coulomb_kernel",
        "exchange_q0",
        "two_electron_operator",
        "gdf_method",
        "rsgdf_ke_cutoff",
        "ao_linear_dependence_threshold",
        "auxiliary_metric_linear_dependence_threshold",
        "auxiliary_phase_convention",
        "smearing_temperature",
    }
    if not isinstance(value, dict) or set(value) != required:
        return False
    if value.get("schema") != REAL_GAMMA_CONTROL_INPUT_VERSION:
        return False

    physical = value.get("system")
    if not isinstance(physical, dict) or set(physical) != {
        "dim",
        "charge",
        "multiplicity",
        "lattice_bohr",
        "atoms",
    }:
        return False
    if (
        type(physical.get("dim")) is not int
        or physical.get("dim") != contract.get("dim")
        or type(physical.get("charge")) is not int
        or type(physical.get("multiplicity")) is not int
        or physical.get("multiplicity", 0) < 1
    ):
        return False
    lattice = physical.get("lattice_bohr")
    if not (
        isinstance(lattice, list)
        and len(lattice) == 3
        and all(_finite_vector(row, 3) for row in lattice)
    ):
        return False
    lattice_float = [
        [float(component) for component in row]
        for row in lattice
    ]
    determinant = (
        lattice_float[0][0]
        * (
            lattice_float[1][1] * lattice_float[2][2]
            - lattice_float[1][2] * lattice_float[2][1]
        )
        - lattice_float[0][1]
        * (
            lattice_float[1][0] * lattice_float[2][2]
            - lattice_float[1][2] * lattice_float[2][0]
        )
        + lattice_float[0][2]
        * (
            lattice_float[1][0] * lattice_float[2][1]
            - lattice_float[1][1] * lattice_float[2][0]
        )
    )
    if not isfinite(determinant) or abs(determinant) <= 1.0e-12:
        return False
    atoms = physical.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        return False
    for atom in atoms:
        if (
            not isinstance(atom, dict)
            or set(atom) != {"atomic_number", "xyz_bohr"}
            or type(atom.get("atomic_number")) is not int
            or atom.get("atomic_number", 0) < 1
            or not _finite_vector(atom.get("xyz_bohr"), 3)
        ):
            return False

    if value.get("system_name") != record.get("system"):
        return False
    if _normalized_text(value.get("electronic_method")) != _normalized_text(
        record.get("method")
    ):
        return False
    functional = value.get("functional")
    if functional is not None and _normalized_text(functional) is None:
        return False
    if _normalized_text(functional) != _normalized_text(record.get("functional")):
        return False
    if _normalized_text(value.get("orbital_basis")) != _normalized_text(
        record.get("basis")
    ):
        return False
    if _normalized_text(value.get("auxiliary_basis")) != _normalized_text(
        contract.get("aux_basis")
    ):
        return False
    if _mesh_shape(value, "finite_torus_mesh") != (
        _mesh_shape(record, "mesh") or _mesh_shape(record, "nrep")
    ):
        return False

    for input_key, contract_key in (
        ("boundary_model", "boundary_model"),
        ("coulomb_kernel", "coulomb_kernel"),
        ("exchange_q0", "exchange_q0"),
        ("two_electron_operator", "two_electron_operator"),
        ("gdf_method", "gdf_method"),
        ("auxiliary_phase_convention", "auxiliary_phase_convention"),
    ):
        if _normalized_text(value.get(input_key)) != _normalized_text(
            contract.get(contract_key)
        ):
            return False
    for key in (
        "rsgdf_ke_cutoff",
        "ao_linear_dependence_threshold",
        "auxiliary_metric_linear_dependence_threshold",
        "smearing_temperature",
    ):
        if (
            _finite_float(value.get(key)) is None
            or _finite_float(value.get(key)) != _finite_float(contract.get(key))
        ):
            return False
    return True


def _record_matches_contract(
    record: dict,
    contract: dict,
    *,
    require_payload: bool = False,
) -> bool:
    """Reject copied contracts that contradict their result payload."""

    if _exchange_q0(record) != _normalized_text(contract.get("exchange_q0")):
        return False
    for key in (
        "boundary_model",
        "coulomb_kernel",
        "aux_basis",
        "gdf_method",
        "auxiliary_phase_convention",
        "two_electron_operator",
    ):
        if not _text_declarations_match(
            record,
            key,
            contract.get(key),
            required=require_payload and key in {
                "boundary_model",
                "coulomb_kernel",
                "aux_basis",
                "gdf_method",
                "auxiliary_phase_convention",
                "two_electron_operator",
            },
        ):
            return False
    for key in (
        "rsgdf_ke_cutoff",
        "ao_linear_dependence_threshold",
        "auxiliary_metric_linear_dependence_threshold",
    ):
        if not _numeric_declaration_matches(
            record,
            key,
            contract.get(key),
            required=require_payload,
        ):
            return False
    if not _smearing_declaration_matches(
        record,
        contract.get("smearing_temperature"),
        required=require_payload,
    ):
        return False

    if require_payload and "dim" not in record:
        return False
    if "dim" in record and record.get("dim") != contract.get("dim"):
        return False
    convention = record.get("finite_torus_convention")
    if isinstance(convention, dict):
        if (
            "periodic_dimension" in convention
            and convention.get("periodic_dimension") != contract.get("dim")
        ):
            return False
        nested_mesh = _mesh_shape(convention, "character_mesh_shape")
        if "character_mesh_shape" in convention and nested_mesh is None:
            return False
        declared_mesh = _mesh_shape(record, "mesh") or _mesh_shape(record, "nrep")
        if nested_mesh is not None and declared_mesh != nested_mesh:
            return False
    return True


def _is_sha256(value: object) -> bool:
    """Require the canonical lowercase, prefixed SHA256 representation."""

    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _is_full_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_identity_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and value == value.strip()
        and value.casefold() not in {"", "unknown", "not-recorded"}
    )


def _valid_preflight_attestation(value: object) -> bool:
    """Validate one exact, successful v2 host/core attestation."""

    return (
        isinstance(value, dict)
        and set(value) == _PREFLIGHT_ATTESTATION_FIELDS
        and value.get("probe_version") == EXPECTED_PROBE_VERSION
        and _is_sha256(value.get("probe_script_id"))
        and value.get("probe_passed") is True
        and _is_identity_text(value.get("host"))
        and _is_identity_text(value.get("vibeqc_version"))
        and _is_sha256(value.get("core_build_id"))
        and _is_full_commit(value.get("source_commit"))
        and value.get("source_clean") is True
    )


def _valid_native_library_versions(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == _NATIVE_LIBRARY_KEYS
        and all(_is_identity_text(version) for version in value.values())
    )


def _valid_producer_payload(
    value: object,
    *,
    expected_stream: str | None,
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "version",
        "stream",
        "files",
    }:
        return False
    stream = value.get("stream")
    if (
        value.get("version") != EXPECTED_PRODUCER_PAYLOAD_VERSION
        or not isinstance(stream, str)
        or stream not in _PRODUCER_STREAM_FILES
        or (expected_stream is not None and stream != expected_stream)
    ):
        return False
    files = value.get("files")
    return (
        isinstance(files, dict)
        and set(files) == _PRODUCER_STREAM_FILES[stream]
        and all(_is_sha256(digest) for digest in files.values())
    )


def _valid_loaded_core_check(value: object) -> bool:
    """Validate the process-local shifted-mesh mirror discriminator."""

    if not isinstance(value, dict) or set(value) != _LOADED_CORE_CHECK_FIELDS:
        return False
    if (
        value.get("version") != EXPECTED_LOADED_CORE_CHECK_VERSION
        or value.get("passed") is not True
        or value.get("fixture") != "lih-sto3g-shifted-mesh/v1"
        or type(value.get("n_ao")) is not int
        or value.get("n_ao") != 6
        or type(value.get("max_l")) is not int
        or value.get("max_l") != 1
        or type(value.get("n_cells")) is not int
        or value.get("n_cells") != 3
        or type(value.get("n_frequencies")) is not int
        or value.get("n_frequencies") != 3
    ):
        return False
    atol = _finite_float(value.get("atol"))
    rtol = _finite_float(value.get("rtol"))
    max_abs = _finite_float(value.get("max_abs_cxx_python"))
    max_scaled = _finite_float(value.get("max_scaled_error"))
    asymmetry = _finite_float(value.get("reference_transpose_asymmetry"))
    return (
        atol == 1.0e-12
        and rtol == 1.0e-9
        and max_abs is not None
        and max_abs >= 0.0
        and max_scaled is not None
        and 0.0 <= max_scaled <= 1.0
        and asymmetry is not None
        and asymmetry > 1.0e-3
    )


def _audit_alias_matches(value: object, expected: object) -> bool:
    """Match a JSON alias or the textual rendering used by ``RUN.meta``."""

    if isinstance(expected, bool):
        if isinstance(value, bool):
            return value is expected
        return (
            isinstance(value, str)
            and value.strip().casefold() == ("true" if expected else "false")
        )
    return isinstance(value, str) and value == expected


def _producer_identity(
    record: dict,
    *,
    expected_stream: str | None = None,
) -> tuple[str, str, str, str, str, str, str, str, str] | None:
    """Return the process-bound identity for a character/real-Gamma control.

    The preflight and the complete producer payload remain auditable inputs to
    the composite attestation, but neither identifies the core that actually
    produced the result.  Equality therefore uses the independently validated
    current-core attestation, linked-library set, and check versions.
    """

    provenance = record.get("provenance")
    required_aliases = {
        "host",
        "vibeqc_version",
        "vibeqc_commit",
        "source_clean",
        "core_build_id",
        "probe_version",
        "probe_script_id",
        "probe_passed",
        "probe_attestation_id",
        "producer_payload_id",
    }
    if (
        not isinstance(provenance, dict)
        or not required_aliases.issubset(provenance)
    ):
        return None
    attestation = provenance.get("probe_attestation")
    if (
        not isinstance(attestation, dict)
        or set(attestation) != _PRODUCER_ATTESTATION_FIELDS
        or attestation.get("attestation_version")
        != EXPECTED_PRODUCER_ATTESTATION_VERSION
        or not _is_sha256(attestation.get("attestor_script_id"))
        or attestation.get("probe_passed") is not True
        or not _is_identity_text(attestation.get("host"))
        or not _is_identity_text(attestation.get("vibeqc_version"))
        or not _is_sha256(attestation.get("core_build_id"))
        or not _is_full_commit(attestation.get("source_commit"))
        or attestation.get("source_clean") is not True
        or type(attestation.get("core_changed_since_preflight")) is not bool
        or attestation.get("result_identity_stable") is not True
    ):
        return None

    libraries = attestation.get("native_library_versions")
    payload = attestation.get("producer_payload")
    preflight = attestation.get("preflight_attestation")
    current = attestation.get("current_core_attestation")
    loaded_check = attestation.get("loaded_core_check")
    if (
        not _valid_native_library_versions(libraries)
        or not _valid_producer_payload(
            payload,
            expected_stream=expected_stream,
        )
        or not _valid_preflight_attestation(preflight)
        or not _valid_preflight_attestation(current)
        or not _valid_loaded_core_check(loaded_check)
    ):
        return None
    assert isinstance(libraries, dict)
    assert isinstance(payload, dict)
    assert isinstance(preflight, dict)
    assert isinstance(current, dict)
    assert isinstance(loaded_check, dict)

    payload_id = _comparison_input_sha256(payload)
    preflight_id = _comparison_input_sha256(preflight)
    current_id = _comparison_input_sha256(current)
    attestation_id = _comparison_input_sha256(attestation)
    if (
        payload_id is None
        or payload_id != attestation.get("producer_payload_id")
        or preflight_id is None
        or preflight_id != attestation.get("preflight_attestation_id")
        or current_id is None
        or current_id != attestation.get("current_core_attestation_id")
        or attestation_id is None
        or payload["files"]["probe_host"]
        != attestation.get("attestor_script_id")
    ):
        return None

    # A deployment update may replace only the native core between preflight
    # and producer launch. Every other source/package/probe identity must stay
    # exact, and the explicit change flag must describe that sole difference.
    for key in _PREFLIGHT_ATTESTATION_FIELDS - {"core_build_id"}:
        if preflight[key] != current[key]:
            return None
    core_changed = preflight["core_build_id"] != current["core_build_id"]
    if attestation["core_changed_since_preflight"] is not core_changed:
        return None

    current_aliases = {
        "host": attestation["host"],
        "vibeqc_version": attestation["vibeqc_version"],
        "core_build_id": attestation["core_build_id"],
        "source_commit": attestation["source_commit"],
        "source_clean": attestation["source_clean"],
        "probe_script_id": attestation["attestor_script_id"],
    }
    if any(current.get(key) != value for key, value in current_aliases.items()):
        return None

    aliases = {
        "host": attestation["host"],
        "vibeqc_version": attestation["vibeqc_version"],
        "vibeqc_commit": attestation["source_commit"],
        "source_clean": attestation["source_clean"],
        "core_build_id": attestation["core_build_id"],
        "probe_version": attestation["attestation_version"],
        "probe_script_id": attestation["attestor_script_id"],
        "probe_passed": attestation["probe_passed"],
        "probe_attestation_id": attestation_id,
        "producer_payload_id": payload_id,
    }
    if any(provenance.get(key) != value for key, value in aliases.items()):
        return None

    # Top-level result fields and curated RUN.meta values are optional audit
    # views. They cannot supply producer evidence and must not contradict it.
    top_level = {key: record[key] for key in aliases if key in record}
    sources = [top_level] if top_level else []
    if "_run_meta" in record:
        run_meta = record["_run_meta"]
        if not isinstance(run_meta, dict) or run_meta.get("_invalid"):
            return None
        sources.append(run_meta)
    for source in sources:
        for key, expected in aliases.items():
            if key in source and not _audit_alias_matches(source[key], expected):
                return None

    libraries_id = _comparison_input_sha256(libraries)
    if libraries_id is None:
        return None
    return (
        attestation["source_commit"],
        attestation["core_build_id"],
        attestation["host"],
        attestation["vibeqc_version"],
        attestation["attestation_version"],
        attestation["attestor_script_id"],
        loaded_check["version"],
        libraries_id,
        current_id,
    )


def _comparison_contract(record: dict) -> dict | None:
    """Return the complete route-independent numerical contract, if declared."""

    contract = record.get("comparison_contract")
    if not isinstance(contract, dict):
        return None
    required = {
        "version",
        "input_sha256",
        "dim",
        "boundary_model",
        "coulomb_kernel",
        "exchange_q0",
        "two_electron_operator",
        "aux_basis",
        "gdf_method",
        "rsgdf_ke_cutoff",
        "ao_linear_dependence_threshold",
        "auxiliary_metric_linear_dependence_threshold",
        "auxiliary_phase_convention",
        "smearing_temperature",
    }
    if set(contract) != required:
        return None
    fingerprint = contract.get("input_sha256")
    comparison_input = record.get("comparison_input")
    computed_fingerprint = _comparison_input_sha256(
        comparison_input
    )
    if not isinstance(fingerprint, str) or fingerprint != computed_fingerprint:
        return None
    if (
        contract.get("version") != REAL_GAMMA_CONTROL_CONTRACT_VERSION
        or type(contract.get("dim")) is not int
        or contract.get("dim") != 3
        or contract.get("boundary_model") != "3d-periodic"
        or contract.get("coulomb_kernel") != "3d-periodic-g0"
        or contract.get("exchange_q0") != EXPECTED_EXCHANGE_Q0
    ):
        return None
    for key in (
        "two_electron_operator",
        "aux_basis",
        "auxiliary_phase_convention",
    ):
        if _normalized_text(contract.get(key)) is None:
            return None
    if contract.get("gdf_method") != "rsgdf":
        return None
    for key in (
        "rsgdf_ke_cutoff",
        "ao_linear_dependence_threshold",
        "auxiliary_metric_linear_dependence_threshold",
    ):
        value = contract.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            or value <= 0.0
        ):
            return None
    smearing = contract.get("smearing_temperature")
    if (
        not isinstance(smearing, (int, float))
        or isinstance(smearing, bool)
        or not isfinite(smearing)
        or float(smearing) != 0.0
    ):
        return None
    if not _comparison_input_matches_record(record, contract):
        return None
    return contract


def _metadata_matches_real_gamma_control(
    b_route_name: str,
    b_record: dict,
    control_route_name: str,
    control_record: dict,
) -> bool:
    """Require like-for-like metadata for one real-Gamma control delta.

    This gate establishes a neutral fitted-torus character-vs-real-Gamma
    representation control.  It does not establish a comparison with the
    Gamma-CCM union-and-weight approach.
    """

    b_route = ROUTES.get(b_route_name)
    control_method = REAL_GAMMA_CONTROL_METHOD_FOR_ROUTE.get(
        control_route_name
    )
    if (
        b_route is None
        or control_method != b_route.method
        or b_record.get("route") != b_route_name
        or control_record.get("route") != control_route_name
        or not _real_gamma_control_identity_matches(control_record)
    ):
        return False
    b_producer = _producer_identity(
        b_record,
        expected_stream="aiccm2026dev-b",
    )
    control_producer = _producer_identity(
        control_record,
        expected_stream="aiccm2026dev-a",
    )
    if b_producer is None or b_producer != control_producer:
        return False
    b_contract = _comparison_contract(b_record)
    control_contract = _comparison_contract(control_record)
    if (
        b_contract is None
        or b_contract != control_contract
        or _normalized_text(b_contract.get("two_electron_operator"))
        != _normalized_text(b_route.backend)
        or not _record_matches_contract(
            b_record,
            b_contract,
            require_payload=True,
        )
        or not _record_matches_contract(
            control_record,
            control_contract,
            require_payload=True,
        )
    ):
        return False

    if not _b_route_identity_matches(b_route_name, b_record):
        return False
    if not _b_construction_identity_matches(b_record):
        return False
    if _b_exact_exchange_assembly_failure(b_record) is not None:
        return False
    if not _b_finite_torus_lattice_matches_input(b_record):
        return False
    b_method = b_record.get("method")
    if _normalized_text(b_method) != _normalized_text(b_route.method):
        return False
    declared_control_method = control_record.get("method")
    if (
        declared_control_method is not None
        and _normalized_text(declared_control_method)
        != _normalized_text(control_method)
    ):
        return False

    b_functional = _normalized_text(
        b_record.get("functional", b_route.functional)
    )
    if b_functional != _normalized_text(b_route.functional):
        return False
    if _normalized_text(control_record.get("functional")) != b_functional:
        return False

    b_basis = _normalized_text(b_record.get("basis"))
    control_basis = _normalized_text(control_record.get("basis"))
    if b_basis is None or b_basis != control_basis:
        return False

    b_mesh = _mesh_shape(b_record, "mesh")
    control_nrep = _mesh_shape(control_record, "nrep")
    if b_mesh is None or b_mesh != control_nrep:
        return False

    b_exchange_q0 = _exchange_q0(b_record)
    control_exchange_q0 = _exchange_q0(control_record)
    return (
        b_exchange_q0 == EXPECTED_EXCHANGE_Q0
        and control_exchange_q0 == EXPECTED_EXCHANGE_Q0
    )


def _matched_real_gamma_control_record(
    system: str,
    b_route_name: str,
    b_record: dict,
    control_records: dict[
        tuple[str, str, str | None] | tuple[str, str],
        dict,
    ],
) -> tuple[str | None, dict | None]:
    """Return one matched real-Gamma control, or no control on a mismatch."""

    control_route_name = REAL_GAMMA_CONTROL_ROUTE_FOR_B.get(b_route_name)
    if control_route_name is None:
        return None, None
    functional = _normalized_text(
        b_record.get("functional", ROUTES[b_route_name].functional)
    )
    control_record = control_records.get(
        (system, control_route_name, functional)
    )
    if control_record is None:
        # Keep direct unit-level callers compatible with the former pair key;
        # ``load()`` itself always emits functional-aware triple keys.
        control_record = control_records.get((system, control_route_name))
    if (
        not _is_converged_success(b_record)
        or not _is_converged_success(
            control_record,
            allow_missing_status=True,
        )
        or _is_retracted(control_record)
        or not _metadata_matches_real_gamma_control(
            b_route_name,
            b_record,
            control_route_name,
            control_record,
        )
    ):
        return None, None
    return control_route_name, control_record


def _crystal_value(refs: dict, system: str, route_name: str) -> float | None:
    value = refs.get(system)
    if isinstance(value, (int, float)):
        return _finite_float(value) if route_name.startswith("rks-pbe-") else None
    if not isinstance(value, dict):
        return None
    route = ROUTES[route_name]
    key = "rhf" if route.method == "RHF" else route.functional
    value = value.get(key)
    return _finite_float(value)


def _fmt(value: float | None, digits: int = 9) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("b_results")
    parser.add_argument(
        "--real-gamma-control-results",
        "--a-results",
        dest="real_gamma_control_results",
        help=(
            "real-Gamma representation-control records; --a-results is a "
            "legacy spelling and does not denote Gamma-CCM evidence"
        ),
    )
    parser.add_argument(
        "--crystal-refs",
        help="JSON {system: {rhf: E/atom, pbe: E/atom, pbe0: E/atom}}",
    )
    parser.add_argument("--csv")
    args = parser.parse_args()

    try:
        b_records = load(args.b_results)
        control_records = (
            load(args.real_gamma_control_results)
            if args.real_gamma_control_results
            else {}
        )
    except ValueError as exc:
        parser.error(str(exc))
    invalid_statuses = [
        (system, route, record.get("status"))
        for (system, route, _), record in _sorted_records(b_records)
        if (
            not isinstance(record.get("status"), str)
            or record.get("status") not in KNOWN_B_STATUSES
        )
    ]
    if invalid_statuses:
        details = ", ".join(
            f"{system}/{route} (status={status!r})"
            for system, route, status in invalid_statuses
        )
        parser.error(
            "refusing chi-CCM-B records with an unknown or missing status: "
            f"{details}"
        )
    unregistered_routes = [
        (system, route)
        for (system, route, _), _record in _sorted_records(b_records)
        if route not in ROUTES
    ]
    if unregistered_routes:
        details = ", ".join(
            f"{system}/{route}" for system, route in unregistered_routes
        )
        parser.error(
            "refusing chi-CCM-B records with an unregistered route: "
            f"{details}"
        )
    retracted = [
        (system, route)
        for (system, route, _), record in _sorted_records(b_records)
        if _is_retracted(record)
    ]
    if retracted:
        details = ", ".join(
            f"{system}/{route}" for system, route in retracted
        )
        parser.error(
            "refusing to compare retracted chi-CCM-B results: "
            f"{details}"
        )
    unreportable = [
        (system, route, record.get("dim"))
        for (system, route, _), record in _sorted_records(b_records)
        if _has_unreportable_b_dimension(record)
    ]
    if unreportable:
        details = ", ".join(
            f"{system}/{route} (dim={dim})"
            for system, route, dim in unreportable
        )
        parser.error(
            "refusing to compare lower-dimensional chi-CCM-B absolute "
            f"energies: {details}"
        )
    invalid_identity = [
        (system, route)
        for (system, route, _), record in _sorted_records(b_records)
        if record.get("status") in ("ok", "not_converged")
        and (
            not _b_route_identity_matches(route, record)
            or not _b_construction_identity_matches(record)
        )
    ]
    if invalid_identity:
        details = ", ".join(
            f"{system}/{route}" for system, route in invalid_identity
        )
        parser.error(
            "refusing chi-CCM-B records whose exact construction identity, "
            "exchange-q=0 applicability, method, functional, or backend "
            "contradicts the route selector: "
            f"{details}"
        )
    revision_bound = [
        (
            system,
            route,
            d102_absolute_energy_revision_failure(
                record.get("ewald_shifted_pair_support"),
                record.get("provenance"),
            ),
        )
        for (system, route, _), record in _sorted_records(b_records)
        if record.get("status") in (None, "ok", "not_converged")
        and d102_absolute_energy_revision_failure(
            record.get("ewald_shifted_pair_support"),
            record.get("provenance"),
        )
        is not None
    ]
    if revision_bound:
        details = ", ".join(
            f"{system}/{route} ({failure})"
            for system, route, failure in revision_bound
        )
        parser.error(
            "refusing to compare revision-bound chi-CCM-B absolute "
            f"energies: {details}"
        )
    invalid_exchange_assembly = [
        (system, route, _b_exact_exchange_assembly_failure(record))
        for (system, route, _), record in _sorted_records(b_records)
        if record.get("status") in ("ok", "not_converged")
        and _b_exact_exchange_assembly_failure(record) is not None
    ]
    if invalid_exchange_assembly:
        details = ", ".join(
            f"{system}/{route} ({failure})"
            for system, route, failure in invalid_exchange_assembly
        )
        parser.error(
            "refusing to compare chi-CCM-B rows without route-consistent "
            f"exact-exchange assembly provenance: {details}"
        )
    unqualified_support = [
        (system, route, _b_two_electron_support_failure(record))
        for (system, route, _), record in _sorted_records(b_records)
        if record.get("status") in (None, "ok", "not_converged")
        and _b_two_electron_support_failure(record) is not None
    ]
    if unqualified_support:
        details = ", ".join(
            f"{system}/{route} ({failure})"
            for system, route, failure in unqualified_support
        )
        parser.error(
            "refusing to compare chi-CCM-B rows without qualified "
            f"two-electron numerical support: {details}"
        )
    unqualified_overlap_fold = [
        (system, route, _b_overlap_fold_support_failure(record))
        for (system, route, _), record in _sorted_records(b_records)
        if record.get("status") in (None, "ok", "not_converged")
        and _b_overlap_fold_support_failure(record) is not None
    ]
    if unqualified_overlap_fold:
        details = ", ".join(
            f"{system}/{route} ({failure})"
            for system, route, failure in unqualified_overlap_fold
        )
        parser.error(
            "refusing to compare chi-CCM-B rows without qualified "
            f"overlap-fold numerical support: {details}"
        )
    revision_bound_controls = [
        (
            system,
            route,
            d102_absolute_energy_revision_failure(
                record.get("ewald_shifted_pair_support"),
                record.get("provenance"),
            ),
        )
        for (system, route, _), record in _sorted_records(control_records)
        if _is_converged_success(record, allow_missing_status=True)
        and d102_absolute_energy_revision_failure(
            record.get("ewald_shifted_pair_support"),
            record.get("provenance"),
        )
        is not None
    ]
    if revision_bound_controls:
        details = ", ".join(
            f"{system}/{route} ({failure})"
            for system, route, failure in revision_bound_controls
        )
        parser.error(
            "refusing revision-bound real-Gamma representation-control "
            f"absolute energies: {details}"
        )
    crystal = (
        json.loads(Path(args.crystal_refs).read_text(encoding="utf-8"))
        if args.crystal_refs
        else {}
    )
    rows = []
    for (system, route_name, _), b_record in _sorted_records(b_records):
        b_energy = _energy_per_atom(b_record)
        control_route, control_record = _matched_real_gamma_control_record(
            system,
            route_name,
            b_record,
            control_records,
        )
        control_energy = _energy_per_atom(
            control_record,
            allow_missing_status=True,
        )
        crystal_energy = _crystal_value(crystal, system, route_name)
        control_status = (
            "defined"
            if control_route is not None
            and b_energy is not None
            and control_energy is not None
            else "not-defined"
        )
        control_defined = control_status == "defined"
        rows.append(
            {
                "system": system,
                "route": route_name,
                "status": b_record.get("status", "unknown"),
                "gamma_ccm_chi_ccm_approach_comparison_status": (
                    "not-defined"
                ),
                "chi_ccm_ha_per_atom": b_energy,
                "real_gamma_control_status": control_status,
                "real_gamma_control_route": (
                    control_route if control_defined else "not-defined"
                ),
                "real_gamma_control_ha_per_atom": (
                    control_energy if control_defined else "not-defined"
                ),
                "crystal_ha_per_atom": crystal_energy,
                "character_minus_real_gamma_control_mha_per_atom": (
                    (b_energy - control_energy) * HA_TO_MHA
                    if control_defined
                    else "not-defined"
                ),
                "chi_ccm_minus_crystal_mha_per_atom": (
                    None
                    if b_energy is None or crystal_energy is None
                    else (b_energy - crystal_energy) * HA_TO_MHA
                ),
                "wall_time_s": b_record.get("wall_time_s"),
            }
        )

    print(
        "| system | χ-CCM route | χ record status | "
        "Γ-CCM/χ-CCM approach comparison status | χ-CCM / Ha atom-1 | "
        "real-Γ control status | real-Γ control route | "
        "real-Γ control / Ha atom-1 | CRYSTAL / Ha atom-1 | "
        "χ character minus real-Γ control / mHa atom-1 | "
        "χ-CRYSTAL / mHa atom-1 |"
    )
    print("|---|---|---|---|---:|---|---|---:|---:|---:|---:|")
    for row in rows:
        control_defined = row["real_gamma_control_status"] == "defined"
        print(
            "| {system} | {route} | {status} | {approach_status} | {b} | "
            "{control_status} | {control_route} | {control_energy} | {x} | "
            "{control_delta} | {dx} |".format(
                system=row["system"],
                route=row["route"],
                status=row["status"],
                approach_status=row[
                    "gamma_ccm_chi_ccm_approach_comparison_status"
                ],
                b=_fmt(row["chi_ccm_ha_per_atom"]),
                control_status=row["real_gamma_control_status"],
                control_route=row["real_gamma_control_route"],
                control_energy=(
                    _fmt(row["real_gamma_control_ha_per_atom"])
                    if control_defined
                    else row["real_gamma_control_ha_per_atom"]
                ),
                x=_fmt(row["crystal_ha_per_atom"]),
                control_delta=(
                    _fmt(
                        row[
                            "character_minus_real_gamma_control_mha_per_atom"
                        ],
                        6,
                    )
                    if control_defined
                    else row[
                        "character_minus_real_gamma_control_mha_per_atom"
                    ]
                ),
                dx=_fmt(row["chi_ccm_minus_crystal_mha_per_atom"], 6),
            )
        )
    if args.csv:
        import csv

        with Path(args.csv).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
