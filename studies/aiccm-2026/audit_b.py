#!/usr/bin/env python3
"""Audit chi-CCM / aiccm2026dev-b JSON result records.

The paper and fleet queues need a quick pre-table gate: every successful B
record must carry the declared finite-torus convention, an exact convergence
attestation, and a finite energy. Failed or non-converged records should be
visible before comparing energies. Both raw ``run_case_b.py`` filenames and
curated nested ``result.json`` records are discovered recursively.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from b_routes import (
    ROUTES,
    d102_absolute_energy_revision_failure,
    exact_exchange_assembly_failure,
    overlap_fold_support_reportability_failure,
    two_electron_support_reportability_failure,
)


EXPECTED_COULOMB_KERNEL = "3d-periodic-g0"
EXPECTED_EXCHANGE_Q0 = "bvk-ewald"
EXPECTED_LATTICE_VECTOR_CONVENTION = "columns"
EXPECTED_CCM_APPROACH = "chi-ccm"
EXPECTED_CCM_CONSTRUCTION = "finite-translation-group-character"
EXPECTED_EVALUATION_REPRESENTATION = "gamma-centred-character-mesh"
KNOWN_STATUSES = {"ok", "not_converged", "unsupported", "error"}


def _is_retracted(record: dict[str, Any]) -> bool:
    """Conservatively treat every truthy retraction marker as withdrawn."""

    return bool(record.get("retracted", False))


def _candidate_json_paths(root: Path) -> list[Path]:
    """Return raw B outputs and records stored by ``curate.py``."""

    paths = set(root.rglob("*__b-*.json"))
    paths.update(root.rglob("result.json"))
    return sorted(paths)


def _path_strongly_identifies_b(path: Path) -> bool:
    """Return whether a path is itself an explicit B-stream declaration."""

    return "__b-" in path.name or (
        path.name == "result.json" and "aiccm2026dev-b" in path.parts
    )


def _record_matches_b(path: Path, record: dict[str, Any]) -> bool:
    """Recognize explicit B records and legacy records under the B subtree."""

    selector = record.get("selector") or record.get("stream")
    if isinstance(selector, str):
        return selector == "aiccm2026dev-b"
    return (
        "__b-" in path.name
        or (path.name == "result.json" and "aiccm2026dev-b" in path.parts)
    )


def _json_records(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in _candidate_json_paths(root):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            if _path_strongly_identifies_b(path):
                raise ValueError(
                    f"{path}: malformed chi-CCM-B JSON: {exc.msg}"
                ) from exc
            continue
        if not isinstance(record, dict):
            if _path_strongly_identifies_b(path):
                raise ValueError(
                    f"{path}: chi-CCM-B JSON record is not an object"
                )
            continue
        if _record_matches_b(path, record):
            records.append((path, record))
    return records


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, (list, tuple)):
        return "x".join(str(item) for item in value)
    return str(value)


def _mesh_shape(value: object) -> tuple[int, int, int] | None:
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


def _finite_matrix(value: object) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            return None
        if any(not _is_finite_number(item) for item in row):
            return None
        matrix.append([float(item) for item in row])
    return matrix


def _bvk_lattice_matches_record(record: dict[str, Any]) -> bool:
    """Require the recorded BvK matrix to equal A @ diag(mesh)."""

    convention = record.get("finite_torus_convention")
    if not isinstance(convention, dict):
        return False
    mesh = _mesh_shape(record.get("mesh"))
    character_mesh = _mesh_shape(convention.get("character_mesh_shape"))
    repetitions = _mesh_shape(
        convention.get("bvk_madelung_supercell_repetitions")
    )
    if (
        mesh is None
        or character_mesh != mesh
        or repetitions != mesh
        or type(convention.get("periodic_dimension")) is not int
        or convention.get("periodic_dimension") != 3
    ):
        return False
    lattice = _finite_matrix(record.get("primitive_lattice_bohr"))
    bvk_lattice = _finite_matrix(
        convention.get("bvk_madelung_supercell_lattice_bohr")
    )
    if lattice is None or bvk_lattice is None:
        return False
    return all(
        abs(bvk_lattice[i][j] - lattice[i][j] * mesh[j]) <= 1.0e-12
        for i in range(3)
        for j in range(3)
    )


def _route_identity_matches(record: dict[str, Any]) -> bool:
    """Require emitted method/backend/functional fields to match the route."""

    route_name = record.get("route")
    if not isinstance(route_name, str):
        return False
    route = ROUTES.get(route_name)
    return route is not None and all(
        key in record and record[key] == expected
        for key, expected in {
            "method": route.method,
            "backend": route.backend,
            "functional": route.functional,
        }.items()
    )


def _expected_exchange_q0_applicability(
    record: dict[str, Any],
) -> str | None:
    """Return the benchmark route's declared full-range seam applicability."""

    route_name = record.get("route")
    route = ROUTES.get(route_name) if isinstance(route_name, str) else None
    return None if route is None else route.exchange_q0_applicability


def _row(path: Path, record: dict[str, Any], root: Path) -> dict[str, Any]:
    raw_convention = record.get("finite_torus_convention")
    convention = raw_convention if isinstance(raw_convention, dict) else {}
    raw_properties = record.get("properties")
    properties = raw_properties if isinstance(raw_properties, dict) else {}
    raw_diagnostics = record.get("convergence_diagnostics")
    diagnostics = raw_diagnostics if isinstance(raw_diagnostics, dict) else {}
    raw_support = record.get("two_electron_support")
    support = raw_support if isinstance(raw_support, dict) else {}
    status = record.get("status", "unknown")
    support_failure = (
        two_electron_support_reportability_failure(
            raw_support,
            record.get("route"),
            record.get("direct_lattice_cutoffs"),
        )
        if status in ("ok", "not_converged")
        else None
    )
    overlap_fold_failure = (
        overlap_fold_support_reportability_failure(
            record.get("overlap_fold_support"),
            record.get("route"),
            record.get("direct_lattice_cutoffs"),
            record.get("mesh"),
        )
        if status in ("ok", "not_converged")
        else None
    )
    revision_failure = (
        d102_absolute_energy_revision_failure(
            record.get("ewald_shifted_pair_support"),
            record.get("provenance"),
        )
        if status in ("ok", "not_converged")
        else None
    )
    exchange_assembly_failure = (
        exact_exchange_assembly_failure(
            record.get("exact_exchange_assembly"),
            record.get("route"),
        )
        if status in ("ok", "not_converged")
        else None
    )
    row = {
        "path": str(path.relative_to(root)),
        "system": record.get("system"),
        "route": record.get("route"),
        "status": status,
        "retracted": _is_retracted(record),
        "converged": record.get("converged"),
        "dim": record.get("dim"),
        "mesh": record.get("mesh"),
        "basis": record.get("basis"),
        "method": record.get("method"),
        "backend": record.get("backend"),
        "functional": record.get("functional"),
        "route_identity_matches": _route_identity_matches(record),
        "aux_basis": record.get("aux_basis"),
        "local_mode": record.get("local_mode"),
        "energy_per_cell_ha": record.get("energy_per_cell_ha"),
        "energy_per_atom_ha": record.get("energy_per_atom_ha"),
        "_raw_energy_per_cell_ha": record.get("energy_per_cell_ha"),
        "_raw_energy_per_atom_ha": record.get("energy_per_atom_ha"),
        "iterations": record.get("iterations"),
        "final_grad_norm": diagnostics.get(
            "final_grad_norm", properties.get("final_scf_grad_norm")
        ),
        "imaginary_residual": properties.get("inverse_bloch_imaginary_residual"),
        "ccm_approach": convention.get("ccm_approach"),
        "ccm_construction": convention.get("ccm_construction"),
        "evaluation_representation": convention.get(
            "evaluation_representation"
        ),
        "boundary_model": convention.get("boundary_model"),
        "coulomb_kernel": convention.get("coulomb_kernel"),
        "exchange_q0": convention.get("exchange_q0"),
        "exchange_q0_applicability": convention.get(
            "exchange_q0_applicability"
        ),
        "expected_exchange_q0_applicability": (
            _expected_exchange_q0_applicability(record)
        ),
        "lattice_vector_convention": convention.get(
            "lattice_vector_convention"
        ),
        "bvk_lattice_matches_input": _bvk_lattice_matches_record(record),
        "exact_exchange_assembly_failure": exchange_assembly_failure,
        "two_electron_support_qualification": support.get("qualification"),
        "two_electron_support_failure": support_failure,
        "overlap_fold_support_failure": overlap_fold_failure,
        "absolute_energy_revision_failure": revision_failure,
        "reason": record.get("reason"),
        "retraction_reason": record.get("retraction_reason"),
    }
    energy_reportable = (
        status == "ok"
        and row["converged"] is True
        and not row["retracted"]
        and _is_finite_number(row["_raw_energy_per_atom_ha"])
        and not _has_bad_route_identity(row)
        and not _has_bad_convention(row)
        and not _has_unreportable_dimension(row)
        and not _has_bad_exact_exchange_assembly(row)
        and not _has_unqualified_two_electron_support(row)
        and not _has_unqualified_overlap_fold_support(row)
        and not _has_revision_bound_absolute_energy(row)
    )
    if not energy_reportable:
        row["energy_per_cell_ha"] = None
        row["energy_per_atom_ha"] = None
    return row


def _has_bad_convention(row: dict[str, Any]) -> bool:
    if row["status"] not in ("ok", "not_converged"):
        return False
    return (
        row["ccm_approach"] != EXPECTED_CCM_APPROACH
        or row["ccm_construction"] != EXPECTED_CCM_CONSTRUCTION
        or row["evaluation_representation"]
        != EXPECTED_EVALUATION_REPRESENTATION
        or row["coulomb_kernel"] != EXPECTED_COULOMB_KERNEL
        or row["exchange_q0"] != EXPECTED_EXCHANGE_Q0
        or row["exchange_q0_applicability"]
        != row["expected_exchange_q0_applicability"]
        or row["lattice_vector_convention"]
        != EXPECTED_LATTICE_VECTOR_CONVENTION
        or row["bvk_lattice_matches_input"] is not True
    )


def _has_bad_route_identity(row: dict[str, Any]) -> bool:
    """Return whether a reportable record contradicts its B route selector."""

    return (
        row["status"] in ("ok", "not_converged")
        and row["route_identity_matches"] is not True
    )


def _has_unreportable_dimension(row: dict[str, Any]) -> bool:
    """Return whether a record tries to report a lower-dimensional B energy."""

    return row["status"] in ("ok", "not_converged") and row["dim"] != 3


def _has_unqualified_two_electron_support(row: dict[str, Any]) -> bool:
    """Return whether a quantitative-status row lacks D82 support."""

    return (
        row["status"] in ("ok", "not_converged")
        and row["two_electron_support_failure"] is not None
    )


def _has_bad_exact_exchange_assembly(row: dict[str, Any]) -> bool:
    """Return whether a quantitative-status row lacks D98 v1 provenance."""

    return (
        row["status"] in ("ok", "not_converged")
        and row["exact_exchange_assembly_failure"] is not None
    )


def _has_unqualified_overlap_fold_support(row: dict[str, Any]) -> bool:
    """Return whether a quantitative-status row lacks D103 support."""

    return (
        row["status"] in ("ok", "not_converged")
        and row["overlap_fold_support_failure"] is not None
    )


def _has_revision_bound_absolute_energy(row: dict[str, Any]) -> bool:
    """Return whether the D102/D104 per-record gate rejects this row."""

    return (
        row["status"] in ("ok", "not_converged")
        and row["absolute_energy_revision_failure"] is not None
    )


def _is_finite_number(value: Any) -> bool:
    """Return whether ``value`` is a finite JSON number, excluding booleans."""

    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _record_consistency_failures(row: dict[str, Any]) -> list[str]:
    """Return contradictions between terminal status and result payload."""

    path = row["path"]
    status = row["status"]
    converged = row["converged"]
    if status == "ok":
        failures = []
        if converged is not True:
            failures.append(
                f"{path}: status=ok requires converged=true exactly; "
                f"got {converged!r}"
            )
        energy = row["_raw_energy_per_atom_ha"]
        if not _is_finite_number(energy):
            failures.append(
                f"{path}: status=ok requires finite numeric "
                f"energy_per_atom_ha; got {energy!r}"
            )
        return failures
    if (
        isinstance(status, str)
        and status in KNOWN_STATUSES
        and converged is True
    ):
        return [f"{path}: status={status} must not claim converged=true"]
    return []


def _print_table(rows: list[dict[str, Any]]) -> None:
    columns = (
        "system",
        "route",
        "status",
        "retracted",
        "converged",
        "dim",
        "mesh",
        "basis",
        "method",
        "backend",
        "functional",
        "route_identity_matches",
        "energy_per_atom_ha",
        "iterations",
        "final_grad_norm",
        "ccm_approach",
        "ccm_construction",
        "evaluation_representation",
        "boundary_model",
        "coulomb_kernel",
        "exchange_q0",
        "exchange_q0_applicability",
        "expected_exchange_q0_applicability",
        "lattice_vector_convention",
        "bvk_lattice_matches_input",
        "exact_exchange_assembly_failure",
        "two_electron_support_qualification",
        "two_electron_support_failure",
        "overlap_fold_support_failure",
        "absolute_energy_revision_failure",
        "reason",
        "retraction_reason",
    )
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        print("| " + " | ".join(_fmt(row[column]) for column in columns) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="directory containing fetched B JSON records")
    parser.add_argument("--csv", help="write the audit rows to CSV")
    parser.add_argument(
        "--allow-not-converged",
        action="store_true",
        help="do not fail the audit on status=not_converged",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="do not fail the audit on status=error",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="exit successfully if no B JSON records are found",
    )
    args = parser.parse_args()

    root = Path(args.results)
    try:
        records = _json_records(root)
    except ValueError as exc:
        print("# records=unavailable")
        print("# audit failures:")
        print(f"# - {exc}")
        raise SystemExit(2) from exc
    rows = [_row(path, record, root) for path, record in records]
    counts = Counter(_fmt(row["status"]) for row in rows)

    print(f"# records={len(rows)}")
    if counts:
        status_items = ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))
        print("# statuses=" + status_items)
    _print_table(rows)

    if args.csv:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as handle:
            fieldnames = (
                [key for key in rows[0] if not key.startswith("_")]
                if rows
                else [
                "path",
                "system",
                "route",
                "status",
                ]
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                {
                    key: value
                    for key, value in row.items()
                    if not key.startswith("_")
                }
                for row in rows
            )

    failures: list[str] = []
    if not rows and not args.allow_missing:
        failures.append("no aiccm2026dev-b JSON records found")
    failures.extend(
        f"{row['path']}: unexpected status={row['status']}"
        for row in rows
        if (
            not isinstance(row["status"], str)
            or row["status"] not in KNOWN_STATUSES
        )
    )
    failures.extend(
        f"{row['path']}: retracted result is failure evidence only"
        for row in rows
        if row["retracted"]
    )
    for row in rows:
        failures.extend(_record_consistency_failures(row))
    if not args.allow_not_converged:
        failures.extend(
            f"{row['path']}: status=not_converged"
            for row in rows
            if row["status"] == "not_converged"
        )
    if not args.allow_errors:
        failures.extend(
            f"{row['path']}: status=error"
            for row in rows
            if row["status"] == "error"
        )
    failures.extend(
        f"{row['path']}: route identity={row['route']}/"
        f"{row['method']}/{row['backend']}/{row['functional']}; "
        "method, backend, and functional must match b_routes.ROUTES exactly"
        for row in rows
        if _has_bad_route_identity(row)
    )
    failures.extend(
        f"{row['path']}: construction identity="
        f"{row['ccm_approach']}/{row['ccm_construction']}/"
        f"{row['evaluation_representation']}; Hamiltonian convention="
        f"{row['coulomb_kernel']}/{row['exchange_q0']}/"
        f"applicability-{row['exchange_q0_applicability']}"
        f"-expected-{row['expected_exchange_q0_applicability']}/"
        f"lattice-{row['lattice_vector_convention']}/"
        f"bvk-match-{row['bvk_lattice_matches_input']}"
        for row in rows
        if _has_bad_convention(row)
    )
    failures.extend(
        f"{row['path']}: periodic dimension={row['dim']}; "
        "lower-dimensional chi-CCM-B absolute energies are not reportable"
        for row in rows
        if _has_unreportable_dimension(row)
    )
    failures.extend(
        f"{row['path']}: invalid exact-exchange assembly provenance: "
        f"{row['exact_exchange_assembly_failure']}"
        for row in rows
        if _has_bad_exact_exchange_assembly(row)
    )
    failures.extend(
        f"{row['path']}: unreportable two-electron numerical support: "
        f"{row['two_electron_support_failure']}"
        for row in rows
        if _has_unqualified_two_electron_support(row)
    )
    failures.extend(
        f"{row['path']}: unreportable overlap-fold numerical support: "
        f"{row['overlap_fold_support_failure']}"
        for row in rows
        if _has_unqualified_overlap_fold_support(row)
    )
    failures.extend(
        f"{row['path']}: revision-bound absolute energy: "
        f"{row['absolute_energy_revision_failure']}"
        for row in rows
        if _has_revision_bound_absolute_energy(row)
    )
    if failures:
        print("# audit failures:")
        for failure in failures:
            print(f"# - {failure}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
