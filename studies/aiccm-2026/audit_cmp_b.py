#!/usr/bin/env python3
"""Audit D101 χ route-control JSON without rendering an SCF energy."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_case_cmp_b import (
    EVIDENCE_ROLE,
    QUANTITATIVE_STATUS,
    _json_exact,
    campaign_record_failures,
    _is_full_commit,
    _strict_json_loads,
    _validation_receipt,
)


def audit_path(
    path: Path,
    expected_source_sha: str,
    validation_record: Path | None = None,
) -> list[str]:
    """Return every D101 contract failure for one result path."""

    try:
        record = _strict_json_loads(path.read_bytes())
    except Exception:  # noqa: BLE001 - CLI audit must fail closed
        return ["cannot read strict D101 JSON"]
    failures = campaign_record_failures(
        record,
        expected_source_sha=expected_source_sha,
    )
    if isinstance(record, dict) and record.get("system") in {"1d", "2d"}:
        provenance = record.get("provenance")
        current_provenance = provenance if isinstance(provenance, dict) else {}
        binding, receipt_failures = _validation_receipt(
            validation_record,
            expected_source_sha,
            current_provenance,
        )
        failures.extend(receipt_failures)
        if binding is not None and not _json_exact(
            record.get("validation_receipt"), binding
        ):
            failures.append(
                "serialized receipt does not equal the supplied 3d record binding"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument(
        "--validation-record",
        type=Path,
        help="exact fetched 3d JSON required when auditing 1d/2d records",
    )
    args = parser.parse_args()

    expected_source_sha = args.expected_source_sha.strip().lower()
    if not _is_full_commit(expected_source_sha):
        parser.error("--expected-source-sha must be one full 40-hex SHA")

    failed = False
    for path in args.paths:
        failures = audit_path(path, expected_source_sha, args.validation_record)
        if failures:
            failed = True
            print(f"{path}: invalid D101 diagnostic")
            for failure in failures:
                print(f"  - {failure}")
        else:
            print(
                f"{path}: contract-valid nonquantitative {EVIDENCE_ROLE}; "
                f"quantitative_status={QUANTITATIVE_STATUS}"
            )
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
