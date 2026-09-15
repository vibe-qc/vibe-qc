"""Run AICCM cases and store JSON records in the test-set layout.

The nested output layout is:

    <TESTSET>/<selector>/<system>/<basis>/<route>/{result.json,RUN.meta}

By default the helper targets a sibling ``qc-input-library/aiccm2026testset``
checkout when one is present. Use ``--testset-root`` or
``AICCM2026_TESTSET_ROOT`` to point at another copy.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import compare_b


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class StreamConfig:
    selector: str
    runner: str
    output_name_template: str
    result_glob: str
    energy_key: str

    def output_name(self, system: str, route: str) -> str:
        return self.output_name_template.format(system=system, route=route)


@dataclass(frozen=True)
class IngestSummary:
    matched: int
    written: int


STREAMS = {
    "aiccm2026dev-a": StreamConfig(
        selector="aiccm2026dev-a",
        runner="run_case.py",
        output_name_template="{system}__{route}.json",
        result_glob="*__*.json",
        energy_key="energy_per_atom",
    ),
    "aiccm2026dev-b": StreamConfig(
        selector="aiccm2026dev-b",
        runner="run_case_b.py",
        output_name_template="{system}__b-{route}.json",
        result_glob="*__b-*.json",
        energy_key="energy_per_atom_ha",
    ),
}


def _sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=HERE,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _provenance() -> dict[str, str]:
    """Which machine and which *build* produced this number.

    ``host`` + ``date`` alone cannot answer "was this computed by a stale core?".
    On 2026-07-10 the curated c-diamond ``aiccm-ri`` energy (-37.390860, dated
    2026-07-08) turned out to be irreproducible on current main (-37.415346, a
    -24.5 mHa/atom shift), and nothing in RUN.meta said which vibeqc, which commit,
    or which compiled core had produced it. ``f8c213e8``
    (cpp/src/aopair_ft.cpp) is a plausible mechanism because it corrupted
    momentum-shifted multi-k GDF fits in that window, but causality was not
    established for the archived diamond row. A stale ``.so`` returns wrong
    numbers rather than failing to import. Record enough to settle the question
    next time.
    """
    import platform

    # short hostname, matching the existing `host: workstation (local)` style
    node = (platform.node() or "unknown").split(".")[0]
    prov = {
        "host": node,
        "vibeqc_version": "unknown",
        "vibeqc_commit": _sha(),
        "source_clean": "unknown",
        "core_built": "unknown",
        "core_build_id": "unknown",
        "probe_version": "not-recorded",
        "probe_script_id": "unknown",
        "probe_passed": "false",
    }
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=HERE,
            text=True,
        )
        prov["source_clean"] = "true" if not status.strip() else "false"
    except Exception:
        pass
    try:
        import hashlib

        import vibeqc

        prov["vibeqc_version"] = str(vibeqc.__version__)
        from vibeqc import _vibeqc_core

        so = Path(getattr(_vibeqc_core, "__file__", "") or "")
        if so.is_file():
            prov["core_built"] = datetime.datetime.fromtimestamp(
                so.stat().st_mtime).isoformat(timespec="seconds")
            digest = hashlib.sha256()
            with so.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            prov["core_build_id"] = "sha256:" + digest.hexdigest()
    except Exception:  # noqa: BLE001 -- provenance must never break curation
        pass
    return prov


def _recorded_provenance(record: dict[str, Any]) -> dict[str, str]:
    """Read producer provenance from an ingested result without inventing it."""

    nested = record.get("provenance")
    provenance = nested if isinstance(nested, dict) else {}
    result: dict[str, str] = {}
    for key in (
        "host",
        "vibeqc_version",
        "vibeqc_commit",
        "source_clean",
        "core_built",
        "core_build_id",
        "probe_version",
        "probe_script_id",
        "probe_passed",
        "probe_attestation_id",
        "producer_payload_id",
    ):
        value = provenance.get(key)
        if isinstance(value, bool):
            result[key] = "true" if value else "false"
        else:
            result[key] = (
                str(value) if value not in (None, "") else "not-recorded"
            )
    return result


def default_testset_root() -> Path:
    env = os.environ.get("AICCM2026_TESTSET_ROOT")
    if env:
        return Path(env).expanduser()
    # HERE is <checkout>/studies/aiccm-2026; parents[2] is the directory
    # holding the checkout, where qc-input-library is a sibling clone.
    sibling = HERE.parents[2] / "qc-input-library" / "aiccm2026testset"
    if sibling.exists():
        return sibling
    return HERE / "_curated_testset"


def _finite_convention(record: dict[str, Any]) -> dict[str, Any]:
    convention = record.get("finite_torus_convention")
    if isinstance(convention, dict):
        return convention
    return {}


def _exchange_q0(record: dict[str, Any]) -> Any:
    convention = _finite_convention(record)
    return convention.get("exchange_q0", record.get("exchange_q0"))


def _state(record: dict[str, Any], return_code: int | None) -> str:
    if isinstance(record.get("status"), str):
        return str(record["status"])
    if return_code == 0:
        return "completed"
    return "failed"


def _write_meta(
    path: Path,
    *,
    stream: StreamConfig,
    system: str,
    route: str,
    basis: str,
    record: dict[str, Any],
    return_code: int | None,
    source: str,
    provenance: dict[str, str] | None = None,
) -> None:
    state = _state(record, return_code)
    energy = record.get(stream.energy_key)
    prov = _provenance() if provenance is None else provenance
    meta = (
        f"host: {prov['host']}\n"
        f"date: {datetime.date.today().isoformat()}\n"
        f"vibeqc_version: {prov['vibeqc_version']}\n"
        f"vibeqc_commit: {prov['vibeqc_commit']}\n"
        f"source_clean: {prov['source_clean']}\n"
        f"core_built: {prov['core_built']}\n"
        f"core_build_id: {prov['core_build_id']}\n"
        f"probe_version: {prov['probe_version']}\n"
        f"probe_script_id: {prov.get('probe_script_id', 'not-recorded')}\n"
        f"probe_passed: {prov['probe_passed']}\n"
        f"probe_attestation_id: "
        f"{prov.get('probe_attestation_id', 'not-recorded')}\n"
        f"producer_payload_id: "
        f"{prov.get('producer_payload_id', 'not-recorded')}\n"
        f"state: {state}\n"
        f"return_code: {'not-recorded' if return_code is None else return_code}\n"
        f"code: {stream.selector}\n"
        f"system: {system}\n"
        f"basis: {basis}\n"
        f"route: {route}\n"
        f"energy_per_atom_ha: {energy}\n"
        f"converged: {record.get('converged')}\n"
        f"exchange_q0: {_exchange_q0(record)}\n"
        f"source: {source}\n"
    )
    path.write_text(meta, encoding="utf-8")


def _store_record(
    record: dict[str, Any],
    *,
    stream: StreamConfig,
    testset_root: Path,
    return_code: int | None,
    source: str,
    system_fallback: str | None = None,
    route_fallback: str | None = None,
    basis_fallback: str | None = None,
    provenance: dict[str, str] | None = None,
) -> Path:
    stored_record, system, route, basis = _prepare_stored_record(
        record,
        stream=stream,
        system_fallback=system_fallback,
        route_fallback=route_fallback,
        basis_fallback=basis_fallback,
    )
    dest = testset_root / stream.selector / system / basis / route
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "result.json").write_text(
        json.dumps(stored_record, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_meta(
        dest / "RUN.meta",
        stream=stream,
        system=system,
        route=route,
        basis=basis,
        record=stored_record,
        return_code=return_code,
        source=source,
        provenance=provenance,
    )
    return dest


def _prepare_stored_record(
    record: dict[str, Any],
    *,
    stream: StreamConfig,
    system_fallback: str | None = None,
    route_fallback: str | None = None,
    basis_fallback: str | None = None,
) -> tuple[dict[str, Any], str, str, str]:
    """Normalize one record exactly as it will be identified on disk."""

    system = str(record.get("system") or system_fallback or "unknown-system")
    route = str(record.get("route") or route_fallback or "unknown-route")
    basis = str(record.get("basis") or basis_fallback or "unknown-basis")
    stored_record = dict(record)
    declared_selector = stored_record.get("selector") or stored_record.get(
        "stream"
    )
    if declared_selector not in (None, stream.selector):
        raise ValueError(
            f"record selector {declared_selector!r} does not match "
            f"requested stream {stream.selector!r}"
        )
    stored_record.setdefault("selector", stream.selector)
    stored_record.setdefault("system", system)
    stored_record.setdefault("route", route)
    stored_record.setdefault("basis", basis)
    return stored_record, system, route, basis


def _record_matches_stream(
    path: Path,
    record: dict[str, Any],
    stream: StreamConfig,
) -> bool:
    selector = record.get("selector") or record.get("stream")
    if isinstance(selector, str):
        return selector == stream.selector
    has_b_name = "__b-" in path.name
    return has_b_name if stream.selector == "aiccm2026dev-b" else not has_b_name


def _has_trusted_producer_attestation(
    record: dict[str, Any],
    stream: StreamConfig,
) -> bool:
    """Validate producer evidence before admitting a successful energy."""

    return (
        compare_b._producer_identity(
            record,
            expected_stream=stream.selector,
        )
        is not None
    )


def curate(
    system: str,
    route: str,
    basis: str,
    extra: list[str],
    *,
    stream: StreamConfig,
    testset_root: Path,
    tmp_dir: Path | None = None,
) -> bool:
    out = tmp_dir or HERE / "_curate_tmp"
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / stream.output_name(system, route)
    result_path.unlink(missing_ok=True)
    cmd = ["bash", str(HERE / "run.sh")]
    if stream.selector == "aiccm2026dev-b":
        cmd.append("--b")
    cmd.extend(
        [
            system,
            route,
            "--basis",
            basis,
            "--out",
            str(out),
            *extra,
        ]
    )
    environment = dict(os.environ)
    environment["VIBEQC_PYTHON"] = sys.executable
    run = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=environment,
    )
    if not result_path.exists():
        tail = (run.stderr or run.stdout)[-200:].strip()
        print(f"[FAIL] {system} {route} {basis}: rc={run.returncode} {tail}")
        return False

    record = json.loads(result_path.read_text(encoding="utf-8"))
    status = _state(record, run.returncode)
    reportable_success = record.get("converged") is True and status in {
        "ok",
        "completed",
    }
    if reportable_success and not _has_trusted_producer_attestation(record, stream):
        print(
            f"[FAIL] {system} {route} {basis}: producer attestation is not trusted"
        )
        return False
    dest = _store_record(
        record,
        stream=stream,
        testset_root=testset_root,
        return_code=run.returncode,
        source=f"local curate.py ({stream.runner}, HEAD {_sha()})",
        system_fallback=system,
        route_fallback=route,
        basis_fallback=basis,
        provenance=_recorded_provenance(record),
    )

    energy = record.get(stream.energy_key)
    print(
        f"[{status.upper():11s}] {system:18s} {route:18s} {basis:14s} "
        f"E/atom={energy} -> {dest.relative_to(testset_root)}"
    )
    return run.returncode == 0 or status == "unsupported"


def ingest_results(
    results_root: Path,
    *,
    stream: StreamConfig,
    testset_root: Path,
) -> IngestSummary:
    """Import already-fetched run_case JSON records into the nested layout."""
    candidates: list[tuple[Path, dict[str, Any]]] = []
    identities: dict[tuple[str, str, str], Path] = {}
    matched = 0
    for path in sorted(results_root.rglob(stream.result_glob)):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not _record_matches_stream(path, record, stream):
            continue
        if "system" not in record or "route" not in record:
            continue
        stored_record, system, route, basis = _prepare_stored_record(
            record,
            stream=stream,
        )
        identity = (system, basis, route)
        previous = identities.get(identity)
        if previous is not None:
            raise ValueError(
                "duplicate destination identity in ingest batch: "
                f"{stream.selector}/{system}/{basis}/{route} from "
                f"{previous.relative_to(results_root)} and "
                f"{path.relative_to(results_root)}"
            )
        identities[identity] = path

        existing_path = (
            testset_root
            / stream.selector
            / system
            / basis
            / route
            / "result.json"
        )
        if existing_path.exists():
            try:
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise ValueError(
                    f"refusing to overwrite unreadable record {existing_path}"
                ) from exc
            if existing != stored_record:
                raise ValueError(
                    f"refusing to overwrite differing record {existing_path}"
                )
            matched += 1
            continue
        candidates.append((path, stored_record))
        matched += 1

    written = 0
    for path, record in candidates:
        dest = _store_record(
            record,
            stream=stream,
            testset_root=testset_root,
            return_code=None,
            source=f"ingest {path.relative_to(results_root)} (HEAD {_sha()})",
            provenance=_recorded_provenance(record),
        )
        status = _state(record, None)
        energy = record.get(stream.energy_key)
        print(
            f"[{status.upper():11s}] {record.get('system', ''):18s} "
            f"{record.get('route', ''):18s} {record.get('basis', ''):14s} "
            f"E/atom={energy} -> {dest.relative_to(testset_root)}"
        )
        written += 1
    return IngestSummary(matched=matched, written=written)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", nargs="?")
    parser.add_argument("routes", nargs="?", help="Comma-separated route names")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument(
        "--stream",
        choices=tuple(STREAMS),
        default="aiccm2026dev-a",
        help="AICCM implementation line to run and store",
    )
    parser.add_argument(
        "--testset-root",
        type=Path,
        default=None,
        help="Root containing aiccm2026dev-a/ and aiccm2026dev-b/",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Scratch directory for run_case JSON output",
    )
    parser.add_argument(
        "--ingest-results",
        type=Path,
        default=None,
        help="Import existing run_case JSON records from this fetched-results tree",
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, extra = _parse_args(sys.argv[1:] if argv is None else argv)
    stream = STREAMS[args.stream]
    testset_root = (
        args.testset_root.expanduser()
        if args.testset_root is not None
        else default_testset_root()
    )
    if args.ingest_results is not None:
        summary = ingest_results(
            args.ingest_results.expanduser(),
            stream=stream,
            testset_root=testset_root,
        )
        print(
            f"=== {stream.selector}: {summary.matched} matched; "
            f"{summary.written} written ==="
        )
        return 0 if summary.matched else 1
    if not args.system or not args.routes:
        raise SystemExit("system and routes are required unless --ingest-results is used")
    routes = [item for item in args.routes.split(",") if item]
    ok = sum(
        curate(
            args.system,
            route,
            args.basis,
            extra,
            stream=stream,
            testset_root=testset_root,
            tmp_dir=args.tmp_dir,
        )
        for route in routes
    )
    print(
        f"=== {stream.selector} {args.system}/{args.basis}: "
        f"{ok}/{len(routes)} curated ==="
    )
    return 0 if ok == len(routes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
