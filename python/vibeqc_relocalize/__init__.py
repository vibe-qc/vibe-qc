"""Dependency-light launcher for the vibe-qc relocalization protocol.

Kept outside ``vibeqc`` so a broken native install can still answer a probe.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from importlib.metadata import PackageNotFoundError, version

PROTOCOL = "vibeqc.relocalize"
PROTOCOL_VERSION = 1
WORKER_VERSION = "1.0.0"
MAX_REQUEST_BYTES = 16 * 1024 * 1024


class RequestError(ValueError):
    """A stable, machine-readable refusal, with no substitute calculation."""

    def __init__(self, code: str, message: str, field: str | None = None):
        super().__init__(message)
        self.code = code
        self.field = field


def package_version():
    try:
        return version("vibe-qc")
    except PackageNotFoundError:
        return "unknown"


def event(kind, request_id=None, **payload):
    return dict(
        protocol=PROTOCOL,
        protocol_version=PROTOCOL_VERSION,
        worker_version=WORKER_VERSION,
        id=request_id,
        event=kind,
        **payload,
    )


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RequestError("invalid_json", f"Duplicate object key: {key}")
        result[key] = value
    return result


def parse_request(line):
    def nonfinite(value):
        raise RequestError("invalid_json", f"Non-finite JSON number: {value}")

    try:
        # json.loads(bytes) also auto-detects UTF-16/32. Decode explicitly so
        # stdin really follows the UTF-8 JSONL transport contract.
        if isinstance(line, (bytes, bytearray)):
            line = line.decode("utf-8")
        value = json.loads(line, object_pairs_hook=_object, parse_constant=nonfinite)
    except (ValueError, UnicodeError) as exc:
        raise RequestError("invalid_json", str(exc)) from exc
    if not isinstance(value, dict):
        raise RequestError("invalid_request", "Request must be an object")
    if (
        value.get("protocol") != PROTOCOL
        or type(value.get("protocol_version")) is not int
        or value["protocol_version"] != 1
    ):
        raise RequestError(
            "unsupported_protocol", "Expected vibeqc.relocalize protocol_version 1"
        )
    if not isinstance(value.get("id"), str) or not 1 <= len(value["id"]) <= 128:
        raise RequestError(
            "invalid_request", "id must be a string of 1 to 128 characters", "id"
        )
    if value.get("operation") not in ("capabilities", "localize"):
        raise RequestError(
            "unsupported_operation", "Expected capabilities or localize", "operation"
        )
    return value


def capabilities():
    """Report tested readiness, not just whether a package can be found."""
    from ._capabilities import method_specs

    report = {
        "backend": "vibe-qc",
        "backend_version": package_version(),
        "ready": False,
        "native_core_ready": False,
        "methods": method_specs(),
        "limits": {
            "max_request_bytes": MAX_REQUEST_BYTES,
            "max_ao": 512,
            "max_periodic_torus_ao": 128,
        },
        "fresh_rhf": {"ready": False, "molecular_only": True},
    }
    try:
        from vibeqc.relocalize import probe

        probe(report)
    except Exception as exc:  # noqa: BLE001 - process/probe boundary
        traceback.print_exc(file=sys.stderr)
        report["error"] = {"code": "backend_unavailable", "message": str(exc)}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="vibe-qc relocalization JSONL worker (one request per process)"
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="emit tested capabilities; do not read stdin",
    )
    args = parser.parse_args(argv)
    # Preserve a protocol-only descriptor BEFORE importing the numerical package.
    # Redirect fd 1 as well as Python stdout: C/C++ diagnostics must go to stderr.
    sys.stdout.flush()
    protocol_out = os.fdopen(
        os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1
    )
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def emit(kind, request_id=None, **payload):
        protocol_out.write(
            json.dumps(
                event(kind, request_id, **payload),
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        protocol_out.flush()

    request_id = None
    with contextlib.redirect_stdout(sys.stderr):
        try:
            if args.probe:
                report = capabilities()
                emit("capabilities", capabilities=report)
                return 0 if report["ready"] else 3
            line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
            if len(line) > MAX_REQUEST_BYTES:
                raise RequestError("request_too_large", "Request exceeds 16 MiB")
            request = parse_request(line)
            request_id = request["id"]
            if request["operation"] == "capabilities":
                if set(request) - {"protocol", "protocol_version", "id", "operation"}:
                    raise RequestError(
                        "invalid_request", "Capabilities request has unexpected fields"
                    )
                report = capabilities()
                emit("capabilities", request_id, capabilities=report)
                return 0 if report["ready"] else 3
            emit("started", request_id, stage="validating")
            try:
                from vibeqc.relocalize import localize
            except Exception as exc:
                raise RequestError("backend_unavailable", str(exc)) from exc
            result = localize(request)
            emit("result", request_id, result=result)
            return 0
        except RequestError as exc:
            emit(
                "error",
                request_id,
                error={"code": exc.code, "message": str(exc), "field": exc.field},
            )
            return 2
        except Exception as exc:  # noqa: BLE001 - process/probe boundary
            traceback.print_exc(file=sys.stderr)
            emit(
                "error",
                request_id,
                error={
                    "code": "computation_failed",
                    "message": str(exc),
                    "field": None,
                },
            )
            return 3
        finally:
            protocol_out.close()
