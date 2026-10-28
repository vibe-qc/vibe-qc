#!/usr/bin/env bash
# Optional external observer for a fresh, single-stage CMake build.
# This helper records no provenance itself and is not a cache-admission gate.

vibeqc_enable_native_collector() {
    if [ -z "${VIBEQC_NATIVE_BUILD_COLLECTOR:-}" ]; then
        if [ -n "${VIBEQC_NATIVE_RECEIPT_DIR:-}${VIBEQC_NATIVE_SOURCE_ARCHIVE:-}${VIBEQC_NATIVE_SEED_CONTEXT:+enabled}" ]; then
            echo "receipt/archive settings require VIBEQC_NATIVE_BUILD_COLLECTOR" >&2
            return 3
        fi
        return 0
    fi
    if [ "${VIBEQC_NATIVE_SEED_CONTEXT:-0}" != 0 ]; then
        if [ "${VIBEQC_NATIVE_SEED_CONTEXT:-}" != 1 ] || [ "$1" != fftw ] || [ -z "${VIBEQC_NATIVE_SOURCE_ARCHIVE:-}" ]; then
            echo "seed context requires opt-in unpatched FFTW archive collection" >&2
            return 3
        fi
        [ ! -e "$REPO_ROOT/.vq-native-fftw.json" ] && [ ! -L "$REPO_ROOT/.vq-native-fftw.json" ] || {
            echo "fresh seed envelope already exists" >&2; return 3;
        }
    fi
    # Validate before removing/configuring a build tree. Resolve symlinks so an
    # external-looking receipt directory cannot write inside the product tree.
    python3 - "$VIBEQC_NATIVE_BUILD_COLLECTOR" "${VIBEQC_NATIVE_RECEIPT_DIR:-}" \
        "$REPO_ROOT" "$INSTALL_DIR" "$1" <<'PYTHON'
import os
from pathlib import Path
import sys

collector, directory, root, install, name = sys.argv[1:]
try:
    executable = Path(collector)
    destination = Path(directory)
    if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("collector must be an absolute executable file")
    if not destination.is_absolute() or not destination.is_dir():
        raise ValueError("receipt directory must be an absolute existing directory")
    destination = destination.resolve(strict=True)
    for boundary in (Path(root).resolve(), Path(install).resolve()):
        if destination == boundary or boundary in destination.parents:
            raise ValueError("receipt directory must be outside the product and install trees")
    receipt = destination / (name + ".json")
    if receipt.exists() or receipt.is_symlink():
        raise ValueError("receipt already exists; select a fresh external receipt directory")
except (OSError, ValueError) as error:
    print("native collector: " + str(error), file=sys.stderr)
    raise SystemExit(3)
PYTHON
    local status=$?
    [ "$status" -eq 0 ] || return "$status"
    _vibeqc_collector_name="$1"
    _vibeqc_collector_spec=""
    if [ -n "${VIBEQC_NATIVE_SOURCE_ARCHIVE:-}" ]; then
        # Only the unpatched pinned FFTW archive has an authenticated contract.
        # Never infer source provenance from installed bytes or a version name.
        [ "$1" = fftw ] || { echo "archive collector supports unpatched FFTW only" >&2; return 3; }
        _vibeqc_collector_spec="$VIBEQC_NATIVE_RECEIPT_DIR/fftw-spec.json"
        python3 - "$VIBEQC_NATIVE_SOURCE_ARCHIVE" "${FFTW_SHA256:-}" \
            "${FFTW_VERSION:-}" "$SRC_DIR" "$BUILD_DIR" "$INSTALL_DIR" \
            "$SCRIPT_DIR" "$REPO_ROOT" "$_vibeqc_collector_spec" <<'PYTHON'
import hashlib
import json
from pathlib import Path
import re
import sys

archive, expected, version, source, build, install, scripts, root, output = sys.argv[1:]
try:
    archive = Path(archive)
    if not archive.is_absolute() or not archive.is_file() or archive.is_symlink():
        raise ValueError("source archive must be an absolute regular file")
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", version):
        raise ValueError("missing FFTW recipe pin")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("source archive differs from the FFTW recipe pin")
    spec = {"sources": {"fftw": source},
            "authenticated_sources": {"fftw": {"archive": str(archive.resolve()),
                "sha256": expected, "prefix": "fftw-" + version}},
            "recipes": [str(Path(scripts) / name) for name in
                ("build_fftw.sh", "_native_collector.sh", "_safe_build_env.sh",
                 "_verify_source.sh", "_build_lock.sh")],
            "cmake_builds": [build], "install": install, "artifacts": [],
            "artifact_manifest": str(Path(build) / "install_manifest.txt"), "cwd": root}
    # The external collector must authenticate extracted bytes before/after the
    # build. This file merely passes the recipe's pin, not a qualification.
    with Path(output).open("x") as stream:
        json.dump(spec, stream, sort_keys=True)
except (OSError, ValueError) as error:
    print("native collector: " + str(error), file=sys.stderr)
    raise SystemExit(3)
PYTHON
        local spec_status=$?
        [ "$spec_status" -eq 0 ] || return "$spec_status"
    fi
    # Defined only when explicitly enabled. Neither cache hits nor fetch-only
    # callers reach here. The original CMake command lines stay unchanged.
    cmake() {
        case "${1:-}" in
            -S)
                command cmake "$@" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
                ;;
            --build)
                if [ -n "$_vibeqc_collector_spec" ]; then
                    if [ "${VIBEQC_NATIVE_SEED_CONTEXT:-0}" = 1 ]; then
                        python3 "$SCRIPT_DIR/native_seed_context.py" before "$BUILD_DIR" "$REPO_ROOT" \
                            "$VIBEQC_NATIVE_RECEIPT_DIR/fftw-context.json" || return $?
                    fi
                    "$VIBEQC_NATIVE_BUILD_COLLECTOR" record \
                        --spec "$_vibeqc_collector_spec" \
                        --output "$VIBEQC_NATIVE_RECEIPT_DIR/$_vibeqc_collector_name.json" \
                        -- cmake "$@"
                    local collector_status=$?
                    [ "$collector_status" -eq 0 ] || return "$collector_status"
                    if [ "${VIBEQC_NATIVE_SEED_CONTEXT:-0}" = 1 ]; then
                        python3 "$SCRIPT_DIR/native_seed_context.py" seal "$REPO_ROOT" "$BUILD_DIR" \
                            "$VIBEQC_NATIVE_RECEIPT_DIR/fftw.json" "$VIBEQC_NATIVE_RECEIPT_DIR/fftw-context.json" || return $?
                    fi
                    return 0
                fi
                "$VIBEQC_NATIVE_BUILD_COLLECTOR" record \
                    --source "$_vibeqc_collector_name=$SRC_DIR" \
                    --recipe "$SCRIPT_DIR/build_$_vibeqc_collector_name.sh" \
                    --recipe "$SCRIPT_DIR/_native_collector.sh" \
                    --recipe "$SCRIPT_DIR/_safe_build_env.sh" \
                    --recipe "$SCRIPT_DIR/_verify_source.sh" \
                    --recipe "$SCRIPT_DIR/_build_lock.sh" \
                    --cmake-build "$BUILD_DIR" --install "$INSTALL_DIR" \
                    --artifact-manifest "$BUILD_DIR/install_manifest.txt" \
                    --cwd "$REPO_ROOT" \
                    --output "$VIBEQC_NATIVE_RECEIPT_DIR/$_vibeqc_collector_name.json" \
                    -- cmake "$@"
                # Propagate failure. A collector may already have run the build;
                # never retry or fall through to an unobserved install.
                ;;
            *) command cmake "$@" ;;
        esac
    }
}
