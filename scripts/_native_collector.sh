#!/usr/bin/env bash
# Optional external observer for a fresh, single-stage CMake build.
# This helper records no provenance itself and is not a cache-admission gate.

vibeqc_enable_native_collector() {
    if [ -z "${VIBEQC_NATIVE_BUILD_COLLECTOR:-}" ]; then
        if [ -n "${VIBEQC_NATIVE_RECEIPT_DIR:-}" ]; then
            echo "VIBEQC_NATIVE_RECEIPT_DIR requires VIBEQC_NATIVE_BUILD_COLLECTOR" >&2
            return 3
        fi
        return 0
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
    # Defined only when explicitly enabled. Neither cache hits nor fetch-only
    # callers reach here. The original CMake command lines stay unchanged.
    cmake() {
        case "${1:-}" in
            -S)
                command cmake "$@" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
                ;;
            --build)
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
