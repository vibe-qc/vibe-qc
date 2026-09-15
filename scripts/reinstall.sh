#!/bin/bash
# Failure-atomically reinstall the current vibe-qc checkout.
#
# USAGE
#     ./scripts/reinstall.sh [OPTIONS]
#
# OPTIONS
#     --venv PATH          Explicit virtualenv path. Default: first existing
#                          .venv / venv / .venv-vibeqc / venv-vibeqc.
#     --python BIN         Replacement interpreter. Default: preserve the
#                          existing virtualenv's recorded base interpreter.
#     --extras GROUP       Pip extras profile (default: test). Supports the
#                          same profiles as install.sh, including mpi,
#                          dispersion,mpi, and the co-located profiles.
#     --with-openblas      Build/retain vendored OpenBLAS.
#     --skip-native-deps   Reuse already-built native dependency installs.
#     --adopt-legacy       Permit an unmarked legacy venv only when trusted
#                          PEP 610 metadata links it to this exact checkout.
#     --dry-run            Preview the replacement without changing files.
#     -h, --help           Show this help.
#
# Reinstall never fetches or changes Git refs. The prior environment remains
# available for rollback until the replacement installs and verifies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPLICIT_VENV=""
PYTHON_BIN=""
EXTRAS_GROUP="test"
WITH_OPENBLAS=0
SKIP_NATIVE_DEPS=0
ADOPT_LEGACY=0
DRY_RUN=0

print_help() {
    awk '/^# USAGE/ {p=1} p && !/^#/ {exit} p { sub(/^# ?/, ""); print }' "$0"
}

require_value() {
    local option="$1"
    local kind="$2"
    local value="${3-}"
    [ -n "$value" ] || {
        echo "Error: $option requires a non-empty $kind." >&2
        exit 1
    }
    case "$value" in
        -*) echo "Error: $option requires a $kind, not option '$value'." >&2; exit 1 ;;
    esac
}

while [ $# -gt 0 ]; do
    case "$1" in
        --venv)
            require_value "$1" "path" "${2-}"
            EXPLICIT_VENV="$2"
            shift 2
            ;;
        --python)
            require_value "$1" "executable" "${2-}"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --extras)
            require_value "$1" "profile" "${2-}"
            EXTRAS_GROUP="$2"
            shift 2
            ;;
        --with-openblas) WITH_OPENBLAS=1; shift ;;
        --skip-native-deps) SKIP_NATIVE_DEPS=1; shift ;;
        --adopt-legacy) ADOPT_LEGACY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) print_help; exit 0 ;;
        *)
            echo "Error: unknown argument '$1'. Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# shellcheck source=_venv_helpers.sh
. "$SCRIPT_DIR/_venv_helpers.sh"
# shellcheck source=_setup_helpers.sh
. "$SCRIPT_DIR/_setup_helpers.sh"

cd "$REPO_ROOT"
vibeqc_extras_to_pip_spec "$EXTRAS_GROUP" _REINSTALL_EXTRAS_SPEC

VENV=""
vibeqc_detect_venv_target VENV "$EXPLICIT_VENV"
if [ -z "$VENV" ]; then
    echo "Error: no existing virtualenv to reinstall." >&2
    echo "Use ./scripts/install.sh for a fresh installation." >&2
    exit 1
fi
VENV_REQUESTED="$VENV"
vibeqc_resolve_venv_path VENV "$VENV_REQUESTED"
vibeqc_assert_removable_venv "$VENV" "$REPO_ROOT"

# Prove lifecycle ownership with a trusted external interpreter before reading
# pyvenv.cfg or resolving any interpreter path recorded by the target.
INSPECTION_REQUESTED="${PYTHON_BIN:-python3}"
vibeqc_select_trusted_python \
    INSPECTION_PYTHON "$INSPECTION_REQUESTED" "$VENV"
vibeqc_assert_owned_venv \
    "$VENV" "$REPO_ROOT" "$INSPECTION_PYTHON" "$ADOPT_LEGACY"
if [ "$DRY_RUN" != "1" ]; then
    vibeqc_acquire_lifecycle_lock \
        "$VENV" "$REPO_ROOT" reinstall-preflight "$INSPECTION_PYTHON"
    trap 'vibeqc_lifecycle_cleanup' EXIT
    # Preflight evidence is not authority after waiting for a lock. Recheck
    # ownership under the lock immediately before reading target metadata.
    vibeqc_assert_owned_venv \
        "$VENV" "$REPO_ROOT" "$INSPECTION_PYTHON" "$ADOPT_LEGACY"
fi

if [ -z "$PYTHON_BIN" ]; then
    if ! vibeqc_venv_base_python PYTHON_BIN "$VENV"; then
        echo "Error: could not determine the existing venv's base interpreter." >&2
        echo "Re-run with --python /path/to/python3." >&2
        exit 1
    fi
fi
vibeqc_select_trusted_python PYTHON_BIN "$PYTHON_BIN" "$VENV"
vibeqc_resolve_base_python PYTHON_BIN "$PYTHON_BIN"

INSTALL_ARGS=(
    --current
    --force
    --venv "$VENV"
    --python "$PYTHON_BIN"
    --extras "$EXTRAS_GROUP"
)
[ "$WITH_OPENBLAS" = "1" ] && INSTALL_ARGS+=(--with-openblas)
[ "$SKIP_NATIVE_DEPS" = "1" ] && INSTALL_ARGS+=(--skip-native-deps)
[ "$ADOPT_LEGACY" = "1" ] && INSTALL_ARGS+=(--adopt-legacy)
[ "$DRY_RUN" = "1" ] && INSTALL_ARGS+=(--dry-run)

if [ "$DRY_RUN" != "1" ]; then
    vibe_toolset_prepare_lifecycle_lock_handoff
    trap - EXIT
fi
exec "$SCRIPT_DIR/install.sh" "${INSTALL_ARGS[@]}"
