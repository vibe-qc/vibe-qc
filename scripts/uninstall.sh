#!/bin/bash
# Remove a standalone vibe-qc virtualenv without touching source or native deps.
#
# USAGE
#     ./scripts/uninstall.sh [OPTIONS]
#
# OPTIONS
#     --venv PATH    Explicit virtualenv path. Default: first existing one of
#                    .venv / venv / .venv-vibeqc / venv-vibeqc.
#     --python BIN   Trusted external Python used to inspect ownership
#                    metadata (default: python3). The macOS system Python is
#                    sufficient; it is not used to run vibe-qc.
#     --adopt-legacy Permit an unmarked legacy venv only when trusted PEP 610
#                    metadata proves it came from this exact checkout.
#     --dry-run      Show the validated removal target without changing it.
#     -h, --help     Show this help.
#
# This removes only a recognisable Python virtualenv. The checkout,
# third_party native dependencies, build caches, and user calculation data are
# retained. Run scripts/install.sh later to install again.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPLICIT_VENV=""
PYTHON_BIN="python3"
PYTHON_OPTION_SET=0
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
            PYTHON_OPTION_SET=1
            shift 2
            ;;
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
# shellcheck source=_build_lock.sh
. "$SCRIPT_DIR/_build_lock.sh"

cd "$REPO_ROOT"
VENV=""
vibeqc_detect_venv_target VENV "$EXPLICIT_VENV"
if [ -z "$VENV" ]; then
    echo "==> vibe-qc is not installed in a recognised checkout virtualenv."
    echo "    Nothing to remove."
    exit 0
fi

VENV_REQUESTED="$VENV"
vibeqc_resolve_venv_path VENV "$VENV_REQUESTED"
if [ "$PYTHON_OPTION_SET" = "1" ]; then
    vibe_toolset_select_external_python INSPECTION_PYTHON "$PYTHON_BIN" "$VENV"
else
    vibe_toolset_find_external_python INSPECTION_PYTHON "$VENV"
fi
vibeqc_assert_owned_venv \
    "$VENV" "$REPO_ROOT" "$INSPECTION_PYTHON" "$ADOPT_LEGACY"

echo "==> vibe-qc uninstall"
echo "    virtualenv:  $VENV"
echo "    retained:    checkout, third_party native deps, build caches, user data"
if [ "$DRY_RUN" = "1" ]; then
    echo "    --dry-run: no files will be changed"
    exit 0
fi

vibeqc_acquire_lifecycle_lock \
    "$VENV" "$REPO_ROOT" uninstall "$INSPECTION_PYTHON"
trap 'vibeqc_lifecycle_cleanup' EXIT
vibeqc_acquire_build_lock
vibeqc_assert_owned_venv \
    "$VENV" "$REPO_ROOT" "$INSPECTION_PYTHON" "$ADOPT_LEGACY"
vibeqc_remove_venv "$VENV" "$REPO_ROOT"
vibeqc_release_lifecycle_lock

echo
echo "==> Uninstall complete. Reinstall later with ./scripts/install.sh"
