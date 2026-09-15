#!/usr/bin/env bash
# Safely remove the standalone vibe-basis environment owned by this checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_venv_helpers.sh
. "$SCRIPT_DIR/_venv_helpers.sh"

PYTHON_BIN="python3"
PYTHON_OPTION_SET=0
VENV_INPUT="${VIBE_BASIS_VENV:-.venv}"
DRY_RUN=0

print_help() {
    cat <<'EOF'
USAGE
    ./vibe-basis/scripts/uninstall.sh [OPTIONS]

OPTIONS
    --python BIN          Trusted Python used to resolve paths (default: python3).
    --venv PATH           Standalone venv (default: vibe-basis/.venv).
    --dry-run             Validate ownership and show what would be removed.
    -h, --help            Show this help.

Only the marker-owned standalone venv is removed. Source, calculations,
configuration, caches, and external-program outputs are preserved.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --python) vibe_basis_require_value "$1" "${2-}" "executable"; PYTHON_BIN="$2"; PYTHON_OPTION_SET=1; shift 2 ;;
        --venv) vibe_basis_require_value "$1" "${2-}" "path"; VENV_INPUT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) print_help; exit 0 ;;
        *) vibe_basis_die "unknown argument '$1' (run with --help)" ;;
    esac
done

PYTHON_BIN="$(vibe_basis_command_path "$PYTHON_BIN")"
VENV_PATH="$(vibe_basis_resolve_venv "$PYTHON_BIN" "$VENV_INPUT")"
vibe_basis_assert_safe_target "$VENV_PATH"

echo "==> vibe-basis uninstall"
echo "    venv:      $VENV_PATH"
echo "    user data: preserved"
[ "$DRY_RUN" = "1" ] && echo "    --dry-run: no files will be changed"
echo

if [ ! -e "$VENV_PATH" ]; then
    echo "No standalone vibe-basis environment was found; nothing to remove."
    exit 0
fi
vibe_basis_assert_removable_venv "$VENV_PATH"

if [ "$DRY_RUN" = "1" ]; then
    echo "Would remove marker-owned virtualenv -> $VENV_PATH"
    exit 0
fi

vibe_basis_assert_regular_user
if [ "$PYTHON_OPTION_SET" = "1" ]; then
    vibe_toolset_select_external_python LOCK_PYTHON "$PYTHON_BIN" "$VENV_PATH"
else
    vibe_toolset_find_external_python LOCK_PYTHON "$VENV_PATH"
fi
vibe_basis_acquire_target_lock "$VENV_PATH" "$LOCK_PYTHON"
trap 'vibe_basis_release_target_lock' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
vibe_basis_assert_removable_venv "$VENV_PATH"
vibe_basis_remove_owned_venv "$VENV_PATH"
echo "==> vibe-basis uninstall complete. Source and user data were retained."
