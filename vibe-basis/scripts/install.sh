#!/usr/bin/env bash
# Install standalone vibe-basis from this checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_venv_helpers.sh
. "$SCRIPT_DIR/_venv_helpers.sh"

PYTHON_BIN="python3"
VENV_INPUT="${VIBE_BASIS_VENV:-.venv}"
PROFILE="standard"
WITH_VQ=0
FORCE=0
DRY_RUN=0

print_help() {
    cat <<'EOF'
USAGE
    ./vibe-basis/scripts/install.sh [OPTIONS]

OPTIONS
    --extras PROFILE     core, standard (default), optimizers, all, or test.
                         standard includes SciPy; optimizers adds NLopt and
                         iminuit. vq is installed separately with --with-vq.
    --with-vq            Install the co-located vq CLI too (Python >=3.12).
    --python BIN         Python used to create the venv (default: python3).
    --venv PATH          Venv path (default: vibe-basis/.venv).
    --force              Replace an owned installation with automatic rollback.
    --dry-run            Resolve and validate without changing files.
    -h, --help           Show this help.

The installer supports macOS and Linux and does not build vibe-qc.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --extras) vibe_basis_require_value "$1" "${2-}" "profile"; PROFILE="$2"; shift 2 ;;
        --with-vq) WITH_VQ=1; shift ;;
        --python) vibe_basis_require_value "$1" "${2-}" "executable"; PYTHON_BIN="$2"; shift 2 ;;
        --venv) vibe_basis_require_value "$1" "${2-}" "path"; VENV_INPUT="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) print_help; exit 0 ;;
        *) vibe_basis_die "unknown argument '$1' (run with --help)" ;;
    esac
done

PYTHON_BIN="$(vibe_basis_command_path "$PYTHON_BIN")"
VENV_PATH="$(vibe_basis_resolve_venv "$PYTHON_BIN" "$VENV_INPUT")"
vibe_basis_assert_safe_target "$VENV_PATH"
EXTRAS_SPEC="$(vibe_basis_profile_spec "$PROFILE")"

if [ -e "$VENV_PATH" ]; then
    [ "$FORCE" = "1" ] || vibe_basis_die \
        "'$VENV_PATH' exists; use update.sh, reinstall.sh, or --force"
    vibe_basis_assert_removable_venv "$VENV_PATH"
fi
PYTHON_BIN="$(vibe_basis_resolve_base_python "$PYTHON_BIN" "$VENV_PATH")"
[ "$WITH_VQ" != "1" ] || vibe_basis_assert_vq_python "$PYTHON_BIN"

echo "==> vibe-basis install"
echo "    source:       $VIBE_BASIS_PROJECT_DIR"
echo "    venv:         $VENV_PATH"
echo "    python:       $PYTHON_BIN"
echo "    capabilities: $PROFILE${EXTRAS_SPEC:+ $EXTRAS_SPEC}"
echo "    vq CLI:       $([ "$WITH_VQ" = "1" ] && echo included || echo separate)"
[ "$DRY_RUN" = "1" ] && echo "    --dry-run:    no files will be changed"
echo

if [ "$DRY_RUN" = "1" ]; then
    echo "==> Dry-run complete."
    exit 0
fi

vibe_basis_assert_regular_user
vibe_basis_acquire_target_lock "$VENV_PATH" "$PYTHON_BIN"
trap 'vibe_basis_release_target_lock' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
vibe_basis_install_environment "$VENV_PATH" "$PYTHON_BIN" "$PROFILE" "$EXTRAS_SPEC" "$WITH_VQ"

echo
echo "==> vibe-basis install complete."
echo "    Activate: source \"$VENV_PATH/bin/activate\""
echo "    Start:    $VENV_PATH/bin/vb --help"
if [ "$WITH_VQ" != "1" ]; then
    echo "    Remote:   install vq separately, or rerun with --with-vq on Python >=3.12"
fi
