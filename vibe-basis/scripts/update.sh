#!/usr/bin/env bash
# Update the checkout (optionally) and rebuild standalone vibe-basis with rollback.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_venv_helpers.sh
. "$SCRIPT_DIR/_venv_helpers.sh"

PYTHON_INPUT=""
VENV_INPUT="${VIBE_BASIS_VENV:-.venv}"
PROFILE_INPUT=""
WITH_VQ_INPUT=""
BRANCH_INPUT=""
BRANCH_SOURCE=""
SKIP_GIT=0
DRY_RUN=0

print_help() {
    cat <<'EOF'
USAGE
    ./vibe-basis/scripts/update.sh [OPTIONS]

OPTIONS
    --skip-git            Rebuild from the current checkout without Git changes.
    --branch NAME         Fast-forward this checkout to origin/NAME first.
    --dev                 Alias for --branch main.
    --release             Alias for --branch release.
    --extras PROFILE      Override the installed capability profile.
    --with-vq             Install/update the co-located vq CLI.
    --without-vq          Keep vq separate.
    --python BIN          Override the existing environment's base interpreter.
    --venv PATH           Venv path (default: vibe-basis/.venv).
    --recreate-venv       Accepted for consistency; updates always rebuild safely.
    --dry-run             Show the update without changing Git or the venv.
    -h, --help            Show this help.

Without a branch selector, Git fast-forwards the currently checked-out branch.
EOF
}

set_branch() {
    _vb_branch="$1"
    _vb_source="$2"
    [ -z "$BRANCH_SOURCE" ] || vibe_basis_die \
        "$_vb_source conflicts with branch selector $BRANCH_SOURCE"
    BRANCH_INPUT="$_vb_branch"
    BRANCH_SOURCE="$_vb_source"
}

set_vq_choice() {
    _vb_choice="$1"
    _vb_source="$2"
    [ -z "$WITH_VQ_INPUT" ] || vibe_basis_die \
        "$_vb_source conflicts with the previous vq selection"
    WITH_VQ_INPUT="$_vb_choice"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-git) SKIP_GIT=1; shift ;;
        --branch) vibe_basis_require_value "$1" "${2-}" "name"; set_branch "$2" "--branch $2"; shift 2 ;;
        --dev) set_branch "main" "--dev"; shift ;;
        --release) set_branch "release" "--release"; shift ;;
        --extras) vibe_basis_require_value "$1" "${2-}" "profile"; PROFILE_INPUT="$2"; shift 2 ;;
        --with-vq) set_vq_choice "1" "--with-vq"; shift ;;
        --without-vq) set_vq_choice "0" "--without-vq"; shift ;;
        --python) vibe_basis_require_value "$1" "${2-}" "executable"; PYTHON_INPUT="$2"; shift 2 ;;
        --venv) vibe_basis_require_value "$1" "${2-}" "path"; VENV_INPUT="$2"; shift 2 ;;
        --recreate-venv) shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) print_help; exit 0 ;;
        *) vibe_basis_die "unknown argument '$1' (run with --help)" ;;
    esac
done

if [ "$SKIP_GIT" = "1" ] && [ -n "$BRANCH_SOURCE" ]; then
    vibe_basis_die "--skip-git conflicts with branch selector $BRANCH_SOURCE"
fi

RESOLVER_PYTHON="$(vibe_basis_command_path "${PYTHON_INPUT:-python3}")"
VENV_PATH="$(vibe_basis_resolve_venv "$RESOLVER_PYTHON" "$VENV_INPUT")"
vibe_basis_assert_removable_venv "$VENV_PATH"

PROFILE="${PROFILE_INPUT:-$(vibe_basis_profile_from_marker "$VENV_PATH")}"
WITH_VQ="${WITH_VQ_INPUT:-$(vibe_basis_vq_from_marker "$VENV_PATH")}"
EXTRAS_SPEC="$(vibe_basis_profile_spec "$PROFILE")"
if [ -n "$PYTHON_INPUT" ]; then
    PYTHON_BIN="$(vibe_basis_resolve_base_python \
        "$RESOLVER_PYTHON" "$VENV_PATH")"
else
    PYTHON_BIN="$(vibe_basis_base_python "$VENV_PATH")"
    PYTHON_BIN="$(vibe_basis_resolve_base_python "$PYTHON_BIN" "$VENV_PATH")"
fi
[ "$WITH_VQ" != "1" ] || vibe_basis_assert_vq_python "$PYTHON_BIN"

if [ "$SKIP_GIT" != "1" ]; then
    git -C "$VIBE_BASIS_REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1 || \
        vibe_basis_die "update requires a Git checkout, or use --skip-git"
    if [ -z "$BRANCH_INPUT" ]; then
        BRANCH_INPUT="$(git -C "$VIBE_BASIS_REPO_ROOT" symbolic-ref --quiet --short HEAD || true)"
        [ -n "$BRANCH_INPUT" ] || vibe_basis_die "detached checkout; select --branch, --dev, --release, or --skip-git"
    fi
fi

echo "==> vibe-basis update"
echo "    checkout:     $VIBE_BASIS_REPO_ROOT"
echo "    Git:          $([ "$SKIP_GIT" = "1" ] && echo unchanged || echo "fast-forward origin/$BRANCH_INPUT")"
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
if [ "$SKIP_GIT" != "1" ]; then
    # The common lifecycle lock serializes every Git-mutating toolset command;
    # the build lock also interoperates with direct native build entry points.
    [ -f "$VIBE_BASIS_REPO_ROOT/scripts/_build_lock.sh" ] || vibe_basis_die \
        "shared native build lock helper is missing"
    # shellcheck source=../../scripts/_build_lock.sh
    . "$VIBE_BASIS_REPO_ROOT/scripts/_build_lock.sh"
    vibeqc_acquire_build_lock
    git -C "$VIBE_BASIS_REPO_ROOT" diff --quiet && \
        git -C "$VIBE_BASIS_REPO_ROOT" diff --cached --quiet || \
        vibe_basis_die "checkout has local changes; commit or remove them before update"
    git -C "$VIBE_BASIS_REPO_ROOT" fetch origin --prune --tags
    if git -C "$VIBE_BASIS_REPO_ROOT" show-ref --verify --quiet "refs/heads/$BRANCH_INPUT"; then
        git -C "$VIBE_BASIS_REPO_ROOT" switch "$BRANCH_INPUT"
    else
        git -C "$VIBE_BASIS_REPO_ROOT" switch --track -c "$BRANCH_INPUT" "origin/$BRANCH_INPUT"
    fi
    git -C "$VIBE_BASIS_REPO_ROOT" pull --ff-only origin "$BRANCH_INPUT"
fi

# Re-enter the freshly checked-out installer so its safety protocol, packaging
# metadata, and verification logic all come from the target revision.
INSTALL_ARGS=(--force --venv "$VENV_PATH" --python "$PYTHON_BIN" --extras "$PROFILE")
[ "$WITH_VQ" = "1" ] && INSTALL_ARGS+=(--with-vq)
vibe_basis_prepare_target_lock_handoff
exec "$SCRIPT_DIR/install.sh" "${INSTALL_ARGS[@]}"
