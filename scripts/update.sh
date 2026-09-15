#!/bin/bash
# Update an existing vibe-qc checkout to the latest release (or to
# another branch / tag when requested).
#
# Pulls the target ref, rebuilds any native deps that changed, refreshes
# the pybind11 module and Python deps, and prints the new banner.
#
# USAGE
#     ./scripts/update.sh [OPTIONS]
#
# OPTIONS
#     --release                 Switch to and update the `release`
#                               branch, or origin's newest stable vX.Y.Z
#                               tag when that branch is absent (default).
#     --dev                     Switch to and update the `main` branch
#                               (bleeding-edge development version,
#                               banner reads X.Y.devN). Alias for
#                               `--branch main`.
#     --branch NAME             Git branch or tag to update to. Default:
#                               release. Common values: release / main /
#                               basissetdev / vX.Y.Z. Cannot be combined
#                               with --release / --dev.
#     --ref REF                 Older spelling of --branch (kept for
#                               back-compat). Same semantics.
#     --extras GROUP            Pip extras group: test (default), dev,
#                               docs, viewer, viewer-gpu, basisopt, ase,
#                               dispersion, mpi, dispersion,mpi, mace, all,
#                               none. `all`
#                               installs dev + docs + viewer; `mace` adds
#                               the heavy, Python <=3.13-only MACE MLIP
#                               stack (deliberately not in `all`); `none`
#                               skips the [...] suffix entirely.
#     --venv PATH               Explicit venv path. Default: auto-detect
#                               (.venv / venv / .venv-vibeqc /
#                               venv-vibeqc, first hit wins). Useful for
#                               side-by-side per-branch venvs.
#     --python BIN              Interpreter for --recreate-venv. Without
#                               this flag, update.sh preserves the existing
#                               venv's recorded base interpreter.
#     --with-openblas           Build or retain vendored OpenBLAS. A prior
#                               vendored install is preserved automatically
#                               during full native rebuilds.
#     --rebuild-native-deps     Remove every pinned native source/build/install
#                               tree before setup_native_deps.sh. Normally the
#                               build stamp detects version, recipe, or artifact
#                               drift and rebuilds only the affected dependency;
#                               this flag forces a complete native rebuild.
#     --libint-max-am SPEC      libint's max angular momentum per
#                               derivative order, as N_N_N for orders
#                               0/1/2. Each part 1..6. Default 5_4_4
#                               (g-function analytic Hessians, ~1.5-2 h);
#                               5_4_3 is the pre-2026-08-06 ~30 min build
#                               without them; 6_5_4 adds i-function
#                               energies for cc-pV6Z-class work.
#                               Changing it registers as native-dep
#                               drift, so libint (and only libint)
#                               rebuilds on the next update, in any
#                               direction. See scripts/_libint_max_am.sh.
#     --recreate-venv           Failure-atomically replace the venv before
#                               reinstalling, with rollback on failure. Use this
#                               when the venv is in a broken hybrid
#                               state, or after a Python interpreter
#                               upgrade.
#     --adopt-legacy            With --recreate-venv, permit an unmarked legacy
#                               venv only when trusted PEP 610 metadata proves
#                               it was installed from this exact checkout.
#     --clean                   Equivalent to --rebuild-native-deps plus
#                               --recreate-venv, and also removes Python build
#                               caches. The nuclear option: takes 15-40 min.
#     --dry-run                 Print what update.sh *would* do (target
#                               commit, which build_*.sh files would
#                               re-run, venv status) and exit without
#                               touching anything. Safe to run at any
#                               time.
#     -h, --help                Show this help.
#
# EXAMPLES
#     ./scripts/update.sh                       # → latest tagged release
#     ./scripts/update.sh --release             # → same, explicit
#     ./scripts/update.sh --dev                 # → bleeding-edge main
#     ./scripts/update.sh --branch basissetdev  # → basissetdev branch
#     ./scripts/update.sh --branch v0.4.2       # → pin to specific tag
#     ./scripts/update.sh --rebuild-native-deps # → force native rebuild
#     ./scripts/update.sh --libint-max-am 5_4_3 # → rebuild libint smaller/faster
#     ./scripts/update.sh --recreate-venv       # → fix a broken venv
#     ./scripts/update.sh --recreate-venv --python python3.13
#     ./scripts/update.sh --with-openblas        # → add/retain vendored OpenBLAS
#     ./scripts/update.sh --clean               # → nuke everything, rebuild from source
#     ./scripts/update.sh --dry-run             # → preview without changing anything
#     ./scripts/update.sh --dev --rebuild-native-deps   # combine
#
# DUAL CHECKOUT WORKFLOW
#     If you keep two checkouts side-by-side — one for the public
#     release, one for dev — run update.sh inside the release tree
#     (no flag, or --release) and update.sh --dev inside the dev
#     tree. The lighter-weight zsh ``vibe-update`` helper in
#     docs/updating.md knows about both trees and updates them in
#     one shot. For a third experimental tree (e.g. basissetdev),
#     `--branch basissetdev` keeps it on the right ref.
#
# This is a wrapper around `git fetch + setup_native_deps.sh +
# pip install`. If your setup is unusual (different venv path, custom
# CMake flags, partial checkout), run those steps by hand — see
# docs/updating.md for the manual equivalent.

set -euo pipefail

# ---------------------------------------------------------------------------
# Source the shared install/update helpers (set_branch + extras mapping
# + branch-checkout dispatch). Same file is sourced by install.sh.
# ---------------------------------------------------------------------------
_UPDATE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_helpers.sh
. "$_UPDATE_SCRIPT_DIR/_setup_helpers.sh"
# --libint-max-am validation + export (shared with install.sh /
# update_native_deps.sh / build_libint.sh).
# shellcheck source=_libint_max_am.sh
. "$_UPDATE_SCRIPT_DIR/_libint_max_am.sh"

_UPDATE_BRANCH_SUGGESTION='Pick one of --release / --dev / --branch NAME / --ref NAME.'

# ---------------------------------------------------------------------------
# Argv parsing — before sourcing _safe_build_env.sh so --help and
# --dry-run don't trigger nice/ionice re-exec on Linux.
# ---------------------------------------------------------------------------

# Capture the original argv BEFORE the parser shifts each flag out.
# _safe_build_env.sh's nice/ionice re-exec needs the original argv to
# survive — without this it re-execs ``bash scripts/update.sh`` with
# an empty $@, which then defaults to --release (the line ~179
# fallback) and silently overrides whatever --dev / --branch / --tag
# the caller actually passed. Surfaced 2026-05-25 when a `vq admin
# update vibeqc-dev compute-host-a` (configured update_script="scripts/update.sh
# --dev") repeatedly switched compute-host-a's vibeqc-dev clone to the release
# branch + left the working tree dirty with basis-file modifications.
_VIBEQC_UPDATE_ORIG_ARGS=("$@")

BRANCH=""
BRANCH_SOURCE=""
EXTRAS_GROUP="test"
EXPLICIT_VENV=""
PYTHON_BIN=""
WITH_OPENBLAS=false
REBUILD_NATIVE_DEPS=false
RECREATE_VENV=false
ADOPT_LEGACY=false
CLEAN=false
DRY_RUN=false

print_help() {
    # Walks from `# USAGE` until the first non-comment line. No magic
    # sentinel — survives header-comment edits that would have silently
    # broken the older awk-range approach.
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
        --release)
            vibeqc_set_branch "release" "--release" "$_UPDATE_BRANCH_SUGGESTION"
            shift
            ;;
        --dev)
            vibeqc_set_branch "main" "--dev" "$_UPDATE_BRANCH_SUGGESTION"
            shift
            ;;
        --branch)
            require_value "$1" "branch or tag" "${2-}"
            vibeqc_set_branch "$2" "--branch $2" "$_UPDATE_BRANCH_SUGGESTION"
            shift 2
            ;;
        --ref)
            require_value "$1" "Git ref" "${2-}"
            vibeqc_set_branch "$2" "--ref $2" "$_UPDATE_BRANCH_SUGGESTION"
            shift 2
            ;;
        --extras)
            require_value "$1" "profile" "${2-}"
            EXTRAS_GROUP="$2"
            shift 2
            ;;
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
        --libint-max-am)
            require_value "$1" "max_am spec" "${2-}"
            vibeqc_libint_max_am_accept "$2" || exit 1
            shift 2
            ;;
        --with-openblas)
            WITH_OPENBLAS=true
            shift
            ;;
        --rebuild-native-deps)
            REBUILD_NATIVE_DEPS=true
            shift
            ;;
        --recreate-venv)
            RECREATE_VENV=true
            shift
            ;;
        --adopt-legacy)
            ADOPT_LEGACY=true
            shift
            ;;
        --clean)
            CLEAN=true
            REBUILD_NATIVE_DEPS=true
            RECREATE_VENV=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1'." >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

if [ -n "$PYTHON_BIN" ] && ! $RECREATE_VENV; then
    echo "Error: --python is only used with --recreate-venv (or --clean)." >&2
    exit 1
fi
if $ADOPT_LEGACY && ! $RECREATE_VENV; then
    echo "Error: --adopt-legacy requires --recreate-venv (or --clean)." >&2
    exit 1
fi

# Default branch when no flag was given.
[ -z "$BRANCH" ] && BRANCH="release"
RELEASE_TAG_FALLBACK=0
if [ "$BRANCH" = release ] && { [ -z "$BRANCH_SOURCE" ] || [ "$BRANCH_SOURCE" = --release ]; }; then
    RELEASE_TAG_FALLBACK=1
fi

vibeqc_extras_to_pip_spec "$EXTRAS_GROUP" EXTRAS_SPEC

# ---------------------------------------------------------------------------
# SCRIPT_DIR + helper sourcing. --dry-run skips the build-pressure
# helper so we don't accidentally re-exec under nice/ionice for what is
# meant to be a quick read-only preview.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$_UPDATE_SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$DRY_RUN" = "false" ]; then
    # Suppress the helper's minimal cmake+ninja check — setup_native_deps.sh
    # (invoked below) runs the comprehensive preflight.
    export VIBEQC_TOOLING_CHECKED=1
    # shellcheck source=_safe_build_env.sh
    . "$SCRIPT_DIR/_safe_build_env.sh"
fi

# shellcheck source=_venv_helpers.sh
. "$SCRIPT_DIR/_venv_helpers.sh"
# shellcheck source=_native_stamp.sh
. "$SCRIPT_DIR/_native_stamp.sh"
# shellcheck source=_vq_cooperation.sh
. "$SCRIPT_DIR/_vq_cooperation.sh"
# shellcheck source=_build_lock.sh
. "$SCRIPT_DIR/_build_lock.sh"

cd "$REPO_ROOT"

echo "==> vibe-qc update"
echo "    repo:         $REPO_ROOT"
echo "    target:       $BRANCH"
echo "    extras:       $EXTRAS_GROUP  (pip suffix: '${EXTRAS_SPEC:-<none>}')"
[ -z "$VIBEQC_COLOCATED_EXTRA" ] || \
    echo "    companion:    local $VIBEQC_COLOCATED_EXTRA project"
[ -n "$EXPLICIT_VENV" ]      && echo "    venv:         $EXPLICIT_VENV (explicit)"
[ -n "$PYTHON_BIN" ]         && echo "    python:       $PYTHON_BIN (explicit replacement interpreter)"
$WITH_OPENBLAS                && echo "    --with-openblas: vendored OpenBLAS will be built/retained"
$REBUILD_NATIVE_DEPS         && echo "    --rebuild-native-deps: third_party/*/install will be wiped first"
$RECREATE_VENV               && echo "    --recreate-venv: venv will be replaced with rollback protection"
$ADOPT_LEGACY                && echo "    --adopt-legacy: trusted PEP 610 ownership proof required"
$CLEAN                       && echo "    --clean: also wiping Python build caches"
$DRY_RUN                     && echo "    --dry-run: no files will be changed"
echo

# Discover and validate the environment before Git or native-dependency
# mutation. --recreate-venv must see broken-but-recognisable environments;
# an ordinary update requires a runnable interpreter.
VENV=""
if $RECREATE_VENV; then
    vibeqc_detect_venv_target VENV "$EXPLICIT_VENV"
else
    vibeqc_detect_venv VENV "$EXPLICIT_VENV"
fi
if [ -n "$VENV" ]; then
    VENV_REQUESTED="$VENV"
    vibeqc_resolve_venv_path VENV "$VENV_REQUESTED"
fi
ADOPT_LEGACY_VALUE=0
$ADOPT_LEGACY && ADOPT_LEGACY_VALUE=1
if $RECREATE_VENV && [ -n "$VENV" ]; then
    INSPECTION_REQUESTED="${PYTHON_BIN:-python3}"
    vibeqc_select_trusted_python \
        INSPECTION_PYTHON "$INSPECTION_REQUESTED" "$VENV"
    vibeqc_assert_owned_venv \
        "$VENV" "$REPO_ROOT" "$INSPECTION_PYTHON" "$ADOPT_LEGACY_VALUE"
    if [ -z "$PYTHON_BIN" ]; then
        vibeqc_venv_base_python PYTHON_BIN "$VENV" || true
    fi
elif [ -n "$VENV" ]; then
    # An activated selected venv may put its own python3 first on PATH. Find a
    # separate isolated interpreter for locking without consulting pyvenv.cfg;
    # the in-place update executes the installed environment only after locks.
    vibe_toolset_find_external_python INSPECTION_PYTHON "$VENV"
fi

# ---------------------------------------------------------------------------
# Dry-run path: peek at the target SHA, compare build-stamp, detect
# venv state — then exit before touching anything.
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" = "true" ]; then
    echo "==> Dry-run preview (no changes will be made)"
    echo

    # Working-tree state.
    DRY_RUN_WORKTREE_STATE=0
    vibeqc_worktree_state || DRY_RUN_WORKTREE_STATE=$?
    if [ "$DRY_RUN_WORKTREE_STATE" -eq 2 ]; then
        echo "    [git] git could not read this repository — update.sh would refuse."
    elif [ "$DRY_RUN_WORKTREE_STATE" -eq 1 ]; then
        echo "    [git] working tree is DIRTY — update.sh would refuse to proceed."
    else
        echo "    [git] working tree is clean."
    fi

    # Target ref availability + would-pull commit.
    echo
    echo "    [git] target branch/tag: $BRANCH"
    if [ "$RELEASE_TAG_FALLBACK" = 1 ]; then
        echo "          (if origin has no release branch, apply selects its newest stable vX.Y.Z tag)"
    fi
    echo "          (read-only preview uses currently cached refs; apply will fetch)"
    if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
        target_sha=$(git rev-parse --short "origin/$BRANCH" 2>/dev/null || git rev-parse --short "$BRANCH")
        echo "          → would fast-forward local '$BRANCH' to $target_sha"
    elif git show-ref --verify --quiet "refs/tags/$BRANCH"; then
        target_sha=$(git rev-parse --short "$BRANCH")
        echo "          → would check out tag '$BRANCH' (detached HEAD) at $target_sha"
    elif git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
        target_sha=$(git rev-parse --short "origin/$BRANCH")
        echo "          → would create local '$BRANCH' tracking origin/$BRANCH at $target_sha"
    else
        echo "          → not present in cached refs; apply will fetch and validate it."
    fi

    # Native-dep stamp drift.
    echo
    if vibeqc_stamp_check; then
        :
    fi

    # Venv presence + health.
    echo
    if [ -z "$VENV" ]; then
        echo "    [venv] none found at the default paths or at '$EXPLICIT_VENV'."
        echo "           update.sh would REFUSE before changing Git or native deps;"
        echo "           use ./scripts/install.sh for a fresh setup."
    else
        echo "    [venv] using $VENV"
        if $RECREATE_VENV; then
            echo "           → checkout-owned virtualenv; would replace atomically"
            if [ -n "$PYTHON_BIN" ]; then
                echo "           → replacement interpreter: $PYTHON_BIN"
            else
                echo "           → base interpreter unavailable; --python would be required"
            fi
        elif vibeqc_check_venv_health "$VENV" 2>/dev/null; then
            echo "           → health OK"
        else
            echo "           → health check FAILED (re-run with --recreate-venv to fix)"
        fi
    fi

    echo
    echo "==> Dry-run complete. Re-run without --dry-run to apply."
    exit 0
fi

if [ -z "$VENV" ]; then
    echo >&2
    if [ -n "$EXPLICIT_VENV" ]; then
        echo "Error: no usable virtualenv at '$EXPLICIT_VENV'. Nothing was changed." >&2
    else
        echo "Error: no usable virtualenv found at .venv/ (or venv/, .venv-vibeqc/," >&2
        echo "venv-vibeqc/). Nothing was changed." >&2
    fi
    echo "Create one with ./scripts/install.sh, then re-run update.sh." >&2
    exit 1
fi

vibeqc_assert_safe_venv_target "$VENV" "$REPO_ROOT"
vibeqc_acquire_lifecycle_lock \
    "$VENV" "$REPO_ROOT" update "${INSPECTION_PYTHON:-python3}"
trap 'vibeqc_lifecycle_cleanup' EXIT
if $RECREATE_VENV; then
    if [ ! -e "$VENV" ] && [ ! -L "$VENV" ]; then
        echo "Error: virtualenv target changed before replacement could begin:" >&2
        echo "       $VENV" >&2
        exit 1
    fi
    vibeqc_assert_owned_venv \
        "$VENV" "$REPO_ROOT" "$INSPECTION_PYTHON" "$ADOPT_LEGACY_VALUE"
    if [ -z "$PYTHON_BIN" ]; then
        echo "Error: could not determine the existing venv's base interpreter." >&2
        echo "Re-run with --recreate-venv --python /path/to/python3." >&2
        exit 1
    fi
    vibeqc_resolve_base_python PYTHON_BIN "$PYTHON_BIN"
    vibeqc_begin_venv_replacement "$VENV" "$REPO_ROOT" 1
else
    vibeqc_check_venv_health "$VENV"
fi

# ---------------------------------------------------------------------------
# Serialize against any other native build on this checkout. Acquired after
# the --dry-run early-exit and before the checkout + native-deps build, so a
# second concurrent update.sh / install.sh / setup_native_deps.sh bails with a
# clear message instead of corrupting the shared third_party/*/build trees
# (the 2026-06-05 compute-host-b collision). Re-entrant with the setup_native_deps.sh
# invoked below. Rationale + edge cases in scripts/_build_lock.sh.
# ---------------------------------------------------------------------------
vibeqc_acquire_build_lock

# ---------------------------------------------------------------------------
# Refuse if working tree has uncommitted changes.
# ---------------------------------------------------------------------------
WORKTREE_STATE=0
vibeqc_worktree_state || WORKTREE_STATE=$?
if [ "$WORKTREE_STATE" -eq 2 ]; then
    echo
    echo "Error: git could not read this repository ($PWD)." >&2
    echo "Run update.sh from inside a vibe-qc checkout." >&2
    echo "    git status -s" >&2
    exit 1
fi
if [ "$WORKTREE_STATE" -eq 1 ]; then
    echo
    echo "Error: working tree has uncommitted changes." >&2
    echo "Commit, stash, or revert them before running update.sh." >&2
    echo "    git status -s" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Fetch + check out the requested branch via the shared helper.
# ---------------------------------------------------------------------------
if ! vibeqc_checkout_ref "$BRANCH" "$RELEASE_TAG_FALLBACK"; then
    echo "Try: --release, --dev, --branch <name>, --branch vX.Y.Z, or --ref <commit>" >&2
    exit 1
fi

vibeqc_assert_colocated_extra_available \
    "$REPO_ROOT" "$VIBEQC_COLOCATED_EXTRA"

# ---------------------------------------------------------------------------
# Build-stamp drift check before deciding whether to keep the existing
# install/ trees. If --rebuild-native-deps wasn't passed but drift is
# detected, alert the user; they can re-run with the flag. We don't
# auto-force the rebuild — wiping third_party/*/install is a 15–40 min
# action and the user should consent.
# ---------------------------------------------------------------------------
if ! $REBUILD_NATIVE_DEPS; then
    echo
    # stamp_check returns 0=ok, 1=drift, 2=stamp missing (legacy install).
    # `|| stamp_rc=$?` keeps set -e happy while letting us branch on rc.
    stamp_rc=0
    vibeqc_stamp_check || stamp_rc=$?
    if [ "$stamp_rc" -eq 1 ]; then
        echo
        echo "Drift detected — auto-rebuilding the changed native dep(s)"
        echo "(targeted: only the dep whose build_*.sh changed gets rebuilt."
        echo "Pass --rebuild-native-deps to force a full rebuild of every dep)."
        "$SCRIPT_DIR/update_native_deps.sh"
    elif [ "$stamp_rc" -eq 2 ]; then
        echo
        echo "Legacy install (no build stamp) — drift can't be auto-detected."
        echo "If the post-install banner shows wrong library versions, re-run with:"
        echo "    ./scripts/update.sh --branch $BRANCH --rebuild-native-deps"
        echo "which will rebuild and write a fresh stamp for next time."
    fi
fi

# ---------------------------------------------------------------------------
# Optional: nuke vendored installs (and the build/ cache, on --clean).
# ---------------------------------------------------------------------------
HAD_VENDORED_OPENBLAS=false
if [ -d third_party/openblas/install ] || \
   { [ -f "$VIBEQC_STAMP_PATH" ] && \
     awk '$1 == "openblas" { found=1 } END { exit !found }' \
         "$VIBEQC_STAMP_PATH"; }; then
    HAD_VENDORED_OPENBLAS=true
fi
if $HAD_VENDORED_OPENBLAS; then
    # Once selected, vendored OpenBLAS is part of this checkout's native
    # configuration. A full rebuild must not silently fall back to system BLAS.
    WITH_OPENBLAS=true
fi

if $REBUILD_NATIVE_DEPS; then
    echo
    echo "==> Removing pinned native source/build/install trees..."
    rm -rf third_party/libint/src third_party/libint/build third_party/libint/install \
           third_party/libxc/src third_party/libxc/build third_party/libxc/install \
           third_party/spglib/src third_party/spglib/build third_party/spglib/install \
           third_party/fftw/src third_party/fftw/build third_party/fftw/install \
           third_party/libecpint \
           third_party/openblas/src third_party/openblas/build third_party/openblas/install
    rm -f "$VIBEQC_STAMP_PATH"
fi

if $CLEAN; then
    echo "==> Removing Python build caches..."
    rm -rf build/ _skbuild/
fi

# ---------------------------------------------------------------------------
# Cooperative vq pause: if vq jobs are running on the same host, pause
# them so the upcoming cmake+ninja+pip-install build doesn't fight
# them for RAM. The trap guarantees resume even on build failure or
# Ctrl-C. No-op if vq isn't installed / daemon is down / no jobs.
# Rationale + edge cases in scripts/_vq_cooperation.sh.
# ---------------------------------------------------------------------------
vibeqc_vq_install_trap
vibeqc_vq_pause

# ---------------------------------------------------------------------------
# Re-run the native-deps orchestrator. Idempotent on existence — only
# the deps with no install/ tree get rebuilt. After a vanilla `git
# pull` this is usually a fast no-op.
# ---------------------------------------------------------------------------
echo
echo "==> Rebuilding native deps (libint, libxc, spglib, fftw, libecpint)..."
if $WITH_OPENBLAS; then
    WITH_OPENBLAS=1 ./scripts/setup_native_deps.sh
else
    ./scripts/setup_native_deps.sh
fi

# ---------------------------------------------------------------------------
# Failure-atomic replacement starts only after Git/native work succeeds. The
# transaction was armed before mutation, so its EXIT trap restores the prior
# venv after any creation, pip, build, or banner failure.
# ---------------------------------------------------------------------------
if $RECREATE_VENV; then
    echo
    echo "==> --recreate-venv: building a verified replacement for $VENV..."
    vibeqc_assert_owned_venv \
        "$VENV" "$REPO_ROOT" "$PYTHON_BIN" "$ADOPT_LEGACY_VALUE"
    vibeqc_start_venv_replacement \
        "$VENV" "$REPO_ROOT" "$PYTHON_BIN" "$ADOPT_LEGACY_VALUE"
    vibeqc_create_venv "$VENV" "$PYTHON_BIN"
fi

# ---------------------------------------------------------------------------
# Upgrade pip + reinstall the editable package.
# ---------------------------------------------------------------------------
echo
vibeqc_pip_upgrade "$VENV"

echo
vibeqc_pip_install_editable "$VENV" "$EXTRAS_SPEC" "."
vibeqc_install_colocated_extra "$VENV" "$REPO_ROOT" "$VIBEQC_COLOCATED_EXTRA"

# ---------------------------------------------------------------------------
# Banner. The banner is also our drift-detector for the *post-install*
# state: if the banner's `linked: libint X.Y.Z` line disagrees with
# the build_*.sh pins, the stale-install footgun bit and the user
# needs --rebuild-native-deps.
# ---------------------------------------------------------------------------
echo
echo "==> Verifying:"
vibeqc_verify_install "$VENV/bin/python" "$EXTRAS_GROUP"
if $RECREATE_VENV; then
    vibeqc_mark_owned_venv "$VENV" "$REPO_ROOT" "$PYTHON_BIN"
    vibeqc_commit_venv_replacement
fi
vibeqc_release_lifecycle_lock

echo
echo "==> Update complete."
