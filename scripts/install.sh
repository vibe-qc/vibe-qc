#!/bin/bash
# Fresh-install wrapper for vibe-qc.
#
# Drives the full bootstrap: optional branch checkout → vendored native
# deps → venv → editable pip install → banner. Idempotent on the
# native-deps step (each build_*.sh short-circuits if its install/
# tree exists), so re-running this is cheap.
#
# USAGE
#     ./scripts/install.sh [OPTIONS]
#
# OPTIONS
#     --release                 Check out and install from the `release`
#                               branch, or origin's newest stable vX.Y.Z
#                               tag when that branch is absent (default).
#     --dev                     Check out and install from the `main`
#                               branch (bleeding-edge dev; banner reads
#                               X.Y.devN). Alias for --branch main.
#     --current                 Install the currently checked-out source
#                               without fetching or changing Git refs.
#     --branch NAME             Any other branch or tag — e.g.
#                               `--branch basissetdev`, `--branch v0.7.5`.
#                               Cannot be combined with --release / --dev /
#                               --current.
#     --extras GROUP            Pip extras group: test (default), dev,
#                               docs, viewer, viewer-gpu, basisopt, ase,
#                               dispersion, mpi, dispersion,mpi, mace, all,
#                               none. `all`
#                               installs dev + docs + viewer; `mace` adds
#                               the heavy, Python <=3.13-only MACE MLIP
#                               stack (deliberately not in `all`); `none`
#                               skips the [...] suffix.
#     --python BIN              Python interpreter to base the venv on
#                               (default: python3). Use to force a
#                               specific version, e.g. `--python python3.13`.
#     --venv PATH               Where to create the venv
#                               (default: .venv in the repo root).
#                               Useful for keeping per-branch venvs
#                               side-by-side, e.g. `--venv .venv-bsd`
#                               for a basissetdev checkout.
#     --with-openblas           Build vendored OpenBLAS (sets
#                               WITH_OPENBLAS=1 for setup_native_deps.sh).
#                               Use when system BLAS is reference netlib
#                               or absent.
#     --libint-max-am SPEC      libint's max angular momentum per
#                               derivative order, as N_N_N for orders
#                               0/1/2. Each part 1..6. Default 5_4_4.
#                                 5_4_3  ~30 min; f-function Hessians
#                                        only (the pre-2026-08-06 build)
#                                 5_4_4  ~1.5-2 h; g-function Hessians
#                                 6_5_4  longer; adds i-function
#                                        energies (cc-pV6Z class)
#                                 6_6_6  longest; the table's ceiling
#                               Only decides which basis sets each
#                               derivative order accepts, no computed
#                               number changes. Sets
#                               VIBEQC_LIBINT_MAX_AM for the build
#                               scripts; see scripts/_libint_max_am.sh.
#     --force                   Failure-atomically replace an existing,
#                               checkout-owned venv at the target path.
#     --adopt-legacy            With --force, permit an unmarked legacy venv
#                               only when trusted PEP 610 metadata proves it
#                               was installed from this exact checkout.
#     --skip-native-deps        Don't (re-)run setup_native_deps.sh.
#                               For use after a previous successful
#                               build when you only want to rebuild the
#                               venv. Bails up-front if any
#                               third_party/<dep>/install/ tree is
#                               missing (else pip install would explode
#                               later with a cryptic CMake error).
#     --dry-run                 Print what install.sh *would* do (target
#                               branch, venv path, would the venv get
#                               clobbered, native-deps state) and exit
#                               without touching anything. Safe to run
#                               at any time.
#     -h, --help                Show this help.
#
# EXAMPLES
#     ./scripts/install.sh                              # release branch, .venv/, [test]
#     ./scripts/install.sh --dev                        # main branch, bleeding edge
#     ./scripts/install.sh --current                    # do not change the checkout
#     ./scripts/install.sh --branch basissetdev         # basis-set paper-writing branch
#     ./scripts/install.sh --branch v0.7.5              # pin to a tag
#     ./scripts/install.sh --extras dev                 # tests + dispersion + py-spy
#     ./scripts/install.sh --extras viewer-gpu          # + co-located vibe-view browser
#     ./scripts/install.sh --extras basisopt             # + co-located vibe-basis
#     ./scripts/install.sh --extras mpi                 # + mpi4py
#     ./scripts/install.sh --extras dispersion,mpi      # dispersion + mpi4py
#     ./scripts/install.sh --extras mace                # + MACE MLIP stack (Python <=3.13)
#     ./scripts/install.sh --python python3.13          # force 3.13 over default python3
#     ./scripts/install.sh --venv .venv-bsd --branch basissetdev   # side-by-side
#     ./scripts/install.sh --force                      # replace existing .venv safely
#     ./scripts/install.sh --with-openblas              # vendor OpenBLAS
#     ./scripts/install.sh --libint-max-am 5_4_3        # faster libint, no g-function Hessians
#     ./scripts/install.sh --dry-run --branch basissetdev   # preview without changes
#
# WORKTREE-PRIVATE VENVS
#     If you run install.sh from a git worktree, the resolved repo root
#     is the worktree (not the main checkout), so the default .venv
#     lands inside the worktree. Each worktree gets its own venv —
#     don't reuse a sibling worktree's .venv, because the editable
#     pip install pins the C++ extension to that worktree's
#     third_party/ trees.
#
# See docs/installation.md for the manual recipe and the build
# prerequisites the preflight check enforces.

set -euo pipefail

# _safe_build_env.sh may re-exec this script after argv parsing on Linux.
# Preserve the original arguments so flags survive that re-exec.
_VIBEQC_UPDATE_ORIG_ARGS=("$@")

# ---------------------------------------------------------------------------
# Source the shared install/update helpers (set_branch + extras mapping
# + branch-checkout dispatch). Same file is sourced by update.sh.
# ---------------------------------------------------------------------------
_INSTALL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_setup_helpers.sh
. "$_INSTALL_SCRIPT_DIR/_setup_helpers.sh"
# --libint-max-am validation + export (shared with update.sh /
# update_native_deps.sh / build_libint.sh).
# shellcheck source=_libint_max_am.sh
. "$_INSTALL_SCRIPT_DIR/_libint_max_am.sh"

# ---------------------------------------------------------------------------
# Argv parsing — defer the heavier _safe_build_env sourcing until after,
# so --help and --skip-native-deps don't trigger nice/ionice re-exec.
# ---------------------------------------------------------------------------

BRANCH=""
BRANCH_SOURCE=""
EXTRAS_GROUP="test"
PYTHON_BIN="python3"
VENV_PATH=".venv"
WITH_OPENBLAS=0
FORCE=0
ADOPT_LEGACY=0
SKIP_NATIVE_DEPS=0
DRY_RUN=0

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
        --release)            vibeqc_set_branch "release" "--release"; shift ;;
        --dev)                vibeqc_set_branch "main"    "--dev";     shift ;;
        --current)            vibeqc_set_branch "__current__" "--current"; shift ;;
        --branch)
            require_value "$1" "branch or tag" "${2-}"
            vibeqc_set_branch "$2" "--branch $2"
            shift 2
            ;;
        --extras)
            require_value "$1" "profile" "${2-}"
            EXTRAS_GROUP="$2"
            shift 2
            ;;
        --python)
            require_value "$1" "executable" "${2-}"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --venv)
            require_value "$1" "path" "${2-}"
            VENV_PATH="$2"
            shift 2
            ;;
        --with-openblas)      WITH_OPENBLAS=1;        shift ;;
        --libint-max-am)
            require_value "$1" "max_am spec" "${2-}"
            vibeqc_libint_max_am_accept "$2" || exit 1
            shift 2
            ;;
        --force)              FORCE=1;                shift ;;
        --adopt-legacy)       ADOPT_LEGACY=1;         shift ;;
        --skip-native-deps)   SKIP_NATIVE_DEPS=1;     shift ;;
        --dry-run)            DRY_RUN=1;              shift ;;
        -h|--help)            print_help; exit 0 ;;
        *)
            echo "Error: unknown argument '$1'." >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# Bare install means the public release. --current is the explicit escape
# hatch for contributors and local source snapshots.
if [ -z "$BRANCH" ]; then
    BRANCH="release"
elif [ "$BRANCH" = "__current__" ]; then
    BRANCH=""
fi
RELEASE_TAG_FALLBACK=0
if [ "$BRANCH" = release ] && { [ -z "$BRANCH_SOURCE" ] || [ "$BRANCH_SOURCE" = --release ]; }; then
    RELEASE_TAG_FALLBACK=1
fi

vibeqc_extras_to_pip_spec "$EXTRAS_GROUP" EXTRAS_SPEC

# ---------------------------------------------------------------------------
# Set up SCRIPT_DIR + REPO_ROOT, source build-pressure helper (re-execs
# under nice/ionice on Linux), source the venv helpers.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$_INSTALL_SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Suppress the helper's minimal cmake+ninja probe — setup_native_deps.sh
# runs a much more comprehensive preflight and we want its consolidated
# report to be the one the user sees.
export VIBEQC_TOOLING_CHECKED=1
# shellcheck source=_safe_build_env.sh
. "$SCRIPT_DIR/_safe_build_env.sh"

# shellcheck source=_venv_helpers.sh
. "$SCRIPT_DIR/_venv_helpers.sh"
# shellcheck source=_vq_cooperation.sh
. "$SCRIPT_DIR/_vq_cooperation.sh"
# shellcheck source=_build_lock.sh
. "$SCRIPT_DIR/_build_lock.sh"

cd "$REPO_ROOT"

VENV_REQUESTED="$VENV_PATH"
vibeqc_resolve_venv_path VENV_PATH "$VENV_REQUESTED"
vibeqc_assert_safe_venv_target "$VENV_PATH" "$REPO_ROOT"
VENV_HAD_ORIGINAL=0
if [ -e "$VENV_PATH" ] || [ -L "$VENV_PATH" ]; then
    VENV_HAD_ORIGINAL=1
fi
if [ "$ADOPT_LEGACY" = "1" ] && [ "$FORCE" != "1" ]; then
    echo "Error: --adopt-legacy requires --force." >&2
    exit 1
fi
if [ "$ADOPT_LEGACY" = "1" ] && [ "$VENV_HAD_ORIGINAL" != "1" ]; then
    echo "Error: --adopt-legacy requires an existing legacy virtualenv." >&2
    exit 1
fi
if [ "$VENV_HAD_ORIGINAL" = "1" ] && [ "$FORCE" = "1" ]; then
    vibeqc_select_trusted_python INSPECTION_PYTHON "$PYTHON_BIN" "$VENV_PATH"
    vibeqc_assert_owned_venv \
        "$VENV_PATH" "$REPO_ROOT" "$INSPECTION_PYTHON" "$ADOPT_LEGACY"
    PYTHON_BIN="$INSPECTION_PYTHON"
fi
if [ -z "$BRANCH" ]; then
    vibeqc_assert_colocated_extra_available \
        "$REPO_ROOT" "$VIBEQC_COLOCATED_EXTRA"
fi

echo "==> vibe-qc install"
echo "    repo root:    $REPO_ROOT"
echo "    venv path:    $VENV_PATH"
echo "    extras:       $EXTRAS_GROUP  (pip suffix: '${EXTRAS_SPEC:-<none>}')"
[ -z "$VIBEQC_COLOCATED_EXTRA" ] || \
    echo "    companion:    local $VIBEQC_COLOCATED_EXTRA project"
echo "    python:       $PYTHON_BIN"
if [ -n "$BRANCH" ]; then
    echo "    branch:       $BRANCH"
else
    echo "    branch:       <leave working tree as-is>"
fi
[ "$WITH_OPENBLAS" = "1" ]    && echo "    --with-openblas: vendored OpenBLAS will be built"
[ "$FORCE" = "1" ]            && echo "    --force: existing venv at $VENV_PATH will be replaced with rollback protection"
[ "$ADOPT_LEGACY" = "1" ]     && echo "    --adopt-legacy: trusted PEP 610 ownership proof required"
[ "$SKIP_NATIVE_DEPS" = "1" ] && echo "    --skip-native-deps: native build step will be skipped"
[ "$DRY_RUN" = "1" ]          && echo "    --dry-run: no files will be changed"
echo

# ---------------------------------------------------------------------------
# Dry-run path: peek at the target SHA, would-be venv status, and
# native-deps presence — then exit without touching anything.
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" = "1" ]; then
    echo "==> Dry-run preview (no changes will be made)"
    echo

    # Working-tree state (only matters if --branch is in play).
    if [ -n "$BRANCH" ]; then
        if git diff --quiet && git diff --cached --quiet; then
            echo "    [git] working tree is clean."
        else
            echo "    [git] working tree is DIRTY — install.sh would refuse the checkout."
        fi
        echo
        echo "    [git] target branch/tag: $BRANCH"
        if [ "$RELEASE_TAG_FALLBACK" = 1 ]; then
            echo "          (if origin has no release branch, apply selects its newest stable vX.Y.Z tag)"
        fi
        echo "          (read-only preview uses currently cached refs; apply will fetch)"
        target_sha=""
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
    else
        echo "    [git] --current selected — would install whatever is currently checked out."
        head_sha=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
        head_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
        echo "          (current: $head_branch @ $head_sha)"
    fi

    # Venv state.
    echo
    if [ -n "$VIBEQC_COLOCATED_EXTRA" ]; then
        echo "    [companion] would install local $VIBEQC_COLOCATED_EXTRA from this checkout."
        echo
    fi
    if [ -e "$VENV_PATH" ]; then
        if [ "$FORCE" = "1" ]; then
            echo "    [venv] $VENV_PATH is owned by this checkout and would be replaced atomically (--force)."
        else
            echo "    [venv] $VENV_PATH exists — install.sh would REFUSE without --force."
        fi
    else
        echo "    [venv] $VENV_PATH does not exist — would be created with $PYTHON_BIN."
    fi

    # Native-deps state.
    echo
    if [ "$SKIP_NATIVE_DEPS" = "1" ]; then
        missing_dry=()
        for dep in libint libxc spglib fftw libecpint; do
            if [ ! -d "third_party/$dep/install" ]; then
                missing_dry+=("$dep")
            fi
        done
        if [ ${#missing_dry[@]} -gt 0 ]; then
            echo "    [native deps] --skip-native-deps set but missing: ${missing_dry[*]}"
            echo "                  install.sh would refuse to proceed."
        else
            echo "    [native deps] --skip-native-deps set; all install/ trees already present."
        fi
    else
        present=()
        absent=()
        for dep in libint libxc spglib fftw libecpint; do
            if [ -d "third_party/$dep/install" ]; then
                present+=("$dep")
            else
                absent+=("$dep")
            fi
        done
        # libint's max_am is not visible from the install/ tree's existence
        # alone, so a "no-op" claim has to check the marker too: a different
        # --libint-max-am means libint gets rebuilt.
        am_want="$(vibeqc_libint_max_am_spec)"
        am_have="$(vibeqc_libint_installed_spec)"
        am_rebuild=0
        if [ -d "third_party/libint/install" ] \
           && [ "$am_have" != "$am_want" ] \
           && ! { [ "$am_have" = "unknown" ] && [ "$am_want" = "$VIBEQC_LIBINT_MAX_AM_DEFAULT" ]; }; then
            am_rebuild=1
        fi

        if [ ${#absent[@]} -eq 0 ] && [ "$am_rebuild" = "0" ]; then
            echo "    [native deps] all install/ trees present; setup_native_deps.sh would be a no-op."
        elif [ ${#present[@]} -eq 0 ]; then
            echo "    [native deps] none present — would build all five (15–40 min)."
        else
            echo "    [native deps] partial: present=${present[*]} | missing=${absent[*]}"
            echo "                  setup_native_deps.sh would build the missing ones."
        fi
        if [ "$am_rebuild" = "1" ]; then
            echo "    [libint]      max_am $am_have installed, $am_want requested — would rebuild libint."
        else
            echo "    [libint]      max_am $am_want"
        fi
    fi

    echo
    echo "==> Dry-run complete. Re-run without --dry-run to apply."
    exit 0
fi

# Validate every destructive/input prerequisite before checkout or native
# build work. A bad interpreter or unsafe/non-venv target must leave Git and
# third_party untouched.
vibeqc_acquire_lifecycle_lock "$VENV_PATH" "$REPO_ROOT" install "$PYTHON_BIN"
trap 'vibeqc_lifecycle_cleanup' EXIT
vibeqc_resolve_base_python PYTHON_BIN "$PYTHON_BIN"
VENV_REPLACEMENT_EXPECTED_ORIGINAL=0
if [ -e "$VENV_PATH" ] || [ -L "$VENV_PATH" ]; then
    VENV_REPLACEMENT_EXPECTED_ORIGINAL=1
    if [ "$FORCE" != "1" ]; then
        cat >&2 <<EOF
Error: '$VENV_PATH' already exists.

To preserve it and update in place, use:
    ./scripts/update.sh${BRANCH:+ --branch $BRANCH}${EXTRAS_GROUP:+ --extras $EXTRAS_GROUP}

To replace a recognisable virtualenv with rollback protection, re-run with:
    ./scripts/install.sh --force ...

Or pick a different venv path:
    ./scripts/install.sh --venv .venv-${BRANCH:-other} ...
EOF
        exit 1
    fi
    vibeqc_assert_owned_venv \
        "$VENV_PATH" "$REPO_ROOT" "$PYTHON_BIN" "$ADOPT_LEGACY"
fi

vibeqc_begin_venv_replacement \
    "$VENV_PATH" "$REPO_ROOT" "$VENV_REPLACEMENT_EXPECTED_ORIGINAL"

# ---------------------------------------------------------------------------
# Serialize against any other native build on this checkout. Acquired here,
# after the --dry-run early-exit (a dry run must take no lock) and before the
# checkout + native-deps build, so a second concurrent install.sh / update.sh
# / setup_native_deps.sh bails with a clear message instead of corrupting the
# shared third_party/*/build trees (the 2026-06-05 compute-host-b collision). Re-entrant:
# the setup_native_deps.sh this script invokes sees the lock already held.
# Rationale + edge cases in scripts/_build_lock.sh.
# ---------------------------------------------------------------------------
vibeqc_acquire_build_lock

# ---------------------------------------------------------------------------
# 1. Optional branch checkout.
#
# Refuse if working tree is dirty (silently stashing user work is worse
# than failing loud). Dispatch to vibeqc_checkout_ref for the three-way
# local-branch / tag / remote-only switch — same helper update.sh uses.
# ---------------------------------------------------------------------------
if [ -n "$BRANCH" ]; then
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Error: working tree has uncommitted changes." >&2
        echo "Commit, stash, or revert them before checking out a different branch:" >&2
        echo "    git status -s" >&2
        exit 1
    fi

    if ! vibeqc_checkout_ref "$BRANCH" "$RELEASE_TAG_FALLBACK"; then
        echo "Try: --release, --dev, or --branch <known-branch-or-tag>" >&2
        exit 1
    fi
    echo
fi

# The selected ref may have a different companion layout from the source tree
# inspected before checkout. Recheck now, before native builds or venv changes.
vibeqc_assert_colocated_extra_available \
    "$REPO_ROOT" "$VIBEQC_COLOCATED_EXTRA"

# ---------------------------------------------------------------------------
# Cooperative vq pause: if vq jobs are running on the same host, pause
# them so the upcoming cmake+ninja+pip-install build doesn't fight
# them for RAM. The trap guarantees resume even on build failure or
# Ctrl-C. Fires here (before --skip-native-deps gating) because pip
# install -e . in step 4 also runs a heavy CMake build of the
# pybind11 extension, so the bracket needs to cover the full install
# regardless of whether native-deps got skipped.
# Rationale + edge cases in scripts/_vq_cooperation.sh.
# ---------------------------------------------------------------------------
vibeqc_vq_install_trap
vibeqc_vq_pause

# ---------------------------------------------------------------------------
# 2. Native deps (idempotent; runs the full preflight).
#
# --skip-native-deps is only safe when every install/ tree already
# exists; otherwise the later `pip install -e .` blows up with a
# cryptic "Cannot find Libint2 / Libxc / Spglib / FFTW3 / libecpint"
# error from CMake. Guard upfront with a per-dep check so the user
# gets an actionable message instead.
# ---------------------------------------------------------------------------
if [ "$SKIP_NATIVE_DEPS" = "1" ]; then
    missing_deps=()
    for dep in libint libxc spglib fftw libecpint; do
        if [ ! -d "third_party/$dep/install" ]; then
            missing_deps+=("$dep")
        fi
    done
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo "Error: --skip-native-deps was passed but third_party/<dep>/install/ is missing for:" >&2
        for d in "${missing_deps[@]}"; do
            echo "    - $d" >&2
        done
        echo >&2
        echo "Either run without --skip-native-deps (will build them now)," >&2
        echo "or build them up-front:" >&2
        echo "    ./scripts/setup_native_deps.sh" >&2
        exit 1
    fi
    echo "==> Skipping setup_native_deps.sh (--skip-native-deps; all install/ trees present)"
    echo
else
    echo "==> Running setup_native_deps.sh (libint max_am $(vibeqc_libint_max_am_spec)) ..."
    if [ "$WITH_OPENBLAS" = "1" ]; then
        WITH_OPENBLAS=1 "$SCRIPT_DIR/setup_native_deps.sh"
    else
        "$SCRIPT_DIR/setup_native_deps.sh"
    fi
    echo
fi

# ---------------------------------------------------------------------------
# 3. Venv: replace at the final path, keeping the prior environment as a
# same-parent rollback until installation and verification both pass.
# ---------------------------------------------------------------------------
# The transaction repeats ownership proof immediately before moving an
# existing target and refuses any existence change since begin.
vibeqc_start_venv_replacement \
    "$VENV_PATH" "$REPO_ROOT" "$PYTHON_BIN" "$ADOPT_LEGACY"
vibeqc_create_venv "$VENV_PATH" "$PYTHON_BIN"
echo

# ---------------------------------------------------------------------------
# 4. Upgrade pip + install vibe-qc + extras.
# ---------------------------------------------------------------------------
vibeqc_pip_upgrade "$VENV_PATH"
echo
vibeqc_pip_install_editable "$VENV_PATH" "$EXTRAS_SPEC" "."
vibeqc_install_colocated_extra \
    "$VENV_PATH" "$REPO_ROOT" "$VIBEQC_COLOCATED_EXTRA"

# ---------------------------------------------------------------------------
# 5. Banner.
# ---------------------------------------------------------------------------
echo
echo "==> Verifying:"
vibeqc_verify_install "$VENV_PATH/bin/python" "$EXTRAS_GROUP"
vibeqc_mark_owned_venv "$VENV_PATH" "$REPO_ROOT" "$PYTHON_BIN"
vibeqc_commit_venv_replacement
vibeqc_release_lifecycle_lock

echo
echo "==> Install complete."
echo
echo "    Activate the venv:"
echo "        source $VENV_PATH/bin/activate"
echo
echo "    Or run vibe-qc directly:"
echo "        $VENV_PATH/bin/python -c 'import vibeqc; vibeqc.print_banner()'"
echo
