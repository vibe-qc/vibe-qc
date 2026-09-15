# scripts/_setup_helpers.sh — shared install.sh / update.sh logic.
#
# Sourced (NOT executed) by install.sh and update.sh. Three functions:
#
#   vibeqc_set_branch NEW_VALUE SOURCE_FLAG [SUGGESTION_LINE]
#       Records that an argv flag chose a branch/ref. Errors loud if
#       the caller's BRANCH global is already set, so the user can't
#       silently combine --release with --dev / --branch / --ref. Uses
#       the caller's BRANCH and BRANCH_SOURCE shell globals (bash
#       doesn't have great named-arg ergonomics; globals are the
#       cleanest version of "these two scripts collaborate"). The
#       optional 3rd arg overrides the default "Pick one branch flag."
#       suggestion line — update.sh adds --ref to its suggestion,
#       install.sh leaves it out (no --ref support there).
#
#   vibeqc_extras_to_pip_spec GROUP OUT_VAR
#       Maps the user-friendly --extras GROUP value (test / dev / docs
#       / viewer / viewer-gpu / basisopt / ase / dispersion / mpi /
#       dispersion,mpi / mace / all / none) to the pip extras
#       suffix the editable install needs ("[test]", "[dev]", "", …).
#       Stores the result in OUT_VAR via printf -v. Errors loud + non-
#       zero on a bogus group so the caller's `set -e` propagates. For a
#       co-located package, VIBEQC_COLOCATED_EXTRA records the second install.
#
#   vibeqc_install_colocated_extra VENV_PATH REPO_ROOT GROUP
#       Installs and verifies the local vibe-view or vibe-basis package when
#       GROUP is viewer-gpu or basisopt. pip does not read [tool.uv.sources],
#       so the shell lifecycle must resolve these paths explicitly.
#
#   vibeqc_assert_colocated_extra_available REPO_ROOT GROUP
#       Fails before native or Python installation work if the selected local
#       companion is absent from the checked-out source tree.
#
#   vibeqc_checkout_ref BRANCH [ALLOW_RELEASE_TAG_FALLBACK]
#       Fetches origin + checks out BRANCH via the same three-way
#       dispatch install.sh and update.sh used to copy-paste:
#         * local branch  → checkout + ff-pull
#         * tag           → checkout (detached HEAD)
#         * remote-only   → create local tracking branch
#         * commit-ish    → detached checkout
#         * unknown       → return 1 (caller emits the suggestion line)
#       After a successful checkout, prints `now at <sha>: <msg>`.
#       With the second argument 1, a missing release branch selects the
#       newest stable vX.Y.Z tag advertised by origin. Explicit refs do not
#       enable this fallback. Local-only and prerelease tags are ignored.
#       Caller is responsible for the dirty-tree check beforehand
#       (install.sh only requires it when BRANCH is provided, update.sh
#       always requires it; the policy difference doesn't belong here).
#
# All functions assume `set -euo pipefail` in the caller.

vibeqc_set_branch() {
    local new_value="$1"
    local source_flag="$2"
    local suggestion="${3:-Pick one of --release / --dev / --branch NAME.}"
    if [ -n "${BRANCH:-}" ]; then
        echo "Error: '$source_flag' conflicts with previously-set branch (${BRANCH_SOURCE:-?} → $BRANCH)." >&2
        echo "$suggestion" >&2
        exit 1
    fi
    BRANCH="$new_value"
    BRANCH_SOURCE="$source_flag"
}

vibeqc_extras_to_pip_spec() {
    local group="$1"
    local out_var="$2"
    local spec
    VIBEQC_COLOCATED_EXTRA=""
    case "$group" in
        none)       spec="" ;;
        test)       spec="[test]" ;;
        dev)        spec="[dev]" ;;
        docs)       spec="[docs]" ;;
        viewer)     spec="[viewer]" ;;
        viewer-gpu) spec=""; VIBEQC_COLOCATED_EXTRA="viewer-gpu" ;;
        basisopt)   spec=""; VIBEQC_COLOCATED_EXTRA="basisopt" ;;
        ase)        spec="[ase]" ;;
        dispersion) spec="[dispersion]" ;;
        mpi)        spec="[mpi]" ;;
        dispersion,mpi) spec="[dispersion,mpi]" ;;
        mace)       spec="[mace]" ;;
        # 'all' deliberately omits [mace]: the MACE MLIP stack is a heavy
        # (~650 MB torch/e3nn) Python <=3.13-only opt-in. Folding it into
        # 'all' would bloat every "give me everything" install and would
        # no-op on the 3.14 dev .venv anyway (pyproject marker
        # python_version < '3.14'). Request it explicitly: --extras mace.
        all)        spec="[dev,docs,viewer]" ;;
        *)
            echo "Error: --extras must be one of: test / dev / docs / viewer / viewer-gpu / basisopt / ase / dispersion / mpi / dispersion,mpi / mace / all / none (got '$group')." >&2
            return 1
            ;;
    esac
    printf -v "$out_var" '%s' "$spec"
}

vibeqc_verify_install() {
    local python="$1"
    local extras_group="$2"

    case ",$extras_group," in
        *,mpi,*)
            # A plain mpi4py import normally calls MPI_Init. Some site MPI
            # stacks require their launcher even for one rank, so the generic
            # install lifecycle proves the extension/native-library link
            # without singleton initialization. Cluster deployers own the
            # separate launcher-backed execution gate.
            "$python" - <<'PY'
import mpi4py

mpi4py.rc.initialize = False
import vibeqc
from mpi4py import MPI

if MPI.Is_initialized():
    raise SystemExit("generic MPI verification unexpectedly initialized MPI")
MPI.Get_version()
MPI.get_vendor()
MPI.Get_library_version()
vibeqc.print_banner()
PY
            ;;
        *)
            "$python" -c "import vibeqc; vibeqc.print_banner()"
            ;;
    esac
}

vibeqc_assert_colocated_extra_available() {
    local repo_root="$1"
    local group="$2"
    local project=""

    case "$group" in
        "") return 0 ;;
        viewer-gpu)
            # vibe-view is a separate repository now. There is no co-located
            # directory to install from; point the user at a sibling checkout
            # rather than failing with a confusing "path not found".
            echo "Error: the 'viewer-gpu' extra installs vibe-view, which is" >&2
            echo "       now a separate project and is no longer co-located." >&2
            echo "       Clone it alongside this checkout and install it:" >&2
            echo "         git clone <vibe-view repo> ../vibe-view" >&2
            echo "         \"\$venv/bin/python\" -m pip install -e '../vibe-view[viewer]'" >&2
            return 1
            ;;
        basisopt)   project="$repo_root/vibe-basis" ;;
        *)
            echo "Error: unsupported co-located extra '$group'." >&2
            return 1
            ;;
    esac

    if [ ! -f "$project/pyproject.toml" ]; then
        echo "Error: co-located project is missing: $project" >&2
        return 1
    fi
}

vibeqc_install_colocated_extra() {
    local venv="$1"
    local repo_root="$2"
    local group="$3"
    local project=""
    local install_spec=""
    local verify=""

    case "$group" in
        "") return 0 ;;
        viewer-gpu)
            # vibe-view is a separate repository; nothing co-located to install.
            # vibeqc_assert_colocated_extra_available() below reports it.
            echo "Error: 'viewer-gpu' installs vibe-view, which is now a" >&2
            echo "       separate project. Install it from its own checkout:" >&2
            echo "         \"\$venv/bin/python\" -m pip install -e '../vibe-view[viewer]'" >&2
            return 1
            ;;
        basisopt)
            project="$repo_root/vibe-basis"
            install_spec="$project"
            verify='import vibe_basis'
            ;;
        *)
            echo "Error: unsupported co-located extra '$group'." >&2
            return 1
            ;;
    esac

    vibeqc_assert_colocated_extra_available "$repo_root" "$group"
    echo "==> Installing co-located $group package from this checkout..."
    "$venv/bin/python" -m pip install --upgrade -e "$install_spec"
    "$venv/bin/python" -c "$verify"
}

vibeqc_checkout_ref() {
    local branch="$1"
    local release_tag_fallback="${2:-0}"

    echo "==> Fetching from origin..."
    git fetch --quiet origin --tags || return

    local selected_tag=""
    if [ "$branch" = release ] && [ "$release_tag_fallback" = 1 ] \
        && ! git show-ref --verify --quiet refs/heads/release; then
        # Query origin, not just cached refs: a single-branch clone may omit
        # its release branch, and local tags need not be published releases.
        local advertised oid ref release_oid="" tag_oid=""
        advertised=$(git ls-remote --refs --sort=-version:refname origin \
            refs/heads/release 'refs/tags/v*') || return
        while IFS=$'\t' read -r oid ref; do
            if [ "$ref" = refs/heads/release ]; then
                release_oid="$oid"
            elif [ -z "$selected_tag" ] \
                && [[ "$ref" =~ ^refs/tags/v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
                selected_tag="$ref"
                tag_oid="$oid"
            fi
        done <<< "$advertised"
        if [ -n "$release_oid" ]; then
            selected_tag=""
            git fetch --quiet origin \
                refs/heads/release:refs/remotes/origin/release || return
        elif [ -z "$selected_tag" ]; then
            echo "Error: origin publishes neither a release branch nor a stable vX.Y.Z tag." >&2
            return 1
        elif [ "$(git rev-parse --verify "$selected_tag" 2>/dev/null)" != "$tag_oid" ]; then
            echo "Error: release tag changed during fetch; inspect the refs and retry." >&2
            return 1
        fi
    fi

    if [ -n "$selected_tag" ]; then
        echo "==> No release branch; checking out '${selected_tag#refs/tags/}' (detached HEAD)..."
        git checkout --quiet --detach "$selected_tag" || return
    elif git show-ref --verify --quiet "refs/heads/$branch"; then
        echo "==> Switching to local branch '$branch' and pulling..."
        git checkout --quiet "$branch" || return
        git pull --ff-only --quiet origin "$branch" || return
    elif git show-ref --verify --quiet "refs/tags/$branch"; then
        echo "==> Checking out tag '$branch' (detached HEAD)..."
        git checkout --quiet "$branch" || return
    elif git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
        echo "==> Creating local branch '$branch' tracking origin/$branch..."
        git checkout --quiet -b "$branch" "origin/$branch" || return
    elif git cat-file -e "$branch^{commit}" 2>/dev/null; then
        echo "==> Checking out commit '$branch' (detached HEAD)..."
        git checkout --quiet --detach "$branch" || return
    else
        echo "Error: '$branch' is not a known branch, tag, or commit." >&2
        return 1
    fi

    local sha msg
    sha="$(git rev-parse --short HEAD)" || return
    msg="$(git log -1 --format='%s')" || return
    echo "    now at $sha: $msg"
}
