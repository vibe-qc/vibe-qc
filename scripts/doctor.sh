#!/bin/bash
# Read-only health check for an existing vibe-qc checkout.
#
# Answers: is this install in a usable state? Reports on:
#
#   * Working-tree state (clean / dirty, current branch, current SHA)
#   * Vendored native deps (which install/ trees exist; build-stamp
#     drift against current build_*.sh)
#   * Venv (path, python version, pip shebang sanity, requires-python
#     match)
#   * Banner output (which library versions are linked)
#
# No builds. No installs. No checkouts. Safe to run any time, including
# in the middle of an in-flight calculation.
#
# USAGE
#     ./scripts/doctor.sh [OPTIONS]
#
# OPTIONS
#     --venv PATH               Explicit venv path. Default: auto-detect
#                               (.venv / venv / .venv-vibeqc /
#                               venv-vibeqc, first hit wins).
#     -h, --help                Show this help.
#
# Exits 0 if every check passes, 1 if any warning was emitted.

set -euo pipefail

EXPLICIT_VENV=""

print_help() {
    # Walks from `# USAGE` until the first non-comment line. No magic
    # sentinel — survives header-comment edits that would have silently
    # broken the older awk-range approach.
    awk '/^# USAGE/ {p=1} p && !/^#/ {exit} p { sub(/^# ?/, ""); print }' "$0"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --venv)
            [ $# -ge 2 ] || { echo "Error: --venv requires an argument." >&2; exit 1; }
            EXPLICIT_VENV="$2"
            shift 2
            ;;
        -h|--help) print_help; exit 0 ;;
        *)
            echo "Error: unknown argument '$1'." >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=_venv_helpers.sh
. "$SCRIPT_DIR/_setup_helpers.sh"
. "$SCRIPT_DIR/_venv_helpers.sh"
# shellcheck source=_native_stamp.sh
. "$SCRIPT_DIR/_native_stamp.sh"

# Failure budget — increment on each non-fatal warning. Return code at
# the bottom is 0 if budget is 0, 1 otherwise. That makes doctor.sh
# scriptable: a wrapper can `if ./scripts/doctor.sh; then …`.
warnings=0

section() {
    echo
    echo "─── $1 ─────────────────────────────────────"
}

warn() {
    echo "  ! $1"
    warnings=$((warnings + 1))
}

ok() {
    echo "  ✓ $1"
}

note() {
    echo "    $1"
}

echo "==> vibe-qc doctor"
echo "    repo: $REPO_ROOT"

# ---------------------------------------------------------------------------
# Git state.
# ---------------------------------------------------------------------------
section "Git"

if git rev-parse --git-dir >/dev/null 2>&1; then
    head_sha=$(git rev-parse --short HEAD)
    head_branch=$(git rev-parse --abbrev-ref HEAD)
    head_msg=$(git log -1 --format='%s')
    ok "branch '$head_branch' @ $head_sha"
    note "tip: $head_msg"

    doctor_worktree_state=0
    vibeqc_worktree_state || doctor_worktree_state=$?
    if [ "$doctor_worktree_state" -eq 2 ]; then
        warn "git could not read this repository (update.sh would refuse to run)"
    elif [ "$doctor_worktree_state" -eq 1 ]; then
        warn "working tree has uncommitted changes (update.sh would refuse to run)"
    else
        ok "working tree clean"
    fi

    # How far ahead/behind origin/main is the current HEAD? Cheap signal
    # for "is an update available?". Skips silently if there's no
    # origin/main (e.g. someone's local-only fork) or fetch fails.
    if git rev-parse --verify --quiet origin/main >/dev/null 2>&1; then
        behind=$(git rev-list --count "HEAD..origin/main" 2>/dev/null || echo 0)
        ahead=$(git rev-list --count "origin/main..HEAD" 2>/dev/null || echo 0)
        if [ "$behind" -eq 0 ] && [ "$ahead" -eq 0 ]; then
            ok "in sync with origin/main"
        elif [ "$behind" -gt 0 ] && [ "$ahead" -eq 0 ]; then
            note "behind origin/main by $behind commit(s) — './scripts/update.sh --dev' to fast-forward"
        elif [ "$behind" -eq 0 ] && [ "$ahead" -gt 0 ]; then
            note "ahead of origin/main by $ahead commit(s) — local work not yet pushed"
        else
            note "diverged from origin/main: $ahead ahead, $behind behind — needs rebase"
        fi
    fi
else
    warn "not a git repository"
fi

# ---------------------------------------------------------------------------
# Native deps.
# ---------------------------------------------------------------------------
section "Native deps"

found_any=0
for dep in libint libxc spglib fftw libecpint openblas; do
    if [ -d "third_party/$dep/install" ]; then
        ok "third_party/$dep/install/ present"
        found_any=1
    fi
done

# libint's per-derivative-order max_am is a build-time choice that nothing
# else on disk reveals, and it decides whether analytic Hessians accept a
# g-function basis. Report what the tree was built with vs what the current
# environment would build (scripts/_libint_max_am.sh).
if [ -d "third_party/libint/install" ]; then
    _am_have="$(vibeqc_libint_installed_spec)"
    _am_want="$(vibeqc_libint_max_am_spec)"
    if [ "$_am_have" = "unknown" ]; then
        note "libint max_am: not recorded (install predates the marker); would build $_am_want"
    elif [ "$_am_have" = "$_am_want" ]; then
        ok "libint max_am: $_am_have (deriv 0/1/2)"
    else
        warn "libint max_am: built $_am_have, environment requests $_am_want — libint will rebuild"
    fi
    if [ "$_am_have" = "5_4_3" ]; then
        note "analytic Hessians are limited to f functions; 5_4_4 adds g-function support"
    fi
fi

if [ "$found_any" = "0" ]; then
    warn "no third_party/<dep>/install/ trees exist yet"
    note "run ./scripts/install.sh (or ./scripts/setup_native_deps.sh) to build them"
else
    # Stamp drift.
    echo
    if vibeqc_stamp_check --quiet; then
        ok "build stamp: all current"
    else
        # vibeqc_stamp_check has already printed the per-dep table; bump the budget.
        warn "build stamp drift detected (see table above)"
    fi
fi

# ---------------------------------------------------------------------------
# Venv.
# ---------------------------------------------------------------------------
section "Venv"

detected=""
vibeqc_detect_venv detected "$EXPLICIT_VENV"

if [ -z "$detected" ]; then
    if [ -n "$EXPLICIT_VENV" ]; then
        warn "no venv at '$EXPLICIT_VENV'"
    else
        warn "no venv found at .venv / venv / .venv-vibeqc / venv-vibeqc"
    fi
    note "run ./scripts/install.sh to create one"
else
    ok "venv: $detected"
    py_version=$("$detected/bin/python" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo "?")
    note "python: $py_version"
    if vibeqc_check_venv_health "$detected" 2>/dev/null; then
        ok "health checks passed"
    else
        warn "health checks failed — re-run with verbose:"
        note "  ./scripts/doctor.sh   # the diagnostic is hidden when stderr is /dev/null"
        # Re-run loud so the user actually sees it.
        echo
        vibeqc_check_venv_health "$detected" || true
    fi

    # Confirm the editable install is actually wired. `pip show vibe-qc`
    # lists a Location: that should match the repo root. If something
    # else (system-pip leak, sibling worktree's venv copied in) ended up
    # satisfying `import vibeqc`, doctor catches it here even though
    # the venv health checks and `import vibeqc` both passed.
    pip_loc=$("$detected/bin/python" -m pip show vibe-qc 2>/dev/null | awk -F': ' '/^Location:/ {print $2; exit}')
    pip_editable=$("$detected/bin/python" -m pip show vibe-qc 2>/dev/null | awk -F': ' '/^Editable project location:/ {print $2; exit}')
    if [ -z "$pip_loc" ]; then
        warn "'pip show vibe-qc' returns nothing — package not installed in this venv"
        note "run ./scripts/install.sh --force --venv $detected"
    elif [ -n "$pip_editable" ]; then
        # Newer pip (>=21.3) reports Editable project location separately.
        if [ "$pip_editable" = "$REPO_ROOT" ]; then
            ok "editable install rooted at $pip_editable"
        else
            warn "editable install rooted at $pip_editable (expected $REPO_ROOT)"
        fi
    elif [ "$pip_loc" = "$REPO_ROOT" ] || [ "$pip_loc" = "$REPO_ROOT/python" ]; then
        # Older pip puts the repo path in Location for editable installs.
        ok "editable install at $pip_loc"
    else
        warn "vibe-qc installed at $pip_loc — NOT this checkout ($REPO_ROOT)"
        note "the venv is wired to a different tree; ./scripts/install.sh --force to re-pin"
    fi
fi

# ---------------------------------------------------------------------------
# Banner.
# ---------------------------------------------------------------------------
section "Banner"

if [ -n "$detected" ] && [ -x "$detected/bin/python" ]; then
    if "$detected/bin/python" -c "import vibeqc" >/dev/null 2>&1; then
        "$detected/bin/python" -c "import vibeqc; vibeqc.print_banner()" || warn "banner print raised"
    else
        warn "vibeqc not importable from $detected"
        note "the venv exists but 'import vibeqc' fails — try:"
        note "  ./scripts/update.sh   # or"
        note "  ./scripts/install.sh --force"
    fi
else
    note "skipped — no usable venv"
fi

# ---------------------------------------------------------------------------
# Summary.
# ---------------------------------------------------------------------------
echo
if [ "$warnings" -eq 0 ]; then
    echo "==> doctor: clean (0 warnings)"
    exit 0
else
    echo "==> doctor: $warnings warning(s) — see the items prefixed with '!' above."
    exit 1
fi
