#!/usr/bin/env bash
# scripts/update_native_deps.sh — rebuild only the vendored native deps
# whose build_<dep>.sh recipe changed (or whose install/ tree is
# missing), then refresh the build stamp.
#
# Why this exists, vs the two adjacent scripts:
#   * setup_native_deps.sh builds every dep but is idempotent ON
#     EXISTENCE — once third_party/<dep>/install/ exists it short-
#     circuits, so a *changed* build_<dep>.sh (e.g. a libint max-am
#     bump) is NOT picked up by a plain re-run.
#   * update.sh --rebuild-native-deps fixes that but is ALL-OR-NOTHING:
#     it wipes every dep's install/ and rebuilds the lot (15-40+ min),
#     even when only one recipe changed.
# This reads the build stamp (scripts/_native_stamp.sh) and rebuilds
# just the dep(s) that drifted. update.sh calls it automatically when it
# detects drift; it is also safe to run by hand after editing a
# build_<dep>.sh, so every installation ends up built the same way.
#
# Each build_<dep>.sh already applies its own build-pressure safety
# (nice/ionice + memory-safe parallel cap, via _safe_build_env.sh), so
# this orchestrator does NOT re-source that helper — avoiding the
# argv-on-re-exec footgun that bit update.sh on compute-host-d (2026-05-25).
# It only sequences the rebuilds and pauses any running vq jobs.
#
# USAGE
#     ./scripts/update_native_deps.sh [OPTIONS]
#
# OPTIONS
#     --dep NAME    Force-rebuild this dep regardless of drift
#                   (repeatable). One of: libint libxc spglib fftw
#                   libecpint openblas.
#     --all         Rebuild every known dep (like
#                   update.sh --rebuild-native-deps, without the git +
#                   venv steps).
#     --libint-max-am SPEC
#                   libint's max angular momentum per derivative order,
#                   as N_N_N for orders 0/1/2, each part 1..6. Default
#                   5_4_4 (g-function analytic Hessians, ~1.5-2 h);
#                   5_4_3 is the pre-2026-08-06 ~30 min build without
#                   them; 6_5_4 adds i-function energies. Passing a spec
#                   that differs from what the installed tree was built
#                   with is itself drift, so libint gets rebuilt without
#                   needing --dep libint.
#     --dry-run     Print what would be rebuilt and exit.
#     -h, --help    Show this help.
#
# After a rebuild, re-run `pip install -e . --no-build-isolation`
# (or scripts/update.sh, which does it for you) so the vibe-qc
# extension links against the refreshed deps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

print_help() {
    awk '/^# USAGE/ {p=1} p && !/^#/ {exit} p { sub(/^# ?/, ""); print }' "$0"
}

# Sourced before the argv loop so --libint-max-am can be validated at parse
# time. _native_stamp.sh (sourced below) pulls the same file in; re-sourcing
# only redefines the functions, which is harmless.
# shellcheck source=_libint_max_am.sh
. "$SCRIPT_DIR/_libint_max_am.sh"

FORCE_ALL=false
DRY_RUN=false
EXPLICIT_DEPS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --dep)
            [ $# -ge 2 ] || { echo "Error: --dep requires an argument." >&2; exit 1; }
            EXPLICIT_DEPS+=("$2"); shift 2 ;;
        --all)     FORCE_ALL=true; shift ;;
        --libint-max-am)
            [ $# -ge 2 ] || { echo "Error: --libint-max-am requires an argument." >&2; exit 1; }
            vibeqc_libint_max_am_accept "$2" || exit 1
            shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Error: unknown argument '$1'. Run with --help for usage." >&2; exit 1 ;;
    esac
done

# A real updater must inspect stamps and install sentinels under the same lock
# it will use for removal/rebuild. Otherwise it can race a direct build and
# incorrectly report "all current" from transient state. Dry-run is a strictly
# read-only preview and intentionally remains lock-free.
if ! $DRY_RUN; then
    # shellcheck source=_build_lock.sh
    . "$SCRIPT_DIR/_build_lock.sh"
    vibeqc_acquire_build_lock
fi

# shellcheck source=_native_stamp.sh
. "$SCRIPT_DIR/_native_stamp.sh"

STAMP_WRITE_SAFE=true
if [ ! -f "$VIBEQC_STAMP_PATH" ]; then
    for _legacy_dep in libint libxc spglib fftw libecpint openblas; do
        if [ -d "third_party/$_legacy_dep/install" ]; then
            STAMP_WRITE_SAFE=false
            break
        fi
    done
fi

# Canonical dep list + recipe paths, straight from the stamp pairs (the
# single source of truth shared with setup_native_deps.sh). Parallel
# indexed arrays — no associative arrays (macOS /bin/bash is 3.2).
DEP_NAMES=()
DEP_SCRIPTS=()
while read -r _dep _script; do
    [ -z "$_dep" ] && continue
    DEP_NAMES+=("$_dep")
    DEP_SCRIPTS+=("$_script")
done < <(_vibeqc_stamp_pairs)

_script_for() {
    local i
    for i in "${!DEP_NAMES[@]}"; do
        if [ "${DEP_NAMES[$i]}" = "$1" ]; then
            printf '%s\n' "${DEP_SCRIPTS[$i]}"
            return 0
        fi
    done
    if [ "$1" = "openblas" ]; then
        printf '%s\n' "scripts/build_openblas.sh"
        return 0
    fi
    return 1
}

_is_known_dep() {
    case "$1" in
        libint|libxc|spglib|fftw|libecpint|openblas) return 0 ;;
        *) return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Decide the target set, preserving canonical (dependency) order.
# ---------------------------------------------------------------------------
TARGETS=()
if $FORCE_ALL; then
    TARGETS=("${DEP_NAMES[@]}")
elif [ ${#EXPLICIT_DEPS[@]} -gt 0 ]; then
    for d in "${EXPLICIT_DEPS[@]}"; do
        if ! _is_known_dep "$d"; then
            echo "Error: unknown dep '$d'. Known: ${DEP_NAMES[*]}" >&2
            exit 1
        fi
    done
    # OpenBLAS is opt-in and therefore absent from _vibeqc_stamp_pairs until
    # installed. An explicit --dep openblas must still be able to add it.
    for x in "${EXPLICIT_DEPS[@]}"; do
        if [ "$x" = "openblas" ]; then
            _openblas_listed=0
            for d in "${DEP_NAMES[@]}"; do
                [ "$d" = "openblas" ] && _openblas_listed=1
            done
            if [ "$_openblas_listed" = "0" ]; then
                DEP_NAMES+=("openblas")
                DEP_SCRIPTS+=("scripts/build_openblas.sh")
            fi
            break
        fi
    done
    for d in "${DEP_NAMES[@]}"; do
        for x in "${EXPLICIT_DEPS[@]}"; do
            if [ "$d" = "$x" ]; then
                TARGETS+=("$d")
                break
            fi
        done
    done
else
    # Default: drift-driven (changed recipe or missing install).
    _DRIFTED=()
    while IFS= read -r _line; do
        [ -n "$_line" ] && _DRIFTED+=("$_line")
    done < <(vibeqc_stamp_drifted_deps)
    if [ ${#_DRIFTED[@]} -gt 0 ]; then
        for d in "${DEP_NAMES[@]}"; do
            for x in "${_DRIFTED[@]}"; do
                if [ "$d" = "$x" ]; then
                    TARGETS+=("$d")
                    break
                fi
            done
        done
    fi
fi

# A forced subset is additive, not permission to certify unrelated drift as
# current. Rebuild any other dependency the existing stamp says is stale
# before the shared stamp is refreshed.
if [ ${#EXPLICIT_DEPS[@]} -gt 0 ] && [ -f "$VIBEQC_STAMP_PATH" ]; then
    _EXTRA_DRIFTED=()
    while IFS= read -r _line; do
        [ -n "$_line" ] && _EXTRA_DRIFTED+=("$_line")
    done < <(vibeqc_stamp_drifted_deps)
    for d in "${DEP_NAMES[@]}"; do
        _needs_add=0
        for x in "${_EXTRA_DRIFTED[@]}"; do
            [ "$d" = "$x" ] && _needs_add=1
        done
        if [ "$_needs_add" = "1" ]; then
            _already_targeted=0
            for x in "${TARGETS[@]}"; do
                [ "$d" = "$x" ] && _already_targeted=1
            done
            [ "$_already_targeted" = "1" ] || TARGETS+=("$d")
        fi
    done
fi

# --all rebuilds every core dependency plus an already-selected OpenBLAS, so
# it establishes a complete trusted baseline even for a legacy tree.
if $FORCE_ALL; then
    STAMP_WRITE_SAFE=true
fi

if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "==> Native deps: all current — nothing to rebuild."
    echo "    (Edit a scripts/build_<dep>.sh, or pass --all / --dep NAME, to force one.)"
    exit 0
fi

echo "==> Native deps to rebuild: ${TARGETS[*]}"
if $DRY_RUN; then
    echo "    --dry-run: no changes made."
    exit 0
fi

# ---------------------------------------------------------------------------
# Cooperative vq pause for the build window (mirrors update.sh). The trap
# guarantees resume on exit / failure / Ctrl-C. No-op if vq isn't
# installed / the daemon is down / there are no jobs. build_<dep>.sh
# applies its own nice/ionice + parallel cap, so we don't here.
# ---------------------------------------------------------------------------
# shellcheck source=_vq_cooperation.sh
. "$SCRIPT_DIR/_vq_cooperation.sh"
vibeqc_vq_install_trap
vibeqc_vq_pause

for dep in "${TARGETS[@]}"; do
    script="$(_script_for "$dep")"
    echo
    echo "==> Rebuilding $dep from its pinned source — then $script ..."
    if [ "$dep" = "libecpint" ]; then
        # libecpint carries three sibling source/build trees under this root.
        rm -rf "third_party/$dep"
    else
        rm -rf "third_party/$dep/src" \
               "third_party/$dep/build" \
               "third_party/$dep/install"
    fi
    bash "$script"
done

# Refresh provenance only when every pre-existing install had a trusted
# baseline. A targeted build beside legacy unstamped binaries must not certify
# those unrelated binaries as current.
if $STAMP_WRITE_SAFE; then
    vibeqc_stamp_write
else
    echo "==> Build stamp left absent: unrelated legacy install trees were not rebuilt."
    echo "    Run --all to rebuild and certify the complete native set."
fi

echo
echo "==> Native-dep rebuild complete: ${TARGETS[*]}"
echo "    Re-run 'pip install -e . --no-build-isolation' (or scripts/update.sh)"
echo "    so the vibe-qc extension links against the refreshed deps."
