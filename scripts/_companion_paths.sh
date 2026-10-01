#!/usr/bin/env bash
# Canonical answer to "where does a companion component live?".
#
# The 2026-09 repository split moved vibe-view, vibe-queue and the agentic
# loop OUT of this tree. They are siblings of the vibe-qc checkout
# (../vibe-view), never children of it, and no checkout of this repository --
# developer, release, or lane mirror -- contains them. vibe-basis is the one
# component that is still co-located, so it remains a child.
#
# Every consumer must resolve companions through this file rather than
# repeating "$REPO_ROOT/<component>". A missing sibling is a normal, supported
# configuration: a user who installed only vibe-qc has none of them, and the
# resolver reports that as a fact instead of handing back a path that cannot
# exist.
#
# API
#     vibeqc_companion_expected_root COMPONENT OUTVAR
#         Always sets OUTVAR to the path this checkout would use, whether or
#         not anything is there. Returns non-zero only for an unknown
#         component.
#
#     vibeqc_companion_root COMPONENT OUTVAR
#         Sets OUTVAR the same way and returns 0 only when that directory
#         exists. Callers that need a particular file still check for it.
#
#     vibeqc_companion_report_missing COMPONENT
#         Prints the standard explanation, the expected path and the override
#         to stderr.
#
# Each companion honours one override environment variable so a checkout kept
# somewhere other than the sibling slot can still be used:
#     vibe-view   VIBE_VIEW_ROOT
#     vibe-queue  VIBE_QUEUE_ROOT
#     vibe-basis  VIBE_BASIS_ROOT
#
# The agentic loop is a sibling too, but nothing in this repository resolves
# it: the one test that did now lives in the loop's own repository.

# Resolved from this file's own location so the table has a single origin and
# does not depend on the caller's cwd or on the caller's own REPO_ROOT.
VIBEQC_COMPANION_SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P
)"
VIBEQC_COMPANION_REPO_ROOT="$(
    cd "$VIBEQC_COMPANION_SCRIPT_DIR/.." && pwd -P
)"

# Deliberately a case rather than an associative array: bash 3.2 ships on
# macOS and the portability contract forbids bash-4 associative-array maps.
vibeqc_companion_layout() {
    case "$1" in
        vibe-view|vibe-queue) echo "sibling" ;;
        vibe-basis)           echo "in-tree" ;;
        *)                    return 1 ;;
    esac
}

vibeqc_companion_env_var() {
    case "$1" in
        vibe-view)  echo "VIBE_VIEW_ROOT" ;;
        vibe-queue) echo "VIBE_QUEUE_ROOT" ;;
        vibe-basis) echo "VIBE_BASIS_ROOT" ;;
        *)          return 1 ;;
    esac
}

vibeqc_companion_expected_root() {
    local _vqc_component="$1"
    local _vqc_out="$2"
    local _vqc_layout=""
    local _vqc_env=""
    local _vqc_override=""
    local _vqc_root=""

    if ! _vqc_layout="$(vibeqc_companion_layout "$_vqc_component")"; then
        echo "Error: unknown companion component '$_vqc_component'." >&2
        return 1
    fi
    _vqc_env="$(vibeqc_companion_env_var "$_vqc_component")"
    eval "_vqc_override=\${$_vqc_env:-}"

    if [ -n "$_vqc_override" ]; then
        _vqc_root="$_vqc_override"
    elif [ "$_vqc_layout" = "sibling" ]; then
        # The parent always exists, so normalise it and keep the reported
        # path free of a "..'" segment even when the sibling is absent.
        _vqc_root="$(
            cd "$VIBEQC_COMPANION_REPO_ROOT/.." && pwd -P
        )/$_vqc_component"
    else
        _vqc_root="$VIBEQC_COMPANION_REPO_ROOT/$_vqc_component"
    fi

    # Normalise only when the directory is really there; a path that does not
    # exist must still be reportable, and `cd` cannot normalise it.
    if [ -d "$_vqc_root" ]; then
        _vqc_root="$(cd "$_vqc_root" && pwd -P)"
    fi

    eval "$_vqc_out=\$_vqc_root"
}

vibeqc_companion_root() {
    local _vqc_root_component="$1"
    local _vqc_root_out="$2"
    local _vqc_root_value=""

    vibeqc_companion_expected_root "$_vqc_root_component" _vqc_root_value \
        || return 1
    eval "$_vqc_root_out=\$_vqc_root_value"
    [ -d "$_vqc_root_value" ]
}

vibeqc_companion_report_missing() {
    local _vqc_miss_component="$1"
    local _vqc_miss_layout=""
    local _vqc_miss_env=""
    local _vqc_miss_root=""

    if ! _vqc_miss_layout="$(
        vibeqc_companion_layout "$_vqc_miss_component"
    )"; then
        echo "Error: unknown companion component '$_vqc_miss_component'." >&2
        return 1
    fi
    _vqc_miss_env="$(vibeqc_companion_env_var "$_vqc_miss_component")"
    vibeqc_companion_expected_root "$_vqc_miss_component" _vqc_miss_root \
        || return 1

    if [ "$_vqc_miss_layout" = "sibling" ]; then
        echo "Error: no $_vqc_miss_component checkout was found." >&2
        echo "       $_vqc_miss_component is a separate repository since the" >&2
        echo "       2026-09 split; this checkout expects it as a sibling:" >&2
        echo "         $_vqc_miss_root" >&2
        echo "       Clone it there, or name an existing checkout:" >&2
        echo "         $_vqc_miss_env=/path/to/$_vqc_miss_component <command>" >&2
    else
        echo "Error: the co-located $_vqc_miss_component project is missing:" >&2
        echo "         $_vqc_miss_root" >&2
        echo "       It ships inside this repository; a checkout without it" >&2
        echo "       is incomplete. $_vqc_miss_env overrides the location." >&2
    fi
    return 0
}
