#!/usr/bin/env bash
# Transactionally provision the lean vibe-view environment used by vq's
# vibeview-dev capture program.
#
# USAGE
#     ./scripts/update_vibeview_capture_env.sh [OPTIONS]
#
# OPTIONS
#     --ref REF             Assert that checkout HEAD is exactly REF. This is
#                           the source-pin form injected by `vq admin update
#                           --expected-sha`; it does not fetch or switch refs.
#     --branch NAME         Assert that checkout HEAD is exactly NAME. This is
#                           the tag/branch form injected by pinned vq updates;
#                           it does not fetch or switch refs.
#     --python BIN          Python used to create the venv (default: python3).
#     --venv PATH           Capture venv (default: .venv-vibeview). Relative
#                           paths are resolved from the vibe-qc checkout.
#     --adopt-legacy        Replace an older/custom local vibe-view venv after
#                           proving its source. Required whenever an existing
#                           venv has no capture-specific ownership marker.
#     --dry-run             Validate and preview without changing files.
#     -h, --help            Show this help without requiring an environment.
#
# The replacement is built at the final path while the previous environment
# waits in a same-filesystem backup. Installation, import checks, and the real
# headless capture self-test must all pass before the backup is removed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
VIEW_SCRIPT_DIR="$REPO_ROOT/vibe-view/scripts"

PYTHON_BIN="${PYTHON:-python3}"
PYTHON_REQUESTED_EXPLICITLY=0
[ -z "${PYTHON:-}" ] || PYTHON_REQUESTED_EXPLICITLY=1
VENV_INPUT="${VIBE_VIEW_CAPTURE_VENV:-.venv-vibeview}"
ADOPT_LEGACY=0
DRY_RUN=0
SOURCE_ASSERTION_OPTION=""
SOURCE_ASSERTION_REF=""
SOURCE_ASSERTION_SHA=""

print_help() {
    awk '/^# USAGE/ {p=1} p && !/^#/ {exit} p {sub(/^# ?/, ""); print}' "$0"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --ref|--branch)
            option="$1"
            [ $# -ge 2 ] && [ -n "${2-}" ] || {
                echo "Error: $option requires a source ref (non-empty)." >&2
                exit 1
            }
            case "$2" in
                -*) echo "Error: $option requires a source ref, not option '$2'." >&2; exit 1 ;;
            esac
            if [ -n "$SOURCE_ASSERTION_OPTION" ]; then
                echo "Error: choose only one of --ref or --branch." >&2
                exit 1
            fi
            SOURCE_ASSERTION_OPTION="$option"
            SOURCE_ASSERTION_REF="$2"
            shift 2
            ;;
        --python)
            [ $# -ge 2 ] && [ -n "$2" ] || {
                echo "Error: --python requires an argument (non-empty)." >&2
                exit 1
            }
            case "$2" in
                -*) echo "Error: --python requires an executable, not option '$2'." >&2; exit 1 ;;
            esac
            PYTHON_BIN="$2"
            PYTHON_REQUESTED_EXPLICITLY=1
            shift 2
            ;;
        --venv)
            [ $# -ge 2 ] && [ -n "$2" ] || {
                echo "Error: --venv requires an argument (non-empty)." >&2
                exit 1
            }
            case "$2" in
                -*) echo "Error: --venv requires a path, not option '$2'." >&2; exit 1 ;;
            esac
            VENV_INPUT="$2"
            shift 2
            ;;
        --adopt-legacy) ADOPT_LEGACY=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help) print_help; exit 0 ;;
        *)
            echo "Error: unknown argument '$1'." >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

# vq has already fetched and checked out its accepted-report pin before it
# invokes this script. Re-resolve both names here, before any environment or
# lock mutation, so the capture build cannot silently use a different tree.
# These options are assertions rather than checkout operations deliberately:
# source selection remains owned by vq's managed update path.
if [ -n "$SOURCE_ASSERTION_OPTION" ]; then
    if ! SOURCE_ASSERTION_SHA="$(
        git -C "$REPO_ROOT" rev-parse --verify \
            "${SOURCE_ASSERTION_REF}^{commit}" 2>/dev/null
    )"; then
        echo "Error: cannot resolve $SOURCE_ASSERTION_OPTION source ref '$SOURCE_ASSERTION_REF'." >&2
        exit 1
    fi
    if ! CHECKOUT_HEAD_SHA="$(
        git -C "$REPO_ROOT" rev-parse --verify 'HEAD^{commit}' 2>/dev/null
    )"; then
        echo "Error: cannot resolve checkout HEAD under $REPO_ROOT." >&2
        exit 1
    fi
    if [ "$CHECKOUT_HEAD_SHA" != "$SOURCE_ASSERTION_SHA" ]; then
        echo "Error: $SOURCE_ASSERTION_OPTION '$SOURCE_ASSERTION_REF' does not match checkout HEAD." >&2
        echo "       expected: $SOURCE_ASSERTION_SHA" >&2
        echo "       actual:   $CHECKOUT_HEAD_SHA" >&2
        exit 1
    fi
fi

if [ ! -f "$VIEW_SCRIPT_DIR/_venv_helpers.sh" ]; then
    echo "Error: canonical vibe-view lifecycle helpers were not found under:" >&2
    echo "       $VIEW_SCRIPT_DIR" >&2
    exit 1
fi

# shellcheck source=../vibe-view/scripts/_venv_helpers.sh
. "$VIEW_SCRIPT_DIR/_venv_helpers.sh"

vibe_view_refuse_privileged_lifecycle

CAPTURE_MARKER_NAME=".vibe-view-capture.json"
CAPTURE_OWNERSHIP_MODE="new"

capture_marker_is_owned() {
    local venv="$1"
    local python_bin="$2"

    PYTHONNOUSERSITE=1 "$python_bin" -I -S - \
        "$venv" "$VIBE_VIEW_PROJECT_DIR" "$CAPTURE_MARKER_NAME" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

venv = Path(sys.argv[1]).resolve()
project = Path(sys.argv[2]).resolve()
marker = venv / sys.argv[3]
if marker.is_symlink() or not marker.is_file():
    raise SystemExit(1)
try:
    payload = json.loads(marker.read_text())
    marker_project = Path(payload["project"])
    if not marker_project.is_absolute():
        raise ValueError("project path is not absolute")
except (OSError, UnicodeError, KeyError, TypeError, ValueError):
    raise SystemExit(1)
raise SystemExit(
    not (
        payload.get("schema") == 1
        and payload.get("kind") == "vibe-view-capture-environment"
        and marker_project.resolve() == project
    )
)
PY
}

capture_assert_replaceable() {
    local venv="$1"
    local python_bin="$2"

    vibe_view_assert_removable_venv "$venv"
    if capture_marker_is_owned "$venv" "$python_bin"; then
        CAPTURE_OWNERSHIP_MODE="capture-owned"
        return 0
    fi

    if [ "$ADOPT_LEGACY" != "1" ]; then
        echo "Error: refusing to replace an unmarked vibe-view environment:" >&2
        echo "       $venv" >&2
        echo "Inspect it and pass --adopt-legacy only if it may become capture-only." >&2
        return 1
    fi

    # Prove source ownership without executing the environment selected for
    # replacement. The trusted base interpreter reads its marker/PEP 610 data.
    vibe_view_assert_owned_standalone_venv "$venv" "$python_bin" 1
    CAPTURE_OWNERSHIP_MODE="legacy-adopted"
}

capture_write_marker() {
    local venv="$1"

    "$venv/bin/python" - "$venv" "$VIBE_VIEW_PROJECT_DIR" "$CAPTURE_MARKER_NAME" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

venv = Path(sys.argv[1])
project = Path(sys.argv[2]).resolve()
marker = venv / sys.argv[3]
temporary = venv / f"{marker.name}.{os.getpid()}.tmp"
temporary.write_text(
    json.dumps(
        {
            "schema": 1,
            "kind": "vibe-view-capture-environment",
            "project": str(project),
        },
        indent=2,
    )
    + "\n"
)
os.replace(temporary, marker)
PY
}

case "$VENV_INPUT" in
    /*) RAW_VENV_TARGET="$VENV_INPUT" ;;
    *)  RAW_VENV_TARGET="$REPO_ROOT/$VENV_INPUT" ;;
esac

# Pick one stable interpreter outside the environment that may be replaced.
# The automatic path deliberately ignores an activated target environment;
# an explicit --python/PYTHON request instead fails closed if it points there.
if ! command -v vibe_toolset_find_external_python >/dev/null 2>&1 || \
   ! command -v vibe_toolset_select_external_python >/dev/null 2>&1; then
    echo "Error: shared toolset lifecycle Python helpers are unavailable." >&2
    exit 1
fi
if [ "$PYTHON_REQUESTED_EXPLICITLY" = "1" ]; then
    vibe_toolset_select_external_python \
        PYTHON_BIN "$PYTHON_BIN" "$RAW_VENV_TARGET" 3 11
else
    vibe_toolset_find_external_python PYTHON_BIN "$RAW_VENV_TARGET" 3 11
fi
vibe_view_assert_python "$PYTHON_BIN"

if [ -e "$RAW_VENV_TARGET" ] || [ -L "$RAW_VENV_TARGET" ]; then
    vibe_view_resolve_removal_path VENV_PATH "$RAW_VENV_TARGET" "$PYTHON_BIN"
    capture_assert_replaceable "$VENV_PATH" "$PYTHON_BIN"
else
    vibe_view_resolve_venv_path VENV_PATH "$RAW_VENV_TARGET" "$PYTHON_BIN"
fi
vibe_view_assert_safe_venv_target "$VENV_PATH"

echo "==> vibe-view capture environment update"
echo "    source:   $VIBE_VIEW_PROJECT_DIR"
[ -z "$SOURCE_ASSERTION_OPTION" ] || echo "    source ref: $SOURCE_ASSERTION_REF ($SOURCE_ASSERTION_OPTION)"
[ -z "$SOURCE_ASSERTION_SHA" ] || echo "    source commit: $SOURCE_ASSERTION_SHA"
echo "    venv:     $VENV_PATH"
echo "    python:   $PYTHON_BIN"
echo "    profile:  core (headless capture, no browser server)"
[ "$DRY_RUN" = "1" ] && echo "    dry-run:  no files will be changed"
[ "$ADOPT_LEGACY" = "1" ] && echo "    adoption: explicit legacy-environment approval"
echo

if [ "$DRY_RUN" = "1" ]; then
    if [ -e "$VENV_PATH" ]; then
        echo "    [venv] $CAPTURE_OWNERSHIP_MODE environment would be replaced transactionally."
    else
        echo "    [venv] a new environment would be created."
    fi
    echo "    [pip] vibe-view core would be installed from this checkout."
    echo "    [verify] imports and a real offscreen PNG render would be checked."
    echo
    echo "==> Dry-run complete."
    exit 0
fi

capture_lifecycle_cleanup() {
    local status="$?"
    local cleanup_status=0

    set +e
    vibe_view_abort_venv_replacement || cleanup_status=$?
    if ! vibe_view_release_target_lock; then
        [ "$cleanup_status" != "0" ] || cleanup_status=1
    fi
    if [ "$status" = "0" ] && [ "$cleanup_status" != "0" ]; then
        status="$cleanup_status"
    fi
    return "$status"
}

vibe_view_acquire_target_lock "$VENV_PATH" "capture update" "$PYTHON_BIN"
trap capture_lifecycle_cleanup EXIT

# Everything that authorizes replacement is repeated while the common
# checkout-and-target lock is held. This closes the gap between the dry
# preflight above and the first filesystem mutation.
vibe_view_assert_safe_venv_target "$VENV_PATH"
CAPTURE_EXPECTED_ORIGINAL=0
if [ -e "$VENV_PATH" ] || [ -L "$VENV_PATH" ]; then
    CAPTURE_EXPECTED_ORIGINAL=1
    capture_assert_replaceable "$VENV_PATH" "$PYTHON_BIN"
fi

vibe_view_begin_venv_replacement "$VENV_PATH" "$CAPTURE_EXPECTED_ORIGINAL"
vibe_view_start_venv_replacement \
    "$VENV_PATH" "$PYTHON_BIN" "$ADOPT_LEGACY" capture

echo "==> Creating replacement virtualenv: $VENV_PATH"
"$PYTHON_BIN" -m venv "$VENV_PATH"
vibe_view_install_environment "$VENV_PATH" ""
vibe_view_verify_environment "$VENV_PATH" core 0

# Provide a portable xvfb-run entry point for vq health checks. Linux uses a
# system xvfb-run when available; macOS and OSMesa-capable Linux hosts fall
# back to PyVista's direct offscreen rendering.
cat >"$VENV_PATH/bin/xvfb-run" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd -P)"
SELF_PATH="$SELF_DIR/$(basename "$0")"
CANDIDATES=()
OLD_IFS="$IFS"
IFS=:
for path_dir in ${PATH:-}; do
    [ -n "$path_dir" ] || path_dir="."
    CANDIDATES+=("$path_dir/xvfb-run")
done
IFS="$OLD_IFS"
CANDIDATES+=(
    /usr/bin/xvfb-run
    /bin/xvfb-run
    /usr/local/bin/xvfb-run
    /opt/homebrew/bin/xvfb-run
    /opt/X11/bin/xvfb-run
)

for candidate in "${CANDIDATES[@]}"; do
    if [ -x "$candidate" ]; then
        [ "$candidate" -ef "$SELF_PATH" ] && continue
        candidate_dir="$(cd "$(dirname "$candidate")" 2>/dev/null && pwd -P || true)"
        candidate_path="$candidate_dir/$(basename "$candidate")"
        if [ -n "$candidate_dir" ] && [ "$candidate_path" != "$SELF_PATH" ]; then
            exec "$candidate_path" "$@"
        fi
    fi
done

while [ "$#" -gt 0 ]; do
    case "$1" in
        -a|--auto-servernum) shift ;;
        -e|--error-file|-f|--auth-file|-n|--server-num|-s|--server-args|-w|--wait)
            shift
            [ "$#" -gt 0 ] && shift
            ;;
        --) shift; break ;;
        -*) shift ;;
        *) break ;;
    esac
done

if [ "$#" -eq 0 ]; then
    echo "xvfb-run fallback: no command provided" >&2
    exit 2
fi

export PYVISTA_OFF_SCREEN="${PYVISTA_OFF_SCREEN:-True}"
exec "$@"
SH
chmod +x "$VENV_PATH/bin/xvfb-run"

"$VENV_PATH/bin/python" - <<'PY'
from __future__ import annotations

import importlib.util

missing = [
    name
    for name in ("vibeview", "pyvista")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit("missing required module(s): " + ", ".join(missing))

unexpected = [
    name
    for name in ("trame", "trame_vtk", "trame_vuetify")
    if importlib.util.find_spec(name) is not None
]
if unexpected:
    raise SystemExit("unexpected web module(s): " + ", ".join(unexpected))
PY

PYVISTA_OFF_SCREEN=True \
    "$VENV_PATH/bin/xvfb-run" -a "$VENV_PATH/bin/vibe-view" capture-selftest

vibe_view_mark_standalone_venv "$VENV_PATH"
capture_write_marker "$VENV_PATH"
vibe_view_commit_venv_replacement

echo
echo "==> vibe-view capture environment ready: $VENV_PATH"
echo "    Health check: $VENV_PATH/bin/xvfb-run -a $VENV_PATH/bin/vibe-view capture-selftest"

vibe_view_release_target_lock
trap - EXIT
