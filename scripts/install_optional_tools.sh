#!/usr/bin/env bash
# Install optional third-party tools into a selected vibe-qc virtualenv.
#
# USAGE
#     ./scripts/install_optional_tools.sh [OPTIONS] [TOOL ...]
#
# OPTIONS
#     --python BIN          Install with this Python interpreter.
#     --venv PATH           Install into PATH/bin/python. Relative paths are
#                           resolved from the vibe-qc checkout.
#     --upgrade             Upgrade tools that are already installed.
#     -y, --yes             Install without interactive prompts.
#     --dry-run             Preview the selected environment and pip actions.
#     -h, --help            Show this help without requiring an environment.
#
# TOOL
#     moltui                Terminal viewer for Molden, cube, fchk, gbw,
#                           hess, and xyz files. It can display output from
#                           vibe-qc and other quantum-chemistry programs.
#
# With neither --python nor --venv, the script prefers an activated venv and
# then <checkout>/.venv. The standalone vibe-view, vq, and vibe-basis tools
# have their own install/update/uninstall/reinstall scripts and are not added
# to vibe-qc's compiled environment by this helper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

YES=0
UPGRADE=0
DRY_RUN=0
PYTHON_INPUT=""
VENV_INPUT=""
WANTED=()

print_help() {
    awk '/^# USAGE/ {p=1} p && !/^#/ {exit} p {sub(/^# ?/, ""); print}' "$0"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --python)
            [ $# -ge 2 ] && [ -n "$2" ] || {
                echo "Error: --python requires an argument (non-empty)." >&2
                exit 1
            }
            case "$2" in
                -*) echo "Error: --python requires an executable, not option '$2'." >&2; exit 1 ;;
            esac
            PYTHON_INPUT="$2"
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
        --upgrade) UPGRADE=1; shift ;;
        -y|--yes)  YES=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) print_help; exit 0 ;;
        --)
            shift
            while [ $# -gt 0 ]; do
                WANTED+=("$1")
                shift
            done
            ;;
        -*)
            echo "Error: unknown option '$1'." >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
        *) WANTED+=("$1"); shift ;;
    esac
done

if [ -n "$PYTHON_INPUT" ] && [ -n "$VENV_INPUT" ]; then
    echo "Error: choose only one of --python or --venv." >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/_venv_helpers.sh" ]; then
    echo "Error: vibe-qc lifecycle helpers were not found under '$SCRIPT_DIR'." >&2
    exit 1
fi
# shellcheck source=_venv_helpers.sh
. "$SCRIPT_DIR/_venv_helpers.sh"

resolve_from_repo() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)  printf '%s/%s\n' "$REPO_ROOT" "$1" ;;
    esac
}

resolve_python_command() {
    local out_var="$1"
    local requested="$2"
    local resolved=""
    local parent=""

    resolved="$(command -v "$requested" 2>/dev/null || true)"
    case "$resolved" in
        /*) ;;
        */*)
            parent="$(cd "$(dirname "$resolved")" 2>/dev/null && pwd -P)" || return 1
            resolved="$parent/$(basename "$resolved")"
            ;;
        *) return 1 ;;
    esac
    [ -x "$resolved" ] || return 1
    printf -v "$out_var" '%s' "$resolved"
}

# Read only interpreter metadata before acquiring the mutation lock. -I -S
# prevents PYTHONPATH, user site packages, .pth files, and sitecustomize from
# running. Python <= 3.13 initializes venv prefixes in site.py, so the
# pyvenv.cfg beside sys.executable is the isolated compatibility fallback.
inspect_optional_python() {
    local python_bin="$1"
    local prefix_out="$2"
    local base_python_out="$3"
    local metadata=""
    local marker=""
    local prefix=""
    local base_python=""
    local is_venv=""
    local line_count=""

    if ! metadata="$(PYTHONNOUSERSITE=1 "$python_bin" -I -S -c '
import os
import sys

prefix = os.path.realpath(sys.prefix)
base_prefix = os.path.realpath(sys.base_prefix)
executable = os.path.abspath(sys.executable)
candidate = os.path.dirname(os.path.dirname(executable))
config = os.path.join(candidate, "pyvenv.cfg")
is_venv = prefix != base_prefix
if not is_venv and os.path.isfile(config) and not os.path.islink(config):
    prefix = os.path.realpath(candidate)
    is_venv = True
base_executable = os.path.realpath(
    getattr(sys, "_base_executable", sys.executable)
)
print("VIBEQC_OPTIONAL_PYTHON_V1")
print(prefix)
print(base_executable)
print("1" if is_venv else "0")
' 2>/dev/null)"; then
        echo "Error: could not inspect selected Python '$python_bin' safely." >&2
        return 1
    fi
    line_count="$(printf '%s\n' "$metadata" | awk 'END {print NR}')"
    marker="$(printf '%s\n' "$metadata" | sed -n '1p')"
    prefix="$(printf '%s\n' "$metadata" | sed -n '2p')"
    base_python="$(printf '%s\n' "$metadata" | sed -n '3p')"
    is_venv="$(printf '%s\n' "$metadata" | sed -n '4p')"
    if [ "$line_count" != "4" ] || \
       [ "$marker" != "VIBEQC_OPTIONAL_PYTHON_V1" ] || \
       [ -z "$prefix" ] || [ -z "$base_python" ]; then
        echo "Error: selected Python returned invalid isolated metadata." >&2
        return 1
    fi
    if [ "$is_venv" != "1" ]; then
        echo "Error: '$python_bin' is not a virtualenv interpreter." >&2
        echo "Pass --venv with the vibe-qc environment; global installs are refused." >&2
        return 1
    fi
    printf -v "$prefix_out" '%s' "$prefix"
    printf -v "$base_python_out" '%s' "$base_python"
}

SELECTED_BY_PYTHON=0
RAW_VENV_TARGET=""
if [ -n "$PYTHON_INPUT" ]; then
    SELECTED_BY_PYTHON=1
    PYTHON="$PYTHON_INPUT"
elif [ -n "$VENV_INPUT" ]; then
    RAW_VENV_TARGET="$(resolve_from_repo "$VENV_INPUT")"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    RAW_VENV_TARGET="$(resolve_from_repo "$VIRTUAL_ENV")"
else
    RAW_VENV_TARGET="$REPO_ROOT/.venv"
fi

VENV_PATH=""
LOCK_PYTHON=""
if [ "$SELECTED_BY_PYTHON" = "1" ]; then
    if resolve_python_command RESOLVED_PYTHON "$PYTHON"; then
        PYTHON="$RESOLVED_PYTHON"
    elif [ "$DRY_RUN" != "1" ]; then
        echo "Error: no usable vibe-qc Python was found at '$PYTHON'." >&2
        echo "Install vibe-qc first, activate its venv, or pass --venv/--python." >&2
        exit 1
    fi
    if [ "$DRY_RUN" != "1" ]; then
        inspect_optional_python "$PYTHON" VENV_PATH BASE_PYTHON
        vibeqc_resolve_venv_path CANONICAL_VENV "$VENV_PATH"
        VENV_PATH="$CANONICAL_VENV"
        vibeqc_assert_removable_venv "$VENV_PATH" "$REPO_ROOT"
        # The probe may have been reached through a PATH shim. All health and
        # mutation commands use the interpreter anchored inside the locked
        # prefix, so replacing an out-of-prefix shim cannot redirect pip.
        PYTHON="$VENV_PATH/bin/python"
    fi
else
    vibeqc_resolve_venv_path VENV_PATH "$RAW_VENV_TARGET"
    vibeqc_assert_removable_venv "$VENV_PATH" "$REPO_ROOT"
    PYTHON="$VENV_PATH/bin/python"
fi
if [ "$DRY_RUN" != "1" ]; then
    command -v vibe_toolset_find_external_python >/dev/null 2>&1 || {
        echo "Error: shared toolset lifecycle Python helpers are unavailable." >&2
        exit 1
    }
    # Locking needs an isolated inspector, not code selected by pyvenv.cfg.
    # The target interpreter is first executed only after this lock is held.
    vibe_toolset_find_external_python LOCK_PYTHON "$VENV_PATH" 3 8
fi

if [ "$DRY_RUN" != "1" ] && [ ! -x "$PYTHON" ]; then
    echo "Error: no usable vibe-qc Python was found at '$PYTHON'." >&2
    echo "Install vibe-qc first, activate its venv, or pass --venv/--python." >&2
    exit 1
fi
PYTHON_SCRIPTS_DIR="$(dirname "$PYTHON")"

prompt_yes() {
    local question="$1"
    local response=""
    if [ "$YES" = "1" ]; then
        return 0
    fi
    if [ ! -t 0 ]; then
        echo "Error: interactive input is unavailable; pass --yes." >&2
        return 2
    fi
    read -r -p "  $question [Y/n] " response
    case "${response:-Y}" in
        [Yy]*) return 0 ;;
        *)     return 1 ;;
    esac
}

has_module() {
    [ -x "$PYTHON" ] && "$PYTHON" -c "import $1" 2>/dev/null
}

pip_install() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '  Would run: %s -m pip install --upgrade' "$PYTHON"
        printf ' %s' "$@"
        printf '\n'
        return 0
    fi
    "$PYTHON" -m pip install --upgrade "$@"
}

# Parallel arrays keep this compatible with macOS /bin/bash 3.2.
TOOL_NAMES=()
TOOL_IMPORTS=()
TOOL_PIP_SPECS=()
TOOL_COMMANDS=()
TOOL_BLURBS=()

register_tool() {
    TOOL_NAMES+=("$1")
    TOOL_IMPORTS+=("$2")
    TOOL_PIP_SPECS+=("$3")
    TOOL_COMMANDS+=("$4")
    TOOL_BLURBS+=("$5")
}

verify_tool() {
    local tool_name="$1"
    local import_name="$2"
    local command_name="$3"
    local command_path="$PYTHON_SCRIPTS_DIR/$command_name"

    if ! has_module "$import_name"; then
        echo "Error: pip returned success, but '$import_name' cannot be imported." >&2
        return 1
    fi
    if [ ! -f "$command_path" ] || [ -L "$command_path" ] || \
       [ ! -x "$command_path" ]; then
        echo "Error: pip returned success, but '$command_name' was not installed in the venv." >&2
        return 1
    fi
    if ! "$command_path" --help >/dev/null 2>&1; then
        echo "Error: the installed '$tool_name' command failed its --help check." >&2
        return 1
    fi
}

tool_is_healthy() {
    local import_name="$1"
    local command_name="$2"
    local command_path="$PYTHON_SCRIPTS_DIR/$command_name"

    has_module "$import_name" && \
        [ -f "$command_path" ] && \
        [ ! -L "$command_path" ] && \
        [ -x "$command_path" ] && \
        "$command_path" --help >/dev/null 2>&1
}

tool_index() {
    local name="$1"
    local i
    for i in "${!TOOL_NAMES[@]}"; do
        if [ "${TOOL_NAMES[$i]}" = "$name" ]; then
            printf '%s\n' "$i"
            return 0
        fi
    done
    return 1
}

register_tool \
    moltui moltui "moltui>=0.1" moltui \
    "MolTUI: terminal viewer for Molden, cube, fchk, gbw, hess, and xyz"

OFFER_IDX=()
if [ "${#WANTED[@]}" -gt 0 ]; then
    for tool_name in "${WANTED[@]}"; do
        if ! tool_idx="$(tool_index "$tool_name")"; then
            echo "Error: unknown tool '$tool_name'." >&2
            echo "Known tools: ${TOOL_NAMES[*]}" >&2
            exit 1
        fi
        OFFER_IDX+=("$tool_idx")
    done
else
    for tool_idx in "${!TOOL_NAMES[@]}"; do
        OFFER_IDX+=("$tool_idx")
    done
fi

optional_tools_cleanup() {
    local status="$?"
    local cleanup_status=0

    set +e
    vibe_toolset_release_lifecycle_lock || cleanup_status=$?
    if [ "$status" = "0" ] && [ "$cleanup_status" != "0" ]; then
        status="$cleanup_status"
    fi
    return "$status"
}

if [ "$DRY_RUN" != "1" ]; then
    vibe_toolset_acquire_lifecycle_lock \
        "$LOCK_PYTHON" "$REPO_ROOT" "$VENV_PATH" "vibe-qc optional tools"
    trap optional_tools_cleanup EXIT

    # Repeat every path and interpreter proof while the common lock is held,
    # before the first health import or pip execution. A stale command symlink
    # or target swap must fail closed instead of redirecting the mutation.
    vibeqc_resolve_venv_path LOCKED_VENV "$VENV_PATH"
    vibeqc_assert_removable_venv "$LOCKED_VENV" "$REPO_ROOT"
    if [ "$LOCKED_VENV" != "$VENV_PATH" ] || [ ! -x "$PYTHON" ] || \
       [ ! -d "$VENV_PATH/bin" ] || [ -L "$VENV_PATH/bin" ]; then
        echo "Error: selected vibe-qc environment changed before it was locked." >&2
        exit 1
    fi
    inspect_optional_python "$PYTHON" LOCKED_PREFIX LOCKED_BASE_PYTHON
    vibeqc_resolve_venv_path LOCKED_PREFIX "$LOCKED_PREFIX"
    if [ "$LOCKED_PREFIX" != "$VENV_PATH" ]; then
        echo "Error: selected Python no longer belongs to the locked virtualenv." >&2
        exit 1
    fi
    case "$LOCKED_BASE_PYTHON" in
        /*) ;;
        *) echo "Error: selected virtualenv reported an invalid base Python." >&2; exit 1 ;;
    esac
    case "$LOCKED_BASE_PYTHON" in
        "$VENV_PATH"|"$VENV_PATH"/*)
            echo "Error: selected virtualenv reported a target-contained base Python." >&2
            exit 1
            ;;
    esac
    vibeqc_assert_python "$PYTHON"
    if ! "$PYTHON" -c 'import vibeqc' >/dev/null 2>&1; then
        echo "Error: vibe-qc is not installed in the selected environment." >&2
        echo "Run ./scripts/install.sh first, then retry this helper." >&2
        exit 1
    fi
    REPORTED_SCRIPTS_DIR="$("$PYTHON" -c \
        'import sysconfig; print(sysconfig.get_path("scripts"))' \
        2>/dev/null || true)"
    if [ -z "$REPORTED_SCRIPTS_DIR" ] || \
       ! PYTHON_SCRIPTS_DIR="$(cd "$REPORTED_SCRIPTS_DIR" 2>/dev/null && pwd -P)"; then
        echo "Error: selected Python reported an invalid scripts directory." >&2
        exit 1
    fi
    EXPECTED_SCRIPTS_DIR="$(cd "$VENV_PATH/bin" && pwd -P)"
    if [ "$PYTHON_SCRIPTS_DIR" != "$EXPECTED_SCRIPTS_DIR" ]; then
        echo "Error: selected Python reported a scripts directory outside the locked virtualenv." >&2
        exit 1
    fi
fi

echo "==> Optional vibe-qc tools"
echo "    python:  $PYTHON"
[ -z "$VENV_PATH" ] || echo "    venv:    $VENV_PATH"
[ "$UPGRADE" = "1" ] && echo "    mode:    install missing tools and upgrade installed tools"
[ "$DRY_RUN" = "1" ] && echo "    dry-run: no files will be changed"
echo

CHANGED=()
SKIPPED=()
for tool_idx in "${OFFER_IDX[@]}"; do
    tool_name="${TOOL_NAMES[$tool_idx]}"
    import_name="${TOOL_IMPORTS[$tool_idx]}"
    pip_spec="${TOOL_PIP_SPECS[$tool_idx]}"
    command_name="${TOOL_COMMANDS[$tool_idx]}"
    blurb="${TOOL_BLURBS[$tool_idx]}"
    installed=0
    if [ "$DRY_RUN" != "1" ]; then
        if tool_is_healthy "$import_name" "$command_name"; then
            installed=1
        elif has_module "$import_name"; then
            echo "Existing $tool_name installation is incomplete; repairing it."
        fi
    fi

    if [ "$installed" = "1" ] && [ "$UPGRADE" != "1" ]; then
        echo "Already installed: $tool_name (use --upgrade to refresh it)."
        SKIPPED+=("$tool_name")
        continue
    fi

    echo "$blurb"
    action="Install"
    [ "$installed" = "1" ] && action="Upgrade"
    if [ "$DRY_RUN" = "1" ] || prompt_yes "$action $tool_name?"; then
        pip_install "$pip_spec"
        if [ "$DRY_RUN" != "1" ]; then
            verify_tool "$tool_name" "$import_name" "$command_name"
        fi
        CHANGED+=("$tool_name")
    else
        prompt_status=$?
        [ "$prompt_status" -ne 2 ] || exit 1
        SKIPPED+=("$tool_name")
    fi
    echo
done

if [ "$DRY_RUN" = "1" ]; then
    echo "==> Dry-run complete."
else
    echo "==> Optional-tool setup complete."
fi
[ "${#CHANGED[@]}" -eq 0 ] || echo "    selected: ${CHANGED[*]}"
[ "${#SKIPPED[@]}" -eq 0 ] || echo "    unchanged/skipped: ${SKIPPED[*]}"
echo
echo "Standalone suite tools use their own lifecycle scripts:"
echo "    vibe-view:  ./vibe-view/scripts/install.sh"
echo "    vq:         ./vibe-queue/scripts/install.sh"
echo "    vibe-basis: ./vibe-basis/scripts/install.sh"

if [ "$DRY_RUN" != "1" ]; then
    vibe_toolset_release_lifecycle_lock
    trap - EXIT
fi
