# scripts/_venv_helpers.sh — venv detection + health-check helpers.
#
# Sourced (NOT executed) by install.sh, update.sh, and doctor.sh.
# Centralises the venv state-machine so the three entry points agree on
# what "broken venv" means and how to recover.
#
# Functions exposed:
#
#   vibeqc_detect_venv VAR_NAME [EXPLICIT_PATH]
#       Sets VAR_NAME (via printf -v) to a venv path. If EXPLICIT_PATH
#       is non-empty, validates that and uses it. Otherwise tries the
#       conventional candidates in order: .venv venv .venv-vibeqc
#       venv-vibeqc. Sets VAR_NAME to the empty string if nothing
#       suitable is found.
#
#   vibeqc_check_venv_health VENV_PATH
#       Runs three health checks against an existing venv:
#         (a) $VENV/bin/python actually runs
#         (b) $VENV/bin/pip wrapper's shebang resolves
#         (c) pyvenv.cfg's `version =` matches `$VENV/bin/python -V`
#       Prints a diagnostic + recovery recipe on the first failure and
#       returns non-zero. Returns 0 if all three pass. The caller
#       decides whether to abort or auto-recreate.
#
#   vibeqc_resolve_venv_path VAR_NAME PATH
#       Resolves PATH without following a final symlink. The parent directory
#       must already exist. All destructive operations use the resolved path.
#
#   vibeqc_assert_removable_venv VENV_PATH [REPO_ROOT]
#       Refuses dangerous targets, symlinks, and existing directories that do
#       not carry a regular pyvenv.cfg marker. This is only the structural
#       guard; user-facing destructive commands must additionally call
#       vibeqc_assert_owned_venv.
#
#   vibeqc_assert_owned_venv VENV_PATH REPO_ROOT TRUSTED_PYTHON [ALLOW_LEGACY]
#       Requires this checkout's lifecycle marker. With ALLOW_LEGACY=1, an
#       unmarked pre-marker install is accepted only when regular PEP 610
#       metadata points exactly at REPO_ROOT. TRUSTED_PYTHON must be external
#       to VENV_PATH; code from the candidate environment is never executed.
#
#   vibeqc_begin_venv_replacement / vibeqc_start_venv_replacement /
#   vibeqc_abort_venv_replacement / vibeqc_commit_venv_replacement
#       Failure-atomic replacement transaction. The old environment is moved
#       to a same-parent backup and restored on any later failure.
#
#   vibeqc_create_venv VENV_PATH PYTHON_BIN
#       Creates a venv inside an active replacement transaction. It never
#       removes the target itself.
#
#   vibeqc_pip_upgrade VENV_PATH
#       Upgrades pip inside VENV_PATH using ``python -m pip`` (sidesteps
#       any stale shebang on the bare pip wrapper).
#
#   vibeqc_pip_install_editable VENV_PATH EXTRAS_SPEC [REPO_ROOT]
#       Runs `pip install -e "$REPO_ROOT$EXTRAS_SPEC"`. EXTRAS_SPEC is
#       passed verbatim — pass "[test]" / "[dev]" / "" depending on what
#       the caller wants. REPO_ROOT defaults to ``.`` (current dir).
#
# All functions assume `set -euo pipefail` in the caller. None of them
# `exit` on their own — they return non-zero so the caller controls
# control flow.

_VIBEQC_VENV_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=_lifecycle_lock.sh
. "$_VIBEQC_VENV_HELPER_DIR/_lifecycle_lock.sh"
unset _VIBEQC_VENV_HELPER_DIR

# ---------------------------------------------------------------------------
# vibeqc_detect_venv VAR [PATH]
# ---------------------------------------------------------------------------
vibeqc_detect_venv() {
    local _vh_out="$1"
    local _vh_explicit="${2:-}"
    local _vh_found=""

    if [ -n "$_vh_explicit" ]; then
        if [ -x "$_vh_explicit/bin/python" ]; then
            _vh_found="$_vh_explicit"
        fi
    else
        local candidate
        for candidate in .venv venv .venv-vibeqc venv-vibeqc; do
            if [ -x "$candidate/bin/python" ]; then
                _vh_found="$candidate"
                break
            fi
        done
    fi

    printf -v "$_vh_out" '%s' "$_vh_found"
}

# Select an existing target even when its interpreter is broken. This is used
# by --recreate-venv and uninstall; vibeqc_detect_venv intentionally remains
# restricted to runnable interpreters for doctor.sh and ordinary updates.
vibeqc_detect_venv_target() {
    local _vh_out="$1"
    local _vh_explicit="${2:-}"
    local _vh_found=""

    if [ -n "$_vh_explicit" ]; then
        if [ -e "$_vh_explicit" ] || [ -L "$_vh_explicit" ]; then
            _vh_found="$_vh_explicit"
        fi
    else
        local candidate
        for candidate in .venv venv .venv-vibeqc venv-vibeqc; do
            if [ -e "$candidate" ] || [ -L "$candidate" ]; then
                _vh_found="$candidate"
                break
            fi
        done
    fi

    printf -v "$_vh_out" '%s' "$_vh_found"
}

# Resolve a target through its physical parent without following the final
# path component. Requiring an existing parent keeps this safety operation
# read-only and makes symlink rejection unambiguous on both macOS and Linux.
vibeqc_resolve_venv_path() {
    local _vh_out="$1"
    local _vh_raw="$2"
    local _vh_parent _vh_name _vh_parent_real _vh_resolved

    if [ -z "$_vh_raw" ]; then
        echo "Error: virtualenv path must not be empty." >&2
        return 2
    fi
    if [ -L "$_vh_raw" ]; then
        echo "Error: refusing to operate on a virtualenv through a symbolic link:" >&2
        echo "       $_vh_raw" >&2
        return 1
    fi

    _vh_parent="$(dirname "$_vh_raw")"
    _vh_name="$(basename "$_vh_raw")"
    case "$_vh_name" in
        ""|.|..|/) echo "Error: unsafe virtualenv path '$_vh_raw'." >&2; return 1 ;;
    esac
    if ! _vh_parent_real="$(cd -P "$_vh_parent" 2>/dev/null && pwd)"; then
        echo "Error: parent directory for virtualenv does not exist: $_vh_parent" >&2
        echo "Create the parent directory explicitly, then re-run." >&2
        return 1
    fi
    if [ "$_vh_parent_real" = "/" ]; then
        _vh_resolved="/$_vh_name"
    else
        _vh_resolved="$_vh_parent_real/$_vh_name"
    fi
    printf -v "$_vh_out" '%s' "$_vh_resolved"
}

vibeqc_assert_safe_venv_target() {
    local target="$1"
    local repo_root="${2:-}"
    local home_real=""
    local home_uid=""
    local scripts_real=""
    local safe_user_location=0

    if [ -n "${HOME:-}" ]; then
        home_real="$(cd -P "$HOME" 2>/dev/null && pwd || true)"
        if [ -n "$home_real" ]; then
            home_uid="$(stat -c '%u' "$home_real" 2>/dev/null || \
                stat -f '%u' "$home_real" 2>/dev/null || true)"
        fi
    fi
    if [ -n "$repo_root" ] && [ -d "$repo_root/scripts" ]; then
        scripts_real="$(cd -P "$repo_root/scripts" && pwd)"
    fi

    case "$target" in
        ""|/|/Applications|/Library|/System|/Users|/Volumes|/bin|/boot|/dev|/etc|/home|/nix|/opt|/private|/private/tmp|/proc|/root|/run|/sbin|/snap|/srv|/sys|/tmp|/usr|/var|"$repo_root"|"$scripts_real"|"$home_real")
            echo "Error: refusing unsafe virtualenv target '$target'." >&2
            return 1
            ;;
    esac

    # A canonical, current-user-owned account home and known temporary/runtime
    # roots are valid places for isolated environments. Do not trust arbitrary
    # HOME, TMPDIR, or XDG values: they could point at another user's home or a
    # protected system location and defeat this guard.
    if [ "$home_uid" = "$EUID" ]; then
        case "$home_real" in
            /Users/*|/home/*|/var/home/*|/root)
                case "$target" in
                    "$home_real"/*) safe_user_location=1 ;;
                esac
                ;;
        esac
    fi
    case "$target" in
        /tmp/*|/private/tmp/*|/var/tmp/*|/private/var/tmp/*|/var/folders/*|/private/var/folders/*|/run/user/"$EUID"/*)
            safe_user_location=1
            ;;
    esac

    case "$target" in
        .git|.git/*|*/.git|*/.git/*)
            echo "Error: refusing virtualenv target '$target': it is inside Git metadata." >&2
            return 1
            ;;
    esac
    if [ "$safe_user_location" != "1" ]; then
        case "$target" in
            /Applications/*|/Library/*|/System/*|/Users/*|/bin/*|/boot/*|/dev/*|/etc/*|/home/*|/nix/*|/opt/*|/private/*|/proc/*|/root/*|/run/*|/sbin/*|/snap/*|/srv/*|/sys/*|/usr/*|/var/*)
                echo "Error: refusing unsafe virtualenv target inside a protected system location: '$target'." >&2
                return 1
                ;;
        esac
    fi

    # Never allow a target that contains the checkout. Even a directory with
    # a stray pyvenv.cfg must not be able to erase this repository or one of
    # its parents. A venv *inside* the checkout remains the normal case.
    if [ -n "$repo_root" ]; then
        case "$repo_root/" in
            "$target/"*)
                echo "Error: refusing virtualenv target '$target': it contains the checkout." >&2
                return 1
                ;;
        esac
    fi
    if [ -n "$home_real" ]; then
        case "$home_real/" in
            "$target/"*)
                echo "Error: refusing virtualenv target '$target': it contains the home directory." >&2
                return 1
                ;;
        esac
    fi
    if [ -e "$target" ] && [ ! -d "$target" ]; then
        echo "Error: virtualenv target exists but is not a directory: $target" >&2
        return 1
    fi
}

vibeqc_assert_removable_venv() {
    local target="$1"
    local repo_root="${2:-}"

    vibeqc_assert_safe_venv_target "$target" "$repo_root" || return
    if [ -L "$target" ]; then
        echo "Error: refusing to remove a virtualenv through a symbolic link:" >&2
        echo "       $target" >&2
        return 1
    fi
    if [ ! -e "$target" ]; then
        return 0
    fi
    if [ ! -d "$target" ] || [ ! -f "$target/pyvenv.cfg" ] || \
       [ -L "$target/pyvenv.cfg" ]; then
        echo "Error: refusing to remove '$target': it is not recognisably a virtualenv." >&2
        echo "Move or remove it manually after checking its contents." >&2
        return 1
    fi
}

# Resolve an interpreter path without executing it. Destructive lifecycle
# commands use this before reading ownership metadata so an activated target
# environment cannot smuggle its own python3 into the proof step.
vibeqc_command_path() {
    local _vh_out="$1"
    local command_name="$2"
    local resolved_command=""
    local link_value=""
    local link_dir=""
    local hops=0

    resolved_command="$(command -v "$command_name" 2>/dev/null || true)"
    case "$resolved_command" in
        /*) ;;
        */*)
            if ! resolved_command="$(cd "$(dirname "$resolved_command")" 2>/dev/null && pwd -P)/$(basename "$resolved_command")"; then
                echo "Error: could not resolve Python interpreter '$command_name'." >&2
                return 1
            fi
            ;;
        *)
            echo "Error: Python interpreter '$command_name' was not found as an executable path." >&2
            return 1
            ;;
    esac
    [ -x "$resolved_command" ] || {
        echo "Error: Python interpreter '$resolved_command' is not executable." >&2
        return 1
    }

    while [ -L "$resolved_command" ]; do
        [ "$hops" -lt 40 ] || {
            echo "Error: too many symbolic links while resolving '$command_name'." >&2
            return 1
        }
        link_value="$(readlink "$resolved_command")"
        link_dir="$(cd "$(dirname "$resolved_command")" && pwd -P)"
        case "$link_value" in
            /*) resolved_command="$link_value" ;;
            *)  resolved_command="$link_dir/$link_value" ;;
        esac
        resolved_command="$(cd "$(dirname "$resolved_command")" && pwd -P)/$(basename "$resolved_command")"
        hops=$((hops + 1))
    done
    resolved_command="$(cd "$(dirname "$resolved_command")" && pwd -P)/$(basename "$resolved_command")"
    printf -v "$_vh_out" '%s' "$resolved_command"
}

vibeqc_select_trusted_python() {
    local _vh_out="$1"
    local requested="$2"
    local protected_target="$3"
    local selected_path=""
    local lexical_target="$protected_target"

    vibeqc_command_path selected_path "$requested" || return
    while [ "$lexical_target" != "/" ]; do
        case "$lexical_target" in
            */.) lexical_target="${lexical_target%/.}" ;;
            */)  lexical_target="${lexical_target%/}" ;;
            *)   break ;;
        esac
    done
    if [ -L "$lexical_target" ]; then
        echo "Error: refusing to replace/remove a virtualenv through a symbolic link:" >&2
        echo "       $lexical_target" >&2
        return 1
    fi
    if [ -d "$lexical_target" ]; then
        lexical_target="$(cd "$lexical_target" && pwd -P)"
    fi
    case "$selected_path" in
        "$lexical_target"|"$lexical_target"/*)
            echo "Error: refusing to use Python from the virtualenv selected for replacement/removal:" >&2
            echo "       $selected_path" >&2
            echo "Deactivate it or pass --python with an external interpreter." >&2
            return 1
            ;;
    esac
    vibeqc_assert_python "$selected_path" || return
    printf -v "$_vh_out" '%s' "$selected_path"
}

vibeqc_mark_owned_venv() {
    local venv="$1"
    local repo_root="$2"
    local trusted_python="$3"

    PYTHONNOUSERSITE=1 "$trusted_python" -I -S - \
        "$venv" "$repo_root" <<'PY'
import json
import os
import sys
from pathlib import Path

venv = Path(sys.argv[1])
project = Path(sys.argv[2]).resolve()
marker = venv / ".vibeqc-lifecycle.json"
temporary = venv / f".vibeqc-lifecycle.{os.getpid()}.tmp"
if marker.is_symlink() or (marker.exists() and not marker.is_file()):
    raise SystemExit("refusing an unsafe vibe-qc lifecycle marker")
temporary.write_text(
    json.dumps(
        {
            "schema": 1,
            "kind": "vibe-qc-lifecycle-venv",
            "project": str(project),
        },
        indent=2,
    )
    + "\n"
)
os.replace(temporary, marker)
PY
}

vibeqc_assert_owned_venv() {
    local venv="$1"
    local repo_root="$2"
    local trusted_python="$3"
    local allow_legacy_pep610="${4:-0}"
    local proof_rc=0

    vibeqc_assert_removable_venv "$venv" "$repo_root" || return
    [ -e "$venv" ] || return 0
    PYTHONNOUSERSITE=1 "$trusted_python" -I -S - \
        "$venv" "$repo_root" "$allow_legacy_pep610" <<'PY' || proof_rc=$?
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

venv = Path(sys.argv[1]).resolve()
project = Path(sys.argv[2]).resolve()
allow_legacy_pep610 = sys.argv[3] == "1"
marker = venv / ".vibeqc-lifecycle.json"

if marker.is_symlink():
    raise SystemExit(1)
if marker.exists():
    if not marker.is_file():
        raise SystemExit(1)
    try:
        payload = json.loads(marker.read_text())
        raw_project = Path(payload["project"])
        if not raw_project.is_absolute():
            raise ValueError("marker project must be absolute")
        marker_project = raw_project.resolve()
    except (OSError, UnicodeError, KeyError, TypeError, ValueError):
        raise SystemExit(1)
    if (
        payload.get("schema") == 1
        and payload.get("kind") == "vibe-qc-lifecycle-venv"
        and marker_project == project
    ):
        raise SystemExit(0)
    # An existing marker is authoritative. Never reinterpret a foreign or
    # malformed marked environment as an adoptable legacy installation.
    raise SystemExit(1)

if not allow_legacy_pep610:
    raise SystemExit(1)

patterns = (
    "lib/python*/site-packages/vibe_qc-*.dist-info/direct_url.json",
    "lib64/python*/site-packages/vibe_qc-*.dist-info/direct_url.json",
    "Lib/site-packages/vibe_qc-*.dist-info/direct_url.json",
)
records = []
for pattern in patterns:
    records.extend(venv.glob(pattern))
for record in records:
    if record.is_symlink() or not record.is_file():
        continue
    try:
        payload = json.loads(record.read_text())
        url = payload.get("url")
        if not isinstance(url, str):
            continue
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
            continue
        source = Path(urllib.request.url2pathname(parsed.path))
        if source.resolve() == project:
            raise SystemExit(0)
    except (OSError, UnicodeError, TypeError, ValueError):
        continue
raise SystemExit(1)
PY
    if [ "$proof_rc" -ne 0 ]; then
        echo "Error: refusing to remove or replace '$venv': ownership is not proven." >&2
        echo "A current install must carry this checkout's .vibeqc-lifecycle.json marker." >&2
        if [ "$allow_legacy_pep610" = "1" ]; then
            echo "No trusted PEP 610 record linked this legacy environment to this checkout." >&2
        else
            echo "For an unmarked legacy install from this checkout, re-run with --adopt-legacy." >&2
        fi
        echo "A foreign or invalid ownership marker is never adopted." >&2
        return 1
    fi
}

vibeqc_remove_venv() {
    local target="$1"
    local repo_root="${2:-}"

    vibeqc_assert_removable_venv "$target" "$repo_root" || return
    [ -e "$target" ] || return 0
    echo "==> Removing virtualenv: $target"
    rm -rf -- "$target"
}

vibeqc_acquire_lifecycle_lock() {
    local target="$1"
    local repo_root="$2"
    local action="$3"
    local requested_python="${4:-python3}"
    local canonical_target=""
    local lock_python=""

    vibeqc_resolve_venv_path canonical_target "$target" || return
    vibeqc_assert_safe_venv_target "$canonical_target" "$repo_root" || return
    vibe_toolset_select_external_python \
        lock_python "$requested_python" "$canonical_target" || return
    vibe_toolset_acquire_lifecycle_lock \
        "$lock_python" "$repo_root" "$canonical_target" "vibe-qc $action"
}

vibeqc_release_lifecycle_lock() {
    vibe_toolset_release_lifecycle_lock
}

vibeqc_lifecycle_cleanup() {
    local status="$?"
    local cleanup_status=0

    set +e
    vibeqc_abort_venv_replacement || cleanup_status=$?
    if ! vibeqc_release_lifecycle_lock; then
        [ "$cleanup_status" != "0" ] || cleanup_status=1
    fi
    if [ "$status" = "0" ] && [ "$cleanup_status" != "0" ]; then
        status="$cleanup_status"
    fi
    return "$status"
}

vibeqc_assert_python() {
    local python_bin="$1"

    if ! command -v "$python_bin" >/dev/null 2>&1; then
        echo "Error: Python interpreter '$python_bin' was not found." >&2
        return 1
    fi
    if ! PYTHONNOUSERSITE=1 "$python_bin" -I -S -c \
        'import sys; raise SystemExit(sys.version_info < (3, 11))' \
        >/dev/null 2>&1; then
        echo "Error: vibe-qc requires Python 3.11 or newer." >&2
        echo "Found: $(PYTHONNOUSERSITE=1 "$python_bin" -I -S --version 2>&1 || true)" >&2
        return 1
    fi
}

# Resolve a requested interpreter to its external base executable. This keeps
# --python "$VENV/bin/python" from disappearing halfway through replacement
# when the old venv is moved aside, and avoids accidentally creating a nested
# virtualenv from another environment.
vibeqc_resolve_base_python() {
    local _vh_out="$1"
    local requested="$2"
    local _vh_resolved=""

    vibeqc_assert_python "$requested" || return
    _vh_resolved="$(PYTHONNOUSERSITE=1 "$requested" -I -S -c \
        'import os, sys; print(os.path.realpath(sys._base_executable))' \
        2>/dev/null || true)"
    if [ -z "$_vh_resolved" ] || [ ! -x "$_vh_resolved" ]; then
        echo "Error: could not resolve the base executable for '$requested'." >&2
        return 1
    fi
    printf -v "$_vh_out" '%s' "$_vh_resolved"
}

# Read the recorded external base executable without running code from the
# environment selected for replacement. Python 3.11+ records this executable
# in pyvenv.cfg, which also lets a user repair a broken environment without
# silently switching to a different python3 on PATH.
vibeqc_venv_base_python() {
    local _vh_out="$1"
    local venv="$2"
    local _vh_found=""

    if [ -f "$venv/pyvenv.cfg" ] && [ ! -L "$venv/pyvenv.cfg" ]; then
        _vh_found="$(awk -F'[[:space:]]*=[[:space:]]*' \
            '/^executable[[:space:]]*=/ {print $2; exit}' \
            "$venv/pyvenv.cfg" 2>/dev/null || true)"
    fi
    if [ -z "$_vh_found" ] || [ ! -x "$_vh_found" ]; then
        return 1
    fi
    printf -v "$_vh_out" '%s' "$_vh_found"
}

# ---------------------------------------------------------------------------
# vibeqc_check_venv_health VENV
#
# Returns 0 if the venv is in a usable state, non-zero otherwise.
# Prints diagnostics to stderr on failure.
# ---------------------------------------------------------------------------
vibeqc_check_venv_health() {
    local venv="$1"

    if [ -z "$venv" ] || [ ! -d "$venv" ]; then
        echo "vibeqc_check_venv_health: '$venv' is not a directory." >&2
        return 2
    fi

    # (a) python runs
    if ! PYTHONNOUSERSITE=1 "$venv/bin/python" -I -S -c \
        'import sys' >/dev/null 2>&1; then
        cat >&2 <<EOF

Error: $venv/bin/python doesn't run.
The venv is broken. Recreate it:
    rm -rf $venv && python3 -m venv $venv
    $venv/bin/python -m pip install -e '.[test]'

Or run update.sh --recreate-venv (or install.sh --force) to do this automatically.
EOF
        return 1
    fi

    # (b) pip shebang is reachable. Direct invocation of $VENV/bin/pip
    # fails when the script's absolute-path shebang points at a moved /
    # deleted python (most commonly when a venv was copied across trees).
    # update.sh itself works because it goes through `python -m pip`, but
    # the user's day-to-day `$VENV/bin/pip` invocations would fail with a
    # confusing "bad interpreter" error.
    if [ -x "$venv/bin/pip" ] && ! "$venv/bin/pip" --version >/dev/null 2>&1; then
        local shebang
        shebang=$(head -n 1 "$venv/bin/pip" | sed 's|^#!||')
        cat >&2 <<EOF

Warning: $venv/bin/pip is broken (stale shebang).
    shebang points at: $shebang
    expected:          $(pwd)/$venv/bin/python
Most common cause: this venv was copied from another tree.
Fix:
    rm -rf $venv && python3 -m venv $venv
    $venv/bin/python -m pip install -e '.[test]'

Or re-run with --recreate-venv to do this automatically.
EOF
        return 1
    fi

    # (c) pyvenv.cfg vs runtime version mismatch. Happens when
    # `python3 -m venv $VENV` is rerun on top of an existing venv with a
    # *different* python3 on PATH — the generic `python` / `python3`
    # symlinks get replaced (often pointing at Apple's stock
    # /usr/bin/python3 = 3.9.x) while pyvenv.cfg still claims the original
    # version. Subsequent `pip install -e .` fails with a confusing
    #     ERROR: Package 'vibe-qc' requires a different Python: 3.9.6 not in '>=3.11'
    if [ -f "$venv/pyvenv.cfg" ]; then
        local cfg_ver bin_ver cfg_majorminor
        cfg_ver=$(awk -F'[[:space:]]*=[[:space:]]*' '/^version[[:space:]]*=/ {print $2; exit}' "$venv/pyvenv.cfg" | tr -d '[:space:]')
        bin_ver=$(PYTHONNOUSERSITE=1 "$venv/bin/python" -I -S -c \
            'import sys; print("%d.%d.%d" % sys.version_info[:3])' \
            2>/dev/null || echo "")
        if [ -n "$cfg_ver" ] && [ -n "$bin_ver" ] && [ "$cfg_ver" != "$bin_ver" ]; then
            cfg_majorminor=$(echo "$cfg_ver" | awk -F. '{print $1"."$2}')
            cat >&2 <<EOF

Error: $venv is in a broken hybrid state.
    $venv/pyvenv.cfg says python $cfg_ver
    $venv/bin/python actually reports $bin_ver

Likely cause: 'python3 -m venv $venv' was rerun on top of an existing
venv with a *different* python3 on PATH. venv silently replaces the
generic python / python3 symlinks (typically to Apple's stock
/usr/bin/python3, currently 3.9.x and below vibe-qc's >=3.11 minimum)
without touching pyvenv.cfg.

Recreate the venv:
    rm -rf $venv
    python${cfg_majorminor} -m venv $venv   # or: python3 -m venv $venv
    $venv/bin/python -m pip install -e '.[test]'

Or re-run with --recreate-venv to do this automatically.
EOF
            return 1
        fi
    fi

    return 0
}

# ---------------------------------------------------------------------------
# Failure-atomic virtualenv replacement
# ---------------------------------------------------------------------------
# Python virtualenvs embed their final path, so a replacement cannot be built
# elsewhere and moved into place. Instead, move the old venv to a same-parent
# backup, build at the final path, and keep an ownership token in the partial
# replacement. The EXIT trap may remove only a target carrying that token.

VIBEQC_VENV_TX_ACTIVE=0
VIBEQC_VENV_TX_TARGET=""
VIBEQC_VENV_TX_REPO_ROOT=""
VIBEQC_VENV_TX_BACKUP=""
VIBEQC_VENV_TX_TOKEN=""
VIBEQC_VENV_TX_HAD_ORIGINAL=0
VIBEQC_VENV_TX_MUTATION_STARTED=0

vibeqc_begin_venv_replacement() {
    local target="$1"
    local repo_root="${2:-}"
    local expected_had_original="${3:-}"
    local actual_had_original=0
    local backup=""

    if [ "$VIBEQC_VENV_TX_ACTIVE" = "1" ]; then
        echo "Error: a virtualenv replacement transaction is already active." >&2
        return 1
    fi
    case "$expected_had_original" in
        0|1) ;;
        *)
            echo "Error: virtualenv replacement requires an expected target state." >&2
            return 1
            ;;
    esac
    vibeqc_assert_removable_venv "$target" "$repo_root" || return
    if [ -e "$target" ] || [ -L "$target" ]; then
        actual_had_original=1
    fi
    if [ "$actual_had_original" != "$expected_had_original" ]; then
        echo "Error: virtualenv target changed before replacement could begin:" >&2
        echo "       $target" >&2
        return 1
    fi

    if [ "$actual_had_original" = "1" ]; then
        if ! backup="$(mktemp -d "${target}.previous.XXXXXX")"; then
            echo "Error: could not reserve a rollback path beside '$target'." >&2
            return 1
        fi
        rmdir "$backup"
    fi

    VIBEQC_VENV_TX_ACTIVE=1
    VIBEQC_VENV_TX_TARGET="$target"
    VIBEQC_VENV_TX_REPO_ROOT="$repo_root"
    VIBEQC_VENV_TX_BACKUP="$backup"
    VIBEQC_VENV_TX_TOKEN="${BASHPID:-$$}.${RANDOM}.${RANDOM}"
    VIBEQC_VENV_TX_HAD_ORIGINAL="$actual_had_original"
    VIBEQC_VENV_TX_MUTATION_STARTED=0
}

vibeqc_start_venv_replacement() {
    local target="$1"
    local repo_root="${2:-}"
    local trusted_python="${3:-}"
    local allow_legacy_pep610="${4:-0}"
    local actual_had_original=0
    local marker

    if [ "$VIBEQC_VENV_TX_ACTIVE" != "1" ] || \
       [ "$target" != "$VIBEQC_VENV_TX_TARGET" ]; then
        echo "Error: virtualenv replacement does not match the active transaction." >&2
        return 1
    fi
    if [ "$repo_root" != "$VIBEQC_VENV_TX_REPO_ROOT" ]; then
        echo "Error: virtualenv replacement ownership scope changed during the transaction." >&2
        return 1
    fi
    if [ "$VIBEQC_VENV_TX_MUTATION_STARTED" = "1" ]; then
        echo "Error: virtualenv replacement has already started." >&2
        return 1
    fi
    if [ -e "$target" ] || [ -L "$target" ]; then
        actual_had_original=1
    fi
    if [ "$actual_had_original" != "$VIBEQC_VENV_TX_HAD_ORIGINAL" ]; then
        echo "Error: virtualenv target changed before replacement mutation:" >&2
        echo "       $target" >&2
        return 1
    fi
    if [ "$actual_had_original" = "1" ]; then
        [ -n "$trusted_python" ] || {
            echo "Error: virtualenv replacement requires a trusted ownership inspector." >&2
            return 1
        }
        vibeqc_assert_owned_venv \
            "$target" "$repo_root" "$trusted_python" "$allow_legacy_pep610" || return
    fi

    VIBEQC_VENV_TX_MUTATION_STARTED=1
    if [ "$VIBEQC_VENV_TX_HAD_ORIGINAL" = "1" ]; then
        mv "$target" "$VIBEQC_VENV_TX_BACKUP"
    fi
    mkdir "$target"
    marker="$target/.vibeqc-venv-transaction"
    printf '%s\n' "$VIBEQC_VENV_TX_TOKEN" > "$marker"
}

vibeqc_abort_venv_replacement() {
    local marker
    local marker_token=""
    local original_still_in_place=0
    local restore_failed=0

    [ "$VIBEQC_VENV_TX_ACTIVE" = "1" ] || return 0
    if [ "$VIBEQC_VENV_TX_MUTATION_STARTED" = "1" ]; then
        echo "==> Virtualenv replacement failed; restoring the previous state..." >&2
        marker="$VIBEQC_VENV_TX_TARGET/.vibeqc-venv-transaction"
        if [ -f "$marker" ]; then
            marker_token="$(sed -n '1p' "$marker" 2>/dev/null || true)"
        fi
        if [ "$VIBEQC_VENV_TX_HAD_ORIGINAL" = "1" ] && \
           [ -e "$VIBEQC_VENV_TX_TARGET" ] && \
           [ ! -e "$marker" ] && \
           [ ! -e "$VIBEQC_VENV_TX_BACKUP" ]; then
            original_still_in_place=1
        fi
        if [ -e "$VIBEQC_VENV_TX_TARGET" ]; then
            if [ "$original_still_in_place" = "1" ]; then
                :
            elif [ "$marker_token" = "$VIBEQC_VENV_TX_TOKEN" ]; then
                if vibeqc_assert_safe_venv_target \
                    "$VIBEQC_VENV_TX_TARGET" "$VIBEQC_VENV_TX_REPO_ROOT"; then
                    rm -rf -- "$VIBEQC_VENV_TX_TARGET"
                else
                    restore_failed=1
                fi
            elif ! rmdir "$VIBEQC_VENV_TX_TARGET" 2>/dev/null; then
                echo "Error: refusing to remove an unowned replacement target:" >&2
                echo "       $VIBEQC_VENV_TX_TARGET" >&2
                restore_failed=1
            fi
        fi
        if [ -n "$VIBEQC_VENV_TX_BACKUP" ] && \
           [ -e "$VIBEQC_VENV_TX_BACKUP" ]; then
            if [ "$restore_failed" = "0" ] && \
               ! mv "$VIBEQC_VENV_TX_BACKUP" "$VIBEQC_VENV_TX_TARGET"; then
                restore_failed=1
            fi
        fi
    fi

    VIBEQC_VENV_TX_ACTIVE=0
    if [ "$restore_failed" = "1" ]; then
        echo "Error: automatic virtualenv rollback was incomplete." >&2
        if [ -n "$VIBEQC_VENV_TX_BACKUP" ]; then
            echo "The previous environment remains at: $VIBEQC_VENV_TX_BACKUP" >&2
        fi
    fi
    return "$restore_failed"
}

vibeqc_commit_venv_replacement() {
    local backup="$VIBEQC_VENV_TX_BACKUP"
    local marker="$VIBEQC_VENV_TX_TARGET/.vibeqc-venv-transaction"

    if [ "$VIBEQC_VENV_TX_ACTIVE" != "1" ] || \
       [ "$VIBEQC_VENV_TX_MUTATION_STARTED" != "1" ]; then
        echo "Error: cannot commit an incomplete virtualenv replacement." >&2
        return 1
    fi
    vibeqc_check_venv_health "$VIBEQC_VENV_TX_TARGET" || return

    # Disable rollback only after the final-path environment is healthy. A
    # backup-cleanup failure leaves the verified new venv in service.
    VIBEQC_VENV_TX_ACTIVE=0
    if ! rm -f -- "$marker"; then
        echo "Warning: could not remove transaction marker '$marker'." >&2
    fi
    if [ -n "$backup" ] && \
       ! vibeqc_remove_venv "$backup" "$VIBEQC_VENV_TX_REPO_ROOT"; then
        echo "Warning: the old virtualenv remains at '$backup'." >&2
        echo "Remove it manually after confirming the update." >&2
    fi
    VIBEQC_VENV_TX_BACKUP=""
}

vibeqc_create_venv() {
    local venv="$1"
    local python_bin="$2"

    if [ "$VIBEQC_VENV_TX_ACTIVE" != "1" ] || \
       [ "$VIBEQC_VENV_TX_MUTATION_STARTED" != "1" ] || \
       [ "$venv" != "$VIBEQC_VENV_TX_TARGET" ]; then
        echo "Error: refusing to create a virtualenv outside an active replacement transaction." >&2
        return 1
    fi
    vibeqc_assert_python "$python_bin" || return
    echo "==> Creating venv at $venv (using $python_bin)..."
    "$python_bin" -m venv "$venv"
}

# ---------------------------------------------------------------------------
# vibeqc_pip_upgrade VENV
#
# Always goes through `python -m pip` to sidestep any stale shebang on
# the bare pip wrapper. --quiet keeps the transcript readable; pip
# prints a one-line "Successfully installed pip-X.Y.Z" only when it
# actually upgrades.
# ---------------------------------------------------------------------------
vibeqc_pip_upgrade() {
    local venv="$1"
    echo "==> Upgrading pip in $venv/ ..."
    "$venv/bin/python" -m pip install --quiet --upgrade pip
}

# ---------------------------------------------------------------------------
# vibeqc_pip_install_editable VENV EXTRAS [ROOT]
# ---------------------------------------------------------------------------
vibeqc_pip_install_editable() {
    local venv="$1"
    local extras="$2"
    local root="${3:-.}"

    # Keep build requirements in the persistent target venv. pip's default
    # isolated build environment lives at a new temporary path on every run;
    # that path change makes CMake rediscover pybind11 and invalidates the
    # binding translation unit even when no source changed. Explicit build
    # requirements + --no-build-isolation give the stable scikit-build tree
    # and ccache a stable include path without relaxing the pyproject floors.
    echo "==> Ensuring editable-build requirements in $venv/ ..."
    "$venv/bin/python" -m pip install --quiet \
        "scikit-build-core>=0.10" "pybind11>=2.12"

    echo "==> Installing vibe-qc into $venv/ (reuses unchanged C++ objects)..."
    "$venv/bin/python" -m pip install --no-build-isolation -e "${root}${extras}"
}
