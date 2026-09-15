#!/usr/bin/env bash
# Shared, Bash-3.2-compatible lifecycle helpers for vibe-basis.

set -euo pipefail

VIBE_BASIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIBE_BASIS_PROJECT_DIR="$(cd "$VIBE_BASIS_SCRIPT_DIR/.." && pwd)"
VIBE_BASIS_REPO_ROOT="$(cd "$VIBE_BASIS_PROJECT_DIR/.." && pwd)"
VIBE_BASIS_TOOLSET_LOCK_HELPER="$VIBE_BASIS_REPO_ROOT/scripts/_lifecycle_lock.sh"
if [ -f "$VIBE_BASIS_TOOLSET_LOCK_HELPER" ]; then
    # shellcheck source=../../scripts/_lifecycle_lock.sh
    . "$VIBE_BASIS_TOOLSET_LOCK_HELPER"
fi
VIBE_BASIS_MARKER=".vibe-basis-standalone"
VIBE_BASIS_REPLACEMENT_MARKER=".vibe-basis-replacement"
VIBE_BASIS_REPLACEMENT_ACTIVE=0
VIBE_BASIS_REPLACEMENT_TARGET=""
VIBE_BASIS_REPLACEMENT_BACKUP=""

vibe_basis_die() {
    echo "Error: $*" >&2
    exit 1
}

vibe_basis_require_value() {
    _vb_option="$1"
    _vb_value="${2-}"
    _vb_kind="$3"
    [ -n "$_vb_value" ] || vibe_basis_die "$_vb_option requires a non-empty $_vb_kind"
    case "$_vb_value" in
        -*) vibe_basis_die "$_vb_option requires a $_vb_kind, not option '$_vb_value'" ;;
    esac
}

vibe_basis_command_path() {
    _vb_command="$1"
    case "$_vb_command" in
        /*) _vb_resolved="$_vb_command" ;;
        *)  _vb_resolved="$(command -v -- "$_vb_command" 2>/dev/null || true)" ;;
    esac
    [ -n "$_vb_resolved" ] || vibe_basis_die "Python executable not found: $_vb_command"
    [ -x "$_vb_resolved" ] || vibe_basis_die "Python is not executable: $_vb_resolved"
    printf '%s\n' "$_vb_resolved"
}

vibe_basis_assert_python() {
    _vb_python="$1"
    PYTHONNOUSERSITE=1 "$_vb_python" -I -S - <<'PY' || exit 1
import sys
if sys.version_info < (3, 11):
    raise SystemExit(
        "Error: vibe-basis requires Python 3.11 or newer; found "
        + sys.version.split()[0]
    )
PY
}

vibe_basis_assert_vq_python() {
    _vb_python="$1"
    PYTHONNOUSERSITE=1 "$_vb_python" -I -S - <<'PY' || exit 1
import sys
if sys.version_info < (3, 12):
    raise SystemExit("Error: the co-located vq CLI requires Python 3.12 or newer")
PY
}

vibe_basis_project_path() {
    _vb_input="$1"
    case "$_vb_input" in
        /*) printf '%s\n' "$_vb_input" ;;
        *)  printf '%s\n' "$VIBE_BASIS_PROJECT_DIR/$_vb_input" ;;
    esac
}

vibe_basis_resolve_venv() {
    _vb_python="$1"
    _vb_input="$2"
    _vb_raw="$(vibe_basis_project_path "$_vb_input")"
    _vb_lexical="$_vb_raw"
    while [ "$_vb_lexical" != "/" ]; do
        case "$_vb_lexical" in
            */)  _vb_lexical="${_vb_lexical%/}" ;;
            */.) _vb_lexical="${_vb_lexical%/.}" ;;
            *) break ;;
        esac
    done
    [ ! -L "$_vb_lexical" ] || vibe_basis_die \
        "refusing a symlink virtualenv target: $_vb_lexical"
    _vb_parent="$(dirname "$_vb_lexical")"
    _vb_name="$(basename "$_vb_lexical")"
    case "$_vb_name" in
        ""|.|..|/) vibe_basis_die "unsafe virtualenv target: $_vb_input" ;;
    esac
    _vb_parent_real="$(cd -P "$_vb_parent" 2>/dev/null && pwd)" || \
        vibe_basis_die "virtualenv parent directory does not exist: $_vb_parent"
    if [ "$_vb_parent_real" = "/" ]; then
        _vb_resolved="/$_vb_name"
    else
        _vb_resolved="$_vb_parent_real/$_vb_name"
    fi
    [ ! -L "$_vb_resolved" ] || vibe_basis_die \
        "refusing a symlink virtualenv target: $_vb_resolved"
    printf '%s\n' "$_vb_resolved"
}

vibe_basis_assert_safe_target() {
    _vb_target="$1"
    _vb_safe_user_location=0
    _vb_home=""
    _vb_home_uid=""
    if [ -n "${HOME:-}" ]; then
        _vb_home="$(cd "$HOME" 2>/dev/null && pwd -P || true)"
        if [ -n "$_vb_home" ]; then
            _vb_home_uid="$(stat -c '%u' "$_vb_home" 2>/dev/null || \
                stat -f '%u' "$_vb_home" 2>/dev/null || true)"
        fi
    fi
    [ -n "$_vb_target" ] || vibe_basis_die "virtualenv path is empty"
    case "$_vb_target" in
        /|/Applications|/bin|/boot|/dev|/etc|/home|/Library|/nix|/opt|/private|/private/tmp|/proc|/root|/run|/sbin|/snap|/srv|/sys|/System|/tmp|/usr|/Users|/var|/Volumes)
            vibe_basis_die "refusing dangerous virtualenv target: $_vb_target"
            ;;
    esac
    # HOME only exempts descendants when it is a canonical account home owned
    # by this uid. Caller-controlled HOME/TMPDIR/XDG values are not trusted.
    if [ "$_vb_home_uid" = "$EUID" ]; then
        case "$_vb_home" in
            /Users/*|/home/*|/var/home/*|/root)
                case "$_vb_target" in
                    "$_vb_home"/*) _vb_safe_user_location=1 ;;
                esac
                ;;
        esac
    fi
    case "$_vb_target" in
        /tmp/*|/private/tmp/*|/var/tmp/*|/private/var/tmp/*|/var/folders/*|/private/var/folders/*|/run/user/"$EUID"/*)
            _vb_safe_user_location=1
            ;;
    esac
    case "$_vb_target" in
        .git|.git/*|*/.git|*/.git/*)
            vibe_basis_die "refusing virtualenv target inside Git metadata: $_vb_target"
            ;;
    esac
    if [ "$_vb_safe_user_location" != "1" ]; then
        case "$_vb_target" in
            /Applications/*|/Library/*|/System/*|/Users/*|/bin/*|/boot/*|/dev/*|/etc/*|/home/*|/nix/*|/opt/*|/private/*|/proc/*|/root/*|/run/*|/sbin/*|/snap/*|/srv/*|/sys/*|/usr/*|/var/*)
                vibe_basis_die "refusing dangerous virtualenv target inside a protected system location: $_vb_target"
                ;;
        esac
    fi
    case "$VIBE_BASIS_PROJECT_DIR/" in
        "$_vb_target/"*) vibe_basis_die "virtualenv target contains the vibe-basis source tree: $_vb_target" ;;
    esac
    case "$VIBE_BASIS_REPO_ROOT/" in
        "$_vb_target/"*) vibe_basis_die "virtualenv target contains the repository: $_vb_target" ;;
    esac
    if [ -n "${HOME:-}" ]; then
        if [ -n "$_vb_home" ]; then
            [ "$_vb_target" != "$_vb_home" ] || vibe_basis_die "refusing the home directory as a virtualenv target"
            case "$_vb_home/" in
                "$_vb_target/"*) vibe_basis_die "virtualenv target contains the home directory: $_vb_target" ;;
            esac
        fi
    fi
}

vibe_basis_marker_value() {
    _vb_marker_file="$1"
    _vb_marker_key="$2"
    sed -n "s/^${_vb_marker_key}=//p" "$_vb_marker_file" | head -n 1
}

vibe_basis_assert_removable_venv() {
    _vb_target="$1"
    vibe_basis_assert_safe_target "$_vb_target"
    [ ! -L "$_vb_target" ] || vibe_basis_die "refusing symlink virtualenv: $_vb_target"
    [ -d "$_vb_target" ] || vibe_basis_die "standalone virtualenv does not exist: $_vb_target"
    [ -f "$_vb_target/pyvenv.cfg" ] && [ ! -L "$_vb_target/pyvenv.cfg" ] || \
        vibe_basis_die "not a recognizable regular virtualenv: $_vb_target"
    [ -f "$_vb_target/$VIBE_BASIS_MARKER" ] && \
        [ ! -L "$_vb_target/$VIBE_BASIS_MARKER" ] || vibe_basis_die \
        "refusing unowned virtualenv (missing $VIBE_BASIS_MARKER): $_vb_target"
    _vb_owner="$(vibe_basis_marker_value "$_vb_target/$VIBE_BASIS_MARKER" project)"
    [ "$_vb_owner" = "$VIBE_BASIS_PROJECT_DIR" ] || vibe_basis_die \
        "virtualenv belongs to a different checkout: $_vb_target"
}

vibe_basis_assert_owned_venv() {
    _vb_target="$1"
    vibe_basis_assert_removable_venv "$_vb_target"
    [ -x "$_vb_target/bin/python" ] || vibe_basis_die \
        "virtualenv has no executable Python: $_vb_target"
}

vibe_basis_profile_spec() {
    case "$1" in
        core)       printf '%s\n' "" ;;
        standard)   printf '%s\n' "[scipy]" ;;
        optimizers) printf '%s\n' "[nlopt,iminuit,scipy]" ;;
        all)        printf '%s\n' "[all]" ;;
        test)       printf '%s\n' "[test]" ;;
        *) vibe_basis_die "unknown extras profile '$1' (use core, standard, optimizers, all, or test)" ;;
    esac
}

vibe_basis_profile_from_marker() {
    _vb_target="$1"
    _vb_profile="$(vibe_basis_marker_value "$_vb_target/$VIBE_BASIS_MARKER" profile)"
    case "$_vb_profile" in
        core|standard|optimizers|all|test) printf '%s\n' "$_vb_profile" ;;
        *) printf '%s\n' "standard" ;;
    esac
}

vibe_basis_vq_from_marker() {
    _vb_target="$1"
    _vb_with_vq="$(vibe_basis_marker_value "$_vb_target/$VIBE_BASIS_MARKER" with_vq)"
    [ "$_vb_with_vq" = "1" ] && printf '1\n' || printf '0\n'
}

vibe_basis_base_python() {
    _vb_target="$1"
    _vb_python="$(vibe_basis_marker_value \
        "$_vb_target/$VIBE_BASIS_MARKER" python)"
    [ -n "$_vb_python" ] || vibe_basis_die \
        "the install marker has no base interpreter; rerun with --python BIN"
    case "$_vb_python" in
        "$_vb_target"/*)
            _vb_python="$(awk -F'[[:space:]]*=[[:space:]]*' \
                '/^executable[[:space:]]*=/ {print $2; exit}' \
                "$_vb_target/pyvenv.cfg" 2>/dev/null || true)"
            [ -n "$_vb_python" ] || vibe_basis_die \
                "the legacy install recorded its own interpreter; rerun with --python BIN"
            ;;
    esac
    printf '%s\n' "$_vb_python"
}

vibe_basis_resolve_base_python() {
    _vb_requested="$1"
    _vb_target="$2"
    _vb_candidate="$(vibe_basis_command_path "$_vb_requested")"
    case "$_vb_candidate" in
        "$_vb_target"/*)
            _vb_candidate="$(awk -F'[[:space:]]*=[[:space:]]*' \
                '/^executable[[:space:]]*=/ {print $2; exit}' \
                "$_vb_target/pyvenv.cfg" 2>/dev/null || true)"
            [ -n "$_vb_candidate" ] || vibe_basis_die \
                "cannot replace a venv using its own interpreter; pass an external --python BIN"
            _vb_candidate="$(vibe_basis_command_path "$_vb_candidate")"
            ;;
    esac
    vibe_basis_assert_python "$_vb_candidate"
    _vb_base="$(PYTHONNOUSERSITE=1 "$_vb_candidate" -I -S - <<'PY'
import os
import sys
print(os.path.realpath(sys._base_executable))
PY
)"
    _vb_base="$(vibe_basis_command_path "$_vb_base")"
    case "$_vb_base" in
        "$_vb_target"/*) vibe_basis_die \
            "refusing a replacement interpreter inside the target venv: $_vb_base" ;;
    esac
    vibe_basis_assert_python "$_vb_base"
    printf '%s\n' "$_vb_base"
}

vibe_basis_assert_regular_user() {
    if [ -n "${SUDO_USER:-}" ] || [ "$(id -u)" -eq 0 ]; then
        vibe_basis_die "run vibe-basis lifecycle commands as your regular user, without sudo"
    fi
}

vibe_basis_acquire_target_lock() {
    _vb_shared_target="$1"
    _vb_shared_python="$2"
    command -v vibe_toolset_acquire_lifecycle_lock >/dev/null 2>&1 || {
        echo "Error: shared toolset lifecycle lock helper is missing." >&2
        return 1
    }
    vibe_toolset_acquire_lifecycle_lock \
        "$_vb_shared_python" "$VIBE_BASIS_REPO_ROOT" \
        "$_vb_shared_target" "vibe-basis lifecycle"
}

vibe_basis_release_target_lock() {
    command -v vibe_toolset_release_lifecycle_lock >/dev/null 2>&1 || return 0
    vibe_toolset_release_lifecycle_lock
}

vibe_basis_prepare_target_lock_handoff() {
    vibe_toolset_prepare_lifecycle_lock_handoff
}

vibe_basis_abort_replacement() {
    [ "$VIBE_BASIS_REPLACEMENT_ACTIVE" = "1" ] || return 0
    if [ -n "$VIBE_BASIS_REPLACEMENT_TARGET" ] && \
       { [ -e "$VIBE_BASIS_REPLACEMENT_TARGET" ] || [ -L "$VIBE_BASIS_REPLACEMENT_TARGET" ]; }; then
        vibe_basis_assert_safe_target "$VIBE_BASIS_REPLACEMENT_TARGET"
        _vb_replacement_marker="$VIBE_BASIS_REPLACEMENT_TARGET/$VIBE_BASIS_REPLACEMENT_MARKER"
        if [ -f "$_vb_replacement_marker" ] && [ ! -L "$_vb_replacement_marker" ] && \
           grep -Fqx "project=$VIBE_BASIS_PROJECT_DIR" "$_vb_replacement_marker" && \
           grep -Fqx "pid=$$" "$_vb_replacement_marker"; then
            rm -rf -- "$VIBE_BASIS_REPLACEMENT_TARGET"
        else
            _vb_failed="$VIBE_BASIS_REPLACEMENT_TARGET.vibe-basis-failed.$$"
            mv -- "$VIBE_BASIS_REPLACEMENT_TARGET" "$_vb_failed"
            echo "Preserved an unrecognized failed replacement at: $_vb_failed" >&2
        fi
    fi
    if [ -n "$VIBE_BASIS_REPLACEMENT_BACKUP" ] && [ -e "$VIBE_BASIS_REPLACEMENT_BACKUP" ]; then
        mv -- "$VIBE_BASIS_REPLACEMENT_BACKUP" "$VIBE_BASIS_REPLACEMENT_TARGET"
        echo "Restored the previous vibe-basis environment after failure." >&2
    fi
    VIBE_BASIS_REPLACEMENT_ACTIVE=0
}

vibe_basis_begin_replacement() {
    _vb_target="$1"
    VIBE_BASIS_REPLACEMENT_TARGET="$_vb_target"
    VIBE_BASIS_REPLACEMENT_BACKUP="$_vb_target.vibe-basis-backup.$$"
    [ ! -e "$VIBE_BASIS_REPLACEMENT_BACKUP" ] || vibe_basis_die \
        "replacement backup already exists: $VIBE_BASIS_REPLACEMENT_BACKUP"
    if [ -e "$_vb_target" ]; then
        vibe_basis_assert_removable_venv "$_vb_target"
        mv -- "$_vb_target" "$VIBE_BASIS_REPLACEMENT_BACKUP"
    else
        VIBE_BASIS_REPLACEMENT_BACKUP=""
    fi
    VIBE_BASIS_REPLACEMENT_ACTIVE=1
    mkdir "$_vb_target"
    {
        printf 'project=%s\n' "$VIBE_BASIS_PROJECT_DIR"
        printf 'pid=%s\n' "$$"
    } >"$_vb_target/$VIBE_BASIS_REPLACEMENT_MARKER"
}

vibe_basis_write_marker() {
    _vb_target="$1"
    _vb_profile="$2"
    _vb_with_vq="$3"
    _vb_python="$4"
    {
        printf 'project=%s\n' "$VIBE_BASIS_PROJECT_DIR"
        printf 'profile=%s\n' "$_vb_profile"
        printf 'with_vq=%s\n' "$_vb_with_vq"
        printf 'python=%s\n' "$_vb_python"
    } >"$_vb_target/$VIBE_BASIS_MARKER"
}

vibe_basis_verify_environment() {
    _vb_target="$1"
    _vb_profile="$2"
    _vb_with_vq="$3"
    "$_vb_target/bin/python" -c "import vibe_basis; print('vibe-basis', vibe_basis.__version__)"
    "$_vb_target/bin/vb" --version
    case "$_vb_profile" in
        standard|optimizers|all|test) "$_vb_target/bin/python" -c "import scipy" ;;
    esac
    if [ "$_vb_with_vq" = "1" ]; then
        "$_vb_target/bin/vq" --version
    fi
    "$_vb_target/bin/python" -m pip check
}

vibe_basis_stamp_vq_source_marker() {
    _vb_target="$1"
    if ! git -C "$VIBE_BASIS_REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
        echo "    (no Git checkout; vq SOURCE-SHA marker not written)"
        return 0
    fi
    _vb_sha="$(git -C "$VIBE_BASIS_REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
    [ -n "$_vb_sha" ] || vibe_basis_die \
        "could not read HEAD for the installed vq SOURCE-SHA marker"
    echo "==> Recording the bundled vq SOURCE-SHA marker..."
    "$_vb_target/bin/vq" source-sha --write-marker "$_vb_sha" >/dev/null || \
        vibe_basis_die "could not write the installed vq SOURCE-SHA marker"
    _vb_readback="$("$_vb_target/bin/vq" source-sha 2>/dev/null || true)"
    [ "$_vb_readback" = "$_vb_sha" ] || vibe_basis_die \
        "installed vq SOURCE-SHA verification mismatch"
    echo "    $_vb_sha"
}

vibe_basis_install_environment() {
    _vb_target="$1"
    _vb_python="$2"
    _vb_profile="$3"
    _vb_spec="$4"
    _vb_with_vq="$5"

    trap 'vibe_basis_abort_replacement; vibe_basis_release_target_lock' EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
    vibe_basis_begin_replacement "$_vb_target"

    echo "==> Creating virtualenv: $_vb_target"
    "$_vb_python" -m venv "$_vb_target"
    "$_vb_target/bin/python" -m pip install --upgrade pip
    "$_vb_target/bin/python" -m pip install "$VIBE_BASIS_PROJECT_DIR$_vb_spec"
    if [ "$_vb_with_vq" = "1" ]; then
        vibe_basis_assert_vq_python "$_vb_target/bin/python"
        "$_vb_target/bin/python" -m pip install "$VIBE_BASIS_REPO_ROOT/vibe-queue"
        vibe_basis_stamp_vq_source_marker "$_vb_target"
    fi
    rm -f -- "$_vb_target/$VIBE_BASIS_REPLACEMENT_MARKER"
    vibe_basis_write_marker "$_vb_target" "$_vb_profile" "$_vb_with_vq" "$_vb_python"
    vibe_basis_verify_environment "$_vb_target" "$_vb_profile" "$_vb_with_vq"

    VIBE_BASIS_REPLACEMENT_ACTIVE=0
    if [ -n "$VIBE_BASIS_REPLACEMENT_BACKUP" ]; then
        vibe_basis_assert_removable_venv "$VIBE_BASIS_REPLACEMENT_BACKUP"
        if ! rm -rf -- "$VIBE_BASIS_REPLACEMENT_BACKUP"; then
            echo "Warning: verified replacement is active, but the old backup could not be fully removed:" >&2
            echo "         $VIBE_BASIS_REPLACEMENT_BACKUP" >&2
        fi
    fi
    trap 'vibe_basis_release_target_lock' EXIT
}

vibe_basis_remove_owned_venv() {
    _vb_target="$1"
    vibe_basis_assert_removable_venv "$_vb_target"
    _vb_quarantine="$_vb_target.vibe-basis-remove.$$"
    [ ! -e "$_vb_quarantine" ] || vibe_basis_die \
        "uninstall quarantine already exists: $_vb_quarantine"
    mv -- "$_vb_target" "$_vb_quarantine"
    _vb_quarantine_owner=""
    if [ -f "$_vb_quarantine/$VIBE_BASIS_MARKER" ] && \
       [ ! -L "$_vb_quarantine/$VIBE_BASIS_MARKER" ]; then
        _vb_quarantine_owner="$(vibe_basis_marker_value \
            "$_vb_quarantine/$VIBE_BASIS_MARKER" project)"
    fi
    if [ ! -f "$_vb_quarantine/pyvenv.cfg" ] || \
       [ -L "$_vb_quarantine/pyvenv.cfg" ] || \
       [ "$_vb_quarantine_owner" != "$VIBE_BASIS_PROJECT_DIR" ]; then
        mv -- "$_vb_quarantine" "$_vb_target"
        vibe_basis_die "ownership changed during uninstall; restored the environment"
    fi
    rm -rf -- "$_vb_quarantine"
}
