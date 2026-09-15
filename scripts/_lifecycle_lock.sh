#!/usr/bin/env bash
# Shared cross-component lifecycle lock for the vibe toolset.
#
# Callers provide a trusted Python outside the selected virtualenv. The caller
# shell holds two descriptors whose shared open-file descriptions carry
# non-blocking fcntl locks: one for the canonical monorepo checkout and one for
# the canonical venv target. A one-shot isolated Python validates and locks the
# exact SHA-256-derived lock-file inodes. Keeping the descriptors in the caller
# removes the former helper-process death window while preserving locks across
# an exec handoff. Persistent lock files are harmless; the kernel releases the
# locks on every exit, including crashes. This works on stock macOS and Linux
# without flock(1).

if [ -n "${VIBE_TOOLSET_LIFECYCLE_LOCK_HELPER_LOADED:-}" ]; then
    return 0
fi
VIBE_TOOLSET_LIFECYCLE_LOCK_HELPER_LOADED=1

VIBE_TOOLSET_LIFECYCLE_LOCK_ACTIVE=0
VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT=""
VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET=""
VIBE_TOOLSET_LIFECYCLE_LOCK_ACTION=""
VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT_PATH=""
VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET_PATH=""
VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT_FD=""
VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET_FD=""
_VIBE_TOOLSET_DIRECT_LOCK_PRESENT=0

_VIBE_TOOLSET_ADMIN_LOCK_PID="${VIBE_TOOLSET_ADMIN_LOCK_PID:-}"
_VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT="${VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT:-}"
_VIBE_TOOLSET_ADMIN_LOCK_TARGET="${VIBE_TOOLSET_ADMIN_LOCK_TARGET:-}"
_VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT_PATH="${VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT_PATH:-}"
_VIBE_TOOLSET_ADMIN_LOCK_TARGET_PATH="${VIBE_TOOLSET_ADMIN_LOCK_TARGET_PATH:-}"
_VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT_FD="${VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT_FD:-}"
_VIBE_TOOLSET_ADMIN_LOCK_TARGET_FD="${VIBE_TOOLSET_ADMIN_LOCK_TARGET_FD:-}"
_VIBE_TOOLSET_ADMIN_LOCK_PYTHON="${VIBE_TOOLSET_ADMIN_LOCK_PYTHON:-}"
_VIBE_TOOLSET_ADMIN_LOCK_PRESENT=0
unset VIBE_TOOLSET_ADMIN_LOCK_PID VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT \
    VIBE_TOOLSET_ADMIN_LOCK_TARGET VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT_PATH \
    VIBE_TOOLSET_ADMIN_LOCK_TARGET_PATH VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT_FD \
    VIBE_TOOLSET_ADMIN_LOCK_TARGET_FD VIBE_TOOLSET_ADMIN_LOCK_PYTHON

if [ -n "$_VIBE_TOOLSET_ADMIN_LOCK_PID$_VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT$_VIBE_TOOLSET_ADMIN_LOCK_TARGET$_VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT_PATH$_VIBE_TOOLSET_ADMIN_LOCK_TARGET_PATH$_VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT_FD$_VIBE_TOOLSET_ADMIN_LOCK_TARGET_FD$_VIBE_TOOLSET_ADMIN_LOCK_PYTHON" ]; then
    if [ -z "$_VIBE_TOOLSET_ADMIN_LOCK_PID" ] || \
       [ -z "$_VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT" ] || \
       [ -z "$_VIBE_TOOLSET_ADMIN_LOCK_TARGET" ] || \
       [ -z "$_VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT_PATH" ] || \
       [ -z "$_VIBE_TOOLSET_ADMIN_LOCK_TARGET_PATH" ] || \
       [ -z "$_VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT_FD" ] || \
       [ -z "$_VIBE_TOOLSET_ADMIN_LOCK_TARGET_FD" ] || \
       [ -z "$_VIBE_TOOLSET_ADMIN_LOCK_PYTHON" ]; then
        echo "Error: refusing incomplete inherited admin lifecycle-lock state." >&2
        return 1
    fi
    case "$_VIBE_TOOLSET_ADMIN_LOCK_PID" in ''|*[!0-9]*) return 1 ;; esac
    case "$_VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT_FD" in ''|*[!0-9]*) return 1 ;; esac
    case "$_VIBE_TOOLSET_ADMIN_LOCK_TARGET_FD" in ''|*[!0-9]*) return 1 ;; esac
    [ "$_VIBE_TOOLSET_ADMIN_LOCK_PID" = "$PPID" ] || {
        echo "Error: inherited admin lifecycle lock has the wrong parent." >&2
        return 1
    }
    case "$_VIBE_TOOLSET_ADMIN_LOCK_PYTHON" in /*) ;; *) return 1 ;; esac
    [ -x "$_VIBE_TOOLSET_ADMIN_LOCK_PYTHON" ] || return 1
    if ! PYTHONNOUSERSITE=1 "$_VIBE_TOOLSET_ADMIN_LOCK_PYTHON" -I -S -c '
import fcntl, hashlib, os, pathlib, stat, sys
checkout, target, checkout_path, target_path, cfd, tfd = sys.argv[1:]

def resource_owner(value):
    current = pathlib.Path(value)
    while True:
        try:
            return current.lstat().st_uid
        except FileNotFoundError:
            parent = current.parent
            if parent == current:
                raise
            current = parent

for scope, resource, path, fd_text in (
    ("checkout", checkout, checkout_path, cfd),
    ("target", target, target_path, tfd),
):
    uid = resource_owner(resource)
    root = pathlib.Path("/tmp") / ("vibe-toolset-lifecycle-locks-%d" % uid)
    root_stat = root.lstat()
    if (not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != uid or
            stat.S_IMODE(root_stat.st_mode) != 0o700):
        raise SystemExit(1)
    digest = hashlib.sha256((scope + ":" + resource).encode("utf-8")).hexdigest()
    expected = root / (scope + "-" + digest + ".lock")
    if not resource.startswith("/") or pathlib.Path(path) != expected:
        raise SystemExit(1)
    fd = int(fd_text)
    opened = os.fstat(fd)
    named = os.stat(path, follow_symlinks=False)
    if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        raise SystemExit(1)
    if (not stat.S_ISREG(opened.st_mode) or opened.st_uid != uid or
            opened.st_nlink != 1 or stat.S_IMODE(opened.st_mode) != 0o600):
        raise SystemExit(1)
    # A descriptor inherited from the real owner shares its flock description
    # and succeeds. An arbitrary same-user file descriptor cannot bypass a
    # concurrently held expected lock.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
' "$_VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT" "$_VIBE_TOOLSET_ADMIN_LOCK_TARGET" \
      "$_VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT_PATH" "$_VIBE_TOOLSET_ADMIN_LOCK_TARGET_PATH" \
      "$_VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT_FD" "$_VIBE_TOOLSET_ADMIN_LOCK_TARGET_FD"; then
        echo "Error: inherited admin lifecycle lock is invalid." >&2
        return 1
    fi
    VIBE_TOOLSET_LIFECYCLE_LOCK_ACTIVE=1
    VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT="$_VIBE_TOOLSET_ADMIN_LOCK_CHECKOUT"
    VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET="$_VIBE_TOOLSET_ADMIN_LOCK_TARGET"
    VIBE_TOOLSET_LIFECYCLE_LOCK_ACTION="vq-admin-update"
    _VIBE_TOOLSET_ADMIN_LOCK_PRESENT=1
fi

_VIBE_TOOLSET_INHERITED_PID="${VIBE_TOOLSET_INHERITED_PID:-}"
_VIBE_TOOLSET_INHERITED_CHECKOUT="${VIBE_TOOLSET_INHERITED_CHECKOUT:-}"
_VIBE_TOOLSET_INHERITED_TARGET="${VIBE_TOOLSET_INHERITED_TARGET:-}"
_VIBE_TOOLSET_INHERITED_ACTION="${VIBE_TOOLSET_INHERITED_ACTION:-}"
_VIBE_TOOLSET_INHERITED_CHECKOUT_PATH="${VIBE_TOOLSET_INHERITED_CHECKOUT_PATH:-}"
_VIBE_TOOLSET_INHERITED_TARGET_PATH="${VIBE_TOOLSET_INHERITED_TARGET_PATH:-}"
_VIBE_TOOLSET_INHERITED_CHECKOUT_FD="${VIBE_TOOLSET_INHERITED_CHECKOUT_FD:-}"
_VIBE_TOOLSET_INHERITED_TARGET_FD="${VIBE_TOOLSET_INHERITED_TARGET_FD:-}"
_VIBE_TOOLSET_LEGACY_HELPER_PID="${VIBE_TOOLSET_INHERITED_HELPER_PID:-}"
_VIBE_TOOLSET_LEGACY_STATE="${VIBE_TOOLSET_INHERITED_STATE:-}"
_VIBE_TOOLSET_INHERITED_PRESENT=0
unset VIBE_TOOLSET_INHERITED_PID VIBE_TOOLSET_INHERITED_CHECKOUT \
    VIBE_TOOLSET_INHERITED_TARGET VIBE_TOOLSET_INHERITED_ACTION \
    VIBE_TOOLSET_INHERITED_CHECKOUT_PATH VIBE_TOOLSET_INHERITED_TARGET_PATH \
    VIBE_TOOLSET_INHERITED_CHECKOUT_FD VIBE_TOOLSET_INHERITED_TARGET_FD \
    VIBE_TOOLSET_INHERITED_HELPER_PID VIBE_TOOLSET_INHERITED_STATE

if [ -n "$_VIBE_TOOLSET_LEGACY_HELPER_PID$_VIBE_TOOLSET_LEGACY_STATE" ]; then
    echo "Error: refusing legacy FIFO/helper toolset lifecycle-lock state." >&2
    return 1
fi

if [ -n "$_VIBE_TOOLSET_INHERITED_PID" ] || \
   [ -n "$_VIBE_TOOLSET_INHERITED_CHECKOUT" ] || \
   [ -n "$_VIBE_TOOLSET_INHERITED_TARGET" ] || \
   [ -n "$_VIBE_TOOLSET_INHERITED_ACTION" ] || \
   [ -n "$_VIBE_TOOLSET_INHERITED_CHECKOUT_PATH" ] || \
   [ -n "$_VIBE_TOOLSET_INHERITED_TARGET_PATH" ] || \
   [ -n "$_VIBE_TOOLSET_INHERITED_CHECKOUT_FD" ] || \
   [ -n "$_VIBE_TOOLSET_INHERITED_TARGET_FD" ]; then
    if [ -z "$_VIBE_TOOLSET_INHERITED_PID" ] || \
       [ -z "$_VIBE_TOOLSET_INHERITED_CHECKOUT" ] || \
       [ -z "$_VIBE_TOOLSET_INHERITED_TARGET" ] || \
       [ -z "$_VIBE_TOOLSET_INHERITED_ACTION" ] || \
       [ -z "$_VIBE_TOOLSET_INHERITED_CHECKOUT_PATH" ] || \
       [ -z "$_VIBE_TOOLSET_INHERITED_TARGET_PATH" ] || \
       [ -z "$_VIBE_TOOLSET_INHERITED_CHECKOUT_FD" ] || \
       [ -z "$_VIBE_TOOLSET_INHERITED_TARGET_FD" ]; then
        echo "Error: refusing incomplete inherited toolset lifecycle-lock state." >&2
        return 1
    fi
    case "$_VIBE_TOOLSET_INHERITED_PID" in
        ''|*[!0-9]*)
            echo "Error: inherited lifecycle-lock process PID is invalid." >&2
            return 1
            ;;
    esac
    case "$_VIBE_TOOLSET_INHERITED_CHECKOUT_FD" in
        194) ;;
        *)
            echo "Error: inherited lifecycle-lock checkout descriptor is invalid." >&2
            return 1
            ;;
    esac
    case "$_VIBE_TOOLSET_INHERITED_TARGET_FD" in
        195) ;;
        *)
            echo "Error: inherited lifecycle-lock target descriptor is invalid." >&2
            return 1
            ;;
    esac
    [ "$_VIBE_TOOLSET_INHERITED_PID" = "$$" ] || {
        echo "Error: refusing an inherited lifecycle lock from another process." >&2
        return 1
    }
    case "$_VIBE_TOOLSET_INHERITED_CHECKOUT" in
        /*) ;;
        *)
            echo "Error: inherited lifecycle-lock checkout is invalid." >&2
            return 1
            ;;
    esac
    case "$_VIBE_TOOLSET_INHERITED_TARGET" in
        /*) ;;
        *)
            echo "Error: inherited lifecycle-lock target is invalid." >&2
            return 1
            ;;
    esac
    [ "$_VIBE_TOOLSET_ADMIN_LOCK_PRESENT" = "0" ] || {
        echo "Error: refusing simultaneous admin and shell lifecycle-lock handoffs." >&2
        return 1
    }
    _VIBE_TOOLSET_INHERITED_PRESENT=1
fi

_vibe_toolset_try_external_python() {
    local out_var="$1"
    local requested="$2"
    local protected_target="$3"
    local min_major="${4:-3}"
    local min_minor="${5:-8}"
    local resolved=""
    local target_real="$protected_target"
    local link_value=""
    local link_dir=""
    local hops=0

    resolved="$(command -v -- "$requested" 2>/dev/null || true)"
    case "$resolved" in
        /*) ;;
        */*)
            resolved="$(cd "$(dirname "$resolved")" 2>/dev/null && pwd -P)/$(basename "$resolved")" || return 1
            ;;
        *) return 1 ;;
    esac
    [ -x "$resolved" ] || return 1
    while [ -L "$resolved" ]; do
        [ "$hops" -lt 40 ] || return 1
        link_value="$(readlink "$resolved")" || return 1
        link_dir="$(cd "$(dirname "$resolved")" 2>/dev/null && pwd -P)" || return 1
        case "$link_value" in
            /*) resolved="$link_value" ;;
            *)  resolved="$link_dir/$link_value" ;;
        esac
        resolved="$(cd "$(dirname "$resolved")" 2>/dev/null && pwd -P)/$(basename "$resolved")" || return 1
        hops=$((hops + 1))
    done
    resolved="$(cd "$(dirname "$resolved")" 2>/dev/null && pwd -P)/$(basename "$resolved")" || return 1

    if [ -d "$target_real" ]; then
        target_real="$(cd "$target_real" 2>/dev/null && pwd -P)" || return 1
    fi
    case "$resolved" in
        "$target_real"|"$target_real"/*) return 1 ;;
    esac
    if ! PYTHONNOUSERSITE=1 "$resolved" -I -S -c \
        'import sys; raise SystemExit(sys.version_info < tuple(map(int, sys.argv[1:3])))' \
        "$min_major" "$min_minor" \
        >/dev/null 2>&1; then
        return 1
    fi
    printf -v "$out_var" '%s' "$resolved"
}

# Select an isolated metadata/locking interpreter without consulting or
# executing the selected environment. Python 3.8+ is sufficient for these
# helpers; package creation and installation retain their Python 3.11+ gates.
vibe_toolset_select_external_python() {
    local out_var="$1"
    local requested="$2"
    local protected_target="$3"
    local min_major="${4:-3}"
    local min_minor="${5:-8}"
    local selected=""

    if ! _vibe_toolset_try_external_python \
        selected "$requested" "$protected_target" "$min_major" "$min_minor"; then
        echo "Error: trusted metadata Python must be an isolated external interpreter:" >&2
        echo "       $requested" >&2
        return 1
    fi
    printf -v "$out_var" '%s' "$selected"
}

vibe_toolset_find_external_python() {
    local out_var="$1"
    local protected_target="$2"
    local min_major="${3:-3}"
    local min_minor="${4:-8}"
    local selected=""
    local candidate=""
    local path_dir=""
    local old_ifs="$IFS"
    local candidates=(
        /usr/bin/python3
        /usr/local/bin/python3
        /opt/homebrew/bin/python3
        /opt/local/bin/python3
        python3
    )

    for candidate in "${candidates[@]}"; do
        if _vibe_toolset_try_external_python \
            selected "$candidate" "$protected_target" "$min_major" "$min_minor"; then
            printf -v "$out_var" '%s' "$selected"
            return 0
        fi
    done
    IFS=:
    for path_dir in ${PATH:-}; do
        [ -n "$path_dir" ] || path_dir="."
        if _vibe_toolset_try_external_python \
            selected "$path_dir/python3" "$protected_target" "$min_major" "$min_minor"; then
            IFS="$old_ifs"
            printf -v "$out_var" '%s' "$selected"
            return 0
        fi
    done
    IFS="$old_ifs"
    echo "Error: no isolated external Python is available for lifecycle metadata and locking." >&2
    return 1
}

_vibe_toolset_validate_inherited_lifecycle_lock() {
    local python_bin="$1"
    local checkout="$_VIBE_TOOLSET_INHERITED_CHECKOUT"
    local target="$_VIBE_TOOLSET_INHERITED_TARGET"
    local checkout_path="$_VIBE_TOOLSET_INHERITED_CHECKOUT_PATH"
    local target_path="$_VIBE_TOOLSET_INHERITED_TARGET_PATH"
    local checkout_fd="$_VIBE_TOOLSET_INHERITED_CHECKOUT_FD"
    local target_fd="$_VIBE_TOOLSET_INHERITED_TARGET_FD"

    if ! PYTHONNOUSERSITE=1 "$python_bin" -I -S -c '
import fcntl
import hashlib
import os
import pathlib
import stat
import sys

checkout, target, checkout_path, target_path, cfd, tfd = sys.argv[1:]

def resource_owner(value):
    current = pathlib.Path(value)
    while True:
        try:
            return current.lstat().st_uid
        except FileNotFoundError:
            parent = current.parent
            if parent == current:
                raise
            current = parent

for scope, resource, path, fd_text in (
    ("checkout", checkout, checkout_path, cfd),
    ("target", target, target_path, tfd),
):
    uid = resource_owner(resource)
    root = pathlib.Path("/tmp") / ("vibe-toolset-lifecycle-locks-%d" % uid)
    root_stat = root.lstat()
    if (not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != uid or
            stat.S_IMODE(root_stat.st_mode) != 0o700):
        raise SystemExit(1)
    digest = hashlib.sha256((scope + ":" + resource).encode("utf-8")).hexdigest()
    expected = root / (scope + "-" + digest + ".lock")
    if not resource.startswith("/") or pathlib.Path(path) != expected:
        raise SystemExit(1)
    fd = int(fd_text)
    opened = os.fstat(fd)
    named = os.stat(path, follow_symlinks=False)
    if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        raise SystemExit(1)
    if (not stat.S_ISREG(opened.st_mode) or opened.st_uid != uid or
            opened.st_nlink != 1 or stat.S_IMODE(opened.st_mode) != 0o600):
        raise SystemExit(1)
    # Success proves this descriptor either shares the original lock open-file
    # description or itself acquired the exact expected lock. It can never
    # accept an unlocked substitute while another operation owns the resource.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
' "$checkout" "$target" "$checkout_path" "$target_path" \
      "$checkout_fd" "$target_fd" \
        >/dev/null 2>&1; then
        echo "Error: inherited toolset lifecycle-lock descriptors are not active and valid." >&2
        return 1
    fi
}

if [ "$_VIBE_TOOLSET_INHERITED_PRESENT" = "1" ]; then
    _VIBE_TOOLSET_INHERITED_PYTHON=""
    vibe_toolset_find_external_python \
        _VIBE_TOOLSET_INHERITED_PYTHON "$_VIBE_TOOLSET_INHERITED_TARGET" || return 1
    _vibe_toolset_validate_inherited_lifecycle_lock \
        "$_VIBE_TOOLSET_INHERITED_PYTHON" || return 1
    VIBE_TOOLSET_LIFECYCLE_LOCK_ACTIVE=1
    VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT="$_VIBE_TOOLSET_INHERITED_CHECKOUT"
    VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET="$_VIBE_TOOLSET_INHERITED_TARGET"
    VIBE_TOOLSET_LIFECYCLE_LOCK_ACTION="$_VIBE_TOOLSET_INHERITED_ACTION"
    VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT_PATH="$_VIBE_TOOLSET_INHERITED_CHECKOUT_PATH"
    VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET_PATH="$_VIBE_TOOLSET_INHERITED_TARGET_PATH"
    VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT_FD="$_VIBE_TOOLSET_INHERITED_CHECKOUT_FD"
    VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET_FD="$_VIBE_TOOLSET_INHERITED_TARGET_FD"
    _VIBE_TOOLSET_DIRECT_LOCK_PRESENT=1
fi
unset _VIBE_TOOLSET_INHERITED_PID _VIBE_TOOLSET_INHERITED_CHECKOUT \
    _VIBE_TOOLSET_INHERITED_TARGET _VIBE_TOOLSET_INHERITED_ACTION \
    _VIBE_TOOLSET_INHERITED_CHECKOUT_PATH _VIBE_TOOLSET_INHERITED_TARGET_PATH \
    _VIBE_TOOLSET_INHERITED_CHECKOUT_FD _VIBE_TOOLSET_INHERITED_TARGET_FD \
    _VIBE_TOOLSET_LEGACY_HELPER_PID _VIBE_TOOLSET_LEGACY_STATE \
    _VIBE_TOOLSET_INHERITED_PRESENT _VIBE_TOOLSET_INHERITED_PYTHON

_vibe_toolset_resolve_lifecycle_resources() {
    local checkout="$1"
    local target="$2"
    local target_parent=""
    local target_name=""
    local target_suffix=""
    local target_ancestor=""

    _VIBE_TOOLSET_RESOLVED_CHECKOUT="$(cd "$checkout" 2>/dev/null && /bin/pwd -P)" || {
        echo "Error: cannot resolve lifecycle checkout '$checkout'." >&2
        return 1
    }
    # An existing target has an authoritative on-disk spelling.  Resolve the
    # whole directory rather than retaining the caller's final basename: on a
    # case-insensitive macOS volume, `.VENV` and `.venv` name the same inode and
    # must hash to the same cross-language lock identity.
    if [ -d "$target" ] && [ ! -L "$target" ]; then
        _VIBE_TOOLSET_RESOLVED_TARGET="$(cd "$target" 2>/dev/null && /bin/pwd -P)" || {
            echo "Error: cannot resolve lifecycle target '$target'." >&2
            return 1
        }
        return 0
    fi
    if [ -e "$target" ] || [ -L "$target" ]; then
        echo "Error: lifecycle target is not a real directory: '$target'." >&2
        return 1
    fi
    target_parent="$(dirname "$target")"
    target_suffix="$(basename "$target")"
    # Fresh installs may name several missing parent components. Resolve only
    # the nearest existing directory, then retain the missing suffix and final
    # component lexically. This gives every component the same stable lock key
    # without following a final symlink introduced during a race.
    while [ ! -d "$target_parent" ]; do
        if [ -e "$target_parent" ] || [ -L "$target_parent" ]; then
            echo "Error: lifecycle target parent is not a directory: '$target_parent'." >&2
            return 1
        fi
        target_name="$(basename "$target_parent")"
        case "$target_name" in
            ""|.|..|/)
                echo "Error: cannot resolve lifecycle target parent for '$target'." >&2
                return 1
                ;;
        esac
        target_suffix="$target_name/$target_suffix"
        target_ancestor="$(dirname "$target_parent")"
        [ "$target_ancestor" != "$target_parent" ] || {
            echo "Error: cannot resolve lifecycle target parent for '$target'." >&2
            return 1
        }
        target_parent="$target_ancestor"
    done
    target_parent="$(cd "$target_parent" 2>/dev/null && /bin/pwd -P)" || {
        echo "Error: cannot resolve lifecycle target parent for '$target'." >&2
        return 1
    }
    # A missing component has no stored spelling to recover with `pwd -P`.
    # Collapse ASCII case on Darwin so differently-cased callers cannot take
    # separate locks and then create the same path on case-insensitive APFS.
    # On a case-sensitive Darwin volume this merely over-serializes two future
    # paths; it never weakens exclusion.
    if [ "$(uname -s 2>/dev/null || true)" = "Darwin" ]; then
        target_suffix="$(LC_ALL=C printf '%s' "$target_suffix" | tr 'A-Z' 'a-z')"
    fi
    if [ "$target_parent" = "/" ]; then
        _VIBE_TOOLSET_RESOLVED_TARGET="/$target_suffix"
    else
        _VIBE_TOOLSET_RESOLVED_TARGET="$target_parent/$target_suffix"
    fi
}

# One exact checkout-metadata receipt is also the durable admission marker for
# single-user -> multi-user migration. Derive it only from the canonical
# checkout and bind the exact serving target inside the strict receipt; ambient
# vq state/config variables must not select this path. Keeping it below .git
# prevents the marker itself from falsifying the accepted dirty-tree proof.
vibe_toolset_multi_user_migration_receipt_path() {
    local out_var="$1"
    local checkout="$2"
    local target="$3"
    local receipt=""

    _vibe_toolset_resolve_lifecycle_resources "$checkout" "$target" || return
    [ -d "$_VIBE_TOOLSET_RESOLVED_CHECKOUT/.git" ] \
        && [ ! -L "$_VIBE_TOOLSET_RESOLVED_CHECKOUT/.git" ] || {
        echo "Error: lifecycle checkout has no real .git directory." >&2
        return 1
    }
    receipt="$_VIBE_TOOLSET_RESOLVED_CHECKOUT/.git/vq-multi-user-bootstrap.json"
    printf -v "$out_var" '%s' "$receipt"
}

_vibe_toolset_refuse_active_multi_user_migration() {
    local python_bin="$1"
    local checkout="$2"
    local target="$3"
    local action="$4"
    local receipt=""

    # The deployer alone may hold the ordinary checkout+serving-target lock
    # while creating, validating, or recovering this marker.  Every supported
    # mutator uses a different fixed action and therefore fails closed.
    [ "$action" = "vq-multi-user-deploy" ] && return 0
    receipt="$checkout/.git/vq-multi-user-bootstrap.json"
    if ! PYTHONNOUSERSITE=1 "$python_bin" -I -S -c '
import os
import sys

raise SystemExit(1 if os.path.lexists(sys.argv[1]) else 0)
' "$receipt"; then
        echo "Error: durable multi-user migration admission is active for $target." >&2
        echo "       Resume the exact deploy-multi-user.sh transaction before mutating this checkout or virtualenv." >&2
        return 1
    fi
}

vibe_toolset_acquire_lifecycle_lock() {
    local python_bin="$1"
    local checkout="$2"
    local target="$3"
    local action="$4"
    local checkout_real=""
    local target_real=""
    local lock_paths=""
    local checkout_lock_path=""
    local target_lock_path=""
    local status=""

    _vibe_toolset_resolve_lifecycle_resources "$checkout" "$target" || return
    checkout_real="$_VIBE_TOOLSET_RESOLVED_CHECKOUT"
    target_real="$_VIBE_TOOLSET_RESOLVED_TARGET"

    if [ "$VIBE_TOOLSET_LIFECYCLE_LOCK_ACTIVE" = "1" ]; then
        if [ "$VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT" = "$checkout_real" ] && \
           [ "$VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET" = "$target_real" ]; then
            _vibe_toolset_refuse_active_multi_user_migration \
                "$python_bin" "$checkout_real" "$target_real" "$action"
            return
        fi
        echo "Error: this process already holds a different toolset lifecycle lock." >&2
        return 1
    fi
    [ -x "$python_bin" ] || {
        echo "Error: trusted lifecycle-lock Python is not executable: $python_bin" >&2
        return 1
    }

    # Prepare the two exact persistent lock inodes securely. The paths contain
    # only a fixed prefix and hexadecimal SHA-256 digest, so splitting the two
    # output lines in the shell never interprets caller-controlled path text.
    if ! lock_paths="$(PYTHONNOUSERSITE=1 "$python_bin" -I -S -c '
import hashlib
import os
import pathlib
import stat
import sys

checkout, target = sys.argv[1:]

def resource_owner(value):
    current = pathlib.Path(value)
    while True:
        try:
            return current.lstat().st_uid
        except FileNotFoundError:
            parent = current.parent
            if parent == current:
                raise
            current = parent

try:
    paths = []
    for scope, resource in (("checkout", checkout), ("target", target)):
        uid = resource_owner(resource)
        if os.geteuid() not in (0, uid):
            raise RuntimeError("current user does not own %s lifecycle resource" % scope)
        lock_root = pathlib.Path("/tmp") / ("vibe-toolset-lifecycle-locks-%d" % uid)
        if lock_root.is_symlink():
            raise RuntimeError("symlinked %s lifecycle lock root" % scope)
        created_root = False
        try:
            lock_root.mkdir(mode=0o700)
            created_root = True
        except FileExistsError:
            pass
        if created_root and os.geteuid() == 0 and uid != 0:
            os.chown(lock_root, uid, -1)
        if created_root:
            os.chmod(lock_root, 0o700)
        root_stat = lock_root.lstat()
        if (not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != uid or
                stat.S_IMODE(root_stat.st_mode) != 0o700):
            raise RuntimeError("unsafe %s lifecycle lock root" % scope)

        digest = hashlib.sha256((scope + ":" + resource).encode("utf-8")).hexdigest()
        lock_path = lock_root / (scope + "-" + digest + ".lock")
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        created_file = False
        try:
            fd = os.open(str(lock_path), flags | os.O_CREAT | os.O_EXCL, 0o600)
            created_file = True
        except FileExistsError:
            fd = os.open(str(lock_path), flags)
        try:
            if created_file and os.geteuid() == 0 and uid != 0:
                os.fchown(fd, uid, -1)
            if created_file:
                os.fchmod(fd, 0o600)
            file_stat = os.fstat(fd)
            named = os.stat(lock_path, follow_symlinks=False)
            if ((file_stat.st_dev, file_stat.st_ino) !=
                    (named.st_dev, named.st_ino)):
                raise RuntimeError("changed %s lifecycle lock inode" % scope)
            if (not stat.S_ISREG(file_stat.st_mode) or file_stat.st_uid != uid or
                    file_stat.st_nlink != 1 or
                    stat.S_IMODE(file_stat.st_mode) != 0o600):
                raise RuntimeError("unsafe %s lifecycle lock file" % scope)
        finally:
            os.close(fd)
        paths.append(lock_path)
    print(*paths, sep="\n")
except Exception as exc:
    print("error:%s" % exc, file=sys.stderr)
    raise SystemExit(1)
' "$checkout_real" "$target_real")"; then
        echo "Error: could not prepare toolset lifecycle lock files." >&2
        return 1
    fi
    case "$lock_paths" in
        *$'\n'*) ;;
        *)
            echo "Error: lifecycle lock-file preparation returned invalid paths." >&2
            return 1
            ;;
    esac
    checkout_lock_path="${lock_paths%%$'\n'*}"
    target_lock_path="${lock_paths#*$'\n'}"
    case "$target_lock_path" in
        *$'\n'*|"")
            echo "Error: lifecycle lock-file preparation returned invalid paths." >&2
            return 1
            ;;
    esac

    if [ -e /dev/fd/194 ] || [ -e /dev/fd/195 ]; then
        echo "Error: lifecycle lock descriptors 194/195 are already in use." >&2
        return 1
    fi
    if ! exec 194<>"$checkout_lock_path"; then
        echo "Error: could not open the checkout lifecycle lock descriptor." >&2
        return 1
    fi
    if ! exec 195<>"$target_lock_path"; then
        exec 194>&-
        echo "Error: could not open the target lifecycle lock descriptor." >&2
        return 1
    fi

    # Lock the shell-owned open-file descriptions. flock locks survive the
    # one-shot validator exiting because descriptors 194/195 are references to
    # those same descriptions. They also survive the legitimate exec handoff.
    if status="$(PYTHONNOUSERSITE=1 "$python_bin" -I -S -c '
import fcntl
import hashlib
import os
import pathlib
import stat
import sys

checkout, target, checkout_path, target_path = sys.argv[1:]

def resource_owner(value):
    current = pathlib.Path(value)
    while True:
        try:
            return current.lstat().st_uid
        except FileNotFoundError:
            parent = current.parent
            if parent == current:
                raise
            current = parent

try:
    for scope, resource, path, fd in (
        ("checkout", checkout, checkout_path, 194),
        ("target", target, target_path, 195),
    ):
        uid = resource_owner(resource)
        root = pathlib.Path("/tmp") / ("vibe-toolset-lifecycle-locks-%d" % uid)
        root_stat = root.lstat()
        if (not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != uid or
                stat.S_IMODE(root_stat.st_mode) != 0o700):
            raise RuntimeError("unsafe %s lifecycle lock root" % scope)
        digest = hashlib.sha256((scope + ":" + resource).encode("utf-8")).hexdigest()
        expected = root / (scope + "-" + digest + ".lock")
        if pathlib.Path(path) != expected:
            raise RuntimeError("wrong %s lifecycle lock path" % scope)
        opened = os.fstat(fd)
        named = os.stat(path, follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise RuntimeError("changed %s lifecycle lock inode" % scope)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_uid != uid or
                opened.st_nlink != 1 or stat.S_IMODE(opened.st_mode) != 0o600):
            raise RuntimeError("unsafe %s lifecycle lock file" % scope)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("busy:%s" % scope)
            raise SystemExit(2)
    print("acquired")
except SystemExit:
    raise
except Exception as exc:
    print("error:%s" % exc)
    raise SystemExit(1)
' "$checkout_real" "$target_real" "$checkout_lock_path" "$target_lock_path" \
        194>&194 195>&195)"; then
        :
    else
        exec 194>&-
        exec 195>&-
        case "$status" in
            busy:checkout)
                echo "Error: another toolset lifecycle operation already active on this checkout." >&2
                ;;
            busy:target)
                echo "Error: another toolset lifecycle operation already active on this target." >&2
                ;;
            error:*)
                echo "Error: could not acquire toolset lifecycle lock: ${status#error:}" >&2
                ;;
            *)
                echo "Error: lifecycle lock validation failed." >&2
                ;;
        esac
        return 1
    fi
    [ "$status" = "acquired" ] || {
        exec 194>&-
        exec 195>&-
        echo "Error: lifecycle lock validation returned an invalid result." >&2
        return 1
    }

    if ! _vibe_toolset_refuse_active_multi_user_migration \
        "$python_bin" "$checkout_real" "$target_real" "$action"; then
        exec 194>&-
        exec 195>&-
        return 1
    fi

    VIBE_TOOLSET_LIFECYCLE_LOCK_ACTIVE=1
    VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT="$checkout_real"
    VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET="$target_real"
    VIBE_TOOLSET_LIFECYCLE_LOCK_ACTION="$action"
    VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT_PATH="$checkout_lock_path"
    VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET_PATH="$target_lock_path"
    VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT_FD=194
    VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET_FD=195
    _VIBE_TOOLSET_DIRECT_LOCK_PRESENT=1
}

vibe_toolset_prepare_lifecycle_lock_handoff() {
    [ "$VIBE_TOOLSET_LIFECYCLE_LOCK_ACTIVE" = "1" ] || {
        echo "Error: cannot hand off an inactive toolset lifecycle lock." >&2
        return 1
    }
    [ "$_VIBE_TOOLSET_DIRECT_LOCK_PRESENT" = "1" ] || {
        echo "Error: only a shell-owned lifecycle lock can use shell exec handoff." >&2
        return 1
    }
    VIBE_TOOLSET_INHERITED_PID="$$"
    VIBE_TOOLSET_INHERITED_CHECKOUT="$VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT"
    VIBE_TOOLSET_INHERITED_TARGET="$VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET"
    VIBE_TOOLSET_INHERITED_ACTION="$VIBE_TOOLSET_LIFECYCLE_LOCK_ACTION"
    VIBE_TOOLSET_INHERITED_CHECKOUT_PATH="$VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT_PATH"
    VIBE_TOOLSET_INHERITED_TARGET_PATH="$VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET_PATH"
    VIBE_TOOLSET_INHERITED_CHECKOUT_FD="$VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT_FD"
    VIBE_TOOLSET_INHERITED_TARGET_FD="$VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET_FD"
    export VIBE_TOOLSET_INHERITED_PID VIBE_TOOLSET_INHERITED_CHECKOUT \
        VIBE_TOOLSET_INHERITED_TARGET VIBE_TOOLSET_INHERITED_ACTION \
        VIBE_TOOLSET_INHERITED_CHECKOUT_PATH VIBE_TOOLSET_INHERITED_TARGET_PATH \
        VIBE_TOOLSET_INHERITED_CHECKOUT_FD VIBE_TOOLSET_INHERITED_TARGET_FD
}

# The parent PID alone is never authority to identify an outer admin
# transaction. update.sh calls this only after resolving its exact venv.
# Success proves only that the independently validated admin-owned lock
# descriptors cover that checkout/target and serialize mutation. It does not
# prove the daemon is quiescent; update.sh re-probes that separately.
vibe_toolset_require_admin_restart_capability() {
    local checkout="$1"
    local target="$2"
    local parent_pid="$3"

    [ "$_VIBE_TOOLSET_ADMIN_LOCK_PRESENT" = "1" ] && \
    [ "$parent_pid" = "$_VIBE_TOOLSET_ADMIN_LOCK_PID" ] && \
    [ "$parent_pid" = "$PPID" ] || {
        echo "Error: the outer-admin restart handshake lacks a validated lifecycle-lock capability." >&2
        return 1
    }
    _vibe_toolset_resolve_lifecycle_resources "$checkout" "$target" || return
    if [ "$VIBE_TOOLSET_LIFECYCLE_LOCK_ACTIVE" != "1" ] || \
       [ "$VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT" != "$_VIBE_TOOLSET_RESOLVED_CHECKOUT" ] || \
       [ "$VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET" != "$_VIBE_TOOLSET_RESOLVED_TARGET" ]; then
        echo "Error: the outer-admin lifecycle-lock capability does not cover this exact checkout and virtualenv." >&2
        return 1
    fi
}

vibe_toolset_release_lifecycle_lock() {
    [ "$VIBE_TOOLSET_LIFECYCLE_LOCK_ACTIVE" = "1" ] || return 0
    if [ "$_VIBE_TOOLSET_ADMIN_LOCK_PRESENT" = "1" ]; then
        VIBE_TOOLSET_LIFECYCLE_LOCK_ACTIVE=0
        return 0
    fi
    if [ "$_VIBE_TOOLSET_DIRECT_LOCK_PRESENT" = "1" ]; then
        case "$VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT_FD" in
            194) exec 194>&- ;;
            *)
                echo "Error: refusing to close an unexpected checkout lifecycle-lock descriptor." >&2
                return 1
                ;;
        esac
        case "$VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET_FD" in
            195) exec 195>&- ;;
            *)
                echo "Error: refusing to close an unexpected target lifecycle-lock descriptor." >&2
                return 1
                ;;
        esac
    fi
    VIBE_TOOLSET_LIFECYCLE_LOCK_ACTIVE=0
    VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT=""
    VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET=""
    VIBE_TOOLSET_LIFECYCLE_LOCK_ACTION=""
    VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT_PATH=""
    VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET_PATH=""
    VIBE_TOOLSET_LIFECYCLE_LOCK_CHECKOUT_FD=""
    VIBE_TOOLSET_LIFECYCLE_LOCK_TARGET_FD=""
    _VIBE_TOOLSET_DIRECT_LOCK_PRESENT=0
}
