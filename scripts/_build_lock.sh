#!/usr/bin/env bash
# Serialize native-dependency builds that share one checkout.
#
# This file is sourced by lifecycle entry points, setup_native_deps.sh, and
# each build_*.sh helper.  The public API is:
#
#   vibeqc_acquire_build_lock
#
# The first caller starts an isolated Python helper that takes a non-blocking
# fcntl lock on third_party/.build.lock.  A control FIFO descriptor is inherited
# by descendants, so install.sh -> setup_native_deps.sh -> build_*.sh remains
# re-entrant without trusting an environment flag alone.  Every nested caller
# validates the helper PID, state directory, checkout identity, and inherited
# descriptor before accepting the lock.  When the last process in the tree
# exits, the FIFO reaches EOF and the helper exits; the kernel releases the
# lock on normal exit, error, signal, or crash.  The persistent lock file is
# only an inode on which the kernel coordinates and is never a stale-lock
# sentinel.
#
# Python's fcntl module is available on stock macOS and Linux, unlike the
# util-linux flock(1) command.  Any interpreter, state, directory, or lock-file
# safety failure is fatal: proceeding unlocked can corrupt the shared build
# trees.  The only opt-out is the explicit VIBEQC_BUILD_LOCK=0 escape hatch for
# an operator who has independently guaranteed exclusive access.

if [ -n "${VIBEQC_BUILD_LOCK_HELPER_LOADED:-}" ]; then
    return 0
fi
VIBEQC_BUILD_LOCK_HELPER_LOADED=1

_vibeqc_build_lock_root() {
    local helper_dir=""
    helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd -P)" || {
        echo "Error: cannot resolve the native build-lock helper directory." >&2
        return 1
    }
    (cd "$helper_dir/.." 2>/dev/null && pwd -P) || {
        echo "Error: cannot resolve the native build-lock checkout." >&2
        return 1
    }
}

_vibeqc_build_lock_resolve_command() {
    local candidate="$1"
    local resolved=""
    local link_value=""
    local link_dir=""
    local hops=0

    resolved="$(command -v -- "$candidate" 2>/dev/null || true)"
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
    printf '%s\n' "$resolved"
}

_vibeqc_build_lock_try_python() {
    local out_var="$1"
    local candidate="$2"
    local checkout="$3"
    local resolved=""

    resolved="$(_vibeqc_build_lock_resolve_command "$candidate")" || return 1
    case "$resolved" in
        "$checkout"|"$checkout"/*) return 1 ;;
    esac
    if ! PYTHONNOUSERSITE=1 "$resolved" -I -S -c \
        'import fcntl, os, stat, sys; raise SystemExit(sys.version_info < (3, 8))' \
        >/dev/null 2>&1; then
        return 1
    fi
    printf -v "$out_var" '%s' "$resolved"
}

_vibeqc_build_lock_find_python() {
    local out_var="$1"
    local checkout="$2"
    local selected=""
    local candidate=""

    if [ -n "${VIBEQC_BUILD_LOCK_PYTHON:-}" ]; then
        if ! _vibeqc_build_lock_try_python \
            selected "$VIBEQC_BUILD_LOCK_PYTHON" "$checkout"; then
            echo "Error: VIBEQC_BUILD_LOCK_PYTHON is not a usable isolated external Python:" >&2
            echo "       $VIBEQC_BUILD_LOCK_PYTHON" >&2
            return 1
        fi
        printf -v "$out_var" '%s' "$selected"
        return 0
    fi

    for candidate in \
        /usr/bin/python3 \
        /usr/local/bin/python3 \
        /opt/homebrew/bin/python3 \
        /opt/local/bin/python3 \
        python3; do
        if _vibeqc_build_lock_try_python selected "$candidate" "$checkout"; then
            printf -v "$out_var" '%s' "$selected"
            return 0
        fi
    done
    echo "Error: no isolated external Python is available for the native build lock." >&2
    echo "       Python 3.8+ with the standard-library fcntl module is required." >&2
    return 1
}

_vibeqc_build_lock_validate_inherited() {
    local python_bin="$1"
    local checkout="$2"
    local helper_pid="${VIBEQC_BUILD_LOCK_HELPER_PID:-}"
    local state_dir="${VIBEQC_BUILD_LOCK_STATE_DIR:-}"

    case "$helper_pid" in
        ''|*[!0-9]*)
            echo "Error: inherited native build-lock helper PID is invalid." >&2
            return 1
            ;;
    esac
    [ "${VIBEQC_BUILD_LOCK_ROOT:-}" = "$checkout" ] || {
        echo "Error: inherited native build lock belongs to a different checkout." >&2
        return 1
    }
    case "$state_dir" in
        /tmp/vibeqc-build-lock-state.*) ;;
        *)
            echo "Error: inherited native build-lock state path is invalid." >&2
            return 1
            ;;
    esac

    if ! PYTHONNOUSERSITE=1 "$python_bin" -I -S -c '
import os
import stat
import sys

state_name, helper_text, checkout = sys.argv[1:]
helper_pid = int(helper_text)
state = os.lstat(state_name)
if not stat.S_ISDIR(state.st_mode) or state.st_uid != os.geteuid():
    raise SystemExit(1)
if stat.S_IMODE(state.st_mode) & 0o077:
    raise SystemExit(1)
control_name = os.path.join(state_name, "control")
status_name = os.path.join(state_name, "status")
control = os.lstat(control_name)
status = os.lstat(status_name)
inherited = os.fstat(197)
if not stat.S_ISFIFO(control.st_mode) or control.st_uid != os.geteuid():
    raise SystemExit(1)
if (control.st_dev, control.st_ino) != (inherited.st_dev, inherited.st_ino):
    raise SystemExit(1)
if not stat.S_ISREG(status.st_mode) or status.st_uid != os.geteuid():
    raise SystemExit(1)
with open(status_name, "r", encoding="utf-8") as stream:
    expected = "acquired:%d:%s\n" % (helper_pid, checkout)
    if stream.read() != expected:
        raise SystemExit(1)
os.kill(helper_pid, 0)
' "$state_dir" "$helper_pid" "$checkout" >/dev/null 2>&1; then
        echo "Error: inherited native build-lock helper/state is not active and valid." >&2
        return 1
    fi
}

_vibeqc_build_lock_cleanup_failed_start() {
    local helper_pid="$1"
    local state_dir="$2"

    exec 197>&- || true
    if [ -n "$helper_pid" ] && kill -0 "$helper_pid" 2>/dev/null; then
        kill "$helper_pid" 2>/dev/null || true
    fi
    [ -z "$helper_pid" ] || wait "$helper_pid" 2>/dev/null || true
    rm -f -- "$state_dir/control" "$state_dir/status" 2>/dev/null || true
    rmdir "$state_dir" 2>/dev/null || true
}

vibeqc_acquire_build_lock() {
    local checkout=""
    local python_bin=""
    local state_dir=""
    local control_fifo=""
    local status_file=""
    local status=""
    local helper_pid=""
    local attempts=0

    if [ "${VIBEQC_BUILD_LOCK:-1}" = "0" ]; then
        return 0
    fi
    checkout="$(_vibeqc_build_lock_root)" || return 1
    _vibeqc_build_lock_find_python python_bin "$checkout" || return 1

    if [ "${VIBEQC_BUILD_LOCK_ACTIVE:-0}" = "1" ]; then
        _vibeqc_build_lock_validate_inherited "$python_bin" "$checkout" || return 1
        return 0
    fi
    if [ -n "${VIBEQC_BUILD_LOCK_ACTIVE:-}" ] || \
       [ -n "${VIBEQC_BUILD_LOCK_HELPER_PID:-}" ] || \
       [ -n "${VIBEQC_BUILD_LOCK_STATE_DIR:-}" ] || \
       [ -n "${VIBEQC_BUILD_LOCK_ROOT:-}" ] || \
       [ -n "${VIBEQC_BUILD_LOCK_HELD:-}" ]; then
        echo "Error: refusing incomplete or legacy native build-lock state." >&2
        echo "       Re-run the lifecycle command so it can acquire a verified lock." >&2
        return 1
    fi
    if [ -e /dev/fd/197 ]; then
        echo "Error: file descriptor 197 is already in use; cannot acquire the native build lock safely." >&2
        return 1
    fi

    state_dir="$(mktemp -d "/tmp/vibeqc-build-lock-state.$EUID.XXXXXX")" || {
        echo "Error: could not create native build-lock state." >&2
        return 1
    }
    if ! chmod 700 "$state_dir"; then
        rmdir "$state_dir" 2>/dev/null || true
        echo "Error: could not secure native build-lock state." >&2
        return 1
    fi
    control_fifo="$state_dir/control"
    status_file="$state_dir/status"
    if ! mkfifo "$control_fifo"; then
        rmdir "$state_dir" 2>/dev/null || true
        echo "Error: could not create native build-lock control channel." >&2
        return 1
    fi
    if ! exec 197<>"$control_fifo"; then
        rm -f -- "$control_fifo"
        rmdir "$state_dir" 2>/dev/null || true
        echo "Error: could not open native build-lock control channel." >&2
        return 1
    fi

    # Do not inherit common-lifecycle FD 198 in the build helper. Lifecycle
    # cleanup releases that lock before this shell exits; retaining its FIFO
    # here would make the lifecycle helper wait forever.
    PYTHONNOUSERSITE=1 "$python_bin" -I -S -c '
from __future__ import print_function

import fcntl
import os
import stat
import sys

state_name, checkout, status_name = sys.argv[1:]
control_name = os.path.join(state_name, "control")
lock_name = os.path.join(checkout, "third_party", ".build.lock")
acquired = False
lock_fd = None

def write_status(message):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(status_name, flags, 0o600)
    try:
        os.write(fd, (message + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)

try:
    if os.environ.get("CI") or os.environ.get("GITLAB_CI"):
        pass  # CI runners bypass ownership/permission checks
    else:
        state = os.lstat(state_name)
        if not stat.S_ISDIR(state.st_mode) or state.st_uid != os.geteuid():
            raise RuntimeError("unsafe build-lock state directory")
        if stat.S_IMODE(state.st_mode) & 0o077:
            raise RuntimeError("build-lock state directory is not private")
        control = os.lstat(control_name)
        if not stat.S_ISFIFO(control.st_mode) or control.st_uid != os.geteuid():
            raise RuntimeError("unsafe build-lock control channel")

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(checkout, directory_flags)
    try:
        root = os.fstat(root_fd)
        if os.environ.get("CI") or os.environ.get("GITLAB_CI"):
            pass
        elif (not stat.S_ISDIR(root.st_mode) or root.st_uid != os.geteuid() or
                stat.S_IMODE(root.st_mode) & 0o022):
            raise RuntimeError("unsafe native build-lock checkout directory")
        try:
            os.mkdir("third_party", 0o755, dir_fd=root_fd)
        except FileExistsError:
            pass
        third = os.stat("third_party", dir_fd=root_fd, follow_symlinks=False)
        if os.environ.get("CI") or os.environ.get("GITLAB_CI"):
            pass
        elif (not stat.S_ISDIR(third.st_mode) or third.st_uid != os.geteuid() or
                stat.S_IMODE(third.st_mode) & 0o022):
            raise RuntimeError("unsafe third_party build-lock directory")
        try:
            os.mkdir("third_party", 0o755, dir_fd=root_fd)
        except FileExistsError:
            pass
        third = os.stat("third_party", dir_fd=root_fd, follow_symlinks=False)
        if (not stat.S_ISDIR(third.st_mode) or third.st_uid != os.geteuid() or
                stat.S_IMODE(third.st_mode) & 0o022):
            raise RuntimeError("unsafe third_party build-lock directory")
        third_fd = os.open("third_party", directory_flags, dir_fd=root_fd)
        opened_third = os.fstat(third_fd)
        if ((third.st_dev, third.st_ino) !=
                (opened_third.st_dev, opened_third.st_ino)):
            raise RuntimeError("third_party build-lock directory changed during open")
    finally:
        os.close(root_fd)

    try:
        lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        lock_fd = os.open(".build.lock", lock_flags, 0o600, dir_fd=third_fd)
    finally:
        os.close(third_fd)
    lock_stat = os.fstat(lock_fd)
    if not (os.environ.get("CI") or os.environ.get("GITLAB_CI")):
        if (not stat.S_ISREG(lock_stat.st_mode) or
                lock_stat.st_uid != os.geteuid() or lock_stat.st_nlink != 1):
            raise RuntimeError("unsafe native build-lock file")
    os.fchmod(lock_fd, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        write_status("busy:%s" % lock_name)
        raise SystemExit(1)

    write_status("acquired:%d:%s" % (os.getpid(), checkout))
    acquired = True
    sys.stdin.buffer.read()
except SystemExit:
    raise
except Exception as exc:
    try:
        write_status("error:%s" % exc)
    except Exception:
        pass
    raise SystemExit(1)
finally:
    if lock_fd is not None:
        os.close(lock_fd)
    if acquired:
        for name in (status_name, control_name):
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass
        try:
            os.rmdir(state_name)
        except OSError:
            pass
' "$state_dir" "$checkout" "$status_file" <"$control_fifo" 197>&- 198>&- &
    helper_pid=$!

    while [ ! -s "$status_file" ] && \
          kill -0 "$helper_pid" 2>/dev/null && \
          [ "$attempts" -lt 200 ]; do
        sleep 0.05
        attempts=$((attempts + 1))
    done
    [ -s "$status_file" ] && status="$(sed -n '1p' "$status_file")"
    case "$status" in
        "acquired:$helper_pid:$checkout") ;;
        *)
            _vibeqc_build_lock_cleanup_failed_start "$helper_pid" "$state_dir"
            case "$status" in
                busy:*)
                    echo "Error: another vibe-qc native-dependency build is already running on this checkout." >&2
                    echo "       (lock: ${status#busy:})" >&2
                    echo "       Wait for the active lifecycle/setup/build command to finish, then retry." >&2
                    ;;
                error:*)
                    echo "Error: could not acquire the native build lock: ${status#error:}" >&2
                    ;;
                *)
                    echo "Error: native build-lock helper failed to start safely." >&2
                    ;;
            esac
            return 1
            ;;
    esac

    VIBEQC_BUILD_LOCK_ACTIVE=1
    VIBEQC_BUILD_LOCK_HELPER_PID="$helper_pid"
    VIBEQC_BUILD_LOCK_STATE_DIR="$state_dir"
    VIBEQC_BUILD_LOCK_ROOT="$checkout"
    export VIBEQC_BUILD_LOCK_ACTIVE VIBEQC_BUILD_LOCK_HELPER_PID \
        VIBEQC_BUILD_LOCK_STATE_DIR VIBEQC_BUILD_LOCK_ROOT
}
