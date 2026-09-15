# scripts/_build_log.sh — quiet-build logging helper.
#
# Sourced (NOT executed) by build scripts whose compile step emits more
# output than a log consumer can hold. The motivating incident: the three
# v0.15.131 release-gate build-test jobs each failed ~162 min in, but the
# libint codegen chatter had overflowed the GitLab runner's 4 MB trace cap
# at minute ~35 — the actual error was never collected, three times over
# (the 5_4_3-era cold build already wrote 3.9 MB of codegen dump; 5_4_4
# writes more). Routing the bulk output to a file keeps the visible trace
# small enough that the failure line always survives.
#
# Usage (from a build_*.sh, after set -euo pipefail):
#
#     . "$SCRIPT_DIR/_build_log.sh"
#     vibeqc_build_log_begin "$DEP_DIR/build.log" "libint2"
#     cmake --build "$BUILD_DIR" --target install    # chatter -> build.log
#     vibeqc_build_log_end
#
# Contract:
#
#   * Between begin and end, the shell's stdout+stderr go to the log file.
#     The original stdout receives one heartbeat line every
#     VIBEQC_BUILD_LOG_HEARTBEAT_SECS (default 60) seconds: elapsed
#     minutes, the last ninja "[N/M]" progress line seen in the log, and
#     disk/memory headroom — the numbers a post-mortem needs. Note that
#     ninja buffers a step's output and flushes it when the step FINISHES,
#     so the log legitimately stays still during long single steps (libint
#     codegen is one 20-30 min step); elapsed time is the liveness signal,
#     not log growth.
#
#   * If the shell exits between begin and end (set -e after a failed
#     build), the EXIT trap restores the original stdout, replays the last
#     200 log lines plus a disk/memory snapshot, and preserves the exit
#     code. The full log file is retained either way.
#
#   * VIBEQC_BUILD_VERBOSE=1 turns begin/end into no-ops: output streams
#     to the terminal exactly as it did before this helper existed.
#
# The helper is logging-only — it never alters what gets built — so build
# scripts source and call it from inside `vibeqc-recipe-hash:
# lifecycle-only` regions and changes here never register as recipe drift
# (no fleet rebuilds for a logging tweak; see _native_stamp.sh).

_VIBEQC_BLOG_FILE=""
_VIBEQC_BLOG_LABEL="build"
_VIBEQC_BLOG_T0=0
_VIBEQC_BLOG_HB_PID=""
_VIBEQC_BLOG_ACTIVE=0

# One "  disk_avail=…G[  mem_avail=…G]" fragment, no trailing newline.
# mem_avail only where /proc/meminfo exists (Linux — the environment the
# incident lives in); macOS callers just get the disk number.
_vibeqc_build_log_resources() {
    df -Pk "$(dirname "$_VIBEQC_BLOG_FILE")" 2>/dev/null \
        | awk 'NR==2 {printf "  disk_avail=%.1fG", $4/1048576}'
    if [ -r /proc/meminfo ]; then
        awk '/^MemAvailable:/ {printf "  mem_avail=%.1fG", $2/1048576}' /proc/meminfo
    fi
}

# Background loop writing heartbeats to the saved original stdout (fd 3).
# Runs with set +e: a heartbeat must never die because grep found no
# "[N/M]" line yet (pipefail would sink the whole subshell otherwise).
_vibeqc_build_log_heartbeat() {
    set +e
    # Drop every inherited descriptor except stdio and the saved stdout
    # (fd 3) — plus bash's own script fd (255, a harmless regular file).
    # The calling build script holds long-lived descriptors this loop must
    # not pin: concretely, _build_lock.sh keeps its lock FIFO's write end
    # at a high fd, and an orphaned sleep inheriting it kept the lock
    # holder — and with it the caller's stdout pipe — alive for up to one
    # full interval after the script had already exited.
    local fd
    for fd in $(command ls /dev/fd 2>/dev/null); do
        case "$fd" in
            0|1|2|3|255) ;;
            *) eval "exec ${fd}>&-" 2>/dev/null ;;
        esac
    done
    local interval
    interval="${VIBEQC_BUILD_LOG_HEARTBEAT_SECS:-60}"
    while :; do
        # Two deliberate quirks make the `kill` in end()/fail() take effect
        # immediately instead of after up to a full interval:
        #   * `sleep &` + the `wait` BUILTIN — bash defers an untrapped
        #     SIGTERM until the foreground command finishes, but signals
        #     interrupt `wait` at once (observed as a 60 s exit hang
        #     otherwise);
        #   * 3>&- 4>&- — the orphaned sleep must not inherit the saved
        #     stdout/stderr fds, or it keeps the caller's stdout pipe open
        #     (an `install.sh | tee` style consumer would block on EOF
        #     until the sleep expires).
        sleep "$interval" 3>&- 4>&- &
        wait $! 2>/dev/null
        {
            printf '[%s build +%smin]' "$_VIBEQC_BLOG_LABEL" \
                "$((($(date +%s) - _VIBEQC_BLOG_T0) / 60))"
            tail -c 100000 "$_VIBEQC_BLOG_FILE" 2>/dev/null \
                | grep -E '^\[[0-9]+/[0-9]+\]' | tail -n 1 \
                | awk '{printf " %s", substr($0, 1, 110)}'
            _vibeqc_build_log_resources
            printf '\n'
        } >&3 2>/dev/null
    done
}

# EXIT trap while the redirect is active: restore stdout/stderr, replay
# the log tail, keep the exit code.
_vibeqc_build_log_fail() {
    rc=$?
    set +e
    trap - EXIT
    if [ -n "${_VIBEQC_BLOG_HB_PID:-}" ]; then
        kill "$_VIBEQC_BLOG_HB_PID" 2>/dev/null
    fi
    exec 1>&3 3>&- 2>&4 4>&-
    echo "ERROR: $_VIBEQC_BLOG_LABEL build failed (exit $rc) after $((($(date +%s) - _VIBEQC_BLOG_T0) / 60)) min."
    echo "------ last 200 lines of $_VIBEQC_BLOG_FILE ------"
    tail -n 200 "$_VIBEQC_BLOG_FILE" 2>/dev/null
    echo "------ end of log tail (full log kept at $_VIBEQC_BLOG_FILE) ------"
    _vibeqc_build_log_resources
    echo
    exit "$rc"
}

vibeqc_build_log_begin() {
    _VIBEQC_BLOG_FILE="$1"
    _VIBEQC_BLOG_LABEL="${2:-build}"
    if [ "${VIBEQC_BUILD_VERBOSE:-0}" = "1" ]; then
        _VIBEQC_BLOG_ACTIVE=0
        return 0
    fi
    : > "$_VIBEQC_BLOG_FILE"
    echo "  (full $_VIBEQC_BLOG_LABEL build output -> $_VIBEQC_BLOG_FILE;"
    echo "   one heartbeat line per minute here. Stream it live with:"
    echo "       tail -f $_VIBEQC_BLOG_FILE"
    echo "   or set VIBEQC_BUILD_VERBOSE=1 to stream into this terminal.)"
    _VIBEQC_BLOG_T0="$(date +%s)"
    exec 3>&1 4>&2 >>"$_VIBEQC_BLOG_FILE" 2>&1
    _vibeqc_build_log_heartbeat &
    _VIBEQC_BLOG_HB_PID=$!
    _VIBEQC_BLOG_ACTIVE=1
    trap '_vibeqc_build_log_fail' EXIT
}

vibeqc_build_log_end() {
    if [ "${_VIBEQC_BLOG_ACTIVE:-0}" != "1" ]; then
        return 0
    fi
    trap - EXIT
    kill "$_VIBEQC_BLOG_HB_PID" 2>/dev/null || true
    wait "$_VIBEQC_BLOG_HB_PID" 2>/dev/null || true
    _VIBEQC_BLOG_HB_PID=""
    _VIBEQC_BLOG_ACTIVE=0
    exec 1>&3 3>&- 2>&4 4>&-
    echo "$_VIBEQC_BLOG_LABEL build finished in $((($(date +%s) - _VIBEQC_BLOG_T0) / 60)) min (full log: $_VIBEQC_BLOG_FILE)"
}
