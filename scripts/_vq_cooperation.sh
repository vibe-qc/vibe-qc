# Cooperative local queue pause/resume for build scripts. Source this file,
# call vibeqc_vq_install_trap, then vibeqc_vq_pause before heavy compilation.
# VIBEQC_VQ_COOPERATE=0 disables cooperation; VQ_BIN overrides PATH lookup.
#
# Only the outer shell owns cleanup. Its exported owner/tag let nested build
# subprocesses borrow the pause without resuming it when their own EXIT fires.
# Re-sourcing in the same shell preserves ownership and the installed trap.
_VIBEQC_VQ_BIN="${VQ_BIN:-$(command -v vq 2>/dev/null || true)}"

vibeqc_vq_pause() {
    if [ "${VIBEQC_VQ_COOPERATE:-1}" = "0" ] || [ -z "$_VIBEQC_VQ_BIN" ]; then
        return 0
    fi
    if [ -n "${VIBEQC_VQ_PAUSE_OWNER:-}" ]; then
        return 0
    fi

    local result probe_status
    # localhost is explicit: a configured default_host may be remote.
    if "$_VIBEQC_VQ_BIN" daemon ping localhost --timeout 2 >/dev/null 2>&1; then
        :
    else
        probe_status=$?
        if [ "$probe_status" = 1 ]; then
            echo "[vq-cooperation] no local daemon; skipping queue cooperation" >&2
        else
            echo "[vq-cooperation] local daemon probe failed; skipping queue cooperation" >&2
        fi
        return 0
    fi

    VIBEQC_VQ_PAUSE_OWNER="${BASHPID:-$$}"
    VIBEQC_VQ_PAUSE_TAG="vibeqc-build-$VIBEQC_VQ_PAUSE_OWNER"
    export VIBEQC_VQ_PAUSE_OWNER VIBEQC_VQ_PAUSE_TAG
    # Retain cleanup ownership even on failure: a bulk command can have
    # paused some jobs before reporting an error or an unexpected response.
    if result=$("$_VIBEQC_VQ_BIN" pause localhost --all --paused-by "$VIBEQC_VQ_PAUSE_TAG" 2>&1); then
        if [[ "$result" =~ ^paused\ [0-9]+\ jobs?(\ \(.+\))?$ ]] && [[ "$result" != *$'\n'* ]] && [[ ! "$result" =~ [1-9][0-9]*\ error\(s\) ]]; then
            echo "[vq-cooperation] local queue: $result" >&2
        else
            echo "[vq-cooperation] warning: local pause returned an unexpected or error summary; tagged cleanup retained" >&2
        fi
    else
        echo "[vq-cooperation] warning: local pause failed; tagged cleanup retained; continuing" >&2
    fi
    return 0
}

vibeqc_vq_resume() {
    if [ "${VIBEQC_VQ_PAUSE_OWNER:-}" != "${BASHPID:-$$}" ] || [ -z "${VIBEQC_VQ_PAUSE_TAG:-}" ]; then
        return 0
    fi
    local result
    if result=$("$_VIBEQC_VQ_BIN" resume localhost --all --paused-by "$VIBEQC_VQ_PAUSE_TAG" 2>&1); then
        if [[ "$result" =~ ^resumed\ [0-9]+\ jobs?(\ \(.+\))?(\ \[.+\])?$ ]] && [[ "$result" != *$'\n'* ]] && [[ ! "$result" =~ [1-9][0-9]*\ error\(s\) ]]; then
            echo "[vq-cooperation] local queue: $result" >&2
            unset VIBEQC_VQ_PAUSE_OWNER VIBEQC_VQ_PAUSE_TAG
        else
            echo "[vq-cooperation] warning: local resume returned an unexpected or error summary; check tagged local jobs" >&2
        fi
    else
        echo "[vq-cooperation] warning: local resume failed; check tagged local jobs" >&2
    fi
    return 0
}

vibeqc_vq_install_trap() {
    # This marker is shell-local. A subprocess must install its own trap,
    # whose resume is a no-op because it borrowed the outer shell's pause.
    if [ "${_VIBEQC_VQ_TRAP_OWNER:-}" = "${BASHPID:-$$}" ]; then
        return 0
    fi
    _VIBEQC_VQ_TRAP_OWNER="${BASHPID:-$$}"
    local printed existing
    printed=$(trap -p EXIT 2>/dev/null)
    if [ -z "$printed" ]; then
        trap vibeqc_vq_resume EXIT
        return 0
    fi
    # Bash prints shell-quoted trap bodies; eval decodes that quoting without
    # executing the body. Preserve quotes, backslashes and multiline traps.
    existing=${printed#trap -- }
    existing=${existing% EXIT}
    eval "existing=$existing"
    # shellcheck disable=SC2064
    trap "vibeqc_vq_resume; $existing" EXIT
}
