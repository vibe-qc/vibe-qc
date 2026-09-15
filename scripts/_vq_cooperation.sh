# scripts/_vq_cooperation.sh — cooperative pause/resume of the vq queue
# around heavy builds.
#
# Sourced (NOT executed) by install.sh and update.sh. Provides three
# functions; the caller decides when to pause and where to install the
# resume trap.
#
# Background
# ----------
# 2026-05-18 ~07:50 EDT compute-host-d wedged hard (kernel OOM cascade →
# systemd-logind watchdog → unresponsive, manual reboot at 12:43).
# Forensics in journalctl pinned the trigger as combined interactive
# desktop load (Steam) + running vq job pushing past the 125 GB cliff.
#
# The vq side ships its own auto-pause in v0.6.20+ (host-pressure
# watchdog, lower default RSS ceiling). This helper closes the
# complementary script-side case: an operator runs `bash scripts/update.sh`
# or `bash scripts/install.sh` directly while vq jobs are running on
# the same host. The wrapper scripts trigger heavy compilation
# (cmake + ninja with parallel cc1plus workers) that can transiently
# spike tens of GB; combined with a running vq job, the box can
# repeat the 2026-05-18 cascade.
#
# `vq admin update` already does this internally — it pauses the
# queue, runs the build, resumes the queue. This script-side helper
# is for when the operator bypasses that wrapper.
#
# See vibe-queue/docs/handover-update-script-cooperative-pause.md for
# the full post-mortem and edge-case rationale.
#
# Functions exposed
# -----------------
#
#   vibeqc_vq_pause
#       SIGSTOP every running vq job via `vq pause --all`. Records
#       that *this* script did the pausing in VIBEQC_VQ_PAUSED_BY_US
#       so the paired resume is a no-op if pause didn't fire (vq not
#       installed, daemon down, etc.). Idempotent.
#
#   vibeqc_vq_resume
#       SIGCONT every suspended vq job via `vq resume --all`. Only
#       fires if vibeqc_vq_pause set the flag — won't resume jobs we
#       didn't pause. Idempotent.
#
#   vibeqc_vq_install_trap
#       Installs `vibeqc_vq_resume` as an EXIT trap so the queue gets
#       resumed even on build failure, Ctrl-C, or signal. Composes
#       with an existing EXIT trap (preserves it) rather than
#       clobbering — defensive for callers that might add their own
#       cleanup later.
#
# Edge cases handled
# ------------------
#
#   * vq not on PATH       — pause silently no-ops; resume also no-ops.
#                            Useful on fresh hosts that don't have vq
#                            installed yet, and on laptops where the
#                            script is run in a worktree without a
#                            daemon.
#
#   * vq daemon down       — `vq pause --all` exits non-zero; we warn
#                            and continue. A down daemon means there
#                            are no running jobs to protect anyway.
#
#   * script crashes       — the EXIT trap fires resume on any exit
#                            (clean, error, or signal). Queue doesn't
#                            get stranded paused.
#
#   * no jobs in queue     — `vq pause --all` reports "paused 0 jobs"
#                            and exits 0; resume similarly reports 0.
#                            No-op end-to-end.
#
#   * pre-existing pauses  — `vq pause --all` skips already-SUSPENDED
#                            jobs. `vq resume --all` resumes EVERY
#                            suspended job, including ones the
#                            operator paused manually before invoking
#                            this script. Known behavior difference vs
#                            `vq admin update`, which tracks the exact
#                            jobids IT paused. Acceptable for compute-host-d
#                            where operators don't pause jobs by hand.
#                            See the vq handover for the more-precise
#                            per-jobid variant.
#
#   * set -e in caller     — every external call uses `|| true` so the
#                            helper never trips the caller's set -e /
#                            pipefail. The script's existing exit-on-
#                            error semantics are preserved for the
#                            actual build steps.
#
# Opt-out
# -------
#
#   VIBEQC_VQ_COOPERATE=0  — skip pause/resume entirely. Useful for CI
#                            (no daemon to coordinate with) and for
#                            scripts running inside `vq admin update`
#                            itself (it already paused the queue —
#                            double-pause would be harmless but
#                            confusing in the logs).

# Resolve vq binary once. VQ_BIN env var wins (lets the operator
# point at a non-default install); otherwise PATH lookup. The
# `|| true` shields a `command -v` miss from set -e in the caller.
_VIBEQC_VQ_BIN="${VQ_BIN:-$(command -v vq 2>/dev/null || true)}"
VIBEQC_VQ_PAUSED_BY_US=0

vibeqc_vq_pause() {
    if [ "${VIBEQC_VQ_COOPERATE:-1}" = "0" ]; then
        return 0
    fi
    if [ -z "$_VIBEQC_VQ_BIN" ]; then
        return 0
    fi
    if "$_VIBEQC_VQ_BIN" pause --all >/dev/null 2>&1; then
        VIBEQC_VQ_PAUSED_BY_US=1
        echo "[vq-cooperation] paused vq queue for the build" >&2
    else
        echo "[vq-cooperation] vq pause --all failed (daemon down?); continuing" >&2
    fi
}

vibeqc_vq_resume() {
    if [ "$VIBEQC_VQ_PAUSED_BY_US" != "1" ]; then
        return 0
    fi
    if "$_VIBEQC_VQ_BIN" resume --all >/dev/null 2>&1; then
        echo "[vq-cooperation] resumed vq queue" >&2
    else
        echo "[vq-cooperation] vq resume --all failed; check 'vq queue --active' manually" >&2
    fi
}

vibeqc_vq_install_trap() {
    # Compose with any pre-existing EXIT trap so we don't clobber a
    # caller-installed cleanup. `trap -p EXIT` prints either nothing
    # (no trap set) or "trap -- 'BODY' EXIT" — bash's format is stable
    # across versions back to 4.x. If non-empty, extract BODY and
    # prepend our resume call.
    local printed existing
    printed=$(trap -p EXIT 2>/dev/null)
    if [ -z "$printed" ]; then
        trap vibeqc_vq_resume EXIT
        return
    fi
    existing=$(printf '%s' "$printed" | sed -E "s/^trap -- '(.*)' EXIT$/\1/")
    # shellcheck disable=SC2064  # intentional early expansion of $existing
    trap "vibeqc_vq_resume; $existing" EXIT
}
