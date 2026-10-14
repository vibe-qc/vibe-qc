# scripts/_safe_build_env.sh — build-pressure safety helper.
#
# Sourced (NOT executed) from every entry-point script that triggers
# heavy compilation: update.sh, setup_native_deps.sh, and each
# build_*.sh. Three side-effects, all idempotent:
#
#   1. On Linux, self-re-execs the *caller* under `nice -n 19 ionice -c 3`
#      so the build runs at idle CPU + IO priority. Guarded by the
#      VIBEQC_BUILD_NICED env var to prevent infinite recursion when a
#      chain of scripts each source the helper (e.g. update.sh →
#      setup_native_deps.sh → build_libint.sh). Only the outermost
#      caller re-execs; inner scripts inherit the niced process.
#
#   2. Verifies cmake + ninja are on PATH and bails with a per-distro
#      install hint if not. setup_native_deps.sh runs a much more
#      comprehensive preflight, but each build_*.sh can also be invoked
#      directly (e.g. for a single-dep rebuild after a version bump),
#      in which case we still want a clean upfront error rather than a
#      cryptic "cmake: command not found" or "could not find ninja"
#      mid-configure. Guarded by VIBEQC_TOOLING_CHECKED so a chain of
#      sourced helpers only runs the check once.
#
#   3. Sets CMAKE_BUILD_PARALLEL_LEVEL to a memory-safe default if the
#      caller hasn't set it already (#411):
#         min(nproc, max(1, usable_mb // budget_mb), 8)
#      where usable = min(MemAvailable, headroom of every enclosing
#      cgroup) — the memory the build can actually get, not MemTotal —
#      and budget is the measured per-worker peak of the phase (5000 MB
#      for libint's library, 15000 MB for everything else). The level is
#      recomputed at each build start (each build_*.sh sourcing, and
#      before the core's pip build), and the hard cap of 8 prevents
#      ninja's link/IO contention regime on monster boxes. Details and
#      measurements at section (3) below.
#
#      Pre-existing CMAKE_BUILD_PARALLEL_LEVEL in env is honored
#      verbatim — explicit user / parent-script intent wins.
#
# Example: a host whose GPU driver reserves part of RAM (MemTotal ~60 GB,
# of which ~36 GB is held by the amdgpu driver, MemAvailable ~24 GB): the
# MemTotal formula gave 4 workers everywhere and was OOM-killed (#411); now
# libint builds with 4 workers and the core with 1 (with a warning line).
#
# macOS / non-Linux: the /proc/meminfo gate fails, both side-effects
# skip, and the caller's existing PHYS_CORES fallback governs
# parallelism. macOS dev boxes have less RAM but clang is more
# memory-efficient than gcc, and the user is at the keyboard — Ctrl-C
# is always available. The build-pressure trap is specifically a
# Linux + gcc + compute-host-d-shaped-host problem.
#
# History: the helper was extracted on 2026-05-16 after two hard
# resets on compute-host-d caused by unbounded `ninja` parallelism. Before
# the helper, each build_*.sh script unconditionally set
# CMAKE_BUILD_PARALLEL_LEVEL=PHYS_CORES, which silently undid any cap
# set by the calling update.sh. Centralizing the logic here ensures
# the cap propagates through the whole script chain.
#
# USAGE (in any entry-point script):
#     set -euo pipefail
#     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     # shellcheck source=_safe_build_env.sh
#     . "$SCRIPT_DIR/_safe_build_env.sh"
#
# Source BEFORE any heavy work, AFTER SCRIPT_DIR is set. The re-exec
# block depends on BASH_SOURCE[1] resolving to the caller's path.
#
# OPT-OUTS:
#   CMAKE_BUILD_PARALLEL_LEVEL=N   — pin parallelism (skips cap formula)
#   VIBEQC_BUILD_NICED=1           — skip nice/ionice re-exec
#   VIBEQC_BUILD_ENV_QUIET=1       — suppress the cap announcement line
#   VIBEQC_SKIP_PREFLIGHT=1        — skip the cmake+ninja tooling check
#   VIBEQC_TOOLING_CHECKED=1       — internal marker set by side-effect (2)

# ---------------------------------------------------------------------------
# (1) Re-exec the caller under nice/ionice on Linux, once.
# ---------------------------------------------------------------------------
if [ -z "${VIBEQC_BUILD_NICED:-}" ] && [ -r /proc/meminfo ]; then
    _vqsafe_prefix=()
    if command -v nice >/dev/null 2>&1; then
        _vqsafe_prefix+=(nice -n 19)
    fi
    if command -v ionice >/dev/null 2>&1; then
        _vqsafe_prefix+=(ionice -c 3)
    fi
    if [ ${#_vqsafe_prefix[@]} -gt 0 ]; then
        export VIBEQC_BUILD_NICED=1
        # BASH_SOURCE[1] is the script that sourced us; we want to
        # restart it under the niced prefix with its ORIGINAL argv.
        #
        # Naively using "$@" here breaks if the caller already shifted
        # args out before sourcing us (update.sh does this: parses
        # --dev / --branch / etc. before the source). In that case "$@"
        # is empty, the re-exec drops every flag, and update.sh's
        # default-branch fallback silently kicks in (~line 179:
        # `[ -z "$BRANCH" ] && BRANCH="release"`).
        #
        # _VIBEQC_UPDATE_ORIG_ARGS is the contract: a caller that
        # parses argv before sourcing us snapshots its original argv
        # in this array. Callers that source us BEFORE their parser
        # runs (current state for build_*.sh) have nothing to
        # snapshot and fall back to "$@" — which is correct because
        # their argv hasn't been shifted yet.
        #
        # `exec` replaces the calling bash process, so the entire
        # caller restarts under the prefix. Child scripts sourced
        # later inherit VIBEQC_BUILD_NICED=1 and skip this block.
        if [ "${_VIBEQC_UPDATE_ORIG_ARGS+set}" = "set" ]; then
            exec "${_vqsafe_prefix[@]}" bash "${BASH_SOURCE[1]}" "${_VIBEQC_UPDATE_ORIG_ARGS[@]}"
        else
            exec "${_vqsafe_prefix[@]}" bash "${BASH_SOURCE[1]}" "$@"
        fi
    fi
    unset _vqsafe_prefix
fi

# ---------------------------------------------------------------------------
# (2) Minimal tooling check — cmake + ninja must be on PATH for any
# build_*.sh to succeed. setup_native_deps.sh runs a much more
# comprehensive preflight, but each build_*.sh can also be invoked
# directly (e.g. for a single-dep rebuild after a version bump), in
# which case we still want a clean upfront error rather than letting
# `cmake -GNinja` fail mid-configure with a much less obvious message.
#
# Guarded by VIBEQC_TOOLING_CHECKED so a chain of sourced helpers
# only runs the check once. Skipped when VIBEQC_SKIP_PREFLIGHT=1
# (parent script already preflighted) or VIBEQC_BUILD_ENV_QUIET=1
# (suppress all helper noise).
# ---------------------------------------------------------------------------
if [ -z "${VIBEQC_TOOLING_CHECKED:-}" ] && [ "${VIBEQC_SKIP_PREFLIGHT:-0}" != "1" ]; then
    _vqsafe_missing=""
    command -v cmake >/dev/null 2>&1 || _vqsafe_missing="$_vqsafe_missing cmake"
    command -v ninja >/dev/null 2>&1 || _vqsafe_missing="$_vqsafe_missing ninja"
    if [ -n "$_vqsafe_missing" ]; then
        echo "Error: required build tool(s) not found on PATH:$_vqsafe_missing" >&2
        echo >&2
        case "$(uname -s)" in
            Darwin)
                echo "    brew install cmake ninja" >&2
                ;;
            Linux)
                echo "    Arch / Manjaro:  sudo pacman -S cmake ninja" >&2
                echo "    Debian / Ubuntu: sudo apt install cmake ninja-build" >&2
                echo "    Fedora / RHEL:   sudo dnf install cmake ninja-build" >&2
                ;;
        esac
        echo >&2
        echo "For the full preflight (BLAS, headers, python, …), run:" >&2
        echo "    ./scripts/setup_native_deps.sh" >&2
        unset _vqsafe_missing
        exit 1
    fi
    unset _vqsafe_missing
    export VIBEQC_TOOLING_CHECKED=1
fi

# ---------------------------------------------------------------------------
# (3) Size CMAKE_BUILD_PARALLEL_LEVEL from the memory the build can get.
#
# #411: this used to read MemTotal. MemTotal is not headroom: on a host whose
# GPU driver reserves part of RAM, ~36 GB of its ~60 GB is held by the amdgpu
# driver, MemAvailable was ~24 GB, and the MemTotal-sized level (4) was
# OOM-killed. The usable figure is now
#
#     usable = min(MemAvailable, cgroup headroom of every enclosing cgroup)
#
# where a cgroup's headroom is its limit minus its working set
# (usage - inactive_file, the figure the kernel reclaims last). The level is
#
#     min(nproc, max(1, usable_mb // budget_mb), 8)
#
# evaluated again every time the helper is sourced (each build_*.sh start) and
# by vibeqc_pip_install_editable before the core build, so a long update
# resizes each phase from the memory free at that moment. A value this helper
# derived carries the marker VIBEQC_BUILD_PARALLEL_AUTO; any other preset
# CMAKE_BUILD_PARALLEL_LEVEL is explicit intent and wins verbatim.
#
# The per-worker budget (VIBEQC_BUILD_MEM_PER_JOB_MB, default below) comes
# from the measured peak RSS of the heaviest translation units; see the
# comment on its default. A floor of 1 means a host with less usable memory
# than one budget builds serially and says so, rather than being OOM-killed.
#
# Non-Linux (no /proc/meminfo): unchanged — nothing is set and the callers'
# PHYS_CORES fallback governs.
#
# Test seams (internal): _VIBEQC_MEMINFO, _VIBEQC_PROC_SELF_CGROUP,
# _VIBEQC_CGROUP_ROOT, _VIBEQC_NPROC replace /proc/meminfo, /proc/self/cgroup,
# /sys/fs/cgroup and nproc.
# ---------------------------------------------------------------------------

# Print the tightest cgroup headroom in MB for this process, or nothing when
# no enclosing cgroup has a memory limit (or cgroups are unreadable).
_vqsafe_cgroup_headroom_mb() {
    local self="${_VIBEQC_PROC_SELF_CGROUP:-/proc/self/cgroup}"
    local root="${_VIBEQC_CGROUP_ROOT:-/sys/fs/cgroup}"
    [ -r "$self" ] || return 0
    local rel dir best="" max cur inactive head
    # cgroup v2: the single "0::<path>" line.
    rel=$(awk -F: '$1 == "0" && $2 == "" { print $3; exit }' "$self")
    if [ -n "$rel" ] && [ -r "$root/cgroup.controllers" ]; then
        dir="$root${rel%/}"
        while :; do
            if [ -r "$dir/memory.max" ]; then
                max=$(cat "$dir/memory.max" 2>/dev/null)
                if [ -n "$max" ] && [ "$max" != "max" ]; then
                    cur=$(cat "$dir/memory.current" 2>/dev/null || echo 0)
                    inactive=$(awk '$1 == "inactive_file" { print $2; exit }' \
                        "$dir/memory.stat" 2>/dev/null)
                    head=$(( (max - cur + ${inactive:-0}) / 1048576 ))
                    [ "$head" -lt 0 ] && head=0
                    if [ -z "$best" ] || [ "$head" -lt "$best" ]; then
                        best=$head
                    fi
                fi
            fi
            [ "$dir" = "$root" ] && break
            case "$dir" in "$root"/*) dir="${dir%/*}" ;; *) break ;; esac
        done
        [ -n "$best" ] && echo "$best"
        return 0
    fi
    # cgroup v1: the memory controller's hierarchical limit already folds in
    # every ancestor; treat values above physical memory as "no limit".
    rel=$(awk -F: '$2 ~ /(^|,)memory(,|$)/ { print $3; exit }' "$self")
    [ -n "$rel" ] || return 0
    dir="$root/memory${rel%/}"
    [ -r "$dir/memory.stat" ] || dir="$root/memory"
    [ -r "$dir/memory.stat" ] || return 0
    max=$(awk '$1 == "hierarchical_memory_limit" { print $2; exit }' "$dir/memory.stat")
    inactive=$(awk '$1 == "total_inactive_file" { print $2; exit }' "$dir/memory.stat")
    cur=$(cat "$dir/memory.usage_in_bytes" 2>/dev/null || echo 0)
    [ -n "$max" ] || return 0
    # 2^62 bytes or more is the kernel's "unlimited".
    [ "${#max}" -ge 19 ] && return 0
    head=$(( (max - cur + ${inactive:-0}) / 1048576 ))
    [ "$head" -lt 0 ] && head=0
    echo "$head"
}

# Per-worker memory budget (MB) for a build phase, from the measured peak RSS
# of its heaviest translation units (vibe-qc#411, single-TU /usr/bin/time
# measurements, 2026-09-29, GCC 16 and Apple clang):
#   * libint's library phase (~4.7k unity TUs): <= 1.8 GiB with GCC, <= 4.0 GiB
#     with clang per TU -> 5000 MB.
#   * the vibe-qc core (guess.cpp, df.cpp, periodic_*.cpp, bindings.cpp): 5-8
#     GiB with GCC per TU -> 15000 MB, the long-standing budget; the other
#     dependencies are unmeasured and keep it too.
# VIBEQC_BUILD_MEM_PER_JOB_MB overrides every phase.
_vqsafe_budget_mb() {
    case "${1:-}" in
        build_libint.sh) echo 5000 ;;
        *)               echo 15000 ;;
    esac
}

# vibeqc_refresh_build_parallelism [LABEL [BUDGET_MB]]
#   (Re)derive CMAKE_BUILD_PARALLEL_LEVEL unless it was set explicitly.
vibeqc_refresh_build_parallelism() {
    local meminfo="${_VIBEQC_MEMINFO:-/proc/meminfo}"
    local label="${1:-}"
    local default_budget="${2:-$(_vqsafe_budget_mb "$label")}"
    if [ -n "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ] && \
       [ "${CMAKE_BUILD_PARALLEL_LEVEL}" != "${VIBEQC_BUILD_PARALLEL_AUTO:-}" ]; then
        return 0
    fi
    [ -r "$meminfo" ] || return 0
    local nproc total avail source usable cgroup budget level note=""
    nproc="${_VIBEQC_NPROC:-$(nproc 2>/dev/null || echo 4)}"
    total=$(awk '/^MemTotal:/ { print int($2 / 1024); exit }' "$meminfo")
    avail=$(awk '/^MemAvailable:/ { print int($2 / 1024); exit }' "$meminfo")
    if [ -n "$avail" ]; then
        source="MemAvailable"; usable=$avail
    elif [ -n "$total" ]; then
        # Kernels before 3.14 have no MemAvailable.
        source="MemTotal(no MemAvailable)"; usable=$total
    else
        return 0
    fi
    cgroup=$(_vqsafe_cgroup_headroom_mb)
    if [ -n "$cgroup" ] && [ "$cgroup" -lt "$usable" ]; then
        usable=$cgroup; source="cgroup"
    fi
    budget="${VIBEQC_BUILD_MEM_PER_JOB_MB:-$default_budget}"
    case "$budget" in ''|*[!0-9]*|0) budget=$default_budget ;; esac
    level=$(( usable / budget ))
    if [ "$level" -lt 1 ]; then
        level=1
        note="; WARNING usable memory is below one worker's budget"
    fi
    [ "$level" -gt "$nproc" ] && level=$nproc
    [ "$level" -gt 8 ] && level=8
    export CMAKE_BUILD_PARALLEL_LEVEL=$level
    export VIBEQC_BUILD_PARALLEL_AUTO=$level
    if [ -z "${VIBEQC_BUILD_ENV_QUIET:-}" ]; then
        echo "==> CMAKE_BUILD_PARALLEL_LEVEL=$level${label:+ for $label} (nproc=$nproc, MemTotal=${total:-?}MB, MemAvailable=${avail:-?}MB, cgroup_headroom=${cgroup:-none}${cgroup:+MB}, usable=${usable}MB from $source; ${budget}MB-per-worker, cap@8$note)"
    fi
}

_vqsafe_caller="${BASH_SOURCE[1]:-}"
vibeqc_refresh_build_parallelism "${_vqsafe_caller##*/}"
unset _vqsafe_caller
