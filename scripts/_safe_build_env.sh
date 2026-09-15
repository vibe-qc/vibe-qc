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
#      caller hasn't set it already:
#         min(nproc, max(2, mem_mb // 15000), 8)
#      ...which keeps `ninja` from firing more cc1plus workers than the
#      host has RAM headroom for. Each cc1plus on vibe-qc's
#      template-heavy headers (libint integrals, periodic_*.cpp,
#      gradient.cpp) peaks at 8–10 GB resident; budgeting 15 GB/worker
#      leaves a clean 50% margin, and the hard cap of 8 prevents
#      ninja's link/IO contention regime on monster boxes.
#
#      Pre-existing CMAKE_BUILD_PARALLEL_LEVEL in env is honored
#      verbatim — explicit user / parent-script intent wins.
#
# Concrete fleet numbers under this cap:
#
#   compute-host-d  (32 th, 125 GB) → 8 workers (≤80 GB peak, ≥45 GB headroom)
#   compute-host-a     (16 th,  62 GB) → 4 workers (≤40 GB peak, ≥22 GB headroom)
#   macbook  (10 th,  32 GB) → cap not engaged (no /proc/meminfo)
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
# (3) Set CMAKE_BUILD_PARALLEL_LEVEL if unset on Linux.
# ---------------------------------------------------------------------------
if [ -z "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ] && [ -r /proc/meminfo ]; then
    _vqsafe_nproc=$(nproc 2>/dev/null || echo 4)
    _vqsafe_mem_mb=$(awk '/^MemTotal:/ {print int($2/1024); exit}' /proc/meminfo)
    if [ -n "${_vqsafe_mem_mb:-}" ] && [ "$_vqsafe_mem_mb" -gt 0 ]; then
        _vqsafe_cap=$(( _vqsafe_mem_mb / 15000 ))
        [ "$_vqsafe_cap" -lt 2 ] && _vqsafe_cap=2
        [ "$_vqsafe_cap" -gt "$_vqsafe_nproc" ] && _vqsafe_cap=$_vqsafe_nproc
        [ "$_vqsafe_cap" -gt 8 ] && _vqsafe_cap=8
        export CMAKE_BUILD_PARALLEL_LEVEL=$_vqsafe_cap
        if [ -z "${VIBEQC_BUILD_ENV_QUIET:-}" ]; then
            echo "==> CMAKE_BUILD_PARALLEL_LEVEL=$_vqsafe_cap (nproc=$_vqsafe_nproc, mem=${_vqsafe_mem_mb}MB; cap@8 / 15GB-per-worker)"
        fi
    fi
    unset _vqsafe_nproc _vqsafe_mem_mb _vqsafe_cap
fi
