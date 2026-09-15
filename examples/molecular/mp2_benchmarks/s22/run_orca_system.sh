#!/usr/bin/env bash
#
# vq-side driver: run every .inp under one S22 system directory through
# ORCA via vibe-queue/contrib/run-orca.sh. Designed to be invoked by
# `vq submit -d ...` so the daemon dispatches one job per S22 system,
# and that job sequentially runs all 15 ORCA invocations (5 variants
# × 3 bodies) for the system.
#
# Why one vq job per system and not one per input:
#   * compute-host-d is at --max-jobs 1 (test phase) — flooding the queue
#     with 330 tiny jobs would serialize them anyway and the queue
#     metadata becomes unreadable. One coherent job per system keeps
#     the queue list usable and lets `vq tail` show progress.
#   * ORCA per-input runs are short (water dimer: ~10s), so the
#     overhead of repeated launches is negligible compared to the
#     full-system MP2 cost.
#
# Usage (inside the workspace the vq daemon hands us):
#   bash run_orca_system.sh
#
# Expects the workspace to be the system directory (e.g.
# inputs/orca/s22-02-water-dimer/) containing N×.inp files. Output
# .out files land alongside the inputs. Scratch is auto-cleaned by
# run-orca.sh.

set -euo pipefail

# Override these two via environment to match your install layout:
#   VIBEQC_QUEUE_CHECKOUT  — vibe-queue checkout root
#                            (wrapper: $VIBEQC_QUEUE_CHECKOUT/contrib/run-orca.sh)
#   ORCA_BIN               — absolute path to the orca binary
# The defaults below assume a separate `~/gitlab/vibe-queue` checkout +
# an ORCA install under `~/bin/orca_*/orca` (one match expected).
: "${VIBEQC_QUEUE_CHECKOUT:=$HOME/gitlab/vibe-queue}"
WRAPPER="$VIBEQC_QUEUE_CHECKOUT/contrib/run-orca.sh"
if [[ ! -f "$WRAPPER" ]]; then
    echo "FATAL: $WRAPPER not found; set VIBEQC_QUEUE_CHECKOUT to the separate vibe-queue checkout on this execution host" >&2
    exit 2
fi
if [[ -z "${ORCA_BIN:-}" ]]; then
    # Pick the first orca binary under ~/bin/orca_*/orca (compute-host-d
    # layout) or ~/bin/orca/orca (mac layout, no version suffix on
    # the install dir). Override ORCA_BIN explicitly if you have
    # multiple ORCA versions or a non-standard install path.
    for cand in "$HOME"/bin/orca_*/orca "$HOME"/bin/orca/orca; do
        if [[ -x "$cand" ]]; then
            ORCA_BIN="$cand"
            break
        fi
    done
fi
export ORCA_BIN

if [[ ! -x "$ORCA_BIN" ]]; then
    echo "FATAL: ORCA binary not found at $ORCA_BIN" >&2
    exit 127
fi

# Make the bundled OpenMPI's mpirun reachable for any %pal block.
export PATH="$(dirname "$ORCA_BIN"):$PATH"

# macOS hardened-runtime workaround: ORCA 6.x ad-hoc-signed dylibs are
# rejected by `library load disallowed by system policy` unless found
# via DYLD_LIBRARY_PATH (which bypasses the rpath check that triggers
# the policy denial). Harmless on Linux — DYLD_LIBRARY_PATH is macOS-
# specific. Linux platforms ignore the variable.
if [[ "$(uname)" == "Darwin" && -d "$(dirname "$ORCA_BIN")/lib" ]]; then
    export DYLD_LIBRARY_PATH="$(dirname "$ORCA_BIN")/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

cd "$(dirname "$0")"

shopt -s nullglob
inputs=(*.inp)
if [[ ${#inputs[@]} -eq 0 ]]; then
    echo "FATAL: no .inp files in $(pwd)" >&2
    exit 2
fi

echo "system dir: $(pwd)"
echo "inputs:     ${#inputs[@]}"
echo

failures=0
for inp in "${inputs[@]}"; do
    out="${inp%.inp}.out"
    if [[ -s "$out" ]]; then
        # Idempotent — skip already-completed runs (lets us re-run after partial failure)
        echo "[skip] $inp (output already present: $out)"
        continue
    fi
    echo "[run]  $inp -> $out"
    t0=$SECONDS
    if bash "$WRAPPER" "$inp" "$out"; then
        echo "       ok ($((SECONDS - t0))s)"
    else
        echo "       FAILED ($((SECONDS - t0))s)" >&2
        failures=$((failures + 1))
    fi
done

if [[ $failures -gt 0 ]]; then
    echo "Done with $failures FAILURE(s)" >&2
    exit 1
fi
echo "Done — all ${#inputs[@]} inputs completed cleanly."
