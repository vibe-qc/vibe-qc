#!/usr/bin/env bash
#
# Local-side driver: run every .py under one S22 system directory
# through the vibe-qc venv. Mirrors run_orca_system.sh (which runs on
# compute-host-d via vq) so local + remote workflows have the same shape.
#
# Idempotent: skip an input whose .result file already exists. Stop on
# first failure (`set -e`) so partial-progress is visible from the
# shell exit code.
#
# Usage:
#   bash run_vibeqc_system.sh <system_dir>
#       e.g. bash run_vibeqc_system.sh inputs/vibeqc/s22-02-water-dimer

set -euo pipefail

# Override with VIBEQC_PY=/path/to/.venv/bin/python on the command line
# or in your shell rc. The default below resolves to the .venv inside
# the vibe-qc checkout that contains this script — works for the standard
# editable install (`pip install -e .[test]`) without further config.
VENV_PY="${VIBEQC_PY:-$(cd "$(dirname "$0")/../../../../" && pwd)/.venv/bin/python}"

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <system_dir>" >&2
    exit 2
fi
SYSTEM_DIR="$1"
if [[ ! -d "$SYSTEM_DIR" ]]; then
    echo "$0: not a directory: $SYSTEM_DIR" >&2
    exit 2
fi

cd "$SYSTEM_DIR"
shopt -s nullglob
inputs=(*.py)
if [[ ${#inputs[@]} -eq 0 ]]; then
    echo "$0: no .py inputs in $(pwd)" >&2
    exit 2
fi

echo "vibe-qc inputs: ${#inputs[@]} in $(pwd)"
failures=0
for inp in "${inputs[@]}"; do
    result="${inp%.py}.result"
    if [[ -s "$result" ]]; then
        echo "[skip] $inp (result already present)"
        continue
    fi
    echo "[run]  $inp"
    t0=$SECONDS
    log="${inp%.py}.log"
    if "$VENV_PY" "$inp" > "$log" 2>&1; then
        echo "       ok ($((SECONDS - t0))s)"
    else
        echo "       FAILED ($((SECONDS - t0))s) — see $log" >&2
        failures=$((failures + 1))
    fi
done

if [[ $failures -gt 0 ]]; then
    echo "Done with $failures FAILURE(s)" >&2
    exit 1
fi
echo "Done — all ${#inputs[@]} inputs completed cleanly."
