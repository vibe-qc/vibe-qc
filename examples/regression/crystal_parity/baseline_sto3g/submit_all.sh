#!/usr/bin/env bash
# Submit all baseline-STO3G CRYSTAL14 inputs via vq from the laptop.
# The daemon's --max-jobs 1 cap runs them serially (each ~10-60s).
#
# Usage (from repo root, after committing + pushing the inputs):
#   bash examples/regression/crystal_parity/baseline_sto3g/submit_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARITY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

inputs=(
    "baseline_sto3g/lih-rhf-sto3g.d12"
    "baseline_sto3g/mgo-rhf-sto3g.d12"
    "baseline_sto3g/nacl-rhf-sto3g.d12"
    "baseline_sto3g/lif-rhf-sto3g.d12"
    "baseline_sto3g/c-diamond-rhf-sto3g.d12"
    "baseline_sto3g/si-diamond-rhf-sto3g.d12"
)

job_ids=()
for input in "${inputs[@]}"; do
    echo "submitting $input ..."
    jobid=$(cd "$PARITY_DIR/.." && vq submit \
        -d "$PARITY_DIR" \
        --cpus 14 --wall-time-seconds 1800 -- \
        bash run-reference.sh "$input" 2>&1 | tail -1)
    echo "  jobid: $jobid"
    job_ids+=("$jobid")
done

echo
echo "All jobs submitted. Job IDs:"
printf '  %s\n' "${job_ids[@]}"
echo
echo "Poll with:  vq queue | head -10"
echo "Fetch a result:  vq status <jobid> -n 0 | tail -40"
