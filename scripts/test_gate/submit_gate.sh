#!/usr/bin/env bash
# Submit the full-suite gate (fast + slow lanes) to a vq host, detached.
# Reads results back from the job workdir; runs the verdict locally.
#
# Usage:
#   submit_gate.sh [HOST] [GATE_PYTHON] [WORKTREE]
# Defaults target the localhost-now setup (this Mac, 128 GB, manual daemon).
# On the managed fleet (once repaired) pass the host + its dev venv + checkout.
set -euo pipefail

HOST="${1:-localhost}"
PY="${2:-/tmp/vqc_gate_venv/bin/python}"          # gate venv with a fresh-built vibeqc + vibe-view
WT="${3:-/private/tmp/vqc_th_wt}"                  # source under test
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "gate -> host=$HOST py=$PY wt=$WT"

# Fast lane: everything except @slow. Frequent gate.
FAST=$(vq submit "$HOST" --job-name gate-fast -d "$HERE" -- \
    "$PY" run_full_suite.py --wt "$WT" --py "$PY" \
    --markexpr "not slow" --jobs 6 --test-timeout 300 --file-timeout 600 \
    --out "\$VQ_WORKDIR/fast.jsonl" 2>&1 | tail -1)
echo "fast lane job: $FAST"

# Slow/heavy lane: @slow tests, heavy env, low concurrency (some files tens of GB).
SLOW=$(vq submit "$HOST" --job-name gate-slow -d "$HERE" -- \
    "$PY" run_full_suite.py --wt "$WT" --py "$PY" \
    --markexpr "slow" --heavy-env --jobs 2 --test-timeout 1800 --file-timeout 2400 \
    --out "\$VQ_WORKDIR/slow.jsonl" 2>&1 | tail -1)
echo "slow lane job: $SLOW"

cat <<EOF

Watch:   vq logs $HOST $FAST -f
         vq logs $HOST $SLOW -f
Verdict (when both terminal):
  vq fetch $HOST $FAST --workdir -o /tmp/gate && vq fetch $HOST $SLOW --workdir -o /tmp/gate
  cat /tmp/gate/*/fast.jsonl /tmp/gate/*/slow.jsonl > /tmp/gate/all.jsonl
  $PY $HERE/gate_verdict.py /tmp/gate/all.jsonl $HERE/known_reds_baseline.json
EOF
