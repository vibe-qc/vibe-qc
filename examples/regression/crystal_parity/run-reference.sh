#!/usr/bin/env bash
# vq-submission wrapper for CRYSTAL14 inputs in this directory.
# Ensures ~/bin (where CRYSTAL14 binaries live on compute-host-d) is on
# PATH for the run-crystal.sh helper, then dispatches.
#
# Usage (via vq):
#   vq submit -d ./examples/regression/crystal_parity/ \
#       --cpus 14 --wall-time-seconds 7200 -- \
#       bash run-reference.sh mgo-rhf-pobtzvp-no-stabilisers.d12
set -e
export PATH="$HOME/bin:$PATH"
# VIBEQC_QUEUE_CHECKOUT names the separate queue checkout on the execution host.
queue_checkout="${VIBEQC_QUEUE_CHECKOUT:-$HOME/gitlab/vibe-queue}"
wrapper="$queue_checkout/contrib/run-crystal.sh"
if [[ ! -f "$wrapper" ]]; then
    echo "FATAL: $wrapper not found; set VIBEQC_QUEUE_CHECKOUT to the separate vibe-queue checkout" >&2
    exit 2
fi
exec bash "$wrapper" "$@"
