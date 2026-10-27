#!/usr/bin/env bash
# Resume a retained legacy build without editing its immutable product source.
# Usage: recover_libint_build.sh ROOT EXACT_SHA [--apply]
set -euo pipefail
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo 'Usage: recover_libint_build.sh ROOT EXACT_SHA [--apply]' >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$1" && pwd)"
. "$ROOT/scripts/_safe_build_env.sh"
EXPECTED="$2"
shift 2
# The lock belongs to the target tree, not the helper's checkout.
. "$ROOT/scripts/_build_lock.sh"
vibeqc_acquire_build_lock
exec python3 "$SCRIPT_DIR/_native_build_resume.py" recover "$ROOT" --expected-source-sha "$EXPECTED" "$@"
