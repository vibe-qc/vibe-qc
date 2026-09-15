#!/usr/bin/env bash
# Rebuild vibe-basis with rollback, without changing Git.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    cat <<'EOF'
USAGE
    ./vibe-basis/scripts/reinstall.sh [OPTIONS]

Rebuild the marker-owned standalone environment from the current checkout.
The existing profile, vq choice, and base interpreter are preserved unless
overridden with update.sh options. Git is never changed.

OPTIONS
    --extras PROFILE      Override the installed capability profile.
    --with-vq             Install/update the co-located vq CLI.
    --without-vq          Keep vq separate.
    --python BIN          Override the existing venv's base interpreter.
    --venv PATH           Venv path (default: vibe-basis/.venv).
    --recreate-venv       Accepted for lifecycle consistency; always rebuilt.
    --dry-run             Preview the rebuild without changing files.
    -h, --help            Show this help.

--skip-git is implied. Branch selectors are intentionally not accepted.
EOF
    exit 0
fi

for arg in "$@"; do
    case "$arg" in
        --branch|--branch=*|--dev|--release|--skip-git)
            echo "Error: reinstall uses the current checkout; '$arg' is not accepted." >&2
            echo "Use ./vibe-basis/scripts/update.sh for Git branch changes." >&2
            exit 1
            ;;
    esac
done

exec "$SCRIPT_DIR/update.sh" --skip-git "$@"
