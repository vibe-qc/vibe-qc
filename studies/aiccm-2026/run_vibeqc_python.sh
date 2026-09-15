#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "usage: run_vibeqc_python.sh SCRIPT [ARG ...]" >&2
    exit 2
fi

shopt -s nullglob
candidates=(
    "${VIBEQC_PYTHON:-}"
    "$HOME/vibeqc-dev/.venv311/bin/python"
    "$HOME/gitlab/vibeqc-dev/.venv311/bin/python"
    "$HOME/vibeqc-dev/.venv/bin/python"
    "$HOME/gitlab/vibeqc-dev/.venv/bin/python"
    /mnt/*/gitlab/vibeqc-dev/.venv311/bin/python
    /mnt/*/gitlab/vibeqc-dev/.venv/bin/python
)
for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        exec "$candidate" "$@"
    fi
done

echo "could not locate the vibeqc-dev Python interpreter" >&2
exit 127
