#!/bin/bash
# Build the vibe-qc docs locally for offline browsing.
#
# Usage:
#   ./scripts/build_docs.sh            — build into docs/_build/html
#   ./scripts/build_docs.sh --clean    — clean before building
#   ./scripts/build_docs.sh -h         — this help message
#
# Prerequisites (already covered by `pip install -e '.[test]'`):
#   - Sphinx + furo + myst_parser + extensions in docs/conf.py
#   - vibe-qc installed (for autodoc to resolve stubs)
#
# The --clean flag removes docs/_build/ so you start fresh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="$SCRIPT_DIR/docs/_build"
VENV_DIR="$SCRIPT_DIR/.venv"

# Required pip packages for the docs build. Keep in sync with the
# ``docs`` extra in pyproject.toml and the extensions/myst_enable_extensions
# lists in docs/conf.py:
#   - sphinx-reredirects  -> conf.py ``extensions`` ("sphinx_reredirects")
#   - linkify-it-py       -> conf.py ``myst_enable_extensions`` ("linkify")
DOC_DEPS=(sphinx furo myst-parser sphinx-copybutton sphinx-design sphinx-reredirects linkify-it-py)

# --- Helper: ask-then-install ---

ask_install() {
    printf '\nRequired docs dependencies are missing from %s:\n' "$VENV_DIR/bin"
    printf '  %s\n' "${DOC_DEPS[@]}"
    printf '\nInstall them now? [y/N] '
    read -r reply
    case "$reply" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *) printf 'Aborted.\n'; exit 1 ;;
    esac
}

if [[ ! -d "$VENV_DIR" ]]; then
    printf 'Error: virtual environment not found at %s\n' "$VENV_DIR"
    printf 'Run: python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[test]"\n'
    exit 1
fi

# Check each dep individually so we know exactly what's missing.
MISSING=()
for dep in "${DOC_DEPS[@]}"; do
    # pip show returns 1 if the package is not installed.
    if ! "$VENV_DIR/bin/pip" show "$dep" >/dev/null 2>&1; then
        MISSING+=("$dep")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    ask_install
    "$VENV_DIR/bin/pip" install "${MISSING[@]}"
fi

SPHINX_BUILD="$VENV_DIR/bin/sphinx-build"
CLEAN=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean|-c)  CLEAN=1; shift ;;
        --verbose|-v) VERBOSE=1; shift ;;
        -h|--help)
            sed -n '4,14p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ $CLEAN -eq 1 ]]; then
    rm -rf "$BUILDDIR"
fi

if [[ $VERBOSE -eq 1 ]]; then
    OPTS="-v"
else
    OPTS=""
fi

cd "$SCRIPT_DIR/docs"
$SPHINX_BUILD -b html $OPTS \
    -D autodoc_mock_imports=vibeqc._vibeqc_core \
    . "$BUILDDIR/html"

INDEX="$BUILDDIR/html/index.html"
printf '\ndocs built successfully.\n'
printf 'Open with:\n'
printf '  open "%s"\n' "$INDEX"
printf '  xdg-open "%s"\n' "$INDEX"
printf '  firefox "%s"\n' "$INDEX"
