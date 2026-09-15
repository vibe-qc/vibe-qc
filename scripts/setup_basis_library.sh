#!/usr/bin/env bash
# Assemble the basis-library from libint's standard bases + our custom
# ones. By default writes into build/basis_library/basis/ — a build-output
# directory that is gitignored (build/) so the working tree stays clean
# across pip install -e . and fleet deploys.
#
# The committed python/vibeqc/basis_library/basis/ ships unchanged inside
# the Python wheel; the build-output copy is an overlay that the runtime
# (vibeqc._install_basis_library) prefers when present.
#
# To regenerate the committed basis/ directory (e.g. after adding a new
# file to custom/ or upgrading libint), pass the explicit target:
#
#     ./scripts/setup_basis_library.sh python/vibeqc/basis_library
#
# Idempotent: safe to re-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --target-dir lets the caller override where the basis/ directory
# is assembled. Default: build/basis_library (gitignored, won't
# dirty the tree). Pass python/vibeqc/basis_library to regenerate
# the committed copy.
TARGET_DIR="${REPO_ROOT}/build/basis_library"
while [ $# -gt 0 ]; do
    case "$1" in
        --target-dir)
            [ $# -ge 2 ] || { echo "Error: --target-dir requires an argument" >&2; exit 1; }
            TARGET_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--target-dir PATH]" >&2
            echo "  Default target: build/basis_library (gitignored build output)" >&2
            echo "  For committing: --target-dir python/vibeqc/basis_library" >&2
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1'" >&2
            exit 1
            ;;
    esac
done

CUSTOM_DIR="$REPO_ROOT/python/vibeqc/basis_library/custom"
BASIS_DIR="$TARGET_DIR/basis"
LIBINT_BASIS="$REPO_ROOT/third_party/libint/install/share/libint"

# Find libint's actual version directory (e.g. 2.13.1/basis).
if [ ! -d "$LIBINT_BASIS" ]; then
    echo "Error: libint not found at $LIBINT_BASIS." >&2
    echo "Run ./scripts/build_libint.sh first." >&2
    exit 1
fi
LIBINT_BASIS_DIR=$(find "$LIBINT_BASIS" -maxdepth 2 -name basis -type d | head -1)
if [ -z "$LIBINT_BASIS_DIR" ]; then
    echo "Error: couldn't find libint's basis directory under $LIBINT_BASIS" >&2
    exit 1
fi

echo "Populating $BASIS_DIR"
echo "  standard bases from: $LIBINT_BASIS_DIR"
echo "  custom bases from:   $CUSTOM_DIR"

mkdir -p "$CUSTOM_DIR"
rm -rf "$BASIS_DIR"
mkdir -p "$BASIS_DIR"

# Standard first.
cp "$LIBINT_BASIS_DIR"/*.g94 "$BASIS_DIR"/
n_standard=$(ls -1 "$BASIS_DIR" | wc -l | tr -d ' ')

# Custom last, so a custom file with the same name wins. Copy any
# pre-split ECP sidecars as well: custom orbital files such as vdzp.g94
# are already libint-clean, so the splitter cannot recreate their .ecp
# siblings from the .g94 alone.
n_custom_g94=0
n_custom_ecp=0
if compgen -G "$CUSTOM_DIR"/*.g94 > /dev/null; then
    cp "$CUSTOM_DIR"/*.g94 "$BASIS_DIR"/
    n_custom_g94=$(ls -1 "$CUSTOM_DIR"/*.g94 | wc -l | tr -d ' ')
fi
if compgen -G "$CUSTOM_DIR"/*.ecp > /dev/null; then
    cp "$CUSTOM_DIR"/*.ecp "$BASIS_DIR"/
    n_custom_ecp=$(ls -1 "$CUSTOM_DIR"/*.ecp | wc -l | tr -d ' ')
fi

# ECP-aware split: BSE-format files for vDZP, LANL2DZ, dhf-*, etc.
# bundle orbital basis blocks AND ECP definition blocks into one
# .g94 file. libint2 cannot parse the ECP blocks ("invalid angular
# momentum label"), so we split each into a libint-clean
# <name>.g94 (orbital only) plus a <name>.ecp sidecar that vibe-qc
# reads via libecpint when needed. See
# scripts/basisset_dev/split_ecp_g94.py for details.
echo ""
echo "Splitting ECP blocks out of any ECP-bearing .g94 files…"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" "$SCRIPT_DIR/basisset_dev/split_ecp_g94.py" "$BASIS_DIR"

n_total=$(ls -1 "$BASIS_DIR" | wc -l | tr -d ' ')
n_g94=$(ls -1 "$BASIS_DIR"/*.g94 2>/dev/null | wc -l | tr -d ' ')
n_ecp=$(ls -1 "$BASIS_DIR"/*.ecp 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "Done: $n_standard standard .g94 + $n_custom_g94 custom .g94 + $n_custom_ecp custom .ecp source files"
echo "      produced $n_total total files ($n_g94 .g94 orbital files, $n_ecp .ecp sidecars)"
echo ""
if [ "$TARGET_DIR" = "${REPO_ROOT}/build/basis_library" ]; then
    echo "Written to build/basis_library/ (runtime overlay; not committed)."
    echo "To regenerate the committed basis/ directory instead, run:"
    echo "  ./scripts/setup_basis_library.sh --target-dir python/vibeqc/basis_library"
else
    echo "Written to $TARGET_DIR."
    echo "These files are committed to the repo and ship inside the wheel."
fi
