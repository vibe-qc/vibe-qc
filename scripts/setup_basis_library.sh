#!/usr/bin/env bash
# Materialise the basis library for libint from its QVF-Basis records.
#
# python/vibeqc/basis_library/qvf/*.qvf.json are the records (the source of
# truth, #406). This renders them into <target>/basis/ as the .g94 orbital
# files libint reads and the .ecp sidecars libecpint reads, then writes the
# content receipt the runtime checks before it will use the overlay (#409).
#
# By default the target is build/basis_library/, a gitignored build-output
# directory the runtime prefers when present. The committed rendering under
# python/vibeqc/basis_library/basis/ ships inside the wheel and is pinned
# byte for byte to the records by tests/test_basis_qvf_parity.py; refresh it
# after changing a record with:
#
#     ./scripts/setup_basis_library.sh --target-dir python/vibeqc/basis_library
#
# Needs Python 3.11 or newer: an explicit PYTHON_BIN, else the project .venv,
# the active VIRTUAL_ENV, then python3 (#516). Nothing here imports vibeqc or
# the native core, so it runs before the core is built. Idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIBRARY="$REPO_ROOT/python/vibeqc/basis_library"

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
            echo "  Default target: build/basis_library (gitignored runtime overlay)" >&2
            echo "  For the committed rendering: --target-dir python/vibeqc/basis_library" >&2
            echo "  Python >=3.11: PYTHON_BIN override, project .venv, active VIRTUAL_ENV, then python3" >&2
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1'" >&2
            exit 1
            ;;
    esac
done

# Resolve an interpreter matching pyproject.toml before changing any output.
# An explicit nonempty PYTHON_BIN is authoritative, including on failure.
# Otherwise prefer usable project/active environments over the system PATH.
if [ -z "${PYTHON_BIN:-}" ]; then
    PYTHON_BIN="python3"
    for candidate in "$REPO_ROOT/.venv/bin/python" "${VIRTUAL_ENV:-}/bin/python"; do
        [ "$candidate" != "/bin/python" ] || continue
        if [ -x "$candidate" ] && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
if ! PYTHON_VERSION=$("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0]); sys.exit(sys.version_info < (3, 11))' 2>&1); then
    echo "Error: PYTHON_BIN='$PYTHON_BIN' must run Python >=3.11 before basis setup can write output." >&2
    echo "  Interpreter check: $PYTHON_VERSION" >&2
    echo "Set PYTHON_BIN to a supported Python interpreter and run setup again." >&2
    exit 1
fi

echo "Rendering $(ls -1 "$LIBRARY/qvf"/*.qvf.json | wc -l | tr -d ' ') QVF-Basis records into $TARGET_DIR/basis"
mkdir -p "$TARGET_DIR/basis"
"$PYTHON_BIN" "$SCRIPT_DIR/render_basis_library.py" --target "$TARGET_DIR/basis"

if [ "$TARGET_DIR" != "$LIBRARY" ]; then
    # The committed rendering must agree with the records, or the receipt
    # below would make every import refuse the overlay it just built.
    if ! "$PYTHON_BIN" "$SCRIPT_DIR/render_basis_library.py" --check; then
        echo "Error: the committed rendering under python/vibeqc/basis_library/basis/ is out of" >&2
        echo "step with the records. Refresh it first:" >&2
        echo "  $0 --target-dir python/vibeqc/basis_library" >&2
        exit 1
    fi
    # Prepare content receipts while setup owns the catalog. Imports are read-only.
    "$PYTHON_BIN" "$REPO_ROOT/python/vibeqc/_basis_overlay.py" "$TARGET_DIR" "$LIBRARY"
fi

n_g94=$(ls -1 "$TARGET_DIR/basis"/*.g94 2>/dev/null | wc -l | tr -d ' ')
n_ecp=$(ls -1 "$TARGET_DIR/basis"/*.ecp 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "Done: $n_g94 .g94 orbital files and $n_ecp .ecp sidecars under $TARGET_DIR/basis"
if [ "$TARGET_DIR" = "${REPO_ROOT}/build/basis_library" ]; then
    echo "Written to build/basis_library/ (runtime overlay; not committed)."
else
    echo "Written to $TARGET_DIR. The committed rendering ships inside the wheel."
fi
