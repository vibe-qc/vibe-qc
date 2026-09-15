#!/usr/bin/env bash
# Diagnostic for "is everything wired up to cross-validate?".
# Probes every QC code that vibe-qc's benchmark framework knows
# about (vibe-qc itself, PySCF, ORCA, Psi4, NWChem, Gaussian,
# Q-Chem, GAMESS-US, Turbomole) and prints a colour-coded
# availability table.
#
# Run from the vibe-qc repo root after the venv is set up::
#
#     ./scripts/check_external_codes.sh
#
# Honours $VIRTUAL_ENV when set; falls back to ./.venv. Exits
# 0 when at least vibe-qc + one external code are available
# (i.e. cross-validation is possible), 1 otherwise — useful in
# CI to gate cross-validation jobs.

set -euo pipefail

# --- Resolve which python we should ask --------------------------
#
# Look in three places, in order:
#   1. $VIRTUAL_ENV — honour an active venv, regardless of cwd
#   2. <repo>/.venv — the standard vibe-qc layout
#   3. <main-repo>/.venv — when running from a `git worktree add`
#      checkout, the worktree itself doesn't have .venv but the
#      main repo's does. `git rev-parse --git-common-dir` always
#      points at the main repo's .git directory regardless of
#      which worktree you're in.

PYTHON=""
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
fi
if [[ -z "$PYTHON" ]]; then
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO="$(cd "$HERE/.." && pwd)"
    if [[ -x "$REPO/.venv/bin/python" ]]; then
        PYTHON="$REPO/.venv/bin/python"
    fi
fi
if [[ -z "$PYTHON" ]]; then
    if common_dir=$(git rev-parse --git-common-dir 2>/dev/null); then
        # `--git-common-dir` returns the .git of the main repo;
        # the toplevel is its parent.
        common_dir="$(cd "$common_dir" && pwd)"
        main_repo="$(dirname "$common_dir")"
        if [[ -x "$main_repo/.venv/bin/python" ]]; then
            PYTHON="$main_repo/.venv/bin/python"
        fi
    fi
fi
if [[ -z "$PYTHON" ]]; then
    echo "error: no vibe-qc venv detected." >&2
    echo "       Activate one with 'source .venv/bin/activate', run" >&2
    echo "       from the vibe-qc repo root after creating .venv, or" >&2
    echo "       set VIRTUAL_ENV explicitly." >&2
    exit 1
fi

# --- Run the framework's availability probe ---------------------

echo "vibe-qc external-code availability"
echo "──────────────────────────────────────────────────────────────"
echo "Python: $PYTHON"
echo

"$PYTHON" -c "
from vibeqc.benchmark import detect_calculators

avail = detect_calculators()

# ANSI colours; degrade to plain text when stdout isn't a tty.
import sys
if sys.stdout.isatty():
    GREEN, RED, RESET, DIM = '\\033[32m', '\\033[31m', '\\033[0m', '\\033[2m'
else:
    GREEN = RED = RESET = DIM = ''

# Order: vibe-qc + python-native first, then external binaries.
ordered = sorted(
    avail.items(),
    key=lambda kv: (
        not kv[1].available,
        not kv[1]._is_python_native,
        kv[0],
    ),
)

n_yes = 0
n_external_yes = 0
for name, info in ordered:
    if info.available:
        marker = f'{GREEN}✓ available{RESET}'
        n_yes += 1
        if not info._is_python_native or name not in {'vibeqc'}:
            if name in {'orca', 'psi4', 'nwchem', 'gaussian',
                       'qchem', 'gamess_us', 'turbomole'}:
                n_external_yes += 1
        if info.binary_path is not None:
            details = str(info.binary_path)
        elif info._is_python_native and info.importable:
            details = 'Python-native'
        else:
            details = info.notes
    else:
        marker = f'{RED}✗ missing  {RESET}'
        details = info.notes
    print(f'  {name:<11}{marker}  {DIM}({details}){RESET}')

print()
print(f'Available: {n_yes}/{len(avail)}')
if n_yes >= 2:
    print('Cross-validation: ✓ possible (vibe-qc + at least one peer)')
else:
    print('Cross-validation: ✗ install at least one peer code')
    sys.exit(1)
"
