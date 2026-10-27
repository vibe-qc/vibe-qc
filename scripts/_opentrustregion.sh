#!/bin/bash
# Optional native optimizer selection shared by the lifecycle scripts.
# Source only: no fetching, compiler execution, or filesystem mutation here.

VIBEQC_OTR_CHOICE="auto"

vibeqc_otr_select() {
    if [ "$VIBEQC_OTR_CHOICE" != auto ] && [ "$VIBEQC_OTR_CHOICE" != "$1" ]; then
        echo "Error: choose only one of --with-opentrustregion / --without-opentrustregion." >&2
        return 1
    fi
    VIBEQC_OTR_CHOICE="$1"
}

vibeqc_otr_configure() {
    local venv="$1"
    local python="${2:-python3}"
    # Parse CMAKE_ARGS as arguments, never as shell code. Explicit lifecycle
    # flags win, followed by CMAKE_ARGS, the last verified venv selection,
    # and legacy CMake caches. Reading these files also works for --clean or
    # a broken venv that is about to be replaced; importing it would not.
    VIBEQC_OTR_ENABLED=$("$python" -I - "$VIBEQC_OTR_CHOICE" "$venv" "$REPO_ROOT" <<'PY'
import os
from pathlib import Path
import shlex
import sys

choice, venv, root = sys.argv[1:]
if choice != "auto":
    print(choice)
    raise SystemExit

def boolean(value):
    value = value.strip().upper()
    if value in {"ON", "TRUE", "YES", "Y", "1"}:
        return "ON"
    if value in {"OFF", "FALSE", "NO", "N", "0", ""}:
        return "OFF"
    raise SystemExit("Invalid OpenTrustRegion build selection; use --with-opentrustregion or --without-opentrustregion.")

args = shlex.split(os.environ.get("CMAKE_ARGS", ""))
selected = None
for i, arg in enumerate(args):
    value = args[i + 1] if arg == "-D" and i + 1 < len(args) else arg.removeprefix("-D")
    key, sep, setting = value.partition("=")
    if sep and key.split(":")[0] == "VIBEQC_ENABLE_OPENTRUSTREGION":
        selected = boolean(setting)
if selected is None and venv:
    marker = Path(venv) / ".vibeqc-opentrustregion"
    if marker.is_file():
        selected = boolean(marker.read_text())
if selected is None:
    cached = set()
    for cache in (Path(root) / "build").glob("*/CMakeCache.txt"):
        for line in cache.read_text().splitlines():
            if line.startswith("VIBEQC_ENABLE_OPENTRUSTREGION:BOOL="):
                cached.add(boolean(line.partition("=")[2]))
    if len(cached) > 1:
        raise SystemExit("Conflicting OpenTrustRegion build caches; use --with-opentrustregion or --without-opentrustregion.")
    selected = next(iter(cached), "OFF")
print(selected)
PY
    ) || return 1
    export VIBEQC_OTR_ENABLED
    export CMAKE_ARGS="${CMAKE_ARGS:+$CMAKE_ARGS }-DVIBEQC_ENABLE_OPENTRUSTREGION=$VIBEQC_OTR_ENABLED"
    WITH_OPENTRUSTREGION=0
    [ "$VIBEQC_OTR_ENABLED" != ON ] || WITH_OPENTRUSTREGION=1
    export WITH_OPENTRUSTREGION
    echo "    OpenTrustRegion: $VIBEQC_OTR_ENABLED (pinned source fetched and linked when enabled)"
}

vibeqc_otr_preflight() {
    [ "${WITH_OPENTRUSTREGION:-0}" = 1 ] || return 0
    local python="${1:-python3}"
    # Honor CMake's explicit compiler first, then FC, then common compilers.
    # This checks availability; CMake validates compilation and linkage later.
    "$python" -I - <<'PY'
import os
import shlex
import shutil
import sys

compiler = None
args = shlex.split(os.environ.get("CMAKE_ARGS", ""))
for i, arg in enumerate(args):
    value = args[i + 1] if arg == "-D" and i + 1 < len(args) else arg.removeprefix("-D")
    key, sep, setting = value.partition("=")
    if sep and key.split(":")[0] == "CMAKE_Fortran_COMPILER":
        compiler = setting
if compiler is None and os.environ.get("FC"):
    compiler = shlex.split(os.environ["FC"])[0]
available = shutil.which(compiler) if compiler else any(
    shutil.which(name) for name in ("gfortran", "flang-new", "flang", "ifx", "ifort", "nvfortran")
)
if not available or not shutil.which("git"):
    sys.exit("OpenTrustRegion requires Git and a Fortran compiler, plus LP64 BLAS/LAPACK.\n"
             "Install prerequisites before retrying:\n"
             "  macOS: brew install git gcc (BLAS/LAPACK: system Accelerate)\n"
             "  Debian/Ubuntu: sudo apt install git gfortran libopenblas-dev liblapack-dev\n"
             "  Fedora/RHEL: sudo dnf install git gcc-gfortran openblas-devel lapack-devel\n"
             "  Arch: sudo pacman -S git gcc-fortran openblas\n"
             "For a custom toolchain, set FC or CMAKE_Fortran_COMPILER in CMAKE_ARGS.")
PY
}

vibeqc_otr_verify() {
    local python="$1"
    local venv="$2"
    "$python" -I - "$VIBEQC_OTR_ENABLED" "$venv" <<'PY'
from pathlib import Path
import sys
import vibeqc

expected = sys.argv[1] == "ON"
actual = bool(getattr(vibeqc, "has_opentrustregion", lambda: False)())
if actual != expected:
    raise SystemExit("OpenTrustRegion installation verification failed: requested "
                     + sys.argv[1] + ", but loaded native capability disagrees.")
print("OpenTrustRegion capability verified: " + sys.argv[1])
# Save only after the installed extension loads and agrees. The marker moves
# with the venv during the existing replacement/rollback transaction.
marker = Path(sys.argv[2]) / ".vibeqc-opentrustregion"
temporary = marker.with_suffix(".tmp")
temporary.write_text(sys.argv[1] + "\n")
temporary.replace(marker)
PY
}
