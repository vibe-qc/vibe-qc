#!/usr/bin/env bash
# Build OpenBLAS from source into third_party/openblas/install/.
#
# Opt-in vendored BLAS+LAPACK backend. Not invoked by
# scripts/setup_native_deps.sh by default — pass WITH_OPENBLAS=1 to
# that script, or call this script directly. The CMake build picks
# the vendored install up automatically once it exists (top-level
# CMakeLists.txt prepends third_party/openblas/install to
# CMAKE_PREFIX_PATH, and cpp/CMakeLists.txt sets BLA_VENDOR=OpenBLAS
# when the install dir is present).
#
# Why vendor OpenBLAS at all:
#
#   - vibe-qc's C++ core uses Eigen, which delegates dense matrix
#     products and (when EIGEN_USE_LAPACKE is set) dense solvers to
#     the linked BLAS+LAPACK. Without an optimised BLAS, every dgemm
#     / dsyev / dpotrf runs Eigen's generic-C++ kernels — the speed
#     gap vs ORCA on HF / def2-TZVP / RIJCOSX is dominated by this.
#
#   - On macOS, Apple Accelerate (ships with the OS) is the right
#     answer; this script is a no-op there. The vendored OpenBLAS
#     path is mainly a Linux story.
#
#   - On Linux, distros split BLAS provision: Arch / Manjaro ship
#     reference netlib BLAS by default (slow), with `openblas` /
#     `blas-openblas` as upgrade packages; Debian / Ubuntu need
#     `libopenblas-dev` + `liblapacke-dev` for full coverage;
#     Fedora has `openblas-devel`. Users without sudo (HPC, locked-
#     down workstations) can't always install these. The vendored
#     build closes that gap.
#
#   - Reproducibility: a known-good OpenBLAS pin built into
#     third_party/openblas/install/ means CI / dev / prod boxes get
#     byte-identical BLAS behaviour, the same way we vendor libint /
#     libxc / spglib for the same reason.
#
# Build options chosen:
#
#   - DYNAMIC_ARCH=1 — single binary that picks the right CPU kernel
#     at runtime (Haswell, Skylake, Zen, etc.). Crucial when the
#     build box has different SIMD support than the user's box.
#
#   - USE_LAPACK=1 USE_LAPACKE=1 — bundle netlib LAPACK 3.12.x's
#     Fortran sources and the C-interface (lapacke.h) into
#     libopenblas.so. One linked library carries BLAS + LAPACK +
#     LAPACKE, so EIGEN_USE_BLAS *and* EIGEN_USE_LAPACKE both
#     activate.
#
#   - USE_THREAD=1 USE_OPENMP=0 NUM_THREADS=128 — pthreads-internal
#     threading capped at 128. We then set OPENBLAS_NUM_THREADS=1
#     from python/vibeqc/__init__.py (respecting user overrides) so
#     BLAS-internal threads don't contend with vibe-qc's OpenMP
#     layer. USE_OPENMP=0 avoids the nested-OpenMP gotcha; if the
#     user explicitly wants BLAS to thread, they export
#     OPENBLAS_NUM_THREADS=N themselves.
#
#   - NO_AFFINITY=1 — disables CPU pinning so OpenBLAS doesn't
#     fight taskset / OpenMP placement.
#
# Hard build-time dependency: gfortran (or another Fortran compiler
# OpenBLAS recognises). LAPACK's reference Fortran sources can't be
# skipped if we want LAPACKE — that's the whole point of
# USE_LAPACK=1. NO_FORTRAN=1 would give a BLAS-only build but lose
# the dense-solver delegation, which is most of the win for SCF.
#
# Re-running is safe: idempotent if the install is already there.
# Set REBUILD=1 to force a fresh build.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build-pressure safety (see _safe_build_env.sh comment block). OpenBLAS
# uses Make (not CMake), so the helper's CMAKE_BUILD_PARALLEL_LEVEL is used
# as the `-j` value below rather than read by CMake.
# shellcheck source=_safe_build_env.sh
. "$SCRIPT_DIR/_safe_build_env.sh"

# Supply-chain integrity (see _verify_source.sh comment block).
# shellcheck source=_verify_source.sh
. "$SCRIPT_DIR/_verify_source.sh"

# Serialize before the install sentinel or any source/build-tree mutation.
# vibeqc-recipe-hash: lifecycle-only-begin
# shellcheck source=_build_lock.sh
. "$SCRIPT_DIR/_build_lock.sh"
vibeqc_acquire_build_lock
# vibeqc-recipe-hash: lifecycle-only-end

OPENBLAS_VERSION="0.3.33"
# Commit SHA the upstream tag v$OPENBLAS_VERSION pointed at when this
# script was pinned. Re-resolve via
#   git ls-remote --tags https://github.com/OpenMathLib/OpenBLAS.git v$OPENBLAS_VERSION
# any time OPENBLAS_VERSION is bumped.
OPENBLAS_COMMIT_SHA="62bcfb0dc9f1cfa685fc04135c50e2780c303137"
OPENBLAS_DIR="$REPO_ROOT/third_party/openblas"
SRC_DIR="$OPENBLAS_DIR/src"
INSTALL_DIR="$OPENBLAS_DIR/install"

# Sentinel file we look for to short-circuit: the C-interface header
# lapacke.h is only there if USE_LAPACKE=1 succeeded. If a previous
# partial build dropped libopenblas.so but failed before installing
# lapacke.h, we want to redo the build.
SENTINEL="$INSTALL_DIR/include/lapacke.h"

if [ "${REBUILD:-0}" != "1" ] && [ -f "$SENTINEL" ]; then
    echo "OpenBLAS $OPENBLAS_VERSION already installed at $INSTALL_DIR"
    exit 0
fi

# Fortran compiler check — bail with a helpful message if missing.
if ! command -v gfortran >/dev/null 2>&1; then
    cat >&2 <<EOF
build_openblas.sh: gfortran not found in PATH.

OpenBLAS bundles netlib LAPACK's Fortran sources to give us
EIGEN_USE_LAPACKE-grade dense solvers. Building from source needs
a Fortran compiler. Install one and retry:

    Arch / Manjaro:  sudo pacman -S gcc-fortran
    Debian / Ubuntu: sudo apt install gfortran
    Fedora / RHEL:   sudo dnf install gcc-gfortran
    macOS:           brew install gcc      (provides gfortran)

Alternative: skip the vendored build and rely on whatever BLAS is
already on the system — set VIBEQC_USE_BLAS=ON (default) and CMake's
FindBLAS will pick up the system install. On macOS, Apple Accelerate
is automatically picked and is much faster than building OpenBLAS
locally, so this script isn't needed there.
EOF
    exit 1
fi

# macOS: warn that Accelerate is almost always the better answer.
# Allow the build to continue — sometimes you want OpenBLAS even on
# macOS (reproducibility, MKL-incompatible code paths) — but tell
# the user what they're trading off.
if [ "$(uname -s)" = "Darwin" ]; then
    cat <<EOF
build_openblas.sh: macOS detected.

Apple Accelerate ships with the OS, is highly tuned for Apple
Silicon and recent Intel Macs, and is what vibe-qc's CMake picks
by default on macOS via FindBLAS(BLA_VENDOR=Apple). Building
OpenBLAS here is rarely a win on macOS unless you're after a
reproducible cross-platform build.

To force OpenBLAS on macOS once this script finishes, pass
-DVIBEQC_BLAS_VENDOR=OpenBLAS to the vibe-qc CMake configure.

Continuing in 3 s — Ctrl-C to abort.
EOF
    sleep 3
fi

if [ ! -d "$SRC_DIR/.git" ]; then
    echo "Cloning OpenBLAS $OPENBLAS_VERSION..."
    rm -rf "$SRC_DIR"
    git_clone_pinned https://github.com/OpenMathLib/OpenBLAS.git \
        "v$OPENBLAS_VERSION" "$OPENBLAS_COMMIT_SHA" "$SRC_DIR"
fi

case "$(uname -s)" in
    Darwin) PHYS_CORES="$(sysctl -n hw.physicalcpu 2>/dev/null || sysctl -n hw.ncpu)" ;;
    Linux)  PHYS_CORES="$(nproc 2>/dev/null || echo 4)" ;;
    *)      PHYS_CORES=4 ;;
esac

# Honor _safe_build_env.sh's cap on Linux; macOS falls back to PHYS_CORES.
BUILD_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$PHYS_CORES}"

echo "Building OpenBLAS $OPENBLAS_VERSION with $BUILD_JOBS parallel jobs..."

# OpenBLAS uses Make natively, not CMake. The CMake build path
# exists but is less battle-tested. We use the Make path, which is
# what every distro packager and the upstream maintainers
# recommend.
#
# Build flags rationale documented in the header comment above.
make -C "$SRC_DIR" -j "$BUILD_JOBS" \
    DYNAMIC_ARCH=1 \
    USE_LAPACK=1 USE_LAPACKE=1 \
    USE_THREAD=1 USE_OPENMP=0 \
    NUM_THREADS=128 \
    NO_AFFINITY=1 \
    NO_STATIC=0 NO_SHARED=0 \
    NO_WARMUP=1

echo
echo "Installing OpenBLAS $OPENBLAS_VERSION to $INSTALL_DIR..."
# OpenBLAS installs to PREFIX/lib + PREFIX/include. lapacke.h ends
# up under include/ — that's the sentinel file we check above.
make -C "$SRC_DIR" install PREFIX="$INSTALL_DIR" \
    DYNAMIC_ARCH=1 \
    USE_LAPACK=1 USE_LAPACKE=1 \
    USE_THREAD=1 USE_OPENMP=0 \
    NUM_THREADS=128

if [ ! -f "$SENTINEL" ]; then
    echo "build_openblas.sh: install completed but $SENTINEL is missing." >&2
    echo "Something went wrong with USE_LAPACKE=1 — check the make output above." >&2
    exit 1
fi

echo
echo "OpenBLAS $OPENBLAS_VERSION installed to $INSTALL_DIR"
echo "  lib:     $INSTALL_DIR/lib/libopenblas.{so,dylib}"
echo "  include: $INSTALL_DIR/include/{cblas.h,lapacke.h,openblas_config.h}"
echo
echo "vibe-qc's CMake will pick this up automatically on the next"
echo "build (find_package(BLAS BLA_VENDOR=OpenBLAS) will resolve to"
echo "this install via CMAKE_PREFIX_PATH)."
