#!/usr/bin/env bash
# Build FFTW3 from source into third_party/fftw/install/.
#
# FFTW3 ships a CMake build since 3.3.10. We pin that version because it's
# long-stable and matches what most distros actually carry. We build with
# OpenMP and threads enabled to feed the Ewald-decomposed long-range
# Hartree path in periodic SCF.
#
# Re-running is safe: idempotent if the install is already there.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build-pressure safety (see _safe_build_env.sh comment block).
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

FFTW_VERSION="3.3.10"
# SHA-256 of fftw-${FFTW_VERSION}.tar.gz as published at
# https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz. Re-resolve via
#   curl -fsSL https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz | shasum -a 256
# any time FFTW_VERSION is bumped.
FFTW_SHA256="56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467"
FFTW_DIR="$REPO_ROOT/third_party/fftw"
SRC_DIR="$FFTW_DIR/src"
BUILD_DIR="$FFTW_DIR/build"
INSTALL_DIR="$FFTW_DIR/install"

for candidate in \
        "$INSTALL_DIR/lib/cmake/fftw3/FFTW3Config.cmake" \
        "$INSTALL_DIR/lib64/cmake/fftw3/FFTW3Config.cmake"; do
    if [ -f "$candidate" ]; then
        echo "FFTW $FFTW_VERSION already installed at $INSTALL_DIR"
        exit 0
    fi
done

if [ ! -f "$SRC_DIR/CMakeLists.txt" ]; then
    echo "Downloading FFTW $FFTW_VERSION..."
    mkdir -p "$FFTW_DIR" "$SRC_DIR"
    TARBALL="$FFTW_DIR/fftw-${FFTW_VERSION}.tar.gz"
    curl -fsSL "https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz" -o "$TARBALL"
    verify_tarball_sha256 "$TARBALL" "$FFTW_SHA256"
    tar -xzf "$TARBALL" -C "$SRC_DIR" --strip-components=1
    rm -f "$TARBALL"
fi

# Fetch-only mode for offline-HPC pre-staging (docs/cluster_setup.md):
# stage source on an internet-connected login node, compile on an offline
# compute node. Guarded on VIBEQC_FETCH_ONLY, so default builds are
# unchanged.
if [ "${VIBEQC_FETCH_ONLY:-0}" = "1" ]; then
    echo "VIBEQC_FETCH_ONLY=1 -- FFTW source staged at $SRC_DIR; skipping build."
    exit 0
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

case "$(uname -s)" in
    Darwin) PHYS_CORES="$(sysctl -n hw.physicalcpu 2>/dev/null || sysctl -n hw.ncpu)" ;;
    Linux)  PHYS_CORES="$(nproc 2>/dev/null || echo 4)" ;;
    *)      PHYS_CORES=4 ;;
esac

echo "Configuring FFTW $FFTW_VERSION (double precision, OpenMP + threads)..."
# CMAKE_POLICY_VERSION_MINIMUM=3.5: FFTW 3.3.10's CMakeLists declares
# cmake_minimum_required(VERSION 2.x) — CMake 4.x dropped that compat.
# This flag tells CMake to behave as if the minimum is 3.5. -Wno-dev
# quiets upstream's CMake-hygiene warnings (also pre-3.5 era code).
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -GNinja -Wno-dev \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DBUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DENABLE_OPENMP=ON \
    -DENABLE_THREADS=ON

# Honor _safe_build_env.sh's cap if it set one; fall back to PHYS_CORES.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$PHYS_CORES}"
echo "Building FFTW with $CMAKE_BUILD_PARALLEL_LEVEL parallel jobs..."
cmake --build "$BUILD_DIR" --target install

echo
echo "FFTW $FFTW_VERSION installed to $INSTALL_DIR"

# ---------------------------------------------------------------------------
# Optional: drop the build tree once the install is verified.
#
# third_party/<dep>/build is never read again after install/ exists -- nothing
# in install/ symlinks into it and the rpath baked into _vibeqc_core resolves
# to install/lib{,64}. It is ~96% of what a built third_party/ costs (libint's
# build tree alone is ~7.7 GB against a 286 MB install).
#
# Gated on a real artifact check, not just on the env var: never drop the only
# copy of a build whose install did not actually land. Costs a cold rebuild
# next time, which ccache absorbs when base_dir/hash_dir are set (CLAUDE.md 3).
# ---------------------------------------------------------------------------
if [ "${VIBEQC_PRUNE_BUILD_TREES:-0}" = "1" ]; then
    _pruned_ok=0
    for _libdir in "$INSTALL_DIR/lib" "$INSTALL_DIR/lib64"; do
        [ -d "$_libdir" ] || continue
        if find "$_libdir" -maxdepth 1 \( -name '*.so*' -o -name '*.dylib' -o -name '*.a' \) \
           -print -quit 2>/dev/null | grep -q .; then _pruned_ok=1; break; fi
    done
    if [ "$_pruned_ok" = "1" ]; then
        if [ -d "$BUILD_DIR" ]; then
            echo "pruning $BUILD_DIR (VIBEQC_PRUNE_BUILD_TREES=1)"
            rm -rf "$BUILD_DIR"
        fi
    else
        echo "NOT pruning fftw build tree: no library found under $INSTALL_DIR/lib{,64}" >&2
    fi
    unset _pruned_ok _libdir
fi
