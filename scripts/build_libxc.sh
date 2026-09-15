#!/usr/bin/env bash
# Build libxc from source into third_party/libxc/install/.
#
# We pin a known-good libxc version because the system package on different
# distros (Arch, Ubuntu, Fedora, conda) ships everything from 5.x to 7.x with
# subtle ABI differences. Vendoring eliminates that whole class of surprise.
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

LIBXC_VERSION="7.0.0"
# Commit SHA the upstream tag $LIBXC_VERSION pointed at when this
# script was pinned. Re-resolve via
#   git ls-remote --tags https://gitlab.com/libxc/libxc.git $LIBXC_VERSION
# any time LIBXC_VERSION is bumped.
LIBXC_COMMIT_SHA="7bd5bb41415968db94c499a2f093309c9a2dcf53"
LIBXC_DIR="$REPO_ROOT/third_party/libxc"
SRC_DIR="$LIBXC_DIR/src"
BUILD_DIR="$LIBXC_DIR/build"
INSTALL_DIR="$LIBXC_DIR/install"

# libxc may install its CMake config to lib/ or lib64/ depending on distro.
for candidate in \
        "$INSTALL_DIR/lib/cmake/Libxc/LibxcConfig.cmake" \
        "$INSTALL_DIR/lib64/cmake/Libxc/LibxcConfig.cmake"; do
    if [ -f "$candidate" ]; then
        echo "libxc $LIBXC_VERSION already installed at $INSTALL_DIR"
        exit 0
    fi
done

if [ ! -f "$SRC_DIR/CMakeLists.txt" ]; then
    echo "Cloning libxc $LIBXC_VERSION..."
    git_clone_pinned https://gitlab.com/libxc/libxc.git \
        "$LIBXC_VERSION" "$LIBXC_COMMIT_SHA" "$SRC_DIR"
fi

# Fetch-only mode for offline-HPC pre-staging (docs/cluster_setup.md):
# stage source on an internet-connected login node, compile on an offline
# compute node. Guarded on VIBEQC_FETCH_ONLY, so default builds are
# unchanged.
if [ "${VIBEQC_FETCH_ONLY:-0}" = "1" ]; then
    echo "VIBEQC_FETCH_ONLY=1 -- libxc source staged at $SRC_DIR; skipping build."
    exit 0
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

case "$(uname -s)" in
    Darwin) PHYS_CORES="$(sysctl -n hw.physicalcpu 2>/dev/null || sysctl -n hw.ncpu)" ;;
    Linux)  PHYS_CORES="$(nproc 2>/dev/null || echo 4)" ;;
    *)      PHYS_CORES=4 ;;
esac

echo "Configuring libxc $LIBXC_VERSION..."
# -DCMAKE_POLICY_VERSION_MINIMUM=3.5: libxc 7.0.0's CMakeLists declares
# cmake_minimum_required(VERSION <3.5), which CMake 4.x dropped support
# for. This flag tells CMake "yes, treat that as 3.5 anyway." Real fix
# is upstream libxc bumping their declared minimum; until then we
# override the policy on the consume side.
# -Wno-dev: silence libxc's "project() before cmake_minimum_required()"
# dev warning at the top of their CMakeLists. Cosmetic.
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -GNinja -Wno-dev \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DENABLE_FORTRAN=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON

# Honor _safe_build_env.sh's cap if it set one; fall back to PHYS_CORES.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$PHYS_CORES}"
echo "Building libxc with $CMAKE_BUILD_PARALLEL_LEVEL parallel jobs..."
cmake --build "$BUILD_DIR" --target install

echo
echo "libxc $LIBXC_VERSION installed to $INSTALL_DIR"

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
        echo "NOT pruning libxc build tree: no library found under $INSTALL_DIR/lib{,64}" >&2
    fi
    unset _pruned_ok _libdir
fi
