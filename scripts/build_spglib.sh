#!/usr/bin/env bash
# Build spglib from source into third_party/spglib/install/.
#
# spglib's API has shifted a few times during the 1.x → 2.x transition;
# vendoring a known-good version means our space-group analysis output
# never depends on whatever the user happens to have on their system.
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

SPGLIB_VERSION="v2.7.0"
# Commit SHA the upstream tag $SPGLIB_VERSION pointed at when this
# script was pinned. Re-resolve via
#   git ls-remote --tags https://github.com/spglib/spglib.git $SPGLIB_VERSION
# any time SPGLIB_VERSION is bumped.
SPGLIB_COMMIT_SHA="12355c77fb7c505a55f52cae36341d73b781a065"
SPGLIB_DIR="$REPO_ROOT/third_party/spglib"
SRC_DIR="$SPGLIB_DIR/src"
BUILD_DIR="$SPGLIB_DIR/build"
INSTALL_DIR="$SPGLIB_DIR/install"

for candidate in \
        "$INSTALL_DIR/lib/cmake/Spglib/SpglibConfig.cmake" \
        "$INSTALL_DIR/lib64/cmake/Spglib/SpglibConfig.cmake"; do
    if [ -f "$candidate" ]; then
        echo "spglib $SPGLIB_VERSION already installed at $INSTALL_DIR"
        exit 0
    fi
done

if [ ! -f "$SRC_DIR/CMakeLists.txt" ]; then
    echo "Cloning spglib $SPGLIB_VERSION..."
    git_clone_pinned https://github.com/spglib/spglib.git \
        "$SPGLIB_VERSION" "$SPGLIB_COMMIT_SHA" "$SRC_DIR"
fi

# Fetch-only mode for offline-HPC pre-staging (docs/cluster_setup.md):
# stage source on an internet-connected login node, compile on an offline
# compute node. Guarded on VIBEQC_FETCH_ONLY, so default builds are
# unchanged.
if [ "${VIBEQC_FETCH_ONLY:-0}" = "1" ]; then
    echo "VIBEQC_FETCH_ONLY=1 -- spglib source staged at $SRC_DIR; skipping build."
    exit 0
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

case "$(uname -s)" in
    Darwin) PHYS_CORES="$(sysctl -n hw.physicalcpu 2>/dev/null || sysctl -n hw.ncpu)" ;;
    Linux)  PHYS_CORES="$(nproc 2>/dev/null || echo 4)" ;;
    *)      PHYS_CORES=4 ;;
esac

echo "Configuring spglib $SPGLIB_VERSION..."

# OpenMP detection. AppleClang on macOS doesn't ship libomp by
# default, so CMake's FindOpenMP fails the spglib configure if
# we unconditionally pass -DSPGLIB_USE_OMP=ON.
#
# Strategy:
#   - On macOS, look for Homebrew's libomp. If found, point
#     OpenMP_ROOT at it so AppleClang can use OpenMP. If not
#     found, fall back to building spglib without OpenMP — the
#     symmetry-finder is fast enough single-threaded that this
#     is a small loss, and it lets the install complete instead
#     of failing loudly. The user gets a one-line note + a
#     suggestion to ``brew install libomp`` for the parallel
#     build.
#   - On Linux, gcc / clang ship libomp / libgomp by default and
#     CMake finds it without help. Pass SPGLIB_USE_OMP=ON
#     unchanged.
SPGLIB_OMP_FLAG="-DSPGLIB_USE_OMP=ON"
if [[ "$(uname)" == "Darwin" ]]; then
    if command -v brew >/dev/null 2>&1 && brew list libomp >/dev/null 2>&1; then
        export OpenMP_ROOT="$(brew --prefix libomp)"
        echo "  OpenMP: using Homebrew libomp at $OpenMP_ROOT"
    else
        SPGLIB_OMP_FLAG="-DSPGLIB_USE_OMP=OFF"
        echo "  OpenMP: libomp not found via brew — building spglib"
        echo "          single-threaded. To enable parallel symmetry"
        echo "          finding: brew install libomp && re-run this script."
    fi
fi

# CMAKE_POLICY_VERSION_MINIMUM=3.5: defensive — CMake 4.x dropped
# support for cmake_minimum_required(VERSION <3.5). If the vendored
# upstream still declares <3.5 anywhere in its tree, this lets us
# build anyway. -Wno-dev silences any deprecation noise from upstream
# CMakeLists hygiene we don't control.
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -GNinja -Wno-dev \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    $SPGLIB_OMP_FLAG \
    -DSPGLIB_WITH_TESTS=OFF \
    -DSPGLIB_WITH_Fortran=OFF \
    -DBUILD_SHARED_LIBS=ON

# Honor _safe_build_env.sh's cap if it set one; fall back to PHYS_CORES.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$PHYS_CORES}"
echo "Building spglib with $CMAKE_BUILD_PARALLEL_LEVEL parallel jobs..."
cmake --build "$BUILD_DIR" --target install

echo
echo "spglib $SPGLIB_VERSION installed to $INSTALL_DIR"

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
        echo "NOT pruning spglib build tree: no library found under $INSTALL_DIR/lib{,64}" >&2
    fi
    unset _pruned_ok _libdir
fi
