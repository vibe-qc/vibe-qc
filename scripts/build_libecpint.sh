#!/usr/bin/env bash
# Build libecpint (and its hard runtime deps) from pinned source into
# third_party/libecpint/install/.
#
# Why vendor everything: libecpint links pugixml (XML parser) and
# libcerf (complex error function library). Both are common but have
# version drift across distros / Homebrew bottles, and we want a
# byte-identical build across machines. Pinning all three matches the
# pattern used for libint (third_party/libint/install/).
#
# Pinned versions (matching the current Homebrew bottles for libecpint
# 1.0.7):
#   - pugixml 1.15
#   - libcerf 3.3
#   - libecpint 1.0.7
#
# Runtime artefact: ``third_party/libecpint/install/{lib,include}``
# with an ``ECPINT::ecpint`` (alias) target exposed via
# ``lib/cmake/ecpint/ecpint-config.cmake``. The shared libraries
# (pugixml, cerf, ecpint) all live under ``install/lib`` and the
# vibe-qc build bakes RPATH so ``import vibeqc`` finds them.
#
# Re-running is safe: it skips to completion if the install is
# already there. Set REBUILD=1 to force-rebuild from scratch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build-pressure safety (see _safe_build_env.sh comment block).
# shellcheck source=_safe_build_env.sh
. "$SCRIPT_DIR/_safe_build_env.sh"

# Supply-chain integrity (see _verify_source.sh comment block).
# shellcheck source=_verify_source.sh
. "$SCRIPT_DIR/_verify_source.sh"

# Serialize before REBUILD handling, install probes, or source-tree mutation.
# vibeqc-recipe-hash: lifecycle-only-begin
# shellcheck source=_build_lock.sh
. "$SCRIPT_DIR/_build_lock.sh"
vibeqc_acquire_build_lock
# vibeqc-recipe-hash: lifecycle-only-end

LIBECPINT_VERSION="v1.0.7"
PUGIXML_VERSION="v1.15"
LIBCERF_VERSION="v3.3"

# Commit SHAs each upstream tag pointed at when this script was pinned.
# Re-resolve any time the matching VERSION is bumped:
#   git ls-remote --tags https://github.com/robashaw/libecpint.git $LIBECPINT_VERSION
#   git ls-remote --tags https://github.com/zeux/pugixml.git $PUGIXML_VERSION
#   git ls-remote --tags https://jugit.fz-juelich.de/mlz/libcerf.git $LIBCERF_VERSION
LIBECPINT_COMMIT_SHA="f07f8b963d3719ec2667879fceb49deb4e2c6826"
PUGIXML_COMMIT_SHA="ee86beb30e4973f5feffe3ce63bfa4fbadf72f38"
LIBCERF_COMMIT_SHA="7e6b637031b5c6cbc89a5f7220bf4ef07518d035"

ROOT_DIR="$REPO_ROOT/third_party/libecpint"
INSTALL_DIR="$ROOT_DIR/install"

# Clean re-build via REBUILD=1 ./scripts/build_libecpint.sh
if [ "${REBUILD:-0}" = "1" ]; then
    echo "Forcing rebuild — removing $ROOT_DIR..."
    rm -rf "$ROOT_DIR"
fi

# The radial patch changes compiled code, so an old install must rebuild.
RADIAL_PATCH_VERSION="periodic-radial-v8-1"
if [ -f "$INSTALL_DIR/lib/cmake/ecpint/ecpint-config.cmake" ] && \
   [ "$(cat "$INSTALL_DIR/.vibeqc-radial-quadrature" 2>/dev/null || true)" = "$RADIAL_PATCH_VERSION" ]; then
    echo "libecpint already installed at $INSTALL_DIR"
    exit 0
fi

mkdir -p "$ROOT_DIR"

# Fetch-only mode for offline-HPC pre-staging (docs/cluster_setup.md):
# clone all three sources (pugixml, libcerf, libecpint) on an
# internet-connected login node, then compile on an offline compute node
# where they are already present. Guarded on VIBEQC_FETCH_ONLY, so the
# default build path is byte-for-byte unchanged.
if [ "${VIBEQC_FETCH_ONLY:-0}" = "1" ]; then
    { [ -d "$ROOT_DIR/pugixml-src/.git" ] || [ -f "$ROOT_DIR/pugixml-src/CMakeLists.txt" ]; } || \
        git_clone_pinned https://github.com/zeux/pugixml.git \
            "$PUGIXML_VERSION" "$PUGIXML_COMMIT_SHA" "$ROOT_DIR/pugixml-src"
    { [ -d "$ROOT_DIR/libcerf-src/.git" ] || [ -f "$ROOT_DIR/libcerf-src/CMakeLists.txt" ]; } || \
        git_clone_pinned https://jugit.fz-juelich.de/mlz/libcerf.git \
            "$LIBCERF_VERSION" "$LIBCERF_COMMIT_SHA" "$ROOT_DIR/libcerf-src"
    { [ -d "$ROOT_DIR/libecpint-src/.git" ] || [ -f "$ROOT_DIR/libecpint-src/CMakeLists.txt" ]; } || \
        git_clone_pinned https://github.com/robashaw/libecpint.git \
            "$LIBECPINT_VERSION" "$LIBECPINT_COMMIT_SHA" "$ROOT_DIR/libecpint-src"
    echo "VIBEQC_FETCH_ONLY=1 -- libecpint (+pugixml, libcerf) sources staged under $ROOT_DIR; skipping build."
    exit 0
fi

# Use Ninja if available, otherwise CMake's default generator.
GENERATOR_FLAG=""
if command -v ninja >/dev/null 2>&1; then
    GENERATOR_FLAG="-GNinja"
fi

# Honor _safe_build_env.sh's CMAKE_BUILD_PARALLEL_LEVEL cap on Linux (set
# earlier in this script by sourcing the helper). On macOS / non-Linux,
# fall back to the previous getconf-derived nproc count.
NCPU="${CMAKE_BUILD_PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

# ---------------------------------------------------------------------------
# 1. pugixml — lightweight XML parser used by libecpint to read ECP files.
# ---------------------------------------------------------------------------

PUGIXML_SRC="$ROOT_DIR/pugixml-src"
PUGIXML_BUILD="$ROOT_DIR/pugixml-build"

if [ ! -f "$INSTALL_DIR/lib/cmake/pugixml/pugixml-config.cmake" ] && \
   [ ! -f "$INSTALL_DIR/lib/cmake/pugixml/pugixmlConfig.cmake" ]; then
    if [ ! -d "$PUGIXML_SRC/.git" ] && [ ! -f "$PUGIXML_SRC/CMakeLists.txt" ]; then
        echo "Cloning pugixml $PUGIXML_VERSION..."
        git_clone_pinned https://github.com/zeux/pugixml.git \
            "$PUGIXML_VERSION" "$PUGIXML_COMMIT_SHA" "$PUGIXML_SRC"
    fi

    echo "Configuring pugixml..."
    rm -rf "$PUGIXML_BUILD"
    # CMAKE_POLICY_VERSION_MINIMUM=3.5 + -Wno-dev: defensive against
    # CMake 4.x dropping pre-3.5 compatibility from upstream
    # CMakeLists. Same one-liner pattern as build_libxc.sh.
    cmake -S "$PUGIXML_SRC" -B "$PUGIXML_BUILD" $GENERATOR_FLAG -Wno-dev \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=ON

    echo "Building pugixml..."
    cmake --build "$PUGIXML_BUILD" -j "$NCPU"
    cmake --install "$PUGIXML_BUILD"
fi

# ---------------------------------------------------------------------------
# 2. libcerf — complex error function library used by libecpint for the
#    Boys-function machinery.
# ---------------------------------------------------------------------------

LIBCERF_SRC="$ROOT_DIR/libcerf-src"
LIBCERF_BUILD="$ROOT_DIR/libcerf-build"

if [ ! -f "$INSTALL_DIR/lib/cmake/libcerf/libcerfTargets.cmake" ] && \
   [ ! -f "$INSTALL_DIR/lib/cmake/cerf/cerfConfig.cmake" ]; then
    if [ ! -d "$LIBCERF_SRC/.git" ] && [ ! -f "$LIBCERF_SRC/CMakeLists.txt" ]; then
        echo "Cloning libcerf $LIBCERF_VERSION..."
        git_clone_pinned https://jugit.fz-juelich.de/mlz/libcerf.git \
            "$LIBCERF_VERSION" "$LIBCERF_COMMIT_SHA" "$LIBCERF_SRC"
    fi

    echo "Configuring libcerf..."
    rm -rf "$LIBCERF_BUILD"
    cmake -S "$LIBCERF_SRC" -B "$LIBCERF_BUILD" $GENERATOR_FLAG -Wno-dev \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=ON

    echo "Building libcerf..."
    cmake --build "$LIBCERF_BUILD" -j "$NCPU"
    cmake --install "$LIBCERF_BUILD"
fi

# ---------------------------------------------------------------------------
# 3. libecpint itself.
# ---------------------------------------------------------------------------

LIBECPINT_SRC="$ROOT_DIR/libecpint-src"
LIBECPINT_BUILD="$ROOT_DIR/libecpint-build"

if [ ! -d "$LIBECPINT_SRC/.git" ] && [ ! -f "$LIBECPINT_SRC/CMakeLists.txt" ]; then
    echo "Cloning libecpint $LIBECPINT_VERSION..."
    git_clone_pinned https://github.com/robashaw/libecpint.git \
        "$LIBECPINT_VERSION" "$LIBECPINT_COMMIT_SHA" "$LIBECPINT_SRC"
fi

# Apply the cerf::cerf → cerf::cerfcpp fix that the Homebrew formula
# uses (see https://github.com/robashaw/libecpint/issues/65). Idempotent.
SRC_CMAKE="$LIBECPINT_SRC/src/CMakeLists.txt"
if [ -f "$SRC_CMAKE" ] && grep -q "cerf::cerf$" "$SRC_CMAKE"; then
    echo "Patching libecpint to link cerf::cerfcpp..."
    # Use sed without -i.bak portability footgun (BSD vs GNU).
    python3 - "$SRC_CMAKE" <<'PY'
import sys, re, pathlib
path = pathlib.Path(sys.argv[1])
text = path.read_text()
text = re.sub(r'\bcerf::cerf\b(?!cpp)', 'cerf::cerfcpp', text)
path.write_text(text)
PY
fi

# Correct tabulated radial quadrature (#758). Keep upstream MIT notices.
# Exhaust the available nested grid so an early coarse estimate cannot miss
# a displaced Gaussian; compare refinements in integral units and retain all
# angular channels even if one reports non-convergence. Direct trigonometric
# nodes avoid accumulated recurrence error at the original tight tolerance.
python3 - "$LIBECPINT_SRC/src/lib" <<'PY_RADIAL'
from pathlib import Path
import sys
root = Path(sys.argv[1])
changes = [
    ('gaussquad.cpp', [
        (
            (
                '\t\t\tzi = zi1;\n'
                '\t\t\tsi = si1;\n'
                '\t\t\tci = ci1;'
            ),
            (
                '\t\t\tzi = (n + 1) * z1;\n'
                '\t\t\tsi = std::sin((n + 1) * z1);\n'
                '\t\t\tci = std::cos((n + 1) * z1);'
            ),
        ),
        (
            (
                '\t\t\twhile (n < maxN && !converged) {\n'
                '\t\t\t\t// Compute T_{2n+1}\n'
                '\t\t\t\tT2n1 = Tn + sumTerms(f, params, n, start, end, p, 2);\n'
                '\t\t\t\n'
                '\t\t\t\t// Check convergence\n'
                '\t\t\t\tdT = T2n1 - 2.0*Tn;\n'
                '\t\t\t\tn = 2*n + 1;\n'
                '\t\t\t\tif (dT*dT <= fabs(T2n1 - Tn12)*tolerance) {\n'
                '\t\t\t\t\tconverged = true;  \n'
                '\t\t\t\t} else {\n'
                '\t\t\t\t\tTn12 = 4.0 * Tn; \n'
                '\t\t\t\t\tTn = T2n1;\n'
                '\t\t\t\t\tp /= 2; \n'
                '\t\t\t\t}\n'
                '\t\t\t}'
            ),
            (
                '\t\t\t// The integrand is already tabulated on the full grid. A coarse\n'
                '            // nested grid can entirely miss a displaced Gaussian peak.\n'
                '            // Exhaust that grid and report its final refinement error.\n'
                '            while (n < maxN) {\n'
                '\t\t\t\t// Compute T_{2n+1}\n'
                '\t\t\t\tT2n1 = Tn + sumTerms(f, params, n, start, end, p, 2);\n'
                '\t\t\t\n'
                '\t\t\t\t// Check convergence\n'
                '\t\t\t\tdT = T2n1 - 2.0*Tn;\n'
                '\t\t\t\tn = 2*n + 1;\n'
                '                // T values grow with grid size. Test in integral units,\n'
                '                // as the TWOPOINT branch already does. Requiring agreement\n'
                '                // with both preceding nested grids also handles cancellation\n'
                '                // in the extrapolation denominator without dividing by zero.\n'
                '                const double scale = 16.0 / (3.0 * (n + 1.0));\n'
                '                const double recent_change = scale * dT;\n'
                '                const double older_change = scale * (T2n1 - Tn12);\n'
                '                converged = std::isfinite(T2n1) &&\n'
                '                    std::fabs(recent_change) <= tolerance &&\n'
                '                    std::fabs(older_change) <= tolerance;\n'
                '                Tn12 = 4.0 * Tn;\n'
                '                Tn = T2n1;\n'
                '                p /= 2;\n'
                '\t\t\t}'
            ),
        ),
    ]),
    ('radial_quad.cpp', [
        (
            (
                'int test;'
            ),
            (
                'int test = 1;'
            ),
        ),
        (
            (
                'test = integral_and_test.second;'
            ),
            (
                'test = test && integral_and_test.second;'
            ),
        ),
        (
            (
                'if (test == 0) break;'
            ),
            (
                '// Evaluate all angular channels before reporting convergence.'
            ),
        ),
    ]),
]
for filename, edits in changes:
    path = root / filename
    text = path.read_text()
    for old, new in edits:
        if text.count(old) == 1:
            text = text.replace(old, new, 1)
        elif text.count(new) != 1:
            raise SystemExit(f"libecpint radial patch: unexpected upstream {filename}; review source")
    path.write_text(text)
PY_RADIAL

# Preserve generated angular coefficients and complete primitive integrals.
# A bare radial estimate is not a bound on the final contracted AO element:
# ECP coefficients and translated angular factors can amplify omitted terms.
# The outer shell-pair prescreen remains; only this unbudgeted inner drop goes.
python3 - "$LIBECPINT_SRC" <<'PY_PROJECTOR'
from pathlib import Path
import sys

root = Path(sys.argv[1])
generator = root / "src/generate.cpp"
text = generator.read_text()
old = "std::ofstream outfile(ofname);"
new = old + "\n    outfile.precision(17); // Preserve double angular coefficients."
if new not in text:
    if text.count(old) != 1:
        raise SystemExit("libecpint angular patch: unexpected generator; review source")
    text = text.replace(old, new, 1)
generator.write_text(text)

radial = root / "src/lib/radial_gen.cpp"
text = radial.read_text()
old = "if (estimate_type2(k, i, j, u.a, a, b, A, B) > tolerance){"
new = "{ // Retain primitive terms until contraction and angular assembly."
if text.count(old) == 2:
    text = text.replace(old, new)
elif text.count(new) != 2:
    raise SystemExit("libecpint projector patch: unexpected screening; review source")
old = "while (not_in_tail && i < gridSize)"
new = ("// Complete the allocated primitive quadrature table.\n"
       "\t\twhile (i < gridSize)")
if text.count(old) == 1:
    text = text.replace(old, new, 1)
elif text.count(new) != 1:
    raise SystemExit("libecpint projector patch: unexpected radial table; review source")
radial.write_text(text)
PY_PROJECTOR

echo "Configuring libecpint..."
rm -rf "$LIBECPINT_BUILD"
# CMAKE_INSTALL_RPATH='$ORIGIN' (literal, not shell-expanded — single
# quotes prevent the bash side from substituting): tells the linker
# to embed a runtime rpath that the dynamic loader expands to the
# directory of the running .so file. Since libecpint.so, libcerf.so,
# and libpugixml.so all install side-by-side in $INSTALL_DIR/lib/,
# libecpint can find its libcerf and libpugixml deps via $ORIGIN
# alone — no $LD_LIBRARY_PATH, no consumer-side rpath transitivity
# games. macOS gets @loader_path equivalent semantics from CMake's
# MACOSX_RPATH machinery; the $ORIGIN entry on macOS is silently
# ignored (no harm) but having it makes the Linux build self-rooted.
cmake -S "$LIBECPINT_SRC" -B "$LIBECPINT_BUILD" $GENERATOR_FLAG -Wno-dev \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DLIBECPINT_USE_CERF=ON \
    -DLIBECPINT_BUILD_TESTS=OFF \
    -DLIBECPINT_BUILD_DOCS=OFF \
    -DBUILD_SHARED_LIBS=ON

echo "Building libecpint..."
cmake --build "$LIBECPINT_BUILD" -j "$NCPU"
cmake --install "$LIBECPINT_BUILD"
printf '%s\n' "$RADIAL_PATCH_VERSION" > "$INSTALL_DIR/.vibeqc-radial-quadrature"

echo "libecpint installed at $INSTALL_DIR"

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
        if [ -d "$LIBECPINT_BUILD" ]; then
            echo "pruning $LIBECPINT_BUILD (VIBEQC_PRUNE_BUILD_TREES=1)"
            rm -rf "$LIBECPINT_BUILD"
        fi
        if [ -d "$PUGIXML_BUILD" ]; then
            echo "pruning $PUGIXML_BUILD (VIBEQC_PRUNE_BUILD_TREES=1)"
            rm -rf "$PUGIXML_BUILD"
        fi
    else
        echo "NOT pruning libecpint build tree: no library found under $INSTALL_DIR/lib{,64}" >&2
    fi
    unset _pruned_ok _libdir
fi
