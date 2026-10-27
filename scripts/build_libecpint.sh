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
RADIAL_PATCH_VERSION="periodic-radial-v12-retry"
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

# The patches below are written against upstream text. A kept source tree
# carries the previous patch set (this point is only reached when the install
# marker differs or no install exists), and a later stage can rewrite an
# earlier stage's output so that the earlier one no longer recognises it
# (v11's diagnostics edit inside the radial gaussquad block). Start from the
# pinned upstream files whenever the source is a git checkout.
if [ -d "$LIBECPINT_SRC/.git" ]; then
    git -C "$LIBECPINT_SRC" checkout -q -- .
    git -C "$LIBECPINT_SRC" clean -fdq
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

# libecpint's generated type2(triples, nbase, ...) sizes its scratch buffer
# `values` as nbase+2, where generate.cpp fixes nbase at code-generation time
# from the shell pair's angular momenta alone. The switch below it reads up
# to values[k - 2 + l2] with k = N + u.n + 2 (N and l2 from the radial
# triple), so the read depends on the ECP primitive's radial power u.n, known
# only at runtime. libecpint stores u.n = n - 2 of the Gaussian-format power:
# every record shipped before the pob-TZVP-rev2 Hf, Ta and W ECPs had n <= 2,
# so u.n <= 0 fitted. Those three records carry n = 4 (u.n = 2) and read past
# the end of the heap buffer: a silent over-read on the analytic branch, not
# the "Quadrature failed" fallback, that made the Hf/Ta/W ECP operator
# nondeterministic and wrong by up to ~1e195 Ha (vibe-qc #358). Size the
# buffer from the largest N + l2 among the triples and this primitive's u.n.
# When u.n <= 0 the generated nbase stays in force, so every other record
# keeps its previous code path. The patch recognises an upstream source, the
# earlier v9 attempt (which budgeted N + u.n and missed the + l2 term), and
# its own output, so a kept source tree upgrades in place.
python3 - "$LIBECPINT_SRC" <<'PY_NBASE'
from pathlib import Path
import sys

root = Path(sys.argv[1])
radial = root / "src/lib/radial_gen.cpp"
text = radial.read_text()


def patch(label, upstream, earlier, current):
    global text
    if text.count(current) == 1:
        return
    for old in (upstream, earlier):
        if text.count(old) == 1:
            text = text.replace(old, current, 1)
            return
    raise SystemExit(f"libecpint nbase patch: unexpected source ({label}); review radial_gen.cpp")


patch(
    "triple scan",
    (
        "\t\tint npB = shellB.nprimitive();\n"
        "\t\t\n"
        "\t\t// Loop over primitives in ECP, only considering correct ang. momentum\n"
    ),
    (
        "\t\tint npB = shellB.nprimitive();\n"
        "\n"
        "\t\t// vibe-qc: track the largest radial triple index so the base-integral\n"
        "\t\t// buffer below can be grown past nbase for a high-radial-power ECP\n"
        "\t\t// primitive (see the block comment above this patch in\n"
        "\t\t// build_libecpint.sh).\n"
        "\t\tint max_triple_N = 0;\n"
        "\t\tfor (const Triple& triple : triples) {\n"
        "\t\t\tint triple_N = std::get<0>(triple);\n"
        "\t\t\tif (triple_N > max_triple_N) max_triple_N = triple_N;\n"
        "\t\t}\n"
        "\t\t\n"
        "\t\t// Loop over primitives in ECP, only considering correct ang. momentum\n"
    ),
    (
        "\t\tint npB = shellB.nprimitive();\n"
        "\n"
        "\t\t// vibe-qc #358: the generated switch reads values[N + u.n + l2];\n"
        "\t\t// find the largest N + l2 so the buffer below can cover u.n > 0.\n"
        "\t\tint max_triple_Nl2 = 0;\n"
        "\t\tfor (const Triple& triple : triples) {\n"
        "\t\t\tint triple_Nl2 = std::get<0>(triple) + std::get<2>(triple);\n"
        "\t\t\tif (triple_Nl2 > max_triple_Nl2) max_triple_Nl2 = triple_Nl2;\n"
        "\t\t}\n"
        "\t\t\n"
        "\t\t// Loop over primitives in ECP, only considering correct ang. momentum\n"
    ),
)

patch(
    "values buffer",
    (
        "\t\t\t\t\t\t// Compute base integrals\n"
        "\t\t\t\t\t\tdouble *values = new double[nbase+2]; \n"
        "\t\t\t\t\t\tcompute_base_integrals(2, 3+nbase, p, o_root_p, P1, P2, P1_2, P2_2, X1, X2, oP1, oP2, values); \n"
    ),
    (
        "\t\t\t\t\t\t// Compute base integrals. nbase alone does not budget for this\n"
        "\t\t\t\t\t\t// primitive's radial power u.n (see the block comment above this\n"
        "\t\t\t\t\t\t// patch in build_libecpint.sh); grow the buffer when u.n needs\n"
        "\t\t\t\t\t\t// more room than nbase already has.\n"
        "\t\t\t\t\t\tint local_nbase = nbase;\n"
        "\t\t\t\t\t\tif (max_triple_N + u.n > local_nbase) local_nbase = max_triple_N + u.n;\n"
        "\t\t\t\t\t\tdouble *values = new double[local_nbase+2]; \n"
        "\t\t\t\t\t\tcompute_base_integrals(2, 3+local_nbase, p, o_root_p, P1, P2, P1_2, P2_2, X1, X2, oP1, oP2, values); \n"
    ),
    (
        "\t\t\t\t\t\t// Compute base integrals. vibe-qc #358: the largest index read\n"
        "\t\t\t\t\t\t// is max(N + l2) + u.n, so the buffer needs max(N + l2) + u.n + 1\n"
        "\t\t\t\t\t\t// entries; nbase + 2 only covers that when u.n <= 0.\n"
        "\t\t\t\t\t\tint local_nbase = nbase;\n"
        "\t\t\t\t\t\tif (max_triple_Nl2 + u.n - 1 > local_nbase) local_nbase = max_triple_Nl2 + u.n - 1;\n"
        "\t\t\t\t\t\tdouble *values = new double[local_nbase+2]; \n"
        "\t\t\t\t\t\tcompute_base_integrals(2, 3+local_nbase, p, o_root_p, P1, P2, P1_2, P2_2, X1, X2, oP1, oP2, values); \n"
    ),
)

radial.write_text(text)
PY_NBASE

# libecpint reported radial-quadrature trouble by printing to the process's
# stdout/stderr and then used the unconverged value (vibe-qc #392). vibe-qc's
# native code never writes to a stream (cpp/include/vibeqc/diagnostics.hpp);
# a vendored library must not either. Record each event in a per-thread
# QuadratureDiagnostics instead: the site, the ECP angular channel, and the
# final refinement change and magnitude of the worst failure. The values are
# used exactly as before, so this patch changes no integral. vibe-qc reads the
# record around every operator build and reports it once through its
# diagnostics channel. The same patch adds ECPIntegral::shell_pair_negligible,
# the exact screen compute_shell_pair applies (#382): true iff that call would
# return an all-zero block, so a caller may skip it without changing a sum.
python3 - "$LIBECPINT_SRC" <<'PY_DIAGNOSTICS'
from pathlib import Path
import sys

root = Path(sys.argv[1])


def edit(relative, old, new, count=1):
    path = root / relative
    text = path.read_text()
    if text.count(new) == count:
        return
    if text.count(old) != count:
        raise SystemExit(f"libecpint diagnostics patch: unexpected {relative}; review source")
    path.write_text(text.replace(old, new))


def create(relative, content):
    path = root / relative
    if path.exists():
        existing = path.read_text()
        if "vibe-qc #650" in existing:
            return  # already extended by the retry patch below
        if existing != content:
            raise SystemExit(f"libecpint diagnostics patch: foreign {relative}; review source")
    path.write_text(content)


create("include/libecpint/diagnostics.hpp", """\
// vibe-qc patch (vibe-qc #392): radial quadrature outcomes are recorded per
// thread instead of being printed. Values are unchanged by this record.
#ifndef LIBECPINT_VIBEQC_DIAGNOSTICS_HEAD
#define LIBECPINT_VIBEQC_DIAGNOSTICS_HEAD

#include <array>
#include <cstdint>

#include "config.hpp"

namespace libecpint {

\t/// Sites that can leave a radial integral unconverged.
\tenum QuadratureSite {
\t\tQUADRATURE_SMALL = 0,     ///< radial_gen.cpp integrate_small (128-point table)
\t\tQUADRATURE_TYPE1 = 1,     ///< radial_quad.cpp type 1 (big grid)
\t\tQUADRATURE_TYPE2_BIG = 2, ///< radial_quad.cpp type 2 after the big-grid retry
\t\tN_QUADRATURE_SITES = 3
\t};

\tstruct QuadratureDiagnostics {
\t\t/// failures[site][l]: l is the ECP angular channel (type 1: the local channel).
\t\tstd::array<std::array<std::uint64_t, LIBECPINT_MAX_L + 2>, N_QUADRATURE_SITES> failures{};
\t\t/// Largest final refinement change among failures, and |integral| there.
\t\tstd::array<double, N_QUADRATURE_SITES> max_change{};
\t\tstd::array<double, N_QUADRATURE_SITES> value_at_max{};
\t\t/// Sum of failures over sites and channels (cheap "anything new?" test).
\t\tstd::uint64_t total = 0;
\t\t/// BesselFunction asked for more angular channels than it tabulated.
\t\tstd::uint64_t bessel_lmax_exceeded = 0;
\t\t/// Set by every quadrature: final refinement change and integral.
\t\tdouble last_change = 0.0;
\t\tdouble last_value = 0.0;
\t};

\t/// This thread's record. Reset it before, and read it after, a batch.
\tQuadratureDiagnostics& quadrature_diagnostics();

\t/// Count one unconverged integral at ``site`` for ECP channel ``l``.
\tvoid record_quadrature_failure(QuadratureSite site, int l);
}

#endif
""")

create("src/lib/diagnostics.cpp", """\
// vibe-qc patch (vibe-qc #392); see include/libecpint/diagnostics.hpp.
#include "diagnostics.hpp"

#include <algorithm>
#include <cmath>

namespace libecpint {

\tQuadratureDiagnostics& quadrature_diagnostics() {
\t\tthread_local QuadratureDiagnostics record;
\t\treturn record;
\t}

\tvoid record_quadrature_failure(const QuadratureSite site, const int l) {
\t\tQuadratureDiagnostics& record = quadrature_diagnostics();
\t\tconst int channel = std::max(0, std::min(l, LIBECPINT_MAX_L + 1));
\t\trecord.failures[site][channel] += 1;
\t\trecord.total += 1;
\t\tconst double change = std::fabs(record.last_change);
\t\tif (!(change <= record.max_change[site])) {  // also keeps a NaN visible
\t\t\trecord.max_change[site] = change;
\t\t\trecord.value_at_max[site] = std::fabs(record.last_value);
\t\t}
\t}
}
""")

# Every quadrature leaves its final refinement change in the record.
edit("src/lib/gaussquad.cpp", '#include "gaussquad.hpp"\n',
     '#include "gaussquad.hpp"\n#include "diagnostics.hpp"\n')
edit("src/lib/gaussquad.cpp",
     "\t\tbool converged = false; // 0 for converged, -1 for not converged\n",
     "\t\tbool converged = false; // 0 for converged, -1 for not converged\n"
     "\t\tdouble last_change = 0.0; // vibe-qc #392: final refinement change\n")
edit("src/lib/gaussquad.cpp",
     "                    std::fabs(older_change) <= tolerance;\n",
     "                    std::fabs(older_change) <= tolerance;\n"
     "                last_change = std::fmax(std::fabs(recent_change), std::fabs(older_change));\n")
edit("src/lib/gaussquad.cpp",
     "\t\treturn {I, converged};\n",
     "\t\tQuadratureDiagnostics& record = quadrature_diagnostics();\n"
     "\t\trecord.last_change = last_change;\n"
     "\t\trecord.last_value = I;\n"
     "\t\treturn {I, converged};\n")

# integrate_small: two print-and-continue sites.
edit("src/lib/radial_gen.cpp", '#include "mathutil.hpp"\n',
     '#include "mathutil.hpp"\n#include "diagnostics.hpp"\n')
edit("src/lib/radial_gen.cpp",
     'if (!quadval.second) std::cout << "Quadrature failed" << std::endl;',
     "if (!quadval.second) record_quadrature_failure(QUADRATURE_SMALL, lam);",
     count=2)

# Type 1 and the type 2 big-grid retry. The multi-channel integrate keeps the
# worst failing channel so type 1 records it, not merely the last channel.
edit("src/lib/radial_quad.cpp", '#include "mathutil.hpp"\n',
     '#include "mathutil.hpp"\n#include "diagnostics.hpp"\n')
edit("src/lib/radial_quad.cpp", "\t\tint test = 1;\n",
     "\t\tint test = 1;\n"
     "\t\tdouble worst_change = 0.0, worst_value = 0.0; // vibe-qc #392\n")
edit("src/lib/radial_quad.cpp",
     "\t\t\ttest = test && integral_and_test.second;\n",
     "\t\t\ttest = test && integral_and_test.second;\n"
     "\t\t\tif (!integral_and_test.second\n"
     "\t\t\t    && !(std::fabs(quadrature_diagnostics().last_change) <= worst_change)) {\n"
     "\t\t\t\tworst_change = std::fabs(quadrature_diagnostics().last_change);\n"
     "\t\t\t\tworst_value = integral_and_test.first;\n"
     "\t\t\t}\n")
edit("src/lib/radial_quad.cpp", "\t\treturn test;\n",
     "\t\tif (!test) {\n"
     "\t\t\tquadrature_diagnostics().last_change = worst_change;\n"
     "\t\t\tquadrature_diagnostics().last_value = worst_value;\n"
     "\t\t}\n"
     "\t\treturn test;\n")
edit("src/lib/radial_quad.cpp",
     'if (test == 0) std::cerr << "Failed to converge" << std::endl;',
     "if (test == 0) record_quadrature_failure(QUADRATURE_TYPE1, U.getL());")
edit("src/lib/radial_quad.cpp",
     'if (!integral_and_test.second) std::cerr << "Failed at second attempt" << std::endl;',
     "if (!integral_and_test.second) record_quadrature_failure(QUADRATURE_TYPE2_BIG, l);")

# A Bessel request beyond the table leaves channels unset: count it; vibe-qc refuses.
edit("src/lib/bessel.cpp", '#include "mathutil.hpp"\n',
     '#include "mathutil.hpp"\n#include "diagnostics.hpp"\n')
edit("src/lib/bessel.cpp",
     '\t\t\tstd::cout << "Asked for " << maxL << " but only initialised to maximum L = " << lMax << "\\n";\n',
     "\t\t\tquadrature_diagnostics().bessel_lmax_exceeded += 1;\n")

# The exact screen of compute_shell_pair, callable on its own.
edit("include/libecpint/ecpint.hpp",
     "        TwoIndex<double> &values, int shiftA = 0, int shiftB = 0) const;\n",
     "        TwoIndex<double> &values, int shiftA = 0, int shiftB = 0) const;\n"
     "\n"
     "\t\t/// vibe-qc #382: true iff compute_shell_pair(U, shellA, shellB, values)\n"
     "\t\t/// (no derivative shift) would skip type 1 and every type 2 channel,\n"
     "\t\t/// i.e. return an all-zero block. Same data, bound and tolerance.\n"
     "\t\tbool shell_pair_negligible(\n"
     "\t\t    const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB) const;\n")
edit("src/lib/ecpint.cpp",
     "\tvoid ECPIntegral::compute_shell_pair(\n",
     "\tbool ECPIntegral::shell_pair_negligible(\n"
     "\t    const ECP &U, const GaussianShell &shellA, const GaussianShell &shellB) const {\n"
     "\t\t// Mirrors the data compute_shell_pair builds for its screen (shift 0).\n"
     "\t\tShellPairData data;\n"
     "\t\tconst double* C = U.center();\n"
     "\t\tfor (int d = 0; d < 3; d++) {\n"
     "\t\t\tdata.A[d] = shellA.center()[d] - C[d];\n"
     "\t\t\tdata.B[d] = shellB.center()[d] - C[d];\n"
     "\t\t}\n"
     "\t\tdata.LA = shellA.am();\n"
     "\t\tdata.LB = shellB.am();\n"
     "\t\tdata.maxLBasis = data.LA > data.LB ? data.LA : data.LB;\n"
     "\t\tdata.ncartA = (data.LA+1)*(data.LA+2)/2;\n"
     "\t\tdata.ncartB = (data.LB+1)*(data.LB+2)/2;\n"
     "\t\tdata.A2 = data.A[0]*data.A[0] + data.A[1]*data.A[1] + data.A[2]*data.A[2];\n"
     "\t\tdata.Am = sqrt(data.A2);\n"
     "\t\tdata.A_on_ecp = (data.Am < 1e-6);\n"
     "\t\tdata.B2 = data.B[0]*data.B[0] + data.B[1]*data.B[1] + data.B[2]*data.B[2];\n"
     "\t\tdata.Bm = sqrt(data.B2);\n"
     "\t\tdata.B_on_ecp = (data.Bm < 1e-6);\n"
     "\t\tdouble RAB[3] = {data.A[0] - data.B[0], data.A[1] - data.B[1], data.A[2] - data.B[2]};\n"
     "\t\tdata.RAB2 = RAB[0]*RAB[0] + RAB[1]*RAB[1] + RAB[2]*RAB[2];\n"
     "\t\tdata.RABm = sqrt(data.RAB2);\n"
     "\t\tdouble screens[U.getL() + 1];\n"
     "\t\testimate_type2(U, shellA, shellB, data, screens);\n"
     "\t\tif (!U.noType1() && screens[U.getL()] > tolerance) return false;\n"
     "\t\tfor (int l = 0; l < U.getL(); l++)\n"
     "\t\t\tif (screens[l] > tolerance) return false;\n"
     "\t\treturn true;\n"
     "\t}\n"
     "\n"
     "\tvoid ECPIntegral::compute_shell_pair(\n")
PY_DIAGNOSTICS

# A radial integral that misses its convergence test is recomputed on a larger
# grid before it is used (vibe-qc #650): integrate_small on 2047 instead of 127
# points, type 1 and the type 2 big-grid stage on 4x the big grid. The record
# counts retries, the retries that converged, and the largest
# |I(retry) - I(first)|. Only an integral whose retry also misses the test is
# counted as a failure, and vibe-qc refuses such a build. A converged first
# attempt runs exactly the code it ran before, so its value is unchanged.
# set_quadrature_retry_enabled(false) restores the single-grid behaviour; it
# exists so a test can force the refusal.
python3 - "$LIBECPINT_SRC" <<'PY_RETRY'
from pathlib import Path
import sys

root = Path(sys.argv[1])


def edit(relative, old, new):
    path = root / relative
    text = path.read_text()
    if text.count(new) == 1:
        return
    if text.count(old) != 1:
        raise SystemExit(f"libecpint retry patch: unexpected {relative}; review source")
    path.write_text(text.replace(old, new, 1))


edit("include/libecpint/diagnostics.hpp",
     "\t\tdouble last_value = 0.0;\n\t};\n",
     "\t\tdouble last_value = 0.0;\n"
     "\t\t/// vibe-qc #650: integrals recomputed on the retry grid after the first\n"
     "\t\t/// grid missed the test, the retries that converged, and the largest\n"
     "\t\t/// |I(retry) - I(first)| with |I(retry)| there. ``failures`` counts only\n"
     "\t\t/// integrals that missed the test on the retry grid as well.\n"
     "\t\tstd::array<std::uint64_t, N_QUADRATURE_SITES> retried{};\n"
     "\t\tstd::array<std::uint64_t, N_QUADRATURE_SITES> resolved{};\n"
     "\t\tstd::array<double, N_QUADRATURE_SITES> max_retry_change{};\n"
     "\t\tstd::array<double, N_QUADRATURE_SITES> value_at_max_retry{};\n"
     "\t};\n")
edit("include/libecpint/diagnostics.hpp",
     "\tvoid record_quadrature_failure(QuadratureSite site, int l);\n",
     "\tvoid record_quadrature_failure(QuadratureSite site, int l);\n"
     "\n"
     "\t/// vibe-qc #650: one integral recomputed on the retry grid; ``converged``\n"
     "\t/// is the retry's own test.\n"
     "\tvoid record_quadrature_retry(QuadratureSite site, double first, double retry, bool converged);\n"
     "\n"
     "\t/// vibe-qc #650: process-wide retry switch, on by default. Off reproduces\n"
     "\t/// the single-grid integrals, so a test can force an unconverged build.\n"
     "\tbool quadrature_retry_enabled();\n"
     "\tvoid set_quadrature_retry_enabled(bool enabled);\n")
edit("src/lib/diagnostics.cpp",
     "#include <algorithm>\n#include <cmath>\n",
     "#include <algorithm>\n#include <atomic>\n#include <cmath>\n")
edit("src/lib/diagnostics.cpp",
     "\t\t\trecord.value_at_max[site] = std::fabs(record.last_value);\n\t\t}\n\t}\n}\n",
     "\t\t\trecord.value_at_max[site] = std::fabs(record.last_value);\n\t\t}\n\t}\n"
     "\n"
     "\t// vibe-qc #650\n"
     "\tvoid record_quadrature_retry(\n"
     "\t    const QuadratureSite site, const double first, const double retry, const bool converged) {\n"
     "\t\tQuadratureDiagnostics& record = quadrature_diagnostics();\n"
     "\t\trecord.retried[site] += 1;\n"
     "\t\tif (converged) record.resolved[site] += 1;\n"
     "\t\tconst double change = std::fabs(retry - first);\n"
     "\t\tif (!(change <= record.max_retry_change[site])) {  // also keeps a NaN visible\n"
     "\t\t\trecord.max_retry_change[site] = change;\n"
     "\t\t\trecord.value_at_max_retry[site] = std::fabs(retry);\n"
     "\t\t}\n"
     "\t}\n"
     "\n"
     "\tnamespace {\n"
     "\t\tstd::atomic<bool> retry_enabled{true};\n"
     "\t}\n"
     "\n"
     "\tbool quadrature_retry_enabled() { return retry_enabled.load(std::memory_order_relaxed); }\n"
     "\n"
     "\tvoid set_quadrature_retry_enabled(const bool enabled) {\n"
     "\t\tretry_enabled.store(enabled, std::memory_order_relaxed);\n"
     "\t}\n"
     "}\n")

# The one-point test is absolute (tolerance 1e-15 for type 1 and type 2), but
# an integral of |I| ~ 1 cannot change by less than a few ulp between nested
# grids: on NaCl/LANL2DZ 132 type 1 integrals with |I| = 1.6 stopped at a
# final change of 1.5e-15 on both grids. Accept a change within 32 eps |I|,
# a rounding floor rather than a looser tolerance. The loop always exhausts
# the grid, so this changes the converged flag only, never a value.
edit("src/lib/gaussquad.cpp",
     "                converged = std::isfinite(T2n1) &&\n"
     "                    std::fabs(recent_change) <= tolerance &&\n"
     "                    std::fabs(older_change) <= tolerance;\n",
     "                // vibe-qc #650: rounding floor of the absolute test.\n"
     "                const double rounding = 32.0 * DBL_EPSILON * std::fabs(scale * T2n1);\n"
     "                const double accepted = std::fmax(tolerance, rounding);\n"
     "                converged = std::isfinite(T2n1) &&\n"
     "                    std::fabs(recent_change) <= accepted &&\n"
     "                    std::fabs(older_change) <= accepted;\n")
edit("src/lib/gaussquad.cpp",
     '#include "diagnostics.hpp"\n',
     '#include "diagnostics.hpp"\n#include <cfloat>\n')

# The retry grids live beside the grids they back up.
edit("include/libecpint/radial.hpp",
     "\t\tGCQuadrature primGrid; \n",
     "\t\tGCQuadrature primGrid; \n"
     "\t\t/// vibe-qc #650: retry grids for primGrid and bigGrid\n"
     "\t\tGCQuadrature primRetryGrid;\n"
     "\t\tGCQuadrature retryGrid;\n")
edit("include/libecpint/radial.hpp",
     "\t\t    int N, int l1, int l2, double n, double a, double b, double A, double B) const;\n",
     "\t\t    int N, int l1, int l2, double n, double a, double b, double A, double B,\n"
     "\t\t    const GCQuadrature *grid = nullptr) const;\n")
edit("src/lib/radial_quad.cpp",
     "\t\tprimGrid.initGrid(128, ONEPOINT); \n",
     "\t\tprimGrid.initGrid(128, ONEPOINT); \n"
     "\t\tprimRetryGrid.initGrid(2048, ONEPOINT); // vibe-qc #650\n"
     "\t\tretryGrid.initGrid(4 * large, ONEPOINT);\n")

# integrate_small: without a grid, try primGrid, then primRetryGrid.
edit("src/lib/radial_gen.cpp",
     "      const double a, const double b, const double A, const double B) const {\n"
     "\t\tint gridSize = primGrid.getN();\n",
     "      const double a, const double b, const double A, const double B,\n"
     "      const GCQuadrature *grid) const {\n"
     "\t\tif (grid == nullptr) {  // vibe-qc #650\n"
     "\t\t\tconst auto first = integrate_small(N, l1, l2, n, a, b, A, B, &primGrid);\n"
     "\t\t\tif (first.second || !quadrature_retry_enabled()) return first;\n"
     "\t\t\tconst auto retry = integrate_small(N, l1, l2, n, a, b, A, B, &primRetryGrid);\n"
     "\t\t\trecord_quadrature_retry(QUADRATURE_SMALL, first.first, retry.first, retry.second);\n"
     "\t\t\treturn retry;\n"
     "\t\t}\n"
     "\t\tint gridSize = grid->getN();\n")
edit("src/lib/radial_gen.cpp",
     "\t\tauto transformedGrid = primGrid;\n",
     "\t\tauto transformedGrid = *grid;\n")
edit("src/lib/radial_gen.cpp",
     "return transformedGrid.integrate(intgd, Ftab, 1e-12, 0, primGrid.getN() - 1);",
     "return transformedGrid.integrate(intgd, Ftab, 1e-12, 0, grid->getN() - 1);")

# Type 1: the per-primitive-pair tabulation becomes a function of the grid.
edit("src/lib/radial_quad.cpp",
     "\t\t\t\t// Reset grid starting points\n"
     "\t\t\t\tGCQuadrature newGrid = bigGrid;\n",
     "\t\t\t\t// vibe-qc #650: tabulate and integrate this primitive pair on base;\n"
     "\t\t\t\t// retried on retryGrid when bigGrid misses the convergence test.\n"
     "\t\t\t\tauto quadrature = [&](const GCQuadrature &base, std::vector<double> &out) -> int {\n"
     "\t\t\t\tconst int gridSize = base.getN();\n"
     "\t\t\t\tTwoIndex<double> intValues(maxL+1, gridSize, 0.0);\n"
     "\t\t\t\tTwoIndex<double> besselValues(maxL+1, gridSize, 0.0);\n"
     "\t\t\t\t// Reset grid starting points\n"
     "\t\t\t\tGCQuadrature newGrid = base;\n")
edit("src/lib/radial_quad.cpp",
     "\t\t\t\tint test = integrate(maxL, gridSize, intValues, newGrid, tempValues, start, end, offset, 2);\n"
     "\t\t\t\tif (test == 0) record_quadrature_failure(QUADRATURE_TYPE1, U.getL());\n",
     "\t\t\t\treturn integrate(maxL, gridSize, intValues, newGrid, out, start, end, offset, 2);\n"
     "\t\t\t\t};\n"
     "\t\t\t\tint test = quadrature(bigGrid, tempValues);\n"
     "\t\t\t\tif (test == 0 && quadrature_retry_enabled()) {\n"
     "\t\t\t\t\tstd::vector<double> retryValues;\n"
     "\t\t\t\t\ttest = quadrature(retryGrid, retryValues);\n"
     "\t\t\t\t\tdouble change = -1.0, first = 0.0, retry = 0.0;\n"
     "\t\t\t\t\tfor (int l = offset; l <= maxL; l+=2) {\n"
     "\t\t\t\t\t\tconst double d = std::fabs(retryValues[l] - tempValues[l]);\n"
     "\t\t\t\t\t\tif (!(d <= change)) { change = d; first = tempValues[l]; retry = retryValues[l]; }\n"
     "\t\t\t\t\t}\n"
     "\t\t\t\t\trecord_quadrature_retry(QUADRATURE_TYPE1, first, retry, test != 0);\n"
     "\t\t\t\t\ttempValues = retryValues;\n"
     "\t\t\t\t}\n"
     "\t\t\t\tif (test == 0) record_quadrature_failure(QUADRATURE_TYPE1, U.getL());\n")

# Type 2: a channel that misses the test on the big grid is redone on retryGrid.
edit("src/lib/radial_quad.cpp",
     "\t\t\t\t\tdouble params2[gridSize]; \n",
     "\t\t\t\t\t// vibe-qc #650: one (l1, l2) channel of this primitive pair on retryGrid.\n"
     "\t\t\t\t\tauto retry_channel = [&](const int l1, const int l2) {\n"
     "\t\t\t\t\t\tconst int n_retry = retryGrid.getN();\n"
     "\t\t\t\t\t\tGCQuadrature grid = retryGrid;\n"
     "\t\t\t\t\t\tgrid.transformRMinMax(p(a, b), (zeta_a * A + zeta_b * B)/p(a, b));\n"
     "\t\t\t\t\t\tstd::vector<double> &r = grid.getX();\n"
     "\t\t\t\t\t\tstd::vector<double> u(n_retry);\n"
     "\t\t\t\t\t\tbuildU(U, l, N, grid, u.data());\n"
     "\t\t\t\t\t\tTwoIndex<double> fa(l1end+1, n_retry, 0.0), fb(l2end+1, n_retry, 0.0);\n"
     "\t\t\t\t\t\tbuildBessel(r, n_retry, l1end, fa, 2.0*zeta_a*A);\n"
     "\t\t\t\t\t\tbuildBessel(r, n_retry, l2end, fb, 2.0*zeta_b*B);\n"
     "\t\t\t\t\t\tstd::vector<double> values2(n_retry);\n"
     "\t\t\t\t\t\tfor (int i = 0; i < n_retry; i++) {\n"
     "\t\t\t\t\t\t\tconst double ra = r[i] - A, rb = r[i] - B;\n"
     "\t\t\t\t\t\t\tvalues2[i] = std::exp(-zeta_a*ra*ra - zeta_b*rb*rb) * u[i] * fa(l1, i) * fb(l2, i);\n"
     "\t\t\t\t\t\t}\n"
     "\t\t\t\t\t\treturn grid.integrate(intgd, values2.data(), tolerance, 0, n_retry - 1);\n"
     "\t\t\t\t\t};\n"
     "\t\t\t\t\tdouble params2[gridSize]; \n")
edit("src/lib/radial_quad.cpp",
     "\t\t\t\t\t\t\t\tconst auto integral_and_test =\n"
     "\t\t\t\t\t\t\t\t    newGrid.integrate(intgd, params2, tolerance, start, end);\n"
     "\t\t\t\t\t\t\t\tif (!integral_and_test.second) record_quadrature_failure(QUADRATURE_TYPE2_BIG, l);\n",
     "\t\t\t\t\t\t\t\tauto integral_and_test =\n"
     "\t\t\t\t\t\t\t\t    newGrid.integrate(intgd, params2, tolerance, start, end);\n"
     "\t\t\t\t\t\t\t\tif (!integral_and_test.second && quadrature_retry_enabled()) {\n"
     "\t\t\t\t\t\t\t\t\tconst auto retry = retry_channel(l1, l2);\n"
     "\t\t\t\t\t\t\t\t\trecord_quadrature_retry(QUADRATURE_TYPE2_BIG, integral_and_test.first,\n"
     "\t\t\t\t\t\t\t\t\t                        retry.first, retry.second);\n"
     "\t\t\t\t\t\t\t\t\tintegral_and_test = retry;\n"
     "\t\t\t\t\t\t\t\t}\n"
     "\t\t\t\t\t\t\t\tif (!integral_and_test.second) record_quadrature_failure(QUADRATURE_TYPE2_BIG, l);\n")
PY_RETRY

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
