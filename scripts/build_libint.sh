#!/usr/bin/env bash
# Build libint2 from source into third_party/libint/install/.
#
# Configuration: max_am for derivative orders 0/1/2 — 5/4/4 by default,
# overridable via VIBEQC_LIBINT_MAX_AM (or the --libint-max-am flag on
# install.sh / update.sh / update_native_deps.sh); see
# scripts/_libint_max_am.sh. The setting applies to:
#   * 1-body integrals (overlap, kinetic, nuclear-attraction);
#   * 4-centre ERI (μν|λσ);
# The auxiliary centers of 3-centre (P|μν) and 2-centre (P|Q) fitting
# integrals separately support i functions at orders 0/1/2. The paired
# orbital centers of (P|μν) retain libint's global orbital limit.
# Derivative-order 2 unlocks the analytic RHF Hessian pipeline
# (Phase 17b): ∂²S, ∂²(T+V), and ∂²(μν|λσ) all contracted directly
# into a 3N × 3N skeleton Hessian. Derivative-order 1 covers nuclear
# gradients (Phase 16) for both the four-index and the DF (RI-J,
# RI-K, DF-MP2) two-electron paths.
#
# Build cost note: enabling deriv_order=2 adds substantially to the
# generated source — libint emits one source file per integral kernel,
# and the count grows roughly as max_am^4 × (deriv_order+1). The
# deriv_order=2 stratum was raised from max_am=3 to max_am=4 on
# 2026-08-06, so that stratum's kernel count grows by ~(4/3)^4 ≈ 3×;
# budget roughly 1.5–2 h total on an M-class macbook (it was ~30 min
# at max_am=3, and ~10 min for deriv_order=1 only). What that buys is
# analytic Hessians on g-function basis sets — def2-TZVPP heavy
# atoms, cc-pVQZ — which the f-function tier could not do at all.
# max_am still tightens as derivative order rises, matching the
# convention used by MPQC, NWChem, ORCA, and PySCF; the 2nd-derivative
# assembly step gets memory-bound at high L.
#
# The spec is a choice, not a fixed cost. `--libint-max-am 5_4_3`
# restores the pre-2026-08-06 ~30 min build, giving up only g-function
# analytic Hessians; `6_5_4` goes the other way and adds i-function
# (L = 6) energies for cc-pV6Z-class work, at a substantially longer
# build again. 6 is the ceiling, and it is vibe-qc's rather than
# libint's — see the CEILING note in scripts/_libint_max_am.sh.
#
# This script refuses to short-circuit on an install/ tree built with a
# different spec, and the native-dep stamp folds the spec into libint's
# recipe hash, so switching is a normal `./scripts/update.sh` away in
# any direction.
#
# Cross-platform: works on macOS (Homebrew) and Linux (system packages).
# Re-running is safe: it skips to completion if the install is already there.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build-pressure safety: on Linux, self-re-exec under nice/ionice and set
# a memory-safe CMAKE_BUILD_PARALLEL_LEVEL. macOS / non-Linux: no-op (the
# PHYS_CORES default below governs parallelism). See _safe_build_env.sh.
# shellcheck source=_safe_build_env.sh
. "$SCRIPT_DIR/_safe_build_env.sh"

# Supply-chain integrity (see _verify_source.sh comment block).
# shellcheck source=_verify_source.sh
. "$SCRIPT_DIR/_verify_source.sh"

# Serialize before even the idempotence probe: another lifecycle command may
# be replacing the install tree while this direct helper inspects it. Nested
# setup/install callers reuse the validated process-tree lock.
# vibeqc-recipe-hash: lifecycle-only-begin
# shellcheck source=_build_lock.sh
. "$SCRIPT_DIR/_build_lock.sh"
vibeqc_acquire_build_lock
# vibeqc-recipe-hash: lifecycle-only-end

# Per-derivative-order max_am, from VIBEQC_LIBINT_MAX_AM (default 5_4_4).
# Publishes VIBEQC_LIBINT_AM_{SPEC,LIST,GLOBAL}; hard-fails on a malformed
# spec before anything is cloned or compiled.
# shellcheck source=_libint_max_am.sh
. "$SCRIPT_DIR/_libint_max_am.sh"
vibeqc_libint_max_am_resolve || exit 1

# Physical def2 fitting bases include i functions even for spdf orbitals.
# This is independent of the expensive four-center orbital kernel setting.
# Keep the literal in the recipe so changing it invalidates the native stamp.
LIBINT_FITTING_AM_SPEC="6_6_6"
LIBINT_FITTING_AM_LIST="6;6;6"

LIBINT_VERSION="v2.13.1"
# Commit SHA the upstream tag $LIBINT_VERSION pointed at when this
# script was pinned. Re-resolve via
#   git ls-remote --tags https://github.com/evaleev/libint.git $LIBINT_VERSION
# any time LIBINT_VERSION is bumped.
LIBINT_COMMIT_SHA="5fb07b4862f219749c51d59ee08073d20a56d506"
LIBINT_DIR="$REPO_ROOT/third_party/libint"
SRC_DIR="$LIBINT_DIR/src"
BUILD_DIR="$LIBINT_DIR/build"
INSTALL_DIR="$LIBINT_DIR/install"

# Libint 2.13.1's Chebyshev table covers [0, 117). At exactly 117,
# the original strict comparison indexes one interval beyond its allocation
# (#754, found by ASan in the periodic GDF double-image SR build). Boys
# evaluation is in the C++ API header; generated integral kernels are unchanged.
# Keep this patch in the recipe itself so the native stamp records it.
patch_boys_endpoint() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
start = text.index("class FmEval_Chebyshev7")
end = text.index("  }  // eval()", start)
section = text[start:end]
old, new = "if (x > T_crit) {", "if (x >= T_crit) {"
if section.count(old) == 1 and new not in section:
    section = section.replace(old, new)
    path.write_text(text[:start] + section + text[end:])
elif section.count(new) != 1 or old in section:
    raise SystemExit("libint Boys endpoint patch: unexpected evaluator; review the upstream header")
PY
}

# Both markers must match. A legacy install with only an orbital marker
# predates the independent auxiliary limits and must be rebuilt.
REBUILD_FOR_MAX_AM=0
INSTALLED_AM_SPEC="$(vibeqc_libint_installed_spec "$INSTALL_DIR/.vibeqc-max-am")"
INSTALLED_FITTING_AM_SPEC="$(vibeqc_libint_installed_spec "$INSTALL_DIR/.vibeqc-fitting-max-am")"

if [ -f "$INSTALL_DIR/lib/cmake/libint2/libint2-config.cmake" ]; then
    patch_boys_endpoint "$INSTALL_DIR/include/libint2/boys.h"
    if [ "$INSTALLED_AM_SPEC" = "$VIBEQC_LIBINT_AM_SPEC" ] \
       && [ "$INSTALLED_FITTING_AM_SPEC" = "$LIBINT_FITTING_AM_SPEC" ]; then
        echo "libint2 already installed at $INSTALL_DIR (orbital max_am $INSTALLED_AM_SPEC; fitting max_am $INSTALLED_FITTING_AM_SPEC)"
        exit 0
    fi
    echo "libint2 at $INSTALL_DIR has orbital/fitting max_am $INSTALLED_AM_SPEC/$INSTALLED_FITTING_AM_SPEC,"
    echo "but $VIBEQC_LIBINT_AM_SPEC/$LIBINT_FITTING_AM_SPEC was requested — rebuilding."
    REBUILD_FOR_MAX_AM=1
fi

# --- Source ---------------------------------------------------------------
if [ ! -d "$SRC_DIR/.git" ] && [ ! -f "$SRC_DIR/CMakeLists.txt" ]; then
    echo "Cloning libint $LIBINT_VERSION..."
    git_clone_pinned https://github.com/evaleev/libint.git \
        "$LIBINT_VERSION" "$LIBINT_COMMIT_SHA" "$SRC_DIR"
else
    echo "Using existing libint source at $SRC_DIR"
fi

patch_boys_endpoint "$SRC_DIR/include/libint2/boys.h"

# Fetch-only mode for offline-HPC pre-staging (docs/cluster_setup.md): on
# an internet-connected login node, stage the source here, then run the
# compile on an offline compute node where the source is already present.
# Exits before the prefix/header preflight + build, both of which need the
# full toolchain. Guarded on VIBEQC_FETCH_ONLY, so default builds are
# byte-for-byte unchanged.
if [ "${VIBEQC_FETCH_ONLY:-0}" = "1" ]; then
    echo "VIBEQC_FETCH_ONLY=1 -- libint source staged at $SRC_DIR; skipping configure/build."
    exit 0
fi

# --- Platform-aware prefix detection --------------------------------------
# libint's code generator pulls in Boost (header-only), Eigen3 (header-only)
# and GMP (with C++ bindings). We don't vendor these — they're tiny system
# packages and GMP in particular has architecture-specific assembly that's
# best left to the distro. We do help CMake find them on macOS where Homebrew
# installs under non-standard paths. On Linux the system paths are searched
# by default; we sanity-check the headers and bail with a copy-paste install
# hint if any are missing.
EXTRA_PREFIX_PATH=""
case "$(uname -s)" in
    Darwin)
        if ! command -v brew >/dev/null 2>&1; then
            echo "Error: Homebrew not found. Install via https://brew.sh and retry." >&2
            echo "Required formulae: boost eigen gmp cmake ninja" >&2
            exit 1
        fi
        HOMEBREW_PREFIX="$(brew --prefix)"
        BOOST_PREFIX="$(brew --prefix boost)"
        EIGEN_PREFIX="$(brew --prefix eigen)"
        GMP_PREFIX="$(brew --prefix gmp)"
        EXTRA_PREFIX_PATH="$BOOST_PREFIX;$EIGEN_PREFIX;$GMP_PREFIX;$HOMEBREW_PREFIX"
        PHYS_CORES="$(sysctl -n hw.physicalcpu 2>/dev/null || sysctl -n hw.ncpu)"
        ;;
    Linux)
        MISSING=()

        # Conda/mamba toolchain (e.g. the compute-cluster cluster's Miniforge env —
        # docs/cluster_setup.md): boost / eigen / gmp live under
        # $CONDA_PREFIX/include, not /usr/include. Fold the active prefix
        # into the header search set and into CMAKE_PREFIX_PATH (consumed
        # below) so both this preflight and libint's own find_package()
        # resolve them. Guarded on $CONDA_PREFIX, so a non-conda build is
        # byte-for-byte unchanged.
        _vqc_conda_inc=""
        if [ -n "${CONDA_PREFIX:-}" ]; then
            _vqc_conda_inc="$CONDA_PREFIX/include"
            EXTRA_PREFIX_PATH="$CONDA_PREFIX"
        fi

        # gmpxx.h — search known prefixes. Some distros put it at
        # /usr/include/gmpxx.h, others under a multi-arch subdir like
        # /usr/include/x86_64-linux-gnu/gmpxx.h.
        if ! find /usr/include /usr/local/include ${_vqc_conda_inc:+"$_vqc_conda_inc"} \
                -maxdepth 3 -name gmpxx.h 2>/dev/null | grep -q .; then
            MISSING+=("gmp + C++ bindings")
        fi

        # Eigen — pkg-config first (works on every distro that ships
        # eigen3.pc, including Arch's eigen-5 which keeps the same .pc
        # name; conda's PKG_CONFIG_PATH makes it work there too), then
        # fall back to common include paths.
        if ! pkg-config --exists eigen3 2>/dev/null \
                && [ ! -e /usr/include/eigen3/Eigen/Core ] \
                && [ ! -e /usr/include/Eigen/Core ] \
                && [ ! -e "${_vqc_conda_inc}/eigen3/Eigen/Core" ]; then
            MISSING+=("eigen3 / eigen5")
        fi

        # Boost (header-only)
        if [ ! -e /usr/include/boost/version.hpp ] \
                && [ ! -e /usr/local/include/boost/version.hpp ] \
                && [ ! -e "${_vqc_conda_inc}/boost/version.hpp" ]; then
            MISSING+=("boost (headers)")
        fi

        if [ ${#MISSING[@]} -gt 0 ]; then
            echo "Error: libint's code generator needs system packages that aren't installed:" >&2
            for m in "${MISSING[@]}"; do echo "    - $m" >&2; done
            echo >&2
            echo "Install with one of:" >&2
            echo "    Arch / Manjaro:  sudo pacman -S gmp eigen boost cmake ninja" >&2
            echo "                     (Arch's 'eigen' is now v5; libint 2.13.1 supports it.)" >&2
            echo "    Debian / Ubuntu: sudo apt install libgmp-dev libgmpxx4ldbl libeigen3-dev libboost-dev cmake ninja-build" >&2
            exit 1
        fi
        PHYS_CORES="$(nproc 2>/dev/null || echo 4)"
        ;;
    *)
        echo "Error: unsupported platform $(uname -s)." >&2
        exit 1
        ;;
esac

# --- Configure + build + install ------------------------------------------
rm -rf "$BUILD_DIR"
# A max_am change also has to take the install tree with it: libint bakes
# the per-stratum limits into installed headers (libint2_params.h), so a
# tree half-overwritten by a differently-configured install is worse than
# no tree at all.
if [ "$REBUILD_FOR_MAX_AM" = "1" ]; then
    rm -rf "$INSTALL_DIR"
fi
mkdir -p "$BUILD_DIR"

CMAKE_ARGS=(
    -GNinja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
    -DLIBINT2_REQUIRE_CXX_API=ON
    # Derivative orders 0, 1, 2 enabled for both one-body and ERI. The
    # `LIBINT2_ENABLE_*` flag value is the highest-derivative-order to
    # generate source for — set to 2 here to unlock the analytic
    # Hessian (Phase 17b).
    -DLIBINT2_ENABLE_ONEBODY=2
    # Enable geometric derivatives of the 1-body PROPERTY integrals
    # (emultipole{1,2,3} multipole moments). libint disables these by
    # default (only overlap/kinetic/elecpot 1-body ops get derivatives);
    # turning the disable OFF generates ∂⟨r^n⟩/∂R, which powers the
    # analytic EXT EL-SPHEROPOLE gradient
    # (compute_ext_el_spheropole_gradient_lattice) and, more generally,
    # dipole/quadrupole gradients (IR intensities, polarizability
    # gradients). Adds modestly to the generated-source build time/size.
    -DLIBINT2_DISABLE_ONEBODY_PROPERTY_DERIVS=OFF
    -DLIBINT2_ENABLE_ERI=2
    # 3-centre (BraKet::xs_xx) and 2-centre (BraKet::xs_xs) ERI for
    # density fitting. compute_2c_eri / compute_3c_eri (deriv 0) feed
    # the SCF DF Fock build; the deriv-1 variants feed the DF analytic
    # gradient (commits 4a/4b). Auxiliary-center limits are independent
    # of the four-index ERI and onebody integral tiers.
    -DLIBINT2_ENABLE_ERI3=2
    -DLIBINT2_ENABLE_ERI2=2
    # Global cap. libint requires it to cover every per-order stratum,
    # so the resolver sets it to the largest of the three.
    -DLIBINT2_MAX_AM="$VIBEQC_LIBINT_AM_GLOBAL"
    # Per-derivative-order max-am lists. List index k = derivative
    # order, value = max angular momentum to support at that order.
    # Tightening max_am as deriv_order increases is standard QC
    # practice (the source-code-generator output explodes with both).
    # Default 5;4;4 — every order >= 4 (g functions): deriv 0 and 1
    # always were, and deriv 2 was raised 3 -> 4 on 2026-08-06 so
    # analytic Hessians cover g-function basis sets. Override with
    # VIBEQC_LIBINT_MAX_AM; see the build cost note above.
    -DLIBINT2_ERI_MAX_AM="$VIBEQC_LIBINT_AM_LIST"
    -DLIBINT2_ERI3_MAX_AM="$LIBINT_FITTING_AM_LIST"
    -DLIBINT2_ERI2_MAX_AM="$LIBINT_FITTING_AM_LIST"
    -DLIBINT2_ONEBODY_MAX_AM="$VIBEQC_LIBINT_AM_LIST"
    -DBUILD_SHARED_LIBS=ON
)
if [ -n "$EXTRA_PREFIX_PATH" ]; then
    CMAKE_ARGS+=( -DCMAKE_PREFIX_PATH="$EXTRA_PREFIX_PATH" )
fi

echo "Configuring libint2 (max_am ${VIBEQC_LIBINT_AM_LIST} for deriv 0/1/2; 1body + ERI)..."
# -Wno-dev: silence CMake dev-warnings from libint's CMakeLists. We're
# consuming libint, not developing it. Notably this hides CMP0167's
# "FindBoost module is removed" warning from CMake 3.30+ — fix is
# upstream's responsibility (adopt find_package(Boost CONFIG REQUIRED)
# + Boost::headers); meanwhile the build works fine.
cmake -Wno-dev -S "$SRC_DIR" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"

# Throttle ninja parallelism. Honor pre-existing CMAKE_BUILD_PARALLEL_LEVEL
# (set by _safe_build_env.sh on Linux, or by the user) and fall back to
# PHYS_CORES on macOS / when the helper didn't fire. Before 2026-05-16
# this line unconditionally set PHYS_CORES, which silently undid update.sh's
# cap — that's the root cause of the two compute-host-d hard resets that day.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$PHYS_CORES}"

echo "Building libint2 with $CMAKE_BUILD_PARALLEL_LEVEL parallel jobs..."
# vibeqc-recipe-hash: lifecycle-only-begin
# Quiet build. The generated-source build emits megabytes of codegen +
# compile chatter — enough that a GitLab CI trace hits the runner's 4 MB
# log cap mid-build, and a failure past that point leaves no error text
# at all (the v0.15.131 release-gate post-mortem: three cold builds died
# ~127 min after codegen with their diagnostics discarded). Full output
# goes to third_party/libint/build.log; the terminal gets one heartbeat
# line per minute (elapsed, last ninja step, disk/memory headroom) and,
# on failure, the last 200 log lines plus a resource snapshot.
# VIBEQC_BUILD_VERBOSE=1 restores full streaming. Logging only — the
# built artifact is identical either way, which is why this block is
# lifecycle-only: it must not rotate libint's recipe hash and trigger
# fleet-wide rebuilds (tests/test_build_log_contract.py pins that).
# shellcheck source=_build_log.sh
. "$SCRIPT_DIR/_build_log.sh"
vibeqc_build_log_begin "$LIBINT_DIR/build.log" "libint2"
# vibeqc-recipe-hash: lifecycle-only-end
cmake --build "$BUILD_DIR" --target install
# vibeqc-recipe-hash: lifecycle-only-begin
vibeqc_build_log_end
# vibeqc-recipe-hash: lifecycle-only-end

# --- openSUSE lib64 reconciliation -----------------------------------------
# libint's superbuild compiles the library in a sub-project that *installs*
# into <build>/.../library-install-stage/, which the top-level install then
# copies into $INSTALL_DIR. On distros whose CMake GNUInstallDirs resolves
# the libdir to "lib64" (openSUSE / Tumbleweed -- the compute-cluster compute nodes;
# docs/cluster_setup.md), the staged libdir (lib64) and the top-level copy
# (lib) diverge, so the compiled libint2.so / libint2-cxx.so never reach
# $INSTALL_DIR even though the exported CMake targets still reference them
# ("$INSTALL_DIR/lib64/libint2.so"). Complete the copy from libint's own
# staged install tree. No-op where the library already installed correctly
# (Arch / Ubuntu -> "lib"); guarded on the shared lib being absent.
# The .dylib arm makes macOS (correct lib/libint2.dylib) match too, so
# this block no-ops there instead of the hard guard below false-aborting.
# Two single-pattern `ls` calls ANDed — NOT one `ls .so* .dylib`: ls exits
# non-zero if *any* operand is missing, so the unmatched .so* glob would
# sink the combined call on macOS and defeat the dylib recognition.
if ! ls "$INSTALL_DIR"/lib*/libint2.so* >/dev/null 2>&1 \
   && ! ls "$INSTALL_DIR"/lib*/libint2.dylib >/dev/null 2>&1; then
    stage_dir="$(find "$BUILD_DIR" -type d -name library-install-stage 2>/dev/null | head -1)"
    if [ -n "$stage_dir" ]; then
        echo "Reconciling libint install from staged tree (lib/lib64 divergence): $stage_dir"
        for sub in lib lib64; do
            [ -d "$stage_dir/$sub" ] || continue
            mkdir -p "$INSTALL_DIR/$sub"
            cp -a "$stage_dir/$sub/." "$INSTALL_DIR/$sub/"
        done
    fi
fi
if ! ls "$INSTALL_DIR"/lib*/libint2.so* >/dev/null 2>&1 \
   && ! ls "$INSTALL_DIR"/lib*/libint2.dylib >/dev/null 2>&1; then
    echo "Error: libint built but libint2.{so,dylib} is not installed under $INSTALL_DIR/lib(64)." >&2
    find "$BUILD_DIR" \( -name 'libint2*.so*' -o -name 'libint2*.dylib' \) 2>/dev/null | sed 's/^/    staged: /' >&2
    exit 1
fi

# Record what this tree was actually built with, so a later run (and
# doctor.sh) can tell a 5_4_4 install from a 5_4_3 one. Written only after
# the artifact check above passed, so the marker never claims a spec for a
# tree that did not finish installing.
{
    echo "# libint per-derivative-order max_am this install tree was built with."
    echo "# Read by scripts/_libint_max_am.sh (vibeqc_libint_installed_spec),"
    echo "# scripts/build_libint.sh and scripts/doctor.sh. Do not edit by hand."
    echo "$VIBEQC_LIBINT_AM_SPEC"
} > "$INSTALL_DIR/.vibeqc-max-am"
{
    echo "# Auxiliary-center max_am for derivative orders 0/1/2. Do not edit."
    echo "$LIBINT_FITTING_AM_SPEC"
} > "$INSTALL_DIR/.vibeqc-fitting-max-am"

echo
echo "libint2 installed to $INSTALL_DIR (max_am $VIBEQC_LIBINT_AM_SPEC)"
ls "$INSTALL_DIR/lib/cmake/libint2/"

# Hint at the next step. End users running build_libint.sh directly
# (rather than via setup_native_deps.sh) need libxc + spglib + fftw +
# libecpint too before vibe-qc's pip install will work.
echo
echo "Next: build the remaining native deps (libxc, spglib, fftw,"
echo "libecpint) and assemble the basis library. The orchestrator"
echo "script calls libint as a no-op since it's already built:"
echo "    ./scripts/setup_native_deps.sh"

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
        echo "NOT pruning libint build tree: no library found under $INSTALL_DIR/lib{,64}" >&2
    fi
    unset _pruned_ok _libdir
fi
