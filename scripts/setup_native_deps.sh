#!/usr/bin/env bash
# Bootstrap every native (C/C++) dependency vibe-qc links against.
#
# Each per-dep script is idempotent: if the install is already there
# it short-circuits. So running this on every checkout is cheap, and
# running it after `git pull` will only rebuild whatever has changed.
#
# Order matters: setup_basis_library.sh reads from the libint install,
# so libint must be built first.
#
# Before any building happens, this script runs a comprehensive
# preflight check that verifies every build-time prerequisite is
# present:
#
#   * Build tooling — cmake (≥3.20), ninja, make, git, curl, tar,
#     pkg-config, a C++17 compiler.
#   * Headers libint's code generator needs — Boost, Eigen, GMP (with
#     C++ bindings, i.e. gmpxx.h).
#   * OpenMP — comes for free on Linux gcc/clang; on macOS we look for
#     Homebrew's libomp because AppleClang ships without OpenMP.
#   * An optimised BLAS+LAPACK — Apple Accelerate on macOS (always
#     present; we just sanity-check the framework is reachable); on
#     Linux either OpenBLAS / MKL (good) or reference netlib BLAS
#     (functional but slow — we warn) or nothing (we hard-fail with
#     install instructions).
#
# Missing items are accumulated into a single report so the user
# doesn't go round-trip-by-round-trip on `brew install X; rerun;
# brew install Y; rerun`. The report includes the per-distro install
# one-liner.
#
# Opt-in environment flags:
#
#   VIBEQC_PRUNE_BUILD_TREES=1
#                       Drop each third_party/<dep>/build tree once its
#                       install/ is verified to contain a library. The
#                       build trees are never read again and are ~96% of
#                       what a built third_party/ costs (libint alone is
#                       ~7.7 GB against a 286 MB install). Costs a cold
#                       rebuild next time; ccache absorbs it when
#                       base_dir/hash_dir are set (CLAUDE.md 3). A dep
#                       whose install did not land is never pruned.
#   WITH_OPENBLAS=1     Also build a vendored OpenBLAS + LAPACK +
#                       LAPACKE into third_party/openblas/install/.
#                       Default OFF — most systems already have *some*
#                       BLAS (system OpenBLAS / MKL / Apple
#                       Accelerate). The vendored build is for HPC /
#                       no-sudo / reproducibility-sensitive setups,
#                       and for boxes where only reference netlib BLAS
#                       is available (which doesn't beat Eigen-generic
#                       at SCF matrix sizes). See scripts/build_openblas.sh
#                       for the rationale + per-distro alternatives.
#
#   VIBEQC_LIBINT_MAX_AM=N_N_N
#                       libint's max angular momentum per derivative
#                       order (0/1/2), each part 1..6. Default 5_4_4:
#                       analytic Hessians on g-function basis sets,
#                       ~1.5-2 h to build. 5_4_3 is the pre-2026-08-06
#                       recipe — ~30 min, f-function Hessians only.
#                       6_5_4 adds i-function (L = 6) energies for
#                       cc-pV6Z-class work, at a longer build again.
#                       No computed number changes either way; the spec
#                       only decides which basis sets each derivative
#                       order accepts. install.sh / update.sh /
#                       update_native_deps.sh expose this as
#                       --libint-max-am; see scripts/_libint_max_am.sh
#                       for the full rationale and the L = 6 ceiling.
#                       Changing it registers as native-dep drift, so
#                       libint rebuilds on the next update rather than
#                       silently staying stale.
#
#   VIBEQC_SKIP_PREFLIGHT=1
#                       Skip the preflight check. Use only if you know
#                       what you're doing — the per-build-script
#                       error messages are less helpful than a single
#                       preflight report.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Suppress the helper's minimal cmake+ninja check — we run our own
# more comprehensive preflight just below, and want its consolidated
# report (libomp, BLAS, python version, headers, …) to be the one the
# user sees, not the helper's two-line subset. Children build_*.sh
# inherit VIBEQC_TOOLING_CHECKED=1 from this export, so the helper
# also no-ops for them after our preflight has passed.
export VIBEQC_TOOLING_CHECKED=1

# Build-pressure safety (Linux): nice/ionice re-exec + memory-safe
# CMAKE_BUILD_PARALLEL_LEVEL cap. The child build_*.sh scripts each source
# the same helper, which is idempotent — only the outermost caller re-execs.
# Running this script standalone (without update.sh wrapping it) is also
# safe: this is the entry point that applies the cap.
# shellcheck source=_safe_build_env.sh
. "$SCRIPT_DIR/_safe_build_env.sh"

# Serialize against any other native-dependency build on this checkout.
# A no-op when invoked via install.sh / update.sh (they already hold the
# lock); when setup_native_deps.sh is run directly it takes the lock itself,
# so two direct runs — or a direct run racing an install.sh — can't clobber
# the shared third_party/*/build trees (the 2026-06-05 compute-host-b collision).
# Rationale + edge cases in scripts/_build_lock.sh.
# shellcheck source=_build_lock.sh
. "$SCRIPT_DIR/_build_lock.sh"
vibeqc_acquire_build_lock


# ---------------------------------------------------------------------------
# Preflight: verify every build prerequisite exists. Bail with one
# consolidated report (and the per-distro install command) if anything
# is missing — much friendlier than letting one of the build_*.sh
# scripts fail mid-flight with a CMake error nobody can decode.
# ---------------------------------------------------------------------------

UNAME_S="$(uname -s)"

# Each entry of MISSING is a human-readable label of something that's
# absent. Each entry of WARNINGS is a non-fatal note we'll print at
# the end of the preflight (e.g. "you have netlib BLAS, consider
# OpenBLAS").
MISSING=()
WARNINGS=()

_check_command() {
    # _check_command CMD LABEL
    # Adds LABEL to MISSING if CMD is not on $PATH.
    if ! command -v "$1" >/dev/null 2>&1; then
        MISSING+=("$2")
    fi
}

_check_header_any() {
    # _check_header_any LABEL HEADER_PATH...
    # Adds LABEL to MISSING unless at least one of the listed header
    # paths exists. Used for "is the dev package for X installed?"
    # checks where the canonical header location differs across
    # distros / Homebrew prefixes.
    local label="$1"; shift
    for h in "$@"; do
        if [ -e "$h" ]; then
            return 0
        fi
    done
    MISSING+=("$label")
}

_check_pkgconfig() {
    # _check_pkgconfig LABEL PKGNAME [PKGNAME...]
    # Adds LABEL to MISSING unless pkg-config can resolve at least one
    # of the named packages.
    local label="$1"; shift
    if command -v pkg-config >/dev/null 2>&1; then
        for pkg in "$@"; do
            if pkg-config --exists "$pkg" 2>/dev/null; then
                return 0
            fi
        done
    fi
    MISSING+=("$label")
}

_check_eigen() {
    # Eigen probe — pkg-config is authoritative when available
    # (brew installs eigen3.pc at /opt/homebrew/opt/eigen/share/pkgconfig
    # which `pkg-config eigen3` resolves; Linux distros all ship eigen3.pc
    # too via libeigen3-dev / eigen3-devel / eigen). Falls back to
    # known header paths when pkg-config isn't present or doesn't know
    # the package — covers custom prefix installs and stripped-down
    # toolchains.
    if command -v pkg-config >/dev/null 2>&1 \
            && pkg-config --exists eigen3 2>/dev/null; then
        return 0
    fi
    local h
    for h in \
            /opt/homebrew/include/eigen3/Eigen/Core \
            /opt/homebrew/opt/eigen/include/eigen3/Eigen/Core \
            /usr/local/include/eigen3/Eigen/Core \
            /usr/local/opt/eigen/include/eigen3/Eigen/Core \
            /usr/include/eigen3/Eigen/Core \
            /usr/include/Eigen/Core; do
        [ -e "$h" ] && return 0
    done
    MISSING+=("Eigen headers (eigen / libeigen3-dev / eigen3-devel; pkg-config eigen3 also tried)")
    # Accumulate-and-continue, like every other _check_* helper — NEVER
    # return non-zero here. The preflight calls `_check_eigen` *bare* under
    # `set -euo pipefail`, so a non-zero return aborted the entire script the
    # moment Eigen was missing, BEFORE the consolidated "missing prerequisites"
    # report below could print (it just died after "checking build
    # prerequisites…" with exit 1 — the exact opposite of this preflight's
    # whole purpose). Return 0 so the report lists Eigen with everything else.
    return 0
}

_cmake_version_ok() {
    # Returns 0 if cmake is on PATH and ≥3.20.
    local v
    if ! command -v cmake >/dev/null 2>&1; then
        return 1
    fi
    v="$(cmake --version 2>/dev/null | awk '/cmake version/ {print $3; exit}')"
    [ -n "$v" ] || return 1
    # Compare major.minor numerically.
    local major minor
    major="${v%%.*}"
    minor="${v#*.}"
    minor="${minor%%.*}"
    if [ "$major" -gt 3 ]; then
        return 0
    fi
    if [ "$major" -eq 3 ] && [ "$minor" -ge 20 ]; then
        return 0
    fi
    return 1
}

_have_cxx_compiler() {
    # Honor user's $CXX first; otherwise look for the usual suspects.
    if [ -n "${CXX:-}" ] && command -v "$CXX" >/dev/null 2>&1; then
        return 0
    fi
    for c in c++ clang++ g++; do
        command -v "$c" >/dev/null 2>&1 && return 0
    done
    return 1
}

_python_version_ok() {
    # Returns 0 if python3 is on PATH and ≥3.11 (vibe-qc's minimum
    # per pyproject.toml: requires-python = ">=3.11"). Also sets
    # PYTHON_VERSION_FOUND for use in the diagnostic message.
    PYTHON_VERSION_FOUND=""
    if ! command -v python3 >/dev/null 2>&1; then
        return 1
    fi
    PYTHON_VERSION_FOUND="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")"
    [ -n "$PYTHON_VERSION_FOUND" ] || return 1
    local major minor
    major="${PYTHON_VERSION_FOUND%%.*}"
    minor="${PYTHON_VERSION_FOUND#*.}"
    if [ "$major" -gt 3 ]; then
        return 0
    fi
    if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
        return 0
    fi
    return 1
}

_preflight_macos() {
    if ! command -v brew >/dev/null 2>&1; then
        cat >&2 <<'EOF'

Preflight: Homebrew not found.

vibe-qc's macOS build path uses Homebrew formulae (boost, eigen, gmp,
libomp, plus cmake / ninja / pkg-config / git as toolchain). Install
Homebrew first:

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Then re-run this script.
EOF
        exit 1
    fi

    local brew_prefix
    brew_prefix="$(brew --prefix)"

    _have_cxx_compiler || MISSING+=("C++17 compiler (Xcode command line tools — xcode-select --install)")
    _cmake_version_ok  || MISSING+=("cmake ≥3.20")
    _check_command ninja       "ninja"
    _check_command make        "make"
    _check_command git         "git"
    _check_command curl        "curl"
    _check_command tar         "tar"
    _check_command pkg-config  "pkg-config"
    # Python ≥3.11 — Apple's stock /usr/bin/python3 is currently 3.9.x
    # (Xcode Command Line Tools stub), which doesn't meet vibe-qc's
    # pyproject minimum. Even if a newer python3 is installed under
    # /opt/homebrew/bin, it has to actually win on PATH — without
    # `eval "$(/opt/homebrew/bin/brew shellenv)"` in your shell rc,
    # /usr/bin/python3 may still be first.
    if ! _python_version_ok; then
        if [ -n "${PYTHON_VERSION_FOUND:-}" ]; then
            MISSING+=("python ≥3.11 on PATH (found python3 = $PYTHON_VERSION_FOUND — likely Apple's stub /usr/bin/python3)")
        else
            MISSING+=("python3 ≥3.11")
        fi
    fi

    # libomp — AppleClang doesn't ship OpenMP, several vendored deps
    # need it. Check via brew first (gives the user the most actionable
    # error), then fall back to a header lookup so a manual install in
    # /usr/local also satisfies us.
    if ! brew list libomp >/dev/null 2>&1 && \
            [ ! -e "$brew_prefix/opt/libomp/include/omp.h" ] && \
            [ ! -e /usr/local/include/omp.h ]; then
        MISSING+=("libomp (OpenMP for AppleClang)")
    fi

    # Header deps for libint's code generator.
    _check_header_any "Boost headers" \
        "$brew_prefix/include/boost/version.hpp" \
        /usr/local/include/boost/version.hpp \
        /opt/homebrew/include/boost/version.hpp

    _check_eigen

    _check_header_any "GMP + C++ bindings (gmpxx.h)" \
        "$brew_prefix/include/gmpxx.h" \
        /usr/local/include/gmpxx.h \
        /opt/homebrew/include/gmpxx.h

    # BLAS: Apple Accelerate is a framework that ships with macOS.
    # The framework lives at /System/Library/Frameworks/Accelerate.framework;
    # if that's gone, the user is on a stripped-down image and needs
    # the vendored OpenBLAS path.
    if [ ! -d /System/Library/Frameworks/Accelerate.framework ]; then
        WARNINGS+=("Apple Accelerate framework not found at /System/Library/Frameworks. Build with WITH_OPENBLAS=1 to vendor OpenBLAS, or check your macOS install.")
    fi

    if [ ${#MISSING[@]} -gt 0 ]; then
        cat >&2 <<EOF

Preflight: the following build prerequisites are missing on this macOS box:

EOF
        for m in "${MISSING[@]}"; do echo "    - $m" >&2; done
        cat >&2 <<'EOF'

Install everything we need with Homebrew:

    brew install cmake ninja pkg-config libomp boost eigen gmp git python@3.14

Apple's stock /usr/bin/python3 is currently 3.9.x and does not meet
vibe-qc's ≥3.11 minimum. Homebrew's python@3.14 is the recommended
runtime. After installing, make sure Homebrew's bin wins on PATH so
`python3` resolves to the new install — `brew install` normally
links python@3.14 as /opt/homebrew/bin/python3, but only the
standard `brew shellenv` config puts /opt/homebrew/bin ahead of
/usr/bin. If `python3 --version` still shows 3.9.x after the brew
install:

    eval "$(/opt/homebrew/bin/brew shellenv)"          # current shell
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc   # persist
    brew link --overwrite --force python@3.14          # if still not linked

Optional but recommended for the vendored OpenBLAS escape hatch
(WITH_OPENBLAS=1):

    brew install gcc      # provides gfortran

If Xcode command line tools are missing (the C++17 compiler):

    xcode-select --install

Re-run this script after installing.
EOF
        exit 1
    fi
}

_preflight_linux() {
    _have_cxx_compiler || MISSING+=("C++17 compiler (g++ / clang++)")
    _cmake_version_ok  || MISSING+=("cmake ≥3.20")
    _check_command ninja       "ninja / ninja-build"
    _check_command make        "make"
    _check_command git         "git"
    _check_command curl        "curl"
    _check_command tar         "tar"
    _check_command pkg-config  "pkg-config"

    # Python ≥3.11. Arch / Fedora rolling are usually fine; Debian
    # stable and Ubuntu LTS may carry an older default — distro
    # install commands below pin python3 explicitly.
    if ! _python_version_ok; then
        if [ -n "${PYTHON_VERSION_FOUND:-}" ]; then
            MISSING+=("python ≥3.11 on PATH (found python3 = $PYTHON_VERSION_FOUND — upgrade or use pyenv / deadsnakes)")
        else
            MISSING+=("python3 ≥3.11")
        fi
    fi

    # python3-venv check: on Debian/Ubuntu `python3-venv` is a separate
    # package; without it `python3 -m venv` fails with a helpful but
    # late message. Catch it here.
    if command -v python3 >/dev/null 2>&1; then
        if ! python3 -c "import venv" >/dev/null 2>&1; then
            MISSING+=("python3-venv (python3 -m venv module)")
        fi
    fi

    # Boost (header-only) — common install locations, plus the active
    # conda/mamba prefix (compute-cluster cluster Miniforge — docs/cluster_setup.md).
    # The ${CONDA_PREFIX:-/nonexistent} guard makes the extra entry a
    # harmless non-match on non-conda builds.
    _check_header_any "Boost headers" \
        /usr/include/boost/version.hpp \
        /usr/local/include/boost/version.hpp \
        "${CONDA_PREFIX:-/nonexistent}/include/boost/version.hpp"

    _check_eigen

    # GMP with C++ bindings. Multi-arch (Debian) puts gmpxx.h under
    # /usr/include/x86_64-linux-gnu/ — sweep two levels deep.
    if ! find /usr/include /usr/local/include ${CONDA_PREFIX:+"$CONDA_PREFIX/include"} \
            -maxdepth 3 -name gmpxx.h 2>/dev/null | grep -q .; then
        MISSING+=("GMP + C++ bindings (gmpxx.h)")
    fi

    # BLAS detection. Three buckets:
    #   1. OpenBLAS / MKL present → great, no action.
    #   2. Only reference netlib BLAS present → soft warning (the
    #      build will work but is slow; suggest OpenBLAS).
    #   3. Nothing present → hard fail (CMake's FindBLAS will either
    #      fail or fall back to Eigen-generic anyway, but the user
    #      should know upfront).
    local has_openblas="" has_mkl="" has_netlib=""
    for d in /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/lib ${CONDA_PREFIX:+"$CONDA_PREFIX/lib"}; do
        [ -d "$d" ] || continue
        for f in "$d"/libopenblas.so* "$d"/libopenblas.so; do
            [ -e "$f" ] && has_openblas="$f" && break
        done
        for f in "$d"/libmkl_rt.so*; do
            [ -e "$f" ] && has_mkl="$f" && break
        done
        for f in "$d"/libblas.so.3 "$d"/libblas.so; do
            [ -e "$f" ] && has_netlib="$f" && break
        done
        [ -n "$has_openblas$has_mkl" ] && break
    done

    if [ "${WITH_OPENBLAS:-0}" = "1" ]; then
        # User opted into vendored OpenBLAS — gfortran is required.
        if ! command -v gfortran >/dev/null 2>&1; then
            MISSING+=("gfortran (WITH_OPENBLAS=1 needs it to build OpenBLAS+LAPACK)")
        fi
    elif [ -z "$has_openblas$has_mkl" ]; then
        if [ -n "$has_netlib" ]; then
            WARNINGS+=("BLAS: only reference netlib BLAS detected at $has_netlib. Functional but ~no win over Eigen-generic at SCF size. Install OpenBLAS (see below) or rerun with WITH_OPENBLAS=1.")
        else
            MISSING+=("BLAS+LAPACK (no libopenblas / libmkl_rt / libblas found)")
        fi
    fi

    if [ ${#MISSING[@]} -gt 0 ]; then
        cat >&2 <<EOF

Preflight: the following build prerequisites are missing on this Linux box:

EOF
        for m in "${MISSING[@]}"; do echo "    - $m" >&2; done
        cat >&2 <<'EOF'

Install everything we need (pick whichever fits your distro):

    Arch / Manjaro:
        sudo pacman -S base-devel cmake ninja pkg-config git curl \
            gmp eigen boost python blas-openblas
        # Optional (WITH_OPENBLAS=1): sudo pacman -S gcc-fortran

    Debian / Ubuntu:
        sudo apt update
        sudo apt install \
            build-essential cmake ninja-build pkg-config git curl \
            libeigen3-dev libboost-dev libgmp-dev libgmpxx4ldbl \
            libopenblas-dev liblapacke-dev \
            python3 python3-dev python3-venv
        # Optional (WITH_OPENBLAS=1): sudo apt install gfortran

    Fedora / RHEL:
        sudo dnf install \
            @development-tools cmake ninja-build pkgconfig git curl \
            eigen3-devel boost-devel gmp-devel gmp-c++ \
            openblas-devel lapack-devel \
            python3 python3-devel
        # Optional (WITH_OPENBLAS=1): sudo dnf install gcc-gfortran

Re-run this script after installing.
EOF
        exit 1
    fi
}

if [ "${VIBEQC_SKIP_PREFLIGHT:-0}" != "1" ] && [ "${VIBEQC_FETCH_ONLY:-0}" != "1" ]; then
    echo "==> Preflight: checking build prerequisites..."
    # Robustness backstop (the 2026-06-05 Eigen silent-abort). The _check_*
    # helpers report missing deps by appending to MISSING[] and must each
    # return 0 so the loop reaches the consolidated report below. If one ever
    # ends on a non-zero statement (a bare probe, a stray `return 1`), `set -e`
    # would abort the whole script *here* — after "checking build
    # prerequisites..." but before the report — turning a clear "you're missing
    # X" into a silent exit 1. Relax errexit just around the dispatch so an
    # accidental non-zero is swallowed; the report's own explicit `exit 1`
    # (when MISSING is non-empty) still fires, which is exactly what we want.
    set +e
    case "$UNAME_S" in
        Darwin) _preflight_macos ;;
        Linux)  _preflight_linux ;;
        *)
            echo "Preflight: unsupported platform '$UNAME_S' — skipping." >&2
            echo "vibe-qc is tested on macOS and Linux. Proceeding anyway." >&2
            ;;
    esac
    set -e
    echo "    all prerequisites present."

    if [ ${#WARNINGS[@]} -gt 0 ]; then
        echo
        for w in "${WARNINGS[@]}"; do
            echo "    note: $w"
        done
    fi
    echo
fi


# ---------------------------------------------------------------------------
# Reconcile stamped recipe drift before idempotent build scripts can skip it.
# ---------------------------------------------------------------------------
# A plain build_*.sh invocation short-circuits when install/ exists. Writing a
# fresh stamp after such a skip used to relabel an old binary as if the current
# recipe had built it. With an existing stamp, rebuild every drifted target
# first. With legacy install trees and no stamp, keep the state explicitly
# unstamped rather than making an unverifiable provenance claim.
VIBEQC_STAMP_WRITE_SAFE=1
if [ "${VIBEQC_FETCH_ONLY:-0}" != "1" ]; then
    # shellcheck source=_native_stamp.sh
    . "$SCRIPT_DIR/_native_stamp.sh"
    _vibeqc_any_install=0
    for _vibeqc_existing_dep in libint libxc spglib fftw libecpint openblas; do
        if [ -d "third_party/$_vibeqc_existing_dep/install" ]; then
            _vibeqc_any_install=1
            break
        fi
    done

    if [ -f "$VIBEQC_STAMP_PATH" ]; then
        _vibeqc_drifted=()
        while IFS= read -r _vibeqc_drifted_dep; do
            [ -n "$_vibeqc_drifted_dep" ] && \
                _vibeqc_drifted+=("$_vibeqc_drifted_dep")
        done < <(vibeqc_stamp_drifted_deps)
        if [ ${#_vibeqc_drifted[@]} -gt 0 ]; then
            echo "==> Native recipe/artifact drift detected: ${_vibeqc_drifted[*]}"
            echo "    Rebuilding those dependencies before refreshing provenance..."
            "$SCRIPT_DIR/update_native_deps.sh"
        fi
    elif [ "$_vibeqc_any_install" = "1" ]; then
        VIBEQC_STAMP_WRITE_SAFE=0
        echo "==> Legacy native install trees found without a build stamp."
        echo "    Their recipe provenance cannot be certified; the stamp will remain absent."
        echo "    Run ./scripts/update.sh --rebuild-native-deps to rebuild and certify them."
    fi
    unset _vibeqc_any_install _vibeqc_existing_dep _vibeqc_drifted_dep
fi


# ---------------------------------------------------------------------------
# Self-heal wiped native libraries before the (idempotent-on-existence)
# build_<dep>.sh calls below.
# ---------------------------------------------------------------------------
# Each build_<dep>.sh short-circuits when its CMake *config* file
# (lib/cmake/<Dep>/...Config.cmake) is present — but that file, the headers,
# and the build stamp all survive even when the actual shared library
# (libint2.so / libxc.so / libsymspg.so / ...) is deleted out from under a
# built install/ tree. The result is a venv that throws "libint2.so: cannot
# open shared object file" at `import vibeqc` yet never self-heals: git is
# current, the stamp says "built", the config file still exists, so the
# rebuild is short-circuited at every layer. Before this guard the only
# recovery was the magic `update.sh --rebuild-native-deps` force flag (the
# 2026-06-13 compute-host-b/compute-host-e fleet incident).
#
# For any core dep whose install/ tree exists but whose primary library is
# gone, wipe just that dep's build+install trees so the build_<dep>.sh below
# stops short-circuiting and rebuilds it. The line is printed loudly so the
# repair shows up in `vq admin update` logs. openblas is intentionally
# excluded: it is opt-in (WITH_OPENBLAS) and its rebuild is gated below, so
# healing it here could wipe an install we then don't rebuild.
#
# Skipped under VIBEQC_FETCH_ONLY (we are staging sources, not repairing a
# built tree). A no-op after update.sh --rebuild-native-deps (install/ trees
# were already wiped by the caller — nothing left to detect).
if [ "${VIBEQC_FETCH_ONLY:-0}" != "1" ]; then
    # shellcheck source=_native_stamp.sh
    . "$SCRIPT_DIR/_native_stamp.sh"
    for _heal_dep in libint libxc spglib fftw libecpint; do
        [ -d "third_party/$_heal_dep/install" ] || continue
        if ! _vibeqc_dep_artifact_present "$_heal_dep"; then
            _heal_lib="$(_vibeqc_dep_lib_basename "$_heal_dep" 2>/dev/null || echo "$_heal_dep")"
            echo "==> native lib ${_heal_lib} missing for ${_heal_dep} (install tree present, shared library gone) → forcing rebuild of ${_heal_dep}"
            rm -rf "third_party/$_heal_dep/build" "third_party/$_heal_dep/install"
        fi
    done
    unset _heal_dep _heal_lib
fi

# ---------------------------------------------------------------------------
# Build the vendored native deps in dependency order.
# ---------------------------------------------------------------------------

"$SCRIPT_DIR/build_libint.sh"
"$SCRIPT_DIR/build_libxc.sh"
"$SCRIPT_DIR/build_spglib.sh"
"$SCRIPT_DIR/build_fftw.sh"
"$SCRIPT_DIR/build_libecpint.sh"

# Fetch-only mode for offline-HPC pre-staging (docs/cluster_setup.md): the
# build_*.sh above each cloned/downloaded their source and exited before
# compiling. Stop here too -- setup_basis_library.sh reads the (not-yet-
# built) libint install, and the stamp/openblas steps assume a real build.
# Re-run WITHOUT VIBEQC_FETCH_ONLY on a toolchain host to compile offline.
if [ "${VIBEQC_FETCH_ONLY:-0}" = "1" ]; then
    echo
    echo "VIBEQC_FETCH_ONLY=1 -- native-dep sources staged under third_party/."
    exit 0
fi

# Basis-library populate — writes to build/basis_library/ by default
# (a gitignored build-output directory). The working tree stays clean
# across pip install -e . and fleet deploys.
#
# The committed python/vibeqc/basis_library/basis/ ships inside the
# wheel and serves as the fallback at runtime when no build output
# exists. To regenerate the committed copy after a custom/ or libint
# change, run the script explicitly with the committed target:
#   ./scripts/setup_basis_library.sh --target-dir python/vibeqc/basis_library
"$SCRIPT_DIR/setup_basis_library.sh"

if [ "${WITH_OPENBLAS:-0}" = "1" ]; then
    echo
    echo "WITH_OPENBLAS=1 set — building vendored OpenBLAS..."
    "$SCRIPT_DIR/build_openblas.sh"
fi


# ---------------------------------------------------------------------------
# Post-build BLAS quality recap (informational; no action taken)
# ---------------------------------------------------------------------------
#
# Preflight has already validated BLAS is present; this final recap
# tells the user which one they ended up with so they can record it
# alongside the build artefact. vibe-qc's runtime banner carries the
# same info (`blas Accelerate` / `blas OpenBLAS +LAPACKE` / etc.), so
# this is just a heads-up at setup time.

if [ "${WITH_OPENBLAS:-0}" = "1" ]; then
    echo
    echo "BLAS backend: vendored OpenBLAS at third_party/openblas/install/"
    echo "  vibe-qc's CMake will pick it up automatically."
elif [ "$UNAME_S" = "Darwin" ]; then
    echo
    echo "BLAS backend: Apple Accelerate (system framework). No action needed."
elif [ "$UNAME_S" = "Linux" ]; then
    HAS_OPENBLAS=""
    HAS_MKL=""
    HAS_NETLIB=""
    for d in /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/lib ${CONDA_PREFIX:+"$CONDA_PREFIX/lib"}; do
        [ -d "$d" ] || continue
        for f in "$d"/libopenblas.so* "$d"/libopenblas.so; do
            [ -e "$f" ] && HAS_OPENBLAS="$f" && break
        done
        for f in "$d"/libmkl_rt.so*; do
            [ -e "$f" ] && HAS_MKL="$f" && break
        done
        for f in "$d"/libblas.so.3 "$d"/libblas.so; do
            [ -e "$f" ] && HAS_NETLIB="$f" && break
        done
        [ -n "$HAS_OPENBLAS$HAS_MKL" ] && break
    done

    echo
    if [ -n "$HAS_MKL" ]; then
        echo "BLAS backend detected: Intel MKL at $HAS_MKL"
        echo "  Expected vibe-qc SCF perf: good. No action needed."
    elif [ -n "$HAS_OPENBLAS" ]; then
        echo "BLAS backend detected: OpenBLAS at $HAS_OPENBLAS"
        echo "  Expected vibe-qc SCF perf: good. No action needed."
    elif [ -n "$HAS_NETLIB" ]; then
        cat <<EOF
BLAS backend detected: reference netlib BLAS at $HAS_NETLIB
  Expected vibe-qc SCF perf: ~no win over Eigen-generic. Netlib
  is single-threaded reference Fortran; at SCF-sized matrices it
  doesn't beat the C++ generic path. For real perf, install an
  optimised BLAS:

    Arch / Manjaro:  sudo pacman -S blas-openblas
    Debian / Ubuntu: sudo apt install libopenblas-dev liblapacke-dev
    Fedora / RHEL:   sudo dnf install openblas-devel lapack-devel

  Or build the vendored OpenBLAS into this checkout (no sudo
  needed, needs gfortran):

    WITH_OPENBLAS=1 ./scripts/setup_native_deps.sh

  Then re-run pip install -e . — CMake will auto-pick whichever
  optimised BLAS is available.
EOF
    fi
fi


# ---------------------------------------------------------------------------
# Record provenance of every build_*.sh in a stamp file. Reader scripts
# (install.sh / update.sh / doctor.sh) compare against the
# currently-checked-out build_*.sh to detect drift between an
# install/ tree and the recipe that produced it. See
# scripts/_native_stamp.sh for the format + rationale.
# ---------------------------------------------------------------------------
# shellcheck source=_native_stamp.sh
. "$SCRIPT_DIR/_native_stamp.sh"
if [ "$VIBEQC_STAMP_WRITE_SAFE" = "1" ]; then
    vibeqc_stamp_write
else
    echo "==> Skipped $VIBEQC_STAMP_PATH (legacy binaries were not rebuilt)."
fi

echo
echo "==========================================================="
echo " All native dependencies built. Next step:"
echo
echo "   ./scripts/install.sh                  # fresh install (creates .venv)"
echo "   ./scripts/install.sh --branch NAME    # or pin a non-default branch"
echo
echo " Or by hand:"
echo "   python3 -m venv .venv"
echo "   .venv/bin/pip install -e '.[test]'"
echo
echo "==========================================================="
