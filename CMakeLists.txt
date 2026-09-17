cmake_minimum_required(VERSION 3.20)
project(vibeqc LANGUAGES CXX VERSION 0.0.1)

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Reuse unchanged translation units when ccache is installed. This is optional
# for developer/fleet hosts and enabled in CI; CMake falls back to invoking the
# compiler directly when ccache is absent.
option(VIBEQC_USE_CCACHE "Use ccache as the C++ compiler launcher" ON)
if(VIBEQC_USE_CCACHE)
    find_program(CCACHE_EXECUTABLE ccache)
    if(CCACHE_EXECUTABLE)
        message(STATUS "Using ccache: ${CCACHE_EXECUTABLE}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_EXECUTABLE}")
    else()
        message(STATUS "ccache not found; compiling without a compiler cache")
    endif()
endif()

# Use modern FindPython inside pybind11 (silences CMP0148 deprecation warning
# from pybind11's FindPythonLibsNew.cmake).
set(PYBIND11_FINDPYTHON ON)

# Prefer source-built dependencies under third_party/<name>/install/ over any
# system install. Each is populated by scripts/setup_native_deps.sh (or the
# matching per-dep script: build_libint.sh, build_libxc.sh, build_spglib.sh,
# build_fftw.sh, build_libecpint.sh).
#
# Vendoring everything we link against eliminates the entire "which version
# of libxc / spglib does THIS system have?" failure mode and means every
# checkout produces an identical native ABI. CMake's list(PREPEND) rule
# guarantees the third_party install wins over any system find_package
# result. Missing third_party/<dep>/install/ silently falls back to system
# discovery, so users who want to point at their own builds still can.
#
# CI / release builds that need the vendored-ABI guarantee can flip the
# fallback into a hard error by setting VIBEQC_REQUIRE_VENDORED (either
# -DVIBEQC_REQUIRE_VENDORED=ON at configure time or the matching env var,
# which CI tends to find more ergonomic). The check covers the five
# always-vendored deps; openblas is opt-in (WITH_OPENBLAS=1 to
# setup_native_deps.sh) and stays excluded.
option(VIBEQC_REQUIRE_VENDORED
    "Fail configure if any always-vendored third_party/<dep>/install is missing"
    OFF)
if(NOT VIBEQC_REQUIRE_VENDORED AND DEFINED ENV{VIBEQC_REQUIRE_VENDORED})
    set(VIBEQC_REQUIRE_VENDORED $ENV{VIBEQC_REQUIRE_VENDORED})
endif()

set(_vibeqc_required_vendored libint libxc spglib fftw libecpint)
set(_vibeqc_missing_vendored "")
foreach(_dep IN ITEMS libint libxc spglib fftw libecpint openblas)
    set(_local "${CMAKE_SOURCE_DIR}/third_party/${_dep}/install")
    if(EXISTS "${_local}")
        message(STATUS "Using source-built ${_dep} at: ${_local}")
        list(PREPEND CMAKE_PREFIX_PATH "${_local}")
    elseif(_dep IN_LIST _vibeqc_required_vendored)
        list(APPEND _vibeqc_missing_vendored "${_dep}")
    endif()
endforeach()

if(VIBEQC_REQUIRE_VENDORED AND _vibeqc_missing_vendored)
    string(JOIN ", " _missing_joined ${_vibeqc_missing_vendored})
    message(FATAL_ERROR
        "VIBEQC_REQUIRE_VENDORED is set, but these always-vendored "
        "dependencies have no third_party/<dep>/install/: ${_missing_joined}. "
        "Run scripts/setup_native_deps.sh (or the matching per-dep "
        "build_*.sh) before re-configuring. To allow the system-library "
        "fallback for a one-off build, unset VIBEQC_REQUIRE_VENDORED.")
endif()

# Auto-detect Homebrew prefix on macOS so find_package() picks up Eigen3 and
# libomp without the user needing to set CMAKE_PREFIX_PATH.
if(APPLE)
    find_program(HOMEBREW_EXECUTABLE brew)
    if(HOMEBREW_EXECUTABLE)
        execute_process(
            COMMAND ${HOMEBREW_EXECUTABLE} --prefix
            OUTPUT_VARIABLE HOMEBREW_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        message(STATUS "Homebrew prefix: ${HOMEBREW_PREFIX}")
        list(APPEND CMAKE_PREFIX_PATH
            "${HOMEBREW_PREFIX}"
            "${HOMEBREW_PREFIX}/opt/libomp")
    endif()
endif()

# -----------------------------------------------------------------------------
# Core C++ module
# -----------------------------------------------------------------------------
include(cpp/cmake/OpenTrustRegion.cmake)
add_subdirectory(cpp)
