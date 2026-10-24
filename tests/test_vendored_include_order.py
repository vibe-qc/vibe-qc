"""Vendored dependency headers win the include search (vibe-qc #461).

libint's installed package links ``Boost::headers``. On macOS its include
directory is the whole Homebrew prefix (``/opt/homebrew/include``), and it
reached the compile line ahead of the vendored libxc, spglib, fftw and
libecpint include directories. ``#include <libecpint/ecpint.hpp>`` then found
Homebrew's unpatched 2021 header while ``libecpint/diagnostics.hpp``, which
only the vendored install has, came from third_party: ``cpp/src/ecp.cpp``
failed with "no member named 'shell_pair_negligible'". ``<xc.h>`` resolved to
Homebrew's libxc 7.1.2 against the vendored 7.0.0 library and compiled
without a word.

``cpp/cmake/VendoredIncludeDirectories.cmake`` collects the vendored include
directories so ``cpp/CMakeLists.txt`` can prepend them. These tests rebuild
that situation in a throwaway CMake project: a vendored header and a
poisoned copy under a shared prefix that the libint-like target pulls in
first. The control proves the project really shadows; the check proves the
prepended directories cure it.

``cpp/cmake/RequireCurrentLibecpintPatch.cmake`` refuses a vendored libecpint
whose recorded patch is older than ``scripts/build_libecpint.sh``'s, since a
plain ``pip install -e .`` relinks that install without rerunning the script
(#460).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CMAKE_MODULES = REPO_ROOT / "cpp" / "cmake"
CMAKE = shutil.which("cmake")

pytestmark = pytest.mark.skipif(CMAKE is None, reason="cmake is not on PATH")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _cmake(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [CMAKE, *args], capture_output=True, text=True, timeout=300, check=False
    )


# --- include order --------------------------------------------------------

_SHADOW_PROJECT = """\
cmake_minimum_required(VERSION 3.20)
project(vibeqc_include_order_probe LANGUAGES CXX)

# The shapes of the real packages: a header-only Boost whose include
# directory is a whole shared prefix, linked by libint's C++ target through
# a generator expression, ahead of libxc and libecpint (whose include
# directory carries a trailing slash and a nested libecpint/ entry).
add_library(Boost::headers INTERFACE IMPORTED)
set_target_properties(Boost::headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SHARED_PREFIX}/include")
add_library(Libint2::int-shared INTERFACE IMPORTED)
set_target_properties(Libint2::int-shared PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/third_party/libint/install/include")
add_library(Libint2::cxx INTERFACE IMPORTED)
set_target_properties(Libint2::cxx PROPERTIES INTERFACE_LINK_LIBRARIES
    "Libint2::int-shared;$<$<TARGET_EXISTS:Boost::headers>:Boost::headers>")
add_library(Libxc::xc INTERFACE IMPORTED)
set_target_properties(Libxc::xc PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/third_party/libxc/install/include/")
add_library(ECPINT::ecpint INTERFACE IMPORTED)
set_target_properties(ECPINT::ecpint PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    "${CMAKE_SOURCE_DIR}/third_party/libecpint/install/include/libecpint;${CMAKE_SOURCE_DIR}/third_party/libecpint/install/include")

add_library(probe STATIC probe.cpp)
target_link_libraries(probe PUBLIC Libint2::cxx Libxc::xc ECPINT::ecpint)

include("${VIBEQC_CMAKE_MODULES}/VendoredIncludeDirectories.cmake")
vibeqc_vendored_include_directories(_vendored Libint2::cxx Libxc::xc ECPINT::ecpint)
message(STATUS "VENDORED_INCLUDES=${_vendored}")
if(PREFER_VENDORED)
    target_include_directories(probe SYSTEM BEFORE PUBLIC ${_vendored})
endif()
"""

_PROBE_SOURCE = """\
#include <libecpint/ecpint.hpp>
#include <xc.h>
static_assert(VIBEQC_TEST_VENDORED_ECPINT && VIBEQC_TEST_VENDORED_XC,
              "vendored headers were not the ones found");
int vibeqc_include_order_probe() { return 0; }
"""


@pytest.fixture
def shadow_project(tmp_path: Path) -> Path:
    if shutil.which("c++") is None and shutil.which("g++") is None:
        pytest.skip("no C++ compiler on PATH")
    src = tmp_path / "src"
    vendored = src / "third_party"
    _write(vendored / "libint/install/include/libint2.h", "#pragma once\n")
    _write(
        vendored / "libxc/install/include/xc.h",
        "#pragma once\n#define VIBEQC_TEST_VENDORED_XC 1\n",
    )
    _write(
        vendored / "libecpint/install/include/libecpint/ecpint.hpp",
        "#pragma once\n#define VIBEQC_TEST_VENDORED_ECPINT 1\n",
    )
    shared = tmp_path / "shared-prefix" / "include"
    _write(shared / "xc.h", '#error "shared-prefix xc.h shadowed the vendored one"\n')
    _write(
        shared / "libecpint/ecpint.hpp",
        '#error "shared-prefix ecpint.hpp shadowed the vendored one"\n',
    )
    _write(src / "CMakeLists.txt", _SHADOW_PROJECT)
    _write(src / "probe.cpp", _PROBE_SOURCE)
    return src


def _configure_and_build(src: Path, build: Path, prefer_vendored: bool):
    generator = ["-G", "Ninja"] if shutil.which("ninja") else []
    configure = _cmake(
        "-S", str(src), "-B", str(build), *generator,
        f"-DSHARED_PREFIX={src.parent / 'shared-prefix'}",
        f"-DVIBEQC_CMAKE_MODULES={CMAKE_MODULES}",
        f"-DPREFER_VENDORED={'ON' if prefer_vendored else 'OFF'}",
    )
    assert configure.returncode == 0, configure.stdout + configure.stderr
    return configure, _cmake("--build", str(build))


def test_shared_prefix_shadows_without_the_prepended_directories(shadow_project):
    _, build = _configure_and_build(
        shadow_project, shadow_project.parent / "build-control", prefer_vendored=False
    )
    assert build.returncode != 0, "the probe must reproduce the shadowing"
    assert "shadowed the vendored one" in build.stdout + build.stderr


def test_prepended_vendored_directories_win(shadow_project):
    configure, build = _configure_and_build(
        shadow_project, shadow_project.parent / "build", prefer_vendored=True
    )
    assert build.returncode == 0, build.stdout + build.stderr

    match = re.search(r"VENDORED_INCLUDES=(.*)", configure.stdout)
    assert match, configure.stdout
    collected = match.group(1).split(";")
    root = f"{shadow_project}/third_party/"
    assert collected == [
        f"{root}libxc/install/include/",
        f"{root}libecpint/install/include/libecpint",
        f"{root}libecpint/install/include",
        f"{root}libint/install/include",
    ]


# --- libecpint patch marker -----------------------------------------------

_MARKER_PROJECT = """\
cmake_minimum_required(VERSION 3.20)
project(vibeqc_libecpint_marker_probe LANGUAGES NONE)
include("${VIBEQC_CMAKE_MODULES}/RequireCurrentLibecpintPatch.cmake")
vibeqc_require_current_libecpint_patch()
"""


def _marker_tree(tmp_path: Path, script: str, marker: str | None) -> Path:
    src = tmp_path / "src"
    _write(src / "CMakeLists.txt", _MARKER_PROJECT)
    _write(src / "scripts/build_libecpint.sh", script)
    install = src / "third_party/libecpint/install"
    install.mkdir(parents=True)
    if marker is not None:
        (install / ".vibeqc-radial-quadrature").write_text(marker + "\n")
    return src


def _configure(src: Path, require_vendored: bool) -> subprocess.CompletedProcess[str]:
    return _cmake(
        "-S", str(src), "-B", str(src.parent / "build"),
        f"-DVIBEQC_CMAKE_MODULES={CMAKE_MODULES}",
        f"-DVIBEQC_REQUIRE_VENDORED={'ON' if require_vendored else 'OFF'}",
    )


_SCRIPT = 'set -e\nRADIAL_PATCH_VERSION="radial-v2"\n'


def test_current_libecpint_patch_configures(tmp_path):
    result = _configure(_marker_tree(tmp_path, _SCRIPT, "radial-v2"), True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Vendored libecpint patch: radial-v2" in result.stdout


@pytest.mark.parametrize("marker", ["radial-v1", None], ids=["stale", "unmarked"])
def test_stale_libecpint_patch_fails_when_vendoring_is_required(tmp_path, marker):
    result = _configure(_marker_tree(tmp_path, _SCRIPT, marker), True)
    assert result.returncode != 0
    output = " ".join((result.stdout + result.stderr).split())
    assert "'radial-v2'" in output
    assert f"'{marker or '(no patch marker)'}'" in output
    assert "Run scripts/build_libecpint.sh" in output


def test_stale_libecpint_patch_warns_otherwise(tmp_path):
    result = _configure(_marker_tree(tmp_path, _SCRIPT, "radial-v1"), False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CMake Warning" in result.stderr
    assert "'radial-v1'" in " ".join(result.stderr.split())


def test_system_libecpint_is_not_checked(tmp_path):
    src = _marker_tree(tmp_path, _SCRIPT, "radial-v1")
    shutil.rmtree(src / "third_party")
    result = _configure(src, True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Vendored libecpint patch" not in result.stdout
    assert "carries libecpint patch" not in result.stderr


def test_real_build_script_marker_is_parsed(tmp_path):
    script = (REPO_ROOT / "scripts" / "build_libecpint.sh").read_text()
    match = re.search(r'^RADIAL_PATCH_VERSION="([^"]+)"$', script, re.MULTILINE)
    assert match, "build_libecpint.sh no longer defines RADIAL_PATCH_VERSION"
    result = _configure(_marker_tree(tmp_path, script, match.group(1)), True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Vendored libecpint patch: {match.group(1)}" in result.stdout
