"""Real configure/link outcomes for the shared libint admission gate (#271).

Tiny generated C API libraries are deliberately capability fixtures, not libint
implementations or numerical references. CMake compiles these artifacts and the
shipped witness against a selected imported target. Nothing executes the witness
or imports vibeqc here. The ordinary repository conftest does import vibeqc;
isolated build-only runs can use pytest --noconftest.

Run the same packet with VIBEQC_TEST_CMAKE=/path/to/cmake-3.20 to check the
minimum supported CMake, as well as the current CMake. A current-version pass
alone does not establish minimum-version compatibility.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import shutil
import subprocess

import pytest


MODULE = (Path(__file__).resolve().parents[1] / "cpp" / "cmake"
          / "RequireLibintOneBodyDerivatives.cmake")
OPERATORS = ("overlap", "kinetic", "elecpot")
SYMBOLS = tuple(f"libint2_{kind}_{op}1"
                for op in OPERATORS for kind in ("build", "need_memory"))
CMAKE = os.environ.get("VIBEQC_TEST_CMAKE", "cmake")
pytestmark = pytest.mark.skipif(shutil.which(CMAKE) is None,
                                reason="CMake is required for configure regressions")


def _command(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    # These tests select only their own package and compile one job at a time.
    env.pop("CMAKE_PREFIX_PATH", None)
    env.pop("CMAKE_TOOLCHAIN_FILE", None)
    build_flag = "-B" if "-B" in args else "--build"
    log_dir = Path(args[args.index(build_flag) + 1]) / "fixture-command-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ordinal = len(list(log_dir.glob("*-start.json")))
    stem = log_dir / f"command-{ordinal:03d}"
    stem.with_name(stem.name + "-start.json").write_text(json.dumps({
        "args": args, "CXXFLAGS": env.get("CXXFLAGS"),
        "CMAKE_BUILD_PARALLEL_LEVEL": env.get("CMAKE_BUILD_PARALLEL_LEVEL"),
    }, indent=2) + "\n")
    try:
        result = subprocess.run(args, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=60)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        stem.with_suffix(".log").write_text(output)
        stem.with_name(stem.name + "-result.json").write_text(
            json.dumps({"status": "timeout", "returncode": None}) + "\n")
        raise
    stem.with_suffix(".log").write_text(result.stdout)
    stem.with_name(stem.name + "-result.json").write_text(
        json.dumps({"status": "completed", "returncode": result.returncode}) + "\n")
    return result


def _provider(root: Path, *, support: int | None = 1, order: int | None = 1,
              missing: tuple[str, ...] = (), alias: bool = True) -> Path:
    """Build a fixture archive and a transitive Libint2::cxx imported target."""
    root.mkdir(parents=True, exist_ok=True)
    include = root / "include"
    include.mkdir(exist_ok=True)
    macros = ""
    for name, value in (("SUPPORT_ONEBODY", support), ("DERIV_ONEBODY_ORDER", order)):
        if value is not None:
            macros += f"#define LIBINT2_{name} {value}\n"
    # The minimum ordinary-gradient contract must not demand property order 1.
    macros += "#define LIBINT2_DERIV_ONEBODY_PROPERTY_ORDER 0\n"
    declarations = ""
    definitions = ""
    for op in OPERATORS:
        table = f"libint2_build_{op}1"
        memory = f"libint2_need_memory_{op}1"
        declarations += (f"extern void (*{table}[2][2])(const Libint_t*);\n"
                         f"std::size_t {memory}(int);\n")
        if table not in missing:
            definitions += f"void (*{table}[2][2])(const Libint_t*) = {{}};\n"
        if memory not in missing:
            definitions += f"std::size_t {memory}(int) {{ return 0; }}\n"
    (include / "libint2.h").write_text(
        '#pragma once\n#include <cstddef>\n' + macros
        + 'struct Libint_t {};\nextern "C" {\n' + declarations + '}\n')
    (root / "kernels.cpp").write_text(
        '#include "libint2.h"\nextern "C" {\n' + definitions + '}\n')
    (root / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.20)
project(capability_fixture LANGUAGES CXX)
add_library(fixture STATIC kernels.cpp)
target_include_directories(fixture PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/archive-$<CONFIG>.txt"
     CONTENT "$<TARGET_FILE:fixture>")
''')
    build = root / "build"
    for args in ([CMAKE, "-S", str(root), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
                 [CMAKE, "--build", str(build), "--config", "Release", "--parallel", "1"]):
        result = _command(args)
        assert result.returncode == 0, result.stdout
    archive = (build / "archive-Release.txt").read_text()
    package = root / "package"
    package.mkdir(exist_ok=True)
    cxx_target = "Libint2::int-cxx-headeronly-shared" if alias else "Libint2::cxx"
    (package / "Libint2Config.cmake").write_text(f'''
add_library(Libint2::fixture STATIC IMPORTED)
set_target_properties(Libint2::fixture PROPERTIES
    IMPORTED_LOCATION "{Path(archive).as_posix()}"
    INTERFACE_INCLUDE_DIRECTORIES "{include.as_posix()}")
add_library({cxx_target} INTERFACE IMPORTED)
set_target_properties({cxx_target} PROPERTIES
    INTERFACE_LINK_LIBRARIES Libint2::fixture)
''' + (f"add_library(Libint2::cxx ALIAS {cxx_target})\n" if alias else ""))
    # Same version for all capability cases: version selection must not save us.
    (package / "Libint2ConfigVersion.cmake").write_text('''
set(PACKAGE_VERSION "2.13.1")
set(PACKAGE_VERSION_COMPATIBLE TRUE)
''')
    return package


def _configure(root: Path, package: Path, *, vendored: bool = False) -> subprocess.CompletedProcess[str]:
    root.mkdir(parents=True, exist_ok=True)
    # Directory presence must not excuse the selected target's missing kernels.
    (root / "third_party" / "libint" / "install").mkdir(parents=True, exist_ok=True)
    (root / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.20)
project(libint_admission LANGUAGES CXX)
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /WX")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Werror")
endif()
find_package(Libint2 2.7 CONFIG REQUIRED)
include("{MODULE.as_posix()}")
vibeqc_require_libint_onebody_derivatives()
# The gate's executable override must not escape its function scope.
if(NOT CMAKE_TRY_COMPILE_TARGET_TYPE STREQUAL "STATIC_LIBRARY")
    message(FATAL_ERROR "The capability gate changed the caller's probe type")
endif()
''')
    return _command([CMAKE, "-S", str(root), "-B", str(root / "build"),
                     f"-DLibint2_DIR={package}", "-DCMAKE_BUILD_TYPE=Release",
                     "-DCMAKE_TRY_COMPILE_CONFIGURATION=Release",
                     "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
                     f"-DVIBEQC_REQUIRE_VENDORED={'ON' if vendored else 'OFF'}"])


def _assert_rejected(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode != 0, result.stdout
    # CMake wraps message(FATAL_ERROR) text to the terminal width.
    message = " ".join(result.stdout.split())
    assert "ordinary one-body first derivatives" in message, result.stdout
    assert "scripts/build_libint.sh" in message, result.stdout
    assert "Selected package:" in message, result.stdout


def _assert_optimized_probe(root: Path) -> None:
    log = (root / "build" / "CMakeFiles" / "vibeqc-libint-onebody.log").read_text()
    assert ("-O3" in log and "-Werror" in log) or ("/O2" in log and "/WX" in log), log


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("alias", [False, True], ids=["direct-target", "imported-alias"])
def test_supported_ordinary_derivatives_without_property_derivatives(tmp_path, order, alias):
    package = _provider(tmp_path / "provider", order=order, alias=alias)
    root = tmp_path / "consumer"
    result = _configure(root, package)
    assert result.returncode == 0, result.stdout
    assert "compile/link check passed" in result.stdout
    _assert_optimized_probe(root)


@pytest.mark.parametrize("support,order,diagnostic", [
    (1, 0, "LIBINT2_DERIV_ONEBODY_ORDER=0"),
    (1, -1, "LIBINT2_DERIV_ONEBODY_ORDER=-1"),
    (0, 1, "LIBINT2_SUPPORT_ONEBODY=0"),
    (None, 1, "LIBINT2_SUPPORT_ONEBODY=unknown"),
    (1, None, "LIBINT2_DERIV_ONEBODY_ORDER=unknown"),
])
def test_unsupported_or_unknown_headers_fail_configure(tmp_path, support, order, diagnostic):
    package = _provider(tmp_path / "provider", support=support, order=order)
    result = _configure(tmp_path / "consumer", package)
    _assert_rejected(result)
    assert diagnostic in result.stdout


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_supported_headers_with_missing_library_symbol_fail_link(tmp_path, symbol):
    package = _provider(tmp_path / "provider", missing=(symbol,))
    root = tmp_path / "consumer"
    result = _configure(root, package)
    _assert_rejected(result)
    assert symbol in result.stdout
    # Real linker failure, despite a caller requesting static-library probes.
    assert any(word in result.stdout.lower() for word in ("undefined", "unresolved"))
    _assert_optimized_probe(root)


@pytest.mark.parametrize("vendored", [False, True])
def test_install_presence_and_vendoring_flag_cannot_bypass_gate(tmp_path, vendored):
    package = _provider(tmp_path / "provider", order=0)
    _assert_rejected(_configure(tmp_path / "consumer", package, vendored=vendored))


def test_selected_package_changes_recheck_same_cmake_cache(tmp_path):
    unsupported = _provider(tmp_path / "unsupported", order=0)
    supported = _provider(tmp_path / "supported", order=1)
    root = tmp_path / "consumer"
    _assert_rejected(_configure(root, unsupported))
    result = _configure(root, supported)
    assert result.returncode == 0, result.stdout
    _assert_rejected(_configure(root, unsupported))


def test_library_replacement_at_same_path_invalidates_prior_success(tmp_path):
    provider = tmp_path / "provider"
    package = _provider(provider)
    root = tmp_path / "consumer"
    result = _configure(root, package)
    assert result.returncode == 0, result.stdout
    _provider(provider, missing=SYMBOLS)
    _assert_rejected(_configure(root, package))
