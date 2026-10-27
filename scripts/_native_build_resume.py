"""Lifecycle-only libint build recovery. Called with the checkout build lock held.

A saved contract binds source content, compiler bytes, recipe and CMake caches.
Legacy recovery additionally requires an explicit clean product SHA and the exact
reviewed libint recipe. It preserves the original source tree and Ninja objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

LEGACY_RECIPE = "5bf93e44e0ffedba0604cd0ab7fb3b1bbd64248a8ca9e2483ee746e07568a5e7"


def run(argv, **kwargs):
    return subprocess.check_output(argv, text=True, **kwargs).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache(path):
    return {
        m[1]: m[2]
        for line in path.read_text().splitlines()
        if (m := re.match(r"([^#/][^:=]*):[^=]+=(.*)$", line))
    }


def recipe(root):
    return run(
        [
            "bash",
            "-c",
            (
                '. "$1/scripts/_native_stamp.sh"; '
                '_vibeqc_recipe_hash "$1/scripts/build_libint.sh"'
            ),
            "resume",
            str(root),
        ]
    )


def locations(root):
    dep = root / "third_party/libint"
    return dep, dep / "src", dep / "build", dep / ".resume-contract.json"


def compiler_tools(compiler):
    tools = {str(Path(compiler).resolve()): digest(compiler)}
    for tool in ("cc1plus", "as", "ld"):
        try:
            result = subprocess.run(
                [str(compiler), "-print-prog-name=" + tool],
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except OSError as error:
            raise ValueError("compiler changed or is unavailable") from error
        name = result.stdout.strip()
        path = (
            Path(name)
            if name.startswith("/")
            else Path(shutil.which(name) or "/nonexistent")
        )
        if result.returncode == 0 and path.is_file():
            tools[str(path.resolve())] = digest(path)
    return tools


def inputs(root):
    _, source, build, _ = locations(root)
    compiler = (
        cache(build / "CMakeCache.txt").get("CMAKE_CXX_COMPILER")
        if (build / "CMakeCache.txt").exists()
        else None
    )
    compiler = compiler or os.environ.get("CXX") or shutil.which("c++")
    if not compiler or not Path(compiler).is_file():
        raise ValueError("cannot bind the C++ compiler executable")
    entries = run(["git", "-C", str(source), "ls-files", "--stage", "-z"]).split("\0")
    source_hash = hashlib.sha256()
    for entry in sorted(filter(None, entries)):
        metadata, name = entry.split("\t", 1)
        mode, object_id, stage = metadata.split()
        if stage != "0":
            raise ValueError("libint source has unresolved index entries")
        path = source / name
        source_hash.update(name.encode() + b"\0")
        if mode == "160000":
            if (path / ".git").exists() and (
                run(["git", "-C", str(path), "rev-parse", "HEAD"]) != object_id
                or run(["git", "-C", str(path), "status", "--porcelain"])
            ):
                raise ValueError("libint submodule differs from its pinned content")
            source_hash.update(("gitlink:" + object_id).encode())
        else:
            source_hash.update(path.read_bytes())
    return {
        "source": str(source.resolve()),
        "source_commit": run(["git", "-C", str(source), "rev-parse", "HEAD"]),
        "source_sha256": source_hash.hexdigest(),
        "recipe": recipe(root),
        "compiler": str(Path(compiler).resolve()),
        "compiler_sha256": digest(compiler),
        "toolchain": compiler_tools(compiler),
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "CC",
                "CXX",
                "CFLAGS",
                "CXXFLAGS",
                "CPPFLAGS",
                "LDFLAGS",
                "CMAKE_PREFIX_PATH",
                "VIBEQC_LIBINT_MAX_AM",
                "CPATH",
                "CPLUS_INCLUDE_PATH",
                "C_INCLUDE_PATH",
                "LIBRARY_PATH",
                "LD_PRELOAD",
            )
        },
    }


def configurations(build):
    return {
        str(p.relative_to(build)): digest(p)
        for p in sorted(build.rglob("CMakeCache.txt"))
    }


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def prepare(root):
    _, _, build, contract = locations(root)
    actual = inputs(root)
    if contract.exists():
        saved = json.loads(contract.read_text())
        if saved["inputs"] != actual:
            raise ValueError(
                "interrupted libint source/recipe/compiler changed; preserved build requires explicit rebuild"
            )
        actual_configs = configurations(build)
        recorded = saved.get("configuration", {})
        if any(actual_configs.get(name) != value for name, value in recorded.items()):
            raise ValueError(
                "interrupted libint CMake configuration changed; preserved build requires review"
            )
        # A superbuild creates its nested cache after the first configure returns.
        # Accept that first appearance only with the parent compiler/flags/options.
        parent = cache(build / "CMakeCache.txt")
        for name in actual_configs.keys() - recorded.keys():
            child = cache(build / name)
            if "CMAKE_CXX_COMPILER" not in child:
                continue  # C-only compiler feature probes do not build libint.
            for key in child.keys() & parent.keys():
                if (
                    key.startswith("LIBINT2_")
                    or key
                    in {
                        "CMAKE_CXX_COMPILER",
                        "CMAKE_CXX_FLAGS",
                        "CMAKE_CXX_FLAGS_RELEASE",
                        "CMAKE_BUILD_TYPE",
                    }
                ) and not option_matches(child[key], parent[key]):
                    raise ValueError(f"new nested CMake configuration changed {key}")
        saved["configuration"] = actual_configs
        write(contract, saved)
        if not (build / "build.ninja").is_file():
            raise ValueError("saved libint build graph is missing")
        print("resume")
    elif (build / "CMakeCache.txt").exists():
        raise ValueError(
            "unrecorded libint build preserved; use recover_libint_build.sh with its exact source SHA"
        )
    else:
        write(contract, {"inputs": actual, "configuration": {}})
        print("fresh")


def configured(root):
    _, _, build, contract = locations(root)
    saved = json.loads(contract.read_text())
    actual = inputs(root)
    if saved["inputs"] != actual:
        raise ValueError("source/compiler changed during configuration")
    saved["configuration"] = configurations(build)
    write(contract, saved)


def option_matches(actual, expected):
    if expected in {"ON", "OFF"}:
        accepted = (
            {"ON", "TRUE", "YES", "1"}
            if expected == "ON"
            else {"OFF", "FALSE", "NO", "0", ""}
        )
        return actual is not None and actual.upper() in accepted
    return actual == expected


def verify_legacy(root, expected):
    dep, source, build, _contract = locations(root)
    for key in (
        "CFLAGS",
        "CXXFLAGS",
        "CPPFLAGS",
        "LDFLAGS",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "C_INCLUDE_PATH",
        "LIBRARY_PATH",
        "LD_PRELOAD",
        "CMAKE_PREFIX_PATH",
        "CMAKE_TOOLCHAIN_FILE",
    ):
        if os.environ.get(key):
            raise ValueError(
                f"legacy recovery does not adopt an unrecorded {key} override"
            )

    other_installs = [
        p.name
        for p in dep.parent.iterdir()
        if p.name != "libint" and (p / "install").exists()
    ]
    if other_installs:
        raise ValueError(
            "legacy recovery will not restamp other native installations: "
            + ", ".join(other_installs)
        )
    if run(["git", "-C", str(root), "rev-parse", "HEAD"]) != expected:
        raise ValueError("product source does not match the requested immutable SHA")
    if run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("product source has tracked edits")
    if recipe(root) != LEGACY_RECIPE:
        raise ValueError("legacy recovery does not recognize this exact libint recipe")
    if (
        run(["git", "-C", str(source), "rev-parse", "HEAD"])
        != "5fb07b4862f219749c51d59ee08073d20a56d506"
    ):
        raise ValueError("libint source commit differs from the pinned recipe")
    # Recreate only the two reviewed header fixes on a disposable original.
    builder = (root / "scripts/build_libint.sh").read_text()
    definitions = builder[
        builder.index("patch_boys_endpoint() {") : builder.index(
            "# Both markers must match."
        )
    ]
    changed = run(
        ["git", "-C", str(source), "diff", "--name-only", "HEAD"]
    ).splitlines()
    if any(name != "include/libint2/boys.h" for name in changed):
        raise ValueError("libint source has changes outside the recipe's header fixes")
    with tempfile.TemporaryDirectory() as temporary:
        header = Path(temporary) / "boys.h"
        header.write_text(
            run(["git", "-C", str(source), "show", "HEAD:include/libint2/boys.h"])
            + "\n"
        )
        subprocess.run(
            [
                "bash",
                "-c",
                definitions + '\npatch_boys_endpoint "$1"\npatch_erfc_scratch "$1"',
                "resume",
                str(header),
            ],
            check=True,
        )
        if header.read_bytes() != (source / "include/libint2/boys.h").read_bytes():
            raise ValueError("libint header differs from the exact recipe patches")
    values = cache(build / "CMakeCache.txt")
    wanted = {
        "CMAKE_HOME_DIRECTORY": str(source),
        "CMAKE_INSTALL_PREFIX": str(dep / "install"),
        "CMAKE_GENERATOR": "Ninja",
        "CMAKE_BUILD_TYPE": "Release",
        "LIBINT2_REQUIRE_CXX_API": "ON",
        "LIBINT2_ENABLE_ONEBODY": "2",
        "LIBINT2_DISABLE_ONEBODY_PROPERTY_DERIVS": "OFF",
        "LIBINT2_ENABLE_ERI": "2",
        "LIBINT2_ENABLE_ERI3": "2",
        "LIBINT2_ENABLE_ERI2": "2",
        "LIBINT2_MAX_AM": "5",
        "LIBINT2_ERI_MAX_AM": "5;4;4",
        "LIBINT2_ONEBODY_MAX_AM": "5;4;4",
        "LIBINT2_ERI3_MAX_AM": "6;6;6",
        "LIBINT2_ERI2_MAX_AM": "6;6;6",
        "BUILD_SHARED_LIBS": "ON",
        "LIBINT2_CARTGAUSS_ORDERING": "standard",
        "LIBINT2_SHGAUSS_ORDERING": "standard",
        "LIBINT2_SHELL_SET": "standard",
        "LIBINT2_ERI_STRATEGY": "1",
        "LIBINT2_MULTIPOLE_MAX_ORDER": "4",
        "LIBINT2_GENERATE_FMA": "OFF",
        "LIBINT2_ENABLE_GENERIC_CODE": "OFF",
        "LIBINT2_PROFILE": "OFF",
        "LIBINT2_FLOP_COUNT": "OFF",
        "LIBINT2_ACCUM_INTS": "OFF",
        "LIBINT2_CONTRACTED_INTS": "ON",
        "LIBINT2_SINGLE_EVALTYPE": "ON",
        "LIBINT2_USE_COMPOSITE_EVALUATORS": "ON",
        "LIBINT2_" + "REALTYPE": "double",
        "LIBINT2_ALIGN_SIZE": "0",
        "LIBINT2_ENABLE_MPFR": "OFF",
    }
    for key, value in wanted.items():
        if not option_matches(values.get(key), value):
            raise ValueError(f"legacy CMake {key} does not match the reviewed recipe")
    nested = build / "src/lib/libint/library-prefix/src/library-build/CMakeCache.txt"
    if not nested.exists():
        raise ValueError("retained generated-library configuration is missing")
    for config in (values, cache(nested)):
        for key, value in {
            "CMAKE_CXX_FLAGS": "",
            "CMAKE_CXX_FLAGS_RELEASE": "-O3 -DNDEBUG",
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_CXX_COMPILER": values["CMAKE_CXX_COMPILER"],
        }.items():
            if config.get(key) != value:
                raise ValueError(
                    f"legacy toolchain flag {key} differs from the supported recipe"
                )
        if any(
            config.get(key)
            for key in ("CMAKE_TOOLCHAIN_FILE", "CMAKE_PROJECT_TOP_LEVEL_INCLUDES")
        ):
            raise ValueError("legacy toolchain injection cannot be adopted")
    cmake_version = ".".join(
        values["CMAKE_CACHE_" + key + "_VERSION"] for key in ("MAJOR", "MINOR", "PATCH")
    )
    if (
        run([values["CMAKE_COMMAND"], "--version"]).splitlines()[0]
        != "cmake version " + cmake_version
    ):
        raise ValueError("CMake version changed since configuration")
    compiler = Path(values["CMAKE_CXX_COMPILER"]).resolve()
    records = list((build / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
    if len(records) != 1:
        raise ValueError("ambiguous recorded compiler configuration")
    text = records[0].read_text()
    version = re.search(r'set\(CMAKE_CXX_COMPILER_VERSION "([^"]+)"\)', text)
    if not version or 'set(CMAKE_CXX_COMPILER_ID "GNU")' not in text:
        raise ValueError(
            "legacy adoption currently requires recorded GNU compiler identity"
        )
    if (
        run([str(compiler), "-dumpfullversion"]) != version[1]
        or compiler.stat().st_mtime > records[0].stat().st_mtime
    ):
        raise ValueError("compiler identity changed since original configuration")
    if any(
        Path(tool).stat().st_mtime > records[0].stat().st_mtime
        for tool in compiler_tools(compiler)
    ):
        raise ValueError("compiler subtool changed since original configuration")
    if not (build / "build.ninja").is_file() or not list(build.rglob(".ninja_log")):
        raise ValueError("original Ninja graph/log unavailable")
    return {
        "inputs": inputs(root),
        "configuration": configurations(build),
        "legacy_product_sha": expected,
        "adoption": "recorded CMake recipe and GNU version; compiler predates configure",
    }


def held_build_lock(root):
    if os.environ.get("VIBEQC_BUILD_LOCK_ACTIVE") != "1":
        raise ValueError("recovery apply requires the original checkout build lock")
    os.fstat(197)
    subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            '. "$1/scripts/_build_lock.sh"; vibeqc_acquire_build_lock',
            "resume",
            str(root),
        ],
        pass_fds=(197,),
        check=True,
    )
    return (197,)


def recover(root, expected, apply):
    dep, _, build, contract = locations(root)
    proof = verify_legacy(root, expected)
    if (
        contract.exists()
        and json.loads(contract.read_text())["inputs"] != proof["inputs"]
    ):
        raise ValueError("saved resume contract conflicts with this legacy attempt")
    if not apply:
        print(json.dumps(proof, indent=2))
        return
    lock_fds = held_build_lock(root)
    write(contract, proof)
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "install"],
        check=True,
        pass_fds=lock_fds,
    )
    if inputs(root) != proof["inputs"]:
        raise ValueError("source/compiler changed while resuming libint")
    # Run the pinned builder's own artifact checks and metadata finalization.
    # No product file is patched, and no source/build stamp is synthesized here.
    builder = (root / "scripts/build_libint.sh").read_text()
    tail = builder[builder.index("# --- openSUSE lib64 reconciliation") :]
    env = dict(
        os.environ,
        SCRIPT_DIR=str(root / "scripts"),
        REPO_ROOT=str(root),
        BUILD_DIR=str(build),
        INSTALL_DIR=str(dep / "install"),
        VIBEQC_LIBINT_AM_SPEC="5_4_4",
        LIBINT_FITTING_AM_SPEC="6_6_6",
    )
    subprocess.run(["bash", "-euo", "pipefail", "-c", tail], env=env, check=True)
    verify_legacy(
        root, expected
    )  # recheck source and configuration before provenance writes
    attestation = json.loads(
        run(
            [
                "python3",
                str(root / "scripts/attest_native_image.py"),
                "libint",
                str(dep / "install"),
            ]
        )
    )
    # The original stamp writer supports targeted partial builds. Every present
    # install here is this freshly completed libint; missing deps are omitted.
    subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            '. "$1/scripts/_native_stamp.sh"; vibeqc_stamp_write',
            "resume",
            str(root),
        ],
        cwd=root,
        check=True,
    )
    write(
        dep / "resume-result.json",
        proof | {"installed": True, "attestation": attestation},
    )
    print(
        "Resumed libint installed and runtime identity checked; run the original product installer next."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "configured", "recover"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-source-sha")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.operation == "recover":
        if not args.expected_source_sha or not re.fullmatch(
            r"[0-9a-f]{40}", args.expected_source_sha
        ):
            parser.error("recovery requires the exact 40-character source SHA")
        recover(root, args.expected_source_sha, args.apply)
    else:
        {"prepare": prepare, "configured": configured}[args.operation](root)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError) as error:
        raise SystemExit(f"Native build resume refused: {error}") from error
