#!/usr/bin/env python3
"""Key the CI dependency cache by the existing build-relevant recipe contract.

Run --write after changing a native recipe. --check refuses stale configuration
before any cached install is consumed. This is cache selection, not provenance
attestation: the builders still authenticate each restored library themselves.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PROFILE = "debian12-gcc12-x86_64-default-flags-am5_4_4-openblas0"
PREFIX = "third-party-debian12-gcc12-"
KEY_LINE = re.compile(r"(?m)^    - key: (third-party-debian12-gcc12-[0-9a-f]+)$")
DEPENDENCIES = ("libint", "libxc", "spglib", "fftw", "libecpint", "openblas")
# Preserve the cache namespace of the existing candidate, only for its exact
# complete recipe signature. No fallback to an older or partly matching cache.
LEGACY_SIGNATURE = "b41773b173e27640e830f6b748061641e67086a9091c2ee666ffea56d9b63d7f"
LEGACY_KEY = PREFIX + "0458cd1f5f39acdf69743b7c65fe3f1a9256c7c3"


def recipe_signature(root: Path = ROOT) -> str:
    paths = [root / "scripts" / f"build_{name}.sh" for name in DEPENDENCIES]
    paths += [root / "scripts" / name for name in
              ("_libint_max_am.sh", "_verify_source.sh")]
    hashes = subprocess.check_output([
        "bash", "-euc",
        '. "$1/scripts/_native_stamp.sh"; shift; '
        'for recipe in "$@"; do _vibeqc_recipe_hash "$recipe"; done',
        "native-cache", str(root), *map(str, paths),
    ], env={**os.environ, "VIBEQC_LIBINT_MAX_AM": "5_4_4"}, text=True)
    rows = [PROFILE] + [f"{path.relative_to(root)} {digest}"
                        for path, digest in zip(paths, hashes.splitlines(), strict=True)]
    return hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()


def cache_key(root: Path = ROOT) -> str:
    signature = recipe_signature(root)
    return LEGACY_KEY if signature == LEGACY_SIGNATURE else PREFIX + signature


def verify(root: Path = ROOT) -> str:
    expected = cache_key(root)
    actual = KEY_LINE.findall((root / ".gitlab-ci.yml").read_text())
    if actual != [expected]:
        raise ValueError("native dependency recipes changed; run "
                         "python scripts/ci/native_cache.py --write and commit the CI key")
    return expected


def verify_runtime() -> None:
    # CI caches are extracted before before_script. Refuse a profile override
    # before install.sh can mistake that cache for a compatible installation.
    for name, default in (("VIBEQC_LIBINT_MAX_AM", "5_4_4"), ("WITH_OPENBLAS", "0")):
        if os.environ.get(name, default) != default:
            raise ValueError(f"{name} differs from the native CI cache profile")
    for name in ("CC", "CXX", "FC", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "FFLAGS", "FCFLAGS",
                 "CMAKE_TOOLCHAIN_FILE", "CMAKE_PREFIX_PATH"):
        if os.environ.get(name):
            raise ValueError(f"{name} requires a separate native CI cache profile")
    release = platform.freedesktop_os_release()
    if (release.get("ID"), release.get("VERSION_ID"), platform.machine()) != ("debian", "12", "x86_64"):
        raise ValueError("native CI cache requires Debian 12 x86_64")
    for compiler in ("gcc", "g++", "gfortran"):
        version = subprocess.check_output([compiler, "-dumpversion"], text=True).strip()
        if version.split(".")[0] != "12":
            raise ValueError(f"native CI cache requires {compiler} 12, found {version}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    if args.write:
        path = ROOT / ".gitlab-ci.yml"
        updated, count = KEY_LINE.subn("    - key: " + cache_key(), path.read_text())
        if count != 1:
            raise ValueError("expected exactly one native dependency cache key")
        path.write_text(updated)
    print(verify())
    if args.runtime:
        verify_runtime()


if __name__ == "__main__":
    main()
