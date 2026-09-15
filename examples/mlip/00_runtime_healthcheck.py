#!/usr/bin/env python3
"""Check a vibe-qc MACE runtime without downloading model weights.

Run from a vibe-qc checkout:

    .venv-mace/bin/python examples/mlip/00_runtime_healthcheck.py

The script verifies the optional stack, the direct periodic API, the
registered default model, and the expected cache root. It reports the
source revision when the installed package is backed by a Git checkout.
It deliberately does not construct a calculator, so it is safe to run
before pre-warming the model cache.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import ase
import e3nn
import mace
import torch
import vibeqc
from vibeqc import run_job, run_neb
from vibeqc.banner import build_info
from vibeqc.mlip import mace_model_registry, resolve_model, save_mace_model_registry
from vibeqc.mlip.mace import (
    MACEModel,
    PeriodicMACEEvaluator,
    mace_cache_root,
    mace_calculator,
    optimize_periodic_mace_cell,
    optimize_periodic_mace_positions,
    run_periodic_mace,
)


def package_version(distribution: str, module: object) -> str:
    """Return installed distribution version with a module fallback."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the file at once."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_git_state() -> dict[str, str | bool] | None:
    """Return Git state for the checkout backing the imported package."""
    package_dir = Path(vibeqc.__file__).resolve().parent
    try:
        root = subprocess.check_output(
            ["git", "-C", str(package_dir), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        values = {
            "root": root,
            "branch": subprocess.check_output(
                ["git", "-C", root, "branch", "--show-current"], text=True
            ).strip(),
            "head": subprocess.check_output(
                ["git", "-C", root, "rev-parse", "HEAD"], text=True
            ).strip(),
            "origin_main": subprocess.check_output(
                ["git", "-C", root, "rev-parse", "refs/remotes/origin/main"],
                text=True,
            ).strip(),
            "dirty": bool(
                subprocess.check_output(
                    ["git", "-C", root, "status", "--porcelain"], text=True
                ).strip()
            ),
        }
    except (OSError, subprocess.CalledProcessError):
        return None
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-main-current",
        action="store_true",
        help=(
            "fail unless the imported checkout is clean, on main, and at its "
            "locally fetched origin/main"
        ),
    )
    parser.add_argument(
        "--require-asl-unset",
        action="store_true",
        help="fail if VIBEQC_ACCEPT_ASL globally acknowledges the OFF23 ASL",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if sys.version_info >= (3, 14):
        raise SystemExit(
            "MACE currently requires Python 3.13 or earlier; "
            f"this interpreter is {platform.python_version()}."
        )

    api = {
        "run_job": run_job,
        "run_neb": run_neb,
        "MACEModel": MACEModel,
        "mace_calculator": mace_calculator,
        "PeriodicMACEEvaluator": PeriodicMACEEvaluator,
        "run_periodic_mace": run_periodic_mace,
        "optimize_periodic_mace_cell": optimize_periodic_mace_cell,
        "optimize_periodic_mace_positions": optimize_periodic_mace_positions,
        "mace_cache_root": mace_cache_root,
        "mace_model_registry": mace_model_registry,
        "save_mace_model_registry": save_mace_model_registry,
    }
    if not all(callable(value) for value in api.values()):
        raise RuntimeError("one or more public periodic MACE APIs are unavailable")

    info = resolve_model(None)
    registry = mace_model_registry()
    registered_keys = {row["key"] for row in registry["model"]}
    if info.key != "medium-mpa-0" or info.key not in registered_keys:
        raise RuntimeError("the default MACE model registry entry is inconsistent")
    if (info.loader, info.license) != ("mace_mp", "MIT"):
        raise RuntimeError("the default MACE loader or license is inconsistent")

    source = build_info()
    git_state = source_git_state()
    if args.require_main_current:
        if git_state is None:
            raise SystemExit("MACE runtime is not backed by a readable Git checkout")
        problems = []
        if git_state["branch"] != "main":
            problems.append(f"branch={git_state['branch'] or 'detached'}")
        if git_state["dirty"]:
            problems.append("worktree=dirty")
        if git_state["head"] != git_state["origin_main"]:
            problems.append(
                f"HEAD={str(git_state['head'])[:12]} "
                f"origin/main={str(git_state['origin_main'])[:12]}"
            )
        if problems:
            raise SystemExit("MACE runtime is not current clean main: " + ", ".join(problems))

    asl_env = os.environ.get("VIBEQC_ACCEPT_ASL", "")
    asl_set = asl_env.strip().lower() not in ("", "0", "false", "no", "off")
    if args.require_asl_unset and asl_set:
        raise SystemExit(
            "VIBEQC_ACCEPT_ASL is globally set; OFF23 acknowledgment must be "
            "explicit and calculation-local"
        )

    print("vibe-qc MACE runtime health: OK")
    print(f"python:     {platform.python_version()}")
    print(f"vibe-qc:   {vibeqc.__version__}")
    print(f"git SHA:   {source.get('sha_full') or 'not available'}")
    print(f"git dirty: {source.get('dirty', 'not available')}")
    if git_state is not None:
        print(f"git branch:{git_state['branch'] or 'detached':>11}")
        print(f"origin/main: {str(git_state['origin_main'])[:12]}")
    print(f"ASE:       {package_version('ase', ase)}")
    print(f"torch:     {package_version('torch', torch)}")
    print(f"e3nn:      {package_version('e3nn', e3nn)}")
    print(f"MACE:      {package_version('mace-torch', mace)}")
    print(f"default:   {info.key}")
    print(f"loader:    {info.loader}")
    print(f"license:   {info.license}")
    print(f"citation:  {info.citation} ({info.doi})")
    print(f"ASL env:   {'set' if asl_set else 'unset'}")
    cache = mace_cache_root()
    print(f"cache:     {cache}")
    cached_files = sorted(path for path in cache.glob("*") if path.is_file())
    if cached_files:
        for path in cached_files:
            print(
                f"weight:    {path.name} "
                f"sha256={file_sha256(path)} size={path.stat().st_size}"
            )
    else:
        print("weight:    none cached; first model use will fetch weights")
    print(f"APIs:      {', '.join(api)}")


if __name__ == "__main__":
    main()
