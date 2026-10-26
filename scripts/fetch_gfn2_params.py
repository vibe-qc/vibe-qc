#!/usr/bin/env python3
"""Prepare or check a complete published GFN2 cache using the runtime loader.

Run with the target runtime's Python. Set VIBEQC_GFN2_CACHE_DIR before import
(or use --cache-dir) to prepare a separate deployment cache. --check never
fetches or changes files; use it on the execution host before admission.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def fetch_params(force: bool = False, *, check: bool = False) -> dict:
    from vibeqc.semiempirical.methods.gfn2_params import (
        GFN2ParameterUnavailableError,
        gfn2_parameter_lineage,
        load_gfn2_params,
    )

    params = load_gfn2_params(force_refetch=force, allow_fetch=not check)
    lineage = gfn2_parameter_lineage(params)
    if params.n_elements() != 86 or lineage is None:
        raise GFN2ParameterUnavailableError("Deployment requires all 86 elements and cache lineage")
    if lineage["cache_source_sha256"] == "unavailable":
        raise GFN2ParameterUnavailableError("Deployment requires the published source SHA256")
    return {**lineage, "elements": 86, "license": "LGPL-3.0-or-later"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="Refresh on a connected preparation host")
    mode.add_argument("--check", action="store_true", help="Validate offline without changing the cache")
    mode.add_argument("--print", dest="print_path", action="store_true", help="Print resolved cache path")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--expected-cache-sha256")
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    if (args.expected_cache_sha256 or args.expected_source_sha256) and not args.check:
        parser.error("expected hashes require --check (no refresh or writes)")
    if args.cache_dir:
        os.environ["VIBEQC_GFN2_CACHE_DIR"] = str(args.cache_dir.expanduser())
    from vibeqc.semiempirical.methods.gfn2_params import gfn2_parameter_cache_path

    if args.print_path:
        print(gfn2_parameter_cache_path())
        return 0
    receipt = fetch_params(args.force, check=args.check)
    for expected, key in ((args.expected_cache_sha256, "cache_sha256"),
                          (args.expected_source_sha256, "cache_source_sha256")):
        if expected is not None and receipt[key] != expected:
            parser.error(f"{key} does not match the expected identity")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
