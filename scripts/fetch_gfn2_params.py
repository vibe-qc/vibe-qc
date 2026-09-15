#!/usr/bin/env python3
"""Fetch GFN2-xTB parameters from the xTB GitHub repository.

On-demand fetcher per ADR-002: downloads the published GFN2-xTB
parameter set (Grimme group, LGPL-3.0) and converts it to vibe-qc's
TOML format.  Parameters are cached locally.

Usage:
    python scripts/fetch_gfn2_params.py              # fetch + cache
    python scripts/fetch_gfn2_params.py --force       # re-download
    python scripts/fetch_gfn2_params.py --print       # print location

The cached file is at ~/.cache/vibeqc/gfn2_xtb_params.toml
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "vibeqc"
CACHE_FILE = CACHE_DIR / "gfn2_xtb_params.toml"
XTB_REPO_URL = "https://raw.githubusercontent.com/grimme-lab/xtb/main"


def get_cache_path() -> Path:
    """Return the cache file path."""
    return CACHE_FILE


def is_cached() -> bool:
    """Check if parameters are already cached."""
    return CACHE_FILE.exists()


def fetch_params(force: bool = False) -> Path:
    """Fetch GFN2-xTB parameters, return path to cached TOML file.

    If the file is already cached and force=False, returns it immediately.
    Otherwise downloads and converts.

    Currently returns a placeholder with instructions since the xTB
    parameter format requires parsing that will be implemented when
    the LGPL licensing situation is resolved.
    """
    if not force and is_cached():
        return CACHE_FILE

    print("GFN2-xTB parameters: on-demand fetch from Grimme group repository.")
    print(f"  Source: {XTB_REPO_URL}")
    print(f"  Cache:  {CACHE_FILE}")
    print()
    print("NOTE: GFN2-xTB parameters are LGPL-3.0 licensed.")
    print("      vibe-qc is MPL-2.0. Parameters are fetched at runtime")
    print("      and cached locally; they are NOT bundled with vibe-qc.")
    print("      A request for LGPL exception has been filed with the")
    print("      Grimme group (ADR-002).")
    print()

    try:
        import urllib.request
    except ImportError:
        print("ERROR: urllib.request not available", file=sys.stderr)
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Download the GFN2-xTB parameter file
    # The xTB repo stores parameters in param/ directory
    # Key files: param_gfn2.xtb (global), .par_* (element-specific)
    param_url = f"{XTB_REPO_URL}/param_gfn2-xtb.txt"

    try:
        print(f"Downloading {param_url} ...")
        with urllib.request.urlopen(param_url) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        print(f"ERROR: could not download parameters: {e}", file=sys.stderr)
        print("Falling back to placeholder parameter set.", file=sys.stderr)

        # Write a placeholder TOML with instructions
        placeholder = """# GFN2-xTB parameter placeholder
# Run: python scripts/fetch_gfn2_params.py --force
# to re-attempt download.
# See ADR-002 for licensing details.

[global]
note = "Placeholder — GFN2-xTB parameters could not be downloaded."
source = "https://github.com/grimme-lab/xtb"
license = "LGPL-3.0"
"""
        CACHE_FILE.write_text(placeholder)
        return CACHE_FILE

    # Convert xTB format to vibe-qc TOML
    # The xTB param_gfn2.xtb format is a custom key-value format.
    # For now, write a structured placeholder that records the attempt.
    sha = hashlib.sha256(raw.encode()).hexdigest()[:16]

    toml_content = f"""# GFN2-xTB parameters — fetched from xTB repository
# Source: {param_url}
# SHA-256 (first 16 chars): {sha}
# License: LGPL-3.0 (Grimme group, 2019)
# DOI: 10.1021/acs.jctc.8b01176
#
# NOTE: Full parameter conversion from xTB format to vibe-qc TOML
# is pending.  The raw parameter data was successfully downloaded
# and is available for parsing.

[global]
note = "Raw parameters downloaded; TOML conversion pending."
source_url = "{param_url}"
sha256_prefix = "{sha}"

# Raw parameter content (first 2KB)
raw_start = \"\"\"
{raw[:2000]}
\"\"\"
"""
    CACHE_FILE.write_text(toml_content)
    print(f"Parameters cached at {CACHE_FILE}")
    print("Full TOML conversion pending — raw data downloaded successfully.")
    return CACHE_FILE


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GFN2-xTB parameters from Grimme group repository"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if cached"
    )
    parser.add_argument(
        "--print", action="store_true", help="Print cache file path and exit"
    )
    args = parser.parse_args()

    if getattr(args, "print"):
        print(CACHE_FILE)
        return

    path = fetch_params(force=args.force)
    print(f"\nCache: {path}")
    print("To use in vibe-qc:")
    print("  from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params")
    print("  params = load_gfn2_params()")


if __name__ == "__main__":
    main()
