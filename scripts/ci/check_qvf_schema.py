#!/usr/bin/env python3
"""Assert vibe-qc's manifest schema matches the QVF specification.

vibe-qc writes QVF and carries its own copy of the manifest schema; the
normative copy lives in the qvf repository. Before the 2026-09 split the
viewer's copy was a symlink into this tree and the guards that compared them
skipped whenever a sibling checkout was absent -- which, post-split, is their
permanent state. This is the producer half of the replacement.

Build-time only: qvf is a reference, not a runtime dependency.

Usage:  check_qvf_schema.py --qvf-root /path/to/qvf/checkout
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

VENDORED = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python" / "vibeqc" / "output" / "formats" / "qvf_manifest.schema.json"
)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qvf-root", required=True, type=pathlib.Path)
    ap.add_argument("--vendored", type=pathlib.Path, default=VENDORED)
    args = ap.parse_args(argv)

    normative = args.qvf_root / "spec" / "qvf_manifest.schema.json"
    for p in (normative, args.vendored):
        if not p.is_file():
            print(f"error: missing {p}", file=sys.stderr)
            return 2

    ha = hashlib.sha256(normative.read_bytes()).hexdigest()
    hb = hashlib.sha256(args.vendored.read_bytes()).hexdigest()
    print(f"normative {ha}\nvendored  {hb}")
    if ha != hb:
        print(
            f"error: {args.vendored} has diverged from the QVF specification.\n"
            "       Re-vendor it, or bump QVF_TAG deliberately if the format\n"
            "       changed.",
            file=sys.stderr,
        )
        return 1
    print("schema: identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
