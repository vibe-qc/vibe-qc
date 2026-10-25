#!/usr/bin/env python3
"""Check the vendored QVF schema and reference pages against one release.

The CI checkout selected by QVF_TAG is the source of truth. All vendored files
must remain byte-identical; keep provenance in docs/qvf/index.md.

Usage: check_qvf_schema.py --qvf-root /path/to/qvf/checkout
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
VENDORED = ROOT / "python/vibeqc/output/formats/qvf_manifest.schema.json"
VENDORED_DOCS = ROOT / "docs/qvf/_vendored"
DOC_SOURCES = {
    "qvf-format-spec.md": "spec/qvf-format-spec.md",
    "GOVERNANCE.md": "GOVERNANCE.md",
    "README.md": "conformance/README.md",
    "integration_guide.md": "docs/integration_guide.md",
    "library_guide.md": "docs/library_guide.md",
    "orca_integration.md": "docs/orca_integration.md",
}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qvf-root", required=True, type=pathlib.Path)
    ap.add_argument("--vendored", type=pathlib.Path, default=VENDORED)
    ap.add_argument("--vendored-docs", type=pathlib.Path, default=VENDORED_DOCS)
    args = ap.parse_args(argv)
    pairs = [(args.qvf_root / "spec/qvf_manifest.schema.json", args.vendored)]
    pairs.extend((args.qvf_root / source, args.vendored_docs / name)
                 for name, source in DOC_SOURCES.items())
    result = 0
    for normative, vendored in pairs:
        missing = [path for path in (normative, vendored) if not path.is_file()]
        if missing:
            for path in missing:
                print(f"error: missing {path}", file=sys.stderr)
            result = 2
            continue
        expected = hashlib.sha256(normative.read_bytes()).hexdigest()
        actual = hashlib.sha256(vendored.read_bytes()).hexdigest()
        if expected != actual:
            print(f"error: {vendored} differs from {normative}. "
                  "Re-vendor from QVF_TAG, or deliberately update the pin.",
                  file=sys.stderr)
            result = max(result, 1)
        else:
            print(f"{vendored.name}: identical ({actual})")
    return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
