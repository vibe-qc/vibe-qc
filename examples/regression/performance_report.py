"""Assess a frozen performance matrix and paired execution receipts (#574)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core.performance import assess_performance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("samples", type=Path)
    args = parser.parse_args(argv)
    try:
        report = assess_performance(json.loads(args.manifest.read_text()),
                                    json.loads(args.samples.read_text()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
