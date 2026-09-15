"""Parity STUB: urea molecular crystal RHF/3-21G at SHRINK 8 8.

CRYSTAL14 reference (``crystal_demos/urea_bulk_321G.d12``):
  - SG 113 (P-42_1m, tetragonal non-centrosymmetric)
  - a=5.565 Å, c=4.684 Å, 5-atom asymmetric unit (C / O / N / H / H)
  - CRYSTAL auto-expands to 16 atoms / primitive cell
  - Inline 3-21G basis (atom-specific, 3 shells per heavy atom + 2 H)
  - SHRINK 8 8

**Blockers**:
  1. SG 113 asymmetric-unit expansion: vibe-qc's ``PeriodicSystem``
     does not expand from a space-group + asymmetric unit; we need
     either spglib-driven expansion or hand-listing of all 16 atoms.
  2. Inline-basis parser for the 3-21G block.
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def main() -> int:
    print("=== urea molecular crystal (3-21G) parity STUB ===")
    print("  BLOCKED:")
    print("    1) SG 113 asymmetric-unit expansion (5 → 16 atoms)")
    print("    2) inline-basis parser for 3-21G")
    print("  Action: spglib-driven asymmetric-unit expansion helper, "
          "then inline-basis parser.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
