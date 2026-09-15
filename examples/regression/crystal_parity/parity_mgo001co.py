"""Parity STUB: CO on MgO(001) slab RHF/extended at SHRINK 4 4.

CRYSTAL14 reference (``crystal_demos/mgo001co.d12``):
  - MgO(001) 1-layer slab + adsorbed CO molecule on one side
  - Uses ``Z=108`` ghost-atom basis for the molecular oxygen (CRYSTAL
    convention for separating molecular vs crystal-O basis sets)
  - Inline atom-specific extended basis for Mg / crystal-O /
    C / molecular-O
  - SHRINK 4 4, FMIXING 30, MAXCYCLE 50

**Blockers**:
  1. Inline-basis parser.
  2. ``dim=2`` BIPOLE driver path.
  3. ``Z=108`` ghost-atom convention (or vibe-qc's equivalent
     atom-specific basis override).
  4. Asymmetric slab handling (CO on one side only — breaks
     mirror symmetry; CRYSTAL handles via ``ATOMSYMM``).

Geometry not constructed by the builder — too many open questions.
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def main() -> int:
    print("=== CO on MgO(001) slab parity STUB ===")
    print("  BLOCKED: requires inline-basis parser, dim=2 driver, "
          "ghost-atom Z=108 convention, asymmetric-slab handling.")
    print("  This is the most-blocked demo in the suite; revisit "
          "after the simpler 3D + STO-3G demos pass parity.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
