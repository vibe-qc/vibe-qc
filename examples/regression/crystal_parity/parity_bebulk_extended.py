"""Parity STUB: Be bulk RHF/extended (4-shell s-only) at SHRINK 12 24.

CRYSTAL14 reference (``crystal_demos/bebulk.d12``):
  - Same geometry as ``be_sto3g.d12`` (SG 194 HCP, a=2.29, c=3.59 Å)
  - Inline 4-shell s-only basis (5s + 1s + 1sp + 1sp)
  - SHRINK 12 24, FMIXING 30, MAXCYCLE 130

**Blockers**:
  1. Inline-basis parser.
  2. Metallic (Be bulk) — needs Fermi-Dirac smearing (see
     ``parity_be_sto3g.py``).
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_bebulk_extended_geometry


def main() -> int:
    system = build_bebulk_extended_geometry()
    print("=== Be bulk EXTENDED basis (4-shell s-only) parity STUB ===")
    print(f"  geometry: dim={system.dim}, n_atoms={len(system.unit_cell)}, "
          f"n_elec={system.n_electrons()}")
    print("  BLOCKED: needs inline-basis parser + metallic smearing.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
