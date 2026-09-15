"""Parity STUB: MgO(001) 1-layer slab RHF/extended at SHRINK 8 8.

CRYSTAL14 reference (``crystal_demos/mgo001.d12``):
  - 1-layer slab cut from MgO rocksalt, surface plane = (001),
    a=4.21 Å; built via ``SLABCUT`` from the bulk SG 225 input
  - Inline atom-specific extended basis (8411G/8511G)
  - SHRINK 8 8, TOLINTEG 7 7 7 7 14, FMIXING 30

**Blockers**:
  1. Inline-basis parser.
  2. ``dim=2`` BIPOLE driver path verification.
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_mgo001_sto3g_geometry


def main() -> int:
    system = build_mgo001_sto3g_geometry()
    print("=== MgO(001) 1-layer slab parity STUB ===")
    print(f"  geometry: dim={system.dim}, n_atoms={len(system.unit_cell)}, "
          f"n_elec={system.n_electrons()}")
    print("  BLOCKED: needs inline-basis parser + verified dim=2 BIPOLE "
          "path.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
