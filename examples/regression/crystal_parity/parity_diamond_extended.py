"""Parity STUB: diamond (C) RHF/6-21G+d-polarization at SHRINK 8 8.

CRYSTAL14 reference (``crystal_demos/diamond.d12``):
  - Same geometry as ``diamond_sto3g.d12`` (SG 227, a=3.57 Å)
  - Inline 6-21G basis modified + d-polarization (4 contracted shells)
  - SHRINK 8 8

**Blocker**: inline-basis parser (same as MgO extended).
Geometry verified by
:func:`crystal_demos.builders.build_diamond_extended_geometry`.
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_diamond_extended_geometry


def main() -> int:
    system = build_diamond_extended_geometry()
    print("=== diamond EXTENDED basis (6-21G+d) parity STUB ===")
    print(f"  geometry verified: dim={system.dim}, n_atoms="
          f"{len(system.unit_cell)}, n_elec={system.n_electrons()}")
    print("  BLOCKED: needs inline-basis parser for the 6-21G+d block.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
