"""Parity STUB: silicon bulk RHF/6-21G(modified) at SHRINK 8 8.

CRYSTAL14 reference (``crystal_demos/sibulk.d12``):
  - Same geometry as ``sibulk_sto3g.d12`` (SG 227, a=5.42 Å)
  - Inline 6-21G modified (4 contracted shells; one polarization)
  - SHRINK 8 8, TOLDEE 7

**Blocker**: inline-basis parser.
Geometry verified by
:func:`crystal_demos.builders.build_sibulk_extended_geometry`.
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_sibulk_extended_geometry


def main() -> int:
    system = build_sibulk_extended_geometry()
    print("=== silicon bulk EXTENDED basis (6-21G mod) parity STUB ===")
    print(f"  geometry verified: dim={system.dim}, n_atoms="
          f"{len(system.unit_cell)}, n_elec={system.n_electrons()}")
    print("  BLOCKED: needs inline-basis parser.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
