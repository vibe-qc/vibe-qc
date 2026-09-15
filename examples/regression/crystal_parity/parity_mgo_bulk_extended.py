"""Parity STUB: MgO bulk RHF/[8411G(Mg)+8511G(O)] at SHRINK 8 8.

CRYSTAL14 reference (``crystal_demos/mgo_bulk.d12``):
  - Same geometry as ``mgo_sto3g.d12`` (SG 225, a=4.21 Å)
  - Inline atom-specific basis:
      Mg: 8411G (8s GTO + 4sp + 1sp + 1sp = 3 contracted shells)
      O:  8511G (8s + 5sp + 1sp + 1sp = 3 contracted shells)
  - SHRINK 8 8, FMIXING 30

**Blocker**: requires the inline-basis parser to translate the .d12
basis block into a ``vibeqc.BasisSet``. Geometry verified by
:func:`crystal_demos.builders.build_mgo_bulk_extended_geometry`.
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_mgo_bulk_extended_geometry


def main() -> int:
    system = build_mgo_bulk_extended_geometry()
    print("=== MgO bulk EXTENDED basis parity STUB ===")
    print(f"  geometry verified: dim={system.dim}, n_atoms="
          f"{len(system.unit_cell)}, n_elec={system.n_electrons()}")
    print()
    print("  BLOCKED: needs inline-basis parser to import the 8411G(Mg)/"
          "8511G(O) atom-specific basis from crystal_demos/mgo_bulk.d12.")
    print("  Action: implement scripts/basisset_dev/parse_crystal_inline.py "
          "→ vibeqc.BasisSet adapter (planned for the basissetdev branch).")
    return 3


if __name__ == "__main__":
    sys.exit(main())
