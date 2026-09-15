"""Parity STUB: NiO bulk UHF/extended (with polarization) at SHRINK 8 8.

CRYSTAL14 reference (``crystal_demos/nio.d12``):
  - Same geometry as ``nio_sto3g.d12`` (SG 225, a=4.164 Å)
  - Inline 7-shell Ni basis with d-polarization + 4-shell O basis
  - UHF, SPINLOCK 2 15 (M_s=1 → multiplicity=3), LEVSHIFT 3 1,
    TOLINTEG 7 7 7 7 14, TOLDEE 7, FMIXING 30

**Blockers**:
  1. Inline-basis parser.
  2. UHF driver (see ``parity_nio_sto3g.py``).
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_nio_extended_geometry


def main() -> int:
    system = build_nio_extended_geometry()
    print("=== NiO bulk EXTENDED basis UHF parity STUB ===")
    print(f"  geometry: dim={system.dim}, n_atoms={len(system.unit_cell)}, "
          f"n_elec={system.n_electrons()}, mult={system.multiplicity}")
    print("  BLOCKED: needs inline-basis parser + UHF BIPOLE driver.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
