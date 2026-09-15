"""Parity STUB: (SN)x polymer RHF/STO-3G at SHRINK 13 13.

CRYSTAL14 reference (``crystal_demos/sn_polym_sto3g.d12``):
  - 1D POLYMER, a=4.431 bohr, S + N (2 atoms / cell, 23 electrons → odd)
  - STO-3G inline
  - SHRINK 13 13

**Blockers**:
  1. ``run_pbc_bipole_rhf`` not exercised at ``dim=1``; 1D direct-space
     cutoff needs linear-truncation handling.
  2. Odd electron count → open-shell, metallic. Needs UHF driver +
     Fermi-Dirac smearing.
Geometry verified by :func:`crystal_demos.builders.build_sn_polym_sto3g`
(multiplicity=2 doublet placeholder).
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_sn_polym_sto3g


def main() -> int:
    system, basis = build_sn_polym_sto3g()
    print("=== (SN)x polymer STO-3G parity STUB ===")
    print(f"  geometry: dim={system.dim}, n_atoms={len(system.unit_cell)}, "
          f"n_elec={system.n_electrons()}, mult={system.multiplicity}")
    print(f"  basis: STO-3G ({basis.nbasis} BFs / {basis.nshells} shells)")
    print()
    print("  BLOCKED: dim=1 needs linear-truncation cutoff + open-shell "
          "metallic SCF (UHF + Fermi-Dirac smearing).")
    return 3


if __name__ == "__main__":
    sys.exit(main())
