"""Parity STUB: Be bulk RHF/STO-3G at SHRINK 12 24 (HCP, metallic).

CRYSTAL14 reference (``crystal_demos/be_sto3g.d12``):
  - SG 194 (HCP P6_3/mmc), a=2.29 Å, c=3.59 Å, Be @ Wyckoff 2c
    (2 atoms / cell)
  - STO-3G inline
  - SHRINK 12 24 (denser k-mesh for Fermi-surface integration)
  - FMIXING 30, MAXCYCLE 130

**Blocker**: bulk Be is metallic (partially occupied bands at E_F).
``run_pbc_bipole_rhf`` uses fixed integer occupations (n_occ doublet-
filled) which cannot describe the partially occupied Fermi surface.
Activating this parity test requires Fermi-Dirac smearing + adaptive
occupations + chemical-potential bisection at each iter — a metallic
SCF aid stack that's not yet on the BIPOLE driver. Geometry verified
by :func:`crystal_demos.builders.build_be_sto3g`.
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_be_sto3g


def main() -> int:
    system, basis = build_be_sto3g()
    print("=== Be bulk STO-3G (metallic HCP) parity STUB ===")
    print(f"  geometry verified: dim={system.dim}, n_atoms="
          f"{len(system.unit_cell)}, n_elec={system.n_electrons()}")
    print(f"  basis: STO-3G ({basis.nbasis} BFs / {basis.nshells} shells)")
    print()
    print("  BLOCKED: Be bulk is metallic — fixed integer occupations "
          "in run_pbc_bipole_rhf cannot represent the partially-occupied "
          "Fermi surface.")
    print("  Action: add Fermi-Dirac smearing + chemical-potential "
          "bisection + adaptive per-k occupations to BIPOLE driver.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
