"""Parity STUB: graphite (single layer = graphene) RHF/STO-3G at SHRINK 8 16.

CRYSTAL14 reference (``crystal_demos/graphite_sto3g.d12``):
  - 2D ``SLAB`` p6/mmm-equivalent, a=2.47 Å, 2-atom honeycomb
  - STO-3G inline (6 1 0 3 + 6 1 1 3)
  - SHRINK 8 16

**Blockers**:
  1. ``run_pbc_bipole_rhf`` currently exercised only at ``dim=3``;
     2D direct-space cutoff handling (cylindrical-shell vs spherical-
     shell summation) needs verification.
  2. Graphene is **semi-metallic** at the Dirac point; fixed occupations
     in the driver may need adjustment for k-meshes that include K-point.
Geometry verified by :func:`crystal_demos.builders.build_graphite_sto3g`
(2-atom honeycomb expansion of the CRYSTAL single-Wyckoff input).
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_graphite_sto3g


def main() -> int:
    system, basis = build_graphite_sto3g()
    print("=== graphite (2D graphene) STO-3G parity STUB ===")
    print(f"  geometry: dim={system.dim}, n_atoms={len(system.unit_cell)}, "
          f"n_elec={system.n_electrons()}")
    print(f"  basis: STO-3G ({basis.nbasis} BFs / {basis.nshells} shells)")
    print()
    print("  BLOCKED: BIPOLE is a bulk (dim=3) route. As of the 2D-slab "
          "convention change it must not be used on a dim=2 slab.")
    print("  Action: run this parity check through the vacuum-free 2D "
          "gauge instead: run_periodic_job(..., jk_method='auto') on a "
          "dim=2 system resolves to SLAB_EWALD_2D.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
