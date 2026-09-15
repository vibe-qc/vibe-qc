"""Parity: MgO primitive RHF/STO-3G at SHRINK 8 8 via the BIPOLE driver.

CRYSTAL14 reference (local run on M5 Max, ``crystal_demos/mgo_sto3g.d12``):
  - SG 225 rocksalt, a=4.21 Å, Mg(0,0,0) + O(½,½,½), STO-3G inline
  - SHRINK 8 8, TOLDEE 6, TOLINTEG 6 6 6 6 12 (CRYSTAL defaults)
  - 7 cycles, E_total = -271.21814374982 Ha/FU (sealed)
  - canonical CRYSTAL community "first run" test
  - Per-component breakdown via sealed ENECYCLE run:
      KINETIC          = +268.018 Ha
      TOTAL E-N + N-E  = -511.818 Ha  (Ewald)
      TOTAL E-E        =   +46.211 Ha (after EXT EL-POLE subtraction)
      BIELET ZONE E-E  =  +570.696 Ha (raw direct-zone analogue)
      TOTAL N-N        =   -73.084 Ha  (Ewald)

Current vibe-qc analytic-V_ne Ewald-J result (cutoff 14, SHRINK 8):
  -271.217774850852 Ha/FU (Δ ≈ +0.369 mHa)

The driver now defaults to the CRYSTAL-gauge Ewald-J two-electron
composition for 3D systems. Pass ``use_ewald_j_split=False`` only for
the legacy direct-only diagnostic path described in the 2026-05-17
BIPOLE handover.
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_mgo_sto3g
from crystal_demos.runner import run_demo_parity


def main() -> int:
    return run_demo_parity(
        label="MgO primitive (STO-3G, SHRINK 8 8)",
        build_fn=build_mgo_sto3g,
        kmesh_size=(8, 8, 8),
        crystal14_ref_ha_per_fu=-271.21814374982,
        cutoff_bohr=14.0,
        use_diis=True,
        diis_start_iter=2,
        damping=0.3,
        max_iter=12,
        target_millihartree=1.0,
        ewald_precision=1e-6,
        conv_tol_energy=1e-7,
        conv_tol_grad=1e-4,
    )


if __name__ == "__main__":
    sys.exit(main())
