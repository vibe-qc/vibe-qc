"""Parity: silicon (Si) bulk RHF/STO-3G at SHRINK 8 8 via the BIPOLE driver.

CRYSTAL14 reference (local CRYSTAL23-demo run on M5 Max):
  - SG 227 diamond-structure, a=5.42 Å, Si @ Wyckoff 8a origin-2
    (2 atoms / cell)
  - STO-3G inline (14 1 0 3 + 14 1 1 3 + 14 1 1 3)
  - SHRINK 8 8, FMIXING 30, TOLDEE / TOLINTEG defaults
  - 6 cycles, E_total = -571.32081220815 Ha/cell

Current vibe-qc analytic-V_ne Ewald-J result (cutoff 14, SHRINK 8):
  -571.321471579800 Ha/cell (Δ ≈ -0.659 mHa)
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_sibulk_sto3g
from crystal_demos.runner import run_demo_parity


def main() -> int:
    return run_demo_parity(
        label="silicon bulk (STO-3G, SHRINK 8 8)",
        build_fn=build_sibulk_sto3g,
        kmesh_size=(8, 8, 8),
        crystal14_ref_ha_per_fu=-571.32081220815,
        cutoff_bohr=14.0,
        use_diis=True,
        diis_start_iter=2,
        damping=0.3,
        max_iter=12,
        target_millihartree=1.0,
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        conv_tol_energy=1e-7,
        conv_tol_grad=1e-4,
        use_symmetry=False,
    )


if __name__ == "__main__":
    sys.exit(main())
