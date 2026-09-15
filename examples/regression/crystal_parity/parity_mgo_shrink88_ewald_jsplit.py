"""Parity: MgO primitive RHF/STO-3G at SHRINK 8 8 via multi-k Ewald J-split.

Multi-k extension of ``parity_mgo_gamma_ewald_jsplit.py`` at full
SHRINK 8 8 — same k-mesh CRYSTAL used for its sealed reference. This
is the headline-number target of the v0.9.0 multi-k Ewald-J-split
extension: closing the +1.88 Ha Γ-vs-converged-k-mesh gap on MgO.

CRYSTAL14 reference (SHRINK 8 8 converged):
  -271.218144 Ha/FU

Current vibe-qc analytic-V_ne result (cutoff 14, ewald_precision=1e-6):
  -271.217775 Ha/FU  (Δ ≈ +0.37 mHa)

Pass target: ±1 mHa on this MgO sign-off case.
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_mgo_sto3g
from crystal_demos.runner import run_demo_parity


def main() -> int:
    # ``use_symmetry=False`` is REQUIRED at multi-k for the Ewald-J-split
    # path — see ``parity_mgo_shrink22_ewald_jsplit.py`` for the reason.
    return run_demo_parity(
        label="MgO primitive (STO-3G, SHRINK 8 8, Ewald J-split + DIIS)",
        build_fn=build_mgo_sto3g,
        kmesh_size=(8, 8, 8),
        crystal14_ref_ha_per_fu=-271.21814374982,
        cutoff_bohr=14.0,
        use_diis=True,
        diis_start_iter=2,
        damping=0.3,
        max_iter=25,
        target_millihartree=1.0,
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        conv_tol_energy=1e-7,
        conv_tol_grad=1e-4,
        use_symmetry=False,
    )


if __name__ == "__main__":
    sys.exit(main())
