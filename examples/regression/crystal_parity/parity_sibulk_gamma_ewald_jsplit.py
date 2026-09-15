"""Parity: silicon bulk RHF/STO-3G at Γ-only via Ewald J-split.

CRYSTAL14 reference (sealed; SHRINK 8 8 converged):
  -571.320812 Ha/cell (2 Si/cell)

Same Ewald J-split path as parity_mgo_gamma_ewald_jsplit.py.
Largest of the three RUNNABLE STO-3G demos (28 electrons,
3rd-row atoms with diffuse 3sp basis).

vibe-qc empirical result (cutoff=12, DIIS, 2026-05-18):
  iter  1 E = -569.625 Ha   (close to CRYSTAL CYC 0 within ~mHa)
  iter  8 E = -570.876 Ha   (CONVERGED with DIIS)
  Δ vs CRYSTAL SHRINK 8 8 = +444 mHa ✓ PASS

Si has the BEST Γ-vs-SHRINK 8 8 parity of the three RUNNABLE
demos (vs MgO +1.88 Ha, diamond +2.53 Ha) — Si's larger lattice
constant gives less band dispersion than diamond, and Si is less
ionic than MgO. Even without DIIS Si converged in 7 iter (the
SCF landscape is intrinsically smoother); DIIS adds ~1 iter
robustness margin.
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
        label="silicon bulk (STO-3G, Γ-only, Ewald J-split + DIIS)",
        build_fn=build_sibulk_sto3g,
        kmesh_size=(1, 1, 1),
        crystal14_ref_ha_per_fu=-571.32081220815,
        cutoff_bohr=12.0,
        use_diis=True,
        diis_start_iter=2,
        damping=0.3,
        max_iter=15,
        target_millihartree=2200.0,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        conv_tol_energy=1e-7,
        conv_tol_grad=1e-4,
    )


if __name__ == "__main__":
    sys.exit(main())
