"""Parity: MgO primitive RKS PBE/STO-3G at SHRINK 2 via the BIPOLE driver.

Part of the dense-ionic-crystal DFT parity ladder
(``parity_mgo_{svwn,pbe,r2scan}_sto3g.py``). See ``parity_mgo_svwn_sto3g.py``
for the cross-cell P0 fix summary and the residual characterization, and
``parity_mgo_r2scan_sto3g.py`` for the definition-parity audit.

Sealed CRYSTAL23 reference:
  - SG 225 rocksalt, a=4.21 Å, Mg(0,0,0)+O(½,½,½), STO-3G internal lib
  - DFT PBE / XLGRID, SHRINK 2 2, TOLINTEG 7 7 7 7 14, TOLDEE 8 — 7 cycles,
    E = -271.76561383103 Ha/FU (sealed). vibe-qc ``functional="pbe"`` drives the
    matching libxc GGA_X_PBE + GGA_C_PBE pair, 0 % EXX.

CROSS-CELL PERIODIC-XC P0 — FIXED in v0.14.x:
  PBE is the GGA rung of the functional-independent dense-crystal periodic-XC
  bug. Pre-fix it could not even converge — it limit-cycled around E ≈ -271.367
  Ha/FU (~+399 mHa, ||[F,DS]|| stuck ~1e-2, dE sign-flipping). The wrong
  cross-cell density (bra anchored home in ``build_xc_periodic``) starved the
  GGA σ Fock terms of a self-consistent density. The fix (both AOs over lattice
  cells, ρ = Σ_a Σ_s χ_a P(s−a) χ_s) restores convergence: PBE now descends
  monotonically and cleanly converges. The σ-resolved density assembly is
  pinned in ``tests/test_periodic_xc_cross_cell.py`` (pbe case, 1e-7).

REMAINING RESIDUAL (a SEPARATE, smaller issue, NOT the cross-cell bug — see
``parity_mgo_svwn_sto3g.py``): vibe-qc's converged BIPOLE energy sits a few mHa
below grid-converged PySCF.pbc GDF, a BIPOLE Ewald-J / exxdiv pure-DFT Coulomb
gauge difference + cutoff-14 lattice truncation. Tracked separately; this is
why the ±2 mHa target below is not yet met.

Exit code: 0 = PASS, 1 = converged off-target (the separate gauge residual),
2 = SCF did not converge.
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_mgo_sto3g
from crystal_demos.runner import run_demo_parity_rks


def main() -> int:
    return run_demo_parity_rks(
        label="MgO primitive (PBE/STO-3G, SHRINK 2 2)",
        build_fn=build_mgo_sto3g,
        functional="pbe",
        kmesh_size=(2, 2, 2),
        crystal_ref_ha_per_fu=-271.76561383103,
        cutoff_bohr=14.0,
        use_diis=True,
        diis_start_iter=2,
        damping=0.3,
        max_iter=50,
        use_symmetry=False,
        target_millihartree=2.0,
        ewald_precision=1e-8,
        conv_tol_energy=1e-7,
        conv_tol_grad=1e-4,
    )


if __name__ == "__main__":
    sys.exit(main())
