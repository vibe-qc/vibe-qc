"""Parity: MgO primitive RKS LDA(SVWN)/STO-3G at SHRINK 2 via the BIPOLE driver.

Part of the dense-ionic-crystal DFT parity ladder
(``parity_mgo_{svwn,pbe,r2scan}_sto3g.py``). See the full definition-parity
audit in ``parity_mgo_r2scan_sto3g.py`` and memory
``periodic-mgga-parity-reference-choice``.

Sealed CRYSTAL23 reference:
  - SG 225 rocksalt, a=4.21 Å, Mg(0,0,0)+O(½,½,½), STO-3G internal lib
  - DFT SVWN (Slater exchange + VWN5 correlation) / XLGRID, SHRINK 2 2,
    TOLINTEG 7 7 7 7 14, TOLDEE 8 — 7 cycles, E = -270.49836634642 Ha/FU (sealed).
  vibe-qc ``functional="lda"`` is the SAME pair: XC_LDA_X + XC_LDA_C_VWN (VWN5),
  0 % EXX (cpp/src/xc.cpp:146) — apples-to-apples definition match.

CROSS-CELL PERIODIC-XC P0 — FIXED in v0.14.x:
  Pre-fix, vibe-qc converged to E = -270.07046 Ha/FU (Δ = +427.9 mHa) because
  ``build_xc_periodic`` assembled the grid density with the bra AO anchored in
  the home cell (ρ_code = Σ_s χ_0 P(s) χ_s) — NOT lattice-periodic, pointwise
  wrong by O(1) in the bonding region, yet still integrating to N electrons so
  ∫ρ dr could not see it. The fix sums BOTH AOs over lattice cells
  (ρ = Σ_a Σ_s χ_a P(s−a) χ_s); the assembly is pinned directly in
  ``tests/test_periodic_xc_cross_cell.py`` (C++ vs an independent NumPy oracle,
  functional-resolved, 1e-7). With the fix this converges cleanly (11 iters,
  monotonic) to E ≈ -270.50226 Ha/FU, Δ ≈ -3.9 mHa (was +427.9 mHa).

REMAINING RESIDUAL (~3.9 mHa — a SEPARATE, smaller issue, NOT the cross-cell
bug, so the ±2 mHa target below is not yet met):
  Characterized 2026-06-24: the bra/ket screen cutoff and the radial AND
  angular DFT grids are all converged (vibe-qc E_xc shifts < 0.01 mHa from the
  default 17×36 product grid to Lebedev-974). PySCF.pbc KRKS-LDA GDF is
  grid-converged at -270.4993 Ha (−0.94 mHa from CRYSTAL; levels 3/5/7 within
  0.07 mHa). vibe-qc sits ~2.9 mHa below that grid-converged PySCF value — a
  difference in the BIPOLE Ewald-J / exxdiv Coulomb gauge of the pure-DFT
  (no-exchange) RKS path, plus the documented cutoff-14 S(k)-fold lattice
  truncation (~0.8 mHa). RHF on this cell matches CRYSTAL to 0.369 mHa because
  its J+K exchange-divergence cancellation is absent in the J-only RKS path.
  Tracked separately from the (now-fixed) cross-cell P0.

Exit code: 0 = PASS (Δ within target), 1 = converged off-target (currently:
the separate ~3.9 mHa gauge/truncation residual), 2 = SCF did not converge.
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
        label="MgO primitive (LDA/SVWN/STO-3G, SHRINK 2 2)",
        build_fn=build_mgo_sto3g,
        functional="lda",
        kmesh_size=(2, 2, 2),
        crystal_ref_ha_per_fu=-270.49836634642,
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
