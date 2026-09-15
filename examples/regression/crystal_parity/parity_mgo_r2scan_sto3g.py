"""Parity: MgO primitive RKS r2SCAN/STO-3G at SHRINK 2 via the BIPOLE driver.

*** Cross-cell periodic-XC P0 FIXED in v0.14.x; a separate ~mHa residual
    remains (see the BUG STATUS section below). ***

Sealed CRYSTAL23 reference (local run):
  - SG 225 rocksalt, a=4.21 Å, Mg(0,0,0) + O(½,½,½), STO-3G internal lib
  - DFT R2SCAN / XLGRID, SHRINK 2 2, TOLINTEG 7 7 7 7 14, TOLDEE 8
  - 8 cycles, E_total = -271.92903755119 Ha/FU  (DE 2.3E-13, sealed)
  - E_xc = E_exch + E_corr = -24.5376659543 + -0.7302681461
         = -25.26793410035 Ha  (sealed)
  - indirect band gap 3.20 eV (direct 6.34 eV) — a wide-gap insulator.
Independent cross-check: PySCF KRKS r2SCAN .density_fit() (GDF) gives
-271.93006737 Ha (~1.0 mHa from CRYSTAL23). Two independent Gaussian-basis
periodic codes agree on the reference; vibe-qc does not.

This is the heavy cross-code half of the periodic meta-GGA / CRYSTAL-parity
validation (CLAUDE.md §10 — CRYSTAL run strictly out-of-process, sealed
reference, no QC-program import). The fast pytest guard
(``tests/test_periodic_xc_mgga.py::test_r2scan_multik_gdf_molecular_limit``)
covers the multi-k Bloch + meta-GGA τ/Vτ path on a *dilute* H2 cell; this
script is the *dense-ionic-crystal* accuracy check, where a different,
currently-open bug shows up.

===========================================================================
DEFINITION-PARITY AUDIT (2026-06-22) — done BEFORE calling this a §7 bug.
===========================================================================
A 384 mHa gap is far too large to be a functional-definition or grid
artifact, but per the standing discipline ("verify the definition, not
just the number") all three axes were pinned before localizing the bug to
the periodic path. A definition-matched, grid-converged comparison still
disagrees → it is a genuine periodic-RKS bug, not a parity-setup mistake.

1. FUNCTIONAL DEFINITION — matched.
   * vibe-qc "r2scan" resolves (cpp/src/xc.cpp:395) to the libxc pair
       XC_MGGA_X_R2SCAN (id 497, w=1.0) + XC_MGGA_C_R2SCAN (id 498, w=1.0),
     hf_exchange_fraction = 0.0, kind = MGGA, is_hybrid = False
     (verified live via vibeqc.Functional("r2scan")). r²SCAN of Furness,
     Kaplan, Ning, Perdew, Sun, JPCL 11, 8208 (2020).
   * CRYSTAL ran its NATIVE r2SCAN, not libxc: the .out reports
       "(EXCHANGE)[CORRELATION] FUNCTIONAL:(r2SCAN)[r2SCAN]"  and
       "ENERGY EXPRESSION=HARTREE+FOCK EXCH*0.00000+(r2SCAN EXCH)*1.0+r2SCAN CORR"
     — confirmed r²SCAN (not SCAN / rSCAN), same generation, 0 % HF.
   * Native-vs-libxc gap is bracketed at ~1 mHa by the PySCF .density_fit()
     cross-check (PySCF drives the SAME libxc 497/498 pair and lands 1.0 mHa
     from CRYSTAL-native). And vibe-qc's own libxc r²SCAN matches PySCF's to
     ~µHa at the molecular level (tests/test_xc.py:233, H2O/def2-SVP,
     -76.3118 Ha, tol 5e-4). So CRYSTAL-native ≈ libxc ≈ vibe-qc all within
     ~1 mHa: the functional definition is NOT the source of the 384 mHa.

2. EXX FRACTION / VWN VARIANT — not applicable, and matched.
   r²SCAN is a pure meta-GGA: 0 % exact exchange on both sides (vibe-qc
   hf_exchange_fraction 0.0; CRYSTAL FOCK EXCH*0.00000) and it carries no
   VWN-LDA component, so the VWN3/VWN5 split that bites B3LYP-type hybrids
   does not arise here. (When a hybrid — e.g. r2scan0/B3LYP — is added to
   this suite, pin its EXX fraction AND VWN variant on both sides here.)

3. INTEGRATION GRID — comparable; grid-converged delta reported separately.
   * CRYSTAL: input XLGRID, auto-upgraded to XXXLGRID for the SCAN family
     (.out lines 256/261): 75 radial points × pruned Lebedev up to 974
     angular (SIZE OF GRID 1292; .out lines 377/383–387).
   * vibe-qc default periodic Becke grid: n_radial=75, n_theta=17, n_phi=36
     (612-point product angular), lebedev_order=29 — comparable radial,
     somewhat lighter angular.
   * r²SCAN was re-regularized specifically to be grid-insensitive: vibe-qc
     reproduces PySCF (grids.level=5, fine) on the DEFAULT medium grid to
     <0.5 mHa molecularly (tests/test_xc.py:225–233; SCAN by contrast needs
     a 2e-3 tol). Measured directly on THIS periodic system, the SAD-density
     iter-1 energy (pure E_xc grid effect, density held fixed) moves by only
       default grid (n_radial 75, 17×36)  E_iter1 = -271.23198846 Ha
       dense   grid (n_radial 99, Leb 41) E_iter1 = -271.23203978 Ha
       grid shift = 0.051 mHa  —  ~7500× smaller than the 384 mHa gap.
     (Reproduce by setting opts.grid.n_radial / n_theta / n_phi / lebedev_order
     on the run_demo_parity_rks recipe.) The grid neither hides nor causes the
     discrepancy: the grid-converged delta is the same +384 mHa.

Conclusion: with the functional definition matched (libxc 497/498, 0 % EXX,
both sides) and the comparison grid-converged, vibe-qc's PERIODIC MgO r2SCAN
SCF still lands 384 mHa high. The bug is fenced in by what already matches:
  - RHF analog matches CRYSTAL to 0.369 mHa  → J / Hcore / Ewald / gauge OK;
  - molecular r2SCAN matches PySCF to µHa     → libxc functional OK;
  - dilute-periodic r2SCAN matches molecular  → periodic XC OK once there is
    NO cross-cell density (no P(g≠0) overlap);
  - no-smearing run is a true insulator       → not a smearing/occupation bug.

The bug is FUNCTIONAL-INDEPENDENT (measured 2026-06-22): the same no-smearing
BIPOLE ladder on this exact MgO/STO-3G [2,2,2] cell, each vs its own sealed
CRYSTAL23 reference, gives LDA/SVWN +427.9 mHa, PBE ~+399 mHa, r2SCAN
+384.1 mHa. LDA is ρ-only (no σ, no τ), so the common ~400 mHa error is NOT
in the σ/τ (GGA/mGGA) Fock terms or in V_τ — it is in the periodic XC piece
EVERY functional shares: the cross-cell density ρ(r) assembled on the periodic
Becke grid for dense (overlapping-image) crystals (``build_xc_periodic``). RHF
(no grid density) matches; dilute cells (no cross-cell P(g≠0)) match molecular.
So the trigger is the non-negligible cross-cell density blocks feeding the
periodic grid evaluation — a genuine §7 periodic-RKS bug, NOT a definition,
grid, or meta-GGA-specific mismatch. (First probes for the fixer: integrate
∫ρ(r)dr on the periodic Becke grid for the converged dense density — should be
N_elec=20 — and sweep ``becke_image_radius_bohr``. See memory
``periodic-mgga-parity-reference-choice`` for the full ladder + a separate,
sub-µHa V_τ non-conservation follow-up that is NOT this cause.)
===========================================================================

BUG STATUS (updated 2026-06-24):
  The +384 mHa gap the definition-parity audit above fenced in was the
  functional-independent dense-crystal periodic-XC cross-cell density bug:
  ``build_xc_periodic`` anchored the bra AO in the home cell (ρ_code = Σ_s χ_0 P(s) χ_s), which is not lattice-periodic and
  is pointwise wrong by O(1) in the bonding region of dense crystals, yet still
  integrates to N electrons. This is now FIXED in v0.14.x — the grid density
  sums BOTH AOs over lattice cells (ρ = Σ_a Σ_s χ_a P(s−a) χ_s), pinned
  functional-resolved (incl. the meta-GGA τ path) in
  ``tests/test_periodic_xc_cross_cell.py`` (C++ vs an independent NumPy oracle,
  1e-7). r2SCAN now converges cleanly much closer to CRYSTAL (LDA went
  +427.9 → -3.9 mHa; PBE went from non-convergent to convergent).

  A SEPARATE, much smaller residual remains (the ±2 mHa target is not yet met):
  vibe-qc's converged BIPOLE energy sits a few mHa below grid-converged
  PySCF.pbc GDF (LDA: vibe-qc -270.5023 vs PySCF -270.4993, both grid-
  converged). It is NOT the cross-cell density and NOT the grid (vibe-qc E_xc
  shifts < 0.01 mHa from the 17×36 product grid to Lebedev-974; PySCF is grid-
  converged to 0.07 mHa over levels 3/5/7). It is the BIPOLE Ewald-J / exxdiv
  pure-DFT (no-exchange) Coulomb gauge + the documented cutoff-14 lattice
  truncation — the RHF analog (``parity_mgo_primitive_shrink88.py``) matches
  CRYSTAL to 0.369 mHa precisely because its J+K exchange-divergence
  cancellation is absent in the J-only RKS path. Tracked separately.

Exit code: 0 = PASS (matches CRYSTAL within tol), 1 = converged off-target
(currently: the separate gauge/truncation residual), 2 = SCF did not converge.
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, "examples/regression/crystal_parity")
from crystal_demos.builders import build_mgo_sto3g
from crystal_demos.runner import run_demo_parity_rks


def main() -> int:
    # functional="r2scan" resolves to libxc XC_MGGA_X_R2SCAN (id 497) +
    # XC_MGGA_C_R2SCAN (id 498), 0 % EXX (cpp/src/xc.cpp:395) — the SAME
    # pair PySCF drives. CRYSTAL ran its native r2SCAN (0 % HF); the two
    # definitions agree to ~1 mHa (see the DEFINITION-PARITY AUDIT above).
    # Sealed CRYSTAL23 target: E_total = -271.92903755119 Ha/FU,
    # E_xc = -25.26793410035 Ha (XXXLGRID, 75 radial × Lebedev≤974).
    return run_demo_parity_rks(
        label="MgO primitive (r2SCAN/STO-3G, SHRINK 2 2)",
        build_fn=build_mgo_sto3g,
        functional="r2scan",
        kmesh_size=(2, 2, 2),
        crystal_ref_ha_per_fu=-271.92903755119,
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
