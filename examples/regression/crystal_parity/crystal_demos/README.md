# CRYSTAL14 / CRYSTAL23 demo inputs

Reference test systems for periodic SCF parity, downloaded from the
CRYSTAL evaluation-version "Try It" page:
https://www.crystalsolutions.eu/try-it.html

These are the canonical "does your periodic SCF work" systems used
by the CRYSTAL community for first-time setup verification. They
were chosen for v0.8.0 BIPOLE parity testing after the lesson
(2026-05-17) that anchoring on LiH primitive @ small kmesh was
methodologically wrong — LiH/STO-3G at SHRINK 2 2 / 4 4 is
intrinsically unconvergeable in CRYSTAL itself; the demo systems
here all converge smoothly in CRYSTAL at their stated kmeshes.

## CRYSTAL23 demo binaries

`crystal23demo` and `properties23demo` are available in `~/bin` on:
- this M5 Max (local; `~/bin/crystal23demo`)
- compute-small
- compute-reference (dispatchable via `vq submit` + `contrib/run-crystal.sh`)

All functionalities of the full version are active; the only limit
is **≤ 10 atoms per primitive cell**. All systems below satisfy that.

## Test systems

### Minimal basis (STO-3G)

| File | System | Geom | Spec |
|---|---|---|---|
| `be_sto3g.d12`         | Beryllium bulk      | HCP P6₃/mmc (SG 194), a=2.29 Å, c=3.59 Å, Be at (1/3, 2/3, 1/4); 2 atoms | SHRINK 12 24, MAXCYCLE 130, FMIXING 30 |
| `mgo_sto3g.d12`        | MgO bulk            | FCC rocksalt SG 225, a=4.21 Å, Mg(0,0,0)+O(½,½,½); 2 atoms | SHRINK 8 8 |
| `nio_sto3g.d12`        | NiO bulk            | FCC rocksalt SG 225, a=4.164 Å; UHF, SPINLOCK, LEVSHIFT 3 1, FMIXING 30 | SHRINK 8 8 |
| `diamond_sto3g.d12`    | Diamond C           | FCC diamond SG 227, a=3.57 Å, C at (1/8,1/8,1/8); 2 atoms in prim | SHRINK 8 8 |
| `sibulk_sto3g.d12`     | Silicon bulk        | FCC diamond SG 227, a=5.42 Å, Si at (1/8,1/8,1/8); 2 atoms in prim | SHRINK 8 8, FMIXING 30 |
| `graphite_sto3g.d12`   | Graphite monolayer  | 2D SLAB, p6/mmm (SG 77), a=2.47 Å, C at fractional 2D coords | SHRINK 8 16 |
| `sn_polym_sto3g.d12`   | (SN)x polymer       | 1D POLYMER, a=4.431 bohr, S+N atoms | SHRINK 13 13 |

### Dense-ionic DFT parity (MgO — sealed CRYSTAL23)

vibe-qc-authored DFT variants of the MgO/STO-3G cell. All three are the same
SG 225 rocksalt cell at **SHRINK 2 2** / XLGRID / TOLINTEG 7 7 7 7 14 /
TOLDEE 8; the CRYSTAL23 `.out` is the sealed reference (run out-of-process per
CLAUDE.md §10 — never imported). **k-setting:** SHRINK 2 2 (IS=2) corresponds to
the vibe-qc Monkhorst-Pack mesh **(2,2,2)** (`kmesh_size=(2,2,2)`,
`use_symmetry=False` in the `parity_mgo_<func>_sto3g.py` scripts) — NOT Γ. The
sealed energies below are only reproducible at SHRINK 2 / (2,2,2); a Γ-only run
gives a materially different energy.

**Status (post-v0.14.x):** the large functional-independent gap these cells were
built to localize was the **cross-cell periodic-XC density P0** (`build_xc_periodic`
anchored the bra AO in the home cell → ρ not lattice-periodic) — now **FIXED**
(ρ = Σ_a Σ_s χ_a P(s−a) χ_s; pinned
in `tests/test_periodic_xc_cross_cell.py`, C++-vs-NumPy oracle, 1e-7). The
parity scripts went from ~+400 mHa to a few mHa.

| File | Functional | CRYSTAL23 E (Ha/FU) | pre-fix Δ | post-fix Δ |
|---|---|---|---|---|
| `mgo_svwn_sto3g.d12`   | SVWN / LDA (Slater+VWN5) | -270.49836634642 | +427.9 mHa | **-3.9 mHa** |
| `mgo_pbe_sto3g.d12`    | PBE (GGA)                | -271.76561383103 | ~+399 mHa (oscillated) | a few mHa (now convergent) |
| `mgo_r2scan_sto3g.d12` | r2SCAN (meta-GGA)        | -271.92903755119 | +384.1 mHa | a few mHa |

**The remaining residual is a SEPARATE, smaller, still-open issue** (so the
±2 mHa target is not yet met): vibe-qc's converged BIPOLE RKS-DFT energy sits a
few mHa below grid-converged PySCF.pbc GDF (LDA: vibe-qc -270.5023 vs PySCF
-270.4993, both grid-converged) — the BIPOLE Ewald-J / exxdiv **pure-DFT
(no-exchange) Coulomb gauge** + the cutoff-14 S(k)-fold truncation (~0.8 mHa),
NOT the cross-cell density and NOT the grid. RHF on the same cell matches
CRYSTAL to 0.369 mHa (its J+K exchange-divergence cancellation is absent in the
J-only RKS path; `parity_mgo_primitive_shrink88.py`). Historically tracked in
the retired v0.14 bug inventory P1. **NB the "~1 mHa" figure is CRYSTAL23 ↔ PySCF.pbc
GDF agreement (two external references), not vibe-qc ↔ CRYSTAL.** See
`parity_mgo_r2scan_sto3g.py` (definition-parity audit) and memory
`periodic-mgga-parity-reference-choice` for the full localization.

### Extended basis (POB / Pople 6-21G with polarization / atom-specific)

| File | System | Notes |
|---|---|---|
| `bebulk.d12`           | Beryllium bulk      | extended s-only basis (4 contracted shells) |
| `mgo_bulk.d12`         | MgO bulk            | 8411G(Mg) / 8511G(O) extended basis |
| `mgo001.d12`           | MgO (001) surface   | 1-layer slab via `SLABCUT` |
| `mgo001co.d12`         | CO on MgO (001)     | 1-layer slab + adsorbed CO molecule (uses ghost-atom Z=108 for the molecular oxygen) |
| `nio.d12`              | NiO bulk            | extended basis, UHF, SPINLOCK 2 15, LEVSHIFT 3 1, FMIXING 30 |
| `diamond.d12`          | Diamond C           | 6-21G + d-polarization |
| `sibulk.d12`           | Silicon bulk        | 6-21G modified (4 contracted shells) |
| `urea_bulk_321G.d12`   | Urea bulk           | molecular crystal, SG 113 (P-42₁m), 3-21G basis, 5 atoms (C, O, N, 2H) |

## Recommended parity test order

1. **`mgo_sto3g.d12`** — first target. Smooth 10-cycle SCF in CRYSTAL14 to -271.21814 Ha/FU. Clean valence structure, no near-degeneracies. The standard CRYSTAL community "first run" system.
2. `diamond_sto3g.d12` — covalent bonding, also clean SCF.
3. `sibulk_sto3g.d12` — larger covalent, tests basis size scaling.
4. `mgo_bulk.d12` (extended basis MgO) — moves to extended basis.
5. `nio_sto3g.d12` — first UHF parity test (requires spin polarization).
6. `bebulk.d12` — metallic (Be), tests Fermi-energy bisection.
7. `graphite_sto3g.d12` — first 2D-periodic test.
8. `urea_bulk_321G.d12` — molecular crystal, larger primitive cell.

Skip Be / NiO until the closed-shell insulator path is solid; they
add UHF + smearing complications on top of the basic BIPOLE
periodic SCF.
