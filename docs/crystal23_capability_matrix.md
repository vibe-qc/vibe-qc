# CRYSTAL23 capability matrix

This page is the explicit feature and configuration matrix of the PBC
CRYSTAL23 parity programme (tracker issue #593). It lists the ground-state
capabilities that the CRYSTAL23 User's Manual, Section 1.2 "CRYSTAL23 Program
Features", and Erba et al., *J. Chem. Theory Comput.* **19**, 6891 (2023),
[DOI 10.1021/acs.jctc.2c00958](https://doi.org/10.1021/acs.jctc.2c00958),
advertise, and records for each one what vibe-qc has established on `main`.
It is an inventory of evidence classes, not a parity claim: a row can only be
promoted by the acceptance named in its "remaining gap" column, and a passed
bounded example never promotes an entire method or backend.

The machine-readable source is
[`benchmarks/crystal23_capability_matrix.json`](../benchmarks/crystal23_capability_matrix.json).
The tables below are generated from it by
`scripts/crystal23_capability_matrix.py --write`; `--check` fails when the
page and the JSON disagree, and `tests/test_crystal23_capability_matrix.py`
runs that check together with the schema rules. Edit the JSON, not the
tables.

## How to read the states

Every row carries exactly one state, the strongest evidence class achieved
for the bounded scope named in its evidence column:

* `numerically-accepted`: a matched reference comparison passed on a recorded
  source and native runtime within a stated tolerance. The evidence names the
  system, reference, tolerance and runtime. This is the only state that counts
  toward parity, and only for the scope it names.
* `source-tested`: implemented and covered by focused repository tests of
  source functionality, refusals or internal identities, without a matched
  external reference.
* `implemented`: a public route exists and runs, with no focused test or
  reference comparison claimed here.
* `unverified`: implemented, but an open numerical defect, a failed or
  contradictory comparison, or a missing reference comparison is on record.
  The evidence column states the recorded number and its runtime.
* `unsupported`: absent or explicitly refused. A precise refusal is an
  acceptable release boundary; it is not parity.

Each row links the tracker issues that own its remaining gap. Required rows
that stay `unverified` or `unsupported` are blockers for #593, not completed
work. The narrower 0.18 release inventory lives in
[PBC 0.18 supported configurations](user_guide/pbc_018_supported_configurations.md)
and is not widened by this page.

## Acceptance criteria shared by the rows

* **Matched inputs.** Lattice orientation, positions, charge and spin,
  primitive and contraction data, ECP, functional definition, grid,
  occupation objective, k coordinates and weights, integral precision,
  exchange zero-mode convention and SCF residual are matched and recorded
  (#573, #645, #647, #648). SHRINK dimensions alone do not match a mesh.
* **Separated error sources.** Basis, fitting or rank, image support, grid,
  BZ sampling, finite-size exchange and SCF termination errors are reported
  separately. Tolerances come from convergence ladders and reference
  evidence, never from one passing fixture.
* **Recorded identity.** Source commit, native build identity, inputs,
  measured results, reference provenance, errors and cost are recorded on
  the issue; operational receipts stay outside the product checkout.
* **Performance claims** need same-accuracy paired full-SCF timings with
  setup, final confirmation, memory and uncertainty (#574); kernel speedups
  do not establish time-to-solution parity.

## Matrix

<!-- matrix:begin -->

Updated 2026-10-24; 61 rows.

| State | Rows | Meaning |
|---|---:|---|
| `numerically-accepted` | 5 | A matched reference comparison passed on a recorded source/native runtime within a stated tolerance, for the bounded scope named in the evidence; not whole-method parity. |
| `source-tested` | 21 | Implemented and covered by focused repository tests of source functionality, refusals or internal identities; no matched external numerical reference. |
| `implemented` | 7 | A public route exists and runs; no focused repository test or reference comparison is claimed for this row. |
| `unverified` | 15 | Implemented, but an open numerical defect, a failed or contradictory comparison, or a missing reference comparison is on record. |
| `unsupported` | 13 | Absent or explicitly refused; a precise refusal is an acceptable release boundary, not a parity claim. |

### Hamiltonian and reference

| Row | CRYSTAL23 capability | vibe-qc route | State | Evidence | Remaining gap | Issues |
|---|---|---|---|---|---|---|
| `rhf-3d` | Restricted Hartree-Fock (RHF), 3D, SHRINK mesh | BIPOLE `run_pbc_bipole_rhf` / `run_periodic_job(method='RHF', jk_method='bipole')`; GDF and GPW routes | `numerically-accepted` | LiH/STO-3G RSGDF Gamma and 2x2x2 within 1.3e-9 and 4.2e-9 Ha of matched PySCF GDF (#179 notes 50997/51683, runtime 63031b08); MgO/STO-3G BIPOLE 2x2x2 within 0.075 mHa of PySCF exxdiv='ewald' at converged cutoff (docs/periodic_jk_routes.md, 2026-06-16, older build) | No CRYSTAL23 row at a matched mesh has passed through the #573 harness; the CRYSTAL comparison is exchange-convention confounded at coarse SHRINK and k-mesh confounded at SHRINK 8 8 (+4.7 mHa) | #573, #593 |
| `uhf-3d` | Unrestricted Hartree-Fock (UHF) | BIPOLE `run_pbc_bipole_uhf`; Ewald multi-k UHF; GDF open shell (explicit `jk_method='gdf'`) | `numerically-accepted` | Ewald UHF Cl2 anchors agree with a matched periodic direct-integral reference within 1.5e-9 Ha (#573 description, 6b5e102e); H2 triplet UKS/PBE0 2x1x1 within 3.2e-9 Ha of PySCF AFTDF | Magnetic-state transport and a CRYSTAL23 open-shell crystal row remain open (#647); no matched CRYSTAL23 UHF total energy yet | #647, #573 |
| `rohf-roks` | Restricted open-shell HF (ROHF) | Ewald `run_rohf_periodic_multi_k_ewald3d`, GDF `run_krohf_periodic_gdf`, GPW Gamma; ROKS Ewald/GPW (maintained preview) | `source-tested` | Preview routes with pinned fixtures (docs/user_guide/periodic_methods.md section 3.8); Fock-unification migration open | No external reference comparison; GDF/GPW ROHF are previews; little-group and broken-symmetry handling open | #601, #597, #577 |
| `ghf-soc` | Generalized HF / two-component spinors, spin-orbit coupling | none | `unsupported` | No two-component periodic route exists in `python/vibeqc/` | Outside the #593 ground-state goal; relativistic/noncollinear extensions are a separate roadmap milestone | #593 |
| `rks-semilocal` | Kohn-Sham DFT, semilocal LDA/GGA, collinear spin | BIPOLE `run_pbc_bipole_rks`/`uks`; GDF; GPW/GAPW; libxc-routed functionals | `unverified` | #187: 12 of 12 pure-PBE crystals over-bound against CRYSTAL23 at matched SHRINK 2 2 by 1.5 to 35 mHa (v0.15.114); after the physical-density and XC-domain fixes the LiH13/PBE 2x2x2 fine-grid gap is -8.0 uHa on one fixture (note 43219); diamond Gamma PBE0 XC domain converged to 6.0e-7 Ha (#465, closed) | Re-measure the twelve #187 crystals at production defaults on current main with a >=4-cell grid ladder; #405 diffuse-basis XC image sum (3.8 mHa) open | #187, #405, #575 |
| `meta-gga` | meta-GGA (tau-dependent): SCAN, r2SCAN, Minnesota families | libxc-routed tau functionals on the periodic KS routes (periodic Becke grids) | `unverified` | #179 compares LDA/PBE/r2SCAN XC increments against CRYSTAL23 SHRINK 2 and finds the increments consistent while the Coulomb block carries the offset; no dedicated meta-GGA acceptance row | Add an r2SCAN row to the #573 matrix; Minnesota functionals have no periodic acceptance evidence | #573, #179 |
| `noncollinear` | Non-collinear spin DFT, spin-current DFT | none | `unsupported` | No noncollinear density or spin-current functional path | Separate roadmap milestone, not in #593 | #593 |
| `global-hybrids` | Global hybrids: B3LYP (VWN5), PBE0, PBESOL0, B1WC, WC1LYP, B97H, PW1PW | BIPOLE/GDF/GPW with `functional='b3lyp' / 'pbe0' / 'pw1pw'/...`; `vq.define_functional` for custom mixes | `unverified` | B3LYP VWN5 flavour verified against CRYSTAL14 to 1e-9 Ha on H2/STO-3G; diamond PBE0/def2-SVP Gamma GDF within 6.0e-6 Ha of PySCF Becke level 7 at converged cutoff (#163 note 40656); but #163 records a 440.8 mHa BIPOLE-versus-GDF Coulomb disagreement on diamond PBE0 Gamma on v0.15.133 that has not been re-measured | Re-run the #163 cross-route case on current main (BIPOLE cost was 35 h); add a matched CRYSTAL23 hybrid row | #163, #573 |
| `screened-hybrids` | Screened-Coulomb hybrids: HSE06, HSEsol, SC-BLYP | `functional='hse06'` on BIPOLE, Ewald and slab routes (short-range exact exchange) | `source-tested` | Gamma image-shell erfc exchange gate re-derived in MR !280 (#501); multi-k short-range exchange pinned in the BIPOLE lanes | No matched CRYSTAL23 or PySCF HSE06 crystal comparison recorded; HSEsol/SC-BLYP untested | #501, #573 |
| `lc-hybrids-production` | Long-range / middle-range corrected hybrids: LC-wPBE, HISS, wB97X, CAM-B3LYP | Corrected 3D physical-pair BIPOLE RKS/UKS implements single-range CAM/LC at Gamma/multi-k; GDF and multiple ranges remain refused | `unverified` | tests/test_periodic_screened_exchange.py: signed-kernel admission, independent fixed-density AFT exchange matrices and RKS/UKS limits; matched CRYSTAL crystal total and cost missing | Matched CRYSTAL totals and cost, support convergence, GDF and multiple-range HISS remain open (#657) | #657, #593 |
| `lc-hybrids-rsgapw` | LC hybrids on the range-separated GAPW route | `RSGAPW` dedicated entry point, 3D Gamma RHF/RKS | `implemented` | docs/user_guide/periodic_methods.md section 4.7 lists wB97X / CAM-B3LYP on RSGAPW | Experimental route, no multi-k, no reference comparison | #593 |
| `sc-hybrids` | Self-consistent global hybrids (sc-hyb) | none | `unsupported` | No dielectric-dependent exact-exchange fraction | Not a #593 deliverable | none |
| `skala` | (no CRYSTAL23 analogue) SKALA machine-learned functional | checkpoint-backed extension on the common periodic infrastructure | `unverified` | #83 runtime campaign; stub results and common-infrastructure timings are explicitly not checkpoint-backed acceptance (#593) | Complete RKS/UKS Gamma and multi-k calculations with the actual checkpoint, invariance and provider derivatives | #83 |

### Dispersion and composite corrections

| Row | CRYSTAL23 capability | vibe-qc route | State | Evidence | Remaining gap | Issues |
|---|---|---|---|---|---|---|
| `dft-d3` | DFT-D3 (automated) and DFT-D2 London corrections | `dispersion='d3' / 'd3bj'` with `compute_d3bj_periodic` (built-in supercell sum or dftd3 backend) | `source-tested` | `python/vibeqc/dispersion_periodic.py` tests; D2 is not offered | No CRYSTAL23 D3 crystal comparison recorded; D2 unsupported | none |
| `dft-d4` | (CRYSTAL23 offers D3; D4 listed for completeness) | molecular D4 generic; periodic D4 not wired into `compute_d3bj_periodic` | `unverified` | #548 propagates generic D4 to molecular optimizers and the FD Hessian; periodic D4 has no entry | Decide whether periodic D4 is a deliverable | #548 |
| `gcp-3c` | gCP counterpoise and composite 3c methods (HF-3c, PBEh-3c, HSE-3c, B97-3c, sol-3c) | `vibeqc.composites` registry and `vibeqc.gcp` (molecular) | `unverified` | Composite recipes exist for molecules; periodic 3c/sol-3c execution has no acceptance evidence | No periodic 3c row in the matrix; sol-3c variants absent | none |

### k sampling, occupations and SCF control

| Row | CRYSTAL23 capability | vibe-qc route | State | Evidence | Remaining gap | Issues |
|---|---|---|---|---|---|---|
| `shrink-mesh` | SHRINK Monkhorst-Pack net, anisotropic and shifted meshes | `vq.monkhorst_pack(system, [n1,n2,n3])`, explicit `KPoints`, IBZ reduction with full-mesh provenance | `source-tested` | #645 anisotropic CRYSTAL dimensions and printed-factor admission landed (46103c8b); #129 hexagonal meshes use multiples of 3 (756b5b0a) | Actual printed reciprocal coordinate and weight qualification against CRYSTAL output still open (#645) | #645, #129 |
| `gilat-net` | SHRINK ISP Gilat net for Fermi energy and DOS | `bz_integration='gilat'` on the multi-k drivers (self-consistent for insulators, post-SCF for metals) | `source-tested` | docs/user_guide/from_crystal.md and metallic_bz_integration.md; pinned fixtures | Metallic total energies still come from smearing; no CRYSTAL23 metal comparison | #155 |
| `smear` | SMEAR Fermi-Dirac smearing | `smearing_temperature` on BIPOLE/GDF/GPW single points | `source-tested` | Finite-temperature BIPOLE results documented; NOITER entropy from the evaluated density fixed (#649, 9bc34e25) | Graphene BZ ladder non-convergent (#129); metallic/magnetic rows are explicit #593 coverage rows | #129, #155, #649 |
| `spinlock-atomspin` | SPINLOCK and ATOMSPIN | `spinlock_mode`/`spinlock_value`/`spinlock_iterations` and `atomic_spins` on BIPOLE UHF/UKS | `source-tested` | Spin-schedule recursion pinned in the BIPOLE UHF/UKS lanes | Genuine charge/spin/magnetic-state transport through the comparison matrix is #647 | #647 |
| `guessp-restart` | GUESSP density restart, NOITER evaluation | `initial_guess=READ`/`read_from`, QVF/Molden restarts, `max_iter=0` zero-iteration evaluation | `source-tested` | #649 restricted and unrestricted multi-k NOITER corrected (4f541cb6, 9bc34e25); front-door restart refusals pinned (#656) | Restart from accepted source-consistent states and conserved charge (#472, #386) remain acceptance items | #649, #472, #386 |
| `convergence-aids` | FMIXING, LEVSHIFT, DIIS, Broyden, Anderson | `fock_mixing`, `level_shift`, EDIIS+DIIS; Anderson/Broyden+Kerker density mixers on GDF and RIJCOSX only | `implemented` | Molecular and periodic convergence lanes; BIPOLE has no Anderson/Broyden mixer | Same-state convergence on difficult cases (#469, #497 acceptance); mixers absent on BIPOLE | #469, #497 |
| `convergence-thresholds` | TOLDEE / TOLDEG / MAXCYCLE | `conv_tol_energy`, `conv_tol_grad`, `max_iter`; terminal exact-Fock density check | `source-tested` | Exact committed-density confirmation and density-closure gate (#187 note 43205, 184 tests) | none beyond the method rows | none |

### Electrostatics, exchange and truncation

| Row | CRYSTAL23 capability | vibe-qc route | State | Evidence | Remaining gap | Issues |
|---|---|---|---|---|---|---|
| `coulomb-truncation` | TOLINTEG T1/T2 Coulomb overlap and penetration thresholds | BIPOLE physical AO-pair support (`cutoff_bohr`), SR pair-interaction reach (`sr_image_precision`), fold-drift guard, and the #531 omitted-shell Coulomb estimate with opt-in `sr_truncation_budget_ha` | `source-tested` | Relabelling-invariant physical domain is the default (#316 migration); 100-bohr Ne molecular limit coefficient gate < 1e-8 at the shipped default; omitted-shell estimate bounds the measured 9.21e-8 Ha residual (this landing) | #316 overlap relabelling <= 1e-12 and #531 two-rung vq acceptance on a recorded runtime remain open; #44 native-family migration partial | #531, #316, #44 |
| `exchange-truncation` | TOLINTEG T3/T4 exchange overlap and pseudo-overlap thresholds | Direct erfc exchange with Schwarz and charge-pair screening; reciprocal long-range exchange; no quartet bipolar replacement | `source-tested` | Screening thresholds recorded in the resolved domain; bipolar quartet far field fails closed | Measured direct J/K bottlenecks (#186, #490); no CRYSTAL-identical algorithm | #186, #490 |
| `exchange-zero-mode` | (CRYSTAL applies no exchange G=0 correction) | `exchange_exxdiv='ewald'` probe-charge Madelung correction or `'none'` | `source-tested` | Convention identity at fixed density documented; BIPOLE coarse mesh matches PySCF exxdiv='ewald' (0.075 mHa MgO) | Comparison with CRYSTAL totals only at k convergence; a CRYSTAL-style uncorrected finite-k mode is out of scope | #573 |
| `ewald-nuclear` | Ewald treatment of nuclear and electron-nuclear electrostatics | 3D Ewald V_ne and E_nn sharing alpha; analytic nuclear source radius on physical support | `numerically-accepted` | Point-charge limit bound per Ye and Berkelbach (2021); 1/L monopole gate on the Ne molecular limit; #457 face-nuclei defect fixed | #483 2-D/GAPW nuclear images around the origin needs verification; #462/#488 physical 1D/2D Coulomb | #483, #462, #488 |
| `gdf-coulomb` | (no CRYSTAL analogue) Gaussian density fitting Coulomb tails | `jk_method='gdf'` RSGDF; `fit_source='classic'`; MDF | `unverified` | LiH matched within 1e-7 (#179); dense-core cells still carry a fitted-tail deficit (MgO -0.46 Ha Gamma, #159/#115 historical); #466 classic pairs physical default landed | Dense-cell default-policy accuracy, #275 component interpretation and #172 MDF definiteness open | #179, #115, #159, #466, #275, #172 |
| `xc-grid` | DFT integration grid and image domain | Periodic Becke partition, product angular grids, XC density image domain (2 r_ao + span) | `numerically-accepted` | #465 diamond def2-SVP PBE0 within 6.0e-7 Ha of the converged 25-bohr domain on runtime f1ad7714; continuous-grid Becke steps fixed (!230) | #405 diffuse-basis image sum (3.8 mHa), #381 multi-k GDF Becke radius, #268 ROKS grid weights open | #465, #405, #381, #268 |
| `symmetry` | Point and space-group symmetry exploitation | spglib space groups; SYM3b atom-pair-orbit Fock reduction; IBZ k reduction with full-mesh expansion | `source-tested` | SYM3b reconstruction bit-identical to the full build (docs/periodic_jk_routes.md); #466 IBZ classic-physical | #171 NaCl GDF reduction exceeds a 1225 GB guard; #470 little-group invariance at wedge representatives; #577 unified-Fock integration | #171, #470, #577, #469 |

### Basis sets and pseudopotentials

| Row | CRYSTAL23 capability | vibe-qc route | State | Evidence | Remaining gap | Issues |
|---|---|---|---|---|---|---|
| `gtf-am` | Gaussian basis functions s, p, d, f, g | libint-backed integrals; derivative orders 0/1/2 built with max_am 5/4/4 by default | `source-tested` | `scripts/build_libint.sh` configuration; f functions exercised by def2 bases | g functions depend on the libint build; no periodic g-function fixture | none |
| `pople-bases` | Standard Pople STO-nG, 3-21G, 6-21G libraries | 255 bundled `.g94` files plus BSE overlays (`BasisSet(mol, 'sto-3g')`) | `implemented` | docs/user_guide/basis_sets.md | none | none |
| `pob-library` | Internal POB-TZVP library | Bundled pob-* family (H-Br) and pob-TZVP-rev2 QVF records | `source-tested` | Basis overlay receipts and catalog tests | Coverage beyond Br and f-in-valence lanthanide/actinide bases absent | none |
| `basis-optimizer` | Internal basis set optimizer | `vibe-basis` companion package (OPTBASIS-style optimiser) | `implemented` | docs/roadmap.md: shipped and moved to `vibe-basis` | Separate repository; not part of this parity matrix | none |
| `ecp-gdf-ewald` | Pseudopotentials: Hay-Wadt, Stuttgart-Cologne AREP libraries | `ecp_library='lanl2dz' / 'ecpNNmdf'` on k-GDF, Gamma RIJCOSX, Ewald open-shell and ROHF/GDF routes (#35) | `numerically-accepted` | Cl2/LANL2DZ UKS PBE Gamma and 3x1x1: molecular offset 6.0e-6 Ha and bond force versus Richardson FD 6.0e-9 Ha/bohr on runtime f1ad7714 (#557 note 53206); NaCl/LANL2DZ GDF RHF within 4.4e-6 Ha of mapped PySCF after the #652 projector-domain repair (note 52563) | #652 shared UHF driver acceptance and density-grid artifact; general ECP forces/stress (#579); these are same-code controls, not CRYSTAL force parity | #35, #652, #579 |
| `ecp-bipole` | Pseudopotentials on the BIPOLE (CRYSTAL-family) route | Neutral 3D corrected physical-pair RHF/RKS/UHF/UKS scalar single points, effective ionic charges, valence populations and midpoint projectors | `unverified` | Cl2/LANL2DZ Gamma and 3x1x1 HF reference and raw Hcore controls; PBE0 convergence and open-shell guess checks (#658); retained operator identity | Broad bulk/reference and support-convergence acceptance, cost, symmetry, external XC and derivatives remain open; #652 failures are not superseded; completed #557 scope is unchanged | #658, #35, #593 |
| `sorep` | Spin-orbit (SOREP) pseudopotentials | none | `unsupported` | No two-component route | Outside #593 | none |

### Periodicity and geometry

| Row | CRYSTAL23 capability | vibe-qc route | State | Evidence | Remaining gap | Issues |
|---|---|---|---|---|---|---|
| `dim-3d` | 3D crystals, 230 space groups | `PeriodicSystem(dim=3)`, spglib symmetry | `source-tested` | All periodic lanes | none | none |
| `dim-2d` | 2D slabs and surfaces, 80 layer groups | `vq.slab_2d`; AUTO selects SLAB_EWALD_2D; explicit closed-shell slab GDF with analytic gradients; BIPOLE refuses dim=2 | `source-tested` | docs/user_guide/periodic_methods.md section 5.1 | Physical 2D Coulomb/exchange on the BIPOLE family (#462/#488); layer groups not exploited | #462, #488 |
| `dim-1d` | 1D polymers, rod and helical symmetry, nanotubes | AUTO and GDF refuse dim=1; model as a dim=3 cell with transverse vacuum | `unsupported` | docs/user_guide/periodic_methods.md section 3.4; #462 physical wires | Physical 1D route (#462); helical symmetry and nanotube builders absent | #462 |
| `geometry-editing` | Automated geometry editing: slab cuts, supercells, clusters, nanotubes | Python/ASE tooling (`make_supercell`, CIF/POSCAR I/O, `vq.slab`) | `implemented` | docs/user_guide/periodic_systems.md | No CRYSTAL-keyword equivalents; nanorod/nanotube/Wulff builders absent | none |

### Derivatives, relaxation and lattice dynamics

| Row | CRYSTAL23 capability | vibe-qc route | State | Evidence | Remaining gap | Issues |
|---|---|---|---|---|---|---|
| `forces-analytic` | Analytic nuclear gradients (HF and DFT, all-electron and ECP) | GDF closed-shell Gamma/multi-k/slab analytic gradients; BIPOLE analytic gradient gated preview; ECP UKS semi-numerical preview | `unverified` | #459 Gamma GDF RKS(PBE) analytic minus FD 3.8e-5 Ha/bohr; #482 multi-k KUKS 2.4e-6 Ha/bohr; Cl2 ECP preview within 6.0e-9 Ha/bohr of FD (#557) | Per-route derivative capability table (#578); independent force acceptance versus CRYSTAL/ORCA | #578, #579, #459, #482 |
| `forces-fd` | (production BIPOLE forces) | Finite-difference forces and fixed-cell atomic relaxation on BIPOLE (`optimize=True`) | `source-tested` | Route-preserving FD optimizer lanes; freeze masks; periodic NEB | Same-objective references for structural relaxation (#578, #175) | #578, #175 |
| `stress-cell` | Analytic cell gradients, full cell optimization, constant-pressure optimization | No analytic periodic stress; FD stress on GPW/GAPW and semiempirical; variable-cell BIPOLE/GDF fails closed | `unsupported` | docs/user_guide/periodic_methods.md section 5.5 | Matched stress and cell relaxation is #578; MANTA stress is #637 | #578, #637 |
| `ts-search` | Transition-state search (periodic) | Periodic NEB on the BIPOLE force stack; no periodic TS optimizer | `implemented` | #564 NEB process workers | No periodic saddle-point optimizer | #564 |
| `frequencies` | Harmonic Gamma frequencies, IR/Raman intensities, phonon dispersion, ADPs, VDOS | GAPW finite-displacement Gamma dynamical matrix (`periodic_gapw_phonon.PhononCalculator`); molecular Hessians | `unverified` | GAPW route only, on analytic GAPW forces; no BIPOLE/GDF periodic Hessian; no intensities | Phonons/QHA are a separate roadmap milestone, not #593 | #593 |
| `anharmonic` | Anharmonic X-H, VSCF/VCI | none | `unsupported` | none | Separate roadmap scope | none |
| `elastic-eos-qha` | Elastic tensor, equation of state, quasi-harmonic thermodynamics, piezoelectric and photoelastic tensors | `vibeqc.eos` fits an E(V) scan; no elastic, QHA or piezo workflows | `unsupported` | EOS fitting is tooling, not an automated workflow with qualified strains | Separate roadmap scope | none |
| `cphf-dielectric` | CPHF/CPKS dielectric and (hyper)polarizability up to fourth order | Molecular CPHF only | `unsupported` | `vibeqc.cphf` is molecular | Response spectroscopy is a separate roadmap milestone | none |
| `solid-solutions` | Solid-solution configuration tools | none | `unsupported` | none | Not planned | none |

### Properties and wave-function analysis

| Row | CRYSTAL23 capability | vibe-qc route | State | Evidence | Remaining gap | Issues |
|---|---|---|---|---|---|---|
| `bands` | Electronic band structure | `vibeqc.bands`, `band_edges`, QVF/plot output | `unverified` | #176: Si non-converged, 2D h-BN fails closed, negative indirect gap from per-k occupations; 'ewald' gauge default and recorded exchange q0 (!103) | Bands from accepted source-consistent states (#472, #176) | #176, #472 |
| `dos` | Density of states, band- and AO-projected | `pdos` on GDF properties; rebuilt operator gated on the 1e-10 Hermitian contract (#547) | `source-tested` | #547 DOS gate and bond orders from the accepted density | Accepted-state consistency (#176/#472) | #547, #176 |
| `population` | Mulliken population analysis, atomic multipoles | Mulliken and Mayer bond orders on every non-AICCM periodic route | `source-tested` | #547 bond orders from the accepted density (e3939146, 905f5f1d) | none beyond accepted-state items | #547 |
| `density-maps` | Charge and spin density maps | Cube output and QVF BvK density artifact | `unverified` | #386: QVF primitive-cell grid integrates 11.98 instead of 10 electrons | Fix the BvK density artifact normalisation | #386 |
| `hirshfeld-periodic` | Hirshfeld-I partitioning (periodic) | Molecular Hirshfeld only | `unsupported` | `vibeqc.properties` is molecular | Not planned in #593 | none |
| `qtaim` | Topological analysis (TOPOND / QTAIM) | Molecular `vibeqc.qtaim`; MANTA QTAIM slice (#618) | `unverified` | #618 QTAIM capability slice c6b17b10 posted | Periodic QTAIM acceptance | #618 |
| `coop-cohp` | Crystal orbital overlap/Hamiltonian populations | `vibeqc.coop_cohp` on GDF properties | `source-tested` | Pinned fixtures | Accepted-state gate as for DOS | #547 |
| `wannier` | Localized Wannier functions (Boys) | `periodic_localise` Boys localisation for local correlation | `implemented` | Used by periodic DLPNO; not a properties-module LOCALWF workflow | No Wannier-based IR/polarization properties | none |
| `other-properties` | Electrostatic potential, EFG, Fermi contact, Mossbauer, Compton profiles, structure factors, transport, spontaneous polarization | none | `unsupported` | none | Properties suite is a separate roadmap milestone | none |

### Performance and parallelism

| Row | CRYSTAL23 capability | vibe-qc route | State | Evidence | Remaining gap | Issues |
|---|---|---|---|---|---|---|
| `openmp` | OpenMP and MPI parallel SCF and gradients | OpenMP native kernels; MPI output-cell farming for the BIPOLE direct ERI build; fleet runtimes | `unverified` | #96: GDF thread efficiency saturates near 17 threads (0.27 at 64 cores) | Measured scaling recommendation (#96); production MPI is a 1.0 milestone | #96 |
| `time-to-solution` | (parity target) Same-accuracy time-to-solution versus CRYSTAL23 | #574 paired full-SCF benchmark manifest; #644 CRYSTAL engine-time extraction | `unverified` | No paired same-accuracy benchmark has been accepted; BIPOLE diamond PBE0 Gamma cost 35 h on 20 cores versus GDF 1.5 h (#163, old runtime) | Target ratio <= 1 per declared production case with uncertainty (#574) | #574, #644 |

<!-- matrix:end -->

## Material gaps without a dedicated owner

Rows whose remaining gap has no bounded tracker issue of its own as of the
update date:

* `meta-gga`, `gcp-3c`, `dft-d4`: no periodic acceptance rows exist for
  meta-GGA functionals, composite 3c methods or periodic D4.

Gaps that received their own issue on 2026-10-24: long-range corrected
hybrids on the production routes (#657, row `lc-hybrids-production`) and
pseudopotentials on the BIPOLE route (#658, row `ecp-bipole`).

Lattice dynamics, elastic and thermodynamic workflows, response properties,
two-component and noncollinear Hamiltonians, and the properties suite are
deliberately outside the #593 ground-state goal and carry their own roadmap
milestones; their `unsupported` rows are listed for completeness only.

## Updating the matrix

1. Edit `benchmarks/crystal23_capability_matrix.json`: change the row's
   `state`, `evidence`, `gap` and `issues`, and bump `updated`.
2. Run `python scripts/crystal23_capability_matrix.py --write`.
3. Run `python -m pytest tests/test_crystal23_capability_matrix.py -q`.
4. Link the tracker note that carries the evidence from the issue named in
   the row. A promotion to `numerically-accepted` needs the recorded runtime
   identity and the matched reference in that note.
