# Issue renumbering, 2026-09-08

When vibe-qc moved to its own repository, the 188 open issues were moved
from `mpei/vibeqc` (project 19) to `mpei/vibe-qc` (project 34). GitLab's move
creates a new issue with a new number and closes the original with a
"moved to" note, so **every issue number changed**.

The mapping is not derivable from the numbers: moves were applied in
ascending old order while new numbers were assigned in sequence, so the
ordering inverted -- issue 8 became 188. There are 188
distinct offsets across 188 issues.

This table exists because those old numbers are cited elsewhere and cannot be
followed otherwise:

* ~125 `(#NNN)` references in `CHANGELOG.md`, and more in commit subjects;
* the JCC manuscript's defect provenance, via the article repository's
  handovers;
* `bugctl` grant and landing records.

Closed issues were deliberately NOT moved. All 776 of them remain on project
19, which is retained as the archive; their numbers are unchanged there.

A machine-readable copy of this table, plus a full pre-move snapshot of every
issue including its original labels, is kept outside the repository at
`_split-archive/` on the maintainer's machine.

| old (project 19) | new (project 34) | title |
|---|---|---|
| 8 | 188 | BUG 95 [LOW] compute-managed immutable release omits advertised D4 dependency |
| 20 | 187 | BIPOLE-PURE-FUNCTIONAL-SYSTEMATIC-OVERBINDING [HIGH] |
| 21 | 186 | perf-bipole-periodic-hf-hours-per-scf-iteration [HIGH] |
| 22 | 185 | BH9-TYPE-I-ISOMERISATION-OUTLIER [MEDIUM] |
| 43 | 184 | BUG 027 [HIGH] GFN2-xTB total-energy residual vs xtb 6.7.1: +245.7 mHa on adenine after the E_re |
| 65 | 183 | Multi-k GAPW fails to converge on expanded cells (missing global-BZ Aufbau and SCF density mixin |
| 66 | 182 | BIPOLE fold-guard known-red block: re-measured re-cut cutoffs await a dedicated edit session |
| 74 | 181 | [HIGH] compute-reference fleet host: memory preflight reads effective allocation 0 MB and refuses every ca |
| 78 | 180 | [HIGH] SCC-DFTB pi-conjugated systems still fail at v0.15.130 (finite-T retry not the default) |
| 81 | 179 | Periodic GDF total-energy offset vs PySCF: DF Coulomb-tail completeness deficit on dense-core ce |
| 82 | 178 | BIPOLE multi-k total energy vs CRYSTAL 23: exchange finite-size gauge (+6.43e-1 Ha on MgO) |
| 84 | 177 | Multi-route band-gap comparison open: LiF pob-TZVP-REV2 (2,2,2) vs GPAW |
| 85 | 176 | Bands/DOS: Si non-converged; 2D h-BN fails closed; negative indirect gap from per-k occupations |
| 86 | 175 | Bulk equation-of-state observables (a0, B, band gap): Si/MgO/LiF scans non-converged or runaway |
| 87 | 174 | Brillouin-zone mesh convergence: unguarded 38.9 GiB Ewald-cache allocation; k=6 non-converged |
| 89 | 173 | DFT+U: NiO target fails with a memory-allocation failure |
| 91 | 172 | Mixed density fitting for heavy cores: overlap matrix loses positive definiteness |
| 92 | 171 | Space-group symmetry demonstration: NaCl reduction exceeds a 1225 GB GDF memory guard |
| 94 | 170 | DF-CCSD(T): explicit DF-vs-conventional pair comparison on the corrected fitting path pending |
| 95 | 169 | CPCM/COSMO cross-code validation: compare by solvent-response components, not raw solvated total |
| 97 | 168 | MgO cluster cross-code RHF consensus contract open at 1.1e-1 Ha |
| 99 | 167 | Native D4 backend: parity coverage H-Ne only; external dftd4 remains default |
| 116 | 166 | BIPOLE-TERMINAL-REBUILD-WITHDRAWS-CONVERGENCE [HIGH]: non-incremental post-loop Fock rebuild dis |
| 117 | 165 | PAPER-ROW-CARRIER-REFUSAL-OUT-DROPS-RUNG-FIELDS [LOW]: an rc-9 .out header reports 'cutoff=shipp |
| 120 | 164 | CARRIER-PER-CASE-TIMEOUT-KILLS-INSIDE-A-LONG-RESERVATION [HIGH]: 206 members lost to a ~900-1200 |
| 121 | 163 | BIPOLE-VS-GDF-CROSSROUTE-440MHA-ON-DIAMOND [HIGH/P1]: same code, same job, same runtime, Coulomb |
| 127 | 162 | [HIGH] GFN2-xTB parameter cache is per-host and unpinned: compute-cluster compute nodes offline -> all 78 G |
| 135 | 161 | PAPER-ROW-CARRIER-TYPEERROR-POST-SCF [HIGH]: band/density extraction raises 'TypeError: only len |
| 139 | 160 | VALIDATION-RUNS-ENERGY-HA-CARRIES-SCF-TOTAL-FOR-CORRELATED-RUNS (reffill wave + v0.15.28/30 legs |
| 142 | 159 | GDF-KPOINT-COLLAPSE-ON-POB-TZVP-REV2 [HIGH]: Si-diamond RKS-PBE (2,2,2) rc 0 at -582.128 Ha, 3.4 |
| 145 | 158 | REFERENCE-DECK-DEFECT-CPCM-RUNS-SMD: the CPCM h2o reference family (mb075/ml075*/mf075) runs SMD |
| 147 | 157 | RP217J-REGISTRATION-WRITES-SCF-TOTAL-FOR-CORRELATED-METHODS: the judged-registration tooling sti |
| 149 | 156 | CASE-METHOD-VS-DECK-METHOD-MISMATCH-DLPNO-MONOTONICITY: 8 liakos rows' case says dlpno-ccsd whil |
| 154 | 155 | PERIODIC-SCF-STAGNATION-SI-K666-GDF [MEDIUM]: energy pinned 14 consecutive iterations at -582.08 |
| 160 | 154 | compute-study-PM6-NORBORNADIENE-SILENT-E-ZERO [MEDIUM, legacy BUG-024]: compute-study-only PM6/norbornadiene retur |
| 161 | 153 | MACE-NOT-STAGED-ON-FLEET [MEDIUM, legacy BUG-014]: MACE models unavailable on fleet hosts - stag |
| 162 | 152 | SE-CAMPAIGN-GENERIC-FILENAMES [LOW, legacy BUG-020]: calculation artifacts named input.py/out/qv |
| 207 | 151 | Nonorthogonal SECCM adapters feed indefinite WS overlap matrices to generalized eigensolvers |
| 208 | 150 | Semiempirical periodic tier vocabularies disagree; C++ PeriodicTier is dormant and the .out neve |
| 242 | 149 | GAMMA-CCM-UNION12-VARIATIONAL-COLLAPSE-ON-NON-MINIMAL-BASIS [HIGH]: eq-18 weighted Coulomb super |
| 246 | 148 | compute-cluster-AMD-CAPACITY-INVISIBLE: three nodes offline; ppn=128 requests are unschedulable and queued  |
| 253 | 147 | W1-RV-DECKS-CASE-NOT-JSON-DECODED: 57 archived decks across 7 bundles die at CASE["charge"] with |
| 254 | 146 | ARTIFACT-STEM-TRUNCATION-AT-DOT [HIGH]: three instances; now manufacturing a FALSE route-mismatc |
| 296 | 145 | Periodic GFN2-xTB energy is not invariant under a lattice translation of an input atom (85 mHa o |
| 303 | 144 | χ-CCM four-center SCF lacks distributed finite-translation output-cell farming |
| 308 | 143 | run_ccm_rhf_gdf at nrep=(1,1,1) is pathological: N=1 costs more than N=5; ~2900 core-hours burne |
| 311 | 142 | Agentic loop: move coordination, fences and rollout identity out of prose into a machine-checked |
| 338 | 141 | Periodic GFN2-xTB analytic gradient/stress are not derivatives of the periodic energy |
| 348 | 140 | GFN2-SECCM AES moment integrals depend on the typed torus representative |
| 408 | 139 | Periodic GFN2 Gamma driver rejects anisotropic supercells: 'H0 coordination pair distance must b |
| 409 | 138 | SECCM broyden_eyert mixes shell charges only; tblite's Broyden state vector also carries atomic  |
| 421 | 137 | GFN2-SECCM converges into symmetry-forbidden / hopped SCC attractors that pass every screen (Ge  |
| 423 | 136 | GFN2-SECCM OpenMP speedup is x0.94-1.44 on 20 threads; serial Python build dominates (87% of wal |
| 425 | 135 | k-route cutoff ladder unconverged at 200 bohr for dim>=2: mHa-scale tails on every 2-D/3-D dense |
| 427 | 134 | ascent reference runner swallows ORCA error termination (exit 0, vq completed, no energy); atom- |
| 429 | 133 | Gaussian-stack image selection is translation-ball, not pair-ball: a Gamma supercell wider than  |
| 432 | 132 | GFN2 native D4 dispersion is absent for Z > 10: frozen refdata covers 8 elements, and TM totals  |
| 435 | 131 | k-route indirect gap uses an occupancy-partition convention: 1022/1668 gapless rows report a pos |
| 436 | 130 | vibeqc-inhouse-dftb-screening-v1 leaves bulk_si and hbn gapless; benchmark set has no independen |
| 437 | 129 | graphene BZ ladder is non-convergent (K-point parity split): mesh spread is 127% of the reported |
| 445 | 128 | Lattice row/column orientation is silently accepted either way: three independent transposes fou |
| 456 | 127 | .system manifest can escape the job output directory into the checkout root (seen in a multi-fil |
| 473 | 126 | Validation DB: reassemble the 10 BH9 03_12R2-derived comparisons now that the species converges |
| 475 | 125 | SECCM iteration-count target: adenine needs 24.5x xtb (514 iterations) and the ~3x bar has no re |
| 493 | 124 | real-Γ direct-torus SCF has no convergence aids: 160-atom LiH slab+H2 fails to converge in 128 i |
| 503 | 123 | Block Davidson: wrong roots reported as converged for small n_eig, and convergence never signall |
| 505 | 122 | NO PUBLIC PER-K BAND ACCESSOR: every carrier re-derives frontier-band extraction and the class h |
| 507 | 121 | Verification is dispatched by hand: the broker records LANDED but emits no queue a distinct acto |
| 513 | 120 | Article sealed-reference table still fills the BIPOLE MgO row from CRYSTAL's RETIRED coarse-mesh |
| 516 | 119 | CASSCF analytic gradient: compute_wz=True yields a spurious x component with d-function bases (H |
| 517 | 118 | Article repo commits author home paths and has no pre-commit hook; its DB validator only runs in |
| 523 | 117 | Citation DOIs diverge between code and article repos: emitted .bibtex sends readers to preprints |
| 525 | 116 | Semiempirical manuscript describes a converging periodic OMx driver that has failed closed since |
| 526 | 115 | Dense-core multi-k KRKS residual: -3.8 mHa vs PySCF / -4.4 mHa vs CRYSTAL survives a parity-size |
| 529 | 114 | Ideal-cubic Gamma DFTB0 T=0 ensemble stress is symmetry-correct but differs from dE/deps by up t |
| 536 | 113 | test_geomopt.py::test_run_periodic_geomopt_imports_and_runs is a deterministic BipoleFoldUnrelia |
| 539 | 112 | Six tests/test_out_format_snapshot.py golden tests are red on main (molecular rhf/rks/hessian/td |
| 547 | 111 | vq queue directory is never reaped: 16,374 specs, 99.9% terminal, delay new scheduler dispatch b |
| 550 | 110 | Staged ORCA UKS reference decks emit the non-keyword UPBE; the stager fix reached one of 25 copi |
| 551 | 109 | ORCA CCSD(T) reference for H2/STO-3G aborts in MDCI (K(C)-AO-direct not implemented) while exiti |
| 552 | 108 | RP248 ladder payload forces write_molden_file=True and blocks 37 of 86 cases before any chemistr |
| 554 | 107 | The reaction-field step is implemented twice: the Gaussian macro-iteration and the MSINDO fock_e |
| 555 | 106 | qcil submit_wave.py cannot submit to compute-cluster at any version: it reads release[binary], which every  |
| 556 | 105 | S22 validation decks pass max_iter/conv_tol_energy to run_job, which accepts neither: 22 runs al |
| 558 | 104 | Implement the COSMO-RS/COSMOSPACE thermodynamic layer: conductor-surface record, sigma profiles, |
| 559 | 103 | import vibeqc raises when the native core cannot satisfy SKALA registration, breaking docs-build |
| 567 | 102 | vibe-qc DFT single points sit 17 to 44 uEh/atom above ORCA 6.1.1 on all six qcil wave001 DFT pai |
| 568 | 101 | qcil wave tooling is pinned to 0.15.45 and mis-parses 0.15.157 artifacts: validator strips .inpu |
| 573 | 100 | TREXIO I/O: writer and reader in vibe-qc (HDF5 and text back ends) and a TREXIO reader in vibe-v |
| 577 | 1 | fleet: unify the vibe-qc deployment layout across nodes (three different layouts on five hosts) |
| 638 | 99 | LiH/pob-TZVP-rev2 PBE multi-k GDF holds 29.4 GB peak RSS at nbasis=13 (19x the MgO/TZVP leg) on  |
| 646 | 98 | GAPW and GDF disagree by 0.3007 Ha on LiH in the molecular limit: the no-blow-up guard overshoot |
| 648 | 97 | Third CWD-artifact leak class: 37 call sites pass output= a bare relative string, which the guar |
| 649 | 96 | Periodic GDF does not scale to a full node: T_eff saturates near 17 threads, 0.27 efficiency at  |
| 650 | 95 | A convergence aid silently routes Gamma GDF onto the legacy fallback, which never auto-sizes the |
| 652 | 94 | test_uno_reference::test_orthonormal_and_occupations_vs_pyscf is red on main and absent from the |
| 655 | 93 | AICCM front door M2 (gamma half): pin the per-unit-cell artefact contract of the two neutral pro |
| 656 | 92 | χ-CCM D126: compact point-symmetry quotient for translation-pair work |
| 658 | 91 | bugctl: the L157 verification-brief sequence cannot work as documented (land closes the grant pr |
| 659 | 90 | AICCM front door M3a: wire the four-centre runner arm (method="aiccm", variant="four-center") |
| 661 | 89 | χ-CCM D127: compact real-torus AO point-action kernel |
| 662 | 88 | The RSGDF dense-core held class covers every all-electron pob-TZVP-rev2 cell, not one: 28 to 200 |
| 664 | 87 | Multi-k GDF is over-bound by a mesh-independent Madelung-scale amount: 0.46 Ha from the Gamma fa |
| 665 | 86 | Fleet PySCF 2.6.2 cannot converge a periodic meta-GGA grid and silently returns a 28 mHa-wrong r |
| 670 | 85 | AICCM four-center external-XC front door drops grid/domain controls and certifies unsupported dr |
| 671 | 84 | Periodic alternative backends misexecute and misreport non-SAP initial guesses |
| 672 | 83 | SKALA next-release compute-cluster runtime and scientific validation campaign |
| 674 | 82 | BIPOLE corrected exchange split cuts its erfc arm at cutoff_bohr with CRYSTAL's unbounded alpha  |
| 676 | 81 | test_multik_roks_fused_kernels_match_serial_fold is nondeterministic: fused vs serial iteration  |
| 678 | 80 | Bare KS options still select the legacy grid on run_wb97x_d, the double-hybrid core, optimize_mo |
| 680 | 79 | Periodic SAD initial guess is un-normalized: tr(D S) = 5.31 for a 4-electron LiH cell (+0.73 e) |
| 681 | 78 | ROHF/GDF public route refuses its own default initial guess (SAD) since bbe7b9001: six tests of  |
| 683 | 77 | χ-CCM D128: bind localized occupied action to its physical overlap |
| 684 | 76 | Molecular density-mode initial guesses do not preserve target charge and spin populations |
| 686 | 75 | Direct native SCF-with-JK entry points ignore selected initial guesses |
| 687 | 74 | READ initial guesses lose basis provenance and accept invalid periodic occupations |
| 688 | 73 | Nested SCF and ASE workflows drop the selected initial guess |
| 689 | 72 | DLPNO-CCSD extended residual domain is 300-400x slower than the legacy recipe and barely truncat |
| 690 | 71 | BIPOLE multi-k RKS/UKS discard constructed SAD and MINAO densities for HCORE |
| 693 | 70 | bugctl has no reopen verb: the specified REOPENED transition (reopen, drop resolution::verified, |
| 694 | 69 | SCC-DFTB-SECCM residual odd-MgO and slab state anomalies misattributed to GFN2 in #421 |
| 695 | 68 | Wave submitter routes oversized periodic legs to compute-study, blocking its strict-order local lane |
| 697 | 67 | χ-CCM D129: finite-torus spatial group and occupied cocycle witness |
| 700 | 66 | DLPNO-CCSD: implement the Riplinger-Neese section II C term-by-term PNO-basis residual so the ex |
| 701 | 65 | DLPNO PNO occupation numbers use a pair density that is neither the Riplinger-Neese Eq. 23 MP2 n |
| 702 | 64 | CCM citation selector credits exact Ewald to MSINDO frozen truncated 1-D Madelung sum |
| 703 | 63 | χ-CCM D130: connect full-mesh RHF snapshots to shared native orbital sewing |
| 704 | 62 | χ four-center multi-k HF: half-cell translation Fock covariance is not qualified for pair reduct |
| 705 | 61 | GDF multi-k termination mixes D_used energy with next-Fock occupation entropy |
| 706 | 60 | Multi-k Ewald UHF/UKS mix selected seed densities with stale HCORE orbitals |
| 708 | 59 | BIPOLE UHF Gilat H-atom energy differs from its fixed-point pin by 6.03 microhartree |
| 709 | 58 | GFN2 feature documentation gate still requires retired exact wording |
| 711 | 57 | Multipole invalid-order test still rejects supported L_max=4 |
| 712 | 56 | CASSCF backend parity fixture assumes a unique H2O basin despite roundoff-triggered 7 mHa root c |
| 713 | 55 | UKS SH tight-tolerance trajectory stalls from a legal spin-density seed in the unchanged native  |
| 714 | 54 | Periodic GDF: RHF and UHF(M=1) energies of the same all-electron cell differ by 2.3 Ha at kpoint |
| 715 | 53 | Periodic RKS/PBE on a pob-TZVP-rev2 Ag cell allocates >80 GB in the XC build (RHF/UHF on the sam |
| 716 | 52 | Periodic SAD guess ignores ECPs: RHF on an ECP cell stops at higher stationary points (AgCl: -53 |
| 717 | 51 | Shared symmetry M1: bounded group and space contracts with chi and molecular consumers |
| 718 | 50 | Verification independence should key on run_id (the chat), not actor_id (the brand): two chats s |
| 719 | 49 | M4a: flip the A-line RI correlation reference to reference="direct" and add parallel neutral rou |
| 720 | 48 | run_periodic_job rejects an invalid initial_guess before the insulator-smearing guard can warn s |
| 721 | 47 | test_bipole_production_guards fold-truncation snapshot is stale since 6da75abe7 added the 'Exact |
| 722 | 46 | run_krohf_periodic_gdf rejects initial_guess=AUTO before its ROKS functional refusal since bbe7b |
| 723 | 45 | 2D UKS multi-k QVF density artifact falls back with a primitive-cell grid integral of 0.025 inst |
| 724 | 44 | #429 stage 2, Family 3: two-electron real-space builders on the pair-complete list, the Ewald nu |
| 727 | 43 | run_periodic_job's published API reference documents initial_guess's default as '<object object> |
| 728 | 42 | test_periodic_gdf_smearing_diis::test_multik_krhf_smeared_diis_occupations_are_fock_selfconsiste |
| 729 | 41 | Derive the analytic nuclear gradient of the COSMO FINE cavity so CFC geometry optimization is po |
| 730 | 40 | vq submit queues a command whose argv[0] is a vq flag: a misplaced --idempotency-key is swallowe |
| 733 | 39 | run_krohf_periodic_gdf rejects its own default initial_guess: test_open_shell_gdf_drivers_report |
| 734 | 38 | IBZ-native symmetry-broken-state guard no longer fires: test_ibz_native_refuses_a_symmetry_broke |
| 736 | 37 | vq submit returns rc=0 with a job id for a scheduler submission that then fails at workspace sta |
| 737 | 36 | run_validation_smoke.py pins EXPECTED_VERSION = 0.15.28 and refuses every release since 2026-07- |
| 738 | 35 | Periodic ECPs run only on the k-point GDF route: RIJCOSX, Ewald open-shell, ROHF/ROKS-on-GDF and |
| 739 | 34 | Periodic ECP gradients, forces and stress are unimplemented: no dV_ECP/dR lattice kernel |
| 740 | 33 | CASCI/MRCI/CASSCF/NEVPT2/CASPT2 and OVGF still refuse an ECP reference |
| 741 | 32 | QVF DOS/PDOS/COOP/COHP payload is skipped on ECP cells: it rebuilds H from bare Z with no V_ECP |
| 742 | 31 | Periodic ECP energies have no cross-code parity pin: add a PySCF.pbc check for the GDF route |
| 743 | 30 | Basis registry follow-ups after the ECP unification: def2-sv, cc-pVnZ-PP, alias resolution |
| 744 | 29 | CFC segment coarsening yields an indefinite CPCM A matrix at ~1 grid spacing in 6 |
| 745 | 28 | CPCM gradient hard-codes switching_sigma_bohr=0.5 while SolventModel exposes it |
| 746 | 27 | Assemble the CFC energy gradient: cpcm_gradient still refuses a FINE cavity |
| 748 | 26 | periodic_runner.py:12847 reads _last.delta_e on a dict, so the SCF diagnostic raises AttributeEr |
| 749 | 25 | Roughly forty validation cases request combinations vibe-qc correctly refuses, so the suite repo |
| 751 | 24 | True multi-k periodic GDF cannot return the lattice-cell density required by QVF and density-gri |
| 752 | 23 | EWALD_3D exchange gauge is still chosen by the mesh when exchange_exxdiv is left at its default: |
| 753 | 22 | AO symmetry maps shells by position and accepts incompatible radial contractions |
| 754 | 21 | Libint Boys interpolation reads beyond its table at T=117 and corrupts periodic GDF integrals |
| 755 | 20 | BIPOLE erfc QQR attenuation drops diffuse angular quartets above the screening threshold |
| 757 | 19 | CFC energy is discontinuous: step-6 basis-point assignment moves whole segment areas between seg |
| 758 | 18 | libecpint radial quadrature falsely converges before an off-centre Gaussian projector peak |
| 760 | 17 | Dormant far-field symmetry reconstruction accepts unqualified orbits and returns incorrect Fock  |
| 762 | 16 | Dormant far-field native adapter mixes row-major and column-major AO-pair indices |
| 763 | 15 | All-electron POB cells fail initial-guess ECP preflight on charge-only metadata |
| 765 | 14 | Periodic COHP projects bare Hcore instead of the analyzed HF/KS Hamiltonian |
| 767 | 13 | Periodic Mayer bond orders quadruple an unrestricted singlet and assume PS is Hermitian |
| 768 | 12 | periodic_runner.py:12809 picks the SCF trace row with isinstance(_trace, list), but the GPW/GAPW |
| 769 | 11 | CFC energy is not rotationally invariant: segment directions are lab-fixed, skipping the paper's |
| 770 | 10 | CFC energy has a C1 kink where step-6's projection flag flips |
| 771 | 9 | CFC step-5 basis-point-to-atom assignment is a hard argmin, leaving a residual energy jump |
| 772 | 8 | Periodic Becke partition changes under atom-image relabelling at fixed periodic density |
| 773 | 7 | Citation entry maurer_ochsenfeld_qqr_2012 is unreachable after the QQR screening removal, so tes |
| 774 | 6 | bugctl land hangs instead of refusing when --project-id names an unregistered project: the allow |
| 775 | 5 | BIPOLE SR screening still calls itself QQR in a user-facing run-log line, in troubleshooting.md  |
| 776 | 4 | Periodic XC gradients reload translated custom bases by name and can abort or change the basis |
| 777 | 3 | Molecular GFN2 adenine converges to four states spanning 29.4 mHa under settings that should not |
| 778 | 2 | M4b/M4c: correlation= keyword on the AICCM gamma variants, with the compound citation keys that  |
