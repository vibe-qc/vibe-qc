# AICCM real-Gamma production track

> **Experimental.** The real-Gamma producer stays under
> `AICCM2026DevAExperimentalWarning` until the maintainer decides otherwise.
> This page is the plan and the ledger for getting it there: what "production
> ready" means for this line, the target architecture, and the increments
> that land it in small steps on `main`. It is updated as increments land.

## 1. Scope

The **real-Gamma producer** of the neutral fitted-torus Hamiltonian:
`run_ccm_rhf_direct`, `run_ccm_rks_direct`, `run_ccm_uhf_direct` and
`run_ccm_uks_direct` in `vibeqc.periodic.ccm.direct`, the neutral cderi fold
and its block storage (`neutral.py`, `block_cderi.py`), the runner arm
`run_real_gamma_scf` and `run_periodic_job(method="aiccm", variant="real-gamma")`.
Its sibling is the neutral-Bloch producer (`run_ccm_rhf_gdf`,
`run_ccm_rks_gdf`), the Bloch representation of the same Hamiltonian.

Out of scope: the union-and-weight four-centre lineage (`aiccm2026dev-a`
construction routes) and chi-CCM (`aiccm2026dev-b`). Nothing here changes
their Hamiltonians or labels.

## 2. What production ready means for this line

Each criterion is checkable; the increments in section 4 name the test that
checks it.

1. **One Hamiltonian, two representations.** Real-Gamma and neutral-Bloch
   agree to 1e-10 Ha/cell (the Theorem-1 identity) on the ladder set: LiH,
   MgO, diamond and fcc Be at STO-3G and pob-DZVP-rev2, meshes (1,1,1) to
   (2,2,2), HF, PBE and PBE0, closed and open shell, with the RSGDF tail
   resolved at every N (#441, landed) and no parity hold.
2. **Same-source evidence.** Every result carries a source identity: fit
   source, auxiliary basis, kinetic-energy cutoff and tail, cell-list cutoff,
   exchange seam convention, storage mode and symmetry plan. A comparison
   across different sources is refused by identity, never averaged.
3. **Evaluated state (#472).** The terminal Fock, density, energy components,
   source and frame identities are frozen in a `FockBuild`. DIIS trial Focks
   are solver operators and never the reported physical operator. The
   `.system` manifest and the QVF vendor section carry the build identities.
4. **Robust SCF.** SAD, READ and Bloch-seeded initial densities on every
   direct driver and in the runner arm, a frontier-gap report and an internal
   stability check after convergence, so a second stationary point is found
   rather than reported (#442).
5. **Cost.** Block storage is the default for `N_c > 1` with the dense fold as
   the reference; the XC image domain matches the Bloch side at the Bloch
   side's cost; the fold cost is tracked (#258); symmetry reduction is opt-in
   and records its qualification evidence.
6. **Interchangeable providers.** The producer runs on the classic RSGDF fit
   (the libint/native FFT builder of the multi-k GDF driver) and on MANTA
   periodic GDF tiles. Each is a named source with its own Bloch control.
7. **Promotion evidence.** The identity tests move from T3 `experimental` to
   T2 rows with measured runtimes, the manual pages follow behaviour, and
   parity against PySCF and CRYSTAL23 is recorded in the AICCM-2026 set.

## 3. Target architecture

### 3.1 Fit-source provider boundary

Today the real-Gamma cderi is built by `ccm_neutral_cderi_fold`: per kept
momentum transfer `q`, one shared-q batch of unit-cell Bloch-pair fits
`L_{k,k+q}[P,mu,nu]` from `vibeqc.aux_basis.build_lpq_bloch_native_fft_shared_q`
(the multi-k GDF driver's classic RSGDF builder), folded to the torus or kept
as block storage. The per-channel fits are the natural provider boundary,
and since RG-7 it is explicit (`vibeqc.periodic.ccm.fit_source`):

```
fit source               bind(FoldContext) -> bound source with
                           channel_fits(qi, bras) -> {ai: L_{ka,ka+q}[P,mu,nu]}
                             in the canonical auxiliary frame, ket unfolded
                           supports_time_reversal, supports_star_plan
                           engine, transport, receipt()
ClassicRSGDFFitSource    the builder above (shared-q or per-pair batching)
DriverFitCacheSource     the multi-k driver's retained fit of the same operator
MANTAFitSource (RG-8)    vibeqc.manta.PeriodicGDFTiles per q over the mesh,
                         projected with the RI-Q builder's positive projector
```

`ccm_neutral_cderi_fold` and `ccm_neutral_cderi_blocks` take `fit_source=`;
the fold keeps its orchestration (kept channels, time reversal, the star
plan, the fold or the block storage) and applies time reversal and the star
plan only to a source that declares them. `CCMBlockCderi` stores what it
gets and contracts J and K exactly as before. The source receipt becomes part
of `cderi_info` and of the `SourceProvenance` of section 3.3. MANTA kernels
stay on their admitted native path (`PeriodicGDFTiles.evaluate`, resource
budgets, receipts); the adapter copies tiles into the block storage it owns
and never wraps a generic copying consumer around the native loop.

A MANTA-sourced torus is a **different neutral source** from the classic one
(`docs/design/manta_scientific_contract.md`, section 6): its Bloch control is
MANTA's own whole-mesh RI build (`PeriodicRIQBuilder` at the full BvK mesh),
not `run_ccm_rhf_gdf`. The Theorem-1 identity is proved per source.

### 3.2 Storage

Block storage (`cderi_storage="block"`, commit cccadc6e) keeps `16 N_kept N_c
n_aux n_mu^2` bytes against `8 n_aux n_mu^2 N_c^3` for the dense fold and
evaluates J and K in the mesh representation on the circulant projection of
the torus density. It is wired for RHF and RKS. The open-shell drivers fail
closed on it. The track extends it to UHF and UKS (per-spin exchange through
the same per-k factor) and then makes it the runner default for `N_c > 1`,
keeping the dense fold as the reference switch and the `tests/test_ccm_block_cderi.py`
identity at 1e-10 Ha/cell as the gate.

### 3.3 Fock build and evaluated state

The SCF loops (`ri._rhf_loop`, `ri._rks_loop`, the inline open-shell loops in
`direct.py`) assemble `F = h + J - 1/2 [K + xi_N S D S]` (plus `V_xc` and the
`a_x` fraction for KS) and the energy from the same `D`. The #472 contract is
adopted without changing those formulas:

* a `SourceProvenance` per run (provider `ccm-neutral-real-gamma`, details
  from `cderi_info`, the one-electron fold's lattice options, the exchange
  seam, the XC domain and grid);
* an `OperatorFrame` built from the supercell `S` and the canonical
  orthonormaliser `X` the loops already use;
* at the accepted state, `FockBuild.from_dense` with `EnergyComponents`
  (`hcore`, `coulomb`, `exchange`, `exchange_q0_seam`, `xc`) whose sum is the
  electronic energy the loop reports; one restricted channel at Gamma, or
  alpha and beta channels;
* DIIS trial Focks described through `FockBuild.solver_operator` and never
  stored on the result as `fock`;
* `result.fock_build` and `FockBuild.output_fields()` in the runner manifest
  and the QVF vendor section.

Acceptance follows the #472 pattern of the Gamma Ewald producers:
same-density components, the energy derivative `F = dE/dD` by finite
differences on a trial density, terminal-state ownership (mutating result
arrays does not change the frozen build), and the output identities.

### 3.4 Symmetry

`vibeqc.periodic.ccm.symmetry` already supplies the cluster-invariant
space-group subgroup, the supercell AO rotations, the atom-pair orbits and the
fold's k-pair star plan (`cderi_symmetry`, opt-in). The shared symmetry work
supplies `periodic_k_symmetry.star_operations` and
`expand_k_matrices_to_full` (the multi-k GDF exchange wedge) and the
qualification contract of `symmetry_shared`: `OperatorContract`,
`QualificationEvidence` and `audit_equivariance`, where a passed probe is not
a production authorization.

The real-Gamma producer consumes these, never reimplements them:

* **channel stars.** Block storage keeps one representative per `q` star and
  transports the others through the star operation at contraction time
  (the fold's star plan already reconstructs them at build time);
* **wedge exchange.** `Kh(k)` is built at wedge `k` only and transported
  across each star with the same transport the multi-k GDF wedge uses, so
  both producers share one implementation and one admission rule (the classic
  fit admits equal-lattice-shift operations; the pair-centred fit admits all);
* **qualification.** At build time the producer runs `audit_equivariance` on
  a nonsymmetric trial density (`F[g.D] = g.F[D]` for every admitted
  operation) and records the `QualificationEvidence` in `cderi_info`. The
  reduction is opt-in (`symmetry=True` or a `CCMSymmetry`), never a default,
  and refuses an operation the audit fails.

### 3.5 XC

Both producers evaluate the XC term with `build_xc_periodic` on the periodic
Becke grid over the image domain of `periodic_xc_domain.xc_lattice_opts`
(#465, #512; c9d32436 grew the real-Gamma domain to match). That domain made
the real-Gamma LiH/STO-3G PBE SCF fourteen times slower. The track measures
where the cost goes (AO-reach screening statistics against the dense
counts) and recovers it inside the shared kernel, so the Bloch side gains
the same; the identity to the Bloch side at 1e-10 Ha/cell is the gate.

## 4. Increments

Each row lands on its own in a day or two, keeps `main` working and is gated
by the named test. Sizes: S under a day, M one to two days, L a week.

| Id | Increment | Size | Depends on | Acceptance | Status |
|---|---|---|---|---|---|
| RG-1 | XC cost after the grown domain: measure the kernel at 15 bohr and at the #465 bound on LiH/STO-3G PBE, then recover the cost without changing a number | M | none | `E_xc` unchanged to 1e-10 Ha at both domains; LiH PBE (1,1,1) real-Gamma SCF within 2x of its pre-#512 time; `tests/test_ccm_xc_domain.py` green; the #512 xfail in `tests/test_ccm_block_cderi.py` removed once the LiH PBE (2,2,2) row passes | RG-1a landed: the builders ride the kernel's finite-torus regrouping (identity to rounding). RG-1b landed as an opt-in: a profile of the grouped build put about 80% of it in the torus AO sums, because the ball-shaped pair domain makes the rectangle cover degenerate to about one rectangle per image; `xc_torus_complete=True` on the KS drivers (kernel `torus_complete=True`) sums every pair of images reaching a batch through its torus residue instead. It is the domain-converged XC term, not the truncated one: E_xc at the 31.4-bohr bound and at 40 to 60 bohr agree to 7e-12 Ha on LiH (1,1,1), but on the H2 cube (2,1,1) the bound (14.0 bohr) truncates E_xc by 3e-10 Ha, which the complete density removes (it equals the ungrouped kernel at 30 bohr to 3e-13). LiH PBE (1,1,1): XC build 4.2 s against 75.9 s (pair graph) and 214.9 s (ungrouped), six threads; SCF 40 s against 444 s, 2.9e-11 Ha/cell apart, and 61 s for the pre-#512 15-bohr kernel. Because it changes the term at the bound's truncation level (up to 1.5e-10 Ha/cell) while the Bloch producer keeps the pair graph, it is opt-in and recorded in the neutral-source identity; making it the default on both producers is a section 5 decision. The #512 xfail is removed: `test_block_identities_rocksalt_222[LiH-pbe]` passes at 1e-10 Ha/cell (30 min on six threads with the pair graph) |
| RG-2 | Source provenance: a `SourceProvenance` record on every direct and Bloch result, in the manifest and QVF vendor section; cross-source Theorem-1 comparisons refused by identity | S | none | `tests/test_ccm_neutral_source.py`: both producers and both runner arms agree on the H2 cube (2,1,1); every Hamiltonian-selecting knob changes the identity; an injected array is refused; the direct-vs-GDF gate of `tests/test_ccm_direct.py` checks it | landed |
| RG-3 | FockBuild for the closed-shell loops (`_rhf_loop`, `_rks_loop`), `result.fock_build`, output fields | M | RG-2 | `tests/test_ccm_fock_build.py` on the H2 cube (2,1,1), HF and PBE: the terminal build is the reported state, its Fock is the derivative of an independent energy functional (2e-9), it owns its arrays, an iteration-capped run reports an evaluated state, the manifest carries the identities; energies unchanged to rounding | landed |
| RG-4 | FockBuild for the open-shell loops (UHF, UKS) | M | RG-3 | `tests/test_ccm_fock_build.py` open-shell rows on the Li doublet cube (2,1,1): the two-channel terminal build is the reported state, its Focks are the per-spin derivatives of an independent energy functional, an iteration-capped run reports an evaluated state, the UKS build carries `E_xc`, the manifest carries the identities; the UHF loop now tests convergence before DIIS | landed |
| RG-5 | Block storage for UHF and UKS | M | none | `tests/test_ccm_block_cderi_open_shell.py`: identity to the dense fold at 1e-10 Ha/cell on the Li doublet cube (2,1,1) for UHF, PBE and PBE0; the drivers build block storage themselves; the Wigner-Seitz partition reaches them | landed |
| RG-6 | Block storage as the runner default for `N_c > 1`; dense stays the reference switch | S | RG-5 | runner parity rows; `cderi_storage` recorded in the manifest | the opt-in landed: `run_periodic_job(method="aiccm", variant="real-gamma", aiccm_cderi_storage="block")` (or `"block-star"`) and `run_real_gamma_scf(cderi_storage=...)`, recorded as `cderi_storage_executed` in the manifest and printed in the `.out` header; refused off real-gamma and with `correlation=`. On the H2 cube (2,2,1) the block and star runs equal the dense energy to 1e-10 Ha/cell with one neutral-source identity (`tests/test_periodic_ccm_real_gamma_adapter.py`). Open: the default itself, a maintainer decision (section 5) |
| RG-7 | Fit-source boundary: `ClassicRSGDFFitSource` wrapping the shared-q and per-pair builders, `DriverFitCacheSource` wrapping the fit cache; the fold and the block storage consume a bound source | M | RG-2 | `tests/test_ccm_fit_source.py` on the H2 cube (2,1,1) and (3,1,1): the default source is the historical fold bit for bit, a foreign source serving recorded fits reproduces the fold and the block storage exactly, the batchings and the driver cache are one operator, undeclared relations are refused | landed |
| RG-8a | `MANTAFitSource` over `PeriodicGDFTiles` and the RI-Q factor, on the domains of MANTA's fitted RI HF, one tile source per kept channel | M | RG-7 | `tests/test_ccm_manta_fit_source.py` on the H2 cube (2,1,1) and (3,1,1): block-storage J/K on MANTA fits equal MANTA's whole-mesh `PeriodicRIFockBuilder` to 1e-16, the dense fold of the fits gives the same J/K, the source records its engine and channels, the SCF converges on it | landed |
| RG-8b | MANTA one-electron source (S, H and nuclear repulsion from MANTA's finite supports) on the HF direct drivers (`oneelectron=`), so a MANTA-sourced torus is MANTA's whole Hamiltonian | M | RG-8a | `tests/test_ccm_manta_one_electron.py`: the one-electron torus equals MANTA's RI HF overlap/core Hamiltonian to 1e-14 and its nuclear repulsion to rounding; real-Gamma RHF on the H2 cube (2,1,1)/(3,1,1) and UHF on the Li doublet (1,1,1) equal MANTA's RI HF energies to 1e-10 Ha/cell; the supercell Madelung constants agree to 7e-14. Open: at N_c > 1 a doublet cell is a supercell singlet in `CCMSystem` but per-cell spin in MANTA (42 mHa/cell apart on Li (2,1,1)); a maintainer decision | landed |
| RG-8c | MANTA-sourced KS: MANTA's periodic XC quadrature as the XC source of the real-Gamma KS drivers | M | RG-8b | real-Gamma RKS/UKS on MANTA sources equals MANTA's fitted KS jobs at the same mesh | landed: `MANTAXCSource` (`xc_source=` and `oneelectron=` on the KS drivers); MANTA's KS domains have the translations and Gamma vectors of its RI HF domains, so the RG-8a/8b sources on the twin HF options serve the KS run. `tests/test_ccm_manta_xc_source.py`: RKS PBE on the H2 cube (1,1,1)/(2,1,1)/(3,1,1) equals `run_periodic_ri_ks_from_cutoffs` to 7e-16 Ha/cell, UKS PBE on the Li doublet (1,1,1) to 1.4e-14; the torus potential is the derivative of the torus energy. Semilocal only |
| RG-9 | Guesses and stability: SAD and Bloch-seeded densities on the direct drivers and the runner arm; frontier-gap report; RHF-to-RHF stability check | M | RG-3 | MgO pob-DZVP-rev2 RHF (1,1,1) lands on the Bloch solution (#442); the HCORE seed on the same cell is reported as a higher stationary point | source landed: an explicit SAD on the four direct drivers, `run_real_gamma_scf` and the runner (AUTO stays HCORE, through `periodic_guess_capabilities(route, requested=...)`); the Bloch seed is READ of `ccm_reference_from_bloch(...).density`; `frontier_gap` on the convention record and `frontier_gap_ha` in the manifest; `stability_check` / `follow_instability` on the RHF driver (`vibeqc.periodic.ccm.stability`: singlet and triplet sectors by Davidson, translation-invariant following). `tests/test_ccm_rhf_stability.py`: the Davidson eigenvalues equal the explicit Hessian, the energy curvature along the mode is four times the eigenvalue, and the uniform equidistant carbon chain at (5,1,1) is a singlet saddle (-0.081 Ha) that the following lowers by 7.8 mHa/cell. Found and fixed on the way: the HF loops reported the orbitals of the last DIIS trial Fock. Acceptance met on the untailed Hamiltonian of #442 (MgO pob-DZVP-rev2 (1,1,1), classic fit at 200 Ha, no RSGDF tail): the Bloch control and the HCORE real-Gamma run reproduce #442's -274.8696749719 and -274.6156706666 Ha/cell exactly; the HCORE state is a singlet saddle (-0.126 Ha); following it (one step), SAD and the Bloch seed all land on -274.8696749719. The Bloch state is singlet-stable (+0.139 Ha) with a weak triplet instability (-0.0074 Ha). Converging the Bloch SCF to 1e-12/1e-10 lets `ccm_reference_from_bloch` accept at its 1e-8 density tolerance (#442's open residual). Open: the tailed variant, whose tail (54,492 Ha, about 7.7e7 plane waves) puts the fit in the #258 cost class |
| RG-10 | Channel-star reduction of the block storage with `audit_equivariance` evidence | M | RG-7 | reduced and unreduced J and K agree to 1e-12 on random nonsymmetric densities for diamond Fd-3m and MgO (2,2,2); SCF energy unchanged to 1e-10 Ha/cell; evidence in `cderi_info` | landed: `cderi_storage="block-star"` keeps the star plan's representative fits and a recipe (operation, two orbital maps) per member; the unitary auxiliary map cancels in the exchange sum, so the reduction is exact for any density. Evidence: `cderi_info["star_storage"]` (accounting and a build-time recipe check, 3e-16 on the H2 cube). `tests/test_ccm_block_star.py`: on compact H2 (2,2,2), 42/64 fits kept, J/K on random nonsymmetric densities equal the materialised storage to 3e-17 and the fully built one to 1.1e-15, RHF/UHF energies through the drivers equal the dense path; the diamond and MgO (2,2,2) rows are slow. Acceptance met on both cells: diamond Fd-3m (2,2,2) passes its slow row; MgO Fm-3m (2,2,2), whose pytest row exceeds 3 h single-threaded (the unreduced classic fold is in the #258 cost class), was run with the same checks on four threads: J and K against the fully built storage 8.3e-13 and 8.8e-13 (K scale 8.2), recipe check 2.2e-15, 24 of 64 fits stored (reduction 2.67), RHF energy 3.4e-12 Ha/cell from the built storage; the three folds took 2340, 2145 and 4573 s. The evidence is the identity check rather than `audit_equivariance`, which tests a different property (equivariance of one operator) |
| RG-11 | Wedge exchange on the block storage through the shared k-star transport | M | RG-10 | as RG-10, plus the measured flop and time reduction recorded on this page | landed: `exchange_wedge=True` builds K on one bra per k-star of the plan (read off its q=0 channel) and transports it with `Om = P_ao diag(e^{i (Rk).S})`, the `periodic_k_symmetry` convention; every call measures the density's distance from the plan's symmetry and builds K in full when it is off (`cderi_info["exchange_wedge"]`). `tests/test_ccm_block_wedge.py`: H2 (2,2,2) builds 6/8 bras and equals the full K on the SCF density to 1e-13; RHF and UHF through the wedge equal the full build with no fallback. Diamond (2,2,2) STO-3G: 4/8 bras, so the bra work of K halves; measured K build 3 ms full against 2 ms wedge on a loaded host (the per-member transport is n_mu^3). The reduction grows with the star size, up to the cluster group order for a general k |
| RG-12 | Fold cost (#258): profile `ao_pair_fourier_transform_bloch` on diamond and MgO (2,2,1) at a 25-bohr list and recover the regression | L | none | the three `tests/test_ccm_symmetry.py` fixtures finish inside their 1800 s budget | partly landed by another route: `RangeSeparatedFitSource` puts the fold on the driver's default SR/LR fit, which needs no tail. On MgO pob-DZVP-rev2 (1,1,1), whose classic tail (54,492 Ha) kept the Bloch fit in `ao_pair_fourier_transform_bloch_multi` for over four hours, the SR control takes 66 s and the SR fold 53 s, and the two producers agree to 5e-11 Ha/cell (`tests/test_ccm_range_separated_source.py`). The SR source admits the star plan and time reversal (961243fd; its canonical auxiliary frame is covariant): on diamond (2,2,2) the SR fold takes 399 s with the plan and 385 s as star storage against 996 s unreduced (four threads), 24 of 64 fits stored, J and K within 4.8e-13 and 5.7e-12 of the unreduced fold at a K scale of 4, the level of the SR builder's integral screen (1.25e-11), whose decisions are not exactly symmetric; on the H2 cube (2,2,2) they agree to 3e-15. The per-channel cost itself is #543: each SR channel recomputes its real-space integrals per `q` and its reciprocal pair transforms per ket, 50 to 140 times PySCF's time at (2,2,2). The fold's SR source now shares the q-independent real-space sources across its channels through the multi-q builder of #543: diamond (2,2,2) unreduced 161.6 s against 833.8 s per channel (four threads), bitwise the same fits. Open: the rest of #543 (owned by the GDF builder), the classic kernel's own cost, and making SR the CCM default on both producers (a maintainer decision, see section 5) |
| RG-14 | Explicit spin-state contract: one `spin_state` keyword on `CCMSystem`, the direct and Bloch producers, both runner arms and `run_periodic_job(aiccm_spin_state=)`; the sector resolved before the shell-aware dispatch; the executed counts on every result, in the manifest and the QVF record; a comparison refused by sector as well as by operator | M | RG-2, RG-4 | `tests/test_ccm_spin_state.py`: the legacy mismatch reproduced with its counts and refused by sector; the per-cell doublet on `(2,1,1)`, `(3,1,1)` (Li) and `(2,2,1)`, `(2,2,2)` (H cube) equal on both producers to 1e-10 Ha/cell with the Bloch projectors folded onto the torus; UKS PBE at `(2,1,1)`; impossible requests, restricted drivers on an open sector, a foreign-sector READ and a non-uniform total on the Bloch producer refused; manifest round trip | landed: `vibeqc.periodic.ccm.spin_state` (`CCMSpinState`, `resolve_ccm_spin_state`, `assert_compatible_spin_state`, `assert_comparable_neutral_results`). `"legacy"` (default) changes no number: real-Gamma pairs a doublet cell by supercell parity, the Bloch producer replicates the cell's counts, and the mismatch (Li STO-3G (2,1,1): 3/3 against 4/2, 4.83 mHa/cell on the classic fit; the 42 mHa/cell of RG-8b was MANTA's def2-SVP-JK source) is now recorded and refused instead of silent. `"per-cell"` and an explicit total multiplicity select one sector on both producers; the Bloch producer refuses a sector that is not `N_c` copies of one pair (Theorem 2 of the theory notes holds for block-circulant per-spin densities with `Tr[D_sigma S] = N_c n_sigma`, which is also the state assumption behind the open-shell representation comparison; Paper 1 states Theorem 1 for closed shells). Found on the way: the Li per-cell quintet at `(2,2,1)` has a degenerate alpha frontier that the multi-k driver occupies fractionally (an ensemble the real-Gamma determinant cannot hold, 11 mHa/cell apart, refused by `ccm_reference_from_bloch`), so a same-sector comparison also needs integer occupations; the H cube doublet carries the multidimensional rows. A total multiplicity fixes `n_alpha - n_beta` only; antiferromagnetic or other broken-symmetry arrangements need a per-cell seed the producers do not take and are not claimed. Scientific acceptance through vq against PySCF: see the #519 note of this increment |
| RG-15 | Occupation identity: an `OccupationRecord` on every result of both producers (integer or ensemble, largest departure, per-k counts, frontier gap, degenerate flag), in both convention records and the manifest (`aiccm_occupation`); `assert_comparable_neutral_results` refuses an ensemble by name before any fold | S | RG-14 | `tests/test_ccm_spin_state.py`: the Li per-cell quintet at `(2,2,1)` is recorded as an ensemble (departure 0.5) and refused; the gapped rows carry equal frontier gaps on both producers; record round trip and manifest | landed. The fix itself, a fixed per-k integer fill, belongs to the shared multi-k driver (pbc-gdf): until it exists a degenerate-frontier torus is a real-Gamma-only row with its recorded frontier |
| RG-13 | Promotion evidence: ladder parity against PySCF and CRYSTAL23 recorded in `studies/aiccm-2026`; the identity tests moved to T2 rows with measured runtimes; manual pages current | M | RG-1 to RG-6 | manifest rows with `measured_runtime_s`; the AICCM-2026 tables | landed except the move to T2, a section 5 decision. Measured runtimes: the 18 identity files under the AICCM lanes' `-m "not slow"` through `scripts/test_gate/run_full_suite.py` (3 jobs, source be62b6aa); 17 pass, 10 to 398 s each and 34 min in sum, and their manifest rows carry `measured_runtime_s` (the tier stays T3, see section 5). `tests/test_ccm_rks_direct.py` did not finish: the per-test 1800 s cap stopped `test_rks_direct_matches_gdf_pure_gamma_post_fix` in the multi-k XC build of LiH PBE (1,1,1), the RG-1 cost. The manual documents the real-Gamma Python controls (storage, wedge, fit, one-electron and XC sources, guesses and stability). PySCF ladder (`studies/aiccm-2026/ladder_pyscf.py`, README section RG-13): the SR fit equals PySCF KRHF `rs_density_fit` with the transported shells to 0.0 to 18 uHa/cell on diamond, LiH, MgO, fcc Be STO-3G (1,1,1) to (2,2,2) and MgO pob-DZVP-rev2 (1,1,1), PySCF's own GDF/RSGDF spread; the classic fit sits 32 to 215 uHa/cell above PySCF GDF. The SR route costs 50 to 140 times PySCF's time at (2,2,2) (#543). CRYSTAL23 (SHRINK 8, `TOLINTEG 8 8 8 8 16`, the transported STO-3G; SHRINK 12 moves diamond by 6 uHa): from (4,4,4) on the PySCF sequences, which the real-Gamma producer equals rung by rung, approach it from below and sit 2.5, 1.0 and 2.5 mHa/cell away at the largest mesh run (diamond (6,6,6), LiH (6,6,6), MgO (5,5,5)); the torus exchange of the two-atom cells converges slowly, so no extrapolated limit is quoted. Open: the RG-10 MgO (2,2,2) row outside pytest; promotion to T2 (section 5) |

## 5. Maintainer decisions

These are not taken by the track; they are listed so they can be taken.

* Removing `AICCM2026DevAExperimentalWarning` from the real-Gamma producer.
* Making block storage the default (RG-6) and, later, any default symmetry
  reduction (the track keeps symmetry opt-in).
* Whether a MANTA-sourced real-Gamma run is exposed through the runner under
  its own selector, and under which name.
* Promoting the identity tests to T2 rows (RG-13).
* The spin convention of a doublet cell on a torus of more than one cell
  (RG-8b, RG-14): the default `spin_state="legacy"` keeps the supercell
  parity pairing on the real-Gamma producer and the per-cell replication on
  the Bloch producer and MANTA; `"per-cell"` makes every producer run the
  uniform aligned sector. Whether `"per-cell"` becomes the default.
* Making SAD the AUTO default of the real-Gamma variant, and running the RHF
  stability check by default as the molecular drivers do (RG-9).
* Making the complete torus XC density (RG-1) the default of the real-Gamma
  KS drivers, together with the Bloch producer's XC kernel so that the two
  stay one operator; it is the domain-converged XC term and much cheaper.
* Moving both neutral producers from the classic fit to the SR/LR fit, the
  multi-k driver's default 3-D source (RG-12): the pin to the classic fit in
  `ri._pin_classic_fit_source` lifts on both halves together.

## 6. Coordination

* **#512** (real-Gamma and Bloch KS split): closed by c9d32436 (option 1,
  real-Gamma grows its XC domain). `codex/qc512-ccm-xc-domain` is superseded;
  its remaining extras are already on `main`. The dormant xfail in
  `tests/test_ccm_block_cderi.py` is removed under RG-1.
* **#513** (stale diamond assertion): `codex/qc513-classic-fit-assertion`.
* **#404** (classic-fit gradient): landed as 4715634f.
* **#441** (tail at every N): landed, changelog d6f60fb6.
* **#442** (READ densities): landed as 6a7a09fd; the SAD, Bloch-seeded and
  stability parts are RG-9.
* **#472** (unified Fock): the landed `vibeqc.fock_state` API is consumed as
  is; `codex/fock-rhf-evaluated-state` and `codex/unified-operator-contract`
  continue on the Gamma producers. The CCM adoption lives under
  `periodic/ccm/` only.
* **Shared symmetry** (the `codex/symmetry-*` merges): consumed through
  `symmetry_shared`, `periodic_k_symmetry` and `ccm/symmetry.py`; no change
  to those modules from this track. Requests go on the tracker.
* **MANTA**: consumed through `vibeqc.manta` (`PeriodicGDFTiles`,
  `PeriodicRIQBuilder`) and the receipts they carry. A binding the adapter
  needs and does not find is requested on the tracker, not added here.
