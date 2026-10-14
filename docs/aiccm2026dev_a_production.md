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
| RG-1 | XC cost after the grown domain: measure the kernel at 15 bohr and at the #465 bound on LiH/STO-3G PBE, then recover the cost without changing a number | M | none | `E_xc` unchanged to 1e-10 Ha at both domains; LiH PBE (1,1,1) real-Gamma SCF within 2x of its pre-#512 time; `tests/test_ccm_xc_domain.py` green; the #512 xfail in `tests/test_ccm_block_cderi.py` removed once the LiH PBE (2,2,2) row passes | RG-1a landed: the builders ride the kernel's finite-torus regrouping (identity to rounding; 427 s to 202 to 237 s per build at the bound, 20 s to 7.5 s at 15 bohr, loaded host). Open: the remaining cost at the bound, measured on LiH (1,1,1): the ungrouped kernel keeps 614813 of 754364 image shells and 30.9 M of 102 M pairs live because the AO-reach screen works at amplitude 1e-12 while the domain bound was derived at 1e-4; the grouped build still spends its time in the kernel's AO evaluation and torus sums. Next: a single-thread profile of the grouped build, then the per-shell reach in the shared kernel |
| RG-2 | Source provenance: a `SourceProvenance` record on every direct and Bloch result, in the manifest and QVF vendor section; cross-source Theorem-1 comparisons refused by identity | S | none | `tests/test_ccm_neutral_source.py`: both producers and both runner arms agree on the H2 cube (2,1,1); every Hamiltonian-selecting knob changes the identity; an injected array is refused; the direct-vs-GDF gate of `tests/test_ccm_direct.py` checks it | landed |
| RG-3 | FockBuild for the closed-shell loops (`_rhf_loop`, `_rks_loop`), `result.fock_build`, output fields | M | RG-2 | `tests/test_ccm_fock_build.py` on the H2 cube (2,1,1), HF and PBE: the terminal build is the reported state, its Fock is the derivative of an independent energy functional (2e-9), it owns its arrays, an iteration-capped run reports an evaluated state, the manifest carries the identities; energies unchanged to rounding | landed |
| RG-4 | FockBuild for the open-shell loops (UHF, UKS) | M | RG-3 | `tests/test_ccm_fock_build.py` open-shell rows on the Li doublet cube (2,1,1): the two-channel terminal build is the reported state, its Focks are the per-spin derivatives of an independent energy functional, an iteration-capped run reports an evaluated state, the UKS build carries `E_xc`, the manifest carries the identities; the UHF loop now tests convergence before DIIS | landed |
| RG-5 | Block storage for UHF and UKS | M | none | `tests/test_ccm_block_cderi_open_shell.py`: identity to the dense fold at 1e-10 Ha/cell on the Li doublet cube (2,1,1) for UHF, PBE and PBE0; the drivers build block storage themselves; the Wigner-Seitz partition reaches them | landed |
| RG-6 | Block storage as the runner default for `N_c > 1`; dense stays the reference switch | S | RG-5 | runner parity rows; `cderi_storage` recorded in the manifest | planned |
| RG-7 | Fit-source boundary: `ClassicRSGDFFitSource` wrapping the shared-q and per-pair builders, `DriverFitCacheSource` wrapping the fit cache; the fold and the block storage consume a bound source | M | RG-2 | `tests/test_ccm_fit_source.py` on the H2 cube (2,1,1) and (3,1,1): the default source is the historical fold bit for bit, a foreign source serving recorded fits reproduces the fold and the block storage exactly, the batchings and the driver cache are one operator, undeclared relations are refused | landed |
| RG-8a | `MANTAFitSource` over `PeriodicGDFTiles` and the RI-Q factor, on the domains of MANTA's fitted RI HF, one tile source per kept channel | M | RG-7 | `tests/test_ccm_manta_fit_source.py` on the H2 cube (2,1,1) and (3,1,1): block-storage J/K on MANTA fits equal MANTA's whole-mesh `PeriodicRIFockBuilder` to 1e-16, the dense fold of the fits gives the same J/K, the source records its engine and channels, the SCF converges on it | landed |
| RG-8b | MANTA one-electron source (S, H, nuclear repulsion and the BvK Madelung seam from MANTA's finite supports) on the direct drivers, so a MANTA-sourced torus is MANTA's whole Hamiltonian | M | RG-8a | real-Gamma on MANTA fits and MANTA one-electron equals `run_periodic_ri_rhf_from_cutoffs` at the same mesh to 1e-10 Ha/cell on the H2 cube and LiH | planned |
| RG-9 | Guesses and stability: SAD and Bloch-seeded densities on the direct drivers and the runner arm; frontier-gap report; RHF-to-RHF stability check | M | RG-3 | MgO pob-DZVP-rev2 RHF (1,1,1) lands on the Bloch solution (#442); the HCORE seed on the same cell is reported as a higher stationary point | planned |
| RG-10 | Channel-star reduction of the block storage with `audit_equivariance` evidence | M | RG-7 | reduced and unreduced J and K agree to 1e-12 on random nonsymmetric densities for diamond Fd-3m and MgO (2,2,2); SCF energy unchanged to 1e-10 Ha/cell; evidence in `cderi_info` | planned |
| RG-11 | Wedge exchange on the block storage through the shared k-star transport | M | RG-10 | as RG-10, plus the measured flop and time reduction recorded on this page | planned |
| RG-12 | Fold cost (#258): profile `ao_pair_fourier_transform_bloch` on diamond and MgO (2,2,1) at a 25-bohr list and recover the regression | L | none | the three `tests/test_ccm_symmetry.py` fixtures finish inside their 1800 s budget | planned |
| RG-13 | Promotion evidence: ladder parity against PySCF and CRYSTAL23 recorded in `studies/aiccm-2026`; the identity tests moved to T2 rows with measured runtimes; manual pages current | M | RG-1 to RG-6 | manifest rows with `measured_runtime_s`; the AICCM-2026 tables | planned |

## 5. Maintainer decisions

These are not taken by the track; they are listed so they can be taken.

* Removing `AICCM2026DevAExperimentalWarning` from the real-Gamma producer.
* Making block storage the default (RG-6) and, later, any default symmetry
  reduction (the track keeps symmetry opt-in).
* Whether a MANTA-sourced real-Gamma run is exposed through the runner under
  its own selector, and under which name.
* Promoting the identity tests to T2 rows (RG-13).

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
