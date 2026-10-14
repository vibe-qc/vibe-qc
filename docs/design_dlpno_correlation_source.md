# DLPNO-CCSD(T) as one engine: the correlation-source contract for molecules and crystals

Proposed 2026-10-09. This page records the design under which the molecular
DLPNO-CCSD(T) engine becomes the single local coupled-cluster engine for
molecules and for periodic systems, and what "production ready" means for
each. It changes no number until its first increment lands, and the decisions
listed at the end belong to the maintainer. Evidence for the molecular state
is in the [DLPNO guide](user_guide/dlpno_mp2.md#accuracy) and on #190 and
#506.

## Where the code stands

**One engine, measured.** `python/vibeqc/dlpno/ccsd_local_solver.py` and
`python/vibeqc/dlpno/uccsd_local_solver.py` solve the closed- and open-shell
DLPNO-CCSD equations in per-pair PNO spaces through the compiled kernels in
`cpp/src/dlpno_ccsd.cpp` and `cpp/src/dlpno_uccsd.cpp`, followed by the
triples of `python/vibeqc/dlpno/triples_local.py`. On 17 closed-shell
molecules at the shipped defaults the engine matches ORCA 6.1.1 on CCSD and
recovers more of the canonical (T). Its inputs are narrow: MO coefficients,
the AO Fock matrix and the AO overlap, the reference energy and electron
count, and a density-fitting object that answers `mo_transform(C_left,
C_right)` with the three-centre factors `B[P, p, q]` over a real orthonormal
auxiliary frame.

**Historical periodic routes, none production.** The toroidal MP2 and
CCSD(T), CCM coupled CCSD(T), and chi MP2 and closed- and open-shell
wrappers pass explicit correlation sources to their engines on a real
Gamma supercell. Their finite Hamiltonians and research defaults remain
distinct. The CCM union-PNO calculation stays a small-cell oracle, and the
native chain is separate.

| Route | What it runs | Reference and factors | Engine reuse | Label |
| --- | --- | --- | --- | --- |
| Megacell, `periodic_megacell_mp2.py` | DLPNO-MP2, CCSD, CCSD(T) on an open finite supercell | molecular RHF and molecular DF | complete, through `run_job` | shipped, no warning, no catalog row |
| Toroidal, `periodic_toroidal_mp2.py` | the same on the Gamma BvK supercell | Gamma GDF HF and the Stage 4 compensated-cell factors | MP2 and CCSD(T) through `toroidal-compcell` source with explicit cyclic overlap; CCSD(T) retains pair mode | Stage 5a to 5c, no warning |
| CCM neutral torus, `periodic/ccm/dlpno*.py` | union-PNO CCSD(T), coupled CCSD(T), UMP2, UCCSD(T) | neutral GDF factors, Bloch results folded to the real supercell | coupled CCSD(T) through `ccm_neutral_source` and the shared engine; union-PNO and spin-orbital oracles retained | `AICCM2026DevAExperimentalWarning` |
| chi, `periodic/chi/posthf.py` | MP2, DLPNO-MP2, UMP2 and DLPNO-UMP2, CCSD(T), UCCSD(T) pilot | k-mesh RSGDF or GDF fit turned into an exact real torus through `_BRealDFAdapter` | local MP2 and CC wrappers through `chi-real-torus` sources with explicit cyclic overlap and compact home-auxiliary factors; UHF keeps both spin spaces; CCSD(T) retains pair mode; screens fail closed | `AICCM2026DevBExperimentalWarning`, library only |
| Native chain, `cpp/src/periodic_gaussian_*`, `bounded_restricted_*` | pair MP2, Galerkin pair CCSD, TNO coupled (T) | its own finite Gaussian reciprocal source, full Gamma-centred mesh | none; its header says it is "not the pair-specific production DLPNO algorithm" | private bindings, `production_dlpno()` false |

**The subsystems the periodic engine must respect.**

* *Unified Fock build.* The shared chokepoint is the `JKBuilder` interface
  (`cpp/include/vibeqc/jk_builder.hpp`), which serves molecular SCF and
  periodic Gamma. Every multi-k driver assembles its own Fock from retained
  GDF factor caches (`RSGDFFitCache`, kept with `retain_fit_cache=True`) and
  applies the exchange zero-mode convention itself. There is no unified Fock
  object, and `docs/design_shared_symmetry.md` states that one is not a
  prerequisite. Molecular post-HF today builds a fresh `DensityFitting` with
  the RI auxiliary basis; the CCM line reuses the SCF fit cache.
* *Symmetry reduction.* `symmetry_reduce_k` reduces the multi-k exchange cost
  only; the orbitals stay on the full mesh. The EWALD3D wedge drivers keep
  orbitals at IBZ points only, and no production helper transports them back;
  `cpp/include/vibeqc/periodic_mean_field_state.hpp` rejects IBZ-only input by
  contract, and `periodic_correlation_admitted_reference.hpp` refuses symmetry
  reduction until a verified unfold and orbital-sewing object exists. No
  correlated method consumes IBZ-only quantities.
* *MANTA.* `manta.run_periodic_rhf` and its UHF and KS siblings return per-k
  overlap, core Hamiltonian, Fock, coefficients and orbital energies on an
  explicit finite support, for full regular meshes only. MANTA has no periodic
  density-fitting factors (its periodic RI is stage P3b, unimplemented), and
  nothing in correlation imports it.
* *BIPOLE.* The Ewald-split periodic SCF route in the CRYSTAL gauge, a sibling
  of GDF rather than a density-fitting method. No correlated code consumes
  BIPOLE quantities; its finite correlation source exists for tests only and
  reports `physical_hamiltonian_certified=False`. Decision D93 in
  `docs/aiccm2026dev_b_decisions.md` marks post-HF on such references as not
  qualified until a correlation-factor support contract exists.

## The design

### One engine

The molecular solvers stay the engine: PNO-space CCSD in the compiled kernels,
the closed-shell `(T1)` and local `(T0)`, the open-shell `(T0)`, the options
and result objects, the output and citation routes. Nothing periodic gets its
own coupled-cluster loop. The CCM transcription is retired and the chi and
toroidal MP2 and CC wrappers use the contract below; the native chain
keeps its role as an independent oracle and as the producer of translational
factors for the k-aware increment.

### The contract

A `CorrelationSource` protocol in `python/vibeqc/dlpno/source.py` is the
only thing the engine reads. It has three parts.

* *Reference*: reference kind (RHF, UHF, ROHF), per-spin MO coefficients and
  AO Fock matrices, AO overlap, reference energy, electron count, frozen-core
  count, and a real orthonormal AO frame. For a crystal this is the real BvK
  supercell fold of a full-mesh Bloch result, built once by the adapter.
* *Factors*: `mo_transform(C_left, C_right)` returning real
  `B[P, p, q]`, the auxiliary dimension, the auxiliary-basis provenance, and
  an identity: the SHA-256 of everything that defines the fitted Coulomb
  operator (auxiliary basis, lattice options, mesh, supports, Ewald parameter,
  exchange zero-mode convention). Periodic adapters report the minimum-image
  distance function the pair screens and `coupling_radius` must use.
* *Conventions*: the exchange zero-mode convention of the reference
  (`exxdiv`, Madelung share), the Bloch normalisation, the k-mesh descriptor,
  and `same_source`, true only when reference and factors come from one
  Hamiltonian.

`run_local_dlpno_ccsd(molecule, basis, rhf, df, options)` keeps its
signature as a shim that wraps its arguments into the molecular adapter, so
every molecular result stays bit-identical. The `.system` manifest and the
structured log record the source identity next to the thresholds.

Adapters:

| Adapter | Reference | Factors | State |
| --- | --- | --- | --- |
| `MolecularDFSource` | `RHFResult`, `UHFResult`, ROHF | `DensityFitting` with the RI auxiliary basis | production |
| `GammaGDFSource` | `run_pbc_gdf_rhf` and siblings | `PeriodicGammaDF` from `build_periodic_gamma_df` with the SCF's options | increment B |
| `BvKSupercellSource` | full-mesh `run_krhf_periodic_gdf` and siblings with `retain_fit_cache=True`, folded to the real supercell by `bloch_reference.as_real_reference` | the retained fit cache folded by `ccm_neutral_cderi_fold` or the chi home factors | increment B |
| `MANTAReferenceSource` | `PeriodicMANTAResult` folded to the real supercell | a GDF-type fit on the same supports until MANTA stage P3b provides factors; `same_source` false and refused by default | increment E |
| BIPOLE references | refused with the reason: no certified same-source factors | | until a certified source exists |
| IBZ-only references | refused with the reason: no certified orbital transport | | until the unfold object exists |

### Consistency rules

1. **Same source.** A production correlation energy pairs a reference with
   factors from the same Hamiltonian, and the output carries their identity.
   Mixing is a diagnostic that must be requested explicitly and is labelled
   as such. This is the BIPOLE and chi principle applied everywhere.
2. **Full mesh at the boundary.** The engine never sees an IBZ. Symmetry
   reduction lives in the SCF (exchange acceleration) and, later, in an
   unfold-and-audit step that reconstructs the full mesh with
   `apply_periodic_ao_bloch_operation` and the orbital-sewing audits before
   admission. Reductions inside the correlation step, such as unique pairs,
   are validated against the unreduced build, as the shared symmetry gates
   require.
3. **The reference's Fock and conventions.** Denominators and the
   semicanonical steps use the reference's own Fock, including its exchange
   zero-mode term; the factors use the bare periodic Coulomb kernel on the
   torus with the G=0 handling the reference used, and the manifest records
   both.
4. **Reuse the SCF fit where it exists.** Periodic adapters consume the
   retained fit cache rather than refitting with other parameters. The
   molecular adapter keeps the RI auxiliary basis, and the difference from
   the JK fit is documented, not hidden.
5. **No new coupled-cluster loop.** A new reference kind gets an adapter and
   an exactness ratchet (full domains equal the canonical method on the same
   source to within the convergence tolerance), never a transcription of the
   engine.

## What production means

A route is production when all of the following hold, and the result object,
the `.out` block and the docs say so with numbers.

1. An exactness ratchet against the canonical method on the same source.
2. A matched accuracy measurement at the shipped defaults against canonical
   references and against ORCA where ORCA can run the same inputs.
3. Measured cost against molecule or cell size, stated as measured; no
   scaling claim before the measurement exists.
4. Same-source identity in every artifact; refusals that name their reason.
5. Documentation, changelog fragment and citation routes for every published
   algorithm the route executes.

Molecular closed shell meets 1, 2, 4 and 5; 3 is the open C20/C40/C60 ladder
on #190. The 17-molecule TightPNO table is now complete: CCSD / CCSD(T)
mean absolute errors are 0.140 / 0.194 kcal/mol against canonical
DF-CCSD(T), including C10H22 at +0.595 / +0.831 kcal/mol. Matched-build
triples timings and n-butane threshold probes are in the
[guide](user_guide/dlpno_mp2.md#accuracy); they do not establish CCSD size
scaling or change any default. Independent verification of the (T1)
screening fix remains open on #506.

Molecular open shell has its full-domain exactness tests, source artifacts
and citation routes, but its broader accuracy gate is open: HCO/cc-pVDZ
returned non-finite local (T0) corrections at NormalPNO and TightPNO after
CCSD convergence in the 2026-10-10 campaign. The native kernel now checks
the triple-space SVD for finiteness, orthonormality and its retained range,
and retries a failed decomposition with Jacobi SVD;
the solver rejects any remaining non-finite correction with an error.
The closed-shell local `(T0)` kernel reproduced the same basis-dependent
failure on an algebraic pair-space control and shares the checked union
builder and non-finite-result policy.
All nine radicals converge with finite energies at both presets in the
rebuilt-core check; canonical C6/C8 radical references are still missing.
The broader radical accuracy table and cost gate remain open. The periodic
GDF route has the H2 exactness evidence described under B1 and B2
and source identity in the library and runner artifacts. Its broader
canonical comparisons, matched accuracy on crystals and cost against cell
size remain open; it is not a production route.

## Increments

Status 2026-10-11: A, B1, the B2 fold, the B2b front door and C are on main;
the periodic routes carry their exactness evidence and are not yet
production routes (see the gates above).

* **A. Contract and molecular adapter.** Landed: `vibeqc.dlpno.source`,
  the molecular adapter and the shim, the `source` keyword on both
  engines, the source identity in the manifest and the structured log,
  and tests that every molecular DLPNO result is bit-identical before
  and after. No periodic behaviour changed.
* **B1. Gamma-point GDF route.** Landed: `vibeqc.periodic_dlpno.run_periodic_dlpno_ccsd`
  and `gamma_gdf_source`. Two facts shaped it: the k-mesh GDF driver retains
  its fit at Gamma only with `fit_source="classic"` (the default source's
  Gamma fast path keeps nothing), and the Gamma eigenvectors come back
  complex with arbitrary phases, so the adapter realifies them block by
  block and re-diagonalises the real Fock inside each block. Evidence: the
  Coulomb energy rebuilt from the factors equals the SCF's, and full domains
  reproduce canonical CCSD on the same source. Still open for B1 to count as
  production: the matched accuracy table and the measured cost.
* **B2. BvK-supercell route.** Landed: `bvk_supercell_source` and
  `run_periodic_dlpno_ccsd(kmesh=...)`. The SCF runs on the unreduced
  Gamma-centred mesh in cyclic order with its classic fit retained; the
  per-k overlap, Fock and every retained `(k_i, k_j)` factor are
  inverse-transformed to the real torus with the finite-character kernels
  the chi line owns (Theorem 1 of `periodic/ccm/bloch_reference.py`), the
  real supercell Fock is re-diagonalised in the canonical orthogonal frame
  and its spectrum checked against the union of the per-k spectra, and the
  engine solves one real supercell problem with the minimum-image distance
  of the supercell. Evidence: on the H2 chain in STO-3G the folded (1,1,2)
  and (1,1,3) meshes reproduce the explicit Gamma supercell to 2e-13 Ha in
  the reference energy and 1e-15 Ha in the correlation energy, with
  triples, and full domains equal the canonical CCSD on the folded source.
  Follow-up for the maintainer: the fold kernels are private helpers of
  `periodic/chi/posthf.py`; promoting them to a shared periodic module is
  a small move that touches a file the chi sessions edit every few days,
  so it was left for an explicit decision.
* **B2b. The `run_periodic_job` front door.** Landed:
  `run_periodic_job(method="dlpno-ccsd(t)")` for 3D GDF references at Gamma
  and on full meshes, through the two periodic adapters, with
  the minimum-image distance in the pair screens and the shipped thresholds
  rather than the pre-#140/#448 pins the toroidal and chi routes carry.
  The reference runs on the k-mesh GDF engine with `fit_source="classic"`
  and its fit retained; the result lands on the `.out`, the `.system`
  manifest, the structured log, the citations (route keys
  `periodic-gdf-dlpno-ccsd` and `periodic-gdf-dlpno-ccsd(t)` plus the
  runtime-gated triples kernel) and the QVF. Refused before any SCF work,
  with the reason: BIPOLE and symmetry-reduced references (decision 2),
  functionals, `dim < 3`, smearing, gradients and Hessians. Open shells
  use increment C below.
  The provenance helpers both runners share live in
  `vibeqc.dlpno.provenance`.
  For a universal JK fitting basis shared by both codes, pass
  `--aux def2-universal-jkfit` to the parity script. Its legacy spelling
  `--aux def2-universal-jk` resolves to that same catalog name for every
  Vibe-QC call, including the explicit supercell check, and for PySCF.
  Per-zeta `def2-{svp,tzvp,tzvpp,qzvp,qzvpp}-jk` selections retain their
  Vibe-QC names and use the corresponding PySCF `-jkfit` aliases. These
  per-zeta records match the auxiliary blocks those PySCF aliases load.
  The script resolves the periodic driver's default and checks PySCF can
  load every selected auxiliary element before running Vibe-QC. Missing
  reference blocks fail explicitly; the script never substitutes a default.
  Evidence measured on 2026-10-10 with
  `examples/regression/parity_periodic_dlpno_vs_pyscf.py` (H2 molecular
  crystal, STO-3G, same def2-SVP-JK auxiliary): E_HF and the full-domain
  CCSD(T) correlation energy per cell equal PySCF KRHF/GDF and KRCCSD(T) to
  0.00 µHa on (1,1,1), (1,1,2) and (2,2,2) when PySCF keeps the Ewald
  exchange term in the CC Fock (`keep_exxdiv=True`), and in def2-SVP the
  (T) correction per cell on (1,1,2) agrees to 0.001 µHa; the fold equals the
  explicit Gamma supercell to 1e-13 Ha; the shipped NormalPNO thresholds sit
  23 µHa (1,1,2) and 4 µHa (2,2,2) per cell above full domains. Still open:
  the exactness ratchet against the canonical supercell CCSD(T) of the chi
  and CCM lines, the n-alkane chain as the 1-D polymer limit of the
  molecular table, accuracy at the shipped thresholds on real crystals, and
  cost against cell size.

  **Toroidal and chi migration:** toroidal DLPNO-MP2 and CCSD(T), chi
  DLPNO-MP2, UMP2/DLPNO-UMP2 and the unrestricted CC pilot pass the cyclic
  overlap through `CorrelationSource`. The closed-shell CCSD(T) migration
  preceded the MP2 and unrestricted migration. No wrapper replaces the
  global molecular overlap function during the solve. The MP2, UMP2 and
  unrestricted pilot engines keep their molecular positional calls and also
  accept a `source` keyword. Their results and wrapper results name the
  source and its identity. The identity hashes the actual Fock, overlap,
  orbitals and stored factors, with the lattice, mesh and exchange
  convention. Chi retains its compressed home-auxiliary factors; the
  identity does not require translated AO copies. Validated classic fit
  reuse reports `same_source=True`; a separate chi correlation refit or the
  toroidal compensated-cell fit reports false.

  The unrestricted source preserves separate alpha and beta coefficients,
  Fock matrices, orbital energies and electron counts on the common cyclic
  AO metric. It does not use the restricted real-torus source constructor.
  Frozen-core spaces, the total-to-per-cell normalisation, exchange
  zero-mode convention and admission guards are unchanged. Wrappers copy
  caller options before applying route controls and engines preserve the
  supplied reference, including when a calculation raises.

  This migration preserves the historical thresholds and pair-mode residual.
  Under PNO truncation that residual projects coupled amplitudes into the
  target pair before contraction, unlike the source-space contractions of
  [Riplinger and Neese 2013](https://doi.org/10.1063/1.4773581), Sec. II C,
  Eqs. (26)-(30). Their Eq. (30) also makes the metric dependence of
  cross-pair PNO overlaps explicit: each contraction must use the metric
  of its own reference. Production settings use the shared `pno` residual;
  correcting a historical residual is a separate scientific change. The
  finite BvK representation agrees with the construction discussed by
  [Ye and Berkelbach 2024](https://doi.org/10.1021/acs.jctc.4c00936), Sec. 2,
  Eq. (1); it does not implement their fragment LNO approximation or the
  unique home-cell pairs of
  [Nejad et al. 2025](https://doi.org/10.1063/5.0290816), Sec. II,
  Eqs. (11)-(16), and Sec. IV C. Translational reduction remains increment D.
  Removing global metric replacements checks calculation isolation, not
  periodic production accuracy. Bounded tests of the finite Hamiltonian do
  not establish broader crystal accuracy or cost scaling.

  **CCM migration (decision 1):** `ccm_dlpno_ccsd_coupled` now delegates
  pair construction, iteration and triples to the shared engine through
  `periodic.ccm.dlpno_source.ccm_neutral_source`. The adapter retains the
  cyclic AO overlap and Fock, uses the supercell minimum-image distance,
  and localises only active occupied orbitals. Its identity hashes the
  actual factors, Fock and overlap. `same_source` is true only for retained
  factors or an identical explicit array; a legacy refit or unverified
  explicit replacement reports false. The experimental label remains.

  The API keeps its Mulliken domains, legacy PNO density, unscreened pair
  list, uncorrected PNO tail and local (T0) default. The residual now
  contracts terms in the source amplitude's PNO space, following Riplinger
  and Neese 2013, Sec. II C, Eqs. (26)-(30), instead of first projecting
  every amplitude into the target pair. This fixes lost pair-interaction
  contributions under truncation; it intentionally changes truncated
  energies. `triples_mode="dense"` uses the shared exact oracle, which
  diagonalises the active occupied and virtual Fock blocks even for a
  pre-localised source. The union-PNO driver remains a separate reference.

  Regression evidence: on the four-cell H2 chain in STO-3G, with
  `tcut_pno=1e-4`, full atom domains, Pipek-Mezey occupieds and full
  occupied coupling, the retired loop's cyclic CCSD correlation energy
  differs from the full-virtual residual by 42.34 µHa. The shared PNO
  engine agrees within 1 nHa at the same retained pair spaces. The tests
  also cover a frozen occupied orbital, canonical CCSD(T) at full domains,
  and a folded (1,1,2) Bloch reference. These check the adapter and residual,
  not production accuracy on real crystals.
* **C. Open shell on crystals.** Implemented through the same Gamma and
  BvK adapters for `run_kuhf_periodic_gdf` and `run_local_dlpno_uccsd`.
  Both spin Fock matrices and occupied spaces are retained independently;
  overlap and the retained fit are shared. On an N-cell mesh the
  multiplicity is `1 + N * (multiplicity_cell - 1)`, and frozen orbitals
  scale by N. Integer occupations, a global occupied-virtual gap in each
  spin, a frozen-active gap when freezing core orbitals, full orbital
  rank, and both folded spectra are checked before the
  engine runs. The library and `run_periodic_job` select UHF for open
  shells; artifacts and citation routes record the reference and spin
  counts. The open-shell accuracy set remains the molecular control;
  this wiring does not qualify the periodic production gate.

  Focused checks in `tests/test_periodic_dlpno_gamma.py` use H3 doublets
  in STO-3G at Gamma and on (1,1,2): full-domain UCCSD and (T) agree with
  the dense spin-orbital reference on the same factors within 1e-8 Ha,
  both with zero frozen orbitals and one frozen orbital per cell. The
  folded result also matches an explicit Gamma supercell within 1e-6 Ha
  in HF and CCSD correlation and 1e-7 Ha in (T). The singlet UHF limit,
  empty beta occupied space, independent spin projectors, occupation
  refusals and front-door artifacts are covered. These are adapter
  correctness checks, not a real-crystal accuracy set.

  Literature comparison: Saitow et al. 2017
  ([doi:10.1063/1.4981521](https://doi.org/10.1063/1.4981521)) use shared
  ROHF/QRO spatial orbitals and NEV-PNOs (Secs. II A and II B). Our engine
  retains independent UHF alpha/beta spaces and two-sided pair densities.
  Ye and Berkelbach
  2024 ([doi:10.1021/acs.jctc.4c00936](https://doi.org/10.1021/acs.jctc.4c00936))
  use the same finite BvK equivalence but a spin-restricted fragment LNO
  approximation. Nejad et al. 2025
  ([doi:10.1063/5.0290816](https://doi.org/10.1063/5.0290816)) exploit
  translationally equivalent local orbitals and unique home-cell pairs;
  that reduction belongs to D. C changes the reference adapter and
  dispatch, not the molecular amplitude equations or PNO ansatz.
* **D. k-aware local CCSD(T).** Unique home-cell pairs and translational
  factors `L(q, P)` from the native chain's producers feed the compiled
  PNO-space kernel without a supercell expansion. This is the roadmap's native
  BvK milestone; the production integration needs B's evidence. Compact chi
  auxiliary storage alone does not supply the pair reduction.

  **D1. Finite-torus pair transport and MP2 contractions.** The bounded
  experimental modules `vibeqc.dlpno.translation` and
  `vibeqc.dlpno.translation_mp2` establish the real, restricted algebra for
  supplied translation-covariant occupied and pair spaces. They do not
  change a solver default or add a CC amplitude equation. One AO virtual
  basis is retained per row of the compact translation-pair plan; ordered
  placed pairs resolve to that row, a common cell translation and an
  endpoint-reversal flag. The input occupied coefficients must already be
  in the declared cell-major gauge. A support permutation, orbital-centre
  label, or invariant occupied density alone does not establish that gauge.
  The boundary checks the actual occupied translates, the covariance of
  the AO overlap and Fock, and the orthonormality and occupied-space
  exclusion of each pair basis. Optional frozen occupied coefficients
  carry complete bands in the same gauge; pair bases exclude the full
  active and frozen occupied union.

  With `P_L` shifting AO coefficients by the finite cell displacement L,
  the overlap between representative pair p translated by A and pair q
  translated by B is

  $$O_{pA,qB}=V_p^T S P_{B-A}V_q.$$

  Cross-pair Fock blocks replace S by F, and occupied couplings are
  $f_{iA,jB}=C_{i0}^T F P_{B-A}C_{j0}$. All cell arithmetic is modulo the
  mesh. A source amplitude contracts as $O T_q O^T$; endpoint reversal
  transposes $T_q$, not the overlap. These are the finite-torus forms of
  the metric-weighted cross-pair overlaps in Riplinger and Neese 2013,
  Sec. II C, Eq. (30), and the occupied-coupling terms in Nejad et al.
  2025, Sec. V, Eqs. (42)-(44). Nejad's Sec. IV C, Eqs. (37)-(41), supplies
  the PNO construction, a separate layer from these contractions.

  For an unweighted ordered spatial-pair energy
  $q_p=\sum_{ab}K^p_{ab}(2T^p_{ab}-T^p_{ba})$, a representative contributes
  $m_p(2-\delta_{i0,jL})q_p/N$ per primitive cell, where $m_p$ counts its
  unordered placed-pair orbit and N is the number of cells. Most
  off-diagonal representatives have weight 2. A same-band pair at a
  nonzero self-inverse displacement ($2L=0$) has $m_p=N/2$ and weight 1,
  as does a home diagonal. The same-band stabilizer must preserve the
  pair span, and the supplied tensors must respect endpoint reversal in
  that span. Treating every non-home pair as weight 2 would double count
  the antipodal pair on an even mesh. Energy orbit weights never multiply
  the residual coupling terms: each occupied band and cell enters those
  sums once.

  `translation_mp2_residual` evaluates one residual per representative,
  including both occupied diagonal terms. `translation_pair_energy_per_cell`
  returns the already normalized primitive-cell energy. Both take the
  pair exchange blocks and amplitudes in the plan's representative order;
  pair virtual Fock blocks need not be diagonal. The tensor consistency
  checks do not authenticate their Coulomb-factor provenance or establish
  `same_source`. The API does not build an SCF reference, select PAO/PNO
  domains, screen pairs, add discarded-space corrections or iterate to a
  correlated solution. It requires real restricted references and an
  explicitly validated translation gauge; it does not
  identify arbitrary localized orbitals with Wannier translates, repair
  phaseful or reordered inputs, or constrain independent UHF spaces to a
  shared restricted frame. D2 below connects these spaces to an admitted
  source and a representative PAO/PNO builder. Neither this algebra nor its
  finite-Hamiltonian comparisons establishes periodic production accuracy
  or cost scaling.

  **D2. Source-bound representative PAO/PNO construction.**
  `vibeqc.dlpno.translation_builder.build_translation_pair_pnos` consumes a
  real restricted `CorrelationSource`, the declared mesh, active occupied
  coefficients and optional frozen occupied coefficients. The active and
  frozen projectors must each match their source subspace, independently;
  equality of their union cannot excuse mixing core and active orbitals.
  Both spaces need complete bands in D1's explicit translation gauge.
  Source mesh, finite-torus normalization, convergence, electron count
  including ECP metadata, and AO dimensions are checked. The builder uses
  the source's actual overlap, Fock and factor provider.

  `pair_ao_domains` supplies one explicit AO-index set per representative;
  the default uses every AO. The builder projects out the complete
  active-plus-frozen occupied union and retains the full AO tails of each
  selected PAO column. It reuses the molecular Gram orthogonalization and
  Fock semicanonicalization, constructs $K_{ab}=(ia|jb)$ with the source's
  `mo_transform`, and forms the semicanonical MP2 seed. This follows
  Riplinger and Neese 2013, Secs. II B1 and II B3, Eqs. (23)-(25), and
  Nejad et al. 2025, Sec. IV C, Eqs. (37)-(41). It does not yet implement
  Nejad's OSV-selected pair domains or periodic local density fitting.

  The untruncated PAO exchange block is recomputed at every translated
  origin and compared with its representative before PNO truncation.
  This audits the contracted Coulomb operator rather than demanding an
  identical auxiliary gauge. It uses the factor-provider interface without
  expanding compact chi home-auxiliary storage into translated AO copies.
  This audit adds work; it is not a reduced-work factor builder or a scaling
  result. A successful audit preserves the source's `same_source` value;
  covariance does not establish that correlation factors are the SCF fit.

  The density convention is explicit: `pno_norm="mp2"` selects the shared
  Riplinger normalization, and the default `tcut_pno=0` retains the full
  admitted PAO space. Positive cutoffs use the existing strict occupation
  comparison and largest-PNO fallback. The retained PNOs are
  recanonicalized, and their K blocks and semicanonical seeds are rebuilt.
  Empty PAO spaces or a retained space that breaks the self-inverse pair's
  translation stabilizer are refused, without silently adding PNOs.

  The distinction between published densities matters. With
  $u=2T-T^T$ and $\delta=\delta_{i0,jL}$, Riplinger Eq. (23) uses
  $\widetilde T=2u/(1+\delta)$ and
  $D=\widetilde T T^T+\widetilde T^T T$. Pinski et al. 2015
  ([doi:10.1063/1.4926879](https://doi.org/10.1063/1.4926879)), the paragraph
  after Eq. (18), also uses the contravariant-amplitude density; it does not
  justify the historical bare-amplitude `legacy` density
  $(TT^T+T^TT)/(1+\delta)$. Nejad Eqs. (38)-(39) write
  $D=2(Tu^T+T^Tu)$ without that diagonal divisor. For the same T this equals
  the shared `mp2` density off diagonal and is twice it on a true home
  diagonal. A same-band nonzero displacement is not a diagonal pair.
  The builder keeps the named shared convention and records it rather than
  silently importing a different occupation scale or threshold.

  `TranslationPNOBuild` carries the pair spaces, K blocks, semicanonical
  amplitudes, PAO/PNO ranks and occupations, and a consumed-input identity
  alongside the original source identity. `pno_correction_per_cell` is the
  orbit-weighted difference between semicanonical PAO and retained-PNO
  energies. It estimates PNO truncation within the selected PAO domains;
  it does not recover omitted AO domains or solve occupied-Fock coupling.
  These are prepared inputs and a correction, not a converged MP2 result.
  D3 below connects these prepared inputs to automatic frame/domain
  selection and a compiled iterative MP2 solver. Public wrapper defaults,
  unrestricted equations and historical numerical conventions are unchanged.

  **D3. Automatic frames, population domains and iterative MP2.**
  `vibeqc.dlpno.translation_driver.run_translation_dlpno_mp2` is an
  explicit experimental entry point for an admitted real restricted
  finite-torus source. It connects `select_translation_occupied`,
  `select_translation_pair_domains`, D2's builder and
  `solve_translation_mp2`. It does not change any molecular or periodic
  job-runner dispatch. The default `tcut_mkn=0` and `tcut_pno=0` retain
  full AO domains and the admitted virtual space; positive thresholds
  request separate domain and PNO approximations.

  The occupied selector constructs the active and frozen frames
  independently. For either source subspace C, its metric projector is
  $Q=CC^T S$. It projects unit-S-norm AO seeds, selecting home-cell seed
  families together with all their translates. A deterministic greedy
  rank/conditioning test chooses these families. If their columns form B,
  symmetric Löwdin orthogonalization gives

  $$W=B(B^T S B)^{-1/2}.$$

  The seed frame intertwines AO and column translations; its Gram matrix
  and principal inverse square root commute with the column translations.
  W therefore supplies the required phase-free cell-major gauge without
  changing the source projector. Actual orthonormality, translation and
  separate active/frozen projector agreement are audited. `n_frozen`
  counts spatial orbitals over the whole torus, must contain complete
  bands and selects the source's first occupied columns. The bounded
  selector refuses a projector for which its greedy common-AO seed search
  cannot find a sufficiently conditioned frame. This does not establish
  that no other Wannier gauge exists, nor does the construction optimize
  spatial localization.

  Domain selection evaluates home-orbital gross Mulliken populations on
  all atoms of the finite torus,
  $q_{iA}=\sum_{\mu\in A}W_{\mu i}(SW)_{\mu i}$, and retains an atom when
  $|q_{iA}|>T_{\rm CutMKN}$. Zero is an explicit full-domain setting.
  Only home-orbital masks are thresholded; other masks are their exact
  cell translates, and representative pair domains are the endpoint
  union. Empty orbital domains are refused. Atom AO groups are read from
  repeated cell-major source shell metadata, or supplied explicitly via
  `home_atom_aos` for model sources. The builder retains all AO tails of
  the selected projected columns. It does not add a domain-omission
  correction or silently grow a domain.

  This gauge and domain policy have a deliberately limited literature
  relationship. Riplinger and Neese 2013, Sec. II B1, motivates the initial
  Mulliken atom criterion and orbital-domain union, but its final PAO-tail
  domain enlargement and extended CC pair domains are additional steps.
  Nejad et al. 2025, Sec. III A, Eqs. (17)-(21), uses a Gamma
  density-Cholesky and per-k polar diabatic starting gauge, then optimizes
  the fourth-power Bloch-IAO Pipek-Mezey metric in Sec. III B,
  Eqs. (22)-(23). Its Sec. IV uses integral-screened PAO domains and OSVs.
  The present projector/Löwdin frame and Mulliken masks do not reproduce
  those localization or screening algorithms. Ye and Berkelbach 2024,
  Sec. 2, Eqs. (3)-(5), likewise requires translation-related orbitals
  spanning the occupied space, but uses PM orbitals or IAOs and a fragment
  LNO approximation. Its IAOs also span part of the virtual space. D3
  remains restricted; it does not implement
  Saitow's shared ROHF/QRO and NEV-PNO construction or an independent-spin
  UHF quotient solver.

  The compiled solver iterates only representative amplitudes with a
  damped Jacobi update. It evaluates D1's full raw residual, including
  every placed occupied coupling and both occupied diagonal terms, and
  tests the energy and residual on the same evaluated amplitude state.
  Orbit weights enter the primitive-cell energy only. There is no Fock
  cutoff, denominator floor or DIIS step. This is the real finite-torus
  MP2 equation of Nejad Sec. V, Eqs. (42)-(44), with explicit cross-pair
  overlap transport; it is not a new CC residual or a transfer of the
  historical CC residual convention. Nonconvergence remains visible in
  the result. The semicanonical `pno_correction_per_cell` is reported
  separately from the iterated energy and does not enter its residual.

  `AutomaticTranslationMP2Result` retains `occupied_selection`,
  `domain_selection`, `pair_build` and `solution`, with consumed-input
  provenance. Source identity and `same_source` keep their original
  meaning; deterministic selection and factor-covariance audits do not
  authenticate a correlation refit as retained SCF factors. Dense AO
  frames, explicit all-origin factor audits and cross-pair transport still
  bound this route to small experiments. It establishes no production
  threshold accuracy, optimized locality or measured scaling. The next
  implementation gap is a factor/transport path that avoids those dense
  preparations and audits while preserving the finite-Hamiltonian checks;
  optimized periodic localization and screened domain construction also
  remain open before cost claims or default integration.
* **E. MANTA.** A reference adapter first; factors once MANTA stage P3b
  exists or a GDF fit on MANTA supports is certified as same-source.
* **F. Symmetry.** The unfold-and-audit admission step, then unique-pair
  reduction inside D, each validated against the unreduced build.

## Decisions

Taken by the maintainer on 2026-10-10 (#508), following the recommendations
of this page.

Pending: which Fock operator the periodic amplitude equations see. The
adapters keep the SCF's Fock with the Ewald exchange zero-mode term, so a
finite-mesh correlation energy is the one PySCF computes with
`keep_exxdiv=True`; PySCF's default strips the term, which raises every
occupied orbital energy by the Madelung constant inside the amplitude
equations and lowers the correlation energy per cell by 4.6 to 8.8 mHa on
the H2 crystal (STO-3G and def2-SVP). The two conventions share the thermodynamic limit and differ
in how fast a finite mesh reaches it; the recommendation of this page is to
keep the SCF Fock (one Hamiltonian for reference and correlation) and to
report the convention on every artifact, as the route does now.

1. The CCM transcription `ccm_dlpno_ccsd_coupled` is retired in favour of the
   engine; the union-PNO driver stays as the small-cell oracle.
2. BIPOLE and IBZ-only references are refused for correlation by default,
   with the reason in the error and in the receipt, not run with a warning.
3. Increment B's production front door is `run_periodic_job` with
   `method="dlpno-ccsd(t)"` on the GDF route. The AICCM experimental warnings
   stay on the AICCM and chi routes only.
4. The molecular adapter accepts the SCF's JK fit cache as an opt-in factor
   source for a same-source molecular run; the RI auxiliary basis remains the
   default and the measured convention.

## Sources

Riplinger and Neese 2013; Riplinger, Sandhoefer, Hansen and Neese 2013;
Riplinger, Pinski, Becker, Valeev and Neese 2016; Saitow, Becker, Riplinger,
Valeev and Neese 2017; Guo et al. 2018 and 2020; Sun, Berkelbach and Chan
2017 for the periodic fitting conventions; Nejad, Zhu, Tew et al. 2025 for
the native BvK lineage. All are in the citation database and fire through
the engine's routes.
