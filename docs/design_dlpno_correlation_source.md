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

  For generic factor providers, the untruncated PAO exchange block is
  recomputed at every translated origin and compared with its
  representative before PNO truncation. This audits the contracted Coulomb
  operator rather than demanding an identical auxiliary gauge, without
  expanding compact chi home-auxiliary storage into translated AO copies.
  D4 below adds a constructive covariance path for a specific immutable
  factor representation. Both paths preserve the source's `same_source`
  value; covariance does not establish that correlation factors are the
  SCF fit.

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
  authenticate a correlation refit as retained SCF factors. D4 below
  removes selected dense preparations and repeated transport work. The
  route remains bounded to small experiments and establishes no
  production threshold accuracy, optimized locality or measured scaling.

  **D4. Compact factor and prepared transport paths.**
  `vibeqc.dlpno.translation_factors.TranslationHomeFactors` snapshots real
  home-auxiliary factors in an immutable representation. The full logical
  factor is defined, rather than sampled, by

  $$B_{tP,\mu\nu}=B_{0P,\mu-t,\nu-t}.$$

  Translating both orbital inputs permutes the complete auxiliary-cell
  sum, so the contracted exchange block is translation covariant by
  construction. `pair_exchange` streams that sum one auxiliary cell at a
  time. It neither constructs translated AO-factor copies nor retains the
  full logical auxiliary MO transform. The compatibility `mo_transform`
  still returns that complete transform for existing consumers.

  The builder's `factor_audit="auto"` uses this construction only for the
  exact immutable implementation on a matching mesh and AO dimension.
  A generic provider, including one declaring itself periodic, keeps the
  all-origin audit. `factor_audit="all-origins"` forces the reference path
  for either kind. Provenance distinguishes the paths and records the
  immutable factor identity. `max_translation_error=None` means no
  numerical origin audit was run; it is not a measured zero error. The
  factor construction proves covariance, not retained-SCF provenance or
  a physical Coulomb boundary condition, and never promotes `same_source`.

  PAO preparation now uses a selected-domain, low-rank form of the same
  projection. With C the complete active-plus-frozen occupied union,
  $H_D=C^T S E_D$ and $A_D=E_D-C H_D$, the domain Gram matrix is
  $G_D=S_{DD}-H_D^T H_D$. Its Fock matrix is evaluated as

  $$F_D=F_{DD}-(FC)_D H_D-H_D^T(FC)_D^T
       +H_D^T(C^TFC)H_D.$$

  Subtraction in $G_D$ can lose relative precision for nearly dependent
  PAOs. After the existing rank cutoff selects X, a retained-space
  Cholesky refinement uses $V_0=E_D X-C H_D X$ and the metric action
  $S V_0=S_{:,D}X-(SC)H_D X$. This uses selected full AO overlap columns
  and shared SC, without a dense S-times-pair-basis product. It refines
  the retained metric without a second rank cutoff or threshold change.

  In PAO basis construction, shared occupied contractions replace a full
  virtual projector and repeated full-AO metric/Fock contractions for
  every domain. Both Fock cross terms remain; no diagonal occupied-Fock
  approximation is used.
  `ProjectedPairBasis` retains the local and occupied-tail coefficients
  of $V_p=E_{D_p}A_p-C B_p$ after PNO rotation. Full AO tails and the
  existing linear-dependence and PNO thresholds are unchanged. This is
  the PAO definition in Riplinger and Neese 2013, Sec. II B1, and Nejad
  et al. 2025, Sec. IV A, Eqs. (26)-(28), evaluated in a different order.

  The default native solver path prepares its fixed cross-pair overlaps
  and coupling routes before iteration. Given $P_L C=C R_L$, the compact
  pair form yields the exact overlap

  $$O_{pq}(L)=A_p^T S_{D_p,D_q+L}A_q-B_p^T R_L B_q.$$

  This selected-block expression is the normal preparation path. A
  conservative estimate of roundoff and the measured occupied Gram,
  projection and gauge defects detects targets whose links risk
  cancellation. Their links instead use
  $M_p=A_p^T S_{D_p,:}-B_p^T(SC)^T$ and contract M with translated source
  coefficient cell blocks. This uses selected full AO overlap columns
  and shared SC; it avoids a full dense S-times-pair-basis product and
  a copied translated coefficient array. It changes no pair space,
  rank cutoff or admission tolerance. The additional AO-row contraction
  occurs only during preparation. `TranslationTransport.stable_links`
  counts these links and is recorded as `stable_selected_column_links`
  in transport and solver provenance.

  Both evaluations represent $V_p^T S P_L V_q$. The prepared solve
  retains pair-space links and fixed pair/Fock data, so no AO translation
  or AO metric contraction is performed inside a Jacobi iteration.
  This follows the metric overlap
  of Riplinger Sec. II C, Eq. (30); that section also describes storing
  amplitude-independent pair overlaps before iteration. It does not
  adopt the paper's pair screening or its CC amplitude equations. Nejad
  Sec. V, Eqs. (42)-(44), remains the residual equation, including every
  placed occupied coupling and the existing energy weights.

  `vibeqc.dlpno.translation_transport.prepare_translation_transport`
  returns a reusable `TranslationTransport`, matched to its consumed
  spaces by an input identity. The solver accepts `transport=...` or
  prepares one with `transport_backend="prepared"`, the default.
  `transport_backend="ao"` retains the independent AO reference path.
  Builds lacking projected-basis metadata contract AO pair coefficients
  directly; their prepared iterations still operate only in pair spaces. The
  default `max_transport_bytes=256*1024**2` covers retained native
  numerical/route storage and a conservative preparation workspace, not
  total process memory. Caller arrays, binding copies, allocator overhead
  and later amplitude/residual workspaces are excluded. Exceeding the
  budget refuses preparation without an automatic backend change.
  Link/coupling counts and retained, workspace and required bytes are
  recorded in transport provenance.

  This factor path is not Nejad's periodic local density fitting in
  Sec. VI, Eqs. (45)-(57), which uses local auxiliary domains, Coulomb
  metrics and charge projection. Its Sec. VI C also distinguishes
  translation symmetry of corrected integrals from individual
  origin-dependent lattice and surface terms. D4 consumes already formed
  finite-torus factors and preserves their source and zero-mode
  conventions. Ye and Berkelbach's finite BvK equivalence and
  translation-related occupied spaces remain relevant, but no fragment
  LNO approximation or new localization algorithm is introduced here.

  These are specific storage and reuse changes, not a linear-scaling
  implementation. For M total AOs, N cells and home auxiliary rank a,
  the retained home factors still require $aM^2$ numbers, versus
  $NaM^2$ for explicit auxiliary-cell copies. D5 below adds an explicit
  cyclic-block representation for source overlap/Fock and streams
  occupied-projector checks. Compatibility pair AO coefficients remain
  dense. The streamed pair contraction still visits all N auxiliary
  cells. Prepared links require $\sum_{\rm links}r_p r_q$ numbers and
  retain unscreened occupied couplings; full domains may retain large pair
  ranks.

  **D5. Cyclic-block metric and Fock preparation.**
  `vibeqc.dlpno.translation_operators.CyclicAOOperator(home_row_blocks,
  mesh)` supplies an immutable real operator through all of its home-row
  displacement blocks, shaped `(N, b, b)` for N cells and b AOs per cell.
  Its convention is

  $$A_{R,T}=A_{0,T-R}=B_{T-R},\qquad
    (AX)_R=\sum_\Delta B_\Delta X_{R+\Delta},$$

  with displacement arithmetic modulo the declared mesh. All N blocks
  are retained. The representation does not introduce a distance cutoff,
  an open-boundary matrix, a cell-normalization factor or a change of AO
  ordering. Block actions, transposed actions and selected columns provide
  the metric and Fock operations without constructing an
  $(Nb)\times(Nb)$ AO matrix.

  Symmetry requires $B_{-\Delta}=B_\Delta^T$, including symmetric blocks
  at self-inverse displacements. For Fourier coefficients proportional
  to $e^{ikR}$, the reciprocal symbol is
  $\widehat A(k)=\sum_\Delta e^{ik\Delta}B_\Delta$.
  The overlap is positive definite exactly when every symbol on the full
  finite mesh is positive definite. Admission checks those symbols;
  it does not infer positivity from the home block, clip eigenvalues or
  shift the metric. This follows the unitary finite Fourier transform
  and block-diagonal Bloch overlap/Fock in Nejad et al. 2025, Sec. II,
  Eqs. (2)-(6). Its Fock transformation in Eqs. (15)-(16) likewise
  preserves the reference operator. Our cyclic Gamma-centred mesh
  conventions and treatment of even-mesh self-inverse cells remain
  unchanged; the paper's odd-grid examples are not a new admission rule.

  The source remains real and restricted, and must explicitly supply both
  `source.overlap` and
  `reference.fock` as matching block operators. Dense sources remain
  supported. There is no automatic conversion or migration of periodic
  adapters, and no implicit array expansion. The legacy
  `CorrelationSource.overlap_matrix()` and `transport_backend="ao"`
  refuse block sources; the prepared transport path consumes their
  blocks directly. The caller remains responsible for supplying the
  actual finite-source operators. Structural covariance and positivity
  do not establish a different Fock as source-equivalent or authenticate
  retained SCF factors. Consumed-input identities record the operator
  representation, while source identity, `same_source`, electron counts,
  frozen-core partition and exchange-zero-mode conventions keep their
  existing meanings.

  Operator actions now serve source admission, automatic occupied-frame
  selection, population domains, PAO construction, pair-space validation
  and native transport preparation. Active and frozen projector
  comparisons and hashes stream AO rows instead of retaining full
  projector matrices. Their existing maximum-entry comparison is
  preserved. Occupied-frame selection forms candidate seed Gram matrices
  as needed rather than a full AO-seed Gram matrix. Selected columns
  supply the D4 retained-space metric refinement and stable transport
  path; these preserve full occupied-projection tails and the existing
  PAO/PNO ranks and tolerances. The metric-weighted pair transport remains
  Riplinger and Neese 2013, Sec. II C, Eq. (30), and the MP2 residual
  remains Nejad Sec. V, Eqs. (42)-(44). Ye and Berkelbach 2024, Sec. 2,
  Eq. (1), supplies the same finite unit-cell/supercell equivalence, not
  a new approximation to the interaction or a fragment LNO step.

  Each metric/Fock operator stores $Nb^2$ numbers rather than $N^2b^2$.
  This removes the dense AO overlap/Fock dependency from the explicit
  block-source preparation path; it does not make the whole calculation
  sparse. Source and selected occupied coefficients, pair AO coefficients,
  occupied Fock, pair Gram matrices and home-auxiliary factors still use
  dense arrays. Full AO domains can still create projected matrices of
  dimension Nb, filled by row blocks without a second source AO matrix,
  and all-displacement block actions do not screen work. Existing
  dense adapters still allocate their original matrices. D6 below adds
  an explicit chi adapter that supplies these operators directly from
  its periodic reference. D5 adds no production accuracy, timing or
  scaling claim.

  **D6. Direct chi Bloch-to-block source preparation.**
  `vibeqc.periodic.chi.translation.prepare_chi_translation_source` runs
  a fresh restricted chi RI-HF reference with the accepted-reference
  SCF controls and prepares a `CorrelationSource` for the experimental
  translation driver. The source carries cyclic overlap and Fock
  operators and compact home-auxiliary factors. It does not construct
  dense finite-torus AO overlap or Fock matrices. This is a separate
  explicit entry point with unchanged periodic-wrapper and job-runner
  defaults. D7 below adds optional `ChiAcceptedReference` reuse.

  With the Bloch convention in D5, its home-row transform is

  $$B_\Delta=\frac{1}{N}\sum_k e^{-ik\cdot R_\Delta}A(k),
    \qquad A_{R,T}=B_{T-R},\qquad A\in\{S,F\}.$$

  The inverse sign and single factor of $1/N$ follow from Nejad et al.
  2025, Sec. II, Eqs. (2)-(6), with BvK orbital coefficients in
  Eqs. (9)-(10). The forward symbol in D5 recovers the original Bloch
  operator. Admission requires the full Gamma-centred cyclic mesh and
  uniform weights, compatible time-reversal partners, real home-row
  blocks within tolerance, and positive-definite overlap symbols.
  Per-k orbital spaces must be full rank and satisfy the supplied
  generalized Fock eigenproblem. The adapter refuses unsupported
  matrices or fillings. The existing global overlap-rank tolerance and
  transpose average after admission are preserved, with no new metric
  cutoff, phase averaging or Fock shift.

  Real canonical finite-torus orbitals are formed from time-reversal
  partner eigenspaces with block operator actions. Arbitrary orbital
  phases and rotations within degenerate eigenspaces do not require
  coefficient-by-coefficient conjugate matching. The preserved occupied
  and virtual spaces have a global gap; a nonempty frozen/active split
  requires its own global gap and complete bands. `n_frozen` is the
  total number of frozen spatial orbitals on the torus, and the caller
  passes that same count to `run_translation_dlpno_mp2`. Reference
  energy and electron counts remain extensive, while the MP2 driver
  reports energy per primitive cell with its existing normalization.
  The supplied SCF Fock retains the existing BvK exchange-zero-mode
  term.

  `fit_source="classic"` reuses the canonical factors only after the
  existing chi admission validates the executed source, factor digest,
  auxiliary frame, mesh and numerical support; this is the
  `same_source=True` case. `fit_source="auto"` retains the existing
  separate SCF and classic correlation fits and records
  `same_source=False`. Equal requested cutoffs or constructive factor
  covariance do not promote a refit to retained-source identity. The
  home-auxiliary transformation preserves the supplied finite fit and
  does not introduce Nejad's local auxiliary domains, chargeless
  projection or surface correction from Sec. VI, Eqs. (45)-(65).

  Nejad's Sec. III A, Eqs. (17)-(21), motivates treating the Bloch gauge
  explicitly, but this realification is not its diabatic Wannier or
  optimized IBO procedure. Ye and Berkelbach 2024, Sec. 2, Eq. (1),
  and Sec. 2.1, Eqs. (3)-(5), supply the finite-representation equivalence
  and occupied-span requirement, not their fragment LNO approximation.
  Nejad's odd Monkhorst-Pack grids do not replace our full cyclic
  even/odd mesh contract. This adapter remains restricted, real and
  three-dimensional; it does not widen UHF, ECP or explicit fitted-support
  admission.

  Canonical orbital coefficients remain dense, as do the downstream
  occupied and pair coefficients, projected domain matrices and compact
  home factors described above. Removing dense source S/F folds is a
  representation change, not a new scientific approximation or a
  measured cost claim. D7 below adds generic GDF and accepted-reference
  block adapters; smaller remaining orbital/factor payloads and qualified
  localization and screening remain implementation gaps.
  Periodic DLPNO production accuracy is not established by this increment.

  **D7. Attested generic GDF inputs and reusable chi block references.**
  `vibeqc.dlpno.translation_gdf.gdf_translation_source(system, basis,
  result, mesh=..., aux_basis_name=None, n_frozen=0)` prepares a block
  source from an existing direct multi-k GDF RHF result. It does not run
  SCF or build a replacement fit. Admission requires the executed classic
  source descriptor and its retained canonical AO-pair factor mapping,
  with matching numerical digest, orbital/auxiliary bases, mesh and
  finite support. A legacy cache object alone is not evidence that the
  factors are those used in SCF. The initial route refuses eigenmode
  auxiliary frames rather than inferring an unrecorded auxiliary
  rotation. `same_source=True` follows only after this attestation.

  The adapter preserves the executed `physical_pairs` or
  `historical_index_ball` support policy. It admits `BvK-ewald` exchange
  zero mode and keeps the source Fock and one-electron support. It does
  not silently replace a historical finite fit with the physical-pair
  fit, strip the exchange shift, refit the interaction or change
  normalization. Cartesian-only k-point data are supplemented with
  derived fractional coordinates on a detached input view and checked
  against the requested complete cyclic mesh. The caller's result stays
  unchanged. D6's full-mesh, realification, global-rank and band-gap
  guards apply, and `n_frozen` counts spatial orbitals over the entire
  torus. UHF, ECP, COSX, SR and explicit fitted-support sources,
  explicit reciprocal tails and symmetry-unfolded references remain
  outside this generic adapter.

  `prepare_chi_reference(..., matrix_representation="cyclic-blocks",
  n_frozen=...)` builds a process-local accepted handle with the direct
  block S/F transform and compact home factors. The existing default is
  `matrix_representation="dense"`; legacy consumers continue to require
  dense handles. `prepare_chi_translation_source(..., reference=handle)`
  validates the original geometry/basis, controls, process/runtime,
  state, factor and metadata seals. Fractional and Cartesian k points,
  numerical tensors and their representation are included in consumed
  state validation. A block handle also binds the exact total frozen
  count and reuses all transforms. A dense handle is migrated from its
  sealed Bloch state directly to block S/F, with a fresh audit of the
  requested frozen partition; its existing compact home factors are
  reused. This migration does not undo dense storage already retained by
  the original handle.

  Returned sources have detached orbital/energy arrays and provenance;
  only immutable block operators and compact factors are shared.
  Reusing an `auto` handle reuses its already separate correlation fit
  and remains `same_source=False`. Its `correlation_fit_status="reused"`
  describes an execution stage, not equality to the SCF fit. Accepted
  reference provenance records the seal, consumed state and real-state
  identities, original matrix representation, and whether the Bloch
  transform was reused or built from the sealed Bloch state. Reuse is
  process-local, with no disk cache or relaxed source admission.

  The scientific distinction is between changing a tensor representation
  and changing its finite operator. [Nejad et al. 2025](https://doi.org/10.1063/5.0290816),
  Sec. II, Eqs. (2)-(10) and (15)-(16), and
  [Ye and Berkelbach 2024](https://doi.org/10.1021/acs.jctc.4c00936),
  Sec. 2, Eq. (1), justify the unitary finite Bloch/BvK transform. They
  do not authenticate a cached fit or make a refit source-equivalent.
  [Riplinger and Neese 2013](https://doi.org/10.1063/1.4773581),
  Sec. II C, Eq. (30), likewise requires the metric of the actual pair
  spaces for cross-pair transport; stored transport is reusable only
  for its admitted inputs. D7 changes source preparation and reuse, not
  that pair algebra, PNO thresholds or MP2 residual conventions.

  A bounded generic-GDF comparison uses all-electron RHF H2/STO-3G,
  a 1.4-bohr H-H bond centred in a cubic 6-bohr cell, the `(2, 1, 1)`
  cyclic mesh, `physical_pairs` classic factors, BvK Ewald exchange,
  no frozen core and full AO/PNO spaces. The
  [focused regression](../tests/test_periodic_dlpno_gamma.py) compares
  three contractions of the same executed finite Hamiltonian:

  | MP2 representation | Correlation energy (Eh/cell) |
  | --- | ---: |
  | Block source and representative translation solver | -0.00970871869654531 |
  | Dense finite-torus canonical contraction | -0.009708718696547205 |
  | Independent character-space contraction | -0.009708718696547193 |

  The largest energy difference is $1.90\times10^{-15}$ Eh/cell and
  the maximum raw iterative residual is $3.32\times10^{-14}$. Rebuilding
  SCF Coulomb energy from the retained factors gives zero recorded
  discrepancy for this case. This is an internal finite-Hamiltonian
  comparison; external reference accuracy, mesh/basis convergence and
  broad periodic qualification remain open.

  A separate accepted-reference chi check at the same geometry, basis and
  mesh retains chi's own finite support. It gives
  -0.00970871869562221 Eh/cell through the block translation solver and
  -0.009708718695624109 Eh/cell through its dense finite reference, a
  difference of $1.90\times10^{-15}$ Eh/cell with maximum raw residual
  $3.32\times10^{-14}$. This checks reuse within the chi source; the
  generic and chi comparisons do not establish equivalence between their
  distinct fitted operators.

  Real folded canonical coefficients remain dense. D9 below streams their
  complex preparation from Bloch data. Compact home factors, compatibility
  pair AO coefficients and unscreened pair links remain substantial
  payloads. Reducing these, qualifying localization/screening and extending
  separately attested source coverage are the next implementation gaps. The new
  adapters establish neither production accuracy nor cost scaling.

  **D8. Retained classic chi block adapter.**
  D6-D7 cover conventional GDF and ChiAcceptedReference blocks. The subsequent
  `vibeqc.periodic.chi.local_mp2.build_chi_translation_source` consumes an
  admitted classic physical-pair RI RHF `ChiHFReference`. Direct Fourier
  home-row overlap/Fock blocks preserve the actual SCF Hamiltonian and
  its BvK exchange zero mode. No dense supercell AO overlap/Fock is built.
  Active and frozen occupied subspaces are separately realified and their
  original complex metric projectors audited in streamed rows. The native
  compact home-auxiliary transform must recover every complex input factor
  block on a forward transform. No refit, SCF replay, rank change or
  historical torus source is used.

  `run_chi_local_mp2` connects this source to D3-D5 with independent domain
  and PNO cutoffs, iteration controls and explicit experimental output.
  Raw energy and the semicanonical discarded-PNO estimate are separate;
  no domain correction or pair screening is added. Scalar ECP metadata is
  preserved without claiming #530 acceptance. Unsupported spin, SR,
  reduced/fractional/overlap-truncated and non-real spaces refuse. Dense
  orbital, factor, pair and transport payloads remain; the exposed budget
  covers prepared native transport only. The [chi local-MP2 guide](chi_local_mp2.md)
  specifies the math, memory scope, source checks and reproducible vq cases.
  Scientific acceptance and D93 qualification remain separate from delivery.

  **D9. Streamed canonical orbital realification.**
  The D6-D7 adapters now form each complete degenerate eigenspace directly
  from the primitive Bloch coefficients. For an orbital labelled by k and
  band p, its finite-torus AO column is

  $$C_{R\mu,kp}=N^{-1/2}e^{ik\cdot R}C_{\mu p}(k).$$

  This is [Nejad et al. 2025](https://doi.org/10.1063/5.0290816), Sec. II,
  Eqs. (3) and (9), evaluated only for the columns currently needed.
  The adapter sorts orbital indices and energies, without constructing
  the full complex folded coefficient array or its globally reordered copy.
  Phase panels are formed only for the current eigenspace. The dense and
  streamed entry points use the same real-span rank test, metric orthogonalization, Fock
  rediagonalization and eigenvalue tolerance. A complete degenerate space
  is never split to reduce memory. Frozen/active and occupied/virtual
  gap guards, complete-band counts and arbitrary complex gauges are
  unchanged. Final orthogonality and stationarity checks visit every
  orbital in b-column panels, including overlaps between different
  eigenspaces, with the same maximum-entry residual definitions.

  For M = Nb total AOs and largest degenerate space d, the constructed
  complex coefficient block has Md entries instead of M squared. Phase
  panels have Nd entries. Final validation uses Mb-entry panels instead
  of another full S-times-C array and square Gram/residual workspaces.
  These are array dimensions, not process memory or wall-time bounds;
  eigensolvers and block actions have additional workspace. Broad
  degeneracy can still make d large. Provenance records
  `max_folded_orbital_columns` and `orbital_validation_columns`.

  The default source snapshots and accepted-reference seals retain the real
  M-by-M canonical array. Selection reads its active and frozen occupied
  columns; PAO preparation projects their complete union;
  factor contractions and transport then consume occupied and pair AO
  coefficients. The prepared native solver receives pair-space data.
  D10 below replaces that retained array with optional compact occupied
  Bloch projectors across admission, selection and seals.
  This increment changes no PAO domain, PNO density, factor fit,
  transport overlap, residual, energy normalization or `same_source` claim.

  [Riplinger and Neese 2013](https://doi.org/10.1063/1.4773581), Sec. II C,
  Eqs. (26)-(30), still requires the actual pair metrics and mixed-pair
  contractions downstream. [Ye and Berkelbach 2024](https://doi.org/10.1021/acs.jctc.4c00936),
  Sec. 2, supplies the finite BvK equivalence, not a reason to introduce
  fragment LNO truncation here. [Saitow et al. 2017](https://doi.org/10.1063/1.4981521),
  Sec. II, assumes one ROHF/QRO orbital set for its NEV-PNO construction;
  it does not justify replacing independent UHF spaces. Molecular and
  unrestricted adapters retain their existing calls and orbital spaces.
  This storage change does not establish production accuracy or scaling.

  **D10. Compact occupied projectors across source consumers.**
  `compact_occupied=True` on chi translation preparation, generic GDF
  adaptation or chi block-reference preparation retains independent
  frozen and active home-row metric projectors instead of the real
  M-by-M canonical array. For each occupied partition X,

  $$Q_X(k)=C_X(k)[C_X(k)^\dagger S(k)C_X(k)]^{-1}C_X(k)^\dagger S(k).$$

  The Gram inverse removes normalization roundoff within each partition;
  it cannot rotate frozen orbitals into the active space. The inverse
  finite Fourier transform yields `Q[R,T]=Q[0,T-R]`. These metric
  projectors are generally nonsymmetric and are not transpose averaged.
  This is the occupied projection of
  [Riplinger and Neese 2013](https://doi.org/10.1063/1.4773581), Sec. II B1,
  in the Bloch convention of
  [Nejad et al. 2025](https://doi.org/10.1063/5.0290816), Sec. II,
  Eqs. (3) and (9). It does not implement their optimized Wannier gauge.

  Preparation still performs the complete D9 real canonical audit,
  including virtual orbitals and all existing gap, rank, orthogonality,
  stationarity and reality guards. Both compact projectors are compared
  against that audited canonical split in streamed AO row blocks.
  The maximum entrywise projection error, occupied orthogonality defect
  and imaginary residual are retained and must also meet the consumer's
  tolerance. Compact data bind the mesh, electron count, frozen count
  and exact overlap/Fock identities. The compact reference exposes no
  `mo_coeffs` fallback. D11 below removes the temporary canonical array
  used by the initial D10 implementation.

  Selection applies each projector to the same normalized AO seed
  families. The seed Gram is the corresponding submatrix of
  $Q_X^T S Q_X$, computed from cyclic symbols without assuming exact
  numerical idempotence. Greedy scores, tie order, linear dependence
  threshold and symmetric Lowdin construction remain unchanged. The
  resulting real frame passes the existing metric, projector, partition
  and translation-gauge checks. Cyclic covariance is constructive, so
  compact projectors require no all-origin covariance scan. Provenance
  and accepted-reference seals hash their home blocks and audit metadata;
  they do not expand AO projectors or restore canonical coefficients.
  Reuse shares only immutable projectors and operators while detaching
  mutable reference state, energies and provenance. Block handles require
  the same explicit occupied representation and frozen partition.

  Retained projector storage is two N-by-b-by-b real arrays rather than
  an M-by-M real canonical array; the no-frozen case currently retains a
  zero frozen array. Full one-dimensional energies remain available.
  This is a retained-storage reduction, not a peak preparation-memory
  bound: selected occupied frames, PAO/pair coefficients, home factors and
  solver payloads remain. D11 addresses the temporary canonical audit.
  D9's literature distinctions,
  experimental scope, independent-spin boundaries and unchanged downstream
  metrics/residuals still apply. Existing defaults remain unchanged.

  **D11. Replayed complete-spectrum admission.**
  Compact preparation now uses a replayable eigenspace realifier shared
  with the default dense adapter. It stores sorted energies, indices and
  eigenspace boundaries, and produces one complete real block at a time.
  Its real-span Gram rank test, metric orthogonalization, projected Fock
  diagonalization and eigenvalue tolerance remain unchanged. All occupied
  and virtual blocks are visited. Degeneracy and the global frozen/active
  and occupied/virtual gap rules are unchanged.

  The audit replays blocks to evaluate every entry of $C^T S C-I$,
  including entries between distinct eigenspaces, and every AO entry of
  $FC-SC\epsilon$. It retains the original maximum-entry residuals and
  `1e-8` admission threshold. The occupied-only Gram defect is recorded
  separately for the subsequent consumer tolerance. Neither per-k checks
  nor a unitary-invariant norm replaces these global checks.

  Frozen and active projector comparisons accumulate
  $\sum_X C_X[\mathrm{rows}]C_X^T S$ one AO row tile at a time, where X
  runs over the complete eigenspaces in that partition. Every AO row is
  compared with its compact projector at the original maximum-entry
  tolerance. Agreement of the occupied union, or of the home row alone,
  cannot certify a frozen split. This retains D10's numerical comparison;
  it adds no all-origin covariance scan or localization approximation.

  The assembled M-by-M canonical array and the full occupied audit frame
  are absent. Workspace contains a fixed number of M-by-d eigenspace
  arrays, real-span Gram/eigensolver work of order d squared, and b-by-M
  row tiles, where d is the largest complete degeneracy and M = Nb.
  Broad degeneracy can still require quadratic workspace. Replay trades
  work for storage: B spectral blocks require B squared block
  reconstructions for the global audit, plus N times the occupied block
  count for the projector comparison. D11 originally repeated both
  eigendecompositions on each reconstruction; D12 below caches bounded
  reconstruction recipes. `orbital_audit="replayed-complete-eigenspaces"`
  and `orbital_audit_block_evaluations` expose every reconstruction; the
  existing column-width diagnostics remain available. No time or
  total-memory scaling claim follows from eliminating the assembled array.

  D10's cited Bloch/projector formulas and scientific scope still apply.
  Bounded regressions compare the replayed audit with the assembled
  reference, including complex phases, large degenerate blocks, virtual
  residuals, cross-eigenspace overlaps and frozen-space mixing. Default
  dense, molecular and independent-spin consumers keep their assembled
  outputs. Downstream occupied and pair storage remains a separate gap.

  **D12. Bounded eigenspace reconstruction recipes.**
  Compact preparation now retains small transforms for repeated visits to
  the same complete eigenspace. For a d-column complex block, let R contain
  its real and imaginary columns. The recipe consists of the retained
  Gram eigenvectors V, square roots w of their eigenvalues, projected Fock
  rotation U and resulting energies. Each visit reconstructs
  $C=(RV/w)U$ in the original multiplication and division order. No AO
  coefficient block is cached, combined into a full canonical array or
  retained on the accepted reference.

  The four owned, immutable recipe arrays require $8(3d^2+2d)$ payload
  bytes. Their total payload is bounded by one primitive Bloch coefficient
  tensor, $16Nb^2$ bytes for N cells and b primitive AOs. Admission is first
  fit without eviction; a complete recipe that exceeds the remaining
  budget is recomputed on every visit. The budget covers retained numeric
  payload, not Python metadata or transient eigensolver workspace. Large
  degeneracies are never split or silently admitted to exceed this budget.

  Recipes belong only to one preparation call over its private Bloch
  coefficient snapshot and immutable cyclic operators. They are cached
  only after the unchanged real-span rank and Fock eigenvalue checks pass.
  Replays still reconstruct and check finite input columns, and every
  global overlap, AO stationarity and separate frozen/active projector
  comparison from D11 still runs. Returned energies are detached from the
  recipe. Default dense, molecular and independent-spin consumers do not
  enable this cache. The Bloch-transform `orbital_audit_recipe_cache`
  record reports the policy, byte budget, stored payload, entry count,
  hits and fresh realifications; the existing block-evaluation count still
  includes both fresh and cached reconstructions.

  This reschedules the same D9/D10 literature-based transformations; it
  introduces no new scientific approximation or orbital gauge. Bounded
  checks compare against the assembled reference, forbid dense fallback,
  count fresh decompositions, exercise insufficient cache budgets and
  preserve failure and caller-state guards. Global block reconstruction
  and contraction work, uncached wide eigenspaces, and downstream dense
  occupied/pair storage remain concrete optimization gaps. D14-D15 below
  address retained pair storage and compact chi selected-frame preparation.
  **D13. Retained-Bloch Coulomb contractions for chi local MP2.**
  The D8 adapter now selects `TranslationBlochFactors` by default. It borrows
  the sealed executed canonical AO factors, including null modes, metric
  ranks, physical supports and source identity. The shared pair builder calls
  its bounded `pair_exchange`; PAO/PNO construction and the iterative solver
  are unchanged. Explicit `factor_provider="dense"` retains the former
  home-auxiliary oracle, with no automatic fallback.

  With the finite unitary convention of Nejad et al. 2025, Sec. II,
  Eqs. (2)-(10), the occupied/virtual contraction for momentum transfer q is
  $X_q=N^{-1}\sum_{j-i=q}c_i^{\dagger}L_{ij}c_j$. The auxiliary-cell
  transform is $B_t=\sum_q e^{iqt}X_q$. For real pair coefficients,
  character orthogonality yields $(ia|jb)=N\sum_{qP}X_{q,Pia}X^*_{q,Pjb}$.
  The shared energy normalization remains separate. The
  [chi guide](chi_local_mp2.md) derives the phases, conjugations and factors
  of N, and specifies the complete source-recovery and covariance audits.
  Every AO/auxiliary entry is checked at the existing `1e-10` tolerance in
  primitive-AO-pair/auxiliary tiles; no full folded tensor enters preparation,
  provenance, validation or the pair contractions.

  A separate `max_factor_bytes` control covers serial provider numerical
  payload, including temporary transforms, dtype conversions, returned
  buffers and a bounded phase cache. The implementation does not call a
  native factor binding. The cap excludes the retained source, reference
  admission, caller inputs, previously returned buffers and downstream work;
  it is not a total-memory bound. The minimum audit tile remains quadratic
  in character count. [Bounded measurements](../benchmarks/chi_streamed_factors.md)
  record both reduced allocation and the slower complete Python audit.

  Riplinger and Neese 2013, Sec. II B3, Eqs. (23)-(25), and II C,
  Eq. (30), and Nejad Sec. V, Eqs. (42)-(44), still define the shared PNO
  and residual conventions. This increment adds no local fit, integral
  screening, optimized Wannier procedure or domain correction. D9-D12's
  orbital audits and reconstruction recipes are not duplicated. Scalar ECP
  metadata and separate frozen projectors retain D8's admission scope.
  Source tests compare both providers with the independent same-source
  canonical limit and compare raw energies, residuals and separate PNO
  estimates. Scientific vq acceptance remains separately recorded; chi is
  experimental and D93 fitted/post-HF is not-qualified. D14 addresses retained
  selected occupied and pair-AO storage; the audit's minimum character tile
  remains an implementation gap.

  **D14. Shared selected columns and recipe-only pair AO storage.**
  Chi local MP2 defaults to `compact_storage=True`; the generic shared
  driver/builder opt in explicitly. The phase-free gauge gives
  $C_{R\mu,Ti}=C_{(R-T)\mu,0i}$ with finite-torus modulo arithmetic.
  `TranslationOccupiedColumns` retains immutable active and frozen home
  columns separately. Every reconstructed column is compared with the
  original selection at the existing maximum-entry tolerance. Complete
  source-projector admission then consumes that reconstruction, including
  the frozen partition; covariance alone does not establish same-source
  identity. Selection, source inputs and pair spaces share those buffers.

  The existing `ProjectedPairBasis` supplies $V_p=E_{D_p}A_p-C B_p$.
  `ProjectedPairCoefficients` reconstructs individual PAO/PNO bases by
  summing the translated occupied home families, retaining all tails.
  Every PAO and rotated PNO is compared entrywise with the dense algebra.
  Full orthonormality, occupied exclusion, self-inverse stabilizer and
  factor covariance checks keep their original tolerances. No all-pair
  AO list is retained or passed to pybind. The native prepared transport
  accepts empty AO coefficients only alongside complete validated recipes.
  Its existing stable-link evaluation reconstructs one source pair and
  preflights an additional largest AO rectangle within its transport cap.
  Complete recipe/source hashes seal reuse without AO reconstruction.

  Nejad Sec. II after Eq. (13) states the translated-column identity;
  Sec. IV A, Eqs. (26)-(28), gives BvK PAOs. Riplinger and Neese Sec. II B1
  supplies occupied projection and II C Eq. (30) supplies cross-pair overlap.
  This storage change reuses those equations and D4's stable evaluation;
  it adds no localization optimization, cutoff, refit or new solver.
  Dense coefficient storage remains an explicit independent oracle.
  Source checks cover complex Bloch phases, nonorthogonal bases, odd/even
  meshes, frozen partitions, ill-conditioned domains, budget boundaries,
  raw iteration/residual agreement and the independent canonical limit.
  [Bounded measurements](../benchmarks/chi_pair_storage.md) report retained
  bytes and timing without claiming linear scaling or total-memory bounds.
  Full selector Gram matrices, transient full selected frames, SC/FC,
  projected domain matrices and the prepared link graph remain separate
  gaps, with the selected-frame preparation addressed by D15 below.
  Scientific vq acceptance remains pending and separately recorded;
  chi stays experimental and D93 fitted/post-HF stays not-qualified.

  **D15. Cyclic occupied selection and streamed preparation.**
  Compact chi preparation decomposes the translated normalized-AO seed Gram
  into character blocks. Its complete spectrum is the union of those blocks;
  greedy scores, global relative conditioning, candidate/tie order and the
  symmetric Lowdin gauge are unchanged. Home columns use the inverse transform
  of $Q(k)E_JD^{-1}G(k)^{-1/2}$ with normalization $1/N$. Complete
  degeneracies remain in the matrix function; frozen and active partitions
  are independent. The [chi guide](chi_local_mp2.md) derives the spectrum,
  normalization, inverse root and real reconstruction, including self-inverse
  characters. The dense selector remains an explicit small-system oracle.

  The chi adapter evaluates its original real-span Gram spectrum with
  conjugate-character blocks, avoiding the full folded occupied span. It
  compares all entries of the normalized span projector with the original
  complex source projector. Canonical source metric/stationarity admission
  and the other adapters' D9-D12 replay remain unchanged. Selection, domain
  preparation, PAO construction and source provenance consume streamed columns
  or home buffers. All global selected Gram, separate source-projector,
  frozen-cross and generator checks remain exhaustive at their original
  tolerances. Full selected C, SC and FC are absent, including the native
  transport boundary. Stable transport reconstructs one occupied column and
  one source pair at a time and charges them to its existing cap.

  `max_selection_bytes` preflights controlled numerical array payload for the
  compact occupied-source/selector phase, with explicit charges for eigensolver
  inputs/outputs, transform copies, home snapshots, reconstruction and audit
  buffers. Native library internal scratch, caller arrays, canonical admission
  and downstream pair work are excluded; this is not a process-memory cap.
  Infeasible budgets refuse with no dense fallback. The
  [bounded ladder](../benchmarks/chi_occupied_preparation.md) reports retained
  and traced preparation bytes and time. Occupied Fock, projected domain
  matrices, individual pair bases, amplitudes and retained transport links
  remain dense; all-entry replay can cost more time as the mesh grows.

  Nejad Sec. II Eqs. (2)-(10) supplies the unitary convention and translation
  identity; Riplinger and Neese Sec. II B1 supplies the PAO projection. The
  derivation reschedules those operations without optimizing a new Wannier
  gauge, screening tails or changing the PNO/residual conventions. Source
  checks retain independent same-source canonical MP2, raw residuals and
  separately reported corrections. Scientific acceptance remains pending;
  chi is experimental and D93 fitted/post-HF is not-qualified. The next
  bounded gap was exhaustive occupied-audit replay cost, addressed by D16;
  the minimum quadratic
  factor-audit tile and retained transport graph remain separate work.

  **D16. Complete occupied audits through verified cyclic blocks.**
  Immutable selected home columns and cyclic overlap now give the complete
  Gram symbols $c_X(k)^\dagger S(k)c_Y(k)$ and separate projector symbols
  $c_X(k)c_X(k)^\dagger S(k)$. Every displacement block is inverse transformed
  before evaluating the original maximum real-entry norm, with the identity
  subtracted only from the zero-displacement same-partition Gram. The active
  and frozen projector targets remain independently source-bound. Admitted
  overlap asymmetry is preserved; no metric adjoint or idempotence shortcut
  changes a contraction.

  The chi source realifier checks its actual complete character table and
  ordering. Its real conjugate-pair Gram uses separate real/imaginary entry
  maxima, not the complex symbol magnitude. Both original and real-operator
  metric actions, complete self-inverse blocks and all complex source-projector
  displacement entries remain checked. Translation is the verified permutation
  identity over these complete blocks. Unverified tables, dense coefficients
  and custom representations retain streamed replay. Near a floating-point
  admission boundary, the original arithmetic determines acceptance at the
  unchanged tolerance. The [manual](chi_local_mp2.md) derives both original
  representations and their norms.

  A single preparation context reuses one overlap and independent active/frozen
  symbol tensors plus at most sixteen scalar measurements. Exact source,
  mesh, operator, partition and column-content identities bind reuse. The
  selection ledger charges all intermediates, phase snapshots, cache payload
  and outputs before allocation. The cache expires on return or exception;
  no full selected C/SC/FC, translated seed Gram or dense AO projector returns.
  The native transport boundary retains its independent streamed checks.
  Provenance reports measured defects, formulations, displacement work,
  cache limits/hits and threshold-boundary replay. Equivalent checks are not
  described as skipped or as new scientific acceptance.

  The [bounded benchmark](../benchmarks/chi_occupied_preparation.md) separates
  source realification, selected-source admission, selection and complete
  occupied preparation. It records time and storage regressions as well as
  improvements, with operation guards instead of timing assertions. The
  independent canonical MP2 comparison, raw residuals, domains, PNO spaces
  and separately reported corrections remain in the focused tests. Nejad
  Sec. II Eqs. (3), (9), (10) and the translation identity after (13), and
  Riplinger/Neese Sec. II B1, supply the same equations as D15. No optimized
  Wannier gauge, screening or new scientific scope is introduced. Chi remains
  experimental and D93 fitted/post-HF remains not-qualified; scientific vq
  acceptance remains separate. Repeated greedy candidate-Gram construction
  is the next occupied-preparation cost; one-pair reconstruction, the factor
  audit's minimum character tile and the retained transport graph remain.

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
