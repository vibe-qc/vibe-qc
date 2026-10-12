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

**Historical periodic routes, none production.** Three drive this engine
on a real Gamma supercell through private density-fitting stand-ins. The
CCM coupled route now uses the correlation-source contract and the same
compiled engine; its union-PNO calculation remains a small-cell oracle.
The native chain is separate.

| Route | What it runs | Reference and factors | Engine reuse | Label |
| --- | --- | --- | --- | --- |
| Megacell, `periodic_megacell_mp2.py` | DLPNO-MP2, CCSD, CCSD(T) on an open finite supercell | molecular RHF and molecular DF | complete, through `run_job` | shipped, no warning, no catalog row |
| Toroidal, `periodic_toroidal_mp2.py` | the same on the Gamma BvK supercell | Gamma GDF HF and the Stage 4 compensated-cell factors through `_PeriodicDFAdapter` | `run_dlpno_mp2`, `run_local_dlpno_ccsd` in `residual_domain="pair"` with a swapped overlap | Stage 5a to 5c, no warning |
| CCM neutral torus, `periodic/ccm/dlpno*.py` | union-PNO CCSD(T), coupled CCSD(T), UMP2, UCCSD(T) | neutral GDF factors, Bloch results folded to the real supercell | coupled CCSD(T) through `ccm_neutral_source` and the shared engine; union-PNO and spin-orbital oracles retained | `AICCM2026DevAExperimentalWarning` |
| chi, `periodic/chi/posthf.py` | DLPNO-MP2, CCSD(T), UCCSD(T) pilot | k-mesh RSGDF or GDF fit turned into an exact real torus through `_BRealDFAdapter` | `run_dlpno_mp2`, `run_local_dlpno_ccsd` in pair mode; screens fail closed | `AICCM2026DevBExperimentalWarning`, library only |
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
own coupled-cluster loop. The CCM transcription and the chi and toroidal
stand-ins are replaced by adapters to the contract below; the native chain
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
  cost against cell size. The toroidal and chi stand-ins remain to be
  converted to the contract.

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
  ROHF/QRO spatial orbitals and NEV-PNOs. Our engine retains independent
  UHF alpha/beta spaces and two-sided pair densities. Ye and Berkelbach
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
  BvK milestone; it starts only when B has its evidence.
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
