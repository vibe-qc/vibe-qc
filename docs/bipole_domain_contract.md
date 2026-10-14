# BIPOLE physical-domain contract and acceptance

This design covers ab initio RHF, UHF, RKS and UKS BIPOLE. It reuses the
physical-pair kernels and the resolved-domain record; it does not replace the
integral implementation. Semiempirical methods, SECCM, custom-basis projects,
periodic correlation and the GDF Coulomb discrepancy in #163 are out of scope.

## Issue, code and acceptance matrix

Historical issue numbers refer to the archived monorepo. Their mapping is in
[the renumbering table](issue_renumbering_2026_09.md). Current numbers below
refer to the product tracker. Verification of a subset is not whole-issue
acceptance, and a source test cannot establish numerical correctness.

| Issue (historical) | Implementation and preserved behavior | Required acceptance and disposition |
|---|---|---|
| #316 | `lattice_pair_cells.hpp`, `lattice_integrals.cpp`, `periodic_fock.cpp`: physical AO pairs and quartets, selected by default in the four ab initio BIPOLE drivers. `bipole_domain.py`: immutable settings, shared Fock checks and output disclosure. | Default migration implemented; awaiting independent numerical acceptance. LiH/def2-SVP, KI/STO-3G, MgO/STO-3G must satisfy Gamma overlap relabelling <=1e-12 at the selected default; non-Gamma matrices must satisfy raw Bloch covariance. Fixed-density kinetic evidence is not a converged total-energy pair. |
| #44 (724) | Physical 1e, midpoint nuclear sources and erfc/reciprocal partners; J integral products and K AC/BD products with wider exchange output/density support. | Opt-in nuclear/quartet subset independently verified (note 32594). BIPOLE default and shared consumer migration implemented; broader native-family migration remains blocked: native Ewald, ECP centers/derivatives, COSX and legacy/symmetry combinations require supported implementation or explicit refusal. A boolean default change alone is insufficient. |
| #187 (20) | `grid.cpp`: point-centered periodic Becke partition. `bipole_gradient.py`: recorded grid/profile/radius forwarded to both KS response helpers. | Partial XC/gradient verification (note 32827); all twelve original PBE crystals at production defaults and a >=4-cell grid ladder remain awaiting independent verification. Preserve the LiH fixed-density +1.5670045538 mHa solely-XC causal witness. |
| #186 (21) | `periodic_fock.cpp`: pair traversal and charge bounds. `pbc_bipole_fock.py` and drivers: incremental builds, terminal exact-build/XC reuse. | Partial cost improvements verified (note 32470). Final-domain full-node LiH/MgO/NaCl HF and PW1PW profiles remain outstanding. Include setup, 1e/Ewald, SR J/K, reciprocal J/K, XC, exact confirmation and output. The historical 7-17% hybrid erfc share does not bound the remainder. |
| #182 (66) | Fold guard, fixture re-cuts and capability refusals in the four linked test files. | Still blocked on complete supported slow-node evidence and baseline reconciliation. Note 34927 fails closure criteria, with no numerical counterexample; note 32833 is an inconclusive timeout. Keep the cutoff-6 negative guard, explicit expensive MgO skip and known-red row until applicable evidence exists. |
| #200 (BIPOLE only) | `test_bipole_fock_ewald.py`, test-gate manifests. | Remeasure then retarget, as already decided in note 30867. A >2400-second bounded file budget and actual wall/RSS evidence are necessary. Do not reinterpret timeout, skip or unmeasured as pass. |
| #166 (116) | `bipole_terminal_check`, `IncrementalJK.seed`, all four drivers: exact committed-density confirmation and reseeding before convergence. | Already fixed and independently closed (notes 33115/33119). Preserve exact/incremental agreement, genuine refusal and emitted residuals after migration. The verifier's small smeared entropy mismatch is separate from incremental-chain drift. |
| #220 | `guess.py`: all-space SAP quadrature for localized AO products, separate from per-cell XC. | Already fixed and independently closed (notes 32592/32593). Preserve raw overlap <=2e-6 and raw Gamma SAP asymmetry <=1e-3 controls. No converged-energy or GDF acceptance follows from a guess correction. |

## Evidence boundaries

The audit on #163 (note 28210, corrections 28421, 28545, 28826 and 28952)
is a set of hypotheses and measurements, not an instruction to reinterpret
all failures as one bug. Source inspection confirms the native default still
selects the historical index ball and the physical kernels already exist.
The claim that two policies prevent one shared resolver is too strong: one
resolver can make the policy explicit and share its result. It cannot make
historical support invariant.

#316 note 28820 establishes a -5.498756 mHa relabelling difference in the
fixed-density kinetic term for KI/STO-3G at 18 bohr. Its truncated nuclear
attraction comparison is not the production Ewald operator. Neither number
is the missing converged-total difference. Timings on a loaded host do not
establish a controlled speedup. Relabelling covariance at finite support and
convergence to the infinite periodic limit are distinct acceptance tests.

## Shared finite operator

* AO-product support uses physical center separation at `cutoff_bohr`.
  Short-range quartet support additionally uses product-midpoint separation,
  with its own resolved interaction reach. K retains its AC/BD products;
  imposing an external AB projector removes valid exchange terms.
* `resolve_bipole_ewald_state` resolves alpha, real reach, reciprocal reach
  and precision together. Nuclear erfc preparation uses the same state.
  Energy and derivative consumers must not silently choose another state.
* The lattice snapshot records overlap/exchange and energy/force Schwarz
  thresholds. The intentionally tighter force threshold is preserved, not
  replaced with the energy threshold.
* XC records all `GridOptions`, density-domain selection, periodic partition
  selection and image reach. SAP all-space quadrature is a different operator.
* The record identifies settings, not geometry, basis, compiled kernels or
  retained tensor membership. A force comparison must also check finite-domain
  boundary crossings. Derivative paths that cannot reproduce a selected
  operator must refuse explicitly.

## Fock operators, MANTA and symmetry integration

The unified operator proposal (#472) is a consumer boundary for a completed
Hermitian Fock or KS matrix. It does not replace the BIPOLE builder or admit
another integral source. Until that proposal lands, BIPOLE keeps its existing
solver entry points. Its exported states can already be checked against the
draft independently of a native SCF run; the reproducible saved-Si diagnostic
is in `studies/pbc-operator-compatibility/`.

The compatibility requirements for subsequent implementation are:

1. **Preserve the evaluated density epoch.** Each spin/k Fock snapshot must
   identify its AO frame, k point, spin and build generation. Keep the evaluated
   density, Fock, component energies and exact-confirmation result together.
   Newly diagonalized orbitals define a candidate next density; they do not
   retroactively replace the density used to build that Fock or energy.
   Incremental-J/K caches remain builder-owned and a confirmation build keeps
   its existing cold-build and reseeding semantics.
2. **Keep source identities separate from solver metadata.** The resolved
   BIPOLE supports, actual basis shells, Ewald envelope, exchange zero mode,
   screening, XC grid and runtime provenance accompany the method result.
   A matching matrix shape or solver frame label cannot establish equality
   with a MANTA source. Component comparisons require matching finite terms
   before comparing converged total energies. A hybrid retains its own exact
   exchange coefficients; a pure semilocal KS witness does not admit SKALA or
   another external XC functional.
3. **Retain memory admission.** MANTA's bounded solver may consume borrowed
   fixed matrices internally and export owned snapshots after the calculation.
   Do not insert the copying Python operator adapter into that loop without
   accounting for its live matrices and solver workspace. Integral source
   selection and eigensolver selection remain independent capabilities.
4. **Preserve full-mesh semantics.** Weights enter density folds, component
   energies and electron counts, never orbital normalization. A full uniform
   BvK source cannot consume irreducible representatives by changing only the
   weights. Use the shared k-star transport for an admitted subgroup, with
   explicit AO phases, reciprocal wraps and antiunitary operations. Export
   enough frame information to distinguish full and representative meshes.
5. **Qualify reduction on the actual source and state.** Physical BIPOLE
   exchange output extends beyond its one-electron support. Keep its existing
   refusal of the legacy projector. Validate unreconstructed J/K/XC covariance
   on nonsymmetric trial densities before reducing work; then check the
   converged state's subgroup, including unrestricted broken symmetry.
   Averaging F or labelling a solver block is not source qualification.
   Keep the unreduced calculation available as the reference.

See the [shared symmetry design](design_shared_symmetry.md) and
[MANTA scientific contract](design/manta_scientific_contract.md) for their
source-specific requirements. Integration acceptance proceeds separately:
exported snapshot/solver parity, matched finite-source component parity,
whole-mesh versus admitted reduction, fresh SCF with energy and potential
refinement, then matched end-to-end CRYSTAL23 timing. Passing an earlier stage
does not close a later stage or the complete HF/DFT/hybrid/SKALA matrix.

## Approved BIPOLE production migration

The maintainer-approved implementation is a BIPOLE-only migration. Keyword-only
`historical_domain=False` on the four ab initio drivers and
`bipole_historical_domain=False` on `run_periodic_job` select the policy once
on a private copy of lattice options before preparing any operator: False
selects physical pairs; True selects the historical index ball. The generic
native `LatticeSumOptions` default stays unchanged pending cross-family proof.
Independent numerical acceptance is still pending. The migration changes the
finite operator, not its convergence tolerances.

Concrete consequences:

1. Existing BIPOLE calls, including explicit `pair_complete_1e=False`, select
   physical support unless migrated to the named historical option. The manual
   documents this precedence; an old False is not a historical request.
2. The implementation forwards the selection through all runner dispatches, spin schedules,
   optimization and finite differences. An explicit historical request is rejected on
   inactive routes. Caller objects are preserved so one calculation cannot change
   the next calculation's policy or cutoff.
3. Physical AUTO uses the full direct build where the historical symmetry
   projector omits exchange output. Explicit incompatible projector, COSX,
   grid/gauge and ECP combinations retain a checked refusal until supported.
4. Equal nominal cutoffs can retain different terms and change numerical pins,
   fold drift and cost. Publish per-component parent/candidate differences;
   accept reference updates only after operator attribution and independent
   verification. No tolerance change, damping or arbitrary radius increase is
   part of this migration.
5. Production-default closure requires the invariance tests to use that
   default, without setting `pair_complete_1e=True` in the witness. Prepared
   regressions now use the default resolver and direct drivers. Earlier opt-in
   results cannot witness completion of this migration.

The native `LatticeSumOptions.pair_complete_1e` default remains False. That
field is no longer the public policy selector for these four BIPOLE drivers:
`historical_domain` takes precedence even when the caller supplied an explicit
False or True native flag. ROHF/ROKS and other method families are unchanged.
The requested lattice snapshot is retained alongside the resolved operator;
analytic previews accept either original or resolved options and reject other
settings. Omitted derivative controls restore the resolved snapshot.

The runner records the planned policy in dry-run output and the actual resolved
settings in the completed manifest. See the [manual](user_guide/bipole.md) and
[worked tutorial](tutorial/bipole_physical_domain.md) for migration and replay.

## Deferred numerical validation

Execute under a bounded validation plan. Use independent matching native builds,
sealed input/test hashes and identical parent/candidate controls. Record wall,
CPU and process-tree peak memory outside product checkouts. No automatic retry;
stop at the phase cap. A pytest per-test timeout alone is not a resource cap.

| Phase | Prepared selection | Wall / memory ceiling | Acceptance |
|---|---|---|---|
| Compact ionic/diffuse S/T | `test_periodic_1electron.py -k physical_domain_ionic_bloch_relabelling_316` | 900 s / 4 GiB | S <=1e-12, T and fixed-density kinetic trace <=1e-10, Gamma and non-Gamma; no projection |
| Finite J/K and nuclear terms | `test_pbc_pair_complete_consumers.py test_bipole_jk_domains.py test_nuclear_erfc_lattice.py -m 'not slow'` | 3600 s / 8 GiB | Existing independent Gaussian, density-adjoint, relabelling and FD thresholds |
| Returned density / exact builds | `test_bipole_final_density_phase.py test_bipole_fock_ewald_exchange.py -k 'final_density or terminal_check or seed_resyncs' -m 'not slow'` | 3600 s / 8 GiB | Raw component closure <=1e-12; original energy/commutator criteria |
| Nondefault grid / Ewald replay | `test_bipole_gradient.py -k 'ks_gradient_nondefault_xc_radius_matches_fd or nondefault_ewald_precision_force_matches_displaced_scf'` | 3600 s / 8 GiB | Original force thresholds; explicit/omitted grid controls agree <=1e-12; physical and historical controls |
| Broader forces | `test_bipole_gradient.py -m slow` | 14400 s / 8 GiB | Existing derivative thresholds, same resolved controls; skipped MgO is still a gap |
| Slow Ewald baseline | `test_bipole_fock_ewald.py -m slow -k 'not multipole_far_field_smoke'` | 7200 s / 8 GiB | Classify every supported node; no baseline retirement from partial output |
| KS/basin controls | `test_bipole_ks_convergence.py test_pbc_bipole_diis_converged_basin.py` | 7200 s / 8 GiB | Existing convergence, refusal and basin assertions |
| SAP preservation | `test_periodic_sap.py` | 1800 s / 4 GiB | Raw pre-projection controls; no per-cell substitution |

Run one process, OMP=2 and BLAS=1 for these bounded correctness phases. The
separate twelve-crystal parity matrix, six full-node performance cases and
converged KI historical/physical pair need archived inputs and explicit resource
admission before a runnable payload can be sealed. Do not substitute guessed
geometries, a relaxed 1 mHa bar, or a small box for those acceptance rows.

## Independent XC density and partition radii

BIPOLE RKS and UKS resolve `options.lattice_opts.becke_image_radius_bohr`
as the density-image reach: a nonpositive (unset) value uses `cutoff_bohr`,
and a positive value uses the smaller of the request and `cutoff_bohr`.
Nonfinite values and nonpositive lattice cutoffs are rejected.
`options.becke_image_radius_bohr` independently controls the Becke partition,
retaining its 10-bohr default. Increasing the partition radius alone does not
extend the density support. Caller options are preserved; `resolved_domain`
retains the requested lattice controls, effective density reach, and partition.

AUTO still detects periodic versus molecular-limit density in the native XC
kernel. The driver reserves the image differences needed if AUTO selects
periodic density; it does not force a molecular-limit density to become periodic.
Explicit density-domain restrictions, including analytic-gradient refusals, remain.
Supported gradients recover the effective density reach from the SCF snapshot
and restore the partition separately. Historical results without a snapshot
retain the old partition-based restoration behavior.

`xc_grid_electrons` reports the unmodified integral from the terminal accepted
XC density, alongside `e_xc`, for built-in and external providers and both spins.
`xc_periodic_density_bras` records the kernel's actual density selection.
For periodic density, `xc_grid_electron_error` is `(N_grid - N) / N`, with UKS
using the total alpha-plus-beta count. A magnitude above 1e-3, or a nonfinite
error, emits one `PeriodicXCGridChargeWarning`, including on an iteration-capped
return. Molecular-limit densities have no per-cell relative error. The check
performs no extra XC evaluation and changes no density or occupations.

This repairs the accidental coupling and missing diagnostics (Refs #434).
A quiet diagnostic is not an energy or force accuracy guarantee. Independent
converged-SCF, matched-reference, provider, spin and force acceptance remains
required before treating the repaired route as numerically validated.
