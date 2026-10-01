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
