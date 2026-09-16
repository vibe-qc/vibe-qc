---
orphan: true
---

# BIPOLE erfc continuation after the repository split

The physical quartet/nuclear-domain integration is recovered and remains
opt-in. Terminal XC reuse, Schwarz self-norm screening and screened derivative
buffer handling are corrected. LiH and Si native SCFs and fixed-density
grid/cutoff checks are complete. The 840-case run completed with 836 passes,
two diagnosed grid-fixture failures, one known expected failure and one
historical skip. Both reviewed fixtures then passed their reruns, and the
historically skipped MgO force comparison passed separately. Production code
was unchanged between these runs. The global default and issue closure
remain subject to the acceptance limits below.

## Recovery, 2026-09-08

The continuation uses the standalone `vibe-qc/vibe-qc` repository, branch
`codex/bipole-erfc-resume`, starting from `bdec7d2`. The initial pull was
followed by the development install's main synchronization. The preserved
candidate was based on monorepo commit `5d49d0c28573`.

The 50 surviving source, documentation and test files were restored from
`codex-bipole-full-wip-paused-20260908.patch` under
`~/vqruns/bipole-erfc-20260906/physical-coupled-wip/`. The old handover
directory was removed in the split; the archived patch retains its history.
The suite manifest was regenerated from this repository's current inventory
instead of restoring the monorepo's generated manifest.

Historical issue numbers in this note refer to the monorepo. The outstanding
work was tracked as #21, #66, #674, #724 and #20. Their new repository IDs
are #186, #182, #82, #44 and #187, respectively; see
[the issue mapping](issue_renumbering_2026_09.md).
At the user's handoff, cluster deployment and bugctl SHA validation still
targeted the monorepo. Their migration was not exercised by this local work;
these results do not establish deployment or independent issue closure.

## Operator and acceptance boundaries

The recovered physical-domain implementation is opt-in through
`pair_complete_1e`. It coordinates AO-pair separation, quartet midpoint
interaction reach, wider exchange output support, nuclear-source images,
reciprocal support and their derivatives. Independent Gaussian sums,
image relabelling, density adjoints and finite differences exercise the
retained finite operator. Convergence of that finite operator toward the
periodic limit is a separate requirement.

For a retained integral `(ab|cd)`, both physical product separations must be
within the pair cutoff, and the separation of `(A+B)/2` and `(C+D)/2` must
be within the interaction cutoff. Coulomb and exchange apply this predicate
to their respective integral pairings. In exchange, the external matrix
indices need not themselves form a retained product pair. Restricting that
external pair to the overlap support would truncate a different operator.
The union of pair-centered spheres with radius equal to pair cutoff plus
interaction cutoff encloses the required images without growing merely
because an atom is given a distant lattice-image label.

Nuclear source selection uses a sphere about each AO-product midpoint.
Reciprocal nuclear and Hartree terms use the same AO-pair support as their
real-space partners. Derivatives preserve that support away from sharp
cutoff crossings. Tests at a finite cutoff therefore establish the
derivative of that retained sum, not uniform convergence across a cutoff
boundary.

The global default remains off. Periodic COSX, diagnostic grid Hartree and
explicit legacy Fock projection refuse the unsupported physical-domain
combination. The #66 fixture retirement follows the prior maintainer decision
to withdraw asymmetric legacy-HF converged-SCF claims; it does not loosen
terminal stationarity tolerances.

The old native LDA finite-domain energy pin was reviewed separately from
the independent periodic two-Gaussian LDA reference. Restoring only the
historical grid partition for that exact fixture reproduces the old
historical-domain and physical-domain values to 1.3e-13 and 1.4e-10 Ha.
This establishes why the native regression anchors must follow the grid
correction; it does not replace any independent physical reference.

On this build the historical-domain LDA result is -34.66112304044817 Ha
(+0.25327789 mHa from its old pin); physical-domain LDA is
-33.875342453487875 Ha (+0.25775326 mHa). These are measured discrepancies,
now reviewed native regression anchors. Both independent HF split values and all four
RKS/UKS periodic LDA reference witnesses passed in the domain run.

## Validation record

The initially recovered extension's SHA256 is
`fcc50e6eacf0d4034865095021bdfa47b298021e9a135eec2db1d6b767efd991`.
All 568 recorded native build inputs were unchanged when validation started.
The package and extension both resolve inside this checkout.

Two binding checks passed. Four new RKS/UKS terminal-XC tests failed before
the reuse change, each at the redundant call after the exact final-density
phase. The complete terminal/helper/output selection then passed all 181
tests after the change, including converged and iteration-capped returns.
The tests allow XC during a required final build and reject only subsequent
quadrature. The returned XC energy must equal the exact build's stored value.
This three-file change is saved separately as local commit `3792411`.
The recovered physical-domain candidate was validated before committing the integration.

The full 424-case domain/consumer run completed in 906.63 seconds with
422 passes and two failures. Besides the preserved LDA pin, a runner test
invoked a queue test file removed in the split. Replacing that external
dependency with a synthetic selected pytest node preserves subprocess
isolation and caller-state checks; all 48 gate tests pass after that repair.
That repair is saved independently as local commit `cdc6790`.
The original 424-case result remains recorded as a failed run.

Before correcting Schwarz self norms, at the saved LiH/PBE reference density,
physical domains with a 20-bohr pair
cutoff, 35.47-bohr resolved SR interaction reach and `sr_image_precision=1e-8`
give -8.103457093990926 Ha. Relative to the level-9 independent reference,
the differences are +23.54956 microhartree total, -10.46735 microhartree
Coulomb and +34.01692 microhartree XC. Kinetic and nuclear-attraction
differences are at most 2.2e-11 Ha. This one-density evaluation took
515.92 seconds with one native worker; it is neither a converged SCF result
nor a matched performance benchmark. Grid and screening ladders were
then needed to explain the residual.

Tightening the native Schwarz threshold from 1e-12 to 1e-14 changes this
Coulomb component by only -3.41e-10 Ha. An independent PySCF 2.14.0
fixed-density RSJK calculation reproduces the archived Coulomb reference
at omega 0.4 and 0.7 within 5e-12 Ha. The native values at those split
parameters differ by 7.18512 microhartree, which motivated the
operator-domain/split investigation below. Independent long-range-only AFT
calculations localize the discrepancy to the short-range erfc sum: after
removing the common analytic background, long-range errors are -4.2e-14
and -9.65e-11 Ha at omega 0.4 and 0.7. The corresponding short-range
errors are -10.46769 and -3.28248 microhartree.
With charge-pair screening enabled at 1e-14, extending the physical
interaction radius from 35.47 to 45 bohr leaves the Coulomb result
bit-for-bit unchanged. The explicit-radius path does not automatically
enable charge-pair screening. The home-cell J energy contraction agrees between charge
screening on and off to about 1.4e-12 Ha. The full unattenuated-Schwarz
45-bohr run was stopped after these more targeted checks; its last row is
incomplete and is not a convergence result.

### Schwarz squared-norm correction

The residual comes from primitive screening in the Schwarz pre-pass itself.
Libint's default absolute precision is machine epsilon. A self integral
below that precision can be discarded even though its square root permits
mixed integrals many orders larger than the requested production cutoff.
Reducing the caller's Schwarz threshold cannot recover a bound already
stored as zero.

Local commit `da0534a` disables primitive screening only while computing
the molecular and periodic Schwarz self norms. The production ERI
threshold is unchanged. Fourteen closed-form Gaussian regressions fail
before this change and pass afterward, covering direct, molecular-limit
and lattice J/K builders, bare Coulomb, two erfc kernels and two thresholds.
Seven molecular incremental-Fock regressions also pass. The original three
periodic screening checks pass, including their performance check.

At the same LiH reference density, the corrected Coulomb energies are
1.995852262476314 and 1.995852262374664 Ha at omega 0.4 and 0.7.
They agree with the independent values to 1.7e-12 and 1.1e-10 Ha,
respectively. This resolves the previously observed 10.47-microhartree
Coulomb residual. The isolated first-build timings are 152.13 and 93.63
seconds versus 124.63 and 77.61 seconds before the fix; these observations
are not a controlled performance benchmark.

The intermediate Schwarz-corrected extension's SHA256 is
`de0e5158dfdf5e6eefc3a909c8d38315c5fa6a13cc1243fa9005ddbebe777a81`.
Only `cpp/src/schwarz.cpp` differs among the 568 native source inputs.
The older gradient run was stopped as superseded and remains incomplete.
The 816-test domain, gradient, convergence, terminal, output and screening
selection exposed the derivative-buffer defect below and was superseded.
Its completed shard and partial receipts are retained as historical results.

The native 15-bohr HF regression anchors move from -23.19211691121152
to -23.19210137102612 Ha for physical domains and from -22.93998258398289
to -22.939983004503866 Ha for historical domains. The original binary
reproduces both old anchors within 8e-14 Ha; the native source audit
isolates the change to Schwarz self norms. The image-relabeling invariant
passes. The anchors have been updated without changing tolerances or
independent physical references.

### Screened derivative-buffer correction

Libint marks a fully screened two-electron quartet by clearing only the
first result pointer. The periodic gradient contractions previously inspected
each of the twelve derivative pointers independently, allowing stale pointers
from an earlier quartet to contribute. Tightening force screening then made
the error worse by admitting more calls that Libint subsequently screened.

Local commit `74cdf04` checks the whole-quartet sentinel independently for
J and K before consuming derivative buffers. Eight of twelve new analytic
Gaussian force cases fail before the change; all twelve pass after it.
The MgO short-range J witness improves from a 1.34684e-6 discrepancy to
3.01438e-12 Ha/bohr against finite differences, with force balance restored.
The complete periodic screening file passes all 29 cases in the isolated
corrected build. Neither finite-difference step sizes nor tolerances were
loosened.

The final production extension SHA256 is
`853dca4cc0aa2d16238bdd70b7c2c35f86447cc8d90b67f1e4971376c703876b`.
Relative to the original native build inputs, only `cpp/src/schwarz.cpp`
and `cpp/src/periodic_gradient.cpp` changed. A fresh complete 840-case
selection includes the symmetry file and new derivative regressions.
Four local shards use 1/1/2/1 native threads and one BLAS thread each,
with exact collection inventory, source hashes and per-test receipts.
Under shared-host load, the original two-hour monitor was replaced with
another bounded two-hour monitor retaining the 16-GiB limit. The test
coordinator and workers continued without restarting. The deliberate
termination of the original monitor is recorded separately from pytest
outcomes in `verified/gate-guard-handoff.json`.
That attempt subsequently exceeded its aggregate memory cap: the monitor
observed 17,425,219,584 bytes against a 16-GiB limit and stopped it. Its
614 passing calls and one strict expected failure are incomplete receipts;
no test assertion failed. `memory-limited-gate-stop.json` records the stop.

The full 840-case selection is therefore restarted from scratch under
`verified-bounded/`, with at most two concurrent workers, four native
threads each, a 24-GiB aggregate limit and a four-hour wall limit. This run
records every setup/call/teardown phase and hashes the Python production
sources as well as the native inputs and selected tests. The source and
extension are unchanged; only execution concurrency and resource limits
have changed.
This run exposed two further grid-related test assertions. The legacy RKS
spheropole fixture's old native energy pin is -1.1156375078921 Ha; the
current partition gives -1.1156334486137565 Ha. Replacing only its grid
builder with the historical partition gives -1.115637508045234 Ha, within
1.6e-10 Ha of the old pin. The spheropole is identical in both calculations.
The recovered and final native binaries also give identical current-grid
energies. `legacy-rks-pin-review.json` retains this isolation.

The LiH moving-grid XC witness reached its old minimum-correction assertion
with a correction of 1.27478e-4 Ha/bohr, below the previous 5e-4 floor.
A separate diagnostic passes its unchanged 1e-9 finite-difference accuracy
requirement: the corrected error is 6.7e-13 Ha/bohr, while omitting grid
motion gives 1.27477e-4 Ha/bohr. The replacement non-vacuity condition
requires the omitted-correction error to exceed 1,000 times the accuracy
tolerance, rather than preserving the old partition's correction magnitude.
Its first attempt hit a 16-GiB memory cap at 17.634 GB; the unchanged
diagnostic completed under a 24-GiB cap with an 18.411-GB observed peak.
`grid-motion-witness-review.json` retains the force tensors and provenance.
The gate's source files stayed unchanged until the full run and separate MgO test completed. Only the two reviewed test functions changed for their reruns; the rest of both modules was compared structurally and is unchanged.
The selection retains two baseline markers: a strict expected
`NotImplementedError` for padded multi-k RHF CPHF negative curvature,
and a runtime-budget skip for the MgO corrected-Gamma RHF gradient.
The expected failure exercises historical padded domains with
`pair_complete_1e=False`; physical coupled SCF finite-difference cases are
maintained separately. Their decorators were compared structurally with
standalone base `bdec7d2`
and are unchanged. They must be reported separately from passing tests.
The MgO row was also run independently with eight native threads,
removing only its runtime skip while retaining its assertions and settings.
Its monitor was also extended under the same 16-GiB limit, retaining the
exact kernel-reported exit status. `verified/budgeted-guard-handoff.json`
distinguishes the monitor replacement from the pytest outcome.
At 03:16 UTC on September 9, the allowance was extended to six hours total,
ending at 05:55:54 UTC, because the fixture's recorded historical runtime
was 5.4 hours. The same pytest process continued under the same memory cap.
The replacement armed its kernel exit watch before stopping the old monitor;
that monitor's exit 137 is distinct from the eventual test exit. The first
extension receipt remains in `budgeted-guard-handoff-first-extension.json`.

The CRYSTAL CYC0 component fixture now injects its sealed local SAD density
explicitly. Production guesses normalize in the periodic overlap metric,
whereas the sealed diagnostic input used the molecular metric. Fixing the
diagnostic input preserves the independent CRYSTAL values and the checks
that distinguish CYC0 from the returned terminal density. The renamed
fixed-density component test passes; production guess normalization is
unchanged.

For XC convergence, an independent contraction of finite Bloch AO image
sums matches the native physical-domain energy to 2.9e-12 Ha. Both bra
and ket image domains must be specified: the default native bra reach is
10 bohr and the ket reach here is 20 bohr. The unrestricted periodic AO
sum is a different diagnostic. Native and PySCF XC functionals agree on
the same density and gradient. The energy-only contraction avoids forming
the XC potential matrices and is used for the subsequent quadrature ladder.

At partition radius 18 bohr and fixed 10-bohr bra support, increasing
quadrature from 75/17/36 to 150/25/50 and 200/35/72 gives XC energies
-2.262343324060281, -2.262383493413342 and -2.262382287809644 Ha.
On the finest grid, expanding bra support to 14 and 20 bohr gives
-2.262380998300450 and -2.262380998501048 Ha. The final value differs
from the independent level-9 XC reference by +0.87709 microhartree; the
last support increment changes it by only -2.01e-10 Ha. This completes
the energy-only XC ladder, not a native SCF convergence ladder.

The fresh Si/PBE reference is converged at grid levels 5, 7 and 9:
-578.6580706412801, -578.6580370246670 and -578.6580312081404 Ha.
The last grid increment is +5.81653 microhartree. All levels use the same
44-function basis, 28 electrons and zone-centered 2x2x2 mesh. Level 9
reuses the analytic one-electron matrices from levels 5/7 and rebuilds the
independent RSJK operator. This completes the previously interrupted
reference calculation; the native Si comparison is recorded below.

At the level-7 Si reference density, changing only the native grid partition
from the historical image weights to the corrected point weights shifts XC
by +20.52974479 mHa. Both evaluations use 75 radial, 17 polar and 36
azimuthal points per atom and a 10-bohr partition radius. The integrated
electron count changes from 28.08477868 to 27.99970363. Increasing the
corrected partition radius to 18 bohr changes XC by -48.71248 microhartree;
increasing quadrature there to 150/25/50 changes it by another -2.37605
microhartree. These are fixed-density component comparisons, not converged
native SCF results. An independent check of all 44 Si AOs at 800 sample
points agrees with the native basis to 7.2e-15 absolute error after the
documented angular-order permutation.

At the fresh level-9 Si density with matched 18-bohr partition and bra
support, quadratures 150/25/50 and 200/35/72 yield XC energies
-41.6639507896482 and -41.66395319182573 Ha. The last increment is
-2.40218 microhartree and the finest value is +5.68555 microhartree from
the independent reference. Its integrated electron count is 27.99999915.
The ket-image support remains 20 bohr. These fixed-density results retain
the independent-basis and native-functional cross-checks.

Increasing physical pair and nuclear cutoffs from 20 to 24 bohr at the
same LiH reference density changes the non-XC energy sum by
+1.47e-11 Ha. The automatically resolved SR interaction reach increases
from 35.47199 to 39.47199 bohr. At 24 bohr, all non-XC components agree
with the independent reference within 3.7e-12 Ha, and J changes from its
corrected 20-bohr value by +5.34e-12 Ha. The 20-bohr comparison combines
the recorded one-electron witness with the separately corrected J witness;
the archive identifies both inputs explicitly. This is a fixed-density
cutoff check, not another SCF ladder. Each driver normalizes the supplied
reference density in its finite overlap metric; the receipts record those
factors, which approach unity as support increases. For both LiH and Si, expanding only
ket-image support from 20 to 24 bohr on the default grid leaves XC energy
and integrated electron count bit-for-bit unchanged.

The corresponding Si 20-to-24-bohr check changes the non-XC energy sum
by -2.92e-10 Ha. Its largest individual component change is 2.45e-9 Ha;
all 24-bohr non-XC components agree with the independent reference within
2.13e-9 Ha. J is 174.42057194564217 Ha and the resolved SR interaction
reach is 38.98511 bohr. These runs used four native workers under concurrent
host load; their 587.75-second LiH and 1549.29-second Si timings are not
controlled performance measurements.

The pre-split 175 passes plus one failure and separate 17 passes were
interrupted sessions. They are historical partial receipts, not completed
gates for this checkout. The development environment is rebuilt locally;
new run inputs and logs are held under `/tmp/bipole-erfc-resume/` while the
continuation is active. Durable snapshots, including full patches and
checksummed inventories, are preserved separately under
`~/vqruns/bipole-erfc-20260908-standalone/`. The original handoff archive
is unchanged.

The native LiH/PBE SCF, warm-started from the independent level-9 density,
converged in four iterations and passed a cold exact-Fock confirmation.
Its total energy is -8.103446644267258 Ha, +33.99929 microhartree from
the independent result. The exact commutator norm is 1.356e-7 and the
confirmation changes the loop energy by -4.893e-11 Ha. This is consistent
with the +34.01692-microhartree default-grid XC discrepancy measured at
fixed density. The run used the default 75/17/36 quadrature and 10-bohr
partition radius, a 20-bohr pair/nuclear cutoff, and the resolved
35.47-bohr SR interaction reach. It took 2197.55 seconds with one native
worker; the reference density is a warm start, so this is not a cold-start
or performance benchmark.

LiH loaded the Schwarz-corrected extension `de0e5158...` before the
screened-derivative fix was installed atomically. The already-running
process retained that loaded image; its SCF did not call the changed
gradient routine. The raw result's hash was sampled from disk at return
and therefore names the newer file. `lih-native-scf-verified.json` corrects
the loaded-image field while preserving that raw hash and linking the
replacement receipt. The raw result is retained unchanged.

At the level-9 Si reference density, native kinetic, nuclear-attraction,
nuclear-repulsion and Coulomb components all agree with the independent
values within 4.0e-9 Ha. Native J is 174.42057194496707 Ha, a difference
of -2.02e-9 Ha. This check used the final native build and four workers;
it took 254.00 seconds.

The full native Si/PBE SCF also converged in four iterations and passed
its cold exact-Fock confirmation. Its energy is -578.6579424366462 Ha,
+88.77149 microhartree from the independent level-9 result. The exact
commutator norm is 3.682e-7; the confirmation changes the loop energy by
+2.956e-12 Ha. The final density evaluation reuses this confirmed build.
The run used the default grid, a 20-bohr pair/nuclear cutoff, a resolved
34.9851-bohr SR interaction reach, and eight native workers. Its loaded
extension hash was captured at startup and matches the final build. Wall
time was 783.20 seconds, with a peak observed RSS of 5.61 GB. As for LiH,
the independent density is a warm start and this is not a timing benchmark.
At the exact level-9 reference density, the default-grid Si XC value is
-41.66387009198551 Ha, +88.78539 microhartree from the fine reference.
Together with the non-XC component check, this accounts for the converged
energy difference; the remaining density relaxation contributes about
-0.0122 microhartree. The default-grid integrated electron count is
27.99970363. The finer-grid/support ladder above reduces the fixed-density
XC discrepancy to +5.68555 microhartree.

Several early diagnostic scripts accidentally shared one result filename.
Their scalar logs survive, and the final Si and corrected LiH matrix
snapshots were recovered into distinct files. Earlier full matrices could
not be recovered and are not claimed as available. The archive's
`native-result-recovery.json` identifies the affected runs, authoritative
outputs and retained scalar evidence. Exact executed scripts are kept
separately from corrected reproduction scripts.

Outstanding acceptance work includes broader crystal convergence checks
and a controlled performance review. The padded multi-k RHF CPHF
limitation is resolved by the follow-up below. The LiH and Si
20-to-24-bohr fixed-density cutoff checks above are complete. The global
physical-domain default stays off pending that broader review.
Terminal XC reuse and the replacement standalone gate regression have
completed their local validation.

The complete 840-case run reconciled on September 9 in 9,965.81 seconds,
with a 20.895-GB peak observed RSS. All setup/call/teardown and JUnit
identities agree, with no omitted or duplicate cases. All 568 native inputs,
600 Python sources and 17 selected test files matched their recorded hashes.
The two failures are exactly the grid assertions reviewed above; the raw
full-run result remains a failed run. See `completed-gate-reconciliation.json`.

The reviewed-fixture rerun passed both cases under the same native binary.
The legacy RKS pin now uses a strict absolute 1e-9-Ha tolerance (relative
slack disabled) and explicitly checks its unchanged spheropole. The moving-grid
force accuracy remains 1e-9 Ha/bohr; omitting the correction must fail by at
least 1,000 times that tolerance. The complete run plus these two reruns
establish 838 passing selected cases, with the one known padded CPHF expected
failure and one unchanged historical runtime skip reported separately.
`final-gate-reconciliation.json` preserves the separate full-run and rerun counts.

The historically skipped MgO corrected-Gamma RHF gradient passed all test
phases independently in 19,163.01 seconds with eight native threads. All three
SCFs converged, net force satisfied its bound, and the nonzero oxygen force
matched the central finite difference without changing settings or tolerances.
The exact process exit was zero; see `budgeted-gradient-reconciliation.json`.


## September 9 follow-up: padded RHF response after `294d3e5`

The former strict expected failure for padded multi-k RHF now passes as an
ordinary regression. Its asymmetric BeH2/STO-3G [2,1,1] fixture retains the
fold-reliable 13-bohr cutoff, all SCF settings, and the complete analytic-vs-FD
assertion body, including the 5e-5-Ha/bohr tolerance. No native code changed.

The previous finite-difference CG callback failed at iteration 42 with
`p^T A p = -1.715e-21`. Explicit unit-direction assembly gives a full-rank
48-dimensional response matrix; CG on that stored matrix converges in eight
iterations. In consistent complex orbital coordinates, its symmetric-part
minimum eigenvalue is 0.2600601 and its condition number is 20.69282. These
observations support a numerical callback failure rather than the former
exception's implication of an SCF instability.

The corrected response assembles the Jacobian using unit rotations, solves
its transpose, and includes the k-point weights in the energy RHS. Occupied
rotations use the conjugate-transpose convention matching the complex
occupied-virtual residual. The final contraction does not apply those weights
again. The assembled equation must satisfy the existing 1e-9 relative
residual tolerance; inconsistent singular systems still fail closed. Dense
matrix assembly remains a scaling cost, so this does not certify large-system
gradient performance or change the preview status.

Five small-model cases differentiate independently constructed normalized AO
density energies along a coupled SCF constraint. The four real/complex and
uniform/unequal-weight cases fail before the change and pass after it; the
singular inconsistency remains refused. Test-only forward and symmetrized
solves fail all four derivative cases. Restoring the old complex convention
or applying weights after solving each fails two cases.

The patched production process passed all seven selected cases: those five
models and both padded/unpadded BeH2 native gradient regressions. The separate
fast gradient and test-gate run passed 93 cases, giving 100 unique targeted
passes. Per-phase events and JUnit identities agree; the production source
hashes stayed unchanged throughout the run. A separate old-code diagnostic
with an explicit solver override passed its BeH2 finite-difference check at
7.07e-8 Ha/bohr maximum error; it is not counted as a production pass.

The local continuation archive `bipole-erfc-20260909-cphf` contains
`acceptance-summary.json`, `operator-analysis.json`, the raw native and model
results, the original expected-failure receipt, and a full patch against
`294d3e5`. Its inventory and isolated patch reconstruction preserve exact
source provenance. The earlier 840-case record remains unchanged; this
follow-up does not relabel that historical full run as green. Broader crystal
and cold-start/grid SCF checks, controlled performance/default review, and
fleet/tracker closure remain outside this targeted follow-up.

## September 9 crystal convergence and default review after `70aa93a`

This follow-up checks LiH rocksalt and Si diamond with the same PBE,
pob-tzvp-rev2 basis and unreduced [2,2,2] mesh as the earlier independent
level-9 comparisons. The native runs start from HCORE, with no supplied
reference density. They use physical 20-bohr pair/nuclear support,
omega 0.4, Ewald precision 1e-12, Schwarz threshold 1e-14 and SR image
precision 1e-8. The grid retains the default 75 radial by 17 theta by
36 phi points and 10-bohr image radius. The energy and commutator exit
criteria remain 1e-9 Ha and 1e-6; the iteration cap is 80. Two native
threads run the crystals sequentially on a shared host, so their wall
times are validation costs rather than a performance benchmark.

Both cold starts passed the exact, non-incremental terminal Fock check
and reused that confirmed build for final density evaluation. Their
energies agree with the earlier reference-density warm starts well
inside the predeclared 1e-8-Ha agreement tolerance. All production
Python source files and the native binary matched their recorded startup
hashes at return.

| Crystal | Iterations | Energy (Ha/cell) | Cold minus warm (Ha) | Exact commutator |
|---|---:|---:|---:|---:|
| LiH | 7 | -8.1034466442673 | -4.441e-14 | 5.581e-09 |
| Si | 10 | -578.6579424366507 | -4.547e-12 | 4.988e-08 |

The maximum AO density-element differences from the warm starts are 8.784e-09
for LiH and 1.566e-07 for Si. The occupied subspaces are reconstructed
from the saved native MO coefficients, with positive indirect gaps
checked before assuming equal integer occupations at every k point.
Wall times are 4178.05 and 4443.24 seconds; the combined guard's
peak observed RSS is 6.589 GB.

The separate generic fixed-density grid survey uses the independent level-9
PySCF density and finite-image AO sums on the native grid points and
weights. Bra-image reach equals the partition radius; ket-image reach
stays at 20 bohr. It does not wrap points or substitute unrestricted
periodic AO sums, which would change this finite operator. Native and
PySCF LibXC evaluations of the resulting density agree within 1e-9 Ha
in all 16 cases. Both default-grid values also reproduce the earlier
sealed calculations within 5e-10 Ha. All recorded input hashes match.

| Grid | Radius (bohr) | Points/cell | LiH XC error (µHa) | Si XC error (µHa) |
|---|---:|---:|---:|---:|
| default | 10 | 91,800 | +34.017 | +88.785 |
| leb100-29-r14 | 14 | 60,400 | -216.729 | -437.482 |
| leb150-35-r14 | 14 | 130,200 | -72.364 | -185.281 |
| leb200-41-r14 | 14 | 236,000 | -24.562 | -72.011 |
| leb200-41-r18 | 18 | 236,000 | -25.953 | -84.954 |
| leb200-47-r14 | 14 | 308,000 | -9.189 | -34.460 |
| leb200-53-r14 | 14 | 389,600 | -2.006 | -12.305 |
| leb200-53-r18 | 18 | 389,600 | -2.621 | -16.003 |

Here `lebN-L-rR` means N radial points, Lebedev order L and image radius
R bohr, with no angular pruning. Errors are relative to each independent
level-9 XC value at its fixed density, per primitive cell. These are
quadrature comparisons, not self-consistent fine-grid energies.

The archived LiH reference used PySCF 2.6.2; Si used 2.14.0. Their
"level 9" labels therefore do not specify identical radial and density-pruning
defaults. The explicit historical LiH radial formula and pruning rule reproduce
its 189,282 points and XC value within 3e-15 Ha under the current runtime.
Current 2.14 settings give 313,990 points and shift fixed-density LiH XC by
-0.062443 microhartree. This explains the version-sensitive reference check;
it does not certify either quadrature as converged.

The small Lebedev rules do not justify replacing the product grid:
order 29 is worse for both crystals despite using fewer points. At
200 radial points and radius 14, orders 41, 47 and 53 progressively
reduce the discrepancy. Increasing radius 14 to 18 at order 53 changes
XC by -0.616 microhartree for LiH and -3.698 microhartree for Si. This
identifies an improved grid candidate, but does not certify the grid
limit or replace the earlier, finer product-grid evidence. A complete
native fine-grid SCF ladder remains outstanding.

Four additional cases used the bundled `PySCFLevel3` atomic-grid profile,
which overrides the generic radial/angular/pruning controls as a complete
protocol. It produces 23,530 points for LiH and 38,360 for Si. At radii
10/14 bohr the LiH XC errors are -241.183/-299.164 microhartree, and
Si's are -2.385/-182.610 microhartree. The apparent Si agreement at the
smaller radius is not stable under increased support: its integrated
electron errors are +0.00058179/+0.00139717, both larger than the default
product-grid error magnitude of 0.00029637. This cheaper profile therefore
does not justify a default replacement. These four cases also passed
native/PySCF LibXC agreement and source-hash checks, bringing the complete
fixed-density survey to 20 cases.

The actual defaults were read from the imported native options and
Python driver signature. The validation overrides are material:

| Setting | Direct RKS API default | Cold-start validation |
|---|---|---|
| Physical pair domain | Off | On |
| Pair / nuclear cutoff | 15 / 25 bohr | 20 / 20 bohr |
| Ewald omega | Automatic | 0.4 |
| Ewald precision | 1e-8 | 1e-12 |
| Schwarz threshold | 1e-12 | 1e-14 |
| SR image precision | 1e-6 | 1e-8 |
| Initial guess | AUTO | HCORE |
| Energy / commutator tolerance | 1e-8 / 1e-6 | 1e-9 / 1e-6 |

The runner can resolve cutoffs differently; this table describes the
direct API used here. In physical mode at a fixed 20-bohr pair cutoff
and omega 0.4, the default SR precision resolves interaction radii of
33.2066 bohr for LiH and 32.7910 bohr for Si, versus 35.4720 and
34.9851 bohr at the tighter validation precision. The radius policy
also enables charge-pair Schwarz screening. Its precision parameter
is a radial-tail estimate, not a bound on the total lattice error.

The fixed-density native comparison changed all three screening precisions
back to their defaults, holding the physical 20-bohr pair/nuclear support
and omega 0.4 fixed. The non-XC energy changes relative to the tight
settings are -1.55e-12 Ha for LiH and -9.61e-10 Ha for Si. Maximum
Coulomb-matrix element changes are 1.90e-8 and 2.89e-9 Ha, respectively.
Every measured default-screening non-XC component remains within
3.96e-9 Ha of its independent reference. These changes are much smaller
than the default-grid XC discrepancies. They do not certify automatic
omega, the default cutoffs or a fully default SCF.

The first screening recorder failed after the LiH native build because
it read driver settings from the nested Fock callback's keyword arguments.
That attempt supplied no accepted J receipt; its owned Si child was
stopped when the shared recorder defect was identified. The corrected
callbacks passed isolated checks before both numerical probes were rerun.
Raw failed and completed attempts are preserved separately.

The read-only fleet inventories captured for this follow-up do not attest
this branch: compute-cluster's registered development runtime reports source
`5ad944fbeffe` and
version 0.15.162; compute-managed's executable registrations expose no source
SHA. No cluster calculation or deployment was used for these results.
The issue snapshot records all five migrated issues as open: #187 (old #20),
#186 (old #21), #182 (old #66), #82 (old #674), and #44 (old #724). The two-crystal
checks do not cover the complete affected crystal set, automatic-omega
and cutoff defaults, or controlled HF/hybrid performance. The global
physical-domain switch remains off; no tracker closure is claimed.

The `bipole-erfc-20260909-crystal-convergence` archive records the exact
executed scripts, startup provenance, raw logs, grid tables, screening
comparison, cold/warm reconciliation and required earlier references.
Its full patch reconstructs the committed tree from the standalone
base, and its inventory verifies every archived file.

## Rebased native fine-grid follow-up (2026-09-09)

This follow-up rebased the continuation onto standalone main
`ed5deffae394ec195fdb2f000b4779728e85a1f3` and rebuilt the native extension.
Its SHA256 is `089312df98c2a410b75b46f9a569a487039b2e81a0936940f8f0f4dff1d672ba`.
The rebased integration checks passed 181 binding/domain/terminal/CPHF tests,
506 upstream physical-product/finite-source/Gram tests, and the two maintained
padded/unpadded multi-k RHF finite-difference force gates. These 689 checks are
separate from the historical raw 840-case run described above.

The rebuilt default-grid LiH and Si continuations converge in two iterations
and reproduce the prior HCORE solutions within 8e-12 Ha. Independent
PySCF/LibXC XC contractions agree within 8e-12 and 4.5e-9 Ha respectively.
Fresh independent RSJK Coulomb operators plus the sealed one-electron
matrices reproduce total energies within 1.3e-11 and 2.3e-9 Ha, and Fock
matrices within 1.4e-9 and 3.4e-9. These default checks reconstruct the returned
canonical-MO densities; they do not claim to use the exact accepted lattice
density, which the initial default recorder did not serialize.

The first fine-grid attempt exposed an AUTO-domain radius propagation bug:
the quadrature used the requested 14-bohr radius but XC's AO bra images still
used the unset native 10-bohr default. Commit `6dee928` propagates the requested
periodic-grid radius in RKS and UKS. At the saved LiH seed, independent XC
reproduces the defective iteration's energy within 5e-11 Ha, and increasing
only the bra radius from 10 to 14 shifts XC by +1.3043 microhartree. The
partially run defective case is retained and excluded from fine-grid acceptance.
The fix passed 144 terminal/provider tests and 47 KS integration tests.

The corrected native fine-grid protocol uses 200 radial points, unpruned
Lebedev order 53 and radius 14 (389,600 points per cell), physical pair and
nuclear cutoffs of 20 bohr, omega 0.4, Schwarz 1e-14, Ewald precision 1e-12
and SR precision 1e-8. Every intercepted native XC call records and checks its
actual 14-bohr bra radius and 20-bohr density cutoff. The recorder saves the
accepted lattice density, its validated per-k representation, and the final
Fock/overlap/hcore matrices. Canonical orbitals are retained separately.

Native LiH converged in four iterations to -8.103482649513941 Ha/cell.
Its exact accepted-density commutator is 1.464e-7; the exact rebuild differs
from the loop energy by +4.831e-11 Ha and is reused for final evaluation.
All five XC calls use the requested 14/20-bohr support. Independent XC at
the saved accepted density agrees within 2.3e-15 Ha, and the independent
full operator agrees within 1.1e-11 Ha in energy and 8.3e-10 in the Fock
matrix. The density round-trip error is below 6e-17.

A separate PySCF AO/LibXC/RSJK SCF loop, with its own DIIS, diagonalization
and stopping rules, starts from the independent level-9 density. It shares
only the native quadrature and uses sealed reference one-electron matrices.
For LiH it converges in four iterations to -8.103482649503272 Ha/cell,
with residual 5.797e-8. Its energy differs from the native result by
1.1e-11 Ha; the orthonormal-density difference is 5.9e-8. The corrected
fine-grid native LiH result is 36.005 microhartree below the default grid
and 2.006 microhartree below the finite level-9 reference.

Native Si converged in five iterations to -578.6580435165043 Ha/cell.
The exact accepted-density residual is 1.199e-8, and the confirmation changes
energy by +5.093e-11 Ha. Five loop XC builds plus one confirmation are
recorded, with no additional terminal build. The density round-trip error
is below 5e-16. Independent accepted-density XC agrees within 3.6e-14 Ha;
the independent full operator agrees within 2.3e-9 Ha in energy and 2.1e-9
in the Fock matrix.

The independent Si loop converges in six iterations to
-578.6580435142816 Ha/cell with residual 1.327e-8. Native and independent
energies differ by 2.3e-9 Ha, and the orthonormal-density difference is
1.6e-8. The fine-grid native result is 101.080 microhartree below the default
grid and 12.308 microhartree below the finite level-9 reference. Thus both
independent loops reach the same fine-grid solution within the predeclared
energy/density/residual gates. They do not establish the infinite grid
limit. Native driver wall times are 5844.05 seconds for LiH and 6851.46
for Si, under shared host load; these are not controlled benchmarks.

The native Si case was started concurrently with LiH using a byte-identical
script and the same four-thread, four-hour and 24-GiB case limits. The
original serial controller's later duplicate was stopped and its -15 exit
preserved. Accepted-case assembly requires a zero exit from the concurrent
Si controller and explicitly excludes that cancelled duplicate. The raw
serial campaign is retained as incomplete; it is not relabelled successful.

The associated gradient radius correction is committed as `c0c3737`,
with test-only strengthening through `529c2a1`. It copies lattice options
without changing the caller's fields and propagates the grid radius through
the fixed-grid and moving-grid XC terms. The new tests fail in eight cases
on the old source and pass all 12 with the correction. The preservation
regression enumerates all 13 public native option fields independently of
the helper. A deliberate omitted-field mutant fails in both periodic KS
cases, while all eight unmodified preservation combinations pass.

The checkout's own native environment completed the four maintained Gamma
LDA/PBE RKS/UKS force witnesses and the remaining slow matrix, with 42 passes
and one historical skip. These processes loaded the exact committed gradient
module without changing the package files held by concurrent native SCFs.
All startup source hashes match at return. Every preexisting test function
and fixture is AST-identical to the committed final test file.

After integration, the checkout's own native environment passed all 57
fast gradient cases in 147.28 seconds. Combining those with the four Gamma
KS witnesses, 42 remaining slow passes and two rebased padded/unpadded RHF
witnesses accounts for every current gradient test exactly once: 105 pass
and one historical high-cost MgO skip, across 106 collected nodes. No
preexisting test tolerance was relaxed. The skip is not promoted to a pass
on this binary; its prior unskipped result belongs to the earlier archive.
The complete follow-up contains 965 distinct passing test nodes across the
rebased integration and radius/gradient gates, with that one skip. This is
selected-gate evidence across the recorded commits, not a full-inventory run
on a single final SHA.

H2 RKS LDA/PBE force errors after that correction are below 3e-9 Ha/bohr;
H3 doublet UKS errors are about 5e-6, versus errors above 1e-3 before the fix.
The exploratory 1e-7 criterion failed for UKS and remains a recorded failure.
The smaller remaining error depends on the preview's internal grid-motion
finite-difference step. The permanent UKS regression uses 1e-5, checks two
smaller relaxed-SCF difference steps, and changes no existing tolerance.
This does not broaden analytic-gradient certification.

A separate fixed-state probe enables physical domains but otherwise uses
direct-API cutoff/screening/automatic-omega defaults. The shared options clamp
the requested nuclear cutoff from 25 to 15 bohr; private Ewald source support
is resolved separately. The non-XC sums differ from the independent reference
by -0.224 microhartree for LiH and -149.872 for Si. The Si input electron trace
is 27.99997991996 before normalization; its scale factor is 1.000000717145.
These component differences include that density normalization and cannot be
attributed to a single cutoff term. This one-build probe stops before XC and
is not a full default SCF. It reinforces the decision to retain opt-in physical
domains and to avoid promoting the current cutoff/grid defaults.

An off-tree finite-Bloch XC factorization reproduces native default-grid
energies within 3e-14 Ha and raw potential matrices within 2.5e-13 on both
crystals. Non-TRIM complex-density controls discriminate the phase convention;
directional derivative and batch-size checks also pass. One-thread observed
wall times are 72/345 seconds for LiH and 76/315 for Si (prototype/native),
under shared host load. This is neither a controlled benchmark nor a
production optimization. A production path still needs validated accepted
per-k densities, bounded AO batching, and separate spin/meta-GGA/external
provider/gradient coverage.

The final `bipole-erfc-20260909-native-fine-grid` archive contains the four
accepted native SCFs, authoritative fine-grid densities, independent
XC/operator/loop checks, gradient collection and coverage accounting,
expected negative controls, source snapshots, and the exact tested binary.
Its continuation patch reconstructs the committed tree from the standalone
base, and its inventory hashes every archived file. Local acceptance does
not close the global physical-default, complete crystal-set, asymptotic-grid,
HF/hybrid performance or independent-person verification gates. No cluster
run, deployment, tracker mutation or upstream push was used.

## Explicit reference and angular-grid refinement, 2026-09-09

The follow-up separates radial, angular and image-radius effects at both the
saved native accepted density and the archived independent reference density.
All nine profiles per material are accounted for, giving 36 primary density
evaluations plus the repeated controls.
Failed attempts remain separate from completed continuations and their repeated
controls. Both time-guarded attempts are resumed with an equivalent optimized
contraction order, with real/complex controls and a repeated native-grid check.
Native and PySCF LibXC energies agree within 1e-9 Ha throughout. The initial LiH
current-version reference check remains a recorded failed attempt; its
historical-grid reproduction and the version-aware rerun are separate receipts.

Five additional independent PBE reference SCFs use explicit atom-specific
Treutler radial grids, NWChem angular pruning and no density pruning under
PySCF 2.14.0. The unreduced [2,2,2] meshes, bases, sealed analytic one-electron
matrices and fresh exact RSJK Coulomb builder remain fixed. Every accepted
state passes the fresh final Fock check, with maximum commutator
5.689e-10. Source hashes remain unchanged.

| Radial / angular points | LiH energy (Ha/cell) | Si energy (Ha/cell) |
|---|---:|---:|
| 300 / 2030 | -8.103479811961442 | -578.6580270441226 |
| 400 / 2702 | -8.103479770512775 | -578.6580259849007 |
| 500 / 3470 | Not run | -578.6580257401564 |

The last reference refinement changes LiH by +0.041449
microhartree and Si by +0.244744. Si's extra
500/3470 calculation follows the 1.059222-microhartree change between the
first two refinements. These are measured steps, not an exact-limit bound.

Two independent finite-image SCFs replace the 200/L53 angular rule with a
200-radial, 35-theta, 72-phi product grid at the same 14-bohr bra/partition
radius and 20-bohr ket support. They start from the previously verified
independent Lebedev-grid states. Their AO evaluation, XC contraction, RSJK,
DIIS, diagonalization and convergence logic are independent of the native
BIPOLE SCF; the native quadrature and sealed reference one-electron matrices
are shared. Both pass the 1e-10-Ha energy-change and 1e-7-commutator criteria.

| Crystal | Independent product-grid energy (Ha/cell) | Minus last reference (µHa) |
|---|---:|---:|
| LiH | -8.103479620614127 | +0.149899 |
| Si | -578.658026129749146 | -0.389593 |

The first product-loop energies match the fixed-density grid-change predictions
within 1e-8 Ha. Subsequent density relaxation changes LiH and Si by
-0.000115 and -0.001396
microhartree, respectively. The following controls use the saved native
accepted densities, with all other parameters held fixed unless stated.

| Grid change | LiH XC change (µHa) | Si XC change (µHa) |
|---|---:|---:|
| Lebedev 53: radial 200 to 300, R14 | -0.527701 | -3.072054 |
| Lebedev 53: radius 14 to 18, radial 200 | -0.615610 | -3.698320 |
| L53 to product 35/72, radial 200, R14 | +3.029003 | +17.385929 |
| Product 35/72: radius 14 to 18, radial 200 | -0.145861 | +0.607076 |
| Product 200/35/72 to 300/47/96, R18 | +0.135633 | +0.096939 |

All numerical workers finish before the record is sealed. The
`bipole-erfc-20260909-grid-limit` archive preserves scripts, failed and
completed attempts, authoritative densities, current and historical reference
sources, runtime hashes and the continuation patch. The production extension
remains the previously tested `089312df` binary.

The completed native SCFs still use 200/L53/R14; the new product-grid SCFs
are independent loops, not additional native SCF acceptances. This study
does not promote grid/cutoff defaults or close broader crystal, HF/hybrid
performance, independent-person review or global physical-domain gates.

## Midpoint-aware nuclear support, 2026-09-09

The physical nuclear selector centres its source ball on the arithmetic
AO-pair midpoint. The Python Ewald helper previously retained legacy
cell-origin padding, so merely relabelling one atom by a lattice vector
unnecessarily enlarged that ball. For positive primitive exponents, the
Gaussian product centre lies at most half the pair separation from the
midpoint. The physical branch now adds half the pair cutoff to the smeared
erfc range. The legacy branch and explicit larger source cutoffs are preserved.
Energy and analytic derivatives continue to use the same helper.

Three atom-image relabelling controls fail on the old helper and pass with
the correction. All 24 nuclear tests pass, including 1D/2D/3D labels, diffuse
and asymmetric s/p/d products, larger-domain energy/gradient comparisons and
caller-option preservation. No native extension rebuild is required: the
tracked C++ sources and tested `089312df` extension are unchanged.

The saved native fine-grid LiH and Si densities also compare the exact old
helper with the corrected helper, using the same native extension. These
are fixed-density nuclear-component witnesses, not new native SCFs.

| Crystal | Old / midpoint radius (bohr) | Nuclear energy change (Ha) | Max matrix change | Max gradient change |
|---|---:|---:|---:|---:|
| lih | 42.954732 / 26.271058 | -1.776e-15 | 1.488e-14 | 4.770e-17 |
| si | 40.358462 / 25.914408 | +2.274e-13 | 4.707e-14 | 7.262e-16 |

The two H2/SVWN retirement pins are corrected separately. Both failed on
the unchanged checkout. Restoring only the historical grid partition
reproduces both old pins within their original 5e-9-Ha tolerance; the
corrected partition shifts both routes by +4.05943148 microhartree entirely
in XC. The two numeric anchors are the only AST changes in that test file.
All eight tests pass on rerun, with unchanged bodies and tolerances.

The 67-file fast BIPOLE lane initially records 1,906 passes and three failures.
The reviewed retirement rerun resolves two of those failures. Two files have
only slow cases and therefore no selected fast tests; 31 non-gradient slow
cases remain deselected across the lane. The explicit slow-gradient batches
cover all 49 additional gradient nodes. Together with the 57 fast nodes,
the gradient matrix has 105 passes and one historical expensive MgO skip.
The original 14-GiB-guarded batch returned no accepted tests; its complete
18-node rerun is recorded separately under a 24-GiB/four-hour guard.

The other two original gradient workers retain their processes, threads,
source and 14-GiB limits. Their local time budget is extended from two to
four hours, measured from their original starts, through an overlapping
supervisor handoff. Darwin exit-status observation is checked against
normal, nonzero and signal exits before use. The original supervisor is
recorded as superseded; its old time budget is not retroactively labelled
a successful bounded run.

After exact accounting, these selected checks contain 1,980
distinct passes, one historical gradient skip and one unresolved existing
SAP failure. This is not a clean full repository gate or independent-person
verification. The raw lane failures and stopped resource-guard attempts
remain in the archive.

`test_vsap_gamma_symmetric_and_attractive` fails identically on the unchanged
checkout and candidate: raw SAP Gamma asymmetry is 0.001311559 against 0.001.
The midpoint helper is not called. Restoring the historical partition gives
0.000108793; current radius-14/radius-18 and physical-domain controls still
exceed the unchanged threshold. No test tolerance, skip or production default
is changed to absorb this failure. SAP/grid analysis remains open alongside
broader physical-default, crystal-set, HF/hybrid performance and other-person
verification gates.

## SAP quadrature and upstream integration, 2026-09-12

The continuation is rebased onto standalone core main `3a5f1d5`. The merge
resolution retains the new ECP SAP dispatch inside physical ERI-cell mapping,
and keeps both the upstream MDF and BIPOLE test-lane assertions. The native
extension is rebuilt against this checkout's vendored dependencies; its
SHA256 begins `263b78c8`. The September 9 source and numerical archives remain
separate. Their 1,980-pass count and refined SCF values are not re-certified
for this rebased tree.

The previously unresolved SAP symmetry test exposed a quadrature contract
mismatch. Each SAP lattice matrix block integrates one localized AO pair
over all space, while the shared Python caller supplied periodic per-cell
weights without summing bra images. The smooth potential is periodic; the
localized AO product is not. The caller now uses the same molecular Becke
partition as the native SAP drivers. The direct integral tests use that
contract as well. Neither the periodic potential nor the existing symmetry
threshold is changed.

Two new regressions inspect the grid at the real Python SAP call boundary.
For a constant potential, its quadrature must reproduce the analytical
localized overlap within 2e-6. They also check the raw SAP matrix before the
helper's existing Hermitian projection. Fresh-core diagnostics give:

| LiH cell edge (bohr) | Periodic-weight overlap error | All-space overlap error | SAP asymmetry, periodic / all-space |
|---|---:|---:|---:|
| 7.6 | 0.289839 | 7.951e-8 | 0.0346592 / 0.000229442 |
| 12.0 | 0.0712418 | 4.226e-7 | 0.00131156 / 0.000108793 |

The rebuilt baseline fails both new regressions and the old symmetry test.
A stale generated basis overlay was then refreshed. Its STO-3G file bytes
changed, so all three failures were reproduced again with the refreshed
basis data before testing the fix. The SAP table stayed byte-identical.
Final test receipts attest the refreshed overlay alongside Python source
and the rebuilt extension.

The completed targeted groups record 285 distinct passes. They
include the full SAP test file, explicit open-shell SAP caller checks,
150 physical/nuclear cases and 93 binding, lane and caller-integration checks.
A broader 189-case SAP/open-shell attempt is stopped during GDF Fourier-factor
setup for a NaH ECP AUTO/SAD case, before reaching its guess builder. Its
partial passes are not counted; the relevant SAP cases are rerun separately.
That broader attempt also exposes a PATOM first-cycle comparison failure.
The PATOM failure reproduces with the
pre-fix helper restored in memory and all SAP construction calls forbidden;
that route never enters the changed construction. Its first-cycle energies
are -1.198539021862297 Ha (GDF) and -1.196328119109694 Ha (direct Ewald),
a -2.210903-mHa difference. These are one-cycle states, not converged SCFs.
The comparison remains unresolved, with the assertion unchanged.

These are targeted integration results, not a full BIPOLE lane or complete
gradient-matrix rerun. The historical MgO gradient skip, broader
physical-default and crystal-set work, and independent-person review remain
open. The separate `bipole-erfc-20260912-sap` archive preserves the rebase,
failed baselines, completed results, source and basis hashes, rebuilt binary
and exact-tree continuation patch. No upstream push, fleet operation,
tracker mutation or issue closure is performed.

## Standalone issue review, 2026-09-13

The continuation was rebased onto standalone main `9010c00`, producing
`e654cb8`. All 576 recorded native build inputs match the September 12
vendored build, extension SHA256
`263b78c8f53a13c2a47d78c3e1aa2ef7dc4d3d54b50b7ca9191687f1c57e6289`.
The rebase refreshed source timestamps, so the timestamp-only stale-core
warning is present; the native content comparison has no mismatches.

Standalone issue #216 groups the retirement and legacy-gauge reference
shifts. Six new converged SCFs repeat the paired partition experiment on
this native build. Only the grid partition changes within each pair.

| Route | Historical partition (Ha) | Point-centered partition (Ha) | Change (microhartree) |
|---|---:|---:|---:|
| SR/LR | -1.1211961982135232 | -1.1211921387820452 | +4.059431478 |
| Exact FT | -1.1211968598808090 | -1.1211928004493320 | +4.059431477 |
| Legacy gauge | -1.1156375080452340 | -1.1156334486137565 | +4.059431478 |

Every historical value recovers its original pin within the original
absolute tolerance. Kinetic, nuclear-attraction, nuclear-repulsion and
J/K components are unchanged to 1.2e-16 Ha; the energy change is wholly XC.
The exact-FT minus SR/LR gap stays -0.661667286 microhartree within
9e-16 Ha. The legacy spheropole is exactly
0.005559511758728281 Ha in both runs. These controls establish the shared
partition cause without a gauge offset or altered acceptance tolerance.
The two affected test files pass on this build.

Additional completed checks give 125 passes for native binding, gate,
Schwarz and nuclear coverage, and two passes for the H2/H-atom
BIPOLE-versus-Ewald controls at the existing 1e-7-Ha tolerance. The broad
fast and slow runs are distinct from these completed selections. The fast
run encounters a per-test timeout in the wide reference build of
`test_padded_internal_cells_reproduce_wide_build`; its raw failed receipt
has no completed pytest count. A shared resource hold prevents queued
files from starting. Neither held targets nor partial output count as
passes, and the old four-file known-red entry is not retired on this basis.

Issue #220 now tracks the September 12 all-space SAP quadrature correction.
The separate PATOM first-cycle diagnostic remains unresolved: suppressing
only GDF's one-electron cutoff growth from 7 to 29.826949 bohr reduces the
GDF-minus-Ewald difference from -2.210903 to -0.296401 mHa, without entering
SAP. This local diagnostic does not propose suppressing that growth in
production or establish a converged-SCF discrepancy.

Issue #163 also reports missing hybrid XC output. The renderer now reads
`result.e_xc` when printing BIPOLE's terminal component history. Four PBE0
RKS/UKS probes, covering converged and capped returns, retain energy
-2.837407679083663 Ha and display the stored XC value
-0.8198135756723072 Ha. No quadrature is added. Three pure rendering checks
cover absent, zero and nonzero XC values. This reporting repair does not
resolve the issue's separate BIPOLE/GDF physical comparison.

The global physical-domain default, all 12 crystal comparisons and the
full-node HF/hybrid performance target remain open. The historical numerical
archives retain their original source and native identities; a new rebase
is not evidence of a fresh execution of those full campaigns.
