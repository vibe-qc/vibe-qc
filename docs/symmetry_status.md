# Symmetry capabilities and production-reduction status

The shared mathematical infrastructure is implemented and experimental.
Production reduction is qualified separately for each numerical method,
finite support, electronic state and retained space. There is no blanket
completion or production-authorization claim for symmetry reduction.

## Shared infrastructure

| Capability | Implementation and boundary |
| --- | --- |
| Finite groups and lattice image cocycles | Native exact algebra checks, scalar antiunitary parity and point-group conjugacy classes are available. A valid table does not establish geometric completeness. |
| Compact space actions and orbit operations | Budgeted block actions, typed density/operator transport, stabilizers and adjoint scatter/gather are available without dense all-space matrices. |
| Molecular and periodic AO transport | Molecular compact actions support mixed pure/Cartesian shells through `l=6`, sharing the periodic polynomial kernel. Periodic AO/Bloch transport retains atom images, fractional translations, mesh wraps and scalar time reversal. The legacy dense molecular API requires pure shells beyond s. |
| State and retained-space diagnostics | Metric, group-law, operator and leakage checks can bind a native periodic state and selected spaces. A successful numerical witness does not certify a producer or authorize pair skipping. |
| Production method integration | An experimental SR/LR GDF retained-auxiliary tensor diagnostic compares independently built batches. Production GDF scheduling, PAO/PNO/TNO transport, reduced correlation execution and analytic derivatives retain separate implementation and acceptance gates. |
| Complex Abelian tensor execution | Experimental `symmetry_abelian` validates complete ordinary character tables and a supplied finite chemist ERI tensor, stores allowed blocks and contracts J/K on arbitrary complex densities. No source generation, production SCF dispatch, projective little groups or magnetic corepresentations are enabled. |
| Ordinary non-Abelian product algebra | Experimental `symmetry_products` resolves supplied ordinary unitary matrix irreps into multiplicity and partner spaces, converts with the actual finite coordinate inverse, and composes compatible reduced maps. Complete tensor audits and explicit recoupling remain required. C3v/A4 algebra and the corrected ammonia J/K harness are accepted below; no producer, CC residual, periodic or derivative admission follows. |
| Molecular representative integral execution | Experimental `symmetry_quartets` audits the complete finite MANTA P0 libint source, then evaluates shell representatives for arbitrary-density J/K. C2v and full C3v AO transport, real pure/Cartesian shells and stabilizers are tested. The optional `symmetry_quartets_streaming` schedule retains quadratic pair maps instead of the stored quartet inventory. Complete audits remain available; optional native Q/V bounds skip reads for omitted whole orbits while auditing retained members. Arbitrary-density contraction terms remain explicit. No production routing, complex integral source or reduced non-Abelian CC tensor is enabled. |
| Native MD quartet source | Experimental low-level admission extends the complete ordered-tile/stabilizer audit to MANTA `DirectJKBuilder`, with distinct source/Fock provenance and an identical-source ordered fallback. Affected regressions and complete ammonia RHF/UHF acceptance are recorded below. Libint pre-audit bounds and default prepared-source dispatch are not transferred. |
| Native quartet execution | Opt-in `executor="native"` compiles angular transport, member permutations and complex-density J/K contractions for either admitted schedule. Selected families share resolved matrix addresses across density channels; scalar shells bypass angular index loops. Integral generation, source digests and scientific tolerances remain unchanged; the chosen qualification policy controls audit reads. It is a provider-independent executor, not MANTA kernels or production SCF dispatch. |
| Shared-density quartet batches | `jk_batch` reuses each representative integral and angular transform for independent complex density channels. NumPy/native paths preserve per-channel J/K and source-error bounds, aggregate actual work, immutable density snapshots and whole-batch refusal. Optional per-channel J/K selectors skip unrequested contractions and output panels, with explicit compact-output mappings. An optional diagnostic checks method-assembled snapshots against the separate Fock-object draft. No spin/state selection or production driver dispatch is implied. |
| Fixed-orbit molecular screening | Opt-in `QuartetScreening` binds a density-independent whole-orbit mask to a stored or compact streamed P0 qualification. Both batch executors skip actual representative calls and contractions, retain separate source/policy identities and report per-input bounds. Optional MANTA Q/V envelopes omit orbits before reads; retained-only audits require that exact mask. Mutable callbacks, producer-side primitive screening and periodic sources remain outside this admission. |
| Native hierarchical bound candidates | Experimental `symmetry_quartets_hierarchy` consumes MANTA bubble traversal and order-zero full Gaussian-tail bounds. Native candidate blocks spend an ordered-AO tensor budget; only their intersection with complete quartet orbits may omit reads. Retained audits and fixed-mask requirements remain. No default dispatch, tree equivariance or periodic admission is inferred. |
| Explicit molecular HF bridge | `symmetry_molecular_hf` connects one admitted fixed P0 mask to native real RHF/UHF and method-owned HF `FockBuild` snapshots. `symmetry_molecular_scf` exposes explicit public RHF/UHF options and consistent evaluated density, energy, Fock and natural-orbital QVF output, including capped and NOITER states. Native UHF jointly requests total-density J and independent spin K; spatially broken states remain allowed. Default/job-deck dispatch, zero-copy native capture, periodic/complex-source/derivative admission remain separate. |
| Measured molecular route selection | Opt-in `executor="auto"` compares ordered retained-tile evaluation with representative reconstruction on nonsymmetric probes. Five bounded native HF signatures have separate cost decisions. Producer/audit/source/policy identities stay fixed; calibration is separately accounted. This does not bypass retained-source audits or enable default, periodic, RI/BUPO or complex-source SCF routing. |
| Explicit molecular job selection | `run_job` accepts the qualified RHF/UHF options or prepared source. Executable QVF records carry a strict fresh-source request, never saved audit authority. Normal output uses the physical evaluated state and labels natural orbitals; canonical frontier diagnostics refuse. Scientific job evidence is recorded below. |
| Exact terminal J/K reuse | Public qualified RHF/UHF keeps bounded method-local evaluations and assembles terminal FockBuild without more integrals when source, ordered channels and density bytes match. Separate native RHF J/K calls can be joined. Mismatches retain the cold path; prepared sources contain no state. |
| Reusable qualified molecular sources | `prepare_molecular_symmetry` owns one actual audit, fixed mask and checked one-electron source. Independent public RHF/UHF states can reuse it and its bounded scalar cost profiles after exact geometry/basis checks, with fresh state and physical counters. New sources still require qualification; changed geometry, radial content, overlapping use and cross-process reuse refuse. |
| Molecular Abelian RI HF | Experimental public RHF/UHF and job entries consume MANTA-owned selected factors with a complete AO/auxiliary retained-space audit, complex character selection, native block contractions, measured dense/blocked routing and exact terminal reuse. Accepted same-source and matched-reference calculations are recorded below. Native metric/rank cuts stay fixed; reduced factor generation, non-Abelian RI, periodic sources and derivatives require separate qualification. |
| Multi-k GDF exchange wedge | `symmetry_reduce_k` (driver `ibz_native`) builds exchange at wedge bras only and admits each transporting operation against the executed exchange source before any integral: every space-group operation, glides and screws included, on the pair-centred SR/LR fit; only equal-lattice-shift operations (the image shift computed with the fractional translation) on the origin-centred classic fit; none on other sources. A fractional translation is carried by the atom-image shift of the Bloch transport, with no separate `exp(-i k.tau)` factor in the AO-centred convention. Transported and directly built exchange agree to roundoff on nonsymmetric complex trial densities for diamond Fd-3m and hcp P6_3/mmc. Occupations are filled once per star on the wedge representatives, with levels split within the measured star spectral spread sharing their electrons, so a star-degenerate zero-temperature frontier receives the symmetric ensemble fill instead of a noise-selected one (#428). The state gate transports every representative through its little group as well as every member through its star operation, so a wedge-native run that passes it is invariant under every admitted operation (#470). Cartesian shells beyond s, a proof of electronic-state invariance (the SCF gate is a calibrated threshold) and floating-point membership of pairs lying exactly on a cutoff surface remain outside this admission. |

The [developer reference](developer_shared_symmetry.md) defines the implemented
APIs and their resource contracts. The [architecture page](design_shared_symmetry.md)
describes the remaining production gates. All shared-layer evidence retains
`production_reduction_authorized == False`; a method must establish its own
admission policy against an unreduced reference. Method-specific calculation acceptance of a
particular change belongs to its tracker record, not to a general capability
checkmark.

The multi-k GDF wedge also binds every executed k point and normalized weight
to the declared full mesh before integral work. It preserves half-step shifts
and reciprocal wraps, and refuses reordered or inconsistent point lists.
This quadrature prerequisite does not strengthen the calibrated electronic
state gate or qualify additional finite sources, local spaces or derivatives.
For an already reduced input, the wedge path also validates the original
representatives, orbit weights and full-grid mapping before expansion can
replace them. Consistent representative reordering and reciprocal wraps are
preserved; inconsistent reductions are refused before integral work.

## October 2026 literature increment and measured results

The [literature assessment](symmetry_literature.md) covers Melega's geometry
supplement, the C3v scope of Hellmann and Gauss, the explicitly labeled
Kitsaras and Stopkowicz preprint, MolSym's two supplements and libmsym,
with connections to the molecular and periodic foundations. The
[design](design_shared_symmetry.md) specifies the non-Abelian tensor and
MANTA/BUPO integration requirements. The historical measurements below start
with the bounded P0 reference-tile adapter; later sections record molecular
HF, RI, routing and ordinary product-algebra increments. Integral kernels and
BUPO hierarchy implementations remain owned by MANTA.

Measured on the current matching native extension with Python 3.14 and
NumPy 2.5.3, one requested OpenMP/BLAS thread, 31 alternating timed
contractions after warmup, seed 6128, and source tolerance `1e-10`.
The density is complex Hermitian and not symmetry invariant. Both paths
use the identical unreduced finite molecular integral source; the block
path omits only forbidden entries below its recorded aggregate tolerance.
All ERIs are computed first, then transformed with native compact actions
and overlap-metric-normalized projectors. No geometry or final Fock matrix
is symmetrized. These are development measurements, not speed guarantees.

| Case / admitted group | AO count / irrep populations | Full / retained ERI values | Full / block J+K scalar products | Block calls | Median full / block time (ms) |
| --- | --- | --- | --- | --- | --- |
| NH3 STO-3G / C3 subgroup of C3v | 8 / 4,2,2 | 4096 / 1376 | 8192 / 2752 | 54 | 0.0167 / 0.3525 |
| NH3 cc-pVDZ / C3 subgroup of C3v | 29 / 11,9,9 | 707281 / 235771 | 1414562 / 471542 | 54 | 2.8650 / 1.4416 |
| H2O cc-pVDZ / C2v | 24 / 11,4,2,7 | 331776 / 88456 | 663552 / 176912 | 128 | 1.3674 / 1.4425 |

| Case | Full complex ERI / retained array payload (bytes) | Preparation / plan time (ms) | Process peak RSS (bytes) | Maximum J / K error against direct AO contraction | Trial RHF energy-functional error (hartree) |
| --- | --- | --- | --- | --- | --- |
| NH3 STO-3G | 65536 / 22384 | 4.92 / 1.23 | 158433280 | 2.23e-15 / 4.71e-15 | 1.43e-14 |
| NH3 cc-pVDZ | 11316496 / 3773040 | 30.67 / 14.70 | 212959232 | 1.79e-13 / 1.53e-13 | 5.28e-13 |
| H2O cc-pVDZ | 5308416 / 1416096 | 18.55 / 9.90 | 212353024 | 2.67e-14 / 3.06e-14 | 1.15e-13 |

Retained payload includes indices and character tables, but excludes native
group and Python metadata. RSS is for the whole fresh benchmark process,
including imports and both reference and reduced arrays; it is not the
incremental memory of the plan. Conservative setup byte reservations are
318150, 45421446 and 21401928, respectively. Forbidden ERI Frobenius norms
are 8.53e-15, 1.75e-13 and 1.05e-13. Each case makes one full `compute_eri`
call. Native primitive/shell-quartet evaluations are not instrumented here:
**no integral-work reduction is claimed**.

Only the larger ammonia case is faster (about 1.99x for contractions), and
its plan setup takes about 11 saved builds to amortize. Water is about 5%
slower and the tiny ammonia case about 21x slower despite fewer arithmetic
terms. Production dispatch needs a measured cost gate or a lower-overhead
executor. The unreduced path remains necessary for both incompatibility and
cost. There is no end-to-end SCF or CC speedup claim.

At independently converged native STO-3G RHF densities, water's total
reference energy is -74.96386817084978 hartree and ammonia's is
-55.450387020207344 hartree for the benchmark fixture geometries. Rebuilding
with block J/K changes their electronic energy functionals by 2.85e-14 and
8.33e-17 hartree, respectively; maximum Fock differences from the native
reference are 8.68e-15 and 8.89e-15. These terminal-density checks supplement,
rather than replace, the nonsymmetric-density tests. The new adapter does
not itself run the SCF iteration.

Focused regression coverage includes native molecular C2v/C3 ERIs, complex
nonsymmetric Hermitian and non-Hermitian densities, Cartesian metric-aware
actions, independently transformed AO J/K and energy functionals, native
converged RHF Fock comparisons, density-energy finite differences, separate
spin calls, dense within-irrep gauges, separately conjugated time-reversal data, displaced
sources and rejection of unsupported non-Abelian/antiunitary/cocycle tables.
All scientific tolerances and production defaults are unchanged.

Validation commands and completed results:

* `pytest tests/test_symmetry_abelian.py -q`: 17 passed.
* `pytest tests/test_symmetry_shared.py tests/test_symmetry_gdf.py
  tests/test_periodic_k_symmetry.py tests/test_symmetry_salc.py -q`:
  468 passed (run together with the first 13 new tests: 481 passed).
* `pytest tests/test_citation_database_integrity.py
  tests/test_citation_no_dead_entries.py -q`: 1379 passed. These are
  citation/schema checks, not numerical chemistry cases.
* `pytest tests/test_test_gate_lanes.py -q`: 76 passed after explicitly
  excluding the new experimental T3 file from shipped/release lanes.
* `pytest tests/test_periodic_ao_bloch_transport.py
  tests/test_periodic_orbital_sewing.py tests/test_periodic_mean_field_state.py
  -q`: 312 passed in 2.45 s, with 16 existing spglib deprecation warnings.
  This bounded rerun covers odd/even meshes, nonsymmorphic phases,
  Cartesian metrics, degenerate retained spaces and native finite-Gaussian
  states independently of the incomplete larger run below.
* Candidate-byte privacy preflight and `git diff --check`: passed.

An additional six-file periodic/CCM run collected 633 cases and reached
90% progress without a reported failure, but did not finish a long native
integral build. It was interrupted and then terminated when the native
call did not return. This is incomplete validation, not a passing CCM lane.
The attempted files were `test_periodic_ao_bloch_transport.py`,
`test_periodic_orbital_sewing.py`, `test_periodic_mean_field_state.py`,
`test_periodic_k_gdf_ibz_nonsymmorphic.py`,
`test_periodic_aiccm2026dev_b_symmetry.py` and `test_ccm_symmetry.py`.

All commands use the matching `.venv/bin/python -m` interpreter and
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Native source provenance was
checked before numerical work; that complex-Abelian increment changed no
C++ code. The later MANTA provider integration was rebuilt as described below.

Next missing production capabilities, in order:

1. Scalable producer-side shell/factor schedules with orbit-consistent
   screening and cost-based dispatch through the shared Fock interface.
   The bounded reference-tile scheduler below supplies representative and
   stabilizer execution plus native call counters. The streaming alternative
   removes the full retained shell inventory. Q/V bounds can now skip some
   audit reads, but all-member bound queries, retained audits and repeated
   Python schedule generation are not a production planner.
2. Full non-Abelian partner/product/multiplicity and recoupling machinery;
   C3v energy and complete CC residual oracles before generalization.
3. Covariant auxiliary, frozen/occupied/virtual and PAO/PNO/TNO selections,
   momentum-aware periodic contraction schedules, with CCM and chi keeping
   separate source/support/weight admission.
4. Method-specific relaxed gradients and stress, including response, Pulay,
   fitting and finite-support derivatives, checked by displaced/strained
   finite differences. None is supplied by this J/K increment.

## Molecular representative-tile increment

The subsequent experimental `symmetry_quartets` increment implements
complete finite-source admission, real-ERI permutation/spatial orbits,
stabilizers and streamed representative J/K execution over the coordinated
MANTA P0 libint reference provider. The native extension was rebuilt after
integrating that provider. C2v water, full C3v ammonia and nonorthogonal
Cartesian s+d actions are tested. This is full-block AO transport, not
non-Abelian reduced-product resolution. No production defaults, periodic
sources or derivative capabilities are changed.

The following fresh-process measurements use the same immutable source,
zero primitive-screening precision, one requested OpenMP/BLAS thread,
seven timed builds after warmup and a complex Hermitian, noninvariant trial
density (seed 679). `none`, `permutation` and `spatial` all use the same
Python tile executor and a private reusable native session. These times
include tile evaluation, receipt validation, digest checks, transport and
J/K contractions. They do not compare with MANTA's native J/K executor and
do not establish a production or end-to-end SCF speedup. Other development
work was active on the machine; timings are descriptive.

| Case / group / AO count | Native shell calls, ordered / permutation / spatial | Requested primitive quartets, ordered / permutation / spatial | Median build ms, ordered / permutation / spatial | Full-source spatial admission ms |
| --- | --- | --- | --- | --- |
| NH3 STO-3G / C3v / 8 | 1296 / 231 / 81 | 104976 / 18711 / 6561 | 33.754 / 12.775 / 11.171 | 44.30 |
| NH3 cc-pVDZ / C3v / 29 | 50625 / 7260 / 1893 | 1874161 / 306834 / 125343 | 980.517 / 352.266 / 324.284 | 851.35 |
| H2O STO-3G / C2v / 7 | 625 / 120 / 76 | 50625 / 9720 / 6156 | 11.009 / 4.163 / 3.939 | 16.45 |
| One-center Cartesian s+d / C3v / 7 | 16 / 6 / 6 | 16 / 6 / 6 | 0.328 / 0.239 / 0.200 | 6.41 |

Every admission evaluates all ordered shell quartets once. The independent
oracle makes another complete ordered pass, separately counted by the
benchmark. Primitive counts describe requested primitive combinations,
not instrumentation inside libint recurrences. These cases returned no
null tiles. The one-center case saves no additional integral calls from
spatial symmetry; its small timing difference is not a spatial speedup
claim. For the larger ammonia case, charging the entire spatial admission
against the measured saving over permutation-only builds requires about
31 subsequent builds to amortize. The setup cost is not omitted.

| Case | J+K scalar products / block contractions, all modes | Extra spatial transport scalar products | Maximum spatial J / K error | Trial interaction energy error (hartree) |
| --- | --- | --- | --- | --- |
| NH3 STO-3G | 8192 / 2592 | 1800 | 5.33e-15 / 1.78e-15 | 1.81e-15 |
| NH3 cc-pVDZ | 1414562 / 101250 | 713398 | 4.31e-14 / 1.35e-14 | 1.20e-13 |
| H2O STO-3G | 4802 / 1250 | 711 | 3.11e-15 / 3.59e-15 | 3.67e-15 |
| One-center Cartesian s+d | 4802 / 32 | 0 | 4.48e-16 / 3.15e-16 | 6.67e-16 |

Arbitrary-density contraction work remains `2*n_AO**4`, performed in
`2*n_shells**4` blocks. The complex-Abelian block consumer above is the
separate increment that omits forbidden contraction terms. Combining the
two schedules without materializing a complete transformed tensor remains
future work. No scalar orbit multiplicity substitutes for missing density
blocks. Tests also use non-Hermitian densities and independent spin channels,
checking `U^dagger F[U D U^dagger] U = F[D]` with metric-aware AO actions.

| Spatial plan case | Factory logical bytes / completed-plan logical bytes | Action arrays / stabilizer-mask bytes | Native provider modeled bytes | Whole-process peak RSS bytes |
| --- | --- | --- | --- | --- |
| NH3 STO-3G | 11992280 / 10851544 | 728 / 486 | 8806200 | 159039488 |
| NH3 cc-pVDZ | 81872044 / 80102572 | 3920 / 11358 | 25527888 | 210714624 |
| H2O STO-3G | 10720544 / 9980704 | 464 / 304 | 8803256 | 158171136 |
| One-center Cartesian s+d | 10505444 / 9476324 | 1800 / 36 | 8698992 | 158564352 |

Logical bytes conservatively reserve inventories and scratch; they are not
measured allocations or an RSS cap. Array and mask payloads exclude Python
schedule objects. RSS includes imports, the schedule, provider and the
independent dense oracle (32768, 5658248, 19208 and 19208 bytes respectively).
The plan itself retains no complete AO ERI tensor. The largest spatial
source residual is `2.07e-14` in Frobenius norm and the largest stabilizer
residual is `2.78e-15`, against an unchanged `1e-10` source tolerance.
Roundoff remains separate from the exact-arithmetic source residual bound.

Reproduce with `scripts/benchmark_symmetry_quartets.py`; its JSON output
contains separate setup/build/reference counts and each measured quantity.
The focused quartet, complex-Abelian and MANTA reference tests pass **70/70**.
A combined run adding shared groups, AO/SALC/GDF, k symmetry, periodic AO
transport, retained-space sewing, finite mean-field states, citation integrity
and lane classification passes **2325 tests**, with 16 existing spglib
deprecation warnings. That count includes metadata/schema tests, not 2325
chemical systems. It preserves odd/even, fractional-translation, complex,
degenerate-space and incompatible-domain coverage in the existing shared
and periodic foundations. No CCM or chi Hamiltonian is modified or newly
qualified by this run.

New regressions cover native ordered-source J/K oracles, actual call counts,
asymmetric-source and displaced-source rejection, exact unreduced fallback,
radial-source mismatch, ephemeral tile lifetime, reordered non-Abelian
operations, energy density-direction finite differences and refusal of
complex-AO/antiunitary/projective sources. An independent source review found
a stabilizer storage underestimate; compact permutation-bit masks correct it.
High-stabilizer regressions and exact/one-short factory budgets now pass,
with insufficient caps refused before snapshots or native sessions. The
review covered source algebra and resource admission, not independent
physical reproduction. Analytic geometry gradients, stress, CC residuals,
RI/CD and screened periodic sources remain separate acceptance targets.

## Streaming pair-stabilizer increment

`schedule="streaming"` now selects an experimental pair-stabilizer
enumeration behind the same MANTA factory. It retains pair maps and one
aggregate representative digest, rather than every ordered quartet and
every representative hash. Source admission remains complete. It does not
infer producer equivariance from geometry or reduce the admission integral
count. The default `schedule="stored"` and all three reduction modes remain
available. No native MANTA code, periodic Hamiltonian or solver routing was
changed by this increment.

Davidson's double-coset intersection and inverse-pair redundancy now have
an execution consumer. Independent S3 and S4 index models, including
reordered identities, repeated shell orbits and fixed shells, compare the
streaming enumeration with exhaustive quartet partitions. A Burnside
census checks the orbit count separately. An index-only 60/120-shell test
constructs schedules without expanding a quartet and observes quadratic
array scaling; this is not a large-molecule numerical benchmark.

Fresh-process spatial-schedule comparisons use seven repeated builds after
warmup and the same fixtures, source tolerance, density seed and one-thread
settings as above. Both schedules have identical source/build receipts,
residuals and J/K errors. The cc-pVDZ regression also requires bitwise equal
J/K arrays, the same representative order and the same transporters.

| Case | Factory logical bytes, stored / streaming | Streaming pair-array bytes | Whole-process peak RSS bytes, stored / streaming | Admission ms, stored / streaming | Median build ms, stored / streaming |
| --- | --- | --- | --- | --- | --- |
| NH3 STO-3G | 11992280 / 10710212 | 2104 | 160317440 / 159023104 | 48.08 / 52.17 | 16.182 / 15.169 |
| NH3 cc-pVDZ | 81872044 / 29606236 | 10776 | 212140032 / 202194944 | 1753.34 / 2065.91 | 792.113 / 1138.456 |
| H2O STO-3G | 10720544 / 10112352 | 1240 | 160055296 / 160186368 | 50.31 / 46.61 | 11.328 / 13.213 |
| One-center Cartesian s+d | 10505444 / 10539676 | 408 | 161267712 / 159629312 | 14.41 / 14.88 | 0.705 / 0.991 |

These timings were taken under heavier concurrent development load than
the previous table; they must not be used as a regression comparison with
that earlier run or as a general dispatch threshold. Streaming regenerates
topology per build and was slower in three of these four comparisons.
Tiny systems need not save logical memory either. The option addresses
retained schedule memory, not a universal speedup.

A separate `--trace-admission-memory` probe on NH3/cc-pVDZ measured
**8198006 / 80630 bytes** live and **9979706 / 125583 bytes** peak for
stored / streaming admission in Python's allocation tracer. This excludes
opaque native allocations and perturbs timing; the table above uses
untraced runs. The completed streaming plan reserves 27836764 logical bytes,
including 25527888 modeled provider bytes. RSS also includes imports and
the independent complete-source oracle. A smaller logical budget is not
an RSS guarantee.

Integral and contraction work is unchanged between schedules. For
NH3/cc-pVDZ, each audits 50625 ordered shell tiles and 1874161 requested
primitive quartets, then uses 1893 shell calls and 125343 requested primitive
quartets per spatial build. Each build executes 1414562 J+K scalar products,
101250 block contractions and 713398 transport scalar products. Maximum
J/K differences against the independent source remain `4.31e-14 / 1.35e-14`;
the trial interaction energy error is `1.20e-13` hartree. All counters are
measured execution work; the pair-orbit census is not substituted for them.

Current validation: **1586 passed** across quartet, complex-Abelian,
MANTA-reference, citation integrity and lane-classification tests. This
includes **80 quartet cases**, running the source, covariance, Cartesian,
non-Abelian, displaced-source, energy-density finite-difference, fallback,
mutation and budget checks with both schedules. Citation and schema checks
are included in the larger count. Full source audit and exact/one-short
factory refusal remain enforced before snapshots or native sessions.
A separate foundation run passed **797 tests** across shared groups,
AO/SALC/GDF, k symmetry, periodic AO transport, retained-space sewing and
finite mean-field states, with 16 existing spglib deprecation warnings.
The two disjoint runs therefore passed **2383 tests** in total.

Independent source review compared the actual enumeration against 66
separate cyclic, dihedral, A4 and S4 cases covering 53028 ordered members.
Synthetic finite-array tests, including a nonorthogonal angular action,
found maximum J/K Frobenius error `9.33e-16` on arbitrary complex densities.
The review checked late-representative mutation and exact/one-short factory
boundaries, and found no actionable defect. This is source/index/finite-array
review evidence, distinct from the native-provider measurements above.

Remaining production gates are producer-side source/screening admission,
measured cost-based execution policy, and combination with
selection-rule block contractions. The ordinary real-AO pair quotient
does not supply complex integral conjugation, antiunitary corepresentations,
non-Abelian reduced products, periodic image/momentum scheduling, local
projector transport or relaxed derivatives. Existing separate contracts
for those capabilities remain in force.

## Native quartet executor increment

`executor="native"` now moves the four-leg inverse-action transport,
permutation indexing and arbitrary-complex-density contractions into C++.
One call consumes one representative and all its admitted members. The
stored/streaming schedules, P0 integral provider, complete numerical audit,
source identity and representative digest checks are preserved. Native
construction snapshots density and angular matrices; checked tile buffers
are borrowed only during each serial call. NumPy remains the default and
explicit fallback. No screening, state projection, MANTA integral kernel,
SCF routing, periodic support or derivative capability is enabled.

The native core was rebuilt and checked against all 603 native/build inputs
before validation. Fresh-process measurements use seven timed builds after
warmup, one requested OpenMP/BLAS thread, the same source and complex
Hermitian noninvariant density (seed 679), and the unchanged `1e-10`
source tolerance. Native times include allocation/packing, owned density
snapshots, provider calls, receipt/digest checks, angular transforms,
contractions and immutable result export. Complete admission is separate.

| Case / mode / schedule | Shell calls | Median NumPy / native build ms | Native admission ms | Maximum native J / K entry error |
| --- | --- | --- | --- | --- |
| NH3 cc-pVDZ / none / stored | 50625 | 751.786 / 321.368 | 1374.11 | 1.12e-14 / 5.41e-15 |
| NH3 cc-pVDZ / permutation / stored | 7260 | 302.753 / 52.538 | 660.71 | 2.00e-14 / 1.24e-14 |
| NH3 cc-pVDZ / spatial / stored | 1893 | 310.821 / 25.368 | 770.93 | 2.82e-14 / 1.36e-14 |
| NH3 cc-pVDZ / spatial / streaming | 1893 | 366.755 / 78.003 | 797.81 | 2.82e-14 / 1.36e-14 |
| NH3 STO-3G / spatial / stored | 81 | 7.712 / 0.825 | 29.16 | 4.00e-15 / 9.94e-16 |
| H2O STO-3G / spatial / stored | 76 | 4.421 / 0.676 | 19.05 | 1.34e-15 / 1.41e-15 |
| One-center Cartesian s+d / spatial / stored | 6 | 0.168 / 0.100 | 6.36 | 2.78e-16 / 4.45e-16 |

For the larger ammonia spatial case, native execution is about 12.3x
faster with stored topology and 4.7x with streaming topology in this run.
Streaming still spends time regenerating Python orbit objects. These are
tile-adapter comparisons under development load, not a comparison against
MANTA's native ordered J/K builder or an end-to-end SCF speedup. Earlier
timing tables were taken under different load. No timing threshold selects
this path automatically.

All integral receipts and contraction counts match the corresponding NumPy
build. NH3/cc-pVDZ spatial execution requests 125343 primitive quartets,
accumulates 1414562 J+K scalar products in 101250 blocks and performs
713398 transport scalar products. It still audits 50625 ordered shell tiles
at setup. This increment accelerates the executed operations without
claiming another integral or contraction-count reduction. Different
floating-point accumulation order changes roundoff: the largest native
trial interaction energy error in these measurements is `1.74e-13` hartree.
The existing source-residual bound continues to exclude arithmetic roundoff.

| Case / mode / schedule | Logical live plan / combined native execution bytes | Process peak RSS NumPy / native bytes |
| --- | --- | --- |
| NH3 cc-pVDZ / none / stored | 83398312 / 83615480 | 235929600 / 235749376 |
| NH3 cc-pVDZ / permutation / stored | 80102572 / 80319740 | 214679552 / 214548480 |
| NH3 cc-pVDZ / spatial / stored | 80102572 / 80319740 | 211910656 / 212041728 |
| NH3 cc-pVDZ / spatial / streaming | 27836764 / 28053932 | 200654848 / 200228864 |
| NH3 STO-3G / spatial / stored | 10851544 / 10943784 | 160841728 / 161480704 |
| H2O STO-3G / spatial / stored | 9980704 / 10066832 | 160104448 / 160415744 |
| One-center Cartesian s+d / spatial / stored | 9476324 / 9666276 | 159956992 / 161775616 |

The combined reservation includes the live admitted plan plus native input,
owned target/export, packed action and tile-scratch allowances. It is
checked before packing or native construction. RSS includes imports,
provider, plan and the independent complete-source oracle, and is not
bounded by the logical reservation. No complete AO ERI tensor is retained
by the executor.

The corrected native artifact passes **171 focused tests**, including
**126 quartet cases**. Coverage includes native/Python J/K and covariance
on nonsymmetric Hermitian and non-Hermitian densities, metric-aware
Cartesian actions, reordered non-Abelian groups, separate spin calls,
density-energy finite differences, exact unreduced fallback, residual bounds,
source mutations, strict array types/layout/alignment, checked integer
overflow and exact/one-short native plus aggregate resource admission.
An independent unequal-axis four-leg einsum checks every real-ERI
permutation and ownership of captured density/actions.

Another **2258 compatibility tests** pass across shared symmetry, fitting,
k-point symmetry, SALCs, AO/Bloch actions, orbital sewing, mean-field state,
citation integrity and test-lane classification. The two disjoint runs total
**2429 passed**; the compatibility run reports 16 existing spglib deprecation
warnings. This preserves the earlier foundations without expanding the
native executor's admitted domains.

Independent source review identified a binding-error path that could export
earlier partial accumulator contents after malformed input. Dispatched
binding failures now invalidate the accumulator, as native arithmetic/index
failures already did. Five permanent regressions submit a valid partial
contribution followed by dtype, layout, rank, alignment or nonarray failure,
then require both export and further accumulation to refuse. The final
review found no additional issue; its separate index-arithmetic comparison
covered 12 unequal-axis fixtures and 192 member visits with maximum relative
J/K error `6.03e-16`. That review is distinct from compiled runtime testing.

Next production work remains provider-qualified screening and scalable
source admission, native schedule/producer batching to reduce Python
callbacks, and measured dispatch through the shared Fock boundary. Complex
integral tensors, irreducible-product CC contractions, periodic momentum and
local-space transport, gradients and stress retain their separate gates.

## Shared-density batch increment

The next coherent execution increment is `jk_batch`: a shared source traversal
for independent spin or response trial densities. Stored and streaming
schedules preserve the same complete admission, representative digests and
`1e-10` source tolerance. No density is symmetrized, made Hermitian or assumed
to be related to another channel. Each channel has its own J/K and residual
bound; the actual integral and angular work is counted once for the batch.
This follows linearity of the existing contraction and adds no new group
or source admission.

The matching native extension was rebuilt and checked against all 603
native/build inputs. Fresh processes use seven timed builds after warmup,
one requested OpenMP/BLAS thread and seed 679. Each separate/shared pair
uses identical independent complex Hermitian noninvariant densities and
the same unscreened MANTA P0 source. An independent all-ordered-tile tensor
supplies each channel's oracle. Timings include snapshots, allocation,
source/digest checks and immutable result export. They exclude complete
source admission and do not measure MANTA's native J/K builder or SCF.

| Case / executor / schedule / densities | Shell calls separate / shared | Primitive requests separate / shared | Median separate / shared ms | Maximum shared J / K entry error |
| --- | --- | --- | --- | --- |
| NH3 cc-pVDZ / native / stored / 2 | 3786 / 1893 | 250686 / 125343 | 46.110 / 23.934 | 2.82e-14 / 1.43e-14 |
| NH3 cc-pVDZ / native / stored / 8 | 15144 / 1893 | 1002744 / 125343 | 182.216 / 28.370 | 4.09e-14 / 2.81e-14 |
| NH3 cc-pVDZ / native / streaming / 2 | 3786 / 1893 | 250686 / 125343 | 137.347 / 69.245 | 2.82e-14 / 1.43e-14 |
| NH3 cc-pVDZ / NumPy / stored / 2 | 3786 / 1893 | 250686 / 125343 | 577.195 / 323.802 | 4.31e-14 / 1.97e-14 |
| H2O STO-3G / native / stored / 2 | 152 / 76 | 12312 / 6156 | 1.252 / 0.635 | 1.36e-15 / 1.79e-15 |
| One-center Cartesian s+d / native / stored / 2 | 12 / 6 | 12 / 6 | 0.161 / 0.086 | 4.58e-16 / 1.78e-15 |

The two/eight-channel stored native comparisons improve by about 1.93x/6.42x.
Each reports the same maximum J/K errors as its separate-build comparison.
Across native measurements the largest trial interaction-energy error is
`1.74e-13` hartree. Full source admission remains mandatory: ammonia takes
0.728 to 0.807 s in these processes, still auditing 50625 ordered shell tiles
and 1874161 primitive requests. Setup is paid once for either reuse policy.

| Case / densities | J+K scalar products, both policies | Block contractions, both policies | Angular scalar products separate / shared |
| --- | --- | --- | --- |
| NH3 cc-pVDZ / 2 | 2829124 | 202500 | 1426796 / 713398 |
| NH3 cc-pVDZ / 8 | 11316496 | 810000 | 5707184 / 713398 |
| H2O STO-3G / 2 | 9604 | 2500 | 1422 / 711 |
| One-center Cartesian s+d / 2 | 9604 | 64 | 0 / 0 |

| Case / executor / schedule / densities | Live-plan + execution logical bytes separate / shared | Process peak RSS bytes separate / shared |
| --- | --- | --- |
| NH3 cc-pVDZ / native / stored / 2 | 80319740 / 80400604 | 213401600 / 213221376 |
| NH3 cc-pVDZ / native / stored / 8 | 80319740 / 80885788 | 213286912 / 213123072 |
| NH3 cc-pVDZ / native / streaming / 2 | 28053932 / 28134796 | 202194944 / 201654272 |
| NH3 cc-pVDZ / NumPy / stored / 2 | 80102572 / 80329836 | 213286912 / 213483520 |
| H2O STO-3G / native / stored / 2 | 10066832 / 10071664 | 161529856 / 162201600 |
| One-center Cartesian s+d / native / stored / 2 | 9666276 / 9671108 | 160972800 / 161054720 |

The separate execution budget covers one active call, while the shared
budget covers the entire batch. The benchmark additionally owns density
and collected result buffers: 80736 bytes for two ammonia channels,
322944 for eight, and 4704 for the smaller cases. RSS includes these buffers,
imports, source, plan and the independent dense oracle. Logical reservation
and RSS are distinct, and small RSS changes are not claimed as memory
savings. Shared execution reserves `O(batch*n**2)` density/target storage;
it does not retain a full ERI tensor or one angular tile per channel.

The focused quartet, complex-Abelian and MANTA reference run passes
**225 tests**, including **54 new batch cases**. They cover both schedules
and executors, C2v/C3v/metric-aware Cartesian actions, all reduction modes,
non-Hermitian and zero channels, nonsymmetric-density covariance, singleton
shape, independent ordered MANTA integrals, immutable outputs and density
snapshots, exact/one-short caps, invalid/overflowing batch counts and a late
tile/source failure. A failure in one native channel poisons the complete
batch. Broken-spin UHF assembly agrees with the identical unreduced source;
channel-resolved density-energy finite differences and linear J/K response
agree without relaxing existing tolerances. These density derivatives are
not nuclear gradients or stress.

Another **2258 compatibility, citation and lane-classification tests** pass
on this rebuilt core, for **2483 passed** across the two disjoint runs.
There are 16 existing spglib deprecation warnings. The compatibility run
preserves the earlier shared, fitting, multi-k, AO/Bloch, SALC, orbital-sewing
and retained-state foundations; it does not authorize those domains for the
molecular batch executor.

Independent bounded source review found no actionable issue. A separate
finite-array exercise ran 36 cases through the actual NumPy batch function
with synthetic admitted-plan metadata (maximum relative J/K error
`3.864e-16`). Another 36 cases compared an independent transcription of the
native channel/transport indexing with four-leg einsum (`5.006e-16`). These
source/algebra checks cover budget/census arithmetic, ownership, per-channel
bounds, whole-batch refusal and shared receipts; they are separate from the
compiled molecular tests and timing measurements above.

Next production gaps are unchanged: provider-qualified screening/scalable
source admission, native schedule/provider batching and method-owned Fock
dispatch with measured costs. At this increment, batches computed both J
and K for every channel; selective targets are implemented in the subsequent
increment below. Complex integral sources, non-Abelian reduced CC products,
periodic momentum/local-space transport and analytic derivatives require
their own acceptance evidence. NumPy and separate-build fallbacks remain
explicit; no solver default or production-authorization flag changes.

## Selective J/K target increment, October 2026

`jk_batch(..., targets=...)` now accepts one immutable `j`, `k` or `jk`
selector per input density. The default still requests both. NumPy and
native execution skip unrequested contractions and target allocations;
compact immutable outputs carry explicit input-channel mappings. J-only
and K-only batches return an empty array for the absent family, rather than
a matrix of zeros. Complete source admission, residual norms, covariance
conventions and representative digests remain unchanged.

For an unrestricted build, three inputs `[D_alpha+D_beta,D_alpha,D_beta]`
with `('j','k','k')` replace four outputs from the two-spin both-target
batch. This saves one quarter of the actual contraction terms and returned
target bytes. The extra total-density input means total reservation is not
reduced. Method-owned spin factors, Fock construction and energy assembly
remain outside the executor; no production solver is enabled.

The final native artifact was rebuilt and checked against all 603 native/build
inputs. The following fresh-process measurements use 11 timed builds after
warmup, one requested OpenMP/BLAS thread, seed 679, the identical unscreened
MANTA P0 source and unchanged `1e-10` source tolerance. Trial densities are
complex Hermitian and noninvariant. Both-target native requests retain the
original paired J/K loop. Timings include the full repeated-build adapter,
but exclude mandatory admission. These measurements were made under
development load and are not directly comparable with the earlier tables.

| Case / executor / schedule | Request | Scalar products | Block contractions | Returned target bytes | Median build ms |
| --- | --- | --- | --- | --- | --- |
| NH3 cc-pVDZ / native / stored | 8 densities, both | 11316496 | 810000 | 215296 | 43.740 |
| NH3 cc-pVDZ / native / stored | 8 densities, J only | 5658248 | 405000 | 107648 | 53.709 |
| NH3 cc-pVDZ / native / stored | 8 densities, K only | 5658248 | 405000 | 107648 | 52.863 |
| NH3 cc-pVDZ / native / stored | 2 spins, both | 2829124 | 202500 | 53824 | 44.925 |
| NH3 cc-pVDZ / native / stored | Total-J and spin-K | 2121843 | 151875 | 40368 | 48.259 |
| NH3 cc-pVDZ / native / streaming | 2 spins, both | 2829124 | 202500 | 53824 | 130.453 |
| NH3 cc-pVDZ / native / streaming | Total-J and spin-K | 2121843 | 151875 | 40368 | 137.400 |
| NH3 cc-pVDZ / NumPy / stored | 2 spins, both | 2829124 | 202500 | 53824 | 626.022 |
| NH3 cc-pVDZ / NumPy / stored | Total-J and spin-K | 2121843 | 151875 | 40368 | 713.182 |
| H2O STO-3G / native / stored | 2 spins, both | 9604 | 2500 | 3136 | 1.239 |
| H2O STO-3G / native / stored | Total-J and spin-K | 7203 | 1875 | 2352 | 1.246 |
| One-center Cartesian s+d / native / stored | 2 spins, both | 9604 | 64 | 3136 | 0.143 |
| One-center Cartesian s+d / native / stored | Total-J and spin-K | 7203 | 48 | 2352 | 0.194 |

The saved terms and output storage did **not** establish a wall-time gain:
the stored native ammonia spin request is about 7.4% slower in this run,
and its eight-density J-only and K-only requests are about 23% and 21%
slower. Selected-target execution therefore remains explicit. These are
execution/memory capabilities, not an automatic performance policy. Native
selection specialization and stable cost measurements remain prerequisites
for default dispatch. No timing threshold or scientific tolerance was relaxed.

All ammonia requests still make 1893 shell calls, request 125343 primitive
quartets and execute 713398 angular scalar products per build. H2O uses
76 calls, 6156 primitive requests and 711 angular terms; the Cartesian case
uses 6 calls, 6 primitive requests and no angular terms. No additional
integral or angular reduction is claimed. Ammonia admission takes 1.188 to
1.610 s here and still evaluates 50625 ordered tiles and 1874161 primitive
requests. Sampling or weakening that audit is not enabled by target selection.

Against independent all-ordered-tile references, the largest native J/K
entry errors are below `4.09e-14`/`2.81e-14`; the NumPy selected-spin errors
are below `5.18e-14`/`1.97e-14`. The largest measured native unrestricted
interaction-energy error is `2.40e-13` hartree, with NumPy below `2.48e-13`.
J-only and K-only runs do not report a complete interaction energy. Source
error bounds remain indexed by input density and exclude arithmetic roundoff.

| Case / executor / schedule | Live-plan + execution bytes both / selected | Peak process RSS bytes both / selected |
| --- | --- | --- |
| NH3, 8 densities / native / stored, J only | 80885788 / 80670492 | 220512256 / 219152384 |
| NH3, 8 densities / native / stored, K only | 80885788 / 80670492 | 220512256 / 219348992 |
| NH3, spin pair / native / stored | 80400604 / 80400732 | 219594752 / 219119616 |
| NH3, spin pair / native / streaming | 28134796 / 28134924 | 207814656 / 208044032 |
| NH3, spin pair / NumPy / stored | 80329836 / 80329964 | 220561408 / 217989120 |
| H2O, spin pair / native / stored | 10071664 / 10071792 | 167788544 / 167575552 |
| Cartesian s+d, spin pair / native / stored | 9671108 / 9671236 | 167198720 / 167247872 |

One-family ammonia execution saves 215296 logical bytes, including the
omitted targets and their immutable exports. For the selected spin request,
an additional total-density snapshot replaces the omitted output storage;
selector metadata raises the logical total by 128 bytes. RSS also includes
imports, source, plan, the independent dense oracle and benchmark-owned
buffers. The spin benchmark additionally retains its original two-spin
input: 26912 bytes for ammonia and 1568 for the small cases. RSS differences
are not attributed to the logical reservation or claimed as general savings.

The final core passes **259 focused quartet, complex-Abelian and MANTA tests**,
including **34 new target-selection cases**. New coverage includes mixed
channel mappings and nonsymmetric-density covariance for C2v/C3v and
metric-aware Cartesian actions, J-only/K-only empty outputs, exact/one-short
reservations before packing, invalid selectors and native descriptor types,
selector ownership, and whole-result refusal after a late source mutation.
A native test makes the omitted contraction overflow if evaluated: the
selected request succeeds and the both-target request refuses. This checks
actual omission, not just returned shapes or nominal counts. Independent
dense integrals validate broken-spin Focks, unrestricted energies, empty
beta channels, complex transition densities and channel-resolved density
finite differences. These are not nuclear-gradient or stress tests.

Another **2258 compatibility, citation and lane-classification tests** pass,
for **2517 passed** across the two disjoint final-core runs. There are 16
existing spglib deprecation warnings. Earlier shared, fitting, multi-k,
AO/Bloch, SALC, orbital-sewing and retained-state foundations remain covered;
their coverage does not admit those sources to the molecular target executor.

Remaining production gaps are source-qualified screening and scalable
admission, native schedule/provider batching, and measured method-owned
dispatch. Complex integral kernels, antiunitary/projective actions,
non-Abelian reduced CC products, periodic momentum/local-space transport,
gradients and stress still require their separate representation and source
acceptance. The existing both-target, NumPy and unreduced-source fallbacks
remain explicit; no production-authorization flag changes.

## Selected native contractions and Fock-state compatibility, October 9, 2026

Native selected-target execution now resolves the J or K matrix strides once
per member, shares those addresses across density channels, and handles
scalar-shell tiles directly. It retains every finite-arithmetic check and
each output's accumulation order. The paired both-target path, complete source
audit, source digests, tolerances, budgets and public API remain unchanged.
No additional integral, transport or contraction reduction is claimed.

The comparison uses the previous native artifact and the rebuilt artifact
in alternating fresh processes, with three processes per case/artifact and
21 timed builds after warmup. Both artifacts use the same Python adapter,
seed 679, unscreened P0 source and `1e-10` source tolerance. One OpenMP/BLAS
thread is requested. Timings include provider calls, transport, contraction
and result export; mandatory source admission remains separate. Each table
entry is the median of process medians, with their minimum/maximum in brackets.
The benchmark also exports all individual build timings.

| Native case / request | Previous build ms [min, max] | Current build ms [min, max] |
| --- | --- | --- |
| NH3 cc-pVDZ stored, two spins both | 23.323 [23.144, 24.132] | 23.284 [23.130, 24.454] |
| NH3 cc-pVDZ stored, total-J/spin-K | 23.749 [23.498, 23.990] | 23.985 [23.537, 24.376] |
| NH3 cc-pVDZ stored, eight J targets | 25.585 [25.551, 25.590] | 25.091 [25.012, 26.278] |
| NH3 cc-pVDZ stored, eight K targets | 25.685 [25.431, 25.823] | 25.032 [24.911, 25.068] |
| NH3 cc-pVDZ streaming, total-J/spin-K | 69.207 [68.898, 69.774] | 69.187 [69.175, 69.388] |
| H2O STO-3G stored, total-J/spin-K | 0.630 [0.626, 0.634] | 0.635 [0.630, 0.639] |
| Cartesian s+d stored, total-J/spin-K | 0.095 [0.092, 0.129] | 0.092 [0.091, 0.094] |

The eight-density J/K-only medians decrease about 1.9%/2.5%; the J ranges
overlap. Mixed spin requests do not establish a timing improvement. Current
stored selected-spin execution remains about 3.0% slower than both-target
execution despite 25% fewer contractions. These bounded measurements do not
justify automatic dispatch or a universal speedup. NumPy and unreduced-source
fallbacks remain available, and scientific acceptance thresholds are unchanged.

For both artifacts, every ammonia build still evaluates 1893 representative
shell tiles and 125343 primitive requests, with 713398 angular scalar products.
Two-spin both/selected execution performs 2829124/2121843 scalar products and
202500/151875 block contractions. Eight J-only or K-only channels perform
5658248 scalar products and 405000 contractions. Admission still evaluates
50625 ordered tiles and 1874161 primitive requests; it took 0.705 to 0.764 s
across these runs. H2O remains at 76 shell calls, 6156 primitive requests,
711 angular terms, 7203 selected scalar products and 1875 contractions.
Cartesian s+d remains at 6 shell calls/primitive requests, no angular terms,
7203 selected scalar products and 48 contractions.

| Case / request | Logical live-plan + execution bytes, unchanged | Peak RSS bytes, previous / current |
| --- | --- | --- |
| NH3 stored, two spins both | 80400604 | 217300992 / 217235456 |
| NH3 stored, total-J/spin-K | 80400732 | 217055232 / 217219072 |
| NH3 stored, eight J targets | 80670492 | 216940544 / 217038848 |
| NH3 stored, eight K targets | 80670492 | 217104384 / 216907776 |
| NH3 streaming, total-J/spin-K | 28134924 | 205340672 / 205340672 |
| H2O stored, total-J/spin-K | 10071792 | 165363712 / 165265408 |
| Cartesian s+d stored, total-J/spin-K | 9671236 | 165085184 / 165281792 |

RSS is the maximum over each artifact's three fresh processes and includes
the benchmark's independent reference and Python/native runtime. Its small
differences are not attributed to storage reduction. Returned target bytes
are unchanged: ammonia both/selected spin 53824/40368, eight J-only or K-only
107648, and small selected-spin cases 2352. Source identities, receipts,
counts, logical bytes and reference errors agree between compared artifacts.
Maximum native J/K errors remain below `4.09e-14`/`2.81e-14` and the largest
trial unrestricted interaction-energy difference below `2.40e-13` hartree.

The rebuilt core is newer than all 603 native/build inputs. **2539 tests pass**,
including **22 new cases** for unequal shell axes and every ERI permutation,
complex nonsymmetric transition densities, exact selected/both-target results,
owned inputs and whole-result refusal on later-channel overflow. Existing
source rejection, scalar-target omission, energy/density-response, fitting,
Gamma/multi-k, AO/Bloch, sewing and retained-state checks remain covered.
There are 16 existing spglib deprecation warnings. This coverage preserves
the periodic foundations without admitting a periodic source to this executor.

`scripts/check_symmetry_fock_contract.py` also passes **24 optional snapshot
crossings** against explicitly supplied Fock-workstream source files. It reads
only the draft operator/state modules into an isolated namespace and records
their hashes, using this checkout's native core. Water, NH3 and Cartesian s+d,
both schedules/executors and full/empty beta channels agree with independent
all-ordered P0 electronic-energy/Fock references. Maximum Fock/electronic-energy
errors are `3.56e-15`/`7.11e-15`. The diagnostic sets the nuclear component to
zero and claims no converged total energy. Snapshots survive subsequent builds
and input mutation and refuse write enablement. Reported retained state arrays
are 4032 bytes for water/Cartesian and 5248 for NH3, additional to executor
storage; this is not a bounded production memory admission.

The Fock object remains a separate, uncommitted draft with unresolved energy
ownership/identity review findings. This crossing does not accept those
semantics or enable production routing. The shared design now records the
method-owned immutable density/Fock/energy boundary and PBC Parity's physical
AC/BD exchange support, wider K outputs, k weights, AO gauges, zero-mode and
screening constraints. MANTA continues to own kernels, screening and bubble
hierarchies. Next production capabilities remain scalable source-qualified
admission/screening, provider/schedule batching and method-owned dispatch.
Complex integral sources, non-Abelian reduced CC products, periodic/local
space execution, nuclear gradients and stress require separate acceptance.

## Source-qualified whole-orbit screening

The next increment adds `QuartetScreening` and `jk_batch(screening=...)`.
The [literature assessment](symmetry_literature.md) distinguishes measured
finite-source omission norms from rigorous kernel bounds and empirical
QQR/CSA estimates. The [design](design_shared_symmetry.md) specifies mask
closure, correct density/operator variance, identities and the MANTA/Fock/PBC
boundary. The [developer reference](developer_shared_symmetry.md) gives the
API, budgets and explicit fallbacks. Production defaults remain unchanged.

Each stored audit now records full ordered-source bytes in a digest and
complete-orbit source/reconstruction norms. The immutable native P0 producer
and session are required before future omitted reads can be skipped. A fixed
mask contains whole orbits and has its own numerical-source identity;
screen-policy identity and density-specific bounds stay separate. The
existing `1e-10` source tolerance is unchanged. This is a bounded finite
numerical certificate, with norm and contraction roundoff excluded, not a
producer analytic error bound or scalable prescreening implementation.

Measurements below use three fresh processes per case and mask state, 21
timed native builds after warmup per process, alternating masked/unmasked
order, one requested OpenMP/BLAS thread, seed 679 and total-J/alpha-K/beta-K
channels. Both use the same immutable P0 source and the current stored audit.
The explicit tensor omission budget is `1e-12`. Timings include source calls,
transport, contraction, receipt checks and exports; audit and mask creation
are separate. The complete ordered reference is contracted both before and
after applying the identical mask. These are trial-density measurements,
not converged SCF, CC or gradient results.

| Case | Actual representative calls, unmasked / masked | Requested primitive quartets, unmasked / masked | J/K scalar products, unmasked / masked | Block contractions, unmasked / masked | Median build ms, unmasked / masked |
| --- | --- | --- | --- | --- | --- |
| H2O STO-3G, C2v | 76 / 68 | 6156 / 5508 | 7203 / 6267 | 1875 / 1755 | 0.6350 / 0.5774 |
| NH3 cc-pVDZ, C3v | 1893 / 1763 | 125343 / 104454 | 2121843 / 2059815 | 151875 / 149271 | 24.0571 / 22.6035 |
| Cartesian s+d, C3v action | 6 / 6 | 6 / 6 | 7203 / 7203 | 48 / 48 | 0.09337 / 0.09358 |

The three process medians span 0.631-0.645 / 0.574-0.582 ms for water,
23.600-24.411 / 22.111-22.766 ms for ammonia and
0.09317-0.09642 / 0.09358-0.09592 ms for Cartesian s+d. These measurements
show about 9% and 6% lower build medians in water and ammonia; the Cartesian
case omits nothing and has no demonstrated speedup. No automatic dispatch
policy follows from this small set. Ammonia angular scalar products change
from 713398 to 712318. Water's 711 and Cartesian s+d's zero are unchanged.

| Case | Logical live-plan/execution bytes, unmasked / masked | Total process peak RSS range, bytes | Complete audit tiles / primitive requests | Audit ms range | Mask creation ms range |
| --- | --- | --- | --- | --- | --- |
| H2O STO-3G | 10087152 / 10110704 | 167247872-168083456 | 625 / 50625 | 18.2-28.6 | 0.057-0.073 |
| NH3 cc-pVDZ | 81330012 / 81818716 | 219578368-220528640 | 50625 / 1874161 | 973-1086 | 0.887-0.899 |
| Cartesian s+d | 9672004 / 9677636 | 167182336-167542784 | 16 / 16 | 5.08-5.89 | 0.019-0.028 |

The full stored inventory remains live. Masks increase the conservative
reservation; they do not reduce output storage or claim an RSS saving.
Target arrays remain 2352, 40368 and 2352 bytes respectively. RSS includes
imports, provider, plan, densities, targets and independent dense references.
The new per-orbit norms/digest increase admission work relative to the
earlier audit measurements; complete source evaluation counts are unchanged.
This setup cost is never included in a claimed repeated-build saving.

Water omits 40 ordered tiles containing numerical zeros. Ammonia omits 868
ordered tiles and 20676 scalar ERI values, with independently checked omitted
Frobenius norm `4.15022e-16`; Cartesian s+d omits none at this cap. Maximum
J/K differences against the identical masked ordered source are
`8.92e-15 / 1.79e-15`, `3.52e-14 / 1.43e-14` and
`8.96e-16 / 1.78e-15`. The unscreened comparisons stay inside the same
rounded envelopes. The largest trial UHF interaction-energy difference is
`2.43e-13` hartree against the masked reference and `2.42e-13` against the
unscreened reference. Ammonia per-input matrix bounds are approximately
`1.771e-12`, `1.266e-12`, `1.191e-12`; these include retained reconstruction
residual and omission, with arithmetic roundoff still separate.

Regression tests also use deliberately large diagnostic budgets to exercise
nonzero omissions, never as proposed production tolerances. They verify
identical-source integrals/J/K, per-density and method-weighted energy bounds,
broken-spin assembly, non-Hermitian covariance, density-energy finite
differences, selected and absent targets, all-omitted zero-work execution,
immutable receipts, exact/one-short budgets and incompatible-domain refusal.
A three-center Cartesian-d fixture measures an orbit norm of 0.544744,
where a representative norm times square-root multiplicity gives 0.516083.
The latter would wrongly admit a 0.52 tensor budget; the implemented
complete-orbit norm refuses that omission. This is separate from the smaller
Cartesian s+d timing fixture above.

The matching native core was unchanged and newer than all 603 native/build
inputs. Validation with Python 3.14 and NumPy 2.5.3 completed **2560 passed,
9 skipped**, with 16 existing spglib warnings. The focused quartet, Abelian
and MANTA suite contributes **302 passed, 9 skipped**; skips are explicitly
inapplicable streaming-screen cases, and streaming refusal is tested.
The larger run repeats the shared, AO/SALC, fitting, periodic k/Bloch,
orbital-sewing, retained-state, citation and test-manifest files from the
previous increment. It preserves odd/even meshes, fractional translations,
complex actions and degenerate-space evidence without admitting a periodic
screened source. Fitting factors, CC residuals and nuclear derivatives are
unchanged and are not acceptance claims for this mask.
The Sphinx documentation build and staged privacy preflight pass; Sphinx
retains the two pre-existing unrelated cross-reference warnings.

Coordination confirmed that MANTA owns kernel bounds, screening and bubble
hierarchies; its current source/tile seam has no general rigorous omission
certificate. The Fock workstream consumed 24 original and 12 screened
snapshot crossings against its updated energy-owning draft. Maximum screened
Fock/electronic-energy errors are `1.78e-15 / 1.07e-14`. Fixed mask receipts
travel in source provenance; per-density bounds remain separate. This
optional crossing is not a dependency or production memory certificate.
PBC Parity confirmed unchanged physical AC/BD support, wider exchange output
support, metric-domain sensitivity, weights, image gauges and Ewald rules.

### Complete screened molecular SCF acceptance

On 2026-10-11 a complete RHF calculation through `vq` exercised the native
screened quartet executor at every iteration on runtime commit `ed9bfb42`,
which contains the symmetry landing `5857803f`. The case is neutral singlet
NH3, pure cc-pVDZ, 29 AOs and 15 shells. In bohr, N is at `(0,0,0.3)` and
the three H atoms are at `(1.7 cos(t),1.7 sin(t),-0.7)`, with
`t = 0, 2*pi/3, 4*pi/3`. The calculation starts from an orbital-rotated
core guess. The immutable P0 source, `1e-10` source tolerance and explicit
`1e-12` tensor omission budget are unchanged.

Three complete SCF calculations use the screened native executor, an
independently evaluated ordered tensor with the identical mask, and that
ordered tensor without the mask. The latter two are dense correctness
oracles, not timing baselines. All converge in 16 iterations without
averaging the produced J/K or Fock matrices.

| Reference | Total energy, hartree | Absolute difference from screened SCF, hartree |
| --- | --- | --- |
| Screened native quartet SCF | -56.18855210065033 | 0 |
| Identical-mask ordered tensor SCF | -56.18855210065026 | 7.11e-14 |
| Unmasked ordered tensor SCF | -56.18855210065033 | 0 at reported precision |
| Native full-source RHF driver | -56.18855210065040 | 7.11e-14 |
| ORCA 6.1.1 RHF/cc-pVDZ, VeryTightSCF, NoRI | -56.188552099947 | 7.03e-10 |

The existing ORCA calculation was reused; its energy difference passes the
declared `1e-8` hartree comparison threshold. ORCA uses its own integral and
screening conventions, so this chemical reference does not replace the
identical-source checks. Final same-mask density/J/K differences are
`8.78e-14 / 2.13e-14 / 1.15e-14`. Reassessment of the saved S/H/D/J/K arrays
gives ten electrons to `4e-15`, maximum `D S D - 2 D` residual `2.00e-15`,
Fock Hermiticity residual `5.55e-17`, metric-orthogonalized commutator norm
`3.35e-10` and density closure `8.57e-11`. The energy-change, commutator and
closure gates remain `1e-11`, `1e-9` and `1e-10` respectively.

An arbitrary complex non-Hermitian trial density also passes the raw
operator checks: same-mask J/K errors are `1.78e-14 / 6.40e-15`, and the
maximum covariance error over the six C3v operations is `2.98e-14`, using
`D' = U D U^dagger` and `U^dagger F[D'] U = F[D]`. These checks are separate
from the converged state. The independently measured omitted tensor norm is
`5.51e-16`; the retained reconstruction residual is `2.25e-14`.

Each screened SCF build performs 1763 shell calls, 104454 requested primitive
quartets, 1373210 J/K scalar products and 99514 block contractions. The 16
builds total 28208 shell calls and 1671264 primitive requests. Qualification
separately evaluates 50625 ordered tiles and 1874161 primitive quartets.
With one requested OpenMP/BLAS thread, the single-run admission takes
2.608 s, mask creation 1.578 ms, median screened build 45.19 ms and the
screened SCF loop 0.731 s. The entire acceptance job, including all oracles
and trial-density checks, takes 10.84 s and peaks at 313028608 bytes RSS.
The logical execution reservation is 81737738 bytes; the two dense tensor
oracles alone retain 11316496 bytes. These are measured costs, not a new
cross-machine speedup or memory-saving claim.

This accepts the bounded opt-in molecular screened SCF experiment. It does
not enable production dispatch or extend admission to native MANTA kernels,
periodic sources, fitting, CC residuals, nuclear gradients or stress.

Next production capabilities are scalable producer-issued bound/support
certificates, compact streaming mask evidence, native provider/schedule
batching and method-owned acceptance/dispatch. Density-dependent screening,
complex integral sources, RI/CD and retained local spaces, non-Abelian
reduced CC tensors and periodic/BIPOLE/CCM/chi screening require their own
source and representation admission. Analytic gradients and stress remain
separate method targets including response, Pulay, fitting and support/mask
derivatives. The unscreened, NumPy and existing streaming fallbacks remain
available with unchanged scientific tolerances.

## Native molecular HF bridge acceptance

The explicit `symmetry_molecular_hf` consumer now runs the admitted fixed P0
mask through the existing native RHF/UHF drivers and constructs method-owned
`FockBuild` snapshots. This closes the standalone-diagnostic-to-native-host
integration gap. It remains opt-in and experimental; no default runner,
periodic producer, fitting, CC or derivative admission changes.

Fourteen focused regressions passed. They cover C2v water, full C3v ammonia,
nonunitary Cartesian s+d actions, nonsymmetric real densities, NumPy/native
execution, byte admission, complex-input and unsupported-kernel refusal,
owned H/D/frame states, rectangular retained frames, independent spin channels,
and finite differences of the HF energy with respect to density. These are
density-response checks, not nuclear-gradient or stress acceptance.

A complete queued calculation used the candidate Python module with a healthy
already-built native development runtime. Loaded runtime files were hashed
before and after the calculation and stayed unchanged. This is candidate-module
acceptance, not evidence of installed default production dispatch. The same
NH3 geometry, cc-pVDZ basis, source tolerance `1e-10`, fixed omitted tensor
budget `1e-12`, native energy tolerance `1e-11` and gradient tolerance `1e-9`
as the preceding acceptance were retained. The existing ORCA reference was
reused. Two additional UHF/STO-3G calculations used H2 at 4.00 and 4.02 bohr
with opposite atom-localized initial spin densities.

| Calculation | Energy, hartree | Difference from full-source native SCF, hartree | Ordered identical-mask Fock max error |
| --- | --- | --- | --- |
| NH3 RHF/cc-pVDZ | -56.188552100650256 | 7.11e-14 | 3.42e-14 |
| H2 UHF, 4.00 bohr | -0.9358423283198118 | 4.45e-16 | 1.39e-17 |
| H2 UHF, 4.02 bohr | -0.9357442504982110 | 2.23e-16 | 1.12e-16 |

The NH3 ORCA difference is `7.03e-10` hartree, below the unchanged `1e-8`
comparison gate. Re-evaluated native energy, common HF state energy and the
independent identical-mask ordered contraction agree within `5.69e-14`
hartree. H2 converges in five iterations at each geometry, with final
`||Da - Db||_F = 1.39525 / 1.39596` and one electron in each spin to roundoff.
The integral reduction therefore does not force the electronic state to obey
atom-exchange symmetry. No separate ORCA reference was acquired for these
broken-spin finite-source H2 checks.

NH3 qualification costs 50625 ordered shell calls and 1874161 requested
primitive quartets. A terminal fused J/K build uses 1763 shell calls,
104454 primitive requests, 1373210 scalar products, 99514 block contractions
and 712318 angular-transport scalar products. The native driver makes 68
actual J/K calls over 15 iterations, including its additional solver work,
for 119884 shell calls, 7102872 primitive requests, 56988215 scalar products
and 4129831 block contractions. The counts are not inferred from iteration
count. Each H2 native UHF run makes 24 calls, 96 shell calls and 7776 primitive
requests. Its separate terminal three-channel HF-state evaluation shares four
representative calls instead of the native driver's three separate traversals.

With one requested OpenMP/BLAS thread, the NH3 audit takes 8.010 s and native
SCF 9.324 s; the terminal fused build takes 72.96 ms. The two H2 SCFs take
8.60 and 5.27 ms. These single-run integration timings are not isolated
speedup benchmarks and are not compared across machines with earlier results.
Peak RSS for the combined tests and acceptance is 253706240 bytes. NH3's
maximum J/K logical reservation is 81964746 bytes; the evaluated-state array
storage is separately 27144 bytes. For H2 those values are 9193288 and 224
bytes, with 9194568 bytes reserved for the terminal three-channel batch.
Full-audit/stored-inventory cost remains a production scaling limitation.

Next work is compact producer-bound screening and method dispatch with source
and accuracy admission, plus native UHF channel batching and terminal/output
state integration. MANTA owns rigorous bounds, support receipts and bubble
hierarchies. Fock unification owns shared evaluated-state semantics; this
adapter supplies only molecular HF factors and its source receipt. Periodic
Seitz, full image shifts, reciprocal wraps, phases, weights and finite-domain
qualification, as well as distinct CCM/chi sources, retained-space transport,
non-Abelian reduced CC tensors, gradients and stress retain their own gates.

## Compact streamed screening acceptance

The streaming pair-stabilizer plan now accepts an explicit
`maximum_omitted_norm` during its complete source audit and exposes a fixed
`plan.screening` certificate. It retains one packed bit per orbit and one
retained-representative digest, instead of complete quartet member lists,
norms and hashes. The schedule-order first-fit policy can differ from the
stored smallest-first policy; each result is checked against its own exact
masked source. Full actual-member norms preserve the Cartesian metric
boundary. The unchanged molecular `JKBuilder` and HF `FockBuild` adapter
consume either certificate. Default dispatch remains experimental and off.

Twenty-eight focused regressions passed, comprising fourteen new compact
screen tests and the fourteen existing HF bridge tests. New coverage includes
C2v water, C3v ammonia, nonunitary Cartesian s+d and three-center Cartesian d
shells, general complex non-Hermitian density covariance, native and NumPy
execution, all/zero omission, selected channels, HF density finite differences,
resource refusal before reads, mutable-source refusal, foreign audits and
changed retained-tile digests. No final-matrix symmetrization substitutes for
these operator checks.

A queued calculation used candidate Python modules with a healthy already-built
native runtime; all 375 loaded files hashed before and after stayed unchanged.
The NH3 geometry, cc-pVDZ basis, source tolerance `1e-10`, omitted tensor budget
`1e-12`, native energy/gradient tolerances `1e-11 / 1e-9` and ORCA comparison
gate `1e-8` were unchanged. The two H2 UHF/STO-3G cases retain opposite
atom-localized initial spin densities and the 4.00/4.02 bohr separations.
Each case ran reduced, independent ordered identical-mask, and full-source
native SCFs, followed by independent terminal Fock and energy checks.

| Calculation | Energy, hartree | Identical-mask SCF difference, hartree | Full-source SCF difference, hartree | Terminal identical-mask Fock max error |
| --- | --- | --- | --- | --- |
| NH3 RHF/cc-pVDZ | -56.188552100650256 | 4.27e-14 | 7.11e-14 | 3.42e-14 |
| H2 UHF, 4.00 bohr | -0.9358423283198118 | 0 | 4.45e-16 | 1.39e-17 |
| H2 UHF, 4.02 bohr | -0.9357442504982110 | 2.23e-16 | 2.23e-16 | 1.12e-16 |

NH3 again agrees with the existing ORCA reference within `7.03e-10` hartree
and converges in 15 iterations. Both H2 cases converge in five iterations,
retain spin-density differences `1.39525 / 1.39596`, and have one electron
per spin to roundoff. The displaced calculation is an energy check; nuclear
gradients and stress remain unqualified. The H2 reference is the independently
contracted identical source, with no separate external-program parity claim.

NH3's 1893 orbits need 237 mask payload bytes and 10776 retained pair-schedule
array bytes. These exclude object overhead; the full conservative J/K logical
reservation is 28289620 bytes, including source, plan, native accumulator and
bridge buffers. HF state arrays separately occupy 27144 bytes. The mask's
worst-case storage remains quartic in shell count despite its compact encoding.
Whole-job peak RSS, including tests, source oracles and all SCFs, is 282673152
bytes; neither the mask payload nor the logical reservation is an RSS bound.

The audit still evaluates 50625 ordered shell tiles and 1874161 requested
primitive quartets. The mask omits 130 representatives, 868 ordered tiles and
20676 integral values, with omitted norm `4.15e-16` and retained reconstruction
residual `2.06e-14`. Each terminal fused J/K uses 1763 actual shell calls,
104454 primitive requests, 1373210 scalar products, 99514 block contractions
and 712318 angular-transport products. The native NH3 solver's 68 actual
J/K builds total 119884 shell calls, 7102872 primitive requests, 56988215
scalar products and 4129831 block contractions. Each H2 solver uses 24 builds,
96 shell calls, 7776 primitive requests and 384 scalar/block contractions;
its terminal three-channel evaluation shares four representative calls.

With one requested OpenMP/BLAS thread, the single-run NH3 audit takes 2.825 s,
native screened SCF 12.634 s, and terminal J/K 133.0 ms. H2 SCFs take 23.83
and 20.26 ms. These include streaming schedule regeneration and are not an
isolated speedup comparison with the preceding stored-plan measurements.
The change reduces retained schedule metadata; it does not claim faster SCF.

Next production capabilities remain rigorous MANTA-issued bound/support
certificates that avoid a full numerical audit, source/accuracy-qualified
public dispatch, native UHF channel batching and terminal/output state
integration. MANTA owns kernel bounds and BUPO/bubble construction; symmetry
supplies compatible representatives, stabilizers and mask closure. Periodic
Seitz/images/wraps/phases/weights/momentum, distinct finite CCM/chi sources,
RI/CD and local retained spaces, reduced non-Abelian CC tensors, complex
integral kernels/antiunitary actions, gradients and stress keep their separate
admission and acceptance requirements. The unmasked and NumPy fallbacks remain.

## Producer-bound audit screening acceptance

`MolecularCoulombBounds` now binds MANTA's existing native full-space Q/V
product envelopes to the immutable molecular P0 source. It supplies bounds
for complete ordered AO shell tiles without evaluating Coulomb integrals.
The compact streaming scheduler can omit entire orbits before numerical
reads; every retained member and stabilizer still passes the unchanged
equivariance audit. This avoids the full quartet integral audit when bounds
are sufficiently small. It is an opt-in experimental molecular HF capability,
with no default dispatch or new periodic admission.

The adapter uses the native envelope producer, not a second implementation
of MANTA kernels or BUPO. Its receipt records kernel, normalization, source,
precision, pair work and memory. Equal extent-search radii avoid unused search;
the selected radius is not a cutoff. The inequality bounds analytic integrals
with floating-point headroom and excludes an interval proof of arithmetic
roundoff. The source receipt distinguishes retained-orbit qualification and
upper-bound omission from a measured full-source audit. Unmasked reconstruction
from a partial plan refuses; the original provider and complete audit remain
explicit fallbacks.

Thirty-seven focused regressions pass, including nine new envelope/admission
checks and the existing compact-screening and HF-bridge tests. Coverage includes
C2v and C3v molecular sources, nonunitary Cartesian shells, all-ordered integral
bounds, native and NumPy contractions on nonsymmetric complex non-Hermitian
trial densities, raw operator covariance, actual skipped-read instrumentation,
source/basis displacement refusal, immutable ownership, resource refusal and
positive-underflow/overflow behavior. The bridge's existing density-response
finite differences remain passing. These are not nuclear derivative tests.

A queued acceptance calculation used the candidate Python modules with an
already-built healthy native runtime. All 377 loaded files watched before and
after remained unchanged. Source tolerance `1e-10`, fixed omitted tensor
budget `1e-12`, native SCF energy/gradient tolerances `1e-11 / 1e-9` and the
NH3 ORCA comparison gate `1e-8` were unchanged. Each case ran reduced,
independent ordered identical-mask and full-source native SCFs, then terminal
Fock/energy checks. H4 contains two H2 fragments with 1.4 bohr bonds and
30 bohr between centers; its admitted group is D2h. H2 UHF retains the broken
spin seeds from preceding acceptance.

| Calculation | Energy, hartree | Identical-mask SCF difference | Full-source SCF difference | Terminal identical-mask Fock max error |
| --- | --- | --- | --- | --- |
| NH3 RHF/cc-pVDZ | -56.18855210065033 | 2.85e-14 | 0 | 5.13e-14 |
| H4 separated RHF/STO-3G | -2.2334286410805104 | 8.89e-16 | 8.89e-16 | 4.45e-16 |
| H2 UHF, 4.00 bohr | -0.9358423283198118 | 0 | 4.45e-16 | 1.39e-17 |
| H2 UHF, 4.02 bohr | -0.9357442504982110 | 2.23e-16 | 2.23e-16 | 1.12e-16 |

NH3 converges in 15 iterations and differs from the existing ORCA reference
by `7.03e-10` hartree. H4 converges in two iterations. Both H2 cases converge
in five, retain `||Da-Db||_F = 1.39525 / 1.39596` and one electron per spin
to roundoff. H4 and broken-spin H2 use the independent identical-source oracle;
no separate external-program parity is claimed for them. The displaced H2
calculation tests source-dependent energies, not gradients or stress.

| Actual setup work | NH3 | Separated H4 |
| --- | --- | --- |
| Native bound shell pairs / primitive pairs | 120 / 771 | 10 / 90 |
| Envelope evaluations / integral calls for bounds | 240 / 0 | 20 / 0 |
| Ordered-member bound queries | 50625 | 256 |
| Audit shell calls / full ordered count | 50625 / 50625 | 64 / 256 |
| Audit primitive quartet requests | 1874161 | 5184 |
| Omitted representatives / ordered tiles | 0 / 0 | 11 / 192 |
| Measured omitted tensor norm / reported upper bound | 0 / 0 | 1.17e-34 / 2.01e-33 |
| Bound arrays / packed mask bytes | 2040 / 237 | 192 / 3 |
| Retained pair-schedule array bytes | 10776 | 1312 |
| Conservative J/K logical reservation, bytes | 62621129 | 26680564 |

The H4 audit avoids 75% of actual ordered integral calls. Its terminal fused
J/K uses eight representative calls, 648 primitive requests and 128 scalar
products/block contractions, compared with 256 calls, 20736 primitive requests
and 512 products/blocks in the independent ordered full-source contraction.
The native screened SCF makes 19 actual J/K calls totaling 152 shell calls,
12312 primitive requests and 1344 products/blocks. Bound visitation is still
quartic; neither subquartic planning nor an SCF speedup is inferred.

NH3 bounds omit nothing at the unchanged cap, so its terminal J/K retains all
1893 representatives, with 125343 primitive requests, 1414562 scalar products,
101250 block contractions and 713398 angular-transport products. Its 68 solver
calls total 128724 shell calls, 8523324 primitive requests, 58704323 scalar
products and 4201875 block contractions. This policy can retain more work than
the preceding measured-norm mask; bound-only omission is deliberately explicit.
Each H2 solver uses 24 builds, 96 shell calls, 7776 primitive requests and
384 scalar products/blocks. Bounds omit no H2 orbit.

With one requested OpenMP/BLAS thread, bound construction takes 0.529 ms for
NH3 and 0.152 ms for H4. Their audits take 2.891 s and 12.24 ms, native reduced
SCFs 13.904 s and 59.12 ms, and terminal J/K 191.68 ms and 1.320 ms.
H2 SCFs take 5.57 and 6.78 ms. These are single-run integration timings,
including Python scheduling, not isolated speedup measurements. Whole-job
peak RSS including tests and independent source oracles is 300482560 bytes.
Logical reservations conservatively retain producer/bound construction costs
and are not RSS guarantees; state array storage is separately 27144 bytes
for NH3, 544 for H4 and 224 per H2 case.

Next capabilities are tighter producer bounds and hierarchical screening,
producer-issued equivariance/support attestations, measured cost-aware public
dispatch, native UHF channel batching and terminal/output state integration.
MANTA owns envelope/kernel/bubble construction; symmetry owns whole-orbit
closure and qualification; the shared FockBuild contract is unchanged.
Periodic finite domains, Seitz/image/wrap/phase/weight/momentum data, distinct
CCM/chi Hamiltonians, auxiliary rank cuts, local retained spaces, non-Abelian
reduced CC tensors, complex kernels, gradients and stress keep their existing
separate gates. No scientific tolerance or fallback was relaxed.

## Native hierarchical screening acceptance

The optional `MolecularHierarchicalBounds` adapter consumes MANTA's existing
native bubble traversal and order-zero error model. A node magnitude takes
the minimum of Q/V and a monopole magnitude plus all compact, penetration
and full-moment-tail errors. Ordered AO dimensions and pair orientations
weight each disjoint candidate block. Only complete quartet orbits contained
in accepted blocks are omitted. A generic native tree need not be symmetric;
it supplies bounds, not an operator-equivariance certificate.

Two queued acceptances pass 36 distinct focused regressions: thirteen new
hierarchy tests, nine existing producer-bound tests and fourteen molecular
HF-bridge tests. Separated H8 accepts internal-node blocks covering multiple
pair interactions, and deliberately tight H4 caps produce partially covered
orbits that must remain retained. These two additional checks and the complete
H8 SCF run use a pinned source/native snapshot.
Independent ordered integrals check both the hierarchy's
own bound and the Q/V minimum, including C2v, C3v and nonunitary Cartesian
sources. Native/NumPy arbitrary complex non-Hermitian density contractions,
raw operator covariance, actual omitted-read instrumentation, foreign and
displaced sources, changed caps, resource refusal and immutable ownership
are covered. The existing HF density-response tests remain passing.

Five complete reduced, independently contracted identical-mask and full-source
native SCFs pass with unchanged source/omission tolerances `1e-10 / 1e-12`,
native SCF energy/gradient tolerances `1e-11 / 1e-9` and ORCA gate `1e-8`.
The candidate numerical module bytes match those executed on the healthy
existing native runtime; all 378 watched loaded files stayed unchanged.

| Case | Energy, hartree | Identical-mask / full-source SCF differences, hartree | Terminal Fock max error |
| --- | --- | --- | --- |
| NH3 RHF/cc-pVDZ | -56.18855210065033 | 2.85e-14 / 0 | 5.13e-14 |
| H4 separated RHF/STO-3G | -2.2334286410805104 | 8.89e-16 / 8.89e-16 | 4.45e-16 |
| H2 UHF, 4.00 bohr | -0.9358423283198118 | 0 / 4.45e-16 | 1.39e-17 |
| H2 UHF, 4.02 bohr | -0.9357442504982110 | 2.23e-16 / 2.23e-16 | 1.12e-16 |
| H8 separated RHF/STO-3G | -4.466857260869869 | 1.78e-15 / 0 | 6.67e-16 |

NH3 agrees with the reused matching ORCA reference within `7.03e-10` hartree.
H2 retains the broken spin states and one electron per spin. H4, H8 and H2
comparisons use independent contractions of the same finite source, without
external-program parity. The displaced H2 result is an energy check, not
nuclear-gradient or stress acceptance.

| Measured setup work | NH3 | Separated H4 |
| --- | --- | --- |
| Native tree nodes / depth | 199 / 5 | 15 / 3 |
| Envelope evaluations / integral calls for envelopes | 6734 / 0 | 350 / 0 |
| Native node visits / bound queries | 12462 / 7260 | 79 / 69 |
| Far-error calls / tightened node bounds | 0 / 0 | 14 / 14 |
| Accepted candidate blocks / ordered AO entries | 0 / 0 | 34 / 192 |
| Orbit-member coverage requests / ancestor-map probes | 1893 / 59178 | 200 / 300 |
| Actual audit calls / full ordered count | 50625 / 50625 | 64 / 256 |
| Audit primitive quartet requests | 1874161 | 5184 |
| Actual omitted norm / upper bound | 0 / 0 | 1.17e-34 / 2.01e-33 |
| Q/V/size arrays / candidate-block payload, bytes | 2040 / 0 | 192 / 816 |
| Ancestor index payload / fixed mask, bytes | 5424 / 237 | 256 / 3 |
| Conservative maximum J/K logical reservation, bytes | 71041033 | 26815268 |

H4 still skips 192 ordered audit calls and eleven representatives; this case
does not skip more integrals than the preceding Q/V-only policy. The hierarchy
performs fewer native node-bound queries than flat ordered-member bounds,
but also incurs extent searches, traversal and coverage lookups. These are
separate counts, not a claimed net work or wall-time speedup. NH3 has no far
candidates or omissions and illustrates the need for a dispatch cost gate.

Terminal H4 J/K uses eight calls, 648 primitive requests and 128 scalar
products/block contractions; its 19 native solver calls total 152 shell calls,
12312 primitive requests and 1344 products/blocks. NH3 terminal J/K retains
1893 calls, 125343 primitive requests, 1414562 scalar products, 101250 blocks
and 713398 angular-transport products. Its 68 native calls total 128724 shell
calls, 8523324 primitive requests, 58704323 scalar products and 4201875 blocks.
Each H2 solver uses 24 builds, 96 shell calls and 7776 primitive requests.

With one requested OpenMP/BLAS thread, NH3/H4 bound construction takes
18.21 / 0.843 ms, audits 1.828 s / 12.55 ms, native reduced SCFs 8.829 s /
25.41 ms and terminal J/K 99.02 / 1.307 ms. H2 SCFs take 6.07 and 6.69 ms.
Whole-job peak RSS is 302579712 bytes. Payload counts exclude object overhead;
logical admission reserves worst-case candidate storage and native construction
workspace. These single-run measurements are not isolated comparisons with
prior runs. Worst-case storage remains quadratic in pair count; orbit
iteration and retained-source audits remain explicit costs.

The H8 extension consists of four H2 fragments with centers at
`(x,y,0)`, `x,y = +/-15` bohr, and hydrogen offsets `z = +/-0.7` bohr.
Its 351 accepted blocks cover 588 unordered pair interactions, including
107 blocks containing an internal node. Thus the hierarchy compresses actual
candidate interactions, rather than only representing a theoretical orbit
count. Closure permits omission of 3840 of 4096 ordered audit calls and
90 representatives. The actual omitted tensor norm is `2.56e-34`, below its
`6.28e-33` recorded bound and the unchanged `1e-12` cap.

| Additional H8 measured work/storage | Value |
| --- | --- |
| Tree nodes / depth; envelope evaluations / integral calls | 58 / 4; 732 / 0 |
| Native node visits / bound queries; far calls / tightened bounds | 667 / 500; 250 / 81 |
| Member coverage requests / ancestor-map probes | 3856 / 15069 |
| Retained audit calls / primitive quartet requests | 256 / 20736 |
| Terminal J/K calls / primitive requests / products / blocks | 16 / 1296 / 512 / 512 |
| SCF builds / shell calls / primitive requests / products / blocks | 28 / 448 / 36288 / 7680 / 7680 |
| Q/V/size arrays / candidate blocks / ancestor payload, bytes | 640 / 8424 / 1296 |
| Mask / schedule arrays / state arrays, bytes | 14 / 4096 / 2112 |
| Conservative maximum J/K logical reservation, bytes | 28081834 |

With one requested thread, H8 bounds/audit/reduced SCF/terminal J/K take
0.0514 / 3.507 / 29.121 / 0.538 seconds; whole-job peak RSS is 239697920
bytes. The two focused checks take 46.93 seconds in this queued run versus
0.52 seconds in their separate focused invocation, so these loaded-run
timings do not establish a speedup or a scaling comparison. The candidate
implementation and test bytes match the landed source, and all 378 watched
loaded files remain unchanged during this run. The H8 reference is the same
finite source with independent ordered contractions, not an external code.

The shared FockBuild and native molecular HF interfaces are unchanged.
Production defaults stay off. Further work includes measured cost-aware
routing, tighter source-specific bounds and efficient orbit/hierarchy joins,
producer equivariance/support attestations and native channel/output-state
integration. Periodic finite domains and Seitz/image/wrap/phase/weight/momentum
handling, distinct CCM/chi sources, auxiliary rank cuts, local projectors,
complex integral kernels, reduced non-Abelian CC products, gradients and
stress retain their separate qualification requirements.

## Joint native UHF symmetry builds

`JKBuilder.build_jk_uhf(Da, Db)` returns independent physical
`(J[Da + Db], K[Da], K[Db])` matrices. Native NOITER, ordinary iterations and
the terminal rebuild use this additive boundary. Its default executes the
original incremental slots in order: J0, K-alpha0, K-beta1. The opt-in
`ScreenedMolecularJK` implementation shares one retained representative read
and angular transport across the three selected channels. Neither density
is projected, and equality of spin densities is not required.

The new focused checks exercise arbitrary nonsymmetric independent densities
with C2v, C3v and metric-aware Cartesian actions in both NumPy and native
executors. They compare raw J/K against ordered identical-mask contractions
and check covariance. Other checks preserve incremental slot results bit for
bit across resets and threshold changes, cover NOITER/capped/converged UHF,
and reject malformed inputs or insufficient three-channel resources before
integral work. Direct incremental UHF, terminal F/D energy consistency and
DF closed-shell/doublet regressions cover the inherited sequential fallback.

Thirty distinct regressions pass across focused queued runs on the rebuilt
native core. Five complete UHF calculations compare the joint and sequential
builds, an independently contracted identical mask and the full P0 source.
All 3457 recorded source/test files and the loaded core remain unchanged
through the final run; 377 loaded module files are recorded. Source/omission
caps remain `1e-10 / 1e-12`, SCF energy/gradient tolerances `1e-11 / 1e-9`
and the reused ORCA gate `1e-8`.

| UHF case | Energy, hartree | Identical-mask / full-source energy difference | Terminal Fock max error |
| --- | --- | --- | --- |
| NH3/cc-pVDZ | -56.18855210065030 | 2.85e-14 / 4.27e-14 | 1.98e-14 |
| Separated H4/STO-3G | -2.2334286410805104 | 0 / 8.89e-16 | 4.45e-16 |
| H2/STO-3G, 4.00 bohr | -0.9358423283198118 | 0 / 4.45e-16 | 1.39e-17 |
| H2/STO-3G, 4.02 bohr | -0.9357442504982110 | 2.23e-16 / 2.23e-16 | 1.12e-16 |
| H doublet/STO-3G | -0.46658184955727544 | 0 / 0 | 0 |

Joint and sequential builds return identical energies for all five cases.
NH3 retains equal spin densities and matches the existing ORCA closed-shell
reference within `7.03e-10` hartree. Both H2 cases retain broken spin densities
with norms of `Da - Db` above 1.39. H4 starts from broken spin seeds but
converges to equal spin densities; its 192 omitted ordered tiles remain bounded
by `2.01e-33`, with actual omitted norm `1.17e-34`. The H doublet has zero beta
density and agrees with its one-electron generalized eigenvalue within
`1e-11` hartree. H4/H2 use same-source numerical references, not an external
program. The displaced H2 energy is not a gradient or stress acceptance.

| Whole native solver work, joint / sequential | NH3 | H4 | Each H2 | H doublet |
| --- | --- | --- | --- | --- |
| Joint invocations / other builds in joint path | 16 / 45 | 8 / 18 | 6 / 6 | 3 / 0 |
| Shell calls | 115473 / 176049 | 208 / 336 | 48 / 96 | 3 / 9 |
| Primitive quartet requests | 7645923 / 11656899 | 16848 / 27216 | 3888 / 7776 | 243 / 729 |
| Angular transport products | 43517278 / 66346014 | 0 / 0 | 0 / 0 | 0 / 0 |
| Density scalar products, equal in both paths | 65777133 | 2688 | 384 | 9 |
| Block contractions, equal in both paths | 4708125 | 2688 | 384 | 9 |
| Maximum logical reservation, bytes | 71202761 / 70987209 | 26818596 / 26814244 | 26366892 / 26365612 | 26313524 / 26313012 |

One terminal NH3 joint build evaluates 1893 shells and 125343 primitive
quartets, with 713398 angular transport products, 2121843 density products
and 151875 blocks. The corresponding sequential calls require exactly three
times its integral and transport work, with equal density contractions.
NH3/H4 mask, schedule and evaluated-state arrays occupy respectively
`237 / 10776 / 40832` and `3 / 1312 / 832` bytes. Whole-job peak RSS is
263979008 bytes; logical reservations are conservative buffer admissions,
not measured peak allocation or complete runtime RSS.

With one requested thread, joint/sequential SCFs take 81.506 / 109.461 seconds
for NH3, 0.376 / 1.106 for H4, 0.160 / 0.0124 and 0.103 / 0.0121 for the two
H2 cases, and 0.00444 / 0.00326 for H. NH3 bound construction/audit take
0.979 / 29.540 seconds; H4 takes 0.000676 / 0.314 seconds. These shared-host
observations include scheduling variation and show no consistent wall-time
speedup. The measured savings are actual integral/transport counts, with
slightly higher joint buffer reservations; cost-aware routing remains needed.

Post-convergence stability analysis and stateless Hessian/response builds
retain their existing separate calls. Joint builds therefore save two
representative traversals each; total solver work also includes those other
calls and must be measured separately. Density contraction terms remain
unchanged. UKS, periodic drivers and native output-state migration are outside
this increment. Source masks, scientific tolerances, default dispatch and the
shared FockBuild contract remain unchanged.

This completes the joint native UHF consumer increment. Next production work
is measured cost-aware routing, efficient orbit/hierarchy joins and stronger
producer source/support attestations, followed by native terminal/output-state
integration with FockBuild. Further response batching requires its own cache
and source contract. MANTA retains ownership of integral kernels and bounds;
its separate finite periodic remainder API does not qualify this molecular
adapter for periodic sources. Complex-source SCF, reduced non-Abelian CC
products, auxiliary/local retained spaces, gradients and stress still require
their separate method admissions.

## Orbit-local hierarchical coverage reuse

The bounded candidate audit now reuses coverage results for identical
unordered real-ERI shell-pair labels within one orbit. It does not equate
spatially related members or alter MANTA's tree, envelopes or candidate blocks.
Every ordered member still contributes to the norm in the original order.
The 128-entry cache is discarded after each orbit; exhausted capacity falls
back to actual queries. Empty candidate sets require no ancestor-map probes.
A conservative 36864-byte reservation covers cache construction and objects.

Nineteen focused regressions pass on a healthy native runtime with the exact
changed Python modules. They cover independent ordered integrals, C2v/C3v and
metric-aware Cartesian actions, arbitrary complex nonsymmetric density J/K
and raw covariance, partial-orbit refusal, displaced/foreign sources, resource
refusal, and exact cached/uncached bounds, masks, source identities and audit
work. Multi-pair leaves and cache capacities zero, one and two preserve the
same results. Instrumented coverage calls agree with request-minus-hit counts.

Three complete native UHF cases pass against independently contracted
identical-mask and full-source references. Source/omission tolerances remain
`1e-10 / 1e-12`, SCF energy/gradient tolerances `1e-11 / 1e-9`, and the ORCA
gate `1e-8`. Cached and uncached qualification produce identical source
identities, masks, omitted norms, retained integral work and member counts.

| Case | Energy, hartree | Identical-mask / full-source energy difference | Terminal Fock max error |
| --- | --- | --- | --- |
| NH3/cc-pVDZ | -56.18855210065030 | 2.85e-14 / 4.27e-14 | 1.98e-14 |
| Separated H8/STO-3G | -4.466857260869869 | 1.78e-15 / 0 | 5.56e-16 |
| Broken-spin H2/STO-3G, 4.00 bohr | -0.9358423283198118 | 0 / 4.45e-16 | 1.39e-17 |

NH3 retains equal spin densities and agrees with the matching reused ORCA
reference within `7.03e-10` hartree. H8 and H2 use same-source numerical
references, without external-code parity. No gradient/stress claim follows.

| Measured admission work | NH3 | H8 | H2 |
| --- | --- | --- | --- |
| Ordered member requests, unchanged | 1893 | 3856 | 4 |
| Actual coverage calls / cache hits | 1893 / 0 | 604 / 3252 | 4 / 0 |
| Ancestor-map probes, cached / original | 0 / 59178 | 2218 / 15069 | 0 / 16 |
| Peak cache entries | 1 | 8 | 1 |
| Candidate blocks / native bound queries, unchanged | 0 / 7260 | 351 / 500 | 0 / 6 |
| Retained audit shell calls / primitive requests, unchanged | 50625 / 1874161 | 256 / 20736 | 16 / 1296 |
| Native UHF shell calls / primitive requests | 115473 / 7645923 | 384 / 31104 | 48 / 3888 |
| UHF density products / block contractions | 65777133 / 4708125 | 7680 / 7680 | 384 / 384 |
| Maximum J/K logical reservation, bytes | 71239625 | 28131242 | 26403756 |

H8 avoids 3252 actual coverage calls and 12851 ancestor-map probes. The
cache does not reduce integral calls or density contractions beyond the
already admitted mask. Cache-key construction and dictionary operations add
work; the probe reduction is not a claim that all setup work falls equally.
No new persistent quartet inventory or leaf-pair expansion is retained.

With one requested thread, bound construction takes 0.01760 / 0.00294 /
0.000212 seconds for NH3/H8/H2. Cached versus original audits take 1.985 /
1.416, 0.02564 / 0.02885 and 0.001307 / 0.001136 seconds. Reduced SCFs take
5.576 / 0.1181 / 0.002421 seconds. Whole-job peak RSS is 311820288 bytes.
These single-run timings show no general wall-time speedup. Logical memory
reservations increase by 36864 bytes; the cache occupancy is reported separately
from total RSS. All 678 recorded runtime/candidate files remain unchanged
through acceptance, with 409 loaded modules recorded.

This completes bounded reuse in the existing molecular coverage audit.
Default dispatch remains off. Next production work is cost-aware source
routing and producer-issued equivariance/support attestations, followed by
terminal/output-state integration with the shared FockBuild. MANTA still owns
integral/bound generation and bubble traversal. Its periodic finite/infinite
source work, multi-k phase/domain contracts, distinct CCM/chi sources,
auxiliary/local projector transport, complex-source SCF, reduced non-Abelian
CC products, gradients and stress retain separate method acceptance targets.

## Same-source cost routing and producer qualification

The explicit molecular bridge now admits `ordered` and `auto` execution in
addition to the existing native and NumPy choices. The ordered route evaluates
every retained ordered tile from the same immutable MANTA P0 session, without
angular transport or final symmetrization. Both routes preserve the exact
mask, source/policy identity, selected density channels and error budget.
Qualification combines the exact native constructor contract, per-tile
provider/kernel/range checks and the existing complete or retained-only audit.
It is not a new algebraic producer attestation or permission to skip retained
audits. Default method entry points remain unchanged.

Calibration probes are nonsymmetric and alternate the two execution orders.
Five native HF channel signatures have separate bounded decisions. The
fastest ordered minus slowest reduced sample must repay complete calibration
cost over the caller's expected remaining builds; otherwise ordered execution
remains selected. Timing noise can change the choice but cannot change the
source. Only scalar diagnostics are cached, with a 131072-byte reservation.
Probe work never counts as physical SCF builds or enters FockBuild provenance.

Numerical implementation `3c80e0aa` passes 36 focused regression cases.
A subsequent receipt-only addition labels each probe executor explicitly;
its focused calibration functionality check passes without changing algebra. Queued acceptance
first exercises both routes and full unmasked references, then reuses those
accepted arrays for the bounded multi-signature fix, with ten routing
regressions and three complete UHF calculations. This preserves the independent
ordered-source and external-reference evidence without repeating those solves.
The native extension is unchanged: runtime source `203393d4`, native SHA-256
`851b221a8b2fb1e13e55159448c39576d2dead6438d5586e183be6af5a2c0116`.
Seven candidate Python modules load explicitly; 684 recorded files remain
unchanged through the final calculation, with 410 loaded modules recorded.

| Case | Final routed energy, Eh | Maximum energy difference between routes, Eh | Full-source energy difference, Eh | Final Fock maximum error, Eh |
| --- | --- | --- | --- | --- |
| NH3 cc-pVDZ, C3v | -56.18855210065030 | 5.69e-14 | 4.27e-14 | 1.98e-14 |
| Four separated H2 fragments, STO-3G | -4.466857260869869 | 1.78e-15 | 0 | 5.56e-16 |
| H2 at 4 bohr, broken-spin STO-3G | -0.9358423283198127 | 1.78e-15 | 1.34e-15 | 2.23e-16 |

NH3 agrees with the matching ORCA 6.1.1 UHF/cc-pVDZ VeryTightSCF NoRI reference
`-56.188552099947 Eh` within `7.033e-10 Eh`, below the unchanged `1e-8` gate.
H8 and H2 use independent same-source numerical references; no new external
parity for those cases is claimed. Source tolerance remains `1e-10`, omission
cap `1e-12`, and SCF energy/gradient convergence `1e-11`/`1e-9`. Broken-spin
H2 remains broken. C2v/C3v, Cartesian shells, complex nonsymmetric J/K,
retained-frame physical Focks, unsupported kernels/images, budget refusal and
calibration disagreement are covered. No gradient or stress is accepted.

| Actual physical SCF work | NH3 | H8 | H2 |
| --- | --- | --- | --- |
| Ordered shell calls | 3088125 | 6144 | 192 |
| Routed shell calls | 115473 | 384 | 120 |
| Ordered primitive requests | 114323821 | 497664 | 15552 |
| Routed primitive requests | 7645923 | 31104 | 9720 |
| Density scalar products, both routes | 65777133 | 7680 | 384 |
| Block contractions, both routes | 4708125 | 7680 | 384 |
| Routed angular scalar products | 43517278 | 0 | 0 |
| Routed maximum logical bytes | 71370697 | 28262314 | 26533548 |

Ordered execution has zero angular transport. Symmetry saves integral
work here, not density scalar products. The final route selects native for
NH3/H8 joint UHF, J and K. H2's joint batch stays ordered because the timing
intervals overlap; its standalone J/K select native. Unmeasured signatures
remain ordered and require their own calibration to claim a benefit.

| Setup and calibration, separate from SCF | NH3 | H8 | H2 |
| --- | --- | --- | --- |
| Source/bound setup seconds | 1.0881 | 0.02742 | 0.001606 |
| Retained audit shell calls | 50625 | 256 | 16 |
| Calibration seconds, three signatures | 2.7424 | 0.06589 | 0.003293 |
| Calibration shell calls / primitive requests | 315108 / 11997024 | 1632 / 132192 | 120 / 9720 |
| Calibration density products / blocks | 14145620 / 1012500 | 5120 / 5120 | 320 / 320 |
| Calibration angular products | 4280388 | 0 | 0 |
| Calibration maximum logical bytes | 71759177 | 28352426 | 26601900 |
| Routed SCF seconds | 5.1572 | 0.11256 | 0.003634 |
| Earlier identical-mask ordered SCF seconds | 30.0159 | 0.17784 | 0.002900 |

One thread was requested. Final whole-job peak RSS is 237027328 bytes.
Timing rows come from successive acceptance runs, not a controlled speedup
benchmark. H8's calibration consumes its observed saving and H2 is too small
to amortize calibration in this run. Expected reuse is an estimate, not a
performance guarantee. The initial one-signature implementation passed
numerically but routed standalone calls poorly: NH3 took 21.63 seconds and
2308413 shell calls. The bounded signature cache resolves that measured
problem; explicit native execution remains available without calibration.

Remaining production capabilities are pre-audit cost admission, reusable
producer-issued operation/support attestations, and native terminal
FockBuild capture without an additional build. MANTA owns immutable kernel/domain descriptors, bounds
and bubble hierarchies. No molecular cost decision admits periodic BIPOLE,
MANTA finite/infinite domains, CCM or chi; their supports, weights, image
shifts, Seitz products, reciprocal wraps, Bloch phases and momentum remain
separate. Complex-source SCF, reduced non-Abelian CC products, covariant
auxiliary/local projectors, analytic gradients and stress also remain separate.

## Public molecular HF method and output acceptance

The explicit `symmetry_reduction=MolecularSymmetryOptions(...)` public RHF/UHF
entry is available for qualified all-electron real molecular P0 sources.
It remains opt-in and experimental. It checks the actual molecule/basis and
one-electron operator, qualifies the fixed screened source, and uses the
existing native solver. Execution selection changes neither the Hamiltonian
nor the source identities. Analytic gradients explicitly refuse this result.

Seventeen focused regressions passed through the calculation queue on source
`4ab94c28`, using native runtime `203393d4` with explicitly loaded and hashed
candidate Python modules and the shared QVF orbital-label change. All 692
expected files stayed unchanged; 394 loaded modules were recorded. Tests
cover native and ordered RHF/UHF at convergence, one iteration and NOITER,
fractional READ densities, rotated degenerate retained coordinates, metric
leakage refusal, incompatible sources, one-electron covariance rejection and
actual QVF writing. Density-defining natural orbitals preserve the evaluated
state without a state-symmetry assumption or next-step extrapolation.

Five complete public-method calculations also passed. Source tolerance
`1e-10`, omission cap `1e-12`, SCF energy `1e-11` and gradient-norm `1e-9`
thresholds are unchanged. Each calculation was compared with an independent
unmasked native P0 SCF and explicit J/K contractions of the complete AO ERI
tensor. The hierarchy omitted no source elements in the compact examples;
H8 omitted 3840 of 4096 ordered quartets with bound `6.273e-33`.

| Case / method | Energy (hartree) | Energy difference from full source | Maximum Fock element difference from independent ERIs | Density reconstruction error |
| --- | ---: | ---: | ---: | ---: |
| C3v NH3 cc-pVDZ / UHF | -56.18855210065031 | 2.843e-14 | 2.243e-14 | 6.912e-15 |
| Four separated H2 fragments, STO-3G / RHF | -4.466857260869866 | 1.777e-15 | 6.662e-16 | 3.775e-15 |
| Broken-spin H2, 4 bohr, STO-3G / UHF | -0.9358423283198123 | 0 | 2.165e-15 | 2.221e-16 |
| NH3 with N displaced 0.02 bohr along z, cc-pVDZ / RHF | -56.18712146664339 | 2.843e-14 | 6.040e-14 | 2.343e-14 |
| He with Cartesian s+d test shells / RHF | -2.300337779324880 | 0 | 8.882e-16 | 1.555e-15 |

The same-geometry NH3 ORCA 6.1.1 UHF/cc-pVDZ VeryTightSCF NoRI reference
is `-56.188552099947` hartree. The difference `7.0331e-10` passes the existing
`1e-8` gate. Other cases use the independent same-source checks above; no new
external reference, CC residual, analytic gradient or stress acceptance is
claimed. The displacement tests a newly qualified source, not a derivative.
Maximum converged density difference from the unmasked runs is `4.838e-14`.

The following are actual integral and contraction counts, including native
stability requests in the SCF totals. NH3 used measured automatic routing;
the other cases selected native execution explicitly. Terminal costs are
additional to SCF, audit and calibration; no theoretical orbit count is
substituted for work performed.

| Work / storage | NH3 | H8 | Broken H2 | Displaced NH3 | Cartesian He |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source-audit shell calls | 50625 | 256 | 16 | 50625 | 16 |
| SCF shell calls | 115473 | 448 | 48 | 128724 | 150 |
| SCF primitive quartet requests | 7645923 | 36288 | 3888 | 8523324 | 150 |
| SCF density scalar products | 65777133 | 7680 | 384 | 58704323 | 72030 |
| SCF block contractions | 4708125 | 7680 | 384 | 4201875 | 480 |
| Additional terminal shell calls | 1893 | 16 | 4 | 1893 | 6 |
| Additional terminal density scalar products | 2121843 | 512 | 48 | 1414562 | 4802 |
| Maximum J/K logical bytes | 71370697 | 28118698 | 26403756 | 71077897 | 26950975 |
| Additional method reservation bytes | 926720 | 131072 | 69632 | 926720 | 115712 |
| Owned FockBuild state bytes | 40832 | 2112 | 224 | 27144 | 1624 |
| Source setup seconds | 0.98868 | 0.01935 | 0.001152 | 0.99239 | 0.005103 |
| SCF J/K seconds | 4.32545 | 0.09749 | 0.001511 | 4.69549 | 0.006321 |
| Additional terminal J/K seconds | 0.07428 | 0.003478 | 0.000132 | 0.06870 | 0.000248 |
| Total public method seconds | 7.72271 | 0.12213 | 0.003789 | 5.76480 | 0.01297 |

NH3 calibration separately used 315108 shell calls, 14145620 density scalar
products and 2.32266 seconds; all three host signatures selected native
representatives. Earlier identical-mask ordered SCF evidence records 3088125
shell calls for the same NH3 61-build workload. Contraction products are not
reduced by this representative-tile schedule; integral generation is.
One-electron qualification additionally builds S/T/V once each and performs
585336, 16384, 64, 585336 and 8232 scalar products, respectively. Maximum
overlap and Hcore covariance residuals are `3.576e-15` and `2.240e-14`.
One thread was requested; whole-job peak RSS was 278495232 bytes, including
the independent references and QVF serialization. Logical workspace counts
are not a process-memory cap, and these timings are not a scaling benchmark.

The method consumes Fock unification's existing frame/source/build/run_fields
interfaces and method-declared QVF orbital labels. MANTA's separately landed
periodic exchange bounds certify fixed-mesh evaluated-density errors, not
symmetry; they are not admitted by this molecular route. Periodic source
support, Seitz composition, full integer image shifts, reciprocal wraps,
Bloch phases, k weights, momentum and distinct CCM/chi Hamiltonians retain
their existing boundaries. Next production work is job-deck admission,
pre-audit cost selection, producer-issued operation/support qualification and
direct native terminal-state capture. RI/CD, complex-source SCF, non-Abelian
reduced CC tensors, local/auxiliary subspaces and method-specific derivatives
remain separate capabilities.

## Prepared-source reuse acceptance

Explicit preparation now removes repeated qualification work for independent
HF calculations on the identical all-electron molecular source. It owns the
actual immutable producer, audited mask and checked one-electron matrices;
no source hash alone authorizes reuse. Full nuclei/order, centers, shell
representation and radial content are checked before tile reads. Charge,
spin, density and retained-space choices remain method-owned. One process-local
lease protects the native workspace, and each method receives fresh physical
counters and state. Only the five bounded scalar cost profiles may persist.

Source `a9d4b1e8` passed 22 regressions and six complete HF calculations through
the calculation queue, using the same identified native runtime `203393d4`
with explicit candidate Python modules. All 695 expected files stayed stable;
394 loaded modules were recorded. The tests include neutral RHF followed by
charged UHF and another fresh RHF state using native, ordered and automatic
execution. Changed geometry, nuclei, same-size radial content and basis size
refuse before reads. Overlapping use, cross-process reuse and pickling refuse;
an invalid READ state releases the lease for the next valid calculation.

Each of NH3 cc-pVDZ UHF, separated H8 STO-3G RHF and broken-spin H2 at 4 bohr
STO-3G UHF was run twice from explicitly supplied initial conditions. The
second H2 calculation swaps the alpha/beta READ densities. Complete independent
AO ERIs again check physical J/K and energies; the prior accepted full-source
SCF arrays and ORCA reference are reused after exact geometry/S/H matching.
All six results match the prior energy, density and Fock arrays exactly,
with the expected spin-channel swap for H2. Maximum independent ERI Fock and
energy differences are `2.243e-14` and `4.264e-14`. NH3 retains the ORCA energy
difference `7.0331e-10` hartree under the unchanged `1e-8` gate. The source,
omission and SCF tolerances are unchanged. QVF source and preparation
identities are checked on all six written artifacts.

| Actual work / storage | NH3 | H8 | Broken H2 |
| --- | ---: | ---: | ---: |
| Once-paid source audit shell calls | 50625 | 256 | 16 |
| First-call calibration shell calls | 315108 | 0 | 0 |
| Second-call audit / calibration shell calls | 0 / 0 | 0 / 0 | 0 / 0 |
| Physical SCF shell calls, each calculation | 115473 | 448 | 48 |
| Physical SCF primitive quartet requests, each | 7645923 | 36288 | 3888 |
| Physical SCF density scalar products, each | 65777133 | 7680 | 384 |
| Physical SCF block contractions, each | 4708125 | 7680 | 384 |
| Additional terminal shell calls, each | 1893 | 16 | 4 |
| Additional terminal density scalar products, each | 2121843 | 512 | 48 |
| Persistent S/H bytes | 13456 | 1024 | 64 |
| Geometry/basis signature reservation bytes | 70144 | 57344 | 23552 |
| Maximum J/K logical bytes | 71370697 | 28118698 | 26403756 |
| Preparation seconds | 1.69886 | 0.03265 | 0.002221 |
| First-call calibration seconds | 4.26287 | 0 | 0 |
| First / second SCF J/K seconds | 8.27719 / 8.90774 | 0.15994 / 0.16159 | 0.002510 / 0.002520 |
| First / second total method seconds | 12.69907 / 9.06118 | 0.16826 / 0.16964 | 0.004310 / 0.004192 |

Relative to preparing and calibrating independently twice, reuse avoids
365733 NH3 shell calls, 256 H8 calls and 16 H2 calls. Preparation makes three one-electron calls, one each for S/T/V, and
either prepared method call makes zero. NH3's first calibration performs 14145620 density scalar products;
its second call reuses all three host signatures and performs no probes.
The independent-state regressions fill all five allowed signatures without
keeping densities or adding an unbounded cache. The whole-job peak RSS is
293666816 bytes, including independent references and QVF serialization.
One thread was requested. These wall times reflect current load and are not
a controlled cross-run speedup benchmark; skipped calls provide direct work
evidence. Physical SCF and terminal work are unchanged.

A fresh geometry/basis/policy still needs a new source audit. There is no new
producer-issued algebraic attestation, periodic source admission or derivative
acceptance. MANTA owns the missing producer operation/support contract;
pre-audit cost admission for unseen sources, public job-deck selection and
native terminal capture remain production increments. This explicit reuse
interface provides a place to consume stronger producer evidence without
weakening the current audit or confusing source and electronic state.

## Historical reduction milestones

The following inventory records the original implementation milestones as of
2026-05-23. It is retained to explain module names and legacy APIs. It is not a
current all-method validation matrix, and its checks do not override the
limitations below. In particular, force symmetrization is not an analytic
gradient implementation, and non-symmorphic transport support does not qualify
every finite integral domain or reconstruction path.

| # | Milestone | Status |
|---|---|---|
| M1 | `symmetry=True` primitive reduction | ✅ |
| M2 | 1e compute reduction (S, T) | ✅ |
| M2a | Wire M2 into 5 drivers | ✅ |
| M3 | `build_jk_gamma_molecular_limit_explicit` | ✅ |
| M3b | `build_jk_2e_real_space_explicit` | ✅ |
| M3b | `build_jk_pair_contributions` (C++) | ✅ |
| M3b | `compute_jk_gamma_reduced` (Python) | ✅, origin-fixed cells only |
| M4 | IBZ-native SCF (4 drivers) | ✅ |
| M5 | `symmetry_scf.py` | ✅ |
| M5a | `symmetry_stabilize` in GDF | ✅ |
| SYM5 | `symmetrize_forces()` | ✅ |
| - | Non-symmorphic support | ✅, diamond Fd-3m validated |
| SYM6 | Character tables (Oh, Td, D4h, C6v) | ✅ |

## Legacy reduction limitations

1. **M3b: origin-fixed cells only.** `compute_jk_gamma_reduced` works for single-atom
   primitive cells (Mg, He) but not multi-atom cells (NaCl), per-atom lattice
   shifts break cell-pair Wigner-D reconstruction. The historical single-atom
   checks do not establish a general multi-atom or multi-k reduction.

2. **M2a: S/T template workaround.** The GDF driver computes full S/T as a template
   when using the symmetry-reduced path, because pybind11's LatticeMatrixSet has
   read-only nbf/cells/blocks. The ERI compute reduction is unaffected.

## Module map

| Module | Purpose |
|---|---|
| `symmetry_shared.py`, native `symmetry_shared` | Shared group, space, orbit and qualification contracts |
| `symmetry_abelian.py` | Experimental complex Abelian selection and finite-ERI block J/K |
| `symmetry_quartets.py` | Experimental source-audited molecular representative-tile execution |
| `symmetry_quartets_screening.py` | Fixed complete-orbit masks and finite-source omission receipts for stored or compact streamed P0 batches |
| `symmetry_quartets_bounds.py` | Source-bound adapter to native MANTA Q/V envelopes, whole-orbit omission before reads and retained-only audit admission |
| `symmetry_quartets_hierarchy.py` | Native bubble/error-model candidates intersected with complete quartet orbits before retained-source auditing |
| `symmetry_quartets_ordered.py` | Same-mask ordered fallback and descriptive immutable-producer qualification evidence |
| `symmetry_molecular_scf.py` | Explicit qualified molecular RHF/UHF entry and evaluated FockBuild/QVF state |
| `symmetry_core.py` | Real Wigner rotations, including stable near-pole extraction |
| `symmetry_ao.py` | Compact pure/Cartesian molecular actions and legacy dense pure-shell matrices |
| native `periodic_ao_bloch_transport`, `periodic_orbital_sewing` | Periodic AO/Bloch transport and state/retained-space audits |
| `symmetry_lattice.py` | Lattice-cell orbits (SYM2b) |
| `symmetry_lattice_c.py` | Atom-pair orbits (SYM2c) |
| `symmetry_integrals.py` | Storage-reduced 1e integrals (SYM3a) |
| `symmetry_integrals_reduced.py` | Compute-reduced 1e integrals (SYM3b) |
| `symmetry_fock_reduced.py` | Compute-reduced 2e Fock (M3b) |
| `symmetry_scf.py` | Group-averaging + force symmetrization (M5, SYM5) |
| `symmetry_salc.py` | Character tables + irrep projection (SYM6) |
| `periodic_runner.py` | `symmetry`/`reduce_to_primitive`/`symmetry_reduce_fock` kwargs |
| `periodic_rhf_gdf.py` | GDF driver with M2a/M5a/M3b integration |
| `pbc_bipole*.py` | BIPOLE drivers with M2a/M4 integration |

## Native legacy entry points

| File | Addition |
|---|---|
| `lattice_integrals.cpp` | `compute_1e_lattice_matrix_explicit` + wrappers |
| `periodic_fock.cpp` | `build_jk_gamma_molecular_limit_explicit`, `build_jk_2e_real_space_explicit`, `build_jk_pair_contributions` |
| `bindings.cpp` | pybind11 for all above |

## Qualified molecular jobs and executable source requests

Source `60936dd9` connects the existing cost-aware fixed-source RHF/UHF entry
to `run_job`. It adds no new integral formula, source certificate or symmetry
representation. The literature-to-code gap here was consumer integration:
Davidson/Dupuis-King representative execution and MANTA-owned bounds were
already available, but the normal job could not select them or preserve their
source request. The new job interface retains exact source guards, physical
FockBuild output and a fresh-qualification QVF recipe. Producer-owned
operation/support attestations and cost admission before an unseen-source
audit remain missing; none of the recent molecular or periodic literature
justifies inferring those from a geometry group or a serialized identity.

The queue acceptance ran 130 focused regressions and four complete HF jobs
using candidate Python modules and identified native runtime `203393d4`.
All 707 expected files stayed stable and 426 loaded modules were recorded.
The tests exercise real RHF/UHF output, prepared reuse, strict operation and
budget records, fresh QVF execution, unsupported method refusal, and a stopped
calculation that cannot enter the legacy SAD retry. Existing canonical orbital
and cube tests also pass. Natural-orbital tables now report actual occupations
and physical-Fock Rayleigh quotients without false canonical frontier gaps.

The larger jobs were NH3 cc-pVDZ UHF through both `run_job` and an executable
QVF, separated H8 STO-3G RHF through a prepared source, and broken-spin H2 at
4 bohr STO-3G UHF from same-basis READ arrays. The direct jobs reproduce the
previous accepted full-source D/F/E arrays exactly. NH3 is
`-56.18855210065031` Eh; the retained ORCA 6.1.1 UHF/cc-pVDZ VeryTightSCF NoRI
reference is `-56.188552099947` Eh, an error of `7.0331e-10` Eh below the
unchanged `1e-8` gate. Independent complete AO ERIs give Fock and energy
errors no larger than `2.243e-14` and `4.264e-14` Eh respectively. The source
tolerance `1e-10`, omission cap `1e-12`, energy convergence `1e-11` and gradient
norm `1e-9` were unchanged.

The QVF Angstrom-to-bohr round trip changes a coordinate by `2.22e-16` bohr.
It therefore correctly produces a fresh source identity instead of reusing
an old audit. A separate queued reference reconstructs that exact source,
confirms its prepared identity, and contracts full AO ERIs with the returned
densities: Fock error `1.577e-14`, energy error `1.422e-14` Eh. Relative to the
original geometry the exported D/F differences are at most `1.99e-14` and
`3.56e-14`; energy is identical. The tiny change affects native stability
work, so the two jobs do not claim identical build counts.

| Calculation | Physical J/K builds | SCF shell calls | Primitive quartets requested | Density scalar products | Block contractions | SCF seconds | Complete job seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NH3 direct job | 61 | 115473 | 7645923 | 65777133 | 4708125 | 12.5425 | 23.5831 |
| NH3 QVF execution | 67 | 126831 | 8397981 | 70020819 | 5011875 | 12.6568 | 23.0585 |
| H8 prepared job | 28 | 448 | 36288 | 7680 | 7680 | 0.27468 | 1.08543 |
| H2 broken-spin job | 12 | 48 | 3888 | 384 | 384 | 0.01203 | 0.30662 |

Each NH3 additionally pays 50625 audit calls, 315108 calibration calls and
14145620 calibration scalar products. Setup takes 3.24660/3.19407 seconds;
calibration takes 6.89984/6.39517 seconds. All three measured signatures select
native reconstruction and agree with their nonsymmetric ordered probes within
`2.101e-14`. The corresponding ordered SCF traversals would request
3088125/3391875 shell tiles with the same density contraction counts. Terminal
capture separately costs 1893 calls and 2121843 scalar products per NH3 job.
This integration adds no further integral or contraction reduction beyond the
already accepted execution path.

H8 preparation pays 256 audit calls once (0.05598 seconds); its job pays none.
H2 pays 16 audit calls. Their terminal captures cost 16/4 calls and 512/48
scalar products. Maximum logical J/K reservations are 71370697 bytes (NH3),
28118698 (H8) and 26403756 (H2); NH3 calibration reserves up to 71759177 bytes.
Owned output states occupy 40832/2112/224 bytes. Whole-job peak RSS is
475365376 bytes including tests, full reference ERIs and normal output
analysis. These observations are not a controlled timing speedup or a hard
SCF memory limit. Default populations, localization, Molden and QVF exports
are validated but their work is not counted as symmetry savings.

This remains an explicit experimental all-electron molecular P0 route.
MANTA owns producer bounds, bubble hierarchies and future attestations; the
job parser cannot issue them. The incoming cutoff-selected periodic BUPO
source work is independent and supplies no molecular or periodic symmetry
permission. Fock unification's source/frame/state fields are reused unchanged.
PBC/CCM/chi sources, complex-source SCF, non-Abelian reduced CC products,
local/auxiliary/frozen projectors and method-specific gradients/stress retain
their separate production acceptance targets.


## Exact terminal evaluation reuse acceptance

Source `0804b885` removes the redundant terminal quartet traversal when the
native method has already evaluated exactly the returned density. This is
reuse of an actual fixed-source evaluation, not a new group-theory theorem,
a geometry/state-symmetry assumption or a source audit bypass. Full UHF
channels preserve total-density J and alpha/beta K order. The native RHF
convergence path calls J and K separately; their exact source/density bytes
can be joined before the method assembles its FockBuild. Standalone response
requests do not authorize a different state. Single-bit density changes,
spin permutations, absent/reset captures and failed replacements cannot
reuse that evaluation; the existing counted cold path remains available.

The method owns at most one full-channel snapshot and one pending J request.
Immutable byte owners isolate them from native input changes and returned
writable J/K matrices. Prepared sources still own no density, J/K or Fock
history. An additional `4096 + 128*n_AO**2` byte reservation is admitted before
new calibration probes. Only immutable scalar route receipts can return to
the prepared source. The reused terminal receipt reports zero new integral,
contraction and transport work and names its original method-local build
numbers in `reused_from_builds`.

The queued acceptance ran 101 regressions and four complete calculations on
candidate Python source and identified native runtime `203393d4`. All 708
expected files stayed stable; 418 loaded modules were recorded. Regression
coverage includes exact FockBuild identity against cold evaluation, native
and ordered execution, separate RHF J/K, nonsymmetric inputs, Cartesian
shells, retained spaces, caps/NOITER and fractional READ states, spin ordering,
owned buffers, changed-bit/signed-zero misses, reset/failure fallback, and
resource refusal before new cost probes. All scientific tolerances remain
unchanged. Ordinary job/QVF source and output checks also pass.

The complete jobs repeat NH3 cc-pVDZ UHF via normal job and executable QVF,
prepared separated H8 STO-3G RHF, and broken-spin H2 at 4 bohr STO-3G UHF.
Energies and direct-job D/F arrays match the preceding full-source acceptance
exactly. NH3 remains `-56.18855210065031` Eh, `7.0331e-10` Eh from the existing
ORCA 6.1.1 UHF/cc-pVDZ VeryTightSCF NoRI reference, below the unchanged `1e-8`
gate. Independent complete AO ERIs give maximum Fock/energy errors of
`2.243e-14`/`4.264e-14` Eh. The QVF check uses its actual reloaded geometry:
its full-source Fock/energy errors are `1.577e-14`/`1.422e-14` Eh. Physical
error bounds, source identities and spin-broken densities are preserved.

| Calculation | Original builds reused | Avoided terminal shell calls | Avoided primitive requests | Avoided density scalar products | Retained capture bytes | Capture seconds | Complete job seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NH3 direct job | 16 | 1893 | 125343 | 2121843 | 40368 | 0.000575 | 15.4488 |
| NH3 QVF execution | 16 | 1893 | 125343 | 2121843 | 40368 | 0.000688 | 17.6643 |
| H8 prepared job | 3 and 4 | 16 | 1296 | 512 | 2560 | 0.000929 | 0.82694 |
| H2 broken-spin job | 6 | 4 | 324 | 48 | 192 | 0.000138 | 0.22879 |

Every terminal receipt contains zero additional shell/primitive calls,
density scalar products, block contractions and angular transport products.
The avoided block counts are 151875 per NH3, 512 for H8 and 48 for H2; each
NH3 also avoids 713398 angular transport scalar products. The original
physical SCF work stays 115473/126831 shell calls and 65777133/70020819 density
products for the two NH3 jobs, 448/7680 for H8 and 48/384 for H2. The work is
reused, not removed from the original SCF counters. The capture copies
80736 values in 16 snapshots per NH3, 1600 values in H8 and 144 in H2.
H8's retained bytes include its 1024-byte pending response J capture.

Logical J/K reservations increase by 111744 bytes for NH3, 12288 for H8 and
4608 for H2, reaching 71482441/28130986/26408364 bytes respectively. Whole-job
peak RSS is 426328064 bytes including tests, complete reference ERIs and
normal output. NH3 still pays 50625 audit and 315108 calibration calls per
fresh source; setup takes 2.34352/2.22801 seconds and calibration
4.28323/5.04610 seconds. SCF times are 8.36332/9.88929 seconds for NH3,
0.22127 for H8 and 0.00977 for H2. These timings come from a different load
than the prior acceptance and do not establish a controlled wall-clock
speedup. Avoided source work is established by zero-work receipts and tests
that reject any additional quartet execution on an exact match.

The bridge remains explicit and experimental. Its bounded capture closes
the redundant terminal-build gap in these admitted RHF/UHF paths without
changing native solver loops or FockBuild ownership. Zero-copy native capture
could reduce copying further. MANTA producer-issued operation/support and
roundoff attestations, cost admission before fresh audits, complex/fitted/
periodic sources, reduced non-Abelian CC products, covariant local/frozen/
auxiliary spaces and method-specific derivative acceptance remain open.


## Ordered-source qualification and short-use routing acceptance

The next increment separates a producer-qualified fixed mask from permission
to reconstruct its angular images. Hierarchical `ordered` preparations read
zero integral tiles. Auto requests with one or two expected builds also skip
paired calibration and choose that admission before the audit. Longer-use
auto requests retain the existing measured routes and retained-orbit audit.
All reduced entry points reject an ordered-only certificate. The actual
producer/mask identity is independent of audit and cost evidence; integral
and molecular Fock source hashes use version 2. Stored historical identities
are not rewritten. Geometry, numerical-operator, electronic-state and
retained-subspace symmetry remain separate.

Numerical source commit `e2e3384d` passed 154 regressions and four complete
queued HF calculations using an explicitly overlaid Python candidate and
the previously healthy native runtime. Exact source/native hashes, loaded
module inventory, queue identity, outputs and arrays are retained outside the
product checkout. All 711 expected files and the loaded runtime were checked
before and after execution. The native molecular kernels were unchanged by
this increment; no new periodic or derivative acceptance is inferred.
Tests include Abelian water, non-Abelian C3v ammonia, metric-aware Cartesian
actions, complex nonsymmetric densities, exact ordered/reduced mask and
Fock-source identity, producer/kernel refusal, resource admission, impossible
probe repayment and attempts to invoke reduced executors without an audit.

Complete calculations used NH3/cc-pVDZ UHF through both `run_job` and executable
QVF, separated H8/STO-3G RHF with a prepared source, and broken-spin H2/STO-3G
UHF at 4 bohr from the existing READ state. Source tolerance `1e-10`, omission
cap `1e-12`, energy convergence `1e-11` and SCF gradient norm `1e-9` are
unchanged. NH3 energies are -56.188552100650256 and -56.18855210065030 hartree.
The existing ORCA 6.1.1 UHF/cc-pVDZ VeryTightSCF NoRI reference is
-56.188552099947 hartree; differences are at most 7.0330e-10, below the
unchanged 1e-8 acceptance limit. Independent full AO-ERI contractions of each
actual final density, including the QVF's reloaded geometry, differ by at
most 3.2863e-14 in Fock entries and 2.8422e-14 hartree in energy. Maximum
density difference from previous accepted results is 1.2144e-13. H8 and H2
use the independent full-integral reference; no external comparison is added
for those cases.

All four preparations performed zero audit tile/shell/primitive calls, and
all short-use route receipts contain zero probe samples and zero probe work.
Each NH3 preparation avoided the previous 50,625 audit calls and 315,108
calibration calls. H8 and H2 avoided 256 and 16 audit calls. Source setup
times were 0.25157/0.31331 seconds for NH3, 0.06560 for H8 and 0.00231 for H2.
These are measured times under current load, not a controlled speedup ratio.
Native envelope/hierarchy work remains explicitly reported in producer
receipts and is not counted as zero setup work.

| Case | Physical J/K builds | Shell calls | Primitive requests | Density products | Block contractions | Maximum logical J/K bytes | SCF seconds / complete job seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NH3 job | 61 | 3,088,125 | 114,323,821 | 65,777,133 | 4,708,125 | 71,465,881 | 96.0524 / 97.3653 |
| NH3 QVF | 67 | 3,391,875 | 125,568,787 | 70,020,819 | 5,011,875 | 71,465,881 | 98.8094 / 100.1143 |
| H8 prepared job | 28 | 7,168 | 580,608 | 7,680 | 7,680 | 28,115,754 | 0.4701 / 1.4329 |
| H2 broken-spin job | 12 | 192 | 15,552 | 384 | 384 | 26,406,284 | 0.0140 / 0.3040 |

All angular-transport counts are zero, and exact terminal reuse adds zero
integral/contraction calls. H8 omits 3,840 of 4,096 ordered tiles per build
with an aggregate bound of 6.2726e-33; actual physical requests fall from
114,688 for the unmasked ordered source to 7,168. Logical reservations remain
conservative; whole-batch measured peak RSS is 428,834,816 bytes, including
tests, full-integral references and output work.

The deliberately short NH3 estimate is not its actual converged workload:
ordered execution makes more physical integral calls than the previously
accepted reduced route and is slower overall. This increment removes
unnecessary setup for explicitly ordered/short-use requests; it does not
establish a new default or promise an overall speedup. A better estimate must
request a newly audited handle. A numerical source identity or bound-only
receipt cannot upgrade ordered qualification into angular equivariance.

Supported execution remains experimental real molecular P0 all-electron
RHF/UHF with explicit fixed masks. Producer operation/support/roundoff
attestation for cheap reduced-source admission is still missing. MANTA owns
that extension and its kernels, envelopes and bubble hierarchy. FockBuild
continues to separate source, frame and evaluated state. Periodic finite
supports, Seitz/image/wrap/phase and momentum conditions, distinct CCM/chi
Hamiltonians, complex kernels and separate antiunitary actions, non-Abelian
reduced products/recoupling, covariant local/auxiliary spaces, CC residuals,
gradients and stress retain their separate acceptance requirements.

## Source-qualified molecular RI block execution

Numerical source `cfee8c49` passed 89 focused regressions and four complete
queued RI-HF cases using a candidate Python overlay and unchanged healthy
native runtime. All 709 expected files and the 396 loaded module files were
recorded; the inputs/runtime hashes were unchanged after execution. Queue,
host, full runtime identity, outputs and factor/density arrays are retained
outside the product tree. This acceptance adds an explicit developer API;
normal method/job defaults do not change.

The source uses native MANTA metric factorization with its lower-triangle
convention and unchanged absolute/relative cut. It evaluates all ordered
three-center tiles once, then retains only allowed complex-Abelian factors.
Complete auxiliary spaces may mix through unitary matrices; split partner
spaces, noninvariant metrics, non-Abelian character tables, antiunitary
transport and an artificially perturbed finite coupling on symmetric
geometry are rejected. Cartesian and pure-shell actions, nonsymmetric complex
trial densities, rank cuts, immutable source snapshots, resource refusals,
HF density finite differences and the existing P0 Fock bridge are covered.
No metric projector or electronic density is repaired by symmetrization.

The complete calculations use water/STO-3G RHF, NH3/cc-pVDZ UHF restricted to
the C3 Abelian subgroup, H2/STO-3G UHF at 4 bohr with independently localized
spin seeds, and water with symmetric x displacements of 0.001 bohr. Auxiliary
and orbital bases are deliberately identical in these finite-source tests;
this is not an accuracy qualification of those small fitting bases. Source
tolerance remains 1e-10, SCF energy tolerance 1e-11 and gradient norm 1e-9.
Broken-spin density survives. The displaced case validates requalification,
not an analytic gradient or finite-difference derivative implementation.

| Case | RI-HF energy / hartree | Native full RI energy difference | Same-factor dense SCF difference | Independent libint Fock maximum error |
| --- | ---: | ---: | ---: | ---: |
| Water | -74.89503754718201 | 0 | 2.8422e-14 | 1.5877e-14 |
| NH3 | -56.45804593400345 | 1.4211e-14 | 7.1055e-14 | 2.8422e-14 |
| Broken-spin H2 | -0.9530462725814317 | 0 | 0 | 2.2205e-16 |
| Displaced water | -74.89507584326651 | 5.6844e-14 | 0 | 1.0659e-14 |

The independent reference recomputes metric and three-center integrals with
libint and independently diagonalizes the metric using the identical cut.
Maximum independent energy error is 1.4211e-13 hartree; maximum complex
nonsymmetric J/K error is 7.0346e-14. Same-selected-factor dense density error
is at most 1.2491e-14. The existing ORCA reference uses NoRI and therefore a
different Hamiltonian. No matched external RI result is available; comparison
with ORCA/CRYSTAL for this auxiliary/rank convention remains unclaimed.

| Case | Metric / three-center setup calls | Stored / dense complex factors | J/K complex products, blocked / dense | Block contractions | Median J/K ms, blocked / dense / native streaming |
| --- | ---: | ---: | ---: | ---: | --- |
| Water | 25 / 125 | 124 / 343 | 1,888 / 5,488 | 48 | 0.487 / 0.126 / 6.692 |
| NH3 | 225 / 3,375 | 8,135 / 24,389 | 488,100 / 1,463,340 | 72 | 1.453 / 2.457 / 121.894 |
| Broken-spin H2 | 4 / 8 | 4 / 8 | 24 / 48 | 24 | 0.132 / 0.034 / 0.154 |
| Displaced water | 25 / 125 | 124 / 343 | 1,888 / 5,488 | 48 | 0.183 / 0.048 / 3.404 |

The corresponding metric/three-center primitive requests are 225/3,375,
1,369/50,653, 36/216 and 225/3,375. All repeated symmetry J/K builds make zero
integral calls. One native streaming J/K benchmark requires 375, 20,250, 24
and 375 coupling calls, respectively (3, 6, 3 and 3 passes); its primitive
requests are 10,125, 303,918, 648 and 10,125. The saving against native streaming
includes caching and is not attributed entirely to symmetry. No four-center
integrals are evaluated by this RI source.

NH3 retained-factor storage is 130,160 bytes versus 390,224 bytes for a dense
complex factor array; orbital/auxiliary transforms add 26,912/13,456 bytes,
and retained native metric/factor arrays add 13,456 bytes. Setup reserves
42,439,624 logical bytes; the method maximum is 43,889,384. Whole acceptance
peak RSS is 262,832,128 bytes, including Python, tests and independent dense
references. Source setup took 0.01247, 0.03603, 0.00285 and 0.00693 seconds.
Blocked SCF times were 0.01782, 0.08133, 0.00231 and 0.01054 seconds; NH3's
dense-factor/native-streamed SCF times were 0.15255/9.95887 seconds. Timings
are measured under current load with five warmed fixed-density samples, not
a general performance guarantee. Python block overhead loses to dense
contraction on the smaller cases, so both explicit executors remain available.

This closes bounded fitted-source execution and auxiliary-space transport for
molecular complex-Abelian factors, not scalable default routing. Still missing
are producer-issued operation/support/roundoff authority and reduced fitting
generation, measured automatic RI cost routing, bounded native block digestion,
production-sized auxiliary sets and matched external RI evidence. Non-Abelian
reduced products/multiplicities/recoupling, periodic finite-domain and Seitz/
Bloch/momentum admission, distinct CCM/chi sources, covariant frozen/local
correlation spaces, CC residuals, analytic gradients and stress remain separate
experimental or unimplemented capabilities as described above. No periodic
or derivative gate is relaxed by this molecular acceptance.

## Finite native RI Gram preservation and larger fitting bases

Larger auxiliary spaces exposed a false refusal in the initial RI adapter:
water and NH3 with def2-SVP/def2-universal-JKFIT have native retained-factor
Gram defects of 3.2779e-10 and 2.5777e-10. Comparing those factors to an ideal
identity at the 1e-10 source tolerance rejected them before checking the
actual fitted tensor. Whitening the factors or loosening the cutoff would
change the source and is not an acceptable fix.

Numerical source `1637c38c` instead keeps G = F^dagger M F as supplied by
MANTA. A QR coordinate solve measures transport and leakage in the original
factor span. The complete assembled irrep rotation receives a unitary polar
correction; this preserves F F^dagger rather than normalizing F. Original
partner relations and the final selected-factor tensor error are then checked
at the unchanged tolerance. AO orthonormal-panel auditing is unchanged.
The native Gram defect is descriptive evidence, separate from isometry,
leakage and source-error acceptance. No native factorization or rank-cut
implementation changes.

The queued candidate passed 49 regressions plus three complete calculations,
each with blocked, same-selected-factor dense, and native unreduced RI SCF.
There are now matched ORCA 6.1.1 RI-JK references for these cases, also run
through the queue. The external inputs explicitly supply the same QVF
orbital/AuxJK primitive data and bohr coordinates, all-electron charge zero
and multiplicity one. ORCA uses spherical functions, RI for both J and K,
no point-group reduction, and VeryTightSCF with energy 1e-12, RMS/max density
1e-10/1e-9, DIIS error 1e-9 and integral thresholds 1e-14/1e-15. Its printed
auxiliary dimensions are 113, 131 and 139. ORCA uses a Cholesky metric inverse;
MANTA's eigenfactor retains every auxiliary column in these cases. Thus the
external comparison is of the same full-rank fitted approximation, with
different numerical inverse algorithms, not identical floating-point source
bytes. Native and independent-libint checks establish the latter boundary.
The ORCA manual, Sections 2.7.6 and 2.8.5, was used to match custom auxiliary
input and RI-JK semantics; no derivative capability is inferred.

| Case | Orbital / auxiliary dimensions | Blocked energy / hartree | ORCA energy / hartree | Absolute difference |
| --- | ---: | ---: | ---: | ---: |
| Water RHF, def2-SVP / def2-universal-JKFIT | 24 / 113 | -75.95160765116219 | -75.951607651057 | 1.0519e-10 |
| NH3 UHF, def2-SVP / def2-universal-JKFIT | 29 / 131 | -56.140538613389054 | -56.140538613308 | 8.1052e-11 |
| NH3 UHF, cc-pVDZ / cc-pVDZ-JKFIT | 29 / 139 | -56.188539942151536 | -56.188539942070 | 8.1535e-11 |

All differences pass the unchanged 1e-8 external energy limit. Molecular
source tolerance remains 1e-10, SCF energy convergence 1e-11 and gradient
norm 1e-9. Maximum native unreduced energy difference is 5.3291e-12 hartree;
same-selected-factor dense energy and density differences are at most
7.1055e-14 hartree and 5.7455e-15. Independent libint metric/three-center
integrals and an independent eigendecomposition at the same cut give maximum
energy, Fock and complex nonsymmetric J/K errors of 5.4144e-12 hartree,
2.2454e-12 and 2.2574e-12. This includes complete native RI calculations,
not only agreement at a symmetry-projected state. Earlier broken-state,
Cartesian/pure, rank-split and incompatible-source refusals remain covered.
A synthetic nonidentity Gram test explicitly detects any accidental whitening.

The auxiliary isometry residuals are 1.4056e-11, 3.8146e-11 and 1.0402e-11;
coefficient leakage is at most 1.0220e-15. Final tensor error bounds are
5.4421e-12, 1.3743e-11 and 8.3266e-12. These remain below the original 1e-10
cap and explicitly exclude arithmetic roundoff. Metric ranks are unchanged.

| Case | Metric / coupling setup calls | Stored / dense complex factor values | J/K products, blocked / dense | Median J/K milliseconds, blocked / dense / native streaming |
| --- | ---: | ---: | ---: | --- |
| Water | 1,369 / 5,328 | 17,834 / 65,088 | 891,700 / 3,254,400 | 2.111 / 4.345 / 244.307 |
| NH3 def2 | 1,849 / 9,675 | 36,745 / 110,171 | 2,204,700 / 6,610,260 | 3.168 / 8.211 / 436.912 |
| NH3 cc | 2,601 / 11,475 | 38,985 / 116,899 | 2,339,100 / 7,013,940 | 3.408 / 8.617 / 586.532 |

Metric/coupling primitive requests are 2,401/23,716, 3,249/41,553 and
2,601/69,819. Repeated symmetry builds perform zero integral calls; one native
streamed J/K uses 90,576, 183,825 and 229,500 calls across 17, 19 and 20
coupling passes. Native primitive requests per benchmark are 403,172,
789,507 and 1,396,380. Block calls are 160, 72 and 72 versus four dense calls.
Saving against native streaming includes cached factors, not just symmetry.
Complex factor arrays occupy 285,344, 587,920 and 623,760 bytes; the dense
complex comparison occupies 1,041,408, 1,762,736 and 1,870,384 bytes. Native
unrotated factors are real, so these complex-array ratios are not claimed as
native storage ratios. Transform and retained native arrays remain separately
reported in the receipts.

Setup times are 0.04785, 0.08675 and 0.07760 seconds. Blocked SCF times are
0.10322, 0.19520 and 0.22186 seconds, against dense-factor times 0.21371,
0.53025 and 0.58251 seconds. Conservative method reservations peak at
98,440,936, 126,641,928 and 134,823,240 bytes; whole acceptance RSS is
301,416,448 bytes, including independent references and tests. Fixed-density
timings are medians of five warmed samples under current load. No default
routing threshold is inferred from these three cases.

Candidate Python source and the unchanged native runtime were recorded with
710 input files and 396 loaded modules; before/after hashes match. Queue
identities, reference inputs/outputs, arrays and native hashes are archived
outside the product checkout. This supersedes the earlier absence of matched
external RI evidence for these three full-rank molecular sources only.
Automatic RI cost routing, native bounded block digestion, producer-owned
factor export and reduced fitting generation remain next capabilities.
Periodic source/image/phase/momentum admission, non-Abelian reduced products,
local/frozen spaces, CC residuals, gradients and stress retain separate gates.

## Measured routing for prepared RI factors

Numerical source `fec402ba` adds explicit `executor='auto'` to the RI native
JKBuilder bridge. Dense and blocked routes consume the identical selected
factor source; unknown signatures use dense. The same conservative payback
rule as quartet routing compares the minimum dense time with the maximum
blocked time and charges all calibration overhead. Fixed nonsymmetric probes
alternate order, are checked at the unchanged source tolerance, and produce
only bounded scalar profiles. Short-use requests skip all probes when even
the baseline halves cannot repay. No timing evidence changes factor selection,
producer qualification, metric rank, physical Fock identity or state symmetry.
Preparation work is no longer incorrectly charged again in each contraction's
work reservation. Its full costs remain in the source receipt.

The queued candidate passed 79 focused regressions and four complete
RI-HF calculations against explicit dense contraction. The suite also retains
quartet routing and physical Fock bridge coverage. It checks each native
channel signature, independent profiles, bounded calibration storage,
resource refusal before work, corrupted-probe rejection, zero-work short
requests, immutable source identity and profile reuse with fresh counters.
There is no online retuning or method default change.

Water/def2-SVP/def2-universal-JKFIT RHF and NH3/cc-pVDZ/cc-pVDZ-JKFIT UHF
use the exact source identities accepted in the preceding increment. Their
existing ORCA 6.1.1 explicit-basis RI-JK and native unreduced evidence is
reused. Energies remain -75.95160765116219 and -56.188539942151536 hartree,
with ORCA differences 1.0519e-10 and 8.1535e-11, below the unchanged 1e-8
limit. Small water/STO-3G with the same auxiliary basis and broken-spin
H2/STO-3G at 4 bohr additionally run fresh native RI references. Their
energies are -74.8950375471821 and -0.9530462725814317 hartree; native energy
differences are 8.5266e-14 and zero. Broken spin remains present.

All auto/dense energy differences are zero in the reported calculations;
maximum density difference is 4.8850e-15. Independent libint metric/three-center
references give maximum energy and Fock errors of 5.4144e-12 hartree and
2.2454e-12. Same-source calibration differences are at most 9.4330e-15.
Source tolerance remains 1e-10 and SCF energy/gradient criteria remain
1e-11/1e-9. The routed and explicit FockBuild source identities agree, and
forked profile reuse evaluates the same physical state with fresh counters.

Five signatures were calibrated with two probe pairs each and an estimate
of 100 future builds per signature, except H2's explicit short-use estimate
of two. These estimates are cost inputs, not measured iteration counts.
Every selected signature chose blocked for the large cases and dense for
small water. H2 selected dense with no probe calls. Preparing unused signatures
is included in the measured overhead below; applications can request only
the signatures they expect to use.

| Case | Calibration seconds | Auto SCF seconds | Dense SCF seconds | Calibration plus auto SCF seconds | Physical J/K builds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Water large | 0.07417 | 0.11155 | 0.23174 | 0.18572 | 73 |
| NH3 large | 0.16351 | 0.25371 | 0.59635 | 0.41722 | 68 |
| Water small | 0.00349 | 0.00408 | 0.00296 | 0.00757 | 42 |
| Broken-spin H2 | 0 | 0.00113 | 0.00092 | 0.00113 | 13 |

Both large runs recover the complete five-profile calibration cost in this
measured workload. Small water pays to discover that dense execution is
faster, and auto therefore costs more than an explicit dense request. H2
avoids that probe overhead but still has ordinary dispatch overhead. These
are individual runs under current load, not guaranteed speedup ratios.
Source preparation takes another 0.04906, 0.07976, 0.00598 and 0.00276 seconds
respectively, equally applicable to either executor.

| Case | Probe complex products / block calls / AO transport products | Physical complex products / block calls / AO transport products | Maximum logical bytes, including calibration |
| --- | --- | --- | ---: |
| Water large | 48,758,136 / 1,764 / 1,880,064 | 47,509,776 / 8,064 / 4,506,624 | 98,969,320 |
| NH3 large | 110,365,872 / 792 / 3,316,904 | 156,407,820 / 4,284 / 9,950,712 | 135,342,792 |
| Water small | 82,908 / 528 / 46,648 | 163,268 / 104 / 64,484 | 26,538,280 |
| Broken-spin H2 | 0 / 0 / 0 | 720 / 54 / 864 | 25,847,464 |

All routing and repeated physical builds perform zero integral calls. Native
setup remains 1,369/5,328, 2,601/11,475, 25/125 and 4/8 metric/coupling calls;
primitive requests are 2,401/23,716, 2,601/69,819, 225/3,375 and 36/216.
Actual physical counts were independently summed from per-build receipts;
probe work never enters those totals. The largest paired-calibration work
reservation is 5,244,979,584 modeled units, charging execution rather than
repeated preparation. Source-factor and transform storage are unchanged.
Whole acceptance peak RSS is 301,121,536 bytes including tests and independent
references. All 710 expected files and 396 loaded modules were recorded with
matching before/after hashes; exact queue/runtime provenance and outputs are
archived outside the checkout.

This closes bounded measured routing for the prepared RI developer bridge.
Normal method/job/QVF integration, automatic workload estimates, native bounded
block digestion and producer-owned factor export/reduced fitting generation
remain next execution capabilities. Molecular source qualification does not
supply periodic finite-support, Seitz/Bloch/momentum, non-Abelian reduced-product,
local/frozen-space, CC residual, gradient or stress acceptance. Those original
goal requirements and their explicit fallbacks remain separate.


## Qualified RI methods, jobs and QVF

Numerical source `a9e45532` connects the existing finite complex-Abelian RI
source and measured routing to explicit molecular RHF/UHF, `run_job`, and
executable QVF requests. It reuses `PreparedMolecularSymmetry`, native SCF,
and the shared evaluated `MolecularSymmetryResult`/FockBuild. The actual S/H
pass covariant invariance before admission. The auxiliary basis, native rank
cutoffs and complete characters are explicit immutable inputs. Fresh jobs
requalify; serialized requests carry neither audit authority nor profiles.
The ordinary job bibliography now includes the executed RI symmetry papers.

Queued acceptance passed 159 regressions and three complete ordinary jobs,
with same-factor dense method comparisons. Water and ammonia also passed
fresh executable QVF replay. The suite covers both native method spin routes,
complex characters, capped and zero-step states, independent integral/Fock
checks, nonsymmetric trial-density covariance, prepared reuse with fresh
counters, changed geometry, incompatible one-electron operators, resource
refusal and malformed/unqualified request schemas. Existing quartet, native
bridge and cost-routing regressions remain passing. Scientific tolerances
stay at 1e-10 source qualification and 1e-11/1e-9 SCF energy/gradient.

| Case | Job energy / hartree | Dense method energy difference | External/native energy difference | Independent Fock maximum error | QVF replay energy difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Water/def2-SVP/def2-universal-JKFIT RHF | -75.95160765116213 | 1.4211e-14 | 1.0514e-10 (ORCA) | 2.2454e-12 | 6.7218e-12 |
| NH3/cc-pVDZ/cc-pVDZ-JKFIT UHF | -56.18853994215148 | 2.8422e-14 | 8.1478e-11 (ORCA) | 2.8245e-13 | 7.8160e-13 |
| Broken-spin H2/STO-3G, same auxiliary, 4 bohr | -0.9530462725814317 | 0 | 0 (native RI) | 2.2205e-16 | Not repeated |

The two large sources exactly match their earlier ORCA 6.1.1 RI-JK/native
qualified source identities, allowing reuse of those matched references.
Maximum independent energy error is 5.4428e-12 hartree. H2 retains distinct
spin densities. QVF unit conversion can change geometry's last bits, so replay
qualifies its actual producer; equality of a copied digest is not an
acceptance criterion. The unchanged external energy limit is 1e-8 hartree.
No matched external reference is claimed for the broken-spin minimal case.

| Case | Source setup / calibration / SCF / terminal seconds | Physical complex products / block calls / AO transport products | Probe complex products / block calls / AO transport products | Maximum source/J-K logical bytes |
| --- | --- | --- | --- | ---: |
| Water | 2.11202 / 1.44030 / 6.18560 / 0.32815 | 125,148,764 / 3,366 / 4,340,736 | 32,505,424 / 1,176 / 1,216,512 | 98,821,864 |
| NH3 | 0.12651 / 0.09378 / 0.32218 / 0.00777 | 156,407,820 / 4,284 / 9,950,712 | 55,494,704 / 416 / 1,951,120 | 135,342,792 |
| H2 | 0.00395 / 0 / 0.00935 / 0.00014 | 720 / 54 / 864 | 0 / 0 / 0 | 25,847,464 |

Physical counts include the cold terminal FockBuild contraction; this RI
consumer has no terminal matrix capture yet. Routing probes are separate.
All repeated J/K builds perform zero integral calls. Source setup uses
1,369/5,328, 2,601/11,475 and 4/8 metric/coupling shell calls respectively,
with 2,401/23,716, 2,601/69,819 and 36/216 primitive requests. Stored selected
complex factor arrays occupy 285,344, 623,760 and 64 bytes; conservative
logical source/J-K reservations include much more staging/workspace.
Separate method/signature reservations are 655,360/54,272, 926,720/70,144 and
69,632/23,552 bytes. Whole acceptance peak RSS is 406,421,504 bytes.
One-electron preparation makes three calls per source plus the overlap call
inside RI qualification, and checks at most 7.5247e-14 operator residual.

Only method-needed signatures are calibrated. Water chose dense for standalone
K and blocked for its other signatures; ammonia chose blocked; H2's short-use
estimate skipped every probe and chose dense. These decisions and timings
reflect this loaded run, not a speedup guarantee. The earlier isolated routing
measurement remains the relevant performance comparison; no new speed claim
is made from job/output timings. Full job times were 15.8793, 0.8494 and
0.1900 seconds, including non-SCF output work.

Validation used the healthy native runtime with an explicit candidate Python
overlay. All 714 expected files and 420 loaded modules retained their recorded
hashes. The compatibility overlay omits only unrelated newer periodic BUPO
exports absent from that runtime; no periodic runtime acceptance is inferred.
Queue, runtime, native hash, exact input/output and overlay provenance are
archived outside the product checkout.

Supported here is explicit all-electron molecular RHF/UHF over this qualified
finite RI source, including normal output and fresh QVF execution. It remains
experimental and opt-in. Producer-owned factor export, reduced fitting
construction, native bounded block execution, automatic workload estimates and
RI terminal capture remain next capabilities. Non-Abelian products, periodic
Seitz/Bloch/momentum and CCM/chi sources, CC residuals, frozen/local projectors,
response, gradients and stress retain their separate qualification and
acceptance requirements. This method entry does not promote any of them.

## Producer-owned RI factors

Numerical source `917dd74d`, validated on main `b16999d2`, moves fitted-factor
construction into MANTA's existing `RIJKBuilder`. The factory-owned export
binds the actual retained metric factor, coupling source, rounded payload and
executed census. Its plan admits arrays and integral/contraction work before
allocation. The symmetry consumer uses the export by default, keeps complete
retained-space and tensor-selection qualification, and offers the previous
`ordered-tiles` construction explicitly. Methods, prepared handles and QVF
carry that choice through the existing FockBuild interface.

This removes the consumer's full raw three-center array. It does not reduce
the ordered coupling census or certify a spatial operation. The native
auxiliary rotation remains unconjugated; complex AO variance, reconstructed
four-index conjugation, actual finite Gram and native rank are unchanged.
Different contraction order can change rounded factors and their selected
source identities. Reduced/dense comparisons use the same selected payload;
the independent exporter comparison requires identical parent producer,
coupling and metric-factor identities, not identical rounded exports.

All 206 focused checks pass, covering independent fitted integrals, native
streamed J/K, real and complex nonsymmetric trial densities, Cartesian/pure
actions, full/reduced/zero native rank, resource refusal before export work,
owned-copy lifetime and mutation isolation, foreign-source rejection, and
method/job/QVF admission. Native zero-rank export performs no coupling work;
the symmetry consumer continues to refuse an empty retained space. Seven
initial job checks encountered a stale generated basis overlay before any
calculation; rebuilding that overlay resolved them. The five acceptance
basis files were byte-identical to the earlier accepted inputs.

Queued scientific acceptance completed three ordinary jobs, same-payload
dense controls, independent ordered-tile controls and two fresh QVF replays.
The two large systems retain the earlier qualified parent source, coupling
and metric-factor identities, so their matched ORCA 6.1.1 RI-JK references
were reused. Independent libint metric/three-center reconstruction checks
the actual terminal Fock matrices and energy. No external reference is
claimed for the broken-spin minimal case; native RI supplies its control.

| Case | Energy / hartree | Same-payload dense energy difference | Ordered-tile energy difference | ORCA/native energy difference | Independent Fock maximum error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Water/def2-SVP/def2-universal-JKFIT RHF | -75.95160765116752 | 1.4211e-14 | 5.4002e-12 | 1.1052e-10 (ORCA) | 7.5496e-14 |
| NH3/cc-pVDZ/cc-pVDZ-JKFIT UHF, complex C3 | -56.188539942151195 | 0 | 3.1264e-13 | 8.1194e-11 (ORCA) | 1.1547e-13 |
| Broken-spin H2/STO-3G, same auxiliary, 4 bohr | -0.9530462725814317 | 0 | 0 | 0 (native RI) | 2.2205e-16 |

Maximum independent energy error is 1.4211e-13 hartree; replay differences
are 3.4107e-13 and 1.8475e-13 hartree. H2 retains distinct spin densities.
Source tolerance 1e-10, SCF energy/gradient thresholds 1e-11/1e-9 and external
energy limit 1e-8 hartree remain unchanged. Arithmetic roundoff is still
excluded from the selection-error bound.

| Case | Coupling shell calls / primitive products / integral values | Export products / compensated accumulations | Real exported factor bytes | Raw coupling bytes avoided |
| --- | --- | --- | ---: | ---: |
| Water | 5,328 / 23,716 / 65,088 | 7,354,944 / 2,408,256 | 520,704 | 520,704 |
| NH3 | 11,475 / 69,819 / 116,899 | 16,248,961 / 5,961,849 | 935,192 | 935,192 |
| H2 | 8 / 216 / 8 | 16 / 16 | 64 | 64 |

Both exporters use these same coupling calls. Metric construction separately
uses 1,369/2,401, 2,601/2,601 and 4/36 shell calls/primitive products. Export
is one coupling pass, with no independent Python fitting provider. Repeated
J/K uses zero integral calls. Selected complex factor storage remains
285,344, 623,760 and 64 bytes respectively.

| Case | Source setup / calibration / SCF / terminal seconds | Physical complex products / block calls / AO transport products | Probe complex products / block calls / AO transport products | Maximum source/J-K logical bytes |
| --- | --- | --- | --- | ---: |
| Water | 0.07074 / 0.06340 / 0.16064 / 0.00255 | 47,509,776 / 8,064 / 4,506,624 | 32,505,424 / 1,176 / 1,216,512 | 100,472,288 |
| NH3 | 0.09553 / 0.08636 / 0.29348 / 0.00853 | 156,407,820 / 4,284 / 9,950,712 | 55,494,704 / 416 / 1,951,120 | 138,535,482 |
| H2 | 0.00296 / 0 / 0.00408 / 0.00012 | 720 / 54 / 864 | 0 / 0 / 0 | 25,804,188 |

Physical counts include the cold terminal contraction. Water and NH3 selected
blocked execution for their requested signatures; H2's expected-use estimate
of two skipped every probe and selected dense. Routing remains measured and
source-preserving. Full job times were 1.06812, 0.83770 and 0.15758 seconds.

Three alternating-order preparation samples measured the following. These
times include Python allocation tracing and varied substantially under load;
they do not establish an uninstrumented speedup.

| Case / exporter | Preparation seconds, median (range) | Python-traced peak bytes, median |
| --- | --- | ---: |
| Water / producer | 2.10592 (0.08196-3.10736) | 10,022,600 |
| Water / ordered tiles | 14.15873 (3.04529-16.42959) | 10,544,578 |
| NH3 / producer | 0.43900 (0.30827-3.06279) | 17,794,757 |
| NH3 / ordered tiles | 3.87049 (1.88787-21.98737) | 18,731,578 |

Python tracing excludes native and BLAS allocations. Whole scientific
acceptance peak RSS was 476,905,472 bytes, not a controlled exporter memory
comparison. Producer source reservations are 97,869,280/133,994,330 bytes
versus ordered-tile 95,969,000/130,413,160 bytes for water/NH3: the conservative
producer plan explicitly reserves compensation, binding and digest copies.
Removing the raw array therefore does not imply a smaller logical reservation
or a demonstrated process-RSS reduction.

The refreshed native module contains the implementation directly, without a
compatibility overlay. All 1,457 expected input/source/native hashes and 424
loaded module hashes remained consistent; unchanged molecular source/object
hashes justify reuse of the focused checks across the unrelated main merge.
Queue, build, core hash, exact input/output and comparison evidence are
archived outside the product checkout.

Explicit molecular Coulomb RI RHF/UHF remains experimental and opt-in. The
native exporter also has focused Coulomb/erf/erfc coverage; that does not
admit range-separated kernels through this symmetry method. Next production
capabilities are producer-issued operation/roundoff and screening-closure
certificates for representative fitting generation, native bounded block
execution, automatic workload estimates and shared RI terminal capture.
MANTA/BUPO owns producer screening and hierarchy bounds. Fock unification
continues to consume the same evaluated source/density result. PBC Parity,
non-Abelian products, CC residuals, local/frozen projectors, response,
gradients and stress keep their separate source and acceptance requirements.

## Shared exact-input RI terminal capture

Source `2eee59fe` extends the existing method-local quartet capture to
`MolecularRIJK` through one shared implementation. It retains one owned
full-channel evaluation and at most one pending RHF J evaluation. Full UHF
uses ordered total/alpha/beta inputs; separately evaluated RHF J and K can
join only when their source and density bytes match. The selected fitted
source identity is required, not merely its parent producer or metric factor.
No state symmetry, tolerance-based density match or spin permutation is used.

Public RI method forks enable capture after resource admission and before
new route probes. Reusable prepared sources remain free of densities and
Fock matrices; forks inherit only source ownership and scalar cost profiles.
The shared FockBuild still assembles the evaluated energy, physical Focks,
frames and output state. Successful reuse reports zero new integral,
factor-contraction and AO-transport work with original build numbers. Missing
or changed inputs, reset state and failed full replacements keep the cold
path. This does not change factor construction, integral counts, metric rank,
selection rules, SCF tolerances or source/state admission.

All 178 focused checks pass. The exact-input capture tests now run against
both quartet and fitted sources: independent input/output ownership,
nonsymmetric trial densities, bit changes including signed zero, spin order,
cross-source capture rejection, response calls, split RHF joining, reset and
failed replacement, and budget refusal before work. RI method tests cover
converged, capped and zero-step results and prepared reuse without state
retention. Existing complex-factor covariance and native/ordered quartet
checks remain passing. No native code changed; validation reused the prior
qualified producer-export binary with unchanged native source/core hashes.

The queued RI phase completed three ordinary jobs, same-selected-source
dense controls, explicit ordered-tile controls, independent integral/Fock
checks and two fresh QVF replays. Capture-enabled and capture-disabled dense
SCF controls had exactly equal FockBuild identities and energies. The two
large sources match their prior qualified identities and ORCA 6.1.1 RI-JK
references; broken-spin H2 retains distinct spin densities and its native RI
control. All external and source tolerances remain unchanged.

| Case | Energy / hartree | ORCA/native energy difference | Independent Fock maximum error | Reused physical builds |
| --- | ---: | ---: | ---: | --- |
| Water/def2-SVP/def2-universal-JKFIT RHF | -75.95160765116754 | 1.1054e-10 (ORCA) | 7.1721e-14 | 17 and 18 |
| NH3/cc-pVDZ/cc-pVDZ-JKFIT UHF | -56.188539942151195 | 8.1194e-11 (ORCA) | 1.1547e-13 | 16 |
| Broken-spin H2/STO-3G, same auxiliary, 4 bohr | -0.9530462725814317 | 0 (native RI) | 2.2205e-16 | 6 |

Maximum independent energy error is 1.4211e-13 hartree. Water/NH3 QVF
replay energy differences are 3.4107e-13/1.8475e-13 hartree. Ordered-tile
control differences are at most 5.4144e-12 hartree. The selected factor source
is unchanged by capture; different qualified exporter rounding remains
separate from exact-input reuse.

| Case | Dense control's terminal products / block calls / AO transport products, all avoided by capture | Captures / copied values / compared values | Retained snapshot + pending bytes / reserved bytes | Capture seconds / terminal reuse seconds |
| --- | --- | --- | --- | --- |
| Water | 3,254,400 / 4 / 82,944 | 17 / 50,112 / 21,888 | 13,824 + 9,216 / 77,824 | 0.0027244 / 0.0000061 |
| NH3 | 13,794,082 / 6 / 292,668 | 16 / 80,736 / 2,523 | 40,368 + 0 / 111,744 | 0.1128835 / 0.0000168 |
| H2 | 80 / 6 / 96 | 6 / 144 / 12 | 192 + 0 / 4,608 | 0.0517244 / 0.0000117 |

The avoided counts are measured in the explicit dense cold controls; capture
counts, storage and timing are from the ordinary jobs, which can choose
different routes. All three ordinary terminal
receipts report zero new contractions and transport. Their original physical
SCF products remain counted: 86,216,928, 154,456,666 and 640 respectively,
with 4,423,680, 9,658,044 and 768 AO-transport products. Fitting preparation
and route calibration are unchanged. Maximum J/K logical reservations are
100,550,112, 138,647,226 and 25,808,796 bytes, including the additional
`4096 + 128*n_AO**2` capture reservation. RI acceptance peak RSS was
389,283,840 bytes; it is not a controlled memory comparison.

Loaded-run source/calibration/SCF times were 9.39177/6.98070/14.59322 seconds
for water, 7.16153/7.03748/19.26765 for NH3 and 0.03132/0/0.92151 for H2.
Capture-disabled dense terminal contractions took 0.60665, 1.19599 and
0.12276 seconds. Scheduling variation is large; these observations do not
establish an end-to-end speedup. The bounded byte copies and comparisons are
reported rather than treated as free work.

The shared quartet path also completed NH3/cc-pVDZ UHF and an exact
cold-versus-captured FockBuild comparison. Its energy is
-56.18855210065031 hartree, 7.0331e-10 from the matched ORCA no-RI reference.
The saved result has exactly the earlier accepted density and Fock bytes,
verified against its complete density/FockBuild digests with current source
metadata. Its unchanged producer, complete audit and zero-omission mask
justify reuse of the independent full-ERI errors: 2.2427e-14 for Fock and
4.2633e-14 hartree for energy. Terminal evaluation reuses build 16 with zero
new quartet, primitive, density-product and angular-transport work.
Its 61 physical SCF builds retain 115,473 shell calls, 7,645,923 primitive
requests, 65,777,133 density products and 43,517,278 angular products.
Sixteen captures copy 80,736 values and compare 2,523, retaining 40,368 bytes
within a 111,744-byte reservation. Capture and terminal-reuse times are
0.0007918 and 0.0000272 seconds; source preparation and SCF took 50.68368
and 219.93979 seconds in this loaded run.

The initial queue run reached its wall limit after all RI checks passed;
only the unfinished quartet phase was submitted again. That calculation
passed convergence and exact cold/captured equality, then stopped at a stale
harness comparison between v1 and v2 source digests. The already landed
`e2e3384d` source-format change explains that difference. A corrected queued
assessment of the saved output passed without repeating SCF: producer/full
audit and omitted mask matched, and exact D/F bytes matched the accepted
independent reference. Original failure records and outputs are retained.
All 1,478 pinned input/source/native hashes were checked unchanged after the
interrupted phases and during the successful result assessment. Queue,
runtime, references and detailed provenance are archived outside this tree.

The current molecular quartet and RI entries share exact terminal reuse;
this closes that execution gap. Native bounded RI block contractions,
automatic workload estimates, and producer-issued operation/roundoff and
screening-closure certificates remain next capabilities. RI remains explicit
and experimental. MANTA/BUPO producer ownership, FockBuild state ownership,
and PBC Parity's finite-domain/Seitz/Bloch/momentum contracts are unchanged.
No new non-Abelian CC, local/frozen-space, periodic, response, gradient or
stress acceptance is claimed.

## Native execution of qualified RI blocks

Source `3822f763` replaces repeated Python block dispatch in the existing
`blocked` route with an immutable native factor layout and one auxiliary
exchange panel. The source factory still qualifies the complete producer
export, retained Gram/rank, orbital and auxiliary representations and
selection-error bound. The numerical source identities match the earlier
accepted producer exports exactly. Neither integral generation nor source,
state or retained-space admission changes. The low-level `numpy` executor
preserves the former blocked implementation; `dense` uses the identical
selected factors. Public method options and measured route selection retain
their existing interfaces.

All 166 focused checks passed, including independent arbitrary complex
rectangular factors, nonsymmetric densities, empty irrep groups, input/output
ownership, overflow and resource refusal, failed calls, complex covariance,
Cartesian/pure shells, retained-space cuts, source isolation and method-owned
terminal Fock reuse. The first queue attempt failed at import before any
calculation because an incoming periodic binding required a refreshed native
module. After rebuilding, both binding checks passed; unchanged molecular
object hashes justify reuse of the focused numerical checks. The corrected
queued acceptance completed successfully. It ran explicit native-block RHF
and UHF jobs, same-source dense SCF controls, independent integral/Fock/energy
checks, two fresh QVF replays and alternating-order contraction benchmarks.
No scientific tolerance was changed.

| Case | Energy / hartree | ORCA/native energy difference | Independent Fock maximum error | Dense SCF energy difference |
| --- | ---: | ---: | ---: | ---: |
| Water/def2-SVP/def2-universal-JKFIT RHF | -75.95160765116749 | 1.1049e-10 (ORCA) | 7.7050e-14 | 4.2633e-14 |
| NH3/cc-pVDZ/cc-pVDZ-JKFIT UHF, complex C3 | -56.18853994215115 | 8.1152e-11 (ORCA) | 1.1192e-13 | 4.2633e-14 |
| Broken-spin H2/STO-3G, same auxiliary, 4 bohr | -0.9530462725814317 | 0 (native RI) | 2.2205e-16 | 0 |

The matched ORCA 6.1.1 RI-JK references and exact producer/factor identities
were reused. H2 retains distinct spin densities; it has a native RI control,
not an external-program comparison. Maximum independent energy error is
1.7054e-13 hartree. Water/NH3 QVF replay errors are 3.5528e-13/1.2790e-13
hartree. All 1,482 pinned input/source/native hashes and 424 loaded module
hashes agree with the recorded runtime before/after evidence.

The following benchmark contracts a fixed nonsymmetric complex AO density,
including its AO/adapted transport. It uses one requested OpenMP/BLAS thread,
one warmup and eight timed trials with alternating executor order. These are
same-process measurements on the identical selected source, not comparisons
against earlier jobs under different host load.

| Case | Complex products, native/NumPy blocks vs dense | Native panel / NumPy batched / dense contraction calls | Median native / NumPy / dense time (ms) | Maximum executor difference |
| --- | --- | --- | --- | ---: |
| Water | 891,700 vs 3,254,400 | 4,520 / 160 / 4 | 1.0559 / 2.3816 / 4.5019 | 3.9721e-15 |
| NH3 | 2,339,100 vs 7,013,940 | 3,336 / 72 / 4 | 3.3712 / 3.9344 / 9.5978 | 2.2645e-15 |
| H2 | 24 vs 48 | 24 / 24 / 4 | 0.02335 / 0.12044 / 0.03727 | 0 |

Native execution is 2.26/1.17/5.16 times faster than the previous NumPy block
path in this measurement, and 4.26/2.85/1.60 times faster than dense factors.
The scalar-product reduction is unchanged from the qualified block algebra;
the new improvement removes Python dispatch and full-rank exchange
intermediates. Panel and batched contraction-call counts have different
granularity, explicitly shown above. Separate nonsymmetric real HF probes
selected native blocks for water J/K and NH3 total-J/two-spin-K at expected
use 100; calibration took 0.01152/0.05191 seconds. Expected use two for H2
skipped all probes and retained dense execution, preserving the cost gate.

| Case | Additional owned native factor bytes | Native layout logical reservation | Physical SCF products / panel calls / AO transport products | Maximum J/K logical bytes |
| --- | ---: | ---: | --- | ---: |
| Water | 285,344 | 721,600 | 44,870,344 / 215,152 / 4,257,792 | 101,140,640 |
| NH3 | 623,760 | 1,432,160 | 142,607,130 / 180,978 / 9,072,708 | 139,948,314 |
| H2 | 64 | 67,520 | 320 / 320 / 768 | 25,745,244 |

The physical runs used explicit native blocks for 69/61/12 builds; automatic
probe costs above are separate. Terminal reuse retained zero new products,
integrals and AO transport. Fitting still uses 5,328/11,475/8 coupling shell
calls and 23,716/69,819/216 primitive-product requests respectively. No
integral-work reduction is attributed to this increment. Peak process RSS
was 471,515,136 bytes for the complete acceptance, including controls and
references; it is not an incremental-memory comparison. Additional native
factor ownership is charged, not treated as free or as a memory saving.
Source setup / physical SCF times were 0.06093/0.06589 seconds for water,
0.08927/0.21575 for NH3 and 0.00273/0.00447 for H2. Ordinary job wall times
were 0.50018/0.59204/0.14787 seconds. No end-to-end speedup over previous
loaded-host runs is inferred from these timings.

MANTA received the reusable contraction interface; its producer-owned
operation/roundoff and screening-closure certificates remain the next
requirement for omitting factor-generation work. Automatic workload estimates
also remain missing. FockBuild continues to own the evaluated physical state,
and PBC Parity retains its separate source, Seitz/image/Bloch/momentum and
quadrature requirements. Molecular RI remains explicit and experimental;
this acceptance adds no periodic, non-Abelian CC, local-projector, response,
gradient or stress capability. Original failure evidence, exact runtime,
queue records and comparisons are archived outside the product tree.

## Demand-driven molecular route calibration

Source `0d98618a` removes eager calibration of unrequested RHF/UHF channel
signatures. Each actual first request obtains or reuses its own scalar cost
profile. Native NOITER has an exact single-evaluation horizon and skips new
probes; that temporary baseline decision is not saved onto a prepared source
for future iterative runs. Already paid profiles remain reusable. Source,
spin state, retained spaces and FockBuild ownership keep their distinct
admission rules. This increment does not predict general iterative workloads
or replace producer qualification with timings.

All 192 focused checks passed. After adding the explicit `targets` field to
zero-probe receipts, 93 relevant cases passed again; these are included in the
192, not additional unique tests. Coverage includes arbitrary nonsymmetric
trial densities, actual-request-only probes, failed and invalid probes,
resource refusal before calibration, prepared-source reuse, and NOITER
followed by iterative use for both quartet and RI builders. Existing complex,
Cartesian/pure, source/rank and physical-state checks remain in the focused
set. The source was landed before queued acceptance; the compatible native
module was rebuilt from `955ec2c8`, whose native inputs are unchanged by this
Python-only increment. All 2,288 pinned files and 425 loaded module hashes
passed the runtime before/after check.

Queued acceptance completed in 31.954 seconds with one CPU and a 2 GiB
reservation. It exercises ordinary auto-routed jobs, same-source dense RI
SCFs, independent integral/Fock/energy references, two fresh QVF replays,
three NOITER calculations and cold versus reused terminal evaluation.
Matched ORCA references were reused; H2 uses an independent native RI control
and has no external reference here. All existing tolerances are unchanged.

| Case | Energy / hartree | ORCA/native energy difference | Independent Fock maximum error |
| --- | ---: | ---: | ---: |
| Water/def2-SVP/def2-universal-JKFIT RHF | -75.95160765116749 | 1.1049e-10 (ORCA) | 7.7050e-14 |
| NH3/cc-pVDZ/cc-pVDZ-JKFIT UHF, complex C3 | -56.18853994215115 | 8.1152e-11 (ORCA) | 1.1192e-13 |
| Broken-spin H2/STO-3G, same auxiliary, 4 bohr | -0.9530462725814317 | 0 (native RI) | 2.2205e-16 |
| NH3/cc-pVDZ quartet UHF, full C3v | -56.18855210065031 | 7.0331e-10 (ORCA) | 2.2427e-14 |

Maximum dense RI SCF energy difference is 4.2633e-14 hartree; maximum
independent energy difference is 1.7054e-13. Water/NH3 QVF replay errors are
3.5528e-13/1.2790e-13 hartree. Fresh NOITER energy differences from their
accepted RI densities are zero, with exactly one physical build and zero new
probes in each case. Terminal reuse performs zero new integral, contraction
or AO-transport work throughout.

The former eager signature schedule was reproduced on each identical source.
The following counts include both baseline and candidate probes. Timings are
single calibration observations, including transport and comparisons, not a
paired end-to-end speedup measurement. Automatic routing selected native RI
blocks for water/NH3, native quartet reconstruction for NH3, and the dense
fallback without probes for H2's declared two-use workload.

| Case | Probes, eager to demand | Probe scalar products, eager to demand | Probe AO-transport products, eager to demand | Calibration seconds, eager to demand |
| --- | --- | --- | --- | --- |
| Water RI RHF | 16 to 12 | 32,505,424 to 16,584,400 | 1,216,512 to 774,144 | 0.05937 to 0.04565 |
| NH3 RI UHF | 12 to 12 | 55,494,704 to 55,494,704 | 1,951,120 to 1,951,120 | 0.13730 to 0.15587 |
| H2 RI UHF | 0 to 0 | 0 to 0 | 0 to 0 | 0 to 0 |
| NH3 quartet UHF | 12 to 12 | 14,145,620 to 14,145,620 | 4,280,388 to 4,280,388 | 5.05519 to 4.88062 |

Water never requests the separate two-K signature: its removal saves
15,921,024 scalar products and four probes. Both UHF cases with probes request
all three signatures, so there is no probe-count reduction. The quartet
probes still perform 315,108 shell calls and 11,997,024 primitive-quartet
requests. RI probe integrals remain zero because the selected factors already
exist. A fresh NOITER uses only its physical baseline build; its source setup
and qualification are still paid.

| Case | Physical SCF products / contraction calls / AO transport products | Maximum physical J/K logical bytes | Source setup / physical SCF / elapsed SCF seconds |
| --- | --- | ---: | --- |
| Water RI | 44,870,344 / 215,152 / 4,257,792 | 101,271,712 | 0.09818 / 0.11863 / 0.16428 |
| NH3 RI | 142,607,130 / 180,978 / 9,072,708 | 140,079,386 | 0.18181 / 0.43077 / 0.58665 |
| H2 RI | 640 / 48 / 768 | 25,876,316 | 0.00401 / 0.00590 / 0.00590 |
| NH3 quartets | 65,777,133 / 4,708,125 / 43,517,278 | 71,482,441 | 2.08186 / 8.78897 / 13.66959 |

Elapsed SCF includes demand calibration; physical SCF excludes that separately
recorded cost. There were 69/61/12/61 physical builds respectively. Maximum
calibration logical reservations were 101,447,840 / 140,467,866 / 25,876,316 /
71,870,921 bytes. The RI acceptance portion, including controls and references,
peaked at 442,744,832 bytes RSS; this is not an incremental memory comparison
or the later quartet peak. Prepared sources store only bounded scalar
profiles; method-local capture remains charged. No memory reduction is claimed.

RI coupling calls remain 5,328 / 11,475 / 8, with identical producer, coupling,
metric-factor, rank and selected-source identities to the accepted exports.
The quartet source pays a fresh 50,625-call audit, then 115,473 physical shell
calls and 7,645,923 primitive-quartet requests. Qualification records separately
bind producer, audit, source and screening policy. No source-audit or
factor-generation savings are attributed to this change.

The Fock consumer handoff records the changed route/timing receipt semantics.
MANTA still owns source generation, bounds and screening closure; its current
screening work does not itself certify symmetry-compatible omission. PBC
Parity keeps separate finite supports, full image shifts, wraps, Bloch phases,
weights, momentum and state admission. The next missing producer capability is
an operation- and pair-bound certificate including arithmetic error and
screening closure for reduced factor generation. General workload prediction,
non-Abelian reduced CC products, local/frozen projectors, periodic source
reduction and method-specific derivatives remain separate. These molecular
routes remain explicit and experimental. Queue/runtime/reference evidence is
archived outside the product tree.

Incoming MANTA source `9c396ff6` separately adds fixed-metric periodic
coupling-only screening, with paired actions, reference-source identity and
fixed-density J/energy bounds. Its contract explicitly excludes integral,
factor, phase and contraction roundoff and supplies no operation closure.
That native change is outside this molecular acceptance runtime; its own
producer evidence is recorded in the linked
[MANTA contract](design/manta_scientific_contract.md). It is not an admission
to molecular RI reduction or a reason to repeat unchanged molecular checks.

### Ordinary product algebra and finite coordinate correction

The matrix-irrep increment supplies ordinary finite unitary product
resolution, repeated-irrep multiplicity spaces, aligned partners and explicit
recoupling. MolSym's complete projector (Eq. 3) and Hellmann-Gauss's C3v
partner and missing-element equations remain the mathematical starting point.
The C3v examples do not establish general non-Abelian CCSD. Independent
C3v and tetrahedral A4 product references, complex C3 variance, Cartesian and
pure molecular pair actions, rectangular maps, gauge refusal and resource
admission cover the finite algebra. No new CC residual, derivative, periodic
source or producer certificate is supplied.

The interrupted finite-Gram correction is now on main as `d5a519cb`. Reduction,
application and reconstruction use the actual retained coordinate inverse.
Reduced composition cancels that inverse against the intermediate basis;
it no longer silently inserts the finite Gram matrix. Source-error propagation
includes measured frame conditioning and retained storage counts the inverse.
Nine affected regression cases passed. The new regression introduces a Gram
defect above 1e-11 but below the unchanged 1e-10 representation tolerance, then
requires inverse residual below 2e-15 and repeated reconstruction/application
within 3e-14. Full source audits and all scientific tolerances remain intact.

The earlier queued ammonia/cc-pVDZ reference is reused. It used 29 AOs,
841 pair coordinates, C3v multiplicities (182, 101, 279), a complete 707,281
value libint ERI tensor and native RHF/UHF drivers with a Python J/K harness.
The prior RHF/UHF energies were -56.18855210065037/-56.18855210065047 hartree,
within 7.035e-10 of matched ORCA and 1.706e-13 of same-source dense SCF.
This was an explicit algebra harness, not a public method dispatch.

Before the inverse correction, combined source construction, basis preparation
and complete qualification cost 1.65182 s. Reduced RHF/UHF SCF cost
0.05646/0.07161 s versus 0.01756/0.02123 s for dense controls. Reduced
composition alone took a median 0.001784 s versus 0.010992 s for a full
product, but reconstruction added 0.023393 s. Those observations already
rule out an end-to-end speedup claim. They are historical controls, not
measurements of the corrected inverse or a cross-host timing comparison.

The final-inverse calculation is accepted after collection and numerical
assessment on 2026-10-25. It ran through vq with the byte-identical
`symmetry_products.py` from `d5a519cb` on the healthy native runtime
`63031b08`; it did not substitute an older module or rebuild unrelated code.
All six payload hashes, the reused density/reference files and all 420 loaded
module hashes matched, including the native binary before and after execution.
Queue, native-binary and payload identities and complete outputs are archived
outside the product tree. The nine affected source checks above are reused;
no unchanged test suite or external-reference calculation was repeated.

The neutral singlet NH3 fixture has N at (0, 0, 0.3) bohr and three H atoms
at radius 1.7 bohr, z = -0.7 bohr, separated by 120 degrees. Both complete
cc-pVDZ SCFs use HCORE and unchanged energy/gradient thresholds 1e-11/1e-9,
converging in 15 iterations. The stored dense SCF control and matched ORCA
energy -56.188552099947 hartree were reused; fresh independent dense AO
contractions check the final density's Fock and energy.

| Method | Energy / hartree | ORCA absolute difference / hartree | Independent Fock maximum error | Independent energy error / hartree |
| --- | ---: | ---: | ---: | ---: |
| RHF | -56.18855210065030 | 7.0330e-10 | 3.2530e-14 | 8.5266e-14 |
| UHF | -56.18855210065043 | 7.0343e-10 | 2.8644e-14 | 5.6844e-14 |

The new energies differ from the earlier accepted reduced SCFs by at most
7.106e-14 hartree. Applying the corrected maps to the archived densities
reproduces their Focks within 3.198e-14 and energies within 1.279e-13 hartree.
Full J/K map reconstruction maximum errors are 2.718e-14/2.767e-14; complete
source-audit Frobenius errors are 2.263e-13/2.678e-13. The independent full
composed tensor differs by at most 2.841e-14. Arbitrary nonsymmetric complex
trial J/K error is 4.711e-14; covariance error over all six operations is
8.632e-14 with the metric-aware AO density push and operator pull. The basis
Gram and inverse residuals are 1.770e-13 and 2.777e-13. A converged symmetric
state is therefore not the only evidence of equivariance.

| Measured setup component | Seconds |
| --- | ---: |
| Geometry and irrep table | 0.01300 |
| Orbital metric and actions | 0.00633 |
| Pair resolution, group-law/partner audits and inverse | 12.78148 |
| Full ERI generation | 0.03365 |
| Four-index basis transport | 0.01858 |
| Complete J/K source qualification, including reconstruction | 1.49519 |
| Total measured setup | 14.34834 |

RHF/UHF iterative SCF took 0.23568/0.28178 s, including every input/output
basis transport and reconstructed AO J/K. Setup plus those SCF segments is
14.58402/14.63012 s per fresh source. These sums exclude separate one-electron
setup and acceptance diagnostics; the complete acceptance process took
24.46 s, including imports, both SCFs and all comparisons. No same-host dense
SCF timing was repeated, so the historical dense timings above are not a
current speed ratio.

RHF executed 80 J/K map applications, 15,920,560 reduced contraction products
and 120,969,440 basis-transport products. UHF executed 99 applications,
19,701,693 contraction products and 149,699,682 transport products. These
counters cover the timed SCF; later independent validation work is separate.
The complete 707,281-value ERI source was still generated and audited. Its
producer shell-call count was not instrumented here, and no integral-generation
or source-audit work reduction is claimed.

Six alternating paired composition timings, after one warm-up, give median
reduced contraction 0.01703 s, full contraction 0.17443 s and full reconstruction
0.35359 s. The median of reduced contraction plus its reconstruction is
0.37066 s. The actual product counts are 28,776,508 versus 594,823,321, but
reconstruction alone exceeds the saved contraction time. Setup further adds
the costs above. This is not end-to-end acceleration or a production routing
recommendation; a useful consumer must remain reduced across multiple steps
and still pay for its eventual full outputs.

Each map stores 121,166 complex reduced entries (1,938,656 bytes), versus
707,281 full entries. The basis and actual inverse each occupy 11,316,496
bytes. Pair-frame retained logical storage is 22,654,348 bytes; each map's
24,597,484-byte retained reservation includes that shared frame, so two maps
sharing it total 26,540,620 bytes, not twice the per-map figure. Resolution
and per-map qualification reservations are 497,959,212 and 158,468,684 bytes.
The complete acceptance process peaked at 439,824,384 bytes RSS. These include
reference and audit storage; they do not establish an incremental memory
saving over an unreduced production pipeline.

The inspected QC0.18 candidate (`2b2ed883`) contains the experimental module
but no included production route imports or calls it. MANTA's bounded release
controls are unreduced; #635 explicitly retains producer certification.
No release regression attributable to this layer was identified. The finite
algebra and this correction do not expand #557 or gate the release. The
[0.18 scope](release_018_scope.md#explicit-exclusions) also explicitly excludes
symmetry-reduced MANTA variants from the frozen control profiles. Existing
molecular quartet/RI routes and their accepted evidence remain separate.

After release, continue producer qualification under #635 with shared
molecular consumer #608 and periodic parity #577, then measure a consumer
that can remain in reduced coordinates. Require support, screening/bubble
closure, auxiliary metric/rank and arithmetic evidence before avoiding a
full source audit. Optimize complete workloads under #638, counting setup,
AO and coupled-basis transport, reconstructed outputs and peak memory.
Correlation-specific projectors, spin/frozen spaces and full-mesh admission
remain #589 obligations. Antiunitary/projective actions, distinct CCM/chi
sources, general CC residuals, gradients and stress are still outside this
ordinary finite layer. No operation-count ratio establishes end-to-end speedup.


### Native MD molecular quartet source

This increment adds no new group mathematics. It connects MANTA's immutable
native MD full-Coulomb producer to the existing complete finite-source audit,
Dupuis-King shell transport, stabilizers, fixed whole-orbit screening and native
batched J/K contractions. Molecular RHF/UHF consume it through the existing
JKBuilder and shared FockBuild interfaces. MD source/Fock identities remain
distinct from libint, and every ordered tile is audited before representatives
are admitted. No libint Q/V or hierarchy certificate transfers to native MD.

This is experimental low-level source admission. Public prepared-source,
run_job and QVF symmetry selection still use the existing libint route. The
ordinary non-Abelian product algebra above remains a separate experimental
consumer; MD shell scheduling does not implement reduced CC. There is no new
periodic, complex-integral, antiunitary, fitting, range-kernel, gradient or stress
admission, and no QC0.18 dependency or release prerequisite.

Thirty-eight affected checks passed, with two existing stored-only admission
checks skipped for the streaming parameterization. They cover both schedules,
native/NumPy execution, water C2v, ammonia C3v, pure and Cartesian s+d actions,
independent ordered-tile J/K, nonsymmetric complex trial densities, independent
spin densities, terminal capture and cold fallback. Wrong kernels, radial
sources, budgets and foreign producer bounds refuse before evaluation.
No scientific tolerance changed. Local dependency problems stopped earlier
attempts before tests; the corrected native build and its dependency identities
are preserved with the successful evidence, rather than counted as test passes.

The complete vq calculation was accepted on 2026-10-26 using native source base
`323fe4db`, the byte-identical four-module adapter payload and the same fixture,
basis, charge/spin, HCORE guess and convergence settings as the preceding
ammonia acceptance. All nine input hashes and 416 loaded module identities were
recorded; native image and initially loaded modules stayed unchanged. Native
image, queue, payload, reference and output provenance are archived outside
the product tree. The earlier accepted ammonia control and matched ORCA energy
-56.188552099947 hartree were reused; neither was recalculated.

| Method | Energy / hartree | ORCA absolute difference / hartree | Same-source dense SCF difference / hartree | Independent final Fock maximum error |
| --- | ---: | ---: | ---: | ---: |
| RHF | -56.18855210065024 | 7.0324e-10 | 0.0 | 2.3315e-14 |
| UHF | -56.18855210065038 | 7.0338e-10 | 5.6843e-14 | 2.5313e-14 |

Both methods converged in 15 iterations at unchanged energy/gradient thresholds
1e-11/1e-9. Independent final-density energy errors were at most 1.4211e-13
hartree; differences from the earlier accepted SCFs were at most 1.2790e-13.
The complete 707,281-value MD tensor differed from independently generated
libint ERIs by at most 8.8818e-15. Arbitrary nonsymmetric complex J/K
reconstruction differed from the dense same-mask tensor by 1.5106e-14 and
from ordered MD execution by 9.9301e-15. Complete source and stabilizer
residuals were 1.6490e-14 and 3.0958e-15. These are finite numerical witnesses,
not producer arithmetic proofs or evidence based only on a symmetric SCF state.

| Executed work for the 29-AO, 15-shell source | Full qualification | Representative J/K | Same-mask ordered J/K |
| --- | ---: | ---: | ---: |
| Shell calls | 50,625 | 1,763 | 49,757 |
| Primitive quartets requested | 1,874,161 | 104,454 | 1,755,773 |
| Integral values received | 707,281 | 29,963 | 686,605 |

The fixed 1e-12 Frobenius omission cap removed 130 representatives covering
868 ordered tiles and 20,676 scalar entries. Their measured finite-source norm
was zero; this is not an analytic-zero certificate. J/K still performed
1,373,210 scalar contraction products, equal to ordered execution, plus
712,318 angular-transport products. The full audit was paid before any of
these representative savings. No pre-audit screening bound was used, and the
omission receipt still excludes arithmetic roundoff.

| Measured cost | Seconds |
| --- | ---: |
| Geometry/basis and MD producer | 0.00162 |
| Complete source audit, shell actions, stabilizers and mask | 1.24526 |
| One-electron source, metric and shared frame | 0.00038 |
| Total setup | 1.24726 |
| Complex representative J/K including transport and reconstruction | 0.07902 |
| Complex same-mask ordered J/K | 0.52103 |
| RHF / UHF iterative SCF, including all AO reconstructions | 8.05727 / 6.89421 |
| Fresh setup plus RHF / UHF SCF | 9.30453 / 8.14147 |
| Captured terminal FockBuild, RHF / UHF | 0.00045 / 0.00064 |
| Cold terminal FockBuild, RHF / UHF | 0.10704 / 0.14938 |

Captured terminal builds used zero additional integral calls; cold fallback
used 1,763 calls and reproduced the identical physical build identity. Fresh
setup, SCF and captured terminal construction total 9.30498/8.14211 s. These
sums charge setup separately to each method; the acceptance process actually
shared one qualified source. Its independent ordered MD tensor construction
cost 0.39451 s, and dense same-tensor RHF/UHF controls took 0.04077/0.03882 s.
The small dense controls are much faster once their tensor is stored. These
single-run measurements do not establish production-scale end-to-end speedup.

RHF executed 92 build calls, 162,196 shell calls, 9,609,768 primitive quartets,
73,466,735 contraction products and 65,533,256 transport products. UHF executed
61 builds, 107,543 shell calls, 6,371,694 primitive quartets, 63,854,265
contraction products and 43,451,398 transport products. Qualification,
independent references and terminal checks are additional work, not hidden
inside those SCF counters.

A separate producer-specific routing calibration cost 1.49264 s and selected
the native representative executor for 20 expected J/K builds. Its ordered
samples cost 0.66972/0.65200 s and representative samples 0.08724/0.08328 s,
with maximum numerical difference 1.8392e-14. Calibration is excluded from the
explicit-native SCF totals above and included in the complete acceptance cost.
It compares those two admitted executors only; it is not evidence against a
stored-tensor production route or an uncertainty-qualified speed benchmark.

Factory logical reservation was 12,821,228 bytes and plan reservation
11,039,572 bytes. Compact plan arrays retained 3,920 bytes plus a 237-byte mask;
these exclude producer and Python object storage. Maximum RHF/UHF consumer
logical reservations were 11,595,492/11,757,220 bytes. The complete acceptance
process, including dense references, reached 304,431,104 bytes RSS and took
19.04201 s after imports (20.05322 s measured command wall time). Logical
reservations and measured process memory are distinct quantities.

Further work remains under #635 with molecular Fock #608, periodic parity #577
and reduced-pipeline optimization #638. MANTA must issue operation-bound
support, screening/bubble and arithmetic evidence before a source can avoid
the complete audit. Auxiliary metric/rank and retained-projector transport,
periodic supports/phases, derivatives and correlation require separate
qualification. No release route or default dispatch is expanded by this result.


### Streamed quartet metadata cost

Profiling the accepted native-MD ammonia path identified image expansion as
about two thirds of profiled J/K call time. The old iterator expanded every
orbit before consulting the fixed mask and constructed an image object even
when a stabilizer duplicate already occupied that dictionary key. This
increment separates representative enumeration from expansion, selects the
already qualified mask first, and creates only unique image descriptors using
the authoritative real-ERI permutation order. It changes no integral kernel,
source admission, tensor transformation or floating-point accumulation order.

Seventy-four affected checks passed, with four expected streaming-mode skips.
Coverage includes independent complete S1-S4 quartet partitions, identity and
non-Abelian actions, stored/streamed bit identity, exact resource caps,
quadratic retained storage, no expansion of omitted orbits, empty/full masks,
complex arbitrary-density reconstruction, late source-digest failure, and
ordered-only producer-bound refusal of angular reconstruction. Forty-eight
passes from the first run were reused; sixteen checks initially stopped on an
unrelated MANTA facade/native export mismatch. Those sixteen passed with the
matching facade from the reused native revision. No scientific check or
scientific tolerance was bypassed, and no unrelated native rebuild was needed.

Complete paired NH3/cc-pVDZ RHF/UHF calculations ran through vq on the same
native `323fe4db` image with the checked Python overlay and its matching MANTA
facade. The old scheduler/batch implementation was loaded from the source
baseline for in-process comparison, with original methods restored between
variants. Fifteen payload hashes, 419 loaded module identities and native image
identity were retained outside the product tree. The earlier ORCA reference
and finite-Gram acceptance were reused, not recalculated. All four SCFs
converged in 15 iterations at unchanged 1e-11/1e-9 thresholds.

Both versions produced bit-identical J/K, final densities, physical Fock
matrices, evaluated-state identities and energies. Source/audit identities,
mask receipts, complete qualification counts and physical build counters were
also identical. RHF/UHF energies were -56.18855210065024/-56.18855210065038
hartree, within 7.0338e-10 of the matched ORCA reference. Independent final
Fock errors were 2.3315e-14/2.5313e-14; energy errors were at most 1.4211e-13
hartree. Nonsymmetric complex trial J/K error was 1.0396e-14.

| Measured component | Previous scheduler / s | Selected unique-image scheduler / s |
| --- | ---: | ---: |
| Complete source setup and audit | 1.21637 | 1.21114 |
| Median J/K build, eight alternating pairs | 0.10296 | 0.08460 |
| RHF iterative SCF | 9.73863 | 7.45541 |
| UHF iterative SCF | 6.43223 | 4.96993 |
| RHF setup, SCF and captured terminal build | 10.95905 | 8.67055 |
| UHF setup, SCF and captured terminal build | 7.65273 | 6.18494 |
| RHF cold terminal build | 0.10892 | 0.07087 |
| UHF cold terminal build | 0.10980 | 0.10942 |

The complete-method sums include geometry/basis (0.00151 s), one-electron
source/metric (0.00209 s), full qualification, every AO transport and
reconstruction, and physical terminal construction. Independent dense-source
assembly took another 0.37424 s and is diagnostic work outside those sums.
The whole paired acceptance process, including profiles and reference checks,
took 34.56869 s after imports. These are observed same-process costs, not a
production crossover: eight build samples span 0.07648-0.14516 s before and
0.07232-0.10470 s after, with two pairs slower after the change. Each complete
SCF variant was timed once. The preceding dense controls remain much faster
once their tensor is stored; no general acceleration over an unreduced method
or uncertainty-qualified speed ratio is claimed.

Instrumented orbit expansions fell from 1,893 to 1,763 per retained J/K build;
the omitted 130 orbits no longer allocate image descriptors. All stabilizer
bits and first-occurrence member order remain intact. Full qualification still
makes 50,625 shell calls and requests 1,874,161 primitive quartets. Each
representative J/K still makes 1,763 calls, requests 104,454 primitive quartets,
and performs 1,373,210 contraction plus 712,318 transport products. RHF/UHF
SCF counters are unchanged from the native-MD acceptance above. This is actual
metadata work avoided, not new integral, contraction or audit savings.

The conservative plan reservation remains 11,039,572 bytes, with 3,920 bytes
of retained plan arrays and a 237-byte mask; no quartet inventory or evaluated
tile cache was added. RHF/UHF consumer reservations remain
11,595,492/11,757,220 bytes. The combined before/after process, holding both
qualified plans and its references, peaked at 259,194,880 bytes RSS. That is
not a per-route memory reduction measurement. Shared Fock and PBC interfaces,
defaults and experimental/source-qualification boundaries are unchanged.
Producer arithmetic/support/bubble certification under #635 and larger reduced
pipelines under #638 remain separate obligations, with no new release gate.


### Compiled single-orbit expansion

The next bounded #638 increment moves selected streamed orbit expansion into
the existing native quartet accumulator. It does not change integral kernels,
source admission, the fixed mask, metric-aware AO transport or floating-point
member order. Source qualification still performs the complete Python orbit
and numerical audit. The checked Python-image fallback remains available for
older native runtimes. No new source certificate or release dependency follows.

A matching native rebuild and 92 affected checks passed. Independent S2-S4
partitions exercise identity, permutation and spatial modes with reordered
operation IDs; compiled and explicit image paths give bit-identical results
and agree with independent dense tensors for nonsymmetric complex densities.
Coverage also includes pure/Cartesian molecular actions, selected J/K channels,
exact resource caps, immutable source snapshots, malformed descriptor poisoning,
and refusal to export partial results. A regression forbids Python image
expansion during compiled execution and then verifies the older-runtime
fallback produces identical J/K and work counts.

The paired NH3/cc-pVDZ acceptance ran through vq with a native build from
`78bddc98` plus the checked expansion changes. Build input hashes were stable,
the native API was required, and payload/module/native identities were retained
outside the product tree. Both implementations ran in one process against that
same native image. The baseline Python scheduler/batch came from `78bddc98`.
The earlier ORCA and finite-Gram results were reused. All four RHF/UHF runs
converged in 15 iterations at unchanged 1e-11/1e-9 thresholds.

J/K, final densities, physical Focks, evaluated-state identities and energies
were bit-identical. Audit identities, screening receipts and all physical work
counters also matched. RHF/UHF energies were
-56.18855210065024/-56.18855210065038 hartree, within 7.0338e-10 of ORCA.
Independent final AO Fock errors were 2.3315e-14/2.5313e-14; independent energy
errors were at most 1.4211e-13 hartree. Nonsymmetric complex trial J/K agreed
with the complete masked ordered tensor within 1.0396e-14. This execution
increment leaves the separately accepted finite-Gram inverse conversion and
its full reconstruction evidence above unchanged.

| Measured component | Python images / s | Compiled images / s |
| --- | ---: | ---: |
| Complete source setup and audit | 1.14598 | 1.14517 |
| Median J/K build, eight alternating pairs | 0.05921 | 0.03033 |
| RHF iterative SCF | 5.34474 | 2.71647 |
| UHF iterative SCF | 3.56868 | 1.80578 |
| RHF setup, SCF and captured terminal build | 6.49387 | 3.86472 |
| UHF setup, SCF and captured terminal build | 4.71779 | 2.95406 |
| RHF cold terminal build | 0.05937 | 0.02946 |
| UHF cold terminal build | 0.06001 | 0.03066 |

Complete-method sums include geometry/basis (0.00153 s), one-electron
source/metric (0.00134 s), full qualification, all AO transport and J/K
reconstruction, and captured physical terminal construction. Cold terminal
rebuilds are additional diagnostics, not hidden inside those sums. Independent
dense-source assembly took 0.36714 s. The paired process, including profiles
and comparisons, took 17.53451 s after imports. Build samples span
0.05886-0.06431 s before and 0.03005-0.03171 s after; all eight pairs were
faster after. Each complete SCF variant was timed once. This is a measured
comparison between two executions of the same admitted source, not evidence
of a production crossover against a stored-tensor or direct native method.
Historical timings on other runtimes or loads are not a current speed ratio.

Instrumentation records 1,763 compiled expansions and zero Python expansions
per retained build, versus 1,763 Python expansions before. The 130 omitted
orbits remain unexpanded during execution. Complete qualification still makes
50,625 shell calls and requests 1,874,161 primitive quartets. Each J/K build
still makes 1,763 calls, requests 104,454 primitive quartets, reads 29,963
values, and performs 1,373,210 contraction plus 712,318 transport products.
RHF/UHF SCF counts remain 92/61 builds, 162,196/107,543 shell calls,
9,609,768/6,371,694 primitive requests, 73,466,735/63,854,265 contraction
products and 65,533,256/43,451,398 transport products. No integral or source
audit reduction is attributed to compiled metadata.

The logical plan reservation remains 11,039,572 bytes; retained arrays and
mask remain 3,920 and 237 bytes. RHF/UHF consumer reservations remain
11,595,492/11,757,220 bytes. One orbit's temporary native map, set and rows
use the existing metadata reservation; there is no persistent quartet cache.
The combined paired process peaked at 260,440,064 bytes RSS, including both
qualified plans and reference tensors. That is not a per-route memory saving.

The new profile places most remaining build cost in producer tile reads and
their checked Python boundary. Further batching or producer sessions must
preserve snapshots, source digests, counters and finite-source qualification;
they must not duplicate MANTA kernels or infer arithmetic certification from
timing. #635 retains operation-bound support, screening/bubble and arithmetic
evidence; #638 retains reduced-pipeline cost optimization. Fock #608 and PBC
#577 interfaces and source/state/subspace distinctions remain unchanged.
General non-Abelian CC, periodic producer admission, gradients and stress are
still outside this experimental molecular increment and do not gate QC0.18.
