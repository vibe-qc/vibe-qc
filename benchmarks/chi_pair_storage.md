# Bounded selected and pair coefficient storage

Run `.venv/bin/python -m benchmarks.chi_pair_storage` from the source checkout
with development test dependencies. The ladder reuses the nonorthogonal,
occupied-coupled frozen-core finite model from the source tests: four primitive
AOs, one active and one frozen occupied band, full domains and zero PNO cutoff.
Model/source construction precedes measurement. No physical QC acceptance is
claimed. The script emits actual source/native runtime identities through
`vibeqc.output`.

Historical D14 coefficient-storage run (seconds are observations, not performance promises):

| Cells | Selected bytes dense / compact | Pair AO bytes dense / compact | Recipe bytes both | Solve/preparation seconds dense / compact | Traced peak bytes dense / compact |
|---|---|---|---|---|---|
| 1 | 128 / 64 | 64 / 0 | 128 | 0.0105 / 0.0091 | 211582 / 64123 |
| 2 | 512 / 128 | 512 / 0 | 896 | 0.0113 / 0.0146 | 62596 / 67279 |
| 4 | 2048 / 256 | 3072 / 0 | 4992 | 0.0214 / 0.0282 | 101368 / 92032 |
| 8 | 8192 / 512 | 20480 / 0 | 32000 | 0.0592 / 0.0839 | 302306 / 220086 |

Selected bytes count unique buffers in the returned automatic result: dense
selection and pair spaces have separate snapshots; compact storage shares
home columns. Recipe bytes include indices, local A and occupied-tail B.
Recipes are shared by pair storage and build metadata. At eight cells this
scope falls from 60672 to 32512 bytes. The full-domain recipe itself is larger
than a single AO pool; the improvement removes its duplicate dense pool.
Native pair-AO input bytes are zero in compact mode. Recipe-only transport
accounts an extra one-pair reconstruction rectangle: required native bytes
at eight cells are 210024 versus 205928 for dense coefficient inputs. The
pre-existing native cap excludes binding copies of other inputs.

Largest raw-energy difference is below `3e-17` Ha/cell. Every solve converges;
maximum final residual is below `2.6e-13`. PNO corrections are reported
separately by the script. Complete pair reconstruction defects in this model
are below `1.1e-31`; other nonorthogonal/truncated and ill-conditioned cases
are covered by source tests at the unchanged `1e-10` admission tolerance.

Tracemalloc includes Python overhead and cold imports and does not measure
all native allocations or process RSS. No repeated-run timing statistics,
total-memory bound, linear scaling or production accuracy follows from this
small ladder. Selector eigensystems, transient full selected frames, source
storage, full-domain Gram matrices, amplitudes and transport links remain.
Scientific acceptance through a containing vq runtime remains separate.

## Bounded native links (#585)

The current script measures the prepared and bounded transport backends with
compact recipes. Run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` and
`--cells 1 2 4 8 16 32`. Pair construction and its initial bounded snapshot
are measured separately; the transport columns below reuse that same build.
Each backend prepares at its exact accounted cap and refuses one byte less.
When bounded requires less than prepared, the script also runs the complete
automatic driver at the bounded cap and verifies prepared transport refuses.
No full numerical graph is constructed in that bounded driver run.

Baseline before target-metric reuse: single-run timings on an arm64 development runtime, release 0.17.26;
no repeated-run statistics or speed guarantee. The native core SHA-256 was
`3ccbbdb50ccf460fcd9ec33f508a93cd6ce752a3da8a318d34afc3d6a8727be0`.
Final binding-only preflight guards were checked separately; these timings
reuse the unchanged transport kernels. Source tests and native compilation
are implementation evidence, not vq
scientific acceptance. Traced values measure Python allocations, not Eigen
payload, library packing scratch or total process RSS.

| Cells | Backend | Prepare s | Iterate s | Retained bytes | Accounted cap bytes | Prep traced peak bytes | Iteration traced peak bytes | Link reconstructions / hits |
|---|---|---|---|---|---|---|---|---|
| 1 | prepared | 0.0016 | 0.0011 | 544 | 2416 | 15994 | 16346 | 0 / 0 |
| 1 | bounded | 0.0015 | 0.0009 | 1252 | 10784 | 16106 | 15202 | 1 / 3 |
| 2 | prepared | 0.0024 | 0.0013 | 2416 | 5968 | 19103 | 22999 | 0 / 0 |
| 2 | bounded | 0.0017 | 0.0013 | 3468 | 24232 | 19113 | 22477 | 56 / 8 |
| 4 | prepared | 0.0045 | 0.0020 | 15792 | 25024 | 24520 | 31132 | 0 / 0 |
| 4 | bounded | 0.0039 | 0.0046 | 12492 | 81016 | 24846 | 31050 | 168 / 24 |
| 8 | prepared | 0.0141 | 0.0037 | 171760 | 201952 | 44373 | 76522 | 0 / 0 |
| 8 | bounded | 0.0137 | 0.0139 | 64716 | 388504 | 44085 | 76438 | 657 / 63 |
| 16 | prepared | 0.0628 | 0.0142 | 2355312 | 2465824 | 158429 | 399035 | 0 / 0 |
| 16 | bounded | 0.0650 | 0.3367 | 413388 | 2316760 | 158653 | 398851 | 2730 / 150 |
| 32 | prepared | 0.5579 | 0.1353 | 35368816 | 35793568 | 846381 | 2882275 | 0 / 0 |
| 32 | bounded | 0.5156 | 9.8217 | 2947788 | 15727192 | 846613 | 2880251 | 10570 / 310 |

Prepared and bounded results in this ladder have identical raw energies,
residuals and separately reported PNO corrections. The one-entry bounded cache
is empty after each solve; it does not persist on the reusable payload.
Prepared retains its numerical links. Native admission still computes and
checks every link once in both modes, so preparation does not promise a
speedup. At this baseline stable link reconstruction repeats target selected-column
and occupied-tail work during iteration; this is a substantial regression when memory permits
the prepared backend. Gamma timings are too small to infer an improvement.

| Cells | Numerical links prepared / bounded | Metadata bytes | Recipe snapshot bytes bounded | Existing occupied Fock bytes | Existing projected recipes bytes | One reconstructed pair bytes | Solver returned arrays bytes |
|---|---|---|---|---|---|---|---|
| 1 | 32 / 0 | 416 | 740 | 8 | 128 | 64 | 64 |
| 2 | 768 / 0 | 880 | 1820 | 32 | 896 | 256 | 512 |
| 4 | 9728 / 0 | 1968 | 6428 | 128 | 4992 | 1024 | 3072 |
| 8 | 141312 / 0 | 5872 | 34268 | 512 | 32000 | 4096 | 20480 |
| 16 | 2170880 / 0 | 20592 | 228956 | 2048 | 225792 | 16384 | 147456 |
| 32 | 34111488 / 0 | 77680 | 1690460 | 8192 | 1688576 | 65536 | 1114112 |

At 16 cells the bounded complete driver converged at 2316760 bytes in 4.430 s (Python traced peak 1003951 bytes); prepared transport refused that cap.

At 32 cells the bounded complete driver converged at 15727192 bytes in 60.190 s (Python traced peak 6591431 bytes); prepared transport refused that cap.

Bounded snapshots and expanded accounting can require a larger cap on small
meshes despite retaining fewer numerical links. The prepared cap retains its
historical exclusion of binding/Python copies, while bounded explicitly
charges them. These scopes must not be mistaken for total-memory comparisons.
Full-domain recipe matrices, Fock blocks, routing metadata and solver arrays
still grow; removal of numerical-link retention does not make the entire
calculation memory bounded independently of mesh size. The next section measures bounded per-target metric reuse. Routing metadata,
amplitude and projected-domain storage remain separate work.


## One target metric reused within the solve (#585 follow-up)

The same command and fixtures were measured after adding the solve-local
single-entry target-metric cache. The rebuilt arm64 native core SHA-256 is
`cfd6643562416aa956f38fa4821693e05cca8a985220cca3072b5aa5cd02f170`.
These are single-run observations, not statistical speed guarantees; Gamma
and small-mesh timings remain dominated by overhead and run-to-run variation.
The script now requires bit-identical prepared/bounded amplitudes, residuals,
raw energies and separately reported PNO corrections, as well as a bound of
one target construction per visited target row. Exact/below-cap checks pass.

| Cells | Prepared iterate s | Bounded prepare s | Bounded iterate s | Bounded accounted cap bytes | Target cache reservation bytes | Target constructions / hits | Prep / iteration traced peak bytes |
|---|---|---|---|---|---|---|---|
| 1 | 0.000793 | 0.001129 | 0.000879 | 10912 | 120 | 0 / 0 | 16206 / 15480 |
| 2 | 0.001653 | 0.001498 | 0.001458 | 24552 | 312 | 0 / 0 | 19213 / 22611 |
| 4 | 0.001810 | 0.004510 | 0.002452 | 82104 | 1080 | 0 / 0 | 24948 / 31234 |
| 8 | 0.003170 | 0.011116 | 0.012397 | 392664 | 4152 | 0 / 0 | 44085 / 76550 |
| 16 | 0.011954 | 0.056244 | 0.111154 | 2333208 | 16440 | 90 / 2640 | 158621 / 398755 |
| 32 | 0.199990 | 0.444103 | 2.172412 | 15792792 | 65592 | 170 / 10400 | 846245 / 2879427 |

At 32 cells, target constructions fall from one per stable link reconstruction
(10570 in the baseline) to 170, with 10400 target hits and 169 evictions.
Link construction counts stay at 10570: no coupling is omitted. Observed
bounded iteration falls from 9.8217 to 2.1724 s, about 4.5 times faster, but
remains about 11 times slower than the prepared oracle in the new run.
At 16 cells, target constructions fall from 2730 to 90 and iteration from
0.3367 to 0.1112 s. Smaller fixtures select the nonstable branch, so target
counts are zero and no target-reuse speedup is claimed.

The 32-cell cache reserves 65536 numerical bytes plus its 56-byte native
descriptor. A new payload counter adds another eight retained bytes; bounded
retention is 2947796 bytes and the complete transport reservation is 15792792
bytes. Numerical link retention remains zero. The extra reservation is
conservative even when the stable branch is never used. It covers one target
coexisting with link/source/contraction work; old targets are freed before
replacement. No pool of target metrics is constructed.

At the exact bounded cap the complete 16-cell driver converges in 3.7441 s
(Python traced peak 1006687 bytes), and the 32-cell driver in 48.2411 s
(6586603 bytes). Prepared transport refuses those caps. The maximum final
residual across the ladder is 2.55e-13. Tracemalloc still excludes Eigen
allocations and external scratch: reservations and these traced peaks are
not total-process-memory guarantees. Prepared retention/caps increase by
eight descriptor bytes; existing Fock, projected recipes, single-pair and
solver-returned numerical arrays retain the baseline sizes above.

254 focused transport/chi checks pass, including independent same-source
canonical limits, ill-conditioned asymmetric dense and cyclic metrics,
active/frozen partitions, eviction, caller mutation, failure recovery,
exact/below-budget refusal and Python AO-reconstruction guards. The largest
canonical error is 5.52e-15 Ha/cell. These are development source checks;
scientific acceptance through a containing vq runtime remains separate.
The next transport gaps are repeated source-pair and route reconstruction
and retained routing metadata. Occupied Fock, amplitudes and projected-domain
storage remain separate work; chi remains experimental and D93 fitted/post-HF
not-qualified.
