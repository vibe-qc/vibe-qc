# Reusable periodic RI Gaussian preparation

The ordered finite GDF producer now owns bounded, lazily published Gaussian
extents, moment panels and support-count scalars. Preparation is shared across
auxiliary shells, forward/adjoint passes and subsequent SCF builds. Both paired
actions keep their existing numerical map, finite metric and source identity.
The shared image selector, bubble tree, erfc ledger and Cartesian moment
functions remain the producers of the numerical data.

## Reproduction and measurement

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/manta_periodic_ri_preparation.py --workers 3 --repeats 3
```

Three separate processes each measure the first build and three subsequent
builds on arm64 with one numerical thread. The fixture has two k points, two
AO functions and six auxiliary functions with full retained rank six. Every
RI build uses two coupling passes. The medians below compare streaming and
reused preparation on the same auxiliary-image hierarchy. Cold times include
preparation on first use; builder construction and metric factorization are
outside both timings. The JSON retains source/native hashes, original Git
state, all samples, counters and peak process memory.

| Path | Order | Coupling epsilon | Cold ms | Warm ms | Cold / warm new panels | Cold / warm envelope evaluations | Modeled peak bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 0 | 0 | 23.632 | 22.867 | 0 / 0 | 0 / 0 | 25478848 |
| streaming tree | 0 | 1 | 88.066 | 85.546 | 400 / 400 | 11276 / 11276 | 40356912 |
| reused tree | 0 | 1 | 83.884 | 83.109 | 36 / 0 | 1053 / 0 | 43573760 |
| streaming tree | 2 | 100 | 114.083 | 113.523 | 408 / 408 | 11276 / 11276 | 44394144 |
| reused tree | 2 | 100 | 111.253 | 111.574 | 38 / 0 | 1053 / 0 | 48195392 |
| streaming tree | 4 | 100 | 236.025 | 235.052 | 394 / 394 | 11276 / 11276 | 54085216 |
| reused tree | 4 | 100 | 230.225 | 228.370 | 37 / 0 | 1053 / 0 | 58844880 |

Reuse removes all Gaussian envelope and moment construction from the warm
paired build. The cold cached cases construct only 36-38 distinct moment
panels and perform 1,053 envelope evaluations. Warm builds reuse 394-408
panels and 432 extents. The extra worst-case storage reservation is about
3.2-4.8 MB in this fixture; it is admitted before the source copies inputs.
`coupling_multipole_reuse=False` preserves the lower-storage streaming path.
The cache is disabled for exact and reciprocal-only sources.

Warm wall time improves only about 2-3% here. Node construction, translated
node envelopes, erfc error queries and contractions remain the dominant work.
All approximate paths remain slower than direct evaluation for this small
case. This is useful preparation reuse, not a demonstrated production speedup
or linear-scaling result.

The numerical results and bounds are unchanged by reuse. Maximum J errors
for orders zero, two and four are 1.786e-6, 3.684e-8 and 2.035e-12; their
reported bounds are 5.613e2, 4.136e5 and 8.461e5. The finite metric and
retained factor are bit-identical to the direct source. These deliberately
conservative bounds exclude arithmetic/phase/factor roundoff, finite-domain
and basis error, and density relaxation.

## Focused checks and remaining scope

507 scientific source checks pass: 21 preparation cases, 57 existing hierarchy
and leaf-multipole cases, and 429 shared raw/paired RI, metric, Fock and public
job compatibility cases. The preparation checks include bitwise
cold/warm/streaming/dense equality, concurrent first publication, independent
source lifetimes, read-only diagnostic snapshots, lazy always-near behavior,
every resource-cap failure before numerical input inspection, grouped-node
RHF/UHF SCF and RKS/UKS using the existing XC provider.

Only Gaussian preparation and integer support counts are retained. There is
no full T/B tensor or auxiliary-by-pair image interaction list. Cross-pass
tree/partition reuse, global density-compressed upward/downward actions,
automatic whole-method accuracy and source-qualified symmetry remain open.
Exact exchange, Madelung/nuclear terms and XC keep their existing owners;
this source grants no new BIPOLE/CRYSTAL or symmetry qualification. The full
MANTA 0.18.x goal remains unfinished.


## Complete calculation acceptance

An actual local vq submission passed on implementation commit
`e9da9ab27482c0fb69bf3f0f18024c692d8035cb`, using native SHA-256
`89e8f8af0d660302872942d05ddfdfd58c4ee64be14cbdfa1582d158563caafe`.
The run verifies 1,603 source/reference pins before and after execution and
63 managed output hashes. A separate NumPy-only assessment of the fetched
arrays passes. The benchmark uses this same native runtime and matching
source content; its original pre-commit provenance is retained.

Seven complete public states cover shifted-mesh Be RHF, empty-beta H UHF,
converged and capped Li UHF, Be PBE RKS, and converged and capped Li PBE UKS.
All four HF cases use the 12-bohr auxiliary cutoff and accept multi-image
nodes. Be RHF and Li UHF represent 1,124 images in 1,102 far blocks, with
2,376 nodes and 1,168 upward translations across the paired Fock passes.
Their final builds reuse 108 moment panels and 108 extents; all seven final
builds perform zero new Gaussian moment or envelope evaluations. Four raw
complex Gamma/nonzero-q cases at orders zero/two also pass and are bitwise
identical with cold, warm and streaming preparation.

Maximum errors against separately contracted and independently relaxed
references are 4.969e-15 for Fock elements, 9.238e-14 Eh for same-density
energy, 8.705e-14 Eh for relaxed energy and 1.519e-9 for density elements.
Gates remain 8e-11 for Fock/energy and 5e-9 for density. Raw action and
duality errors are below 5.1e-15. The largest same-density changes from
replacing exact finite coupling are 1.482e-8 in Fock elements and
1.787e-7 Eh in energy, within the reported approximation bounds.
Shared Fock application, energy bookkeeping, unchanged exact exchange and
Madelung ownership, XC and saved-density consistency all pass. Capped states
remain nonconverged and export their evaluated density.

No matched ORCA or CRYSTAL calculation is available for these custom finite
domains. References use independent finite Gaussian/Boys/Fourier formulas,
libint erfc integrals, NumPy metric contractions and independent SciPy SCF;
KS reference contractions use the same libxc functional. The approximate
partition is not independently reimplemented. This qualifies the bounded
finite-source change and does not establish external-program parity.
