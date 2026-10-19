# Ordered periodic RI auxiliary-image hierarchy

This opt-in increment groups the auxiliary images actually selected by the
ordered GDF capsule, then translates their phase-weighted Cartesian moments
upward. It reuses the native bubble tree, radial envelopes, moment translations
and isolated erfc error ledger. Both existing paired actions consume the same
tiles. The finite metric and retained factors remain bit-identical to the
direct finite source; exact exchange, reciprocal panels and zero terms retain
their existing owners.

## Reproduction and measurement

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/manta_periodic_ri_hierarchy.py --workers 3 --repeats 3
```

Three separate processes, each with three warm build measurements, on arm64
with one numerical thread. The fixture has two k points, two compact AO
functions and six auxiliary functions, full retained rank six and two coupling
passes. Values compare with the same finite direct-coupling metric and factor.
The benchmark exports native/source identities and process memory as JSON.

| Path | Order | Requested coupling epsilon | Median ms | Error queries | Accepted blocks | Represented images | Max J error | J bound | Modeled peak bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 0 | 0 | 23.350 | 0 | 0 | 0 | 0.000e+00 | 0.000e+00 | 25478848 |
| leaf | 0 | 1 | 95.213 | 14706 | 0 | 13716 | 1.786e-06 | 4.715e+02 | 25888424 |
| tree | 0 | 1 | 82.471 | 11010 | 9930 | 13716 | 1.786e-06 | 5.613e+02 | 40356912 |
| leaf | 2 | 100 | 110.534 | 14706 | 0 | 13656 | 3.684e-08 | 1.739e+05 | 25916432 |
| tree | 2 | 100 | 107.787 | 11170 | 9952 | 13656 | 3.684e-08 | 4.136e+05 | 44394144 |
| leaf | 4 | 100 | 211.065 | 14706 | 0 | 13218 | 2.050e-12 | 2.483e+05 | 26115736 |
| tree | 4 | 100 | 225.611 | 11612 | 9740 | 13218 | 2.035e-12 | 8.461e+05 | 54085216 |

The tree cuts error queries by about one quarter. Order zero is about 13%
faster than the leaf path, order two is similar, and order four is about 7%
slower in this small measurement. All approximate paths remain slower than
direct evaluation here. Tree construction, envelope setup and repeated tile
passes still dominate. Bounds remain very conservative and grow when larger
nodes use uniform descendant envelopes. These measurements do not establish
linear scaling, a production speedup or automatic total-energy accuracy.

## Numerical coverage and remaining scope

253 distinct focused checks pass: 32 hierarchy cases, 25 existing ordered
multipole cases and 196 shared bubble, radial-error, translation, erfc and
repository-contract checks. The hierarchy cases include Gamma and nonzero-q
complex adjoints, exact finite references, atom-image and reciprocal gauges,
retained ranks zero/one/full, energy derivatives, signed Cartesian/spherical
shells, tight and always-near direct limits, empty selected support, all joint
resource caps, public HF/KS states and the J-only correlated-fit export guard.

The shared envelope bounds each descendant separately. An N-image accepted
node contributes N times its uniform erfc bound, with no cancellation credit
from complex phases. The bound excludes arithmetic/phase/factor roundoff,
finite-domain and auxiliary-basis error, and density relaxation. The
approximate partition is checked against independent exact finite integrals;
it is not independently reimplemented. Public references contract separately
exported approximate coupling with NumPy and independent SCF drivers.

Only one selected image tree is retained per ordered pair image. Cross-pass
partition reuse, global density-compressed upward/downward actions, automatic
accuracy and source-qualified symmetry remain unfinished. This bounded
increment does not close the full MANTA 0.18.x design or claim BIPOLE/CRYSTAL
parity.


## Complete calculation acceptance

Two actual local vq submissions passed on implementation commit
`2c35f6e0ed14cbdbe28fc7a7d8191dad4a620956`, using native SHA-256
`92bc21d45dbd256e27b24437f708398b70c5d473475794f9acc52c5e7f64c6d7`.
The runs verify 1,601 source/reference pins before and after execution and
72 managed output hashes. A separate NumPy-only assessment of the fetched
arrays also passes. The timing evidence above uses this same native runtime
and matching source pins; its original pre-commit provenance is retained.

Eight complete public states cover shifted-mesh Be RHF, empty-beta H UHF,
converged and capped Li UHF, Be PBE RKS, converged and capped Li PBE UKS, and
Li UHF with auxiliary cutoff 12 bohr. The final case exercises accepted
multi-image nodes during a converged SCF: 1,124 represented images in 1,102
far blocks, 2,376 tree nodes and 1,168 upward translations across the paired
Fock passes. The smaller cells exercise descent to singleton far nodes.
Four additional raw complex cases at Gamma/nonzero-q and orders zero/two
also exercise multi-image nodes. Capped states remain nonconverged and save
their evaluated density; no unevaluated final proposal is exported.

Maximum errors against separately contracted and independently relaxed
references are 5.052e-15 for Fock elements, 1.191e-13 Eh for same-density
energy, 1.120e-13 Eh for independently relaxed energy and 1.519e-9 for
density elements. Acceptance gates remain 8e-11 for Fock/energy and 5e-9
for density. Raw action and duality errors are below 5.1e-15. The largest
same-density changes from replacing the exact finite coupling are 4.340e-8
in Fock elements and 4.808e-7 Eh in energy, inside the deliberately conservative
reported bounds. Shared Fock application, method energy bookkeeping, exact
exchange/Madelung ownership, XC and saved-density consistency all pass.

No matched ORCA or CRYSTAL calculation is available for these custom finite
domains. References use independent finite Gaussian/Boys/Fourier formulas,
libint erfc integrals, NumPy metric contractions and independent SciPy SCF;
KS reference contractions use the same libxc functional. This establishes
the qualified finite-source implementation, not external-program parity.
