# Global ordered pair-tree periodic RI-BUPO

The optional `coupling_multipole_pair_tree=True` source aggregates certified
far interactions across ordered AO pair images using the shared GDF producer.
It keeps exact finite masks, a fixed metric, paired complex maps and the
existing reciprocal/zero producer. Gamma completes conjugate source pairs
only after qualifying exact finite support. The approximation has a distinct
source identity. It is experimental and remains opt-in.

## Reproduction

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/manta_periodic_ri_pair_tree.py --workers 3 --repeats 3
```

The script compares the existing call-wide auxiliary hierarchy with the new
pair-tree route at the same requested coupling tolerances. Their partitions
and actual approximation errors differ. Three sequential processes measure
one cold and three warm calls each, recording source/native hashes, operation
counts and resource admission. Metric preparation is outside RI-J timing.
Angular and many-k rows measure both raw map directions, not complete SCF.

The public manual describes source qualification and limits. Whole-method
automatic accuracy, representative bulk scaling and the remaining MANTA
0.18.x capability matrix remain unfinished.


## Initial pair-tree measurement

The compact RI-J fixture has two k points, two AOs and six auxiliary functions
with a fixed rank-six metric. Orders zero, two and four request coupling
tolerances 1, 100 and 100. These deliberately loose tolerances expose far
interactions; they are not recommended accuracy settings. Angular fixtures
use signed contracted Cartesian/spherical d shells. The many-k fixture has
16 auxiliary functions, two AOs and 16 arbitrary momenta.

| Case | Order | Route | First ms | Warm ms | Action products | Pair groups | Modeled peak bytes |
|---|---:|---|---:|---:|---:|---:|---:|
| RI-J | 0 | auxiliary hierarchy | 43.969 | 12.342 | 31286 | 0 | 185129728 |
| RI-J | 0 | pair tree | 53.762 | 15.247 | 57446 | 2332 | 226188384 |
| RI-J | 2 | auxiliary hierarchy | 63.033 | 21.484 | 1861448 | 0 | 196900096 |
| RI-J | 2 | pair tree | 73.709 | 19.586 | 2722632 | 2276 | 250667720 |
| RI-J | 4 | auxiliary hierarchy | 179.076 | 85.805 | 20783194 | 0 | 227624784 |
| RI-J | 4 | pair tree | 143.295 | 45.498 | 30344580 | 2128 | 353194480 |
| Cartesian-d | 2 | auxiliary hierarchy | 2.853 | 2.296 | 7584 | 0 | 206510545 |
| Cartesian-d | 2 | pair tree | 2.858 | 2.419 | 16164 | 0 | 217490953 |
| spherical-d | 2 | auxiliary hierarchy | 3.738 | 3.183 | 5540 | 0 | 201585713 |
| spherical-d | 2 | pair tree | 3.750 | 3.327 | 12660 | 0 | 212099881 |
| many-auxiliary-many-k | 2 | auxiliary hierarchy | 163.651 | 50.922 | 4857480 | 0 | 199628285 |
| many-auxiliary-many-k | 2 | pair tree | 145.589 | 24.171 | 3578268 | 2908 | 253418565 |

First-call times are medians across three processes; warm times are medians
across nine calls. Order-four RI-J is about 1.9 times faster warm, and the
many-k action pair about 2.1 times faster. Order two gains about 9%, whereas
order zero and angular cases are slower. Gamma completes conjugate source
pairs with two raw passes, so its counted products can increase despite a
wall-time improvement from fewer repeated translations and certifications.
The angular cases have no accepted multi-pair blocks. These small measurements
do not establish a general production or whole-SCF crossover. The option
stays disabled by default. Process peak RSS across all cases is 399-431 MB;
this is not a per-case memory measurement.

| Pair-tree RI-J order | Max J difference from exact finite coupling | Energy difference (hartree) | Evaluated coupling bound | Propagated J bound |
|---:|---:|---:|---:|---:|
| 0 | 2.455e-06 | 2.137e-06 | 0.984734 | 4.212e+05 |
| 2 | 6.690e-08 | 4.725e-08 | 98.4618 | 9.720e+08 |
| 4 | 5.645e-12 | 2.187e-12 | 99.6193 | 9.946e+08 |

All actual differences satisfy the reported bounds. The uniform envelope and
global positive-magnitude bounds are very conservative: these numbers do not
yet supply useful automatic accuracy control. The old and new partitions
differ, so matching requested tolerances does not mean equal actual accuracy.
Metric factors are bitwise unchanged and repeated calls within each route
are bitwise identical. Source-wide error allocation, useful-accuracy profiles
and representative scaling remain required work.

## Initial pair-tree scientific acceptance

672 focused scientific checks pass, including 61 new pair-tree checks. They
cover independent finite-source errors, both complete complex maps, dense/raw
exports, Gamma/nonzero q, orders zero/two/four, signed angular contractions,
physical erfc, image and reciprocal gauges, concurrent call-private state,
fixed ranks, weighted energy derivatives, public HF/KS and exact/one-below
resource limits at source, action, dense and RI parent boundaries. Gamma
cutoff ties that break literal finite capsule conjugacy and near-Gamma
representatives outside roundoff are explicitly refused. Uncached antichain
storage is reserved even with a zero partition-cache limit.

Actual local vq execution of source `cff7dea6c00227f6489306754103ca25f10f3610` passed seven complete
public states and ten raw complex-action cases. The native SHA-256 is
`b65d317cec32128bf1294020873cfcef4684ceebd163aa938192210c81a04dad`.
Source checks and benchmark use these same native bytes and matching source
content; a subsequent documentation-only fragment-format correction accounts
for their earlier commit identity. All 1,876 source/reference pins,
63 managed output hashes and 17 result-array hashes verified. An independent
NumPy-only assessment of the fetched arrays passed. Documentation links,
changelog fragment formatting, released-section pins and staged privacy pass.

Public cases are shifted Be RHF, empty-beta H UHF, converged and capped Li
UHF, Be PBE RKS, and converged and capped Li PBE UKS. Capped cases remain
explicitly nonconverged. Shared Fock apply/energy, exact exchange and Madelung,
XC and evaluated-density QVF are checked. Maximum Fock disagreement with the
independently contracted approximate coupling is 4.913e-15; density difference
is 1.518e-9 and same-density/relaxed energy differences are 9.770e-14 hartree.
Maximum same-density approximation errors against the exact finite source are
1.482e-8 in Fock and 1.786e-7 hartree in energy, within their recorded ledgers.

Raw maps cover Gamma/nonzero q at orders zero/two/four, signed Cartesian and
spherical d shells, and physical erfc at Gamma/nonzero q. Maximum map
disagreement with separately materialized coupling is 8.882e-16 and complex
duality error is 3.972e-15. Maximum coupling approximation error against the
independent exact finite reference is 2.022e-6 and satisfies its source bound.

No matched ORCA/CRYSTAL reference is available for these custom finite domains.
References use independent Gaussian/Boys/Fourier formulas, libint erfc,
NumPy retained-metric contractions and independent SciPy SCF; KS shares libxc.
The approximate Gaussian partition is not independently reimplemented.
The algorithm reuses Neese et al., DOI `10.1021/acs.jpca.4c07415`, equations
29-36, with the explicit periodic source/mask/Gamma extensions in the manual.
Current main's physical BIPOLE density-qualified symmetry and Fourier fixes
were integrated before the native build. No BIPOLE/CRYSTAL parity or symmetry
reduction permission is inferred for this new approximate coupling. Automatic
accuracy, low-dimensional kernels, remaining source adapters, derivatives,
acceleration and the full MANTA 0.18.x matrix remain unfinished.


## Block error ledger measurement

The next increment changes only error and magnitude bookkeeping. Accepted
interactions accumulate per-node bounds which propagate to original ordered
pair leaves. Images of the same orbital shell pair are summed; unrelated
output blocks contribute through maxima. Input block L1 norms provide action
component and Euclidean bounds, and RI propagation uses the forward Euclidean
bound with the unchanged retained metric. There is no cancellation credit,
new partition or new source identity. One auxiliary shell is processed at a
time, with explicit scratch/work admission.

The same fixtures, tolerances and three-process/nine-warm-call protocol give:

| RI-J order | Evaluated coupling bound | Previous J bound | Block J bound | J bound reduction | Block energy bound (hartree) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0277581 | 421171 | 1183.42 | 355.9 | 21.2851 |
| 2 | 1.17299 | 9.72026e+08 | 161429 | 6021.4 | 116235 |
| 4 | 1.63983 | 9.94625e+08 | 189133 | 5258.9 | 113384 |

Actual J and energy differences remain the values in the initial table.
Coupling bounds are 35-84 times tighter; J bounds improve by 356-6021 times
and energy bounds by about 12000-23000 times. Nevertheless, these deliberately
loose requested tolerances still produce bounds far above the actual errors.
This improvement does not qualify automatic useful-accuracy selection.
The many-auxiliary/many-k action component bounds improve by 134 times forward
and 122 times adjoint; signed angular action bounds improve by 4.8-6.5 times.

| Pair-tree case | Order | First ms | Warm ms | Modeled peak bytes |
|---|---:|---:|---:|---:|
| RI-J | 0 | 52.208 | 15.181 | 226566368 |
| RI-J | 2 | 64.589 | 17.656 | 251045704 |
| RI-J | 4 | 135.771 | 43.220 | 353572464 |
| Cartesian-d | 2 | 2.951 | 2.549 | 217539049 |
| spherical-d | 2 | 3.941 | 3.465 | 212147977 |
| many-auxiliary-many-k | 2 | 141.478 | 22.846 | 253796869 |

The additional admitted ledger workspace is about 378 kB for the compact RI-J
fixture and 48 kB for the angular cases. Process peak RSS across all cases is
399-426 MB. The timing differences from the initial measurements are small
and are not an isolated speedup claim. Numerical moment-action counts and
accepted multi-pair groups are unchanged. Broad production crossover and
whole-method useful-accuracy profiles remain open.

## Block ledger scientific acceptance

All 517 focused scientific checks pass: 80 pair-tree checks, including 19 new
block-ledger cases, and 437 affected-source regressions. Twelve documentation
checks pass. New checks cover sparse/complex inputs, phase and scale behavior,
zero bounds, duplicated auxiliary outputs, stable Euclidean bounds for inputs
scaled by 1e180, and RI propagation at retained ranks zero, one and six.
The existing resource, source, gauge, Gamma, physical erfc, angular, public
HF/KS and fixed-energy derivative gates remain in force.

Complete local vq calculation of source `32cd3a01cbab06d2f60809529da377a9ecbc33ff`
passes seven public HF/KS states and ten raw map cases. The native SHA-256 is
`02e7ebf5bc1a399dacd5b23529ad0d579621f217c17f10d19e47a7396fd570bf`.
All 1,876 source/reference pins, 63 managed outputs and 17 result
archives verify. A NumPy-only assessment of fetched arrays confirms both action
norm bounds, fixed-density energy/J bounds and the independent finite-source
references described above. It also compares 146 arrays bit for bit with the
accepted predecessor: couplings, raw maps, metric, Fock, density, energy and
saved spin density are unchanged. Coupling, reference and fixed-factor
identities match. The previous approximation/reference discrepancies are
unchanged, while the bounds decrease.

Source checks and benchmarks use the same native bytes and relevant source
hashes. A rebase brought in the separately accepted Python BIPOLE symmetry
residual implementation before vq acceptance; it changed no MANTA/native
source. Unified Fock and source-qualified symmetry remain compatible without
granting this approximate coupling any additional reduction permission.
No matched ORCA/CRYSTAL reference is available for these finite fixtures;
the same Gaussian/Boys/Fourier, libint and NumPy/SciPy references and limitations
apply. The full MANTA 0.18.x goal remains unfinished.


## Later error-model version

The subsequent shared isolated erfc radial/interior-moment bound changes
accepted approximate partitions and versions their source identities. The
historical measurements above retain their original meaning and provenance.
The same twelve-row regression benchmark passes with the new model; tighter
explicit tolerances and complete RHF costs are recorded in the
[accuracy benchmark](manta_periodic_ri_accuracy.md). Exact finite metric and
reference identities are preserved. The later increment does not turn these
historical loose-tolerance measurements into an automatic accuracy profile.
