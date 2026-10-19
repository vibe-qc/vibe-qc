# Compressed periodic RI hierarchical actions

The existing selected auxiliary-image hierarchy now supports paired actions
that contract inputs into moments before far translations and project fields
back afterward. Forward density moments and downward auxiliary fields are
matched to the adjoint's upward auxiliary charges and accumulated pair fields.
The same producer chooses the same ordered finite partition in both directions.
Far blocks no longer materialize auxiliary-by-AO-pair coupling values. Direct
near, reciprocal and finite-zero terms retain the common residual tile path.

Select `coupling_multipole_compress=True` with positive multipole error and
`coupling_multipole_hierarchy=True`. Compression remains opt-in (default False)
because the measured benefit is not yet consistent. It changes contraction
ordering and therefore may change roundoff, but not the mathematical coupling,
finite metric, retained factor or numerical source identity. Source preparation
reuse works with either contraction mode.

## Reproduction and measurement

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/manta_periodic_ri_compressed.py --workers 3 --repeats 3
```

Three separate processes each measure the first call and three warm calls,
on arm64 with one numerical thread. Every comparison explicitly selects its
compression setting and reuses Gaussian preparation. The small RI-J fixture
has two k points, two AO functions, six auxiliary functions and full metric
rank six. Angular cases apply both raw complex directions using signed
contracted d shells: 36 Cartesian or 25 spherical AO pairs and respectively
six or five auxiliary components. They use expansion order two and the same
finite source for the materialized and compressed actions. Metric construction
is outside timing; raw angular calls contain no metric solve.

| Case | Order | Coupling epsilon | Action | Cold ms | Warm ms | Compressed pair images / nodes | Modeled peak bytes |
|---|---:|---:|---|---:|---:|---:|---:|
| RI-J | 0 | 0 | direct | 21.292 | 21.158 | 0 / 0 | 25478848 |
| RI-J | 0 | 1 | materialized | 76.913 | 76.786 | 0 / 0 | 43573760 |
| RI-J | 0 | 1 | compressed | 81.005 | 79.402 | 352 / 9930 | 44263936 |
| RI-J | 2 | 100 | materialized | 102.265 | 102.647 | 0 / 0 | 48195392 |
| RI-J | 2 | 100 | compressed | 110.624 | 110.073 | 360 / 9952 | 53511424 |
| RI-J | 4 | 100 | materialized | 212.791 | 210.910 | 0 / 0 | 58844880 |
| RI-J | 4 | 100 | compressed | 258.094 | 258.418 | 346 / 9740 | 77228112 |
| Cartesian-d | 2 | 1 | materialized | 3.302 | 3.181 | 0 / 0 | 65822929 |
| Cartesian-d | 2 | 1 | compressed | 3.244 | 3.082 | 2 / 60 | 71199121 |
| spherical-d | 2 | 1 | materialized | 4.027 | 3.982 | 0 / 0 | 60916273 |
| spherical-d | 2 | 1 | compressed | 4.059 | 4.112 | 2 / 44 | 66274289 |

Cartesian d-shell actions are about 3% faster. The spherical d case and the
small RI-J cases are about 3-23% slower. Positive magnitude envelopes and
downward translations add work, while image-tree construction and erfc error
queries still dominate. The workspace plan conservatively retains the old
producer reservation and adds compressed fields, charges and magnitude data;
the reported peak therefore increases. These measurements do not establish
a production speedup, linear scaling or a larger-cell crossover.

Both modes use identical metric factors and coupling partitions. The compact
RI-J maximum errors relative to direct finite coupling remain approximately
1.786e-6, 3.684e-8 and 2.035e-12 at orders zero, two and four. Raw angular
actions agree with separately materialized approximate tiles within 4e-12;
independent libint and Gaussian Fourier references qualify their source
approximation. New counters expose compressed pair/node counts, input/field
products and additional downward translations; residual-tile element counters
continue to describe only the materialized residual contraction.

The raw magnitude diagnostic bounds the full coupling without reconstructing
far tiles: positive `|W|` upward envelopes bound auxiliary moment rows, pair
moment maxima bound AO rows, and `b^T |I| p` bounds each accepted block. Sum
these and add the residual-tile maximum. RI uses this upper bound in its
existing fixed-density J estimate. It may be more conservative than measuring
the largest full tile element. The source approximation ledger is unchanged;
arithmetic/phase/factor roundoff, finite-domain and basis error, and density
relaxation remain outside that ledger.

The JSON retains original Git/source/native identities, every timing sample,
counters and peak process memory. These measurements used explicit controls
before the default was set to False. The implementation and benchmark control
values were unchanged; the original header/test source snapshots are retained
to verify the narrowly limited default-policy difference. Timing evidence is
reused without substituting the final runtime's identity for the measured one.

## Validation and remaining scope

208 focused source checks pass, including 40 new compressed-action cases and
168 affected hierarchy, preparation, leaf, raw-action, paired RI and Fock cases.
The new cases check every operator column and adjoint, arbitrary complex inputs,
zero-density structural activity, atom-image and reciprocal gauges, signed
Cartesian/spherical d shells, factored magnitude bounds, all source/parent RI
resource caps, rank-zero/one/full factors, weighted energy derivatives and
complete public HF/KS receipts. Concurrent source-preparation tests remain
covered. No symmetry reduction or new exchange/XC owner is introduced.

This compresses the selected auxiliary-image hierarchy. Global AO-pair bubble
aggregation, cross-pass tree/partition reuse, automatic whole-method accuracy,
source-qualified symmetry and the other approved MANTA 0.18.x capabilities
remain unfinished. The public capability matrix records this distinction.


## Complete calculation acceptance

An actual local vq submission explicitly enabled compression on implementation
commit `6f64692b1de81a38239fb7f46bcdc9761e7e1c40` with native SHA-256
`c0c22e4bd86c80719dc34069690f5bfc69705b30e8881d9fd742cd2cb43e7582`.
All 1,604 source/reference pins match before and after execution, and all
63 managed output hashes verify. A separate NumPy-only assessment of the
fetched arrays passes. The final runtime also passes all 208 focused source
checks and five documentation checks, including the unchanged default route.

Seven complete public states cover shifted-mesh Be RHF, empty-beta H UHF,
converged and capped Li UHF, Be PBE RKS, and converged and capped Li PBE UKS.
Every final evaluated Fock records compressed pairs/nodes, input/field products
and downward translations. Grouped Be/Li HF builds use 54 compressed pair
images, 1,102 accepted nodes and 584 additional downward translations across
the paired actions. Warm Gaussian preparation remains reused. Capped states
remain nonconverged and save their evaluated density.

Six raw cases cover complex Gamma/nonzero-q actions at orders zero/two and
signed Cartesian/spherical d shells. Every compressed result agrees with the
separately materialized approximate coupling; materialized and compressed
source identities and partitions match. The magnitude diagnostic bounds the
full coupling without reconstructing its far blocks. Max forward, adjoint
and duality discrepancies are below 1.9e-15, 1.0e-15 and 3.6e-15 respectively.

Maximum Fock discrepancy is 4.997e-15; same-density and independently relaxed
energy discrepancies are 1.013e-13 and 8.705e-14 Eh, and density discrepancy
is 1.519e-9. Gates remain 8e-11 for Fock/energy and 5e-9 for density. The
largest same-density changes relative to exact finite coupling are 1.482e-8
in Fock elements and 1.787e-7 Eh in energy, within the reported bounds.
Shared Fock application, energy bookkeeping, exact exchange/Madelung and XC
ownership, and QVF density consistency pass.

No matched ORCA or CRYSTAL calculation is available for these custom finite
domains. References use independent finite Gaussian/Boys/Fourier formulas,
libint erfc integrals, NumPy metric contractions and independent SciPy SCF;
KS uses the same libxc functional. The approximate partition is not
independently reimplemented. No external-program parity claim follows.

The timing runtime SHA-256 is
`72c2cd04531a6f8db2a6c6b1e7efb5b46e0bcc0db7fdbc577cb3bdd93a0cd383`.
Its explicit compression controls and action implementation match acceptance;
source snapshots verify only the documented default declaration and explicit
test-selection differences. This preserves the measured runtime's identity
and reuses valid timing evidence without repeating the benchmark.
