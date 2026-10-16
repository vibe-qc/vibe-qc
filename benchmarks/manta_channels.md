# MANTA contraction-sharing microbenchmark

This bounded diagnostic isolates primitive sharing across contraction channels.
It is not a molecular SCF, BUPO, symmetry or scaling benchmark. Run the rebuilt
editable checkout with:

```sh
.venv/bin/python benchmarks/manta_channels.py > /tmp/manta-channels.json
```

The fixed fixture contains three pure d channels on one center, 15 AOs and
two arbitrary real symmetric, indefinite densities. General channels share
both exponents; partial channels have overlapping exponent sets; disjoint
channels share none. Signed coefficients are normalized by the same basis
adapter in all variants. All builds compute J and K for both densities.
The existing optimized direct libint path uses its public J/K entry points.
The ordered libint and MANTA paths share one ordered traversal across the two
densities. MANTA channel sharing uses batches of 16 quartets. No approximation
threshold is introduced.

Each row comes from a fresh single-thread process, one complete warm-up and
five measured builds. The table records constructor setup separately from
median build wall time and its min/max range. Every measured output is checked
against the existing zero-screening direct libint builder. Peak RSS includes
imports, the fixture and reference validation; it is not a measured kernel
workspace. These tiny cases are dominated in memory by the loaded runtime.

The October 2026 run used ARM64 macOS, Python 3.14.8 and the same native
extension for all variants, with native optimization enabled and binding LTO
disabled. The native binary SHA-256 was
`040ba671425170e84307148659660cc6d9e31211436e5c2085cfdd78094bac5c`.
The script records versions, core identity, timings, work and errors without
machine names or local paths.

| Case | Variant | Setup ms | Build median ms | Build range ms | Peak process MiB | Evaluated MD primitive quartets |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| General | Existing libint direct | 1.652 | 2.458 | 2.419-2.480 | 144.59 | unavailable |
| General | Ordered libint | 0.014 | 2.461 | 2.445-2.468 | 144.81 | unavailable |
| General | Segmented MANTA | 0.018 | 16.701 | 16.443-17.043 | 144.73 | 1296 |
| General | Shared MANTA | 0.013 | 2.524 | 2.489-2.556 | 145.42 | 96 |
| Partial | Existing libint direct | 1.367 | 1.393 | 1.382-1.418 | 144.53 | unavailable |
| Partial | Ordered libint | 0.015 | 1.361 | 1.325-1.373 | 144.59 | unavailable |
| Partial | Segmented MANTA | 0.014 | 8.348 | 8.079-8.647 | 144.70 | 625 |
| Partial | Shared MANTA | 0.013 | 4.010 | 3.930-4.070 | 145.06 | 226 |
| Disjoint | Existing libint direct | 1.374 | 2.331 | 2.299-2.362 | 144.45 | unavailable |
| Disjoint | Ordered libint | 0.013 | 2.443 | 2.357-2.484 | 144.45 | unavailable |
| Disjoint | Segmented MANTA | 0.015 | 17.006 | 16.745-17.115 | 144.78 | 1296 |
| Disjoint | Shared MANTA | 0.014 | 18.942 | 18.893-19.076 | 145.27 | 1296 |

The ordered variants each consume 81 shell quartets and 50,625 integral
elements, accumulating 202,500 density terms for the four targets. Shared
MANTA reduces actual MD evaluations from 1296 to 96 for general channels and
from 625 to 226 for partial channels. The requested primitive counts remain
unchanged. Libint's internal primitive evaluations are not inferred from
requested products. The maximum absolute J/K discrepancy across all rows was
`8.53e-14`.

Sharing improved MANTA's measured build time by 6.62 times for general
channels and 2.08 times for partial channels. It was 11.4 percent slower for
disjoint channels. Existing libint was still faster in each of these cases.
These results motivate opt-in sharing and further dispatch measurements;
they do not establish an end-to-end speed advantage or transfer a published
SHARK speedup to this implementation.
