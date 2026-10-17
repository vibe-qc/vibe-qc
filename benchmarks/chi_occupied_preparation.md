# Bounded cyclic occupied-preparation ladder

Reproduce with `python -m benchmarks.chi_occupied_preparation` in the development
environment. The deterministic model has 16 primitive AOs, six occupied bands
(two frozen), a nonorthogonal metric, and complete translated seed degeneracies.
Sources and imports are prepared before measurement. Both selectors consume
the same immutable Bloch projectors. Every selected seed and coefficient is
compared against the explicit dense selector; no physical accuracy is claimed.

| Cells | Dense retained bytes | Compact retained bytes | Dense traced peak | Compact traced peak | Dense seconds | Compact seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 768 | 768 | 28,641 | 25,206 | 0.00750 | 0.00359 |
| 2 | 3,072 | 1,536 | 48,857 | 36,731 | 0.01273 | 0.00711 |
| 4 | 12,288 | 3,072 | 95,657 | 62,571 | 0.03001 | 0.01907 |
| 8 | 49,152 | 6,144 | 279,123 | 116,859 | 0.09626 | 0.09347 |

Measured with Python 3.14 and NumPy 2.5.3 in one local run. Timings include tracemalloc
and are observations, not performance promises. Largest complete coefficient
difference: `5.56e-16`. The eight-cell final active dense Gram would contain
8,192 bytes; the compact selector diagonalizes at most a 4-by-4 active block
and a 2-by-2 frozen block. The source-realification helper separately uses up
to twice the occupied band count per character. The conservative selection
payload reservation at eight cells is 2,386,944 bytes under the default 64 MiB
cap, including a 1 MiB NumPy iterator allowance. It is deliberately larger
than the traced Python/NumPy peak and is not an observed high-water mark.

The same script separately measures both independent source partitions,
starting from identical Bloch columns with complex phases and comparing every
projector entry with the original dense real-span oracle:

| Cells | Dense retained bytes | Compact retained bytes | Dense traced peak | Compact traced peak | Dense seconds | Compact seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 768 | 4,096 | 26,104 | 45,886 | 0.00029 | 0.00124 |
| 2 | 3,072 | 8,192 | 61,896 | 72,398 | 0.00033 | 0.00254 |
| 4 | 12,288 | 16,384 | 179,624 | 198,103 | 0.00044 | 0.00962 |
| 8 | 49,152 | 32,768 | 593,576 | 387,183 | 0.00100 | 0.06084 |

The compact source retains two AO projector home rows; this can exceed dense
occupied storage for small meshes. Its exhaustive real-column metric and AO
projector replay is substantially slower. These costs are included rather
than hidden behind the selector's savings. The source phase table is supplied
before timing; factor preparation is excluded. The largest complete projector
difference on this run was `5e-16`.

This ladder measures source realification and selection separately, excluding
factor construction, pair preparation and native transport. The test suite
separately exercises the complete chi route with complex Bloch phases, reordered
characters, odd/even and multidimensional meshes, independent frozen partitions,
near-dependent seeds and full-domain canonical MP2. Allocation guards prohibit
full occupied conversion, dense selection and the original full real-span
routine through a complete compact solve; the native input is home columns.

Retained source projectors and operator symbols cost O(N b squared), but the
complete metric/projector replay performs many reconstructions and contractions.
It is not a linear-time algorithm. The adapter's shared phase table remains
quadratic in cell count. Domain projected matrices, occupied Fock, individual
pair AO bases, amplitudes and the prepared overlap graph remain dense. Native
library scratch, pre-existing arrays, downstream outputs, allocator overhead
and process RSS are outside this preparation ledger. Scientific runtime
acceptance and D93 qualification remain separate from these measurements.
