# Retained chi factor storage ladder

`benchmarks/chi_streamed_factors.py` compares the dense native home-factor
oracle with the streamed retained-Bloch provider. This synthetic model has
16 primitive AOs, 32 primitive canonical auxiliary entries including one
null mode, random real symmetric home factors, two occupied test columns
and eight virtual test columns. It is not an SCF calculation or scientific
acceptance. Source construction occurs before measurement.

Measured on 2026-10-16 with Python source from this increment and native
source `6e3439a86102825d8fbeca3a0ed97ba67767ae48`, native SHA-256
`607355178f6c2d684911b58621392849839d240b971fb5028de853301f164ff0`.
The source-delivery commit is recorded on #508. The provider used a 2 MiB
cap, auxiliary tile 4 and exactly enough cache for its phase table. All
values below are bytes; the cap's iterator allowance is 1,048,576 bytes.

| Cells | Existing Bloch source | Removed home tensor | Stream cache | Accounted provider peak | Largest returned matrix |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 131072 | 65536 | 32 | 1070144 | 512 |
| 2 | 524288 | 262144 | 128 | 1080576 | 512 |
| 4 | 2097152 | 1048576 | 512 | 1101824 | 512 |
| 8 | 8388608 | 4194304 | 2048 | 1145856 | 512 |

The dense path additionally snapshots its home tensor. The streamed path
retains zero home-factor bytes. The accounted peak is the conservative
numerical-array ledger, including output buffers until return. It is not
a measured RSS. Independent `tracemalloc` peaks, including Python overhead,
were measured around each call:

| Cells | Dense preparation | Stream preparation | Dense pair | Stream pair |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 528225 | 13278 | 17960 | 29860 |
| 2 | 938555 | 6142 | 42536 | 30292 |
| 4 | 2758330 | 10990 | 77864 | 35412 |
| 8 | 9572781 | 31086 | 148008 | 45652 |

Single-call timings in seconds, with tracing enabled, are deliberately
reported alongside storage. These are instrumented measurements, without
resource isolation or repeated statistical sampling. The streamed complete
audit trades runtime for storage, and these data do not show a speedup:

| Cells | Dense preparation | Stream preparation | Dense pair | Stream pair | Maximum pair difference |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00143 | 0.121 | 0.000132 | 0.000335 | 6.94e-17 |
| 2 | 0.000881 | 0.144 | 0.000197 | 0.000709 | 7.77e-16 |
| 4 | 0.00373 | 0.280 | 0.000354 | 0.00222 | 2.89e-15 |
| 8 | 0.0256 | 0.843 | 0.000734 | 0.00825 | 3.96e-14 |

Across the ladder, maximum factor imaginary, symmetry, full recovery and
phase-covariance defects were respectively 4.36e-17, 1.39e-17, 3.20e-16 and
8.80e-15. The largest pair-contraction imaginary defect was 4.32e-14.
All use the unchanged 1e-10 admission threshold. The regression suite also
checks odd and multidimensional meshes, complex orbital phases,
nonorthogonal references, frozen core, source rejection, every-origin pair
covariance, full-space canonical energies and separately truncated domains
and PNOs.

Reproduce with the identified containing checkout environment:

```text
python benchmarks/chi_streamed_factors.py
python -m pytest tests/test_chi_translation_mp2.py -q
```

Keep operational output outside the product checkout. The source, selected
orbitals, pair spaces and downstream solver still have substantial dense
storage and unscreened work. These measurements imply neither linear scaling,
total-memory bounds nor production accuracy. The [manual](../docs/chi_local_mp2.md)
documents memory scope and the separate vq scientific-acceptance cases.
