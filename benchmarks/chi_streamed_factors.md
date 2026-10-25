# Retained chi factor storage ladder

The first section preserves the original D13 measurements. The D17 section
below measures the current character/displacement-tiled implementation.

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

## Character/displacement-tiled audit (D17)

Measured on 2026-10-21 with native source
`1b1a8155e087cd3ed8f5ca796ff156941cac6c34`, release 0.17.24, native SHA-256
`4fc54c93965709af41277016a27222d82c9540d80d9e1f1620e687f3370c352b`.
The Python provider SHA-256 is
`92d67b8b377cbf95658604cea0b34b7b5e58e0e2cbc0b65bc561aebff172e3a4`.
The delivery commit is recorded on #508. The complete-table implementation
is loaded explicitly from `f1f74442` for comparison without changing the
checkout. Both providers use the same immutable source, 2 MiB workspace cap,
auxiliary tile ceiling 4 and 1 MiB cache ceiling. The new cache retains only
one bounded phase rectangle even under this generous ceiling.

These are single-call instrumented timings with tracemalloc enabled, without
resource isolation or statistical sampling. Source/model construction precedes
measurement; Python overhead is included in traced peaks. Values are bytes
and seconds. The first ladder retains the original 16-AO/32-auxiliary model.

| Cells | Complete audit time | Tiled audit time | Complete traced peak | Tiled traced peak | Tiled accounted audit peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.12540 | 0.17131 | 7742 | 5166 | 1049472 |
| 2 | 0.16174 | 1.91572 | 5622 | 5532 | 1050368 |
| 4 | 0.30694 | 2.96830 | 10934 | 8622 | 1055744 |
| 8 | 0.90868 | 6.27907 | 31014 | 19406 | 1077248 |

| Cells | Complete pair time | Tiled pair time | Complete pair peak | Tiled pair peak | Tiled accounted whole-call peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.000702 | 0.000379 | 29860 | 27756 | 1070208 |
| 2 | 0.000801 | 0.001096 | 30292 | 30485 | 1080576 |
| 4 | 0.002496 | 0.002719 | 35412 | 35797 | 1101824 |
| 8 | 0.008376 | 0.008674 | 45748 | 46805 | 1145856 |

All four measured audit defects match the complete-table values exactly on
this ladder. The largest pair difference from the dense oracle is 3.97e-14.
At eight cells, the retained Bloch source remains 8,388,608 bytes. The new
provider still retains zero home-factor bytes. It does not allocate a full
character table or quadratic auxiliary audit tile. The default eight-cell
accounted whole-call peak is unchanged because contraction buffers dominate;
the Gamma ledger increases by 64 bytes. Some pair-call peaks and times
regress. The complete audit is substantially
slower, particularly for two through eight cells, because inverse blocks
are recomputed for each target character tile. This is a workspace change,
not a general speed improvement.

The minimum reservation is `I + 320*N` versus the former `I + 160*N*N`, where
`I = 1048576` is the unchanged iterator allowance. Gamma's new minimum is
160 bytes larger; two cells tie. Every minimum below succeeds, while one byte
less refuses before phase/audit allocation. This second ladder uses two AOs
and three canonical auxiliary entries, including one null mode, to make
minimum-budget recomputation practical.

| Cells | Former minimum | New minimum | Minimum audit time | Minimum traced peak | 2 MiB audit time | 2 MiB traced peak |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1048736 | 1048896 | 0.00126 | 4926 | 0.00057 | 5286 |
| 2 | 1049216 | 1049216 | 0.01042 | 5086 | 0.00352 | 5326 |
| 4 | 1051136 | 1049856 | 0.09378 | 5294 | 0.00672 | 7822 |
| 8 | 1058816 | 1051136 | 1.21722 | 9758 | 0.02053 | 16138 |

Every minimum plan uses d=p=1. Its accounted audit peak equals its cap.
The pair call then correctly refuses: audit feasibility does not promise
room for the output and orbital transforms. At eight cells, minimum-budget
preparation evaluates 6,912 inverse blocks and generates 148,288 phase
entries; the 2 MiB preparation uses d=4, p=3 and 48 inverse blocks. Both
recover all 768 retained entries and check all 512 translation identities.
Audit defects again match exactly at both budgets. Recomputations are
reported as work, not as skipped or sampled checks.

The same small model extends to larger meshes under 2 MiB:

| Cells | d, p | Complete audit time | Tiled audit time | Complete traced peak | Tiled traced peak | Tiled accounted audit peak |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 16 | 8, 3 | 0.05707 | 0.10466 | 87070 | 49062 | 1138688 |
| 32 | 8, 3 | 0.49874 | 1.19908 | 402942 | 160942 | 1228800 |

All audit differences are zero here too; pair differences remain below
3.89e-14. The phase/cache ledger covers concurrent blocks and snapshot copies,
with d capped at eight independently of N. Source storage, canonical adapter
metadata, earlier reference preparation, caller arrays, returned arrays after
return and downstream work remain outside the provider cap. None of these
observations establishes linear scaling, total-memory bounds or scientific
acceptance.

Reproduce all three ladders with the containing runtime:

```text
python -m benchmarks.chi_streamed_factors --baseline f1f74442
python -m benchmarks.chi_streamed_factors --baseline f1f74442 --aos 2 --aux 3 --budgets minimum 2097152
python -m benchmarks.chi_streamed_factors --baseline f1f74442 --aos 2 --aux 3 --cells 16 32
```

Use `--cache-bytes 0` for the uncached policy. Allocation tests guard against
full character tables and quadratic audit arrays in preparation and both
contractions; exact cap/output tests replace wall-clock assertions. Late
real-space, character and null-mode perturbations, including adjacent-float
threshold cases, retain the original admission decisions. The independent
canonical MP2 limit and separate domain/PNO/residual comparisons remain
implementation checks. Managed vq runtime acceptance remains pending; chi
is experimental and D93 fitted/post-HF is not-qualified.
