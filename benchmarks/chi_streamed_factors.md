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

## One-slab audit reuse (D18, #583)

Measured on 2026-10-23 with native source `49317c5db2920cdae4b5d675304825221ac2624b`,
release 0.17.26, native SHA-256
`c0b76f59cbac5ce0177432f2e4fd8f205215e8c5919d41e9cbc3d45b9f047589`.
The Python provider SHA-256 is
`96bdc2733918f0921daa357512847b173130985b39c608bc228147b61f7cd36a`.
The delivery commit is recorded on #583. The D17 baseline is loaded from
`49317c5d`, without changing the checkout. Each comparison uses the same
immutable source, byte cap, auxiliary tile ceiling 4 and cache ceiling 1 MiB.
These are single-call timings with tracing enabled on a shared machine,
without resource isolation or statistical sampling. Times are seconds; memory
values are bytes. Source construction precedes each measurement.

When the budget has spare capacity, only the first audited displacement slab is snapshotted.
It is reused for subsequent target character tiles and released before the
next AO/auxiliary tile. If T is the number of character tiles, each AO/auxiliary
tile previously used `T*T*(1+T)` inverse blocks. Reuse removes
`(T-1)*(1+T)` of them. All original entrywise defects and every recovered
source entry are still measured, with the same summation order. No complete
character table, home factor tensor or quadratic audit array is retained.

The original 16-AO/32-auxiliary ladder at 2 MiB:

| Cells | D17 audit time | D18 audit time | D17 traced peak | D18 traced peak | D17 accounted audit peak | D18 accounted audit peak | Inverse blocks, D17 to D18 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 0.18913 | 0.21464 | 9406 | 5022 | 1049472 | 1049472 | 4096 to 4096 |
| 2 | 2.64985 | 2.15491 | 5220 | 5837 | 1050368 | 1050496 | 24576 to 18432 |
| 4 | 3.83279 | 3.58974 | 8150 | 9567 | 1055744 | 1056256 | 24576 to 18432 |
| 8 | 10.85870 | 8.73418 | 19094 | 21759 | 1077248 | 1079296 | 24576 to 18432 |

At eight cells, 25% fewer inverse blocks cost 2,048 extra accounted audit
bytes. The retained Bloch source is still 8,388,608 bytes, outside this
workspace cap; retained home-factor storage after preparation remains zero.
The whole-call accounted peak remains 1,145,856 bytes because pair-contraction
buffers dominate. Gamma has no repeated target tile and gains no reuse.
Its timing regresses in this sample. No general speedup is claimed.

The two-AO/three-auxiliary model tests both tight and larger meshes:

| Cells | Budget | D17 time | D18 time | D17 traced peak | D18 traced peak | D18 accounted audit peak | Inverse blocks, D17 to D18 |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | minimum | 0.00164 | 0.00179 | 9166 | 4894 | 1048896 | 24 to 24 |
| 1 | 2 MiB | 0.00073 | 0.00073 | 4886 | 4718 | 1049280 | 8 to 8 |
| 2 | minimum | 0.01341 | 0.01457 | 4774 | 5230 | 1049216 | 144 to 144 |
| 2 | 2 MiB | 0.00456 | 0.00679 | 5142 | 5439 | 1050080 | 48 to 36 |
| 4 | minimum | 0.14758 | 0.13877 | 4822 | 5374 | 1049856 | 960 to 960 |
| 4 | 2 MiB | 0.01013 | 0.01216 | 7554 | 8215 | 1054592 | 48 to 36 |
| 8 | minimum | 1.72023 | 1.83559 | 9686 | 9758 | 1051136 | 6912 to 6912 |
| 8 | 2 MiB | 0.02878 | 0.02524 | 15842 | 17559 | 1072640 | 48 to 36 |
| 16 | 2 MiB | 0.18825 | 0.16240 | 50014 | 55775 | 1144832 | 48 to 36 |
| 32 | 2 MiB | 2.04365 | 1.83316 | 160910 | 175959 | 1241088 | 320 to 260 |

All minimum plans retain exactly `I + 320*N` bytes of accounted workspace,
with no slab reuse. One byte less refuses before audit allocation; pair
outputs still refuse at these audit-only minima. Zero cache capacity also
keeps complete recomputation. The exact optional slab reservation enables
reuse; one byte less in either workspace or cache capacity disables reuse
without changing admission or the chosen tile widths. Small two- and
four-cell timings regress despite fewer inverse blocks; snapshots and runtime
variation matter. At 32 cells the extra reservation is 12,288 bytes and
inverse blocks fall from 320 to 260, an 18.75% reduction.

All four audit differences from D17 are exactly zero across these ladders.
The separate complete-table oracle `f1f74442` also gives exactly the same
defects on the small model at 1, 2, 4, 8, 16 and 32 cells. At 32 cells its
accounted audit peak is 1,409,024 bytes versus 1,241,088 for D18; traced peaks
are 403,022 versus 176,079 bytes, and times 0.76096 versus 2.02623 seconds.
The complete-table oracle remains faster there and is never an automatic
fallback. The largest raw pair difference from the dense home oracle across
all ladders is 3.97e-14.

Focused regression coverage includes 147 translation tests plus 578 shared
source, canonical MP2, reference-transport and checkpoint tests. Cache
immutability/lifetime, reordered characters, late/null-component and
adjacent-float threshold defects, complex MO contractions, global residuals,
PNO ranks/corrections and independent frozen partitions remain checked.
These are implementation checks, not independent scientific acceptance.
Managed vq/reference qualification remains tracked separately; chi stays
experimental and D93 fitted/post-HF stays not-qualified.

Reproduce with the containing checkout environment:

```text
python -m benchmarks.chi_streamed_factors --baseline 49317c5d
python -m benchmarks.chi_streamed_factors --baseline 49317c5d --aos 2 --aux 3 --budgets minimum 2097152
python -m benchmarks.chi_streamed_factors --baseline 49317c5d --aos 2 --aux 3 --cells 16 32
python -m benchmarks.chi_streamed_factors --baseline f1f74442 --aos 2 --aux 3 --cells 1 2 4 8 16 32
```

The benchmark now measures each baseline at the requested byte cap and
records preparation/pair refusals explicitly under `baseline_provider`.
Traced Python overhead and the conservative numerical-payload reservation
remain distinct; neither is process RSS or a total-memory bound.


## Raw reversed-AO slab reuse (D20, #583)

Measured 2026-10-24 against D18 (`35b6adaf`) at identical budgets and tile
ceilings. The Python base is `55f751b8` with the D20 changes; the native
module was rebuilt from `855e57b4` (native sources unchanged at that Python
base), release 0.17.26. Its SHA-256 is
`15c245e0a4be4a3460b0d6b3fe1deaa1ea3f09a44504f012f0c80bdae0e45d2e`.
The Python provider SHA-256 is
`1c282d8632cfc66f0ad6df0f85e2e77135317b33a8f7edf1d1a0f5a28782ed1f`.
Both providers use the same immutable model. These remain single-call,
traced timings on a shared machine; they establish no general speedup or
process-memory guarantee. Source construction precedes measurement.

D20 retains a separately budgeted raw complex reversed-AO slab for the first
S tile and all R. Later R and target-character tiles use it in the original
symmetry comparison and real transpose average. The averaged first-R slab
and raw first-S slab remain distinct, immutable and local to one AO/auxiliary
tile. No symmetry assumption, projected audit, complete canonical array or
quadratic audit tensor is introduced. Original full-axis reduction and
recovery orders, null components and phase-action checks are preserved.

For T character tiles, D18 computes `H=T*T-(T-1)` home slabs per AO/auxiliary
tile and uses `H*(1+T)` inverse blocks. D20 saves another `H-1` blocks,
leaving `H*T+1`. The first reverse block now covers all R, so inverse-entry
counts separately describe its larger output; block counts alone are not
operation counts. Each optional cache adds `16*N*d*p` accounted bytes and
must fit alongside the phase snapshot. The tile plan and audit minimum stay
unchanged. Both caches are released before leaving the AO/auxiliary tile.

The 16-AO/32-auxiliary ladder at 2 MiB:

| Cells | D18 audit time | D20 audit time | D18 traced peak | D20 traced peak | D18 accounted audit peak | D20 accounted audit peak | Inverse blocks, D18 to D20 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 0.22319 | 0.25851 | 9406 | 5390 | 1049472 | 1049472 | 4096 to 4096 |
| 2 | 1.81776 | 1.84718 | 5621 | 6654 | 1050496 | 1050624 | 18432 to 14336 |
| 4 | 2.99356 | 2.99005 | 8935 | 10832 | 1056256 | 1056768 | 18432 to 14336 |
| 8 | 7.25264 | 6.61955 | 21511 | 24560 | 1079296 | 1081344 | 18432 to 14336 |

At eight cells, inverse blocks fall 22.22% and inverse entries fall from
1,572,864 to 1,441,792 (8.33%). The extra raw snapshot reserves 2,048 bytes.
The whole-call accounted peak remains 1,145,856 bytes because contractions
dominate. Source storage remains 8,388,608 bytes outside this cap; no home
factor storage survives preparation. Gamma has no second target tile and
keeps the same work count. Gamma and two-cell timings regress in this sample.

The two-AO/three-auxiliary ladder includes exact minimum budgets and more
than two character tiles:

| Cells | Budget | D18 time | D20 time | D18 traced peak | D20 traced peak | D20 accounted audit peak | Inverse blocks, D18 to D20 |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | minimum | 0.00149 | 0.00233 | 9166 | 5262 | 1048896 | 24 to 24 |
| 1 | 2 MiB | 0.00068 | 0.00074 | 4886 | 5086 | 1049280 | 8 to 8 |
| 2 | minimum | 0.01317 | 0.01373 | 4774 | 5598 | 1049216 | 144 to 144 |
| 2 | 2 MiB | 0.00347 | 0.00413 | 5511 | 6224 | 1050176 | 36 to 28 |
| 4 | minimum | 0.11202 | 0.13421 | 4822 | 5742 | 1049856 | 960 to 960 |
| 4 | 2 MiB | 0.00732 | 0.00824 | 8243 | 9320 | 1054976 | 36 to 28 |
| 8 | minimum | 1.55141 | 1.53575 | 9686 | 9758 | 1051136 | 6912 to 6912 |
| 8 | 2 MiB | 0.02826 | 0.03018 | 17715 | 19816 | 1074176 | 36 to 28 |
| 16 | 2 MiB | 0.14042 | 0.16974 | 56607 | 62432 | 1150976 | 36 to 28 |
| 32 | 2 MiB | 1.66069 | 1.64313 | 173615 | 188936 | 1253376 | 260 to 212 |

At 32 cells the additional reservation is 12,288 bytes, inverse blocks fall
from 260 to 212 and inverse entries from 79,872 to 72,960. Several small-model
timings regress despite less inverse work. Snapshot and slicing overhead,
unchanged covariance work and runtime variation still matter. Further
repeated inverse work remains; this is a bounded scheduling increment.

All four audit differences from D18 are exactly zero across these ladders.
The maximum raw pair discrepancy from the dense home oracle is 3.97e-14.
Minimum plans keep `I+320*N` accounted bytes and no audit caches; one byte
less still refuses before allocation. At those audit-only minima, pair
outputs still refuse. Exact cache/workspace edge regressions cover zero,
one and two audit slabs without changing admission or tile widths.

The bounded regression run passed 423 tests across translation MP2, occupied
preparation, shared source, canonical MP2 and HF reference files. It includes
complete-table and adjacent-float threshold comparisons, cached and late
null-component defects, immutable snapshots and release on success/failure,
complex MO contractions, frozen partitions, PNO corrections/ranks and global
residuals. These are implementation checks, not independent vq/reference
scientific acceptance. Chi remains experimental; D93 fitted/post-HF remains
not-qualified, with the PBC/MANTA production gates unchanged.

Reproduce with the containing checkout environment:

```text
python -m benchmarks.chi_streamed_factors --baseline 35b6adaf
python -m benchmarks.chi_streamed_factors --baseline 35b6adaf --aos 2 --aux 3 --budgets minimum 2097152
python -m benchmarks.chi_streamed_factors --baseline 35b6adaf --aos 2 --aux 3 --cells 16 32
```
