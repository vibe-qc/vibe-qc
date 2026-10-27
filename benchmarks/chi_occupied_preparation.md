# Bounded cyclic occupied-preparation ladder

Run `python -m benchmarks.chi_occupied_preparation` in the development
environment; add `--profile` for function counts and cumulative times.
Use `--baseline ac2929cb` to run the original audit modules from Git in the
same process environment, without changing the checkout.
The deterministic model has 16 primitive AOs, six occupied bands (two frozen),
a nonorthogonal metric, complex Bloch phases and complete translated seed
degeneracies. Both partitions and selected frames are compared against the
explicit dense oracle. These are finite algebra measurements, not science.

The following before/after run compares `ac2929cb` occupied preparation with
D16 on the same Python 3.14 / NumPy 2.5.3 environment. The three original
preparation/storage/admission modules were loaded from that baseline for the
before run; both runs used the extended benchmark and identical source data.
Timings include tracemalloc and exclude cProfile. They are single-run
observations, not assertions or scaling promises. Imports, prescribed Bloch
operands and phase tables are prepared before timing.

## Source realification and binding

| Cells | Before seconds | After seconds | Before traced peak | After traced peak |
|---:|---:|---:|---:|---:|
| 1 | 0.001075 | 0.001437 | 45,606 | 65,348 |
| 2 | 0.002633 | 0.001574 | 72,350 | 109,985 |
| 4 | 0.010158 | 0.002854 | 198,191 | 208,201 |
| 8 | 0.059840 | 0.004136 | 387,351 | 399,809 |

## Selected-source admission

| Cells | Before seconds | After seconds | Before traced peak | After traced peak |
|---:|---:|---:|---:|---:|
| 1 | 0.000576 | 0.000813 | 18,665 | 25,509 |
| 2 | 0.001861 | 0.000796 | 31,401 | 43,005 |
| 4 | 0.008522 | 0.000984 | 56,785 | 78,869 |
| 8 | 0.074538 | 0.000903 | 107,473 | 150,517 |

## Selection, including selected-source admission

| Cells | Before seconds | After seconds | Before traced peak | After traced peak |
|---:|---:|---:|---:|---:|
| 1 | 0.003785 | 0.004352 | 27,732 | 41,777 |
| 2 | 0.007650 | 0.006856 | 36,891 | 47,350 |
| 4 | 0.019595 | 0.011361 | 63,859 | 84,919 |
| 8 | 0.098643 | 0.020909 | 117,931 | 159,959 |

## Complete occupied preparation

| Cells | Before seconds | After seconds | Before traced peak | After traced peak |
|---:|---:|---:|---:|---:|
| 1 | 0.004651 | 0.005728 | 45,534 | 63,097 |
| 2 | 0.010343 | 0.008298 | 72,374 | 109,865 |
| 4 | 0.030842 | 0.013628 | 197,215 | 207,065 |
| 8 | 0.159948 | 0.025166 | 387,311 | 399,721 |

Complete occupied preparation measures source realification/binding plus
selection in one call. It includes selection's admission, so the timing rows
must not be added together. Factor construction/admission, PAO/PNO building,
native preparation and iteration are outside this benchmark.

## Work and storage tradeoffs

A separate cProfile run at eight cells counted 140 real-column reconstructions
and 90 cyclic AO actions in the two source partitions before this change.
The selector made another 152 AO actions; its admission accumulated 16 AO
projector row tiles. The new source uses two real-character audits and no
folded-column replay. The selector uses three complete displacement Gram
checks and two independent displacement projector comparisons with no AO
actions or complete-column replay. Tests forbid those reconstructions and
assert bounded cache reuse; there are no wall-clock assertions.

At eight cells the source profile fell from 0.0664 to 0.00554 seconds.
Selected-source admission fell from 0.0881 to 0.00178 seconds. Greedy cyclic
seed construction remained about 0.026 seconds and now dominates selection;
repeated candidate-Gram construction was the next bounded implementation gap,
addressed by the D19 measurements below.
These profiled times include instrumentation and are separate from the table.

The improvement costs storage and small-system overhead. Gamma source
realification, admission and complete preparation are slower. At eight cells,
complete traced peak increases from 387,311 to 399,721 bytes; selector peak
increases from 117,931 to 159,959 bytes. Immutable source snapshots and bounded
Fourier intermediates are counted, including regressions. The conservative
payload reservation rises from 2,386,944 to 2,833,536 bytes under the same
64 MiB default cap. It includes 45,056 bytes for at most one overlap plus two
occupied symbol tensors, 128 bytes for sixteen scalar measurements and the
new intermediate/phase-validation charges. This is a reservation, not an
observed high-water mark. Caches expire with the preparation context.

Retained source projectors are unchanged at `16*N*b*b` bytes for the two
partitions (32,768 bytes at eight cells). Selected home columns are unchanged
at `8*N*b*m` (6,144 bytes), versus 49,152 bytes for complete dense selected
frames. The largest dense-oracle projector difference in this run was
`5.65e-16`; the largest selected coefficient difference was `7.78e-16`.
Every displacement is included in each metric/projector maximum, preserving
the original norm and separate frozen/active source binding. Near threshold
boundaries and for unverified inputs the original exhaustive replay remains.

The focused suite additionally compares optimized and exhaustive defects with
non-home errors, broken translation tables, frozen/active mixing, nonorthogonal
and slightly asymmetric metrics, complex phases, reordered characters,
degeneracies and odd/even multidimensional meshes. It preserves domains,
PNO spaces, raw residuals, separate PNO corrections and the independently
evaluated same-source canonical MP2 limit at full domains/zero cutoff.

No linear scaling or total-memory bound follows. The shared phase table and
its validation remain quadratic in cell count. Candidate seed Grams,
occupied Fock, projected domain matrices, individual pair bases, amplitudes
and prepared transport links remain costly. Native scratch, pre-existing
arrays, downstream outputs, Python metadata, allocator overhead and process
RSS are outside the occupied payload ledger. Scientific vq acceptance remains
pending; chi is experimental and D93 fitted/post-HF remains not-qualified.


## D19: bounded reference phases and greedy Gram reuse (#584)

Measured on 2026-10-23, Python 3.14, NumPy 2.5.3, release 0.17.26.
Both sides used the rebuilt native source `49317c5d`; native SHA-256
`cf0f49ec6f6eb43f2f2cc1b4c983d08d09181c74f60ae37d55c42993938565bb` (39,439,248 bytes). The new preparation module SHA-256 is
`ce0f609cfcb137987e8019791664e94248188392ae3b2ad66cc06ccef955ac6b`. Source delivery commits are recorded on #584.
The old side loads preparation, selector, audit, source and storage modules
from `49317c5d` in memory, without modifying the checkout. The deterministic
model remains 16 AOs, six occupied bands, two frozen bands and a nonorthogonal
metric. Both partitions and selected columns are checked against the dense
oracle. Maximum projector and column differences are respectively 1.66e-15
and 1.40e-15. These are finite algebra checks, not scientific acceptance.

The following are sequential single-call timings with tracemalloc enabled,
without cProfile, process isolation or statistical sampling. Thread counts
were fixed to one for OpenMP and OpenBLAS. Source/model construction precedes
measurement. Phase setup is now included in total preparation. A bounded
phase object initially stores only immutable coordinates and its identity;
phase generation is charged to the transforms that request it. These stages
overlap and their times must not be added. Bytes below are traced peaks,
including Python overhead, unless explicitly labelled reserved.

### Reference phase setup

| Cells | Before seconds | After seconds | Before traced peak | After traced peak |
|---:|---:|---:|---:|---:|
| 1 | 0.000019 | 0.000083 | 1480 | 2313 |
| 2 | 0.000027 | 0.000106 | 1600 | 2298 |
| 4 | 0.000037 | 0.000123 | 2080 | 2763 |
| 8 | 0.000027 | 0.000154 | 4000 | 3609 |
| 16 | 0.000035 | 0.000199 | 11680 | 5385 |
| 32 | 0.000050 | 0.000273 | 42400 | 8344 |

### Complete source realification and binding

| Cells | Before seconds | After seconds | Before traced peak | After traced peak |
|---:|---:|---:|---:|---:|
| 1 | 0.001826 | 0.001717 | 65244 | 65987 |
| 2 | 0.001862 | 0.004131 | 109985 | 109736 |
| 4 | 0.003431 | 0.003602 | 208201 | 207840 |
| 8 | 0.005344 | 0.005240 | 399697 | 398872 |
| 16 | 0.009985 | 0.010849 | 786753 | 783216 |
| 32 | 0.023574 | 0.025493 | 1566385 | 1551552 |

### Selection including its admission

| Cells | Before seconds | After seconds | Before traced peak | After traced peak |
|---:|---:|---:|---:|---:|
| 1 | 0.005819 | 0.005155 | 36915 | 41248 |
| 2 | 0.009195 | 0.008262 | 47143 | 48576 |
| 4 | 0.017225 | 0.012548 | 84935 | 86064 |
| 8 | 0.028772 | 0.024183 | 159975 | 161056 |
| 16 | 0.054309 | 0.044577 | 310519 | 311600 |
| 32 | 0.109362 | 0.075747 | 611910 | 613688 |

### Independent selected-source audit

| Cells | Before seconds | After seconds | Before traced peak | After traced peak |
|---:|---:|---:|---:|---:|
| 1 | 0.001123 | 0.000962 | 25373 | 25805 |
| 2 | 0.001492 | 0.001065 | 43093 | 43485 |
| 4 | 0.001293 | 0.001039 | 78997 | 79349 |
| 8 | 0.001292 | 0.001402 | 150677 | 150909 |
| 16 | 0.001544 | 0.001898 | 294037 | 294269 |
| 32 | 0.001783 | 0.001622 | 581045 | 581213 |

### Total occupied preparation

| Cells | Before seconds | After seconds | Before traced peak | After traced peak |
|---:|---:|---:|---:|---:|
| 1 | 0.007390 | 0.007931 | 63241 | 63923 |
| 2 | 0.011684 | 0.010259 | 110025 | 110635 |
| 4 | 0.022878 | 0.017487 | 207417 | 207931 |
| 8 | 0.033248 | 0.028828 | 400841 | 401627 |
| 16 | 0.064058 | 0.055325 | 791689 | 788051 |
| 32 | 0.122052 | 0.098448 | 1586057 | 1560675 |

### Candidate-Gram work and reservation

The following same-run microbenchmark compares the original compact
contraction with reuse, including the final Lowdin construction. It uses the
active partition only. Counts include complete primitive cache construction,
winning-score replay and final gauge contractions. Candidate eigenspectra
remain exhaustive: 58 per character. This model needs no boundary replay;
near-tie and near-dependent tests explicitly exercise that slower path.

| Cells | Original Gram builds | Reused Gram builds | Original seconds | Reused seconds | Original traced peak | Reused traced peak |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 59 | 6 | 0.002439 | 0.002067 | 17080 | 24488 |
| 2 | 118 | 12 | 0.003567 | 0.002913 | 28464 | 36856 |
| 4 | 236 | 24 | 0.007923 | 0.005981 | 52544 | 73864 |
| 8 | 472 | 48 | 0.014897 | 0.010905 | 101696 | 123144 |
| 16 | 944 | 96 | 0.029548 | 0.022148 | 200000 | 221704 |
| 32 | 1888 | 192 | 0.060369 | 0.043526 | 396608 | 418824 |

| Cells | Before reserved payload | After reserved payload | Scoped primitive Gram cache | Phase tile |
|---:|---:|---:|---:|---:|
| 1 | 1617216 | 1621376 | 4096 | 1 |
| 2 | 1790592 | 1798784 | 8192 | 1 |
| 4 | 2137728 | 2154112 | 16384 | 2 |
| 8 | 2833536 | 2866304 | 32768 | 4 |
| 16 | 4231296 | 4296832 | 65536 | 8 |
| 32 | 7051392 | 7149696 | 131072 | 8 |

The old phase reservation was `64*N*N`; the new phase reservation is
`128*N*d`, with `d=min(8,max(1,N//2))`. The separate primitive Gram cache
adds `16*N*b*b`. Thus the total reservation grows on this ladder, even
though mandatory quadratic phase storage is removed. Gamma total time
increases, phase setup generally costs more, and several source/audit times
and traced peaks regress. No universal speedup or total-memory bound is
claimed. At 32 cells, total time falls from 0.122052 to 0.098448 seconds;
traced total peak falls from 1,586,057 to 1,560,675 bytes. The Gram-only
traced peak increases because of the cache. Its per-partition cache is
released on success and failure; no phase arrays are retained.

The unavoidable source projector blocks and selected homes remain linear
in cells but quadratic in primitive AO dimension where applicable. Retained
Bloch factors, canonical momentum metadata, occupied Fock, projected domains,
amplitudes and transport links retain their separate scaling outside this
cap. Remaining selection work includes all candidate eigenspectra; source
realification still constructs and audits every displacement block. #583
owns repeated factor inverse recovery separately.

Reproduce both sides, keeping output outside the checkout:

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m benchmarks.chi_occupied_preparation --baseline 49317c5d --cells 1 2 4 8 16 32
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m benchmarks.chi_occupied_preparation --cells 1 2 4 8 16 32
python -m pytest tests/test_chi_occupied_preparation.py tests/test_chi_translation_mp2.py -q
```

Add `--profile` to inspect phase transforms, candidate construction and
audits separately. The benchmark also emits the new standalone inverse
reference-transform measurement and phase-generation counters. Exact/below
budget tests and operation/allocation guards replace timing assertions.
The existing hash-bound H2 Gamma/odd and LiH even/frozen vq cases in the
manual remain the scientific handoff. The local containing runtime passed
164 focused tests before integration and 177 after preserving #583 on main
(overlapping runs, not additive); managed vq runtime and matched external acceptance remain
separate. Chi remains experimental and D93 fitted/post-HF not-qualified;
production periodic DLPNO is deferred beyond the bounded 0.18 release (#557).
