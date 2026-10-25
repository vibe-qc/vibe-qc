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
repeated candidate-Gram construction is the next bounded implementation gap.
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
