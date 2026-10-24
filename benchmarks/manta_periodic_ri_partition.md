# Bounded periodic RI tree and partition reuse

The shared finite GDF source now retains a bounded working set of selected
auxiliary-image trees and certified interaction decisions. Both raw directions,
materialized tiles and later SCF builds consume the same immutable partition.
No density data, full coupling tensor or metric solution is retained. Numerical
source identities, arithmetic order and fixed-density error ledgers are
unchanged. Reconstruction remains available with zero capacity.

`coupling_multipole_partition_bytes` defaults to 64 MiB when hierarchical
multipoles and Gaussian preparation reuse are enabled. A FIFO evicts old
entries, and an entry whose conservative bound exceeds the capacity streams.
The source reserves that capacity twice, for retained data and an in-flight
or borrowed entry, plus bookkeeping and the existing producer workspace.
This prevents an unbounded image-interaction cache but can admit substantially
more memory than a small fixture actually retains. Exact, leaf and reciprocal
routes reserve no partition cache; disabling preparation reuse also disables it.

## Reproduction

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/manta_periodic_ri_partition.py --workers 3 --repeats 3
```

Three independent processes each measure the first call and three warm calls,
with one numerical thread on arm64. The compact RI-J fixture has two k points,
two AOs, six auxiliary functions and fixed full metric rank six. Its metric
construction is outside timing. Angular raw cases apply both complex directions
using signed contracted Cartesian or spherical d shells. Every pair of capacity
settings uses the same coupling approximation and unchanged factors.

| Case | Order | Epsilon | Action | Cache MiB | Cold ms | Warm ms | Warm partition builds / reuses | Modeled peak bytes |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| RI-J | 0 | 0 | materialized | 0 | 22.635 | 22.751 | 0 / 0 | 25478848 |
| RI-J | 0 | 1 | materialized | 0 | 84.277 | 83.581 | 384 / 0 | 43573760 |
| RI-J | 0 | 1 | materialized | 64 | 48.356 | 11.853 | 0 / 384 | 177799680 |
| RI-J | 0 | 1 | compressed | 0 | 86.158 | 86.149 | 384 / 0 | 44263936 |
| RI-J | 0 | 1 | compressed | 64 | 50.704 | 14.369 | 0 / 384 | 178489856 |
| RI-J | 2 | 100 | materialized | 0 | 112.020 | 110.845 | 384 / 0 | 48195392 |
| RI-J | 2 | 100 | materialized | 64 | 64.009 | 16.731 | 0 / 384 | 182421312 |
| RI-J | 2 | 100 | compressed | 0 | 118.280 | 119.241 | 384 / 0 | 53511424 |
| RI-J | 2 | 100 | compressed | 64 | 73.311 | 24.493 | 0 / 384 | 187737344 |
| RI-J | 4 | 100 | materialized | 0 | 232.091 | 234.087 | 384 / 0 | 58844880 |
| RI-J | 4 | 100 | materialized | 64 | 142.825 | 45.074 | 0 / 384 | 193070800 |
| RI-J | 4 | 100 | compressed | 0 | 290.795 | 280.378 | 384 / 0 | 77228112 |
| RI-J | 4 | 100 | compressed | 64 | 190.940 | 91.395 | 0 / 384 | 211454032 |
| Cartesian-d | 2 | 1 | materialized | 0 | 3.490 | 3.346 | 2 / 0 | 65822929 |
| Cartesian-d | 2 | 1 | materialized | 64 | 3.009 | 2.373 | 0 / 2 | 200048849 |
| Cartesian-d | 2 | 1 | compressed | 0 | 3.548 | 3.250 | 2 / 0 | 71199121 |
| Cartesian-d | 2 | 1 | compressed | 64 | 2.932 | 2.333 | 0 / 2 | 205425041 |
| spherical-d | 2 | 1 | materialized | 0 | 4.525 | 4.250 | 2 / 0 | 60916273 |
| spherical-d | 2 | 1 | materialized | 64 | 3.771 | 3.231 | 0 / 2 | 195142193 |
| spherical-d | 2 | 1 | compressed | 0 | 4.383 | 4.300 | 2 / 0 | 66274289 |
| spherical-d | 2 | 1 | compressed | 64 | 3.923 | 3.198 | 0 / 2 | 200500209 |

The zero-error first row is exact direct finite coupling. For the small RI-J
fixture, retained partitions improve warm materialized actions by about 5-7x
and compressed actions by about 3-6x. The cold paired build also benefits
because its adjoint reuses the forward partition. All 384 warm pair-image
visits reuse retained decisions with zero tree builds or new erfc certification
queries. Angular paired actions improve by about 24-29%. The order-zero and
order-two materialized cases are faster than direct finite RI-J here; their
approximation errors remain part of the comparison.

The compact maximum J errors relative to direct finite coupling remain about
1.786e-6, 3.684e-8 and 2.035e-12 for orders zero, two and four. They are bounded
fixed-density changes, not basis/domain or relaxed-state accuracy guarantees.
Every warm result is bitwise equal to its first result. Separate source tests
also require bitwise equality with zero-capacity reconstruction, for both maps,
dense tiles, eviction and oversized-entry fallback. Angular references retain
independent libint erfc and Gaussian Fourier checks. The cache changes execution,
not the approximate partition or metric. Compression remains independently opt-in.

These measurements establish a working-set benefit in the declared fixtures,
not linear scaling, a large-system crossover or automatic whole-method accuracy.
The exact selector still enumerates each requested finite support. Node charges,
fields and translations are still recomputed, and global AO-pair aggregation
is not implemented. A working set larger than capacity can repeatedly evict
entries. Model peaks include the conservative cache and borrowed-entry allowance;
they are not measured resident memory. The raw angular caches retain only
67,976 and 61,576 modeled bytes respectively despite the larger admitted cap.

## Validation

243 affected scientific source checks pass, including 35 new cache tests.
They cover both contraction modes, Gamma/nonzero-q and orders zero/two/four,
complex adjoints, every operator column, weighted derivatives, exact factors,
signed angular shells, atom/reciprocal gauges, public HF/KS receipts, concurrency,
bounded eviction/fallback and full resource admission before input inspection.
The source-native runtime includes the concurrent BIPOLE fixed-order erfc
optimization from main. No alternate Fock, exchange, XC or symmetry owner is
introduced. The full MANTA 0.18.x capability matrix remains open.


## Complete calculation acceptance

Actual local vq execution on source `f12c5aea9c7f9176d00648cc9c3502f65c4ea39c` passed seven complete
public HF/KS states and six raw complex-action cases. The native SHA-256 is
`b3c93fd91031d0f6017bb3a12b22112462bc0e3ecc059b4abb543d32a318a80b`; benchmark and acceptance use the same native bytes
and matching source pins. All 1,605 source/reference pins matched before and
after execution, and 63 managed output hashes verified. A separate NumPy-only
assessment of the fetched arrays passed. Five documentation checks and the
205-section released-changelog guard also passed.

Public cases cover shifted-mesh Be RHF, empty-beta H UHF, converged/capped Li
UHF, Be PBE RKS and converged/capped Li PBE UKS. Every final evaluated Fock
reused its selected partitions with zero new tree construction or certification
queries. Be/Li cases reuse 54 pair-image partitions across the paired actions;
H reuses 24. Gaussian preparation remains reused, and Fock application,
energy bookkeeping, exact exchange/Madelung, XC ownership and saved QVF
density remain consistent. Capped states remain explicitly nonconverged.

Raw complex Gamma/nonzero-q order-zero/two and signed Cartesian/spherical
d-shell cases are bitwise equal to zero-capacity reconstruction in both
directions. They also agree with the separately materialized coupling and
its complex adjoint. Canonical fixtures retain 64 partitions, with modeled
live storage of 1,886,640 bytes at order zero and 3,290,208 bytes at order two;
each subsequent direction reuses all 64 without eviction or certification.
Angular cases retain one partition. All approximation bounds and source
identities remain unchanged by cache history.

Maximum Fock discrepancy is 4.997e-15; same-density and independently relaxed
energy discrepancies are 1.013e-13 and 8.705e-14 Eh, with density discrepancy
1.519e-9. Gates remain 8e-11 for Fock/energy and 5e-9 for density. Forward,
adjoint and duality discrepancies are below 1.9e-15, 1.0e-15 and 3.6e-15.

No matched ORCA/CRYSTAL reference is available for these custom finite
domains. References use independent finite Gaussian/Boys/Fourier formulas,
libint erfc, NumPy metric contractions and independent SciPy SCF; KS uses
the same libxc functional. The approximate partition is not independently
reimplemented. Fixed-density bounds exclude roundoff, finite-domain/basis
error and density relaxation; no external-program parity claim follows.

The native runtime includes the concurrent BIPOLE fixed-order erfc change.
Later release metadata was integrated before submission without changing
scientific source or rebuilding an already healthy runtime. Timing retains
its original precommit content pins. Global AO-pair aggregation, automatic
accuracy and the remaining approved MANTA capabilities remain unfinished.
