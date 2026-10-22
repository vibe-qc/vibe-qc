# Call-wide periodic RI moment actions

The optional `coupling_multipole_global_actions=True` path reuses each ordered
pair charge across auxiliary shells and projects each accumulated pair field
once. Its matched forward stage sums auxiliary fields before projecting each
auxiliary shell once; the adjoint shares that shell's compressed input. All
moment panels come from the existing immutable GDF preparation. All charges
and fields are private to the current call. The exact selected sphere/capsule
domains, certified far partition and finite metric remain unchanged.

This advances the input-moment/output-field organization of RI-BUPO (Neese
et al., DOI `10.1021/acs.jpca.4c07415`, equations 29-36). It is not yet a global
spatial pair-bubble traversal. The canonical Ewald pair inventory cannot
replace this producer's pair-dependent auxiliary capsule masks.

## Reproduction

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/manta_periodic_ri_global.py --workers 3 --repeats 3
```

Three sequential independent processes each measure one first call and three
warm calls on arm64 with one numerical thread. All rows use compression,
Gaussian preparation reuse and a 64-MiB partition cache. The compact RI-J
fixture has two k points, two AOs and six auxiliary functions with unchanged
rank-six metric factors; factor construction is outside timing. Orders zero,
two and four use the existing coupling tolerances 1, 100 and 100 respectively.
The angular cases apply both complex directions with signed contracted
Cartesian/spherical d shells. The many-auxiliary/many-k operator fixture has
16 auxiliary functions, two AOs and 16 arbitrary momenta; it is not a complete
SCF or a representative bulk scaling case.

| Case | Order | Action | First ms | Warm ms | Action products | Modeled peak bytes |
|---|---:|---|---:|---:|---:|---:|
| RI-J | 0 | per-tile | 44.064 | 12.975 | 39032 | 178489856 |
| RI-J | 0 | call-wide | 45.553 | 12.432 | 31286 | 185129728 |
| RI-J | 2 | per-tile | 62.548 | 20.791 | 1871948 | 187737344 |
| RI-J | 2 | call-wide | 62.689 | 20.940 | 1861448 | 196900096 |
| RI-J | 4 | per-tile | 164.065 | 79.414 | 20800731 | 211454032 |
| RI-J | 4 | call-wide | 162.747 | 79.283 | 20783194 | 227624784 |
| Cartesian-d | 2 | per-tile | 2.942 | 2.369 | 9204 | 205425041 |
| Cartesian-d | 2 | call-wide | 2.938 | 2.373 | 7584 | 206510545 |
| spherical-d | 2 | per-tile | 3.843 | 3.316 | 6480 | 200500209 |
| spherical-d | 2 | call-wide | 3.853 | 3.314 | 5540 | 201585713 |
| many-auxiliary-many-k | 2 | per-tile | 164.232 | 50.282 | 4898644 | 187836184 |
| many-auxiliary-many-k | 2 | call-wide | 161.604 | 50.662 | 4857480 | 199628312 |

Numbers are median first-call times across processes and median warm times
across nine calls. Pair gather/scatter counts in the compact cases are
60/64/62 unique visits across both directions, with 292/296/284 reuses and
30/32/31 pair projections. Each direction touches six auxiliary panels; the
forward performs only six final auxiliary projections. The larger operator
fixture has 60 pair visits, 884 reuses, 30 pair projections and 16 auxiliary
projections. These counters quantify actual contraction reuse; existing tree,
far-node and represented-image counts retain their original meaning.

The order-zero compact case saves about 20% of the counted action products
and about 4% of measured warm time. Higher-order and angular timings are
essentially unchanged; the larger operator fixture is slightly slower. These
small timings do not establish a production crossover. Translation and
remaining per-pair traversal dominate, so the new path stays opt-in. Modeled
memory rises by about 1-16 MB because admission reserves every candidate pair
entry even when few are used. Measured process peaks, including all cases,
range from 396 to 422 MB and are not per-case memory measurements.

Maximum raw-map difference from per-tile compression is 1.987e-15; compact
J differences are at most 1.388e-17 and metric factors are bitwise unchanged.
All repeated results are bitwise equal within the selected execution route.
Cross-route summation order changes, so bitwise equality is not required.
The selected multipole approximation itself remains the same: earlier
fixed-density errors and qualified error ledgers still apply. This execution
comparison does not establish finite-domain, basis or relaxed-density accuracy.

## Source validation

359 affected scientific source checks pass, including 40 new call-wide action
checks. They cover independent materialized maps, exhaustive complex columns
and adjoints, Gamma/nonzero-q, orders zero/two/four, signed angular shells,
atom/reciprocal gauges, zero/linear inputs, concurrent call-private state,
fixed metric ranks, weighted energy derivatives, all resource limits and public
HF/KS Fock receipts. Existing paired, screened, hierarchical, compressed,
preparation and partition behavior is also covered.

The source/native runtime includes the concurrent BIPOLE fixed-order erfc
implementation. Unified Fock, exact exchange and its Madelung convention, XC,
output density and source-qualified symmetry keep their existing owners.
No new symmetry or CRYSTAL parity qualification follows. Global spatial
pair-bubble aggregation, automatic accuracy and the full MANTA 0.18.x
capability matrix remain unfinished.


## Complete calculation acceptance

Actual local vq execution on source `a8d2c15c85b91de82fa239df775424a8ad02bc95` passed seven complete
public HF/KS states and six raw complex-action cases. The native SHA-256 is
`1e55a238910c54a40c91144e6cf62248aca9f520c9aa8254dc2f9e635bd692d7`. The benchmark uses the same native bytes and matching
source-content pins. All 1,606 source/reference pins matched before and after
execution; 63 managed output hashes verified. A separate NumPy-only assessment
of the fetched arrays passed. Five documentation checks, the 205-section
released-changelog guard, diff and staged privacy checks also passed.

Public states cover shifted-mesh Be RHF, empty-beta H UHF, converged/capped
Li UHF, Be PBE RKS and converged/capped Li PBE UKS. The terminal HF builds
reuse pair entries 36 times for Be/Li and 16 times for H across both actions,
projecting nine pair panels for Be/Li, four for H, and three auxiliary
panels in either case. The KS builds reuse ten pair entries,
projecting eight pair panels and two auxiliary panels. Every terminal Fock
also reuses prepared Gaussian data and selected partitions without new
certification. Shared Fock application and energy, exact exchange/Madelung,
XC ownership and saved QVF density remain consistent. Capped states remain
explicitly nonconverged.

Maximum Fock discrepancy is 4.968e-15; same-density and independently
relaxed energy discrepancies are 9.770e-14 and 9.059e-14 Eh. Density
discrepancy is 1.518e-09. Gates remain 8e-11 for Fock/energy and
5e-9 for density. Raw forward, adjoint and duality discrepancies are at most
1.831e-15, 9.155e-16 and 4.965e-16. These maps agree with per-tile
compression and separately materialized coupling; zero-capacity reconstruction
and repeated calls are bitwise equal within the global-action route.

No matched ORCA/CRYSTAL reference is available for these custom finite domains.
References use independent finite Gaussian/Boys/Fourier formulas, libint erfc,
NumPy metric contractions and independent SciPy SCF; KS uses the same libxc
functional. The approximate partition is not independently reimplemented.
Maximum same-density approximation differences from exact finite coupling
are 1.482e-08 in Fock and 1.786e-07 Eh in energy, within the
reported bounds. Those bounds exclude roundoff, finite-domain/basis error and
density relaxation. No bulk parity or automatic accuracy claim follows.

Global spatial pair-bubble traversal, tighter accuracy profiles and the
remaining approved MANTA 0.18.x capabilities remain required.
