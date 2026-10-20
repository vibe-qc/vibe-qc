# Joint periodic quadrature, domains and KS

Measured 2026-10-09 with three fresh CPU workers, each using one cold run and
five repetitions. OpenMP, OpenBLAS and Accelerate thread counts are one.
The whole timed call includes grid generation, cutoff domains, physical SCF,
terminal confirmation and owned receipts. No speedup or RSS claim is made.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_ks_grid.py
```

These are finite-source consistency fixtures: 12 radial points, Lebedev order
17, an 8-bohr periodic Becke partition, the established physical pair support,
a reciprocal kinetic cutoff of 2 hartree, and an 11-bohr XC image sphere.
The full shifted three-point k mesh is retained. H2 and HeH use PBE, while the
empty-beta H fixture uses BLYP. The separate documented PBE endpoint
conditioning limit is unchanged. These deliberately small grids do not
qualify a production accuracy profile.

| Profile | Grid points | Cold (s) | Median (s) | Range (s) | Iterations |
|---|---:|---:|---:|---:|---:|
| rks | 2640 | 0.025037 | 0.025810 | 0.025577 to 0.026084 | 3 |
| uks | 2640 | 0.107649 | 0.108308 | 0.104333 to 0.109652 | 9 |
| empty-spin | 1320 | 0.037243 | 0.037246 | 0.036359 to 0.037584 | 10 |

The joint call is bitwise identical to separately constructing the same grid,
domains and physical KS calculation. Independent Gaussian/libxc contractions
and SciPy eigensolves provide a separate numerical witness:

| Profile | Density max error | Fock max error (Ha) | Energy error (Ha) | Population max error |
|---|---:|---:|---:|---:|
| rks | 2.220446e-15 | 1.665349e-15 | 1.554312e-15 | 2.220446e-16 |
| uks | 1.329266e-10 | 5.329102e-15 | 4.884981e-15 | 1.776357e-15 |
| empty-spin | 4.623457e-11 | 2.053913e-15 | 4.996004e-16 | 8.881784e-16 |

The plan charges the retained shared grid and explicit source copies during
all domain generation, SCF and terminal builds, and subtracts generation work
from downstream caps. The maximum peak spans both grid generation and SCF:

| Profile | Retained grid bytes | Count-only peak bound | Actual-source modeled peak |
|---|---:|---:|---:|
| rks | 233920 | 79836067 | 79787523 |
| uks | 233920 | 79926649 | 79846905 |
| empty-spin | 133344 | 79608285 | 79559741 |

These byte counts include backend allowances and conservative ownership
models; they are not measured process RSS. Tests cover exact simultaneous
caps and each one-short memory/shell/primitive/scalar/kernel cap, downstream
external-live reservations, and refusal before a deliberately invalid
partition can execute. Capped evaluated epochs retain independently checked
same-density Fock and energy data. Owned arrays and grid receipts survive
input mutation and parent lifetimes. Explicit-grid callers retain their
previous unset quadrature metadata.

The native core SHA-256 was
`048e9f90eb45c1e2204a90bfede1d519471ac8681f3e3ffe99e9a43275e339d2`.

The independent quadrature refinement study remains in
`manta_periodic_quadrature.md`. The subsequent public-job measurement is in
`manta_periodic_ks_job.md`. Broad grid accuracy qualification remains required;
numerical SCF convergence does not close that capability.
