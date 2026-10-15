# Bounded periodic atomic quadrature

Measured 2026-10-09. This is an explicit finite-grid adapter benchmark, not
a production grid default or complete KS job qualification. Three fresh CPU
workers each perform one cold build and five repetitions with OpenMP, OpenBLAS
and Accelerate thread counts fixed to one. The fixture has one He center in a
6-bohr cubic cell, two s functions, two non-Gamma momenta, and interior spin
densities. Both spin channels of UKS are populated; the previously documented
empty-beta PBE conditioning limit is not requalified here.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_quadrature.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_quadrature.py --refinement
```

All timing profiles use 40 radial points, Lebedev order 59, an 8-bohr
partition radius, and 48,080 points. Times below include native grid
generation and its receipt, but exclude public array exports and XC evaluation.
They are measured costs; no speedup claim is made.

| Profile | Cold (s) | Median (s) | Range (s) | XC evaluation (s) |
|---|---:|---:|---:|---:|
| lda-rks | 0.030534 | 0.028137 | 0.027833 to 0.028373 | 0.045647 |
| pbe-uks | 0.028615 | 0.027406 | 0.025734 to 0.028241 | 0.100114 |
| pbe-uks-stratmann | 0.025334 | 0.024288 | 0.023811 to 0.024584 | 0.100073 |

The native core SHA-256 was
`3b7aedc6d300cc2a1b619c6130f28b8a1ced1328d2f43a20d25e81958356170d`.

Independent Gaussian polynomials/gradients and raw libxc contractions on the
same generated grid agree within `2.443e-15` in energy, potential and
populations. The separate tensor-cell reference uses NumPy Gauss-Legendre
rules and those independent contractions; order 64 versus 80 differs by at
most `6.662e-15` hartree in energy and `4.525e-14` hartree in potential.
This validates the integration comparison, not libxc internals.

| Profile | Cell-reference energy error (Ha) | Potential max error (Ha) | Population max error |
|---|---:|---:|---:|
| lda-rks | 6.737940e-06 | 1.153282e-05 | 4.312382e-05 |
| pbe-uks | 7.011547e-06 | 1.327019e-05 | 2.468071e-05 |
| pbe-uks-stratmann | 8.152533e-06 | 1.153585e-05 | 2.415010e-05 |

Each grid plan admits 12,777,152 bytes, including an 8 MiB backend
margin, 3,654,080 grid/export bytes and 734,464 workspace bytes.
It bounds 1,331 atlas and 2,197 neighborhood candidates,
29,740,574,247,552 scalar terms and 951,698,375,921,664 kernel terms.
These conservative geometric and partition-work bounds are much larger than
executed work. The benchmark explicitly raises the work budgets. Peak process
RSS in its JSON includes Python, independent references and XC arrays; it is
not a proof of the grid allocation bound. A later KS call needs a separate
external-live reservation for this retained source.

Independent tests reconstruct the radial rule and degree-five spherical rule
without native tables, enumerate oversized integer-image neighborhoods,
normalize Becke/Stratmann products, and check canonical folding. Analytic
periodized Gaussian and reciprocal cosine integrals give additional refinement
witnesses. Exact/one-short caps, overflow, unsupported geometry, array ownership
and one-versus-three-thread identical source bytes are checked.

The PBE UKS refinement witness varies partition support separately from
radial/angular resolution. Errors are relative to the independent order-80
cell integral.

| Radial | Lebedev order | Partition radius (bohr) | Energy error (Ha) | Potential max error (Ha) |
|---:|---:|---:|---:|---:|
| 20 | 29 | 8 | 4.234064e-05 | 3.458157e-05 |
| 40 | 59 | 8 | 7.011547e-06 | 1.327019e-05 |
| 80 | 95 | 8 | 1.317755e-06 | 1.847784e-06 |
| 80 | 95 | 16 | 1.336849e-08 | 4.151055e-08 |
| 120 | 95 | 16 | 6.457614e-08 | 1.202164e-07 |
| 120 | 95 | 20 | 6.236327e-10 | 2.628399e-09 |

Refinement is not monotone: at radius 16, increasing 80 to 120 radial points
increases both errors. Radius 20 with 120 radial points attains energy
`6.237e-10` and potential `2.629e-9` hartree for this fixture. This is evidence
for one selected source, not a transferable automatic accuracy promise.
The broad method, lattice, species and basis qualification remains open.
