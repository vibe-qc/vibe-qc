# Whole-mesh physical range RI exchange

This benchmark measures explicit-mesh `PeriodicRIFockBuilder` K-only sources
with physical erfc/erf kernels. Ordinary J retains its own Coulomb source.
Complete bounded range KS is qualified separately in the
[method benchmark](manta_periodic_ri_range_ks.md); public range-fitting jobs
remain gated.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_ri_range_fock.py
```

Four isolated workers use two or four explicit momenta, normalized nonuniform
weights, physical omega 0.43, numerical omega 0.65, 5 bohr pair/auxiliary
supports and the 27-vector shifted supports of the existing RI Fock fixture.
Each worker times five complete source constructions and five spin responses.
The Gamma owner supplies diagonal K; each ordered off-diagonal pair has its
own fixed metric/projector with reverse-source audits. H is supplied once.

| Kernel | k points | Setup median (ms) | Response median (ms) | Metrics | Actual metric shell calls | Actual response shell calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| erfc | 2 | 0.325 | 8.301 | 3 | 110 | 5904 |
| erfc | 4 | 1.457 | 40.016 | 13 | 470 | 25104 |
| erf | 2 | 0.173 | 1.356 | 3 | 12 | 864 |
| erf | 4 | 0.571 | 5.370 | 13 | 52 | 3744 |

Different kernels execute different work; these rows are not a speedup claim.
The source retains 263,832 bytes at two k points and 1,143,144 bytes at four.
Conservative modeled peaks are 25,799,950 and 26,803,966 bytes respectively,
including the existing unused SR reservations for erf. They are not RSS.
The independent libint/Gaussian/NumPy contractions agree within 1.60e-14 for
K, 4.00e-15 for F and 5.33e-15 hartree for all component energies.

Complete RKS/UKS diagnostics additionally use the shifted `(2, 1, 1)` regular
mesh, Be/Li basis, grid and finite controls of the
[range-fitting source benchmark](manta_periodic_range_fitting.md). Their
explicit semilocal PBE weights and 0.25 physical-range exchange are diagnostic
finite functionals, not an HSE/LC method claim. A separate native bare-Coulomb
RI J builder and native XC component complete the response; an independent
per-q reference and SciPy fixed point check the energy. The Python diagnostic
does not claim aggregate admission by the production bounded method driver.

| Kernel | Spin | Total energy (hartree/cell) | Independent energy error (hartree) |
| --- | --- | ---: | ---: |
| erfc | RKS | -11.503272155695836 | 1.78e-15 |
| erfc | UKS | -6.247587922970695 | 3.55e-15 |
| erf | RKS | -11.235226812373227 | 3.55e-15 |
| erf | UKS | -6.047536878424118 | 2.66e-15 |

Maximum density error is 1.07e-14, same-density Fock error 4.45e-15 and
energy-component error 8.89e-15. The grid is intentionally coarse; agreement
does not establish domain or quadrature convergence. No matched ORCA/CRYSTAL
reference is available for these custom finite functionals/supports, so
external parity remains untested.

Focused tests also cover one-k limits, zero weights, signed exchange scale,
truncated/empty fixed projectors, weighted variation and reciprocity, shared
image transport, source/frame/density separation and resource/refusal rules.
All 936 saved default Coulomb Fock fields from 18 cases retain their exact
arrays, scalar counts and identities.

Native core SHA-256:
`5db693750fd347b4e006bf976c363cf0bef6bb3a3e2b603ff3f6842c5ad61372`.
