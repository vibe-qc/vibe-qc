# Selected periodic UHF job evidence

Run `.venv/bin/python benchmarks/manta_periodic_uhf.py --source job --repeats 5`
from a freshly rebuilt editable checkout. Twelve fresh single-thread workers
cover hydrogen and lithium, Gamma and two-point meshes with and without
shifts, and both explicit exchange conventions. The finite 2-hartree
reciprocal cutoff and 3-bohr pair radius match the independently integrated
[cutoff UHF fixtures](manta_periodic_uhf.md). They qualify execution, not
physical cutoff or basis convergence.

Each timed call includes the public job adapter, resource admission, source
generation, spin SCF and confirmation, standard output, geometry, citations,
structured log and QVF. One cold call precedes five measured repetitions;
each worker reuses a temporary output stem. Times do not measure durable
storage or remote delivery. The independent quadrature/libint and SciPy SCF
oracle checks the final timed state. QVF readback must exactly preserve both
spin densities. Worker RSS includes imports and the oracle outside timing.

Measured on 2026-10-08, arm64 macOS, Python 3.14.8, NumPy 2.5.3.
BLAS and OpenMP use one thread. Core SHA256:
`57d677b8fc2d6baad79b89b9b4ac4c8e3dbc1a9c171552562e0c358261d365e3`.
Small-case timings include interpreter and file-delivery overhead and
scheduling noise; no speedup or scaling conclusion is drawn.

| Atom | Mesh | Exchange | Cold s | Median s (min, max) | CPU s | RSS MiB | Modeled MiB | Iterations / Focks |
|---|---|---|---:|---:|---:|---:|---:|---:|
| H | gamma | strict-zero-mode | 0.3808 | 0.1617 (0.1565, 0.1724) | 0.1432 | 172.8 | 83.218 | 2 / 3 |
| H | gamma | BvK-ewald | 0.4257 | 0.1729 (0.1628, 0.1915) | 0.1507 | 173.8 | 83.414 | 2 / 3 |
| H | multi-k | strict-zero-mode | 0.4010 | 0.1665 (0.1475, 0.1750) | 0.1401 | 173.8 | 83.273 | 12 / 13 |
| H | multi-k | BvK-ewald | 0.4211 | 0.1884 (0.1689, 0.2252) | 0.1600 | 174.7 | 83.481 | 18 / 19 |
| H | shifted | strict-zero-mode | 0.8343 | 0.3725 (0.3471, 0.4007) | 0.2983 | 171.7 | 83.273 | 12 / 13 |
| H | shifted | BvK-ewald | 1.2233 | 0.5515 (0.4317, 0.6010) | 0.3048 | 172.1 | 83.481 | 18 / 19 |
| Li | gamma | strict-zero-mode | 1.3280 | 0.4486 (0.2147, 0.6823) | 0.2954 | 173.9 | 83.304 | 9 / 10 |
| Li | gamma | BvK-ewald | 0.4049 | 0.1857 (0.1657, 0.1881) | 0.1615 | 173.7 | 83.501 | 14 / 15 |
| Li | multi-k | strict-zero-mode | 0.3894 | 0.1827 (0.1584, 0.1886) | 0.1588 | 173.2 | 83.390 | 10 / 11 |
| Li | multi-k | BvK-ewald | 0.4152 | 0.1722 (0.1671, 0.1902) | 0.1536 | 172.8 | 83.598 | 14 / 15 |
| Li | shifted | strict-zero-mode | 0.3805 | 0.1739 (0.1659, 0.1879) | 0.1541 | 173.2 | 83.390 | 10 / 11 |
| Li | shifted | BvK-ewald | 0.4235 | 0.1820 (0.1663, 0.1875) | 0.1591 | 174.0 | 83.598 | 14 / 15 |

Maximum errors: density 1.808e-10, Fock 1.275e-13
hartree, total energy 1.918e-13 hartree and energy components
1.972e-13 hartree. Every QVF spin-density comparison was exact.

`tests/test_manta_periodic_uhf_job.py` independently checks complete-job
energies, Focks and densities, spin-resolved occupations and owned exports,
QVF validity and histories, empty-beta null diagnostics, shifted AO image
gauges, capped-state readback, pre-computation refusals, dry runs and delivery
failure semantics. The broader RHF and periodic restart regressions protect
existing behavior. These jobs retain the neutral all-electron full-mesh
fixed-spin profile; they do not qualify KS, RI/BUPO, reduced dimensionality,
magnetization stability, symmetry reduction or periodic derivatives.
