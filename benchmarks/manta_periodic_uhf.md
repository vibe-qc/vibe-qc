# Finite-source physical periodic UHF

Run `.venv/bin/python benchmarks/manta_periodic_uhf.py --repeats 5` from a
freshly rebuilt editable checkout. Twelve fresh single-thread workers cover
hydrogen (two s Gaussians, one alpha and zero beta electrons) and lithium
(three s Gaussians, two alpha and one beta electron), Gamma, two-point and
shifted two-point meshes, and both explicit exchange conventions.

The portable fixtures are defined in `tests/test_manta_periodic_uhf.py`.
Their skew cell, explicit small reciprocal cubes, origin-only AO support,
physical nuclear radius and numerical tolerances match the independent
Gaussian-quadrature/libint source. These finite supports are intentionally
small and do not establish cutoff convergence or a realistic basis limit.

Each timed call regenerates physical S/T/V, nuclear Ewald energy, all spin/k
Focks, the optional separately admitted BvK correction, scalar SCF history
and the independent terminal confirmation build. One cold call precedes
five measured repetitions. NumPy/SciPy SCF on independent integral witnesses
checks the actual timed output's energy components, Fock and evaluated spin
densities. The benchmark measures the native explicit-support driver;
selected-job delivery and QVF are not yet part of this UHF path.

Reported peak process RSS includes imports, fixtures, repetitions and the
independent oracle. Modeled bytes reserve native owners, admitted component
peaks and typed exports, with backend headroom; they are distinct from RSS.
SCF iterations and Fock calls are separate because confirmation adds one
call. Shell and primitive-pair counts are executed source work. No speedup,
scaling law or transferred literature performance is claimed.

The optional refined PySCF AFT checks are separate from timed workers. They
use hydrogen at Gamma and a shifted two-point mesh, a 25-cubed reciprocal
grid, AO images through three cells per direction, an 18-bohr pair cutoff
and a 16-bohr nuclear cutoff. PySCF receives explicit supercell populations
and a fixed-spin occupation callback because its empty-beta diagnostic
references an uninitialized orbital-energy array. Its independent integral,
Fock and SCF implementations remain unchanged.

## Measured explicit-support driver

Measured on 2026-10-08 with Python 3.14.8, NumPy 2.5.3 and an
arm64 macOS native build. Each worker sets BLAS and OpenMP threads to one.
The core SHA256 was
`0f3f5437d7f2243d73deb3492f214477e8e532cab1f557b57c5fea1a02e47646`.
All twelve independently checked workers passed. Times are seconds; memory
is MiB. Short millisecond calls show scheduling noise in the measured range.

| Atom | Mesh | Exchange | Cold | Median (min, max) | CPU median | RSS | Modeled | Iterations / Focks |
|---|---|---|---:|---:|---:|---:|---:|---:|
| H | gamma | strict-zero-mode | 0.00081 | 0.00064 (0.00057, 0.00086) | 0.00064 | 160.7 | 80.502 | 2 / 3 |
| H | gamma | BvK-ewald | 0.00349 | 0.00055 (0.00053, 0.00060) | 0.00055 | 160.2 | 80.699 | 2 / 3 |
| H | multi-k | strict-zero-mode | 0.00305 | 0.00283 (0.00278, 0.00287) | 0.00284 | 158.8 | 80.518 | 12 / 13 |
| H | multi-k | BvK-ewald | 0.00705 | 0.00458 (0.00400, 0.00655) | 0.00430 | 159.5 | 80.726 | 17 / 18 |
| H | shifted | strict-zero-mode | 0.00358 | 0.00310 (0.00283, 0.00434) | 0.00301 | 159.7 | 80.518 | 12 / 13 |
| H | shifted | BvK-ewald | 0.00953 | 0.00440 (0.00406, 0.00490) | 0.00409 | 158.7 | 80.726 | 17 / 18 |
| Li | gamma | strict-zero-mode | 0.00294 | 0.00234 (0.00221, 0.00290) | 0.00229 | 160.3 | 80.539 | 9 / 10 |
| Li | gamma | BvK-ewald | 0.01244 | 0.00749 (0.00589, 0.00827) | 0.00576 | 160.8 | 80.736 | 14 / 15 |
| Li | multi-k | strict-zero-mode | 0.02067 | 0.01050 (0.00887, 0.01474) | 0.00789 | 158.8 | 80.563 | 10 / 11 |
| Li | multi-k | BvK-ewald | 0.02607 | 0.01707 (0.01615, 0.02196) | 0.01098 | 158.7 | 80.771 | 14 / 15 |
| Li | shifted | strict-zero-mode | 0.00799 | 0.02086 (0.01199, 0.03472) | 0.00798 | 158.6 | 80.563 | 10 / 11 |
| Li | shifted | BvK-ewald | 0.01078 | 0.00621 (0.00612, 0.07945) | 0.00622 | 159.6 | 80.771 | 14 / 15 |

| Atom | Mesh | Exchange | Executed shell calls | Executed primitive pairs | Reserved history bytes |
|---|---|---|---:|---:|---:|
| H | gamma | strict-zero-mode | 96 | 96 | 92800 |
| H | gamma | BvK-ewald | 96 | 96 | 92800 |
| H | multi-k | strict-zero-mode | 648 | 648 | 92800 |
| H | multi-k | BvK-ewald | 888 | 888 | 92800 |
| H | shifted | strict-zero-mode | 648 | 648 | 92800 |
| H | shifted | BvK-ewald | 888 | 888 | 92800 |
| Li | gamma | strict-zero-mode | 594 | 594 | 92800 |
| Li | gamma | BvK-ewald | 864 | 864 | 92800 |
| Li | multi-k | strict-zero-mode | 1242 | 1242 | 92800 |
| Li | multi-k | BvK-ewald | 1674 | 1674 | 92800 |
| Li | shifted | strict-zero-mode | 1242 | 1242 | 92800 |
| Li | shifted | BvK-ewald | 1674 | 1674 | 92800 |

Maximum absolute errors across timed outputs were 2.133e-10
in density, 1.278e-13 hartree in Fock,
1.927e-13 hartree in total energy and
1.973e-13 hartree in the energy components.
The two separately run refined PySCF comparisons passed their original
tolerances: 2e-9 hartree total energy, 3e-10 S/H, 3e-8 spin density,
3e-9 hartree spin Fock, 3e-11 hartree nuclear energy and 3e-12 Madelung.

## Cutoff-generated UHF

Run the same benchmark with `--source cutoffs --repeats 5`. These workers
use a 2-hartree reciprocal kinetic cutoff and the same 3-bohr physical pair
radius as the small explicit fixtures. Each timed call includes geometric
planning, certified source generation, conjugacy audit, complete spin SCF
and confirmation. The independent integral/SCF oracle uses the actual
generated supports. Planner bounds and domain storage are included in
admission; counts below are accepted source sizes and executed Fock calls.
This is a finite-cutoff benchmark, not a cutoff-convergence study.

Measured on 2026-10-08 with the same runtime and single-thread settings.
Native SHA256:
`57d677b8fc2d6baad79b89b9b4ac4c8e3dbc1a9c171552562e0c358261d365e3`.

| Atom | Mesh | Exchange | Cold s | Median s (min, max) | CPU s | RSS MiB | Modeled MiB | Domain MiB | G / Q vectors | Iterations / Focks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H | gamma | strict-zero-mode | 0.00116 | 0.00119 (0.00085, 0.00152) | 0.00098 | 161.4 | 80.551 | 0.048 | 28 / 28 | 2 / 3 |
| H | gamma | BvK-ewald | 0.01094 | 0.00171 (0.00112, 0.00802) | 0.00110 | 160.0 | 80.748 | 0.048 | 28 / 28 | 2 / 3 |
| H | multi-k | strict-zero-mode | 0.01164 | 0.01441 (0.00976, 0.01825) | 0.00618 | 158.5 | 80.585 | 0.065 | 28 / 112 | 12 / 13 |
| H | multi-k | BvK-ewald | 0.03223 | 0.01465 (0.01030, 0.03033) | 0.00872 | 160.4 | 80.793 | 0.065 | 28 / 112 | 18 / 19 |
| H | shifted | strict-zero-mode | 0.00374 | 0.00294 (0.00291, 0.00339) | 0.00293 | 159.8 | 80.585 | 0.065 | 28 / 112 | 12 / 13 |
| H | shifted | BvK-ewald | 0.00899 | 0.00510 (0.00473, 0.00533) | 0.00511 | 160.6 | 80.793 | 0.065 | 28 / 112 | 18 / 19 |
| Li | gamma | strict-zero-mode | 0.00245 | 0.00212 (0.00209, 0.00253) | 0.00212 | 159.9 | 80.588 | 0.048 | 28 / 28 | 9 / 10 |
| Li | gamma | BvK-ewald | 0.00522 | 0.00326 (0.00281, 0.00330) | 0.00326 | 160.0 | 80.785 | 0.048 | 28 / 28 | 14 / 15 |
| Li | multi-k | strict-zero-mode | 0.00418 | 0.00433 (0.00395, 0.00451) | 0.00433 | 159.6 | 80.630 | 0.065 | 28 / 112 | 10 / 11 |
| Li | multi-k | BvK-ewald | 0.00802 | 0.00620 (0.00568, 0.00683) | 0.00585 | 159.5 | 80.838 | 0.065 | 28 / 112 | 14 / 15 |
| Li | shifted | strict-zero-mode | 0.00448 | 0.00513 (0.00412, 0.00623) | 0.00494 | 159.7 | 80.630 | 0.065 | 28 / 112 | 10 / 11 |
| Li | shifted | BvK-ewald | 0.01263 | 0.00698 (0.00659, 0.00811) | 0.00648 | 161.1 | 80.838 | 0.065 | 28 / 112 | 14 / 15 |

All workers passed. Maximum absolute errors were 1.808e-10
in density, 1.275e-13 hartree in Fock,
1.918e-13 hartree in total energy and
1.972e-13 hartree in components. All fixtures selected one
AO translation; that deliberately small support keeps the independent oracle
bounded. Separately refined cutoff tests use 60 hartree, 18-bohr pair and
16-bohr nuclear radii and pass the PySCF tolerances stated above.
