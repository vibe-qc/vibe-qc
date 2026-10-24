# Finite periodic local-exchange source

Run `python benchmarks/manta_periodic_local_exchange.py` with the current
own-checkout editable native build. The harness verifies native provenance,
sets BLAS/OpenMP/Accelerate threads to one, starts three fresh workers and
reports cold execution plus five timed repeats per worker. It releases each
previous result before starting the next admitted call. JSON goes to stdout;
keep operational records outside the product checkout.

The skew-cell fixtures use three asymmetric momenta with weights 0.2/0.3/0.5,
27 AO images, 1000 Gauss-Legendre cell points, a 64-point workspace batch,
and contracted s/p/d Gaussian shells. Restricted, spin-polarized pure and
spin-polarized Cartesian cases include complex positive density matrices.
The accuracy oracle evaluates Gaussian monomials directly over the whole
grid and calls independent libxc `lda_x` (the native `slater` alias) for energy and potential. Every
timed result must match energy, V and integrated populations within 2e-10.
The oracle uses different storage and is not a production speed baseline.

Reported resource admission includes caller/binding copies, bounded panels,
compensated output and one exported matrix owner, basis metadata and an
8 MiB backend allowance. Peak RSS includes Python, libraries, inputs and the
full-grid validation oracle. No RSS cap, infinite-image tail bound, production
accuracy profile, SCF acceleration or complete periodic KS support is claimed.
Native work counts are executed shell evaluations and primitive/grid pairs;
the budget's legacy `primitive_quartets` field caps those primitive/grid pairs.

## Measured qualification

A fresh native build on 2026-10-08 passed all 30 local-exchange tests,
42 joint-functional tests and 1026 focused solver, physical HF/job, AO,
runner, citation and repository-gate checks after integrating the shared
AO-workspace update (1098 total; two optional PySCF cases deselected).
An additional independent SciPy real-spherical-harmonic/libxc calculation
for 24 AO functions, nine asymmetric k points and 343 cell points matched
both restricted and spin-polarized sources: maximum energy error 1.12e-16 Ha,
potential error 1.00e-15 Ha and integrated-population error 2.23e-15.
It did not reuse MANTA's polynomial transforms or AO evaluator.

The table is an isolated rerun after the regression workload completed,
with five repeats in each fresh process. Times are absolute observations;
these small fixtures establish correctness and bounded panel use, not a
speedup or asymptotic scaling claim. Process RSS includes the full-grid oracle.

| Profile | Cold (s) | Median (s) | Range (s) | CPU median (s) | Modeled peak (bytes) | Panel (bytes) | RSS (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| restricted | 0.010752 | 0.009975 | 0.009733 to 0.013935 | 0.009543 | 8576461 | 34304 | 167.8 |
| spin | 0.011443 | 0.011490 | 0.011126 to 0.012767 | 0.011225 | 8595901 | 34304 | 166.4 |
| cartesian | 0.010957 | 0.010708 | 0.010629 to 0.010915 | 0.010703 | 8608605 | 37888 | 164.4 |

Each case executed 1296 shell evaluations and 162000 primitive/grid pairs.
Maximum timed-result errors were 1.421e-14 Ha in energy,
3.109e-15 Ha in potential elements and
2.132e-14 electrons in quadrature populations.

Native binary SHA256: `bf80ae61a69b21f86fba6e3a4da50eff29c4f3ec50bee502e87ceda2fdfc71d0`.

A separate reviewer compiled the actual source/binding and post-update AO
evaluator in an isolated module. Its 230 native calls covered Cartesian/pure
s-through-g shells (35/25 AOs), both spin conventions, five asymmetric k
points, seven asymmetric images and 113 grid points. Independent monomials
or SciPy real spherical harmonics plus libxc gave maximum energy error
1.11e-16 Ha, potential error 6.67e-16 Ha and population error 6.66e-16.
Thirty-six real/imaginary derivative directions agreed within 3.43e-12;
shell-gauge and reciprocal-relabel errors stayed below 7.52e-16. Grid
permutation, k-point splitting, batch changes, C/F/unaligned arrays and all
five exact/one-short budgets also passed. This review qualifies the finite
component only; complete physical KS composition remains a separate gate.
