# Finite complex periodic LDA/GGA component

Run `python benchmarks/manta_periodic_semilocal_xc.py` in the fresh own-checkout
editable runtime. The script verifies native provenance, sets OpenMP/BLAS/
Accelerate threads to one, and runs three fresh workers. Each records cold
execution and five timed repeats, releasing the previous result before the
next admitted call. Operational JSON stays outside the product checkout.

The fixtures use a skew cell, three asymmetric k points with unequal positive
weights, 27 canonical AO images, 1000 tensor Gauss-Legendre cell points,
64-point batches and contracted s/p/d shells. LDA restricted, PBE spin-polarized
and BLYP Cartesian spin-polarized cases use complex positive density matrices.
The independent oracle evaluates Gaussian polynomials and their analytic
spatial derivatives over the whole grid, evaluates libxc point energies and
potentials, and contracts the full density-gradient adjoint separately.
Every timed result must match energy, V and integrated populations within
2e-10. Oracle wall time is reported for attribution, not as a production
speed baseline.

Admission includes the existing source copies and compensated output plus
bounded AO/gradient panels, local density/gradient/invariant/response vectors
and a conservative libxc per-component allowance. RSS includes Python,
libraries, fixtures and the full-grid oracle. These small tests establish
same-source correctness and bounded panel allocation, not end-to-end SCF
speedups, production grid accuracy or infinite-lattice tail control.

## Measured qualification

The fresh native build on 2026-10-08 passed 738 focused checks covering the
62 semilocal tests, the unchanged Dirac source, AO evaluation, XC, the joint
functional solver, citations and repository gates. Sphinx and source privacy
checks passed. Measurements below followed the regression workload, using
libxc 7.0.0 and five repeats in each fresh process.

| Profile | Cold (s) | Median (s) | Range (s) | CPU median (s) | Modeled peak (bytes) | Panel (bytes) | RSS (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| lda-restricted | 0.007389 | 0.008242 | 0.007119 to 0.009051 | 0.008036 | 11787737 | 34304 | 182.0 |
| pbe-spin | 0.024242 | 0.022870 | 0.022692 to 0.025336 | 0.022407 | 11910089 | 137216 | 179.5 |
| blyp-cartesian | 0.018158 | 0.020533 | 0.018390 to 0.041499 | 0.019623 | 11933549 | 151552 | 180.8 |

Each case executed 1296 shell evaluations, 162000 primitive/grid pairs and
16 functional batches. Maximum timed-result errors were 1.066e-14 Ha in
energy, 3.775e-15 Ha in potential elements and 2.132e-14 electrons in
quadrature populations. Independent full-grid oracle times were 0.075 to
0.086 s. These are measured small cases, not a scaling or speedup claim.

Native binary SHA256: `ba30ba55dc8ee52d85d18519e959e31becaeefc0c2b9287bf58194a7c9cde15e`.

A separate reviewer compiled the actual source/bindings in an isolated
module. Its 400 native calls covered LDA/PBE/BLYP, both spin conventions,
Cartesian/pure s-through-g shells (35/25 AOs), five asymmetric k points and
seven asymmetric images. Independent monomials or SciPy real spherical
harmonics with five-point spatial derivatives gave maximum energy error
4.89e-15 Ha, potential error 5.44e-14 Ha and 72 real/imaginary density
variation errors below 2.89e-12. Omitting the mixed-spin GGA term produced
potential errors of 0.0195 to 0.0254 Ha, so that contraction is discriminated.
Shell-gauge and reciprocal-relabel errors stayed below 9.10e-16.

All five exact/one-short caps, unsupported functional refusals, nonfinite
weights, custom mixtures, captured alias identity and returned ownership
were checked independently. Four old/refactored Dirac profiles were bitwise
identical in values, energy, populations and source hashes. A 20000-character
functional-name increase admitted exactly 80000 additional bytes.

Grid refinement is separate from algebraic agreement. For the fixed-density
PBE fixture, the 80- and 96-point-per-axis tensor rules differed by
1.080e-8 Ha; the 64-point rule differed from the 96-point rule by
6.355e-8 Ha. Coarser rules converge nonmonotonically, so no general production
grid accuracy is inferred. Independent image refinement also remains required.
Complete physical KS drivers and their own acceptance gates remain open.
