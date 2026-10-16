# Full periodic exchange bounds

`PeriodicExchangeBoundSource` bounds omitted reciprocal and AO images for
the full-Coulomb exchange functional at fixed original mesh and density.
Explicit periodic BUPO RHF/UHF admission includes the overlap-dependent
correction at the same fixed Madelung coefficient. Numerical values and
finite-source identities remain unchanged by certification.

Run `.venv/bin/python benchmarks/manta_periodic_exchange_bounds.py --repeats 3`.
Each case runs in an isolated process with one numerical thread. Immutable
pair/BUPO setup is excluded from certificate construction timing; full HF
timing includes certificate construction, every evaluated density and the
terminal confirmation. Process RSS includes Python and imported references.
These small cases make no scaling or speedup claim.

Three isolated Darwin arm64 workers use Python 3.14.8, NumPy 2.5.3 and three
repetitions. The measured MANTA implementation matches source `5fdd40ea`;
the tested native SHA-256 is
`3ffcab7c814fcb854f9cea3f8c06c084cc0a460963e1331dfae87bcf376b48dd`.

| Case | Q vectors | Constructor ms | Density bound microseconds | Complete HF seconds | Bound model MB | HF model MB | RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse-sp | 498 | 0.0818 | 29.63 | - | 8.540 | - | 205.12 |
| certified-rhf | 226980 | 49.65 | 12.33 | 1.223 | 37.579 | 282.513 | 194.19 |
| certified-empty-beta | 907922 | 197.70 | 16.87 | 4.356 | 124.744 | 533.161 | 250.23 |

The maximum observed method-scaled exchange matrix and energy bounds are
1.052e-12 and 4.131e-12 for RHF, and 1.607e-12 and 4.521e-12 for UHF,
respectively. Empty beta has exactly zero exchange error. The deliberately
poor sparse source has an unscaled K bound of 2.512e9 and fails any tight
production target. Small model/timing values do not imply chemical accuracy.

The sparse s/p case uses signed contractions, a skew cell, two arbitrary
k points and a deliberately incomplete image/cutoff domain. Its large error
bound is not a production target. Certified He RHF and empty-beta H UHF
use a cubic side of 6 bohr, exponents 4 and 6, original image cube 1,
physical cutoff 1 bohr, reciprocal exchange cube 30 and the original mesh's
BvK correction. Matrix/energy targets stay at 1e-8; overlap and kinetic
targets also remain unchanged.

Independent contour-shifted Gauss-Hermite quadrature integrates the complex
Cartesian polynomials exactly through the tested angular degrees and remains
stable at large Q. Direct real-axis quadrature checks the small-Q convention.
Unreduced ordered-image sums compare finite, image-enlarged and jointly
image/reciprocal-enlarged K responses for general complex signed densities.
Separate reciprocal-shell and AO-image refinements check the reference.
The complete HF reference uses independent Gaussian electrostatics and a
separate generalized-eigenproblem SCF, with the same finite exchange supports.

Seven scenarios were assessed through actual local `vq` calculations: two
full-exchange comparisons and five HF cases, four converged and one stopped
at its iteration cap. Twelve accepted native calculations include five
uncertified HF controls; eight previously completed calculations were reused.
Certification leaves D, F, H, S and total energy bitwise unchanged in every
control. A separate fetched-array assessment recomputes the matrix enclosures,
weighted energy bounds, BvK overlap correction and HF energy traces.

| HF case | Energy, hartree | Maximum observed K matrix bound | Maximum observed exchange energy bound |
|---|---:|---:|---:|
| He RHF, Gamma, strict zero mode | 1.174898184838615 | 1.037e-12 | 4.074e-12 |
| He RHF, shifted mesh, BvK | 0.704358935947042 | 1.123e-12 | 4.410e-12 |
| H UHF, empty beta, Gamma, BvK | 1.990966734222167 | 1.504e-12 | 4.232e-12 |
| Li UHF, shifted mesh, BvK | 4.577272041683743 | 6.433e-20 | 1.003e-17 |
| Li UHF, one iteration, BvK | 4.671722068618250 | 5.809e-20 | 8.107e-18 |

The HF cases use the same 6-bohr cell and exponents 4 and 6, with exponent
9 added for Li. Li uses AO image cube 2, cutoff 6.1 bohr, fixed Ewald eta
0.5, real/reciprocal Ewald cubes 4/10 and exchange cube 40. The initial
smaller image source and eta 0.6 correctly fail electrostatic energy
admission. The final configurations retain all original accuracy targets.

The largest HF reference differences are 1.250e-11 hartree in energy,
4.104e-13 in F and 8.772e-10 in D. The largest recomputed native energy-trace
residual is 9.666e-12 hartree. Finite K agrees within 7.120e-14; separate
reciprocal-shell and image refinements change its reference by at most
1.637e-23 and 2.129e-24, respectively. Numerical comparison allowances are
3e-11 for energy/matrices and 3e-9 for density, separate from the analytical
bounds, which exclude roundoff. Two tiny exchange targets are refused.

For the larger Li reciprocal sum, the independent reference uses compensated
scalar summation and reuses its fixed tiny-system operator. Its 1e-13 SCF
stopping tolerance measures the occupied projector in orthonormal coordinates;
the raw AO coefficients amplify roundoff with overlap condition number 1472.
These reference refinements resolve the earlier comparison failures against
the preserved, bitwise-identical native result without changing native
equations, solver settings or acceptance tolerances. The capped case reuses
that independently integrated operator and converged reference.

The comparison qualifies errors at the evaluated density and fixed mesh and
Madelung coefficient. It does not bound k-mesh convergence, Madelung evaluation,
basis incompleteness, relaxed-state or overlap-rank sensitivity, or roundoff.
There is no matched ORCA/CRYSTAL finite-support exchange certificate;
independent numerical references and domain refinement provide the comparison.
Whole-method profiles, public job selection and q-aware RI-BUPO remain open.
