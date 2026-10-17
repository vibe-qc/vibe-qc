# Screened periodic exchange benchmark

Run `.venv/bin/python benchmarks/manta_periodic_exchange.py --kernel erfc --omega 0.6`.
The fixture, twelve fresh single-thread workers, one warmup, five measured
builds and reference conventions match the [full-kernel benchmark](manta_periodic_exchange.md).
Positive-range erfc includes explicit Q=0 in each diagonal transfer support:
seven vectors at Gamma and 63 over the three-point mesh, including one/three
finite zero modes. No exchange singularity correction is applied.

The reference weights its existing analytical Fourier factors with the
screened reciprocal coefficient, evaluated with `expm1`, and the finite
`pi/omega^2` zero-mode limit. Independent quadrature and real-space Gaussian
sum tests separately check the scientific convention.

Measured on Darwin arm64, Python 3.14.8 and NumPy 2.5.3. Native core SHA-256:
`d96477732a9542a225d238940d1fed12a0ca860cce5a56eca938c4e06cecfed8`.

| Mesh | D channels | Provider | Batch | Median build ms | Median CPU ms | Total worker s | Peak process MiB |
|---|---:|---|---:|---:|---:|---:|---:|
| gamma | 1 | reference | 32 | 0.624 | 0.625 | 1.291 | 144.69 |
| gamma | 1 | manta | 4 | 2.243 | 2.205 | 1.226 | 146.48 |
| gamma | 1 | manta | 32 | 1.150 | 1.150 | 0.900 | 146.45 |
| gamma | 2 | reference | 32 | 0.522 | 0.522 | 0.786 | 146.17 |
| gamma | 2 | manta | 4 | 1.635 | 1.638 | 0.779 | 145.91 |
| gamma | 2 | manta | 32 | 1.192 | 1.188 | 0.806 | 145.64 |
| multi-k | 1 | reference | 32 | 4.857 | 4.858 | 0.811 | 145.91 |
| multi-k | 1 | manta | 4 | 14.638 | 14.190 | 0.869 | 146.06 |
| multi-k | 1 | manta | 32 | 9.763 | 9.685 | 0.841 | 146.55 |
| multi-k | 2 | reference | 32 | 8.039 | 7.972 | 1.284 | 146.52 |
| multi-k | 2 | manta | 4 | 19.117 | 19.118 | 1.283 | 145.30 |
| multi-k | 2 | manta | 32 | 13.238 | 13.238 | 1.262 | 146.00 |

Maximum matrix error is `1.41e-14`; exchange-energy error is `1.28e-13`.
Fourier work is independent of density-channel count. At three momenta,
batch 4/32 uses 18/9 panels, 1134/567 shell calls and 4536/2268 primitive-pair
evaluations. Both consume 15876 vector-primitive products and exactly three
zero modes. Matrix products increase from 126 to 252 with a second density.
Modeled peak storage is 16.26 to 16.35 MiB, distinct from process RSS.

MANTA is slower on this small fixture. These timings qualify screened-kernel
accuracy, finite-zero-mode handling and measured work sharing, not a speedup
claim or complete periodic SCF/convergence.
