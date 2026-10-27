# Finite periodic exchange benchmark

Run `.venv/bin/python benchmarks/manta_periodic_exchange.py` from the rebuilt
editable checkout. Twelve fresh single-thread workers cover Gamma and three
momenta, one/two density channels, and the existing Fourier reference versus
MANTA reciprocal batches 4/32. Each warms up once and measures five complete
exchange builds, including owned exports.

The fixture has nine pure s/p/d AOs, two primitives per shell and seven AO
images on a skew lattice. Each target/source pair uses seven reciprocal labels
shifted by the momentum difference, with exact Q=0 removed. Gamma retains six
vectors; the three-point mesh retains 60 across nine transfers. Both providers
share Fourier factors across all density channels in each build. The reference
uses the existing analytical native Bloch Fourier operator at the negative
source momentum, its physical AO normalization, and the existing linear
exchange contraction. MANTA uses its native target-momentum convention.

These are finite full-Coulomb operators with Q=0 omitted and no Madelung
correction. This benchmark does not establish reciprocal/image convergence,
corrected periodic HF, SCF, range separation, derivatives or BUPO performance.
Build timings exclude validation, which is included in total worker time and
RSS. JSON also reports minimum/maximum build times and actual work.

Measured on Darwin arm64, Python 3.14.8 and NumPy 2.5.3. Native core SHA-256:
`2eab6d945c0d83a8b697d51cc903727f5873f4f6b85c20ee2437e6415305ccf7`.

| Mesh | D channels | Provider | Batch | Median build ms | Median CPU ms | Total worker s | Peak process MiB |
|---|---:|---|---:|---:|---:|---:|---:|
| gamma | 1 | reference | 32 | 0.766 | 0.768 | 1.908 | 144.84 |
| gamma | 1 | manta | 4 | 2.204 | 2.100 | 1.229 | 145.45 |
| gamma | 1 | manta | 32 | 1.426 | 1.429 | 1.289 | 146.44 |
| gamma | 2 | reference | 32 | 0.725 | 0.726 | 1.353 | 145.27 |
| gamma | 2 | manta | 4 | 2.344 | 2.345 | 1.314 | 145.17 |
| gamma | 2 | manta | 32 | 1.492 | 1.493 | 1.267 | 145.33 |
| multi-k | 1 | reference | 32 | 6.869 | 6.872 | 1.264 | 145.86 |
| multi-k | 1 | manta | 4 | 18.662 | 18.343 | 1.279 | 145.73 |
| multi-k | 1 | manta | 32 | 9.782 | 9.740 | 0.941 | 146.09 |
| multi-k | 2 | reference | 32 | 5.023 | 5.022 | 0.823 | 146.78 |
| multi-k | 2 | manta | 4 | 14.138 | 14.032 | 0.849 | 146.17 |
| multi-k | 2 | manta | 32 | 8.486 | 8.486 | 0.812 | 146.30 |

Maximum matrix error is `1.57e-13`; exchange-energy error is `1.64e-13`.
Doubling density channels leaves all Fourier work unchanged, while doubling
matrix products as expected. At three momenta, increasing the batch from 4
to 32 reduces Fourier panel calls from 18 to 9, shell calls from 1134 to 567
and primitive-pair evaluations from 4536 to 2268. Vector-primitive products
remain 15120. Gamma shows the same factor-of-two reduction. Modeled native
peak storage ranges from 16.26 to 16.35 MiB, including conservative backend
workspace; process RSS also includes Python, libraries and reference caches.

MANTA is slower on this small fixture. Timing variation between separate
workers is visible, so the shorter two-channel measurements do not imply
that adding a density improves speed. The work counters demonstrate sharing;
these measurements make no general speedup claim.
