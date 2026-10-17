# Finite periodic lattice multipole evidence

Run `.venv/bin/python benchmarks/manta_lattice_tail.py` from the rebuilt
editable checkout. Six fresh single-thread processes cover cubic and skew
cells at total degrees 2, 6 and 12. Each warms up once and measures five
complete finite lattice builds and one owned matrix export. Timing includes
admission, canonical domain sorting, radial/reciprocal differentiation,
compensated accumulation and the constant gauge. Input integer-domain
generation is outside this interval. Each source contains 27 real images,
342 nonzero reciprocal indices and the origin-only near set.

Validation is outside build timing and inside process time/RSS. Every matrix
entry is compared with independent Gauss-Legendre integration of differentiated
Gaussians using SciPy Hermite polynomials. Order-12 workers additionally
evaluate nine charged Gaussian component pairs and their neutral combined
energy, using 343 real images, 6858 reciprocal indices and 27 near images.
Their independent reference is the analytic Gaussian Fourier Green function
on a larger reciprocal support. The near interactions include full Gaussian
self interactions; there is no nuclear point self subtraction.

Measured on Darwin arm64, Python 3.14.8 and NumPy 2.5.3, with native SHA-256
`440b89f5bdfc8a479c4d9dce98a4fd7465f66b048efe3d72c63191b08d35b372`.

| Cell | Order | Median wall ms | Median CPU ms | Process peak MiB | Modeled peak bytes |
|---|---:|---:|---:|---:|---:|
| Cubic | 2 | 0.093 | 0.093 | 159.08 | 8444008 |
| Cubic | 6 | 0.255 | 0.255 | 159.83 | 8575640 |
| Cubic | 12 | 0.956 | 0.957 | 175.86 | 11919728 |
| Skew | 2 | 0.073 | 0.074 | 159.25 | 8444008 |
| Skew | 6 | 0.255 | 0.255 | 159.86 | 8575640 |
| Skew | 12 | 1.281 | 1.281 | 175.83 | 11919728 |

The maximum independent coefficient difference is `8.89e-16`. Modeled peaks
include the default 8 MiB native margin. Process RSS also includes Python,
NumPy/SciPy and reference validation. Counts are 270/2268/12285 evaluated
radial coefficients and 3420/28728/155610 reciprocal coefficients at orders
2/6/12; these are not estimated skipped lattice interactions.

The cubic Gaussian energy is `0.057647852252410904` hartree, compared with
`0.05764785225241087` from the independent Fourier sum. The skew-cell values
are `0.05774618288965329` and `0.057746182889657396` hartree. Maximum errors in
individual charged pair interactions are `4.45e-15` and `1.33e-14` hartree.
Testing the individual pairs prevents a neutral total from hiding a dropped
constant or trace contribution.

These qualify finite coefficients and the stated smooth-charge fixtures.
They do not certify lattice/Taylor/Gaussian error bounds, a periodic BUPO
traversal, SCF energies, external ORCA/CRYSTAL parity or a speedup against a
production periodic Coulomb implementation. Other checks cover near-set
transfers, Ewald split/refinement, Poisson trace, retained quadratic/constant
terms, rotations, expansion-center changes, source ownership and exact
resource boundaries. Explicitly insufficient domains remain unqualified.
