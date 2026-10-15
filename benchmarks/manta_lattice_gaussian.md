# Finite lattice Gaussian approximation ledger evidence

Run `.venv/bin/python benchmarks/manta_lattice_gaussian.py` from a rebuilt
editable checkout containing the native API. Six fresh single-thread workers
cover cubic/skew cells at total degrees 0, 6 and 12. After one warm-up, each
measures five joint coefficient, compact remainder and Gaussian error-ledger
constructions. Timing includes admission, input snapshots, canonical domains,
reciprocal sorting and immutable results. Native producer envelopes are
prepared beforehand. The domains contain 27 real images, 124 nonzero
reciprocal indices and one near image.

Independent validation, Gaussian moment construction and exact near
interactions are outside the timed interval and inside process wall time
and peak RSS. Measurements use Darwin arm64, Python 3.14.8 and NumPy 2.5.3.
Native SHA-256:
`37ae150a52d885aaa4ddd3b0cdd1122d5fd73b9772aabe4d257eeddaae319d0b`.

| Cell | Order | Median wall ms | Median CPU ms | Process peak MiB | Modeled peak bytes |
|---|---:|---:|---:|---:|---:|
| Cubic | 0 | 0.096 | 0.095 | 167.81 | 8465984 |
| Cubic | 6 | 0.272 | 0.272 | 166.39 | 8600952 |
| Cubic | 12 | 2.186 | 2.186 | 170.58 | 11945040 |
| Skew | 0 | 0.096 | 0.096 | 167.11 | 8465984 |
| Skew | 6 | 0.276 | 0.276 | 166.53 | 8600952 |
| Skew | 12 | 2.225 | 2.225 | 176.31 | 11945040 |

The modeled peak includes the default 8 MiB backend margin. Evaluated
full/tail moment products number 1, 924 and 18564 at these orders; each
construction visits all 124 reciprocal vectors for penetration bounds.
The modeled storage/work is neither measured process RSS nor measured FLOPs.

The complete calculation is the finite Ewald electrostatic energy of three
normalized Gaussian sources with charges 1, -0.7 and -0.3. It includes all
nine charged-pair interactions and finite Gaussian self integrals. Exact
listed near interactions plus the native lattice Taylor tail are compared
with independent analytic Gaussian convolutions of the same finite Green
function. The reported energy bound contracts every absolute pair bound
with absolute charges, without relying on neutrality to cancel terms.

| Cell | Order | Absolute energy error (Eh) | Gaussian approximation bound (Eh) |
|---|---:|---:|---:|
| Cubic | 0 | 1.102e-10 | 3.173e-01 |
| Cubic | 6 | 2.220e-16 | 1.693e-01 |
| Cubic | 12 | 0.000e+00 | 9.284e-02 |
| Skew | 0 | 1.538e-10 | 3.208e-01 |
| Skew | 6 | 4.441e-16 | 1.767e-01 |
| Skew | 12 | 2.220e-16 | 9.898e-02 |

The bounds are deliberately conservative. The separate timed pair/auxiliary
fixture has a penetration bound of about 132 in its declared coefficient
normalization; the measured integral error decreases from 4.15e-3 at order 0
to 2.21e-10 at order 12. A consumer must inspect the actual bound and refine
the support or use direct evaluation when its target is not met. Tests also
demonstrate an explicit 1e-8 target in the tight-Gaussian limit and check
signed angular contractions against independent polynomial Gaussian
quadrature, including quadrature refinement.

The three ledger terms cover compact Taylor truncation, Gaussian penetration
and replacing interior moments by full moments. They exclude omitted lattice
and pair-image domains, coefficient/moment/contraction roundoff, point-nuclear
self conventions and density-dependent Fock aggregation. There is no matched
ORCA/CRYSTAL reference for this arbitrary finite Gaussian-source primitive.
External parity, production periodic BUPO-J/RI-BUPO-J, SCF accuracy profiles,
derivatives and symmetry omission remain separate qualifications.

A complete calculation through `vq`, using source `1b0f36f6` and the native
binary above, passed 24 cases: cubic/skew cells, two Ewald splitting
parameters, two near domains and orders 0, 6 and 12. Every charged-pair and
total-energy error stayed within its respective bound. The maximum energy
errors at those orders were 4.02e-8, 6.86e-13 and 8.89e-16 Eh; the largest
individual pair error at order 12 was 1.78e-15 Eh. Four signed angular
pair/auxiliary combinations passed the independent quadrature and signed
density-contraction comparisons. Their largest component error was 8.37e-5
in the declared unnormalized coefficient convention, covered by the
corresponding conservative Gaussian bound.

Four tight-source calculations reached an explicit 1e-8 bound at order 5.
Their maximum reported bound was 3.10e-9 and largest measured error was
1.67e-16. Source and native hashes were unchanged before and after the run;
all fetched numerical artifacts matched their recorded hashes. These results
qualify the finite Gaussian approximation with the exclusions above.
