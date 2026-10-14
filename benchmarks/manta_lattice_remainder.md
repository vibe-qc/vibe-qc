# Finite lattice compact Taylor remainder evidence

Run `.venv/bin/python benchmarks/manta_lattice_remainder.py` from a rebuilt
editable checkout containing the new native API. Six isolated single-thread
workers cover cubic/skew cells at total degrees 2, 6 and 12. After one warm-up,
each measures five complete remainder constructions, including admission,
canonical domain snapshots, support validation, positive integral envelopes,
logarithmic accumulation and immutable results. Inputs contain 27 real images,
342 nonzero reciprocal indices and one near image. No matrix is built in
the timing interval. Coefficient construction and independent validation are
outside that interval and inside process wall time and peak RSS.

Measured on Darwin arm64, Python 3.14.8 and NumPy 2.5.3. Native SHA-256:
`b84d0699dbe4b93c6b044e8b7ce8d88b73b1fbac966c47104fd68f691a29a9eb`.

| Cell | Order | Median wall ms | Median CPU ms | Process peak MiB | Modeled peak bytes |
|---|---:|---:|---:|---:|---:|
| Cubic | 2 | 0.107 | 0.108 | 159.88 | 8440512 |
| Cubic | 6 | 0.120 | 0.120 | 158.36 | 8440512 |
| Cubic | 12 | 0.164 | 0.164 | 163.77 | 8440512 |
| Skew | 2 | 0.115 | 0.115 | 158.42 | 8440512 |
| Skew | 6 | 0.145 | 0.140 | 159.16 | 8440512 |
| Skew | 12 | 0.163 | 0.163 | 163.30 | 8440512 |

The modeled peak includes the default 8 MiB backend margin. Actual radial
integral counts are 54, 108 and 189 at orders 2, 6 and 12; reciprocal counts
are 342 in each case. The memory/work model is not process RSS or measured
FLOPs. Independent SciPy incomplete-gamma evaluations are covered component
by component by the positive native envelopes.

The complete calculation is the cross electrostatic energy of two signed
compact point clouds, with three charges each. Exact listed near interactions
plus the native lattice Taylor tail are compared with a direct finite Ewald
Green sum over all nine charged pairs. A neutral source component still uses
its nonzero absolute-charge norm in the bound. These finite measures require
no Gaussian-tail substitution or nuclear point self subtraction.

| Cell | Order | Absolute cross-energy error (Eh) | Compact Taylor bound (Eh) |
|---|---:|---:|---:|
| Cubic | 2 | 1.342e-05 | 1.053e-02 |
| Cubic | 6 | 4.937e-11 | 2.725e-06 |
| Cubic | 12 | 1.180e-16 | 7.855e-11 |
| Skew | 2 | 1.342e-05 | 1.054e-02 |
| Skew | 6 | 7.032e-11 | 2.908e-06 |
| Skew | 12 | 3.331e-16 | 1.010e-10 |

The bound is deliberately conservative. At order 12 the measured error is
at the scale of binary64 evaluation noise, which is not part of the Taylor
ledger. Tests additionally check adaptive order, near-set transfers, support
singularities, covariance, scaling, source binding, positive underflow and
optional 70-digit integral references. The bound assumes valid caller radii
and absolute charge norms. It excludes finite-domain lattice error, Gaussian
penetration/full-moment tails and coefficient/contraction roundoff.

There is no matched ORCA/CRYSTAL reference for this arbitrary compact-source
primitive. External parity, production periodic BUPO-J/RI-BUPO-J, SCF
accuracy profiles, derivatives and symmetry skipping remain unqualified.

An additional complete calculation through `vq`, using source `8bca4fac`
and the native binary above, passed 24 cases: cubic/skew cells, two Ewald
splitting parameters, two near domains and orders 2, 6 and 12. All nine
charged-pair values per case and the total cross energy stayed within their
respective bounds. At order 12, the maximum cross-energy error was
2.38e-15 Eh and the maximum individual pair error was 1.14e-14 Eh. Four
adaptive calculations reached a 1e-8 Eh bound at orders 9 or 10; the largest
observed energy error was 2.71e-15 Eh. Independent incomplete-gamma values
were covered component by component. These results qualify the same finite
compact-source approximation, with the exclusions above.
