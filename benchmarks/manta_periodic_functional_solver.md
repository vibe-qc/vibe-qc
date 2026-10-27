# Bounded periodic functional numerical contract

Run `.venv/bin/python benchmarks/manta_periodic_functional_solver.py --repeats 5`
from a freshly rebuilt editable checkout. Fresh single-thread workers exercise
restricted, unequal-spin and empty-beta models with three k points and three
basis functions. One cold call precedes five timed repetitions. Each timed
call includes admission, overlap analysis, eigensolves, whole-density joint
callbacks, diagnostics and owned numerical result construction. Imports and
the independent oracle are outside timing but inside peak process RSS.
The previous result is released before each timed call. Modeled numerical
totals cover the solver, borrowed S/H and joint provider; other Python
fixtures and oracle arrays are outside this leaf and included only in RSS.

The model is deliberately nonlinear: with
`rho=sum_spin,k Tr(D A)/K`, the interaction energy is
`a*rho^2/2+b*rho^3/3` and its potential is `(a*rho+b*rho^2)*A`.
The cubic term makes the HF half-trace formula wrong. Independent NumPy
fixed-point iterations and finite-difference derivatives check both the
potential and energy, including complex off-diagonal densities. This is
numerical solver evidence, not a physical XC quadrature or KS job benchmark.

A separate compiled C++ API harness also compared six cases with an
independent SciPy overlap/NumPy SCF oracle: K=1,2,3,4,9 and n=5,6; restricted,
unequal and empty/reversed spin populations; full and truncated metrics.
Maximum errors were 7.032e-11 in density, 4.993e-16 hartree in potential/Fock,
and 1.865e-14 hartree in total energy. These dimensions exceed the tiny
Python diagnostic interface, which intentionally caps K at eight and n at
four. The harness uses the same public C++ entry points as future physical
adapters, and does not call the Python diagnostic provider.

The existing HF entry points retain their signatures, provider layouts and
energy convention. Functional entry points use a separate joint provider and
charge its declared retained bytes, peak workspace and work, plus eight
bytes for the energy scalar and 32 bookkeeping work units per evaluation.
Output numerical ownership remains the same as the corresponding HF leaf.
Missing energy, including a missing write after an earlier successful epoch,
is rejected; no previous iteration's scalar is reused. Potential projection
diagnostics do not silently redefine the functional energy.

## Measured tiny numerical models

Measured on 2026-10-08, arm64 macOS, Python 3.14.8, with BLAS/OpenMP
threads set to one. Native SHA256:
`c4627380d79982f3c4443c3a8e098f83410030cfa8c265ca5935e9ba73869fdb`.
Times are microseconds. These sub-millisecond fixtures measure interface and
small-matrix overhead; they establish no scaling law or physical KS speedup.

| Profile | Cold | Median (min, max) | CPU median | RSS MiB | Owned bytes | Total numerical bytes | Evaluations | Charged work units |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| restricted | 391.46 | 263.92 (260.46, 301.83) | 264.00 | 155.23 | 2400 | 3784 | 7 | 97554656 |
| unequal-spin | 746.46 | 570.63 (469.38, 1344.13) | 498.00 | 156.50 | 3768 | 5584 | 6 | 165086912 |
| empty-beta | 320.21 | 226.38 (214.08, 263.25) | 227.00 | 156.80 | 3768 | 5584 | 5 | 142057376 |

All three workers passed. Maximum errors across timed results were
3.843e-11 in density, 1.085e-16 hartree in Fock and
1.332e-15 hartree in total energy. Restricted provider retention
is 456 bytes; each unrestricted provider retains
888 bytes. Every profile admits 64 bytes of
provider workspace. These declared numerical quantities are distinct from RSS.
