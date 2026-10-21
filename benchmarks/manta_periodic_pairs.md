# Periodic pair-image source evidence

Run `.venv/bin/python benchmarks/manta_periodic_pairs.py` from a rebuilt
editable checkout. Six fresh single-thread workers cover cubic/skew cells
at moment orders 0, 6 and 12. Each measures five constructions, forward
density contractions and adjoint responses after a warm-up. The fixture has
four AOs, five explicit images, three non-TRIM momenta and unequal weights.
Its 20 ordered shell/image products become 11 conjugate representatives.
The source streams 11 moment panels per pass; it does not retain a complete
AO/image/moment tensor. Setup includes native Gaussian extent construction.

Independent validation is outside the timed regions and inside process wall
time and peak RSS. Measurements use Darwin arm64, Python 3.14.8 and
NumPy 2.5.3. Native SHA-256:
`8cfe730d6cbb5334c7c52f79f492f84169a44a59330479ecde04c7e3881acd3b`.

| Cell | Order | Setup wall/CPU ms | Forward wall/CPU ms | Adjoint wall/CPU ms | Process peak MiB | Modeled peak bytes |
|---|---:|---:|---:|---:|---:|---:|
| Cubic | 0 | 0.296/0.296 | 0.042/0.042 | 0.036/0.036 | 168.08 | 8499473 |
| Cubic | 6 | 22.885/3.583 | 0.951/0.952 | 0.918/0.917 | 166.92 | 8648417 |
| Cubic | 12 | 15.877/12.105 | 28.137/8.495 | 11.486/8.721 | 168.12 | 9288113 |
| Skew | 0 | 0.309/0.309 | 0.043/0.042 | 0.037/0.037 | 167.19 | 8499473 |
| Skew | 6 | 4.728/3.735 | 1.081/1.036 | 0.971/0.971 | 168.33 | 8648417 |
| Skew | 12 | 27.657/11.471 | 44.696/8.409 | 46.010/8.671 | 167.73 | 9288113 |

The differences between wall and CPU time reflect scheduling
variability in these short runs. The modeled peak includes the default
8 MiB backend margin. Modeled work/storage is neither measured FLOPs nor
process RSS. The plan covers setup and one forward/adjoint cycle; it is not
a lifetime call quota or a complete SCF resource plan.

The complete calculation is a finite reciprocal Coulomb energy of the
paired AO density. Raw Cartesian moments generate a declared Fourier Taylor
approximation at each representative midpoint. The native adjoint returns
its AO response, and the weighted half trace recovers the same energy.
Independent unreduced, ordered Gaussian Fourier integrals supply the exact
finite-source reference. Independent polynomial Gaussian quadrature checks
the same-order moments separately.

| Cell | Order | Absolute Coulomb-energy error (Eh) | Fourier approximation bound (Eh) |
|---|---:|---:|---:|
| Cubic | 0 | 1.116e-01 | 1.283e+03 |
| Cubic | 6 | 7.640e-03 | 4.027e+01 |
| Cubic | 12 | 6.078e-06 | 1.256e-02 |
| Skew | 0 | 1.313e-01 | 1.293e+03 |
| Skew | 6 | 9.132e-03 | 4.668e+01 |
| Skew | 12 | 9.033e-06 | 1.535e-02 |

The source/adjoint implementation and the Taylor approximation are distinct.
The largest same-order Fourier discrepancy is 3.39e-15, and the energy
half-trace discrepancies are at most 1.67e-16 Eh. The order-12 approximation
still has errors up to 9.04e-6 Eh in this diffuse fixture. It is not an
automatic microhartree accuracy profile.

The diagnostic bound uses the retained absolute radial moment of degree p:
bound the exponential remainder after degree p-1 by `|G|^p F_p / p!`,
weight each pair by its absolute density-coefficient norm, then add the
absolute last retained Taylor term. If this bounds the Fourier error by b,
`sum_G v_G (|rho_G| b_G + b_G^2/2)` bounds the energy error, where
`v_G = 4*pi/(volume*G^2)`. Floating-point error remains separate.

Tests additionally cover signed pure/Cartesian high-angular contractions,
independent ordered Bloch pairing, weighted adjoint energy derivatives,
zero-weight momenta, ownership, canonical domains and all resource caps.
An arbitrary nonreciprocal Fourier counterexample explicitly excludes the
nonperiodic use of the conjugate-image identity. The existing molecular
bubble-tree admission remains unchanged.

No matched ORCA/CRYSTAL reference exists for this arbitrary finite density
and Fourier domain. No external parity, production periodic BUPO/RI-BUPO,
complete nuclear/background policy, SCF accuracy profile, spatial symmetry
or derivative capability follows from this source primitive.

## Complete local calculation acceptance

The MANTA source at `035312bdf67cb3fdcf469fb3a0ed8ddf28c3f90b` matches the validated
snapshot `11768f966e24c9fc5b9214c35ce30a020c9dde83`. Local `vq` job
`e44e368f614f` used the native SHA above. Before/after and rebase checks
pin the native bytes and 16 implementation/reference files. All 28 fetched
numerical artifacts and the run/input hashes were verified. This is
acceptance of `finite-pair-image-moments-and-adjoint` with the stated scope.

Twenty-four complete finite Fourier Coulomb calculations cover cubic/skew
cells, pure/Cartesian orbital shells, two general complex Hermitian density
seeds and orders 0, 6 and 12. Four additional fixed quadratic density
functionals check the weighted adjoint as an energy derivative, including
zero-weight momenta. Subnormal-density and unaligned-input checks pass.

The largest same-order Gaussian Fourier discrepancy is 4.75e-15; the largest
energy half-trace difference is 5.00e-16 Eh. The finite-difference derivative
discrepancy is at most 5.41e-15. Maximum finite Fourier energy errors at
orders 0, 6 and 12 are 1.810e-1, 2.373e-2 and 2.264e-5 Eh. Every Fourier
and energy error is below its explicit bound; the largest energy error/bound
ratio is 0.00217. The wider density set reinforces the distinction between
source correctness and Taylor truncation. Relative convergence rates depend
on cancellation in the density, so a universal order-to-order reduction
factor is not an acceptance criterion.

The 39 focused source tests pass. The unchanged molecular extent algorithm
also retains its 37 passing tests and 24 bitwise fixture/identity comparisons.
Documentation, manifest, citation and changelog checks pass. No matched
ORCA/CRYSTAL finite-density reference or complete SCF calculation is claimed
for this source primitive.
