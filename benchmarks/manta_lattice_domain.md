# Infinite Ewald cube-domain error evidence

Run `.venv/bin/python benchmarks/manta_lattice_domain.py` from the rebuilt
editable checkout. Six fresh single-thread workers cover cubic/skew cells
and centered real/reciprocal integer cubes of radii 1, 3 and 6. Five native
bound constructions follow a warm-up. Sorting, validation, inverse-residual
metric checks and result exports are timed. Infinite omitted cells are not
enumerated. Independent full ionic energies run outside the timed region
and inside process wall time and peak RSS.

Measurements use Darwin arm64, Python 3.14.8 and NumPy 2.5.3. Native SHA-256:
`1921724cd0d4b4c0f0bc964cefe1f78f0632460f7f3ec2fff4f86137a48dec08`.

| Cell | Cube radius | Wall/CPU ms | Process peak MiB | Modeled peak bytes |
|---|---:|---:|---:|---:|
| Cubic | 1 | 0.017/0.017 | 173.52 | 8410176 |
| Cubic | 3 | 0.108/0.108 | 171.92 | 8470848 |
| Cubic | 6 | 0.768/0.767 | 172.64 | 8826816 |
| Skew | 1 | 0.017/0.017 | 170.27 | 8410176 |
| Skew | 3 | 0.110/0.110 | 170.11 | 8470848 |
| Skew | 6 | 0.788/0.788 | 169.95 | 8826816 |

The modeled peak includes the default 8 MiB backend margin and all admitted
copies. Modeled memory/work is neither process RSS nor measured FLOPs. Short
wall/CPU differences include scheduling variability.

The calculation is the complete Ewald energy of a neutral two-ion cell,
including the regular point-self term. Independent SciPy erfc/cosine sums
at real radius 7 and reciprocal radius 15 supply the reference. The finite
kernel is assembled from native order-zero lattice-tail coefficients and
exactly the matching near bare terms, avoiding a duplicated reciprocal part.

| Cell | Cube radius | Energy error (Eh) | Omitted-domain energy bound (Eh) |
|---|---:|---:|---:|
| Cubic | 1 | 3.553e-05 | 6.257e+01 |
| Cubic | 3 | 9.992e-16 | 6.083e-04 |
| Cubic | 6 | 2.220e-16 | 1.511e-13 |
| Skew | 1 | 4.703e-05 | 8.247e+01 |
| Skew | 3 | 9.936e-15 | 8.080e-04 |
| Skew | 6 | 1.110e-16 | 1.861e-13 |

Bounds use the count `24*m*m+2` of integer maximum-norm shells, a conservative
lattice metric, the positive erfc Gaussian envelope, and a first-term-plus-
integral upper bound for the remaining series. Absolute charge norms multiply
separate real and reciprocal errors. The source binds its finite kernel and
domains; changing the common near subtraction changes identity without
changing the omitted-domain bound. See the scientific contract for the proof.

Tests check the analytical formula against independent positive infinite
shell sums, signed compact charge energies, translated-source periodicity
defects and explicit 1e-8 Eh adaptive cube targets. The finite real sum is
not declared exactly periodic. Native roundoff is excluded from these domain
bounds; comparisons retain a separate 5e-14 Eh numerical allowance. The
unchanged coefficient and compact Taylor routines preserve eight saved
matrices, plans, identities and remainders bitwise.

Ewald (1921), DOI 10.1002/andp.19213690304, section IV.1, Eqs. 37-40
supplies the neutral split and regular point-self reference. The cube-envelope
bound is a separate MANTA derivation. No matched ORCA/CRYSTAL finite ionic
kernel calculation is available; no external parity or electronic SCF claim
is made. This conditional compact-source qualification excludes unbounded
Gaussian tails, finite AO pair-image omission, roundoff, full charge-policy
selection and production periodic BUPO/RI-BUPO admission.

## Complete local calculation acceptance

The MANTA source at `fee1d1d8c6df91b3dca8bb32df3db2ec1a7970f0` matches the validated
snapshot `938442b30adb7402dd5231854937d1eccd9bc25c`. Local `vq` job
`ac47dc84e4bb` used the native SHA above. Before/after and rebase checks
pin ten implementation/reference files and the native binary. All 44 fetched
numerical artifacts match their original queue workspace, and ionic energies
were recomputed from the fetched matrices and charge vectors.

The run covers 24 fixed complete two-ion energies: cubic/skew cells, two
Ewald splits, original/lattice-translated ion representatives and three cube
sizes. Eight adaptive calculations enlarge the cube until its energy bound
is below 1e-8 Eh. They succeed at radii 5 through 8; the largest accepted
bound is 9.657e-9 Eh and the largest measured error is 3.89e-16 Eh. Including
intermediate adaptive steps, 42 unique ionic calculations were evaluated;
valid earlier cases were reused. The largest fixed-case error is 5.50e-8 Eh,
covered by its domain bound with the separate numerical allowance above.

Two signed compact-source checks measure periodicity defects of 2.48e-8
and 7.26e-8 Eh. Their conservative real-domain bounds are 53.31 and 33.73 Eh
at the deliberately short fixed cutoffs. This confirms the error accounting,
not a useful accuracy profile at those cutoffs. Enlarging the cube is required
when a requested bound is not met, as the adaptive ionic cases demonstrate.
No exact finite-kernel covariance or Gaussian-source certificate is inferred.

All 31 focused tests pass; the two translation cases also pass the stronger
real-only defect bound. Eight prior coefficient/compact-remainder fixtures
are bitwise unchanged. Manifest, citation, changelog, documentation and staged
privacy checks pass. The benchmark and the local calculation use independent
numerical references with the scope and external-reference limitation above.
