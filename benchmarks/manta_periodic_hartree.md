# MANTA finite periodic Hartree microbenchmark

Run the rebuilt editable checkout with:

```sh
.venv/bin/python benchmarks/manta_periodic_hartree.py > /tmp/manta-periodic-hartree.json
```

This benchmark computes the same finite 3D zero-average reciprocal Hartree
operator using the existing bounded reference interface and the selected
MANTA producer. It has nine pure s/p/d AOs, seven explicit images, a skew
lattice and a 6-Ha reciprocal cutoff (14 nonzero G vectors). Gamma and
three arbitrary complex Hermitian density blocks use identical basis and
support conventions; the latter has nonuniform weights 0.2, 0.3 and 0.5.
This is a small operator benchmark, not a solid, SCF, BUPO, symmetry, exchange
or thermodynamic-limit benchmark.

Each row runs in a fresh single-thread process with one warm-up and five
measured builds. Build timing includes grid generation, admission, integral
generation, contractions and owned J(k) export. Fixture setup and independent
reference validation are separately timed in the JSON, which also records
CPU time and total worker time. The reference route retains its angular
conversion; MANTA already produces physical AO normalization. Every output
is checked against the same-source reference J and Hartree energy.

The October 2026 measurement used ARM64 macOS, Python 3.14.8, NumPy 2.5.3,
native optimization and binding LTO disabled. The core SHA-256 was
`4620f3afdf436bbed48e6039e0b14014caa38944bdcb7ec6b0f0749bf768e117`.
Other work was running concurrently, so wall-time ranges include contention.

| Case | Provider | G batch | Build median ms | Build range ms | Median CPU ms | Peak process MiB |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Gamma | Reference | 4 | 2557.604 | 2474.441-2606.052 | 2304.534 | 144.70 |
| Gamma | MANTA | 4 | 16.377 | 15.945-18.636 | 14.270 | 144.91 |
| Gamma | Reference | 32 | 633.080 | 613.710-648.821 | 584.188 | 145.25 |
| Gamma | MANTA | 32 | 8.522 | 7.580-9.735 | 8.161 | 144.89 |
| Multi-k | Reference | 4 | 2863.956 | 1536.321-2998.243 | 2308.649 | 144.06 |
| Multi-k | MANTA | 4 | 13.196 | 12.923-13.705 | 12.445 | 145.02 |
| Multi-k | Reference | 32 | 714.465 | 689.223-778.459 | 588.848 | 145.44 |
| Multi-k | MANTA | 32 | 6.235 | 5.829-7.410 | 6.112 | 145.00 |

The largest J discrepancy was `5.80e-15`, and the largest Hartree energy
discrepancy was `2.98e-14`. MANTA was faster on this fixture. The reference
per-cell Fourier interface has substantial setup/Python overhead on tiny
panels; these ratios do not predict a speed advantage over a cached reference
source, a complete SCF calculation or another integral implementation.

MANTA's measured shell calls fell from 504 to 126 when the entire reciprocal
support fit in one batch. Actual primitive pairs fell from 2016 to 504;
vector-primitive products stayed at 7056. These generation counts are the
same for Gamma and multi-k, because density-side Fourier work is shared
across momenta. Phase accumulation rose from 15876 to 31752, and contraction
terms from 9639 to 13041. These are executed counters; no internal reference
primitive count is inferred. No BUPO or symmetry reduction is involved.

Modeled peak storage ranged from 16.38 to 16.65 MiB. Process peak RSS includes
imports, the fixture, the builds and reference validation, and is dominated
by the runtime. It is not a native workspace measurement. Neither numerical
identity, small errors nor these timings establish convergence of the finite
image or reciprocal support for a physical crystal.
