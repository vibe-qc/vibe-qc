# Public periodic MANTA KS jobs

The pure-KS profiles were measured 2026-10-09 in three fresh CPU workers, each with one cold run and
five repetitions. OpenMP, OpenBLAS and Accelerate use one thread. Timings
include joint grid/domain/physical SCF admission, numerical generation,
terminal confirmation and all nine standard output artifacts. No speedup
or process RSS claim is made.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_ks_job.py
```

The fixtures use a full shifted three-point mesh, 12 radial points, Lebedev
order 17, an 8-bohr Becke partition, an 11-bohr XC-image sphere and the same
finite physical domains as `manta_periodic_ks_grid.py`. H2 RKS and HeH UKS use
PBE. The empty-beta H fixture uses BLYP; this does not qualify PBE minority
potentials at the fully polarized libxc endpoint. The deliberately small
grids are consistency witnesses, not production accuracy profiles.

| Profile | Cold (s) | Median (s) | Range (s) | Iterations | Fock calls |
|---|---:|---:|---:|---:|---:|
| rks | 0.226279 | 0.101766 | 0.100957 to 0.103406 | 3 | 4 |
| uks | 0.275591 | 0.177297 | 0.159868 to 0.180017 | 9 | 10 |
| empty-spin | 0.214523 | 0.106372 | 0.105088 to 0.110673 | 10 | 11 |

Independent Gaussian integral and libxc point contractions, nuclear lattice
sums and SciPy generalized eigensolves check the full SCF and separate
physical energy terms. QVF is validated and its evaluated spin density is
read back without orbital reconstruction.

| Profile | Density max error | Fock max error (Ha) | Energy error (Ha) | Grid-population reference error | QVF density error |
|---|---:|---:|---:|---:|---:|
| rks | 2.220446e-15 | 1.665349e-15 | 1.887379e-15 | 2.220446e-16 | 0 |
| uks | 1.329266e-10 | 5.329102e-15 | 5.329071e-15 | 1.776357e-15 | 0 |
| empty-spin | 4.623457e-11 | 2.053913e-15 | 5.551115e-16 | 8.881784e-16 | 0 |

| Profile | Job reservation (bytes) | Actual-source modeled peak (bytes) | Shell calls | Primitive pairs | Primitive/grid products |
|---|---:|---:|---:|---:|---:|
| rks | 2752512 | 82572803 | 9160 | 88 | 570240 |
| uks | 2852864 | 82732537 | 34434 | 414 | 2138400 |
| empty-spin | 2777088 | 82369597 | 12674 | 200 | 784080 |

The actual-source modeled peak already includes the job reservation, Python
receipt storage and native backend allowances. It is not measured process
memory. Grid-vs-metric population discrepancies are 0.001607, 0.001596 and
0.013755 electrons respectively; close agreement with the independent
finite-grid reference does not eliminate these finite-support errors.
Both energy and every spin/k potential require independent refinement.

The native core SHA-256 was
`048e9f90eb45c1e2204a90bfede1d519471ac8681f3e3ffe99e9a43275e339d2`.

The job tests additionally cover Gamma LDA, restricted UKS, Becke/Stratmann,
iteration caps, exact QVF density preservation, AO-image gauge transport,
immutable exports, functional and source refusals, each aggregate budget,
dry runs and output failure status. They do not enable symmetry or alter
BIPOLE dispatch.

## Draft operator boundary check

The actual draft `FockOperator` Python module at revision `7b19b5474a05`
(SHA-256 `04469508826b8ed8130c08d89a84f268efc53ac62482e504603bf3a799a89980`)
was evaluated with the MANTA checkout's existing dense eigensolver. No build
or installation was changed in the neighboring checkout. Nine complete jobs
cover the three profiles at caps 1, 2 and converged termination: 45 separate
spin/k operators with explicit full metrics, frame identities and evaluated
generations. Mutating export inputs after construction leaves each operator
unchanged. Block actions agree exactly; maximum spectrum, generalized
residual, metric orthogonality and occupied-projector errors are respectively
1.888e-15, 2.042e-15, 2.806e-15 and 2.443e-15. This checks result export into
the current draft boundary, not the forthcoming build-state API or a generic
operator adapter inside MANTA's bounded loop.

BIPOLE/CRYSTAL parity and source-specific symmetry admission remain separate
requirements. Shared kernels, numerical SCF convergence and this operator
snapshot check do not qualify them. General accuracy profiles and the other
open capability-matrix rows remain required work.

## Complete global-hybrid public jobs

Measured 2026-10-09 in six fresh CPU workers, one per spin profile and exchange
convention. Each worker ran one cold job followed by five repetitions with
OpenMP, OpenBLAS and Accelerate restricted to one thread. Every timed call
includes joint grid/domain/SCF admission, numerical generation, terminal
confirmation and all nine output artifacts. These are finite-source timing
witnesses, without a speedup claim or matched-accuracy comparison.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_ks_job.py --hybrid-q0 strict-zero-mode
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_ks_job.py --hybrid-q0 BvK-ewald
```

The H2 RKS and HeH UKS fixtures use PBE0. The empty-beta H fixture uses
20% full-Coulomb exchange, 80% B88 exchange and full LYP correlation, as
registered by the reference fixture. This avoids claiming tight PBE0 or
B3LYP minority-potential accuracy at the libxc fully polarized endpoint.
All use the full shifted three-point mesh, 2-hartree reciprocal cutoff,
3-bohr physical pair cutoff, 4-bohr nuclear cutoff, 11-bohr XC-image radius,
12 radial points, order-17 Lebedev rule and 8-bohr Becke partition. There are
264 exchange Q vectors, 27 XC images and 2640 grid points for H2/HeH, 1320
for H. The nuclear Ewald alpha is 0.7; BvK uses its independent default alpha
and 1e-12 tolerance. The small domains/grids are consistency fixtures, not
production accuracy profiles.

| Convention | Profile | Cold (s) | Median (s) | Range (s) | Iterations | Fock calls |
|---|---|---:|---:|---:|---:|---:|
| strict-zero-mode | rks | 0.188263 | 0.101247 | 0.100175 to 0.102387 | 5 | 6 |
| strict-zero-mode | uks | 0.288519 | 0.193679 | 0.192900 to 0.194947 | 13 | 14 |
| strict-zero-mode | empty-spin | 0.190349 | 0.102438 | 0.101500 to 0.102655 | 11 | 12 |
| BvK-ewald | rks | 0.199695 | 0.096570 | 0.095842 to 0.097740 | 4 | 5 |
| BvK-ewald | uks | 0.286023 | 0.199593 | 0.199087 to 0.205439 | 14 | 15 |
| BvK-ewald | empty-spin | 0.189472 | 0.104672 | 0.103790 to 0.108150 | 12 | 13 |

Independent Gaussian/libxc contractions, nuclear lattice sums, a separate BvK
probe-charge sum and SciPy generalized eigensolves verify the evaluated
density, full physical Fock and all five electronic energy terms. Maximum
errors over the six workers are 1.821025e-10 in D, 9.769975e-15 hartree in F,
1.065814e-14 hartree in total energy, 1.707662e-14 hartree in any component,
and 5.995204e-15 electrons against the finite-grid population reference.
All QVF files validate and reproduce the evaluated spin density exactly.
The BvK grid-vs-metric population discrepancies remain 0.001607, 0.001567
and 0.014096 electrons; SCF and artifact correctness do not remove them.

| Convention | Profile | Actual-source modeled peak (bytes) | Job reservation (bytes) | Shell calls | Primitive pairs | Primitive/grid products |
|---|---|---:|---:|---:|---:|---:|
| strict-zero-mode | rks | 98539869 | 2752512 | 14160 | 552 | 855360 |
| strict-zero-mode | uks | 98713487 | 2852864 | 50454 | 2826 | 2993760 |
| strict-zero-mode | empty-spin | 99386539 | 2777088 | 14688 | 1080 | 855360 |
| BvK-ewald | rks | 98805213 | 2752512 | 11804 | 464 | 712800 |
| BvK-ewald | uks | 98978831 | 2852864 | 54054 | 3024 | 3207600 |
| BvK-ewald | empty-spin | 99651883 | 2777088 | 15910 | 1168 | 926640 |

Modeled peaks include the job reservation and a 49152-byte Python source
receipt reservation. These are the realized finite-source plans, not process
RSS measurements; count-only preflight may reserve larger candidate domains.
The native core SHA-256 was
`3a0b7e18b7a7c195bf55ac936d82f3a8486f34a349b8e3365e58d22b538ddf4a`.

Public-job tests also cover Gamma B3LYP and a custom LDA hybrid, restricted
UKS, both partitions, custom Madelung controls, capped iterations, source AO
image transport and nuclear-center QVF, all simultaneous exact/one-short
resource caps, dry runs and delivery failures. Artifacts explicitly distinguish
method-scaled exchange energies from the unscaled K child receipt. They retain
the correction separately without double counting it in total energy.

The actual FockBuild draft at base revision `7b19b5474a05` was also checked
using nine public hybrid jobs (caps 1, 2 and converged termination) and their
delivered provenance. The inspected operator and build-state module SHA-256
digests were respectively
`04469508826b8ed8130c08d89a84f268efc53ac62482e504603bf3a799a89980` and
`a54bd861d1d894f9b3c5feb3a02a379a4c6fb85debbfc1a37530b75c02d3c895`.
All 45 spin/k operators preserve physical frame, immutable evaluated D/F,
method-owned energy/source evidence and solver-modification parentage. Density
and action errors are zero; total-energy error is 4.441e-16 hartree. Maximum
spectrum, residual, metric and next-projector errors are 2.220e-15, 2.407e-15,
2.648e-15 and 2.554e-15. Capped evaluated densities differ from the next
projector by up to 0.124841, so the two states are tested separately.
This qualifies exported public-job snapshots against that draft, not a landed
adapter or Python allocation inside the bounded native solver. BIPOLE/CRYSTAL
parity, shared symmetry admission and the remaining MANTA capabilities keep
their independent gates.

## Complete range-separated public jobs

Measured 2026-10-09 after 2235 regression checks and a successful Sphinx build.
Six fresh single-thread workers each ran one cold job and five repetitions,
including joint admission, source generation, SCF confirmation and all nine
standard artifacts. The commands are:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_ks_job.py --range-separated --hybrid-q0 strict-zero-mode
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_ks_job.py --range-separated --hybrid-q0 BvK-ewald
```

The profiles are H2/HSE06 RKS, unequal-spin HeH/CAM-B3LYP UKS and empty-beta
H/LC-BLYP UKS. They use the same small grid and cutoff controls as the global
hybrid fixtures above. There are 264 base exchange vectors; each erfc component
adds three exact zero vectors, with finite analytic weights. HSE06 contains
0.25 erfc exchange at omega 0.11. CAM-B3LYP resolves to 0.65 Coulomb and -0.46
erfc exchange at omega 0.33; LC-BLYP uses Coulomb minus erfc at omega 0.33.
All ranges are in inverse bohr. Only the long-range Coulomb coefficient can
receive the BvK correction. The requested policy is preserved even when
HSE06's actual correction is `not-applicable`.

| Requested policy | Profile | Cold (s) | Median (s) | Range (s) | Iterations | Fock calls |
|---|---|---:|---:|---:|---:|---:|
| strict-zero-mode | rks | 0.209280 | 0.117588 | 0.114544 to 0.123436 | 6 | 7 |
| strict-zero-mode | uks | 0.327993 | 0.226575 | 0.222372 to 0.229686 | 14 | 15 |
| strict-zero-mode | empty-spin | 0.226942 | 0.127317 | 0.124142 to 0.129173 | 14 | 15 |
| BvK-ewald | rks | 0.215832 | 0.118796 | 0.114208 to 0.124038 | 6 | 7 |
| BvK-ewald | uks | 0.351572 | 0.267265 | 0.261031 to 0.269612 | 18 | 19 |
| BvK-ewald | empty-spin | 0.235535 | 0.133410 | 0.129320 to 0.135661 | 17 | 18 |

Independent SCF, analytic kernel contractions, separate nuclear/BvK sums and
libxc integrands verify the same finite source. Across all six workers, maximum
errors are 1.604e-10 in density, 4.341e-14 hartree in Fock, 3.953e-14 hartree in
total energy, 4.436e-14 hartree in individual energy components and 3.109e-15
electrons in finite-grid populations. QVF validation passes and spin-density
readback is exact. The grid-vs-metric population discrepancies remain 0.001607,
0.001455 and 0.014367 electrons: these fixtures do not establish continuum
accuracy or production grid settings. Timings do not claim a speedup or
matched accuracy with another exchange convention.

| Requested policy | Profile | Actual-source modeled peak (bytes) | Job reservation (bytes) | Shell calls | Primitive pairs | Primitive/grid products |
|---|---|---:|---:|---:|---:|---:|
| strict-zero-mode | rks | 98650545 | 2818048 | 16516 | 640 | 997920 |
| strict-zero-mode | uks | 115805797 | 2983936 | 56484 | 5454 | 3207600 |
| strict-zero-mode | empty-spin | 115417869 | 2908160 | 19434 | 2424 | 1069200 |
| BvK-ewald | rks | 98650545 | 2818048 | 16516 | 640 | 997920 |
| BvK-ewald | uks | 116071141 | 2983936 | 71532 | 6894 | 4062960 |
| BvK-ewald | empty-spin | 115683213 | 2908160 | 23316 | 2904 | 1283040 |

Modeled peaks include the job reservation and 49152 base bytes for Python
source provenance. The job reservation additionally charges 32768 bytes per
possible canonical component, with twice the input descriptor count as its
bound. This covers nested receipts, owned copies and serialization even for
cancelled descriptors. These are modeled numerical allowances, not RSS bounds.
The native binary SHA-256 is
`e49b857cb7567f0a3fd20c1d82f9c599d8a4f06811625cb4356d767439d51f91`.

The 65 public range-job tests additionally cover Gamma LC-omegaPBE/omegaB97X,
restricted UKS, multiple ranges, cancelled exchange descriptors, both grid
partitions, requested versus inactive Madelung controls, capped iterations,
AO-image gauge transport and exact/one-short resource caps. Structured output
preserves every signed coefficient, range, raw K receipt and method-scaled
energy. The finite erfc zero is already in each uncorrected component; the
singular correction remains separate. The global-hybrid benchmark still passes
with the extended harness.

Nine public range-job states at caps 1, 2 and convergence were exported into
the actual patched FockBuild and OperatorFrame drafts. Operator module SHA-256
was `04469508826b8ed8130c08d89a84f268efc53ac62482e504603bf3a799a89980`;
build-state module SHA-256 was
`d2f9476e8ab36e69413e80f014280ee0a3e75f1bec44f853cd650d81d0650210`.
All 45 spin/k blocks preserve delivered source provenance, physical frames,
owned evaluated D/F and method-supplied energy components. D/F actions and
energy agree exactly. Maximum spectrum, residual, metric and next-projector
errors are 2.443e-15, 2.478e-15, 2.425e-15 and 2.332e-15. Capped densities differ
from the next projector by up to 0.114365. These are export checks; no Python
adapter or copies are inserted inside the native bounded loop. The generic
commutator diagnostic does not replace method-specific convergence rules.
BIPOLE/CRYSTAL source parity, shared symmetry qualification and the remaining
capability-matrix rows retain their own acceptance gates.
