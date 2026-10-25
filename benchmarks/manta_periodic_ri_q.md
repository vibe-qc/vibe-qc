# MANTA streamed periodic RI transfer components

This checkpoint evaluates a fixed positive auxiliary-metric projector over
the finite Coulomb GDF source, with streamed retained-mode J/K contractions.
Gamma J uses the complete supplied ket batch; each K matrix is a source-k
weighted contribution at bra k-q. No SCF, exchange singular correction or
physical range-separated fitting is included.

## Reproduction and scope

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_ri_q.py
```

Measured on 2026-10-10 with fresh own-checkout native SHA256
`14961451ed9c23fb2e5bf937f186876e064d41918e177d2f3d29f90bb270e891`.
The driver checks provenance and freshness. Each profile uses its own
single-thread process, one source/projector construction, one cold build
and five repeated builds, including owned J/K exports. The dense comparison
constructs the same finite source with libint, factors it with NumPy and
retains all fitted factors for five repeated contractions. Its construction
timing includes the first libint call. These are different storage/work
strategies; the dense repeated-build timing performs no integral evaluation.

Both s profiles use two orbital and two auxiliary functions, two ket
momenta, numerical omega 0.65 and pair/auxiliary radii of 5 bohr. Gamma builds
J/K; the non-TRIM transfer builds K. The d profile uses four orbital and ten
auxiliary functions with signed five-primitive contractions and radii
4.3/5.1 bohr. Every profile uses the 27 G+q vectors from integer triples in
[-1,1], complex indefinite alpha/beta trial densities and weights 0.23/0.77.
There is no SR screening. Mode blocks are one for s and four for d. These
are finite-source consistency fixtures, not converged accuracy profiles.

## Results

| Profile | Construction / cold build (ms) | Streamed median / range (ms) | Dense construction (ms) | Cached dense median / range (ms) | Maximum K error | Maximum J error |
|---|---:|---:|---:|---:|---:|---:|
| s, Gamma | 0.110 / 1.571 | 1.577 / 1.537-1.611 | 1.512 | 0.038 / 0.031-0.107 | 2.04e-14 | 3.22e-15 |
| s, non-TRIM | 0.126 / 1.737 | 1.628 / 1.551-1.649 | 1.455 | 0.033 / 0.026-0.096 | 5.69e-14 | Not requested |
| d, non-TRIM | 0.341 / 1.408 | 1.336 / 1.336-1.338 | 1.946 | 0.049 / 0.042-0.119 | 2.66e-15 | Not requested |

Independent analytic Gaussian/Boys image sums and Gaussian Fourier integrals
give maximum s J/K differences of 3.06e-14 and a Gamma Coulomb energy
difference of 3.61e-16 hartree. The Gamma metric-equation residual is 8.85e-16
for right-hand-side norm 1.89735. Response Hermiticity residuals are below
4e-16. These residuals describe the finite source, not continuum error.

Streaming is slower than cached dense contractions here. Repeated passes
trade integral work for bounded factor storage. No crossover, whole-method
speedup or production accuracy claim follows from these small profiles.

## Admission and executed work

| Profile | Rank / passes | Largest raw coupling tile (complex values) | Tile-source peak bytes | Joint RI peak bytes | Metric shell / primitive calls | Build shell / primitive calls |
|---|---:|---:|---:|---:|---:|---:|
| s, Gamma | 2 / 2 | 2 | 17035126 | 25430902 | 38 / 38 | 2064 / 2064 |
| s, non-TRIM | 2 / 2 | 2 | 17035126 | 25430902 | 36 / 36 | 1920 / 1920 |
| d, non-TRIM | 10 / 3 | 90 | 17079774 | 25515998 | 8 / 120 | 120 / 912 |

| Profile | Joint shell bound | Primitive-product bound | Scalar-work bound | Kernel-work bound |
|---|---:|---:|---:|---:|
| s, both | 8541018 | 8541018 | 4459718720 | 3799192055568 |
| d | 2849722 | 56604994 | 8625890400 | 36855637710144 |

The counts include repeated auxiliary and AO-pair Fourier evaluations caused
by shell/mode tiling. Bounds enclose all real-space image candidates and
reserve a zero panel even for nonzero q. Construction plus one build is
admitted at maximum possible rank before metric evaluation. The model
includes snapshots, dense metric/eigensolver/factor/projector, density and
output copies, compensation, one shell tile, bounded fitted-mode blocks and
backend margins. It is a conservative inventory, not measured RSS, and work
caps are not lifetime call quotas.

A separate count-only test uses 80 orbital functions, 256 auxiliary functions
and four momenta. Its joint peak bound is 62054872 bytes, below the
104857600 bytes required for one complete complex coupling tensor alone.
It constructs only the immutable shell source and computes the RI admission
plan; it does not execute this larger metric or J/K workload.

## Scientific and interface checks

The tests reconstruct the dense metric/coupling bitwise from shell tiles and
compare independent Gaussian integrals and libint sources. Pure and Cartesian
signed angular contractions are checked through f auxiliaries. The legacy
auxiliary Fourier route refuses Cartesian L>0; those cases use independent
polynomial quadrature for LR plus libint SR. Converged bare-Coulomb reciprocal
fits agree for two numerical splits at Gamma and non-TRIM q.

Weighted directional derivatives and conjugate-transfer energy closure check
J/K contractions. AO-image changes transport the density from ket k and K to
bra k-q, including a nontrivial truncated auxiliary projector. Duplicate,
truncated and zero-rank spaces, negative-metric and non-Hermitian-response
refusals, all five exact/one-short caps and owned immutable captures have
separate tests. No positivity requirement is imposed on trial spin densities.

A read-only compatibility probe against the current shared FockBuild and
OperatorFrame draft consumed nine finite kinetic-plus-fitted-interaction
snapshots, totaling 36 spin/k operators. Full, truncated and zero-rank fits
preserved evaluated densities, method-owned energy components, weighted
commutators, operator actions and owned exports. This is an interface probe,
not SCF acceptance; generic commutators do not replace method convergence or
terminal-density closure.

Whole-mesh fitted Fock/SCF selection, charge/normalization and exchange-seam
acceptance, physical range-separated fitting, BIPOLE/CRYSTAL parity,
symmetry reduction and projector response remain required. No full T/B
materialization, silent metric repair or automatic symmetry admission is
introduced by this increment.
