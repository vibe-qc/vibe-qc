# MANTA periodic short-range fitting

This checkpoint measures the jointly admitted raw erfc auxiliary metric and
shared-q three-center source. MANTA and the existing libint evaluator use the
same GDF sphere/capsule domains and normalized shell payloads. No metric
projection, long-range sum, finite-zero subtraction, exchange correction or
SCF is included. These small finite domains are consistency fixtures, not
production accuracy profiles.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_fitting.py
```

Measured on 2026-10-10, using a fresh own-checkout native module with SHA256
`e417ed5321b96e530b7881b2642c86e73b564298e086886de45fddd07c2f843d`.
The script verifies the module location and freshness. Each profile runs in
a fresh single-thread process, records one cold MANTA call and five repeated
MANTA calls, then five calls to the existing metric/three-center evaluator.
The libint range includes its first call. Reported times cover each API's
joint result creation; the legacy helper also exports its NumPy arrays.
Timing is specific to this build and these fixtures.

The two s profiles use two orbital and two auxiliary functions, omega=0.65,
and pair/auxiliary radii of 5 bohr in a skew cell. The Gamma profile uses
q=k=0. The multi profile uses a general non-TRIM transfer and two ket momenta.
The d profile uses the signed, five-primitive auxiliary d contractions of
`_weighted_gradient_fixture`, p/s orbital shells, pair radius 4.3 bohr,
auxiliary radius 5.1 bohr and the same omega. All profiles disable integral
envelope screening. Exact fixture definitions are in the benchmark script
and its test helpers.

## Accuracy and time

Errors below are maximum absolute element errors. The independent s-shell
references sum analytic Gaussian/Boys integrals over the physical image
domains. A general complex AO-pair trial checks the fitted action
`T.conj().T @ solve(M, T @ trial)` against the same finite libint source.
This trial is a source-consistency witness, not a qualified complete fit.

| Profile | MANTA cold / median (ms) | MANTA range (ms) | libint median / range (ms) | Metric / coupling error | Independent metric / coupling error | Complex action error |
|---|---:|---:|---:|---:|---:|---:|
| s, Gamma | 0.698 / 0.657 | 0.645-0.665 | 0.129 / 0.127-1.283 | 1.43e-14 / 8.89e-15 | 4.98e-14 / 5.78e-15 | 8.12e-15 |
| s, two k | 0.860 / 0.680 | 0.636-0.820 | 0.135 / 0.133-1.272 | 2.14e-14 / 1.07e-14 | 5.00e-14 / 5.02e-15 | 1.02e-14 |
| d, two k | 0.379 / 0.324 | 0.323-0.328 | 0.037 / 0.034-1.311 | 7.82e-14 / 2.14e-14 | Separate high-l witnesses below | 1.30e-14 |

MANTA is slower than the existing evaluator on all three microbenchmarks.
These results establish numerical substitution and resource accounting; they
do not satisfy the full-method performance gate.

## Resource receipts

| Profile | Orbital / auxiliary / k counts | Output bytes | Modeled peak bytes | Actual metric / three-center calls | Pair-shell images | Actual primitive products | Phase products / accumulations |
|---|---:|---:|---:|---:|---:|---:|---:|
| s, Gamma | 2 / 2 / 1 | 192 | 17017742 | 32 / 816 | 64 | 848 | 912 / 912 |
| s, two k | 2 / 2 / 2 | 320 | 17018318 | 32 / 816 | 64 | 848 | 976 / 976 |
| d, two k | 4 / 10 / 2 | 6720 | 17071006 | 4 / 8 | 8 | 260 | 28 / 580 |

| Profile | Admitted shell-call bound | Primitive-product bound | Scalar-work bound | Kernel-work bound |
|---|---:|---:|---:|---:|
| s, Gamma | 4254444 | 4254444 | 1635199584 | 1895460344472 |
| s, two k | 4254444 | 4254444 | 1636692640 | 1895472288920 |
| d, two k | 942564 | 18858140 | 2669893408 | 12296644801896 |

Count-only admission deliberately bounds every candidate in the enclosing
integer image boxes before geometric rejection. These conservative bounds
are much larger than executed work. The extra ket momentum changes phase
contractions and output storage while reusing primitive evaluations. Modeled
memory includes native owners, compensation, binding/output copies and MD
workspace; it is not measured RSS. Unmodeled caller-retained owners must be
charged through `other_live_bytes`.

## Validation scope

The 48 focused tests additionally cover unequal image radii, signed long
contractions, pure/Cartesian shells, atom-image gauge covariance, reciprocal
relabeling, a left-handed cell representation, independent source identities,
owned exports and all five exact/one-short resource limits. Independent
central-potential radial quadrature checks orbital angular momenta through
l=6; an analytic Gaussian Fourier radial integral and spherical orthogonality
check the auxiliary l=7 limit, beyond the linked libint fitting limit.

Screening tests engage the existing absolute shell-envelope rule. With pair
radius 8 and auxiliary radius 20 bohr, the unscreened source evaluates 296
metric and 3240 three-center shell images. Error allowances of 1e-10, 1e-5 and
1e-2 reduce the three-center counts to 2436, 1375 and 866, respectively; the
measured raw-entry differences remain within their allowances. This does not
bound finite-image truncation, fitting error or floating-point arithmetic.

The broader change-relevant MANTA/citation/test-gate regression completed
2086 tests; 40 selected existing GDF enumeration, screening and workspace
tests also passed. Full SR/LR assembly, retained-space projection, selected
fitted SCF, derivatives, symmetry certification and BIPOLE/CRYSTAL parity
remain separate acceptance stages.
