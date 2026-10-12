# MANTA complete periodic Coulomb fitting source

This checkpoint measures complete finite auxiliary metrics and shared-q
three-center blocks assembled from MANTA erfc and Fourier kernels. The
existing GDF sphere/capsule selectors and jellium kernel define the source.
It is unfactored: no retained-space projection, exchange correction or SCF is
included. Finite cutoffs and explicit reciprocal lists still need convergence
for a physical calculation.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_gdf.py
```

Measured on 2026-10-10 with fresh own-checkout native SHA256
`8817dab3a8a1061c7fae2e65c0d02bab9b0c5b70c6490f540ae3d1087bc1306d`.
The script checks build provenance and freshness. Each profile runs in a
fresh single-thread process: one cold MANTA call, five MANTA repeats and five
calls to the existing combined GDF source. Both timings include owned NumPy
metric/coupling exports. The libint range includes its first call. Numbers
are specific to this build and these small fixtures.

The s fixtures have two orbital and two auxiliary functions in a skew cell,
omega=0.65 and pair/auxiliary radii of 5 bohr. Gamma uses q=k=0; the second
profile uses a non-TRIM q and two ket momenta. The d fixture has ten auxiliary
functions from signed five-primitive contractions, four orbital functions,
and radii 4.3/5.1 bohr. All three use the 27 `G+q` vectors from integer triples
in [-1,1], with no SR envelope screening. They are matched-source consistency
fixtures, not accuracy profiles or a full-method performance workload.

## Accuracy and timings

Errors are maximum absolute element errors against the existing combined
GDF source. Independent s references use analytic Gaussian/Boys real-space
sums, Gaussian Fourier integrals and an explicit finite-zero outer product.
A general complex pair trial compares `T.conj().T @ solve(M,T@trial)` on the
same finite source. That witness does not select a production fitting rule.

| Profile | MANTA cold / median (ms) | MANTA range (ms) | Existing GDF median / range (ms) | Metric / coupling error | Independent metric / coupling error | Complex action error |
|---|---:|---:|---:|---:|---:|---:|
| s, Gamma | 0.764 / 0.701 | 0.696-0.706 | 0.159 / 0.157-1.194 | 1.60e-14 / 1.07e-14 | 4.76e-14 / 5.11e-15 | 8.15e-15 |
| s, two k | 0.784 / 0.701 | 0.696-0.746 | 0.181 / 0.181-1.262 | 2.28e-13 / 5.69e-14 | 1.14e-13 / 4.33e-14 | 9.96e-14 |
| d, two k | 0.510 / 0.420 | 0.419-0.432 | 0.200 / 0.197-1.461 | 8.66e-14 / 2.49e-14 | Separate quadrature witnesses | 2.67e-15 |

MANTA is slower on all three microbenchmarks. The full-method performance
gate remains open. The minimum metric eigenvalues here are 4.6306, 5.1284 and
21.4186, respectively. Positivity on these fixtures is not a guarantee for
arbitrary finite cutoffs or a certificate for a retained fitting subspace.

## Resource receipts

| Profile | Output bytes | Modeled peak bytes | SR metric / coupling calls | Auxiliary / pair FT calls | Total primitive products | Vector-primitive products | Phase products / accumulations | Final contraction terms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s, Gamma | 192 | 25552264 | 32 / 816 | 6 / 96 | 950 | 952 | 1008 / 1808 | 336 |
| s, two k | 320 | 25553480 | 32 / 816 | 4 / 64 | 916 | 918 | 1104 / 2704 | 540 |
| d, two k | 6720 | 25659348 | 4 / 8 | 4 / 8 | 312 | 702 | 44 / 1444 | 11340 |

| Profile | Admitted shell-call bound | Primitive-product bound | Scalar-work bound | Kernel-work bound |
|---|---:|---:|---:|---:|
| s, Gamma | 4263198 | 4263198 | 2192502624 | 1913012095824 |
| s, two k | 4263198 | 4263198 | 2205595040 | 1913128779600 |
| d, two k | 946686 | 18874634 | 2832966176 | 12320196476704 |

Planning bounds all enclosing image candidates and reserves a zero panel for
every q. The Gamma source executes that extra panel to remove the finite SR
zero exactly once. Reciprocal panels contain at most 16 vectors; one
shell-pair Fourier tile is reused across all ket momenta. No full AO-square
by reciprocal-vector transient is allocated. Conservative memory includes
inputs, result owners/exports, compensation and MD workspace; it is not RSS.
Caller-retained storage is separately charged through `other_live_bytes`.

## Scientific gates

The 38 focused tests compare converged omega=0.45 and 0.8 results against a
separate bare-Coulomb reciprocal sum, at Gamma and non-TRIM q. With pair radius
12 and auxiliary radius 18 bohr, both splittings meet the 3e-11 absolute plus
2e-12 relative matrix/tensor tolerance. Omitting zero from the reciprocal list
preserves the finite-zero subtraction, including reciprocal Gamma aliases.

Other tests cover complex actions, image gauges, transfer conjugacy,
left-handed cells, ordered-domain identities, pure/Cartesian signed
contractions and high-angular Fourier assembly through orbital l=6 and
auxiliary l=7 against separable polynomial quadrature. Engaged screening
changes only the retained SR source. Exact/one-short tests cover all five
joint resource caps before numerical copies, with and without metric output.
The combined change-relevant native regression passed 212 tests; Sphinx
documentation also built successfully.

Selected fitted SCF, retained-space projection, physical erf/erfc fitted
kernels, derivatives and symmetry certificates remain subsequent increments.
This source does not alter BIPOLE AC/BD domains or assert CRYSTAL parity.
