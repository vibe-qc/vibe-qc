# Pure-KS cutoff source generation

Run `python benchmarks/manta_periodic_ks_domains.py` with the fresh
own-checkout editable core. Three fresh one-thread workers execute H2/PBE
RKS, HeH/PBE UKS and empty-beta H/BLYP UKS, with cold execution plus five
repeats. Operational JSON stays outside the product checkout.

The sources use a skew cell, shifted three-point full mesh, physical pair
cutoff 3 bohr, reciprocal kinetic cutoff 2 Ha and canonical XC image radius
11 bohr. Quadrature is an explicitly supplied 8-point-per-axis tensor
Gauss-Legendre rule (512 points), batched in 64-point panels. These are finite
source-composition fixtures, not converged grid or infinite-support references.

Independent integer enumeration checks direct and reciprocal membership,
including a more strongly skew cell. Physical shell-image shifts exercise
pair completeness and complex Bloch transport. Separate Gaussian quadrature,
AO polynomials/gradients, libxc and SciPy SCF check energies, potentials and
densities. Generated sources also match the explicit KS driver bitwise.

Modeled admission includes source boxes, sorting scratch, native owners and
exports, plus a worst-case KS plan for box counts and the supplied grid size.
The actual physical KS plan can be smaller. Domain arrays and generation work
are independent of k-mesh size; no exchange-transfer support is generated.
Peak RSS includes the interpreter, libraries and full-grid validation oracle.

## Empty-spin PBE conditioning witness

The separate empty-beta PBE fixture retains bitwise equality between generated
and explicit MANTA sources. Energy agrees with independent contraction to
roundoff and the occupied-spin potential agrees within 1e-11 Ha. In libxc
7.0.0, the minority-spin potential differed by 2.684e-6 Ha between independent
fp64 contractions. Direct raw-libxc experiments isolate the sensitivity:
a one-epsilon relative change in alpha density alone moved the contracted
minority potential by up to 3.934e-6 Ha (point potentials by 1.111e-4 Ha).
Adding a beta density of 1e-6 times alpha reduced the independent contraction
discrepancy to 1.089e-12 Ha without changing runtime regularization.

The regression retains the actually empty PBE case, checks exact domain
agreement, tight energy/occupied-spin agreement, raw-backend sensitivity and
the strict interior-density comparison. It does not certify a uniform
1e-10 Ha minority-potential accuracy at the empty-spin PBE endpoint. Empty-beta
BLYP provides the ordinary strict full-SCF fixture. Neither solver nor
quadrature convergence is inferred from minority-spin conditioning evidence.

## Measured qualification

After integrating current main and rebuilding the native core, 706 focused
checks passed, with four optional PySCF checks deselected. This includes all
26 domain tests, physical KS, earlier HF domains, semilocal XC, citations and
repository gates. Sphinx succeeded. The tests include the exact Gamma
reciprocal boundary and its adjacent floating-point cutoffs, oversized
requests, all five jointly exact caps and external live storage.

The following measurements use fresh processes after regressions completed.
The complete wall times include count planning, source generation and SCF;
source-only generation is separately observed once per worker, after repeats.

| Profile | Cold (s) | Median (s) | Range (s) | CPU median (s) | Generation (s) | Domain live (bytes) | Actual modeled peak (bytes) | Box-based peak bound (bytes) | RSS (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rks | 0.005513 | 0.005278 | 0.005189 to 0.005598 | 0.005272 | 0.000025 | 76000 | 79417411 | 79465955 | 162.7 |
| uks | 0.028609 | 0.027924 | 0.027105 to 0.029211 | 0.027853 | 0.000041 | 76000 | 79476793 | 79556537 | 163.6 |
| empty-spin | 0.016682 | 0.017580 | 0.016852 to 0.017767 | 0.017530 | 0.000039 | 76000 | 79374685 | 79423229 | 162.2 |

Each case selected 28 nonzero G vectors, one physical translation and
27 canonical XC images. RKS used three iterations/four physical builds;
UKS and empty-beta BLYP used ten iterations/eleven builds. Maximum independent
errors were 7.090e-11 in converged density, 3.553e-15 Ha in same-density Fock,
6.218e-15 Ha in energy and 1.333e-15 electrons in grid populations.
Independent oracle times were 0.029 to 0.084 s. These small absolute timing
observations establish no speedup or scaling claim.

Native binary SHA256: `57b4e9b52b06bc0ec18f29ee9d6d1db071e3192b77f82ba184504b1f406c22d9`.
