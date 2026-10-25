# Diagnosing the chi scalar-ECP comparison

The #530 validation result remains **not accepted** against the conditional
`1e-6` Ha/cell external HF/correlation target. The following diagnosis reuses
its completed RHF checkpoints and exact ORCA/CRYSTAL comparisons. It neither
reruns SCF nor applies a correction to a reported energy.

## Separate the error sources

The recorded H2S/LANL2DZ scalar-ECP Gamma results are:

| Cell / bohr | Chi minus matched CRYSTAL / Ha | Gap times cell volume |
|---|---:|---:|
| 24 x 26 x 28 | -0.003067580973 | -53.60 |
| 32 x 34 x 36 | -0.001383542965 | -54.19 |
| 40 x 42 x 44 | -0.000748407251 | -55.32 |

This approximate inverse-volume trend motivates an exchange finite-size
diagnosis; it is not an extrapolated acceptance result. The prior validation
already found unchanged energies on independently extending one-electron,
pair and ECP-projector images from 12 to 16 bohr, complex covariance within
`1.5e-14`, and native molecular ECP HF agreement with ORCA/CRYSTAL near
`1e-10` Ha. Those checks are reused rather than repeated as new jobs.

At each saved density, `chi_finite_size.py` compares the retained fitted
J/K with direct AO-pair Fourier contractions on the same reciprocal sphere.
The direct contraction shares the native pair transform but uses no auxiliary
basis, metric, whitening or fitted factor. It separately compares molecular
four-center ERIs and the leading analytical Ewald image term. It verifies
home-only physical AO-pair support, real character-independent density,
archived normalized shells and molecular/periodic overlap agreement before
making this isolated-cell comparison. It refuses more general crystals;
it is not an independent oracle for the native Fourier transform itself.

For the original 20 Ha source, fixed-density fitted minus unfitted `J+K`
is approximately +7.27 microhartree at Gamma and +7.31 microhartree on
mesh (3,1,1). Direct Fourier refinement from 20 to 80 Ha shifts `J+K` by
about +41.4 microhartree. These are separate from the millihartree exchange
finite-size difference. The prior self-consistent fitted 40 to 80 Ha change
is still 4.37 microhartree; neither axis is certified converged at `1e-6` Ha.

The completed fixed-density diagnostic gives the following energy differences
in microhartree per primitive cell (negative means lower than the molecular
operator at the same density):

| Contribution | Gamma | Mesh (3,1,1) |
|---|---:|---:|
| Fitted minus direct J+K, identical 20 Ha sphere | +7.270 | +7.308 |
| Direct J+K, 20 to 80 Ha change | +41.385 | +41.390 |
| Direct 80 Ha exchange minus molecular | -3033.815 | -897.006 |
| Leading Ewald-Hessian exchange prediction | -3017.887 | -888.478 |
| Direct 80 Ha electrostatic minus molecular | -60.882 | -61.132 |

The leading exchange estimate explains the dominant scale and mesh response
without an adjustable parameter; the exchange remainder is still about 16
and 9 microhartree, respectively. The molecular fixed-density HF energies
are -11.030579976356 and -11.030580484131 Ha; they differ slightly from the
separately converged molecular reference because these are periodic orbitals.
The diagnostic therefore identifies the principal finite-size contribution,
not a complete convergence proof or an ECP-specific Hamiltonian defect.

## Finite-size equation

The declared zero-average periodic Coulomb Green function near its origin is

```text
v_BvK(r) = 1/|r| - xi + (1/2) sum_ab H_ab r_a r_b + O(|r|^4).
```

The HF Madelung term restores `xi` in exchange. It does not remove the
quadratic and higher image terms. The diagnostic differentiates the Ewald
real, reciprocal and self terms to obtain `H`. It checks
`trace(H)=4*pi/V_BvK`; in a cubic cell, `H=4*pi/(3*V_BvK) I`.
For spin-summed closed-shell AO density `D`, define overlap `S`, dipole
integrals `X_a` and second-moment integrals `Q_ab`. The leading exchange error
relative to the molecular Coulomb operator at that density is

```text
Delta E_x = -(1/8) sum_ab H_ab [
  Tr(D S D Q_ab) + Tr(D Q_ab D S)
  - Tr(D X_a D X_b) - Tr(D X_b D X_a)].
```

This predicts inverse-volume scaling at fixed shape, while a one-axis mesh
change also changes the Hessian anisotropy. The value comes from the density
moments and lattice, without fitting the measured energy gap. Higher moments,
SCF relaxation, finite reciprocal support and fitting error remain distinct
remainders. The quadratic expression is diagnostic, not an automatic energy
or Fock correction, and is not a bound.

Sources read through the literature library: Ewald (1921), IV.1, equations
37-40, DOI [10.1002/andp.19213690304](https://doi.org/10.1002/andp.19213690304);
Carrier et al. (2007), section II, equations 3-12,
DOI [10.1103/PhysRevB.75.205126](https://doi.org/10.1103/PhysRevB.75.205126);
Sun et al. (2017), section II A, equations 3-16,
DOI [10.1063/1.4998644](https://doi.org/10.1063/1.4998644);
Xing and Lin (2024), section II B, equations 7-11, and section III,
DOI [10.1103/PhysRevX.14.011059](https://doi.org/10.1103/PhysRevX.14.011059).
The last paper distinguishes corrections to orbital energies and general
ERIs; the related canonical CC convention is documented in
[chi_ccsd.md](../docs/chi_ccsd.md).

## Reproduce without duplicating an SCF job

Use the retained RHF mesh 1 or mesh 3 HF archive from #530, with the matching
runtime and this script. The script rebuilds only small molecular integral
diagnostics from the archived shells. It does not rebuild fitted factors,
optimize orbitals or submit jobs. It writes public-safe JSON through
`vibeqc.output`, including archive/native/source identities and measured
contractions. Keep operational evidence outside the checkout.

```sh
OMP_NUM_THREADS=2 python benchmarks/chi_finite_size.py rhf.vqr \
    --cutoffs 20 40 80 > fixed-density.json
```

For a Validator execution, stage only the script and an existing archive in
an external payload directory and submit them through `vq` on an identified
runtime. Reuse a completed diagnostic result when it already exists. A
two-worker, 2 GiB allocation suffices for the recorded 12-AO case; the
diagnostic limits the molecular tensor to 32 AOs, reciprocal candidates to
2,000,000, and Fourier chunks to 1,024 vectors.

The finite-size evidence does not resolve the separate ORCA/CRYSTAL ROHF
root/formulation difference, qualify periodic CC accuracy, or admit SR or
spin-orbit ECPs. Chi remains experimental, D93 fitted/post-HF not-qualified,
and the chi/CCM construction boundary is unchanged.
