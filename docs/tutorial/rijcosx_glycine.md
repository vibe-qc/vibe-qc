# RIJCOSX SCF and analytic gradients on glycine

This tutorial exercises the v0.8.0 `JKBuilder` polymorphic
Fock build via the **RIJCOSX** kernel: RI-J for the Coulomb
piece, seminumerical chain-of-spheres exchange (COSX) for the
K piece. Molecular RIJCOSX is a **production** route (label vocabulary in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels);
the periodic RIJCOSX line is a separate experimental route and is not covered
here). Its validation, from the [feature matrix](../features.md):

- the SCF **energy** agrees with **ORCA 6.1.1 to 0.26 mHa** on glycine /
  def2-TZVP;
- the **analytic gradient** matches finite differences of the RIJCOSX energy
  surface up to the neglected grid-motion band (about 2e-4 Ha/bohr on
  H2O/def2-TZVP, about 3e-3 Ha/bohr in the worst case tested, OH/def2-SVP);
- against ORCA 6.1.1 EnGrad on glycine / def2-TZVP the gradient norm is
  0.08696 against 0.08703 Ha/bohr, with a componentwise RMS difference of
  3.3e-3 and a maximum of 1.1e-2 Ha/bohr. The residual is the documented
  fitted-Fock response term plus the neglected grid-motion and weight
  derivatives, which the [Theory](#theory) section explains.

**You will**:

- Build a 10-atom glycine molecule.
- Run RHF with the four-index (default) Fock build for
  reference.
- Re-run with RIJCOSX, comparing energy + analytic gradient.
- Time the two paths and see where the crossover lies.

## Theory

The Fock matrix needs two two-electron contributions with very different
structure. The Coulomb matrix
$J_{\mu\nu} = \sum_{\lambda\sigma} D_{\lambda\sigma}(\mu\nu|\lambda\sigma)$
depends on the density only through a single scalar field, and the exchange
matrix $K_{\mu\nu} = \sum_{\lambda\sigma} D_{\lambda\sigma}(\mu\lambda|\nu\sigma)$
couples the orbitals directly, which is why exchange is the expensive part of
a hybrid functional. RIJCOSX treats them with different approximations.

**RI-J: fitting the Coulomb field.** The Coulomb matrix is assembled by
expanding orbital-pair densities in an auxiliary basis $\{P\}$ with the
Coulomb-metric fit,

$$
(\mu\nu|\lambda\sigma) \approx \sum_{PQ} (\mu\nu|P)\,[V^{-1}]_{PQ}\,(Q|\lambda\sigma),
\qquad V_{PQ} = (P|Q) ,
$$

so that $J_{\mu\nu} = \sum_P (\mu\nu|P)\,d_P$ with fitted density coefficients
$d_P = \sum_Q [V^{-1}]_{PQ} \sum_{\lambda\sigma}(Q|\lambda\sigma)D_{\lambda\sigma}$.
The four-index tensor is never formed, and the cost is $O(N^2 M)$ for $M$
auxiliary functions. The auxiliary family is the def2-JKFIT set
(`default_aux_basis_for("def2-tzvp", kind="jk")`). The fit error is second
order in the fit residual, as in
[RI-MP2](ri_mp2_aux_verification.md#theory).

**COSX: exchange on a grid.** The chain-of-spheres algorithm
({vibeqc-cite}`neese_rijcosx_2009`) integrates one of the two electron
coordinates numerically on a Becke grid $\{\mathbf r_g, w_g\}$ and does the
other analytically, which turns the four-index contraction into a sequence of
cheap matrix products,

$$
K_{\mu\nu} \approx \sum_g w_g\,\chi_\mu(\mathbf r_g)
  \sum_{\sigma} A_{\nu\sigma}(\mathbf r_g) \sum_{\lambda} D_{\lambda\sigma}\,\chi_\lambda(\mathbf r_g),
\qquad
A_{\nu\sigma}(\mathbf r_g) = \int \frac{\chi_\nu(\mathbf r)\,\chi_\sigma(\mathbf r)}{|\mathbf r - \mathbf r_g|}\,d\mathbf r ,
$$

where $A_{\nu\sigma}(\mathbf r_g)$ is the analytic nuclear-attraction-type
integral of the product $\chi_\nu\chi_\sigma$ with a unit charge sitting at the
grid point. Because each $\chi_\mu(\mathbf r_g)$ is significant only for grid
points near its centre, the sums are sparse, which is the "chain of spheres"
and gives asymptotically quadratic cost with a small prefactor. The error is
the grid error, and the exchange grid is much sparser than the XC grid (the
`cosx_grid_level` tiers). vibe-qc uses the overlap-fitted variant by default
(`cosx_variant` AUTO resolves to FITTED): the analytic overlap is reproduced
at quadrature by a global fit, $\mathbf K = \mathbf Q\mathbf K_\text{naive}\mathbf Q^{\mathsf T}$
with $\mathbf Q = \mathbf S\mathbf S_\text{grid}^{-1}$, the global precursor of the
per-point fit of {vibeqc-cite}`izsak_neese_cosx_2011`. The unfitted STANDARD
build is cheaper but can destabilise some open-shell SCFs.

**The gradient and its residual.** The analytic gradient differentiates the
RI-J and COSX integrals with the density held fixed. Two pieces of the exact
derivative of the *approximate* energy are missing. The grid is tied to the
nuclei, so moving an atom moves grid points and changes weights; that
grid-motion response is neglected, and it is the reason the gradient agrees
with finite differences of its own energy only to the "band" quoted at the
top. The second is the response of the fitted Fock matrix. Both are small and
smooth, and they are the difference between this gradient and a finite
difference of the RIJCOSX energy or a reference program's analytic gradient.
For geometry optimisation the residual acts as a noise floor in the
forces, so a force tolerance tighter than the band quoted above is not
meaningful.

**Where it helps and where it fails.** RIJCOSX pays off for hybrid functionals
and HF on larger systems, where the exchange build dominates; the crossover
against the four-index path depends on basis and hardware, so measure it as
in the timings below. It gives nothing for pure GGA or LDA (no exchange to assemble; use
`density_fit=True, cosx=False`, i.e. RI-J alone), and on tiny systems the
grid and auxiliary set-up dominate. For range-separated hybrids the
attenuated exchange has no fitted path yet (see the
[ωB97X tutorial](range_separated_wb97x.md)).

For the implementation:
[`user_guide/density_fitting.md`](../user_guide/density_fitting.md)
documents the JKBuilder polymorphic dispatch and the three concrete
kernels (`FourIndexJKBuilder`, `DFJKBuilder`, `COSXJKBuilder`).

## Setup

Start by building the 10-atom glycine test molecule from an XYZ file
(`Molecule.from_xyz` takes a path); this is the system both Fock-build paths
below run on. The runs use def2-SVP (95 basis functions), which finishes in
minutes; the def2-TZVP figures quoted at the top are the documented validation
against ORCA, where glycine has 185 functions.

```python
from pathlib import Path
import time

import numpy as np
import vibeqc as vq

# Glycine, a 10-atom amino acid: C, N, O, H and an intramolecular contact.
Path("glycine.xyz").write_text("""\
10
glycine, B3LYP/6-31G* optimized geometry
N    -1.5012   -0.1265    0.0000
C    -0.4080    0.7959    0.0000
C     0.9189    0.0670    0.0000
O     1.0150   -1.1438    0.0000
O     1.9583    0.7993    0.0000
H    -2.3543    0.4169    0.0000
H    -1.5012   -0.7196    0.8189
H    -0.4773    1.4360    0.8843
H    -0.4773    1.4360   -0.8843
H     2.7892    0.2748    0.0000
""")
glycine = vq.Molecule.from_xyz("glycine.xyz")
```

## Reference: four-index RHF

First run RHF with the default four-index Fock build to get the reference
energy and wall-clock time the RIJCOSX run is measured against:

```python
basis = vq.BasisSet(glycine, "def2-svp")   # 95 BFs

t0 = time.perf_counter()
ref = vq.run_rhf(glycine, basis, vq.RHFOptions())   # default: four-index (direct) build
t_4idx = time.perf_counter() - t0

print(f"4-index RHF energy: {ref.energy:.6f} Ha, {ref.n_iter} iterations")
print(f"  wall-clock: {t_4idx:.1f} s")
# 4-index RHF energy: -282.610589 Ha, 14 iterations
#   wall-clock: 33 s   (heavily loaded shared machine)
```

## RIJCOSX

Now re-run the same SCF with RI-J for the Coulomb piece and COSX for
exchange, and compare the energy and timing against the four-index
reference. The option objects are configured by attribute assignment:

```python
aux_name = vq.default_aux_basis_for("def2-svp", kind="jk")
# -> "def2-svp-jk"

opts = vq.RHFOptions()
opts.density_fit = True
opts.aux_basis = aux_name
opts.cosx = True              # -> COSXJKBuilder; the exchange grid tier is
                              # chosen from the basis (cosx_grid_level = -1)

t0 = time.perf_counter()
rijcosx = vq.run_rhf(glycine, basis, opts)
t_cosx = time.perf_counter() - t0

print(f"RIJCOSX energy:    {rijcosx.energy:.6f} Ha, {rijcosx.n_iter} iterations")
print(f"  Δ vs 4-index:    {1000*(rijcosx.energy - ref.energy):+.3f} mHa")
print(f"  wall-clock:      {t_cosx:.1f} s")
# RIJCOSX energy:    -282.610285 Ha, 18 iterations
#   Δ vs 4-index:    +0.304 mHa
#   wall-clock:      297 s   (same loaded machine)
```

The 18 iterations are three SCF stages, because RIJCOSX runs a multi-stage
grid progression (a coarse exchange grid first, then finer ones, after
Helmich-Paris et al. 2021): the first stage converged in 10 iterations at
`-282.610971` Ha, the second in 7, and the final, finest stage in 1, at
`-282.610285` Ha. The energy differs from the four-index result by 0.30 mHa
here, the same order as the 0.26 mHa the def2-TZVP comparison with ORCA gives.

## Analytic gradient

The COSX path includes an analytic gradient (Bykov et al.,
*Mol. Phys.* **113**, 1961, 2015). Compare it against the four-index
analytic gradient, which is the exact Hartree-Fock gradient. The gradient
options carry their own copy of the RIJCOSX settings, so set them to match the
SCF (including `cosx_grid_level`):

```python
g_ref = np.asarray(vq.compute_gradient(glycine, basis, ref))   # (n_atoms, 3)

gopts = vq.GradientOptions()
gopts.density_fit = True
gopts.aux_basis = aux_name
gopts.cosx = True
gopts.cosx_grid_level = opts.cosx_grid_level       # -1, the same tier as the SCF
g_cosx = np.asarray(vq.compute_gradient(glycine, basis, rijcosx, gopts))

diff = np.abs(g_cosx - g_ref)
print(f"Max |Δgradient| per atom (mHa/bohr):")
for atom, d in zip(glycine.atoms, 1000 * diff.max(axis=1)):
    print(f"  Z={atom.Z:<3} {d:6.3f}")
print(f"Overall max: {1000 * diff.max():.3f} mHa/bohr, "
      f"RMS {1000 * np.sqrt((diff**2).mean()):.3f} mHa/bohr")
print(f"|g|: four-index {np.linalg.norm(g_ref):.5f}, RIJCOSX {np.linalg.norm(g_cosx):.5f} Ha/bohr")
# Overall max: 12.848 mHa/bohr, RMS 4.375 mHa/bohr
# |g|: four-index 0.13426, RIJCOSX 0.11736 Ha/bohr
```

This is a much larger difference than the energy difference suggests, and it
is the documented behaviour, not a failed run: the RIJCOSX gradient omits the
grid-motion and fitted-Fock response terms (see the [Theory](#theory)
section), which do not show up in an energy comparison at all. The def2-SVP maximum of 12.8 mHa/bohr, RMS 4.4, is
the same size as the def2-TZVP comparison against ORCA 6.1.1 (maximum
11 mHa/bohr, componentwise RMS 3.3 mHa/bohr), where the gradient norms agree to
0.08696 against 0.08703 Ha/bohr. Leaving `cosx_grid_level` at the
`GradientOptions` default of `0` (the legacy `cosx_grid`) instead of matching
the SCF gives 13.8 mHa/bohr maximum and 4.6 RMS here, so keep the two tiers
equal.

## What this tells you

* **RIJCOSX energies are accurate to a fraction of a millihartree.** 0.30 mHa
  against the four-index energy here (0.26 mHa against ORCA on the
  def2-TZVP glycine validation case) is well below the inherent basis-set
  error and most chemical accuracy targets (~1.6 mHa per kcal/mol).
* **Its forces are not at the same level.** The analytic gradient differs
  from the exact Hartree-Fock gradient by several mHa/bohr per component, so
  use RIJCOSX geometries as converged-to-the-approximation, and compare
  against a four-index or finite-difference gradient (or a reference program)
  before drawing conclusions from small force differences or tight
  optimisation thresholds.
* **The COSX K kernel is where any time is saved.** RI-J alone
  (`density_fit=True, cosx=False`) helps less because J scales O(N²·M)
  anyway; the exchange is the part that grows steeply. But on this 95-function
  case RIJCOSX was slower than the direct four-index build on the loaded
  machine used for this pass (297 s against 33 s), because the auxiliary
  basis, the multi-stage grid and the exchange grid cost more than the
  screened four-index exchange at this size. The crossover to an actual
  speedup is at larger systems, and this pass did not locate it; speedups
  reported for RIJCOSX grow with system size, and you should measure them on
  your own hardware before relying on a number.
* **Use this combination** for hybrid-DFT or HF work on larger systems after
  measuring the crossover on your own hardware; below it the four-index
  path's setup-free simplicity wins.

## When NOT to use RIJCOSX

* **Pure GGA / LDA** (no exact exchange): no exchange to
  assemble, so COSX is wasted. Use `density_fit=True,
  cosx=False` (= `DFJKBuilder`) and let RI-J accelerate the
  Coulomb piece.
* **Tiny systems** (< 20 BFs): the auxiliary basis + grid
  setup time dominates. Stay with the default 4-index path.
* **Production gradients above 50 BFs def2-TZVP** with
  `cosx=False` (the RIJK path): see the
  {ref}`v0.8.x DF gradient bug warning <historical-gradient-bug-status>`.
  The RIJCOSX path goes through the COSX gradient and is
  unaffected by this bug.

## What the `.out` reports

A RIJCOSX job run through `run_job` (here `run_job(glycine, basis="def2-svp",
method="rhf", rhf_options=opts)` with the options above) writes an `.out`
whose memory estimate names the two fitted pieces and whose SCF trace shows the
grid progression as a sequence of restarts (iteration counter back to 1 at each
new grid):

```text
  vibe-qc estimates this calculation will require ~0.41 GB of memory:
      RI-J tensors                     134.8 MB
      COSX grid workspace               41.1 MB
      Fock + density + 1e               0.55 MB
      DIIS history                       1.1 MB
      ...

  iter     energy (Ha)            dE          ||[F,DS]||   DIIS
     1     -282.4450157019        --     1.436e+00    1
     ...
    10     -282.6109711303  -3.031e-08  6.451e-05    8
     1     -282.6109669502        --     5.933e-04    1
     ...
     7     -282.6109662137  +1.355e-10  6.450e-07    7
     1     -282.6102851537        --     3.778e-02    1
  initial_guess = AUTO -> PATOM (transport: READ)
  converged in 18 iterations; E = -282.6102851537 Ha
```

The "RI-J tensors" and "COSX grid workspace" lines in the memory block are how
you confirm the route actually taken, which is a useful sanity check when
`density_fit=` or `cosx=` get tangled across multiple option structs. The
energy changes by about 0.7 mHa between the coarse-grid and final stages, which
is why the last stage, not the first converged one, is the number to report.

## Resources

Glycine / def2-SVP (95 functions, a 0.41 GB memory estimate): the four-index reference took 33 s and the RIJCOSX
run 297 s on a heavily loaded shared machine at four threads, so absolute
times are not representative, and the ratio says only that RIJCOSX is not a
win at this size. The def2-TZVP runs of this page (185 functions) were started
but did not finish in the time available on that machine, which is why the
page's measured numbers are def2-SVP.

## References

- **COSX algorithm.** F. Neese, F. Wennmohs, A. Hansen, U. Becker,
  "Efficient, approximate and parallel Hartree-Fock and hybrid DFT
  calculations. A 'chain-of-spheres' algorithm for the Hartree-Fock
  exchange," *Chem. Phys.* **356**, 98 (2009).
  [doi:10.1016/j.chemphys.2008.10.036](https://doi.org/10.1016/j.chemphys.2008.10.036).
- **RI-J / RIJK.** F. Weigend, "A fully direct RI-HF algorithm:
  Implementation, optimised auxiliary basis sets, demonstration of
  accuracy and efficiency," *Phys. Chem. Chem. Phys.* **4**, 4285
  (2002). [doi:10.1039/B204199P](https://doi.org/10.1039/B204199P).
- **RIJCOSX analytic gradient.** S. Bykov, T. Petrenko, R. Izsák,
  S. Kossmann, U. Becker, E. Valeev, F. Neese, "Efficient
  implementation of the analytic second derivatives of
  Hartree-Fock and hybrid DFT energies: a detailed analysis of
  different approximations," *Mol. Phys.* **113**, 1961 (2015).
  [doi:10.1080/00268976.2015.1025114](https://doi.org/10.1080/00268976.2015.1025114).
  The COSX-gradient implementation in vibe-qc follows this paper's
  derivation.
- **Auxiliary-basis convention.** F. Weigend, "Hartree-Fock exchange
  fitting basis sets for H to Rn," *J. Comput. Chem.* **29**, 167
  (2008). [doi:10.1002/jcc.20702](https://doi.org/10.1002/jcc.20702).
  The def2-*-jk aux family the RIJCOSX kernel routes by default.

The RIJCOSX combination of these papers fires in the bundled
[citation database](../user_guide/citations.md) routes
`methods["cosx"]`, `methods["rij"]`, and the matching aux-basis
entries, every `run_job` using `cosx=True` writes them into the
auto-generated `.bibtex`.

## See also

- [`user_guide/density_fitting.md`](../user_guide/density_fitting.md), 
  full JKBuilder reference + recommended-aux table.
- [`user_guide/functionals.md`](../user_guide/functionals.md), 
  hybrid functionals where RIJCOSX shines.
- [Pt cluster with LANL2DZ: heavy-element ECPs](pt_cluster_lanl2dz.md), 
  pairs RIJCOSX with ECPs.
- [Stiff molecular SCF: rescuing oscillation with EDIIS+DIIS](ediis_diis_hybrid.md), the
  default SCF accelerator paired with RIJCOSX in this tutorial;
  see what it does in stiff cases.
