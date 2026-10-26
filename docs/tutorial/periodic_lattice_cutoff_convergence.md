---
myst:
  html_meta:
    "description": "Tutorial: converging a periodic Kohn-Sham energy with respect to the lattice-sum cutoff on the GDF route. The cubic Al2O3 model with STO-3G and PBE at the Gamma point: cutoff_bohr, rcut_strategy='pyscf_auto' against 'flat', the exchange-correlation image domain and the xc_image_cutoff_bound helper, with the Gaussian decay estimates behind them."
    "og:title": "vibe-qc tutorial: lattice-cutoff convergence of periodic Kohn-Sham energies"
---

# Lattice-cutoff convergence of periodic Kohn-Sham energies

A periodic Gaussian calculation sums over lattice images: of the
one-electron integrals, of the fitted Coulomb terms and, in Kohn-Sham DFT,
of the basis-function products that make up the density on the
integration grid. Each sum is truncated at a radius, and a radius that is
too small gives an energy that looks converged but is not. This tutorial
converges the PBE energy of a compact oxide against that radius, shows what
the default strategy does for you, and gives the estimate you can use to
check any cell before you run it.

```{admonition} Status
:class: note

Periodic Kohn-Sham DFT on the Gaussian density-fitting (GDF) route is
production for 3-D cells. The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

**Prerequisites.** The [periodic DFT tutorial](periodic_dft.md) and the
[multi-k SCF guide](../user_guide/multi_k_scf.md), which lists every GDF
option. Run the script from an empty scratch directory. The numbers were
measured with the vibe-qc 0.17.15 development build (main, 2026-10-11) on a
heavily loaded laptop, so the wall times show ratios, not absolute cost.
Since v0.18.0 the periodic Becke partition switches atom images off
smoothly (#482), which moves periodic Kohn-Sham energies at the
quadrature-error level, about 1e-7 Ha on the default grid. Expect the last
digits of the energies below to differ on a current build; the
differences between rows, which are what the tutorial is about, are far
larger.

## The test cell

The cubic Al2O3 model of the examples directory is a fluorite-derived cell,
$a = 5.05$ Å, with two aluminium and three oxygen atoms. It is not
corundum, but it is compact and ionic, which is the regime where truncated
lattice sums fail first. STO-3G has 33 functions in this cell. A small
Fermi-Dirac smearing keeps the Gamma-point PBE occupation well defined.

```python
import time
import numpy as np
import vibeqc as vq
from vibeqc.periodic_k_gdf import run_krks_periodic_gdf
from vibeqc.periodic_xc_domain import xc_image_cutoff_bound

ANG = 1.0 / 0.529177210903
A = 5.05 * ANG
atoms = ([vq.Atom(13, [f * A for f in fr]) for fr in [(.25, .25, .25), (.75, .75, .75)]]
         + [vq.Atom(8, [f * A for f in fr]) for fr in [(.5, .5, 0), (.5, 0, .5), (0, .5, .5)]])
cell = vq.PeriodicSystem(3, np.diag([A, A, A]), atoms)
basis = vq.BasisSet(cell.unit_cell_molecule(), "sto-3g")

print(f"XC image bound: {xc_image_cutoff_bound(cell, basis):.3f} bohr")

def run(strategy, cutoff):
    opts = vq.PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = cutoff
    opts.smearing_temperature = 0.01          # Hartree
    opts.conv_tol_energy = 1e-7
    t0 = time.perf_counter()
    r = run_krks_periodic_gdf(cell, basis, (1, 1, 1), opts, functional="pbe",
                              aux_basis="def2-svp-jk", rcut_strategy=strategy,
                              progress=False)
    return r, time.perf_counter() - t0

for strategy, cutoff in [("pyscf_auto", 15.0), ("flat", 15.0), ("flat", 20.0),
                         ("flat", 25.0), ("flat", 30.0)]:
    r, dt = run(strategy, cutoff)
    print(f"{strategy:10s} {cutoff:5.1f}  E = {r.energy:.8f}  "
          f"E_xc = {r.e_xc:.8f}  it = {r.n_iter}  {dt:.0f} s")
```

## The measurement

```text
XC image bound: 23.157 bohr
```

| `rcut_strategy` | `cutoff_bohr` | E, Ha | E_xc, Ha | E minus converged, mHa | E_xc minus converged, mHa | Iterations | Wall, s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pyscf_auto` | 15 | -700.84930373 | -62.72525253 | 0.0000 | 0.0000 | 16 | 745 |
| `flat` | 15 | -700.85461920 | -62.73382313 | -5.3155 | -8.5706 | 15 | 288 |
| `flat` | 20 | -700.84930389 | -62.72525201 | -0.0002 | +0.0005 | 16 | 455 |
| `flat` | 25 | -700.84930373 | -62.72525255 | 0.0000 | 0.0000 | 16 | 903 |
| `flat` | 30 | -700.84930373 | -62.72525255 | reference | reference | 16 | 1126 |

Read the table in two directions.

- **Down the `flat` rows.** With the caller's cutoff applied to everything,
  15 bohr is 5.3 mHa too low and almost all of it is exchange-correlation
  energy. At 20 bohr the error falls to 0.2 µHa, and from 25 bohr on the
  energy is converged to the printed digits. The cost grows with the
  number of images, roughly as the cube of the radius: 30 bohr costs four
  times as much as 15.
- **The first row.** The default strategy, `rcut_strategy="pyscf_auto"`,
  starts from the same 15 bohr and still gives the converged energy. It
  grows each domain to a bound computed from the basis: the one-electron
  terms to the separation where their overlap elements fall below 1e-16 of
  the largest, and the exchange-correlation image domain to the bound
  printed first, 23.2 bohr here. It costs more than a flat 15 bohr and less
  than a flat 25.

So the default takes the one-electron and exchange-correlation domains off
your hands, but not everything. The fitted Coulomb and exchange lattice sums
of the default range-separated fit still use `cutoff_bohr` as given, and
`rcut_strategy` does not size them. On this cell at Gamma 15 bohr already
converges the fit to the printed digits; on rock-salt LiH on a (2, 2, 2)
mesh, 15 bohr sits 1.6 mHa from PySCF and 30 bohr reaches microhartree
agreement ([multi-k SCF guide](../user_guide/multi_k_scf.md#lattice-cutoff-convergence)).
So keep one flat scan of `cutoff_bohr` in your convergence checks, after
the k mesh. Use `rcut_strategy="flat"` when you need the historical
behaviour everywhere, for example to reproduce an old number.

## What the exchange-correlation domain is

The density on a grid point $\mathbf r$ is a lattice sum of basis-function
products,

$$
\rho(\mathbf r) = \sum_{\mathbf R, \mathbf S}\sum_{\mu\nu}
D_{\mu\nu}(\mathbf S - \mathbf R)\,
\chi_\mu(\mathbf r - \mathbf R)\,\chi_\nu(\mathbf r - \mathbf S),
$$

and a term contributes only where both functions are non-negligible at the
same point. The bound returned by `xc_image_cutoff_bound` is

$$
R_{\mathrm{xc}} = 2\,r_{\mathrm{ao}} + d_{\mathrm{cell}},
$$

where $r_{\mathrm{ao}}$ is the largest radius at which any shell's absolute
Gaussian envelope $\sum_p |c_p|\, r^{l} e^{-\alpha_p r^2}$ still exceeds
$10^{-4}$, and $d_{\mathrm{cell}}$ is the largest distance between two shell
centres in the home cell. Two functions further apart than $2r_{\mathrm{ao}}$
cannot both exceed the threshold anywhere. The most diffuse shell sets
$r_{\mathrm{ao}}$, so the bound grows with diffuse functions: about 18.5 bohr
for diamond in def2-SVP, 23.2 bohr here, and above 30 bohr for rock-salt
LiH, whose lithium 2sp shell in STO-3G is very diffuse. Up to the v0.17.15
release, the Kohn-Sham GDF routes built the exchange-correlation
density on the caller's 15 bohr, which left this Al2O3 cell 1.55 mHa away
from PySCF.

## Theory

**Gaussian decay sets every cutoff.** The product of two Gaussians with
exponents $\alpha$ and $\beta$ centred a distance $R$ apart is a Gaussian
of exponent $\alpha + \beta$ with prefactor

$$
\exp\!\left(-\frac{\alpha\beta}{\alpha+\beta}\,R^2\right),
$$

so overlap, kinetic and density terms between an AO and its image decay as
a Gaussian in the separation, controlled by the most diffuse exponents
present. A cutoff that keeps every pair whose prefactor exceeds a threshold
$\epsilon$ needs a radius that grows as $\sqrt{\ln(1/\epsilon)/\alpha_{\min}}$.
That is the criterion of the lattice truncation schemes of CRYSTAL
({vibeqc-cite}`pisani_dovesi_lattice_truncation_1980`,
{vibeqc-cite}`causa_exchange_truncation_1988`) and of the per-shell radii
PySCF uses for Gaussian density fitting ({vibeqc-cite}`sun_berkelbach_gdf_2017`),
which the strategy's name refers to.

**The sign of the error.** A truncated density is not the density of any
determinant: it misses the products that cross the cutoff, and those can
carry either sign. Its exchange-correlation energy can therefore come out
too high or too low; here it is too low. Do not rely on the sign: converge
the number.

**Cost.** The number of image cells inside a sphere of radius $R$ grows as
$4\pi R^3 / (3\,\Omega)$ for a cell of volume $\Omega$. The
exchange-correlation build screens each grid batch to the shells that reach
it, so the wall time grows more slowly than the image count, but it still
grows, as the table shows.

## Variations

- **A diffuse basis.** Repeat the scan with def2-SVP: the bound grows and
  the flat 15 bohr error grows with it.
- **Rock-salt LiH.** Run the same comparison on the LiH primitive cell of
  the [LiH multi-k tutorial](lih_multi_k.md); its bound is above 30 bohr.
- **The bound alone.** `xc_image_cutoff_bound(cell, basis)` costs nothing.
  Print it for every new cell and basis before you plan a calculation.

## Pitfalls

- **A flat scan that stops too early looks converged.** Between 25 and 30
  bohr nothing moves here, but between 15 and 20 the energy moved by 5 mHa.
  Always include a point well beyond the bound.
- **The cutoff is not the only knob.** The k mesh, the auxiliary basis, the
  fit's kinetic-energy cutoff and the integration grid each need their own
  convergence check.
- **Compare like with like.** Two runs with different strategies differ in
  their domains even at the same `cutoff_bohr`; record the strategy with
  the number.

## See also

- [Multi-k SCF](../user_guide/multi_k_scf.md), the GDF options and guards.
- [Periodic methods](../user_guide/periodic_methods.md), the GDF route and
  its fit sources.
- [Periodic SCF convergence](periodic_scf_convergence.md) for the SCF side
  of a difficult cell.
- [k-points and Bloch functions](kpoints_brillouin_bloch.md) for the mesh.
