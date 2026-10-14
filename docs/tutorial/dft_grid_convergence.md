---
myst:
  html_meta:
    "description": "Tutorial: converging a Kohn-Sham energy with respect to the integration grid in vibe-qc. Water with PBE and r2SCAN: the named grid levels, an angular series of Lebedev rules up to order 95 (3074 points), a radial series of Treutler-Ahlrichs shells, and the Becke/Stratmann partition and quadrature theory behind them."
    "og:title": "vibe-qc tutorial: DFT integration-grid convergence"
---

# DFT integration-grid convergence

The exchange-correlation energy of Kohn-Sham DFT has no analytic Gaussian
integrals. vibe-qc evaluates it on a molecular grid: radial shells around
each atom, an angular quadrature on each shell, and a smooth partition that
splits space between the atoms. This tutorial measures how the energy of
water depends on each of these pieces, for a GGA (PBE) and a meta-GGA
(r2SCAN), using the named grid levels and explicit Lebedev rules up to the
3074-point order-95 rule. You will see which knob matters for which
functional and how much accuracy each step buys.

```{admonition} Status
:class: note

Molecular Kohn-Sham DFT and its grids are production. The label vocabulary
is defined in [Methods: theory and status](../methods.md#how-to-read-the-status-labels).
```

**Prerequisites.** The [molecular DFT tutorial](molecular_dft.md) and the
grid section of the [functionals guide](../user_guide/functionals.md). Run
the script from an empty scratch directory. The numbers were measured with
the vibe-qc 0.17.17 development build (main, 2026-10-14); the wall times are
from a laptop and include the whole job.

## Named grid levels

`run_job` takes a `grid_level` keyword. The default is `"orca-defgrid3"`:

| `grid_level` | Radial shells | Angular rule | Pruning | Partition |
|---|---:|---|---|---|
| `"coarse"` | 50 | Lebedev order 23 (194 points) | NWChem | Stratmann |
| `"orca-defgrid3"` (default) | 75 | Lebedev order 29 (302 points) | NWChem | Stratmann |
| `"fine"` | 99 | Lebedev order 35 (434 points) | NWChem | Stratmann |
| `"legacy"` | 75 | product Gauss-Legendre 17 by 36 | none | Becke |
| `"skala"` | PySCF level 3 parity profile for SKALA-1.1 | | | |

```python
import numpy as np
import vibeqc as vq

th = np.deg2rad(104.5 / 2)
r = 0.9572 / 0.529177210903          # experimental O-H, bohr
water = vq.Molecule([vq.Atom(8, [0.0, 0.0, 0.0]),
                     vq.Atom(1, [r * np.sin(th), r * np.cos(th), 0.0]),
                     vq.Atom(1, [-r * np.sin(th), r * np.cos(th), 0.0])])

for functional in ("pbe", "r2scan"):
    for level in ("coarse", "orca-defgrid3", "fine"):
        res = vq.run_job(water, basis="def2-svp", method="rks",
                         functional=functional, grid_level=level,
                         output=f"{functional}_{level}")
        print(functional, level, f"{res.energy:.10f}")
```

Errors relative to the largest grid of the series below (order 59, 200
radial shells):

| `grid_level` | PBE error, µHa | r2SCAN error, µHa |
|---|---:|---:|
| `"coarse"` | +6.3 | +119.4 |
| `"orca-defgrid3"` | +1.8 | +8.3 |
| `"fine"` | +0.3 | +1.3 |

The default is accurate to a few microhartree for PBE and to under ten for
r2SCAN on this molecule, and `"fine"` is within about a microhartree for
both. `"coarse"` is fine for a GGA pre-optimisation but costs r2SCAN a
tenth of a millihartree. Meta-GGAs depend on the kinetic-energy density,
which varies rapidly near the nuclei and in bonds, and are known to need
denser grids than GGAs.

## Explicit grids: the angular series

For a convergence study, set the grid explicitly through the options
object. Here the radial grid is held at 99 shells, pruning is off, so every
shell carries the full rule, and the Lebedev order runs from 23 to 95:

```python
def energy(functional, n_radial, order):
    opts = vq.RKSOptions()
    opts.grid.n_radial = n_radial
    opts.grid.angular = "lebedev"
    opts.grid.lebedev_order = order
    opts.grid.angular_pruning = "none"
    opts.grid.partition = "stratmann"
    opts.conv_tol_energy = 1e-10
    res = vq.run_job(water, basis="def2-svp", method="rks", functional=functional,
                     rks_options=opts, output=f"{functional}_L{order}_R{n_radial}")
    return res.energy

for order in (23, 29, 35, 41, 47, 53, 59, 65, 77, 95):
    print(order, f"{energy('pbe', 99, order):.10f}", f"{energy('r2scan', 99, order):.10f}")
```

| Lebedev order | Points per shell | PBE error, µHa | r2SCAN error, µHa | PBE wall, s |
|---:|---:|---:|---:|---:|
| 23 | 194 | +16.49 | +21.69 | 0.6 |
| 29 | 302 | +1.69 | +3.06 | 0.8 |
| 35 | 434 | +0.25 | +1.26 | 1.0 |
| 41 | 590 | -0.03 | +1.02 | 1.3 |
| 47 | 770 | +0.10 | +1.10 | 1.6 |
| 53 | 974 | +0.01 | +1.05 | 1.9 |
| 59 | 1202 | +0.03 | +1.07 | 3.3 |
| 65 | 1454 | +0.03 | +1.06 | 2.9 |
| 77 | 2030 | +0.06 | +1.08 | 4.0 |
| 95 | 3074 | +0.04 | +1.07 | 6.2 |

From order 41 on, the angular rule no longer limits PBE: the remaining
scatter of a few hundredths of a microhartree is the SCF tolerance and the
radial grid. r2SCAN reaches its plateau at order 35 as well, but the plateau
sits about 1.07 µHa above the reference. That offset does not come from the
angular rule, and the next series shows where it does come from.

Orders 59, 65, 77 and 95 are the convergence-study rules added in v0.17.16.
Through `run_job` they need a build of main from 2026-10-14 or later (after
v0.17.17): before that, the memory estimate refused them before the SCF
started, with "unsupported molecular Lebedev order 59".

## Explicit grids: the radial series

Hold the angular rule at order 59 and vary the number of Treutler-Ahlrichs
radial shells:

| Radial shells | PBE error, µHa | r2SCAN error, µHa |
|---:|---:|---:|
| 50 | -1.42 | +108.06 |
| 75 | -0.10 | +6.11 |
| 99 | +0.03 | +1.07 |
| 150 | 0.00 | +0.04 |
| 200 | reference | reference |

For PBE 75 shells suffice to a tenth of a microhartree. r2SCAN needs about
twice as many: its error falls from 108 µHa at 50 shells to 6 µHa at 75 and
1 µHa at 99, and only 150 shells remove the last microhartree. On this
molecule the meta-GGA's grid requirement is radial, not angular.

## What to take from this

- **Converge the radial grid first for meta-GGAs.** The angular rule
  saturated at order 35 to 41 for both functionals; the radial grid set the
  r2SCAN error.
- **The default grid is a good production grid.** `"orca-defgrid3"` is
  within 2 µHa (PBE) and 8 µHa (r2SCAN) of the converged energy, far below
  method errors. Use `"fine"` for benchmarks and for energy differences
  smaller than about 0.01 kcal/mol.
- **The extended rules are for checking, not for production.** Order 95
  costs five times order 41 and changes nothing here. Use it once, to prove
  that a lower order is converged for your system.

## Theory

**Atomic partition.** The molecular integral is split into atomic
contributions with weights $w_A(\mathbf r)$ that sum to one everywhere,

$$
E_{\mathrm{xc}} = \int f(\mathbf r)\,\mathrm d\mathbf r
= \sum_A \int w_A(\mathbf r)\, f(\mathbf r)\,\mathrm d\mathbf r ,
\qquad
w_A(\mathbf r) = \frac{P_A(\mathbf r)}{\sum_B P_B(\mathbf r)} .
$$

Becke's cell functions $P_A$ are products of smoothed step functions of the
elliptical coordinate between each pair of atoms
({vibeqc-cite}`becke_grid_1988`). Stratmann, Scuseria and Frisch replaced the
iterated polynomial by a piecewise function that is exactly zero or one far
from the cell boundaries, which lets most points skip the partition
altogether ({vibeqc-cite}`stratmann_grid_1996`); vibe-qc's named levels use
it.

**Radial quadrature.** Each atomic integral is done in spherical
coordinates around its nucleus. The radial coordinate is mapped onto a
finite interval and integrated with a Gauss-Chebyshev rule; vibe-qc uses the M4
mapping of Treutler and Ahlrichs ({vibeqc-cite}`treutler_ahlrichs_1995`),
which places many shells near the nucleus, where the density and the
kinetic-energy density vary fastest. Mura and Knowles analysed the accuracy
of such mappings ({vibeqc-cite}`mura_knowles_radial_1996`).

**Angular quadrature.** On each shell the angular integral uses a Lebedev
rule ({vibeqc-cite}`lebedev_laikov_1999`). A rule of order $L$ integrates
every spherical harmonic $Y_{lm}$ with $l \le L$ exactly with octahedral
symmetry and positive weights, using about $(L+1)^2/3$ points, roughly a
third of what a product Gauss-Legendre grid needs for the same degree:

$$
\int_{S^2} g(\Omega)\,\mathrm d\Omega \approx 4\pi \sum_{i=1}^{N_L} w_i\, g(\Omega_i),
\qquad
\text{exact for } g = Y_{lm},\ l \le L .
$$

**Pruning.** Near the nucleus the density is close to spherical, so the
named levels reduce the angular order on the inner shells
and on the far shells, where little density is left (the NWChem scheme
in the SG-1 tradition of Murray, Handy and Laming), which saves points without
measurable loss. The explicit series above turned pruning off to isolate
the angular rule.

## Variations

- **A heavier atom.** Repeat the radial series on a molecule with a
  third-row atom; the core needs more shells.
- **Geometry derivatives.** Gradients are more grid-sensitive than
  energies; compare `"orca-defgrid3"` and `"fine"` forces on a distorted
  geometry.
- **Range-separated and VV10 functionals.** VV10's non-local kernel uses a
  coarser second grid (`vv10_grid_factor`); converge it separately.

## Pitfalls

- **An unconverged SCF masks the grid.** Tighten `conv_tol_energy` to
  1e-10 for a grid study, as above, so that the SCF noise stays well below
  the grid differences you want to see.
- **Do not mix grids in a reaction energy.** Every structure in an energy
  difference needs the same grid level.
- **A grid you customise wins over `grid_level`.** Once you change any field
  of `opts.grid`, the named level no longer applies to that options object.

## See also

- [Functionals](../user_guide/functionals.md), including the grid keywords.
- [Molecular DFT](molecular_dft.md) and [functional comparison](functional_comparison.md).
- [Lattice-cutoff convergence](periodic_lattice_cutoff_convergence.md), the
  periodic counterpart for the exchange-correlation image domain.
- [`examples/molecular/input-h2o-grid-convergence.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-grid-convergence.py),
  the runnable script.
