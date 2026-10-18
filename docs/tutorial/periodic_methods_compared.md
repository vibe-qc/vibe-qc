---
myst:
  html_meta:
    "description": "Hands-on guide to choosing a periodic Coulomb-build route in vibe-qc: GDF (Gaussian density fitting), BIPOLE (an Ewald J-split with a shared Ewald gauge), the GPW / GAPW plane-wave family, and RIJCOSX (density-fitted J plus chain-of-spheres K). How to switch routes with one keyword, what each is, and a decision table. GPW and GAPW are vibe-qc's own grid implementation; the external GPAW is a parity reference only."
    "og:title": "vibe-qc tutorial - choosing a periodic method (GDF, BIPOLE, GPW/GAPW)"
---

# Choosing a periodic method: GDF, BIPOLE, and GPW / GAPW

A molecular calculation has exactly one way to build the Coulomb (and
exchange) matrices. A *periodic* one has a choice, because the bare
lattice sum of $1/r$ over an infinite crystal diverges, and there is more
than one good way to tame it. This page is the map: what the routes are,
how to switch between them with a single keyword, and how to pick the
right one.

```{admonition} Route status at a glance
:class: note

The labels below follow [Methods: theory and status](../methods.md#how-to-read-the-status-labels).

| Route (`jk_method=`) | Status |
|---|---|
| `"gdf"` | **production** for Gamma and multi-k RHF, RKS, UHF, UKS in 3-D (a 1-D wire is modelled as a 3-D cell with transverse vacuum; `dim=1` is refused) and the bounded closed-shell 2-D slab route; periodic ROHF on GDF is a **preview** |
| `"bipole"` | **production** energies and finite-difference forces (Gamma and multi-k RHF, UHF, RKS, UKS, 3-D); analytic gradients are a **preview**; the quartet multipole far field is **gated** |
| `"gpw"` | **production** within its shipped envelope (Gamma RHF, UHF, RKS, UKS; multi-k pure-DFT RKS, ROKS, UKS; Gamma analytic gradients; DFT+U; D3-BJ); GPW ROHF/ROKS are a **preview**; the `.out` route line carries an experimental tag and the grid infrastructure emits `GAPWExperimentalWarning` once per grid build while the GPW/GAPW family is completed |
| `"gapw"` | **experimental** |
| `"rijcosx"` | **experimental** |
| `"auto"` on a `dim=2` slab (SLAB_EWALD_2D) | **production** for RHF, RKS and UKS |
| `method="aiccm"` with any `variant` | **experimental** |
```

```{seealso}
This is the hands-on companion to the reference page
[periodic-SCF methods](../user_guide/periodic_methods.md), which derives
*why* the lattice sum diverges and *how* each route fixes it (Ewald
splitting, density fitting, FFT-Poisson on a grid). Read that page when
you want the theory; read this one when you want to run something.

When you go to **validate** a route against an external program, see
[periodic JK routes & parity policy](../periodic_jk_routes.md): each route
is checked against a program from its own method family (GDF ↔ PySCF,
BIPOLE ↔ CRYSTAL, GPW / GAPW ↔ CP2K / GPAW), and cross-family comparison
is a trap.
```

## One system, several routes

You select the route with the `jk_method` keyword on `run_periodic_job`.
The system, basis, and method stay the same, only the Coulomb engine
underneath changes:

```python
import vibeqc as vq
# ... define `system` (a PeriodicSystem) and `basis` ...

# Production default for closed-shell RHF / RKS (Gamma or multi-k):
res = vq.run_periodic_job(system, basis, method="RKS", functional="pbe",
                          kpoints=[4, 4, 4], jk_method="gdf")

# BIPOLE: the Ewald J-split route (open shell, hybrids, CRYSTAL14 cross-validation):
res = vq.run_periodic_job(system, basis, method="RKS", functional="pbe",
                          kpoints=[4, 4, 4], jk_method="bipole")

# Plane-wave routes: GPW is production within its envelope, GAPW is experimental.
res = vq.run_periodic_job(system, basis, method="RKS", functional="pbe",
                          jk_method="gpw")    # or jk_method="gapw"

# Independent finite cyclic-cluster RHF/RKS (experimental):
res = vq.run_periodic_job(system, basis, method="RKS", functional="pbe0",
                          jk_method="aiccm2026dev-b",
                          aiccm_lattice_extension=[4, 4, 4],
                          aiccm_backend="rijcosx")
```

If you omit `jk_method` (or pass `"auto"`), vibe-qc chooses for you. It
branches on dimensionality first: a **`dim=2` slab** always goes to
**SLAB_EWALD_2D**, the vacuum-free 2D Ewald gauge, and every bulk route
raises there. Otherwise it is **GDF** for closed-shell RHF/RKS and
**BIPOLE** for open-shell UHF/UKS. The default is already a
production-quality route; you reach for this page when the default is not
the one you want. (Open-shell GDF also ships and is reachable with an
explicit `jk_method="gdf"`; AUTO keeps BIPOLE as the open-shell default by
design.)

## The routes

**GDF, Gaussian density fitting.** The production default for
closed-shell HF and KS, at $\Gamma$ and at a full k-mesh, with open-shell
UHF / UKS available too (Gamma and multi-k). It projects the AO pair
density onto a charge-compensated auxiliary basis, which cancels the
divergent monopole at the source and leaves a finite, fast Coulomb
metric ({vibeqc-cite}`sun_berkelbach_gdf_2017`). Best all-round choice for ionic and covalent insulators. Deep
dive: [`density_fitting.md`](../user_guide/density_fitting.md).

**BIPOLE, an Ewald J-split.** Splits $1/r$ into a short-range
piece summed in real space and a long-range piece summed in reciprocal
space ({vibeqc-cite}`ewald_lattice_sum_1921`), sharing a single Ewald
parameter across the electron-nucleus,
electron-electron, and nucleus-nucleus terms, with the Gaussian-density
electrostatics of {vibeqc-cite}`saunders_bipole_1992` and the CRYSTAL
conventions of {vibeqc-cite}`dovesi_crystal14_2014`. This is the route for **open-shell** systems, **hybrid functionals
at multiple k-points**, and **cross-validating against a CRYSTAL14 reference**. Production
for 3D; the multi-k corrected-gauge exchange is an active frontier. Deep
dive: [`bipole.md`](../user_guide/bipole.md).

**GPW, Gaussian + plane-wave** ({vibeqc-cite}`lippert_hutter_gpw_1997`).
Carries the density on a uniform grid and
solves the Poisson equation by FFT (an $\mathcal{O}(N_g\log N_g)$ Hartree
build), pinning the $\mathbf{G}=0$ component to zero for a neutral cell.
It is production within its shipped envelope: $\Gamma$-point RHF, RKS, UHF,
and UKS plus multi-k pure-DFT RKS, ROKS and UKS, with Gamma analytic forces,
the finite-difference Hessian, DFT+U, smearing, and D3(BJ) dispersion; GPW
ROHF and ROKS are a maintained preview. It carries an all-electron density on
the grid, so no pseudopotential files are involved. The `.out` route
line still carries an experimental tag and the plane-wave grid infrastructure
emits `GAPWExperimentalWarning` once per grid build while the GPW/GAPW family
is completed; GDF and BIPOLE are the longer-validated routes. Deep dive:
[`gapw.md`](../user_guide/gapw.md).

**GAPW, Gaussian-augmented plane-wave**
({vibeqc-cite}`lippert_hutter_parrinello_gapw_1999`,
{vibeqc-cite}`krack_parrinello_allelectron_2000`). GPW plus a per-atom radial
augmentation that sharpens all-electron accuracy by splitting the Hartree
potential into a smooth grid part and a hard atomic correction. Experimental:
molecular-limit Gamma HF and block-construction DFT ship, compact-crystal
augmentation validation is pending, and the library drivers emit
`GAPWExperimentalWarning`. Deep dive: [`gapw.md`](../user_guide/gapw.md).

**RIJCOSX, density-fitted J plus chain-of-spheres K.** Builds the Coulomb
matrix by GDF density fitting and the exchange by a seminumerical
chain-of-spheres (COSX, {vibeqc-cite}`neese_rijcosx_2009`) sum on a
periodic Becke grid. It runs a full
$\Gamma$-point RHF SCF today and true multi-k RHF, RKS, UHF, and UKS on
uniform time-reversal-symmetric meshes through the GDF/COSX backend. The
dedicated $\Gamma$ route is at parity with GDF for vacuum-padded
(molecular-limit) cells; its tight-cell exchange K is still being brought
to GDF parity, so use GDF for production tight-cell hybrids meanwhile.
Periodic RIJCOSX is experimental. Deep dive:
[`density_fitting.md`](../user_guide/density_fitting.md).

```{note}
**The plane-wave routes are vibe-qc's own grid / Gaussian-augmented
implementation.** GPW and GAPW are computed entirely by vibe-qc's own
kernels (GAPW is the all-electron variant, a hard/soft Gaussian density
partition rather than the projector-augmented-wave method), and the ASE
Calculator and command-line interface mirror GPAW's so a GPAW user is at
home. The separate external GPAW code is used only as an out-of-process
parity reference; vibe-qc never imports it or calls it to compute an energy.
```

```{note}
**Beyond the k-point routes: the cyclic cluster model (CCM).** vibe-qc's
signature periodic approach is real-space rather than reciprocal-space: the
[cyclic cluster model](cyclic_cluster_model.md) treats a crystal as a single
finite cyclic cluster rather than sampling the Brillouin zone
({vibeqc-cite}`bredow_geudtner_jug_ccm_2001`, {vibeqc-cite}`peintinger_ccm_2014`). It is
available today at the semi-empirical (MSINDO) level; two distinct
experimental ab-initio CCM approaches ship separately -
[Γ-CCM / `aiccm2026dev-a`](../aiccm2026dev_a.md) (union-and-weight/
Wigner-Seitz integral weighting) and
[χ-CCM / `aiccm2026dev-b`](../user_guide/aiccm2026dev_b.md)
(finite-translation-group characters; 3-D only) - and are under active
head-to-head study. No cross-approach delta is currently reportable. See the
[CCM tutorial](cyclic_cluster_model.md) and
the [CCM reference](../user_guide/cyclic_cluster_model.md).
```

## How to choose

All of these target the *same* periodic Hamiltonian, but their total
energies agree only once the finite Coulomb and exchange-divergence
conventions are matched (see [Running a real comparison](#running-a-real-comparison),
whose MgO GDF and BIPOLE pair differs by about 0.5 Ha). They differ in cost,
in which methods they support, and in maturity. Pick by your situation:

| Your situation | Route | Why |
|---|---|---|
| Closed-shell RHF / RKS, $\Gamma$ or multi-k | **GDF** (default) | fastest production route; chemical accuracy on insulators |
| Open-shell UHF / UKS | **BIPOLE** (AUTO default) or **GDF** | both run open-shell $\Gamma$ + multi-k; AUTO keeps BIPOLE |
| Full-range hybrid at multiple k-points | **BIPOLE** or **RIJCOSX** | both carry exact-exchange K at $\mathbf{k}\neq 0$; RIJCOSX uses the composed COSX backend |
| Cross-validating against CRYSTAL | **BIPOLE** | the shared-$\alpha$ Ewald J-split gives directly comparable energies |
| Tight ionic crystal at $\Gamma$ | **GDF** | auxiliary fit avoids grid / Ewald overhead |
| Plane-wave Hartree-J, large smooth-density cell | **GPW** | smooth-grid FFT-Poisson Hartree; $\Gamma$ RHF/RKS/UHF/UKS + multi-k pure-DFT RKS/ROKS/UKS (production within that envelope) |
| Plane-wave with all-electron augmentation | **GAPW** | per-atom augmentation, no pseudopotential files (experimental) |
| Seminumerical exchange at $\Gamma$ or uniform multi-k | **RIJCOSX** | GDF-J + chain-of-spheres COSX-K; use GDF for tight-cell $\Gamma$ hybrids |

```{important}
The automatic default (**GDF** for closed-shell, **BIPOLE** for
open-shell) already covers closed- and open-shell work, and both are
the longer-validated production routes. **GPW** is production within its
envelope and is reached by an explicit `jk_method="gpw"`; **GAPW** and
**RIJCOSX** also run a full SCF today but are experimental (RIJCOSX has
multi-k coverage, but its dedicated $\Gamma$ tight-cell exchange is not yet
at GDF parity). For a $\Gamma$-point hybrid functional on a tight ionic
crystal, stay on GDF or BIPOLE until that gap closes.
```

## Running a real comparison

The honest way to compare routes is to run the *same* crystal through more
than one and watch the total energy land on the same value after the finite
Coulomb and exchange-divergence conventions are matched. Then they are
different engines for the same declared Hamiltonian, so any disagreement beyond
the fitting/grid tolerance is a bug, not a feature.

Good starting points, all with verified reference output you can diff
against:

- The periodic foundations: [periodic_hf](periodic_hf.md) and
  [periodic_dft](periodic_dft.md).
- A full multi-k walkthrough on the default GDF route:
  [lih_multi_k](lih_multi_k.md) and
  [lih_pob_tzvp_solid_state](lih_pob_tzvp_solid_state.md).
- A current MgO route fixture:
  {download}`bundle README <../_static/examples/mgo-route-gdf-bipole/README.md>`,
  [GDF input](../_static/examples/mgo-route-gdf-bipole/input-mgo-rhf-sto3g-gdf-k222.py),
  [GDF `.out`](../_static/examples/mgo-route-gdf-bipole/output-mgo-rhf-sto3g-gdf-k222.out),
  [GDF `.qvf`](../_static/examples/mgo-route-gdf-bipole/output-mgo-rhf-sto3g-gdf-k222.qvf),
  [BIPOLE `.out`](../_static/examples/mgo-route-gdf-bipole/output-mgo-rhf-sto3g-bipole-k222.out),
  [BIPOLE `.qvf`](../_static/examples/mgo-route-gdf-bipole/output-mgo-rhf-sto3g-bipole-k222.qvf),
  and [vibe-view captures](../_static/examples/mgo-route-gdf-bipole/captures/mgo-gdf-structure.png).
  The GDF run converged in 14 iterations to
  `-271.7175700123166` Ha per primitive cell on a `2 x 2 x 2`
  Monkhorst-Pack mesh. The symmetry-reduced BIPOLE run converged in 23
  iterations to `-271.21454466172247` Ha per primitive cell. The bundle is
  useful for inspecting route-specific inputs, logs, QVFs, and viewer output;
  do not read this pair as a GDF/BIPOLE parity assertion.
- The runnable scripts under [`examples/periodic/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic/) (LiH, MgO, NaCl, and more); change `jk_method` and re-run to compare the routes.

For the k-space machinery every periodic route shares (the Brillouin zone,
Bloch phase, and why a k-mesh is needed at all), read
[kpoints_brillouin_bloch](kpoints_brillouin_bloch.md).
