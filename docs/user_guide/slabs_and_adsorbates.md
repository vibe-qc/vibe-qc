# Slabs and adsorbates

`vibeqc.build` is the native (no-ASE) helper for surface-catalysis
inputs. It generates genuine **two-dimensional** :class:`PeriodicSystem`
objects (`dim=2`): two in-plane lattice vectors, atoms at their real
Cartesian `z`, and **no vacuum gap**.

The submodule is deliberately small: closed-form surface unit cells for
the most common low-index facets of fcc, bcc, and hcp metals. For
exotic facets or non-metals, build a `PeriodicSystem` directly from
coordinates (see [crystal_lattices](crystal_lattices.md)).

## A slab needs no vacuum

A Gaussian-basis code does not need a vacuum gap to describe a surface.
Only the two in-plane lattice vectors are periodic; the direction normal
to the slab is simply never summed over. vibe-qc therefore treats a slab
as a true `dim=2` object and evaluates its electrostatics with the
rigorous **vacuum-free 2D Ewald** gauge (Parry 1975; de Leeuw and Perram
1979), exposed as `CoulombMethod.SLAB_EWALD_2D` and selected
automatically. This is the same choice CRYSTAL makes for its `SLAB`
mode.

`PeriodicSystem` still stores a full 3x3 lattice, because AO integrals
and spglib both want a 3D cell. For `dim=2` the third column `a3` is
**auto-synthesized** along the slab normal and is **not physical**: it is
bookkeeping only, and the SCF energy is invariant to its length. Do not
treat it as a vacuum gap, and do not tune it.

```{warning}
**Migration from the old vacuum convention.** Before v0.15.32,
`vq.slab(...)` returned a `dim=3` cell whose third lattice vector was a
load-bearing vacuum gap, and slab jobs were routed through the bulk 3D
Coulomb builders. That was a plane-wave idiom, and it was wrong: a slab
run as a 3D crystal of sheets `a3` apart yields an energy that depends
on `a3` *and* on the k-mesh.

* `vq.slab(...)` now returns `dim=2`. Pass **`periodic_z=True`** to get
  the legacy `dim=3`-with-vacuum cell, which is useful only as a
  3D-with-vacuum reference.
* For `dim=2` the `vacuum=` argument no longer affects the physics. It
  is retained purely as viewer / density-grid extent metadata on
  `SlabInfo`.
* Slab energies computed before this change are not comparable to the
  new ones. The new ones are `a3`-invariant.
```

Build a slab straight from lattice vectors with `vq.slab_2d`, passing
the atoms at their real `z` (bohr):

```python
import vibeqc as vq

# A graphene layer: two in-plane vectors + real-z atoms. No vacuum.
sheet = vq.slab_2d(
    [4.6487, 0.0000, 0.0],
    [2.3244, 4.0259, 0.0],
    [vq.Atom(6, [0.0000, 0.0000, 0.0]),
     vq.Atom(6, [2.3244, 1.3420, 0.0])],
)
assert sheet.dim == 2   # a3 is synthesized for you, and is non-physical
```

## A 5-layer Fe(100) slab with N₂ adsorbed

```python
import vibeqc as vq

# 5-layer Fe(100), 2×2 lateral cell. This is a dim=2 slab: no vacuum.
# The default lattice constant (2.866 Å) comes from
# vq.BULK_LATTICE_CONSTANTS; override with a=...
slab_sys, info = vq.slab(
    "Fe", facet=(1, 0, 0), n_layers=5,
    supercell=(2, 2),
    multiplicity=21,  # 5×4 = 20 atoms × 4 unpaired e-/Fe ≈ FM start guess
)
assert slab_sys.dim == 2

# Side-on N₂ at a bridge site, 1.9 Å above the top Fe layer.
slab_with_n2 = vq.place_adsorbate(
    slab_sys, "N2", info=info,
    site="bridge", orientation="side-on", height=1.9,
    bond_length=1.10,
)
```

The `info` object (a :class:`SlabInfo`) carries the per-atom layer
index, which feeds the frozen-substrate relaxation pattern below.

### Built-in adsorbate library

`vq.build_molecule(name, bond_length=...)` returns a small library of
common adsorbate geometries (in Å):

| Name | Atoms | Default geometry |
|------|-------|-------------------|
| `H`, `H2`, `N2`, `O2` | 1-2 | linear along z |
| `CO`, `OH`, `NH` | 2 | linear along z |
| `H2O`, `NH3`, `CH4` | 3-5 | gas-phase neutral geometry |

For anything else, pass an explicit `((symbol, (x, y, z)), …)` sequence
in Å.

### Adsorption sites

`place_adsorbate(slab, ads, site=..., info=info)` resolves named sites
against the primitive surface cell:

| Site | fcc(100), bcc(100) | fcc(111), hcp(0001) | bcc(110) |
|------|---------------------|----------------------|-----------|
| `"top"` | atop a metal | atop a metal | atop a metal |
| `"bridge"` | 2-fold bridge | 2-fold bridge | short bridge |
| `"hollow"` | 4-fold hollow | (degenerate w/ fcc/hcp) | quasi-3-fold |
| `"fcc-hollow"` | - | hcp(110) absent, fcc-only | - |
| `"hcp-hollow"` | - | offset 3-fold hollow | - |

For full control, pass `position=(x, y)` in Å (relative to the lattice
origin) instead of `site=`.

## Frozen-substrate relaxation

```{important}
Frozen-substrate relaxation and periodic Gaussian NEB need **`dim=3`**
today: they run on the BIPOLE force stack, which does not accept a
`dim=2` slab. Build the cell with `periodic_z=True` for these
workflows, and accept the vacuum-image error that convention carries.
Periodic BIPOLE Hessians fail closed; do not treat `dim=3` as a Hessian
workaround.
Plain closed-shell atomic relaxation of a `dim=2` slab (no freeze
masks) ships since 2026-07-30 on the slab-GDF analytic gradient:
`run_periodic_job(jk_method="gdf", optimize=True)`. Single-point
energies should use the `dim=2` default.
```

The standard pattern: freeze the bottom layers and relax only the top
layers + adsorbate. :func:`vibeqc.relax_atoms` takes a
`freeze_indices=` argument; :meth:`SlabInfo.bottom_layer_indices`
produces the list:

```python
# dim=3 with a real vacuum gap: required for forces.
slab3d, info3d = vq.slab("Fe", facet=(1, 0, 0), n_layers=5,
                         vacuum=12.0, supercell=(2, 2),
                         multiplicity=21, periodic_z=True)
slab3d_n2 = vq.place_adsorbate(slab3d, "N2", info=info3d,
                               site="bridge", orientation="side-on",
                               height=1.9, bond_length=1.10)

freeze = info3d.bottom_layer_indices(3)  # bottom 3 of 5 layers
opt = vq.relax_atoms(
    slab3d_n2,
    basis_name="sto-3g",
    kmesh=vq.monkhorst_pack(slab3d_n2, [2, 2, 1]),
    method="UKS",
    functional="pbe",
    freeze_indices=freeze,
)
```

Internally the optimizer pins the fractional coordinates of the frozen
atoms via L-BFGS-B box bounds and zeros their gradient components, so
the reported `|grad|` reflects only the free degrees of freedom.

### Animate the relaxation in vibe-view

Pass `output_trajectory="stem"` and `relax_atoms` writes a
vibe-view-renderable QVF archive on exit, one frame per accepted
L-BFGS-B step, with the initial geometry as frame 0 and the
converged geometry as the last frame:

```python
opt = vq.relax_atoms(
    slab_with_n2,
    basis_name="sto-3g",
    kmesh=vq.monkhorst_pack(slab_with_n2, [2, 2, 1]),
    method="UKS",
    functional="pbe",
    freeze_indices=freeze,
    output_trajectory="slab_n2_relax",   # → slab_n2_relax.qvf
)
```

The archive ships as **QVF v2** (per-frame lattice + dim) for any
`PeriodicSystem` input, so vibe-view's renderer (per the
"[Periodic reaction paths (QVF v2)](vibe_view.md)" section)
draws the unit cell and wraps atoms across in-plane periodic
boundaries automatically. Default `output_trajectory=None` is a
no-op, no per-step capture overhead.

## Open-shell + multi-k for metallic slabs

On a `dim=2` slab, open-shell **UKS** routes through the vacuum-free
`SLAB_EWALD_2D` gauge, at Gamma and multi-k. Metallic slabs typically
need a finite k-mesh in the surface plane, pass `kpoints=` to
:func:`vibeqc.run_periodic_job`. The slab normal is not periodic, so its
entry is always `1`:

```python
vq.run_periodic_job(
    slab_with_n2,
    basis=vq.BasisSet(slab_with_n2.unit_cell_molecule(), "sto-3g"),
    method="UKS",
    functional="pbe",
    jk_method="auto",           # -> SLAB_EWALD_2D on a dim=2 slab
    kpoints=[4, 4, 1],          # k-mesh in surface plane, 1 along the normal
)
```

For multi-k UKS slabs, the default QVF includes the real-space density grid
plus total and projected DOS sections. Output preparation preserves every
converged lattice-density block by cell index and zero-extends only the outer
cells requested by its longer visualization template.

`method="UHF"` on a slab is not supported yet and raises; use `UKS`.
On a **`dim=3`** cell (`periodic_z=True`), open-shell dispatches via
BIPOLE as before.

```{warning}
**Metallic slabs and smearing.** `smearing_temperature` is wired through
the GDF / GPW / GAPW / RIJCOSX routes, none of which accept a `dim=2`
slab, so a slab that needs Fermi-Dirac smearing to converge cannot be
smeared on the 2D route today. Until slab smearing lands, run such
systems as `dim=3` with thick vacuum (`vq.slab(..., periodic_z=True)`).
```

## Dispersion correction for periodic systems

`vq.compute_d3bj_periodic` returns the D3-BJ dispersion energy per
unit cell. Two backends:

* **`backend="dftd3"`** (recommended), Grimme's reference Fortran
  library via the optional `dftd3` Python package. Bit-exact periodic
  D3-BJ. Install with `pip install dftd3` or
  `pip install -e '.[dispersion]'`.
* **`backend="builtin"`**, vibe-qc's native C++ D3-BJ on a supercell
  expansion of the unit cell. Approximate (CN values near the
  supercell boundary are wrong); the per-cell energy converges with
  larger supercell. Use for elements not yet covered by the dftd3
  install or to stay external-dependency-free.
* **`backend="auto"`** (default), picks dftd3 when available.

Standalone call:

```python
import vibeqc as vq
slab_sys, info = vq.slab("Fe", facet=(1, 0, 0), n_layers=5,
                         supercell=(2, 2), multiplicity=21)
res = vq.compute_d3bj_periodic(slab_sys, "pbe", with_gradient=True)
print(res.energy, "Ha per cell;", res.backend, res.supercell)
```

`compute_d3bj_periodic` is dimensionality-aware: on a `dim=2` slab it
replicates only in-plane, so the reported `supercell` looks like
`(3, 3, 1)`.

Or inline with `run_periodic_job` via the `dispersion=` keyword. Leave
`jk_method` at its default `"auto"`, which resolves a `dim=2` slab to
the vacuum-free `SLAB_EWALD_2D` gauge:

```python
vq.run_periodic_job(
    slab_with_n2,
    basis=vq.BasisSet(slab_with_n2.unit_cell_molecule(), "sto-3g"),
    method="UKS",
    functional="pbe",
    jk_method="auto",            # -> SLAB_EWALD_2D for dim=2
    kpoints=[4, 4, 1],           # the normal axis is never sampled
    dispersion="pbe",            # or True (uses the current functional)
    dispersion_backend="auto",
)
```

```{note}
Passing an explicit **bulk** `jk_method` (`"gdf"`, `"bipole"`, `"gpw"`,
`"gapw"`, `"rijcosx"`) to a `dim=2` slab now raises
`NotImplementedError` rather than silently returning an `a3`-dependent
energy. Use `"auto"` (or `"slab_ewald_2d"`). `method="UHF"` on a slab
also raises for now; use `RHF`, `RKS`, or `UKS`.
```

The dispersion piece is logged as a separate block in the `.out`
file and the SCF result is wrapped with a `_DispersionAugmented`
proxy that exposes `.energy_total = E_SCF + E_disp` (the bare
`.energy` is unchanged).

Pass the slab `PeriodicSystem` unchanged. For a `dim=2` slab the lattice
sum runs over the two in-plane axes only, so the dispersion sum is
naturally bounded along the normal.

## Screened hybrids (HSE06) on slabs

HSE06 runs correctly on `dim=2` slabs through the AUTO /
`slab_ewald_2d` route (and on 3-D cells through the `bipole` and
Ewald drivers): the exact exchange is the genuine short-range
`0.25 · K_erfc(ω = 0.11 bohr⁻¹)` direct lattice sum: screened, not
the full-range PBE0 kernel that every periodic route silently
substituted before 2026-07-12.

On a symmetry-reduced Monkhorst-Pack mesh, RKS and UKS rotate every
irreducible density through its complete 2D star before folding the
real-space J/K/XC density. This applies to orthogonal and oblique slabs.
An explicit weighted k-point list has no star provenance and retains the
quadrature supplied by the caller.

```python
result = vq.run_periodic_job(
    slab, basis, method="RKS", functional="hse06",
    jk_method="auto",           # dim=2 -> SLAB_EWALD_2D
    kpoints=(4, 4, 1),
)
```

The slab total stays invariant to the synthesized `a3` length and
k-mesh-convergent, same §-gates as PBE. Long-range-corrected RSH
(ωB97X, CAM-B3LYP, LC-ωPBE) fail closed on all periodic routes, and
routes without an erfc kernel (GDF, GPW/GAPW, COSX) fail closed on
every range-separated functional; see
[Functionals § HSE06](functionals.md#hse06-screened-range-separated-hybrid).
Analytic slab gradients do not cover screened hybrids (they fail
closed; see below).

## What's not here yet

The following surface-catalysis features are tracked in
[`docs/roadmap.md`](../roadmap.md) but not yet shipped:

* **Slab analytic gradients.** The `SLAB_EWALD_2D` gauge ships energies
  (RHF / RKS / UKS, Gamma and multi-k); its analytic gradient is still
  being written, so geometry optimisation and forces on a `dim=2` slab
  raise. As a stopgap, build the cell with `periodic_z=True` and
  optimise the `dim=3`-with-vacuum system, accepting the `a3`-dependent
  error that convention carries.
* **Open-shell HF (`UHF`) on a slab.** Use `UKS` instead.
* **Density-fitted (GDF) and grid (GPW) 2D Coulomb**, for large slabs
  where the 2D-Ewald sum becomes the bottleneck.
* Relaxed bond-length / angle scans on the surface.
* Nudged elastic band (NEB / CI-NEB).
* Periodic D4 (the molecular D4 path already exists; periodic
  generalisation pending).

The Hubbard-U correction (`+U`) for strongly-correlated `d/f`
electrons is now shipped, see
[DFT+U](dft_plus_u.md) for the full surface (`vq.HubbardSite(...)`
on `run_periodic_job`, energies + Fock + per-spin per-k path);
periodic +U gradients are still being plumbed.

If you need any of these urgently, file an issue.
