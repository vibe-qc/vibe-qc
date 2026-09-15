# Open-shell Green's-function surface embedding

```{admonition} Experimental method
:class: warning

Green's-function surface embedding is **experimental** and not yet
validated to thick-slab [GDF](../user_guide/density_fitting.md) parity.
The numbers in this tutorial are **qualitative**: they exercise the
machinery and the open-shell plumbing, not a converged adsorption
energy. See the [user-guide page](../user_guide/surface_embedding.md)
for the feature-status detail.
```

Surface embedding treats an adsorbate plus a patch of substrate as a
localized perturbation on a laterally periodic, *semi-infinite* clean
surface, reaching the dilute-limit adsorbate with no slab-thickness
artifact and no adsorbate-adsorbate image interaction. This tutorial
drives the **open-shell** path: an odd-electron cell solved with a
spin-polarised (UHF) embedded density, through the ASE calculator.

**You will**:

- Build an odd-electron slab cell (a 3-atom H chain, a doublet).
- Run it through `EmbeddedSurfaceCalculator` with `multiplicity=2` and
  `scf_method="uhf"`.
- Get the embedded energy and numerical forces, the way an ASE optimiser
  would.
- See why an open-shell cell needs *both* the right multiplicity and a
  spin-polarised density route.

**Background**: the [surface-embedding user
guide](../user_guide/surface_embedding.md) for the API and the two
layers; [Open-shell Mg⁺•](open_shell_mg_cation.md) for open-shell
*periodic SCF* in the conventional supercell picture; [ASE
integration](../user_guide/ase_integration.md) for the calculator
interface.

## Why open-shell needs two ingredients

A cell with an odd electron count cannot be a closed-shell singlet. Two
independent things have to agree:

1. **Multiplicity.** `2S + 1` must match the electron-count parity. A
   3-electron cell is a doublet (`multiplicity=2`). If you leave the
   default `multiplicity=1`, the unit-cell molecule fails its parity
   check (*"n_electrons and multiplicity are inconsistent"*) **before**
   any embedding work runs.
2. **A spin-polarised density.** The default density method
   (`one_shot`/`hf`) is closed-shell. Spin-up and spin-down must be
   allowed to differ, so you pass `scf_method="uhf"`.

Set one without the other and you either trip the parity error or
silently restrict the density to closed-shell.

## Setup

The geometry is a 3-atom H chain stacked along the surface normal in a
slab cell (a long *c*-axis marks it as a 2D-periodic slab). Three H atoms
carry three electrons, an open-shell doublet. Centring the chain in the
lateral plane forces the in-plane forces to vanish by symmetry, leaving
only the surface-normal (*z*) force.

```python
import numpy as np
from ase import Atoms
from vibeqc.periodic_embedding.ase_calc import EmbeddedSurfaceCalculator

# 3-atom H chain (slab cell): 3 electrons -> doublet.
a, c, sp = 4.0, 18.0, 1.5            # Angstrom
z0 = (c - 2 * sp) / 2.0
atoms = Atoms(
    "H3",
    positions=[(a / 2, a / 2, z0 + i * sp) for i in range(3)],
    cell=[a, a, c], pbc=[True, True, True],
)
atoms.set_tags([0, 0, 1])            # bottom 2 = substrate, top = region I
```

The `EmbeddedSurfaceCalculator` carries the open-shell choice:
`multiplicity=2` (threaded into the `PeriodicSystem` it builds) and
`scf_method="uhf"` (forwarded to the runner). The contour window
(`e_bottom`, `e_fermi`) is fixed explicitly so the numerical-force loop
holds it constant across displacements; `force_atom_indices=[2]` makes
only the region-I atom mobile.

```python
atoms.calc = EmbeddedSurfaceCalculator(
    basis_name="sto-3g",
    surface_k_mesh=(1, 1),
    contour_n_nodes=24,
    e_bottom=-2.2, e_fermi=-1.0,
    charge=0,
    multiplicity=2,                  # 3 electrons -> doublet
    scf_method="uhf",                # spin-polarised density
    force_atom_indices=[2],          # only the region-I atom is mobile
)
```

## Run it

```python
energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()             # eV / Angstrom

print(f"Embedded energy: {energy:.4f} eV")
for i, f in enumerate(forces):
    print(f"  atom {i}: {f[2]:+.4f} eV/A (z)")
print(f"Region-I occupation: {atoms.calc.last_result.occupation:.4f} e-")
```

Expected (coarse 24-node contour, STO-3G, qualitative):

```
Embedded energy: 128.0366 eV
  atom 0: +0.0000 eV/A (z)
  atom 1: +0.0000 eV/A (z)
  atom 2: +11.5797 eV/A (z)
Region-I occupation: 0.2420 e-
```

The substrate atoms (outside `force_atom_indices`) keep exactly zero
force; the in-plane components of the mobile atom vanish by the chain's
x/y symmetry; only the surface-normal force is nonzero. The force is the
exact central-difference gradient of the returned energy, so as the
energy model matures the force tracks it automatically.

```{note}
The embedded "energy" here is the placeholder Layer-A total
(`band_energy + nuclear_repulsion`), not a converged adsorption energy,
hence the un-physical magnitude. What this tutorial validates is the
*open-shell plumbing*: the doublet cell builds, the UHF density runs, and
energy plus forces come back finite and symmetry-correct.
```

The complete script is `examples/embedding/open_shell_adsorbate.py`.

## What this tells you

- **Open-shell embedded surfaces run via `multiplicity` plus
  `scf_method="uhf"`.** `charge` and `multiplicity` are threaded into the
  `PeriodicSystem` the calculator builds; the UHF route gives the
  spin-polarised density.
- **Numerical forces come for free.** `get_forces()` central-differences
  the embedded energy with the contour window held fixed, so any ASE
  optimiser or vibrational driver works unchanged.
- **The method is experimental.** Absolute energies are qualitative until
  thick-slab GDF parity lands; use it to explore the embedding workflow,
  not for production adsorption energies yet.

## Where this generalises

- **Closed-shell surfaces**: drop `multiplicity`/`scf_method` and use the
  defaults (singlet, one-shot). See
  `examples/embedding/simple_adsorbate.py` for the Layer A plus Layer B
  (Lloyd adsorption energy) walkthrough.
- **Spin-polarised DFT (UKS)**: a roadmap item. The open-shell route is
  UHF (J + K) today; UKS with a `Vxc` augmentation is future work
  (`handovers/HANDOVER_GF_EMBEDDING.md`).
- **Conventional supercell slabs**: for the non-embedding picture, see
  [Slabs and adsorbates](../user_guide/slabs_and_adsorbates.md) and the
  [slab-adsorbate DFT tutorial](slab_adsorbate_dft.md).

## See also

- [Green's-function surface embedding](../user_guide/surface_embedding.md):
  the user-guide reference for the runner and the calculator.
- [Open-shell Mg⁺• in a periodic box](open_shell_mg_cation.md):
  open-shell periodic SCF with the Makov-Payne correction.
- [Open-shell systems (UHF, UKS, UMP2)](open_shell.md): molecular
  open-shell basics.
