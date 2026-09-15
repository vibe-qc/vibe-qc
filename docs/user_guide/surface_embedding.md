# Green's-function surface embedding

```{admonition} Feature status — experimental
:class: warning

Green's-function surface embedding is **experimental**. The analytic 1D
foundation (semi-infinite tight-binding substrate) and the 3D Gaussian
pipeline are validated and regression-tested, but the method has **not
yet** been shown to reproduce a thick-slab [GDF](density_fitting.md)
reference to tight tolerance (see the project's periodic-SCF discipline,
CLAUDE.md §7). Treat absolute energies and densities as **qualitative**
until that parity lands. Every embedded-surface run emits an
``EmbeddedSurfaceExperimentalWarning``. The API lives under
``vibeqc.periodic_embedding`` and is **not** exported from the top-level
``vibeqc`` namespace.
```

To silence the per-run warning:

```python
import warnings
from vibeqc.periodic_embedding import EmbeddedSurfaceExperimentalWarning
warnings.filterwarnings("ignore", category=EmbeddedSurfaceExperimentalWarning)
```

## What it is

For an adsorbate in the dilute (zero-coverage) limit, a finite supercell
slab forces an unphysical choice: a thin slab has spurious quantum-size
oscillations along the surface normal, and a laterally small cell makes
the adsorbate interact with its own periodic images. Surface embedding
removes both.

* **Layer A, surface-normal embedding (Inglesfield).** The finite slab
  in *z* is replaced by a true semi-infinite substrate via an
  energy-dependent, non-local embedding potential `Σ_emb` on a dividing
  plane. Only *region I* (the surface plus adsorbate) is solved; lateral
  periodicity is kept via Bloch k-points in the 2D surface Brillouin
  zone. `Σ_emb` comes from the bulk Green function, built by
  Sancho-Rubio principal-layer decimation. *No slab-thickness artifact.*
* **Layer B, lateral impurity embedding (Dyson).** The clean-surface
  Green function `G0` from Layer A is perturbed by the adsorbate as a
  localized `ΔV` of finite real-space support; `G = G0 + G0 ΔV G` is
  solved on that region and the adsorption energy comes from Lloyd's
  formula. *No adsorbate-adsorbate lateral interaction.*

The numerical backbone, shared by both layers, is: density by
complex-energy contour integration, Ishida energy-linearization of
`Σ_emb`, Sancho-Rubio decimation, and Lloyd's formula for energetics.

For the conventional finite-supercell slab route (no embedding), see
[Slabs and adsorbates](slabs_and_adsorbates.md).

## The runner

`run_embedded_surface` is the high-level entry point. It takes a clean
slab `PeriodicSystem`, a `BasisSet`, and per-atom integer **tags**
(`0` = substrate, `1` = region I), and returns an `EmbeddedSurfaceResult`
carrying the region-I density, occupation, band energy, contour, and
per-k `Σ_emb`.

```python
import numpy as np
from vibeqc._vibeqc_core import Atom, BasisSet, PeriodicSystem
from vibeqc.periodic_embedding import (
    TAG_REGION_I,
    TAG_SUBSTRATE,
    run_embedded_surface,
)

# A 6-layer H slab. dim=2: the third column is bookkeeping, not a
# vacuum gap (the embedding builds its own Lpq, so it accepts dim=2).
a, c, sp = 10.0, 40.0, 2.0  # bohr
lat = np.array([[a, 0, 0], [0, a, 0], [0, 0, c]])
z0 = (c - 5 * sp) / 2
atoms = [Atom(1, [a / 2, a / 2, z0 + i * sp]) for i in range(6)]
system = PeriodicSystem(dim=2, lattice=lat, unit_cell=atoms)  # singlet
basis = BasisSet(system.unit_cell_molecule(), "sto-3g")

tags = [TAG_SUBSTRATE] * 4 + [TAG_REGION_I] * 2   # 4 sub + 2 region I
result = run_embedded_surface(
    system, basis, tags,
    surface_k_mesh=(2, 2),
    contour_n_nodes=40,
    e_bottom=-2.2, e_fermi=-1.0,   # fix the contour window (recommended)
    scf_method="one_shot",
)
print(result.occupation, result.band_energy)
```

### Density methods (`scf_method`)

| `scf_method` | Density model |
|--------------|---------------|
| `"one_shot"` (default) | Non-self-consistent: `Hcore + Σ_emb` only. |
| `"hartree"` | Diagonal Hartree self-consistency. |
| `"hf"` | Full Hartree-Fock with GDF J and K. |
| `"uhf"` | Spin-polarised UHF (J + K). **The open-shell route.** |

`Σ_emb` is built once and held fixed. The full two-step Ishida
self-consistency that rebuilds `Σ_emb` from the SCF substrate density is
in progress; a substrate-SCF mean field is available via `sigma_scf=True`.
[`examples/embedding/two_step_scf_and_forces.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/embedding/two_step_scf_and_forces.py)
runs both the Hcore and `sigma_scf=True` embeddings side by side on a
4-atom H-chain substrate + region-I system (the SCF-level potential
shifts the substrate bands up relative to Hcore, as expected), and
drives `EmbeddedSurfaceCalculator`'s numerical forces through `ase`
so a standard `ase.optimize` relaxer can move the region-I geometry.

### The contour window

`e_fermi` and `e_bottom` (Ha) bound the complex-energy contour. If
omitted they are auto-estimated from the substrate's `S⁻¹H` spectrum at
Γ, but for geometry optimisation you should **pass explicit values**: the
numerical-force loop then holds the window fixed across displacements.
This is the force at fixed Fermi level (the semi-infinite substrate is an
electron reservoir), and it also removes finite-difference noise.

## The ASE calculator

`EmbeddedSurfaceCalculator` wraps the runner as an
[ASE `Calculator`](ase_integration.md), giving `get_potential_energy()`
(eV) and `get_forces()` (eV/Å). Forces are **central-difference
numerical** gradients of the returned energy (`F = -dE/dR`); the
analytical `Σ_emb` gradient is future work. Use `force_atom_indices` to
differentiate only the mobile atoms: a slab then pays 6 displaced runs
per mobile atom and nothing for the frozen substrate.

```python
from ase import Atoms
from vibeqc.periodic_embedding.ase_calc import EmbeddedSurfaceCalculator

atoms = Atoms("H4", positions=..., cell=[30, 30, 30], pbc=True)
atoms.set_tags([0, 0, 1, 1])           # 2 substrate, 2 region I

atoms.calc = EmbeddedSurfaceCalculator(
    basis_name="sto-3g",
    surface_k_mesh=(2, 2),
    e_bottom=-2.2, e_fermi=-1.0,        # hold the contour window fixed
    force_atom_indices=[2, 3],          # only region-I atoms are mobile
)
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV / Å
```

## Open-shell (spin-polarised) cells

By default the calculator builds a closed-shell singlet
(`charge=0, multiplicity=1`). An **odd-electron** cell, or any
deliberately non-singlet cell, needs two things:

1. `multiplicity > 1` on the calculator (or, for the runner, on the
   `PeriodicSystem` itself). The unit cell's electron count and
   `multiplicity` must share parity, otherwise the
   `PeriodicSystem`/molecule constructor raises
   *"n_electrons and multiplicity are inconsistent"* **before** any
   embedding work runs.
2. A spin-polarised density route: `scf_method="uhf"`.

```python
from ase import Atoms
from vibeqc.periodic_embedding.ase_calc import EmbeddedSurfaceCalculator

# 3-atom H chain in a slab cell: 3 electrons -> doublet.
atoms = Atoms(
    "H3",
    positions=[(2.0, 2.0, 7.5 + 1.5 * i) for i in range(3)],
    cell=[4.0, 4.0, 18.0], pbc=True,
)
atoms.set_tags([0, 0, 1])              # 2 substrate, 1 region I

atoms.calc = EmbeddedSurfaceCalculator(
    basis_name="sto-3g",
    surface_k_mesh=(1, 1),
    contour_n_nodes=24,
    e_bottom=-2.2, e_fermi=-1.0,
    charge=0,
    multiplicity=2,                    # 3 electrons -> doublet
    scf_method="uhf",                  # spin-polarised density
    force_atom_indices=[2],
)
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV / Å
```

`charge` and `multiplicity` are forwarded straight into the
`PeriodicSystem` the calculator builds from the ASE `Atoms`; any extra
keyword (`scf_method=`, `scf_mixing=`, and so on) is forwarded to
`run_embedded_surface`.

```{note}
The open-shell route here is **UHF** (J + K only). Spin-polarised
*Kohn-Sham* embedded surfaces (UKS with a `Vxc` augmentation) are a
roadmap item (see ``handovers/HANDOVER_GF_EMBEDDING.md``).
```

## Runnable example

A complete, runnable open-shell script ships at
`examples/embedding/open_shell_adsorbate.py`; the closed-shell Layer A
plus Layer B (Lloyd adsorption energy) walkthrough is
`examples/embedding/simple_adsorbate.py`. The
[surface-embedding tutorial](../tutorial/surface_embedding.md) walks
through the open-shell case step by step.

## See also

- [Slabs and adsorbates](slabs_and_adsorbates.md): the conventional
  finite-supercell slab builder (`vq.slab`, `vq.place_adsorbate`).
- [ASE integration](ase_integration.md): using vibe-qc calculators as
  ASE `Calculator` objects for relaxations and workflows.
- [Open-shell Mg⁺• tutorial](../tutorial/open_shell_mg_cation.md):
  open-shell periodic SCF in the conventional supercell picture.
- [Density fitting (GDF)](density_fitting.md): the thick-slab reference
  the embedding method is being validated against.

## References

- J. E. Inglesfield, *J. Phys. C* **14**, 3795 (1981),
  [doi:10.1088/0022-3719/14/26/015](https://doi.org/10.1088/0022-3719/14/26/015).
- H. Ishida, *Phys. Rev. B* **63**, 165409 (2001),
  [doi:10.1103/PhysRevB.63.165409](https://doi.org/10.1103/PhysRevB.63.165409).
- M. P. López Sancho, J. M. López Sancho and J. Rubio, *J. Phys. F*
  **15**, 851 (1985),
  [doi:10.1088/0305-4608/15/4/009](https://doi.org/10.1088/0305-4608/15/4/009).
- P. Lloyd, *Proc. Phys. Soc.* **90**, 207 (1967).
