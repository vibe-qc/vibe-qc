# Green's-function surface embedding

```{admonition} Status: experimental
:class: warning

Green's-function surface embedding is an experimental route. It emits
`EmbeddedSurfaceExperimentalWarning` on every embedded-surface run, and its
API lives under `vibeqc.periodic_embedding`, not in the top-level `vibeqc`
namespace. The analytic 1-D foundation (semi-infinite tight-binding
substrate) and the 3-D Gaussian pipeline are validated and regression-tested,
but the method has not yet been shown to reproduce a thick-slab
[GDF](density_fitting.md) reference to tight tolerance, so absolute energies
and densities are qualitative. Its numbers are research output, not validated
results, and it is not part of the release gate. The label vocabulary is
defined in [Methods: theory and status](../methods.md#how-to-read-the-status-labels).
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

## Theory

Write the one-electron problem of the whole semi-infinite system in the
Gaussian basis with Hamiltonian $\mathbf H$ and overlap $\mathbf S$, and split
the basis into region I (surface layers plus adsorbate) and the substrate
$S$. The Green's function of region I at complex energy $z$ is obtained by
eliminating the substrate exactly,

$$
\mathbf G_I(z) = \big[\,z\,\mathbf S_{II} - \mathbf H_{II} - \boldsymbol\Sigma_{\mathrm{emb}}(z)\,\big]^{-1},
\qquad
\boldsymbol\Sigma_{\mathrm{emb}}(z) = (\mathbf H_{IS} - z\,\mathbf S_{IS})\,
\mathbf g_S(z)\,(\mathbf H_{SI} - z\,\mathbf S_{SI}),
$$

where $\mathbf g_S(z)$ is the surface Green's function of the substrate with
region I removed. $\boldsymbol\Sigma_{\mathrm{emb}}$ is the embedding
self-energy: energy dependent, non-Hermitian for $\operatorname{Im} z \ne 0$,
and non-local, but supported only on the substrate orbitals that couple to
region I. It is the LCAO form of Inglesfield's embedding potential
({vibeqc-cite}`inglesfield_embedding_1981`), which replaces the
semi-infinite substrate by a boundary term on the dividing surface so that the
region-I wavefunction matches onto the exact bulk continuum instead of onto a
truncated slab. In vibe-qc the potential is built per 2-D surface k-point by
`build_surface_sigma` and recovered as $\boldsymbol\Sigma_{\mathrm{emb}} =
z\,\mathbf 1 - \mathbf H_{00} - \mathbf g_{\mathrm{surf}}^{-1}$ in the
principal-layer basis (`EmbeddingPotential`).

The surface Green's function comes from the decimation of Lopez Sancho,
Lopez Sancho and Rubio ({vibeqc-cite}`lopez_sancho_rubio_1985`,
`sancho_rubio_surface_gf`). With the substrate cut into principal layers
that couple only to their neighbours, on-site block $\mathbf H_{00}$ and
inter-layer block $\mathbf H_{01}$, the recursion

$$
\boldsymbol\alpha_{i+1} = \boldsymbol\alpha_i\,(z - \boldsymbol\epsilon_i)^{-1}\boldsymbol\alpha_i,\quad
\boldsymbol\beta_{i+1} = \boldsymbol\beta_i\,(z - \boldsymbol\epsilon_i)^{-1}\boldsymbol\beta_i,\quad
\boldsymbol\epsilon_{i+1} = \boldsymbol\epsilon_i
+ \boldsymbol\alpha_i(z - \boldsymbol\epsilon_i)^{-1}\boldsymbol\beta_i
+ \boldsymbol\beta_i(z - \boldsymbol\epsilon_i)^{-1}\boldsymbol\alpha_i,\quad
\boldsymbol\epsilon^{s}_{i+1} = \boldsymbol\epsilon^{s}_i
+ \boldsymbol\alpha_i(z - \boldsymbol\epsilon_i)^{-1}\boldsymbol\beta_i,
$$

started from $\boldsymbol\epsilon_0 = \boldsymbol\epsilon^s_0 = \mathbf
H_{00}$, $\boldsymbol\alpha_0 = \mathbf H_{01}$ and $\boldsymbol\beta_0 =
\mathbf H_{01}^\dagger$, doubles the effective stack depth at every step; when
$\boldsymbol\alpha_i, \boldsymbol\beta_i \to 0$ the surface and bulk Green's
functions are $\mathbf g_{\mathrm{surf}} = (z - \boldsymbol\epsilon^s)^{-1}$
and $\mathbf g_{\mathrm{bulk}} = (z - \boldsymbol\epsilon)^{-1}$. A stack of
$2^{20}$ layers is reached in twenty iterations, which is why a converged
semi-infinite substrate costs no more than a few slab-sized inversions per
energy.

Because $\boldsymbol\Sigma_{\mathrm{emb}}$ depends on $z$, the region-I
problem is not a single eigenvalue problem. Ishida's linearisation
({vibeqc-cite}`ishida_embedding_2001`, `EmbeddingPotential.linearize`)
expands it about a reference energy $z_0$ inside the occupied window,

$$
\boldsymbol\Sigma_{\mathrm{emb}}(z) \approx \boldsymbol\Sigma_{\mathrm{emb}}(z_0)
+ (z - z_0)\,\frac{\mathrm d\boldsymbol\Sigma_{\mathrm{emb}}}{\mathrm dz}\Big|_{z_0},
$$

so that the energy-dependent term folds into a modified Hamiltonian
$\mathbf H_{II} + \boldsymbol\Sigma(z_0) - z_0\,\boldsymbol\Sigma'(z_0)$ and
overlap $\mathbf S_{II} - \boldsymbol\Sigma'(z_0)$. The region-I density is
then the contour integral of the Green's function up to the Fermi level,

$$
\mathbf D_I = -\frac{1}{\pi}\operatorname{Im}\int_{E_{\mathrm{bottom}}}^{E_F}
\mathbf G_I(E + i0^{+})\,\mathrm dE ,
$$

evaluated on a semicircle in the upper half plane with Gauss-Legendre nodes
(`EnergyContour`, `contour_n_nodes`), where the integrand is smooth; the
adsorbate step (Layer B) perturbs the clean-surface $\mathbf G_0$ by a
localised $\Delta\mathbf V$ through the Dyson equation $\mathbf G = \mathbf G_0
+ \mathbf G_0\,\Delta\mathbf V\,\mathbf G$ and takes the change in the
integrated density of states and band energy from Lloyd's formula,
$\Delta N(E) = -\tfrac{1}{\pi}\operatorname{Im}\ln\det[\mathbf 1 - \mathbf
G_0(E)\Delta\mathbf V]$.

The approximations that enter are the finite region I (its depth sets how
far the surface perturbation is allowed to reach), the linearisation of
$\boldsymbol\Sigma_{\mathrm{emb}}$ about $z_0$, the numerical contour, and, in
the current code, an embedding potential built once from the substrate
Hamiltonian and held fixed rather than rebuilt from the converged substrate
density (`sigma_scf=True` uses a substrate-SCF mean field for it). The method
helps exactly where a supercell slab fails: dilute adsorbates, with no
slab-thickness quantum-size oscillation and no lateral image interaction. It
fails, today, where absolute numbers are needed: the thick-slab GDF parity is
pending, forces are finite differences of the embedded energy, and the
open-shell route is UHF only.

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
roadmap item; see [the roadmap](../roadmap.md).
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
  [doi:10.1088/0022-3719/14/26/015](https://doi.org/10.1088/0022-3719/14/26/015),
  {vibeqc-cite}`inglesfield_embedding_1981`: the embedding potential.
- H. Ishida, *Phys. Rev. B* **63**, 165409 (2001),
  [doi:10.1103/PhysRevB.63.165409](https://doi.org/10.1103/PhysRevB.63.165409),
  {vibeqc-cite}`ishida_embedding_2001`: energy linearisation of the
  embedding potential in an LCAO basis.
- M. P. López Sancho, J. M. López Sancho and J. Rubio, *J. Phys. F*
  **15**, 851 (1985),
  [doi:10.1088/0305-4608/15/4/009](https://doi.org/10.1088/0305-4608/15/4/009),
  {vibeqc-cite}`lopez_sancho_rubio_1985`: the decimation that yields the
  surface Green's function.
- P. Lloyd, *Proc. Phys. Soc.* **90**, 207 (1967): Lloyd's formula for the
  integrated density of states of a localised perturbation.
