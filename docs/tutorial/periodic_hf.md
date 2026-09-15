# Periodic HF on a 1D H chain

The simplest non-trivial periodic system: a one-dimensional chain of
H₂ dimers.

```python
import numpy as np
import vibeqc as vq
from vibeqc import (
    Atom, BasisSet, KPoints, PeriodicSystem, run_periodic_job,
)

# Unit cell: H2 oriented along the chain, 4.0 bohr between unit cells.
unit_cell = [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]
lattice = np.diag([4.0, 30.0, 30.0])   # 1D axis on x; y, z are vacuum

sysp = PeriodicSystem(
    dim=1,
    lattice=lattice,
    unit_cell=unit_cell,
)

basis = BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")
kpoints = KPoints.monkhorst_pack(sysp, [6, 1, 1])

result = run_periodic_job(
    sysp,
    basis,
    method="RHF",
    jk_method="gdf",       # maintained 1D Coulomb route
    kpoints=kpoints,
    conv_tol_energy=1e-10,
)
print(f"E_RHF per cell = {result.energy:.10f}  ({result.n_iter} iters)")
```

``run_periodic_job`` is the recommended user-facing dispatcher. For 1D
wires, ``jk_method="auto"`` also selects GDF; spelling it explicitly here
makes the Coulomb objective reproducible in the example. The historical
``run_rhf_periodic_scf`` direct-truncated branch is retained only for
vacuum-padded molecular-limit diagnostics, not quantitative 1D solids.
For a Gamma-only calculation, pass ``KPoints.gamma(sysp)``.

## Other ways to build a k-mesh, `vibeqc.KPoints`

The `KPoints` builder fronts every common k-point input mode in a
single API. The five most useful for everyday work:

```python
# 1. Monkhorst-Pack (classical convention; auto-shifts even meshes by ½)
kp = KPoints.monkhorst_pack(sysp, [6, 1, 1])

# 2. Γ-centred: required for hex/trigonal cells (the classical MP shift
#    breaks 3-fold symmetry); the MP constructor will refuse non-zero
#    shifts on those spacegroups with an actionable error.
kp = KPoints.gamma_centred(sysp, [6, 6, 6])

# 3. Symmetry-reduced to the irreducible BZ (via spglib).
#    Requires `vq.attach_symmetry(sysp)` first.
vq.attach_symmetry(sysp)
kp = KPoints.monkhorst_pack(sysp, [4, 4, 4], symmetry=True)
# Equivalent two-step form: kp.symmetry_reduce()

# 4. Density-based auto-mesh (KPPRA / kspacing / VASP `Auto`).
#    Picks the mesh from a target k-point density rather than from
#    a fixed [N, N, N] shape.
kp = KPoints.from_kppra(sysp, n_kpts_per_atom=1000)  # AFLOW / Curtarolo
kp = KPoints.from_kspacing(sysp, kspacing=0.04)      # Materials Project / ASE
kp = KPoints.auto(sysp, length=40)                   # VASP Auto

# 5. Band path: auto-detects Bravais lattice via spglib, returns
#    the canonical Hinuma 2017 (HPKOT) k-path via seekpath.
band = KPoints.band_path(sysp)
print(' → '.join(label for _, label in band.labels))
```

The legacy free-function ``vq.monkhorst_pack(sysp, [N, M, L])`` is
still supported and returns a raw ``BlochKMesh``, the periodic
SCF entry points accept both. For new code, prefer ``KPoints``:
it carries the symmetry-reduction state, supports band paths, and
gives an actionable error on the hex/trigonal MP-shift trap.

End-to-end demo across every mode (Γ-only, MP scan, IBZ reduction,
KPPRA scan, HPKOT band path) on cubic Mg:
[`examples/periodic/input-k-mesh-convergence.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-k-mesh-convergence.py).
The full reference is [user_guide/k_points.md](../user_guide/k_points.md).

## What the options mean

* `PeriodicSystem.dim`, number of periodic axes. `dim=1` makes only
  `lattice.col(0)` a lattice vector; the other two columns are
  interpreted as vacuum with enough separation (30 bohr here) that
  images don't overlap. `dim=2` treats cols 0 and 1 as in-plane
  lattice vectors; col 2 is synthesized bookkeeping, not a vacuum gap,
  and the slab energy is invariant to it (build these with
  `vq.slab_2d`). `dim=3` is full bulk.
* `lattice_opts.cutoff_bohr`, the real-space distance beyond which
  (μν) pair integrals are dropped from the lattice sum. Bump it up
  until your energy stops changing.
* `lattice_opts.nuclear_cutoff_bohr`, separate, usually larger,
  cutoff for the nuclear-attraction term (1/r tail decays more slowly
  than AO overlap).

## Choosing a k-mesh

For the SCF to represent an insulator correctly, the mesh needs to
sample enough of the Brillouin zone that the occupied manifold stays
separated from the virtual. Start with `[N, 1, 1]` for 1D, `[N, N, 1]`
for 2D, `[N, N, N]` for 3D, and increase `N` until the energy per cell
stops changing. For small H-chain tests a 4×1×1 mesh is usually
plenty; for covalent 3D crystals in TZ basis, 4×4×4 is a reasonable
starting point.

## What's validated today

* **Molecular limit**: a big-box (50 bohr) unit cell produces the same
  energy as molecular RHF, machine precision across dim ∈ {1, 2, 3}
  and multiple k-mesh sizes.
* **Bloch-sum machinery**: kinetic-only folding equivalence between
  1-cell × K k-points and K-cell × Γ, machine precision.
* **Non-trivial 1D SCF convergence** at tight cell spacings.

3D bulk calculations with the direct-truncated Coulomb sum are
cutoff-dependent (the infinite Coulomb series is conditionally
convergent in 3D). For quantitative bulk results set
``opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D``,
the dispatcher then routes to vibe-qc's full-metric Ewald backend. The
supported route uses exact erfc-screened real-space ERIs plus reciprocal
long-range Coulomb; it does not activate the fail-closed quartet multipole
prototype. See [the Ewald user
guide](../user_guide/ewald.md) for details on the full-metric FFT
long-range solver and the FFTDF/GDF parity track.

```{tip}
**SCF won't converge?** [`examples/debug/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/debug)
ships a kit of focused diagnostic scripts (Ne / Ar / diamond C
known-good baselines, trajectory recorder, convergence-aid sweep,
PySCF cross-check). [Periodic SCF convergence: damping, DIIS, and level shifts](periodic_scf_convergence.md)
walks through the level-shift / DIIS / SAD-guess aids the SCF
exposes. Every dispatcher accepts ``progress=True`` (v0.5.1) for
live per-iter logging.
```

## Theory

The sections below build up the periodic-HF machinery the code runs: Bloch's theorem and the Bloch-summed AO basis, the k-diagonal generalised eigenvalue problem it produces, Monkhorst-Pack sampling of the Brillouin zone, the Γ-only and molecular-limit checks, and the conditionally convergent 3D Coulomb sum that motivates the Ewald backend.

### Bloch's theorem

For a Hamiltonian that commutes with every translation by a
lattice vector $\mathbf{T}$, the eigenfunctions can be labeled by
a crystal momentum $\mathbf{k}$ in the first Brillouin zone and a
band index $n$:

$$
\psi_{n\mathbf{k}}(\mathbf{r}) = e^{i \mathbf{k} \cdot \mathbf{r}} \, u_{n\mathbf{k}}(\mathbf{r}),
\qquad
u_{n\mathbf{k}}(\mathbf{r} + \mathbf{T}) = u_{n\mathbf{k}}(\mathbf{r}).
$$

The lattice-periodic function $u_{n\mathbf{k}}$ is expanded in a
*Bloch-summed* basis: take each atom-centered Gaussian $\chi_\mu$ in
the reference unit cell and form the infinite symmetry-adapted
combination

$$
\chi_{\mu\mathbf{k}}(\mathbf{r}) = \sum_{\mathbf{T}} e^{i \mathbf{k} \cdot \mathbf{T}} \, \chi_\mu(\mathbf{r} - \mathbf{T}).
$$

These Bloch-summed AOs are the natural basis for crystalline
orbitals in the Roothaan-style framework (CRYSTAL, vibe-qc's
periodic driver, QE's hybrid-DFT implementation all do this).

### The generalised eigenvalue problem at every k

Plug the Bloch-summed basis into the HF variational equation and
the resulting matrices are **$\mathbf{k}$-diagonal**: for each
$\mathbf{k}$ in the sampled mesh, you get an independent
generalised eigenvalue problem

$$
\mathbf{F}(\mathbf{k}) \, \mathbf{C}(\mathbf{k}) = \mathbf{S}(\mathbf{k}) \, \mathbf{C}(\mathbf{k}) \, \boldsymbol{\varepsilon}(\mathbf{k}),
$$

where each k-space matrix is a **Fourier transform** of the real-
space lattice-summed matrix:

$$
\mathbf{F}(\mathbf{k}) = \sum_{\mathbf{T}} e^{i \mathbf{k} \cdot \mathbf{T}} \, \mathbf{F}(\mathbf{T}),
\qquad
\mathbf{F}(\mathbf{T})_{\mu\nu} = \int \chi_\mu(\mathbf{r}) \, \hat{F} \, \chi_\nu(\mathbf{r} - \mathbf{T}) \, \mathrm{d}^3 r.
$$

The SCF loop alternates between two representations: build the
Fock contributions in real space (lattice-summed one- and
two-electron integrals), transform to $\mathbf{k}$-space, diagonalize
per $\mathbf{k}$, inverse-transform the density back to real space.
This "real-space build, $\mathbf{k}$-space diagonalize" pattern is
what CRYSTAL pioneered for Gaussian crystalline orbitals.

### Monkhorst-Pack k-sampling

Integrals over the Brillouin zone, including the density

$$
\rho(\mathbf{r}) = \sum_n^{\text{occ}} \int_\text{BZ} |\psi_{n\mathbf{k}}(\mathbf{r})|^2 \, \frac{\mathrm{d}^3 k}{(2\pi)^3}
$$

- are replaced by weighted sums over a discrete mesh. The
**Monkhorst-Pack scheme** constructs a uniform grid of k-points
offset from the BZ origin, optionally reduced to the irreducible
wedge by the space-group operations. `monkhorst_pack(sysp, [N, M, L])`
builds the grid with $N \times M \times L$ points along each
reciprocal-lattice direction.

For insulators, convergence with $N$ is exponential, the error
shrinks as $e^{-\alpha N}$ because the occupied bands are well
separated from the virtual ones. For metals, convergence degrades
to a power law because the Fermi surface cuts bands; smearing or
tetrahedron methods (not yet in vibe-qc) accelerate convergence
there.

### The Γ-only limit and molecular-limit validation

Setting the mesh to `[1, 1, 1]` samples only $\mathbf{k} = \mathbf{0}$
(the Γ point), equivalent to using purely real Bloch sums. This is
the cheapest periodic calculation and the right default for a large
unit cell where a denser mesh doesn't change anything.

The **molecular-limit test**, make the cell so big that periodic
images don't interact, then the per-cell energy must equal the
isolated-molecule HF energy, is vibe-qc's cleanest correctness
check. Run it at every k-mesh size: the per-cell energy must be
independent of the mesh (because in the molecular limit, no
$\mathbf{k}$-dependent structure exists to resolve). vibe-qc's
regression suite asserts this at machine precision across dim ∈ {1,
2, 3}.

### The Coulomb-sum problem in 3D

The last term of the Fock matrix, the Coulomb contribution from
the infinite lattice of nuclei and electrons, is where periodic
HF gets hard in 3D. The per-unit-cell sum

$$
\sum_{\mathbf{T}} \frac{1}{|\mathbf{r} - \mathbf{T}|}
$$

is **conditionally convergent**: direct truncation gives shape-
dependent answers (see [Madelung constants via Ewald summation](madelung_with_ewald.md) for
the full story). Mixed-boundary 1D and 2D systems also require
dimension-specific Coulomb kernels. The public routes use GDF for 1D and
``SLAB_EWALD_2D`` (or explicit slab GDF) for 2D. ``DIRECT_TRUNCATED`` is
only a vacuum-padded molecular-limit diagnostic. For 3D, set
``opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D``;
the dispatcher routes to vibe-qc's full-metric Ewald backend, with exact
real-space ERIs rather than the fail-closed quartet multipole prototype
(validated to ω-invariance to 1e-8 Ha across H₂ /
LiH / MgO / Ne in the bulk benchmark suite). The native FFT
long-range solver supports skew-cell metrics; FFTDF/GDF parity is
the active production path for tight 3D ionic crystals.

For the full menu of periodic Coulomb/exchange routes that tame this sum
(BIPOLE, GDF, GPW / GAPW) and which external program validates each, see
[periodic-SCF methods](../user_guide/periodic_methods.md) and the
[periodic JK routes & parity policy](../periodic_jk_routes.md).

## Resources

~150 MB peak RAM, ~2 s on one core (Apple M2 baseline) for the
H₂-chain pob-TZVP / 6-k example. Cost scales linearly in the
k-mesh size and as $\mathcal{O}(N_\text{bf}^4)$ in the per-cell
basis. EWALD_3D adds roughly 10-30% wall time for the FFT-based
long-range piece on top of the direct-truncated baseline.

## References

- **Bloch's theorem.** F. Bloch, "Über die Quantenmechanik der
  Elektronen in Kristallgittern," *Z. Phys.* **52**, 555 (1929).
- **CRYSTAL-style Gaussian crystalline orbitals.** C. Pisani and
  R. Dovesi, "Exact-exchange Hartree-Fock calculations for periodic
  systems. I. Illustration of the method," *Int. J. Quantum Chem.*
  **17**, 501 (1980); C. Pisani, R. Dovesi, and C. Roetti,
  *Hartree-Fock Ab Initio Treatment of Crystalline Systems*,
  Lecture Notes in Chemistry **48**, Springer (1988).
- **Monkhorst-Pack k-mesh.** H. J. Monkhorst and J. D. Pack,
  "Special points for Brillouin-zone integrations," *Phys. Rev. B*
  **13**, 5188 (1976).
- **Textbook, crystalline electronic structure.** N. W. Ashcroft
  and N. D. Mermin, *Solid State Physics*, Cengage Learning reprint
  (1976), chapter 8 for Bloch's theorem, chapter 10 for the
  tight-binding formulation that most closely matches Gaussian
  crystalline orbitals.
- **Gaussian-density 3D electrostatics.** V. R. Saunders, C.
  Freyria-Fava, R. Dovesi, L. Salasco, and C. Roetti, "On the
  electrostatic potential in crystalline systems where the charge
  density is expanded in Gaussian functions," *Mol. Phys.* **77**,
  629 (1992). This supplies electrostatic-potential, penetration, and
  spheropole context, not the implemented quartet dispatcher.
- **Periodic bipolar quartet expansion.** C. Pisani, R. Dovesi, and
  C. Roetti, *Hartree-Fock Ab Initio Treatment of Crystalline Systems*,
  Lecture Notes in Chemistry 48, Springer (1988), Chapter II.4c.
  [DOI 10.1007/978-3-642-93385-1](https://doi.org/10.1007/978-3-642-93385-1).
