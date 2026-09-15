# Solid-state walkthrough: LiH rocksalt with `pob-TZVP`

A complete v0.5 solid-state calculation: build a real ionic crystal,
run multi-k periodic RHF with the full Ewald Coulomb sum, read off
properties (gap, bands, DOS), and visualize a crystalline orbital.
LiH was the canonical testbed for the pob-TZVP basis (Peintinger,
Vilela Oliveira & Bredow, 2013), so this tutorial doubles as a
calibration check that your install reproduces the published numbers.

**Wall:** ~5-15 min on a laptop, ~2-3 min on a 32-core box.
Pass ``progress=True`` to ``run_rhf_periodic_scf`` (v0.5.1) to
stream per-iteration SCF lines to stdout, without it the script
looks frozen during the multi-minute Fock build. Under
``nohup python my_script.py > log 2>&1 &`` + ``tail -f log`` the
flushed output shows progress unfold in real time.

**Companion script:**
[`examples/periodic/input-lih-pob-tzvp.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-lih-pob-tzvp.py)
reproduces every code block on this page end-to-end.

## Why LiH + pob-TZVP

* **LiH rocksalt** is the simplest 3D ionic crystal: one Li⁺ and one
  H⁻ per primitive (FCC) cell, lattice parameter 4.084 Å, gap of
  ~5 eV. Small enough to converge in minutes, big enough to exercise
  every periodic kernel, multi-k Ewald Coulomb, IBZ symmetry
  reduction, lattice ERI sums.
* **pob-TZVP** is the modern standard solid-state Gaussian basis
  (Peintinger, Vilela Oliveira & Bredow 2013). Designed by removing
  or re-tuning the smallest-exponent primitives in TZVP that cause
  linear dependencies in periodic calculations. LiH was one of the
  test systems in the original paper, H is among the cleanest
  pob-TZVP atoms.

Anything you can do on LiH with this script you can do on any other
3D crystal of similar size, MgO, CaO, NaCl, simple oxides, ionic
fluorides. Replace the lattice + atom block, keep the rest.

## Step 1: Build the cell

Lay out the rocksalt structure as a conventional cubic cell, 4 Li plus 4
H at their fractional sites, and attach the pob-TZVP basis:

```python
import numpy as np
import vibeqc as vq

# Conventional cubic cell: 4 Li + 4 H, 8 atoms total. Easier to
# reason about visually than the 2-atom primitive FCC cell, at the
# cost of a 4× larger SCF.
A_ANG = 4.084                                  # experimental, RT
A = A_ANG / 0.529177210903                     # bohr
lat = A * np.eye(3)

LI_FRAC = [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]
H_FRAC  = [(0.5,0.5,0.5), (0.5,0,0), (0,0.5,0), (0,0,0.5)]

unit_cell = []
for fx, fy, fz in LI_FRAC:
    unit_cell.append(vq.Atom(3, [fx*A, fy*A, fz*A]))    # Li
for fx, fy, fz in H_FRAC:
    unit_cell.append(vq.Atom(1, [fx*A, fy*A, fz*A]))    # H

system = vq.PeriodicSystem(dim=3, lattice=lat, unit_cell=unit_cell)
basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp")
print(f"Basis: pob-TZVP, {basis.nbasis} basis functions per cell")
```

The conventional cell has 8 atoms; pob-TZVP gives 52 basis functions
per cell on this geometry. The primitive FCC cell has 2 atoms (and
~13 basis functions), same chemistry, smaller compute, less obvious
geometry. Pick the conventional cell whenever the visualization
matters, the primitive when wall-time matters.

For a quick "is this a known crystal" sanity check, attach
spglib symmetry and read off the spacegroup:

```python
# doc-audit: skip - continues the system constructed in the preceding cell
vq.attach_symmetry(system, symprec=1e-4)
print(system.symmetry.international_symbol)   # → 'Fm-3m'
print(system.symmetry.number)                  # → 225
```

Rocksalt is Fm-3m (#225, point group O_h). If your build returns
something else, double-check the atom assignment.

## Step 2: SCF with EWALD_3D + IBZ-reduced k-mesh

Run periodic RHF with the unconditionally convergent Ewald Coulomb sum
on a 4×4×4 Monkhorst-Pack mesh that spglib reduces to the irreducible
Brillouin zone:

```python
# doc-audit: skip - continues the system and basis constructed above
opts = vq.PeriodicSCFOptions()
opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
opts.lattice_opts.cutoff_bohr = 12.0
opts.lattice_opts.nuclear_cutoff_bohr = 25.0
opts.conv_tol_energy = 1e-8
opts.max_iter = 80

# 4×4×4 Monkhorst-Pack mesh, reduced to the IBZ via spglib.
# Fm-3m at this mesh has only 4 irreducible k-points; the SCF is
# 4× cheaper than the naïve 64-point full mesh.
kpts = vq.KPoints.monkhorst_pack(system, [4, 4, 4], symmetry=True)
print(f"k-mesh: 4×4×4 → {len(kpts)} IBZ points")

result = vq.run_rhf_periodic_scf(system, basis, kpts, opts, progress=True)
print(f"E/cell    = {result.energy:.8f} Ha")
print(f"Iters     = {result.n_iter} ({'converged' if result.converged else 'NOT'})")
```

`progress=True` (v0.5.1) streams a flushed banner + per-stage
timings + per-iteration SCF lines to stdout while the SCF runs.
Critical for any multi-minute job: without it the script looks
frozen between the "running SCF..." print and the next print. Set
``VIBEQC_LIVE_LOGGING=0`` in the environment to silence it for
batch scripts.

`CoulombMethod.EWALD_3D` is the right default for any 3D bulk: the
direct-truncated Coulomb sum is conditionally convergent (see
[Madelung constants via Ewald summation](madelung_with_ewald.md)) and
gives shape-dependent answers. The Ewald path is unconditionally
convergent and routes through the same dispatcher
(`run_rhf_periodic_scf`).

The `symmetry=True` flag wires through spglib's IBZ reduction (Phase
K2). For high-symmetry crystals like rocksalt this is a 16× saving
on a 4×4×4 mesh; for low-symmetry P1 cells it's a no-op.

```{tip}
**Doesn't converge?** Ionic insulators like LiH / NaCl / MgO with
deep cores are exactly the regime where the default Hcore initial
guess is furthest from the converged density. Try ``opts.initial_guess
= vq.InitialGuess.SAD`` (v0.6.1), superposition of atomic densities
is much closer to the rocksalt picture and routinely rescues these
cases. The [`examples/debug/scf_sad_vs_hcore.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/debug/scf_sad_vs_hcore.py)
script A/B-tests this on LiH / NaCl / MgO directly. For a deeper
forensic dive, see [`examples/debug/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/debug)
and [Periodic SCF convergence: damping, DIIS, and level shifts](periodic_scf_convergence.md).
```

## Step 3: Read the fundamental gap

vibe-qc doesn't expose per-site Mulliken charges for periodic systems
yet (deferred to post-v0.6; periodic Mulliken implementation is on
the backlog and not gated on G1). The per-k orbital energies are
right there in `result.mo_energies` and give you the HOMO-LUMO gap
for free:

```python
# doc-audit: skip - continues the converged periodic result from Step 2
n_occ = sum(int(a.Z) for a in system.unit_cell) // 2
homo = max(float(eps[n_occ - 1]) for eps in result.mo_energies)
lumo = min(float(eps[n_occ])     for eps in result.mo_energies)
print(f"HOMO max  = {homo:.4f} Ha   ({homo * 27.2114:.3f} eV)")
print(f"LUMO min  = {lumo:.4f} Ha   ({lumo * 27.2114:.3f} eV)")
print(f"Gap       = {(lumo - homo) * 27.2114:.3f} eV")
```

For LiH at HF/pob-TZVP with this k-mesh, expect a gap around
8-9 eV. **HF systematically over-estimates band gaps** by roughly
3 eV in ionic crystals (no electron correlation in the conduction
band). Experimental optical gap is ~5 eV; PBE/pob-TZVP gives ~3 eV
(DFT under-estimates by a similar amount); HSE06 and beyond give
the right answer. The number HF gives you is *consistent*, 
useful for trends across systems, not for matching a single
experimental gap.

## Step 4: Band structure along the standard HPKOT path

Sample the Hcore eigenvalues along the canonical high-symmetry k-path
for an Fm-3m crystal, which seekpath derives automatically from the
Bravais lattice:

```python
# doc-audit: skip - continues the system, basis, and occupancy from earlier steps
# Auto-detect Bravais → seekpath returns the canonical Hinuma 2017
# k-path. For Fm-3m: Γ-X-W-K-Γ-L-U-W-L-K|U-X.
band_kp = vq.KPoints.band_path(system)
print(f"Path: {' → '.join(label for _, label in band_kp.labels)}")

bands = vq.band_structure_hcore(
    system, basis, band_kp.to_kpath(),
    n_electrons_per_cell=2 * n_occ,
)
```

These are **Hcore bands**, the non-interacting eigenvalues of T+V along
the path. They exercise the reciprocal-space path and Bloch
diagonalization, but they are not an approximation to the converged HF
bands in a controlled sense. Electron-electron terms can change band
ordering, dispersion, extrema, and topology, not merely shift every
eigenvalue. Do not quote an Hcore gap or compare this plot with
experiment. For scientific bands, choose the post-SCF helper for the
calculation route in
[Band structure and DOS](../user_guide/band_structure.md#programmatic-scf-routes).

```{note}
The path `vq.KPoints.band_path(system)` returns is the
**HPKOT (Hinuma 2017)** standard, the same convention used by
Materials Project, AFLOW, and seekpath itself. Cross-checking
your bands against published reference data is straightforward
because the path is fixed by Bravais type.
```

## Step 5: DOS + atom-projected DOS

The total and atom-projected density of states are sampled on a 4×4×4
mesh with Gaussian broadening, decomposing the spectrum into Li and H
contributions per angular momentum:

```python
# doc-audit: skip - continues the system, basis, and occupancy from earlier steps
dos = vq.density_of_states_hcore(
    system, basis, [4, 4, 4],
    sigma=0.01, n_electrons_per_cell=2 * n_occ,
)

# atom + l projection: separates Li-s, H-s, plus any p contribution
# at higher energies.
pdos = vq.density_of_states_projected_hcore(
    system, basis, [4, 4, 4],
    projection="atoms_l",            # or "atoms" for atom-only
    sigma=0.01,
    n_electrons_per_cell=2 * n_occ,
)
```

Within this Hcore teaching model, the low occupied states have strong
H-s character and the low virtual states gain Li-s/Li-p character.
That is useful for learning how to read the projection labels, but it
is not a self-consistent assignment of the LiH valence and conduction
bands. Confirm any chemical interpretation with a converged SCF PDOS.

## Step 6: HOMO Bloch orbital cube

Write the HOMO Bloch orbital at Γ to a cube file, tiled across a 2×2×2
supercell so the crystalline orbital is ready to open in VESTA or
XCrySDen:

```python
# doc-audit: skip - continues the k-point mesh and SCF result from earlier steps
# Find Γ in the IBZ list; typically index 0, but verify.
k_cart = np.asarray(kpts.kpoints_cart)
gamma_idx = int(np.argmin(np.linalg.norm(k_cart, axis=1)))

# 2×2×2 supercell tile of the orbital, ready to open in VESTA / XCrySDen.
vq.write_cube_mo_periodic(
    "lih-homo-gamma.cube",
    system, basis,
    np.asarray(result.mo_coeffs[gamma_idx]),
    k_cart[gamma_idx],
    n_occ - 1,                              # HOMO band index
    n_replica=(2, 2, 2),
    spacing_bohr=0.25,
    title="LiH HOMO @ Γ (RHF/pob-TZVP)",
)
```

Open the cube with **VESTA** (`vesta lih-homo-gamma.cube`) or
**XCrySDen** (`xcrysden --cube lih-homo-gamma.cube`). At Γ the
orbital is real and the isosurface shows H 1s spheres on every
H site, slightly delocalized into the Li 2s neighbors via small
amplitude tails. That's the picture: H⁻ holds the electron, with a
~10% admixture of Li 2s, the "ionic with covalent character"
description that pob-TZVP is designed to capture.

For non-Γ k-points the orbital is complex; pass
`component="real"`, `"imag"`, `"abs"`, or `"abs2"` to
`write_cube_mo_periodic` to control which scalar field gets
written. The default is `"real"`.

```{tip}
**Why a cube and not the ``moltui`` terminal viewer?** moltui
([Viewing geometries, orbitals, and vibrations with MolTUI](moltui_terminal_viewer.md)) is a great molecular
viewer but doesn't yet handle periodic-cell metadata, so the cube
file's lattice vectors, the BZ, image-atom replication. Open
discussion with the upstream developer to extend it for solids;
in the meantime VESTA / XCrySDen / VMD are the right tools for
crystalline orbitals.
```

## Theory

The sections below unpack the four ingredients the script leans on: the
Bloch construction in a Gaussian basis, why pob-TZVP exists, why the 3D
Coulomb sum needs Ewald, and how IBZ reduction buys back the cost.

### Bloch's theorem in a Gaussian basis

For a translationally invariant Hamiltonian, eigenstates are labeled
by a band index $n$ and a crystal momentum $\mathbf{k}$ in the first
Brillouin zone:

$$
\psi_{n\mathbf{k}}(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}}\,u_{n\mathbf{k}}(\mathbf{r}),
\qquad
u_{n\mathbf{k}}(\mathbf{r} + \mathbf{T}) = u_{n\mathbf{k}}(\mathbf{r}).
$$

Each AO $\chi_\mu$ in the reference cell is replaced by a
Bloch-summed combination:

$$
\chi_{\mu\mathbf{k}}(\mathbf{r}) = \sum_\mathbf{T} e^{i\mathbf{k}\cdot\mathbf{T}}\,\chi_\mu(\mathbf{r}-\mathbf{T}).
$$

The Roothaan eigenvalue problem becomes block-diagonal in $\mathbf{k}$:

$$
\mathbf{F}(\mathbf{k})\,\mathbf{C}(\mathbf{k}) = \mathbf{S}(\mathbf{k})\,\mathbf{C}(\mathbf{k})\,\boldsymbol{\varepsilon}(\mathbf{k}),
$$

with each k-space matrix the discrete Fourier transform of the
real-space lattice-summed matrix
$\mathbf{F}(\mathbf{k}) = \sum_\mathbf{T} e^{i\mathbf{k}\cdot\mathbf{T}}\mathbf{F}(\mathbf{T})$.
This is the "build in real space, diagonalize in k-space" pattern
that CRYSTAL pioneered for Gaussian crystalline orbitals
([Periodic HF on a 1D H chain](periodic_hf.md) covers it in detail).

### Why pob-TZVP

Standard molecular Gaussian basis sets (TZVP, def2-TZVP,
cc-pVTZ, …) include a long tail of small-exponent primitives that
extend deep into neighboring cells and produce a near-singular
overlap matrix when summed across the lattice. Stripping those
primitives (or replacing them with sharper alternatives) gives
**pob-\***, the "Periodic, Optimized, Bredow" family. Three variants
ship today:

| Basis | Source | Strengths |
|---|---|---|
| pob-DZVP | Bredow group, 2013 | Cheapest; minimal redundancy |
| **pob-TZVP** | PVO-Bredow, 2013 | Standard production basis; this tutorial |
| pob-TZVP-rev2 | PVO-Bredow, 2018 | Refined valence space; pairs well with PBE |
| pob-DZVP-rev2 | PVO-Bredow, 2018 | Cheaper rev2 variant |

For more on the design, see [Why solid-state calculations use `pob-TZVP`](pob_tzvp.md).

### Coulomb sum convergence in 3D

The per-cell electron-electron Coulomb sum
$\sum_\mathbf{T} 1/|\mathbf{r}-\mathbf{T}|$ is **conditionally
convergent**: in 3D it sums to different values depending on the
shape over which you truncate (sphere, cube, slab). Direct truncation
(`CoulombMethod.DIRECT_TRUNCATED`, vibe-qc's default for
back-compat) gives shape-dependent energies and 0.1-1 Ha per-cell
errors on bulk. The fix is **Ewald summation**, split each $1/r$
into an erfc-screened short-range piece (real-space, exponential
convergence) plus a Gaussian-charge long-range piece (FFT-based
reciprocal-space convolution, exponential convergence too). vibe-qc's
EWALD_3D path is validated to ω-invariance at $10^{-8}$ Ha across
H₂ / LiH / MgO / Ne reference cells, see
[user_guide/ewald.md](../user_guide/ewald.md).

### IBZ reduction and the speed-up

For an MP mesh of $N$ k-points and a system with point-group order
$|G|$, the irreducible Brillouin zone contains roughly $N/|G|$
distinct k-points (modulo time-reversal symmetry). Per-iteration
SCF cost scales as $O(N_\text{IBZ} \cdot N_\text{bf}^4)$, for LiH
at 4×4×4 with $|G|=48$, that's a **16× saving** over the full mesh.

Hex / trigonal cells have a subtle trap: the classical
Monkhorst-Pack convention shifts even meshes by half a step, which
breaks the three-fold rotational symmetry and inflates the IBZ
count for no accuracy gain. vibe-qc's K2 phase refuses non-zero
shifts on spacegroups 143-194 with an actionable error pointing
at `KPoints.gamma_centred(...)`. For Fm-3m (this tutorial) the
classical shift is fine and gets applied automatically.

## Next

* **Phonons + thermochemistry of LiH.** Needs Phase 21 (periodic
  phonons via finite-difference on top of Phase G1 periodic
  gradients). G1a-e (the gradient stack) shipped in v0.6; periodic
  phonons remain queued for a later v0.6.x or v0.7.
* **Full SCF bands** instead of Hcore bands. Use a helper documented for
  the selected SCF route; generic periodic results do not all expose the
  same real-space Fock representation. The current route table is in
  [Band structure and DOS](../user_guide/band_structure.md#programmatic-scf-routes).
* **DFT bands** with PBE / B3LYP. First converge the intended RKS route,
  then use a post-SCF helper that evaluates that same Kohn-Sham operator.
  Converge the k mesh and state the energy reference; do not infer gap
  accuracy from the functional name alone.
* **Other ionic crystals.** MgO is the next standard pob-TZVP
  testbed; CaO, NaCl, KCl follow.

## References

* **Original pob-TZVP paper.** M. F. Peintinger, D. Vilela Oliveira,
  T. Bredow, "Consistent Gaussian basis sets of triple-zeta valence
  with polarization quality for solid-state calculations,"
  *J. Comp. Chem.* **34**, 451 (2013).
* **HPKOT band-path standardization.** Y. Hinuma, G. Pizzi, Y. Kumagai,
  F. Oba, I. Tanaka, "Band structure diagram paths based on
  crystallography," *Comp. Mat. Sci.* **128**, 140 (2017).
  Implemented via [seekpath](https://seekpath.readthedocs.io/).
* **Gaussian-density 3D electrostatics.** V. R. Saunders et al.,
  "On the electrostatic potential in crystalline systems where the
  charge density is expanded in Gaussian functions," *Mol. Phys.*
  **77**, 629 (1992). This supplies electrostatic-potential and penetration
  context; it is not the source of the quartet dispatcher.
* **Monkhorst-Pack k-sampling.** H. J. Monkhorst, J. D. Pack,
  "Special points for Brillouin-zone integrations," *Phys. Rev. B*
  **13**, 5188 (1976).
* **CRYSTAL-style Gaussian crystalline orbitals.** C. Pisani,
  R. Dovesi, C. Roetti, *Hartree-Fock Ab Initio Treatment of
  Crystalline Systems*, Lecture Notes in Chemistry **48**,
  Springer (1988).

## See also

* [Periodic HF on a 1D H chain](periodic_hf.md), 
  Bloch-sum machinery, simpler 1D system
* [Periodic KS-DFT](periodic_dft.md), same
  machinery for DFT functionals
* [Madelung constants via Ewald summation](madelung_with_ewald.md)
  - what conditional convergence in 3D actually means
* [Band structure and density of states](band_structure.md), Hcore
  bands on the H-chain
* [Why solid-state calculations use `pob-TZVP`](pob_tzvp.md), basis design
* [User guide, linear dependence in periodic SCF](../user_guide/linear_dependence.md)
  - what to do if a tighter cell or a non-pob basis trips the
  v0.7 critical-overlap preflight on this system
* [Projected density of states (PDOS)](pdos.md), PDOS in detail
* [Periodic orbital cubes and XSF files](periodic_orbital_cubes.md)
  - the Bloch-orbital cube path in depth
* [User guide, Ewald](../user_guide/ewald.md), the EWALD_3D
  dispatch
* [User guide, k-points](../user_guide/k_points.md), the
  ``KPoints`` builder API
* [User guide, Crystal lattices](../user_guide/crystal_lattices.md)
  - Bravais types + WS cells + BZ shapes
