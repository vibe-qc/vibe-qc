# XSF and BXSF: periodic volumetric visualisation

Visualise crystal structures, electron densities, Bloch orbitals,
and Fermi surfaces with XCrySDen's native formats, directly from
vibe-qc.

**Native path:** vibe-qc's own visualization format for periodic
results is QVF + vibe-view. `run_periodic_job(..., output_qvf=True)`
writes a `.qvf` archive (structure, density, and bands / DOS where
computed) that [vibe-view](../user_guide/vibe_view.md) renders
directly; see [The QVF file format, end to end](qvf_file_format.md) and the
[QVF tech spec](../design_qvf_format.md). This tutorial covers the
**XSF / BXSF interchange** formats, for XCrySDen, VESTA, and moltui.

**You'll learn:** how to write XSF files for crystal structures,
electron densities, and single Bloch orbitals; how to write BXSF
files for Fermi-surface visualisation; and how to open them in
moltui, VESTA, and XCrySDen.

**Why:** XSF is the de-facto format for periodic volumetric data.
Unlike Gaussian cube, which has no concept of a lattice, XSF
stores the unit-cell vectors and tells the viewer to tile the cell.
The result is correct periodicity with exactly one set of voxels,
no supercell hacks. BXSF extends this to reciprocal space so
XCrySDen can render Fermi surfaces directly.

**Prerequisites:** a working vibe-qc install; optional
[moltui](https://github.com/kszenes/moltui) for terminal viewing.

```sh
# Install moltui as a vibe-qc extra:
.venv/bin/pip install '.[viewer]'
```

## The three verbs: structure, volumetric, bands

vibe-qc ships four writers in the XSF family:

| Writer | What it writes | Best viewer |
|---|---|---|
| ``write_xsf_structure`` | Structure-only XSF (PRIMVEC + PRIMCOORD) | VESTA, XCrySDen, moltui |
| ``write_xsf_mo`` | One Bloch orbital on the primitive cell | VESTA, XCrySDen |
| ``write_xsf_density`` | Total SCF density on the primitive cell | VESTA, XCrySDen, moltui |
| ``write_bxsf`` | Band energies on a regular k-mesh (Fermi surface) | XCrySDen, moltui |

The first three work in real space; the last one works in
**reciprocal** (k) space.  All are callable as free functions from
``vibeqc``; ``write_xsf_density`` is also wired into
``run_periodic_job(write_density=True)`` for one-line access.

```{tip}
For *molecular* volumetric output — cube files, molden, orbital
visualisation — see [Orbital and density visualization](orbital_visualization.md) and the
[volumetric data user guide](../user_guide/volumetric_data.md).
```

## 1. XSF crystal structure

The simplest XSF file carries only the crystal, lattice vectors and
atom positions, with no volumetric data.  Useful as a quick visual
check on your ``PeriodicSystem`` before running the SCF.

```python
import vibeqc as vq
import numpy as np

# NaCl rocksalt primitive cell. Lattice vectors are columns.
a = 5.640 / 0.529177210903  # angstrom to bohr
a1 = np.array([0.0, a / 2, a / 2])
a2 = np.array([a / 2, 0.0, a / 2])
a3 = np.array([a / 2, a / 2, 0.0])
lattice = np.column_stack((a1, a2, a3))
cl_position = 0.5 * (a1 + a2 + a3)
system = vq.PeriodicSystem(
    3,
    lattice,
    [vq.Atom(11, [0.0, 0.0, 0.0]), vq.Atom(17, cl_position)],
)

vq.write_xsf_structure("nacl_primitive.xsf", system)
# → nacl_primitive.xsf — open in VESTA, XCrySDen, or moltui
```

The file contains a ``CRYSTAL`` block with ``PRIMVEC`` (lattice
vectors in ångström) and ``PRIMCOORD`` (atom positions in ångström).
vibe-qc converts from bohr automatically; XSF's native unit is
ångström.  A 2D slab opens with ``SLAB`` instead, and a 1D polymer
with ``POLYMER``; see [XSF format conventions](#xsf-format-conventions).

For the one-call workflow, ``run_periodic_job`` writes a
``{stem}.xsf`` alongside the other output files:

```python
# doc-audit: skip - continues the periodic system constructed above
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

result = vq.run_periodic_job(
    system, basis,
    method="RHF",
    output="nacl",
    write_xsf_structure_file=True,   # default: True
    write_density=False,
)
# → nacl.out, nacl.system, nacl.molden, nacl.xsf (.structure.xsf
#   when write_density=True), nacl.POSCAR, nacl.cif
```

When ``write_density=True`` (see next section), the density lands
in ``{stem}.xsf`` and the structure-only XSF is routed to
``{stem}.structure.xsf`` to avoid a suffix collision.

## 2. XSF density

Pass ``write_density=True`` to ``run_periodic_job`` and the job emits
a ``{stem}.xsf`` with the SCF total electron density sampled on the
primitive cell:

```python
# doc-audit: skip - continues the periodic system and basis constructed above
vq.run_periodic_job(
    system, basis,
    method="RHF",
    output="nacl",
    write_density=True,
    density_spacing_bohr=0.2,        # default
)
# → nacl.xsf — CRYSTAL block + DATAGRID_3D_density
```

The density ρ(r) = Σ D_{μν} χ_μ(r) χ_ν(r) is evaluated on a regular
fractional-coordinate grid spanning one primitive cell.  The
``BEGIN_DATAGRID_3D_density`` block holds the voxel values; VESTA
renders them as a scalar isosurface, moltui as a raycast isosurface.

When both ``write_density=True`` and ``write_xsf_structure_file=True``,
the density lands in ``{stem}.xsf`` and the structure-only file is
routed to ``{stem}.structure.xsf`` to avoid a suffix collision.

The standalone `vq.write_xsf_density` is a lower-level API for a
workflow that already owns an explicit real-space `LatticeMatrixSet`
density. Do not assume that a generic periodic result has a
`density_real` attribute. The high-level call above performs the
route-specific density conversion and is the supported default.

## 3. XSF Bloch orbitals

To visualise a single crystalline (Bloch) orbital, 
ψ_{n,k}(r), on the primitive cell, use ``write_xsf_mo``.  This is
the periodic counterpart of ``write_cube_mo`` for molecules.

```python
# doc-audit: skip - continues the Gamma-point SCF calculation above
# Assumes a Γ-only SCF converged above.  At Γ the coefficients are
# real (up to an arbitrary sign) and k = (0, 0, 0).

homo = system.n_electrons() // 2 - 1

vq.write_xsf_mo(
    "nacl_homo.xsf", system, basis,
    np.asarray(result.mo_coeffs),       # C(k) — at Γ this is real
    np.array([0.0, 0.0, 0.0]),          # k = Γ
    band_index=homo,
    spacing_bohr=0.2,
    component="real",                    # Re ψ — the standard "MO picture"
)
```

``write_xsf_mo`` constructs a primitive-cell grid, evaluates the Bloch
orbital

$$
\psi_{n,k}(\mathbf{r}) = \sum_{\mathbf{T}} e^{i\mathbf{k} \cdot \mathbf{T}}
    \sum_\mu C_{\mu n}(\mathbf{k}) \, \chi_\mu(\mathbf{r} - \mathbf{T})
$$

at every voxel, and writes the result as a ``DATAGRID_3D`` block.
The ``component=`` keyword selects what scalar to write:

| Component | Scalar | Use when… |
|---|---|---|
| ``"real"`` | Re ψ | Orbitals at Γ or time-reversal-invariant k (standard MO picture) |
| ``"imag"`` | Im ψ | Gauge-paired view at general k |
| ``"abs"`` | \|ψ\| | Gauge-invariant magnitude |
| ``"density"`` | \|ψ\|² | Cross-k comparison; no phase ambiguity |

For a full walk-through of Bloch-orbital evaluation, the
`evaluate_bloch_orbital` kernel, the `PrimitiveCellGrid` factory,
the cube-file workaround for molecular viewers, and the gauge-
invariance considerations, see
[periodic orbital cubes](periodic_orbital_cubes.md).

## 4. BXSF Fermi surfaces

BXSF is the ASCII **Band XSF** format. It stores band energies sampled on a regular 3D
k-mesh across the Brillouin zone.  XCrySDen / moltui render an
isosurface at the Fermi energy, the **Fermi surface**, showing
which bands cross the Fermi level and where in k-space.

### What you need

``write_bxsf`` expects:

* A ``PeriodicSystem`` (for the reciprocal-lattice vectors),
* A 4D ``energies`` array of shape ``(n_kx, n_ky, n_kz, n_bands)``
  in **Hartree** (the writer converts to eV automatically),
* An optional ``e_fermi`` in Hartree.

The k-mesh must be a **regular fractional-coordinate grid** over
``[0, 1)`` along each reciprocal-lattice direction, exactly what
``np.linspace(0, 1, nk, endpoint=False)`` produces.

### End-to-end: NaCl rocksalt Hcore format demonstration

This example diagonalizes Hcore on a dense k-mesh and writes a BXSF
file. Hcore contains only kinetic and nuclear-attraction terms. It is a
format and reciprocal-grid teaching model, not the RHF calculation used
for the density above. Although physical NaCl is a wide-gap insulator, the
non-SCF Hcore model is not guaranteed to preserve that gap. A production
Fermi surface requires a metallic system, a converged self-consistent
operator, and a physically defined chemical potential.

```python
# doc-audit: skip - continues the NaCl system and basis constructed above
import numpy as np
import vibeqc as vq

# The `system` and `basis` objects are the NaCl primitive cell built above.

# 1. Build the real-space Hcore and overlap lattice blocks once.
opts = vq.LatticeSumOptions()
opts.cutoff_bohr = 12.0
opts.nuclear_cutoff_bohr = 25.0

overlap_real = vq.compute_overlap_lattice(basis, system, opts)
kinetic_real = vq.compute_kinetic_lattice(basis, system, opts)
nuclear_real = vq.compute_nuclear_lattice(basis, system, opts)

# 2. Sample [0, 1) in fractional reciprocal coordinates.
nk = 8
mesh = (nk, nk, nk)
energies = np.empty((*mesh, basis.nbasis), dtype=float)
reciprocal = np.asarray(system.reciprocal_lattice())

for index in np.ndindex(mesh):
    k_fractional = np.asarray(index, dtype=float) / np.asarray(mesh)
    k_cartesian = reciprocal @ k_fractional
    overlap_k = vq.bloch_sum(overlap_real, k_cartesian)
    hcore_k = (
        vq.bloch_sum(kinetic_real, k_cartesian)
        + vq.bloch_sum(nuclear_real, k_cartesian)
    )
    overlap_k = 0.5 * (overlap_k + overlap_k.conj().T)
    hcore_k = 0.5 * (hcore_k + hcore_k.conj().T)
    solution = vq.diagonalize_bloch(hcore_k, overlap_k)
    energies[index] = np.asarray(solution.energies, dtype=float)

# 3. Place the display reference midway between the Hcore band edges.
n_occupied = system.n_electrons() // 2
valence_maximum = float(energies[..., n_occupied - 1].max())
conduction_minimum = float(energies[..., n_occupied].min())
e_reference = 0.5 * (valence_maximum + conduction_minimum)
hcore_edge_separation = conduction_minimum - valence_maximum

if hcore_edge_separation <= 0.0:
    print("Hcore manifolds overlap; displayed surfaces are nonphysical")

# 4. Write the ASCII Band XSF file.
vq.write_bxsf(
    "nacl_hcore_bands.bxsf",
    system,
    energies,
    e_fermi=e_reference,
)
# Open nacl_hcore_bands.bxsf in XCrySDen or moltui.
```

The loop evaluates H(k) = T(k) + V(k) at every k-point of the dense
mesh and diagonalizes. Do not assume a generic periodic SCF result has
a real-space Fock lattice set. For a scientific spectrum, use the
post-SCF helper belonging to that SCF route and verify that it evaluates
the same converged operator and energy reference. If the Hcore occupied
and virtual manifolds overlap, the midpoint used above can cut an Hcore
band and produce an isosurface. That surface remains a format demonstration,
not a physical prediction for NaCl.

```{tip}
**For metals, the Fermi surface is non-empty.** Use a converged metallic
SCF workflow and its chemical potential. One or more self-consistent
bands can then cross E_F, and XCrySDen renders the corresponding Fermi
sheets. Merely substituting a metallic geometry in this Hcore recipe
does not make the result production quality.
```

### Performance notes

* An `nk`³ mesh diagonalizing `N_bf` bands costs
  $O(nk^3 N_\mathrm{bf}^3)$ operations.
* BXSF is ASCII. File size scales as $O(nk^3 N_\mathrm{bf})$ with
  roughly 14 to 16 bytes per formatted value in the current writer.
* Increase the teaching mesh only after confirming the small grid. A
  production mesh must be converged for the surface feature being
  reported.

## 5. Viewing in moltui

moltui reads XSF and BXSF natively.  Launch it on any of the files
written above:

```sh
moltui nacl_primitive.xsf          # crystal structure
moltui nacl.xsf                    # density isosurface
moltui nacl_homo.xsf               # Bloch orbital isosurface
moltui nacl_hcore_bands.bxsf       # Hcore band-grid format demonstration
```

**Keyboard controls** (from the geometry / isosurface view):

| Key | Action |
|---|---|
| `h j k l` / arrow keys | Rotate the structure |
| `+` / `-` | Zoom in / out |
| `b` | Toggle periodic replication (XSF only) |
| `m` | Cycle view mode (geometry / orbitals / normal modes) |
| `o` | Toggle isosurface visibility |
| `n` / `p` | Navigate grids / bands (when multiple are present) |
| `Tab` / `Shift-Tab` | Cycle through bands (BXSF) |
| `i` / `I` | Decrease / increase isovalue |
| `V` | Open visual panel (isovalue, style, lighting) |
| `e` | Export PNG screenshot |
| `q` | Quit |

```{tip}
**Periodic replication** (key `b`) tiles the primitive cell across
the display.  This is particularly useful for XSF structure and
density files — you see the full crystal, not just one cell.  The
replication respects the lattice vectors, so off-diagonal cells
(monoclinic, triclinic) tile correctly.
```

For more on moltui, including molecular-orbital viewing, normal-mode
animation, and trajectory playback, see
[viewing with moltui](moltui_terminal_viewer.md).

## 6. Viewing in VESTA / XCrySDen

Beyond the terminal, two desktop GUIs read these files directly: VESTA
for structures and density isosurfaces, and XCrySDen for the same plus
BXSF Fermi surfaces and band-path overlays.

### VESTA

[VESTA](https://jp-minerals.org/vesta/en/) (free, cross-platform
GUI) reads XSF files directly:

```sh
# macOS:
open -a VESTA nacl.xsf

# Linux:
vesta nacl.xsf
```

* **Structure:** File → Open → select the `.xsf`.  VESTA displays
  atoms + bonds with the unit-cell wireframe.  Right-click → Style
  to switch to polyhedral / ball-and-stick / space-filling.
* **Volumetric data:** VESTA automatically detects the
  ``DATAGRID_3D`` block.  Edit → Volumetric Data → adjust the
  isosurface level and colour map.
* **Periodic replication:** Objects → Boundary → set the number of
  replicated cells along each axis.

### XCrySDen

[XCrySDen](http://www.xcrysden.org/) (free, Linux/macOS) is the
reference viewer for the format:

```sh
# XSF structure + volumetric data:
xcrysden --xsf nacl.xsf

# BXSF Fermi surface:
xcrysden --bxsf nacl_bands.bxsf
```

XCrySDen's ``--bxsf`` mode opens the Brillouin-zone wireframe and
renders an isosurface for every band that crosses the Fermi level.
The wireframe is the reciprocal-lattice bounding box; the surfaces
inside it are the Fermi sheets.  Use ``Tools → k-path`` to overlay
a band-structure path on the same display.

```{tip}
**moltui vs VESTA vs XCrySDen.**  moltui is the terminal-native
triage tool — fast, runs over SSH, no GUI.  VESTA is the standard
crystallographic viewer — publication-quality figures, polyhedral
rendering, diffraction patterns.  XCrySDen is the specialist for
Fermi surfaces and band-structure overlay; its BXSF reader is the
reference implementation.  Pick the right viewer for the job.
```

## Theory

The sections below set out the on-disk conventions for both formats:
the real-space XSF block structure, the reciprocal-space BXSF variant,
how the two grids relate, and what a Fermi surface is in terms of the
band energies BXSF stores.

### XSF format conventions

The XSF format (Kokalj 2003) was designed for periodic crystals.
A file opens with a keyword naming the **dimensionality** of the
system, and vibe-qc writes it from ``system.dim``:

| Keyword | ``dim`` | Periodic directions |
|---|---|---|
| ``CRYSTAL`` | 3 | all three PRIMVEC rows |
| ``SLAB`` | 2 | the first two rows (the *xy*-plane) |
| ``POLYMER`` | 1 | the first row only (the *x* axis) |

This keyword matters more than it looks.  A slab or a polymer still
needs a full-rank 3×3 lattice for the AO integrals and for spglib, so
vibe-qc pads the unused ``PRIMVEC`` rows with an arbitrary long vector.
That padding is **not a vacuum gap and not a cell edge**: the SCF total
energy is provably invariant to it.  Yet it is numerically
indistinguishable from a real vacuum gap in a 3D cell.  No amount of
inspecting the matrix can tell them apart; the keyword is the only
signal.  A viewer that ignores it will draw a 2D slab as a thin sheet
adrift inside a large empty box.

The block then contains:

* **PRIMVEC**, three row vectors (in ångström) defining the
  primitive-cell lattice.  vibe-qc converts from its internal bohr
  representation at write time.  For ``SLAB`` and ``POLYMER`` only the
  first ``dim`` rows are physical.
* **PRIMCOORD**, atom count, a format flag (``1`` for fractional
  coordinates), and one line per atom (``Z  x  y  z`` in ångström).
* **DATAGRID_3D_<name>** (optional), a 3D scalar grid spanning
  a chosen real-space volume.  The volume is specified by an
  ``origin`` (ångström) and three ``span`` vectors (ångström);
  for a primitive-cell density these are the lattice vectors.
  Voxel ``(0, 0, 0)`` is at the origin; voxel ``(n_a − 1, n_b − 1,
  n_c − 1)`` is one step short of the opposite corner.  The voxel
  at the far corner, which coincides with the periodic image of
  ``(0, 0, 0)``, is **not** stored.  XSF / VESTA treat it as
  implicit, and including it introduces a spurious stripe at every
  cell boundary (the "off-by-one trap").

The grid data runs with the first index varying fastest, then the
second, then the third, the Fortran-order convention that vibe-qc
honours via ``np.transpose(data, (2, 1, 0)).ravel()``.

### BXSF format conventions

BXSF ("band XSF") uses the same block-structured syntax but in
**reciprocal space**:

* The spanning vectors are the reciprocal-lattice vectors
  $\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$ in **1/Å**.
* The origin is $(0, 0, 0)$, the Γ point.
* Band energies are in **eV** (vibe-qc converts from Hartree
  internally using 1 Ha = 27.211386 eV).
* Each band is one ``BAND:`` block; the band index counts from 1.
* The ``Fermi Energy`` in the ``BEGIN_INFO`` block is in eV.

XCrySDen renders the band structure as $N_\text{bands}$ isosurfaces
inside the BZ wireframe, one per band, at the Fermi energy.  Bands
that never reach $E_F$ produce no visible isosurface; bands that
cross $E_F$ produce Fermi sheets, the textbook Fermi surface.

### Real-space grids (XSF) vs k-space grids (BXSF)

This table contrasts the two formats side by side: domain, grid
convention, the scalar each stores, what its isosurface means, the
span vectors, and the units.

| | XSF | BXSF |
|---|---|---|
| **Domain** | Real space $\mathbf{r} \in$ unit cell | Reciprocal space $\mathbf{k} \in$ BZ |
| **Grid** | Primitive-cell fractional coordinates | Monkhorst-Pack fractional coordinates |
| **Scalar** | ρ(r), ψ_{n,k}(r), or any f(r) | ε_n(k), band energy |
| **Isosurface meaning** | Shape of the density / orbital | Shape of the Fermi surface |
| **Span vectors** | Lattice $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$ | Reciprocal lattice $\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$ |
| **Units** | Å for coordinates, a.u. for data | 1/Å for reciprocal vectors, eV for energies |

The real-space grid tells you *where the electrons are*; the k-space
grid tells you *which electrons cross the Fermi level and at what
crystal momentum*.  Together they give the complete one-electron
picture of a periodic system.

### Fermi surfaces and the BZ

A Fermi surface is the set of k-points where

$$
\varepsilon_n(\mathbf{k}) = E_F
$$

for some band $n$.  In an **insulator**, no band satisfies this
condition, the occupied bands all lie below $E_F$ and the virtual
bands all lie above it, so the BXSF isosurface is empty.  In a
**metal**, at least one band crosses $E_F$, and the resulting
isosurface encloses the occupied region of k-space, the Fermi
sea.  The shape of the Fermi surface determines transport
properties (conductivity, thermopower, magnetoresistance) and is
the central object of metallic electronic structure.

The mesh density `nk` controls the smoothness of the isosurface.
For publication-quality figures, 32³ or denser is recommended;
16³ is enough for orientation and triage.

## References

* **Kokalj, A.** "Computer graphics and graphical user interfaces as
  tools in simulations of matter at the atomic scale," *Comput.
  Mater. Sci.* **28**, 155 (2003).  The XSF format specification.
  Updated at <http://www.xcrysden.org/doc/XSF.html>.
* **Bloch, F.** "Über die Quantenmechanik der Elektronen in
  Kristallgittern," *Z. Phys.* **52**, 555 (1928).  Bloch's theorem
  - the foundation of crystalline orbitals and band theory.
* **Ashcroft, N. W.; Mermin, N. D.** *Solid State Physics*.  Holt,
  Rinehart and Winston (1976), Chapters 8-10 and 15.  The standard
  textbook on Bloch states, band structures, and Fermi surfaces.
  Chapter 15 is the canonical introduction to Fermi-surface
  topology and measurement.

## Next

- [Periodic orbital cubes](periodic_orbital_cubes.md)
  - the `write_xsf_mo` deep-dive: gauge invariance, component
  selection, and the `evaluate_bloch_orbital` kernel.
- [Viewing with moltui](moltui_terminal_viewer.md)
  - full moltui reference: geometries, molecular orbitals, normal
  modes, and trajectory animations in the terminal.
- [Volumetric data user guide](../user_guide/volumetric_data.md)
  - the complete reference for cube, XSF, and BXSF writers,
  including molecular cube files and the `run_periodic_job`
  integration.
- [Periodic HF](periodic_hf.md), the foundation
  for periodic calculations, including k-mesh setup and the
  Monkhorst-Pack scheme.
- [Band structure](band_structure.md), the
  energy-vs-kpath context for the Fermi-surface view BXSF
  provides.
