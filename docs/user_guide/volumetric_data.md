# Volumetric data: cube, XSF, and BXSF files

vibe-qc's native visualization format is **QVF** (``.qvf``): a single
archive carrying the structure, density, orbitals, and basis together,
opened in **vibe-view**. `run_job` and `run_periodic_job` write it by
default (`output_qvf=False` disables it), and it
is the recommended way to visualize vibe-qc results: see
[output files](output_files.md#output-h2oqvf-native-visualization-archive-default),
the [QVF tech spec](../design_qvf_format.md), and
[The QVF file format, end to end](../tutorial/qvf_file_format.md).
For periodic QVF consumers, the downloadable
[`chi-ccm-b-qvf`](../example_outputs.md#static-reference-bundles) fixture
bundle provides validated 1D and 3D finite-BvK archives with density,
orbital, replicated-cell, and Wannier-centre overlay captures.

This page covers the **interchange** formats, for when an external
solid-state or molecular viewer is what you need:

* **Gaussian cube** (``.cube``), molecular electron densities,
  molecular orbitals, and ESP. Read by VMD, Avogadro, PyMOL,
  ChimeraX, and moltui.
* **XCrySDen XSF** (``.xsf``), periodic crystal structures and
  volumetric data on a primitive-cell grid.  Read by VESTA,
  XCrySDen, and moltui (since v0.8.x+ moltui).
* **XCrySDen BXSF** (``.bxsf``), band energies on a 3D k-mesh
  for Fermi-surface plots. Read by XCrySDen and moltui.

## Quick-reference: which writer for which viewer

| Viewer | Format | vibe-qc writer |
|---|---|---|
| **vibe-view** (native) | **QVF** | high-level runners by default; keep `output_qvf=True` explicit when required |
| VESTA, XCrySDen | XSF | ``write_xsf_structure``, ``write_xsf_volume``, ``write_xsf_density``, ``write_xsf_mo``, ``run_periodic_job(... write_density=True)`` |
| XCrySDen (Fermi surface) | BXSF | ``write_bxsf`` |
| VMD, Avogadro, PyMOL, ChimeraX | Cube | ``write_cube_density``, ``write_cube_mo``, ``write_cube_mos``, ``run_job(... write_cube=...)`` |
| moltui (terminal) | XSF, BXSF, Cube | Any of the above, moltui reads all three |

## What you sample

A cube file is a uniformly-spaced 3D scalar field. For an electron
density,

$$
\rho(\mathbf{r}) \;=\; \sum_{\mu\nu} D_{\mu\nu}\, \chi_\mu(\mathbf{r})\, \chi_\nu(\mathbf{r}),
$$

vibe-qc evaluates $\rho$ on every voxel and writes the values in
Bohr$^{-3}$ in the Gaussian-95 cube convention (header, atom block,
then a flat float stream with the third lattice index running
fastest). For a single molecular orbital,

$$
\phi_i(\mathbf{r}) \;=\; \sum_\mu C_{\mu i}\, \chi_\mu(\mathbf{r}),
$$

the same evaluator is invoked, but the values written are
$\phi_i(\mathbf{r})$ (signed amplitude) rather than $|\phi_i|^2$,
so isosurface viewers (Avogadro, VMD, ChimeraX) can render both
lobes of a $\pi$ orbital. To render a density-of-states-like view
of a single MO, square the file post-hoc in your viewer.

The file format itself is documented in Gaussian's
[cube specification](https://gaussian.com/cubegen/);
XSF is described in the
[XCrySDen XSF Format reference](http://www.xcrysden.org/doc/XSF.html)
and the ASCII Band XSF (BXSF) syntax in the same document under
"BXSF file format".

## Molecular: cube files

```python
import vibeqc as vq

mol = vq.Molecule.from_xyz("h2o.xyz")
basis = vq.BasisSet(mol, "6-31g*")
res = vq.run_rhf(mol, basis)

# Density: ρ(r) = Σ D_μν χ_μ χ_ν
vq.write_cube_density("rho.cube", res.density, basis, mol)

# A single MO (zero-based index)
homo = mol.n_electrons() // 2 - 1
vq.write_cube_mo("homo.cube", res.mo_coeffs, homo, basis, mol)

# Multiple MOs in one file (multi-volume cube; viewer flips between them)
vq.write_cube_mos("mos.cube", res.mo_coeffs, [homo - 1, homo, homo + 1, homo + 2],
                   basis, mol)
```

The grid is built automatically: an axis-aligned bounding box around
the molecule plus 4 bohr of padding, with 0.2 bohr cubic voxels (~10⁶
voxels for a small molecule). Override with ``spacing=`` and
``padding=``, or pass a fully custom :class:`vibeqc.CubeGrid`:

```python
# doc-audit: skip - continues the molecule, basis, and result constructed above
grid = vq.make_uniform_grid(mol, spacing=0.1, padding=6.0)
vq.write_cube_density("rho-fine.cube", res.density, basis, mol, grid=grid)
```

For UHF / UKS pass ``D = D_alpha + D_beta`` (the total density).
A typical sanity check: integrating ρ over the box should return the
electron count to ~1% on the default grid, exact in the limit.

## Periodic: XSF files

vibe-qc offers three paths for XSF output, from simplest to most
flexible:

### 1. Automatic via `run_periodic_job`

The simplest route.  Pass ``write_density=True`` and the job writes
``{stem}.xsf`` alongside the other output files. If structure output is
also enabled, the structure-only sibling is ``{stem}.structure.xsf``:

```python
import numpy as np
import vibeqc as vq

# NaCl rocksalt primitive cell, sto-3g.
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
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

vq.run_periodic_job(
    system,
    basis,
    method="RHF",
    output="nacl",
    write_density=True,
    density_spacing_bohr=0.2,
)
# The density is nacl.xsf; the structure-only sibling is nacl.structure.xsf.
```

The density grid spans the primitive cell with ``density_spacing_bohr``
voxel spacing (default 0.2 bohr).  The lattice vectors are written in
Ångström; VESTA / XCrySDen / moltui will tile the cell automatically.

### 2. Structure-only via `write_xsf_structure`

```python
# doc-audit: skip - continues the periodic system constructed above
vq.write_xsf_structure("crystal.xsf", system)
```

Writes the structure block only (PRIMVEC + PRIMCOORD), no volumetric
data.  Useful for quick visual sanity checks on a `PeriodicSystem`
before running anything expensive.  Opens in VESTA / XCrySDen / moltui
as the bare unit cell.

The block keyword follows `system.dim`: `CRYSTAL` for bulk, `SLAB` for
a 2D system, `POLYMER` for a 1D one.  For `dim < 3` the padded PRIMVEC
rows are bookkeeping, not cell edges; see
[XSF format conventions](../tutorial/xsf_bxsf_visualization.md#xsf-format-conventions).

### 3. Custom volumetric data via `write_xsf_volume`

```python
# doc-audit: skip - schematic custom field evaluated by user code
import numpy as np

# Evaluate your scalar field on a primitive-cell grid …
rho = evaluate_density_on_grid(...)  # shape (n1, n2, n3), in bohr

vq.write_xsf_volume(
    "rho.xsf", system,
    data=rho, name="density",
    origin=np.zeros(3),          # Bohr — position of voxel (0,0,0)
    span=system.lattice.T,        # Bohr — grid spanning vectors
)
```

The ``span=`` rows define how far the grid extends, for a
primitive-cell density, pass the lattice vectors.  For a supercell,
pass multiples thereof.  The ``origin=`` is the position of voxel
(0, 0, 0) in Bohr; use ``np.zeros(3)`` for a cell-aligned grid.

The most common XSF grid shape is the **primitive-cell orbital**
evaluated via [Periodic orbital cubes and XSF files](../tutorial/periodic_orbital_cubes.md):

```python
# doc-audit: skip - continues explicit periodic lattice objects from the linked workflow
vq.write_xsf_mo("lih_bonding.xsf", system, basis, C, k, band_index=1,
                spacing_bohr=0.15)
```

### 4. XSF density writer

```python
# doc-audit: skip - continues an explicit lattice-resolved density workflow
vq.write_xsf_density("density.xsf", system, basis, density_real,
                     spacing_bohr=0.2)
```

This lower-level writer requires `density_real` to be an explicit
real-space `LatticeMatrixSet`, such as the output of
`vq.real_space_density_from_kpoints`. A generic periodic result does not
promise a `density_real` attribute. Use the automatic runner route above
unless your workflow already owns the lattice-resolved density.

## Periodic: BXSF files (regular k-space band grids)

BXSF is the ASCII **Band XSF** format. It stores band energies sampled
on a regular three-dimensional reciprocal-space grid. Each band is one
ASCII ``BAND:`` block. XCrySDen renders a constant-energy isosurface,
normally at the Fermi energy. Only a self-consistent metallic spectrum
with a physically defined chemical potential produces a scientific
Fermi surface.

### End-to-end: NaCl Hcore format demonstration

The complete recipe below uses public vibe-qc APIs to build an Hcore
band grid for a true two-atom rocksalt primitive cell. Hcore contains
only kinetic and nuclear-attraction terms. The file is useful for
testing BXSF layout and viewer operation, but it is not an RHF or DFT
band structure and must not be interpreted as a NaCl Fermi surface.

```python
import numpy as np
import vibeqc as vq

# 1. True NaCl rocksalt primitive cell. Lattice vectors are columns.
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
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

# 2. Build the real-space Hcore and overlap lattice blocks once.
opts = vq.LatticeSumOptions()
opts.cutoff_bohr = 12.0
opts.nuclear_cutoff_bohr = 25.0
overlap_real = vq.compute_overlap_lattice(basis, system, opts)
kinetic_real = vq.compute_kinetic_lattice(basis, system, opts)
nuclear_real = vq.compute_nuclear_lattice(basis, system, opts)

# 3. Sample [0, 1) in fractional reciprocal coordinates.
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

# 4. Put the display reference midway between the Hcore band edges.
n_occupied = system.n_electrons() // 2
valence_maximum = float(energies[..., n_occupied - 1].max())
conduction_minimum = float(energies[..., n_occupied].min())
e_reference = 0.5 * (valence_maximum + conduction_minimum)
hcore_edge_separation = conduction_minimum - valence_maximum

if hcore_edge_separation <= 0.0:
    print("Hcore manifolds overlap; displayed surfaces are nonphysical")

vq.write_bxsf(
    "nacl_hcore_bands.bxsf",
    system,
    energies,
    e_fermi=e_reference,
)
```

NaCl is physically an insulator, but that fact does not guarantee that
this non-SCF Hcore model has a positive indirect gap. The chosen reference
is only the midpoint between its occupied and virtual band edges. If those
manifolds overlap, the viewer may draw an Hcore isosurface; it is still not
a physical Fermi surface. For a production metallic surface, obtain the
dense-grid eigenvalues and chemical potential from a route-specific
converged SCF workflow. Generic periodic result objects do not expose a
universal real-space Fock lattice set.

### Units and conversions

* BXSF is plain ASCII, not a binary container.
* BXSF energies are conventionally in **eV**.  vibe-qc's internal
  energies are in Hartree; ``write_bxsf`` converts automatically.
* The reciprocal lattice spanning vectors are in **1/Å**.
* The k-mesh origin is (0, 0, 0); the three spanning vectors are the
  reciprocal lattice vectors $\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$.
* Grid data runs with the **first** k-index varying fastest (same
  Fortran-order convention as XSF DATAGRID_3D).

## Viewing in moltui

moltui reads XSF and BXSF natively alongside its existing cube
support:

```sh
# Install (one-time)
./scripts/install_optional_tools.sh moltui

# View a crystal structure
moltui nacl.xsf

# View a density isosurface inside the unit cell
moltui nacl.xsf

# Inspect the Hcore band grid (NaCl has no physical Fermi surface here)
moltui nacl_hcore_bands.bxsf
```

Keyboard controls in moltui:

| Key | Action |
|---|---|
| `b` | Toggle periodic replication |
| `m` | Cycle view mode (geometry / orbitals / normal modes) |
| `o` | Toggle isosurface visibility |
| `n` / `p` | Navigate grids/bands (when multiple are present) |
| `V` | Open visual panel (isovalue, style, lighting) |
| `e` | Export PNG screenshot |

Set the XSF viewer as your default with the ``moltui`` command; for
desktop-quality rendering, pipe the same file to VESTA or XCrySDen.

## End-to-end example: HOMO / LUMO of water

A reproducible water HOMO/LUMO cube generation, ready to copy:

```python
# input-water-cubes.py
from pathlib import Path
import vibeqc as vq

HERE = Path(__file__).parent

mol = vq.Molecule([
    vq.Atom(8, [ 0.0,  0.00,  0.00]),   # bohr
    vq.Atom(1, [ 0.0,  1.43, -0.98]),
    vq.Atom(1, [ 0.0, -1.43, -0.98]),
])
basis = vq.BasisSet(mol, "6-31g*")
result = vq.run_rhf(mol, basis)

# Closed-shell H2O has 10 electrons → 5 occupied MOs.
homo = mol.n_electrons() // 2 - 1            # MO index 4
lumo = homo + 1                              # MO index 5

vq.write_cube_density("rho.cube", result.density, basis, mol)
vq.write_cube_mo("homo.cube", result.mo_coeffs, homo, basis, mol)
vq.write_cube_mo("lumo.cube", result.mo_coeffs, lumo, basis, mol)
```

Drag the three files into VMD / Avogadro / ChimeraX. `rho.cube`
shows the total electron density (canonical isovalue 0.05 e/bohr$^3$
for a textbook bond-density picture); `homo.cube` and `lumo.cube`
render with isovalue ±0.05 a.u. to show the lobes (red = negative
amplitude, blue = positive, viewer-dependent).

Sanity check: $\int \rho \,d^3r$ should integrate to 10 (vibe-qc's
test suite verifies this to ~1% on the default grid). $\int |\phi_i|^2
\,d^3r$ should integrate to 1 for any occupied MO.

## Validation

`tests/test_cube.py` checks:

- the writer header parses back as expected (atom count, voxel
  vectors, grid shape, multi-value flag),
- ρ(r) integrates to the electron count on a fine grid,
- single-MO files have ⟨φ|φ⟩ ≈ 1.

`tests/test_xsf.py` checks:

- ångström conversion is correct for lattice and atom positions,
- XSF DATAGRID traversal order (first index runs fastest, then j,
  then k), required by VESTA/XCrySDen,
- BXSF conventions: 1/ångström reciprocal lattice and eV energies.
