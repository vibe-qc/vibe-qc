# Viewing geometries, orbitals, and vibrations with MolTUI

**You'll learn:** how to inspect vibe-qc output files, 
geometries, molecular orbitals, normal modes, directly in the
terminal, without firing up Avogadro / Jmol / VMD.

**Why:** when you're SSH'd into a remote machine, iterating
quickly on a screen full of input files, or just allergic to GUI
context-switches, [MolTUI](https://github.com/kszenes/moltui) is
the right tool. It renders 3D structures inline using Unicode
block characters; you navigate with the keyboard.

**Prerequisites:** vibe-qc installed in a venv, plus MolTUI on
top. Two install paths:

```sh
# As a vibe-qc extra (single command, captures the dependency):
.venv/bin/pip install '.[viewer]'

# Or via the interactive helper:
./scripts/install_optional_tools.sh
```

After install, ``moltui`` is on the venv's PATH:

```sh
.venv/bin/moltui --version
```

## What MolTUI reads

MolTUI opens a spread of molecular and periodic formats; the table below
maps each one to what it renders and to the vibe-qc call that emits it,
flagging which are written today versus not-yet-supported.

| Format | What it shows | Where vibe-qc writes it |
|---|---|---|
| `.xyz` | Geometry (single frame) | `Molecule.to_xyz()`, ASE export |
| `.molden` | Geometry + MOs + normal modes | `run_job(...)` by default |
| `.cube` | Volumetric data (orbitals, density) | `write_cube_mo()`, `write_cube_density()` |
| `.fchk` | Geometry + MOs + normal modes (Gaussian fchk) | (not yet emitted by vibe-qc) |
| `.gbw` | ORCA's native binary (needs `orca_2mkl`) | (not emitted by vibe-qc; ORCA-specific) |
| `.hess` | ORCA's normal-modes file | ✅ ``vq.write_orca_hess(path, mol, hess)`` (M1) |
| `.xsf` | Crystal structure + volumetric data (density, Bloch orbitals) | ✅ `write_xsf_structure`, `write_xsf_volume`, `write_xsf_mo`, `run_periodic_job(write_density=True)` |
| `.bxsf` | Fermi surfaces (band energies on k-mesh) | ✅ `write_bxsf` |
| Multi-frame `.xyz` | Trajectory animations | ✅ ``vq.write_xyz_trajectory(path, frames)`` (M2) |
| `.opt` | Geometry-optimization history (multi-XYZ + per-step E) | ✅ ``vq.write_opt_trajectory(path, frames, energies)`` (M2) |

The two formats vibe-qc writes today out of the box are
**`.molden`** (full molecular-orbital data, plus geometry, plus
normal modes if a Hessian was computed) and **`.cube`** (volumetric
data for individual orbitals or the total density). **`.xsf`** is
the periodic counterpart for crystal structures and Bloch orbitals.
All three work with MolTUI directly.

## Step 1: Generate the files

Run a quick PBE calculation on water through `run_job`, which writes a
`.molden` carrying the geometry, MOs, and occupations that the rest of
this walkthrough inspects.

```python
from vibeqc import Atom, Molecule, run_job

mol = Molecule([
    Atom(8, [0.0,  0.00,  0.00]),
    Atom(1, [0.0,  1.43, -0.98]),
    Atom(1, [0.0, -1.43, -0.98]),
])

run_job(
    mol,
    basis="6-31g*",
    method="rks",
    functional="PBE",
    output="water",
)
# → water.out, water.molden
```

That gives you `water.molden` with the geometry + MOs + occupations.
Add a cube for density / orbital visualization:

```python
import numpy as np
import vibeqc as vq

basis = vq.BasisSet(mol, "6-31g*")
result = vq.run_rks(mol, basis, vq.RKSOptions(functional="PBE"))

# HOMO orbital cube
homo = mol.n_electrons() // 2 - 1
vq.write_cube_mo("water-homo.cube",
                 np.asarray(result.mo_coeffs), homo, basis, mol,
                 spacing=0.2, padding=3.0)

# Total density cube
vq.write_cube_density("water-density.cube",
                      np.asarray(result.density), basis, mol,
                      spacing=0.2, padding=3.0)
```

## Step 2: View geometries

Launch MolTUI on the .molden:

```sh
moltui water.molden
```

The default view shows the molecular geometry rendered inline:

```text
                    ┌──────── water.molden ────────┐
                    │                              │
                    │              o               │
                    │                              │
                    │           h     h            │
                    │                              │
                    │                              │
                    └──────────────────────────────┘
                    [g] geometry  [o] orbitals  [v] vibrations
```

(rendering varies by terminal size + Unicode font; on a typical
full-screen 100×40 character terminal you get a recognisable 3D
structure)

**Keyboard navigation** (defaults, see `moltui --help` for the
authoritative list):

| Key | Action |
|---|---|
| arrow keys / `h j k l` | Rotate the structure |
| `+` / `-` | Zoom in / out |
| `g` | Switch to geometry view |
| `o` | Switch to orbitals view (.molden / .cube / .fchk) |
| `v` | Switch to vibrations view (.molden / .hess / .fchk) |
| `Tab` / `Shift-Tab` | Cycle frame / orbital / mode |
| `q` | Quit |

## Step 3: View orbitals

From the geometry view, press `o` to switch to orbitals. MolTUI
parses the molden file's MO-coefficient block and renders an
isosurface inline:

```text
              ┌───── HOMO   ε = -6.09 eV   occ = 2 ─────┐
              │                                         │
              │              ▓▒░       ░▒▓              │
              │             ▓▒░         ░▒▓             │
              │             ▓▒░    o    ░▒▓             │
              │             ▒░           ░▒             │
              │            h                h           │
              │                                         │
              │                                         │
              └─────────────────────────────────────────┘
              [Tab]/[Shift-Tab] cycle MO  [+/-] zoom  [q] quit
```

Light shading is the positive isosurface lobe, darker is the
negative. The HOMO of water is the in-plane oxygen lone pair, 
two lobes opposite the H atoms.

**`Tab`** cycles through the MOs; the header shows the eigenvalue
and the occupation. Useful for sanity-checking that:

* The HOMO has the right symmetry for your molecule (lone pairs,
  π systems, etc.).
* The LUMO ordering is what you expect (no surprise σ\* /
  Rydberg crossings).
* The orbital energies (header) match what's in the `.out`
  file's orbital table.

For finer-grained orbital views, explicit isovalue control,
publication-quality renderings, open the `.molden` or one of the
cube files in Avogadro / Jmol / VMD instead. MolTUI is for
*quick triage*, not for paper figures.

## Step 4: View cube files (volumetric data)

For more detailed per-orbital control:

```sh
moltui water-homo.cube
moltui water-density.cube
```

Cube files are dense volumetric grids, MolTUI raycasts an
isosurface at a default isovalue (usually 0.05 a.u.). Adjust with:

```text
[i]/[I]  decrease/increase isovalue
[r]      reset isovalue to default
```

A **density cube** (sum over occupied orbitals) shows the total
electron density, the "shape" of the molecule. A **single-MO
cube** shows one orbital at a time (HOMO, LUMO, anything you
asked `write_cube_mo` to output).

## Step 5: View vibrational modes

If your `.molden` has a `[FREQ]` and `[FR-COORD]` block (which
vibe-qc emits when you compute a Hessian), MolTUI renders the
normal modes as animated displacement vectors. Press `v` from
any view to switch.

```text
        ┌── mode 7   ω = 1751.3 cm⁻¹   IR = 67.8 km/mol ──┐
        │                                                 │
        │              ↑   o   ↑                          │
        │              ↑       ↑                          │
        │              h       h                          │
        │              (symmetric stretch)                │
        │                                                 │
        └─────────────────────────────────────────────────┘
        [Tab]/[Shift-Tab] cycle mode  [Space] play/pause
```

The arrows (or oscillating atoms, depending on the terminal +
animation toggle) show the eigenvector of that normal mode.
For water you'll see three real modes, bend, symmetric stretch,
antisymmetric stretch, and six near-zero translation/rotation
modes that MolTUI typically hides.

To produce a `.molden` *with* vibrational data:

```python
import vibeqc as vq

# Compute analytic Hessian + frequencies
mol = vq.Molecule.from_xyz("water-equilibrium.xyz")
basis = vq.BasisSet(mol, "6-31g*")
result = vq.run_rhf(mol, basis)
hess = vq.compute_hessian_rhf_analytic(
    mol, basis, result, basis_name="6-31g*",
)
# Write a molden file with the normal modes attached
vq.write_molden("water-with-freqs.molden", mol, basis, result,
                hessian=hess)
```

```{tip}
**Direct ORCA-format `.hess` export** is available via
:func:`vibeqc.write_orca_hess` (Phase M1). Useful when you'd
rather inspect just the vibrational data than parse the full
molden bundle — and it's the format moltui reads natively for
ORCA-style normal-modes display:

    hess = vq.compute_hessian_rhf_analytic(mol, basis, result, basis_name="6-31g*")
    vq.write_orca_hess("water.hess", mol, hess)
    # then:  moltui water.hess

The same ASCII format means cross-code visual diff'ing works:
``moltui vibeqc.hess`` and ``moltui orca.hess`` should show
identical modes when the underlying CPHF kernels agree.
See [Cross-validating against ORCA, Psi4, and PySCF](cross_validation.md)
for the cross-validation script that asserts ~0.5 cm⁻¹ agreement
on H₂O.
```

## Tips and tricks

* **Over SSH:** MolTUI runs entirely in your terminal, no X11
  forwarding, no GUI windows. Just `ssh` in, `cd` to your
  output directory, and `moltui ...`.
* **Quick triage:** for a calculation that just finished, look
  at the `.out` file first (text, energy, convergence,
  warnings), then `moltui ...molden` to spot-check the geometry
  + HOMO shape. Catches "the SCF converged to the wrong state"
  bugs immediately.
* **Animations** (Phase M2): vibe-qc writes multi-XYZ trajectories
  natively, geometry-optimization paths, NEB images, normal-mode
  movies, MD snapshots, and moltui animates them frame-by-frame.

  **Optimization history → moltui movie**:

  ```python
  import vibeqc as vq
  geoms, energies, rms_grads = run_my_optimization(mol)
  vq.write_opt_trajectory("h2o.opt", geoms, energies,
                            rms_grad=rms_grads)
  ```

  ```sh
  moltui h2o.opt          # animated optimisation path with per-step E in the
                          # comment line — Tab cycles iterates
  ```

  **Normal-mode movie**, paired helper builds the displaced frames
  for one vibrational mode:

  ```python
  hess = vq.compute_hessian_rhf_analytic(mol, basis, result, basis_name="6-31g*")
  frames = vq.normal_mode_trajectory(mol, hess, mode_index=8,
                                       amplitude=0.5, n_frames=20)
  vq.write_xyz_trajectory("h2o-asym-stretch.xyz", frames)
  ```

  ```sh
  moltui h2o-asym-stretch.xyz   # animated symmetric-stretch movie
  ```

  Falling back via ASE, convert any ASE `.traj` to multi-XYZ, 
  also still works:

  ```sh
  ase convert output-h2o-opt.traj output-h2o-opt.xyz
  moltui output-h2o-opt.xyz
  ```
* **Keep `moltui` open** in a separate terminal while you iterate
  on input files; reload by re-running `moltui <file>` after
  each calculation finishes.

## Crystalline orbitals and periodic data

Since v0.8.x+, MolTUI supports **XSF and BXSF**, XCrySDen's native
periodic formats, alongside its existing molecular formats:

| Format | What it shows | Where vibe-qc writes it |
|---|---|---|
| `.xsf` | Crystal structure + volumetric data (density, Bloch orbitals) | `write_xsf_structure`, `write_xsf_volume`, `write_xsf_density`, `write_xsf_mo`, `run_periodic_job(write_density=True)` |
| `.bxsf` | Fermi surfaces (band energies on k-mesh) | `write_bxsf` |

**What works:**

* **Unit-cell wireframe**, lattice vectors drawn as the bounding box.
* **Periodic replication**, press `b` to tile the cell N×N×N along
  the lattice vectors.
* **Isosurfaces**, density or Bloch orbitals rendered inside the cell
  (same isosurface pipeline as cube files).
* **Band cycling**, for multi-grid XSF or BXSF, `Tab` cycles through
  grids/bands.
* **Fermi-surface rendering**, BXSF files show each band's
  intersection with the Fermi energy.

```sh
# View a crystal structure
moltui nacl.xsf

# View density inside the unit cell
moltui nacl.density.xsf

# Cycle through Fermi surfaces
moltui nacl_bands.bxsf
```

See [XSF and BXSF: periodic volumetric visualisation](xsf_bxsf_visualization.md) for the full
end-to-end workflow, from SCF → XSF/BXSF → MolTUI.

**What doesn't work yet:**

* **.molden periodic extension**, the format has no cell/lattice
  concept.  Use XSF or cube instead.
* **Per-axis replication**, in released MolTUI the `b` toggle
  replicates all three axes uniformly and draws the padded lattice
  rows of a 1D/2D system as if they were cell edges, so a slab
  renders as a sheet inside a large empty box.  vibe-qc now gives the
  viewer what it needs to do better: the XSF it writes opens with
  ``SLAB`` / ``POLYMER`` rather than ``CRYSTAL`` (see [XSF format
  conventions](xsf_bxsf_visualization.md#xsf-format-conventions)), and
  a fix that honours the keyword is pending upstream in MolTUI.
* **k-point navigation**, you view one crystalline orbital at one
  k-point per file.  No in-app k-point picker.

For desktop-quality rendering, pipe the same XSF/BXSF file to VESTA
or XCrySDen after MolTUI triage.

## When MolTUI isn't enough

MolTUI is a triage tool, fast, terminal-native, low-resolution.
For:

* **Publication figures**, Avogadro 2 (cross-platform GUI),
  VMD (scriptable, scientific-grade), ChimeraX (high-quality
  surfaces), Jmol (web-friendly).
* **Detailed orbital analysis**, natural-bond-orbital
  decompositions, ELF/NCI plots, charge-transfer maps, use
  Multiwfn (free, Windows/Wine on macOS+Linux) or pymol.
* **Crystal structures**, VESTA (free), pymatgen visualizations.

vibe-qc writes formats every one of these tools reads (`.molden`,
`.cube`, `.xyz`, future `.gbw`/`.hess`); pick the right viewer for
the job. MolTUI's superpower is **no GUI, no setup, runs over
SSH**, that's the niche.

### MolTUI or vibe-view's terminal mode?

vibe-view has [its own terminal renderer](vibe_view_terminal.md) that shares
that niche. The two are not competitors so much as different entry points:

| | MolTUI | `vibe-view show` / `tui` |
|---|---|---|
| Reads | `.molden`, `.cube`, `.xyz`, `.xsf`, `.fchk`, `.gbw`, CIF | `.qvf` only |
| Source | any code that writes those formats | vibe-qc (or any QVF producer) |
| Shows | geometry, MOs, normal modes, cells | those **plus** bands, DOS, spectra, SCF trail, charges, bond orders, run record, citations |
| Install | separate tool (`.[viewer]` extra) | ships with vibe-view; `[tui]` extra only for the interactive app |

Use MolTUI when you have loose files, or output from a code other than
vibe-qc. Use terminal mode when you have a `.qvf`: because the archive is
self-describing, one command reaches every section the calculation wrote
rather than the one file you happened to open.

## See also

* [installation: optional terminal viewer](../installation.md#optional-terminal-viewer-moltui)
  - install paths
* [output files user guide](../user_guide/output_files.md), 
  what vibe-qc writes
* [orbital visualization](orbital_visualization.md)
  - generating cube files for visualization
* [vibrational frequencies](vibrational_frequencies.md)
  - generating Hessians for normal-mode analysis
* [reading a `.qvf` in the terminal, over SSH](vibe_view_terminal.md)
  - vibe-view's own terminal renderer, for QVF archives
* [MolTUI repo](https://github.com/kszenes/moltui), 
  authoritative docs for keybindings + supported formats
