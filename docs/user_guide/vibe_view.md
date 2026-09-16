---
myst:
  html_meta:
    "description": "vibe-view is an independently installed 3D viewer for QVF (.qvf) archives from conforming producers. Install, launch, sidebar walkthrough, per-section rendering, viewer_defaults hints."
    "og:title": "vibe-view: independent interactive viewer"
    "og:description": "Open a .qvf in vibe-view: structure, density isosurfaces, orbitals, band structures, spectra, trajectories, vibrational modes."
    "og:image": "https://vibe-qc.com/docs/_static/logo/vibe-view-social.png"
    "og:image:width": "1200"
    "og:image:height": "630"
    "og:image:alt": "vibe-view: See the structure. Understand the result."
    "twitter:card": "summary_large_image"
    "twitter:image": "https://vibe-qc.com/docs/_static/logo/vibe-view-social.png"
    "twitter:image:alt": "vibe-view: See the structure. Understand the result."
---

# vibe-view: interactive viewer

```{figure} ../_static/logo/vibe-view-social.svg
:alt: vibe-view: See the structure. Understand the result.
:width: 1200px
:class: product-art

Open the [vibe-view manual](https://vibe-qc.com/vibe-view/docs/) for the independently maintained viewer. This page covers its use with vibe-qc.
```

Source commands on this page run from the **separate vibe-view checkout**.
Clone [vibe-qc/vibe-view](https://github.com/vibe-qc/vibe-view) and follow
[viewer setup](../tutorial/vibe_view_getting_started.md). Its `.venv`, `scripts/`, and `electron/` belong to that repository;
vibe-qc does not contain or install the viewer.

vibe-view is a standalone browser, desktop, terminal, conversion, and headless
viewer for quantum-chemistry results. It opens
[QVF](../design_qvf_format.md) (`.qvf`) archives from any conforming producer
and common loose formats from other codes, without requiring vibe-qc. For QVF
it renders every section the calculation produced: structure, electron
density, molecular orbitals, band structures, spectra, geometry-optimisation
trajectories, vibrational modes, NEB / IRC reaction paths, and more.

```{figure} ../_static/plots/vibe_view/15-orbital-homo.png
vibe-view in the browser: the section sidebar (left), the
GPU-accelerated 3D viewport (centre), and per-section controls (right).
Here, a molecular orbital with both signed lobes.
```

The [vibe-view manual](https://vibe-qc.com/vibe-view/docs/) is published
independently. Start with its [installation guide](https://vibe-qc.com/vibe-view/docs/installation.html),
[quickstart](https://vibe-qc.com/vibe-view/docs/quickstart.html), or
[CLI reference](https://vibe-qc.com/vibe-view/docs/cli.html) for the current
viewer release. This core guide covers integration with vibe-qc; standalone
viewer behavior and installation requirements are maintained by the viewer project.

## What you need on disk

Install vibe-view, then start with any one of these inputs:

1. No file at all: `vibe-view demo --open` creates and opens the bundled
   water demo.
2. A `.qvf` from vibe-qc or another conforming producer. In vibe-qc,
   `run_job` and `run_periodic_job` default to `output_qvf=True`, which writes
   `{stem}.qvf` beside `{stem}.out`. Set `output_qvf=False` only when you
   deliberately do not want the archive.
3. A supported loose XYZ, CIF, PDB, MOL2, GRO, SDF/Mol, Cube, or Gaussian
   input file. Run `vibe-view formats` for the live importer inventory.

vibe-view has its own repository, version line, and release tags. The QVF
file is the interface between the engine and the viewer.

## Install

vibe-view is installed independently. For a
standalone install with the command-line, browser, desktop-server, and TUI
modes, run the viewer's own installer from the checkout root:

```sh
./scripts/install.sh
source .venv/bin/activate

# Later: update the current branch and refresh the same environment.
./scripts/update.sh

# Include the source-backed Electron desktop application.
./scripts/update-desktop.sh

# Repair the local environment without changing Git.
./scripts/reinstall.sh
./scripts/reinstall.sh --desktop

# Preview, then remove the standalone source installation.
./scripts/uninstall.sh --dry-run
./scripts/uninstall.sh
```

The environment stays under `.venv` and does not build or install
vibe-qc. Electron downloads automatically on the first `vibe-view desktop`
launch; add `--with-electron` to the installer to fetch it up front.
`update-desktop.sh` refreshes only the app owned by this checkout. Packaged
apps installed from a DMG keep their separate packaged-update lifecycle.

`reinstall.sh` safely replaces the environment from the current checkout and
restores the old one if creation, installation, verification, or optional
desktop setup fails. Quit all desktop windows and running CLI/browser-server
launches before a desktop reinstall or uninstall. `uninstall.sh` removes the
dedicated viewer environment, the recognizable checkout Electron runtime, and
only a macOS source app owned by this checkout. It preserves packaged and
foreign apps, QVF files, settings, recents, logs, and the app-managed
`$XDG_DATA_HOME/vibe-view/venv` by default.
Use `--keep-desktop` when you want to retain the Electron download for a later
reinstall.

If one Python process needs both packages, follow
[Install both](../getting_started.md#install-both). That recipe installs two
independent checkouts explicitly; the core `viewer-gpu` extra cannot fetch
the companion source from GitHub or GitLab.

The dependency footprint is pure-pip. The core install (QVF reading plus
headless screenshot capture) is PyVista, VTK, matplotlib, Plotly, Click,
Pydantic, and jsonschema. The interactive web viewer adds Trame and
uvicorn via the `[viewer]` extra.

Opening structures in other formats is covered below; `.xyz`, `.cif`,
`.pdb`, `.mol2`, `.gro`, `.sdf`, `.cube` and Gaussian input are read
directly, and the long tail needs one more extra:

```sh
python -m pip install -e '.[ase]'  # POSCAR, .extxyz, .traj, ...
```

vibe-view will tell you, naming that command, if you open a file that
needs it. No conda, no JavaScript build step, no
npm. The heaviest single dep is VTK (about a 200 MB wheel), so the first
install takes a minute or two.

Verify the install:

```sh
vibe-view --version
# vibe-view 2.15.2  -- Roothaan's Roadrunner   (your version may differ)
```

## Produce a .qvf file from vibe-qc

Both runners default to `output_qvf=True`. Keeping it explicit in a
publication input can make the artifact policy obvious. The flag stacks with
`write_cube`, `write_molden_file`, and the other artefact
toggles.

### Molecular: H2O / PBE / 6-31G*

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
    optimize=True,                  # writes a geometry trajectory
    output="water",
    output_qvf=True,                # produces water.qvf
    write_cube=["density", "homo", "lumo"],  # embeds volumetric grids
    write_molden_file=True,         # embeds wavefunction.gto + MOs
)
```

Produces `water.qvf` alongside the usual `water.out` / `water.molden`
/ `water.traj` / `water.bibtex` siblings.

### Periodic: two-site MgO teaching cell / RHF / sto-3g

```python
import numpy as np
import vibeqc as vq

a = 4.21 / 0.529177                                    # Angstrom to bohr
mgo = vq.PeriodicSystem(
    3,
    np.eye(3) * a,
    [vq.Atom(12, [0.0, 0.0, 0.0]),
     vq.Atom(8,  [a/2, a/2, a/2])],
)
basis = vq.BasisSet(mgo.unit_cell_molecule(), "sto-3g")

kpath = vq.kpath_from_segments(
    mgo,
    segments=[
        ([0.0, 0.0, 0.0], "G", [0.5, 0.0, 0.5], "X"),
        ([0.5, 0.0, 0.5], "X", [0.5, 0.5, 0.5], "L"),
        ([0.5, 0.5, 0.5], "L", [0.0, 0.0, 0.0], "G"),
    ],
    points_per_segment=20,
)

vq.run_periodic_job(
    mgo,
    basis=basis,
    method="RHF",
    output="mgo",
    output_qvf=True,
    write_density=True,
    band_structure=vq.band_structure_hcore(mgo, basis, kpath),  # not RHF bands
)
```

The optional `band_structure=` kwarg embeds a band-structure section
in the archive so vibe-view can render the interactive Plotly band plot
without having to re-diagonalise.

This simple-cubic body-centered geometry is CsCl-type, not rocksalt. The band
object in this compact viewer example uses Hcore and is not the RHF spectrum.
For scientific interpretation, use a physically intended cell and a
route-specific helper that diagonalizes the converged SCF operator.

## Launch

### Command line

```sh
vibe-view open water.qvf
```

Boots the Trame server on `http://127.0.0.1:8080` and opens the
default browser at that URL. Flags:

```sh
vibe-view open water.qvf --port 9876        # bind to a different port
vibe-view open water.qvf --no-browser       # skip the auto-open
vibe-view open water.qvf --host 0.0.0.0     # bind to all interfaces (remote use)
```

```{warning}
`--host 0.0.0.0` exposes the viewer server to every reachable network
interface, and the server has no authentication layer. Use it only on a
trusted network. For a remote machine, prefer an SSH tunnel while leaving
vibe-view bound to its default `127.0.0.1`; the
[standalone tutorial](../tutorial/vibe_view_getting_started.md#platform-notes)
has the copy-paste command.
```

### Programmatic

```python
# doc-audit: skip - shows alternative launch forms; qvf_bytes comes from the caller
import io

from vibeview import launch_qvf

launch_qvf("water.qvf")                          # path on disk
launch_qvf(open("water.qvf", "rb"))              # any seekable file-like
launch_qvf(io.BytesIO(qvf_bytes), open_browser=False)  # in-memory
```

`launch_qvf` blocks until the Trame server stops (Ctrl+C). Pass an
already-constructed `QVFReader` instance if you want to inspect the
archive before launching the UI.

### Terminal mode: no browser, no OpenGL

`vibe-view open` needs a browser and a GL context, and so does
`vibe-view capture`. Neither is available on a typical compute node reached
over SSH, which is where a `.qvf` is usually produced. For that case the
viewer has a third front-end that renders into the terminal itself, using a
pure-numpy software rasterizer and Unicode braille characters:

```sh
vibe-view show water.qvf                     # one frame, then exit
vibe-view show water.qvf --section homo --isovalue 0.03
vibe-view show water.qvf --info              # provenance + section inventory
vibe-view tui  water.qvf                     # interactive, `?` for keys
```

```{figure} ../_static/plots/vibe_view/tui-02-orbital.svg
`vibe-view tui` showing a molecular orbital. Isosurfaces come from VTK's
marching-cubes filter, which is pure computation and needs no render window,
then go through the same software rasterizer as the atoms and are depth
tested against them.
```

`vibe-view show` runs on the base install. The interactive `vibe-view tui`
needs the `[tui]` installer profile, which adds only Textual: pure Python, no
native build. The [standalone tutorial](../tutorial/vibe_view_getting_started.md)
shows both checkout and wheel installations.

Structures, isosurfaces, trajectories and normal modes draw in 3D; bands,
DOS, spectra, SCF traces, equations of state, phonons and scan surfaces
chart; citations, run records, charges, bond orders and MO listings get text
panes. Element colours and radii come from the same `cpk_color` /
`cpk_radius` the browser viewer uses, so the two agree on what carbon looks
like.

From Python, the text counterpart of `capture_structure`:

```python
from vibeview import render_terminal

print(render_terminal("water.qvf", size=(100, 30)))
print(render_terminal("water.qvf", "homo", isovalue=0.03))
```

Full option and key reference:
[terminal mode](vibe_view_terminal_mode.md).
Worked example: [reading a `.qvf` in the terminal, over
SSH](../tutorial/vibe_view_terminal.md).

## Opening structure files

vibe-view's own format is QVF, but it opens ordinary structure files too.
That is handy for building a system, or for looking at something a
calculation has not produced yet. Pass explicit files or let your shell expand
a glob:

```sh
vibe-view open water.xyz
vibe-view open ./structures/*.xyz
vibe-view serve ./structures/         # browser-based directory picker
```

**Read directly, no extra needed:** `.qvf`, `.xyz`, `.cif`, `.cube`, `.pdb`,
`.mol2`, `.gjf` / `.com`, `.gro`, `.sdf` / `.mol`, and `.py` (a vibe-qc input
script).

**Read through ASE, needs the `[ase]` extra:** `.traj`, `.extxyz`, `.vasp`,
`.poscar`, and VASP's extensionless `POSCAR` / `CONTCAR`. Those are the names
explicitly routed to ASE; other formats supported by ASE are not automatically
accepted by vibe-view.

ASE is optional because it is a large dependency that most people opening a
QVF never need. Select the `all` profile in the checkout installer or install
the downloadable wheel with its `[ase]` extra. In a current checkout or the
legacy 2.15.2 wheel, run `vibe-view formats` to see whether it is available.

`open` converts a loose file in memory. A current checkout (and the
legacy 2.15.2 wheel) can keep the result with `vibe-view import INPUT`; to
understand exactly what each importer preserves, read
[Input formats and interoperability](vibe_view_formats.md).

Do not pass a directory to `open`. `serve DIRECTORY` scans a directory, while
`import DIRECTORY` creates one persistent QVF per recognized source file.

## The startup banner

Before the server starts, vibe-view prints a summary banner to
stdout that lists every section in the archive and what it will do
with each one:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  QVF file: water.qvf                                                         ║
║  Source:   vibe-qc <version> - water                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Section ID         Kind                         Status                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  structure          structure                    rendered                    ║
║  density            volume.density               rendered                    ║
║  homo               volume.orbital               rendered                    ║
║  lumo               volume.orbital               rendered                    ║
║  ir                 spectra.ir                   rendered                    ║
║  traj0              trajectory                   rendered                    ║
║  x_custom.notes     x_custom.notes               skipped, vendor namespace   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

`rendered` means vibe-view has a renderer for the section's kind and
will draw it. `skipped, unsupported` means the kind is not in the
viewer's [`SUPPORTED_KINDS`](#supported-kinds) registry. `skipped,
vendor namespace` is the same thing but for `x_<vendor>.*` kinds
that producers register without the viewer needing to know about
them. A known vendor section can still feed an overlay even when the
section itself is listed as vendor metadata; for example,
`x_ccm.wannier_centers` drives optional Wannier-centre markers. Unknown
non-critical sections never abort the open; they appear in the sidebar
with the same status string.

## The UI

When the browser opens you get three regions:

* **Top bar**: the source banner (program, version, calculation
  name) and the section-status pill.
* **Sidebar** (left): the section tree. Click a section to make it
  the active section (loads its binary data on first click). Active
  sections drive the main viewport.
* **Name header**: when the standalone `vibeqc_naming` engine is
  available, the sidebar starts with a non-clickable IUPAC name entry
  carrying its provenance and confidence icon. The source installer
  wires the checkout's `python/` directory onto the viewer environment
  path, so naming works in the standalone desktop app, and loose files
  opened on the fly (`.py` inputs, `.xyz`, ...) are named from their
  structure atoms just like on-disk QVF archives.
* **Viewport** (centre): 3D scene for structure / volume / vibrations
  / trajectory sections; interactive Plotly chart for bands /
  spectra; table for `atom_properties` and `scf_history`.

### Comparing multiple files

Open several archives at once (`vibe-view open a.qvf b.qvf ...`); a **Files**
dropdown in the top bar switches the active file. With more than one open, a
**Compare Files** card appears in the right panel: the *Overlay all structures*
switch draws every loaded structure together in one viewport, one translucent
colour per file, with a colour-keyed legend of file names. This is handy for
seeing how a geometry shifts between two relaxations, methods, or polymorphs.
Structures are shown at their stored coordinates; toggle *Align (RMSD fit to
first file)* to Kabsch-superpose each onto the first file (when the atom counts
match), with the per-file RMSD to that reference shown in the legend. Clicking
any section exits compare mode.

For two files that both carry a `volume.density` section on the same grid, the
**Density Difference** card renders ρ_A − ρ_B as a two-colour isosurface (red =
accumulation where A > B, blue = depletion) over file A's structure, handy for
deformation densities or before/after-adsorption maps.

### Structure section

Atoms are drawn as CPK spheres with element-coloured surfaces and
the standard van-der-Waals radii. Bonds come from the explicit
`bonds` section if the producer wrote one; otherwise vibe-view
infers them from covalent radii. Crystals show the unit-cell
wireframe.

```{figure} ../_static/plots/vibe_view/01-structure.png
The structure panel: CPK spheres (Jmol colours, Z ≤ 96), inferred or
explicit bonds, and the unit-cell wireframe for crystals.
```

Sidebar controls:

* **Replication** (periodic only): set Nx, Ny, Nz to replicate the
  cell along the lattice vectors. Atoms, isosurfaces, and the cell
  wireframe all expand in lock-step.
* **Atom radii / colours**: choose between CPK (default), van der
  Waals, and unit-radius schemes.

### Building & editing (edit mode)

Press `e` (or the pencil icon in the top bar) to toggle **edit
mode** on the structure section. Click an atom to select it, click
empty space to place an atom of the element chosen in the *Atom
Editor* card; the card also changes the element of the selection,
deletes it, inserts fragments (methyl, hydroxyl, phenyl, …),
saturates open valences with hydrogens, and offers undo/redo
(`Ctrl-Z` / `Ctrl-Y`). Edits compound in place; *Export vibe-qc
input (.py)* and vq submission always use the geometry as edited.

* **Auto-optimize** (Atom Editor card): while on, every edit pause
  relaxes the sketch in the background in a separate subprocess,
  streaming each optimizer step into the viewport so the atoms ease
  toward the relaxed geometry. The status line under the switch shows
  the running step / energy / max gradient, then `relaxed in N
  steps`. Editing again mid-relax cancels and reschedules; one undo
  entry per relax run returns to the sketch as drawn. Needs `vibeqc`
  importable in the viewer's environment: when it isn't, the switch
  snaps back off with the reason, and the viewer works exactly as
  before. Structures beyond 80 atoms are skipped to keep the
  background evaluations interactive.
* **Auto-optimize engines**: the default is vibe-qc's **MSINDO**
  semi-empirical model (analytic gradients, elements H through Xe).
  When vibe-qc's `[mace]` extra is installed (PyTorch stack, Python
  3.13 or older), an **Engine** picker appears offering **MACE**
  (the MIT-licensed MACE-MPA-0 foundation model, CPU): often better
  geometries, heavier per step, and the first use downloads the model
  weights (the status line says so). MACE energies are the model's
  reference-shifted DFT-surface values, not vibe-qc total energies;
  MACE also ignores charge and spin. The ASL-gated academic MACE
  models are not offered here.

### `volume.density`, `volume.orbital`, `volume.spin`, `volume.elf`, `volume.difference`, `volume.generic`, `volume.potential`

All volume kinds drive the same renderer: VTK marching cubes over
the grid `.dat` payload, producing an isosurface at the
configurable isovalue. Sidebar controls:

* **Isovalue**: slider over the volume's data range (default from
  `viewer_defaults`, falling back to a kind-specific heuristic:
  0.05 e/bohr^3 for densities, 0.04 bohr^(-3/2) for MOs).
* **Colour map** (signed-volume kinds): `volume.orbital`,
  `volume.spin`, `volume.difference`, and `volume.potential` (the
  electrostatic potential, signed in hartree/e) get a divergent map so the
  +/- lobes show in different colours.
* **Opacity**: 0 to 1 alpha.

Volume `.dat` blobs are **lazy-loaded**: nothing is read from the
zip until you activate the section. Large MO grids stay on disk
until you click them.

```{figure} ../_static/plots/vibe_view/02-density.png
A scalar-field isosurface (electron density) from VTK marching cubes,
drawn translucent over the structure. Signed fields such as orbitals, spin,
and difference densities render both ± lobes in contrasting colours.
```

#### Periodic torus grids and Wannier overlays

For periodic QVF archives, vibe-view draws `structure.lattice_vectors` as row
vectors in Angstrom and treats the volume grid as bohr-space data converted at
render time. If a grid spans the full Born-von Karman cell, periodic
replication makes adjacent density and orbital surfaces abut without a gap.
Localized periodic orbital grids can also be rolled to the cell centre with
the wrap-to-centre control; delocalized or already centred fields are left
unchanged.

The chi-CCM-B writer may add an `x_ccm.wannier_centers` vendor section. It is
not a sidebar panel of its own, but vibe-view recognizes it as an optional
gold marker overlay in the Display controls. The validated fixtures in the
{ref}`QVF tutorial <validated-periodic-chi-ccm-b-qvf-fixtures>` include
downloadable 3D periodic QVF archives plus headless captures:
[contact sheet](../_static/examples/chi-ccm-b-qvf/vibe-view-contact-sheet.png),
[H-chain QVF](../_static/examples/chi-ccm-b-qvf/output-chi-ccm-b-hchain-ri-n4-wannier.qvf),
and [H2-pair QVF](../_static/examples/chi-ccm-b-qvf/output-chi-ccm-b-h2pair-3d-ri-n2-wannier.qvf).

### `volume.rdg` (NCI analysis)

Non-covalent interaction (NCI) surface. vibe-view contours the reduced density
gradient s(r) at the standard isovalue (0.3) and colours the surface by
sign(λ₂)ρ (the second Hessian eigenvalue of the density times ρ), computed from
a co-present `volume.density` section. Colour convention: blue = attractive
(e.g. H-bonds), green = van der Waals, red = repulsive/steric, clamped to ±0.05
a.u. With no density section in the archive the RDG surface renders uncolored.
Method: Johnson et al., *JACS* **132**, 6498 (2010); NCIPLOT conventions,
Contreras-García et al., *JCTC* **7**, 625 (2011).

### `bands`

Interactive Plotly plot of every band along the k-path. The Fermi
level (from the `bands.fermi` field) is drawn as a horizontal
reference line. Hover any band to see its energy in eV at that
k-point. Energies are stored in eV in QVF v1 (see
[QVF spatial and energy units](../design_qvf_format.md#3-units-and-numeric-conventions)).

```{figure} ../_static/plots/vibe_view/08-bands-dos.png
The band structure and, when a `dos.total` section is also present, the
density of states, drawn as one figure on a shared, Fermi-referenced
energy axis.
```

### `phonon_bands`, `phonon_dos`

Lattice-dynamics panels for periodic systems. `phonon_bands` plots the
phonon dispersion (frequency along the Brillouin-zone q-path); `phonon_dos`
plots the phonon density of states. Both use cm⁻¹ on the frequency axis and
share the same 2D side panel as the electronic bands/DOS. Soft (imaginary)
modes are drawn below the zero-frequency line rather than hidden, so a
dynamical instability is visible at a glance. vibe-view plots the
producer-computed frequencies directly; cite your phonon method on the
producer side.

### `equation_of_state`

Volume-energy panel. The sampled (V, E) points are plotted as markers; the
producer's fitted curve (3rd-order Birch-Murnaghan or Murnaghan, per
`fit.model`) is overlaid by evaluating the published energy form at the fitted
V0/E0/B0/B0', and the equilibrium volume V0 is marked. The fitted bulk modulus
B0, B0', and the model reference (e.g. Birch 1947) are annotated in the panel.
Units follow spec §4.14 (Angstrom³ volumes, eV energies, GPa bulk modulus).

### `fermi_surface`

Fermi surface in **reciprocal space** (Å⁻¹). For each band near E_F the archive
stores E(k) − E_F on a γ-centered Monkhorst-Pack mesh; vibe-view builds the
reciprocal lattice from the real-space `lattice_vectors` and contours each band
at E = E_F, drawing one sheet per band (distinct colours) inside the
reciprocal-cell wireframe. Because this is k-space rather than real space, the
molecular structure is hidden while the Fermi surface is shown; click any
real-space section to bring the structure back. When more than one band crosses
E_F, a *Fermi Surface Bands* multi-select in the right panel toggles which
sheets are drawn, so an individual sheet can be isolated. Spec §4.12.

### `spectra.ir`, `spectra.uvvis`, `spectra.raman`, `spectra.ecd`, `spectra.vcd`, `spectra.nmr`, `spectra.generic`

Each spectrum renders as a Plotly stem chart of intensities vs
frequency, with hover tooltips showing the per-peak metadata. The
x-axis unit follows the spectrum kind (cm^-1 for IR / Raman, eV
for UV-Vis, ppm for NMR).

```{figure} ../_static/plots/vibe_view/09-ecd-spectrum.png
A 1-D spectrum (ECD shown) as a Plotly chart with hover tooltips.
Signed spectra (ECD / VCD) keep their negative Cotton bands.
```

### `trajectory` and `reaction.path`

Frame-by-frame animation with a play / pause / step button strip
underneath the viewport. Bonds are inferred per-frame from
covalent radii so they update with the geometry. A small energy
chart above the timeline plots `metadata.energies` vs frame index
for geometry-optimisation runs.

```{figure} ../_static/plots/vibe_view/anim-trajectory.gif
Frame-by-frame trajectory playback; the play / pause / step strip
drives the timeline and the energy chart tracks the current frame.
```

`reaction.path` has the same binary layout as `trajectory` but
additionally renders waypoint markers (reactant / TS / intermediate
/ product) on the timeline.

**Periodic reaction paths.** When the frames are ``PeriodicSystem`` instances
(slabs, surfaces, periodic NEB trajectories), the archive stays
``qvf_version: 1`` and carries optional per-frame lattice + dimensionality
metadata so the renderer can draw the unit cell and wrap atoms across periodic
boundaries. The earlier ``qvf_version: 2`` stamp for this case was withdrawn;
vibe-view still accepts those older archives as v1-compatible.

The schema additions are minimal: an optional ``lattice`` binary member on the
``reaction.path`` section (columns = a, b, c, in bohr, matching
``vibeqc.PeriodicSystem.lattice``) plus a ``dim`` integer in the metadata JSON.
A shared lattice across all frames stores once as shape ``[3, 3]``; per-frame
lattices store as ``[n_frames, 3, 3]`` (forward-compat with variable-cell
scans).

When a periodic ``reaction.path`` is activated, vibe-view draws the unit-cell
wireframe in the 3D scene (the same parallelepiped style the structure renderer
uses) and wraps every frame's atom positions into the central cell along the
first ``dim`` lattice vectors. For a slab (``dim=2``) that means a + b are
wrapped while the non-periodic c direction stays open. Variable-cell paths
(``[n_frames, 3, 3]`` lattices) re-emit the box per frame, so the cell animates
with the geometry.

### `vibrations`

Mode-selector dropdown to pick the normal mode (sorted by
frequency); the structure animates with the displacement vector
applied sinusoidally. Frequencies are in cm^-1.

```{figure} ../_static/plots/vibe_view/anim-vibration.gif
A normal mode animated about the equilibrium geometry. Mode labels
carry the IR intensity when a companion `spectra.ir` section is present.
```

### `atom_properties`

Tabular display of `mulliken_charge` / `loewdin_charge` /
`hirshfeld_charge` / `spin_population` (whichever the producer wrote). Color-coded
per-atom highlighting in the 3D viewport.

```{figure} ../_static/plots/vibe_view/05-atomic-properties.png
Atomic charges as a table plus a 3D overlay; "colour by charge" tints
each atom (red positive, blue negative). On replicated periodic cells
the tint follows every replica.
```

### `wavefunction.gto`

Browse the molecular orbitals as a list (with energies and
occupations) and click any MO to render its isosurface. vibe-view
resamples the orbital from the embedded GTO basis + MO coefficients
on a grid of its own choosing, so you can inspect any MO without
the producer having to pre-evaluate it. **Compute total density** sums
the occupied MOs (ρ = Σ occ_i |ψ_i|²) on the fly into a density isosurface
without needing a stored `volume.density` section; the status reports
∫ρ dV as an electron-count check.

### `scf_history`

Two-pane chart: energy vs iteration (top) and `|DIIS error|` vs
iteration (bottom). Hover any iteration to see the exact numbers.

### `structure.symmetry`

Compact info panel with the spglib output: space group name,
Hall number, international symbol, Wyckoff positions, equivalent
atoms.

```{figure} ../_static/plots/vibe_view/11-symmetry.png
The symmetry panel surfaces every key spglib reports for the structure.
```

### `citations`

The embedded BibTeX bundle is rendered as a copy-paste-friendly
block. Pairs with the [citations user guide](citations.md): the
viewer shows you what to cite, the producer wrote the database
entries.

```{figure} ../_static/plots/vibe_view/13-citations.png
The citations panel renders the embedded BibTeX bundle as a
copy-paste block.
```

## Running and monitoring jobs (vq)

vibe-view is also a cockpit for the vibe-queue (`vq`) job runner, not
just a file viewer. Open the **Job Manager** from the server icon
(`mdi-server-network`) in the app bar. It lists the local `vq` queue,
tracks job state, and opens finished results straight into the viewer:

* Each row shows a colour-coded **state chip** (`pending` grey,
  `running` blue, `suspended` amber, `completed` green,
  `failed`/`killed`/`interrupted` red), the job name and short id,
  **elapsed time**, and an **open** action that fetches a completed job
  and loads its `.qvf` files.
* A **queue-overview strip** summarizes the local queue at a glance:
  daemon health, host and `vq` version, capacity (max CPUs / concurrent
  jobs), and current load (running / pending CPUs).
* **Live monitor** (a switch) includes every state (queued, running,
  suspended) and auto-refreshes every 5 seconds, with a per-job status
  and stdout/stderr **log tail** so you can watch a job progress without
  leaving the viewer.
* **Submit from the viewer**: the submit dialog generates a vibe-qc
  input script from the current structure and submits it through `vq`
  with a job name and tag.

### Live reload

Flip **Auto-reload on file change** (the Display card in the left panel)
and vibe-view watches the open file and hot-reloads it whenever its
content changes. The watcher is content-based, not timestamp-based: it
fingerprints the QVF manifest per section, waits for the file to settle,
holds off on a half-written archive, and then reloads only what actually
moved: the active panel alone for a change confined to a 2D-panel
section, or the full scene (your camera preserved, the active section
restored) for anything structural. If you are parked on the last frame
of a growing trajectory, the reload keeps you on the head as new frames
arrive.

### Streaming a running job

A vibe-qc job started with `checkpoint_qvf=...` rewrites a live QVF
snapshot as it runs, carrying `provenance.run_status`, a monotonic
`provenance.checkpoint`, and `partial` on still-growing sections (see
[the streaming section of the QVF reference](../consumer_qvf_reference.md)).
vibe-view surfaces that stream:

* In the Job Manager, a running job that has written a checkpoint shows
  a **Watch live** action (the eye icon). Clicking it opens the job's
  checkpoint QVF and turns auto-reload on for you (no need to know the path).
  Jobs submitted from the viewer stream a checkpoint by default, so a
  job you just launched is immediately watchable.
* An **app-bar chip** reflects the job status: it pulses **running**
  (with `#seq · iter N · E … Eh` from the latest checkpoint), then
  settles green **converged** or red **failed** on the terminal
  snapshot, and the status bar announces the finish.
* A section the producer is still growing (an optimization
  `trajectory`, SCF history mid-run) is badged **streaming** in the
  sidebar and reloads in place as it grows.

When `run_status` leaves `running`, the checkpoint you are already
watching *is* the settled result, so no separate fetch is needed.

## Exporting animations to video

`vibe-view animate` renders an animatable section to MP4 (needs
`ffmpeg` on `PATH`) or GIF (needs only Pillow):

```sh
vibe-view animate opt.qvf                            # auto-detect the section
vibe-view animate neb.qvf --kind reaction            # NEB band: reactant -> TS -> product
vibe-view animate vib.qvf --kind vibration --mode 7  # one normal mode
vibe-view animate opt.qvf -k trajectory --fps 8 --format gif
```

`--kind auto` (the default) picks the first of `trajectory`,
`reaction.path`, `vibrations`, or `wavefunction.gto` it finds. The same
renderers are importable (`render_trajectory_video`,
`render_reaction_video`, `render_vibration_video`, and
`render_orbital_animation` in `vibeview.animation`), each returning the
output path (or `None` when the archive has no matching section):

```python
from vibeview.qvf import QVFReader
from vibeview.animation import render_reaction_video

render_reaction_video(QVFReader("neb.qvf"), "neb.mp4", fps=5, format="mp4")
```

(viewer-defaults-producer-side-hints)=
## `viewer_defaults`: producer-side hints

The producer can suggest defaults to the viewer in the manifest
under `viewer_defaults`. vibe-view picks these up on load and
applies them before rendering anything.

```python
# doc-audit: skip - manifest fragment for a surrounding write_qvf call
write_qvf(
    "water.qvf",
    ...,
    viewer_defaults={
        "auto_open": ["density"],          # activate the density section by default
        "density": {                       # per-section render hints
            "isovalue": 0.04,
            "colormap": "viridis",
            "opacity": 0.55,
        },
        "bookmarks": [                     # camera bookmarks
            {"name": "front",  "camera": {...}},
            {"name": "above",  "camera": {...}},
        ],
    },
)
```

Hints are non-binding: the user can override any of them through
the UI, and unknown viewer_defaults fields are ignored rather than
rejected.

## Supported kinds

The viewer's renderer registry lives at
`src/vibeview/kinds.py::SUPPORTED_KINDS`. The full
writer / viewer support matrix is in the
[QVF design doc § 4](../design_qvf_format.md#4-canonical-section-kinds).
Anything not in the registry is classified as "skipped,
unsupported" by the viewer and listed in the banner. The viewer
never aborts on an unknown kind; it just leaves that sidebar entry
unclickable.

If you write a custom kind under the `x_<vendor>.*` namespace, the
viewer reports it as "skipped, vendor namespace" with the vendor
name extracted. To get a kind rendered, add it to
`SUPPORTED_KINDS` and wire a matching renderer under
`src/vibeview/renderers/`. A small number of known vendor
sections may still feed overlays on another panel; for example,
`x_ccm.wannier_centers` appears as vendor metadata in the banner but drives
the optional Wannier-centre markers on the structure/volume view.

## Common pitfalls

### "vibe-view: command not found"

Install the separate viewer with the source recipe above, then activate its
`.venv` so the `vibe-view` command is on PATH. For a combined Python process,
use [Install both](../getting_started.md#install-both).

### "ManifestValidationError"

The archive's `manifest.json` does not satisfy the canonical JSON
Schema at `python/vibeqc/output/formats/qvf_manifest.schema.json`.
This usually means the producer is older than the viewer or vice
versa. Both implementations carry schema copies checked against the published QVF
contract; the viewer no longer links to a core source file with a symlink; if the archive was
produced by a third party, ask them to validate with
[`validate_qvf`](output_files.md) before sharing.

### "SHA256MismatchError"

A binary payload in the archive has a different sha256 than the
manifest claims. Usually means the archive was edited after
writing (zip-shuffled, copied truncated, …). The viewer reports
this per-section: other sections still load.

### "the browser opened but the page is blank"

Trame uses Vue 3 + WebSocket. A very strict ad-blocker or
corporate proxy can break the WebSocket handshake. Either
allow the local page in the blocker or try another browser profile. If the
viewer runs on another machine, use the SSH-tunnel route in the
[standalone tutorial](../tutorial/vibe_view_getting_started.md#platform-notes).
Changing the bind address does not repair a broken local WebSocket and can
expose the unauthenticated server.

### Firefox on macOS 15+ "Unable to connect to 127.0.0.1"

macOS 15 (Sequoia) added a system-level **Local Network**
permission gate that is denied to Firefox by default. The
symptom is unmistakable: Safari and Chrome connect to
`http://127.0.0.1:8080` fine, but Firefox shows "Unable to
connect" with a hint about Local Network permissions in
macOS Privacy & Security. Two fixes:

* **Grant Firefox the permission.** System Settings →
  Privacy & Security → Local Network → toggle Firefox on.
  Refresh the page. (vibe-view does not need elevated
  permissions itself; only the browser does.)
* **Or use a different browser.** Safari and Chrome are
  unaffected because they were grandfathered into the
  permission system. Open `http://127.0.0.1:8080` there
  while the vibe-view server keeps running in the terminal.

vibe-view itself binds to `127.0.0.1` only by default; nothing
on the server side needs reconfiguring. If you launched with
`--host 0.0.0.0` for remote access the same Local Network
gate applies to any LAN browser hitting the box, not just
Firefox.

### "vibe-view server starting on ..." printed but the browser cannot connect for a few seconds

vibe-view announces "ready" only once uvicorn has actually
bound the TCP port (the watcher polls the listener in a
background thread, so the "ready" line and the automatic
browser-open happen post-bind). If you set `--no-browser`
and open the URL by hand immediately after launching, you
may still race the bind on slow machines; refresh once and
you should connect. If "ready" never appears, see the next
entry.

### "vibe-view: warning, server did not bind within 10s"

uvicorn entered the asyncio main loop but never made it to
binding the TCP socket. Two common causes:

* **Port already in use.** Another process is bound to the
  same port (a stale vibe-view, an aborted SSH tunnel, Docker
  Desktop, …). `vibe-view open` now normally catches this
  *before* launch and aborts with a "port … is already in use"
  message instead (see the entry below); the 10 s warning only
  remains for the rarer case where the socket is held in a state
  the pre-check misses. Either way, pick a different port with
  `--port 9876` and retry, or find the offender:

  ```sh
  lsof -nP -iTCP:8080 -sTCP:LISTEN
  ```

* **uvicorn import-time failure.** A missing dependency in
  the venv (the historical case was the now-fixed missing
  `uvicorn` declaration in vibe-view's pyproject; reinstall
  if you see a `ModuleNotFoundError` in the stderr just
  before the warning).

### "vibe-view: port 8080 is already in use"

`vibe-view open` pre-checks the port and stops with this message
when something is already listening there, most often a vibe-view
you left running in another terminal (servers stay up until you
press Ctrl+C; they have no idle timeout). Stop the other server, or
start this one on a free port:

```sh
lsof -nP -iTCP:8080 -sTCP:LISTEN   # find what owns the port
pkill -f "vibe-view open"           # …or just stop stale vibe-view servers
vibe-view open water.qvf --port 8090
```

This guard exists because, before it, `open` would connect to the
existing listener, announce "ready", and open your browser at the
**stale** server while the new one died unseen; see the next entry.

### The viewer shows a structure but the mouse does nothing (can't rotate / zoom)

Almost always a **stale server**, not a viewer bug; rotation, zoom,
and pan work in Chrome, Safari, and Firefox. Because vibe-view servers
run until Ctrl+C, a browser tab can outlive the server behind it (you
stopped it, your machine slept, or you re-launched and the new process
could not bind the busy port). The tab keeps showing the *last rendered
frame* (the structure looks fine) but its WebSocket is gone, so
nothing responds to the mouse. Start fresh and use the new tab:

```sh
pkill -f "vibe-view open"     # stop any stale servers
vibe-view open water.qvf       # interact with the tab this one opens
```

### Large MO sets making the sidebar slow

vibe-view loads MO metadata eagerly but MO `.dat` blobs lazily.
If you wrote dozens of orbitals as separate `volume.orbital`
sections (one per occupied MO), the sidebar tree is still fast
because the binaries are not read until a click. Producers with
many orbitals should prefer one `wavefunction.gto` section over
N `volume.orbital` sections; the viewer resamples MOs on demand.

## Programmatic API

vibe-view's API is intentionally minimal. Most users want
`launch_qvf`:

```python
from vibeview import launch_qvf, QVFReader, QVFOpenError

# One-line launch
launch_qvf("water.qvf")

# In-memory archive (no temp file).
def launch_generated_archive(qvf_bytes: bytes) -> None:
    import io

    launch_qvf(io.BytesIO(qvf_bytes))

# Inspect first, then launch
try:
    reader = QVFReader("water.qvf")
except QVFOpenError as e:
    print(f"bad archive: {e}")
else:
    print(reader.sections)             # list of Section objects
    print(reader.source.version)       # producer version
    launch_qvf(reader)                 # reuses the open reader
```

`QVFReader` accepts a path, raw zip bytes, or any seekable binary
file-like (`BytesIO`, opened file). It validates the manifest at
construction time and lazy-loads section payloads on demand.

### Build a structure from a SMILES string

`smiles_to_qvf` turns a SMILES string into an in-memory QVF you can hand
straight to the viewer, which is the quickest way to get a molecule on screen
without a calculation or a structure file:

```python
from vibeview import launch_qvf
from vibeview.converters import smiles_to_qvf

launch_qvf(smiles_to_qvf("c1ccccc1"))          # benzene
launch_qvf(smiles_to_qvf("CCO", add_hydrogens=False))
```

It needs the `[smiles]` extra, which pulls RDKit. Select the `all` profile in
the checkout installer or install the downloadable wheel with its `[smiles]`
extra. The distribution name is `vibeview`, but it is not on PyPI yet; see
[Getting started with vibe-view alone](../tutorial/vibe_view_getting_started.md)
for installable sources.

```{important}
The geometry is an **RDKit ETKDG embedding, with no force-field cleanup**.
That is a chemically sensible starting structure, not an optimised one, and it
is not a substitute for a calculation. Treat it as a starting point: hand it to
vibe-qc, or to the viewer's live optimisation, before reading anything
quantitative off it.
```

This is a Python entry point today. There is no `vibe-view` subcommand for it
and no SMILES file format the open dialog accepts, so `smiles_to_qvf` is the
only route.

The full reading-side API for callers who want to consume `.qvf`
archives outside vibe-view is in the
[QVF consumer reference](../consumer_qvf_reference.md).

## See also

* [Biomolecules](vibe_view_biomolecules.md): cartoon / ribbon rendering,
  chains, residues, secondary structure and b-factors, plus how the metadata
  gets into a `.qvf` in the first place.
* [vq Job Manager](vibe_view_vq_jobs.md) and
  [Live reload](vibe_view_live_reload.md): submitting, monitoring and watching
  a running job from the viewer.
* [QVF format design](../design_qvf_format.md): the producer / consumer
  contract, manifest schema, per-section payload conventions.
* [QVF consumer reference](../consumer_qvf_reference.md): the
  reading-side Python API for callers that want to ingest `.qvf`
  archives without launching vibe-view.
* [vibe-view walkthrough](../tutorial/vibe_view_walkthrough.md):
  short end-to-end example that runs a calc, opens the result in
  vibe-view, and walks through what each panel does.
* [reading a `.qvf` in the terminal, over SSH](../tutorial/vibe_view_terminal.md):
  worked example of vibe-view's own terminal mode, reading the same archive
  as the walkthrough above without a browser or a GL context.
* [moltui terminal viewer](../tutorial/moltui_terminal_viewer.md):
  a format-agnostic terminal viewer for loose `.molden` / `.cube` / `.xyz`
  files from any code, where terminal mode reads `.qvf` only.
* [Output files reference](output_files.md): the full family of
  artefacts vibe-qc emits alongside the `.qvf`.
* Background, [*QVF v1 grows up: basis sets, full reactions, and a
  reference viewer*](https://vibe-qc.com/2026/05/25/qvf-v1-basis-reactions-vibe-view/)
  (vibe-qc.com, May 2026): the post that introduced vibe-view, with its
  companion [*Quantum chemistry needs a modern file
  format*](https://vibe-qc.com/2026/05/21/qc-needs-a-modern-file-format-qvf/).
