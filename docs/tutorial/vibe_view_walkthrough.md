# vibe-view: an end-to-end walkthrough

Viewer lifecycle commands on this page run from the **separate vibe-view
checkout**. Run calculation inputs with the core environment and give the
viewer the resulting QVF path; the two checkouts need not be adjacent.
Clone [mpei/vibe-view](https://github.com/vibe-qc/vibe-view) and follow
[viewer setup](vibe_view_getting_started.md). Its `.venv`, `scripts/`, and `electron/` belong to that repository;
vibe-qc does not contain or install the viewer.

**You will learn:** how to produce a `.qvf` archive from a routine
vibe-qc job, launch [vibe-view](../user_guide/vibe_view.md), and
inspect every section the calculation wrote: structure, electron
density, the HOMO / LUMO orbitals, the optimisation trajectory, the
IR spectrum, and the per-job citation bundle.

**Prerequisites:**

* `vibeqc` installed and a working SCF setup (any tutorial under
  [Fundamentals](index.md) shows this).
* `vibe-view` installed from its own repository. Use separate engine and viewer
  environments, or follow [Install both](../getting_started.md#install-both)
  when a single Python process needs both. VTK is about 200 MB, so the first
  install can take a minute.

* A working modern browser (Chrome, Firefox, Safari, Edge). vibe-view
  uses Trame + WebSocket; very strict ad-blockers can interfere.

Time to complete: about 5 minutes.

```{figure} ../_static/plots/vibe_view/01-structure.png
vibe-view in the browser: the section sidebar on the left, the
GPU-accelerated 3D viewport in the centre, and the source/provenance
banner along the top.
```

## 1. Produce a `.qvf`

Anything in vibe-qc that runs through `run_job` or `run_periodic_job`
emits a `.qvf` by default. Here is a small
RKS / PBE / 6-31G* water calculation with everything that vibe-view
knows how to render:

```python
# input-water-vibe-view.py
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
    optimize=True,                 # writes a trajectory section
    hessian=True,                  # writes vibrations + IR spectrum
    output="water",
    output_qvf=True,               # default, kept explicit for this tutorial
    write_cube=["density", "homo", "lumo"],  # embeds volumetric grids
    write_molden_file=True,        # embeds wavefunction.gto + MO blobs
)
```

Run it (a few seconds on any modern laptop):

```sh
~/path/to/vibeqc/.venv/bin/python input-water-vibe-view.py
```

Output:

```
water.out          .  banner + SCF trace + geometry opt log
water.molden       .  MO coefficients (Avogadro / Jmol)
water.traj         .  ASE trajectory (ase gui water.traj)
water.bibtex       .  auto-assembled BibTeX entries
water.references   .  Chicago-style reference list
water.system       .  TOML manifest with hardware + library versions
water.qvf          .  what we will open in vibe-view in a second
```

Have a peek at the manifest to confirm the sections vibe-view will
render:

```python
import zipfile, json
manifest = json.loads(zipfile.ZipFile("water.qvf").read("manifest.json"))
for s in manifest["sections"]:
    print(f"  {s['id']:<15s} {s['kind']}")
```

Typical output:

```
  structure       structure
  density         volume.density
  homo            volume.orbital
  lumo            volume.orbital
  wavefunction    wavefunction.gto
  traj0           trajectory
  vib             vibrations
  ir              spectra.ir
  citations       citations
  scf_history     scf_history
```

## 2. Launch vibe-view

Point the `vibe-view open` subcommand at the archive you just wrote;
it reads the sections out of `water.qvf` and starts the viewer:

```sh
vibe-view open water.qvf
```

Two things happen:

1. **Startup banner** prints to stdout (the same table you printed
   from the manifest above, with per-section render status):

   ```
   ╔══════════════════════════════════════════════════════════════════════════════╗
   ║  QVF file: water.qvf                                                         ║
   ║  Source:   vibe-qc <version> - water                                         ║
   ╠══════════════════════════════════════════════════════════════════════════════╣
   ║  structure          structure                    rendered                    ║
   ║  density            volume.density               rendered                    ║
   ║  homo               volume.orbital               rendered                    ║
   ║  lumo               volume.orbital               rendered                    ║
   ║  wavefunction       wavefunction.gto             rendered                    ║
   ║  traj0              trajectory                   rendered                    ║
   ║  vib                vibrations                   rendered                    ║
   ║  ir                 spectra.ir                   rendered                    ║
   ║  citations          citations                    rendered                    ║
   ║  scf_history        scf_history                  rendered                    ║
   ╚══════════════════════════════════════════════════════════════════════════════╝
   ```

2. **The default browser opens** at `http://127.0.0.1:8080` with the
   3D viewport. The server stays in the foreground; press Ctrl+C in
   the terminal to stop it.

Tips:

```sh
vibe-view open water.qvf --port 9876        # different port
vibe-view open water.qvf --no-browser       # do not auto-open the browser
vibe-view open water.qvf --host 0.0.0.0     # bind to all interfaces (remote use)
```

Because the server runs until you press Ctrl+C, re-running `vibe-view
open` while an earlier viewer is still up stops with `port 8080 is
already in use` (rather than opening your browser at the stale server);
stop the first one (`pkill -f "vibe-view open"`) or pass `--port`.
Likewise, if a tab that *was* working stops responding to the mouse, the
server behind it has been stopped; start a fresh one and use its new
tab. See [Common pitfalls](../user_guide/vibe_view.md#common-pitfalls).

## 3. Walk through what each panel does

The browser window has three regions:

* **Top bar**: source banner (program + version + calculation name)
  and the active-section pill.
* **Sidebar** (left): every section from the manifest, classified as
  rendered / skipped / error. Click any section to make it active.
* **Viewport** (centre): 3D scene for structure / volume /
  trajectory / vibrations, interactive Plotly for bands / spectra,
  table for `atom_properties` / `scf_history` / `citations`.

### Structure

The first section that auto-activates is `structure`. You see the
H2O molecule with CPK-coloured atoms (red oxygen, white hydrogens)
and bonds drawn from covalent radii. Rotate with left-drag, pan
with middle-drag, zoom with the scroll wheel.

```{figure} ../_static/plots/vibe_view/14-structure-h2co.png
The structure panel -- CPK-coloured atoms and covalent bonds in the 3D
viewport (formaldehyde, from the bundled showcase archive).
```

### Electron density

Click **`density`** in the sidebar. vibe-view marches the cubes at
the default isovalue of `0.05 e/bohr^3` and renders the resulting
isosurface translucent over the structure. A sidebar control lets
you change the isovalue (lower values give a larger lobe; higher
values squeeze in toward the nuclei).

This is the right place to see what the SCF actually converged on.
For water the lone-pair region behind the oxygen should be the
largest pocket of density.

```{figure} ../_static/plots/vibe_view/02-density.png
An electron-density isosurface, rendered translucent over the
structure so you can see both at once.
```

### Molecular orbitals (HOMO + LUMO)

Click **`homo`**. vibe-view marches the cubes on the orbital `.dat`
payload (real-valued for closed-shell molecular MOs). Because MOs
have signed lobes the viewer uses a divergent colormap by default,
so you can see the `+` and `-` regions. Set the isovalue to ~0.04
bohr^(-3/2) for the standard "balloon" view of an HF-style HOMO.

```{figure} ../_static/plots/vibe_view/15-orbital-homo.png
A molecular orbital isosurface. Because an MO is a signed field, both
lobes are drawn in contrasting colours (blue = positive, red =
negative) -- for stored `volume.orbital` sections as well as the
on-demand `wavefunction.gto` evaluation.
```

Click **`lumo`** to see the LUMO instead. Both stay loaded; the
viewport always shows the active section's volume.

### The full MO set via `wavefunction.gto`

For more orbitals than just HOMO / LUMO, click **`wavefunction`**.
vibe-view shows a list of every MO (with energy and occupation) and
re-samples each MO from the basis + coefficients when you click it.
This is cheaper to produce than writing every MO as a separate
`volume.orbital` section.

### Geometry-optimisation trajectory

Click **`traj0`**. The viewport switches to a frame-by-frame
animation of the geometry-optimisation steps. A play / pause / step
strip below the viewport drives the timeline; a small energy chart
above it plots the per-frame energies (in Hartree) so you can see
the SCF energy decrease monotonically.

For our water optimisation there are only a few frames because the
input geometry was already near the minimum; the energy chart drops
by a fraction of a mHa and the structure barely moves.

```{figure} ../_static/plots/vibe_view/anim-trajectory.gif
Trajectory playback steps through the optimisation frames; the
play / pause / step strip drives the timeline.
```

### Vibrational modes

Click **`vib`**. The viewport renders the equilibrium geometry; a
dropdown selector lets you pick a normal mode by frequency (in
cm^-1). The atoms oscillate along the displacement vector with the
mode's amplitude.

For water at RKS/PBE/6-31G* you should see three modes corresponding to the
bend, symmetric stretch, and antisymmetric stretch. Exact harmonic frequencies
depend on the method, basis, optimized geometry, and numerical settings.

```{figure} ../_static/plots/vibe_view/anim-vibration.gif
A selected normal mode, animated: the atoms oscillate along their
(un-mass-weighted, Cartesian) displacement vectors.
```

### IR spectrum

Click **`ir`**. The viewport switches to an interactive Plotly stem
chart of intensities vs frequency. Hover any peak to see the exact
frequency and intensity (in km / mol). The chart respects the
viewer-defaults x-axis range if the producer set one.

### Citations

Click **`citations`** at the bottom of the sidebar. The viewport
shows the embedded BibTeX bundle as a copy-paste block: every paper
the run cites (vibe-qc itself, libint, libxc, PBE, the basis set,
plus any dispersion / SCF-accelerator references). This is the same
bundle written to `water.bibtex` on disk; see the
[citations user guide](../user_guide/citations.md) for the format
spec.

```{figure} ../_static/plots/vibe_view/13-citations.png
The citations panel renders the embedded BibTeX bundle as a
copy-paste block, ready for a paper's reference list.
```

### SCF history

Click **`scf_history`**. Two stacked Plotly panes:

* Top: SCF energy vs iteration (Hartree).
* Bottom: `|DIIS error|` vs iteration (log scale).

Hover any iteration to see the exact numbers. This is the same
information vibe-qc writes to `water.out` as a text table, presented
as a chart you can zoom and pan.

```{figure} ../_static/plots/vibe_view/12-scf-convergence.png
The SCF-history panel: total energy (top) and the DIIS error on a log
axis (bottom), per iteration.
```

## 4. Periodic example: MgO bands

Periodic systems work the same way. This example uses a simple-cubic,
two-site MgO teaching cell for an RHF calculation and embeds a separately
computed Hcore band model:

```python
# input-mgo-vibe-view.py
import numpy as np
import vibeqc as vq

a = 4.21 / 0.529177                                       # Angstrom to bohr
mgo = vq.PeriodicSystem(
    3,
    np.eye(3) * a,
    [vq.Atom(12, [0.0, 0.0, 0.0]),
     vq.Atom(8,  [a/2, a/2, a/2])],
)
basis = vq.BasisSet(mgo.unit_cell_molecule(), "sto-3g")

# Compute a band structure along a high-symmetry path. Each segment
# is (start_frac, start_label, end_frac, end_label).
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

The atom at the body center makes this a CsCl-type teaching cell, not rocksalt
MgO. The embedded bands diagonalize Hcore only and must not be interpreted as
the RHF spectrum. Use a route-specific converged-operator band helper for a
scientific calculation.

```sh
vibe-view open mgo.qvf
```

Two extra sidebar entries appear compared to the molecular case:

* **`structure`** now shows a unit-cell wireframe. Sidebar gains a
  replication control (Nx, Ny, Nz) to extend the cell along the
  lattice vectors. Atoms, the cell box, and any active isosurface
  all expand together.
* **`bands`** opens the Plotly band plot: every band's eigenvalue
  along the k-path, with a horizontal line at the Fermi energy.
  Hover any band to see the energy in eV.

```{figure} ../_static/plots/vibe_view/16-periodic-structure.png
A periodic crystal: the unit-cell wireframe is drawn automatically, and
the replication controls tile the cell along the lattice vectors.
```

```{figure} ../_static/plots/vibe_view/08-bands-dos.png
When the archive carries both a `bands` and a `dos.total` section,
vibe-view draws them as one figure on a shared, Fermi-referenced energy
axis -- the standard solid-state plot.
```

```{figure} ../_static/plots/vibe_view/17-periodic-dos.png
The density of states for a periodic crystal (NaCl), referenced to the
Fermi level at 0 eV -- Cl-3p valence below, Na conduction above the gap.
```

## 5. Tips and gotchas

The sections below cover producer-side `viewer_defaults` hints,
launching from an in-memory archive without touching disk, viewing a
remote run over SSH, and what to do when the viewer reports a section
as unsupported.

### Producer-side hints with `viewer_defaults`

To make a `.qvf` open with the right defaults (which section to
activate, isovalue, colormap, camera bookmarks), pass
`viewer_defaults=` to the QVF writer or build it inside
`run_job` via the output-plan API. vibe-view picks up the hints
on load. See the {ref}`viewer_defaults section of the user guide
<viewer-defaults-producer-side-hints>`.

### In-memory QVF (skip the disk round-trip)

If you're scripting vibe-view from a notebook and do not want a
`.qvf` on disk:

```python
import io
from vibeview import launch_qvf

# Build the archive in memory.
buf = io.BytesIO()
# ... use the QVF writer with `buf` as the path ...

buf.seek(0)
launch_qvf(buf, open_browser=False)
```

`launch_qvf` accepts any seekable binary file-like, not just paths.

### Remote viewing over SSH

The simplest answer is not to move the archive or forward anything at all:
vibe-view renders into the terminal you are already sitting in.

```sh
ssh remote
vibe-view show run.qvf      # one frame, then exit
vibe-view tui  run.qvf      # interactive
```

```{figure} ../_static/plots/vibe_view/tui-01-structure.svg
`vibe-view tui run.qvf` over a plain SSH session. Section browser on the
left, the structure drawn with Unicode braille in the centre, and a status
line naming what you are looking at. No browser, no OpenGL, no X
forwarding: the rasterizer is pure NumPy.
```

`vibe-view show` runs with the core profile; `vibe-view tui` needs the `tui`
profile. Install it from the repository root with
`./scripts/install.sh --extras tui`, or refresh an existing viewer
environment with `./scripts/update.sh --skip-git --extras tui`. The
terminal renderer reads the same archive and uses the same element colours
and radii as the browser viewer, so the two agree on what you are looking
at. It covers structures, isosurfaces, trajectories and normal modes in 3D;
bands, DOS, spectra and SCF traces as charts; and citations, run records,
and charges as tables. A `wavefunction.gto` section is interactive rather
than read-only: it renders a default orbital, opens a selectable surface
table on the right, and lets you render canonical, alpha/beta, natural, or
localized rows with <kbd>Up</kbd>/<kbd>Down</kbd> plus <kbd>Enter</kbd>.
<kbd>D</kbd> computes total density when the archive declares electron
occupations; natural transition weights and ambiguous legacy natural values
are refused. <kbd>S</kbd> computes spin density
only for unrestricted sets. See
[reading a `.qvf` in the terminal, over SSH](vibe_view_terminal.md).

If you specifically want the *browser* UI against a remote run, two options:

1. Forward the port: `ssh -L 8080:127.0.0.1:8080 remote`, then run
   `vibe-view open run.qvf` on the remote and open
   `http://127.0.0.1:8080` in your local browser.
2. Bind vibe-view to the public interface: `vibe-view open run.qvf
   --host 0.0.0.0 --no-browser`. Only do this on a trusted network;
   the Trame server has no authentication.

For loose `.molden` / `.cube` / `.xyz` files, or output from a code other
than vibe-qc, [moltui](moltui_terminal_viewer.md) is the format-agnostic
terminal viewer; terminal mode reads `.qvf` only.

### vibe-view says "skipped, unsupported" on a section I care about

The kind is not yet in `src/vibeview/kinds.py::SUPPORTED_KINDS`.
Either:

* The producer wrote a kind under the `x_<vendor>.*` namespace,
  which the viewer never tries to render by design (the namespace
  is reserved for producer-specific extensions).
* The kind is implemented on the writer but not yet on the viewer.
  The full matrix is in the
  [QVF design doc § 4](../design_qvf_format.md#4-canonical-section-kinds).
  Adding a renderer is a three-step contribution: add the kind to
  `SUPPORTED_KINDS`, drop a renderer module in
  `src/vibeview/renderers/`, and wire the activation
  branch in `src/vibeview/app.py`. The existing renderers
  are short (~100 to ~200 lines each) and serve as templates.

## Next

* [vibe-view user guide](../user_guide/vibe_view.md): full
  reference (install routes, programmatic API, every sidebar
  control).
* [QVF design](../design_qvf_format.md): the producer / consumer
  format contract.
* [QVF consumer reference](../consumer_qvf_reference.md): read
  `.qvf` files programmatically without launching the viewer.
* [moltui](moltui_terminal_viewer.md): the
  terminal-only sibling viewer (XSF / BXSF / Molden) for SSH
  contexts without browser access.
* [auto citations](auto_citations.md): how the
  `citations` section vibe-view rendered in step 3 is assembled.
* Background reading: [*QVF v1 grows up: basis sets, full reactions, and a
  reference viewer*](https://vibe-qc.com/2026/05/25/qvf-v1-basis-reactions-vibe-view/)
  (vibe-qc.com) introduces vibe-view as QVF's reference viewer.
