# Terminal mode

Source commands on this page run from the **separate vibe-view checkout**.
Clone [vibe-qc/vibe-view](https://github.com/vibe-qc/vibe-view) and follow
[viewer setup](../tutorial/vibe_view_getting_started.md). Its `.venv`, `scripts/`, and `electron/` belong to that repository;
vibe-qc does not contain or install the viewer.

`vibe-view tui` and `vibe-view show` render QVF archives as **text**: a full
3D viewer, band structures, spectra and data tables drawn with Unicode braille
characters and 24-bit colour, inside any terminal.

The point is where it runs. The interactive web viewer (`vibe-view open`) and
the headless PNG capture (`vibe-view capture`) both drive VTK through an
OpenGL context. A compute node reached over SSH usually has no display server,
no GL, and no X forwarding, and that node is exactly where your `.qvf` lands.
Terminal rasterization is pure NumPy: it paints spheres, bonds, isosurface
triangles and plot lines into a depth-buffered pixel array, then packs that
array into braille glyphs. Isosurface extraction uses VTK computational
filters, without creating a render window or OpenGL context.

```sh
ssh compute-reference
vibe-view show ~/scratch/job.qvf          # one frame, then exit
vibe-view tui  ~/scratch/job.qvf          # interactive
```

## Two commands

| | `vibe-view show` | `vibe-view tui` |
|---|---|---|
| Output | one frame to stdout, then exits | interactive full-screen app |
| Needs | core profile, with no display or GL context | the `tui` profile (adds Textual) |
| Good for | pipes, CI logs, scripts, a quick look | browsing an archive section by section |

From the repository root, install the terminal profile in its dedicated
environment:

```sh
./scripts/install.sh --extras tui
source .venv/bin/activate
```

For an existing checkout installation, refresh that same profile with:

```sh
./scripts/update.sh --skip-git --extras tui
```

`vibe-view show` needs nothing beyond the core profile. The terminal renderer
itself is a NumPy software rasterizer, but the base distribution also carries
the shared archive and plotting dependencies; "no display" does not mean a
NumPy-only package install. vibeview is not currently published on PyPI, so a
bare `pip install 'vibeview[tui]'` does not resolve.

## `vibe-view show`

```sh
vibe-view show job.qvf                          # the structure
vibe-view show job.qvf --section vol_mo_3       # a specific section
vibe-view show job.qvf -s vol_mo_3 --isovalue 0.03
vibe-view show job.qvf --info                   # provenance + section inventory
vibe-view show job.qvf --all --plain > frames.txt
```

Useful options:

- `--size COLSxROWS`, fix the character grid instead of asking the terminal.
- `--mode braille|half`, see [Rendering modes](#rendering-modes).
- `--plain`, no colour, plain characters. Also honours `NO_COLOR`.
- `--representation ball_and_stick|licorice|spacefill|wireframe|points|backbone`
- `--color-by element|chain|secondary|bfactor`
- `--rotate X,Y,Z`, Euler angles in degrees.
- `--replicate nx,ny,nz`, supercell view; only periodic axes replicate.
- `--labels`, overlay atom indices.
- `--frame N`, pick a frame of a trajectory, reaction path or normal mode.
- `--chart`, for a reaction path or trajectory, draw the energy profile
  instead of the geometry.

Colour escapes survive a pipe, so `vibe-view show job.qvf | less -R` keeps the
shading. Pass `--plain` when you want text a log file can hold.

## `vibe-view tui`

The interactive viewer opens on the structure with a section browser on the
left, the viewport in the middle, and a status line naming what you are
looking at. Press <kbd>?</kbd> for the key map.

### Keys

| | |
|---|---|
| <kbd>Tab</kbd> / <kbd>Shift</kbd>+<kbd>Tab</kbd> | next / previous section |
| arrows or <kbd>h</kbd> <kbd>j</kbd> <kbd>k</kbd> <kbd>l</kbd> | rotate |
| <kbd>H</kbd> <kbd>J</kbd> <kbd>K</kbd> <kbd>L</kbd> | pan |
| <kbd>+</kbd> / <kbd>-</kbd> | zoom |
| <kbd>r</kbd> | reset the camera and re-fit |
| <kbd>m</kbd> | cycle representation |
| <kbd>c</kbd> | cycle colour scheme |
| <kbd>b</kbd> / <kbd>u</kbd> / <kbd>#</kbd> | bonds / unit cell / atom indices |
| <kbd>d</kbd> | switch braille to half-block and back |
| <kbd>x</kbd> <kbd>y</kbd> <kbd>z</kbd> (<kbd>X</kbd> <kbd>Y</kbd> <kbd>Z</kbd>) | replicate (un-replicate) along a, b, c |
| <kbd>n</kbd> / <kbd>p</kbd> | next / previous stored volume, or next / previous orbital while a wavefunction is active |
| <kbd>i</kbd> / <kbd>I</kbd> | isovalue down / up |
| <kbd>o</kbd> | isosurface on / off |
| <kbd>Up</kbd> / <kbd>Down</kbd>, then <kbd>Enter</kbd> | highlight and render a row in the wavefunction surface table |
| <kbd>D</kbd> / <kbd>S</kbd> | render total density / spin density from the active wavefunction |
| <kbd>Space</kbd> | play / pause an animation |
| <kbd>[</kbd> / <kbd>]</kbd> | step one frame |
| <kbd>&lt;</kbd> / <kbd>&gt;</kbd> | previous / next normal mode |
| <kbd>g</kbd> | switch between geometry and energy-profile view (reaction paths, trajectories) |
| <kbd>s</kbd> / <kbd>t</kbd> | sidebar / data table |
| <kbd>w</kbd> | write the current frame beside the archive as `.txt` |
| <kbd>q</kbd> | quit |

## What each section kind shows

**Drawn in 3D**: `structure`, `trajectory`, `reaction.path`, `vibrations`,
every `volume.*`, `basis.ao`, and `wavefunction.gto` in the interactive TUI.
Volumes draw their isosurface over the structure; signed fields (orbitals,
spin, difference densities) get positive and negative lobes in orange and
blue, while unsigned density gets one surface. `trajectory` and
`reaction.path` also have an energy-profile chart (<kbd>g</kbd> in the TUI,
`--chart` for `show`).

Activating `wavefunction.gto` evaluates a default orbital (the frontier when
one is well-defined) and
opens a focusable surface table on the right. Use <kbd>Up</kbd> and
<kbd>Down</kbd> to highlight a row and <kbd>Enter</kbd> to render it, or use
<kbd>n</kbd>/<kbd>p</kbd> to step through orbitals directly. The table keeps
the complete spin identity, so alpha and beta orbitals with the same index
remain distinct. It works the same way for canonical, natural, and localized
sets. Canonical rows are identified by energy and occupation; natural rows by
their declared occupation or transition weight; localized rows by the atoms
on which they are centred, because a localized orbital has no meaningful
orbital-energy ordering.

When the archive declares electron-occupation semantics, the first table row
computes total density from the embedded coefficients and occupations.
<kbd>D</kbd> is its shortcut. A natural-transition-orbital section instead
declares transition-weight semantics, so the TUI omits the total-density row
and explains the refusal if <kbd>D</kbd> is pressed. Legacy natural sections
without either declaration still render every individual orbital, but omit
density because a coincidental value sum cannot distinguish the two meanings.
An unrestricted wavefunction also offers signed spin density, rho-alpha minus
rho-beta, with <kbd>S</kbd> as the shortcut. A restricted wavefunction omits
that row and reports why if <kbd>S</kbd> is pressed. Computed fields are cached,
so changing the isovalue re-contours the sampled grid instead of re-evaluating
every basis function.

The on-demand evaluator currently supports basis shells through `l = 3`
(s, p, d, and f). If an orbital has more than 0.5% of its coefficient weight
in g or higher shells, the status line marks the surface incomplete. Periodic
wavefunctions are real Gamma-point fields evaluated from central-cell AOs;
periodic-image AO tails are not added. See the
[QVF wavefunction contract](../design_qvf_format.md) for these display limits.

**Charted**: `bands`, `dos.total`, `dos.projected`, `dos.coop`, `dos.cohp`,
every `spectra.*` stick spectrum, `scf_history`, `equation_of_state`,
`phonon_bands`, `phonon_dos`, and `scan.surface` (as a colour heat map).

Chart conventions match the interactive Plotly renderers so the two tell the
same story: bands and DOS are Fermi-referenced when E_F falls inside the data
window, and labelled absolute (naming E_F) when it does not; SCF convergence
is |ΔE| on a log axis; spectrum X-axes carry each kind's native unit (cm⁻¹
for IR/Raman/VCD, eV for UV-Vis/ECD), with the Lorentzian envelope width
floored in that same unit.

**Tabulated**: `citations` (BibTeX ready to paste), `run.record` (the
invocation, its verbatim input, and the tail of its log), `job.spec`,
`atom_properties`, `bond_orders`, `structure.symmetry`,
`spectra.nmr`, `spectra.epr`, `topology.qtaim`.

Any kind the viewer does not draw still appears in the section browser with
the honest reason from the kind registry ("not yet rendered" is a different
statement from "unsupported"), so the list is a true inventory of the archive
rather than a list of what happens to be implemented.

## Biomolecules

An all-atom render of a solvated protein is not a picture of a protein. A
typical MD system is overwhelmingly water, and even the solute alone is
thousands of overlapping spheres that resolve to a solid mass at terminal
resolution. Use the backbone trace instead:

```sh
vibe-view show protein.qvf --representation backbone
```

That draws one point per alpha carbon, joined along each chain, coloured by
secondary structure: red for helix, yellow for strand, grey for coil. Chains
are traced separately, so no bond is ever drawn between one chain's
C-terminus and the next chain's N-terminus.

`--color-by chain` colours each chain instead. `--color-by bfactor` ramps
blue to red across the temperature factors, and falls back to element
colouring when the column carries no variation, which is common: many files
write a constant 0.00 rather than omitting the field, and scaling that would
paint every atom one flat colour. Whenever a requested scheme cannot be
honoured the status line says which one is being shown and why.

## Rendering modes

**braille** (default) packs a 2x4 dot matrix into every character, so an 80x24
terminal is a 160x96 pixel canvas. Eight times the geometric detail of block
characters, at the cost of one colour per cell (the eight dots share the
average colour of what they cover). Best for structures and isosurfaces,
where shape carries the meaning.

**half** (`--mode half`, or <kbd>d</kbd> in the TUI) splits each cell into two
independently coloured pixels. A quarter of the vertical resolution, but two
true colours per cell, better for heat maps and anything where colour is the
data.

Both need a terminal that speaks 24-bit colour and has a font with braille
coverage (most modern monospace fonts do). If braille renders as boxes, use
`--mode half`.

## Consistency with the GUI

Element colours and radii come from the same `cpk_color` / `cpk_radius`
functions the interactive viewer and every PNG capture use, including any
per-element override you have set: a terminal frame and a GUI screenshot of
the same archive agree on what carbon looks like.

Periodic systems follow the same rules as the 3D renderer: only axes flagged
in `pbc` are drawn as cell edges (a 2D slab gets the in-plane parallelogram,
never a box around the vacuum), and bonds that cross a cell face are drawn to
the nearest periodic image rather than stretched across the box.

## From Python

`render_terminal` is on the public SDK, next to the PNG capture functions:
same renderers, text instead of an image, and no GL context required.

```python
from vibeview import render_terminal

print(render_terminal("job.qvf", size=(100, 30)))
print(render_terminal("job.qvf", "vol_mo_3", isovalue=0.03))

# plain=True drops the ANSI colour, for a log file
open("frame.txt", "w").write(render_terminal("job.qvf", plain=True))
```

It accepts a path, a file-like object, or an already-open `QVFReader` (a
reader you pass in stays open: the SDK only closes readers it opened
itself). Keyword arguments match the `show` flags: `mode`, `representation`,
`color_mode`, `replication`, `isovalue`, `rotation`, `show_labels`, `frame`,
`chart`. A section with no graphical form returns a short explanation rather
than raising, so sweeping every section needs no pre-filtering by kind.

For finer control, the layers underneath are importable directly:
`vibeview.tui.show.render_section` returns the `CellGrid` before it is
stringified, `vibeview.tui.plots` builds charts, `vibeview.tui.panes` builds
the text panes, and `vibeview.tui.scene` + `vibeview.tui.raster` are the
scene builder and the rasterizer.

A runnable script covering all of it (representations, orbitals,
supercells, charts, text panes, animation, and the full sweep) ships at
`examples/terminal_mode.py` in the separate vibe-view checkout:

```sh
python examples/terminal_mode.py job.qvf
python examples/terminal_mode.py job.qvf --plain
```

It branches on what the archive actually contains, so it is safe to point at
any `.qvf`.

## Limits

- No mouse rotation: terminals report clicks, not smooth drags. Use the keys.
- One colour per braille cell, so two differently coloured atoms sharing a
  cell blend. Zoom in, or use `--mode half`.
- Large volumes contour on every isovalue change. A 100³ grid is quick; a
  300³ grid takes a moment.
- Metadata text panes are read-only. The wavefunction surface table is the
  exception: it is focusable, and <kbd>Enter</kbd> renders its highlighted row.
  Use `vibe-view export` to get data out.

## See also

- [Reading a `.qvf` in the terminal, over SSH](../tutorial/vibe_view_terminal.md),
  the worked tutorial: produce an archive, then read every section of it
  from a login shell.
- [`vq` Job Manager](vibe_view_vq_jobs.md), fetching results from the queue.
- [vibe-view: interactive viewer](vibe_view.md), the browser viewer whose
  conventions this one mirrors.
- `vibe-view capture`, PNG rendering when a GL context *is* available.
- `vibe-view info`, the same metadata as `show --info`, machine-readable.
- [MolTUI](https://github.com/kszenes/moltui), a format-agnostic terminal
  viewer for loose `.molden` / `.cube` / `.xyz` files from any code, where
  terminal mode reads `.qvf` only. See
  [the MolTUI tutorial](../tutorial/moltui_terminal_viewer.md).
