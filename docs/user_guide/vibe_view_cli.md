---
myst:
  html_meta:
    "description": "Every vibe-view command in one place: open, tui, show, capture, animate, export, table, info, validate, diff, slice, merge, supercell, from-vq and the rest, with the extra each one needs."
    "og:title": "vibe-view command reference"
    "og:description": "The full vibe-view CLI, grouped by task, with which install extra each command requires."
---

# vibe-view command reference

Source commands on this page run from the **separate vibe-view checkout**.
Clone [mpei/vibe-view](https://github.com/vibe-qc/vibe-view) and follow
[viewer setup](../tutorial/vibe_view_getting_started.md). Its `.venv`, `scripts/`, and `electron/` belong to that repository;
vibe-qc does not contain or install the viewer.

Every `vibe-view` command, grouped by what you are trying to do. Each command
takes `--help` for its exact option list.

The product and command are spelled `vibe-view`. The Python distribution and
import package are spelled `vibeview`. The distribution is not on PyPI yet,
so use the checkout installer or validated companion wheel from the getting-started guide,
not a bare `pip install vibeview` or `pipx install vibeview`.

```{important}
The legacy 2.15.2 wheel includes the `demo`, `doctor`,
`formats`, and persistent `import` commands below.
Run `vibe-view --help` to inspect any installed version.
```

```{note}
**Which extra do I need?** Most of this works on the base install
(`pip install .`). `open`, `compare`, `serve` and `dashboard` need
the browser stack (`[viewer]`); `tui` needs Textual (`[tui]`). Two more extras
widen what you can read rather than adding commands: `[ase]` for the long tail
of structure formats, and `[smiles]` for building a structure from a SMILES
string. `[all]` is every one of them. See
[Getting started with vibe-view alone](../tutorial/vibe_view_getting_started.md)
for the install itself. Run a command whose extra is missing and vibe-view
names the extra and the exact install command.

From a source checkout, `./scripts/install.sh` installs the base,
`[viewer]`, and `[tui]` surfaces together into `.venv`; its updater
keeps that environment and the current Git branch current. Use
`./scripts/update-desktop.sh` when the source-backed Electron app
should be refreshed too; packaged desktop artifacts update separately where a
feed or replacement build is published.
```

## Look at an archive

| Command | Does | Needs |
|---|---|---|
| `vibe-view open INPUT_FILES...` | Open one or more QVFs or supported loose files and launch the interactive viewer. Loose inputs are converted in memory. | `[viewer]` |
| `vibe-view tui QVF_FILE` | Open a QVF in the interactive terminal viewer. | `[tui]` |
| `vibe-view show QVF_FILE` | Print one frame of a QVF to the terminal and exit. | base |
| `vibe-view desktop FILE` | Launch the native Electron window from a source checkout. | `[viewer]` + source checkout |
| `vibe-view compare QVF_A QVF_B` | Open two `.qvf` files side by side with compare mode enabled. | `[viewer]` |
| `vibe-view serve DIRECTORY` | Start a web-based QVF file browser for a directory. | `[viewer]` |
| `vibe-view dashboard PATTERNS...` | Render a grid preview of multiple QVF files. | `[viewer]` |

`open` takes `--port`, `--host`, `--no-browser`, `-s/--section` to start on a
named section, and `--auto-compare` when you pass several files. `show` takes
the terminal-rendering flags (`--size`, `--mode`, `--plain`,
`--representation`, `--color-by`, `--rotate`, `--replicate`, `--isovalue`,
`--frame`, `--chart`, `--all`, `--info`); see
[Terminal mode](vibe_view_terminal_mode.md). `dashboard` takes `--cols`,
`--cell-size` and `--html`.

Do not pass a directory to `open`. Use `vibe-view serve DIRECTORY` for the
browser file picker, or pass explicit files and shell globs. See
[Input formats and interoperability](vibe_view_formats.md) for the accepted
loose inputs and their data-preservation limits.

## Import data from another producer

| Command | Does | Needs |
|---|---|---|
| `vibe-view formats` | List built-in and plugin importers, extensions, preserved data kinds, and availability. | base |
| `vibe-view import INPUT...` | Convert loose files into persistent, validated QVF archives. | base; a selected importer may require an extra |

One input writes a sibling `.qvf` unless `-o/--output` names a file. Multiple
inputs, or one directory, write one QVF per source into an output directory;
the default is `./vibe-view-imports`.

```sh
vibe-view formats
vibe-view formats --json
vibe-view import molecule.xyz
vibe-view import density.cube -o density.qvf
vibe-view import ./results -o ./qvf-results/
vibe-view import calculation.data --from xyz -o calculation.qvf
```

Outputs are not replaced by default. `--force` permits an intentional
replacement. `--from FORMAT` selects an importer by the name printed by
`formats`. Each input is converted independently; this command does not merge
calculation sidecars. Import first before passing a loose file to QVF-only
commands such as `show`, `tui`, `capture`, `validate`, `slice`, or `merge`.

## Render and export

| Command | Does | Needs |
|---|---|---|
| `vibe-view capture QVF_FILE` | Render a section to PNG via the headless capture API. | base |
| `vibe-view animate QVF_FILE` | Render animated sections (trajectory, reaction paths, vibrations, orbitals) to MP4/GIF. | base |
| `vibe-view batch PATTERNS...` | Render a gallery of PNGs from many `.qvf` files (offscreen). | base |
| `vibe-view export QVF_FILE` | Export geometry from a `.qvf`. | base |
| `vibe-view table QVF_FILE` | Dump tabular data (frequencies, charges, MO energies) from a `.qvf`. | base |

`export` writes 12 formats via `-f`/`--format`: structure (`xyz`, `cif`,
`cml`, `json`, or a regenerated vibe-qc input script via `py`), meshes
(`obj`, `gltf`), scenes (`pov`, `blend`), and pages/figures (`html`,
`svg`, `pdf`). `py` reuses the QVF's own provenance (method, basis,
functional) so the regenerated script reproduces the original
calculation, and carries the lattice so periodic files export a
`PeriodicSystem` script rather than silently degrading to molecular.
Rendering needs no browser, which is the point: this is the group to reach for
in a script, a CI job, or a figure pipeline. If a render fails complaining
about a display or a GL context, set `PYVISTA_OFF_SCREEN=True`.

## Inspect, verify, compare

| Command | Does | Needs |
|---|---|---|
| `vibe-view info QVF_FILE` | Print detailed metadata and section summary for a `.qvf` file. | base |
| `vibe-view validate QVF_FILES...` | Validate QVF file integrity: schema, SHA-256 hashes, member presence. | base |
| `vibe-view diff QVF_A QVF_B` | Compare two `.qvf` files: energy delta, section differences, geometry RMSD. | base |
| `vibe-view batch-compare QVF_FILES...` | Compare multiple `.qvf` files: energies, RMSD, sections. | base |
| `vibe-view stats DIRECTORY` | Show aggregate statistics for all `.qvf` files in a directory. | base |
| `vibe-view recent` | Show recently opened QVF files. | base |

`info --short` is a single line per archive, which is what you want in a loop.
`validate` is the one to run before you publish or hand on an archive: it
checks every member against the sha256 recorded in the manifest, so a
truncated download or a corrupted transfer is caught rather than half-rendered.
`diff` and `batch-compare` both take `--json` for scripting.

## Transform archives

| Command | Does | Needs |
|---|---|---|
| `vibe-view slice QVF_FILE` | Extract a subset of sections into a new `.qvf` file. | base |
| `vibe-view merge QVF_FILES...` | Combine sections from multiple `.qvf` files into one archive. | base |
| `vibe-view supercell QVF_FILE` | Build a supercell by replicating the unit cell Nx x Ny x Nz times. | base |
| `vibe-view h-add QVF_FILE` | Add hydrogen atoms to saturate all open valences. | base |

`slice` is how you make a big archive small enough to email: keep the
structure and the one section your colleague needs and drop the volumetric
data. `h-add` writes `<stem>_h.qvf` unless you pass `-o`.

## Work with the queue

| Command | Does | Needs |
|---|---|---|
| `vibe-view from-vq JOB_IDS...` | Fetch job outputs from vq and open their `.qvf` files in vibe-view. | `[viewer]`, `vq` |
| `vibe-view vq-features [FEATURE]` | Show what vibe-qc needs to produce for specific visualizations. | base |
| `vibe-view quickstart` | Run a demo calculation and open it in vibe-view, the fastest path to a 3D molecule. | `vibeqc` |

`vq-features` answers the question that comes up constantly when a panel is
empty: *what did the calculation have to write for this to render?* Run it
with no argument for the list, or name a feature. `quickstart` runs a real
calculation, so unlike everything else on this page it needs vibe-qc
installed; see the [vq Job Manager](vibe_view_vq_jobs.md) for the same loop
inside the viewer.

## Setup and self-check

| Command | Does | Needs |
|---|---|---|
| `vibe-view demo` | Write and validate a deterministic water structure demo; optionally open it. | base; `[viewer]` for `--open` |
| `vibe-view examples` | Show common workflows; optionally copy bundled examples. | base |
| `vibe-view config` | Manage vibe-view configuration. | base |
| `vibe-view doctor` | Diagnose Python, packaged resources, optional modes, and source-desktop support. | base |
| `vibe-view capture-selftest` | Validate that headless PyVista capture can render a nonblank image. | base |

`demo` is the no-input first-run path. It does not run a chemistry code and
does not need vibe-qc:

```sh
vibe-view demo
vibe-view demo -o water.qvf --open
vibe-view demo -o water.qvf --force
vibe-view demo --open --port 9876 --no-browser
```

Without `--open`, it prints browser, TUI, desktop, and headless next steps.
`examples --copy DIRECTORY` copies the installed examples without requiring a
source checkout; `--force` permits replacement. `doctor --json` and
`formats --json` provide machine-readable diagnostics.

`capture-selftest` is the first thing to run on a new headless machine: it
answers whether the offscreen renderer works at all, separately from whether
your archive is fine. A blank PNG from `capture` on a server usually means the
GL stack, not the data. On success it prints the pyvista and VTK versions it
rendered with; `-o FILE` keeps the test PNG instead of discarding it, and
`--size WxH` sets its resolution.

## See also

* [Getting started with vibe-view alone](../tutorial/vibe_view_getting_started.md),
  install and first run on Linux and macOS.
* [vibe-view: interactive viewer](vibe_view.md), the browser UI in depth.
* [Input formats and interoperability](vibe_view_formats.md), other chemistry
  codes, import behavior, and the importer plugin contract.
* [Terminal mode](vibe_view_terminal_mode.md), the full `show` and `tui`
  surface.
* [Working with vibe-view](../tutorial/working_with_vibe_view.md), worked
  examples of the capture and export APIs from Python.
* [QVF and vibe-view](../visualization.md), the section index.
