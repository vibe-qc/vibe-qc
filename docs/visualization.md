---
myst:
  html_meta:
    "description": "QVF and vibe-view: the one-file calculation archive vibe-qc writes, and the browser, terminal, desktop and headless viewers that read it. Format spec, tutorials, and how to emit QVF from another quantum-chemistry code."
---

# QVF and vibe-view

A vibe-qc calculation does not have to leave behind a scatter of `.cube`,
`.molden`, `.xyz`, `.xsf` and log files. It can leave behind **one file**: a
`.qvf` archive carrying the structure, the wavefunction, scalar fields,
spectra, bands, trajectories, the run's provenance, and its citations, each
member typed and checksummed.

**QVF** is that format. **vibe-view** is a standalone viewer and conversion
tool that reads it in a browser, in a terminal, as a desktop app, or
headlessly from Python. It also opens selected structure and volume formats
from Gaussian, ORCA, VASP, molecular-dynamics tools, and other producers.

This section is the whole story in one place: what a `.qvf` is, how to produce
one, the four ways to read one, and how to emit QVF from a quantum-chemistry
code that is not vibe-qc.

## Install the reader you need

| Situation | Installation |
|---|---|
| You only received a `.qvf` | follow [viewer setup](tutorial/vibe_view_getting_started.md) for a source install or validated companion artifact |
| You have repository access and want samples or the source desktop launcher | run `./scripts/install.sh` from the separate vibe-view checkout |
| You already installed vibe-qc and want one combined environment | follow [Install both](getting_started.md#install-both) |
| You are on an SSH compute node | install the core or TUI profile; no display server is required |

Opening and reading a QVF does not import or execute the vibe-qc calculation
engine. The archive is the boundary between producer and viewer, so basic
inspection can happen in a different environment or on another machine.
Optional live-calculation and submission features do require vibe-qc.

## Choose your route

Everything below is also in the sidebar. Start with the row that describes
what you already have:

| Goal | Start here |
|---|---|
| Try the viewer with no calculation or input file | Install the separate vibe-view checkout, then run `vibe-view demo` in [Getting started with vibe-view alone](tutorial/vibe_view_getting_started.md). |
| Open a QVF in a browser | `vibe-view open job.qvf`; use the [browser walkthrough](tutorial/vibe_view_walkthrough.md) for the panels. |
| Work over SSH with no display | `vibe-view show job.qvf` or `vibe-view tui job.qvf`; see [Terminal mode](user_guide/vibe_view_terminal_mode.md). |
| Open a file from another chemistry code | Check [Input formats and interoperability](user_guide/vibe_view_formats.md), then use `vibe-view open INPUT`; current checkout/next-wheel installs can persist it with `import`. |
| Use a native desktop window | Follow the [desktop app guide](user_guide/vibe_view_desktop.md). |
| Add QVF output to another code | Start with [Adopting QVF in your own code](tutorial/qvf_adapt_and_writer.md). |
| Inspect or automate QVF from Python | Use the [QVF consumer reference](consumer_qvf_reference.md). |

The product and shell command are spelled `vibe-view`. Its Python
distribution and import package are spelled `vibeview`.

## Learn the format and its tools

**1. Understand the format.** Start with
[The QVF file format, end to end](tutorial/qvf_file_format.md): what the
archive contains, the manifest, and the section-kind catalog group by group.
Then [Running a calculation as a QVF container](tutorial/qvf_job_containers.md)
for the other direction, a `.qvf` that carries the *request* as well as the
result and settles in place when it runs.

**2. Produce one.** Every `run_job` / `run_periodic_job` call can write a
`.qvf` directly; [QVF job containers](user_guide/qvf_containers.md) is the
user-guide reference for the container lifecycle. If you want the archive
without going through the runner, the writer API is covered in the format
tutorial above.

**3. Read one.** Four routes, in rough order of how much you need on the
machine you are sitting at:

| Route | Command | Needs |
|---|---|---|
| Browser viewer | `vibe-view open job.qvf` | a GL context, the `[viewer]` extra |
| Terminal | `vibe-view show` / `vibe-view tui` | base install (`[tui]` extra for the interactive half); no display or GL context |
| Desktop app | `vibe-view desktop job.qvf` | `[viewer]` plus a source checkout; packaged artifacts are separate |
| Headless / Python | `vibe-view capture`, `render_terminal`, `QVFReader` | base install |

* **Only want the viewer?**
  [Getting started with vibe-view alone](tutorial/vibe_view_getting_started.md)
  installs it without building vibe-qc, on Linux or macOS, and opens one of
  the sample archives that ship in the clone.
* The browser viewer, panel by panel:
  [vibe-view: an end-to-end walkthrough](tutorial/vibe_view_walkthrough.md),
  then [vibe-view: interactive viewer](user_guide/vibe_view.md) as the
  reference.
* The terminal UI:
  [Reading a `.qvf` in the terminal, over SSH](tutorial/vibe_view_terminal.md)
  for the worked example, [Terminal mode](user_guide/vibe_view_terminal_mode.md)
  for the full `show` / `tui` surface, key map and rendering modes.
* The desktop app:
  [vibe-view desktop app](user_guide/vibe_view_desktop.md).
* Headless capture, exports, compare mode and the scripted workflows:
  [Working with vibe-view](tutorial/working_with_vibe_view.md).
* Every command in one table, and which install extra each one needs:
  [vibe-view command reference](user_guide/vibe_view_cli.md).
* Loose-file imports, data-preservation limits, and routes from other codes:
  [Input formats and interoperability](user_guide/vibe_view_formats.md).
* Proteins and other biomolecules, cartoon rendering, chains, residues and
  b-factors:
  [Biomolecules](user_guide/vibe_view_biomolecules.md).
* Reading a `.qvf` from your own Python, with the verify-before-use contract:
  [QVF consumer reference](consumer_qvf_reference.md).

**4. Watch a job as it runs.** [Live reload](user_guide/vibe_view_live_reload.md)
hot-reloads the scene as a running calculation rewrites its checkpoint, and the
[vq Job Manager](user_guide/vibe_view_vq_jobs.md) turns the viewer into a
cockpit for the [`vq` queue](user_guide/queue.md): submit, monitor, fetch, open.

**5. Emit QVF from your own code.**
[Adopting QVF: how to implement a writer in your own code](tutorial/qvf_adapt_and_writer.md)
is the tutorial;
[The QVF format toolkit](qvf/index.md) hosts the normative specification, the
Apache-2.0 reference writers in Python and C++, the integration and library
guides, a worked ORCA mapping, and the governance model.

## Related pages elsewhere in the docs

* [QVF format design](design_qvf_format.md), the design document behind the
  format (rationale, trade-offs, the full section-kind reference).
* [QVF `basis.ao` section design](design_basis_ao_section.md), the contract for
  a single atomic orbital sampled on a grid.
* [QVF-Basis design](design_qvf_basis.md), the canonical basis-set
  representation behind vibe-qc's basis-set library.
* [Viewing geometries, orbitals, and vibrations with MolTUI](tutorial/moltui_terminal_viewer.md),
  a format-agnostic terminal viewer for the loose `.molden` / `.cube` / `.xyz`
  files vibe-qc also writes.
* [Orbital visualization](tutorial/orbital_visualization.md) and
  [XSF / BXSF visualization](tutorial/xsf_bxsf_visualization.md), for the
  non-QVF output routes.

```{toctree}
:maxdepth: 1
:caption: The QVF format

tutorial/qvf_file_format
tutorial/qvf_job_containers
user_guide/qvf_containers
consumer_qvf_reference
```

```{toctree}
:maxdepth: 1
:caption: Reading a QVF with vibe-view

tutorial/vibe_view_getting_started
tutorial/vibe_view_walkthrough
user_guide/vibe_view
tutorial/vibe_view_terminal
user_guide/vibe_view_terminal_mode
tutorial/working_with_vibe_view
user_guide/vibe_view_cli
user_guide/vibe_view_python_api
user_guide/vibe_view_formats
user_guide/vibe_view_biomolecules
user_guide/vibe_view_desktop
```

```{toctree}
:maxdepth: 1
:caption: Live jobs

user_guide/vibe_view_vq_jobs
user_guide/vibe_view_live_reload
```

```{toctree}
:maxdepth: 1
:caption: Adopting QVF in another code

tutorial/qvf_adapt_and_writer
qvf/index
```

```{toctree}
:maxdepth: 1
:caption: Other viewers

tutorial/moltui_terminal_viewer
```
