---
myst:
  html_meta:
    "description": "Install vibe-view on its own with one setup script, without building vibe-qc, then open a QVF in the browser, desktop app, or terminal."
    "og:title": "Getting started with vibe-view alone"
    "og:description": "The viewer without the chemistry build. One installer sets up browser, desktop-server, terminal, and headless modes."
---

# Getting started with vibe-view alone

The viewer project now publishes its own
[installation guide](https://vibe-qc.com/vibe-view/docs/installation.html) and
[quickstart](https://vibe-qc.com/vibe-view/docs/quickstart.html). They track the
current viewer release independently of this core tutorial. Use them for
standalone setup, profiles, and the first demo; continue here for workflows
that connect viewer output with the vibe-qc documentation.

Source commands on this page run from the **separate vibe-view checkout**.
Clone [vibe-qc/vibe-view](https://github.com/vibe-qc/vibe-view) and follow
the source installation steps below. Its `.venv`, `scripts/`, and `electron/` belong to that repository;
vibe-qc does not contain or install the viewer.

**You will learn:** how to install **only the viewer**, without compiling
vibe-qc, and get a first result without supplying a calculation file. Then
choose a browser, source-checkout desktop window, interactive terminal user
interface (TUI), one-shot terminal render, or headless PNG workflow. Commands
are given for both Linux and macOS.

This is the page for you if somebody sent you a `.qvf` file, if you want to
look at the archives that ship with the repository, or if you work on another
quantum-chemistry code and want to see what the format looks like from the
consumer side before deciding whether to write it. You do not need to run
vibe-qc calculations for any of that.

For the short install, update, repair, and uninstall decision table shared by
the engine and its companion tools, see
[Install and maintain the vibe toolset](../toolset_lifecycle.md).

```{note}
**vibe-view does not need vibe-qc.** Its Python dependencies install from
prebuilt wheels: no C++ compiler, no `libint`/`libxc`/`FFTW` build, and no
`setup_native_deps.sh`. VTK is a compiled dependency, but pip downloads its
prebuilt wheel instead of compiling it locally. If you *do* want to run
vibe-qc calculations, follow [installation](../installation.md) as well. The
default vibe-qc install does not add vibe-view; keep the dedicated viewer
environment, or explicitly install the separate viewer source into a combined environment.
```

The names have different jobs:

| Context | Spelling |
|---|---|
| Product and shell command | `vibe-view` |
| Python distribution and import package | `vibeview` |
| Independent source repository | [vibe-qc/vibe-view](https://github.com/vibe-qc/vibe-view) |

**Prerequisite:** **Python 3.11 or newer** (`python3 --version`). The checkout
route also needs `git`. Debian and Ubuntu users need the matching
`python3-venv` package so Python can create the dedicated environment.

**Time to complete:** about five minutes, most of it the download.

## Choose a path

| You want to... | Continue with... |
|---|---|
| Try the viewer without an input file | Install from the checkout, then run `vibe-view demo --open` (the next wheel release will include it too). |
| Open a QVF or loose structure/volume file in a browser | Install the browser profile, then use `vibe-view open INPUT`. |
| Work over SSH with no display | Install the TUI profile, then use `vibe-view show` or `vibe-view tui`. |
| Use a native desktop window | Use the source checkout and `vibe-view desktop`; the wheel does not contain Electron. |
| Bring data from Gaussian, ORCA, VASP, or another producer | Read [Input formats and interoperability](../user_guide/vibe_view_formats.md), then use `open` or `import`. |
| Add QVF output to another program | Use [Adopting QVF in another code](qvf_adapt_and_writer.md). |

## 1. Choose the checkout or wheel

(fastest-route-download-the-wheel)=

### Companion release artifacts

Look for a validated wheel on the [vibe-view release page](https://github.com/vibe-qc/vibe-view/releases).
The `vibeview-2.15.2` wheel retained under this documentation site's
`_static/downloads/` predates the split. It is a legacy artifact, not a
verified current release. Until an appropriate companion artifact is
published, use the source route below.

### Source route: clone the repository

vibe-view has its own public source repository:

```sh
git clone https://github.com/vibe-qc/vibe-view.git
cd vibe-view
```

The checkout is the recommended route because it also
contains the Electron desktop source, install/update scripts, and larger
example archives. It is also the route that provides the unreleased onboarding
commands on this page today.

```{note}
The public clone needs no credentials. Canonical maintainer development access
is configured separately, and public snapshots can lag GitLab. A wheel does not contain the Electron source
tree, so source-backed desktop mode requires a clone.
```

## 2. Install the modes you need

From the repository root, run the installer and activate the environment it
creates:

```sh
./scripts/install.sh
source .venv/bin/activate
```

That one installer checks for Python 3.11+, creates a dedicated virtualenv,
installs a regular source build from this independent checkout with its
`[viewer,tui]` extras, and verifies the standalone CLI, browser, desktop-server, and
interactive TUI commands. It does not compile or install vibe-qc. The heaviest
download is VTK at roughly 200 MB, so give the first run a minute or two.

Run install, update, reinstall, and uninstall as your regular login user,
without `sudo`. This keeps the checkout, virtualenv, and Electron runtime
owned by the same account and matches Electron's desktop security model.

"Standalone" here means a separate Python environment without the vibe-qc
package or native build. Run the script from the full repository checkout;
the viewer keeps using the canonical QVF schemas and Electron sources stored
alongside it.

The virtualenv avoids the **externally managed** (PEP 668) error from
Homebrew and Linux system Pythons. Do not use `--break-system-packages`.

Electron is about 120 MB and downloads automatically on the first
`vibe-view desktop` launch. To make desktop mode ready during the initial
installation, use this command *instead of* the first install command above:

```sh
./scripts/install.sh --with-electron
```

Add `--dock` to pin the source-backed app to the macOS Dock and
`--link-bin DIR` to write a marked `vibe-view` launcher into an existing
writable directory on your PATH (Homebrew macOS: `/opt/homebrew/bin`):

```sh
./scripts/install.sh --with-electron --dock \
    --link-bin /opt/homebrew/bin
```

Uninstall removes the command link and unpins the Dock tile with the
installation; re-running `install.sh` with the same flags refreshes both
after the checkout moves.

If the environment already exists, update the checkout, Python environment,
and source-backed desktop application together:

```sh
./scripts/update-desktop.sh
```

Quit the desktop window first. The updater installs Electron if it is missing,
synchronizes it to the reviewed lockfile version, and refreshes the Applications
copy and its displayed vibe-view version on macOS. This is the source-backed
desktop app; there is no signed downloadable desktop release yet. On Linux it
synchronizes Electron and the launch wrappers, then you continue to start the
window with `vibe-view desktop`; Linux has no Applications copy.
`install.sh --with-electron` downloads the runtime and installs the macOS
Applications copy in the same step.
If the source app reports that another checkout owns it, inspect that path and
use `--adopt-desktop` only when this checkout should take ownership.

To repair the standalone environment from the current checkout without a Git
fetch, use the transactional reinstall command:

```sh
./scripts/reinstall.sh
./scripts/reinstall.sh --desktop  # also refresh Electron + source app
```

The old environment remains available until the replacement installs and
passes all mode checks. A failed Python, pip, verification, or Electron step
restores it automatically.

Environments newly created or transactionally recreated by the lifecycle
scripts carry a checkout-specific ownership marker. An ordinary in-place
update does not claim an older unmarked environment. Reinstall and other
replacement commands refuse an unmarked or foreign environment. For a
standalone install created before markers existed, inspect it first and add
`--adopt-legacy --python /path/to/external/python3`; trusted PEP 610 metadata
must point to this exact vibe-view checkout. A foreign marker is never adopted.

To remove the source installation, quit every vibe-view desktop window and any
running CLI or browser-server launch, then preview the exact scope first:

```sh
./scripts/uninstall.sh --dry-run
./scripts/uninstall.sh
```

This removes the dedicated standalone environment, a recognizable
direct-download Electron runtime in this checkout, and only a source-backed
macOS app that belongs to this checkout. Packaged apps, another checkout's app,
QVF files, settings, recents, logs, and the app-managed environment under
`$XDG_DATA_HOME/vibe-view/venv` are preserved. Add `--keep-desktop` if you want
to remove only the Python environment and reuse the Electron download later.
The sole user-data housekeeping is a cached interpreter selection: uninstall
removes it only when it points into the environment being deleted.

For a downloaded wheel, which does not contain these source scripts, use the
small manual equivalent:

```sh
python3 -m venv .venv-viewer
source .venv-viewer/bin/activate
python3 -m pip install './vibeview-X.Y.Z-py3-none-any.whl[all]'
```

The distribution is not on PyPI yet, so bare `pip install vibeview`,
`pipx install vibeview`, and equivalent public-index `uv tool install`
commands do not resolve. Use the local wheel path above or the checkout
installer. For a combined environment, follow [Install both](../getting_started.md#install-both)
and supply the separate viewer checkout explicitly.

## 3. Verify, customise, and update

Check the installed environment and optional modes (current checkout or the
hosted wheel):

```sh
vibe-view --version
vibe-view doctor
```

The legacy 2.15.2 wheel includes `vibe-view doctor`; `capture-selftest`
remains available to check the headless rendering path.

```text
vibe-view 2.15.2  -- Roothaan's Roadrunner
Python    3.14.6
PyVista   0.48.4
VTK       (9, 6, 2)
```

Your version numbers will differ. `doctor` checks Python, packaged resources,
the selected extras, and source-desktop availability. Use
`vibe-view doctor --json` when another program needs to consume the result.
On a headless host, also test the actual offscreen render path:

```sh
vibe-view capture-selftest
```

```{admonition} If the install failed, it is almost certainly one of these
:class: warning

**Python is too old.** Install Python 3.11 or newer, then pass its command with
`--python`, for example `--python python3.13`.

**`.venv` already exists.** Preserve and update it with
`./scripts/update.sh --skip-git`, or replace this checkout's marked
virtualenv with `./scripts/install.sh --force`. The installer refuses
unmarked, foreign, unsafe, and non-virtualenv paths. A pre-marker install from
this checkout requires the explicit `--adopt-legacy` proof described above.

**A package download failed.** Re-run the same command once connectivity is
back. The native VTK wheel is large, but there is no compiler fallback step.
```

(installing-less)=

```{admonition} Installing less
:class: tip

The profiles are tiered, so you can install only what you need:

| Installer | Gets you | Needs |
|---|---|---|
| `install.sh --extras core` | reading archives, `show`, `info`, `capture`, `export`, `diff` | NumPy, VTK, matplotlib, Plotly |
| `install.sh --extras tui` | core plus the interactive terminal viewer | adds Textual |
| `install.sh --extras viewer` | core plus browser and desktop-server modes | adds Trame and uvicorn |
| `install.sh` | core, browser, desktop-server, and TUI modes | `[viewer,tui]` |
| `install.sh --extras all` | all modes plus ASE, SMILES, and Jupyter integrations | full optional stack |

A headless capture box, a CI job, or a compute node needs **no extra at all**:
rendering a PNG and reading an archive are in the base install. If you run a
command whose extra is missing, vibe-view names the extra and the exact
install command rather than failing obscurely.

The checkout installer uses a regular, non-editable source install. This lets
`install.sh --force` and `update.sh --recreate-venv` restore the previous
installed code and dependency set if replacement fails, rather than silently
following a checkout that already advanced. PEP 610 source metadata keeps the
desktop command connected to the co-located `electron/` source; `update.sh`
explicitly reinstalls after a successful fast-forward.
```

From the repository root, update later with:

```sh
./scripts/update.sh          # fast-forward the current branch
./scripts/update.sh --dev    # switch to and update main
./scripts/update.sh --release
./scripts/update-desktop.sh  # include the source-backed desktop app
```

The updater refuses a dirty tracked tree and uses fast-forward-only pulls. Git
operations apply only to the vibe-view repository: no
selector updates the current branch, while `--dev`, `--release`, or
`--branch NAME` can switch the whole checkout. Use `--skip-git` to reinstall
from the current tree without fetching.

The default update and reinstall profile is `modes`. Repeat
`--extras core|viewer|tui|all|test` when maintaining a non-default profile;
an in-place pip update does not prune packages left by a broader profile. The
desktop wrapper accepts compatible updater options because it delegates to
`update.sh --desktop`, and its profile must include the viewer. A selected
branch, tag, or commit must contain the safe desktop-update protocol or the
command refuses it before running that revision's installer.

## 4. Create and open the built-in demo

Create a deterministic water structure demo and open it in the browser:

```sh
vibe-view demo -o water.qvf --open
```

The command writes a project-authored QVF, validates it, and then launches the
browser. It does not run a chemistry calculation and does not need vibe-qc.
Without `--open`, it prints copy-paste next steps for browser, TUI, desktop,
and headless modes. Pass `--force` to replace an existing output deliberately;
`--port 9876` and `--no-browser` control the browser server when `--open` is
present.

The viewer starts a small local web server and opens
`http://127.0.0.1:8080` in your default browser. The section browser is on
the left, the 3D viewport in the middle, and the panels for the active section
on the right. Click through the sections to see each one render. Press
<kbd>Ctrl</kbd>+<kbd>C</kbd> in the terminal to stop the server.

The **vibe-qc checkout**, not the vibe-view checkout, carries these larger
calculation archives. Paths in this table are relative to the core repository;
give the viewer an absolute path when the checkouts are in different places:

| File | What is in it |
|---|---|
| `examples/vibe_view/runs/qvf_showcase/water.qvf` | H2O RHF/6-31G\*, 13 sections: structure, density, an orbital, the wavefunction, an optimisation trajectory, vibrations, an IR spectrum, charges, bond orders, SCF history, citations |
| `examples/vibe_view/runs/h2co_showcase/h2co.qvf` | formaldehyde, same shape without the trajectory |
| `examples/vibe_view/output-nacl-showcase.qvf` | periodic NaCl RKS/STO-3G: structure, density, total and projected DOS |

For a viewer-only installation, copy the examples carried by the installed
vibe-view distribution. This supplies `water.xyz`, `water.qvf`, and a README
without a core checkout:

```sh
vibe-view examples --copy ./vibe-view-examples
```

Useful flags:

```sh
vibe-view open water.qvf --port 9876          # if 8080 is taken
vibe-view open water.qvf --no-browser         # start the server, open no browser
vibe-view open water.qvf -s vol_dens_0        # start on a named section
vibe-view open a.qvf b.qvf                    # a Files dropdown switches between them
vibe-view open a.qvf b.qvf --auto-compare     # ... and overlay them straight away
```

Open the same source-checkout installation in its native desktop window:

```sh
vibe-view desktop water.qvf
```

The first desktop launch downloads Electron if you did not fetch it during
installation. This command needs the source checkout because the Electron
shell is not included in the standalone wheel.

For the panel-by-panel tour of what you are looking at, read
[vibe-view: an end-to-end walkthrough](vibe_view_walkthrough.md).

## 5. Look at an archive without any graphics

You do not need a browser, a display, or an OpenGL context. Terminal mode
renders the same scenes as Unicode braille:

```sh
vibe-view show water.qvf
vibe-view tui  water.qvf
```

`show` prints one frame and exits, so it pipes into a log or a file. `tui` is
the interactive full-screen version; press <kbd>?</kbd> for the key map and
<kbd>q</kbd> to quit. This is the mode to use over SSH on a machine with no
display server, which is usually where a `.qvf` is produced in the first
place. Details in [Terminal mode](../user_guide/vibe_view_terminal_mode.md),
worked through in
[reading a `.qvf` in the terminal, over SSH](vibe_view_terminal.md).

Before opening anything, you can also just ask what an archive holds:

```sh
vibe-view info water.qvf
vibe-view info water.qvf --short
```

## 6. Render a figure without opening anything

The base install can write images and other formats directly, which is what
you want on a headless machine or in a script:

```sh
vibe-view capture water.qvf -o structure.png
vibe-view export  water.qvf -f xyz -o water.xyz
vibe-view batch   './results/*.qvf' -o gallery/  # a PNG per matching archive
```

If a capture fails complaining about a display or a GL context, set
`PYVISTA_OFF_SCREEN=True` in the environment:

```sh
PYVISTA_OFF_SCREEN=True vibe-view capture water.qvf -o structure.png
```

If that still fails, `vibe-view capture-selftest` separates a rendering-stack
problem from a problem in the archive. Browser logs default to
`~/.cache/vibe-view/vibe-view.log` unless `XDG_CACHE_HOME` is set.

## Platform notes

**Both.** Everything above installs through pip from prebuilt wheels. There is
no local compiler step, no conda requirement, and no JavaScript build, so the
commands are the same on Linux and macOS.

**Linux.** The browser viewer and PNG capture render through VTK, which wants
a working OpenGL stack; on a minimal server or container image you may need
your distribution's Mesa/GL runtime (`libGL.so.1`) before `open` or `capture`
will work. Run the Electron desktop mode as a regular user, without `sudo`, so
its security sandbox remains enabled. Electron also needs the normal GTK3
desktop runtime; the [desktop guide](../user_guide/vibe_view_desktop.md)
includes the Debian/Ubuntu package command and restricted-sandbox guidance.
Terminal mode has no such requirement: it uses a software rasteriser without
an OpenGL render window, so
`vibe-view show` and `vibe-view tui` work on a bare compute node. If you are on
a remote machine and want the browser UI anyway, forward the port:

```sh
ssh -L 8080:127.0.0.1:8080 you@remote-host
# then, on the remote host:
vibe-view open results.qvf --no-browser
```

and open `http://127.0.0.1:8080` in the browser on your own machine.

**macOS.** Nothing extra is needed on either Apple silicon or Intel; the VTK
and PyVista wheels are prebuilt. The source-backed
[desktop app](../user_guide/vibe_view_desktop.md) can register `.qvf` files.
Signed standalone desktop artifacts have not landed yet.

## Coming from another quantum-chemistry code

You do not need vibe-qc to get value out of the viewer, and you do not need a
`.qvf` to start. Point `vibe-view open` at files you already have and it
converts them to an in-memory archive as it reads them:

```sh
vibe-view open geometry.xyz          # xyz -> qvf, done in memory
vibe-view open density.cube          # a scalar field on its grid
vibe-view open complex.pdb           # chains and residues, so the cartoon works
```

To keep the converted result, import it:

```sh
vibe-view import geometry.xyz
vibe-view import density.cube -o density.qvf
vibe-view import ./results -o ./qvf-results/
```

`vibe-view formats` reports the built-in and plugin importers in this
environment. The exact format-by-format preservation limits and producer
routes are in [Input formats and interoperability](../user_guide/vibe_view_formats.md).

Two important limits. `.molden` is **not** among them, so orbitals from a code
that writes Molden files do not open directly today; for those,
[MolTUI](moltui_terminal_viewer.md) reads loose `.molden` / `.cube` / `.xyz`
in a terminal. And a converted file only carries what the file carries: a
`.cube` has no wavefunction, no spectra, and no provenance, so most of the
viewer's panels stay empty.

That gap is the point of the format rather than a missing importer. Anything
inside one archive gets a panel; anything scattered across sidecars does not.
If you maintain a code and want its results to open with the full surface,
the writer side is deliberately easy to adopt:

* [Adopting QVF in your own code](qvf_adapt_and_writer.md), the tutorial:
  container model, the wavefunction normalization contract, the vendor
  namespace, and certification against the conformance corpus.
* [The QVF format toolkit](../qvf/index.md), the normative specification plus
  reference writers in Python and in zero-dependency C++17, distributed under
  **Apache 2.0** so it can be vendored into any code, including a proprietary
  one, independently of vibe-qc's own MPL-2.0 licence.
* [ORCA integration guide](../qvf/orca_integration.md), a complete worked
  mapping from a production code's data (a GBW-style wavefunction, spectra,
  EPR, scalar properties) onto QVF sections.

A conforming producer is a few dozen lines on top of the reference library,
and the archives it writes open in vibe-view with no viewer-side changes at
all.

## Getting your own `.qvf` files

Five routes, depending on where the data comes from:

* **From the built-in demo.** `vibe-view demo -o water.qvf` creates and
  validates a QVF without running a chemistry code.
* **From a loose file.** `vibe-view import INPUT` converts one supported file
  to a persistent QVF. It does not merge calculation sidecars.
* **From vibe-qc.** Every `run_job` / `run_periodic_job` call already writes
  one, since `output_qvf` defaults to `True`. That needs the full vibe-qc
  install, so follow [installation](../installation.md).
  `vibe-view quickstart` also runs a demo calculation and opens it, and it
  likewise needs vibe-qc present.
* **From the queue.** `vibe-view from-vq JOB_ID` fetches a finished
  [`vq`](../user_guide/queue.md) job straight into the viewer, and the
  [vq Job Manager](../user_guide/vibe_view_vq_jobs.md) panel does it by
  clicking.
* **From another program.** QVF is an open format with a standalone,
  Apache-2.0 reference writer, so a code that is not vibe-qc can emit
  archives vibe-view opens. See
  [Adopting QVF in your own code](qvf_adapt_and_writer.md).

vibe-view also opens its explicitly supported structure and volume formats
directly, converting them to an in-memory archive for that session. Run
`vibe-view formats` for the installed list.

## Next steps

* [vibe-view: an end-to-end walkthrough](vibe_view_walkthrough.md), what every
  panel does.
* [vibe-view command reference](../user_guide/vibe_view_cli.md), commands
  grouped by task.
* [Input formats and interoperability](../user_guide/vibe_view_formats.md),
  other-code routes and an honest data-preservation matrix.
* [vibe-view: interactive viewer](../user_guide/vibe_view.md), the reference
  for the browser UI.
* [Working with vibe-view](working_with_vibe_view.md), the full feature tour:
  compare mode, exports, the capture API, presets, scripted workflows.
* [The QVF file format, end to end](qvf_file_format.md), what is actually
  inside the file you just opened.
* [QVF and vibe-view](../visualization.md), the index for all of it.
