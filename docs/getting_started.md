---
myst:
  html_meta:
    "description": "Start here with vibe-qc and vibe-view: choose the calculation engine, viewer-only, combined, or learning path, install it, verify it, and run a first calculation."
    "og:title": "Start here with vibe-qc and vibe-view"
---

# Start here

Use this page to choose which independently installed products your workflow
needs:

- **vibe-qc** is the calculation engine. It has a compiled C++ core and needs
  the native build prerequisites described in [Installation](installation.md).
- **vibe-view** reads `.qvf` calculation archives. It can be installed by
  itself and does not need the vibe-qc C++ core.
- **vibe-queue** schedules commands on compute hosts. Its CLI and import
  package are `vq`; it can run other codes without installing vibe-qc.

## Choose your route

| I want to... | Install | First destination |
|---|---|---|
| Run molecular or periodic calculations | vibe-qc | [Engine installation](installation.md) |
| Open a `.qvf` somebody sent me | vibe-view only | [Viewer-only installation](tutorial/vibe_view_getting_started.md) |
| Calculate and inspect results on one machine | vibe-qc and vibe-view | [Install both](#install-both) |
| Learn molecular quantum chemistry | vibe-qc | [Molecular learning path](tutorial/index.md#molecular-path) |
| Model crystals, slabs, or wires | vibe-qc | [Periodic learning path](tutorial/index.md#periodic-path) |
| Read results over SSH without graphics | vibe-view core or TUI | [Terminal mode](tutorial/vibe_view_terminal.md) |
| Submit and monitor remote work | `vq`, plus the code being run | [Remote-job tutorial](tutorial/vq_queue_remote_job.md) |

Each product has its own GitHub repository. See
[repositories and downloads](installation.md#repositories-and-downloads)
for clone instructions and source tags. A published viewer wheel can be
installed without cloning the viewer repository.

## Install the calculation engine

The recommended installer owns the entire bootstrap: it checks the host,
builds the pinned native libraries, creates `.venv`, installs the Python
package, and prints a verification banner.

```sh
git clone https://github.com/vibe-qc/vibe-qc.git
cd vibe-qc
./scripts/install.sh --dev
```

The example above selects `main`. The flag-free installer selects the newest
stable tag advertised by origin. Each companion has its own releases; a core
tag does not update a viewer or queue checkout.

The first native build normally takes 15 to 40 minutes. Later runs reuse the
finished dependency trees. Choose a different source line explicitly when
needed:

```sh
./scripts/install.sh --current       # retain this checkout revision
./scripts/install.sh --dev           # rolling main development branch
./scripts/install.sh --branch <tag-or-branch>
```

Do not reuse `.venv` from another checkout. An editable installation records
the checkout and native-library paths it was built against.

### Verify the engine

Run the interpreter from the environment you just created:

```sh
.venv/bin/python -c "from vibeqc import print_banner; print_banner()"
.venv/bin/python -c "import vibeqc; print(vibeqc.__file__)"
```

The first command should print the vibe-qc version and the linked libint,
libxc, spglib, libecpint, FFTW, and BLAS versions. The second path should point
into this checkout. If it points elsewhere, stop and repair the environment
before running calculations.

The complete platform package lists, manual build, BLAS choices, cluster
notes, and failure messages live in [Installation](installation.md).

## Install only the viewer

The companion's [installation guide](https://vibe-qc.com/vibe-view/docs/installation.html)
and [quickstart](https://vibe-qc.com/vibe-view/docs/quickstart.html) cover its
current release, optional profiles, and first rendered molecule.

vibe-view needs no local compiler or native build. Its Python package installs
compiled VTK as a prebuilt wheel; CMake, libint, and the vibe-qc engine are not
required.

### Release downloads

Get viewer artifacts from the [vibe-view releases](https://github.com/vibe-qc/vibe-view/releases).
The wheel retained under this documentation site's downloads is a legacy
monorepo artifact, not a verified current companion release. Use the source
route below until the companion release provides a suitable artifact.

### From a source checkout

```sh
git clone https://github.com/vibe-qc/vibe-view.git
cd vibe-view
./scripts/install.sh
source .venv/bin/activate
vibe-view --version
```

Add `--with-electron` to download the reviewed Electron runtime during
installation instead of at first desktop launch. The standalone viewer
installer does not build or install vibe-qc.

(install-both)=

## Install both

Two environments are the easiest arrangement to understand and update:

```text
<vibe-qc-checkout>/.venv/             calculation engine
<vibe-view-checkout>/.venv/          viewer (separate repository)
```

Install each with its own script, then activate the one needed for the
current command. The `.qvf` file is the interface between them, so the two
processes do not need to share a Python environment.

If one combined environment is important, activate the vibe-qc environment
and install the local viewer package into it:

```sh
source .venv/bin/activate
python -m pip install -e '../vibe-view[viewer,tui]'
vibe-view --version
python -c "import vibeqc; print('vibe-qc import: ok')"
```

This example assumes independent `vibe-qc` and `vibe-view` clones next to
each other and is run from `vibe-qc`. Use the actual viewer checkout path if
you chose another layout. There is no viewer source directory in vibe-qc;
`viewer-gpu` alone cannot obtain the private companion from PyPI.

## Run the first calculation

Keep calculations outside the source checkout. This separates source files
from output archives and makes `git status` meaningful.

```sh
mkdir -p ~/vibeqc-runs/water
cd ~/vibeqc-runs/water
```

Save this as `water.py`:

```python
from vibeqc import Atom, Molecule, run_job

mol = Molecule([
    Atom(8, [0.0,  0.00,  0.00]),
    Atom(1, [0.0,  1.43, -0.98]),
    Atom(1, [0.0, -1.43, -0.98]),
])

result = run_job(
    mol,
    basis="6-31g*",
    method="rhf",
    output="water",
    output_qvf=True,
)

print(f"converged = {result.converged}")
print(f"energy    = {result.energy:.10f} Ha")
```

Run it with the engine environment's interpreter:

```sh
<vibe-qc-checkout>/.venv/bin/python water.py
```

Check three things before moving on:

1. `converged = True` appears.
2. `water.out` contains the version banner, SCF trace, and final energy.
3. `water.qvf` exists and passes `vibe-view info water.qvf` when the viewer
   is installed.

Then follow [Quickstart](quickstart.md) for open-shell, periodic, and orbital
examples, or [Planning a calculation](tutorial/planning_a_calculation.md)
before choosing a production method and basis.

## A result is ready to use when

Installation success and scientific success are different checks. Before
using a number in a report or paper, verify:

- the SCF and any post-SCF method converged;
- the charge, multiplicity, geometry units, basis, and method are the ones
  intended;
- the basis and numerical grid are converged for the quantity being compared;
- the memory estimate fits the actual job allocation;
- the `.system`, `.references`, and `.bibtex` provenance files are retained;
- energy differences compare like with like, including dispersion,
  solvation, frozen-core, smearing, and reference-state choices.

[Good practices](good_practices.md) turns this checklist into a repeatable
working convention. [Troubleshooting](troubleshooting.md) starts from the
exact error or numerical symptom when a check fails.

## Where to go next

| Goal | Continue with |
|---|---|
| Learn what the first script does | [Molecular Hartree-Fock](tutorial/molecular_hf.md) |
| Choose a method, basis, and resource budget | [Planning a calculation](tutorial/planning_a_calculation.md) |
| Browse runnable inputs by task | [Examples](https://github.com/vibe-qc/vibe-qc/tree/main/examples) |
| Understand every output file | [Output files](user_guide/output_files.md) |
| Explore a QVF visually | [QVF and vibe-view](visualization.md) |
| Use the API as a reference | [User guide](user_guide/index.md) and [API](api/index.md) |
