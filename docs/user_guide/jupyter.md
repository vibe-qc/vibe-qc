---
myst:
  html_meta:
    "description": "Running vibe-qc from Jupyter Lab, three installation patterns (Jupyter in the venv, system-wide Jupyter + ipykernel registration, JupyterHub) + notebook patterns + 3D visualisation."
    "og:title": "vibe-qc, Jupyter Lab integration"
    "og:description": "Three ways to use vibe-qc from Jupyter Lab: install JL into the vibe-qc venv, or register the venv as a kernel for a system-wide JL. Plus py3Dmol / NGLView visualisation, OpenMP caveats, remote tunnel."
---

# Running vibe-qc from Jupyter Lab

[Jupyter Lab](https://jupyter.org/) is great for exploratory
quantum-chemistry work, interactive geometry tweaks, live
SCF traces, inline 3D viewers for orbitals and crystal cells,
mixed code + math + prose in one document. vibe-qc works with
Jupyter Lab through one of three install patterns; pick the
one that matches how Jupyter is already on your system.

## Decision tree

| Your situation | Pattern | Effort |
|----------------|---------|--------|
| **No Jupyter yet, single user** | [Pattern A](#pattern-a-jupyter-inside-the-vibe-qc-venv), install Jupyter Lab into vibe-qc's venv | ~2 min |
| **Jupyter Lab already installed system-wide** (Homebrew / pacman / your distro's package) | [Pattern B](#pattern-b-system-wide-jupyter-vibe-qc-kernel), register the vibe-qc venv as an `ipykernel` | ~3 min |
| **JupyterHub or shared Jupyter server** | [Pattern C](#pattern-c-jupyterhub--shared-jupyter-server), admin registers a kernel, or per-user `ipykernel install --user` | varies |

All three patterns reach the same end-state: a Jupyter
notebook in which `import vibeqc` works and uses the vibe-qc
build you actually want to use.

## Pattern A, Jupyter inside the vibe-qc venv

The simplest option: install Jupyter Lab as an extra Python
package inside the same `.venv/` where vibe-qc lives. Jupyter
runs with vibe-qc on its `sys.path` automatically.

```sh
# From inside your vibe-qc checkout.
cd ~/path/to/vibe-qc
.venv/bin/pip install jupyterlab     # ~80 MB of deps; one-time
.venv/bin/jupyter lab                # opens http://localhost:8888/lab
```

That's the whole setup. The notebook server inherits the
venv's Python → `import vibeqc` works without anything else.

### Pros / cons

* ✅ **One venv, one toolchain.** No kernel-registration
  ceremony; updating vibe-qc updates the notebook environment.
* ✅ **Works offline.** No system-package dependencies.
* ❌ **Duplicated install if you have multiple vibe-qc venvs**
  (e.g. dev + release clones). Each needs its own Jupyter Lab.
* ❌ **Bigger venv.** Adds ~80 MB to the venv tree.

### Optional: extras for richer notebooks

```sh
.venv/bin/pip install jupyterlab ipywidgets py3Dmol nglview matplotlib
```

* `ipywidgets`, interactive controls (sliders for lattice
  constant sweeps, dropdowns for functional choice).
* `py3Dmol`, embedded 3D molecule / crystal viewer (the
  primary recommendation; see § Visualisation below).
* `nglview`, alternative viewer with crystallographic
  features (cell box, fractional coordinates, supercells).
* `matplotlib`, already a vibe-qc transitive dep but
  helpful to ensure the inline backend is registered.

(pattern-b-system-wide-jupyter-vibe-qc-kernel)=
## Pattern B, System-wide Jupyter + vibe-qc kernel

If Jupyter Lab is already installed at the system or
per-user level, e.g. `brew install jupyterlab` on macOS,
`pacman -S jupyterlab` on Arch, or `pipx install jupyterlab`
- register the vibe-qc venv as an additional **kernel**.
Jupyter will offer it in the launcher.

```sh
# 1. Activate the vibe-qc venv (so ipykernel installs in it).
source ~/path/to/vibe-qc/.venv/bin/activate

# 2. Install ipykernel into the vibe-qc venv (one-time):
pip install ipykernel

# 3. Register the venv as a kernel for the system Jupyter.
python -m ipykernel install --user \
    --name vibeqc-dev \
    --display-name "Python (vibe-qc dev)"

deactivate
```

Verify:

```sh
jupyter kernelspec list
# should show:
# vibeqc-dev  /Users/USER/Library/Jupyter/kernels/vibeqc-dev  (macOS)
# vibeqc-dev  /home/USER/.local/share/jupyter/kernels/vibeqc-dev  (Linux)
```

Launch your system Jupyter Lab, the kernel picker now offers
"Python (vibe-qc dev)" alongside whatever else you have
registered. Notebooks opened with that kernel `import
vibeqc` from the vibe-qc venv; `print(sys.executable)` shows
the venv path.

### Multiple vibe-qc clones (dev + release)

Register each as its own kernel:

```sh
source ~/vibeqc-dev/.venv/bin/activate
pip install ipykernel
python -m ipykernel install --user \
    --name vibeqc-dev \
    --display-name "Python (vibe-qc dev — main)"
deactivate

source ~/vibeqc-release/.venv/bin/activate
pip install ipykernel
python -m ipykernel install --user \
    --name vibeqc-release \
    --display-name "Python (vibe-qc release)"
deactivate
```

Two kernels in the launcher; pick at notebook-open time
which vibe-qc version to run against. Same logical pairing
as vq's `--branch main / --branch release`.

### Pros / cons

* ✅ **One Jupyter Lab install** serves every project.
* ✅ **Multi-version vibe-qc** without duplicating Jupyter.
* ✅ **Updates to Jupyter Lab** don't disturb vibe-qc.
* ❌ **Kernel needs re-registering** if you delete + recreate
  the venv (the kernelspec's `argv` points at the venv's
  python path; orphans cause `KernelDied` errors).
* ❌ **No ipywidgets / py3Dmol** unless you ALSO install them
  in the vibe-qc venv (the system Jupyter Lab's
  installed-package set isn't visible to your kernel's
  Python).

## Pattern C, JupyterHub / shared Jupyter server

If you're on a shared Jupyter Hub (uni cluster, HPC center,
hosted research environment), the admin chooses how kernels
are exposed. Two common setups:

### Admin-side kernel registration

The hub admin installs the vibe-qc venv on the shared
filesystem and registers it as a system-wide kernel with
`python -m ipykernel install` (no `--user`). Every Hub user
sees it. This is the cleanest pattern for a research-group
shared install.

If you're the admin, follow Pattern B but drop `--user`:

```sh
sudo /opt/vibeqc/.venv/bin/python -m ipykernel install \
    --name vibeqc-shared \
    --display-name "Python (vibe-qc shared)"
```

### Per-user kernel registration on a Hub

If the Hub allows users to install their own kernels, the
recipe is identical to Pattern B:

```sh
# Inside your Hub terminal (or `!` in a notebook cell):
~/path/to/your-vibeqc-clone/.venv/bin/pip install ipykernel
~/path/to/your-vibeqc-clone/.venv/bin/python -m ipykernel install --user \
    --name vibeqc-mine \
    --display-name "Python (vibe-qc — my clone)"
```

The hub picks up `--user`-installed kernels at next launch.
Some Hubs require a Hub restart for new kernels to appear;
ask your admin if the kernel doesn't show.

## A first vibe-qc notebook

Once the kernel is wired up, the smallest useful notebook:

```python
# Cell 1 — quick sanity check
import vibeqc as vq
vq.print_banner()
```

Expect a 5-line banner showing the build (release vs dev,
SHA, linked libraries). If it doesn't render, you're on the
wrong kernel.

```python
# Cell 2 — water RHF
import vibeqc as vq

mol = vq.Molecule([
    vq.Atom(8, [0.0,  0.00,  0.00]),
    vq.Atom(1, [0.0,  1.43, -0.98]),
    vq.Atom(1, [0.0, -1.43, -0.98]),
])

basis = vq.BasisSet(mol, "6-31g*")
result = vq.run_rhf(mol, basis)
print(f"E = {result.energy:.6f} Ha")
print(f"converged in {result.n_iter} iters")
```

```python
# Cell 3 — periodic example: MgO rocksalt
import vibeqc as vq

mgo = vq.PeriodicSystem(
    3,
    [[0.0, 2.106, 2.106],
     [2.106, 0.0, 2.106],
     [2.106, 2.106, 0.0]],
    [vq.Atom(12, [0.0, 0.0, 0.0]),
     vq.Atom(8,  [2.106, 2.106, 2.106])],
)
basis = vq.BasisSet(mgo.unit_cell_molecule(), "sto-3g")
result = vq.run_krhf_periodic_gdf(mgo, basis, kmesh=(1, 1, 1))
print(f"E = {result.energy_per_cell_ha:.6f} Ha/cell")
```

## 3D visualisation in notebooks

For inline 3D viewers, install one of:

```sh
.venv/bin/pip install py3Dmol      # lightweight, recommended
.venv/bin/pip install nglview      # crystal-aware; needs jupyter extension
```

### py3Dmol, molecules and crystals

```python
import py3Dmol
import vibeqc as vq

mol = vq.Molecule.from_xyz("""\
3
water
O   0.0   0.00  0.00
H   0.0   1.43 -0.98
H   0.0  -1.43 -0.98
""")

# Convert vibe-qc Molecule → XYZ string → py3Dmol view:
xyz_str = mol.to_xyz()
view = py3Dmol.view(width=400, height=300)
view.addModel(xyz_str, "xyz")
view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
view.zoomTo()
view.show()
```

### NGLView, periodic cells

NGLView handles crystallographic cells well. The trick: round-
trip through ASE:

```python
import nglview
import vibeqc as vq

mgo = vq.PeriodicSystem( ... )                 # as above
atoms = vq.ase.to_atoms(mgo)                   # vibeqc.ase → ase.Atoms
view = nglview.show_ase(atoms)
view.add_representation("ball+stick")
view.add_unitcell()
view                                            # auto-renders in the cell
```

See [`ase_integration.md`](ase_integration.md) for the full
vibe-qc ↔ ASE bridge.

## Watching long-running SCFs

vibe-qc's `ProgressLogger` writes to stdout, which Jupyter
captures and re-renders per cell. Long SCFs work fine, but
note:

* **Per-iter output buffering** can lag in Jupyter. Force
  flushing with `sys.stdout.flush()` between iterations if
  the trace updates only at the end:

  ```python
  import sys, vibeqc as vq
  opts = vq.RHFOptions()
  opts.progress = vq.ProgressLogger(flush_every=1)
  result = vq.run_rhf(mol, basis, opts)
  sys.stdout.flush()
  ```

* **Interrupting an SCF** with the Jupyter "Interrupt Kernel"
  button (or `Ctrl-I`, `I,I`) sends `KeyboardInterrupt` to the
  Python thread. The C++ side may not respond immediately for
  inner loops; in pathological cases (a stuck libint call)
  you'll need "Restart Kernel" instead. The `vq` queue is the
  right tool for truly batch-shaped long runs.

* **Working-directory effects**: `run_job`, `run_periodic_job`
  write `.out` / `.molden` / `.traj` to the **kernel's
  current working directory**, which is the notebook's
  directory by default. To redirect, pass `output=`:

  ```python
  vq.run_job(mol, basis="sto-3g", method="rhf",
             output="./scratch/h2o")
  # → ./scratch/h2o.out / .molden / .traj
  ```

  Or `os.chdir(...)` at the top of the notebook.

## OpenMP threading in notebooks

vibe-qc's C++ kernels are OpenMP-parallel. The Python notebook
process inherits `OMP_NUM_THREADS` from its environment. **If
you launched Jupyter from a shell that didn't set
`OMP_NUM_THREADS`, libgomp defaults to 1 thread**, your SCF
will run single-threaded and feel slow.

Two fixes:

```sh
# Option A: set in the shell before launching Jupyter:
OMP_NUM_THREADS=8 .venv/bin/jupyter lab
```

```python
# Option B: set inside the notebook BEFORE importing vibeqc.
# Must come first — OpenMP reads the env var at the first
# parallel region, which is at vibeqc-module-load time.
import os
os.environ["OMP_NUM_THREADS"] = "8"

import vibeqc as vq                # now uses 8 threads
```

If you forget and import vibe-qc with `OMP_NUM_THREADS` unset,
you may need to **Restart Kernel** before the new value takes
effect.

A future vibe-qc release will warn at import time if
`OMP_NUM_THREADS` is unset (a v0.8.x improvement filed by
the experimental-spikes chat).

## Remote Jupyter via SSH tunnel

If your big-memory compute box runs Jupyter (Pattern A or B),
you can drive notebooks from your laptop without exposing the
Jupyter port publicly. Tunnel via SSH:

```sh
# 1. On the remote, start Jupyter bound to localhost only:
ssh remote 'OMP_NUM_THREADS=16 ~/vibeqc-dev/.venv/bin/jupyter lab \
    --no-browser --port 8889 --ip 127.0.0.1'
# (prints a URL with a token in the remote terminal — copy the token)

# 2. From your laptop, tunnel port 8889 over SSH:
ssh -N -L 8889:127.0.0.1:8889 remote

# 3. Open http://localhost:8889/?token=<the-token> in your laptop browser.
# Notebooks now run on the remote's CPU + RAM + vibe-qc venv.
```

For workflow scale-up, same script, many parameters, use
[`vq submit`](queue.md) instead. Jupyter is for *interactive*
exploration; vq is for *batch* sweeps.

## Caveat list

* **Don't `pip install` from inside a running notebook** if the
  package extends vibe-qc, Python's import cache won't pick it
  up until you restart the kernel. (Restart Kernel is the safe
  default after any `pip install`.)
* **Notebook output cells can get huge** with verbose SCF
  traces. `Cell → All Output → Clear` before checkpointing
  the notebook to git; otherwise the diff is unreadable.
* **`%autoreload` works** for the Python side of vibe-qc but
  not for the C++ binding (`_vibeqc_core.so`). Native code
  changes require a kernel restart.

  ```python
  %load_ext autoreload
  %autoreload 2          # auto-reload edits to vibeqc/*.py
  ```

* **GIL behaviour**: vibe-qc's C++ kernels release the GIL
  during long compute regions, so you *can* run multiple SCFs
  in parallel threads inside a single notebook, but that
  competes with OpenMP for cores. In practice, **either
  thread-pool your Python OR use OpenMP, not both**.

* **JupyterLab v4 vs v3**, both work. JupyterLab v4 deprecates
  the classic notebook server; if you see "Notebook deprecated"
  warnings, that's expected and harmless.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: No module named 'vibeqc'` | wrong kernel | check kernel picker; should say "Python (vibe-qc …)" |
| Banner shows wrong vibe-qc version | kernel points at a different venv | re-register kernel from the right venv (Pattern B) |
| SCF feels slow despite a beefy machine | `OMP_NUM_THREADS` unset | set it in shell or set before `import vibeqc` |
| `import vibeqc` raises `ImportError: libint2.so` | wrong Python (system instead of venv) | check `sys.executable`; if not the vibe-qc venv, switch kernel |
| Kernel keeps dying mid-SCF | OOM | check the notebook's process memory; lower `--mem-mb` analog by using a smaller basis or fewer k-points |
| 3D viewer doesn't render | missing ipywidgets / py3Dmol install | `.venv/bin/pip install ipywidgets py3Dmol` + restart kernel |

## See also

* [`running.md`](../running.md), running vibe-qc outside of
  Jupyter (venv activation, batch logs, nohup).
* [`queue.md`](queue.md), vq for batch-shaped sweeps; complementary
  to Jupyter for interactive work.
* [`ase_integration.md`](ase_integration.md), vibe-qc ↔ ASE,
  required for NGLView crystal viewer.
* [`output_files.md`](output_files.md), what `.out` /
  `.molden` / `.traj` / `.system` files contain.
