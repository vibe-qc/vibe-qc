# Quickstart, your first 30 minutes

This page gets you from a working install to four working
calculations in under a minute of wall time. Open
[`examples/quickstart.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/quickstart.py)
in another window and follow along, it's the runnable script for
exactly what's on this page. For a wider tour of the API surface
(`run_job`, ASE calculator, logging, custom basis sets, the test
suite), see [tour](tour.md).

If you have not chosen between the calculation engine and the standalone
viewer yet, begin at [Start here](getting_started.md). If the software is
already installed but you are choosing a production method, basis, or memory
budget, read [Planning a calculation](tutorial/planning_a_calculation.md)
after completing this quickstart.

## 0. Install

vibe-qc is **not on PyPI yet**, there is no `pip install vibe-qc`.
The install is "clone the repo, install your platform's build
prerequisites (compiler, CMake, Ninja, headers, a BLAS), run
`./scripts/setup_native_deps.sh` to fetch + build the vendored
native libraries (libint, libxc, spglib, FFTW3, libecpint), then
`pip install -e .` in a virtualenv". The one-liner per platform
(macOS Homebrew, Arch / Manjaro, Debian / Ubuntu) and full
troubleshooting live in [installation](installation.md), read
that first.

Once `pip install -e .` succeeds, the package ships with every
basis set vibe-qc supports, ~90 standard libint sets (sto-3g,
6-31G\*, cc-pVxZ, def2-\*, ANO-RCC, …) plus the CRYSTAL pob-TZVP
/ pob-DZVP-rev2 / pob-TZVP-rev2 sets for periodic crystals. No
`LIBINT_DATA_PATH` to set, no separate basis-set fetch.

### How to run the snippets below

Each numbered step is a standalone Python snippet. Save it to a file
(``step1.py``, ``step2.py``, …) **outside the repo** so any
side-effect files (cube grids etc.) don't land in the source tree:

```sh
mkdir -p ~/vibeqc-runs/quickstart
cd ~/vibeqc-runs/quickstart
# save step1.py here
```

Run it with the **virtual-env's Python** so ``import vibeqc``
resolves, give the full path since you're no longer in the repo:

```sh
~/path/to/vibe-qc/.venv/bin/python step1.py
```

Replace ``~/path/to/vibe-qc/`` with wherever ``git clone`` landed.

Or activate the venv once per shell session and use the bare
``python`` command (works from any directory):

```sh
source ~/path/to/vibe-qc/.venv/bin/activate    # bash / zsh; one-time per terminal
python step1.py                                # uses the venv's python automatically
```

If you see ``ModuleNotFoundError: No module named 'vibeqc'``, you
ran the wrong Python, make sure you're either calling the venv's
``python`` by full path or have activated the venv first. The
example output (``E(SCF) = -76.006678 Ha``, etc.) appears on
stdout; nothing is written to disk unless a snippet explicitly
calls a writer like ``write_cube_mo``.

For the wider story (controlling threads, capturing output for long
runs, running on a remote machine via SSH, common errors), see
[running](running.md).

## 1. Molecular HF on H₂O

```python
import vibeqc as vq
import numpy as np

mol = vq.Molecule([
    vq.Atom(8, [ 0.0,  0.00,  0.00]),
    vq.Atom(1, [ 0.0,  1.43, -0.98]),
    vq.Atom(1, [ 0.0, -1.43, -0.98]),
])
basis = vq.BasisSet(mol, "6-31g*")
result = vq.run_rhf(mol, basis)

eps = np.asarray(result.mo_energies)
homo = mol.n_electrons() // 2 - 1
print(f"E(SCF)  = {result.energy:.6f} Ha")
print(f"ε(HOMO) = {eps[homo]*27.211386:.2f} eV")
print(f"gap     = {(eps[homo+1] - eps[homo])*27.211386:.2f} eV")
```

What you should see:

```
E(SCF)  = -76.006678 Ha
ε(HOMO) = -13.56 eV
gap     = 19.67 eV
```

10 SCF iterations, ~6 ms. Atomic positions are in **bohr** (vibe-qc's
internal unit), pass Ångström values via `from_xyz()` if you'd
rather work in those.

### What a full vibe-qc output looks like

The bare-API call above just prints the four lines you asked for.
If you'd rather have vibe-qc's full **banner + SCF trace + orbital
table + properties block** written to a file (the format you'd paste
into a paper's SI), use [`run_job`](tour.md).

```python
# doc-audit: skip - continues the water setup from the runnable block above
# The same calculation as above, via run_job with full output:
result = vq.run_job(mol, basis="6-31g*", method="rhf")

# For large basis sets, pick an iterative solver:
result = vq.run_job(mol, basis="def2-svp", method="rhf", solver="davidson")
# solver="lobpcg" is the fastest iterative option for periodic
# drivers and ROHF; for molecular RHF/UHF/RKS/UKS it falls back
# to the Davidson path. See the solver framework guide for the
# full matrix.
```

That's what the
canonical [`examples/molecular/input-h2o-rhf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-rhf.py)
script does. It loads `examples/h2o.xyz` from the parent directory, so preserve
that parent/`molecular/` layout when copying the example. It writes `output-h2o-rhf.out`,
`output-h2o-rhf.system`, `output-h2o-rhf.molden`,
`output-h2o-rhf.xyz`, and the citation siblings beside the script
or in whatever project directory you copied it to. The top of the
`.out` file starts with the provenance banner:

```text
vibe-qc <version> "<codename>"
git: <sha> (<branch>[, dirty])
libint: <version>  libxc: <version>  spglib: <version>
```

The banner block at the top records vibe-qc's version, codename,
git revision, and linked native-library versions, that's your
provenance line for any calculation that ends up in a paper. See
[good practices: reproducibility](good_practices.md#reproducibility)
for the convention.

Browse [more example scripts](example_outputs.md) covering DFT,
open-shell, periodic, ASE, and visualization calculations.

→ Full theory + variants in [Molecular Hartree-Fock](tutorial/molecular_hf.md).

## 2. Open-shell SCF on OH•

vibe-qc's spin information lives **on the `Molecule`**, not on the
SCF-driver options. Set `multiplicity` (= 2S + 1) when you build the
molecule and the rest is automatic:

```python
import vibeqc as vq

oh = vq.Molecule(
    atoms=[vq.Atom(8, [0, 0, 0]), vq.Atom(1, [0, 0, 1.832])],
    multiplicity=2,                   # doublet, one unpaired electron
)
basis = vq.BasisSet(oh, "6-31g*")
result = vq.run_uhf(oh, basis)

# UHF returns separate alpha and beta MO sets — that's the open-shell
# signature. Sorted ascending in energy on each spin.
eps_a = result.mo_energies_alpha
eps_b = result.mo_energies_beta

# Spin = multiplicity − 1 = number of unpaired electrons.
# n_α = (n_e + spin)/2,  n_β = (n_e − spin)/2.
spin  = oh.multiplicity - 1            # 1 for a doublet
n_e   = oh.n_electrons()
n_a   = (n_e + spin) // 2              # α electrons
n_b   = (n_e - spin) // 2              # β electrons
homo_a, homo_b = n_a - 1, n_b - 1      # last occupied on each spin

# SOMO = highest singly-occupied orbital. For a doublet, that's the
# alpha HOMO (β has one fewer electron, so this orbital is empty in β).
somo_eps = max(eps_a[homo_a], eps_b[homo_b])

print(f"E(UHF)  = {result.energy:.6f} Ha   "
      f"({result.n_iter} iters)")
print(f"ε(SOMO) = {somo_eps*27.211386:.2f} eV")
print(f"gap (α) = {(eps_a[homo_a+1] - eps_a[homo_a])*27.211386:.2f} eV")
print(f"gap (β) = {(eps_b[homo_b+1] - eps_b[homo_b])*27.211386:.2f} eV")
print(f"⟨S²⟩    = {result.s_squared:.4f}  (ideal {result.s_squared_ideal:.4f})")
# E(UHF)  = -75.380943 Ha   (23 iters)
# ε(SOMO) = -13.71 eV
# gap (α) = 20.97 eV
# gap (β) = 17.32 eV
# ⟨S²⟩    = 0.7553  (ideal 0.7500)
```

The α / β split, different HOMO-LUMO gaps for the two spin sets, is
the open-shell signature: in a doublet, alpha has one more occupied
orbital than beta (the SOMO), so the two spin spectra converge to
slightly different eigenvalues. RHF would force them equal and miss
this physics.

The `⟨S²⟩` diagnostic shows the spin contamination in this UHF
solution: 0.7553 vs. the ideal 0.75 for a pure doublet. **A small
excess (here ~0.005) is normal**, larger excesses (e.g. ⟨S²⟩ > 0.85
on a doublet) indicate a UHF state that has mixed in a higher-spin
configuration, usually a sign that you should switch to a multi-
reference method or accept the open-shell singlet was metastable.

There is **no `spin` field on `UHFOptions`**, the molecule is the
single source of truth, which keeps every RHF / UHF / RKS / UKS run
against the same system consistent. Triplet O₂ would just be
`Molecule(..., multiplicity=3)`. See the
[molecules user guide](user_guide/molecules.md#configuring-open-shell-systems)
for the full story.

```{note}
**ROHF and ROKS ship in v0.13.0+ (Wisesa's Fox).** Restricted
open-shell Hartree-Fock (`method="rohf"`) and Kohn-Sham (`method="roks"`)
have shipped since v0.13.0, with analytic ROHF gradients, ASE Calculator
integration, and spin-pure `cas_reference="rohf"`. See the
[ROHF user guide](user_guide/rohf.md) and the [open-shell tutorial](tutorial/open_shell.md).
A multireference treatment (CASCI / selected-CI / DMRG via `vibeqc.solvers`,
orbital-optimised CASSCF with analytic gradients, NEVPT2 / CASPT2) is
also available.
```

→ Full open-shell coverage in [Open-shell systems (UHF, UKS, UMP2)](tutorial/open_shell.md).

## 3. Periodic SCF on a real solid

This is the headline. Set up a 2-atom cubic LiH cell, point the
SCF dispatcher at the EWALD_3D Coulomb method, and run:

```python
import vibeqc as vq
import numpy as np

a = 4.5                                          # cubic edge, bohr
shift = 0.05                                     # off-FFT-grid shift
sysp = vq.PeriodicSystem(
    dim=3, lattice=np.eye(3) * a,
    unit_cell=[
        vq.Atom(3, [shift,         shift,         shift        ]),
        vq.Atom(1, [a/2 + shift,   a/2 + shift,   a/2 + shift  ]),
    ],
)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

opts = vq.PeriodicSCFOptions()
opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
opts.lattice_opts.cutoff_bohr = 10.0
opts.lattice_opts.nuclear_cutoff_bohr = 15.0

result = vq.run_rhf_periodic_gamma_scf(
    sysp, basis, opts, omega=0.5, spacing_bohr=0.5,
)
print(f"E(RHF)/cell = {result.energy:.6f} Ha   "
      f"converged = {result.converged}")
# E(RHF)/cell = -268.738877 Ha   converged = True
```

Two things are doing real work here:

1. The dispatcher (`run_rhf_periodic_gamma_scf` /
   `run_rhf_periodic_scf`) inspects
   `opts.lattice_opts.coulomb_method` and routes to either the
   direct-truncated path (default) or the **EWALD_3D** path
   (recommended for any 3D bulk, the long-range Coulomb sum is
   conditionally convergent in 3D and Ewald is the textbook fix).
2. `omega = 0.5` is the screening parameter that splits the Coulomb
   sum into a short-range real-space piece + a long-range
   reciprocal-space piece. The total is **ω-invariant**, the same
   total energy comes out for any ω in (0.1, 1.0). See
   [the Ewald user guide](user_guide/ewald.md) for the full story.

This step takes ~30 s on a single core because the FFT grid build
dominates at small system size. Multi-k beats Γ-only for production
work, `monkhorst_pack(sysp, [4, 4, 4])` and `run_rhf_periodic_scf`
take a kmesh argument; see [Periodic HF on a 1D H chain](tutorial/periodic_hf.md).

For DFT on a solid, the matching dispatcher is
`run_rks_periodic_scf`. Multi-k KS-DFT via GDF, GPW, and BIPOLE
routes all shipped; the EWALD_3D legacy route is superseded.
See [Periodic KS-DFT](tutorial/periodic_dft.md)
and [Tight-cell DFT with the periodic Becke partition](tutorial/tight_cell_dft.md) for the
periodic-Becke-partition story on tight DFT cells.

## 4. Visualize an orbital

Write the H₂O HOMO to a Gaussian cube file:

```python
import vibeqc as vq
import numpy as np

# Reproduce step 1's H2O / RHF / 6-31G* result so this snippet runs
# standalone. (If you appended this to step1.py instead, drop these
# next 6 lines — `mol`, `basis`, `result` are already in scope.)
mol = vq.Molecule([
    vq.Atom(8, [ 0.0,  0.00,  0.00]),
    vq.Atom(1, [ 0.0,  1.43, -0.98]),
    vq.Atom(1, [ 0.0, -1.43, -0.98]),
])
basis = vq.BasisSet(mol, "6-31g*")
result = vq.run_rhf(mol, basis)

homo_idx = mol.n_electrons() // 2 - 1
vq.write_cube_mo(
    "/tmp/h2o_homo.cube",
    np.asarray(result.mo_coeffs), homo_idx, basis, mol,
    spacing=0.2, padding=3.0,
)
```

Open the result in Avogadro, VMD, PyMOL, ChimeraX, or any other
cube-aware viewer:

```sh
open -a Avogadro2 /tmp/h2o_homo.cube           # macOS
vmd /tmp/h2o_homo.cube                          # cross-platform
```

```{tip}
**You do not need a cube file for this.** The snippet above uses the
low-level `run_rhf` + `write_cube_mo` path to show what a cube is. A
normal `run_job` already writes a `.qvf` archive (it is the default),
and asking it for orbitals puts them inside that one file:

    run_job(mol, basis="6-31g*", method="rhf",
            write_cube=["homo", "lumo"], output="water")

Then `vibe-view open water.qvf` for the browser viewer, or
`vibe-view tui water.qvf` to read it over SSH with no display server.
See [QVF and vibe-view](visualization.md).
```

You'll see the H₂O lone-pair π orbital, two lobes perpendicular to
the molecular plane.

→ Cube + Molden output, periodic Bloch orbitals, density grids,
electrostatic potentials in
[Orbital and density visualization](tutorial/orbital_visualization.md) and
[Periodic orbital cubes and XSF files](tutorial/periodic_orbital_cubes.md).

## What just happened in 30 seconds

You ran:

| Calculation | Type | Wall time | What it demonstrates |
|---|---|---|---|
| H₂O / RHF / 6-31G\* | Closed-shell molecular SCF | ~6 ms | Standard molecular HF |
| OH• / UHF / 6-31G\* | Open-shell molecular SCF | ~6 ms | Spin lives on Molecule |
| 2-atom LiH cubic / RHF / sto-3g via EWALD_3D | 3D periodic SCF | ~30 s | Dispatcher + Ewald summation |
| H₂O HOMO → .cube | Visualization | ~40 ms | Cube-file output |

…using **only the public Python API**, **only basis sets bundled
with the wheel**, and **no environment variables**. Quick, clean, reproducible.

## Where to go next

- **Read this once: [good practices](good_practices.md)**, the
  working conventions nobody tells you (file layout, naming,
  reproducibility, when to trust a number, what to try when SCF
  diverges). Five minutes now saves a lot of "why isn't this
  converging" later.
- **The full API tour**, [tour](tour.md) covers `run_job` (the
  classic-QC-program input-file experience), the ASE Calculator,
  Python `logging` integration, dropping in custom basis sets, and
  how to run the test suite.
- **Theory + variants for what you just did**, work through the
  numbered tutorials. [Molecular Hartree-Fock](tutorial/molecular_hf.md)
  through [Submitting a job to a remote machine with `vq`](tutorial/vq_queue_remote_job.md),
  grouped by topic in [the tutorial index](tutorial/index.md).
  New: [non-mean-field solvers](tutorial/non_hf_solvers.md),
  Selected-CI, DMRG, v2RDM, and transcorrelated methods.
- **API reference**, [api/index](api/index.md) lists every
  exported symbol with its full signature.
- **Topic-focused user guides**, [basis sets](user_guide/basis_sets.md),
  [Ewald summation](user_guide/ewald.md),
  [k-point sampling](user_guide/k_points.md),
  [SCF convergence](user_guide/scf_convergence.md), and more under
  [user_guide/index](user_guide/index.md).
- **Examples directory**, [`examples/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples)
  has runnable input scripts in the style of classic QC programs
  (Gaussian / ORCA / NWChem). Each one is self-contained and writes
  its output alongside.

If something here didn't work, check
[installation](installation.md) for environment-specific notes or
open an issue at
<https://github.com/vibe-qc/vibe-qc/issues>.
