# Tour of the vibe-qc API

This page walks through the main ways to use vibe-qc, `run_job`, the
programmatic SCF entry points, the ASE Calculator, logging, custom
basis sets, and running the test suite. For a much shorter "I just
installed it, what's the smallest interesting thing?" walkthrough,
see [quickstart](quickstart.md) instead.

Everything below assumes you've finished [installation.md](installation.md)
and activated the virtualenv:

```sh
source .venv/bin/activate   # bash/zsh
# or run commands as .venv/bin/python / .venv/bin/pytest
```

For the full "how to invoke vibe-qc scripts" reference, venv
activation patterns, controlling OpenMP threads, capturing output
for long runs, SSH workflows, common errors, see
[running](running.md).

Any time you want to confirm which vibe-qc you're running, print the
banner:

```python
from vibeqc import print_banner
print_banner()
```

The following historical release banner illustrates the layout; use your
installed banner for the actual version, codename and library versions:

```
╔════════════════════════════════════════════════════════════════════════╗
║ Release v0.15.0 "Neese's Cheetah"  --  Quantum chemistry for molecules  ║
║ and solids                                                              ║
║ © Michael F. Peintinger · MPL 2.0  ·  https://vibe-qc.com               ║
║ linked: libint 2.13.1 · libxc 7.0.0 · spglib 2.7.0 · libecpint 1.0.7    ║
║         · fftw3 3.3.10                                                  ║
╚════════════════════════════════════════════════════════════════════════╝
```

A development banner also records the source revision. Retain it with the
calculation, and check the loaded native module after a rebuild: a source
revision alone does not prove an older extension was rebuilt. A historical
example of the development layout is:

```
╔════════════════════════════════════════════════════════════════════════╗
║ dev <version> "Neese's Cheetah" (main @ <sha>)  --  Quantum             ║
║ chemistry for molecules and solids                                      ║
║ © Michael F. Peintinger · MPL 2.0  ·  https://vibe-qc.com               ║
║ linked: libint 2.13.1 · libxc 7.0.0 · spglib 2.7.0 · libecpint 1.0.7    ║
║         · fftw3 3.3.10                                                  ║
╚════════════════════════════════════════════════════════════════════════╝
```

A `dirty` flag appears in the descriptor when the working tree has
uncommitted changes. The formatted SCF trace
(`vibeqc.format_scf_trace`) prepends this banner by default, so a
persisted SCF log always carries its provenance, see
[release_process.md](release_process.md) for how to pin a calculation
to a specific build.

## Choose an entry point

| Task | Entry point | What to expect |
|---|---|---|
| Molecular calculation with output files | `run_job` | Method dispatch and coordinated text, QVF, geometry and provenance output |
| Periodic calculation | `run_periodic_job` | Periodic route selection and cell/k-point controls; see [periodic methods](user_guide/periodic_methods.md) |
| Compose a calculation programmatically | `run_rhf`, `run_uhf`, `run_rks`, and related drivers | A result object; use the output helpers when you need persistent files |
| Use ASE structures or optimizers | `vibeqc.ase.VibeQC` | ASE energies in eV, positions in angstrom and forces in eV/angstrom |
| Solve a supplied eigenproblem | `vibeqc.solvers` | Eigenpairs and solver diagnostics; see [solver framework](user_guide/solver_framework.md) |

Run examples in a calculation directory outside the source checkout, with the
installed environment active. Blocks marked as continuations reuse objects
from earlier sections; the isolated-atom example uses a separate variable so
it does not replace the water molecule used in the DFT example.

## 0. The simple way: `run_job`

If you just want the classic QC-program experience, write an input
script, run it, read a text log, `vibeqc.run_job` dispatches everything
for you:

```python
from vibeqc import Atom, Molecule, run_job

# Coordinates in bohr; no external input file is required.
mol = Molecule([
    Atom(8, [0.0, 0.0, 0.0]),
    Atom(1, [0.0, 1.5, -1.16]),
    Atom(1, [0.0, -1.5, -1.16]),
])

result = run_job(
    mol,
    basis="6-31g*",
    method="rhf",              # or "rks" / "uhf" / "uks" / "auto"
    output="output-h2o-rhf",   # file name stem, relative to this run directory
    write_molden_file=True,     # request this optional export explicitly
)
if not result.converged:
    raise RuntimeError("SCF did not converge; inspect output-h2o-rhf.out")
print(f"E(RHF) = {result.energy:.10f} Ha")
```

Produces:

- `output-h2o-rhf.out`, banner, atom table, SCF iteration trace,
  energy components (for DFT), orbital-energy table with HOMO/LUMO
  markers, and the HOMO-LUMO gap.
- `output-h2o-rhf.molden`, molecular orbitals for any molden-aware
  viewer (Jmol, Avogadro, Molden, IQmol, MolView).
- `output-h2o-rhf.qvf`, the same calculation as one self-describing
  archive: structure, wavefunction, atom properties, SCF history, the
  run record and the citations. This is written by default. Open it
  with `vibe-view open output-h2o-rhf.qvf`, or `vibe-view tui` to read
  it in a terminal with no display server. See
  [QVF and vibe-view](visualization.md).
- `output-h2o-rhf.system`, the machine-readable run record; preserve it with
  the input, `.out` file and native runtime identity.
- `output-h2o-rhf.xyz` and the citation files, including `.bibtex` and
  `.references`, for the geometry and methods used. See
  [output files](user_guide/output_files.md) for defaults and optional exports.

A result object and an output archive are not, by themselves, proof of
convergence or scientific validity. Inspect the convergence status, selected
method and state before comparing energies. See
[planning a calculation](tutorial/planning_a_calculation.md).

Geometry optimization is one flag:

```python
# doc-audit: skip - continues the run_job setup directly above
run_job(
    mol, basis="6-31g*", method="rhf",
    output="output-h2o-opt",
    optimize=True, fmax=0.01,    # eV/Å; ASE BFGS, or native L-BFGS-B, or Brent
)
# Adds: output-h2o-opt.traj — per-step trajectory, animated via
#       `ase gui output-h2o-opt.traj`.
```

A spread of runnable example inputs ships under
[`examples/molecular/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/molecular),
single-point RHF / DFT (`input-h2o-rhf.py`, `input-h2o-dft.py`),
geometry optimization (`input-h2o-opt.py`), open-shell UHF
(`input-oh-radical.py`), MP2 / double hybrids
(`input-h2o-scs-mp2.py`, `input-h2o-b2plyp.py`), the SCF
Fock-build modes (`input-h2o-rhf-direct.py`,
`input-h2o-rhf-conventional.py`), cube output, dispersion-corrected
optimization, and more, plus
[`examples/periodic/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic)
for the solid-state side. Copy any of them as a template for your
own systems.

### Methods `run_job` understands

The following table is a selection of molecular entry points, not the full
method inventory. See [methods and status](methods.md) for correlation,
semiempirical and composite routes and their qualification limits.

| `method=` | What runs |
|---|---|
| `"rhf"` / `"uhf"` | Restricted / unrestricted Hartree-Fock |
| `"rks"` / `"uks"` | Restricted / unrestricted Kohn-Sham DFT (set `functional=`) |
| `"rohf"` / `"roks"` | Restricted-open HF / KS; see the [route-specific controls](user_guide/rohf.md) |
| `"auto"` | picks RHF / UHF / RKS / UKS from the molecule + `functional=` |
| `"fci"` | exact Full CI in the (active) orbital space |
| `"selected_ci"` | CIPSI-style selected CI |
| `"dmrg"` | two-site DMRG |
| `"v2rdm"` | variational 2-RDM |
| `"transcorrelated_ci"` | transcorrelated-Hamiltonian selected CI |

The wavefunction methods accept an `active_space=(n_orbitals,
n_electrons)` kwarg for a frozen-core / CAS-style window, plus
per-solver option objects (`selected_ci_options=`, `dmrg_options=`,
`v2rdm_options=`, `transcorrelated_options=`). See the
[non-mean-field solvers user guide](user_guide/non_hf_solvers.md)
and the [`non_hf_solvers` tutorial](tutorial/non_hf_solvers.md).

The remainder of this tour covers the programmatic API directly,
useful when `run_job` isn't enough and you want to wire things up by
hand.

## 1. Load a molecule

From an XYZ file in your run directory, such as a copy of the repository's
`examples/h2o.xyz` saved as `h2o.xyz`:

```python
from vibeqc import from_xyz
mol = from_xyz("h2o.xyz", charge=0, multiplicity=1)
```

XYZ positions are in angstrom; `from_xyz` converts them to internal bohr.
Charge and multiplicity describe the supplied nuclei: changing water's charge
does not remove a hydrogen atom to make hydroxide. For a hydroxide anion, use
an O-H geometry and a singlet with charge -1:

```python
from vibeqc import Atom, Molecule
hydroxide = Molecule([
    Atom(8, [0.0, 0.0, 0.0]),
    Atom(1, [0.0, 0.0, 1.83]),  # illustrative O-H distance in bohr
], charge=-1, multiplicity=1)
```

The hydroxide example defines a system only; it does not establish an optimized
geometry or a suitable production basis for an anion.

Programmatic construction (positions in **bohr**, not Å):

```python
from vibeqc import Atom, Molecule
mol = Molecule([
    Atom(8, [0.0, 0.0,  0.0]),
    Atom(1, [0.0, 1.5, -1.16]),
    Atom(1, [0.0, -1.5, -1.16]),
])
```

Or via ASE (positions in Å), which also handles CIF, PDB, POSCAR, Gaussian
inputs, and more:

```python
from ase.io import read
atoms = read("h2o.xyz")          # .xyz / .cif / .pdb / .poscar ...
# use atoms.calc = VibeQC(...) pattern below
```

## 2. Hartree-Fock energy

```python
# doc-audit: skip - uses the molecule constructed in section 1
from vibeqc import BasisSet, run_rhf, RHFOptions

basis = BasisSet(mol, "6-31g*")
result = run_rhf(mol, basis)
print(f"E(RHF/6-31G*) = {result.energy:.6f} Ha")
if not result.converged:
    raise RuntimeError(f"RHF did not converge after {result.n_iter} iterations")
print(f"converged in {result.n_iter} iterations")
```

Key options (`RHFOptions`):

| Field | Default | Meaning |
|---|---|---|
| `max_iter` | 100 | hard limit on SCF cycles |
| `conv_tol_energy` | 1e-8 | \|ΔE\| threshold (Hartree) |
| `conv_tol_grad` | 1e-6 | ‖F·D·S − S·D·F‖ threshold |
| `damping` | 0.5 | initial fraction of previous density to mix in; 0 disables static density damping |
| `dynamic_damping` | True | adapt the damping fraction during SCF |
| `use_diis` | True | enable the selected Fock extrapolator |
| `scf_accelerator` | `SCFAccelerator.EDIIS_DIIS` | hybrid energy-DIIS / commutator-DIIS accelerator |
| `diis_start_iter` | 2 | DIIS begins at this iteration |
| `diis_subspace_size` | 8 | history length; 2-12 for EDIIS, ADIIS, and their hybrids |
| `initial_guess` | `InitialGuess.AUTO` | conservative system- and route-aware selection; see [initial guesses](user_guide/initial_guess.md) for explicit constructions and limits |
| `scf_mode` | `SCFMode.AUTO` | Fock-build mode: `AUTO`, `CONVENTIONAL` (in-core ERI tensor), or `DIRECT` (on-the-fly screened integrals) |

These are the molecular `RHFOptions` defaults in this source revision, not a
prescription for all periodic or open-shell routes. AUTO chooses a supported
seed construction; it does not promise the lowest electronic state. Static
and dynamic damping are separate controls, so setting `damping=0` alone does
not turn off adaptive damping. See [SCF convergence](user_guide/scf_convergence.md)
for how these options interact.

For large systems, set `scf_mode = SCFMode.DIRECT` to avoid retaining the
4-index ERI tensor. The AUTO default in this version switches above 140 basis
functions; memory-capped workflows can lower
`scf_mode_auto_threshold` to 107 so the dense tensor stays below 1 GiB. See
[Direct SCF or in-core integrals?](tutorial/direct_scf_memory_tradeoff.md) for
the CPU-time and memory tradeoff, and
[SCF Fock-build modes](user_guide/scf_modes.md) for every option.

`result: RHFResult` fields include `energy`, `e_electronic`, `n_iter`,
`converged`, `mo_energies` (ascending), `mo_coeffs` (AO-to-MO matrix,
columns are MOs), `density`, `fock`, `scf_trace`.

### Open-shell HF (UHF)

```python
from vibeqc import Atom, BasisSet, Molecule, run_uhf, UHFOptions

h_atom = Molecule([Atom(1, [0, 0, 0])], multiplicity=2)   # H atom (doublet)
result = run_uhf(h_atom, BasisSet(h_atom, "sto-3g"))
print(f"E(UHF) = {result.energy:.6f} Ha")
print(f"<S^2> = {result.s_squared:.4f}  (ideal {result.s_squared_ideal:.4f})")
```

`UHFResult` has separate α/β attributes: `mo_energies_alpha`,
`mo_energies_beta`, `mo_coeffs_alpha`, `mo_coeffs_beta`, `density_alpha`,
`density_beta`.

## 3. Density Functional Theory (DFT)

Closed-shell Kohn-Sham:

```python
# doc-audit: skip - continues with the molecule constructed in section 1
from vibeqc import BasisSet, run_rks, RKSOptions

basis = BasisSet(mol, "6-31g*")
opts = RKSOptions()
opts.functional = "B3LYP"
# opts.grid.n_radial = 99   # optional — bump quality
result = run_rks(mol, basis, opts)
print(f"E(B3LYP/6-31G*) = {result.energy:.6f} Ha")
print(f"  E_core   = {result.e_electronic - result.e_coulomb - result.e_hf_exchange - result.e_xc:.4f}")
print(f"  E_J      = {result.e_coulomb:.4f}")
print(f"  α·E_K    = {result.e_hf_exchange:.4f}   (= 0 for pure DFT)")
print(f"  E_xc     = {result.e_xc:.4f}")
print(f"  E_nuc    = {result.e_nuclear:.4f}")
```

Supported functional names:

| Name | Family | HF-exchange fraction | libxc components |
|---|---|---:|---|
| `LDA` / `SVWN` | LDA | 0 | Slater + VWN5 |
| `SLATER` | LDA | 0 | Slater exchange only |
| `PBE` | GGA | 0 | XC_GGA_X_PBE + XC_GGA_C_PBE |
| `BLYP` | GGA | 0 | XC_GGA_X_B88 + XC_GGA_C_LYP |
| `B3LYP` (= `B3LYP5`) | hybrid GGA | 0.20 | XC_HYB_GGA_XC_B3LYP5 (VWN5; ORCA / CRYSTAL convention) |
| `B3LYP/G` (= `B3LYPG`) | hybrid GGA | 0.20 | XC_HYB_GGA_XC_B3LYP (VWN-RPA; Gaussian convention) |

Any libxc ID or comma-separated list of IDs works too, e.g.
`"1,7"` (Slater + VWN5) or `"101,130"` (PBE). `B3LYP` follows
the ORCA definition (VWN5 local correlation, also TURBOMOLE's,
ADF's, CRYSTAL's, and CP2K's reading of the name; `B3LYP5` is the
explicit spelling); `B3LYP/G` / `B3LYPG` select the
Gaussian-compatible VWN-RPA flavor that Gaussian, libxc, PySCF,
and Psi4 mean by the bare name, see
[user_guide/functionals.md](user_guide/functionals.md) for the
full story.

### DFT grid quality

```python
from vibeqc import RKSOptions

opts = RKSOptions()
opts.functional = "PBE"
opts.grid.n_radial = 75      # Treutler-Ahlrichs radial points per atom (default)
opts.grid.n_theta  = 17      # Gauss-Legendre θ points per radial shell
opts.grid.n_phi    = 36      # uniform φ points per radial shell
opts.grid.becke_k  = 3       # Becke cell smoothing iterations
# Total points per atom ≈ n_radial · n_theta · n_phi = 46k by default.
```

## 4. ASE Calculator, the easy path

```python
from ase.build import molecule
from ase.optimize import BFGS
from vibeqc.ase import VibeQC

# Hartree-Fock single point
atoms = molecule("H2O")
atoms.calc = VibeQC(basis="6-31g*")
print(f"E = {atoms.get_potential_energy():.4f} eV")

# Geometry optimization (HF forces work)
BFGS(atoms).run(fmax=0.01)      # fmax in eV/Å
print(f"Optimised: r(O-H) = {atoms.get_distance(0, 1):.4f} Å")

# DFT single point
atoms.calc = VibeQC(basis="6-31g*", functional="B3LYP")
print(f"E(B3LYP) = {atoms.get_potential_energy():.4f} eV")

# Open-shell HF (radical): multiplicity=2 → routes to UHF automatically
from ase import Atoms
oh = Atoms(symbols=["O", "H"], positions=[(0, 0, 0), (0.97, 0, 0)])
oh.calc = VibeQC(basis="sto-3g", multiplicity=2)
print(f"E(OH) = {oh.get_potential_energy():.4f} eV")
```

### Method routing matrix

| `functional`            | multiplicity | driver | forces? |
|---|---:|---|:-:|
| `None`                  | 1 | RHF | ✅ |
| `None`                  | ≥ 2 | UHF | ✅ |
| any functional string   | 1 | RKS | ✅ |
| any functional string   | ≥ 2 | UKS | ✅ |

### Reading arbitrary structure files

```python
from ase.io import read
from vibeqc.ase import VibeQC

atoms = read("my_structure.cif")       # .cif, .xyz, .poscar, .pdb, Gaussian .com, …
atoms.calc = VibeQC(basis="def2-tzvp", functional="PBE")
energy = atoms.get_potential_energy()
```

## 5. Logging the SCF

vibe-qc uses the stdlib `logging` module; the library is silent until you
configure it:

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# or per-logger:
logging.getLogger("vibeqc.scf").setLevel(logging.DEBUG)   # per-iteration table
```

Loggers:

- `vibeqc`, top-level namespace (nothing emits directly)
- `vibeqc.scf`, SCF iteration tables (DEBUG) + convergence summary (INFO)
- `vibeqc.ase`, method header when the ASE Calculator fires

Post-hoc log of a result you already have:

```python
# doc-audit: skip - formats the result produced in an earlier section
from vibeqc import log_scf_trace, format_scf_trace
log_scf_trace(result)              # emits via logging
print(format_scf_trace(result))    # returns a multi-line string
```

## 6. Geometry optimization + visualization

```python
# doc-audit: skip - continues with the ASE atoms and calculator defined above
from ase.optimize import BFGS
from ase.visualize import view

atoms.calc = VibeQC(basis="6-31g*")
opt = BFGS(atoms, trajectory="opt.traj")   # writes every step
opt.run(fmax=1e-3)                          # eV/Å

# Replay:
from ase.io import read
frames = read("opt.traj", index=":")
view(frames)                                # opens the ASE GUI
```

## 7. Custom basis sets

A basis set is a QVF-Basis record; the `.g94` text libint reads is
rendered from it. Turn your Gaussian `.g94` file into a record and render it
into a directory libint reads:

```sh
vibe-qc-basis convert my-basis.g94 -o my-basis.qvf.json --kind hand-written
mkdir -p my-library/basis && cp my-basis.g94 my-library/basis/
export LIBINT_DATA_PATH=$PWD/my-library
```

See
[`python/vibeqc/basis_library/README.md`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/basis_library/README.md)
for how a record joins the bundled library.

```python
# doc-audit: skip - uses the molecule constructed in section 1
basis = BasisSet(mol, "my-basis")      # libint picks it up by name
```

A user-set `LIBINT_DATA_PATH` wins over the bundled library, so a
directory of your own that carries a complete library plus your file
replaces a standard basis without touching the shipped records.

## 8. Accessing the full API

The top-level `vibeqc` module exposes everything you're likely to need:

```python
import vibeqc
dir(vibeqc)
# Atom, Molecule, BasisSet, RHFOptions, RHFResult, UHFOptions, UHFResult,
# RKSOptions, RKSResult, Functional, Grid, GridOptions, SCFIteration,
# from_xyz, run_rhf, run_uhf, run_rks, compute_overlap, compute_kinetic,
# compute_nuclear, compute_eri, compute_gradient, compute_gradient_uhf,
# build_grid, evaluate_ao, evaluate_ao_with_gradient, log_scf_trace,
# format_scf_trace, hello, libint_version, ...
```

Everything returning matrices returns NumPy arrays (zero-copy views of
Eigen data where possible).

## 9. Check a change with the relevant tests

Use the checkout's interpreter and select the smallest relevant test file or
lane. A routine source edit does not require the complete research inventory
or a clean native build. The [testing policy](testing.md) distinguishes source
checks, numerical regressions and release validation; the
[lane inventory](developer_test_lanes.md) lists focused commands.

For a native calculation test, first confirm that the loaded Python package
and compiled extension match the intended source, as described in
[contributor setup](contributor_setup.md). Tests comparing with an optional
reference program can skip when that program is unavailable. A skipped
comparison is not a numerical validation result. Documentation-only work
needs a focused text, link or example check.
