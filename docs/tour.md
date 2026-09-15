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

For an installed wheel of a tagged release, the banner reads:

```
╔════════════════════════════════════════════════════════════════════════╗
║ Release v0.15.0 "Neese's Cheetah"  --  Quantum chemistry for molecules  ║
║ and solids                                                              ║
║ © Michael F. Peintinger · MPL 2.0  ·  https://vibe-qc.com               ║
║ linked: libint 2.13.1 · libxc 7.0.0 · spglib 2.7.0 · libecpint 1.0.7    ║
║         · fftw3 3.3.10                                                  ║
╚════════════════════════════════════════════════════════════════════════╝
```

Running from a development checkout, the banner instead carries the
exact git revision so a calculation log unambiguously identifies the
build that produced it:

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

## 0. The simple way: `run_job`

If you just want the classic QC-program experience, write an input
script, run it, read a text log, `vibeqc.run_job` dispatches everything
for you:

```python
from vibeqc import Molecule, run_job

mol = Molecule.from_xyz("examples/h2o.xyz")

run_job(
    mol,
    basis="6-31g*",
    method="rhf",              # or "rks" / "uhf" / "uks" / "auto"
    output="output-h2o-rhf",   # file name stem
)
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

The `method=` keyword accepts the four mean-field SCF methods plus
the `vibeqc.solvers` wavefunction methods and an `"auto"` resolver:

| `method=` | What runs |
|---|---|
| `"rhf"` / `"uhf"` | Restricted / unrestricted Hartree-Fock |
| `"rks"` / `"uks"` | Restricted / unrestricted Kohn-Sham DFT (set `functional=`) |
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

From an XYZ file:

```python
from vibeqc import Molecule, from_xyz
mol = from_xyz("examples/h2o.xyz")                # uses defaults charge=0, mult=1
# or:
mol = from_xyz("examples/h2o.xyz", charge=-1, multiplicity=1)  # hydroxide
```

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
atoms = read("examples/h2o.xyz")          # .xyz / .cif / .pdb / .poscar ...
# use atoms.calc = VibeQC(...) pattern below
```

## 2. Hartree-Fock energy

```python
# doc-audit: skip - uses the molecule constructed in section 1
from vibeqc import BasisSet, run_rhf, RHFOptions

basis = BasisSet(mol, "6-31g*")
result = run_rhf(mol, basis)
print(f"E(RHF/6-31G*) = {result.energy:.6f} Ha")
print(f"converged in {result.n_iter} iterations")
```

Key options (`RHFOptions`):

| Field | Default | Meaning |
|---|---|---|
| `max_iter` | 100 | hard limit on SCF cycles |
| `conv_tol_energy` | 1e-8 | \|ΔE\| threshold (Hartree) |
| `conv_tol_grad` | 1e-6 | ‖F·D·S − S·D·F‖ threshold |
| `damping` | 0.5 | fraction of previous density to mix in (0 = off, required ~0.5 for some systems) |
| `use_diis` | True | Pulay DIIS extrapolation |
| `diis_start_iter` | 2 | DIIS begins at this iteration |
| `diis_subspace_size` | 8 | history length; 2-12 for EDIIS, ADIIS, and their hybrids |
| `initial_guess` | `InitialGuess.SAD` | `HCORE` or `SAD` (superposition of atomic densities) |
| `scf_mode` | `SCFMode.AUTO` | Fock-build mode: `AUTO`, `CONVENTIONAL` (in-core ERI tensor), or `DIRECT` (on-the-fly screened integrals) |

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

mol = Molecule([Atom(1, [0, 0, 0])], multiplicity=2)      # H atom (doublet)
result = run_uhf(mol, BasisSet(mol, "sto-3g"))
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
from vibeqc import run_rks, RKSOptions

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

Drop a Gaussian `.g94` file into `basis_library/custom/` and re-run
`./scripts/setup_basis_library.sh`. See
[`python/vibeqc/basis_library/README.md`](https://github.com/vibe-qc/vibe-qc/blob/main/python/vibeqc/basis_library/README.md)
for the file format.

```sh
cp my-basis.g94 basis_library/custom/my-basis.g94
./scripts/setup_basis_library.sh
```

```python
# doc-audit: skip - uses the molecule constructed in section 1
basis = BasisSet(mol, "my-basis")      # libint picks it up by name
```

A custom file with the same name as a standard basis (e.g.
`basis_library/custom/6-31g.g94`) overrides the standard one, useful
for replacing a contracted basis with a more accurate variant without
touching the shipped library.

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

## 9. Running the tests

```sh
.venv/bin/python -m pytest tests/ -v
```

By category:

```sh
pytest tests/test_rhf.py         # HF energies vs PySCF
pytest tests/test_uhf.py         # UHF vs PySCF
pytest tests/test_rks.py         # DFT vs PySCF
pytest tests/test_gradient.py    # analytic gradients vs PySCF
pytest tests/test_ase.py         # ASE calculator + geometry opt
```
