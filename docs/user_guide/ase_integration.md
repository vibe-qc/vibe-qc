# ASE integration

[ASE](https://ase-lib.org/) (Atomic Simulation Environment) is the
de-facto standard Python toolbox for atomic-scale simulation:
geometry optimization, molecular dynamics, vibrational analysis,
NEB, basin hopping, structure I/O, periodic-cell handling. ASE's
**Calculator** interface is calculator-agnostic, anything that can
return energy + forces (and optionally stress / dipole / Hessian /
polarizability / magmoms) plugs in directly.

`vibeqc.ase.VibeQC` is vibe-qc's ASE Calculator. With it, every
ASE workflow that works for VASP / GPAW / ORCA / PySCF works for
vibe-qc by **changing one line**:

```python
atoms.calc = VibeQC(basis="6-31g*", functional="PBE")
```

This page is the reference for what the calculator exposes, how to
build / read / constrain structures via ASE, how to bridge to
`vqfetch` and the `vq` queue, and how cross-validation against
external codes (PySCF / ORCA / CRYSTAL) works through the
**subprocess-runner pattern** (per
[Archived CLAUDE.md § 10](https://vibe-qc.com/docs/);
external programs are run out-of-process, not imported as backends).

Page layout, feel free to jump:

* [Quickstart](#quickstart), minimal one-liner usage
* [Supported methods](#supported-methods) + [property surface](#property-surface)
* [Building structures with ASE](#building-structures-with-ase)
* [Reading structures via ASE I/O](#reading-structures-via-ase-io)
* [Constraints](#constraints), frozen atoms / planes
* [Workflow examples](#workflow-examples), opt / vibrations / NEB / MD / surface
* [Dispersion](#dispersion-d3-bj)
* [ASE + `vqfetch`](#ase-vqfetch), fetched-structure → SCF
* [ASE + `vq` queue](#ase-vq-queue), remote dispatch
* [Cross-validation against external codes](#cross-validation-against-external-codes)
* [Periodic calculations](#periodic-calculations-current-state)
* [Options + logging + versions](#ase-option-translation)

## Quickstart

```python
from ase.build import molecule
from ase.optimize import BFGS
from vibeqc.ase import VibeQC

atoms = molecule("H2O")
atoms.calc = VibeQC(basis="6-31g*", functional="PBE")

BFGS(atoms).run(fmax=0.01)        # ASE's BFGS, vibe-qc forces
print(atoms.get_potential_energy())   # eV
print(atoms.get_dipole_moment())       # e Angstrom, Phase A (v0.5)
```

## Supported methods

The driver is selected automatically from the calculator's
`functional` and the molecule's multiplicity:

| `functional=` | `multiplicity` | Driver |
|---|---|---|
| `None` (default) | 1 | RHF (`run_rhf`) |
| `None` | > 1 | UHF (`run_uhf`) |
| set | 1 | RKS (`run_rks`), closed-shell DFT |
| set | > 1 | UKS (`run_uks`), open-shell DFT |

You don't pass a `method=` argument, the calculator infers it.
This keeps every RHF / UHF / RKS / UKS run on the same `Molecule`
consistent (the multiplicity is the single source of truth).

```python
# Closed-shell HF
VibeQC(basis="6-31g*")

# Open-shell HF: multiplicity comes from atoms.info["multiplicity"]
# or atoms.get_initial_magnetic_moments(); set explicitly via the
# `multiplicity=` kwarg to be unambiguous.
VibeQC(basis="6-31g*", multiplicity=2)

# Closed-shell DFT
VibeQC(basis="def2-tzvp", functional="B3LYP")

# Open-shell DFT
VibeQC(basis="def2-tzvp", functional="B3LYP", multiplicity=3)
```

## Property surface

As of v0.5, `VibeQC` implements the following ASE properties. The
ASE accessor (left column) triggers the calculator only on first
call; subsequent calls return the cached result until the geometry
or other input changes.

| ASE accessor | Property | Method coverage | Notes |
|---|---|---|---|
| `atoms.get_potential_energy()` | `energy` (eV) | RHF / UHF / RKS / UKS | + D3(BJ) when `dispersion=` set |
| `atoms.get_potential_energy(force_consistent=True)` | `free_energy` | All | Alias for `energy` (vibe-qc has no smearing entropy at the molecular SCF level). |
| `atoms.get_forces()` | `forces` (eV/Å) | All | Analytic nuclear gradients. The gradient call inherits the SCF options' `density_fit` / `aux_basis` / `cosx` / ECP settings, and the public wrapper verifies the exact XML or inline ECP provenance recorded by the SCF result. On ECP systems the ECP centres are moved onto the current atoms at every call (#643); the `*_options` you passed then describe the last geometry evaluated. ECP+CPCM gradients remain unsupported. |
| `atoms.get_dipole_moment()` | `dipole` (eÅ) | All | 3-vector. Phase A (v0.5). |
| `atoms.calc.get_property("hessian", atoms)` | `hessian` (eV/Å²) | RHF / UHF / RKS analytic; UKS via FD | 3N×3N matrix. RHF uses Phase 17b-3 CPHF; UHF uses Phase 17c per-spin CPHF; RKS uses 17c+; UKS falls back to `compute_hessian_fd`. Gas-phase ECP systems always take the FD route (the analytic kernels take no ECP options, #643); CPCM Hessians fail closed. |
| `atoms.calc.get_property("polarizability", atoms)` | `polarizability` (Å³) | RHF only (CPHF, Phase 17b-1) | 3×3 tensor. UHF / KS variants on the roadmap. |

For a gas-phase ECP calculation, ASE geometry optimization and the
finite-difference Hessian are supported for ground-state RHF/UHF/RKS/UKS. The
analytic Hessian kernels, CPCM Hessians, MR/post-HF derivatives, and
excited-state gradient/optimization routes do not accept ECPs; the calculator
automatically chooses the supported FD Hessian route for the mean-field
methods.

Properties not yet implemented (raise `PropertyNotImplementedError`):

* `magmoms`: per-atom spin density. The data exists on the result
  (`density_alpha - density_beta`) but vibe-qc has no exposed
  Mulliken-spin-population API yet; tracked as a tiny engineering
  ask.
* Periodic **stress** through the `ase.Atoms` interface. Energy +
  forces now ship via `VibeQCPeriodic` (see [Periodic
  calculations](#periodic-calculations-current-state)); true periodic
  stress is not wired onto the calculator. BIPOLE variable-cell
  optimization also fails closed in the native entry points.

## Building structures with ASE

ASE's structure-builder utilities (`ase.build.*`,
`ase.spacegroup.crystal`) produce `Atoms` objects you can pass
straight to `VibeQC` (molecular path) or convert to a
`PeriodicSystem` for the periodic path. The combination removes
hand-rolling of lattice vectors + fractional coordinates.

### Molecules: `ase.build.molecule`

```python
from ase.build import molecule
from vibeqc.ase import VibeQC

# ASE knows ~120 common small molecules by name (G2 / G3 / G2/97 set):
atoms = molecule("H2O")
atoms.calc = VibeQC(basis="6-31g*", functional="PBE")
atoms.get_potential_energy()
```

Try `molecule("benzene")`, `molecule("NH3")`, `molecule("urea")`,
`molecule("naphthalene")`, etc., the full list is in
`ase.collections.g2`.

### Periodic bulk: `ase.build.bulk`

```python
from ase.build import bulk
import vibeqc as vq

# ASE knows common crystal prototypes by name. Pass through to
# vibe-qc's PeriodicSystem via the ase↔vibeqc bridge:
si = bulk("Si", "diamond", a=5.431)              # ase.Atoms (PBC=True)
sysp = vq.ase.from_atoms(si)                      # → PeriodicSystem
basis = vq.BasisSet(sysp, "pob-tzvp")
result = vq.run_krhf_periodic_gdf(sysp, basis, kmesh=(1, 1, 1))
```

The `bulk(...)` helper covers `"fcc"`, `"bcc"`, `"hcp"`,
`"diamond"`, `"zincblende"`, `"rocksalt"`, `"cesiumchloride"`, …
- the same set documented at
[`crystal_lattices.md`](crystal_lattices.md).

### Spacegroup-driven: `ase.spacegroup.crystal`

For more exotic lattices, give the spacegroup + basis explicitly:

```python
from ase.spacegroup import crystal
import vibeqc as vq

# Rutile TiO2 (P4_2/mnm, #136):
tio2 = crystal(
    symbols=["Ti", "O"],
    basis=[(0.0, 0.0, 0.0), (0.305, 0.305, 0.0)],
    spacegroup=136,
    cellpar=[4.594, 4.594, 2.959, 90, 90, 90],
)
sysp = vq.ase.from_atoms(tio2)
basis = vq.BasisSet(sysp, "pob-tzvp")
```

This pattern produces every example shipped under
`examples/periodic/<system>/`, the rutile, corundum, α-quartz,
wurtzite, perovskite scripts use exactly this idiom.

### Surfaces: `ase.build.surface`

```python
from ase.build import bulk, surface
import vibeqc as vq

cu = bulk("Cu", "fcc", a=3.615)
slab = surface(cu, (1, 1, 1), layers=4, vacuum=10.0)
sysp = vq.ase.from_atoms(slab)

# Periodic-G1a-e pure-DFT forces flow through the molecular
# VibeQC calculator today (HF + hybrid K-gradient queued for
# v0.6.x). Surface relaxation in pure DFT works:
from ase.optimize import BFGS
slab.calc = VibeQC(basis="pob-dzvp-rev2", functional="PBE")
# BFGS(slab).run(fmax=0.05)        # uncomment for full opt
```

See [`crystal_lattices.md`](crystal_lattices.md) for the full
lattice catalogue + visualisations.

## Reading structures via ASE I/O

ASE reads ~30 structure formats out of the box. Read once via
ASE, then forward to vibe-qc:

```python
from ase.io import read
import vibeqc as vq

# CIF (from COD / Materials Project / your own):
atoms = read("MgO.cif")
sysp = vq.ase.from_atoms(atoms)

# POSCAR (VASP 5):
atoms = read("POSCAR", format="vasp")
sysp = vq.ase.from_atoms(atoms)

# XYZ molecular:
atoms = read("water.xyz")
atoms.calc = vq.ase.VibeQC(basis="sto-3g")

# Multi-frame trajectories (.traj, .xyz):
frames = read("optimization.traj", index=":")   # all frames
```

vibe-qc itself reads POSCAR / XYZ natively
(`vq.read_poscar`, `vq.Molecule.from_xyz`); the ASE path is
the right pick when the file format is exotic (CIF, FHI-aims
geometry, Gaussian input, Q-Chem input, etc.) or when you
want ASE's lattice-cell handling.

## Constraints

ASE's constraint system works unchanged with vibe-qc, they
project forces during the BFGS / FIRE / NEB step:

```python
from ase.build import molecule
from ase.constraints import FixAtoms, FixBondLength, FixedPlane
from ase.optimize import BFGS
from vibeqc.ase import VibeQC

water = molecule("H2O")
water.calc = VibeQC(basis="6-31g*", functional="PBE")

# Freeze the oxygen atom; H atoms relax around it:
water.set_constraint(FixAtoms(indices=[0]))    # atom 0 = O
BFGS(water).run(fmax=0.01)

# Or fix a bond length:
water.set_constraint(FixBondLength(1, 2))      # H1-H2 distance

# Or constrain motion to a plane (e.g. a slab atom in z):
water.set_constraint(FixedPlane(indices=[0], direction=[0, 0, 1]))
```

Same pattern works for periodic geometry optimisation
(`atoms.pbc = True`, pure-DFT only at v0.8.0 per the
[periodic section](#periodic-calculations-current-state) below).

(ase-vqfetch)=
## ASE + `vqfetch`

`vqfetch` (the [external-structures](external_structures.md)
fetcher) emits both a vibe-qc input script and a regression
`PeriodicSpec`. For ASE workflows, the SPEC's lattice + atom
list converts trivially to an `ase.Atoms`:

```python
# Pull a structure from Materials Project:
# $ vqfetch mp --id mp-1265   ->   examples/regression/systems/periodic/mp_mp-1265.py

from examples.regression.systems.periodic.mp_mp_1265 import mp_mp_1265
import vibeqc as vq

# Build a PeriodicSystem from the SPEC; then bridge to ASE:
sysp = vq.PeriodicSystem.from_spec(mp_mp_1265)    # vibe-qc native
atoms = vq.ase.to_atoms(sysp)                      # -> ase.Atoms with PBC
print(atoms.get_chemical_formula())                # "Mg4O4"
print(atoms.cell)                                  # cell + pbc preserved

# Or import the spec's provenance for citation tracking:
print(mp_mp_1265.provenance)
# Provenance(source_db='materials_project', source_id='mp-1265', ...)
```

The full provenance bundle (source DB, ID, DOI, license) is
preserved through both round-trip directions, citation
information stays attached to whatever you compute.

(ase-vq-queue)=
## ASE + `vq` queue

ASE-driven workflows (BFGS opt, NEB, MD, basin hopping) are
batch-shaped and run well on a remote compute box via the
[`vq` queue](queue.md). Write the workflow as a normal
Python script + submit:

```python
# my_opt.py, standalone script
from ase.build import molecule
from ase.optimize import BFGS
from vibeqc.ase import VibeQC

atoms = molecule("naphthalene")
atoms.calc = VibeQC(basis="def2-tzvp", functional="b3lyp",
                    dispersion="d3bj")
BFGS(atoms, trajectory="naphthalene.traj").run(fmax=0.01)
```

```sh
# Submit via vq:
vq submit my_opt.py --cpus 8 --mem-mb 16000 --wall-time-seconds 7200
# Tail the SCF trace + BFGS step output live:
vq tail <jobid> -f
# Or just block until it finishes:
vq wait <jobid>
# Pull the .traj file back when done:
vq fetch <jobid> ./outputs/
ase gui ./outputs/naphthalene.traj
```

This is the recommended pattern for long ASE workflows,
laptop runs Jupyter for development; remote runs vq for the
real batch. See [`queue.md`](queue.md) for the full vq story.

## Dispersion (D3-BJ)

Pass `dispersion=` to fold Grimme D3-BJ dispersion into the
energy + forces:

```python
# Use the SCF functional's D3-BJ parameters automatically.
atoms.calc = VibeQC(basis="6-31g*", functional="PBE",
                    dispersion="d3bj")

# Or specify a different functional's parameters explicitly:
atoms.calc = VibeQC(basis="6-31g*", functional="PBE",
                    dispersion="b3lyp")  # uses B3LYP's D3-BJ params

# Or pass a vibeqc.D3BJParams object directly for fine control.
```

The dispersion energy is logged at INFO level on the
`vibeqc.ase` logger, and the per-component breakdown is exposed
on the calculator's results dict:

```python
atoms.get_potential_energy()
print(atoms.calc.results["e_scf"])         # SCF energy in eV
print(atoms.calc.results["e_dispersion"])  # D3(BJ) correction in eV
```

See [dispersion](../tutorial/dispersion.md) for
the full theory + variants.

```{warning}
`atoms.get_potential_energy()` with `dispersion=` set returns the
**total** (SCF + D3-BJ) energy, matching the forces. The high-level
`vq.run_job(...)` result keeps the opposite convention: `.energy` is
the SCF energy and the total lives on `.energy_total` (with the
correction on `.e_dispersion`). When collecting energies from mixed
ASE / `run_job` workflows, always compare totals with totals.
```

## Choosing an ASE optimizer

Which optimizer classes exist depends on the installed ASE version:
`FIRE2` and `RFO` are recent additions that ASE 3.22.x does not have,
and on ASE < 3.23 the line-search optimizers (`BFGSLineSearch`,
`LBFGSLineSearch`) fail to import against SciPy >= 1.14 (they import
the removed `scipy.integrate.cumtrapz`). Both failure modes surface
*after* a batch job has been queued, which is the expensive place to
find out.

vibe-qc ships helpers that probe the active installation up front:

```python
from vibeqc.ase import (
    available_ase_optimizers,      # ["bfgs", "bfgs-linesearch", ...]
    resolve_ase_optimizer,         # name -> optimizer class, or raise
    format_ase_optimizer_report,   # human-readable availability table
    ensure_ase_scipy_compat,       # cumtrapz alias shim for old ASE
)

print(format_ase_optimizer_report())

# Fails early (with the full report) if the deployed ASE lacks it:
Optimizer = resolve_ase_optimizer("ase-bfgs-linesearch")
Optimizer(atoms).run(fmax=0.05)
```

`resolve_ase_optimizer` accepts the canonical keys, the ASE class
names, and the `ase-` prefixed spellings batch matrices use. It calls
`ensure_ase_scipy_compat()` for the line-search entries automatically;
importing `vibeqc.ase` also applies the shim, so a plain
`from ase.optimize import BFGSLineSearch` placed *after* the
`vibeqc.ase` import works on old-ASE + new-SciPy environments too.

Practical guidance for expensive (DFT-quality) calculators, from the
glycine PBE0/def2-TZVP optimizer matrix:

* `BFGS`, `LBFGS`, `BFGSLineSearch`, and `GoodOldQuasiNewton` are the
  reliable workhorses.
* `GPMin`'s surrogate model can propose very large steps and drive the
  geometry into regions where the SCF no longer converges; prefer a
  quasi-Newton optimizer, or bound the step.
* `MDMin`'s default `dt=0.2` is tuned for cheap force fields and
  overshoots on ab initio surfaces; use `MDMin(atoms, dt=0.05)` or
  smaller.

When the SCF fails to converge mid-optimization the calculator raises
`vibeqc.ase.VibeQCSCFConvergenceError`, whose message names the
closest atom pair and suggests step-size remedies, so batch harnesses
can classify the failure without scraping tracebacks.

The calculator additionally rejects collapsed geometries *before*
running any SCF: if the closest atom pair falls below
`min_pair_distance` (default 0.25 Angstrom, far inside any bound
internuclear distance) it raises `vibeqc.ase.VibeQCGeometryError`
immediately instead of burning the full SCF iteration budget on a
meaningless Hamiltonian. Disable with
`VibeQC(min_pair_distance=None)` for deliberate ultra-compressed
scans.

## Workflow examples

The `examples/ase_workflows/` directory ships runnable scripts for
the common ASE patterns, all using `VibeQC` as the backing
calculator:

| Workflow | Script | Demonstrates |
|---|---|---|
| Geometry optimization | [`optimize-via-ase-bfgs.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/ase_workflows/optimize-via-ase-bfgs.py) | ASE's `BFGS` on H₂O / 6-31G* with energy + force convergence plot |
| Vibrational analysis | [`vibrations-via-ase-vibrations.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/ase_workflows/vibrations-via-ase-vibrations.py) | `ase.vibrations.Vibrations` (FD on analytic gradients); asserts the three real frequencies fall in the HF/6-31G* band |
| NEB transition state | [`nh3-neb-via-ase.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/ase_workflows/nh3-neb-via-ase.py) | Climbing-image NEB on NH₃ umbrella inversion |
| NVE molecular dynamics | [`md-water-nve.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/ase_workflows/md-water-nve.py) | Velocity-Verlet conservation diagnostic, drift < 10 meV / 25 fs is the force-energy consistency test |
| Surface adsorption setup | [`surface-h2-pt111-singlepoint.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/ase_workflows/surface-h2-pt111-singlepoint.py) | Pt(111) slab + H₂ atop geometry fixture. Periodic forces are available through the native periodic drivers; the ASE-calculator-side `Atoms`-to-`PeriodicSystem` adapter is still missing, so the script writes the committed XYZ setup and skips SCF for now. |

```{tip}
**For ASE's standard tutorials**, see the
[ASE workflow-tutorials parity matrix](../roadmap.md). Most
molecular tutorials work with vibe-qc as a one-line calculator
swap. Periodic workflows currently require driving the
periodic-system path directly from Python until the ASE-calculator
adapter for periodic systems
lands.
```

## Cross-validation against external codes

vibe-qc's policy ([Archived CLAUDE.md § 10](https://vibe-qc.com/docs/)):
**other QC programs (PySCF, ORCA, Psi4, CRYSTAL, …) are external
programs run out-of-process**, not imported as backends. The
**subprocess-runner pattern** at
[`examples/regression/runner_*.py`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/regression)
spawns each external program, parses its output, and produces
a comparison artefact alongside the vibe-qc result.

```python
# examples/regression/run_suite.py walks a curated test set:
#   - vibe-qc runs in-process (it's *our* code).
#   - PySCF / ORCA / CRYSTAL run as subprocesses (out-of-process).
#   - All parity numbers are captured + tabled in summary.md.

# Single-system smoke check, command-line:
python -m examples.regression.run_suite \
    --include water_6_31gs \
    --references pyscf orca \
    --output-md ./parity-report.md
```

For an ASE-driven idiom (the same `Atoms` against vibe-qc +
external runner subprocess output), use the regression suite's
runner modules directly:

```python
from ase.build import molecule
from vibeqc.ase import VibeQC
from examples.regression.core import runner_pyscf, runner_orca

atoms = molecule("H2O")
atoms.calc = VibeQC(basis="6-31g*")
e_vibeqc = atoms.get_potential_energy()

# PySCF as an external program: spawned, parsed, never imported:
pyscf_result = runner_pyscf.run_rhf(atoms, basis="6-31g*")
e_pyscf = pyscf_result.energy_ev      # parsed from PySCF's stdout

# ORCA likewise:
orca_result = runner_orca.run_rhf(atoms,
                                   orcasimpleinput="HF 6-31G* EnGrad")
e_orca = orca_result.energy_ev

print(f"vibe-qc: {e_vibeqc:.6f} eV")
print(f"PySCF:   {e_pyscf:.6f} eV   Δ = {1000*(e_vibeqc - e_pyscf):.3f} meV")
print(f"ORCA:    {e_orca:.6f} eV    Δ = {1000*(e_vibeqc - e_orca):.3f} meV")
```

The runner modules handle the program-specific bits:
* **PySCF runner**, writes a minimal Python driver script,
  spawns `python <script>` (with the PySCF venv on PATH),
  parses energy / forces / dipole from JSON output.
* **ORCA runner**, emits the input file (`! <orcasimpleinput>`
  + coordinate block), spawns `orca input.inp`, parses
  `.engrad` for forces and the `.out` file for energy /
  dipole.
* **CRYSTAL runner**, emits the `.d12` input file
  (lattice + basis + SCF block), spawns Pcrystal via the
  `contrib/run-crystal.sh` wrapper, parses
  `out.out` for energy.

Expected output for the H₂O / RHF / 6-31G\* example (verified
end-to-end via the runner-subprocess pattern):

```text
vibe-qc: -2068.294643 eV
PySCF:   -2068.294643 eV   Δ = 0.000 meV
ORCA:    -2068.294643 eV   Δ = 0.000 meV
```

The runners gracefully skip when an external program isn't
available (PySCF not in any venv; `orca` not on `$PATH`).
Missing rows show as `(unavailable)` in the report, the
non-skipped rows still produce parity numbers.

```{note}
**Retirement of the in-process `vibeqc.benchmark.PySCFCalculator` /
`make_psi4_calculator` shims.** Earlier vibe-qc versions exposed
ASE-style calculators that imported `pyscf` / `psi4` inside
vibe-qc's process. The CLAUDE.md sec 10 policy (commits
`545db04` / `7ab9384` / `d451b84` for PySCF; `6910ce5d` for the
final PySCF / Psi4 retirement in v0.8.0) moves all external
programs out-of-process. The shims are gone from
`vibeqc.benchmark` and `import` will fail; migrate to the
subprocess runner pattern above. `make_orca_calculator` survives:
ORCA's ASE wrapper is itself a subprocess driver, so it does not
violate the in-process boundary.
```

See:

* [cross-validation tutorial](../tutorial/cross_validation.md)
  - H₂O / NaCl / water-dimer worked walkthrough, now using
  the subprocess runners.
* [external codes user guide](external_codes.md), installing
  ORCA / Psi4 / CRYSTAL14 alongside vibe-qc, plus licensing
  realities.
* [`examples/regression/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/regression)
  - full regression-suite runner directory.

## ASE option translation

The `VibeQC` calculator accepts both ASE-style kwargs and vibe-qc
options objects. The ASE-style ones map to common defaults; the
vibe-qc options structs let you reach every knob.

```python
# Common cases via kwargs:
VibeQC(basis="def2-tzvp")                                # RHF defaults
VibeQC(basis="def2-tzvp", functional="PBE")              # RKS defaults
VibeQC(basis="def2-tzvp", functional="PBE",
       charge=-1, multiplicity=2)                        # OH⁻ doublet
VibeQC(basis="def2-tzvp", dispersion="d3bj",
       functional="PBE")                                  # PBE-D3(BJ)

# Full SCF-options control:
import vibeqc as vq
opts = vq.RHFOptions()
opts.max_iter = 200
opts.level_shift = 0.4
opts.diis_history = 20
VibeQC(basis="def2-tzvp", rhf_options=opts)
```

Available options structs: `rhf_options=`, `uhf_options=`,
`rks_options=`, `uks_options=`. Pass whichever matches the method
the calculator will dispatch to.

For RKS and UKS, an implicit options object uses
`grid_level="orca-defgrid3"`, matching `run_job` and the direct KS
wrappers. Set `grid_level="fine"`, `"coarse"`, or `"legacy"` to select
another preset. If you pass `rks_options=` or `uks_options=` with a
customised grid, that grid wins and `grid_level` leaves it alone; an
options object whose grid you never touched receives the preset, so passing
`RKSOptions()` to set `max_iter` does not silently switch the calculator to
the legacy grid (#663).

## Logging

The ASE calculator logs progress on the `vibeqc.ase` logger.
Configure as you would any Python logger:

```python
import logging
logging.basicConfig(level=logging.INFO)
# Now atoms.get_potential_energy() prints:
#   INFO:vibeqc.ase:RHF / sto-3g  n_atoms=3  n_electrons=10  charge=0  mult=1
#   INFO:vibeqc.scf:SCF converged in 10 iterations  E = -75.984...
```

The underlying SCF trace logs on `vibeqc.scf` (per-iteration
energy, gradient norm, DIIS error vector). Set `level=logging.DEBUG`
for additional detail (per-iteration timing, integral cache hits).

## Periodic calculations (current state)

`vibeqc.ase.from_atoms(atoms)` converts an ASE `Atoms` with
`pbc=True` + non-trivial `cell` directly to a vibe-qc
`PeriodicSystem`. From there, the native GDF drivers handle supported
Gamma and multi-k RHF/RKS/UHF/UKS calculations. Closed-shell RHF/RKS
retain the sub-microhartree external-parity anchors, while the public
multi-k routes enforce their documented full-mesh and gradient envelopes.
See [`multi_k_scf.md`](multi_k_scf.md) for the current sampling contract.

```python
from ase.build import bulk
import vibeqc as vq

# ASE-built MgO rocksalt, primitive cell.
atoms = bulk("MgO", "rocksalt", a=4.211)
sysp = vq.ase.from_atoms(atoms)
basis = vq.BasisSet(sysp, "sto-3g")

# Gamma native GDF example; supported multi-k tuples use the same API.
result = vq.run_krhf_periodic_gdf(sysp, basis, kmesh=(1, 1, 1))
print(f"E = {result.energy_per_cell_ha:.6f} Ha/cell")
```

For **forces** on a periodic system today: Phase **G1** analytic
gradients ship for RHF, RKS (LDA / GGA / hybrid), plus the UHF /
UKS open-shell counterparts; BIPOLE production forces use the FD
energy-gradient route documented in the BIPOLE guide. The periodic
ASE-calculator adapter (`VibeQCPeriodic`, below) now drives energy +
forces from an `ase.Atoms` directly. BIPOLE variable-cell work is not
currently supported.
The recommended workflow today is:

* **Geometry generation + cell setup**: ASE (`bulk`, `surface`,
  `crystal`, CIF reader).
* **Energy + supported Gamma/multi-k single-point**: native GDF driver.
* **Forces**: the route-specific native GDF analytic-gradient envelope.
* **Geometry optimisation** (fixed cell): `VibeQCPeriodic` + any
  `ase.optimize` driver (BFGS / FIRE), at Γ or multi-k. Variable-cell
  BIPOLE optimization fails closed.

For full periodic-SCF detail see:

* [crystal lattices](crystal_lattices.md), Bravais
  classification + per-lattice visualisations + worked-example
  index.
* [k-point meshes](k_points.md), Monkhorst-Pack, IBZ.
* [multi-k SCF](multi_k_scf.md), KRHF / KRKS current state,
  parity target, scope caveats.
* [Ewald summation](ewald.md), long-range Coulomb.
* tutorials [4](../tutorial/periodic_hf.md),
  [5](../tutorial/periodic_dft.md),
  [12](../tutorial/band_structure.md),
  [33](../tutorial/mgo_from_materials_project.md),
  [34](../tutorial/lih_multi_k.md),
  [35](../tutorial/open_shell_mg_cation.md).

### `VibeQCPeriodic`, periodic ASE calculator (available today)

`vibeqc.ase_periodic.VibeQCPeriodic` mirrors the molecular calculator
for fully-periodic `Atoms` (`pbc=True` on all axes), so ASE optimisers,
`ase.mep.NEB`, and MD drive the periodic SCF directly. Energy + analytic
forces come from a single SCF per geometry, and ASE caches the result
until `atoms.positions` change:

```python
from ase.build import bulk
from ase.optimize import BFGS
from vibeqc.ase_periodic import VibeQCPeriodic

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = VibeQCPeriodic(basis="sto-3g", functional="pbe", kpts=(2, 2, 2))
e = atoms.get_potential_energy()   # eV
f = atoms.get_forces()             # eV/Å
BFGS(atoms).run(fmax=0.05)         # ASE-driven periodic relaxation (fixed cell)
```

On the default `backend="ewald"` it supports closed-shell HF and
RKS / UKS (Γ and multi-k); open-shell HF (UHF) periodic gradients are
not available on that backend and raise. Passing `backend="gdf"`
(2026-07-29, G-PBC-002 milestones 5-6) routes supported Gamma and multi-k
jobs through the GDF drivers with the analytic gradient ladder: RHF, RKS,
**UHF**, and UKS, pure and global-hybrid functionals. The fit is the
production PySCF-parity `gdf_method="rsgdf"` by default; pass
`gdf_method="compcell"` for the AFT-corrected compensated-charge
fallback. Tailed dense-core rsgdf fits are supported too since the
2026-07-29 e_nuc Ewald-gauge fix. Multi-k GDF analytic gradients
landed 2026-07-30 (G-PBC-002 Item 4): `run_krhf_periodic_gdf` /
`run_kuhf_periodic_gdf` and the KRKS/KUKS wrappers accept
`compute_gradient=True` (rsgdf, Γ-centered full meshes, no
smearing/IBZ; Schwarz-screened fits are differentiated at the SCF's
fixed pair mask since 2026-07-30; FD-gated at 1e-8-class on H2
(2,1,1)),
and the calculator routes `kpts != (1,1,1)` through them (rsgdf
only). Convention note: on `backend="gdf"` the tuple `kpts` is a
**Γ-centered** mesh (the multi-k GDF driver convention), while the
`ewald` backend samples a shifted Monkhorst-Pack mesh for even
`kpts` -- the two backends deliberately disagree on even meshes.
The high-level runner is objective-consistent too:
`run_periodic_job(..., jk_method="gdf", optimize=True)` relaxes on the
same GDF analytic-gradient surface its SCF reported (Γ and multi-k),
never on BIPOLE forces.
Stress is not wired onto the calculator. `run_periodic_job` and the direct
BIPOLE/GDF optimizers fail closed for variable-cell work.

## Versions

`VibeQC` requires `ase>=3.22`, and a normal vibe-qc install includes
it because `run_job(optimize=True)` uses ASE for semiempirical and MLIP
optimization routes. The `[ase]` extra remains as a compatibility alias
for existing setup scripts:

```sh
pip install .                  # from the repo, includes ASE
pip install '.[ase]'           # compatibility alias
```

The cross-validation framework in :mod:`vibeqc.benchmark` is part
of the base install. PySCF is no longer driven in-process, the
former ``PySCFCalculator`` ASE shim was retired (CLAUDE.md § 10
forbids in-process imports of external QC programs under
``python/vibeqc/``); PySCF parity now runs out-of-process via
the subprocess runner at
[`examples/regression/core/runner_pyscf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/regression/core/runner_pyscf.py).

## See also

* [external codes user guide](external_codes.md), installing
  ORCA / Psi4 / Gaussian / NWChem alongside vibe-qc.
* [cross-validation tutorial](../tutorial/cross_validation.md)
  - end-to-end walkthrough.
* [example scripts and generated outputs](../example_outputs.md),
  runnable `examples/ase_compare/` and `examples/ase_workflows/`
  drivers plus the local artifacts they generate.
* [ASE tutorial parity matrix](../roadmap.md), which ASE
  tutorials currently work with vibe-qc, which need v0.5+
  features.
