---
myst:
  html_meta:
    "description": "Side-by-side translation guide for users coming from PySCF, molecules, basis sets, SCF drivers, options, k-meshes, results unpacking, dispersion, and ECPs."
    "og:title": "PySCF → vibe-qc translation guide"
    "og:description": "A 10-minute crosswalk for PySCF users moving to vibe-qc. Same chemistry, slightly different idioms."
---

# Moving from PySCF

vibe-qc validates its molecular SCF stack against
[PySCF](https://pyscf.org/) to machine precision (see
[external_codes](external_codes.md) § PySCF and the regression
suite under `examples/regression/`), so users moving from PySCF
have the easiest migration path of any QC code. The chemistry is
the same; the idioms differ at a handful of well-defined points.

This page is a side-by-side crosswalk. Each section pairs a
PySCF snippet with the equivalent vibe-qc call and notes any
behavioural difference worth knowing.

## Molecules

### PySCF

```python
from pyscf import gto

mol = gto.M(
    atom = """O 0 0 0
              H 0 0.757  0.587
              H 0 -0.757 0.587""",
    basis = "6-31g*",
    charge = 0,
    spin = 0,                 # 2S = 0  (singlet)
)
```

### vibe-qc

```python
import vibeqc as vq

mol = vq.Molecule([
    vq.Atom(8, [ 0.0,  0.000,  0.000]),    # bohr, not Angstrom
    vq.Atom(1, [ 0.0,  1.431, -0.987]),
    vq.Atom(1, [ 0.0, -1.431, -0.987]),
], charge=0, multiplicity=1)                # 2S+1, not 2S

basis = vq.BasisSet(mol, "6-31g*")           # separate from Molecule
```

```{important}
Three differences worth pinning to:

1. **Units.** vibe-qc uses **bohr** internally for everything. The
   convenient `Molecule.from_xyz("h2o.xyz")` parser reads Ångström
   from the file (XYZ convention) and converts on the way in. PySCF
   defaults to Ångström.

2. **Spin convention.** PySCF's `mol.spin` is `2S` (number of unpaired
   electrons). vibe-qc's `Molecule.multiplicity` is `2S+1`
   (multiplicity). PySCF singlet = 0; vibe-qc singlet = 1.
   The Molecule constructor validates `multiplicity` against the
   electron count.

3. **Basis sets are a separate object.** PySCF folds the basis into
   the `Mole`. vibe-qc keeps `Molecule` and `BasisSet` as two
   distinct objects so the same geometry can be evaluated at
   multiple basis qualities without rebuilding the molecule.
```

For XYZ-on-disk geometries the convenience method matches PySCF's
ergonomics:

```python
mol = vq.Molecule.from_xyz("h2o.xyz")        # Angstrom in file
# Equivalent to: gto.M(atom="h2o.xyz", basis="6-31g*")
```

## Restricted Hartree-Fock

### PySCF

```python
from pyscf import scf

mf = scf.RHF(mol)
mf.conv_tol = 1e-9
mf.max_cycle = 100
mf.kernel()
print(mf.e_tot)
print(mf.mo_energy)
print(mf.mo_coeff)
```

### vibe-qc

```python
opts = vq.RHFOptions()
opts.conv_tol_energy = 1e-9
opts.max_iter        = 100

result = vq.run_rhf(mol, basis, opts)
print(result.energy)
print(result.mo_energies)
print(result.mo_coeffs)
print(result.converged)        # bool - never silent like mf.converged
```

| PySCF | vibe-qc |
|---|---|
| `mf.e_tot` | `result.energy` |
| `mf.mo_energy` | `result.mo_energies` |
| `mf.mo_coeff` | `result.mo_coeffs` |
| `mf.mo_occ` | `result.occupations` |
| `mf.make_rdm1()` | `result.density` |
| `mf.get_fock()` | `result.fock` |
| `mf.get_ovlp()` | `result.overlap` |
| `mf.scf_summary` | the formatted block in `output-*.out` |

The `result` object is a frozen dataclass, `mf` is a stateful
object that re-runs SCF on `.kernel()` calls.

## Unrestricted / open-shell

### PySCF

```python
mol = gto.M(atom="...", basis="6-31g*", spin=1)   # doublet
mf = scf.UHF(mol).run()
```

### vibe-qc

```python
mol = vq.Molecule([...], multiplicity=2)          # doublet (2S+1=2)
result = vq.run_uhf(mol, basis)

result.mo_energies_alpha      # per-spin blocks (flat attrs, not nested)
result.mo_coeffs_alpha
result.mo_energies_beta
result.mo_coeffs_beta
result.density_alpha
result.density_beta
result.fock_alpha
result.fock_beta
result.s_squared              # <S^2> expectation value
result.s_squared_ideal        # S(S+1) for the requested multiplicity
result.s_squared_deviation    # <S^2> - S(S+1), i.e. spin contamination
```

For UHF and UKS jobs, the same three values are recorded in verbose text
output, the structured log, the ``.system`` manifest, and QVF provenance.

```{note}
The single source of truth for spin in vibe-qc is
`Molecule.multiplicity` - there is **no `spin` field on `UHFOptions`**.
This is a common source of confusion for PySCF users; see
[molecules](molecules.md) § Configuring open-shell systems.
```

## Kohn-Sham DFT

### PySCF

```python
from pyscf import dft

mf = dft.RKS(mol, xc="PBE")
mf.grids.level = 3
mf.kernel()
```

### vibe-qc

```python
opts = vq.RKSOptions()
opts.functional   = "PBE"
opts.grid.level   = 3        # 0=coarse, 5=ultrafine; default 3
result = vq.run_rks(mol, basis, opts)
```

For hybrid functionals just change the name, both codes resolve
through [libxc](https://libxc.gitlab.io/) and accept the same
short aliases:

```python
opts.functional = "B3LYP"     # PySCF: dft.RKS(mol, xc="B3LYP")
opts.functional = "PBE0"
opts.functional = "wb97x-d"   # range-separated hybrid + CHG dispersion (v0.9.0)
```

```{warning}
**B3LYP convention difference.** vibe-qc's `b3lyp` keyword
resolves to libxc id 475 (VWN5 - the ORCA / CRYSTAL definition);
PySCF evaluates the VWN-RPA / Gaussian flavor (libxc id 402) for
**both** `xc="b3lyp"` and `xc="b3lypg"` (verified on PySCF 2.13 /
libxc 7.0.0). The two flavors differ by ~10-15 mHa per heavy
atom. For strict parity, match the flavor on both sides:

* vibe-qc `functional="b3lyp"` ↔ PySCF `xc="b3lyp5"`
  (identical to <1e-9 Ha on H₂/STO-3G), or
* vibe-qc `functional="b3lyp/g"` (or `"b3lypg"`) ↔ PySCF
  `xc="b3lyp"` (also <1e-9 Ha).

See [functionals](functionals.md) for details.
```

## Geometry optimisation

### PySCF (via PyBerny / geomeTRIC)

```python
from pyscf.geomopt.berny_solver import optimize
mol_opt = optimize(mf, maxsteps=200)
```

### vibe-qc (via ASE / BFGS)

```python
from vibeqc import run_job

run_job(
    mol, basis="6-31g*", method="rks", functional="PBE",
    optimize=True,
    fmax=1e-3,
    max_opt_steps=200,
    output="output-h2o-opt",
)
```

`run_job` writes the trajectory to `output-h2o-opt.traj` (an ASE
binary trajectory, `ase gui output-h2o-opt.traj` animates it),
the final geometry to `output-h2o-opt.xyz`, and the SCF log to
`output-h2o-opt.out`. See [output_files](output_files.md) for the
full file family.

For programmatic optimisation outside `run_job`, use the
[ASE Calculator integration](ase_integration.md):

```python
from ase.optimize import BFGS
from ase.build import molecule
from vibeqc.ase import VibeQC

atoms = molecule("H2O")
atoms.calc = VibeQC(method="rks", functional="PBE", basis="6-31g*")
BFGS(atoms).run(fmax=1e-3)
```

## Periodic systems

### PySCF.pbc

```python
from pyscf.pbc import gto, scf

cell = gto.Cell()
cell.atom = "Mg 0 0 0; O 2.105 2.105 2.105"
cell.a = [[4.21, 0, 0], [0, 4.21, 0], [0, 0, 4.21]]    # Angstrom
cell.basis = "pob-tzvp"
cell.build()

kpts = cell.make_kpts([4, 4, 4])
mf = scf.KRHF(cell, kpts=kpts).run()
print(mf.e_tot)
```

### vibe-qc

```python
import numpy as np

sysp = vq.PeriodicSystem(
    dim=3,
    lattice=np.eye(3) * 7.957,     # bohr (4.21 Å)
    unit_cell=[
        vq.Atom(12, [0.0, 0.0, 0.0]),               # Mg
        vq.Atom(8, [3.979, 3.979, 3.979]),          # O - bohr
    ],
)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "pob-tzvp")
kmesh = vq.monkhorst_pack(sysp, [4, 4, 4])

opts = vq.PeriodicRHFOptions()
opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
result = vq.run_rhf_periodic(sysp, basis, kmesh, opts)
print(result.energy)
```

| PySCF.pbc | vibe-qc |
|---|---|
| `gto.Cell` | `vq.PeriodicSystem(dim=3, ...)` |
| `cell.a` (Å, rows) | `lattice` (bohr, columns) |
| `cell.make_kpts([n,n,n])` | `vq.monkhorst_pack(sysp, [n,n,n])` |
| `scf.KRHF(cell, kpts=kpts)` | `vq.run_rhf_periodic(sysp, basis, kmesh, opts)` |
| `scf.KRKS(cell, kpts=kpts, xc="PBE")` | `vq.run_rks_periodic(sysp, basis, kmesh, opts)` with `opts.functional = "PBE"` |
| `mf.with_df = df.GDF(cell)` | the default for Γ-only via `run_rhf_periodic_gamma_gdf` |

```{important}
**Lattice vectors are columns, in bohr.** PySCF.pbc stores them as
rows in Ångström. The conversion is
`lattice_bohr_cols = lattice_angstrom_rows.T * 1.8897259886`.

For 1D wires, set `dim=1` and put generous vacuum (30+ bohr) on the
non-periodic axes. For 2D slabs, do **not** add vacuum: use
`vq.slab_2d(a1, a2, atoms)`, which takes only the two in-plane vectors
and synthesizes the (non-physical) third column itself. PySCF.pbc uses
the `dimension` keyword on `Cell`, but its `dimension=2` still wants a
vacuum gap; vibe-qc's 2D Ewald gauge does not.
```

## DFT-D3 / D4 dispersion

### PySCF

```python
# PySCF wires dispersion through dft.RKS via the xc string:
mf = dft.RKS(mol, xc="PBE-D3BJ").run()
# or via the dftd3/dftd4 PySCF plugin
```

### vibe-qc

```python
# Via run_job (recommended):
run_job(mol, basis="def2-tzvp", method="rks",
        functional="PBE", dispersion="d3bj",
        output="output-pbe-d3bj")

# Via run_rks directly:
opts = vq.RKSOptions()
opts.functional = "PBE"
opts.dispersion = "d3bj"        # or "d3", "d4"
result = vq.run_rks(mol, basis, opts)
print(result.energy)              # = E_DFT + E_disp
print(result.dispersion_energy)   # = E_disp alone
```

D3(BJ) is wired through the molecular runner via `compute_d3bj`.
D4 (Caldeweyher-Bannwarth-Grimme, 2019) ships with two backends:
the optional `dftd4` package (production default, bit-exact to
upstream) and an in-tree MPL-2.0 native implementation
(`compute_d4(mol, func, backend="native")`) that is parity-validated
for **H-Ne** (correlated CPKS/PBE38 reference C6 within a few percent
of dftd4). Use the default `dftd4` backend for elements outside that
set / full periodic-table coverage.

## Density fitting (RIJ / RIJK / RIJCOSX)

### PySCF

```python
mf = scf.RHF(mol).density_fit(auxbasis="def2-svp-jkfit")
mf.kernel()
```

### vibe-qc

```python
opts = vq.RHFOptions()
opts.density_fit = True
opts.aux_basis = "def2-svp-jk"     # or vq.default_aux_basis_for("def2-svp", kind="jk")
opts.cosx = False                  # set True for RIJCOSX (large hybrid DFT)
result = vq.run_rhf(mol, basis, opts)
```

The `JKBuilder` polymorphic Fock build picks one of three concrete
kernels, direct four-index, DF (RIJK), or DF + COSX, based on
the flags. See [density_fitting](density_fitting.md) for the
when-to-use-which table.

## MP2 / RI-MP2

### PySCF

```python
from pyscf import mp
mp2 = mp.MP2(mf, frozen=0).run()
print(mp2.e_corr, mp2.e_tot)
```

### vibe-qc

```python
mp2_options = vq.MP2Options()
mp2_options.n_frozen_core = 0
mp2_result = vq.run_mp2(mol, basis, result, mp2_options)
print(mp2_result.e_corr)
print(mp2_result.e_total)      # = e_HF + e_corr

# Same-spin / opposite-spin decomposition:
print(mp2_result.e_ss, mp2_result.e_os)

# RI-MP2 (much cheaper at large basis):
mp2_options.density_fit = True
mp2_options.aux_basis = "def2-tzvp-rifit"
mp2_result = vq.run_mp2(mol, basis, result, mp2_options)
```

The explicit zeros make this an all-electron cross-code comparison. Bare
PySCF MP2 is all-electron, whereas bare vibe-qc MP2 follows the published
chemical-core convention; neither implicit default should be used as an
equivalence claim. UMP2 uses the analogous `vq.UMP2Options()` and
`vq.run_ump2(mol, basis, uhf_result, options)` calls.

## Effective Core Potentials

### PySCF

```python
mol = gto.M(
    atom = "Pt 0 0 0",
    basis = "lanl2dz",
    ecp = "lanl2dz",        # ECP carried by the basis
)
```

### vibe-qc

```python
mol = vq.Molecule([vq.Atom(78, [0.0, 0.0, 0.0])])
basis = vq.BasisSet(mol, "lanl2dz")

opts = vq.UHFOptions()
opts.ecp_centers, opts.ecp_library = vq.auto_ecp_centers(mol, "lanl2dz")

result = vq.run_uhf(mol, basis, opts)
```

See [ecp](ecp.md) for the full table mapping basis-set families to
ECP libraries.

## Initial guess

### PySCF

```python
mf.init_guess = "minao"     # or "atom", "1e", "huckel"
mf.kernel()
```

### vibe-qc

```python
opts = vq.RHFOptions()
opts.initial_guess = vq.InitialGuess.SAP       # SAP / SAD / HCORE / AUTO / ...
result = vq.run_rhf(mol, basis, opts)
```

`InitialGuess.AUTO` (default in v0.8.0) inspects the system and
picks SAP (light closed-shell) or SAD (transition metal /
periodic). See [initial_guess](initial_guess.md) for the full
table.

## SCF convergence acceleration

### PySCF

```python
mf.DIIS = scf.EDIIS            # diagonal switch
mf.diis_space = 8
mf.level_shift = 0.2
mf.damp = 0.5
```

### vibe-qc

```python
opts.scf_accelerator = vq.SCFAccelerator.EDIIS_DIIS   # default v0.8.0+
opts.diis_subspace_size = 8
opts.level_shift = 0.2
opts.damping = 0.5

# Second-order finalizer:
opts.newton_threshold = 1.0    # Phase D2c Newton
```

vibe-qc defaults to the EDIIS+DIIS hybrid (Garza-Scuseria 2012);
PySCF still defaults to plain DIIS as of pyscf 2.x. See the
[stiff-convergence tutorial](../tutorial/ediis_diis_hybrid.md)
for when each matters.

## Properties

### PySCF

```python
charges = mf.mulliken_pop()[1]
dip = mf.dip_moment()
```

### vibe-qc

```python
mul = vq.mulliken_charges(result, basis, mol)
low = vq.loewdin_charges(result, basis, mol)
bonds = vq.mayer_bond_orders(result, basis, mol)
dip = vq.dipole_moment(result, basis, mol)
print(dip.total_debye)
```

`run_job` writes a formatted properties block to `output-*.out`
automatically, see [properties](properties.md).

## What vibe-qc does that PySCF doesn't (yet)

Things you don't get out of the box in PySCF that vibe-qc gives
you:

* **Auto-citations.** Every `run_job` writes a `.bibtex` /
  `.references` pair listing every paper to cite for the
  functional / basis / dispersion combination used. See
  [citations](citations.md).
* **Pre-flight memory estimator.** Aborts before the SCF starts
  if the dense-ERI / DFT-grid / MP2 working set exceeds available
  RAM. See [memory](memory.md).
* **`.system` manifest.** TOML manifest with hardware, library
  versions, plan + outputs status. Lets the
  [`vq` queue](queue.md) detect crashed jobs and fetch only the
  declared artefacts. See [output_files](output_files.md).
* **pob-* basis sets.** The Peintinger-Vilela Oliveira-Bredow
  solid-state basis family is bundled. See
  [basis_sets](basis_sets.md) and
  [Why solid-state calculations use `pob-TZVP`](../tutorial/pob_tzvp.md).
* **CRYSTAL-format basis parser.** Drop a CRYSTAL-style per-element
  file under `basis_library/custom/` and `setup_basis_library.sh`
  picks it up.

## What PySCF does that vibe-qc doesn't (yet)

Most of the gaps that existed in earlier vibe-qc releases have now
closed.  The features below were formerly on this list and shipped
in the releases noted:

* **ωB97X-V / ωB97M-V**: shipped in v0.14.0 with the VV10 nonlocal
  correlation kernel (Vydrov & Van Voorhis 2010).  `xc="wb97x-v"`
  and `xc="wb97m-v"` resolve as complete functionals; the periodic-K
  RSH wiring for HSE06 also landed in v0.15.0.
* **CCSD / CCSD(T)**: shipped in v0.14.0 with a validated C++
  closed-shell engine (Stanton-Gauss-Watts-Bartlett 1991 equations,
  Raghavachari (T)).  DLPNO-CCSD(T) followed in v0.15.0.
* **TD-DFT / excited states**: shipped in v0.12.0 (Casida + TDA for
  RHF/RKS/UHF/UKS; open-shell Casida in v0.15.0).  Natural transition
  orbitals (NTOs) and UV/Vis stick spectra are included.
* **MCSCF / multi-reference**: shipped in v0.12.0 (CASCI +
  state-averaged CASSCF) and hardened through v0.15.0 (CASSCF analytic
  gradient, CASPT2/NEVPT2, MS/XMS-CASPT2, selected-CI, DMRG).
* **Multi-k UHF / UKS**: shipped in v0.12.0 (multi-k unrestricted
  Ewald and GDF drivers) and hardened through v0.15.0 (analytic
  gradients, BIPOLE open-shell, smearing, per-spin Gilat occupations).
* **CCSD-DLPNO / domain-based local correlation**: shipped in
  v0.14.0 (DLPNO-CCSD pilot) and hardened to production in v0.15.0
  (closed-shell local solver, open-shell DLPNO-UCCSD, (T) triples).

Genuine remaining gaps where PySCF still offers more:

* **ωB97M(2) double hybrid.**  Gated: the building-block kernels
  landed in v0.14.0 but the self-consistent SCF driver is follow-up;
  the functional raises a clear message pointing to ωB97M-V.
* **Periodic TD-DFT beyond Γ-point.**  Only Γ-point TDDFT is wired;
  multi-k periodic linear response is future work.
* **EOM-CCSD.**  Not yet implemented.
* **Analytic Hessians for periodic systems.**  Molecular analytic
  Hessians are shipped; periodic frequencies use finite differences.

For any of these, drop back to PySCF or another QC code; vibe-qc
plays well with everything via the
[external_codes](external_codes.md) framework, so you can do an
HF reference in vibe-qc and the post-SCF correlation in PySCF /
ORCA / Q-Chem on the same `Atoms`.

## Validating vibe-qc against PySCF

Drop-in cross-validation lives at
[`examples/regression/core/runner_pyscf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/regression/core/runner_pyscf.py).
Per [Archived CLAUDE.md § 10](https://vibe-qc.com/docs/)
vibe-qc never imports PySCF at runtime, the runner subprocesses
PySCF, parses its output, and produces a side-by-side comparison
artefact. The molecular HF / DFT / MP2 paths in vibe-qc match
PySCF to machine precision on the runnable examples.

## See also

- [ase_integration](ase_integration.md), the ASE Calculator
  interface (a clean way to use vibe-qc through PySCF-adjacent
  ASE workflows).
- [external_codes](external_codes.md), the cross-code validation
  framework. Run the same Atoms through vibe-qc + PySCF + ORCA
  and compare.
- [Cross-validating against ORCA, Psi4, and PySCF](../tutorial/cross_validation.md)
  - end-to-end worked example of the same.
- [citations](citations.md), vibe-qc's auto-bibliography surface.
- [output_files](output_files.md), what every `run_job` writes.
