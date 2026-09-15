---
myst:
  html_meta:
    "description": "Translation guide for users coming from Gaussian (G09 / G16) to vibe-qc. Maps the # route line, Link 0 % directives, SCF= keywords, GEN basis blocks, Pseudo=Read, IOp(...) flags, and the B3LYP convention difference."
    "og:title": "Gaussian → vibe-qc translation guide"
    "og:description": "A keyword-block crosswalk for Gaussian users moving to vibe-qc."
---

# Moving from Gaussian

[Gaussian](https://gaussian.com/) (G09 / G16) is the historical
QC workhorse for many groups, and a sizable share of vibe-qc's
early adopters are migrating from there. The chemistry maps
directly; the **input idiom** is different, Gaussian takes a
fixed-grammar deck with a Link 0 header + route line + title +
molecule + (optional) per-element basis / ECP blocks, while
vibe-qc takes a Python script. This page is a directive-by-
directive crosswalk.

For other migrations: [PySCF](from_pyscf.md), [CRYSTAL](from_crystal.md),
[ORCA](from_orca.md).

## Input file shape

### Gaussian `.gjf`

```text
%chk=h2o.chk
%nprocshared=4
%mem=4000MB
#p RB3LYP/6-31G(d) Opt=Tight SCF=Tight Int=UltraFine EmpiricalDispersion=GD3BJ

Water — RKS B3LYP/6-31G* opt with D3(BJ)

0 1
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471160
H  0.000000 -0.755453 -0.471160

```

### vibe-qc Python

```python
import vibeqc as vq

mol = vq.Molecule.from_xyz("h2o.xyz")           # Å in file, bohr internal

vq.run_job(
    mol,
    method="rks",
    functional="B3LYP/G",                       # ← Gaussian's VWN-RPA flavor (or "b3lypg")
    basis="6-31g*",
    dispersion="d3bj",                          # EmpiricalDispersion=GD3BJ
    optimize=True,
    fmax=1e-4,                                  # Opt=Tight
    num_threads=4,                              # %nprocshared=4
    output="output-h2o-b3lyp",
    options=vq.RKSOptions(
        conv_tol_energy=1e-9,                   # SCF=Tight
        grid=vq.GridOptions(level=4),           # Int=UltraFine
    ),
)
```

```{important}
**B3LYP convention difference.** Gaussian's `B3LYP` keyword uses
the VWN-RPA correlation parametrisation ("VWN(III)" in Gaussian's
manuals; libxc id 402). vibe-qc's bare `b3lyp` is the **VWN5**
flavor (libxc id 475, the ORCA / TURBOMOLE / CRYSTAL definition)
— the two differ by ~10–15 mHa per heavy atom. For Gaussian
parity, pass `functional="b3lyp/g"` (or the equivalent spelling
`"b3lypg"`) — measured on H₂/STO-3G it matches the
Gaussian-flavor reference (PySCF 2.13's `b3lyp`) to <1e-9 Ha.
See [functionals](functionals.md) § B3LYP for the details.
```

The vibe-qc Python file is **executable**, so anything you'd
normally do via Gaussian's `--Link1--` chaining or
`linker-controlled batch` (multiple jobs in one input) lands in
a plain Python loop:

```python
for func in ["B3LYP/G", "PBE0", "wB97X-D"]:
    vq.run_job(
        mol,
        method="rks",
        functional=func,
        basis="6-31g*",
        output=f"output-h2o-{func.lower().replace('/', '-')}",
    )
```

## Keyword crosswalk

### Link 0 directives (`%...`)

| Gaussian `%` | vibe-qc |
|---|---|
| `%nprocshared=n` | `num_threads=n` on `run_job` (OpenMP). vibe-qc is shared-memory only, no `%nproclinda` / multi-node equivalent. |
| `%mem=4000MB` | Not a separate kwarg, vibe-qc estimates memory pre-flight and aborts if exceeded; pass `memory_override=True` to bypass. See [memory](memory.md). |
| `%chk=h2o.chk` | The `.molden` sibling (orbital coefficients) + `.system` manifest serve the same provenance role. The `.chk` analogue for re-using a converged density (`Guess=Read`) is roadmap. |
| `%rwf=...`, `%int=...` | No analogue, vibe-qc keeps all intermediate state in RAM by default. |
| `%save` / `%nosave` | Always cleans up, opt in to extra logs via `perf_log=True` / `structured_log=True`. |
| `%KJob` (split a long job) | Use the [`vq` queue](queue.md) for batch dispatch. |

### Route-line (`#`) keywords

| Gaussian `#` keyword | vibe-qc |
|---|---|
| `RHF`, `UHF`, `ROHF` | `method="rhf"` / `"uhf"` / `"rohf"`, see [ROHF](rohf.md) |
| `RB3LYP`, `UB3LYP` | `method="rks"` / `"uks"`, `functional="B3LYP/G"` (Gaussian's VWN-RPA flavor; also spelled `"b3lypg"`), bare `"B3LYP"` is the VWN5 / ORCA flavor |
| `RPBEPBE` | `functional="PBE"` (vibe-qc resolves the X+C pair through libxc) |
| `RPBE1PBE` | `functional="PBE0"` |
| `MP2`, `UMP2` | `vq.run_mp2(...)`, `vq.run_ump2(...)` |
| `B2PLYP` | `vq.run_b2plyp(...)`, see [Double hybrid: B2PLYP step by step](../tutorial/double_hybrid_b2plyp.md) |
| `6-31G(d)`, `6-31G(d,p)`, `6-31+G(d,p)` | `basis="6-31g*"`, `"6-31g**"`, `"6-31+g**"` |
| `cc-pVDZ`, `cc-pVTZ` | `basis="cc-pvdz"`, `"cc-pvtz"` (case-insensitive) |
| `Def2TZVP` / `Def2TZVPP` | `basis="def2-tzvp"` / `"def2-tzvpp"` |
| `GEN` (general basis from input block) | Drop the per-element basis into `basis_library/custom/`; see [basis_sets](basis_sets.md). |
| `Pseudo=Read` (read ECP from input block) | `opts.ecp_centers = [ECPCenter(Z=..., xyz=...)]` + `opts.ecp_library = "lanl2dz"` / `"ecp46mdf"` / ... See [ecp](ecp.md). |
| `EmpiricalDispersion=GD3` | `dispersion="d3"` |
| `EmpiricalDispersion=GD3BJ` | `dispersion="d3bj"` |
| `EmpiricalDispersion=GD4` | `dispersion="d4"` |
| `Density=Current` (use the SCF density) | Default, `result.density` always carries the converged density. |
| `Pop=Full / Pop=Reg / Pop=NaturalOrbitals` | Mulliken / Löwdin / Mayer are always reported by `run_job`; natural orbitals via [Natural orbitals and the idempotency diagnostic](../tutorial/natural_orbitals.md). |
| `Opt` / `Opt=Tight` | `run_job(..., optimize=True, fmax=1e-3)` (default) or `fmax=1e-4` for Tight. |
| `Opt=TS / Opt=NoEigen` (transition state) | Roadmap (vibe-qc's optimiser is BFGS via ASE; saddle-point opt + NEB are queued). |
| `Freq=Numerical / Freq=Analytic` | `run_job(..., compute_hessian=True)`, analytic Hessian for RHF/RKS/UHF/UKS LDA + GGA, finite-diff otherwise. See [Vibrational frequencies](../tutorial/vibrational_frequencies.md). |
| `Int=FineGrid / UltraFine / SuperFine` | `opts.grid = vq.GridOptions(level=3 / 4 / 5)` for RKS/UKS (defaults to 3 = FineGrid equivalent). |
| `SCF=Tight / VeryTight` | `opts.conv_tol_energy = 1e-9 / 1e-10` |
| `SCF=NoSymm` | (default, vibe-qc doesn't yet exploit symmetry in the SCF) |
| `SCF=QC` (quadratically convergent) | `opts.newton_threshold = 1.0`, the v0.8.0 Newton finalizer. See [scf_convergence](scf_convergence.md). |
| `SCF=DM` (direct minimization) | Roadmap, D2 second-order family covers Newton / TRAH / SOSCF today. |
| `SCF=NoVarAcc` (disable DIIS) | `opts.use_diis = False` and rely on plain `damping=` mixing. The `scf_accelerator` enum has no off-state value; `use_diis` is the explicit on/off flag. |
| `Guess=Mix` (broken-symmetry guess) | Manual SAD + MOM workflow today; one-liner is roadmap. |
| `Guess=Huckel / Guess=Harris` | `opts.initial_guess = vq.InitialGuess.HUECKEL` (parameter-free extended Hückel) or `PATOM` (re-polarised SAD, the ORCA PAtom analogue). Both ship as of v0.9.x; molecular only. |
| `Symmetry=NoInt` / `Symmetry=Loose` | No-ops; vibe-qc doesn't yet use molecular symmetry. |
| `IOp(2/15=N)` | Generally not supported, `IOp` was Gaussian's escape-hatch for internal flags. Most useful `IOp`s have explicit vibe-qc API equivalents; the rest are queued. |

### Molecule specification

Gaussian's molecule block looks like:

```text
0 1                    ← charge, multiplicity
O  x  y  z             ← atomic symbol or number
H  x  y  z
H  x  y  z
                       ← blank line terminator
```

The vibe-qc equivalent uses `Molecule` + `Atom`:

```python
mol = vq.Molecule([
    vq.Atom(8, [0.0, 0.0, 0.117790 * 1.8897259886]),   # bohr
    vq.Atom(1, [0.0,  0.755453 * 1.8897259886, -0.471160 * 1.8897259886]),
    vq.Atom(1, [0.0, -0.755453 * 1.8897259886, -0.471160 * 1.8897259886]),
], charge=0, multiplicity=1)
```

For typical `.xyz` files (which Gaussian also reads), use the
convenience method that converts Å → bohr on the way in:

```python
mol = vq.Molecule.from_xyz("h2o.xyz")
```

**Spin convention.** Gaussian's `multiplicity` is `2S+1`, 
**same as vibe-qc**. (PySCF uses `2S` = number of unpaired
electrons; that's the gotcha for the PySCF migration, not here.)

### GEN basis blocks

Gaussian's `GEN` keyword switches to a per-element basis block at
the bottom of the input:

```text
#p RHF/GEN

Title

0 1
Fe  0.0 0.0 0.0
...

Fe 0
S   1 1.00
   100.0    1.0
****
H  0
S   1 1.00
    1.0    1.0
****

```

vibe-qc's CRYSTAL-format parser at `vibeqc.basis_crystal` handles
the equivalent per-element format. Drop the file under
`basis_library/custom/my_basis.g94` and re-run
`./scripts/setup_basis_library.sh`; then load it by name:

```python
basis = vq.BasisSet(mol, "my_basis")
```

For one-off jobs without the basis-library rebuild step, build a
`BasisSet` from an in-memory dict (roadmap; today the rebuild is
the canonical path).

### Pseudo=Read

Gaussian's ECP-from-input pattern:

```text
#p RHF/GEN Pseudo=Read

...

Pt 0
S-D   4    1.000000
   1234.0  0.001
...
```

vibe-qc keeps the orbital basis and ECP as separate fields. Use
the bundled ECP library that matches the orbital basis's element
row (see [ecp](ecp.md) § Library-selection table):

```python
opts = vq.UHFOptions()
opts.ecp_centers = [vq.ECPCenter(Z=78, xyz=[0.0, 0.0, 0.0])]
opts.ecp_library = "ecp46mdf"      # Stuttgart MDF for Pt
# or "lanl2dz" for the Hay-Wadt convention
```

See [Pt cluster with LANL2DZ: heavy-element ECPs](../tutorial/pt_cluster_lanl2dz.md) for the
full Pt-cluster recipe.

## DFT integration grid

| Gaussian `Int=...` | vibe-qc |
|---|---|
| `Int=FineGrid` (default) | `GridOptions(level=3)`, 75 radial, 302 angular |
| `Int=UltraFine` | `GridOptions(level=4)`, 99 radial, 590 angular |
| `Int=SuperFine` | `GridOptions(level=5)`, 150 radial, 974 angular |
| `Int(Grid=N)` (explicit grid spec) | `GridOptions(n_radial=N_r, n_angular=N_a)` |

vibe-qc's grid quality matches Gaussian's at the same `level=`
within ~0.1 mHa on standard hybrid DFT.

## Frequencies and thermodynamics

| Gaussian | vibe-qc |
|---|---|
| `Freq` | `run_job(..., compute_hessian=True)`; result carries normal modes + frequencies |
| `Freq=Numerical` | `compute_hessian="fd"`, finite-difference fallback |
| `Freq=Analytic` (default for HF/DFT) | `compute_hessian="analytic"` for RHF / RKS / UHF / UKS-LDA. UKS-GGA analytic Hessian is roadmap (Phase 17e). |
| `Temperature=N` in route line | `ThermoOptions(temperature=N)` (Kelvin) on `run_job` |
| `Pressure=p` | `ThermoOptions(pressure=p)` |
| ZPE / enthalpy / Gibbs free energy printed in summary | Always reported when `compute_hessian=True`. See [Thermodynamics at temperature](../tutorial/thermodynamics.md). |

## Output files

| Gaussian | vibe-qc |
|---|---|
| `.log` (text log) | `{stem}.out`, same role |
| `.chk` / `.fchk` (binary / formatted checkpoint) | `.molden` (orbitals) + `.system` (TOML manifest). `.fchk` writer is roadmap. |
| `.wfn` (wavefunction file) | Not yet, `.molden` covers most viewers (Jmol, Avogadro, MolTUI). |
| `.cube` (volumetric data via `cubegen`) | `vq.write_cube_density / write_cube_mo`, see [Orbital and density visualization](../tutorial/orbital_visualization.md). |
| `Test.FChk` for property reads | `result` object has every result quantity as a Python attribute. |

vibe-qc's **native** single-file visualization format has no Gaussian
analogue: `run_job(..., output_qvf=True)` writes a `.qvf` archive
(structure + density + orbitals) for [vibe-view](vibe_view.md), which
supersedes wiring `.molden` and `.cube` files together for a
third-party viewer. See the [QVF tech spec](../design_qvf_format.md)
and [The QVF file format, end to end](../tutorial/qvf_file_format.md).

vibe-qc additionally always writes (v0.8.x+) `.xyz`, `.bibtex`,
`.references`, and (when applicable) `.POSCAR` / `.xsf` /
`.population.{txt,json}`. See [output_files](output_files.md) and
[citations](citations.md).

## What Gaussian does that vibe-qc doesn't (yet)

Use Gaussian for these for now:

* **Coupled cluster, CASSCF, multi-reference perturbation theory.**
  CCSD(T) is the canonical "gold-standard" workhorse, vibe-qc
  tops out at MP2 and won't ship CCSD until v2.x (CCM-coupled).
* **TD-DFT and excited states.** Roadmap.
* **Range-separated hybrids beyond ωB97X / ωB97X-D.** vibe-qc
  v0.9.0 ships ωB97X and ωB97X-D (the latter via
  `vibeqc.run_wb97x_d` + the CHG dispersion kernel); CAM-B3LYP and
  HSE06 / LC-ωPBE / LRC-ωPBE are queued, and the RSH machinery
  for periodic K is queued.
* **Polarizable continuum models other than CPCM/COSMO** (PCM,
  CPCM-X, IEFPCM, IPCM, SMD). vibe-qc v0.9.0 ships CPCM/COSMO
  (see [solvation](solvation.md)); other implicit-solvent
  variants are roadmap.
* **NBO analysis.** Mulliken, Löwdin, Mayer, and natural orbitals
  are in; full NBO 7 integration is queued.
* **NMR shielding tensors / coupling constants.** Roadmap.
* **Composite methods G3 / G4 / W1.** The "3c" family
  (HF-3c through ωB97X-3c) ships in v0.9.0 via `method="hf-3c"`
  etc. with gCP + D3/D4 wired through the molecular runner.
  G3 / G4 / W1 themselves are still roadmap; the building blocks
  (HF, MP2, B3LYP, B2PLYP, CCSD via the in-tree CC foundation)
  are present but the composite-method orchestration for those
  specific recipes is not yet wired.
* **Counterpoise (BSSE) correction** as a built-in route keyword.
  Manual CP by running each fragment in the dimer basis is
  straightforward in Python; the one-liner is roadmap.

## What vibe-qc does that Gaussian doesn't (or does differently)

Things you gain by switching:

* **MPL-2.0 license.** No per-seat or per-institution licensing;
  drop into any project, including closed-source software.
* **Python scripting.** Sweeps over functionals / basis sets /
  conformers are a few-line `for` loop, not a chain of
  `--Link1--` blocks.
* **Periodic SCF.** Gaussian's periodic capabilities are limited;
  vibe-qc has 1D / 2D / 3D periodic HF and KS-DFT today.
* **Auto-citations.** Every job writes a `.bibtex` /
  `.references` pair listing every paper to cite. See
  [citations](citations.md). The Gaussian equivalent is to read
  the manual.
* **pob-* basis sets bundled.** The Peintinger-Vilela Oliveira-
  Bredow solid-state basis family ships in `basis_library/`.
  See [basis_sets](basis_sets.md).
* **`vq` queue with vibe-qc-aware preflight.** Submit jobs to a
  remote box with output-aware fetch. See [Submitting a job to a remote machine with `vq`](../tutorial/vq_queue_remote_job.md).
* **Pre-flight memory estimator.** Aborts before the SCF starts
  if the dense-ERI / DFT-grid / MP2 working set exceeds available
  RAM. See [memory](memory.md).
* **Open source you can hack on.** All the way down through the
  C++ integral kernels and the libint glue.

## Validating vibe-qc against Gaussian

Gaussian is **not** in vibe-qc's bundled benchmark suite (the
[external_codes](external_codes.md) framework lists it as
user-installed; no out-of-process runner ships with vibe-qc by
default because Gaussian licensing precludes redistribution).

Manual parity check:

```sh
# 1. Run in Gaussian:
g16 my-input.gjf > my-input.gjf.log

# 2. Run in vibe-qc:
python my-input.py             # writes my-input.out / .bibtex / etc.

# 3. Compare:
grep "SCF Done:" my-input.gjf.log
grep "Total energy:" my-input.out
```

Expected agreement after **using `b3lyp/g` (or `b3lypg`) for the
Gaussian flavor**: <0.5 mHa per heavy atom for closed-shell HF /
DFT on small molecules. The remaining discrepancy is dominated by
integral-screening conventions (Gaussian's defaults are tighter
than vibe-qc's), pass
`opts.lattice_opts.cutoff_bohr = 30` and
`opts.conv_tol_energy = 1e-10` for sub-µHa parity.

## See also

- [from_pyscf](from_pyscf.md), sister migration guide for the
  PySCF code (the canonical open-source comparison reference).
- [from_orca](from_orca.md), sister migration guide for ORCA
  (the canonical hybrid-DFT + RIJCOSX reference vibe-qc validates
  against).
- [from_crystal](from_crystal.md), sister migration guide for
  CRYSTAL (the canonical solid-state Gaussian-basis reference).
- [user_guide/functionals](functionals.md) § B3LYP, the VWN3
  vs VWN5 convention difference that bit users moving between
  Gaussian and any modern code.
- [user_guide/external_codes](external_codes.md), the
  cross-validation framework. The Gaussian calculator is on the
  supported list once a user-installed binary is on `$PATH`.
