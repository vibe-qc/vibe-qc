---
myst:
  html_meta:
    "description": "Translation guide for users coming from ORCA to vibe-qc. Maps `!` keywords, %scf / %basis / %method blocks, RIJCOSX, TIGHTSCF, KDIIS, and B3LYP/G to the corresponding vibe-qc API."
    "og:title": "ORCA → vibe-qc translation guide"
    "og:description": "A keyword-block crosswalk for ORCA users moving to vibe-qc."
---

# Moving from ORCA

vibe-qc's molecular SCF stack validates against
[ORCA 6.1.1](https://orcaforum.kofo.mpg.de/) for parity at the
0.13 mHa level on glycine / def2-TZVP / RIJCOSX (see
[RIJCOSX SCF and analytic gradients on glycine](../tutorial/rijcosx_glycine.md)). The chemistry
maps cleanly; the **input idiom** is different, ORCA takes a
keyword-driven plain-text deck while vibe-qc takes a Python
script. This page is a side-by-side crosswalk.

For users coming from PySCF instead, see
[from_pyscf](from_pyscf.md). For CRYSTAL users, see
[from_crystal](from_crystal.md).

## Input file shape

### ORCA `.inp`

```text
! RKS PBE0 D3BJ DEF2-TZVP RIJCOSX TIGHTSCF OPT

%scf
  MaxIter 200
  KDIIS true
  CNVDIIS 1.0e-7
end

%pal
  nprocs 4
end

* xyz 0 1
O      0.000000   0.000000   0.117790
H      0.000000   0.755453  -0.471160
H      0.000000  -0.755453  -0.471160
*
```

### vibe-qc Python

```python
import vibeqc as vq

mol = vq.Molecule.from_xyz("h2o.xyz")          # Å in file, bohr internal

vq.run_job(
    mol,
    method="rks",
    functional="PBE0",
    basis="def2-tzvp",
    dispersion="d3bj",                          # D3BJ keyword
    density_fit=True, cosx=True,                # RIJCOSX
    optimize=True,                              # OPT
    num_threads=4,                              # %pal nprocs 4
    output="output-h2o-pbe0",
    options=vq.RKSOptions(
        max_iter=200,
        scf_accelerator=vq.SCFAccelerator.KDIIS,  # KDIIS
        conv_tol_energy=1e-9,                   # ~ TIGHTSCF
    ),
)
```

ORCA's compact `!` keyword line trades concision for grammar
constraints (every keyword has to be recognised by the parser).
vibe-qc's Python input trades concision for sweep-friendliness,
the same script can loop over functionals / basis sets in plain
Python without a templating layer.

## Keyword crosswalk

### Method + basis (the `!` line)

| ORCA | vibe-qc |
|---|---|
| `! HF` | `method="rhf"` (or `"uhf"` for open-shell) |
| `! RKS PBE0` | `method="rks"`, `functional="PBE0"` |
| `! UKS B3LYP` | `method="uks"`, `functional="B3LYP"` (same VWN5 flavor in both codes, see the B3LYP-convention box below) |
| `! MP2` | `method="rmp2"` (after `run_rhf`); also `run_mp2(...)` directly |
| `! RI-MP2` | Same plus `density_fit=True, aux_basis="def2-tzvp-rifit"` |
| `! B2PLYP` | `method="rks", functional="B2PLYP"` via `run_b2plyp(...)` |
| `! DEF2-TZVP` | `basis="def2-tzvp"` |
| `! 6-31G*` | `basis="6-31g*"` |
| `! CC-PVDZ` | `basis="cc-pvdz"` |
| `! ANO-RCC` | `basis="ano-rcc"` |
| `! RIJCOSX` | `density_fit=True, cosx=True` |
| `! RIJK` | `density_fit=True, cosx=False` |
| `! D3BJ` | `dispersion="d3bj"` |
| `! D4` | `dispersion="d4"` |
| `! TIGHTSCF` | `opts.conv_tol_energy = 1e-9` (vibe-qc's default is already 1e-8, TIGHTSCF is the next step) |
| `! VERYTIGHTSCF` | `opts.conv_tol_energy = 1e-10` |
| `! OPT` | `run_job(..., optimize=True)` |
| `! FREQ` | `run_job(..., compute_hessian=True, mode="finite-diff")` (analytic Hessian roadmap) |
| `! NORMALPRINT / MINIPRINT / NORMALPRINT` | `verbose=` kwarg on `run_job` (PySCF-convention 0-6 levels, see [output_files](output_files.md) § Verbosity) |
| `! NOITER` | Not a separate flag, set `opts.max_iter = 0`; vibe-qc evaluates the initial guess once, returns `n_iter = 0` and `converged = False`, and performs no SCF iteration |

```{important}
**B3LYP convention.** vibe-qc and ORCA both mean the **VWN5**
flavor by the bare name (libxc id 475), so bare `B3LYP` matches
across the two codes with no translation, and the keyword pairs
mirror each other exactly:

* ORCA `! B3LYP` ↔ vibe-qc `functional="b3lyp"` (explicit
  spelling: `"b3lyp5"`).
* ORCA `! B3LYP/G` ↔ vibe-qc `functional="b3lyp/g"` (PySCF-style
  spelling: `"b3lypg"`), the Gaussian / VWN-RPA flavor, ~10-15
  mHa per heavy atom apart from the default.

Measured on H₂/STO-3G at 1.4 bohr: ORCA `B3LYP` −1.158669 vs
vibe-qc `b3lyp` −1.1586001; ORCA `B3LYP/G` −1.165470 vs vibe-qc
`b3lyp/g` −1.1654009. The ~69 µHa residual is **not** the grid:
it is ORCA's default Coulomb approximation (see the next
paragraph). With `! NORI` in the ORCA input the two `B3LYP`
energies agree to 3 nHa; the DFT grid (ORCA DefGrid2 vs vibe-qc
`orca-defgrid3`) accounts for 0.01 µHa of the difference. See
[functionals.md](functionals.md) § B3LYP.

**RI-J / RIJCOSX is on by default in ORCA, off by default in
vibe-qc.** ORCA ≥ 5 silently turns on Split-RI-J for pure
functionals and RIJCOSX for hybrids (its output says "RI-
approximation to the Coulomb term is turned on"), with the
`def2/J` fitting basis auto-attached. vibe-qc's `run_job`
computes the exact four-index J and exact K unless you ask for
density fitting (`density_fit=True` plus a fitting basis). The
resulting offset is the Weigend RI-J error, up to ~50 µHa per
atom (0.02 to 0.04 mHa/atom measured on the H₂, CH₄, H₂O and NH₃
def2-SVP pairs; issue #567), so an unmatched comparison
disagrees at the tens-of-µHa level while the HF pair agrees to
1e-10 Ha. To compare like with like, either add `! NORI` to the
ORCA input (exact J and K on both sides) or run vibe-qc with
`density_fit=True` and the same fitting basis (`def2-universal-
jfit` matches ORCA's `def2/J`).
```

### `%scf` block

| ORCA `%scf` | vibe-qc |
|---|---|
| `MaxIter n` | `opts.max_iter = n`; zero is the `NOITER` path above, while negative values are rejected before SCF work |
| `ConvForced 1` (force run despite non-convergence) | (vibe-qc returns a result object with `converged = False` either way) |
| `CNVDIIS n` (DIIS error threshold) | `opts.conv_tol_grad = n` |
| `DIIS true/false` | `opts.scf_accelerator = vq.SCFAccelerator.DIIS` / `.EDIIS_DIIS` |
| `KDIIS true` (ORCA's opt-in `!KDIIS`; Kollmar 1997) | `opts.scf_accelerator = vq.SCFAccelerator.KDIIS` |
| `SOSCF true` | `opts.soscf_threshold = 1.0` (with `opts.soscf_opts.trust_radius = ...`) |
| `Damp DampFac p / DampErr e` | `opts.damping = p`; the error-driven damping isn't yet exposed |
| `Shift Shift n` (level shift in eV) | `opts.level_shift = n / 27.2114` (Hartree) |
| `Guess HCore / PModel / PAtom / Hueckel / Read` | `opts.initial_guess = vq.InitialGuess.HCORE / SAD / PATOM / HUECKEL / READ` |
| `BrokenSym n1 n2` | Manual SAD with custom multiplicity per fragment + MOM (Phase 17h). Not a one-liner today. |
| `STABPerform true` (Hessian stability check) | `opts.stability_check = True` (default-on for UHF and for UKS where the complete response is available; UKS meta-GGA, range-separated, VV10, DFT+U, active hybrid RIJCOSX, and coupled-solvent requests fail closed; RHF/RKS not yet covered) |
| `SCFConvForced 1` | Force `result.converged = False` paths to still return a result, vibe-qc always returns the result + `converged` flag. |

### `%basis` block

| ORCA `%basis` | vibe-qc |
|---|---|
| `NewGTO 78 "LANL2DZ" end` (per-element basis override) | Use a `basis_library/custom/*.g94` file; vibe-qc applies it per element. See [basis_sets](basis_sets.md) § Custom basis sets. |
| `NewECP 78 "LANL2DZ" end` | `opts.ecp_centers = [vq.ECPCenter(Z=78, xyz=...)]` + `opts.ecp_library = "lanl2dz"` |
| `Aux "def2/J"` | `opts.aux_basis = "def2-universal-jkfit"` |
| `AuxJ "def2-svp/J"` / `AuxC "cc-pVDZ/C"` | `opts.aux_basis_j` / `opts.aux_basis_c` (these split-aux fields are roadmap; today `aux_basis` is the single field) |

### `%method` block

| ORCA `%method` | vibe-qc |
|---|---|
| `Functional B3LYP` | `opts.functional = "B3LYP"` (same flavor in both codes; see the B3LYP-convention box above) |
| `Method DFT` | implicit by calling `run_rks` / `run_uks` |
| `RI on` | `opts.density_fit = True` |
| `RunTyp Energy / Gradient / Opt / Freq` | call `run_job(optimize=True)` / `run_job(compute_hessian=True)` / `compute_gradient_rhf(...)` etc. directly |

### `%pal` block

| ORCA `%pal` | vibe-qc |
|---|---|
| `nprocs n` | `num_threads=n` on `run_job`, or `OMP_NUM_THREADS=n` env var. **Production parallelism is OpenMP (shared-memory, single node).** The optional `[mpi]` extra (v0.12.0, `mpi4py`) adds distributed-memory shell-pair distribution and k-point / state task farming; the GPW grid overlay is experimental, see [features](../features.md). |

### `%output` block

| ORCA `%output` | vibe-qc |
|---|---|
| `Print[ P_MOs ] 1` (print orbitals to .out) | Default, vibe-qc always prints the orbital table (up to `opts.n_virtual = 5`). |
| `Print[ P_OverlapM ] 1` (print overlap matrix) | Not directly, `result.overlap` is the NumPy array for programmatic access. |
| `XYZFile true` | Always-on as of v0.8.x. The final geometry lands in `{stem}.xyz`. |
| `MOPrintLimit 0.001` | `opts.n_virtual = N` controls how many virtuals appear in the orbital table. |
| `MullikenPop true` | Default, vibe-qc always reports Mulliken + Löwdin + Mayer in the post-SCF properties block. See [properties](properties.md). |

## RIJCOSX in detail

ORCA's flagship hybrid-DFT recipe is `!PBE0 D3BJ DEF2-TZVP RIJCOSX
TIGHTSCF`. The vibe-qc equivalent is:

```python
opts = vq.RKSOptions()
opts.functional   = "PBE0"
opts.dispersion   = "d3bj"
opts.density_fit  = True
opts.cosx         = True
opts.aux_basis    = "def2-tzvp-jk"     # default; auto-picked by basis name
opts.cosx_grid    = vq.default_cosx_grid_options()  # = ORCA's default GridX
opts.conv_tol_energy = 1e-9             # TIGHTSCF
```

The RIJCOSX SCF + analytic gradient match ORCA's to **0.13 mHa**
on glycine / def2-TZVP per [RIJCOSX SCF and analytic gradients on glycine](../tutorial/rijcosx_glycine.md).

## ECPs

ORCA's `! DEF2-TZVP` automatically loads the matching ECPs for
heavy atoms. vibe-qc keeps the orbital basis and the ECP **as two
separate fields** so you can mix-and-match (e.g. def2-TZVP orbital
with `ecp46mdf` if you want the Stuttgart MDF instead of the
def2-bundled SDD). The shipped `vq.auto_ecp_centers(mol, basis_name)`
helper covers the common "use the matching ECP for this bundled basis"
case:

```python
# Manual recipe (always works):
opts.ecp_centers = [
    vq.ECPCenter(Z=78, xyz=[0.0, 0.0, 0.0]),
]
opts.ecp_library = "ecp46mdf"   # or "lanl2dz" for HW pseudopotential

# Auto recipe:
opts.ecp_centers, opts.ecp_library = vq.auto_ecp_centers(mol, "def2-tzvp")
```

See [ecp](ecp.md) and [Pt cluster with LANL2DZ: heavy-element ECPs](../tutorial/pt_cluster_lanl2dz.md).

## What ORCA does that vibe-qc doesn't (yet)

Most of the gaps listed in earlier editions of this guide have closed
across v0.9.0--v0.15.x. The items below track the current state
(August 2026, v0.15.x):

* **Coupled-cluster (DLPNO-CCSD(T)).** Available since v0.14.0
  (closed-shell) and v0.15.x (open-shell DLPNO-UCCSD(T)). Both routes
  carry the full truncation-control surface (PAO, PNO, weak-pair,
  occupied-coupling, TNO). See [CCSD and coupled-cluster methods](ccsd.md).
* **MR-CI / CASSCF / NEVPT2.** CASSCF shipped in v0.11.0; CASPT2 /
  NEVPT2 / MS-CASPT2 with analytic gradients in v0.12.0.
  State-averaged CASSCF, selected-CI backends, and IC-CASPT2 gradients
  are available. See [non-HF solvers](non_hf_solvers.md).
* **TD-DFT and excited states.** Shipped in v0.12.0 (Casida + TDA for
  RHF / RKS / UHF / UKS) with NTO analysis and spectra output.
  See the [TD-DFT tutorial](../tutorial/excited_states_tddft.md).
* **Range-separated hybrids beyond ωB97X / ωB97X-D.** Shipped.
  ωB97X-V and ωB97M-V landed in v0.14.0 with VV10 nonlocal
  correlation. HSE06, CAM-B3LYP, and the RSH machinery shipped in
  v0.9.0. See [functionals](functionals.md).
* **Meta-GGAs (some).** TPSS, TPSSh, M06-L, M06-2X, SCAN, r²SCAN,
  and r²SCAN01 shipped in v0.9.0--v0.14.0 with τ-density analytic
  gradients. See [functionals](functionals.md).
* **Composite "3c" methods.** All seven composites (HF-3c, PBEh-3c,
  B97-3c, B3LYP-3c, r²SCAN-3c, ωB97X-3c, HSE-3c) shipped in v0.9.0.
  Use `method="hf-3c"` etc. via `run_job`. See
  [composites](composites.md).
* **Periodic SCF in ORCA-style.** ORCA does molecular only, but if
  you're moving from ORCA to do solids, vibe-qc's periodic stack is
  what you want (see [periodic_systems](periodic_systems.md)).

## What vibe-qc does that ORCA doesn't (or does differently)

Things you gain by switching:

* **Auto-citations.** Every job writes a `.bibtex` /
  `.references` pair; ORCA prints the citation list to the
  `.out` but doesn't give you the BibTeX file. See
  [citations](citations.md).
* **Python scripting.** Sweep over functionals / basis / k-mesh
  in a `for` loop, not by maintaining N input files. The
  [PySCF guide](from_pyscf.md) elaborates on the workflow shape.
* **Periodic SCF.** ORCA is molecular only; vibe-qc has 1D / 2D /
  3D periodic HF and KS-DFT today.
* **MPL-2.0 license.** Drop into any project, including
  closed-source software.
* **`vq` queue.** Submit jobs to a remote box with a
  vibe-qc-aware preflight; ORCA users typically run on SLURM/PBS
  via `sbatch`. See [Submitting a job to a remote machine with `vq`](../tutorial/vq_queue_remote_job.md).
* **Pre-flight memory estimator.** Aborts before the SCF starts
  if the dense-ERI / DFT-grid / MP2 working set exceeds available
  RAM. See [memory](memory.md).

## Validating vibe-qc against ORCA

ORCA is **the** cross-validation reference for vibe-qc's molecular
hybrid-DFT + RIJCOSX surface. The
[parity_matrix_orca/](https://github.com/vibe-qc/vibe-qc/tree/main/examples/regression/parity_matrix_orca)
suite runs the same `Atoms` through vibe-qc and out-of-process
ORCA, parses both `.out` files, and produces a side-by-side
comparison artefact.

Quick parity check on your own system:

```sh
# 1. Run in ORCA:
orca my-input.inp > my-input.orca.out

# 2. Run in vibe-qc:
python my-input.py            # writes my-input.out / .bibtex / etc.

# 3. Compare the two total energies:
grep "FINAL SINGLE POINT ENERGY" my-input.orca.out
grep "Total energy:" my-input.out
```

Expected agreement after the v0.8.0 B3LYP-convention alignment:
~µHa per atom for closed-shell HF / DFT; ~0.13 mHa per atom for
RIJCOSX hybrids; ~10 µHa per atom for D3(BJ) corrections (the
dispersion is bit-identical because both codes use
[Grimme's reference implementation](https://github.com/dftd3/simple-dftd3)).

## See also

- [from_pyscf](from_pyscf.md), migration guide for the other
  most-common starting code.
- [from_crystal](from_crystal.md), migration guide for the
  CRYSTAL family (solid-state focus).
- [user_guide/external_codes](external_codes.md) § ORCA, the
  out-of-process ORCA runner the parity suite uses.
- [user_guide/density_fitting](density_fitting.md), RIJCOSX,
  RIJK, four-index reference paths.
- [RIJCOSX SCF and analytic gradients on glycine](../tutorial/rijcosx_glycine.md)
  - RIJCOSX SCF + gradient parity check against ORCA 6.1.1.
- [Cross-validating against ORCA, Psi4, and PySCF](../tutorial/cross_validation.md)
  - full end-to-end same-Atoms-through-N-codes recipe.
