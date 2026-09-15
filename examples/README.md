# vibe-qc examples

Self-contained input scripts in the style of classic QC programs
(Gaussian / ORCA / NWChem). Each one is a runnable Python file —
edit and rerun freely.

## Which repository supplies each tool?

These examples live in [vibe-qc](https://github.com/vibe-qc/vibe-qc).
Install the core using [getting started](../docs/getting_started.md), then add
only the companions needed by your chosen example:

| Repository | Role in the examples | Setup |
|---|---|---|
| [vibe-qc](https://github.com/vibe-qc/vibe-qc) | Runs calculations and writes QVF; retains the `vibe-basis/` toolkit | [Core installation](../docs/installation.md) |
| [vibe-view](https://github.com/vibe-qc/vibe-view) | Opens generated QVF archives; installed separately | [Viewer setup](../docs/tutorial/vibe_view_getting_started.md) |
| [vibe-queue](https://github.com/vibe-qc/vibe-queue) | Schedules jobs through the `vq` CLI and import package | [Queue setup](../docs/user_guide/queue.md) |
| [qvf](https://github.com/vibe-qc/qvf) | Publishes the format specification, conformance corpus, and optional reference implementations | [Format reference](../docs/qvf/index.md) |

The QVF repository is not a runtime dependency of vibe-qc. The independent
implementations exchange files validated against its published format. Each
repository releases independently; a core tag does not select a viewer or
queue version. [Repositories and downloads](../docs/installation.md#repositories-and-downloads)
lists all four GitLab sources and GitHub mirrors, current access requirements,
and the owning projects' tags and release artifacts.

The companion manuals are published independently at
[vibe-view docs](https://vibe-qc.com/vibe-view/docs/) and
[vibe-queue docs](https://vibe-qc.com/vibe-queue/docs/). These core calculation
examples remain here and are released with vibe-qc; viewer-only examples and
queue examples belong to the corresponding companion checkout.

## Running an example

You need to use the **virtual-env's Python** so `import vibeqc`
resolves. The recommended workflow copies an input into a calculation
directory outside the checkout:

```sh
mkdir -p ~/vibeqc-runs/examples/h2o-rhf
cp examples/molecular/input-h2o-rhf.py ~/vibeqc-runs/examples/h2o-rhf/
cd ~/vibeqc-runs/examples/h2o-rhf
<vibe-qc-checkout>/.venv/bin/python input-h2o-rhf.py
```

Replace `<vibe-qc-checkout>` with the clone path. Outputs land alongside the
copied script because examples use `Path(__file__).parent` as the output
stem. This keeps `.out`, `.qvf`, `.molden`, trajectories, and plots out of the
source tree.

For a quick disposable run, this also works from the repository root:

```sh
.venv/bin/python examples/molecular/input-h2o-rhf.py
```

That form writes beside the original example. Inspect the files, then remove
the generated artifacts before committing. `git status --short` should not
show calculation output when you finish.

Curated reference bundles for the docs are regenerated on `vq` and
published under [`docs/_static/examples/`](../docs/_static/examples/)
instead. Those bundles include the copied input, full stdout/stderr,
matched output files, QVF archives where available, and vibe-view
screenshots for QVF-backed examples.

## Layout

```
examples/
├── molecular/    — Molecule-only HF / DFT / MP2, opt, vibrations, cubes
├── periodic/     — PeriodicSystem (1D chains + 3D crystals)
├── mlip/         — optional MACE runtime checks and calculation workflows
├── qvf_containers/ — pending-to-settled single-file job lifecycle
├── workflows/    — multi-step protocols (NEB, …)
├── ase_compare/  — cross-code validation vs PySCF / ORCA / Psi4
├── ase_workflows/— pure-ASE workflows with vibe-qc as calculator
├── debug/        — diagnostic scripts for SCF debugging
├── plots/        — matplotlib plot regenerators for tutorial figures
├── regression/   — automated parity test suite
└── quickstart.py — bare-bones first calculation
```

## Choose an example by goal

| Goal | Start with | Continue with |
|---|---|---|
| Verify a new installation | [`quickstart.py`](quickstart.py) | [`molecular/input-h2o-rhf.py`](molecular/input-h2o-rhf.py) |
| Run molecular DFT and optimize | [`molecular/input-h2o-rks-pbe-d3-opt.py`](molecular/input-h2o-rks-pbe-d3-opt.py) | geometry-optimization section below |
| Treat an open-shell molecule | [`molecular/input-oh-radical.py`](molecular/input-oh-radical.py) | [`molecular/input-oh-ump2.py`](molecular/input-oh-ump2.py) |
| Run Microsoft SKALA neural XC | [`molecular/input-h2-rks-skala.py`](molecular/input-h2-rks-skala.py) | [worked SKALA tutorial](../docs/tutorial/skala_neural_xc.md), [experimental periodic smoke run](periodic/input-h2-cell-rks-skala-gdf.py) |
| Compare direct and conventional SCF | [`molecular/input-h2o-rhf-direct.py`](molecular/input-h2o-rhf-direct.py) | [direct-SCF tutorial](../docs/tutorial/direct_scf_memory_tradeoff.md) |
| Start a periodic calculation | [`periodic/input-h-chain-uniform.py`](periodic/input-h-chain-uniform.py) | [`periodic/input-lih-pob-tzvp.py`](periodic/input-lih-pob-tzvp.py) |
| Optimize or characterize through ASE | [`ase_workflows/optimize-via-ase-bfgs.py`](ase_workflows/optimize-via-ase-bfgs.py) | vibrations, NEB, and MD entries below |
| Run a multi-step reaction workflow | [`workflows/input-nh3-umbrella-neb.py`](workflows/input-nh3-umbrella-neb.py) | [NEB tutorial](../docs/tutorial/neb_reaction_path.md) |
| Produce or inspect QVF containers | [`qvf_containers/build_h2_job.py`](qvf_containers/build_h2_job.py) | [container tutorial](../docs/tutorial/qvf_job_containers.md) |
| Validate against another program | [`ase_compare/compare-h2o-hf.py`](ase_compare/compare-h2o-hf.py) | cross-code validation section below |

Each example should answer four questions before you modify it: which system
is represented, which method and basis are used, where files will be written,
and what numerical result or diagnostic defines success. Change one of those
at a time and preserve the unmodified input beside any comparison run.

`ModuleNotFoundError: No module named 'vibeqc'` means you used the
wrong Python — see [the running guide](../docs/running.md) for the
full story (venv activation, `OMP_NUM_THREADS`, output capture for
long runs, etc.).

**Resource expectations** (Apple M2 baseline, single core):

| Example class | Peak RAM | Wall time |
|---|---|---|
| Molecular HF / DFT at sto-3g / 6-31G\*\* on small systems | ~70–250 MB | <1 s |
| Geometry optimization, water dimer / trimer | ~150 MB | 5–60 s |
| Vibrational frequencies (finite-difference Hessian) | ~250 MB | ~30 s on H₂O, scales as 6N |
| 1D periodic SCF (H-chain, sto-3g, k=8) | ~200 MB | a few seconds |
| 3D bulk SCF via EWALD_3D (LiH, sto-3g) | ~250 MB | ~10–60 s |
| Madelung / point-charge Ewald | <50 MB | <0.05 s |
| NEB (NH₃ inversion, 7 images, HF/STO-3G) | ~150 MB | ~30 s |

SKALA is omitted from this timing table until a supported Linux reference run
is published. Real evaluation requires Python 3.11 through 3.13 and the
optional `[skala]` extra, and its full-grid differentiable model needs more
memory than an ordinary semilocal DFT grid. Both source examples accept
`--dry-run`, which imports no PyTorch and downloads no checkpoint.

Use the examples as ballparks: pob-TZVP on a real solid bumps the
basis count by 5–10×, which costs roughly $\mathcal{O}(N_{bf}^4)$ in
the Fock build. Multi-threaded scaling and tight-cell convergence
guidance live in [tutorial 18 (parallel)](../docs/tutorial/parallel_execution.md)
and [tutorial 24 (periodic SCF convergence)](../docs/tutorial/periodic_scf_convergence.md).

## Round-trip QVF job containers - `qvf_containers/`

These examples start before a calculation has run. They create a pending
`.qvf` carrying a structure plus declarative `job.spec`; `vibeqc run` updates
that same archive in place with results, a complete run record, and terminal
status.

```sh
.venv/bin/python examples/qvf_containers/build_h2_job.py
.venv/bin/vibeqc run examples/qvf_containers/runs/h2-rhf.qvf
.venv/bin/python examples/qvf_containers/inspect_job.py \
    examples/qvf_containers/runs/h2-rhf.qvf
```

| Script | What it does |
| --- | --- |
| [`input-qvf-container-roundtrip.py`](input-qvf-container-roundtrip.py) | Runs the complete pending-to-settled molecular workflow in one script and checks the embedded log |
| [`qvf_containers/build_h2_job.py`](qvf_containers/build_h2_job.py) | Builds a pending molecular H2/RHF/STO-3G container for the separate `vibeqc run` CLI step |
| [`qvf_containers/build_periodic_he_job.py`](qvf_containers/build_periodic_he_job.py) | Builds a tiny pending periodic He/SCC-DFTB Gamma-only container |
| [`qvf_containers/inspect_job.py`](qvf_containers/inspect_job.py) | Validates a pending or settled container and prints its spec, lifecycle status, section kinds, and run history |

See the
[complete job-container tutorial](../docs/tutorial/qvf_job_containers.md) for
local execution, first-class vq submit/fetch, terminal viewing, graphical
viewing, failure semantics, and deliberate re-runs.

## Single-point energies — `molecular/`

| Script | What it does |
| --- | --- |
| [`molecular/input-h2o-rhf.py`](molecular/input-h2o-rhf.py) | Closed-shell Hartree-Fock on H2O / 6-31G\* |
| [`molecular/input-h2o-dft.py`](molecular/input-h2o-dft.py) | Kohn-Sham DFT with the PBE functional |
| [`molecular/input-h2-rks-skala.py`](molecular/input-h2-rks-skala.py) | H2 / def2-SVP fixed-geometry RKS with Microsoft SKALA-1.1; supports a model-free `--dry-run` |
| [`molecular/input-h2o-rks-pbe-d3-opt.py`](molecular/input-h2o-rks-pbe-d3-opt.py) | RKS-PBE / 6-31G\* + D3(BJ) dispersion + geometry opt — the homepage example |
| [`molecular/input-oh-radical.py`](molecular/input-oh-radical.py) | Open-shell UHF on the OH radical |
| [`molecular/input-oh-ump2.py`](molecular/input-oh-ump2.py) | UMP2 post-SCF correlation on the OH radical |
| [`molecular/input-h2o-cube.py`](molecular/input-h2o-cube.py) | H2O density + frontier MOs as Gaussian cube files for VMD / Avogadro |
| [`molecular/input-h2o-vibrations.py`](molecular/input-h2o-vibrations.py) | RHF + analytic Hessian + harmonic frequencies |

## Geometry optimization — `molecular/`

| Script | What it does |
| --- | --- |
| [`molecular/input-h2o-opt.py`](molecular/input-h2o-opt.py) | Relaxation of H2O at HF/6-31G\* (via ASE BFGSLineSearch) |
| [`molecular/input-h2o-dimer-opt.py`](molecular/input-h2o-dimer-opt.py) | Water dimer (H-bonded), R(OO) ≈ 2.98 Å at HF/6-31G\* |
| [`molecular/input-h2o-trimer-opt.py`](molecular/input-h2o-trimer-opt.py) | Cyclic water trimer, asymmetric "uud" ring minimum |

H-bonded clusters are weakly bound — the optimizer uses a conservative
``fmax=0.2`` eV/Å tolerance (about 0.005 Ha/bohr) because the force
precision at the default SCF tolerance is around 0.1 eV/Å. Tighten
``conv_tol_grad`` in the RHF options if you need a stricter fit.

## Correlated + wavefunction methods — `molecular/`

| Script | What it does |
| --- | --- |
| [`molecular/input-oh-ump2.py`](molecular/input-oh-ump2.py) | UMP2 post-SCF correlation on the open-shell OH radical |
| [`molecular/input-h2o-scs-mp2.py`](molecular/input-h2o-scs-mp2.py) | Spin-component-scaled MP2 (SCS-MP2) on H2O |
| [`molecular/input-h2o-b2plyp.py`](molecular/input-h2o-b2plyp.py) | B2PLYP double hybrid — hybrid SCF + scaled RI-MP2 in one call |
| [`molecular/input-h2o-ri-mp2-residual.py`](molecular/input-h2o-ri-mp2-residual.py) | RI-MP2 with the auxiliary-fit residual diagnostic |
| [`molecular/input-h2-fci.py`](molecular/input-h2-fci.py) | Exact Full CI vs Hartree-Fock — the basis-set correlation energy of H2 |

The `vibeqc.solvers` wavefunction methods (`fci`, `selected_ci`,
`dmrg`, `v2rdm`, `transcorrelated_ci`) are also reachable through
`run_job(method=…)`; see
[`docs/user_guide/non_hf_solvers.md`](../docs/user_guide/non_hf_solvers.md)
and [`docs/tutorial/non_hf_solvers.md`](../docs/tutorial/non_hf_solvers.md).

## MACE machine-learning interatomic potentials — `mlip/`

MACE requires a separate Python 3.13-or-earlier environment with the
optional `[mace]` extra. Start with the dependency-only health check:

```sh
.venv-mace/bin/python examples/mlip/00_runtime_healthcheck.py
```

| Script | What it does |
| --- | --- |
| [`mlip/00_runtime_healthcheck.py`](mlip/00_runtime_healthcheck.py) | Reports runtime versions, source revision when available, public periodic APIs, default model metadata, and cache root without loading weights |
| [`mlip/01_water_mace.py`](mlip/01_water_mace.py) | Molecular H₂O single point and optimization through `run_job(method="mace")` |
| [`mlip/02_periodic_mace_silicon.py`](mlip/02_periodic_mace_silicon.py) | Direct periodic Si two-point strain probe with one retained model; optional cell relaxation |
| [`mlip/03_neb_mace.py`](mlip/03_neb_mace.py) | NH₃ umbrella-inversion NEB through the molecular MACE dispatch |
| [`mlip/04_surface_adsorption.py`](mlip/04_surface_adsorption.py) | Exploratory materials-model adsorption protocol, not reference adsorption data |
| [`mlip/05_mace_pm6_relative_curve.py`](mlip/05_mace_pm6_relative_curve.py) | Independently normalized MACE and PM6 H₂O stretch curves; never mixes their absolute energies |
| [`mlip/06_optimize_water_off23_for_dlpno.py`](mlip/06_optimize_water_off23_for_dlpno.py) | Explicitly licensed MACE-OFF23 geometry stage for the molecular DLPNO handoff |
| [`mlip/07_dlpno_ccsdt_on_mace_geometry.py`](mlip/07_dlpno_ccsdt_on_mace_geometry.py) | DLPNO-CCSD(T) energy and RHF-reference properties on the MACE-optimized geometry |
| [`mlip/08_relax_h2o_hbn_then_periodic_dft.py`](mlip/08_relax_h2o_hbn_then_periodic_dft.py) | Fixed-cell true-sheet MACE relaxation, then optional periodic PBE |

Periodic MACE does not dispatch through `run_periodic_job`. Use
`run_periodic_mace` for one point, `PeriodicMACEEvaluator` for a
sequential fixed-model scan, and `optimize_periodic_mace_cell` for cell
relaxation of 3D crystals. Use `optimize_periodic_mace_positions` for a
true slab or fixed-cell crystal. See the
[MACE tutorial](../docs/tutorial/mace_mlip.md), the
[geometry-to-electronic-structure tutorial](../docs/tutorial/mace_geometry_to_electronic_structure.md), and
[MLIP reference](../docs/user_guide/mlip.md) for model licensing,
provenance, cache behavior, and scientific claim boundaries.

## Periodic systems — `periodic/`

The high-level ``run_periodic_job`` is the periodic counterpart of
``run_job`` — it dispatches to the right periodic driver and writes
the same artefact family (``.out`` / ``.system`` / ``.bibtex`` /
``.references`` / ``.xyz`` / ``.POSCAR`` / ``.xsf``). Several
examples below predate it and call the periodic drivers directly,
writing their own ``.out`` / ``.system`` / ``.perf`` files via
``perf_log`` + ``write_system_manifest``; both styles are valid.

### 1D periodic SCF

| Script | What it does |
| --- | --- |
| [`periodic/input-h-chain-uniform.py`](periodic/input-h-chain-uniform.py) | 1D H2 molecular crystal, k-mesh convergence study (HF / pob-TZVP) |
| [`periodic/input-h-chain-peierls.py`](periodic/input-h-chain-peierls.py) | Peierls-dimerisation scan on a 1D H-chain — E(δ) drops as the uniform chain distorts toward H2 units |
| [`periodic/input-h-chain-rks-pbe.py`](periodic/input-h-chain-rks-pbe.py) | DFT counterpart of `input-h-chain-uniform.py` — RKS-PBE on the 1D H2 chain |
| [`periodic/input-h-chain-bands.py`](periodic/input-h-chain-bands.py) | Hcore band structure + density of states for the same 1D chain |
| [`periodic/input-k-mesh-convergence.py`](periodic/input-k-mesh-convergence.py) | Walks every `vibeqc.KPoints` flavour (Γ, MP, IBZ-reduced, …) on the same H-chain |

### 3D crystals — periodic SCF

| Script | What it does |
| --- | --- |
| [`periodic/input-lih-pob-tzvp.py`](periodic/input-lih-pob-tzvp.py) | LiH cubic rocksalt — pob-TZVP / RHF / EWALD_3D, multi-k IBZ, full v0.5.x logging surface |
| [`periodic/input-h2-cell-rks-skala-gdf.py`](periodic/input-h2-cell-rks-skala-gdf.py) | Experimental 3D H2 / STO-3G / Gamma GDF SKALA API smoke run; not a periodic benchmark |
| [`periodic/input-mgo-pob-tzvp.py`](periodic/input-mgo-pob-tzvp.py) | MgO rocksalt — pob-TZVP / RHF / EWALD_3D (CRYSTAL Tutorial port) |
| [`periodic/input-mgo-rocksalt-rhf.py`](periodic/input-mgo-rocksalt-rhf.py) | MgO rocksalt — STO-3G / RHF / Γ-only Ewald-3D — vibe-qc half of the v0.7 RHF parity test vs PySCF.pbc |
| [`periodic/input-mgo-rocksalt-rks-pbe.py`](periodic/input-mgo-rocksalt-rks-pbe.py) | MgO rocksalt — STO-3G / RKS-PBE / Γ-only Ewald-3D — DFT counterpart of the parity test |
| [`periodic/input-al2o3-cubic-rhf.py`](periodic/input-al2o3-cubic-rhf.py) | Al2O3 cubic stuffed-fluorite model — STO-3G / RHF / Γ-only Ewald-3D — vibe-qc half of the v0.7 RHF parity test vs PySCF.pbc |
| [`periodic/input-al2o3-cubic-rks-pbe.py`](periodic/input-al2o3-cubic-rks-pbe.py) | Al2O3 cubic — STO-3G / RKS-PBE / Γ-only Ewald-3D — DFT counterpart |
| [`periodic/input-nacl-sto3g-dft.py`](periodic/input-nacl-sto3g-dft.py) | NaCl rocksalt — STO-3G / RKS-LDA / EWALD_3D — debug-friendly solid demo (CRYSTAL Tutorial port) |
| [`periodic/input-si-bands.py`](periodic/input-si-bands.py) | Silicon diamond — Hcore band structure (CRYSTAL Tutorial port) |

### 3D lattice electrostatics (Ewald)

| Script | What it does |
| --- | --- |
| [`periodic/input-madelung-constants.py`](periodic/input-madelung-constants.py) | Madelung constants for NaCl, CsCl, ZnS via `ewald_point_charge_energy` — reproduces textbook values to 8+ digits |

### Symmetry & infrastructure

| Script | What it does |
| --- | --- |
| [`periodic/input-nacl-symmetry.py`](periodic/input-nacl-symmetry.py) | NaCl Pm-3m — 19× compression of the overlap `LatticeMatrixSet` via atom-pair orbits (SYM2c) + Wigner-D check |

## SCF infrastructure showcases — `molecular/`

| Script | What it does |
| --- | --- |
| [`molecular/input-h2-tight-canonical-orth.py`](molecular/input-h2-tight-canonical-orth.py) | Tight H2 with aug-cc-pVTZ — pre-flight linear-dependence diagnostic + canonical-orthogonalisation threshold sweep |
| [`molecular/input-progress-demo.py`](molecular/input-progress-demo.py) | Live SCF progress logging via the v0.5.x ProgressLogger surface |
| [`molecular/input-h2o-initial-guess-comparison.py`](molecular/input-h2o-initial-guess-comparison.py) | HCORE / SAD / SAP / AUTO initial-guess comparison on H2O |
| [`molecular/input-h2o-rhf-direct.py`](molecular/input-h2o-rhf-direct.py) | Direct (integral-on-the-fly) SCF Fock-build mode |
| [`molecular/input-h2o-rhf-conventional.py`](molecular/input-h2o-rhf-conventional.py) | Conventional (in-core 4-index ERI) SCF Fock-build mode |
| [`molecular/input-glycine-rhf-direct.py`](molecular/input-glycine-rhf-direct.py) | Explicit DIRECT SCF on glycine / def2-SVP (95 BF), avoiding the 621.42 MiB dense ERI tensor |

See [`docs/user_guide/scf_modes.md`](../docs/user_guide/scf_modes.md)
for the AUTO / CONVENTIONAL / DIRECT Fock-build dispatch.

## Multi-step workflows — `workflows/`

| Script | What it does |
| --- | --- |
| [`workflows/input-nh3-umbrella-neb.py`](workflows/input-nh3-umbrella-neb.py) | NH3 umbrella inversion via climbing-image NEB / RHF / STO-3G — transition-state search demo |

## ASE workflows

Pure-ASE workflows where vibe-qc is the backing calculator.
Drives `ase.optimize.BFGS` / `ase.vibrations.Vibrations` /
`ase.mep.NEB` / `ase.md.VelocityVerlet` directly — no glue code,
no extra ceremony beyond `atoms.calc = VibeQC(...)`.

Lives under [`ase_workflows/`](ase_workflows/). All scripts run
out of the box with `pip install '.[ase]'`.

| Script | What it does |
| --- | --- |
| [`ase_workflows/optimize-via-ase-bfgs.py`](ase_workflows/optimize-via-ase-bfgs.py) | H₂O / 6-31G\* geometry opt via ASE's BFGS, with energy + |F|max convergence plot |
| [`ase_workflows/vibrations-via-ase-vibrations.py`](ase_workflows/vibrations-via-ase-vibrations.py) | H₂O vibrational frequencies via `ase.vibrations.Vibrations` (FD on analytic gradients) |
| [`ase_workflows/nh3-neb-via-ase.py`](ase_workflows/nh3-neb-via-ase.py) | NH₃ umbrella inversion via climbing-image NEB |
| [`ase_workflows/md-water-nve.py`](ase_workflows/md-water-nve.py) | Velocity-Verlet NVE on H₂O — energy-conservation diagnostic (drift < 10 meV / 25 fs is the force-energy consistency test) |
| [`ase_workflows/surface-h2-pt111-singlepoint.py`](ase_workflows/surface-h2-pt111-singlepoint.py) | Pt(111) slab + atop H₂ geometry fixture for the periodic ASE adapter; writes the committed XYZ setup and skips SCF until the adapter lands |

See [tutorial 26 (cross-validation)](../docs/tutorial/cross_validation.md)
for the same calculations driven through the comparison framework
that puts vibe-qc next to PySCF / ORCA / Psi4.

## Cross-code validation

Validate vibe-qc results against external QC codes (PySCF, ORCA,
Psi4, …) using the [`vibeqc.benchmark`](../python/vibeqc/benchmark.py)
framework. Each script auto-detects which codes are installed,
skips missing ones gracefully (run on a developer laptop with just
PySCF; full table on a cross-checking workstation with ORCA), and
asserts numerical agreement at the end.

Lives under [`ase_compare/`](ase_compare/). To enable ORCA / Psi4
comparisons, see [the external codes user guide](../docs/user_guide/external_codes.md).

| Script | What it does |
| --- | --- |
| [`ase_compare/compare-h2o-hf.py`](ase_compare/compare-h2o-hf.py) | H₂O / RHF / 6-31G\* across vibe-qc + PySCF + ORCA — tight tolerance (1e-5 eV) since the SCF skeleton is identical |
| [`ase_compare/compare-h2o-dft.py`](ase_compare/compare-h2o-dft.py) | H₂O / RKS-PBE / 6-31G\* — looser tolerance (5 mHa) for per-code DFT-grid differences; adds dipole comparison |
| [`ase_compare/compare-water-dimer-dispersion.py`](ase_compare/compare-water-dimer-dispersion.py) | Water dimer interaction energy with and without D3-BJ — verifies vibe-qc's Grimme-parameter implementation matches ORCA's |
| [`ase_compare/compare-basis-convergence.py`](ase_compare/compare-basis-convergence.py) | H₂O / RHF across 5 basis sets × 3 codes; agreement at every basis to <1 meV |
| [`ase_compare/compare-nh3-neb-barrier.py`](ase_compare/compare-nh3-neb-barrier.py) | Full NH₃ NEB barrier vs ORCA NEB on the same protocol; 1 kcal/mol agreement |

```{tip}
**Sanity-check the setup** with the diagnostic script before
running the comparison scripts:

    ./scripts/check_external_codes.sh

It probes every supported calculator and prints the colour-coded
availability table. Exits 0 when at least vibe-qc + one peer code
are available (cross-validation possible), 1 otherwise.
```

## Output files

Every script writes three output families, named by the script's `output=`
stem:

- `{stem}.out` — plain-text log: banner, atom table, SCF trace,
  energy breakdown (DFT), orbital-energy table with HOMO/LUMO markers.
- `{stem}.molden` — molecular orbitals for any molden-aware viewer
  (Jmol, Avogadro, Molden itself, IQmol, MolView).
- `{stem}.traj` — ASE binary trajectory, written only for geometry
  optimization. View as an animation with:
  ```sh
  ase gui {stem}.traj
  ```

## Geometry input

The newer examples (`molecular/input-h2o-rks-pbe-d3-opt.py`,
`molecular/input-oh-ump2.py`, every `periodic/input-*.py`) **inline
the geometry** directly in the source as a list of `Atom(Z, [x, y, z])`
calls — bohr coordinates, fully self-contained, no external files
required.

The original H2O / H2 family (`molecular/input-h2o-*.py`) read from
shared XYZ files at `examples/h2.xyz`, `examples/h2o.xyz`,
`examples/h2o_dimer.xyz`, `examples/h2o_trimer.xyz` via
`Molecule.from_xyz(...)`. Swap in your own XYZ file to run these
methods on other molecules — both patterns are plain Python and
trivial to edit.
