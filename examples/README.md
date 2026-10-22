# vibe-qc examples

Self-contained input scripts in the style of classic QC programs
(Gaussian, ORCA, NWChem). Each one is a runnable Python file: edit and
rerun freely. This page is the complete catalog of what lives under
`examples/`, what each script does, where it writes, and which manual
page explains it. Status labels (production, preview, experimental,
gated) follow [Methods: theory and status](../docs/methods.md#how-to-read-the-status-labels).

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

## Requirements beyond the core install

* **ASE is a core dependency** of vibe-qc (it is listed in
  `[project.dependencies]`; `run_job(optimize=True)` drives semiempirical
  and MLIP optimisations through ASE calculators). The `[ase]` extra still
  exists as a compatibility alias for older instructions that say
  `pip install '.[ase]'`; it installs nothing the base install lacks.
* **matplotlib is not declared** anywhere in `pyproject.toml`, yet these
  scripts import it: every script under `plots/` except `bond-orbitals.py`,
  the band-structure and diagnostics inputs `periodic/input-h-chain-bands.py`,
  `periodic/input-si-bands.py`, `periodic/input-nacl-sto3g-dft.py`,
  `periodic/coop_cohp_h2_chain.py` and
  `periodic/aiccm2026dev_diamond_bonds_bands_compare.py`,
  `debug/scf_iteration_recorder.py`, the ASE scripts
  `ase_workflows/optimize-via-ase-bfgs.py`, `ase_workflows/md-water-nve.py`
  and `ase_workflows/nh3-neb-via-ase.py`, and
  `ase_compare/compare-basis-convergence.py` and
  `ase_compare/compare-nh3-neb-barrier.py`. Install it yourself
  (`pip install matplotlib`) before running those.
* Optional extras named per section: `[skala]`, `[mace]`, `[trexio]`,
  `[fetch]`, `[basisopt]`, the OpenTrustRegion backend, and PySCF or ORCA
  as external references for the comparison scripts.

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

Replace `<vibe-qc-checkout>` with the clone path. Scripts that read a
shared geometry file (`Molecule.from_xyz(HERE.parent / "h2o.xyz")`) expect
the `.xyz` one directory above the script, so copy `examples/h2o.xyz` and
friends alongside or recreate that layout.

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

`ModuleNotFoundError: No module named 'vibeqc'` means you used the
wrong Python; see [the running guide](../docs/running.md) for the
full story (venv activation, `OMP_NUM_THREADS`, output capture for
long runs).

## Where the output goes

There is no single rule; four conventions coexist. Check the script's
`output=` argument (or its docstring) before you run it. Many scripts,
mostly under `ase_compare/`, `ase_workflows/`, `basisset_dev/`, `embedding/`,
`plots/` and much of `semiempirical/` and `wavefunction/`, call library
functions rather than `run_job` and write nothing but what they print or a
single named report or figure.

1. **Next to the script** (the majority of `run_job` / `run_periodic_job`
   callers). The script defines `HERE = Path(__file__).parent` and passes
   `output=HERE / "<stem>"`, so `<stem>.out` and its siblings land beside
   the `.py` wherever you copied it. This covers nearly every
   `molecular/input-*.py`, the `periodic/input-*.py`, `gpw-*.py` and
   `gapw-*.py` inputs, the parity-grid scripts, `workflows/`, the
   `wavefunction/01_` and `02_casscf_*` scripts and
   `plots/water-dimer-d3-curve.py`.
2. **An explicit `--output-dir`** you must supply, chosen outside the
   checkout: `opentrustregion/`, `trexio/`, `qcschema/` and
   `iao_population_comparison.py`. They refuse to run without it.
3. **The current working directory.** These pass a bare stem, so their
   files land wherever you launched Python; run them from a scratch
   directory: the AICCM demos (`periodic/aiccm2026dev_*.py`),
   `periodic/coop_cohp_h2_chain.py`,
   `molecular/input-h2-tight-canonical-orth.py`,
   `semiempirical/30_gfn2_single_point.py`,
   `wavefunction/02_selected_ci_at_scale.py` and
   `vibe_view/showcase_natural_orbitals.py` (which also accepts a stem as
   its first argument). `run_job` without any `output=` would write
   `./output.*`; no cataloged script does that any more.
4. **A fixed location elsewhere.** `plots/*.py` write their figure into
   `docs/_static/plots/`; `plots/h2o-opt-trajectory-animation.py` runs
   `molecular/input-h2o-opt.py` as a subprocess (its files land in
   `molecular/`) and `plots/bond-orbitals.py` works under
   `/tmp/bond-orbitals` (or `$BOND_VIZ_WORKDIR`). `qvf_containers/build_*.py`
   write into `qvf_containers/runs/`, `vibe_view/showcase_formaldehyde.py`
   into `vibe_view/runs/h2co_showcase/`, `periodic/input-symmetry-demo.py`
   and `wavefunction/benchmark_all_methods.py` under `/tmp`, the `debug/`
   health checks under `debug/output/<scriptname>/`, and the QCSchema
   README uses `/tmp/vibeqc-qcschema-example` as its example directory.

The artifact family that `run_job` / `run_periodic_job` write for a stem
(a run of `molecular/input-h2o-rhf.py` on the 0.17.10 development build
produced exactly this list):

| File | Content |
|---|---|
| `{stem}.out` | plain-text log: banner, atom table, SCF trace, energy breakdown, orbital energies with HOMO/LUMO markers |
| `{stem}.system` | TOML manifest of host, build, options and the files written |
| `{stem}.bibtex`, `{stem}.references` | the auto-assembled citation bundle for the methods used ([citations](../docs/user_guide/citations.md)) |
| `{stem}.xyz` | final geometry |
| `{stem}.molden` | molecular orbitals for Jmol, Avogadro, Molden, IQmol |
| `{stem}.qvf` | the vibe-view archive (default on since v0.14; `output_qvf=False` disables it) |
| `{stem}.population.txt`, `{stem}.population.json` | Mulliken and Loewdin populations |
| `{stem}.traj` | ASE trajectory, geometry optimisations only (`ase gui {stem}.traj`) |
| `{stem}.hess`, `{stem}-*.cube` | ORCA-format normal modes and Gaussian cube files, where the script asks for them |
| `{stem}.POSCAR`, `{stem}.xsf`, `{stem}.cif` | periodic runs: structure files, plus a density `.xsf` when `write_density=True` |

Each `write_*` keyword on `run_job` / `run_periodic_job` (for example
`write_molden_file=False`, `citations=False`) switches one member off;
[output files](../docs/user_guide/output_files.md) lists them.

## Layout

```
examples/
├── quickstart.py                    first four calculations after installing
├── casscf_gradient.py               CASSCF analytic gradient diagnostics
├── embed_convergence.py             embedded-cluster convergence protocol (experimental)
├── iao_population_comparison.py     IAO charges against Knizia 2013 (needs --output-dir)
├── input-qvf-container-roundtrip.py pending-to-settled QVF container in one script
├── input-vqfetch-optimade-mgo.py    fetch MgO from OPTIMADE, emit and run
├── input-vqfetch-reference-h2o.py   fetch CCCBDB reference data for H2O
├── h2.xyz  h2o.xyz  h2o_dimer.xyz  h2o_trimer.xyz  h2s.xyz   shared geometries
├── molecular/        HF, DFT (incl. range-separated, double hybrids, SKALA), ROHF/ROKS,
│   │                 MP2 variants, CCSD(T), DLPNO, CASSCF/CASPT2, FCI, optimisation,
│   │                 vibrations, cubes, SCF-mode and initial-guess showcases
│   ├── solvation/    CPCM single point, optimisation and dielectric scan
│   └── mp2_benchmarks/  S22 and open-shell MP2 benchmark against ORCA (own README)
├── opentrustregion/  optional second-order orbital optimiser, READ restart, saddle following
├── qcschema/         QCSchema molecular JSON in, atomic results out
├── periodic/         1-D chains, 3-D crystals, 2-D slabs; Ewald-3D, GDF, GPW, GAPW, BIPOLE
│   │                 and AICCM routes; bands, DOS, COOP/COHP, symmetry, DFT+U, XSF/BXSF
│   └── <System>/     the GDF parity grid, one directory per crystal (see below)
├── periodic_pyscf/   PySCF twins of the parity grid (external reference, optional)
├── mlip/             MACE machine-learned potentials (optional [mace] extra)
├── qvf_containers/   pending-to-settled single-file job lifecycle
├── trexio/           wavefunction exchange, backend conversion, READ restarts ([trexio])
├── workflows/        NEB reaction paths, molecular and periodic
├── ase_compare/      cross-code validation against PySCF, ORCA, Psi4
│   └── cross_code_regression/  head-to-head basis scans and periodic checks (own README)
├── ase_workflows/    pure-ASE drivers (BFGS, Vibrations, NEB, MD) with vibe-qc as calculator
├── semiempirical/    MSINDO, DFTB0/SCC-DFTB, GFN2-xTB, PM6, OM1/OM2/OM3, SECCM
├── wavefunction/     CASSCF, CASPT2/NEVPT2, FCI, selected CI, DMRG, v2RDM, transcorrelated CI
├── embedding/        Green's-function surface embedding (experimental)
├── vibe_view/        showcase inputs whose .qvf exercises every viewer section (own README)
├── jk_builder/       driving the periodic Gamma SCF through the molecular JKBuilder seam
├── basisset_dev/     pob-paper calibration set and basis-development tooling (own README)
├── benchmarks/       eigensolver benchmark
├── crystal23_demo/   CRYSTAL23 demo .d12 decks mirrored for smoke tests (own README)
├── experimental/     the coulomb-method-zoo spike, historical (own README)
├── debug/            70 diagnostic scripts, grouped in debug/README.md
├── plots/            regenerators for the tutorial figures (own README)
└── regression/       parity test suite against PySCF and ORCA (own README)
```

## Choose an example by goal

| Goal | Start with | Continue with |
|---|---|---|
| Verify a new installation | [`quickstart.py`](quickstart.py) | [`molecular/input-h2o-rhf.py`](molecular/input-h2o-rhf.py) |
| Run molecular DFT and optimize | [`molecular/input-h2o-rks-pbe-d3-opt.py`](molecular/input-h2o-rks-pbe-d3-opt.py) | geometry-optimization section below |
| Treat an open-shell molecule | [`molecular/input-oh-radical.py`](molecular/input-oh-radical.py) | [`molecular/input-oh-ump2.py`](molecular/input-oh-ump2.py), [`molecular/input-oh-rohf.py`](molecular/input-oh-rohf.py) |
| Get a coupled-cluster number | [`molecular/input-h2o-ccsd-t.py`](molecular/input-h2o-ccsd-t.py) | [`molecular/input-h2o-dlpno-ccsd-t.py`](molecular/input-h2o-dlpno-ccsd-t.py) |
| Multireference | [`molecular/input-h2o-casscf-caspt2.py`](molecular/input-h2o-casscf-caspt2.py) | [`wavefunction/`](wavefunction/) |
| Exchange a QCSchema molecular job | [`qcschema/compare_h2.py`](qcschema/compare_h2.py) | [QCSchema tutorial](../docs/tutorial/qcschema_interchange.md), [HF derivatives](qcschema/hf_derivatives.py) |
| Compare IAO populations with literature | [`iao_population_comparison.py`](iao_population_comparison.py) (requires `--output-dir`) | [IAO numerical comparisons](../docs/user_guide/iao_validation.md) |
| Run Microsoft SKALA neural XC | [`molecular/input-h2-rks-skala.py`](molecular/input-h2-rks-skala.py) | [worked SKALA tutorial](../docs/tutorial/skala_neural_xc.md), [experimental periodic smoke run](periodic/input-h2-cell-rks-skala-gdf.py) |
| Compare direct and conventional SCF | [`molecular/input-h2o-rhf-direct.py`](molecular/input-h2o-rhf-direct.py) | [direct-SCF tutorial](../docs/tutorial/direct_scf_memory_tradeoff.md) |
| Compare orbital optimizers and follow an electronic saddle | [`opentrustregion/01_compare_scf.py`](opentrustregion/01_compare_scf.py) | [OpenTrustRegion examples](opentrustregion/README.md), [worked tutorials](../docs/tutorial/opentrustregion.md) |
| Start a periodic calculation | [`periodic/input-h-chain-uniform.py`](periodic/input-h-chain-uniform.py) | [`periodic/input-lih-pob-tzvp.py`](periodic/input-lih-pob-tzvp.py) |
| Pick a periodic J/K route | [`periodic/input-mgo-rocksalt-rhf.py`](periodic/input-mgo-rocksalt-rhf.py) (Gamma-only GDF) | GDF, GPW, GAPW, BIPOLE sections below and [periodic methods](../docs/user_guide/periodic_methods.md) |
| Fast semiempirical screening | [`semiempirical/01_water_dftb0.py`](semiempirical/01_water_dftb0.py) | [`semiempirical/10_msindo_validate.py`](semiempirical/10_msindo_validate.py), [semiempirical guide](../docs/user_guide/semiempirical.md) |
| Optimize or characterize through ASE | [`ase_workflows/optimize-via-ase-bfgs.py`](ase_workflows/optimize-via-ase-bfgs.py) | vibrations, NEB, and MD entries below |
| Run a multi-step reaction workflow | [`workflows/input-nh3-umbrella-neb.py`](workflows/input-nh3-umbrella-neb.py) | [NEB tutorial](../docs/tutorial/neb_reaction_path.md) |
| Produce or inspect QVF containers | [`qvf_containers/build_h2_job.py`](qvf_containers/build_h2_job.py) | [container tutorial](../docs/tutorial/qvf_job_containers.md) |
| Exchange TREXIO wavefunctions | [`trexio/molecular_restart.py`](trexio/molecular_restart.py) | [TREXIO examples](trexio/README.md), [molecular tutorial](../docs/tutorial/trexio_exchange.md), [CI and periodic tutorial](../docs/tutorial/trexio_correlated_periodic.md) |
| Validate against another program | [`ase_compare/compare-h2o-hf.py`](ase_compare/compare-h2o-hf.py) | cross-code validation section below |
| Look at results in 3-D | [`vibe_view/showcase_formaldehyde.py`](vibe_view/showcase_formaldehyde.py) | [vibe-view showcase README](vibe_view/README.md) |
| Fit a periodic equation of state | [`periodic/input-lih-eos.py`](periodic/input-lih-eos.py) | [equation of state](../docs/user_guide/equation_of_state.md) |
| Read a CIF or POSCAR into a calculation | [`periodic/input-nacl-from-cif.py`](periodic/input-nacl-from-cif.py) | [external structures](../docs/user_guide/external_structures.md) |
| Colour bands by atomic character | [`periodic/input-lih-chain-fat-bands.py`](periodic/input-lih-chain-fat-bands.py) | [band structure](../docs/tutorial/band_structure.md) |
| Try density mixing in a periodic SCF | [`periodic/input-mgo-density-mixing.py`](periodic/input-mgo-density-mixing.py) | [periodic SCF convergence](../docs/tutorial/periodic_scf_convergence.md) |
| Compare coupled-cluster variants, TDDFT or IRC | [`molecular/input-h2o-cc-hierarchy.py`](molecular/input-h2o-cc-hierarchy.py) | [`molecular/input-h2o-tddft-tda-nto.py`](molecular/input-h2o-tddft-tda-nto.py), [`molecular/input-nh3-irc.py`](molecular/input-nh3-irc.py) |

Each example should answer four questions before you modify it: which system
is represented, which method and basis are used, where files will be written,
and what numerical result or diagnostic defines success. The docstrings of the
introductory scripts state that success criterion explicitly. Change one of
those at a time and preserve the unmodified input beside any comparison run.

**Resource expectations** (order-of-magnitude guide, Apple M2, one to
four threads; a loaded machine or a debug build is slower):

| Example class | Peak RAM | Wall time |
|---|---|---|
| Molecular HF / DFT at STO-3G / 6-31G* on small systems | 70 to 250 MB | seconds |
| Geometry optimization, water dimer / trimer | ~150 MB | 5 to 60 s |
| Vibrational frequencies (`input-h2o-vibrations.py`, analytic RHF Hessian) | ~250 MB | seconds on H2O |
| ASE finite-difference Hessian (`ase_workflows/vibrations-via-ase-vibrations.py`) | ~250 MB | ~30 s on H2O, scales as 6N gradients |
| 1D periodic SCF (H-chain, STO-3G, k=8) | ~200 MB | a few seconds |
| 3D bulk SCF via EWALD_3D (LiH, STO-3G) | ~250 MB | 10 to 60 s |
| Madelung / point-charge Ewald | <50 MB | <0.05 s |
| NEB (NH3 inversion, 7 images, HF/STO-3G) | ~150 MB | ~30 s |
| CCSD(T) or DLPNO-CCSD(T) on H2O / cc-pVDZ | <1 GB | tens of seconds |
| pob-TZVP on a real solid with an 8x8x8 mesh (`MgO-rocksalt/*mp8*`) | GBs | minutes to an hour |

SKALA is omitted from this timing table until a supported Linux reference run
is published. Real evaluation requires Python 3.11 through 3.13 and the
optional `[skala]` extra, and its full-grid differentiable model needs more
memory than an ordinary semilocal DFT grid. Both source examples accept
`--dry-run`, which imports no PyTorch and downloads no checkpoint.

Use the examples as ballparks: pob-TZVP on a real solid bumps the
basis count by 5 to 10 times, which costs roughly $\mathcal{O}(N_{bf}^4)$ in
the Fock build. Multi-threaded scaling and tight-cell convergence
guidance live in [parallel execution](../docs/tutorial/parallel_execution.md)
and [periodic SCF convergence](../docs/tutorial/periodic_scf_convergence.md).

## First calculations: `quickstart.py`

[`quickstart.py`](quickstart.py) walks through molecular RHF on H2O /
6-31G*, UHF on the OH radical (spin lives on the `Molecule`, not on the
driver options), a periodic Ewald-3D SCF on bulk LiH and the Madelung
constant of NaCl, all in one file, and prints what each number means. Pair
it with [quickstart](../docs/quickstart.md) and
[molecular HF](../docs/tutorial/molecular_hf.md).

## QCSchema molecular interchange in `qcschema/`

`h2_input.json` defines a singlet H2 molecule at a 1.4011 bohr bond length
with an HF/STO-3G energy request. Run the scripts from the repository root:

```sh
mkdir -p /tmp/vibeqc-qcschema-example
.venv/bin/python examples/qcschema/compare_h2.py --output-dir /tmp/vibeqc-qcschema-example
.venv/bin/python examples/qcschema/hf_derivatives.py --output-dir /tmp/vibeqc-qcschema-example
```

`compare_h2.py` writes HF, MP2, and FCI atomic result JSON files and compares
their energies with a published Born-Oppenheimer value. `hf_derivatives.py`
writes gradient and Hessian atomic results. See the
[worked tutorial](../docs/tutorial/qcschema_interchange.md) for the physical
interpretation and JSON field layout.

## Round-trip QVF job containers in `qvf_containers/`

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
| [`qvf_containers/build_h2_job.py`](qvf_containers/build_h2_job.py) | Builds a pending molecular H2/RHF/STO-3G container under `qvf_containers/runs/` for the separate `vibeqc run` CLI step |
| [`qvf_containers/build_periodic_he_job.py`](qvf_containers/build_periodic_he_job.py) | Builds a tiny pending periodic He/SCC-DFTB Gamma-only container |
| [`qvf_containers/inspect_job.py`](qvf_containers/inspect_job.py) | Validates a pending or settled container and prints its spec, lifecycle status, section kinds, and run history |

See the
[complete job-container tutorial](../docs/tutorial/qvf_job_containers.md) for
local execution, first-class vq submit/fetch, terminal viewing, graphical
viewing, failure semantics, and deliberate re-runs, and the
[container README](qvf_containers/README.md).

## Single-point energies in `molecular/`

| Script | What it does | Manual page |
| --- | --- | --- |
| [`molecular/input-h2o-rhf.py`](molecular/input-h2o-rhf.py) | Closed-shell Hartree-Fock on H2O / 6-31G* | [molecular HF](../docs/tutorial/molecular_hf.md) |
| [`molecular/input-h2o-dft.py`](molecular/input-h2o-dft.py) | Kohn-Sham DFT with the PBE functional, 6-31G*, energy components and timings | [molecular DFT](../docs/tutorial/molecular_dft.md) |
| [`molecular/input-h2-rks-skala.py`](molecular/input-h2-rks-skala.py) | H2 / def2-SVP fixed-geometry RKS with Microsoft SKALA-1.1 (experimental); supports a model-free `--dry-run` | [SKALA](../docs/tutorial/skala_neural_xc.md) |
| [`molecular/input-h2o-rks-pbe-d3-opt.py`](molecular/input-h2o-rks-pbe-d3-opt.py) | RKS-PBE / 6-31G* + D3(BJ) dispersion + geometry optimisation, the homepage example | [dispersion](../docs/tutorial/dispersion.md) |
| [`molecular/input-oh-radical.py`](molecular/input-oh-radical.py) | Open-shell UHF on the OH radical, 6-31G*, SAD guess | [open shell](../docs/tutorial/open_shell.md) |
| [`molecular/input-oh-ump2.py`](molecular/input-oh-ump2.py) | UMP2 post-SCF correlation on the OH radical, 6-31G*, with the spin-contamination diagnostic | [MP2 and double hybrids](../docs/user_guide/mp2_and_double_hybrids.md) |
| [`molecular/input-h2o-cube.py`](molecular/input-h2o-cube.py) | H2O density + frontier MOs as Gaussian cube files for VMD / Avogadro | [orbital visualisation](../docs/tutorial/orbital_visualization.md) |
| [`molecular/input-h2o-vibrations.py`](molecular/input-h2o-vibrations.py) | RHF + analytic CPHF Hessian + harmonic frequencies; writes the `.hess` normal-modes file and the Molden `[FREQ]` blocks | [vibrational frequencies](../docs/tutorial/vibrational_frequencies.md) |
| [`molecular/input-h2o-range-separated.py`](molecular/input-h2o-range-separated.py) | wB97X, CAM-B3LYP, HSE06 and PBE0 on H2O / def2-SVP: total energies, HOMO/LUMO levels, and the finite-difference fallback for range-separated optimisations (GitLab #571) | [tutorial](../docs/tutorial/range_separated_wb97x.md) |
| [`molecular/input-h2o-meta-gga.py`](molecular/input-h2o-meta-gga.py) | TPSS, SCAN and r2SCAN against PBE on H2O / def2-SVP: energies and Kohn-Sham gaps | [tutorial](../docs/tutorial/functional_comparison.md) |
| [`molecular/input-h2o-dimer-3c.py`](molecular/input-h2o-dimer-3c.py) | HF-3c and PBEh-3c binding energy of the water dimer, with the D3(BJ), gCP and SRB contributions rebuilt from `composite_correction_terms` and `compute_gcp` | [user guide](../docs/user_guide/composites.md) |
| [`molecular/input-hi-ecp.py`](molecular/input-hi-ecp.py) | HI / def2-SVP with the def2 effective core potential attached from the bundled sidecar; `ecp_requirements` and the `ecp_*` result fields | [user guide](../docs/user_guide/ecp.md) |
| [`molecular/input-h2o-thermochemistry.py`](molecular/input-h2o-thermochemistry.py) | RHF / 6-31G\* finite-difference Hessian, harmonic frequencies and RRHO thermochemistry at 298.15 K and 500 K via `compute_thermochemistry` | [tutorial](../docs/tutorial/thermodynamics.md) |

## Geometry optimization in `molecular/`

| Script | What it does | Manual page |
| --- | --- | --- |
| [`molecular/input-h2o-opt.py`](molecular/input-h2o-opt.py) | Relaxation of H2O at HF/6-31G* (via ASE BFGSLineSearch) | [geometry optimisation](../docs/tutorial/geometry_optimization.md) |
| [`molecular/input-h2o-dimer-opt.py`](molecular/input-h2o-dimer-opt.py) | Water dimer (H-bonded), R(OO) about 2.98 A at HF/6-31G* | [post-SCF properties](../docs/tutorial/post_scf_properties.md) |
| [`molecular/input-h2o-trimer-opt.py`](molecular/input-h2o-trimer-opt.py) | Cyclic water trimer, asymmetric "uud" ring minimum | [geometry optimisation](../docs/tutorial/geometry_optimization.md) |

H-bonded clusters are weakly bound and their potential surfaces are flat,
so the trimer example stops at a loose ``fmax=0.2`` eV/A (about
0.004 Ha/bohr) to keep the run short. That is a choice of stopping
criterion, not a precision limit: analytic HF forces at the default SCF
tolerance are accurate to a few 1e-8 Ha/bohr. Lower ``fmax`` (and, for
very tight thresholds, ``conv_tol_grad`` in the RHF options) when you need
the minimum more precisely.

### Optimizer backends

| Script | What it does | Manual page |
| --- | --- | --- |
| [`molecular/input-h2o-brent-opt.py`](molecular/input-h2o-brent-opt.py) | H2O optimisation with `optimizer_backend="brent"`: steepest descent + Brent line search, never uphill, no Hessian model; QVF archive of the path | [geometry optimisation guide](../docs/user_guide/geometry_optimization.md) |
| [`molecular/input-h2o-pbe-d3-brent-opt.py`](molecular/input-h2o-pbe-d3-brent-opt.py) | The homepage RKS-PBE-D3(BJ) optimisation redone on the Brent backend, to compare with the ASE BFGS run | [geometry optimisation guide](../docs/user_guide/geometry_optimization.md) |

`run_job(optimize=True)` with a post-SCF method (MP2, CCSD, DLPNO) relaxes
the mean-field reference surface; FCI, NEVPT2, CASPT2 and MRCI need
`optimizer_backend="native"` or `"brent"` for a finite-difference
optimisation on the correlated surface. CASSCF has its analytic gradient.

## MP2 and double hybrids in `molecular/`

| Script | What it does | Manual page |
| --- | --- | --- |
| [`molecular/input-h2o-scs-mp2.py`](molecular/input-h2o-scs-mp2.py) | Spin-component-scaled MP2 (SCS-MP2) on H2O through RI-MP2 | [MP2 and double hybrids](../docs/user_guide/mp2_and_double_hybrids.md) |
| [`molecular/input-h2o-b2plyp.py`](molecular/input-h2o-b2plyp.py) | B2PLYP double hybrid: hybrid SCF + scaled RI-MP2 in one call | [double hybrid B2PLYP](../docs/tutorial/double_hybrid_b2plyp.md) |
| [`molecular/input-h2o-pwpb95.py`](molecular/input-h2o-pwpb95.py) | PWPB95, the meta-GGA and spin-opposite-scaled double hybrid | [double hybrid B2PLYP](../docs/tutorial/double_hybrid_b2plyp.md) |
| [`molecular/input-h2o-revdsd-pbep86.py`](molecular/input-h2o-revdsd-pbep86.py) | revDSD-PBEP86 with its asymmetric os/ss MP2 scaling | [double hybrid B2PLYP](../docs/tutorial/double_hybrid_b2plyp.md) |
| [`molecular/input-h2o-ri-mp2-residual.py`](molecular/input-h2o-ri-mp2-residual.py) | RI-MP2 on H2O / cc-pVTZ with the auxiliary-fit residual diagnostic across two aux bases | [RI-MP2 aux verification](../docs/tutorial/ri_mp2_aux_verification.md) |
| [`molecular/input-h2-fci.py`](molecular/input-h2-fci.py) | Exact Full CI against Hartree-Fock: the basis-set correlation energy of H2 / 6-31G | [non-HF solvers](../docs/tutorial/non_hf_solvers.md) |

The larger MP2 validation set (S22 and an open-shell complement against
ORCA 6.1.1) lives in [`molecular/mp2_benchmarks/`](molecular/mp2_benchmarks/README.md).

## Coupled cluster and local correlation in `molecular/`

| Script | What it does | Manual page |
| --- | --- | --- |
| [`molecular/input-h2o-ccsd-t.py`](molecular/input-h2o-ccsd-t.py) | Canonical DF-CCSD(T) on H2O / cc-pVDZ through `run_job(method="ccsd(t)")`; reproduces ORCA 6.1.1 to about 1 mHa (the density-fitting error) | [coupled cluster](../docs/tutorial/coupled_cluster_ccsd_t.md), [CCSD guide](../docs/user_guide/ccsd.md) |
| [`molecular/input-ch3-uccsd-t.py`](molecular/input-ch3-uccsd-t.py) | Open-shell UCCSD(T) on the methyl radical, cc-pVDZ, on an automatic UHF reference; `ccsd_reference="rohf"` selects the spin-pure alternative | [coupled cluster](../docs/tutorial/coupled_cluster_ccsd_t.md) |
| [`molecular/input-h2o-dlpno-mp2.py`](molecular/input-h2o-dlpno-mp2.py) | DLPNO-MP2 on H2O / cc-pVDZ (Pinski 2015): Foster-Boys localisation, PNO compression, measured recovery against RI-MP2 | [DLPNO local correlation](../docs/tutorial/dlpno_local_correlation.md) |
| [`molecular/input-h2o-dlpno-ccsd.py`](molecular/input-h2o-dlpno-ccsd.py) | DLPNO-CCSD on H2O / cc-pVDZ: per-pair local solver in PNO bases with the extended-PAO residual | [DLPNO local correlation](../docs/tutorial/dlpno_local_correlation.md) |
| [`molecular/input-h2o-dlpno-ccsd-t.py`](molecular/input-h2o-dlpno-ccsd-t.py) | DLPNO-CCSD(T) on H2O / cc-pVDZ: the local solver plus DLPNO-(T1) triples (Guo, Riplinger 2018) | [DLPNO guide](../docs/user_guide/dlpno_mp2.md) |
| [`periodic/input-lih-dlpno-ccsd-t.py`](periodic/input-lih-dlpno-ccsd-t.py) | DLPNO-CCSD(T) on rock-salt LiH / STO-3G at Gamma and on a (2,2,2) mesh folded to the BvK supercell, energies per cell (preview route) | [local correlation on crystals](../docs/user_guide/periodic_local_correlation.md), [tutorial](../docs/tutorial/periodic_dlpno_ccsd_t.md) |
| [`regression/parity_periodic_dlpno_vs_pyscf.py`](regression/parity_periodic_dlpno_vs_pyscf.py) | External reference of the periodic DLPNO route: full-domain DLPNO-CCSD(T) against PySCF KRCCSD(T) on the H2 crystal or a regression-suite crystal, with both PySCF Fock conventions (PySCF runs in a child process) | [local correlation on crystals](../docs/user_guide/periodic_local_correlation.md#evidence) |
| [`molecular/benchmark-dlpno-ccsd-t.py`](molecular/benchmark-dlpno-ccsd-t.py) | Retained validation evidence: DLPNO-CCSD(T) against canonical CCSD(T) on H2O, NH3, CH4, HF, CO at def2-SVP with an explicitly pinned legacy threshold recipe | [DLPNO local correlation](../docs/tutorial/dlpno_local_correlation.md) |
| [`molecular/input-h2o-cc-hierarchy.py`](molecular/input-h2o-cc-hierarchy.py) | CCD, LCCD, CEPA(0/1), QCISD(T), CC2, CCSD(T), BCCD(T), CC3 and CCSDT on H2O / STO-3G, tabulated against CCSD(T); CC2 is the outlier | [user guide](../docs/user_guide/ccsd.md), [tutorial](../docs/tutorial/coupled_cluster_family.md) |
| [`molecular/input-h2o-cisd-mrci.py`](molecular/input-h2o-cisd-mrci.py) | CISD, MRCI on a (4e,4o) CAS and FCI on H2O / STO-3G, plus a two-H2 size-extensivity test that separates CISD's truncation error from its size-extensivity error | [user guide](../docs/user_guide/non_hf_solvers.md) |
| [`molecular/input-h2o-ovgf.py`](molecular/input-h2o-ovgf.py) | Second-order Green's function (GF2) quasiparticle IPs and pole strengths against Koopmans on H2O / cc-pVDZ, plus the renormalised third-order HOMO IP printed in the `.out` | [user guide](../docs/user_guide/ovgf.md), [tutorial](../docs/tutorial/ovgf_ionization.md) |
| [`molecular/input-h2o-tddft-tda-nto.py`](molecular/input-h2o-tddft-tda-nto.py) | TDA and full Casida TD-DFT (PBE0 / def2-SVP) with natural transition orbitals and a UV/Vis spectrum; tables read back from the `.out`; oscillator strengths are currently over-scaled, see the TDDFT guide | [tutorial](../docs/tutorial/excited_states_tddft.md) |

## Multireference and determinant solvers

CASSCF, its PT2 partners and the determinant solvers are reachable through
`run_job(method=...)`; the small-active-space references (FCI, selected CI,
v2RDM) are production, transcorrelated CI is a preview, DMRG is an
experimental teaching implementation limited to six spatial orbitals, and
SC-CASPT2 stays gated behind `VIBEQC_EXPERIMENTAL_MULTIREF=1`. Read
[CASSCF and multireference](../docs/tutorial/casscf_multireference.md),
[non-HF solvers](../docs/tutorial/non_hf_solvers.md) and the
[solver guide](../docs/user_guide/non_hf_solvers.md).

| Script | What it does |
| --- | --- |
| [`molecular/input-h2o-casscf-caspt2.py`](molecular/input-h2o-casscf-caspt2.py) | CASSCF(4,4) reference + internally contracted CASPT2 on H2O; `casscf_options=` switches the PT2 reference from CASCI-on-HF to the optimised orbitals |
| [`casscf_gradient.py`](casscf_gradient.py) | The CASSCF analytic nuclear gradient (the production force) with its components next to the in-repo finite-difference oracle, H2 / 6-31G |
| [`wavefunction/01_casscf_h2o.py`](wavefunction/01_casscf_h2o.py) | CASCI, CASSCF (single-state and state-averaged), NEVPT2 and CASPT2 on H2O / STO-3G CAS(2,2) and CAS(4,4) |
| [`wavefunction/02_casscf_gradient.py`](wavefunction/02_casscf_gradient.py) | CASSCF(4,4) analytic-gradient diagnostic on H2O / STO-3G: components and the translational-invariance residual |
| [`wavefunction/03_casscf_geometry_optimization.py`](wavefunction/03_casscf_geometry_optimization.py) | Relaxes stretched H2 / 6-31G on the CAS(2,2) surface by central finite differences of the full CASSCF energy |
| [`wavefunction/02_selected_ci_at_scale.py`](wavefunction/02_selected_ci_at_scale.py) | Selected CI (multi-root CIPSI) past the dense-determinant wall: CASCI, CASSCF and PT2 on active spaces the dense engine cannot hold |
| [`wavefunction/fci_h2.py`](wavefunction/fci_h2.py) | Full CI on H2 / STO-3G, the two-determinant textbook case |
| [`wavefunction/selected_ci_h2.py`](wavefunction/selected_ci_h2.py) | Selected-CI convergence study on H2 / STO-3G against RHF and FCI |
| [`wavefunction/selected_ci_h4.py`](wavefunction/selected_ci_h4.py) | Selected CI on a stretched H4 square (R = 2.0 bohr), a multireference system |
| [`wavefunction/lih_selected_ci.py`](wavefunction/lih_selected_ci.py) | Selected CI on LiH / STO-3G: convergence to FCI, leading determinants, multireference character |
| [`wavefunction/h2_dissociation.py`](wavefunction/h2_dissociation.py) | H2 dissociation curve, RHF against selected CI against FCI, 0.6 to 6.0 bohr |
| [`wavefunction/n2_dissociation_active_space.py`](wavefunction/n2_dissociation_active_space.py) | N2 triple-bond breaking, where a single determinant fails and an active space is needed |
| [`wavefunction/benchmark_h2o_fci.py`](wavefunction/benchmark_h2o_fci.py) | Correlation-energy benchmark on H2O / STO-3G: RHF, selected CI, FCI (441 determinants) |
| [`wavefunction/benchmark_h4_linear.py`](wavefunction/benchmark_h4_linear.py) | Linear H4, the minimal strong-correlation model, compressed against stretched |
| [`wavefunction/benchmark_bh_dissociation.py`](wavefunction/benchmark_bh_dissociation.py) | BH dissociation, the ionic-to-covalent transition of a polar bond |
| [`wavefunction/benchmark_ch2_st.py`](wavefunction/benchmark_ch2_st.py) | Methylene singlet-triplet gap with FCI and selected CI |
| [`wavefunction/benchmark_oh_radical.py`](wavefunction/benchmark_oh_radical.py) | Open-shell correlation benchmark on OH: correlation energy, spin densities, UHF contamination |
| [`wavefunction/benchmark_all_methods.py`](wavefunction/benchmark_all_methods.py) | Side-by-side energy and timing table of every `run_job` solver on H2O / STO-3G: HF, KS-DFT, FCI, selected CI, DMRG, v2RDM, transcorrelated CI |
| [`wavefunction/benchmark_timing.py`](wavefunction/benchmark_timing.py) | Wall time of selected CI against FCI across active-space sizes on H2O / STO-3G |
| [`wavefunction/dmrg_h2.py`](wavefunction/dmrg_h2.py) | Two-site DMRG with bond-dimension scheduling on H2 / STO-3G (experimental teaching solver) |
| [`wavefunction/tc_dmrg_h2.py`](wavefunction/tc_dmrg_h2.py) | Transcorrelated Hamiltonian + DMRG on H2 / STO-3G, scanning the Gaussian correlator strength |
| [`wavefunction/transcorrelated_h2.py`](wavefunction/transcorrelated_h2.py) | Transcorrelated Hamiltonian + selected CI on H2 / STO-3G (preview) |
| [`wavefunction/v2rdm_h2.py`](wavefunction/v2rdm_h2.py) | Variational 2-RDM with PQG constraints on H2 / STO-3G, trace and PSD residuals |
| [`wavefunction/non_hf_orbitals.py`](wavefunction/non_hf_orbitals.py) | The solvers do not need HF orbitals: canonical HF, Loewdin-orthogonalised AOs and random orthonormal orbitals compared |

## More DFT in `molecular/`

| Script | What it does | Manual page |
| --- | --- | --- |
| [`molecular/input-h2o-wb97m-v.py`](molecular/input-h2o-wb97m-v.py) | wB97M-V on H2O / cc-pVDZ: range-separated meta-GGA hybrid with self-consistent VV10 | [VV10 and modern functionals](../docs/tutorial/vv10_modern_functionals.md) |
| [`molecular/input-h2o-grid-convergence.py`](molecular/input-h2o-grid-convergence.py) | Water / def2-SVP, PBE and r2SCAN on the named grid levels, Lebedev orders 23 to 95 and 50 to 200 radial shells; energies against the largest grid | [DFT grid convergence](../docs/tutorial/dft_grid_convergence.md) |
| [`molecular/input-h2o-pwpb95.py`](molecular/input-h2o-pwpb95.py), [`input-h2o-revdsd-pbep86.py`](molecular/input-h2o-revdsd-pbep86.py) | The two double hybrids beyond B2PLYP (listed above) | [double hybrid B2PLYP](../docs/tutorial/double_hybrid_b2plyp.md) |
| [`molecular/input-ch3-roks.py`](molecular/input-ch3-roks.py) | ROKS with PBE on the methyl radical, 6-31G*, spin-pure S^2 = 0.75 | [ROHF and ROKS](../docs/user_guide/rohf.md) |
| [`molecular/input-oh-rohf.py`](molecular/input-oh-rohf.py) | ROHF on the OH radical, 6-31G*, one spin-restricted orbital set | [ROHF and ROKS](../docs/user_guide/rohf.md) |
| [`molecular/define_functional_demo.py`](molecular/define_functional_demo.py) | `vq.define_functional`: a PW1PW clone defined at runtime and checked against the built-in on H2O / def2-SVP | [functionals](../docs/user_guide/functionals.md) |
| [`molecular/d4_pilot_c6.py`](molecular/d4_pilot_c6.py) | Native D4 pilot: RHF, imaginary-frequency CPHF polarisabilities and the Casimir-Polder integral give molecular C6 coefficients compared with DOSD literature values | [dispersion](../docs/tutorial/dispersion.md) |
| [`molecular/optimize_basis_h2o_pbe.py`](molecular/optimize_basis_h2o_pbe.py) | Re-optimise a molecular basis at the DFT level with the analytic BDIIS optimiser (`vibeqc.basis_optimization.recipes.molecular`) | [basis optimisation](../docs/user_guide/basis_optimization.md) |

Range-separated hybrids and VV10 are production for energies; their
analytic gradients omit terms, so optimisers fall back to finite differences.

## Properties and analysis in `molecular/`

| Script | What it does | Manual page |
| --- | --- | --- |
| [`molecular/input-h2o-population-analysis.py`](molecular/input-h2o-population-analysis.py) | Mulliken, Loewdin, Hirshfeld and IAO charges, Mayer / Wiberg / IAO-Wiberg bond orders, dipole and CPHF polarizability from one RHF / 6-31G\* run | [tutorial](../docs/tutorial/post_scf_properties.md) |
| [`molecular/input-water-dimer-eda.py`](molecular/input-water-dimer-eda.py) | Water-dimer counterpoise interaction energy (Z = 0 ghost atoms) and its LMO-EDA and Kitaura-Morokuma partition (electrostatic, exchange, Pauli repulsion, polarisation plus charge transfer) through `vibeqc.eda`, which is preview and closed-shell RHF only | [user guide](../docs/user_guide/bond_analysis.md), [tutorial](../docs/tutorial/eda_and_entanglement.md) |
| [`molecular/input-h2o-qtaim.py`](molecular/input-h2o-qtaim.py) | QTAIM critical points, densities, Laplacians and `atom_pair` connectivity of RHF / 6-31G\* water through `run_job(qtaim=True)` and `qtaim_analysis` | [user guide](../docs/user_guide/qtaim.md) |

## Reaction paths and dynamics in `molecular/`

| Script | What it does | Manual page |
| --- | --- | --- |
| [`molecular/input-nh3-irc.py`](molecular/input-nh3-irc.py) | Intrinsic reaction coordinate from the planar NH3 saddle at RHF / STO-3G, both directions down to the two mirror-image pyramids | [user guide](../docs/user_guide/irc_and_dimer.md), [tutorial](../docs/tutorial/irc_from_transition_state.md) |
| [`molecular/input-h2o2-relaxed-scan.py`](molecular/input-h2o2-relaxed-scan.py) | HOOH torsion scan (rigid for a four-atom dihedral, and the script says why) plus a relaxed O-O bond scan with `relaxed_scan` | [tutorial](../docs/tutorial/relaxed_pes_scan.md) |
| [`molecular/input-h2o-md-nvt.py`](molecular/input-h2o-md-nvt.py) | 20 steps of Nose-Hoover NVT Born-Oppenheimer MD on RHF / STO-3G water through `vibeqc.md.run_nvt`; temperature, energies and the conserved-quantity drift | [tutorial](../docs/tutorial/molecular_dynamics.md) |

## Solvation in `molecular/solvation/`

CPCM/COSMO is production; COSMO-RS is experimental. The CPCM scripts pair with
[solvation of water](../docs/tutorial/solvation_water.md) and the
[solvation guide](../docs/user_guide/solvation.md); the directory has its own
[README](molecular/solvation/README.md).

| Script | What it does |
| --- | --- |
| [`molecular/solvation/input-ch2o-water-single-point.py`](molecular/solvation/input-ch2o-water-single-point.py) | Formaldehyde gas-phase against CPCM water single points: polarisation energy and total solvation free energy |
| [`molecular/solvation/input-ch2o-water-opt.py`](molecular/solvation/input-ch2o-water-opt.py) | Geometry optimisation in gas and in water with the analytic CPCM gradient; reports the C=O elongation (about 0.7 pm) |
| [`molecular/solvation/input-ch2o-dielectric-scan.py`](molecular/solvation/input-ch2o-dielectric-scan.py) | Loops over every bundled solvent preset and shows the conductor-screening saturation of E_solv with the dielectric constant |
| [`molecular/solvation/input-methanol-cosmors.py`](molecular/solvation/input-methanol-cosmors.py) | COSMO-RS solvation free energy of methanol in water with the solvent given by name; the solvent surface is computed once and cached (experimental) |

## SCF infrastructure showcases in `molecular/`

| Script | What it does | Manual page |
| --- | --- | --- |
| [`molecular/input-h2-tight-canonical-orth.py`](molecular/input-h2-tight-canonical-orth.py) | Tight H2 with aug-cc-pVTZ: pre-flight linear-dependence diagnostic + canonical-orthogonalisation threshold sweep | [linear dependence](../docs/user_guide/linear_dependence.md) |
| [`molecular/input-progress-demo.py`](molecular/input-progress-demo.py) | Live SCF progress logging via the ProgressLogger surface | [logging](../docs/user_guide/logging.md) |
| [`molecular/input-h2o-initial-guess-comparison.py`](molecular/input-h2o-initial-guess-comparison.py) | HCORE / SAD / SAP / AUTO initial-guess comparison on H2O and OH | [initial guess walkthrough](../docs/tutorial/initial_guess_walkthrough.md) |
| [`molecular/input-h2o-rhf-direct.py`](molecular/input-h2o-rhf-direct.py) | Direct (integral-on-the-fly) SCF Fock-build mode | [direct SCF](../docs/tutorial/direct_scf_memory_tradeoff.md) |
| [`molecular/input-h2o-rhf-conventional.py`](molecular/input-h2o-rhf-conventional.py) | Conventional (in-core 4-index ERI) SCF Fock-build mode | [SCF modes](../docs/user_guide/scf_modes.md) |
| [`molecular/input-glycine-rhf-direct.py`](molecular/input-glycine-rhf-direct.py) | Explicit DIRECT SCF on glycine / def2-SVP (95 BF), avoiding the 621.42 MiB dense ERI tensor | [SCF modes](../docs/user_guide/scf_modes.md) |
| [`molecular/input-h2o-stretched-scf-accelerators.py`](molecular/input-h2o-stretched-scf-accelerators.py) | DIIS, EDIIS+DIIS, ADIIS+DIIS and KDIIS on doubly stretched water at UHF / 6-31G, with the broken-symmetry UHF-versus-RHF gap: iteration counts and final energies | [tutorial](../docs/tutorial/ediis_diis_hybrid.md) |
| [`molecular/input-glycine-rijcosx.py`](molecular/input-glycine-rijcosx.py) | Glycine B3LYP/def2-SVP with the conventional four-index Fock build against RIJCOSX (`density_fit=True, cosx=True`, auxiliary basis autodetected): both energies, their difference in mHa and the wall times | [RIJCOSX on glycine](../docs/tutorial/rijcosx_glycine.md), [density fitting](../docs/user_guide/density_fitting.md) |
| [`molecular/input-h2o-integral-engines.py`](molecular/input-h2o-integral-engines.py) | Water / cc-pVDZ RHF and PBE on libint and on the experimental MANTA engine with exact, RI, RI-JK and BUPO Coulomb, printing the executed provenance | [integral engines](../docs/user_guide/integral_engines.md) |
| [`tutorials/unified_operators.py`](tutorials/unified_operators.py) | The shared Hermitian operator contract: cold and warm Davidson, complex block callbacks, a factorised AO operator with an overlap metric, diagnostics and nearest-root selection; writes no files | [eigensolver tutorial](../docs/tutorial/eigensolver_operators.md) |

## Semiempirical methods in `semiempirical/`

Status per method: MSINDO is production within its documented scope (INDO H
to Xe; NDDO H, Li to F, Na to Cl); DFTB0 and SCC-DFTB are a preview
(screening and preoptimisation with in-house parameters, no DFTB+ parity,
absolute energies not claimable while the repulsive is a placeholder);
GFN2-xTB and PM6 are experimental (`GFN2ExperimentalWarning`); OM1, OM2 and
OM3 are experimental and a retirement target; the semiempirical cyclic
cluster model (SECCM) is experimental overall, with the MSINDO route the
most mature. Manual pages: [MSINDO](../docs/tutorial/msindo.md) and its
[guide](../docs/user_guide/msindo.md), [DFTB](../docs/tutorial/semiempirical_dftb.md),
[PM6 and GFN2](../docs/tutorial/pm6_and_gfn2.md), the
[GFN2-xTB workshop](../docs/tutorial/gfn2_xtb_workshop.md),
[periodic semiempirical and validation](../docs/tutorial/semiempirical_periodic_and_validation.md),
[cyclic cluster model](../docs/tutorial/cyclic_cluster_model.md) and the
[semiempirical guide](../docs/user_guide/semiempirical.md). Scripts that
write a Markdown report put it next to themselves.

### MSINDO (production within scope) and SECCM (experimental)

| Script | What it does |
| --- | --- |
| [`semiempirical/10_msindo_validate.py`](semiempirical/10_msindo_validate.py) | vibe-qc's independent MSINDO re-implementation against reference MSINDO numbers |
| [`semiempirical/16_comparison_msindo.py`](semiempirical/16_comparison_msindo.py) | MSINDO on H2O and H2S (the d-shell test), part of the three-tier semiempirical/MLIP comparison |
| [`semiempirical/11_msindo_ccm.py`](semiempirical/11_msindo_ccm.py) | Periodic MSINDO via the cyclic cluster model: bulk, surface, adsorption (SECCM, experimental) |
| [`semiempirical/22_msindo_neb.py`](semiempirical/22_msindo_neb.py) | NH3 umbrella inversion NEB with MSINDO energies and gradients |
| [`semiempirical/23_msindo_cosmo.py`](semiempirical/23_msindo_cosmo.py) | MSINDO COSMO implicit solvation energies in water |
| [`semiempirical/24_msindo_md.py`](semiempirical/24_msindo_md.py) | Born-Oppenheimer MD on the MSINDO surface (H2O) |
| [`semiempirical/25_msindo_metadynamics.py`](semiempirical/25_msindo_metadynamics.py) | Well-tempered metadynamics: an analytic model first, then MSINDO |
| [`semiempirical/26_msindo_conical.py`](semiempirical/26_msindo_conical.py) | CIS excited-state gradients and a conical-intersection search on MSINDO (H2O) |
| [`semiempirical/27_msindo_cisd.py`](semiempirical/27_msindo_cisd.py) | Variational CISD over the INDO reference through the generic CI (H2, H2O) |
| [`semiempirical/38_seccm_msindo_h_chain.py`](semiempirical/38_seccm_msindo_h_chain.py) | Experimental SECCM through `run_job(method="seccm", ccm_options=CCMOptions(...))` on a 1-D LiH cyclic cluster of 2 to 5 units: energy per LiH unit against cluster size, the Wigner-Seitz topology diagnostics, and the Madelung embedding energy (`madelung=True` against `False`) See [cyclic cluster model](../docs/tutorial/cyclic_cluster_model.md). |

### DFTB0 and SCC-DFTB (preview)

| Script | What it does |
| --- | --- |
| [`semiempirical/01_water_dftb0.py`](semiempirical/01_water_dftb0.py) | DFTB0 energy, analytic gradient and translational invariance on water |
| [`semiempirical/02_formic_acid_scc.py`](semiempirical/02_formic_acid_scc.py) | SCC-DFTB with charge mixing, Mulliken charges and the energy decomposition on formic acid |
| [`semiempirical/03_carbon_chain_periodic.py`](semiempirical/03_carbon_chain_periodic.py) | Periodic DFTB0 on a 1-D carbon chain: Gamma, Monkhorst-Pack mesh, band path |
| [`semiempirical/04_optimize_and_preoptimize.py`](semiempirical/04_optimize_and_preoptimize.py) | DFTB0 optimisation via ASE, SCC comparison, D3(BJ), preoptimisation handoff |
| [`semiempirical/05_production_workflow.py`](semiempirical/05_production_workflow.py) | Spline-fitted repulsives, TOML parameter I/O, DFTB0 against SCC against SCC+D3, optimisation |
| [`semiempirical/09_dftb_preoptimize.py`](semiempirical/09_dftb_preoptimize.py) | The recommended pattern: DFTB0 preoptimisation, then PBE/def2-SVP refinement |
| [`semiempirical/13_comparison_dftb.py`](semiempirical/13_comparison_dftb.py) | DFTB0 and SCC-DFTB on H2O for the three-tier comparison |
| [`semiempirical/21_frequencies.py`](semiempirical/21_frequencies.py) | DFTB0 harmonic frequencies, ZPVE and 298 K thermochemistry from finite differences of gradients |
| [`semiempirical/fit_repulsive.py`](semiempirical/fit_repulsive.py) | Tool: fit a cubic-spline repulsive from DFT reference energies along a bond stretch |

### GFN2-xTB (experimental)

| Script | What it does |
| --- | --- |
| [`semiempirical/30_gfn2_single_point.py`](semiempirical/30_gfn2_single_point.py) | Parameter auto-fetch, single points on water, methane, ammonia, CO2, energy components, charges |
| [`semiempirical/31_gfn2_spin_polarization.py`](semiempirical/31_gfn2_spin_polarization.py) | Unrestricted GFN2 on OH, O2 triplet and CH3 with an S^2 estimate |
| [`semiempirical/32_gfn2_geometry_optimization.py`](semiempirical/32_gfn2_geometry_optimization.py) | `run_job(optimize=True)`, the native `run_geomopt` + `SemiempiricalProvider` route, and the analytic-against-FD gradient check |
| [`semiempirical/33_gfn2_frequencies.py`](semiempirical/33_gfn2_frequencies.py) | Finite-difference Hessian, frequencies, ZPVE and thermochemistry |
| [`semiempirical/34_gfn2_pes_scan.py`](semiempirical/34_gfn2_pes_scan.py) | Bond, angle and 2-D PES scans of water with minimum and curvature |
| [`semiempirical/35_gfn2_molecular_dynamics.py`](semiempirical/35_gfn2_molecular_dynamics.py) | NVE and Berendsen NVT dynamics through an ASE calculator wrapper |
| [`semiempirical/36_gfn2_periodic.py`](semiempirical/36_gfn2_periodic.py) | Periodic GFN2 on graphene and h-BN, stress tensor, molecular-limit check, fail-closed multi-k |
| [`semiempirical/37_gfn2_convergence_tuning.py`](semiempirical/37_gfn2_convergence_tuning.py) | Mixers, damping and electronic temperature for the SCC loop |
| [`semiempirical/14_comparison_gfn2.py`](semiempirical/14_comparison_gfn2.py) | GFN2 on H2O for the three-tier comparison |

### PM6 (experimental)

| Script | What it does |
| --- | --- |
| [`semiempirical/05_pm6_validate.py`](semiempirical/05_pm6_validate.py) | Stewart 2007 against MOPAC-loaded PM6 parameters on reference molecules |
| [`semiempirical/06_pm6_co2_optimize.py`](semiempirical/06_pm6_co2_optimize.py) | CO2 optimised with both parameter sets, so energies are compared at their own minima |
| [`semiempirical/10_pm6_vs_mopac.py`](semiempirical/10_pm6_vs_mopac.py) | PM6 energies against MOPAC 2016 reference values |
| [`semiempirical/08_periodic_pm6.py`](semiempirical/08_periodic_pm6.py) | Periodic PM6 on 1-D chains: Gamma energy, FD gradient and stress, cutoff convergence |
| [`semiempirical/09_periodic_pm6_tutorial.py`](semiempirical/09_periodic_pm6_tutorial.py) | The end-to-end periodic NDDO workflow including variable-cell optimisation and the molecular-limit parity |
| [`semiempirical/15_comparison_pm6.py`](semiempirical/15_comparison_pm6.py) | PM6 on H2O and H2 for the three-tier comparison |
| [`semiempirical/20_atomization.py`](semiempirical/20_atomization.py) | PM6 atomisation energies and a bond-length scan against reference values |

### OM1, OM2, OM3 (experimental, retirement target)

| Script | What it does |
| --- | --- |
| [`semiempirical/17_comparison_omx.py`](semiempirical/17_comparison_omx.py) | The three orthogonalisation-corrected NDDO variants on water with the Dral 2016 parameter tables |

### Across methods

| Script | What it does |
| --- | --- |
| [`semiempirical/07_cross_validate.py`](semiempirical/07_cross_validate.py) | Smoke comparison of all molecular semiempirical methods; the only ordering it checks is SCC-DFTB not above DFTB0 |
| [`semiempirical/11_periodic_graphene.py`](semiempirical/11_periodic_graphene.py) | DFTB0, SCC-DFTB and GFN2 on the two-atom graphene cell: energy and gap |
| [`semiempirical/12_benchmark_suite.py`](semiempirical/12_benchmark_suite.py) | Markdown benchmark report across DFTB0, SCC-DFTB, GFN2 and PM6 on a standard set |
| [`semiempirical/18_comparison_all.py`](semiempirical/18_comparison_all.py) | All three tiers on water: ab initio DFT, semiempirical, and MACE when installed |
| [`semiempirical/19_gradient_validation.py`](semiempirical/19_gradient_validation.py) | Analytic against finite-difference gradients for DFTB0, GFN2 and PM6, with a Markdown report |

## MACE machine-learning interatomic potentials in `mlip/`

MACE requires a separate Python 3.13-or-earlier environment with the
optional `[mace]` extra. It is a production interface to an external
pre-trained model, not a vibe-qc method. Start with the dependency-only
health check:

```sh
.venv-mace/bin/python examples/mlip/00_runtime_healthcheck.py
```

| Script | What it does |
| --- | --- |
| [`mlip/00_runtime_healthcheck.py`](mlip/00_runtime_healthcheck.py) | Reports runtime versions, source revision when available, public periodic APIs, default model metadata, and cache root without loading weights |
| [`mlip/01_water_mace.py`](mlip/01_water_mace.py) | Molecular H2O single point and optimization through `run_job(method="mace")` |
| [`mlip/02_periodic_mace_silicon.py`](mlip/02_periodic_mace_silicon.py) | Direct periodic Si two-point strain probe with one retained model; optional cell relaxation |
| [`mlip/03_neb_mace.py`](mlip/03_neb_mace.py) | NH3 umbrella-inversion NEB through the molecular MACE dispatch |
| [`mlip/04_surface_adsorption.py`](mlip/04_surface_adsorption.py) | Exploratory materials-model adsorption protocol, not reference adsorption data |
| [`mlip/05_mace_pm6_relative_curve.py`](mlip/05_mace_pm6_relative_curve.py) | Independently normalized MACE and PM6 H2O stretch curves; never mixes their absolute energies |
| [`mlip/06_optimize_water_off23_for_dlpno.py`](mlip/06_optimize_water_off23_for_dlpno.py) | Explicitly licensed MACE-OFF23 geometry stage for the molecular DLPNO handoff |
| [`mlip/07_dlpno_ccsdt_on_mace_geometry.py`](mlip/07_dlpno_ccsdt_on_mace_geometry.py) | DLPNO-CCSD(T) energy and RHF-reference properties on the MACE-optimized geometry |
| [`mlip/08_relax_h2o_hbn_then_periodic_dft.py`](mlip/08_relax_h2o_hbn_then_periodic_dft.py) | Fixed-cell true-sheet MACE relaxation, then optional periodic PBE |

Periodic MACE does not dispatch through `run_periodic_job`. Use
`run_periodic_mace` for one point, `PeriodicMACEEvaluator` for a
sequential fixed-model scan, and `optimize_periodic_mace_cell` for cell
relaxation of 3D crystals. Use `optimize_periodic_mace_positions` for a
true slab or fixed-cell crystal. See the
[MACE tutorial](../docs/tutorial/mace_mlip.md), the
[geometry-to-electronic-structure tutorial](../docs/tutorial/mace_geometry_to_electronic_structure.md),
the [MLIP reference](../docs/user_guide/mlip.md) and the
[mlip README](mlip/README.md) for model licensing, provenance, cache
behavior, and scientific claim boundaries.

## Periodic systems in `periodic/`

The high-level `run_periodic_job` is the periodic counterpart of `run_job`:
it dispatches to the selected J/K route (`jk_method="ewald"`, `"gdf"`,
`"gpw"`, `"gapw"`, `"bipole"`, or `method="aiccm"` with a `variant`) and
writes the same artifact family plus `.POSCAR`, `.xsf` and `.cif`. Several
older examples call the periodic drivers directly and write their own
`.out` / `.system` / `.perf` files via `perf_log` + `write_system_manifest`;
both styles are valid. Route status: Ewald-3D, GDF and BIPOLE energies are
production; GPW is production within its shipped envelope (Gamma
RHF/UHF/RKS/UKS, multi-k pure-DFT, Gamma analytic gradients, DFT+U, D3-BJ)
while the runner still prints an experimental tag in the `.out` route line;
GAPW and periodic RIJCOSX are experimental; BIPOLE analytic gradients and
GPW ROHF/ROKS are previews; AICCM is experimental. See
[periodic methods](../docs/user_guide/periodic_methods.md) and
[periodic methods compared](../docs/tutorial/periodic_methods_compared.md).

### 1D periodic SCF

| Script | What it does | Manual page |
| --- | --- | --- |
| [`periodic/input-h-chain-uniform.py`](periodic/input-h-chain-uniform.py) | 1D H2 molecular crystal, k-mesh convergence study (RHF / pob-TZVP) | [pob-TZVP](../docs/tutorial/pob_tzvp.md), [k-points](../docs/user_guide/k_points.md) |
| [`periodic/input-h-chain-peierls.py`](periodic/input-h-chain-peierls.py) | Peierls-dimerisation scan on a 1D H-chain (RHF / pob-TZVP): E(delta) drops as the uniform chain distorts toward H2 units | [Peierls](../docs/tutorial/peierls.md) |
| [`periodic/input-h-chain-rks-pbe.py`](periodic/input-h-chain-rks-pbe.py) | DFT counterpart of `input-h-chain-uniform.py`: RKS-PBE / pob-TZVP on the 1D H2 chain | [periodic DFT](../docs/tutorial/periodic_dft.md) |
| [`periodic/input-h-chain-bands.py`](periodic/input-h-chain-bands.py) | Hcore band structure + density of states for the same 1D chain at STO-3G (matplotlib) | [band structure](../docs/tutorial/band_structure.md) |
| [`periodic/input-h-chain-plus-u.py`](periodic/input-h-chain-plus-u.py) | Periodic H2 in a 50 bohr box, RKS-LDA / cc-pVDZ with U = 4 eV on the H p channel: the +U surface in a few seconds | [DFT+U](../docs/tutorial/dft_plus_u.md) |
| [`periodic/input-h-chain-bvk-sweep.py`](periodic/input-h-chain-bvk-sweep.py) | Born-von Karman sweep at STO-3G: the same chain in five unit-cell representations must give the same per-pair energy | [H8 chain and the CCM ancestor](../docs/tutorial/h8_chain_ccm_ancestor.md) |
| [`periodic/input-k-mesh-convergence.py`](periodic/input-k-mesh-convergence.py) | Walks every `vibeqc.KPoints` flavour (Gamma, Monkhorst-Pack, IBZ-reduced, KPPRA, band path) on a cubic Mg cell at STO-3G | [periodic HF](../docs/tutorial/periodic_hf.md), [k-points](../docs/user_guide/k_points.md) |

### 3D crystals on Ewald-3D and GDF

| Script | What it does | Manual page |
| --- | --- | --- |
| [`periodic/input-lih-pob-tzvp.py`](periodic/input-lih-pob-tzvp.py) | LiH cubic rocksalt, pob-TZVP / RHF / EWALD_3D, multi-k IBZ, bands, PDOS and a crystalline-orbital cube | [LiH solid state](../docs/tutorial/lih_pob_tzvp_solid_state.md) |
| [`periodic/input-h2-cell-rks-skala-gdf.py`](periodic/input-h2-cell-rks-skala-gdf.py) | Experimental 3D H2 / STO-3G / Gamma GDF SKALA API smoke run; not a periodic benchmark | [SKALA](../docs/user_guide/skala.md) |
| [`periodic/input-mgo-pob-tzvp.py`](periodic/input-mgo-pob-tzvp.py) | MgO rocksalt, pob-TZVP / RHF / EWALD_3D (CRYSTAL Tutorial port, conventional 8-atom cell) | [MgO from Materials Project](../docs/tutorial/mgo_from_materials_project.md) |
| [`periodic/input-mgo-rocksalt-rhf.py`](periodic/input-mgo-rocksalt-rhf.py) | MgO rocksalt, STO-3G / RHF / Gamma-only GDF; the vibe-qc half of the RHF parity test against PySCF.pbc | [periodic methods](../docs/user_guide/periodic_methods.md) |
| [`periodic/input-mgo-rocksalt-rks-pbe.py`](periodic/input-mgo-rocksalt-rks-pbe.py) | MgO rocksalt, STO-3G / RKS-PBE / Gamma-only GDF; DFT counterpart of the parity test | [periodic DFT](../docs/tutorial/periodic_dft.md) |
| [`periodic/input-al2o3-cubic-rhf.py`](periodic/input-al2o3-cubic-rhf.py) | Al2O3 cubic stuffed-fluorite model, STO-3G / RHF / Gamma-only GDF (a cubic model, not corundum; the docstring says why) | [periodic HF](../docs/tutorial/periodic_hf.md) |
| [`periodic/input-al2o3-cubic-rks-pbe.py`](periodic/input-al2o3-cubic-rks-pbe.py) | Al2O3 cubic model, STO-3G / RKS-PBE / Gamma-only GDF with Fermi-Dirac smearing (no PBE gap at Gamma) | [periodic DFT](../docs/tutorial/periodic_dft.md) |
| [`periodic/input-nacl-sto3g-dft.py`](periodic/input-nacl-sto3g-dft.py) | NaCl rocksalt, STO-3G / RKS-LDA / EWALD_3D, a debug-friendly solid demo with bands and DOS (CRYSTAL Tutorial port) | [periodic DFT](../docs/tutorial/periodic_dft.md) |
| [`periodic/input-si-bands.py`](periodic/input-si-bands.py) | Silicon diamond, Hcore band structure + DOS at STO-3G (CRYSTAL Tutorial port) | [band structure](../docs/tutorial/band_structure.md) |
| [`periodic/input-lih-eos.py`](periodic/input-lih-eos.py) | LiH rocksalt equation of state: Gamma-only RHF/STO-3G on the GDF route at five lattice constants around the model's own minimum (about 3.2 angstrom), then a third-order Birch-Murnaghan fit with `vibeqc.eos.fit_birch_murnaghan` giving V0, B0, B0' and the fit quality; refuses to report V0 if the minimum is not bracketed Heavy: on a shared, oversubscribed host each single point took minutes to about an hour; idle-machine times were not measured. | [equation of state](../docs/user_guide/equation_of_state.md), [periodic geometry optimisation](../docs/tutorial/periodic_geometry_optimization.md) |
| [`periodic/input-lih-mdf-vs-rsgdf.py`](periodic/input-lih-mdf-vs-rsgdf.py) | The three GDF fit backends (`gdf_method="rsgdf"`, `"compcell"`, `"mdf"`) on LiH at Gamma, with pairwise energy differences and the refusal message printed if a backend declines the cell Heavy: on a shared, oversubscribed host each single point took minutes to about an hour; idle-machine times were not measured. | [periodic methods](../docs/user_guide/periodic_methods.md) |
| [`periodic/input-mgo-density-mixing.py`](periodic/input-mgo-density-mixing.py) | MgO RKS-LDA/STO-3G on multi-k GDF with Fock DIIS, Anderson, Broyden and Kerker-preconditioned density mixing; prints iteration counts (13, 13, 13 and 37 here) and shows all four reach the same energy to 3e-8 Ha; the Kerker run emits a GAPWExperimentalWarning Heavy: on a shared, oversubscribed host each single point took minutes to about an hour; idle-machine times were not measured. | [periodic SCF convergence](../docs/tutorial/periodic_scf_convergence.md) |
| [`periodic/input-nacl-from-cif.py`](periodic/input-nacl-from-cif.py) | NaCl read from the committed `nacl.cif` (8-atom conventional cell) and `nacl.POSCAR` (2-atom primitive cell), symmetrised to the same Fm-3m primitive cell, then one Gamma RHF/STO-3G GDF single point from each; the two energies agree Heavy: on a shared, oversubscribed host each single point took minutes to about an hour; idle-machine times were not measured. | [external structures](../docs/user_guide/external_structures.md), [crystal lattices](../docs/user_guide/crystal_lattices.md) |

### GPW route (`gpw-*.py`)

Gaussian and plane waves: J on an FFT grid, everything else in the Gaussian
basis. Pair these with [periodic GPW](../docs/tutorial/periodic_gpw.md).

| Script | What it does |
| --- | --- |
| [`periodic/gpw-h2-sto3g-rhf.py`](periodic/gpw-h2-sto3g-rhf.py) | The simplest end-to-end GPW job: H2 in a 16-bohr cube, RHF / STO-3G through `run_periodic_job(jk_method="gpw")` |
| [`periodic/gpw-he-sto3g-lda.py`](periodic/gpw-he-sto3g-lda.py) | He in a 16-bohr cube, RKS-LDA: the closed-shell DFT extension of the GPW path |
| [`periodic/gpw-h2-sto3g-rks-pbe-with-output.py`](periodic/gpw-h2-sto3g-rks-pbe-with-output.py) | RKS-PBE with the full output family and `write_density=True`, a template for a production-style GPW job |
| [`periodic/gpw-h-atom-uhf.py`](periodic/gpw-h-atom-uhf.py) | Single H atom, open-shell UHF through `run_periodic_uhf_gpw` with explicit per-spin occupations |
| [`periodic/gpw-li-sto3g-rohf.py`](periodic/gpw-li-sto3g-rohf.py) | Li doublet, ROHF / STO-3G: the smallest periodic restricted-open-shell case (preview: 3-D, Gamma-only, integer occupations) |
| [`periodic/gpw-h2-sto3g-pbe-multi-k.py`](periodic/gpw-h2-sto3g-pbe-multi-k.py) | Multi-k RKS-PBE on a 2x2x2 mesh through `run_periodic_rks_gpw_multi_k` (pure DFT only on the multi-k path) |
| [`periodic/gpw-h2-sto3g-smearing.py`](periodic/gpw-h2-sto3g-smearing.py) | Fermi-Dirac smearing on the multi-k GPW route: free energy F = E - TS and the entropy term ([smearing tutorial](../docs/tutorial/fermi_dirac_smearing.md)) |
| [`periodic/gpw-h2-dimer-d3bj.py`](periodic/gpw-h2-dimer-d3bj.py) | Two H2 molecules in a 20-bohr cube, PBE-D3(BJ) on the GPW RKS path, `e_dispersion` on the breakdown |
| [`periodic/gpw-h2-sto3g-dos.py`](periodic/gpw-h2-sto3g-dos.py) | Post-SCF DOS and HOMO/LUMO helpers (`compute_dos_from_result`, `compute_homo_lumo`) on a GPW result |
| [`periodic/gpw-h2-sto3g-restart.py`](periodic/gpw-h2-sto3g-restart.py) | `save_gpw_result` / `load_gpw_result`: a converged GPW result as one compressed `.npz` |
| [`periodic/gpw-h2-sto3g-geomopt-ase.py`](periodic/gpw-h2-sto3g-geomopt-ase.py) | Geometry relaxation with ASE BFGS and the `VibeqcGPW` calculator ([periodic geometry optimisation](../docs/tutorial/periodic_geometry_optimization.md)) |
| [`periodic/gpw-h2-sto3g-tddft.py`](periodic/gpw-h2-sto3g-tddft.py) | GPW ground state, then TDA/CIS vertical excitation energies of H2 in a box ([TDDFT tutorial](../docs/tutorial/excited_states_tddft.md)) |

### GAPW route (`gapw-*.py`, experimental)

| Script | What it does | Manual page |
| --- | --- | --- |
| [`periodic/gapw-he-sto3g-rhf.py`](periodic/gapw-he-sto3g-rhf.py) | He in a 12-bohr cube, all-electron GAPW RHF / STO-3G: FFT-Poisson J with per-atom augmentation, Ewald-gauge V_ne and E_nn | [periodic GAPW](../docs/tutorial/periodic_gapw.md), [GAPW guide](../docs/user_guide/gapw.md) |

### BIPOLE route (`input-bipole-*.py`)

CRYSTAL-gauge Ewald J split with a shared alpha across V_ne, E_nn and the
long-range J. Energies and finite-difference forces are production, the
analytic gradient is a maintained preview, the quartet multipole far field
is gated. Pair with the [BIPOLE guide](../docs/user_guide/bipole.md) and
[the physical-domain tutorial](../docs/tutorial/bipole_physical_domain.md).

| Script | What it does |
| --- | --- |
| [`periodic/input-bipole-mgo-rhf.py`](periodic/input-bipole-mgo-rhf.py) | MgO primitive cell, RHF / STO-3G / Gamma-only BIPOLE through the driver; docstring lists the CRYSTAL14 and GDF numbers to compare with |
| [`periodic/input-bipole-mgo-rks-pbe.py`](periodic/input-bipole-mgo-rks-pbe.py) | The RKS-PBE counterpart on the BIPOLE RKS driver |
| [`periodic/input-bipole-runner-mgo-rhf.py`](periodic/input-bipole-runner-mgo-rhf.py) | The same MgO RHF dispatched through `run_periodic_job(jk_method="bipole")` |
| [`periodic/input-bipole-runner-mgo-multik.py`](periodic/input-bipole-runner-mgo-multik.py) | Multi-k (2x2x2) BIPOLE through `run_periodic_job` |
| [`periodic/input-bipole-workflow-mgo.py`](periodic/input-bipole-workflow-mgo.py) | The supported fixed-cell workflow: single point with the analytic-gradient preview diagnostic, position relaxation, energy and gradient checks |
| [`periodic/input-bipole-li-uhf.py`](periodic/input-bipole-li-uhf.py) | Isolated Li atom in a 10-bohr box, UHF / STO-3G: the molecular limit of the BIPOLE gauge |
| [`periodic/input-bipole-nio-uks-plus-u.py`](periodic/input-bipole-nio-uks-plus-u.py) | NiO, UKS-LDA+U / STO-3G, U_eff in {0, 4, 7} eV on Ni 3d: the textbook DFT+U recipe ([DFT+U](../docs/tutorial/dft_plus_u.md)) |
| [`periodic/input-bipole-domain.py`](periodic/input-bipole-domain.py) | LiH BIPOLE domain selection; nothing runs without `--run`, and `--historical` selects reproduction support |

### The GDF parity grid (`periodic/<System>/` and `periodic_pyscf/`)

Ten crystals, each in its own directory, with an RHF, an RKS-PBE and an
RKS-B3LYP input at pob-TZVP (STO-3G for the two large cells), all through
the native Gamma-only GDF driver: `Al2O3-corundum`, `C-diamond`,
`LiH-rocksalt`, `MgO-rocksalt`, `NaCl-rocksalt`, `Ne-fcc`, `Si-diamond`,
`SiO2-alpha-quartz`, `TiO2-rutile`, `ZnO-wurtzite`. Every script embeds
its geometry and runs standalone. `periodic_pyscf/<System>/` holds the
PySCF twin of each input; PySCF is an optional external reference and is
never imported by vibe-qc itself. The gallery page
[periodic crystal gallery](../docs/tutorial/periodic_crystal_gallery.md)
walks through them.

| Helper | What it does |
| --- | --- |
| [`periodic/_systems.py`](periodic/_systems.py) | The geometry registry (lattice parameters and Wyckoff positions) as `ase.Atoms` |
| [`periodic/_gen.py`](periodic/_gen.py) | Regenerates every `<System>/<System>-<Method>-<Basis>.py` and its PySCF twin from the registry |
| [`periodic/run_all.py`](periodic/run_all.py) | Runs every grid script in its directory and prints a vibe-qc against PySCF parity table |
| [`periodic/run_missing.py`](periodic/run_missing.py) | Re-runs only the inputs whose `.out` lacks a parseable final energy |
| [`periodic/MgO-rocksalt/*-mp8-gdf.py`](periodic/MgO-rocksalt/README-crystal14-parity.md) | Three MgO primitive-cell inputs with an 8x8x8 mesh on the native multi-k GDF backend, paired with CRYSTAL14 `.d12` decks |

Three `gpw-*` inputs have committed output siblings (`.xyz`, `.POSCAR`,
`.cif`); they are reference artifacts, not something a run should recreate
in the tree.

### AICCM (experimental)

The ab initio cyclic cluster model and everything under
`vibeqc.periodic.ccm` / `vibeqc.periodic.chi` are experimental research
lines; their numbers are not validated results and are not release-gated.
The front door is `run_periodic_job(method="aiccm", variant=...)` with the
four variants `real-gamma`, `neutral-bloch` (the two producers of the
neutral Gamma-CCM construction), `four-center` (the 2014 union-and-weight
lineage, development line `aiccm2026dev-a`) and `chi` (chi-CCM, the
finite-translation-group character construction, `aiccm2026dev-b`). Read
[the AICCM guide](../docs/user_guide/aiccm.md),
[the quickstart](../docs/tutorial/aiccm_quickstart.md) and
[the chi-CCM tutorial](../docs/tutorial/aiccm2026dev_b.md). These demos
write into the current directory.

| Script | What it does |
| --- | --- |
| [`periodic/aiccm2026dev_a_demo.py`](periodic/aiccm2026dev_a_demo.py) | The full stack on the four-center Gamma-CCM construction |
| [`periodic/aiccm2026dev_b_demo.py`](periodic/aiccm2026dev_b_demo.py) | chi-CCM RHF and RKS through `method="aiccm", variant="chi"` with the four-center, RI or RIJCOSX backend; `--dim 1` or `2` verifies the fail-closed path |
| [`periodic/aiccm2026dev_b_mp2.py`](periodic/aiccm2026dev_b_mp2.py) | The canonical 3-D RI-MP2 route on chi-CCM |
| [`periodic/aiccm2026dev_b_local_correlation.py`](periodic/aiccm2026dev_b_local_correlation.py) | Local-PNO MP2 and CCSD(T) on a finite chi-CCM torus |
| [`periodic/aiccm2026dev_b_localization.py`](periodic/aiccm2026dev_b_localization.py) | Occupied localisation on a 3-D H2 control, printing invariants only ([tutorial](../docs/tutorial/aiccm2026dev_b_localization.md)) |
| [`periodic/aiccm2026dev_b_open_shell.py`](periodic/aiccm2026dev_b_open_shell.py) | Unrestricted chi-CCM on a Li doublet |
| [`periodic/aiccm2026dev_diamond_bonds_bands_compare.py`](periodic/aiccm2026dev_diamond_bonds_bands_compare.py) | Side-by-side Gamma-CCM and chi-CCM property bundles for diamond: energies, gaps, charges, Mayer bond orders |
| [`periodic/chi_rohf_rmp2.py`](periodic/chi_rohf_rmp2.py) | Proposed LiH2 chi ROHF/RMP2 validation case with checkpoint and resume; scientific acceptance pending ([chi ROHF-MP2](../docs/chi_rohf_mp2.md)) |

### Bonding, visualisation and infrastructure

| Script | What it does | Manual page |
| --- | --- | --- |
| [`periodic/coop_cohp_h2_chain.py`](periodic/coop_cohp_h2_chain.py) | COOP/COHP and periodic Mayer bond orders for a chain of H2 molecules: a GDF RHF SCF in a 3-D vacuum box with `coop_cohp=True`, then the explicit-operator API on a strictly 1-D chain (production on the GDF route; matplotlib; writes into the current directory) | [COOP/COHP](../docs/user_guide/coop_cohp.md), [COOP and COHP tutorial](../docs/tutorial/coop_cohp_bonding.md) |
| [`periodic/input-xsf-bxsf-nacl.py`](periodic/input-xsf-bxsf-nacl.py) | NaCl SCF density as XSF plus a teaching-model BXSF of Hcore energies | [XSF and BXSF](../docs/tutorial/xsf_bxsf_visualization.md) |
| [`periodic/gilat-be-fcc-dos.py`](periodic/gilat-be-fcc-dos.py) | Be fcc metal: parameter-free Gilat-Raubenheimer Fermi level and DOS after the SCF | [Gilat net for metals](../docs/tutorial/gilat_net_metals.md) |
| [`periodic/input-symmetry-demo.py`](periodic/input-symmetry-demo.py) | `symmetry="auto"`, primitive reduction, `attach_symmetry`, symmetrisation operators and k-mesh reduction in one walk-through | [symmetry storage](../docs/tutorial/symmetry_storage.md) |
| [`periodic/input-nacl-symmetry.py`](periodic/input-nacl-symmetry.py) | NaCl Pm-3m: 19x compression of the overlap `LatticeMatrixSet` via atom-pair orbits plus a Wigner-D check | [symmetry storage](../docs/tutorial/symmetry_storage.md) |
| [`periodic/periodic-megacell-local-correlation.py`](periodic/periodic-megacell-local-correlation.py) | Megacell (finite-supercell) route to periodic correlation: per-cell MP2 and DLPNO-CCSD(T) of an H2 chain converging with supercell size | [DLPNO guide](../docs/user_guide/dlpno_mp2.md) |
| [`periodic/ecp_diagnostics.py`](periodic/ecp_diagnostics.py) | One-line pre-SCF ECP sanity summary: catches silent all-electron fallbacks and wrong valence counts | [ECPs](../docs/user_guide/ecp.md) |
| [`periodic/input-madelung-constants.py`](periodic/input-madelung-constants.py) | Madelung constants for NaCl, CsCl, ZnS via `ewald_point_charge_energy`; reproduces textbook values to 8+ digits | [Madelung with Ewald](../docs/tutorial/madelung_with_ewald.md) |
| [`periodic/input-lih-chain-fat-bands.py`](periodic/input-lih-chain-fat-bands.py) | Hcore fat bands of a true 1-D LiH chain: `vibeqc.bands.band_structure_projected` gives Li-s, Li-p and H-s Mulliken character along Gamma to X, printed as a table (valence band H-s, lowest empty band switching Li-p to Li-s); an opt-in `WRITE_QVF` flag embeds the bands in a QVF through `run_periodic_job(..., band_structure=...)`. Non-self-consistent characters, labelled as such | [band structure](../docs/tutorial/band_structure.md), [projected DOS](../docs/tutorial/pdos.md) |

A note on 1-D cells: the GDF route fails closed on a strictly one-dimensional
(`dim=1`) `PeriodicSystem` combined with a k mesh, because the bare nuclear
lattice sums have no consistent Coulomb gauge; it raises
`NotImplementedError`. `periodic/coop_cohp_h2_chain.py` hit exactly this at
HEAD and was repaired by putting the chain in a 3-D cell with 10 bohr of
vacuum and `kpoints=[8, 1, 1]`, the recipe of the COOP and COHP tutorial. Its
explicit-operator block still uses a `dim=1` chain, which needs no SCF.

## Multi-step workflows in `workflows/`

| Script | What it does | Manual page |
| --- | --- | --- |
| [`workflows/input-nh3-umbrella-neb.py`](workflows/input-nh3-umbrella-neb.py) | NH3 umbrella inversion via climbing-image NEB / RHF / STO-3G, a transition-state search demo | [NEB reaction path](../docs/tutorial/neb_reaction_path.md) |
| [`workflows/input-h3-neb-warm-start-benchmark.py`](workflows/input-h3-neb-warm-start-benchmark.py) | H + H2 exchange, climbing-image NEB: warm-start against cold-start timing, two QVF animations | [NEB guide](../docs/user_guide/neb.md) |
| [`workflows/input-periodic-h2plus-neb-plus-u.py`](workflows/input-periodic-h2plus-neb-plus-u.py) | `run_neb` on a `PeriodicSystem`: H2+ stretching with and without DFT+U on the H 1s channel | [NEB guide](../docs/user_guide/neb.md), [DFT+U](../docs/tutorial/dft_plus_u.md) |

## ASE workflows in `ase_workflows/`

Pure-ASE workflows where vibe-qc is the backing calculator. They drive
`ase.optimize.BFGS`, `ase.vibrations.Vibrations`, `ase.mep.NEB` and
`ase.md.VelocityVerlet` directly, with no glue beyond
`atoms.calc = VibeQC(...)`. ASE is a core dependency, so they run out of the
box. See the [ASE integration guide](../docs/user_guide/ase_integration.md).

| Script | What it does |
| --- | --- |
| [`ase_workflows/optimize-via-ase-bfgs.py`](ase_workflows/optimize-via-ase-bfgs.py) | H2O / 6-31G* geometry opt via ASE's BFGS, with an energy and max-force convergence plot (matplotlib) |
| [`ase_workflows/vibrations-via-ase-vibrations.py`](ase_workflows/vibrations-via-ase-vibrations.py) | H2O vibrational frequencies via `ase.vibrations.Vibrations` (finite differences of analytic gradients) |
| [`ase_workflows/nh3-neb-via-ase.py`](ase_workflows/nh3-neb-via-ase.py) | NH3 umbrella inversion via climbing-image NEB, vibe-qc as the only backend (3 to 5 minutes) |
| [`ase_workflows/md-water-nve.py`](ase_workflows/md-water-nve.py) | Velocity-Verlet NVE on H2O, an energy-conservation diagnostic (drift below 10 meV over 25 fs is the force-energy consistency test) |
| [`ase_workflows/surface-h2-pt111-singlepoint.py`](ase_workflows/surface-h2-pt111-singlepoint.py) | Pt(111) slab + atop H2 geometry fixture; writes the XYZ and prints the `vibeqc.ase_periodic.VibeQCPeriodic` + BFGS recipe (the calculator exists; the Pt slab itself is a production-size job, so no SCF runs) |

## Cross-code validation in `ase_compare/`

Validate vibe-qc results against external QC codes (PySCF, ORCA, Psi4)
using the [`vibeqc.benchmark`](../python/vibeqc/benchmark.py) framework.
Each script auto-detects which codes are installed, skips missing ones
gracefully, and asserts numerical agreement at the end. To enable ORCA /
Psi4 comparisons, see [the external codes user guide](../docs/user_guide/external_codes.md)
and [cross-validation](../docs/tutorial/cross_validation.md).

| Script | What it does |
| --- | --- |
| [`ase_compare/compare-h2o-hf.py`](ase_compare/compare-h2o-hf.py) | H2O / RHF / 6-31G* across vibe-qc + PySCF + ORCA; tight tolerance (1e-5 eV) since the SCF skeleton is identical |
| [`ase_compare/compare-h2o-dft.py`](ase_compare/compare-h2o-dft.py) | H2O / RKS-PBE / 6-31G*; looser tolerance (5 mHa) for per-code DFT-grid differences; adds a dipole comparison |
| [`ase_compare/compare-h2o-vibrations.py`](ase_compare/compare-h2o-vibrations.py) | H2O analytic CPHF Hessian and `.hess` normal-modes file from vibe-qc and ORCA, frequencies compared |
| [`ase_compare/compare-water-dimer-dispersion.py`](ase_compare/compare-water-dimer-dispersion.py) | Water dimer interaction energy with and without D3-BJ; verifies the Grimme parameters against ORCA |
| [`ase_compare/compare-basis-convergence.py`](ase_compare/compare-basis-convergence.py) | H2O / RHF across 5 basis sets x 3 codes; agreement at every basis to below 1 meV (matplotlib plot) |
| [`ase_compare/compare-nh3-neb-barrier.py`](ase_compare/compare-nh3-neb-barrier.py) | Full NH3 NEB barrier against ORCA NEB on the same protocol; 1 kcal/mol agreement |

[`ase_compare/cross_code_regression/`](ase_compare/cross_code_regression/README.md)
holds the developer head-to-head set: RHF, RKS-PBE and MP2 basis scans of
H2O against ORCA and PySCF, and two periodic checks (easy wide-gap systems
and the vacuum-box molecular limit) against PySCF.pbc. They import PySCF in
process, which is allowed under `examples/` but never inside `python/vibeqc/`.

Sanity-check the setup with `./scripts/check_external_codes.sh` before
running the comparison scripts; it probes every supported calculator and
prints an availability table (exit 0 when vibe-qc plus at least one peer
code is available).

## External structures and reference data (`vqfetch`)

Both scripts need the optional `[fetch]` extra and network access, and pair
with [the external data fetcher tutorial](../docs/tutorial/external_data_fetcher.md)
and [external structures](../docs/user_guide/external_structures.md).

| Script | What it does |
| --- | --- |
| [`input-vqfetch-optimade-mgo.py`](input-vqfetch-optimade-mgo.py) | Pulls MgO rocksalt from the Materials Project OPTIMADE endpoint, inspects the `PeriodicSpec` and provenance, emits a spec module and runs it |
| [`input-vqfetch-reference-h2o.py`](input-vqfetch-reference-h2o.py) | Pulls the NIST CCCBDB reference page for H2O (atomisation energy, formation enthalpy) and round-trips it through vibe-qc |

## Wavefunction exchange, optimisers and populations

| Directory or script | What it does | Manual page |
| --- | --- | --- |
| [`trexio/`](trexio/README.md) | Export and restart water RHF, OH UHF and NaH ECP RHF through TREXIO, a two-k-point He cell, LiH CASCI/CASSCF/FCI wavefunctions, and backend conversion; needs `[trexio]` and `--output-dir` | [TREXIO exchange](../docs/tutorial/trexio_exchange.md), [correlated and periodic](../docs/tutorial/trexio_correlated_periodic.md) |
| [`opentrustregion/`](opentrustregion/README.md) | Native against OpenTrustRegion SCF for water and its cation, a READ restart, and following a spin-breaking instability of stretched H2; needs the optional backend and `--output-dir` | [OpenTrustRegion](../docs/tutorial/opentrustregion.md), [stability](../docs/tutorial/opentrustregion_stability.md) |
| [`iao_population_comparison.py`](iao_population_comparison.py) | Reproduces the IAO guide's comparison with Knizia 2013 Table 1 and the open-shell examples; needs `--output-dir` | [IAO validation](../docs/user_guide/iao_validation.md) |

## Surface embedding in `embedding/` (experimental)

Green's-function embedding of an adsorbate on a periodic slab (Layer A
embedded density, Layer B Lloyd's-formula adsorption energy). Experimental;
see [surface embedding](../docs/tutorial/surface_embedding.md) and the
[guide](../docs/user_guide/surface_embedding.md).

| Script | What it does |
| --- | --- |
| [`embedding/simple_adsorbate.py`](embedding/simple_adsorbate.py) | The complete pipeline on a 2-D H-atom slab: tag substrate and region I, embedded density, clean-surface G0, adsorption energy |
| [`embedding/open_shell_adsorbate.py`](embedding/open_shell_adsorbate.py) | The `EmbeddedSurfaceCalculator` ASE route with `charge` and `multiplicity` for a spin-polarised cell |
| [`embedding/two_step_scf_and_forces.py`](embedding/two_step_scf_and_forces.py) | `sigma_scf=True` (embedding potential from the SCF substrate Fock, Ishida two-step) plus numerical forces on an H-chain surface |
| [`embed_convergence.py`](embed_convergence.py) | Embedded-cluster convergence protocol on MgO and NaCl hosts: QM cluster size and AIMP shell thickness against the Madelung stabilisation |

## vibe-view showcase inputs in `vibe_view/`

Eight `run_job` / `run_periodic_job` inputs whose `.qvf` exercises the
viewer: formaldehyde (every common section), NaCl rocksalt and CO on
MgO(001) (periodic rendering), the whole QVF section surface, basis
functions, CASSCF natural orbitals, NEB reaction paths, and H2S with MSINDO.
The directory [README](vibe_view/README.md) has the run commands, the
viewer launch and the section coverage. `output-nacl-showcase.qvf` and the
two archives under `vibe_view/runs/` are committed reference outputs. Pair
with [the vibe-view guide](../docs/user_guide/vibe_view.md).

## JK builder in `jk_builder/`

[`jk_builder/periodic_gamma_via_jk_builder.py`](jk_builder/periodic_gamma_via_jk_builder.py)
drives a Gamma-only periodic RHF SCF through the same C++ `JKBuilder`
interface the molecular SCF drivers use (`run_rhf_scf_with_jk`), showing
that the iteration body (orthogonalisation, DIIS, damping, level shift) is
shared. Pair with [density fitting](../docs/user_guide/density_fitting.md).

## Basis-set development in `basisset_dev/`

The pob-paper calibration set as runnable inputs (39 cubic compounds with
CRYSTAL14 `.d12` sidecars), the generator that emits them from
`vibe_basis.io.structures`, the cohesive-energy pipeline demo, the ATOMBSSE
ghost-shell convergence tools and the Gamma-phonon gate. Needs the
`[basisopt]` extra for the generator and the CRYSTAL parity path; read the
[basisset_dev README](basisset_dev/README.md), the
[basis toolkit guide](../docs/user_guide/basis_toolkit.md) and
[cohesive energies](../docs/user_guide/cohesive_energies.md).

## Benchmarks, experiments, CRYSTAL decks

* [`benchmarks/benchmark_eigensolvers.py`](benchmarks/benchmark_eigensolvers.py)
  compares the dense, Davidson, LOBPCG and Lanczos eigensolvers on Fock-like
  matrices: runtime, matrix-vector products, accuracy, matrix-free mode and
  warm starts. Davidson, LOBPCG and dense are production; see the
  [solver framework guide](../docs/user_guide/solver_framework.md).
* [`experimental/coulomb-method-zoo/`](experimental/coulomb-method-zoo/README.md)
  is a closed spike that compared alternative periodic Coulomb J builders
  (Ewald-3D, Wolf, analytic Ewald, ADFT fitting) on H2 in a box against the
  old Gamma-only Ewald driver. Its README says which scripts exist and its
  POSTMORTEM records the verdicts; nothing in it is a current benchmark.
* [`crystal23_demo/`](crystal23_demo/README.md) mirrors the CRYSTAL23 demo
  `.d12` decks for smoke-testing the external `crystal23demo` binaries; no
  Python here.

## Debug and plots

* [`debug/`](debug/README.md): 70 diagnostic scripts grouped by the bug
  hunt they served (periodic SCF health checks with a decision tree, gauge
  probes, GDF and MDF development, AO-convention calibration, BIPOLE
  gradients) plus a "historical, no longer runnable" list.
* [`plots/`](plots/README.md): one script per tutorial figure or animation,
  writing into `docs/_static/plots/`; all but `bond-orbitals.py` need matplotlib.
* [`regression/`](regression/README.md): the parity suite against PySCF and
  ORCA with its own operator guide, design notes and expected values.

## Geometry input

The newer examples (`molecular/input-h2o-rks-pbe-d3-opt.py`,
`molecular/input-oh-ump2.py`, every `periodic/input-*.py`) **inline
the geometry** directly in the source as a list of `Atom(Z, [x, y, z])`
calls: bohr coordinates, fully self-contained, no external files
required.

The original H2O / H2 family (`molecular/input-h2o-*.py`) reads from
shared XYZ files at `examples/h2.xyz`, `examples/h2o.xyz`,
`examples/h2o_dimer.xyz`, `examples/h2o_trimer.xyz` via
`Molecule.from_xyz(...)`; `examples/h2s.xyz` serves the two MSINDO H2S
inputs (`semiempirical/16_comparison_msindo.py`,
`vibe_view/showcase_msindo_h2s.py`). Swap in your own XYZ file to run these
methods on other molecules; both patterns are plain Python and trivial to
edit.
