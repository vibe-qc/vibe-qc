# Example scripts and generated outputs

The repository keeps runnable example inputs under `examples/`. Local
runs still write `.out`, `.system`, `.molden`, `.traj`, `.cube`,
`.png`, `.csv`, or citation files next to the script that produced
them; those ad hoc files are local working products and should stay out
of commits.

For tutorials and reference pages, the docs also carry curated static
bundles under `docs/_static/examples/`; the root manifest is
[`artifact-index.json`](_static/examples/artifact-index.json). Those
bundles are regenerated on `vq` with the current development build, then
copied into the docs site with the input, full `stdout.txt`, full
`stderr.txt`, matched output files, QVF archives where the example
produces them, and vibe-view PNG captures for QVF-backed runs.

```{tip}
Every text output starts with the runtime banner: a labelled box
recording vibe-qc version, codename, git revision, and linked
native-library versions. For manuscript or SI work, copy that banner
into your paper files alongside the `.system` manifest. Use the static
bundles below as the documented reference outputs; for new exploratory
runs, use a scratch directory outside the checkout and commit only the
input script or a small data table needed to reproduce the figure.
```

## Source Locations

The canonical examples live in:

- [`examples/molecular/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/molecular),
  molecule-only HF / DFT / MP2, optimization, vibrations, and cubes.
- [`examples/periodic/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/periodic),
  one-dimensional chains, bulk crystals, GPW/GAPW demos, and periodic
  output formats.
- [`examples/ase_workflows/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/ase_workflows),
  ASE-native BFGS, NEB, vibrations, and MD workflows.
- [`examples/ase_compare/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/ase_compare),
  cross-code validation drivers that create comparison tables locally.

## Static Reference Bundles

Most bundles below were regenerated with `scripts/regenerate_doc_examples.py`
through `vq`; curated validation fixtures note their own provenance in their
README. Each directory includes `README.md`, `artifact-status.json`, the copied
input script or scripts, and the full text logs available for that run. The
root index is [`artifact-index.json`](_static/examples/artifact-index.json);
the vibe-view capture status is
[`screenshot-status.json`](_static/examples/screenshot-status.json).

| Bundle | Main downloads | Notes |
| --- | --- | --- |
| [`h2o-rhf`](_static/examples/h2o-rhf/artifact-status.json) | [`out`](_static/examples/h2o-rhf/output-h2o-rhf.out), [`system`](_static/examples/h2o-rhf/output-h2o-rhf.system), [`qvf`](_static/examples/h2o-rhf/output-h2o-rhf.qvf), [`screenshot`](_static/examples/h2o-rhf/output-h2o-rhf-structure.png) | Closed-shell H2O RHF / `6-31G*`. |
| [`h2o-dft`](_static/examples/h2o-dft/artifact-status.json) | [`out`](_static/examples/h2o-dft/output-h2o-dft.out), [`system`](_static/examples/h2o-dft/output-h2o-dft.system), [`qvf`](_static/examples/h2o-dft/output-h2o-dft.qvf), [`screenshot`](_static/examples/h2o-dft/output-h2o-dft-structure.png) | H2O RKS-PBE / `6-31G*`. |
| [`h2o-opt`](_static/examples/h2o-opt/artifact-status.json) | [`out`](_static/examples/h2o-opt/output-h2o-opt.out), [`traj`](_static/examples/h2o-opt/output-h2o-opt.traj), [`qvf`](_static/examples/h2o-opt/output-h2o-opt.qvf), [`screenshot`](_static/examples/h2o-opt/output-h2o-opt-structure.png) | ASE/BFGS H2O relaxation. |
| [`h2o-dimer-opt`](_static/examples/h2o-dimer-opt/artifact-status.json) | [`out`](_static/examples/h2o-dimer-opt/output-h2o-dimer-opt.out), [`traj`](_static/examples/h2o-dimer-opt/output-h2o-dimer-opt.traj), [`qvf`](_static/examples/h2o-dimer-opt/output-h2o-dimer-opt.qvf), [`screenshot`](_static/examples/h2o-dimer-opt/output-h2o-dimer-opt-structure.png) | Water-dimer relaxation. |
| [`h2o-trimer-opt`](_static/examples/h2o-trimer-opt/artifact-status.json) | [`out`](_static/examples/h2o-trimer-opt/output-h2o-trimer-opt.out), [`traj`](_static/examples/h2o-trimer-opt/output-h2o-trimer-opt.traj), [`qvf`](_static/examples/h2o-trimer-opt/output-h2o-trimer-opt.qvf), [`screenshot`](_static/examples/h2o-trimer-opt/output-h2o-trimer-opt-structure.png) | Cyclic water-trimer relaxation. |
| [`oh-radical`](_static/examples/oh-radical/artifact-status.json) | [`out`](_static/examples/oh-radical/output-oh-radical.out), [`system`](_static/examples/oh-radical/output-oh-radical.system), [`qvf`](_static/examples/oh-radical/output-oh-radical.qvf), [`screenshot`](_static/examples/oh-radical/output-oh-radical-structure.png) | Open-shell UHF OH doublet. |
| [`h2o-vibrations`](_static/examples/h2o-vibrations/artifact-status.json) | [`out`](_static/examples/h2o-vibrations/output-h2o-vibrations.out), [`hess`](_static/examples/h2o-vibrations/output-h2o-vibrations.hess), [`qvf`](_static/examples/h2o-vibrations/output-h2o-vibrations.qvf), [`screenshot`](_static/examples/h2o-vibrations/output-h2o-vibrations-structure.png) | Analytic Hessian and ORCA-format Hessian export. |
| [`h2o-cube`](_static/examples/h2o-cube/artifact-status.json) | [`qvf`](_static/examples/h2o-cube/output-h2o-cube.qvf), [`density cube`](_static/examples/h2o-cube/output-h2o-density.cube), [`HOMO cube`](_static/examples/h2o-cube/output-h2o-mo-homo.cube), [`LUMO cube`](_static/examples/h2o-cube/output-h2o-mo-lumo.cube), [`MO stack`](_static/examples/h2o-cube/output-h2o-mos.cube) | Cube writer example plus a vibe-view archive carrying the same RHF state. |
| [`h2-tight-canonical-orth`](_static/examples/h2-tight-canonical-orth/artifact-status.json) | [`out`](_static/examples/h2-tight-canonical-orth/output-h2-tight-canonical-orth.out), [`qvf`](_static/examples/h2-tight-canonical-orth/output-h2-tight-canonical-orth.qvf), [`stdout`](_static/examples/h2-tight-canonical-orth/stdout.txt) | Diagnostic script prints its report to stdout and writes the default SCF archive. |
| [`h-chain-uniform`](_static/examples/h-chain-uniform/artifact-status.json) | [`out`](_static/examples/h-chain-uniform/output-h-chain-uniform.out), [`stdout`](_static/examples/h-chain-uniform/stdout.txt) | Legacy multi-k convergence scan; QVF is deliberately not expected. |
| [`h-chain-peierls`](_static/examples/h-chain-peierls/artifact-status.json) | [`out`](_static/examples/h-chain-peierls/output-h-chain-peierls.out), [`stdout`](_static/examples/h-chain-peierls/stdout.txt) | Legacy Peierls dimerisation scan; QVF is deliberately not expected. |
| [`h-chain-bands`](_static/examples/h-chain-bands/artifact-status.json) | [`plot`](_static/examples/h-chain-bands/output-h-chain-bands.png), [`stdout`](_static/examples/h-chain-bands/stdout.txt) | Legacy Hcore band/DOS figure; QVF is deliberately not expected. |
| [`madelung-constants`](_static/examples/madelung-constants/artifact-status.json) | [`out`](_static/examples/madelung-constants/output-madelung-constants.out), [`stdout`](_static/examples/madelung-constants/stdout.txt) | Legacy Ewald Madelung-constant table; QVF is deliberately not expected. |
| [`nacl-symmetry`](_static/examples/nacl-symmetry/artifact-status.json) | [`stdout`](_static/examples/nacl-symmetry/stdout.txt), [`stderr`](_static/examples/nacl-symmetry/stderr.txt) | Legacy symmetry-compression diagnostic; QVF is deliberately not expected. |
| [`nh3-umbrella-neb`](_static/examples/nh3-umbrella-neb/artifact-status.json) | [`out`](_static/examples/nh3-umbrella-neb/output-nh3-umbrella-neb.out), [`traj`](_static/examples/nh3-umbrella-neb/output-nh3-umbrella-neb.traj), [`qvf`](_static/examples/nh3-umbrella-neb/output-nh3-umbrella-neb.qvf), [`screenshot`](_static/examples/nh3-umbrella-neb/output-nh3-umbrella-neb-structure.png) | NH3 umbrella-inversion climbing-image NEB. |
| [`chi-ccm-b-qvf`](_static/examples/chi-ccm-b-qvf/artifact-status.json) | {download}`README <_static/examples/chi-ccm-b-qvf/README.md>`, [`H-chain QVF`](_static/examples/chi-ccm-b-qvf/output-chi-ccm-b-hchain-ri-n4-wannier.qvf), [`H2-pair QVF`](_static/examples/chi-ccm-b-qvf/output-chi-ccm-b-h2pair-3d-ri-n2-wannier.qvf), [`contact sheet`](_static/examples/chi-ccm-b-qvf/vibe-view-contact-sheet.png) | Curated chi-CCM-B periodic QVF validation fixtures with full `.out`, sanitized `.system`, and vibe-view captures. |
| [`mgo-route-gdf-bipole`](_static/examples/mgo-route-gdf-bipole/artifact-status.json) | {download}`README <_static/examples/mgo-route-gdf-bipole/README.md>`, [`GDF QVF`](_static/examples/mgo-route-gdf-bipole/output-mgo-rhf-sto3g-gdf-k222.qvf), [`BIPOLE QVF`](_static/examples/mgo-route-gdf-bipole/output-mgo-rhf-sto3g-bipole-k222.qvf), [`GDF density capture`](_static/examples/mgo-route-gdf-bipole/captures/mgo-gdf-density.png), [`BIPOLE structure capture`](_static/examples/mgo-route-gdf-bipole/captures/mgo-bipole-structure.png) | Historical MgO primitive RHF/STO-3G `2 x 2 x 2` GDF and symmetry-reduced BIPOLE route fixtures from `vq`. The committed BIPOLE QVF predates exact returned-density export and remains density-less until regeneration. The committed `.out` files also predate the `of which q=0 finite-size` exchange row (2026-08-28, #82), so their energy-components block has one row fewer than a current run's. |

## Running Examples

From the checkout root, use the project virtual environment:

```sh
.venv/bin/python examples/molecular/input-h2o-rhf.py
```

Many examples intentionally write beside themselves because that is
the most convenient pattern when the script is copied into a project
directory. If you run from the repository, remove the generated files
before committing:

```sh
find examples -type f \
  \( -name '*.out' -o -name '*.system' -o -name '*.molden' \
     -o -name '*.traj' -o -name '*.cube' -o -name '*.png' \
     -o -name '*.csv' -o -name '*.references' -o -name '*.bibtex' \
     -o -name '*.hess' -o -name '*.engrad' -o -name 'stdout.txt' \
     -o -name 'stderr.txt' -o -name '*.population.txt' \
     -o -name '*.population.json' \) -delete
```

## Molecular Examples

| Script | Generates | Expected result |
| --- | --- | --- |
| [`input-h2o-rhf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-rhf.py) | `output-h2o-rhf.*` | closed-shell H2O RHF / 6-31G* |
| [`input-h2o-dft.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-dft.py) | `output-h2o-dft.*` | H2O RKS-PBE / 6-31G* |
| [`input-h2-rks-skala.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2-rks-skala.py) | `output-h2-rks-skala.*` | H2 RKS / SKALA-1.1 / def2-SVP fixed-geometry single point; source only until a supported Linux bundle is published |
| [`input-oh-radical.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-oh-radical.py) | `output-oh-radical.*` | open-shell UHF doublet with alpha/beta orbital blocks |
| [`input-h2o-opt.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-opt.py) | `output-h2o-opt.*` | ASE/BFGS H2O relaxation |
| [`input-h2o-dimer-opt.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-dimer-opt.py) | `output-h2o-dimer-opt.*` | water-dimer relaxation |
| [`input-h2o-trimer-opt.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-trimer-opt.py) | `output-h2o-trimer-opt.*` | cyclic water-trimer relaxation |
| [`input-h2o-cube.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-cube.py) | `output-h2o-cube.*`, `output-h2o-*.cube` | QVF archive plus density, HOMO, LUMO, and multi-orbital cube files |
| [`input-h2o-vibrations.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/input-h2o-vibrations.py) | `output-h2o-vibrations.*` | analytic Hessian and ORCA-format Hessian export |

## Periodic Examples

| Script | Generates | Expected result |
| --- | --- | --- |
| [`input-h-chain-uniform.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-h-chain-uniform.py) | `output-h-chain-uniform.out` | k-mesh convergence table for a 1D H2 chain |
| [`input-h-chain-peierls.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-h-chain-peierls.py) | `output-h-chain-peierls.out` | Peierls dimerisation scan |
| [`input-h-chain-bands.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-h-chain-bands.py) | `output-h-chain-bands.png` | H-chain band structure and DOS figure |
| [`input-madelung-constants.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-madelung-constants.py) | `output-madelung-constants.out` | Ewald Madelung-constant table |
| [`input-nacl-sto3g-dft.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-nacl-sto3g-dft.py) | `output-nacl-sto3g-dft.*` | debug-friendly NaCl RKS-LDA periodic run |
| [`input-h2-cell-rks-skala-gdf.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-h2-cell-rks-skala-gdf.py) | `output-h2-cell-rks-skala-gdf.*` | experimental 3D H2 RKS / SKALA-1.1 / Gamma GDF smoke calculation; not a reference energy |
| [`input-xsf-bxsf-nacl.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/input-xsf-bxsf-nacl.py) | `output-nacl-xsf.*` | XSF/BXSF structure and band-grid export |

## ASE Workflows

| Script | Generates | Expected result |
| --- | --- | --- |
| [`optimize-via-ase-bfgs.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/ase_workflows/optimize-via-ase-bfgs.py) | trajectory, log, and convergence plot | ASE BFGS relaxation driven by `vibeqc.ase.VibeQC` |
| [`vibrations-via-ase-vibrations.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/ase_workflows/vibrations-via-ase-vibrations.py) | displacement cache and summary text | finite-difference vibrational analysis |
| [`nh3-neb-via-ase.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/ase_workflows/nh3-neb-via-ase.py) | MEP table, plot, and trajectory | ammonia umbrella-inversion NEB |
| [`md-water-nve.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/ase_workflows/md-water-nve.py) | time-series CSV, plot, and trajectory | NVE energy-conservation diagnostic |
| [`surface-h2-pt111-singlepoint.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/ase_workflows/surface-h2-pt111-singlepoint.py) | local Pt(111) + H2 XYZ fixture | periodic ASE geometry setup without launching SCF |

## Cross-Validation

The comparison examples in `examples/ase_compare/` and
`examples/regression/` are source drivers. They may create ORCA,
PySCF, CRYSTAL, GPAW, or CP2K logs when those programs are available,
but the generated logs are not committed. Keep the input decks and
small JSON/data summaries; regenerate bulky output locally when you
need to audit a number.

## Output File Reference

For the meaning of each output family, see
[Input scripts and output files](user_guide/output_files.md). For
run-directory hygiene, including why examples should normally be copied
outside the checkout before production work, see
[Good practices](good_practices.md).
