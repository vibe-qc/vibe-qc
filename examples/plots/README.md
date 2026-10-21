# Plot-generation scripts

Each script in this directory regenerates one figure or animation for
the tutorial that walks through it. Outputs land at
`docs/_static/plots/<name>.png` (or `.gif`) and are referenced from
`docs/tutorial/*.md` via standard MyST image syntax:

```markdown
![Caption](../_static/plots/<name>.png)
```

These scripts write their figure or animation into `docs/_static/plots/`,
not next to themselves. Most compute with library calls and write nothing
else. The exceptions: `water-dimer-d3-curve.py` calls `run_job` and writes
`water-dimer-d3-curve.*` (`.out`, `.system`, `.qvf` and companions) beside
itself, `h2o-opt-trajectory-animation.py` runs
`examples/molecular/input-h2o-opt.py` as a subprocess so that example's
output lands in `examples/molecular/`, and `bond-orbitals.py` works under
`/tmp/bond-orbitals` (override with `BOND_VIZ_WORKDIR`). Remove any
generated files before committing.

## Requirements

* `matplotlib` for every script except `bond-orbitals.py`. It is not
  declared in `pyproject.toml`; install it into the environment yourself
  (`pip install matplotlib`).
* `ase` (a core dependency of vibe-qc) for the trajectory readers.
* The three animations use matplotlib's `PillowWriter` (matplotlib 3.7
  or newer).
* `bond-orbitals.py` renders through the real vibe-view viewport in a
  headless browser, so it needs the separately installed viewer, Pillow
  and Playwright (with a browser installed) instead of matplotlib; it is
  the only script here that does.

## Convention

* One script per figure, named to match the figure file.
* Scripts are idempotent: re-running regenerates the file (modulo
  non-determinism in the underlying calculation, which is pinned where
  possible).
* Figures are 300 DPI PNG, about 5 to 6 inches wide, sized to read
  cleanly at the docs page width.
* Each script prints what it wrote and any sanity-check numbers (barrier
  height, peak position, fit residual) so you see at a glance whether the
  underlying calculation looks right.

## Running

Most scripts re-run the small calculation they plot. Two read the output
of a parent example and raise with a directive message if it is missing:

```sh
.venv/bin/python examples/workflows/input-nh3-umbrella-neb.py   # produces the .traj
.venv/bin/python examples/plots/nh3-umbrella-neb-mep.py         # reads it, writes the .png
```

## Catalog

| Script | Reads or re-runs | Writes | Embedded in |
|---|---|---|---|
| `nacl-ewald-alpha-invariance.py` | Ewald alpha sweep for the NaCl Madelung constant (under 1 s) | `nacl-ewald-alpha-invariance.png` | [Madelung with Ewald](../../docs/tutorial/madelung_with_ewald.md) |
| `h2o-opt-trajectory-animation.py` | runs `examples/molecular/input-h2o-opt.py` and reads its `.traj` (~10 s) | `h2o-opt-trajectory-animation.gif` | [Geometry optimisation](../../docs/tutorial/geometry_optimization.md) |
| `water-vibrational-frequencies.py` | HF/STO-3G optimisation + ASE finite-difference Hessian (~30 s) | `water-vibrational-frequencies.png` | [Vibrational frequencies](../../docs/tutorial/vibrational_frequencies.md) |
| `water-normal-mode-animation.py` | HF/STO-3G optimisation + analytic RHF Hessian, 60 frames (~30 s) | `water-normal-mode-animation.gif` | [Vibrational frequencies](../../docs/tutorial/vibrational_frequencies.md) |
| `water-frontier-orbitals.py` | HF/6-31G* single point, orbitals evaluated with `evaluate_ao` (~1 s) | `water-frontier-orbitals.png` | [Orbital visualisation](../../docs/tutorial/orbital_visualization.md) |
| `bond-orbitals.py` | RHF on H2, ethylene and acetylene, then screenshots of headless vibe-view | `bond-orbitals.png` | [Orbital visualisation](../../docs/tutorial/orbital_visualization.md) |
| `h-chain-bands-dos.py` | Hcore bands and DOS of the 1-D H2 chain (~1 s) | `h-chain-bands-dos.png` | [Band structure](../../docs/tutorial/band_structure.md) |
| `h-chain-crystalline-orbitals.py` | Hcore Bloch orbitals at Gamma and X (~2 s) | `h-chain-crystalline-orbitals.png` | [Band structure](../../docs/tutorial/band_structure.md) |
| `water-basis-convergence.py` | HF on cc-pVDZ, cc-pVTZ, cc-pVQZ plus the Helgaker extrapolation (~30 s) | `water-basis-convergence.png` | [Basis convergence](../../docs/tutorial/basis_convergence.md) |
| `water-dimer-d3-curve.py` | PBE/6-31G* dimer scan with and without D3(BJ) (~2 min) | `water-dimer-d3-curve.png` | [Dispersion](../../docs/tutorial/dispersion.md) |
| `water-functional-comparison.py` | HF, SVWN, PBE, BLYP, B3LYP, PBE0 on water (~5 s) | `water-functional-comparison.png` | [Functional comparison](../../docs/tutorial/functional_comparison.md) |
| `h-chain-peierls-energy.py` | the Peierls SCF scan on a finer delta grid (~30 s) | `h-chain-peierls-energy.png` | [Peierls](../../docs/tutorial/peierls.md) |
| `h-chain-peierls-bands.py` | Hcore bands at delta = 0 and 1.1 bohr (~1 s) | `h-chain-peierls-bands.png` | [Peierls](../../docs/tutorial/peierls.md) |
| `h-chain-peierls-animation.py` | the Peierls scan, animated frame by frame | `h-chain-peierls-animation.gif` | [Peierls](../../docs/tutorial/peierls.md) |
| `openmp-scaling.py` | one RKS/PBE glycine/cc-pVDZ SCF at 1, 2, 4, 8 threads (under a minute) | `openmp-scaling.png` | [Parallel execution](../../docs/tutorial/parallel_execution.md) |
| `nh3-umbrella-neb-mep.py` | `examples/workflows/output-nh3-umbrella-neb.traj` (run the NEB first) | `nh3-umbrella-neb-mep.png` | [NEB reaction path](../../docs/tutorial/neb_reaction_path.md) |
| `nh3-umbrella-neb-animation.py` | the same `.traj`, ping-pong through the seven images | `nh3-umbrella-neb-animation.gif` | [NEB reaction path](../../docs/tutorial/neb_reaction_path.md) |
| `natural-orbital-occupations.py` | H2O RHF and O2 triplet UHF at 6-31G*, natural-orbital analysis | `natural-orbital-occupations.png` | [Natural orbitals](../../docs/tutorial/natural_orbitals.md) |
| `lih-chain-pdos.py` | Hcore bands + projected DOS of a 1-D LiH chain (under 1 s) | `lih-chain-pdos.png` | [Projected DOS](../../docs/tutorial/pdos.md) |
| `lih-chain-bloch-density.py` | Bloch-orbital densities of the LiH chain at Gamma and X | `lih-chain-bloch-density.png` | [Periodic orbital cubes](../../docs/tutorial/periodic_orbital_cubes.md) |
| `periodic-becke-weights.py` | molecular against periodic Becke weights on cubic H2 cells, a = 4 to 20 bohr | `periodic-becke-weights.png` | [Tight-cell DFT](../../docs/tutorial/tight_cell_dft.md) |
| `level-shift-scf-trace.py` | three Gamma Ewald SCFs on a tight LiH cell with level shifts 0.0, 0.3, 0.7 Ha | `level-shift-scf-trace.png` | [Periodic SCF convergence](../../docs/tutorial/periodic_scf_convergence.md) |
