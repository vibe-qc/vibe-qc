# MACE examples

These scripts exercise vibe-qc's optional ACEsuit MACE interface. They
require Python 3.13 or earlier and a source install with the `mace` extra:

```sh
python3.13 -m venv .venv-mace
.venv-mace/bin/pip install -e '.[mace]'
.venv-mace/bin/python examples/mlip/00_runtime_healthcheck.py
```

The first model use downloads its weights into the user's MACE cache.
vibe-qc does not bundle model weights.

For a managed development runtime that should follow the latest clean
`main`, strengthen the health check with both operational gates:

```sh
.venv-mace/bin/python examples/mlip/00_runtime_healthcheck.py \
    --require-main-current --require-asl-unset
```

`--require-main-current` requires a clean `main` checkout whose `HEAD`
matches its locally fetched `origin/main`. `--require-asl-unset` prevents an
environment-wide MACE-OFF23 license acknowledgement. The runtime may move as
`main` moves; record the exact source SHA in each job receipt or calculation
artifact instead of freezing the development environment to one commit.

For VQ, register the development program with `branch = "main"`,
`auto_update_policy = "branch"`, the normal `.[mace]` update command, and no
`expected_git_sha` or `expected_import_version`. Poll it from an operator
timer with:

```sh
vq admin auto-update vibeqc-mace-dev localhost
```

Branch policy selects what the poller follows; it is not itself a background
timer. When drift is found, current VQ submits a supervised `build-env` job.
Every calculation still receives the exact source SHA observed at submission
as its immutable runtime receipt, so a later development update cannot
silently change an already queued job.

## Scripts

| Script | Purpose |
| --- | --- |
| `00_runtime_healthcheck.py` | Verify Python, VibeQC, ASE, Torch, e3nn, MACE, the supported public API, registry metadata, and the expected cache root without loading model weights. |
| `01_water_mace.py` | Molecular H2O single point and geometry optimization through `run_job(..., method="mace")`. |
| `02_periodic_mace_silicon.py` | Direct periodic bulk-Si energy, forces, stress, a compact strain probe with one retained model, and optional variable-cell relaxation. |
| `03_neb_mace.py` | Molecular NH3 umbrella-inversion NEB with MACE forces. |
| `04_surface_adsorption.py` | Exploratory adsorption probes with the materials model. These are protocol examples, not reference adsorption energies. |
| `05_mace_pm6_relative_curve.py` | A paper-safe H2O stretch comparison: each MACE and PM6 curve is normalized to its own minimum before curve shapes are compared. |
| `06_optimize_water_off23_for_dlpno.py` | Stage 1 of the molecular handoff: explicitly acknowledged ASL MACE-OFF23 geometry optimization. |
| `07_dlpno_ccsdt_on_mace_geometry.py` | Stage 2: DLPNO-CCSD(T) energy plus RHF-reference electronic properties on the MACE geometry. |
| `08_relax_h2o_hbn_then_periodic_dft.py` | True 2D H2O/h-BN relaxation with a frozen substrate, then optional periodic PBE. |

Run an example from the repository root:

```sh
.venv-mace/bin/python examples/mlip/02_periodic_mace_silicon.py
```

## Supported model scope

- `medium-mpa-0` is the default MIT-licensed materials model.
- The registered MACE-MP/MPA-0 keys use the `mace_mp` loader.
- The registered MACE-OFF23 keys use the `mace_off` loader and require
  explicit Academic Software License acknowledgement for academic,
  non-commercial use.
- Unregistered weights, arbitrary URLs, OMAT, MATPES, MH, and MDP are
  rejected outside the documented vibe-qc MACE scope.

Do not set `VIBEQC_ACCEPT_ASL` globally merely to make an example run.
Pass `MLIPOptions(accept_academic_license=True)` only for a specific
OFF23 calculation after confirming that its license covers the use.

## Scientific interpretation

MACE energies use a model-specific reference scale. Never compare their
absolute values across MACE models or directly with PM6, DFTB, MSINDO,
GFN2-xTB, HF, or DFT totals. Valid comparisons include:

- relative energies within one fixed model and composition,
- geometries and force directions,
- stress or strain trends from one fixed periodic model,
- externally referenced observables with a documented protocol.

A zero force or stress at one symmetric structure is a smoke result, not
an equation-of-state validation. Performance, startup, memory use, and
convergence behavior are diagnostics and must not be presented as
accuracy evidence.

For true `dim=2` slabs, only energy and forces are exposed. The synthetic
third bookkeeping lattice vector has no physical volume, so slab stress and
cell relaxation fail closed. Use `optimize_periodic_mace_positions` with
frozen bottom-layer indices, then pass its returned `PeriodicSystem` to a
periodic HF/DFT single point.
