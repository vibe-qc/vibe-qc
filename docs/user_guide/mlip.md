---
myst:
  html_meta:
    "description": "Run the ACEsuit MACE machine-learning interatomic potential in vibe-qc via method=\"mace\": model selection, the ASL license gate, the energy-scale caveat, and citations."
    "og:title": "vibe-qc, MACE machine-learning interatomic potential"
---

# Machine-learning interatomic potentials (MACE)

vibe-qc can run [ACEsuit **MACE**](https://github.com/ACEsuit/mace), a
pre-trained *E(3)*-equivariant graph-neural-network **interatomic
potential**, as a first-class method:

```python
from vibeqc import Molecule
from vibeqc.runner import run_job

mol = Molecule.from_xyz("h2o.xyz")
run_job(mol, method="mace", output="h2o_mace")
```

MACE predicts energy, forces, and stress on a DFT-fitted potential-energy
surface, it does **not** solve the Schrödinger equation and builds no
wavefunction. vibe-qc drives MACE's pre-trained forward pass and
attributes it as an external pre-trained model (see `CLAUDE.md` § 10);
the energy is MACE's, not a vibe-qc ab-initio total. Within its trained
domain it provides a fast model surface, but accuracy must be established
against an external observable for the system and protocol in question.
It gives no electronic structure (no orbitals / charges / gap) and only
covers its trained elements.

## Installation, the `[mace]` extra (Python ≤ 3.13)

MACE is an **optional** dependency (a heavy PyTorch + e3nn stack), not
part of the default install:

```sh
pip install 'vibe-qc[mace]'
```

> **Python ≤ 3.13 only.** MACE's `matscipy` dependency has no Python 3.14
> wheel yet, so the `[mace]` extra is marked `python_version < "3.14"` and
> the runtime import is gated. On Python 3.14, `method="mace"` raises a
> clear error telling you to use a ≤ 3.13 environment.

The foundation-model **weights are fetched on demand** into the XDG cache
(`~/.cache/mace/`) on first use, vibe-qc bundles none of them.

Run the dependency-only health check before downloading weights:

```sh
python examples/mlip/00_runtime_healthcheck.py
```

For a reproducible runtime, record the VibeQC source SHA and version,
Python version, MACE/Torch/e3nn/ASE versions, device, dtype, model key,
loader, cache path, and weight-file digest. The health-check example
reports everything available before a model is loaded. A calculation
manifest should add the resolved weight path and digest after first use.
Moving or clearing the cache causes the upstream loader to fetch weights
again; it does not change the model key recorded by vibe-qc.

## Selecting a model, `MLIPOptions`

The default is the MIT-licensed **MACE-MPA-0** (materials). Pass an
`MLIPOptions` to choose another registered model, the device, or the
dtype. Organic-family work uses MACE-OFF23 and must acknowledge its
license explicitly:

```python
from vibeqc import Molecule, run_job
from vibeqc.mlip import MLIPOptions

ethanol = Molecule.from_xyz("ethanol.xyz")
options = MLIPOptions(
    model="off23-medium",
    accept_academic_license=True,
    device="cpu",
    dtype="float64",
)
run_job(
    ethanol,
    method="mace",
    mlip_options=options,
    output="ethanol_off23",
)
```

| Model key | Family | Domain | License |
|-----------|--------|--------|---------|
| `medium-mpa-0` (default), `small`, `medium`, `large` | MACE-MP / MPA | materials (89 elements) | **MIT** |
| `off23-small`, `off23-medium`, `off23-large` | MACE-OFF23 | organic (H, C, N, O, F, P, S, Cl, Br, I) | **ASL** |

Aliases (`mpa-0`, `off23`, …) resolve to the canonical keys; see
`vibeqc.mlip.MACE_MODELS`. The supported public scope contains only
registered MACE-MP/MPA-0 and MACE-OFF23 keys. OMAT, MATPES, MH, MDP,
arbitrary URLs, local weight paths, and other unregistered weights are
outside this scope.

Unknown keys are rejected before the MACE backend imports. A license
acknowledgement does not make an unregistered model supported or
scientifically appropriate.

The registry is available without importing Torch or MACE:

```python
from vibeqc.mlip import mace_model_registry, save_mace_model_registry

registry = mace_model_registry()
save_mace_model_registry("mace-models.toml")
```

The exported rows contain the exact model key, loader and loader
argument, license, domain, element coverage, training data, theory,
citation key, and DOI.

## The ASL license gate

MACE's *code* is MIT, but the foundation-model *weights* are licensed
separately. The materials line (MP-0 / MPA-0) is **MIT**, free for
commercial and academic use. The organic **MACE-OFF23** model is under the
[Academic Software License (ASL)](https://github.com/gabor1/ASL),
**academic, non-commercial use only**.

Selecting an ASL model raises `PermissionError` unless you acknowledge
the academic license, `MLIPOptions(accept_academic_license=True)` or the
environment variable `VIBEQC_ACCEPT_ASL=1`. MIT models are never gated.
This prevents a commercial workflow from silently pulling academic-only
weights. Full licensing detail: [license.md](../license.md).

## Energy-scale caveat

A `method="mace"` energy is on a **model-specific reference scale** (each
model subtracts its own per-element atomic energies), **not** a vibe-qc
total electronic energy, and **not comparable across models**. Two MACE
families can assign very different absolute values to the same H₂O
geometry while both provide internally consistent forces.

Use MACE energies for **relative** energetics (geometry optimization,
reaction energies at fixed composition + model, MD), not as absolute
totals, and never mix them with vibe-qc SCF or semiempirical totals.
For method comparisons, normalize each fixed-composition curve to its
own reference point, then compare curve shapes or externally referenced
observables. The `.out` prints a provenance block naming the model, its
license, and this caveat.

## Supported public APIs

| Task | Public API | Result and units |
|---|---|---|
| Molecular single point | `run_job(molecule, method="mace", mlip_options=...)` | `.energy` in Ha and `.gradient()` in Ha/bohr; writes provenance and citations |
| Molecular optimization | `run_job(..., method="mace", optimize=True)` | optimized structure and ASE trajectory |
| Molecular reaction path | `run_neb(..., method="mace", mlip_options=...)` | fixed-model relative path energies and forces |
| Bare ASE workflow | `mace_calculator(options)` | upstream ASE calculator in eV and Angstrom units |
| Periodic single point | `run_periodic_mace(system, options)` | 1D/2D/3D `.energy` in Ha and `.gradient()` in Ha/bohr; 3D-only `.stress()` in Ha/bohr³; runtime/model provenance |
| Sequential periodic scan | `PeriodicMACEEvaluator(options).run(system)` | same result, with one retained model and calculator |
| Fixed-cell periodic/slab relaxation | `optimize_periodic_mace_positions(system, options, fixed_indices=...)` | result containing the relaxed `PeriodicSystem`, convergence, forces, and provenance |
| 3D periodic cell relaxation | `optimize_periodic_mace_cell(system, options, fmax=...)` | relaxed 3D `PeriodicSystem` |
| Registry export | `mace_model_registry()` / `save_mace_model_registry(path)` | dependency-free provenance metadata |

`MACEModel` is the lower-level molecular wrapper behind these APIs.
Most user workflows should prefer `run_job` so output manifests,
citations, and provenance are written automatically.

`run_periodic_job(method="mace")` is not supported. It is an SCF runner;
use the direct periodic MACE APIs above.

## macOS note

On macOS, vibe-qc's native core and PyTorch each link an OpenMP runtime;
to avoid an abort (OMP Error #15) vibe-qc auto-sets
`KMP_DUPLICATE_LIB_OK=TRUE` for the MACE run, with a one-time warning. It
also caps the MACE process to one OpenMP/BLAS thread by default, because
duplicate OpenMP runtimes can still segfault when both sides create worker
pools. The MACE forward pass is verified to produce identical energies
with the duplicate-runtime workaround set. To opt into a different MACE
thread count, set `VIBEQC_MACE_OPENMP_THREADS=N` before importing vibe-qc.

## Citations

Every `method="mace"` run auto-emits the MACE method paper (Batatia et
al. 2022) and the foundation-model paper for the model used (MACE-MP →
Batatia et al. 2024; MACE-OFF23 → Kovács et al. 2023) to the `.bibtex` /
`.references` siblings. **Cite them in published work.** PyTorch / e3nn
are infrastructure (their versions appear in the banner + `.system`
manifest), not cited.

## Visualization

A `method="mace"` run writes the same structural output as any vibe-qc job,
so the geometry, and, for `optimize=True`, the relaxation trajectory,
visualize out of the box:

- **MolTUI** (terminal viewer; the `[viewer]` extra) renders the `.xyz`
  geometry: `moltui output-h2o-mace.xyz`.
- **[vibe-view](vibe_view.md)** (GPU 3D viewer, installed from its separate repository)
  opens the `.qvf` archive written with `output_qvf=True`, which carries the
  structure, the optimization-trajectory animation, and the citations:
  `vibe-view open output.qvf`.

There is **no orbital / density / band visualization**. MACE produces no
wavefunction or AO density (it is a learned potential, not an SCF), so the
default capability-aware output plan contains neither `.molden` nor
`.population.{txt,json}` for `method="mace"`. Explicitly requesting either
with `write_molden_file=True` or `write_population_file=True` raises before
model loading; `None` (the default) records them as inapplicable by omitting
the corresponding guaranteed plan rows.

## Implementation, runtime, dispatch, and evidence

Keep these five statements separate:

1. **Implemented in VibeQC.** The model registry, license gate, molecular
   wrapper, periodic direct driver, reusable periodic evaluator, stress,
   cell relaxation, and citation plumbing exist in the source tree.
2. **Runtime available.** Real calculations additionally require Python
   3.13 or earlier plus the `[mace]` extra and accessible model weights.
   A normal Python 3.14 VibeQC runtime does not satisfy this condition.
3. **Molecular dispatch supported.** `run_job(..., method="mace")` handles
   molecular single points and optimization; `run_neb(method="mace")`
   handles molecular reaction paths.
4. **Direct periodic support.** `run_periodic_mace`,
   `PeriodicMACEEvaluator`, `optimize_periodic_mace_positions`, and
   `optimize_periodic_mace_cell` are public. Periodic MACE is not routed
   through `run_periodic_job`.
5. **Scientifically validated landed evidence.** Import checks, wrapper
   parity with the upstream calculator, finite energies, symmetric
   forces, stress parity, and smoke relaxations establish execution and
   unit conversion. They do not establish reference-EOS accuracy or
   general model accuracy. Those claims need external comparisons.

## Limitations

- **Molecular** single-point energies + forces and **geometry
  optimization** (`optimize=True`) work today.
- **Periodic** energy and forces support 1D, genuine 2D slabs, and 3D through
  direct drivers. Fixed-cell position optimization supports frozen atom
  indices, including `SlabInfo.bottom_layer_indices(n)`. Stress and
  variable-cell relaxation are intentionally 3D-only: a true 2D slab's
  third lattice vector is non-physical bookkeeping, so reporting a
  volume-normalized stress or relaxing that vector would be misleading.

  ```python
  from vibeqc.mlip.mace import (
      PeriodicMACEEvaluator,
      optimize_periodic_mace_cell,
      optimize_periodic_mace_positions,
      run_periodic_mace,
  )

  res = run_periodic_mace(system)        # .energy (Ha), .gradient() (Ha/bohr),
                                         # .stress() (Ha/bohr³, 3×3)
  relaxed = optimize_periodic_mace_cell(system, fmax=0.05)   # variable-cell
  slab_opt = optimize_periodic_mace_positions(
      slab,
      fmax=0.05,
      fixed_indices=slab_info.bottom_layer_indices(2),
  )
  relaxed_slab = slab_opt.system

  # A fixed-model scan loads the model once and reuses its calculator.
  evaluator = PeriodicMACEEvaluator()
  scan_results = [evaluator.run(point) for point in strained_systems]
  ```

  (`method="mace"` is not yet wired into `run_periodic_job`; use these
  drivers directly.) A `PeriodicMACEEvaluator` is sequential; create one per
  worker when scan points run in parallel.
- No electronic structure (orbitals / charges / gap).
- Trained-domain only, fails out-of-distribution.
- Charge and spin do not select an electronic state. Non-neutral and
  open-shell inputs warn because the learned neutral potential ignores
  those quantum numbers.
- A zero return, finite energy, or low force at one symmetric geometry is
  not an EOS or accuracy validation.
- Startup, wall time, peak RSS, device, dtype, and model-reload behavior
  are diagnostics. Record discrepancies rather than turning them into
  scientific claims.

## Examples

Runnable scripts (require the `[mace]` extra, Python ≤ 3.13) live in
[`examples/mlip/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/mlip):

- `00_runtime_healthcheck.py`, dependency, source, registry, API, and cache
  preflight without loading model weights.
- `01_water_mace.py`, molecular single-point + geometry optimization on H₂O.
- `02_periodic_mace_silicon.py`, periodic energy/forces/stress + variable-cell
  relaxation on bulk Si, with a reusable two-point strain probe by default.
- `03_neb_mace.py`, NH₃ umbrella-inversion reaction path via
  `run_neb(method="mace")` (climbing-image NEB on the MACE potential).
- `04_surface_adsorption.py`, H₂O + NH₃ adsorption energies on MgO,
  CaO, Al₂O₃, TiO₂, Pt, and Cu slabs with MACE-MPA-0 as an exploratory
  protocol, not reference adsorption data.
- `05_mace_pm6_relative_curve.py`, independently normalized MACE and PM6
  H₂O stretch curves for a safe cross-family comparison.
- `06_optimize_water_off23_for_dlpno.py` and
  `07_dlpno_ccsdt_on_mace_geometry.py`, a two-runtime molecular geometry
  handoff from explicitly acknowledged MACE-OFF23 to DLPNO-CCSD(T).
- `08_relax_h2o_hbn_then_periodic_dft.py`, a genuine 2D H2O/h-BN
  fixed-cell relaxation with a frozen sheet, followed optionally by a
  periodic PBE single point on the relaxed structure.

MACE also drives **reaction-path searches**: `run_neb(...,
method="mace")` runs a Nudged Elastic Band on the MACE potential,
analytic forces, no SCF, and (for periodic bands) no finite-difference
gradient. See [`neb.md`](neb.md) § MACE backend.

See also: [`ase_integration.md`](ase_integration.md),
[`external_codes.md`](external_codes.md), and the
[roadmap](../roadmap.md) § MACE MLIP interface.

## Scope relative to upstream MACE

vibe-qc is **not feature-par with the whole upstream MACE project**, and is
not intended to be its training frontend. It exposes a license-reviewed,
reproducible inference subset:

| Capability | Upstream MACE | vibe-qc wrapper |
|---|---|---|
| Registered foundation-model inference | yes | yes: MP/MPA-0 and OFF23 only |
| Molecular energy, forces, optimization | yes through ASE | yes |
| Periodic bulk energy, forces, stress | yes | yes; stress is 3D-only |
| True slab energy, forces, fixed-cell optimization | yes through ASE PBC/constraints | yes; exact 2D PBC and frozen indices |
| Training, fine-tuning, custom checkpoints | yes | no |
| MD, descriptors, Hessians, per-atom outputs, committees | yes | use the bare upstream calculator where scientifically appropriate; no first-class vibe-qc result contract |
| Charge-aware/polar/dipole models and newer foundation families | available upstream | outside the registered scope |

This distinction matters for surface work. The MACE-MP foundation-model
paper includes interfaces and catalysis examples, but also reports systems
with material deviations from the reference DFT and cases where fine-tuning
was needed. A relaxed foundation-model structure is therefore a candidate
geometry to validate with an electronic method, not proof of reference-level
surface accuracy. See Batatia et al.,
[MACE method](https://doi.org/10.48550/arXiv.2206.07697) and
[MACE-MP foundation model](https://doi.org/10.1063/5.0297006) (*J. Chem.
Phys.* **163**, 184110 (2025)), and Kovács et al.,
[MACE-OFF](https://doi.org/10.1021/jacs.4c07099) (*J. Am. Chem. Soc.*
**147**, 17598 (2025)); the emitted `.bibtex` cites these versions of
record, not the earlier arXiv preprints.
