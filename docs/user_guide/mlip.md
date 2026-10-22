---
myst:
  html_meta:
    "description": "Run the ACEsuit MACE machine-learning interatomic potential in vibe-qc via method=\"mace\": model selection, the ASL license gate, the energy-scale caveat, and citations."
    "og:title": "vibe-qc, MACE machine-learning interatomic potential"
---

# Machine-learning interatomic potentials (MACE)

## Which MACE?

Two separate projects use this name:

| Project | Purpose | Relation to vibe-qc |
| --- | --- | --- |
| [ACEsuit MACE](https://github.com/ACEsuit/mace), installed as `mace-torch` | Machine-learned energy, forces and stress | Provides `method="mace"` and the optional `[mace]` extra |
| [Mendoza Automated Crystal Engine](https://www.egr.msu.edu/~jmendoza/code.html) | DFT workflow automation for CRYSTAL, including structure preparation, job management and analysis | Separate workflow application |

Mendoza's v1.1.5 announcement, browser GUI and `mace gui connect` command
refer to the workflow application. Our MLIP interface uses ACEsuit's
calculator API. Install the `mace-torch` distribution for this interface;
a package named `mace` is not a substitute. Keep applications that occupy
the same Python import namespace in separate environments.

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
attributes it as an external pre-trained model in the citation block;
the energy is MACE's, not a vibe-qc ab-initio total. Within its trained
domain it provides a fast model surface, but accuracy must be established
against an external observable for the system and protocol in question.
It gives no electronic structure (no orbitals / charges / gap) and only
covers its trained elements.

**Status.** The vibe-qc interface (registry, licence gate, molecular and
periodic drivers, provenance) is production. The model behind it is an
external pre-trained potential and not a vibe-qc method: vibe-qc computes
none of its numbers and cannot vouch for their accuracy beyond what the
model papers report. The label vocabulary is defined in
[Methods: theory and status](../methods.md#how-to-read-the-status-labels).

## Background: what MACE is

MACE ({vibeqc-cite}`batatia_mace_2022`) is a **message-passing neural
network interatomic potential**. The total energy is a sum of atomic site
energies,

$$
E = \sum_i E_i, \qquad E_i = E_i\big(\{\mathbf r_{ij}, Z_j : |\mathbf r_{ij}| < r_{\mathrm{cut}}\}\big),
$$

each a learned function of the atom's neighbourhood within a cutoff. Forces
and the stress are the analytic negative derivatives of $E$ with respect to
positions and strain, obtained by automatic differentiation, so they are
exactly consistent with the energy. Three ideas make MACE what it is.

*Equivariance.* Every internal feature is a set of irreducible
representations of $E(3)$ (scalars, vectors, rank-2 tensors, ...) built
from spherical harmonics of the neighbour directions and a learned radial
basis. Rotating or reflecting the input rotates the features accordingly and
leaves the energy unchanged, so symmetry is exact by construction rather
than learned from augmented data.

*Body order from the atomic cluster expansion.* The message an atom
receives in one layer is not a sum of pair terms but a many-body
expansion: MACE forms symmetric tensor products of the neighbour features
up to a correlation order $\nu$ (the published models use $\nu = 3$, so
one message already contains up to four-body information) using the
atomic cluster expansion, and two such layers are sufficient in practice.
This is why MACE reaches its accuracy with few layers and a modest
receptive field.

*Training data.* The parameters are fitted to DFT energies, forces and
stresses of large datasets. **MACE-MP-0** ({vibeqc-cite}`batatia_mace_mp_2024`)
is trained on MPtrj, the relaxation trajectories of the Materials Project
(PBE and PBE+U), and **MACE-MPA-0** on MPtrj plus a subsample of the
Alexandria database; both cover 89 elements and are released under the MIT
licence. **MACE-OFF23** ({vibeqc-cite}`kovacs_mace_off_2023`) is trained on
the SPICE dataset of organic molecules at the
$\omega$B97M-D3(BJ)/def2-TZVPPD level, covers H, C, N, O, F, P, S, Cl, Br
and I, and is released under the Academic Software License (academic,
non-commercial), which is why vibe-qc gates it. The absolute energy of each
family sits on the reference scale of its training functional minus a
per-element offset, hence the energy-scale caveat below.

What follows from this for a user: a MACE energy is a regression estimate
of the training functional on the training distribution. Inside that
distribution it is fast and typically close to the DFT it learned; outside
it (unusual oxidation states, charged or open-shell species, elements or
bonding motifs absent from the data, very short contacts) it fails without
warning, because there is no variational principle, no SCF residual and no
electronic structure to inspect. The registry exported by
`mace_model_registry()` records the training data and theory level of each
model so that this provenance travels with every result.
Single-point evaluation validates the backend energy, forces, and applicable
3D stress before returning or caching a result. Missing properties, wrong
array shapes, and non-finite values raise `ValueError`; backend exceptions
propagate. A failed evaluation cannot leave a partially accepted energy in
the cache. These checks validate the result contract, not model accuracy.

Molecular and periodic optimizers require a finite positive `fmax` and a
non-negative integer step budget (`max_opt_steps` for molecular `run_job`,
`max_steps` for periodic drivers). They reject invalid controls before loading
a model. Molecular `run_job` also rejects these controls before creating output.
A zero-step budget is allowed to assess the initial structure against the
requested force criterion; it does not imply convergence.

The fixed-cell optimizer's `gradient()` is the physical energy derivative
in Ha/bohr, including forces on frozen atoms. Its `max_force_eva` and
`converged` fields use the constrained forces, with frozen atoms excluded
from the stopping criterion.
`fixed_indices` requires integer atom indices; fractional values and strings
are rejected instead of converted. Both periodic optimizers check the final
energy and physical forces for finite values before returning. Cell
relaxation also checks the final stress.

MACE checks that coordinates have finite values and shape `(n_atoms, 3)`
and that a supplied cell has finite values and shape `(3, 3)` before
loading the backend. Explicit `pbc` masks contain three booleans or integer
0/1 flags. The model copies the supplied cell. A rejected structure update
in a reusable evaluator preserves the previous structure and cached result.
These checks validate input representation, not a geometry's physical
suitability for a trained model.

## Installation, the `[mace]` extra (Python ≤ 3.13)

MACE is an **optional** dependency (a heavy PyTorch + e3nn stack), not
part of the default install:

```sh
pip install 'vibe-qc[mace]'
```

The extra selects `mace-torch>=0.3.16,<0.4`. ACEsuit's
[supported 0.3.x line](https://github.com/ACEsuit/mace#project-status)
provides the calculator and serialized checkpoint APIs used here.
Its planned 1.0 rewrite changes those APIs and requires checkpoint conversion.
When installed `mace-torch` version metadata is available, other API series
raise an installation error before importing the ML stack or changing its
runtime environment. Source trees without distribution metadata still pass
through the calculator import checks. A missing stack or calculator reports
the ACEsuit package name and installation command, preserving the original
import failure as the exception cause.

> **Supported installation: Python ≤ 3.13.** The `[mace]` extra is marked
> `python_version < "3.14"` and installs no MACE stack on Python 3.14.
> If the backend is missing, `method="mace"` raises an installation error.
> A manually installed newer stack may import and run; this does not extend
> the supported interpreter range or qualify that stack for production.

Apple MPS requires `MLIPOptions(device="mps", dtype="float32")`.
vibe-qc loads and converts the checkpoint on CPU before transferring the
model to MPS; inference stays on the requested device. Invalid precision
choices and unavailable MPS devices fail explicitly.
MACE 0.3.16 has a separate forward-pass float64 cast that prevents MPS
inference even with float32 weights. MPS also requires an upstream build
containing [the node-energy fix, commit 22f0809](https://github.com/ACEsuit/mace/commit/22f0809735bd4dd1deba80cf8e16f89913a35ff4).
Installing the ordinary extra with MACE 0.3.16 does not satisfy that
requirement. The real MPS energy/force/stress regression remains active on
MPS hardware so this dependency limitation is not hidden.

The foundation-model **weights are fetched on demand** into the XDG cache
(`~/.cache/mace/`) on first use, vibe-qc bundles none of them.

Run the dependency-only health check before downloading weights:

```sh
python examples/mlip/00_runtime_healthcheck.py
```

For a reproducible runtime, record the VibeQC source SHA and version,
Python version, MACE/Torch/e3nn/ASE versions, device, dtype and model key.
The health-check example reports everything available before a model is
loaded. Loading snapshots the registered checkpoint and computes a SHA256
digest of the same bytes passed to PyTorch. `MACEModel.weight_identity` and
periodic result objects expose this digest, byte count, registered URL and
checkpoint loader. The `loader` model metadata names the upstream registry
selector; `checkpoint.loader` identifies VibeQC's snapshot loader.
The URL identifies the registry entry, not proof of the cached file's origin.
The digest identifies the serialized checkpoint, not subsequent in-memory
modifications or the scientific accuracy of its predictions. Upstream cache
names and the bundled MPA-0 preference are preserved; clearing a cache fetches
the registered weights again.

Molecular `run_job` records the executed model, device and dtype in
`.system` under `[run.mlip]` and in QVF `provenance.mlip`, together with
the loader, license, citation and energy scale. Its QVF job specification
preserves those runtime choices for replay, resolving aliases to the
canonical model name. `.out`, `.system` and QVF also record the checkpoint
identity. Replay supplies its digest as `MLIPOptions(weight_sha256=...)` and
rejects different bytes before deserialization. This option can also pin a
checkpoint for a new job. Older archives without a digest still load the
selected model, but cannot enforce checkpoint identity.
Output records do not copy academic-license acknowledgment;
replaying an OFF23 calculation still requires the caller's acknowledgment.
QVF provenance marks the forward pass as `scf_applicable=false` and omits
SCF convergence and iteration claims.

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
| Molecular or periodic reaction path | `run_neb(..., method="mace", mlip_options=...)` | fixed-model relative path energies and forces; preserves 1D/2D/3D periodic axes and skew cells |
| Bare ASE workflow | `mace_calculator(options)` | upstream ASE calculator in eV and Angstrom units |
| Periodic single point | `run_periodic_mace(system, options)` | 1D/2D/3D `.energy` in Ha and `.gradient()` in Ha/bohr; 3D-only `.stress()` in Ha/bohr³; runtime/model provenance |
| Sequential periodic scan | `PeriodicMACEEvaluator(options).run(system)` | same result, with one retained model and calculator |
| Fixed-cell periodic/slab relaxation | `optimize_periodic_mace_positions(system, options, fixed_indices=...)` | result containing the relaxed `PeriodicSystem`, convergence, forces, and provenance |
| 3D periodic cell relaxation | `optimize_periodic_mace_cell(system, options, fmax=...)` | converged 3D `PeriodicSystem`; raises on step exhaustion |
| Registry export | `mace_model_registry()` / `save_mace_model_registry(path)` | dependency-free provenance metadata |

`MACEModel` is the lower-level molecular wrapper behind these APIs.
Most user workflows should prefer `run_job` so output manifests,
citations, and provenance are written automatically.
`MACEModel` and `PeriodicMACEEvaluator` copy the supplied `MLIPOptions` when
constructed. Later edits to those options do not change the loaded model or
its reported device and precision. `MACEModel.options` returns a detached copy;
construct a new model to use different runtime settings.

Molecular `run_job(..., method="mace", optimize=True)` retains the optimizer's
loaded model for the final energy and gradient. The reported result and its
checkpoint digest therefore describe the same potential as the trajectory,
even if the cached checkpoint or caller's options change during optimization.

MACE reaction-path QVF archives mark SCF as inapplicable and omit SCF
convergence and iteration counts. Path convergence is the separate NEB force
criterion exposed by `NEBResult.converged`.
The results of `run_neb`, `run_dimer`, and `run_irc` with `method="mace"`
retain `model_info`, `device`, `dtype`, and `weight_identity` from the loaded
potential. `NEBResult.write_qvf` includes these in `provenance.mlip`, including
the checkpoint digest and model license. Changing the caller's options after
loading does not relabel a path. Reaction-path archives do not currently
provide a replayable NEB job specification.

`run_periodic_job(method="mace")` is not supported. It is an SCF runner;
use the direct periodic MACE APIs above.

Variable-cell relaxation checks ASE's combined atomic and cell-force criterion.
Zero atomic forces alone do not qualify a strained cell. If the step budget
is exhausted, `optimize_periodic_mace_cell` raises
`PeriodicMACECellOptimizationError` (a `RuntimeError`), preserving the last
`PeriodicSystem` in `error.system`, the completed and allowed step counts in
`error.n_steps` and `error.max_steps`, and the terminal cell-filter force norm
and requested threshold in `error.final_max_force` and `error.target_max_force`.
Those last two values use ASE's filter scaling, not raw physical stress units.
An already converged starting structure may return with `max_steps=0`.
Both the relaxed structure and `error.system` retain the input charge and
spin multiplicity for subsequent electronic-structure calculations. These
labels do not change the neutral MACE potential or suppress its charge/spin
warning.
Optimizer convergence does not establish dynamical stability or predictive
accuracy of the selected model.

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
gradient. Each image must return a finite scalar energy and a finite force
vector for every atom. Invalid model output stops the calculation before it
can enter the band or its optimizer state. See [`neb.md`](neb.md) § MACE backend.

`run_dimer(..., method="mace")` uses the same checked energy and force
evaluations for a saddle search from one starting structure. For periodic
systems, every evaluation retains the input's 1D, 2D or 3D periodic axes
and cell, including skew cells.
Force tolerances (`conv_tol_force`, `rotation_force_tol`), `dimer_separation`,
`max_step`, and `initial_step` must be finite and positive. Invalid values
are rejected before loading a model or evaluating forces; an infinite force
threshold cannot certify a nonstationary structure as a converged saddle.
The positive integer `max_iter` limits evaluated dimer centres. When that
budget is exhausted, the result keeps the last evaluated geometry, its
energy and force residual, and `converged=False`; it does not return an
unevaluated final displacement.

`run_irc(..., method="mace")` follows molecular paths from an explicit
`transition_mode`, for example a dimer mode. `step_size` and `conv_tol_grad`
must be finite and positive; `max_points` must be a nonnegative integer.
Invalid controls are rejected before loading a model. Each branch evaluates
the initial displacement and allows up to `max_points` further step attempts,
including rejected trials. The convergence flag reflects the maximum absolute
Cartesian gradient at the last accepted point, including a final accepted
step. Repeated rejected steps cannot certify convergence. A stalled or
exhausted branch retains its accepted path and energies, with its convergence
flag false unless the endpoint satisfies `conv_tol_grad` in Ha/bohr. Check
that flag before treating `forward_minimum` or `reverse_minimum` as a minimum
or interpreting the corresponding barrier as a converged result.

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
