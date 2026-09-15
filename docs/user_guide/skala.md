---
myst:
  html_meta:
    "description": "Run the Microsoft SKALA-1.1 neural exchange-correlation functional in vibe-qc for molecular RKS, UKS, ROKS, and experimental periodic AO-grid calculations."
    "og:title": "vibe-qc, Microsoft SKALA-1.1 neural XC"
---

# Microsoft SKALA-1.1 neural XC

vibe-qc can evaluate the official Microsoft SKALA-1.1 neural
exchange-correlation checkpoint through the ordinary molecular and periodic
Kohn-Sham drivers. The adapter is independent of the upstream `skala` Python
package and PySCF. It uses a generic differentiable full-grid XC boundary, so
the same model can be combined with different Coulomb builders, Gaussian
bases, spin treatments, and periodic boundary conditions where the route uses
the shared AO-grid XC projector.

```{admonition} Current scope
:class: important

SKALA currently provides self-consistent XC energies and potentials for
fixed-geometry single points. Molecular RKS, UKS, and ROKS are available.
Periodic support is experimental and limited to the routes listed below.
Analytic forces, geometry or cell optimization, stress, Hessians, TDDFT, and
Newton/TRAH response paths are not available and fail closed.
```

## At a glance

| Item | Current contract |
|---|---|
| Functional name | `skala-1.1` |
| Accepted aliases | `skala`, `skala-1.1-rev1` |
| Molecular methods | RKS, UKS, ROKS fixed-geometry single points |
| Periodic methods | Selected AO-grid RKS/UKS routes; experimental |
| Grid | Exact PySCF 2.14 level-3 atomic-grid protocol selected by vibe-qc |
| Elements | H through Lr, Z=1 through 103, as limited by the pinned parity-profile table |
| Runtime | CPU PyTorch 2.12 or 2.13 on Python 3.11 through 3.13 |
| Checkpoint | Downloaded on demand from an immutable revision and SHA-256 verified |
| Molecular dispersion | Not automatic; request the checkpoint-recommended D3(BJ) setting with `dispersion="b3lyp5"` |

## Release availability

| Version | SKALA availability |
|---|---|
| v0.15.158 | Molecular RKS/UKS/ROKS single points and selected experimental periodic AO-grid routes; GPW/GAPW and CCM still gated |
| v0.15.160 | Adds the qualified GPW/GAPW and CCM routes in the periodic matrix below |
| After v0.15.160 | Four-center AICCM grid/domain forwarding and ECP/hybrid dry-run preflight corrected by [Archived #670](https://vibe-qc.com/docs/), commit `4d7b93b01`; use a build containing that fix for this route |

The route matrix describes the current implementation. Availability does not
establish chemical accuracy for a chosen molecule, solid, or surface. The
real-checkpoint compute-cluster validation campaign is now tracked in
[core #83](https://github.com/vibe-qc/vibe-qc/issues), moved from
[archived #672](https://vibe-qc.com/docs/).

## Install the optional runtime

SKALA is not part of the base install because PyTorch wheels are large. Install
the optional extra into the same environment as vibe-qc:

```sh
.venv/bin/pip install -e '.[skala]'
```

The declared runtime window is Python 3.11 through 3.13 with
`torch>=2.12,<2.14`. On Python 3.14 the environment marker intentionally does
not install Torch, even though `pip install -e '.[skala]'` itself succeeds and
PyTorch 2.13 publishes a Python 3.14 wheel. The combined vibe-qc native core,
Torch, and OpenMP runtime is not validated there yet. Use a Python 3.13 or
earlier environment for SKALA until that acceptance canary is green.

```{warning}
Real SKALA evaluation is currently enabled only on Linux. On macOS, the
vibe-qc native core and current PyTorch wheels can initialize different
`libomp` copies in one process; other operating systems have not passed the
same compiled-core acceptance tests. The adapter rejects non-Linux evaluation
before it imports Torch. Importing vibe-qc, prefetching the model, inspecting
provenance, and `dry_run=True` remain safe because they do not import Torch.
```

The upstream Microsoft `skala` package and PySCF are neither installed nor
imported. Importing `vibeqc` registers a lazy callback but does not import
PyTorch, access the network, or load the checkpoint.

## Molecular quickstart

This complete H2 example runs bare SKALA RKS. `Atom` coordinates are in bohr.

```python
import vibeqc as vq

h2 = vq.Molecule(
    [
        vq.Atom(1, [0.0, 0.0, -0.7]),
        vq.Atom(1, [0.0, 0.0, 0.7]),
    ],
    charge=0,
    multiplicity=1,
)

result = vq.run_job(
    h2,
    basis="def2-svp",
    method="rks",
    functional="skala-1.1",
    output="h2-skala",
    progress=False,
)

print(f"E(SKALA) = {result.energy:.10f} Eh")
```

The high-level runner recognizes the exact SKALA aliases and selects the
pinned parity profile automatically. The first real XC evaluation verifies or
downloads the pinned checkpoint and then loads it on the CPU.

The same calculation is available as the runnable
[`input-h2-rks-skala.py`](../../examples/molecular/input-h2-rks-skala.py)
source example, including a safe `--dry-run` mode.

### Add the checkpoint-recommended D3(BJ) correction

The TorchScript artifact is the XC functional only. Its metadata records
`b3lyp5` as the expected D3 setting, but vibe-qc does not silently add
dispersion. Request it explicitly when you intend to reproduce that molecular
protocol:

```python
result = vq.run_job(
    h2,
    basis="def2-svp",
    method="rks",
    functional="skala-1.1",
    dispersion="b3lyp5",
    output="h2-skala-d3",
    progress=False,
)

print(f"E(SCF)   = {result.energy:.10f} Eh")
print(f"E(D3-BJ) = {result.e_dispersion:+.10f} Eh")
print(f"E(total) = {result.energy_total:.10f} Eh")
```

`result.energy` remains the bare SCF value, `result.e_dispersion` is the
additive correction, and `result.energy_total` is the corrected total. Use
`energy_total` in downstream molecular workflows when dispersion was
requested. The D3 setting is part of the published molecular protocol; it is
not evidence for a validated periodic SKALA+D3 method.

### Open-shell UKS and ROKS

Choose UKS for separate alpha and beta orbitals, or ROKS for the restricted
open-shell, spin-pure orbital treatment:

```python
import vibeqc as vq

oh = vq.Molecule(
    [
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.834]),
    ],
    charge=0,
    multiplicity=2,
)

method = "uks"  # Change to "roks" for restricted open-shell KS.

result = vq.run_job(
    oh,
    basis="def2-svp",
    method=method,
    functional="skala-1.1",
    output=f"oh-skala-{method}",
    progress=False,
)

print(f"E({method.upper()}) = {result.energy:.10f} Eh")
```

UKS can exhibit spin contamination, as with any unrestricted Kohn-Sham
calculation. See [ROHF and ROKS](rohf.md) for the general UKS-versus-ROKS
choice.

## The accepted grid contract

SKALA is non-local over the complete atom-major quadrature grid. Grid points,
weights, atomic grouping, and ordering are therefore part of the model input,
not a freely interchangeable numerical accuracy setting within one benchmark.
Microsoft's implementation allows configurable grids, and the training data
used multiple grid constructions. vibe-qc deliberately accepts one pinned
profile: the PySCF 2.14 level-3 default used by Microsoft's PySCF inference
and benchmark protocol:

- element-period radial counts from 50 through 105;
- Lebedev order 29, 302 points, through Ne and order 35, 434 points,
  thereafter;
- atom-specific Treutler-Ahlrichs radial scaling;
- the radius-based five-region NWChem pruning rule;
- the square-root Bragg-radius heteroatomic Becke adjustment; and
- complete atom-major blocks with the required angular and radial ordering.

For normal high-level use, do not set a different grid:

```python
vq.run_job(
    h2,
    basis="def2-svp",
    method="rks",
    functional="skala-1.1",
    grid_level="skala",  # Optional: this is selected automatically.
    output="h2-skala-explicit-grid",
)
```

If you pass an options object whose grid you customised, that grid is
authoritative and `grid_level=` leaves it alone; an untouched grid on a
supplied object receives the `skala` profile like an implicit one (#663). To
choose the profile yourself, set it on that object:

```python
options = vq.RKSOptions()
options.functional = "skala-1.1"
options.grid.atomic_grid_profile = "pyscf-level3"

result = vq.run_job(
    h2,
    basis="def2-svp",
    method="rks",
    rks_options=options,
    output="h2-skala-options",
)
```

`UKSOptions` works the same way. `ROKSOptions.grid` may initially be `None`, so
create the grid before setting its profile:

```python
options = vq.ROKSOptions()
options.functional = "skala-1.1"
options.grid = vq.GridOptions()
options.grid.atomic_grid_profile = "pyscf-level3"
```

The profile spellings `pyscf-level3`, `pyscf_level3`, `pyscflevel3`, and
`skala` select the same coherent grid. A different explicit profile is
rejected by vibe-qc's current acceptance contract before the model callback
runs. Individual radial or angular controls are ignored and replaced by the
profile's coherent PySCF-level-3 settings.

## Periodic SKALA, experimental

The full-grid provider is independent of the Coulomb and exchange builder, but
every enabled route must present one complete atom-major periodic AO density
and project the matching variational XC potential. The native AO-grid routes
do that directly. GPW and GAPW retain their plane-wave or augmented Hartree
builders but bypass their ordinary pointwise or split XC path and use the same
full-grid periodic adapter. The example below is an API smoke calculation in a
deliberately simple 3D cell. It is not a converged chemistry benchmark.

```python
import numpy as np
import vibeqc as vq

a = 8.0
system = vq.PeriodicSystem(
    dim=3,
    lattice=np.eye(3) * a,
    unit_cell=[
        vq.Atom(1, [a / 2, a / 2, a / 2 - 0.7]),
        vq.Atom(1, [a / 2, a / 2, a / 2 + 0.7]),
    ],
)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

result = vq.run_periodic_job(
    system,
    basis,
    method="RKS",
    functional="skala-1.1",
    jk_method="gdf",
    output="h2-cell-skala",
    progress=False,
)

print(f"E(SKALA)/cell = {result.energy:.10f} Eh")
```

The periodic runner takes a `BasisSet` object rather than a basis-name string.
Gamma-only sampling is the default; pass a complete Monkhorst-Pack mesh with
`kpoints=(n1, n2, n3)` when required.

The runnable
[`input-h2-cell-rks-skala-gdf.py`](../../examples/periodic/input-h2-cell-rks-skala-gdf.py)
keeps this experimental smoke calculation separate from validated periodic
reference examples and includes a safe `--dry-run` mode.

```{warning}
Periodic SKALA with `dispersion=` is rejected before setup. The current
periodic output lifecycle cannot yet preserve the corrected energy and full D3
citation provenance. Run bare experimental periodic SKALA, or use the
molecular SKALA+D3 workflow above.
```

Hamiltonian-derived periodic post-processing is also fail-closed. Requests
for `band_structure`, `dos_kmesh`, or `coop_cohp` are rejected because the
generic QVF reconstruction cannot yet include SKALA's nonlocal XC potential.
The current non-BIPOLE periodic population writer uses a molecular
Gamma-block proxy rather than a validated periodic population formula. Do not
use that sidecar for scientific bond-analysis claims. Molecular SKALA
population and QTAIM workflows are unaffected.

### Periodic route matrix

Every row marked available below remains experimental for SKALA. Normal route
restrictions from [Periodic-SCF methods](periodic_methods.md) still apply.
All periodic full-grid external-XC routes are currently zero-temperature
only. Positive electronic smearing fails before dry-run or integral setup;
finite-iteration Mermin energy, entropy, occupations, and the returned density
must first be made one coherent state in every host driver.

| Periodic route | SKALA methods and sampling | Qualification |
|---|---|---|
| GDF | RKS and UKS, Gamma or complete multi-k mesh | Uses the shared periodic AO-grid projector; all-electron bases only, plus the normal dimensional and GDF restrictions |
| RIJCOSX | RKS and UKS, complete multi-k mesh only | All-electron bases only. The Gamma/single-k route uses a molecular XC kernel that cannot provide periodic metadata and fails closed; multi-k uses the periodic GDF/COSX path |
| BIPOLE | 3D RKS and UKS, Gamma or complete mesh | Requires the corrected Ewald exchange gauge and a complete Monkhorst-Pack mesh or valid expandable IBZ; a lone non-Gamma twist, ad-hoc multi-k list, explicit legacy gauge, and ECP-bearing basis are rejected |
| 2D slab Ewald | Not available | The slab drivers do not yet expose the exact difference-closed periodic XC density domain |
| Periodic ROKS | Not available | The maintained GPW/BIPOLE ROKS drivers do not yet expose the explicit periodic-lattice XC density domain |
| Neutral finite-BvK torus, `real-gamma` | 3D RKS and UKS on the real Gamma supercell | Uses the periodic unit-cell atom grid and the explicit periodic-lattice density domain; zero temperature, no DFT+U, and all-electron bases only |
| Neutral finite-BvK torus, `neutral-bloch` | Not available | Fails closed at both the unified runner and low-level RKS/UKS producers: representation invariance of the atom-block periodic provider partition has not yet been established |
| GPW | 3D RKS, Gamma or complete unreduced multi-k mesh | FFT-Poisson remains the Hartree builder; pointwise FFT XC is bypassed. ECP-bearing jobs and symmetry-reduced representative meshes remain gated |
| GAPW | 3D RKS, Gamma only | The augmented Hartree construction remains active, while smooth/hard/soft split XC is bypassed and SKALA sees the complete AO density once; multi-k remains gated |
| Four-centre WSSC lineage, `aiccm2026dev-a` | 3D RKS and UKS through `run_periodic_job(method="aiccm", variant="four-center")` or the low-level `run_ccm_rks` / `run_ccm_uks` API | Pure full-grid providers use the periodic reference-cluster grid. Zero temperature, no DFT+U, and all-electron bases only; external-provider hybrids remain gated |
| Chi-CCM, `aiccm2026dev-b` | 3D RKS and UKS with `aiccm_backend="four_center"` | Uses the complete Gamma-centred finite-torus mesh and explicit periodic-lattice density domain. Zero temperature, no DFT+U, and all-electron bases only; RI, RIJCOSX, lower-dimensional, and external-provider hybrid requests remain gated |

Microsoft's published validation protocols and reference grids are molecular. vibe-qc
keeps the home-atom grid exact and extends the Becke partition to explicitly
numbered periodic image atoms. That image partition and its finite image set
are a vibe-qc periodic adaptation. There is no published periodic SKALA
reference result against which to certify absolute energies.

The `real-gamma` row is a neutral finite-Born-von-Karman-torus representation
control. It is not the literal union-and-weight/four-centre WSSC
`aiccm2026dev-a` construction, and it is not assigned Gamma-CCM construction
identity. Support and validation for either representation do not transfer to
the other. Similarly, `aiccm_backend="four_center"` selects the direct backend
inside the distinct Chi-CCM construction; it does not turn Chi-CCM into the
literal WSSC construction.

The image set used by the periodic Becke partition is currently an
implementation-pinned part of this experimental adaptation, not an independent
high-level convergence keyword.

For any periodic SKALA calculation:

1. Converge the public controls for the selected route, including the cell and
   k-point representation, `rsgdf_ke_cutoff` for RSGDF, or
   `bipole_cutoff_bohr` for BIPOLE as applicable.
2. Repeat the calculation with tighter controls and compare energy differences,
   not only SCF convergence.
3. Validate the intended observable against an independent method appropriate
   to the system.
4. Report the route and that periodic SKALA support is experimental.

The periodic bridge passes the raw orbital kinetic-energy density assembled
from the finite real-space density matrix. It does not apply the
`tau >= tau_W` repair used by some pointwise meta-GGAs, because clipping would
change the checkpoint input and its density-matrix derivative. Small negative
values can arise from finite lattice truncation, which is another reason to
converge the real-space representation explicitly.

## Checkpoint download, cache, and offline use

The adapter pins one immutable artifact:

| Field | Value |
|---|---|
| Repository | `microsoft/skala-1.1` |
| Revision | `99b5ed87e5f69d9216e1f9e30148b922eaea1241` |
| File | `skala-1.1-rev1.fun` |
| SHA-256 | `7f3e8622e1eb520ccd88a55464c3e359ac4d7e5ccbd1fb77a26afa1e1c20a5cd` |
| Device | CPU |

The default path is:

```text
$XDG_CACHE_HOME/vibeqc/skala/microsoft/skala-1.1/<revision>/skala-1.1-rev1.fun
```

When `XDG_CACHE_HOME` is unset, the root is
`~/.cache/vibeqc/skala`. Prefetch and verify the model on a connected login
node with:

```sh
.venv/bin/python -c \
  "import vibeqc as vq; print(vq.ensure_skala_model())"
```

`ensure_skala_model()` verifies an existing cache entry. If it is missing or
corrupt and the network is available, the function downloads a replacement,
enforces the pinned 2,388,988-byte size before hashing, verifies the digest
before an atomic install, and writes `provenance.json` plus
`MICROSOFT-SKALA-SOURCE-MIT.txt` beside it. The latter is explicitly identified
as the source-repository notice, not as checkpoint redistribution clearance.
Invalid bytes are never passed to TorchScript deserialization. If download or
offline staging fails, evaluation stops.

For a shared offline cache, set `XDG_CACHE_HOME` consistently for the prefetch
and every compute job:

```sh
export XDG_CACHE_HOME=/shared/vibeqc-cache
.venv/bin/python -c \
  "import vibeqc as vq; print(vq.ensure_skala_model())"
```

Copy that entire cache tree to the offline node and export the corresponding
root there. You can ask for the exact target without downloading:

```sh
.venv/bin/python -c \
  "from vibeqc.skala import model_cache_path; print(model_cache_path())"
```

Low-level helpers accept `cache_dir=` or an explicit verified `model_path=`,
but `run_job` and `run_periodic_job` do not. A custom high-level cache location
must therefore be selected with `XDG_CACHE_HOME` before the process starts.

```{warning}
TorchScript is an executable serialization format. Stage only the official
file with the pinned digest. Never bypass the SHA-256 check or substitute an
untrusted `.fun` file.
```

## Dry-run planning and reproducibility

Use `dry_run=True` to validate dispatch and obtain the output plan without
importing PyTorch, downloading the checkpoint, or evaluating XC:

```python
vq.run_job(
    h2,
    basis="def2-svp",
    method="rks",
    functional="skala-1.1",
    output="h2-skala-plan",
    dry_run=True,
)
```

Dry run returns `None` and writes the `.system` manifest. It proves that the
input can be routed and planned; it does not prove that Torch imports, the
checkpoint is locally available, or the real calculation succeeds.

Memory estimation during dry run is opt-in. Set
`VIBEQC_DRY_RUN_ESTIMATE=1` in the process environment to run and record the
SKALA-aware estimate:

```sh
VIBEQC_DRY_RUN_ESTIMATE=1 \
  .venv/bin/python examples/molecular/input-h2-rks-skala.py --dry-run
```

The immutable model record is also available without Torch or network access:

```python
import vibeqc as vq

for key, value in vq.skala_model_provenance().items():
    print(f"{key}: {value}")
```

Normal and dry-run `.system` manifests record the external-XC backend,
checkpoint repository, revision and digest, protocol version, CPU device,
feature and model dtypes, model chunk target, expected D3 setting, and grid
policy. The cache path is intentionally excluded because it is machine-local.

Archive these files with a result:

- `.system`, for checkpoint and execution provenance;
- `.out`, for the method and energy breakdown;
- `.bibtex` and `.references`, for the generated citations; and
- the structure and wavefunction artifacts required to reproduce the job.

## Performance and memory

SKALA is a full-grid differentiable model, not a pointwise libxc call. Its
feature tensors and first derivatives make the XC phase materially more
memory-intensive than an ordinary meta-GGA on the same quadrature. vibe-qc
includes this workspace in molecular and periodic memory planning.

The model runs on the CPU. Evaluation chunks preserve complete atoms and group
equal-sized atomic grids, with a default target of 8192 model points. A single
atomic grid is never split, so the actual peak can exceed that target for an
oversized atom block. The full host-side grid state remains resident even when
model evaluation is chunked.

Run a dry plan with `VIBEQC_DRY_RUN_ESTIMATE=1` before a large calculation and
keep the normal memory admission check enabled. `memory_override=True` only
bypasses the safety stop; it does not reduce SKALA's memory requirement.

## Scientific domain and evidence

Atomic number is not a neural-model feature, so the checkpoint is
mathematically element-agnostic. The executable grid is not: the pinned
atom-specific Treutler table ends at Lr, Z=103, and vibe-qc rejects larger
atomic numbers rather than inventing a radius.

Microsoft trained primarily on H through Xe systems and reports tests covering
that range plus Pb and Bi. The strongest evidence remains molecular main-group
chemistry. Transition-metal, heavy-element, and periodic-solid results should
be treated as exploratory and checked against an appropriate independent
reference. A successful SCF is not by itself an accuracy validation.

The offline real-checkpoint regression in vibe-qc covers a fixed-density
He/def2-SVP molecular XC energy and AO potential. The broader provider and
periodic tests exercise the generic boundary and route guards with synthetic
providers. They do not establish a periodic SKALA accuracy benchmark.

## Unsupported operations

The following requests stop with an error rather than substituting libxc,
changing the grid, clipping model inputs, or dropping a requested derivative:

- molecular or periodic analytic gradients and forces;
- molecular geometry optimization and periodic atom or cell optimization;
- cell stress;
- Hessians, TDDFT, TDDFT gradients, and XC second derivatives;
- Newton and TRAH second-order SCF paths;
- pointwise `Functional.eval_*` evaluation;
- periodic SKALA combined with a dispersion correction;
- periodic `band_structure`, `dos_kmesh`, or `coop_cohp` post-processing;
- every 2D external-XC route, including slab Ewald and slab GDF;
- every periodic ROKS external-XC route;
- GDF and RIJCOSX external XC with an ECP-bearing basis;
- GPW external XC outside 3D RKS, including ECP-bearing jobs and
  symmetry-reduced representative meshes;
- GAPW external XC outside 3D Gamma-only RKS, including ECP-bearing jobs;
- Gamma/single-k RIJCOSX XC execution;
- literal four-centre WSSC (`aiccm2026dev-a`) external XC below three periodic
  dimensions, with finite-temperature smearing, DFT+U, an ECP-bearing basis,
  or an external-provider hybrid;
- Chi-CCM (`aiccm2026dev-b`) external XC with RI or RIJCOSX, in fewer than
  three periodic dimensions, with finite-temperature smearing, DFT+U, an ECP,
  or an external-provider hybrid;
- neutral `real-gamma` external XC with finite-temperature smearing, DFT+U,
  or an ECP; and
- every other periodic full-grid external-XC route with positive electronic
  smearing, and BIPOLE with the legacy gauge or an ad-hoc multi-k list; and
- elements above Lr, Z=103, under the pinned parity profile.

Whole-SCF finite differences can be orchestrated externally for research use,
but they are not an analytic SKALA force implementation and do not make the
high-level optimization routes supported.

## Troubleshooting

| Symptom | Meaning and action |
|---|---|
| `No module named 'torch'` | Use Python 3.11 through 3.13 and install `.[skala]` in the active vibe-qc environment. On Python 3.14 the extra intentionally installs no Torch dependency. |
| SKALA reports that the platform is unsupported | Real evaluation is Linux-only. On macOS this also avoids the duplicate-`libomp` conflict described above; other operating systems are not accepted yet. Use a compatible Linux CPU environment. |
| Model download fails | Prefetch on a connected node, copy the complete XDG cache tree, and export the same `XDG_CACHE_HOME` on the compute node. |
| SHA-256 mismatch | Do not bypass the check. Online `ensure_skala_model()` attempts an atomic repair; offline, replace the cache entry with the exact pinned artifact. |
| Pinned grid profile mismatch | Let the high-level runner build options, pass `grid_level="skala"`, or set `options.grid.atomic_grid_profile = "pyscf-level3"` on an explicit options object. |
| Old or mismatched native extension | Rebuild or reinstall the editable package so the Python adapter and compiled external-XC capability boundary come from the same checkout. |
| Memory preflight refuses the job | Reduce system size where scientifically valid or use a larger allocation. Bypassing the check does not lower the full-grid/autograd peak. |
| `NotImplementedError` for a derivative or backend | The requested route is outside the support matrix. Use a fixed-geometry single point on a supported route. |

## Licensing and citations

vibe-qc ships its independent adapter under MPL 2.0 and fetches, rather than
bundles, the official model. The Microsoft source is MIT licensed, and the
checkpoint's model card labels it MIT. The exact-grid adaptation retains
PySCF's Apache-2.0 license and NOTICE. The repository audit found no copyright
or training-data-license blocker to distributing the adapter and hash-pinned
on-demand fetch path.

MIT does not contain an express patent-license clause. The repository audit is
not a legal opinion or a commercial freedom-to-operate review. See the full
[SKALA redistribution audit](../license.md#microsoft-skala-11-optional-neural-xc-functional-skala-extra)
for the reviewed immutable sources, notices, and patent caveat.

Every SKALA job routes its defining references from the central citation
database into `.bibtex` and `.references`; it does not cite libxc because the
model uses the external full-grid backend. See [Citations](citations.md) for
the generated-reference workflow.

## See also

- [Worked SKALA tutorial](../tutorial/skala_neural_xc.md), from runtime
  preflight through molecular RKS, D3(BJ), and provenance.
- [Functionals](functionals.md), conventional libxc functionals and the generic
  `define_external_functional` interface.
- [Installation](../installation.md#optional-microsoft-skala-11-functional),
  the short optional-runtime setup.
- [Periodic-SCF methods](periodic_methods.md), backend selection and periodic
  convergence controls.
- [Memory budget](memory.md), dry-run planning and admission checks.
- [Microsoft Research SKALA project](https://www.microsoft.com/en-us/research/project/dft/),
  the upstream project page.
