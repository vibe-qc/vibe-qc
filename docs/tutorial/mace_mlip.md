# Machine-learning interatomic potentials with MACE

[ACEsuit **MACE**](https://github.com/ACEsuit/mace) is a pre-trained
*E(3)*-equivariant graph-neural-network **interatomic potential**. It
predicts energy, forces, and stress on a DFT-fitted potential-energy
surface. Runtime and accuracy depend on the model, structure, device,
and whether the chemistry lies inside the training domain. vibe-qc
exposes molecular `method="mace"` dispatch plus direct periodic drivers.

This tutorial uses the `mace-torch` distribution from ACEsuit. Mendoza
Automated Crystal Engine, including its `mace gui connect` workflow command,
is a separate application. See [Which MACE?](../user_guide/mlip.md#which-mace)
for the project and installation distinction.

MACE is a *learned potential*, not a quantum-chemistry method: it solves
no Schrödinger equation and produces no wavefunction. vibe-qc drives
MACE's pre-trained forward pass and attributes it as an external model, 
the energy is MACE's, not a vibe-qc ab-initio total. This tutorial walks
from a single molecule through geometry optimization, periodic crystals,
surface probes, and a comparison with
vibe-qc's own semi-empirical methods.

> **What you'll learn:** run MACE on molecules and crystals; optimize
> geometries and relax unit cells; reuse one model for a periodic strain
> probe; choose a model and respect its license; and compare relative
> curves without mixing incompatible energy references.

## Background

**What MACE is.** MACE ({vibeqc-cite}`batatia_mace_2022`) is a message-passing
neural network potential. The total energy is a sum of atomic site energies,
each a learned function of the neighbours within a cutoff radius, and forces
and stress are its analytic derivatives obtained by automatic differentiation,

$$
E = \sum_i E_i\bigl(\{\mathbf r_{ij}, Z_j\}_{|\mathbf r_{ij}| < r_\text{cut}}\bigr),
\qquad
\mathbf F_i = -\nabla_{\mathbf r_i} E ,
$$

where $\mathbf r_{ij}$ are the vectors to the neighbours of atom $i$, $Z_j$
their elements and $r_\text{cut}$ the cutoff. *Equivariant* message passing
means the internal features are irreducible representations of $E(3)$ built
from spherical harmonics of the neighbour directions, so the symmetry under
rotation, reflection and translation is exact by construction, and the
messages combine many neighbours at once (a high body order from the atomic
cluster expansion) so that two layers are enough.

**Training data and families.** The parameters are fitted by the model
authors to DFT energies, forces and stresses. MACE-MP-0 and MACE-MPA-0
({vibeqc-cite}`batatia_mace_mp_2024`) are trained on Materials Project
relaxation data (MPA-0 adds a subsample of a second materials database),
cover 89 elements and are released under the MIT licence. MACE-OFF23
({vibeqc-cite}`kovacs_mace_off_2023`) is trained on organic molecules,
covers H, C, N, O, F, P, S, Cl, Br and I, and is released under the
Academic Software License (academic and non-commercial use only), which is
why vibe-qc gates it behind an explicit acknowledgement. Section 3 lists the
model keys; the [MACE user guide](../user_guide/mlip.md#background-what-mace-is)
gives the longer account.

**What it is not.** MACE is an **external pre-trained model, not a vibe-qc
method.** vibe-qc drives its forward pass, records the provenance and cites
the papers, but computes none of the numbers itself and cannot vouch for their
accuracy beyond what the model papers report. The interface (model registry,
licence gate, molecular and periodic drivers) is a production route; the
model behind it carries the limits of any learned potential. Inside its
training distribution it is fast and typically close to the DFT it learned;
outside it (unusual oxidation states, charged or open-shell species, elements
or bonding motifs absent from the data, very short contacts) it fails without
warning, because there is no variational principle, no SCF residual and no
electronic structure to inspect. The run must therefore be judged against an
external observable, and its absolute energies never compared with an
electronic-structure total.

```{note}
This tutorial's code needs the optional `[mace]` extra (PyTorch, e3nn, MACE)
including on Python 3.14. The interpreter used to check this documentation
pass carries neither, so the calls below were not re-run on this build and
no new output is quoted; the numbers and behaviour described come from the
existing examples and the user guide. Run the health check in the
Installation section before relying on them.
```

## Installation

MACE is an optional dependency (a heavy PyTorch + e3nn stack), behind the
`[mace]` extra, including on Python 3.14:

```sh
pip install 'vibe-qc[mace]'
```

The extra requires `mace-torch>=0.3.16,<0.4`; ACEsuit's planned 1.0 API
and checkpoint migration need a separate integration update. The runtime
health check uses the same dependency guard as calculations and prints
actionable installation errors without downloading weights.

On Python 3.14 the extra requests PyTorch >=2.14.1. No older interpreter
is needed. A missing backend raises a clear installation error.
On macOS, vibe-qc auto-sets
`KMP_DUPLICATE_LIB_OK=TRUE` (its native core and PyTorch each link an
OpenMP runtime) and caps MACE's OpenMP/BLAS pools to one thread by
default, verified not to change results. Set
`VIBEQC_MACE_OPENMP_THREADS=N` before importing vibe-qc to opt into a
different MACE thread count. Foundation-model weights download on first
use into `~/.cache/mace/`.

Validate the environment before downloading weights:

```sh
python examples/mlip/00_runtime_healthcheck.py
```

The health check reports Python and package versions, source revision
when available, the default model registry entry, public periodic APIs,
and the expected cache root.

## 1. A first calculation

`method="mace"` plugs into `run_job` exactly like an SCF method, minus
the basis set (MACE needs none):

```python
from vibeqc import Molecule, run_job

mol = Molecule.from_xyz("h2o.xyz")
run_job(mol, method="mace", output="h2o_mace")
```

`h2o_mace.out` carries the energy, a MACE **provenance block**
(model, license, training/theory, the energy-scale caveat), and the
references; `h2o_mace.bibtex` / `.references` cite the MACE method
paper + the foundation-model paper. There is no `.molden`, MACE has no
orbitals.

The single-point wrappers require a finite scalar energy, one three-component
force per atom, and a finite six-component stress for a 3D periodic run.
Missing, malformed, or non-finite backend results raise `ValueError` before
an energy or converged periodic result is returned. Failed evaluations do
not populate the result cache; backend loading and evaluation errors propagate.

> **Energy scale.** A MACE energy lives on a **model-specific reference
> scale** (each model subtracts its own per-element atomic energies),
> *not* a vibe-qc total and *not comparable across models*. Use MACE
> energies for relative quantities within one fixed model and
> composition, geometry optimization, model-internal reaction curves,
> adsorption energies, and MD. Never compare their absolute values with
> semi-empirical or electronic total energies.

## 2. Geometry optimization

Add `optimize=True`, ASE's BFGS drives the MACE calculator directly:

```python
run_job(mol, method="mace", optimize=True, fmax=0.02, output="h2o_opt")
```

You get `h2o_opt.traj` with one frame per step. A lower final energy and
small terminal force validate execution on the selected model surface;
they do not establish agreement with experiment or a reference method.

## 3. Choosing a model, and its license

The default is the MIT-licensed **MACE-MPA-0** (materials, 89 elements).
Use it for the materials-family examples in this tutorial. For an
organic-family calculation, select registered MACE-OFF23 weights and
acknowledge their license explicitly:

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

| model key | family | domain | license |
|-----------|--------|--------|---------|
| `medium-mpa-0` (default), `small`, `medium`, `large` | MACE-MP / MPA | materials (89 elts) | **MIT** |
| `off23-small`, `off23-medium`, `off23-large` | MACE-OFF23 | organic (H,C,N,O,F,P,S,Cl,Br,I) | **ASL** |

The organic **MACE-OFF23** weights are under the Academic Software
License, **academic, non-commercial use only**. vibe-qc raises a
`PermissionError` if you select an ASL model without acknowledging it
(`accept_academic_license=True` or `VIBEQC_ACCEPT_ASL=1`). MIT models are
never gated. Picking an organic model for an out-of-domain element (say a
metal) raises a clear error pointing you at a materials model.

The documented scope contains only the registered MACE-MP/MPA-0 and
MACE-OFF23 families. Arbitrary URLs, unregistered weights, OMAT, MATPES,
MH, and MDP are rejected before the backend imports. License acknowledgement
is deliberately not an escape hatch for unsupported weights.

## 4. Periodic crystals

Periodic systems use direct drivers. MACE is not wired into
`run_periodic_job`, which is SCF-only. Following the periodic
semi-empirical precedent. `run_periodic_mace` returns energy and forces for
1D, true 2D slabs, and 3D systems. Stress and variable-cell relaxation are
3D-only. `optimize_periodic_mace_positions` keeps the cell fixed and is the
surface-geometry path:

```python
import numpy as np
from ase.build import bulk
from vibeqc.ase_periodic import atoms_to_periodic_system
from vibeqc.mlip.mace import (
    PeriodicMACEEvaluator,
    optimize_periodic_mace_cell,
    optimize_periodic_mace_positions,
    run_periodic_mace,
)

systems = [
    atoms_to_periodic_system(bulk("Si", "diamond", a=a))
    for a in (5.43, 5.47)
]

evaluator = PeriodicMACEEvaluator()
results = [evaluator.run(system) for system in systems]
energies = np.array([result.energy for result in results])
relative_mev = (energies - energies.min()) * 27211.386245988

for system, result, delta in zip(systems, results, relative_mev):
    print(delta)                         # meV, normalized within this model
    print(np.diag(result.stress()) * 29421.0)  # GPa

relaxed = optimize_periodic_mace_cell(systems[0], fmax=0.01)

# True dim=2 slab: keep its non-physical bookkeeping a3 fixed.
slab_opt = optimize_periodic_mace_positions(
    slab,
    fmax=0.05,
    fixed_indices=slab_info.bottom_layer_indices(2),
)
relaxed_slab = slab_opt.system
```

The evaluator keeps one fixed model, device, and dtype. It is sequential;
parallel scans should construct one evaluator per worker. Start with a
small strain probe such as the two points above before scheduling a full
equation-of-state scan. A finite energy, symmetric force, or near-zero
stress at one point is runtime evidence, not EOS validation. The
downloadable Si example prints model identity with the probe:
[`02_periodic_mace_silicon.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/mlip/02_periodic_mace_silicon.py).

For a `dim=2` result, calling `.stress()` raises instead of returning a
number normalized by the arbitrary third bookkeeping vector. Likewise,
`optimize_periodic_mace_cell` rejects a slab. Relax positions and frozen
substrate layers with `optimize_periodic_mace_positions`.

## 5. Molecules on surfaces, adsorption energies

For a surface and adsorbate inside the materials model's documented
element coverage, the standard adsorption-energy construction is

```
E_ads = E(slab + molecule) − E(slab) − E(molecule)
```

all evaluated with the *same* MACE model so the per-element reference
energies cancel. The exploratory
[`04_surface_adsorption.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/mlip/04_surface_adsorption.py)
script demonstrates calculator reuse, fixed bottom layers, gas-phase
boxes, and the subtraction.

Its output is not reference data. Before making a scientific claim,
converge slab thickness, lateral cell size, adsorption site, vacuum,
relaxation tolerance, and molecular versus dissociated state. Compare
against external reference data appropriate to the observable. Record
model key, loader, weight provenance, cache digest, license, device,
dtype, and citations with the result.

For the supported first-class slab handoff, see
[MACE geometries to electronic structure](mace_geometry_to_electronic_structure.md).
It relaxes H2O on h-BN with a genuine `(True, True, False)` boundary mask and
then passes the same `PeriodicSystem` to periodic PBE. Whole-slab molecular
DLPNO-CCSD(T) is not supported; DLPNO is demonstrated separately for an
isolated molecule.

## 6. MACE versus vibe-qc's semi-empirical methods

vibe-qc ships its own semi-empirical platform (DFTB, GFN2-xTB, PM6,
OMx, see [`semiempirical`](../user_guide/semiempirical.md)). Both are
fast, non-ab-initio energy engines, but they sit in different places:

| | **MACE** (this tutorial) | **semi-empirical** (vibe-qc's own) |
|---|---|---|
| nature | learned potential (pre-trained GNN) | physics-based, vibe-qc's own code |
| reference target | model-training DFT surface | method-specific parameterization |
| electronic structure | **none** (no orbitals/charges/gap) | yes, charges, orbitals, HOMO-LUMO |
| charge / spin states | ignored (neutral potential) | represented |
| elements | registered model coverage | method parameter coverage |
| validation question | domain and reference-observable agreement | implementation parity and reference-observable agreement |
| typical use | model-internal structures, forces, stress, MD | electronic quantities and fast energy surfaces |

The trade-off is that semi-empirical methods can provide electronic
quantities and represent supported charge and spin states, while MACE
provides none of those and applies only within its training domain. A
method comparison therefore needs an observable shared by both methods,
not their absolute total energies. Detailed production boundaries live in
[`semiempirical_mlip_comparison`](../user_guide/semiempirical_mlip_comparison.md).

The runnable comparison example evaluates the same fixed-composition H₂O
stretch with the MIT MACE-MPA-0 model and PM6:

```sh
python examples/mlip/05_mace_pm6_relative_curve.py
```

It subtracts each method's own sampled minimum and prints
`mace_delta_ev` and `pm6_delta_ev`. Comparing curve shapes and sampled
minima is meaningful. Comparing the two raw absolute energies is not.
Neither curve is an external reference, so disagreement is a diagnostic
and agreement is not an accuracy validation.

## 7. Visualization

A MACE run writes the same structural output as any vibe-qc job, so
geometry and (with `optimize=True`) the relaxation trajectory visualize
out of the box: `moltui h2o_opt.xyz` (terminal), or pass
`output_qvf=True` and open the `.qvf` archive in **vibe-view** for the
3-D structure + trajectory animation + citations. There is no
orbital/density/band visualization, there is no wavefunction to plot.
See [`mlip`](../user_guide/mlip.md) § Visualization.

## 8. Limitations + good practice

- **No electronic structure**, no charges, orbitals, gaps, spectra.
- **Trained-domain only**, accuracy degrades out of distribution; MACE
  cannot tell you when you've left its domain. Sanity-check against DFT
  for unusual chemistry.
- **Charge / spin ignored**, vibe-qc warns if you pass a non-neutral or
  open-shell system; the energy is the neutral value regardless.
- **Energy scale**, model-specific reference, never an absolute total or
  comparable across models.
- **Validation level**, import checks, nonzero energies, symmetric
  forces, and successful optimization establish runtime execution only.
  Scientific accuracy needs an external observable comparison.
- **Attribution**, every run cites the MACE papers; cite them in
  published work (vibe-qc drives MACE, it does not claim MACE's energy as
  its own).

## See also

- [`mlip`](../user_guide/mlip.md), the MACE user-guide reference.
- [`examples/mlip/`](https://github.com/vibe-qc/vibe-qc/tree/main/examples/mlip)
  - runtime health, molecular, periodic, NEB, surface, and normalized
  MACE-PM6 comparison scripts.
- [`semiempirical_mlip_comparison`](../user_guide/semiempirical_mlip_comparison.md)
  - detailed MACE-vs-semi-empirical comparison.
- [`neb`](../user_guide/neb.md) § MACE backend, `run_neb(method="mace")`
  drives reaction-path searches on the MACE potential (analytic forces,
  no SCF).
- [Geometry optimization](geometry_optimization.md),
  [Slabs and adsorbates](../user_guide/slabs_and_adsorbates.md).
