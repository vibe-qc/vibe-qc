---
myst:
  html_meta:
    "description": "Optimize molecules and true 2D adsorbate slabs with MACE, then hand the geometry to DLPNO-CCSD(T) or periodic DFT without comparing incompatible total-energy scales."
    "og:title": "MACE geometry to electronic structure in vibe-qc"
---

# From a MACE geometry to electronic structure

MACE is excellent at proposing structures quickly; it does not provide a
wavefunction. This tutorial makes the boundary explicit with two workflows:

1. optimize an isolated molecule with the registered organic MACE-OFF23
   model, then run molecular DLPNO-CCSD(T) on the optimized geometry;
2. optimize an adsorbate on a genuine two-dimensional slab with the
   MIT-licensed MACE-MPA-0 materials model, then run a periodic electronic
   single point on the whole relaxed slab.

The MACE energy is not carried into either electronic calculation. It lives
on a model-specific reference scale and selected the geometry only.

## Choose the workflow before choosing the method

| Structure after optimization | Valid electronic handoff | Not implied |
|---|---|---|
| Isolated molecule | molecular HF/DFT/MP2/CC/DLPNO, with charge and spin stated | that MACE represented the chosen charge/spin state |
| Whole periodic slab + adsorbate | `run_periodic_job` with a supported periodic HF/DFT route | molecular DLPNO-CCSD(T) on an infinite slab |
| Finite cluster cut from a slab | molecular methods only after an explicit cluster, termination, charge, and spin protocol | equivalence to the original periodic surface |

vibe-qc's molecular `run_job(method="dlpno-ccsd(t)")` cannot accept a
`PeriodicSystem`. Periodic CCM/DLPNO capabilities are not a general
whole-slab DLPNO replacement. For an adsorbate surface, use periodic HF/DFT
for the intact slab or define and validate a separate embedded/finite-cluster
model.

## Runtime split

MACE currently needs Python 3.13 or earlier. DLPNO-CCSD(T) does not need the
optional Torch stack, so the two molecular stages can run in separate
environments:

```sh
python3.13 -m venv .venv-mace
.venv-mace/bin/pip install -e '.[mace]'

python3.14 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

The source checkout can follow current `main`; every queued calculation must
still record the exact Git SHA and imported package versions it actually ran.
The MACE health check records Python, VibeQC, ASE, Torch, e3nn, MACE, API,
registry, cache files, and cached-weight digests:

```sh
.venv-mace/bin/python examples/mlip/00_runtime_healthcheck.py
```

## Workflow A: MACE-OFF23 to molecular DLPNO-CCSD(T)

MACE-OFF23 is the organic-family model in the supported VibeQC scope. Its
weights use the Academic Software License, so the example refuses to run
without a calculation-local acknowledgement:

```sh
.venv-mace/bin/python \
  examples/mlip/06_optimize_water_off23_for_dlpno.py \
  --accept-asl --work-dir mace-handoff-output
```

The input calls:

```python
options = MLIPOptions(
    model="off23-medium",
    accept_academic_license=True,
    device="cpu",
    dtype="float64",
)
result = run_job(
    molecule,
    method="mace",
    mlip_options=options,
    optimize=True,
    fmax=0.02,
    output="mace-to-dlpno-water-opt",
)
```

The resulting `mace-to-dlpno-water-opt.xyz` is the only scientific datum
handed to stage 2. Model identity, loader, ASL notice, device, dtype, cache
location, and citations stay in the MACE siblings.

Run the electronic single point:

```sh
.venv/bin/python examples/mlip/07_dlpno_ccsdt_on_mace_geometry.py \
  --work-dir mace-handoff-output
```

Its central call is:

```python
molecule = Molecule.from_xyz(
    "mace-to-dlpno-water-opt.xyz",
    charge=0,
    multiplicity=1,
)
result = run_job(
    molecule,
    basis="cc-pvdz",
    method="dlpno-ccsd(t)",
    write_molden_file=True,
    write_population_file=True,
    output="mace-to-dlpno-water-dlpno-ccsd-t",
)
```

`result.energy_total` is the DLPNO-CCSD(T) electronic total. The emitted
Molden orbitals, Mulliken-style populations, bond orders, and dipole are from
the RHF reference wavefunction: VibeQC does not yet build a correlated DLPNO
one-particle property density. Label those properties accordingly.

Do not subtract the raw MACE and DLPNO totals. To assess the geometry, compare
bond lengths/angles, electronic energies evaluated on multiple geometries
with the same electronic method, or an external experimental/reference
observable.

## Workflow B: MACE-MPA-0 adsorbate/slab relaxation to periodic DFT

The direct periodic MACE wrapper now preserves the dimensionality of a
`PeriodicSystem`:

- `dim=1` becomes ASE PBC `(True, False, False)`;
- `dim=2` becomes `(True, True, False)`;
- `dim=3` becomes `(True, True, True)`.

That distinction is essential. VibeQC's true slab stores a synthetic third
lattice vector for bookkeeping, not a physical vacuum repeat. A slab MACE
calculation therefore exposes energy and forces but refuses stress and cell
relaxation: both would normalize against or optimize a non-physical volume.

Run the geometry stage for a compact H2O/h-BN demonstration:

```sh
.venv-mace/bin/python \
  examples/mlip/08_relax_h2o_hbn_then_periodic_dft.py \
  --work-dir mace-handoff-output
```

The relevant API is:

```python
sheet = hbn_2x2()
adsorbed = vq.place_adsorbate(
    sheet,
    "H2O",
    position=(x_hollow, y_hollow),
    height=2.90,
    orientation="flat",
    anchor="O",
)
relaxed = optimize_periodic_mace_positions(
    adsorbed,
    MLIPOptions(model="medium-mpa-0", device="cpu", dtype="float64"),
    fmax=0.05,
    max_steps=200,
    fixed_indices=range(len(sheet.unit_cell)),
)
if not relaxed.converged:
    raise RuntimeError("do not launch the electronic stage")
relaxed_slab = relaxed.system
```

The cell, dimension, charge, multiplicity, atom order, and frozen-layer
choice remain explicit. The helper in the runnable example constructs the
2 x 2 h-BN sheet from explicit lattice vectors and atoms. The example writes
an Extended XYZ with
`pbc="T T F"` for inspection.

To run the whole-slab electronic stage in the same process:

```sh
.venv-mace/bin/python \
  examples/mlip/08_relax_h2o_hbn_then_periodic_dft.py \
  --electronic --work-dir mace-handoff-output
```

It passes `relaxed.system` directly to `run_periodic_job` with periodic PBE.
For production, converge the slab thickness, lateral supercell, adsorption
sites, k mesh, basis, Coulomb route, electronic convergence, and geometry
tolerance. The 2 x 2 cell and fixed ideal sheet are tutorial choices, not an
adsorption benchmark.

## What the literature supports

The original MACE paper establishes higher-order equivariant message passing
and analytic force prediction from the learned energy. The MACE-MP paper
demonstrates broad bulk, molecular, interface, and catalysis use, but also
shows that a foundation model can deviate materially from the reference DFT
and that quantitative surface work may need task-specific fine-tuning. The
OFF23 paper evaluates organic chemistry and geometry/force use at its own
training level.

Read the primary sources:

- Batatia et al., [MACE architecture](https://doi.org/10.48550/arXiv.2206.07697)
- Batatia et al., [MACE-MP foundation model](https://doi.org/10.1063/5.0297006), *J. Chem. Phys.* **163**, 184110 (2025)
- Kovács et al., [MACE-OFF](https://doi.org/10.1021/jacs.4c07099), *J. Am. Chem. Soc.* **147**, 17598 (2025)

The examples use `medium-mpa-0`, not the exact fine-tuned or dispersion-added
variants used in every published catalysis subsection. They establish a
reproducible geometry workflow, not transferred accuracy from a different
weight or protocol.

## Checklist before launching expensive electronic work

- Record exact VibeQC SHA/version and Python/MACE/Torch/e3nn/ASE versions.
- Record model key, loader and argument, license, device, dtype, cache path,
  cached-weight digest, and citations.
- Keep the OFF23 acknowledgement calculation-local; never globalize it.
- Confirm element coverage and the model's training domain.
- Inspect the geometry, boundary mask, frozen indices, terminal force, step
  count, and `converged` flag.
- Treat a capped or failed optimization as a failed row, not a structure.
- Preserve charge and multiplicity explicitly for the electronic stage;
  MACE did not choose or validate them.
- Compare MACE relative energies only within one fixed model and composition.
- Never claim reference surface accuracy without an external comparison.
