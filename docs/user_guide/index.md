# User guide

Reference-style background: each page describes a single concept / API
area in depth. Read the [tutorials](../tutorial/index.md) first if
you're new to vibe-qc.

## Find the page from the task

| I need to... | Start here | Then check |
|---|---|---|
| define charge, spin, and coordinates | [Molecules](molecules.md) | [External structures](external_structures.md) |
| choose or inspect a Gaussian basis | [Basis sets](basis_sets.md) | [Linear dependence](linear_dependence.md), [ECPs](ecp.md) |
| choose HF, DFT, or correlation | [Functionals](functionals.md), [MP2 and double hybrids](mp2_and_double_hybrids.md), [CCSD](ccsd.md) | [Planning tutorial](../tutorial/planning_a_calculation.md) |
| run Microsoft SKALA neural XC | [SKALA-1.1](skala.md) | [worked tutorial](../tutorial/skala_neural_xc.md), [Functionals](functionals.md), [memory budget](memory.md) |
| reduce molecular SCF memory | [SCF modes](scf_modes.md) | [Memory budget](memory.md), [direct-SCF tutorial](../tutorial/direct_scf_memory_tradeoff.md) |
| improve difficult SCF convergence | [SCF convergence](scf_convergence.md) | [Initial guess](initial_guess.md), [solver framework](solver_framework.md) |
| construct a crystal, slab, or wire | [Periodic systems](periodic_systems.md) | [Crystal lattices](crystal_lattices.md), [slabs](slabs_and_adsorbates.md) |
| choose a periodic Coulomb route | [Periodic methods](periodic_methods.md) | [Periodic JK routes](../periodic_jk_routes.md) |
| run the ab initio cyclic cluster model (experimental) | [AICCM](aiccm.md) | [chi-CCM](aiccm2026dev_b.md), [Gamma-CCM](../aiccm2026dev_a.md) |
| converge a crystal calculation | [k-points](k_points.md) | [Smearing](smearing.md), [density fitting](density_fitting.md) |
| understand files and provenance | [Output files](output_files.md) | [Logging](logging.md), [citations](citations.md) |
| analyze IAO charges, spins and bonds | [IAO analysis](iao_population.md) | [Numerical comparisons](iao_validation.md) |
| hand a wavefunction to another program | [TREXIO](trexio.md) | [Output files](output_files.md), [citations](citations.md) |
| open or automate a QVF result | [QVF and vibe-view](../visualization.md) | [vibe-view CLI](vibe_view_cli.md) |
| run on another host | [Queue](queue.md) | [Running](../running.md) |

## Tutorial, guide, or API reference?

- Use a **tutorial** when learning a method or workflow. Tutorials include a
  runnable calculation, theory, expected output, and interpretation.
- Use this **user guide** for supported behavior, options, defaults, and
  limitations organized by concept.
- Use the **[API reference](../api/index.md)** when you already know the
  object or function name and need its exact signature.
- Use the **[examples catalog](https://github.com/vibe-qc/vibe-qc/blob/main/examples/README.md)**
  to find a complete input closest to the job you want to run.

```{toctree}
:maxdepth: 1
:caption: Molecular methods

molecules
basis_sets
basis_optimization
cohesive_energies
naming
functionals
skala
rohf
mp2_and_double_hybrids
ccsd
dlpno_mp2
non_hf_solvers
composites
mlip
semiempirical
semiempirical_mlip_comparison
msindo
dft_plus_u
ecp
```

```{toctree}
:maxdepth: 1
:caption: Periodic methods

periodic_systems
crystal_lattices
slabs_and_adsorbates
surface_embedding
k_points
multi_k_scf
ewald
bipole
gapw
periodic_methods
../periodic_jk_routes
cyclic_cluster_model
aiccm
aiccm2026dev_b
../aiccm2026dev_a
../design_aiccm2026dev_b
../aiccm2026dev_b_decisions
coop_cohp
```

```{toctree}
:maxdepth: 1
:caption: SCF & convergence

scf_convergence
opentrustregion
opentrustregion_theory
scf_modes
solver_framework
initial_guess
smearing
metallic_bz_integration
linear_dependence
density_fitting
solvation
settings
memory
```

```{toctree}
:maxdepth: 1
:caption: Workflows & properties

relaxed_scan
neb
molecular_dynamics
properties
bond_analysis
iao_population
iao_validation
band_structure
volumetric_data
data_library
reference_data
qtaim
```

```{toctree}
:maxdepth: 1
:caption: Import & export

ase_integration
external_codes
external_structures
from_pyscf
from_crystal
from_orca
from_gaussian
output_files
trexio
logging
citations
```

```{toctree}
:maxdepth: 1
:caption: Tooling

blas
queue
jupyter
keyword_index
basis_toolkit
```

```{admonition} Looking for vibe-view or QVF?
:class: note

The viewer and format reference pages (interactive viewer, terminal mode,
desktop app, vq Job Manager, live reload, QVF job containers) are grouped with
their tutorials and the format toolkit under
[QVF and vibe-view](../visualization.md).
```
