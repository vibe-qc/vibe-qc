# Tutorial

These tutorials double as a **course** in computational quantum chemistry.
If you are new to vibe-qc or to the field, start with the
[Introduction](introduction.md) for the background, the molecular/periodic
split, and the learning path; otherwise work down from wherever you like.
Each page is a complete worked example you can paste straight into a Python
interpreter.

## Choose a learning path

You do not have to read all of the tutorials. Pick the path that matches the
calculation you want to perform, and use the user guide when you need a
reference table rather than a lesson.

| Path | Read in order | You will finish able to... |
|---|---|---|
| First calculation, about 45 minutes | [Introduction](introduction.md) -> [Planning](planning_a_calculation.md) -> [Molecular HF](molecular_hf.md) -> [Molecular DFT](molecular_dft.md) | choose a model, run an SCF calculation, and judge convergence |
| Molecular structure and properties | [Molecular HF](molecular_hf.md) -> [Geometry optimization](geometry_optimization.md) -> [Vibrations](vibrational_frequencies.md) -> [Thermodynamics](thermodynamics.md) | move from an electronic energy to an optimized, characterized structure |
| Correlated molecular energies | [Basis convergence](basis_convergence.md) -> [RI-MP2](ri_mp2_aux_verification.md) -> [CCSD(T)](coupled_cluster_ccsd_t.md) -> [DLPNO](dlpno_local_correlation.md) | choose a correlation hierarchy and track its cost and approximations |
| Periodic materials | [Periodic HF](periodic_hf.md) -> [Bloch and k-points](kpoints_brillouin_bloch.md) -> [Periodic DFT](periodic_dft.md) -> [Method routes](periodic_methods_compared.md) | construct a crystal calculation and converge its sampling and Coulomb route |
| Results and visualization | [QVF format](qvf_file_format.md) -> [vibe-view setup](vibe_view_getting_started.md) -> [Viewer walkthrough](vibe_view_walkthrough.md) | inspect, compare, export, and archive results |
| Portable molecular inputs and results | [Molecular HF](molecular_hf.md) -> [QCSchema interchange](qcschema_interchange.md) | exchange a molecule and calculation result as JSON and check it against a literature energy |
| Wavefunction exchange | [TREXIO molecular exchange](trexio_exchange.md) -> [CI and periodic TREXIO](trexio_correlated_periodic.md) | export, inspect, convert and restart wavefunctions while retaining default QVF output |
| Remote operation | [Parallel execution](parallel_execution.md) -> [vq remote jobs](vq_queue_remote_job.md) -> [Reference outputs](reference_outputs.md) | submit, monitor, fetch, and preserve remote calculations |

(molecular-path)=

### Molecular path

Start with the foundations section, then choose properties, accuracy, or
correlation. Read [Direct SCF or in-core integrals?](direct_scf_memory_tradeoff.md)
before using a large molecular basis: basis growth affects memory much faster
than atom count alone suggests.

(periodic-path)=

### Periodic path

Read the four periodic foundations pages before choosing GDF, BIPOLE, GPW,
GAPW, or CCM. A periodic calculation adds lattice sums, k-point sampling,
dimensionality, and basis linear-dependence questions that do not occur in an
isolated molecule.

### What every tutorial tells you

A complete tutorial should answer five questions: what physical problem is
being solved, which approximations enter, how to run it, what successful
output looks like, and what to test before trusting the result. When you only
need option names and defaults, jump to the [user guide](../user_guide/index.md)
or [keyword index](../user_guide/keyword_index.md).

For downloadable inputs, full logs, QVF archives, and vibe-view captures, see
[reference outputs](reference_outputs.md) and the full
[example-output catalog](../example_outputs.md). The catalog includes the
curated chi-CCM-B periodic QVF validation fixtures for checking finite-BvK
cells, torus-aligned grids, and Wannier-centre overlays.

```{toctree}
:maxdepth: 1
:caption: Introduction

introduction
planning_a_calculation
```

```{toctree}
:maxdepth: 1
:caption: Molecular: foundations

molecular_hf
direct_scf_memory_tradeoff
molecular_dft
qcschema_interchange
open_shell
initial_guess_walkthrough
```

```{toctree}
:maxdepth: 1
:caption: Molecular: properties & workflows

post_scf_properties
orbital_visualization
geometry_optimization
vibrational_frequencies
thermodynamics
molecular_dynamics
neb_reaction_path
relaxed_pes_scan
solvation_water
```

```{toctree}
:maxdepth: 1
:caption: Molecular: methods & accuracy

basis_convergence
pt_cluster_lanl2dz
functional_comparison
skala_neural_xc
dispersion
double_hybrid_b2plyp
range_separated_wb97x
vv10_modern_functionals
rijcosx_glycine
ediis_diis_hybrid
adaptive_diis
second_order_scf
opentrustregion
opentrustregion_stability
excited_states_tddft
```

```{toctree}
:maxdepth: 1
:caption: Molecular: correlation & fast methods

natural_orbitals
ri_mp2_aux_verification
coupled_cluster_ccsd_t
dlpno_local_correlation
non_hf_solvers
casscf_multireference
msindo
semiempirical_dftb
pm6_and_gfn2
gfn2_xtb_workshop
semiempirical_periodic_and_validation
mace_mlip
mace_geometry_to_electronic_structure
```

```{toctree}
:maxdepth: 1
:caption: Periodic: foundations

periodic_hf
periodic_dft
madelung_with_ewald
kpoints_brillouin_bloch
```

```{toctree}
:maxdepth: 1
:caption: Periodic: the cyclic cluster model

cyclic_cluster_model
aiccm_quickstart
aiccm2026dev_a
aiccm2026dev_b
aiccm2026dev_b_localization
h8_chain_ccm_ancestor
equidistant_h_chain
```

```{toctree}
:maxdepth: 1
:caption: Periodic: methods

periodic_methods_compared
periodic_gpw
periodic_gapw
```

```{toctree}
:maxdepth: 1
:caption: Periodic: electronic structure

band_structure
lih_multi_k
pdos
fermi_dirac_smearing
gilat_net_metals
dft_plus_u
```

```{toctree}
:maxdepth: 1
:caption: Periodic: basis & real systems

pob_tzvp
lih_pob_tzvp_solid_state
periodic_crystal_gallery
periodic_geometry_optimization
mgo_from_materials_project
slab_adsorbate_dft
surface_embedding
```

```{toctree}
:maxdepth: 1
:caption: Periodic: phenomena, convergence & visualization

peierls
open_shell_mg_cation
periodic_orbital_cubes
tight_cell_dft
periodic_scf_convergence
symmetry_storage
xsf_bxsf_visualization
```

```{toctree}
:maxdepth: 1
:caption: Operations

trexio_exchange
trexio_correlated_periodic
parallel_execution
vq_queue_remote_job
external_data_fetcher
reference_outputs
auto_citations
cross_validation
```

```{admonition} Visualization tutorials moved
:class: note

The QVF and vibe-view tutorials (the file format, job containers, adopting QVF
in your own code, the browser walkthrough, terminal mode, the full vibe-view
feature tour, and MolTUI) now live together in
[QVF and vibe-view](../visualization.md), alongside the matching reference
pages and the format toolkit. The tutorial pages themselves did not move; only
their place in the navigation did.
```

New to vibe-qc or to the field? The [Introduction](introduction.md) lays
out the background, the molecular and periodic tracks, and how to run these
examples. The molecular tutorials use the built-in basis sets (STO-3G,
6-31G\*, cc-pVDZ) that ship with vibe-qc; the periodic tutorials use the
[pob-\* basis sets](../user_guide/basis_sets.md) designed to avoid linear
dependence in crystals.
