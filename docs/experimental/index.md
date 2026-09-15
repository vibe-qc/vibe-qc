# Experimental

vibe-qc ships a handful of features behind an **experimental gate**. They are
reachable, but not yet production-certified and subject to change without the
usual deprecation cycle. Each one emits an `ExperimentalWarning` when used.

```{toctree}
:hidden:

catalog
```

## What's in this section

- The **[feature catalog](catalog.md)** is the complete table of every gated
  feature, its opt-in, its caveat, and its quantitative-status column. Start
  there if you want to know what's gated and why.
- **Basis set development**, developer documentation for the basis-set
  optimiser toolchain, design notes, and verification reports.
- **AICCM**, the ab-initio cyclic cluster model, both Γ-CCM
  (`aiccm2026dev-a`, union-and-weight/Wigner--Seitz integral weighting) and
  χ-CCM[^xccm-convention] (`aiccm2026dev-b`, finite-translation-group
  characters).

[^xccm-convention]: Γ-CCM and χ-CCM are distinct CCM construction approaches:
    union-and-weight/Wigner-Seitz integral weighting and finite-translation-group
    characters, respectively. A declared common exchange-q=0 convention is a
    comparison constraint, not an identification of the constructions. Their
    distinction is not a choice of Coulomb kernels, and equality for a specified
    operator and route must be established rather than inferred from the labels.

## Basis set development

```{toctree}
:maxdepth: 1
:caption: Basis set development

../basisset_dev/index
```

## AICCM: ab-initio cyclic cluster model

The cyclic cluster model is vibe-qc's signature real-space approach to
periodic systems. Two independent implementations are maintained while the
common limit is being established:

- **[`aiccm2026dev-a`](../aiccm2026dev_a.md)**, the union-and-weight
  Gamma-supercell CCM line (`vibeqc.periodic.ccm`). Its construction methods
  are HF, KS, MP2, UMP2, and CCSD(T), with analytic gradients and derivable
  properties. The same namespace contains neutral fitted-torus UCCSD(T) and
  DLPNO controls that are not assigned a Γ-CCM or χ-CCM identity.
- **[`aiccm2026dev-b`](../user_guide/aiccm2026dev_b.md)**, χ-CCM, the
  finite-character (Γ-centred character-mesh) CCM line
  (`vibeqc.periodic.chi`). It uses the explicit Γ-centred
  character net, with 3D RHF/RKS/UHF/UKS plus RI-MP2 and local-PNO CCSD(T).
  Every 1D/2D absolute-energy backend fails closed.

Open-shell capabilities across both lines are documented in
[Open-shell AICCM](aiccm2026dev_open_shell.md).

```{toctree}
:maxdepth: 1
:caption: AICCM reference

../aiccm2026dev_a
../aiccm2026dev_a_followon
../user_guide/aiccm2026dev_b
../design_aiccm2026dev_b
../aiccm2026dev_b_decisions
```

```{toctree}
:maxdepth: 1
:caption: AICCM tutorials

../tutorial/cyclic_cluster_model
../tutorial/aiccm_quickstart
../tutorial/aiccm2026dev_a
../tutorial/aiccm2026dev_b
../tutorial/aiccm2026dev_b_localization
../tutorial/h8_chain_ccm_ancestor
../tutorial/equidistant_h_chain
```

```{toctree}
:maxdepth: 1
:caption: AICCM experimental

aiccm2026dev_route_matrix
aiccm2026dev_b_pno
aiccm2026dev_b_symmetry
aiccm2026dev_diamond_compare
aiccm2026dev_open_shell
aiccm2026dev_compare
aiccm2026dev_basis
aiccm2026dev_viz
aiccm2026dev_troubleshooting
```

### AICCM examples

The runnable examples live under `examples/periodic/`:

- [`aiccm2026dev_a_demo.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/aiccm2026dev_a_demo.py),
  Γ-CCM union-and-weight stack: 8-fold ERI symmetry check on 1-D/2-D/3-D
  lattices and the HF→MP2→CCSD(T) correlation ladder. It also reports the
  neutral four-center Madelung-background diagnostic as a separate control.
- [`aiccm2026dev_b_demo.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/aiccm2026dev_b_demo.py),
  χ-CCM: exercises all three ER backends in 3D; 1D/2D invocations demonstrate
  the intentional fail-close and do not return absolute energies.
- [`aiccm2026dev_b_mp2.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/aiccm2026dev_b_mp2.py),
  canonical RI-MP2 in 3D.
- [`aiccm2026dev_b_local_correlation.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/aiccm2026dev_b_local_correlation.py),
  DLPNO-MP2 and DLPNO-CCSD(T) on the χ-CCM finite torus.
- [`aiccm2026dev_diamond_bonds_bands_compare.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/periodic/aiccm2026dev_diamond_bonds_bands_compare.py),
  side-by-side diamond HF/KS property bundles and localized-orbital QVF archives;
  it records the cross-approach comparison as `not-defined` and emits no delta.
- [`benchmark_aiccm2026dev_b.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/regression/benchmark_aiccm2026dev_b.py),
  H4 convergence comparison against the historical and Γ-CCM weights.

The B/CRYSTAL fleet and future Γ-CCM/χ-CCM study inputs live under
[`studies/aiccm-2026/`](https://github.com/vibe-qc/vibe-qc/tree/main/studies/aiccm-2026).

### AICCM documentation map

| you want to… | read this |
|---|---|
| Run a minimal example | [Quickstart](../tutorial/aiccm_quickstart.md) |
| Run your first Γ-CCM calculation | [Γ-CCM tutorial](../tutorial/aiccm2026dev_a.md) |
| Run your first χ-CCM calculation | [χ-CCM tutorial](../tutorial/aiccm2026dev_b.md) |
| See all Γ-CCM methods and routes | [Γ-CCM reference](../aiccm2026dev_a.md) |
| See all χ-CCM methods, backends, and caveats | [χ-CCM user guide](../user_guide/aiccm2026dev_b.md) |
| See which route supports HF, DFT, MP2, DLPNO, or analytic gradients | [Route support matrix](aiccm2026dev_route_matrix.md) |
| Compare Γ-CCM UHF/UMP2, neutral-control UCCSD(T), and χ-CCM open-shell APIs | [Open-shell AICCM](aiccm2026dev_open_shell.md) |
| Study Γ-CCM and χ-CCM side by side (no current approach delta) | [Cross-stream comparison](aiccm2026dev_compare.md) |
| Choose the right basis set | [Basis sets for AICCM](aiccm2026dev_basis.md) |
| Debug a failing calculation | [Troubleshooting](aiccm2026dev_troubleshooting.md) |
| Generate orbital/density visualizations | [Visualization](aiccm2026dev_viz.md) |
| Understand the broad CCM concept | [CCM tutorial](../tutorial/cyclic_cluster_model.md) |
| Understand the χ-CCM derivation | [χ-CCM derivation](../design_aiccm2026dev_b.md) |
| Understand the theory (Γ-CCM paper) | [Archived Γ-CCM theory paper](https://vibe-qc.com/docs/) |

## See also

- [Roadmap](../roadmap.md), the v2.x CCM track and where this work fits.
- [Troubleshooting](../troubleshooting.md), known bugs and workarounds.
- The [Archived comparative manuscript](https://vibe-qc.com/docs/),
  detailed theory record for the Γ-CCM and χ-CCM streams.
