# Basis set development

vibe-qc ships production tooling for designing, optimising, and verifying
Gaussian basis sets, from molecular re-parametrisation to full periodic
library construction. This section collects the developer-facing
documentation for the basis-set toolchain.

```{toctree}
:maxdepth: 1
:caption: User guides

OPTIMISING_A_BASIS
```

```{toctree}
:maxdepth: 1
:caption: Design documents

ENERGY_GRADIENT_DESIGN
PERIODIC_GRADIENT_DESIGN
PLAN
GOAL4_DESIGN
GOAL8_MPEI_TZVP
STAGE0_IMPLEMENTATION_PLAN
```

```{toctree}
:maxdepth: 1
:caption: Verification & review

VERIFICATION_REPORT
REVIEW_BASIS_SETS_2026-05-08
HF_REV2_FAILURE_SCOUT_2026-05-18
REQUIREMENTS-PERIODIC
GENERATING_THE_BASIS_LIBRARY
ROADMAP_BASIS_LIBRARY
```

## See also

- [Basis sets](../user_guide/basis_sets.md), the bundled basis library.
- [Basis optimisation](../user_guide/basis_optimization.md), the in-process
  molecular optimiser user guide.
- [`examples/molecular/optimize_basis_h2o_pbe.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/molecular/optimize_basis_h2o_pbe.py),
  the runnable molecular optimisation example.
