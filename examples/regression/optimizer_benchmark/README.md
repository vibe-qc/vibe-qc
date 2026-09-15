# Optimizer Benchmark Suite

Systematic comparison of gradient-based geometry-optimization methods
for quantum chemistry, driven by vibe-qc's SCF + analytic gradients.

## Quick start

```bash
# List available systems and optimizers
python examples/regression/optimizer_benchmark/run_benchmark.py --list-systems
python examples/regression/optimizer_benchmark/run_benchmark.py --list-opts

# Run a quick benchmark — RHF/STO-3G, main optimizers
python examples/regression/optimizer_benchmark/run_benchmark.py \
    --systems h2o,ch4,c2h4 \
    --optimizers BFGSLineSearch,BFGS,LBFGS,FIRE,native \
    --output results/quick_test.json --report

# Full benchmark — RKS/PBE/def2-SVP, all systems & optimizers
python examples/regression/optimizer_benchmark/run_benchmark.py \
    --systems h2o,ch4,c2h4,benzene,h2o_dimer,stacked_benzene,glycine,aspirin \
    --optimizers BFGSLineSearch,BFGS,LBFGS,LBFGSLineSearch,FIRE,MDMin,native \
    --method rks --functional PBE --basis def2-svp \
    --output results/full_benchmark.json --report

# Post-hoc report with convergence plots
python examples/regression/optimizer_benchmark/run_benchmark.py \
    --load results/full_benchmark.json --report --plot-dir results/plots/

# Markdown tables (for paper)
python examples/regression/optimizer_benchmark/run_benchmark.py \
    --load results/full_benchmark.json --report --markdown
```

## Test systems

| System | Atoms | Category | Description |
|--------|------:|----------|-------------|
| `h2o` | 3 | covalent | H₂O — bent triatomic |
| `ch4` | 5 | covalent | CH₄ — tetrahedral |
| `c2h4` | 6 | covalent | C₂H₄ — planar alkene |
| `benzene` | 12 | covalent | C₆H₆ — aromatic ring |
| `h2o_dimer` | 6 | hbonded | (H₂O)₂ — H-bonded, flat PES |
| `stacked_benzene` | 24 | dispersion | (C₆H₆)₂ — π-stacked, dispersion-bound |
| `glycine` | 10 | torsion | Glycine — flexible amino acid |
| `aspirin` | 19 | medium_organic | Aspirin — ester + aromatic ring |

All starting geometries are perturbed from equilibrium so every
optimizer must descend.

## Available optimizers

| Optimizer | Family | Hessian? | Description |
|-----------|--------|----------|-------------|
| `BFGSLineSearch` | quasi_newton | approx | BFGS with line search (**vibe-qc default**) |
| `BFGS` | quasi_newton | approx | BFGS without line search |
| `LBFGS` | quasi_newton | approx | Limited-memory BFGS |
| `LBFGSLineSearch` | quasi_newton | approx | L-BFGS with line search |
| `FIRE` | inertial | none | Fast Inertial Relaxation Engine |
| `MDMin` | inertial | none | Velocity-Verlet MD + damping |
| `GPMin` | bayesian | surrogate | Gaussian Process minimizer |
| `ODE12r` | ode | none | Adaptive ODE-based optimizer |
| `GoodOldQuasiNewton` | quasi_newton | approx | Legacy ASE quasi-Newton |
| `SciPyFminBFGS` | quasi_newton | approx | SciPy BFGS via ASE |
| `SciPyFminCG` | cg | none | SciPy conjugate gradient via ASE |
| `native` | quasi_newton | approx | scipy L-BFGS-B (vibe-qc native, no ASE) |

## Architecture

```
optimizer_benchmark/
├── __init__.py          # Package docstring
├── test_systems.py       # Molecular system definitions (perturbed geometries)
├── benchmark_driver.py   # Core runner — ASE + native optimizer dispatch
├── report.py             # Results analysis, tables, convergence plots
├── run_benchmark.py      # CLI entry point
└── README.md             # This file
```

- **ASE optimizers** drive vibe-qc through `vibeqc.ase.VibeQC` (the
  ASE Calculator bridge). Per-step data is collected via ASE's
  optimizer callback mechanism.
- **Native optimizer** uses `vibeqc.molecular_optimize.optimize_molecule`
  (scipy L-BFGS-B with analytic SCF gradients). No ASE required.
- **Results** are JSON-serializable and can be loaded post-hoc for
  analysis — no need to re-run expensive calculations.

## Metrics collected

Per optimizer×system pair:
- Steps to convergence (gradient norm ≤ fmax)
- SCF evaluations (energy + gradient calls)
- Wall-clock time
- Initial and final energy (Ha)
- Initial and final max gradient component (Ha/bohr)
- Per-step energy trajectory (for convergence plots)

## Design rules (vibe-qc project)

- §10-compliant: no imports from other QC programs (PySCF, ORCA,
  Gaussian, …). All energies and gradients come from vibe-qc's own
  SCF + analytic gradient implementations.
- ASE is a *library* (infrastructure), not a QC program — allowed.
- Geometries stored as code (not external files) for reproducibility.
- Fixed random seed for deterministic perturbed starting geometries.

## Paper outline

This benchmark suite targets a methods-comparison paper:

1. **Introduction** — why optimizer choice matters for QC geometry
   optimization (flat PES, H-bonds, dispersion, large systems).
2. **Methods** — description of each optimizer family (quasi-Newton,
   inertial, conjugate gradient, Bayesian).
3. **Computational details** — vibe-qc version, SCF methods, basis
   sets, convergence criteria.
4. **Results** — per-family comparison tables, convergence trajectory
   plots, scaling analysis.
5. **Discussion** — which optimizer for which system class; Hessian
   approximation vs. line search trade-off; recommendations.
6. **Comparison with CRYSTAL / ORCA** — out-of-process reference
   calculations at identical method/basis/fmax.
