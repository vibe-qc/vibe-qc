# Internal translation CCSD residual benchmark

Run `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m benchmarks.chi_ccsd_residual --cells 1 2 3 4 8`.
This is one arbitrary-amplitude residual/energy snapshot of the synthetic
nonorthogonal frozen-partition model, with full virtual domains. It does not
measure iterative convergence or scientific acceptance. Measurements below
use a rebuilt native module, Python 3.14, single-thread BLAS on arm64.

| Cells | Transport preparation (s) | Residual (s) | Unreduced oracle (s) | CC reserved bytes | Python traced peak bytes | Pair reconstructions |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0015 | 0.0108 | 0.0018 | 35136 | 35293 | 1 |
| 2 | 0.0019 | 0.0581 | 0.0024 | 196864 | 29095 | 65 |
| 3 | 0.0021 | 0.1674 | 0.0061 | 626240 | 25205 | 235 |
| 4 | 0.0033 | 0.5786 | 0.0126 | 1496064 | 34475 | 757 |
| 8 | 0.0123 | 8.1413 | 0.1490 | 13398016 | 186745 | 5474 |

Maximum residual disagreement with the shared unreduced engine is
5.35e-16; maximum raw per-cell energy disagreement is 1.74e-18 Ha.
The translation route is 6 to 55 times slower on these small cases. Repeated
pair/factor reconstruction and callback dispatch dominate; this increment
establishes complete compiled algebra, not improved runtime scaling.
The diagnostic `terms=True` output requires a larger reservation.

Tracemalloc measures Python allocations and excludes native Eigen/BLAS scratch.
The CC reservation covers controlled numerical payload, not total RSS. The
benchmark separately reports transport reservation, projected recipes,
occupied Fock and returned residual arrays. At eight cells these are 392664,
32000, 512 and 10368 bytes respectively. Source storage and construction are
outside the timed snapshot. Quadratic occupied-pair overlap scratch remains;
full-domain pair ranks also grow with the mesh. No MP2 memory claim is inherited.

See [the internal contract](../docs/translation_ccsd_residual.md) for supported
scope, cache cleanup, budget exclusions and remaining iterative-driver work.
