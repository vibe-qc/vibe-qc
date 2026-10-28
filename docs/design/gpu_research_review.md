# GPU concept: literature and architecture review

Research date: 2026-10-27. This supplements the [GPU concept](gpu_support.md),
using the configured paper-library catalog, selected full-text methods sections
and supporting information, recent primary online research, and source
inspection at `9cae35c7`. It authorizes no implementation, dependency, licence
change or vendor contact. No GPU or scientific acceptance calculation was run.

Round 3 rechecks source at `a3242f5c` and adds standalone Eigen reproductions,
not production calculations. The [G0/G1m plan](gpu_g0_g1m_plan.md) is the
actionable next-stage specification. Labels there distinguish repository,
primary-source, reproduced, proposed and hypothetical claims.

**Repository fact** means inspected source. **Primary-source fact** means
material in the cited paper or vendor documentation. **Proposal** is a design
choice; **hypothesis** requires measurement or qualification. Published timings
are not vibe-qc results. This is a focused 2025-2026 research update with older
foundational work, not an exhaustive literature survey.

## Paper-library research

| Paper and catalog key | Material examined | Consequence |
|---|---|---|
| Laqua et al., JCTC 2020, [10.1021/acs.jctc.9b00860][laqua20], `laqua_snlink_2020` | Exchange background and SI FLOP accounting, together with its 2021 extension | Grid contractions and three-center one-electron integrals are distinct workloads; collocation alone is not an exchange engine |
| Laqua et al., JCP 2021, [10.1063/5.0045084][laqua21], `laqua2021accelerating` | Sections II-IV, equations 5-12, section IV C/Figure 2 | Add a Metal seminumerical-exchange experiment; preserve contribution-dependent precision and the fixed-anchor update assumptions |
| Wu et al., WIREs CMS 2025, [10.1002/wcms.70008][wu25], `wu2025enhancing` | Implementation overview, DF workflow and separate SCF/gradient/Hessian benchmark descriptions | Measure complete workflows and qualify derivatives separately; package speedups are not portable predictions |
| Palethorpe and Barca, JCTC 2025, [10.1021/acs.jctc.5c00775][palethorpe25], `palethorpe2025high` | Local ChemRxiv v2 manuscript sections 2 and 4-5, SI batching/timing descriptions, publisher section 5 cross-check | Direct J remains worthwhile; register pressure, intermediate lifetimes and redundant work require dedicated design |
| Li et al., March 2026 preprint, [arXiv:2603.24881v1][gpw26], `pass20261012_li2026_implementationmultigridgaussianplanewave` | Sections II-IV, equations 28-29, algorithms 1-2 and grid parallelism | Give periodic GPW density/potential construction a concrete experiment; grid blocking and global-memory writes matter alongside FFT offload |

The local Palethorpe main manuscript is a preprint associated with the journal
DOI, not the publisher version. Its SI contains unresolved table references;
verify the publisher SI before adopting fixtures. Laqua 2021 SI was not
authenticated in this pass and is not used as evidence for its method or
benchmark settings. The 2021 main paper was read.

The Asadchev/Valeev three-center and four-center papers are catalogued under
[10.1021/acs.jctc.2c00995][threecenter] and
[10.1021/acs.jpca.3c04574][fourcenter]. They remain relevant G4 reading; this
pass does not claim a complete recurrence/SI review or adopt their algorithms.
Earlier reading of Barca, Pokhilko and Datta remains in the main concept.

## Recent online research

| Source, status and reading depth | Finding | Proposed use and limit |
|---|---|---|
| Huang, Shao and Hammond, April 2026, [10.1021/acs.jpca.6c00225][huang-journal]; full author text [arXiv:2601.08077v3][huang26], method and conclusion sections | Adaptive INT8 GEMM emulation targets DF-K, retaining J in FP64 and returning to native FP64 as required; the authors call the switching policy empirical without a general convergence guarantee | Strengthens the consumer-GPU DF-K experiment independently of cuEST; do not promote empirical qualification to a proof |
| Hayes and Gordon, April 2026, [10.1021/acs.jctc.5c02025][juliachem26]; publisher-indexed sections 3.2-3.3 and performance passages | Dense and screened DF algorithms have different crossover points; the AMD implementation is a proof of concept without the grouped batched products used on NVIDIA | Cite for algorithm selection, not a portable AMD performance expectation; full text/SI were not completely retrieved, so no screening implementation is prescribed |
| Li et al., JPCA 2025, [10.1021/acs.jpca.4c05876][li25]; [author manuscript][li-preprint] overview | Contracted-basis ERIs serve integral-direct Fock and nuclear-gradient construction | Keep direct J/K and gradients explicit; distinguish this paper from the Wu DF-focused contribution |
| Ramakrishnan and Motamarri, October 2026 revision, [arXiv:2604.26037v3][paw26]; sections 4.4-4.5 and method overview | Reduced precision and communication overlap are coupled to a particular residual-based subspace solver; benefits differ across platforms | Precision and asynchronous execution need operation-specific contracts; finite-element PAW evidence does not qualify Gaussian SCF or Metal |
| Lygatsika et al., [arXiv:2604.11139v3][abinit-preprint], revised August 31, 2026; author abstract rechecked | The Abinit plane-wave GPU port compares LOBPCG and Chebyshev filtering and emphasizes algorithmic changes alongside vendor libraries | Reprofile after offload; this abstract does not support our earlier vendor-specific or Rayleigh-Ritz statements, which are withdrawn |

None of these sources establishes a validated M5/M6 Gaussian HF/DFT route
meeting our intended accuracy. Apple performance remains a hypothesis, not a
reason to drop Apple from the objective or to declare it supported already.

## Corrections to the supplied refinement

### Operation boundary and scheduling

**Proposal:** adopt allocation, transfers, GEMM, strided-batched GEMM,
SYRK/HERK, TRSM, segmented reductions and grid operations as needed. Complex
support requires conjugation-aware operations. Capability records should
describe operation-specific precision, workspace and layout limits.

**Hypothesis:** concurrent J/K/XC work may help, but three streams can compete
for bandwidth, occupancy and scratch. Use plan-owned handles/workspaces and an
explicit dependency graph; sequential execution is a valid baseline. Measure
overlap rather than mandate it. Host waits should not occupy spinning OpenMP
teams.

A 2,000-by-2,000 FP64 matrix is 32 MB, but D in and J/K/Vxc out already move
128 MB per restricted DFT step, before factor traffic and packing. At an
assumed sustained 12.8-25.6 GB/s this is 5-10 ms: an arithmetic planning range,
not a measured PCIe result. DF-K taking seconds at that size is a hypothesis
to test. Complex/open-shell work increases the volume. Apple shared buffers
may avoid explicit copies but still incur traffic and synchronization.

### DF storage, streaming and factorization

**Repository fact:** [df.cpp](../../cpp/src/df.cpp), constructor around lines
295-419, retains `T_flat_` and `B_per_P_`, with additional construction
temporaries. `T_flat_` also serves `compute_B_metric` and `contract_B_with`
around lines 850 and 869. Removing it globally is not automatically safe.

**Repository fact:** [DFJKBuilder](../../cpp/src/jk_builder.cpp), lines
332-375, uses density-form K, not its separate occupied-factor or streaming
variants. `build_K_mo` exists in [df.cpp](../../cpp/src/df.cpp), lines 747-759.

**Repository fact and standalone reproduction:** the unused streaming methods
apply the full `n_aux`-row triangular solve to an `nb`-row RHS when
`nb < n_aux` (lines 775-776 and 813-814). A debug Eigen reproduction asserts.
Full auxiliary blocks avoid that mismatch but still map column-major tensor
rows as contiguous row-major matrices, contrary to
[df.hpp](../../cpp/include/vibeqc/df.hpp), lines 353-354 and the packing at
df.cpp lines 341-348. A deterministic `n=3`, `n_aux=4`, `L=I`, `D=I`
example with `B[P,i,j]=sin(1+P+2*(i+j)+i*j)` (zero-based indices) gives
max K error 5.95973 against reference max 6.12590. This reproduces the
algebra/layout bug, not a complete native QC calculation. No block-size
regime makes this a generally correct implementation; special inputs may
coincidentally agree. No in-tree callers or Python bindings were found.
Delete or rewrite it with a test in G0; no production code changes here.

**Repository fact:** after construction, the other `T_flat_` consumers are
`compute_B_metric` and `contract_B_with`. Their intended definitions depend
only on B: `S=B_flat B_flat^T`, `alpha=B_flat vec(M)`, using the same pair
ordering for both operands. The robust-COSX function currently returns its
input K unchanged ([cosx_robust.cpp](../../cpp/src/cosx_robust.cpp), lines
15-37); the DF derivative kernels do not consume `T_flat_`.

**Additional reproduced defect:** the current metric helper solves with
`L^T` on the left of the transposed intermediate (df.cpp lines 854-857).
This produces `L^-1 W L^-1`, not the documented `L^-1 W L^-T`.
The same deterministic tensor with lower-triangular rows `(2,0,0,0)`,
`(1,3,0,0)`, `(.4,.5,2,0)`, `(.2,.3,.4,1)` gives max discrepancy
0.882744 from `B_flat B_flat^T`. Use the direct B definition as the test
oracle; preserving this inactive helper's old output is not correctness.

**Proposal:** distinguish construction peak, steady-state host/device storage,
derivative requirements and vendor scratch. Blocking RHS columns preserves the
full auxiliary dimension of `L B = T`; blocking auxiliary rows requires proper
block forward substitution and its dependencies. T has the same dense size as
B. Do not promise that memory is halved until the selected consumer and peak
have been measured.

One resident B representation can remove the persistent raw tensor once these
consumers and the public C++ `T_flat()` accessor contract are addressed. Do
not keep a full B-flat copy alongside `B_per_P_` and call that one tensor.
The G0 plan specifies blocked views/packing and separate peak measurements.

Where occupied coefficients and nonnegative occupations are available, a
weighted factor `X = C sqrt(f)` gives `D = X X^T`. Preserve spin conventions.
At the current density-only boundary, pivoted Cholesky is an alternative to
assess, not an unqualified prerequisite hidden in G0. Rank truncation and
factorization residuals need an explicit K/Fock error budget. Negative-pivot
handling alone is insufficient. Indefinite input densities need the density
path, not silent projection onto the positive cone. Separate this algorithmic
optimization from changing GPU arithmetic so accuracy and speed comparisons
remain interpretable.

### Precision and convergence

**Primary-source fact:** [cuBLAS section 1.5][cublas] distinguishes dynamic
mantissa control with native fallback from fixed mantissa control, and documents
special-value handling, version-dependent defaults and workspace effects.
An emulated GEMM does not establish FP64 arithmetic for every operation.

**Proposal:** record `native64`, `emulated` and `mixed32` per operation, with
settings and actual fallback. Keep initial `strict` native FP64. A later policy
could admit accuracy-qualified emulation with an explicit input/error contract,
runtime controls and independent checks. A passing device benchmark alone does
not establish all IEEE semantics or a universal accuracy guarantee.

Likewise, `1e-10 Ha` is a possible small-case regression target, not a consequence
of using FP64. Conditioning, reduction length/order and nonlinear state selection
matter. Compare fixed-density J/K/Vxc matrices and energy contributions before
full SCF, holding fitting, grids and screening constant.

Accept resetting DIIS and builder caches at the switch to strict arithmetic.
However, two consecutive converged iterations do not prove global ground-state
identity or stability. Do not demand Aufbau occupations or unchanged orbital
ordering for all routes: [mom.hpp](../../cpp/include/vibeqc/mom.hpp), lines
1-17 and 47-51, explicitly supports non-Aufbau occupied-subspace selection.
Degenerate rotations and fractional occupations also defeat a simple ordering
test. Preserve the requested occupation policy and compare occupied subspaces
in the overlap metric, densities and applicable stability diagnostics.

A final diagonalization can change D. Report energy, residual, Fock, density
and occupations from a mutually consistent final state; rebuild/check after a
density change. Treat `tau32`, stagnation counts and iteration limits as
experiment parameters with documented norm normalization. Exhausting a recovery
time budget may justify continuing strictly from the recovered state; tie full
restarts to numerical-integrity or state failures. Call this a verification
record, not a universal mixed-precision certificate.

### Apple exchange

**Primary-source fact:** Laqua 2021 section IV A partitions integral
contributions into omitted, FP32 and FP64 categories using contribution bounds.
Section IV C uses a mixed-precision anchor, then FP32 increments relative to
that fixed anchor with tighter screening. This differs from recursive FP32
Fock accumulation and from a generic all-FP32 SCF followed by an energy check.

**Proposal:** measure Metal collocation, DF-shaped products and complete
seminumerical K batches separately. Significant FP64 contributions need CPU
evaluation or another qualified implementation. Budget routing, merging and
recovery costs. Sn-LinK and our COSX route are related, but their grids,
screening and overlap corrections cannot be assumed identical. Reproducing a
paper's arithmetic policy requires demonstrating those correspondences.

### Roadmap and business scope

Qualify each molecular route as it lands; do not wait for all grid/Metal work.
G0 should produce measurements and a bounded storage design, not require an
unmeasured factorization rewrite or guaranteed memory reduction. A one-week
Metal experiment is a timebox, not guaranteed feasibility or failure evidence.
Raw machine-specific profiling logs stay outside the product checkout; only
public-safe benchmark definitions and summaries belong in documentation.

Retain the open-core recommendation and optional cuEST route. Copyright tracking
does not itself confer relicensing rights over outside contributions.
User-installed dynamic loading does not settle all distribution/compatibility
questions. The refinement's categorical licence approval is not adopted; the
main concept's scoped assessment still applies.

## Next concept round

1. **G0:** profile explicit direct, actual AUTO default, DF and RIJ-COSX at
   several sizes; repair/delete dead streaming code, audit storage, and specify
   spin, layout, occupation and error contracts. Establish a CPU shortlist.
2. **G1:** qualify a blocked DF-K primitive and minimal CUDA/HIP adapter.
   Keep factorized-density acceleration an independent comparison.
3. **G1m:** test Metal collocation, DF-shaped GEMM and complete exchange batches
   early; select the first Apple path by error and complete-cost evidence.
4. **G2:** integrate and qualify molecular DF-J/K if the G0 plus G1 gate
   favors it over a grid pilot. CPU-only G0 cannot demonstrate a device win.
5. **G3:** extend XC/seminumerical exchange, compare native work with GauXC,
   and qualify gradients separately. G1m may change Apple's ordering.
6. **G4-G6:** retain measured-bottleneck integral, periodic and correlation
   extensions. Periodic grid work now has a concrete 2026 GPW reference;
   molecular atom-centered quadrature and periodic uniform grids remain distinct.
7. **N1:** keep cuEST independent, using native math for its first correctness
   study and separately reported emulation experiments.

No stage is implemented here. Device allocation, numerical admission limits,
measured crossover points, timing targets and the commercial decision remain
open for another review round.

[laqua20]: https://doi.org/10.1021/acs.jctc.9b00860
[laqua21]: https://doi.org/10.1063/5.0045084
[wu25]: https://doi.org/10.1002/wcms.70008
[palethorpe25]: https://pubs.acs.org/doi/10.1021/acs.jctc.5c00775
[threecenter]: https://doi.org/10.1021/acs.jctc.2c00995
[fourcenter]: https://doi.org/10.1021/acs.jpca.3c04574
[huang-journal]: https://pubs.acs.org/doi/10.1021/acs.jpca.6c00225
[huang26]: https://arxiv.org/html/2601.08077v3
[juliachem26]: https://pubs.acs.org/doi/10.1021/acs.jctc.5c02025
[li25]: https://doi.org/10.1021/acs.jpca.4c05876
[li-preprint]: https://arxiv.org/html/2407.09700v1
[gpw26]: https://arxiv.org/abs/2603.24881v1
[paw26]: https://arxiv.org/html/2604.26037v3
[abinit26]: https://www.sciencedirect.com/science/article/pii/S0010465526004492
[abinit-preprint]: https://arxiv.org/abs/2604.11139v3
[cublas]: https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation
