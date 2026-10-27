# GPU support concept and implementation plan

**Status: discussion draft, 2026-10-27.** This is a proposal for iterative
technical and product review, not an approved implementation, dependency
addition, licensing change or performance claim. No GPU calculation was run
for this document. The source assessment starts at `0cbc072f`.

## 1. Recommended direction

Build an optional, open-source acceleration layer inside vibe-qc, with a
CPU reference implementation and CUDA, HIP/ROCm and Metal backends. Share
mathematical operations, layouts, error policies and tests across vendors.
Do not assume that one kernel language or one arithmetic mode will perform
well on every GPU.

Start with density-fitting contractions and XC quadrature. Bring Apple into
the initial feasibility phase, alongside NVIDIA and AMD, so portability is
tested before interfaces become entrenched. Keep the initial scope to
molecular single-point HF, pure DFT and global hybrid DFT. Periodic kernels
remain an explicit next phase, not a promised consequence of molecular support.

Develop the layer so it could later become a reusable library. Keep it in
this repository until a second real consumer establishes the required API.
Do not build a general tensor compiler, new BLAS or a complete GPU QC program
before delivering one accelerated vibe-qc workflow.

Treat NVIDIA cuEST as a separate, lower-priority optional backend and
collaboration experiment. The portable implementation must work without it.

## 2. Hardware and programming model

| Target | Proposed backend | Arithmetic and support policy |
|---|---|---|
| NVIDIA datacenter | CUDA, cuBLAS, cuSOLVER, cuFFT as needed | Native FP64 reference where efficient; qualify accelerated arithmetic separately for each architecture |
| NVIDIA RTX professional and GeForce | Same CUDA backend | Professional branding does not guarantee high FP64 throughput; qualify native FP64 and controlled mixed precision independently |
| AMD Instinct | HIP with rocBLAS/hipBLAS, rocSOLVER/hipSOLVER and rocFFT | Native FP64 starting point on appropriate devices; use the tested ROCm and OS combination |
| AMD Radeon PRO and gaming Radeon | Same HIP backend | Admit exact device, driver, library and OS combinations; do not promise every Radeon or every Windows configuration |
| Apple M5/M6 family | Metal, Metal Performance Shaders where useful, custom Metal kernels | FP32 GPU work with explicit error control and CPU FP64 recovery; no assumption of native GPU FP64 |
| CPU-only systems | Existing Eigen/BLAS/LAPACK and native kernels | Always installable and usable without a GPU SDK |

CUDA is NVIDIA's platform. ROCm is AMD's software stack; HIP is a programming
interface that can target AMD and NVIDIA. Neither supplies the Apple route.
Prefer a small CUDA/HIP-compatible custom kernel subset, with vendor BLAS
behind our own numerical interface. Avoid warp-width assumptions: subgroup
sizes, atomics, reductions and workgroup limits are backend capabilities.
See AMD's [HIP porting guide][hip] and [ROCm platform information][rocm].

Apple's [Metal language specification][metal], section 2.1, excludes `double`.
Its [feature tables][metal-features] distinguish GPU families and capabilities.
Apple has announced [M6 hardware][m6], but that announcement does not establish
FP64 capability or QC performance. Query runtime capabilities and qualify each
M5/M6 target; never infer numerical support from a chip name. GPU matrix
accelerators and the separate Neural Engine are not interchangeable. The
Neural Engine is outside this initial plan.

Alternative portability layers deserve a bounded experiment, not a platform
commitment. [Kokkos][kokkos] lists CUDA, HIP and SYCL device backends, without a
Metal backend in that configuration guide. [AdaptiveCpp's Metal backend][acpp]
now exists, but is explicitly experimental and lacks `double`. This makes
SYCL a credible experiment rather than a guaranteed production solution for
all three vendors. CuPy, MLX, PyTorch and OpenMP offload can help prototypes,
but replacing NumPy calls alone would not accelerate the native SCF drivers.

## 3. What should run on GPUs?

The ordering below is a hypothesis to test with complete calculations.
Kernel suitability does not establish that the kernel dominates our runtime.

| Work | Opportunity | Initial decision and limitation |
|---|---|---|
| DF/RI exchange K and AO-to-MO transforms | Large matrix products with reusable three-index factors | First contraction target; block over auxiliary and orbital indices |
| DF/RI Coulomb J | Regular contractions, but often bandwidth limited | Implement with K; measure transfers and memory traffic rather than assuming GEMM-like speedup |
| XC integration | Batched AO values/derivatives, density formation and potential accumulation | First DFT target; start LDA/GGA, then meta-GGA and derivatives |
| Three-center and two-center integrals | Repeated Gaussian work with more manageable storage than four-index ERIs | Second wave; initially generate factors on CPU and reuse on GPU |
| Direct four-center J/K | Large potential gain, difficult screening, recurrence and reductions | Later dedicated work; bin shell classes and fuse integral evaluation with contraction |
| Dense eigensolves, Cholesky and triangular solves | Good at sufficient matrix size on suitable hardware | Use vendor solvers selectively; keep rank decisions and Apple FP64 solves on CPU initially |
| Periodic FFT/Poisson and GPW grid work | Regular transforms and grid contractions | Later periodic phase; preserve normalization, boundary conditions and complex phases |
| Periodic GDF and k-point contractions | Repeated complex matrix products and block reuse | Later periodic phase; exploit k/q batching without replicating all tensors |
| RI-MP2, CCSD and perturbative triples | Substantial tensor contractions | Reuse the contraction layer after SCF; amplitudes and intermediates still constrain memory |
| DLPNO and local correlation | Many small tensor operations | Group compatible batches; sparsity and tiny domains can favor CPU |
| Gradients, response, Hessians and MD | Repeated accelerated builds, derivative integrals and contractions | Separate acceptance after energies; energy accuracy alone cannot qualify forces |
| Semiempirical SCF and parameter fitting | Batched independent systems and selected linear algebra | Useful later; small individual calculations may remain faster on CPU |
| Parsing, basis lookup, SCF decisions, DIIS coefficient solve, output | Little bulk parallel work | Keep orchestration and small solves on CPU |
| Irregular graph searches and screening setup | Divergence and unpredictable memory access | CPU preparation initially; offload only after profiling |

There is no fundamental rule that integrals are the only GPU work, or that
correlation and periodic calculations cannot run there. The practical limits
are precision, memory, parallel structure, transfer cost and implementation
effort. GPU execution does not remove unfavorable asymptotic scaling.

For real molecular DF factors, the existing operations already expose useful
building blocks:

```text
g[P]       = sum_mn B[P,m,n] D[m,n]
J[m,n]     = sum_P  B[P,m,n] g[P]
T[P,m,i]   = sum_n  B[P,m,n] C_occ[n,i]
K[m,n]     = sum_Pi T[P,m,i] T[P,n,i]
```

The occupied-orbital form requires the correct occupations and spin factors.
Keep a density-based K operation for fractional occupations and density guesses.
Complex periodic forms require the appropriate conjugations and momentum
selection, not a mechanical reuse of these real molecular expressions.

Memory is a design constraint from day one. An unpacked FP64 B tensor with
2,000 orbitals and 6,000 auxiliary functions occupies 192 GB in decimal units,
before workspace. A 16-24 GB gaming card therefore needs blocking, streaming,
recomputation or a more local representation. Unified memory avoids some
explicit copies on Apple but does not remove capacity, bandwidth or
synchronization limits. Do not rely on operating-system swapping for performance.

## 4. Where the layer fits in the current code

The native [JKBuilder](../../cpp/include/vibeqc/jk_builder.hpp) already
separates Coulomb, exchange, paired-spin exchange and range-separated exchange.
[DFJKBuilder](../../cpp/src/jk_builder.cpp) delegates to the native
[density-fitting implementation](../../cpp/src/df.cpp). The molecular
[RKS driver](../../cpp/src/rks.cpp) selects this builder. This is the production
integration seam; the [Python DF implementation](../../python/vibeqc/density_fitting.py)
is also useful for small independent reference checks.

The present builder returns host `Eigen::MatrixXd` objects. A first adapter can
retain B and scratch space on the device and return J/K to the CPU. That gives
a bounded integration, but not a fully resident SCF loop. Measure that boundary
before extending it with device-resident operations. A faster backend hidden
behind mandatory full-tensor copies would defeat the design.

Proposed layers:

```text
vibe-qc methods, SCF control, conventions and scientific acceptance
    |
    +-- JKBuilder / XC integration / periodic operator adapters
            |
            +-- device context, buffers, execution plans, error policy
                    |
                    +-- CPU numerical operations
                    +-- CUDA numerical operations
                    +-- HIP numerical operations
                    +-- Metal numerical operations
                    +-- optional cuEST component adapter
```

Implement internal C++ interfaces first, with thin Python bindings where
needed. A later independent library can expose a versioned C ABI with opaque
handles. The reusable portion should accept explicit basis metadata or tensor
blocks, not depend on a vibe-qc SCF object, output writer or job scheduler.

The initial operation set should be deliberately small: device allocation and
transfers, batched GEMM, reductions, DF-J/K plans and grid batches. FFTs,
eigensolvers and generic tensor contractions enter when a real consumer needs
them. Avoid exposing proprietary handles or Eigen types in an eventual public
ABI. Synchronization, ownership, array strides and scratch-space budgets must
be explicit. Keep buffers resident across iterations and reuse execution plans.

Backend selection and numerical-method selection are separate. Requesting a
GPU must not silently change direct HF into density-fitted HF, change a grid,
or substitute an auxiliary basis. An automatic hardware choice may fall back
with a recorded reason; an explicitly required unsupported backend should fail
with an actionable message. Record actual device, backend/library versions,
precision policy, fallback and source/native identity through `vibeqc.output`.
All proposed option names and interfaces remain subject to review.

The native RKS route currently rejects range-separated hybrids with
`density_fit=true`. A cuEST capability or new GPU kernel does not automatically
remove that restriction: route semantics, range operators and derivatives need
their own admission and validation.

## 5. Precision is part of the algorithm

Propose two separately qualified modes:

1. **Strict:** native FP64 operations on CPU or suitable GPU; no implicit TF32,
   FP16 or other reduced-precision substitution. Strict Apple operation may
   execute accuracy-critical work on CPU.
2. **Verified mixed:** accelerate selected work at lower precision, then test
   residuals and observables using an independent FP64 path. Refine or return
   to FP64 computation when the error budget cannot be met.

For the Apple pilot, retain FP64 source factors on the CPU, construct FP32
device blocks, and evaluate early SCF contractions there. Keep overlap/metric
factorization, linear-dependence decisions, eigensolves and sensitive scalar
reductions on CPU. At the transition, rebuild from the original FP64 data and
perform as many strict SCF iterations as required to meet the requested
residual and energy tolerances. A single final energy recomputation is not a
convergence certificate. Recheck the density, occupations and state identity;
near-degenerate states may require a full strict restart.

FP64 accumulation cannot recover input information already discarded by FP32
conversion. Cancellation, diffuse bases, ill-conditioned metrics, tiny gaps
and long reductions need targeted tests. Pairwise or compensated sums help
with accumulation error but do not replace precision analysis. Threshold
decisions that control screening or certified omission bounds need conservative
arithmetic; an ordinary mixed-precision kernel must not inherit a MANTA bound
or certificate merely because it returns a similar energy.

Two-FP32 representations and integer/tensor-engine emulation are possible
research paths. They are not automatically IEEE FP64 and are outside the first
Apple prototype. NVIDIA documents tunable [GEMM emulation in cuEST][emulation],
including an explicit native-FP64 mode; the adapter must set and report its
policy rather than silently accept vendor defaults.

Published mixed-precision evidence is method-specific. Pokhilko et al.,
[10.1021/acs.jctc.8b00321][pokhilko], section 2 and SI section 1/Figure S2,
retain the SCF/integral preparation in double precision and use cleanup in
correlated iterations. This supports testing selective precision and recovery,
not claiming that all HF/DFT work is safe in FP32. Our Apple SCF recovery is a
proposed extension requiring its own evidence.

## 6. Implementation sequence and roadmap fit

The existing [acceleration program](../roadmap.md#acceleration-program-multi-release)
places general GPU production capability in `v1.x`, and already names GPU-XC1.
Keep that production theme. Pull profiling, interface work and experimental
prototypes forward now, without making them a new prerequisite for molecular
or periodic parity and without assigning a release version prematurely.

| Stage | Deliverable | Exit evidence and dependency |
|---|---|---|
| G0: measure and design | Component timings, memory estimates, operation contracts and CPU baselines | Representative HF, PBE and PBE0 workloads; identify real bottlenecks and read accepted scientific contracts |
| G1: feasibility on three vendors | One blocked DF-K operation on CUDA, HIP and Metal, with CPU DF factors | Same inputs and explicit precision; measure copies, initialization, peak memory and crossover; Apple includes strict recovery cost |
| G2: molecular integration | Optional native GPU DF-J/K behind JKBuilder; RHF/UHF and selected RKS/UKS global hybrids | Complete `vq` calculations against CPU and matched ORCA; missing SDK and low-memory fallback checked |
| G3: DFT grid and derivatives | AO collocation, density and XC matrix batches; then gradients | Start LDA/GGA; independent grid and functional parity, energy and force acceptance; connect to GPU-XC1 |
| G4: remove measured bottlenecks | Native 2c/3c generation, larger resident workflow and selected solvers | Implement only where G2/G3 profiles justify it; direct four-center builds are a separate subproject |
| G5: periodic acceleration | One existing validated Gamma route, then complex multi-k GDF or GPW operations | CRYSTAL comparisons where applicable, phase/symmetry tests, energies per cell, forces/stress as claimed |
| G6: correlated scaling | RI-MP2 then selected CCSD/(T) and batched local-correlation work | Memory-bounded tensors and scientific references; fits the production correlated-scaling `v2.x` theme |
| N1: optional NVIDIA study | cuEST component adapter and bounded molecular demonstration | Separate licensing/packaging assessment, exact method equivalence and permitted publication; does not block G1-G6 |

G2 depends on G1, G3 on the common device infrastructure and an admitted SCF
route, and G4 on measurements. G5 and G6 can proceed independently once their
shared operations and CPU routes qualify. Multi-GPU execution follows a useful
single-GPU implementation and the existing MPI plan; it is not an initial
prerequisite. Parameter fitting can later reuse qualified backends and batch
independent cases without changing its objective or reference protocol.

Planning allowance, not a delivery commitment: G0-G1 about 3-6 engineer-weeks;
G2 another 4-8; G3 another 4-8. These assume access to NVIDIA and AMD hardware,
GPU expertise and existing valid reference cases. A robust direct integral
engine, periodic coverage and CC are separate multi-month efforts. Re-estimate
after G1, especially if Apple recovery or device-memory traffic erases the gain.

The first prototype should compare at least a small overhead-dominated system,
a medium molecular DF hybrid, and a larger memory-constrained case. Carry the
same blocked contraction to all three backends before expanding method scope.
GauXC is a serious reuse candidate for G3: its project supports CUDA and HIP
XC/K integrators, but does not advertise a Metal backend. Compare an optional
GauXC adapter with native kernels for fidelity, dependencies and maintainability
before choosing. Do not substitute its grid/functional choices silently.
See [GauXC's implementation and dependency description][gauxc].

## 7. Acceptance and performance policy

Each feature has source checks and subsequent scientific acceptance through
`vq`; development landing does not wait for unrelated builds or calculations.
Ordinary development skips CI under repository policy. Release candidates run
their required CI; GPU feature claims also require real device evidence.

Numerical qualification has two levels: backend versus the same CPU algorithm,
then the complete calculation versus ORCA or CRYSTAL with matched geometry,
basis/ECP, fit, grid, functional convention, spin, thresholds and units. Separate
floating-point error from density-fitting/grid approximations. Do not call an
exact-versus-DF comparison a pure hardware comparison.

Suggested starting budgets for small/medium molecular qualification cases are
`1e-8 Ha` total-energy backend difference and `1e-6 Ha/bohr` maximum gradient
component difference, while also satisfying the requested SCF residual and
occupation/state checks. These are proposed engineering targets, not universal
chemistry tolerances or observed results. Establish size-dependent budgets and
relative-energy/response requirements before admitting larger systems. External
reference tolerances must account explicitly for any unavoidable algorithmic
differences rather than loosening the backend comparison.

Include open-shell densities, fractional occupations, diffuse/high-angular-
momentum bases, near-linear dependence, low-memory tiling, device failures and
unsupported functional requests. Numerical reproducibility means agreement
within documented tolerances; cross-vendor bitwise equality is not promised.
Use deterministic reductions where needed for diagnosis.

Benchmark cold and warm complete jobs, including initialization/JIT, integral
preparation, transfers, strict recovery, iteration count and output. Compare
against a tuned CPU build using its intended native module and BLAS, not an
accidentally serial baseline. Publish total time, component times, peak device
and host memory, and accuracy. A possible G2 product target is at least 2x
end-to-end speedup for a declared workload, while small cases choose CPU;
this target is subject to G0/G1 findings and is not a promised result.

Amdahl's law illustrates why this matters: if 80% of runtime becomes 10x
faster, the maximum total speedup is `1 / (0.2 + 0.8/10) = 3.57x`, before
overheads. Keep existing locality, screening and convergence work: GPUs
complement those algorithms. Neither advertised AI throughput nor an isolated
GEMM benchmark establishes useful QC throughput.

## 8. Reusable library and commercial boundary

**Recommendation: keep the common backend API, native portable kernels,
precision policy and correctness tests under vibe-qc's MPL-2.0 licence.**
GPU support improves the core product and community reach. Reserving all GPU
execution for a proprietary edition would weaken vendor portability, public
reproducibility and contributions, and create a second scientific code path.

A commercial company can build paid deployment, managed compute, workflow
automation, support, qualification services and independently developed
extensions around the open core. For Cyclic Labs, this is the recommended
boundary; it is a product proposal, not a decision already made. If a future
standalone GPU library has outside consumers, reconsider packaging and any
additional licensing offer then, with contributor rights accounted for.

Mozilla's [MPL FAQ][mpl] describes file-level copyleft and combination with
separate proprietary code. It does not erase other dependency obligations.
The repository's [licensing inventory](../license.md) already flags FFTW's
GPL implications for distributed binaries. Review the actual dependency/link
graph before offering a proprietary combined package. A plugin boundary or
dynamic loading alone is not a legal exemption. No licensing changes are made
by this draft.

## 9. Lower-priority NVIDIA collaboration route

The library in question is **cuEST**, a CUDA electronic-structure component
library, distinct from cuQuantum quantum-circuit simulation tools. It is
[publicly obtainable][cuest-install], with C and Python interfaces and
[sample code][cuest-samples]. Public availability does not mean the library
is open source. NVIDIA currently describes it as beta on its
[product page][cuest-product].

The [cuEST overview][cuest] specifies Gaussian bases with **nonperiodic vacuum
boundary conditions** and **density-fitted J/K**. It also describes XC,
range-separated exchange and derivative components. It supplies building
blocks; vibe-qc remains responsible for SCF, occupations and observables.
Proposed use: HF consumes J/K, pure DFT consumes J/XC, and global hybrids add
scaled K. Range-separated hybrids need separate operator/convention admission.
No general periodic, arbitrary ERI-output or post-HF capability follows from
that scope. Molecular/finite-cluster use does not establish periodic CCM
correctness.

The installation guide currently describes Linux x86-64 and ARM64, CUDA
12/13, and Python bindings requiring Python 3.12 or newer. Its tested GPU
list includes A100/H100/H200/B200 and RTX Pro 6000, not an explicit GeForce
qualification list. Match the chosen release and hardware before promising
gaming support; the C adapter can avoid making these Python bindings a core
Python-version requirement. This is not a macOS backend.

Technical diligence must cover shell ordering/normalization, spherical versus
Cartesian bases, auxiliary fitting and metric treatment, spin/occupation
factors, grids, B3LYP variants and range parameters. In particular, vibe-qc
documents patched Libxc spin-fraction kernels while cuEST documents its own
Libxc build. Equal functional names therefore require numerical equivalence
checks, including polarized low-density limits. Use the lower-level XC
interface if that is necessary to retain the intended functional evaluation.

The current [cuEST licence][cuest-license], sections 1 and 2, imposes
NVIDIA-platform and redistribution conditions. Section 2.12 requires prior
written permission for disclosing specified benchmark/performance outputs.
Keep vendor binaries separate initially, assess the actual distribution, and
resolve publication rights before proposing a public comparative benchmark.
Do not infer that freely downloadable software can be relicensed as MPL.

Proposed collaboration package: one molecular PBE0 demonstration, a matched
native-versus-cuEST correctness study, documented integration gaps, and a
request for engineering access, hardware/cloud resources and sponsored
implementation work. Agree on publication rights, maintainership and vendor
independence. Do not make funding or exclusivity prerequisites for the portable
project. This document authorizes no outreach or funding application.

[NVIDIA Inception][inception] is a possible startup relationship route, not
evidence of a cash award. The [Academic Grant Program][grants] page currently
says it is not accepting new applications and describes academic eligibility;
do not present it as an available company grant. A direct sponsored engineering
agreement would require a separate discussion. No funding is assumed in the
implementation estimates.

## 10. Scientific sources and review questions

Local-library reading supporting the design, beyond vendor API documentation:

| Source | Sections examined | Design consequence |
|---|---|---|
| Barca et al., 2021, [10.1021/acs.jctc.1c00720][barca] | Sections 4-5, screening/binning equations 23-24 and shell-batch organization | Screen and group integral work before offload; minimize transfers; do not copy a CPU quartet loop verbatim |
| Pokhilko et al., 2018, [10.1021/acs.jctc.8b00321][pokhilko] | Section 2, plus SI section 1/Figure S2 | Selective precision and strict cleanup are testable strategies; the evidence is not blanket FP32 approval for SCF |
| Datta and Gordon, 2023, [10.1021/acs.jctc.3c00876][datta] | RI amplitude formulation around equations 1-16 and GPU contraction/transfer discussion | Reuse three-index factors and tile dominant CC contractions; a useful general library extends beyond integrals |

These papers are implementation precedents, not vibe-qc benchmark results.
No new recurrence, mixed-precision proof or derivative algorithm is adopted
here. Before implementing those stages, complete the method-specific primary
paper/SI review and citation entries through the configured literature library.
Vendor emulation is assessed as a black-box optional component in this draft.

Questions for the next design iteration:

1. Is the priority order DF-K plus J, XC, then integral generation right for
   the actual workloads, or do G0 profiles justify a different first kernel?
2. Can the first Metal experiment recover strict SCF accuracy cheaply enough
   to be useful? What cases should force CPU from the start?
3. Should G3 use optional GauXC on CUDA/HIP, and how should it share grids and
   functional definitions with Metal and existing CPU routes?
4. Is an internal C++ layer with a later C ABI sufficient, or is there already
   a concrete external library consumer that changes this priority?
5. Does the proposed open core and commercial services/extensions boundary
   fit the intended business, without creating duplicate science engines?
6. Which exact NVIDIA, AMD and Apple devices and reference cases can qualify
   G1/G2? Device access is an implementation dependency, not a reason to claim
   untested coverage.
7. Does cuEST's molecular scope and publication/licensing policy justify N1
   as a bounded collaboration experiment?

[hip]: https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.0/how-to/hip_porting_guide.html
[rocm]: https://www.amd.com/en/products/software/rocm.html
[metal]: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
[metal-features]: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
[m6]: https://www.apple.com/newsroom/2026/08/apple-unveils-a-more-powerful-mac-mini-featuring-the-all-new-m6-and-m5-pro/
[kokkos]: https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html
[acpp]: https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/install-metal.md
[gauxc]: https://github.com/wavefunction91/GauXC
[mpl]: https://www.mozilla.org/en-US/MPL/2.0/FAQ/
[cuest]: https://docs.nvidia.com/cuda/cuest/
[cuest-product]: https://developer.nvidia.com/cuda/cuda-x-libraries/cuest
[cuest-install]: https://docs.nvidia.com/cuda/cuest/installation.html
[cuest-license]: https://docs.nvidia.com/cuda/cuest/license.html
[cuest-samples]: https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuEST
[emulation]: https://docs.nvidia.com/cuda/cuest/gemm_emulation.html
[inception]: https://www.nvidia.com/en-us/startups/
[grants]: https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/
[barca]: https://doi.org/10.1021/acs.jctc.1c00720
[pokhilko]: https://doi.org/10.1021/acs.jctc.8b00321
[datta]: https://doi.org/10.1021/acs.jctc.3c00876
