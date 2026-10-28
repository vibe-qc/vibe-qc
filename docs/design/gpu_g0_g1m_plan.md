# GPU round 3: G0 and G1m implementation plan

Status: ACCEPTED and integrated into the
[roadmap](../roadmap.md#gpu-delivery-stages). Source inspected: `a3242f5c`.
The first implementation task is G0-A, CPU profiling and provenance; later
tasks follow the dependencies below. Execution is authorized by the relevant
implementation request, not by this document alone. No dependency, licence
change, vendor contact or library edit is authorized by this plan. None of
its profiling, GPU benchmarks or scientific acceptance runs has been performed.

Update: G0-A has landed. It adds opt-in SCF component timing, the
`examples/regression/gpu_profile.py` harness and an H2O-only pilot in
[`benchmarks/gpu_g0.md`](../../benchmarks/gpu_g0.md). The M and L
systems, noise floors and the G0-B/G0-C tasks remain open, so G0 is not
complete.

Labels: **[repo]** inspected repository source; **[source]** primary paper or
vendor documentation; **[repro]** standalone reproduction; **[proposal]** a
decision or numerical admission threshold; **[hypothesis]** an unmeasured
outcome. Unless explicitly labelled otherwise, tasks, schedules and thresholds
below are proposals. Thresholds are experimental admission limits, not
published guarantees or measured performance.

## Accepted amendments and corrections

| Round 3 item | Resolution and evidence |
|---|---|
| V1 / D5 | [repo], [repro] Record the unused streaming builders as wrong code, not a concern. Partial auxiliary blocks violate solve dimensions; full blocks retain the layout defect. See the deterministic reproduction in the [research review](gpu_research_review.md). This does not mean every special input produces a nonzero error. |
| V2 | [repo] The surviving raw-tensor consumers are B-only operations and the robust correction is inactive. Plan one resident B tensor, with construction peak measured separately. `T_flat()` is a public C++ accessor; no in-tree caller is not proof that removing it is source compatible. |
| V2, extra finding | [repo], [repro] `compute_B_metric` also has a transpose error at df.cpp lines 854-857. Compare the rewrite with `B B^T`, not with its erroneous old output. Preserve old-kernel parity for the active J/K builders. |
| V7 / D4 | [source] Cite the [Abinit v3 abstract](https://arxiv.org/abs/2604.11139v3) for LOBPCG versus Chebyshev filtering. Withdraw vendor-specific and Rayleigh-Ritz claims from that abstract-level evidence. |
| D1 | [proposal] G1 is CUDA/HIP vendor-BLAS plumbing and fixed-density native-FP64 DF-J/K replay. Metal and mixed-precision feasibility belong to G1m; no G1 FP32 DF-K SCF recovery pilot. |
| D2 | [proposal] Keep 128 MB at 2,000 real AOs as a transfer-volume example. The 5-10 ms estimate assumes 12.8-25.6 GB/s sustained bandwidth; it is not a measurement. Measure packing, synchronization, residency and host/device contention too. |
| D3 | [proposal] Accept the priority gate, with its measurement moved to the first G1 device replay after G0 CPU baselines. CPU-only G0 cannot show a device win. A failed gate moves a bounded grid pilot ahead of G2; final production ordering still follows evidence. |
| Hardware premise | [source] G1m requires an Apple GPU plus Metal/MPS development tools. Only its fixture preparation and CPU references are GPU-free. A week is a timebox, not a promised implementation or a verdict about untested M5/M6 devices. |

[repo] `RHFOptions` has `density_fit=false`, `SCFMode::AUTO` and threshold
140 ([rhf.hpp](../../cpp/include/vibeqc/rhf.hpp), lines 299, 328-329).
AUTO selects conventional in-core or direct, rather than always direct;
the shared resolver is declared in
[jk_builder.hpp](../../cpp/include/vibeqc/jk_builder.hpp), lines 44-72.
Measure the actual resolved default as well as explicitly forced direct.
The historical [speed report](../../benchmarks/orca_vs_vibeqc_speed.md)
is motivation to remeasure, not a baseline for current source or hardware.

[source] The [JuliaChem publisher excerpts](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.5c02025)
confirm an AMD proof of concept without grouped batching and tunable GPU
crossover settings. They also report AMD timings, so those measurements do
exist; they are not a prediction for our HIP backend. Do not transplant a
500/800-AO threshold into vibe-qc. The cited CPU-to-GPU factor transfers are
excluded from its reported GPU Fock times; our full-cost comparison includes them.

## G0: CPU performance and design, 2-3 engineer-weeks

### Fixed workload manifest

Use the exact coordinates in `GEOMETRIES` in
[cases.py](../../examples/regression/parity_matrix_orca/cases.py) and its
single Angstrom-to-Bohr conversion. Do not substitute the different H2O
geometry in another regression catalog. Freeze the geometry and resolved
basis content hashes in the manifest.

| Size | Regression key | Orbital basis | State |
|---|---|---|---|
| S | `H2O` | `def2-svp` | neutral singlet, restricted |
| M | `naphthalene` | `def2-svp` | neutral singlet, restricted |
| L | `n-hexadecane` | `def2-svp` | neutral singlet, restricted |

These are three existing regression systems, not a claim to cover 2,000 AOs.
Record actual `n_ao`, `n_aux`, occupied count and grid sizes at runtime.
Add H2O/def2-TZVP as an angular-momentum diagnostic if the primary matrix
completes; it is not a substitute for a missing large-system result.

For each size run RHF, RKS/PBE and RKS/PBE0 on:

1. **Explicit direct:** `density_fit=false`, `cosx=false`, `scf_mode=DIRECT`.
2. **DF:** `density_fit=true`, `cosx=false`, explicit
   `def2-universal-jkfit` auxiliary basis for this controlled comparison.
3. **RIJCOSX:** `density_fit=true`, `cosx=true`, explicit
   `def2-universal-jfit`; record resolved COSX variant and grid stages.
4. **Actual default builder:** `density_fit=false`, `cosx=false`,
   `scf_mode=AUTO`, threshold 140. Deduplicate a run only when all resolved
   settings are identical to the direct case; keep both manifest labels.

PBE has zero exact exchange: K is **not applicable**, not a measured zero-cost
K build. The RIJCOSX-labelled PBE control may resolve to RI-J with no COSX-K;
record that fact rather than invent an exchange timing.

Pin SAD, charge/spin, orbital ordering, basis representation, initial density,
grids, screening, convergence norms, maximum iterations, incremental settings,
stability policy and symmetry identically where applicable. Use energy
convergence `1e-9 Ha`, initial commutator target `1e-6` in the driver's stated
normalization and at most 200 iterations as the initial profiling recipe;
record actual resolved values and any route refusal. Pin the inspected XC
grid defaults explicitly: Generic profile, 75 radial points, product angular
grid with 17 theta and 36 phi points, Becke partition with `becke_k=3`, no
angular pruning. Use COSX level 2 with its resolved stages. Serialize remaining
grid options and their resolved point counts too. Tighten the final grid/screen
for a separate accuracy diagnostic when necessary, never only for one timed
competitor. Route approximation errors and arithmetic errors are separate.

[repo] The current
[vibeqc_timing_job.py](../../examples/regression/parity_matrix_orca/vibeqc_timing_job.py)
clamps direct-route energy convergence to `1e-7` in `_apply_fock_path`.
Do not use it unchanged for matched-accuracy conclusions. Build an isolated
G0 runner reusing geometry/queue plumbing, with explicit serialized settings
and refusals. If the current incremental direct path cannot meet the common
recipe, retain that failure; add a matched comparison with incremental builds
disabled and report the configuration difference from the default.

Use OMP=1 and OMP=N, where N is the granted physical-core allocation, and
record requested and observed teams. Start with BLAS threads=1 to avoid nested
oversubscription; separately measure the supported BLAS-parallel setting for
the dominant DF operation. Do not handicap the CPU in the device comparison:
use the fastest qualified CPU configuration within the same allocation.

One untimed warm-up and three independent timed process runs per qualified
cell/thread setting are the initial protocol. Separate cold process/setup cost
from warm kernel replay. Randomize run order and exclude overlapping jobs.
Use a proposed 60-minute per-run budget initially; timeout is incomplete
evidence, not a performance number. Rebudget explicitly after pilot timings.
Do not launch the whole matrix unbounded or turn it into routine CI.

### Instrumentation and evidence

Record per iteration and in aggregate: integral/DF construction, J, K or
**fused JK** time, AO/grid construction, density/XC evaluation and Vxc
assembly, diagonalization, DIIS/other accelerator, density construction,
recovery/rebuild work, stability, output and unclassified overhead. Record
SCF wall time, iteration counts, convergence status, energy, residual norms,
grid-stage changes and actual screening thresholds.

[repo] Direct `build_g_rhf` can be fused; do not split its wall time into
invented J/K fractions or replace it with two builds for the production
baseline. Separate J and K replay diagnostics may supplement the fused timer.
The existing SCF trace `wall_s` is a whole-iteration clock, not component
instrumentation. Exclusive phase totals must reconcile with the enclosing
wall time; nested timings are explicitly marked inclusive.

Memory evidence has three distinct fields: tracked live tensor bytes after
construction, tracked construction peak, and process RSS high-water mark.
RSS includes allocator/BLAS effects and is not proof of live tensor storage.
Sample construction boundaries and account for transient T, T-flat, B-flat,
per-P matrices, metric, scratch and packing. Run before/after measurements
in fresh processes with the same input, compiler, BLAS and thread settings.

Raw records include `vibeqc.__file__`, native module `__file__`, build-info
receipt, source commit, binary digest, dependency/compiler versions, BLAS
backend and thread controls, process allocation, command/deck and provenance.
Require source/runtime identity match before accepting a numerical result.
Raw logs, binary fixtures and absolute paths stay under an explicitly selected
external evidence root through existing output-path helpers, never anywhere
inside the checkout, including ignored directories.

Proposed repository summaries: `benchmarks/gpu_g0.md` and
`benchmarks/gpu_g0_summary.json`, with no placeholder measured values.
One row per case/method/resolved route/thread setting contains public case
ID, input hashes, public source/binary IDs, compiler/BLAS versions, generic
CPU/GPU model, thread counts, dimensions, median/min/max times, phase totals,
memory fields, iterations, numerical errors and PASS/FAIL/INCOMPLETE.
Exclude host/account/job identifiers, internal URLs and machine paths.
Runtime provenance is retained externally; the summary says whether the
identity check passed. User-facing calculation output goes through
`vibeqc.output`; the summary is derived evidence, not a new calculation engine.

### Bounded DF tasks and contracts

**G0-B1: retire defective streaming.** Prefer deleting the unused declarations
and definitions in `df.hpp`/`df.cpp`. Record the public C++ compatibility
decision before deletion; absence of Python bindings is not licence to break
an externally supported API. If compatibility requires retaining the entry
points, explicitly refuse the unsupported streaming route pending a tested
replacement, rather than silently returning a dense-memory algorithm.
There are no production callers to migrate in the inspected tree.

A replacement, if chosen, blocks pair columns `(mu,nu)` with the full auxiliary
dimension in every triangular solve. Auxiliary-row blocking would require
proper block forward substitution. Correct construction blocking alone does
not establish bounded-memory exchange; design the K contraction's dependent
tiles/recomputation separately. Do not leave T resident solely to preserve
the defective methods. Cover partial/full/oversized blocks and invalid sizes,
nontrivial triangular factors, symmetric random densities, unequal dimensions
and strided/layout sentinels. An explicit refusal path needs refusal tests.

**G0-B2: remove duplicate residency.** Express `compute_B_metric` as
`B_flat B_flat^T` and `contract_B_with(M)` as `B_flat vec_pair(M)`.
Keep one owning B allocation; a B-flat mathematical view need not be a second
physical tensor. Start with existing `B_per_P_` storage and bounded packed
panels/direct Frobenius contractions for these inactive helpers, preserving
the active K layout. Do not allocate a permanent flat copy. Release the
construction raw tensor after the transform and necessary unpacking.

The metric test oracle is explicit per-P Frobenius products, including its
symmetry/positive-semidefinite property. The old metric has a reproduced
transpose defect. The active `build_J` and `build_K_density` must retain
fixed-density matrix and energy parity against the old kernel. Preserve the
OMP-independent accumulation and aligned triangular-solve partition where
possible; any changed rounding needs the measured tolerance protocol below.

[repo] `T_flat()` returns a const reference in the public header. Decide its
deprecation/removal contract before dropping its storage; lazily retaining
a reconstructed raw tensor defeats the one-tensor residency claim. In-tree
DF derivative kernels use separate integral paths, but rerun their focused
checks after storage changes. Do not activate robust COSX as part of this work.

**G0-B3: construction peak.** After B2, measure whether in-place solves and
pack/unpack lifetime changes actually lower peak. The current half-transform
accounts for four tensor extents including a scratch allowance; that is an
accounting policy, not a measured allocation trace. Avoid bundling a risky
in-place rewrite into B2. One resident tensor is a design goal; neither a
50% process-memory reduction nor a particular peak reduction is promised.

**G0-C: conventions.** Add `docs/design/gpu_operation_contracts.md` before G1:

- Declare physical strides separately from logical indices. Current raw
  T-flat is column-major `(n_aux,n_ao^2)`, pair index `mu*n_ao+nu`;
  each B-per-P matrix is column-major `(n_ao,n_ao)`. A row of T-flat is
  strided. Require pack/unpack round trips with unequal dimensions and an
  asymmetric index sentinel so symmetry cannot hide a transpose.
- Restricted `D=2 C_occ C_occ^T`, `F=H+J-a_x K/2+Vxc`,
  `E_J=Tr(D J)/2`, `E_K=-a_x Tr(D K)/4`. PBE0 uses `a_x=1/4`;
  RHF uses 1 and PBE 0. Unrestricted contracts use `D_alpha+D_beta` for
  J and each unit-occupation spin density for K, with exchange energy
  `-a_x sum_sigma Tr(D_sigma K_sigma)/2`. Do not double occupations twice.
- `X=C sqrt(f)` is admissible only for the actual density being built;
  driver orbitals do not necessarily factor a damped or extrapolated input.
  Keep the density path for indefinite inputs. Fractional/MOM occupations
  retain their requested policy. Factorized K stays an independent experiment.
- Initial G1 is real molecular arithmetic. Complex data later requires
  conjugate transposes, Hermitian norms and separate k-point conventions;
  real-only qualification is not periodic support.

At fixed density compare J, K and Vxc separately using max-absolute error,
Frobenius relative error with a documented zero-denominator rule, symmetry
defect, and the actual energy contractions above. Hold basis/auxiliary basis,
grid, screens and density identical. Include SAD, mid-SCF, converged and
small signed perturbation inputs. Measure CPU-repeat and OMP=1 versus N
noise for every observable before comparing a new implementation.

Set each later strict tolerance from a recorded CPU noise floor and a
scientific error budget, for example require both error <= ten times the
larger of measured noise and a justified roundoff estimate, and error <= the
predeclared scientific budget. A large noise floor does not automatically
relax the budget: it makes the test unresolved. No universal `1e-10 Ha`
criterion follows from FP64. G0 must publish actual limits and their units
before G1; this plan does not manufacture the missing noise measurements.

### Files, tests and exits

All new paths below are proposed, not files implemented by this change.
New tests must be entered in `CURATED` in
`scripts/test_gate/update_suite_manifest.py` and generated with
`python3 scripts/test_gate/update_suite_manifest.py`; never hand-edit JSON.
Existing tests keep their classification. Rebuild only affected native code
using the contributor setup, then verify the loaded build before native tests.

| Task | Files/interfaces | Test and classification | Focused check after implementation | Exit evidence |
|---|---|---|---|---|
| G0-A manifest, runner, privacy | New `examples/regression/gpu_profile.py`; reuse `parity_matrix_orca/cases.py`, queue/output-path helpers; route user output through `python/vibeqc/output/` | New `tests/test_gpu_profile.py`: T2, under-review, owner `performance`; synthetic manifest/provenance/refusal checks | `.venv/bin/python -m pytest tests/test_gpu_profile.py -q` | Exact manifest; matched settings; no hidden tolerance clamp; redaction and provenance tested |
| G0-A component timers | `cpp/include/vibeqc/rhf.hpp`, `cpp/src/rhf.cpp`, `cpp/src/rks.cpp`, `cpp/src/jk_builder.cpp`, bindings and output mapping; opt-in aggregate diagnostic fields | New `tests/test_scf_component_timing.py`: T2, under-review, owner `molecular/DFT` | `.venv/bin/python -m pytest tests/test_scf_component_timing.py tests/test_scf_stability_phase_timing.py -q` | Positive clocks for executed phases; fused work labelled; no double counting; overhead measured against timing disabled |
| G0-B1 streaming | `cpp/include/vibeqc/df.hpp`, `cpp/src/df.cpp`; internal C++ fixture access only if needed | New `tests/test_df_storage_contract.py`: T2, under-review, owner `basis/integrals`; refusal or numerical layout/block tests according to chosen API path | `.venv/bin/python -m pytest tests/test_df_storage_contract.py -q` | Defective path unavailable or independently correct; full/partial block coverage; compatibility decision recorded |
| G0-B2 helpers/residency | Same DF files; narrow binding of diagnostic helpers if necessary in `cpp/src/bindings.cpp` | Extend `tests/test_df_core.py` and new storage test; existing core T2/verified | `.venv/bin/python -m pytest tests/test_df_core.py tests/test_df_storage_contract.py tests/test_df_rhf.py tests/test_df_gradient_kernels.py -q` | Direct-B metric/alpha oracle; old active J/K parity; derivatives checked; one tensor demonstrated |
| G0-B3 peak | Half-transform/constructor in `df.cpp`; byte counters and external profiler | Extend storage test for accounting; performance measured outside pytest | `.venv/bin/python -m pytest tests/test_df_storage_contract.py -q` | Before/after live, construction and RSS measures; no threshold evasion or uncounted scratch |
| G0-C contracts/noise | New conventions document; profile fixture exporter; existing DF/grid interfaces | Extend new storage/profile tests and existing `tests/test_cosx_batched.py` (T2/verified) | `.venv/bin/python -m pytest tests/test_gpu_profile.py tests/test_df_storage_contract.py tests/test_cosx_batched.py -q` | Spin/layout formula checks, measured noise table, fixed-density protocol and unresolved cases |

Use a complete `vq` calculation with matched ORCA reference for acceptance of
scientific DF changes, reusing qualifying cases when possible. Unit tests and
standalone algebra reproductions do not replace that verdict. G0 profiling is
also submitted through `vq`; its commands must target the future implemented
runner and configured external output root. No speculative working CLI is
claimed here. Ordinary development skips CI; release qualification remains
under the existing policy.

Week 1: manifest, identity, instrumentation and pilot measurements. Week 2:
bounded DF repair/storage work and CPU comparisons. Week 3 contingency:
remaining runs, reference acceptance, noise/contract report and prioritization.
Do not label all G0 complete if the large case or reference evidence is pending.

## G1m: Apple feasibility, one engineer-week alongside G0

Requires a real Apple GPU, macOS, supported Metal compiler/runtime, MPS and
Accelerate. Record the actual GPU family, OS and tool versions. Do not infer
M5/M6 hardware from the request or extrapolate one model's result to another.
No CUDA/HIP installation is needed. CPU fixture preparation can start first.
The [MPS matrix API](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
is a backend for an experiment, not a scientific accuracy guarantee.

Use real G0 basis/grid/density fixtures, with no host paths in serialized
metadata. Proposed isolated harness: `examples/benchmarks/gpu/apple/`, with
`main.mm`, `collocation.metal`, `cosx_batch.metal` and build instructions.
Nothing is linked into the production runtime in this stage. Freeze CPU
references before tuning, retain tail/tight-core/diffuse and cancellation
batches as held-out checks (add H2O/def2-SVPD with a fixed SAD density for
the diffuse diagnostic), and test awkward batch sizes as well as 256,
1,024 and 4,096 points where present.

### Three experiments

| Experiment | Work and CPU comparator | Proposed numerical admission |
|---|---|---|
| A: collocation | Metal FP32 values and all three spatial derivatives on identical points against `evaluate_ao_with_gradient`; same AO normalization, contractions, spherical transform, ordering and screens | For each of values and derivatives, normalized max error <= `2e-5` and normalized Frobenius error <= `5e-6`; include a density/gradient contraction check <= `2e-5` using actual D |
| B: DF-shaped GEMM | MPS FP32 versus Accelerate DGEMM using identical real B/D or B/X inputs; `(n,n)*(n,n)` and `(n,n)*(n,n_occ)` for actual G0 shapes, repeated over auxiliary panels; do not substitute a large square-only benchmark | Normalized Frobenius and max error <= `1e-5` per product; then assemble fixed-density K and apply the full-K/energy budget below before calling this a DF candidate |
| C: complete COSX-K batch | FP32/hybrid batch versus CPU FP64 `compute_cosx_k` on identical real grid batches; include active-pair selection, integral evaluation and all contractions | Sum of absolute exchange-energy errors over all batches of each complete grid <= `1e-5 Ha` per molecule and assembled K normalized Frobenius/max error <= `1e-5`; no nonfinite outputs |

For these pilot ratios use `||delta A||_max / max(1,||A_ref||_max)` and
`||delta A||_F / max(1,||A_ref||_F)` in the declared atomic units. Also retain
unscaled absolute errors and near-zero-tail diagnostics so the floor is not
mistaken for a relative guarantee. Pilot limits intentionally differ from
the later strict CPU-derived limits. Failure sends work to qualified CPU
FP64 or refinement; it does not loosen the scientific production target.

For B, apply the same `1e-5 Ha` exchange-energy and `1e-5` K pilot budgets
after complete auxiliary-panel reduction. Neither GEMM success nor A's AO
error bound proves SCF accuracy. For A, recompute a CPU FP64 density/XC
diagnostic from the candidate AO arrays; reject if the fixed-density XC
energy changes by more than `1e-5 Ha` on a full-grid test. These are proposed
screening thresholds to select work, not final HF/DFT acceptance.

### What counts as a complete exchange batch

[repo] `compute_cosx_k` in [cosx.cpp](../../cpp/src/cosx.cpp) contracts
`D chi`, screened one-electron attraction integrals at grid centers, weighted
exchange contributions, then symmetrizes and applies the selected overlap
fit (lines 900-935). `grid_batch.hpp` supplies primary-AO caches; AO-only
offload is not complete exchange.

[source] [Laqua 2021](https://doi.org/10.1063/5.0045084), sections II-IV,
separates contribution-based omission, FP32 and FP64 work and uses a fixed
mixed-precision anchor for incremental updates. Adopt that structure only
after mapping its quantities to our COSX variant; the algorithms are not
interchangeable. The mislabelled local 2021 SI remains excluded.

A complete measured batch covers input/pair packing; AO values; D-chi
contraction; contribution bounds and list construction; three-center
one-electron integral evaluation for active pairs; FP32 and routed CPU-FP64
contractions; weighted reduction/scatter into K; synchronization and merging.
Measure final global symmetrization/overlap correction and its setup separately
and include their amortized/full costs. Do not apply a different Q to each
batch: use the same global correction as the reference. Include reference
full-grid summation to detect accumulation errors hidden by isolated batches.

If an integral kernel is only a CPU FP64 producer followed by FP32 contraction,
label that experiment a hybrid contraction slice and time the producer. It
does not establish a complete GPU exchange engine. If incremental work is
tested, first make the qualified anchor, use `D-D_anchor`, tighten screening
as required, and include anchor/rebuild cost. Recursive FP32 accumulation is
not a substitute. If this cannot be built or assessed in a week, report
INCOMPLETE and revise the estimate; no-go is not proof of impossibility.

### Timing, tests and first-route decision

Use command completion/event times for device kernels and wall time for the
complete operation. Include conversion, allocation, packing, FP64 fallback,
merging and synchronization; separate startup from reusable-buffer execution.
Shared memory may eliminate copies but not memory traffic or synchronization.
Warm up; use at least five independent timing groups with enough repetitions
to amortize timer resolution, and report median/range and thermal conditions.

Proposed performance admission: on both M and L, complete candidate operation
median <= `0.8 * CPU median`, with nonoverlapping group ranges and all numeric
checks passing. This is a chosen margin, not a predicted speedup. The S case
establishes dispatch overhead/crossover and may select CPU. A kernel-only win
without a complete-operation win does not pass. Missing hardware, compiler,
complete-batch implementation or trustworthy timing is INCOMPLETE.

New `tests/test_gpu_apple_contract.py`: T3, experimental, owner `GPU`, added
through `CURATED`. Validate fixtures and numerical comparisons in ordinary
Python; hardware cases explicitly skip with a reason off Metal hardware.
Run `.venv/bin/python -m pytest tests/test_gpu_apple_contract.py -q` on the
Apple target, and existing `tests/test_ao_eval.py` / `tests/test_cosx_batched.py`
for unchanged CPU reference contracts. Reuse existing CPU bindings; any
new batch-capture binding in `cpp/src/bindings.cpp` is diagnostic only and
requires native rebuild/provenance. Skips cannot satisfy hardware acceptance.
Record the benchmark in proposed `benchmarks/gpu_g1m.md`, raw data externally.

Choose the next Apple pilot using measured removable G0 time and full measured
candidate cost, not isolated FLOP/s:

1. If A passes and improves the whole measured grid slice, prioritize AO/grid
   integration, keeping XC and required FP64 work on CPU initially.
2. If C passes as a complete batch and gives greater measured savings after
   fallback/anchor costs, prioritize seminumerical exchange instead.
3. B passing admits further fixed-density DF work. It does not admit FP32
   SCF: strict CPU recovery and complete SCF must be measured in a separately
   scoped follow-up before DF becomes the first production Apple route.
4. If none pass, retain CPU execution and record FAIL or INCOMPLETE per
   candidate. No emulated-FP64 Apple implementation is assumed.

Days 1-2: freeze fixtures/build harness and A/B. Days 3-4: C or an explicitly
limited slice. Day 5: held-out errors, full-cost accounting and decision.
Native integral recurrence work exceeding the timebox gets a separate estimate.

## G1 prerequisites, device gate and roadmap

G0 must settle operation shapes, strides/transposes, ownership/lifetimes,
spin/occupation factors, symmetry, error-reporting semantics, allowed aliasing,
workspace requirements, asynchronous dependencies and fallback behavior.
Publish CPU noise and accuracy limits, immutable replay densities, admitted
route settings and measured CPU phase costs. Each backend reports actual
native/emulated/mixed precision per operation; initial G1 uses native FP64.

Memory admission must charge resident factors, metric, all D/J/K outputs,
double buffering when used, packing/conversion, vendor scratch and concurrent
plans. Use the smaller of user budget and currently available device memory,
with an explicit recorded reserve; never hard-code total VRAM as available.
Probe workspace before execution, derive block size from the complete budget,
and handle allocation failure by a smaller qualified block or explicit CPU
fallback. No silent method/basis change or unmanaged host spill. Shared Apple
memory has a joint CPU/GPU budget; it is not extra independent VRAM.

G1 needs one accessible NVIDIA device supported by its installed CUDA/BLAS
stack and one AMD device supported by its installed HIP/ROCm/BLAS stack,
with a known CPU baseline and enough memory for at least a blocked M/L replay.
Record the exact board, driver, toolkit, native FP64 capability and memory.
Minimum development access is not product qualification: workstation and
gaming devices from each vendor require their own later capability/performance
rows. No card-class support or vendor funding is promised. Apple is separate.

G0 ranks removable work and can reject an unattractive DF hypothesis, but
cannot measure a GPU win. In G1 replay the entire sequence of G0 densities
using resident device factors, not just the last cheap iteration. Define a
**projected** DF run cost as measured CPU DF setup plus first upload plus
measured device replay/transfer/merge plus measured remaining CPU SCF work.
Include initialization and no unmeasured concurrency credit. Compare with
actual default-route CPU totals at matched numerical acceptance, keeping
DF approximation error separate from backend error.

Prioritize G2 if this projection, with timing spread included, is lower on
both M and L for RHF and PBE0 on at least one vendor, and the fixed-density
numerical gate passes. PBE is a separate J/XC decision, not an exchange win.
The S result defines CPU crossover. This is a prioritization test only;
after minimal integration, G2 must confirm a complete measured SCF win with
iteration counts and final-state checks. If the projection fails or remains
inconclusive, prioritize the measured dominant grid/COSX pilot ahead of G2
and revise the implementation estimate. Do not assert that grid work will win
merely because DF failed. The broader periodic/correlation roadmap and optional
cuEST study retain their independent gates in the main concept.

## Items closed from repository evidence, and what remains open

| Closed [repo] question | Evidence |
|---|---|
| Where to integrate, and which K algorithm currently runs? | [jk_builder.cpp](../../cpp/src/jk_builder.cpp), DFJKBuilder lines 332-375, calls density-form K; [df.cpp](../../cpp/src/df.cpp) lines 747-759 already implement occupied-factor K. |
| Must raw T remain for active derivatives or robust COSX? | df.cpp lines 761-870 contain all post-construction consumers apart from the header accessor; derivative kernels start below them. [cosx_robust.cpp](../../cpp/src/cosx_robust.cpp) returns K unchanged. No streaming call/binding was found by repository-wide symbol search. |
| Do upstream orbitals guarantee a factor for the builder input? | [rhf.cpp](../../cpp/src/rhf.cpp), lines 494-500, constructs a mixed `D_used` before the JK call; lines 753 onward form new orbitals. Their availability does not make them a factor of every input density. |
| What is the CPU default? | [rhf.hpp](../../cpp/include/vibeqc/rhf.hpp), lines 299 and 328-329, plus the shared SCF-mode resolver: DF off, AUTO threshold 140. |
| Can existing interfaces supply Apple reference batches? | [ao_eval.hpp](../../cpp/include/vibeqc/ao_eval.hpp) defines values and gradients; [grid_batch.hpp](../../cpp/include/vibeqc/grid_batch.hpp) defines cached batches; [test_cosx_batched.py](../../tests/test_cosx_batched.py) already checks batched/unbatched K. |
| Are all requested component timings already available? | No: `SCFIteration::wall_s` in rhf.hpp lines 478-485 covers the whole iteration. The G0 plan adds explicit component instrumentation without fabricating a split of fused JK. |
| Must new benchmark geometries or a new test policy be invented? | No: `parity_matrix_orca/cases.py` supplies the three systems/basis family; `update_suite_manifest.py` supplies CURATED and manifest generation. |

Closed [repro]: the two streaming defect regimes and the metric transpose
defect, using standalone Eigen algebra. This is not full QC validation.

Still open: measured bottlenecks, CPU noise, one-tensor and peak-memory gains,
Apple route feasibility, M5/M6 qualification, GPU crossover, complete SCF
recovery cost, public C++ accessor compatibility decision, staffing and exact
schedule. Repository evidence cannot settle vendor licensing permissions,
outside-contribution relicensing rights, funding or the commercial boundary.
The open-source acceleration recommendation remains a proposal. Library SI
repair remains a separately authorized library intake task.
