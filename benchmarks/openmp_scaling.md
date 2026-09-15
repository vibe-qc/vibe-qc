# OpenMP / BLAS-thread scaling - per-method cores-per-job map

Measured with `scripts/bench_omp_scaling.py` + `scripts/bench_omp_scaling_sweep.sh`
on an Apple-Silicon dev box (18 cores = 6 performance + 12 efficiency, macOS,
Accelerate BLAS, `lapacke_enabled=False` so the dense eigensolve is serial
Eigen). `run_job` end-to-end wall time, file writes off, min-of-2 reps.

**Caveat:** the box was shared/contended during the run (load ~8). Absolute
times are noisy and 1-thread baselines are somewhat inflated (so per-thread
speedups are upper bounds); the **shape of each curve and the OMP-vs-BLAS axis
split are robust**, and those are what the cores-per-job map rests on. Re-run on
a quiet box / per-deployment to pin exact absolutes.

The decisive design choice: vibe-qc **pins BLAS to 1 thread by default**
(`python/vibeqc/__init__.py` sets `OPENBLAS_NUM_THREADS` /
`VECLIB_MAXIMUM_THREADS` / `MKL_NUM_THREADS` = 1) so the OpenMP regions don't
oversubscribe a threaded BLAS. Every Eigen / numpy GEMM therefore runs
single-threaded unless the user overrides those vars. So the sweep measures
**two axes separately**: `OMP_NUM_THREADS` (the `omp` rows) and the BLAS thread
count (the `blas` rows), plus both together (`both`).

## Raw speedups (vs 1 thread)

### OMP axis (BLAS pinned to 1) - the default config

| method / system | 1 | 2 | 4 | 8 | 16 | saturates | cap |
|---|---|---|---|---|---|---|---|
| RKS/PBE glycine def2-SVP (~115 bf) | 1.0 | 1.38 | 1.78 | **1.90** | 1.68 ↓ | **~8** | memory bandwidth + serial Eigen eigensolve; E-cores hurt at 16 |
| CCSD h2o-trimer cc-pVDZ (~75 bf) | 1.0 | 1.28 | 3.58 | 6.38 | **10.1** | **>16** | scales - wall time is in the *threaded scalar intermediates* |
| UCCSD CH₃ cc-pVDZ (small) | 1.0 | 1.74 | 2.17 | **2.42** | 2.34 | **~8** | OMP-scalar `so_residuals`; system too small to scale further |
| MP2 h2o-trimer cc-pVDZ | 1.0 | 1.66 | 1.64 | 3.52 | 3.94 | (noisy, small) | OMP transform loops; see BLAS axis |
| CCSD(T) h2o-trimer cc-pVDZ | 1.0 | - | 3.53 | 5.66 | **8.74** | **>16** | scales - scalar intermediates + the threaded (T) worklist (`9b1f0ce3`) |
| TD-DFT/PBE glycine def2-SVP (TDA) | 1.0 | - | 1.90 | 2.14 | **2.29** | **~8** | tracks the DFT SCF (it dominates here); the Python TDA build is the small part |
| RKS/PBE glycine **def2-TZVP** (~250 bf) | 1.0 | - | **2.31** | - | - | later than SVP | larger basis → more parallel work before the BW/serial-diag wall |

### BLAS axis (OMP pinned to 1) - does threaded BLAS/LAPACK help?

| method | blas=1 | blas=2 | blas=4 | blas=8 | verdict |
|---|---|---|---|---|---|
| CCSD h2o3 | 30.8 s | 31.7 | 29.2 | 28.6 | **flat (~1.08×)** - BLAS ladders are NOT the wall; threaded LAPACK won't help CCSD |
| MP2 h2o3 | 3.27 s | 1.73 | 1.73 | 1.72 | **1.9×** - the DF energy GEMM is single-threaded and *does* benefit |

### Both axes (OMP = BLAS = N)

| method | both=2 | both=4 | both=8 | vs OMP-only@8 | note |
|---|---|---|---|---|---|
| CCSD h2o3 | 15.4 (2.0×) | 8.1 (3.8×) | 4.80 (6.4×) | = omp8 (4.83) | BLAS adds nothing; no nesting penalty |
| MP2 h2o3 | 1.08 (3.0×) | 0.73 (4.5×) | **0.57 (5.8×)** | better than omp8 (0.93) | MP2 wants **both** axes |

## What the data corrected vs the static read

The static analysis (CCSD's rate-limiting *FLOPs* are the BLAS `tau·Wabefᵀ` /
`Wmnijᵀ·tau` ladders → "BLAS-bound, won't scale on OMP") was **wrong about
wall-clock**: the unvectorised scalar `collapse(2)` intermediate loops (the ring
Ws, F-intermediates, the fused T2 residual) are far slower *per flop* than the
BLAS ladders, so they **dominate wall time and they thread** - CCSD scales ~10×
to 16 threads while threaded BLAS does nothing (the `blas` row is flat). MP2 is
the opposite: its single big DF GEMM is the only thing that benefits from BLAS
threads. This is exactly why the task said *profile, don't guess*.

## Per-method scaling map (static read + measured, all methods)

`OMP` = scales with `OMP_NUM_THREADS`. `BLAS` = scales with the BLAS thread
count (`OPENBLAS_/VECLIB_/MKL_NUM_THREADS`). `serial` = single-threaded
(Python numpy / GIL / serial Eigen).

| Method | What threads (verified) | Scales on | Saturates | Why it stops |
|---|---|---|---|---|
| RHF / RKS | C++ Fock + XC-grid (OMP); serial Eigen eigensolve | OMP | ~8 (med), later for large basis | memory bandwidth, then the serial `SelfAdjointEigenSolver` for very large n |
| MP2 (DF) | OMP AO→MO transform + 1 single-threaded DF GEMM | OMP **and** BLAS | scales with both to 8 | both axes wanted (measured 5.8× at omp=blas=8) |
| RCCSD | scalar `collapse(2)` intermediates (OMP); BLAS ladders (1 thread) | **OMP** | >16 (size-dep) | scalar intermediates dominate wall time and thread; BLAS threads do nothing |
| CCSD(T) | + (T) worklist (OMP, `9b1f0ce3`) | OMP | >16 | as CCSD; (T) is well-threaded |
| UCCSD | scalar `so_residuals` (OMP), no BLAS | OMP | size-dependent | OMP-scalar; small systems saturate ~8 |
| CASSCF | C++ CI sigma build (OMP) | OMP | active-space-dependent | CI Davidson sigma is the threaded part |
| CASPT2 | Python numpy einsum / 4-RDM | **serial** | ~1-2 | single-threaded numpy BLAS + GIL; only the underlying SCF threads |
| TD-DFT | Python einsum AO→MO + numpy `eigh`; SCF underneath (OMP) | OMP (small) → serial (large) | ~8 (SCF-bound) | small systems track the DFT SCF (measured 2.3× to 16); only the O(n⁴) response build is Python-bound, and it dominates only for large systems |
| periodic RKS/UKS | C++ Fock/XC **per k** (OMP); `for k in k_points` **serial** | OMP inside each k | per-k saturates like molecular DFT | core count is NOT sized by n_k (unlike CRYSTAL) |
| periodic Γ-CCM | C++ aiccm kernels (OMP, 15 pragmas) + Python einsum | OMP (kernels) | medium | the C++ kernels thread; the Python einsum glue is single-threaded |

**Coverage:** *thread-scaling measured* this run - RHF/RKS (def2-SVP +
def2-TZVP), MP2, CCSD, CCSD(T), UCCSD, TD-DFT. The harness now also runs
CASSCF and CASPT2 (`cas_h2o`, active spaces (8,8) / (6,6)); a quick functional
check on the saturated dev box confirmed the cost split qualitatively
(CASSCF(8,8) ~3.8 s at 1 thread = the C++ CI sigma build; CASPT2(6,6) >> 120 s
= the single-threaded Python 4-RDM / IC contractions, validating the "serial"
prediction), but a clean 1-vs-N *scaling* curve for the CAS pair and for the
periodic methods needs a quiet box and is deferred to a dedicated `vq`/compute-reference
run. The cores numbers for CASSCF/CASPT2/periodic below remain predictions
backed by the static read + this cost check, not measured speedup curves.

## Cores-per-job recommendation (hand to the queue chat)

Default node target on the shared boxes; "pack" = run several jobs of this
class concurrently for throughput, "dedicate" = give it most of the node.

| Method class | cores | pack vs dedicate | extra env |
|---|---|---|---|
| RHF/RKS, ≤200 bf | **4-8** | **pack** 2-4 per node (saturates ~8, wastes >8) | - |
| RHF/RKS, ≥500 bf | **8-16** | dedicate / 1-2 per node | - |
| MP2 | **8** | dedicate | also set `OPENBLAS_/VECLIB_NUM_THREADS=cpus` (the one common method where threaded BLAS helps) |
| CCSD / CCSD(T) | **16+ (all)** | **dedicate** - one big job per node | - |
| UCCSD | **8-16** | dedicate | - |
| CASSCF | **4-8** | medium | - |
| CASPT2 | **1-4** | **pack** many per node | extra cores wasted (Python-bound; static read, not measured) |
| TD-DFT, small/med | **4-8** | medium | tracks the DFT SCF (measured) |
| TD-DFT, large | **1-4** | **pack** | the O(n⁴) Python response build dominates |
| periodic RKS/UKS | **4-8** (per-k work) | medium | size by per-k cost, **not** by n_k |
| periodic Γ-CCM | **8-16** | dedicate | - |

### Queue-side actions (queue chat owns these)

1. **Make `--cpus` real.** The daemon does not inject `OMP_NUM_THREADS` today,
   so every job spawns all-hardware-core threads regardless of `--cpus` and
   oversubscribes shared nodes (compute-small). Inject `OMP_NUM_THREADS=spec.cpus`. This
   is the single biggest win and a prerequisite for any of the above to hold.
2. **MP2 only:** also export `OPENBLAS_NUM_THREADS=spec.cpus` /
   `VECLIB_MAXIMUM_THREADS=spec.cpus` - MP2 is the one common method whose DF
   GEMM benefits from threaded BLAS (~1.9× alone, 5.8× combined). For every
   other method, leave BLAS pinned to 1 (it does nothing or risks nesting).
3. **Scheduling:** pack DFT(small)/CASPT2/TD-DFT (saturate early or are
   Python-bound); dedicate CCSD/CCSD(T)/MP2/CCM (scale to the node).

### Not done (assessed, with reasons)

* **Parallelize the serial k-point loop (CRYSTAL-style).** Only pays when
  `n_k > cores` *and* per-k work is small (small cells, dense mesh). For the
  flagship slab/surface workloads the per-k Fock/XC already saturates the cores,
  so threading inside-each-k is correct and cross-k parallelism adds nothing.
  Worth it only if a many-k-small-cell workload appears; deferred (roadmap).
* **Threaded LAPACK eigensolve.** `lapacke_enabled=False` on Apple (Accelerate);
  on Linux+OpenBLAS, `EIGEN_USE_LAPACKE` would thread `dsyev`, but the data shows
  DFT saturates from memory bandwidth *before* the eigensolve dominates at these
  sizes, and CCSD's eigensolve is negligible. Only matters for very large DFT
  (≥1000 bf); roadmap, not a quick win.
* **Code fixes for serial bottlenecks** were landed earlier this session as the
  parallelism half of the perf sweep (see `HANDOVER_PERF_OPT.md`): per-cell
  Schwarz, grid Becke partition (4.2×), AO-eval fork-join (1.9→3.2×), (T)
  triples worklist, periodic-XC grid-batch - each with a before/after number.
