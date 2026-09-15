# MPI Parallelization Strategy

This note records the code review and implementation strategy for
production MPI parallelism in vibe-qc. It distinguishes the current
OpenMP and experimental MPI surface from the distributed-memory model we
should build next.

Status as of 2026-08-24 (updated; original audit below is from 2026-06-30):

- OpenMP is real and broad across the C++ core.
- `mpi4py` is already an optional extra, with collective helpers in
  `python/vibeqc/mpi.py`.
- Production SCF and post-SCF drivers do not yet use MPI as their complete
  execution model. Multi-k GDF has two distributed seams, and restricted 3D
  χ-CCM four-center RHF now farms its direct short-range output blocks; the
  surrounding SCF work remains replicated.
- **MPI-P1 has landed**: `vibeqc.mpi.KPointPartition` (2026-07-28)
  provides the split/reassembly primitive (`local_indices` +
  `allgather_ordered`), with real `mpirun -np {2,3}` tests
  (`tests/test_mpi_kpoint_multirank.py`).
- **MPI-P3 has now landed in full for multi-k GDF**, in two slices:
  first, multi-k GDF's cderi (density-fitting 3-center integral) build
  farms momentum-transfer q-groups across ranks via `KPointPartition`,
  one partition/gather per SCF *setup* (`python/vibeqc/periodic_k_gdf.py`,
  `run_krhf_periodic_gdf` / `run_kuhf_periodic_gdf`, and
  `run_krohf_periodic_gdf` in `periodic_rohf_gdf.py`, which calls the
  same builder). Second -- landed 2026-08-05, `HANDOVER_MPI.md` item
  4c -- the **per-iteration** `n_k²` exchange contraction is farmed too,
  over the **bra** k-index: each rank computes complete `K(k_i)`
  matrices for its own share (exact, not a partial-sum approximation)
  and `KPointPartition.allgather_ordered` reassembles them in global k
  order. Every rank already holds the full replicated cderi cache from
  the setup-time farming, which is exactly what a bra-partitioned rank
  needs. Verified under real `mpirun` at 1/2/3 ranks to <1e-12 relative
  vs. the serial reference (`test_multik_gdf_exchange_farming_matches_serial`).
  Measured on one node (H2/6-31g**, (2,2,2), 8 cores fixed): 1×8
  threads 47.69 s, 2×4 threads 31.86 s (1.50x), 4×2 threads 28.36 s
  (1.68x) -- unlike the setup-time farming, this MPI split pays even on
  a single node here, because the per-call GEMMs are individually small
  enough that fewer threads per rank recovers parallel efficiency;
  multi-node scaling remains unmeasured (no cluster in the dev
  environment).
- **D114 adds a construction-neutral complete-output scheduler**:
  `LatticeOutputPartition` partitions ordered real-space cell keys with exact
  ownership and fingerprinted allgather validation. Restricted 3D χ-CCM RHF
  uses it only for the padded direct-ERI short-range J/K output list. Every
  task retains the full internal translation sum; reciprocal long-range
  Hartree, long-range exchange, the q=0 seam, and the rest of SCF remain
  replicated. This is direct-output scheduling, not Γ-CCM WSC farming or χ
  finite-group residue farming.
- Beyond the above, the other code path that computes a matrix
  with MPI is the experimental GPW Hartree-J z-slab overlay,
  `GpwJBuilder(mpi_aware=True)`, in
  `python/vibeqc/periodic_gapw_j.py`.

## Current Code Audit

### OpenMP Substrate

The shared-memory model is centered on
`cpp/include/vibeqc/thread_pool.hpp`, which wraps
`omp_get_max_threads`, `omp_get_thread_num`, `omp_set_num_threads`, and
the per-thread libint engine pool. The Python binding exposes
`get_num_threads()` and `set_num_threads()` in
`cpp/src/bindings.cpp`, and `python/vibeqc/runner.py` reports the active
thread count in the output.

The C++ core uses OpenMP across the main expensive kernels:

- Molecular one- and two-electron integrals:
  `cpp/src/integrals.cpp`, `cpp/src/fock.cpp`, `cpp/src/df.cpp`.
- Molecular SCF and DFT grid work:
  `cpp/src/rks.cpp`, `cpp/src/uks.cpp`.
- Gradients and Hessian-related integral kernels:
  `cpp/src/gradient.cpp`, `cpp/src/hessian_integrals.cpp`.
- MP2, UMP2, CCSD, UCCSD, and triples:
  `cpp/src/mp2.cpp`, `cpp/src/ump2.cpp`, `cpp/src/ccsd.cpp`,
  `cpp/src/uccsd.cpp`, `cpp/src/ccsd_triples.cpp`.
- Periodic kernels:
  `cpp/src/periodic_fock.cpp`, `cpp/src/periodic_xc.cpp`,
  `cpp/src/ewald.cpp`, `cpp/src/periodic_bloch.cpp`.
- Multireference helpers:
  `cpp/src/casci.cpp`, `cpp/src/selected_ci.cpp`,
  `cpp/src/nevpt2_ops.cpp`.

Most pybind11 calls into heavy C++ kernels use
`py::call_guard<py::gil_scoped_release>()`, so Python can orchestrate
work while the C++ side uses OpenMP. `cpp/src/periodic_bloch.cpp`
already parallelizes the multi-k Bloch sum over the k-loop.

The benchmark note `benchmarks/openmp_scaling.md` is the current
performance map. It shows the important practical rule: vibe-qc pins
BLAS threads to 1 by default and lets OpenMP own the outer parallelism.
MP2 is the main exception where raising BLAS threads also helps.

### Existing MPI Substrate

`pyproject.toml` defines an optional `mpi` extra with `mpi4py>=3.1`.
`python/vibeqc/mpi.py` provides:

- `MPIWorld` and active-world detection.
- Basic collectives: allreduce, reduce, bcast, gather, allgather,
  barrier.
- Contiguous shell-pair partitioning via `distribute_shell_pairs`.
- Contiguous task partitioning via `distribute_tasks`.
- Exact ordered real-space output-task partitioning via
  `LatticeOutputPartition`.
- Grid z-slab partitioning via `grid_slab_partition`.
- GPW helper routines for density collocation and potential projection.

Installing `[mpi]` enables capability but does not activate MPI. In an
ordinary process, `vibeqc.mpi` does not import `mpi4py.MPI`, so even calls to
the serial identity helpers leave MPI uninitialized. Complete launcher-rank
evidence from Open MPI, PMI/Intel/MPICH, MVAPICH, an actual Slurm task step, or
PMIx admits native loading and `COMM_WORLD` binding; a wrapper may instead
declare the generic required contract and its exact expected size.
Communicator rank and size remain authoritative and must agree with the
launcher. Detected launcher
failures never fall back to multiple independent serial worlds.

The implemented call sites are narrow:

- `python/vibeqc/periodic_gapw_j.py` uses the grid z-slab helpers when
  `GpwJBuilder(..., mpi_aware=True)` is selected. It collocates density
  on rank-local z-slabs, assembles a replicated full grid, runs the FFT
  Poisson solve on every rank, and allreduces the projected AO Hartree-J
  matrix. This is correct by construction for the fixed post-v0.12
  simulated-rank tests, but still marked experimental until a real
  multi-rank `mpirun` validation is in the gate.
- `python/vibeqc/mpi_tddft.py` can farm TDDFT state-property analysis
  across ranks, but the normal `run_job(..., tddft=True)` path does not
  appear to route through it today.
- `distribute_shell_pairs` is exported and tested but is not wired into
  the production molecular SCF/Fock builders.
- Restricted 3D `aiccm2026dev-b` RHF with `four_center` uses
  `LatticeOutputPartition` around the BIPOLE domains kernel. It partitions
  only complete direct short-range output blocks and allgathers them before
  incremental-Fock state is updated. RKS/UHF/UKS, fitted and post-HF χ routes,
  reciprocal layers, and rank-aware output remain outside this slice.

The repo therefore has an MPI substrate, exact distributed seams in multi-k
GDF and restricted χ four-center RHF, and one experimental periodic grid
overlay. It does not yet have a completely distributed HF, DFT, MP2, TDDFT,
or periodic multi-k SCF driver.

## Design Principles

1. Keep MPI optional. A source checkout without `mpi4py`, or an unlaunched
   process with it installed, must retain the exact serial identity behavior.
2. Use hybrid MPI x OpenMP. The default cluster shape should be a small
   number of MPI ranks per node or per NUMA domain, each rank using
   `OMP_NUM_THREADS` for the local kernels.
3. Start with replicated data. Every rank may hold the full AO basis,
   density, Fock, and small SCF histories. This avoids ScaLAPACK and
   is the right first step for Ethernet clusters and most chemistry
   sizes.
4. Make the communication model explicit. Ethernet clusters need coarse
   tasks and infrequent reductions. Low-latency clusters can use finer
   shell-pair and matrix-block reductions.
5. Keep rank-0 ownership of files. Output files, QVF archives, Molden,
   cubes, and manifests should be written by rank 0 after gathering or
   broadcasting the final result.
6. Preserve bit-level expectations where possible. MPI reductions can
   change floating-point summation order. Tests should use tight numeric
   tolerances, but user-visible methods must reproduce serial chemistry
   within the existing parity bands.
7. Prefer rank-local heavy kernels over Python pickling. MPI
   orchestration can stay in Python for phase 1, but the data shipped
   between ranks should be NumPy arrays and scalars, not large Python
   object graphs.

## Cluster Tiers

| Cluster class | Recommended mode | Why |
| --- | --- | --- |
| Single workstation or single fat node | OpenMP only | No network cost, current kernels already scale. |
| Multi-node Ethernet | Coarse MPI tasks plus OpenMP inside each rank | Useful for k-points, states, pairs, and grid slabs where communication is limited. Fine shell-pair reductions may lose. |
| Low-latency interconnect | Coarse MPI first, then optional finer reductions | Exact-exchange shell-pair and distributed matrix work can pay once latency and bandwidth are good. |

The scheduler should never auto-spread a small job across nodes just
because MPI is installed. MPI should be opt-in or selected by a size
heuristic that can explain itself in the log.

## Periodic Strategy

### Natural First Target: K-Point Farming

For periodic calculations, k-points are the cleanest MPI axis:

- Per-k overlap/Hcore Bloch sums and orthogonalizers.
- Per-k diagonalization.
- Per-k occupations, density matrices, residuals, error vectors, and
  energy traces.
- Band, DOS, COOP/COHP, fat-band, and post-SCF k-resolved properties.

Rank ownership should be over a stable list of k-point indices, including
IBZ weights. Each rank computes its owned k-points, then the driver
reduces scalar totals and gathers the ordered per-k arrays on all ranks
or on rank 0 depending on the next phase of the SCF loop.

The first production implementation should target the drivers whose
Python structure already keeps per-k lists:

- `python/vibeqc/periodic_rhf_multi_k_ewald.py`
- `python/vibeqc/periodic_rks_multi_k_ewald.py`
- `python/vibeqc/periodic_uhf_multi_k_ewald.py`
- `python/vibeqc/periodic_gapw_j.py` multi-k RKS
- the native multi-k GDF drivers in `python/vibeqc/periodic_k_gdf.py`

Important caveat: not every periodic cost is per-k. In the legacy Ewald
path, much of the expensive two-electron work is built once as real-space
blocks and then Bloch-folded to all k-points. Distributing only the
Bloch-fold and diagonalization steps helps dense k-meshes but does not
parallelize the whole Fock build. By contrast, GDF, COSX long-range
exchange, response properties, and post-SCF k/q workloads have stronger
rank-local k- or q-point work.

### Periodic Phase Plan

**MPI-P0: make the existing substrate honest.**

- Keep `mpi4py` optional.
- Add a real `mpirun -np 2` smoke test for `python/vibeqc/mpi.py`.
- Validate `GpwJBuilder(mpi_aware=True)` under a real multi-rank run,
  not only simulated ranks.
- Make feature docs say "MPI substrate" and "experimental GPW overlay",
  not "HF/DFT/MP2 MPI shipped".

**MPI-P1: rank-local periodic utility layer.**

- ✅ Add a small `KPointPartition` helper over `distribute_tasks`
  (landed 2026-07-28, `python/vibeqc/mpi.py`).
- ✅ Add ordered gather/allgather helpers for per-k matrices, vectors,
  and scalar contributions (`KPointPartition.allgather_ordered`).
- Add a rank-aware progress/log policy: rank 0 prints SCF rows; other
  ranks stay quiet unless debugging is enabled.
- Record MPI size, rank policy, and threads-per-rank in `.out` and
  `.system`.

**MPI-P2: production multi-k SCF farming.**

- Start with pure RKS/RHF multi-k paths where per-k diagonalization,
  errors, occupations, density rebuilds, and energy traces are cleanly
  separated.
- Preserve serial code paths by making the partition helper return the
  full range when `mpi_size() == 1`.
- Reduce the per-k weighted energy, free energy, entropy, and residual
  norms with `Allreduce`.
- Gather ordered `C(k)`, `eps(k)`, `D(k)`, `F(k)`, and error lists only
  when the next accelerator or writer needs the full list.

**MPI-P3: k/q exchange and periodic post-SCF.**

- 🟡 For GDF and COSX exact exchange, distribute the natural momentum
  transfer or k-pair worklists where the algorithm couples k-points.
  Landed for multi-k GDF in full: the cderi *build* (2026-07-28,
  `periodic_k_gdf.py` q-group farming over `KPointPartition`, one
  partition/gather per SCF setup) and the per-iteration exchange
  *contraction* (2026-08-05, farmed over the bra k-index, exact
  reassembly via `allgather_ordered`). COSX and the other periodic
  drivers (Ewald and GPW/GAPW) are untouched. BIPOLE now has the separate
  D114 complete direct-output-block slice used by restricted χ four-center
  RHF; it is not a k/q exchange partition and leaves reciprocal work
  replicated.
- For periodic MP2 and CCM-like finite-character work, distribute
  momentum-conserving tuple batches. This is more valuable than merely
  farming diagonalizations because the actual correlation loops scale
  with k-tuples.
- For band structure, DOS, COOP/COHP, fat bands, and QTAIM sampling,
  use embarrassingly parallel k-point and band batches.

**MPI-P4: GPW/GAPW grid decomposition.**

- Replace the current zero-padded allreduce density assembly with
  `Allgatherv` or equivalent slab assembly.
- Decide whether a distributed FFT is worth a new optional dependency.
  Until then, the Poisson solve remains replicated and MPI only saves
  collocation/projection work.
- Extend the model from Hartree-J to XC grid projection and GAPW
  augmentation grids only after Gamma parity and real multi-rank tests
  are stable.

## Molecular Strategy

Molecules do not have k-points, so the MPI axes depend on the method.

### SCF HF And DFT

The straightforward molecular SCF model is replicated density and
rank-local matrix contributions:

- For HF exact exchange, distribute shell-pair or shell-quartet tasks.
  Each rank accumulates local J/K matrices, then allreduces the dense
  AO matrices once per SCF iteration.
- For DFT, distribute grid batches or atom-grid blocks. Each rank
  accumulates local `E_xc`, quadrature diagnostics, and `V_xc`, then
  allreduces scalars and the AO matrix.
- For DF-J/K, distribute auxiliary shell blocks or fit-vector batches
  when the `Lpq` tensor is too large for a single rank's memory.

This is best for very large molecules where each rank has enough local
integral or grid work to amortize a dense `nbf x nbf` allreduce. On
Ethernet, it should stay opt-in and guarded by a size heuristic. For
small and medium molecules, OpenMP plus job packing remains faster.

### MP2 And Double Hybrids

MP2 has a better distributed axis than SCF:

- Distribute occupied-pair batches `(i, j)` or spin-channel pair batches.
- Each rank transforms or reads the needed DF/MO blocks, computes local
  opposite-spin and same-spin energy contributions, and reduces scalar
  energies.
- For memory-bound large molecules, distribute the `B_ia^P` or `ovv`
  blocks rather than replicating the whole transformed tensor.

This is a strong early molecular target because the communication can be
mostly scalar reductions, and the local compute is large.

### Canonical CCSD And Triples

Canonical CCSD is harder:

- Current C++ kernels already OpenMP-parallelize scalar intermediates and
  triples worklists.
- A replicated-amplitude MPI version would allreduce large residual
  tensors every iteration. That can work on fast interconnects but is a
  poor first target for Ethernet.
- The `(T)` correction is easier: distribute occupied triples or the
  existing triples worklist, reduce the scalar triples energy.

Recommendation: implement MPI for `(T)` and selected tensor-build
worklists only after MP2 and DLPNO pair MPI are in place. Defer full
canonical distributed CCSD residuals until profiling shows it beats the
local-correlation route.

### DLPNO And Local Correlation

DLPNO is the best multi-node molecular correlation target:

- Pair domains are naturally rank-local.
- Pair energies, pair residual norms, and local triples corrections can
  be reduced or gathered with small messages.
- Sparse pair lists already express a work queue that can be partitioned
  statically or dynamically.

The first scalable molecular MPI production feature should be DLPNO pair
farming, not distributed canonical CCSD.

### Multireference And Selected CI

The multireference stack has several useful MPI axes:

- Selected-CI determinant batches for sigma-vector builds, selection, and
  deterministic PT2.
- CASPT2/NEVPT2 perturber classes or excitation-case partitions.
- State-specific CASPT2 solves in multi-state CASPT2.
- Finite-difference gradient displacements when the analytic gradient is
  unavailable.

The design should avoid shipping full determinant lists repeatedly. Rank
0 should build or load the determinant space, broadcast compact masks and
integral arrays, then each rank owns a deterministic range of determinants
or perturber classes.

### TDDFT And Response

`python/vibeqc/mpi_tddft.py` currently farms state-property analysis, but
that is a small part of the TDDFT cost. Production MPI should target:

- AO->MO ERI transform shell batches for hybrid TDDFT.
- Matrix-vector products for iterative Davidson/Lanczos TDDFT.
- State/NTO/spectrum post-processing as a low-risk first step.

Full dense Casida/TDA diagonalization should remain serial or threaded
until an iterative response solver is the default for large systems.

## User-Facing API

Do not make users import `vibeqc.mpi` for normal runs. The proposed
surface is:

```python
run_job(
    mol,
    basis="def2-svp",
    method="mp2",
    parallel={"backend": "mpi", "mode": "auto", "threads_per_rank": 8},
)
```

and for periodic:

```python
run_periodic_job(
    system,
    basis="pob-tzvp",
    method="rks",
    functional="pbe",
    kpoints=(6, 6, 6),
    parallel={"backend": "mpi", "decompose": "kpoints"},
)
```

The same policy should be reachable from environment variables for batch
scripts:

- `VIBEQC_PARALLEL_BACKEND=omp|mpi|auto`
- `VIBEQC_MPI_DECOMPOSE=auto|kpoints|grid|pairs|shells`
- `VIBEQC_THREADS_PER_RANK=N`

If MPI is active but the selected method has no production MPI path, the
driver should log a clear message and use OpenMP on each rank only if
that is exactly what the user asked for. It should not silently run the
same full calculation on every rank and write duplicate outputs.

## Scheduling Rules

Default rules for `vq` and cluster examples:

- Small molecules: one MPI rank, 4-8 OpenMP threads, pack many jobs per
  node.
- Large molecular DFT: MPI grid/shell mode only when `nbf` and grid size
  make the per-rank work large; otherwise OpenMP.
- MP2: 1 rank per NUMA domain or per node, OpenMP plus BLAS threads inside
  rank, distribute occupied pairs only for large cases.
- DLPNO: many ranks can help, but keep OpenMP modest per rank so pair
  queues stay balanced.
- Periodic dense k-meshes: ranks should not exceed useful k/q work items.
  Use fewer ranks with more OpenMP threads when the k-mesh is small.
- GPW/GAPW grid: use MPI only for grids large enough that slab
  collocation/projection dominates the replicated FFT.

Every MPI run should log:

- world size and host names if available,
- threads per rank,
- decomposition mode,
- local task counts per rank,
- allreduce/gather timing totals,
- a recommendation if MPI overhead exceeds the saved compute time.

## Validation Gates

Each MPI phase needs these gates before becoming user-facing:

1. `mpi4py` absent: import and normal tests still pass.
2. `mpirun -np 1`: bitwise or near-bitwise identity to serial.
3. `mpirun -np 2` and `-np 3`: parity against serial within existing
   method tolerances, including uneven task partitions.
4. Rank-local output discipline: only rank 0 writes user files.
5. Oversubscription guard: `OMP_NUM_THREADS`, BLAS thread variables, and
   `VIBEQC_THREADS_PER_RANK` are reported and respected.
6. Performance smoke: one Ethernet-like test where MPI is not selected
   for a too-small workload, and one larger workload where MPI improves
   wall time.
7. Restart/manifest: output records MPI mode so a calculation can be
   audited after the fact.

## Roadmap Placement

The v0.12.0 work should be described as "MPI substrate plus experimental
GPW z-slab overlay", not as production MPI for HF/DFT/MP2. Production MPI
belongs after the v0.15.x release-paper hardening line and before a
feature-complete v1.0 claim.

Recommended roadmap slot:

- The production MPI and large-system scaling milestone. Open milestones
  are themes, numbered only at the cut; see the [roadmap](roadmap.md).
- Dependencies: native BvK periodic local correlation for k/q post-SCF
  workloads, periodic CC for periodic tuple batching, and the v0.15.x
  hardening fixes for GPW/GDF/BIPOLE correctness.
- ScaLAPACK or another distributed dense eigensolver is not a phase-1
  requirement. It stays a later option if replicated dense matrices
  become the actual memory wall.

## Immediate Next Steps

1. Correct public feature docs to say the current MPI surface is a
   substrate plus experimental GPW overlay.
2. Add `handovers/HANDOVER_MPI.md` as the workstream tracker.
3. Add a real multi-rank smoke test for `vibeqc.mpi` and
   `GpwJBuilder(mpi_aware=True)`.
4. Implement `KPointPartition` and rank-aware ordered gather helpers.
5. Wire the first production periodic k-point farming path, starting with
   the lowest-risk pure DFT multi-k driver.
6. Benchmark on two hardware profiles: Ethernet multi-node and
   low-latency interconnect.
7. Only after that, pick the first molecular MPI target: MP2 occupied-pair
   farming or DLPNO pair farming.
