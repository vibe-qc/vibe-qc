# GPU G0-A: CPU component-timing pilot (H2O)

Status: **pilot only.** This is the first G0-A measurement from the
[G0 plan](../docs/design/gpu_g0_g1m_plan.md). It covers the small H2O case
only. Naphthalene and n-hexadecane have not been run, so this report does not
rank bottlenecks for the G0 stage and makes no GPU claim. The machine-readable
version is [`gpu_g0_summary.json`](gpu_g0_summary.json).

## What ran

- System: H2O from `parity_matrix_orca/cases.py` (single Angstrom-to-Bohr
  conversion), def2-SVP, 24 basis functions, 5 occupied orbitals.
- Methods: RHF, RKS/PBE, RKS/PBE0. Routes: forced direct, DF
  (`def2-universal-jkfit`, 113 auxiliary functions), RIJCOSX
  (`def2-universal-jfit`, 71 auxiliary functions, GridX level 2, FITTED
  variant, stage grids of 3954, 9444 and 23862 points) and resolved AUTO.
  AUTO resolved to the conventional in-core builder at 24 basis functions, so
  it was not deduplicated against forced direct.
- Recipe: energy tolerance 1e-9 Eh, commutator tolerance 1e-6
  (`||FDS - SDF||_F`), at most 200 iterations, SAD guess, incremental direct
  Fock build at its default (on). XC grid pinned to Generic, 75 radial, product
  17 x 36 angular, Becke k=3, no pruning (137700 points). Stability analysis at
  its default (on). No tolerance clamp.
- Protocol: one queued `vq` job on a 4-CPU allocation; every run is a fresh
  process. Per case and thread setting, one untimed warm-up, then three timed
  runs with component timing on and three with it off, in a seeded random
  order (seed 20261027). OpenMP 1 and 4 threads, BLAS 1 thread.
- Runtime: source `679f03e5`, clean tree; native core SHA-256
  `f7ba7161beb89d52074e0472403013e38050705b4d46a060072ddb101ab6d623`,
  Apple LLVM 21.0.0, OpenMP 202011, Eigen 3.5.0, Accelerate, Libint 2.13.1,
  Libxc 7.0.0, Python 3.14.9, NumPy 2.5.4, macOS 27 on an Apple M5 Max.
  Source and runtime identity matched for every row. Raw records, logs and
  `.out` files are kept outside the repository.

Regenerate with `examples/regression/gpu_profile.py submit --systems H2O
--threads 1,4 --compare-timing-off ...` and `summarize --md-out`.

## Results

Status PASS means converged, identical source/runtime identity, clean tree,
repeat energies within 1e-8 Eh and reconciled timers. "Last residual" is the
largest final trace commutator over the repeats.

| Case | Threads | Builder | Status | Iter | Energy (Eh) | Last residual | SCF wall median [min, max] (s) | Peak RSS (MiB) |
|---|---|---|---|---|---|---|---|---|
| H2O__def2-svp__PBE0__auto | 1 | conventional_four_index | PASS | 9 | -76.2733120507 | 2.7e-07 | 4.516 [4.459, 4.545] | 263 |
| H2O__def2-svp__PBE0__auto | 4 | conventional_four_index | PASS | 9 | -76.2733120507 | 2.7e-07 | 1.619 [1.601, 1.895] | 296 |
| H2O__def2-svp__PBE0__df | 1 | df_j_k | PASS | 9 | -76.2733177277 | 2.7e-07 | 4.494 [4.435, 4.494] | 264 |
| H2O__def2-svp__PBE0__df | 4 | df_j_k | PASS | 9 | -76.2733177277 | 2.7e-07 | 1.733 [1.567, 1.741] | 296 |
| H2O__def2-svp__PBE0__direct | 1 | direct | PASS | 12 | -76.2733120507 | 1.0e-07 | 5.076 [4.902, 5.111] | 264 |
| H2O__def2-svp__PBE0__direct | 4 | direct | PASS | 12 | -76.2733120507 | 1.0e-07 | 1.796 [1.757, 1.799] | 306 |
| H2O__def2-svp__PBE0__rijcosx | 1 | ri_j_cosx_k | PASS | 13 | -76.2734210466 | 3.8e-03 (above target) | 3.784 [3.724, 3.964] | 216 |
| H2O__def2-svp__PBE0__rijcosx | 4 | ri_j_cosx_k | PASS | 13 | -76.2734210466 | 3.8e-03 (above target) | 1.235 [1.230, 1.365] | 268 |
| H2O__def2-svp__PBE__auto | 1 | conventional_four_index | PASS | 10 | -76.2709676966 | 2.6e-07 | 4.406 [4.405, 4.556] | 245 |
| H2O__def2-svp__PBE__auto | 4 | conventional_four_index | PASS | 10 | -76.2709676966 | 2.6e-07 | 1.598 [1.551, 1.685] | 281 |
| H2O__def2-svp__PBE__df | 1 | df_j_only | PASS | 10 | -76.2709944630 | 2.6e-07 | 4.428 [4.426, 4.453] | 248 |
| H2O__def2-svp__PBE__df | 4 | df_j_only | PASS | 10 | -76.2709944630 | 2.6e-07 | 1.647 [1.501, 1.761] | 284 |
| H2O__def2-svp__PBE__direct | 1 | direct | PASS | 13 | -76.2709676966 | 6.9e-09 | 5.013 [4.966, 5.159] | 245 |
| H2O__def2-svp__PBE__direct | 4 | direct | PASS | 13 | -76.2709676966 | 6.9e-09 | 1.822 [1.808, 2.004] | 281 |
| H2O__def2-svp__PBE__rijcosx | 1 | ri_j_only | PASS | 12 | -76.2710544933 | 7.9e-09 | 7.502 [7.406, 7.749] | 262 |
| H2O__def2-svp__PBE__rijcosx | 4 | ri_j_only | PASS | 12 | -76.2710544933 | 7.9e-09 | 3.034 [2.795, 3.471] | 330 |
| H2O__def2-svp__RHF__auto | 1 | conventional_four_index | PASS | 11 | -75.9547597040 | 1.4e-07 | 0.026 [0.025, 0.027] | 173 |
| H2O__def2-svp__RHF__auto | 4 | conventional_four_index | PASS | 11 | -75.9547597040 | 1.4e-07 | 0.011 [0.011, 0.012] | 180 |
| H2O__def2-svp__RHF__df | 1 | df_j_k | PASS | 11 | -75.9547034344 | 1.4e-07 | 0.018 [0.017, 0.019] | 174 |
| H2O__def2-svp__RHF__df | 4 | df_j_k | PASS | 11 | -75.9547034344 | 1.4e-07 | 0.018 [0.018, 0.020] | 182 |
| H2O__def2-svp__RHF__direct | 1 | direct | PASS | 13 | -75.9547597040 | 7.3e-08 | 0.097 [0.096, 0.102] | 172 |
| H2O__def2-svp__RHF__direct | 4 | direct | PASS | 13 | -75.9547597040 | 7.3e-08 | 0.036 [0.035, 0.037] | 180 |
| H2O__def2-svp__RHF__rijcosx | 1 | ri_j_cosx_k | PASS | 15 | -75.9549216556 | 1.5e-02 (above target) | 1.270 [1.260, 1.300] | 206 |
| H2O__def2-svp__RHF__rijcosx | 4 | ri_j_cosx_k | PASS | 15 | -75.9549216556 | 1.5e-02 (above target) | 0.396 [0.384, 0.414] | 242 |

Median exclusive phase seconds (component timing on):

| Case | Threads | setup | df_construction | grid_construction | jk_fused | j | k | k_correction | ao_xc_vxc | accelerator | diagonalization | density | final_rebuild_other | stability | unclassified |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H2O__def2-svp__PBE0__auto | 1 | 0.003 | - | 0.005 | - | 0.002 | 0.002 | - | 1.847 | 0.001 | 0.000 | 0.000 | 0.000 | 2.629 | 0.000 |
| H2O__def2-svp__PBE0__auto | 4 | 0.003 | - | 0.002 | - | 0.001 | 0.001 | - | 0.629 | 0.001 | 0.000 | 0.000 | 0.000 | 1.017 | 0.000 |
| H2O__def2-svp__PBE0__df | 1 | 0.004 | 0.001 | 0.005 | - | 0.001 | 0.004 | - | 1.829 | 0.001 | 0.000 | 0.000 | 0.000 | 2.645 | 0.000 |
| H2O__def2-svp__PBE0__df | 4 | 0.004 | 0.001 | 0.002 | - | 0.001 | 0.005 | - | 0.604 | 0.001 | 0.000 | 0.000 | 0.000 | 1.096 | 0.000 |
| H2O__def2-svp__PBE0__direct | 1 | 0.002 | - | 0.004 | - | 0.016 | 0.016 | - | 2.340 | 0.000 | 0.000 | 0.000 | 0.000 | 2.646 | 0.000 |
| H2O__def2-svp__PBE0__direct | 4 | 0.002 | - | 0.002 | - | 0.005 | 0.005 | - | 0.744 | 0.000 | 0.000 | 0.000 | 0.000 | 1.034 | 0.000 |
| H2O__def2-svp__PBE0__rijcosx | 1 | 0.006 | 0.002 | 0.022 | - | 0.000 | 1.047 | 0.062 | 2.645 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.001 |
| H2O__def2-svp__PBE0__rijcosx | 4 | 0.005 | 0.003 | 0.010 | - | 0.000 | 0.283 | 0.068 | 0.871 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.001 |
| H2O__def2-svp__PBE__auto | 1 | 0.003 | - | 0.004 | - | 0.002 | - | - | 1.872 | 0.001 | 0.000 | 0.000 | 0.000 | 2.522 | 0.000 |
| H2O__def2-svp__PBE__auto | 4 | 0.003 | - | 0.002 | - | 0.001 | - | - | 0.644 | 0.001 | 0.000 | 0.000 | 0.000 | 0.946 | 0.000 |
| H2O__def2-svp__PBE__df | 1 | 0.004 | 0.001 | 0.004 | - | 0.001 | - | - | 1.910 | 0.001 | 0.000 | 0.000 | 0.000 | 2.512 | 0.000 |
| H2O__def2-svp__PBE__df | 4 | 0.004 | 0.001 | 0.002 | - | 0.001 | - | - | 0.643 | 0.001 | 0.000 | 0.000 | 0.000 | 0.993 | 0.000 |
| H2O__def2-svp__PBE__direct | 1 | 0.002 | - | 0.005 | - | 0.017 | - | - | 2.463 | 0.000 | 0.001 | 0.000 | 0.000 | 2.523 | 0.000 |
| H2O__def2-svp__PBE__direct | 4 | 0.002 | - | 0.002 | - | 0.006 | - | - | 0.828 | 0.000 | 0.001 | 0.000 | 0.000 | 1.033 | 0.000 |
| H2O__def2-svp__PBE__rijcosx | 1 | 0.006 | 0.002 | 0.021 | - | 0.001 | - | - | 2.409 | 0.000 | 0.001 | 0.000 | 0.000 | 5.015 | 0.001 |
| H2O__def2-svp__PBE__rijcosx | 4 | 0.005 | 0.003 | 0.013 | - | 0.001 | - | - | 0.921 | 0.001 | 0.001 | 0.000 | 0.000 | 2.087 | 0.001 |
| H2O__def2-svp__RHF__auto | 1 | 0.003 | - | - | 0.003 | 0.000 | 0.000 | - | - | 0.001 | 0.000 | 0.000 | 0.000 | 0.017 | 0.000 |
| H2O__def2-svp__RHF__auto | 4 | 0.003 | - | - | 0.001 | 0.000 | 0.000 | - | - | 0.001 | 0.000 | 0.000 | 0.000 | 0.005 | 0.000 |
| H2O__def2-svp__RHF__df | 1 | 0.004 | 0.001 | - | 0.000 | 0.001 | 0.008 | - | - | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| H2O__def2-svp__RHF__df | 4 | 0.004 | 0.001 | - | 0.000 | 0.001 | 0.008 | - | - | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| H2O__def2-svp__RHF__direct | 1 | 0.002 | - | - | 0.016 | 0.001 | 0.001 | - | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.075 | 0.000 |
| H2O__def2-svp__RHF__direct | 4 | 0.002 | - | - | 0.005 | 0.000 | 0.000 | - | - | 0.000 | 0.000 | 0.000 | 0.000 | 0.027 | 0.000 |
| H2O__def2-svp__RHF__rijcosx | 1 | 0.006 | 0.002 | 0.017 | 0.000 | 0.000 | 1.178 | 0.064 | - | 0.001 | 0.001 | 0.000 | 0.000 | 0.000 | 0.001 |
| H2O__def2-svp__RHF__rijcosx | 4 | 0.005 | 0.003 | 0.009 | 0.000 | 0.000 | 0.308 | 0.066 | - | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.001 |

Instrumentation on versus off:

| Case | Threads | dE (Eh) | Same iterations | Median on (s) | Median off (s) | Overhead | Ranges overlap |
|---|---|---|---|---|---|---|---|
| H2O__def2-svp__PBE0__auto | 1 | 0.0e+00 | True | 4.516 | 4.450 | +1.5% | True |
| H2O__def2-svp__PBE0__auto | 4 | 0.0e+00 | True | 1.619 | 1.587 | +2.0% | False |
| H2O__def2-svp__PBE0__df | 1 | 0.0e+00 | True | 4.494 | 4.527 | -0.7% | True |
| H2O__def2-svp__PBE0__df | 4 | 0.0e+00 | True | 1.733 | 1.611 | +7.6% | True |
| H2O__def2-svp__PBE0__direct | 1 | 0.0e+00 | True | 5.076 | 4.936 | +2.8% | True |
| H2O__def2-svp__PBE0__direct | 4 | 2.8e-14 | True | 1.796 | 1.825 | -1.6% | True |
| H2O__def2-svp__PBE0__rijcosx | 1 | 0.0e+00 | True | 3.784 | 3.710 | +2.0% | True |
| H2O__def2-svp__PBE0__rijcosx | 4 | 0.0e+00 | True | 1.235 | 1.323 | -6.6% | True |
| H2O__def2-svp__PBE__auto | 1 | 0.0e+00 | True | 4.406 | 4.386 | +0.5% | False |
| H2O__def2-svp__PBE__auto | 4 | 0.0e+00 | True | 1.598 | 1.641 | -2.6% | True |
| H2O__def2-svp__PBE__df | 1 | 0.0e+00 | True | 4.428 | 4.523 | -2.1% | False |
| H2O__def2-svp__PBE__df | 4 | 0.0e+00 | True | 1.647 | 1.803 | -8.7% | True |
| H2O__def2-svp__PBE__direct | 1 | 0.0e+00 | True | 5.013 | 5.057 | -0.9% | True |
| H2O__def2-svp__PBE__direct | 4 | -2.8e-14 | True | 1.822 | 1.902 | -4.2% | True |
| H2O__def2-svp__PBE__rijcosx | 1 | 0.0e+00 | True | 7.502 | 7.665 | -2.1% | True |
| H2O__def2-svp__PBE__rijcosx | 4 | 0.0e+00 | True | 3.034 | 3.071 | -1.2% | True |
| H2O__def2-svp__RHF__auto | 1 | 0.0e+00 | True | 0.026 | 0.025 | +5.8% | True |
| H2O__def2-svp__RHF__auto | 4 | 0.0e+00 | True | 0.011 | 0.011 | -1.0% | True |
| H2O__def2-svp__RHF__df | 1 | 0.0e+00 | True | 0.018 | 0.018 | +0.5% | True |
| H2O__def2-svp__RHF__df | 4 | 0.0e+00 | True | 0.018 | 0.017 | +6.0% | True |
| H2O__def2-svp__RHF__direct | 1 | 0.0e+00 | True | 0.097 | 0.100 | -2.5% | True |
| H2O__def2-svp__RHF__direct | 4 | 2.8e-14 | True | 0.036 | 0.034 | +6.9% | True |
| H2O__def2-svp__RHF__rijcosx | 1 | 0.0e+00 | True | 1.270 | 1.271 | -0.1% | True |
| H2O__def2-svp__RHF__rijcosx | 4 | 0.0e+00 | True | 0.396 | 0.397 | -0.3% | True |

Phase groups are exclusive seconds and, with `unclassified`, sum to the SCF
session wall time. `jk_fused` is one RHF `build_g_rhf` call. For direct and
conventional builders it contains all J and K work, which is not split.
`j`/`k` collect separate driver or kernel phases, including the final rebuild.
`ao_xc_vxc` is AO collocation, XC evaluation and Vxc assembly, which run
batched together. `stability` is the post-convergence stability analysis.

## Observations

- **Instrumentation does not change results.** Across all 24 on/off pairs the
  energies differ by at most 3e-14 Eh, with identical iteration counts. The
  median wall-time overhead is -0.5% (range -9% to +8%). The on and off
  ranges overlap in 21 of 24 pairs, with no consistent sign, so the overhead is below the measured run
  noise at this size, not a resolved number.
- **Stability analysis dominates most DFT runs.** It takes 2.5-2.6 s of a
  4.4-5.1 s single-thread SCF for PBE and PBE0 on the direct, DF and AUTO routes,
  and 5.0 s of 7.5 s for PBE on RIJCOSX. It is skipped (about 0 s) on the PBE0
  and RHF COSX routes. It should be measured separately from SCF bottlenecks.
- **XC on this grid is the next cost.** `ao_xc_vxc` takes 1.8-2.6 s single
  thread. In H2O, J/K costs at most 1.2 s, all of it COSX-K; DF construction
  takes 1-3 ms.
- **COSX-K** costs 1.05-1.18 s single thread for 13-15 iterations at this
  size, and the one-center correction about 0.06 s.
- **Converged with residual above target.** The RHF and PBE0 RIJCOSX runs
  report convergence while the last trace commutator is 1.5e-2 and 3.8e-3. The
  staged GridX path finishes on its final grid stage, and this pilot does not
  establish whether that meets the common recipe. Treat these rows as not
  matched to the 1e-6 commutator target until that is resolved.
- **Memory.** Process peak RSS is 170-330 MiB and mostly interpreter and
  library baseline at this size. Tracked DF tensors are 1.1 MB live and
  2.4 MB at the construction boundary for the JK-fit basis, and 0.7 MB live and
  1.4 MB at that boundary for the J-fit basis. They do not include BLAS or
  solver scratch.
- RHF on this size runs in 10-100 ms, so its phase times are near clock and
  process noise.

## Measurement gaps

- Naphthalene and n-hexadecane, the H2O/def2-TZVP diagnostic and CPU noise
  floors are not yet measured.
- AO collocation is not timed separately from XC evaluation and Vxc assembly.
- Observed OpenMP team sizes per kernel are not exposed by the high-level
  drivers. The records hold the requested count, the runtime maximum and the
  RKS XC batch workers. BLAS threads are set through the environment and not
  observed.
- No BLAS-parallel run of the dominant DF operation was made.
- Tracked memory covers DF construction only. COSX grid batches, AO caches and
  the in-core four-index tensor are not tracked. Peak RSS is the process
  high-water mark only.
- The stage-change grid build in staged COSX runs outside a named phase and is
  counted as `unclassified`. Incremental-direct recovery rebuilds are inside
  the Fock phase of the iteration that does them, without a separate timer.
- UHF/UKS and periodic drivers are not instrumented.
