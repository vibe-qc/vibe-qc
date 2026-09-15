# ORCA vs vibe-qc — wall-clock speed benchmark (direct SCF)

_Generated 2026-05-20T13:23:16+00:00 on compute-small (i9-10885H, 4 threads)
by `examples/regression/parity_matrix_orca/run_speed_benchmark.py`._

* **ORCA**: 6.1.1
* **vibe-qc**: 0.8.1.dev0 (main @ 83a5da8, after mol-direct audit fixes)

Same compute-small box, same core count (4 threads), matched tight SCF tolerance
(ORCA `VeryTightSCF` / vibe-qc `conv_tol_energy=1e-10`). Wall time is
`finished - started` from `vq status` (queue-wait-free).
`ratio = vibe-qc / ORCA` — **< 1 means vibe-qc is faster**, the
maintainer's goal.

**This run exercises the direct (integral-driven, Schwarz-screened)
Fock build only** — the path that replaced the in-core 4-index ERI
tensor after the mol-direct audit (commits `56ffec2`, `f1da5e6`,
`f389a8c`). RIJCOSX and DFT cells are not included; they belong to
the COSX-perf track.

**vibe-qc at or above ORCA speed on 3/6 evaluated cells.**

| cell | ORCA wall (s) | vibe-qc wall (s) | vibe-qc SCF (s) | ratio | vibe-qc ≥ ORCA? | |ΔE| (Ha) |
|---|---:|---:|---:|---:|:--:|---:|
| H2O__def2-svp__RHF | 15.03 | 2.01 | 0.424 | 0.13 | ✓ | 9.72e-11 |
| glycine__def2-svp__RHF | 13.02 | 5.03 | 3.647 | 0.39 | ✓ | 3.93e-10 |
| naphthalene__def2-svp__RHF | 45.10 | 31.09 | 29.615 | 0.69 | ✓ | 5.56e-10 |
| ferrocene__def2-svp__RHF | 179.54 | 360.91 | 358.557 | 2.01 | ✗ | 1.90e+01 ⚠ |
| n-decane__def2-svp__RHF | 67.15 | 226.30 | 224.759 | 3.37 | ✗ | 1.11e-06 |
| n-hexadecane__def2-svp__RHF | 185.28 | 639.15 | 637.508 | 3.45 | ✗ | 2.36e-06 |

⚠ **Ferrocene (Fe d⁶ LS)**: the 19 Ha energy delta is a convergence
failure — the SAD initial guess + default (0.5) damping does not
converge for this transition-metal system. With `damping=0.7` it
converges to E = −1646.3163055 Ha (matches ORCA to 0.5 µHa) in
96 iterations. The raw timing data is included for transparency but
should be disregarded for ratio analysis. This is a pre-existing
SCF convergence issue unrelated to the direct Fock build.

## What this means for direct SCF

| Size range | nbf | vibe-qc vs ORCA | Status |
|---|---|---|---|
| Tiny (H2O) | 24 | 8× faster | ORCA startup overhead dominates |
| Small (glycine) | 95 | 2.6× faster | In-core tensor wins |
| Medium (naphthalene) | 180 | 1.4× faster | Near crossover — direct path holding |
| Large (n-decane) | 250 | 3.4× slower | Direct Fock build gap opens |
| Very large (n-hexadecane) | 394 | 3.5× slower | Gap plateaus — ~3.5× from here up |

**Energy agreement**: all converged cells match ORCA to ≤ 2.4 µHa —
the direct kernel is producing the right numbers.

**Memory wall**: closed. n-hexadecane at 394 BF no longer OOMs. The
May-15 baseline had `vibeqc-error` for this cell (192 GB ERI tensor
under the old in-core path).

**Performance**: the ~3.5× gap on the large alkane cells is the real
remaining Fock-build gap vs ORCA's direct SCF. This is a 4.6×
improvement over the pre-fix state (16× at 1 cpu on compute-small), but still
short of the "at least as fast as ORCA" bar at scale. The per-iteration
Fock build is the bottleneck — ORCA's direct kernel is still faster.

## Next steps

1. **Profile the direct Fock hot loop** on n-hexadecane — the libint
   quartet evaluation + 4-index accumulation is the candidate hotspot.
2. **Benchmark with `incremental_fock=True`** on the alkane ladder —
   we observed a 2× speedup locally, but the tight `conv_tol_energy=1e-10`
   in this benchmark can't be met by the incremental ΔP path (drift floor
   ~1e-7 Ha). Raise convergence to `1e-7` for incremental benchmarks.
3. **Re-run ferrocene** with `damping=0.7` to get a clean TM data point.
4. **COSX-perf track** — the RIJCOSX gap (18× on butanethiol) is the
   higher-leverage target; direct SCF is now within a small multiple.
