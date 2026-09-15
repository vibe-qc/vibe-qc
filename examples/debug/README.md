# `examples/debug/` — substrate scripts for debugging vibe-qc

Pinned reproducers for SCF + Fock-build debugging. Each script
answers ONE question. All wrap in `vibeqc.crash_dump_context` +
`vibeqc.perf_log` so failures leave artifacts on disk for
post-mortem inspection without re-running.

| Script | Question it answers | Wall |
|---|---|---|
| [`scf_easy_periodic.py`](scf_easy_periodic.py) | Is the periodic SCF stack itself healthy, or just unhappy with my system? Runs three known-good closed-shell wide-gap insulators (Ne / Ar / diamond C) — must converge in <10 iters. | ~30 s |
| [`scf_minimal_reproducer.py`](scf_minimal_reproducer.py) | Is the bug in the chemistry or in the SCF orchestration? Single He atom in a 3 Å cubic cell, sto-3g (1 bf!), Γ-only RKS-LDA. Smallest possible 3D periodic SCF. | ~30 s |
| [`scf_size_bisection.py`](scf_size_bisection.py) | At what system size does the bug start? Five-system scan: He → Mg → NaCl primitive (XFAIL) → NaCl conv → MgO conv. Honours `VIBEQC_FAST_DEBUG=1` to coarsen all knobs. | ~5–15 min (~1–3 min FAST) |
| [`scf_convergence_sweep.py`](scf_convergence_sweep.py) | Which convergence aid fixes my divergence? Sweeps initial guess × damping × level shift × DIIS × smearing on NaCl-LDA — 16 configs, all dumped on failure. | ~2–10 min |
| [`scf_sad_vs_hcore.py`](scf_sad_vs_hcore.py) | Does SAD (v0.6.1) rescue divergences that HCORE couldn't handle? A/B test on three hard ionic insulators (LiH / NaCl / MgO). | ~5–15 min |
| [`scf_iteration_recorder.py`](scf_iteration_recorder.py) | What does my failed SCF trajectory actually look like? Captures every iteration's `(E, ΔE, ‖[F,DS]‖, DIIS dim)` and plots a 4-panel forensics figure. | varies (script wraps any system you point it at) |
| [`scf_molecular_limit_check.py`](scf_molecular_limit_check.py) | Does my periodic SCF give the right molecular limit (vacuum-padded box → same energy as molecular RHF)? Detects the Madelung self-image leak that v0.6.0 currently has. Reports observed vs predicted over-bind, scales H₂ across L = 30/50/100 bohr to verify 1/L. | ~1 min |
| [`scf_vs_pyscf.py`](scf_vs_pyscf.py) | Is my vibe-qc divergence a vibe-qc bug or a hard system? Head-to-head against PySCF's `pbc.dft.RKS` on Ne / Ar / diamond C — same basis, same XC, same k-mesh. Skip PySCF column if not installed. | ~1-2 min |
| [`scf_perf_scaling.py`](scf_perf_scaling.py) | What's the wall-time vs precision trade-off for each EWALD_3D knob? Scans `spacing_bohr` × `cutoff_bohr` × `omega` on NaCl/STO-3G/RKS-LDA, reports per-config wall + ΔE vs the tightest reference. | ~3-10 min |

## Output convention

Each script writes under `examples/debug/output/<scriptname>/`:

  - `<label>.out`  — line-buffered live SCF log
  - `<label>.perf` — per-stage timings (post-mortem text report)
  - `<label>.dump` — TOML crash dump on SCF failure
  - `<label>.dump.density.npy` — last-iter density on failure
  - `summary.csv` — table of all results (sweep / bisection scripts)

Inspect a failure post-hoc:

```python
import vibeqc as vq
dump = vq.load_dump("examples/debug/output/sad-vs-hcore/NaCl_HCORE.dump")
print(dump["crash"]["phase"], "→", dump["hint"]["likely_cause"])
last = dump["scf.last_iter"]   # iter, energy, dE, grad_norm, diis
```

## Decision tree

  1. **"Is anything in vibe-qc periodic SCF working?"**
     → start with `scf_easy_periodic.py`. If all three pass, the
        framework is fine. If any fails, that's a regression and
        the rest of the questions don't matter yet.

  2. **"Why is my specific system diverging?"**
     → run `scf_iteration_recorder.py` on it (edit `build_system()`
        at the bottom). The 4-panel plot tells you whether it's
        DIIS instability (oscillating E), bad initial guess
        (monotone climb), or stuck saddle (flat ‖[F,DS]‖).

  3. **"Is there a convergence-aid combination that works?"**
     → `scf_convergence_sweep.py` for ALL knobs at once, or the
        more targeted `scf_sad_vs_hcore.py` for just the initial
        guess (now that v0.6.1 wired SAD).

  4. **"Is the bug system-size-dependent?"**
     → `scf_size_bisection.py`. The smallest reproducing case
        gives you the fastest turnaround on any subsequent fix.

  5. **"Is this a vibe-qc bug or just a hard system?"**
     → start with `scf_easy_periodic.py` to rule out framework
        regression. Then for the hard case, cross-check against
        PySCF (`pyscf.pbc.dft.RKS`) — if both codes diverge it's
        the system, not vibe-qc.

## Adding more debug scripts

Keep them ≤250 lines, one question each, output under
`output/<scriptname>/`, wrap in `crash_dump_context` + `perf_log`,
emit a CSV summary if multi-config. Update this README.
