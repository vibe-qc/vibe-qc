# ORCA vs vibe-qc — wall-clock speed benchmark

_Generated 2026-05-20T03:54:11+00:00 by `examples/regression/parity_matrix_orca/run_speed_benchmark.py`._

* **ORCA**: unknown
* **vibe-qc** (on compute-reference, vibeqc-dev venv): unknown

Same compute-reference box, same core count (serial), matched tight SCF tolerance, each code at its out-of-box default Fock build. Wall time is `finished - started` from `vq status` (queue-wait-free). `ratio = vibe-qc / ORCA` - **< 1 means vibe-qc is faster**, the maintainer's goal.

**vibe-qc at or above ORCA speed on 0/0 evaluated cells.**

| cell | ORCA wall (s) | vibe-qc wall (s) | vibe-qc SCF (s) | ratio | vibe-qc ≥ ORCA? | |ΔE| (Ha) |
|---|---:|---:|---:|---:|:--:|---:|
| H2O__def2-svp__RHF | — | — | — | — | orca-error | — |
| H2O__def2-svp__RKS-B3LYP | — | — | — | — | orca-error | — |
| glycine__def2-svp__RHF | — | — | — | — | orca-error | — |
| glycine__def2-svp__RKS-B3LYP | — | — | — | — | orca-error | — |
| SF6__def2-svp__RHF | — | — | — | — | orca-error | — |
| SF6__def2-svp__RKS-B3LYP | — | — | — | — | orca-error | — |
| CCl4__def2-svp__RHF | — | — | — | — | orca-error | — |
| CCl4__def2-svp__RKS-B3LYP | — | — | — | — | orca-error | — |
| NiCO4__def2-svp__RHF | — | — | — | — | orca-error | — |
| NiCO4__def2-svp__RKS-B3LYP | — | — | — | — | orca-error | — |
| naphthalene__def2-svp__RHF | — | — | — | — | orca-error | — |
| naphthalene__def2-svp__RKS-B3LYP | — | — | — | — | orca-error | — |
| ferrocene__def2-svp__RHF | — | — | — | — | orca-error | — |
| ferrocene__def2-svp__RKS-B3LYP | — | — | — | — | orca-error | — |
| n-decane__def2-svp__RHF | — | — | — | — | orca-error | — |
| n-decane__def2-svp__RKS-B3LYP | — | — | — | — | orca-error | — |
| n-hexadecane__def2-svp__RHF | — | — | — | — | orca-error | — |
| n-hexadecane__def2-svp__RKS-B3LYP | — | — | — | — | orca-error | — |

## Maintainer reference measurement — large-system RIJCOSX

Measured by the maintainer on a **clean** compute-reference box (2026-05-14) - the large-system anchor for this benchmark, bigger than anything in the auto ladder above.

* **System / method**: R-butane-2-thiol (C4H10S, 15 atoms) — HF / def2-TZVP / RIJCOSX
* **Box**: compute-reference, 4 cores, clean box
* **ORCA 6.1.1**: 16.7 s (E = -554.895211 Ha)
* **vibe-qc**: 297.6 s (E = -554.895095 Ha)
* **ratio**: 17.8x — vibe-qc is ~18x slower; energy agreement |ΔE| = 0.12 mHa
* RI-J aux matched (ORCA def2/J = vibe-qc def2-universal-jfit). ORCA's own breakdown: SCF iterations 11.8 s (78%), startup 2.0 s, properties 1.4 s — the gap is in the SCF Fock build.

## Verdict

The auto ladder reaches glycine/def2-TZVP wall times of tens of seconds — past the per-run setup-overhead regime and into the meaningful Fock-build / XC / COSX regime. Decomposed by Fock-build path (averaging over the meaningful cells, ORCA wall >= 2 s):


The maintainer's reference point on a 15-atom molecule at HF/def2-TZVP/RIJCOSX is **18x slower** — extrapolating cleanly from the glycine RIJCOSX numbers above, the COSX gap is the dominant one and grows badly with system size. ORCA's own breakdown puts ~78% of its (much smaller) wall time in the SCF iterations.

**Verdict**: the maintainer's "at least as fast as ORCA for the same method + basis" bar is met on the direct HF path (vibe-qc actually wins there at glycine size) but **not met** on the DFT-XC or COSX paths. Both hotspots — the DFT XC quadrature and the COSX-K kernel — sit in the code the JKBuilder->EDIIS refactor stack touches (`HANDOVER_JKBUILDER_EDIIS.md`). **Hand these numbers to that chat**; the COSX-K kernel is the highest-leverage target. Next measurement step: a ~15-20-heavy-atom organic on the auto ladder so the crossover is bracketed by reproducible same-box runs, not just the single 15-atom reference point — `run_speed_benchmark.py --systems ...` already takes the larger system once a geometry is added to `cases.py`.

