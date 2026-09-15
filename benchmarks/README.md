# benchmarks/

Performance evaluations of vibe-qc against external reference codes.
Distinct from `tests/` (correctness / parity) and from
`examples/regression/parity_matrix_orca/` (per-piece energy-decomposition
parity, where the question is "do we get the same number?"). The
question here is "do we get to the same number as fast?".

## Current reports

| file | what it measures | how to regenerate |
|---|---|---|
| [`orca_vs_vibeqc_speed.md`](orca_vs_vibeqc_speed.md) | wall-clock parity of ORCA 6.1.1 vs vibe-qc on a 3 → 50 atom ladder, direct + RIJCOSX paths, HF + B3LYP | `python -m examples.regression.parity_matrix_orca.run_speed_benchmark` |

The script that generates `orca_vs_vibeqc_speed.md` lives at
[`examples/regression/parity_matrix_orca/run_speed_benchmark.py`](../examples/regression/parity_matrix_orca/run_speed_benchmark.py)
- it submits both codes to compute-reference via `vq` so the comparison is
same-box. The generated report includes the ORCA and vibe-qc
(vibeqc-dev venv) versions in its header for reproducibility.

## Conventions

* **Reports are committed.** They are the *artifacts*; the regenerator
  in `examples/regression/...` is the *recipe*. Re-running it
  overwrites the report — the prior numbers stay in git history.
* **Same-box only.** Comparing a laptop run against a compute-reference run is
  not a benchmark, it is noise. Both codes go through `vq` to the same
  host. The reports list the host + core count in the header.
* **Versions in the header.** Always: the external code's version + the
  vibe-qc commit/release the numbers were measured against.
* **Verdicts route to the chat that owns the relevant code.** The
  current `orca_vs_vibeqc_speed.md` verdict routes the COSX-K and
  DFT-XC hotspots to the molecular-perf / direct-SCF arc; the
  underlying JKBuilder + EDIIS+DIIS infrastructure landed on
  `origin/main` @ `d2099de` (2026-05-15).
