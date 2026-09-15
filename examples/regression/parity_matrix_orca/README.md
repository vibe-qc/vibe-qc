# ORCA parity matrix + ORCA-vs-vibe-qc speed benchmark

The **ORCA axis** of the HF/DFT cross-code parity matrix, plus the
**ORCA-vs-vibe-qc wall-clock speed benchmark**. Both deliverables came from
the retired ORCA-parity handover (2026-06-13; recover via git history).

`tests/test_parity_hf_dft.py` already certifies vibe-qc's
per-intermediate energy decomposition against **PySCF**. This package
adds **ORCA 6.1.1** as a second, independent reference, and measures
whether vibe-qc keeps up with ORCA on wall-clock time.

ORCA runs **out-of-process on compute-host-d via the `vq` queue** — vibe-qc
never imports ORCA (CLAUDE.md § 10). ORCA results are **cached as
committed JSON** under `cache/`, so `tests/test_parity_vs_orca.py`
runs fast in CI with no ORCA / vq / network.

## Layout

| file | role |
|---|---|
| `cases.py` | cell definitions (`PARITY_CELLS`, the speed ladder) + geometries |
| `orca_input.py` | `(cell) -> ORCA input deck` |
| `parse_orca.py` | `ORCA .out -> decomposition dict` + mandatory self-check |
| `orca_vq.py` | submit / poll / fetch one ORCA cell through the `vq` CLI |
| `vibeqc_compare.py` | the **only** module that imports vibe-qc: runs the vibe-qc SCF + decomposition, compares against ORCA |
| `run_parity.py` | deliverable 1 driver — run/cache ORCA, compare, write `<output-root>/<run-id>/PARITY_REPORT.md` |
| `vibeqc_timing_job.py` | runs *on compute-host-d*: times one vibe-qc SCF, emits a result marker |
| `run_speed_benchmark.py` | deliverable 2 driver — ORCA vs vibe-qc wall clock, writes `<output-root>/<run-id>/orca_vs_vibeqc_speed.md` |
| `cache/*.json` | committed ORCA decompositions (the CI fixture) |
| [`contrib/run-orca.sh` in vibe-queue](https://github.com/vibe-qc/vibe-queue/blob/main/contrib/run-orca.sh) | copied from the separate queue checkout into each job workspace |

Tests: `tests/test_parse_orca.py` (parser, hermetic — uses the
committed ORCA samples), `tests/test_parity_vs_orca.py` (the matrix —
vibe-qc live vs cached ORCA, skip-if-no-cache).

## Running it

For live runs, install [vibe-queue](https://github.com/vibe-qc/vibe-queue)
separately and put its `vq` command on PATH; see [queue setup](../../../docs/user_guide/queue.md).
Set `VIBEQC_QUEUE_CHECKOUT` to its absolute checkout path on the **submission
machine** (default: `~/gitlab/vibe-queue`). The driver copies
`contrib/run-orca.sh` from there into the job workspace. No queue source is
present in the core checkout. Cached comparisons need neither this checkout
nor a running queue.

```sh
# deliverable 1 — compare every cell that has a cached ORCA result
python -m examples.regression.parity_matrix_orca.run_parity

# run ONE cell end-to-end on compute-host-d first (nail the loop), keep the
# regenerated cache copy in the run directory
python -m examples.regression.parity_matrix_orca.run_parity \
    --first --run --output-root ~/vibeqc-runs

# fill in / refresh every run-local cache entry by running ORCA on compute-host-d
python -m examples.regression.parity_matrix_orca.run_parity \
    --run --output-root ~/vibeqc-runs

# intentionally refresh the committed CI fixture cache as well
python -m examples.regression.parity_matrix_orca.run_parity \
    --run --update-fixture-cache --output-root ~/vibeqc-runs

# deliverable 2 — ORCA vs vibe-qc wall clock, both on compute-host-d
python -m examples.regression.parity_matrix_orca.run_speed_benchmark \
    --output-root ~/vibeqc-runs
```

Reports, vq workspaces, fetched raw outputs, and regenerated cache
copies default to `~/vibeqc-runs/<run-id>/`, with `--output-root` and
`VIBEQC_RUNS_DIR` following the main regression-suite precedence. The
committed `cache/*.json` fixture is read by default but is written only
with `--update-fixture-cache`. `PARITY_REPORT.md` in this directory is a
historical calibration snapshot, not the default report target.

compute-host-d is a shared family machine and the `vq` daemon is in a
`--max-jobs 1` phase — ORCA cells are submitted serial (`--cpus 1`).
Be sparing with `--run`.

## What ORCA actually exposes — and where the handover was wrong

Three things the original ORCA-parity handover (rev 2; retired) got wrong
about ORCA 6.1.1; the code is built around what ORCA *actually* does.

1. **No Coulomb / exchange split.** The handover's parser table
   expected `Coulomb Energy :` and `Exchange Energy :` lines. ORCA's
   standard `TOTAL SCF ENERGY` block prints neither — just a combined
   `Two Electron Energy :`. So the Fock-build comparison uses a single
   **`e_coulomb_plus_exchange`** bucket (ORCA `Two Electron Energy -
   E(XC)`; vibe-qc `e_coulomb + e_exchange`). Five energy buckets, not
   six — a failing bucket still localises to integrals / Fock build /
   XC quadrature; only the J-vs-K resolution is lost.

2. **Hybrid "final integration".** For a hybrid functional ORCA
   recomputes E_x on a finer grid at the end. `Nuc + 1e + 2e` is the
   **SCF-grid** total; `FINAL SINGLE POINT ENERGY` is that plus the
   finer-grid correction (~1e-5 Ha). vibe-qc decomposes on its own SCF
   grid with no final integration, so the parser returns the SCF-grid
   total as `e_total` (and keeps the post-integration value as
   `e_total_final`). The self-check accounts for the correction.

3. **The `vq` daemon does not prepend registered-program dirs to
   PATH.** A job's PATH is the bare `/usr/local/sbin:/usr/local/bin:
   /usr/bin`. `orca_vq.py` reads ORCA's path from `vq programs --json`
   and hands it to `run-orca.sh` via an explicit `ORCA_BIN` env var
   (the same pattern `tests/integration_smoke.py` uses for CRYSTAL).

Two further matched-settings decisions, made so the comparison
reflects the *implementations* rather than incidental setup
differences:

* **Geometry in Bohr.** ORCA reads the `*xyz` block with the `Bohrs`
  keyword, fed the exact Bohr coordinates vibe-qc uses (one shared
  Angstrom→Bohr conversion in `cases.py`). Without this, each code's
  own Angstrom→Bohr rounding shifts E_nuc by ~4e-8 Ha between the two
  codes.
* **`VeryTightSCF`.** Plain `TightSCF` (~1e-8 energy) stops ORCA with
  the density still only ~5e-5 converged — a ~1e-4 cross-code leak
  into E_1e / E_coulomb. `VeryTightSCF` brings ORCA's density close to
  vibe-qc's (`conv_tol_energy = 1e-12`).

## Coverage

`PARITY_CELLS` mirrors `test_parity_hf_dft.py`'s diagonal: H2O / H2CO /
OH· × {def2-svp, def2-tzvp} × {RHF, UHF, RKS-PBE, RKS-B3LYP,
RKS-PBE0, UKS-PBE} × {direct, density-fitted, RIJCOSX}. DF and
RIJCOSX cells are HF / hybrid only — ORCA's `RIJK` (and the cosx-K
piece of RIJCOSX) needs HF exchange (a pure-GGA DF cell would need
`RI` with the `def2/J` AuxJ slot, which the default regression suite
exercises via benzene/def2-SVP/RKS-PBE-DF). The pure-DFT cosx flag is a no-op
(`tests/test_rijcosx.py::test_rijcosx_pure_dft_flag_is_noop`), so a
pure-GGA RIJCOSX cell adds no coverage either.

**RIJCOSX cells: like-with-like across codes.** Both vibe-qc and
ORCA implement RIJCOSX (RI-J + seminumerical chain-of-spheres K,
Neese 2009), and vibe-qc's COSX was implemented to match ORCA
6.1.1 — so vibe-qc-RIJCOSX vs ORCA-RIJCOSX is a legitimate
cross-code cell (BRIEF.md § Scope-1 like-with-like doctrine). The
cells certify implementation consistency between two
independently-written codes of the same algorithm. PySCF doesn't
ship a turnkey RIJCOSX driver, so the RIJCOSX axis is **ORCA-only**
on the cross-code matrix (the PySCF axis stays at direct + DF).
`tests/test_rijcosx.py` continues to pin RIJCOSX's *absolute*
correctness against direct SCF (sub-mHa fit-error band) — the new
ORCA-axis cells add the cross-code certification on top.

The **speed-benchmark ladder** defaults to H2O + glycine (cheap,
< 5 min on a clean compute-host-d box). For probing the regime where the
ORCA gap actually lives, `cases.py` also defines an
`EXTENDED_SPEED_LADDER_SYSTEMS` set, opt-in via `--systems`:

| name | atoms | rationale |
|---|---:|---|
| `SF6` | 7 | small + row-3 (S, F), Oh |
| `CCl4` | 5 | small + row-3 (Cl), Td |
| `NiCO4` | 9 | metal carbonyl, Ni(0) d¹⁰, Td |
| `naphthalene` | 18 | ~20-atom flat aromatic (PAH), D₂ₕ |
| `ferrocene` | 21 | metal-organic complex, Fe(II) d⁶ LS, D₅ₕ |
| `n-decane` | 32 | ~30-atom all-trans alkane |
| `n-hexadecane` | 50 | 50-atom all-trans alkane |

All built from first principles (point-group symmetry + standard bond
lengths), all closed-shell. Run individually or in groups — be sparing
on compute-host-d; the upper end is hours per cell at
B3LYP/def2-TZVP/RIJCOSX. Example::

    python -m examples.regression.parity_matrix_orca.run_speed_benchmark \\
        --systems naphthalene,ferrocene --methods RKS-B3LYP --paths rijcosx

See the run directory's `PARITY_REPORT.md` for a fresh parity matrix.
The checked-in `PARITY_REPORT.md` and
[`benchmarks/orca_vs_vibeqc_speed.md`](../../../benchmarks/orca_vs_vibeqc_speed.md)
are historical snapshots used to explain the tolerance calibration; new
benchmark reports are written under the selected output root unless
`--report` points elsewhere.
