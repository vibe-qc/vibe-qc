# Real-Γ direct-torus studies

Validation / cost-probe payloads for the neutral fitted-torus real-Γ route
(`vibeqc.periodic.ccm.direct`, `jk_method='real-gamma'`). Owner: the real-Γ
AICCM development chat; workstream tracker
[`handovers/HANDOVER_AICCM_DIRECT_TORUS.md`](../../handovers/HANDOVER_AICCM_DIRECT_TORUS.md).

* `aic7.py` — canonical source for the rung-isolated AIC7 wave payload. The
  driver supports convergence, Madelung, parity, pathology, iteration-cost,
  and direct-route diagnostic plans. Comma-separated CLI method lists and
  their plus-separated tag form are equivalent. Ladder targets run in
  descending mesh order so the global budget does not systematically strand
  the requested high-mesh points. A result is `complete` with exit code 0 only
  when every planned rung is terminal `ok`; otherwise it is `partial` with a
  nonzero exit and explicit per-rung budget/timeout provenance. Submit future
  copies from this tracked source. Fetched historical payloads are sealed run
  evidence and must not be edited in place.

* The machine-readable issue-#415 adjudication lives with immutable campaign
  evidence at
  `agentic-loop/runs/2026-08-28-aic7-status-backfill/manifest.json`; its reader
  is `agentic-loop/artifact_status_adjudication.py`. The manifest binds each
  raw `results.json`, affected rung file, and external OOM receipt by SHA-256;
  the reader produces effective `partial` top-level and per-rung records
  without changing those source bytes. The canonical campaign-verdict recorder
  invokes this reader and refuses a `complete` artifact gate for a listed job.
  Other judges must likewise read the effective root `runs` array, or request
  one effective rung with `--rung-id`; they must not consume the listed raw
  `status: complete` or `status: running` fields directly. For example:

  ```sh
  python agentic-loop/artifact_status_adjudication.py \
      <vq-job-id> <artifact-root> [--rung-id <rung-id>]
  ```

* `eff_ladder_v2.py` — efficiency ladder v2 (2026-08-21). Re-measures the
  July ladder's five rungs with the occupied-rank factored exchange live
  (main `2effb3ad7`) and adds adsorption-class cost probes (molecule on
  3x3 / 4x4 in-plane slab supercells with real vacuum). Per rung: fold
  time (symmetry=True), SCF time + iterations, E/cell, exchange_q0,
  factored-vs-dense K timing and their max difference, L shape/GB,
  occupied rank, peak RSS (unit-corrected). Writes `eff_ladder_v2.json`
  to `$VQ_WORKDIR`.

  Run on vq per CLAUDE.md § 15 (one node, full cores, generous walltime;
  **compute-small requires explicit `--mem-mb`/`--cpus`** or the job is
  cgroup-capped at 4 GB). The B rungs are cost/feasibility probes, not
  chemistry — their energies are not adsorption energies (no BSSE, no
  relaxation, minimal basis).
