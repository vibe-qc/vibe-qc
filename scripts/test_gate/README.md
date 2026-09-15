# vibe-qc release-tier test gate

The machine-readable policy is
[`suite_manifest.json`](suite_manifest.json). Every `tests/**/test_*.py` file
has exactly one method maturity and one gate tier:

| Tier | Role | Verdict | Default concurrency | Budget |
|---|---|---|---|---|
| T0 | build-break | blocks every push | 0 pytest files | under 2 minutes |
| T1 | pre-cut sentinels | blocks the release cut | 2 | at most 15 minutes |
| T2 | post-cut inventory | cannot block the cut | 1 | unbounded |
| T3 | advisory/research | never blocks | 1 | unbounded |

The maturity values are `production`, `verified`, `under-review`, and
`experimental`. Experimental files must be T3. T0/T1 files must have a named
owner. Pytest applies `tier0` through `tier3` and
`maturity_production`, `maturity_verified`, `maturity_under_review`, or
`maturity_experimental` during collection from this manifest, so ordinary
selection is mechanical:

```sh
$PY -m pytest -m tier0
$PY -m pytest -m 'tier1 and maturity_production'
```

For isolated release-sentinel execution use the tier selector. T0 is the
native compile/import check and therefore has no pytest artifact:

```sh
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --tier T1 --out t1.jsonl
$PY scripts/test_gate/gate_verdict.py t1.jsonl known_reds_baseline.json \
    --reverify --wt "$WT" --py "$PY"
```

`gate_verdict.py` accepts only classified T0/T1 records. It returns usage
error 2 before verdict calculation if an artifact contains T2, T3, or legacy
unclassified records. T3 red therefore cannot redden a cut, even if someone
accidentally supplies its artifact to the blocking command.

The older named lanes remain available for development-area selection. They
are compatibility views, not the release-cut source of truth; the release-cut
role consumes `suite_manifest.json` and may not reinterpret it.

The build and `import vibeqc` command form T0; no pytest file runs on ordinary
pushes. The static cut gate adds the consolidated production sentinel in T1,
plus any file its owning chat has pinned to T1 in `suite_manifest.json`.
Native-binding and gate-framework tests are T2 and run when those areas change,
not on every unrelated push. Patch-affected route evidence is
produced by the owning change lane and reused only when its exact SHA and
impact-clearance key remain valid. The release cut does not rerun broad parity,
output, basis, dispatch, or method matrices merely because they exist.

A repeatable, **executable** full-suite gate that runs every test to completion
and turns a clear green/red verdict — so regressions can no longer hide on
`main`.

## Why this exists

Native CI (`.gitlab-ci.yml` on the 32 GB `build-runner` runner) executes no
regression pytest on ordinary `main` pushes: compilation plus import is T0.
Release-candidate branches additionally execute the sole T1 file. The tag
reuses the green exact-SHA candidate evidence and does not run CI again.
CI never collects the monolithic suite.
Broad lane/full-suite correctness remains an isolated local or nightly gate.

## The key insight — the OOM is cumulative, not per-test

pytest holds every test's allocations in one long-lived process, so peak RSS is
the *sum* of leaks, not the heaviest test. **Per-file process isolation** (each
file in its own `pytest` process, releasing on exit) bounds peak RSS to ~the
single heaviest file. Measured: the bulk of files peak **< 1 GB**; a handful
hit 1–6 GB; exactly one (`test_direct_scf_smoke`, the n⁴ memory-wall boundary)
needs ~49 GB. So an isolated suite fits even on 32 GB — the fix is *isolation*,
not a bigger box. (See the memory-tier table below for routing.)

## Components

* **`run_full_suite.py`** — per-target isolation runner. Spawns `pytest <target>`
  per test file or explicit pytest node id, polls peak RSS of the process tree
  (psutil), enforces a per-test timeout (`pytest-timeout`) *and* a hard
  per-target wall-clock kill
  (the real backstop — pytest-timeout's thread method cannot interrupt a
  GIL-holding C++ call), classifies each target
  (`PASS`/`FAIL`/`SEGFAULT`/`OOM_KILLED`/`SIGKILLED`/`TIMEOUT`/`COLLECT_ERR`/…;
  a SIGKILL is `OOM_KILLED` only when the sampled peak RSS reached
  `--oom-fraction` of physical memory, otherwise `SIGKILLED` with the cause
  unassigned and the memory numbers stored on the row), and writes
  a **resumable** `triage.jsonl` (re-running skips targets already recorded).
  Lanes via `--lane <name>` from `lane_manifest.json`, comma-separated lane
  lists via `--lane lane-a,lane-b`, explicit tiers via `--tier T0,T1`, release profiles via
  `--profile release-core --profile-mode blocking|advisory|all`, impact
  selection via `--changed-since <ref> --impact-mode dev|release`, or raw
  `--markexpr` (`"not slow"` / `"slow"`); `--heavy-env` sets
  `VIBEQC_RUN_HEAVY_TESTS=1` for the env-gated heavy tests.
* **`gate_verdict.py`** — accepts blocking-tier artifacts only and compares a `triage.jsonl` against
  `known_reds_baseline.json`. **Green iff every non-PASS file is a tracked,
  owned bug in the baseline**; any *new* non-PASS file → RED (exit 1). Also
  flags baseline entries that now PASS (remove them). With `--reverify` it
  re-runs each NEW (unlisted) non-PASS file once in isolation before reding it,
  so a contention artifact on a loaded box can't fake a red (see
  [Contention re-verify](#contention-re-verify) below).
* **`known_reds_baseline.json`** — the tracked-known reds (owner + ref +
  reason). A red here does not gate; it is someone's tracked bug.
* **`suite_manifest.json`** — exhaustive per-file maturity, tier, owner,
  memory tier, measured runtime, disposition, rationale, and verification SHA.
  Regenerate classifications and fold in fresh jsonl measurements with
  `update_suite_manifest.py --measurements RUN.jsonl --verified-sha SHA`. Pass
  the commit actually tested when importing a long-running or resumed artifact;
  otherwise the updater uses the current HEAD.

  Regeneration rebuilds **every** row, so a classification the updater cannot
  derive from `lane_manifest.json` must be recorded in that script's `CURATED`
  table or the next chat to add a test file will silently overwrite it. Pin only
  the fields that are genuinely hand-chosen — a tier, an owner, a written
  rationale — and leave the rest derived. `tests/test_test_gate_lanes.py`
  asserts that a fresh run reproduces the checked-in file exactly, so an
  unrecorded hand-edit fails a test rather than surviving until someone reads a
  ~700-row JSON diff. Pinning a `tier` of T0 or T1 puts the file in the blocking
  release gate on its own, because the cut selects with `--tier`, not by lane;
  do that only as the file's owning chat.

  That guard turns red for an unrecorded hand-edit, but it goes green again
  either way — by registering the row *or* by regenerating over it. So the
  updater also refuses to write a row it would weaken relative to the committed
  manifest: a tier demotion, a maturity downgrade, or a hand-written rationale
  reverting to boilerplate. It names the affected rows and exits non-zero
  without touching the file. Register them in `CURATED`; if a demotion is
  genuinely intended, pass `--allow-demotion <path>` once per file and say why
  in the commit message.
* **`lane_manifest.json`** — the partition map. It names the smoke, molecular,
  PBC, GDF, BIPOLE, GAPW, semiempirical, external-reference, full-fast, and
  slow-nightly lanes, with default timeouts, owner, cadence, file globs,
  optional explicit pytest node ids, trigger paths, `method_maturity`,
  `lane_class`, linked global bug/gate items, target release, and the full
  matched calculation required before a scientific value can be promoted.

## Partition policy

The old split was only `not slow` versus `slow`. That is too blunt for a codebase
where a molecular-output change should not pay for AICCM real-Gamma/Gamma/chi
research tests, and a molecular or molecular-semiempirical change should not
run the whole PBC stack.

Default developer rule:

1. Run `smoke` when native bindings or gate policy change.
2. Run the lane for every code area touched.
3. For release or patch candidates, run the single T1 file after the T0
   compile/import check.
   The former
   `release-core` profile and slim impact selection remain development
   compatibility views and are not authoritative cut inputs.
4. Run advisory release-profile lanes separately only for nonblocking status
   on non-AICCM areas. AICCM/CCM/SECCM are experimental manual research lanes
   and are intentionally absent from release advisory profiles.
5. Run `full-fast` only for shared infrastructure, cross-cutting numerical
   primitives, final pre-tag confidence, or when no recent nightly covers the
   tree.
6. Run `slow-nightly` and `external-reference` before release tags or when those
   areas are touched.
7. Do not use AICCM real-Gamma/Gamma/chi, CCM, MSINDO-CCM, or SECCM as release,
   advisory-profile, or automatic impact tests. They are explicit manual
   research checks only.

List available lanes:

```sh
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --list-lanes
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --list-profiles
```

Preview the targets a lane will run:

```sh
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --lane pbc-gdf --dry-run
```

Preview the lanes and targets affected by a release or patch candidate:

```sh
BASE=v0.15.41
$PY scripts/test_gate/run_full_suite.py --wt "$WT" \
    --changed-since "$BASE" --impact-mode release --list-affected-lanes
$PY scripts/test_gate/run_full_suite.py --wt "$WT" \
    --changed-since "$BASE" --impact-mode release --dry-run
```

The default `--impact-mode dev` is broader and selects full area lanes for
touched code. Use it while developing, or when you deliberately want the full
calculation lane. Release mode is deliberately slim: it keeps only explicitly
opted-in cheap release-impact lanes blocking, and leaves
calculation-heavy scientific lanes to the responsible dev chat, advisory
status, nightly runs, or full published calculation checks.

The manifest has three lane classes. `pre-cut blocking` is the default
one-hour patch critical path. `post-cut evaluation` is on-touch, advisory,
nightly, final-confidence, or immutable-release evaluation work. The
`implementing-chat-only` class is experimental work owned by the implementing
chat. Every lane also declares one maturity (`production`, `verified`,
`under-review`, or `experimental`) and `scientific_acceptance: false`: pytest
and CI are change-safety evidence only. Method or paper acceptance still
requires the full production-shaped matched-reference calculation named by
`required_full_calculation`.

Run a focused lane:

```sh
# Run smoke when native bindings or gate policy changed.
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --lane smoke --out smoke.jsonl

# Then run the touched area, for example molecular SCF/DFT only.
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --lane molecular-scf-dft --out molecular-scf-dft.jsonl

# Periodic GDF changes run this lane, not every PBC/GAPW test.
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --lane pbc-gdf --out pbc-gdf.jsonl

# Molecular semiempirical work does not run periodic or experimental bridge tests.
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --lane semiempirical-molecular --out semiempirical-molecular.jsonl

# Periodic semiempirical work has its own lane.
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --lane semiempirical-periodic --out semiempirical-periodic.jsonl

# Release or patch candidate: compile/import supplies T0, then run T1.
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --tier T1 --out t1.jsonl
$PY scripts/test_gate/gate_verdict.py t1.jsonl \
    scripts/test_gate/known_reds_baseline.json --reverify --wt "$WT" --py "$PY" \
    --markexpr "tier1"

# Advisory nonblocking status for non-AICCM lanes. AICCM/CCM/SECCM are not part
# of this profile.
$PY scripts/test_gate/run_full_suite.py --wt "$WT" --py "$PY" \
    --profile release-core --profile-mode advisory --out release-advisory.jsonl
```

Useful lane names:

| Lane | When to run |
|------|-------------|
| `smoke` | Native-binding or gate-policy changes. |
| `release-core-cheap` | Blocking release profile lane for cheap shipped, non-experimental feature coverage. Uses selected pytest node ids for otherwise heavy files. |
| `output-docs` | `vibeqc.output`, citations, QVF, docs-symbol, manifest, or golden output work. |
| `molecular-scf-dft` | Molecular HF/DFT, grids, SCF, molecular gradients, Hessians, properties, dispersion. |
| `molecular-correlation` | MP2, CC, DLPNO, CAS, CI, TDDFT, and solver work. |
| `basis-ecp-integrals` | Basis, ECP, integral, DF core, and basissetdev architecture work. |
| `semiempirical-molecular` | Molecular MSINDO, PM6/OMx, GFN2/xTB, DFTB0, runner/output, PES, MD, NEB, and route-plan work. |
| `semiempirical-periodic` | Periodic and k-point semiempirical adapters, DFTB smearing/convergence/FD, and native periodic route checks. |
| `semiempirical` | Aggregate semiempirical lane for broad refactors. Prefer the narrower sublanes. |
| `pbc-core` | Periodic core excluding GDF, BIPOLE, GAPW/GPW, and embedding. |
| `pbc-gdf` | GDF/RSGDF/MDF/compcell/AO-pair-FT changes. |
| `pbc-bipole` | BIPOLE, screened exchange, pair-resolved SR domains, and BIPOLE-backed periodic NEB. |
| `pbc-gapw-gpw` | GAPW/GPW, plane-wave grids, PW reference, and FFT-Poisson. |
| `embedding-surfaces-neb` | Periodic embedding, slabs, surface reactions, ASE periodic, and NEB. |
| `external-reference` | ORCA/CRYSTAL/CP2K/GPAW/PySCF parity boundaries and `examples/regression` runner tests. |
| `full-fast` | Non-slow shipped-feature pytest files, excluding experimental AICCM/CCM/SECCM. |
| `slow-nightly` | Slow/heavy shipped-feature tests on a large box, excluding experimental AICCM/CCM/SECCM. |

## Running the gate

Needs a **built** vibe-qc (fresh `_vibeqc_core.so`) + a venv with `pytest`,
`pytest-timeout`, `psutil`, `pyscf`, and the in-repo **`vibe-view`** package
(QVF round-trip tests import `vibeview`). Build per the
`release-gate-build-recipe` (throwaway venv + worktree, reuse prebuilt
`third_party`); `pip install -e ./vibe-view` for the viewer.

```sh
PY=/path/to/gate-venv/bin/python      # venv with a fresh-built vibeqc
WT=/path/to/checkout-or-worktree      # the source under test

# Fast lane for shipped features (everything except @slow and experimental
# AICCM/CCM/SECCM) — the frequent gate, minutes:
$PY run_full_suite.py --wt "$WT" --py "$PY" --markexpr "not slow" \
    --jobs 6 --test-timeout 300 --file-timeout 600 --out fast.jsonl

# Manifest-backed shipped-feature fast lane:
$PY run_full_suite.py --wt "$WT" --py "$PY" --lane full-fast --out fast.jsonl

# Blocking release run: only T0/T1 may feed the verdict.
$PY run_full_suite.py --wt "$WT" --py "$PY" \
    --tier T0 --out t0.jsonl
$PY run_full_suite.py --wt "$WT" --py "$PY" \
    --tier T1 --out t1.jsonl

$PY gate_verdict.py t0.jsonl known_reds_baseline.json \
    --reverify --wt "$WT" --py "$PY" --markexpr "tier0"
$PY gate_verdict.py t1.jsonl known_reds_baseline.json \
    --reverify --wt "$WT" --py "$PY" --markexpr "tier1"

# Advisory profile: nonblocking status for non-AICCM lanes. Save it and
# summarize it, but do not pass it to the blocking verdict.
$PY run_full_suite.py --wt "$WT" --py "$PY" \
    --profile release-core --profile-mode advisory --out release-advisory.jsonl

# Slow/heavy shipped-feature lane (@slow tests; the heavy/parity SCFs) —
# nightly/pre-tag, on the 128 GB box, low concurrency:
$PY run_full_suite.py --wt "$WT" --py "$PY" --markexpr "slow" --heavy-env \
    --jobs 2 --test-timeout 1800 --file-timeout 2400 --out slow.jsonl

# Manifest-backed shipped-feature slow lane:
$PY run_full_suite.py --wt "$WT" --py "$PY" --lane slow-nightly --out slow.jsonl

# These broad/slow artifacts are inventory evidence. Summarize or retain them,
# but never pass them to gate_verdict.py; it rejects their T2 records.
```

### Contention re-verify

(`gate_verdict.py --reverify`)


The historical fast lane ran many files concurrently (`--jobs 6`). On a shared box under
heavy load a perfectly healthy file can be starved into a
`FAIL`/`ABORT`/`TIMEOUT`/`OOM_KILLED`/`SIGKILLED` that it does **not** reproduce when run
alone — a *contention artifact*, not a code regression. The v0.13.0 cut hit
exactly this: a fast-lane run at `--jobs 6` on a box at load ~80 red-flagged ~8
slow/memory-heavy files, every one of which **passed** when re-run in isolation
at `--jobs 1`.

T1 now defaults to two workers, sized for its documented light/medium memory
tiers. `--reverify` remains a final safety net. Before a NEW (unlisted) non-PASS
file is allowed to gate RED, `gate_verdict.py` re-runs it **once on its own**
(`--jobs 1`, a generous timeout — `--reverify-file-timeout`, default 1800s — so
the re-run itself can't false-TIMEOUT). It only stays red if it fails the
second time too; a file that recovers is reported as a contention artifact and
does not gate. It is tier-aware by construction because the verdict rejects
T2/T3 input before re-verification; only a T0/T1 candidate can pay this cost.
Pass `--markexpr` matching the blocking tier being verdicted.

A genuine regression still fails the isolated re-run and gates RED — re-verify
clears contention noise, it does not mask real bugs. The partition logic is
unit-tested in [`tests/test_gate_verdict_reverify.py`](../../tests/test_gate_verdict_reverify.py).

## Lane health — the periodic report (not a gate)

(`lane_health.py`)

The gate above answers "may this cut ship?". It cannot answer "what is red on
`main` right now?", because it deliberately runs T0+T1 only and
`gate_verdict.py` rejects T2/T3 input outright. That leaves ~90% of the
inventory with **no periodic signal at all**: a commit that reddens a T2/T3
lane produces nothing at push time, nothing at release time, and nothing ever,
until someone runs that lane for an unrelated reason. GitLab issue #450
collected six lanes found exactly that way in two days — one of them a
default-path SCF convergence regression that shipped in three releases.

`lane_health.py` is the missing periodic report. It consumes the artifacts the
runner already produces and publishes what is red:

```sh
# Whole inventory, per-file isolation, every tier:
$PY run_full_suite.py --wt "$WT" --py "$PY" --tier T0,T1,T2,T3 --out health.jsonl

# Report it. Feed the previous run's JSON back in so "last passed" tightens.
$PY lane_health.py health.jsonl --wt "$WT" \
    --previous lane_health.json \
    --out LANE_HEALTH.md --json-out lane_health.json
```

Per red file it reports the status, tier, owner, whether it is a tracked
known-red or a **new** one, the SHA it last passed at, and the commit range
since — including the commits in that range that touched the file. The
last-passed anchor comes from the previous report when one is supplied and
from `suite_manifest.json`'s `last_verified_sha` otherwise.

It also reports what it did **not** run. Any inventory file with no record in
the supplied artifacts is listed under *Not observed*: unknown, not green. A
partial run that read as a clean bill of health would recreate the blind spot
the report exists to remove.

**It is not a gate and must not be wired into one.** It accepts every tier
(the mirror image of `gate_verdict.py`, which refuses everything but T0/T1)
and always exits 0 on a successful report, however red the tree is. Use
`gate_verdict.py` for verdicts. Published text is redacted of local absolute
paths so the report can be committed or pasted (CLAUDE.md § 12). The logic is
unit-tested in
[`tests/test_test_gate_lane_health.py`](../../tests/test_test_gate_lane_health.py).

Running it on a timer is still an open ask to the queue chat — the same
host/timer the nightly cadence below waits on.

## Running it via vq (detached; survives long runs)

`submit_gate.sh <host>` submits the lanes to a vq host and points the verdict
at the workdir. On **localhost** (this Mac, 128 GB, manual daemon
`~/bin/vq-daemon-local.sh`) it runs against a local gate venv + worktree.

**Fleet status 2026-06-10:** the distributed (compute-study/compute-small/compute-medium-fanned) gate is
blocked on fleet repair - compute-small's managed venv can't import vibeqc
(`libint2.so`), compute-study is commits-stale, compute-medium's configured python path is
missing, and none have `pytest-forked`/`pytest-timeout`. See
`handovers/HANDOVER_TEST_HEALTH.md` § fleet for the asks to the queue chat. Until then the
gate runs on localhost.

## Memory-tier table (for shard routing)

From the 2026-06-10 full-suite per-file profile (368/369 on the 128 GB box):

| Tier | Peak RSS | Count | Route to | Files |
|------|----------|-------|----------|-------|
| light | < 1 GB | 324 | anywhere (incl. 32 GB) | the bulk |
| medium | 1–6 GB | 39 | ≥ 64 GB (or 32 GB at `--jobs 1`) | `test_pbc_gdf_compcell` 5.8G, `test_ewald_composed` 5.7G, `test_basis_filter` 5.3G, `test_periodic_rhf_ewald_diis` 5.2G, `test_parity_hf_dft` 5.1G, `test_parity_vs_orca` 5.1G, `test_periodic_{u,r}ks_multi_k_ewald` 4.2–4.3G, … |
| heavy | 6–32 GB | 4 | ≥ 64 GB | `test_periodic_rijcosx` 14.5G, `test_periodic_gradient_g1c` 7.7G, `test_periodic_gradient_g1d` 7.5G, `test_periodic_rhf_multi_k_ewald` 6.6G |
| **XL** | **> 32 GB** | **2** | **128 GB only** | `test_direct_scf_smoke` **49.2G** (n⁴ DIRECT boundary), `test_pw1pw` **48.7G** |

Routing rule: `@mem_high` (XL) → the 128 GB box at `--jobs 1`; everything else
→ any ≥ 64 GB box. With per-file isolation even the medium tier fits 32 GB at
`--jobs 1`. NOTE: the two XL files' ~49 GB in nominally O(n²) DIRECT/PW1PW paths
is high — flagged as a possible memory-efficiency follow-up (not OOM on 128 GB).

## Cadence

* **Patch or release-candidate impact gate**: run the combined
  `release-core` profile plus `--changed-since <base> --impact-mode release`
  command and verdict it first. This catches cheap contract regressions without
  paying for calculation-heavy molecular, PBC, GAPW, or external-reference
  work. AICCM/CCM/SECCM are not part of release or advisory gates.
* **Pre-tag final confidence**: run shipped-feature `full-fast` plus
  `slow-nightly`, or cite a recent green nightly for unchanged slow/heavy
  shipped-feature lanes; verdict must be green.
* **Nightly** (once a host/timer is wired by the queue chat): both lanes; a new
  red pings the owning chat.
* **Lane health** (nightly or every few days, same host/timer): the whole
  inventory at `--tier T0,T1,T2,T3`, reported through `lane_health.py`. This is
  the only periodic signal T2/T3 has; it reports, it never blocks.
* When a baseline entry starts PASSing, `gate_verdict.py` flags it — remove it
  from `known_reds_baseline.json` in the same change that confirms the fix.
