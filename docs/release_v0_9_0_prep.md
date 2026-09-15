---
orphan: true
---

# v0.9.0 release-day docs sprint, checklist + pre-staged copy

```{note}
This is a historical pre-split release checklist for the archived monorepo.
Its co-located component paths and old branch/tag instructions are not a
current setup recipe. Use [release process](release_process.md) and
[toolset lifecycle](toolset_lifecycle.md) for the independent repositories.
```

This page is the **release-day playbook for the v0.9.0 vibe-qc
release**. Copy-and-adapted from `docs/release_v0_8_0_prep.md` per the
per-tag mechanical-sprint convention in
[`docs/release_process.md`](release_process.md) § "Documentation
cadence". Marked `orphan: true` so it doesn't appear in the toctree.

## Scope, what v0.9.0 is, and what it is NOT

> **Read this before writing any user-facing copy.** The v0.8.0 cut
> shipped late corrections because the prep doc overstated periodic-GDF
> maturity (the release chat caught it in a pre-flight reconciliation
> hours before tag). v0.9.0 starts from an honest scope so that does
> not repeat.

**v0.9.0 codename: _Knowles's Kingfisher_** (release-chat recommendation
pending Mike's confirmation, Peter Knowles, the Knowles-Handy
determinant-driven Full-CI algorithm, 1984; the conceptual core of
vibe-qc's new FCI solver. Kingfisher: a precise, fast hunter that
catches its exact prey, exact diagonalisation. Alternatives floated:
_Handy's Heron_, _Shavitt's Salamander_.)

### v0.9.0 headline + included tracks

| Track | Status on `main` | Notes |
|---|---|---|
| **Wavefunction methods**, FCI / Selected-CI / DMRG / v2RDM | **Headline.** `python/vibeqc/solvers/` package; `method='fci'` via `run_job`; auto-method routing; numerical-gradient geometry opt; user-guide docs; `examples/wavefunction/`; parity test suites | The marquee v0.9.0 feature |
| Direct SCF (molecular RHF/UHF/RKS/UKS) | Production-ready (mol-direct audit complete) | |
| BIPOLE periodic driver, Γ-only | "v0.8.x complete"; RKS/UKS + UHF parity; DIIS validated on MgO/diamond/Si | Multi-k BIPOLE is in progress, Γ-only is the v0.9.0 surface |
| COSX, `OneCenterCorrection` | Complete (GridBatches-consistent K_cosx_1c) | |
| ωB97X (RSH machinery, milestone A) | Landed | ωB97X**-D** is deferred, see below |
| MP2 / RI-MP2 parity matrix vs ORCA | M1-M4 milestones landed | |

### Deferred OUT of v0.9.0, do NOT claim these in v0.9.0 copy

| Deferred item | Target | Why |
|---|---|---|
| **Periodic GDF** (compcell + multi-k + AFT) | **v0.10.0** | Not converging, LiH primitive FCC +22 Ha, MgO 8-atom ~+1500 Ha off PySCF. Deferred v0.8.0 → v0.9.0 → v0.10.0. See [`HANDOVER_GDF_V0_11_2026_05_29.md`](https://vibe-qc.com/docs/). |
| **ωB97X-D** (RSH milestone B) | v0.9.x | Blocked on a dispersion reference, ωB97X-D's D2/CHG dispersion has no out-of-process oracle (PySCF lists `wb97x-d` UNSUPPORTED_DISP; dftd3/dftd4 do D3/D4 only). CLAUDE.md § 8 forbids shipping a hand-transcribed table with no reference. |
| Native D4 dispersion | v0.10.0 | Design / milestone-plan phase only |
| Solvation analytic gradients | v0.9.1 | Energy-only CPCM/COSMO may be assessed for v0.9.0 inclusion; the analytic-gradient work is explicitly v0.9.1 per its handover |
| Coupled cluster (canonical CCSD(T)) | v0.14.0 | Just started, roadmap commit `9601697` places canonical CCSD(T) at v0.14.0. **Gates nothing before then.** |

The periodic surface v0.9.0 actually ships is the **Ewald-3D stack at
its v0.7.x maturity** plus **BIPOLE Γ-only**. v0.9.0 copy must not imply
periodic GDF parity.

## Pre-flight (before tagging)

Confirm:

1. **Dev-chat tag-ready signals.** Each contributing chat drops a
   `.release-status/v0.9.0/<chat-id>.md` per CLAUDE.md § 3 stating
   tag-ready Y/N. Gating siblings: the wavefunction-methods chat,
   mol-direct, bipole, cosx, xc/RSH, the QA chat. Do not tag without
   QA's tag-ready YES.

2. **`CHANGELOG.md` `[Unreleased]` block** is the v0.9.0 release-notes
   draft. On tag day:
   - Promote `[Unreleased]` → `## [v0.9.0], <YYYY-MM-DD>, 
     *Knowles's Kingfisher*, <one-line theme>`.
   - Replace `[Unreleased]` with the v0.9.x maintenance banner.
   - **Audit every periodic-GDF claim out**, the `[Unreleased]`
     block accumulated GDF entries that must move to a v0.10.0
     forward-looking note, exactly as the v0.8.0 cut had to strip
     them. Same for ωB97X-D.

3. **`docs/roadmap.md`**, flip v0.9.0 entries from "in flight" /
   "queued" to "shipped" / "✓". Anything not in the cut → move to a
   v0.9.x / v0.10.0 section. The GDF deferral admonition on the
   `v0.8.0 — periodic SCF chain` entry (landed 2026-05-20) already
   records the v0.10.0 target, leave it.

4. **`docs/index.md`** homepage admonitions, swap to the pre-staged
   v0.9.0 versions below.

5. **`pyproject.toml`** version bump to `0.9.0`.

6. **`CITATION.cff`** version + `date-released` bump.

7. **`docs/citing.md`** version-bump in the APA + BibTeX examples.

8. **README.md**, verify the headline-feature paragraph mentions the
   v0.9.0 wavefunction-methods deliverable.

9. **No banner coupled-fix this cycle.** The v0.8.0 libecpint banner
   coupled fix is done and shipped; v0.9.0 has no analogous
   native-dependency banner change. (If the wavefunction-methods work
   introduced a new vendored native dep, re-check CLAUDE.md § 6, but
   FCI/CI/DMRG/v2RDM are believed to be pure in-tree.)

## Release-branch reconciliation, the `merge -s ours` step

> **This is new vs the v0.8.0 prep doc, which discovered the problem
> live.** Pre-documenting so the v0.9.0 cut does not stall.

`release` and `main` diverged at `2b5531d` (v0.7.3 prep) and have NOT
re-converged. As of v0.8.3, `release` carries ~23 commits (the
v0.7.13 → v0.8.3 patch line) that are NOT on `main` as those SHAs.
Therefore `release` is **not an ancestor of `main`**, and
`git merge --ff-only v0.9.0` will fail unless the v0.9.0 tag commit
has the release line in its ancestry.

**Fix, same procedure used at the v0.8.0 cut** (commit `91ab737`):

```sh
# On a fresh branch off current origin/main:
git merge -s ours --no-ff origin/release -m "Merge v0.8.3 release line into main pre-v0.9.0 (history-only)"
# Then the release: v0.9.0 commit goes on top; tag it; release ffs cleanly.
```

`merge -s ours` records the ancestry (so `release`'s tip becomes an
ancestor of the v0.9.0 commit) **without importing release's
working-tree state**, which would conflict with the 200+ feature
commits `main` has since the divergence. Carry the `[v0.8.0]` …
`[v0.8.3]` CHANGELOG sections forward into `main`'s CHANGELOG in the
release commit (same as the v0.8.0 cut carried `[v0.7.4]`-`[v0.7.14]`).

## On tag day, order of operations

1. **Pre-flight + reconciliation.** Branch off current `origin/main`;
   run the `merge -s ours` step above; run `pytest tests/` (must pass).
2. **Version + metadata commit.** Bump `pyproject.toml`,
   `CITATION.cff`, `docs/citing.md`; carry the v0.8.x CHANGELOG
   sections forward; promote `[Unreleased]` → `[v0.9.0]`. Commit
   `release: v0.9.0`.
3. **Regenerate bundled reference outputs**, 
   `scripts/regenerate_doc_examples.py`. **NOTE**: at the v0.8.0 cut
   this script no-op'd because `scripts/doc_examples.toml` was stale
   (every input script missing). If still stale at v0.9.0 tag time,
   fix the manifest first (tracked task) or document the skip.
4. **Tag `v0.9.0`** annotated (`vibe-qc 0.9.0 — Knowles's Kingfisher`).
   Push the tag.
5. **Fast-forward `release`** to the tag; push `release` (fires the
   docs-deploy CI to vibe-qc.com).
6. **Post-release:** bump `main` to `0.9.1.dev0`.
7. Docs chat applies the homepage-admonition + landing-page flips
   below in one commit per surface.
8. Confirm `vibe-qc.com` rebuilds; verify each admonition renders.

## Pre-staged copy: homepage admonitions for v0.9.0

### Block 1, release-banner admonition

Auto-substituted via MyST `{{release}}` / `{{codename}}` from
`pyproject.toml` + the `RELEASE_CODENAMES` catalog in
`python/vibeqc/banner.py`. **No manual edit needed**, but the
`"0.9.0": "Knowles's Kingfisher"` catalog entry MUST be added to
`banner.py` in the same commit that bumps `pyproject.toml` (per the
catalog's own docstring). Verify the substitution resolves on rebuild.

### Block 2, replace the v0.8.0 mol-methods headline admonition

**Currently** (search `Grimme's Gecko` / the v0.8.0 headline block in
`docs/index.md`): the v0.8.0 molecular-methods headline.

**Replace with** (v0.9.0 headline):

> ✅ Wavefunction methods, exact and near-exact, in v0.9.0
>
> v0.9.0 (codename *Knowles's Kingfisher*) adds a wavefunction-methods
> suite under `vibeqc.solvers`: **Full Configuration Interaction**
> (`method='fci'`), **Selected-CI** (restricted + unrestricted),
> **DMRG**, and **variational 2-RDM** (v2RDM with Q/G constraints), 
> with auto-method routing and numerical-gradient geometry
> optimisation. FCI parity is validated against reference
> diagonalisation; see `examples/wavefunction/` and the
> `vibeqc.solvers` user-guide page.
>
> v0.9.0 also ships **production-grade molecular direct SCF**
> (RHF/UHF/RKS/UKS), the **BIPOLE periodic driver** (Γ-only,
> closed- and open-shell, DIIS-validated on MgO / diamond / Si),
> the **COSX one-centre correction**, **ωB97X** (range-separated
> hybrid machinery), and the **MP2 / RI-MP2 cross-code parity
> matrix vs ORCA**. See
> [CHANGELOG](https://vibe-qc.com/docs/)
> for the full notes.

### Block 3, replace the v0.8.x "Open in the v0.8.x maintenance window" warning admonition

Refresh the open-issues warning to the v0.9.x window. **Carry
forward** the still-open carryovers; **add** anything new; **drop**
what closed in v0.9.0.

* **Closes at v0.9.0** (drop): DF analytic-gradient mis-attribution
  (fixed v0.8.3, `4e59a4d`); the homepage-admonition staleness
  (fixed v0.8.3, `bdc3102`).
* **Carries v0.8.x → v0.9.x** (keep): periodic GDF not at parity
  (now explicitly v0.10.0); analytic-gradient-with-f-shells; B3LYP
  behaviour-change informational; the `scripts/doc_examples.toml`
  stale manifest.
* **New at v0.9.0** (add): ωB97X-D deferred (dispersion-reference
  blocker); solvation analytic gradients land in v0.9.1; multi-k
  BIPOLE still in progress.

### Block 4, funding admonition

Unchanged from v0.8.0 unless the sponsor surface changed. Verify URLs.

## Pre-staged copy: CHANGELOG `[Unreleased]` reset

After promoting the current `[Unreleased]` content to `## [v0.9.0]`,
replace `[Unreleased]` with:

```markdown
## [Unreleased]

> Heading toward v0.9.x maintenance + v0.10.0. v0.9.x will close:
> ωB97X-D (pending a D2/CHG dispersion reference); solvation
> analytic gradients (v0.9.1); multi-k BIPOLE. v0.10.0 is reserved
> for the periodic-GDF chain (compcell + multi-k + AFT — deferred
> from v0.8.0 → v0.9.0 → v0.10.0 because it does not yet reach
> PySCF parity) and native D4 dispersion. Coupled cluster
> (canonical CCSD(T)) is roadmapped for v0.14.0.
```

## Pre-staged copy: roadmap.md "shipped" sweep

For each v0.9.0 entry in `docs/roadmap.md`:

1. Add a leading `**Status: shipped in v0.9.0 (<YYYY-MM-DD>) as part
   of *Knowles's Kingfisher*.**` line.
2. Move slipped sub-items to a v0.9.x / v0.10.0 section.
3. The `v0.8.0 — periodic SCF chain` entry already carries the
   2026-05-20 GDF-deferral admonition, leave it; just confirm it
   still reads correctly post-v0.9.0.

## What's NOT in this checklist

- **Engineering tasks** (merging feature branches, the
  wavefunction-methods final validation, BIPOLE multi-k): owned by
  the respective dev chats.
- **CI verification** (docs deploy, banner render): verify by
  visiting https://vibe-qc.com/ after the deploy.
- **Periodic GDF**, explicitly v0.10.0; not part of v0.9.0 at all.
- **Coupled cluster**, v0.14.0; do not mention as imminent.

## Estimated wall-clock for the docs sprint

Assuming engineering has merged + the reconciliation is done:

- `merge -s ours` reconciliation + CHANGELOG carry-forward: ~20 min
- CHANGELOG promotion + `[Unreleased]` reset: ~10 min
- Homepage admonitions swap (Blocks 2-3): ~15 min
- Landing-page rework (intro / capabilities / status): ~25 min
- Roadmap "shipped" sweep: ~30 min
- `pyproject.toml` + `CITATION.cff` + `docs/citing.md` + `banner.py`
  codename: ~10 min
- README.md headline refresh: ~10 min
- Regenerate doc examples (if manifest fixed): ~15 min
- Verify deploy + cross-link audit: ~30 min

**Total: ~2.5-3 hours** of focused release/docs work on tag day.

## After the dust settles

Queue for v0.9.x maintenance:

- The v0.8.0 post-tag docs sprint was never run, its homepage
  admonition + roadmap sweep items are folded into this v0.9.0 sprint.
- `scripts/doc_examples.toml` manifest fix (stale since before v0.8.0).
- ωB97X-D once a dispersion reference exists.
- The standing `merge -s ours` reconciliation will be needed again at
  every minor while `release` carries a patch line, consider whether
  a periodic main↔release true-merge is worth doing once to reset.
