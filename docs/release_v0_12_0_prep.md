# v0.12.0, *Knuth's Beaver*, cut record (2026-06-12)

```{note}
This is a historical pre-split release checklist for the archived monorepo.
Its co-located component paths and old branch/tag instructions are not a
current setup recipe. Use [release process](release_process.md) and
[toolset lifecycle](toolset_lifecycle.md) for the independent repositories.
```

The release-chat record of the v0.12.0 cut: what gated it, what was
verified, what shipped with a known issue, and the follow-ups. Written
post-tag per the documentation cadence
([release_process](release_process.md#documentation-cadence)); also the
copy-and-adapt playbook for the next minor cut.

## Anatomy

| Step | Commit |
|---|---|
| Cut verification (gauge-refusal gates, re-pins, baseline 4→1) | `6af56d91` |
| Release commit (`release: v0.12.0 — Knuth's Beaver`; pyproject + CHANGELOG promotion + CITATION.cff) | `055cb2d9` |
| Reference-output regeneration (first since v0.5.0) | `d071593b` |
| **Tag `v0.12.0`** | → `d071593b` |
| Absorb merge (`-s ours` over the v0.11.x hotfix ancestry; tree == tag tree) | `d6b399e6` |
| `release` advanced (ff) | `bd779d6b` → `d6b399e6` |
| Post-release bump to `0.12.1.dev0` | `223650fb` |

The commit-msg version guard (`.githooks/commit-msg`) validated
version==tag at the release commit, armed via
`git config core.hooksPath .githooks` on the cutting clone (it is OFF by
default on fresh clones; arm it before every cut).

## Cut verification (fresh core, release-gate venv)

Run on a fresh `_vibeqc_core` built at the freeze tip in a throwaway
venv (`release-gate-build-recipe`); the shared `.venv` `.so` was stale
again (selected-CI/xc commits post-rebuild).

* **known_reds 4 → 1.** Three dispositioned at the cut:
  - `test_bipole_gradient` / `test_dft_plus_u`: the Phase-4a/4b Γ
    gauge default-flip post-dated the 2026-06-10 triage; the analytic
    BIPOLE UHF gradient silently disagreed with the driver FD by
    5.3 Ha/bohr. Fixed by completing the corrected-gauge **refusal**
    across UHF/RKS/UKS (mirroring RHF) + pinning every Γ
    analytic-vs-FD/CPHF test to the legacy gauge on BOTH the SCF and
    the FD reference (`6af56d91`).
  - `test_eigs_preflight`: re-pinned to the physical value, the
    `86c39851` gauge restoration recovers the isolated-molecule limit
    (−1.1167 Ha; `run_rhf` cross-check −1.1167143).
* **Shipped known issue (the 1 remaining tracked red)**: the analytic
  EWALD Γ gradient (`compute_gradient_periodic_rhf_gamma`, including
  ASE periodic forces on the EWALD route) differentiates the
  pre-restoration assembly, 0.1927 Ha/bohr vs the agreeing driver-FD
  + molecular-limit 0.1486 on a dilute H₂ box. FD periodic gradients
  are correct. Documented in the `[v0.12.0]` CHANGELOG; the gauge
  migration of the analytic kernel is the periodic-SCF chat's next
  milestone.
* **Restoration proofs green on the fresh build**:
  `test_audit_20260531_gauge_consistency` +
  `test_audit_20260530_periodic` 12/12 (the b4a6faba gauge cluster
  restoration `86c39851` + multi-k GDF compcell completion `3f1e8d89`).
* Also green: D4 (66+2xf incl. the strict-xfail native-parity
  un-gating bar), CPCM 34/34 (post-`38dfffa2` factor-2 fix, FD-exact),
  B3LYP convention 11, compiled-submodule registration, release
  version guard, dft_plus_u 72/72.

## In flight at tag time

* **compute-small `c70866f8d904`** (*bipole-phase4b-validation*, the round of
  record for `66ac0252`) was still running. The identical suites
  passed locally; if the deployment round reds, use the
  `Patch-candidate: v0.12.x` trailer flow.
* **BIPOLE Phase 3 (multi-k corrected gauge, `6aa34bc2` + `5d6d4a17`)
  landed ~4 h after the freeze line**, deliberately NOT in v0.12.0
  (un-verified by the cut gate; its own Phase-5 re-pin sweep pending).
  Rides the next release.

## Lessons → follow-ups

1. **Reference-output regen had been skipped since v0.5.0**, every
   release through v0.11.4 shipped `Release v0.5.0 "Wilson's Otter"`
   banners in the bundled examples. v0.12.0 regenerated all of them
   (`d071593b`).
2. **`regenerate_doc_examples.py` traps** (both hit at this cut):
   (a) run from a worktree, its interpreter fallback silently picks
   the MAIN repo's `.venv` → stale-SHA dirty banners, always pass
   `--python <cut-venv>`; (b) the editable install's reported version
   is baked at pip-install time, re-`pip install -e` after the
   version bump or the banners read `.dev0`. Script-hardening chip
   filed (assert runtime version == pyproject before running jobs).
   *Landed post-cut*: the script now runs a version preflight that
   refuses any job interpreter whose `vibeqc.__version__` ≠
   pyproject `[project] version` (regression-tested in
   `tests/test_regenerate_doc_examples_preflight.py`; flow in
   [release_process.md](release_process.md) step 3).
3. **Banner "Release vX.Y.Z" rendering requires exact-tag-at-HEAD**
   (`build_info.is_release`), which is structurally impossible for
   outputs committed before the tag, v0.12.0's bundle reads
   `dev 0.12.0 "Knuth's Beaver" (HEAD @ 055cb2d)`. Acceptable;
   a regen-script release-mode (banner override or post-tag regen
   flow) is part of the same chip. *Landed post-cut*: the banner
   override won, `--release-banner` runs jobs with
   `VIBEQC_BANNER_FORCE_RELEASE=1` (rendering-only; `.system`
   manifests keep truthful `is_release`/SHA provenance). The
   post-tag-regen alternative was rejected because `release` ff's
   to the tag, so every tagged tree would permanently carry the
   *previous* release's bundle.
4. **Audit-flag staleness cuts both ways**: composites-3c was stale
   (fixed v0.10.5), d4-native + cpcm-gradient were real. Always check
   the audit note's RESOLUTION section + `git log -S` before
   reproducing, and before spawning chats.
5. Slow-lane reds can hide behind triage timing: the Γ gauge
   default-flip landed after the full triage, so `test_bipole_gradient`'s
   slow lane was unjudged until the cut verification. The cut MUST
   re-run the slow lane of any file whose defaults flipped since the
   last triage.

## Phase E

Drop-box dirs `.release-status/{v0.10.x,v0.11.0,v0.12.0}/` deleted
post-tag (key numbers extracted into the CHANGELOG entries and this
record). `.release-status/audit-2026-05-31/` retained, 
`scripts/test_gate/gate_verdict.py` references it and its notes carry
the d4/cpcm/composites resolution trail.
