# v0.15.0 release prep -- cut plan + checklist (Neese's Cheetah)

```{note}
This is a historical pre-split release checklist for the archived monorepo.
Its co-located component paths and old branch/tag instructions are not a
current setup recipe. Use [release process](release_process.md) and
[toolset lifecycle](toolset_lifecycle.md) for the independent repositories.
```

Forward-looking playbook for the v0.15.0 cut, copy-adapted from the per-tag
mechanical-sprint playbook (docs/release_v0_12_0_prep.md cut record). This file
is the durable plan; the live per-chat coordination lives in the local
`.release-status/v0.15.0/` drop-box. Rewritten into a cut record at tag time.

v0.15.0 is the **release-paper baseline** (moved up from v0.14 because main
progressed fast). That raises the bar: every method the paper claims must be
correct or explicitly gated.

## FEATURE FREEZE (declared 2026-06-26)

**v0.15.0 freezes at `e94221f2`.** Scope = everything ready as of the freeze
(comprehensive -- includes the TD-DFT program, AICCM-a 3-D, the periodic-XC
cross-cell fix, the eigensolver framework, CASSCF gradients + geom-opt, DLPNO,
COOP/COHP, QTAIM, etc. -- regardless of prior roadmap timing). The roadmap is
being reworked to match what shipped rather than gating scope by roadmap-due.

The freeze is a *pinned commit*, not a branch lock: `main` stays open and other
chats keep committing normally (only `release` is protected). The release chat
cuts v0.15.0 from `e94221f2` and cherry-picks **only gate-red fixes** onto the
freeze line; everything pushed to `main` after `e94221f2` rides v0.15.1 / v0.16.
Both gate reds found on the freeze are fixed (periodic-XC param-gradient
convention; gilat-net density-mixer well-posed cell) and re-verified green at
`e94221f2`; the full-suite gate is running on it.

## Scope (high level, since v0.14.0)

A minor's worth of work: the unified eigensolver framework (Davidson/LOBPCG;
JD/GPLHR experimental), CASSCF analytic gradients + the geometry-optimization
framework, molecular DLPNO (MP2 + local CCSD foundation), NEVPT2/CASPT2 energies
(+ experimental gradients), COOP/COHP + QTAIM + fat-bands analysis, vibe-view
QVF v1.2, the dense-crystal periodic-XC correctness fix, and the experimental
AICCM cyclic-cluster lines (gated, ride v2.0+).

## Gating status

Resolved on main (were blockers):

* P0 dense-crystal real-space periodic XC (cross-cell density) -- fixed
  `295f7ada`; MgO/STO-3G now within ~1 mHa of CRYSTAL23, regression
  `tests/test_periodic_xc_cross_cell.py`.
* P0 CASSCF analytic gradient (now full-energy-FD-tight) -- `2201eb8b`.
* QTAIM gradient/Hessian crash -- fixed; analytic-Hessian upgrade landed.
* P2 COOP/COHP spin-polarized Fermi level -- fixed `060e65ab`.
* §8 geom-opt optimizer citations + §1 license inventory + CHANGELOG/roadmap
  parity + user-guide stubs -- `a715de4f`, `13b18f08`.
* GAPW experimental warning un-silenced on the run_periodic_job path -- `33ca4972`.

Remaining hard blocker (gates the freeze):

* **P1 selected-CI CASSCF (spin-pure) convergence regression** -- live gate red,
  `tests/test_selected_casci.py`. Routed to the CASSCF chat (likely cause
  `9540b2da`). Freeze waits on this going green or being correctly gated.

Scope decision (maintainer, 2026-06-25): **v0.15.0 ships everything that
landed.** Do not curate a narrow release scope -- include all landed features;
mark the experimental ones clearly and keep them gated (§14), but they ride in
the release. The effort goes into **reworking the roadmap** to reflect reality,
not into excluding work from the tag.

Experimental / gated (ship IN the release, clearly marked experimental, NOT a
paper-method claim): AICCM -a/-b CCM (import-guarded, not run_job-reachable;
roadmap rework to settle its long-term v2.0 line), GAPW all-electron, BIPOLE
meta-GGA analytic gradient, native D4 backend, GFN2-xTB, JD/GPLHR eigensolvers,
MS-CASPT2 gradient (preliminary, SA->MS shift limitation documented),
slab-Ewald-2D.

Carried-forward open bugs: see handovers/HANDOVER_OPEN_BUGS_V015.md (+ the
still-open v0.14 entries it references).

## Codename -- Neese's Cheetah (decided 2026-06-25)

DLPNO local-correlation headline. Frank Neese (DLPNO-CCSD(T), Riplinger-Neese
2013); cheetah = the speed local correlation buys. PRE-STAGED in
`python/vibeqc/banner.py` RELEASE_CODENAMES (a 0.14.x build still renders
Bartlett's Goose; locked at the v0.15.0 cut).

## Pre-freeze checklist (must hold before tagging)

1. [x] P1 selected-CI CASSCF red green or correctly gated. (Resolved; dense fallback at full-selection limit, 114 tests green.)
2. [x] Scope decided (2026-06-25): ship everything that landed (experimental
       marked + gated). Codename Neese's Cheetah pre-staged in banner.py.
3. [x] Roadmap REWORKED (maintainer ask): shipped-sweep every landed feature,
       re-plan the forward roadmap given how far ahead main is, settle the
       AICCM long-term line + the open codename slots. Routed to the docs chat.
4. [ ] CHANGELOG [Unreleased] is COMPREHENSIVE (everything that landed; mostly
       done `13b18f08`; re-verify COOP/COHP, QTAIM, CASSCF-gradient entries).
5. [ ] docs/license.md current (done `13b18f08`; re-verify scikit-learn [ml],
       k8f model, gCP tables).
6. [ ] Full-suite gate green-modulo-known on a freshly built core. This is a
       FEATURE release -- the v0.14.1 gate-inheritance shortcut does NOT apply;
       run scripts/test_gate/ for real (delete the jsonl first).
7. [x] Experimental guards verified (CASPT2/NEVPT2 correlation gradients now
   use relaxed full-energy FD when compute_corr_grad=True; the historical
   Z-vector shortcut is not advertised as paper-grade. JD/GPLHR gated.)

## Cut sequence (mechanical sprint, at freeze)

1. Pick a gate-green freeze SHA on main.
2. Pre-stage chosen codename in banner.py RELEASE_CODENAMES.
3. Version bump 0.14.2.dev0 -> 0.15.0 (pyproject.toml, CITATION.cff +
   date-released, docs/citing.md).
4. Promote CHANGELOG [Unreleased] -> "## [v0.15.0] -- DATE -- *Codename*".
5. Homepage docs/index.md "New in ..." admonition swap to v0.15.0; roadmap
   "shipped" sweep.
6. Example smoke pass: build a 0.15.0 venv, run the linked example inputs in
   a scratch directory, and leave the generated `.out` / `.system` /
   visualization artifacts uncommitted.
7. Release commit "release: v0.15.0" (commit-msg hook checks pyproject ==
   0.15.0 and a "## [v0.15.0]" CHANGELOG header).
8. Annotated tag v0.15.0 "v0.15.0 <Codename>".
9. Advance release: detach at v0.15.0; merge -s ours --no-ff origin/release;
   verify tree == tag tree; push HEAD:release (fast-forward).
10. Carry-back: main -> 0.15.1.dev0 + CHANGELOG [v0.15.0] section + fresh
    [Unreleased]; rebase-then-push.
11. Watch the release pipeline: docs-build (sphinx-reredirects is now in CI per
    v0.14.1) + docs-deploy; confirm vibe-qc.com flips to v0.15.0.
12. Post-tag: vq admin update the compute hosts to v0.15.0 for the paper batch;
    Phase E (delete the drop-box; fold numbers into this file as the cut record);
    update release-chat memory.

## Notes

* vibe-view must be installed in the gate venv (pip install -e ./vibe-view) or
  the QVF-reader tests fail on import (not a code bug).
* Keep docs/ ASCII-dash-clean (.githooks/check_no_em_dashes.py). CHANGELOG.md is
  at repo root and is exempt.
