# Release operations

Public working notes for the vibe-qc release lane. Private operations,
host and deployment detail live outside this repository since #261
(`086bdd6`, 2026-09-14), which moved every handover out; this file was
re-added on 2026-09-16 at the maintainer's request and carries public
release state only.

**Who cuts.** The maintainer decided on 2026-09-13 (#230, closed): the agentic
loop tags `vX.Y.Z` and fast-forwards `release`; the vibe-qc release chat
prepares each release, proves its candidate, hands over the exact SHA and the
green pipeline id, and bumps `main` afterwards. `docs/release_process.md` is
the procedure.

## Current state, 2026-09-17

Re-measure rather than quote; `main` moves constantly.

| | value | re-measure with |
|---|---|---|
| current release | `v0.17.6` = `1c02daf`, candidate pipeline 5805 | `git describe --exact-match --tags origin/release` |
| `release` branch | at `1c02daf` | `git ls-remote --heads origin refs/heads/release` |
| `main` version | `0.17.7.dev0` (`0fea5c2`) | `awk -F'"' '/^version/{print $2; exit}' pyproject.toml` |
| live docs | `Current source: 0.17.6` | `curl -s https://vibe-qc.com/docs/ \| grep -o 'Current source: [0-9.]*'` |
| release-candidate branches | one: the loop's abandoned `v0.17.6` (`1528a5d`), left in place so the #230 record stays visible; this lane's was deleted at the tag | `git ls-remote --heads origin 'refs/heads/release-candidate/*'` |
| pinned changelog sections | 193 on `main`, `check` and `audit` clean | `python scripts/test_gate/changelog_guard.py check` and `... audit` |

## Cuts verified by this chat

Each was checked against the remote, not taken from a report.

| release | tag peels to | candidate pipeline | notes |
|---|---|---|---|
| v0.17.2 | `94322fb` | 5345, all five gate jobs green | prepared and handed over by this chat on 2026-09-13 |
| v0.17.3 | `1701bd2` | 5382 | release commit `6bb328d` is an ancestor; one documentation commit sits on top, and the tag is on the proven candidate SHA |
| v0.17.4 | `df9903b` | 5499 | `release` fast-forwarded; docs deploy green; live docs report 0.17.4 |
| v0.17.5 | `e197820` | 5619 | prepared by another session on `release-candidate/v0.17.5`; release pipeline 5643 green, live docs report 0.17.5 |
| v0.17.6 | `1c02daf` | 5805, all five jobs green | prepared by this chat; CI T1 job 12262 pass=10 notests=1 NEW-red=0; tagger `mpei@vibe-qc.com`; release pipeline 5813 green; live docs report 0.17.6; `main` bumped to `0.17.7.dev0` in `0fea5c2` |

Tagger identity on all three: `mpei@vibe-qc.com`. No candidate branch was left
behind, and the changelog pins audit reports no unexplained difference from any
tag.

## Open items

- **#206**, the range-separated GDF slowdown carried since v0.17.1. A verifier
  claim was granted on 2026-09-13 and **expired on 2026-09-14**, so `6818031`
  is still unverified. The baseline and the comparison trap are in note 25243.
- **#196**, the one-electron AO cutoff widening. !38 merged as `69e8306` and
  ships in v0.17.6: the nuclear sums no longer inherit the widened domain
  (V_ne unchanged to 1e-16 Ha, 4.9x faster on LiH/STO-3G). The **widening
  itself stays**, because it implements the #179 estimator-floor decision of
  2026-09-06, so the issue remains open as a maintainer decision rather than a
  tuning job.
- **#306** is **closed**: the auto-stabilisation ladder budget fix `efdb1d2`
  ships in v0.17.6. Its open successor is **#307**, which still carries the
  262-row v0.17.1-to-v0.17.4 comparison; that comparison has not been re-run
  against v0.17.6, and thymine's -1.370e-02 Ha has only a hypothesis (a
  finite-temperature rung reporting `A = E - T*S`).
- **#358** (P1, filed 2026-09-17), the periodic ECP lattice operator is
  nondeterministic and silently corrupt for pob-TZVP-rev2 W, Hf and Ta. Carried
  into v0.17.6 as a known issue. It passes the existing suite because the ECP
  pins never reach the radial quadrature.
- **#308** (P1), the 2-D slab QVF density artifact, 1.0716 where 2.0 is
  expected. Candidate fix open as **!52** (#45), unmerged. **#257** (P2, in
  progress) is carried too.
- **#227**, the changelog misfiling, fixed on `main` as `1a1fb81`. The re-check
  assigned to the loop's verifier lane (note 26236) has not happened; note
  26253 carries the acceptance to use, because the older one was superseded.
- **#229**, the guard against entries landing in released sections. Landed as
  `f87a560` (history correction) and `e8d6821` (the guard). Independent
  verification requested in note 26252 and still open. Do not close it yourself.
- **#259**, roadmap pass 2. Pass 1 landed as `848112a`.
- **v0.17.5 trigger** (maintainer, 2026-09-13): #206 verified resolved and #196
  fixed, or sooner for a verified fix to a wrong-answer bug. The second clause
  fired: **#281**, the chi DLPNO frozen-core wrong answer (P1), landed as
  `75c72bb`. #206 and #196 are carried into v0.17.5 as known issues, unfixed.

## v0.17.5 preparation, 2026-09-16

Prepared on `release-candidate/v0.17.5`, branched from `origin/main` at
`86d0a8946401868f2028cba28a1532166dca3eef`. Scope is the 31 commits between
`df9903b` and that SHA.

- The `[Unreleased]` section became `## [v0.17.5] - 2026-09-16 - *Tew's Tern*`,
  every entry keeping its author's wording; a preamble and an explicit #206 /
  #196 known-issues carry-forward are the only new prose, plus one entry for
  the QCSchema/QCElemental compatibility boundary that `45b358f` documented.
- Version sites moved to 0.17.5 in `pyproject.toml`, `CITATION.cff` and
  `docs/citing.md`; `docs/relocalization_worker.md` now says the worker ships
  in 0.17.5 rather than being an unreleased 0.17.5.dev0 addition.
- `VIBE_VIEW_TAG` moves to **v2.17.1**, the current viewer release. Viewer
  `main` carries unreleased 2.18.0 preparation and is not pinned. `QVF_TAG`
  stays at `v0.1.0-docs.1`, still the newest qvf tag.
- **Excluded** from this candidate: **!43 (#222)** and **!44 (#282)** both
  merged to `main` *after* the branch point (`28d2604` and `d660fae`), so they
  land in the next patch; **!42 (#284)**, **!38 (#196)** and **!28 (#241)** are
  still open with CI not green. `main` has since moved to `6ed795b`, also
  picking up #285, #237 and the optional OpenTrustRegion optimizer. The
  candidate was **not** rebased and stays at `86d0a89`.
- **Cut and released 2026-09-17**: tag `e197820`, tagger `mpei@vibe-qc.com`,
  candidate pipeline 5619 and release pipeline 5643 both green, no candidate
  branch left behind. Verified against the remote by this chat.
- Tagging, the `release` fast-forward and the post-tag `.dev0` bump are the
  coordinator's, from this candidate's proven SHA and pipeline.

## v0.17.6 preparation, 2026-09-17

Prepared on `main`. The release commit is `2c722e1`, with `1c02daf` carrying
#358 and #257 into its known issues on top, the shape v0.17.5 used
(`0187aca` + `e197820`). The candidate is `release-candidate/v0.17.6-main` at
`1c02daf`, pipeline **5805 green on all five jobs** (build-test 56.6 min,
docs-build 7.0 min). Scope is the 76 commits between `e197820` and `1c02daf`
(`git rev-list --count e197820..1c02daf`, 20 of them merges). CI's T1 receipt
is job 12262: `import OK 0.17.6`, pass=10 notests=1 NEW-red=0, with
`tests/test_periodic_molden_gamma_export.py` at 634.3 s against 623.0 s at
v0.17.5. Handed over on #230 note 29806.

- `[Unreleased]` was reconciled first (`812fce1`). Parallel merges had left it
  with 36 blocks under 25 headings: seven byte-identical duplicates, one
  "k-pair star planner" copy carrying a foreign #282 bullet appended by a
  merge, and four separate bare `### Fixed` blocks. The duplicates are gone and
  the four `### Fixed` blocks are one block with all eight bullets. No entry was
  reworded: the set of non-blank lines is equal before and after, both ways.
- The two `changelog.d/` fragments were folded in with
  `scripts/assemble_changelog.py --clean` before the promotion, the first cut to
  use the fragment workflow (`2c92d37`).
- `[Unreleased]` became `## [v0.17.6] - 2026-09-17 - *Tew's Tern*`, entries
  ordered fixes, changes, additions, each keeping its author's wording. New
  prose is the preamble and the known-issues carry-forward only.
- Version sites moved to 0.17.6 in `pyproject.toml`, `CITATION.cff` and
  `docs/citing.md`; the new section is pinned (193 sections).
- **Known issues carried:** #206 (unverified since v0.17.2), #307 (the open
  successor of the closed #306; the ladder-budget fix ships here but the 262-row
  comparison has not been re-run), #308 (the 2-D slab QVF density artifact,
  candidate fix open as !52). **#196 changed state:** its fix `69e8306` landed
  after v0.17.5, so the nuclear sums no longer inherit the widened one-electron
  AO domain; the widening itself stays under the #179 estimator-floor decision,
  so the issue remains open and the changelog says only that much.
- `VIBE_VIEW_TAG` v2.17.1 and `QVF_TAG` v0.1.0-docs.1 are unchanged, and both
  are still the newest published tags of their repositories, checked with
  `git ls-remote --tags` against each.
- **`main` moved during preparation.** #154 (`2293957`, molecular PM6 basin
  ambiguity) landed while the promotion was being written, so the release
  commit was rebased onto it and #154 is *in* this release. Its fragment was
  folded into the v0.17.6 section by hand, since `assemble_changelog.py` folds
  only into `[Unreleased]`, and the preamble names it.
- **Pre-flight caught trap four on the first run.** #154 changed
  `cpp/src/semiempirical/methods/nddo/pm6_fock.cpp`, so the core built minutes
  earlier was older than a `cpp/` source and the gate would have measured a
  tree without that change. Rebuilt, then re-run. Worth recording separately:
  before today's rebuild this checkout's venv still held a **0.17.2** install,
  which is the same trap two cuts wide and exactly what #285 now turns from a
  warning into a refusal.
- **A second v0.17.6 preparation exists and is unresolved.**
  `release-candidate/v0.17.6` on origin is `1528a5d` (parent `2fe3e99`),
  branched from `2293957`, with
  pipelines 5768 and 5789 green. It is **not on `main`**, its base is six
  commits behind, its `[v0.17.6]` section carries the duplicate headings this
  lane reconciled in `4d0c8ac`, and it has no entry for #325 whose fragment its
  base never folded. It named #358 and #257, which this lane had missed and has
  since carried. The collision is on #230 note 29770 for the maintainer; this
  lane has not force-pushed, deleted or tagged anything, and pushed its own
  candidate under a separate branch name rather than over that one.
- Tagging, the `release` fast-forward and the post-tag `.dev0` bump are the
  coordinator's, from the SHA the maintainer picks and its proven pipeline.
- **Local gate on `1c02daf`:** PREFLIGHT PASSED. T1 is 9 pass, 1 notests,
  NEW-red 0, and `tests/test_periodic_molden_gamma_export.py` was classified a
  contention artifact by `gate_verdict.py` itself: FAIL at 606.6 s under load
  against the 600 s per-test cap, PASS alone in 1020.9 s. That lane sits at the
  cap because of #206 and passed in CI at v0.17.5 in 623.0 s (pipeline 5619,
  job 11854).

## Handed to dev lanes, 2026-09-17

A capability verification for the release paper measured the method-support
tables against v0.17.1, and every finding was re-verified against `main` at
`a13ad47`. Nothing had been fixed in the 318 commits since v0.17.1: the files a
fix would touch carry no commits in that range. The release chat wrote a
strategy and one prompt per owning lane, and gave both to the maintainer.

The pattern behind most findings: `run_job` accepts a request and answers it on
a different surface than the user asked for, with nothing in the output saying
so. That matters to this lane, because a release cannot claim a capability whose
optimizer walks another surface.

| finding | owning lane | state on main |
|---|---|---|
| solvated optimization drops the solvent (default backend passes none; other routes differentiate the gas-phase surface) | solvation, output-docs | still holds; the manual claims the opposite |
| D4, gCP, SRB outside the optimized surface; FD Hessian has no dispersion | molecular-scf-dft | still holds |
| correlated `hessian=True` differentiates the mean field, output names no surface | molecular-correlation | still holds |
| FCI, NEVPT2, CASPT2, MRCI with `optimize=True` take the mean-field fallback | molecular-correlation | still holds; the source flags it as a maintainer decision |
| `density_fit` and `cosx` silently dropped for ROHF/ROKS; Casida silently runs TDA on ROHF | molecular-scf-dft | still holds; the ROHF options already exist, so it is missing wiring |
| double hybrids have no `run_job` route | molecular-correlation | intentional and test-pinned; a gap to decide |
| #206 and #196 periodic GDF cost regressions | pbc-gdf | unresolved; #206's verifier claim expired 2026-09-14 |

None of these is tracked yet: no open issue among the 211 covers them. Each
prompt carries a ready-made issue title so the owning lane files its own.

## Tooling this lane owns

- `scripts/test_gate/changelog_guard.py` with `changelog_pins.toml`: every
  released changelog section is pinned by content hash. **Pin the new section in
  the release commit** (`... pin vX.Y.Z`), or the T1 test
  `tests/test_changelog_released_sections.py` fails and `.githooks/pre-push`
  refuses the push. A deliberate edit to a released section needs
  `pin vX.Y.Z --amended "why"`.
- `AGENTS.md`, `CLAUDE.md` and `CODEX.md` at the repository root carry the agent
  rules (`762d01d`), relaxed from the monorepo's.

## Learnings worth keeping

- **A gate is evidence about the tree the interpreter loaded**, not the tree you
  meant. Rebuild after C++ changes; the installed version cannot tell two builds
  apart inside one `.dev0` cycle.
- **Compare a released changelog section with its own tag**, not only with the
  latest tag. Comparing against the latest tag hid six misattributions that the
  tag itself had shipped.
- **Check "shipped" claims against public API, docs and tests**, not changelog
  text. Of seven features the roadmap called planned and the changelog called
  shipped, only two shipped without limits.
- **A check script's exit code is a claim too.** A chain whose `set -e` did not
  fire reported success after a printed failure; put each check in a script that
  exits, and print a final PASSED line only at the end.
- **A timeout on a loaded machine is not yet a regression.** Re-run a new red in
  isolation (`gate_verdict.py --reverify`) before reading it as one.
- **zsh applies history modifiers to `$VAR:r`,** so brace variables in refspecs:
  `"${SHA}:refs/heads/..."`.
- **A check that reads stdin ignores the filename you gave it.**
  `.githooks/private_terms.py` takes its input from stdin, so
  `private_terms.py FILE` scans nothing and exits 0. Pipe the content in, the
  way `.githooks/pre-commit` does, and note that it only runs at all where
  `privacy.termsFile` is configured.
- **Let the gate classify its own red.** A file that fails under load and
  passes alone is reported as a contention artifact, not a gate red. Read the
  verdict block rather than the per-file line.
- **A moving `main` is part of the cut.** Three rebases during this preparation
  pulled four commits into v0.17.6; each one's changelog fragment had to be
  folded into the promoted section by hand, because
  `scripts/assemble_changelog.py` folds into `[Unreleased]` only. Check for a
  user-visible change that arrived without a fragment: `901c486` put its entry
  straight into `[Unreleased]` and would otherwise have been promoted into the
  next release's section while its code shipped in this one.
