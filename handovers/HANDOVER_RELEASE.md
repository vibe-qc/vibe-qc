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

## Current state, 2026-09-16

Re-measure rather than quote; `main` moves constantly.

| | value | re-measure with |
|---|---|---|
| current release | `v0.17.4` = `df9903b`, candidate pipeline 5499 | `git describe --exact-match --tags origin/release` |
| `release` branch | at `df9903b` | `git ls-remote --heads origin refs/heads/release` |
| `main` version | `0.17.5.dev0`; this candidate branch carries `0.17.5` | `awk -F'"' '/^version/{print $2; exit}' pyproject.toml` |
| live docs | `Current source: 0.17.4` | `curl -s https://vibe-qc.com/docs/ \| grep -o 'Current source: [0-9.]*'` |
| release-candidate branches | `release-candidate/v0.17.5` | `git ls-remote --heads origin 'refs/heads/release-candidate/*'` |
| pinned changelog sections | 192 on this branch (191 on `main`), `check` and `audit` clean | `python scripts/test_gate/changelog_guard.py check` and `... audit` |

## Cuts verified by this chat

Each was checked against the remote, not taken from a report.

| release | tag peels to | candidate pipeline | notes |
|---|---|---|---|
| v0.17.2 | `94322fb` | 5345, all five gate jobs green | prepared and handed over by this chat on 2026-09-13 |
| v0.17.3 | `1701bd2` | 5382 | release commit `6bb328d` is an ancestor; one documentation commit sits on top, and the tag is on the proven candidate SHA |
| v0.17.4 | `df9903b` | 5499 | `release` fast-forwarded; docs deploy green; live docs report 0.17.4 |

Tagger identity on all three: `mpei@vibe-qc.com`. No candidate branch was left
behind, and the changelog pins audit reports no unexplained difference from any
tag.

## Open items

- **#206**, the range-separated GDF slowdown carried since v0.17.1. A verifier
  claim was granted on 2026-09-13 and **expired on 2026-09-14**, so `6818031`
  is still unverified. The baseline and the comparison trap are in note 25243.
- **#196**, the one-electron AO cutoff widening. A candidate change is open as
  **!38** (`claude/196-screened-nuclear-domain`, screens the periodic
  nuclear-attraction lattice domain); it is unmerged and not in v0.17.5.
- **#306**, molecular GFN2-xTB regressed between v0.17.1 and v0.17.4 (P1).
  Already shipped in v0.17.4, so v0.17.5 inherits it. Norbornadiene and
  p-benzoquinone stopped converging and thymine moved -1.370e-02 Ha. A bisect
  over `b4f6035..df9903b` is running in a separate lane. Carried into the
  v0.17.5 changelog as a known issue; it does not block this cut.
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
- Tagging, the `release` fast-forward and the post-tag `.dev0` bump are the
  coordinator's, from this candidate's proven SHA and pipeline.

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
