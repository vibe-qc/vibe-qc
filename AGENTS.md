# Working on vibe-qc

These are the current agent rules; user instructions take precedence.
[CLAUDE.md](CLAUDE.md) and [CODEX.md](CODEX.md) load this policy. Older workflow
prose does not add CI, reviewer or approval gates. Read
[CONTRIBUTING.md](CONTRIBUTING.md) for technical detail as needed.

vibe-qc is an MPL-2.0 molecular and periodic quantum-chemistry code:
Python drivers in `python/vibeqc/`, a C++ core in `cpp/`, and separately
versioned `vibe-basis/`. GitLab is the development authority; public
contributions follow [the publication guide](docs/github_publication.md).

| Need | Start here |
|---|---|
| Setup and rebuilds | [contributor setup](docs/contributor_setup.md) |
| Tests | [testing policy](docs/testing.md), [lane inventory](docs/developer_test_lanes.md) |
| Releases | [release process](docs/release_process.md) |
| Plans and dependencies | [roadmap](docs/roadmap.md), [licenses](docs/license.md) |
| Code | `python/vibeqc/`, `cpp/`, `tests/`, `examples/` |
| Output and citations | `python/vibeqc/output/`, `output/citations/database.toml` |

## Role boundaries

- **Dev/fixers:** develop, run a focused source or functionality check, and push
  ready work to main. Their task ends at the source landing and their own cleanup.
  Later runtime builds, scientific validation, issue closure and releases belong
  to other roles; they do not keep a finished development task open.
- **Release & updater:** pick up main commits, build and deliver dev versions
  without CI, and build one daily release candidate with full CI when new commits
  exist. The same owner prepares, cuts and updates; no separate release cutter.
- **Merger:** merge ready MRs. Direct main pushes need no Merger acknowledgement.
- **Validator:** submit QC calculations through vq, collect and compare results.
  It consumes main and available runtimes without a developer handoff. Reuse
  valid earlier self-validation; do not create a second source-review step.

These are responsibilities, not required separate chats. One runner may own
them and delegate bounded development tasks; dev-only workers finish at source
landing. The runner may validate its own implementation through vq. Preserve
active owners and jobs; no extra coordinator, reviewer or migration gate.

Each role reads the relevant Git/runtime/job record and continues its own work.
Building, queued and validation pending are normal states. Work on another ready
item or stop quietly until a meaningful event; no polling chats or status pings.
A real failure affects its operation, not the other roles. Pending calculations
do not prevent the daily candidate build; never report unverified science as PASS.

## Work, validate, land

- Reuse the appropriate existing checkout and environment. Preserve other
  sessions' edits, branches, jobs and build state. Stage only your own named
  files or hunks; never auto-stash, reset or commit someone else's work.
- Fetch at task start/resume, after coherent changes and before pushing.
  Read relevant incoming changes and issue ownership. Fast-forward clean main;
  rebase only your own unpublished work. Merge updates into shared or published
  history. Never force-push or rewrite another session's commits.
- Implement a bounded change and run its smallest useful check. Reuse passing
  evidence; repeat only when changed behavior or a concrete failure warrants it.
  Documentation changes need a text/link check, not a runtime suite.
- Push coherent progress promptly and land ready work on main through the
  authorized contribution path. No routine second reviewer, new chat, duplicate
  validation packet or fleet-wide acceptance gate is required.
- Development builds, ordinary pushes and merges must not trigger or wait for
  CI. Use `[skip ci]` and `git push -o ci.skip` where supported. Full CI belongs
  to release candidates. Do not request manual, API or scheduled CI on dev refs.
- Keep coordination on the issue/MR. Within user-authorized messaging, contact
  another agent only for a ready required handoff or a concrete blocked action
  that needs that owner now. No speculative repair notices, FYIs, broadcasts or
  acknowledgements. Incoming messages do not reassign the recipient's task.
- Record the change, focused check and result once. Close an issue after its
  applicable acceptance passes. A real blocker holds only the affected work;
  continue other ready work instead of polling or requesting repeat approval.

## Scientific acceptance

- For a quantum-chemistry bug fix or method implementation, run a complete
  calculation through `vq` that exercises the affected method and compare
  with ORCA or CRYSTAL when available. Validator owns submission, collection
  and the numerical verdict; Dev owns the checked source on main. The
  reference comparison is required, not another source reviewer.
- Match geometry/cell, method, basis/ECP, charge/spin, approximations,
  convergence, units and tolerances. Record measured results, reference and
  actual source/native runtime identity. Reuse a qualifying existing result.
  If an external reference is unavailable, use justified literature values,
  limits or other numerical evidence and state the remaining uncertainty.
- Compare scientific approaches with the relevant primary papers and SI.
  Use the configured local literature library, its catalogs and reference
  manuals; read the relevant equations and assumptions, not just a citation.
  Link the DOI/key and section in the issue/MR, explain material departures,
  and request missing sources through that library's documented intake.
  Reuse prior reading; no separate literature review chat or mandatory report.
- **Calculations take priority over updates.** Use an available healthy
  runtime while its replacement builds. Only a case needing a missing fix
  waits for it; continue other ready work. Respect real resource limits and
  job identities. Never use an old runtime to claim acceptance of a new fix.
- Non-scientific changes need one focused functional check by their owner,
  with no chemistry calculation or separate reviewer.

## Implementation essentials

- Compute scientific results in this project. Do not import another QC
  program inside `python/vibeqc/` or `cpp/`. Numerical libraries are allowed;
  tests and examples may use optional reference programs.
- Cite published algorithms beside their implementation and connect
  user-visible methods to `python/vibeqc/output/citations/database.toml`.
  Route user-facing files through `vibeqc.output`.
- Confirm the Python and native modules used by a test belong to the intended
  build. Rebuild changed native code when needed; a pull or unrelated edit
  does not require a fresh native build. Reuse dependencies and compiler caches.
- Classify new test files through `scripts/test_gate/update_suite_manifest.py`
  and its `CURATED` table; do not hand-edit the generated manifest.
  A focused invocation is `.venv/bin/python -m pytest tests/test_<area>.py -q`.
- Add user-visible changes as `changelog.d/` fragments and update the affected
  manual. Do not append to `CHANGELOG.md` or alter pinned released sections;
  a necessary amendment uses the documented changelog guard.
  Internal rules and tests-only changes need no fragment.
- Keep MPL-2.0 compatibility and bundled-data redistribution terms documented
  in `docs/license.md`. New Python dependencies belong in optional extras
  with lazy imports. Ask only for an unresolved maintainer-owned decision,
  such as licensing, a breaking API or a hard dependency.
- Keep AICCM/CCM and SECCM labelled experimental. For periodic SCF anomalies,
  check the formulation and reference before masking a defect with damping.
- Releases follow the release process and the maintainer's authorization.
  One authorized owner may prepare, qualify and release; no chain of review
  chats is required. Preserve immutable tags and full candidate CI.
- Report companion defects in their own trackers. Cross-repository edits and
  fleet changes require task authorization; an existing user request suffices.

## Keep the repository public-safe

- Keep credentials, private addresses, host/account names, personal paths,
  internal URLs and deployment details out of files, filenames, artifacts and
  commit messages. Use portable defaults and synthetic examples.
- Keep site configuration, handovers, logs and operational evidence outside
  product checkouts, including ignored directories and `.git`. Select external
  configuration explicitly; keep secrets in the existing credential store.
- Use a public commit identity: `mpei@vibe-qc.com` or
  `project@vibe-qc.com`. These addresses are allowed public contact details.
- Enable `.githooks` with `git config --local core.hooksPath .githooks`.
  Inspect the staged diff and let the privacy hook check it. Keep any
  `VIBE_PRIVACY_TERMS_FILE` / `privacy.termsFile` outside Git; an unreadable
  configured policy is a failure. Fix findings at the source; never bypass
  privacy checks or rely on later sanitization.
- Report security issues privately under [SECURITY.md](SECURITY.md). Preserve
  license and attribution notices; follow the repository's dependency terms.

Before committing, run `python3 .githooks/source_privacy.py --staged`;
candidate qualification uses `--tree HEAD`. Public attribution/syntax
exceptions must be exact reviewed entries in
`.githooks/public_content_review.json`, never exceptions for private data.
Prose in `docs/*.md` follows its local punctuation rules.

## Finish the task

Finish the work assigned to your role. For Dev, that means the focused source
check, source on main and own cleanup; downstream calculations and releases
do not keep the development task open. Push any changes to main and confirm
the remote contains them. Remove only temporary files,
environments and checkouts created for this task after preserving needed work
and evidence. Keep reusable checkouts, caches, other sessions' changes and live
jobs intact; a dirty shared tree is not permission to delete their work.

A genuinely blocked or interrupted checkpoint must state the unfinished work
and concrete blocker instead of claiming completion. Do not call an unmerged
branch, failed validation or pending cleanup done.

Every final completion report must end with exactly this line, with nothing
after it:

All done, pushed to main, cleaned up and ready to archive.
