# AGENTS.md

Orientation for AI coding agents working on vibe-qc: Claude Code, Codex, Cursor,
aider and others. [`CLAUDE.md`](CLAUDE.md) and [`CODEX.md`](CODEX.md) point here,
so this is the one place the rules live. Humans get the same ground at more
length in [`CONTRIBUTING.md`](CONTRIBUTING.md).

Keep this file short and current. If a rule here stops matching the
repository, fix the rule.

## Keep this repository public-safe

This product repository must remain ready for public mirroring at every commit.

- Never put private email addresses, real machine names or host aliases,
  internal hostnames or URLs, private IP addresses, account names, personal
  filesystem paths, credentials, tokens or site-specific deployment details
  in tracked files, filenames, generated artifacts or commit messages.
  This includes code, comments, tests, documentation and agent instructions.
- `project@vibe-qc.com` and `mpei@vibe-qc.com` are explicitly allowed public
  email addresses. Use generic placeholders and reserved example addresses
  for tests and documentation; do not copy real private values into fixtures.
- Keep private configuration separate from product code. Store it outside
  the product checkout on the local machine, or in the private agentic loop
  repository. Select it through environment variables, command-line options
  or an explicit external configuration path. Commit only portable defaults,
  schemas and examples without private values.
- Ignored files and custom folders under `.git` are not private configuration
  stores. Keep private operational logs, inventories and release evidence
  outside the product checkout too. Never commit secrets to the private loop
  repository; use the existing credential or secret store.
- Prevent contamination while making the change. Inspect the diff and use the
  existing automated privacy checks before committing. Fix a finding in the
  source; do not rely on a later sanitizer or create sanitation chats for
  routine releases. Never bypass a privacy failure to publish a release.

## What vibe-qc is

A quantum-chemistry code for molecules and solids: Hartree-Fock, DFT,
correlated methods and semiempirical methods, with periodic boundary
conditions throughout. The numerical core is C++ (`cpp/`, bound with pybind11);
the user-facing layer is Python (`python/vibeqc/`), driven mainly through
`run_job` and `run_periodic_job`.

On 2026-09-08 the former monorepo was split into four repositories.
This one is **`vibe-qc/vibe-qc`**. Its licence is MPL-2.0; `vibe-basis/`
stays here as an independently versioned subpackage. GitLab is the canonical
development source. Public source and issues are available on GitHub; see
[`docs/github_publication.md`](docs/github_publication.md) for the publication
gate and the current contribution-routing status. Maintainer access recipes
and tracker identifiers live in private operations documentation.

## Where things live

```
python/vibeqc/        Python package: runners, SCF drivers, properties, output
  output/             every user-facing file goes through here (.out, .system, QVF)
  periodic/           periodic machinery; periodic/ccm and periodic/chi are the experimental AICCM lines
  semiempirical/      MSINDO, PM6/OMx, GFN2-xTB, DFTB
  basis_library/      bundled basis sets and ECPs; regenerate the runtime overlay with scripts/setup_basis_library.sh
cpp/                  native integrals, Fock builds, lattice sums
tests/                pytest suite, classified per file in scripts/test_gate/suite_manifest.json
scripts/test_gate/    test lanes, the T1 release gate, the changelog guard
docs/                 Sphinx site, published from the `release` branch
examples/             runnable inputs (no generated outputs committed)
vibe-basis/           basis-set optimization subpackage with its own versioning
studies/              research studies and benchmark sets
website/              Astro marketing site, deployed from `main`
handovers/            public workstream notes; private operations stay external
```

| Looking for | Start at |
|---|---|
| What is planned | [`docs/roadmap.md`](docs/roadmap.md) |
| What shipped when | [`CHANGELOG.md`](CHANGELOG.md) |
| How a release is cut | [`docs/release_process.md`](docs/release_process.md) |
| Setup and rebuilds | [`docs/contributor_setup.md`](docs/contributor_setup.md) |
| Which tests to run | [`docs/developer_test_lanes.md`](docs/developer_test_lanes.md) |
| Licences of bundled components | [`docs/license.md`](docs/license.md) |
| Open defects | [the issue tracker](https://github.com/vibe-qc/vibe-qc/issues) |

## Setup, tests and docs

```sh
./scripts/install.sh --dev --extras test --venv .venv        # fresh clone; native deps take 15-40 min
VIBEQC_REQUIRE_VENDORED=ON .venv/bin/pip install -e . --no-build-isolation   # rebuild after C++ changes

.venv/bin/python scripts/test_gate/run_full_suite.py --list-lanes
.venv/bin/python scripts/test_gate/run_full_suite.py --wt "$PWD" --py .venv/bin/python --lane <lane>
.venv/bin/python -m pytest tests/test_<area>.py -q
```

- **Check which build you are testing.** An editable install pins the native
  extension to one checkout. After C++ changes, pulls or branch switches,
  rebuild, and confirm the loaded `_vibeqc_core` is newer than the newest
  `cpp/` source. Inside one `.dev0` cycle the installed version cannot tell
  two builds apart.
- **A new test file needs a manifest row.** `tests/conftest.py` refuses an
  unclassified file. Run `scripts/test_gate/update_suite_manifest.py`, and put
  a hand classification in its `CURATED` table rather than editing
  `suite_manifest.json` by hand.
- **Ordinary pushes to `main` run no test pipeline.** CI tests only release
  candidates, so your local run is the evidence. Name the tests you ran in the
  commit message.
- **Docs:** `sphinx-build -b html docs/ docs/_build/html`.

## Ground rules

These are the few rules whose mistakes are hard to undo. Everything else is
judgment; see [Defaults](#defaults).

1. **Never force-push `main` or `release`, and never rewrite pushed history.**
   Rebase, then push.
2. **Releases follow `docs/release_process.md`.** The agentic loop tags
   `vX.Y.Z` and moves `release`; the vibe-qc release chat prepares and proves
   the candidate. Don't create tags, push `release` or `release-candidate/*`,
   or pick codenames unless the maintainer asked you to for that cut.
3. **Keep private data out of the tree:** credentials, tokens, home-directory
   paths, private addresses, employer names. Keep private profiles and operator
   records outside every product checkout, worktree and Git database, under
   `VIBE_PRIVATE_ROOT` or the platform state directory. Ignoring a file does
   not make the checkout a suitable private store. Enable the hooks once per clone
   with `git config --local core.hooksPath .githooks`. If a hook blocks you,
   fix the content; bypass it only for a reviewed exception and say why in the
   commit message. Security issues go by email per [`SECURITY.md`](SECURITY.md).
4. **Licensing.** vibe-qc is MPL-2.0. Check the redistribution terms before
   bundling data or adding a dependency; when they are unclear, fetch on demand
   instead of bundling (as `vibeqc.basis_fetch` does). New Python dependencies
   go into optional extras with lazy imports. Record bundled components in
   `docs/license.md`.
5. **vibe-qc computes its own results.** Never import another
   quantum-chemistry program (PySCF, ORCA, CRYSTAL, xTB, ...) inside
   `python/vibeqc/` or `cpp/`. Numerical libraries such as libint, libxc,
   spglib, libecpint, Eigen, FFTW3 and ASE are fine. Tests and examples may use
   another program as an optional reference, in-process behind
   `pytest.importorskip` or through the subprocess runners in
   `examples/regression/`.
6. **Released changelog sections are pinned.** New entries go under
   `[Unreleased]`. Changing a released section needs
   `scripts/test_gate/changelog_guard.py pin vX.Y.Z --amended "why"` in the
   same commit; the T1 test and `.githooks/pre-push` enforce it.

## Defaults

This is how work normally goes. Use judgment when a case doesn't fit, and say
what you did.

- **Test what you touched**, plus its obvious consumers. Run the affected
  lanes for shared numerical machinery.
- **Keep `main` working.** Land in small increments, and gate unfinished
  features rather than leaving half-wired code paths.
- **CHANGELOG.** Add user-visible changes to `[Unreleased]`. Tests-only and
  internal changes may skip it.
- **Issue numbers.** When a commit fixes a tracked issue, put `(#N)` in the
  subject; triage finds fixes that way. Untracked work needs no number.
- **A second pair of eyes for wrong answers.** A fix to a wrong number, or to
  anything blocking a release, is checked by someone other than its author
  before its issue closes. Ask for that on the issue, with the commit, the
  commands and the numbers before the fix ([`CONTRIBUTING.md`](CONTRIBUTING.md)
  explains the form).
- **Cite what you implement.** Name the paper beside a published algorithm or
  reference value in the code, and give user-visible methods a citation route
  in `python/vibeqc/output/citations/database.toml` so the output can cite
  them.
- **Output goes through `vibeqc.output`.** Don't write user-facing files
  directly from a method module.
- **Docs follow behaviour.** When you change what a user sees, update the
  manual page in the same change. Prose in `docs/*.md` uses no em or en dashes
  (the pre-commit hook checks).
- **Experimental lines stay labelled.** AICCM/CCM and SECCM are experimental
  and not release-gated; don't present their results as production.
- **Ask the maintainer when the decision is theirs:** licensing, breaking a
  public API, a new hard dependency, release scope or codenames, or anything
  that commits another repository. Otherwise go ahead and explain your
  reasoning in the commit.

## Notes that save time

- **Periodic SCF that oscillates or lands far off is usually a bug.** Look at
  the gauge, Madelung and image sums, and reproduce against a reference,
  before reaching for damping, level shifts or tighter DIIS.
- **Stale builds and overlays look like real failures.** Check the four traps
  listed in `docs/release_process.md`: editable install replaced by a copy,
  stale `build/basis_library` overlay, same-second `.pyc`, and a core built
  from another tree.
- **A timeout on a loaded machine is not yet a regression.**
  `scripts/test_gate/gate_verdict.py --reverify` re-runs a new red once in
  isolation.
- **Heavy lanes exist.** Some periodic GDF tests run for minutes; see
  `docs/developer_test_lanes.md` before starting a full inventory.

## Working alongside other sessions

Several agent sessions may work on vibe-qc at once. They commit under the same
git identity, so authorship cannot tell you who changed what.

- **Work in your own clone** when you build or measure; a linked worktree
  shares the venv's editable install with its main checkout.
- **Treat a shared checkout's uncommitted changes as someone else's.** Stage
  files by name; don't stash, reset or check out over changes you didn't make.
- **Keep both sides of a conflict** in files many sessions append to, such as
  `CHANGELOG.md` and handovers. After a rebase, check that your changelog entry
  is still under `[Unreleased]`.
- **Handovers are for workstreams that span sessions.** Public implementation
  notes may live under `handovers/`. Private coordination, configuration and
  evidence belong in the external private state directory. A defect belongs
  on the tracker, not in a handover.
- **Decisions that belong to the maintainer** go on the tracker as a "Needs a
  decision" issue or a note, not a guess.

## Other repositories

- **vibe-view** is the viewer, **vibe-queue** the job queue and fleet tooling,
  **qvf** the file format. Each has its
  own tracker, releases and rules.
- **Report problems where they belong,** and don't edit another repository
  from here; file an issue in its tracker.
- **CI pins companions.** `VIBE_VIEW_TAG` and `QVF_TAG` in `.gitlab-ci.yml`
  select the published tags vibe-qc validates against.
- **Fleet hosts are vibe-queue's domain.** Don't update or reconfigure them
  unless the maintainer asked you to.

## What changed from the monorepo rules

Older comments, handovers and audits cite "CLAUDE.md § N" or "AGENTS.md rule
N". Those refer to historical monorepo instructions retained in the private
archive, rather than to numbered sections of this file.
They were written for dozens of parallel sessions and most were stricter than
this repository needs:

| Monorepo rule | Here |
|---|---|
| A fixed per-topic clone registry and per-release drop-box status files | Any clone of your own. The tracker, CHANGELOG and handovers carry status. |
| `bugctl` claims and leases before touching an issue | Plain issue comments work. The agentic loop may still use `bugctl` for its own lanes. |
| Issue number in every subject was non-negotiable | An issue number when there is one. |
| A full definition-of-done checklist before every push | Test what you touched and keep `main` working. |
| A persistent handover for every chat | Only for workstreams that span sessions. |
| `vibeqc.output` changes requested from the IO chat, never made directly | Make the change with tests, keeping output through `vibeqc.output`. |
| No refactors without approval | Propose large refactors in an issue first; small cleanups are fine. |
| The `basissetdev` standing rule and fleet-host directory rules | Retired here. `basissetdev` was not transferred; fleet rules live in vibe-queue. |
| Release operations owned by one release chat | The agentic loop tags; the release chat prepares (#230). |

Kept, because their mistakes are expensive: licensing, no other QC program at
runtime, privacy, no history rewrites, independent checks of wrong-answer
fixes, and pinned released changelog sections.
