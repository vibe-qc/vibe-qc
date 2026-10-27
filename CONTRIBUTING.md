# Contributing to vibe-qc

Contributions and feedback both welcome. The decision tree below
covers where to direct what.

This repository owns the engine, `vibe-basis/`, and the shared marketing site.
Report viewer work in [vibe-view](https://github.com/vibe-qc/vibe-view),
scheduling work in [vibe-queue](https://github.com/vibe-qc/vibe-queue),
and format/specification work in [QVF](https://github.com/vibe-qc/qvf).
Each companion owns its code, documentation and releases; a core patch does
not update a companion checkout. Public-site ownership and shared CSS are
covered in [site publishing](https://github.com/vibe-qc/vibe-qc/blob/main/docs/site_publishing.md).

## Where to report what

| What | Where |
| --- | --- |
| **Security vulnerability** — exploitable bug, memory corruption, anything you think shouldn't be public-by-default | Email **`mpei@vibe-qc.com`** directly. PGP fingerprint `CC6D 30BB DF96 F694 C615  FBDE 4CD5 65CF 26B1 E7E5` (key at <https://vibe-qc.com/docs/_static/pgp/mpei.asc>). See [SECURITY.md](https://github.com/vibe-qc/vibe-qc/blob/main/SECURITY.md) for what to include and the disclosure timeline. **Do not open a public issue.** |
| **Bug, install problem, missing feature, "this didn't work the way the docs said"** | [GitHub issues](https://github.com/vibe-qc/vibe-qc/issues). Search for an existing issue first; comment on it if there is one. Filing requires a GitHub account. |
| **General feedback, question, "is this a bug or am I doing it wrong?"** | Email **`mpei@vibe-qc.com`**. No GitLab account needed; replies go to your address directly. |
| **Code-of-conduct violation** | Email **`mpei@vibe-qc.com`**. See [CODE_OF_CONDUCT.md](https://github.com/vibe-qc/vibe-qc/blob/main/CODE_OF_CONDUCT.md). |
| **You want to send a patch** | Read the rest of this file. |
| **You fixed something, or you want to check someone else's fix** | The issue it belongs to. Verification requests and verdicts are **comments on the corresponding issue**, never files in the repository — see [How a fix gets verified](#how-a-fix-gets-verified). |
| **You want to fund the project** | [GitHub Sponsors](https://github.com/sponsors/mpeintinger) (recurring monthly) or [Ko-fi](https://ko-fi.com/mpeintinger) (one-time). Full pitch + author bio + funding goals on the [support page](https://vibe-qc.com/docs/support.html). |

All contributors and maintainers are expected to follow the
[Code of Conduct](https://github.com/vibe-qc/vibe-qc/blob/main/CODE_OF_CONDUCT.md).

## How a fix gets verified

Development and scientific acceptance have different completion points. A
source change is ready to land after its focused check passes. A scientific
issue closes after a complete calculation exercises the affected method and
its numerical result is compared with a suitable reference. The same runner
may do both; a second author or separate review chat is not required.

Record evidence once on the corresponding issue or merge request:

- the defect or feature, the landed commit, and the focused check and result;
- the input and command needed to reproduce the calculation;
- the actual Python/native runtime identity, not just the checkout SHA;
- measured values, reference values, units, tolerances, and the verdict;
- any missing comparison, unrelated failure, or remaining uncertainty.

For quantum-chemistry changes, the validator submits complete calculations
through `vq` and compares with ORCA or CRYSTAL when available. Match geometry
or cell, method, basis/ECP, charge/spin, approximations and convergence settings.
If an external reference is unavailable, document justified literature values,
limits or other numerical evidence and the remaining uncertainty. Reuse valid
prior self-validation. Compilation and unit tests alone are not scientific
acceptance, and a pending calculation is not a PASS.

Development workers finish after the checked source lands on main and their
own cleanup is complete. Runtime builds, scientific acceptance and releases
continue under their respective owners. Non-scientific changes need one
focused functional check; documentation changes need a text/link check.

Keep operational logs, machine details and private coordination records
outside product repositories. Public reports and patches follow the
[publication guide](https://vibe-qc.com/docs/github_publication.html);
maintainers link them to canonical GitLab work when routing is needed.

### Tests for optional dependencies must run in every environment

Force the branch you want to test; do not wait for an environment that
happens to lack the dependency.

```python
# Good: exercises the missing-dependency path whether or not trexio is
# installed, so it runs in every venv and in CI.
monkeypatch.setitem(sys.modules, "trexio", None)
with pytest.raises(ImportError) as excinfo:
    trexio_format._require_trexio()
```

The shape to avoid is a test gated to run *only* when a dependency is
absent, typically `@pytest.mark.skipif(HAVE_SOMETHING, ...)`. A broken
assertion inside such a gate is invisible to everyone who has the package
installed, and it can stay broken indefinitely because the people who could
see the failure are the ones not running it. **A test that runs only in the
environment you do not have is a test nobody reads the failures of.**

That is not hypothetical: vibe-view carried two assertions inside a
`skipif(HAVE_RDKIT)` gate that passed in CI only because the checkout
directory happened to be named `vibe-view`, so a substring check against a
path-derived install hint matched by coincidence. They were asserting where
the repository sat on disk, not that the message helped a user. Found and
fixed 2026-09-10.

This suite currently has no such gate, and it is worth keeping that way. To
check:

```sh
grep -rnE 'skipif\(\s*(HAVE_|HAS_|_HAS_|has_)[A-Za-z_0-9]*\s*[,)]' tests/
```

Read the hits rather than counting them, and check the *direction* before
believing either result. Only a condition that is true when the dependency is
**present** hides anything:

| Condition | Test runs when | Verdict |
|---|---|---|
| `skipif(HAVE_RDKIT, ...)` | RDKit **absent** | the trap |
| `skipif(not HAVE_RDKIT, ...)` | RDKit **present** | fine, the ordinary needs-the-extra gate |
| `skipif(shutil.which("git") is None, ...)` | git **present** | fine, the safe inverse |

A non-empty result is therefore not by itself a finding.

The narrow pattern above already excludes both safe forms, so running just it
usually needs no triage at all. Widen it when you want confidence that no
variant slipped past, and triage what the wider net drags in. Quote which
pattern produced a count, because the two do not agree: sweeping vibe-view's
suite, the documented pattern returned **1** and a widened one returned
**7** — six of them the safe `which` form, and the odd one out, present in
both, merely prose inside a docstring rather than a gate at all.

An empty result is also worth nothing until the pattern is shown capable of
matching. Plant the shapes in a throwaway file first, confirm the grep finds
them, then trust the sweep:

```sh
printf 'import pytest\n@pytest.mark.skipif(HAVE_RDKIT)\ndef test_x(): ...\n' \
    > /tmp/gate_control.py
grep -rnE 'skipif\(\s*(HAVE_|HAS_|_HAS_|has_)[A-Za-z_0-9]*\s*[,)]' /tmp/gate_control.py
```

Related but distinct: it is fine for an *extra leg* of a test to skip when a
companion is absent, as the writer-to-viewer integration above does, provided
the core assertions still run unconditionally.

## Branch and checkout policy

Clone the public `vibe-qc/vibe-qc` repository to prepare contributions.
Submit patches as pull requests, with the evidence described below; maintainers
route accepted changes into canonical GitLab development. Automatic routing is
not yet active; see [publication status](https://vibe-qc.com/docs/github_publication.html).

Maintainers working in an authorized upstream checkout land ready changes on
`main` after a focused local check. Reuse the appropriate existing checkout
and environment. Inspect the working tree and fetch before editing, after a
coherent change, and before pushing:

```sh
git status --short
git fetch origin
# Only when main is clean and can fast-forward:
git merge --ff-only origin/main
# After committing your own checked changes with [skip ci]:
git push -o ci.skip origin HEAD:main
```

Stage only your own named files or hunks. Do not auto-stash, reset, delete or
commit another session's work. Rebase only your own unpublished commits;
merge incoming changes into shared or published history. If a fast-forward
would overwrite local work, preserve it and resolve the integration without
rewriting another contributor's commits. Never force-push main or release
refs, or move an existing release tag.

Ordinary development pushes and merges skip CI. Use `[skip ci]` in the
commit message and `git push -o ci.skip` where supported. Full CI belongs to
release candidates, not routine development refs. The authorized release
owner prepares, qualifies and cuts releases; see the
[release process](https://github.com/vibe-qc/vibe-qc/blob/main/docs/release_process.md).
Website publication is described in
[site publishing](https://github.com/vibe-qc/vibe-qc/blob/main/docs/site_publishing.md).

## Release notes

For each user-visible change, add one Markdown file under `changelog.d/`,
named after the issue or a short descriptive slug. Start it with a `### `
heading or a `- ` bullet. Tests-only and internal changes may omit a fragment.
Do not edit `CHANGELOG.md` directly for ordinary contributions. Separate
fragments let concurrent changes land without conflicting at `[Unreleased]`.

Validate the fragments before committing:

```sh
python scripts/assemble_changelog.py --check
```

The release owner assembles them into `[Unreleased]` with
`scripts/assemble_changelog.py` when preparing a release; the command and
cleanup workflow are in `changelog.d/README.md`. Released sections remain
pinned. Deliberate amendments follow the pre-push guard procedure below.

## Personal information

Do not commit personal home-directory paths or unrelated employer
information. Use `~/`, `/home/USER/`, or `<vibe-qc-checkout>` placeholders in
examples. The pre-commit hook checks full staged files for private emails,
internal host suffixes, private IPs, personal paths and external private terms. Its narrow
`runner`, `root`, and `user` allowlist exists for CI accounts and generic
fixtures; it is not permission to add personal data under another name.
The hook and CI use the shared `.githooks/source_privacy.py` implementation.

## Documentation prose

Use commas, parentheses or ordinary hyphens instead of em or en dashes in
Markdown prose under `docs/`. The hook checks each touched file's full staged
content, including older prose. Fenced code blocks, inline code and inline
math are exempt because they may contain literal output or mathematical
notation. Provenance-marked QVF snapshots under `docs/qvf/_vendored/` are
also exempt: change those in the QVF source repository, then deliberately
re-vendor and update the pin.

These public policies are the source of truth for the hooks. Do not bypass
hooks unless the maintainer explicitly authorizes the particular exception;
record that authorization and the reason in the commit message. Privacy failures
must be fixed in source or resolved with an exact public-content review; never
bypass them with `--no-verify`.

## Before you open a merge request

1. Confirm no existing issue already covers the change — if there is
   one, comment on it so we can coordinate.
2. Run a bounded reproduction or focused tests for the change. A whole area lane
   is optional broad validation, not a default pre-push requirement:
   ```sh
   .venv/bin/python scripts/test_gate/run_full_suite.py \
       --wt "$PWD" --py .venv/bin/python --lane <affected-lane>
   ```
3. For documentation prose, check changed text, local links and section
   anchors. A native rebuild or runtime suite is not required. If changing
   Sphinx configuration, directives or page layout, build and inspect the
   affected pages using an existing docs environment. Keep temporary build
   output outside the product checkout:
   ```sh
   .venv/bin/python -m sphinx -b html docs/ /tmp/vibeqc-doc-preview
   ```
4. For anything that adds a dependency or changes public API, open
   an issue first to check scope fit.

### Build against the vendored dependencies before trusting a lane

A missing `third_party/<dep>/install/` makes the top-level CMake configure
fall back to system discovery for that one dependency, and it does so
silently. Set `VIBEQC_REQUIRE_VENDORED=ON` on any build whose test numbers
you intend to report, so that fallback becomes a configure-time error
instead:

```sh
VIBEQC_REQUIRE_VENDORED=ON .venv/bin/pip install -e . \
    --no-build-isolation --config-settings=build-dir=build/<your-build-dir>
```

What this catches is not a wrong number. A system libint built without the
one-body derivative kernels (`overlap1`, `kinetic1`, `elecpot1`) links
fine and passes every lane that never differentiates, then calls a null
kernel pointer the moment one does: `compute_gradient` takes `SIGSEGV`
inside an OpenMP worker, which kills the pytest process and takes the rest
of that lane's results with it. Homebrew's `libint` 2.13.1 is such a
build; the vendored one is not.

This bites hardest in a worktree, which starts with no `third_party/` at
all, so an agent harness that builds in one gets the fallback by default.
See [installation](https://vibe-qc.com/docs/installation.html) for what
`scripts/setup_native_deps.sh` fetches and where it installs.

### Recommended pre-merge verification

Choose the smallest check that exercises the changed behavior. For example:

```sh
.venv/bin/python -m pytest tests/test_<area>.py -q
```

Use [testing policy](https://vibe-qc.com/docs/testing.html) and
[developer test lanes](https://vibe-qc.com/docs/developer_test_lanes.html)
when a broader lane is justified. Confirm that the Python package and native
module belong to the intended build before reporting a numerical result.
Rebuild changed native code when needed; unrelated edits do not require one.

List the exact check, result and source/runtime identity in the issue or MR.
Reuse passing evidence unless behavior changed or a concrete failure warrants
repeating it. Do not claim full-inventory success from a focused test or T1.

## Pre-commit hook (one-time setup)

After cloning, point git at the tracked `.githooks/` directory so the
repo's commit guards run on every commit:

```sh
git config --local core.hooksPath .githooks
```

This single setting activates every hook in `.githooks/` (git resolves
one hooks directory for all hook types):

- **`pre-commit`** — first refuses a commit whose author or committer
  email is not a public address (`mpei@vibe-qc.com`, `project@vibe-qc.com`,
  or one you add with `git config --local --add privacy.commitEmail
  you@example.org`); set the clone's identity with
  `git config --local user.email`. It then refuses commits whose staged file content contains
  absolute paths into the author's home directories or other
  personal-info patterns ([personal information](#personal-information)). It then runs
  `.githooks/check_no_em_dashes.py`, which refuses staged `docs/*.md`
  carrying em or en dashes in prose (code fences, inline code, and
  inline math are exempt). Last, `.githooks/check_changelog_fragments.py`
  runs `scripts/assemble_changelog.py --check` on the staged version of
  each `changelog.d/` fragment the commit adds or changes and refuses one
  that does not open with a `### ` heading or a `- ` bullet.
- **`commit-msg`** — refuses a `release: vX.Y.Z` commit unless
  `pyproject.toml` `[project] version` == `X.Y.Z` and a `## [vX.Y.Z]`
  CHANGELOG header exists. This is the backstop against tagging a
  release from a tree whose version bump was skipped — the defect that
  shipped v0.11.2 / v0.11.3 self-reporting `0.11.1`. See
  [maintainer publication](https://vibe-qc.com/docs/release_process.html#maintainer-publication).
- **`pre-push`** — refuses a push to `main` or `release-candidate/*`
  whose `CHANGELOG.md` changes an already released section without
  re-pinning it in `scripts/test_gate/changelog_pins.toml` (#229).
  It runs at push time because misfiled entries arrive through
  rebases, which never run `pre-commit`. Ordinary unreleased notes belong in
  `changelog.d/` until release assembly; a deliberate amendment to a released
  section is re-pinned with
  `python scripts/test_gate/changelog_guard.py pin vX.Y.Z --amended "why"`.
  The committed guard, pins and changelog must all be readable regular files;
  repairing only the working tree does not repair the pushed commit (#264).
  Historical commits are exempt only when neither policy file occurs in their
  complete reachable history. Deleting the guard or pins does not restore that
  exemption. If shallow history prevents proving it, fetch complete history
  before retrying. Deletions and pushes to unrelated destination refs remain
  exempt. The hook uses the committed guard as trusted project code; it is not
  a substitute for reviewing changes to the guard itself.

Fix hook failures before committing. A maintainer-authorized exception follows
the [documentation and hook policy](#documentation-prose) above.

### Worktrees need the setting applied per worktree

Reuse existing checkouts under the
[branch and checkout policy](#branch-and-checkout-policy). If working in an
existing linked worktree, check its effective hook configuration before
trusting it; a worktree can override the shared setting.

`--local` writes the shared `.git/config`, but once a repo has
`extensions.worktreeConfig = true` — git enables it the first time
anything writes a worktree-scoped setting — a linked worktree can carry
its own `core.hooksPath` in `.git/worktrees/<name>/config.worktree` that
**overrides** the shared value. Check what a worktree actually resolves,
and repair it with `--worktree`:

```sh
git -C <worktree> config --show-scope --get-all core.hooksPath
git -C <worktree> config --worktree core.hooksPath .githooks
```

Keep the path **relative**. An absolute path bakes the checkout
location into each worktree's config, so renaming or moving the clone
leaves `core.hooksPath` pointing at a directory that no longer exists.
Git skips a missing hooks directory **silently** — no warning, no error,
no failed commit — so every guard above goes inert without anyone
noticing. A relative `.githooks` resolves against each worktree's own
top level and survives the move.

To confirm a hook really runs, without making a commit:

```sh
git -C <worktree> hook run pre-commit
```

## AI-assisted contributions

AI-assisted patches follow the same contributor, licensing and verification
policies established for this project, including the policy work tracked in
[#195](https://github.com/vibe-qc/vibe-qc/blob/main/CONTRIBUTING.md).

Disclose AI assistance in the pull request or merge request, including which
parts it helped produce. The contributor remains responsible for understanding
and reviewing the patch, its tests, citations and dependency choices. Check
referenced sources directly and remove generated claims that cannot be verified.

Include tests that exercise the reported failure and the corrected behavior.
For numerical changes, provide the relevant before/after values, reference
settings, tolerances and validation comparisons. Identify the exact commit and
build that were tested, and state missing checks or unrelated failures.

Public contributions follow the
[publication guide](https://vibe-qc.com/docs/github_publication.html).
Maintainers route accepted patches into canonical GitLab development with
the focused checks described above. Routine development work skips CI;
release candidates receive full CI. AI assistance does not change scientific
acceptance, attribution or the licensing terms below.

## Code style

- **C++17.** Four-space indent, brace on same line for control flow.
  Everything in `cpp/` lives under `namespace vibeqc { ... }`. Match
  the patterns in `cpp/src/rhf.cpp` / `cpp/src/integrals.cpp`.
- **Python 3.11+.** PEP 8 with four-space indent. Start every module
  with `from __future__ import annotations`. Type hints on public
  API; local helpers can skip them.
- Prefer editing existing modules over introducing new ones. If a
  new file is the right choice, follow the neighbors' layout.

## Commit messages

- Imperative mood, first line under 72 characters (e.g.
  `Fix SCF divergence on CH3 with SAD guess`).
- Longer rationale in the body if the *what* doesn't explain itself.
- Co-author trailers are fine for pair work.
- To flag a commit for inclusion in a patch release, add a
  `Patch-candidate:` trailer to the commit body (alongside
  `Signed-off-by:` / `Co-Authored-By:`) — values like `v0.7.x`,
  `v0.8.x`, `v0.8.0`, or comma-separated combinations. The release
  chat scans these when preparing patches. Do not tag `vX.Y.Z`,
  push to `release`, or open MRs against `release` directly —
  the authorized release owner performs those under the release process. See
  [branch and checkout policy](#branch-and-checkout-policy) and
  [`docs/release_process.md`](https://vibe-qc.com/docs/release_process.html) for the
  full convention.

## What we won't accept (for now)

- API changes that break public signatures without a deprecation
  path.
- New hard dependencies added without prior discussion — open an
  issue first.
- Changes that regress the test suite without a stated rationale and
  a plan to restore.

## Licensing

By submitting a patch, pull request, or any other contribution to
vibe-qc, you agree that:

1. Your contribution is licensed under the Mozilla Public License 2.0
   (the project license — see
   [`LICENSE`](https://github.com/vibe-qc/vibe-qc/blob/main/LICENSE)).
2. You grant the project owner (Michael F. Peintinger) the right to
   relicense your contribution under alternative terms, including a
   future commercial license, alongside the MPL 2.0 public license.
   You retain copyright.

This is a lightweight alternative to a formal Contributor License
Agreement. If you're not comfortable with (2), please open an issue
before contributing so we can discuss.

vibe-qc's compiled core links
[libint](https://github.com/evaleev/libint) (MPL 2.0),
[libxc](https://libxc.gitlab.io/) (MPL 2.0),
[Eigen](https://eigen.tuxfamily.org/) (MPL 2.0 / BSD-3),
[spglib](https://github.com/spglib/spglib) (BSD-3), and
[pybind11](https://github.com/pybind/pybind11) (BSD-3) — all
MPL-compatible.

### Private privacy policy

The portable guard checks privacy patterns without storing real private names. Operators
can add private literal terms through `VIBE_PRIVACY_TERMS_FILE` or the clone-local
`privacy.termsFile` Git setting. Use an absolute path to a UTF-8 file outside the
source checkout, with one literal term per line. Matching is case-insensitive;
a configured missing, empty or in-tree file blocks the check. Keep the private
terms file and resolved installation paths out of commits. The repository's
privacy tests load the same external policy when configured.

Site wrappers and deployment configuration are maintained separately in private
operations storage. Product changes must use portable examples and preserve the
existing generic installation and configuration APIs.


### Fast source privacy preflight

Enable `git config --local core.hooksPath .githooks` once in each clone.
Run `python3 .githooks/source_privacy.py --staged` before committing and
`python3 .githooks/source_privacy.py --tree HEAD` before requesting a release
candidate build. Candidate qualification runs the current-tree check before the numerical
build. Ordinary development pushes skip CI and rely on the local guards.

All tracked text is covered, including instructions and hooks. Keep
operator logs, machine details and private configuration outside product checkouts.
Use `VIBE_PRIVACY_TERMS_FILE` or local Git `privacy.termsFile` for an external
UTF-8 file with one private literal per line. A configured empty, missing or
in-tree file blocks the check. Canonical CI supplies this file from private CI
configuration; the public repository contains no private term list.

The two public maintainer addresses and reserved documentation domains are
allowed. Public upstream attribution or syntax that resembles a private value
requires independent, exact-context review in `.githooks/public_content_review.json`.
Do not approve private values or generate a blanket baseline. Changing the text
invalidates its exception. Binary assets and historical public objects remain
covered by the automatic publication gate; this fast check does not replace it.
