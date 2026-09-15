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
| **General feedback, question, "is this a bug or am I doing it wrong?"** | Email **`mpei@vibe-qc.com`**. No tracker account needed; replies go to your address directly. |
| **Code-of-conduct violation** | Email **`mpei@vibe-qc.com`**. See [CODE_OF_CONDUCT.md](https://github.com/vibe-qc/vibe-qc/blob/main/CODE_OF_CONDUCT.md). |
| **You want to send a patch** | Read the rest of this file. |
| **You fixed something, or you want to check someone else's fix** | The issue it belongs to. Verification requests and verdicts are **comments on the issue**, never files in the repository — see [How a fix gets verified](#how-a-fix-gets-verified). |
| **You want to fund the project** | [GitHub Sponsors](https://github.com/sponsors/mpeintinger) (recurring monthly) or [Ko-fi](https://ko-fi.com/mpeintinger) (one-time). Full pitch + author bio + funding goals on the [support page](https://vibe-qc.com/docs/support.html). |

All contributors and maintainers are expected to follow the
[Code of Conduct](https://github.com/vibe-qc/vibe-qc/blob/main/CODE_OF_CONDUCT.md).

GitHub is the public contribution entry point; maintainers track accepted work
in the canonical GitLab project. Do not assume GitHub comments reach `bugctl`
until a maintainer posts an upstream issue link. Automated PR import is available
only after its bridge is enabled; the submission requirements below still apply.

## How a fix gets verified

Every fix to a wrong answer, a numerical result, or anything
release-blocking is checked by **someone other than its author** before the
issue closes. That is a project rule, not a formality: the author of a fix
is the person least able to see what it missed.

**The request and the verdict are comments on the issue.** When your fix
lands, comment on the issue it fixes with what a second person needs in
order to attack it:

* the claim, in one paragraph — what was wrong and why the fix is correct;
* the commit it landed as;
* the exact commands to run, copy-pasteable;
* the numbers you measured **before** the fix, so a verifier can reproduce
  the broken behaviour on the parent commit and confirm the diagnosis;
* any test failure that is present but unrelated, named, with why you
  believe it is unrelated;
* what you would attack if you were checking someone else's version of
  this — the assumption you are least sure of.

To check someone else's fix, say so in a comment on the issue, do the work,
and comment the verdict with the evidence you measured. Disagreeing with
the fix is a useful outcome, not a failed one.

**Do not put any of this in a file in the repository.** Contributors work
from different machines and not all of them can push, so a request written
as a file reaches its reader but their answer cannot come back. A comment
needs only an account. Historical request files are archived privately; all new verification
requests and verdicts belong on the issue.

Maintainer-side automation (the `bugctl` broker used by the project's own
agents) reads and writes the canonical GitLab issue comments and labels, so a
contributor commenting by hand and an agent using the tooling are
participating in the same queue. Human comments are not fixer-lease actions
and can be added after landing. An automated fixer instead posts the same
brief after its push succeeds and before `bugctl land`, because `land` closes
the lease that authorizes `bugctl progress`.

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

Work on the new `mpei/vibe-qc` repository. The old `mpei/vibeqc` repository
is a frozen historical archive. Every change lands on `main` first, with
local checks for the affected area. Rebase before pushing:

```sh
git pull --rebase origin main
git push origin main
```

Never force-push `main`, `release`, or release tags, and never amend a commit
that has already been pushed. When a rebase conflicts in a shared changelog
or handover, preserve both contributors' facts and review the merged result.
The agentic loop alone cuts tags and fast-forwards `release` from a tagged
commit on `main`; the vibe-qc release chat prepares and proves each release.
Website changes publish from `main` after their CI gate; core docs publish
from `release`. See [release process](https://github.com/vibe-qc/vibe-qc/blob/main/docs/release_process.md)
and [site publishing](https://github.com/vibe-qc/vibe-qc/blob/main/docs/site_publishing.md).

Use one clone per agent task so its branch, index, environment and generated
files are isolated. Do not reset or delete another task's changes. If an
existing worktree must be used, check its hook configuration as described
[below](#worktrees-need-the-setting-applied-per-worktree).

## Personal information

Do not commit personal home-directory paths or unrelated employer
information. Use `~/`, `/home/USER/`, or `<vibe-qc-checkout>` placeholders in
examples. The pre-commit hook checks staged additions for concrete macOS
and Linux home paths and the project's employer marker. Its narrow
`runner`, `root`, and `user` allowlist exists for CI accounts and generic
fixtures; it is not permission to add personal data under another name.
The hook's pattern and allowlist are mirrored by
`tests/test_basis_no_maintainer_paths.py`.

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
record that authorization and the reason in the commit message.

## Before you open a merge request

1. Confirm no existing issue already covers the change — if there is
   one, comment on it so we can coordinate.
2. Run every test lane affected by the change and confirm it passes:
   ```sh
   .venv/bin/python scripts/test_gate/run_full_suite.py \
       --wt "$PWD" --py .venv/bin/python --lane <affected-lane>
   ```
3. If your change touches documentation, build the Sphinx docs
   locally and skim the affected pages:
   ```sh
   sphinx-build -b html docs/ docs/_build/html
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

Affected lanes are the merge-request gate. Use
`scripts/test_gate/run_full_suite.py --list-lanes` and
`docs/developer_test_lanes.md` to select them. The recommended layering is:

```sh
# 1) Non-mean-field solver tests — fast, no native rebuild needed.
.venv/bin/python -m pytest \
    tests/test_solvers_active_space_api.py \
    tests/test_solvers_dmrg_fixed_n.py \
    tests/test_solvers_v2rdm_constraints.py \
    tests/test_solvers_v2rdm_tc.py \
    tests/test_solvers_integration.py \
    tests/test_solvers_fci_parity.py

# 2) QVF writer + writer-to-viewer integration. The viewer leg is
#    skipped automatically if vibe-view isn't installed; the strict
#    writer-side schema asserts always run.
.venv/bin/python -m pytest \
    tests/test_qvf_writer.py \
    tests/test_qvf_writer_to_viewer.py

# 3) Packaging / extras metadata coherence — pure metadata, no install.
.venv/bin/python -m pytest tests/test_packaging_extras.py

# 4) Every affected lane. Add full-fast only for shared infrastructure.
.venv/bin/python scripts/test_gate/run_full_suite.py \
    --wt "$PWD" --py .venv/bin/python --lane <affected-lane>
```

List the exact lanes and SHA tested in the merge-request description.
Do not claim full-inventory success unless `full-fast` and `slow-nightly`
actually ran on that SHA.

## Pre-commit hook (one-time setup)

After cloning, point git at the tracked `.githooks/` directory so the
repo's commit guards run on every commit:

```sh
git config --local core.hooksPath .githooks
```

This single setting activates every hook in `.githooks/` (git resolves
one hooks directory for all hook types):

- **`pre-commit`** — refuses commits whose staged additions contain
  absolute paths into the author's home directories or other
  personal-info patterns ([personal information](#personal-information)). It then runs
  `.githooks/check_no_em_dashes.py`, which refuses staged `docs/*.md`
  carrying em or en dashes in prose (code fences, inline code, and
  inline math are exempt).
- **`commit-msg`** — refuses a `release: vX.Y.Z` commit unless
  `pyproject.toml` `[project] version` == `X.Y.Z` and a `## [vX.Y.Z]`
  CHANGELOG header exists. This is the backstop against tagging a
  release from a tree whose version bump was skipped — the defect that
  shipped v0.11.2 / v0.11.3 self-reporting `0.11.1`. See
  [docs/release_process.md](https://vibe-qc.com/docs/release_process.html) § "Cutting a
  release".

Fix hook failures before committing. A maintainer-authorized exception follows
the [documentation and hook policy](#documentation-prose) above.

### Worktrees need the setting applied per worktree

Parallel agent tasks get **one clone each**, following the
[branch and checkout policy](#branch-and-checkout-policy). Worktrees still turn up in practice
(agent harnesses create them under `.claude/worktrees/`), and a worktree
does not necessarily inherit the setting above, so check before trusting
it.

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
  the agentic loop performs those, and (since 2026-05-15) GitLab
  branch / tag protection blocks them. See
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

The portable guard checks home paths without storing real user names. Operators
can add private literal terms through `VIBE_PRIVACY_TERMS_FILE` or the clone-local
`privacy.termsFile` Git setting. Use an absolute path to a UTF-8 file outside the
source checkout, with one literal term per line. Matching is case-insensitive;
a configured missing, empty or in-tree file blocks the check. Keep the private
terms file and resolved installation paths out of commits. The repository's
privacy tests load the same external policy when configured.

Site wrappers and deployment configuration are maintained separately in private
operations storage. Product changes must use portable examples and preserve the
existing generic installation and configuration APIs.
