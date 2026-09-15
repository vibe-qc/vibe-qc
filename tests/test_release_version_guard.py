"""Release-commit version guard (``.githooks/commit-msg``).

The hook refuses a ``release: vX.Y.Z`` commit unless, in the tree being
committed, ``pyproject.toml`` ``[project] version`` equals ``X.Y.Z`` and
``CHANGELOG.md`` carries a ``## [vX.Y.Z]`` dated-section header. It is the
mechanical backstop against the v0.11.2 / v0.11.3 defect, where the
manual version-bump step (docs/release_process.md "Cutting a patch
release" step 4) was skipped and the tags shipped with ``pyproject``
still at ``0.11.1`` — so those wheels self-report ``0.11.1`` via
``vibeqc.banner._compute_version`` → ``importlib.metadata.version``.

Like ``test_basis_no_maintainer_paths.py`` mirrors the ``pre-commit``
personal-info hook, this test drives the actual ``commit-msg`` hook
script through throwaway git repositories so its contract is pinned
regardless of per-clone hook activation. Note: CI on the build-runner
runner is collect-only (no test execution), so the *hook itself* — not
this test — is the gate that blocks a bad cut. This test locks the
hook's behaviour for anyone who runs the suite locally and documents the
contract.

Update this test in lockstep with ``.githooks/commit-msg`` if the
release-commit subject convention or the asserted invariants change.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".githooks" / "commit-msg"

# Neutralise the developer's global/system git config (a global
# ``commit.gpgsign = true`` or ``core.hooksPath`` would otherwise leak
# into the throwaway repos and make even the allow-cases behave oddly).
_ISOLATED_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t.t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t.t",
}

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not HOOK.exists(),
    reason="git not available or commit-msg hook missing",
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        env=_ISOLATED_ENV,
        capture_output=True,
        text=True,
    )


def _make_repo(tmp_path: Path, version: str, changelog_header: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    hooks = repo / "hooks"
    hooks.mkdir()
    shutil.copy(HOOK, hooks / "commit-msg")
    (hooks / "commit-msg").chmod(0o755)
    _git(repo, "config", "core.hooksPath", str(hooks))
    _git(repo, "config", "commit.gpgsign", "false")

    (repo / "pyproject.toml").write_text(
        "[build-system]\n"
        "requires = []\n\n"
        "[project]\n"
        'name = "vibe-qc"\n'
        f'version = "{version}"\n\n'
        "[tool.scikit-build]\n"
        'cmake.version = ">=3.20"\n',
        encoding="utf-8",
    )
    body = "# Changelog\n\n## [Unreleased]\n\n"
    if changelog_header:
        body += changelog_header + "\nentry\n"
    else:
        body += "entry\n"
    (repo / "CHANGELOG.md").write_text(body, encoding="utf-8")

    _git(repo, "add", "-A")
    return repo


def _commit(repo: Path, subject: str) -> subprocess.CompletedProcess:
    return _git(repo, "commit", "-m", subject)


# (label, pyproject_version, changelog_header, subject, should_block)
_CASES = [
    ("non-release subject", "9.9.9", "", "fix(scf): handle null overlap", False),
    ("good patch release", "1.2.3", "## [v1.2.3] — 2026-01-01", "release: v1.2.3", False),
    ("good minor + codename", "0.12.0", "## [v0.12.0] — 2026-02-02 — *Foo*",
     "release: v0.12.0 — Foo's Bar", False),
    ("version-bump skipped (the v0.11.2/.3 bug)", "0.11.1",
     "## [v0.11.2] — 2026-06-04", "release: v0.11.2", True),
    ("changelog header missing", "1.2.3", "", "release: v1.2.3", True),
    ("pyproject still .dev0", "1.2.3.dev0", "## [v1.2.3] — 2026-01-01",
     "release: v1.2.3", True),
    ("chore dev-bump does not fire", "0.11.2.dev0", "",
     "chore: bump main to 0.11.2.dev0 + land [v0.11.2]", False),
    ("release subject with .dev does not fire", "0.11.2.dev0", "",
     "release: v0.11.2.dev0", False),
]


@pytest.mark.parametrize(
    "label,version,changelog_header,subject,should_block",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_release_guard(tmp_path, label, version, changelog_header, subject, should_block):
    repo = _make_repo(tmp_path, version, changelog_header)
    result = _commit(repo, subject)

    if should_block:
        assert result.returncode != 0, (
            f"[{label}] expected the commit to be BLOCKED but it succeeded.\n"
            f"stderr:\n{result.stderr}"
        )
        # The commit must not exist.
        log = _git(repo, "log", "--oneline")
        assert log.returncode != 0 or log.stdout.strip() == "", (
            f"[{label}] commit was created despite the guard:\n{log.stdout}"
        )
        # Error must name the offending surface so the cutter can fix it.
        assert "pyproject.toml" in result.stderr or "CHANGELOG.md" in result.stderr
    else:
        assert result.returncode == 0, (
            f"[{label}] expected the commit to be ALLOWED but it was blocked.\n"
            f"stderr:\n{result.stderr}"
        )


def test_no_verify_bypasses_the_guard(tmp_path):
    """``--no-verify`` is the documented reviewed-exception escape hatch
    (CLAUDE.md § 2 / § 12); the guard must honour it like every hook."""
    repo = _make_repo(tmp_path, "0.11.1", "## [v0.11.2]")
    result = _git(repo, "commit", "--no-verify", "-m", "release: v0.11.2")
    assert result.returncode == 0, (
        "--no-verify should bypass the guard:\n" + result.stderr
    )
