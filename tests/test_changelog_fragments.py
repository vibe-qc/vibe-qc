"""The changelog fragment directory and its assembler.

These cover the property the fragment scheme exists for: two changes landing
independently must never touch the same file, and assembling must not disturb
released sections, which `changelog_guard.py` pins.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "assemble_changelog.py"
FRAGMENTS = ROOT / "changelog.d"
HOOKS = ROOT / ".githooks"


def _module():
    spec = importlib.util.spec_from_file_location("assemble_changelog", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_assembler_exists_and_imports():
    assert SCRIPT.is_file(), "scripts/assemble_changelog.py is missing"
    assert hasattr(_module(), "assemble")


def test_every_fragment_is_well_formed():
    """A fragment must read as changelog prose, not as a stray note."""
    mod = _module()
    for name, text in mod.fragments(FRAGMENTS):
        first = text.splitlines()[0]
        assert first.startswith("### ") or first.startswith("- "), (
            "%s starts with %r; a fragment must open with a '### ' heading or a "
            "'- ' bullet so it reads correctly once folded in" % (name, first[:40])
        )


def test_assembling_leaves_released_sections_byte_identical(tmp_path):
    """The guard pins released bodies. Assembly must not be able to break a pin."""
    mod = _module()
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "## [Unreleased]\n\n### Added\n\n- existing entry\n\n"
        "## [v1.2.3]\n\n### Fixed\n\n- a released entry that is pinned\n"
    )
    d = tmp_path / "changelog.d"
    d.mkdir()
    (d / "999-new.md").write_text("### Fixed: something new (#999)\n\nProse.\n")

    out = mod.assemble(changelog, d)
    released = out[out.index("## [v1.2.3]") :]
    assert released == "## [v1.2.3]\n\n### Fixed\n\n- a released entry that is pinned\n"
    assert "### Fixed: something new (#999)" in out[: out.index("## [v1.2.3]")]
    assert "- existing entry" in out


def test_assembly_is_deterministic_regardless_of_discovery_order(tmp_path):
    """Two fragments must fold in filename order, not filesystem order."""
    mod = _module()
    changelog = tmp_path / "CHANGELOG.md"
    base = "## [Unreleased]\n\n## [v1.0.0]\n\n- old\n"
    changelog.write_text(base)
    d = tmp_path / "changelog.d"
    d.mkdir()
    (d / "b-second.md").write_text("- second\n")
    (d / "a-first.md").write_text("- first\n")
    out = mod.assemble(changelog, d)
    assert out.index("- first") < out.index("- second")


def test_readme_is_not_treated_as_a_fragment(tmp_path):
    mod = _module()
    d = tmp_path / "changelog.d"
    d.mkdir()
    (d / "README.md").write_text("# Changelog fragments\n\nInstructions, not an entry.\n")
    assert mod.fragments(d) == []


@pytest.mark.skipif(not FRAGMENTS.is_dir(), reason="no changelog.d in this tree")
def test_the_real_fragment_directory_assembles():
    mod = _module()
    out = mod.assemble(ROOT / "CHANGELOG.md", FRAGMENTS)
    assert out.startswith("## [Unreleased]")


# .githooks/pre-commit runs the assembler's --check on the staged fragments.
# Agents push to main with [skip ci], so the test above only ever ran after a
# prose fragment was already on main.


def _env():
    env = os.environ.copy()
    for key in (
        "GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE", "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_EMAIL", "VIBE_PRIVACY_REQUIRE_TERMS", "VIBE_PRIVACY_TERMS_FILE",
    ):
        env.pop(key, None)
    return env


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], env=_env(), check=True)


def _hook_repo(tmp_path):
    """A disposable repository carrying this tree's hooks and assembler."""
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], env=_env(), check=True)
    _git(repo, "config", "user.name", "Example")
    _git(repo, "config", "user.email", "mpei@vibe-qc.com")
    shutil.copytree(HOOKS, repo / ".githooks", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "scripts").mkdir()
    shutil.copy2(SCRIPT, repo / "scripts" / SCRIPT.name)
    (repo / "changelog.d").mkdir()
    return repo


def _stage(repo, name, text):
    (repo / "changelog.d" / name).write_text(text)
    _git(repo, "add", "changelog.d/" + name)


def _pre_commit(repo):
    return subprocess.run(
        ["sh", str(repo / ".githooks" / "pre-commit")], cwd=repo, env=_env(),
        capture_output=True, text=True, timeout=60, check=False,
    )


def test_pre_commit_refuses_a_staged_prose_fragment(tmp_path):
    repo = _hook_repo(tmp_path)
    _stage(repo, "999-prose.md", "Fixed the thing, written as a stray note.\n")
    hook = _pre_commit(repo)
    check = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--fragments", str(repo / "changelog.d")],
        capture_output=True, text=True, check=False,
    )
    assert check.returncode == 1 and "999-prose.md: must start with" in check.stderr
    assert hook.returncode != 0
    assert check.stderr in hook.stderr


def test_pre_commit_accepts_headed_and_bulleted_fragments(tmp_path):
    repo = _hook_repo(tmp_path)
    _stage(repo, "999-headed.md", "### Fixed: something new (#999)\n\nProse.\n")
    _stage(repo, "999-bullet.md", "- a bullet entry\n")
    # The README opens with '# ', which a fragment may not; it is not a fragment.
    _stage(repo, "README.md", "# Changelog fragments\n\nInstructions, not an entry.\n")
    hook = _pre_commit(repo)
    assert hook.returncode == 0, hook.stderr


def test_pre_commit_checks_the_index_not_the_working_tree(tmp_path):
    repo = _hook_repo(tmp_path)
    entry = repo / "changelog.d" / "999-entry.md"
    _stage(repo, entry.name, "Prose that was staged.\n")
    entry.write_text("### Fixed: a heading only in the working tree\n")
    assert _pre_commit(repo).returncode != 0

    _stage(repo, entry.name, "### Fixed: the staged heading (#999)\n")
    entry.write_text("Prose only in the working tree.\n")
    hook = _pre_commit(repo)
    assert hook.returncode == 0, hook.stderr


def test_pre_commit_leaves_fragments_the_commit_does_not_touch(tmp_path):
    """A malformed fragment already committed must not block an unrelated commit."""
    repo = _hook_repo(tmp_path)
    _stage(repo, "998-old.md", "Prose committed before the hook existed.\n")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-q", "--no-verify", "-m", "old")
    _stage(repo, "999-new.md", "- a new entry\n")
    hook = _pre_commit(repo)
    assert hook.returncode == 0, hook.stderr
