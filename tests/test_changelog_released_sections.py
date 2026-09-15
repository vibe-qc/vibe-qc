"""Released CHANGELOG.md sections stay as released unless an amendment is pinned (#229).

#227 found entries for post-v0.17.1 work inside the already released
``[v0.17.0]`` and ``[v0.17.1]`` sections, and fixing it turned up four more
misattributions that had already shipped in the v0.17.0 and v0.17.1 tags. They
arrived through ordinary rebases: a promotion commit relabels a header, and a
later rebase re-applies an ``[Unreleased]`` hunk at the same context under the
now-released header, without a conflict. A heading comparison misses an entry
under a generic ``### Added`` heading and a paragraph appended to an existing
entry, so ``scripts/test_gate/changelog_guard.py`` pins each released section's
whole body.

The first test runs the guard on this repository's CHANGELOG.md and needs no
git history, so it holds in a shallow CI clone. The rest pin the guard's
contract on synthetic changelogs, and on throwaway git repositories for
``audit`` and ``.githooks/pre-push``.
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
GUARD_PATH = ROOT / "scripts" / "test_gate" / "changelog_guard.py"
PINS_PATH = ROOT / "scripts" / "test_gate" / "changelog_pins.toml"
HOOK = ROOT / ".githooks" / "pre-push"
Z40 = "0" * 40


def _load_guard():
    spec = importlib.util.spec_from_file_location("changelog_guard", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard()

BASE = """## [Unreleased]

### Added: something new

## [v1.1.0] - 2026-02-01 - *B*

### Added

- a released entry

### Fixed: a released fix

It was broken.

## [v1.0.0] - 2026-01-01 - *A*

### Added

- the first release
"""


def _pinned(text: str) -> dict:
    return guard.pin(text, {}, [], all_unpinned=True)


def test_repository_changelog_matches_its_pins():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    problems = guard.check(text, guard.load_pins(PINS_PATH))
    assert problems == [], "\n".join(problems)


def test_untouched_changelog_passes():
    assert guard.check(BASE, _pinned(BASE)) == []


def test_unreleased_edits_pass():
    pins = _pinned(BASE)
    edited = BASE.replace(
        "### Added: something new\n",
        "### Added: something new\n\nMore text.\n\n### Fixed: another\n",
    )
    assert guard.check(edited, pins) == []


@pytest.mark.parametrize(
    "old,new",
    [
        ("- a released entry\n", "- a released entry\n- post-release work\n"),
        ("It was broken.\n", "It was broken.\n\nA later change extended it.\n"),
        (
            "## [v1.1.0] - 2026-02-01 - *B*\n\n",
            "## [v1.1.0] - 2026-02-01 - *B*\n\n### Fixed: landed after the tag\n\nText.\n\n",
        ),
        ("### Fixed: a released fix\n\nIt was broken.\n\n", ""),
    ],
    ids=["generic-heading-entry", "appended-paragraph", "misfiled-under-header", "removed-entry"],
)
def test_edits_to_a_released_section_fail(old, new):
    pins = _pinned(BASE)
    assert old in BASE
    problems = guard.check(BASE.replace(old, new, 1), pins)
    assert len(problems) == 1 and "[v1.1.0]" in problems[0], problems


def test_newly_released_section_needs_a_pin():
    pins = _pinned(BASE)
    promoted = BASE.replace(
        "## [Unreleased]\n\n",
        "## [Unreleased]\n\nNothing yet.\n\n## [v1.2.0] - 2026-03-01 - *C*\n\n",
        1,
    )
    problems = guard.check(promoted, pins)
    assert len(problems) == 1 and "[v1.2.0]" in problems[0] and "no pin" in problems[0]
    assert guard.check(promoted, guard.pin(promoted, pins, ["v1.2.0"])) == []


def test_changing_a_pin_needs_a_stated_amendment():
    pins = _pinned(BASE)
    amended_text = BASE.replace("It was broken.\n", "It was broken; the fix is described here.\n")
    with pytest.raises(guard.GuardError, match="--amended"):
        guard.pin(amended_text, pins, ["v1.1.0"])
    repinned = guard.pin(amended_text, pins, ["v1.1.0"], amended="clarified wording")
    assert repinned["v1.1.0"]["amended"] == "clarified wording"
    assert guard.check(amended_text, repinned) == []
    again = amended_text.replace("described here", "described here in full")
    twice = guard.pin(again, repinned, ["v1.1.0"], amended="second clarification")
    assert twice["v1.1.0"]["amended"] == "clarified wording; second clarification"


def test_stale_pin_and_duplicate_section_fail():
    pins = _pinned(BASE)
    without_first_release = BASE.split("## [v1.0.0]")[0]
    assert any(
        "[v1.0.0]" in problem and "names no section" in problem
        for problem in guard.check(without_first_release, pins)
    )
    duplicated = BASE + "\n## [v1.0.0] - 2026-01-01 - *A*\n\nagain\n"
    assert any("more than once" in problem for problem in guard.check(duplicated, pins))


def test_pins_round_trip_through_the_file(tmp_path):
    pins = _pinned(BASE)
    quoted = BASE.replace("It was broken.\n", 'It was "broken".\n')
    pins = guard.pin(quoted, pins, ["v1.1.0"], amended='quotes " and \\ backslashes survive')
    path = tmp_path / "pins.toml"
    path.write_text(guard.dump_pins(pins, guard.section_order(quoted)), encoding="utf-8")
    assert guard.load_pins(path) == pins


_ISOLATED_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t.t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t.t",
}

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, env=_ISOLATED_ENV, capture_output=True, text=True,
        check=False,
    )


def _repo(tmp_path: Path, text: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "config", "tag.gpgsign", "false")
    (repo / "CHANGELOG.md").write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    assert _git(repo, "commit", "-q", "-m", "init").returncode == 0
    return repo


@needs_git
def test_audit_requires_a_reason_where_a_section_differs_from_its_tag(tmp_path):
    repo = _repo(tmp_path, BASE)
    assert _git(repo, "tag", "v1.1.0").returncode == 0
    later = BASE.replace("It was broken.\n", "It was broken.\n\nLater text.\n")
    silent = _pinned(later)
    problems, checked, skipped = guard.audit(later, silent, repo)
    assert checked == ["v1.1.0"] and skipped == ["v1.0.0"]
    assert len(problems) == 1 and "[v1.1.0]" in problems[0]
    explained = guard.pin(later, silent, ["v1.1.0"], amended="documented follow-up")
    assert guard.audit(later, explained, repo)[0] == []


def _hook(repo: Path, line: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = {**_ISOLATED_ENV, "VIBEQC_PRE_PUSH_PYTHON": sys.executable, **(extra_env or {})}
    return subprocess.run(
        ["sh", str(HOOK), "origin", "ssh://example.invalid/repo"],
        cwd=repo,
        env=env,
        input=line,
        capture_output=True,
        text=True,
        check=False,
    )


@needs_git
def test_pre_push_hook_checks_main_and_release_candidates(tmp_path):
    repo = _repo(tmp_path, BASE)
    (repo / "scripts" / "test_gate").mkdir(parents=True)
    shutil.copy(GUARD_PATH, repo / "scripts" / "test_gate" / "changelog_guard.py")
    (repo / "scripts" / "test_gate" / "changelog_pins.toml").write_text(
        guard.dump_pins(_pinned(BASE), guard.section_order(BASE)), encoding="utf-8"
    )
    _git(repo, "add", "-A")
    assert _git(repo, "commit", "-q", "-m", "guard").returncode == 0
    good = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "CHANGELOG.md").write_text(
        BASE.replace("- a released entry\n", "- a released entry\n- misfiled\n"),
        encoding="utf-8",
    )
    assert _git(repo, "commit", "-q", "-am", "misfile").returncode == 0
    bad = _git(repo, "rev-parse", "HEAD").stdout.strip()

    accepted = _hook(repo, f"refs/heads/main {good} refs/heads/main {Z40}\n")
    assert accepted.returncode == 0, accepted.stderr
    refused = _hook(repo, f"refs/heads/main {bad} refs/heads/main {good}\n")
    assert refused.returncode != 0 and "[v1.1.0]" in refused.stderr
    candidate = _hook(repo, f"{bad} {bad} refs/heads/release-candidate/v1.2.0 {Z40}\n")
    assert candidate.returncode != 0
    topic = _hook(repo, f"refs/heads/topic {bad} refs/heads/topic {Z40}\n")
    assert topic.returncode == 0, topic.stderr
    deletion = _hook(repo, f"(delete) {Z40} refs/heads/main {good}\n")
    assert deletion.returncode == 0, deletion.stderr


@needs_git
def test_pre_push_hook_ignores_a_tree_that_predates_the_guard(tmp_path):
    repo = _repo(tmp_path, BASE)
    sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    result = _hook(repo, f"refs/heads/main {sha} refs/heads/main {Z40}\n")
    assert result.returncode == 0, result.stderr


POLICY_FILES = (
    "scripts/test_gate/changelog_guard.py",
    "scripts/test_gate/changelog_pins.toml",
    "CHANGELOG.md",
)


def _commit_all(repo: Path, message: str) -> str:
    assert _git(repo, "add", "-A").returncode == 0
    result = _git(repo, "commit", "-q", "-m", message)
    assert result.returncode == 0, result.stderr
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _guarded_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = _repo(tmp_path, BASE)
    historical = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "scripts/test_gate").mkdir(parents=True)
    shutil.copy(GUARD_PATH, repo / POLICY_FILES[0])
    (repo / POLICY_FILES[1]).write_text(
        guard.dump_pins(_pinned(BASE), guard.section_order(BASE)), encoding="utf-8"
    )
    return repo, historical, _commit_all(repo, "introduce policy")


@needs_git
@pytest.mark.parametrize("missing", [
    [POLICY_FILES[0]], [POLICY_FILES[1]], [POLICY_FILES[2]],
    list(POLICY_FILES[:2]), list(POLICY_FILES),
], ids=["guard", "pins", "changelog", "guard-and-pins", "all-inputs"])
@pytest.mark.parametrize("destination", ["main", "release-candidate/v1.2.0"])
def test_pre_push_refuses_missing_committed_policy_inputs(tmp_path, missing, destination):
    repo, _, good = _guarded_repo(tmp_path)
    for name in missing:
        (repo / name).unlink()
    bad = _commit_all(repo, "accidental policy removal")
    # A repaired but uncommitted working tree cannot hide the bad pushed tree.
    for name in missing:
        (repo / name).write_text(_git(repo, "show", f"{good}:{name}").stdout)
    result = _hook(repo, f"refs/heads/topic {bad} refs/heads/{destination} {good}\n")
    assert result.returncode != 0, result.stderr
    assert missing[0] in result.stderr


@needs_git
def test_pre_push_uses_intact_commit_despite_missing_working_policy(tmp_path):
    repo, historical, good = _guarded_repo(tmp_path)
    for name in POLICY_FILES:
        (repo / name).unlink()
    for sha in [good, historical]:
        result = _hook(repo, f"refs/heads/topic {sha} refs/heads/main {Z40}\n")
        assert result.returncode == 0, result.stderr


@needs_git
@pytest.mark.parametrize("bad_first", [True, False])
def test_pre_push_checks_every_protected_row_and_keeps_failure(tmp_path, bad_first):
    repo, _, good = _guarded_repo(tmp_path)
    (repo / POLICY_FILES[1]).unlink()
    bad = _commit_all(repo, "remove pins")
    valid = f"refs/heads/good {good} refs/heads/main {Z40}\n"
    invalid = f"refs/heads/bad {bad} refs/heads/release-candidate/v1.2.0 {Z40}\n"
    lines = invalid + valid if bad_first else valid + invalid
    # A deletion or unrelated ref is exempt even if its other object is absent.
    lines += f"(delete) {Z40} refs/heads/main {'f' * 40}\n"
    lines += f"refs/heads/topic {'f' * 40} refs/heads/topic {Z40}\n"
    result = _hook(repo, lines)
    assert result.returncode != 0
    assert POLICY_FILES[1] in result.stderr and "refs/heads/bad" in result.stderr


@needs_git
@pytest.mark.parametrize("destination", ["main", "release-candidate/v1.2.0"])
def test_pre_push_unknown_protected_commit_is_not_historical(tmp_path, destination):
    repo = _repo(tmp_path, BASE)
    result = _hook(repo, f"refs/heads/topic {'f' * 40} refs/heads/{destination} {Z40}\n")
    assert result.returncode != 0
    assert "commit" in result.stderr


@needs_git
@pytest.mark.parametrize("operation", ["show", "ls-tree", "log"])
def test_pre_push_git_read_failure_is_not_historical(tmp_path, operation):
    repo, historical, good = _guarded_repo(tmp_path)
    wrapper = tmp_path / "bin"
    wrapper.mkdir()
    (wrapper / "git").write_text(
        '#!/bin/sh\n'
        'if [ "$1" = "$FAIL_GIT_OPERATION" ]; then exit 74; fi\n'
        'exec "$REAL_GIT" "$@"\n'
    )
    (wrapper / "git").chmod(0o755)
    sha = historical if operation == "log" else good
    result = _hook(repo, f"refs/heads/topic {sha} refs/heads/main {Z40}\n", {
        "PATH": str(wrapper) + os.pathsep + os.environ["PATH"],
        "REAL_GIT": shutil.which("git"), "FAIL_GIT_OPERATION": operation,
    })
    assert result.returncode != 0
    assert "pre-push:" in result.stderr
    if operation == "show":
        assert POLICY_FILES[0] in result.stderr


@needs_git
def test_pre_push_guard_symlink_is_not_executed_as_policy(tmp_path):
    repo, _, good = _guarded_repo(tmp_path)
    (repo / POLICY_FILES[0]).unlink()
    # git show emits the link target, which happens to be valid no-op Python.
    (repo / POLICY_FILES[0]).symlink_to("pass")
    bad = _commit_all(repo, "accidentally replace guard with link")
    result = _hook(repo, f"refs/heads/topic {bad} refs/heads/main {good}\n")
    assert result.returncode != 0
    assert POLICY_FILES[0] in result.stderr


@needs_git
def test_pre_push_shallow_absence_cannot_prove_pre_guard_history(tmp_path):
    repo, _, good = _guarded_repo(tmp_path)
    # Synthetic shallow boundary; no source clone or network operation.
    (repo / ".git/shallow").write_text(good + "\n")
    accepted = _hook(repo, f"refs/heads/topic {good} refs/heads/main {Z40}\n")
    assert accepted.returncode == 0, accepted.stderr
    for name in POLICY_FILES:
        (repo / name).unlink()
    bad = _commit_all(repo, "remove policy beyond shallow boundary")
    (repo / ".git/shallow").write_text(bad + "\n")
    refused = _hook(repo, f"refs/heads/topic {bad} refs/heads/main {Z40}\n")
    assert refused.returncode != 0
    assert "shallow" in refused.stderr


@needs_git
def test_pre_push_policy_introduction_on_merged_branch_is_not_historical(tmp_path):
    repo, historical, good = _guarded_repo(tmp_path)
    assert _git(repo, "checkout", "-q", "-b", "integration", historical).returncode == 0
    # The resulting tree omits policy, but one reachable parent introduced it.
    assert _git(repo, "merge", "-q", "--no-ff", "-s", "ours", good, "-m", "merge").returncode == 0
    merged = _git(repo, "rev-parse", "HEAD").stdout.strip()
    result = _hook(repo, f"refs/heads/integration {merged} refs/heads/main {Z40}\n")
    assert result.returncode != 0
    assert POLICY_FILES[0] in result.stderr


@needs_git
def test_pre_push_policy_process_cannot_consume_following_push_rows(tmp_path):
    repo, _, good = _guarded_repo(tmp_path)
    (repo / POLICY_FILES[1]).unlink()
    bad = _commit_all(repo, "remove pins")
    launcher = tmp_path / "python-launcher"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "sys.stdin.read()\n"
        "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    launcher.chmod(0o755)
    result = _hook(
        repo,
        f"refs/heads/good {good} refs/heads/main {Z40}\n"
        f"refs/heads/bad {bad} refs/heads/release-candidate/v1.2.0 {Z40}\n",
        {"VIBEQC_PRE_PUSH_PYTHON": str(launcher)},
    )
    assert result.returncode != 0
    assert POLICY_FILES[1] in result.stderr and "refs/heads/bad" in result.stderr
