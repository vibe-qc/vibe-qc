"""Regressions for ``git_clone_pinned`` in ``scripts/_verify_source.sh`` (#241).

Every case runs the real helper under Bash against a throwaway upstream
repository on the local filesystem: no network, no native build. What is
pinned:

* the commit-SHA check that refuses a moved tag is unchanged;
* ``VIBEQC_GIT_REFERENCE_DIR`` lets a build take the pinned commit from a
  local repository when upstream is unreachable, and only from one that
  already holds that exact commit;
* an upstream failure is retried a bounded number of times and then reported
  with a pointer at the local-source override;
* a clone that stalls is taken down by a SIGKILL-escalating deadline even
  when it ignores SIGTERM, and Ctrl-C on the build script still stops it.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts" / "_verify_source.sh"
BASH = "/bin/bash" if Path("/bin/bash").is_file() else shutil.which("bash")
GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(
    BASH is None or GIT is None or os.name != "posix" or shutil.which("pgrep") is None,
    reason="git_clone_pinned needs Bash, git and pgrep on a POSIX host",
)

_GIT_IDENTITY = ("-c", "user.email=t@example.invalid", "-c", "user.name=t")


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        [GIT, *_GIT_IDENTITY, *args], cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


@pytest.fixture
def upstream(tmp_path: Path) -> dict[str, str]:
    """A two-commit repository with tags ``v0`` and ``v1``."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git("init", "-q", str(repo))
    _git("commit", "-q", "--allow-empty", "-m", "one", cwd=repo)
    _git("tag", "v0", cwd=repo)
    _git("commit", "-q", "--allow-empty", "-m", "two", cwd=repo)
    _git("tag", "v1", cwd=repo)
    return {
        "path": str(repo),
        "url": repo.as_uri(),
        "v0": _git("rev-parse", "v0", cwd=repo),
        "v1": _git("rev-parse", "v1", cwd=repo),
    }


UNREACHABLE = "file:///nonexistent/evaleev/libint.git"


def _env(**overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("VIBEQC_GIT_") or name.startswith("GIT_HTTP_LOW_SPEED"):
            env.pop(name)
    # Fast failure paths by default; individual cases widen what they test.
    env.setdefault("VIBEQC_GIT_CLONE_ATTEMPTS", "1")
    env.setdefault("VIBEQC_GIT_CLONE_RETRY_DELAY", "0")
    env.update(overrides)
    return env


def _clone(
    tmp_path: Path, url: str, tag: str, expected: str, dest: str, **env: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            BASH,
            "-c",
            f'. "{HELPER}"; git_clone_pinned "$@"',
            "_",
            url,
            tag,
            expected,
            dest,
        ],
        cwd=tmp_path,
        env=_env(**env),
        text=True,
        capture_output=True,
        timeout=120,
    )


def _pids_matching(pattern: str) -> list[str]:
    result = subprocess.run(["pgrep", "-f", pattern], text=True, capture_output=True)
    return [line for line in result.stdout.split() if line]


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return predicate()


# ---------------------------------------------------------------------------
# The commit pin.
# ---------------------------------------------------------------------------


def test_plain_clone_lands_the_pinned_commit(tmp_path: Path, upstream: dict[str, str]) -> None:
    result = _clone(tmp_path, upstream["url"], "v1", upstream["v1"], "dest")
    assert result.returncode == 0, result.stderr
    dest = tmp_path / "dest"
    assert _git("rev-parse", "HEAD", cwd=dest) == upstream["v1"]
    assert _git("rev-parse", "--is-shallow-repository", cwd=dest) == "true"


def test_moved_tag_is_refused_and_the_checkout_removed(
    tmp_path: Path, upstream: dict[str, str]
) -> None:
    # The pin says v0's commit, the tag now points at v1's: refuse.
    result = _clone(tmp_path, upstream["url"], "v1", upstream["v0"], "dest")
    assert result.returncode != 0
    assert "commit-SHA mismatch" in result.stderr
    assert f"expected: {upstream['v0']}" in result.stderr
    assert f"got:      {upstream['v1']}" in result.stderr
    assert not (tmp_path / "dest").exists()


# ---------------------------------------------------------------------------
# VIBEQC_GIT_REFERENCE_DIR: a local source for the pinned commit.
# ---------------------------------------------------------------------------


def _stage_local_source(root: Path, layout: str, upstream: dict[str, str], tag: str) -> Path:
    """Clone upstream at ``tag`` into ``root/<layout>``; ``*.git`` means bare."""
    target = root / layout
    target.parent.mkdir(parents=True, exist_ok=True)
    args = ["clone", "-q", "--depth", "1", "--branch", tag]
    if layout.endswith(".git"):
        args.append("--bare")
    _git(*args, upstream["url"], str(target))
    return target


@pytest.mark.parametrize(
    "layout",
    [
        "libint/src",  # another vibe-qc checkout's third_party/
        "libint.git",  # a bare mirror
        "libint",  # a plain clone named after the repository
        "libint-src",  # libecpint's sub-dependency layout
    ],
)
def test_local_source_serves_the_pinned_commit_without_upstream(
    tmp_path: Path, upstream: dict[str, str], layout: str
) -> None:
    ref_dir = tmp_path / "third_party"
    _stage_local_source(ref_dir, layout, upstream, "v1")
    result = _clone(
        tmp_path,
        UNREACHABLE,
        "v1",
        upstream["v1"],
        "dest",
        VIBEQC_GIT_REFERENCE_DIR=str(ref_dir),
    )
    assert result.returncode == 0, result.stderr
    assert "from local source" in result.stdout
    assert "does not appear to be a git repository" not in result.stderr
    dest = tmp_path / "dest"
    assert _git("rev-parse", "HEAD", cwd=dest) == upstream["v1"]
    # The result reads like an ordinary pinned clone of upstream.
    assert _git("remote", "get-url", "origin", cwd=dest) == UNREACHABLE
    assert _git("rev-parse", "v1", cwd=dest) == upstream["v1"]
    assert _git("rev-parse", "--is-shallow-repository", cwd=dest) == "true"


def test_local_source_dirs_are_colon_separated(
    tmp_path: Path, upstream: dict[str, str]
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    holder = tmp_path / "mirrors"
    _stage_local_source(holder, "libint.git", upstream, "v1")
    result = _clone(
        tmp_path,
        UNREACHABLE,
        "v1",
        upstream["v1"],
        "dest",
        VIBEQC_GIT_REFERENCE_DIR=f"{tmp_path / 'missing'}:{empty}:{holder}",
    )
    assert result.returncode == 0, result.stderr
    assert _git("rev-parse", "HEAD", cwd=tmp_path / "dest") == upstream["v1"]


def test_local_source_at_another_commit_is_refused_then_upstream_is_used(
    tmp_path: Path, upstream: dict[str, str]
) -> None:
    ref_dir = tmp_path / "third_party"
    # The candidate names derive from the URL, so the layout follows upstream's.
    source = _stage_local_source(ref_dir, "upstream/src", upstream, "v0")
    result = _clone(
        tmp_path,
        upstream["url"],
        "v1",
        upstream["v1"],
        "dest",
        VIBEQC_GIT_REFERENCE_DIR=str(ref_dir),
    )
    assert result.returncode == 0, result.stderr
    assert f"local source {source} does not contain {upstream['v1']}" in result.stderr
    assert "from local source" not in result.stdout
    assert _git("rev-parse", "HEAD", cwd=tmp_path / "dest") == upstream["v1"]


def test_local_source_at_another_commit_cannot_substitute_offline(
    tmp_path: Path, upstream: dict[str, str]
) -> None:
    ref_dir = tmp_path / "third_party"
    _stage_local_source(ref_dir, "libint/src", upstream, "v0")
    result = _clone(
        tmp_path,
        UNREACHABLE,
        "v1",
        upstream["v1"],
        "dest",
        VIBEQC_GIT_REFERENCE_DIR=str(ref_dir),
    )
    assert result.returncode != 0
    assert "does not contain" in result.stderr
    assert "giving up on" in result.stderr
    assert "VIBEQC_GIT_REFERENCE_DIR" in result.stderr
    assert not (tmp_path / "dest").exists()


def test_a_directory_inside_a_repository_is_not_a_source(
    tmp_path: Path, upstream: dict[str, str]
) -> None:
    # third_party/ of a vibe-qc checkout is inside that checkout's repository;
    # `git -C` there would walk up to the checkout. It must be skipped without
    # being reported as a source that lacks the commit.
    checkout = tmp_path / "checkout"
    (checkout / "third_party" / "upstream").mkdir(parents=True)
    _git("init", "-q", str(checkout))
    result = _clone(
        tmp_path,
        upstream["url"],
        "v1",
        upstream["v1"],
        "dest",
        VIBEQC_GIT_REFERENCE_DIR=str(checkout / "third_party"),
    )
    assert result.returncode == 0, result.stderr
    assert "does not contain" not in result.stderr
    assert "from local source" not in result.stdout
    assert _git("rev-parse", "HEAD", cwd=tmp_path / "dest") == upstream["v1"]


# ---------------------------------------------------------------------------
# Bounded retries and a deadline that terminates.
# ---------------------------------------------------------------------------


def test_unreachable_upstream_is_retried_a_bounded_number_of_times(
    tmp_path: Path, upstream: dict[str, str]
) -> None:
    result = _clone(
        tmp_path,
        UNREACHABLE,
        "v1",
        upstream["v1"],
        "dest",
        VIBEQC_GIT_CLONE_ATTEMPTS="3",
    )
    assert result.returncode != 0
    assert "attempt 1 of 3" in result.stderr
    assert "attempt 2 of 3" in result.stderr
    assert "attempt 3 of 3" not in result.stderr
    assert "giving up on" in result.stderr and "after 3 attempt(s)" in result.stderr
    assert not (tmp_path / "dest").exists()


def _fake_git(tmp_path: Path, on_clone: str) -> Path:
    """A ``git`` on PATH whose ``clone`` runs ``on_clone`` and never returns."""
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    script = fake_bin / "git"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "for arg in \"$@\"; do\n"
        '    if [ "$arg" = clone ]; then\n'
        f"        {on_clone}\n"
        "    fi\n"
        "done\n"
        f'exec "{GIT}" "$@"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return fake_bin


def test_deadline_kills_a_clone_that_ignores_sigterm(
    tmp_path: Path, upstream: dict[str, str]
) -> None:
    fake_bin = _fake_git(tmp_path, "trap '' TERM; while :; do :; done")
    marker = str(fake_bin / "git")
    started = time.monotonic()
    result = _clone(
        tmp_path,
        upstream["url"],
        "v1",
        upstream["v1"],
        "dest",
        PATH=f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        VIBEQC_GIT_CLONE_TIMEOUT="1",
        VIBEQC_GIT_CLONE_KILL_GRACE="1",
    )
    elapsed = time.monotonic() - started
    assert result.returncode != 0
    assert "no completion after 1s" in result.stderr
    assert "giving up on" in result.stderr and "(last exit 124)" in result.stderr
    assert elapsed < 30, elapsed
    assert _wait_until(lambda: not _pids_matching(marker)), "the stalled clone survived"
    assert not (tmp_path / "dest").exists()


def test_deadline_passes_the_exit_status_through_and_cancels_its_watchdog(
    tmp_path: Path,
) -> None:
    started = time.monotonic()
    result = subprocess.run(
        [
            BASH,
            "-c",
            f'. "{HELPER}"; rc=0; _vqc_run_with_deadline 977 1 bash -c "exit 7" || rc=$?; '
            "echo rc=$rc",
        ],
        cwd=tmp_path,
        env=_env(),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "rc=7" in result.stdout
    assert time.monotonic() - started < 10
    assert _wait_until(lambda: not _pids_matching("sleep 977"), timeout=5), (
        "the watchdog's sleeper was left behind"
    )


def test_interrupting_the_build_script_stops_the_clone(
    tmp_path: Path, upstream: dict[str, str]
) -> None:
    fake_bin = _fake_git(tmp_path, "while :; do sleep 0.2; done")
    marker = str(fake_bin / "git")
    env = _env(
        PATH=f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        VIBEQC_GIT_CLONE_TIMEOUT="600",
    )
    proc = subprocess.Popen(
        [
            BASH,
            "-c",
            f'. "{HELPER}"; git_clone_pinned "$@"',
            "_",
            upstream["url"],
            "v1",
            upstream["v1"],
            "dest",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert _wait_until(lambda: bool(_pids_matching(marker))), "the fake clone never started"
        proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    # Either the re-delivered SIGINT ended Bash (-2) or the relay's exit did.
    assert proc.returncode in (130, -signal.SIGINT), (proc.returncode, stderr)
    assert _wait_until(lambda: not _pids_matching(marker)), "the clone outlived Ctrl-C"
    assert not _pids_matching("sleep 600"), "the deadline watchdog was left behind"
