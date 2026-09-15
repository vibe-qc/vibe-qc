"""Tests for vibe_basis.transports.local.LocalTransport.

LocalTransport runs commands synchronously, so these tests use a
mock runner (no real subprocess) plus one end-to-end test with a
trivial real command (``true`` / ``false``) to confirm the
subprocess path actually works.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

import pytest

from vibe_basis.transports import JobHandle, LocalTransport, TransportError


@dataclass
class _FakeProc:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


class _FakeLocalRunner:
    def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.calls: list[tuple] = []
        self._proc = _FakeProc(stdout=stdout, stderr=stderr, returncode=returncode)

    def __call__(self, command, cwd):
        self.calls.append((tuple(command), cwd))
        return self._proc


def test_submit_runs_command_and_reports_completed(tmp_path):
    runner = _FakeLocalRunner(returncode=0, stdout="ok")
    lt = LocalTransport(runner=runner)
    handle = lt.submit(tmp_path, ["bash", "run.sh", "mgo.d12"])
    assert isinstance(handle, JobHandle)
    assert handle.job_id.startswith("local-")
    assert lt.poll(handle) == "completed"
    # stdout/stderr persisted next to outputs.
    assert (tmp_path / f"{handle.job_id}.stdout").read_text() == "ok"


def test_submit_reports_failed_on_nonzero_returncode(tmp_path):
    runner = _FakeLocalRunner(returncode=1, stderr="boom")
    lt = LocalTransport(runner=runner)
    handle = lt.submit(tmp_path, ["false"])
    assert lt.poll(handle) == "failed"
    assert (tmp_path / f"{handle.job_id}.stderr").read_text() == "boom"


def test_submit_raises_on_missing_work_dir(tmp_path):
    lt = LocalTransport(runner=_FakeLocalRunner())
    with pytest.raises(TransportError):
        lt.submit(tmp_path / "does-not-exist", ["true"])


def test_poll_unknown_job_raises(tmp_path):
    lt = LocalTransport(runner=_FakeLocalRunner())
    stray = JobHandle("local-deadbeef", tmp_path, ("true",))
    with pytest.raises(TransportError):
        lt.poll(stray)


def test_fetch_returns_work_dir(tmp_path):
    runner = _FakeLocalRunner()
    lt = LocalTransport(runner=runner)
    handle = lt.submit(tmp_path, ["true"])
    # Local outputs aren't copied — fetch just points at work_dir.
    assert lt.fetch(handle, tmp_path / "ignored_dest") == tmp_path


def test_wait_returns_immediately_for_synchronous_job(tmp_path):
    runner = _FakeLocalRunner(returncode=0)
    lt = LocalTransport(runner=runner)
    handle = lt.submit(tmp_path, ["true"])
    # Synchronous: by submit's return the job is terminal, so wait
    # polls once and returns without sleeping.
    slept: list[float] = []
    state = lt.wait(handle, sleep=slept.append, clock=lambda: 0.0)
    assert state == "completed"
    assert slept == []


@pytest.mark.skipif(shutil.which("true") is None, reason="needs /bin/true")
def test_end_to_end_real_subprocess(tmp_path):
    """One real-subprocess test: confirm the default runner path
    (subprocess.run) actually executes and classifies exit codes."""
    lt = LocalTransport()  # default runner — real subprocess.run
    ok = lt.submit(tmp_path, ["true"])
    assert lt.poll(ok) == "completed"
    bad = lt.submit(tmp_path, ["false"])
    assert lt.poll(bad) == "failed"
