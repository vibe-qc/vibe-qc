"""Tests for vibe_basis.transports.vq.VqTransport.

The ``vq`` CLI is mocked via an injected runner — these tests run
without vq installed and without compute-host-d access. The canned
stdout shapes match the vq submit/status/fetch formats used in
the CRYSTAL14-on-compute-host-d recipe (README + chat memory).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from vibe_basis.transports import (
    JobHandle,
    JobSubmitError,
    JobTimeoutError,
    TransportError,
    VqTransport,
)


@dataclass
class _FakeProc:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


class _FakeVqRunner:
    """Records argv calls; returns canned output keyed by subcommand.

    ``status_sequence`` is consumed one entry per ``vq status`` call,
    sticking on the last entry — so a [running, running, completed]
    sequence exercises the wait-loop.
    """

    def __init__(
        self,
        *,
        submit_stdout: str = "job-abc123\n",
        submit_rc: int = 0,
        status_sequence: list[str] | None = None,
        fetch_rc: int = 0,
    ) -> None:
        self.calls: list[list[str]] = []
        self.submit_stdout = submit_stdout
        self.submit_rc = submit_rc
        self.status_sequence = status_sequence or ["state: completed"]
        self._status_idx = 0
        self.fetch_rc = fetch_rc

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append(argv)
        sub = argv[1]  # argv == ["vq", "<sub>", ...]
        if sub == "submit":
            return _FakeProc(stdout=self.submit_stdout, returncode=self.submit_rc)
        if sub == "status":
            i = min(self._status_idx, len(self.status_sequence) - 1)
            self._status_idx += 1
            return _FakeProc(stdout=self.status_sequence[i] + "\n")
        if sub == "fetch":
            return _FakeProc(returncode=self.fetch_rc)
        return _FakeProc()


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def test_submit_parses_bare_job_id(tmp_path):
    runner = _FakeVqRunner(submit_stdout="job-abc123\n")
    vq = VqTransport(runner=runner)
    handle = vq.submit(tmp_path, ["bash", "run-crystal.sh", "mgo.d12"])
    assert isinstance(handle, JobHandle)
    assert handle.job_id == "job-abc123"
    assert handle.work_dir == tmp_path
    assert handle.command == ("bash", "run-crystal.sh", "mgo.d12")


def test_submit_parses_id_from_verbose_line(tmp_path):
    """`vq submit` might print 'Submitted job <id>' rather than a bare id."""
    runner = _FakeVqRunner(submit_stdout="Submitted job xyz789\n")
    vq = VqTransport(runner=runner)
    handle = vq.submit(tmp_path, ["true"])
    assert handle.job_id == "xyz789"


def test_submit_builds_correct_argv(tmp_path):
    runner = _FakeVqRunner()
    vq = VqTransport(runner=runner, default_cpus=14, default_wall_time_s=7200)
    vq.submit(tmp_path, ["bash", "run-crystal.sh", "mgo.d12"])
    argv = runner.calls[0]
    assert argv[0:2] == ["vq", "submit"]
    assert "-d" in argv and str(tmp_path) in argv
    assert "--cpus" in argv and "14" in argv
    assert "--wall-time-seconds" in argv and "7200" in argv
    # The `--` separator must precede the user command so vq doesn't
    # try to interpret `bash`/flags as its own options.
    sep = argv.index("--")
    assert argv[sep + 1:] == ["bash", "run-crystal.sh", "mgo.d12"]


def test_submit_explicit_cpus_and_walltime_override_defaults(tmp_path):
    runner = _FakeVqRunner()
    vq = VqTransport(runner=runner, default_cpus=14, default_wall_time_s=7200)
    vq.submit(tmp_path, ["true"], cpus=4, wall_time_s=1800)
    argv = runner.calls[0]
    assert "4" in argv and "1800" in argv
    assert "14" not in argv and "7200" not in argv


def test_submit_raises_on_nonzero_exit(tmp_path):
    runner = _FakeVqRunner(submit_rc=1)
    vq = VqTransport(runner=runner)
    with pytest.raises(TransportError):
        vq.submit(tmp_path, ["true"])


def test_submit_raises_jobsubmiterror_on_empty_stdout(tmp_path):
    runner = _FakeVqRunner(submit_stdout="   \n\n")
    vq = VqTransport(runner=runner)
    with pytest.raises(JobSubmitError):
        vq.submit(tmp_path, ["true"])


# ---------------------------------------------------------------------------
# poll
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "status_line,expected",
    [
        ("state: completed", "completed"),
        ("state: running", "running"),
        ("state:  failed (exit 1)", "failed"),
        ("state: KILLED", "killed"),
        ("STATE: interrupted", "interrupted"),
    ],
)
def test_poll_parses_state(tmp_path, status_line, expected):
    runner = _FakeVqRunner(status_sequence=[status_line])
    vq = VqTransport(runner=runner)
    handle = JobHandle("job-1", tmp_path, ("true",))
    assert vq.poll(handle) == expected


def test_poll_unparseable_status_returns_unknown(tmp_path):
    """No `state:` line → 'unknown' (wait-loop keeps polling rather
    than crashing)."""
    runner = _FakeVqRunner(status_sequence=["job is doing something"])
    vq = VqTransport(runner=runner)
    handle = JobHandle("job-1", tmp_path, ("true",))
    assert vq.poll(handle) == "unknown"


# ---------------------------------------------------------------------------
# wait (the concrete poll-loop in the base class)
# ---------------------------------------------------------------------------

def test_wait_loops_until_terminal(tmp_path):
    runner = _FakeVqRunner(
        status_sequence=["state: running", "state: running", "state: completed"]
    )
    vq = VqTransport(runner=runner)
    handle = JobHandle("job-1", tmp_path, ("true",))
    slept: list[float] = []
    state = vq.wait(
        handle,
        poll_interval_s=30.0,
        sleep=slept.append,          # no-op sleep, records intervals
        clock=lambda: 0.0,           # frozen clock — never times out
    )
    assert state == "completed"
    # running, running, completed → 3 polls → 2 sleeps between them.
    assert slept == [30.0, 30.0]


def test_wait_raises_on_timeout(tmp_path):
    runner = _FakeVqRunner(status_sequence=["state: running"])  # never terminal
    vq = VqTransport(runner=runner)
    handle = JobHandle("job-1", tmp_path, ("true",))
    # Advancing clock: each call returns a later time → trips timeout.
    ticks = iter([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    with pytest.raises(JobTimeoutError):
        vq.wait(
            handle,
            timeout_s=25.0,
            poll_interval_s=1.0,
            sleep=lambda _s: None,
            clock=lambda: next(ticks),
        )


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def test_fetch_returns_job_subdir_and_creates_dest(tmp_path):
    runner = _FakeVqRunner()
    vq = VqTransport(runner=runner)
    handle = JobHandle("job-abc123", tmp_path, ("true",))
    dest = tmp_path / "fetched"
    out = vq.fetch(handle, dest)
    assert out == dest / "job-abc123"
    assert dest.is_dir()
    # The fetch argv must point -o at dest.
    fetch_argv = runner.calls[-1]
    assert fetch_argv[0:3] == ["vq", "fetch", "job-abc123"]
    assert "-o" in fetch_argv and str(dest) in fetch_argv


def test_fetch_raises_on_nonzero_exit(tmp_path):
    runner = _FakeVqRunner(fetch_rc=2)
    vq = VqTransport(runner=runner)
    handle = JobHandle("job-1", tmp_path, ("true",))
    with pytest.raises(TransportError):
        vq.fetch(handle, tmp_path / "fetched")


# ---------------------------------------------------------------------------
# run (submit → wait → fetch convenience)
# ---------------------------------------------------------------------------

def test_run_chains_submit_wait_fetch(tmp_path):
    runner = _FakeVqRunner(
        submit_stdout="job-run-1\n",
        status_sequence=["state: running", "state: completed"],
    )
    vq = VqTransport(runner=runner)
    result = vq.run(
        tmp_path,
        ["bash", "run-crystal.sh", "mgo.d12"],
        tmp_path / "out",
        sleep=lambda _s: None,
        clock=lambda: 0.0,
    )
    assert result.state == "completed"
    assert result.ok
    assert result.output_dir == tmp_path / "out" / "job-run-1"
    # Subcommand order: submit, status*, fetch.
    subs = [c[1] for c in runner.calls]
    assert subs[0] == "submit"
    assert subs[-1] == "fetch"
    assert "status" in subs


def test_run_result_not_ok_on_failed_state(tmp_path):
    runner = _FakeVqRunner(
        submit_stdout="job-fail-1\n",
        status_sequence=["state: failed"],
    )
    vq = VqTransport(runner=runner)
    result = vq.run(
        tmp_path, ["false"], tmp_path / "out",
        sleep=lambda _s: None, clock=lambda: 0.0,
    )
    assert result.state == "failed"
    assert not result.ok
