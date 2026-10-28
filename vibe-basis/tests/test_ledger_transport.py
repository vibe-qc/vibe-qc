"""LedgerTransport: a restarted driver adopts the jobs its predecessor queued.

The inner transport is a stand-in whose job table outlives any one
``LedgerTransport``, as a vq daemon's does; a "restart" is a new
``LedgerTransport`` over the same ledger file.
"""

from __future__ import annotations

import json

import pytest

from vibe_basis.campaign import Campaign
from vibe_basis.transports.base import JobHandle, Transport, TransportError
from vibe_basis.transports.ledger import LedgerTransport, submission_key


class Queue(Transport):
    def __init__(self, jobs=None):
        self.jobs = {} if jobs is None else jobs
        self.submitted = []

    def submit(self, work_dir, command, *, cpus=None, wall_time_s=None, label=None):
        job_id = f"job{len(self.jobs) + 1}"
        self.jobs[job_id] = "running"
        self.submitted.append(job_id)
        return JobHandle(job_id, work_dir, tuple(command), label)

    def poll(self, handle):
        state = self.jobs[handle.job_id]
        if state == "lost":
            raise TransportError(f"{handle.job_id} not found")
        return state

    def fetch(self, handle, dest):
        return dest


@pytest.fixture
def staged(tmp_path):
    wd = tmp_path / "calc"
    wd.mkdir()
    (wd / "MgO.d12").write_text("deck\n")
    return wd


def test_a_restart_adopts_an_unfetched_job_instead_of_resubmitting(tmp_path, staged):
    queue = Queue()
    first = LedgerTransport(queue, tmp_path / "ledger.jsonl")
    handle = first.submit(staged, ["sh", "run.sh"], cpus=4, label="calc/MgO/a1b2c3")
    assert queue.submitted == ["job1"]
    assert [r["job_id"] for r in first.open_submissions()] == ["job1"]

    # The driver dies; a new one asks for the same calculation.
    again = LedgerTransport(Queue(queue.jobs), tmp_path / "ledger.jsonl")
    adopted = again.submit(staged, ["sh", "run.sh"], cpus=4, label="calc/MgO/d4e5f6")
    assert adopted.job_id == handle.job_id and again.inner.submitted == []
    assert again.adoptions == 1
    again.fetch(adopted, tmp_path)
    assert again.open_submissions() == []

    # Fetched: the same calculation asked for later is a new job.
    third = LedgerTransport(Queue(queue.jobs), tmp_path / "ledger.jsonl")
    assert third.submit(staged, ["sh", "run.sh"], cpus=4).job_id == "job2"


def test_a_submission_is_what_it_would_run(tmp_path, staged):
    key = submission_key(staged, ["sh", "run.sh"], 4, 3600)
    assert submission_key(staged, ["sh", "run.sh"], 4, 3600) == key
    assert submission_key(staged, ["sh", "run.sh"], 2, 3600) != key
    (staged / "MgO.d12").write_text("another deck\n")
    assert submission_key(staged, ["sh", "run.sh"], 4, 3600) != key


def test_a_job_that_ended_without_completing_is_not_adopted(tmp_path, staged):
    queue = Queue()
    LedgerTransport(queue, tmp_path / "l.jsonl").submit(staged, ["x"])
    queue.jobs["job1"] = "killed"  # an operator cancelled the orphan
    again = LedgerTransport(Queue(queue.jobs), tmp_path / "l.jsonl")
    assert again.submit(staged, ["x"]).job_id == "job2"
    states = [json.loads(line)["state"] for line in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert states == ["submitted", "abandoned", "submitted"]


def test_an_adopted_job_that_does_not_deliver_is_submitted_afresh(tmp_path, staged):
    queue = Queue()
    LedgerTransport(queue, tmp_path / "l.jsonl").submit(staged, ["x"])
    queue.jobs["job1"] = "lost"  # the host no longer knows it
    again = LedgerTransport(Queue(queue.jobs), tmp_path / "l.jsonl")
    # The poll fails, which says nothing either way: adopted once ...
    assert again.submit(staged, ["x"]).job_id == "job1"
    # ... and when the caller asks again, a fresh job.
    assert again.submit(staged, ["x"]).job_id == "job2"
    assert again.inner.submitted == ["job2"]


def test_a_ledger_cut_short_by_a_kill_still_loads(tmp_path, staged):
    queue = Queue()
    LedgerTransport(queue, tmp_path / "l.jsonl").submit(staged, ["x"])
    with (tmp_path / "l.jsonl").open("a") as fh:
        fh.write('{"key": "trunc')
    again = LedgerTransport(Queue(queue.jobs), tmp_path / "l.jsonl")
    assert again.submit(staged, ["x"]).job_id == "job1"


def test_a_campaign_records_its_submissions_and_adoptions(tmp_path, staged):
    queue = Queue()
    with Campaign.open(tmp_path / "run") as c:
        c.ledger(queue).submit(staged, ["x"])
    with Campaign.open(tmp_path / "run") as c:
        t = c.ledger(Queue(queue.jobs))
        t.fetch(t.submit(staged, ["x"]), tmp_path)
    s1, s2 = json.loads((tmp_path / "run" / "campaign.json").read_text())["sessions"]
    assert (s1["submissions"], s1["adoptions"], s1["open_submissions"]) == (1, 0, ["job1"])
    assert (s2["submissions"], s2["adoptions"], s2["open_submissions"]) == (0, 1, [])
