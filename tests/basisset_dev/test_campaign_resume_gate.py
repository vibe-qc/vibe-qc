"""M4's gate: a 200-evaluation campaign survives ``kill -9`` and resumes identically.

The campaign is the M3 topology scan (#225) driven exactly as on the fleet:
``Crystal23Engine`` emits each deck, a :class:`LedgerTransport` journals the
submissions, and the queue is a stand-in whose jobs live in files, so they
outlive the driver as a vq daemon's do. A job "runs" when it is first polled
and writes a CRYSTAL23-style output whose energy is the model of
``test_topology_scan_recipe.py``. The driver runs in its own process with a
managed git checkout as its working directory.

Pinned: the driver killed while waiting on a job leaves that job queued; the
restarted driver adopts it and every other unfetched job instead of
submitting them again, computes no journalled point twice and ends on the
record an uninterrupted run writes; the checkout is untouched; and a
campaign directory inside a checkout is refused before anything is written.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("vibe_basis")
pytest.importorskip("scipy")

DRIVER = textwrap.dedent('''
    import json, math, os, shutil, signal, sys, threading
    from pathlib import Path
    from types import SimpleNamespace

    from scipy.optimize import minimize
    from vibe_basis.campaign import Campaign
    from vibe_basis.io.structures import STRUCTURES
    from vibe_basis.pipeline import HARTREE_TO_KJ_PER_MOL
    from vibe_basis.transports.base import JobHandle, Transport
    from vibeqc.basis_crystal import parse_crystal_inline_element
    from vibeqc.basis_optimization.recipes import topology_scan as ts

    TARGETS, WEIGHTS = (0.40, 1.20, 3.60), (10.0, 2.0, 0.3)

    def model_ha(basis_text):
        mg = parse_crystal_inline_element(basis_text, 12, source="<model>")
        d = sorted(a for s in mg.shells if s.shell_type == "D" for a in s.exponents)
        gain = sum(WEIGHTS[k] * math.exp(-2.0 * math.log(a / TARGETS[k]) ** 2)
                   for k, a in enumerate(d) if k < len(TARGETS))
        return -275.0 - gain / HARTREE_TO_KJ_PER_MOL

    OUTPUT = """ *                              CRYSTAL23                          *
     CYCLE   1 TOTAL ENERGY(DFT)(AU)(   1)        {e:.11E} DE-1.0E+02

     == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) {e:.11E}

     EEEEEEEEEE TERMINATION  DATE 27 10 2026 TIME 12:00:00.0
    """

    def put(path, text):
        # Atomic, as a queue daemon's own state is: the stand-in runs inside
        # the driver, so a kill must not leave it half written.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)

    class FileQueue(Transport):
        """Jobs are files, so they outlive the driver; a job runs on its first poll."""

        def __init__(self, root):
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)
            self.lock = threading.Lock()

        def submit(self, work_dir, command, *, cpus=None, wall_time_s=None, label=None):
            with self.lock:
                job_id = f"job{len(list(self.root.glob('job*.json'))) + 1:05d}"
                put(self.root / f"{job_id}.json",
                    json.dumps({"work_dir": str(work_dir), "state": "queued"}))
            return JobHandle(job_id, Path(work_dir), tuple(command), label)

        def poll(self, handle):
            path = self.root / f"{handle.job_id}.json"
            job = json.loads(path.read_text())
            if job["state"] == "queued":
                if handle.job_id == os.environ.get("KILL_WAITING_ON"):
                    os.kill(os.getpid(), signal.SIGKILL)
                basis = (Path(job["work_dir"]).parents[1] / "basis.crystal").read_text()
                out = self.root / handle.job_id
                out.mkdir(exist_ok=True)
                put(out / "MgO.out", OUTPUT.format(e=model_ha(basis)))
                job["state"] = "completed"
                put(path, json.dumps(job))
            return job["state"]

        def fetch(self, handle, dest):
            target = Path(dest) / handle.job_id
            shutil.copytree(self.root / handle.job_id, target, dirs_exist_ok=True)
            return target

    def powell(objective, x0, *, bounds, tol, ftol_abs, xtol_abs, max_eval, initial_step):
        res = minimize(objective, x0, method="Powell", bounds=bounds,
                       options={"xtol": xtol_abs, "ftol": 1e-12, "maxfev": max_eval})
        return SimpleNamespace(message=str(res.message), n_evaluations=int(res.nfev))

    workdir, queue = sys.argv[1], sys.argv[2]
    with Campaign.open(workdir, sources={"driver": "gate"}) as campaign:
        transport = campaign.ledger(FileQueue(queue))
        factory = ts.crystal23_engine_factory(transport, crystal_binary="crystal23")
        ts.run_topology_scan(STRUCTURES["MgO"], "Mg", "D", factory, campaign,
                             parallel=2, optimizer=powell, max_eval=200)
''')


def _drive(tmp_path, workdir, queue, cwd, **env):
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    return subprocess.run(
        [sys.executable, str(driver), str(workdir), str(queue)],
        cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True, timeout=600,
    )


def _ledger(workdir):
    return [json.loads(line) for line in (workdir / "submissions.jsonl").read_text().splitlines()]


def _comparable(record):
    """The scan's result: everything but the per-session counters."""
    return {k: v for k, v in record.items() if k not in ("cache", "campaign")}


def _snapshot(root):
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes())
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "README").write_text("a managed checkout\n")
    git = ["git", "-C", str(root), "-c", "user.name=gate", "-c", "user.email=gate@example.invalid"]
    for args in (["init", "-q"], ["add", "README"], ["commit", "-qm", "init"]):
        subprocess.run(git + args, check=True, capture_output=True)
    return root


def test_a_killed_campaign_resumes_identically_and_adopts_its_queued_jobs(tmp_path, checkout):
    before = _snapshot(checkout)

    # The uninterrupted run is the reference.
    clean = tmp_path / "runs" / "clean"
    done = _drive(tmp_path, clean, tmp_path / "queue-clean", checkout)
    assert done.returncode == 0, done.stderr
    reference = json.loads((clean / "scan_record.json").read_text())
    evaluations = len(_ledger(clean)) // 2  # one "submitted", one "fetched" each
    assert evaluations >= 200

    # The same campaign killed while it waits on its 100th job.
    work, queue = tmp_path / "runs" / "killed", tmp_path / "queue-killed"
    killed = _drive(tmp_path, work, queue, checkout, KILL_WAITING_ON="job00100")
    assert killed.returncode == -signal.SIGKILL
    assert not (work / "scan_record.json").exists()
    assert (work / "campaign.lock").exists()
    first = _ledger(work)
    state = {r["key"]: r for r in first}
    unfetched = {k: r["job_id"] for k, r in state.items() if r["state"] == "submitted"}
    assert "job00100" in unfetched.values()
    journalled = len((work / "energies.jsonl").read_text().splitlines())
    # Two narrow windows a kill can hit in the other thread: the queue took a
    # job the ledger never heard of, or a job was fetched but its energy not
    # yet journalled. Each costs one recomputation; neither is a duplicate of
    # a job the ledger knew to be live.
    untracked = len(list(queue.glob("job*.json"))) - len({r["job_id"] for r in first})
    lost = sum(r["state"] == "fetched" for r in state.values()) - journalled

    resumed = _drive(tmp_path, work, queue, checkout)
    assert resumed.returncode == 0, resumed.stderr
    record = json.loads((work / "scan_record.json").read_text())

    # Identical: the same candidates, scores, front and stopping reason.
    assert _comparable(record) == _comparable(reference)

    # Every job left on the queue was adopted, not submitted again, and no
    # journalled point was computed twice.
    second = _ledger(work)[len(first):]
    adopted = {r["key"]: r["job_id"] for r in second if r["state"] == "adopted"}
    assert adopted == unfetched
    assert not [r for r in second if r["state"] == "submitted" and r["key"] in unfetched]
    assert len(list(queue.glob("job*.json"))) == evaluations + untracked + lost

    # The second session took over the dead driver's lock and says so.
    sessions = json.loads((work / "campaign.json").read_text())["sessions"]
    assert [s["stopped_because"] if s["ended_unix"] else None for s in sessions] == [None, "finished"]
    assert sessions[1]["took_over_lock"]["pid"] == sessions[0]["pid"]
    assert sessions[1]["journal_entries_at_start"] == journalled
    assert sessions[1]["adoptions"] == len(unfetched)
    assert sessions[1]["open_submissions"] == []
    assert not (work / "campaign.lock").exists()

    # One directory per evaluation, named by its key: none was overwritten.
    points = [p for p in (work / "points").iterdir() if p.name != "probe"]
    assert len(points) == evaluations

    # Zero writes into the managed checkout the drivers ran in.
    assert _snapshot(checkout) == before
    status = subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--ignored"],
                            capture_output=True, text=True, check=True)
    assert status.stdout == ""


def test_a_campaign_inside_a_checkout_is_refused_before_any_write(tmp_path, checkout):
    before = _snapshot(checkout)
    run = _drive(tmp_path, checkout / "runs" / "scan", tmp_path / "queue", checkout)
    assert run.returncode != 0
    assert "ManagedCheckoutError" in run.stderr
    assert not (checkout / "runs").exists()
    assert _snapshot(checkout) == before
