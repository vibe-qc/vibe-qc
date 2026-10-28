"""The campaign directory (M4): checkout guard, lock, settings binding, budget.

The kill-and-resume gate itself runs a whole scan in a subprocess
(``tests/basisset_dev/test_campaign_resume_gate.py`` in vibe-qc); these pin
each rule on its own.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess

import pytest

from vibe_basis.campaign import (
    BudgetExhausted,
    Campaign,
    CampaignHalted,
    CampaignLocked,
    CampaignMismatch,
    ManagedCheckoutError,
    checkout_root,
    source_identity,
)
from vibe_basis.engine import EngineEnergy


def _dead_pid():
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def test_a_directory_inside_a_checkout_is_refused_before_it_exists(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    target = tmp_path / "repo" / "runs" / "scan"
    assert checkout_root(target) == (tmp_path / "repo").resolve()
    with pytest.raises(ManagedCheckoutError, match="inside the git checkout"):
        Campaign.open(target)
    assert not (tmp_path / "repo" / "runs").exists()
    # A worktree or submodule keeps a .git *file*.
    (tmp_path / "wt").mkdir()
    (tmp_path / "wt" / ".git").write_text("gitdir: elsewhere\n")
    with pytest.raises(ManagedCheckoutError):
        Campaign.open(tmp_path / "wt" / "scan")


def test_a_live_driver_holds_the_lock_and_a_dead_ones_is_taken_over(tmp_path):
    first = Campaign.open(tmp_path)
    with pytest.raises(CampaignLocked, match=f"pid {os.getpid()} on this host"):
        Campaign.open(tmp_path)
    first.close()
    assert not (tmp_path / "campaign.lock").exists()

    dead = _dead_pid()
    (tmp_path / "campaign.lock").write_text(
        json.dumps({"pid": dead, "host": socket.gethostname()})
    )
    with Campaign.open(tmp_path) as again:
        assert again.session["took_over_lock"]["pid"] == dead

    # A lock from another host cannot be checked: refused, with the way out.
    (tmp_path / "campaign.lock").write_text(json.dumps({"pid": dead, "host": "elsewhere"}))
    with pytest.raises(CampaignLocked, match="remove"):
        Campaign.open(tmp_path)


def test_sessions_are_recorded_and_a_killed_one_has_no_end(tmp_path):
    with Campaign.open(tmp_path, sources={"code": "abc"}) as c:
        c.cache.put("c:1", EngineEnergy(energy=-1.0, ok=True, engine="model"))
    killed = Campaign.open(tmp_path)  # never closed, as after kill -9
    os.unlink(tmp_path / "campaign.lock")
    with Campaign.open(tmp_path):
        pass
    sessions = json.loads((tmp_path / "campaign.json").read_text())["sessions"]
    assert [s["stopped_because"] for s in sessions] == ["finished", None, "finished"]
    assert sessions[0]["sources"] == {"code": "abc"}
    assert sessions[1]["ended_unix"] is None and killed.session["journal_entries_at_start"] == 1
    assert not list(tmp_path.glob(".*.tmp"))


def test_a_resumed_session_must_bring_the_campaigns_settings(tmp_path):
    settings = {"threshold": 1.0, "bounds": (0.15, 100.0), "parallel": 2}
    with Campaign.open(tmp_path) as c:
        c.bind(settings, volatile=["parallel"])
    with Campaign.open(tmp_path) as c:
        c.bind({**settings, "parallel": 4}, volatile=["parallel"])  # volatile: allowed
        assert c.session["settings"]["parallel"] == 4
    with pytest.raises(CampaignMismatch, match=r"threshold: 1\.0 -> 0\.5"):
        with Campaign.open(tmp_path) as c:
            c.bind({**settings, "threshold": 0.5}, volatile=["parallel"])
    sessions = json.loads((tmp_path / "campaign.json").read_text())["sessions"]
    assert sessions[-1]["stopped_because"].startswith("CampaignMismatch")


def test_the_budget_counts_every_session_and_a_halt_stops_new_work(tmp_path):
    with Campaign.open(tmp_path, budget=3) as c:
        for i in range(2):
            c.charge()
            c.cache.put(f"c:{i}", EngineEnergy(energy=-1.0, ok=True, engine="model"))
    with pytest.raises(BudgetExhausted, match="budget of 3"):
        with Campaign.open(tmp_path, budget=3) as c:
            c.charge()
            c.cache.put("c:2", EngineEnergy(energy=-1.0, ok=True, engine="model"))
            c.charge()
    sessions = json.loads((tmp_path / "campaign.json").read_text())["sessions"]
    assert [s["stopped_because"] for s in sessions] == ["finished", "budget"]
    assert sessions[1]["evaluations"] == 1

    with Campaign.open(tmp_path / "halt") as c:
        c.halt("host unreachable")
        with pytest.raises(CampaignHalted, match="host unreachable"):
            c.charge()


def test_point_directories_are_named_by_their_key(tmp_path):
    with Campaign.open(tmp_path) as c:
        assert c.point_dir("c:0123abcd") == tmp_path.resolve() / "points" / "c-0123abcd"
        path = c.write_record("scan_record.json", {"a": (1, 2)})
    assert json.loads(path.read_text()) == {"a": [1, 2]}


def test_source_identity_names_a_checkouts_revision(tmp_path):
    assert source_identity(tmp_path) == {"path": str(tmp_path.resolve())}
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    (tmp_path / "f").write_text("x")
    for args in (["init", "-q"], ["add", "f"], ["commit", "-qm", "f"]):
        subprocess.run(git + args, check=True, capture_output=True)
    ident = source_identity(tmp_path)
    assert len(ident["git_sha"]) == 40 and ident["dirty"] is False
    (tmp_path / "f").write_text("y")
    assert source_identity(tmp_path)["dirty"] is True
