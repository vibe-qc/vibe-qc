"""Unit tests for the report-only lane-health view (GitLab issue #450).

There is no per-commit CI and the release gate runs the blocking T0+T1 profile
only, so a commit that reddens a T2/T3 lane emits no signal anywhere. #450
collected six such lanes found by accident in two days, one of them a
default-path SCF convergence regression that shipped in three releases.
``scripts/test_gate/lane_health.py`` is the periodic report that surfaces them.

The properties pinned here are the ones that make the report trustworthy:

* it is **not** a gate — a tree full of untracked reds still exits 0, and it
  accepts the advisory tiers that ``gate_verdict.py`` deliberately refuses;
* a red carries **when it last passed and what landed since**, which is the
  whole point of running it periodically;
* what it did **not** run is reported as unknown rather than omitted — the
  blind spot #450 is about is silence, so a report that silently covered 40%
  of the inventory would recreate it;
* published text carries no local absolute paths (CLAUDE.md par. 12).

The scripts live outside the importable package, so both are loaded by path.
No built core and no git worktree are needed: the git lookup is injected.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

GATE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "test_gate"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, GATE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lh = _load("lane_health_under_test", "lane_health.py")
gv = _load("gate_verdict_for_lane_health", "gate_verdict.py")


def _rec(file, status, tier="T2", **extra):
    base = {
        "file": file,
        "status": status,
        "rc": 1 if status != "PASS" else 0,
        "elapsed_s": 1.0,
        "peak_rss_mb": 100.0,
        "counts": {},
        "tier": tier,
        "tail": "",
    }
    base.update(extra)
    return base


def _manifest_rows(*files, tier="T2", owner="release/test-health", verified=None):
    return {
        f: {
            "file": f,
            "maturity": "verified",
            "tier": tier,
            "owner": owner,
            "last_verified_sha": verified,
        }
        for f in files
    }


def _write_manifest(tmp_path, rows):
    path = tmp_path / "suite_manifest.json"
    path.write_text(json.dumps({"tests": list(rows.values())}), encoding="utf-8")
    return path


def _write_baseline(tmp_path, entries):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"known_reds": entries}), encoding="utf-8")
    return path


def _write_jsonl(tmp_path, records, name="triage.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# It is a report, not a gate.
# --------------------------------------------------------------------------

def test_untracked_reds_do_not_make_the_report_exit_nonzero(tmp_path):
    """A red tree is the subject of the report, not a failure of it.

    If this ever returns non-zero someone will wire it into a blocking path and
    the "not a gate" contract in #450's ask is gone.
    """
    files = ["tests/test_a.py", "tests/test_b.py"]
    rows = _manifest_rows(*files)
    triage = _write_jsonl(tmp_path, [_rec(f, "FAIL") for f in files])
    rc = lh.main([
        str(triage),
        "--suite-manifest", str(_write_manifest(tmp_path, rows)),
        "--baseline", str(_write_baseline(tmp_path, [])),
        "--run-sha", "b" * 40,
        "--json-out", str(tmp_path / "out.json"),
        "--out", str(tmp_path / "out.md"),
    ])
    assert rc == 0
    report = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert [e["file"] for e in report["red"]] == files
    assert all(e["tracked_known_red"] is False for e in report["red"])


def test_accepts_the_advisory_tiers_the_blocking_verdict_refuses(tmp_path):
    """The mirror image of ``gate_verdict.validate_blocking_artifact``.

    T2/T3 is ~90% of the inventory and the part that has no periodic signal, so
    the health report must accept exactly what the blocking verdict rejects.
    """
    records = [_rec("tests/test_t2.py", "FAIL", tier="T2"),
               _rec("tests/test_t3.py", "FAIL", tier="T3")]
    with pytest.raises(ValueError):
        gv.validate_blocking_artifact(records, "artifact.jsonl")

    report = lh.build_report(
        lh.latest_records(records),
        _manifest_rows("tests/test_t2.py", "tests/test_t3.py"),
        {},
        "c" * 40,
    )
    assert {e["file"] for e in report["red"]} == {
        "tests/test_t2.py", "tests/test_t3.py"
    }


# --------------------------------------------------------------------------
# A red says when it last passed and what landed since.
# --------------------------------------------------------------------------

def test_red_carries_commit_range_and_suspects_since_last_pass():
    old, new = "a" * 40, "f" * 40
    seen = {}

    def probe(since, run_sha, file):
        seen[file] = (since, run_sha)
        return 37, ["d85be3cd2 broke it", "0000000 unrelated touch"]

    report = lh.build_report(
        lh.latest_records([_rec("tests/test_ccm_conv_tol_grad.py", "FAIL")]),
        _manifest_rows("tests/test_ccm_conv_tol_grad.py", verified=old),
        {},
        new,
        git_probe=probe,
    )
    entry = report["red"][0]
    assert seen == {"tests/test_ccm_conv_tol_grad.py": (old, new)}
    assert entry["last_pass_sha"] == old
    assert entry["commit_range"] == f"{old}..{new}"
    assert entry["commits_since_last_pass"] == 37
    assert entry["suspect_commits"][0] == "d85be3cd2 broke it"


def test_previous_report_supersedes_the_manifest_for_last_pass():
    """A direct observation beats whatever the last measurement import stamped.

    ``last_verified_sha`` is months stale across most of the manifest; a health
    report that ran yesterday and saw the file green is the tighter anchor, and
    it is the only anchor at all for the rows the manifest never stamped.
    """
    manifest_sha, observed_sha, run_sha = "a" * 40, "b" * 40, "c" * 40
    ranges = {}

    def probe(since, run, file):
        ranges[file] = since
        return 3, []

    report = lh.build_report(
        lh.latest_records([_rec("tests/test_x.py", "FAIL")]),
        _manifest_rows("tests/test_x.py", verified=manifest_sha),
        {},
        run_sha,
        previous={"last_pass": {"tests/test_x.py": observed_sha}},
        git_probe=probe,
    )
    assert ranges["tests/test_x.py"] == observed_sha
    assert report["red"][0]["last_pass_source"] == "previous lane-health report"


def test_a_green_file_records_this_run_as_its_last_pass():
    report = lh.build_report(
        lh.latest_records([_rec("tests/test_green.py", "PASS"),
                           _rec("tests/test_red.py", "FAIL")]),
        _manifest_rows("tests/test_green.py", "tests/test_red.py"),
        {},
        "e" * 40,
        previous={"last_pass": {"tests/test_red.py": "1" * 40}},
    )
    assert report["last_pass"]["tests/test_green.py"] == "e" * 40
    # A file that is red now must keep its older anchor, not adopt this run.
    assert report["last_pass"]["tests/test_red.py"] == "1" * 40


def test_report_carries_an_unresolved_range_through_as_unknown():
    """A probe that cannot answer must surface as unknown in the report."""
    report = lh.build_report(
        lh.latest_records([_rec("tests/test_x.py", "FAIL")]),
        _manifest_rows("tests/test_x.py", verified="9" * 40),
        {},
        "c" * 40,
        git_probe=lambda since, run, file: (None, []),
    )
    assert report["red"][0]["commits_since_last_pass"] is None


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


def _tiny_repo(tmp_path):
    if shutil.which("git") is None:  # pragma: no cover - git is a hard dep here
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    _git(repo.parent, "init", "-q", "repo")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    return repo


def _commit(repo, rel, body, message):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    _git(repo, "add", rel)
    _git(repo, "commit", "-q", "-m", message)
    out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True)
    return out.stdout.strip()


def test_git_probe_returns_unknown_for_an_anchor_this_checkout_cannot_reach(tmp_path):
    """An unreachable anchor must not read as "nothing landed since".

    `last_verified_sha` values outlive rebases and shallow clones. Reporting
    such a range as *0 commits* would say "nothing changed since it passed" --
    the most misleading answer available -- about a file that may have been
    broken for months.
    """
    repo = _tiny_repo(tmp_path)
    head = _commit(repo, "tests/test_x.py", "one\n", "add test")
    probe = lh.make_git_probe(repo)
    assert probe("9" * 40, head, "tests/test_x.py") == (None, [])


def test_git_probe_counts_the_range_and_names_commits_touching_the_file(tmp_path):
    repo = _tiny_repo(tmp_path)
    anchor = _commit(repo, "tests/test_x.py", "one\n", "green here")
    _commit(repo, "tests/other.py", "x\n", "unrelated change")
    _commit(repo, "tests/test_x.py", "two\n", "reddens test_x")
    head = _commit(repo, "tests/other.py", "y\n", "another unrelated change")

    count, suspects = lh.make_git_probe(repo)(anchor, head, "tests/test_x.py")
    assert count == 3
    assert [line.split(" ", 1)[1] for line in suspects] == ["reddens test_x"]


# --------------------------------------------------------------------------
# What it did not run is stated, not omitted.
# --------------------------------------------------------------------------

def test_inventory_files_with_no_record_are_reported_as_not_observed():
    """#450's complaint is silence; a partial run that reads as green repeats it."""
    rows = _manifest_rows("tests/test_ran.py", "tests/test_never_ran.py")
    report = lh.build_report(
        lh.latest_records([_rec("tests/test_ran.py", "PASS")]), rows, {}, "d" * 40
    )
    assert [e["file"] for e in report["not_observed"]] == ["tests/test_never_ran.py"]
    assert report["coverage"] == {
        "inventory": 2,
        "observed": 1,
        "not_observed": 1,
        "observed_outside_inventory": [],
        "by_tier": {"T2": {"inventory": 2, "observed": 1, "red": 0}},
    }
    assert "Not observed" in lh.render_markdown(report)


def test_baseline_entry_that_now_passes_is_flagged_for_removal():
    report = lh.build_report(
        lh.latest_records([_rec("tests/test_fixed.py", "PASS")]),
        _manifest_rows("tests/test_fixed.py"),
        {"tests/test_fixed.py": {"file": "tests/test_fixed.py", "owner": "gdf-chat",
                                 "reason": "multi-k GDF"}},
        "d" * 40,
    )
    assert [e["file"] for e in report["recovered"]] == ["tests/test_fixed.py"]


def test_a_red_node_is_not_erased_by_a_later_green_sibling_node():
    """Resumable artifacts append; per-node records land one node at a time."""
    records = [
        _rec("tests/test_x.py::test_a", "FAIL"),
        _rec("tests/test_x.py::test_b", "PASS"),
    ]
    latest = lh.latest_records(records)
    assert set(latest) == {"tests/test_x.py"}
    assert latest["tests/test_x.py"]["status"] == "FAIL"


def test_a_rerun_of_a_whole_file_supersedes_its_earlier_record():
    records = [_rec("tests/test_x.py", "FAIL"), _rec("tests/test_x.py", "PASS")]
    assert lh.latest_records(records)["tests/test_x.py"]["status"] == "PASS"


# --------------------------------------------------------------------------
# The report is meant to be published.
# --------------------------------------------------------------------------

def test_tail_excerpt_redacts_worktree_and_home_paths(tmp_path):
    """CLAUDE.md par. 12: no author home paths in committed content.

    pytest tails carry them in rootdir lines and tracebacks as a matter of
    course, so redaction belongs here rather than with whoever commits the
    report.
    """
    worktree = Path.home() / "gitlab" / "vibeqc-somewhere"
    record = _rec(
        "tests/test_x.py",
        "FAIL",
        tail=(
            f"rootdir: {worktree}\n"
            f"E   FileNotFoundError: {Path.home()}/data/missing.json\n"
        ),
    )
    excerpt = lh.tail_excerpt(record, worktree)
    assert str(Path.home()) not in excerpt
    assert str(worktree) not in excerpt
    assert "<repo>" in excerpt and "~/data/missing.json" in excerpt


def test_tail_excerpt_redacts_a_home_belonging_to_neither_this_box_nor_the_worktree():
    """An artifact can be produced elsewhere and carry someone else's home.

    Redacting only `Path.home()` and `--wt` would leave those intact, and a
    committed report would then trip the pre-commit personal-info gate -- or,
    worse, not be committed and leak anyway.
    """
    # Assembled rather than written literally: the repo's own personal-info
    # pre-commit gate blocks a "/home/<lowercase>" literal in committed content,
    # which is precisely the leak this redaction exists to prevent.
    foreign = "/home/" + "someone"
    record = _rec(
        "tests/test_x.py",
        "FAIL",
        tail=f"E   FileNotFoundError: {foreign}/scratch/ref.json\n",
    )
    excerpt = lh.tail_excerpt(record, Path("/elsewhere"))
    assert foreign not in excerpt
    assert "/home/USER/scratch/ref.json" in excerpt


def test_tail_excerpt_keeps_the_placeholder_homes_the_repo_documents():
    """`/home/USER/` and the allowlisted accounts are the documented forms."""
    record = _rec("tests/test_x.py", "FAIL",
                  tail="E   missing: /home/USER/gitlab/x and /home/runner/work/y\n")
    excerpt = lh.tail_excerpt(record, Path("/elsewhere"))
    assert "/home/USER/gitlab/x" in excerpt
    assert "/home/runner/work/y" in excerpt


def test_render_markdown_marks_untracked_reds_as_new():
    report = lh.build_report(
        lh.latest_records([_rec("tests/test_new.py", "FAIL"),
                           _rec("tests/test_known.py", "FAIL")]),
        _manifest_rows("tests/test_new.py", "tests/test_known.py"),
        {"tests/test_known.py": {"file": "tests/test_known.py", "owner": "gdf-chat",
                                 "reason": "tracked"}},
        "d" * 40,
    )
    md = lh.render_markdown(report)
    assert "**NEW**" in md
    assert "Not in the known-reds baseline" in md
    assert "Tracked known red (owner: gdf-chat)" in md
