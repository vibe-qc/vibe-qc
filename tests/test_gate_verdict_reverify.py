"""Unit tests for the test-gate verdict's contention re-verify (FIX 3).

The full-suite gate runs the fast lane at high concurrency on a shared box. A
healthy-but-heavy file can be starved into a FAIL/ABORT/TIMEOUT/OOM under load
that it does not reproduce when run alone — a *contention artifact*, not a code
regression (the v0.13.0 cut hit exactly this on ~8 files). ``gate_verdict.py``
gained a ``--reverify`` step: before a NEW (unlisted) non-PASS file gates RED,
it is re-run once in isolation and only stays red if it fails again.

These tests pin that logic. They load ``scripts/test_gate/gate_verdict.py`` by
path (it lives outside the importable package) and inject a fake runner, so no
built core / real pytest spawn is needed.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

GV_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "test_gate" / "gate_verdict.py"
)


def _load_gate_verdict():
    spec = importlib.util.spec_from_file_location("gate_verdict_under_test", GV_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gv = _load_gate_verdict()


def _rec(file, status, **extra):
    base = {"file": file, "status": status, "rc": 1, "elapsed_s": 1.0,
            "peak_rss_mb": 1.0, "counts": {}, "tier": "T1"}
    base.update(extra)
    return base


# --------------------------------------------------------------------------
# reverify_new_reds — the pure partition function
# --------------------------------------------------------------------------

def test_reverify_passing_file_is_recovered_not_red():
    """A candidate that PASSES alone is a contention artifact → recovered."""
    cands = [_rec("tests/test_a.py", "TIMEOUT")]
    runner = lambda f: _rec(f, "PASS")  # noqa: E731
    still_red, recovered = gv.reverify_new_reds(cands, runner)
    assert still_red == []
    assert [r[0]["file"] for r in recovered] == ["tests/test_a.py"]


def test_reverify_failing_file_stays_red():
    """A candidate that FAILS again alone is a real regression → still red."""
    cands = [_rec("tests/test_b.py", "FAIL")]
    runner = lambda f: _rec(f, "FAIL")  # noqa: E731
    still_red, recovered = gv.reverify_new_reds(cands, runner)
    assert [r["file"] for r in still_red] == ["tests/test_b.py"]
    assert recovered == []


def test_reverify_notests_counts_as_recovered():
    """NOTESTS on the isolated re-run (e.g. everything deselected by the lane's
    markexpr) is not a failure — it must not gate."""
    cands = [_rec("tests/test_c.py", "ABORT")]
    still_red, recovered = gv.reverify_new_reds(cands, lambda f: _rec(f, "NOTESTS"))
    assert still_red == []
    assert len(recovered) == 1


def test_reverify_inconclusive_runner_stays_red_conservatively():
    """If the re-run itself can't be carried out (runner returns None), the red
    is NOT cleared — an inconclusive re-run never recovers a file."""
    cands = [_rec("tests/test_d.py", "OOM_KILLED")]
    still_red, recovered = gv.reverify_new_reds(cands, lambda f: None)
    assert [r["file"] for r in still_red] == ["tests/test_d.py"]
    assert recovered == []


def test_reverify_partitions_mixed_batch():
    """Mixed batch: passers recover, repeaters stay red, each routed correctly."""
    cands = [
        _rec("tests/test_pass.py", "TIMEOUT"),
        _rec("tests/test_fail.py", "FAIL"),
        _rec("tests/test_seg.py", "SEGFAULT"),
    ]

    def runner(f):
        return _rec(f, "PASS" if f == "tests/test_pass.py" else "FAIL")

    still_red, recovered = gv.reverify_new_reds(cands, runner)
    assert sorted(r["file"] for r in still_red) == ["tests/test_fail.py", "tests/test_seg.py"]
    assert [r[0]["file"] for r in recovered] == ["tests/test_pass.py"]


# --------------------------------------------------------------------------
# main() — re-verify integrated into the green/red verdict
# --------------------------------------------------------------------------

def _write_inputs(tmp_path, runs, baseline):
    triage = tmp_path / "triage.jsonl"
    triage.write_text("".join(json.dumps(r) + "\n" for r in runs))
    base = tmp_path / "baseline.json"
    base.write_text(json.dumps(baseline))
    return str(triage), str(base)


def test_main_new_red_gates_without_reverify(tmp_path):
    """Back-compat: with no --reverify, a NEW non-PASS file gates RED (exit 1),
    exactly as before this change."""
    runs = [_rec("tests/test_ok.py", "PASS"), _rec("tests/test_new.py", "FAIL")]
    triage, base = _write_inputs(tmp_path, runs, {"known_reds": [], "known_slow": []})
    assert gv.main([triage, base]) == 1


def test_main_all_tracked_is_green(tmp_path):
    """A known-red and a known-slow non-PASS file do not gate → green (0)."""
    runs = [
        _rec("tests/test_ok.py", "PASS"),
        _rec("tests/test_kr.py", "FAIL"),
        _rec("tests/test_ks.py", "TIMEOUT"),
    ]
    baseline = {
        "known_reds": [{"file": "tests/test_kr.py", "reason": "tracked"}],
        "known_slow": [{"file": "tests/test_ks.py", "reason": "slow"}],
    }
    triage, base = _write_inputs(tmp_path, runs, baseline)
    assert gv.main([triage, base]) == 0


def test_main_reverify_recovers_contention_artifact(tmp_path, monkeypatch):
    """With --reverify, a NEW red that PASSES alone is downgraded to a
    contention artifact and the gate goes GREEN. Only the NEW-red file is
    re-run (cheap) — never the PASS or the known-slow file."""
    runs = [
        _rec("tests/test_ok.py", "PASS"),
        _rec("tests/test_ks.py", "TIMEOUT"),       # known-slow, must not be re-run
        _rec("tests/test_flaky.py", "TIMEOUT"),    # NEW red, recovers alone
    ]
    baseline = {"known_reds": [], "known_slow": [{"file": "tests/test_ks.py", "reason": "slow"}]}
    triage, base = _write_inputs(tmp_path, runs, baseline)

    calls = []

    def fake_runner_builder(args):
        def runner(rel):
            calls.append(rel)
            return _rec(rel, "PASS")
        return runner

    monkeypatch.setattr(gv, "_build_isolated_runner", fake_runner_builder)
    rc = gv.main([triage, base, "--reverify", "--wt", "/wt", "--py", "/py"])
    assert rc == 0                      # recovered → green
    assert calls == ["tests/test_flaky.py"]   # only the NEW red was re-run


def test_main_reverify_keeps_real_regression_red(tmp_path, monkeypatch):
    """With --reverify, a NEW red that FAILS again alone stays RED (exit 1) —
    re-verify must not mask a genuine regression."""
    runs = [_rec("tests/test_ok.py", "PASS"), _rec("tests/test_bug.py", "FAIL")]
    triage, base = _write_inputs(tmp_path, runs, {"known_reds": [], "known_slow": []})

    monkeypatch.setattr(
        gv, "_build_isolated_runner",
        lambda args: (lambda rel: _rec(rel, "FAIL")),
    )
    rc = gv.main([triage, base, "--reverify", "--wt", "/wt", "--py", "/py"])
    assert rc == 1


def test_main_reverify_requires_wt_and_py(tmp_path):
    """--reverify without --wt/--py is a usage error (argparse exits 2)."""
    triage, base = _write_inputs(tmp_path, [_rec("tests/t.py", "PASS")],
                                 {"known_reds": [], "known_slow": []})
    with pytest.raises(SystemExit) as exc:
        gv.main([triage, base, "--reverify"])
    assert exc.value.code == 2


def test_main_rejects_t3_artifact_even_when_green(tmp_path):
    """Advisory research output is structurally incapable of gating a cut."""
    triage, base = _write_inputs(
        tmp_path,
        [_rec("tests/test_research.py", "PASS", tier="T3")],
        {"known_reds": [], "known_slow": []},
    )
    assert gv.main([triage, base]) == 2


def test_main_rejects_unclassified_legacy_artifact(tmp_path):
    record = _rec("tests/test_old.py", "PASS")
    record.pop("tier")
    triage, base = _write_inputs(
        tmp_path, [record], {"known_reds": [], "known_slow": []}
    )
    assert gv.main([triage, base]) == 2
