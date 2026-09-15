"""Behavioral coverage for the tracked AIC7 wave driver."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = REPO_ROOT / "studies" / "aiccm-real-gamma" / "aic7.py"
BACKFILL_TOOL_PATH = (
    REPO_ROOT / "agentic-loop" / "artifact_status_adjudication.py"
)

# artifact_status_adjudication.py belongs to the agentic-loop control plane,
# which is internal orchestration tooling and was deliberately not carried
# into the split repositories. The driver under test lives in studies/ and is
# here; the adjudication tool it is checked against is not. Skip rather than
# fail collection, so the gate reports this honestly instead of going red on
# a file that cannot exist in this repository.
if not BACKFILL_TOOL_PATH.is_file():
    pytest.skip(
        "agentic-loop/artifact_status_adjudication.py is internal tooling and "
        "is not part of this repository",
        allow_module_level=True,
    )


def _load_driver():
    spec = importlib.util.spec_from_file_location("aic7_driver", DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_backfill_tool():
    spec = importlib.util.spec_from_file_location(
        "artifact_status_adjudication",
        BACKFILL_TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aic7 = _load_driver()
aic7_backfill = _load_backfill_tool()


def test_live_stamp_uses_banner_submodule_for_library_versions(monkeypatch):
    """The package-level ``banner`` name is a function, not the submodule."""
    package = ModuleType("vibeqc")
    package.__file__ = str(REPO_ROOT / "python" / "vibeqc" / "__init__.py")
    package.__version__ = "test-version"
    package.__path__ = []
    package.banner = lambda: "package-level banner function"
    banner_module = ModuleType("vibeqc.banner")
    banner_module.library_versions = lambda: {
        "libint": "2.9.0",
        "libxc": "7.0.0",
    }
    monkeypatch.setitem(sys.modules, "vibeqc", package)
    monkeypatch.setitem(sys.modules, "vibeqc.banner", banner_module)

    stamp = aic7.live_stamp()

    assert stamp["library_versions"] == {
        "libint": "2.9.0",
        "libxc": "7.0.0",
    }
    assert "library_versions_error" not in stamp


def _write_synthetic_backfill(tmp_path: Path):
    result = {
        "status": "complete",
        "n_rungs_planned": 2,
        "n_rungs_ok": 1,
        "runs": [
            {"rung_id": "000-ok", "status": "ok"},
            {
                "rung_id": "001-timeout",
                "status": "rung_timeout",
                "driver": {"cap_s": 50.0},
            },
        ],
    }
    raw_result = json.dumps(result, sort_keys=True).encode("utf-8")
    results_path = tmp_path / "results.json"
    results_path.write_bytes(raw_result)
    rung_dir = tmp_path / "rungs"
    rung_dir.mkdir()
    raw_rung = json.dumps({"status": "running"}, sort_keys=True).encode(
        "utf-8"
    )
    rung_path = rung_dir / "001-timeout.json"
    rung_path.write_bytes(raw_rung)
    manifest = {
        "schema": aic7_backfill.BACKFILL_SCHEMA,
        "issue": 415,
        "recorded_utc": "2026-08-28T00:00:00Z",
        "jobs": [
            {
                "job_id": "synthetic-job",
                "result_relpath": "results.json",
                "complete_gate": False,
                "results_sha256": hashlib.sha256(raw_result).hexdigest(),
                "observed": {
                    "status": "complete",
                    "exit_code": 0,
                    "n_rungs_planned": 2,
                    "n_rungs_ok": 1,
                },
                "effective": {
                    "status": "partial",
                    "exit_code": 1,
                    "n_rungs_planned": 2,
                    "n_rungs_delivered": 1,
                    "rung_status_counts": {"ok": 1, "rung_timeout": 1},
                },
                "rung_overrides": [
                    {
                        "rung_id": "001-timeout",
                        "observed_results_status": "rung_timeout",
                        "observed_rung_status": "running",
                        "observed_rung_sha256": hashlib.sha256(
                            raw_rung
                        ).hexdigest(),
                        "effective_status": "rung_timeout",
                        "attempted": True,
                        "cause": "ordinary_rung_cap",
                        "allotted_s": 50.0,
                        "driver_wall_s": 50.1,
                        "budget_limited": False,
                        "budget_s": 1000.0,
                        "job_elapsed_s": 60.0,
                    }
                ],
            }
        ],
    }
    manifest_path = tmp_path / "backfill.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return results_path, rung_path, manifest_path


def test_historical_backfill_covers_exact_six_false_green_jobs():
    manifest = aic7_backfill.load_manifest()
    jobs = {job["job_id"]: job for job in manifest["jobs"]}
    assert set(jobs) == {
        "2baaa3bb21fb",
        "8c8ac3eeea0a",
        "0b67f035bb14",
        "d6afb7290ceb",
        "9047bd80710c",
        "f90a95e4a7a2",
    }
    assert Counter(job["reported_date"] for job in jobs.values()) == {
        "2026-08-25": 2,
        "2026-08-26": 4,
    }
    expected_incomplete = {
        "2baaa3bb21fb": ["rung_timeout"],
        "8c8ac3eeea0a": ["rung_timeout"],
        "0b67f035bb14": ["budget_killed", "budget_killed"],
        "d6afb7290ceb": [
            "rung_timeout",
            "rung_timeout",
            "budget_killed",
        ],
        "9047bd80710c": [
            "rung_timeout",
            "rung_timeout",
            "budget_killed",
            "budget_killed",
        ],
        "f90a95e4a7a2": ["oom_killed"],
    }
    for job_id, job in jobs.items():
        observed = job["observed"]
        effective = job["effective"]
        assert observed["status"] == "complete"
        assert observed["exit_code"] == 0
        assert job["complete_gate"] is False
        assert effective["status"] == "partial"
        assert effective["exit_code"] != 0
        assert effective["n_rungs_delivered"] < effective["n_rungs_planned"]
        assert sum(effective["rung_status_counts"].values()) == effective[
            "n_rungs_planned"
        ]
        assert sorted(
            row["effective_status"] for row in job["rung_overrides"]
        ) == sorted(expected_incomplete[job_id])
        assert all(
            row["effective_status"] not in {"running", "skipped_budget"}
            for row in job["rung_overrides"]
        )


def test_historical_backfill_is_digest_bound_and_non_mutating(tmp_path):
    results_path, rung_path, manifest_path = _write_synthetic_backfill(tmp_path)
    source_result = results_path.read_bytes()
    source_rung = rung_path.read_bytes()

    effective = aic7_backfill.load_effective_results(
        "synthetic-job",
        results_path,
        manifest_path,
    )

    assert results_path.read_bytes() == source_result
    assert rung_path.read_bytes() == source_rung
    assert effective["status"] == "partial"
    assert effective["observed_exit_code"] == 0
    assert effective["effective_exit_code"] == 1
    assert effective["n_rungs_delivered"] == 1
    assert effective["n_rungs_incomplete"] == 1
    assert effective["rung_status_counts"] == {"ok": 1, "rung_timeout": 1}
    corrected = effective["runs"][1]
    assert corrected["historical_observed_status"] == "rung_timeout"
    assert corrected["driver"]["allotted_s"] == 50.0
    assert corrected["driver"]["budget_limited"] is False
    assert effective["historical_status_backfill"]["raw_artifact_mutated"] is False
    effective_rung = aic7_backfill.load_effective_rung(
        "synthetic-job",
        tmp_path,
        "001-timeout",
        manifest_path,
    )
    assert effective_rung["status"] == "rung_timeout"
    assert effective_rung["historical_status_backfill"]["issue"] == 415


def test_historical_backfill_fails_closed_when_evidence_changes(tmp_path):
    results_path, _, manifest_path = _write_synthetic_backfill(tmp_path)
    results_path.write_bytes(results_path.read_bytes() + b"\n")

    with pytest.raises(
        aic7_backfill.HistoricalStatusBackfillError,
        match="results digest changed",
    ):
        aic7_backfill.load_effective_results(
            "synthetic-job",
            results_path,
            manifest_path,
        )


def test_historical_backfill_derives_complete_gate_from_rungs(tmp_path):
    results_path, _, manifest_path = _write_synthetic_backfill(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["jobs"][0]["complete_gate"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        aic7_backfill.HistoricalStatusBackfillError,
        match="complete gate disagrees",
    ):
        aic7_backfill.load_effective_results(
            "synthetic-job",
            results_path,
            manifest_path,
        )


def _stub_stamp(monkeypatch):
    monkeypatch.delenv("VQ_WALL_TIME_SECONDS", raising=False)
    monkeypatch.setattr(
        aic7.subprocess,
        "run",
        _write_valid_stamp,
    )


def _write_valid_stamp(command, *args, **kwargs):
    Path(command[-1]).write_text(
        json.dumps(
            {
                "producer_full": {
                    "vibeqc_version": "test-version",
                    "library_versions": {
                        "libint": "2.9.0",
                        "libxc": "7.0.0",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(returncode=0)


def _result_document(workdir: Path) -> dict:
    paths = list(workdir.glob("aic7-*/results.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "producer",
    [
        {"vibeqc_version": "test-version"},
        {"vibeqc_version": "test-version", "library_versions": {}},
    ],
)
def test_main_refuses_missing_library_provenance_before_first_rung(
    monkeypatch,
    tmp_path,
    producer,
):
    monkeypatch.delenv("VQ_WALL_TIME_SECONDS", raising=False)
    monkeypatch.setenv("VQ_WORKDIR", str(tmp_path))

    def write_invalid_stamp(command, *args, **kwargs):
        Path(command[-1]).write_text(
            json.dumps({"producer_full": producer}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(aic7.subprocess, "run", write_invalid_stamp)
    monkeypatch.setattr(
        aic7,
        "run_rungs",
        lambda *args, **kwargs: pytest.fail(
            "a rung started without linked-library provenance"
        ),
    )

    rc = aic7.main(
        [
            "lad",
            "--system",
            "lih",
            "--basis",
            "sto-3g",
            "--route",
            "gdf",
            "--methods",
            "rhf",
            "--meshes",
            "1",
        ]
    )

    doc = _result_document(tmp_path)
    assert rc == 1
    assert doc["status"] == "partial"
    assert doc["runs"] == []
    assert doc["n_rungs_attempted"] == 0
    assert "no non-empty linked-library provenance" in doc["driver_error"]


@pytest.mark.parametrize(
    ("statuses", "expected_status", "expected_delivered", "expected_rc"),
    [
        (("failed", "failed"), "partial", 0, 1),
        (("ok", "failed"), "partial", 1, 1),
        (("ok", "not_converged"), "partial", 1, 1),
        (("ok", "rung_timeout"), "partial", 1, 1),
        (("ok", "ok"), "complete", 2, 0),
    ],
)
def test_main_fails_closed_unless_every_planned_rung_is_ok(
    monkeypatch,
    tmp_path,
    statuses,
    expected_status,
    expected_delivered,
    expected_rc,
):
    _stub_stamp(monkeypatch)
    monkeypatch.setenv("VQ_WORKDIR", str(tmp_path))

    def fake_run_rungs(doc, outdir, rungs, *args):
        assert len(rungs) == len(statuses)
        for index, (spec, status) in enumerate(zip(rungs, statuses, strict=True)):
            doc["runs"].append(
                {
                    "spec": spec,
                    "rung_id": doc["planned_rungs"][index]["rung_id"],
                    "status": status,
                }
            )

    monkeypatch.setattr(aic7, "run_rungs", fake_run_rungs)
    rc = aic7.main(
        [
            "lad",
            "--system",
            "lih",
            "--basis",
            "sto-3g",
            "--route",
            "gdf",
            "--methods",
            "rhf,pbe0",
            "--meshes",
            "1",
        ]
    )

    doc = _result_document(tmp_path)
    assert rc == expected_rc
    assert doc["status"] == expected_status
    assert doc["n_rungs_planned"] == 2
    assert doc["n_rungs_recorded"] == 2
    assert doc["n_rungs_delivered"] == expected_delivered
    assert doc["n_rungs_ok"] == expected_delivered
    assert doc["n_rungs_incomplete"] == 2 - expected_delivered
    assert doc["rung_status_counts"] == {
        status: statuses.count(status) for status in sorted(set(statuses))
    }


def test_main_fails_closed_on_duplicate_and_missing_rung_identities(
    monkeypatch,
    tmp_path,
):
    _stub_stamp(monkeypatch)
    monkeypatch.setenv("VQ_WORKDIR", str(tmp_path))

    def fake_run_rungs(doc, outdir, rungs, *args):
        first_id = doc["planned_rungs"][0]["rung_id"]
        for spec in rungs:
            doc["runs"].append(
                {"spec": spec, "rung_id": first_id, "status": "ok"}
            )

    monkeypatch.setattr(aic7, "run_rungs", fake_run_rungs)
    rc = aic7.main(
        [
            "lad",
            "--system",
            "lih",
            "--basis",
            "sto-3g",
            "--route",
            "gdf",
            "--methods",
            "rhf,pbe0",
            "--meshes",
            "1",
        ]
    )

    doc = _result_document(tmp_path)
    assert rc == 1
    assert doc["status"] == "partial"
    assert doc["n_rungs_delivered"] == 0
    assert doc["n_rungs_unrecorded"] == 1
    assert doc["n_rungs_duplicate_ids"] == 1
    assert doc["missing_rung_ids"] == [doc["planned_rungs"][1]["rung_id"]]


@pytest.mark.parametrize("method_text", ["rhf,pbe0", "rhf+pbe0"])
def test_lad_method_list_round_trips_through_tag_and_targets_first(
    monkeypatch,
    tmp_path,
    method_text,
):
    _stub_stamp(monkeypatch)
    monkeypatch.setenv("VQ_WORKDIR", str(tmp_path))
    captured = []

    def fake_run_rungs(doc, outdir, rungs, *args):
        captured.extend(rungs)
        for index, spec in enumerate(rungs):
            doc["runs"].append(
                {
                    "spec": spec,
                    "rung_id": doc["planned_rungs"][index]["rung_id"],
                    "status": "ok",
                }
            )

    monkeypatch.setattr(aic7, "run_rungs", fake_run_rungs)
    rc = aic7.main(
        [
            "lad",
            "--system",
            "lih",
            "--basis",
            "sto-3g",
            "--route",
            "gdf",
            "--methods",
            method_text,
            "--meshes",
            "2,4,3",
        ]
    )

    assert rc == 0
    assert [(rung["mesh"], rung["method"]) for rung in captured] == [
        (4, "rhf"),
        (4, "pbe0"),
        (3, "rhf"),
        (3, "pbe0"),
        (2, "rhf"),
        (2, "pbe0"),
    ]
    doc = _result_document(tmp_path)
    assert doc["tag"].endswith("-rhf+pbe0")
    assert doc["methods"] == ["rhf", "pbe0"]
    assert doc["n_rungs_planned"] == 6
    assert doc["rung_order_policy"].startswith("target-first")
    assert [row["spec"] for row in doc["planned_rungs"]] == captured


def test_real_gamma_plan_bundles_methods_once_per_target_mesh(
    monkeypatch,
    tmp_path,
):
    _stub_stamp(monkeypatch)
    monkeypatch.setenv("VQ_WORKDIR", str(tmp_path))
    captured = []

    def fake_run_rungs(doc, outdir, rungs, *args):
        captured.extend(rungs)
        for index, spec in enumerate(rungs):
            doc["runs"].append(
                {
                    "spec": spec,
                    "rung_id": doc["planned_rungs"][index]["rung_id"],
                    "status": "ok",
                }
            )

    monkeypatch.setattr(aic7, "run_rungs", fake_run_rungs)
    rc = aic7.main(
        [
            "lad",
            "--system",
            "lih",
            "--basis",
            "sto-3g",
            "--route",
            "real-gamma",
            "--methods",
            "rhf+pbe0",
            "--meshes",
            "1,3",
        ]
    )

    assert rc == 0
    assert [(row["mesh"], row["methods"]) for row in captured] == [
        (3, ["rhf", "pbe0"]),
        (1, ["rhf", "pbe0"]),
    ]
    assert all("method" not in row for row in captured)
    doc = _result_document(tmp_path)
    assert doc["methods"] == ["rhf", "pbe0"]
    assert doc["n_rungs_planned"] == 2


def test_exhausted_budget_records_every_unstarted_rung_and_exits_partial(
    monkeypatch,
    tmp_path,
):
    _stub_stamp(monkeypatch)
    monkeypatch.setenv("VQ_WORKDIR", str(tmp_path))
    rc = aic7.main(
        [
            "lad",
            "--system",
            "lih",
            "--basis",
            "sto-3g",
            "--route",
            "gdf",
            "--methods",
            "rhf",
            "--meshes",
            "1,2",
            "--budget-s",
            "0",
        ]
    )

    doc = _result_document(tmp_path)
    assert rc == 1
    assert doc["status"] == "partial"
    assert doc["n_rungs_delivered"] == 0
    assert doc["n_rungs_budget_killed"] == 2
    assert [row["spec"]["mesh"] for row in doc["planned_rungs"]] == [2, 1]
    assert {row["status"] for row in doc["runs"]} == {"budget_killed"}
    for row in doc["runs"]:
        assert row["attempted"] is False
        assert row["budget_s"] == 0.0
        assert row["elapsed_s"] >= 0.0
        assert row["remaining_at_start_s"] == 0.0

    rung_results = [
        path
        for path in (next(tmp_path.glob("aic7-*")) / "rungs").glob("*.json")
        if not path.name.endswith(".spec.json")
    ]
    assert len(rung_results) == 2
    assert {
        json.loads(path.read_text(encoding="utf-8"))["status"]
        for path in rung_results
    } == {"budget_killed"}


@pytest.mark.parametrize(
    ("budget_s", "expected_status", "expected_allotment"),
    [
        (100.0, "budget_killed", 30.0),
        (1000.0, "rung_timeout", 50.0),
    ],
)
def test_timeout_distinguishes_global_budget_from_ordinary_rung_cap(
    monkeypatch,
    tmp_path,
    budget_s,
    expected_status,
    expected_allotment,
):
    clock = SimpleNamespace(value=10.0)
    monkeypatch.setattr(aic7.time, "time", lambda: clock.value)
    monkeypatch.setattr(aic7.time, "monotonic", lambda: clock.value)
    monkeypatch.setattr(aic7.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(aic7.os, "killpg", lambda pgid, sig: None)

    class FakeProcess:
        pid = 1234

        def __init__(self, *args, **kwargs):
            self.wait_calls = 0

        def wait(self, timeout):
            self.wait_calls += 1
            if self.wait_calls == 1:
                clock.value += timeout
                raise subprocess.TimeoutExpired("aic7 worker", timeout)
            return -signal.SIGTERM

    monkeypatch.setattr(aic7.subprocess, "Popen", FakeProcess)
    (tmp_path / "rungs").mkdir()
    doc = {
        "_t0": 0.0,
        "_budget_started_monotonic": 0.0,
        "runs": [],
    }
    spec = {
        "kind": "lad",
        "system": "lih",
        "basis": "sto-3g",
        "mesh": 4,
        "method": "rhf",
        "route": "gdf",
    }

    aic7.run_rungs(
        doc,
        str(tmp_path),
        [spec],
        1,
        "gdf",
        budget_s,
        50.0,
        lambda: None,
    )

    assert len(doc["runs"]) == 1
    row = doc["runs"][0]
    assert row["status"] == expected_status
    assert row["driver"]["allotted_s"] == expected_allotment
    assert row["driver"]["budget_limited"] is (
        expected_status == "budget_killed"
    )
    persisted = json.loads(
        next(
            path
            for path in (tmp_path / "rungs").glob("*.json")
            if not path.name.endswith(".spec.json")
        ).read_text(encoding="utf-8")
    )
    assert persisted["status"] == expected_status


def test_spawn_time_counts_against_global_budget(monkeypatch, tmp_path):
    clock = SimpleNamespace(value=10.0)
    signals = []
    waits = []
    monkeypatch.setattr(aic7.time, "monotonic", lambda: clock.value)
    monkeypatch.setattr(aic7.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(
        aic7.os,
        "killpg",
        lambda pgid, sig: signals.append(sig),
    )

    class FakeProcess:
        pid = 1234

        def __init__(self, *args, **kwargs):
            clock.value = 100.0

        def wait(self, timeout):
            waits.append(timeout)
            return -signal.SIGKILL

    monkeypatch.setattr(aic7.subprocess, "Popen", FakeProcess)
    (tmp_path / "rungs").mkdir()
    spec = {
        "kind": "lad",
        "system": "lih",
        "basis": "sto-3g",
        "mesh": 4,
        "method": "rhf",
        "route": "gdf",
    }
    doc = {"_budget_started_monotonic": 0.0, "runs": []}

    aic7.run_rungs(
        doc,
        str(tmp_path),
        [spec],
        1,
        "gdf",
        100.0,
        50.0,
        lambda: None,
    )

    assert signals == [signal.SIGKILL]
    assert waits == [0.0]
    assert doc["runs"][0]["status"] == "budget_killed"
    assert doc["runs"][0]["driver"]["allotted_s"] == 0.0


def test_budget_grace_waits_follow_absolute_deadline(monkeypatch, tmp_path):
    clock = SimpleNamespace(value=10.0)
    waits = []
    monkeypatch.setattr(aic7.time, "monotonic", lambda: clock.value)
    monkeypatch.setattr(aic7.os, "getpgid", lambda pid: pid)

    def fake_killpg(pgid, sig):
        clock.value += 1.0

    monkeypatch.setattr(aic7.os, "killpg", fake_killpg)

    class FakeProcess:
        pid = 1234

        def __init__(self, *args, **kwargs):
            self.wait_calls = 0

        def wait(self, timeout):
            self.wait_calls += 1
            waits.append(timeout)
            if self.wait_calls < 3:
                clock.value += timeout
                raise subprocess.TimeoutExpired("aic7 worker", timeout)
            return -signal.SIGKILL

    monkeypatch.setattr(aic7.subprocess, "Popen", FakeProcess)
    (tmp_path / "rungs").mkdir()
    spec = {
        "kind": "lad",
        "system": "lih",
        "basis": "sto-3g",
        "mesh": 4,
        "method": "rhf",
        "route": "gdf",
    }
    doc = {"_budget_started_monotonic": 0.0, "runs": []}

    aic7.run_rungs(
        doc,
        str(tmp_path),
        [spec],
        1,
        "gdf",
        100.0,
        50.0,
        lambda: None,
    )

    assert waits == [30.0, 30.0, 28.0]
    assert clock.value == 72.0
    assert doc["runs"][0]["status"] == "budget_killed"


def test_abnormal_child_exit_cannot_preserve_worker_ok_status(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(aic7.time, "monotonic", lambda: 10.0)

    class FakeProcess:
        pid = 1234

        def __init__(self, command, **kwargs):
            Path(command[-1]).write_text(
                json.dumps({"status": "ok"}),
                encoding="utf-8",
            )

        def wait(self, timeout):
            return -signal.SIGKILL

    monkeypatch.setattr(aic7.subprocess, "Popen", FakeProcess)
    (tmp_path / "rungs").mkdir()
    spec = {
        "kind": "lad",
        "system": "lih",
        "basis": "sto-3g",
        "mesh": 4,
        "method": "rhf",
        "route": "gdf",
    }
    rid = aic7._rung_id(0, spec)
    doc = {
        "_budget_started_monotonic": 0.0,
        "n_rungs_planned": 1,
        "planned_rungs": [{"rung_id": rid, "spec": spec}],
        "runs": [],
    }

    aic7.run_rungs(
        doc,
        str(tmp_path),
        [spec],
        1,
        "gdf",
        1000.0,
        50.0,
        lambda: None,
    )
    rc = aic7._finalize_run(doc)

    assert rc == 1
    assert doc["status"] == "partial"
    assert doc["n_rungs_delivered"] == 0
    assert doc["runs"][0]["worker_reported_status"] == "ok"
    assert doc["runs"][0]["status"] == f"child_exit_{-signal.SIGKILL}"


def test_parent_launch_failure_is_persisted_as_partial(monkeypatch, tmp_path):
    _stub_stamp(monkeypatch)
    monkeypatch.setenv("VQ_WORKDIR", str(tmp_path))

    def fail_launch(*args, **kwargs):
        raise OSError("synthetic launch failure")

    monkeypatch.setattr(aic7.subprocess, "Popen", fail_launch)
    rc = aic7.main(
        [
            "lad",
            "--system",
            "lih",
            "--basis",
            "sto-3g",
            "--route",
            "gdf",
            "--methods",
            "rhf",
            "--meshes",
            "1",
        ]
    )

    doc = _result_document(tmp_path)
    assert rc == 1
    assert doc["status"] == "partial"
    assert doc["n_rungs_recorded"] == 0
    assert doc["n_rungs_unrecorded"] == 1
    assert "synthetic launch failure" in doc["driver_error"]


def test_real_gamma_bundle_requires_every_method_to_succeed(monkeypatch):
    def fake_call(record, function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(aic7, "_call", fake_call)

    def fail_pbe0(*args, **kwargs):
        raise RuntimeError("synthetic PBE0 failure")

    package = ModuleType("vibeqc")
    package.__path__ = []
    periodic = ModuleType("vibeqc.periodic")
    periodic.__path__ = []
    ccm = ModuleType("vibeqc.periodic.ccm")
    ccm.__path__ = []
    direct = ModuleType("vibeqc.periodic.ccm.direct")
    direct.run_ccm_rhf_direct = lambda *args, **kwargs: SimpleNamespace(
        energy=-1.0,
        converged=True,
        n_iter=2,
        exchange_q0="ewald",
    )
    direct.run_ccm_rks_direct = fail_pbe0
    neutral = ModuleType("vibeqc.periodic.ccm.neutral")
    neutral.ccm_neutral_cderi_fold = (
        lambda *args, **kwargs: np.zeros((1, 1, 1))
    )
    ri = ModuleType("vibeqc.periodic.ccm.ri")
    ri.run_ccm_rhf_gdf = lambda *args, **kwargs: None
    ri.run_ccm_rks_gdf = lambda *args, **kwargs: None
    package.periodic = periodic
    periodic.ccm = ccm
    ccm.direct = direct
    ccm.neutral = neutral
    ccm.ri = ri
    for module in (package, periodic, ccm, direct, neutral, ri):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    rec = {"n_cells": 1, "n_atoms_unit": 2, "mesh": 2}
    spec = {
        "route": "real-gamma",
        "methods": ["rhf", "pbe0"],
        "max_iter": 10,
        "conv_tol": 1e-9,
    }

    aic7._rung_lad(rec, lambda: None, np, object(), spec)

    assert rec["n_methods_ok"] == 1
    assert [row["status"] for row in rec["per_method"]] == ["ok", "failed"]
    assert rec["status"] == "failed"


def test_madelung_derived_rows_do_not_change_planned_rung_counts(
    monkeypatch,
    tmp_path,
):
    _stub_stamp(monkeypatch)
    monkeypatch.setenv("VQ_WORKDIR", str(tmp_path))

    def fake_run_rungs(doc, outdir, rungs, *args):
        for index, spec in enumerate(rungs):
            doc["runs"].append(
                {
                    "spec": spec,
                    "rung_id": doc["planned_rungs"][index]["rung_id"],
                    "status": "ok",
                    "mesh": spec["mesh"],
                    "construction": spec["construction"],
                    "energy_per_cell": -1.0 - index * 0.01,
                    "n_atoms_unit": 2,
                }
            )

    monkeypatch.setattr(aic7, "run_rungs", fake_run_rungs)
    rc = aic7.main(
        [
            "mad",
            "--system",
            "lih",
            "--basis",
            "sto-3g",
            "--meshes",
            "1",
        ]
    )

    doc = _result_document(tmp_path)
    assert rc == 0
    assert doc["status"] == "complete"
    assert doc["n_rungs_planned"] == 3
    assert doc["n_rungs_recorded"] == 3
    assert doc["n_rungs_delivered"] == 3
    assert len(doc["runs"]) == 4
    assert doc["runs"][-1]["status"] == "derived"


def test_driver_rejects_budget_that_consumes_vq_finalization_margin(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("VQ_WORKDIR", str(tmp_path))
    monkeypatch.setenv("VQ_WALL_TIME_SECONDS", "1000")

    with pytest.raises(SystemExit) as excinfo:
        aic7.main(
            [
                "lad",
                "--system",
                "lih",
                "--basis",
                "sto-3g",
                "--route",
                "gdf",
                "--budget-s",
                "701",
            ]
        )

    assert excinfo.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--budget-s", "nan"),
        ("--budget-s", "inf"),
        ("--rung-cap-s", "nan"),
        ("--rung-cap-s", "inf"),
    ],
)
def test_driver_rejects_nonfinite_limits_before_creating_output(
    monkeypatch,
    tmp_path,
    option,
    value,
):
    monkeypatch.delenv("VQ_WALL_TIME_SECONDS", raising=False)
    monkeypatch.setenv("VQ_WORKDIR", str(tmp_path))

    with pytest.raises(SystemExit) as excinfo:
        aic7.main(
            [
                "lad",
                "--system",
                "lih",
                "--basis",
                "sto-3g",
                "--route",
                "gdf",
                option,
                value,
            ]
        )

    assert excinfo.value.code == 2
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# GitLab issue #120 -- a fixed per-case cap inside a reservation the carrier
# did not size kills healthy calculations while the node stands idle.
# 206 members were lost that way across 202 cases in the rp217/rp223 pass;
# rp217op016 spent 2.11 h of a 16 h grant and still lost two BH9 members at
# 3540 s and 3660 s. The reservation is not a mystery -- vq exports it as
# VQ_WALL_TIME_SECONDS -- so the limits are derived from it.
# ---------------------------------------------------------------------------

def _lad_argv(*extra: str) -> list[str]:
    return [
        "lad",
        "--system",
        "lih",
        "--basis",
        "sto-3g",
        "--route",
        "gdf",
        "--methods",
        "rhf",
        *extra,
    ]


def _capture_limits(monkeypatch, tmp_path):
    """Run ``main`` with the rung loop stubbed; return what it resolved."""
    seen: dict[str, object] = {}

    def fake_run_rungs(doc, outdir, rungs, cpus, route, budget_s, cap_s,
                       flush, *rest):
        seen["budget_s"] = budget_s
        seen["cap_s"] = cap_s
        seen["rest"] = rest
        seen["n_rungs"] = len(rungs)
        for index, spec in enumerate(rungs):
            doc["runs"].append(
                {
                    "spec": spec,
                    "rung_id": doc["planned_rungs"][index]["rung_id"],
                    "status": "ok",
                }
            )

    monkeypatch.setattr(aic7, "run_rungs", fake_run_rungs)
    monkeypatch.setattr(
        aic7.subprocess,
        "run",
        _write_valid_stamp,
    )
    monkeypatch.setenv("VQ_WORKDIR", str(tmp_path))
    return seen


def test_budget_defaults_to_the_granted_vq_reservation(monkeypatch, tmp_path):
    """Under a 24 h grant the driver may spend ~24 h, not a fixed 18 h."""
    monkeypatch.setenv("VQ_WALL_TIME_SECONDS", "86400")
    seen = _capture_limits(monkeypatch, tmp_path)

    aic7.main(_lad_argv("--meshes", "1,2,3"))

    assert seen["budget_s"] == 86400.0 - aic7.WALL_FINALIZATION_GRACE_S
    doc = _result_document(tmp_path)
    assert doc["budget_source"] == "vq-wall"
    assert doc["vq_wall_time_s"] == 86400.0


def test_short_reservation_scales_the_budget_instead_of_refusing_the_run(
    monkeypatch,
    tmp_path,
):
    """A 4 h grant used to be refused outright by the 18 h constant."""
    monkeypatch.setenv("VQ_WALL_TIME_SECONDS", "14400")
    seen = _capture_limits(monkeypatch, tmp_path)

    rc = aic7.main(_lad_argv("--meshes", "1"))

    assert rc == 0
    assert seen["budget_s"] == 14400.0 - aic7.WALL_FINALIZATION_GRACE_S


def test_rung_cap_becomes_a_floor_under_a_reservation(monkeypatch, tmp_path):
    monkeypatch.setenv("VQ_WALL_TIME_SECONDS", "86400")
    seen = _capture_limits(monkeypatch, tmp_path)

    aic7.main(_lad_argv("--meshes", "1,2,3"))

    assert seen["cap_s"] == 10800.0
    doc = _result_document(tmp_path)
    assert doc["rung_cap_is_floor"] is True
    assert doc["rung_cap_source"] == "default"


def test_explicit_limits_pin_the_declared_per_wave_values(
    monkeypatch,
    tmp_path,
):
    """Issue #120 ask (2): an explicit flag is a hard cap, not a floor."""
    monkeypatch.setenv("VQ_WALL_TIME_SECONDS", "86400")
    seen = _capture_limits(monkeypatch, tmp_path)

    aic7.main(_lad_argv("--meshes", "1", "--budget-s", "3600",
                        "--rung-cap-s", "600"))

    assert seen["budget_s"] == 3600.0
    assert seen["cap_s"] == 600.0
    doc = _result_document(tmp_path)
    assert doc["budget_source"] == "explicit"
    assert doc["rung_cap_source"] == "explicit"
    assert doc["rung_cap_is_floor"] is False


def test_no_reservation_keeps_the_historical_constants_exactly(
    monkeypatch,
    tmp_path,
):
    """Negative control: the same route with no reservation exported.

    This is the feature turned OFF on the identical code path -- not a
    neighbouring one -- so the historical values are asserted exactly.
    """
    monkeypatch.delenv("VQ_WALL_TIME_SECONDS", raising=False)
    seen = _capture_limits(monkeypatch, tmp_path)

    aic7.main(_lad_argv("--meshes", "1,2,3"))

    assert seen["budget_s"] == 64800.0
    assert seen["cap_s"] == 10800.0
    doc = _result_document(tmp_path)
    assert doc["vq_wall_time_s"] is None
    assert doc["budget_source"] == "default"
    assert doc["rung_cap_is_floor"] is False




def _allotments_through_main(monkeypatch, tmp_path, argv, *, wall_s):
    """Drive the REAL rung loop through ``main``, with a fake child.

    Deliberately routed through ``main`` and not through ``run_rungs``
    directly: ``main(argv)`` is an entry point that exists unchanged at the
    parent, so the allotment it records is comparable on both sides of the
    fix rather than raising on a signature the fix introduced (L124).
    """
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr(aic7.time, "monotonic", lambda: clock.value)
    monkeypatch.setattr(aic7.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(aic7.os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(
        aic7.subprocess,
        "run",
        _write_valid_stamp,
    )

    class FakeProcess:
        pid = 4321

        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(aic7.subprocess, "Popen", FakeProcess)
    if wall_s is None:
        monkeypatch.delenv("VQ_WALL_TIME_SECONDS", raising=False)
    else:
        monkeypatch.setenv("VQ_WALL_TIME_SECONDS", str(wall_s))
    monkeypatch.setenv("VQ_WORKDIR", str(tmp_path))

    aic7.main(argv)
    doc = _result_document(tmp_path)
    return doc, [row["driver"] for row in doc["runs"] if "driver" in row]


def test_first_rung_takes_its_share_of_the_remaining_reservation(
    monkeypatch,
    tmp_path,
):
    """The filed loss, measured: 3 rungs inside a 24 h reservation.

    Pre-fix the first rung is killed at the fixed 10800 s cap -- 3 h of a
    24 h grant -- with the rest of the reservation unspent. Its share of
    what is left is (86100 - 60) / 3 = 28680 s.
    """
    doc, drivers = _allotments_through_main(
        monkeypatch,
        tmp_path,
        _lad_argv("--meshes", "1,2,3"),
        wall_s=86400,
    )
    expected = (86400.0 - 300.0 - 60.0) / 3.0
    assert drivers[0]["allotted_s"] == pytest.approx(round(expected, 1))
    assert drivers[0]["cap_basis"] == "reservation-share"
    assert drivers[0]["cap_floor_s"] == 10800.0
    assert drivers[0]["rungs_remaining_at_start"] == 3
    assert drivers[0]["budget_limited"] is False


def test_share_never_shrinks_the_declared_cap(monkeypatch, tmp_path):
    """A floor is a floor: many rungs must not shorten the per-rung cap."""
    meshes = ",".join(str(n) for n in range(1, 41))
    _doc, drivers = _allotments_through_main(
        monkeypatch,
        tmp_path,
        _lad_argv("--meshes", meshes),
        wall_s=86400,
    )
    assert drivers[0]["allotted_s"] == 10800.0
    assert drivers[0]["cap_basis"] == "reservation-share"


def test_no_reservation_leaves_the_rung_allotment_untouched(
    monkeypatch,
    tmp_path,
):
    """Negative control on the rung loop: identical argv, no reservation.

    Same entry point, same rungs, feature off. The allotment must be the
    historical constant exactly.
    """
    _doc, drivers = _allotments_through_main(
        monkeypatch,
        tmp_path,
        _lad_argv("--meshes", "1,2,3"),
        wall_s=None,
    )
    assert drivers[0]["allotted_s"] == 10800.0
    assert drivers[0]["budget_limited"] is False


def test_a_shared_allotment_still_cannot_outlive_the_budget(
    monkeypatch,
    tmp_path,
):
    """Safety property, true on both sides: sharing cannot overrun.

    The allotment is clipped to ``remaining - BUDGET_KILL_GRACE_S`` whether
    it came from a share or from the fixed cap, so the budget stays a hard
    completion bound and not a start-time hint. Asserted because raising
    the cap to a share is only safe while this holds.
    """
    _doc, drivers = _allotments_through_main(
        monkeypatch,
        tmp_path,
        _lad_argv("--meshes", "1", "--budget-s", "600"),
        wall_s=86400,
    )
    assert drivers[0]["allotted_s"] == 600.0 - 60.0
    assert drivers[0]["budget_limited"] is True
