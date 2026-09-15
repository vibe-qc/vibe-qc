"""``vibeqc.output.dry_run_manifest`` — pre-flight manifest writer.

Pins the contract documented in
``docs/design_output_module.md § Phase O3 — --vibeqc-dry-run``:

  1. ``dry_run_manifest(plan)`` writes ``{stem}.system`` with the
     ``[plan]`` section populated and ``[outputs].status = "dry_run"``.
     No SCF is run, no molecule is required.
  2. The manifest is valid TOML at all times (round-trips through
     stdlib ``tomllib``).
  3. ``is_dry_run_requested()`` reads the ``VIBEQC_DRY_RUN`` env var,
     case-insensitively, with falsy values (``""``, ``"0"``,
     ``"false"``, ``"no"``) treated as not-requested.
  4. ``print_dry_run_summary(plan)`` emits one line per declared file
     plus a header / footer to the given stream.

The matching ``run_job(..., dry_run=True)`` short-circuit is exercised
in tests/test_runner.py — those would require an importable C++ core,
which the current parent venv lacks; the assertions here cover the
output-module surface that the runner short-circuit calls into.
"""

from __future__ import annotations

import io
import tomllib
from pathlib import Path

import pytest

from vibeqc.output import (
    OutputPlan,
    dry_run_manifest,
    is_dry_run_requested,
    is_dry_run_estimate_requested,
    print_dry_run_summary,
)


def _plan(tmp_path: Path, **overrides) -> OutputPlan:
    kw = dict(
        output=tmp_path / "h2o",
        method="rks",
        basis="def2-svp",
        functional="PBE",
    )
    kw.update(overrides)
    return OutputPlan.from_run_job_kwargs(**kw)


# ---------------------------------------------------------------------- #
# is_dry_run_requested — env var parsing
# ---------------------------------------------------------------------- #

def test_dry_run_env_var_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    assert is_dry_run_requested() is False


@pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "no", "No"])
def test_dry_run_env_var_falsy(value: str,
                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBEQC_DRY_RUN", value)
    assert is_dry_run_requested() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes",
                                    "on"])
def test_dry_run_env_var_truthy(value: str,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBEQC_DRY_RUN", value)
    assert is_dry_run_requested() is True


# ---------------------------------------------------------------------- #
# is_dry_run_estimate_requested: VIBEQC_DRY_RUN_ESTIMATE env parsing.
# Mirrors is_dry_run_requested; gates the opt-in peak-memory estimate
# pass that feeds [memory].estimate_bytes for `vq submit auto`.
# ---------------------------------------------------------------------- #

def test_dry_run_estimate_env_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VIBEQC_DRY_RUN_ESTIMATE", raising=False)
    assert is_dry_run_estimate_requested() is False


@pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "no", "No"])
def test_dry_run_estimate_env_var_falsy(
    value: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", value)
    assert is_dry_run_estimate_requested() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes", "on"])
def test_dry_run_estimate_env_var_truthy(
    value: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", value)
    assert is_dry_run_estimate_requested() is True


def test_dry_run_estimate_independent_of_dry_run_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The two flags are read independently: VIBEQC_DRY_RUN_ESTIMATE
    # alone (without VIBEQC_DRY_RUN) is truthy here, but the runner only
    # acts on it inside the dry-run short-circuit, so it is harmless.
    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")
    assert is_dry_run_requested() is False
    assert is_dry_run_estimate_requested() is True


# ---------------------------------------------------------------------- #
# dry_run_manifest — writes a [outputs].status="dry_run" manifest
# ---------------------------------------------------------------------- #

def test_dry_run_manifest_writes_system_sibling(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    target = dry_run_manifest(plan, print_summary=False)
    assert target == (tmp_path / "h2o").with_suffix(".system")
    assert target.is_file()


def test_dry_run_manifest_status_is_dry_run(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    target = dry_run_manifest(plan, print_summary=False)
    with target.open("rb") as fh:
        body = tomllib.load(fh)
    assert body["outputs"]["status"] == "dry_run"


def test_dry_run_manifest_records_truthful_self_exclusion(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    target = dry_run_manifest(plan, print_summary=False)
    with target.open("rb") as fh:
        body = tomllib.load(fh)
    rows = {row["path"]: row for row in body["outputs"]["files"]}
    self_row = rows[str(target)]
    assert self_row["written"] is True
    assert self_row["bytes"] == 0
    assert self_row["sha256"] == ""
    assert self_row["checksum_status"] == "self-excluded"


def test_dry_run_manifest_includes_full_plan(tmp_path: Path) -> None:
    plan = _plan(tmp_path, optimize=True, perf_log=True)
    target = dry_run_manifest(plan, print_summary=False)
    with target.open("rb") as fh:
        body = tomllib.load(fh)
    p = body["plan"]
    assert p["method"] == "RKS"
    assert p["basis"] == "def2-svp"
    assert p["functional"] == "PBE"
    declared_paths = {row["path"] for row in p["files"]}
    # optimize=True declares .traj; perf_log=True declares .perf.
    assert any(p.endswith(".traj") for p in declared_paths)
    assert any(p.endswith(".perf") for p in declared_paths)


def test_dry_run_manifest_round_trips_via_tomllib(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    target = dry_run_manifest(plan, print_summary=False)
    # tomllib raises TOMLDecodeError on malformed input — the contract
    # is that every state the writer emits is parseable.
    with target.open("rb") as fh:
        tomllib.load(fh)


def test_dry_run_manifest_preserves_legacy_sections(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    target = dry_run_manifest(plan, print_summary=False)
    with target.open("rb") as fh:
        body = tomllib.load(fh)
    # The pre-Phase-O1 sections must still be present — the dry-run
    # manifest is a superset, not a different file shape.
    for section in ("vibeqc", "host", "cpu", "memory", "python",
                    "libraries", "validation", "run"):
        assert section in body, (
            f"legacy section {section!r} missing from dry-run manifest"
        )


# ---------------------------------------------------------------------- #
# dry_run_manifest(estimate_bytes=…): [memory].estimate_bytes contract.
# Frozen contract consumed by vq's vibeqc_preflight: an int (not bool),
# > 0, in the [memory] section; absent when no estimate was computed.
# ---------------------------------------------------------------------- #

def test_dry_run_manifest_omits_estimate_by_default(tmp_path: Path) -> None:
    # No estimate_bytes passed → the [memory] section is byte-for-byte
    # what the runtime probe produced; the key must be absent so the
    # cheap output-discovery dry-run manifest is unchanged.
    plan = _plan(tmp_path)
    target = dry_run_manifest(plan, print_summary=False)
    with target.open("rb") as fh:
        body = tomllib.load(fh)
    assert "estimate_bytes" not in body["memory"]


def test_dry_run_manifest_writes_estimate_bytes(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    target = dry_run_manifest(plan, print_summary=False,
                              estimate_bytes=123_456_789)
    with target.open("rb") as fh:
        body = tomllib.load(fh)
    val = body["memory"]["estimate_bytes"]
    # vq's parser requires a real int (bool is an int subclass), > 0.
    assert type(val) is int
    assert val == 123_456_789
    # The existing [memory] probe keys must survive alongside it.
    assert "total_gb" in body["memory"]
    assert "available_gb" in body["memory"]


def test_dry_run_manifest_estimate_none_is_omitted(tmp_path: Path) -> None:
    # Explicit None is the best-effort failure sentinel from the runner;
    # it must behave exactly like "not provided".
    plan = _plan(tmp_path)
    target = dry_run_manifest(plan, print_summary=False, estimate_bytes=None)
    with target.open("rb") as fh:
        body = tomllib.load(fh)
    assert "estimate_bytes" not in body["memory"]


# ---------------------------------------------------------------------- #
# print_dry_run_summary — human-readable stdout output
# ---------------------------------------------------------------------- #

def test_print_dry_run_summary_lists_every_declared_file(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, optimize=True, perf_log=True,
                 structured_log=True)
    buf = io.StringIO()
    print_dry_run_summary(plan, stream=buf)
    out = buf.getvalue()
    for f in plan.files:
        assert str(f.path) in out, (
            f"declared file {f.path!r} missing from summary"
        )


def test_print_dry_run_summary_marks_conditional_files(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    # crash dump is conditional (only on failure) — must be flagged.
    buf = io.StringIO()
    print_dry_run_summary(plan, stream=buf)
    out = buf.getvalue()
    # The crash file row should be tagged "conditional".
    crash_lines = [ln for ln in out.splitlines() if ".dump" in ln]
    assert crash_lines, "expected a .dump line in the summary"
    assert "conditional" in crash_lines[0]


def test_print_dry_run_summary_includes_method_and_basis(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    buf = io.StringIO()
    print_dry_run_summary(plan, stream=buf)
    out = buf.getvalue()
    assert "RKS" in out
    assert "def2-svp" in out
    assert "PBE" in out


# ---------------------------------------------------------------------- #
# dry_run_manifest + print_summary together
# ---------------------------------------------------------------------- #

def test_dry_run_manifest_default_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = _plan(tmp_path)
    dry_run_manifest(plan)  # print_summary=True default
    captured = capsys.readouterr()
    assert "vibe-qc dry-run" in captured.out
    assert "RKS" in captured.out
