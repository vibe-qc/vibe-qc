"""Tests for the v0.6 structured (NDJSON) log.

The acceptance bar:

* Every line is a self-contained JSON object that ``json.loads`` (and
  by extension ``jq``) accepts — no NaN / Infinity tokens, no trailing
  commas.
* Records carry a stable ``"event"`` key plus a per-event payload.
* Format-stability: existing event names and their fields don't change
  shape across this test suite. New event names can be added in a
  future patch.
* Numpy scalars / arrays / Path objects encode without error.
* The context manager and the env-var (``VIBEQC_STRUCTURED_LOG``)
  routes are equivalent.

Tests fall into two layers: pure-Python tests that exercise
:mod:`vibeqc.structured_log` directly, and one integration test that
runs an end-to-end ``run_job`` and validates the emitted ``.scf.jsonl``
end-to-end.
"""

from __future__ import annotations

import json
import math
import tomllib
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    PeriodicSystem,
    RHFOptions,
    StructuredLog,
    active_structured_log,
    run_fingerprint,
    run_job,
    run_periodic_job,
    structured_log,
)
from vibeqc.output import output_level
from vibeqc.structured_log import emit as emit_module


# ---------------------------------------------------------------------------
# Fixture: tiny H2 molecule for end-to-end tests.
# ---------------------------------------------------------------------------

@pytest.fixture
def h2_molecule() -> Molecule:
    return Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        0, 1,
    )


# ---------------------------------------------------------------------------
# StructuredLog: JSON encoding fundamentals.
# ---------------------------------------------------------------------------

def _read_records(path: Path) -> list[dict]:
    """Load a .scf.jsonl file as a list of records, with one
    json.loads per line — verifies the NDJSON shape."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            records.append(json.loads(line))
    return records


def test_emit_writes_one_json_record_per_line(tmp_path):
    log_path = tmp_path / "out.scf.jsonl"
    log = StructuredLog(log_path)
    log.emit("hello", n=1, label="first")
    log.emit("hello", n=2, label="second")
    log.close()

    records = _read_records(log_path)
    assert len(records) == 2
    assert records[0]["event"] == "hello"
    assert records[0]["n"] == 1
    assert records[1]["label"] == "second"


def test_every_record_carries_timestamp_and_event(tmp_path):
    log_path = tmp_path / "out.scf.jsonl"
    log = StructuredLog(log_path)
    log.emit("scf_iter", iter=1, energy=-1.117)
    log.close()

    records = _read_records(log_path)
    assert len(records) == 1
    assert "event" in records[0]
    assert "timestamp" in records[0]
    # Timestamp is ISO-8601 with timezone offset.
    assert "T" in records[0]["timestamp"]


def test_disabled_log_is_a_noop(tmp_path):
    log_path = tmp_path / "out.scf.jsonl"
    log = StructuredLog(log_path, enabled=False)
    assert not log.enabled
    log.emit("hello", n=1)
    log.close()
    # File never opened.
    assert not log_path.exists()


def test_log_with_no_path_is_a_noop():
    log = StructuredLog(None)
    assert not log.enabled
    log.emit("hello", n=1)  # must not raise
    log.close()


def test_open_failure_uses_output_warning(tmp_path):
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("block", encoding="utf-8")
    log_path = blocking_file / "out.scf.jsonl"

    with pytest.warns(
        UserWarning,
        match=r"structured_log_open.*FileExistsError",
    ):
        log = StructuredLog(log_path)

    assert not log.enabled
    log.emit("ignored")
    log.close()


def test_write_failure_uses_output_warning(tmp_path):
    class FailingHandle:
        def write(self, _text):
            raise OSError("synthetic structured-log write failure")

        def flush(self):
            pass

        def close(self):
            pass

    log = StructuredLog(tmp_path / "out.scf.jsonl")
    assert log._fh is not None
    log._fh.close()
    log._fh = FailingHandle()

    with pytest.warns(
        UserWarning,
        match=r"structured_log_write.*synthetic structured-log write failure",
    ):
        log.emit("will_fail")

    assert log.n_emitted == 0
    log.close()


# ---------------------------------------------------------------------------
# StructuredLog: numpy / non-finite / Path / nested-dict encoding.
# ---------------------------------------------------------------------------

def test_emit_handles_numpy_scalars_and_arrays(tmp_path):
    log_path = tmp_path / "out.scf.jsonl"
    with structured_log(log_path):
        emit_module(
            "numpy_test",
            int_scalar=np.int64(42),
            float_scalar=np.float64(3.14),
            bool_scalar=np.bool_(True),
            array_1d=np.array([1.0, 2.0, 3.0]),
            array_2d=np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
    records = _read_records(log_path)
    assert records[0]["int_scalar"] == 42
    assert records[0]["float_scalar"] == pytest.approx(3.14)
    assert records[0]["bool_scalar"] is True
    assert records[0]["array_1d"] == [1.0, 2.0, 3.0]
    assert records[0]["array_2d"] == [[1.0, 2.0], [3.0, 4.0]]


def test_emit_coerces_non_finite_floats_to_null(tmp_path):
    log_path = tmp_path / "out.scf.jsonl"
    with structured_log(log_path):
        emit_module("nonfinite",
                    nan=float("nan"),
                    pos_inf=float("inf"),
                    neg_inf=float("-inf"),
                    finite=1.5)
    # Re-parse with strict JSON. allow_nan is False by default in
    # json.loads, but Python's loader accepts NaN unless we explicitly
    # disable — a manual sentinel-string check is the cleaner gate.
    text = log_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    records = _read_records(log_path)
    assert records[0]["nan"] is None
    assert records[0]["pos_inf"] is None
    assert records[0]["neg_inf"] is None
    assert records[0]["finite"] == 1.5


def test_emit_handles_path_objects(tmp_path):
    log_path = tmp_path / "out.scf.jsonl"
    payload_path = tmp_path / "some/nested/path.txt"
    with structured_log(log_path):
        emit_module("path_test", out=payload_path)
    records = _read_records(log_path)
    assert records[0]["out"] == str(payload_path)


def test_emit_handles_nested_dicts_and_lists(tmp_path):
    log_path = tmp_path / "out.scf.jsonl"
    with structured_log(log_path):
        emit_module(
            "memory_estimate",
            total_gb=0.05,
            by_category={
                "ERI tensor": 1024 * 1024,
                "Fock + density + 1e": 4096,
            },
            history=[1, 2, np.float32(3.5)],
        )
    records = _read_records(log_path)
    assert records[0]["by_category"]["ERI tensor"] == 1024 * 1024
    assert records[0]["history"] == [1, 2, pytest.approx(3.5, rel=1e-6)]


# ---------------------------------------------------------------------------
# Context manager + env-var routing.
# ---------------------------------------------------------------------------

def test_context_manager_activates_then_clears(tmp_path):
    log_path = tmp_path / "out.scf.jsonl"
    assert active_structured_log() is None
    with structured_log(log_path):
        assert active_structured_log() is not None
        emit_module("inside", n=1)
    assert active_structured_log() is None
    records = _read_records(log_path)
    assert len(records) == 1
    assert records[0]["event"] == "inside"


def test_emit_outside_context_is_noop(tmp_path):
    # The free function is harmless without an active log.
    emit_module("nowhere", n=1)
    # Nothing written, nothing raised.
    assert active_structured_log() is None


def test_env_var_resolves_path_when_no_explicit_arg(tmp_path, monkeypatch):
    log_path = tmp_path / "envvar.scf.jsonl"
    monkeypatch.setenv("VIBEQC_STRUCTURED_LOG", str(log_path))
    with structured_log() as log:
        assert log.enabled
        log.emit("from_env", flag=True)
    assert log_path.exists()
    records = _read_records(log_path)
    assert records[0]["flag"] is True


def test_explicit_path_wins_over_env_var(tmp_path, monkeypatch):
    env_path = tmp_path / "env.scf.jsonl"
    explicit_path = tmp_path / "explicit.scf.jsonl"
    monkeypatch.setenv("VIBEQC_STRUCTURED_LOG", str(env_path))
    with structured_log(explicit_path) as log:
        log.emit("test", n=1)
    assert explicit_path.exists()
    assert not env_path.exists()


def test_no_path_no_env_yields_disabled_log(tmp_path, monkeypatch):
    monkeypatch.delenv("VIBEQC_STRUCTURED_LOG", raising=False)
    with structured_log() as log:
        assert not log.enabled
        log.emit("nothing", n=1)


def test_enabled_false_overrides_path(tmp_path):
    log_path = tmp_path / "should_not_exist.scf.jsonl"
    with structured_log(log_path, enabled=False) as log:
        assert not log.enabled
        log.emit("never", n=1)
    assert not log_path.exists()


# ---------------------------------------------------------------------------
# run_fingerprint stability.
# ---------------------------------------------------------------------------

def test_run_fingerprint_is_stable_for_same_inputs(h2_molecule):
    fp1 = run_fingerprint(method="rhf", basis="sto-3g",
                          functional=None, molecule=h2_molecule)
    fp2 = run_fingerprint(method="rhf", basis="sto-3g",
                          functional=None, molecule=h2_molecule)
    assert fp1 == fp2


def test_run_fingerprint_changes_when_method_changes(h2_molecule):
    fp_rhf = run_fingerprint(method="rhf", basis="sto-3g",
                             functional=None, molecule=h2_molecule)
    fp_rks = run_fingerprint(method="rks", basis="sto-3g",
                             functional="pbe", molecule=h2_molecule)
    assert fp_rhf != fp_rks


def test_run_fingerprint_changes_when_geometry_changes():
    mol1 = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1)
    mol2 = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.5])], 0, 1)
    fp1 = run_fingerprint(method="rhf", basis="sto-3g",
                          functional=None, molecule=mol1)
    fp2 = run_fingerprint(method="rhf", basis="sto-3g",
                          functional=None, molecule=mol2)
    assert fp1 != fp2


def test_run_fingerprint_is_short_hex():
    mol = Molecule([Atom(1, [0, 0, 0])], 0, 2)
    fp = run_fingerprint(method="uhf", basis="sto-3g",
                         functional=None, molecule=mol)
    assert isinstance(fp, str)
    assert len(fp) == 16
    int(fp, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# Integration: run_job(structured_log=True) emits the full event suite.
# ---------------------------------------------------------------------------

def _opts_loose() -> RHFOptions:
    o = RHFOptions()
    o.max_iter = 80
    o.conv_tol_energy = 1e-8
    o.conv_tol_grad = 1e-6
    return o


def test_run_job_structured_log_true_writes_jsonl(tmp_path, h2_molecule):
    out_stem = tmp_path / "h2"
    run_job(
        h2_molecule,
        basis="sto-3g",
        method="rhf",
        output=out_stem,
        structured_log=True,
        progress=False,
        rhf_options=_opts_loose(),
    )
    jsonl_path = out_stem.with_suffix(".scf.jsonl")
    assert jsonl_path.exists(), \
        f"structured log not written; tmp_path={list(tmp_path.iterdir())}"


def test_run_job_structured_log_false_does_not_write(tmp_path, h2_molecule):
    out_stem = tmp_path / "h2"
    run_job(
        h2_molecule,
        basis="sto-3g",
        method="rhf",
        output=out_stem,
        structured_log=False,
        progress=False,
        rhf_options=_opts_loose(),
    )
    assert not out_stem.with_suffix(".scf.jsonl").exists()


def test_run_job_structured_log_emits_expected_events(tmp_path, h2_molecule):
    out_stem = tmp_path / "h2"
    run_job(
        h2_molecule,
        basis="sto-3g",
        method="rhf",
        output=out_stem,
        structured_log=True,
        progress=False,
        rhf_options=_opts_loose(),
    )
    records = _read_records(out_stem.with_suffix(".scf.jsonl"))
    events = [r["event"] for r in records]
    # The ordered event family for a successful molecular SCF.
    assert "banner" in events
    assert "job_start" in events
    assert "memory_estimate" in events
    assert "scf_iter" in events
    assert "scf_converged" in events
    assert "job_end" in events
    # banner first, job_end last.
    assert events[0] == "banner"
    assert events[-1] == "job_end"


def test_uhf_spin_diagnostic_reaches_all_serializations(tmp_path):
    """BUG 77: one unrestricted run pins every serialization surface."""
    stem = tmp_path / "oh-spin"
    radical = Molecule(
        [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.83])],
        multiplicity=2,
    )
    previous_level = output_level()
    output_level("verbose")
    try:
        result = run_job(
            radical,
            basis="sto-3g",
            method="uhf",
            output=stem,
            structured_log=True,
            progress=False,
        )
    finally:
        output_level(previous_level)

    expected = {
        "s_squared": result.s_squared,
        "s_squared_ideal": result.s_squared_ideal,
        "s_squared_deviation": result.s_squared_deviation,
    }
    assert expected["s_squared_deviation"] > 0.0

    out_text = stem.with_suffix(".out").read_text(encoding="utf-8")
    assert "<S^2> =" in out_text
    assert "ideal" in out_text
    assert "delta =" in out_text

    records = _read_records(stem.with_suffix(".scf.jsonl"))
    spin_event = next(r for r in records if r["event"] == "spin_squared")
    assert spin_event["value"] == pytest.approx(expected["s_squared"])
    assert spin_event["ideal"] == pytest.approx(expected["s_squared_ideal"])
    assert spin_event["deviation"] == pytest.approx(
        expected["s_squared_deviation"]
    )

    with stem.with_suffix(".system").open("rb") as handle:
        system_manifest = tomllib.load(handle)
    for key, value in expected.items():
        assert system_manifest["run"][key] == pytest.approx(value)

    with zipfile.ZipFile(stem.with_suffix(".qvf")) as archive:
        qvf_manifest = json.loads(archive.read("manifest.json"))
    qvf_spin = qvf_manifest["provenance"]["spin_diagnostic"]
    for key, value in expected.items():
        assert qvf_spin[key] == pytest.approx(value)


def test_run_job_structured_log_records_pass_strict_json(
    tmp_path, h2_molecule,
):
    out_stem = tmp_path / "h2"
    run_job(
        h2_molecule,
        basis="sto-3g",
        method="rhf",
        output=out_stem,
        structured_log=True,
        progress=False,
        rhf_options=_opts_loose(),
    )
    text = out_stem.with_suffix(".scf.jsonl").read_text(encoding="utf-8")
    # Strict JSON: no NaN / Infinity tokens that jq would reject.
    assert "NaN" not in text
    assert "Infinity" not in text
    # Each line round-trips through json.loads cleanly.
    for line in text.splitlines():
        if not line:
            continue
        json.loads(line)


def test_scf_iter_records_have_stable_field_names(tmp_path, h2_molecule):
    out_stem = tmp_path / "h2"
    run_job(
        h2_molecule,
        basis="sto-3g",
        method="rhf",
        output=out_stem,
        structured_log=True,
        progress=False,
        rhf_options=_opts_loose(),
    )
    records = _read_records(out_stem.with_suffix(".scf.jsonl"))
    iters = [r for r in records if r["event"] == "scf_iter"]
    assert len(iters) > 0
    for r in iters:
        for field in ("iter", "energy", "dE", "grad_norm", "diis_subspace"):
            assert field in r, \
                f"scf_iter missing {field!r}: {sorted(r)}"
        # iter 1 should have dE = null (placeholder).
        if int(r["iter"]) == 1:
            assert r["dE"] is None


def test_banner_event_carries_run_fingerprint(tmp_path, h2_molecule):
    out_stem = tmp_path / "h2"
    run_job(
        h2_molecule,
        basis="sto-3g",
        method="rhf",
        output=out_stem,
        structured_log=True,
        progress=False,
        rhf_options=_opts_loose(),
    )
    records = _read_records(out_stem.with_suffix(".scf.jsonl"))
    banner = next(r for r in records if r["event"] == "banner")
    assert "run_fingerprint" in banner
    assert "vibeqc_version" in banner
    assert "libint" in banner
    assert "libxc" in banner
    assert "spglib" in banner
    # Same inputs → same fingerprint as the public helper.
    fp = run_fingerprint(method="rhf", basis="sto-3g",
                         functional=None, molecule=h2_molecule)
    assert banner["run_fingerprint"] == fp


def test_job_end_carries_total_wall_and_path(tmp_path, h2_molecule):
    out_stem = tmp_path / "h2"
    run_job(
        h2_molecule,
        basis="sto-3g",
        method="rhf",
        output=out_stem,
        structured_log=True,
        progress=False,
        rhf_options=_opts_loose(),
    )
    records = _read_records(out_stem.with_suffix(".scf.jsonl"))
    job_end = next(r for r in records if r["event"] == "job_end")
    assert job_end["converged"] is True
    assert job_end["total_wall_s"] > 0.0
    assert job_end["out_path"].endswith("h2.out")


def test_run_job_default_structured_log_is_off(tmp_path, h2_molecule):
    """Backwards-compat: default behavior unchanged when caller doesn't
    opt in. No ``.scf.jsonl`` should appear."""
    out_stem = tmp_path / "h2"
    run_job(
        h2_molecule,
        basis="sto-3g",
        method="rhf",
        output=out_stem,
        progress=False,
        rhf_options=_opts_loose(),
    )
    assert not out_stem.with_suffix(".scf.jsonl").exists()


def test_run_job_explicit_path_routes_correctly(tmp_path, h2_molecule):
    out_stem = tmp_path / "h2"
    explicit = tmp_path / "elsewhere.jsonl"
    run_job(
        h2_molecule,
        basis="sto-3g",
        method="rhf",
        output=out_stem,
        structured_log=str(explicit),
        progress=False,
        rhf_options=_opts_loose(),
    )
    assert explicit.exists()
    assert not out_stem.with_suffix(".scf.jsonl").exists()


def test_env_var_routes_run_job(tmp_path, h2_molecule, monkeypatch):
    # Default structured_log=False; the env var has to win when the
    # caller explicitly opts in (passing False blocks the env var).
    out_stem = tmp_path / "h2"
    env_path = tmp_path / "env.scf.jsonl"
    monkeypatch.setenv("VIBEQC_STRUCTURED_LOG", str(env_path))
    # Pass structured_log=None to defer to the env var.
    run_job(
        h2_molecule,
        basis="sto-3g",
        method="rhf",
        output=out_stem,
        structured_log=None,
        progress=False,
        rhf_options=_opts_loose(),
    )
    assert env_path.exists()


def test_records_round_trip_independently(tmp_path, h2_molecule):
    """Format-stability sanity check: shuffle the records, deserialize
    each one in isolation. Stability of NDJSON shape requires every
    line stand alone."""
    out_stem = tmp_path / "h2"
    run_job(
        h2_molecule,
        basis="sto-3g",
        method="rhf",
        output=out_stem,
        structured_log=True,
        progress=False,
        rhf_options=_opts_loose(),
    )
    text = out_stem.with_suffix(".scf.jsonl").read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln]
    # Reverse the order — each line is still valid.
    for line in reversed(lines):
        rec = json.loads(line)
        assert "event" in rec
        assert "timestamp" in rec


# ---------------------------------------------------------------------------
# Periodic runner: VQ_WORKDIR auto-enable + banner/job_start payload.
# ---------------------------------------------------------------------------

def _h2_periodic() -> tuple[PeriodicSystem, BasisSet]:
    length = 12.0
    c = length / 2.0
    sysp = PeriodicSystem(
        3, np.eye(3) * length,
        [Atom(1, [c, c, c - 0.7]), Atom(1, [c, c, c + 0.7])],
    )
    return sysp, BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def test_periodic_job_start_carries_resolved_method(tmp_path, monkeypatch):
    """Regression for the periodic job_start NameError: the banner /
    job_start emission referenced an undefined ``resolved_method`` (the
    periodic runner resolves its method as ``method_upper``), so every
    periodic ``.out`` write crashed before SCF. Introduced e14cf70ce /
    ceb3fb49c, fixed cecfec321. Pins the VQ_WORKDIR auto-enable route
    end-to-end and the method identifier the records carry."""
    monkeypatch.setenv("VQ_WORKDIR", str(tmp_path))
    sysp, basis = _h2_periodic()
    stem = tmp_path / "per_h2"
    run_periodic_job(
        sysp, basis, method="RHF", jk_method="bipole", kpoints=(1, 1, 1),
        max_iter=20, output=stem, output_qvf=False, write_density=False,
        citations=False, progress=False,
    )
    records = _read_records(stem.with_suffix(".scf.jsonl"))
    events = [r["event"] for r in records]
    assert events[0] == "banner"
    assert "job_start" in events
    job_start = next(r for r in records if r["event"] == "job_start")
    assert job_start["method"] == "RHF"
    assert job_start["periodic"] is True
    assert job_start["jk_method"] == "BIPOLE"
    # The banner's fingerprint hashes the same resolved method.
    assert records[0]["run_fingerprint"] == run_fingerprint(
        method="RHF",
        basis=basis.name,
        functional=None,
        molecule=sysp.unit_cell_molecule(),
    )
