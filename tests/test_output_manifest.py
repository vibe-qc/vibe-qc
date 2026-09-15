"""``vibeqc.output.ManifestUpdater`` + ``OutputWriter`` — extended .system.

Pins the contract documented in
``docs/design_output_module.md § .system schema extension``:

  1. Constructing a ManifestUpdater writes ``{stem}.system`` with the
     full runtime-env block + ``[plan]`` + ``[outputs].status="running"``.
  2. ``mark_written`` rewrites the manifest atomically with the new
     ``[[outputs.files]]`` row populated (bytes, sha256/checksum state,
     wall_time_s).
  3. ``finish`` flips ``[outputs].status`` to ``"complete"`` and stamps
     ``finished_at_iso``.
  4. ``crash`` flips ``[outputs].status`` to ``"crashed"``.
  5. The manifest round-trips through stdlib ``tomllib`` at every state
     transition — no stage produces a half-written file.
  6. ``OutputWriter.context()`` calls ``finish`` on clean exit and
     ``crash`` on exception (and the exception still re-raises).
  7. The pre-Phase-O1 ``[vibeqc] [host] [cpu] [memory] [python]
     [libraries] [validation] [run]`` sections are still present and
     unchanged in shape — additivity contract.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from vibeqc.output import (
    IncompleteOutputError,
    ManifestUpdater,
    OutputPlan,
    OutputWriter,
    write_initial_manifest,
)


# Sections that vibeqc.system_info has always emitted and that the
# Phase O1 extension must not break.
_LEGACY_SECTIONS = {
    "vibeqc", "host", "cpu", "memory", "python", "libraries",
    "validation", "run",
}


def _build_plan(tmp_path: Path) -> OutputPlan:
    return OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf", basis="sto-3g", functional=None,
    )


def _read_manifest(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _mark_guaranteed_written(
    plan: OutputPlan,
    recorder,
    *,
    exclude: tuple[Path, ...] = (),
) -> None:
    """Materialize and record every guaranteed non-manifest test artefact."""
    for planned in plan.guaranteed_files():
        if planned.role == "manifest" or planned.path in exclude:
            continue
        planned.path.parent.mkdir(parents=True, exist_ok=True)
        planned.path.write_text(
            f"test payload: {planned.role}/{planned.format}\n",
            encoding="utf-8",
        )
        recorder(planned.path)


# ---------------------------------------------------------------------------
# Initial write
# ---------------------------------------------------------------------------

def test_initial_manifest_includes_plan_and_outputs(
    tmp_path: Path,
) -> None:
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)
    assert updater.path == (tmp_path / "h2o").with_suffix(".system")
    assert updater.path.is_file()
    body = _read_manifest(updater.path)
    assert "plan" in body
    assert "outputs" in body
    assert body["outputs"]["status"] == "running"


def test_basis_library_section_is_present_even_with_nothing_fetched(
    tmp_path: Path,
) -> None:
    """The section never disappears, so a consumer branches on
    ``fetched_count`` rather than on the section being there at all."""
    updater = ManifestUpdater(_build_plan(tmp_path))
    body = _read_manifest(updater.path)
    assert body["basis_library"]["fetched_count"] == 0
    assert body["basis_library"]["resolved_root"] == ""


def test_set_basis_library_records_the_root_and_the_rendered_rows(
    tmp_path: Path,
) -> None:
    updater = ManifestUpdater(_build_plan(tmp_path))
    updater.set_basis_library(
        resolved_root="/somewhere/basis_library",
        fetched=[{
            "name": "sadlej pvtz",
            "bse_version": "0.12",
            "element_count": 25,
            "has_ecp": False,
            "provenance_path": "/cache/basis/sadlej pvtz.provenance.json",
        }],
    )
    section = _read_manifest(updater.path)["basis_library"]
    assert section["resolved_root"] == "/somewhere/basis_library"
    assert section["fetched_count"] == 1
    row = section["fetched"][0]
    assert row["name"] == "sadlej pvtz"
    assert row["bse_version"] == "0.12"
    assert row["has_ecp"] is False
    assert row["provenance_path"].endswith(".provenance.json")


def test_the_basis_library_section_survives_a_later_rewrite(tmp_path: Path) -> None:
    """The manifest is rewritten on every artefact; provenance set once
    early must not be dropped by a later rewrite."""
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)
    updater.set_basis_library(resolved_root="/somewhere", fetched=())
    updater.update_wall_seconds(1.25)
    assert _read_manifest(updater.path)["basis_library"]["resolved_root"] == "/somewhere"


def test_initial_manifest_preserves_legacy_sections(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)
    body = _read_manifest(updater.path)
    for section in _LEGACY_SECTIONS:
        assert section in body, (
            f"legacy section {section!r} missing — "
            "Phase O1 must be additive"
        )


def test_plan_section_carries_method_basis_files(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)
    body = _read_manifest(updater.path)
    p = body["plan"]
    assert p["method"] == "RHF"
    assert p["basis"] == "sto-3g"
    assert p["functional"] == ""
    assert isinstance(p["files"], list)
    assert len(p["files"]) == len(plan.files)
    # Each file row has the documented fields.
    for row in p["files"]:
        assert {"role", "path", "format", "always", "description"} \
            <= set(row.keys())


def test_initial_outputs_files_seeded_for_every_planned_file(
    tmp_path: Path,
) -> None:
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)
    body = _read_manifest(updater.path)
    outputs_paths = [f["path"] for f in body["outputs"]["files"]]
    plan_paths = [f["path"] for f in body["plan"]["files"]]
    assert sorted(outputs_paths) == sorted(plan_paths)
    # None written yet — all rows say written=false with pending checksums.
    for row in body["outputs"]["files"]:
        assert row["written"] is False
        assert row["checksum_status"] == "pending"


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

def test_mark_written_updates_outputs_row(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)
    # Create a real file on disk and report it.
    out_path = (tmp_path / "h2o").with_suffix(".out")
    out_path.write_text("hello vibe-qc")
    updater.mark_written(out_path, wall_time_s=0.123)
    body = _read_manifest(updater.path)
    rows = {row["path"]: row for row in body["outputs"]["files"]}
    matched = rows[str(out_path)]
    assert matched["written"] is True
    assert matched["bytes"] == len("hello vibe-qc")
    assert matched["sha256"]            # non-empty
    assert matched["checksum_status"] == "sha256"
    assert matched["wall_time_s"] == pytest.approx(0.123)


def test_finish_flips_status_to_complete(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)
    _mark_guaranteed_written(plan, updater.mark_written)
    updater.finish(wall_seconds=4.5)
    body = _read_manifest(updater.path)
    assert body["outputs"]["status"] == "complete"
    assert body["outputs"]["finished_at_iso"]   # non-empty timestamp
    assert body["run"]["wall_seconds"] == pytest.approx(4.5)
    rows = {row["path"]: row for row in body["outputs"]["files"]}
    self_row = rows[str(updater.path)]
    assert self_row["written"] is True
    assert self_row["bytes"] == 0
    assert self_row["sha256"] == ""
    assert self_row["checksum_status"] == "self-excluded"


def test_manifest_self_record_never_hashes_pre_finalization_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest cannot truthfully contain a digest of its final bytes."""
    import vibeqc.output.manifest as manifest_module

    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)
    _mark_guaranteed_written(plan, updater.mark_written)

    def _must_not_hash(path: Path) -> tuple[int | None, str | None]:
        raise AssertionError(f"attempted impossible self-hash for {path}")

    monkeypatch.setattr(manifest_module, "_stat_and_hash", _must_not_hash)
    updater.mark_written(updater.path)
    updater.finish()

    body = _read_manifest(updater.path)
    rows = {row["path"]: row for row in body["outputs"]["files"]}
    self_row = rows[str(updater.path)]
    assert self_row["written"] is True
    assert self_row["bytes"] == 0
    assert self_row["sha256"] == ""
    assert self_row["checksum_status"] == "self-excluded"


def test_crash_flips_status_to_crashed(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)
    updater.crash(wall_seconds=0.8)
    body = _read_manifest(updater.path)
    assert body["outputs"]["status"] == "crashed"
    assert body["outputs"]["finished_at_iso"]


def test_undeclared_files_are_appended_to_outputs(tmp_path: Path) -> None:
    """``mark_written`` is permissive — a file not in the plan still
    surfaces to vq via the manifest. The CI gate that asserts "no
    undeclared outputs" runs against the plan, not the manifest."""
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)
    extra = tmp_path / "h2o.bonus.log"
    extra.write_text("logfile")
    updater.mark_written(extra)
    body = _read_manifest(updater.path)
    rows = {row["path"]: row for row in body["outputs"]["files"]}
    assert str(extra) in rows
    assert rows[str(extra)]["written"] is True


def test_manifest_round_trips_through_tomllib_at_every_state(
    tmp_path: Path,
) -> None:
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)
    # initial
    tomllib.loads(updater.path.read_text(encoding="utf-8"))
    # after a mark_written
    f = (tmp_path / "h2o").with_suffix(".out")
    f.write_text("body")
    updater.mark_written(f)
    tomllib.loads(updater.path.read_text(encoding="utf-8"))
    # after finish
    _mark_guaranteed_written(plan, updater.mark_written)
    updater.finish()
    tomllib.loads(updater.path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# OutputWriter coordinator wrapping
# ---------------------------------------------------------------------------

def test_output_writer_initial_status(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    w = OutputWriter(plan)
    assert w.status == "running"
    assert w.manifest_path.is_file()


def test_output_writer_context_clean_exit_marks_complete(
    tmp_path: Path,
) -> None:
    plan = _build_plan(tmp_path)
    w = OutputWriter(plan)
    with w.context():
        _mark_guaranteed_written(plan, w.record)
    assert w.status == "complete"


def test_finish_rejects_pending_guaranteed_files(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "required",
        method="rhf",
        basis="sto-3g",
        functional=None,
        write_xyz=False,
        citations=False,
        write_population=False,
        crash_dump=False,
        output_qvf=False,
    )
    updater = ManifestUpdater(plan)
    molden = (tmp_path / "required").with_suffix(".molden")
    _mark_guaranteed_written(plan, updater.mark_written, exclude=(molden,))

    with pytest.raises(IncompleteOutputError, match="required.molden") as exc:
        updater.finish(wall_seconds=1.25)

    assert exc.value.paths == (molden,)
    body = _read_manifest(updater.path)
    assert body["outputs"]["status"] == "crashed"
    rows = {row["path"]: row for row in body["outputs"]["files"]}
    assert rows[str(molden)]["written"] is False
    assert rows[str(molden)]["checksum_status"] == "failed"
    assert "guaranteed artefact" in rows[str(molden)]["error"]


def test_finish_allows_pending_conditional_files(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "conditional",
        method="rhf",
        basis="sto-3g",
        functional=None,
        write_molden_file=False,
        write_xyz=False,
        citations=False,
        write_population=False,
        crash_dump=True,
        output_qvf=False,
    )
    updater = ManifestUpdater(plan)
    _mark_guaranteed_written(plan, updater.mark_written)

    updater.finish()

    body = _read_manifest(updater.path)
    assert body["outputs"]["status"] == "complete"
    crash_rows = [
        row for row in body["outputs"]["files"]
        if row["path"].endswith(".dump")
    ]
    assert len(crash_rows) == 1
    assert crash_rows[0]["written"] is False
    assert crash_rows[0]["checksum_status"] == "pending"


def test_output_writer_context_exception_marks_crashed(
    tmp_path: Path,
) -> None:
    plan = _build_plan(tmp_path)
    w = OutputWriter(plan)
    with pytest.raises(RuntimeError, match="boom"):
        with w.context():
            raise RuntimeError("boom")
    assert w.status == "crashed"
    # The exception still propagates — bookkeeping must not swallow.
    body = _read_manifest(w.manifest_path)
    assert body["outputs"]["status"] == "crashed"


def test_output_writer_record_threads_wall_time(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    w = OutputWriter(plan)
    f = (tmp_path / "h2o").with_suffix(".out")
    f.write_text("ok")
    w.record(f)
    body = _read_manifest(w.manifest_path)
    rows = {row["path"]: row for row in body["outputs"]["files"]}
    matched = rows[str(f)]
    assert matched["written"] is True
    assert matched["wall_time_s"] >= 0.0


def test_output_writer_updates_late_bound_run_fields(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    w = OutputWriter(
        plan,
        extra_run_fields={"jk_method_executed": "pending"},
    )
    w.update_run_fields(
        {"jk_method_executed": "gdf", "dft_plus_u_route": "none"},
    )
    body = _read_manifest(w.manifest_path)
    assert body["run"]["jk_method_executed"] == "gdf"
    assert body["run"]["dft_plus_u_route"] == "none"


# ---------------------------------------------------------------------------
# write_initial_manifest convenience entry point
# ---------------------------------------------------------------------------

def test_write_initial_manifest_returns_an_updater(tmp_path: Path) -> None:
    plan = _build_plan(tmp_path)
    updater = write_initial_manifest(plan)
    assert isinstance(updater, ManifestUpdater)
    assert updater.path.is_file()


# ---------------------------------------------------------------------------
# system_info() is probed once, not per manifest rewrite
# ---------------------------------------------------------------------------

def test_system_info_probed_once_across_rewrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime-env probe (`system_info()` — does subprocess
    sysctl / platform calls) is invariant across a run. The
    ManifestUpdater snapshots it once at construction; the
    coordinator's per-artefact rewrites (record / finish / crash)
    must reuse the snapshot, not re-probe. Regression guard for the
    pre-v1.0 dispatch-overhaul, where run_job rewrites the manifest
    N times per job instead of once."""
    import vibeqc.output.manifest as _mani

    calls = {"n": 0}
    real = _mani.system_info

    def _counting(*a: object, **k: object):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(_mani, "system_info", _counting)

    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)        # 1 probe at construction
    # Five rewrites — none should re-probe.
    for suffix in (".out", ".molden", ".xyz", ".bibtex", ".references"):
        f = (tmp_path / "h2o").with_suffix(suffix)
        f.write_text("payload")
        updater.mark_written(f)
    _mark_guaranteed_written(plan, updater.mark_written)
    updater.finish(wall_seconds=1.0)

    assert calls["n"] == 1, (
        f"system_info() probed {calls['n']}x — expected exactly 1 "
        "(snapshot at construction, reused on every rewrite)"
    )


# ---------------------------------------------------------------------------
# [progress] section
# ---------------------------------------------------------------------------


def test_progress_section_written_and_readable(tmp_path: Path) -> None:
    """update_progress writes a [progress] section with the given fields."""
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)

    updater.update_progress(
        phase="scf", iteration=6, energy_eh=-73.4963596554,
        gradient_norm=3.3e-08, diis_subspace=6,
    )

    body = _read_manifest(updater.path)
    assert "progress" in body
    p = body["progress"]
    assert p["phase"] == "scf"
    assert p["iteration"] == 6
    assert p["gradient_norm"] == pytest.approx(3.3e-08)
    assert p["diis_subspace"] == 6
    # energy_eh is a float in TOML; check approximate equality.
    assert p["energy_eh"] == pytest.approx(-73.4963596554)


def test_progress_section_absent_by_default(tmp_path: Path) -> None:
    """When update_progress is never called, [progress] is absent."""
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)
    body = _read_manifest(updater.path)
    assert "progress" not in body


def test_progress_section_cleared_with_no_fields(tmp_path: Path) -> None:
    """Calling update_progress() with no fields removes the section."""
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)

    updater.update_progress(iteration=1, energy_eh=-73.0)
    body = _read_manifest(updater.path)
    assert "progress" in body

    updater.update_progress()  # clear
    body = _read_manifest(updater.path)
    assert "progress" not in body


def test_progress_survives_multiple_updates(tmp_path: Path) -> None:
    """Each update_progress call overwrites the section atomically."""
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)

    updater.update_progress(iteration=1, energy_eh=-73.48)
    body = _read_manifest(updater.path)
    assert body["progress"]["iteration"] == 1

    updater.update_progress(iteration=6, energy_eh=-73.49)
    body = _read_manifest(updater.path)
    assert body["progress"]["iteration"] == 6
