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


@pytest.mark.parametrize("converged", [False, True])
def test_periodic_result_status_is_separate_from_delivery(tmp_path, converged):
    from types import SimpleNamespace
    from vibeqc.periodic_runner import (
        _PERIODIC_OUTPUT_WRITER, _periodic_output_lifecycle,
    )

    plan = _build_plan(tmp_path)

    @_periodic_output_lifecycle
    def calculation():
        writer = OutputWriter(plan)
        _PERIODIC_OUTPUT_WRITER.set(writer)
        _mark_guaranteed_written(plan, writer.record)
        return SimpleNamespace(converged=converged)

    assert calculation().converged is converged
    outputs = _read_manifest(tmp_path / "h2o.system")["outputs"]
    assert outputs["status"] == "complete"
    assert outputs["result_status"] == ("converged" if converged else "not_converged")
    assert outputs["result_error"] == ""


def test_delivery_failure_preserves_known_calculation_outcome(tmp_path):
    updater = ManifestUpdater(_build_plan(tmp_path))
    updater.set_result_status("not_converged")
    with pytest.raises(IncompleteOutputError):
        updater.finish()
    updater.crash()
    outputs = _read_manifest(updater.path)["outputs"]
    assert outputs["status"] == "crashed"
    assert outputs["result_status"] == "not_converged"


def test_unreported_result_is_not_inferred_from_delivery(tmp_path):
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)
    _mark_guaranteed_written(plan, updater.mark_written)
    updater.finish()
    assert _read_manifest(updater.path)["outputs"]["result_status"] == "not_reported"


def test_periodic_preflight_refusal_writes_manifest(tmp_path):
    import numpy as np
    import vibeqc as vq

    system = vq.PeriodicSystem(3, np.eye(3) * 12., [vq.Atom(1, [0., 0., 0.])])
    stem = tmp_path / "refused.v1"
    with pytest.raises(NotImplementedError):
        vq.run_periodic_job(
            system, None, method="om2", kpoints=(2, 1, 1),
            output=stem, record_hostname=False,
        )
    body = _read_manifest(tmp_path / "refused.v1.system")
    assert body["outputs"]["status"] == "crashed"
    assert body["outputs"]["result_status"] == "refused"
    assert "NotImplementedError" in body["outputs"]["result_error"]
    assert body["run"]["refusal_stage"] == "preflight"
    assert body["plan"]["method"] == "om2"
    assert body["host"]["hostname"] == "<redacted>"


def test_refusal_recording_failure_preserves_original_error(tmp_path, monkeypatch):
    import vibeqc.periodic_runner as runner

    def unwritable(*args, **kwargs):
        raise OSError("cannot write manifest")

    monkeypatch.setattr(runner, "OutputWriter", unwritable)

    @runner._periodic_output_lifecycle
    def refused(**kwargs):
        raise NotImplementedError("unsupported capability")

    with pytest.warns(UserWarning, match="cannot write manifest"):
        with pytest.raises(NotImplementedError, match="unsupported capability"):
            refused(output=tmp_path / "unwritable")


@pytest.mark.parametrize("existing_writer", [False, True])
@pytest.mark.parametrize("warnings_as_errors", [False, True])
def test_refusal_update_failure_preserves_original_error(
    tmp_path, monkeypatch, existing_writer, warnings_as_errors,
):
    import warnings
    import vibeqc.periodic_runner as runner

    def unwritable(*args, **kwargs):
        raise OSError("cannot update manifest")

    monkeypatch.setattr(OutputWriter, "set_result_status", unwritable)

    @runner._periodic_output_lifecycle
    def refused(**kwargs):
        if existing_writer:
            runner._PERIODIC_OUTPUT_WRITER.set(OutputWriter(_build_plan(tmp_path)))
        raise NotImplementedError("unsupported capability")

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("error" if warnings_as_errors else "always")
        with pytest.raises(NotImplementedError, match="unsupported capability"):
            refused(output=tmp_path / "unwritable")
    if not warnings_as_errors:
        assert any("cannot update manifest" in str(w.message) for w in emitted)
    path = tmp_path / ("h2o.system" if existing_writer else "unwritable.system")
    assert _read_manifest(path)["outputs"]["status"] == "crashed"


@pytest.mark.parametrize("error_type", [OSError, NotImplementedError])
def test_periodic_delivery_error_keeps_driver_convergence(tmp_path, monkeypatch, error_type):
    from types import SimpleNamespace
    import numpy as np
    import vibeqc as vq
    import vibeqc.pbc_bipole as bipole
    import vibeqc.periodic_runner as runner

    system = vq.PeriodicSystem(3, np.eye(3) * 12., [
        vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    monkeypatch.setattr(
        bipole, "run_pbc_bipole_rhf",
        lambda *args, **kwargs: SimpleNamespace(converged=True, scf_trace=[]),
    )

    def write_failure(*args, **kwargs):
        raise error_type("SCF output delivery failed")

    monkeypatch.setattr(runner, "write_scf_trace", write_failure)
    with pytest.raises(error_type, match="SCF output delivery failed"):
        vq.run_periodic_job(
            system, basis, jk_method="bipole", output=tmp_path / "delivery",
            progress=False, record_hostname=False,
        )
    outputs = _read_manifest(tmp_path / "delivery.system")["outputs"]
    assert outputs["status"] == "crashed"
    assert outputs["result_status"] == "converged"
    assert "SCF output delivery failed" in outputs["result_error"]


def test_bipole_nonconvergence_is_recorded_in_manifest(tmp_path):
    import numpy as np
    import vibeqc as vq

    system = vq.PeriodicSystem(3, np.eye(3) * 12., [
        vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.raises(RuntimeError, match="SCF did not converge after 1 iterations"):
        vq.run_periodic_job(
            system, basis, jk_method="bipole", max_iter=1, initial_guess="HCORE",
            bipole_cutoff_bohr=8., output=tmp_path / "capped", output_qvf=False,
            progress=False, record_hostname=False,
        )
    outputs = _read_manifest(tmp_path / "capped.system")["outputs"]
    assert outputs["status"] == "crashed"
    assert outputs["result_status"] == "not_converged"
    assert "SCF did not converge" in outputs["result_error"]


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


@pytest.mark.parametrize("legacy", [False, True])
def test_manifest_round_trips_numpy_source_coordinates(tmp_path: Path, legacy: bool) -> None:
    """Executed source coordinates stay numbers, including with NumPy 2 repr."""
    import json
    import numpy as np

    coordinates = tuple(np.asarray([0.0, -1.25, 1.0e-12], dtype=np.float64))
    source = {
        "q_grouping": [(coordinates, ((0, 0),))],
        "kpoints_cart": [coordinates],
        "metric_threshold": np.float64(1.0e-9),
        "executed_domains_json": json.dumps({"kernel": "periodic Coulomb"}),
    }
    if legacy:
        from vibeqc.output.formats.system_info import _format_manifest
        parsed = tomllib.loads(_format_manifest({"run": {"executed_fit_source": source}}))
    else:
        updater = ManifestUpdater(_build_plan(tmp_path))
        updater.update_run_fields({"executed_fit_source": source})
        parsed = _read_manifest(updater.path)

    restored = parsed["run"]["executed_fit_source"]
    assert restored["q_grouping"] == [[[0.0, -1.25, 1.0e-12], [[0, 0]]]]
    assert restored["kpoints_cart"] == [[0.0, -1.25, 1.0e-12]]
    assert all(type(value) is float for value in restored["kpoints_cart"][0])
    assert restored["metric_threshold"] == 1.0e-9
    assert restored["executed_domains_json"] == source["executed_domains_json"]


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


# ---------------------------------------------------------------------------
# Rewrites stay in the job directory after the process changes directory
# (vibe-qc#127: a ``pilot.system`` escaped into the checkout root while
# ``pilot.out`` landed in the job's tmp_path)
# ---------------------------------------------------------------------------


def _relative_plan(stem: str) -> OutputPlan:
    """A plan spelled the way ``run_job(output="pilot")`` spells it."""
    return OutputPlan.from_run_job_kwargs(
        output=stem, method="rhf", basis="sto-3g", functional=None,
    )


def _job_and_elsewhere(tmp_path: Path) -> tuple[Path, Path]:
    job_dir = tmp_path / "job"
    elsewhere = tmp_path / "elsewhere"
    job_dir.mkdir()
    elsewhere.mkdir()
    return job_dir, elsewhere


def test_relative_stem_is_anchored_to_the_job_start_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir, _ = _job_and_elsewhere(tmp_path)
    monkeypatch.chdir(job_dir)
    updater = ManifestUpdater(_relative_plan("pilot"))
    assert updater.path.is_absolute()
    assert updater.path.resolve() == (job_dir / "pilot.system").resolve()
    assert (job_dir / "pilot.system").is_file()


def test_deferred_rewrites_stay_in_the_job_directory_after_chdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Progress heartbeat, wall-clock stamp and finish all fire after the
    cwd moved. Each must rewrite the manifest where the job started, and
    none may drop a copy where the process happens to be now."""
    job_dir, elsewhere = _job_and_elsewhere(tmp_path)
    monkeypatch.chdir(job_dir)
    plan = _relative_plan("pilot")
    updater = ManifestUpdater(plan)
    _mark_guaranteed_written(plan, updater.mark_written)

    monkeypatch.chdir(elsewhere)
    updater.update_progress(phase="scf", iteration=3, energy_eh=-1.0)
    updater.update_wall_seconds(2.0)
    updater.update_run_fields({"late_bound": "yes"})
    updater.finish(wall_seconds=3.0)

    assert sorted(p.name for p in elsewhere.iterdir()) == []
    body = _read_manifest(job_dir / "pilot.system")
    assert body["outputs"]["status"] == "complete"
    assert body["progress"]["iteration"] == 3
    assert body["run"]["late_bound"] == "yes"
    assert body["run"]["wall_seconds"] == pytest.approx(3.0)


def test_anchoring_does_not_leak_absolute_paths_into_the_manifest_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The anchor is an on-disk concern: rendered rows keep the declared
    relative spelling and the self row stays a single row.

    This guards the fix's design rather than the original symptom. A fix
    that made ``_path`` absolute and looked the self row up by it would
    miss the declared ``pilot.system`` row after a chdir and append an
    absolute duplicate; a consumer keyed on the declared spelling would
    then read a stale ``written = false`` row."""
    job_dir, elsewhere = _job_and_elsewhere(tmp_path)
    monkeypatch.chdir(job_dir)
    plan = _relative_plan("pilot")
    updater = ManifestUpdater(plan)
    _mark_guaranteed_written(plan, updater.mark_written)
    monkeypatch.chdir(elsewhere)
    updater.finish()

    body = _read_manifest(job_dir / "pilot.system")
    plan_paths = [row["path"] for row in body["plan"]["files"]]
    out_paths = [row["path"] for row in body["outputs"]["files"]]
    assert "pilot.system" in plan_paths
    assert "pilot.out" in plan_paths
    assert out_paths.count("pilot.system") == 1
    assert "pilot.out" in out_paths
    assert not any(Path(p).is_absolute() for p in plan_paths + out_paths)
    self_row = next(
        row for row in body["outputs"]["files"] if row["path"] == "pilot.system"
    )
    assert self_row["written"] is True
    assert self_row["checksum_status"] == "self-excluded"


def test_mark_written_probes_the_job_directory_after_chdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recording a plan-relative artefact from another cwd must hash the
    file in the job directory, not report it missing."""
    job_dir, elsewhere = _job_and_elsewhere(tmp_path)
    monkeypatch.chdir(job_dir)
    updater = ManifestUpdater(_relative_plan("pilot"))
    payload = b"reference energy -1.0\n"
    (job_dir / "pilot.out").write_bytes(payload)

    monkeypatch.chdir(elsewhere)
    updater.mark_written(Path("pilot.out"), wall_time_s=0.5)

    assert not (elsewhere / "pilot.system").exists()
    rows = {
        row["path"]: row
        for row in _read_manifest(job_dir / "pilot.system")["outputs"]["files"]
    }
    assert rows["pilot.out"]["written"] is True
    assert rows["pilot.out"]["bytes"] == len(payload)
    assert rows["pilot.out"]["checksum_status"] == "sha256"
    assert len(rows["pilot.out"]["sha256"]) == 64


@pytest.mark.parametrize("spelling", ["absolute", "relative"])
def test_self_record_after_chdir_matches_the_declared_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    """Recording the manifest itself from another cwd, by either its
    absolute on-disk spelling or its declared relative one, must hit the
    declared ``pilot.system`` row, never hash the manifest, and never
    append a second row."""
    import vibeqc.output.manifest as manifest_module

    job_dir, elsewhere = _job_and_elsewhere(tmp_path)
    monkeypatch.chdir(job_dir)
    updater = ManifestUpdater(_relative_plan("pilot"))
    monkeypatch.chdir(elsewhere)

    def _must_not_hash(path: Path) -> tuple[int | None, str | None]:
        raise AssertionError(f"attempted impossible self-hash for {path}")

    monkeypatch.setattr(manifest_module, "_stat_and_hash", _must_not_hash)
    target = updater.path if spelling == "absolute" else Path("pilot.system")
    updater.mark_written(target)

    assert not (elsewhere / "pilot.system").exists()
    body = _read_manifest(job_dir / "pilot.system")
    paths = [row["path"] for row in body["outputs"]["files"]]
    assert paths.count("pilot.system") == 1
    assert not any(Path(p).is_absolute() for p in paths)
    self_row = next(r for r in body["outputs"]["files"] if r["path"] == "pilot.system")
    assert self_row["written"] is True
    assert self_row["checksum_status"] == "self-excluded"
    assert self_row["bytes"] == 0
    assert self_row["sha256"] == ""


def test_output_writer_manifest_path_survives_chdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coordinator's view is the same anchored path, and its own
    late calls (heartbeat, finish) land in the job directory."""
    job_dir, elsewhere = _job_and_elsewhere(tmp_path)
    monkeypatch.chdir(job_dir)
    plan = _relative_plan("pilot")
    w = OutputWriter(plan)
    assert w.manifest_path.is_absolute()
    assert w.manifest_path.resolve() == (job_dir / "pilot.system").resolve()
    _mark_guaranteed_written(plan, w.record)

    monkeypatch.chdir(elsewhere)
    w.update_progress(phase="scf", iteration=1, energy_eh=-1.0)
    w.finish()

    assert sorted(p.name for p in elsewhere.iterdir()) == []
    assert _read_manifest(job_dir / "pilot.system")["outputs"]["status"] == "complete"


# ---------------------------------------------------------------------------
# One verdict contract across every writer (#342 #346 #351).
#
# Pre-fix a certified-stable UHF run and one whose stability analysis produced
# no verdict were byte-identical in .system (``status = "complete"``,
# ``result_status = "not_reported"``), in the QVF (``run_status =
# "converged"``) and in ``job_end``; only .out prose and a warning event
# differed, so a consumer gating on the structured record ingested every
# UNVERIFIED species as complete (#346).
# ---------------------------------------------------------------------------


def _oh_doublet():
    from vibeqc import Atom, Molecule

    ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
    return Molecule(
        [Atom(8, [0.0, 0.0, 0.0]),
         Atom(1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR])],
        charge=0,
        multiplicity=2,
    )


@pytest.mark.parametrize(
    "cap, stability, usable, reason",
    [(60, "verified_stable", True, "converged"), (0, "unverified", False, "not_run")],
)
def test_molecular_run_carries_one_verdict_into_every_writer(
    tmp_path, monkeypatch, cap, stability, usable, reason
):
    import json
    import zipfile

    import vibeqc as vq

    monkeypatch.chdir(tmp_path)
    opts = vq.UHFOptions()
    opts.max_iter = 200
    opts.stability_davidson_max_iter = cap
    vq.run_job(
        _oh_doublet(), basis="sto-3g", method="uhf", uhf_options=opts,
        output="verdict", structured_log="verdict.ndjson", record_hostname=False,
    )

    body = _read_manifest(tmp_path / "verdict.system")
    assert body["outputs"]["status"] == "complete"
    assert body["outputs"]["result_status"] == "converged"
    verdict = body["verdict"]
    assert verdict["scf"] == "converged"
    assert verdict["stability"] == stability
    assert verdict["usable"] is usable
    assert verdict["stability_reason"] == reason
    assert verdict["stability_matvec_cap"] == cap
    # D1: the automatic budget is max(600 s, 3.0 x the SCF loop wall); a
    # sub-second SCF sits on the floor.
    assert verdict["stability_wall_budget_s"] == pytest.approx(600.0)

    with zipfile.ZipFile(tmp_path / "verdict.qvf") as archive:
        provenance = json.loads(archive.read("manifest.json"))["provenance"]
    assert provenance["run_status"] == "converged"
    assert provenance["verdict"] == verdict

    events = [
        json.loads(line)
        for line in (tmp_path / "verdict.ndjson").read_text().splitlines()
        if line.strip()
    ]
    (job_end,) = [e for e in events if e.get("event") == "job_end"]
    assert job_end["verdict"] == verdict
    (scf_converged,) = [e for e in events if e.get("event") == "scf_converged"]
    assert scf_converged["verdict"] == verdict

    from vibeqc.output.verdict import usable_from_manifest

    assert usable_from_manifest(body) == ("usable" if usable else "not_usable")


def test_molecular_nonconvergence_is_recorded_before_the_crash(tmp_path, monkeypatch):
    import vibeqc as vq

    monkeypatch.chdir(tmp_path)
    opts = vq.UHFOptions()
    opts.max_iter = 1
    with pytest.raises(RuntimeError, match="did not converge"):
        vq.run_job(
            _oh_doublet(), basis="sto-3g", method="uhf", uhf_options=opts,
            output="capped", output_qvf=False, record_hostname=False,
        )
    outputs = _read_manifest(tmp_path / "capped.system")["outputs"]
    assert outputs["status"] == "crashed"
    assert outputs["result_status"] == "not_converged"
    assert "did not converge" in outputs["result_error"]
    verdict = _read_manifest(tmp_path / "capped.system")["verdict"]
    assert verdict["scf"] == "not_converged"
    assert verdict["usable"] is False


def test_molecular_preflight_refusal_writes_manifest(tmp_path):
    """#351 criterion 2, molecular half: a refusal raised before the output
    plan exists leaves the same preflight receipt the periodic runner leaves."""
    import vibeqc as vq
    from tests.refusal_receipt import assert_preflight_refusal

    stem = tmp_path / "refused"
    with pytest.raises(NotImplementedError) as info:
        vq.run_job(
            _oh_doublet(), basis="sto-3g", method="rohf",
            tddft=True, tddft_type="casida", output=stem, record_hostname=False,
        )
    assert_preflight_refusal(stem, info.value)
    body = _read_manifest(tmp_path / "refused.system")
    assert body["plan"]["method"] == "rohf"
    assert body["verdict"]["scf"] == "not_reported"
    assert body["verdict"]["usable"] is False


def test_verdict_section_is_always_present_and_derived_when_unset(tmp_path):
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)
    body = _read_manifest(updater.path)
    assert body["verdict"]["scf"] == "not_reported"
    assert body["verdict"]["stability"] == "not_reported"
    assert body["verdict"]["usable"] is False
    updater.set_result_status("converged")
    body = _read_manifest(updater.path)
    assert body["verdict"]["scf"] == "converged"
    assert body["verdict"]["usable"] is True
    updater.set_verdict({
        "scf": "converged", "stability": "unverified",
        "stability_reason": "wall_budget", "stability_eigenvalue": 0.09,
        "stability_restarts": 0, "stability_matvec_used": 17,
        "stability_matvec_cap": 60, "stability_wall_s": 612.4,
        "stability_wall_budget_s": 600.0,
    })
    body = _read_manifest(updater.path)
    assert body["verdict"]["stability"] == "unverified"
    assert body["verdict"]["usable"] is False
    assert body["verdict"]["stability_matvec_used"] == 17
    with pytest.raises(ValueError):
        updater.set_verdict({"scf": "converged", "stability": "fine",
                             "stability_reason": "converged"})


def test_delivery_failure_preserves_a_recorded_verdict(tmp_path):
    """#346 criterion 3: a crashed delivery keeps the usable SCF verdict."""
    plan = _build_plan(tmp_path)
    updater = ManifestUpdater(plan)
    updater.set_result_status("converged")
    updater.set_verdict(
        {"scf": "converged", "stability": "verified_stable",
         "stability_reason": "converged"}
    )
    with pytest.raises(IncompleteOutputError):
        updater.finish()
    updater.crash()
    body = _read_manifest(updater.path)
    assert body["outputs"]["status"] == "crashed"
    assert body["outputs"]["result_status"] == "converged"
    assert body["verdict"]["usable"] is True
    from vibeqc.output.verdict import usable_from_manifest

    assert usable_from_manifest(body) == "usable"
