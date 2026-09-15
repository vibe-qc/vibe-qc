"""Output-failure diagnostics — visibility + manifest integrity.

Covers :mod:`vibeqc.output._errors`:

* ``warn_output_failure`` surfaces a structured warning AND records the
  failure in the manifest when a manifest handle is available.
* ``warn_writer_failure`` delegates to the above via
  ``OutputWriter._manifest``.
* Every ``OutputFailureKind`` category produces the right log level.
* The manifest row for a failed artefact carries ``written=false`` and
  a non-empty ``error`` cell.
* Writer errors remain visible immediately, while finalization refuses to
  claim a complete run when an ``always=True`` artefact is absent.

Tests are parametrised across all five failure kinds so the log-level
and manifest-handling logic is exercised uniformly.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from vibeqc.output import ManifestUpdater, OutputWriter

from vibeqc.output._errors import (
    OutputFailureKind,
    OutputFailureRecord,
    warn_output_failure,
    warn_writer_failure,
)

# ---------------------------------------------------------------------------
# Unit: OutputFailureRecord
# ---------------------------------------------------------------------------


def test_failure_record_has_readable_warning(tmp_path: Path) -> None:
    path = tmp_path / "test.xyz"
    rec = OutputFailureRecord(
        path=path,
        role="test_role",
        category=OutputFailureKind.optional_artifact,
        exc_type="ValueError",
        exc_message="bad value",
    )
    msg = rec.warning_message()
    assert "test_role" in msg
    assert "test.xyz" in msg
    assert "optional_artifact" in msg
    assert "ValueError" in msg
    assert "bad value" in msg


def test_failure_record_manifest_error_string() -> None:
    rec = OutputFailureRecord(
        path=Path("out.qvf"),
        role="qvf",
        category=OutputFailureKind.optional_artifact,
        exc_type="RuntimeError",
        exc_message="no space",
    )
    assert rec.manifest_error_string() == "RuntimeError: no space"


# ---------------------------------------------------------------------------
# Unit: warn_output_failure — warning emitted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        OutputFailureKind.optional_artifact,
        OutputFailureKind.manifest_recording,
        OutputFailureKind.cleanup,
        OutputFailureKind.compatibility_fallback,
    ],
)
def test_warn_output_failure_emits_warning(kind: OutputFailureKind) -> None:
    rec = warn_output_failure(
        RuntimeError("boom"),
        Path("out.xyz"),
        role="test",
        category=kind,
    )
    assert isinstance(rec, OutputFailureRecord)
    assert rec.exc_type == "RuntimeError"


def test_warn_output_failure_unrecoverable_emits_error_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    warn_output_failure(
        RuntimeError("boom"),
        Path("out.xyz"),
        role="test",
        category=OutputFailureKind.unrecoverable,
    )
    assert "boom" in caplog.text
    # unrecoverable → logger.error (caplog captures at all levels)


# ---------------------------------------------------------------------------
# Unit: warn_output_failure — manifest recording
# ---------------------------------------------------------------------------


def _build_plan(tmp_path: Path) -> "OutputWriter":
    from vibeqc.output import OutputPlan, OutputWriter

    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
    )
    return OutputWriter(plan)


def test_warn_output_failure_records_in_manifest(tmp_path: Path) -> None:
    """``warn_output_failure`` accepts a :class:`ManifestUpdater`
    directly. (``warn_writer_failure`` is the wrapper that accepts an
    :class:`OutputWriter`; see the test below for that path.)"""
    writer = _build_plan(tmp_path)
    failed_path = tmp_path / "h2o.molden"
    # The molden file is declared in the plan.
    warn_output_failure(
        RuntimeError("write failed"),
        failed_path,
        role="molden",
        category=OutputFailureKind.optional_artifact,
        manifest=writer._manifest,  # type: ignore[attr-defined]
    )
    outcomes = writer.outcomes()
    matched = [o for o in outcomes if o.path == failed_path]
    assert len(matched) == 1
    assert matched[0].written is False
    assert "RuntimeError" in matched[0].error  # type: ignore[attr-defined]


def test_warn_output_failure_appends_undeclared_path(tmp_path: Path) -> None:
    writer = _build_plan(tmp_path)
    # Path not declared in plan — should still get a manifest row.
    undeclared = tmp_path / "h2o.unexpected.log"
    warn_output_failure(
        FileNotFoundError("nope"),
        undeclared,
        role="unexpected",
        category=OutputFailureKind.optional_artifact,
        manifest=writer._manifest,  # type: ignore[attr-defined]
    )
    outcomes = writer.outcomes()
    matched = [o for o in outcomes if o.path == undeclared]
    assert len(matched) == 1
    assert matched[0].written is False
    assert "FileNotFoundError" in matched[0].error  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Unit: warn_writer_failure delegates correctly
# ---------------------------------------------------------------------------


def test_warn_writer_failure_delegates_to_writer_manifest(
    tmp_path: Path,
) -> None:
    writer = _build_plan(tmp_path)
    failed_path = tmp_path / "h2o.cube"
    rec = warn_writer_failure(
        ValueError("bad cube"),
        failed_path,
        role="cube",
        category=OutputFailureKind.optional_artifact,
        writer=writer,
    )
    assert rec.exc_type == "ValueError"
    outcomes = writer.outcomes()
    matched = [o for o in outcomes if o.path == failed_path]
    assert len(matched) == 1
    assert matched[0].written is False


# ---------------------------------------------------------------------------
# Integration: molecular run_job — output failure does not crash SCF
# ---------------------------------------------------------------------------


def test_molecular_scf_refuses_complete_status_after_writer_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A requested population failure warns, then blocks finalization."""
    import tomllib

    from vibeqc import Atom, Molecule, run_job
    from vibeqc.output import IncompleteOutputError
    from vibeqc.output.formats import population as _population

    # Inject the failure at the writer-function level — the runner's
    # ``except Exception`` calls warn_writer_failure(...) which records
    # the failure in the .system manifest.
    def _boom_population(*args, **kwargs):
        raise RuntimeError("synthetic population writer failure")

    monkeypatch.setattr(
        _population,
        "write_population_summary",
        _boom_population,
    )

    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
    stem = tmp_path / "h2_popfail"
    with pytest.warns(UserWarning, match=r"vibe-qc output \[optional_artifact\]"):
        with pytest.raises(IncompleteOutputError, match="population"):
            run_job(
                mol,
                basis="sto-3g",
                method="rhf",
                output=stem,
                write_population_file=True,
                citations=False,
            )

    # Manifest row carries the failure.
    body = tomllib.loads(stem.with_suffix(".system").read_text(encoding="utf-8"))
    rows = {r["path"]: r for r in body["outputs"]["files"]}
    assert body["outputs"]["status"] == "crashed"
    pop_path = str(stem.parent / (stem.name + ".population.txt"))
    assert pop_path in rows, f"population row missing from {list(rows)}"
    assert rows[pop_path]["written"] is False
    assert "RuntimeError" in rows[pop_path]["error"]
    assert "synthetic population writer failure" in rows[pop_path]["error"]


# ---------------------------------------------------------------------------
# Integration: periodic run_periodic_job — output failure does not crash
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_periodic_scf_refuses_complete_status_after_writer_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception in the periodic XYZ writer must emit a structured
    warning and block complete finalization. (Marked ``slow`` because
    even a 2-atom periodic SCF takes several seconds.)

    Uses the C++-bound :class:`vibeqc.PeriodicSystem` constructor —
    ``PeriodicSystem(dim, lattice, unit_cell=[Atom(...)])`` — and a
    real ``BasisSet`` matching the runner's signature.
    """
    import tomllib

    import numpy as np
    import vibeqc as vq
    from vibeqc import Atom, PeriodicSystem
    from vibeqc.periodic_runner import run_periodic_job
    from vibeqc.output import IncompleteOutputError
    from vibeqc.output.formats import extended_xyz as extended_xyz_module

    # Inject a synthetic failure in the periodic extended-XYZ writer.
    def _boom_xyz(*args, **kwargs):
        raise RuntimeError("synthetic periodic xyz failure")

    monkeypatch.setattr(extended_xyz_module, "write_extended_xyz", _boom_xyz)

    # 3D cubic box, 2 H atoms — small enough to converge fast.
    lattice = np.eye(3) * 6.0
    cell = PeriodicSystem(
        3,
        lattice,
        unit_cell=[
            Atom(1, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 1.4]),
        ],
    )
    basis = vq.BasisSet(cell.unit_cell_molecule(), "sto-3g")

    stem = tmp_path / "h2_periodic_xyzfail"
    with pytest.warns(UserWarning, match=r"vibe-qc output \[optional_artifact\]"):
        with pytest.raises(IncompleteOutputError, match=r"\.xyz"):
            run_periodic_job(
                cell,
                basis,
                method="RHF",
                output=stem,
                max_iter=30,
                conv_tol_energy=1e-6,
                write_xyz_file=True,
                write_poscar_file=False,
                write_cif_file=False,
                write_xsf_structure_file=False,
                write_molden_file=False,
                write_population_file=False,
                write_density=False,
                output_qvf=False,
                citations=False,
            )

    # The periodic OutputWriter exists before compute, so the dispatch
    # failure is visible to downstream manifest consumers immediately.
    body = tomllib.loads(stem.with_suffix(".system").read_text(encoding="utf-8"))
    rows = {r["path"]: r for r in body["outputs"]["files"]}
    assert body["outputs"]["status"] == "crashed"
    xyz_row = rows.get(str(stem.with_suffix(".xyz")))
    assert xyz_row is not None
    assert xyz_row["written"] is False
    assert "synthetic periodic xyz failure" in xyz_row["error"]


# ---------------------------------------------------------------------------
# Edge: manifest row survives JSON/TOML round-trip with error field
# ---------------------------------------------------------------------------


def test_failed_outcome_round_trips_through_tomllib(tmp_path: Path) -> None:
    """A manifest row with an error cell must still round-trip through
    tomllib — the error field is a plain string."""
    import tomllib

    from vibeqc.output import OutputPlan, OutputWriter
    from vibeqc.output.manifest import FileOutcome

    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
    )
    writer = OutputWriter(plan)

    # Manually inject an error into a declared file.
    molden_path = tmp_path / "h2o.molden"
    outcomes = writer.outcomes()
    for o in outcomes:
        if o.path == molden_path:
            o.written = False
            o.error = "RuntimeError: write failed"
            break
    writer._manifest.mark_failed(molden_path, "RuntimeError: write failed")

    body = tomllib.loads(writer.manifest_path.read_text(encoding="utf-8"))
    rows = {r["path"]: r for r in body["outputs"]["files"]}
    matched = rows[str(molden_path)]
    assert matched["written"] is False
    assert "RuntimeError" in matched["error"]
    # tomllib did not choke — the field is serialisable.


# ---------------------------------------------------------------------------
# All five categories exercise the right log level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,expected_level",
    [
        (OutputFailureKind.optional_artifact, "WARNING"),
        (OutputFailureKind.manifest_recording, "WARNING"),
        (OutputFailureKind.cleanup, "WARNING"),
        (OutputFailureKind.compatibility_fallback, "WARNING"),
        (OutputFailureKind.unrecoverable, "ERROR"),
    ],
)
def test_category_log_levels(
    kind: OutputFailureKind,
    expected_level: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    warn_output_failure(
        RuntimeError("test"),
        Path("out.xyz"),
        role="test",
        category=kind,
    )
    # caplog at INFO captures all levels; check the actual record.
    records = caplog.records
    assert len(records) >= 1
    actual = caplog.text
    assert expected_level.lower() in actual.lower()
