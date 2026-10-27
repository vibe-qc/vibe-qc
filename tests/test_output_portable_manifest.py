"""Portable ``.system`` manifest mode (vibe-qc#388).

The default manifest keeps the caller's path spelling as runtime
provenance: a relative stem stays relative, an absolute one stays
absolute (#127 anchoring unchanged). The opt-in portable mode, selected
by ``portable_manifest=True`` or ``VIBEQC_PORTABLE_MANIFEST=1``, writes
every plan/output path and the plan stem relative to the manifest's own
directory, marks artefacts outside it as ``<external>/<name>``, writes
the interpreter relative to its environment and records the mode and its
resolution base in ``[run]``.

Acceptance from the maintainer decision: relative and absolute stems, a
cwd change between plan and write, relocating the output directory, and
artefact digests that match before and after the move.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tomllib
from pathlib import Path

import pytest

from vibeqc.output import (
    ManifestUpdater,
    OutputPlan,
    OutputWriter,
    portable_manifest_requested,
    portable_manifest_scope,
)


@pytest.fixture(autouse=True)
def _no_ambient_portable_env(monkeypatch):
    monkeypatch.delenv("VIBEQC_PORTABLE_MANIFEST", raising=False)


@pytest.fixture
def h2():
    from vibeqc import Atom, Molecule

    return Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])


def _plan(stem: Path | str, **extra) -> OutputPlan:
    return OutputPlan.from_run_job_kwargs(
        output=stem, method="rhf", basis="sto-3g", functional=None, **extra,
    )


def _read(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _write_guaranteed(plan: OutputPlan, writer: OutputWriter, base: Path) -> None:
    """Materialize every guaranteed non-manifest artefact and record it."""
    for planned in plan.guaranteed_files():
        if planned.role == "manifest":
            continue
        target = planned.path if planned.path.is_absolute() else base / planned.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"payload {planned.role}\n", encoding="utf-8")
        writer.record(planned.path)


def _all_paths(manifest: dict) -> list[str]:
    return [row["path"] for row in manifest["plan"]["files"]] + [
        row["path"] for row in manifest["outputs"]["files"]
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_mode_keeps_the_absolute_spelling(tmp_path):
    plan = _plan(tmp_path / "job" / "h2")
    updater = ManifestUpdater(plan)
    body = _read(updater.path)
    assert body["plan"]["stem"] == str(tmp_path / "job" / "h2")
    assert all(Path(p).is_absolute() for p in _all_paths(body))
    assert "path_mode" not in body["run"]
    assert not updater.portable


@pytest.mark.parametrize("stem_kind", ["relative", "absolute"])
def test_portable_paths_are_relative_to_the_manifest_directory(
    tmp_path, monkeypatch, stem_kind,
):
    monkeypatch.chdir(tmp_path)
    stem = Path("job") / "h2" if stem_kind == "relative" else tmp_path / "job" / "h2"
    plan = _plan(stem)
    writer = OutputWriter(plan, portable_manifest=True)
    _write_guaranteed(plan, writer, tmp_path)
    writer.finish()
    body = _read(tmp_path / "job" / "h2.system")
    assert body["run"]["path_mode"] == "portable"
    assert body["run"]["path_base"] == "manifest_directory"
    assert body["plan"]["stem"] == "h2"
    paths = _all_paths(body)
    assert "h2.out" in paths and "h2.system" in paths
    assert all(not Path(p).is_absolute() and "/" not in p for p in paths)
    assert str(tmp_path) not in (tmp_path / "job" / "h2.system").read_text()


def test_cwd_change_between_plan_and_write(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    elsewhere = tmp_path / "elsewhere"
    job_dir.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(job_dir)
    plan = _plan("h2")
    writer = OutputWriter(plan, portable_manifest=True)
    monkeypatch.chdir(elsewhere)
    _write_guaranteed(plan, writer, job_dir)
    writer.finish()
    assert not (elsewhere / "h2.system").exists()
    body = _read(job_dir / "h2.system")
    assert body["plan"]["stem"] == "h2"
    assert {p for p in _all_paths(body)} >= {"h2.out", "h2.system"}
    assert not any(p.startswith("<external>/") for p in _all_paths(body))


def test_relocated_directory_resolves_with_matching_digests(tmp_path):
    original = tmp_path / "original"
    plan = _plan(original / "h2")
    writer = OutputWriter(plan, portable_manifest=True)
    _write_guaranteed(plan, writer, tmp_path)
    writer.finish()
    before = {
        row["path"]: row["sha256"]
        for row in _read(original / "h2.system")["outputs"]["files"]
        if row["checksum_status"] == "sha256"
    }
    assert before
    moved = tmp_path / "published" / "dataset"
    moved.parent.mkdir()
    shutil.move(str(original), str(moved))
    manifest = moved / "h2.system"
    after = _read(manifest)["outputs"]["files"]
    for row in after:
        if row["checksum_status"] != "sha256":
            continue
        # The manifest's own directory is the documented resolution base.
        target = manifest.parent / row["path"]
        assert target.is_file()
        assert _sha256(target) == row["sha256"] == before[row["path"]]


def test_external_artefact_keeps_only_its_name(tmp_path):
    perf = tmp_path / "logs" / "h2.perf"
    plan = _plan(tmp_path / "job" / "h2", perf_log=perf)
    updater = ManifestUpdater(plan, portable_manifest=True)
    body = _read(updater.path)
    assert "<external>/h2.perf" in _all_paths(body)
    assert str(tmp_path) not in updater.path.read_text()


def test_interpreter_is_written_relative_to_its_environment(tmp_path):
    updater = ManifestUpdater(_plan(tmp_path / "h2"), portable_manifest=True)
    executable = _read(updater.path)["python"]["executable"]
    assert executable.startswith(("<env>/", "<external>/"))
    if Path(sys.executable).resolve().is_relative_to(Path(sys.prefix).resolve()):
        assert executable.startswith("<env>/")
        assert Path(sys.prefix, executable.removeprefix("<env>/")).exists()


def test_interpreter_stays_portable_with_hostname_redaction(tmp_path):
    updater = ManifestUpdater(
        _plan(tmp_path / "h2"), record_hostname=False, portable_manifest=True,
    )
    body = _read(updater.path)
    assert body["host"]["hostname"] == "<redacted>"
    assert not body["python"]["executable"].startswith("~")


def test_portable_mode_does_not_redact_the_hostname(tmp_path):
    body = _read(ManifestUpdater(_plan(tmp_path / "h2"), portable_manifest=True).path)
    assert body["host"]["hostname"] != "<redacted>"


def test_basis_library_root_is_package_relative(tmp_path):
    import vibeqc

    updater = ManifestUpdater(_plan(tmp_path / "h2"), portable_manifest=True)
    updater.set_basis_library(
        resolved_root=str(Path(vibeqc.__file__).parent / "basis_library")
    )
    assert _read(updater.path)["basis_library"]["resolved_root"] == (
        "<vibeqc>/basis_library"
    )
    updater.set_basis_library(resolved_root=str(tmp_path / "overlay"))
    assert _read(updater.path)["basis_library"]["resolved_root"] == (
        "<external>/overlay"
    )


@pytest.mark.parametrize("value, expected", [
    ("1", True), ("yes", True), ("0", False), ("false", False), ("", False),
])
def test_environment_lever(tmp_path, monkeypatch, value, expected):
    monkeypatch.setenv("VIBEQC_PORTABLE_MANIFEST", value)
    assert portable_manifest_requested() is expected
    assert ManifestUpdater(_plan(tmp_path / "h2")).portable is expected


def test_scope_only_turns_the_mode_on(tmp_path):
    assert not portable_manifest_requested()
    with portable_manifest_scope(True):
        with portable_manifest_scope(False):
            assert portable_manifest_requested()
            assert ManifestUpdater(_plan(tmp_path / "h2")).portable
    assert not portable_manifest_requested()


def test_run_job_keyword_writes_a_portable_manifest(tmp_path, h2):
    from vibeqc import run_job

    stem = tmp_path / "run" / "h2"
    run_job(h2, basis="sto-3g", method="rhf", output=stem, portable_manifest=True)
    manifest = stem.parent / "h2.system"
    text = manifest.read_text()
    body = tomllib.loads(text)
    assert body["run"]["path_mode"] == "portable"
    assert body["plan"]["stem"] == "h2"
    assert str(tmp_path) not in text
    for row in body["outputs"]["files"]:
        if row["checksum_status"] == "sha256":
            assert _sha256(manifest.parent / row["path"]) == row["sha256"]
    assert not portable_manifest_requested()


def test_run_job_default_is_unchanged(tmp_path, h2):
    from vibeqc import run_job

    stem = tmp_path / "run" / "h2"
    run_job(h2, basis="sto-3g", method="rhf", output=stem)
    body = _read(stem.parent / "h2.system")
    assert "path_mode" not in body["run"]
    assert body["plan"]["stem"] == os.fspath(stem)


def test_run_periodic_job_keyword_writes_a_portable_manifest(tmp_path):
    import numpy as np
    import vibeqc as vq

    system = vq.PeriodicSystem(3, np.eye(3) * 7.0, [vq.Atom(2, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "cell" / "he"
    assert vq.run_periodic_job(
        system, basis, method="RHF", output=stem, dry_run=True,
        portable_manifest=True,
    ) is None
    text = (stem.parent / "he.system").read_text()
    body = tomllib.loads(text)
    assert body["run"]["path_mode"] == "portable"
    assert body["plan"]["stem"] == "he"
    assert str(tmp_path) not in text
    assert not portable_manifest_requested()
