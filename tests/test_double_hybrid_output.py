from __future__ import annotations

import json
import tomllib
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vibeqc import (
    Atom,
    DoubleHybridResult,
    Molecule,
    run_b2plyp,
    run_revdsd_pbep86,
)
from vibeqc.output.formats.qvf import validate_qvf


def _h2o() -> Molecule:
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.4315, -0.9314]),
            Atom(1, [0.0, -1.4315, -0.9314]),
        ]
    )


def _qvf_manifest(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path, "r") as zf:
        return json.loads(zf.read("manifest.json"))


def _assert_content_sized_energy_block(out_text: str) -> None:
    lines = out_text.splitlines()
    title_index = lines.index("  Energy components (Ha)")
    top_rule = lines[title_index + 1]
    bottom_index = lines.index(top_rule, title_index + 2)
    assert bottom_index > title_index + 2
    assert all(
        len(line) == len(top_rule) for line in lines[title_index + 2 : bottom_index]
    )


def _assert_complete_telemetry(stem: Path, result: DoubleHybridResult) -> None:
    structured_path = stem.with_suffix(".scf.jsonl")
    perf_path = stem.with_suffix(".perf")
    assert structured_path.is_file()
    assert perf_path.is_file()
    assert structured_path.stat().st_size > 0
    assert perf_path.stat().st_size > 0

    records = [
        json.loads(line)
        for line in structured_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    events = [record["event"] for record in records]
    assert events[0] == "banner"
    assert events[-1] == "job_end"
    assert {
        "job_start",
        "scf_iter",
        "scf_converged",
        "double_hybrid_mp2_done",
        "double_hybrid_post_scf",
    } <= set(events)

    mp2 = next(
        record for record in records if record["event"] == "double_hybrid_mp2_done"
    )
    assert mp2["e_correlation"] == pytest.approx(
        mp2["c_os"] * mp2["e_os"] + mp2["c_ss"] * mp2["e_ss"]
    )
    assert mp2["e_correlation"] == pytest.approx(result.mp2.e_correlation)
    assert mp2["n_frozen_core"] == 0
    assert mp2["frozen_core_convention"] == "all-electron-explicit"
    assert mp2["solver_iterations"] is None
    assert mp2["iteration_contract"] == "direct_noniterative"
    assert mp2["wall_s"] > 0.0

    post_scf = next(
        record
        for record in records
        if record["event"] == "double_hybrid_post_scf"
    )
    assert post_scf["scaled_os_correlation"] == pytest.approx(
        post_scf["os_scale"] * post_scf["raw_os_correlation"]
    )
    assert post_scf["scaled_ss_correlation"] == pytest.approx(
        post_scf["ss_scale"] * post_scf["raw_ss_correlation"]
    )
    assert post_scf["mp2_correlation"] == pytest.approx(
        post_scf["scaled_os_correlation"]
        + post_scf["scaled_ss_correlation"]
    )
    assert post_scf["assembled_total_energy"] == pytest.approx(
        post_scf["reference_scf_energy"]
        + post_scf["mp2_correlation"]
        + post_scf["dispersion_energy"]
    )
    assert post_scf["n_frozen_core"] == 0
    assert post_scf["frozen_core_convention"] == "all-electron-explicit"
    assert post_scf["reference_scf_iterations"] > 0
    assert post_scf["solver_iterations"] is None
    assert post_scf["iteration_contract"] == "direct_noniterative"
    terminal = records[-1]
    assert terminal["energy_kind"] == "assembled_total"
    assert terminal["energy"] == pytest.approx(result.e_total)

    snapshots = [record for record in records if record["event"] == "memory_snapshot"]
    labels = {record["label"] for record in snapshots}
    assert {
        "job_start",
        "start_of_scf",
        "end_of_scf",
        "end_of_mp2",
        "job_end",
    } <= labels
    assert all(record["rss_mb"] > 0.0 for record in snapshots)

    perf_text = perf_path.read_text(encoding="utf-8")
    assert "double_hybrid.total" in perf_text
    assert "double_hybrid.reference_scf" in perf_text
    assert "double_hybrid.mp2" in perf_text
    assert "double_hybrid.output" in perf_text
    assert "Memory snapshots" in perf_text
    assert "start_of_scf" in perf_text
    assert "end_of_mp2" in perf_text
    assert "SCF iterations" in perf_text


def test_run_b2plyp_output_writes_standard_sidecars_and_qvf(tmp_path: Path) -> None:
    stem = tmp_path / "b2plyp_h2o"
    result = run_b2plyp(
        _h2o(),
        "sto-3g",
        density_fit=False,
        density_fit_mp2=False,
        output=stem,
        citations=False,
        perf_log=True,
        structured_log=True,
    )

    assert isinstance(result, DoubleHybridResult)
    assert np.isfinite(result.e_total)
    for path in (
        stem.with_suffix(".out"),
        stem.with_suffix(".system"),
        stem.with_suffix(".molden"),
        stem.with_suffix(".xyz"),
        tmp_path / "b2plyp_h2o.population.txt",
        tmp_path / "b2plyp_h2o.population.json",
        stem.with_suffix(".qvf"),
        stem.with_suffix(".scf.jsonl"),
        stem.with_suffix(".perf"),
    ):
        assert path.is_file(), path

    out_text = stem.with_suffix(".out").read_text(encoding="utf-8")
    assert "Double-hybrid DFT dispatcher" in out_text
    assert "MP2 correlation" in out_text
    assert "MP2 frozen core:   0 (explicit all-electron model component)" in out_text
    assert f"{result.e_total:18.10f}" in out_text
    assert out_text.count("RKS reference") == 1
    _assert_content_sized_energy_block(out_text)

    system = tomllib.loads(stem.with_suffix(".system").read_text(encoding="utf-8"))
    assert system["outputs"]["status"] == "complete"
    assert system["run"]["frozen_core_orbitals"] == 0
    assert system["run"]["frozen_core_convention"] == "all-electron-explicit"
    planned = {(row["role"], row["format"]) for row in system["plan"]["files"]}
    assert ("qvf", "qvf") in planned
    written = {
        Path(row["path"]).name for row in system["outputs"]["files"] if row["written"]
    }
    assert "b2plyp_h2o.qvf" in written
    assert "b2plyp_h2o.population.json" in written
    assert "b2plyp_h2o.scf.jsonl" in written
    assert "b2plyp_h2o.perf" in written
    guaranteed_paths = {row["path"] for row in system["plan"]["files"] if row["always"]}
    output_rows = {row["path"]: row for row in system["outputs"]["files"]}
    assert all(
        output_rows[path]["written"]
        and output_rows[path]["checksum_status"] != "pending"
        for path in guaranteed_paths
    )
    _assert_complete_telemetry(stem, result)

    report = validate_qvf(stem.with_suffix(".qvf"))
    assert report["valid"], report["errors"]
    manifest = _qvf_manifest(stem.with_suffix(".qvf"))
    assert manifest["provenance"]["method"] == "b2plyp"
    assert manifest["provenance"]["scf_energy"]["value"] == result.rks.energy
    assert isinstance(manifest["dipole_moment"]["origin"], list)
    assert len(manifest["dipole_moment"]["origin"]) == 3
    kinds = {section["kind"] for section in manifest["sections"]}
    assert {
        "structure",
        "wavefunction.gto",
        "atom_properties",
        "run.record",
    } <= kinds
    run_record = next(
        section
        for section in manifest["sections"]
        if section["kind"] == "run.record"
    )
    with zipfile.ZipFile(stem.with_suffix(".qvf")) as zf:
        assert zf.read(run_record["members"]["log"]["path"]) == (
            stem.with_suffix(".out").read_bytes()
        )
    atom_properties = next(
        section
        for section in manifest["sections"]
        if section["kind"] == "atom_properties"
    )
    assert "hirshfeld_charge" in atom_properties["members"]


def test_run_revdsd_pbep86_output_qvf_default_and_opt_out(tmp_path: Path) -> None:
    stem = tmp_path / "revdsd_h2o"
    result = run_revdsd_pbep86(
        _h2o(),
        "sto-3g",
        density_fit=False,
        density_fit_mp2=False,
        dispersion="d4",
        output=stem,
        citations=False,
        write_population_file=False,
        perf_log=True,
        structured_log=True,
    )

    assert isinstance(result, DoubleHybridResult)
    assert stem.with_suffix(".qvf").is_file()
    _assert_content_sized_energy_block(
        stem.with_suffix(".out").read_text(encoding="utf-8")
    )
    report = validate_qvf(stem.with_suffix(".qvf"))
    assert report["valid"], report["errors"]
    assert _qvf_manifest(stem.with_suffix(".qvf"))["provenance"]["method"] == (
        "revdsd-pbep86"
    )
    records = [
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    dispersion = next(
        record
        for record in records
        if record["event"] == "double_hybrid_dispersion_done"
    )
    assert dispersion["model"] == "d4"
    assert dispersion["energy"] == pytest.approx(result.dispersion.energy)
    assert dispersion["solver_iterations"] is None
    assert dispersion["iteration_contract"] == "not_defined"
    assert dispersion["wall_s"] > 0.0
    assert "double_hybrid.dispersion" in stem.with_suffix(".perf").read_text(
        encoding="utf-8"
    )

    opt_out = tmp_path / "revdsd_no_qvf"
    run_revdsd_pbep86(
        _h2o(),
        "sto-3g",
        density_fit=False,
        density_fit_mp2=False,
        dispersion="d4",
        output=opt_out,
        output_qvf=False,
        citations=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
    )
    assert opt_out.with_suffix(".out").is_file()
    assert opt_out.with_suffix(".system").is_file()
    assert not opt_out.with_suffix(".qvf").exists()
    assert not opt_out.with_suffix(".scf.jsonl").exists()
    assert not opt_out.with_suffix(".perf").exists()
