from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from examples.regression.core import runner_vibeqc
from examples.regression.core.spec import MethodSpec
from examples.regression.systems.periodic.ne_fcc import SPEC as NE_FCC


def test_periodic_regression_runner_uses_public_gdf_gamma_route(tmp_path, monkeypatch):
    captured = {}

    def fake_run_periodic_job(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            energy=-1.0,
            converged=True,
            n_iter=2,
            backend="pbc-gdf-rsgdf-uks",
        )

    monkeypatch.setattr(runner_vibeqc.vq, "run_periodic_job", fake_run_periodic_job)

    row = runner_vibeqc.run_periodic_case(
        run_id="unit",
        target="dev",
        code_version="test",
        spec=NE_FCC,
        basis_name="sto-3g",
        method=MethodSpec(id="rks-lda", scf="rks", xc="lda"),
        kmesh=(1, 1, 1),
        log_path=tmp_path / "ne.log",
    )

    assert row.energy_ha == -1.0
    assert row.converged
    assert captured["method"] == "RKS"
    assert captured["functional"] == "lda"
    assert captured["jk_method"] == "gdf"
    assert captured["kpoints"] is None
    assert captured["output_qvf"] is False
    assert row.status == "pending"
    assert row.note == ""


def test_periodic_regression_runner_tail_cutoff_lifts_static_hold(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_run_periodic_job(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            energy=-504.6098275,
            converged=True,
            n_iter=2,
            backend="pbc-gdf-rsgdf-uks",
        )

    monkeypatch.setattr(runner_vibeqc.vq, "run_periodic_job", fake_run_periodic_job)

    row = runner_vibeqc.run_periodic_case(
        run_id="unit",
        target="dev",
        code_version="test",
        spec=NE_FCC,
        basis_name="sto-3g",
        method=MethodSpec(id="rks-lda", scf="rks", xc="lda"),
        kmesh=(1, 1, 1),
        rsgdf_tail_ke_cutoff=10000.0,
        log_path=tmp_path / "ne-tail.log",
    )

    assert captured["rsgdf_tail_ke_cutoff"] == 10000.0
    assert captured["output_qvf"] is False
    assert row.energy_ha == -504.6098275
    assert row.status == "pending"
    assert row.note == ""


def test_periodic_regression_runner_marks_gdf_parity_hold_unavailable(
    tmp_path,
    monkeypatch,
):
    def fake_run_periodic_job(*args, **kwargs):
        return SimpleNamespace(
            energy=-506.1141162,
            converged=True,
            n_iter=2,
            backend="pbc-gdf-rsgdf-uks+PARITY_HELD",
        )

    monkeypatch.setattr(runner_vibeqc.vq, "run_periodic_job", fake_run_periodic_job)

    row = runner_vibeqc.run_periodic_case(
        run_id="unit",
        target="dev",
        code_version="test",
        spec=NE_FCC,
        basis_name="sto-3g",
        method=MethodSpec(id="rks-pbe", scf="rks", xc="pbe"),
        kmesh=(1, 1, 1),
        log_path=tmp_path / "ne-held.log",
    )

    assert row.energy_ha == -506.1141162
    assert row.status == "unavailable"
    assert "parity held" in row.note


def test_periodic_regression_runner_explicit_zero_tail_keeps_static_hold(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_run_periodic_job(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            energy=-506.1141162,
            converged=True,
            n_iter=2,
            backend="pbc-gdf-rsgdf-uks",
        )

    monkeypatch.setattr(runner_vibeqc.vq, "run_periodic_job", fake_run_periodic_job)

    row = runner_vibeqc.run_periodic_case(
        run_id="unit",
        target="dev",
        code_version="test",
        spec=NE_FCC,
        basis_name="sto-3g",
        method=MethodSpec(id="rks-lda", scf="rks", xc="lda"),
        kmesh=(1, 1, 1),
        rsgdf_tail_ke_cutoff=0.0,
        log_path=tmp_path / "ne-zero-tail.log",
    )

    assert captured["rsgdf_tail_ke_cutoff"] == 0.0
    assert row.status == "unavailable"
    assert "Gamma GDF solid absolute-energy parity held" in row.note


def test_periodic_regression_runner_copies_requested_qvf(tmp_path, monkeypatch):
    captured = {}

    def fake_run_periodic_job(*args, **kwargs):
        captured.update(kwargs)
        output_stem = Path(kwargs["output"])
        if kwargs.get("output_qvf"):
            output_stem.with_suffix(".qvf").write_bytes(b"mock qvf archive")
        return SimpleNamespace(
            energy=-1.0,
            converged=True,
            n_iter=2,
            backend="mock-gdf",
        )

    monkeypatch.setattr(runner_vibeqc.vq, "run_periodic_job", fake_run_periodic_job)

    artifact_dir = tmp_path / "case" / "vibeqc"
    row = runner_vibeqc.run_periodic_case(
        run_id="unit",
        target="dev",
        code_version="test",
        spec=NE_FCC,
        basis_name="sto-3g",
        method=MethodSpec(id="rks-pbe", scf="rks", xc="pbe"),
        kmesh=(1, 1, 1),
        log_path=tmp_path / "ne-qvf.log",
        artifact_dir=artifact_dir,
        write_qvf_artifact=True,
    )

    assert row.energy_ha == -1.0
    assert row.converged
    assert captured["output_qvf"] is True
    assert Path(captured["output"]) == artifact_dir / "vibeqc_job"
    assert (artifact_dir / "vibeqc_job.qvf").read_bytes() == b"mock qvf archive"
    assert (artifact_dir / "vibeqc.qvf").read_bytes() == b"mock qvf archive"
