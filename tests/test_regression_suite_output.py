from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.regression import run_suite
from examples.regression.core.compare import annotate_against_reference
from examples.regression.core.report import render_summary
from examples.regression.core import runner_orca, runner_pyscf, runner_vibeqc
from examples.regression.core.case import CaseRecord, CodeRow
from examples.regression.core.spec import (
    AtomCart,
    ExpectedRef,
    MethodSpec,
    MoleculeSpec,
)
from examples.regression.methods.catalog import METHODS
from examples.regression.systems.periodic.ne_fcc import SPEC as NE_FCC


def test_output_root_default_is_home_vibeqc_runs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv(run_suite.RUNS_DIR_ENV, raising=False)

    root = run_suite.resolve_output_root(environ={})

    assert root == (home / "vibeqc-runs").resolve()


def test_output_root_cli_override_wins(tmp_path):
    cli = tmp_path / "cli-root"
    env = tmp_path / "env-root"

    root = run_suite.resolve_output_root(
        str(cli),
        environ={run_suite.RUNS_DIR_ENV: str(env)},
    )

    assert root == cli.resolve()


def test_output_root_env_override_wins_without_cli(tmp_path):
    env = tmp_path / "env-root"

    root = run_suite.resolve_output_root(
        environ={run_suite.RUNS_DIR_ENV: str(env)},
    )

    assert root == env.resolve()


def test_smoke_run_writes_self_contained_output_tree(tmp_path):
    output_root = tmp_path / "runs"
    run_id = "pytest-smoke"

    rc = run_suite.main([
        "--smoke",
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--output-qvf",
    ])

    assert rc == 0
    run_dir = output_root / run_id
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "env.json").is_file()
    assert (run_dir / "results.csv").is_file()
    assert (run_dir / "summary.md").is_file()

    case_dir = run_dir / "cases" / "h2__sto-3g__rhf__mol"
    assert case_dir.is_dir()
    assert (case_dir / "NOTES.md").is_file()

    vibeqc_dir = case_dir / "vibeqc"
    assert (vibeqc_dir / "input.json").is_file()
    assert (vibeqc_dir / "vibeqc.out").is_file()
    assert (vibeqc_dir / "vibeqc.system").is_file()
    assert (vibeqc_dir / "vibeqc.qvf").is_file()
    assert (vibeqc_dir / "stdout.log").is_file()
    assert (vibeqc_dir / "stderr.log").is_file()
    assert (vibeqc_dir / "parsed.json").is_file()

    pyscf_dir = case_dir / "reference" / "pyscf"
    assert (pyscf_dir / "input.json").is_file()
    assert (pyscf_dir / "stdout.log").is_file()
    assert (pyscf_dir / "stderr.log").is_file()
    assert (pyscf_dir / "parsed.json").is_file()


def test_smoke_run_does_not_write_checkout_output(tmp_path):
    checkout_output = run_suite.REGRESSION_ROOT / "output"
    before = set(checkout_output.iterdir()) if checkout_output.exists() else set()

    rc = run_suite.main([
        "--smoke",
        "--output-root",
        str(tmp_path / "runs"),
        "--run-id",
        "pytest-no-checkout-output",
        "--output-qvf",
    ])

    assert rc == 0
    after = set(checkout_output.iterdir()) if checkout_output.exists() else set()
    assert after == before


def test_isolated_periodic_vibeqc_runner_receives_rsgdf_tail_cutoff(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_run(*_args, **kwargs):
        captured["env"] = kwargs.get("env")
        row = CodeRow(
            run_id="pytest",
            target="dev",
            system_id=NE_FCC.id,
            family=NE_FCC.family,
            basis="sto-3g",
            method_id="rks-lda",
            kmesh="1x1x1",
            code="vibeqc",
            code_version="test",
            status="pending",
        )
        return SimpleNamespace(
            returncode=0,
            stdout=run_suite._RESULT_MARKER + json.dumps(row.__dict__) + "\n",
            stderr="",
        )

    monkeypatch.setattr(run_suite.subprocess, "run", fake_run)

    row = run_suite._isolated_runner_call(
        code="vibeqc",
        spec_kind="periodic",
        spec=NE_FCC,
        basis="sto-3g",
        method=METHODS["rks-lda"],
        log_path=tmp_path / "ne.log",
        workdir=None,
        artifact_dir=None,
        run_id="pytest",
        target="dev",
        code_version="test",
        rsgdf_tail_ke_cutoff=4321.0,
    )

    assert row.status == "pending"
    assert captured["env"]["VIBEQC_REGRESSION_RSGDF_TAIL_KE_CUTOFF"] == "4321.0"


def test_mocked_pyscf_reference_artifacts_are_retained(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "reference" / "pyscf"
    payload = {"kind": "molecule", "basis": "sto-3g"}
    parsed = {
        "status": "ok",
        "code_version": "mock-pyscf",
        "energy_ha": -1.0,
        "wall_s": 0.01,
        "converged": True,
        "n_iter": 2,
    }
    stdout = (
        "raw PySCF output\n"
        + runner_pyscf._RESULT_MARKER
        + json.dumps(parsed, sort_keys=True)
        + "\n"
    )

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="raw stderr\n")

    monkeypatch.setattr(runner_pyscf.subprocess, "run", fake_run)

    result = runner_pyscf._run_external_pyscf(
        payload,
        tmp_path / "verbose.log",
        artifact_dir=artifact_dir,
    )

    assert result["energy_ha"] == -1.0
    assert json.loads((artifact_dir / "input.json").read_text()) == payload
    assert "raw PySCF output" in (artifact_dir / "stdout.log").read_text()
    assert "raw stderr" in (artifact_dir / "stderr.log").read_text()
    assert json.loads((artifact_dir / "parsed.json").read_text())["code_version"] == "mock-pyscf"
    assert (artifact_dir / "runner.py").is_file()


def test_orca_parity_matrix_defaults_to_output_root(tmp_path, monkeypatch):
    from examples.regression.parity_matrix_orca import run_parity

    captured = {}

    def fake_evaluate_cell(
        cell,
        *,
        run,
        vq_kw,
        cache_dir,
        workspace_root,
        fetch_root,
        update_fixture_cache=False,
    ):
        captured["run"] = run
        captured["cache_dir"] = cache_dir
        captured["workspace_root"] = workspace_root
        captured["fetch_root"] = fetch_root
        captured["update_fixture_cache"] = update_fixture_cache
        return {"cell": cell, "status": "no-cache", "note": "unit test"}

    monkeypatch.setattr(run_parity, "evaluate_cell", fake_evaluate_cell)

    rc = run_parity.main([
        "--first",
        "--output-root",
        str(tmp_path),
        "--run-id",
        "unit-orca-parity",
    ])

    run_dir = tmp_path.resolve() / "unit-orca-parity"
    assert rc == 0
    assert (run_dir / "PARITY_REPORT.md").is_file()
    assert captured["cache_dir"] == run_dir / "cache"
    assert captured["workspace_root"] == run_dir / "vq_work"
    assert captured["fetch_root"] == run_dir / "vq_fetch"
    assert captured["update_fixture_cache"] is False


def test_orca_parity_live_cache_stays_in_output_root_by_default(tmp_path, monkeypatch):
    from examples.regression.parity_matrix_orca import run_parity

    cell = run_parity.FIRST_CELL
    run_cache = tmp_path / "run-cache"
    fixture_cache = tmp_path / "fixture-cache"

    def fake_run_orca_cell(*_args, **_kwargs):
        return {"code_version": "mock-orca", "raw_output": "kept"}

    monkeypatch.setattr(run_parity, "run_orca_cell", fake_run_orca_cell)
    monkeypatch.setattr(run_parity, "FIXTURE_CACHE_DIR", fixture_cache)

    record = run_parity._obtain_orca(
        cell,
        run=True,
        vq_kw={},
        cache_dir=run_cache,
        workspace_root=tmp_path / "work",
        fetch_root=tmp_path / "fetch",
    )

    assert record["code_version"] == "mock-orca"
    assert (run_cache / f"{cell.cell_id}.json").is_file()
    assert not (fixture_cache / f"{cell.cell_id}.json").exists()


def test_mocked_xtb_live_artifacts_are_retained(tmp_path, monkeypatch):
    from examples.regression.core import runner_xtb

    artifact_dir = tmp_path / "reference" / "xtb"
    stdout = "   * total energy  :   -0.987562 Eh\n   SCF converged in 3 cycles\n"

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="raw xtb stderr\n")

    monkeypatch.setattr(runner_xtb.subprocess, "run", fake_run)

    result = runner_xtb._run_xtb(
        "2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.7\n",
        artifact_dir=artifact_dir,
    )

    assert result["energy"] == -0.987562
    assert (artifact_dir / "input.xyz").is_file()
    assert "total energy" in (artifact_dir / "stdout.log").read_text()
    assert "raw xtb stderr" in (artifact_dir / "stderr.log").read_text()
    assert json.loads((artifact_dir / "parsed.json").read_text())["returncode"] == 0


def test_mocked_mopac_live_artifacts_are_retained(tmp_path, monkeypatch):
    from examples.regression.core import runner_mopac

    artifact_dir = tmp_path / "reference" / "mopac"

    def fake_run(*_args, **kwargs):
        cwd = Path(kwargs["cwd"])
        (cwd / "input.out").write_text(
            "TOTAL ENERGY            =    -27.21140 EV\n"
            "JOB FINISHED\n",
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            stdout="raw mopac stdout\n",
            stderr="raw mopac stderr\n",
        )

    monkeypatch.setattr(runner_mopac.subprocess, "run", fake_run)

    result = runner_mopac._run_mopac(
        ["H 0.0 0.0 0.0"],
        artifact_dir=artifact_dir,
    )

    assert result["energy"] == pytest.approx(-1.0)
    assert (artifact_dir / "input.mop").is_file()
    assert (artifact_dir / "input.out").is_file()
    assert "raw mopac stdout" in (artifact_dir / "stdout.log").read_text()
    assert "raw mopac stderr" in (artifact_dir / "stderr.log").read_text()
    assert json.loads((artifact_dir / "parsed.json").read_text())["returncode"] == 0


def test_mocked_dftbp_live_artifacts_are_retained(tmp_path, monkeypatch):
    from examples.regression.core import runner_dftbp

    artifact_dir = tmp_path / "reference" / "dftbp"

    def fake_run(*_args, **kwargs):
        cwd = Path(kwargs["cwd"])
        (cwd / "detailed.out").write_text(
            "Total energy:    -4.2038719941 H\n"
            "SCC converged\n"
            "SCC iterations: 5\n",
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            stdout="raw dftb+ stdout\n",
            stderr="raw dftb+ stderr\n",
        )

    monkeypatch.setattr(runner_dftbp.subprocess, "run", fake_run)

    result = runner_dftbp._run_dftbp(
        "1\n\nH 0.0 0.0 0.0\n",
        artifact_dir=artifact_dir,
    )

    assert result["energy"] == -4.2038719941
    assert (artifact_dir / "geo.gen").is_file()
    assert (artifact_dir / "dftb_in.hsd").is_file()
    assert (artifact_dir / "detailed.out").is_file()
    assert "raw dftb+ stdout" in (artifact_dir / "stdout.log").read_text()
    assert "raw dftb+ stderr" in (artifact_dir / "stderr.log").read_text()
    assert json.loads((artifact_dir / "parsed.json").read_text())["returncode"] == 0


def test_mocked_openmolcas_live_artifacts_are_retained(tmp_path, monkeypatch):
    from examples.regression.core import runner_openmolcas

    artifact_dir = tmp_path / "reference" / "openmolcas"
    monkeypatch.setenv("OPENMOLCAS_PYMOLCAS", "/mock/pymolcas")
    monkeypatch.setenv("OPENMOLCAS_PYTHON", "python")
    monkeypatch.setenv("MOLCAS", "/mock/molcas")

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            stdout=(
                "Total SCF energy -1.111111\n"
                "RASSCF root number 1 Total energy: -1.234567\n"
                "CASPT2 Root 1 Total energy: -1.345678\n"
            ),
            stderr="raw openmolcas stderr\n",
            returncode=0,
        )

    monkeypatch.setattr(runner_openmolcas.subprocess, "run", fake_run)

    result = runner_openmolcas.run_caspt2(
        [(1, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 1.4))],
        basis="sto-3g",
        n_core=0,
        n_act=2,
        n_elec=2,
        artifact_dir=artifact_dir,
    )

    assert result.scf == -1.111111
    assert result.casci == -1.234567
    assert result.caspt2 == -1.345678
    assert (artifact_dir / "job.xyz").is_file()
    assert (artifact_dir / "job.input").is_file()
    plain_input = (artifact_dir / "job.input").read_text()
    assert "XField" not in plain_input
    assert "Charge =" not in plain_input
    assert "CASPT2 Root" in (artifact_dir / "stdout.log").read_text()
    assert "raw openmolcas stderr" in (artifact_dir / "stderr.log").read_text()
    parsed = json.loads((artifact_dir / "parsed.json").read_text())
    assert parsed["scf"] == -1.111111
    assert parsed["caspt2"] == -1.345678


def test_mocked_openmolcas_xms_nac_input_and_parser(tmp_path, monkeypatch):
    """The canonical oracle can regenerate the recorded XMS coupling."""
    from examples.regression.core import runner_openmolcas

    artifact_dir = tmp_path / "reference" / "openmolcas-xms-nac"
    monkeypatch.setenv("OPENMOLCAS_PYMOLCAS", "/mock/pymolcas")
    monkeypatch.setenv("OPENMOLCAS_PYTHON", "python")
    monkeypatch.setenv("MOLCAS", "/mock/molcas")

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            stdout=(
                "Total SCF energy -7.700000\n"
                "RASSCF root number 1 Total energy: -7.800000\n"
                "RASSCF root number 2 Total energy: -7.700000\n"
                "XMS-CASPT2 Root 1 Total energy: -7.87683288\n"
                "XMS-CASPT2 Root 2 Total energy: -7.74499661\n"
                "Total derivative coupling\n"
                "Li1 0.000000D+00 0.000000D+00 1.94851114390451D-01\n"
                "H2  0.000000D+00 0.000000D+00 -7.19983958016190D-02\n"
            ),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(runner_openmolcas.subprocess, "run", fake_run)

    result = runner_openmolcas.run_caspt2(
        [(3, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 3.0))],
        basis="sto-3g",
        n_core=1,
        n_act=2,
        n_elec=2,
        ci_only=False,
        nroots=2,
        multistate="xms",
        nac_pair=(0, 1),
        artifact_dir=artifact_dir,
    )

    assert result.note == ""
    assert result.ms_roots == [-7.87683288, -7.74499661]
    assert result.nac_pair == (0, 1)
    assert np.allclose(
        result.nac,
        [[0.0, 0.0, 0.194851114390451], [0.0, 0.0, -0.071998395801619]],
    )
    inp = (artifact_dir / "job.input").read_text()
    assert "RICD" in inp
    assert "XMULtistate = all" in inp
    assert "&CASPT2\n" in inp and "NAC = 1 2" in inp
    assert "&MCLR" in inp
    assert "&ALASKA\nNAC = 1 2" in inp
    parsed = json.loads((artifact_dir / "parsed.json").read_text())
    assert parsed["nac_pair"] == [0, 1]
    assert parsed["nac"] == result.nac


def test_mocked_openmolcas_xfield_charged_cluster_input(tmp_path, monkeypatch):
    """The canonical oracle writes XField in atomic units and parses SCF.

    Embedded-cluster parity previously duplicated this support in a study
    script, leaving the reusable OpenMolcas runner unable to express the job.
    """
    from examples.regression.core import runner_openmolcas

    artifact_dir = tmp_path / "reference" / "openmolcas-xfield"
    monkeypatch.setenv("OPENMOLCAS_PYMOLCAS", "/mock/pymolcas")
    monkeypatch.setenv("OPENMOLCAS_PYTHON", "python")
    monkeypatch.setenv("MOLCAS", "/mock/molcas")

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            stdout=(
                "Total SCF energy -1.283281765129D+03\n"
                "RASSCF root number 1 Total energy: -1.283300000000D+03\n"
                "CASPT2 Root 1 Total energy: -1.283350000000D+03\n"
            ),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(runner_openmolcas.subprocess, "run", fake_run)

    result = runner_openmolcas.run_caspt2(
        [(8, (0.0, 0.0, 0.0)), (12, (0.0, 0.0, 3.0))],
        basis="6-31G",
        n_core=8,
        n_act=2,
        n_elec=2,
        charge=2,
        ci_only=False,
        xfield_positions=[(0.0, 0.0, -5.0), (0.0, 0.0, 8.0)],
        xfield_charges=[1.5, -1.5],
        artifact_dir=artifact_dir,
    )

    assert result.scf == pytest.approx(-1283.281765129)
    assert result.casci == pytest.approx(-1283.3)
    assert result.caspt2 == pytest.approx(-1283.35)
    assert result.note == ""

    xyz = (artifact_dir / "job.xyz").read_text()
    assert "O 0.0000000000 0.0000000000 0.0000000000" in xyz
    assert "Mg 0.0000000000 0.0000000000 3.0000000000" in xyz

    inp = (artifact_dir / "job.input").read_text()
    xfield = (
        "XField\n"
        "2 0\n"
        "0.0000000000 0.0000000000 -5.0000000000 1.5000000000\n"
        "0.0000000000 0.0000000000 8.0000000000 -1.5000000000\n"
    )
    assert xfield in inp
    assert inp.index("XField") < inp.index("&SEWARD")
    assert "&SCF\nCharge = 2\n&RASSCF" in inp
    assert "CIonly" not in inp

    parsed = json.loads((artifact_dir / "parsed.json").read_text())
    assert parsed["scf"] == pytest.approx(-1283.281765129)


def test_mocked_openmolcas_nonzero_exit_is_not_silently_accepted(
    tmp_path, monkeypatch,
):
    """Parsed-looking energies do not hide a failed external process."""
    from examples.regression.core import runner_openmolcas

    monkeypatch.setenv("OPENMOLCAS_PYMOLCAS", "/mock/pymolcas")
    monkeypatch.setenv("OPENMOLCAS_PYTHON", "python")
    monkeypatch.setenv("MOLCAS", "/mock/molcas")

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            stdout=(
                "Total SCF energy -1.100000\n"
                "RASSCF root number 1 Total energy: -1.200000\n"
                "CASPT2 Root 1 Total energy: -1.300000\n"
            ),
            stderr="OpenMolcas failed after printing results\n",
            returncode=7,
        )

    monkeypatch.setattr(runner_openmolcas.subprocess, "run", fake_run)

    result = runner_openmolcas.run_caspt2(
        [(1, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 1.4))],
        basis="sto-3g",
        n_core=0,
        n_act=2,
        n_elec=2,
        artifact_dir=tmp_path / "failed-openmolcas",
    )

    assert result.scf == -1.1
    assert result.casci == -1.2
    assert result.caspt2 == -1.3
    assert "OpenMolcas exited with status 7" in result.note
    assert "OpenMolcas failed after printing results" in result.note
    parsed = json.loads(
        (tmp_path / "failed-openmolcas" / "parsed.json").read_text()
    )
    assert parsed["note"] == result.note


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {
                "xfield_positions": [(0.0, 0.0, 1.0)],
                "xfield_charges": [],
            },
            "equal length",
        ),
        (
            {
                "xfield_positions": [(0.0, 1.0)],
                "xfield_charges": [1.0],
            },
            "three coordinates",
        ),
        (
            {
                "xfield_positions": [(0.0, 0.0, 1.0)],
                "xfield_charges": [float("nan")],
            },
            "must be finite",
        ),
        ({"charge": 0.5}, "must be an integer"),
    ],
)
def test_openmolcas_xfield_input_validation_precedes_discovery(kwargs, message):
    """Malformed embeddings fail locally, without needing OpenMolcas."""
    from examples.regression.core import runner_openmolcas

    with pytest.raises(ValueError, match=message):
        runner_openmolcas.run_caspt2(
            [(1, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 1.4))],
            basis="sto-3g",
            n_core=0,
            n_act=2,
            n_elec=2,
            **kwargs,
        )


def test_orca_pure_gga_df_uses_ri_aux_basis():
    simple = runner_orca._orca_simpleinput(METHODS["rks-pbe-df"], "def2-svp")
    tokens = simple.split()

    assert "RI" in tokens
    assert "RIJ" not in tokens
    assert "RIJK" not in tokens
    assert "def2/J" in tokens
    assert "def2/JK" not in tokens

    hybrid_or_hf = runner_orca._orca_simpleinput(METHODS["rhf-df"], "def2-svp")
    assert "RIJK" in hybrid_or_hf.split()
    assert "def2/JK" in hybrid_or_hf.split()


def test_orca_reference_failure_remains_error(tmp_path, monkeypatch):
    import vibeqc.benchmark as benchmark

    class FailingCalculator:
        def get_potential_energy(self, _atoms=None, force_consistent=False):
            Path("orca.out").write_text(
                "INPUT ERROR\n"
                "UNRECOGNIZED OR DUPLICATED KEYWORD(S) IN SIMPLE INPUT LINE\n"
                "  RIJ\n",
                encoding="utf-8",
            )
            raise RuntimeError("ORCA exited with status 4")

    monkeypatch.setattr(benchmark, "find_orca_command", lambda: "/tmp/orca")
    monkeypatch.setattr(
        benchmark,
        "make_orca_calculator",
        lambda **_kwargs: FailingCalculator(),
    )
    monkeypatch.setattr(runner_orca, "_orca_version", lambda _cmd: "mock-orca")

    spec = MoleculeSpec(
        id="h2",
        family="molecule_closed_shell",
        atoms=(
            AtomCart("H", 1, (0.0, 0.0, 0.0)),
            AtomCart("H", 1, (0.0, 0.0, 0.74)),
        ),
    )
    method = MethodSpec(id="rks-pbe-df", scf="rks", xc="pbe", df=True)
    artifact_dir = tmp_path / "reference" / "orca"

    row = runner_orca.run_molecule_case(
        run_id="pytest",
        target="dev",
        spec=spec,
        basis_name="def2-svp",
        method=method,
        log_path=tmp_path / "orca.log",
        workdir=tmp_path / "work",
        artifact_dir=artifact_dir,
    )

    assert row.status == "error"
    assert "ORCA exited" in row.note
    assert "status 4" in row.note
    assert "RIJ" in (artifact_dir / "orca.out").read_text()
    assert json.loads((artifact_dir / "parsed.json").read_text())["status"] == "error"


def test_dense_ionic_ewald_guard_classifies_as_unavailable():
    row = CodeRow(
        run_id="pytest",
        target="dev",
        system_id="lih_rocksalt",
        family="rocksalt",
        basis="sto-3g",
        method_id="rks-lda",
        kmesh="1x1x1",
        code="vibeqc",
        code_version="test",
    )

    runner_vibeqc._classify_periodic_exception(
        row,
        ValueError(
            "Gamma-only EWALD_3D is retired for dense ionic crystals whose "
            "periodic images overlap: the largest cross-cell AO overlap is 0.41"
        ),
    )

    assert row.status == "unavailable"
    assert "retired EWALD_3D" in row.note


def test_reference_annotation_adds_size_normalized_deltas():
    ref = CodeRow(
        run_id="pytest",
        target="dev",
        system_id="benzene",
        family="molecule_aromatic",
        basis="sto-3g",
        method_id="rks-b3lyp",
        kmesh="mol",
        code="orca",
        code_version="mock",
        converged=True,
        energy_ha=-100.000,
        n_atoms=12,
        n_electrons=42,
        n_basis_functions=36,
    )
    vibe = CodeRow(
        run_id="pytest",
        target="dev",
        system_id="benzene",
        family="molecule_aromatic",
        basis="sto-3g",
        method_id="rks-b3lyp",
        kmesh="mol",
        code="vibeqc",
        code_version="mock",
        converged=True,
        energy_ha=-99.9974,
        n_atoms=12,
        n_electrons=42,
        n_basis_functions=36,
    )

    annotate_against_reference(
        [vibe, ref],
        ref_code="orca",
        expected=ExpectedRef(
            system_id="benzene",
            basis="sto-3g",
            method_id="rks-b3lyp",
            kmesh=(0, 0, 0),
            tolerance_ha=1e-3,
        ),
    )

    assert vibe.status == "marginal"
    assert vibe.delta_mha_vs_ref == pytest.approx(2.6)
    assert vibe.abs_delta_mha_per_atom_vs_ref == pytest.approx(2.6 / 12)
    assert vibe.abs_delta_mha_per_electron_vs_ref == pytest.approx(2.6 / 42)
    assert vibe.abs_delta_mha_per_basis_function_vs_ref == pytest.approx(2.6 / 36)
    assert "mHa/atom" in vibe.note


def test_summary_renders_size_aware_accuracy_ranking(tmp_path):
    row = CodeRow(
        run_id="pytest",
        target="dev",
        system_id="benzene",
        family="molecule_aromatic",
        basis="sto-3g",
        method_id="rks-b3lyp",
        kmesh="mol",
        code="vibeqc",
        code_version="mock",
        energy_ha=-99.9974,
        n_atoms=12,
        n_electrons=42,
        n_basis_functions=36,
        delta_ha_vs_ref=0.0026,
        delta_mha_vs_ref=2.6,
        abs_delta_mha_vs_ref=2.6,
        delta_mha_per_atom_vs_ref=2.6 / 12,
        abs_delta_mha_per_atom_vs_ref=2.6 / 12,
        delta_mha_per_electron_vs_ref=2.6 / 42,
        abs_delta_mha_per_electron_vs_ref=2.6 / 42,
        delta_mha_per_basis_function_vs_ref=2.6 / 36,
        abs_delta_mha_per_basis_function_vs_ref=2.6 / 36,
        ref_code="orca",
        status="marginal",
    )
    case = CaseRecord(
        system_id="benzene",
        family="molecule_aromatic",
        basis="sto-3g",
        method_id="rks-b3lyp",
        kmesh=(0, 0, 0),
        rows=[row],
    )

    out_path = tmp_path / "summary.md"
    render_summary(
        run_id="pytest",
        env={},
        cases=[case],
        out_path=out_path,
        csv_path=tmp_path / "results.csv",
    )

    text = out_path.read_text()
    assert "## Accuracy ranking" in text
    assert "abs Δ/atom" in text
    assert "atoms/bf" in text
    assert "0.217" in text


def _parity_rows():
    common = dict(run_id="test", target="dev", system_id="crystal-a",
                  family="ionic", basis="explicit", method_id="uks-pbe",
                  kmesh="2x1x1", code_version="test", converged=True,
                  energy_ha=-12., n_atoms=2, n_electrons=10, n_basis_functions=8)
    return (CodeRow(code="vibeqc", wall_s=2., **common),
            CodeRow(code="crystal", wall_s=4., **common),
            ExpectedRef(system_id="crystal-a", basis="explicit", method_id="uks-pbe",
                        kmesh=(2, 1, 1), tolerance_ha=1e-5))


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("field,value", [
    ("converged", False), ("converged", None),
    ("energy_ha", None), ("energy_ha", float("nan")),
    ("energy_ha", float("inf")), ("energy_ha", -float("inf")),
    ("system_id", "other"), ("basis", "different"),
    ("method_id", "rks-pbe0"), ("kmesh", "1x1x1"),
])
def test_comparison_rejects_invalid_or_mismatched_runs(side, field, value):
    v, r, expected = _parity_rows()
    setattr((v, r)[side], field, value)
    annotate_against_reference([v, r], "crystal", expected)
    assert v.status == "error"
    assert v.delta_ha_vs_ref is None and v.abs_delta_mha_vs_ref is None
    assert not v.ref_code


@pytest.mark.parametrize("ref_status", ["error", "unavailable"])
def test_comparison_preserves_reference_failure(ref_status):
    v, r, expected = _parity_rows()
    r.status, r.note = ref_status, "reference process failed after printing an energy"
    annotate_against_reference([v, r], "crystal", expected)
    assert r.status == ref_status
    assert r.note == "reference process failed after printing an energy"
    assert v.status == "error"


def test_comparison_rejects_duplicate_reference_and_clears_old_pass():
    from dataclasses import replace
    v, r, expected = _parity_rows()
    annotate_against_reference([v, r], "crystal", expected)
    assert v.status == "pass"
    duplicate = replace(r)
    annotate_against_reference([v, r, duplicate], "crystal", expected)
    assert all(x.status == "error" for x in (v, r, duplicate))
    assert v.delta_ha_vs_ref is None and v.abs_delta_mha_per_atom_vs_ref is None
    assert "ambiguous" in v.note


@pytest.mark.parametrize("field", ["n_atoms", "n_electrons", "n_basis_functions"])
def test_comparison_rejects_different_reported_counts(field):
    v, r, expected = _parity_rows()
    setattr(v, field, getattr(r, field) + 1)
    annotate_against_reference([v, r], "crystal", expected)
    assert v.status == "error"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1., True])
def test_comparison_rejects_invalid_tolerance(bad):
    from dataclasses import replace
    from examples.regression.core.compare import classify
    v, r, expected = _parity_rows()
    annotate_against_reference([v, r], "crystal", replace(expected, tolerance_ha=bad))
    assert v.status == "error"
    assert classify(0., bad)[0] == "error"


@pytest.mark.parametrize("delta", [None, float("nan"), float("inf"), -float("inf")])
def test_nonfinite_delta_is_a_missing_comparison(delta):
    from examples.regression.core.compare import classify
    assert classify(delta, 1e-5)[0] == "error"


def test_report_uses_only_accuracy_accepted_timing_pairs(tmp_path):
    from examples.regression.core.compare import paired_wall_ratio
    v, r, expected = _parity_rows()
    annotate_against_reference([v, r], "crystal", expected)
    assert paired_wall_ratio(v, r) == .5
    case = CaseRecord("crystal-a", "ionic", "explicit", "uks-pbe", (2, 1, 1), [v, r])
    output = tmp_path / "report.md"
    def render():
        render_summary(run_id="test", env={}, cases=[case], out_path=output,
                       csv_path=tmp_path / "rows.csv")
        return output.read_text()
    assert "Accuracy-filtered wall ratios" in render()
    case.rows.append(v)
    assert "Accuracy-filtered wall ratios" not in render()
    case.rows.pop()
    for state in ("marginal", "fail", "error", "unavailable", "pending"):
        v.status = state
        assert paired_wall_ratio(v, r) is None
        assert "Accuracy-filtered wall ratios" not in render()
    v.status = "pass"
    for bad in (0., -1., float("nan"), float("inf")):
        v.wall_s = bad
        assert paired_wall_ratio(v, r) is None
    v.wall_s = 2.
    r.converged = False
    assert paired_wall_ratio(v, r) is None


@pytest.mark.parametrize("converged,energy,reference,tolerance,expected", [
    (True, -12., -12., 1., 0), (True, -12., -13., 1., 1),
    (False, -12., -12., 1., 2), (None, -12., -12., 1., 2),
    (True, float("nan"), -12., 1., 2), (True, -12., float("inf"), 1., 2),
    (True, -12., -12., float("inf"), 2), (True, -12., -12., True, 2),
])
def test_legacy_crystal_demo_verdict_requires_valid_calculation(converged, energy, reference, tolerance, expected):
    from examples.regression.crystal_parity.crystal_demos.runner import _demo_parity_verdict
    assert _demo_parity_verdict(SimpleNamespace(converged=converged, energy=energy),
                                reference, tolerance) == expected


@pytest.mark.parametrize("route", ["gdf", "bipole"])
def test_periodic_route_crosses_isolation_without_inherited_controls(tmp_path, monkeypatch, route):
    from examples.regression.core import _isolated_runner
    captured = {}
    row = CodeRow("test", "dev", NE_FCC.id, NE_FCC.family, "sto-3g", "rhf",
                  "1x1x1", "vibeqc", "test")

    def fake_process(*args, **kwargs):
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout=run_suite._RESULT_MARKER
                               + json.dumps(row.__dict__) + "\n", stderr="")

    monkeypatch.setenv("VIBEQC_REGRESSION_PERIODIC_JK_METHOD", "wrong")
    monkeypatch.setenv("VIBEQC_REGRESSION_RSGDF_TAIL_KE_CUTOFF", "99")
    monkeypatch.setattr(run_suite.subprocess, "run", fake_process)
    run_suite._isolated_runner_call(
        code="vibeqc", spec_kind="periodic", spec=NE_FCC, basis="sto-3g",
        method=METHODS["rhf"], log_path=tmp_path / "case.log", workdir=None,
        artifact_dir=None, run_id="test", target="dev", code_version="test",
        periodic_jk_method=route,
    )
    env = captured["env"]
    assert env["VIBEQC_REGRESSION_PERIODIC_JK_METHOD"] == route
    assert "VIBEQC_REGRESSION_RSGDF_TAIL_KE_CUTOFF" not in env

    def fake_job(**kwargs):
        captured["request"] = kwargs
        return row

    monkeypatch.setenv("VIBEQC_REGRESSION_PERIODIC_JK_METHOD", env["VIBEQC_REGRESSION_PERIODIC_JK_METHOD"])
    monkeypatch.delenv("VIBEQC_REGRESSION_RSGDF_TAIL_KE_CUTOFF")
    monkeypatch.setattr(runner_vibeqc, "run_periodic_case", fake_job)
    assert _isolated_runner._run_periodic(
        "vibeqc", NE_FCC, "sto-3g", METHODS["rhf"], tmp_path / "case.log",
        "test", "dev", "test",
    ) is row
    assert captured["request"]["jk_method"] == route
    assert captured["request"]["rsgdf_tail_ke_cutoff"] is None


@pytest.mark.parametrize("dry", [False, True])
def test_bipole_cli_route_survives_final_manifest(tmp_path, monkeypatch, dry):
    captured = []
    monkeypatch.setattr(run_suite, "collect_env", lambda *_: {})

    def fake_case(**kwargs):
        captured.append(kwargs)
        return CaseRecord(NE_FCC.id, NE_FCC.family, "sto-3g", "rks-lda", (1, 1, 1))

    monkeypatch.setattr(run_suite, "run_one_periodic_case_dev", fake_case)
    args = ["--systems", NE_FCC.id, "--periodic-jk-method", "bipole",
            "--run-id", "route-test", "--output-root", str(tmp_path)]
    assert run_suite.main(args + (["--dry-run"] if dry else [])) == 0
    manifest = json.loads((tmp_path / "route-test/manifest.json").read_text())
    assert manifest["status"] == ("dry_run" if dry else "complete")
    assert manifest["planned_cases"][0]["jk_method"] == "bipole"
    assert len(captured) == (0 if dry else 1)
    if captured:
        assert captured[0]["periodic_jk_method"] == "bipole"


def test_periodic_case_forwards_route_only_to_vibeqc(tmp_path, monkeypatch):
    captured = []

    def fake_runner(**kwargs):
        captured.append(kwargs)
        return CodeRow("test", "dev", NE_FCC.id, NE_FCC.family, "sto-3g",
                       "rks-lda", "1x1x1", kwargs["code"], "test",
                       converged=True, energy_ha=-1.)

    monkeypatch.setattr(run_suite, "_isolated_runner_call", fake_runner)
    case = run_suite.run_one_periodic_case_dev(
        run_id="test", system_id=NE_FCC.id, basis="sto-3g", method_id="rks-lda",
        target="dev", env={}, run_dir=tmp_path, periodic_jk_method="bipole",
    )
    assert captured[0]["periodic_jk_method"] == "bipole"
    assert "periodic_jk_method" not in captured[1]
    assert case.rows[0].status == "pass"


def test_bipole_cli_rejects_gdf_tail_before_creating_run(tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_suite.main(["--systems", NE_FCC.id, "--periodic-jk-method", "bipole",
                        "--rsgdf-tail-ke-cutoff", "10", "--output-root", str(tmp_path)])
    assert exc.value.code == 2
    assert list(tmp_path.iterdir()) == []


def _matrix_row(mesh=(1, 1, 1), **changes):
    return dict(system_id=NE_FCC.id, basis="sto-3g", method_id="rhf",
                kmesh=list(mesh), tolerance_ha=1e-6,
                tolerance_rationale="Synthetic test threshold, not chemistry acceptance",
                reference_code="pyscf", **changes)


def _write_periodic_matrix(tmp_path, rows):
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(dict(schema_version=1, jk_method="bipole", cases=rows)))
    return path


def test_periodic_matrix_distinct_meshes_and_hybrid_method(tmp_path):
    from examples.regression.core.periodic_matrix import load_periodic_matrix
    gamma = _matrix_row()
    multi = {**_matrix_row((3, 1, 1)), "method_id": "uks-b3lyp"}
    path = _write_periodic_matrix(tmp_path, [gamma, multi])
    route, cases = load_periodic_matrix(path, load_system=run_suite._load_periodic_spec)
    assert route == "bipole"
    assert [c.kmesh for c in cases] == [(1, 1, 1), (3, 1, 1)]
    assert cases[1].method_id == "uks-b3lyp"
    assert cases[0].tolerance_ha == 1e-6


@pytest.mark.parametrize("changes", [
    {"kmesh": [True, 1, 1]}, {"kmesh": [0, 1, 1]}, {"kmesh": [1., 1, 1]},
    {"kmesh": [1, 1]}, {"kmesh": "1x1x1"}, {"kmesh": [1, 1, 1, 1]},
    {"tolerance_ha": float("nan")}, {"tolerance_ha": float("inf")},
    {"tolerance_ha": -1.}, {"tolerance_ha": True}, {"tolerance_ha": 10**400},
    {"tolerance_rationale": ""},
    {"method_id": "mp2"}, {"method_id": "rhf-df"}, {"method_id": "missing"},
    {"reference_code": "missing"}, {"basis": "../basis"}, {"system_id": "../system"},
    {"charge": 1}, {"shift": [1, 0, 0]},
])
def test_periodic_matrix_rejects_invalid_or_silently_dropped_controls(tmp_path, changes):
    from examples.regression.core.periodic_matrix import load_periodic_matrix
    path = _write_periodic_matrix(tmp_path, [{**_matrix_row(), **changes}])
    with pytest.raises(ValueError):
        load_periodic_matrix(path, load_system=run_suite._load_periodic_spec)


def test_periodic_matrix_duplicate_reference_would_collide_artifacts(tmp_path):
    from examples.regression.core.periodic_matrix import load_periodic_matrix
    path = _write_periodic_matrix(tmp_path, [_matrix_row(),
        {**_matrix_row(), "reference_code": "crystal"}])
    with pytest.raises(ValueError, match="duplicate"):
        load_periodic_matrix(path, load_system=run_suite._load_periodic_spec)


@pytest.mark.parametrize("data", [[], {}, {"schema_version": True, "jk_method": "bipole", "cases": []},
    {"schema_version": 2, "jk_method": "bipole", "cases": []},
    {"schema_version": 1, "jk_method": "bipole", "cases": []}])
def test_periodic_matrix_rejects_invalid_schema(tmp_path, data):
    from examples.regression.core.periodic_matrix import load_periodic_matrix
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_periodic_matrix(path, load_system=run_suite._load_periodic_spec)


@pytest.mark.parametrize("dry", [True, False])
def test_periodic_matrix_cli_preserves_meshes_and_failed_case_identity(tmp_path, monkeypatch, dry):
    path = _write_periodic_matrix(tmp_path, [_matrix_row(), _matrix_row((3, 1, 1))])
    captured = []
    monkeypatch.setattr(run_suite, "collect_env", lambda *_: {})

    def fake_case(**kw):
        captured.append(kw)
        if kw["kmesh"] == (3, 1, 1):
            raise RuntimeError("synthetic failure")
        return CaseRecord(NE_FCC.id, NE_FCC.family, "sto-3g", "rhf", kw["kmesh"])

    monkeypatch.setattr(run_suite, "run_one_periodic_case_dev", fake_case)
    args = ["--periodic-matrix", str(path), "--output-root", str(tmp_path / "runs"),
            "--run-id", "matrix-test"] + (["--dry-run"] if dry else [])
    assert run_suite.main(args) == 0
    manifest = json.loads((tmp_path / "runs/matrix-test/manifest.json").read_text())
    planned = manifest["planned_cases"]
    assert [c["kmesh"] for c in planned] == ["1x1x1", "3x1x1"]
    assert len({c["case_id"] for c in planned}) == 2
    assert all(c["jk_method"] == "bipole" and c["tolerance_ha"] == 1e-6 for c in planned)
    if not dry:
        assert [c["kmesh"] for c in captured] == [(1, 1, 1), (3, 1, 1)]
        assert all(c["tolerance_ha"] == 1e-6 and c["reference_code"] == "pyscf" for c in captured)
        failed = manifest["completed_cases"][1]
        assert failed["kmesh"] == "3x1x1" and failed["case_id"].endswith("3x1x1")
        assert failed["rows"][0]["status"] == "error"


@pytest.mark.parametrize("code", ["vibeqc", "pyscf"])
def test_periodic_matrix_mesh_reaches_isolated_runner(tmp_path, monkeypatch, code):
    from dataclasses import replace
    from examples.regression.core import _isolated_runner
    captured = {}
    row = CodeRow("test", "dev", NE_FCC.id, NE_FCC.family, "sto-3g", "rhf",
                  "3x1x1", code, "test")

    def fake_process(*args, **kw):
        captured["env"] = kw["env"]
        return SimpleNamespace(returncode=0, stdout=run_suite._RESULT_MARKER
                               + json.dumps(row.__dict__) + "\n", stderr="")

    monkeypatch.setenv("VIBEQC_REGRESSION_KMESH", "[9,9,9]")
    monkeypatch.setattr(run_suite.subprocess, "run", fake_process)
    run_suite._isolated_runner_call(
        code=code, spec_kind="periodic", spec=replace(NE_FCC, default_kmesh=(3, 1, 1)),
        basis="sto-3g", method=METHODS["rhf"], log_path=tmp_path / "case.log",
        workdir=None, artifact_dir=None, run_id="test", target="dev", code_version="test",
    )
    assert json.loads(captured["env"]["VIBEQC_REGRESSION_KMESH"]) == [3, 1, 1]
    monkeypatch.setenv("VIBEQC_REGRESSION_KMESH", captured["env"]["VIBEQC_REGRESSION_KMESH"])

    def fake_job(**kw):
        captured["request"] = kw
        return row

    module = runner_vibeqc if code == "vibeqc" else runner_pyscf
    monkeypatch.setattr(module, "run_periodic_case", fake_job)
    _isolated_runner._run_periodic(code, NE_FCC, "sto-3g", METHODS["rhf"],
                                  tmp_path / "case.log", "test", "dev", "test")
    assert captured["request"]["kmesh"] == (3, 1, 1)
    assert captured["request"]["spec"].default_kmesh == (3, 1, 1)


@pytest.mark.parametrize("reference_available", [True, False])
def test_periodic_matrix_uses_requested_reference_and_tolerance(tmp_path, monkeypatch, reference_available):
    from examples.regression.core import runner_crystal
    captured = []

    def fake_runner(**kw):
        code = kw.get("code", "crystal")
        captured.append((code, kw))
        mesh = kw["spec"].default_kmesh
        energy = -2. if code == "pyscf" else -1.
        row = CodeRow("test", "dev", NE_FCC.id, NE_FCC.family, "sto-3g", "rhf",
                       "x".join(map(str, mesh)), code, "synthetic",
                       converged=True, energy_ha=energy)
        if code == "crystal" and not reference_available:
            row.status, row.converged, row.energy_ha = "unavailable", None, None
        return row

    def unexpected_legacy(*args):
        pytest.fail("explicit acceptance must not reuse a legacy Gamma expected file")

    monkeypatch.setattr(run_suite, "_load_expected", unexpected_legacy)
    monkeypatch.setattr(run_suite, "_isolated_runner_call", fake_runner)
    monkeypatch.setattr(runner_crystal, "run_periodic_case", fake_runner)
    case = run_suite.run_one_periodic_case_dev(
        run_id="test", system_id=NE_FCC.id, basis="sto-3g", method_id="rhf", target="dev",
        env={}, run_dir=tmp_path, periodic_jk_method="bipole", kmesh=(3, 1, 1),
        tolerance_ha=1e-9, tolerance_rationale="synthetic threshold",
        reference_code="crystal", include_crystal=True, reference_version="CRYSTAL23",
    )
    assert [r.status for r in case.rows] == (
        ["pass", "fail", "reference"] if reference_available else ["error", "error", "unavailable"]
    )
    assert case.rows[0].ref_code == ("crystal" if reference_available else "")
    assert all(kw["spec"].default_kmesh == (3, 1, 1) for _, kw in captured)
    assert captured[-1][1]["kmesh"] == (3, 1, 1)
    assert captured[-1][1]["expected_version"] == "CRYSTAL23"


def test_periodic_matrix_cli_requires_reference_and_rejects_conflicting_selection(tmp_path):
    path = _write_periodic_matrix(tmp_path, [{**_matrix_row(), "reference_code": "crystal"}])
    base = ["--periodic-matrix", str(path), "--output-root", str(tmp_path / "runs")]
    for extra in ([], ["--dry-run", "--systems", NE_FCC.id],
                  ["--dry-run", "--periodic-jk-method", "gdf"]):
        with pytest.raises(SystemExit) as exc:
            run_suite.main(base + extra)
        assert exc.value.code == 2
    assert not (tmp_path / "runs").exists()


def test_periodic_matrix_shipped_starter_has_explicit_restricted_mesh_pairs():
    from examples.regression.core.periodic_matrix import load_periodic_matrix
    path = run_suite.REGRESSION_ROOT / "matrices/bipole_restricted_starter.json"
    route, cases = load_periodic_matrix(path, load_system=run_suite._load_periodic_spec)
    assert route == "bipole" and len(cases) == 6
    assert {(c.method_id, c.kmesh) for c in cases} == {
        (method, mesh) for method in ("rhf", "rks-pbe", "rks-b3lyp")
        for mesh in ((1, 1, 1), (2, 2, 2))
    }
    assert all(c.reference_code == "crystal" and c.reference_version == "CRYSTAL23"
               and c.tolerance_rationale and c.notes for c in cases)


@pytest.mark.parametrize("code, version", [
    ("pyscf", "CRYSTAL23"), ("cp2k", "CRYSTAL23"), ("crystal", 23),
    ("crystal", "23"), ("crystal", "CRYSTAL23 extra"), ("crystal", True),
])
def test_periodic_matrix_rejects_invalid_reference_versions(tmp_path, code, version):
    from examples.regression.core.periodic_matrix import load_periodic_matrix
    path = _write_periodic_matrix(tmp_path, [{**_matrix_row(), "reference_code": code,
                                            "reference_version": version}])
    with pytest.raises(ValueError, match="reference_version"):
        load_periodic_matrix(path, load_system=run_suite._load_periodic_spec)


@pytest.mark.parametrize("dry", [True, False])
def test_periodic_matrix_version_reaches_dispatch_and_manifest(tmp_path, monkeypatch, dry):
    path = _write_periodic_matrix(tmp_path, [{**_matrix_row(), "reference_code": "crystal",
                                            "reference_version": "CRYSTAL23"}])
    captured = []
    monkeypatch.setattr(run_suite, "collect_env", lambda *_: {})
    def fake_case(**kw):
        captured.append(kw)
        return CaseRecord(NE_FCC.id, NE_FCC.family, "sto-3g", "rhf", kw["kmesh"])
    monkeypatch.setattr(run_suite, "run_one_periodic_case_dev", fake_case)
    args = ["--periodic-matrix", str(path), "--include-crystal", "--output-root",
            str(tmp_path / "runs"), "--run-id", "version-test"] + (["--dry-run"] if dry else [])
    assert run_suite.main(args) == 0
    manifest = json.loads((tmp_path / "runs/version-test/manifest.json").read_text())
    assert manifest["planned_cases"][0]["reference_version"] == "CRYSTAL23"
    assert len(captured) == (0 if dry else 1)
    if not dry:
        assert captured[0]["reference_version"] == "CRYSTAL23"


@pytest.mark.parametrize("changes", [
    {"lattice_cutoff_bohr": 0}, {"lattice_cutoff_bohr": -3.0}, {"lattice_cutoff_bohr": True},
    {"lattice_cutoff_bohr": float("inf")}, {"lattice_cutoff_bohr": "28"},
    {"lattice_cutoff_bohr": 30.0, "nuclear_cutoff_bohr": 20.0},
])
def test_periodic_matrix_rejects_invalid_lattice_cutoffs(tmp_path, changes):
    from examples.regression.core.periodic_matrix import load_periodic_matrix
    path = _write_periodic_matrix(tmp_path, [{**_matrix_row(), **changes}])
    with pytest.raises(ValueError):
        load_periodic_matrix(path, load_system=run_suite._load_periodic_spec)


def test_periodic_matrix_cutoff_ladder_rows_are_distinct_cases(tmp_path):
    from examples.regression.core.periodic_matrix import load_periodic_matrix
    rows = [_matrix_row(), {**_matrix_row(), "lattice_cutoff_bohr": 16.0},
            {**_matrix_row(), "lattice_cutoff_bohr": 20.0, "nuclear_cutoff_bohr": 30}]
    _, cases = load_periodic_matrix(_write_periodic_matrix(tmp_path, rows),
                                    load_system=run_suite._load_periodic_spec)
    assert [c.tag() for c in cases] == ["", "cut16", "cut20_ncut30"]
    with pytest.raises(ValueError, match="duplicate"):
        load_periodic_matrix(_write_periodic_matrix(tmp_path, rows + [rows[1]]),
                             load_system=run_suite._load_periodic_spec)


def test_periodic_matrix_cutoffs_reach_the_isolated_runner(tmp_path, monkeypatch):
    from examples.regression.core import _isolated_runner
    from examples.regression.core.periodic_matrix import apply_lattice_cutoffs
    captured = {}
    row = CodeRow("test", "dev", NE_FCC.id, NE_FCC.family, "sto-3g", "rhf",
                  "1x1x1", "vibeqc", "test")

    def fake_process(*args, **kw):
        captured["env"] = kw["env"]
        return SimpleNamespace(returncode=0, stdout=run_suite._RESULT_MARKER
                               + json.dumps(row.__dict__) + "\n", stderr="")

    monkeypatch.setenv("VIBEQC_REGRESSION_LATTICE_CUTOFFS", '{"lattice_cutoff_bohr": 99}')
    monkeypatch.setattr(run_suite.subprocess, "run", fake_process)
    spec = apply_lattice_cutoffs(NE_FCC, {"lattice_cutoff_bohr": 28.0, "nuclear_cutoff_bohr": 30.0})
    run_suite._isolated_runner_call(
        code="vibeqc", spec_kind="periodic", spec=spec, basis="sto-3g", method=METHODS["rhf"],
        log_path=tmp_path / "case.log", workdir=None, artifact_dir=None,
        run_id="test", target="dev", code_version="test")
    sent = json.loads(captured["env"]["VIBEQC_REGRESSION_LATTICE_CUTOFFS"])
    assert sent == {"lattice_cutoff_bohr": 28.0, "nuclear_cutoff_bohr": 30.0}
    monkeypatch.setenv("VIBEQC_REGRESSION_LATTICE_CUTOFFS", json.dumps(sent))
    monkeypatch.setattr(runner_vibeqc, "run_periodic_case",
                        lambda **kw: captured.setdefault("spec", kw["spec"]) and row)
    _isolated_runner._run_periodic("vibeqc", NE_FCC, "sto-3g", METHODS["rhf"],
                                   tmp_path / "case.log", "test", "dev", "test")
    assert (captured["spec"].default_cutoff_bohr,
            captured["spec"].default_nuclear_cutoff_bohr) == (28.0, 30.0)


@pytest.mark.parametrize("matrix", ["bipole_restricted_starter.json", "open_shell_starter.json"])
def test_shipped_bipole_matrices_pass_fold_support_at_their_declared_cutoffs(matrix):
    """The driver refuses an S(k) fold drift above 1e-2 and only calls it
    converged below 1e-4; every shipped row must be runnable as declared."""
    import vibeqc as vq
    from vibeqc.pbc_bipole_common import s_fold_truncation_drift
    from examples.regression.core.periodic_matrix import (
        apply_lattice_cutoffs, load_periodic_matrix,
    )

    route, cases = load_periodic_matrix(run_suite.REGRESSION_ROOT / "matrices" / matrix,
                                        load_system=run_suite._load_periodic_spec)
    assert route == "bipole"
    for case in cases:
        spec = apply_lattice_cutoffs(run_suite._load_periodic_spec(case.system_id), case.lattice())
        system = runner_vibeqc._build_periodic_system(spec)
        basis = vq.make_basis(system.unit_cell_molecule(), case.basis)
        opts = vq.LatticeSumOptions()
        opts.cutoff_bohr = spec.default_cutoff_bohr
        opts.nuclear_cutoff_bohr = spec.default_nuclear_cutoff_bohr
        k_points = None
        if case.kmesh != (1, 1, 1):
            k_points = list(vq.KPoints.monkhorst_pack(system, list(case.kmesh),
                                                      symmetry=False).kpoints_cart)
        assert s_fold_truncation_drift(basis, system, opts, k_points=k_points) < 1e-4, case


def test_lih_starter_default_cutoff_is_refused_by_the_fold_guard():
    import vibeqc as vq
    from vibeqc.pbc_bipole_common import s_fold_truncation_drift
    spec = run_suite._load_periodic_spec("lih_rocksalt")
    system = runner_vibeqc._build_periodic_system(spec)
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = spec.default_cutoff_bohr
    assert s_fold_truncation_drift(vq.make_basis(system.unit_cell_molecule(), "sto-3g"),
                                   system, opts) > 1e-2


def _performance_samples():
    from dataclasses import asdict
    from examples.regression.core.performance import protocol_digest
    v, r, expected = _parity_rows()
    manifest = dict(schema="vibeqc-performance-v1", repetitions=3, cache_mode="cold",
                    target_ratio=1., cases=[dict(id="required", expected=asdict(expected),
                    reference_code="crystal", accuracy_evidence="accepted-case-record")])
    manifest["sha256"] = protocol_digest(manifest)
    samples = []
    for repeat in range(1, 4):
        pair = dict(case_id="required", repeat=repeat, protocol_sha256=manifest["sha256"])
        for side, row in (("candidate", v), ("reference", r)):
            d = asdict(row)
            d.update(run_id="campaign", n_iter=6, code_version="CRYSTAL23" if side == "reference" else "test",
                     input_qualification="qualified" if side == "candidate" else "reference")
            receipt = dict(hardware_id="synthetic-cpu", cpus=2, memory_limit_mb=4096,
                           execution_identity=f"{side}-{repeat}", ranks=1, threads_per_rank=2,
                           accelerator_identity="none",
                           peak_rss_mb=100, wall_uncertainty_s=.01, cache_mode="cold",
                           timing_scope="setup-through-final-confirmation",
                           phase_wall_s=dict(setup=row.wall_s*.25, scf=row.wall_s*.5, final=row.wall_s*.25),
                           runtime_identity=f"runtime-{side}", input_identity=f"input-{side}",
                           output_identity=f"output-{side}-{repeat}", qualification_evidence="matched-input-record",
                           measurement_evidence=f"measurement-{side}-{repeat}")
            pair[side] = dict(row=d, receipt=receipt)
        samples.append(pair)
    return manifest, samples


def test_performance_protocol_reports_every_required_case_and_worst_repeat():
    from examples.regression.core.performance import assess_performance, protocol_digest
    manifest, samples = _performance_samples()
    report = assess_performance(manifest, samples)
    assert report["status"] == "pass"
    assert report["worst_upper_ratio"] == pytest.approx(2.01/3.99)
    assert report["cases"][0]["upper_ratio_range"] == [2.01/3.99]*2
    assert "not a confidence interval" in report["variability"]
    # One slow required repetition must not disappear behind a fast median.
    samples[2]["candidate"]["row"]["wall_s"] = 5.
    samples[2]["candidate"]["receipt"]["phase_wall_s"]["scf"] = 4.
    report = assess_performance(manifest, samples)
    assert report["status"] == "incomplete" and report["cases"][0]["status"] == "slow"
    manifest["cases"].append(dict(manifest["cases"][0], id="missing-required-case"))
    manifest["sha256"] = protocol_digest(manifest)
    for pair in samples:
        pair["protocol_sha256"] = manifest["sha256"]
    report = assess_performance(manifest, samples)
    assert report["cases"][1]["status"] == "blocked"
    assert report["status"] == "incomplete"


@pytest.mark.parametrize("field,value", [
    ("hardware_id", "different-cpu"), ("cpus", 4), ("memory_limit_mb", 8192),
    ("timing_scope", "scf-only"), ("cache_mode", "warm"), ("runtime_identity", ""),
    ("input_identity", None), ("output_identity", ""), ("qualification_evidence", ""),
    ("measurement_evidence", ""), ("peak_rss_mb", None),
    ("wall_uncertainty_s", float("nan")), ("wall_uncertainty_s", 3.),
    ("phase_wall_s", {"scf": 2.}), ("execution_identity", "candidate-2"),
    ("ranks", 2), ("threads_per_rank", 1), ("accelerator_identity", "different"),
])
def test_performance_protocol_blocks_incomplete_or_unmatched_receipts(field, value):
    from examples.regression.core.performance import assess_performance
    manifest, samples = _performance_samples()
    samples[0]["candidate"]["receipt"][field] = value
    report = assess_performance(manifest, samples)
    assert report["status"] == "incomplete"
    assert report["cases"][0]["status"] == "blocked"


@pytest.mark.parametrize("field,value", [
    ("energy_ha", -11.), ("converged", False), ("input_qualification", ""),
    ("n_iter", 0), ("n_retries", 1), ("run_id", ""),
])
def test_performance_protocol_recomputes_accuracy_and_rejects_reused_runs(field, value):
    from examples.regression.core.performance import assess_performance
    manifest, samples = _performance_samples()
    samples[0]["candidate"]["row"][field] = value
    assert assess_performance(manifest, samples)["status"] == "incomplete"


def test_performance_protocol_freeze_duplicates_and_cli(tmp_path, capsys):
    from examples.regression.core.performance import assess_performance, protocol_digest
    from examples.regression.performance_report import main
    manifest, samples = _performance_samples()
    assert assess_performance(manifest, samples[:-1])["status"] == "incomplete"
    assert assess_performance(manifest, samples + samples[:1])["status"] == "incomplete"
    manifest["cache_mode"] = "warm"
    with pytest.raises(ValueError, match="digest"):
        assess_performance(manifest, samples)
    manifest["cache_mode"] = "cold"
    manifest_path, samples_path = tmp_path/"protocol.json", tmp_path/"samples.json"
    manifest_path.write_text(json.dumps(manifest))
    samples_path.write_text(json.dumps(samples))
    assert main([str(manifest_path), str(samples_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    samples_path.write_text("[]")
    assert main([str(manifest_path), str(samples_path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete"
    manifest["cases"][0]["expected"]["method_id"] = "rks-skala"
    manifest["sha256"] = protocol_digest(manifest)
    with pytest.raises(ValueError, match="SKALA"):
        assess_performance(manifest, samples)


def test_performance_protocol_does_not_pool_changed_runtimes_or_hide_uncertainty():
    from examples.regression.core.performance import assess_performance
    manifest, samples = _performance_samples()
    samples[1]["candidate"]["receipt"]["runtime_identity"] = "changed-build"
    report = assess_performance(manifest, samples)
    assert report["cases"][0]["status"] == "blocked"
    manifest, samples = _performance_samples()
    for pair in samples:
        pair["candidate"]["row"]["wall_s"] = 4.
        pair["candidate"]["receipt"]["phase_wall_s"]["scf"] = 3.
    report = assess_performance(manifest, samples)
    assert report["cases"][0]["status"] == "uncertain"
    assert report["status"] == "incomplete"


def test_performance_protocol_separates_skala_checkpoint_comparison():
    from examples.regression.core.performance import assess_performance, protocol_digest
    manifest, samples = _performance_samples()
    case = manifest["cases"][0]
    case.update(reference_code="checkpoint-reference", checkpoint_identity="checkpoint-content-digest")
    case["expected"]["method_id"] = "uks-skala"
    manifest["sha256"] = protocol_digest(manifest)
    for pair in samples:
        pair["protocol_sha256"] = manifest["sha256"]
        pair["reference"]["row"]["code"] = "checkpoint-reference"
        for side in ("candidate", "reference"):
            pair[side]["row"]["method_id"] = "uks-skala"
            pair[side]["receipt"]["checkpoint_identity"] = case["checkpoint_identity"]
    report = assess_performance(manifest, samples)
    assert report["status"] == "pass"
    assert report["cases"][0]["comparison_kind"] == "same-functional checkpoint"
    samples[0]["reference"]["receipt"]["checkpoint_identity"] = "other-checkpoint"
    assert assess_performance(manifest, samples)["status"] == "incomplete"


@pytest.mark.parametrize("version", ["CRYSTAL14", "CRYSTAL230", "unknown", "23.1", None])
def test_performance_protocol_uses_canonical_crystal_runner_version(version):
    from examples.regression.core.performance import assess_performance
    manifest, samples = _performance_samples()
    samples[0]["reference"]["row"]["code_version"] = version
    assert assess_performance(manifest, samples)["status"] == "incomplete"
