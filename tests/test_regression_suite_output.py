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
