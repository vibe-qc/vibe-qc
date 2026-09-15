"""Tests for QVF job containers (``vibeqc.qvf_job``): build a pending
archive, load it back, reconstruct the system, and map the declarative
job.spec onto the runner kwarg surface (QVF spec § 5.9, § 3.2)."""

from __future__ import annotations

import json
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

import vibeqc
from vibeqc._vibeqc_core import Atom, Molecule, PeriodicSystem
from vibeqc.molecule import ANGSTROM_TO_BOHR
from vibeqc.output.formats.qvf import validate_qvf
from vibeqc.qvf_job import (
    QvfJobContainer,
    build_system,
    load_job_container,
    run_container,
    write_pending_qvf,
)


def _h2() -> Molecule:
    return Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])], 0, 1
    )


def _write_results_archive(stem, molecule):
    """Emit a minimal runner-style results archive at ``stem``.qvf."""
    from vibeqc.output.formats.qvf import write_qvf
    from vibeqc.output.plan import OutputPlan

    plan = OutputPlan.from_run_job_kwargs(
        output=stem, method="rhf", basis="sto-3g", functional=None
    )
    return write_qvf(
        stem,
        plan,
        molecule=molecule,
        run_record={
            "program": "vibe-qc",
            "log_text": "stub runner log\n",
        },
    )


def _mgo() -> PeriodicSystem:
    a = 4.2 * ANGSTROM_TO_BOHR
    return PeriodicSystem(
        dim=3,
        lattice=np.asarray(
            [[a, 0, 0], [0, a, 0], [0, 0, a]], dtype=float, order="F"
        ),
        unit_cell=[
            Atom(12, [0.0, 0.0, 0.0]),
            Atom(8, [a / 2, a / 2, a / 2]),
        ],
        charge=0,
        multiplicity=1,
    )


class TestWritePendingQvf:
    def test_molecular_round_trip(self, tmp_path):
        path = write_pending_qvf(
            _h2(),
            tmp_path / "job",
            method="rhf",
            basis="sto-3g",
            tasks=["single_point"],
        )
        report = validate_qvf(path)
        assert report["valid"], report["errors"]
        container = load_job_container(path)
        assert container.run_status == "pending"
        assert container.spec["job_type"] == "molecular"
        assert container.spec["method"] == "rhf"
        assert container.spec["basis"] == "sto-3g"
        assert container.spec["charge"] == 0
        assert container.spec["multiplicity"] == 1
        assert len(container.structure["atoms"]) == 2

    def test_periodic_round_trip(self, tmp_path):
        path = write_pending_qvf(
            _mgo(),
            tmp_path / "mgo_job",
            method="rks",
            basis="sto-3g",
            functional="pbe",
            kpoints=[2, 2, 2],
        )
        report = validate_qvf(path)
        assert report["valid"], report["errors"]
        container = load_job_container(path)
        assert container.spec["job_type"] == "periodic"
        assert container.spec["kpoints"] == [2, 2, 2]
        assert container.structure["pbc"] == [True, True, True]

    def test_kpoints_on_molecule_refused(self, tmp_path):
        with pytest.raises(ValueError, match="kpoints"):
            write_pending_qvf(_h2(), tmp_path / "bad", kpoints=[2, 2, 2])

    def test_unknown_task_refused(self, tmp_path):
        with pytest.raises(ValueError, match="unknown tasks"):
            write_pending_qvf(_h2(), tmp_path / "bad", tasks=["dance"])

    def test_reserved_option_refused(self, tmp_path):
        with pytest.raises(ValueError, match="restate"):
            write_pending_qvf(
                _h2(), tmp_path / "bad", options={"method": "rhf"}
            )


class TestLoadJobContainer:
    def test_refuses_archive_without_job_spec(self, tmp_path):
        # An ordinary results archive has no job.spec section.
        from vibeqc.output.formats.qvf import write_qvf
        from vibeqc.output.plan import OutputPlan

        plan = OutputPlan.from_run_job_kwargs(
            output=tmp_path / "plain", method="rhf", basis="sto-3g",
            functional=None,
        )
        path = write_qvf(tmp_path / "plain", plan, molecule=_h2())
        with pytest.raises(ValueError, match="job.spec"):
            load_job_container(path)

    def test_refuses_corrupted_spec_member(self, tmp_path):
        path = write_pending_qvf(_h2(), tmp_path / "job", method="rhf")
        # Rewrite the spec member without updating its manifest sha256.
        with zipfile.ZipFile(path, "r") as zf:
            names = {n: zf.read(n) for n in zf.namelist()}
        names["job_spec/spec.json"] = json.dumps(
            {"job_type": "molecular", "method": "tampered"}
        ).encode()
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in names.items():
                zf.writestr(name, data)
        with pytest.raises(ValueError, match="sha256"):
            load_job_container(path)


class TestBuildSystem:
    def _container(self, spec, structure) -> QvfJobContainer:
        from pathlib import Path

        return QvfJobContainer(
            path=Path("x.qvf"),
            manifest={},
            spec=spec,
            structure=structure,
            run_status="pending",
        )

    def test_molecular(self, tmp_path):
        path = write_pending_qvf(_h2(), tmp_path / "job", method="rhf")
        mol = build_system(load_job_container(path))
        assert isinstance(mol, Molecule)
        assert mol.charge == 0
        assert mol.multiplicity == 1
        assert len(mol.atoms) == 2
        # Positions survive the Angstrom round trip.
        assert mol.atoms[1].xyz[2] == pytest.approx(1.4, abs=1e-9)

    def test_periodic(self, tmp_path):
        path = write_pending_qvf(
            _mgo(), tmp_path / "mgo", method="rks", functional="pbe"
        )
        sys_p = build_system(load_job_container(path))
        assert isinstance(sys_p, PeriodicSystem)
        assert sys_p.dim == 3
        lat = np.asarray(sys_p.lattice)
        assert lat[0, 0] == pytest.approx(
            4.2 * ANGSTROM_TO_BOHR, rel=1e-9
        )

    def test_job_type_structure_mismatch(self):
        c = self._container(
            {"job_type": "molecular"},
            {
                "atoms": [{"atomic_number": 1, "position": [0, 0, 0]}],
                "pbc": [True, True, True],
                "lattice_vectors": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            },
        )
        with pytest.raises(ValueError, match="periodic"):
            build_system(c)

    def test_non_prefix_pbc_refused(self):
        c = self._container(
            {"job_type": "periodic"},
            {
                "atoms": [{"atomic_number": 1, "position": [0, 0, 0]}],
                "pbc": [True, False, True],
                "lattice_vectors": [[4, 0, 0], [0, 30, 0], [0, 0, 4]],
            },
        )
        with pytest.raises(ValueError, match="prefix"):
            build_system(c)


class TestRunContainer:
    def test_refuses_non_pending(self, tmp_path, monkeypatch):
        path = write_pending_qvf(_h2(), tmp_path / "job", method="rhf")
        # Flip the archive's status to converged, keeping it valid.
        with zipfile.ZipFile(path, "r") as zf:
            names = {n: zf.read(n) for n in zf.namelist()}
        manifest = json.loads(names["manifest.json"])
        manifest["provenance"]["run_status"] = "converged"
        names["manifest.json"] = json.dumps(manifest).encode()
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in names.items():
                zf.writestr(name, data)

        with pytest.raises(ValueError, match="pending"):
            run_container(path)

        # force=True proceeds (runner stubbed out).
        seen = {}

        def fake_run_job(molecule, **kwargs):
            seen["called"] = True
            return "ok"

        monkeypatch.setattr(vibeqc, "run_job", fake_run_job)
        assert (
            run_container(path, force=True, update_in_place=False) == "ok"
        )
        assert seen["called"]

    def test_molecular_kwarg_mapping(self, tmp_path, monkeypatch):
        path = write_pending_qvf(
            _h2(),
            tmp_path / "job",
            method="rks",
            basis="def2-svp",
            functional="pbe0",
            tasks=["optimize", "hessian"],
            options={"fmax": 0.03},
        )
        seen = {}

        def fake_run_job(molecule, **kwargs):
            seen["molecule"] = molecule
            seen["kwargs"] = kwargs
            return "ok"

        monkeypatch.setattr(vibeqc, "run_job", fake_run_job)
        assert run_container(path, update_in_place=False) == "ok"
        kw = seen["kwargs"]
        assert kw["method"] == "rks"
        assert kw["basis"] == "def2-svp"
        assert kw["functional"] == "pbe0"
        assert kw["optimize"] is True
        assert kw["hessian"] is True
        assert kw["fmax"] == 0.03
        # Output defaults to the container stem (job.qvf -> job.*).
        assert kw["output"].endswith("job")
        assert isinstance(seen["molecule"], Molecule)

    def test_periodic_kwarg_mapping(self, tmp_path, monkeypatch):
        path = write_pending_qvf(
            _mgo(),
            tmp_path / "mgo",
            method="rks",
            basis="sto-3g",
            functional="pbe",
            kpoints=[2, 2, 2],
        )
        seen = {}

        def fake_run_periodic_job(system, basis, **kwargs):
            seen["system"] = system
            seen["basis"] = basis
            seen["kwargs"] = kwargs
            return "ok"

        monkeypatch.setattr(
            vibeqc, "run_periodic_job", fake_run_periodic_job
        )
        assert run_container(path, update_in_place=False) == "ok"
        # The spec's basis name is resolved to a real BasisSet for the
        # periodic runner (it does not accept name strings).
        assert seen["basis"].name == "sto-3g"
        assert seen["kwargs"]["kpoints"] == [2, 2, 2]
        assert isinstance(seen["system"], PeriodicSystem)

    def test_unknown_option_refused(self, tmp_path):
        path = write_pending_qvf(
            _h2(),
            tmp_path / "job",
            method="rhf",
            basis="sto-3g",
            options={"definitely_not_a_run_job_kwarg": 1},
        )
        with pytest.raises(ValueError, match="not understood"):
            run_container(path)

    def test_overrides_win_over_spec_options(self, tmp_path, monkeypatch):
        path = write_pending_qvf(
            _h2(),
            tmp_path / "job",
            method="rhf",
            basis="sto-3g",
            options={"fmax": 0.05},
        )

        def fake_run_job(molecule, **kwargs):
            return kwargs

        monkeypatch.setattr(vibeqc, "run_job", fake_run_job)
        kw = run_container(path, fmax=0.01, update_in_place=False)
        assert kw["fmax"] == 0.01


class TestCliDispatch:
    def test_run_qvf_dispatches_to_container(self, tmp_path, monkeypatch):
        import argparse

        from vibeqc._cli import _cmd_run

        path = write_pending_qvf(
            _h2(), tmp_path / "job", method="rhf", basis="sto-3g"
        )

        seen = {}

        def fake_run_job(molecule, output=None, **kwargs):
            # Emit a minimal results archive at the output stem, like the
            # real runner, so the CLI's in-place container update runs.
            seen["called"] = True
            _write_results_archive(output, molecule)
            return types.SimpleNamespace(converged=True)

        monkeypatch.setattr(vibeqc, "run_job", fake_run_job)
        args = argparse.Namespace(
            script=str(path), force=False, output=None
        )
        assert _cmd_run(args) == 0
        assert seen["called"]
        # The container itself is now the settled archive.
        container = load_job_container(path)
        assert container.run_status == "converged"

    def test_run_qvf_non_pending_exits_2(self, tmp_path, capsys):
        import argparse

        from vibeqc._cli import _cmd_run

        path = write_pending_qvf(_h2(), tmp_path / "job", method="rhf")
        with zipfile.ZipFile(path, "r") as zf:
            names = {n: zf.read(n) for n in zf.namelist()}
        manifest = json.loads(names["manifest.json"])
        manifest["provenance"]["run_status"] = "failed"
        names["manifest.json"] = json.dumps(manifest).encode()
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in names.items():
                zf.writestr(name, data)

        args = argparse.Namespace(
            script=str(path), force=False, output=None
        )
        assert _cmd_run(args) == 2
        assert "pending" in capsys.readouterr().err


class TestInPlaceUpdate:
    def test_runner_exception_settles_complete_failed_record(
        self, tmp_path, monkeypatch
    ):
        path = write_pending_qvf(
            _h2(), tmp_path / "job", method="rhf", basis="sto-3g"
        )
        with zipfile.ZipFile(path) as zf:
            pending_manifest = json.loads(zf.read("manifest.json"))
            pending_spec_section = next(
                s
                for s in pending_manifest["sections"]
                if s["kind"] == "job.spec"
            )
            pending_spec = zf.read(
                pending_spec_section["members"]["spec"]["path"]
            )

        def exploding_run_job(molecule, **kwargs):
            with zipfile.ZipFile(path) as zf:
                manifest = json.loads(zf.read("manifest.json"))
            assert manifest["provenance"]["run_status"] == "running"
            output = Path(kwargs["output"])
            output.with_suffix(".out").write_text(
                "SCF failed after this line\n", encoding="utf-8"
            )
            raise RuntimeError("SCF blew up")

        monkeypatch.setattr(vibeqc, "run_job", exploding_run_job)
        with pytest.raises(RuntimeError, match="SCF blew up"):
            run_container(path)
        report = validate_qvf(path)
        assert report["valid"], report["errors"]
        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["provenance"]["run_status"] == "failed"
            spec_section = next(
                s
                for s in manifest["sections"]
                if s["kind"] == "job.spec"
            )
            assert spec_section == pending_spec_section
            assert zf.read(spec_section["members"]["spec"]["path"]) == (
                pending_spec
            )
            record = next(
                s
                for s in manifest["sections"]
                if s["kind"] == "run.record"
            )
            assert record["sequence"] == 0
            assert record["exit_status"] == 1
            assert zf.read(record["members"]["input"]["path"]) == (
                pending_spec
            )
            assert zf.read(record["members"]["log"]["path"]) == (
                b"SCF failed after this line\n"
            )

    def test_missing_result_archive_is_an_error(
        self, tmp_path, monkeypatch
    ):
        path = write_pending_qvf(
            _h2(), tmp_path / "job", method="rhf", basis="sto-3g"
        )

        def silent_run_job(molecule, output=None, **kwargs):
            return types.SimpleNamespace(converged=True)

        monkeypatch.setattr(vibeqc, "run_job", silent_run_job)
        with pytest.raises(ValueError, match="without a settled result"):
            run_container(path, output=tmp_path / "elsewhere")
        # Finalization failure restores the submitted pending archive.
        assert load_job_container(path).run_status == "pending"

    def test_failure_before_output_channel_embeds_empty_log(
        self, tmp_path, monkeypatch
    ):
        path = write_pending_qvf(
            _h2(), tmp_path / "job", method="rhf", basis="sto-3g"
        )

        def exploding_run_job(molecule, **kwargs):
            raise RuntimeError("failed before log")

        monkeypatch.setattr(vibeqc, "run_job", exploding_run_job)
        with pytest.raises(RuntimeError, match="failed before log"):
            run_container(path)

        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            record = next(
                section
                for section in manifest["sections"]
                if section["kind"] == "run.record"
            )
            assert manifest["provenance"]["run_status"] == "failed"
            assert zf.read(record["members"]["log"]["path"]) == b""
        report = validate_qvf(path)
        assert report["valid"], report["errors"]

    def test_dotted_output_stem_uses_writer_suffix_rules(
        self, tmp_path, monkeypatch
    ):
        path = write_pending_qvf(
            _h2(), tmp_path / "job", method="rhf", basis="sto-3g"
        )

        def fake_run_job(molecule, output=None, **kwargs):
            _write_results_archive(output, molecule)
            return types.SimpleNamespace(converged=True)

        monkeypatch.setattr(vibeqc, "run_job", fake_run_job)
        run_container(path, output=tmp_path / "result.v1")
        assert load_job_container(path).run_status == "converged"
        assert not (tmp_path / "result.qvf").exists()
        assert not (tmp_path / "result.v1.qvf").exists()

    def test_update_in_place_false_leaves_container_pending(
        self, tmp_path, monkeypatch
    ):
        path = write_pending_qvf(
            _h2(), tmp_path / "job", method="rhf", basis="sto-3g"
        )

        def fake_run_job(molecule, output=None, **kwargs):
            _write_results_archive(output, molecule)
            return types.SimpleNamespace(converged=True)

        monkeypatch.setattr(vibeqc, "run_job", fake_run_job)
        run_container(
            path, output=tmp_path / "sidecar", update_in_place=False
        )
        assert load_job_container(path).run_status == "pending"

    def test_stale_truncated_log_is_repaired(self, tmp_path, monkeypatch):
        """The runner archives the log before appending epilogues (the
        periodic optimize=True case): the archive's embedded log is
        stale and flagged truncated. The container update re-embeds the
        complete on-disk log and drops the flag."""
        path = write_pending_qvf(
            _h2(), tmp_path / "job", method="rhf", basis="sto-3g"
        )
        full_log = b"SCF converged\nOPTIMIZATION EPILOGUE\n"
        stale_log = b"SCF converged\n"

        def fake_run_job(molecule, output=None, **kwargs):
            import hashlib

            qvf_path = _write_results_archive(output, molecule)
            with zipfile.ZipFile(qvf_path, "r") as zf:
                names = {n: zf.read(n) for n in zf.namelist()}
            manifest = json.loads(names["manifest.json"])
            names["run_record/log.txt"] = stale_log
            files_doc = {"log": {"filename": "job.out", "truncated": True}}
            files_bytes = json.dumps(files_doc).encode()
            names["run_record/files.json"] = files_bytes
            manifest["sections"].append(
                {
                    "id": "run_record0",
                    "kind": "run.record",
                    "program": "vibe-qc",
                    "members": {
                        "log": {
                            "path": "run_record/log.txt",
                            "format": "binary",
                            "sha256": hashlib.sha256(
                                stale_log
                            ).hexdigest(),
                        },
                        "files": {
                            "path": "run_record/files.json",
                            "format": "json",
                            "sha256": hashlib.sha256(
                                files_bytes
                            ).hexdigest(),
                        },
                    },
                }
            )
            names["manifest.json"] = json.dumps(manifest).encode()
            with zipfile.ZipFile(qvf_path, "w") as zf:
                for name, data in names.items():
                    zf.writestr(name, data)
            # The on-disk log has the epilogue the archive missed.
            with open(f"{output}.out", "wb") as fh:
                fh.write(full_log)
            return types.SimpleNamespace(converged=True)

        monkeypatch.setattr(vibeqc, "run_job", fake_run_job)
        run_container(path)
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            record = [
                s
                for s in manifest["sections"]
                if s["kind"] == "run.record"
            ][0]
            assert zf.read(record["members"]["log"]["path"]) == full_log
            files_doc = json.loads(
                zf.read(record["members"]["files"]["path"])
            )
            assert "truncated" not in files_doc["log"]
        report = validate_qvf(path)
        assert report["valid"], report["errors"]

    def test_run_sidecars_are_streamed_as_attachments(
        self, tmp_path, monkeypatch
    ):
        path = write_pending_qvf(
            _h2(), tmp_path / "job", method="rhf", basis="sto-3g"
        )

        def fake_run_job(molecule, output=None, **kwargs):
            stem = Path(output)
            _write_results_archive(stem, molecule)
            stem.with_suffix(".out").write_text(
                "complete log\n", encoding="utf-8"
            )
            stem.with_suffix(".system").write_text(
                "[outputs]\nstatus = \"complete\"\n", encoding="utf-8"
            )
            stem.with_suffix(".perf").write_text(
                "scf 0.12 s\n", encoding="utf-8"
            )
            stem.with_suffix(".scf.jsonl").write_text(
                '{"event":"done"}\n', encoding="utf-8"
            )
            return types.SimpleNamespace(converged=True)

        monkeypatch.setattr(vibeqc, "run_job", fake_run_job)
        run_container(path)
        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            record = next(
                s
                for s in manifest["sections"]
                if s["kind"] == "run.record"
            )
            files = json.loads(
                zf.read(record["members"]["files"]["path"])
            )
            expected = {
                "attachment.system": b'[outputs]\nstatus = "complete"\n',
                "attachment.perf": b"scf 0.12 s\n",
                "attachment.structured": b'{"event":"done"}\n',
            }
            for role, payload in expected.items():
                assert zf.read(record["members"][role]["path"]) == payload
                assert files[role]["filename"]

    def test_force_rerun_carries_prior_run_record(
        self, tmp_path, monkeypatch
    ):
        path = write_pending_qvf(
            _h2(), tmp_path / "job", method="rhf", basis="sto-3g"
        )

        def fake_run_job(molecule, output=None, **kwargs):
            qvf_path = _write_results_archive(output, molecule)
            # Give the archive a run.record like the real runner does.
            with zipfile.ZipFile(qvf_path, "r") as zf:
                names = {n: zf.read(n) for n in zf.namelist()}
            manifest = json.loads(names["manifest.json"])
            import hashlib

            log = b"fake run log\n"
            names["run_record/log.txt"] = log
            manifest["sections"].append(
                {
                    "id": "run_record0",
                    "kind": "run.record",
                    "program": "vibe-qc",
                    "members": {
                        "log": {
                            "path": "run_record/log.txt",
                            "format": "binary",
                            "sha256": hashlib.sha256(log).hexdigest(),
                        }
                    },
                }
            )
            names["manifest.json"] = json.dumps(manifest).encode()
            with zipfile.ZipFile(qvf_path, "w") as zf:
                for name, data in names.items():
                    zf.writestr(name, data)
            return types.SimpleNamespace(converged=True)

        monkeypatch.setattr(vibeqc, "run_job", fake_run_job)
        run_container(path)
        first = load_job_container(path)
        assert first.run_status == "converged"

        run_container(path, force=True)
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
        records = [
            s
            for s in manifest["sections"]
            if s["kind"] == "run.record"
        ]
        assert len(records) == 2
        sequences = sorted(int(r.get("sequence", -1)) for r in records)
        assert sequences == [0, 1]
        specs = [
            s for s in manifest["sections"] if s["kind"] == "job.spec"
        ]
        assert len(specs) == 1
        report = validate_qvf(path)
        assert report["valid"], report["errors"]


class TestEndToEnd:
    def test_h2_sto3g_single_point_updates_container(self, tmp_path):
        """A pending container runs a real SCF and becomes the settled,
        self-contained archive: results + run.record + complete log +
        preserved job.spec, marked converged, validator-clean."""
        path = write_pending_qvf(
            _h2(),
            tmp_path / "h2_run",
            method="rhf",
            basis="sto-3g",
        )
        with zipfile.ZipFile(path) as zf:
            pending_manifest = json.loads(zf.read("manifest.json"))
            pending_spec_section = next(
                s
                for s in pending_manifest["sections"]
                if s["kind"] == "job.spec"
            )
            pending_spec_bytes = zf.read(
                pending_spec_section["members"]["spec"]["path"]
            )
        result = run_container(path)
        energy = float(getattr(result, "energy", np.nan))
        # H2/STO-3G RHF at 1.4 bohr: E ~ -1.117 Ha.
        assert energy == pytest.approx(-1.117, abs=5e-3)

        log_file = tmp_path / "h2_run.out"
        assert log_file.is_file()

        report = validate_qvf(path)
        assert report["valid"], report["errors"]
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            sections = {
                s["kind"]: s for s in manifest["sections"]
            }
            assert manifest["provenance"]["run_status"] == "converged"
            # The original request is preserved verbatim.
            assert sections["job.spec"] == pending_spec_section
            settled_spec_bytes = zf.read(
                sections["job.spec"]["members"]["spec"]["path"]
            )
            assert settled_spec_bytes == pending_spec_bytes
            spec = json.loads(settled_spec_bytes)
            assert spec["method"] == "rhf"
            assert spec["basis"] == "sto-3g"
            # The embedded log is the complete on-disk log.
            record = sections["run.record"]
            assert record["sequence"] == 0
            assert record["exit_status"] == 0
            assert zf.read(record["members"]["input"]["path"]) == (
                pending_spec_bytes
            )
            embedded = zf.read(record["members"]["log"]["path"])
            assert embedded == log_file.read_bytes()
            files_member = record["members"].get("files")
            assert files_member is not None
            files_doc = json.loads(zf.read(files_member["path"]))
            assert "truncated" not in files_doc.get("log", {})
            # The .system manifest travels inside the run.record, so no
            # sidecar is needed to understand the run.
            sys_member = record["members"]["attachment.system"]
            embedded_system = zf.read(sys_member["path"])
            assert embedded_system == (
                tmp_path / "h2_run.system"
            ).read_bytes()
            assert (
                files_doc["attachment.system"]["filename"]
                == "h2_run.system"
            )
            # Results are in the same archive.
            assert "scf_history" in sections or "structure" in sections

        # Re-running the settled container without force is refused.
        with pytest.raises(ValueError, match="pending"):
            run_container(path)

    def test_periodic_he_box_updates_container(self, tmp_path):
        """A pending *periodic* container runs a real Gamma SCF and is
        updated in place (He/STO-3G in an 8-bohr cube, ~2 s)."""
        a = 8.0
        sys_p = PeriodicSystem(
            dim=3,
            lattice=np.asarray(
                [[a, 0, 0], [0, a, 0], [0, 0, a]], dtype=float, order="F"
            ),
            unit_cell=[Atom(2, [0.0, 0.0, 0.0])],
            charge=0,
            multiplicity=1,
        )
        path = write_pending_qvf(
            sys_p, tmp_path / "he_pbc", method="rhf", basis="sto-3g"
        )
        result = run_container(path)
        assert bool(getattr(result, "converged", False))

        container = load_job_container(path)
        assert container.run_status == "converged"
        assert container.spec["job_type"] == "periodic"
        report = validate_qvf(path)
        assert report["valid"], report["errors"]

    def test_basis_free_periodic_container_updates_in_place(self, tmp_path):
        system = PeriodicSystem(
            dim=3,
            lattice=np.eye(3) * 8.0,
            unit_cell=[Atom(2, [0.0, 0.0, 0.0])],
            charge=0,
            multiplicity=1,
        )
        path = write_pending_qvf(
            system,
            tmp_path / "he_scc",
            method="scc_dftb",
        )
        result = run_container(
            path,
            citations=False,
            write_xyz_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
        )
        assert bool(getattr(result, "converged", False))
        assert load_job_container(path).run_status == "converged"
        report = validate_qvf(path)
        assert report["valid"], report["errors"]


class TestModuleEntryPoint:
    def test_python_dash_m_vibeqc_runs_a_container(self, tmp_path):
        """``python -m vibeqc run job.qvf`` completes the round trip.

        This is the exact invocation vq's first-class QVF dispatch uses
        (``$VENV/bin/python -m vibeqc run job.qvf`` on the resolved
        managed runtime), so the module entry point is a queue contract,
        not a convenience: without ``vibeqc/__main__.py`` every fleet
        container job dies with "No module named vibeqc.__main__".
        """
        import subprocess
        import sys

        path = write_pending_qvf(
            _h2(), tmp_path / "job", method="rhf", basis="sto-3g"
        )
        proc = subprocess.run(
            [sys.executable, "-m", "vibeqc", "run", str(path)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert proc.returncode == 0, proc.stderr
        container = load_job_container(path)
        assert container.run_status == "converged"
        report = validate_qvf(path)
        assert report["valid"], report["errors"]
