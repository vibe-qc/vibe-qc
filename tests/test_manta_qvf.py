"""QVF exports executed molecular engine receipts without inventing sources."""
import hashlib
import io
import json
import zipfile
from types import SimpleNamespace

import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc.output.formats.qvf import qvf_bytes, validate_qvf, write_qvf
from vibeqc.output.plan import OutputPlan
from tests.test_manta_bupo_scf import _system, _options, _bupo
from tests.test_manta_ri_bupo_scf import _coupling


def _run(method="rhf", algorithm="exact", backend="manta", loose=False):
    mol, basis = _system(method); options = _options(method); options.max_iter = 0
    coulomb = vq.CoulombBuildOptions(algorithm, basis if algorithm in ("ri", "ri_bupo") else None,
        bupo=_bupo(loose=loose), ri_bupo=_coupling(loose=loose))
    result = getattr(vq, "run_"+method)(mol, basis, options,
        integral_engine=vq.IntegralEngineOptions(backend=backend), coulomb=coulomb)
    return mol, basis, result


def _read(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        sections = [s for s in manifest["sections"] if s["kind"] == "x_vibeqc.molecular_integral_engine"]
        assert len(sections) == 1
        section = sections[0]; member = section["members"]["data"]
        payload = archive.read(member["path"])
        assert member["sha256"] == hashlib.sha256(payload).hexdigest()
        assert section["critical"] is False
        return manifest, json.loads(payload)


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("algorithm", ["exact", "ri", "bupo", "ri_bupo"])
def test_file_and_memory_exports_retain_complete_executed_receipt(tmp_path, method, algorithm):
    mol, basis, result = _run(method, algorithm)
    plan = OutputPlan.from_run_job_kwargs(output=tmp_path/"engine", method=method, basis="sto-3g", functional=None)
    context = dict(molecule=mol, result=result, method=method, basis="sto-3g", record_hostname=False)
    memory = qvf_bytes(plan, **context)
    path = write_qvf(tmp_path/"engine", plan, atomic=True, **context)
    expected = manta.molecular_engine_provenance(result)
    for data in [memory, path.read_bytes()]:
        manifest, payload = _read(data)
        assert payload == expected
        assert payload["method"] == method and payload["coulomb_algorithm"] == algorithm
        assert payload["schema_version"] == 4
        assert manifest["extensions"]["x_vibeqc"] == {"version": "1.0", "critical": False}
        assert manifest["provenance"]["scf_converged"] is False  # NOITER remains NOITER.
        # An HCORE density can already be stationary; the receipt records
        # that fact without declaring a zero-iteration calculation converged.
        assert payload["stationarity_confirmed"] == result.engine_provenance.stationarity_confirmed
    assert validate_qvf(path)["valid"]


def test_explicit_libint_reference_is_not_relabelled_as_manta(tmp_path):
    mol, _, result = _run(backend="libint")
    plan = OutputPlan.from_run_job_kwargs(output=tmp_path/"libint", method="rhf", basis="sto-3g", functional=None)
    _, payload = _read(qvf_bytes(plan, molecule=mol, result=result, record_hostname=False))
    assert payload["one_electron_provider"] == payload["coulomb_provider"] == payload["exchange_provider"] == "libint"
    assert not payload["primitive_evaluations_measured"] and payload["citation_methods"] == []


@pytest.mark.parametrize("algorithm", ["bupo", "ri_bupo"])
def test_failed_approximate_confirmation_remains_failed_in_export(tmp_path, algorithm):
    mol, _, result = _run(algorithm=algorithm, loose=True)
    plan = OutputPlan.from_run_job_kwargs(output=tmp_path/"failed", method="rhf", basis="sto-3g", functional=None)
    _, payload = _read(qvf_bytes(plan, molecule=mol, result=result, record_hostname=False))
    assert not payload["full_build_confirmed"] and not payload["stationarity_confirmed"]
    assert payload["terminal_fock_delta"] > 1e-6
    assert payload["confirmation_coulomb_algorithm"] == ("exact" if algorithm == "bupo" else "ri")
    assert payload["confirmation_coulomb_source_identity"] != payload["coulomb_source_identity"]
    assert "manta_bupo" in payload["citation_methods"]


@pytest.mark.parametrize("mode", ["file", "bytes"])
def test_preserves_other_extensions_and_never_mutates_caller_metadata(tmp_path, mode):
    mol, _, result = _run()
    plan = OutputPlan.from_run_job_kwargs(output=tmp_path/"mixed", method="rhf", basis="sto-3g", functional=None)
    extensions = {"x_example": {"version": "2.0", "critical": False},
                  "x_vibeqc": {"version": "1.2", "critical": False}}
    extra = [{"id": "other", "kind": "x_example.record", "payload": {"value": 7}}]
    before = json.dumps([extensions, extra], sort_keys=True)
    context = dict(molecule=mol, result=result, extensions=extensions, vendor_json_sections=extra, record_hostname=False)
    data = qvf_bytes(plan, **context) if mode == "bytes" else write_qvf(tmp_path/"mixed", plan, **context).read_bytes()
    manifest, _ = _read(data)
    assert manifest["extensions"] == extensions
    assert any(s["kind"] == "x_example.record" for s in manifest["sections"])
    assert json.dumps([extensions, extra], sort_keys=True) == before


@pytest.mark.parametrize("key,value", [("kind", "x_vibeqc.molecular_integral_engine"),
    ("id", "molecular_integral_engine"), ("path", "provenance/molecular_integral_engine.json")])
def test_caller_cannot_override_executed_source_and_atomic_output_is_preserved(tmp_path, key, value):
    mol, _, result = _run(); plan = OutputPlan.from_run_job_kwargs(output=tmp_path/"reserved", method="rhf", basis="sto-3g", functional=None)
    context = dict(molecule=mol, result=result, record_hostname=False)
    path = write_qvf(tmp_path/"reserved", plan, atomic=True, **context); before = path.read_bytes()
    spec = {"id": "other", "kind": "x_example.override", "payload": {"coulomb_provider": "invented"}, key: value}
    with pytest.raises(ValueError, match="reserved"):
        write_qvf(tmp_path/"reserved", plan, atomic=True, vendor_json_sections=[spec], **context)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="reserved"):
        qvf_bytes(plan, vendor_json_sections=[spec], **context)


def test_legacy_or_missing_result_does_not_infer_provenance_from_requested_options(tmp_path):
    mol, _ = _system("rhf")
    plan = OutputPlan.from_run_job_kwargs(output=tmp_path/"legacy", method="rhf", basis="sto-3g", functional=None)
    for result in [None, SimpleNamespace(energy=-1., converged=False, engine_provenance=None)]:
        data = qvf_bytes(plan, molecule=mol, result=result, integral_engine="manta", coulomb="ri_bupo", record_hostname=False)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            assert not any(s["kind"] == "x_vibeqc.molecular_integral_engine" for s in manifest["sections"])
            assert "extensions" not in manifest
