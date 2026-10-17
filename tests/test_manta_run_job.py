"""Selected molecular engines survive the complete job and retry pipelines."""
import json
import tomllib
import zipfile

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta, properties, runner
from tests.test_manta_bupo_scf import _system, _options, _bupo
from tests.test_manta_ri_bupo_scf import _coupling


def _job(tmp_path, method="rhf", algorithm="exact", backend="manta", exchange="exact", **kwargs):
    mol, basis = _system(method)
    options = kwargs.pop(method+"_options", _options(method))
    engine = kwargs.pop("integral_engine", vq.IntegralEngineOptions(backend=backend))
    coulomb = vq.CoulombBuildOptions(algorithm,
        "sto-3g" if algorithm in ("ri", "ri_bupo") or exchange == "ri" else None,
        bupo=_bupo(), ri_bupo=_coupling())
    stem = tmp_path/"selected"
    result = vq.run_job(mol, method=method, basis="sto-3g", output=stem,
        name_molecule=False, progress=False, record_hostname=False, crash_dump=False,
        integral_engine=engine, coulomb=coulomb, exchange=vq.ExchangeBuildOptions(exchange),
        **{method+"_options": options}, **kwargs)
    return mol, basis, options, result, stem


def _qvf(stem):
    with zipfile.ZipFile(stem.with_suffix(".qvf")) as archive:
        return json.loads(archive.read("manifest.json")), json.loads(
            archive.read("provenance/molecular_integral_engine.json"))


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("algorithm", ["exact", "ri", "bupo", "ri_bupo"])
@pytest.mark.parametrize("exchange", ["exact", "ri"])
def test_complete_job_preserves_selected_source_and_exports(tmp_path, monkeypatch, method, algorithm, exchange):
    # These are the runner's ordinary native entry points and legacy guess
    # preparation. Poison them so a hidden reroute fails the real job.
    def legacy(*args, **kwargs):
        raise AssertionError("Selected job entered the legacy integral/guess route")
    for name in ("run_rhf", "run_uhf", "run_rks", "run_uks", "_prepare_molecular_special_guess"):
        monkeypatch.setattr(runner, name, legacy)
    monkeypatch.setattr(properties, "compute_overlap", legacy)
    monkeypatch.setattr(properties, "compute_dipole", legacy)
    monkeypatch.setattr(properties, "sad_density", legacy)
    from vibeqc import iao
    for name in ("compute_overlap", "compute_overlap_two_basis", "compute_dipole"):
        monkeypatch.setattr(iao, name, legacy)
        monkeypatch.setattr(vq, name, legacy)
    mol, basis, options, result, stem = _job(tmp_path, method, algorithm, exchange=exchange,
                                           structured_log=True)
    assert result.converged
    expected = getattr(vq, "run_"+method)(mol, basis, options,
        integral_engine=vq.IntegralEngineOptions(backend="manta"),
        coulomb=vq.CoulombBuildOptions(algorithm,
            "sto-3g" if algorithm in ("ri", "ri_bupo") or exchange == "ri" else None,
            bupo=_bupo(), ri_bupo=_coupling()), exchange=vq.ExchangeBuildOptions(exchange))
    assert result.energy == pytest.approx(expected.energy, abs=2e-11)
    evidence = manta.molecular_engine_provenance(result)
    assert evidence["coulomb_algorithm"] == algorithm and evidence["exchange_algorithm"] == exchange
    assert evidence["stationarity_confirmed"] and evidence["full_build_confirmed"]
    manifest, exported = _qvf(stem)
    assert exported == evidence
    localized = [section for section in manifest["sections"] if section.get("id", "").startswith("wf_localized")]
    assert len(localized) == (3 if method in ("rhf", "rks") else 0)
    text = stem.with_suffix(".out").read_text()
    assert "Integral engine: manta" in text
    assert "Hirshfeld atomic SAD promolecules" in stem.with_suffix(".population.txt").read_text()
    population = json.loads(stem.with_suffix(".population.json").read_text())
    assert "hirshfeld" in population["unavailable"] and not population["errors"]
    assert "FAILED" not in text
    citations = stem.with_suffix(".bibtex").read_text()
    assert "mcmurchie" in citations.lower() and "neese_shark" in citations
    assert "valeev_libint" not in citations
    assert "haser_ahlrichs_schwarz" not in citations
    assert ("neese_bubblepole" in citations) == (algorithm in ("bupo", "ri_bupo"))
    events = [json.loads(line) for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()]
    assert any("engine_provenance" in event for event in events)
    assert any("property_integral_provenance" in event for event in events)
    with stem.with_suffix(".system").open("rb") as stream:
        system = tomllib.load(stream)
    assert system["outputs"]["result_status"] == "converged"
    # The runner changes only its own options copy.
    assert options.scf_mode == vq.SCFMode.AUTO
    assert options.initial_guess == vq.InitialGuess.HCORE


@pytest.mark.parametrize("backend", ["manta", "libint"])
def test_omitted_options_choose_hcore_and_cite_actual_provider(tmp_path, backend):
    mol, _ = _system("rhf")
    result = vq.run_job(mol, method="rhf", basis="sto-3g", output=tmp_path/"default", name_molecule=False,
        integral_engine=vq.IntegralEngineOptions(backend=backend), progress=False, record_hostname=False)
    assert result.converged and result.engine_provenance.one_electron_provider == backend
    citations = (tmp_path/"default.bibtex").read_text()
    assert ("valeev_libint" in citations) == (backend == "libint")
    assert ("mcmurchie" in citations.lower()) == (backend == "manta")


@pytest.mark.parametrize("algorithm", ["exact", "ri", "bupo", "ri_bupo"])
def test_rks_read_retry_preserves_every_source_and_xc_setting(tmp_path, monkeypatch, algorithm):
    first = []
    original = runner._run_single_point
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        first.append((kwargs, result))
        return result
    monkeypatch.setattr(runner, "_run_single_point", capture)
    monkeypatch.setattr(runner, "_rks_tail_trah_retry_supported", lambda **kwargs: True)
    def legacy(*args, **kwargs):
        raise AssertionError("Selected retry must retain engine options")
    monkeypatch.setattr(runner, "run_rks", legacy)
    opts = _options("rks", "hse06"); opts.max_iter = 0
    _, _, _, result, stem = _job(tmp_path, "rks", algorithm, rks_options=opts,
                                 functional="hse06", exchange="ri")
    assert len(first) == 2 and result.converged
    initial, final = [manta.molecular_engine_provenance(item[1]) for item in first]
    assert not first[0][1].converged
    for key in ("one_electron_source_identity", "coulomb_source_identity", "exchange_source_identity",
                "range_exchange_source_identity", "xc_source_identity", "bupo_partition_identity"):
        assert initial[key] == final[key]
    retry = first[1][0]["rks_options"]
    assert retry.initial_guess == vq.InitialGuess.READ and retry.trah_threshold > 0
    np.testing.assert_array_equal(retry.read_density, first[0][1].density)
    assert opts.max_iter == 0 and opts.initial_guess == vq.InitialGuess.HCORE
    assert _qvf(stem)[1] == final


def test_unconverged_rhf_does_not_restart_via_legacy_sad(tmp_path, monkeypatch):
    attempts = []
    original = runner._run_single_point
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        attempts.append(result)
        return result
    monkeypatch.setattr(runner, "_run_single_point", capture)
    def legacy(*args, **kwargs):
        raise AssertionError("Selected RHF cannot take an unqualified SAD retry")
    monkeypatch.setattr(runner, "_rhf_tail_sad_retry_supported", legacy)
    opts = _options("rhf"); opts.max_iter = 0
    with pytest.raises(RuntimeError, match="did not converge after 0 iterations"):
        _job(tmp_path, rhf_options=opts)
    assert len(attempts) == 1 and not attempts[0].converged
    assert attempts[0].engine_provenance.one_electron_provider == "manta"
    assert not (tmp_path/"selected.qvf").exists()


@pytest.mark.parametrize("configuration", [
    {"method": "mp2"}, {"method": "b2plyp"}, {"method": "seccm"},
    {"optimize": True}, {"hessian": True},
    {"tddft": True}, {"nto": True}, {"atomization": True},
    {"qtaim": True}, {"solvent": "water"}, {"cosmors": object()},
    {"dispersion": "d4"}, {"dft_plus_u": [object()]}, {"read_from": "restart.qvf"},
    {"fragments": []}, {"density_fit": True}, {"aux_basis": "sto-3g"},
    {"cosx": True}, {"smearing_temperature": .01},
    {"citype": "ccsd"}, {"triples": "t0"},
])
def test_unqualified_compositions_refuse_before_basis_and_preserve_failure_manifest(tmp_path, monkeypatch, configuration):
    mol, _ = _system("rhf")
    def unexpected(*args, **kwargs):
        raise AssertionError("Unqualified request reached basis construction")
    monkeypatch.setattr(runner, "BasisSet", unexpected)
    with pytest.raises(NotImplementedError, match="Selected integral engines"):
        vq.run_job(mol, basis="sto-3g", output=tmp_path/"refused",
                    integral_engine=vq.IntegralEngineOptions(backend="manta"), **configuration)
    assert [path.suffix for path in tmp_path.iterdir()] == [".system"]
    with next(tmp_path.glob("*.system")).open("rb") as stream:
        assert tomllib.load(stream)["outputs"]["result_status"] == "refused"


def test_explicit_unsupported_guess_is_not_replaced_or_prepared(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Unqualified guess reached legacy preparation")
    monkeypatch.setattr(runner, "_prepare_molecular_special_guess", unexpected)
    with pytest.raises(ValueError, match="HCORE or supplied-density READ"):
        _job(tmp_path, initial_guess="sad")


def test_resource_refusal_retains_explicit_budget(tmp_path):
    budget = manta.ResourceBudget(); budget.memory_bytes = 1
    with pytest.raises((ValueError, RuntimeError), match="(?i)memory|budget"):
        _job(tmp_path, integral_engine=vq.IntegralEngineOptions(backend="manta", budget=budget))
    assert budget.memory_bytes == 1


def test_coulomb_only_opt_in_and_dry_run_do_not_invent_executed_receipts(tmp_path):
    mol, _ = _system("rhf")
    result = vq.run_job(mol, method="rhf", basis="sto-3g", output=tmp_path/"only", name_molecule=False,
        coulomb=vq.CoulombBuildOptions(), progress=False, record_hostname=False)
    assert result.converged and result.engine_provenance.one_electron_provider == "libint"
    vq.run_job(mol, method="rhf", basis="sto-3g", output=tmp_path/"dry", name_molecule=False,
        integral_engine=vq.IntegralEngineOptions(backend="manta"), dry_run=True)
    assert not (tmp_path/"dry.qvf").exists()


def test_auto_method_cannot_silently_change_a_correlated_choice_to_mean_field(tmp_path):
    mol, _ = _system("rhf")
    with pytest.raises(NotImplementedError, match="resolved to 'fci'"):
        vq.run_job(mol, basis="sto-3g", output=tmp_path/"auto",
                    integral_engine=vq.IntegralEngineOptions(backend="manta"))
    assert [path.suffix for path in tmp_path.iterdir()] == [".system"]


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
def test_same_basis_read_density_survives_job_options_copy(tmp_path, method):
    mol, basis = _system(method)
    opts = _options(method)
    reference = getattr(vq, "run_"+method)(mol, basis, opts,
        integral_engine=vq.IntegralEngineOptions(backend="manta"))
    opts.initial_guess = vq.InitialGuess.READ
    if method in ("uhf", "uks"):
        opts.read_density_alpha, opts.read_density_beta = reference.density_alpha, reference.density_beta
    else:
        opts.read_density = reference.density
    _, _, _, result, _ = _job(tmp_path, method, **{method+"_options": opts})
    assert result.converged and opts.initial_guess == vq.InitialGuess.READ
    assert result.energy == pytest.approx(reference.energy, abs=3e-11)


def test_libint_bupo_near_field_keeps_both_provider_citations(tmp_path):
    mol, _ = _system("rhf")
    result = vq.run_job(mol, method="rhf", basis="sto-3g", output=tmp_path/"mixed",
        name_molecule=False, progress=False, record_hostname=False,
        integral_engine=vq.IntegralEngineOptions(backend="manta"),
        coulomb=vq.CoulombBuildOptions("bupo", bupo=_bupo("libint")))
    assert result.converged and result.engine_provenance.bupo_near_provider == "libint"
    citations = (tmp_path/"mixed.bibtex").read_text()
    assert "valeev_libint" in citations and "mcmurchie_davidson" in citations
    assert "neese_bubblepole" in citations and "haser_ahlrichs_schwarz" not in citations


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
def test_job_iao_population_and_localized_qvf_share_selected_source(tmp_path, monkeypatch, method):
    from vibeqc import iao
    def unexpected(*args, **kwargs):
        raise AssertionError("Selected IAO output entered a legacy integral route")
    for name in ("compute_overlap", "compute_overlap_two_basis", "compute_dipole"):
        monkeypatch.setattr(iao, name, unexpected)
        monkeypatch.setattr(vq, name, unexpected)
    mol, _, _, result, stem = _job(tmp_path, method, "ri_bupo", iao_analysis=True)
    assert result.iao_analysis.available, result.iao_analysis.unavailable_reason
    assert sum(result.iao_analysis.charges) == pytest.approx(mol.charge, abs=2e-10)
    if method in ("uhf", "uks"):
        assert sum(result.iao_analysis.spin_populations) == pytest.approx(mol.multiplicity-1, abs=2e-10)
    manifest, provenance = _qvf(stem)
    assert provenance == manta.molecular_engine_provenance(result)
    assert any(section["kind"] == "x_vibeqc.iao_analysis" for section in manifest["sections"])
    with zipfile.ZipFile(stem.with_suffix(".qvf")) as archive:
        np.testing.assert_allclose(np.frombuffer(archive.read("atom_properties/iao_charge.bin"), dtype="<f8"),
                                   result.iao_analysis.charges, atol=2e-12)


def test_selected_localization_opt_out_is_preserved(tmp_path):
    _, _, _, result, stem = _job(tmp_path, localize=False)
    assert result.converged
    assert not any(section.get("id", "").startswith("wf_localized") for section in _qvf(stem)[0]["sections"])


@pytest.mark.parametrize("algorithm", ["bupo", "ri_bupo"])
def test_failed_terminal_confirmation_checkpoint_retains_executed_receipt(tmp_path, algorithm):
    mol, _ = _system("rhf")
    opts = _options("rhf"); opts.max_iter = 0
    checkpoint = tmp_path/"terminal.qvf"
    with pytest.raises(RuntimeError, match="did not converge"):
        vq.run_job(mol, method="rhf", basis="sto-3g", output=tmp_path/"failed",
            rhf_options=opts, integral_engine=vq.IntegralEngineOptions(backend="manta"),
            coulomb=vq.CoulombBuildOptions(algorithm, "sto-3g" if algorithm == "ri_bupo" else None,
                bupo=_bupo(loose=True), ri_bupo=_coupling(loose=True)),
            checkpoint_qvf=checkpoint, progress=False, name_molecule=False, record_hostname=False)
    manifest, receipt = _qvf(checkpoint)
    assert manifest["provenance"]["run_status"] == "failed"
    assert not manifest["provenance"]["scf_converged"]
    assert receipt["coulomb_algorithm"] == algorithm
    assert not receipt["full_build_confirmed"] and not receipt["stationarity_confirmed"]
    assert receipt["terminal_fock_delta"] > 1e-6
