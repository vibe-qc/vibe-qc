"""Selected HF geometry surfaces preserve basis content and fitting policy."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc.geomopt import ConvergencePolicy, run_geomopt
from tests.test_manta_molecular_hf import _system, _options


def _source(unrestricted=False, j="ri", k="ri", *, custom=False, **kwargs):
    mol, basis = _system(unrestricted)
    auxiliary = vq.BasisSet(mol, "sto-3g")
    if custom:
        shells = auxiliary.shells()
        for shell in shells:
            shell.exponents = [1.23*x for x in shell.exponents]
        auxiliary = vq.BasisSet(mol, shells, "sto-3g", True)
    policy = kwargs.pop("policy", manta.RIOptions()); policy.mode_block = 1
    source = manta.MolecularHFGeometrySource(mol, basis, method="uhf" if unrestricted else "rhf",
        options=kwargs.pop("options", _options(unrestricted)),
        coulomb=vq.CoulombBuildOptions(j, auxiliary if "ri" in (j, k) else None, policy),
        exchange=vq.ExchangeBuildOptions(k), **kwargs)
    return mol, basis, auxiliary, source


def _displace(mol, delta):
    return vq.Molecule([vq.Atom(a.Z, (np.asarray(a.xyz)+d).tolist()) for a, d in zip(mol.atoms, delta)],
                       mol.charge, mol.multiplicity)


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("j,k", [("exact", "exact"), ("ri", "ri"), ("ri", "exact"), ("exact", "ri")])
def test_selected_surface_derivatives_and_geometry_specific_sources(unrestricted, j, k):
    mol, basis, auxiliary, source = _source(unrestricted, j, k, custom=True)
    assert source.last_result is source.last_basis is None
    e, gradient = source(mol)
    first = source.last_result.engine_provenance
    assert source.last_result.converged and first.stationarity_confirmed
    np.testing.assert_array_equal(vq.compute_overlap(source.last_basis), vq.compute_overlap(basis))
    direction = np.array([[.1, -.2, .3], [-.3, .2, .1]]+([[.2, -.1, -.2]] if unrestricted else []))
    for h in (2e-4, 1e-4):
        ep = source(_displace(mol, h*direction))[0]
        em = source(_displace(mol, -h*direction))[0]
        assert np.dot(gradient, direction.ravel()) == pytest.approx((ep-em)/(2*h), abs=5e-8)
    assert first.molecule_identity != source.last_result.engine_provenance.molecule_identity
    assert first.orbital_basis_identity != source.last_result.engine_provenance.orbital_basis_identity
    assert first.coulomb_source_identity != source.last_result.engine_provenance.coulomb_source_identity
    assert first.exchange_source_identity != source.last_result.engine_provenance.exchange_source_identity
    again, again_gradient = source(mol)
    assert again == e
    np.testing.assert_array_equal(again_gradient, gradient)
    assert first.coulomb_source_identity == source.last_result.engine_provenance.coulomb_source_identity


def test_geometry_source_owns_options_metric_and_budget():
    opts = _options(); policy = manta.RIOptions(); budget = manta.ResourceBudget()
    mol, _, _, source = _source(options=opts, policy=policy,
        integral_engine=vq.IntegralEngineOptions(backend="manta", budget=budget))
    expected = source(mol)
    opts.max_iter = 0; policy.mode_block = 0; policy.absolute_cutoff = 1e20; budget.memory_bytes = 1
    actual = source(mol)
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    exported = source.scf_keywords(mol)
    exported["coulomb"].metric.mode_block = 0
    exported["integral_engine"].budget.memory_bytes = 1
    np.testing.assert_array_equal(source(mol)[1], expected[1])


@pytest.mark.parametrize("j,k", [("exact", "exact"), ("ri", "ri"), ("ri", "exact"), ("exact", "ri")])
def test_native_geometry_optimizer_walks_and_finishes_on_selected_surface(j, k):
    mol, _, _, source = _source(j=j, k=k)
    initial, _ = source(mol)
    result = run_geomopt(mol, source, geom_opt="bfgs", geom_conv=ConvergencePolicy(gmax=1e-6),
                        geom_max_iter=50, progress=False)
    assert result.converged
    energy, gradient = source(result.system)
    assert energy < initial-1e-3
    assert np.max(np.abs(gradient)) < 1e-6
    p = source.last_result.engine_provenance
    assert (p.one_electron_provider, p.coulomb_algorithm, p.exchange_algorithm) == ("manta", j, k)


def test_geometry_source_exact_libint_reference_and_failure_keeps_last_success():
    mol, _, _, source = _source(j="exact", k="exact", integral_engine=vq.IntegralEngineOptions(backend="libint"))
    reference = source(mol)
    _, _, _, native = _source(j="exact", k="exact")
    result = native(mol)
    assert result[0] == pytest.approx(reference[0], abs=1e-10)
    np.testing.assert_allclose(result[1], reference[1], atol=3e-9)
    last = source.last_result
    changed = vq.Molecule(mol.atoms, 2, 1)
    with pytest.raises(ValueError, match="must stay fixed"):
        source(changed)
    assert source.last_result is last


def test_geometry_source_refuses_nonconvergence_and_unqualified_sources():
    mol, basis = _system()
    opts = _options(); opts.max_iter = 0
    source = manta.MolecularHFGeometrySource(mol, basis, options=opts)
    with pytest.raises(RuntimeError, match="did not converge"):
        source(mol)
    assert source.last_result is None
    for kwargs, error in [({"method": "rks"}, "RHF or UHF"),
        ({"coulomb": vq.CoulombBuildOptions("bupo")}, "qualified exact/RI"),
        ({"coulomb": vq.CoulombBuildOptions("ri")}, "auxiliary basis"),
        ({"options": vq.RHFOptions()}, "HCORE")]:
        with pytest.raises(ValueError, match=error):
            manta.MolecularHFGeometrySource(mol, basis, **kwargs)
    moved = _displace(mol, np.ones((2, 3))*.1)
    with pytest.raises(ValueError, match="centers/atom map"):
        manta.MolecularHFGeometrySource(moved, basis)


@pytest.mark.parametrize("j,k", [("exact", "exact"), ("ri", "ri"), ("ri", "exact"), ("exact", "ri")])
@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_complete_selected_geometry_job_preserves_final_auxiliary_source(tmp_path, monkeypatch, j, k, method):
    from vibeqc import geomopt, runner
    mol, basis, auxiliary, reference = _source(j=j, k=k, custom=True)
    if method == "uhf":
        selection = reference.scf_keywords(mol)
        mol = vq.Molecule(mol.atoms, 1, 2)
        reference = manta.MolecularHFGeometrySource(mol, basis, method="uhf", options=_options(True), **selection)
    optimized = []
    original = geomopt.run_geomopt
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        optimized.append(result)
        return result
    def legacy(*args, **kwargs):
        raise AssertionError("Selected geometry job reached legacy SCF/provider")
    monkeypatch.setattr(geomopt, "run_geomopt", capture)
    monkeypatch.setattr(geomopt, "MolecularSCFProvider", legacy)
    monkeypatch.setattr(runner, "run_"+method, legacy)
    stem = tmp_path/"selected-geometry"
    result = vq.run_job(mol, method=method, basis="sto-3g", **{method+"_options": _options(method == "uhf")},
        geom_opt="bfgs", geom_conv_gmax=1e-6, max_opt_steps=50, **reference.scf_keywords(mol),
        output=stem, progress=False, name_molecule=False, record_hostname=False, crash_dump=False,
        structured_log=True, localize=False)
    assert len(optimized) == 1 and optimized[0].converged
    expected, gradient = reference(optimized[0].system)
    assert np.max(np.abs(gradient)) < 1e-6
    assert result.energy == pytest.approx(expected, abs=2e-11)
    for name in ("molecule_identity", "orbital_basis_identity", "coulomb_source_identity", "exchange_source_identity"):
        assert getattr(result.engine_provenance, name) == getattr(reference.last_result.engine_provenance, name)
    text = stem.with_suffix(".out").read_text()
    assert "HF gradient: manta" in text and "Optimized geometry" in text
    citations = stem.with_suffix(".bibtex").read_text()
    assert "pople_derivatives_1979" in citations
    assert ("higham_matrix_functions_2008" in citations) == ("ri" in (j, k))


@pytest.mark.parametrize("method,j", [("rks", "exact"), ("uks", "ri"), ("rhf", "bupo"), ("uhf", "ri_bupo")])
def test_unqualified_selected_geometry_jobs_refuse(tmp_path, method, j):
    mol, _ = _system(method in ("uhf", "uks"))
    with pytest.raises(NotImplementedError, match="geometry optimization require RHF/UHF"):
        vq.run_job(mol, method=method, basis="sto-3g", geom_opt="bfgs", output=tmp_path/"refused",
            integral_engine=vq.IntegralEngineOptions(backend="manta"), coulomb=vq.CoulombBuildOptions(j))


def test_selected_exact_and_legacy_geometry_jobs_reach_same_minimum(tmp_path):
    mol, _ = _system()
    results = []
    for name, selection in [("legacy", {}), ("manta", {"integral_engine": vq.IntegralEngineOptions(backend="manta")})]:
        result = vq.run_job(mol, method="rhf", basis="sto-3g", rhf_options=_options(),
            optimize=True, geom_opt="bfgs", geom_conv_gmax=1e-6, max_opt_steps=50,
            output=tmp_path/name, progress=False, name_molecule=False, record_hostname=False,
            localize=False, crash_dump=False, **selection)
        assert result.converged
        results.append(result)
    assert results[0].engine_provenance is None
    assert results[1].engine_provenance.one_electron_provider == "manta"
    assert results[0].energy == pytest.approx(results[1].energy, abs=1e-9)
