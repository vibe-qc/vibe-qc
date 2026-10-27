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
    read = vq.RHFOptions(); read.initial_guess = vq.InitialGuess.READ
    for kwargs, error in [({"method": "rks"}, "RHF or UHF"),
        ({"coulomb": vq.CoulombBuildOptions("bupo")}, "qualified exact/RI"),
        ({"coulomb": vq.CoulombBuildOptions("ri")}, "auxiliary basis"),
        ({"options": read}, "READ projection")]:
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


def test_selected_geometry_source_and_job_walk_the_dispersion_inclusive_surface(tmp_path):
    from vibeqc.dispersion import compute_dispersion, d3bj_params_for
    mol, basis = _system()
    params = d3bj_params_for("pbe0")
    bare = manta.MolecularHFGeometrySource(mol, basis, options=_options())
    dispersed = manta.MolecularHFGeometrySource(mol, basis, options=_options(), dispersion_params=params)
    e0, g0 = bare(mol)
    e, g = dispersed(mol)
    dispersion = compute_dispersion(mol, params, with_gradient=True)
    assert dispersion.energy != 0. and np.max(np.abs(dispersion.gradient)) > 0.
    assert e == pytest.approx(e0+dispersion.energy, abs=1e-12)
    assert np.allclose(g, g0+np.asarray(dispersion.gradient).ravel(), atol=1e-12)
    assert dispersed.last_result.energy == pytest.approx(e0, abs=1e-12)
    results = {}
    for name, selection in [("legacy", {}), ("manta", {"integral_engine": vq.IntegralEngineOptions(backend="manta")})]:
        results[name] = vq.run_job(mol, method="rhf", basis="sto-3g", rhf_options=_options(), dispersion=params,
            optimize=True, geom_opt="bfgs", geom_conv_gmax=1e-6, max_opt_steps=50,
            output=tmp_path/name, progress=False, name_molecule=False, record_hostname=False,
            localize=False, crash_dump=False, **selection)
    assert results["manta"].engine_provenance.one_electron_provider == "manta"
    assert results["legacy"].engine_provenance is None
    assert results["manta"].e_dispersion == pytest.approx(results["legacy"].e_dispersion, abs=1e-9)
    assert results["manta"].energy_total == pytest.approx(results["legacy"].energy_total, abs=1e-9)


def _ecp_fixture(route):
    """HBr with the bromine small-core XML ECP, or RbH with the def2-svp inline sidecar."""
    if route == "xml":
        mol = vq.Molecule([vq.Atom(35, [0., 0., 0.]), vq.Atom(1, [0., 0., 2.7])], 0, 1)
        basis = vq.BasisSet(mol, "sto-3g")
        opts = _options(); opts.max_iter = 200
        opts.ecp_centers = [vq.ECPCenter(Z=35, xyz=[0., 0., 0.])]; opts.ecp_library = "ecp10mdf"
    else:
        mol = vq.Molecule([vq.Atom(37, [0., 0., 0.]), vq.Atom(1, [0., 0., 4.5])], 0, 1)
        basis = vq.BasisSet(mol, "def2-svp")
        opts = _options(); opts.max_iter = 200
    budget = manta.ResourceBudget(); budget.memory_bytes = 2 << 30
    budget.kernel_terms = budget.scalar_terms = budget.primitive_quartets = budget.shell_calls = 1 << 50
    return mol, basis, opts, budget


@pytest.mark.parametrize("route,backend", [("xml", "libint"), ("inline", "libint"), ("inline", "manta")])
def test_ecp_geometry_source_follows_the_atoms_and_matches_the_legacy_ecp_gradient(route, backend):
    from vibeqc.ecp_metadata import ecp_centre_atom_indices, ecp_centres_follow_atoms
    from vibeqc.gradient_options import gradient_options_from_scf
    mol, basis, opts, budget = _ecp_fixture(route)
    engine = vq.IntegralEngineOptions(backend=backend, budget=budget)
    source = manta.MolecularHFGeometrySource(mol, basis, options=opts, integral_engine=engine)
    legacy_opts = vq.RHFOptions(opts)
    vq._attach_inline_ecp_from_basis_sidecar(legacy_opts, mol, basis)
    displaced = _displace(mol, np.array([[0., 0., -.05], [0., 0., .1]]))
    for geometry in (mol, displaced):
        energy, gradient = source(geometry)
        moved = source.basis_at(geometry)
        with ecp_centres_follow_atoms(legacy_opts, mol, geometry):
            reference = vq.run_rhf(geometry, moved, legacy_opts)
            expected = vq.compute_gradient(geometry, moved, reference, gradient_options_from_scf(legacy_opts))
        assert reference.converged and reference.ecp_total_ncore > 0
        assert energy == pytest.approx(reference.energy, abs=3e-10)
        np.testing.assert_allclose(gradient, np.asarray(expected).ravel(), atol=1e-7, rtol=1e-7)
        p = manta.molecular_engine_provenance(source.last_result)
        assert p["one_electron_provider"] == backend
        assert p["ecp_route"] == ("xml-library" if route == "xml" else "inline-primitives")
        assert source.last_result.ecp_total_ncore == reference.ecp_total_ncore
    # The source's own centres are back on the reference atoms after every geometry.
    ecp_centre_atom_indices(source._options, mol)


@pytest.mark.parametrize("route", ["xml", "inline"])
def test_fitted_ecp_geometry_source_gradient_matches_finite_differences(route):
    # The selected RI Lagrangian gradient differentiates V_ne(Z_eff) and the
    # effective nuclear repulsion with the SCF's effective charges and adds
    # the ECP provider's operator derivative; there is no legacy RI source
    # with the same fitting, so the oracle is the selected RI energy itself.
    mol, basis, opts, budget = _ecp_fixture(route)
    source = manta.MolecularHFGeometrySource(mol, basis, options=opts,
        integral_engine=vq.IntegralEngineOptions(backend="manta", budget=budget),
        coulomb=vq.CoulombBuildOptions("ri", "sto-3g" if route == "xml" else "def2-svp"),
        exchange=vq.ExchangeBuildOptions("ri"))
    energy, gradient = source(mol)
    p = manta.molecular_engine_provenance(source.last_result)
    assert p["coulomb_algorithm"] == p["exchange_algorithm"] == "ri" and p["one_electron_provider"] == "manta"
    assert p["ecp_route"] == ("xml-library" if route == "xml" else "inline-primitives")
    assert source.last_result.ecp_total_ncore > 0
    # Derivative options that dropped the SCF's ECP route are refused rather
    # than differentiated as an all-electron Hamiltonian.
    with pytest.raises((ValueError, RuntimeError), match="ECP route"):
        vq.compute_gradient(mol, source.last_basis, source.last_result, vq.GradientOptions())
    gradient = np.asarray(gradient).reshape(-1, 3)
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-5)
    h = 1e-3
    plus = source(_displace(mol, np.array([[0., 0., 0.], [0., 0., h]])))[0]
    minus = source(_displace(mol, np.array([[0., 0., 0.], [0., 0., -h]])))[0]
    assert abs(gradient[1, 2]) > 1e-3
    assert gradient[1, 2] == pytest.approx((plus-minus)/(2*h), abs=5e-6)


def test_selected_ecp_geometry_optimization_moves_the_centres_and_matches_the_legacy_minimum(tmp_path):
    mol, basis, opts, budget = _ecp_fixture("xml")
    engine = vq.IntegralEngineOptions(backend="libint", budget=budget)
    source = manta.MolecularHFGeometrySource(mol, basis, options=opts, integral_engine=engine)
    initial, _ = source(mol)
    optimized = run_geomopt(mol, source, geom_opt="bfgs", geom_conv=ConvergencePolicy(gmax=1e-6),
                            geom_max_iter=50, progress=False)
    assert optimized.converged
    energy, gradient = source(optimized.system)
    assert energy < initial-1e-4 and np.max(np.abs(gradient)) < 1e-6
    centre = source.last_result.ecp_xml_centers[0]
    assert int(centre.Z) == 35
    np.testing.assert_allclose(np.asarray(centre.xyz), np.asarray(optimized.system.atoms[0].xyz), atol=1e-12)
    results = {}
    for name, selection in [("legacy", {}), ("libint", {"integral_engine": engine})]:
        results[name] = vq.run_job(mol, method="rhf", basis="sto-3g", rhf_options=vq.RHFOptions(opts),
            optimize=True, geom_opt="bfgs", geom_conv_gmax=1e-6, max_opt_steps=50,
            output=tmp_path/name, progress=False, name_molecule=False, record_hostname=False,
            localize=False, crash_dump=False, **selection)
    selected, legacy = results["libint"], results["legacy"]
    assert selected.converged and selected.ecp_operator_applied and selected.ecp_total_ncore == 10
    assert selected.engine_provenance.one_electron_provider == "libint"
    assert selected.engine_provenance.ecp_route == "xml-library"
    assert legacy.engine_provenance is None
    assert selected.energy == pytest.approx(legacy.energy, abs=1e-9)
    assert selected.energy == pytest.approx(energy, abs=1e-9)
    assert "Optimized geometry" in (tmp_path/"libint.out").read_text()


def test_geometry_source_accepts_the_shared_guesses_at_every_geometry():
    mol, basis = _system()
    opts = _options(); opts.initial_guess = vq.InitialGuess.SAD
    source = manta.MolecularHFGeometrySource(mol, basis, options=opts)
    hcore = manta.MolecularHFGeometrySource(mol, basis, options=_options())
    e, g = source(mol)
    e0, g0 = hcore(mol)
    assert e == pytest.approx(e0, abs=1e-9) and np.allclose(g, g0, atol=1e-7)
    assert manta.molecular_engine_provenance(source.last_result)["initial_guess_effective"] == "SAD"
    assert source.options.initial_guess == vq.InitialGuess.SAD
