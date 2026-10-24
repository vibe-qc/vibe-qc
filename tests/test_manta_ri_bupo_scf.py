"""Selected molecular RI-BUPO with exact RI terminal confirmation."""
import io
import json

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc import _vibeqc_core as core
from tests.test_manta_bupo_scf import _system, _options, _density


def _coupling(loose=False):
    o = manta.BupoFittingOptions(); o.order = 12; o.opening = .8
    o.coupling_tolerance = 1e-11; o.bubbles.leaf_capacity = 1
    if loose:
        o.order = 0; o.coupling_tolerance = 1e6
        o.extents.minimum_radius = o.extents.maximum_radius = .05
        o.extents.pointwise_tolerance = o.extents.tail_mass_tolerance = 1e6
    return o


def _run(method, mol, basis, options, *, coupling=None, exchange="exact", budget=None):
    engine = vq.IntegralEngineOptions(backend="manta", channel_batch=2,
        **({"budget": budget} if budget is not None else {}))
    coulomb = vq.CoulombBuildOptions("ri" if coupling is None else "ri_bupo", basis,
        **({"ri_bupo": coupling} if coupling is not None else {}))
    return getattr(vq, "run_"+method)(mol, basis, options, integral_engine=engine,
        coulomb=coulomb, exchange=vq.ExchangeBuildOptions(exchange))


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("exchange", ["exact", "ri"])
def test_each_method_and_exchange_composes_and_confirms_same_ri_source(method, exchange):
    mol, basis = _system(method); options = _options(method)
    expected = _run(method, mol, basis, options, exchange=exchange)
    result = _run(method, mol, basis, options, coupling=_coupling(), exchange=exchange)
    assert expected.converged and result.converged
    assert result.energy == pytest.approx(expected.energy, abs=4e-9)
    np.testing.assert_allclose(_density(method, result), _density(method, expected), atol=6e-8)
    p, ref = map(manta.molecular_engine_provenance, [result, expected])
    assert p["schema_version"] == 4
    assert p["coulomb_algorithm"] == "ri_bupo" and p["confirmation_coulomb_algorithm"] == "ri"
    assert p["confirmation_coulomb_source_identity"] == ref["coulomb_source_identity"] != p["coulomb_source_identity"]
    assert p["exchange_source_identity"] == ref["exchange_source_identity"]
    assert p["ri_bupo_forward_source_identity"] == p["ri_bupo_adjoint_source_identity"]
    assert len(p["ri_bupo_forward_source_identity"]) == len(p["ri_bupo_metric_factor_identity"]) == 64
    assert p["ri_bupo_metric_rank"] == basis.nbasis
    assert p["ri_bupo_metric_inverse_norm_bound"] > 0
    assert p["ri_bupo_last_energy_bound"] >= 0 and p["ri_bupo_last_metric_residual"] >= 0
    assert p["bupo_near_provider"] == "manta" and p["bupo_order"] == 12
    assert p["bupo_build_calls"] > 0 and p["confirmation_coulomb_build_calls"] > 0
    assert p["bupo_integral_tolerance"] == 1e-11 and p["bupo_maximum_integral_error"] <= 1e-11
    assert p["full_build_confirmed"] and p["stationarity_confirmed"] and p["primitive_evaluations_measured"]
    assert "manta_bupo" in p["citation_methods"]
    json.dumps(p)
    if method in ("rhf", "uhf"):
        with pytest.raises(ValueError, match="BUPO gradients"):
            (vq.compute_gradient if method == "rhf" else vq.compute_gradient_uhf)(mol, basis, result)


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
def test_separated_fragments_execute_paired_far_fields(method):
    mol, basis = _system(method, separated=True); options = _options(method)
    if method in ("uhf", "uks"):
        restricted = "rhf" if method == "uhf" else "rks"
        fm, fb = _system(restricted); fragment = _run(restricted, fm, fb, _options(restricted))
        assert fragment.converged
        alpha, beta = np.zeros((3, 3)), np.zeros((3, 3))
        alpha[:2, :2] = beta[:2, :2] = .5*fragment.density; alpha[2, 2] = 1.
        options.initial_guess = vq.InitialGuess.READ
        options.read_density_alpha, options.read_density_beta = alpha, beta
    expected = _run(method, mol, basis, options)
    result = _run(method, mol, basis, options, coupling=_coupling())
    assert expected.converged and result.converged
    assert result.energy == pytest.approx(expected.energy, abs=1e-8)
    p = manta.molecular_engine_provenance(result)
    assert p["bupo_far_blocks"] > 0 and p["bupo_moment_panels"] > 0 and p["bupo_moment_primitive_products"] > 0
    assert p["bupo_moment_primitive_pairs"] == 0  # This older counter is for four-center BUPO only.
    assert p["full_build_confirmed"] and p["stationarity_confirmed"]
    assert p["terminal_commutator_norm"] <= options.conv_tol_grad


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("exchange", ["exact", "ri"])
def test_loose_noiter_confirmation_measures_ri_error_and_rejects_false_convergence(method, exchange):
    mol, basis = _system(method); options = _options(method)
    expected = _run(method, mol, basis, options, exchange=exchange); assert expected.converged
    read = getattr(vq, method.upper()+"Options")(options); read.initial_guess = vq.InitialGuess.READ; read.max_iter = 0
    if method in ("uhf", "uks"):
        read.read_density_alpha, read.read_density_beta = expected.density_alpha, expected.density_beta
    else: read.read_density = expected.density
    coupling = _coupling(loose=True)
    result = _run(method, mol, basis, read, coupling=coupling, exchange=exchange)
    d = _density(method, result); p = manta.molecular_engine_provenance(result)
    builder = manta.BupoRIJBuilder(basis, basis, coupling)
    approximate, exact = builder.build(d), builder.build_exact(d)
    delta = exact.J-approximate.J
    assert np.max(abs(delta)) > 1e-6
    assert p["terminal_fock_delta"] == pytest.approx(np.max(abs(delta)), abs=3e-11)
    assert p["terminal_energy_delta"] == pytest.approx(abs(.5*np.sum(d*delta)), abs=3e-11)
    assert p["ri_bupo_last_energy_bound"] == approximate.receipt.approximation_energy_bound
    assert p["bupo_last_j_bound"] == approximate.receipt.approximation_j_bound
    assert not result.converged and not p["full_build_confirmed"] and not p["stationarity_confirmed"]
    np.testing.assert_allclose(d, _density(method, expected), atol=2e-12)
    # Distinguish this confirmation from a four-center Coulomb rebuild.
    four_center = manta.DirectJKBuilder(basis).build_J(d)
    assert np.max(abs(four_center-exact.J)) > 1e-5


@pytest.mark.parametrize("method", ["rks", "uks"])
@pytest.mark.parametrize("functional,exchange", [("pbe", "exact"), ("hse06", "exact"), ("wb97x", "ri"), ("wb97x-v", "exact")])
def test_ri_bupo_preserves_xc_vv10_and_independent_range_exchange(method, functional, exchange):
    mol, basis = _system(method); options = _options(method, functional)
    if functional == "wb97x-v": options.grid.n_radial = 10; options.grid.vv10_grid_factor = 3.
    expected = _run(method, mol, basis, options, exchange=exchange)
    result = _run(method, mol, basis, options, coupling=_coupling(), exchange=exchange)
    assert expected.converged and result.converged
    for field in ["energy", "e_coulomb", "e_hf_exchange", "e_xc"]:
        assert getattr(result, field) == pytest.approx(getattr(expected, field), abs=4e-9)
    p, ref = map(manta.molecular_engine_provenance, [result, expected])
    assert p["xc_source_identity"] == ref["xc_source_identity"]
    assert p["range_exchange_source_identity"] == ref["range_exchange_source_identity"]
    assert p["confirmation_coulomb_algorithm"] == "ri" and p["full_build_confirmed"]


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("exchange", ["exact", "ri"])
def test_memory_boundary_includes_exact_ri_confirmation_and_exchange(method, exchange):
    mol, basis = _system(method); options = _options(method, "hse06"); options.max_iter = 0
    budget = manta.ResourceBudget(); budget.other_live_bytes = 43210
    first = _run(method, mol, basis, options, coupling=_coupling(), exchange=exchange, budget=budget)
    budget.memory_bytes = first.engine_provenance.integral_modeled_peak_bytes
    second = _run(method, mol, basis, options, coupling=_coupling(), exchange=exchange, budget=budget)
    assert second.energy == first.energy
    budget.memory_bytes -= 1
    with pytest.raises(ValueError, match="memory budget"):
        _run(method, mol, basis, options, coupling=_coupling(), exchange=exchange, budget=budget)


def test_approximate_stationary_solution_cannot_pass_exact_ri_confirmation():
    mol, basis = _system("rhf"); options = _options("rhf"); coupling = _coupling(loose=True)
    approximate = manta.BupoRIJBuilder(basis, basis, coupling); exchange = manta.DirectJKBuilder(basis)
    class ApproximateJK(core.JKBuilder):
        def __init__(self): super().__init__()
        def build_J(self, d): return approximate.build_J(d)
        def build_K(self, d): return exchange.build_K(d)
        def build_g_rhf(self, d, alpha_hf=1.): return self.build_J(d)-.5*alpha_hf*self.build_K(d)
    overlap = vq.compute_overlap(basis); hcore = vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol)
    native = vq.run_rhf_scf_with_jk(basis, mol.n_electrons(), overlap, hcore, mol.nuclear_repulsion(), ApproximateJK(), options)
    assert native.converged
    result = _run("rhf", mol, basis, options, coupling=coupling)
    assert not result.converged and not result.engine_provenance.full_build_confirmed
    assert result.energy == pytest.approx(native.energy, abs=3e-10)
    assert result.engine_provenance.terminal_fock_delta > 1e-6


def test_output_and_invalid_selection():
    mol, basis = _system("rhf"); options = _options("rhf")
    from vibeqc.output import OutputChannel
    stream = io.StringIO()
    with OutputChannel.to_stream(stream): _run("rhf", mol, basis, options, coupling=_coupling())
    text = stream.getvalue()
    assert "J=ri_bupo (manta)" in text and "Confirmation J: ri" in text and "RI-BUPO: exact metric rank" in text
    with pytest.raises(ValueError, match="backend='manta'"):
        vq.run_rhf(mol, basis, options, integral_engine=vq.IntegralEngineOptions(backend="libint"),
            coulomb=vq.CoulombBuildOptions("ri_bupo", basis))
    with pytest.raises(ValueError, match="auxiliary_basis"):
        vq.run_rhf(mol, basis, options, integral_engine=vq.IntegralEngineOptions(backend="manta"),
            coulomb=vq.CoulombBuildOptions("ri_bupo"))
