"""Selected BUPO SCF with independent exact terminal Coulomb confirmation."""
import io
import json

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc import _vibeqc_core as core


def _system(method, separated=False):
    unrestricted = method in ("uhf", "uks")
    positions = [[0., 0., 0.], [.2, -.1, 1.8]]
    if separated:
        positions += [[60., .3, -.4]] if unrestricted else [[60., .3, -.4], [60.3, -.2, 1.3]]
    elif unrestricted:
        positions += [[1.3, .4, .7]]
    mol = vq.Molecule([vq.Atom(1, p) for p in positions], 0, 2 if unrestricted else 1)
    return mol, vq.BasisSet(mol, "sto-3g")


def _options(method, functional="pbe0"):
    options = getattr(vq, method.upper()+"Options")()
    options.initial_guess = vq.InitialGuess.HCORE
    options.conv_tol_energy, options.conv_tol_grad = 1e-12, 1e-9
    options.stability_check = False
    if method.endswith("ks"):
        options.functional = functional
        options.grid.n_radial, options.grid.angular, options.grid.lebedev_order = 20, "lebedev", 11
    return options


def _bupo(backend="manta", loose=False):
    options = manta.BupoOptions(); options.order = 12; options.opening = .8
    options.integral_tolerance = 1e-11; options.bubbles.leaf_capacity = 1; options.near_backend = backend
    if loose:
        options.order = 0; options.integral_tolerance = 1e6
        options.extents.minimum_radius = options.extents.maximum_radius = .05
        options.extents.pointwise_tolerance = options.extents.tail_mass_tolerance = 1e6
    return options


def _run(method, mol, basis, options, *, bupo=None, exchange="exact", budget=None):
    engine = vq.IntegralEngineOptions(backend="manta", channel_batch=2,
                                     **({"budget": budget} if budget is not None else {}))
    coulomb = vq.CoulombBuildOptions("exact" if bupo is None else "bupo", basis if exchange == "ri" else None,
                                    **({"bupo": bupo} if bupo is not None else {}))
    return getattr(vq, "run_"+method)(mol, basis, options, integral_engine=engine, coulomb=coulomb,
                                     exchange=vq.ExchangeBuildOptions(exchange))


def _density(method, result):
    return result.density_alpha+result.density_beta if method in ("uhf", "uks") else result.density


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("exchange", ["exact", "ri"])
@pytest.mark.parametrize("near_backend", ["manta", "libint"])
def test_bupo_composes_with_each_method_and_independent_exchange(method, exchange, near_backend):
    mol, basis = _system(method); options = _options(method); bupo = _bupo(near_backend)
    expected = _run(method, mol, basis, options, exchange=exchange)
    result = _run(method, mol, basis, options, bupo=bupo, exchange=exchange)
    assert result.converged and expected.converged
    assert result.energy == pytest.approx(expected.energy, abs=3e-9)
    np.testing.assert_allclose(_density(method, result), _density(method, expected), atol=5e-8, rtol=5e-9)
    p = manta.molecular_engine_provenance(result)
    assert p["schema_version"] == 4 and p["coulomb_algorithm"] == "bupo"
    assert p["confirmation_coulomb_algorithm"] == "exact" and p["confirmation_coulomb_build_calls"] >= 1
    assert p["confirmation_coulomb_source_identity"] != p["coulomb_source_identity"]
    if exchange == "exact":
        assert p["confirmation_coulomb_source_identity"] == p["exchange_source_identity"]
    else:
        assert p["confirmation_coulomb_source_identity"] != p["exchange_source_identity"]
    assert p["full_build_confirmed"] and p["stationarity_confirmed"]
    assert p["bupo_near_provider"] == near_backend and p["bupo_order"] == 12
    assert p["bupo_integral_tolerance"] == 1e-11 and p["bupo_maximum_integral_error"] <= 1e-11
    assert p["bupo_build_calls"] > 0 and p["bupo_near_shell_calls"] > 0
    assert len(p["bupo_partition_identity"]) == 64 and p["bupo_setup_wall_seconds"] > 0
    assert p["primitive_evaluations_measured"] == (near_backend == "manta")
    assert "manta_bupo" in p["citation_methods"]
    assert p["terminal_fock_delta"] <= 1e-9 and p["terminal_energy_delta"] <= 1e-9
    assert p["integral_modeled_peak_bytes"] <= manta.ResourceBudget().memory_bytes
    json.dumps(p)
    if method in ("rhf", "uhf"):
        gradient = vq.compute_gradient if method == "rhf" else vq.compute_gradient_uhf
        with pytest.raises(ValueError, match="BUPO gradients"):
            gradient(mol, basis, result)


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
def test_separated_fragments_execute_far_fields_and_confirm_exact_stationarity(method):
    mol, basis = _system(method, separated=True); options = _options(method)
    if method in ("uhf", "uks"):
        # An isolated H2 closed shell plus a singly occupied distant H is a
        # physical initial state; HCORE instead transfers charge between them.
        fragment_mol, fragment_basis = _system("rhf" if method == "uhf" else "rks")
        fragment_method = "rhf" if method == "uhf" else "rks"
        fragment = _run(fragment_method, fragment_mol, fragment_basis, _options(fragment_method))
        assert fragment.converged
        alpha, beta = np.zeros((3, 3)), np.zeros((3, 3))
        alpha[:2, :2] = beta[:2, :2] = .5*fragment.density
        alpha[2, 2] = 1.
        options.initial_guess = vq.InitialGuess.READ
        options.read_density_alpha, options.read_density_beta = alpha, beta
    expected = _run(method, mol, basis, options)
    assert expected.converged
    result = _run(method, mol, basis, options, bupo=_bupo())
    assert result.converged
    p = manta.molecular_engine_provenance(result)
    assert p["bupo_far_blocks"] > 0 and p["bupo_moment_panels"] > 0 and p["bupo_moment_primitive_pairs"] > 0
    assert p["full_build_confirmed"] and p["stationarity_confirmed"]
    assert result.energy == pytest.approx(expected.energy, abs=1e-8)
    np.testing.assert_allclose(_density(method, result), _density(method, expected), atol=5e-8, rtol=5e-9)
    assert p["terminal_commutator_norm"] <= options.conv_tol_grad


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("exchange", ["exact", "ri"])
def test_loose_bupo_noiter_uses_exact_confirmation_and_refuses_false_convergence(method, exchange):
    mol, basis = _system(method); options = _options(method)
    expected = _run(method, mol, basis, options, exchange=exchange); assert expected.converged
    read = getattr(vq, method.upper()+"Options")(options)
    read.initial_guess = vq.InitialGuess.READ; read.max_iter = 0
    if method in ("uhf", "uks"):
        read.read_density_alpha, read.read_density_beta = expected.density_alpha, expected.density_beta
    else:
        read.read_density = expected.density
    bupo = _bupo(loose=True)
    result = _run(method, mol, basis, read, bupo=bupo, exchange=exchange)
    d = _density(method, result); p = manta.molecular_engine_provenance(result)
    direct = manta.DirectJKBuilder(basis).build_J(d)
    approximate = manta.BupoJBuilder(basis, bupo).build(d).J
    delta = direct-approximate
    assert np.max(abs(delta)) > 1e-6
    assert p["terminal_fock_delta"] == pytest.approx(np.max(abs(delta)), abs=3e-11)
    assert p["terminal_energy_delta"] == pytest.approx(abs(.5*np.sum(d*delta)), abs=3e-11)
    assert not result.converged and not p["full_build_confirmed"] and not p["stationarity_confirmed"]
    assert p["confirmation_coulomb_build_calls"] >= 1 and p["bupo_far_blocks"] > 0
    np.testing.assert_allclose(d, _density(method, expected), atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("method", ["rks", "uks"])
@pytest.mark.parametrize("functional,exchange", [("pbe", "exact"), ("hse06", "exact"), ("wb97x", "ri"), ("wb97x-v", "exact")])
def test_bupo_preserves_pure_range_and_vv10_xc_composition(method, functional, exchange):
    mol, basis = _system(method); options = _options(method, functional)
    if functional == "wb97x-v":
        options.grid.n_radial = 10; options.grid.vv10_grid_factor = 3.
    reference = _run(method, mol, basis, options, exchange=exchange)
    result = _run(method, mol, basis, options, bupo=_bupo(), exchange=exchange)
    assert result.converged and reference.converged
    for field in ["energy", "e_coulomb", "e_hf_exchange", "e_xc"]:
        assert getattr(result, field) == pytest.approx(getattr(reference, field), abs=3e-9)
    p, ref = map(manta.molecular_engine_provenance, [result, reference])
    assert p["xc_source_identity"] == ref["xc_source_identity"]
    assert p["range_exchange_source_identity"] == ref["range_exchange_source_identity"]
    if functional == "pbe":
        assert not p["exchange_active"]
    else:
        assert p["range_exchange_build_calls"] > 0
    assert p["confirmation_coulomb_algorithm"] == "exact" and p["full_build_confirmed"]


def test_native_approximate_stationarity_is_not_sufficient_for_exact_confirmation():
    method = "rhf"; mol, basis = _system(method); options = _options(method); bupo_options = _bupo(loose=True)
    bupo = manta.BupoJBuilder(basis, bupo_options); exact = manta.DirectJKBuilder(basis)
    class ApproximateJK(core.JKBuilder):
        def __init__(self):
            super().__init__()
        def build_J(self, d):
            return bupo.build_J(d)
        def build_K(self, d):
            return exact.build_K(d)
        def build_g_rhf(self, d, alpha_hf=1.):
            return self.build_J(d)-.5*alpha_hf*self.build_K(d)
    overlap = vq.compute_overlap(basis); hcore = vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol)
    native = vq.run_rhf_scf_with_jk(basis, mol.n_electrons(), overlap, hcore, mol.nuclear_repulsion(), ApproximateJK(), options)
    assert native.converged
    selected = _run(method, mol, basis, options, bupo=bupo_options)
    p = manta.molecular_engine_provenance(selected)
    assert not selected.converged and not p["full_build_confirmed"]
    assert p["terminal_fock_delta"] > 1e-6 and p["terminal_energy_delta"] > 1e-6
    assert selected.energy == pytest.approx(native.energy, abs=3e-10)


def test_bupo_memory_reserves_other_providers_and_output_records_both_sources():
    mol, basis = _system("rhf"); options = _options("rhf")
    result = _run("rhf", mol, basis, options, bupo=_bupo(), exchange="ri")
    p = manta.molecular_engine_provenance(result)
    budget = manta.ResourceBudget(); budget.memory_bytes = p["integral_modeled_peak_bytes"]
    repeated = _run("rhf", mol, basis, options, bupo=_bupo(), exchange="ri", budget=budget)
    assert repeated.converged
    budget.memory_bytes -= 1
    with pytest.raises(ValueError, match="budget"):
        _run("rhf", mol, basis, options, bupo=_bupo(), exchange="ri", budget=budget)
    from vibeqc.output import OutputChannel
    stream = io.StringIO()
    with OutputChannel.to_stream(stream):
        _run("rhf", mol, basis, options, bupo=_bupo())
    text = stream.getvalue()
    assert "J=bupo (manta)" in text and "Confirmation J: exact" in text and "near provider manta" in text


def test_invalid_backend_and_exchange_combination_refuse_before_scf():
    mol, basis = _system("rhf"); options = _options("rhf")
    with pytest.raises(ValueError, match="MANTA moments"):
        vq.run_rhf(mol, basis, options, integral_engine=vq.IntegralEngineOptions(backend="libint"),
                   coulomb=vq.CoulombBuildOptions("bupo", bupo=_bupo()))
    with pytest.raises(ValueError, match="exact or ri K"):
        vq.run_rhf(mol, basis, options, integral_engine=vq.IntegralEngineOptions(backend="manta"),
                   coulomb=vq.CoulombBuildOptions("bupo"), exchange=vq.ExchangeBuildOptions("bupo"))


@pytest.mark.parametrize("method", ["rhf", "uhf", "rks", "uks"])
@pytest.mark.parametrize("j_algorithm", ["exact", "ri", "bupo"])
@pytest.mark.parametrize("k_algorithm", ["exact", "ri"])
def test_composed_provider_peak_is_an_admissible_budget(method, j_algorithm, k_algorithm):
    mol, basis = _system(method); options = _options(method, "hse06"); options.max_iter = 0
    budget = manta.ResourceBudget(); budget.other_live_bytes = 43210
    coulomb = vq.CoulombBuildOptions(j_algorithm, basis if "ri" in (j_algorithm, k_algorithm) else None, bupo=_bupo())
    def run():
        return getattr(vq, "run_"+method)(mol, basis, options,
            integral_engine=vq.IntegralEngineOptions(backend="manta", budget=budget),
            coulomb=coulomb, exchange=vq.ExchangeBuildOptions(k_algorithm))
    first = run()
    budget.memory_bytes = first.engine_provenance.integral_modeled_peak_bytes
    second = run()
    assert second.energy == first.energy
    budget.memory_bytes -= 1
    with pytest.raises(ValueError, match="memory budget"):
        run()
