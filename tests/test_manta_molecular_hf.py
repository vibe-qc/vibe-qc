"""Opt-in molecular engines, separate J/K choices and final-density evidence."""
import io
import json

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc import _vibeqc_core as core


def _system(unrestricted=False):
    positions = [[0., 0., 0.], [.2, -.1, 1.8]]+([[1.3, .4, .7]] if unrestricted else [])
    mol = vq.Molecule([vq.Atom(1, p) for p in positions], 0, 2 if unrestricted else 1)
    return mol, vq.BasisSet(mol, "sto-3g")


def _options(unrestricted=False):
    options = vq.UHFOptions() if unrestricted else vq.RHFOptions()
    options.initial_guess = vq.InitialGuess.HCORE
    options.conv_tol_energy, options.conv_tol_grad = 1e-12, 1e-10
    options.stability_check = False
    return options


class _ReferenceJK(core.JKBuilder):
    def __init__(self, basis, auxiliary, j_kind, k_kind):
        super().__init__()
        self.eri = np.asarray(vq.compute_eri(basis))
        self.df = vq.make_df_jk_builder(basis, auxiliary)
        self.j_kind, self.k_kind = j_kind, k_kind

    def build_J(self, d):
        return self.df.build_J(d) if self.j_kind == "ri" else np.einsum("ijkl,kl->ij", self.eri, d)

    def build_K(self, d):
        return self.df.build_K(d) if self.k_kind == "ri" else np.einsum("ijkl,jl->ik", self.eri, d)

    def build_g_rhf(self, d, alpha_hf=1.):
        return self.build_J(d)-.5*alpha_hf*self.build_K(d)


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("j_kind,k_kind", [("exact", "exact"), ("ri", "ri"), ("ri", "exact"), ("exact", "ri")])
def test_selected_molecular_hf_against_independent_same_algorithm(unrestricted, j_kind, k_kind):
    mol, basis = _system(unrestricted); auxiliary = vq.BasisSet(mol, "sto-3g")
    options = _options(unrestricted)
    engine = vq.IntegralEngineOptions(backend="manta")
    coulomb = vq.CoulombBuildOptions(j_kind, auxiliary if "ri" in (j_kind, k_kind) else None)
    exchange = vq.ExchangeBuildOptions(k_kind)
    driver = vq.run_uhf if unrestricted else vq.run_rhf
    result = driver(mol, basis, options, integral_engine=engine, coulomb=coulomb, exchange=exchange)
    s = vq.compute_overlap(basis); h = vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol)
    reference = _ReferenceJK(basis, auxiliary, j_kind, k_kind)
    if unrestricted:
        expected = vq.run_uhf_scf_with_jk(basis, 2, 1, s, h, mol.nuclear_repulsion(), reference, options)
        fields = ["density_alpha", "density_beta", "fock_alpha", "fock_beta"]
    else:
        expected = vq.run_rhf_scf_with_jk(basis, 2, s, h, mol.nuclear_repulsion(), reference, options)
        fields = ["density", "fock"]
    assert result.converged and expected.converged
    assert result.energy == pytest.approx(expected.energy, abs=3e-10)
    for name in fields:
        np.testing.assert_allclose(getattr(result, name), getattr(expected, name), atol=3e-9, rtol=3e-9)
    p = manta.molecular_engine_provenance(result)
    assert p["requested_backend"] == p["one_electron_provider"] == p["coulomb_provider"] == p["exchange_provider"] == "manta"
    assert (p["coulomb_algorithm"], p["exchange_algorithm"]) == (j_kind, k_kind)
    assert p["full_build_confirmed"] and p["stationarity_confirmed"] and p["primitive_evaluations_measured"]
    assert p["terminal_energy_delta"] < 1e-11 and p["terminal_fock_delta"] < 1e-11
    assert p["terminal_commutator_norm"] <= options.conv_tol_grad
    assert p["jk_build_calls"] > result.n_iter and p["jk_shell_calls"] > 0
    assert p["one_electron_shell_calls"] == 3*basis.nshells**2
    assert p["primitive_integrals_evaluated"] > 0
    assert p["integral_modeled_peak_bytes"] <= engine.budget.memory_bytes
    assert p["one_electron_wall_seconds"] > 0 and p["jk_wall_seconds"] > 0
    assert not p["fallback_reason"] and not p["incremental"] and p["screening"] == "none"
    assert len(p["one_electron_source_identity"]) == 64
    assert (p["coulomb_source_identity"] == p["exchange_source_identity"]) == (j_kind == k_kind)
    assert result.ecp_provenance_verified and not result.ecp_operator_applied
    assert not options.density_fit and options.scf_mode == vq.SCFMode.AUTO
    json.dumps(p)
    gradient = vq.compute_gradient_uhf if unrestricted else vq.compute_gradient
    if "ri" in (j_kind, k_kind):
        selected_gradient = gradient(mol, basis, result)
        assert np.isfinite(selected_gradient).all()
        np.testing.assert_allclose(selected_gradient.sum(axis=0), 0., atol=3e-12)
    else:
        plain = driver(mol, basis, options)
        assert plain.engine_provenance is None
        np.testing.assert_allclose(gradient(mol, basis, result), gradient(mol, basis, plain), atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_explicit_libint_reference_default_guess_read_and_owned_provenance(unrestricted):
    mol, basis = _system(unrestricted); driver = vq.run_uhf if unrestricted else vq.run_rhf
    first = driver(mol, basis, integral_engine=vq.IntegralEngineOptions(backend="libint"))
    assert first.converged and first.guess_selection.effective == vq.InitialGuess.HCORE
    assert first.engine_provenance.one_electron_provider == "libint"
    assert not first.engine_provenance.primitive_evaluations_measured
    assert first.engine_provenance.primitive_integrals_evaluated == 0
    options = _options(unrestricted); options.initial_guess = vq.InitialGuess.READ
    if unrestricted:
        options.read_density_alpha, options.read_density_beta = first.density_alpha, first.density_beta
    else:
        options.read_density = first.density
    second = driver(mol, basis, options, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    assert second.converged and second.energy == pytest.approx(first.energy, abs=1e-9)
    p = manta.molecular_engine_provenance(second)
    p["one_electron_provider"] = "changed"
    assert second.engine_provenance.one_electron_provider == "manta"
    with pytest.raises(AttributeError):
        second.engine_provenance.full_build_confirmed = False


def test_selected_engine_output_uses_executed_receipt():
    from vibeqc.output import OutputChannel
    mol, basis = _system(); stream = io.StringIO()
    with OutputChannel.to_stream(stream):
        result = vq.run_rhf(mol, basis, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    assert "Integral engine: manta; J=exact (manta); K=exact (manta)" in stream.getvalue()
    assert f"J/K shell calls: {result.engine_provenance.jk_shell_calls}" in stream.getvalue()
    assert "manta_md" in manta.molecular_engine_provenance(result)["citation_methods"]


@pytest.mark.parametrize("field,value,match", [("initial_guess", vq.InitialGuess.AUTO, "HCORE"),
    ("read_path", "restart.molden", "restart file"), ("density_fit", True, "legacy density_fit"),
    ("aux_basis", "sto-3g", "legacy density_fit"), ("cosx", True, "COSX"),
    ("ecp_library", "example-ecp", "all-electron"), ("scf_mode", vq.SCFMode.CONVENTIONAL, "AO4"),
    ("conv_tol_grad", np.nan, "convergence tolerances")])
def test_unsupported_or_conflicting_method_options_refuse(field, value, match):
    mol, basis = _system(); options = _options(); setattr(options, field, value)
    with pytest.raises(ValueError, match=match):
        vq.run_rhf(mol, basis, options, integral_engine=vq.IntegralEngineOptions(backend="manta"))


@pytest.mark.parametrize("keywords,exception", [
    ({"integral_engine": "manta"}, TypeError),
    ({"integral_engine": vq.IntegralEngineOptions(backend="unknown")}, ValueError),
    ({"integral_engine": vq.IntegralEngineOptions(fallback="libint_same_source")}, NotImplementedError),
    ({"integral_engine": vq.IntegralEngineOptions(channel_batch=True)}, TypeError),
    ({"integral_engine": vq.IntegralEngineOptions(backend="libint", channel_batch=2)}, ValueError),
    ({"coulomb": vq.CoulombBuildOptions("bupo")}, ValueError),
    ({"exchange": vq.ExchangeBuildOptions("none")}, ValueError),
    ({"coulomb": vq.CoulombBuildOptions("ri")}, ValueError),
    ({"coulomb": vq.CoulombBuildOptions("exact", "sto-3g")}, ValueError),
])
def test_configuration_is_explicit_and_strict(keywords, exception):
    mol, basis = _system()
    with pytest.raises(exception):
        vq.run_rhf(mol, basis, **keywords)


def test_selected_budget_and_zero_iteration_have_no_false_convergence():
    mol, basis = _system(); budget = manta.ResourceBudget(); budget.memory_bytes = 1
    with pytest.raises(ValueError, match="memory budget"):
        vq.run_rhf(mol, basis, integral_engine=vq.IntegralEngineOptions(backend="manta", budget=budget))
    options = _options(); options.max_iter = 0
    result = vq.run_rhf(mol, basis, options, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    assert not result.converged and result.n_iter == 0
    assert result.engine_provenance.full_build_confirmed
    with pytest.raises((ValueError, RuntimeError)):
        vq.compute_gradient(mol, basis, result)


def test_selected_gradient_preserves_geometry_and_algorithm_source():
    mol, basis = _system()
    result = vq.run_rhf(mol, basis, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    changed = vq.BasisSet(mol, "6-31g")
    with pytest.raises(ValueError, match="differs from the SCF source"):
        vq.compute_gradient(mol, changed, result)
    options = vq.GradientOptions(); options.density_fit = True; options.aux_basis = "sto-3g"
    with pytest.raises(ValueError, match="conflicts with derivative options"):
        vq.compute_gradient(mol, basis, result, options)


def test_native_selected_ri_also_refuses_incomplete_auxiliary_coverage():
    mol, basis = _system()
    partial = vq.BasisSet(mol, basis.shells()[:1], "partial-auxiliary", True)
    engine = core._manta._MolecularHFEngine(); engine.coulomb = "ri"
    with pytest.raises(ValueError, match="incomplete atom coverage"):
        core._manta._run_rhf(mol, basis, _options(), engine, partial)


def test_selected_gradient_cannot_add_an_unselected_hubbard_source():
    mol, basis = _system()
    result = vq.run_rhf(mol, basis, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    with pytest.raises(ValueError, match="no matching DFT\\+U"):
        vq.compute_gradient(mol, basis, result, dft_plus_u=[object()])


def test_selected_exact_gradient_retains_original_resource_budget():
    mol, basis = _system()
    plan = manta.plan_hf_lagrangian_gradient(mol, basis)
    budget = manta.ResourceBudget(); budget.shell_calls = plan.shell_calls-1
    result = vq.run_rhf(mol, basis, _options(),
        integral_engine=vq.IntegralEngineOptions(backend="manta", budget=budget))
    assert result.converged
    budget.shell_calls = 10000000
    with pytest.raises(ValueError, match="budget"):
        vq.compute_gradient(mol, basis, result)


@pytest.mark.parametrize("field,value", [("atomic_spins", [1, -1, 1]),
    ("spinlock_mode", vq.SpinlockMode.PATTERN_HOLD), ("multi_guess_seeds", [7])])
def test_selected_open_shell_schedule_must_be_explicitly_qualified(field, value):
    mol, basis = _system(True); options = _options(True); setattr(options, field, value)
    with pytest.raises(ValueError, match="spin"):
        vq.run_uhf(mol, basis, options, integral_engine=vq.IntegralEngineOptions(backend="manta"))
