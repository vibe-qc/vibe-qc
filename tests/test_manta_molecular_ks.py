"""Selected molecular KS: resolved exchange kernels and final-density sources."""
import io
import json

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc import _vibeqc_core as core


def _system(unrestricted=False):
    positions = [[0., 0., 0.], [.2, -.1, 1.8]] + ([[1.3, .4, .7]] if unrestricted else [])
    mol = vq.Molecule([vq.Atom(1, p) for p in positions], 0, 2 if unrestricted else 1)
    return mol, vq.BasisSet(mol, "sto-3g")


def _options(functional="pbe0", unrestricted=False):
    opts = vq.UKSOptions() if unrestricted else vq.RKSOptions()
    opts.initial_guess, opts.functional = vq.InitialGuess.HCORE, functional
    opts.conv_tol_energy, opts.conv_tol_grad = 1e-12, 1e-9
    opts.stability_check = False
    opts.grid.n_radial, opts.grid.angular, opts.grid.lebedev_order = 20, "lebedev", 11
    return opts


class _IndependentJK(core.JKBuilder):
    """Small libcint tensors, independently factored kernel-specific metrics."""
    def __init__(self, mol, j_kind, k_kind, omega):
        super().__init__()
        gto = pytest.importorskip("pyscf.gto")
        pm = gto.M(atom=[("H", a.xyz) for a in mol.atoms], unit="Bohr", basis="sto-3g",
                   spin=mol.multiplicity-1, verbose=0)
        self.j_kind, self.k_kind, self.omega = j_kind, k_kind, omega
        self.eri = pm.intor("int2e_sph")
        self.fitted = self._fitted(pm)
        if omega:
            with pm.with_range_coulomb(omega):
                self.range_eri = pm.intor("int2e_sph")
                self.range_fitted = self._fitted(pm)

    @staticmethod
    def _fitted(pm):
        metric = pm.intor("int2c2e_sph")
        coupling = pm.intor("int3c2e_sph").transpose(2, 0, 1)
        ev, u = np.linalg.eigh(metric)
        keep = ev > max(1e-12, 1e-10*ev[-1])
        return np.einsum("pr,pij->rij", u[:, keep]/np.sqrt(ev[keep]), coupling)

    def build_J(self, d):
        if self.j_kind == "ri":
            return np.einsum("rij,rkl,kl->ij", self.fitted, self.fitted, d)
        return np.einsum("ijkl,kl->ij", self.eri, d)

    def build_K(self, d):
        return self._exchange(d, self.eri, self.fitted)

    def build_K_erf(self, d, omega):
        assert omega == self.omega
        return self._exchange(d, self.range_eri, self.range_fitted)

    def _exchange(self, d, eri, fitted):
        if self.k_kind == "ri":
            return np.einsum("rij,jk,rlk->il", fitted, d, fitted)
        return np.einsum("ijkl,jl->ik", eri, d)

    def build_g_rhf(self, d, alpha_hf=1.):
        return self.build_J(d)-.5*alpha_hf*self.build_K(d)


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("functional", ["lda", "pbe0", "tpss", "hse06", "wb97x"])
@pytest.mark.parametrize("j_kind,k_kind", [("exact", "exact"), ("ri", "exact"), ("exact", "ri"), ("ri", "ri")])
def test_selected_ks_all_algorithms_against_independent_integrals(unrestricted, functional, j_kind, k_kind):
    mol, basis = _system(unrestricted)
    if functional == "tpss" and not unrestricted:
        # The exactly single-orbital TPSS limit has a numerically sensitive
        # virtual-space potential in libxc. Compare SCF at multiple occupied
        # orbitals; the single-orbital limit is tested at identical D below.
        mol = vq.Molecule([vq.Atom(1, p) for p in
            [[0., 0., 0.], [.2, -.1, 1.8], [1.3, .4, .7], [2., .9, 2.4]]], 0, 1)
        basis = vq.BasisSet(mol, "sto-3g")
    opts = _options(functional, unrestricted)
    f = vq.Functional(functional, 2 if unrestricted else 1)
    driver = vq.run_uks if unrestricted else vq.run_rks
    engine = vq.IntegralEngineOptions(backend="manta", channel_batch=2)
    result = driver(mol, basis, opts, integral_engine=engine,
                    coulomb=vq.CoulombBuildOptions(j_kind, basis if "ri" in (j_kind, k_kind) else None),
                    exchange=vq.ExchangeBuildOptions(k_kind))
    reference = _IndependentJK(mol, j_kind, k_kind, f.rsh_omega)
    s = vq.compute_overlap(basis); h = vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol)
    grid = core.build_grid(mol, opts.grid)
    if unrestricted:
        expected = core.run_uks_scf_with_jk(basis, 2, 1, s, h, mol.nuclear_repulsion(), reference, grid, opts)
        fields = ["density_alpha", "density_beta", "fock_alpha", "fock_beta"]
    else:
        expected = core.run_rks_scf_with_jk(basis, mol.n_electrons(), s, h, mol.nuclear_repulsion(), reference, grid, opts)
        fields = ["density", "fock"]
    assert result.converged and expected.converged
    for field in ["energy", "e_electronic", "e_coulomb", "e_hf_exchange", "e_xc", "e_nuclear"]:
        assert getattr(result, field) == pytest.approx(getattr(expected, field), abs=3e-9)
    for field in fields:
        np.testing.assert_allclose(getattr(result, field), getattr(expected, field), atol=5e-8, rtol=5e-9)
    p = manta.molecular_engine_provenance(result)
    assert p["method"] == ("uks" if unrestricted else "rks")
    assert p["provider_version"] == "md-fp64-v1" and p["channel_batch"] == 2
    assert p["xc_provider"] == "libxc" and p["functional"].lower() == functional
    assert p["xc_grid_points"] == len(grid.weights) and len(p["xc_source_identity"]) == 64
    assert p["full_build_confirmed"] and p["stationarity_confirmed"]
    assert p["terminal_fock_delta"] < 1e-10 and p["terminal_energy_delta"] < 1e-10
    assert p["integral_modeled_peak_bytes"] <= engine.budget.memory_bytes
    assert p["exchange_active"] == f.is_hybrid
    assert p["full_exchange_fraction"] == f.cam_alpha and p["range_exchange_fraction"] == f.cam_beta
    if f.is_range_separated:
        assert p["range_omega"] == f.rsh_omega and p["range_exchange_build_calls"] > result.n_iter
        assert p["range_exchange_shell_calls"] > 0 and len(p["range_exchange_source_identity"]) == 64
        assert p["range_exchange_source_identity"] != p["exchange_source_identity"]
    else:
        assert not p["range_exchange_source_identity"] and p["range_exchange_shell_calls"] == 0
    json.dumps(p)
    assert opts.initial_guess == vq.InitialGuess.HCORE and opts.max_iter > 0


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("factor", [1., 3.])
def test_vv10_uses_same_full_or_coarse_grid_as_existing_driver(unrestricted, factor):
    mol, basis = _system(unrestricted); opts = _options("wb97x-v", unrestricted)
    opts.grid.n_radial = 10; opts.grid.vv10_grid_factor = factor
    driver = vq.run_uks if unrestricted else vq.run_rks
    selected = driver(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    reference = driver(mol, basis, opts)
    assert selected.converged and reference.converged
    for field in ["energy", "e_xc", "e_coulomb", "e_hf_exchange"]:
        assert getattr(selected, field) == pytest.approx(getattr(reference, field), abs=3e-9)
    p = manta.molecular_engine_provenance(selected)
    assert p["full_build_confirmed"]
    assert (p["vv10_grid_points"] > 0) == (factor != 1.)
    assert reference.engine_provenance is None


@pytest.mark.parametrize("unrestricted", [False, True])
def test_explicit_libint_and_same_basis_read(unrestricted):
    mol, basis = _system(unrestricted); opts = _options("pbe0", unrestricted)
    driver = vq.run_uks if unrestricted else vq.run_rks
    first = driver(mol, basis, opts, integral_engine=vq.IntegralEngineOptions())
    assert first.converged and first.engine_provenance.one_electron_provider == "libint"
    opts.initial_guess = vq.InitialGuess.READ
    if unrestricted:
        opts.read_density_alpha, opts.read_density_beta = first.density_alpha, first.density_beta
    else:
        opts.read_density = first.density
    second = driver(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    assert second.converged and second.energy == pytest.approx(first.energy, abs=1e-9)
    assert second.engine_provenance.xc_source_identity == first.engine_provenance.xc_source_identity
    opts.grid.n_radial += 1
    third = driver(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    assert third.engine_provenance.xc_source_identity != second.engine_provenance.xc_source_identity


@pytest.mark.parametrize("unrestricted", [False, True])
def test_selected_ks_gradient_and_converted_reference_cannot_lose_provenance(unrestricted):
    mol, basis = _system(unrestricted); opts = _options("pbe0", unrestricted)
    driver = vq.run_uks if unrestricted else vq.run_rks
    result = driver(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    gradient = core.compute_gradient_uks if unrestricted else core.compute_gradient_rks
    with pytest.raises(ValueError, match="matching integral, XC and grid derivatives"):
        gradient(mol, basis, result, opts.grid)
    if not unrestricted:
        converted = core.rhf_result_from_rks(result)
        assert converted.engine_provenance.xc_source_identity == result.engine_provenance.xc_source_identity
        with pytest.raises(ValueError, match="cannot be differentiated as a Hartree-Fock"):
            vq.compute_gradient(mol, basis, converted)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_zero_iteration_keeps_returned_density_and_does_not_claim_convergence(unrestricted):
    mol, basis = _system(unrestricted); opts = _options("hse06", unrestricted); opts.max_iter = 0
    driver = vq.run_uks if unrestricted else vq.run_rks
    result = driver(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    assert result.n_iter == 0 and not result.converged
    assert result.engine_provenance.full_build_confirmed


@pytest.mark.parametrize("functional,backend,match", [
    ("b2plyp", "manta", "double-hybrid"), ("b2plyp", "libint", "double-hybrid"),
])
def test_unsupported_functional_source_refuses(functional, backend, match):
    mol, basis = _system(); opts = _options(functional)
    with pytest.raises(ValueError, match=match):
        vq.run_rks(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend=backend))


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("functional", ["hse06", "wb97x"])
def test_range_separated_ks_libint_backend_matches_manta_and_independent(unrestricted, functional):
    """The libint backend's erf exchange is its own reference-provider source."""
    mol, basis = _system(unrestricted)
    opts = _options(functional, unrestricted)
    f = vq.Functional(functional, 2 if unrestricted else 1)
    driver = vq.run_uks if unrestricted else vq.run_rks
    results = {backend: driver(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend=backend))
               for backend in ["libint", "manta"]}
    reference = _IndependentJK(mol, "exact", "exact", f.rsh_omega)
    s = vq.compute_overlap(basis); h = vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol)
    grid = core.build_grid(mol, opts.grid)
    if unrestricted:
        expected = core.run_uks_scf_with_jk(basis, 2, 1, s, h, mol.nuclear_repulsion(), reference, grid, opts)
        fields = ["density_alpha", "density_beta", "fock_alpha", "fock_beta"]
    else:
        expected = core.run_rks_scf_with_jk(basis, mol.n_electrons(), s, h, mol.nuclear_repulsion(), reference, grid, opts)
        fields = ["density", "fock"]
    assert expected.converged and all(r.converged for r in results.values())
    for result in results.values():
        for field in ["energy", "e_electronic", "e_coulomb", "e_hf_exchange", "e_xc"]:
            assert getattr(result, field) == pytest.approx(getattr(expected, field), abs=3e-9)
        for field in fields:
            np.testing.assert_allclose(getattr(result, field), getattr(expected, field), atol=5e-8, rtol=5e-9)
    assert results["libint"].energy == pytest.approx(results["manta"].energy, abs=3e-10)
    provenance = {backend: manta.molecular_engine_provenance(r) for backend, r in results.items()}
    assert provenance["libint"]["exchange_provider"] == "libint" and provenance["manta"]["exchange_provider"] == "manta"
    for p in provenance.values():
        assert p["range_exchange_build_calls"] > 0 and p["full_build_confirmed"] and p["stationarity_confirmed"]
        assert p["range_omega"] == pytest.approx(f.rsh_omega)
    assert provenance["libint"]["range_exchange_source_identity"] != provenance["manta"]["range_exchange_source_identity"]
    # The public libint backend stays exact J/K only, range kernels included.
    with pytest.raises(ValueError, match="exact J and K only"):
        driver(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend="libint"),
               coulomb=vq.CoulombBuildOptions("exact", basis), exchange=vq.ExchangeBuildOptions("ri"))


def test_range_budget_accounts_for_simultaneous_kernel_sources():
    mol, basis = _system(); opts = _options("pbe0"); opts.max_iter = 0
    result = vq.run_rks(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    budget = manta.ResourceBudget(); budget.memory_bytes = result.engine_provenance.integral_modeled_peak_bytes
    opts.functional = "hse06"
    with pytest.raises(ValueError, match="memory budget"):
        vq.run_rks(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend="manta", budget=budget))


def test_selected_ks_default_grid_and_output_receipt():
    from vibeqc.output import OutputChannel
    mol, basis = _system(); stream = io.StringIO()
    with OutputChannel.to_stream(stream):
        result = vq.run_rks(mol, basis, grid_level="coarse", integral_engine=vq.IntegralEngineOptions(backend="manta"))
    assert result.converged and result.guess_selection.effective == vq.InitialGuess.HCORE
    assert "XC:" in stream.getvalue() and "libxc" in stream.getvalue()
    assert "exchange fractions: full=0.0" in stream.getvalue()


def test_external_provider_is_refused_before_callback_and_mixed_ranges_refuse():
    calls = []
    def external(features):
        calls.append(features)
        raise AssertionError("unadmitted external XC callback ran")
    vq.define_external_functional("manta-selected-ks-external-test", external)
    mol, basis = _system(); opts = _options("manta-selected-ks-external-test")
    with pytest.raises(ValueError, match="libxc functional"):
        vq.run_rks(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    assert not calls
    vq.define_functional("manta-selected-ks-mixed-test", [("HYB_GGA_XC_HSE06", .5), ("HYB_GGA_XC_WB97X", .5)])
    opts.functional = "manta-selected-ks-mixed-test"
    with pytest.raises(ValueError, match="range|CAM"):
        vq.run_rks(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend="manta"))


def test_alias_recipe_is_part_of_quadrature_source_identity():
    name = "manta-selected-ks-alias-test"
    mol, basis = _system(); opts = _options(name); opts.max_iter = 0
    engine = vq.IntegralEngineOptions(backend="manta")
    vq.define_functional(name, [("GGA_X_PBE", .75), ("GGA_C_PBE", 1.)], hf_exchange_fraction=.25)
    first = vq.run_rks(mol, basis, opts, integral_engine=engine)
    vq.define_functional(name, [("GGA_X_PBE", .8), ("GGA_C_PBE", 1.)], hf_exchange_fraction=.2)
    second = vq.run_rks(mol, basis, opts, integral_engine=engine)
    assert first.engine_provenance.xc_source_identity != second.engine_provenance.xc_source_identity
    assert first.engine_provenance.full_exchange_fraction == .25
    assert second.engine_provenance.full_exchange_fraction == .2


def test_single_orbital_tpss_potential_matches_at_identical_density():
    mol, basis = _system(); opts = _options("tpss")
    engine = vq.IntegralEngineOptions(backend="manta")
    first = vq.run_rks(mol, basis, opts, integral_engine=engine)
    opts.initial_guess, opts.max_iter, opts.read_density = vq.InitialGuess.READ, 0, first.density
    selected = vq.run_rks(mol, basis, opts, integral_engine=engine)
    reference = _IndependentJK(mol, "exact", "exact", 0.)
    s = vq.compute_overlap(basis); h = vq.compute_kinetic(basis)+vq.compute_nuclear(basis, mol)
    expected = core.run_rks_scf_with_jk(basis, 2, s, h, mol.nuclear_repulsion(), reference,
                                      core.build_grid(mol, opts.grid), opts, first.density)
    np.testing.assert_array_equal(selected.density, expected.density)
    np.testing.assert_allclose(selected.fock, expected.fock, atol=1e-12, rtol=1e-12)
    assert selected.e_xc == pytest.approx(expected.e_xc, abs=1e-12)


def test_pattern_hold_on_selected_uks_matches_the_legacy_route():
    """Deeply stretched H2, PBE: a held [+1, -1] seed keeps the localized state on both routes."""
    mol = vq.Molecule([vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [5., 0., 0.])], 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.UKSOptions()
    opts.functional, opts.initial_guess, opts.atomic_spins = "pbe", vq.InitialGuess.SAD, [1, -1]
    opts.conv_tol_energy, opts.conv_tol_grad, opts.max_iter = 1e-10, 1e-7, 200
    opts.stability_check = False
    opts.grid.n_radial, opts.grid.angular, opts.grid.lebedev_order = 30, "lebedev", 15
    opts.spinlock_mode, opts.spinlock_iterations = vq.SpinlockMode.PATTERN_HOLD, 10
    selected = vq.run_uks(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    legacy = vq.run_uks(mol, basis, opts)
    assert selected.converged and legacy.converged
    assert selected.s_squared > .5 and selected.s_squared == pytest.approx(legacy.s_squared, abs=1e-7)
    assert selected.energy == pytest.approx(legacy.energy, abs=1e-8)
    assert all(it.diis_subspace == 0 for it in selected.scf_trace[:10])
    p = manta.molecular_engine_provenance(selected)
    assert p["full_build_confirmed"] and p["stationarity_confirmed"] and p["initial_guess_effective"] == "SAD"
    seeds = vq.UKSOptions(opts); seeds.multi_guess_seeds = [3]
    chosen = vq.run_uks(mol, basis, seeds, integral_engine=vq.IntegralEngineOptions(backend="manta"))
    assert chosen.converged and chosen.energy <= selected.energy + 1e-8
    assert chosen.engine_provenance is not None and chosen.engine_provenance.stationarity_confirmed
