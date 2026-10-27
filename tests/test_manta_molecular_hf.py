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


@pytest.mark.parametrize("field,value,match", [("initial_guess", vq.InitialGuess.FRAGMO, "FRAGMO"),
    ("read_path", "restart.molden", "restart file"), ("density_fit", True, "legacy density_fit"),
    ("aux_basis", "sto-3g", "legacy density_fit"), ("cosx", True, "COSX"),
    ("scf_mode", vq.SCFMode.CONVENTIONAL, "AO4"),
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


@pytest.mark.parametrize("method,algorithm", [("rhf", "exact"), ("uhf", "exact"), ("rhf", "ri")])
def test_selected_hubbard_gradient_matches_legacy_and_finite_differences(method, algorithm):
    import numpy as np
    from tests.test_manta_bupo_scf import _system as _bupo_system, _options as _bupo_options
    mol, basis = _bupo_system(method)
    opts = _bupo_options(method)
    site = vq.HubbardSite(atom_index=0, l=0, U_ev=4.0)
    engine = vq.IntegralEngineOptions(backend="manta")
    coulomb = vq.CoulombBuildOptions(algorithm, "sto-3g" if algorithm == "ri" else None)
    run = getattr(vq, "run_"+method)
    gradient = vq.compute_gradient if method == "rhf" else vq.compute_gradient_uhf
    result = run(mol, basis, opts, dft_plus_u=[site], integral_engine=engine, coulomb=coulomb)
    assert result.converged and abs(result.e_dft_plus_u) > 1e-4
    values = np.asarray(gradient(mol, basis, result, dft_plus_u=[site]))
    assert values.shape == (len(mol.atoms), 3)
    np.testing.assert_allclose(values.sum(axis=0), 0., atol=1e-8)
    if algorithm == "exact":
        legacy = run(mol, basis, opts, dft_plus_u=[site])
        expected = np.asarray(gradient(mol, basis, legacy, dft_plus_u=[site]))
        np.testing.assert_allclose(values, expected, atol=1e-7)
        without = np.asarray(gradient(mol, basis, result))
        assert np.abs(values-without).max() > 1e-5
    # The Hubbard term is differentiated on the selected energy itself.
    h, atom, axis = 1e-3, 1, 2
    energies = []
    for sign in (1, -1):
        positions = [list(a.xyz) for a in mol.atoms]
        positions[atom][axis] += sign*h
        moved = vq.Molecule([vq.Atom(a.Z, p) for a, p in zip(mol.atoms, positions)], mol.charge, mol.multiplicity)
        energies.append(run(moved, vq.BasisSet(moved, "sto-3g"), opts, dft_plus_u=[site],
                            integral_engine=engine, coulomb=coulomb).energy)
    assert values[atom, axis] == pytest.approx((energies[0]-energies[1])/(2*h), abs=5e-6)


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


def test_multi_guess_selection_runs_one_native_scf_per_seed_and_matches_legacy(monkeypatch):
    """Seeds are composed in Python over restarted selected runs; the native driver never sees them."""
    mol, basis = _system(True); options = _options(True); options.multi_guess_seeds = [7, 11]
    engine = vq.IntegralEngineOptions(backend="manta")
    native = core._manta._run_uhf
    calls = []

    def counted(molecule, basis_, opts, *rest):
        assert list(opts.multi_guess_seeds) == []
        calls.append((opts.restart_opts.enabled, opts.restart_opts.seed))
        return native(molecule, basis_, opts, *rest)
    monkeypatch.setattr(core._manta, "_run_uhf", counted)
    result = vq.run_uhf(mol, basis, options, integral_engine=engine)
    assert len(calls) == 3 and calls[1:] == [(True, 7), (True, 11)]
    legacy = vq.run_uhf(mol, basis, options)
    assert result.converged and legacy.converged
    assert result.energy == pytest.approx(legacy.energy, abs=3e-10)
    assert result.engine_provenance is not None and result.engine_provenance.stationarity_confirmed
    assert list(options.multi_guess_seeds) == [7, 11]
    with pytest.raises(ValueError, match="one seed"):
        seeded = vq.UHFOptions(options)
        native(mol, basis, seeded, core._manta._MolecularHFEngine(), None)


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("guess", ["sad", "sap", "patom", "auto"])
def test_shared_guesses_reach_the_hcore_basin_and_are_recorded(unrestricted, guess):
    """Shared starting guesses converge the selected route to the same state as HCORE."""
    mol, basis = _system(unrestricted)
    driver = vq.run_uhf if unrestricted else vq.run_rhf
    engine = vq.IntegralEngineOptions(backend="manta")
    baseline = driver(mol, basis, _options(unrestricted), integral_engine=engine)
    options = _options(unrestricted)
    options.initial_guess = getattr(vq.InitialGuess, guess.upper())
    result = driver(mol, basis, options, integral_engine=engine)
    assert result.converged and baseline.converged
    assert result.energy == pytest.approx(baseline.energy, abs=1e-9)
    p = manta.molecular_engine_provenance(result)
    assert p["initial_guess_requested"] == guess.upper()
    assert p["initial_guess_effective"] in {"SAD", "SAP", "PATOM", "HCORE"}
    if guess != "auto":
        assert p["initial_guess_effective"] == guess.upper()
    assert p["one_electron_provider"] == p["coulomb_provider"] == "manta"
    assert p["full_build_confirmed"] and p["stationarity_confirmed"]
    assert manta.molecular_engine_provenance(baseline)["initial_guess_effective"] == "HCORE"
    fragmo = _options(unrestricted); fragmo.initial_guess = vq.InitialGuess.FRAGMO
    with pytest.raises(ValueError, match="FRAGMO"):
        driver(mol, basis, fragmo, integral_engine=engine)


@pytest.mark.parametrize("backend", ["manta", "libint"])
def test_atomic_spin_seeds_reach_the_legacy_broken_symmetry_state(backend):
    """Stretched H2 singlet: a [+1, -1] SAD seed lands on the legacy broken-symmetry minimum."""
    mol = vq.Molecule([vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 3.])], 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    engine = vq.IntegralEngineOptions(backend=backend)

    def options(seed):
        opts = _options(True); opts.initial_guess = vq.InitialGuess.SAD
        if seed is not None:
            opts.atomic_spins = seed
        return opts

    broken = vq.run_uhf(mol, basis, options([1, -1]), integral_engine=engine)
    legacy = vq.run_uhf(mol, basis, options([1, -1]))
    symmetric = vq.run_uhf(mol, basis, options(None), integral_engine=engine)
    assert broken.converged and legacy.converged and symmetric.converged
    assert broken.energy == pytest.approx(legacy.energy, abs=3e-10)
    assert broken.energy == pytest.approx(-0.95101800, abs=1e-6)
    assert broken.s_squared == pytest.approx(legacy.s_squared, abs=1e-8)
    assert broken.s_squared == pytest.approx(0.7742, abs=1e-2)
    # Without the seed the stability-off SCF stays on the spin-symmetric saddle.
    assert symmetric.energy == pytest.approx(-0.88527500, abs=1e-6) and abs(symmetric.s_squared) < 1e-3
    np.testing.assert_allclose(broken.density_alpha, legacy.density_alpha, atol=3e-9, rtol=3e-9)
    p = manta.molecular_engine_provenance(broken)
    assert p["initial_guess_effective"] == "SAD" and p["full_build_confirmed"] and p["stationarity_confirmed"]
    assert p["coulomb_provider"] == backend
    flipped = vq.run_uhf(mol, basis, options([-1, 1]), integral_engine=engine)
    assert flipped.energy == pytest.approx(broken.energy, abs=1e-9)
    with pytest.raises((ValueError, RuntimeError), match="atomic_spins"):
        vq.run_uhf(mol, basis, options([1, -1, 0]), integral_engine=engine)


def test_spin_locking_on_selected_uhf_matches_the_legacy_route():
    """PATTERN_HOLD protects a seeded state; SPIN_SCHEDULE locks then releases to the target."""
    engine = vq.IntegralEngineOptions(backend="manta")
    stretched = vq.Molecule([vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 3.])], 0, 1)
    basis = vq.BasisSet(stretched, "sto-3g")
    held = _options(True); held.initial_guess = vq.InitialGuess.SAD; held.atomic_spins = [1, -1]
    held.spinlock_mode, held.spinlock_iterations = vq.SpinlockMode.PATTERN_HOLD, 8
    free = _options(True); free.initial_guess = vq.InitialGuess.SAD; free.atomic_spins = [1, -1]
    selected_held = vq.run_uhf(stretched, basis, held, integral_engine=engine)
    legacy_held = vq.run_uhf(stretched, basis, held)
    selected_free = vq.run_uhf(stretched, basis, free, integral_engine=engine)
    assert selected_held.converged and legacy_held.converged and selected_free.converged
    assert selected_held.energy == pytest.approx(legacy_held.energy, abs=3e-10)
    assert selected_held.energy == pytest.approx(selected_free.energy, abs=1e-8)
    assert selected_held.s_squared == pytest.approx(legacy_held.s_squared, abs=1e-8)
    p = manta.molecular_engine_provenance(selected_held)
    assert p["full_build_confirmed"] and p["stationarity_confirmed"] and p["initial_guess_effective"] == "SAD"

    equilibrium = vq.Molecule([vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4])], 0, 1)
    basis = vq.BasisSet(equilibrium, "sto-3g")
    plain = _options(True); plain.initial_guess = vq.InitialGuess.SAD
    schedule = _options(True); schedule.initial_guess = vq.InitialGuess.SAD
    schedule.spinlock_mode, schedule.spinlock_value, schedule.spinlock_iterations = vq.SpinlockMode.SPIN_SCHEDULE, 2, 5
    selected_plain = vq.run_uhf(equilibrium, basis, plain, integral_engine=engine)
    selected_schedule = vq.run_uhf(equilibrium, basis, schedule, integral_engine=engine)
    legacy_schedule = vq.run_uhf(equilibrium, basis, schedule)
    assert selected_schedule.converged and legacy_schedule.converged
    assert selected_schedule.energy == pytest.approx(selected_plain.energy, abs=1e-7)
    assert selected_schedule.energy == pytest.approx(legacy_schedule.energy, abs=3e-10)
    assert abs(selected_schedule.s_squared) < 1e-3
    p = manta.molecular_engine_provenance(selected_schedule)
    assert p["full_build_confirmed"] and p["stationarity_confirmed"]
    assert p["jk_build_calls"] > manta.molecular_engine_provenance(selected_plain)["jk_build_calls"]
    schedule.spinlock_value = 1
    with pytest.raises(RuntimeError, match="incompatible"):
        vq.run_uhf(equilibrium, basis, schedule, integral_engine=engine)


_WATER = [(8, [0., 0., 0.]), (1, [0., 1.4315, 1.107153]), (1, [0., -1.4315, 1.107153])]


def test_fragment_guess_and_restart_projection_on_the_selected_route():
    """FRAGMO and read_from are prepared into a same-basis READ density before the selected SCF."""
    atoms = [vq.Atom(z, list(xyz)) for z, xyz in _WATER]
    atoms += [vq.Atom(z, [x, y, z_ + 6.]) for z, (x, y, z_) in _WATER]
    dimer = vq.Molecule(atoms, 0, 1); basis = vq.BasisSet(dimer, "sto-3g")
    engine = vq.IntegralEngineOptions(backend="manta")
    frags = [vq.Fragment(atoms=[0, 1, 2]), vq.Fragment(atoms=[3, 4, 5])]
    opts = _options(); opts.initial_guess = vq.InitialGuess.FRAGMO
    selected = vq.run_rhf(dimer, basis, opts, integral_engine=engine, fragments=frags)
    # The selected route prepares its own copy; the caller's options stay untouched.
    assert opts.initial_guess == vq.InitialGuess.FRAGMO and np.asarray(opts.read_density).size == 0
    with pytest.raises(ValueError, match="requires fragments"):
        vq.run_rhf(dimer, basis, opts, integral_engine=engine)
    legacy = vq.run_rhf(dimer, basis, opts, fragments=frags)
    assert selected.converged and legacy.converged
    assert selected.energy == pytest.approx(legacy.energy, abs=3e-10)
    p = manta.molecular_engine_provenance(selected)
    assert p["initial_guess_requested"] == "FRAGMO" and p["initial_guess_effective"] == "FRAGMO"
    assert p["one_electron_provider"] == p["coulomb_provider"] == "manta"
    assert p["full_build_confirmed"] and p["stationarity_confirmed"]

    mol, basis = _system(True)
    first = vq.run_uhf(mol, basis, _options(True), integral_engine=engine)
    restart = _options(True); restart.initial_guess = vq.InitialGuess.READ
    projected = vq.run_uhf(mol, basis, restart, integral_engine=engine, read_from=first)
    assert projected.converged and projected.energy == pytest.approx(first.energy, abs=1e-9)
    assert projected.n_iter < first.n_iter
    p = manta.molecular_engine_provenance(projected)
    assert p["initial_guess_effective"] == "READ" and p["stationarity_confirmed"]


@pytest.mark.parametrize("backend", ["manta", "libint"])
def test_xml_library_ecp_on_the_selected_route_matches_legacy(backend):
    """Zn2+ / 6-31g with the ecp10mdf small core: same valence SCF on both routes."""
    mol = vq.Molecule([vq.Atom(30, [0., 0., 0.])], 2, 1)
    basis = vq.BasisSet(mol, "6-31g")
    opts = _options(); opts.initial_guess = vq.InitialGuess.SAD; opts.max_iter = 200
    opts.ecp_centers = [vq.ECPCenter(Z=30, xyz=[0., 0., 0.])]; opts.ecp_library = "ecp10mdf"
    selected = vq.run_rhf(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend=backend))
    legacy = vq.run_rhf(mol, basis, opts)
    assert selected.converged and legacy.converged
    assert selected.energy == pytest.approx(legacy.energy, abs=3e-10)
    np.testing.assert_allclose(selected.density, legacy.density, atol=3e-9, rtol=3e-9)
    assert selected.ecp_operator_applied and selected.ecp_total_ncore == legacy.ecp_total_ncore == 10
    assert selected.ecp_xml_library == "ecp10mdf" and len(selected.ecp_xml_centers) == 1
    p = manta.molecular_engine_provenance(selected)
    assert p["ecp_route"] == "xml-library" and p["ecp_total_ncore"] == 10 and p["schema_version"] == 6
    assert p["one_electron_provider"] == backend and p["full_build_confirmed"] and p["stationarity_confirmed"]
    mixed = vq.RHFOptions(opts); mixed.ecp_effective_charges = [28.]; mixed.ecp_total_ncore = 2
    with pytest.raises(ValueError):
        vq.run_rhf(mol, basis, mixed, integral_engine=vq.IntegralEngineOptions(backend=backend))


def test_inline_sidecar_ecp_on_the_selected_route_matches_legacy_energy_and_gradient():
    """AgCl / def2-svp: the basis sidecar's inline ECP on the selected route."""
    mol = vq.Molecule([vq.Atom(47, [0., 0., 0.]), vq.Atom(17, [0., 0., 4.3101])], 0, 1)
    basis = vq.BasisSet(mol, "def2-svp")
    opts = _options(); opts.initial_guess = vq.InitialGuess.SAD; opts.max_iter = 200
    # Silver's def2-svp primitives exceed the default work caps; state the limits explicitly.
    budget = manta.ResourceBudget(); budget.memory_bytes = 2 << 30
    budget.kernel_terms = budget.scalar_terms = budget.primitive_quartets = budget.shell_calls = 1 << 50
    selected = vq.run_rhf(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend="manta", budget=budget))
    legacy = vq.run_rhf(mol, basis, opts)
    assert selected.converged and legacy.converged
    assert selected.ecp_operator_applied and selected.ecp_total_ncore == legacy.ecp_total_ncore > 0
    assert selected.energy == pytest.approx(legacy.energy, abs=3e-10)
    p = manta.molecular_engine_provenance(selected)
    assert p["ecp_route"] == "inline-primitives" and p["ecp_total_ncore"] == legacy.ecp_total_ncore
    assert p["one_electron_provider"] == "manta" and p["stationarity_confirmed"]
    # ECP results take the ordinary native gradient path, which carries the ECP derivative.
    def gradient_options(result):
        go = vq.GradientOptions()
        go.ecp_primitive_blocks = list(result.ecp_primitive_blocks)
        go.ecp_primitive_centers = [list(c) for c in result.ecp_primitive_centers]
        go.ecp_effective_charges = list(result.ecp_effective_charges)
        go.ecp_total_ncore = int(result.ecp_total_ncore)
        return go
    assert len(selected.ecp_primitive_blocks) == len(legacy.ecp_primitive_blocks) > 0
    np.testing.assert_allclose(vq.compute_gradient(mol, basis, selected, gradient_options(selected)),
                               vq.compute_gradient(mol, basis, legacy, gradient_options(legacy)), atol=1e-7, rtol=1e-7)
