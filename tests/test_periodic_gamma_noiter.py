"""max_iter=0 (ORCA NOITER) on the exact-Gamma drivers (#649).

Every Gamma Ewald driver evaluates the starting guess once: the energy is
exactly the first trace entry of an iterating run from the same guess, the
result is explicitly unconverged with no iteration, and a negative budget is
refused. The RHF/RKS/UHF/UKS 3-D and the RHF 2-D slab cases live beside their
drivers' other tests; this file covers the native driver, ROHF, the RKS/UKS
slabs, GPW and the output records. Restricted and unrestricted multi-k Ewald tests additionally
cover pure/hybrid DFT, finite temperature and explicit restart densities.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc as vq
import vibeqc._vibeqc_core as core
from vibeqc import periodic_gapw_j as gpw_closed
from vibeqc import periodic_gapw_open_shell as gpw_open
from vibeqc.output.formats.scf_log import INITIAL_GUESS_ONLY, format_scf_trace

LIH = vq.PeriodicSystem(3, 12.0 * np.eye(3), [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [0, 0, 3.0])])
H3 = vq.PeriodicSystem(3, 12.0 * np.eye(3), [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4]),
                                             vq.Atom(1, [0, 1.6, 0.7])], 0, 2)
HE_SLAB = vq.PeriodicSystem(2, np.diag([18., 18., 40.]), [vq.Atom(2, [0., 0., 1.3])])
H2_SLAB = vq.PeriodicSystem(2, np.diag([18., 18., 40.]),
                            [vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4])])


def _basis(system):
    return vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def _options(ks, max_iter):
    opts = vq.PeriodicKSOptions() if ks else vq.PeriodicRHFOptions()
    if ks:
        opts.functional = "lda"
    opts.max_iter, opts.damping, opts.use_diis = max_iter, 0.3, True
    return opts


def _field(result, name):
    return result[name] if isinstance(result, dict) else getattr(result, name)


def _first_energy(trace):
    step = list(trace)[0]
    return step["energy"] if isinstance(step, dict) else step.energy


def _assert_noiter_is_the_guess(run):
    zero = run(0)
    assert _field(zero, "converged") is False
    assert _field(zero, "n_iter") == 0
    assert list(_field(zero, "scf_trace") or []) == []
    assert _field(zero, "energy") == _first_energy(_field(run(2), "scf_trace"))
    return zero


def test_native_gamma_rhf_evaluates_the_guess_instead_of_returning_zero():
    basis = _basis(LIH)
    zero = _assert_noiter_is_the_guess(
        lambda n: core.run_rhf_periodic_gamma(LIH, basis, _options(False, n)))
    assert zero.energy < -18.0
    with pytest.raises(ValueError, match="max_iter must be >= 0"):
        core.run_rhf_periodic_gamma(LIH, basis, _options(False, -1))


@pytest.mark.parametrize("name, system, ks", [
    ("run_rohf_periodic_gamma_ewald3d", H3, False),
    ("run_rks_periodic_gamma_ewald2d", HE_SLAB, True),
    ("run_uks_periodic_gamma_ewald2d", H2_SLAB, True),
])
def test_gamma_ewald_driver_evaluates_the_guess(name, system, ks):
    driver = getattr(vq, name)
    basis = _basis(system)
    accepted = inspect.signature(driver).parameters
    kwargs = {k: v for k, v in (("auto_optimize_truncation", False), ("verbose", 0))
              if k in accepted}
    _assert_noiter_is_the_guess(lambda n: driver(system, basis, _options(ks, n), **kwargs))
    with pytest.raises(ValueError, match="max_iter must be >= 0"):
        driver(system, basis, _options(ks, -1), **kwargs)


@pytest.mark.parametrize("name, ks, open_shell", [
    ("run_pbc_gdf_rhf", False, False), ("run_pbc_gdf_rks", True, False),
    ("run_pbc_gdf_uhf", False, True), ("run_pbc_gdf_uks", True, True),
])
def test_gamma_gdf_driver_evaluates_the_guess_instead_of_returning_zero(name, ks, open_shell):
    """The GDF route run_periodic_job picks for these cells returned E = 0.0."""
    driver = getattr(vq, name)
    system = H3 if open_shell else vq.PeriodicSystem(
        3, 12.0 * np.eye(3), [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])])
    basis = _basis(system)
    zero = _assert_noiter_is_the_guess(lambda n: driver(system, basis, _options(ks, n)))
    assert zero.energy < -0.5
    with pytest.raises(ValueError, match="max_iter must be >= 0"):
        driver(system, basis, _options(ks, -1))


@pytest.mark.parametrize("driver, system, extra", [
    (gpw_closed.run_periodic_rhf_gpw, "h2", {}),
    (gpw_closed.run_periodic_rks_gpw, "h2", {"functional": "lda"}),
    (gpw_open.run_periodic_uhf_gpw, "h3", {}),
    (gpw_open.run_periodic_rohf_gpw, "h3", {}),
])
def test_gpw_driver_evaluates_the_guess(driver, system, extra):
    box = 8.0 * np.eye(3)
    system = (vq.PeriodicSystem(3, box, [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])])
              if system == "h2" else
              vq.PeriodicSystem(3, box, [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4]),
                                         vq.Atom(1, [0, 1.6, 0.7])], 0, 2))
    basis = _basis(system)
    _assert_noiter_is_the_guess(lambda n: driver(system, basis, max_iter=n, **extra))


def test_scf_log_names_a_zero_iteration_result():
    result = SimpleNamespace(converged=False, n_iter=0, scf_trace=[], energy=-1.5)
    text = format_scf_trace(result)
    assert INITIAL_GUESS_ONLY in text
    assert "NOT converged in 0 iterations" not in text


def test_periodic_job_records_that_no_iteration_ran(tmp_path):
    system = vq.PeriodicSystem(3, 12.0 * np.eye(3), [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])])
    with pytest.raises(RuntimeError, match=r"no SCF iteration ran \(max_iter = 0\)"):
        vq.run_periodic_job(system, _basis(system), method="RHF", max_iter=0,
                            output=str(tmp_path / "x"), write_molden_file=False,
                            write_density=False, citations=False)
    assert INITIAL_GUESS_ONLY in (tmp_path / "x.out").read_text()
    manifest = (tmp_path / "x.system").read_text()
    assert "scf_iterations = 0" in manifest
    assert f'scf_evaluation = "{INITIAL_GUESS_ONLY}"' in manifest


@pytest.mark.parametrize("method, functional", [("rhf", None), ("rks", "lda"), ("rks", "b3lyp")])
@pytest.mark.parametrize("temperature", [0.0, 0.03])
def test_restricted_multik_ewald_noiter_evaluates_initial_density(method, functional, temperature):
    """A genuine multi-k guess used to crash RHF or return zero RKS energy."""
    basis = _basis(LIH)
    kmesh = vq.monkhorst_pack(LIH, [2, 1, 1])
    driver = getattr(vq, f"run_{method}_periodic_multi_k_ewald3d")

    def run(n, initial_density_k=None):
        opts = _options(functional is not None, n)
        if functional:
            opts.functional = functional
        opts.initial_guess = vq.InitialGuess.HCORE
        opts.smearing_temperature = temperature
        opts.lattice_opts.cutoff_bohr = 5.0
        opts.lattice_opts.nuclear_cutoff_bohr = 12.0
        return driver(LIH, basis, kmesh, opts, auto_optimize_truncation=False,
                      verbose=0, initial_density_k=initial_density_k)

    zero = run(0)
    ordinary = run(2)
    assert not zero.converged
    assert zero.n_iter == 0
    assert not zero.scf_trace
    assert zero.energy < -1.0
    # The trace contains A = E - TS, so finite-T NOITER must preserve the
    # entropy of the evaluated guess rather than refill the new spectrum.
    assert zero.free_energy == ordinary.scf_trace[0].energy
    assert zero.free_energy == zero.energy - temperature * zero.entropy
    weights = np.asarray(kmesh.weights)
    ne = sum(w * np.trace(d @ s).real for w, d, s in
             zip(weights, zero.density_k, zero.overlap))
    assert ne == pytest.approx(4.0, abs=1e-10)
    for f, s, c, eps in zip(zero.fock, zero.overlap, zero.mo_coeffs, zero.mo_energies):
        np.testing.assert_allclose(f, f.conj().T, atol=1e-12)
        np.testing.assert_allclose(f @ c, (s @ c) * eps, atol=1e-10)
    core_energy = sum(w * np.trace(d @ h).real for w, d, h in
                      zip(weights, zero.density_k, zero.hcore))
    if functional is None:
        electronic = sum(0.5 * w * np.trace(d @ (h + f)).real for w, d, h, f in
                         zip(weights, zero.density_k, zero.hcore, zero.fock))
    else:
        electronic = core_energy + zero.e_coulomb + zero.e_hf_exchange + zero.e_xc
    assert zero.e_electronic == pytest.approx(electronic, abs=1e-12)
    assert zero.energy == pytest.approx(electronic + zero.e_nuclear, abs=1e-12)

    if temperature == 0.0:
        # A non-idempotent restart catches an accidental update to occupied
        # eigenvectors of F[D]. Its explicit density is authoritative.
        restart = [0.75 * d + 0.25 * (4.0 / len(s)) * np.linalg.inv(s)
                   for d, s in zip(zero.density_k, zero.overlap)]
        read = run(0, restart)
        assert read.energy == run(2, restart).scf_trace[0].energy
        assert abs(read.energy - zero.energy) > 1e-4
        assert read.n_iter == 0 and not read.converged and not read.scf_trace
        for actual, expected in zip(read.density_k, restart):
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        from vibeqc.periodic_k_density import real_space_density_from_per_k_density
        expected_real = real_space_density_from_per_k_density(
            restart, kmesh, read.density.cells)
        for actual, expected in zip(read.density.blocks, expected_real.blocks):
            np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("method", ["rhf", "rks"])
def test_restricted_multik_ewald_negative_budget_fails_before_setup(monkeypatch, method):
    import vibeqc.guess as guess

    def unexpected_setup(*args, **kwargs):
        pytest.fail("negative iteration budget reached guess setup")

    monkeypatch.setattr(guess, "select_periodic_driver_guess", unexpected_setup)
    driver = getattr(vq, f"run_{method}_periodic_multi_k_ewald3d")
    with pytest.raises(ValueError, match="max_iter must be >= 0"):
        driver(LIH, _basis(LIH), vq.monkhorst_pack(LIH, [2, 1, 1]), _options(method == "rks", -1))


@pytest.mark.parametrize("method, functional", [("uhf", None), ("uks", "lda"), ("uks", "b3lyp")])
@pytest.mark.parametrize("temperature", [0.0, 0.03])
def test_unrestricted_multik_ewald_noiter_preserves_spin_densities(method, functional, temperature):
    """Evaluate an open-shell guess, including a complex fractional READ."""
    basis = _basis(H3)
    kmesh = vq.monkhorst_pack(H3, [3, 1, 1])
    driver = getattr(vq, f"run_{method}_periodic_multi_k_ewald3d")

    def run(n, initial_density_k=None):
        opts = _options(functional is not None, n)
        if functional:
            opts.functional = functional
        opts.initial_guess = vq.InitialGuess.HCORE
        opts.smearing_temperature = temperature
        opts.lattice_opts.cutoff_bohr = 5.0
        opts.lattice_opts.nuclear_cutoff_bohr = 12.0
        return driver(H3, basis, kmesh, opts, auto_optimize_truncation=False,
                      verbose=0, initial_density_k=initial_density_k)

    zero = _assert_noiter_is_the_guess(run)
    assert zero.energy < -0.5
    assert zero.free_energy == zero.energy - temperature * zero.entropy
    weights = np.asarray(kmesh.weights)

    def check_state(result):
        for spin, n_elec in (("alpha", 2), ("beta", 1)):
            density = getattr(result, f"density_{spin}_k")
            count = sum(w * np.trace(d @ s).real for w, d, s in
                        zip(weights, density, result.overlap))
            assert count == pytest.approx(n_elec, abs=1e-10)
            for f, s, c, eps in zip(getattr(result, f"fock_{spin}"), result.overlap,
                                   getattr(result, f"mo_coeffs_{spin}"),
                                   getattr(result, f"mo_energies_{spin}")):
                np.testing.assert_allclose(f, f.conj().T, atol=1e-12)
                np.testing.assert_allclose(f @ c, (s @ c) * eps, atol=1e-10)
        da, db = result.density_alpha_k, result.density_beta_k
        if functional is None:
            electronic = sum(0.5 * w * np.trace((a + b) @ h + a @ fa + b @ fb).real
                             for w, a, b, h, fa, fb in zip(weights, da, db, result.hcore,
                                                          result.fock_alpha, result.fock_beta))
        else:
            core_energy = sum(w * np.trace((a + b) @ h).real
                              for w, a, b, h in zip(weights, da, db, result.hcore))
            electronic = core_energy + result.e_coulomb + result.e_hf_exchange + result.e_xc
        assert result.e_electronic == pytest.approx(electronic, abs=1e-12)
        assert result.energy == pytest.approx(electronic + result.e_nuclear, abs=1e-12)

    check_state(zero)
    if temperature == 0.0:
        assert zero.s_squared == pytest.approx(0.75, abs=1e-10)
    # A real Gamma density and conjugate +/-k densities, with distinct
    # fractional spin occupations. The analytic spin-orbital overlap is
    # an independent reference for the density-trace diagnostic.
    restart, natural_coeffs = [], []
    natural_occ = [np.array([0.9, 0.7, 0.4]), np.array([0.55, 0.3, 0.15])]
    for spin, occ in enumerate(natural_occ):
        densities, coeffs = [], []
        for k, s in zip(kmesh.kpoints, zero.overlap):
            eig, vec = np.linalg.eigh(s)
            x = (vec / np.sqrt(eig)) @ vec.conj().T
            theta = (0.3 + 0.2 * spin) * np.sin(12 * np.asarray(k)[0])
            u = np.eye(3, dtype=complex)
            u[:2, :2] = [[np.cos(theta), 1j * np.sin(theta)],
                         [1j * np.sin(theta), np.cos(theta)]]
            c = x @ u
            coeffs.append(c)
            densities.append((c * occ) @ c.conj().T)
        restart.append(densities)
        natural_coeffs.append(coeffs)
    assert max(np.max(np.abs(d.imag)) for ds in restart for d in ds) > 1e-3
    read = _assert_noiter_is_the_guess(lambda n: run(n, restart))
    check_state(read)
    expected_s2 = 1.75 - sum(
        float(w) * np.sum(np.outer(*natural_occ) * np.abs(ca.conj().T @ s @ cb) ** 2)
        for w, s, ca, cb in zip(weights, read.overlap, *natural_coeffs)
    )
    assert read.s_squared == pytest.approx(expected_s2, abs=1e-10)
    expected_entropy = -sum(np.sum(o * np.log(o) + (1 - o) * np.log1p(-o))
                            for o in natural_occ) if temperature > 0.0 else 0.0
    assert read.entropy == pytest.approx(expected_entropy, abs=1e-12)
    assert read.free_energy == pytest.approx(read.energy - temperature * expected_entropy, abs=1e-12)
    from vibeqc.periodic_k_density import real_space_density_from_per_k_density
    for spin, expected in zip(("alpha", "beta"), restart):
        for actual, d in zip(getattr(read, f"density_{spin}_k"), expected):
            np.testing.assert_allclose(actual, d, atol=1e-12)
        actual_real = getattr(read, f"density_{spin}")
        expected_real = real_space_density_from_per_k_density(expected, kmesh, actual_real.cells)
        for actual, d in zip(actual_real.blocks, expected_real.blocks):
            np.testing.assert_allclose(actual, d, atol=1e-12)


@pytest.mark.parametrize("method", ["uhf", "uks"])
def test_unrestricted_multik_ewald_negative_budget_fails_before_setup(monkeypatch, method):
    import vibeqc.periodic_k_gdf as gdf

    def unexpected_setup(*args, **kwargs):
        pytest.fail("negative iteration budget reached ECP setup")

    monkeypatch.setattr(gdf, "_periodic_ecp_context", unexpected_setup)
    driver = getattr(vq, f"run_{method}_periodic_multi_k_ewald3d")
    with pytest.raises(ValueError, match="max_iter must be >= 0"):
        driver(H3, _basis(H3), vq.monkhorst_pack(H3, [2, 1, 1]), _options(method == "uks", -1))


@pytest.mark.parametrize("method", ["uhf", "uks"])
def test_unrestricted_multik_noiter_retains_sad_density_at_gamma(monkeypatch, method):
    import importlib

    module = importlib.import_module(f"vibeqc.periodic_{method}_multi_k_ewald")
    original = module.initial_densities_open_shell
    seeds = []

    def capture_seed(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None
        seeds.append([np.array(d, copy=True) for d in result])
        return result

    monkeypatch.setattr(module, "initial_densities_open_shell", capture_seed)
    basis, mesh = _basis(H3), vq.monkhorst_pack(H3, [1, 1, 1])

    def run(n):
        opts = _options(method == "uks", n)
        opts.initial_guess = vq.InitialGuess.SAD
        opts.lattice_opts.cutoff_bohr = 5.0
        opts.lattice_opts.nuclear_cutoff_bohr = 12.0
        return getattr(module, f"run_{method}_periodic_multi_k_ewald3d")(
            H3, basis, mesh, opts, auto_optimize_truncation=False, verbose=0)

    zero = _assert_noiter_is_the_guess(run)
    np.testing.assert_allclose(zero.density_alpha_k[0], seeds[0][0], atol=1e-12)
    np.testing.assert_allclose(zero.density_beta_k[0], seeds[0][1], atol=1e-12)


@pytest.mark.parametrize("omega", [0.0, 0.35])
@pytest.mark.parametrize("adjoint", [False, True])
def test_gamma_molecular_limit_repeated_build_is_exact(omega, adjoint):
    """Repeat at a fixed team size; run with OMP_NUM_THREADS=2 or more."""
    system = vq.PeriodicSystem(
        3, 8.0 * np.eye(3),
        [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4]),
         vq.Atom(1, [0, 1.6, 0.7])], 0, 2)
    basis = _basis(system)
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 12.0
    density = np.array([[0.8, 0.13, -0.07], [0.13, 0.6, 0.04],
                        [-0.07, 0.04, 0.5]])

    def build(d=density, w=omega):
        return core.build_jk_gamma_molecular_limit(
            basis, system, opts, d, w, adjoint_symmetric=adjoint)

    first = build()
    assert np.isfinite(first.J).all() and np.isfinite(first.K).all()
    assert np.linalg.norm(first.K) > 0.0
    for _ in range(5):
        repeated = build()
        np.testing.assert_array_equal(repeated.J, first.J)
        np.testing.assert_array_equal(repeated.K, first.K)
    zero = build(np.zeros_like(density))
    np.testing.assert_array_equal(zero.J, np.zeros_like(density))
    np.testing.assert_array_equal(zero.K, np.zeros_like(density))
    with pytest.raises((ValueError, RuntimeError), match="density shape mismatch"):
        build(np.zeros((2, 2)))
    with pytest.raises((ValueError, RuntimeError), match="omega must be non-negative"):
        build(w=-0.1)
