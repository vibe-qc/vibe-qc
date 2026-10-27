"""Periodic Γ-point ROHF (EWALD_3D) tests.

Runs on a build box (needs the compiled periodic J/K). The Roothaan
coupling, DIIS and occupation are shared with the molecular ROHF driver
(unit-tested in ``tests/test_rohf.py``); these tests pin the *periodic*
integration:

* closed-shell limit equals periodic RHF (the Roothaan effective Fock
  collapses to the closed-shell Fock; same Ewald J / real-space K / e_nuc),
* exact ⟨S²⟩ = S(S+1) for an open shell,
* a one-electron doublet equals periodic UHF (no closed shell → identical),
* ROHF lies at or above UHF (variational; UHF has more freedom) and is
  spin-pure where UHF may be contaminated.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _options():
    o = vq.PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.damping = 0.3
    o.max_iter = 60
    return o


def _h2_closed_shell(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _atom_doublet(Z: int, box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, [vq.Atom(Z, [c, c, c])])
    sysp.multiplicity = 2
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def test_periodic_rohf_closed_shell_matches_rhf():
    """Closed-shell H2 in a box: periodic ROHF == periodic RHF, ⟨S²⟩ = 0."""
    sysp, basis = _h2_closed_shell()
    opts = _options()
    r_rhf = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3
    )
    r_rohf = vq.run_rohf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3
    )
    assert r_rhf.converged and r_rohf.converged
    assert r_rohf.energy == pytest.approx(r_rhf.energy, abs=1e-8)
    assert abs(r_rohf.s_squared) < 1e-12


def test_periodic_rohf_h_atom_equals_uhf():
    """One-electron doublet: no closed shell, so ROHF == UHF exactly, and
    ⟨S²⟩ = 0.75."""
    sysp, basis = _atom_doublet(1)
    opts = _options()
    r_rohf = vq.run_rohf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3
    )
    r_uhf = vq.run_uhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3
    )
    assert r_rohf.converged and r_uhf.converged
    assert r_rohf.s_squared == pytest.approx(0.75, abs=1e-12)
    assert r_rohf.energy == pytest.approx(r_uhf.energy, abs=1e-7)
    # ROHF occupations: one singly-occupied orbital, rest empty.
    assert float(np.sum(r_rohf.mo_occupations)) == pytest.approx(1.0, abs=1e-10)


def test_periodic_rohf_li_atom_spin_pure_and_above_uhf():
    """Li atom in a box (3 e⁻, doublet: 1 closed + 1 open): ROHF is
    spin-pure (⟨S²⟩ = 0.75 exactly) and lies at or above UHF."""
    sysp, basis = _atom_doublet(3)
    opts = _options()
    r_rohf = vq.run_rohf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3
    )
    assert r_rohf.converged
    assert r_rohf.s_squared == pytest.approx(0.75, abs=1e-10)
    # occupations: 1 closed (2.0) + 1 open (1.0) = 3 electrons.
    assert float(np.sum(r_rohf.mo_occupations)) == pytest.approx(3.0, abs=1e-10)

    r_uhf = vq.run_uhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3
    )
    if r_uhf.converged:
        assert r_uhf.energy <= r_rohf.energy + 1e-7


def test_periodic_rohf_invalid_multiplicity_raises():
    box = 30.0
    c = box / 2
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, [vq.Atom(1, [c, c, c])])
    sysp.multiplicity = 1  # 1 electron + singlet is impossible
    # The impossible spin state is rejected fail-fast at molecule construction
    # (Molecule validates n_electrons vs multiplicity, cpp/src/molecule.cpp),
    # before the ROHF driver ever runs. Wrap the whole build+run so the guard
    # is asserted wherever it fires.
    with pytest.raises(ValueError, match="multiplicity"):
        basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
        vq.run_rohf_periodic_gamma_ewald3d(sysp, basis, _options())


@pytest.mark.parametrize("davidson", [False, True])
@pytest.mark.parametrize("rank", [5, 7])
@pytest.mark.parametrize("shift", [0., .25])
def test_effective_fock_shared_solve_preserves_metric_and_open_shell_selection(davidson, rank, shift):
    scipy = pytest.importorskip("scipy.linalg")
    from vibeqc.periodic_rohf_ewald import _solve_and_occupy
    from vibeqc.solvers._scf import FockDiagonalizer

    rng = np.random.default_rng(487)
    u = np.linalg.qr(rng.normal(size=(7, 7)))[0]
    metric_values = np.linspace(.4, 1.6, rank)
    s = (u * np.r_[metric_values, np.full(7-rank, 1e-14)]) @ u.T
    x = u[:, :rank] / np.sqrt(metric_values)
    # The open orbital selected by Fa is deliberately not the first virtual
    # orbital in the effective-Fock ordering.
    energies = np.linspace(-2., 2., rank)
    alpha_energies = np.linspace(-1., 1., rank)
    alpha_energies[2] = -3.
    covariant = s @ x
    fock = (covariant * energies) @ covariant.T
    focka = (covariant * alpha_energies) @ covariant.T
    if rank < 7:
        coupling = .03 * np.outer(u[:, -1], u[:, 0])
        fock += coupling + coupling.T
    dma = x[:, :2] @ x[:, :2].T
    inputs = [a.copy() for a in (fock, focka, x, s, dma)]
    options = _options()
    options.use_davidson = davidson
    options.davidson_min_dim = 0
    diagonalize = FockDiagonalizer(x, s, options, spin="restricted-open", n_required=rank)
    c, eps, occ = _solve_and_occupy(fock, focka, s, dma, shift, 2, 1,
                                   diagonalize=diagonalize)
    shifted = fock + shift * (s - s @ dma @ s)
    frame = u[:, :rank]
    reference = scipy.eigh(frame.T @ shifted @ frame, frame.T @ s @ frame, eigvals_only=True)
    np.testing.assert_allclose(eps, reference, atol=1e-12)
    np.testing.assert_allclose(c.T @ s @ c, np.eye(rank), atol=1e-12)
    expected_occ = np.zeros(rank)
    expected_occ[0], expected_occ[2] = 2., 1.
    np.testing.assert_array_equal(occ, expected_occ)
    result = diagonalize.last_result
    assert result.operator_metadata.spin == "restricted-open"
    assert result.operator_metadata.kpoint == (0., 0., 0.)
    assert result.retained_rank == rank
    assert result.residual_frame == "original"
    residual = shifted @ c - (s @ c) * eps
    np.testing.assert_allclose(result.residuals, np.linalg.norm(residual, axis=0), atol=1e-13)
    assert max(result.projected_residuals) < 1e-12
    if rank < 7:
        assert max(result.residuals) > .01
    else:
        assert max(result.residuals) < 1e-12
    for actual, original in zip((fock, focka, x, s, dma), inputs):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("davidson", [False, True])
@pytest.mark.parametrize("system_kind", ["closed", "open"])
def test_gamma_rohf_shared_solve_matches_previous_iteration(monkeypatch, davidson, system_kind):
    scipy = pytest.importorskip("scipy.linalg")
    from vibeqc import periodic_rohf_ewald as driver
    from vibeqc.solvers import _scf

    system, basis = _h2_closed_shell() if system_kind == "closed" else _atom_doublet(3)
    options = _options()
    options.use_davidson = davidson
    options.davidson_min_dim = 0
    options.level_shift = .2
    # Compare orbitals at a matched SCF accuracy finer than the spectral
    # assertion, rather than comparing different accepted stopping points.
    options.conv_tol_energy = 1e-12
    options.conv_tol_grad = 1e-10
    options.max_iter = 100
    options.davidson.conv_tol = 1e-10
    options.davidson.guess_vectors = np.eye(basis.nbasis)
    guess = np.asarray(options.davidson.guess_vectors).copy()
    original = _scf.FockDiagonalizer.__call__
    generations = []

    def checked(self, fock):
        c, eps = original(self, fock)
        result = self.last_result
        np.testing.assert_allclose(eps, scipy.eigh(fock.matrix, self.overlap, eigvals_only=True), atol=1e-9)
        assert result.operator_metadata.kind == "fock"
        assert result.operator_metadata.spin == "restricted-open"
        assert result.operator_metadata.kpoint == (0., 0., 0.)
        assert result.eigenvalues.size == self.X.shape[1]
        assert max(result.residuals) < 1e-9
        assert result.orthogonality_error < 1e-12
        assert self.iterative == davidson
        generations.append(result.operator_metadata.generation)
        return c, eps

    def run():
        # Finite-grid parity tests the eigensolver migration, not convergence
        # of this fixed Ewald source with respect to its numerical supports.
        return driver.run_rohf_periodic_gamma_ewald3d(
            system, basis, options, omega=.5, grid_shape=(24, 24, 24),
            auto_optimize_truncation=False,
        )

    monkeypatch.setattr(_scf.FockDiagonalizer, "__call__", checked)
    result = run()
    assert result.converged
    assert generations == sorted(set(generations))
    assert len(generations) >= 2  # The separate HCORE seed is not a Fock epoch.
    np.testing.assert_array_equal(options.davidson.guess_vectors, guess)
    assert options.davidson.n_eig == 0

    def previous(fock_eff, focka, s, dma, level_shift, n_alpha, n_beta, *, diagonalize):
        f = fock_eff.matrix + level_shift * (s - s @ dma @ s)
        x = diagonalize.X
        projected = x.T @ f @ x
        eps, vectors = np.linalg.eigh(.5 * (projected + projected.T))
        c = x @ vectors
        alpha = np.einsum("pi,pq,qi->i", c, focka, c)
        return c, eps, driver._roothaan_occupations(eps, alpha, n_alpha, n_beta)

    monkeypatch.setattr(driver, "_solve_and_occupy", previous)
    reference = run()
    assert reference.converged
    assert result.energy == pytest.approx(reference.energy, abs=1e-9)
    assert result.s_squared == pytest.approx(reference.s_squared, abs=1e-12)
    np.testing.assert_allclose(result.mo_energies, reference.mo_energies, atol=1e-9)
    np.testing.assert_array_equal(result.mo_occupations, reference.mo_occupations)
    np.testing.assert_allclose(result.density_alpha, reference.density_alpha, atol=1e-9)
    np.testing.assert_allclose(result.density_beta, reference.density_beta, atol=1e-9)
    np.testing.assert_array_equal(options.davidson.guess_vectors, guess)
    assert options.davidson.n_eig == 0


@pytest.mark.parametrize("defect", ["partial_roots", "excess_roots", "unsupported_option"])
def test_gamma_rohf_rejects_incompatible_solver_before_fock_build(monkeypatch, defect):
    from vibeqc import periodic_rohf_ewald as driver

    system, basis = _atom_doublet(3)
    options = _options()
    options.use_davidson = True
    options.davidson_min_dim = 0
    if defect == "partial_roots":
        # Covers the occupied alpha space, but omits candidates needed by
        # open-shell selection from the complete retained spectrum.
        options.davidson.n_eig = 2
        error, message = ValueError, "Requested Fock roots"
    elif defect == "excess_roots":
        options.davidson.n_eig = basis.nbasis + 1
        error, message = ValueError, "retained orbital space"
    else:
        options.davidson.max_refine = 6
        error, message = NotImplementedError, "max_refine"

    def fail(*args, **kwargs):
        pytest.fail("Incompatible solver controls must fail before building J/K")

    monkeypatch.setattr(driver, "make_ewald_3d_gamma_j_builder", fail)
    monkeypatch.setattr(driver, "_GammaUHFBuilder", fail)
    with pytest.raises(error, match=message):
        driver.run_rohf_periodic_gamma_ewald3d(
            system, basis, options, omega=.5, grid_shape=(24, 24, 24),
            auto_optimize_truncation=False,
        )


@pytest.mark.parametrize("seed", ["HCORE", "SAD", "PATOM", "SAP", "HUECKEL", "MINAO", "READ"])
@pytest.mark.parametrize("scalar", ["real", "complex_guess", "complex_frame", "inactive_guess"])
def test_gamma_rohf_real_only_admission(monkeypatch, seed, scalar):
    """Public options reach the real-only guard before seed or J/K work."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    from vibeqc import guess, linear_dependence, periodic_rohf_ewald as driver, periodic_v_ne
    from vibeqc.solvers import _scf

    options = vq.PeriodicRHFOptions()
    options.initial_guess = getattr(vq.InitialGuess, seed)
    options.use_davidson = scalar != "inactive_guess"
    options.davidson_min_dim = 0
    options.davidson.n_eig = options.davidson.n_guess = 3
    options.davidson.guess_vectors = np.eye(3)
    complex_guess = np.diag([1., 1., 1j])
    if scalar in {"complex_guess", "inactive_guess"}:
        options.davidson.guess_vectors_cplx = complex_guess
    original_guess = np.asarray(options.davidson.guess_vectors).copy()
    original_complex = np.asarray(options.davidson.guess_vectors_cplx).copy()
    events, seen = [], []
    system = SimpleNamespace(dim=3, multiplicity=2, lattice=np.eye(3)*30,
                             n_electrons=lambda: 3, unit_cell_molecule=lambda: None)
    basis = SimpleNamespace(name="synthetic", nbasis=3, nshells=3)
    logger = SimpleNamespace(info=lambda *a: None, stage=lambda *a, **k: nullcontext())
    monkeypatch.setattr(driver, "resolve_progress", lambda *a, **k: logger)
    monkeypatch.setattr(driver, "_gamma_source_details", lambda *a, **k:
                        dict(shells=[], ordering="synthetic"))
    monkeypatch.setattr(driver, "_refuse_if_dense_ionic", lambda *a: None)
    # Seed construction is outside scalar admission; exercise every supported
    # selected seed while retaining the complete public driver body.
    monkeypatch.setattr(guess, "select_periodic_driver_guess", lambda *a, **k:
                        SimpleNamespace(effective=options.initial_guess))
    def overlap(*args):
        events.append("integrals")
        return np.eye(3)
    monkeypatch.setattr(driver, "compute_overlap_lattice", overlap)
    monkeypatch.setattr(driver, "compute_kinetic_lattice", lambda *a: np.diag([-3., -2., -1.]))
    monkeypatch.setattr(periodic_v_ne, "compute_nuclear_lattice_dispatch", lambda *a, **k: np.zeros((3, 3)))
    monkeypatch.setattr(driver, "bloch_sum", lambda matrix, *a: matrix)
    monkeypatch.setattr(driver, "nuclear_repulsion_per_cell", lambda *a: 0.)
    monkeypatch.setattr(linear_dependence, "scf_preflight_overlap_check", lambda *a, **k: None)
    if scalar == "complex_frame":
        monkeypatch.setattr(driver, "_canonical_orthogonalizer", lambda *a, **k: (complex_guess, 3))
    def initial(*args, **kwargs):
        events.append("seed")
        return None if seed == "HCORE" else (np.diag([1., 1., 0.]), np.diag([1., 0., 0.]))
    monkeypatch.setattr(driver, "initial_densities_open_shell", initial)
    class Diagonalizer(_scf.FockDiagonalizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append(self)
    monkeypatch.setattr(driver, "FockDiagonalizer", Diagonalizer)
    class BeforeJK(RuntimeError):
        pass
    def builder(*args, **kwargs):
        events.append("J factory")
        raise BeforeJK("Reached J factory with admitted solver")
    monkeypatch.setattr(driver, "make_ewald_3d_gamma_j_builder", builder)
    def no_jk(*args, **kwargs):
        pytest.fail("No J/K evaluation is needed for scalar admission")
    monkeypatch.setattr(driver, "_GammaUHFBuilder", no_jk)
    errors = {"complex_guess": "complex Davidson guesses", "complex_frame": "real X and overlap"}
    expected = NotImplementedError if scalar in errors else BeforeJK
    with pytest.raises(expected, match=errors.get(scalar, "Reached J factory")):
        driver.run_rohf_periodic_gamma_ewald3d(
            system, basis, options, omega=.5, grid_shape=(2, 2, 2),
            auto_optimize_truncation=False,
        )
    assert events == (["integrals"] if scalar in errors else ["integrals", "seed", "J factory"])
    if scalar not in errors:
        assert len(seen) == 1
        solve = _scf.solve_operator
        def dense_control(operator, solver_options, *, solver, previous):
            assert solver == ("dense" if scalar == "inactive_guess" else "davidson")
            return solve(operator, _scf.SolverOptions(n_roots=solver_options.n_roots), solver="dense")
        monkeypatch.setattr(_scf, "solve_operator", dense_control)
        diagonalize = seen[0]
        c, eps, occ = driver._solve_and_occupy(
            np.diag([-3., -2., -1.]), np.diag([-3., -.5, 1.]),
            np.eye(3), np.diag([1., 1., 0.]), 0., 2, 1, diagonalize=diagonalize,
        )
        assert not np.iscomplexobj(c)
        np.testing.assert_allclose(c.T @ c, np.eye(3), atol=1e-12)
        np.testing.assert_array_equal(occ, [2., 1., 0.])
        with pytest.raises(NotImplementedError, match="complex Fock matrices"):
            diagonalize(np.diag([-3., -2., -1.]).astype(complex))
    np.testing.assert_array_equal(options.davidson.guess_vectors, original_guess)
    np.testing.assert_array_equal(options.davidson.guess_vectors_cplx, original_complex)
    assert options.davidson.n_eig == options.davidson.n_guess == 3


def _evaluated_rohf_setup(kind, davidson=False, max_iter=80):
    if kind == "h2":
        system, basis = _h2_closed_shell()
    elif kind == "lih+":
        system = vq.PeriodicSystem(3, np.eye(3)*30.,
            [vq.Atom(3, [15., 15., 15.]), vq.Atom(1, [15., 15., 18.])],
            charge=1, multiplicity=2)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    else:
        system, basis = _atom_doublet(1 if kind == "h" else 3)
    options = _options()
    options.initial_guess = vq.InitialGuess.PATOM
    options.level_shift = .2
    options.max_iter = max_iter
    options.use_davidson = davidson
    options.davidson_min_dim = 0
    options.davidson.conv_tol = 1e-11
    options.conv_tol_energy = 1e-11
    options.conv_tol_grad = 1e-9
    return system, basis, options


@pytest.mark.parametrize("kind,davidson,max_iter", [
    ("h2", False, 80), ("h", True, 80),
    ("li", False, 80), ("li", True, 80),
    ("lih+", False, 80), ("lih+", False, 1), ("lih+", True, 0),
])
def test_gamma_rohf_evaluated_state(monkeypatch, tmp_path, kind, davidson, max_iter):
    import json
    import zipfile
    from vibeqc import periodic_rohf_ewald as driver
    from vibeqc.solvers import _scf
    from vibeqc.output import OutputPlan
    from vibeqc.output.formats.qvf import write_qvf, qvf_wf_data, scf_history_from_result

    system, basis, options = _evaluated_rohf_setup(kind, davidson, max_iter)
    builders, builds, operators = [], [], []
    init, build = driver._GammaUHFBuilder.__init__, driver._GammaUHFBuilder.__call__
    solve = _scf.solve_operator
    def capture_init(self, *args, **kwargs):
        init(self, *args, **kwargs); builders.append(self)
    def capture_build(self, *args, **kwargs):
        state = build(self, *args, **kwargs); builds.append(state)
        return state
    def capture_solve(operator, *args, **kwargs):
        operators.append(operator)
        return solve(operator, *args, **kwargs)
    monkeypatch.setattr(driver._GammaUHFBuilder, "__init__", capture_init)
    monkeypatch.setattr(driver._GammaUHFBuilder, "__call__", capture_build)
    monkeypatch.setattr(_scf, "solve_operator", capture_solve)
    monkeypatch.setattr(driver, "solve_operator", capture_solve)
    r = driver.run_rohf_periodic_gamma_ewald3d(system, basis, options,
        omega=.5, grid_shape=(24, 24, 24), auto_optimize_truncation=False, progress=False)
    state, effective, builder = r.fock_build, r.effective_fock_operator, builders[0]
    assert r.converged == (max_iter == 80)
    assert state is builds[-1] and state.method == "ROHF"
    assert [b.generation for b in builds] == list(range(len(builds)))
    assert len({b.source.identity for b in builds}) == 1
    assert r.energy == state.energy.total
    assert r.e_electronic + r.e_nuclear == pytest.approx(r.energy, abs=1e-12)
    assert len(r.evaluated_build_history) == r.n_iter
    if r.n_iter:
        assert r.scf_trace[-1].energy == r.energy
        assert r.evaluated_build_history[-1]["build_id"] == state.identity
        assert r.scf_trace[-1].grad_norm == r.effective_fock_residual
    assert state.metric_electron_counts == pytest.approx((r.n_alpha, r.n_beta), abs=1e-11)
    assert effective.metadata.role == "effective"
    assert effective.metadata.build_id == state.identity
    assert effective.metadata.density_id == state.density_id
    assert effective.metadata.spin == "restricted-open"
    assert effective.frame is state.channels[0].operator.frame is state.channels[1].operator.frame
    assert all(c.operator.metadata.role == "physical" for c in state.channels)
    np.testing.assert_array_equal(r.fock, effective.matrix)
    for spin, channel in zip(("alpha", "beta"), state.channels):
        np.testing.assert_array_equal(getattr(r, "fock_"+spin), channel.operator.matrix)
        np.testing.assert_array_equal(getattr(r, "density_"+spin), channel.density)
    c, occ, s = r.mo_coeffs, r.mo_occupations, r.overlap
    if max_iter:
        # max_iter=0 (NOITER, #649) returns the guess density itself, which
        # the orbitals of its Fock need not reproduce.
        np.testing.assert_allclose((c*(occ > 0))@c.T, r.density_alpha, atol=1e-12)
        np.testing.assert_allclose((c*(occ == 2))@c.T, r.density_beta, atol=1e-12)
    np.testing.assert_allclose(c.T@s@c, np.eye(c.shape[1]), atol=1e-12)
    np.testing.assert_allclose(np.diag(c.T@r.fock@c), r.mo_energies, atol=1e-12)
    for occupation in (0., 1., 2.):
        block = c[:, occ == occupation].T @ r.fock @ c[:, occ == occupation]
        np.testing.assert_allclose(block, np.diag(np.diag(block)), atol=1e-12)
    expected = np.linalg.norm(driver._commutator_error(r.fock, r.density, s, effective.frame.X))
    assert r.effective_fock_residual == pytest.approx(expected, abs=1e-14)
    if kind == "lih+" and max_iter == 80:
        assert r.effective_fock_residual < options.conv_tol_grad
        assert state.commutator_norm > 1e-5  # Unconstrained UHF is not the ROHF stationarity test.
    by_id = {b.identity:b for b in builds}
    # The core guess and projected occupation blocks now cross the same
    # solver boundary, but are not density-evaluated physical build epochs.
    assert operators[0].metadata.kind == "one_electron"
    np.testing.assert_array_equal(operators[0].overlap, s)
    built_operators = [op for op in operators if op.metadata.build_id is not None]
    assert [op.metadata.spin for op in built_operators[:2]] == ["alpha", "beta"]  # PATOM
    for op in operators[1:]:
        if op.metadata.build_id is None:
            assert op.metadata.basis_id.startswith("gamma-rohf-occupation-")
            assert op.overlap is None
    for op in built_operators:
        parent = by_id[op.metadata.build_id]
        assert op.metadata.density_id == parent.density_id
        assert op.frame is parent.channels[0].operator.frame
        if op.metadata.role == "solver":
            assert op.metadata.modification
    assert options.davidson.n_eig == 0 and options.davidson.guess_vectors.size == 0

    stem = tmp_path/"gamma_rohf"
    plan = OutputPlan.from_run_job_kwargs(output=stem, method="rohf", basis=basis.name, functional=None)
    path = write_qvf(stem, plan, molecule=system.unit_cell_molecule(), system=system,
        result=r, method="rohf", basis=basis.name, record_hostname=False,
        wf_data=qvf_wf_data(r,basis,system.unit_cell_molecule()), scf_history=scf_history_from_result(r))
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    fields = manifest["provenance"]["run_fields"]
    assert fields["fock_build_id"] == fields["effective_fock_parent_build_id"] == state.identity
    assert fields["effective_fock_role"] == "effective"
    assert manifest["provenance"]["scf_energy"]["value"] == r.energy

    # Physical spin energy derivatives belong to Fa/Fb, not the effective Fock.
    rng = np.random.default_rng(472)
    directions = rng.normal(size=(2, basis.nbasis, basis.nbasis))
    directions = .03*(directions + directions.swapaxes(1, 2))
    densities = np.array([channel.density for channel in state.channels])
    step = 1e-5
    plus, minus = builder(*(densities+step*directions)), builder(*(densities-step*directions))
    derivative = sum(np.einsum("ij,ji->", channel.operator.matrix, direction)
                     for channel, direction in zip(state.channels, directions))
    assert (plus.energy.total-minus.energy.total)/(2*step) == pytest.approx(derivative, abs=3e-9)
    options.lattice_opts.cutoff_bohr += 1.
    with pytest.raises(ValueError, match="numerical source changed"):
        builder(*densities)
    before = state.energy.total
    r.density_alpha[:] = 0.; r.density_beta[:] = 0.; r.fock[:] = 0.; r.fock_alpha[:] = 0.; r.overlap[:] = 0.
    assert state.energy.total == before
    assert state.metric_electron_counts == pytest.approx((r.n_alpha, r.n_beta), abs=1e-11)
    assert np.linalg.norm(effective.matrix) > 0.
    with pytest.raises(ValueError):
        effective.matrix.setflags(write=True)


@pytest.mark.parametrize("occupations", [[2, 2, 0, 0], [1, 1, 0, 0], [2, 1, 1, 0]])
def test_gamma_rohf_occupation_blocks_use_shared_solver(monkeypatch, occupations):
    from vibeqc import periodic_rohf_ewald as driver

    rng = np.random.default_rng(472)
    coefficients, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    fock = rng.normal(size=(4, 4))
    fock = .5 * (fock + fock.T)
    occupations = np.asarray(occupations, dtype=float)
    calls = []
    solve = driver.solve_dense_fock

    def checked(matrix, **kwargs):
        result = solve(matrix, **kwargs)
        expected = np.linalg.eigvalsh(matrix)
        np.testing.assert_allclose(result.eigenvalues, expected, atol=1e-13)
        assert result.residual_frame == "orthogonal"
        calls.append(matrix.shape[0])
        return result

    monkeypatch.setattr(driver, "solve_dense_fock", checked)
    c, energies, occ = driver._canonicalize_final_orbitals(coefficients, occupations, fock)
    assert calls == [int(np.count_nonzero(occupations == x)) for x in (2, 1, 0)
                     if np.any(occupations == x)]
    for selector in (lambda x: x > 0, lambda x: x == 2):
        np.testing.assert_allclose((c * selector(occ)) @ c.T,
                                   (coefficients * selector(occupations)) @ coefficients.T,
                                   atol=1e-13)
    np.testing.assert_allclose(energies, np.diag(c.T @ fock @ c), atol=1e-13)
