"""Phase 12e-c-4b: Γ-point periodic RHF SCF driver using EWALD_3D.

The correctness contract is the ω-invariance of the Ewald identity:
at SCF convergence, the total energy from
``run_rhf_periodic_gamma_ewald3d`` must be independent of the
splitting parameter ω, up to the finite-box / finite-grid residual
measured in 12e-c-4a (about 0.3 % on H2 / STO-3G / 30-bohr / 0.3-bohr
grid).

The driver's energy differs from the existing
``run_rhf_periodic_gamma`` (DIRECT_TRUNCATED) reference by the
Makov–Payne constant shift α · S integrated against the density —
well-defined physics, and the offset matches the prediction
``α ≈ -α_M · Q / L`` (simple-cubic Madelung constant) to within a
percent on suitable test systems.

Systems used
------------

Every test uses H₂ / STO-3G — no tight cores, cleanly resolved on a
0.3-bohr grid. H₂O / STO-3G fails because the O 1s core is too tight
for the default grid; that's the known-xfail case from 12e-c-4a,
documented in :func:`vibeqc.build_j_ewald_3d`.

Tests
-----

1. **Convergence on H₂** — SCF terminates successfully.
2. **ω-invariance** — total energy stable across a spread of ω.
3. **Makov–Payne consistency** — ``E_ewald - E_direct`` matches
   ``0.5 · α_M · n_elec² / L`` for a cubic box.
4. **Result shape correctness** — mo_coeffs / mo_energies / density /
   fock / overlap come out as expected-shape numpy arrays.
5. **Rejection of multi-atom multiplicity mismatches** — open-shell
   structures can't go through the closed-shell RHF driver.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq


def _run_ewald3d(*args, **kwargs):
    """``run_rhf_periodic_gamma_ewald3d`` with the truncation
    auto-optimiser pinned OFF (defensive — on the dilute H₂ boxes here
    it is a no-op since the overlap is already PSD; kept off so the one
    tight LiH-primitive gauge test never pays a cutoff-tuning re-run).

    The actual speed-up for this file is ``use_diis=True`` in
    ``_default_options``: these Γ-Ewald SCFs converge in ~3 iterations
    instead of ~17, and the dominant cost is the per-iteration analytic-FT
    Hartree-J rebuild (PBC-audit E2), so fewer iterations ⇒ proportionally
    fewer rebuilds (≈508→231 s CPU, all assertions unchanged). Note:
    auto_optimize_truncation / grid-spacing were measured *not* to matter
    here (45.8 vs 46.2 s; 100³ vs 38³ grid identical) — the J is analytic.
    """
    kwargs.setdefault("auto_optimize_truncation", False)
    return _run_rhf_periodic_gamma_ewald3d(*args, **kwargs)


_run_rhf_periodic_gamma_ewald3d = vq.run_rhf_periodic_gamma_ewald3d


@pytest.mark.parametrize("max_iter,iterative,quadratic", [
    (1, False, 0), (80, False, 0), (80, True, 0), (2, True, 1)])
def test_rhf_evaluated_state_terminal_and_qvf(monkeypatch, tmp_path, max_iter, iterative, quadratic):
    _check_rhf_evaluated_state(monkeypatch, tmp_path, max_iter, iterative, quadratic, dim=3)


@pytest.mark.parametrize("max_iter,iterative,quadratic", [
    (1, False, 0), (80, False, 0), (80, True, 0), (2, True, 1), (2, False, 0)])
def test_slab_rhf_evaluated_state_terminal_and_qvf(monkeypatch, tmp_path, max_iter, iterative, quadratic):
    _check_rhf_evaluated_state(monkeypatch, tmp_path, max_iter, iterative, quadratic, dim=2)


def _slab_he_setup(max_iter=80):
    # Neutral He with two radial functions gives a nontrivial occupied update
    # without using a charged slab or a large, expensive exchange problem.
    system = vq.PeriodicSystem(2, np.diag([18., 18., 40.]), [vq.Atom(2, [0., 0., 1.3])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "6-31g")
    options = _default_options(iter_limit=max_iter)
    options.initial_guess = vq.InitialGuess.PATOM if max_iter > 2 else vq.InitialGuess.HCORE
    options.level_shift, options.fock_mixing = .2, .15
    options.conv_tol_energy, options.conv_tol_grad = 1e-12, 1e-10
    options.davidson.conv_tol = 1e-12
    return system, basis, options


def _check_rhf_evaluated_state(monkeypatch, tmp_path, max_iter, iterative, quadratic, *, dim):
    """A nontrivial stopped density, physical Fock and output share one epoch."""
    import json
    import zipfile
    from vibeqc import periodic_rhf_ewald as driver
    from vibeqc.output import OutputPlan
    from vibeqc.output.formats.qvf import write_qvf, qvf_wf_data, scf_history_from_result
    from vibeqc.solvers import _scf

    # HeH+ has a changing occupied orbital even in a two-function basis.
    # The charged Ewald source is used consistently by every comparison.
    system = vq.PeriodicSystem(3, np.eye(3) * 30.,
        [vq.Atom(2, [15., 15., 14.3]), vq.Atom(1, [15., 15., 15.7])], charge=1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = _default_options(iter_limit=max_iter)
    if dim == 2:
        system, basis, options = _slab_he_setup(max_iter)
        if max_iter == 2 and not quadratic:
            options.use_diis = False
            options.dynamic_damping = True
    options.level_shift, options.fock_mixing = .2, .15
    options.conv_tol_energy, options.conv_tol_grad = 1e-12, 1e-10
    options.use_davidson, options.davidson_min_dim = iterative, 0
    options.davidson.conv_tol = 1e-12
    options.quadratic_fallback_iter = quadratic
    builds, solvers, instances = [], [], []
    call, init, solve = driver._GammaRHFBuilder.__call__, driver._GammaRHFBuilder.__init__, _scf.solve_operator

    def capture_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        instances.append(self)

    def capture_build(self, *args):
        state = call(self, *args)
        builds.append(state)
        return state

    def capture_solve(operator, *args, **kwargs):
        solvers.append(operator)
        return solve(operator, *args, **kwargs)

    monkeypatch.setattr(driver._GammaRHFBuilder, "__init__", capture_init)
    monkeypatch.setattr(driver._GammaRHFBuilder, "__call__", capture_build)
    monkeypatch.setattr(_scf, "solve_operator", capture_solve)
    run = _run_ewald3d if dim == 3 else vq.run_rhf_periodic_gamma_ewald2d
    result = run(system, basis, options, omega=.5, grid_shape=(24, 24, 24), progress=False)
    state, builder = result.fock_build, instances[0]
    assert result.converged == (max_iter > 2)
    assert state is builds[-1]
    assert [s.generation for s in builds] == list(range(len(builds)))
    assert len({s.source.identity for s in builds}) == 1
    assert len(result.evaluated_build_history) == result.n_iter
    assert result.evaluated_build_history[-1]["build_id"] == state.identity
    assert result.energy == state.energy.total == result.scf_trace[-1].energy
    assert result.scf_trace[-1].grad_norm == pytest.approx(state.commutator_norm)
    assert state.metric_electron_counts == pytest.approx((2.,))
    if quadratic:
        assert result.mo_coeffs.shape[1] == state.channels[0].operator.frame.retained_rank
    np.testing.assert_array_equal(result.density, state.channels[0].density)
    np.testing.assert_array_equal(result.fock, state.channels[0].operator.matrix)
    np.testing.assert_allclose(result.density,
        (result.mo_coeffs * result.occupations) @ result.mo_coeffs.T, atol=1e-12)
    np.testing.assert_allclose(result.mo_energies,
        np.diag(result.mo_coeffs.T @ result.fock @ result.mo_coeffs), atol=1e-12)
    assert state.energy.electronic == pytest.approx(
        .5 * np.einsum("ij,ij->", result.density, builder.hcore + result.fock), abs=1e-12)
    assert result.density_fixed_point_residual == pytest.approx(
        np.max(abs(result.next_density - result.density)))
    if max_iter == 1:
        assert result.density_fixed_point_residual > 1e-5
    by_id = {s.identity: s for s in builds}
    assert solvers[0].metadata.kind == "one_electron"
    for op in solvers[1:]:
        parent = by_id[op.metadata.build_id]
        assert op.frame is parent.channels[0].operator.frame
        assert op.metadata.density_id == parent.density_id
        if op.metadata.role == "solver":
            assert op.metadata.modification

    stem = tmp_path / "evaluated_rhf"
    plan = OutputPlan.from_run_job_kwargs(output=stem, method="rhf", basis=basis.name, functional=None)
    path = write_qvf(stem, plan, molecule=system.unit_cell_molecule(), system=system,
        result=result, method="rhf", basis=basis.name, record_hostname=False,
        wf_data=qvf_wf_data(result, basis, system.unit_cell_molecule()),
        scf_history=scf_history_from_result(result))
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["provenance"]["run_fields"]["fock_build_id"] == state.identity
    assert manifest["provenance"]["scf_energy"]["value"] == state.energy.total

    # The restricted energy gradient is the physical Fock for total D.
    rng = np.random.default_rng(472)
    q = rng.normal(size=result.density.shape)
    direction = .05 * (q + q.T)
    if dim == 2:
        # Keep the slab neutral in the finite-difference direction.
        direction -= np.trace(direction @ result.overlap) / 2. * result.density
        assert dict(state.energy.corrections)["madelung"] == 0.
        details = builder.source_details()
        assert details["nuclear_boundary"] == "open-normal-parry"
        assert details["hartree_zero_mode"] == "parry-z-profile"
        assert dict(details["nuclear_ewald"])["alpha"] == .5
        assert details["grid_shape"] is None
    step = 1e-5
    plus, minus = builder(result.density + step * direction), builder(result.density - step * direction)
    derivative = (plus.energy.total - minus.energy.total) / (2 * step)
    assert derivative == pytest.approx(np.einsum("ij,ij->", result.fock, direction), abs=2e-9)
    old_fock = state.channels[0].operator.matrix.copy()
    old_overlap = state.channels[0].operator.frame.overlap.copy()
    result.fock[:] = 0.
    result.density[:] = 0.
    result.overlap[:] = 0.
    np.testing.assert_array_equal(state.channels[0].operator.matrix, old_fock)
    np.testing.assert_array_equal(state.channels[0].operator.frame.overlap, old_overlap)
    monkeypatch.setenv("VIBEQC_J_EWALD3D_KE", "201.0")
    if dim == 2:
        # Unused 3D controls do not change the slab identity; actual resolved
        # lattice changes do, before any cached J or fresh K evaluation.
        assert builder.source_details() == details
        builder.lattice_options.slab_ewald_alpha = .6
    with pytest.raises(ValueError, match="numerical source changed"):
        builder(state.channels[0].density)


def test_slab_rhf_max_iter_zero_evaluates_the_guess():
    system, basis, _ = _slab_he_setup(0)
    _assert_noiter_is_the_guess(
        lambda options: vq.run_rhf_periodic_gamma_ewald2d(system, basis, options),
        lambda n: _slab_he_setup(n)[2])


def test_rhf_max_iter_zero_evaluates_the_guess():
    system, basis = _h2_in_box()
    _assert_noiter_is_the_guess(
        lambda options: _run_ewald3d(system, basis, options),
        lambda n: _default_options(iter_limit=n))


def _assert_noiter_is_the_guess(run, set_iter):
    """max_iter=0 (NOITER, #649): the guess energy, exactly the first trace
    entry of an iterating run, explicitly unconverged with no iteration."""
    zero = run(set_iter(0))
    assert (zero.converged, zero.n_iter, list(zero.scf_trace)) == (False, 0, [])
    assert zero.energy == run(set_iter(2)).scf_trace[0].energy
    with pytest.raises(ValueError, match="max_iter must be >= 0"):
        run(set_iter(-1))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _h2_in_box(box: float = 30.0):
    """H₂ centered in a cubic box of given side (bohr). Returns
    ``(system, basis)``. Default 30 bohr matches 12e-c-4a tests."""
    c = box / 2
    atoms = [
        vq.Atom(1, [c, c, c - 0.7]),
        vq.Atom(1, [c, c, c + 0.7]),
    ]
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _default_options(iter_limit: int = 60, damping: float = 0.3):
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = damping
    opts.max_iter = iter_limit
    # DIIS converges these H₂/Ewald SCFs in ~3 iterations instead of ~17
    # (identical fixed point), so the per-iteration analytic-FT J rebuild
    # is paid 3× not 17× — the dominant cost here. The old "driver doesn't
    # implement DIIS yet" comment was stale; the Γ-Ewald DIIS path is the
    # subject of test_periodic_rhf_ewald_diis.py.
    opts.use_diis = True
    return opts


# ---------------------------------------------------------------------------
# Convergence on H2
# ---------------------------------------------------------------------------


def test_h2_converges():
    """The SCF terminates successfully on a clean test case."""
    sysp, basis = _h2_in_box(box=30.0)
    opts = _default_options()
    r = _run_ewald3d(sysp, basis, opts, omega=0.5)
    assert r.converged, f"did not converge after {r.n_iter} iterations"
    assert r.n_iter <= opts.max_iter
    # Energy is finite and sensible (H2 RHF is ~ -1 Ha).
    assert -2.0 < r.energy < 0.0


# ---------------------------------------------------------------------------
# ω-invariance (the headline correctness witness)
# ---------------------------------------------------------------------------


def test_energy_is_omega_invariant_on_h2():
    """SCF total energy must be independent of ω to ~0.3 % — this is
    the deep contract of the Ewald decomposition. Any mismatch at the
    mHa scale on a 30-bohr H₂ box indicates a bug in the composed-J
    pipeline (which was validated separately in 12e-c-4a at the matrix
    level)."""
    sysp, basis = _h2_in_box(box=30.0)
    opts = _default_options()

    energies = []
    for omega in (0.3, 0.5, 1.0, 1.5):
        r = _run_ewald3d(
            sysp,
            basis,
            opts,
            omega=omega,
            spacing_bohr=0.3,
        )
        assert r.converged
        energies.append(r.energy)

    spread = max(energies) - min(energies)
    # Spread < 0.5 % of |E| (matching the 12e-c-4a finite-box
    # Makov-Payne bound).
    assert spread < 0.005 * abs(min(energies)), (
        f"ω-invariance violated: {energies}, spread = {spread:.3e}"
    )


# ---------------------------------------------------------------------------
# Makov–Payne consistency with DIRECT_TRUNCATED
# ---------------------------------------------------------------------------


def test_energy_matches_direct_truncated_in_atomic_limit():
    """v0.6.1 Madelung-cancellation fix: after the SCF energy applies
    +α_M(Q_n²+Q_e²)/(2L), the Ewald-3D total energy in a big-enough
    box matches DIRECT_TRUNCATED to within the 12e-c-4a finite-box
    accuracy. Pre-fix, Ewald was off by ~ 0.5·α·n_electrons (the
    Makov-Payne shift). Post-fix that shift is removed by the
    Madelung correction; Ewald and DIRECT both converge to the
    molecular-limit total energy at large L.
    """
    box = 30.0
    sysp, basis = _h2_in_box(box=box)
    opts = _default_options()

    # The direct route needs its cutoff balls to reach the first image
    # shell: at cutoff 12 a 30-bohr box leaves only g = 0, which the #133
    # guard refuses (#430). Both balls at box + 1 hold the same six
    # images (the next shell, 42.4 bohr, is outside), so the direct sum
    # stays neutral and still sits in the atomic limit.
    opts_direct = _default_options()
    opts_direct.lattice_opts.cutoff_bohr = box + 1.0
    opts_direct.lattice_opts.nuclear_cutoff_bohr = box + 1.0

    r_direct = vq.run_rhf_periodic_gamma(sysp, basis, opts_direct)
    r_ewald = _run_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    assert r_direct.converged
    assert r_ewald.converged

    diff = r_ewald.energy - r_direct.energy
    # 10 mHa tolerance — at L=30 the residual is ~3 mHa from the
    # spread-density vs point-charge Madelung approximation.
    assert abs(diff) < 1e-2, (
        f"Ewald - Direct mismatch in atomic limit: "
        f"E_ewald = {r_ewald.energy:.6f}, E_direct = {r_direct.energy:.6f}, "
        f"diff = {diff:.6f} Ha"
    )


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_result_has_expected_shapes():
    sysp, basis = _h2_in_box()
    opts = _default_options()
    r = _run_ewald3d(sysp, basis, opts, omega=0.5)
    n_bf = basis.nbasis
    assert r.density.shape == (n_bf, n_bf)
    assert r.fock.shape == (n_bf, n_bf)
    assert r.overlap.shape == (n_bf, n_bf)
    # Canonical orth may shrink MO space; columns ≤ n_bf.
    assert r.mo_coeffs.shape[0] == n_bf
    assert r.mo_coeffs.shape[1] <= n_bf
    assert r.mo_energies.shape == (r.mo_coeffs.shape[1],)
    # scf_trace entries are SCFIteration dataclass instances (the same
    # type the C++ DIRECT_TRUNCATED driver emits — see scf_trace
    # type unification commit).
    assert len(r.scf_trace) >= 1
    first = r.scf_trace[0]
    assert isinstance(first, vq.SCFIteration)
    assert first.iter >= 1
    assert isinstance(first.energy, float)
    assert isinstance(first.delta_e, float)
    assert isinstance(first.grad_norm, float)
    assert r.grid_shape == (100, 100, 100)  # auto_grid(30 bohr, 0.3 spacing)
    # ω = 0.5 passed explicitly; not auto-derived.
    assert r.omega == pytest.approx(0.5)


def test_result_matrices_are_symmetric():
    sysp, basis = _h2_in_box()
    opts = _default_options()
    r = _run_ewald3d(sysp, basis, opts, omega=0.5)
    assert np.allclose(r.overlap, r.overlap.T, atol=1e-12)
    assert np.allclose(r.fock, r.fock.T, atol=1e-8)
    assert np.allclose(r.density, r.density.T, atol=1e-8)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_accepts_non_orthorhombic_lattice_smoke():
    """Skew cells run through the native FFT metric path."""
    lat = np.array(
        [
            [30.0, 1.5, 0.0],
            [0.0, 30.0, 0.0],
            [0.0, 0.0, 30.0],
        ]
    )
    c = 15.0
    atoms = [
        vq.Atom(1, [c, c, c - 0.7]),
        vq.Atom(1, [c, c, c + 0.7]),
    ]
    sysp = vq.PeriodicSystem(3, lat, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _default_options(iter_limit=1)
    r = _run_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        grid_shape=(8, 8, 8),
    )
    assert r.n_iter == 1
    assert np.isfinite(r.energy)


def test_rejects_open_shell_system():
    """Closed-shell RHF driver must reject odd-electron systems."""
    # H atom (1 electron, multiplicity 2) in a box
    c = 15.0
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * 30.0,
        [vq.Atom(1, [c, c, c])],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _default_options()
    with pytest.raises(ValueError, match="closed-shell"):
        _run_ewald3d(sysp, basis, opts)


def test_auto_enforces_ewald3d_gauge():
    """A1 regression: EWALD_3D driver auto-sets coulomb_method=EWALD_3D."""
    a = 4.084 / 0.529177210903
    system = vq.PeriodicSystem(
        3,
        np.diag([a, a, a]),
        [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [0.5 * a, 0.5 * a, 0.5 * a])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.max_iter = 10
    opts.conv_tol_energy = 1e-7
    opts.use_diis = True  # fewer iterations on the tight LiH-primitive cell
    result = _run_ewald3d(system, basis, opts, progress=False)
    assert opts.lattice_opts.coulomb_method == vq.CoulombMethod.EWALD_3D
    assert -15.0 < result.energy < -1.0, (
        f"EWALD_3D energy out of physical range: {result.energy:.3f} Ha"
    )


@pytest.mark.parametrize("method,dim", [("rhf", 3), ("rhf", 2), ("rks", 3), ("rks", 2)])
@pytest.mark.parametrize("seed", ["HCORE", "SAD", "PATOM"])
@pytest.mark.parametrize("scalar", ["real", "complex_guess", "complex_frame", "complex_fock", "inactive_guess"])
def test_closed_shell_ewald_real_only_admission(monkeypatch, method, dim, seed, scalar):
    """Exercise each driver through seed admission, without J/K or an SCF."""
    import importlib
    from contextlib import nullcontext
    from types import SimpleNamespace
    from vibeqc import guess, linear_dependence, periodic_v_ne
    from vibeqc.solvers import _scf

    driver = importlib.import_module(f"vibeqc.periodic_{method}_ewald")
    options = vq.PeriodicRHFOptions() if method == "rhf" else vq.PeriodicKSOptions()
    options.initial_guess = getattr(vq.InitialGuess, seed)
    options.use_davidson = scalar != "inactive_guess"
    options.davidson_min_dim = 0
    options.davidson.n_eig = options.davidson.n_guess = 2
    options.davidson.guess_vectors = np.eye(2)
    if scalar in {"complex_guess", "inactive_guess"}:
        options.davidson.guess_vectors_cplx = np.eye(2) * np.exp(.4j)
    if method == "rks":
        options.use_periodic_becke = False

    system = SimpleNamespace(dim=dim, multiplicity=1, lattice=np.eye(3)*30,
                             n_electrons=lambda: 2, unit_cell_molecule=lambda: None)
    basis = SimpleNamespace(name="synthetic", nbasis=2, nshells=2)
    log = SimpleNamespace(level=0, info=lambda *a: None, write_raw=lambda *a: None,
                          stage=lambda *a, **k: nullcontext())
    monkeypatch.setattr(driver, "resolve_progress", lambda *a, **k: log)
    monkeypatch.setattr(driver, "_refuse_if_dense_ionic", lambda *a: None)
    monkeypatch.setattr(guess, "select_periodic_driver_guess", lambda *a, **k:
                        SimpleNamespace(effective=options.initial_guess))
    monkeypatch.setattr(driver, "compute_overlap_lattice", lambda *a: np.eye(2))
    monkeypatch.setattr(driver, "compute_kinetic_lattice", lambda *a: np.diag([-1., 1.]))
    monkeypatch.setattr(periodic_v_ne, "compute_nuclear_lattice_dispatch", lambda *a, **k: np.zeros((2, 2)))
    monkeypatch.setattr(driver, "bloch_sum", lambda matrix, *a: matrix)
    monkeypatch.setattr(driver, "nuclear_repulsion_per_cell", lambda *a: 0.)
    monkeypatch.setattr(linear_dependence, "scf_preflight_overlap_check", lambda *a, **k: None)
    if method == "rhf":
        monkeypatch.setattr(driver, "scf_preflight_overlap_check", lambda *a, **k: None)
    else:
        monkeypatch.setattr(driver, "build_grid", lambda *a: None)
        monkeypatch.setattr(driver, "Functional", lambda *a: None)
        monkeypatch.setattr(driver, "resolve_periodic_exchange", lambda *a, **k: None)
    if scalar == "complex_frame":
        monkeypatch.setattr(driver, "_canonical_orthogonalizer", lambda *a, **k: (np.eye(2)*np.exp(.4j), 2))
    monkeypatch.setattr(driver, "initial_density_closed_shell", lambda *a, **k:
                        None if seed == "HCORE" else np.diag([2., 0.]))

    setup = []
    def no_jk(*args, **kwargs):
        pytest.fail("Scalar admission must precede a J/K evaluation")
    def builder(*args, **kwargs):
        setup.append("J factory")
        return no_jk
    monkeypatch.setattr(driver, "make_ewald_3d_gamma_j_builder", builder)
    monkeypatch.setattr(driver, "make_slab_ewald_2d_gamma_j_builder", builder)
    monkeypatch.setattr(driver, "build_jk_gamma_molecular_limit", no_jk)

    # The numerical backend is outside this admission contract. Use its dense
    # implementation for the real control while retaining adapter dispatch.
    solve = _scf.solve_operator
    def dense_control(operator, options, *, solver, previous):
        assert solver == ("dense" if scalar == "inactive_guess" else "davidson")
        return solve(operator, _scf.SolverOptions(n_roots=options.n_roots), solver="dense")
    monkeypatch.setattr(_scf, "solve_operator", dense_control)
    class SeedAdmitted(Exception):
        pass
    diagonalise = _scf.FockDiagonalizer.__call__
    def first_seed(self, fock):
        if scalar == "complex_fock":
            fock = fock.astype(complex)
            fock[0, 1], fock[1, 0] = .2j, -.2j
        coefficients, energies = diagonalise(self, fock)
        assert not np.iscomplexobj(coefficients)
        np.testing.assert_allclose(coefficients.T @ coefficients, np.eye(2), atol=1e-12)
        assert 2*np.trace(coefficients[:, :1] @ coefficients[:, :1].T) == pytest.approx(2.)
        raise SeedAdmitted
    monkeypatch.setattr(_scf.FockDiagonalizer, "__call__", first_seed)
    run = getattr(driver, f"run_{method}_periodic_gamma_ewald{dim}d")
    kwargs = dict(omega=.5, grid_shape=(2, 2, 2))
    if dim == 3:
        kwargs["auto_optimize_truncation"] = False
    errors = {"complex_guess": "complex Davidson guesses", "complex_frame": "real X and overlap",
              "complex_fock": "complex Fock matrices"}
    if scalar in errors:
        with pytest.raises(NotImplementedError, match=errors[scalar]):
            run(system, basis, options, **kwargs)
        if scalar != "complex_fock":
            assert setup == []
    else:
        with pytest.raises(SeedAdmitted):
            run(system, basis, options, **kwargs)
        assert setup == ["J factory"]
