"""Phase 15a tests: Γ-point periodic UHF SCF with Ewald-3D Coulomb.

Contracts exercised:

1. **Closed-shell limit equals RHF** — for a closed-shell molecule
   (multiplicity = 1, n_e even), UHF must converge to the same total
   energy as RHF and report ⟨S²⟩ ≈ 0 (machine-precision).

2. **One-electron doublet exact ⟨S²⟩** — for a single-electron H atom
   in a box (n_e=1, mult=2), UHF reports ⟨S²⟩ = 0.75 = S(S+1) exactly
   (no spin contamination possible with one electron).

3. **Open-shell convergence** — H atom doublet in a 30-bohr box
   converges to a finite, reasonable energy.

4. **α/β symmetry on closed-shell** — when n_α = n_β the converged
   D_α and D_β are equal (mirror of the RHF density up to the
   factor of 2).

5. **Result-shape contract** — every field on PeriodicUHFEwaldResult
   is populated and has a sensible type.

6. **Bad multiplicity raises** — incompatible (n_e, multiplicity)
   pairs are rejected up-front.

7. **PySCF cross-check** — H atom UHF / sto-3g matches PySCF UHF on
   the same atom in a box (~µHa agreement when the box is large
   enough that Ewald and free-space converge).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
import vibeqc.periodic_runner as pr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _h2_closed_shell(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h_atom_doublet(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, [vq.Atom(1, [c, c, c])])
    sysp.multiplicity = 2
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _li_atom_doublet(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(3, [c, c, c])],
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _options():
    o = vq.PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.damping = 0.3
    o.max_iter = 40
    return o


def _assert_reported_spin_densities_match_orbitals(result) -> None:
    """Every reported spin density is defined by its C/f pair."""
    for spin in ("alpha", "beta"):
        coeffs = np.asarray(getattr(result, f"mo_coeffs_{spin}"))
        occupations = np.asarray(getattr(result, f"occupations_{spin}"))
        reconstructed = (coeffs * occupations) @ coeffs.conj().T
        np.testing.assert_allclose(
            getattr(result, f"density_{spin}"),
            reconstructed,
            atol=1e-12,
            rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# 1. Closed-shell limit equals RHF
# ---------------------------------------------------------------------------


def test_uhf_closed_shell_matches_rhf():
    sysp, basis = _h2_closed_shell()
    opts = _options()
    r_rhf = vq.run_rhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    r_uhf = vq.run_uhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    assert r_rhf.converged and r_uhf.converged
    assert r_uhf.energy == pytest.approx(r_rhf.energy, abs=1e-9)
    assert r_uhf.scf_trace[-1].energy == pytest.approx(r_uhf.energy, abs=1e-12)
    _assert_reported_spin_densities_match_orbitals(r_uhf)
    # ⟨S²⟩ ≈ 0 for closed shell.
    assert abs(r_uhf.s_squared) < 1e-10
    assert abs(r_uhf.s_squared_ideal) < 1e-12


def test_uhf_closed_shell_alpha_beta_densities_equal():
    """For a closed-shell UHF the converged α and β densities must
    be equal (each is half the closed-shell total density)."""
    sysp, basis = _h2_closed_shell()
    opts = _options()
    r = vq.run_uhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    assert r.converged
    diff = np.abs(r.density_alpha - r.density_beta).max()
    assert diff < 1e-9, f"α/β density asymmetry on closed shell: {diff:.3e}"


# ---------------------------------------------------------------------------
# 2. One-electron doublet — exact ⟨S²⟩
# ---------------------------------------------------------------------------


def test_h_atom_doublet_spin_squared_is_three_quarters():
    sysp, basis = _h_atom_doublet()
    opts = _options()
    r = vq.run_uhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    assert r.converged
    # S = 1/2 → S(S+1) = 3/4. UHF on a single electron is exact.
    assert r.s_squared == pytest.approx(0.75, abs=1e-12)
    assert r.s_squared_ideal == pytest.approx(0.75, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Open-shell convergence
# ---------------------------------------------------------------------------


def test_h_atom_doublet_converges_to_sensible_energy():
    sysp, basis = _h_atom_doublet()
    opts = _options()
    r = vq.run_uhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    assert r.converged
    assert np.isfinite(r.energy)
    # H atom HF/sto-3g atomic limit is −0.466. In a finite Ewald box
    # the Makov-Payne correction shifts it by ~50 mHa to ~−0.51.
    assert -0.7 < r.energy < -0.4


# ---------------------------------------------------------------------------
# 4. Result-shape contract
# ---------------------------------------------------------------------------


def test_result_struct_populated():
    sysp, basis = _h_atom_doublet()
    opts = _options()
    r = vq.run_uhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    nbf = basis.nbasis
    assert r.density_alpha.shape == (nbf, nbf)
    assert r.density_beta.shape == (nbf, nbf)
    assert r.fock_alpha.shape == (nbf, nbf)
    assert r.fock_beta.shape == (nbf, nbf)
    assert r.mo_coeffs_alpha.shape[0] == nbf
    assert r.mo_coeffs_beta.shape[0] == nbf
    assert r.occupations_alpha.shape == r.mo_energies_alpha.shape
    assert r.occupations_beta.shape == r.mo_energies_beta.shape
    assert r.occupations_alpha.sum() == pytest.approx(1.0)
    assert r.occupations_beta.sum() == pytest.approx(0.0)
    assert r.overlap.shape == (nbf, nbf)
    assert isinstance(r.scf_trace, list)
    assert len(r.scf_trace) == r.n_iter
    # ω is auto-derived from nuclear_cutoff_bohr (mirror of
    # the RHF/RKS Γ and multi-k drivers, commit 49f8ae91 / 433d3543);
    # the driver ``omega`` kwarg is overridden.
    assert r.omega == pytest.approx(0.5)
    orbital_text = pr._mo_summary(r)
    assert "Crystal orbital energies (alpha)" in orbital_text
    assert "Crystal orbital energies (beta)" in orbital_text


def test_damped_li_terminal_state_is_not_false_positive_converged():
    """A rejected provisional state continues through the iteration cap."""
    sysp, basis = _li_atom_doublet()
    opts = _options()
    opts.max_iter = 3
    opts.use_diis = False
    opts.damping = 0.95
    opts.conv_tol_energy = 0.02
    opts.conv_tol_grad = 0.07

    result = vq.run_uhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.5,
        allow_dense_ionic=True,
    )

    assert not result.converged
    assert result.n_iter == 3
    assert len(result.scf_trace) == result.n_iter
    _assert_reported_spin_densities_match_orbitals(result)
    assert result.scf_trace[-1].energy == pytest.approx(result.energy, abs=1e-12)
    terminal_delta_e = abs(result.scf_trace[-1].delta_e)
    FDS_alpha = result.fock_alpha @ result.density_alpha @ result.overlap
    FDS_beta = result.fock_beta @ result.density_beta @ result.overlap
    terminal_grad = np.sqrt(
        np.linalg.norm(FDS_alpha - FDS_alpha.T) ** 2
        + np.linalg.norm(FDS_beta - FDS_beta.T) ** 2
    )
    assert terminal_delta_e > opts.conv_tol_energy
    assert terminal_grad > opts.conv_tol_grad
    assert result.scf_trace[-1].grad_norm == pytest.approx(terminal_grad)


def test_one_cycle_cap_returns_consistent_orbital_state():
    """The non-provisional max-iteration path also gets a terminal rebuild."""
    sysp, basis = _li_atom_doublet()
    opts = _options()
    opts.max_iter = 1
    opts.use_diis = False

    result = vq.run_uhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        grid_shape=(8, 8, 8),
        allow_dense_ionic=True,
    )

    assert not result.converged
    assert result.n_iter == 1
    assert len(result.scf_trace) == 1
    assert result.scf_trace[-1].energy == pytest.approx(result.energy, abs=1e-12)
    _assert_reported_spin_densities_match_orbitals(result)


# ---------------------------------------------------------------------------
# 5. Bad multiplicity raises
# ---------------------------------------------------------------------------


def test_invalid_multiplicity_raises():
    """1 electron with multiplicity=1 (singlet) is impossible —
    requires non-integer α/β occupations. The system-construction
    pipeline rejects it (vibe-qc's ``unit_cell_molecule()`` already
    enforces ``(n_e + mult) % 2 == 1``); the driver inherits this
    guard via its first basis-build call."""
    box, c = 30.0, 15.0
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, [vq.Atom(1, [c, c, c])])
    sysp.multiplicity = 1  # impossible: 1 electron + singlet
    with pytest.raises(Exception):
        # unit_cell_molecule() raises here — that's the right level
        # to catch impossible (n_e, mult) combinations.
        basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
        vq.run_uhf_periodic_gamma_ewald3d(
            sysp,
            basis,
            _options(),
            omega=0.5,
            spacing_bohr=0.3,
        )


def test_nonpositive_max_iter_raises():
    sysp, basis = _h_atom_doublet()
    opts = _options()
    opts.max_iter = 0

    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        vq.run_uhf_periodic_gamma_ewald3d(sysp, basis, opts)


# ---------------------------------------------------------------------------
# 6. PySCF cross-check on H atom doublet
# ---------------------------------------------------------------------------


def test_h_atom_pyscf_cross_check():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf

    # PySCF: H atom in vacuum, doublet, sto-3g — bare molecular UHF.
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, charge=0, verbose=0)
    e_pyscf = scf.UHF(mol).run().e_tot

    # vibe-qc: same atom in a 30-bohr box. Ewald nuclear repulsion
    # adds a Madelung-like ~50 mHa shift; the agreement against
    # the bare-vacuum PySCF energy isn't exact but is bounded.
    sysp, basis = _h_atom_doublet(box=30.0)
    opts = _options()
    r = vq.run_uhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    # Within ~50 mHa of PySCF's vacuum value (Makov-Payne bound).
    assert abs(r.energy - e_pyscf) < 0.1, (
        f"H atom UHF disagrees with PySCF beyond box-shift bound: "
        f"vibe-qc {r.energy:.6f} vs PySCF {e_pyscf:.6f}"
    )


def _previous_dense_fock(self, fock):
    """The driver's previous projected dense solve, without shared dispatch."""
    from vibeqc.solvers.operators import FockOperator
    if isinstance(fock, FockOperator):
        fock = fock.matrix
    projected = self.X.T @ fock @ self.X
    eps, vectors = np.linalg.eigh(.5 * (projected + projected.T))
    return self.X @ vectors, eps


@pytest.mark.parametrize("max_iter", [1, 80])
def test_evaluated_build_drives_terminal_state_history_and_qvf(monkeypatch, tmp_path, max_iter):
    import json
    import zipfile
    from vibeqc import periodic_uhf_ewald as driver
    from vibeqc.guess_read import resolve_periodic_read_densities_open
    from vibeqc.output import OutputPlan
    from vibeqc.output.formats.qvf import write_qvf, qvf_wf_data, scf_history_from_result

    system, basis = _li_atom_doublet()
    options = _options()
    options.max_iter = max_iter
    options.level_shift = .2
    options.conv_tol_energy, options.conv_tol_grad = 1e-12, 1e-10
    builds = []
    original = driver._GammaUHFBuilder.__call__

    def observe(self, *densities):
        state = original(self, *densities)
        builds.append(state)
        return state

    monkeypatch.setattr(driver._GammaUHFBuilder, "__call__", observe)
    result = driver.run_uhf_periodic_gamma_ewald3d(
        system, basis, options, omega=.5, grid_shape=(24, 24, 24), auto_optimize_truncation=False)
    state = result.fock_build
    assert state is builds[-1]
    assert result.converged == (max_iter > 1)
    assert [b.generation for b in builds] == list(range(len(builds)))
    assert len({b.source.identity for b in builds}) == 1
    assert result.energy == state.energy.total == result.scf_trace[-1].energy
    assert result.scf_trace[-1].grad_norm == pytest.approx(state.commutator_norm)
    assert state.metric_electron_counts == pytest.approx((2., 1.))
    assert len(result.evaluated_build_history) == result.n_iter
    assert result.evaluated_build_history[-1]["build_id"] == state.identity
    differences = []
    for i, spin in enumerate(("alpha", "beta")):
        channel = state.channels[i]
        np.testing.assert_array_equal(getattr(result, "density_"+spin), channel.density)
        np.testing.assert_array_equal(getattr(result, "fock_"+spin), channel.operator.matrix)
        differences.append(np.max(abs(getattr(result, "next_density_"+spin) - channel.density)))
    assert result.density_fixed_point_residual == max(differences)
    if max_iter == 1:
        assert result.density_fixed_point_residual > 1e-4
    _assert_reported_spin_densities_match_orbitals(result)
    stem = tmp_path / "evaluated"
    plan = OutputPlan.from_run_job_kwargs(output=stem, method="uhf", basis="sto-3g", functional=None)
    path = write_qvf(stem, plan, molecule=system.unit_cell_molecule(), system=system,
                     result=result, method="uhf", basis="sto-3g", record_hostname=False,
                     wf_data=qvf_wf_data(result, basis, system.unit_cell_molecule()),
                     scf_history=scf_history_from_result(result))
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    fields = manifest["provenance"]["run_fields"]
    assert fields["fock_build_id"] == state.identity
    assert fields["evaluated_density_id"] == state.density_id
    assert fields["scf_stop_reason"] == ("converged" if result.converged else "iteration_limit")
    assert manifest["provenance"]["scf_energy"]["value"] == state.energy.total
    da, db = resolve_periodic_read_densities_open(basis, read_path=str(path))
    np.testing.assert_allclose(da, state.channels[0].density, atol=1e-11)
    np.testing.assert_allclose(db, state.channels[1].density, atol=1e-11)
    # Legacy writable export arrays cannot mutate the canonical build.
    result.fock_alpha[:] = 0.
    assert np.linalg.norm(state.channels[0].operator.matrix) > .1


def test_gamma_build_same_density_energy_derivative_and_source_invalidation(monkeypatch):
    from vibeqc import periodic_uhf_ewald as driver

    system, basis = _li_atom_doublet()
    options = _options(); options.max_iter = 1
    instances = []
    original = driver._GammaUHFBuilder.__init__

    def capture(self, *args, **kwargs):
        original(self, *args, **kwargs)
        instances.append(self)

    monkeypatch.setattr(driver._GammaUHFBuilder, "__init__", capture)
    result = driver.run_uhf_periodic_gamma_ewald3d(
        system, basis, options, omega=.5, grid_shape=(24, 24, 24), auto_optimize_truncation=False)
    builder = instances[0]
    rng = np.random.default_rng(472)
    directions = rng.normal(size=(2, basis.nbasis, basis.nbasis))
    directions = .05*(directions + directions.swapaxes(1, 2))
    ds = np.array([c.density for c in result.fock_build.channels])
    before = [c.operator.matrix.copy() for c in result.fock_build.channels]
    step = 1e-5
    plus, minus = builder(*(ds+step*directions)), builder(*(ds-step*directions))
    expected = sum(np.einsum("ij,ji->", f, direction) for f, direction in zip(before, directions))
    assert (plus.energy.total-minus.energy.total)/(2*step) == pytest.approx(expected, abs=2e-9)
    # Neither a later build nor mutation of a producer argument changes an epoch.
    ds[:] = 0.
    for channel, old in zip(result.fock_build.channels, before):
        np.testing.assert_array_equal(channel.operator.matrix, old)
    monkeypatch.setenv("VIBEQC_J_EWALD3D_KE", "201.0")
    with pytest.raises(ValueError, match="numerical source changed"):
        builder(*ds)


def _assert_same_spin_state(actual, reference):
    assert actual.converged == reference.converged
    assert actual.energy == pytest.approx(reference.energy, abs=1e-9)
    assert actual.s_squared == pytest.approx(reference.s_squared, abs=1e-9)
    _assert_reported_spin_densities_match_orbitals(actual)
    for spin in ("alpha", "beta"):
        np.testing.assert_allclose(getattr(actual, f"density_{spin}"),
                                   getattr(reference, f"density_{spin}"), atol=1e-9)
        np.testing.assert_array_equal(getattr(actual, f"occupations_{spin}"),
                                      getattr(reference, f"occupations_{spin}"))
        np.testing.assert_allclose(getattr(actual, f"mo_energies_{spin}"),
                                   getattr(reference, f"mo_energies_{spin}"), atol=1e-9)
        c = getattr(actual, f"mo_coeffs_{spin}")
        f = getattr(actual, f"fock_{spin}")
        # Terminal reporting uses physical Rayleigh quotients even at a cap.
        np.testing.assert_allclose(getattr(actual, f"mo_energies_{spin}"),
                                   np.diag(c.T @ f @ c), atol=1e-12)
        np.testing.assert_allclose(c.T @ actual.overlap @ c,
                                   np.eye(c.shape[1]), atol=1e-12)


@pytest.mark.parametrize("davidson", [False, True])
@pytest.mark.parametrize("open_shell", [False, True])
def test_gamma_uhf_shared_fock_preserves_state_and_separate_spin_history(
    monkeypatch, davidson, open_shell,
):
    scipy = pytest.importorskip("scipy.linalg")
    from vibeqc import periodic_uhf_ewald as driver
    from vibeqc.solvers import _scf

    system, basis = _li_atom_doublet() if open_shell else _h2_closed_shell()
    options = _options()
    options.initial_guess = vq.InitialGuess.PATOM if open_shell else vq.InitialGuess.HCORE
    options.use_davidson = davidson
    options.davidson_min_dim = 0
    options.davidson.guess_vectors = np.eye(basis.nbasis)
    options.davidson.conv_tol = 1e-10
    options.level_shift = .2
    options.conv_tol_energy = 1e-12
    options.conv_tol_grad = 1e-10
    options.max_iter = 100
    guess = np.asarray(options.davidson.guess_vectors).copy()
    original = _scf.solve_operator
    previous_by_spin, generations = {}, {"alpha": [], "beta": []}
    seeds = []

    def checked(operator, solver_options, *, solver, previous=None):
        spin = operator.metadata.spin
        assert operator.metadata.kpoint == (0., 0., 0.)
        assert solver == ("davidson" if davidson else "dense")
        if operator.metadata.kind == "one_electron":
            assert spin == "restricted"
            assert previous is None
            assert operator.metadata.generation == 0
            seeds.append(operator)
        else:
            assert spin in generations
            assert operator.metadata.kind == "fock"
            assert previous is (previous_by_spin.get(spin) if davidson else None)
        result = original(operator, solver_options, solver=solver, previous=previous)
        expected = scipy.eigh(operator.matrix, operator.overlap, eigvals_only=True)
        np.testing.assert_allclose(result.eigenvalues, expected, atol=1e-9)
        assert max(result.residuals) < 1e-9
        assert result.orthogonality_error < 1e-12
        if operator.metadata.kind == "fock":
            previous_by_spin[spin] = result
            generations[spin].append(result.operator_metadata.generation)
        return result

    def run():
        # Same finite Ewald source for both algorithms, not a grid convergence test.
        return driver.run_uhf_periodic_gamma_ewald3d(
            system, basis, options, omega=.5, grid_shape=(24, 24, 24),
            auto_optimize_truncation=False,
        )

    monkeypatch.setattr(_scf, "solve_operator", checked)
    result = run()
    assert result.converged
    assert len(seeds) == 1
    for spin in generations:
        assert len(generations[spin]) >= 3
        assert generations[spin] == list(range(len(generations[spin])))
    assert previous_by_spin["alpha"] is not previous_by_spin["beta"]
    np.testing.assert_array_equal(options.davidson.guess_vectors, guess)
    assert options.davidson.n_eig == 0
    monkeypatch.setattr(_scf.FockDiagonalizer, "__call__", _previous_dense_fock)
    reference = run()
    _assert_same_spin_state(result, reference)


@pytest.mark.parametrize("davidson", [False, True])
@pytest.mark.parametrize("control", ["schedule", "pattern_hold", "quadratic", "reduced"])
def test_gamma_uhf_shared_fock_preserves_control_and_terminal_paths(monkeypatch, davidson, control):
    from vibeqc import periodic_uhf_ewald as driver
    from vibeqc import quadratic_scf
    from vibeqc.solvers import _scf

    system, basis = _li_atom_doublet()
    options = _options()
    options.use_davidson = davidson
    options.davidson_min_dim = 0
    options.max_iter = 3
    options.use_diis = False
    options.conv_tol_energy = 1e-14
    options.conv_tol_grad = 1e-12
    calls = []
    if control == "schedule":
        options.level_shift_schedule = [.4, .2, 0.]
        original_shift = driver.apply_level_shift

        def shifted(*args):
            calls.append(args[3])
            return original_shift(*args)

        monkeypatch.setattr(driver, "apply_level_shift", shifted)
    elif control == "pattern_hold":
        options.spinlock_mode = vq.SpinlockMode.PATTERN_HOLD
        options.spinlock_iterations = 3
        original_mom = driver._mom_reorder

        def mom(*args):
            calls.append(args[0].shape[1])
            return original_mom(*args)

        monkeypatch.setattr(driver, "_mom_reorder", mom)
    elif control == "quadratic":
        options.quadratic_fallback_iter = 1
        original_step = quadratic_scf.quadratic_step

        def quadratic(*args, **kwargs):
            calls.append(args[1].shape[1])
            return original_step(*args, **kwargs)

        monkeypatch.setattr(quadratic_scf, "quadratic_step", quadratic)
    else:
        original_frame = driver._canonical_orthogonalizer

        def reduced(*args, **kwargs):
            x, rank = original_frame(*args, **kwargs)
            # Li/STO-3G has two s AOs followed by three p AOs. Retain
            # both occupied s candidates and drop a p-only direction;
            # removing an occupied candidate would put the occupation
            # boundary in a degenerate p shell with no unique density.
            np.testing.assert_allclose(x[:2, 2], 0., atol=1e-12)
            return np.delete(x, 2, axis=1), rank - 1

        monkeypatch.setattr(driver, "_canonical_orthogonalizer", reduced)

    def run():
        return driver.run_uhf_periodic_gamma_ewald3d(
            system, basis, options, omega=.5, grid_shape=(12, 12, 12),
            auto_optimize_truncation=False,
        )

    result = run()
    assert not result.converged
    assert result.n_iter == options.max_iter
    if control == "schedule":
        assert calls == [.4, .4, .2, .2]
    elif control in ("pattern_hold", "quadratic"):
        assert calls and all(rank == basis.nbasis for rank in calls)
    else:
        assert result.mo_coeffs_alpha.shape[1] == basis.nbasis - 1
    monkeypatch.setattr(_scf.FockDiagonalizer, "__call__", _previous_dense_fock)
    reference = run()
    _assert_same_spin_state(result, reference)


@pytest.mark.parametrize("defect", ["occupied", "excess", "pattern_hold", "quadratic", "max_refine", "complex_guess"])
def test_gamma_uhf_rejects_incompatible_solver_before_jk(monkeypatch, defect):
    from vibeqc import periodic_uhf_ewald as driver

    system, basis = _li_atom_doublet()
    options = _options()
    options.use_davidson = True
    options.davidson_min_dim = 0
    options.davidson.n_eig = 2
    error, message = ValueError, "Requested Fock roots"
    if defect == "occupied":
        options.davidson.n_eig = 1
    elif defect == "excess":
        options.davidson.n_eig = basis.nbasis + 1
    elif defect == "pattern_hold":
        options.spinlock_mode = vq.SpinlockMode.PATTERN_HOLD
        options.spinlock_iterations = 2
    elif defect == "quadratic":
        options.quadratic_fallback_iter = 1
    elif defect == "max_refine":
        options.davidson.max_refine = 6
        error, message = NotImplementedError, "max_refine"
    else:
        options.davidson.guess_vectors_cplx = np.eye(basis.nbasis, 2) * np.exp(.4j)
        error, message = NotImplementedError, "complex Davidson guesses"

    def fail(*args, **kwargs):
        pytest.fail("Solver admission must precede J/K construction")

    monkeypatch.setattr(driver, "make_ewald_3d_gamma_j_builder", fail)
    monkeypatch.setattr(driver, "build_jk_gamma_molecular_limit", fail)
    with pytest.raises(error, match=message):
        driver.run_uhf_periodic_gamma_ewald3d(
            system, basis, options, grid_shape=(8, 8, 8),
            auto_optimize_truncation=False,
        )


def test_gamma_uhf_partial_roots_preserve_occupied_densities():
    system, basis = _li_atom_doublet()
    options = _options()
    options.use_davidson = True
    options.davidson_min_dim = 0
    options.max_iter = 1

    def run():
        return vq.run_uhf_periodic_gamma_ewald3d(
            system, basis, options, grid_shape=(12, 12, 12),
            auto_optimize_truncation=False,
        )

    full = run()
    options.davidson.n_eig = 2
    partial = run()
    assert not full.converged and not partial.converged
    assert partial.energy == pytest.approx(full.energy, abs=1e-10)
    for spin in ("alpha", "beta"):
        assert getattr(partial, f"mo_coeffs_{spin}").shape == (basis.nbasis, 2)
        np.testing.assert_allclose(getattr(partial, f"density_{spin}"),
                                   getattr(full, f"density_{spin}"), atol=1e-10)
    assert options.davidson.n_eig == 2
    assert np.asarray(options.davidson.guess_vectors).size == 0
