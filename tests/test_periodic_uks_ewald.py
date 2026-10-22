"""Phase 15c-3a: Γ-point periodic UKS SCF driver using EWALD_3D.

Open-shell DFT counterpart of the RKS Ewald tests + UHF Ewald tests.
Pin:

  1. H atom (mult=2) converges with ⟨S²⟩ = 3/4 = ideal.
  2. Closed-shell H₂ through UKS reproduces RKS to ~µHa
     (degenerate-spin-density limit).
  3. Hybrid (B3LYP) path exercises α > 0 — distinguishably non-zero
     ``e_hf_exchange``.
  4. ω-invariance.
  5. Result-shape / energy-decomposition consistency.
  6. Skew-cell FFT metric support.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
import vibeqc.periodic_runner as pr


def _h2_in_box(box: float = 30.0):
    c = box / 2
    atoms = [
        vq.Atom(1, [c, c, c - 0.7]),
        vq.Atom(1, [c, c, c + 0.7]),
    ]
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h_atom_in_box(box: float = 30.0):
    c = box / 2
    atoms = [vq.Atom(1, [c, c, c])]
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box, atoms, charge=0, multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _li_atom_in_box(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(3, [c, c, c])],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h2_slab(box: float = 18.0, vacuum: float = 45.0):
    c = box / 2
    atoms = [
        vq.Atom(1, [c, c, vacuum / 2 - 0.7]),
        vq.Atom(1, [c, c, vacuum / 2 + 0.7]),
    ]
    sysp = vq.PeriodicSystem(2, np.diag([box, box, vacuum]), atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h_atom_slab(box: float = 18.0, vacuum: float = 45.0):
    c = box / 2
    atoms = [vq.Atom(1, [c, c, vacuum / 2])]
    sysp = vq.PeriodicSystem(
        2,
        np.diag([box, box, vacuum]),
        atoms,
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _default_options(functional: str = "PBE", iter_limit: int = 60,
                     damping: float = 0.3):
    opts = vq.PeriodicKSOptions()
    opts.functional = functional
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = damping
    opts.max_iter = iter_limit
    opts.use_diis = True
    return opts


def _slab_options(functional: str = "PBE"):
    opts = _default_options(functional, iter_limit=50, damping=0.3)
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.lattice_opts.slab_ewald_alpha = 0.4
    opts.conv_tol_energy = 1e-7
    opts.conv_tol_grad = 1e-5
    return opts


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
# Convergence
# ---------------------------------------------------------------------------

def test_h_atom_converges_with_correct_spin():
    sysp, basis = _h_atom_in_box()
    opts = _default_options("PBE")
    r = vq.run_uks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert r.converged
    # Doublet ideal: S(S+1) = 0.75 with S=1/2.
    assert abs(r.s_squared - 0.75) < 1e-6
    assert abs(r.s_squared_ideal - 0.75) < 1e-12
    # H atom UKS PBE/STO-3G: ~ -0.5 Ha (after MP shift in 30-bohr box).
    assert -1.0 < r.energy < 0.0
    assert r.fock_build.metric_electron_counts == pytest.approx((1., 0.))
    np.testing.assert_array_equal(r.fock_build.channels[1].density, np.zeros((1, 1)))
    _assert_reported_spin_densities_match_orbitals(r)


def test_closed_shell_uks_matches_rks_to_microhartree():
    """Closed-shell H₂ with mult=1 → UKS should give the same answer
    as RKS to ~µHa (formally identical at the SCF fixed point)."""
    sysp, basis = _h2_in_box()
    opts = _default_options("PBE")
    r_uks = vq.run_uks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    r_rks = vq.run_rks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert r_uks.converged and r_rks.converged
    assert abs(r_uks.energy - r_rks.energy) < 1e-6
    assert r_uks.scf_trace[-1].energy == pytest.approx(r_uks.energy, abs=1e-12)
    _assert_reported_spin_densities_match_orbitals(r_uks)
    # Closed-shell ⟨S²⟩ → 0 (α and β occupied orbitals identical).
    assert abs(r_uks.s_squared) < 1e-6


# ---------------------------------------------------------------------------
# Hybrid path
# ---------------------------------------------------------------------------

def test_b3lyp_path_uses_hf_exchange():
    sysp, basis = _h2_in_box()
    opts = _default_options("B3LYP")
    r = vq.run_uks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert r.converged
    assert abs(r.e_hf_exchange) > 1e-3, (
        f"e_hf_exchange ≈ 0 ({r.e_hf_exchange:.3e}) but B3LYP α = 0.2"
    )


# ---------------------------------------------------------------------------
# ω-invariance
# ---------------------------------------------------------------------------

def test_omega_invariance_h_atom():
    sysp, basis = _h_atom_in_box()
    opts = _default_options("PBE")
    energies = []
    for omega in (0.3, 0.5, 1.0, 1.5):
        r = vq.run_uks_periodic_gamma_ewald3d(
            sysp, basis, opts, omega=omega, spacing_bohr=0.3,
        )
        assert r.converged
        energies.append(r.energy)
    spread = max(energies) - min(energies)
    # 1 % tolerance — single-electron H system has tighter sensitivity
    # to the FFT grid than H₂ but still well within finite-box bounds.
    assert spread < 0.01 * abs(min(energies)), (
        f"ω-invariance violated: {energies}, spread = {spread:.3e}"
    )


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def test_result_has_expected_shapes():
    sysp, basis = _h_atom_in_box()
    opts = _default_options("PBE")
    r = vq.run_uks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    n_bf = basis.nbasis
    assert r.density_alpha.shape == (n_bf, n_bf)
    assert r.density_beta.shape == (n_bf, n_bf)
    assert r.fock_alpha.shape == (n_bf, n_bf)
    assert r.fock_beta.shape == (n_bf, n_bf)
    assert r.occupations_alpha.shape == r.mo_energies_alpha.shape
    assert r.occupations_beta.shape == r.mo_energies_beta.shape
    assert r.occupations_alpha.sum() == pytest.approx(1.0)
    assert r.occupations_beta.sum() == pytest.approx(0.0)
    assert r.overlap.shape == (n_bf, n_bf)
    assert r.functional.upper().startswith("PBE")
    assert len(r.scf_trace) == r.n_iter
    orbital_text = pr._mo_summary(r)
    assert "Crystal orbital energies (alpha)" in orbital_text
    assert "Crystal orbital energies (beta)" in orbital_text


def test_damped_li_terminal_state_is_not_false_positive_converged():
    """A provisional damped iterate cannot certify its fresh density."""
    sysp, basis = _li_atom_in_box()
    opts = _default_options("PBE", iter_limit=2, damping=0.95)
    opts.use_diis = False
    opts.conv_tol_energy = 0.003
    opts.conv_tol_grad = 0.03

    result = vq.run_uks_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.5,
        allow_dense_ionic=True,
    )

    assert not result.converged
    assert result.n_iter == 2
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
    sysp, basis = _li_atom_in_box()
    opts = _default_options("PBE", iter_limit=1)
    opts.use_diis = False

    result = vq.run_uks_periodic_gamma_ewald3d(
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


def test_energy_decomposition_consistent():
    """E_total = e_electronic + e_nuclear for the EWALD_3D gauge.

    The Γ-only UKS Ewald driver now forces EWALD_3D (historical design record F4,
    2026-06-01), so V_ne / e_nuclear are Ewald-gauge-consistent with the
    Hartree J and the v0.6.1 Madelung leak correction is disabled
    (``madelung_energy_correction_for_lat`` returns 0.0 for EWALD_3D).
    Mirrors the RKS Γ decomposition test (test_periodic_rks_ewald.py).
    """
    sysp, basis = _h2_in_box()
    opts = _default_options("PBE")
    r = vq.run_uks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert r.converged
    # For EWALD_3D: E_total = E_elec + E_nuc (no Madelung fix).
    assert abs(r.energy - (r.e_electronic + r.e_nuclear)) < 1e-10
    # Pure DFT — no HF exchange.
    assert abs(r.e_hf_exchange) < 1e-12


# ---------------------------------------------------------------------------
# Slab Ewald
# ---------------------------------------------------------------------------

def test_slab_ewald_2d_h_atom_converges_with_correct_spin():
    sysp, basis = _h_atom_slab()
    opts = _slab_options("PBE")

    r = vq.run_uks_periodic_gamma_ewald2d(sysp, basis, opts)

    assert r.converged
    assert r.s_squared == pytest.approx(0.75, abs=1e-6)
    assert r.s_squared_ideal == pytest.approx(0.75, abs=1e-12)
    assert r.energy == pytest.approx(r.e_electronic + r.e_nuclear, abs=1e-10)
    assert r.omega == pytest.approx(opts.lattice_opts.slab_ewald_alpha)
    assert r.grid_shape == (0, 0, 0)


def test_slab_ewald_2d_closed_shell_uks_matches_rks():
    sysp, basis = _h2_slab()
    opts_uks = _slab_options("PBE")
    opts_rks = _slab_options("PBE")

    r_uks = vq.run_uks_periodic_gamma_ewald2d(sysp, basis, opts_uks)
    r_rks = vq.run_rks_periodic_gamma_ewald2d(sysp, basis, opts_rks)

    assert r_uks.converged and r_rks.converged
    assert r_uks.energy == pytest.approx(r_rks.energy, abs=1e-6)
    assert abs(r_uks.s_squared) < 1e-6


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_nonpositive_max_iter_raises():
    sysp, basis = _h_atom_in_box()
    opts = _default_options(iter_limit=0)

    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        vq.run_uks_periodic_gamma_ewald3d(sysp, basis, opts)


def test_accepts_non_orthorhombic_lattice_smoke():
    lat = np.array([
        [30.0, 1.5, 0.0],
        [0.0, 30.0, 0.0],
        [0.0, 0.0, 30.0],
    ])
    c = 15.0
    atoms = [vq.Atom(1, [c, c, c])]
    sysp = vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=2)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _default_options(iter_limit=1)
    r = vq.run_uks_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, grid_shape=(8, 8, 8),
    )
    assert r.n_iter == 1
    assert np.isfinite(r.energy)


def _uks_evaluated_setup(dim, functional, max_iter=80):
    # Neutral HeH has two occupied alpha orbitals and a changing beta
    # orbital in this two-function basis, without a slab charge background.
    system = vq.PeriodicSystem(dim, np.eye(3) * 30.,
        [vq.Atom(2, [15., 15., 14.3]), vq.Atom(1, [15., 15., 15.7])], multiplicity=2)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = _default_options(functional, iter_limit=max_iter)
    options.level_shift = .2
    options.use_periodic_becke = functional != "PBE0"
    options.initial_guess = vq.InitialGuess.PATOM
    options.conv_tol_energy, options.conv_tol_grad = 1e-12, 1e-10
    return system, basis, options


@pytest.mark.parametrize("dim,functional,max_iter,iterative,quadratic,hold", [
    (3, "PBE", 1, False, 0, False),
    (3, "PBE", 80, False, 0, False),
    (3, "PBE", 80, True, 0, False),
    (3, "PBE0", 2, True, 1, False),
    (3, "PBE0", 80, False, 0, False),
    (2, "hse06", 80, False, 0, False),
    (2, "hse06", 2, True, 0, True),
])
def test_uks_evaluated_state_terminal_and_qvf(
        monkeypatch, tmp_path, dim, functional, max_iter, iterative, quadratic, hold):
    """Same-density joint energy, spin solver histories and output identity."""
    import json
    import zipfile
    from vibeqc import periodic_uks_ewald as driver
    from vibeqc.output import OutputPlan
    from vibeqc.output.formats.qvf import write_qvf, qvf_wf_data, scf_history_from_result
    from vibeqc.solvers import _scf

    system, basis, options = _uks_evaluated_setup(dim, functional, max_iter)
    options.use_davidson, options.davidson_min_dim = iterative, 0
    options.davidson.conv_tol = 1e-12
    options.davidson.guess_vectors = np.eye(2)
    guess_before = np.asarray(options.davidson.guess_vectors).copy()
    options.quadratic_fallback_iter = quadratic
    if hold:
        options.spinlock_mode = vq.SpinlockMode.PATTERN_HOLD
        options.spinlock_iterations = 3
    builds, solvers, instances, returned = [], [], [], {}
    call, init, solve = driver._GammaUKSBuilder.__call__, driver._GammaUKSBuilder.__init__, _scf.solve_operator

    def capture_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        instances.append(self)

    def capture_build(self, *args, **kwargs):
        state = call(self, *args, **kwargs)
        builds.append(state)
        return state

    def capture_solve(operator, *args, **kwargs):
        key = operator.metadata.kind, operator.metadata.spin
        previous = kwargs.get("previous")
        if previous is not None:
            assert returned[id(previous)][0] == key
        elif key[0] == "ks" and iterative:
            assert not any(op.metadata.kind == "ks" and op.metadata.spin == key[1] for op in solvers)
        solvers.append(operator)
        result = solve(operator, *args, **kwargs)
        returned[id(result)] = key, result
        return result

    monkeypatch.setattr(driver._GammaUKSBuilder, "__init__", capture_init)
    monkeypatch.setattr(driver._GammaUKSBuilder, "__call__", capture_build)
    monkeypatch.setattr(_scf, "solve_operator", capture_solve)
    kwargs = dict(omega=.5, grid_shape=(24, 24, 24), progress=False)
    if dim == 3:
        kwargs["auto_optimize_truncation"] = False
    result = getattr(vq, f"run_uks_periodic_gamma_ewald{dim}d")(system, basis, options, **kwargs)
    state, builder = result.fock_build, instances[0]
    assert result.converged == (max_iter > 2)
    assert options.davidson.n_eig == 0
    np.testing.assert_array_equal(options.davidson.guess_vectors, guess_before)
    assert state is builds[-1]
    assert [s.generation for s in builds] == list(range(len(builds)))
    assert len({s.source.identity for s in builds[1:]}) == 1
    assert builds[0].method == "UHF" and state.method == "UKS"
    assert dict(builds[0].energy.components)["xc"] == 0.
    assert builds[0].source.identity != state.source.identity
    assert state.channels[0].operator.frame is state.channels[1].operator.frame
    assert dict(state.energy.components)["xc"] == result.e_xc
    source = builder.source_details()
    assert source["xc"]["spin"] == 2
    assert source["xc"]["cells"] == source["xc"]["beta_cells"]
    assert source["xc"]["density_convention"] == ("gamma-torus" if options.use_periodic_becke else "home-cell-only")
    assert source["exchange_assembly"]["omega_screen"] == pytest.approx(.11 if functional == "hse06" else 0.)
    assert source["hartree_zero_mode"] == ("parry-z-profile" if dim == 2 else "omitted-jellium")
    assert len(result.evaluated_build_history) == result.n_iter
    assert result.evaluated_build_history[-1]["build_id"] == state.identity
    assert result.energy == state.energy.total == result.scf_trace[-1].energy
    assert result.scf_trace[-1].grad_norm == pytest.approx(state.commutator_norm)
    assert state.metric_electron_counts == pytest.approx((2., 1.))
    assert result.s_squared == pytest.approx(.75, abs=1e-12)
    _assert_reported_spin_densities_match_orbitals(result)
    half_trace = 0.
    residuals = []
    for channel, spin in zip(state.channels, ("alpha", "beta")):
        c, d, f = (getattr(result, f"{name}_{spin}") for name in ("mo_coeffs", "density", "fock"))
        assert channel.operator.metadata.kind == "ks" and channel.operator.metadata.spin == spin
        np.testing.assert_array_equal(d, channel.density)
        np.testing.assert_array_equal(f, channel.operator.matrix)
        np.testing.assert_allclose(getattr(result, f"mo_energies_{spin}"), np.diag(c.T @ f @ c), atol=1e-12)
        half_trace += .5 * np.einsum("ij,ij->", d, builder.hcore + f)
        residuals.append(np.max(abs(getattr(result, f"next_density_{spin}") - d)))
        if quadratic or hold:
            assert c.shape[1] == channel.operator.frame.retained_rank
    assert abs(state.energy.electronic - half_trace) > 1e-3
    assert sum(dict(state.energy.components).values()) == pytest.approx(result.e_electronic, abs=1e-12)
    assert result.density_fixed_point_residual == pytest.approx(max(residuals))
    if max_iter == 1:
        assert result.density_fixed_point_residual > 1e-5
    by_id = {s.identity: s for s in builds}
    assert solvers[0].metadata.kind == "one_electron"
    assert [(op.metadata.kind, op.metadata.spin) for op in solvers[1:3]] == [("fock", "alpha"), ("fock", "beta")]
    for op in solvers[1:]:
        parent = by_id[op.metadata.build_id]
        channel = parent.channels[0 if op.metadata.spin == "alpha" else 1]
        assert op.frame is channel.operator.frame
        assert op.metadata.density_id == parent.density_id
        if op.metadata.role == "solver":
            assert op.metadata.modification
            if hold:
                assert "extrapolation" not in op.metadata.modification

    stem = tmp_path / "evaluated_uks"
    plan = OutputPlan.from_run_job_kwargs(output=stem, method="uks", basis="sto-3g", functional=functional)
    path = write_qvf(stem, plan, molecule=system.unit_cell_molecule(), system=system,
        result=result, method="uks", basis="sto-3g", record_hostname=False,
        wf_data=qvf_wf_data(result, basis, system.unit_cell_molecule()),
        scf_history=scf_history_from_result(result))
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["provenance"]["run_fields"]["fock_build_id"] == state.identity
    assert manifest["provenance"]["scf_energy"]["value"] == state.energy.total

    # Perturb one spin at a time: each physical spin Fock differentiates
    # the whole energy, including the polarized XC coupling to both spins.
    rng = np.random.default_rng(472)
    for index, channel in enumerate(state.channels):
        q = rng.normal(size=channel.density.shape)
        direction, step = .05 * (q + q.T), 1e-5
        ds = [c.density.copy() for c in state.channels]
        ds[index] += step * direction
        plus = builder(*ds)
        ds[index] -= 2 * step * direction
        minus = builder(*ds)
        derivative = (plus.energy.total - minus.energy.total) / (2 * step)
        assert derivative == pytest.approx(np.einsum("ij,ij->", channel.operator.matrix, direction), abs=2e-7)
    old_overlap = state.channels[0].operator.frame.overlap.copy()
    for channel, spin in zip(state.channels, ("alpha", "beta")):
        old_fock = channel.operator.matrix.copy()
        getattr(result, f"fock_{spin}")[:] = 0.
        getattr(result, f"density_{spin}")[:] = 0.
        np.testing.assert_array_equal(channel.operator.matrix, old_fock)
    result.overlap[:] = 0.
    np.testing.assert_array_equal(state.channels[0].operator.frame.overlap, old_overlap)
    for obj, name, replacement in ((options, "functional", "BLYP"),
            (options.grid, "n_radial", options.grid.n_radial + 1),
            (builder.lattice_options, "cutoff_bohr", builder.lattice_options.cutoff_bohr + 1)):
        previous = getattr(obj, name)
        setattr(obj, name, replacement)
        with pytest.raises(ValueError, match="numerical source changed"):
            builder(*(c.density for c in state.channels))
        setattr(obj, name, previous)


@pytest.mark.parametrize("dim", [3, 2])
@pytest.mark.parametrize("seed", ["PATOM"])
@pytest.mark.parametrize("scalar", ["real", "complex_guess", "complex_frame", "complex_fock", "inactive_guess",
                                    "truncated_occupied", "truncated_hold", "truncated_quadratic", "unsupported_refine"])
def test_uks_evaluated_state_real_only_admission(monkeypatch, dim, seed, scalar):
    """Exercise each driver through seed admission, without J/K or an SCF."""
    from vibeqc import periodic_uks_ewald as driver
    from contextlib import nullcontext
    from types import SimpleNamespace
    from vibeqc import guess, linear_dependence, periodic_v_ne
    from vibeqc.solvers import _scf

    options = vq.PeriodicKSOptions()
    options.initial_guess = getattr(vq.InitialGuess, seed)
    options.use_davidson = scalar != "inactive_guess"
    options.davidson_min_dim = 0
    options.davidson.n_eig = options.davidson.n_guess = 2
    options.davidson.guess_vectors = np.eye(2)
    if scalar in {"complex_guess", "inactive_guess"}:
        options.davidson.guess_vectors_cplx = np.eye(2) * np.exp(.4j)
    options.use_periodic_becke = False
    if scalar.startswith("truncated"):
        options.davidson.n_eig = 1
    if scalar == "truncated_hold":
        options.spinlock_mode, options.spinlock_iterations = vq.SpinlockMode.PATTERN_HOLD, 2
    if scalar == "truncated_quadratic":
        options.quadratic_fallback_iter = 1
    if scalar == "unsupported_refine":
        options.davidson.max_refine = 6

    system = SimpleNamespace(dim=dim, multiplicity=1, lattice=np.eye(3)*30,
                             n_electrons=lambda: 4 if scalar == "truncated_occupied" else 2, unit_cell_molecule=lambda: None)
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
    monkeypatch.setattr(driver, "build_grid", lambda *a: None)
    monkeypatch.setattr(driver, "Functional", lambda *a: None)
    monkeypatch.setattr(driver, "resolve_periodic_exchange", lambda *a, **k: None)
    if scalar == "complex_frame":
        monkeypatch.setattr(driver, "_canonical_orthogonalizer", lambda *a, **k: (np.eye(2)*np.exp(.4j), 2))
    monkeypatch.setattr(driver, "initial_densities_open_shell", lambda *a, **k:
                        None if seed == "HCORE" else (np.diag([1., 0.]), np.diag([1., 0.])))

    setup = []
    def no_jk(*args, **kwargs):
        pytest.fail("Scalar admission must precede a J/K evaluation")
    def builder(*args, **kwargs):
        setup.append("J factory")
        return no_jk
    monkeypatch.setattr(driver, "make_ewald_3d_gamma_j_builder", builder)
    monkeypatch.setattr(driver, "make_slab_ewald_2d_gamma_j_builder", builder)
    from vibeqc import periodic_ks_fock
    monkeypatch.setattr(periodic_ks_fock, "build_jk_gamma_molecular_limit", no_jk)

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
    run = getattr(driver, f"run_uks_periodic_gamma_ewald{dim}d")
    kwargs = dict(omega=.5, grid_shape=(2, 2, 2))
    if dim == 3:
        kwargs["auto_optimize_truncation"] = False
    errors = {"complex_guess": "complex Davidson guesses", "complex_frame": "real X and overlap",
              "complex_fock": "complex Fock matrices", "unsupported_refine": "max_refine"}
    if scalar.startswith("truncated"):
        with pytest.raises(ValueError, match="required orbital space"):
            run(system, basis, options, **kwargs)
        assert setup == []
    elif scalar in errors:
        with pytest.raises(NotImplementedError, match=errors[scalar]):
            run(system, basis, options, **kwargs)
        if scalar != "complex_fock":
            assert setup == []
    else:
        with pytest.raises(SeedAdmitted):
            run(system, basis, options, **kwargs)
        assert setup == ["J factory"]
