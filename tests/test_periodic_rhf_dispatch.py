"""Phase 12e-c-4 tests: end-to-end SCF dispatch on ``CoulombMethod``.

Contracts exercised:

1. **DIRECT_TRUNCATED routing** — dispatcher routes to the C++ driver
   and produces the same energy as calling ``run_rhf_periodic`` /
   ``run_rhf_periodic_gamma`` directly.

2. **EWALD_3D routing** — dispatcher routes to the Python Ewald driver
   and produces the same energy as calling ``run_rhf_periodic_multi_k_
   ewald3d`` / ``run_rhf_periodic_gamma_ewald3d`` directly. When
   ``lattice_opts.coulomb_method = EWALD_3D`` is set on the options,
   downstream nuclear repulsion also uses Ewald — so the self-
   consistent-Ewald energy differs from a mixed "Ewald-electronic +
   direct-nuclear" configuration by the image nuclear-lattice
   contribution.

3. **None options** — no options argument uses defaults and produces
   a sensible SCF result.

4. **Low-dimensional dispatch** — ``SLAB_EWALD_2D`` routes Gamma-only
   meshes to the dedicated slab Ewald RHF driver and dense meshes to the
   multi-k slab Ewald path; ``NEUTRALIZED_1D`` still raises
   ``NotImplementedError``.

5. **Options translation** — passing either ``PeriodicSCFOptions`` or
   ``PeriodicRHFOptions`` works; fields are translated correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _h2(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]),
         vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h2_slab(box: float = 18.0, vacuum: float = 45.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        2, np.diag([box, box, vacuum]),
        [vq.Atom(1, [c, c, vacuum / 2 - 0.7]),
         vq.Atom(1, [c, c, vacuum / 2 + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _hf_polar_slab(vacuum: float):
    sysp = vq.PeriodicSystem(
        2, np.diag([8.0, 8.0, vacuum]),
        [vq.Atom(9, [4.0, 4.0, 19.13]),
         vq.Atom(1, [4.0, 4.0, 20.87])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _std_options(method):
    opts = vq.PeriodicSCFOptions()
    opts.lattice_opts.cutoff_bohr = 12
    opts.lattice_opts.nuclear_cutoff_bohr = 15
    opts.lattice_opts.coulomb_method = method
    opts.damping = 0.3
    opts.max_iter = 40
    return opts


# ---------------------------------------------------------------------------
# DIRECT_TRUNCATED routing
# ---------------------------------------------------------------------------

def test_dispatcher_routes_direct_truncated_multi_k():
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _std_options(vq.CoulombMethod.DIRECT_TRUNCATED)

    r_dispatch = vq.run_rhf_periodic_scf(sysp, basis, km, opts)
    r_direct = vq.run_rhf_periodic(sysp, basis, km, opts)

    assert r_dispatch.converged and r_direct.converged
    assert r_dispatch.energy == pytest.approx(r_direct.energy, abs=1e-10)


def test_dispatcher_routes_direct_truncated_gamma():
    sysp, basis = _h2()
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12
    opts.lattice_opts.nuclear_cutoff_bohr = 15
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.DIRECT_TRUNCATED
    opts.damping = 0.3
    opts.max_iter = 40

    r_dispatch = vq.run_rhf_periodic_gamma_scf(sysp, basis, opts)
    r_direct = vq.run_rhf_periodic_gamma(sysp, basis, opts)

    assert r_dispatch.converged and r_direct.converged
    assert r_dispatch.energy == pytest.approx(r_direct.energy, abs=1e-10)


# ---------------------------------------------------------------------------
# EWALD_3D routing
# ---------------------------------------------------------------------------

def test_dispatcher_routes_ewald_multi_k():
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _std_options(vq.CoulombMethod.EWALD_3D)

    r_dispatch = vq.run_rhf_periodic_scf(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )

    rhf_opts = vq.PeriodicRHFOptions()
    rhf_opts.lattice_opts.cutoff_bohr = 12
    rhf_opts.lattice_opts.nuclear_cutoff_bohr = 15
    rhf_opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    rhf_opts.damping = 0.3
    rhf_opts.max_iter = 40
    r_direct = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, rhf_opts, omega=0.5, spacing_bohr=0.3,
    )

    assert r_dispatch.converged and r_direct.converged
    assert r_dispatch.energy == pytest.approx(r_direct.energy, abs=1e-10)


def test_dispatcher_routes_ewald_gamma():
    sysp, basis = _h2()
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12
    opts.lattice_opts.nuclear_cutoff_bohr = 15
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.damping = 0.3
    opts.max_iter = 40

    r_dispatch = vq.run_rhf_periodic_gamma_scf(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )
    r_direct = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )

    assert r_dispatch.converged and r_direct.converged
    assert r_dispatch.energy == pytest.approx(r_direct.energy, abs=1e-10)


# ---------------------------------------------------------------------------
# None options → defaults
# ---------------------------------------------------------------------------

def test_dispatcher_accepts_none_options():
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    # No options — picks default PeriodicSCFOptions with DIRECT_TRUNCATED.
    r = vq.run_rhf_periodic_scf(sysp, basis, km)
    assert r.converged


# ---------------------------------------------------------------------------
# Low-dimensional dispatch
# ---------------------------------------------------------------------------

def test_dispatcher_routes_slab_ewald_2d_gamma():
    sysp, basis = _h2_slab()
    opts = _std_options(vq.CoulombMethod.SLAB_EWALD_2D)
    opts.lattice_opts.slab_ewald_alpha = 0.4
    opts.conv_tol_energy = 1e-7
    opts.conv_tol_grad = 1e-5
    opts.max_iter = 35

    result = vq.run_rhf_periodic_gamma_scf(sysp, basis, opts)

    assert result.converged
    assert result.n_iter > 1
    assert result.e_nuclear == pytest.approx(
        vq.nuclear_repulsion_per_cell(sysp, opts.lattice_opts),
        abs=1e-12,
    )
    assert result.omega == pytest.approx(opts.lattice_opts.slab_ewald_alpha)


def test_slab_ewald_2d_gamma_polar_slab_vacuum_invariant():
    opts_a = _std_options(vq.CoulombMethod.SLAB_EWALD_2D)
    opts_b = _std_options(vq.CoulombMethod.SLAB_EWALD_2D)
    for opts in (opts_a, opts_b):
        opts.lattice_opts.cutoff_bohr = 10.0
        opts.lattice_opts.nuclear_cutoff_bohr = 14.0
        opts.lattice_opts.slab_ewald_alpha = 0.4
        opts.conv_tol_energy = 1e-7
        opts.conv_tol_grad = 1e-5
        opts.max_iter = 30

    sys_a, bas_a = _hf_polar_slab(35.0)
    sys_b, bas_b = _hf_polar_slab(55.0)
    res_a = vq.run_rhf_periodic_gamma_scf(sys_a, bas_a, opts_a, progress=False)
    res_b = vq.run_rhf_periodic_gamma_scf(sys_b, bas_b, opts_b, progress=False)

    assert res_a.converged and res_b.converged
    assert res_a.energy == pytest.approx(res_b.energy, abs=1e-9)


def test_slab_ewald_2d_multi_k_routes_through_dispatcher():
    sysp, basis = _h2_slab()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    opts = _std_options(vq.CoulombMethod.SLAB_EWALD_2D)
    opts.lattice_opts.slab_ewald_alpha = 0.4
    opts.conv_tol_energy = 1e-7
    opts.conv_tol_grad = 1e-5
    opts.max_iter = 35

    result = vq.run_rhf_periodic_scf(sysp, basis, km, opts, progress=False)

    assert result.converged
    assert isinstance(result, vq.PeriodicRHFMultiKEwaldResult)
    assert result.n_iter > 1
    assert result.e_nuclear == pytest.approx(
        vq.nuclear_repulsion_per_cell(sysp, opts.lattice_opts),
        abs=1e-12,
    )
    assert result.energy == pytest.approx(
        result.e_electronic + result.e_nuclear,
        abs=1e-10,
    )
    assert result.omega == pytest.approx(opts.lattice_opts.slab_ewald_alpha)
    assert result.grid_shape == (0, 0, 0)


def test_neutralized_1d_raises_not_implemented_gamma():
    sysp, basis = _h2()
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.NEUTRALIZED_1D
    with pytest.raises(NotImplementedError, match="NEUTRALIZED_1D"):
        vq.run_rhf_periodic_gamma_scf(sysp, basis, opts)


# ---------------------------------------------------------------------------
# Options-type translation
# ---------------------------------------------------------------------------

def test_dispatcher_accepts_both_options_types():
    """Both PeriodicSCFOptions and PeriodicRHFOptions should work as
    input — the dispatcher translates fields to the right type for the
    underlying backend."""
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])

    # Path A: PeriodicSCFOptions with EWALD_3D.
    scf_opts = vq.PeriodicSCFOptions()
    scf_opts.lattice_opts.cutoff_bohr = 12
    scf_opts.lattice_opts.nuclear_cutoff_bohr = 15
    scf_opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    scf_opts.damping = 0.3
    scf_opts.max_iter = 40
    r_a = vq.run_rhf_periodic_scf(
        sysp, basis, km, scf_opts, omega=0.5, spacing_bohr=0.3,
    )

    # Path B: PeriodicRHFOptions with EWALD_3D.
    rhf_opts = vq.PeriodicRHFOptions()
    rhf_opts.lattice_opts.cutoff_bohr = 12
    rhf_opts.lattice_opts.nuclear_cutoff_bohr = 15
    rhf_opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    rhf_opts.damping = 0.3
    rhf_opts.max_iter = 40
    r_b = vq.run_rhf_periodic_scf(
        sysp, basis, km, rhf_opts, omega=0.5, spacing_bohr=0.3,
    )

    assert r_a.converged and r_b.converged
    assert r_a.energy == pytest.approx(r_b.energy, abs=1e-12)


# ---------------------------------------------------------------------------
# Ewald vs direct differ predictably on realistic periodic cells
# ---------------------------------------------------------------------------

def test_ewald_and_direct_differ_on_periodic_cell():
    """On a genuinely periodic cell (not the molecular limit), EWALD_3D
    and DIRECT_TRUNCATED produce different energies — the Ewald side
    captures the long-range nuclear + electronic tails the truncated
    direct-sum drops. Both converge; the difference is finite and
    non-trivial.

    H2-chain analogue at a = 15 bohr gives both drivers a clean
    convergence in a few iterations while the Ewald-nuclear image
    sum still contributes non-trivially vs the direct truncation.
    """
    a = 15.0
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * a,
        [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [1, 1, 1])

    opts_direct = _std_options(vq.CoulombMethod.DIRECT_TRUNCATED)
    opts_ewald = _std_options(vq.CoulombMethod.EWALD_3D)

    r_d = vq.run_rhf_periodic_scf(sysp, basis, km, opts_direct)
    r_e = vq.run_rhf_periodic_scf(
        sysp, basis, km, opts_ewald, omega=0.5, spacing_bohr=0.3,
    )
    assert r_d.converged and r_e.converged
    assert np.isfinite(r_d.energy) and np.isfinite(r_e.energy)
    assert r_d.energy < 0.0 and r_e.energy < 0.0
    # Non-trivial difference from the Ewald nuclear-image sum.
    diff = abs(r_d.energy - r_e.energy)
    assert diff > 1e-3
    assert diff < 10.0


@pytest.mark.parametrize("kind", ["RHF", "KS", "SCF"])
def test_guess_option_conversion_preserves_payload_and_ecp(kind):
    from vibeqc.periodic_rhf_dispatch import _copy_options_to_rhf, _copy_options_to_scf
    from vibeqc.periodic_ks_dispatch import _copy_ks_options

    cls, copier = {
        "RHF": (vq.PeriodicRHFOptions, _copy_options_to_rhf),
        "KS": (vq.PeriodicKSOptions, _copy_ks_options),
        "SCF": (vq.PeriodicSCFOptions, _copy_options_to_scf),
    }[kind]
    source = cls()
    source.initial_guess = vq.InitialGuess.SAD
    source.ecp_home_centers = [[0., 0., 0.]]
    source.ecp_effective_charges = [19.]
    source.ecp_total_ncore = 28
    if kind != "SCF":
        source.read_density = np.eye(2)
        source.read_density_alpha = .7 * np.eye(2)
        source.read_density_beta = .3 * np.eye(2)
        source.read_path = "previous.qvf"
        source.atomic_spins = [1]
        source.spinlock_mode = vq.SpinlockMode.PATTERN_HOLD
        source.spinlock_iterations = 3
    copied = copier(source)
    assert copied.initial_guess == source.initial_guess
    for field in ("ecp_home_centers", "ecp_effective_charges", "ecp_total_ncore"):
        np.testing.assert_array_equal(getattr(copied, field), getattr(source, field))
    if kind != "SCF":
        for field in ("read_density", "read_density_alpha", "read_density_beta", "atomic_spins"):
            np.testing.assert_array_equal(getattr(copied, field), getattr(source, field))
        assert copied.read_path == "previous.qvf"
        assert copied.spinlock_mode == vq.SpinlockMode.PATTERN_HOLD
        assert copied.spinlock_iterations == 3


def test_native_option_conversion_refuses_unrepresentable_restart():
    from vibeqc.periodic_rhf_dispatch import _copy_options_to_scf
    source = vq.PeriodicRHFOptions()
    source.read_density = np.eye(2)
    with pytest.raises(NotImplementedError, match="cannot carry.*read_density"):
        _copy_options_to_scf(source)


@pytest.mark.parametrize("method", ["RHF", "RKS"])
@pytest.mark.parametrize("multi", [False, True])
def test_dispatch_rejects_closed_shell_spin_seed_before_output(monkeypatch, method, multi):
    system, basis = _h2()
    options = vq.PeriodicRHFOptions() if method == "RHF" else vq.PeriodicKSOptions()
    options.initial_guess = vq.InitialGuess.SAD
    options.atomic_spins = [1, -1]
    options.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    suffix = "scf" if multi else "gamma_scf"
    driver = getattr(vq, f"run_{method.lower()}_periodic_{suffix}")
    args = (system, basis, vq.monkhorst_pack(system, [3, 1, 1]), options) if multi else (system, basis, options)
    with pytest.raises(NotImplementedError, match="atomic_spins"):
        driver(*args)


@pytest.mark.parametrize("method", ["RHF", "RKS"])
@pytest.mark.parametrize("multi", [False, True])
def test_dispatch_executes_read_payload(method, multi):
    from vibeqc.guess_read import resolve_periodic_read_density_k_closed

    system, basis = _h2()
    options = vq.PeriodicRHFOptions() if method == "RHF" else vq.PeriodicKSOptions()
    options.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    options.lattice_opts.cutoff_bohr = 12.
    options.lattice_opts.nuclear_cutoff_bohr = 15.
    options.max_iter = 40
    options.initial_guess = vq.InitialGuess.SAD
    suffix = "scf" if multi else "gamma_scf"
    driver = getattr(vq, f"run_{method.lower()}_periodic_{suffix}")
    mesh = vq.monkhorst_pack(system, [3, 1, 1])
    args = (system, basis, mesh, options) if multi else (system, basis, options)
    source = driver(*args, grid_shape=24)
    assert source.converged
    options.initial_guess = vq.InitialGuess.READ
    if multi:
        payload = resolve_periodic_read_density_k_closed(
            read_from=source, expected_n_k=3, n_basis=basis.nbasis)
        restarted = driver(*args, grid_shape=24, initial_density_k=payload)
    else:
        options.read_density = np.asarray(source.density)
        options.read_path = "previous.qvf"
        restarted = driver(*args, grid_shape=24)
    assert restarted.converged
    assert restarted.guess_selection.effective == vq.InitialGuess.READ
    assert restarted.energy == pytest.approx(source.energy, abs=1e-8)
