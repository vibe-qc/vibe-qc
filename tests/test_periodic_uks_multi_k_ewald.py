"""Phase 15c-3b: multi-k periodic UKS SCF driver using EWALD_3D."""

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
        3,
        np.eye(3) * box,
        atoms,
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


def _default_options(functional: str = "PBE"):
    opts = vq.PeriodicKSOptions()
    opts.functional = functional
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.3
    opts.max_iter = 60
    opts.use_diis = True
    return opts


def _slab_options(functional: str = "LDA"):
    opts = _default_options(functional)
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D
    opts.lattice_opts.slab_ewald_alpha = 0.4
    opts.conv_tol_energy = 1e-7
    opts.conv_tol_grad = 1e-5
    opts.max_iter = 40
    return opts


def test_slab_ewald_2d_multi_k_closed_shell_uks_matches_rks():
    sysp, basis = _h2_slab()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])
    opts_uks = _slab_options("LDA")
    opts_rks = _slab_options("LDA")

    r_uks = vq.run_uks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, opts_uks, progress=False,
    )
    r_rks = vq.run_rks_periodic_scf(
        sysp, basis, kmesh, opts_rks, progress=False,
    )

    assert r_uks.converged and r_rks.converged
    assert r_uks.energy == pytest.approx(r_rks.energy, abs=1e-10)
    assert r_uks.e_coulomb == pytest.approx(r_rks.e_coulomb, abs=1e-10)
    assert r_uks.e_hf_exchange == pytest.approx(0.0, abs=1e-12)
    assert r_uks.omega == pytest.approx(opts_uks.lattice_opts.slab_ewald_alpha)
    assert r_uks.grid_shape == (0, 0, 0)


def test_uks_smearing_small_t_matches_unsmeared_gapped_system():
    """Per-spin Fermi-Dirac smearing (parity with the RKS multi-k
    driver — the old NotImplementedError gate is gone): on a gapped
    closed-shell system a small temperature must reproduce the
    unsmeared energy (occupations stay essentially integer) and the
    free energy must equal E − T·S."""
    sysp, basis = _h2_in_box()
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])

    r0 = vq.run_uks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _default_options("PBE"), omega=0.5,
    )
    opts = _default_options("PBE")
    opts.smearing_temperature = 1e-3
    r1 = vq.run_uks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, opts, omega=0.5,
    )
    assert r0.converged and r1.converged
    assert r1.smearing_temperature == pytest.approx(1e-3)
    assert r1.entropy >= 0.0
    assert r1.free_energy == pytest.approx(
        r1.energy - 1e-3 * r1.entropy, abs=1e-12
    )
    # Gapped system, T well below the gap: energies agree tightly.
    assert r1.energy == pytest.approx(r0.energy, abs=5e-6)


def test_uks_smearing_terminal_metadata_matches_reported_spectrum_and_density():
    """The terminal diagonalisation must solve and return one FD state."""
    from vibeqc.smearing.fermi_dirac import fermi_dirac_occupations_per_k

    sysp, basis = _h2_in_box()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])
    opts = _default_options("PBE")
    opts.smearing_temperature = 0.05

    result = vq.run_uks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, opts, omega=0.5
    )

    assert result.converged
    expected_alpha, mu_alpha, entropy_alpha = fermi_dirac_occupations_per_k(
        result.mo_energies_alpha,
        kmesh.weights,
        1.0,
        opts.smearing_temperature,
        spin_degeneracy=1.0,
    )
    expected_beta, mu_beta, entropy_beta = fermi_dirac_occupations_per_k(
        result.mo_energies_beta,
        kmesh.weights,
        1.0,
        opts.smearing_temperature,
        spin_degeneracy=1.0,
    )
    for got, expected in zip(result.occupations_alpha, expected_alpha):
        np.testing.assert_allclose(got, expected, atol=1e-12)
    for got, expected in zip(result.occupations_beta, expected_beta):
        np.testing.assert_allclose(got, expected, atol=1e-12)
    assert result.fermi_level_alpha == pytest.approx(mu_alpha, abs=1e-12)
    assert result.fermi_level_beta == pytest.approx(mu_beta, abs=1e-12)
    assert result.entropy == pytest.approx(
        entropy_alpha + entropy_beta, abs=1e-12
    )
    assert result.free_energy == pytest.approx(
        result.energy - opts.smearing_temperature * result.entropy,
        abs=1e-12,
    )

    expected_density_alpha = (
        vq._vibeqc_core.real_space_density_from_kpoints_fractional(
            result.mo_coeffs_alpha,
            result.occupations_alpha,
            kmesh,
            result.density_alpha.cells,
        )
    )
    expected_density_beta = (
        vq._vibeqc_core.real_space_density_from_kpoints_fractional(
            result.mo_coeffs_beta,
            result.occupations_beta,
            kmesh,
            result.density_beta.cells,
        )
    )
    for got, expected in zip(
        result.density_alpha.blocks, expected_density_alpha.blocks
    ):
        np.testing.assert_allclose(got, expected, atol=1e-12)
    for got, expected in zip(
        result.density_beta.blocks, expected_density_beta.blocks
    ):
        np.testing.assert_allclose(got, expected, atol=1e-12)


def test_damped_li_terminal_recheck_continues_and_fails_closed_at_cap():
    """A damped KS plateau cannot certify the look-ahead density."""
    box = 30.0
    centre = box / 2.0
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(3, [centre, centre, centre])],
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _default_options("PBE")
    opts.max_iter = 3
    opts.use_diis = False
    opts.damping = 0.95
    opts.conv_tol_energy = 0.003
    opts.conv_tol_grad = 0.03

    result = vq.run_uks_periodic_multi_k_ewald3d(
        sysp,
        basis,
        kmesh,
        opts,
        spacing_bohr=0.5,
        progress=False,
    )

    provisional = result.scf_trace[1]
    assert abs(provisional.delta_e) < opts.conv_tol_energy
    assert provisional.grad_norm < opts.conv_tol_grad
    assert result.n_iter == opts.max_iter
    assert not result.converged
    assert (
        abs(result.energy - result.scf_trace[-1].energy)
        > opts.conv_tol_energy
    )


def test_uks_smearing_open_shell_doublet_runs():
    """H-atom doublet with per-spin smearing: separate chemical
    potentials at fixed n_alpha=1 / n_beta=0, finite diagnostics."""
    sysp, basis = _h_atom_in_box()
    opts = _default_options("PBE")
    opts.smearing_temperature = 0.005
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = vq.run_uks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, opts, omega=0.5,
    )
    assert r.converged
    assert np.isfinite(r.fermi_level_alpha)
    assert r.entropy >= 0.0
    # n_beta = 0: the beta channel carries no electrons and no entropy
    # at any mu, so the free energy stays close to the bare energy for
    # this tiny gapped system.
    assert abs(r.free_energy - r.energy) < 1e-3


def test_multi_k_at_gamma_mesh_matches_gamma_driver_pbe():
    """[1,1,1] multi-k UKS must match Γ-only UKS to ~µHa for PBE."""
    sysp, basis = _h2_in_box()
    opts = _default_options("PBE")
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    r_multi = vq.run_uks_periodic_multi_k_ewald3d(
        sysp,
        basis,
        kmesh,
        opts,
        omega=0.5,
    )
    r_gamma = vq.run_uks_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
    )
    assert r_multi.converged and r_gamma.converged
    assert abs(r_multi.energy - r_gamma.energy) < 1e-10


def test_multi_k_b3lyp_uses_the_ewald_exxdiv_gauge_not_the_gamma_one():
    """A hybrid's full-range arm rides the corrected Ewald split, so multi-k
    no longer reproduces the Γ driver bit-for-bit.

    The Γ driver sums the *bare* ``1/r`` exchange over its image ball. That
    is legitimate at Γ, where the molecular-limit density decays with image
    distance, and it lands on a truncated gauge carrying no q -> 0
    correction. On a finite k mesh the density is Born-von-Kármán-torus
    periodic, that same sum diverges, and the multi-k driver must use the
    Ewald split with the probe-charge ``q + G = 0`` term instead.

    So the two are different gauges by construction. This replaces an older
    ``< 1e-10`` equality that encoded the pre-fix bare-sum behaviour. The
    threshold is deliberately well below the measured ~74 µHa shift on this
    fixture and well above float noise: the shift is *small* here because a
    30-bohr box at a 12-bohr cutoff holds a single lattice cell, which is
    exactly why this fixture could never have caught the divergence itself.
    """
    sysp, basis = _h2_in_box()
    opts = _default_options("B3LYP")
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    r_multi = vq.run_uks_periodic_multi_k_ewald3d(
        sysp,
        basis,
        kmesh,
        opts,
        omega=0.5,
    )
    r_gamma = vq.run_uks_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
    )
    assert r_multi.converged and r_gamma.converged
    assert abs(r_multi.energy - r_gamma.energy) > 1e-5
    # The whole shift lives in the exchange; nothing else may move.
    assert abs(r_multi.e_xc - r_gamma.e_xc) < 1e-9
    assert abs(
        (r_multi.energy - r_gamma.energy)
        - (r_multi.e_hf_exchange - r_gamma.e_hf_exchange)
    ) < 1e-8


def test_multi_k_uks_hybrid_matches_the_rks_driver_on_a_closed_shell():
    """Closed-shell UKS must equal RKS, energy *and* the reported J/K split.

    This is the gate that catches both plausible port errors at once: a
    wrong prefactor on the corrected k-space term (UKS folds per-spin with
    no 1/2, RKS folds a total density with one), and the corrected arms
    leaking into the J-only reporting build. The second is invisible to any
    energy assertion -- it moves ``e_hf_exchange`` and ``e_coulomb`` by
    ±30.8 mHa while leaving the total bit-identical -- so the split is
    asserted explicitly.
    """
    box = 12.0
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    def _opts():
        o = vq.PeriodicKSOptions()
        o.functional = "PBE0"
        o.lattice_opts.cutoff_bohr = 12.0
        o.lattice_opts.nuclear_cutoff_bohr = 12.0
        o.damping = 0.3
        o.max_iter = 200
        o.conv_tol_energy = 1e-10
        o.conv_tol_grad = 1e-7
        return o

    for mesh in ([1, 1, 1], [2, 1, 1]):
        kmesh = vq.monkhorst_pack(sysp, mesh)
        u = vq.run_uks_periodic_multi_k_ewald3d(
            sysp, basis, kmesh, _opts(), auto_optimize_truncation=False
        )
        r = vq.run_rks_periodic_multi_k_ewald3d(
            sysp, basis, kmesh, _opts(), auto_optimize_truncation=False
        )
        assert u.converged and r.converged
        assert u.energy == pytest.approx(r.energy, abs=1e-9)
        assert u.e_hf_exchange == pytest.approx(r.e_hf_exchange, abs=1e-9)
        assert u.e_coulomb == pytest.approx(r.e_coulomb, abs=1e-9)


def test_multi_k_uks_hybrid_exchange_is_cutoff_independent():
    """The full-range exchange sum must converge in the lattice cutoff.

    Uses a cell whose image ball holds more than one lattice cell -- the
    30-bohr fixture above holds exactly one, which makes any image-sum
    assertion structurally blind.
    """
    box = 12.0
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])

    energies = []
    for cutoff in (12.0, 20.0):
        o = vq.PeriodicKSOptions()
        o.functional = "PBE0"
        o.lattice_opts.cutoff_bohr = cutoff
        o.lattice_opts.nuclear_cutoff_bohr = cutoff
        o.damping = 0.3
        o.max_iter = 200
        o.conv_tol_energy = 1e-10
        o.conv_tol_grad = 1e-7
        r = vq.run_uks_periodic_multi_k_ewald3d(
            sysp, basis, kmesh, o, auto_optimize_truncation=False
        )
        assert r.converged
        assert len(r.density_alpha.cells) > 1
        energies.append(r.energy)
    assert abs(energies[1] - energies[0]) < 1e-5, energies

    # Published target: PySCF 2.14.0 KUKS.density_fit(), xc='pbe0',
    # exxdiv='ewald', cell.spin=0, on the matched cell/basis/mesh:
    # -1.1552192296377888 Ha (exxdiv=None gives -1.117597525253365, so this
    # also pins that the q -> 0 correction is applied at all).
    assert energies[0] == pytest.approx(-1.1552192296377888, abs=2e-4)


def test_h_atom_multi_k_pbe():
    """H atom open-shell UKS at [1,1,1] mesh — same physics as
    Γ-only, ⟨S²⟩ = 3/4."""
    sysp, basis = _h_atom_in_box()
    opts = _default_options("PBE")
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = vq.run_uks_periodic_multi_k_ewald3d(
        sysp,
        basis,
        kmesh,
        opts,
        omega=0.5,
    )
    assert r.converged
    assert abs(r.s_squared - 0.75) < 1e-6


def test_omega_invariance_at_111_mesh():
    sysp, basis = _h2_in_box()
    opts = _default_options("PBE")
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    energies = []
    for omega in (0.3, 0.5, 1.0, 1.5):
        r = vq.run_uks_periodic_multi_k_ewald3d(
            sysp,
            basis,
            kmesh,
            opts,
            omega=omega,
            spacing_bohr=0.3,
        )
        assert r.converged
        energies.append(r.energy)
    spread = max(energies) - min(energies)
    assert spread < 0.005 * abs(min(energies))


def test_result_has_expected_per_k_shapes():
    sysp, basis = _h2_in_box()
    opts = _default_options()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])
    r = vq.run_uks_periodic_multi_k_ewald3d(
        sysp,
        basis,
        kmesh,
        opts,
        omega=0.5,
    )
    n_k = len(kmesh.kpoints)
    n_bf = basis.nbasis
    assert len(r.fock_alpha) == n_k
    assert len(r.fock_beta) == n_k
    assert len(r.occupations_alpha) == n_k
    assert len(r.occupations_beta) == n_k
    assert len(r.overlap) == n_k
    assert len(r.hcore) == n_k
    for k_idx in range(n_k):
        assert r.fock_alpha[k_idx].shape == (n_bf, n_bf)
        assert r.fock_beta[k_idx].shape == (n_bf, n_bf)
        assert r.occupations_alpha[k_idx].shape == r.mo_energies_alpha[k_idx].shape
        assert r.occupations_beta[k_idx].shape == r.mo_energies_beta[k_idx].shape
    assert r.functional.upper().startswith("PBE")
    assert "VBM" in pr._band_summary(r)
    orbital_text = pr._mo_summary(r)
    assert "Crystal orbital energies (alpha)" in orbital_text
    assert "Crystal orbital energies (beta)" in orbital_text
    expected_alpha = (
        vq._vibeqc_core.real_space_density_from_kpoints_fractional(
            r.mo_coeffs_alpha,
            r.occupations_alpha,
            kmesh,
            r.density_alpha.cells,
        )
    )
    expected_beta = (
        vq._vibeqc_core.real_space_density_from_kpoints_fractional(
            r.mo_coeffs_beta,
            r.occupations_beta,
            kmesh,
            r.density_beta.cells,
        )
    )
    for got, expected in zip(r.density_alpha.blocks, expected_alpha.blocks):
        np.testing.assert_allclose(got, expected, atol=1e-12)
    for got, expected in zip(r.density_beta.blocks, expected_beta.blocks):
        np.testing.assert_allclose(got, expected, atol=1e-12)
    # ω is auto-derived from nuclear_cutoff_bohr (mirror of the
    # sibling Ewald drivers, commit 49f8ae91 / 433d3543); the driver
    # ``omega`` kwarg is overridden.
    assert r.omega == pytest.approx(0.5)


def test_accepts_non_orthorhombic_lattice_smoke():
    lat = np.array(
        [
            [30.0, 1.5, 0.0],
            [0.0, 30.0, 0.0],
            [0.0, 0.0, 30.0],
        ]
    )
    c = 15.0
    atoms = [vq.Atom(1, [c, c, c])]
    sysp = vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=2)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _default_options()
    opts.max_iter = 1
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = vq.run_uks_periodic_multi_k_ewald3d(
        sysp,
        basis,
        kmesh,
        opts,
        omega=0.5,
        grid_shape=(8, 8, 8),
    )
    assert r.n_iter == 1
    assert np.isfinite(r.energy)


def test_fused_multi_k_bloch_folds_match_python_fallback(monkeypatch):
    """The fused OpenMP ``bloch_sum_multi_k`` per-iteration F_HF folds
    and the per-k Python ``_bloch_sum_blocks`` fallback converge the
    same SCF. The kernel's cos/sin phase differs from ``np.exp`` by at
    most one ulp, so the converged energies agree far inside the pinned
    tolerances."""
    import vibeqc._vibeqc_core as core

    sysp, basis = _h_atom_in_box()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])
    r_fused = vq.run_uks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _default_options("LDA"), omega=0.5,
        progress=False,
    )

    monkeypatch.delattr(core, "bloch_sum_multi_k")
    r_py = vq.run_uks_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _default_options("LDA"), omega=0.5,
        progress=False,
    )

    assert r_fused.converged and r_py.converged
    assert r_fused.n_iter == r_py.n_iter
    assert abs(r_fused.energy - r_py.energy) < 1e-10
    for F_f, F_p in zip(r_fused.fock_alpha, r_py.fock_alpha):
        assert np.allclose(F_f, F_p, atol=1e-10)
    for F_f, F_p in zip(r_fused.fock_beta, r_py.fock_beta):
        assert np.allclose(F_f, F_p, atol=1e-10)
