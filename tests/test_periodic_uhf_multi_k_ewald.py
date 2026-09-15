"""Phase 15b tests: multi-k periodic UHF SCF with Ewald-3D Coulomb.

Contracts:

1. **Closed-shell limit equals multi-k RHF** — at any k-mesh, when
   ``multiplicity = 1`` and ``n_e`` is even the UHF total energy
   matches the multi-k Ewald RHF energy to ~µHa.

2. **One-electron exact ⟨S²⟩** — H atom doublet at [1,1,1] mesh
   reports ⟨S²⟩ = 0.75 = S(S+1) exactly.

3. **Open-shell convergence** — H atom doublet on a [1,1,1] mesh
   converges to a finite, sensible energy.

4. **α/β density symmetry on closed shell** — converged D_α and
   D_β LatticeMatrixSets match block-by-block.

5. **Result-shape contract** — every field on
   PeriodicUHFMultiKEwaldResult is populated and shaped correctly
   (per-k lists with right length, one density LMS per spin).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
import vibeqc.periodic_runner as pr


def _h2_closed_shell(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]),
         vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h_atom_doublet(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, [vq.Atom(1, [c, c, c])])
    sysp.multiplicity = 2
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _options():
    o = vq.PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    o.damping = 0.3
    o.max_iter = 40
    return o


def test_uhf_smearing_fractional_occupations_converge():
    """Per-spin Fermi-Dirac smearing is wired on the UHF multi-k Ewald
    route. The old guard raised NotImplementedError here; a modest
    temperature on H2/[2,1,1] gives a nonzero entropy and converges the
    Mermin free energy."""
    sysp, basis = _h2_closed_shell()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    opts = _options()
    opts.smearing_temperature = 0.05
    opts.max_iter = 30
    opts.conv_tol_energy = 1e-7
    opts.conv_tol_grad = 1e-5

    r = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    assert r.smearing_temperature == pytest.approx(0.05)
    assert r.entropy > 1e-5
    assert np.isfinite(r.fermi_level_alpha)
    assert np.isfinite(r.fermi_level_beta)
    assert r.free_energy == pytest.approx(r.energy - 0.05 * r.entropy, abs=1e-12)
    assert abs(r.s_squared) < 1e-10
    assert len(r.occupations_alpha) == len(km.kpoints)
    assert len(r.occupations_beta) == len(km.kpoints)
    assert "fractionally occupied / smeared" in pr._band_summary(r)


def test_uhf_smearing_terminal_density_matches_reported_orbitals():
    """The terminal density must be reconstructed from returned C and f."""
    from vibeqc.smearing.fermi_dirac import fermi_dirac_occupations_per_k

    sysp, basis = _h2_closed_shell()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    opts = _options()
    opts.smearing_temperature = 0.05
    opts.max_iter = 30
    opts.conv_tol_energy = 1e-7
    opts.conv_tol_grad = 1e-5

    result = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp,
        basis,
        km,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )

    assert result.converged
    expected_occ_alpha, mu_alpha, entropy_alpha = (
        fermi_dirac_occupations_per_k(
            result.mo_energies_alpha,
            km.weights,
            1.0,
            opts.smearing_temperature,
            spin_degeneracy=1.0,
        )
    )
    expected_occ_beta, mu_beta, entropy_beta = (
        fermi_dirac_occupations_per_k(
            result.mo_energies_beta,
            km.weights,
            1.0,
            opts.smearing_temperature,
            spin_degeneracy=1.0,
        )
    )
    for got, expected in zip(result.occupations_alpha, expected_occ_alpha):
        np.testing.assert_allclose(got, expected, atol=1e-12)
    for got, expected in zip(result.occupations_beta, expected_occ_beta):
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
    expected_alpha = (
        vq._vibeqc_core.real_space_density_from_kpoints_fractional(
            result.mo_coeffs_alpha,
            result.occupations_alpha,
            km,
            result.density_alpha.cells,
        )
    )
    expected_beta = (
        vq._vibeqc_core.real_space_density_from_kpoints_fractional(
            result.mo_coeffs_beta,
            result.occupations_beta,
            km,
            result.density_beta.cells,
        )
    )
    for got, expected in zip(result.density_alpha.blocks, expected_alpha.blocks):
        np.testing.assert_allclose(got, expected, atol=1e-12)
    for got, expected in zip(result.density_beta.blocks, expected_beta.blocks):
        np.testing.assert_allclose(got, expected, atol=1e-12)


def test_damped_li_terminal_recheck_continues_and_fails_closed_at_cap():
    """A damped energy plateau cannot certify the look-ahead density."""
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
    opts = _options()
    opts.max_iter = 3
    opts.use_diis = False
    opts.damping = 0.95
    opts.conv_tol_energy = 0.02
    opts.conv_tol_grad = 0.07

    result = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp,
        basis,
        kmesh,
        opts,
        spacing_bohr=0.5,
        progress=False,
    )

    # Iteration 2 satisfies the former convergence check, but the terminal
    # look-ahead rejects it and consumes the remaining cycle.
    provisional = result.scf_trace[1]
    assert abs(provisional.delta_e) < opts.conv_tol_energy
    assert provisional.grad_norm < opts.conv_tol_grad
    assert result.n_iter == opts.max_iter
    assert not result.converged
    assert (
        abs(result.energy - result.scf_trace[-1].energy)
        > opts.conv_tol_energy
    )


# ---------------------------------------------------------------------------
# 1. Closed-shell H₂: UHF ≡ RHF
# ---------------------------------------------------------------------------

# The 9.3e-4 Ha RHF-vs-UHF disagreement diagnosed 2026-06-10 was the
# b4a6faba merge-drop: the revert had put UHF back on the pre-f7ee5832
# split J (build_fock_2e_real_space J_SR + grid-Poisson J_LR) while
# RHF kept the analytic-FT J. The 2026-06-10/11 merge-drop restoration
# routes both drivers through periodic_fock_multi_k.ewald_3d_j_blocks
# again (one J path, f7ee5832's invariant), and the xfail flipped to
# strict-XPASS — closed-shell UHF ≡ RHF holds. pbc_audit_2026-06.md
# EWALD-driver owner item closed.
def test_closed_shell_uhf_matches_rhf_at_111_mesh():
    sysp, basis = _h2_closed_shell()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _options()
    r_rhf = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    r_uhf = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r_rhf.converged and r_uhf.converged
    assert r_uhf.energy == pytest.approx(r_rhf.energy, abs=1e-9)
    assert abs(r_uhf.s_squared) < 1e-10


# ---------------------------------------------------------------------------
# 2. H atom doublet ⟨S²⟩ exact at [1,1,1]
# ---------------------------------------------------------------------------

def test_h_atom_doublet_spin_squared_is_three_quarters_at_111():
    sysp, basis = _h_atom_doublet()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _options()
    r = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    assert r.s_squared == pytest.approx(0.75, abs=1e-10)
    assert r.s_squared_ideal == pytest.approx(0.75, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Open-shell convergence
# ---------------------------------------------------------------------------

def test_h_atom_doublet_converges():
    sysp, basis = _h_atom_doublet()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _options()
    r = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    assert np.isfinite(r.energy)
    # Multi-k [1,1,1] uses the proper periodic density convention so
    # the H atom energy includes more lattice cells than Γ-only.
    # Bound is wide.
    assert -1.5 < r.energy < 0.0


# ---------------------------------------------------------------------------
# 4. α/β density symmetry on closed shell
# ---------------------------------------------------------------------------

def test_closed_shell_alpha_beta_real_space_densities_equal():
    sysp, basis = _h2_closed_shell()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _options()
    r = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    n_cells = len(r.density_alpha.cells)
    for g in range(n_cells):
        diff = np.abs(
            np.asarray(r.density_alpha.blocks[g])
            - np.asarray(r.density_beta.blocks[g])
        ).max()
        assert diff < 1e-10, (
            f"α/β density asymmetry on closed shell at cell {g}: "
            f"{diff:.3e}"
        )


# ---------------------------------------------------------------------------
# 5. Result-shape contract
# ---------------------------------------------------------------------------

def test_result_struct_populated():
    sysp, basis = _h_atom_doublet()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _options()
    r = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    n_k = len(km.kpoints)
    nbf = basis.nbasis
    assert len(r.mo_energies_alpha) == n_k
    assert len(r.mo_coeffs_alpha) == n_k
    assert len(r.fock_alpha) == n_k
    assert len(r.mo_energies_beta) == n_k
    assert len(r.mo_coeffs_beta) == n_k
    assert len(r.fock_beta) == n_k
    assert len(r.occupations_alpha) == n_k
    assert len(r.occupations_beta) == n_k
    assert len(r.overlap) == n_k
    assert len(r.hcore) == n_k
    for idx in range(n_k):
        assert r.fock_alpha[idx].shape == (nbf, nbf)
        assert r.fock_beta[idx].shape == (nbf, nbf)
        assert r.occupations_alpha[idx].shape == r.mo_energies_alpha[idx].shape
        assert r.occupations_beta[idx].shape == r.mo_energies_beta[idx].shape
    assert isinstance(r.scf_trace, list)
    assert len(r.scf_trace) == r.n_iter
    # Density LMS is well-formed.
    assert len(r.density_alpha.cells) >= 1
    assert len(r.density_beta.cells) == len(r.density_alpha.cells)
    orbital_text = pr._mo_summary(r)
    assert "Crystal orbital energies (alpha)" in orbital_text
    assert "Crystal orbital energies (beta)" in orbital_text


# ---------------------------------------------------------------------------
# 6. set_block bug regression: multi-k Ewald RHF damping path actually mutates
# ---------------------------------------------------------------------------

def test_lattice_matrix_set_set_block_persists():
    """Direct test of the LatticeMatrixSet.set_block C++ binding.
    Element assignment via ``lms.blocks[i] = M`` does NOT persist
    (writes to a transient Python list copy); ``lms.set_block(i, M)``
    must mutate the underlying C++ vector in place."""
    box, c = 30.0, 15.0
    sysp = vq.PeriodicSystem(3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])])
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 12.0
    opts.nuclear_cutoff_bohr = 15.0
    S = vq.compute_overlap_lattice(basis, sysp, opts)
    new = np.full((2, 2), 99.0)
    S.set_block(0, new)
    assert np.allclose(np.asarray(S.blocks[0]), new)


def test_fused_jk_exchange_matches_jfull_ffull_reconstruction():
    """The fused J+K builder's bare exchange equals the former
    K(D) = 2·(J_full(D) − F_full(D)) reconstruction at machine
    precision — the contract behind the 2026-06-10 call-site fusion
    that halved the lattice-ERI traversals per spin in the multi-k
    UHF/UKS Fock builds."""
    import numpy as np
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        build_fock_2e_real_space,
        build_jk_2e_real_space,
        compute_overlap_lattice,
    )

    sysp, basis = _h2_closed_shell(20.0)
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 8.0
    lat.nuclear_cutoff_bohr = 8.0

    # An arbitrary symmetric one-particle density on the lattice set.
    D = compute_overlap_lattice(basis, sysp, lat)
    rng = np.random.default_rng(5)
    nbf = basis.nbasis
    M = rng.standard_normal((nbf, nbf))
    M = 0.1 * (M + M.T)
    for g in range(len(D.cells)):
        scale = 1.0 if (D.cells[g].index == np.array([0, 0, 0])).all() else 0.05
        D.set_block(g, scale * M)

    J_full = build_fock_2e_real_space(basis, sysp, lat, D, 0.0, 0.0)
    F_full = build_fock_2e_real_space(basis, sysp, lat, D, 1.0, 0.0)
    jk = build_jk_2e_real_space(basis, sysp, lat, D, 0.0)

    for g in range(len(D.cells)):
        K_recon = 2.0 * (
            np.asarray(J_full.blocks[g], dtype=float)
            - np.asarray(F_full.blocks[g], dtype=float)
        )
        K_fused = np.asarray(jk.K.blocks[g], dtype=float)
        assert np.abs(K_recon - K_fused).max() < 1e-11, f"cell {g}"


def test_fused_multi_k_fock_assembly_matches_python_fallback(monkeypatch):
    """The fused OpenMP ``assemble_fock_multi_k`` per-iteration path and
    the per-k Python ``_bloch_sum_blocks`` fallback converge the same
    SCF. The kernel's cos/sin phase differs from ``np.exp`` by at most
    one ulp, so the converged energies agree far inside the pinned
    tolerances."""
    import vibeqc._vibeqc_core as core

    sysp, basis = _h_atom_doublet()
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])
    r_fused = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _options(), progress=False,
    )

    monkeypatch.delattr(core, "assemble_fock_multi_k")
    r_py = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, kmesh, _options(), progress=False,
    )

    assert r_fused.converged and r_py.converged
    assert r_fused.n_iter == r_py.n_iter
    assert abs(r_fused.energy - r_py.energy) < 1e-10
    for F_f, F_p in zip(r_fused.fock_alpha, r_py.fock_alpha):
        assert np.allclose(F_f, F_p, atol=1e-10)
    for F_f, F_p in zip(r_fused.fock_beta, r_py.fock_beta):
        assert np.allclose(F_f, F_p, atol=1e-10)


def test_fock_blocks_with_reused_d_total_buffer_match_one_shot():
    """``_build_uhf_fock_blocks_ewald3d`` with a caller-provided reusable
    ``d_total_buffer`` container returns the same per-cell blocks as the
    one-shot path that rebuilds the container from the overlap template,
    including on a second reuse (every block is overwritten, no stale
    data)."""
    from vibeqc.periodic_uhf_multi_k_ewald import _build_uhf_fock_blocks_ewald3d

    sysp, basis = _h2_closed_shell(20.0)
    lat = vq.LatticeSumOptions()
    lat.cutoff_bohr = 8.0
    lat.nuclear_cutoff_bohr = 8.0
    lat.coulomb_method = vq.CoulombMethod.EWALD_3D

    # Arbitrary symmetric one-particle spin densities on the lattice set.
    D_a = vq.compute_overlap_lattice(basis, sysp, lat)
    D_b = vq.compute_overlap_lattice(basis, sysp, lat)
    rng = np.random.default_rng(11)
    nbf = basis.nbasis
    M = rng.standard_normal((nbf, nbf))
    M = 0.1 * (M + M.T)
    for g in range(len(D_a.cells)):
        scale = 1.0 if (D_a.cells[g].index == np.array([0, 0, 0])).all() else 0.05
        D_a.set_block(g, scale * M)
        D_b.set_block(g, 0.5 * scale * M)

    args = (basis, sysp, D_a, D_b, 0.5, lat, (25, 25, 25), None, 0.3)
    F_a_ref, F_b_ref = _build_uhf_fock_blocks_ewald3d(*args)

    buf = vq.compute_overlap_lattice(basis, sysp, lat)
    for _ in range(2):  # second pass proves per-call overwrite
        F_a_buf, F_b_buf = _build_uhf_fock_blocks_ewald3d(
            *args, d_total_buffer=buf,
        )
        for g in range(len(D_a.cells)):
            assert np.allclose(F_a_ref[g], F_a_buf[g], atol=1e-13), f"cell {g}"
            assert np.allclose(F_b_ref[g], F_b_buf[g], atol=1e-13), f"cell {g}"
