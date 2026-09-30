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

6. **Compact cells (GitLab #399)** — the exact exchange takes the RHF
   sibling's gauge: the corrected Ewald split at every mesh (since #23 at a
   single Gamma point too), the molecular-limit kernel on request. Closed-shell UHF equals
   RHF on a 6-bohr cell, is lattice-cutoff converged, and open-shell states
   match the independent BIPOLE UHF route, the Gamma UHF driver and PySCF
   KUHF.
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
    # <S^2> is the ensemble value of the multi-k formula every open-shell
    # route shares (#399): Sz(Sz+1) + n_b - sum_k w_k sum_ij n^a_i n^b_j
    # |<a_i|S|b_j>|^2. For this closed-shell state a == b, so it is
    # n_b - sum_k w_k sum_i n_i^2, small and positive at kT = 0.05 Ha, not 0.
    # (The Gamma-block diagnostic it replaced ignored the occupations.)
    for oa, ob in zip(r.occupations_alpha, r.occupations_beta):
        assert np.allclose(oa, ob, atol=1e-12)
    ensemble = 1.0 - sum(
        float(w) * float(np.sum(np.asarray(o) ** 2))
        for w, o in zip(km.weights, r.occupations_alpha)
    )
    assert r.s_squared == pytest.approx(ensemble, abs=1e-10)
    assert r.s_squared > 0.0
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


# ---------------------------------------------------------------------------
# GitLab #399: compact cells
# ---------------------------------------------------------------------------
#
# Until #399 the driver summed its full-range exchange as a bare 1/r kernel
# over a radial image ball. On a finite k mesh the SCF density is BvK-torus
# periodic, D(g) does not decay, and that sum does not converge; at a single
# Gamma point the inverse fold filled every image block with P(Gamma). Only
# the 30-bohr anchor above hid it. Measured on origin/main b268dce, H2/STO-3G
# in a 6-bohr cube:
#
#   mesh     options            RHF             UHF (M=1)       UHF - RHF
#   (1,1,1)  default          -1.1629042834   -7.9994791730    -6.837
#   (2,1,1)  default          -1.1945076098   -4.4318619148    -3.237
#   (2,1,1)  EWALD_3D, cut 18 -1.1945076136   -6.2234315088    -5.029


def _compact_h2(box: float = 6.0):
    return _h2_closed_shell(box)


def _compact_options(explicit_ewald: bool = False, cutoff=None):
    o = vq.PeriodicRHFOptions()
    o.max_iter = 200
    if explicit_ewald:
        o.lattice_opts.cutoff_bohr = 12.0
        o.lattice_opts.nuclear_cutoff_bohr = 15.0
        o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    if cutoff is not None:
        o.lattice_opts.cutoff_bohr = float(cutoff)
        o.lattice_opts.nuclear_cutoff_bohr = float(cutoff) + 3.0
        o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    return o


@pytest.mark.parametrize(
    "mesh, explicit_ewald",
    [((1, 1, 1), False), ((2, 1, 1), False), ((2, 1, 1), True)],
    ids=["111-default", "211-default", "211-ewald"],
)
def test_closed_shell_uhf_matches_rhf_on_a_compact_cell(mesh, explicit_ewald):
    """#399: UHF(M=1) of a closed-shell state is RHF, on a compact cell too."""
    sysp, basis = _compact_h2()
    km = vq.monkhorst_pack(sysp, list(mesh))
    r = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, options=_compact_options(explicit_ewald)
    )
    u = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, km, options=_compact_options(explicit_ewald)
    )
    assert r.converged and u.converged
    assert u.energy == pytest.approx(r.energy, abs=1e-9)
    assert abs(u.s_squared) < 1e-10


def test_uhf_exchange_is_lattice_cutoff_converged_on_a_compact_cell():
    """#399: the defect's signature was an energy that kept falling as the
    lattice cutoff grew (-2.26 Ha at 10 bohr, -6.22 Ha at 18 bohr). The
    corrected split converges like RHF: 1.4e-7 Ha from 14 to 18 bohr."""
    sysp, basis = _compact_h2()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    e = {}
    for cut in (14.0, 18.0):
        e[cut] = vq.run_uhf_periodic_multi_k_ewald3d(
            sysp, basis, km, options=_compact_options(cutoff=cut),
            auto_optimize_truncation=False,
        ).energy
    assert e[18.0] == pytest.approx(e[14.0], abs=1e-6)
    assert e[18.0] == pytest.approx(-1.1945076136, abs=1e-8)


def _open_shell_h(name: str):
    if name == "h2_triplet":
        atoms, mult = [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])], 3
    else:
        atoms, mult = [vq.Atom(1, [0.0, 0.0, 0.0])], 2
    sysp = vq.PeriodicSystem(3, np.eye(3) * 8.0, atoms, 0, mult)
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


@pytest.mark.parametrize("name", ["h2_triplet", "h_doublet"])
def test_open_shell_uhf_matches_bipole_uhf_on_a_multik_mesh(name):
    """#399: two independent Coulomb routes on one open-shell state.

    BIPOLE UHF assembles its exchange in direct space with its own Ewald
    split; this driver uses the corrected split of
    vibeqc.periodic_corrected_exchange. They agree to ~5e-9 Ha at (2,2,2)
    (measured 2026-09-27); main was 1.7 mHa (triplet) and 1.0 mHa (doublet)
    off even in this dilute 8-bohr cell.
    """
    sysp, basis = _open_shell_h(name)
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    u = vq.run_uhf_periodic_multi_k_ewald3d(sysp, basis, km)
    b = vq.run_pbc_bipole_uhf(sysp, basis, km, progress=False)
    assert u.converged and b.converged
    assert u.energy == pytest.approx(b.energy, abs=1e-7)


# PySCF 2.14.0 KUHF/GDF, exxdiv='ewald', cell.precision = 1e-10,
# conv_tol = 1e-10, run out of process by
# examples/regression/parity_uhf_multik_ewald_vs_pyscf.py (vibe-qc never
# imports it). The triplet is not pinned here: both GDF codes (PySCF and
# vibe-qc's run_kuhf_periodic_gdf, which agree to 1e-10) sit 1.30e-4 Ha
# below the two Ewald routes on that state, a density-fitting error; the
# BIPOLE check above covers it.
PYSCF_KUHF_H_DOUBLET_222_EWALD = -0.4675702852
PYSCF_METHOD_GAP_HA = 5.0e-5


def test_open_shell_uhf_matches_pyscf_kuhf():
    sysp, basis = _open_shell_h("h_doublet")
    u = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, vq.monkhorst_pack(sysp, [2, 2, 2])
    )
    assert u.converged
    assert u.energy == pytest.approx(
        PYSCF_KUHF_H_DOUBLET_222_EWALD, abs=PYSCF_METHOD_GAP_HA
    )


@pytest.mark.parametrize("name", ["h2_triplet", "h_doublet"])
def test_single_gamma_uhf_gauges(name):
    """#399 + #23: at a single Gamma point the default is the corrected
    Ewald split, which BIPOLE UHF also takes there; exchange_exxdiv='none'
    selects the molecular-limit kernel and reproduces
    run_uhf_periodic_gamma_ewald3d (main returned -2.91 / -1.66 Ha before
    #399, in neither gauge)."""
    sysp, basis = _open_shell_h(name)
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    default = vq.run_uhf_periodic_multi_k_ewald3d(sysp, basis, km)
    bipole = vq.run_pbc_bipole_uhf(sysp, basis, km, progress=False)
    assert default.energy == pytest.approx(bipole.energy, abs=1e-6)

    molecular = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, km, exchange_exxdiv="none"
    )
    gamma = vq.run_uhf_periodic_gamma_ewald3d(sysp, basis)
    assert molecular.energy == pytest.approx(gamma.energy, abs=1e-10)
    # The two gauges are genuinely different at a single Gamma point.
    assert abs(default.energy - molecular.energy) > 5.0e-3


def test_exchange_exxdiv_is_validated():
    sysp, basis = _open_shell_h("h_doublet")
    with pytest.raises(ValueError, match="exchange_exxdiv must be"):
        vq.run_uhf_periodic_multi_k_ewald3d(
            sysp, basis, vq.monkhorst_pack(sysp, [1, 1, 1]),
            exchange_exxdiv="madelung",
        )
    with pytest.raises(NotImplementedError, match="only at a single Gamma"):
        vq.run_uhf_periodic_multi_k_ewald3d(
            sysp, basis, vq.monkhorst_pack(sysp, [2, 1, 1]),
            exchange_exxdiv="none",
        )
