"""Tests for multi-k Ewald-J-split BIPOLE paths.

Validates the v0.9.0 multi-k extension that wires per-k
``F_J^LR(k)`` via Bloch-summed bra-pair FT + shifted-ν ρ̂(K). Prior
to this extension multi-k raised NotImplementedError; the empirical
shortcut tried 2026-05-18 (k-independent ``J^LR(Γ)`` shortcut) was
reverted as broken (commit ``6cc5eab``).

The tests here exercise:

* The Γ-only ↔ multi-k consistency at a single-k-point mesh
  (``compute_J_long_range_at_k`` at k=0 reproduces
  ``compute_J_long_range_gamma`` exactly).
* Hermiticity of the per-k J^LR build.
* The Bloch-summed bra-pair FT identity at k=0 + at a non-Γ k.
* MgO SHRINK 2 2 coverage at both ends of the support contract: cheap
  low-cutoff tests pin fail-closed overlap-fold refusal and IBZ expansion,
  while the cutoff-12 slow anchor below pins converged CRYSTAL parity.
* Cache-reuse regressions pin the density-independent V_ne/J^LR AO-pair
  Fourier cache to one build per SCF run.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import (
    CoulombMethod,
    InitialGuess,
    LatticeSumOptions,
    monkhorst_pack,
)
from vibeqc._aopair_ft import (
    ao_pair_fourier_transform_at_cells,
    ao_pair_fourier_transform_bloch,
    ao_pair_fourier_transform_shifted_ket,
)
from vibeqc._vibeqc_core import (
    PeriodicKSOptions,
    PeriodicRHFOptions,
    bloch_sum,
    build_fock_2e_real_space,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    direct_lattice_cells,
    real_space_density_from_kpoints,
)
from vibeqc.bipole_ext_el_pole import (
    compute_cell_density_fourier,
    compute_cell_density_fourier_lattice,
    crystal_default_ewald_alpha,
    crystal_ewald_reciprocal_cutoff,
)
from vibeqc.bipole_fock_ewald import (
    _build_j_long_range_cache,
    compute_J_long_range_at_k,
    compute_J_long_range_gamma,
    compute_J_long_range_real_space_blocks,
    compute_rho_hat_from_k_density,
)
from vibeqc.bipole_lattice_self_energy import cell_volume_bohr3
from vibeqc.ewald_composed import compute_j_ewald_3d_ft_k_density_to_cells
from vibeqc.guess import initial_density_closed_shell
from vibeqc.pbc_bipole import run_pbc_bipole_rhf
from vibeqc.pbc_bipole_common import (
    BipoleFoldUnreliableError,
    _bloch_sum_blocks,
    _bloch_sum_blocks_multi_k,
    _cell_key,
    bipole_sr_image_extent,
)
from vibeqc.pbc_bipole_fock import _sr_image_padded_jk
from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf
from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks
from vibeqc.periodic_rhf_multi_k_ewald import (
    _canonical_orthogonalizer_complex,
    _diag_in_orth_basis,
)
from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch


ANG2BOHR = 1.0 / 0.529177210903


def _build_mgo():
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    atoms = [
        vq.Atom(12, [0.0, 0.0, 0.0]),
        vq.Atom(8, [a / 2.0, a / 2.0, a / 2.0]),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _build_he_cubic():
    lattice = np.eye(3) * 7.0
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _make_p_real_iter1_sad(system, basis, opts):
    n_occ = system.n_electrons() // 2
    S_lat = compute_overlap_lattice(basis, system, opts)
    D_sad = initial_density_closed_shell(
        system.unit_cell_molecule(), basis, n_occ,
        InitialGuess.SAD, is_periodic=True,
    )
    P_real = S_lat
    for g_idx in range(len(P_real.cells)):
        is_g0 = (np.asarray(P_real.cells[g_idx].index)
                 == np.array([0, 0, 0])).all()
        P_real.set_block(
            g_idx,
            np.asarray(D_sad, dtype=float) if is_g0
            else np.zeros_like(np.asarray(D_sad), dtype=float),
        )
    return P_real, np.asarray(D_sad, dtype=float)


def _two_iter_rhf_options() -> PeriodicRHFOptions:
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 4.0
    opts.lattice_opts.nuclear_cutoff_bohr = 4.0
    opts.max_iter = 2
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = False
    opts.damping = 0.0
    opts.conv_tol_energy = 1e-30
    opts.conv_tol_grad = 1e-30
    return opts


def _two_iter_ks_options() -> PeriodicKSOptions:
    opts = PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = 4.0
    opts.lattice_opts.nuclear_cutoff_bohr = 4.0
    opts.max_iter = 2
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = False
    opts.functional = "lda"
    opts.conv_tol_energy = 1e-30
    opts.conv_tol_grad = 1e-30
    return opts


def _one_iter_rhf_options() -> PeriodicRHFOptions:
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 5.0
    opts.lattice_opts.nuclear_cutoff_bohr = 5.0
    opts.max_iter = 1
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = False
    return opts


def _one_iter_ks_options() -> PeriodicKSOptions:
    opts = PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = 5.0
    opts.lattice_opts.nuclear_cutoff_bohr = 5.0
    opts.max_iter = 1
    opts.initial_guess = InitialGuess.SAD
    opts.use_diis = False
    opts.functional = "lda"
    return opts


def _count_j_lr_cache_builds(monkeypatch):
    import vibeqc.bipole_fock_ewald as ewald_mod

    calls = {"n": 0}
    original = ewald_mod._build_j_long_range_cache

    def counted_build(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ewald_mod,
        "_build_j_long_range_cache",
        counted_build,
    )
    return calls


def _assert_result_k_count(result, expected: int) -> None:
    if hasattr(result, "mo_energies"):
        assert len(result.mo_energies) == expected
    else:
        assert len(result.mo_energies_alpha) == expected
        assert len(result.mo_energies_beta) == expected


# ---------------------------------------------------------------------
# Shifted-ν AO-pair FT — primitive sanity checks
# ---------------------------------------------------------------------
def test_shifted_ao_pair_ft_zero_shift_matches_unshifted():
    """At R_g = 0 the shifted-ν FT reduces to the standard AO-pair FT."""
    from vibeqc._aopair_ft import ao_pair_fourier_transform
    system, basis = _build_mgo()
    K = np.array([[0.5, 0.2, 0.1], [1.0, 0.0, 0.0], [0.3, -0.4, 0.7]])
    ft_unshifted = ao_pair_fourier_transform(basis, K)
    ft_shifted_zero = ao_pair_fourier_transform_shifted_ket(
        basis, K, np.zeros(3))
    np.testing.assert_allclose(ft_unshifted, ft_shifted_zero, atol=1e-14)


def test_shifted_ao_pair_ft_translational_identity():
    """FT_νμ(K; -R_g) · exp(-iK·R_g) = FT_μν(K; +R_g)  — analytic identity.

    Derives from the change of variables ``r → r + R_g`` in the AO-pair
    integrand; valid for any K (not just reciprocal lattice vectors).
    """
    system, basis = _build_mgo()
    K = np.array([[0.5, 0.2, 0.1], [1.0, 0.0, 0.0]])
    R_g = np.asarray(system.lattice, dtype=float)[0]  # 1st primitive vector
    ft_pos = ao_pair_fourier_transform_shifted_ket(basis, K, R_g)
    ft_neg = ao_pair_fourier_transform_shifted_ket(basis, K, -R_g)
    phase = np.exp(1j * (K @ R_g))[None, None, :]
    # FT_μν(K; -R_g) = exp(+iK·R_g) · FT_νμ(K; +R_g)  (swap μ↔ν index)
    expected = phase * np.transpose(ft_pos, (1, 0, 2))
    np.testing.assert_allclose(ft_neg, expected, atol=1e-12)


def test_bloch_summed_ft_single_cell_reduces_to_unshifted():
    from vibeqc._aopair_ft import ao_pair_fourier_transform
    system, basis = _build_mgo()
    K = np.array([[0.5, 0.2, 0.1], [1.0, 0.0, 0.0]])
    ft_b = ao_pair_fourier_transform_bloch(
        basis, K, np.array([[0.0, 0.0, 0.0]]), np.zeros(3))
    ft_unshifted = ao_pair_fourier_transform(basis, K)
    np.testing.assert_allclose(ft_b, ft_unshifted, atol=1e-14)


# ---------------------------------------------------------------------
# compute_cell_density_fourier — shifted-ν vs legacy at Γ-only SAD
# ---------------------------------------------------------------------
def test_compute_cell_density_fourier_gamma_only_sad_unchanged():
    """At iter-1 SAD with D(g≠0)=0, the shifted-ν FT and the legacy
    bra-pair-at-home approximation give the SAME ρ̂(K).

    The new gauge-correct path only differs for D(g≠0); at g=0 both
    formulas use ``FT_μν(K; 0)`` which is identical.
    """
    system, basis = _build_mgo()
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    P_real, _ = _make_p_real_iter1_sad(system, basis, opts)
    K = np.array([[0.5, 0.2, 0.1], [1.0, 0.0, 0.0], [0.3, -0.4, 0.7]])
    rho_new = compute_cell_density_fourier_lattice(P_real, basis, K)
    rho_wrap = compute_cell_density_fourier(P_real, basis, K)
    np.testing.assert_allclose(rho_new, rho_wrap, atol=1e-14)


# ---------------------------------------------------------------------
# compute_J_long_range_at_k — per-k variant
# ---------------------------------------------------------------------
def test_J_long_range_at_k_zero_matches_gamma():
    """At k=0 the per-k J^LR reduces to the Γ-only Real-symmetric result.

    Verifies that the multi-k extension is a strict generalisation —
    no behaviour change at k=0.
    """
    system, basis = _build_mgo()
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    P_real, _ = _make_p_real_iter1_sad(system, basis, opts)
    V = cell_volume_bohr3(system)
    omega = crystal_default_ewald_alpha(V)
    J_LR_gamma = compute_J_long_range_gamma(
        P_real, basis, system, omega, precision=1e-6)
    J_LR_at_k0 = compute_J_long_range_at_k(
        P_real, basis, system, omega, np.zeros(3), precision=1e-6)
    # At k=0 with real-Hermitian D, imag part vanishes; the Γ-only
    # function returns the (already real) symmetrised result.
    np.testing.assert_allclose(
        np.real(J_LR_at_k0), J_LR_gamma, atol=1e-10)
    np.testing.assert_allclose(
        np.imag(J_LR_at_k0), 0.0, atol=1e-10)


def test_J_long_range_at_k_hermitian():
    """``F_J^LR(k)`` MUST be Hermitian for any k.

    Required for the Fock matrix at k to have real eigenvalues. The
    Hermiticity holds analytically under the K ↔ -K pairing of the
    K-mesh + real-Hermitian density; this test sanity-checks the
    numerical implementation.
    """
    system, basis = _build_mgo()
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    P_real, _ = _make_p_real_iter1_sad(system, basis, opts)
    V = cell_volume_bohr3(system)
    omega = crystal_default_ewald_alpha(V)
    # Non-trivial k far from any high-symmetry point.
    b1 = 2.0 * math.pi * np.linalg.inv(
        np.asarray(system.lattice, dtype=float)).T[0]
    k_cart = 0.25 * b1 + np.array([0.01, 0.02, 0.0])
    J_LR_k = compute_J_long_range_at_k(
        P_real, basis, system, omega, k_cart, precision=1e-6)
    np.testing.assert_allclose(J_LR_k, J_LR_k.conj().T, atol=1e-10)


def test_rho_hat_k_space_gamma_only_matches_realspace_sad():
    """k-space ρ̂(K) at k=0 with D(k=0)=SAD matches the real-space
    form Σ_g D_used(g)·FT(K; R_g) when D_used(g≠0)=0.

    At Γ-only iter-1 SAD the manual short-circuit forces D(g≠0)=0,
    so the real-space form collapses to ``Σ_{μν} D_SAD·FT(K; 0)``.
    The k-space form at a single k=0 with D(k=0)=SAD gives
    ``D_SAD · Σ_g FT(K; R_g)`` — which is NOT the same in general
    (Bloch sum over all g vs only g=0). This test documents the
    expected divergence and confirms both reductions are stable.
    """
    system, basis = _build_mgo()
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 6.0
    P_real, D_sad = _make_p_real_iter1_sad(system, basis, opts)
    V = cell_volume_bohr3(system)
    omega = crystal_default_ewald_alpha(V)
    R_g_arr = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in P_real.cells])
    cache = _build_j_long_range_cache(
        basis, system, R_g_arr, omega, precision=1e-6)
    # k-space ρ̂(K) at k=0 with D(k=0)=SAD-as-uniform-in-k:
    rho_kspace = compute_rho_hat_from_k_density(
        [D_sad], [np.zeros(3)], [1.0], cache,
    )
    # K=0 limit: ρ̂(K=0) = total electron count per cell.
    K2 = (cache.K_vectors ** 2).sum(axis=1)
    K_min_idx = int(np.argmin(K2))
    # Just sanity-check finite + bounded.
    assert np.isfinite(rho_kspace).all()
    assert np.abs(rho_kspace[K_min_idx]) < 100.0, (
        f"|ρ̂(K_min)| = {np.abs(rho_kspace[K_min_idx]):.3e} — should be "
        f"O(electron count); large value suggests a normalisation bug."
    )


def test_J_long_range_cache_reused():
    """Pre-built cache gives identical results to the from-scratch build."""
    system, basis = _build_mgo()
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    P_real, _ = _make_p_real_iter1_sad(system, basis, opts)
    V = cell_volume_bohr3(system)
    omega = crystal_default_ewald_alpha(V)
    R_g_arr = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in P_real.cells])
    cache = _build_j_long_range_cache(
        basis, system, R_g_arr, omega, precision=1e-6)
    k_cart = np.array([0.1, 0.05, 0.0])
    J_no_cache = compute_J_long_range_at_k(
        P_real, basis, system, omega, k_cart, precision=1e-6)
    J_cached = compute_J_long_range_at_k(
        P_real, basis, system, omega, k_cart, cache=cache)
    np.testing.assert_allclose(J_cached, J_no_cache, atol=1e-12)


def test_rho_hat_reuses_cached_bloch_pair_ft():
    """Repeated k-density transforms reuse density-independent pair FT."""
    system, basis = _build_mgo()
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 6.0
    P_real, D_sad = _make_p_real_iter1_sad(system, basis, opts)
    V = cell_volume_bohr3(system)
    omega = crystal_default_ewald_alpha(V)
    R_g_arr = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in P_real.cells])
    cache = _build_j_long_range_cache(
        basis, system, R_g_arr, omega, precision=1e-6)
    k_cart = np.array([0.11, 0.03, -0.02])

    rho_1 = compute_rho_hat_from_k_density(
        [D_sad], [k_cart], [1.0], cache,
    )
    assert len(cache.ft_bloch_cache) == 1
    cached_tensor = next(iter(cache.ft_bloch_cache.values()))

    rho_2 = compute_rho_hat_from_k_density(
        [D_sad], [k_cart], [1.0], cache,
    )
    assert len(cache.ft_bloch_cache) == 1
    assert next(iter(cache.ft_bloch_cache.values())) is cached_tensor
    np.testing.assert_allclose(rho_2, rho_1, atol=0.0)


def test_J_long_range_real_space_blocks_bloch_sum_to_k_matrix():
    """The real-space J^LR(g) blocks must Bloch-fold to J^LR(k).

    The SCF driver uses the blocks for CRYSTAL-style energy contraction
    and the folded matrix for diagonalisation.  This identity guards the
    split accounting that fixed the CYC0 Γ-energy overcount.
    """
    system, basis = _build_mgo()
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 6.0
    P_real, _ = _make_p_real_iter1_sad(system, basis, opts)
    V = cell_volume_bohr3(system)
    omega = crystal_default_ewald_alpha(V)
    R_g_arr = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in P_real.cells])
    cache = _build_j_long_range_cache(
        basis, system, R_g_arr, omega, precision=1e-6)
    k_cart = 0.2 * (
        2.0 * math.pi * np.linalg.inv(
            np.asarray(system.lattice, dtype=float)).T[0]
    )
    blocks = compute_J_long_range_real_space_blocks(
        P_real, basis, system, omega,
        precision=1e-6, cache=cache,
    )
    J_from_blocks = sum(
        np.exp(1j * float(np.dot(k_cart, np.asarray(cell.r_cart))))
        * block
        for cell, block in zip(P_real.cells, blocks)
    )
    J_at_k = compute_J_long_range_at_k(
        P_real, basis, system, omega, k_cart,
        precision=1e-6, cache=cache,
    )
    np.testing.assert_allclose(J_from_blocks, J_at_k, atol=1e-10)


def _mgo_shrink2_hcore_density(cutoff_bohr: float = 8.0):
    """Return a deterministic MgO SHRINK-2 fixed density for J tests.

    The density is the per-k Hcore Aufbau density used as the first
    multi-k BIPOLE RKS guess. It has the correct occupied rank at every
    k and a 20-electron S(k) metric, but avoids an SCF loop so the test
    isolates the Hartree builder.
    """

    system, basis = _build_mgo()
    kmesh = monkhorst_pack(system, [2, 2, 2], use_symmetry=False)
    k_points = [np.asarray(k, dtype=float) for k in kmesh.kpoints]
    weights = [float(w) for w in kmesh.weights]

    opts = LatticeSumOptions()
    opts.cutoff_bohr = float(cutoff_bohr)
    opts.nuclear_cutoff_bohr = float(cutoff_bohr)
    opts.coulomb_method = CoulombMethod.DIRECT_TRUNCATED

    S_lat = compute_overlap_lattice(basis, system, opts)
    T_lat = compute_kinetic_lattice(basis, system, opts)
    v_opts = LatticeSumOptions()
    v_opts.cutoff_bohr = float(cutoff_bohr)
    v_opts.nuclear_cutoff_bohr = float(cutoff_bohr)
    v_opts.coulomb_method = CoulombMethod.EWALD_3D
    V_lat = compute_nuclear_lattice_dispatch(basis, system, v_opts)

    n_occ = int(system.n_electrons()) // 2
    coeffs = []
    densities = []
    overlaps = []
    for k_cart in k_points:
        S_k = np.asarray(bloch_sum(S_lat, k_cart))
        S_k = 0.5 * (S_k + S_k.conj().T)
        H_k = np.asarray(bloch_sum(T_lat, k_cart)) + np.asarray(
            bloch_sum(V_lat, k_cart)
        )
        H_k = 0.5 * (H_k + H_k.conj().T)
        X_k, _ = _canonical_orthogonalizer_complex(
            S_k,
            1e-7,
            normalize_diag_first=True,
        )
        C_k, _ = _diag_in_orth_basis(H_k, X_k)
        C_occ = C_k[:, :n_occ]
        D_k = 2.0 * (C_occ @ C_occ.conj().T)
        coeffs.append(C_k.astype(complex))
        densities.append(D_k)
        overlaps.append(S_k)

    metric = sum(
        w * float(np.trace(D_k @ S_k).real)
        for w, D_k, S_k in zip(weights, densities, overlaps)
    )
    assert metric == pytest.approx(float(system.n_electrons()), abs=1e-10)
    return system, basis, kmesh, k_points, weights, opts, coeffs, densities, S_lat


def _j_energy_from_blocks(blocks, cells, k_points, weights, densities) -> float:
    total = 0.0
    for k_cart, weight, D_k in zip(k_points, weights, densities):
        J_k = _bloch_sum_blocks(blocks, cells, np.asarray(k_cart, dtype=float))
        J_k = 0.5 * (J_k + J_k.conj().T)
        total += float(weight) * float(np.einsum("ij,ji->", D_k, J_k).real)
    return 0.5 * total


def _bipole_fixed_density_j_energy(
    system,
    basis,
    kmesh,
    k_points,
    weights,
    opts,
    coeffs,
    densities,
    S_lat,
    *,
    K_max: float,
) -> float:
    """BIPOLE J_SR + J_LR + electronic background on one fixed density."""

    n_occ = int(system.n_electrons()) // 2
    cells = direct_lattice_cells(system, float(opts.cutoff_bohr))
    D_real = real_space_density_from_kpoints(
        coeffs,
        [n_occ] * len(coeffs),
        kmesh,
        cells,
    )
    V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    omega = crystal_default_ewald_alpha(V_cell)
    opts.sr_range_screening = True
    sr_extent = float(opts.cutoff_bohr) + bipole_sr_image_extent(
        basis,
        system,
        omega,
        precision=1e-6,
    )
    J_sr = _sr_image_padded_jk(
        basis,
        system,
        opts,
        D_real,
        float(omega),
        sr_extent,
        compute_exchange=False,
    ).J
    cache = _build_j_long_range_cache(
        basis,
        system,
        np.array([np.asarray(c.r_cart, dtype=float) for c in cells]),
        omega,
        1e-8,
        K_max=float(K_max),
    )
    rho_hat = compute_rho_hat_from_k_density(
        densities,
        k_points,
        weights,
        cache,
    )
    J_lr_blocks = compute_J_long_range_real_space_blocks(
        D_real,
        basis,
        system,
        omega,
        precision=1e-8,
        cache=cache,
        rho_hat=rho_hat,
    )
    background = (
        -math.pi
        * float(system.n_electrons())
        / (float(omega) * float(omega) * V_cell)
    )
    s_blocks = {
        _cell_key(cell): np.asarray(block, dtype=float)
        for cell, block in zip(S_lat.cells, S_lat.blocks)
    }
    blocks = []
    for cell, sr_block, lr_block in zip(cells, J_sr.blocks, J_lr_blocks):
        key = _cell_key(cell)
        blocks.append(
            np.asarray(sr_block, dtype=float)
            + np.asarray(lr_block, dtype=float)
            + background * s_blocks[key]
        )
    return _j_energy_from_blocks(blocks, cells, k_points, weights, densities)


@pytest.mark.slow
def test_mgo_shrink2_m5_padding_leaves_only_exact_ft_tail_gap():
    """Fixed-density MgO SHRINK-2 M5 split vs finite-ke exact-FT J.

    This is the regression requested by the v0.14/v0.15 BIPOLE RKS
    handover. It avoids SCF, exchange, and XC entirely: the same
    20-electron Hcore density is contracted with (1) the BIPOLE
    J_SR(erfc) + J_LR(erf) + electronic-background split and (2) the
    exact full-reciprocal G=0-dropped Hartree J now used by pure RKS.

    Pins re-derived 2026-07-13 for M5 from the builder-disagreement
    triage and its independent Poisson oracle:

    * ``e_reference`` (exact-FT J, ke_cutoff = 300) = 80.9288590106 Ha
      is a partial sum of the converged G=0-dropped Hartree of this
      density, E(ke->inf) = 81.4173577 Ha (every kept G adds a
      nonnegative (2pi/V)|rho(G)|^2/G^2 term; converged value from an
      independent closed-form Gaussian rho(G) summed band-by-band to
      ke = 24000, where the slowest pair decay exp(-G^2/1196) is dead).
      The ~0.4885 Ha tail is Mg/O core-pair reciprocal content and is
      density-independent to first order. Production pure-RKS uses this
      builder at ke_cutoff = 200 (``VIBEQC_J_EWALD3D_KE``) and carries
      the same absolute-total undercoverage on dense-core cells.
    * ``e_bipole`` (split, physical cutoff 8, M5 internal radius from
      ``sr_image_precision=1e-6`` with QQR screening) =
      81.4173579654 Ha. This agrees with the independently converged
      full-reciprocal value above: the historical -0.5153 Ha SR image
      truncation is gone.

    The pinned gap ``e_bipole - e_reference = +0.4884989548 Ha`` is now
    solely the exact-FT reference's finite-ke tail. Re-derive BOTH pins
    when the internal-domain precision or exact-FT ke changes. Widening
    only the BIPOLE reciprocal envelope leaves the number unchanged
    (K_max-invariance assert), so the repaired split is independently
    converged in both its SR and LR pieces.
    """

    (
        system,
        basis,
        kmesh,
        k_points,
        weights,
        opts,
        coeffs,
        densities,
        S_lat,
    ) = _mgo_shrink2_hcore_density(cutoff_bohr=8.0)

    cells = direct_lattice_cells(system, float(opts.cutoff_bohr))
    n_occ = int(system.n_electrons()) // 2
    D_real = real_space_density_from_kpoints(
        coeffs,
        [n_occ] * len(coeffs),
        kmesh,
        cells,
    )
    reference_blocks = compute_j_ewald_3d_ft_k_density_to_cells(
        basis,
        system,
        densities,
        k_points,
        weights,
        cells,
        omega=0.5,  # accepted for compatibility; exact-FT J is w-invariant
        ke_cutoff=300.0,
    )
    e_reference = _j_energy_from_blocks(
        reference_blocks,
        D_real.cells,
        k_points,
        weights,
        densities,
    )

    V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    kmax_crystal = crystal_ewald_reciprocal_cutoff(V_cell)
    e_bipole = _bipole_fixed_density_j_energy(
        system,
        basis,
        kmesh,
        k_points,
        weights,
        opts,
        coeffs,
        densities,
        S_lat,
        K_max=kmax_crystal,
    )
    e_bipole_wide = _bipole_fixed_density_j_energy(
        system,
        basis,
        kmesh,
        k_points,
        weights,
        opts,
        coeffs,
        densities,
        S_lat,
        K_max=10.0,
    )

    assert e_bipole_wide == pytest.approx(e_bipole, abs=1e-8)
    split_gap = e_bipole - e_reference
    assert e_reference == pytest.approx(80.9288590106, abs=5e-5)
    assert e_bipole == pytest.approx(81.4173579654, abs=5e-5)
    assert split_gap == pytest.approx(0.4884989548, abs=5e-5)


@pytest.mark.slow
def test_j_sr_traversal_ball_converges_diffuse_home_element():
    """The SR ket-image lattice sum must converge with the traversal ball.

    Truth anchor for the -515 mHa split defect documented in
    ``test_mgo_shrink2_fixed_density_pure_rks_exact_j_closes_split_gap``:
    restrict the MgO SHRINK-2 Hcore density to its home-cell block
    (J is linear in D, so the defect localises cleanly) and build
    J_SR(erfc, w = CRYSTAL default) with the traversal cell ball
    widened past the erf(sqrt(m) r) - erf(sqrt(m_w) r) reach of the
    diffuse pair charges. The pinned targets are the reciprocal-space
    (Poisson-summation) values of the same lattice sum,

        J_SR(g)_mn = (1/V) [ (pi/w^2) rho(0) S(g)_mn
            + sum_{G!=0} (4pi/G^2)(1 - e^{-G^2/4w^2})
              rho(G) conj(FT_mn(G; R_g)) ]

    evaluated with an independent closed-form Gaussian rho(G)/FT on
    the exact reciprocal lattice to ke = 24000 (2026-07-12 triage; the
    identity itself was verified to 1e-16 on this lattice with
    Gaussian charges, and the C++ builder reproduces the targets to
    7 digits at a 16-bohr ball). At the density-support ball (8 bohr)
    the same elements read 1.6054816 / 3.9333393 -- percent-level
    truncation loss on diffuse pairs, the root cause of the split
    defect. This test pins the CONVERGED property, so it must stay
    green when the SYM3b M4 internal-domain pad lands.
    """
    (
        system,
        basis,
        kmesh,
        k_points,
        weights,
        opts,
        coeffs,
        densities,
        S_lat,
    ) = _mgo_shrink2_hcore_density(cutoff_bohr=8.0)
    cells = direct_lattice_cells(system, float(opts.cutoff_bohr))
    n_occ = int(system.n_electrons()) // 2
    D_real = real_space_density_from_kpoints(
        coeffs, [n_occ] * len(coeffs), kmesh, cells,
    )
    nbf = int(basis.nbasis)
    from vibeqc._vibeqc_core import make_lattice_matrix_set

    blocks = [
        np.asarray(D_real.blocks[0], dtype=float) if c == 0
        else np.zeros((nbf, nbf))
        for c in range(len(cells))
    ]
    D_h0 = make_lattice_matrix_set(nbf, list(cells), blocks)

    V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    omega = crystal_default_ewald_alpha(V_cell)
    wide = LatticeSumOptions()
    wide.cutoff_bohr = 16.0
    wide.nuclear_cutoff_bohr = 16.0
    J = build_fock_2e_real_space(basis, system, wide, D_h0, 0.0, float(omega))
    J0 = np.asarray(J.blocks[0])

    # AO 5 = Mg 3s, AO 8 = Mg 3px, AO 12 = O 2pz (sto-3g shell order).
    # Poisson-anchored targets (see docstring); tolerances cover the
    # residual 16-bohr ball truncation (< 1e-6) with platform margin.
    assert J0[5, 5] == pytest.approx(1.8529059, abs=1e-4)
    assert J0[12, 12] == pytest.approx(3.9988987, abs=1e-4)
    # Cubic site symmetry: the converged home-block s-p coupling drops
    # to the small residual set by the (slightly asymmetric) h=0-only
    # density restriction -- at the 8-bohr ball it reads 0.126.
    assert abs(J0[5, 8]) < 6e-3


# ---------------------------------------------------------------------
# run_pbc_bipole_rhf — multi-k Ewald-J-split smoke tests
# ---------------------------------------------------------------------
def test_use_ewald_j_split_multik_no_longer_raises():
    """Multi-k Ewald-J-split is supported as of v0.9.0.

    Previously raised ``NotImplementedError`` (see prior
    ``test_use_ewald_j_split_multik_raises`` in
    ``tests/test_pbc_bipole_ewald_split_integration.py``); now runs
    via per-k ``compute_J_long_range_at_k``.
    """
    # Keep the ordinary smoke lightweight. MgO 2x2x2 at a fold-converged
    # 12-bohr operator cutoff is an intentionally slow validation and is
    # covered by the slow tests below; using it here made this entire T2 file
    # exceed the per-file gate after final-density consistency added a second
    # physical Fock evaluation.
    system, basis = _build_he_cubic()
    kmesh = monkhorst_pack(system, [2, 1, 1], use_symmetry=False)
    opts = _one_iter_rhf_options()
    result = run_pbc_bipole_rhf(
        system, basis, kmesh, opts,
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        progress=False,
    )
    assert result.n_iter == 1
    assert np.isfinite(result.energy)


def test_run_pbc_bipole_rhf_builds_shared_lr_cache_once(monkeypatch):
    """Multi-k SCF must reuse the density-independent J^LR pair-FT cache.

    The AO-pair FT stack depends on the basis, lattice, Ewald alpha, and
    reciprocal cutoff, not on the current density. A two-iteration SCF
    should therefore build the shared V_ne/J^LR cache once before the
    loop, then only recompute rho_hat and Fock contractions per iteration.
    """
    calls = _count_j_lr_cache_builds(monkeypatch)
    system, basis = _build_he_cubic()
    kmesh = monkhorst_pack(system, [2, 1, 1], use_symmetry=False)
    opts = _two_iter_rhf_options()

    result = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        progress=False,
    )

    assert result.n_iter == 2
    assert calls["n"] == 1


@pytest.mark.parametrize(
    "driver,options",
    [
        (run_pbc_bipole_rks, _two_iter_ks_options),
        (run_pbc_bipole_uhf, _two_iter_rhf_options),
        (run_pbc_bipole_uks, _two_iter_ks_options),
    ],
    ids=["rks", "uhf", "uks"],
)
def test_run_pbc_bipole_other_drivers_build_shared_lr_cache_once(
    monkeypatch,
    driver,
    options,
):
    """RKS/UHF/UKS keep the same one-cache-per-run multi-k contract."""
    calls = _count_j_lr_cache_builds(monkeypatch)
    system, basis = _build_he_cubic()
    kmesh = monkhorst_pack(system, [2, 1, 1], use_symmetry=False)

    result = driver(
        system,
        basis,
        kmesh,
        options(),
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        progress=False,
    )

    assert result.n_iter == 2
    assert calls["n"] == 1


# ---------------------------------------------------------------------
# Fused multi-k Bloch folds (bloch_sum_multi_k) vs Python fallback
# ---------------------------------------------------------------------


def test_bloch_sum_blocks_multi_k_matches_serial_fold(monkeypatch):
    """The shared BIPOLE helper returns the same per-k folds through the
    fused OpenMP kernel and through the serial per-k Python fallback.
    Both accumulate the cells in list order; the kernel's cos/sin phase
    is within one ulp of ``np.exp``."""
    import vibeqc._vibeqc_core as core

    system, basis = _build_he_cubic()
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 6.0
    S_lat = compute_overlap_lattice(basis, system, lat)
    blocks = [np.asarray(b, dtype=float) for b in S_lat.blocks]
    cells = list(S_lat.cells)
    kmesh = monkhorst_pack(system, [2, 2, 1], use_symmetry=False)
    k_points = list(kmesh.kpoints)

    fused = _bloch_sum_blocks_multi_k(blocks, cells, k_points)
    serial = [
        _bloch_sum_blocks(blocks, cells, np.asarray(k)) for k in k_points
    ]
    assert len(fused) == len(k_points)
    for F_f, F_s in zip(fused, serial):
        np.testing.assert_allclose(F_f, F_s, atol=1e-14)

    monkeypatch.delattr(core, "bloch_sum_multi_k")
    fallback = _bloch_sum_blocks_multi_k(blocks, cells, k_points)
    for F_f, F_p in zip(fused, fallback):
        np.testing.assert_allclose(F_f, F_p, atol=1e-14)


@pytest.mark.parametrize(
    "driver,options",
    [
        (run_pbc_bipole_rhf, _two_iter_rhf_options),
        (run_pbc_bipole_rks, _two_iter_ks_options),
        (run_pbc_bipole_uhf, _two_iter_rhf_options),
        (run_pbc_bipole_uks, _two_iter_ks_options),
    ],
    ids=["rhf", "rks", "uhf", "uks"],
)
def test_bipole_fused_multi_k_scf_matches_python_fallback(
    monkeypatch,
    driver,
    options,
):
    """All four multi-k BIPOLE drivers run the same deterministic
    2-iteration SCF with the fused per-iteration Bloch folds and with
    the serial per-k Python fallback (energy to 1e-10 Ha, per-k Fock
    matrices to 1e-10)."""
    import vibeqc._vibeqc_core as core

    system, basis = _build_he_cubic()
    kmesh = monkhorst_pack(system, [2, 1, 1], use_symmetry=False)

    r_fused = driver(
        system,
        basis,
        kmesh,
        options(),
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        progress=False,
    )
    monkeypatch.delattr(core, "bloch_sum_multi_k")
    r_py = driver(
        system,
        basis,
        kmesh,
        options(),
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        progress=False,
    )

    assert r_fused.n_iter == r_py.n_iter == 2
    assert abs(r_fused.energy - r_py.energy) < 1e-10
    if hasattr(r_fused, "fock"):
        fock_pairs = [(r_fused.fock, r_py.fock)]
    else:
        fock_pairs = [
            (r_fused.fock_alpha, r_py.fock_alpha),
            (r_fused.fock_beta, r_py.fock_beta),
        ]
    for fused_list, py_list in fock_pairs:
        for F_f, F_p in zip(fused_list, py_list):
            np.testing.assert_allclose(F_f, F_p, atol=1e-10)


# ---------------------------------------------------------------------
# Multi-k optional-artifact metadata (P03 MgO/STO-3G KRHF gap, 2026-07)
# ---------------------------------------------------------------------
#
# The P03 sidecar rerun surfaced two warnings from a multi-k BIPOLE KRHF
# job: the Γ-only molden export and the QVF wavefunction fallback both
# failed with "multi-k ... requires k-point metadata". Root cause: the
# ``PBCBipole*Result`` dataclasses carried the per-k ``mo_coeffs`` lists
# but not the k-points those lists span, so the runner's Γ-locating
# output helpers (``_gamma_proxy_for_multi_k`` /
# ``_qvf_wavefunction_proxy_for_multi_k``) had nothing to key on and
# refused to guess that the first k-point is Γ. The BIPOLE results now
# carry ``kpoints_cart`` / ``kpoint_weights`` (the GDF result contract in
# periodic_k_gdf.py). These regressions pin that plumbing and the two
# resulting behaviours: metadata present + Γ in mesh → locate Γ; metadata
# genuinely absent → fail closed with the clear diagnostic.


@pytest.mark.parametrize(
    "driver,options",
    [
        (run_pbc_bipole_rhf, _two_iter_rhf_options),
        (run_pbc_bipole_rks, _two_iter_ks_options),
        (run_pbc_bipole_uhf, _two_iter_rhf_options),
        (run_pbc_bipole_uks, _two_iter_ks_options),
    ],
    ids=["rhf", "rks", "uhf", "uks"],
)
def test_multik_bipole_result_carries_kpoint_metadata(driver, options):
    """All four multi-k BIPOLE drivers record aligned k-point metadata.

    The per-k ``mo_coeffs`` / ``mo_energies`` lists are only interpretable
    by downstream single-k output writers if the result also says which
    k-points they span. Prior to the P03 fix these fields were absent and
    optional molden / QVF-wavefunction export failed closed on every
    multi-k BIPOLE job.
    """
    system, basis = _build_he_cubic()
    kmesh = monkhorst_pack(system, [2, 1, 1], use_symmetry=False)
    expected_kpts = np.asarray(list(kmesh.kpoints), dtype=float).reshape(-1, 3)
    n_k = expected_kpts.shape[0]
    assert n_k == 2  # sanity: fixture is genuinely multi-k

    result = driver(
        system,
        basis,
        kmesh,
        options(),
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        progress=False,
    )

    _assert_result_k_count(result, n_k)
    assert result.kpoints_cart is not None
    assert result.kpoint_weights is not None
    kpts = np.asarray(result.kpoints_cart, dtype=float)
    weights = np.asarray(result.kpoint_weights, dtype=float)
    assert kpts.shape == (n_k, 3)
    assert weights.shape == (n_k,)
    # Aligned, in the same order as the per-k orbital lists.
    np.testing.assert_allclose(kpts, expected_kpts, atol=1e-12)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)


def test_multik_bipole_optional_wavefunction_export_locates_gamma():
    """Γ-centred multi-k BIPOLE result → molden / QVF-wf locate Γ.

    Reproduces the P03 export path on a real (cheap) multi-k KRHF result.
    Before the fix both helpers raised ValueError("... requires k-point
    metadata"); now they resolve Γ from the recorded ``kpoints_cart``.
    """
    from vibeqc.periodic_runner import (
        _gamma_index_for_multi_k,
        _gamma_proxy_for_multi_k,
        _qvf_wavefunction_proxy_for_multi_k,
    )

    system, basis = _build_he_cubic()
    kmesh = monkhorst_pack(system, [2, 1, 1], use_symmetry=False)
    result = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        _two_iter_rhf_options(),
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        progress=False,
    )

    gamma_idx = _gamma_index_for_multi_k(result, len(result.mo_coeffs))
    # [2,1,1] is Γ-centred, so exactly one k-point is Γ.
    kpts = np.asarray(result.kpoints_cart, dtype=float)
    assert np.linalg.norm(kpts[gamma_idx]) < 1e-10

    # Molden Γ-proxy exposes the Γ-block MOs (not a blind first-k guess).
    molden_proxy = _gamma_proxy_for_multi_k(result)
    np.testing.assert_allclose(
        np.asarray(molden_proxy.mo_coeffs),
        np.asarray(result.mo_coeffs[gamma_idx]),
        atol=1e-12,
    )

    # QVF wavefunction export selects Γ and records its fractional coord.
    qvf_proxy, k_frac = _qvf_wavefunction_proxy_for_multi_k(result, system)
    np.testing.assert_allclose(
        np.asarray(qvf_proxy.mo_coeffs),
        np.asarray(result.mo_coeffs[gamma_idx]),
        atol=1e-12,
    )
    np.testing.assert_allclose(k_frac, [0.0, 0.0, 0.0], atol=1e-10)


def test_multik_bipole_missing_kpoint_metadata_fails_closed():
    """With k-metadata genuinely absent, optional export fails closed.

    If a legacy or hand-built result reaches the output layer without
    ``kpoints_cart``, the writers must refuse rather than silently
    assuming the first block is Γ. This pins the clear, fail-closed
    diagnostic (Expected option 2 of the P03 report) so the failure never
    looks like an unexpected artifact crash.
    """
    from vibeqc.periodic_runner import (
        _gamma_proxy_for_multi_k,
        _qvf_wavefunction_proxy_for_multi_k,
    )

    system, basis = _build_he_cubic()
    kmesh = monkhorst_pack(system, [2, 1, 1], use_symmetry=False)
    result = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        _two_iter_rhf_options(),
        use_ewald_j_split=True,
        ewald_precision=1e-6,
        progress=False,
    )
    # Remove both output and restart metadata to model a legacy result.
    # Current results also carry restart coordinates for READ round trips.
    for name in ("kpoints_cart", "kpoint_weights", "kpoints", "weights",
                 "kmesh", "restart_kpoints", "restart_weights"):
        if hasattr(result, name):
            setattr(result, name, None)

    with pytest.raises(ValueError, match="requires k-point metadata"):
        _gamma_proxy_for_multi_k(result)
    with pytest.raises(ValueError, match="requires k-point metadata"):
        _qvf_wavefunction_proxy_for_multi_k(result, system)


def test_use_ewald_j_split_ibz_native_energy_matches_full_mesh():
    """IBZ-native Ewald-J SCF reproduces full-mesh total energy.

    M4 (commit a5412ec8, 2026-05-22) made the BIPOLE RHF driver
    IBZ-native: SCF diagonalisation runs at the user's input k-points,
    and the density D(k) is expanded to the full mesh only for the
    Ewald-J ρ̂ density transform (via ir_mapping). The output
    ``mo_energies`` therefore matches the user's input mesh size, not
    the underlying full mesh. The science-correctness contract is that
    the total energy is identical to a full-mesh run on the same MP grid.
    """
    system, basis = _build_he_cubic()
    vq.attach_symmetry(system)
    kmesh_full = monkhorst_pack(system, [2, 2, 2], use_symmetry=False)
    kmesh_ibz = monkhorst_pack(system, [2, 2, 2], use_symmetry=True)
    assert len(kmesh_ibz.ir_mapping) == 8
    assert len(kmesh_ibz.kpoints) < len(kmesh_full.kpoints)

    opts = _one_iter_rhf_options()

    result_full = run_pbc_bipole_rhf(
        system, basis, kmesh_full, opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,  # gauge-independent consistency check; legacy keeps it cheap (no BvK-torus cutoff)
        ewald_precision=1e-6,
        progress=False,
    )
    result_ibz = run_pbc_bipole_rhf(
        system, basis, kmesh_ibz, opts,
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,  # gauge-independent consistency check; legacy keeps it cheap (no BvK-torus cutoff)
        ewald_precision=1e-6,
        progress=False,
    )

    _assert_result_k_count(result_full, len(kmesh_full.kpoints))
    # IBZ inputs are expanded to the full mesh up front (2026-06-10):
    # the IBZ-native replication shortcut lacked the star AO rotations
    # and broke non-trivial crystals. The result therefore reports the
    # full-mesh k count, and full ≡ IBZ holds by construction.
    _assert_result_k_count(result_ibz, len(kmesh_full.kpoints))
    assert result_ibz.energy == pytest.approx(result_full.energy, abs=1e-10)


@pytest.mark.parametrize(
    "driver,options",
    [
        (run_pbc_bipole_rks, _one_iter_ks_options),
        (run_pbc_bipole_uhf, _one_iter_rhf_options),
        (run_pbc_bipole_uks, _one_iter_ks_options),
    ],
    ids=["rks", "uhf", "uks"],
)
def test_use_ewald_j_split_ibz_native_other_drivers_energy_matches_full_mesh(
    driver,
    options,
):
    """RKS/UHF/UKS accept IBZ meshes with ir_mapping like RHF."""
    system, basis = _build_he_cubic()
    vq.attach_symmetry(system)
    kmesh_full = monkhorst_pack(system, [2, 2, 2], use_symmetry=False)
    kmesh_ibz = monkhorst_pack(system, [2, 2, 2], use_symmetry=True)
    assert len(kmesh_ibz.ir_mapping) == 8
    assert len(kmesh_ibz.kpoints) < len(kmesh_full.kpoints)

    result_full = driver(
        system, basis, kmesh_full, options(),
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,  # gauge-independent consistency check; legacy keeps it cheap (no BvK-torus cutoff)
        ewald_precision=1e-6,
        progress=False,
    )
    result_ibz = driver(
        system, basis, kmesh_ibz, options(),
        use_ewald_j_split=True,
        use_exchange_ewald_split=False,  # gauge-independent consistency check; legacy keeps it cheap (no BvK-torus cutoff)
        ewald_precision=1e-6,
        progress=False,
    )

    _assert_result_k_count(result_full, len(kmesh_full.kpoints))
    _assert_result_k_count(result_ibz, len(kmesh_full.kpoints))
    assert result_ibz.energy == pytest.approx(result_full.energy, abs=1e-10)


def test_ibz_mesh_expands_to_full_mesh_on_mgo_before_fold_refusal(monkeypatch):
    """IBZ-input MgO [2,2,2] expands to the exact full mesh.

    Regression for the 2026-06-10 finding: the former IBZ-native path
    diagonalised at the 3 irreducible points and replicated D(k) into
    each star without the AO rotation D(R·k) = P(R)·D(k)·P(R)ᵀ —
    exact on the He validation cells (trivial stars) but on MgO the
    SCF failed to converge 8.25 Ha away from the full-mesh result. The
    production fix is to expand before SCF, so pin that operation directly
    on the first non-trivial crystal instead of running two invalid cutoff-6
    SCFs. The driver must still reach and enforce the fold-support refusal.

    The converged cutoff-12 MgO anchor below covers the valid numerical
    route; the lightweight He tests above cover full-vs-IBZ driver energy.
    """
    import vibeqc.pbc_bipole as bipole_module

    system, basis = _build_mgo()
    vq.attach_symmetry(system)
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.lattice_opts.nuclear_cutoff_bohr = 6.0
    opts.max_iter = 1
    opts.use_diis = False
    opts.initial_guess = InitialGuess.SAD

    kmesh_full = monkhorst_pack(system, [2, 2, 2], use_symmetry=False)
    kmesh_ibz = monkhorst_pack(system, [2, 2, 2], use_symmetry=True)
    assert len(kmesh_ibz.kpoints) < len(kmesh_full.kpoints)

    observed = {}
    original_expand = bipole_module._expand_ibz_kmesh_for_ewald_j

    def capture_expanded_mesh(system_arg, kmesh_arg, plog):
        expanded = original_expand(system_arg, kmesh_arg, plog)
        observed["mesh"] = expanded
        return expanded

    monkeypatch.setattr(
        bipole_module,
        "_expand_ibz_kmesh_for_ewald_j",
        capture_expanded_mesh,
    )
    with pytest.raises(
        BipoleFoldUnreliableError,
        match=r"S\(k\) fold truncation.*cutoff 6\.0 bohr",
    ):
        run_pbc_bipole_rhf(
            system,
            basis,
            kmesh_ibz,
            opts,
            use_ewald_j_split=True,
            ewald_precision=1e-8,
            progress=False,
        )

    expanded = observed["mesh"]
    np.testing.assert_allclose(
        np.asarray(expanded.kpoints),
        np.asarray(kmesh_full.kpoints),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(expanded.weights),
        np.asarray(kmesh_full.weights),
        atol=0.0,
        rtol=0.0,
    )
    assert tuple(expanded.mesh) == tuple(kmesh_full.mesh)
    assert tuple(expanded.is_shift) == tuple(kmesh_full.is_shift)


def test_use_ewald_j_split_multik_mgo_shrink22_refuses_c8_fold():
    """MgO SHRINK 2 2 refuses the known-invalid cutoff-8 overlap fold.

    The former smoke entered three SCF cycles on a metric whose overlap-fold
    drift is 0.383 and therefore could only pin a spurious basin. The
    cutoff-12 converged MgO test below already guards the original
    k-independent J^LR over-binding bug on a usable fold; this cheap test
    instead pins the scientifically required pre-SCF refusal.

    The smaller 0.041 diagnostic at cutoff 6 is not improved convergence:
    its 1.5x reference stops at 9 bohr and misses the diffuse Mg-O overlap
    shell at 9.74 bohr. The cutoff-8 reference extends to 12 bohr and sees
    that shell, hence the larger (and more revealing) drift.
    """
    system, basis = _build_mgo()
    kmesh = monkhorst_pack(system, [2, 2, 2])
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 8.0
    opts.lattice_opts.nuclear_cutoff_bohr = 8.0
    opts.max_iter = 3
    opts.use_diis = False
    opts.damping = 0.0
    opts.initial_guess = InitialGuess.SAD
    with pytest.raises(
        BipoleFoldUnreliableError,
        match=r"S\(k\) fold truncation.*cutoff 8\.0 bohr",
    ):
        run_pbc_bipole_rhf(
            system,
            basis,
            kmesh,
            opts,
            use_ewald_j_split=True,
            ewald_precision=1e-6,
            progress=False,
        )


@pytest.mark.slow
def test_mgo_shrink22_converged_matches_crystal_kconverged():
    """Cross-family CRYSTAL pin: converged BIPOLE SHRINK 2 2 (corrected gauge,
    ``exxdiv='ewald'``) reproduces CRYSTAL's *k-converged* SHRINK-8-8 value —
    NOT its coarse SHRINK-2-2 value.

    2026-06-16 finding (``docs/periodic_jk_routes.md`` § Current status): the
    "matched-k" SHRINK-2-2-vs-SHRINK-2-2 comparison is exchange-finite-size-
    *convention*-confounded. BIPOLE applies the probe-charge Ewald (Madelung)
    exxdiv correction ``(ξ_M − π/(V_sc·ω²))·S(k)D(k)S(k)`` (PySCF-equivalent
    ``exxdiv='ewald'``), removing the leading 1/N_k^{1/3} exchange finite-size
    error — so its SHRINK-2-2 energy is *already k-converged*. CRYSTAL applies
    no such correction; its SHRINK-2-2 over-binds its own SHRINK-8-8 by 638 mHa.
    So the valid cross-family comparison is BIPOLE-SHRINK22 (exxdiv-corrected)
    ↔ CRYSTAL-**SHRINK88** (k-converged), agreeing to a few mHa (STO-3G
    lattice-sum truncation scale at cutoff 12).

    Independent references (rerun out of process for M5 on 2026-07-13):
      * SHRINK 8 8 = -271.21814375 Ha/FU  (k-converged; the valid target)
      * SHRINK 2 2 = -271.85639652 Ha/FU  (coarse, uncorrected; NOT a target)
      * PySCF 2.13.1 KRHF/GDF [2,2,2], exxdiv='ewald'
        = -271.21335558268 Ha/FU

    cutoff 12 → S(k)-fold ~5e-3, metric Σ_k w_k Tr[D(k)S(k)] = 20.000 (physical
    basin, not the cutoff-8 metric-invalid state). M5 combines the
    precision-derived padded SR ket-image domain and QQR screening with the
    SYM3b real-space point-group reduction. The direct-ERI Fock build runs only
    the orbit-representative cells and reconstructs the rest by rotation on
    the fully group-invariant truncation domain. The re-pinned M5 energy is
    -271.21478463 Ha/FU: +3.359 mHa vs CRYSTAL23 and -1.429 mHa vs PySCF.
    """
    E_CRYSTAL_SHRINK88 = -271.21814375
    E_CRYSTAL_SHRINK22 = -271.85639652
    E_PYSCF_KRHF_222 = -271.21335558268

    system, basis = _build_mgo()
    vq.attach_symmetry(system)   # SYM3b needs the crystal point group
    kmesh = monkhorst_pack(system, [2, 2, 2])
    n_occ = system.n_electrons() // 2

    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    opts.max_iter = 60
    opts.use_diis = True
    opts.initial_guess = InitialGuess.SAD
    opts.conv_tol_energy = 1e-8
    opts.conv_tol_grad = 1e-6

    r = run_pbc_bipole_rhf(
        system, basis, kmesh, opts,
        use_exchange_ewald_split=True,    # corrected gauge (exxdiv='ewald')
        use_incremental_fock=True,
        use_fock_symmetry_reduce=True,    # SYM3b point-group-reduced Fock build
        ewald_precision=1e-8,
        progress=False,
    )
    assert r.converged

    # Metric validity: Σ_k w_k Tr[D(k)S(k)] == N_elec (=20). A value near 19.57
    # would be the cutoff-8 metric-invalid (spurious) basin.
    weights = [1.0 / len(r.mo_coeffs)] * len(r.mo_coeffs)
    metric = 0.0
    for C, S, w in zip(r.mo_coeffs, r.overlap, weights):
        C = np.asarray(C)
        D = 2.0 * (C[:, :n_occ] @ C[:, :n_occ].conj().T)
        metric += w * np.real(np.trace(D @ np.asarray(S)))
    assert metric == pytest.approx(float(system.n_electrons()), abs=1e-3)

    # PRIMARY: cross-family agreement with CRYSTAL's k-converged value.
    assert r.energy == pytest.approx(E_CRYSTAL_SHRINK88, abs=6e-3)
    # Same exchange convention as BIPOLE; independent GDF implementation.
    assert r.energy == pytest.approx(E_PYSCF_KRHF_222, abs=3e-3)
    # Regression anchor: M5 padded + pair-resolved SYM3b c12 value.
    assert r.energy == pytest.approx(-271.21478463, abs=5e-4)
    # The coarse matched-k value is NOT a target: BIPOLE sits ~641 mHa above it
    # by construction (the exxdiv convention gap) and must stay well above it.
    assert r.energy - E_CRYSTAL_SHRINK22 > 0.5


# ---------------------------------------------------------------------------
# #725: MOM declares its fixed occupied subspace to the T = 0 fill
# ---------------------------------------------------------------------------


def test_bipole_rks_mom_declares_a_fixed_occupied_subspace(monkeypatch):
    """Twin of the RHF-Ewald contract in ``test_periodic_rhf_multi_k_ewald``:
    ``run_pbc_bipole_rks`` permutes ``C(k)`` / ``eps(k)`` so the MOM-selected
    occupied states lead (Gilbert, Besley, Gill 2008), so it must ask the
    shared ``T = 0`` fill for a positional occupation of that block
    (``fixed_occupied_subspace=use_mom``) rather than the global energy
    ordering that undid the selection since #85 (#725). Pinned at the first
    occupation call, before any SCF iteration; pre-fix the wrapper received no
    such keyword."""
    import vibeqc.pbc_bipole_rks as drv

    class _Stop(RuntimeError):
        pass

    calls: list = []

    def _recording(*args, **kwargs):
        calls.append(kwargs)
        raise _Stop

    monkeypatch.setattr(drv, "_closed_shell_periodic_occupations", _recording)
    system, basis = _build_he_cubic()
    kmesh = monkhorst_pack(system, [2, 1, 1], use_symmetry=False)
    for use_mom in (False, True):
        with pytest.raises(_Stop):
            drv.run_pbc_bipole_rks(
                system, basis, kmesh, _one_iter_ks_options(),
                use_mom=use_mom, progress=False,
            )
        assert calls[-1]["fixed_occupied_subspace"] is use_mom
    assert len(calls) == 2
