"""Density-fitted analytic gradient kernels — finite-difference validation.

The DF Coulomb-energy gradient at converged density D is

  ∂E_J/∂R_{A,c} = Σ_P γ_P ∂ρ_P/∂R_{A,c} − (1/2) Σ_{PQ} γ_P γ_Q ∂V_{PQ}/∂R_{A,c}

with γ = V^{-1} ρ, ρ_P = Σ_{μν} D_{μν}(μν|P), V_{PQ} = (P|Q) (Weigend,
PCCP 4, 4285 (2002)). For pure DFT (α_HF = 0) this is the *complete*
DF analytic two-electron gradient. For HF / hybrid DFT, pair with the
DF-K gradient (commit 4b).

These tests pin the new C++ kernels at the gradient level by comparing
against finite differences of the closed-form DF Coulomb energy
E_J = (1/2) ρ^T V^{-1} ρ at fixed density. The reference is independent
of any SCF — it's a direct check of the libint deriv-1 contractions
the kernels expose.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    RHFOptions,
    compute_2c_eri,
    compute_3c_eri,
    compute_df_j_gradient,
    run_rhf,
)

from .conftest import GEOMETRIES


def _make(positions, atom_Zs, orbital_basis_name, aux_basis_name):
    mol = Molecule([Atom(int(Z), list(p)) for Z, p in zip(atom_Zs, positions)])
    basis = BasisSet(mol, orbital_basis_name)
    aux = BasisSet(mol, aux_basis_name)
    return mol, basis, aux


def _e_j_df(D, positions, atom_Zs, orbital_basis_name, aux_basis_name):
    """Closed-form DF Coulomb energy at the given geometry, with the
    density `D` *held fixed* in its (numerical) AO-index form.

    AO basis functions move with their atoms; D doesn't change shape
    because the basis dimension is the same. This is exactly the
    fixed-D / 2n+1-rule construction the analytic gradient assumes.
    """
    _, basis, aux = _make(
        positions, atom_Zs, orbital_basis_name, aux_basis_name,
    )
    V = compute_2c_eri(aux)
    T = compute_3c_eri(basis, aux)
    rho = np.einsum("Pmn,mn->P", T, D)
    gamma = np.linalg.solve(V, rho)
    return 0.5 * rho @ gamma


def _fd_gradient(D, atom_Zs, ref_positions,
                 orbital_basis_name, aux_basis_name, *, h=1e-4):
    n_atoms = len(ref_positions)
    grad = np.zeros((n_atoms, 3))
    for A in range(n_atoms):
        for c in range(3):
            pos_p = [list(p) for p in ref_positions]
            pos_p[A][c] += h
            pos_m = [list(p) for p in ref_positions]
            pos_m[A][c] -= h
            ep = _e_j_df(D, pos_p, atom_Zs, orbital_basis_name, aux_basis_name)
            em = _e_j_df(D, pos_m, atom_Zs, orbital_basis_name, aux_basis_name)
            grad[A, c] = (ep - em) / (2 * h)
    return grad


# ---------------------------------------------------------------------
# FD-validation cases.
# ---------------------------------------------------------------------

DF_GRADIENT_CASES = [
    ("H2",  "def2-svp", "def2-svp-jk"),
    ("H2O", "def2-svp", "def2-svp-jk"),
    ("H2O", "cc-pvdz",  "cc-pvdz-jkfit"),
    ("CH4", "def2-svp", "def2-svp-jk"),
]


@pytest.mark.parametrize(
    "mol_key,orb,aux", DF_GRADIENT_CASES,
    ids=[f"{m}-{o}-{a}" for m, o, a in DF_GRADIENT_CASES],
)
def test_df_j_gradient_matches_fd(mol_key, orb, aux):
    """Analytic DF-J gradient agrees with finite difference of E_J at
    fixed converged density. Truncation error of central FD with h=1e-4
    is ~1e-8 Ha/bohr; tolerance pinned at 1e-6 to leave slack."""
    atoms = GEOMETRIES[mol_key]
    Zs = [Z for Z, _ in atoms]
    ref_positions = [list(xyz) for _, xyz in atoms]

    mol, basis, aux_basis = _make(ref_positions, Zs, orb, aux)

    # Get a physical density via converged RHF.
    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    rhf_opts.conv_tol_grad = 1e-10
    rhf = run_rhf(mol, basis, rhf_opts)
    assert rhf.converged
    D = np.array(rhf.density)

    grad_analytic = compute_df_j_gradient(basis, aux_basis, mol, D)
    grad_fd = _fd_gradient(D, Zs, ref_positions, orb, aux)

    np.testing.assert_allclose(
        grad_analytic, grad_fd, atol=1e-6,
        err_msg=f"{mol_key}/{orb}/{aux}: DF-J analytic vs FD mismatch",
    )


@pytest.mark.parametrize(
    "mol_key,orb,aux", DF_GRADIENT_CASES,
    ids=[f"{m}-{o}-{a}" for m, o, a in DF_GRADIENT_CASES],
)
def test_df_j_gradient_translational_invariance(mol_key, orb, aux):
    """Σ_A ∂E/∂R_A = 0 — the energy is translation-invariant. Holds to
    machine precision because both the 2c and 3c gradient kernels
    distribute their per-shell-pair derivative blocks across all
    contributing centres without losing the sum-to-zero structure."""
    atoms = GEOMETRIES[mol_key]
    Zs = [Z for Z, _ in atoms]
    ref_positions = [list(xyz) for _, xyz in atoms]
    mol, basis, aux_basis = _make(ref_positions, Zs, orb, aux)

    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    rhf_opts.conv_tol_grad = 1e-10
    rhf = run_rhf(mol, basis, rhf_opts)
    D = np.array(rhf.density)

    grad = compute_df_j_gradient(basis, aux_basis, mol, D)
    sum_over_atoms = grad.sum(axis=0)
    np.testing.assert_allclose(sum_over_atoms, 0.0, atol=1e-10)


def test_df_2c_gradient_kernel_translational_invariance():
    """Standalone test of the 2c kernel: with γ uniform over P, the
    gradient sum-over-atoms equals zero. (Strictly: sum over atoms of
    Σ_PQ γ_P γ_Q ∂V_PQ/∂R = 0 because ∂(P|Q)/∂R + ∂(P|Q)/∂R' = 0 when
    P, Q ride on different atoms — and the diagonal P=Q on the same
    atom contributes zero by translational invariance of the
    self-overlap.)"""
    from vibeqc import compute_2c_eri_gradient_contribution
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    aux = BasisSet(mol, "def2-svp-jk")
    rng = np.random.default_rng(42)
    gamma = rng.standard_normal(aux.nbasis)

    grad = compute_2c_eri_gradient_contribution(aux, mol, gamma)
    sum_over_atoms = grad.sum(axis=0)
    np.testing.assert_allclose(sum_over_atoms, 0.0, atol=1e-10)


def test_df_3c_gradient_kernel_translational_invariance():
    """Standalone test of the 3c kernel: with arbitrary D and γ,
    the gradient sum-over-atoms equals zero."""
    from vibeqc import compute_3c_eri_gradient_contribution
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "def2-svp")
    aux = BasisSet(mol, "def2-svp-jk")
    rng = np.random.default_rng(42)
    A = rng.standard_normal((basis.nbasis, basis.nbasis))
    D = 0.5 * (A + A.T)
    gamma = rng.standard_normal(aux.nbasis)

    grad = compute_3c_eri_gradient_contribution(basis, aux, mol, D, gamma)
    sum_over_atoms = grad.sum(axis=0)
    np.testing.assert_allclose(sum_over_atoms, 0.0, atol=1e-10)


def test_df_j_gradient_zero_at_zero_density():
    """Defensive: at D = 0 the J gradient is zero (no electrons → no
    Coulomb)."""
    atoms = GEOMETRIES["H2"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "def2-svp")
    aux = BasisSet(mol, "def2-svp-jk")
    D = np.zeros((basis.nbasis, basis.nbasis))
    grad = compute_df_j_gradient(basis, aux, mol, D)
    np.testing.assert_allclose(grad, 0.0, atol=1e-12)


# ---------------------------------------------------------------------
# K-gradient and combined JK-gradient (HF + hybrid DFT).
# ---------------------------------------------------------------------

def _e_k_df(C_occ, alpha_hf, positions, atom_Zs,
            orbital_basis_name, aux_basis_name):
    """Closed-form DF Exchange energy at fixed (C_occ, α_HF):

        E_K = -α_HF Σ_PQ V^{-1}_PQ (M^P : M^Q)

    with M^P_{ij} = C_occ^T T^P C_occ.
    """
    _, basis, aux = _make(
        positions, atom_Zs, orbital_basis_name, aux_basis_name,
    )
    V = compute_2c_eri(aux)
    T = compute_3c_eri(basis, aux)
    M = np.einsum("mi,Pmn,nj->Pij", C_occ, T, C_occ)
    M_flat = M.reshape(M.shape[0], -1)
    eta_flat = np.linalg.solve(V, M_flat)
    return -alpha_hf * float((M_flat * eta_flat).sum())


def _fd_gradient_with_e_fn(e_fn, positions, h=1e-4):
    n_atoms = len(positions)
    grad = np.zeros((n_atoms, 3))
    for A in range(n_atoms):
        for c in range(3):
            pp = [list(p) for p in positions]; pp[A][c] += h
            pm = [list(p) for p in positions]; pm[A][c] -= h
            grad[A, c] = (e_fn(pp) - e_fn(pm)) / (2 * h)
    return grad


@pytest.mark.parametrize(
    "mol_key,orb,aux", DF_GRADIENT_CASES,
    ids=[f"{m}-{o}-{a}" for m, o, a in DF_GRADIENT_CASES],
)
def test_df_k_gradient_matches_fd(mol_key, orb, aux):
    """DF-K gradient (α_HF = 1) at fixed C_occ matches FD of the
    closed-form -Σ_PQ V^{-1}_PQ (M^P : M^Q) energy."""
    from vibeqc import compute_df_k_gradient
    atoms = GEOMETRIES[mol_key]
    Zs = [Z for Z, _ in atoms]
    ref = [list(xyz) for _, xyz in atoms]
    mol, basis, aux_basis = _make(ref, Zs, orb, aux)

    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    rhf_opts.conv_tol_grad = 1e-10
    rhf = run_rhf(mol, basis, rhf_opts)
    n_occ = mol.n_electrons() // 2
    C_occ = np.array(rhf.mo_coeffs)[:, :n_occ]

    g_an = compute_df_k_gradient(basis, aux_basis, mol, C_occ, 1.0)
    g_fd = _fd_gradient_with_e_fn(
        lambda p: _e_k_df(C_occ, 1.0, p, Zs, orb, aux), ref,
    )
    np.testing.assert_allclose(g_an, g_fd, atol=1e-6)


@pytest.mark.parametrize("alpha_hf", [1.0, 0.2, 0.5])
def test_df_jk_gradient_matches_fd_hybrid(alpha_hf):
    """Combined JK gradient at fixed (D, C_occ) matches FD of
    E_J + α_HF · E_K. Covers HF (α=1) and hybrid-DFT (α=0.2 ≈ B3LYP,
    α=0.5 ≈ BHandH) regimes."""
    from vibeqc import compute_df_jk_gradient
    atoms = GEOMETRIES["H2O"]
    Zs = [Z for Z, _ in atoms]
    ref = [list(xyz) for _, xyz in atoms]
    mol, basis, aux_basis = _make(ref, Zs, "def2-svp", "def2-svp-jk")

    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    rhf_opts.conv_tol_grad = 1e-10
    rhf = run_rhf(mol, basis, rhf_opts)
    D = np.array(rhf.density)
    C_occ = np.array(rhf.mo_coeffs)[:, :5]

    g_an = compute_df_jk_gradient(basis, aux_basis, mol, D, C_occ, alpha_hf)

    def e_total(p):
        ej = _e_j_df(D, p, Zs, "def2-svp", "def2-svp-jk")
        ek = _e_k_df(C_occ, alpha_hf, p, Zs, "def2-svp", "def2-svp-jk")
        return ej + ek

    g_fd = _fd_gradient_with_e_fn(e_total, ref)
    np.testing.assert_allclose(g_an, g_fd, atol=1e-6)


def test_df_jk_gradient_pure_dft_reduces_to_j():
    """JK gradient with α_HF = 0 must equal the J-only gradient
    (machine precision)."""
    from vibeqc import compute_df_jk_gradient
    atoms = GEOMETRIES["H2O"]
    Zs = [Z for Z, _ in atoms]
    ref = [list(xyz) for _, xyz in atoms]
    mol, basis, aux_basis = _make(ref, Zs, "def2-svp", "def2-svp-jk")

    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    rhf_opts.conv_tol_grad = 1e-10
    rhf = run_rhf(mol, basis, rhf_opts)
    D = np.array(rhf.density)
    C_occ = np.array(rhf.mo_coeffs)[:, :5]

    g_jk = compute_df_jk_gradient(basis, aux_basis, mol, D, C_occ, 0.0)
    g_j  = compute_df_j_gradient(basis, aux_basis, mol, D)
    np.testing.assert_allclose(g_jk, g_j, atol=1e-12)


def test_df_k_gradient_translational_invariance():
    """Σ_A ∂E_K/∂R_A = 0 — DF-K gradient is translation-invariant at
    machine precision, mirroring the J case."""
    from vibeqc import compute_df_k_gradient
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "def2-svp")
    aux = BasisSet(mol, "def2-svp-jk")
    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    rhf_opts.conv_tol_grad = 1e-10
    rhf = run_rhf(mol, basis, rhf_opts)
    C_occ = np.array(rhf.mo_coeffs)[:, :5]
    grad = compute_df_k_gradient(basis, aux, mol, C_occ, 1.0)
    np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=1e-10)


def test_df_k_gradient_zero_at_pure_dft():
    """K-gradient with α_HF = 0 returns zero (no exchange)."""
    from vibeqc import compute_df_k_gradient
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "def2-svp")
    aux = BasisSet(mol, "def2-svp-jk")
    rhf_opts = RHFOptions()
    rhf_opts.conv_tol_energy = 1e-12
    rhf = run_rhf(mol, basis, rhf_opts)
    C_occ = np.array(rhf.mo_coeffs)[:, :5]
    grad = compute_df_k_gradient(basis, aux, mol, C_occ, 0.0)
    np.testing.assert_allclose(grad, 0.0, atol=1e-12)
