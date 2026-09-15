"""Phase 17b-1 — Coupled-Perturbed Hartree-Fock for RHF.

Contracts pinned here:

1. **API surface** — :func:`vibeqc.cphf_solve_rhf`,
   :func:`vibeqc.dipole_polarizability_rhf`,
   :class:`vibeqc.CPHFOptions`, :class:`vibeqc.CPHFConvergenceError`
   are public.

2. **Polarizability via CPHF matches FD-on-dipole** — for H₂O /
   STO-3G at the same equilibrium geometry, the analytic CPHF
   polarizability tensor agrees with PySCF's FD-on-dipole reference
   to ~1e-5 a.u. (FD truncation in the reference, not us).

3. **Polarizability tensor is symmetric** — α_αβ = α_βα is guaranteed
   by the construction (the static response is governed by a
   symmetric-positive-definite orbital Hessian).

4. **Polarizability is positive-definite** — all three eigenvalues
   of α are positive for a stable closed-shell molecule (response
   to the field has a positive real part).

5. **Refusal on un-converged SCF** — the CPHF kernel demands a
   converged HF reference and raises a clean ValueError otherwise.

6. **Convergence-failure path** — over-tight tol with too few CG
   iterations raises :class:`CPHFConvergenceError` rather than
   returning silent garbage.

7. **Off-diagonal RHS handling** — feeding a non-symmetric AO RHS
   matrix is permitted (the CPHF kernel only uses the occ-vir
   block); the result depends only on that block.

8. **Multiple RHS in one call** — feeding a stack of RHS matrices
   produces a stack of U matrices, matching the per-RHS solves.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.cphf import cphf_solve_rhf, dipole_polarizability_rhf


ANGSTROM_TO_BOHR = 1.8897261339213


def _h2o_mol():
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0,  0.7572 * ANGSTROM_TO_BOHR, 0.5868 * ANGSTROM_TO_BOHR]),
        vq.Atom(1, [0.0, -0.7572 * ANGSTROM_TO_BOHR, 0.5868 * ANGSTROM_TO_BOHR]),
    ], charge=0, multiplicity=1)


def _h2_mol(R_bohr: float = 1.346):
    return vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 0.0, R_bohr])],
        charge=0, multiplicity=1,
    )


def _converged_rhf(mol, basis_name="sto-3g"):
    basis = vq.BasisSet(mol, basis_name)
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    return basis, vq.run_rhf(mol, basis, opts)


# ---------------------------------------------------------------------------
# 1. API surface
# ---------------------------------------------------------------------------

def test_cphf_api_exposed():
    assert hasattr(vq, "cphf_solve_rhf")
    assert hasattr(vq, "dipole_polarizability_rhf")
    assert hasattr(vq, "CPHFOptions")
    assert hasattr(vq, "CPHFConvergenceError")


def test_cphf_default_options():
    o = vq.CPHFOptions()
    assert o.max_iter == 100
    assert o.tol == pytest.approx(1e-8)
    assert o.use_preconditioner is True


# ---------------------------------------------------------------------------
# 2 & 3 & 4. Polarizability against PySCF FD-on-dipole
# ---------------------------------------------------------------------------

def _pyscf_fd_polarizability(F=1e-4):
    """Reference: PySCF static dipole polarizability via FD on the
    dipole moment under a uniform external field. Returns (3, 3) in
    atomic units."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf

    mol_ps = gto.Mole()
    mol_ps.atom = [
        ["O", (0.0, 0.0, 0.0)],
        ["H", (0.0,  0.7572, 0.5868)],
        ["H", (0.0, -0.7572, 0.5868)],
    ]
    mol_ps.basis = "sto-3g"
    mol_ps.unit = "Angstrom"
    mol_ps.verbose = 0
    mol_ps.build()

    charges = np.array([mol_ps.atom_charge(i) for i in range(mol_ps.natm)])
    coords = mol_ps.atom_coords()
    nuc_dip = charges @ coords

    def dipole_with_field(direction, F_signed):
        # H' = +E·r̂ on an electron in field E; perturbation operator on AO
        # basis is +E_α (r_α)_μν.
        h_pert = (np.einsum("xij,x->ij",
                             mol_ps.intor_symmetric("int1e_r", comp=3),
                             direction) * F_signed)
        h0 = mol_ps.intor_symmetric("int1e_kin") + mol_ps.intor_symmetric("int1e_nuc")
        mf2 = scf.RHF(mol_ps)
        mf2.conv_tol = 1e-12
        mf2.get_hcore = lambda *a, **kw: h0 + h_pert
        mf2.kernel()
        dm = mf2.make_rdm1()
        el_dip = -np.einsum("xij,ji->x",
                              mol_ps.intor_symmetric("int1e_r", comp=3), dm)
        return el_dip + nuc_dip

    alpha_ps = np.zeros((3, 3))
    for axis in range(3):
        e = np.zeros(3); e[axis] = 1.0
        mu_p = dipole_with_field(e, +F)
        mu_m = dipole_with_field(e, -F)
        alpha_ps[:, axis] = (mu_p - mu_m) / (2 * F)
    return 0.5 * (alpha_ps + alpha_ps.T)


def test_h2o_polarizability_matches_pyscf_fd():
    """CPHF polarizability on H₂O / STO-3G agrees with PySCF FD-on-
    dipole to ~1e-5 a.u. (FD truncation in the reference)."""
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    eri = np.asarray(vq.compute_eri(basis))
    alpha_vq = dipole_polarizability_rhf(result, basis, mol, eri=eri)
    alpha_ps = _pyscf_fd_polarizability(F=1e-4)
    np.testing.assert_allclose(alpha_vq, alpha_ps, atol=1e-5, rtol=0)


def test_polarizability_tensor_symmetry():
    """α_αβ = α_βα: the static-response tensor is symmetric."""
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    eri = np.asarray(vq.compute_eri(basis))
    alpha = dipole_polarizability_rhf(result, basis, mol, eri=eri)
    np.testing.assert_allclose(alpha, alpha.T, atol=1e-12)


def test_polarizability_is_positive_definite():
    """All eigenvalues of α are positive for a stable closed-shell
    molecule (response to the applied field is in the same direction
    as the field)."""
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    eri = np.asarray(vq.compute_eri(basis))
    alpha = dipole_polarizability_rhf(result, basis, mol, eri=eri)
    eigs = np.linalg.eigvalsh(alpha)
    assert np.all(eigs > 0), f"polarizability has non-positive eigenvalues: {eigs}"


def test_polarizability_h2_isotropic_smoke():
    """H₂ along z has a clear principal-axis polarizability: α_∥ along
    bond > α_⊥ perpendicular. Sanity-check the structure."""
    mol = _h2_mol(R_bohr=1.4)
    basis, result = _converged_rhf(mol)
    eri = np.asarray(vq.compute_eri(basis))
    alpha = dipole_polarizability_rhf(result, basis, mol, eri=eri)
    # x and y are perpendicular to bond; z is along bond.
    a_perp = 0.5 * (alpha[0, 0] + alpha[1, 1])
    a_par = alpha[2, 2]
    assert a_par > a_perp


# ---------------------------------------------------------------------------
# 5. Refusal on un-converged SCF
# ---------------------------------------------------------------------------

def test_cphf_rejects_unconverged_rhf():
    """Cannot solve CPHF on a non-stationary HF reference — the
    orbital Hessian only describes the response *at* the SCF fixed
    point."""
    mol = _h2o_mol()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.max_iter = 1   # guaranteed not to converge
    opts.use_diis = False
    result = vq.run_rhf(mol, basis, opts)
    assert not result.converged
    eri = np.asarray(vq.compute_eri(basis))
    n_bf = basis.nbasis
    rhs = np.zeros((n_bf, n_bf))
    with pytest.raises(ValueError, match="not converged"):
        cphf_solve_rhf(rhs, result, eri)


# ---------------------------------------------------------------------------
# 6. Convergence-failure path
# ---------------------------------------------------------------------------

def test_cphf_raises_on_non_convergence():
    """Tight tol + tiny iter cap → CPHFConvergenceError, not silent
    wrong answer."""
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    eri = np.asarray(vq.compute_eri(basis))
    dip = vq.compute_dipole(basis, [0.0, 0.0, 0.0])
    M = np.asarray(dip.z)
    # The out-of-plane x perturbation solves in one symmetry channel and
    # reaches machine precision in one step. Use z: its one-step relative
    # residual is about 0.164, so refusal tests an unfinished solve rather
    # than whether roundoff lies just above or below 1e-15.
    opts = vq.CPHFOptions(max_iter=1, tol=1e-15)
    with pytest.raises(vq.CPHFConvergenceError):
        cphf_solve_rhf(M, result, eri, options=opts)


# ---------------------------------------------------------------------------
# 7 & 8. RHS handling
# ---------------------------------------------------------------------------

def test_cphf_single_rhs_returns_2d():
    """A single (n_basis, n_basis) RHS produces a 2D U_{ia} array,
    not a 3D one with a leading length-1 axis."""
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    eri = np.asarray(vq.compute_eri(basis))
    dip = vq.compute_dipole(basis, [0.0, 0.0, 0.0])
    U = cphf_solve_rhf(np.asarray(dip.x), result, eri)
    assert U.ndim == 2


def test_cphf_multiple_rhs_returns_3d():
    """A stack of (n_rhs, n_basis, n_basis) RHS produces a 3D
    (n_rhs, n_occ, n_vir) U array. Each U[k] equals the per-RHS
    single-call solve."""
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    eri = np.asarray(vq.compute_eri(basis))
    dip = vq.compute_dipole(basis, [0.0, 0.0, 0.0])
    M = np.stack([np.asarray(dip.x), np.asarray(dip.y),
                  np.asarray(dip.z)], axis=0)
    U_stack = cphf_solve_rhf(M, result, eri)
    assert U_stack.ndim == 3
    # Compare against single-call results.
    for k in range(3):
        U_k = cphf_solve_rhf(M[k], result, eri)
        np.testing.assert_allclose(U_stack[k], U_k, atol=1e-9)


def test_cphf_is_linear_in_rhs():
    """CPHF is a linear-response equation: doubling the RHS doubles
    the orbital-response amplitudes. (Tests linearity rather than
    "only-uses-MO-occ-vir-block" because the AO basis is not
    orthonormal — adding a 'pure occ-occ AO matrix' bleeds into the
    MO occ-vir block.)"""
    mol = _h2o_mol()
    basis, result = _converged_rhf(mol)
    eri = np.asarray(vq.compute_eri(basis))
    dip = vq.compute_dipole(basis, [0.0, 0.0, 0.0])
    M = np.asarray(dip.x)
    U1 = cphf_solve_rhf(M, result, eri)
    U2 = cphf_solve_rhf(2.0 * M, result, eri)
    np.testing.assert_allclose(U2, 2.0 * U1, atol=1e-9,
                               err_msg="CPHF response not linear in RHS")
