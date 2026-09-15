"""Phase 17c -- analytic UHF Hessian assembly.

Per-spin extension of :func:`vibeqc.compute_hessian_rhf_analytic`
(Phase 17b-3). Solves the UHF coupled-perturbed Hartree-Fock equations
faithfully ported from :func:`pyscf.scf.ucphf.solve` and assembles
the (3N, 3N) Hessian by summing a and b response contributions.

Hessian formula (per PySCF ``hessian/uhf.py``)::

    H_{Aa,Bb} =  d^2E_nuc/dR_AadR_Bb
              +  S_muν (D_a+D_b)_muν . d^2(T+V)/dR                     (1-e skeleton)
              +  S_muνls Γ_UHF . d^2(muν|ls)/dR                          (2-e skeleton)
              -  S_muν (W_a+W_b)_muν . d^2S/dR                          (overlap-Lagrangian)
              +  2 . tr(h1ao_a[A] . dm1_a[B]) + 2 . tr(h1ao_b[A] . dm1_b[B])
              -  2 . tr(s1ao[A] . e_a.dm1_a[B]) - 2 . tr(s1ao[A] . e_b.dm1_b[B])
              -  tr(s1oo_a[A] . mo_e1_a[B]) - tr(s1oo_b[A] . mo_e1_b[B])

The factor 2 (vs RHF's 4) is the spin-restricted convention: each
spin contributes its own factor 2 (for "+c.c."), summed over both
spins.

V1 implementation: per-spin ``h1ao_s[A, d]`` via FD on the AO Fock
matrix at fixed reference density (12N Fock builds, 6N for each
spin). Future optimization: libint deriv_order=1 + shell-slice.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._vibeqc_core import (
    Atom,
    BasisSet,
    GridOptions,
    Molecule,
    UHFOptions,
    UHFResult,
    build_coulomb,
    build_exchange,
    build_fock_g,
    compute_eri,
    compute_eri_hessian_contribution_uhf,
    compute_kinetic,
    compute_kinetic_nuclear_hessian_contribution,
    compute_nuclear,
    compute_overlap,
    compute_overlap_hessian_contribution,
    nuclear_repulsion_hessian,
    run_uhf,
)
from .cphf import CPHFOptions
from .hessian import (
    HessianResult,
    HessianFDOptions,
    _AMU_TO_ELECTRON_MASS,
    _atom_positions_bohr,
    _build_trans_rot_modes,
    _enforce_projected_zero,
    _omega2_to_cm_inv,
    _project_out,
    _resolve_masses,
)
from .hessian_analytic import _displaced_molecule


# ----------------------------------------------------------------------
# Internal -- UHF Fock-at-geometry and FD h1
# ----------------------------------------------------------------------

def _uhf_fock_at_geometry(mol: Molecule, basis_name: str,
                           D_alpha: np.ndarray,
                           D_beta: np.ndarray,
                           alpha_hf: float = 1.0):
    """UHF Fock matrices at the given geometry with the supplied a and
    b densities held fixed (no SCF). Returns ``(F_a, F_b)``.

    F_s = T + V + J(D_a + D_b) - a_HF . K(D_s)
    """
    basis = BasisSet(mol, basis_name)
    T = np.asarray(compute_kinetic(basis))
    V = np.asarray(compute_nuclear(basis, mol))
    Hcore = T + V
    eri = np.asarray(compute_eri(basis))
    D_total = D_alpha + D_beta
    J = np.asarray(build_coulomb(eri, D_total))
    K_a = np.asarray(build_exchange(eri, D_alpha))
    K_b = np.asarray(build_exchange(eri, D_beta))
    F_alpha = Hcore + J - alpha_hf * K_a
    F_beta = Hcore + J - alpha_hf * K_b
    return F_alpha, F_beta


def _build_uhf_h1ao_fd(mol: Molecule, basis_name: str,
                        D_alpha: np.ndarray, D_beta: np.ndarray,
                        alpha_hf: float = 1.0,
                        step_bohr: float = 1e-4):
    """Per-spin AO Fock first-derivative tensors via FD.

    Returns ``(h1ao_a, h1ao_b)`` each of shape
    ``(n_atoms, 3, n_basis, n_basis)``.
    """
    n_atoms = len(mol.atoms)
    n_basis = D_alpha.shape[0]
    h1a = np.zeros((n_atoms, 3, n_basis, n_basis), dtype=np.float64)
    h1b = np.zeros((n_atoms, 3, n_basis, n_basis), dtype=np.float64)
    for A in range(n_atoms):
        for d in range(3):
            mol_p = _displaced_molecule(mol, A, d, +step_bohr)
            mol_m = _displaced_molecule(mol, A, d, -step_bohr)
            Fa_p, Fb_p = _uhf_fock_at_geometry(mol_p, basis_name,
                                                D_alpha, D_beta, alpha_hf)
            Fa_m, Fb_m = _uhf_fock_at_geometry(mol_m, basis_name,
                                                D_alpha, D_beta, alpha_hf)
            h1a[A, d] = (Fa_p - Fa_m) / (2.0 * step_bohr)
            h1b[A, d] = (Fb_p - Fb_m) / (2.0 * step_bohr)
    return h1a, h1b


def _build_s1ao_fd(mol: Molecule, basis_name: str,
                    step_bohr: float = 1e-4) -> np.ndarray:
    """Reuse RHF version -- overlap is spin-independent."""
    from .hessian_analytic import _build_s1ao_fd as _impl
    return _impl(mol, basis_name, step_bohr)


# ----------------------------------------------------------------------
# UHF CPHF -- port of pyscf/scf/ucphf.solve_withs1
# ----------------------------------------------------------------------

def _solve_cphf_uhf_nuclear(uhf_result: UHFResult,
                              eri: np.ndarray,
                              h1ao_a: np.ndarray,
                              h1ao_b: np.ndarray,
                              s1ao: np.ndarray,
                              alpha_hf: float = 1.0,
                              options: Optional[CPHFOptions] = None):
    """Solve the UHF CPHF equations for all nuclear-coordinate
    perturbations. Returns ``(mo1_a, mo1_b, mo_e1_a, mo_e1_b)``.

    Each ``mo1_s[A, d]`` has shape ``(n_basis, n_occ_s)`` (AO basis,
    transformed back from MO solution like PySCF). ``mo_e1_s[A, d]``
    has shape ``(n_occ_s, n_occ_s)``.

    The orbital-Hessian operator couples a and b: ``fvind(mo1)[s] =
    J(dm1_a + dm1_b) - a_HF . K(dm1_s)`` where dm1_s is the AO
    response density built from mo1_s.
    """
    if options is None:
        options = CPHFOptions()
    max_iter = int(options.max_iter)
    tol = float(options.tol)

    if not uhf_result.converged:
        raise ValueError("hessian_analytic_uhf: UHFResult is not converged.")

    Ca = np.asarray(uhf_result.mo_coeffs_alpha, dtype=np.float64)
    Cb = np.asarray(uhf_result.mo_coeffs_beta, dtype=np.float64)
    eps_a_full = np.asarray(uhf_result.mo_energies_alpha, dtype=np.float64)
    eps_b_full = np.asarray(uhf_result.mo_energies_beta, dtype=np.float64)
    Da = np.asarray(uhf_result.density_alpha, dtype=np.float64)
    Db = np.asarray(uhf_result.density_beta, dtype=np.float64)
    n_basis = Ca.shape[0]
    n_occ_a = int(round(np.trace(Da @ Da) ** 0.5 * np.sqrt(np.linalg.norm(Da)
                                                              / max(np.trace(Da), 1e-14))))
    # Cleaner: rank of Da == n_occ_a (since Da = C_a_occ C_a_occ^T)
    n_occ_a = int(round(np.sum(np.linalg.eigvalsh(Da) > 1e-8)))
    n_occ_b = int(round(np.sum(np.linalg.eigvalsh(Db) > 1e-8)))
    n_vir_a = n_basis - n_occ_a
    n_vir_b = n_basis - n_occ_b
    Ca_occ = Ca[:, :n_occ_a]
    Cb_occ = Cb[:, :n_occ_b]
    e_i_a = eps_a_full[:n_occ_a]
    e_a_a = eps_a_full[n_occ_a:]
    e_i_b = eps_b_full[:n_occ_b]
    e_a_b = eps_b_full[n_occ_b:]
    e_ai_a = 1.0 / (e_a_a[:, None] - e_i_a)
    e_ai_b = 1.0 / (e_a_b[:, None] - e_i_b)

    eri_arr = np.asarray(eri, dtype=np.float64)
    n_atoms = h1ao_a.shape[0]
    n_pert = n_atoms * 3

    # Transform h1ao and s1ao to MO basis (full nmo x nocc).
    h1mo_a = np.zeros((n_pert, n_basis, n_occ_a))
    h1mo_b = np.zeros((n_pert, n_basis, n_occ_b))
    s1mo_a = np.zeros((n_pert, n_basis, n_occ_a))
    s1mo_b = np.zeros((n_pert, n_basis, n_occ_b))
    for k, (A, d) in enumerate(((A, d) for A in range(n_atoms) for d in range(3))):
        h1mo_a[k] = Ca.T @ h1ao_a[A, d] @ Ca_occ
        h1mo_b[k] = Cb.T @ h1ao_b[A, d] @ Cb_occ
        s1mo_a[k] = Ca.T @ s1ao[A, d] @ Ca_occ
        s1mo_b[k] = Cb.T @ s1ao[A, d] @ Cb_occ

    hs_a = h1mo_a - s1mo_a * e_i_a[None, None, :]
    hs_b = h1mo_b - s1mo_b * e_i_b[None, None, :]

    # Initial mo1: vir block from -e_ai * hs[vir]; occ block fixed at -1/2.s1_oo.
    mo1_a = hs_a.copy()
    mo1_a[:, n_occ_a:, :] *= -e_ai_a[None, :, :]
    mo1_a[:, :n_occ_a, :] = -0.5 * s1mo_a[:, :n_occ_a, :]
    mo1_b = hs_b.copy()
    mo1_b[:, n_occ_b:, :] *= -e_ai_b[None, :, :]
    mo1_b[:, :n_occ_b, :] = -0.5 * s1mo_b[:, :n_occ_b, :]

    # fvind: applies the UHF orbital-Hessian operator to (mo1_a, mo1_b).
    # For UHF: dm1_s = 2.C_s.mo1_s.C_occ_s.T + transpose; v_s = J(dm1_a + dm1_b)
    # - a_HF . K(dm1_s) ; transform back to MO occ-vir basis.
    # Note: the factor 2 in dm1 is the closed-shell convention from RHF;
    # for UHF we want dm1_s = mo1_s . C_occ_s.T + transpose (no factor 2)
    # because each spin is its own one-particle density. Then J / K
    # follow with their natural prefactors.
    def fvind(m_a, m_b):
        v_a = np.zeros_like(m_a)
        v_b = np.zeros_like(m_b)
        for k in range(m_a.shape[0]):
            dm_a = Ca @ m_a[k] @ Ca_occ.T
            dm_a = dm_a + dm_a.T
            dm_b = Cb @ m_b[k] @ Cb_occ.T
            dm_b = dm_b + dm_b.T
            J_ao = np.asarray(build_coulomb(eri_arr, dm_a + dm_b))
            K_a_ao = np.asarray(build_exchange(eri_arr, dm_a))
            K_b_ao = np.asarray(build_exchange(eri_arr, dm_b))
            v_a_ao = J_ao - alpha_hf * K_a_ao
            v_b_ao = J_ao - alpha_hf * K_b_ao
            v_a[k] = Ca.T @ v_a_ao @ Ca_occ
            v_b[k] = Cb.T @ v_b_ao @ Cb_occ
        return v_a, v_b

    # LGMRES (loose-restart GMRES) on stacked (mo1_a_vir, mo1_b_vir). The
    # a-occ-occ and b-occ-occ blocks stay fixed at -1/2.s1_oo throughout the
    # iteration. We use lgmres rather than vanilla gmres because the
    # UCPHF orbital-Hessian operator on open-shell systems (e.g. OH
    # radical) has a richer eigenvalue spectrum than RHF, and gmres'
    # truncated Krylov subspace can stall while lgmres' "loose-restart"
    # mechanism (it carries vectors across restarts) handles it cleanly.
    from scipy.sparse.linalg import lgmres, LinearOperator

    n_a_vec = n_pert * n_vir_a * n_occ_a
    n_b_vec = n_pert * n_vir_b * n_occ_b
    n_total = n_a_vec + n_b_vec

    mo1_a_scratch = mo1_a.copy()
    mo1_b_scratch = mo1_b.copy()

    def matvec(x_flat):
        xa = x_flat[:n_a_vec].reshape(n_pert, n_vir_a, n_occ_a)
        xb = x_flat[n_a_vec:].reshape(n_pert, n_vir_b, n_occ_b)
        mo1_a_scratch[:, n_occ_a:, :] = xa
        mo1_b_scratch[:, n_occ_b:, :] = xb
        v_a, v_b = fvind(mo1_a_scratch, mo1_b_scratch)
        out_a = (xa + e_ai_a[None, :, :] * v_a[:, n_occ_a:, :]).ravel()
        out_b = (xb + e_ai_b[None, :, :] * v_b[:, n_occ_b:, :]).ravel()
        return np.concatenate([out_a, out_b])

    A = LinearOperator((n_total, n_total), matvec=matvec, dtype=np.float64)
    b_a = mo1_a[:, n_occ_a:, :].ravel().copy()
    b_b = mo1_b[:, n_occ_b:, :].ravel().copy()
    b = np.concatenate([b_a, b_b])
    x0 = b.copy()

    x_solution, info = lgmres(A, b, x0=x0, rtol=tol, atol=tol,
                                maxiter=max_iter)
    if info != 0:
        raise RuntimeError(
            f"hessian_analytic_uhf: LGMRES UCPHF did not converge "
            f"(info = {info}, max_iter = {max_iter}).")

    mo1_a[:, n_occ_a:, :] = x_solution[:n_a_vec].reshape(n_pert, n_vir_a, n_occ_a)
    mo1_b[:, n_occ_b:, :] = x_solution[n_a_vec:].reshape(n_pert, n_vir_b, n_occ_b)

    # Final refinement: hs_s += fvind_s; mo1_s[vir] = hs_s[vir] / (e_i - e_a).
    v_a_final, v_b_final = fvind(mo1_a, mo1_b)
    hs_a_final = hs_a + v_a_final
    hs_b_final = hs_b + v_b_final
    mo1_a[:, n_occ_a:, :] = hs_a_final[:, n_occ_a:, :] / (
        e_i_a[None, None, :] - e_a_a[:, None][None, :, :]
    )
    mo1_b[:, n_occ_b:, :] = hs_b_final[:, n_occ_b:, :] / (
        e_i_b[None, None, :] - e_a_b[:, None][None, :, :]
    )

    # Orbital-energy response per spin
    e_diff_a = e_i_a[:, None] - e_i_a[None, :]
    e_diff_b = e_i_b[:, None] - e_i_b[None, :]
    mo_e1_a = hs_a_final[:, :n_occ_a, :] + mo1_a[:, :n_occ_a, :] * e_diff_a[None, :, :]
    mo_e1_b = hs_b_final[:, :n_occ_b, :] + mo1_b[:, :n_occ_b, :] * e_diff_b[None, :, :]

    # Reshape and transform to AO for the assembler.
    mo1_a = mo1_a.reshape(n_atoms, 3, n_basis, n_occ_a)
    mo1_b = mo1_b.reshape(n_atoms, 3, n_basis, n_occ_b)
    mo_e1_a = mo_e1_a.reshape(n_atoms, 3, n_occ_a, n_occ_a)
    mo_e1_b = mo_e1_b.reshape(n_atoms, 3, n_occ_b, n_occ_b)

    mo1_a_ao = np.zeros((n_atoms, 3, n_basis, n_occ_a))
    mo1_b_ao = np.zeros((n_atoms, 3, n_basis, n_occ_b))
    for A in range(n_atoms):
        for d in range(3):
            mo1_a_ao[A, d] = Ca @ mo1_a[A, d]
            mo1_b_ao[A, d] = Cb @ mo1_b[A, d]
    return mo1_a_ao, mo1_b_ao, mo_e1_a, mo_e1_b


# ----------------------------------------------------------------------
# Public driver
# ----------------------------------------------------------------------

def compute_hessian_uhf_analytic(
    mol: Molecule,
    basis: BasisSet,
    uhf_result: UHFResult,
    *,
    basis_name: Optional[str] = None,
    eri: Optional[np.ndarray] = None,
    alpha_hf: float = 1.0,
    fd_step_bohr: float = 1e-4,
    cphf_options: Optional[CPHFOptions] = None,
    hessian_options: Optional[HessianFDOptions] = None,
) -> HessianResult:
    """Compute the open-shell-UHF Hessian via per-spin CPHF + skeleton
    2nd-derivative integrals."""
    from .ecp_metadata import refuse_molecular_ecp_derivative_route

    refuse_molecular_ecp_derivative_route(
        mol,
        basis,
        result=uhf_result,
        route="compute_hessian_uhf_analytic",
    )
    if basis_name is None:
        raise ValueError(
            "compute_hessian_uhf_analytic requires basis_name explicitly")
    if hessian_options is None:
        hessian_options = HessianFDOptions()
    if eri is None:
        eri = np.asarray(compute_eri(basis), dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)

    if not uhf_result.converged:
        raise ValueError(
            "compute_hessian_uhf_analytic: UHFResult is not converged.")

    Ca = np.asarray(uhf_result.mo_coeffs_alpha, dtype=np.float64)
    Cb = np.asarray(uhf_result.mo_coeffs_beta, dtype=np.float64)
    eps_a = np.asarray(uhf_result.mo_energies_alpha, dtype=np.float64)
    eps_b = np.asarray(uhf_result.mo_energies_beta, dtype=np.float64)
    Da = np.asarray(uhf_result.density_alpha, dtype=np.float64)
    Db = np.asarray(uhf_result.density_beta, dtype=np.float64)
    D_total = Da + Db
    n_basis = Ca.shape[0]
    n_occ_a = int(round(np.sum(np.linalg.eigvalsh(Da) > 1e-8)))
    n_occ_b = int(round(np.sum(np.linalg.eigvalsh(Db) > 1e-8)))
    Ca_occ = Ca[:, :n_occ_a]
    Cb_occ = Cb[:, :n_occ_b]
    e_i_a = eps_a[:n_occ_a]
    e_i_b = eps_b[:n_occ_b]
    # Energy-weighted densities per spin, then summed for the overlap term.
    Wa = Ca_occ @ np.diag(e_i_a) @ Ca_occ.T
    Wb = Cb_occ @ np.diag(e_i_b) @ Cb_occ.T
    W_total = Wa + Wb

    n_atoms = len(mol.atoms)
    Ndof = 3 * n_atoms

    # ---- 1. Skeleton ------------------------------------------------
    H = np.zeros((Ndof, Ndof), dtype=np.float64)
    H += np.asarray(nuclear_repulsion_hessian(mol))
    H += np.asarray(compute_kinetic_nuclear_hessian_contribution(basis, mol, D_total))
    H += np.asarray(compute_eri_hessian_contribution_uhf(basis, mol, Da, Db, alpha_hf))
    H += np.asarray(compute_overlap_hessian_contribution(basis, mol, W_total))

    # ---- 2. Per-spin h1ao + shared s1ao via FD ---------------------
    h1a, h1b = _build_uhf_h1ao_fd(mol, basis_name, Da, Db, alpha_hf,
                                    step_bohr=fd_step_bohr)
    s1ao = _build_s1ao_fd(mol, basis_name, step_bohr=fd_step_bohr)

    # ---- 3. UHF CPHF -----------------------------------------------
    mo1a, mo1b, mo_e1a, mo_e1b = _solve_cphf_uhf_nuclear(
        uhf_result, eri, h1a, h1b, s1ao, alpha_hf=alpha_hf,
        options=cphf_options)

    # ---- 4. Response part of Hessian -------------------------------
    # H_xy_response = +2 tr(h1ao_a[A] . dm1_a[B]) + 2 tr(h1ao_b[A] . dm1_b[B])
    #                -2 tr(s1ao[A] . e_a.dm1_a[B]) - 2 tr(s1ao[A] . e_b.dm1_b[B])
    #                -  tr(s1oo_a[A] . mo_e1_a[B]) -  tr(s1oo_b[A] . mo_e1_b[B])
    for A in range(n_atoms):
        for da in range(3):
            s1oo_a_Ada = Ca_occ.T @ s1ao[A, da] @ Ca_occ
            s1oo_b_Ada = Cb_occ.T @ s1ao[A, da] @ Cb_occ
            for B in range(n_atoms):
                for db in range(3):
                    dm1a = mo1a[B, db] @ Ca_occ.T
                    dm1b = mo1b[B, db] @ Cb_occ.T
                    eps_dm1a = (mo1a[B, db] * e_i_a[None, :]) @ Ca_occ.T
                    eps_dm1b = (mo1b[B, db] * e_i_b[None, :]) @ Cb_occ.T

                    t1a = 2.0 * np.einsum("uv,vu->", h1a[A, da], dm1a)
                    t1b = 2.0 * np.einsum("uv,vu->", h1b[A, da], dm1b)
                    t2a = -2.0 * np.einsum("uv,vu->", s1ao[A, da], eps_dm1a)
                    t2b = -2.0 * np.einsum("uv,vu->", s1ao[A, da], eps_dm1b)
                    t3a = -1.0 * np.einsum("ij,ji->", s1oo_a_Ada, mo_e1a[B, db])
                    t3b = -1.0 * np.einsum("ij,ji->", s1oo_b_Ada, mo_e1b[B, db])
                    H[3*A + da, 3*B + db] += t1a + t1b + t2a + t2b + t3a + t3b

    H = 0.5 * (H + H.T)

    # ---- 5. Mass-weight + diagonalize ------------------------------
    masses_amu = _resolve_masses(mol, hessian_options.atomic_masses_amu)
    masses_e = masses_amu * _AMU_TO_ELECTRON_MASS
    inv_sqrt_m_per_dof = np.repeat(1.0 / np.sqrt(masses_e), 3)
    M_inv_sqrt = inv_sqrt_m_per_dof
    H_mw = (M_inv_sqrt[:, None] * H) * M_inv_sqrt[None, :]
    H_mw = 0.5 * (H_mw + H_mw.T)

    is_linear = False
    if hessian_options.project_trans_rot:
        positions = _atom_positions_bohr(mol)
        zero_modes, is_linear = _build_trans_rot_modes(positions, masses_e)
        H_proj = _project_out(H_mw, zero_modes)
        omega2, modes = np.linalg.eigh(0.5 * (H_proj + H_proj.T))
        n_zero = zero_modes.shape[1]
        omega2, modes = _enforce_projected_zero(omega2, modes, n_zero)
    else:
        omega2, modes = np.linalg.eigh(H_mw)
    order = np.argsort(omega2)
    omega2 = omega2[order]
    modes = modes[:, order]
    freqs = _omega2_to_cm_inv(omega2)
    imag_count = int(np.sum(omega2 < -1e-10))

    return HessianResult(
        hessian=H,
        hessian_mw=H_mw,
        frequencies_cm1=freqs,
        normal_modes=modes,
        imaginary_count=imag_count,
        n_displacements=0,
        is_linear=is_linear,
        masses_amu=masses_amu,
    )


__all__ = ["compute_hessian_uhf_analytic"]
