"""Phase 17d -- analytic RKS Hessian assembly.

Closed-shell-DFT extension of :func:`vibeqc.compute_hessian_rhf_analytic`
(Phase 17b-3). Two new ingredients vs HF:

1. **Hybrid HF-exchange fraction a_HF**. The 2-electron skeleton uses
   :func:`compute_eri_hessian_contribution(alpha_hf=...)` (already
   wired in 17b-2). For pure DFT (LDA, PBE, ...) a_HF = 0 and the
   K-derivative drops out; for hybrids (B3LYP, wB97X, ...) it scales.

2. **XC kernel response in CPHF**. The orbital-Hessian operator
   ``A`` picks up an additional grid-based piece
   :math:`\\int \\chi_\\mu(r) f_{xc}(r) \\rho^v(r) \\chi_\\nu(r) \\, dr`
   where :math:`\\rho^v(r) = \\sum D^v_{\\lambda\\sigma}
   \\chi_\\lambda(r) \\chi_\\sigma(r)` is the AO response density.

   - **LDA**: f_xc = d^2e_xc/dr^2. Single grid integration.
   - **GGA**: also picks up ``v2rhosigma`` and ``v2sigma2`` plus
     gradr-dependent terms via the chain rule.
   The implementation here covers LDA fully and the LDA-only path
   for GGA (i.e., we drop s-coupled fxc terms -- *approximate*
   for GGA, exact for LDA). Phase 17d-2 will add the full GGA
   kernel.

3. **XC skeleton**. We avoid implementing 2nd-derivative AO
   evaluation + 2nd-derivative Becke partition by re-using
   :func:`compute_gradient_rks` at the displaced geometry with the
   *reference* RKSResult. FD-on-gradient at fixed reference
   density gives the full skeleton (incl XC) by construction. Cost:
   6N gradient evaluations per Hessian (each involves one Fock build
   + grid integration but no SCF) instead of 6N full SCFs (the
   17a-1 path) -- typically ~20x faster.

V1 implementation (this commit):
- Skeleton: FD on compute_gradient_rks at fixed reference RKSResult
- h1ao: FD on F_KS = h + J - (a_HF/2) K + V_xc at fixed D
- CPHF: analytic LDA fxc kernel; for GGA, fxc treated approximately
  (LDA terms only -- converges in practice but Hessian numbers will
  differ from PySCF GGA by ~1e-3 on the s-coupled terms)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._vibeqc_core import (
    Atom,
    BasisSet,
    Functional,
    GridOptions,
    Molecule,
    RKSOptions,
    RKSResult,
    XCKind,
    build_coulomb,
    build_exchange,
    build_fock_g,
    build_grid,
    compute_eri,
    compute_eri_hessian_contribution,
    compute_gradient_rks,
    compute_kinetic,
    compute_kinetic_nuclear_hessian_contribution,
    compute_nuclear,
    compute_overlap,
    compute_overlap_hessian_contribution,
    evaluate_ao_with_gradient,
    nuclear_repulsion_hessian,
    run_rks,
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
# Internal -- KS Fock build at given (mol, D) (no SCF)
# ----------------------------------------------------------------------

def _ks_fock_at_geometry(mol: Molecule, basis_name: str, D: np.ndarray,
                         functional_name: str, alpha_hf: float,
                         grid_options: GridOptions):
    """Build F_KS = T + V + J - (a_HF/2).K + V_xc at the given geometry,
    with density D held fixed (no SCF). Returns ``F`` AO matrix and
    the grid + chi values used for V_xc (so the caller can reuse them
    for the kernel response if at the same geometry)."""
    basis = BasisSet(mol, basis_name)
    T = np.asarray(compute_kinetic(basis))
    V = np.asarray(compute_nuclear(basis, mol))
    Hcore = T + V
    eri = np.asarray(compute_eri(basis))
    J = np.asarray(build_coulomb(eri, D))
    if alpha_hf != 0.0:
        K = np.asarray(build_exchange(eri, D))
        F_jk = J - 0.5 * alpha_hf * K
    else:
        F_jk = J
    # XC piece
    grid = build_grid(mol, grid_options)
    chi, dchi_x, dchi_y, dchi_z = evaluate_ao_with_gradient(basis, grid.points)
    chi = np.asarray(chi)
    chiD = chi @ D
    rho = (chiD * chi).sum(axis=1)
    func = Functional(functional_name, spin=1)
    is_gga = (func.kind == XCKind.GGA)
    if is_gga:
        # gradr_a = 2 S (chiD) . dchi_a
        dchi = [np.asarray(dchi_x), np.asarray(dchi_y), np.asarray(dchi_z)]
        gx = 2.0 * (chiD * dchi[0]).sum(axis=1)
        gy = 2.0 * (chiD * dchi[1]).sum(axis=1)
        gz = 2.0 * (chiD * dchi[2]).sum(axis=1)
        sigma = gx**2 + gy**2 + gz**2
    else:
        sigma = np.zeros_like(rho)
        dchi = None
        gx = gy = gz = None
    exc, v_rho, v_sigma = func.eval_unpolarised(rho, sigma)
    weights = np.asarray(grid.weights)
    # LDA piece of V_xc -- V_muν = S_g w(g) v_r(g) chi_mu(g) chi_ν(g)
    # = chi^T (w . v_r) chi implemented as broadcasted elementwise
    # multiply (avoids materialising an (n_pts, n_pts) diag matrix
    # which for typical molecular grids is gigabytes).
    w_vrho = weights * v_rho                          # (n_pts,)
    V_xc = chi.T @ (w_vrho[:, None] * chi)
    if is_gga:
        # GGA piece (same trick): F_grad shape (n_pts, n_basis); the
        # diag(u) sandwich becomes a column-wise scaling by u.
        u = 2.0 * weights * v_sigma
        F_grad = (gx[:, None] * dchi[0]
                  + gy[:, None] * dchi[1]
                  + gz[:, None] * dchi[2])
        Fu = F_grad.T @ (u[:, None] * chi)
        V_xc += Fu + Fu.T
    return Hcore + F_jk + V_xc


def _build_ks_h1ao_fd(mol: Molecule, basis_name: str, D: np.ndarray,
                       functional_name: str, alpha_hf: float,
                       grid_options: GridOptions,
                       step_bohr: float = 1e-4) -> np.ndarray:
    """h1ao for KS: F_KS first derivative via FD at fixed D."""
    n_atoms = len(mol.atoms)
    n_basis = D.shape[0]
    h1ao = np.zeros((n_atoms, 3, n_basis, n_basis), dtype=np.float64)
    for A in range(n_atoms):
        for d in range(3):
            mol_p = _displaced_molecule(mol, A, d, +step_bohr)
            mol_m = _displaced_molecule(mol, A, d, -step_bohr)
            Fp = _ks_fock_at_geometry(mol_p, basis_name, D,
                                       functional_name, alpha_hf, grid_options)
            Fm = _ks_fock_at_geometry(mol_m, basis_name, D,
                                       functional_name, alpha_hf, grid_options)
            h1ao[A, d] = (Fp - Fm) / (2.0 * step_bohr)
    return h1ao


# ----------------------------------------------------------------------
# Skeleton via FD on the analytic gradient at fixed reference RKSResult
# ----------------------------------------------------------------------

def _ks_skeleton_via_fd_gradient(
    mol: Molecule, basis: BasisSet, basis_name: str,
    rks_result: RKSResult, grid_options: GridOptions,
    step_bohr: float = 1e-4) -> np.ndarray:
    """Skeleton (3N, 3N) Hessian via central-difference of the analytic
    KS gradient at fixed reference ``rks_result``. The reference
    result's C, e, D are held fixed numerically across all displaced
    geometries; only the basis (re-built per displacement) and the
    grid (re-built per displacement) reflect the new geometry. This
    captures all geometry-dependent pieces of the gradient *at fixed
    density*, including the XC contribution evaluated on the moving
    grid -- exactly what the skeleton requires.
    """
    n_atoms = len(mol.atoms)
    Ndof = 3 * n_atoms
    H_skel = np.zeros((Ndof, Ndof), dtype=np.float64)
    for A in range(n_atoms):
        for d in range(3):
            mol_p = _displaced_molecule(mol, A, d, +step_bohr)
            mol_m = _displaced_molecule(mol, A, d, -step_bohr)
            basis_p = BasisSet(mol_p, basis_name)
            basis_m = BasisSet(mol_m, basis_name)
            g_p = np.asarray(compute_gradient_rks(mol_p, basis_p, rks_result,
                                                    grid_options))
            g_m = np.asarray(compute_gradient_rks(mol_m, basis_m, rks_result,
                                                    grid_options))
            col = (g_p - g_m).reshape(-1) / (2.0 * step_bohr)
            H_skel[:, 3 * A + d] = col
    return 0.5 * (H_skel + H_skel.T)


# ----------------------------------------------------------------------
# CPHF for KS -- extended fvind with XC kernel
# ----------------------------------------------------------------------

def _solve_cphf_rks_nuclear(rks_result: RKSResult,
                              eri: np.ndarray,
                              h1ao: np.ndarray,
                              s1ao: np.ndarray,
                              functional_name: str,
                              alpha_hf: float,
                              grid_options: GridOptions,
                              mol: Molecule,
                              basis: BasisSet,
                              options: Optional[CPHFOptions] = None):
    """KS CPHF for nuclear-coordinate perturbations. Mirrors the
    closed-shell-RHF version in :mod:`vibeqc.hessian_analytic` but
    extends ``fvind`` with the XC kernel response.

    Currently LDA-exact; GGA approximated by LDA fxc terms only
    (Phase 17d-2 will add the s-coupled GGA pieces).
    """
    if options is None:
        options = CPHFOptions()
    max_iter = int(options.max_iter)
    tol = float(options.tol)

    if not rks_result.converged:
        raise ValueError("hessian_analytic_rks: RKSResult is not converged.")

    C = np.asarray(rks_result.mo_coeffs, dtype=np.float64)
    eps = np.asarray(rks_result.mo_energies, dtype=np.float64)
    D = np.asarray(rks_result.density, dtype=np.float64)
    n_basis = C.shape[0]
    n_occ = int(round(np.sum(np.linalg.eigvalsh(D) > 1e-8)))
    n_vir = n_basis - n_occ
    n_mo = n_basis
    C_occ = C[:, :n_occ]
    C_vir = C[:, n_occ:]
    e_i = eps[:n_occ]
    e_a = eps[n_occ:]
    e_ai = 1.0 / (e_a[:, None] - e_i)

    eri_arr = np.asarray(eri, dtype=np.float64)
    n_atoms = h1ao.shape[0]
    n_pert = n_atoms * 3

    # Precompute grid + AO values at the reference geometry; reused
    # every fvind call.
    grid = build_grid(mol, grid_options)
    chi_g, dchi_x_g, dchi_y_g, dchi_z_g = evaluate_ao_with_gradient(
        basis, grid.points)
    chi_g = np.asarray(chi_g)
    weights = np.asarray(grid.weights)

    # Reference rho on the grid (for fxc evaluation).
    chiD = chi_g @ D
    rho_ref = (chiD * chi_g).sum(axis=1)
    func = Functional(functional_name, spin=1)
    is_gga = (func.kind == XCKind.GGA)
    if is_gga:
        dchi = [np.asarray(dchi_x_g), np.asarray(dchi_y_g),
                np.asarray(dchi_z_g)]
        gx = 2.0 * (chiD * dchi[0]).sum(axis=1)
        gy = 2.0 * (chiD * dchi[1]).sum(axis=1)
        gz = 2.0 * (chiD * dchi[2]).sum(axis=1)
        sigma_ref = gx**2 + gy**2 + gz**2
    else:
        sigma_ref = np.zeros_like(rho_ref)

    # Compute the LDA part of fxc (d^2f/dr^2) at the reference. For
    # GGA, the LDA terms are dominant; full GGA fxc is Phase 17d-2.
    fxc_v2rho2, _, _ = func.eval_unpolarised_fxc(rho_ref, sigma_ref)

    # Transform h1ao + s1ao to MO basis.
    h1mo = np.zeros((n_pert, n_mo, n_occ))
    s1mo = np.zeros((n_pert, n_mo, n_occ))
    for k, (A, d) in enumerate(((A, d) for A in range(n_atoms) for d in range(3))):
        h1mo[k] = C.T @ h1ao[A, d] @ C_occ
        s1mo[k] = C.T @ s1ao[A, d] @ C_occ
    hs = h1mo - s1mo * e_i[None, None, :]

    mo1 = hs.copy()
    mo1[:, n_occ:, :] *= -e_ai[None, :, :]
    mo1[:, :n_occ, :] = -0.5 * s1mo[:, :n_occ, :]

    def _xc_kernel_response_ao(D_resp: np.ndarray) -> np.ndarray:
        """LDA-only XC kernel response: ∫ chi_mu f_xc . r_resp chi_ν dr.
        D_resp is the AO response density; r_resp(r) = S_muν D_resp[mu,ν]
        chi_mu(r) chi_ν(r). Returns the AO matrix V_xc_resp."""
        chi_Dresp = chi_g @ D_resp
        rho_resp = (chi_Dresp * chi_g).sum(axis=1)
        # V_xc_resp_muν = ∫ chi_mu(r) [f_xc(r) . r_resp(r)] chi_ν(r) w(r) dr
        # Broadcast multiply instead of np.diag (avoids n_ptsxn_pts
        # dense diagonal matrix -- for typical molecular grids that's
        # multiple GB and OOMs).
        coef = weights * fxc_v2rho2 * rho_resp           # (n_pts,)
        return chi_g.T @ (coef[:, None] * chi_g)

    def fvind(mo1_arr):
        out = np.zeros_like(mo1_arr)
        for k in range(mo1_arr.shape[0]):
            # AO response density (closed-shell convention: factor 2)
            dm = C @ (mo1_arr[k] * 2.0) @ C_occ.T
            dm = dm + dm.T
            # 2-electron Fock-like response: J - (alpha_hf/2) K
            J_ao = np.asarray(build_coulomb(eri_arr, dm))
            if alpha_hf != 0.0:
                K_ao = np.asarray(build_exchange(eri_arr, dm))
                v_2e = J_ao - 0.5 * alpha_hf * K_ao
            else:
                v_2e = J_ao
            # XC kernel response
            v_xc = _xc_kernel_response_ao(dm)
            v_ao = v_2e + v_xc
            out[k] = C.T @ v_ao @ C_occ
        return out

    from scipy.sparse.linalg import lgmres, LinearOperator
    n_vec_size = n_pert * n_vir * n_occ
    mo1_scratch = mo1.copy()

    def matvec(x_flat):
        x = x_flat.reshape(n_pert, n_vir, n_occ)
        mo1_scratch[:, n_occ:, :] = x
        v = fvind(mo1_scratch)
        return (x + e_ai[None, :, :] * v[:, n_occ:, :]).ravel()

    A = LinearOperator((n_vec_size, n_vec_size), matvec=matvec,
                       dtype=np.float64)
    b = mo1[:, n_occ:, :].ravel().copy()
    x0 = b.copy()
    x_solution, info = lgmres(A, b, x0=x0, rtol=tol, atol=tol,
                                maxiter=max_iter)
    if info != 0:
        raise RuntimeError(
            f"hessian_analytic_rks: LGMRES KS-CPHF did not converge "
            f"(info = {info}, max_iter = {max_iter}).")
    mo1[:, n_occ:, :] = x_solution.reshape(n_pert, n_vir, n_occ)

    hs_final = hs + fvind(mo1)
    mo1[:, n_occ:, :] = hs_final[:, n_occ:, :] / (
        e_i[None, None, :] - e_a[:, None][None, :, :])
    e_i_diff = e_i[:, None] - e_i[None, :]
    mo_e1 = hs_final[:, :n_occ, :] + mo1[:, :n_occ, :] * e_i_diff[None, :, :]

    mo1 = mo1.reshape(n_atoms, 3, n_mo, n_occ)
    mo_e1 = mo_e1.reshape(n_atoms, 3, n_occ, n_occ)
    mo1_ao = np.zeros((n_atoms, 3, n_basis, n_occ))
    for A_idx in range(n_atoms):
        for d in range(3):
            mo1_ao[A_idx, d] = C @ mo1[A_idx, d]
    return mo1_ao, mo_e1


# ----------------------------------------------------------------------
# Public driver
# ----------------------------------------------------------------------

def compute_hessian_rks_analytic(
    mol: Molecule,
    basis: BasisSet,
    rks_result: RKSResult,
    *,
    basis_name: Optional[str] = None,
    functional: Optional[str] = None,
    grid_options: Optional[GridOptions] = None,
    eri: Optional[np.ndarray] = None,
    fd_step_bohr: float = 1e-4,
    cphf_options: Optional[CPHFOptions] = None,
    hessian_options: Optional[HessianFDOptions] = None,
) -> HessianResult:
    """Compute the closed-shell-RKS analytic Hessian.

    Skeleton via FD on the analytic KS gradient at fixed reference
    density (so the XC and a_HF.K skeleton pieces come out naturally
    without needing 2nd-deriv AO grids or d^2(muν|ls) integrations
    beyond what 17b-2 provides). Response part via KS CPHF with
    libxc fxc kernel.

    Currently LDA-exact for the CPHF kernel; GGA approximated by
    LDA fxc terms only (the s-coupled fxc pieces are not yet
    implemented). LDA / hybrid / pure-DFT all work via this entry
    point.
    """
    from .ecp_metadata import refuse_molecular_ecp_derivative_route

    refuse_molecular_ecp_derivative_route(
        mol,
        basis,
        result=rks_result,
        route="compute_hessian_rks_analytic",
    )
    if basis_name is None:
        raise ValueError(
            "compute_hessian_rks_analytic requires basis_name explicitly")
    if functional is None:
        functional = rks_result.functional
    if hessian_options is None:
        hessian_options = HessianFDOptions()
    if grid_options is None:
        grid_options = GridOptions()
    if eri is None:
        eri = np.asarray(compute_eri(basis), dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)

    if not rks_result.converged:
        raise ValueError(
            "compute_hessian_rks_analytic: RKSResult is not converged.")

    func = Functional(functional, spin=1)
    alpha_hf = func.hf_exchange_fraction

    # Reference data
    C = np.asarray(rks_result.mo_coeffs, dtype=np.float64)
    eps = np.asarray(rks_result.mo_energies, dtype=np.float64)
    D = np.asarray(rks_result.density, dtype=np.float64)
    n_basis = C.shape[0]
    n_occ = int(round(np.sum(np.linalg.eigvalsh(D) > 1e-8)))
    C_occ = C[:, :n_occ]
    e_i = eps[:n_occ]
    W = 2.0 * C_occ @ np.diag(e_i) @ C_occ.T

    n_atoms = len(mol.atoms)
    Ndof = 3 * n_atoms

    # ---- 1. Skeleton via FD on the analytic gradient ----------------
    # This includes XC + 1e + 2e + nuclear all at once at fixed
    # reference density. Cost: 6N gradient evaluations.
    H = _ks_skeleton_via_fd_gradient(mol, basis, basis_name, rks_result,
                                       grid_options, step_bohr=fd_step_bohr)

    # ---- 2. h1ao + s1ao via FD on F_KS / S -------------------------
    h1ao = _build_ks_h1ao_fd(mol, basis_name, D, functional, alpha_hf,
                              grid_options, step_bohr=fd_step_bohr)
    from .hessian_analytic import _build_s1ao_fd
    s1ao = _build_s1ao_fd(mol, basis_name, step_bohr=fd_step_bohr)

    # ---- 3. KS CPHF -------------------------------------------------
    mo1, mo_e1 = _solve_cphf_rks_nuclear(
        rks_result, eri, h1ao, s1ao, functional, alpha_hf,
        grid_options, mol, basis, options=cphf_options)

    # ---- 4. Response part of Hessian -------------------------------
    for A in range(n_atoms):
        for da in range(3):
            for B in range(n_atoms):
                for db in range(3):
                    dm1_B_db = mo1[B, db] @ C_occ.T
                    eps_dm1_B_db = (mo1[B, db] * e_i[None, :]) @ C_occ.T
                    s1oo_A_da = C_occ.T @ s1ao[A, da] @ C_occ

                    t1 = 4.0 * np.einsum("uv,vu->",
                                           h1ao[A, da], dm1_B_db)
                    t2 = -4.0 * np.einsum("uv,vu->",
                                            s1ao[A, da], eps_dm1_B_db)
                    t3 = -2.0 * np.einsum("ij,ji->",
                                            s1oo_A_da, mo_e1[B, db])
                    H[3*A + da, 3*B + db] += t1 + t2 + t3

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


__all__ = ["compute_hessian_rks_analytic"]
