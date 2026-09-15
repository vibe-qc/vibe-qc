"""Phase 17e -- analytic UKS Hessian assembly.

Open-shell-DFT extension of :func:`vibeqc.compute_hessian_uhf_analytic`
(Phase 17c) and :func:`vibeqc.compute_hessian_rks_analytic` (Phase
17d). Combines:

  - **Per-spin CPHF** like 17c (UHF) -- a and b densities, two
    coupled mo1 blocks, LGMRES on the stacked vir-occ vectors.
  - **libxc fxc kernel response** like 17d (RKS) -- extends fvind
    with the spin-resolved second-order XC kernel:

      v_xc_resp_a = ∫ chi_mu chi_ν . [f_xc^aa . r_resp_a
                                  + f_xc^ab . r_resp_b] . w dr
      v_xc_resp_b = ∫ chi_mu chi_ν . [f_xc^ab . r_resp_a
                                  + f_xc^bb . r_resp_b] . w dr

    where ``f_xc^ss'`` comes from libxc's polarized fxc evaluation. LDA
    uses the three rho-rho pieces; GGA additionally contracts the full
    5x5 kernel in (rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb).

  - **Skeleton via FD on the analytic UKS gradient** at fixed
    reference UKSResult (avoids 2nd-deriv AO + 2nd-deriv Becke
    machinery, matches the 17d trick).

Hybrid functionals (a_HF.K) work unchanged through the inherited
17c K-derivative skeleton + a_HF-scaled exchange in fvind. Pure
functionals (LDA, PBE, BLYP) take a_HF=0.

The factor 2 (vs RHF's 4) is the spin-restricted convention: each
spin contributes its own factor 2 (for "+c.c."), summed over both
spins. Same response-Hessian assembly as 17c.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    Functional,
    GridOptions,
    Molecule,
    UKSResult,
    XCKind,
    build_coulomb,
    build_exchange,
    build_grid,
    compute_eri,
    compute_gradient_uks,
    compute_kinetic,
    compute_nuclear,
    evaluate_ao_with_gradient,
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
# Internal -- UKS Fock-at-geometry (no SCF) for FD h1ao construction
# ----------------------------------------------------------------------

def _uks_fock_at_geometry(mol: Molecule, basis_name: str,
                            D_alpha: np.ndarray, D_beta: np.ndarray,
                            functional_name: str, alpha_hf: float,
                            grid_options: GridOptions):
    """UKS per-spin Fock matrices F_a, F_b at the given geometry with
    the supplied densities held fixed (no SCF). For LDA::

        F_s = T + V + J(D_a + D_b) - a_HF . K(D_s) + V_xc_s

    with V_xc_s_muν = ∫ chi_mu(r) v_r_s(r) chi_ν(r) w(r) dr from libxc's
    spin-polarized evaluation.

    For GGA we additionally pick up gradchi-coupled terms via v_s_ss'.
    """
    basis = BasisSet(mol, basis_name)
    T = np.asarray(compute_kinetic(basis))
    V = np.asarray(compute_nuclear(basis, mol))
    Hcore = T + V

    eri = np.asarray(compute_eri(basis))
    D_total = D_alpha + D_beta
    J = np.asarray(build_coulomb(eri, D_total))
    if alpha_hf != 0.0:
        K_a = np.asarray(build_exchange(eri, D_alpha))
        K_b = np.asarray(build_exchange(eri, D_beta))
        F_a_jk = J - alpha_hf * K_a
        F_b_jk = J - alpha_hf * K_b
    else:
        F_a_jk = J.copy()
        F_b_jk = J.copy()

    # XC contribution: build V_xc_a and V_xc_b via libxc polarized eval.
    grid = build_grid(mol, grid_options)
    chi, dchi_x, dchi_y, dchi_z = evaluate_ao_with_gradient(basis, grid.points)
    chi = np.asarray(chi)
    weights = np.asarray(grid.weights)

    chiD_a = chi @ D_alpha
    chiD_b = chi @ D_beta
    rho_a = (chiD_a * chi).sum(axis=1)
    rho_b = (chiD_b * chi).sum(axis=1)

    func = Functional(functional_name, spin=2)
    is_gga = (func.kind == XCKind.GGA)
    if is_gga:
        dchi = [np.asarray(dchi_x), np.asarray(dchi_y), np.asarray(dchi_z)]
        gax = 2.0 * (chiD_a * dchi[0]).sum(axis=1)
        gay = 2.0 * (chiD_a * dchi[1]).sum(axis=1)
        gaz = 2.0 * (chiD_a * dchi[2]).sum(axis=1)
        gbx = 2.0 * (chiD_b * dchi[0]).sum(axis=1)
        gby = 2.0 * (chiD_b * dchi[1]).sum(axis=1)
        gbz = 2.0 * (chiD_b * dchi[2]).sum(axis=1)
        sigma_aa = gax**2 + gay**2 + gaz**2
        sigma_ab = gax*gbx + gay*gby + gaz*gbz
        sigma_bb = gbx**2 + gby**2 + gbz**2
    else:
        sigma_aa = np.zeros_like(rho_a)
        sigma_ab = np.zeros_like(rho_a)
        sigma_bb = np.zeros_like(rho_a)
        dchi = None
        gax = gay = gaz = gbx = gby = gbz = None

    (exc, v_rho_a, v_rho_b,
     v_sigma_aa, v_sigma_ab, v_sigma_bb) = func.eval_polarised(
        rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb)

    # LDA piece, per spin: V_xc_s_muν = S_g w(g) v_r_s(g) chi_mu(g) chi_ν(g).
    w_va = weights * v_rho_a
    w_vb = weights * v_rho_b
    V_xc_a = chi.T @ (w_va[:, None] * chi)
    V_xc_b = chi.T @ (w_vb[:, None] * chi)

    if is_gga:
        # GGA s piece, per spin. Following the standard chain rule on
        # s_aa = |gradr_a|^2, s_ab = gradr_a.gradr_b:
        #   dE_xc/dD_s_muν |_GGA = 2.gradr_s . (2 v_s_ss gradchi_mu chi_ν + v_s_s¬s gradchi_mu chi_ν)
        # Per-spin "flow" vectors (matches the gradient-side helper):
        u_a = 2.0 * weights * v_sigma_aa
        u_ab = weights * v_sigma_ab          # cross-spin coupling
        u_b = 2.0 * weights * v_sigma_bb
        # F_grad_a on each grid point: 2.v_s_aa.gradr_a + v_s_ab.gradr_b.
        Fa_x = u_a[:, None] * np.zeros_like(chi)  # initialise correct shape
        # We actually want: each row is a single number times a gradient vector,
        # accumulated along Cartesian dims. Build flux x dchi/dR per row.
        flux_a = (u_a[:, None] * dchi[0]) * gax[:, None] \
                 + (u_a[:, None] * dchi[1]) * gay[:, None] \
                 + (u_a[:, None] * dchi[2]) * gaz[:, None]
        flux_a += (u_ab[:, None] * dchi[0]) * gbx[:, None] \
                  + (u_ab[:, None] * dchi[1]) * gby[:, None] \
                  + (u_ab[:, None] * dchi[2]) * gbz[:, None]
        flux_b = (u_b[:, None] * dchi[0]) * gbx[:, None] \
                 + (u_b[:, None] * dchi[1]) * gby[:, None] \
                 + (u_b[:, None] * dchi[2]) * gbz[:, None]
        flux_b += (u_ab[:, None] * dchi[0]) * gax[:, None] \
                  + (u_ab[:, None] * dchi[1]) * gay[:, None] \
                  + (u_ab[:, None] * dchi[2]) * gaz[:, None]
        Fu_a = flux_a.T @ chi
        Fu_b = flux_b.T @ chi
        V_xc_a += Fu_a + Fu_a.T
        V_xc_b += Fu_b + Fu_b.T

    F_alpha = Hcore + F_a_jk + V_xc_a
    F_beta = Hcore + F_b_jk + V_xc_b
    return F_alpha, F_beta


def _build_uks_h1ao_fd(mol: Molecule, basis_name: str,
                        D_alpha: np.ndarray, D_beta: np.ndarray,
                        functional_name: str, alpha_hf: float,
                        grid_options: GridOptions,
                        step_bohr: float = 1e-4):
    """Per-spin AO Fock first-derivative tensors via FD on
    :func:`_uks_fock_at_geometry` at fixed reference (D_a, D_b).

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
            Fa_p, Fb_p = _uks_fock_at_geometry(
                mol_p, basis_name, D_alpha, D_beta,
                functional_name, alpha_hf, grid_options)
            Fa_m, Fb_m = _uks_fock_at_geometry(
                mol_m, basis_name, D_alpha, D_beta,
                functional_name, alpha_hf, grid_options)
            h1a[A, d] = (Fa_p - Fa_m) / (2.0 * step_bohr)
            h1b[A, d] = (Fb_p - Fb_m) / (2.0 * step_bohr)
    return h1a, h1b


# ----------------------------------------------------------------------
# Skeleton via FD on compute_gradient_uks at fixed reference UKSResult
# ----------------------------------------------------------------------

def _uks_skeleton_via_fd_gradient(
    mol: Molecule, basis: BasisSet, basis_name: str,
    uks_result: UKSResult, grid_options: GridOptions,
    step_bohr: float = 1e-4) -> np.ndarray:
    """(3N, 3N) skeleton Hessian via central-difference of the analytic
    UKS gradient at fixed reference ``uks_result``. Same trick as 17d
    for RKS -- the reference C, e, D_a, D_b are held fixed numerically
    while the basis + grid + AOs at displaced geometries reflect the
    new R. Captures all geometry-dependent skeleton pieces (1-e, 2-e,
    XC, nuclear-repulsion) at fixed density by construction.
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
            g_p = np.asarray(compute_gradient_uks(
                mol_p, basis_p, uks_result, grid_options))
            g_m = np.asarray(compute_gradient_uks(
                mol_m, basis_m, uks_result, grid_options))
            col = (g_p - g_m).reshape(-1) / (2.0 * step_bohr)
            H_skel[:, 3 * A + d] = col
    return 0.5 * (H_skel + H_skel.T)


# ----------------------------------------------------------------------
# UKS CPHF -- port of pyscf/scf/ucphf.solve_withs1 + RKS XC kernel
# ----------------------------------------------------------------------

def _solve_cphf_uks_nuclear(uks_result: UKSResult,
                              eri: np.ndarray,
                              h1ao_a: np.ndarray,
                              h1ao_b: np.ndarray,
                              s1ao: np.ndarray,
                              functional_name: str,
                              alpha_hf: float,
                              grid_options: GridOptions,
                              mol: Molecule,
                              basis: BasisSet,
                              options: Optional[CPHFOptions] = None):
    """Solve the UKS CPHF/CPKS equations for all nuclear-coordinate
    perturbations. Mirrors :func:`_solve_cphf_uhf_nuclear` (17c) and
    extends ``fvind`` with the spin-resolved libxc fxc kernel response.
    """
    if options is None:
        options = CPHFOptions()
    max_iter = int(options.max_iter)
    tol = float(options.tol)

    if not uks_result.converged:
        raise ValueError("hessian_analytic_uks: UKSResult is not converged.")

    Ca = np.asarray(uks_result.mo_coeffs_alpha, dtype=np.float64)
    Cb = np.asarray(uks_result.mo_coeffs_beta, dtype=np.float64)
    eps_a_full = np.asarray(uks_result.mo_energies_alpha, dtype=np.float64)
    eps_b_full = np.asarray(uks_result.mo_energies_beta, dtype=np.float64)
    Da = np.asarray(uks_result.density_alpha, dtype=np.float64)
    Db = np.asarray(uks_result.density_beta, dtype=np.float64)
    n_basis = Ca.shape[0]
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

    # Pre-compute grid + AO values + reference rho_a, rho_b + libxc fxc
    # at the reference geometry. Re-used every fvind call.
    grid = build_grid(mol, grid_options)
    chi_g, dchi_x_g, dchi_y_g, dchi_z_g = evaluate_ao_with_gradient(
        basis, grid.points)
    chi_g = np.asarray(chi_g)
    dchi_g = [
        np.asarray(dchi_x_g),
        np.asarray(dchi_y_g),
        np.asarray(dchi_z_g),
    ]
    weights = np.asarray(grid.weights)

    chi_Da = chi_g @ Da
    chi_Db = chi_g @ Db
    rho_ref_a = (chi_Da * chi_g).sum(axis=1)
    rho_ref_b = (chi_Db * chi_g).sum(axis=1)
    grad_ref_a = [
        2.0 * (chi_Da * dchi_g[dim]).sum(axis=1)
        for dim in range(3)
    ]
    grad_ref_b = [
        2.0 * (chi_Db * dchi_g[dim]).sum(axis=1)
        for dim in range(3)
    ]
    sigma_aa = sum(g * g for g in grad_ref_a)
    sigma_ab = sum(ga * gb for ga, gb in zip(grad_ref_a, grad_ref_b))
    sigma_bb = sum(g * g for g in grad_ref_b)
    func = Functional(functional_name, spin=2)
    if func.kind == XCKind.MGGA:
        raise RuntimeError(
            "hessian_analytic_uks: meta-GGA CPKS kernels require tau/laplacian "
            "second derivatives and are not implemented.")
    (_, _, _,
     v_sigma_aa, v_sigma_ab, v_sigma_bb) = func.eval_polarised(
        rho_ref_a, rho_ref_b, sigma_aa, sigma_ab, sigma_bb)
    fxc = func.eval_polarised_gga_fxc(
        rho_ref_a, rho_ref_b, sigma_aa, sigma_ab, sigma_bb)

    def _rho_and_grad_response(D_resp: np.ndarray):
        chi_D = chi_g @ D_resp
        rho = (chi_D * chi_g).sum(axis=1)
        grad = [
            2.0 * (chi_D * dchi_g[dim]).sum(axis=1)
            for dim in range(3)
        ]
        return rho, grad

    def _dot_cart(lhs, rhs):
        return sum(a * b for a, b in zip(lhs, rhs))

    # Transform h1ao + s1ao to MO basis.
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

    mo1_a = hs_a.copy()
    mo1_a[:, n_occ_a:, :] *= -e_ai_a[None, :, :]
    mo1_a[:, :n_occ_a, :] = -0.5 * s1mo_a[:, :n_occ_a, :]
    mo1_b = hs_b.copy()
    mo1_b[:, n_occ_b:, :] *= -e_ai_b[None, :, :]
    mo1_b[:, :n_occ_b, :] = -0.5 * s1mo_b[:, :n_occ_b, :]

    def _xc_kernel_response_ao(D_resp_a: np.ndarray, D_resp_b: np.ndarray):
        """Spin-resolved LDA/GGA XC kernel response.

        Returns ``(V_xc_resp_a, V_xc_resp_b)`` where
            V_xc_resp_s[mu,ν] = d(V_xc_s[mu,ν]) / dD_resp

        evaluated with the libxc polarized GGA variables
        ``(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb)``.
        """
        rho_resp_a, grad_resp_a = _rho_and_grad_response(D_resp_a)
        rho_resp_b, grad_resp_b = _rho_and_grad_response(D_resp_b)

        dsigma_aa = 2.0 * _dot_cart(grad_ref_a, grad_resp_a)
        dsigma_ab = (
            _dot_cart(grad_resp_a, grad_ref_b)
            + _dot_cart(grad_ref_a, grad_resp_b)
        )
        dsigma_bb = 2.0 * _dot_cart(grad_ref_b, grad_resp_b)

        dv_rho_a = (
            fxc["v2rho2_aa"] * rho_resp_a
            + fxc["v2rho2_ab"] * rho_resp_b
            + fxc["v2rhosigma_a_aa"] * dsigma_aa
            + fxc["v2rhosigma_a_ab"] * dsigma_ab
            + fxc["v2rhosigma_a_bb"] * dsigma_bb
        )
        dv_rho_b = (
            fxc["v2rho2_ab"] * rho_resp_a
            + fxc["v2rho2_bb"] * rho_resp_b
            + fxc["v2rhosigma_b_aa"] * dsigma_aa
            + fxc["v2rhosigma_b_ab"] * dsigma_ab
            + fxc["v2rhosigma_b_bb"] * dsigma_bb
        )
        v_a = chi_g.T @ ((weights * dv_rho_a)[:, None] * chi_g)
        v_b = chi_g.T @ ((weights * dv_rho_b)[:, None] * chi_g)

        if func.kind == XCKind.GGA:
            dv_sigma_aa = (
                fxc["v2rhosigma_a_aa"] * rho_resp_a
                + fxc["v2rhosigma_b_aa"] * rho_resp_b
                + fxc["v2sigma2_aa_aa"] * dsigma_aa
                + fxc["v2sigma2_aa_ab"] * dsigma_ab
                + fxc["v2sigma2_aa_bb"] * dsigma_bb
            )
            dv_sigma_ab = (
                fxc["v2rhosigma_a_ab"] * rho_resp_a
                + fxc["v2rhosigma_b_ab"] * rho_resp_b
                + fxc["v2sigma2_aa_ab"] * dsigma_aa
                + fxc["v2sigma2_ab_ab"] * dsigma_ab
                + fxc["v2sigma2_ab_bb"] * dsigma_bb
            )
            dv_sigma_bb = (
                fxc["v2rhosigma_a_bb"] * rho_resp_a
                + fxc["v2rhosigma_b_bb"] * rho_resp_b
                + fxc["v2sigma2_aa_bb"] * dsigma_aa
                + fxc["v2sigma2_ab_bb"] * dsigma_ab
                + fxc["v2sigma2_bb_bb"] * dsigma_bb
            )

            def _weighted_flux(field):
                return sum(
                    (weights * field[dim])[:, None] * dchi_g[dim]
                    for dim in range(3)
                )

            delta_flux_a = [
                2.0 * dv_sigma_aa * grad_ref_a[dim]
                + 2.0 * v_sigma_aa * grad_resp_a[dim]
                + dv_sigma_ab * grad_ref_b[dim]
                + v_sigma_ab * grad_resp_b[dim]
                for dim in range(3)
            ]
            delta_flux_b = [
                2.0 * dv_sigma_bb * grad_ref_b[dim]
                + 2.0 * v_sigma_bb * grad_resp_b[dim]
                + dv_sigma_ab * grad_ref_a[dim]
                + v_sigma_ab * grad_resp_a[dim]
                for dim in range(3)
            ]
            fu_a = _weighted_flux(delta_flux_a).T @ chi_g
            fu_b = _weighted_flux(delta_flux_b).T @ chi_g
            v_a = v_a + fu_a + fu_a.T
            v_b = v_b + fu_b + fu_b.T

        return v_a, v_b

    def fvind(m_a, m_b):
        v_a = np.zeros_like(m_a)
        v_b = np.zeros_like(m_b)
        for k in range(m_a.shape[0]):
            # Per-spin AO response density (no factor 2 -- UKS convention,
            # each spin is its own one-particle density).
            dm_a = Ca @ m_a[k] @ Ca_occ.T
            dm_a = dm_a + dm_a.T
            dm_b = Cb @ m_b[k] @ Cb_occ.T
            dm_b = dm_b + dm_b.T
            # 2-e Fock-like response.
            J_ao = np.asarray(build_coulomb(eri_arr, dm_a + dm_b))
            if alpha_hf != 0.0:
                K_a_ao = np.asarray(build_exchange(eri_arr, dm_a))
                K_b_ao = np.asarray(build_exchange(eri_arr, dm_b))
                v_a_ao = J_ao - alpha_hf * K_a_ao
                v_b_ao = J_ao - alpha_hf * K_b_ao
            else:
                v_a_ao = J_ao.copy()
                v_b_ao = J_ao.copy()
            # XC kernel response.
            v_xc_a, v_xc_b = _xc_kernel_response_ao(dm_a, dm_b)
            v_a_ao += v_xc_a
            v_b_ao += v_xc_b
            v_a[k] = Ca.T @ v_a_ao @ Ca_occ
            v_b[k] = Cb.T @ v_b_ao @ Cb_occ
        return v_a, v_b

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
            f"hessian_analytic_uks: LGMRES UCPKS did not converge "
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

    # Orbital-energy response per spin.
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
    for A_idx in range(n_atoms):
        for d in range(3):
            mo1_a_ao[A_idx, d] = Ca @ mo1_a[A_idx, d]
            mo1_b_ao[A_idx, d] = Cb @ mo1_b[A_idx, d]
    return mo1_a_ao, mo1_b_ao, mo_e1_a, mo_e1_b


# ----------------------------------------------------------------------
# Public driver
# ----------------------------------------------------------------------

def compute_hessian_uks_analytic(
    mol: Molecule,
    basis: BasisSet,
    uks_result: UKSResult,
    *,
    basis_name: Optional[str] = None,
    functional: Optional[str] = None,
    grid_options: Optional[GridOptions] = None,
    eri: Optional[np.ndarray] = None,
    fd_step_bohr: float = 1e-4,
    cphf_options: Optional[CPHFOptions] = None,
    hessian_options: Optional[HessianFDOptions] = None,
) -> HessianResult:
    """Compute the open-shell-UKS analytic Hessian.

    Skeleton via FD on the analytic UKS gradient at fixed reference
    density (so the XC and a_HF.K skeleton pieces come out naturally
    without 2nd-deriv AO grids -- matches the 17d trick). Response
    part via per-spin UCPKS with libxc spin-polarized fxc kernel.

    The CPKS kernel contracts the full spin-polarized LDA/GGA libxc
    fxc response. Hybrid functionals work via a_HF-scaled K-derivative
    skeleton and a_HF-scaled exchange in fvind.
    """
    from .ecp_metadata import refuse_molecular_ecp_derivative_route

    refuse_molecular_ecp_derivative_route(
        mol,
        basis,
        result=uks_result,
        route="compute_hessian_uks_analytic",
    )
    if basis_name is None:
        raise ValueError(
            "compute_hessian_uks_analytic requires basis_name explicitly")
    if functional is None:
        functional = uks_result.functional
    if hessian_options is None:
        hessian_options = HessianFDOptions()
    if grid_options is None:
        grid_options = GridOptions()
    if eri is None:
        eri = np.asarray(compute_eri(basis), dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)

    if not uks_result.converged:
        raise ValueError(
            "compute_hessian_uks_analytic: UKSResult is not converged.")

    func = Functional(functional, spin=2)
    alpha_hf = func.hf_exchange_fraction

    # Reference data
    Ca = np.asarray(uks_result.mo_coeffs_alpha, dtype=np.float64)
    Cb = np.asarray(uks_result.mo_coeffs_beta, dtype=np.float64)
    eps_a = np.asarray(uks_result.mo_energies_alpha, dtype=np.float64)
    eps_b = np.asarray(uks_result.mo_energies_beta, dtype=np.float64)
    Da = np.asarray(uks_result.density_alpha, dtype=np.float64)
    Db = np.asarray(uks_result.density_beta, dtype=np.float64)
    n_basis = Ca.shape[0]
    n_occ_a = int(round(np.sum(np.linalg.eigvalsh(Da) > 1e-8)))
    n_occ_b = int(round(np.sum(np.linalg.eigvalsh(Db) > 1e-8)))
    Ca_occ = Ca[:, :n_occ_a]
    Cb_occ = Cb[:, :n_occ_b]
    e_i_a = eps_a[:n_occ_a]
    e_i_b = eps_b[:n_occ_b]

    n_atoms = len(mol.atoms)
    Ndof = 3 * n_atoms

    # ---- 1. Skeleton via FD on analytic UKS gradient ---------------
    H = _uks_skeleton_via_fd_gradient(
        mol, basis, basis_name, uks_result, grid_options,
        step_bohr=fd_step_bohr,
    )

    # ---- 2. Per-spin h1ao + shared s1ao via FD ---------------------
    h1a, h1b = _build_uks_h1ao_fd(
        mol, basis_name, Da, Db, functional, alpha_hf,
        grid_options, step_bohr=fd_step_bohr,
    )
    from .hessian_analytic import _build_s1ao_fd
    s1ao = _build_s1ao_fd(mol, basis_name, step_bohr=fd_step_bohr)

    # ---- 3. UKS CPHF/CPKS ------------------------------------------
    mo1a, mo1b, mo_e1a, mo_e1b = _solve_cphf_uks_nuclear(
        uks_result, eri, h1a, h1b, s1ao, functional, alpha_hf,
        grid_options, mol, basis, options=cphf_options,
    )

    # ---- 4. Response part of Hessian (same as 17c UHF) -------------
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


__all__ = ["compute_hessian_uks_analytic"]
