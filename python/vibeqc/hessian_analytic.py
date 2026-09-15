"""Phase 17b-3 -- analytic RHF Hessian assembly.

Combines the skeleton 2nd-derivative integral contractions
(:mod:`vibeqc.hessian_integrals` -- Phase 17b-2) with the CPHF
orbital-rotation amplitudes (:func:`vibeqc.cphf_solve_rhf` -- Phase
17b-1) to produce the exact closed-shell-RHF Hessian without any
finite-difference of total energies or full SCFs.

Hessian formula (Yamaguchi-Osamura-Goddard-Schaefer / Pople-
Krishnan-Schlegel-Binkley convention)::

    H_{Aa,Bb} =  d^2E_nuc/dR_AadR_Bb
              +  S_muν D_muν . d^2(T+V)_muν/dR_AadR_Bb                   (skeleton 1-e)
              +  S_muνls Γ . d^2(muν|ls)_muνls/dR_AadR_Bb                  (skeleton 2-e)
              -  S_muν W_muν . d^2S_muν/dR_AadR_Bb                          (skeleton overlap)
              +  4 . tr(h1ao[A] . dm1[B])                              (response, h1)
              -  4 . tr(s1ao[A] . (mocc.e.mo1[B].T))                   (response, S^x)
              -  2 . tr(s1oo[A] . mo_e1[B])                            (response, e^x)

where ``mo1[B] = dC_occ/dR_Bb`` (CPHF amplitudes in MO basis,
expanded back to AOxocc shape) and ``mo_e1[B] = de/dR_Bb`` (orbital-
energy response).

V1 implementation: ``h1ao[A, d]`` (the AO Fock first derivative when
atom A moves along direction d) is computed by central differences
on the AO Fock matrix at *fixed* AO density. This costs 6N Fock
builds per Hessian -- much cheaper than the 6N full SCFs of the
:func:`vibeqc.compute_hessian_fd` path (Phase 17a-1) but still
finite-difference. A future optimization will swap in libint
``deriv_order=1`` integrals contracted with shell-slice density to
get a true analytic ``h1ao`` (~3-5x additional speedup).

Cost comparison (small molecule, doubly-occupied, no DFT):

==============   ==========================  ====================
 Path             Per-Hessian Fock-equiv.    Speedup vs 17a-1
==============   ==========================  ====================
 17a-1 (FD-E)      6N . 20  =  120N           1x
 17b-3 (this)      6N + 30 + (deriv=2 sweep)  ~20x for medium molecules
 17b-3-opt         (deriv=1 + deriv=2 sweep)  ~50x for large molecules
==============   ==========================  ====================
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._vibeqc_core import (
    Atom,
    BasisSet,
    GridOptions,
    Molecule,
    RHFOptions,
    RHFResult,
    build_fock_g,
    compute_dipole,
    compute_eri,
    compute_kinetic,
    compute_nuclear,
    compute_overlap,
    compute_eri_hessian_contribution,
    compute_kinetic_nuclear_hessian_contribution,
    compute_overlap_hessian_contribution,
    nuclear_repulsion_hessian,
    one_electron_gradient_contribution,
    overlap_gradient_contribution,
    two_electron_gradient_contribution,
    run_rhf,
)
from .cphf import CPHFOptions, _orbital_hessian_action, _ov_block, _pcg
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


# ----------------------------------------------------------------------
# Internal helpers -- h1ao via FD on the Fock matrix
# ----------------------------------------------------------------------

def _fock_at_geometry(mol: Molecule, basis_name: str, D: np.ndarray,
                      alpha_hf: float = 1.0) -> np.ndarray:
    """AO Fock matrix at the given geometry with the supplied density
    held fixed (no SCF). F = T + V + G(D) where G = J - 1/2K (closed
    shell)."""
    basis = BasisSet(mol, basis_name)
    T = np.asarray(compute_kinetic(basis))
    V = np.asarray(compute_nuclear(basis, mol))
    Hcore = T + V
    eri = np.asarray(compute_eri(basis))
    if alpha_hf == 1.0:
        G = np.asarray(build_fock_g(eri, D))   # J - 1/2K
    else:
        # Mixed J / K combination for hybrid DFT. Build pieces and
        # assemble by hand -- closed-shell convention has the 1/2 on K.
        from ._vibeqc_core import build_coulomb, build_exchange
        J = np.asarray(build_coulomb(eri, D))
        K = np.asarray(build_exchange(eri, D))
        G = J - 0.5 * alpha_hf * K
    return Hcore + G


def _displaced_molecule(mol: Molecule, atom_index: int,
                        cart: int, delta: float) -> Molecule:
    """Copy of ``mol`` with one atom shifted by ``delta`` along the
    given Cartesian axis. Mirrors the helper in :mod:`vibeqc.hessian`."""
    atoms = []
    for j, a in enumerate(mol.atoms):
        xyz = [a.xyz[0], a.xyz[1], a.xyz[2]]
        if j == atom_index:
            xyz[cart] += delta
        atoms.append(Atom(int(a.Z), xyz))
    return Molecule(atoms, charge=mol.charge, multiplicity=mol.multiplicity)


def _build_h1ao_fd(mol: Molecule, basis_name: str, D: np.ndarray,
                   alpha_hf: float = 1.0,
                   step_bohr: float = 1e-4) -> np.ndarray:
    """Build the AO Fock first-derivative tensor by central
    differences on the AO Fock matrix at fixed density.

    Returns ``(n_atoms, 3, n_basis, n_basis)``: ``h1ao[A, d, mu, ν] =
    dF_muν / dR_{A,d}`` evaluated with ``D`` held at its reference
    value.
    """
    n_atoms = len(mol.atoms)
    n_basis = D.shape[0]
    h1ao = np.zeros((n_atoms, 3, n_basis, n_basis), dtype=np.float64)
    for A in range(n_atoms):
        for d in range(3):
            mol_p = _displaced_molecule(mol, A, d, +step_bohr)
            mol_m = _displaced_molecule(mol, A, d, -step_bohr)
            F_p = _fock_at_geometry(mol_p, basis_name, D, alpha_hf)
            F_m = _fock_at_geometry(mol_m, basis_name, D, alpha_hf)
            h1ao[A, d] = (F_p - F_m) / (2.0 * step_bohr)
    return h1ao


def _build_s1ao_fd(mol: Molecule, basis_name: str,
                    step_bohr: float = 1e-4) -> np.ndarray:
    """Build the AO overlap first-derivative tensor by FD.

    Returns ``(n_atoms, 3, n_basis, n_basis)``: ``s1ao[A, d, mu, ν] =
    dS_muν / dR_{A,d}``. Same FD as for h1ao but cheaper (just overlap
    integrals, no SCF or eri required).
    """
    n_atoms = len(mol.atoms)
    basis = BasisSet(mol, basis_name)
    n_basis = basis.nbasis
    s1ao = np.zeros((n_atoms, 3, n_basis, n_basis), dtype=np.float64)
    for A in range(n_atoms):
        for d in range(3):
            mol_p = _displaced_molecule(mol, A, d, +step_bohr)
            mol_m = _displaced_molecule(mol, A, d, -step_bohr)
            S_p = np.asarray(compute_overlap(BasisSet(mol_p, basis_name)))
            S_m = np.asarray(compute_overlap(BasisSet(mol_m, basis_name)))
            s1ao[A, d] = (S_p - S_m) / (2.0 * step_bohr)
    return s1ao


# ----------------------------------------------------------------------
# CPHF for nuclear-coordinate perturbations (per atom)
# ----------------------------------------------------------------------

def _solve_cphf_nuclear(
    rhf_result: RHFResult,
    eri: np.ndarray,
    h1ao: np.ndarray,
    s1ao: np.ndarray,
    options: Optional[CPHFOptions] = None,
):
    """Solve CPHF for all nuclear-coordinate perturbations.

    Faithful port of :func:`pyscf.scf.cphf.solve_withs1`. The
    algorithm:

    1. Precompute ``hs = h1mo - s1mo . e_i`` in MO basis (full
       ``(n_mo, n_occ)`` shape, not just the occ-vir block).
    2. Initialise ``mo1``: virtual block <- ``-e_ai . hs[vir]`` (the
       leading-order static-perturbation guess); occ-occ block <-
       ``-1/2 s1_oo`` (forced by orthonormality of the perturbed
       orbitals).
    3. Fixed-point iteration on the virtual block:
       ``mo1[vir] = -e_ai . (hs[vir] - fvind(mo1)[vir])`` where
       ``fvind`` applies the orbital-Hessian operator (closed-shell
       ``J - 1/2K`` Fock-response) to the AO density built from the
       FULL ``mo1`` (occ-occ block stays at -1/2 s1_oo throughout).
    4. After convergence, refine: ``hs += fvind(mo1)`` and set
       ``mo1[vir] = hs[vir] / (e_i - e_a)``.
    5. ``mo_e1 = hs[occ] + mo1[occ] . (e_i - e_j)`` -- orbital-
       energy first derivative in the occ-occ block.

    Returns ``(mo1, mo_e1)``:

    - ``mo1[A, d, mu, i]`` -- first-order MO change dC[:, occ]/dR_{A, d}
      transformed back to AO basis (so ``dm1[A, d] =
      mo1[A, d] @ C_occ.T + transpose`` per closed-shell convention).
      Shape ``(n_atoms, 3, n_basis, n_occ)``.
    - ``mo_e1[A, d, i, j]`` -- orbital-energy response in the occ-occ
      block. Shape ``(n_atoms, 3, n_occ, n_occ)``.
    """
    if options is None:
        options = CPHFOptions()
    max_iter = int(options.max_iter)
    tol = float(options.tol)

    if not rhf_result.converged:
        raise ValueError("hessian_analytic: RHFResult is not converged.")

    C = np.asarray(rhf_result.mo_coeffs, dtype=np.float64)
    eps = np.asarray(rhf_result.mo_energies, dtype=np.float64)
    D_ao = np.asarray(rhf_result.density, dtype=np.float64)
    n_occ = int(round(np.sum(np.linalg.eigvalsh(D_ao) > 1e-8)))
    n_basis = C.shape[0]
    n_vir = n_basis - n_occ
    n_mo = n_basis
    C_occ = C[:, :n_occ]
    C_vir = C[:, n_occ:]
    e_i = eps[:n_occ]
    e_a = eps[n_occ:]
    # Per PySCF: e_ai = 1 / (e_a[:, None] - e_i)  shape (n_vir, n_occ)
    e_ai = 1.0 / (e_a[:, None] - e_i)

    eri_arr = np.asarray(eri, dtype=np.float64)
    n_atoms = h1ao.shape[0]
    n_pert = n_atoms * 3

    # ---- Transform h1ao + s1ao to MO basis (full n_mo x n_occ blocks) ----
    h1mo = np.zeros((n_pert, n_mo, n_occ), dtype=np.float64)
    s1mo = np.zeros((n_pert, n_mo, n_occ), dtype=np.float64)
    for k, (A, d) in enumerate(((A, d) for A in range(n_atoms) for d in range(3))):
        h1mo[k] = C.T @ h1ao[A, d] @ C_occ
        s1mo[k] = C.T @ s1ao[A, d] @ C_occ

    # hs = h1mo - s1mo . e_i. Shape (n_pert, n_mo, n_occ).
    hs = h1mo - s1mo * e_i[None, None, :]

    # Initial mo1: vir block from -e_ai * hs[vir]; occ block fixed at -1/2.s1_oo.
    mo1 = hs.copy()
    mo1[:, n_occ:, :] *= -e_ai[None, :, :]            # vir-occ block
    mo1[:, :n_occ, :] = -0.5 * s1mo[:, :n_occ, :]     # occ-occ block (orthonormality)

    # ---- fvind: orbital-Hessian operator on (n_pert, n_mo, n_occ) ----
    # Per PySCF gen_vind: dm1 = 2.C.mo1.C_occ^T + transpose, then v = G(dm1)
    # in AO, finally v_mo = C^T.v.C_occ. The factor 2 is the closed-shell
    # spin (each spatial orbital is doubly occupied).
    def fvind(mo1_arr):
        out = np.zeros_like(mo1_arr)
        for k in range(mo1_arr.shape[0]):
            dm = C @ (mo1_arr[k] * 2.0) @ C_occ.T
            dm = dm + dm.T
            v_ao = np.asarray(build_fock_g(eri_arr, dm))   # J - 1/2K
            out[k] = C.T @ v_ao @ C_occ
        return out

    # ---- Solve (I + L) x = mo1base[vir] via GMRES ----
    # PySCF uses lib.krylov with `(I + vind_vo) x = mo1base` semantics
    # where vind_vo applies +e_ai . fvind and zeros the occ block.
    # That's the linear equation
    #     x[vir] + e_ai . fvind(x_with_mo1_occ_fixed)[vir] = mo1base[vir]
    # We use scipy GMRES on the vir block; each matvec calls fvind once,
    # which is one Fock-build worth of work. Convergence is much faster
    # than naive fixed-point (which can stall at iteration ratio ~0.85
    # on small molecules with near-degenerate occupied/virtual gaps),
    # typically 5-20 matvecs per RHS.
    from scipy.sparse.linalg import gmres, LinearOperator

    n_vec_size = n_pert * n_vir * n_occ   # flattened (n_pert, n_vir, n_occ)

    # Reusable scratch buffer to feed into fvind (occ block fixed).
    mo1_scratch = mo1.copy()

    def matvec(x_flat):
        x = x_flat.reshape(n_pert, n_vir, n_occ)
        mo1_scratch[:, n_occ:, :] = x
        # mo1_scratch[:, :n_occ, :] stays at -1/2.s1_oo throughout
        v = fvind(mo1_scratch)
        # (I + L) x = x + e_ai . fvind(...)[vir]
        return (x + e_ai[None, :, :] * v[:, n_occ:, :]).ravel()

    A = LinearOperator((n_vec_size, n_vec_size), matvec=matvec,
                       dtype=np.float64)
    b = mo1[:, n_occ:, :].ravel().copy()
    x0 = mo1[:, n_occ:, :].ravel().copy()
    # Both rtol AND atol are critical: high-symmetry molecules (e.g. H2
    # along the bond axis) can have near-zero RHS, in which case the
    # default ``||r|| < rtol . ||b||`` criterion is unreachable. Setting
    # atol to the same tolerance switches to ``||r|| < max(atol, rtol.||b||)``.
    x_solution, info = gmres(A, b, x0=x0, rtol=tol, atol=tol,
                              maxiter=max_iter,
                              restart=min(50, n_vec_size))
    if info != 0:
        # info > 0: did not converge in maxiter iterations (info = number
        # of iterations actually performed). info < 0: illegal input.
        raise RuntimeError(
            f"hessian_analytic: GMRES CPHF did not converge "
            f"(info = {info}, max_iter = {max_iter}). Try increasing "
            f"options.max_iter or loosening options.tol."
        )
    mo1[:, n_occ:, :] = x_solution.reshape(n_pert, n_vir, n_occ)

    # ---- Final refinement (PySCF: hs += fvind(mo1); mo1[vir] = hs[vir]/(e_i - e_a)) ----
    hs_final = hs + fvind(mo1)
    mo1[:, n_occ:, :] = hs_final[:, n_occ:, :] / (e_i[None, None, :] - e_a[:, None][None, :, :])

    # ---- Orbital-energy response in occ-occ block ----
    # mo_e1[i, j] = hs_final[occ][i, j] + mo1[occ][i, j] . (e_i - e_j)
    e_i_diff = e_i[:, None] - e_i[None, :]   # (n_occ, n_occ): [i, j] = e_i - e_j
    mo_e1 = hs_final[:, :n_occ, :] + mo1[:, :n_occ, :] * e_i_diff[None, :, :]

    # Reshape to (n_atoms, 3, ...) for the assembler
    mo1_pert = mo1.reshape(n_atoms, 3, n_mo, n_occ)
    mo_e1_pert = mo_e1.reshape(n_atoms, 3, n_occ, n_occ)

    # PySCF stores mo1 in AO basis (mo1_AO = C @ mo1_MO) for the assembler.
    mo1_ao = np.zeros((n_atoms, 3, n_basis, n_occ), dtype=np.float64)
    for A in range(n_atoms):
        for d in range(3):
            mo1_ao[A, d] = C @ mo1_pert[A, d]

    return mo1_ao, mo_e1_pert


# ----------------------------------------------------------------------
# Public driver
# ----------------------------------------------------------------------

def compute_hessian_rhf_analytic(
    mol: Molecule,
    basis: BasisSet,
    rhf_result: RHFResult,
    *,
    basis_name: Optional[str] = None,
    eri: Optional[np.ndarray] = None,
    alpha_hf: float = 1.0,
    fd_step_bohr: float = 1e-4,
    cphf_options: Optional[CPHFOptions] = None,
    hessian_options: Optional[HessianFDOptions] = None,
) -> HessianResult:
    """Compute the closed-shell-RHF Hessian via analytic CPHF and
    libint deriv_order=2 skeleton + FD-on-Fock h1ao.

    Returns a :class:`vibeqc.HessianResult` with the same shape as
    :func:`vibeqc.compute_hessian_fd` (Phase 17a-1), so all the
    downstream consumers (frequencies, IR intensities, thermo) plug
    in unchanged.
    """
    from .ecp_metadata import refuse_molecular_ecp_derivative_route

    refuse_molecular_ecp_derivative_route(
        mol,
        basis,
        result=rhf_result,
        route="compute_hessian_rhf_analytic",
    )
    if basis_name is None:
        # Recover the basis-set name from the BasisSet object's repr.
        # In v0.5 we always have it from the caller.
        raise ValueError(
            "compute_hessian_rhf_analytic requires basis_name explicitly "
            "(needed for the FD-on-Fock displaced-geometry rebuilds)."
        )
    if hessian_options is None:
        hessian_options = HessianFDOptions()
    if eri is None:
        eri = np.asarray(compute_eri(basis), dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)

    # ---- Reference data ------------------------------------------------
    if not rhf_result.converged:
        raise ValueError(
            "compute_hessian_rhf_analytic: RHFResult is not converged.")

    C = np.asarray(rhf_result.mo_coeffs, dtype=np.float64)
    eps = np.asarray(rhf_result.mo_energies, dtype=np.float64)
    D = np.asarray(rhf_result.density, dtype=np.float64)
    n_basis = C.shape[0]
    n_occ = int(round(np.sum(np.linalg.eigvalsh(D) > 1e-8)))
    C_occ = C[:, :n_occ]
    eps_occ = eps[:n_occ]
    W = 2.0 * C_occ @ np.diag(eps_occ) @ C_occ.T   # energy-weighted density

    n_atoms = len(mol.atoms)
    Ndof = 3 * n_atoms

    # ---- 1. Skeleton (Phase 17b-2) -------------------------------------
    H = np.zeros((Ndof, Ndof), dtype=np.float64)
    H += np.asarray(nuclear_repulsion_hessian(mol))
    H += np.asarray(compute_kinetic_nuclear_hessian_contribution(basis, mol, D))
    H += np.asarray(compute_eri_hessian_contribution(basis, mol, D, alpha_hf))
    H += np.asarray(compute_overlap_hessian_contribution(basis, mol, W))

    # ---- 2. h1ao + s1ao via FD-on-Fock --------------------------------
    h1ao = _build_h1ao_fd(mol, basis_name, D, alpha_hf, step_bohr=fd_step_bohr)
    s1ao = _build_s1ao_fd(mol, basis_name, step_bohr=fd_step_bohr)

    # ---- 3. CPHF response (Phase 17b-1) -------------------------------
    mo1, mo_e1 = _solve_cphf_nuclear(rhf_result, eri, h1ao, s1ao,
                                     options=cphf_options)
    # mo1 shape (n_atoms, 3, n_basis, n_occ); mo_e1 (n_atoms, 3, n_occ, n_occ)

    # ---- 4. Response part of the Hessian ------------------------------
    # H_xy_response = + 4 . tr(h1ao[A] . dm1[B])
    #                - 4 . tr(s1ao[A] . e.dm1[B])
    #                - 2 . tr(s1oo[A] . mo_e1[B])
    # where dm1[B] = mo1[B] . C_occ^T  (closed-shell density response).
    for A in range(n_atoms):
        for da in range(3):
            for B in range(n_atoms):
                for db in range(3):
                    dm1_B_db = mo1[B, db] @ C_occ.T              # (n_basis, n_basis)
                    eps_dm1_B_db = mo1[B, db] * eps_occ[None, :]
                    eps_dm1_B_db = eps_dm1_B_db @ C_occ.T

                    # 4 . tr(h1ao[A,da] . dm1[B,db])
                    term1 = 4.0 * np.einsum(
                        "uv,vu->", h1ao[A, da], dm1_B_db)
                    # -4 . tr(s1ao[A,da] . e.dm1[B,db])
                    term2 = -4.0 * np.einsum(
                        "uv,vu->", s1ao[A, da], eps_dm1_B_db)
                    # -2 . tr(s1oo[A,da] . mo_e1[B,db])
                    s1oo_A_da = C_occ.T @ s1ao[A, da] @ C_occ
                    term3 = -2.0 * np.einsum(
                        "ij,ji->", s1oo_A_da, mo_e1[B, db])

                    H[3*A + da, 3*B + db] += term1 + term2 + term3

    # Symmetrize (the response part should be symmetric to FD precision;
    # any residual asymmetry is FD-truncation noise on h1ao and s1ao).
    H = 0.5 * (H + H.T)

    # ---- 5. Mass-weight + diagonalize (same as 17a-1) -----------------
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
        n_displacements=0,   # analytic -- no FD-on-energy displacements
        is_linear=is_linear,
        masses_amu=masses_amu,
    )


__all__ = ["compute_hessian_rhf_analytic"]
