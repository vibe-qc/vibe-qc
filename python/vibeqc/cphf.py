"""Phase 17b-1 -- Coupled-Perturbed Hartree-Fock (CPHF) kernel for RHF.

Solves the linear-response equation

.. math::
    \\mathbf{A} \\, \\mathbf{U}^x = -\\mathbf{B}^x

where :math:`\\mathbf{U}^x` is the orbital-rotation amplitude in the
occupied-virtual block (response of the MOs to perturbation *x*),
:math:`\\mathbf{A}` is the orbital Hessian (closed-shell, real,
spin-restricted) and :math:`\\mathbf{B}^x` is the "RHS" describing
how perturbation *x* couples to occupied-virtual orbital pairs.

The orbital Hessian acts as

.. math::
    [\\mathbf{A} \\, \\mathbf{v}]_{ai} = (\\varepsilon_a - \\varepsilon_i) v_{ai}
        + \\sum_{bj} \\big[ 4(ai|bj) - (ab|ij) - (aj|ib) \\big] v_{bj}

(the factor 4 / -1 / -1 is the closed-shell-RHF combination of the
exchange + Coulomb response). We never form :math:`\\mathbf{A}`
explicitly. Each matrix-vector product builds a Fock-like contribution
in AO basis (:math:`G[D^v]` with :math:`D^v_{\\mu\\nu} = 2 \\sum_{ai}
v_{ai} (C_{\\mu a} C_{\\nu i} + C_{\\mu i} C_{\\nu a})`, factor 2
because the closed-shell amplitudes describe both spins) and
transforms back to the MO occ-vir block. Cost per iteration: one
:math:`O(N^4)` Fock build, identical to one SCF iteration.

The CG solver uses the diagonal :math:`(\\varepsilon_a -
\\varepsilon_i)^{-1}` as a preconditioner -- the orbital Hessian is
strongly diagonal-dominant for non-pathological systems, so a few
tens of iterations is typical.

API
---
::

    from vibeqc import Atom, Molecule, BasisSet, RHFOptions, run_rhf
    from vibeqc import compute_eri, compute_dipole
    from vibeqc.cphf import cphf_solve_rhf, dipole_polarizability_rhf

    mol = Molecule.from_xyz("water.xyz")
    basis = BasisSet(mol, "6-31g*")
    result = run_rhf(mol, basis, RHFOptions(conv_tol_grad=1e-9))
    alpha = dipole_polarizability_rhf(result, basis, mol)
    print(alpha)   # (3, 3) tensor in atomic units (e^2.a₀^2/E_h)

The first arg of :func:`cphf_solve_rhf` is the AO-basis RHS matrix
(or a list of them). Returns the orbital-response amplitudes ``U``
in occ-vir MO basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    Molecule,
    RHFResult,
    compute_dipole,
    compute_eri,
)
from .properties import center_of_mass


# ----------------------------------------------------------------------
# Public dataclasses
# ----------------------------------------------------------------------

@dataclass
class CPHFOptions:
    """Knobs for :func:`cphf_solve_rhf`.

    max_iter
        Conjugate-gradient iteration cap. Most cases converge in
        20-60 iterations on default basis sets; increase if you see
        ``CPHFConvergenceError``.

    tol
        Relative residual tolerance ``||r|| < tol.||rhs||``. Default
        1e-8 -- tight enough for analytic Hessian work, loose enough
        that SCF residual noise doesn't dominate.

    use_preconditioner
        When True (default), use :math:`(\\varepsilon_a -
        \\varepsilon_i)^{-1}` as a CG preconditioner. Disable only
        for diagnostics -- convergence is much slower without it.
    """
    max_iter: int = 100
    tol: float = 1e-8
    use_preconditioner: bool = True


class CPHFConvergenceError(RuntimeError):
    """Raised when the CG solver fails to converge to ``options.tol``
    within ``options.max_iter`` iterations."""


# ----------------------------------------------------------------------
# Internal helpers -- orbital-Hessian matrix-vector product
# ----------------------------------------------------------------------

def _ao_density_from_ov(v_ia: np.ndarray, C_occ: np.ndarray,
                        C_vir: np.ndarray) -> np.ndarray:
    """Form the symmetric AO-basis "response density"
        D^v_{muν} = S_{ia} v_{ia} (C_mua C_νi + C_mui C_νa)

    No extra spin factor: the closed-shell orbital-Hessian
    coefficients ``4 (ai|jb) - (ab|ij) - (aj|ib)`` already absorb
    the doubly-occupied spin algebra. The factor "2" prefactor on
    G(D^v) in the orbital-Hessian action gives the correct
    combination "2 . (J - 1/2K).D^v = 4 J - K" (in MO occ-vir).

    Indexing convention: ``v_ia[i, a]`` (occ is row, vir is column).
    """
    # T_{mui} = S_a v_{ia} C_mua = (C_vir . v_ia^T)_{mui}
    T = C_vir @ v_ia.T                       # (n_basis, n_occ)
    # D = T C_occ^T + C_occ T^T  (no leading factor of 2)
    D = T @ C_occ.T
    return D + D.T


def _ao_density_antisym_from_ov(v_ia: np.ndarray, C_occ: np.ndarray,
                                C_vir: np.ndarray) -> np.ndarray:
    """Form the *antisymmetric* AO-basis "response density"
        D^v_{muν} = S_{ia} v_{ia} (C_mua C_νi - C_mui C_νa)

    This is the density-like matrix the ``(A - B)`` orbital-Hessian
    action couples to. Its Coulomb build J[D] is identically zero
    (ERI symmetric in ls, D antisymmetric), so only the exchange
    build contributes -- see :func:`_orbital_hessian_minus_action`.

    Indexing convention: ``v_ia[i, a]`` (occ row, vir column),
    matching :func:`_ao_density_from_ov`.
    """
    T = C_vir @ v_ia.T                       # (n_basis, n_occ)
    D = T @ C_occ.T
    return D - D.T


def _ov_block(M_ao: np.ndarray, C_occ: np.ndarray,
              C_vir: np.ndarray) -> np.ndarray:
    """Transform an AO-basis matrix into the occupied-virtual block of
    MO basis. Returns shape ``(n_occ, n_vir)`` in ``[i, a]`` indexing
    (occ row, vir column) -- matches the convention used everywhere
    else in this module.
    """
    return C_occ.T @ M_ao @ C_vir


def _build_g_ao(D_ao: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """Closed-shell Fock-like contribution from a (possibly non-RHF)
    density-like matrix:
        G[D] = J[D] - (1/2) K[D]
        J[D]_{muν} = S_{ls} (muν|ls) D_{ls}
        K[D]_{muν} = S_{ls} (mul|νs) D_{ls}
    Implemented as numpy einsums on the stored 4-index ERI tensor.
    """
    # J: contract ls on D against the right two indices of (muν|ls).
    J = np.einsum("uvls,ls->uv", eri, D_ao)
    # K: contract mu->l swap. (mul|νs) = eri[mu,l,ν,s]; contract ls on D:
    # K_{muν} = S_{ls} (mul|νs) D_{ls}
    K = np.einsum("ulvs,ls->uv", eri, D_ao)
    return J - 0.5 * K


def _build_k_ao(D_ao: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """Plain exchange build (no Coulomb, no 1/2 factor):
        K[D]_{muν} = S_{ls} (mul|νs) D_{ls}

    Used by the ``(A - B)`` orbital-Hessian action. For an
    *antisymmetric* density-like matrix the Coulomb build J[D]
    vanishes identically (the ERI is symmetric under l<->s while D is
    not), so ``(A - B)``'s two-electron part is pure exchange.
    """
    return np.einsum("ulvs,ls->uv", eri, D_ao)


def _orbital_hessian_action(v_ia: np.ndarray, C_occ: np.ndarray,
                            C_vir: np.ndarray, eps_diff_inv: np.ndarray,
                            eri: np.ndarray) -> np.ndarray:
    """Apply the closed-shell RHF orbital Hessian A to a vector
    v_{ia} in MO occ-vir basis. Returns A.v with the same
    ``(n_occ, n_vir)`` shape and ``[i, a]`` indexing.

    A.v decomposes into:
        diag: (e_a - e_i) v_{ia}
        2-electron: 2.G[D^v]^{ov} in the same MO occ-vir basis.

    The "2." prefactor and "+J - (1/2)K" structure together give the
    closed-shell spin-restricted "4 J - K - (exchange-mix)" combination
    of the textbook orbital Hessian.
    """
    # eps_diff_inv has shape (n_occ, n_vir): 1/(e_a - e_i)
    # Diagonal piece: (e_a - e_i) v_{ia} = v_ia / eps_diff_inv.
    diag_part = v_ia / eps_diff_inv
    # 2-electron piece via Fock-like AO build.
    D_v = _ao_density_from_ov(v_ia, C_occ, C_vir)
    G_v = _build_g_ao(D_v, eri)
    g_part_ia = 2.0 * _ov_block(G_v, C_occ, C_vir)
    return diag_part + g_part_ia


def _orbital_hessian_minus_action(v_ia: np.ndarray, C_occ: np.ndarray,
                                  C_vir: np.ndarray,
                                  eps_diff_inv: np.ndarray,
                                  eri: np.ndarray) -> np.ndarray:
    """Apply the closed-shell RHF ``(A - B)`` operator to a vector
    ``v_{ia}`` in MO occ-vir basis. Returns ``(A - B).v`` with the
    same ``(n_occ, n_vir)`` shape and ``[i, a]`` indexing.

    In TD-HF notation the closed-shell singlet super-matrices are

        A_{ia,jb} = (e_a - e_i) d_ij d_ab + 2(ia|jb) - (ij|ab)
        B_{ia,jb} = 2(ia|jb) - (ib|ja)

    so

        (A - B)_{ia,jb} = (e_a - e_i) d_ij d_ab - (ij|ab) + (ib|ja).

    The two-electron part ``(ib|ja) - (ij|ab)`` is exactly the
    occ-vir block of the plain exchange build ``K`` evaluated on the
    *antisymmetric* response density (Coulomb cancels) -- see
    :func:`_ao_density_antisym_from_ov` / :func:`_build_k_ao`. No
    prefactor: the doubly-occupied spin algebra of the closed-shell
    amplitudes is already absorbed, exactly as for ``(A + B)`` in
    :func:`_orbital_hessian_action`.

    ``(A - B)`` is symmetric positive-definite whenever the
    closed-shell RHF reference is stable; it is diagonal for a pure
    (HF-exchange-free) functional and picks up the exchange terms
    above for HF / hybrids.
    """
    diag_part = v_ia / eps_diff_inv
    D_anti = _ao_density_antisym_from_ov(v_ia, C_occ, C_vir)
    K_anti = _build_k_ao(D_anti, eri)
    twoel_part = _ov_block(K_anti, C_occ, C_vir)
    return diag_part + twoel_part


# ----------------------------------------------------------------------
# Conjugate-gradient solver (preconditioned)
# ----------------------------------------------------------------------

def _pcg(matvec, rhs, precond, tol: float, max_iter: int):
    """Single-RHS preconditioned conjugate gradient. Solves
    ``matvec(x) = rhs`` with the symmetric matvec implied by matvec.

    Returns (x, n_iter, converged).
    """
    rhs_norm = np.linalg.norm(rhs)
    if rhs_norm == 0:
        return np.zeros_like(rhs), 0, True

    x = np.zeros_like(rhs)
    r = rhs.copy()           # r0 = b - A.x0 = b
    z = precond(r)
    p = z.copy()
    rz = float(np.vdot(r.ravel(), z.ravel()).real)

    for k in range(1, max_iter + 1):
        Ap = matvec(p)
        denom = float(np.vdot(p.ravel(), Ap.ravel()).real)
        if denom <= 0:
            # Non-PD orbital Hessian (instability or near-degeneracy).
            # In practice this shouldn't happen for converged RHF on
            # closed-shell systems; bail out gracefully.
            raise CPHFConvergenceError(
                f"CPHF: orbital Hessian non-positive at CG iter {k} "
                f"(p^T A p = {denom:.3e}). Likely an RHF instability."
            )
        alpha = rz / denom
        x = x + alpha * p
        r = r - alpha * Ap
        if np.linalg.norm(r) < tol * rhs_norm:
            return x, k, True
        z = precond(r)
        rz_new = float(np.vdot(r.ravel(), z.ravel()).real)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return x, max_iter, False


# ----------------------------------------------------------------------
# Public driver: cphf_solve_rhf
# ----------------------------------------------------------------------

def cphf_solve_rhf(
    rhs_ao,
    rhf_result: RHFResult,
    eri: np.ndarray,
    *,
    options: Optional[CPHFOptions] = None,
) -> np.ndarray:
    """Solve the closed-shell RHF CPHF equations for one or more
    AO-basis perturbations.

    Each perturbation matrix ``M_ao`` (shape ``(n_basis, n_basis)``)
    is transformed to the occupied-virtual block in MO basis to form
    the right-hand side ``B_{ai} = (C_vir^T . M_ao . C_occ)_{ai}``,
    then the CPHF equations
    :math:`\\mathbf{A}\\,\\mathbf{U} = -\\mathbf{B}` are solved by
    preconditioned CG.

    Parameters
    ----------
    rhs_ao :
        Either a single ``(n_basis, n_basis)`` matrix or a sequence
        of them (e.g. shape ``(n_rhs, n_basis, n_basis)``). For
        polarizability the three components are the dipole-integral
        matrices in the three Cartesian directions.
    rhf_result :
        Converged :class:`vibeqc.RHFResult` providing MO coefficients
        and energies.
    eri :
        AO-basis 4-index ERI tensor (e.g. from
        :func:`vibeqc.compute_eri`).
    options :
        :class:`CPHFOptions` for solver knobs.

    Returns
    -------
    np.ndarray
        Orbital-response amplitudes ``U[n_rhs, n_occ, n_vir]`` (or
        ``U[n_occ, n_vir]`` for a single RHS).

    Raises
    ------
    CPHFConvergenceError
        If the CG solver fails to converge to ``options.tol`` in
        ``options.max_iter`` iterations.
    """
    if options is None:
        options = CPHFOptions()

    # --- MO data ---------------------------------------------------------
    C = np.asarray(rhf_result.mo_coeffs, dtype=np.float64)
    eps = np.asarray(rhf_result.mo_energies, dtype=np.float64)
    if not rhf_result.converged:
        raise ValueError(
            "cphf_solve_rhf: RHFResult is not converged. CPHF requires "
            "a stationary HF reference."
        )
    n_basis = C.shape[0]
    # n_occ = (electron count) / 2. Read from the density matrix's trace
    # against the overlap, but RHFResult doesn't expose n_occ directly --
    # infer from the column shape and the eps-vs-occ structure. The
    # cleanest invariant is "number of orbitals with negative occupation
    # eigenvalue", but vibe-qc closed-shell results have no fractional
    # occupations, so use a plain sign-of-density approach -- if the
    # density is 2.C_occ.C_occ^T then trace(D.S) = 2.n_occ. We pull n_occ
    # from the result.density rather than re-computing it.
    D = np.asarray(rhf_result.density, dtype=np.float64)
    # trace(D . S) = 2 n_occ. Easier: use the canonical-MO orthonormality --
    # for closed-shell RHF with frozen-HF density, the rank of D/2 is n_occ.
    # Eigenvalues of D are {2, 2, ..., 2, 0, 0, ..., 0} so n_occ = round(sum(D_diag)/2).
    # Actually that's trace(D)/2 only if S is identity -- which it is in
    # an orthonormal basis but not in AO. Punt: take MO eigenvalues as the
    # signal -- those below the HOMO-LUMO gap. The cheapest approach is to
    # demand the caller knows n_occ from molecule.n_electrons // 2; we do
    # that by inspecting how the density was built. Cleaner: we ask the
    # user to pass it. For now, re-derive from the eigenvalue spectrum of
    # the density matrix in AO basis.
    n_occ = _infer_n_occ(D, C)

    C_occ = C[:, :n_occ]
    C_vir = C[:, n_occ:]
    n_vir = C_vir.shape[1]
    eps_occ = eps[:n_occ]
    eps_vir = eps[n_occ:]
    # eps_diff has shape (n_occ, n_vir): e_a - e_i with a=virtual, i=occ
    eps_diff = eps_vir[None, :] - eps_occ[:, None]
    if np.any(eps_diff <= 0):
        raise ValueError(
            "cphf_solve_rhf: orbital-energy gap non-positive -- the RHF "
            "reference is unstable (HOMO above LUMO)."
        )
    eps_diff_inv = 1.0 / eps_diff

    eri_arr = np.asarray(eri, dtype=np.float64)

    def _matvec(v_ov):
        return _orbital_hessian_action(v_ov, C_occ, C_vir, eps_diff_inv,
                                        eri_arr)

    if options.use_preconditioner:
        def _precond(r):
            return r * eps_diff_inv
    else:
        def _precond(r):
            return r

    # --- Process inputs --------------------------------------------------
    rhs_arr = np.asarray(rhs_ao, dtype=np.float64)
    single = rhs_arr.ndim == 2
    if single:
        rhs_list = [rhs_arr]
    else:
        if rhs_arr.ndim != 3:
            raise ValueError(
                f"cphf_solve_rhf: rhs_ao must be 2D ({n_basis}, {n_basis})"
                f" or 3D (n_rhs, {n_basis}, {n_basis}); got shape {rhs_arr.shape}"
            )
        rhs_list = [rhs_arr[k] for k in range(rhs_arr.shape[0])]

    # --- Solve per RHS ---------------------------------------------------
    U_list = []
    for k, M_ao in enumerate(rhs_list):
        if M_ao.shape != (n_basis, n_basis):
            raise ValueError(
                f"cphf_solve_rhf: RHS {k} has shape {M_ao.shape}, "
                f"expected ({n_basis}, {n_basis})"
            )
        # MO occ-vir block of the RHS, sign flipped: A.U = -B.
        # Shape (n_occ, n_vir), [i, a] indexing.
        B = -_ov_block(M_ao, C_occ, C_vir)
        x, n_iter, converged = _pcg(_matvec, B, _precond,
                                     options.tol, options.max_iter)
        if not converged:
            raise CPHFConvergenceError(
                f"cphf_solve_rhf: CG did not converge for RHS {k} "
                f"in {options.max_iter} iterations "
                f"(residual ||r|| / ||b|| = {np.linalg.norm(_matvec(x) - B) / np.linalg.norm(B):.3e})."
            )
        U_list.append(x)

    if single:
        return U_list[0]
    return np.stack(U_list, axis=0)


# ----------------------------------------------------------------------
# MO setup shared by the static and dynamic solvers
# ----------------------------------------------------------------------

def _cphf_mo_setup(rhf_result: RHFResult):
    """Return ``(C_occ, C_vir, eps_diff, eps_diff_inv, n_occ, n_vir)``
    for a converged closed-shell RHF reference. Raises ValueError on a
    non-stationary or HOMO-above-LUMO reference -- both solvers need a
    stable closed-shell RHF.
    """
    if not rhf_result.converged:
        raise ValueError(
            "cphf: RHFResult is not converged -- CPHF / dynamic response "
            "require a stationary HF reference."
        )
    C = np.asarray(rhf_result.mo_coeffs, dtype=np.float64)
    eps = np.asarray(rhf_result.mo_energies, dtype=np.float64)
    D = np.asarray(rhf_result.density, dtype=np.float64)
    n_occ = _infer_n_occ(D, C)
    C_occ = C[:, :n_occ]
    C_vir = C[:, n_occ:]
    n_vir = C_vir.shape[1]
    eps_diff = eps[n_occ:][None, :] - eps[:n_occ][:, None]  # (n_occ, n_vir)
    if np.any(eps_diff <= 0):
        raise ValueError(
            "cphf: orbital-energy gap non-positive -- the RHF reference "
            "is unstable (HOMO above LUMO)."
        )
    return C_occ, C_vir, eps_diff, 1.0 / eps_diff, n_occ, n_vir


# ----------------------------------------------------------------------
# Dynamic (imaginary-frequency) CPHF / TD-HF linear response
# ----------------------------------------------------------------------

def cphf_solve_dynamic_rhf(
    rhs_ao,
    rhf_result: RHFResult,
    eri: np.ndarray,
    omega: float,
    *,
    options: Optional[CPHFOptions] = None,
) -> np.ndarray:
    """Solve the **imaginary-frequency** closed-shell linear-response
    (TD-HF) equations for one or more AO-basis perturbations.

    The frequency-dependent TD-HF response to a real, symmetric
    one-electron perturbation ``P`` is, with ``U₊ = X + Y`` the
    symmetric response amplitude,

    .. math::
        \\big[ (A + B) + \\omega^2 (A - B)^{-1} \\big] \\, U_+ = -2 P

    evaluated on the **imaginary** frequency axis ``iw`` (``omega`` is
    the real magnitude ``w >= 0``). On the imaginary axis the shifted
    operator is real and symmetric positive-definite -- there are no
    poles -- which is exactly why D3/D4 reference dispersion data is
    built there.

    At ``omega == 0`` this reduces to ``(A + B) U_+ = -2 P``; note the
    factor-2 RHS relative to :func:`cphf_solve_rhf` (which solves
    ``(A + B) U = -P``), so ``U_+ == 2 . U_static`` in the static
    limit. :func:`dynamic_polarizability_rhf` absorbs the factor in
    its contraction prefactor so ``a(i0)`` equals the static
    :func:`dipole_polarizability_rhf` tensor.

    Solver: outer preconditioned CG on the ``w^2``-shifted operator;
    each application of ``(A - B)^{-1}`` is an inner preconditioned CG
    solve against ``(A - B)`` (positive-definite for a stable
    closed-shell RHF). For ``omega == 0`` the inner solve is skipped.

    Parameters
    ----------
    rhs_ao :
        A single ``(n_basis, n_basis)`` perturbation matrix or a
        sequence / 3D stack of them.
    rhf_result :
        Converged closed-shell :class:`RHFResult`.
    eri :
        AO-basis 4-index ERI tensor.
    omega :
        Imaginary-frequency magnitude ``w >= 0`` (atomic units). The
        physical frequency is ``iw``.
    options :
        :class:`CPHFOptions` solver knobs (shared by the inner and
        outer CG).

    Returns
    -------
    np.ndarray
        Symmetric response amplitudes ``U_+`` with shape
        ``(n_rhs, n_occ, n_vir)`` (or ``(n_occ, n_vir)`` for a single
        RHS).
    """
    if options is None:
        options = CPHFOptions()
    omega = float(omega)
    if omega < 0:
        raise ValueError(
            f"cphf_solve_dynamic_rhf: omega must be >= 0 (imaginary-axis "
            f"magnitude); got {omega}"
        )

    C_occ, C_vir, eps_diff, eps_diff_inv, n_occ, n_vir = _cphf_mo_setup(
        rhf_result)
    n_basis = C_occ.shape[0]
    eri_arr = np.asarray(eri, dtype=np.float64)

    def _matvec_plus(v):
        return _orbital_hessian_action(v, C_occ, C_vir, eps_diff_inv, eri_arr)

    def _matvec_minus(v):
        return _orbital_hessian_minus_action(v, C_occ, C_vir, eps_diff_inv,
                                             eri_arr)

    omega_sq = omega * omega

    # Preconditioners. The inner (A-B) solve uses the orbital-energy-
    # difference diagonal 1/(e_a-e_i) -- (A-B) is strongly diagonal-
    # dominant. The *outer* operator is (A+B) + w^2(A-B)⁻¹; its
    # diagonal is ≈ (e_a-e_i) + w^2/(e_a-e_i), so the matching
    # preconditioner is the inverse of that w-shifted diagonal.
    # Using the plain 1/(e_a-e_i) outer preconditioner badly
    # under-damps the w^2(A-B)⁻¹ part and stalls the outer CG at high
    # w (the integrand tail of the Casimir-Polder grid). The shifted
    # preconditioner reduces to the static one at w=0.
    if options.use_preconditioner:
        outer_precond_diag = eps_diff + omega_sq * eps_diff_inv

        def _precond_inner(r):
            return r * eps_diff_inv

        def _precond_outer(r):
            return r / outer_precond_diag
    else:
        def _precond_inner(r):
            return r

        def _precond_outer(r):
            return r

    # Inner solve tolerance: tighter than the outer so the shifted
    # operator is effectively exact and the outer CG doesn't stall on
    # an inconsistent matvec.
    inner_tol = max(options.tol * 1.0e-2, 1.0e-12)

    def _shifted_op(v):
        out = _matvec_plus(v)
        if omega_sq != 0.0:
            w, _, conv = _pcg(_matvec_minus, v, _precond_inner,
                              inner_tol, options.max_iter)
            if not conv:
                raise CPHFConvergenceError(
                    "cphf_solve_dynamic_rhf: inner (A-B) CG did not "
                    f"converge in {options.max_iter} iterations."
                )
            out = out + omega_sq * w
        return out

    rhs_arr = np.asarray(rhs_ao, dtype=np.float64)
    single = rhs_arr.ndim == 2
    if single:
        rhs_list = [rhs_arr]
    else:
        if rhs_arr.ndim != 3:
            raise ValueError(
                f"cphf_solve_dynamic_rhf: rhs_ao must be 2D "
                f"({n_basis}, {n_basis}) or 3D (n_rhs, {n_basis}, "
                f"{n_basis}); got shape {rhs_arr.shape}"
            )
        rhs_list = [rhs_arr[k] for k in range(rhs_arr.shape[0])]

    U_list = []
    for k, M_ao in enumerate(rhs_list):
        if M_ao.shape != (n_basis, n_basis):
            raise ValueError(
                f"cphf_solve_dynamic_rhf: RHS {k} has shape "
                f"{M_ao.shape}, expected ({n_basis}, {n_basis})"
            )
        # RHS is -2 P^{ov} (the factor 2 is the U₊ = X+Y convention).
        rhs = -2.0 * _ov_block(M_ao, C_occ, C_vir)
        x, _, converged = _pcg(_shifted_op, rhs, _precond_outer,
                               options.tol, options.max_iter)
        if not converged:
            raise CPHFConvergenceError(
                f"cphf_solve_dynamic_rhf: outer CG did not converge for "
                f"RHS {k} at omega={omega:.6g} in {options.max_iter} "
                f"iterations (||r||/||b|| = "
                f"{np.linalg.norm(_shifted_op(x) - rhs) / np.linalg.norm(rhs):.3e})."
            )
        U_list.append(x)

    if single:
        return U_list[0]
    return np.stack(U_list, axis=0)


# ----------------------------------------------------------------------
# Application: static dipole polarizability
# ----------------------------------------------------------------------

def dipole_polarizability_rhf(
    rhf_result: RHFResult,
    basis: BasisSet,
    molecule: Molecule,
    *,
    eri: Optional[np.ndarray] = None,
    origin: Optional[Sequence[float]] = None,
    options: Optional[CPHFOptions] = None,
) -> np.ndarray:
    """Static (frequency-zero) dipole polarizability tensor
    :math:`\\alpha_{\\alpha\\beta}` from CPHF.

    .. math::
        \\alpha_{\\alpha\\beta} = -2 \\sum_{ai} U^{\\alpha}_{ai} \\, \\mu_{\\beta, ai}

    where :math:`\\mu_{\\alpha, ai}` is the AO->MO occ-vir block of
    the dipole integral in direction :math:`\\alpha` and
    :math:`U^{\\alpha}_{ai}` solves the CPHF equations with the
    a-direction dipole as the RHS. The factor 2 is for closed-shell
    spin and the negative sign comes from the electronic charge
    convention (electrons have ``Q = -1`` in atomic units).

    Returns
    -------
    np.ndarray
        Symmetric ``(3, 3)`` polarizability tensor in atomic units
        (``e^2.a₀^2/E_h`` = bohr^3 when expressed via dimensional
        analysis of the field-induced dipole). Multiply by 0.14818
        to get Å^3.
    """
    if origin is None:
        origin_vec = center_of_mass(molecule)
    else:
        origin_vec = np.asarray(origin, dtype=np.float64)
    dip = compute_dipole(basis, [float(x) for x in origin_vec])
    M = np.stack([np.asarray(dip.x), np.asarray(dip.y),
                   np.asarray(dip.z)], axis=0)  # (3, n_bf, n_bf)
    # The dipole-perturbation Hamiltonian for an electron is (-r), but
    # the CPHF "RHS" is conventionally mu itself (without the sign);
    # the sign convention is reabsorbed into the polarizability formula
    # below.
    if eri is None:
        eri = np.asarray(compute_eri(basis), dtype=np.float64)
    U = cphf_solve_rhf(M, rhf_result, eri, options=options)  # (3, n_occ, n_vir)

    # Build dipole occ-vir block in MO basis for the contraction.
    # Shape (n_occ, n_vir), [i, a] indexing.
    C = np.asarray(rhf_result.mo_coeffs, dtype=np.float64)
    n_occ = _infer_n_occ(np.asarray(rhf_result.density, dtype=np.float64), C)
    C_occ = C[:, :n_occ]
    C_vir = C[:, n_occ:]
    mu_mo = np.zeros((3, n_occ, C_vir.shape[1]))
    for axis in range(3):
        mu_mo[axis] = C_occ.T @ M[axis] @ C_vir
    # a_ab = -4 S_ia U^b_ia . mu_a,ia  (closed-shell convention).
    # The factor 4 = 2 (closed-shell spin in P = 2 C_occ C_occ^T)
    # x 2 (each orbital contributes via both <dpsi|r|psi> and <psi|r|dpsi>
    # in the dipole expectation, for real orbitals).
    alpha = -4.0 * np.einsum("xia,yia->xy", U, mu_mo)
    # Symmetrize (the response is symmetric for a static field; tiny
    # residual asymmetry is FD/CG noise).
    return 0.5 * (alpha + alpha.T)


# ----------------------------------------------------------------------
# Application: imaginary-frequency dipole polarizability a(iw)
# ----------------------------------------------------------------------

def dynamic_polarizability_rhf(
    rhf_result: RHFResult,
    basis: BasisSet,
    molecule: Molecule,
    frequencies,
    *,
    eri: Optional[np.ndarray] = None,
    origin: Optional[Sequence[float]] = None,
    options: Optional[CPHFOptions] = None,
) -> np.ndarray:
    """Imaginary-frequency dipole polarizability :math:`\\alpha(i\\omega)`.

    Evaluates the dynamic dipole polarizability tensor on the
    imaginary frequency axis,

    .. math::
        \\alpha_{\\alpha\\beta}(i\\omega)
            = -2 \\sum_{ia} U_+^{\\beta}(i\\omega)_{ia}\\;\\mu_{\\alpha,ia}

    where :math:`U_+^{\\beta}(i\\omega)` solves the imaginary-frequency
    TD-HF response equations (see :func:`cphf_solve_dynamic_rhf`).

    This is the quantity the D4 dispersion model needs: the reference
    :math:`C_6` coefficients come from the Casimir-Polder integral
    :math:`C_6 = \\tfrac{3}{\\pi}\\int_0^\\infty \\alpha_A(i\\omega)\\,
    \\alpha_B(i\\omega)\\,d\\omega`, and :math:`\\alpha(i\\omega)` is
    smooth, real and monotone-decaying there (no poles) -- milestone
    D4b-1 of ``handovers/HANDOVER_D4_NATIVE.md``.

    Parameters
    ----------
    rhf_result, basis, molecule :
        Converged closed-shell RHF reference and its system.
    frequencies :
        A scalar imaginary-frequency magnitude :math:`\\omega \\ge 0`,
        or a 1-D sequence of them (atomic units).
    eri :
        Optional precomputed AO 4-index ERI tensor (built once and
        reused across all frequencies when omitted).
    origin :
        Gauge origin for the dipole integrals; defaults to the centre
        of mass (the polarizability is origin-independent for a
        neutral closed-shell molecule, so this only matters for ions).
    options :
        :class:`CPHFOptions` solver knobs.

    Returns
    -------
    np.ndarray
        ``(3, 3)`` tensor for a scalar ``frequencies``; otherwise an
        ``(n_freq, 3, 3)`` stack, one tensor per frequency. Atomic
        units (bohr^3). :math:`\\alpha(i0)` reproduces the static
        :func:`dipole_polarizability_rhf` tensor.
    """
    scalar_input = np.isscalar(frequencies)
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if np.any(freqs < 0):
        raise ValueError(
            "dynamic_polarizability_rhf: imaginary-frequency magnitudes "
            "must be >= 0."
        )

    if origin is None:
        origin_vec = center_of_mass(molecule)
    else:
        origin_vec = np.asarray(origin, dtype=np.float64)
    dip = compute_dipole(basis, [float(x) for x in origin_vec])
    M = np.stack([np.asarray(dip.x), np.asarray(dip.y),
                   np.asarray(dip.z)], axis=0)  # (3, n_bf, n_bf)

    if eri is None:
        eri = np.asarray(compute_eri(basis), dtype=np.float64)

    # Dipole occ-vir MO block, shared across all frequencies.
    C = np.asarray(rhf_result.mo_coeffs, dtype=np.float64)
    n_occ = _infer_n_occ(np.asarray(rhf_result.density, dtype=np.float64), C)
    C_occ = C[:, :n_occ]
    C_vir = C[:, n_occ:]
    mu_mo = np.zeros((3, n_occ, C_vir.shape[1]))
    for axis in range(3):
        mu_mo[axis] = C_occ.T @ M[axis] @ C_vir

    out = np.zeros((freqs.size, 3, 3))
    for f, omega in enumerate(freqs):
        U_plus = cphf_solve_dynamic_rhf(M, rhf_result, eri, omega,
                                        options=options)  # (3, n_occ, n_vir)
        # a_ab(iw) = -2 S U₊^b . mu_a. The U₊ = X+Y convention carries
        # the factor 2 relative to the static U; combined with the -2
        # here it reproduces the static -4 prefactor at w = 0.
        alpha = -2.0 * np.einsum("xia,yia->xy", U_plus, mu_mo)
        out[f] = 0.5 * (alpha + alpha.T)

    return out[0] if scalar_input else out


# ----------------------------------------------------------------------
# n_occ inference
# ----------------------------------------------------------------------

def _infer_n_occ(D_ao: np.ndarray, C: np.ndarray) -> int:
    """Recover the number of doubly-occupied orbitals from the closed-
    shell density matrix and the MO coefficients.

    The MO-basis density matrix is ``C^T . S . D . S . C`` for a
    non-orthonormal AO basis, but we only have the AO Fock and
    density at hand. A cheaper trick: in the *MO basis*, the closed-
    shell RHF density matrix is ``diag(2, 2, ..., 2, 0, ..., 0)`` so
    looking at ``(C^T D)`` and counting columns with non-trivial
    inner product with C_occ would work, but again needs S.

    The simplest robust signal: the density's trace ``Tr(D)`` equals
    twice the integrated density (number of electrons) only when AOs
    are orthonormal. In general we'd need ``Tr(D . S)`` = N.

    For a converged RHF solution though, ``D = 2 . C_occ . C_occ^T``,
    so ``D . C = 2 . C_occ . (C_occ^T . C)`` and this projects onto
    the occupied subspace. ``np.linalg.matrix_rank`` of ``D`` gives
    n_occ since D has rank n_occ exactly (the AO ranks aren't reduced
    by C being non-square here; D's rank is set by the rank of the
    occupied projection).

    Use rank-of-density.
    """
    # Eigenvalues of D are {2, 2, ..., 2, 0, ..., 0} in an orthonormal
    # basis; in AO basis they're scaled but the rank is preserved.
    eigvals = np.linalg.eigvalsh(D_ao)
    # Most singular values are 0; n_occ are nonzero positive.
    threshold = 1e-8 * eigvals.max() if eigvals.max() > 0 else 1e-10
    return int(np.sum(eigvals > threshold))


__all__ = [
    "CPHFOptions",
    "CPHFConvergenceError",
    "cphf_solve_rhf",
    "cphf_solve_dynamic_rhf",
    "dipole_polarizability_rhf",
    "dynamic_polarizability_rhf",
]
