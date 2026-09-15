"""MSINDO analytic nuclear gradient -- port of DEDXYZK/DEDXYZL (Phase 4).

Computes the full Cartesian nuclear gradient of the RHF MSINDO total energy
by contracting the SCF density matrix with per-pair (R,th,phi) integral
derivatives, using the chain rule to convert to Cartesian forces.

Fortran reference: dedxyzk.f, dedxyzl.f, intdxdydz.f.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import msindo_pair_deriv as _pd
from .msindo import (
    ANGSTROM_TO_BOHR,
    _MSINDO_ROOT_PROBE_ELEMENTS,
    _SUPPORTED_NDDO,
    _atom_blocks,
    _build_core_and_gamma,
    _nddo_fock_extra,
    _nddo_params,
    _scf_rhf,
    _scf_rhf_molecular,
    eff_core_charge,
    nddo_dspdd_si,
    nddo_dsppp_si,
    nddo_dspsp_pi,
    nddo_dspsp_si,
    nddo_dspss_si,
    nddo_spdd_si,
    nddo_sppp_si,
    nddo_spsp_pi,
    nddo_spsp_si,
    nddo_spss_si,
)


@lru_cache(maxsize=1)
def _cpp_gradient_kernel():
    try:
        from vibeqc._vibeqc_core.semiempirical import indo as _indo
    except ImportError:
        return None
    try:
        gradient_analytic = _indo.gradient_analytic
        load_params_from_json = _indo.load_params_from_json
    except AttributeError:
        return None
    params = load_params_from_json(
        Path(__file__).with_name("msindo_params.json").read_text()
    )
    return gradient_analytic, params


def _directional_integral_jacobian(value, radial_derivative, distance, direction):
    """Return ``d(value(R) * e_a) / d(C_j-C_i)_k``.

    The radial derivative routines return ``d integral / dR``. This helper adds
    the derivative of the molecular-frame direction cosine, exactly the split
    used by MSINDO ``dnddorhf.f``. Indices are ``(a, k)``.
    """
    direction = np.asarray(direction, dtype=float)
    ee = np.outer(direction, direction)
    return radial_derivative * ee + value * (np.eye(3) - ee) / distance


def _dipole_tensor_jacobian(
    sigma,
    pi,
    dsigma,
    dpi,
    distance,
    direction,
):
    """Return the s-p/s-p tensor and its Cartesian displacement derivative.

    ``S_ab = pi delta_ab + (sigma-pi) e_a e_b``. The returned derivative has
    indices ``(a, b, k)`` for ``dS_ab / d(C_j-C_i)_k``. This compact tensor form
    is algebraically the same as the 27 explicit components in
    ``dspsprhf.f`` and, crucially, differentiates the integral convention used
    by the shipped NDDO energy rather than copying its force signs.
    """
    direction = np.asarray(direction, dtype=float)
    identity = np.eye(3)
    ee = np.outer(direction, direction)
    tensor = pi * identity + (sigma - pi) * ee
    jacobian = np.empty((3, 3, 3), dtype=float)
    for k in range(3):
        de = (identity[:, k] - direction * direction[k]) / distance
        jacobian[:, :, k] = (
            dpi * direction[k] * identity
            + (dsigma - dpi) * direction[k] * ee
            + (sigma - pi)
            * (np.outer(de, direction) + np.outer(direction, de))
        )
    return tensor, jacobian


def _nddo_gradient_correction(atomic_numbers, coordinates_bohr, blocks, density):
    """Derivative terms present in the NDDO energy but absent from INDO.

    This contracts the RHF density with the NDDO HSP core term and with the
    dipole--monopole/dipole--dipole two-electron additions. Dewar--Thiel,
    Theor. Chim. Acta 46, 89 (1977), Eqs. 38--47 and 53--66 define the
    point-multipole energy. MSINDO ``dnddorhf.f`` and ``dspsprhf.f`` provide an
    independent source mapping. The result is ``dE/dR`` in Hartree/bohr; it is
    not a force. At a converged RHF density the orbital-response term vanishes,
    so the contraction holds the density fixed while differentiating every
    explicit geometry dependence of the exact shipped energy.
    """
    Z = list(atomic_numbers)
    C = np.asarray(coordinates_bohr, dtype=float)
    P = np.asarray(density, dtype=float)
    gradient = np.zeros((len(Z), 3), dtype=float)

    def fock_energy_derivative(row, column, derivative):
        # E_2 = 1/2 Tr[P F_2]. F_2 is assembled in the lower triangle and then
        # mirrored, so an off-diagonal derivative carries P[row, column], while
        # a diagonal derivative carries half that weight.
        weight = 0.5 if row == column else 1.0
        return weight * P[row, column] * derivative

    for i, zi in enumerate(Z):
        lo_i, hi_i = blocks[i]
        has_p_i = hi_i - lo_i >= 4
        has_d_i = hi_i - lo_i >= 9
        for j, zj in enumerate(Z):
            if i == j:
                continue
            lo_j, hi_j = blocks[j]
            has_p_j = hi_j - lo_j >= 4
            has_d_j = hi_j - lo_j >= 9
            displacement = C[j] - C[i]
            distance = float(np.linalg.norm(displacement))
            if distance < 1e-12:
                continue
            direction = displacement / distance
            pair_derivative = np.zeros(3, dtype=float)

            # Dipole-on-j integrals used against monopoles on i.
            sssp = nddo_spss_si(zj, zi, distance) if has_p_j else 0.0
            dsssp = nddo_dspss_si(zj, zi, distance) if has_p_j else 0.0
            d_sssp_e = (
                _directional_integral_jacobian(
                    sssp, dsssp, distance, direction
                )
                if has_p_j
                else np.zeros((3, 3), dtype=float)
            )
            ppsp = nddo_sppp_si(zj, zi, distance) if has_p_i and has_p_j else 0.0
            dppsp = (
                nddo_dsppp_si(zj, zi, distance) if has_p_i and has_p_j else 0.0
            )
            d_ppsp_e = (
                _directional_integral_jacobian(
                    ppsp, dppsp, distance, direction
                )
                if has_p_i and has_p_j
                else np.zeros((3, 3), dtype=float)
            )
            ddsp = nddo_spdd_si(zj, zi, distance) if has_p_j and has_d_i else 0.0
            dddsp = (
                nddo_dspdd_si(zj, zi, distance)
                if has_p_j and has_d_i
                else 0.0
            )
            d_ddsp_e = (
                _directional_integral_jacobian(
                    ddsp, dddsp, distance, direction
                )
                if has_p_j and has_d_i
                else np.zeros((3, 3), dtype=float)
            )

            # Dipole-on-i integrals used against monopoles on j.
            spss = nddo_spss_si(zi, zj, distance) if has_p_i else 0.0
            dspss = nddo_dspss_si(zi, zj, distance) if has_p_i else 0.0
            d_spss_e = (
                _directional_integral_jacobian(
                    spss, dspss, distance, direction
                )
                if has_p_i
                else np.zeros((3, 3), dtype=float)
            )
            sppp = nddo_sppp_si(zi, zj, distance) if has_p_i and has_p_j else 0.0
            dsppp = (
                nddo_dsppp_si(zi, zj, distance) if has_p_i and has_p_j else 0.0
            )
            d_sppp_e = (
                _directional_integral_jacobian(
                    sppp, dsppp, distance, direction
                )
                if has_p_i and has_p_j
                else np.zeros((3, 3), dtype=float)
            )
            spdd = nddo_spdd_si(zi, zj, distance) if has_p_i and has_d_j else 0.0
            dspdd = (
                nddo_dspdd_si(zi, zj, distance)
                if has_p_i and has_d_j
                else 0.0
            )
            d_spdd_e = (
                _directional_integral_jacobian(
                    spdd, dspdd, distance, direction
                )
                if has_p_i and has_d_j
                else np.zeros((3, 3), dtype=float)
            )

            if has_p_i:
                # HSP one-electron core attraction in _build_core_and_gamma.
                # Tr[P dH] gives a factor of two for the symmetric s/p pair.
                core_charge_j = float(eff_core_charge(zj))
                for a in range(3):
                    pair_derivative += (
                        -2.0
                        * core_charge_j
                        * P[lo_i + 1 + a, lo_i]
                        * d_spss_e[a]
                    )

            # The remaining updates differentiate _nddo_fock_extra term for
            # term. Keeping its lower-triangle convention makes the 1/2 Tr[PF]
            # factor explicit instead of relying on source-specific force signs.
            if has_p_j:
                p_dipole_j = P[lo_j + 1 : lo_j + 4, lo_j]
                derivative = -2.0 * (p_dipole_j @ d_sssp_e)
                pair_derivative += fock_energy_derivative(lo_i, lo_i, derivative)
                if has_p_i:
                    derivative = -2.0 * (p_dipole_j @ d_ppsp_e)
                    for a in range(3):
                        pair_derivative += fock_energy_derivative(
                            lo_i + 1 + a,
                            lo_i + 1 + a,
                            derivative,
                        )
                if has_d_i:
                    derivative = -2.0 * (p_dipole_j @ d_ddsp_e)
                    for a in range(5):
                        pair_derivative += fock_energy_derivative(
                            lo_i + 4 + a,
                            lo_i + 4 + a,
                            derivative,
                        )

            if has_p_i:
                pss_j = P[lo_j, lo_j]
                ppp_j = (
                    float(np.trace(P[lo_j + 1 : lo_j + 4, lo_j + 1 : lo_j + 4]))
                    if has_p_j
                    else 0.0
                )
                pdd_j = (
                    float(np.trace(P[lo_j + 4 : lo_j + 9, lo_j + 4 : lo_j + 9]))
                    if has_d_j
                    else 0.0
                )
                if has_p_i and has_p_j:
                    sigma = nddo_spsp_si(zj, zi, distance)
                    pi = nddo_spsp_pi(zj, zi, distance)
                    dsigma = nddo_dspsp_si(zj, zi, distance)
                    dpi = nddo_dspsp_pi(zj, zi, distance)
                    _s_tensor, d_s_tensor = _dipole_tensor_jacobian(
                        sigma,
                        pi,
                        dsigma,
                        dpi,
                        distance,
                        direction,
                    )
                    p_dipole_j = P[lo_j + 1 : lo_j + 4, lo_j]
                else:
                    d_s_tensor = np.zeros((3, 3, 3), dtype=float)
                    p_dipole_j = np.zeros(3, dtype=float)
                for a in range(3):
                    derivative = (
                        d_spss_e[a] * pss_j
                        + d_sppp_e[a] * ppp_j
                        + d_spdd_e[a] * pdd_j
                    )
                    if has_p_j:
                        derivative = derivative + 2.0 * np.tensordot(
                            p_dipole_j,
                            d_s_tensor[a],
                            axes=(0, 0),
                        )
                    pair_derivative += fock_energy_derivative(
                        lo_i + 1 + a,
                        lo_i,
                        derivative,
                    )

            if i > j:
                cross_ss = P[lo_i, lo_j]
                if has_p_i:
                    p_pi_sj = P[lo_i + 1 : lo_i + 4, lo_j]
                    pair_derivative += fock_energy_derivative(
                        lo_i,
                        lo_j,
                        -0.5 * (p_pi_sj @ d_spss_e),
                    )
                if has_p_j:
                    p_si_pj = P[lo_i, lo_j + 1 : lo_j + 4]
                    pair_derivative += fock_energy_derivative(
                        lo_i,
                        lo_j,
                        0.5 * (p_si_pj @ d_sssp_e),
                    )
                    for b in range(3):
                        pair_derivative += fock_energy_derivative(
                            lo_i,
                            lo_j + 1 + b,
                            0.5 * cross_ss * d_sssp_e[b],
                        )
                if has_p_i and has_p_j:
                    for b in range(3):
                        p_pi_pjb = P[lo_i + 1 : lo_i + 4, lo_j + 1 + b]
                        pair_derivative += fock_energy_derivative(
                            lo_i,
                            lo_j + 1 + b,
                            -0.5 * (p_pi_pjb @ d_sppp_e),
                        )
                if has_p_i:
                    for a in range(3):
                        pair_derivative += fock_energy_derivative(
                            lo_i + 1 + a,
                            lo_j,
                            -0.5 * cross_ss * d_spss_e[a],
                        )
                if has_p_i and has_p_j:
                    for a in range(3):
                        p_pia_pj = P[lo_i + 1 + a, lo_j + 1 : lo_j + 4]
                        pair_derivative += fock_energy_derivative(
                            lo_i + 1 + a,
                            lo_j,
                            0.5 * (p_pia_pj @ d_ppsp_e),
                        )
                    for a in range(3):
                        for b in range(3):
                            derivative = (
                                -0.5 * P[lo_i, lo_j + 1 + b] * d_sppp_e[a]
                                + 0.5 * P[lo_i + 1 + a, lo_j] * d_ppsp_e[b]
                            )
                            pair_derivative += fock_energy_derivative(
                                lo_i + 1 + a,
                                lo_j + 1 + b,
                                derivative,
                            )

                    pair_derivative += fock_energy_derivative(
                        lo_i,
                        lo_j,
                        -0.5
                        * np.tensordot(
                            P[lo_i + 1 : lo_i + 4, lo_j + 1 : lo_j + 4],
                            d_s_tensor,
                            axes=([0, 1], [0, 1]),
                        ),
                    )
                    for b in range(3):
                        derivative = -0.5 * np.tensordot(
                            P[lo_i + 1 : lo_i + 4, lo_j],
                            d_s_tensor[:, b, :],
                            axes=(0, 0),
                        )
                        pair_derivative += fock_energy_derivative(
                            lo_i,
                            lo_j + 1 + b,
                            derivative,
                        )
                    for a in range(3):
                        derivative = -0.5 * np.tensordot(
                            P[lo_i, lo_j + 1 : lo_j + 4],
                            d_s_tensor[a, :, :],
                            axes=(0, 0),
                        )
                        pair_derivative += fock_energy_derivative(
                            lo_i + 1 + a,
                            lo_j,
                            derivative,
                        )
                    for a in range(3):
                        for b in range(3):
                            pair_derivative += fock_energy_derivative(
                                lo_i + 1 + a,
                                lo_j + 1 + b,
                                -0.5 * cross_ss * d_s_tensor[a, b],
                            )

            # displacement = C_j - C_i
            gradient[j] += pair_derivative
            gradient[i] -= pair_derivative

    return gradient


def msindo_gradient_analytic(
    atomic_numbers,
    coords_angstrom,
    *,
    charge=0,
    multiplicity=1,
    nddo=False,
    max_iter=200,
    conv_tol=1e-10,
):
    """Analytic nuclear gradient (Ha/bohr) of the MSINDO RHF total energy.

    Runs one SCF, then for each atom pair (K,L) contracts the density P
    with the per-pair (R,th,phi) derivative matrices using the DEDXYZK/DEDXYZL
    formulas from MSINDO Fortran.

    Parameters
    ----------
    atomic_numbers : list of int
    coords_angstrom : (natom, 3) array-like, Angstrom

    Returns
    -------
    (natom, 3) ndarray -- gradient in Ha/bohr
    """
    if nddo:
        if multiplicity != 1:
            raise NotImplementedError(
                "MSINDO NDDO analytic gradients are closed-shell (RHF) only; "
                "open-shell NDDO derivatives are not implemented."
            )
        Z = list(atomic_numbers)
        unsupported = sorted({z for z in Z if z not in _SUPPORTED_NDDO})
        if unsupported:
            raise NotImplementedError(
                "MSINDO NDDO analytic gradients are available only for the "
                f"NDDO parameter set; got Z={unsupported}."
            )
        valence_electrons = sum(eff_core_charge(z) for z in Z)
        nelec = valence_electrons - charge
        if nelec < 0:
            raise ValueError(
                f"charge={charge} exceeds the {valence_electrons} "
                "valence electrons"
            )
        if nelec % 2 != 0:
            raise NotImplementedError(
                "MSINDO NDDO analytic gradients are closed-shell (RHF) only; "
                "the valence-electron count is odd."
            )
        # MSINDO nddoparam.f replaces the full parameter set before both SCF
        # and derivatives. The old direct route skipped this context and thus
        # differentiated a hybrid INDO/NDDO Hamiltonian even before its missing
        # multipole derivative terms were considered (#538).
        with _nddo_params():
            return _msindo_gradient_analytic_impl(
                atomic_numbers,
                coords_angstrom,
                charge=charge,
                multiplicity=multiplicity,
                nddo=True,
                max_iter=max_iter,
                conv_tol=conv_tol,
            )
    return _msindo_gradient_analytic_impl(
        atomic_numbers,
        coords_angstrom,
        charge=charge,
        multiplicity=multiplicity,
        nddo=False,
        max_iter=max_iter,
        conv_tol=conv_tol,
    )


def _msindo_gradient_analytic_impl(
    atomic_numbers,
    coords_angstrom,
    *,
    charge=0,
    multiplicity=1,
    nddo=False,
    max_iter=200,
    conv_tol=1e-10,
):
    Z = list(atomic_numbers)

    # Open-shell check -- fall back to FD for now
    cz = [eff_core_charge(z) for z in Z]
    nelec = sum(cz) - charge
    if nelec % 2 != 0:
        from warnings import warn

        warn("Analytic gradient only supports closed-shell (RHF); falling back to FD.")
        from .msindo import msindo_gradient_fd

        return msindo_gradient_fd(
            atomic_numbers,
            coords_angstrom,
            charge=charge,
            multiplicity=multiplicity,
            nddo=nddo,
            max_iter=max_iter,
            conv_tol=conv_tol,
        )

    if multiplicity == 1 and not nddo:
        kernel = _cpp_gradient_kernel()
        if kernel is not None:
            gradient_analytic, params = kernel
            grad_cpp = gradient_analytic(
                Z,
                np.asarray(coords_angstrom, float).tolist(),
                params,
                max_iter=max_iter,
                conv_tol=conv_tol,
                charge=charge,
            )
            if len(grad_cpp) == 0 and len(Z) != 0:
                raise RuntimeError(
                    "SCF not converged; cannot compute analytic gradient."
                )
            return np.asarray(grad_cpp, dtype=float)

    C = np.asarray(coords_angstrom, float) * ANGSTROM_TO_BOHR
    natom = len(Z)
    blocks, nsto = _atom_blocks(Z)
    nocc = nelec // 2

    # SCF once
    H, G = _build_core_and_gamma(Z, C, blocks, nsto, nddo=nddo)
    fock_extra = _nddo_fock_extra(Z, C, blocks) if nddo else None
    proactive_probe = multiplicity == 1 and all(
        z in _MSINDO_ROOT_PROBE_ELEMENTS for z in Z
    )
    if nddo or not proactive_probe:
        # Outside the validated light-element selector scope, analytic
        # derivatives retain their pre-existing strict stationary DIIS route.
        # In particular, never feed the heavy energy-only WICHT density here.
        P, F, e_scf, eps, converged, iters = _scf_rhf(
            H,
            G,
            blocks,
            Z,
            nocc,
            fock_extra=fock_extra,
            max_iter=max_iter,
            conv_tol=conv_tol,
        )
    else:
        P, F, e_scf, eps, converged, iters = _scf_rhf_molecular(
            H,
            G,
            blocks,
            Z,
            nocc,
            max_iter=max_iter,
            conv_tol=conv_tol,
        )
    if not converged:
        raise RuntimeError("SCF not converged; cannot compute analytic gradient.")

    grad = np.zeros((natom, 3))
    n = len(blocks)

    for k in range(natom):
        for l in range(k + 1, natom):
            rk, rl = C[k], C[l]
            dvec = rl - rk
            R = float(np.linalg.norm(dvec))
            if R < 1e-12:
                continue

            # Get per-pair derivative matrices
            pd = _pd._pair_blocks_deriv(Z[k], Z[l], rk, rl)

            # Chain rule factors: d(R,th,phi)/d(r_l - r_k)
            E = dvec / R
            cost, sint = E[2], math.sqrt(max(0.0, 1.0 - E[2] ** 2))
            if sint < 1e-12:
                # z-axis aligned: phi degenerate, skip phi derivatives
                cosphi, sinphi = 1.0, 0.0
                dphidx, dphidy, dphidz = 0.0, 0.0, 0.0
            else:
                cosphi, sinphi = E[0] / sint, E[1] / sint
                c1 = sint * sint * R  # = (1 - costh^2) * R
                dphidx = -E[1] / c1
                dphidy = E[0] / c1
                dphidz = 0.0

            # EDT = dE/dth = [costh cosphi, costh sinphi, -sinth]
            edt1 = cost * cosphi
            edt2 = cost * sinphi
            edt3 = -sint

            # Chain rule: d(R,th,phi)/dx = (dE/dx factors from DEINV)
            drdx, drdy, drdz = E[0], E[1], E[2]
            dthetadx = edt1 / R
            dthetady = edt2 / R
            dthetadz = edt3 / R

            # Extract density sub-blocks
            lo_k, hi_k = blocks[k]
            lo_l, hi_l = blocks[l]
            nk = hi_k - lo_k
            nl = hi_l - lo_l

            P_kk = P[lo_k:hi_k, lo_k:hi_k]
            P_ll = P[lo_l:hi_l, lo_l:hi_l]
            P_kl = P[lo_k:hi_k, lo_l:hi_l]  # (nk x nl), stored as P[K_i, L_j]

            # RHF: total density P, a and b are P/2
            # For the RHF formula in DEDXYZK:
            #   PLK = P_kl.T  (nl x nk, stored as P[L_j, K_i] in Fortran column-major)
            #   PLKA = PLKB = 0.5 * PLK
            PLK = P_kl.T  # (nl x nk)

            # ---- DEDXYZK: 1e terms from K's perspective ----
            # Diagonal (same-atom on K): S_i P_ii * H_ii
            # + off-diagonal (same-atom on K): 2 S_{i<j} P_ij * H_ij
            dx, dy, dz = 0.0, 0.0, 0.0
            for i in range(nk):
                Pii = P_kk[i, i]
                dx += Pii * (
                    pd.HK1_DR[i, i] * drdx
                    + pd.HK1_DT[i, i] * dthetadx
                    + pd.HK1_DP[i, i] * dphidx
                )
                dy += Pii * (
                    pd.HK1_DR[i, i] * drdy
                    + pd.HK1_DT[i, i] * dthetady
                    + pd.HK1_DP[i, i] * dphidy
                )
                dz += Pii * (
                    pd.HK1_DR[i, i] * drdz
                    + pd.HK1_DT[i, i] * dthetadz
                    + pd.HK1_DP[i, i] * dphidz
                )
                for j in range(i + 1, nk):
                    Pij = P_kk[j, i]
                    # H_ij (j,i) in Fortran = P_kk[j,i] in Python
                    # 2 * Pij * dH/dcoord
                    dx += (
                        2.0
                        * Pij
                        * (
                            pd.HK1_DR[j, i] * drdx
                            + pd.HK1_DT[j, i] * dthetadx
                            + pd.HK1_DP[j, i] * dphidx
                        )
                    )
                    dy += (
                        2.0
                        * Pij
                        * (
                            pd.HK1_DR[j, i] * drdy
                            + pd.HK1_DT[j, i] * dthetady
                            + pd.HK1_DP[j, i] * dphidy
                        )
                    )
                    dz += (
                        2.0
                        * Pij
                        * (
                            pd.HK1_DR[j, i] * drdz
                            + pd.HK1_DT[j, i] * dthetadz
                            + pd.HK1_DP[j, i] * dphidz
                        )
                    )

            # ---- DEDXYZK: pair-block terms ----
            # Formula (RHF): for each (iinK, jinL):
            #   F_g = 1/2 . dg_ij/dR . (Pii*Pjj - (Pij/2)^2 - (Pij/2)^2)
            #        = 1/2 . dg_ij/dR . (Pii*Pjj - 1/2 Pij^2)
            #   F_R = Pij * HKL2_DR[j,i]
            #   F_T = Pij * HKL2_DT[j,i]
            #   F_P = Pij * HKL2_DP[j,i]
            # Note: both DEDXYZK and DEDXYZL contribute these same terms,
            # so the total gradient = 2x the single-routine contribution.
            for i in range(nk):
                Pii = P_kk[i, i]
                for j in range(nl):
                    Pjj = P_ll[j, j]
                    Pij = P_kl[i, j]  # K_i, L_j
                    Pij2_half = 0.5 * Pij * Pij  # Pij^2/2 for RHF

                    # Gamma contribution: dg/dR (one copy, total = 2x)
                    F_gam_one = 0.5 * pd.d_gamma_local[i, j] * (Pii * Pjj - Pij2_half)

                    # Resonance terms (per the Fortran DEDXYZK/DEDXYZL indexing):
                    # DEDXYZK: PIJ = PLK(J,I) = P_OC[J,I] = P_CO[I,J],
                    #          FIJR = PIJ * HIJ2DR(J,I) = P_CO[I,J] * HKL2_DR[I,J]
                    # DEDXYZL: PIJ = PLK(I,J) = P_OC[I,J] = P_CO[J,I],
                    #          FIJR = PIJ * HIJ2DR(I,J) = P_CO[J,I] * HKL2_DR[J,I]
                    # Both contribute P_CO[i,j] * HKL2_DR[i,j], total = 2x.
                    F_R_one = Pij * pd.HKL2_DR[i, j]

                    # Both DEDXYZK and DEDXYZL contribute the same (F_gam + F_R).
                    # Total = 2x.
                    dx += (
                        2.0 * (F_gam_one + F_R_one) * drdx
                        + 2.0 * Pij * pd.HKL2_DT[i, j] * dthetadx
                        + 2.0 * Pij * pd.HKL2_DP[i, j] * dphidx
                    )
                    dy += (
                        2.0 * (F_gam_one + F_R_one) * drdy
                        + 2.0 * Pij * pd.HKL2_DT[i, j] * dthetady
                        + 2.0 * Pij * pd.HKL2_DP[i, j] * dphidy
                    )
                    dz += (
                        2.0 * (F_gam_one + F_R_one) * drdz
                        + 2.0 * Pij * pd.HKL2_DT[i, j] * dthetadz
                        + 2.0 * Pij * pd.HKL2_DP[i, j] * dphidz
                    )

            # ---- DEDXYZL: 1e terms from L's perspective ----
            for i in range(nl):
                Pii = P_ll[i, i]
                dx += Pii * (
                    pd.HL1_DR[i, i] * drdx
                    + pd.HL1_DT[i, i] * dthetadx
                    + pd.HL1_DP[i, i] * dphidx
                )
                dy += Pii * (
                    pd.HL1_DR[i, i] * drdy
                    + pd.HL1_DT[i, i] * dthetady
                    + pd.HL1_DP[i, i] * dphidy
                )
                dz += Pii * (
                    pd.HL1_DR[i, i] * drdz
                    + pd.HL1_DT[i, i] * dthetadz
                    + pd.HL1_DP[i, i] * dphidz
                )
                for j in range(i + 1, nl):
                    Pij = P_ll[j, i]
                    dx += (
                        2.0
                        * Pij
                        * (
                            pd.HL1_DR[j, i] * drdx
                            + pd.HL1_DT[j, i] * dthetadx
                            + pd.HL1_DP[j, i] * dphidx
                        )
                    )
                    dy += (
                        2.0
                        * Pij
                        * (
                            pd.HL1_DR[j, i] * drdy
                            + pd.HL1_DT[j, i] * dthetady
                            + pd.HL1_DP[j, i] * dphidy
                        )
                    )
                    dz += (
                        2.0
                        * Pij
                        * (
                            pd.HL1_DR[j, i] * drdz
                            + pd.HL1_DT[j, i] * dthetadz
                            + pd.HL1_DP[j, i] * dphidz
                        )
                    )

            # ---- Nuclear repulsion gradient ----
            # Both DEDXYZK and DEDXYZL add this term, so 2x total.
            zke = float(eff_core_charge(Z[k]))
            zle = float(eff_core_charge(Z[l]))
            zkl = 0.5 * zke * zle  # factor from dedxyzk.f: 1/2 . CZ(K) . CZ(L)
            drkl = -1.0 / (R * R)  # d/dR of 1/R
            dnuc = 2.0 * zkl * drkl  # 2x from both routines
            dx += dnuc * drdx
            dy += dnuc * drdy
            dz += dnuc * drdz

            # Accumulate: DEDXYZK sign convention
            #   EDX(L) += DX, EDX(K) -= DX
            grad[l, 0] += dx
            grad[l, 1] += dy
            grad[l, 2] += dz
            grad[k, 0] -= dx
            grad[k, 1] -= dy
            grad[k, 2] -= dz

    if nddo:
        grad += _nddo_gradient_correction(Z, C, blocks, P)

    return grad
