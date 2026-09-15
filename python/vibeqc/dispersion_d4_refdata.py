"""D4 reference-data machinery -- Phase D4b-1.

The native D4 dispersion energy (Phase D4b, see
``handovers/HANDOVER_D4_NATIVE.md``) needs a reference C6 dataset
re-derived from first principles. The C6 between two systems comes
from the Casimir-Polder relation:

    C6_AB = (3/pi) ∫_0^inf a_A(iw) . a_B(iw) dw

where ``a(iw)`` is the isotropic dynamic dipole polarizability on the
imaginary frequency axis -- smooth, real, monotone-decaying, no poles.

This module provides the **level-agnostic core** of that pipeline:

  * :func:`imaginary_frequency_grid` -- a quadrature grid for the
    semi-infinite ∫_0^inf ... dw, via a rational change of variable +
    Gauss-Legendre. Returns (w_g, w_g) such that
    ∫_0^inf f(w) dw ≈ S_g w_g . f(w_g).

  * :func:`casimir_polder_c6` -- the C6 quadrature itself, given
    a_A and a_B sampled on a grid.

  * :func:`uncoupled_polarizability_imag_freq` -- the first
    a(iw) provider: the independent-particle ("uncoupled", bare
    sum-over-states) dynamic polarizability of a closed-shell RHF
    reference. This is the simplest correct a(iw); the coupled
    CPKS / TD-DFT generalisation (which the production D4 reference
    data will use) is milestone D4b-1's second half and reuses the
    Casimir-Polder + grid machinery here unchanged.

The Casimir-Polder integrator + grid are validated analytically: for
single-pole (Unsöld / Drude) model polarizabilities
a_X(iw) = a0_X / (1 + (w/w0_X)^2), the integral has the closed-form
London result

    C6_AB = (3/2) . a0_A . a0_B . w0_A.w0_B / (w0_A + w0_B).

See ``tests/test_dispersion_d4_refdata.py``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    Molecule,
    RHFOptions,
    RHFResult,
    compute_dipole,
    run_rhf,
)
from .cphf import CPHFOptions, _infer_n_occ, dynamic_polarizability_rhf
from .properties import center_of_mass

__all__ = [
    "imaginary_frequency_grid",
    "casimir_polder_c6",
    "uncoupled_polarizability_imag_freq",
    "coupled_polarizability_imag_freq",
    "coupled_polarizability_imag_freq_dft",
    "london_c6_single_pole",
    "molecular_c6",
]


# ---------------------------------------------------------------------
# Imaginary-frequency quadrature grid.
# ---------------------------------------------------------------------

def imaginary_frequency_grid(n_points: int = 23,
                             omega_scale: float = 0.5):
    """Quadrature grid for the semi-infinite integral ∫_0^inf f(w) dw.

    Uses the rational change of variable w = s.(1+x)/(1-x), which maps
    x in [-1, 1) onto w in [0, inf), with dw = 2s/(1-x)^2 dx, and
    Gauss-Legendre nodes/weights for the x-integral on [-1, 1].

    Parameters
    ----------
    n_points
        Number of quadrature points. D3/D4 reference data uses ~23;
        the integrand (a product of two smooth decaying
        polarizabilities) converges quickly, so 23 is ample.
    omega_scale
        The map parameter ``s`` (atomic units). It sets where the
        nodes cluster; the integral is insensitive to it once
        ``n_points`` is adequate. ~0.5 a.u. (a typical molecular
        response frequency) is a good default.

    Returns
    -------
    (omegas, weights) : tuple of np.ndarray
        Both shape ``(n_points,)``. ``∫_0^inf f(w) dw ≈ S_g
        weights[g] . f(omegas[g])``. ``omegas`` are strictly
        positive and sorted ascending.
    """
    if n_points < 1:
        raise ValueError("imaginary_frequency_grid: n_points must be >= 1")
    if omega_scale <= 0.0:
        raise ValueError("imaginary_frequency_grid: omega_scale must be > 0")

    # Gauss-Legendre nodes/weights on [-1, 1].
    x, wx = np.polynomial.legendre.leggauss(n_points)

    # w = s (1 + x) / (1 - x);  dw/dx = 2 s / (1 - x)^2.
    omegas = omega_scale * (1.0 + x) / (1.0 - x)
    weights = wx * 2.0 * omega_scale / (1.0 - x) ** 2

    # leggauss returns ascending x, so omegas are already ascending.
    return omegas, weights


# ---------------------------------------------------------------------
# Casimir-Polder C6 quadrature.
# ---------------------------------------------------------------------

def casimir_polder_c6(alpha_a: Sequence[float],
                      alpha_b: Sequence[float],
                      weights: Sequence[float]) -> float:
    """C6 dispersion coefficient from the Casimir-Polder integral.

        C6_AB = (3/pi) ∫_0^inf a_A(iw) a_B(iw) dw
              ≈ (3/pi) S_g weights[g] . a_A(iw_g) . a_B(iw_g)

    Parameters
    ----------
    alpha_a, alpha_b
        Isotropic dynamic polarizabilities of systems A and B sampled
        on the imaginary-frequency grid (atomic units, bohr^3). Same
        length as ``weights``.
    weights
        Quadrature weights from :func:`imaginary_frequency_grid` (the
        second return value), already carrying the dw Jacobian.

    Returns
    -------
    float
        The C6 coefficient in atomic units (Hartree.bohr⁶).
    """
    a = np.asarray(alpha_a, dtype=np.float64)
    b = np.asarray(alpha_b, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if not (a.shape == b.shape == w.shape):
        raise ValueError(
            f"casimir_polder_c6: alpha_a {a.shape}, alpha_b {b.shape}, "
            f"weights {w.shape} must all have the same shape.")
    return float(3.0 / np.pi * np.sum(w * a * b))


def london_c6_single_pole(alpha0_a: float, omega0_a: float,
                          alpha0_b: float, omega0_b: float) -> float:
    """Closed-form C6 for single-pole (Unsöld / Drude) model
    polarizabilities a_X(iw) = a0_X / (1 + (w/w0_X)^2).

    The Casimir-Polder integral evaluates exactly to the London
    formula

        C6_AB = (3/2) a0_A a0_B . w0_A w0_B / (w0_A + w0_B).

    Provided as the analytic reference that
    :func:`casimir_polder_c6` + :func:`imaginary_frequency_grid` are
    validated against (see the test module).
    """
    return (1.5 * alpha0_a * alpha0_b
            * omega0_a * omega0_b / (omega0_a + omega0_b))


# ---------------------------------------------------------------------
# Uncoupled (independent-particle) dynamic polarizability.
# ---------------------------------------------------------------------

def uncoupled_polarizability_imag_freq(
        rhf_result: RHFResult,
        basis: BasisSet,
        molecule: Molecule,
        omegas: Sequence[float],
        *,
        origin: Optional[Sequence[float]] = None) -> np.ndarray:
    """Independent-particle isotropic dynamic dipole polarizability
    a⁰(iw) of a closed-shell RHF reference, on the imaginary axis.

    The uncoupled (no CPHF response coupling) sum-over-states form,
    for closed-shell RHF:

        a⁰_uv(iw) = S_{ia} 4 (e_a - e_i) d^u_ia d^v_ia
                            ─────────────────────────────
                                  (e_a - e_i)^2 + w^2

    with d^u_ia = <i|u|a> the occ->vir dipole matrix element. The
    factor 4 = 2 (closed-shell spin) x 2 (the +w and -w
    sum-over-states terms both contribute (e)/[(e)^2+w^2]). The
    isotropic value returned is the trace average
    a⁰(iw) = ⅓.[a⁰_xx + a⁰_yy + a⁰_zz](iw).

    At w = 0 this reduces to the uncoupled static polarizability
    S 4 d^2 / Δe. As w -> inf it decays as (S 4(e_a-e_i)d^2)/w^2.

    This is the *uncoupled* level -- the simplest correct a(iw). The
    production D4 reference data needs the *coupled* CPKS / TD-DFT
    response (milestone D4b-1 second half); that routine will reuse
    :func:`casimir_polder_c6` and :func:`imaginary_frequency_grid`
    here unchanged, swapping only the a(iw) provider.

    Parameters
    ----------
    rhf_result, basis, molecule
        A converged closed-shell RHF reference and its basis/molecule.
    omegas
        Imaginary frequencies (atomic units) at which to evaluate
        a⁰(iw) -- typically from :func:`imaginary_frequency_grid`.
        w = 0 is allowed (gives the static uncoupled polarizability).
    origin
        Gauge origin for the dipole operator. Defaults to the
        molecular centre of mass. The isotropic polarizability of a
        neutral system is gauge-invariant; the origin only matters
        for charged inputs.

    Returns
    -------
    np.ndarray
        ``(len(omegas),)`` isotropic a⁰(iw) values, atomic units
        (bohr^3). Strictly positive and monotone-decreasing in w.
    """
    if not rhf_result.converged:
        raise ValueError(
            "uncoupled_polarizability_imag_freq: RHFResult is not "
            "converged -- a stationary reference is required.")

    C = np.asarray(rhf_result.mo_coeffs, dtype=np.float64)
    eps = np.asarray(rhf_result.mo_energies, dtype=np.float64)
    D = np.asarray(rhf_result.density, dtype=np.float64)
    n_occ = _infer_n_occ(D, C)
    C_occ = C[:, :n_occ]
    C_vir = C[:, n_occ:]

    eps_occ = eps[:n_occ]
    eps_vir = eps[n_occ:]
    # Δe_ia = e_a - e_i, shape (n_occ, n_vir).
    delta = eps_vir[None, :] - eps_occ[:, None]
    if np.any(delta <= 0.0):
        raise ValueError(
            "uncoupled_polarizability_imag_freq: non-positive HOMO-LUMO "
            "gap -- the RHF reference is unstable.")

    # Dipole integrals (AO) -> occ-vir MO blocks.
    if origin is None:
        origin_vec = center_of_mass(molecule)
    else:
        origin_vec = np.asarray(origin, dtype=np.float64)
    dip = compute_dipole(basis, [float(x) for x in origin_vec])
    M_ao = np.stack([np.asarray(dip.x), np.asarray(dip.y),
                     np.asarray(dip.z)], axis=0)  # (3, n_bf, n_bf)

    # |d_ia|^2 summed over the three Cartesian directions, shape
    # (n_occ, n_vir). The isotropic polarizability needs S_u d^u_ia^2;
    # the ⅓ trace average is folded in below.
    d2 = np.zeros_like(delta)
    for axis in range(3):
        d_ia = C_occ.T @ M_ao[axis] @ C_vir
        d2 += d_ia * d_ia

    omega_arr = np.asarray(omegas, dtype=np.float64)
    out = np.empty(omega_arr.shape[0], dtype=np.float64)
    # numerator_ia = 4 (e_a-e_i) |d_ia|^2  (Cartesian-summed)
    numerator = 4.0 * delta * d2
    for g, omega in enumerate(omega_arr):
        # a⁰(iw) = ⅓ S_ia numerator_ia / (Δe^2 + w^2)
        out[g] = (1.0 / 3.0) * np.sum(numerator / (delta * delta + omega * omega))
    return out


# ---------------------------------------------------------------------
# Coupled (CPKS / TD-HF) dynamic polarizability -- milestone D4b-1b.
# ---------------------------------------------------------------------

def coupled_polarizability_imag_freq(
        rhf_result: RHFResult,
        basis: BasisSet,
        molecule: Molecule,
        omegas: Sequence[float],
        *,
        origin: Optional[Sequence[float]] = None,
        eri: Optional[np.ndarray] = None,
        cphf_options: Optional[CPHFOptions] = None) -> np.ndarray:
    """Coupled isotropic dynamic dipole polarizability a(iw) of a
    closed-shell RHF reference -- the **CPKS / TD-HF response** level.

    This is the production a(iw) provider for the D4 reference data:
    it includes the orbital-relaxation (coupled-response) coupling
    that :func:`uncoupled_polarizability_imag_freq` omits. It is a
    thin isotropic wrapper over
    :func:`vibeqc.cphf.dynamic_polarizability_rhf`, which solves the
    imaginary-frequency TD-HF response equation

        [ (A + B) + w^2 (A - B)⁻¹ ] U₊ = -2 P.

    The signature matches :func:`uncoupled_polarizability_imag_freq`
    so it drops straight into the :func:`casimir_polder_c6` +
    :func:`imaginary_frequency_grid` pipeline unchanged -- exactly the
    "swap only the a(iw) provider" plan of
    ``handovers/HANDOVER_D4_NATIVE.md``.

    Parameters
    ----------
    rhf_result, basis, molecule
        A converged closed-shell RHF reference and its basis/molecule.
    omegas
        Imaginary frequencies (atomic units); typically the first
        return value of :func:`imaginary_frequency_grid`. w = 0 gives
        the static coupled polarizability.
    origin
        Gauge origin for the dipole operator (defaults to centre of
        mass).
    eri
        Optional precomputed AO 4-index ERI tensor, built once and
        reused across all frequencies when omitted.
    cphf_options
        Optional :class:`vibeqc.cphf.CPHFOptions` solver knobs.

    Returns
    -------
    np.ndarray
        ``(len(omegas),)`` isotropic a(iw) values, atomic units
        (bohr^3). Strictly positive and monotone-decreasing in w; at
        w = 0 it equals the static coupled CPHF polarizability.
    """
    omega_arr = np.asarray(omegas, dtype=np.float64)
    tensors = dynamic_polarizability_rhf(
        rhf_result, basis, molecule, omega_arr,
        eri=eri, origin=origin, options=cphf_options)   # (n_omega, 3, 3)
    # Isotropic ⅓.tr per frequency.
    return np.einsum("gii->g", tensors) / 3.0


# ---------------------------------------------------------------------
# Coupled (CPKS / TD-DFT adiabatic) dynamic polarizability -- the
# correlated alpha(iw) provider for the production D4 reference data.
# ---------------------------------------------------------------------

def coupled_polarizability_imag_freq_dft(
        rks_result,
        basis: BasisSet,
        molecule: Molecule,
        omegas: Sequence[float],
        functional: str,
        *,
        eri: Optional[np.ndarray] = None,
        origin: Optional[Sequence[float]] = None,
        grid_options=None) -> np.ndarray:
    """Coupled isotropic dynamic dipole polarizability a(iw) at the
    **CPKS / TD-DFT (adiabatic) level** for a closed-shell RKS reference.

    This is the correlated counterpart of
    :func:`coupled_polarizability_imag_freq` (which is TD-HF): the
    response kernel uses the functional's exact-exchange admixture
    ``c_x`` plus its adiabatic XC kernel ``f_xc`` instead of full HF
    exchange, so the resulting a(iw) carries the correlation that TD-HF
    omits. For O/N this lifts the under-bound HF polarizability up to
    the level dftd4's PBE38 reference data was built at -- the
    production D4 reference C6 then meet the dftd4 parity bars.

    The closed-shell singlet Casida super-matrices (Casida 1995, in
    *Recent Advances in Density Functional Methods*, Part I, p. 155;
    hybrid composition Bauernschmitt & Ahlrichs, *Chem. Phys. Lett.*
    **256**, 454 (1996)) are

        A_{ia,jb} = d_ij d_ab (e_a - e_i) + 2(ia|jb) - c_x(ij|ab) + (ia|f_xc|jb)
        B_{ia,jb} = 2(ia|jb) - c_x(ib|ja) + (ia|f_xc|jb)

    and the imaginary-frequency dynamic response amplitude
    ``U_+ = X + Y`` solves, on the imaginary axis iw,

        [ (A + B) + w^2 (A - B)^{-1} ] U_+ = -2 P,

    with ``P`` the dipole-perturbation occ-vir block. The polarizability
    is ``a_xy(iw) = -2 sum_ia U_+^y(iw)_ia mu^x_ia`` -- the same
    contraction + factor convention as
    :func:`vibeqc.cphf.dynamic_polarizability_rhf`, so a(i0) reduces to
    the static coupled polarizability.

    The reference systems are small, so A and B are built and the
    dynamic equation is solved **densely**, reusing the validated TDDFT
    builders :func:`vibeqc.tddft._build_casida_matrices` and
    :func:`vibeqc.tddft._compute_alda_kernel_mo` unchanged -- the
    response kernel is exactly the one vibe-qc's TDDFT excitation
    energies are validated against. The adiabatic ``f_xc`` is
    frequency-independent, so the same kernel matrix enters every grid
    point and ``(A - B)`` is inverted once.

    Validated against the TD-HF path: with ``c_x = 1`` and no XC kernel
    (an HF reference) this reproduces
    :func:`coupled_polarizability_imag_freq` to ~1e-13.

    Parameters
    ----------
    rks_result
        Converged closed-shell RKS result (``functional`` orbitals).
    basis, molecule
        The system.
    omegas
        Imaginary-frequency magnitudes w >= 0 (atomic units).
    functional
        XC functional name (the same the RKS reference used). Must be a
        pure functional or a global hybrid; range-separated and double
        hybrids are refused by the TDDFT kernel resolver.
    eri
        Optional precomputed AO 4-index ERI tensor.
    origin
        Gauge origin for the dipole operator (default centre of mass).
    grid_options
        Optional DFT-grid controls for the XC-kernel evaluation.

    Returns
    -------
    np.ndarray
        ``(len(omegas),)`` isotropic a(iw), atomic units (bohr^3).
    """
    from ._vibeqc_core import compute_dipole, compute_eri
    from .cphf import _infer_n_occ
    from .tddft import (
        _build_casida_matrices,
        _compute_alda_kernel_mo,
        _resolve_exact_exchange_fraction,
        make_eri_provider,
    )

    if not rks_result.converged:
        raise ValueError(
            "coupled_polarizability_imag_freq_dft: RKS reference is not "
            "converged -- a stationary reference is required.")

    C = np.asarray(rks_result.mo_coeffs, dtype=np.float64)
    eps = np.asarray(rks_result.mo_energies, dtype=np.float64)
    D = np.asarray(rks_result.density, dtype=np.float64)
    n_occ = _infer_n_occ(D, C)
    n_vir = C.shape[1] - n_occ

    if eri is None:
        eri = np.asarray(compute_eri(basis), dtype=np.float64)
    provider = make_eri_provider(eri, C, n_occ)

    # Response kernel: c_x exact exchange + adiabatic f_xc of the
    # semilocal part (reuses the validated TDDFT builders).
    c_x = _resolve_exact_exchange_fraction(functional)
    kernel_mo = _compute_alda_kernel_mo(
        molecule, basis, functional, C, D, n_occ, grid_options=grid_options)
    A, B = _build_casida_matrices(
        eps, provider.ovov(), provider.oovv(), n_occ, c_x, kernel_mo)
    ApB = A + B
    # (A - B) is symmetric positive-definite for a stable closed-shell
    # reference; invert once and reuse across all frequencies.
    AmB_inv = np.linalg.inv(A - B)

    # Dipole perturbation, occ-vir block flattened over (i, a).
    if origin is None:
        origin_vec = center_of_mass(molecule)
    else:
        origin_vec = np.asarray(origin, dtype=np.float64)
    dip = compute_dipole(basis, [float(x) for x in origin_vec])
    M_ao = np.stack([np.asarray(dip.x), np.asarray(dip.y),
                     np.asarray(dip.z)], axis=0)
    C_occ, C_vir = C[:, :n_occ], C[:, n_occ:]
    P = np.empty((3, n_occ * n_vir))
    for ax in range(3):
        P[ax] = (C_occ.T @ M_ao[ax] @ C_vir).ravel()

    omega_arr = np.asarray(omegas, dtype=np.float64)
    out = np.empty(omega_arr.shape[0], dtype=np.float64)
    for g, w in enumerate(omega_arr):
        op = ApB + (w * w) * AmB_inv
        # Solve op . U = -2 P for all three Cartesian RHS at once.
        U = np.linalg.solve(op, (-2.0 * P).T).T  # (3, n_pair)
        alpha = -2.0 * (P @ U.T)  # a_xy = -2 P_x . U_y
        alpha = 0.5 * (alpha + alpha.T)
        out[g] = float(np.trace(alpha) / 3.0)
    return out


# ---------------------------------------------------------------------
# Molecular C6 -- the end-to-end pipeline (Phase D4b-2a).
# ---------------------------------------------------------------------

def molecular_c6(mol_a: Molecule,
                 basis_a: BasisSet,
                 mol_b: Optional[Molecule] = None,
                 basis_b: Optional[BasisSet] = None,
                 *,
                 n_freq: int = 23,
                 omega_scale: float = 0.5,
                 coupled: bool = True,
                 rhf_options: Optional[RHFOptions] = None,
                 cphf_options: Optional[CPHFOptions] = None) -> float:
    """Molecular C6 dispersion coefficient between two closed-shell
    molecules, the whole pipeline in one call.

    For each molecule: run RHF, evaluate the isotropic dynamic dipole
    polarizability ``a(iw)`` on an imaginary-frequency grid, then
    Casimir-Polder integrate

        C6_AB = (3/pi) ∫_0^inf a_A(iw) a_B(iw) dw.

    This is the Phase D4b-2 building block -- the same machinery that,
    applied to the D4 *reference systems* (rather than whole
    molecules), produces the reference C6 dataset. Validated here at
    the whole-molecule level against literature molecular C6 values.

    Parameters
    ----------
    mol_a, basis_a
        First molecule and its basis. Must be closed-shell.
    mol_b, basis_b
        Second molecule and its basis. If ``mol_b`` is ``None`` the
        homo-molecular coefficient ``C6(A, A)`` is returned (and
        ``basis_b`` is ignored).
    n_freq, omega_scale
        Imaginary-frequency grid controls -- see
        :func:`imaginary_frequency_grid`.
    coupled
        ``True`` (default) uses the coupled CPHF response
        :func:`coupled_polarizability_imag_freq`; ``False`` uses the
        independent-particle :func:`uncoupled_polarizability_imag_freq`.
    rhf_options
        Options for the RHF reference SCF. If ``None``, a default
        :class:`RHFOptions` tightened to ``conv_tol_energy = 1e-10``
        is used (a converged, stationary reference is required for the
        response step).
    cphf_options
        Solver knobs for the coupled-response path; ignored when
        ``coupled=False``.

    Returns
    -------
    float
        The C6 coefficient in atomic units (Hartree.bohr⁶).

    Notes
    -----
    Accuracy is set by the level of the underlying response -- coupled
    CPHF on a finite Gaussian basis. CPHF omits electron correlation,
    so molecular C6 values come out somewhat below high-accuracy
    dipole-oscillator-strength-distribution references, and a basis
    without diffuse functions underestimates the polarizability
    further. For quantitative reference C6 data, use an
    augmented basis; the pipeline itself (grid + Casimir-Polder) is
    exact to the quadrature tolerance -- see the analytic
    single-pole / London validation in
    ``tests/test_dispersion_d4_refdata.py``.
    """
    omegas, weights = imaginary_frequency_grid(n_points=n_freq,
                                               omega_scale=omega_scale)

    if rhf_options is None:
        rhf_options = RHFOptions()
        rhf_options.conv_tol_energy = 1.0e-10

    def _alpha(mol, basis):
        hf = run_rhf(mol, basis, rhf_options)
        if not hf.converged:
            raise RuntimeError(
                "molecular_c6: RHF reference did not converge "
                f"(n_iter={hf.n_iter}, energy={hf.energy:.6f}).")
        if coupled:
            return coupled_polarizability_imag_freq(
                hf, basis, mol, omegas, cphf_options=cphf_options)
        return uncoupled_polarizability_imag_freq(hf, basis, mol, omegas)

    alpha_a = _alpha(mol_a, basis_a)
    if mol_b is None:
        alpha_b = alpha_a
    else:
        if basis_b is None:
            raise ValueError(
                "molecular_c6: basis_b is required when mol_b is given.")
        alpha_b = _alpha(mol_b, basis_b)

    return casimir_polder_c6(alpha_a, alpha_b, weights)
