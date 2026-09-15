"""Range-separated (w) Coulomb kernels for the GAPW periodic SCF path.

Range-separated hybrids split the Coulomb operator into short-range (SR)
and long-range (LR) pieces via the error function:

    1/r = erf(w r) / r   +   erfc(w r) / r
          ↑ LR (smooth)     ↑ SR (short-range)

where w (bohr⁻¹) is the range-separation parameter. The LR piece is
smooth and handled via FFT-Poisson on the plane-wave grid; the SR piece
is short-ranged and handled via libint erfc-screened integrals.

For the GAPW augmentation, only the **total** (unscreened) Coulomb
potential enters the per-atom correction -- the w-split applies only to
the smooth-grid FFT J and libint K builders.

Supported functional families
-----------------------------

* HSE06    -- w = 0.11 bohr⁻¹, 25% SR-HF exchange, 0% LR-HF exchange
* wB97X-V -- w = 0.30 bohr⁻¹, 16.6% total HF, 100% LR-HF
* wB97M-V -- w = 0.30 bohr⁻¹, 0% total HF, 100% LR-HF
* CAM-B3LYP -- w = 0.33 bohr⁻¹, a = 0.19, b = 0.46 (SR -> LR HF slope)

References
----------
* Heyd, Scuseria, Ernzerhof, JCP 118, 8207 (2003) -- HSE
* Chai & Head-Gordon, JCP 128, 084106 (2008) -- wB97X
* Yanai, Tew, Handy, CPL 393, 51 (2004) -- CAM-B3LYP
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import _vibeqc_core as _core
from .periodic_gapw_grid import GAPWExperimentalWarning, PlaneWaveGrid
from .periodic_gapw_j import (
    GpwEnergyBreakdown,
    _build_v_ne,
    _eigh_safe,
    _evaluate_xc_on_grid,
    _kinetic_lattice_gamma,
    _make_diis,
    _overlap_lattice_gamma,
    _project_vtau_to_ao,
    _project_vxc_to_ao,
    collocate_density_on_grid,
    project_potential_to_ao,
)

__all__ = [
    "RsJBuilder",
    "RsGapwJBuilder",
    "run_periodic_rhf_rsgapw",
    "run_periodic_rks_rsgapw",
    "omega_for_functional",
]


# ============================================================
# w lookup for common range-separated functionals
# ============================================================


def omega_for_functional(functional: str) -> tuple[float, float, float]:
    """Return ``(omega, hf_sr, hf_lr)`` for a range-separated functional.

    Parameters
    ----------
    functional
        Lower-case functional name, e.g. ``"hse06"``, ``"wb97x"``,
        ``"cam-b3lyp"``.

    Returns
    -------
    omega
        Range-separation parameter in bohr⁻¹.
    hf_sr_fraction
        Fraction of short-range exact exchange.
    hf_lr_fraction
        Fraction of long-range exact exchange.
    """
    key = functional.lower().replace("-", "").replace("_", "").replace("*", "")
    _table = {
        # HSE06: w = 0.11, 25% SR-HF, 0% LR-HF
        "hse06": (0.11, 0.25, 0.0),
        "hse": (0.11, 0.25, 0.0),
        # wB97X: w = 0.30, 15.77% total HF, 100% LR-HF
        "wb97x": (0.30, 0.1577, 1.0),
        # wB97X-D: w = 0.20 (reparameterised)
        "wb97xd": (0.20, 0.222, 1.0),
        # wB97X-V: w = 0.30
        "wb97xv": (0.30, 0.166, 1.0),
        # wB97M-V: w = 0.30
        "wb97mv": (0.30, 0.0, 1.0),
        # CAM-B3LYP: w = 0.33, a=0.19 SR-HF, b=0.46 (LR = a+b)
        "camb3lyp": (0.33, 0.19, 0.65),
        # LC-wPBE: w = 0.40, 0% SR, 100% LR
        "lcwpbe": (0.40, 0.0, 1.0),
    }
    if key in _table:
        return _table[key]
    raise ValueError(
        f"Unknown range-separated functional {functional!r}. "
        f"Supported: {list(_table.keys())}"
    )


# ============================================================
# Short-range J via total-minus-long-range
# ============================================================


def _build_j_long_range(
    basis,
    density_matrix: np.ndarray,
    grid: PlaneWaveGrid,
    omega: float,
    *,
    collocation_cache=None,
) -> np.ndarray:
    """Long-range Hartree-J via FFT-Poisson with erf(w r)/r kernel.

    J_LR(w) = V_H[r] with the erf(w r)/r Coulomb kernel.
    """
    D = np.asarray(density_matrix, dtype=float)
    rho = collocate_density_on_grid(basis, D, grid, cache=collocation_cache)
    V_lr = _core.solve_poisson_erf_screened(rho, grid.lattice_bohr, omega)
    J_lr = project_potential_to_ao(basis, V_lr, grid, cache=collocation_cache)
    return 0.5 * (J_lr + J_lr.T)


def _build_j_short_range(
    basis,
    density_matrix: np.ndarray,
    grid: PlaneWaveGrid,
    omega: float,
    *,
    collocation_cache=None,
) -> np.ndarray:
    """Short-range Hartree-J = J_total - J_LR(w).

    J_total via the full FFT-Poisson (1/r kernel); J_LR via the
    erf-screened kernel. The SR piece is the difference.
    """
    D = np.asarray(density_matrix, dtype=float)
    rho = collocate_density_on_grid(basis, D, grid, cache=collocation_cache)
    V_full = _core.solve_poisson_coulomb(rho, grid.lattice_bohr)
    J_full = project_potential_to_ao(basis, V_full, grid, cache=collocation_cache)
    V_lr = _core.solve_poisson_erf_screened(rho, grid.lattice_bohr, omega)
    J_lr = project_potential_to_ao(basis, V_lr, grid, cache=collocation_cache)
    J_sr = J_full - J_lr
    return 0.5 * (J_sr + J_sr.T)


# ============================================================
# Long-range and short-range K via libint
# ============================================================


def _build_k_short_range(
    basis,
    system,
    density_matrix: np.ndarray,
    omega: float,
    *,
    cutoff_bohr: float = 25.0,
) -> np.ndarray:
    """Short-range exchange via libint with the erfc(w r)/r kernel.

    Delegates to the C++ ``build_jk_gamma_molecular_limit`` with
    the ``omega`` parameter. Positive ``omega`` selects libint's
    erfc-screened Coulomb operator, i.e. the short-range piece.
    """
    D = np.asarray(density_matrix, dtype=float)
    lo = _core.LatticeSumOptions()
    lo.cutoff_bohr = cutoff_bohr
    jk = _core.build_jk_gamma_molecular_limit(
        basis,
        system,
        lo,
        D,
        omega=omega,
    )
    return np.asarray(jk.K)


def _build_k_long_range(
    basis,
    system,
    density_matrix: np.ndarray,
    omega: float,
    *,
    cutoff_bohr: float = 25.0,
) -> np.ndarray:
    """Long-range exchange = K_total - K_SR(w)."""
    D = np.asarray(density_matrix, dtype=float)
    lo = _core.LatticeSumOptions()
    lo.cutoff_bohr = cutoff_bohr
    # Total K (w = 0).
    jk_full = _core.build_jk_gamma_molecular_limit(
        basis,
        system,
        lo,
        D,
        omega=0.0,
    )
    K_full = np.asarray(jk_full.K)
    # Short-range K.
    jk_sr = _core.build_jk_gamma_molecular_limit(
        basis,
        system,
        lo,
        D,
        omega=omega,
    )
    K_sr = np.asarray(jk_sr.K)
    K_lr = K_full - K_sr
    return 0.5 * (K_lr + K_lr.T)


# ============================================================
# Range-separated J builder (smooth-grid GPW)
# ============================================================


class RsJBuilder:
    """Range-separated Hartree-J builder on the smooth FFT grid.

    Composes the total J as J = J_SR(w) + J_LR(w) with the
    erf/erfc split. Caches the AO values on the grid for reuse
    across SCF iterations.
    """

    def __init__(
        self,
        basis,
        grid: PlaneWaveGrid,
        omega: float,
        *,
        quiet: bool = False,
        collocation_cache=None,
    ) -> None:
        self._basis = basis
        self._grid = grid
        self._omega = float(omega)
        self._collocation_cache = collocation_cache
        if not quiet:
            warnings.warn(
                f"RsJBuilder: range-separated J with w={omega} -- experimental.",
                category=GAPWExperimentalWarning,
                stacklevel=2,
            )

    def build_J(self, density_matrix: np.ndarray) -> np.ndarray:
        """J = J_SR(w) + J_LR(w)."""
        J_lr = _build_j_long_range(
            self._basis,
            density_matrix,
            self._grid,
            self._omega,
            collocation_cache=self._collocation_cache,
        )
        J_sr = _build_j_short_range(
            self._basis,
            density_matrix,
            self._grid,
            self._omega,
            collocation_cache=self._collocation_cache,
        )
        J = J_lr + J_sr
        return 0.5 * (J + J.T)


# ============================================================
# Range-separated GAPW J builder (with augmentation)
# ============================================================


class RsGapwJBuilder:
    """Range-separated GAPW Hartree-J builder.

    Composes the smooth-grid RS J (via :class:`RsJBuilder`) with
    the GAPW augmentation correction. The augmentation correction
    is applied to the **total** (unscreened) Coulomb potential,
    so it's independent of w.
    """

    def __init__(
        self,
        basis,
        system,
        grid: PlaneWaveGrid,
        omega: float,
        *,
        lmax: int = 3,
        soft_cutoff: float = 3.0,
        n_radial: int = 80,
        lebedev_order: int = 17,
        quiet: bool = False,
    ) -> None:
        self._basis = basis
        self._system = system
        self._grid = grid
        self._omega = float(omega)
        self._rs_j = RsJBuilder(basis, grid, omega, quiet=quiet)

        from .periodic_gapw_augment import GapwJBuilder

        self._gapw = GapwJBuilder(
            basis,
            system,
            grid,
            lmax=lmax,
            soft_cutoff=soft_cutoff,
            n_radial=n_radial,
            lebedev_order=lebedev_order,
            quiet=quiet,
        )
        if not quiet:
            warnings.warn(
                f"RsGapwJBuilder: RS-GAPW with w={omega} -- experimental.",
                category=GAPWExperimentalWarning,
                stacklevel=2,
            )

    def build_J(self, density_matrix: np.ndarray) -> np.ndarray:
        """Range-separated GAPW J = J_RS(smooth) + J_augmentation.

        The smooth-grid piece uses the erf/erfc split. The
        augmentation correction is the full-Coulomb correction
        from GapwJBuilder (same as the unscreened GAPW correction).
        """
        D = np.asarray(density_matrix, dtype=float)
        J_smooth = self._rs_j.build_J(D)
        # Build the total (unscreened) augmentation correction.
        # Extract just the augmentation part from GapwJBuilder.
        J_gapw_total = self._gapw.build_J(D)
        J_gpw_total = self._gapw._gpw.build_J(D)
        J_aug = J_gapw_total - J_gpw_total
        J = J_smooth + J_aug
        return 0.5 * (J + J.T)


# ============================================================
# Range-separated GAPW SCF entry points
# ============================================================


def run_periodic_rhf_rsgapw(
    system,
    basis,
    *,
    omega: float,
    hf_sr_fraction: float = 0.25,
    hf_lr_fraction: float = 0.0,
    grid: Optional[PlaneWaveGrid] = None,
    cutoff_ha: Optional[float] = None,
    max_iter: int = 50,
    conv_tol_energy: float = 1e-9,
    conv_tol_density: float = 1e-7,
    damping: float = 0.0,
    initial_density: Optional[np.ndarray] = None,
    v_ne_convention: str = "ewald",
    functional: Optional[str] = None,
    use_diis: bool = True,
    diis_subspace_size: int = 8,
    diis_start_iter: int = 2,
    quiet: bool = False,
    **gapw_kwargs,
) -> object:
    """Closed-shell range-separated GAPW RHF SCF.

    The Fock matrix includes range-separated exchange:

        F = Hcore + J_SR(w) + J_LR(w)
            - c_SR . K_SR(w) - c_LR . K_LR(w)
            + V_xc (if DFT)

    where c_SR = hf_sr_fraction, c_LR = hf_lr_fraction.

    Parameters
    ----------
    system, basis
        Standard periodic system + basis.
    omega
        Range-separation parameter (bohr⁻¹).
    hf_sr_fraction
        Short-range exact-exchange fraction (c_SR).
    hf_lr_fraction
        Long-range exact-exchange fraction (c_LR).
    Additional SCF parameters match :func:`run_periodic_rhf_gapw`.
    functional
        DFT functional for the semilocal part (e.g. 'pbe' for HSE06).
    gapw_kwargs
        GAPW augmentation kwargs (lmax, soft_cutoff, etc.).

    Returns
    -------
    result
        :class:`GapwScfResult`.
    """
    if not quiet:
        warnings.warn(
            f"run_periodic_rhf_rsgapw: RS-GAPW with w={omega}, "
            f"c_SR={hf_sr_fraction}, c_LR={hf_lr_fraction} -- experimental.",
            category=GAPWExperimentalWarning,
            stacklevel=2,
        )

    if system.dim != 3:
        raise ValueError(f"run_periodic_rhf_rsgapw: dim == 3 only; got {system.dim}")

    n_elec = int(sum(int(a.Z) for a in system.unit_cell))
    if n_elec % 2 != 0:
        raise ValueError("run_periodic_rhf_rsgapw: odd-electron cell.")
    n_occ = n_elec // 2

    # Build grid.
    if grid is None:
        from .periodic_gapw_grid import make_grid as _make_grid

        if cutoff_ha is None:
            cutoff_ha = 300.0
        grid = _make_grid(np.asarray(system.lattice, dtype=float), cutoff_ha=cutoff_ha)

    # One-electron.
    T = _kinetic_lattice_gamma(basis, system)
    V_ne = _build_v_ne(basis, system, v_ne_convention, None, grid)
    S = _overlap_lattice_gamma(basis, system)
    Hcore = T + V_ne
    E_nn = float(_core.ewald_nuclear_repulsion(system, _core.EwaldOptions()))

    # RS-GAPW J builder.
    rs_builder = RsGapwJBuilder(
        basis,
        system,
        grid,
        omega,
        quiet=quiet,
        **gapw_kwargs,
    )

    # Initial density.
    if initial_density is None:
        s_eigs, U = _eigh_safe(S)
        S_half_inv = U @ np.diag(1.0 / np.sqrt(s_eigs)) @ U.T
        e, C_orth = _eigh_safe(S_half_inv @ Hcore @ S_half_inv)
        C = S_half_inv @ C_orth
        D = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
    else:
        D = np.asarray(initial_density, dtype=float).copy()

    # Orthogonalizer.
    s_eigs, U = _eigh_safe(S)
    S_half_inv = U @ np.diag(1.0 / np.sqrt(s_eigs)) @ U.T

    # Functional.
    func = None
    if functional is not None:
        func = _core.Functional(functional, 1)

    c_sr = float(hf_sr_fraction)
    c_lr = float(hf_lr_fraction)

    # DIIS state.
    diis = _make_diis(use_diis, diis_subspace_size)

    # SCF loop.
    E_prev = 0.0
    converged = False
    n_iter = 0
    C = np.zeros((basis.nbasis, basis.nbasis))
    e = np.zeros(basis.nbasis)
    scf_trace: list[dict] = []

    for it in range(1, max_iter + 1):
        D_sym = 0.5 * (D + D.T)
        J = rs_builder.build_J(D_sym)

        # Range-separated exchange.
        # K = c_SR . K_SR(w) + c_LR . K_LR(w)
        K_sr = _build_k_short_range(basis, system, D_sym, omega)
        K_lr = _build_k_long_range(basis, system, D_sym, omega)
        K_rs = c_sr * K_sr + c_lr * K_lr

        F = Hcore + J - 0.5 * K_rs

        # XC piece (semilocal part of the range-separated functional).
        e_xc_iter = 0.0
        if func is not None:
            rho_grid = collocate_density_on_grid(basis, D_sym, grid)
            # basis + density_matrix let meta-GGA functionals build t(r);
            # LDA/GGA ignore them.
            e_xc_iter, v_xc_grid, v_sigma_grid, grad_rho, v_tau_grid = (
                _evaluate_xc_on_grid(
                    rho_grid, grid, func, basis=basis, density_matrix=D_sym
                )
            )
            V_xc = _project_vxc_to_ao(
                basis,
                v_xc_grid,
                grid,
                v_sigma_grid=v_sigma_grid,
                grad_rho=grad_rho,
            )
            if v_tau_grid is not None:
                # t Fock term: 1/2 ∫ v_tau gradchi.gradchi (see
                # periodic_gapw_j._project_vtau_to_ao -- the multiplicative
                # projection is not the derivative of E_xc).
                V_tau = _project_vtau_to_ao(basis, v_tau_grid, grid)
                V_xc = V_xc + V_tau
            F = F + V_xc

        # Energy.
        E_elec = float(np.einsum("ij,ij->", D_sym, Hcore))
        E_elec += 0.5 * float(np.einsum("ij,ij->", D_sym, J))
        E_elec += -0.25 * float(np.einsum("ij,ij->", D_sym, K_rs))
        E_elec += e_xc_iter
        E = E_elec + E_nn

        # DIIS. ``D_sym`` is the density that built ``F``; see
        # tests/test_diis_error_vector_source.py.
        if diis is not None and it >= diis_start_iter:
            err = F @ D_sym @ S - S @ D_sym @ F
            err = S_half_inv @ err @ S_half_inv
            F = diis.extrapolate(F, err)

        # Solve F C = e S C.
        e, C_orth = _eigh_safe(S_half_inv @ F @ S_half_inv)
        C = S_half_inv @ C_orth
        D_new = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        if damping > 0.0:
            D_new = (1.0 - damping) * D_new + damping * D

        dE = E - E_prev
        dD = float(np.linalg.norm(D_new - D))
        n_iter = it
        scf_trace.append(
            {
                "iter": it,
                "energy": float(E),
                "delta_e": float(dE),
                "grad_norm": float(dD),
                "e_xc": float(e_xc_iter),
            }
        )
        if abs(dE) < conv_tol_energy and dD < conv_tol_density and it > 1:
            converged = True
            D = D_new
            break
        D = D_new
        E_prev = E

    D_final = 0.5 * (D + D.T)
    J_final = rs_builder.build_J(D_final)
    K_sr_final = _build_k_short_range(basis, system, D_final, omega)
    K_lr_final = _build_k_long_range(basis, system, D_final, omega)
    K_rs_final = c_sr * K_sr_final + c_lr * K_lr_final

    E_kin = float(np.einsum("ij,ij->", D_final, T))
    E_ne = float(np.einsum("ij,ij->", D_final, V_ne))
    E_H = 0.5 * float(np.einsum("ij,ij->", D_final, J_final))
    E_K = -0.25 * float(np.einsum("ij,ij->", D_final, K_rs_final))
    if func is not None:
        rho_grid = collocate_density_on_grid(basis, D_final, grid)
        e_xc_final, _v_xc_grid, _v_sigma_grid, _grad_rho, _v_tau_grid = (
            _evaluate_xc_on_grid(
                rho_grid,
                grid,
                func,
                basis=basis,
                density_matrix=D_final,
            )
        )
    else:
        e_xc_final = 0.0
    total_energy = E_kin + E_ne + E_H + E_K + float(e_xc_final) + E_nn
    breakdown = GpwEnergyBreakdown(
        e_kinetic=E_kin,
        e_nuclear_attraction=E_ne,
        e_hartree=E_H,
        e_hf_exchange=E_K,
        e_nuclear_repulsion=E_nn,
        e_total=total_energy,
        grid=grid,
        e_xc=float(e_xc_final),
        functional=functional,
    )

    from .periodic_gapw_augment import GapwScfResult

    J_smooth = rs_builder._rs_j.build_J(D_final)
    gapw_correction = 0.5 * float(
        np.einsum("ij,ij->", D_final, J_final - J_smooth)
    )
    return GapwScfResult(
        energy=total_energy,
        breakdown=breakdown,
        density=D,
        mo_coeffs=C,
        mo_energies=e,
        converged=converged,
        n_iter=n_iter,
        grid=grid,
        gapw_correction=gapw_correction,
        scf_trace=tuple(scf_trace),
    )


def run_periodic_rks_rsgapw(system, basis, *, functional, omega, **kwargs):
    """Closed-shell range-separated GAPW RKS SCF.

    Wraps :func:`run_periodic_rhf_rsgapw` with a DFT functional.
    The w, hf_sr_fraction, and hf_lr_fraction are auto-detected
    from the functional name via :func:`omega_for_functional`.
    """
    if not functional:
        raise ValueError("run_periodic_rks_rsgapw requires a functional=.")
    # Auto-detect w from the functional name.
    fname = functional.lower().replace("-", "").replace("_", "").replace("*", "")
    if any(rs in fname for rs in ["hse", "wb97", "cam", "lcwp"]):
        w, c_sr, c_lr = omega_for_functional(functional)
        omega = float(omega) if omega is not None else w
    else:
        # Not a range-separated functional -- fall back to standard GAPW.
        from .periodic_gapw_augment import run_periodic_rks_gapw

        return run_periodic_rks_gapw(system, basis, functional=functional, **kwargs)

    return run_periodic_rhf_rsgapw(
        system,
        basis,
        omega=omega,
        hf_sr_fraction=c_sr,
        hf_lr_fraction=c_lr,
        functional=functional,
        **kwargs,
    )
