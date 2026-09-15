"""DIRECT periodic J/K via the existing C++ build_fock_2e_real_space.

The existing C++ kernel ``build_fock_2e_real_space`` already does the
correct triple-cell-sum for the periodic 4-center Coulomb tensor:

    F_2e(c_g)  =  S_{c_l, c_s}  S_{muν ls}
                     [(mu_0 ν_g | l_l s_s)  P(c_l, c_s)
                      - 1/2 a (mu_0 l_l | ν_g s_s)  P(c_l, c_s)]

with three independent cell indices and per-cell Cauchy-Schwarz
screening. The buggy molecular-limit kernel
``build_jk_gamma_molecular_limit`` is a *Γ-projected* simplification
that hardcodes c_s = c_l -- that's where the missing-s_q convention
bug lived.

This wrapper plugs ``build_fock_2e_real_space`` into the
:mod:`vibeqc.periodic_jk_method` dispatch as the DIRECT method.

The Madelung exxdiv correction (PySCF's ``exxdiv='ewald'`` shift of
``-1/2 . madelung . S . D . S``) is **not** added here -- it lives in
the SCF driver, not in the J/K kernel. The DIRECT path naturally
sums over a finite cell shell so the Madelung shift is only a small
constant relative to the truncated lattice; for parity tests vs
PySCF GDF you'll want to add it back at the SCF level.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    CoulombMethod,
    LatticeSumOptions,
    PeriodicSystem,
    bloch_sum,
    build_fock_2e_real_space,
    compute_overlap_lattice,
)


__all__ = ["jk_via_direct"]


def jk_via_direct(
    system: PeriodicSystem,
    basis: BasisSet,
    D_gamma: np.ndarray,
    *,
    cutoff_bohr: float = 18.0,
    schwarz_threshold: float = 1e-12,
    exchange_scale: float = 1.0,
    omega: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Γ-point F_2e via DIRECT lattice sum, returning J and K.

    Parameters
    ----------
    system, basis
        Periodic system + AO basis (vibe-qc layout).
    D_gamma
        ``(nbf, nbf)`` Γ-point density matrix in vibe-qc AO order.
    cutoff_bohr
        Real-space cutoff for the lattice sum on each of the three
        cell indices. ~18 bohr is enough for sto-3g on MgO; larger
        for diffuse bases.
    schwarz_threshold
        Per-quartet Cauchy-Schwarz cutoff. Default 1e-12 matches
        molecular RHF tolerance.
    exchange_scale
        HF exchange admixture: 1.0 for RHF, 0.0 for pure DFT, the
        hybrid fraction for hybrid DFT (e.g. 0.20 for B3LYP, 0.25
        for PBE0).
    omega
        Range-separation parameter. ``0.0`` (default) -> full Coulomb;
        positive w -> erfc-screened SR Coulomb (for Ewald composition).

    Returns
    -------
    J : np.ndarray of shape (nbf, nbf)
        Hartree matrix at Γ in vibe-qc AO order.
    K : np.ndarray of shape (nbf, nbf)
        Exchange matrix at Γ in vibe-qc AO order. For hybrid DFT
        callers should multiply by the hybrid fraction.

    Notes
    -----
    The function calls ``build_fock_2e_real_space`` *twice*:
      - once with ``exchange_scale=0`` to extract pure J;
      - once with ``exchange_scale=1`` to get J - 1/2 K, then
        K = -2 . (F_with_K - F_pure_J).

    A future optimization would expose a "J only" / "K only" entry
    point on the C++ side to avoid the duplicate work.
    """
    nbf = basis.nbasis
    if D_gamma.shape != (nbf, nbf):
        raise ValueError(
            f"D_gamma shape {D_gamma.shape} != ({nbf}, {nbf})"
        )

    # Always use DIRECT_TRUNCATED cells. The lattice sum cells live
    # in lat_opts; the C++ builds its cell list from cutoff_bohr.
    lat_opts = LatticeSumOptions()
    lat_opts.coulomb_method = CoulombMethod.DIRECT_TRUNCATED
    lat_opts.cutoff_bohr = float(cutoff_bohr)
    lat_opts.nuclear_cutoff_bohr = max(25.0, 2.0 * float(cutoff_bohr))
    lat_opts.schwarz_threshold = float(schwarz_threshold)

    # Wrap D_gamma as a LatticeMatrixSet (block 0 = D, others = 0).
    D_set = compute_overlap_lattice(basis, system, lat_opts)
    zero = np.zeros((nbf, nbf))
    for i in range(len(D_set)):
        D_set.set_block(i, D_gamma if i == 0 else zero)

    # Pure J: F_J = build_fock_2e_real_space(..., exchange_scale=0, ...)
    F_J_set = build_fock_2e_real_space(
        basis, system, lat_opts, D_set, 0.0, float(omega),
    )
    J_gamma = np.real(bloch_sum(F_J_set, np.zeros(3)))
    J_gamma = 0.5 * (J_gamma + J_gamma.T)

    if exchange_scale == 0.0:
        return J_gamma, np.zeros((nbf, nbf))

    # J - 1/2 K: F_JK = build_fock_2e_real_space(..., exchange_scale=1, ...)
    F_JK_set = build_fock_2e_real_space(
        basis, system, lat_opts, D_set, 1.0, float(omega),
    )
    F_JK = np.real(bloch_sum(F_JK_set, np.zeros(3)))
    F_JK = 0.5 * (F_JK + F_JK.T)
    # F_JK = J - 1/2 K -> K = -2 . (F_JK - J)
    K_gamma = -2.0 * (F_JK - J_gamma)
    K_gamma = 0.5 * (K_gamma + K_gamma.T)

    return J_gamma, K_gamma
