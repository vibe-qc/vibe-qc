"""Legacy scalar Madelung orbital-energy shift helper.

This module performs one narrowly defined operation: compute the cell's
Ewald-Madelung scalar ``ξ`` and add ``+ξ`` to occupied orbital energies while
leaving virtual energies unchanged.  It is retained for low-level experiments
and is not wired into the toroidal MP2 producer.

This scalar operation must not be described as the periodic-MP2 correction of
Nejad, Zhu, Sorathia, and Tew (2025).  That method makes fitted charge
distributions chargeless and includes the associated surface-dipole term; a
blanket occupied-level shift does neither.  Consequently this helper has no
claim of restoring a universal MP2 denominator or matching an arbitrary KMP2
finite Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .madelung import madelung_constant_for_cell


@dataclass
class BvKCorrection:
    """Descriptor for the legacy scalar occupied-orbital shift.

    Attributes
    ----------
    xi : float
        Ewald-Madelung constant ``ξ`` (Ha) for the unit cell, from
        :func:`vibeqc.madelung.madelung_constant_for_cell`.  This is
        the scalar used by this helper.
    delta_occ : float
        Occupied-orbital energy shift to apply (``+ξ``).
    n_occ : int
        Number of occupied orbitals (for bookkeeping).
    """

    xi: float
    delta_occ: float
    n_occ: int


def compute_bvk_correction(
    system,
    n_occ: int,
) -> BvKCorrection:
    """Compute the legacy scalar occupied-orbital shift descriptor.

    Returns the Ewald-Madelung constant ``ξ`` and the mechanical
    occupied-orbital shift ``Δ_occ = +ξ``.  No claim is made that this is a
    complete periodic-MP2 finite-size correction.

    Parameters
    ----------
    system : PeriodicSystem
        The unit cell (must be 3-D, dim=3).  Provides the lattice.
    n_occ : int
        Number of doubly-occupied orbitals.

    Returns
    -------
    BvKCorrection
    """
    xi = float(madelung_constant_for_cell(system))
    return BvKCorrection(xi=xi, delta_occ=xi, n_occ=n_occ)


def apply_bvk_correction_to_energies(
    eps_occ: np.ndarray,
    eps_vir: np.ndarray,
    correction: BvKCorrection,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the legacy scalar shift to orbital energies.

    Occupied energies are shifted by ``+ξ``; virtual energies are left
    unchanged.

    Parameters
    ----------
    eps_occ : ndarray, shape (n_occ,)
        Uncoupled occupied orbital energies from periodic HF (with exxdiv='ewald').
    eps_vir : ndarray, shape (n_vir,)
        Virtual orbital energies (unchanged by the correction).
    correction : BvKCorrection
        The pre-computed correction from :func:`compute_bvk_correction`.

    Returns
    -------
    eps_occ_corr : ndarray, shape (n_occ,)
        Shifted occupied energies.
    eps_vir_corr : ndarray, shape (n_vir,)
        Unchanged virtual energies (returned as a copy for consistency).
    """
    eps_occ_arr = np.asarray(eps_occ, dtype=float)
    eps_vir_arr = np.asarray(eps_vir, dtype=float)
    if len(eps_occ_arr) != correction.n_occ:
        raise ValueError(
            f"apply_bvk_correction_to_energies: n_occ mismatch: "
            f"got {len(eps_occ_arr)}, correction.n_occ = {correction.n_occ}"
        )
    return eps_occ_arr + correction.delta_occ, eps_vir_arr.copy()
