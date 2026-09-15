"""Mermin finite-temperature free-energy occupations (Phys. Rev. 137, A1441, 1965).

Mermin established that at finite temperature the variational quantity in
density-functional theory is the grand potential (or the Helmholtz free
energy ``A = E - T·S`` at fixed particle number), not the bare internal
energy ``E``.  The occupation function that makes ``A`` stationary with
respect to the orbital occupations is the Fermi-Dirac distribution --
identical in form to the pragmatic "Fermi-Dirac smearing" used in
plane-wave codes.  The distinction is conceptual: calling it "Mermin
smearing" makes explicit that the reported energy is the free energy,
and that forces / stress are derivatives of ``A``.

This module is the Mermin-flavoured entry point into vibe-qc's smearing
infrastructure.  It supports both periodic (multi-k) and molecular
(single-Gamma) contexts through the same ``apply_smearing`` dispatcher.

Closed-shell (occupations in [0, 2]) and single-spin-channel (g = 1.0)
conventions are supported via the ``spin_degeneracy`` parameter, matching
the Fermi-Dirac module's API.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ._mu_bisection import bracket_from_eigenvalues, find_chemical_potential


def mermin_occupations_per_k(
    eps_per_k: Sequence[np.ndarray],
    weights: Sequence[float],
    n_electrons_per_cell: float,
    temperature: float,
    spin_degeneracy: float = 2.0,
) -> Tuple[List[np.ndarray], float, float]:
    """Mermin finite-temperature occupations for a weighted k-mesh.

    Returns ``(occ_per_k, mu, entropy)`` where ``occ_per_k`` contains
    per-band occupations in ``[0, spin_degeneracy]``, ``mu`` is the
    chemical potential in Hartree, and ``entropy`` is the dimensionless
    electronic entropy ``S/k_B`` per unit cell.

    The particle constraint is
    ``sum_k w_k sum_i n_i(k) = n_electrons_per_cell`` with the
    Fermi-Dirac occupation
    ``n_i(k) = g / (1 + exp((eps_i(k) - mu) / temperature))`` and
    ``g = spin_degeneracy``.  The occupation function is identical to
    :func:`fermi_dirac_occupations_per_k` -- the Mermin framework
    provides the variational justification for reporting the free
    energy ``A = E - T·S`` rather than the bare internal energy.

    ``spin_degeneracy`` selects the channel convention:

    * ``2.0`` (default) -- closed-shell: occupations in ``[0, 2]``.
    * ``1.0`` -- a single open-shell spin channel: occupations in
      ``[0, 1]``, ``n_electrons_per_cell`` is that channel's particle
      count.
    """
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError(
            "mermin_occupations_per_k: temperature must be positive"
        )
    g = float(spin_degeneracy)
    if g <= 0.0:
        raise ValueError(
            "mermin_occupations_per_k: spin_degeneracy must be > 0"
        )

    eps_real = [np.asarray(np.real(e), dtype=float) for e in eps_per_k]
    if not eps_real:
        raise ValueError("mermin_occupations_per_k: no k-points")
    if any(e.ndim != 1 or e.size == 0 for e in eps_real):
        raise ValueError(
            "mermin_occupations_per_k: eigenvalue arrays must be "
            "non-empty 1D arrays"
        )

    w_arr = np.asarray(weights, dtype=float)
    if w_arr.shape != (len(eps_real),):
        raise ValueError(
            "mermin_occupations_per_k: weights length must match "
            "number of k-points"
        )
    if np.any(w_arr < 0.0):
        raise ValueError(
            "mermin_occupations_per_k: weights must be >= 0"
        )
    if not np.isclose(float(w_arr.sum()), 1.0):
        raise ValueError(
            "mermin_occupations_per_k: weights must sum to 1"
        )

    target = float(n_electrons_per_cell)
    max_electrons = sum(
        float(w) * g * float(eps.shape[0])
        for eps, w in zip(eps_real, w_arr)
    )
    if target < -1e-12 or target > max_electrons + 1e-12:
        raise ValueError(
            "mermin_occupations_per_k: electron count is outside "
            "the available band capacity"
        )

    eps_min, eps_max = bracket_from_eigenvalues(eps_real)

    # Empty or completely-full band manifold.
    if target <= 1e-12:
        return [np.zeros_like(e) for e in eps_real], float(eps_min), 0.0
    if target >= max_electrons - 1e-12:
        return [np.full_like(e, g) for e in eps_real], float(eps_max), 0.0

    def particle_count(mu: float) -> float:
        total = 0.0
        for eps, w in zip(eps_real, w_arr):
            arg = np.clip((eps - mu) / temp, -50.0, 50.0)
            n_i = g / (1.0 + np.exp(arg))
            total += float(w) * float(n_i.sum())
        return total

    mu = find_chemical_potential(
        particle_count,
        target=target,
        eps_min=eps_min,
        eps_max=eps_max,
        width=temp,
    )

    occ_per_k: List[np.ndarray] = []
    entropy = 0.0
    for eps, w in zip(eps_real, w_arr):
        arg = np.clip((eps - mu) / temp, -50.0, 50.0)
        n_i = g / (1.0 + np.exp(arg))
        occ_per_k.append(n_i)

        # Fermi-Dirac entropy: S = -k_B Σ [f ln f + (1-f) ln(1-f)]
        f = n_i / g
        f_safe = np.clip(f, 1e-300, 1.0 - 1e-15)
        s_i = -g * (
            f_safe * np.log(f_safe)
            + (1.0 - f_safe) * np.log(1.0 - f_safe)
        )
        entropy += float(w) * float(s_i.sum())

    return occ_per_k, mu, entropy
