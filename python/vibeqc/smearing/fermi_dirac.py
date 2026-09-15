"""Fermi-Dirac occupations and electronic entropy on a k-mesh.

Closed-shell variant for now (occupations in ``[0, 2]``, one
shared chemical potential). Open-shell (per-spin mu) lands at M3
via the same mu-bisection wired with per-spin particle-count
constraints.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ._mu_bisection import bracket_from_eigenvalues, find_chemical_potential


def fermi_dirac_occupations_per_k(
    eps_per_k: Sequence[np.ndarray],
    weights: Sequence[float],
    n_electrons_per_cell: float,
    temperature: float,
    spin_degeneracy: float = 2.0,
) -> Tuple[List[np.ndarray], float, float]:
    """Fermi-Dirac occupations for a weighted k-mesh, one chemical potential.

    Returns ``(occ_per_k, mu, entropy)`` where ``occ_per_k`` contains
    per-band occupations in ``[0, spin_degeneracy]``, ``mu`` is the
    chemical potential in Hartree, and ``entropy`` is the dimensionless
    electronic entropy ``S/k_B`` per unit cell.

    The particle constraint is
    ``sum_k w_k sum_i n_i(k) = n_electrons_per_cell`` with
    ``n_i(k) = g / (1 + exp((eps_i(k) - mu) / temperature))`` and
    ``g = spin_degeneracy``.

    ``spin_degeneracy`` selects the channel convention:

    * ``2.0`` (default) -- closed-shell: occupations in ``[0, 2]``, the
      v0.4.0 contract; one mu conserves the total electron count.
    * ``1.0`` -- a single open-shell spin channel: occupations in
      ``[0, 1]``, ``n_electrons_per_cell`` is that channel's particle
      count, and the returned mu is that channel's mu_s. The open-shell
      driver calls this once per spin (see
      :func:`vibeqc.smearing.apply.apply_smearing_open_shell`).
    """
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError(
            "fermi_dirac_occupations_per_k: temperature must be positive"
        )
    g = float(spin_degeneracy)
    if g <= 0.0:
        raise ValueError(
            "fermi_dirac_occupations_per_k: spin_degeneracy must be > 0"
        )

    eps_real = [np.asarray(np.real(e), dtype=float) for e in eps_per_k]
    if not eps_real:
        raise ValueError("fermi_dirac_occupations_per_k: no k-points")
    if any(e.ndim != 1 or e.size == 0 for e in eps_real):
        raise ValueError(
            "fermi_dirac_occupations_per_k: eigenvalue arrays must be "
            "non-empty 1D arrays"
        )

    w_arr = np.asarray(weights, dtype=float)
    if w_arr.shape != (len(eps_real),):
        raise ValueError(
            "fermi_dirac_occupations_per_k: weights length must match "
            "number of k-points"
        )
    if np.any(w_arr < 0.0):
        raise ValueError(
            "fermi_dirac_occupations_per_k: weights must be >= 0"
        )
    if not np.isclose(float(w_arr.sum()), 1.0):
        raise ValueError(
            "fermi_dirac_occupations_per_k: weights must sum to 1"
        )

    target = float(n_electrons_per_cell)
    max_electrons = sum(
        float(w) * g * float(eps.shape[0])
        for eps, w in zip(eps_real, w_arr)
    )
    if target < -1e-12 or target > max_electrons + 1e-12:
        raise ValueError(
            "fermi_dirac_occupations_per_k: electron count is outside "
            "the available band capacity"
        )

    eps_min, eps_max = bracket_from_eigenvalues(eps_real)

    # Empty or completely-full band manifold: the mu-bisection cannot bracket
    # n = 0 (mu -> -inf) or n = capacity (mu -> +inf). Both are integer endpoints
    # with zero entropy -- a fully spin-polarized channel hits them directly
    # (n_b = 0, or n_a = full capacity for an open-shell determinant). Return
    # the endpoint occupations without invoking the bisection.
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

        f = n_i / g
        f_safe = np.clip(f, 1e-300, 1.0 - 1e-15)
        s_i = -g * (
            f_safe * np.log(f_safe)
            + (1.0 - f_safe) * np.log(1.0 - f_safe)
        )
        entropy += float(w) * float(s_i.sum())

    return occ_per_k, mu, entropy
