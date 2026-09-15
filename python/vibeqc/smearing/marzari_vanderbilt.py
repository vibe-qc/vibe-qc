"""Marzari-Vanderbilt "cold smearing" (Phys. Rev. Lett. 82, 3296, 1999).

Cold smearing minimises the broadening of the Fermi surface by using
a first-order Hermite-polynomial expansion that makes the generalised
entropy vanish at the Fermi level.  Empirically, this converges total
energies to the zero-smearing limit faster than Fermi-Dirac for many
metals.

Closed-shell (occupations in [0, 2]) and single-spin-channel
(g=1.0) conventions are supported via the ``spin_degeneracy``
parameter, matching the Fermi-Dirac module's API.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.special import erfc

from ._mu_bisection import bracket_from_eigenvalues, find_chemical_potential

_SQRT_PI = float(np.sqrt(np.pi))
_SQRT_2 = float(np.sqrt(2.0))


def _mv_occupation(x: np.ndarray) -> np.ndarray:
    """Return the per-spin Marzari-Vanderbilt cold occupation.

    The paper defines ``x_paper = (mu - eps) / sigma`` and a
    spin-degenerate occupation. Here ``x = (eps - mu) / sigma`` and
    the caller applies ``spin_degeneracy``, giving the corresponding
    per-spin form of Eq. (1), Phys. Rev. Lett. 82, 3296 (1999).
    """
    shifted = x + 1.0 / _SQRT_2
    return (
        0.5 * erfc(shifted)
        + np.exp(-shifted * shifted) / (_SQRT_2 * _SQRT_PI)
    )


def _mv_entropy_integrand(x: np.ndarray) -> np.ndarray:
    """Generalised entropy integrand for MV cold smearing.

    This is the per-spin form of Eq. (2), after converting its opposite
    energy-variable sign convention. Together with :func:`_mv_occupation`
    it obeys ``s'(x) = x f'(x)``, making ``E - T*S`` stationary at fixed
    particle number. Generalised entropies need not be positive.
    """
    shifted = x + 1.0 / _SQRT_2
    return (
        0.5
        * np.exp(-shifted * shifted)
        * (1.0 + _SQRT_2 * x)
        / _SQRT_PI
    )


def marzari_vanderbilt_occupations_per_k(
    eps_per_k: Sequence[np.ndarray],
    weights: Sequence[float],
    n_electrons_per_cell: float,
    temperature: float,
    spin_degeneracy: float = 2.0,
) -> Tuple[List[np.ndarray], float, float]:
    """Marzari-Vanderbilt cold-smearing occupations for a weighted k-mesh.

    Returns ``(occ_per_k, mu, entropy)`` -- same contract as
    :func:`fermi_dirac_occupations_per_k`.
    """
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError(
            "marzari_vanderbilt_occupations_per_k: temperature must be positive"
        )
    g = float(spin_degeneracy)
    if g <= 0.0:
        raise ValueError(
            "marzari_vanderbilt_occupations_per_k: spin_degeneracy must be > 0"
        )

    eps_real = [np.asarray(np.real(e), dtype=float) for e in eps_per_k]
    if not eps_real:
        raise ValueError("marzari_vanderbilt_occupations_per_k: no k-points")
    if any(e.ndim != 1 or e.size == 0 for e in eps_real):
        raise ValueError(
            "marzari_vanderbilt_occupations_per_k: eigenvalue arrays must be "
            "non-empty 1D arrays"
        )

    w_arr = np.asarray(weights, dtype=float)
    if w_arr.shape != (len(eps_real),):
        raise ValueError(
            "marzari_vanderbilt_occupations_per_k: weights length must match "
            "number of k-points"
        )
    if np.any(w_arr < 0.0):
        raise ValueError("marzari_vanderbilt_occupations_per_k: weights must be >= 0")
    if not np.isclose(float(w_arr.sum()), 1.0):
        raise ValueError("marzari_vanderbilt_occupations_per_k: weights must sum to 1")

    target = float(n_electrons_per_cell)
    max_electrons = sum(
        float(w) * g * float(eps.shape[0]) for eps, w in zip(eps_real, w_arr)
    )
    if target < -1e-12 or target > max_electrons + 1e-12:
        raise ValueError(
            "marzari_vanderbilt_occupations_per_k: electron count is outside "
            "the available band capacity"
        )

    eps_min, eps_max = bracket_from_eigenvalues(eps_real)

    if target <= 1e-12:
        return [np.zeros_like(e) for e in eps_real], float(eps_min), 0.0
    if target >= max_electrons - 1e-12:
        return [np.full_like(e, g) for e in eps_real], float(eps_max), 0.0

    def particle_count(mu: float) -> float:
        total = 0.0
        for eps, w in zip(eps_real, w_arr):
            x = (eps - mu) / temp
            x = np.clip(x, -30.0, 30.0)
            n_i = g * _mv_occupation(x)
            total += float(w) * float(n_i.sum())
        return total

    # Widen the bracket: MV occupations can extend beyond [0, g].
    margin = 3.0 * temp
    lo_bracket = float(eps_min) - margin
    hi_bracket = float(eps_max) + margin

    mu = find_chemical_potential(
        particle_count,
        target=target,
        eps_min=lo_bracket,
        eps_max=hi_bracket,
        width=temp,
    )

    occ_per_k: List[np.ndarray] = []
    entropy = 0.0
    for eps, w in zip(eps_real, w_arr):
        x = np.clip((eps - mu) / temp, -30.0, 30.0)
        n_i = g * _mv_occupation(x)
        occ_per_k.append(n_i)

        s_i = g * _mv_entropy_integrand(x)
        entropy += float(w) * float(s_i.sum())

    return occ_per_k, mu, entropy
