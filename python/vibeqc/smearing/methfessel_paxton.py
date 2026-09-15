"""Methfessel-Paxton occupation scheme (Phys. Rev. B 40, 3616, 1989).

Implements MP orders N=1 and N=2.  The smearing function replaces
the Fermi-Dirac step with a polynomial expansion that accelerates
k-point convergence for metals -- the negative-occupation tails
cancel the leading k-point integration error term.

Closed-shell (occupations in [0, 2]) and single-spin-channel
(g=1.0) conventions are supported via the ``spin_degeneracy``
parameter, matching the Fermi-Dirac module's API.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.special import erfc

from ._mu_bisection import bracket_from_eigenvalues, find_chemical_potential


# Hermite polynomials (physicists' convention).
def _H1(x: np.ndarray) -> np.ndarray:
    return 2.0 * x


def _H2(x: np.ndarray) -> np.ndarray:
    return 4.0 * x * x - 2.0


def _H3(x: np.ndarray) -> np.ndarray:
    return 8.0 * x**3 - 12.0 * x


def _H4(x: np.ndarray) -> np.ndarray:
    return 16.0 * x**4 - 48.0 * x * x + 12.0


# MP coefficients A_n = (-1)^n / (n! * 4^n)
_MP_A = {1: -1.0 / 4.0, 2: 1.0 / 32.0}

_SQRT_PI = float(np.sqrt(np.pi))


def _mp_occupation(x: np.ndarray, order: int) -> np.ndarray:
    """Methfessel-Paxton occupation S_N(x) for a single argument array."""
    result = 0.5 * erfc(x)
    corr = np.zeros_like(x)
    for n in range(1, order + 1):
        herm = {1: _H1, 2: _H2, 3: _H3}.get(2 * n - 1, lambda y: y * 0.0)
        corr += _MP_A[n] * herm(x)
    result += np.exp(-x * x) / _SQRT_PI * corr
    return result


def _mp_entropy_integrand(x: np.ndarray, order: int) -> np.ndarray:
    """Generalised entropy integrand for MP order N.

    The variational free energy requires ``s_N'(x) = x f_N'(x)`` for
    the MP occupation ``f_N``. Integrating the Hermite expansion gives

    ``s_N(x) = 1/2 A_N H_{2N}(x) exp(-x^2) / sqrt(pi)``,

    where the lower-order terms cancel exactly. This is the generalized
    entropy of Methfessel and Paxton, Phys. Rev. B 40, 3616 (1989).
    It is not positive definite and, in particular, MP order 1 is not
    an entropy-free "cold" approximation.
    """
    hermite = {1: _H2, 2: _H4}[order]
    return (
        0.5
        * _MP_A[order]
        * hermite(x)
        * np.exp(-x * x)
        / _SQRT_PI
    )


def methfessel_paxton_occupations_per_k(
    eps_per_k: Sequence[np.ndarray],
    weights: Sequence[float],
    n_electrons_per_cell: float,
    temperature: float,
    spin_degeneracy: float = 2.0,
    mp_order: int = 1,
) -> Tuple[List[np.ndarray], float, float]:
    """Methfessel-Paxton occupations for a weighted k-mesh.

    Returns ``(occ_per_k, mu, entropy)`` -- same contract as
    :func:`fermi_dirac_occupations_per_k`.
    """
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError(
            "methfessel_paxton_occupations_per_k: temperature must be positive"
        )
    order = int(mp_order)
    if order not in (1, 2):
        raise ValueError(
            f"methfessel_paxton_occupations_per_k: mp_order must be 1 or 2; got {order}"
        )
    g = float(spin_degeneracy)
    if g <= 0.0:
        raise ValueError(
            "methfessel_paxton_occupations_per_k: spin_degeneracy must be > 0"
        )

    eps_real = [np.asarray(np.real(e), dtype=float) for e in eps_per_k]
    if not eps_real:
        raise ValueError("methfessel_paxton_occupations_per_k: no k-points")
    if any(e.ndim != 1 or e.size == 0 for e in eps_real):
        raise ValueError(
            "methfessel_paxton_occupations_per_k: eigenvalue arrays must be "
            "non-empty 1D arrays"
        )

    w_arr = np.asarray(weights, dtype=float)
    if w_arr.shape != (len(eps_real),):
        raise ValueError(
            "methfessel_paxton_occupations_per_k: weights length must match "
            "number of k-points"
        )
    if np.any(w_arr < 0.0):
        raise ValueError("methfessel_paxton_occupations_per_k: weights must be >= 0")
    if not np.isclose(float(w_arr.sum()), 1.0):
        raise ValueError("methfessel_paxton_occupations_per_k: weights must sum to 1")

    target = float(n_electrons_per_cell)
    max_electrons = sum(
        float(w) * g * float(eps.shape[0]) for eps, w in zip(eps_real, w_arr)
    )
    if target < -1e-12 or target > max_electrons + 1e-12:
        raise ValueError(
            "methfessel_paxton_occupations_per_k: electron count is outside "
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
            n_i = g * _mp_occupation(x, order)
            total += float(w) * float(n_i.sum())
        return total

    # Widen the bracket: MP occupations can go slightly outside [0, g],
    # so the apparent particle-count range at finite mu is wider.
    margin = 3.0 * temp * (1.0 + 0.5 * order)
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
        n_i = g * _mp_occupation(x, order)
        occ_per_k.append(n_i)

        s_i = g * _mp_entropy_integrand(x, order)
        entropy += float(w) * float(s_i.sum())

    return occ_per_k, mu, entropy
