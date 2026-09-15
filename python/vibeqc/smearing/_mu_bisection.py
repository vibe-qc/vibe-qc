"""Shared chemical-potential bisection.

Every smearing flavor's particle-count constraint
``S_k w_k S_i n_i(e_i, mu) = n_target`` is solved with the same
bisection. The flavor-specific occupation formula is passed as a
callable; this module owns only the search loop.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def find_chemical_potential(
    particle_count: Callable[[float], float],
    *,
    target: float,
    eps_min: float,
    eps_max: float,
    width: float,
    n_iter: int = 200,
    abs_tol: float = 1e-14,
) -> float:
    """Bisect for ``mu`` such that ``particle_count(mu) == target``.

    ``width`` is a smearing width (in Hartree) used to expand the
    initial bracket beyond ``[eps_min - 10.width - 1, eps_max +
    10.width + 1]`` for safety; the search expands further if the
    bracket does not contain the solution.
    """
    lo = float(eps_min) - 10.0 * float(width) - 1.0
    hi = float(eps_max) + 10.0 * float(width) + 1.0
    expansion = max(100.0, 100.0 * float(width), 1.0)
    for _ in range(20):
        if particle_count(lo) <= target <= particle_count(hi):
            break
        lo -= expansion
        hi += expansion
        expansion *= 2.0
    else:
        raise RuntimeError(
            "find_chemical_potential: failed to bracket the requested "
            "electron count"
        )

    for _ in range(int(n_iter)):
        mid = 0.5 * (lo + hi)
        if particle_count(mid) > target:
            hi = mid
        else:
            lo = mid
        if hi - lo < float(abs_tol):
            break
    return 0.5 * (lo + hi)


def bracket_from_eigenvalues(
    eps_per_k: Sequence[np.ndarray],
) -> tuple[float, float]:
    """Return ``(min e, max e)`` across a list of per-k eigenvalue arrays."""
    if not eps_per_k:
        raise ValueError(
            "bracket_from_eigenvalues: empty eigenvalue list"
        )
    eps_real = [np.asarray(np.real(e), dtype=float) for e in eps_per_k]
    eps_min = float(min(e.min() for e in eps_real))
    eps_max = float(max(e.max() for e in eps_real))
    return eps_min, eps_max
