"""Dyson equation for the localized impurity (Layer B).

Given the clean-surface Green function ``G0`` and a localized
perturbation ``Delta_V`` whose support is confined to a finite
real-space region (the adsorbate + locally perturbed surface atoms), the
perturbed Green function on that region solves the Dyson equation

    G = G0 + G0 Delta_V G   =>   (I - G0 Delta_V) G = G0.

Because ``Delta_V`` has finite support, the equation only ever involves
the region-restricted blocks of ``G0`` -- this is the KKR impurity Green
function construction transplanted to a localized (Gaussian / Wannier)
surface basis. Translational symmetry is broken *locally*, reaching the
genuinely isolated (zero-coverage) adsorbate without a supercell.

Reference: standard multiple-scattering / KKR impurity embedding; see
J. E. Inglesfield, "The Embedding Method for Electronic Structure," IOP
Publishing (2015), doi:10.1088/978-0-7503-1042-0, Ch. on impurity
embedding.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def dyson_solve(
    g0: NDArray[np.complex128] | complex,
    delta_v: NDArray[np.complex128] | complex,
) -> NDArray[np.complex128]:
    """Solve ``G = G0 + G0 Delta_V G`` on the perturbed region.

    Parameters
    ----------
    g0
        Clean-surface Green function restricted to the perturbed region,
        ``(n, n)`` (scalar / ``(1, 1)`` for the single-site model).
    delta_v
        Localized perturbation on the same region, ``(n, n)``.

    Returns
    -------
    G
        Perturbed Green function ``(I - G0 Delta_V)^{-1} G0``, ``(n, n)``.
    """
    g0 = np.atleast_2d(np.asarray(g0, dtype=np.complex128))
    dv = np.atleast_2d(np.asarray(delta_v, dtype=np.complex128))
    n = g0.shape[0]
    if g0.shape != (n, n) or dv.shape != (n, n):
        raise ValueError("g0 and delta_v must be square and the same size")
    ident = np.eye(n, dtype=np.complex128)
    return np.linalg.solve(ident - g0 @ dv, g0)
