"""Python adapter for the C++ pair-centre moment shift.

Wraps :func:`_vibeqc_core.shift_multipole_moments_to_pair_centres`.
Falls back to pure Python on any error.

Provenance
----------
Pisani, Dovesi, and Roetti (1988), Ch. II.4c,
doi:10.1007/978-3-642-93385-1, places periodic product-distribution
multipoles at their centroids. Pisani-Dovesi (1980), Sec. 4, supplies the
adjoined diffuse s-Gaussian screening convention; the standard Gaussian
product theorem supplies its pair centre. The cache layout is an
implementation choice.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .bipole_pair_moments import PairMultipoleMoments

__all__ = ["pair_center_moments_native"]


def pair_center_moments_native(
    M_lat,
    basis,
    *,
    L_target: int = 4,
) -> PairMultipoleMoments:
    """Shift moments to adjoined-Gaussian pair centres using C++ OpenMP."""
    try:
        from ._vibeqc_core import (
            shift_multipole_moments_to_pair_centres,
            LatticeMultipoleSet,
        )
    except ImportError:
        from .bipole_pair_moments import pair_center_moments as _py
        return _py(M_lat, basis, L_target=L_target)

    if not isinstance(M_lat, LatticeMultipoleSet):
        from .bipole_pair_moments import pair_center_moments as _py
        return _py(M_lat, basis, L_target=L_target)

    origin = (float(M_lat.origin[0]), float(M_lat.origin[1]), float(M_lat.origin[2]))
    cpp_result = shift_multipole_moments_to_pair_centres(
        M_lat, basis, int(L_target), origin,
    )

    nbf = cpp_result.nbf
    cells_out = list(cpp_result.cells)
    slices = [(int(s[0]), int(s[1])) for s in cpp_result.shell_slices]

    blocks_out: List[List[np.ndarray]] = []
    for c in range(len(cells_out)):
        comp_blocks = [np.asarray(cpp_result.blocks[c][comp], dtype=float)
                       for comp in range(len(cpp_result.blocks[c]))]
        blocks_out.append(comp_blocks)

    n_sh = len(slices)
    centres_out = []
    for c in range(len(cells_out)):
        flat = cpp_result.centres_flat[c]
        arr = np.zeros((n_sh, n_sh, 3), dtype=float)
        for s1 in range(n_sh):
            for s2 in range(n_sh):
                ct = flat[s1 * n_sh + s2]
                arr[s1, s2, 0] = float(ct[0])
                arr[s1, s2, 1] = float(ct[1])
                arr[s1, s2, 2] = float(ct[2])
        centres_out.append(arr)

    return PairMultipoleMoments(
        nbf=nbf, L_max=L_target, cells=cells_out,
        blocks=blocks_out, centers=centres_out, shell_slices=slices,
    )
