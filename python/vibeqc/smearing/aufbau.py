"""Closed-shell hard Aufbau occupations across a k-mesh.

The ``temperature == 0`` fast path used by every periodic SCF
driver before smearing was wired in. Kept here as part of the
shared utility so the ``T = 0`` and ``T > 0`` branches share a
single home.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def aufbau_occupations_per_k(
    eps_per_k: Sequence[np.ndarray],
    n_occ_each: int,
    occ_value: float = 2.0,
) -> List[np.ndarray]:
    """Hard Aufbau occupations for each k-point.

    The lowest ``n_occ_each`` orbitals receive occupation ``occ_value``;
    all remaining orbitals receive 0.

    ``occ_value`` is ``2.0`` for closed-shell (the historical default)
    and ``1.0`` for a single open-shell spin channel.
    """
    n_occ = int(n_occ_each)
    if n_occ < 0:
        raise ValueError("aufbau_occupations_per_k: n_occ_each must be >= 0")
    occ_per_k: List[np.ndarray] = []
    for eps in eps_per_k:
        n_band = int(np.asarray(eps).shape[0])
        if n_occ > n_band:
            raise ValueError(
                "aufbau_occupations_per_k: n_occ_each exceeds band count"
            )
        occ = np.zeros(n_band, dtype=float)
        occ[:n_occ] = float(occ_value)
        occ_per_k.append(occ)
    return occ_per_k
