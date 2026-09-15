"""Pre-computed quartet interaction tensor cache.

BIPOLE-EXACT-ZONE increment 3f.  The quartet-level bipolar far-field
contractor computes interaction tensors ``T(R)`` for each unique
inter-centre separation vector.  Since the geometry is fixed during
SCF, ALL required tensors can be pre-computed once during setup and
reused across every SCF iteration.

This module provides :class:`QuartetTensorCache` which, given a
penetration dispatch and moment buffer, pre-computes all interaction
tensors indexed by the separation vector and multipole order.
The far-field contractor then does O(1) look-up instead of O(L^4)
tensor assembly per quartet.

Memory: for a system with n_cells ~ 100 and n_sh ~ 10, there are
~10^4 unique separation vectors × up to 5 orders (L=0..4), each
storing a (25×25) float tensor → ~25 MB.  The tensor assembly
time is ~5-50 ms per tensor, so pre-computing 10^4 tensors takes
~50-500 s — compute once, save per SCF iteration.

Provenance
----------
Pisani, Dovesi, and Roetti (1988), Ch. II.4c,
doi:10.1007/978-3-642-93385-1, derives the periodic quartet expansion.
The tensor cache and its keying strategy are implementation prototypes,
not algorithms derived in that source or Saunders et al. (1992).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .bipole_multipole import multipole_interaction_tensor, n_components
from .bipole_quartet_far_field import (
    QuartetBipolarDispatch,
    quartet_effective_screening_parameter,
)

__all__ = [
    "QuartetTensorCache",
    "build_quartet_tensor_cache",
]


@dataclass
class QuartetTensorCache:
    """Pre-computed interaction tensors for a far-field dispatch.

    ``tensors[(R_key, L_order, mu_key)]`` gives the (L+1)^2 × (L+1)^2
    interaction tensor.  ``R_key`` is a rounded separation vector.
    ``mu_key`` is the rounded effective screening parameter (0 for bare).
    """

    tensors: Dict[Tuple, np.ndarray] = field(default_factory=dict)
    decimals: int = 5
    n_quartets: int = 0

    def get(
        self,
        R_sep: np.ndarray,
        L_order: int,
        mu_eff: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Look up a pre-computed tensor or return None."""
        key = (
            round(float(R_sep[0]), self.decimals),
            round(float(R_sep[1]), self.decimals),
            round(float(R_sep[2]), self.decimals),
            int(L_order),
            round(float(mu_eff), self.decimals),
        )
        return self.tensors.get(key)

    def __len__(self) -> int:
        return len(self.tensors)


def build_quartet_tensor_cache(
    moment_buffer,
    dispatch: QuartetBipolarDispatch,
    *,
    ewald_omega: float = 0.0,
    decimals: int = 5,
) -> QuartetTensorCache:
    """Pre-compute all interaction tensors needed for a far-field dispatch.

    Walks the dispatch once, collects all unique (R_sep, L_order) pairs,
    and computes the corresponding interaction tensors.

    Parameters
    ----------
    moment_buffer : SphericalMomentBuffer or SymmetryReducedMomentBuffer
        Buffer providing per-pair product-distribution centres.
    dispatch : QuartetBipolarDispatch
        Far-field quartet dispatch.
    ewald_omega : float
        Ewald splitting parameter.  >0 uses the erfc-screened short-range
        tensor; 0 uses the bare Coulomb tensor.
    decimals : int
        Rounding precision for the separation vector key.

    Returns
    -------
    QuartetTensorCache
    """
    cache = QuartetTensorCache(decimals=decimals)

    # Try the native C++ interaction tensors for speed.  Probe every kernel
    # the C++ branch below calls, not just the first: a ``_vibeqc_core``
    # exposing one without the other would satisfy a single-symbol probe and
    # then raise ImportError mid-loop.  (Both bindings landed together in
    # 1a26e6954, so this is a latent gap, not an observed one.)
    #
    # Both are bound under private ``_cpp_*`` aliases so they cannot shadow
    # the module-level pure-Python names the fallback branch relies on.  A
    # bare ``import`` of a module-level name anywhere in this function makes
    # that name function-local for the WHOLE body -- which is how the
    # fallback branch once shipped raising UnboundLocalError (7e94c7f18).
    try:
        from ._vibeqc_core import (
            multipole_erfc_interaction_tensor as _cpp_erfc_tensor,
        )
        from ._vibeqc_core import multipole_interaction_tensor as _cpp_tensor
        _use_cpp_tensor = True
    except ImportError:
        _use_cpp_tensor = False

    for q in range(len(dispatch)):
        s1, s2, bra_cell = dispatch.bra_pairs[q]
        s3, s4, ket_cell = dispatch.ket_pairs[q]
        L_order = dispatch.truncation_orders[q]
        if L_order <= 0:
            continue

        centre_bra = moment_buffer.get_center(s1, s2, bra_cell)
        centre_ket = moment_buffer.get_center(s3, s4, ket_cell)
        if centre_bra is None or centre_ket is None:
            continue
        R_sep = centre_ket - centre_bra
        r2 = float(np.dot(R_sep, R_sep))
        if r2 < 1e-30:
            continue

        # Compute effective screening (zero for bare Coulomb).
        mu_eff = 0.0
        if ewald_omega > 0:
            gamma_bra = dispatch.bra_widths[q]
            gamma_ket = dispatch.ket_widths[q]
            mu_eff = quartet_effective_screening_parameter(
                gamma_bra, gamma_ket, ewald_omega,
            )

        key = (
            round(float(R_sep[0]), decimals),
            round(float(R_sep[1]), decimals),
            round(float(R_sep[2]), decimals),
            int(L_order),
            round(float(mu_eff), decimals),
        )
        if key in cache.tensors:
            continue

        # Compute the tensor.
        if _use_cpp_tensor:
            if ewald_omega > 0:
                T_mat = np.asarray(_cpp_erfc_tensor(
                    L_order, L_order,
                    float(R_sep[0]), float(R_sep[1]), float(R_sep[2]),
                    float(mu_eff),
                ))
            else:
                T_mat = np.asarray(_cpp_tensor(
                    L_order, L_order,
                    float(R_sep[0]), float(R_sep[1]), float(R_sep[2]),
                ))
        else:
            if ewald_omega > 0:
                from .bipole_multipole import sr_multipole_interaction_tensor
                T_mat = sr_multipole_interaction_tensor(
                    L_order, L_order, R_sep, mu_eff,
                )
            else:
                T_mat = multipole_interaction_tensor(
                    L_order, L_order, R_sep,
                )
        cache.tensors[key] = T_mat
        cache.n_quartets += 1

    return cache
