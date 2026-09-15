"""Conversion from Python QuartetBipolarDispatch to C++ skip-mask format.

BIPOLE-EXACT-ZONE increment 3e.  The C++ function
``build_jk_2e_real_space_bipolar_dispatch`` expects the skip mask as a
vector-of-vectors: ``skip_mask[c_g * n_cells + c_lam]`` is a sorted
list of flat shell-quartet indices ``s1*nsh^3 + s2*nsh^2 + s3*nsh + s4``
for quartets that are FAR-FIELD (skipped from exact ERI evaluation).

The dispatch is fully unfolded (all 4 transposition variants) before
reaching this module, so all shell-pair orderings are covered in the
skip mask.  No Fock-matrix asymmetry remains.
"""

from __future__ import annotations

from typing import List

__all__ = [
    "dispatch_to_bipolar_skip_mask",
]


def dispatch_to_bipolar_skip_mask(
    dispatch: "QuartetBipolarDispatch",
    n_cells: int,
    n_shells: int,
) -> List[List[int]]:
    """Convert penetration dispatch to C++ bipolar_skip_mask.

    Parameters
    ----------
    dispatch : QuartetBipolarDispatch
        The far-field quartet dispatch from the geometric penetration
        criterion.  Only quartets with ``truncation_order > 0``
        (far-field) are included in the skip mask.
    n_cells : int
        Number of lattice cells in the C++ internal sum.
    n_shells : int
        Number of AO shells.

    Returns
    -------
    list of list of int
        ``skip_mask[i]`` where ``i = c_g * n_cells + c_lam`` is a
        sorted list of flat quartet indices.  Empty lists for cell
        pairs with no far-field quartets.
    """
    # Group far-field quartets by (c_g, c_lam) cell pair.
    from collections import defaultdict

    n_sh = n_shells
    groups: dict = defaultdict(list)

    for q in range(len(dispatch)):
        order = dispatch.truncation_orders[q]
        if order <= 0:
            continue  # near-field, not skipped
        s1, s2, g_idx = dispatch.bra_pairs[q]
        s3, s4, lam_idx = dispatch.ket_pairs[q]
        # The dispatch has been unfolded to include all 4 transposition
        # variants, so all (s1,s2) orderings are covered.
        flat = ((s1 * n_sh + s2) * n_sh + s3) * n_sh + s4
        pair_idx = g_idx * n_cells + lam_idx
        groups[pair_idx].append(flat)

    # Build the skip mask: one entry per cell pair.
    n_pairs = n_cells * n_cells
    skip_mask: List[List[int]] = []
    for i in range(n_pairs):
        if i in groups:
            vals = sorted(set(groups[i]))
            skip_mask.append(vals)
        else:
            skip_mask.append([])

    return skip_mask
