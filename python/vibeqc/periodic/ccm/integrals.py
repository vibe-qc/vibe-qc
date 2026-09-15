"""CCM-weighted two-center integrals (overlap, kinetic).

These build the cyclic-cluster overlap ``S^CCM`` and kinetic ``T^CCM``
matrices by folding the real-space lattice blocks ``S(g)``, ``T(g)``
(from vibe-qc's existing lattice-sum integral engine) against the WSSC
weights (:mod:`vibeqc.periodic.ccm.wigner_seitz`):

    # Eq. 5 of Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
    # doi:10.1002/jcc.23550:
    #   S^CCM_{mu nu} = sum_{g} w[g; A(mu), B(nu)] * S(g)_{mu nu}
    # Eq. 6 (kinetic) is identical with T(g) in place of S(g).

This is the Γ-point analogue of a Bloch sum: at Γ an ordinary periodic
overlap is ``sum_g S(g)`` (unit weights); the CCM replaces the uniform
sum with the WSSC weights ``w[g]`` (1 inside the cell, 1/n on a boundary,
0 outside). Three- and four-center weighted integrals (V, ERIs) are the
next milestone and will reuse the same :class:`CCMSystem` geometry.
"""

from __future__ import annotations

import numpy as np

from vibeqc import (  # type: ignore
    compute_kinetic_lattice,
    compute_overlap_lattice,
)

__all__ = ["ccm_overlap", "ccm_kinetic", "fold_lattice_matrix_set"]


def fold_lattice_matrix_set(ccm, lms) -> np.ndarray:
    """Fold a real-space ``LatticeMatrixSet`` into a CCM matrix.

    Applies the WSSC two-center weights to each lattice block and sums:
    ``M^CCM = sum_g (w[g] broadcast to AO) * block[g]`` (eqs 5-6). Only the
    cells that are a minimum image of some atom pair contribute, so the sum
    runs over the handful of cells in ``ccm.cell_weight_matrices()``.

    Parameters
    ----------
    ccm : CCMSystem
        Provides the AO->atom map and the per-cell WSSC weight matrices.
    lms : LatticeMatrixSet
        Real-space blocks ``block[g]`` keyed by integer cell ``cell.index``.

    Returns
    -------
    (nbf, nbf) ndarray -- the CCM-weighted matrix.
    """
    weights_by_cell = ccm.cell_weight_matrices()
    block_by_index = {
        tuple(int(x) for x in np.asarray(c.index)): np.asarray(lms.blocks[i], dtype=float)
        for i, c in enumerate(lms.cells)
    }

    ao = ccm.ao_atom
    rows, cols = ao[:, None], ao[None, :]
    out = np.zeros((ccm.nbf, ccm.nbf), dtype=float)
    for g, w_atom in weights_by_cell.items():
        block = block_by_index.get(g)
        if block is None:
            raise ValueError(
                f"CCM fold: minimum-image cell {g} is absent from the lattice "
                "sum -- increase the lattice-sum cutoff (CCMSystem cutoff_bohr)."
            )
        out += w_atom[rows, cols] * block
    # Symmetrise to clean up floating-point asymmetry (S, T are symmetric).
    return 0.5 * (out + out.T)


def ccm_overlap(ccm) -> np.ndarray:
    """CCM-weighted overlap matrix ``S^CCM`` (eq. 5). Returns ``(nbf, nbf)``."""
    lms = compute_overlap_lattice(ccm.basis, ccm.cluster_system, ccm.lattice_options)
    return fold_lattice_matrix_set(ccm, lms)


def ccm_kinetic(ccm) -> np.ndarray:
    """CCM-weighted kinetic-energy matrix ``T^CCM`` (eq. 6). Returns ``(nbf, nbf)``."""
    lms = compute_kinetic_lattice(ccm.basis, ccm.cluster_system, ccm.lattice_options)
    return fold_lattice_matrix_set(ccm, lms)
