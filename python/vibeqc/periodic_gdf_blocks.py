"""Bloch assembly helpers for cell-resolved periodic GDF blocks.

The C++ DF kernels expose translation-resolved blocks,
``blocks[c] = integral(T_c)``. These helpers apply the same
``exp(+i k.T)`` convention as :func:`vibeqc.bloch_sum`, but for 2c and
3c DF tensors rather than ordinary AO matrices. They are intentionally
small and NumPy-only so the future native multi-k GDF driver can share
one tested phase convention across RHF, RKS, and hybrids.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "bloch_sum_2c_eri_blocks",
    "bloch_sum_3c_eri_blocks",
    "gdf_block_phases",
]


def gdf_block_phases(
    cell_vectors_bohr: np.ndarray,
    k_cart_bohr_inv: Sequence[float],
) -> np.ndarray:
    """Return ``exp(+i k.T_c)`` for each lattice block vector.

    Parameters are Cartesian: cell vectors in bohr, k-vector in bohr⁻¹.
    """
    vectors = np.asarray(cell_vectors_bohr, dtype=float).reshape(-1, 3)
    k = np.asarray(k_cart_bohr_inv, dtype=float).reshape(3)
    return np.exp(1j * (vectors @ k))


def bloch_sum_2c_eri_blocks(
    cell_vectors_bohr: np.ndarray,
    blocks: np.ndarray,
    k_cart_bohr_inv: Sequence[float],
) -> np.ndarray:
    """Bloch-sum cell-resolved 2c DF metric blocks.

    ``blocks`` has shape ``(n_cells, n_aux, n_aux)`` and the return value
    has shape ``(n_aux, n_aux)``.
    """
    block_arr = np.asarray(blocks, dtype=float)
    if block_arr.ndim != 3:
        raise ValueError(
            "bloch_sum_2c_eri_blocks: blocks must have shape "
            f"(n_cells, n_aux, n_aux); got {block_arr.shape}"
        )
    phases = gdf_block_phases(cell_vectors_bohr, k_cart_bohr_inv)
    if phases.shape[0] != block_arr.shape[0]:
        raise ValueError(
            "bloch_sum_2c_eri_blocks: cell vector count does not match "
            f"block count ({phases.shape[0]} != {block_arr.shape[0]})"
        )
    return np.tensordot(phases, block_arr, axes=(0, 0))


def bloch_sum_3c_eri_blocks(
    cell_vectors_bohr: np.ndarray,
    blocks: np.ndarray,
    k_cart_bohr_inv: Sequence[float],
) -> np.ndarray:
    """Bloch-sum cell-resolved 3c DF blocks.

    ``blocks`` has shape ``(n_cells, n_aux, n_orb, n_orb)`` and the
    return value has shape ``(n_aux, n_orb, n_orb)``. Individual
    translation blocks are not symmetrized in the two AO indices; callers
    that need the Γ-only tensor can symmetrize the summed tensor exactly
    as the legacy Γ kernel does.
    """
    block_arr = np.asarray(blocks, dtype=float)
    if block_arr.ndim != 4:
        raise ValueError(
            "bloch_sum_3c_eri_blocks: blocks must have shape "
            f"(n_cells, n_aux, n_orb, n_orb); got {block_arr.shape}"
        )
    phases = gdf_block_phases(cell_vectors_bohr, k_cart_bohr_inv)
    if phases.shape[0] != block_arr.shape[0]:
        raise ValueError(
            "bloch_sum_3c_eri_blocks: cell vector count does not match "
            f"block count ({phases.shape[0]} != {block_arr.shape[0]})"
        )
    return np.tensordot(phases, block_arr, axes=(0, 0))
