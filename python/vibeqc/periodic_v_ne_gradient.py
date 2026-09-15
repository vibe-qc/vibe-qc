"""Analytic gradient of the FT-based Ewald ``V_ne`` used by GDF.

The value route in :mod:`vibeqc.periodic_v_ne` combines the real-space
erfc-screened attraction, the reciprocal-space nuclear structure factor,
the AO-pair Fourier transform, and the finite ``G = 0`` overlap correction.
This module differentiates every one of those terms in the same gauge.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    EwaldOptions,
    LatticeSumOptions,
    PeriodicSystem,
    ao_pair_fourier_transform_gamma_gradient_weighted,
    compute_overlap_lattice,
    direct_lattice_cells,
    nuclear_erfc_lattice_gradient_contribution,
    overlap_lattice_gradient_contribution,
)
from .aux_basis import _ao_scales_for_rsgdf, rsgdf_dense_g_mesh
from .periodic_v_ne import _vne_ft_g_chunk_size


def _set_homogeneous_gamma_blocks(matrix_set, matrix: np.ndarray) -> None:
    """Fill a lattice matrix set with a homogeneous Gamma-point matrix."""
    home_only = all(
        tuple(int(v) for v in np.asarray(cell.index).reshape(3)) == (0, 0, 0)
        for cell in matrix_set.cells
    )
    zero = np.zeros_like(matrix)
    for cell_idx, cell in enumerate(matrix_set.cells):
        index = tuple(int(v) for v in np.asarray(cell.index).reshape(3))
        block = matrix if (index == (0, 0, 0) or not home_only) else zero
        matrix_set.set_block(cell_idx, block)


def compute_v_ne_ewald_3d_ft_gamma_gradient(
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    D: np.ndarray,
    *,
    ewald_options: Optional[EwaldOptions] = None,
    ke_cutoff: float = 200.0,
) -> np.ndarray:
    """Return the fixed-density gradient of ``Tr(D V_ne)`` at Gamma.

    ``basis``, ``system``, ``lat_opts``, ``ewald_options``, and
    ``ke_cutoff`` have the same meaning as in
    :func:`vibeqc.periodic_v_ne.compute_v_ne_ewald_3d_ft_gamma`.
    The result has shape ``(n_atoms, 3)`` and units Hartree/bohr.

    The reciprocal-space derivative is density- and kernel-contracted by the
    native AO-pair kernel while streaming bounded ``G`` chunks. This avoids
    materialising any per-cell AO-pair derivative tensor while still removing
    the former ``6N`` whole-``V_ne`` finite difference.
    """
    if system.dim != 3:
        raise ValueError(
            "compute_v_ne_ewald_3d_ft_gamma_gradient: requires dim == 3; "
            f"got dim = {system.dim}."
        )

    density_input = np.asarray(D)
    if np.iscomplexobj(density_input) and np.any(
        np.abs(np.imag(density_input)) > 1.0e-12
    ):
        raise ValueError(
            "compute_v_ne_ewald_3d_ft_gamma_gradient: D must be real at "
            "Gamma."
        )
    density = np.asarray(np.real(density_input), dtype=np.float64)
    n_orb = int(basis.nbasis)
    if density.shape != (n_orb, n_orb):
        raise ValueError(
            "compute_v_ne_ewald_3d_ft_gamma_gradient: D must have shape "
            f"({n_orb}, {n_orb}); got {density.shape}."
        )
    density = 0.5 * (density + density.T)

    lattice = np.asarray(system.lattice, dtype=np.float64)
    cell_volume = float(abs(np.linalg.det(lattice)))
    n_atoms = len(system.unit_cell)

    if ewald_options is None:
        ewald_options = EwaldOptions()
        ewald_options.real_cutoff_bohr = lat_opts.nuclear_cutoff_bohr
    alpha = float(ewald_options.alpha)
    if alpha <= 0.0:
        alpha = 2.0

    # Real-space erfc-screened attraction.
    density_set = compute_overlap_lattice(basis, system, lat_opts)
    _set_homogeneous_gamma_blocks(density_set, density)
    gradient = np.asarray(
        nuclear_erfc_lattice_gradient_contribution(
            basis, system, density_set, lat_opts, alpha
        ),
        dtype=np.float64,
    )

    # Reciprocal-space long-range attraction.  Its derivative contains both
    # the nuclear structure-factor response and the motion of the AO-pair
    # Fourier transform centres.
    G_all = rsgdf_dense_g_mesh(system, float(ke_cutoff))
    G2_all = np.einsum("gx,gx->g", G_all, G_all)
    nonzero = G2_all > 0.0
    G = np.asarray(G_all[nonzero], dtype=np.float64)
    G2 = np.asarray(G2_all[nonzero], dtype=np.float64)
    damping = np.exp(-G2 / (4.0 * alpha**2))
    significant = damping > 1.0e-14
    G = G[significant]
    G2 = G2[significant]
    damping = damping[significant]

    nuclei_z = np.array([atom.Z for atom in system.unit_cell], dtype=np.float64)
    nuclei_r = np.array(
        [list(atom.xyz) for atom in system.unit_cell], dtype=np.float64
    )
    phases = np.exp(-1j * (nuclei_r @ G.T))
    kernel = damping * (4.0 * np.pi / G2)
    v_long = -np.einsum("a,ag,g->g", nuclei_z, phases, kernel, optimize=True)

    cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
    cell_vectors = np.array(
        [np.asarray(cell.r_cart, dtype=np.float64) for cell in cells],
        dtype=np.float64,
    )
    if cell_vectors.size == 0:
        cell_vectors = np.zeros((1, 3), dtype=np.float64)

    ao_scales = _ao_scales_for_rsgdf(basis)
    pair_scales = np.outer(ao_scales, ao_scales)
    weighted_density = density * pair_scales
    g_chunk = _vne_ft_g_chunk_size(n_orb)

    from ._aopair_ft import ao_pair_fourier_transform_bloch

    k_gamma = np.zeros(3, dtype=np.float64)
    for start in range(0, len(G), g_chunk):
        stop = min(start + g_chunk, len(G))
        G_chunk = G[start:stop]
        kernel_chunk = kernel[start:stop]
        phases_chunk = phases[:, start:stop]
        v_long_chunk = v_long[start:stop]

        pair_ft = ao_pair_fourier_transform_bloch(
            basis, G_chunk, cell_vectors, k_cart=k_gamma
        )
        density_pair_ft = np.einsum(
            "mn,mng,mn->g",
            density,
            pair_ft.conj(),
            pair_scales,
            optimize=True,
        )
        del pair_ft

        # d[-Z_A kernel exp(-iG.R_A)]/dR_A.
        d_structure = (
            -nuclei_z[:, None, None]
            * kernel_chunk[None, :, None]
            * phases_chunk[:, :, None]
            * (-1j * G_chunk[None, :, :])
        )
        gradient += np.real(
            np.einsum(
                "agx,g->ax", d_structure, density_pair_ft, optimize=True
            )
        ) / cell_volume

        gradient += np.asarray(
            ao_pair_fourier_transform_gamma_gradient_weighted(
                basis,
                G_chunk,
                cell_vectors,
                weighted_density,
                v_long_chunk / cell_volume,
                n_atoms,
            ),
            dtype=np.float64,
        )

    # The real-space erfc sum contains the finite short-range G=0 tail,
    # while the GDF/PySCF gauge drops it.  The value route adds
    # (-v_short_G0) S, so its derivative is obtained by passing
    # v_short_G0 D to the helper that returns -Tr(W dS/dR).
    total_nuclear_charge = float(nuclei_z.sum())
    if abs(total_nuclear_charge) > 1.0e-12:
        v_short_g0 = (
            -(np.pi / (alpha**2 * cell_volume)) * total_nuclear_charge
        )
        g0_weight_set = compute_overlap_lattice(basis, system, lat_opts)
        _set_homogeneous_gamma_blocks(g0_weight_set, v_short_g0 * density)
        gradient += np.asarray(
            overlap_lattice_gradient_contribution(
                basis, system, g0_weight_set, lat_opts
            ),
            dtype=np.float64,
        )

    return gradient
