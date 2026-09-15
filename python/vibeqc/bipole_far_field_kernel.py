"""Pre-computed Fock kernel for the dormant quartet far-field prototype.

BIPOLE-EXACT-ZONE increment 7b.  Instead of recomputing the far-field
Fock contribution from multipole moments at every SCF iteration,
pre-compute the effective Fock kernel once during setup and store
it as a sparse dictionary.  Each SCF iteration then contracts the
kernel with the density matrix in a single vectorized operation.

The stored buffer and sparse kernel are implementation cache strategies.
Pisani-Dovesi-Roetti (1988), Ch. II.4c, derives the periodic quartet
expansion but does not derive this kernel construction.

For diamond at cutoff=8 (1,956 far-field quartets), this reduces
the per-iteration far-field contract time from ~0.86s to ~0.02s
(40× speedup).

Memory: the kernel stores one (n1*n2, n3*n4) float matrix per
unique (R_sep, L_order) combination, plus index metadata.
For diamond: 7,542 unique tensors × ~36 floats each × 8 bytes
≈ 2 MB — negligible.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .bipole_multipole import n_components as _n_sph
from .bipole_spherical_moment_buffer import SphericalMomentBuffer
from .bipole_quartet_far_field import (
    QuartetBipolarDispatch,
    QuartetMultipoleFarField,
    quartet_effective_screening_parameter,
)

__all__ = [
    "FarFieldFockKernel",
    "build_far_field_fock_kernel",
    "apply_far_field_fock_kernel",
    "apply_far_field_fock_kernel_with_symmetry",
]


@dataclass
class FarFieldFockKernel:
    """Pre-computed contraction kernel for the dormant prototype.

    ``contributions`` is a list of (bra_key, bra_slice, ket_key, ket_slice,
    K_matrix, dispatch_index) tuples where ``K_matrix`` has shape (n1*n2, n3*n4)
    and ``F_bra[bra_slice] += K_matrix @ D_ket[ket_slice].flatten()``.

    ``dispatch_index`` is the index into the original :class:`QuartetBipolarDispatch`
    that this contribution was built from, needed for symmetry-reconstruction alignment.
    """

    contributions: List[Tuple] = field(default_factory=list)
    nbf: int = 0

    def __len__(self) -> int:
        return len(self.contributions)


def build_far_field_fock_kernel(
    moment_buffer: SphericalMomentBuffer,
    dispatch: QuartetBipolarDispatch,
    *,
    ewald_omega: float = 0.0,
    nbf: int = 0,
    tensor_cache: Optional["QuartetTensorCache"] = None,
) -> FarFieldFockKernel:
    """Pre-compute the prototype far-field Fock kernel for a geometry.

    The kernel encodes: for each far-field quartet, the effective
    linear operator that maps the ket density block to the bra Fock
    block.  This kernel is geometry-dependent but DENSITY-INDEPENDENT,
    so it's computed once and reused across all SCF iterations.

    Returns a :class:`FarFieldFockKernel` that can be applied to any
    density matrix via :func:`apply_far_field_fock_kernel`.
    """
    from .bipole_multipole import (
        multipole_interaction_tensor,
        sr_multipole_interaction_tensor,
    )

    kernel = FarFieldFockKernel(nbf=nbf)
    slices = moment_buffer.shell_slices

    # Group by (R_key, L_order) for tensor reuse.
    groups: Dict[Tuple, List] = defaultdict(list)
    for q in range(len(dispatch)):
        L_order = dispatch.truncation_orders[q]
        if L_order <= 0:
            continue
        s1, s2, bra_cell = dispatch.bra_pairs[q]
        s3, s4, ket_cell = dispatch.ket_pairs[q]
        centre_bra = moment_buffer.get_center(s1, s2, bra_cell)
        centre_ket = moment_buffer.get_center(s3, s4, ket_cell)
        if centre_bra is None or centre_ket is None:
            continue
        R_sep = centre_ket - centre_bra
        if float(np.dot(R_sep, R_sep)) < 1e-30:
            continue
        key = (
            round(float(R_sep[0]), 5),
            round(float(R_sep[1]), 5),
            round(float(R_sep[2]), 5),
            int(L_order),
        )
        groups[key].append((q, s1, s2, bra_cell, s3, s4, ket_cell))

    for (rx, ry, rz, L_order), quartet_list in groups.items():
        R_sep = np.array([rx, ry, rz], dtype=float)
        T_mat = None
        if tensor_cache is not None:
            T_mat = tensor_cache.get(R_sep, L_order)
        if T_mat is None:
            if ewald_omega > 0:
                q0 = quartet_list[0][0]
                gamma_bra = dispatch.bra_widths[q0]
                gamma_ket = dispatch.ket_widths[q0]
                mu_eff = quartet_effective_screening_parameter(
                    gamma_bra, gamma_ket, ewald_omega,
                )
                # Prefer C++ erfc tensor if available.
                try:
                    from ._vibeqc_core import multipole_erfc_interaction_tensor
                except ImportError:
                    multipole_erfc_interaction_tensor = None
                if multipole_erfc_interaction_tensor is not None:
                    T_mat = np.asarray(multipole_erfc_interaction_tensor(
                        L_order, L_order,
                        float(R_sep[0]), float(R_sep[1]), float(R_sep[2]),
                        float(mu_eff),
                    ))
                else:
                    T_mat = sr_multipole_interaction_tensor(
                        L_order, L_order, R_sep, mu_eff,
                    )
            else:
                # Prefer C++ bare tensor.
                try:
                    from ._vibeqc_core import multipole_interaction_tensor as _cpp_T
                except ImportError:
                    _cpp_T = None
                if _cpp_T is not None:
                    T_mat = np.asarray(_cpp_T(
                        L_order, L_order,
                        float(R_sep[0]), float(R_sep[1]), float(R_sep[2]),
                    ))
                else:
                    T_mat = multipole_interaction_tensor(
                        L_order, L_order, R_sep,
                    )

        n_keep = _n_sph(L_order)

        # Collect per-quartet moment data for batched C++ matmul.
        bra_moments_group = []
        ket_moments_group = []
        quartet_metadata = []  # (bra_key, bra_sl, ket_key, ket_sl, q_idx)

        for q_idx, s1, s2, bra_cell, s3, s4, ket_cell in quartet_list:
            mom_bra = moment_buffer.get_moments(s1, s2, bra_cell)
            mom_ket = moment_buffer.get_moments(s3, s4, ket_cell)
            if mom_bra is None or mom_ket is None:
                continue

            b1, n1 = slices[s1]
            b2, n2 = slices[s2]
            b3, n3 = slices[s3]
            b4, n4 = slices[s4]

            bra_cell_obj = moment_buffer.cells[bra_cell]
            bra_key = (
                bra_cell_obj.index[0],
                bra_cell_obj.index[1],
                bra_cell_obj.index[2],
            )
            ket_cell_obj = moment_buffer.cells[ket_cell]
            ket_key = (
                ket_cell_obj.index[0],
                ket_cell_obj.index[1],
                ket_cell_obj.index[2],
            )

            mom_bra_q = np.asarray(mom_bra[:, :, :n_keep], dtype=float)
            mom_ket_q = np.asarray(mom_ket[:, :, :n_keep], dtype=float)
            mom_bra_2d = mom_bra_q.reshape(n1 * n2, n_keep)
            mom_ket_2d = mom_ket_q.reshape(n3 * n4, n_keep)

            bra_moments_group.append(mom_bra_2d)
            ket_moments_group.append(mom_ket_2d)
            quartet_metadata.append((
                bra_key,
                (b1, b1 + n1, b2, b2 + n2),
                ket_key,
                (b3, b3 + n3, b4, b4 + n4),
                q_idx,
            ))

        # Compute all K matrices at once with C++.
        try:
            from ._vibeqc_core import build_far_field_k_matrices_batch
        except ImportError:
            build_far_field_k_matrices_batch = None
        if build_far_field_k_matrices_batch is not None:
            K_mats = build_far_field_k_matrices_batch(
                bra_moments_group, ket_moments_group,
                np.asarray(T_mat, dtype=float),
            )
            for K_ij, (bra_key, bra_sl, ket_key, ket_sl, q_idx) in zip(
                K_mats, quartet_metadata,
            ):
                kernel.contributions.append((
                    bra_key, bra_sl, ket_key, ket_sl,
                    np.asarray(K_ij, dtype=float),
                    q_idx,  # dispatch index for reconstruction alignment
                ))
        else:
            # Fallback: per-quartet numpy matmul.
            for (bra_key, bra_sl, ket_key, ket_sl, q_idx), mom_bra_2d, mom_ket_2d in zip(
                quartet_metadata, bra_moments_group, ket_moments_group,
            ):
                K_mat = mom_bra_2d @ T_mat @ mom_ket_2d.T
                kernel.contributions.append((
                    bra_key, bra_sl, ket_key, ket_sl, K_mat, q_idx,
                ))

    return kernel


def apply_far_field_fock_kernel(
    kernel: FarFieldFockKernel,
    density_blocks: Dict[Tuple[int, int, int], np.ndarray],
) -> QuartetMultipoleFarField:
    """Apply the pre-computed kernel to a density matrix.

    This is the per-iteration step: for each far-field quartet,
    contract the pre-computed kernel with the current density and
    accumulate into the Fock blocks.

    Uses the C++ OpenMP-parallel path when available; falls back
    to a pure-Python loop otherwise.

    Returns the far-field Fock contribution and Coulomb energy.
    """
    # Try C++ native path first.
    try:
        return _apply_far_field_fock_kernel_native(kernel, density_blocks)
    except ImportError:
        pass

    # Pure-Python fallback.
    result = QuartetMultipoleFarField()

    for bra_key, bra_sl, ket_key, ket_sl, K_mat, *_ in kernel.contributions:
        D_ket = density_blocks.get(ket_key)
        if D_ket is None:
            continue
        b1, b1e, b2, b2e = bra_sl
        b3, b3e, b4, b4e = ket_sl
        D_sub = np.asarray(
            D_ket[b3:b3e, b4:b4e], dtype=float,
        )
        if np.max(np.abs(D_sub)) < 1e-30:
            continue

        # F_bra[bra_sl] += K_mat @ D_sub.flatten()
        D_flat = D_sub.reshape(-1)
        F_contrib = K_mat @ D_flat  # (n1*n2,)

        if bra_key not in result.fock_blocks:
            result.fock_blocks[bra_key] = np.zeros(
                (kernel.nbf, kernel.nbf), dtype=float,
            )
        F_bra = result.fock_blocks[bra_key]
        n1 = b1e - b1
        n2 = b2e - b2
        F_bra[b1:b1e, b2:b2e] += F_contrib.reshape(n1, n2)

        # Track energy.
        D_bra_sub = np.asarray(
            density_blocks.get(
                bra_key,
                np.zeros((kernel.nbf, kernel.nbf), dtype=float),
            )[b1:b1e, b2:b2e],
            dtype=float,
        )
        result.e_coulomb_far += 0.5 * float(
            np.sum(D_bra_sub * F_contrib.reshape(n1, n2))
        )
        result.n_quartets += 1

    return result


def _apply_far_field_fock_kernel_native(
    kernel: FarFieldFockKernel,
    density_blocks: Dict[Tuple[int, int, int], np.ndarray],
) -> QuartetMultipoleFarField:
    """Apply the kernel using the C++ OpenMP-parallel path.

    Converts the stored row-major AO-pair indices to the native contractor's
    column-major pair indices and calls the existing native implementation.
    The stored kernel and the Python unreduced contraction are unchanged.
    """
    from ._vibeqc_core import (
        FarFieldFockKernelEntry,
        apply_far_field_fock_kernel as _cpp_apply,
    )

    if not kernel.contributions:
        return QuartetMultipoleFarField()

    # Build typed C++ entries.
    cpp_entries = []
    for bra_key, bra_sl, ket_key, ket_sl, K_mat, *_ in kernel.contributions:
        e = FarFieldFockKernelEntry()
        e.bra_cell_ix = int(bra_key[0])
        e.bra_cell_iy = int(bra_key[1])
        e.bra_cell_iz = int(bra_key[2])
        e.bra_b1 = int(bra_sl[0])
        e.bra_b1e = int(bra_sl[1])
        e.bra_b2 = int(bra_sl[2])
        e.bra_b2e = int(bra_sl[3])
        e.ket_cell_ix = int(ket_key[0])
        e.ket_cell_iy = int(ket_key[1])
        e.ket_cell_iz = int(ket_key[2])
        e.ket_b3 = int(ket_sl[0])
        e.ket_b3e = int(ket_sl[1])
        e.ket_b4 = int(ket_sl[2])
        e.ket_b4e = int(ket_sl[3])
        # Stored/Python pair indices are row-major: (mu, nu) -> mu*n_nu+nu.
        # Eigen's contractor packs density and Fock column-major. Translate
        # both pair axes here; changing only array storage order is insufficient.
        n1, n2 = e.bra_b1e-e.bra_b1, e.bra_b2e-e.bra_b2
        n3, n4 = e.ket_b3e-e.ket_b3, e.ket_b4e-e.ket_b4
        matrix = np.asarray(K_mat, dtype=float)
        if min(n1, n2, n3, n4) <= 0 or matrix.shape != (n1*n2, n3*n4):
            raise ValueError("far-field kernel matrix disagrees with AO-pair dimensions")
        e.K_matrix = (
            matrix.reshape(n1, n2, n3, n4)
            .transpose(1, 0, 3, 2).reshape(n1*n2, n3*n4)
        )
        cpp_entries.append(e)

    # Pack density blocks into parallel lists.
    density_cell_keys_flat = []
    density_matrices = []
    for (ix, iy, iz), D in sorted(density_blocks.items()):
        density_cell_keys_flat.extend([int(ix), int(iy), int(iz)])
        density_matrices.append(np.asarray(D, dtype=float))

    # Call C++ OpenMP kernel apply.
    cpp_result = _cpp_apply(
        cpp_entries,
        density_cell_keys_flat,
        density_matrices,
        kernel.nbf,
    )

    # Convert C++ result back to Python format.
    result = QuartetMultipoleFarField()
    for ix, iy, iz, F_mat in zip(
        cpp_result.cell_ix,
        cpp_result.cell_iy,
        cpp_result.cell_iz,
        cpp_result.fock_matrices,
    ):
        result.fock_blocks[(int(ix), int(iy), int(iz))] = (
            np.asarray(F_mat, dtype=float)
        )
    result.e_coulomb_far = float(cpp_result.coulomb_energy_far)
    result.n_quartets = int(cpp_result.n_quartets_applied)

    return result


# ---------------------------------------------------------------------------
# Dormant symmetry adapter: checked identity execution only
# ---------------------------------------------------------------------------


def apply_far_field_fock_kernel_with_symmetry(
    kernel: FarFieldFockKernel,
    reconstruction: Any,  # SymmetryFockReconstructionMap
    density_blocks: Dict[Tuple[int, int, int], np.ndarray],
) -> QuartetMultipoleFarField:
    """Apply an already-unreduced kernel with checked identity metadata.

    The dormant prototype has no qualified operator or tensor transport for
    nonidentity orbits. Shell dimensions and an orbit inventory cannot prove
    that a canonical contraction can be reused at another shell/cell position
    or for another density block. Refuse that execution before contracting.

    Identity metadata leaves the stored kernel's operator unchanged and uses
    the ordinary unreduced contractor. This does not reconstruct entries
    omitted by a reduced kernel builder or qualify the multipole source.
    """
    from .bipole_symmetry_dispatch import SymmetryFockReconstructionMap

    if not isinstance(reconstruction, SymmetryFockReconstructionMap):
        raise TypeError("far-field reconstruction requires typed orbit metadata")
    count = reconstruction.n_sym_operations
    if (isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer)) or count < 1):
        raise ValueError("far-field reconstruction operation count is invalid")
    if count != 1:
        raise NotImplementedError(
            "Far-field symmetry reconstruction is unqualified; use the unreduced kernel."
        )
    for orbit in reconstruction.orbit_entries:
        if not orbit:
            raise ValueError("far-field reconstruction has an empty orbit")
        if len(orbit) != 1:
            raise NotImplementedError(
                "Far-field symmetry reconstruction is unqualified; use the unreduced kernel."
            )
    # Validate every mapping before producing any Fock contribution. In
    # particular, equal block dimensions do not identify a transpose action.
    for entry in kernel.contributions:
        if len(entry) < 6:
            raise ValueError("far-field identity reconstruction requires dispatch indices")
        index = entry[5]
        if (isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or not 0 <= index < len(reconstruction)):
            raise ValueError("far-field identity reconstruction has an invalid dispatch index")
        if reconstruction.orbit_entries[index][0] != entry[:4]:
            raise ValueError("far-field identity orbit differs from stored kernel positions")
    return apply_far_field_fock_kernel(kernel, density_blocks)
