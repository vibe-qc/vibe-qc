"""Python adapter for the C++ OpenMP far-field gradient contractor.

Wraps :func:`_vibeqc_core.compute_bipolar_far_field_gradient_cpp` with
Python-side data conversion from the existing :class:`SphericalMomentBuffer`
and :class:`QuartetBipolarDispatch` types.

The C++ contractor handles the dominant dT/dR terms (interaction-tensor
gradient).  The sub-dominant dM/dA terms are added in Python using the
precomputed moment_grads in the buffer.

Provenance
----------
Pisani, Dovesi, and Roetti (1988), Ch. II.4c,
doi:10.1007/978-3-642-93385-1, derives the periodic quartet expansion.
The native gradient contractor is an implementation prototype. Saunders
et al. (1992), Sec. 5.3, Eqs. (90)-(92), supports the radial derivative
recurrence but does not derive this full quartet-gradient algorithm.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .bipole_quartet_far_field import QuartetBipolarDispatch
from .bipole_spherical_moment_buffer import SphericalMomentBuffer

__all__ = ["compute_bipolar_far_field_gradient_native"]


def _build_cpp_moment_buffer_for_gradient(
    moment_buffer: SphericalMomentBuffer,
) -> "BipoleMomentBufferCpp":
    """Convert Python SphericalMomentBuffer to C++ BipoleMomentBufferCpp."""
    from ._vibeqc_core import BipoleMomentBufferCpp, BipolePairMomentEntry

    cpp_buf = BipoleMomentBufferCpp()
    cpp_buf.L_max = moment_buffer.L_max
    cpp_buf.n_sph = moment_buffer.n_sph

    cell_positions = []
    cell_indices_list = []
    for cell in moment_buffer.cells:
        cell_positions.append((
            float(cell.r_cart[0]),
            float(cell.r_cart[1]),
            float(cell.r_cart[2]),
        ))
        cell_indices_list.append((
            int(cell.index[0]),
            int(cell.index[1]),
            int(cell.index[2]),
        ))
    cpp_buf.cell_positions = cell_positions
    cpp_buf.cell_indices = cell_indices_list

    shell_slices_list = []
    for bf_offset, bf_count in moment_buffer.shell_slices:
        shell_slices_list.append((int(bf_offset), int(bf_count)))
    cpp_buf.shell_slices = shell_slices_list

    base = moment_buffer
    if hasattr(moment_buffer, '_base'):
        base = moment_buffer._base

    entries_list = []
    for (s1, s2, cell_idx), moments in base.blocks.items():
        centre = base.centers.get((s1, s2, cell_idx))
        if centre is None:
            continue
        entry = BipolePairMomentEntry()
        entry.shell_index_1 = int(s1)
        entry.shell_index_2 = int(s2)
        entry.cell_index = int(cell_idx)
        b1, n1 = moment_buffer.shell_slices[s1]
        b2, n2 = moment_buffer.shell_slices[s2]
        entry.bf_offset_1 = int(b1)
        entry.bf_count_1 = int(n1)
        entry.bf_offset_2 = int(b2)
        entry.bf_count_2 = int(n2)
        mom_arr = np.asarray(moments, dtype=float)
        entry.moments_flat = mom_arr.reshape(n1 * n2, moment_buffer.n_sph)
        entry.centre_x = float(centre[0])
        entry.centre_y = float(centre[1])
        entry.centre_z = float(centre[2])
        entries_list.append(entry)
    cpp_buf.entries = entries_list

    return cpp_buf


def compute_bipolar_far_field_gradient_native(
    moment_buffer: SphericalMomentBuffer,
    dispatch: QuartetBipolarDispatch,
    density_blocks: Dict[Tuple[int, int, int], np.ndarray],
    n_atoms: int,
    *,
    ewald_omega: float = 0.0,
    atom_to_shells: Optional[Dict[int, List[int]]] = None,
    include_moment_derivative: bool = True,
) -> np.ndarray:
    """Compute far-field nuclear gradient via C++ OpenMP + Python dM/dA.

    The dominant dT/dR terms are computed in C++ with OpenMP parallelism.
    The sub-dominant dM/dA terms are added in Python using the
    precomputed ``moment_buffer.moment_grads``.

    Falls back to the pure-Python path if the C++ extension is unavailable.
    """
    # ---- Try C++ dT/dR path -----------------------------------------------
    try:
        from ._vibeqc_core import (
            compute_bipolar_far_field_gradient_cpp,
            BipoleQuartetEntryCpp, DensityBlockCpp,
        )

        cpp_buf = _build_cpp_moment_buffer_for_gradient(moment_buffer)

        cpp_dispatch = []
        for q in range(len(dispatch)):
            s1, s2, bra_cell = dispatch.bra_pairs[q]
            s3, s4, ket_cell = dispatch.ket_pairs[q]
            L_order = dispatch.truncation_orders[q]
            if L_order <= 0:
                continue
            entry = BipoleQuartetEntryCpp()
            entry.shell_1 = int(s1)
            entry.shell_2 = int(s2)
            entry.cell_bra = int(bra_cell)
            entry.shell_3 = int(s3)
            entry.shell_4 = int(s4)
            entry.cell_ket = int(ket_cell)
            entry.truncation_order = int(L_order)
            entry.bra_width = float(dispatch.bra_widths[q])
            entry.ket_width = float(dispatch.ket_widths[q])
            cpp_dispatch.append(entry)

        cpp_density = []
        for (ix, iy, iz), D in density_blocks.items():
            db = DensityBlockCpp()
            db.cell_ix = int(ix)
            db.cell_iy = int(iy)
            db.cell_iz = int(iz)
            db.density = np.asarray(D, dtype=float)
            cpp_density.append(db)

        shell_to_atom_cpp = []
        if atom_to_shells is not None:
            n_sh = len(moment_buffer.shell_slices)
            shell_to_atom_cpp = [[] for _ in range(n_sh)]
            for sh_idx, atom_list in atom_to_shells.items():
                if sh_idx < n_sh:
                    shell_to_atom_cpp[sh_idx] = [int(a) for a in atom_list]

        grad = np.asarray(compute_bipolar_far_field_gradient_cpp(
            cpp_buf, cpp_dispatch, cpp_density,
            ewald_omega, n_atoms, shell_to_atom_cpp, False,
        ), dtype=float)

        # Add dM/dA terms in Python if requested.
        if include_moment_derivative and len(moment_buffer.moment_grads) > 0:
            from .bipole_far_field_gradient import _add_moment_derivative_gradient
            _add_moment_derivative_gradient(
                grad, moment_buffer, dispatch, density_blocks,
                n_atoms, ewald_omega, atom_to_shells,
            )
        return grad

    except (ImportError, TypeError, RuntimeError) as exc:
        pass  # Fall back to Python below.

    # ---- Python fallback ---------------------------------------------------
    from .bipole_far_field_gradient import (
        build_bipolar_far_field_gradient_contribution,
    )
    return build_bipolar_far_field_gradient_contribution(
        moment_buffer, dispatch, density_blocks, n_atoms,
        ewald_omega=ewald_omega,
        atom_to_shells=atom_to_shells,
        include_moment_derivative=include_moment_derivative,
    )
