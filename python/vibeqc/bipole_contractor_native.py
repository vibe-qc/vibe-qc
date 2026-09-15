"""Python adapter for the C++ direct far-field Coulomb contractor.

Wraps :func:`_vibeqc_core.compute_bipolar_coulomb_far_field_cpp` with
Python-side data conversion from the existing :class:`SphericalMomentBuffer`
and :class:`QuartetBipolarDispatch` types.

The C++ contractor replaces :func:`bipole_quartet_far_field.build_bipolar_coulomb_far_field`
and runs the full per-quartet contraction loop in C++ with OpenMP parallelism
and per-thread interaction-tensor caching.

Provenance
----------
Pisani, Dovesi, and Roetti (1988), Ch. II.4c,
doi:10.1007/978-3-642-93385-1, derives the periodic quartet expansion.
The contractor, indexing, and cache strategy wrapped here are
implementation prototypes, not algorithms derived in that source.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .bipole_quartet_far_field import QuartetBipolarDispatch, QuartetMultipoleFarField
from .bipole_spherical_moment_buffer import SphericalMomentBuffer

__all__ = ["compute_bipolar_coulomb_far_field_native"]


def _build_cpp_moment_buffer(
    moment_buffer: SphericalMomentBuffer,
    nbf: int,
) -> "BipoleMomentBufferCpp":
    """Convert a Python SphericalMomentBuffer to C++ BipoleMomentBufferCpp."""
    from ._vibeqc_core import BipoleMomentBufferCpp, BipolePairMomentEntry

    cpp_buf = BipoleMomentBufferCpp()
    cpp_buf.L_max = moment_buffer.L_max
    cpp_buf.n_sph = moment_buffer.n_sph

    # Cell positions and indices.
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

    # Shell slices.
    shell_slices_list = []
    for bf_offset, bf_count in moment_buffer.shell_slices:
        shell_slices_list.append((int(bf_offset), int(bf_count)))
    cpp_buf.shell_slices = shell_slices_list

    # Per-pair moment entries.
    entries_list = []
    # Handle both raw SphericalMomentBuffer (has .blocks) and
    # SymmetryReducedMomentBuffer (wraps ._base with .blocks).
    if hasattr(moment_buffer, '_base'):
        base = moment_buffer._base
    else:
        base = moment_buffer
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
        # Flatten moments: (n1, n2, n_sph) → (n1*n2, n_sph)
        mom_arr = np.asarray(moments, dtype=float)
        entry.moments_flat = mom_arr.reshape(n1 * n2, moment_buffer.n_sph)
        entry.centre_x = float(centre[0])
        entry.centre_y = float(centre[1])
        entry.centre_z = float(centre[2])
        entries_list.append(entry)
    cpp_buf.entries = entries_list

    return cpp_buf


def _build_cpp_dispatch(
    dispatch: QuartetBipolarDispatch,
) -> List["BipoleQuartetEntryCpp"]:
    """Convert a Python QuartetBipolarDispatch to C++ quartet entries."""
    from ._vibeqc_core import BipoleQuartetEntryCpp

    cpp_entries = []
    for q in range(len(dispatch)):
        s1, s2, cell_bra = dispatch.bra_pairs[q]
        s3, s4, cell_ket = dispatch.ket_pairs[q]
        L_order = dispatch.truncation_orders[q]
        if L_order <= 0:
            continue
        entry = BipoleQuartetEntryCpp()
        entry.shell_1 = int(s1)
        entry.shell_2 = int(s2)
        entry.cell_bra = int(cell_bra)
        entry.shell_3 = int(s3)
        entry.shell_4 = int(s4)
        entry.cell_ket = int(cell_ket)
        entry.truncation_order = int(L_order)
        entry.bra_width = float(dispatch.bra_widths[q])
        entry.ket_width = float(dispatch.ket_widths[q])
        cpp_entries.append(entry)
    return cpp_entries


def _build_cpp_density(
    density_blocks: Dict[Tuple[int, int, int], np.ndarray],
) -> List["DensityBlockCpp"]:
    """Convert Python density dict to C++ density block list."""
    from ._vibeqc_core import DensityBlockCpp

    cpp_blocks = []
    for (ix, iy, iz), D in density_blocks.items():
        db = DensityBlockCpp()
        db.cell_ix = int(ix)
        db.cell_iy = int(iy)
        db.cell_iz = int(iz)
        db.density = np.asarray(D, dtype=float)
        cpp_blocks.append(db)
    return cpp_blocks


def compute_bipolar_coulomb_far_field_native(
    moment_buffer: SphericalMomentBuffer,
    dispatch: QuartetBipolarDispatch,
    density_blocks: Dict[Tuple[int, int, int], np.ndarray],
    *,
    ewald_omega: float = 0.0,
    nbf: int = 0,
) -> QuartetMultipoleFarField:
    """Compute Coulomb far-field Fock contribution using the C++ OpenMP contractor.

    This is the high-performance path: converts the Python-format input
    to C++ typed structs, calls :func:`_vibeqc_core.compute_bipolar_coulomb_far_field_cpp`,
    and converts the result back to a :class:`QuartetMultipoleFarField`.

    Falls back to the pure-Python contractor if the C++ path is not available.
    """
    try:
        from ._vibeqc_core import compute_bipolar_coulomb_far_field_cpp
    except ImportError:
        # C++ extension not available; fall back to Python.
        from .bipole_quartet_far_field import build_bipolar_coulomb_far_field
        return build_bipolar_coulomb_far_field(
            moment_buffer, dispatch, density_blocks,
            ewald_omega=ewald_omega, nbf=nbf,
        )

    # Convert inputs to C++ structs.
    cpp_buf = _build_cpp_moment_buffer(moment_buffer, nbf)
    cpp_dispatch = _build_cpp_dispatch(dispatch)
    cpp_density = _build_cpp_density(density_blocks)

    # Call C++ OpenMP contractor.
    cpp_result = compute_bipolar_coulomb_far_field_cpp(
        cpp_buf, cpp_dispatch, cpp_density,
        float(ewald_omega), int(nbf),
    )

    # Convert result back to Python QuartetMultipoleFarField.
    result = QuartetMultipoleFarField()
    result.e_coulomb_far = float(cpp_result.coulomb_energy_far)
    result.n_quartets = int(cpp_result.quartets_processed)
    for i in range(len(cpp_result.fock_cell_ix)):
        key = (
            int(cpp_result.fock_cell_ix[i]),
            int(cpp_result.fock_cell_iy[i]),
            int(cpp_result.fock_cell_iz[i]),
        )
        result.fock_blocks[key] = np.asarray(
            cpp_result.fock_blocks[i], dtype=float,
        )
    return result
