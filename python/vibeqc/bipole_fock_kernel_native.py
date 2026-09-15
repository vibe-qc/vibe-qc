"""Python adapter for the C++ Fock kernel builder.

Wraps :func:`_vibeqc_core.build_far_field_fock_kernel_cpp`.
Falls back to pure Python on error.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .bipole_far_field_kernel import FarFieldFockKernel
from .bipole_quartet_far_field import QuartetBipolarDispatch
from .bipole_spherical_moment_buffer import SphericalMomentBuffer

__all__ = ["build_far_field_fock_kernel_native"]


def build_far_field_fock_kernel_native(
    moment_buffer: SphericalMomentBuffer,
    dispatch: QuartetBipolarDispatch,
    *,
    ewald_omega: float = 0.0,
    nbf: int = 0,
) -> FarFieldFockKernel:
    """Build the Fock kernel using C++ grouping + batched matmul."""
    try:
        from ._vibeqc_core import build_far_field_fock_kernel_cpp
    except ImportError:
        from .bipole_far_field_kernel import build_far_field_fock_kernel as _py
        return _py(moment_buffer, dispatch, ewald_omega=ewald_omega, nbf=nbf)

    # Convert to C++ format using the contractor adapter.
    from .bipole_contractor_native import (
        _build_cpp_moment_buffer,
        _build_cpp_dispatch,
    )
    cpp_buf = _build_cpp_moment_buffer(moment_buffer, nbf)
    cpp_disp = _build_cpp_dispatch(dispatch)

    cpp_entries = build_far_field_fock_kernel_cpp(
        cpp_buf, cpp_disp, float(ewald_omega),
    )

    kernel = FarFieldFockKernel(nbf=nbf)
    for e in cpp_entries:
        kernel.contributions.append((
            (int(e.bra_cell_ix), int(e.bra_cell_iy), int(e.bra_cell_iz)),
            (int(e.bra_b1), int(e.bra_b1e), int(e.bra_b2), int(e.bra_b2e)),
            (int(e.ket_cell_ix), int(e.ket_cell_iy), int(e.ket_cell_iz)),
            (int(e.ket_b3), int(e.ket_b3e), int(e.ket_b4), int(e.ket_b4e)),
            np.asarray(e.K_matrix, dtype=float),
        ))
    return kernel
