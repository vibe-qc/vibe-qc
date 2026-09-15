"""Bravais-lattice geometry utilities for the bipolar far-field pipeline.

Provides dimension-aware helpers for cell volume, inverse length scale,
and lattice-basis rotation that work for all 14 Bravais lattices in 3D
as well as 1D chains and 2D slabs.

All functions accept either a PeriodicSystem (which carries the ``dim``,
``lattice``, and ``unit_cell_molecule`` attributes) or raw lattice
vectors, and return the appropriate dimension-generalised result.

References
----------
International Tables for Crystallography, Vol. A (any edition).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


def cell_dimensionality(system_or_lattice) -> int:
    """Return the periodic dimensionality (1, 2, or 3).

    Accepts either a PeriodicSystem object or a raw (dim, lattice) tuple.
    """
    if hasattr(system_or_lattice, "dim"):
        return int(system_or_lattice.dim)
    # Assume 3D if we can't determine.
    lattice = np.asarray(system_or_lattice, dtype=float)
    if lattice.ndim == 2 and lattice.shape[0] == lattice.shape[1]:
        return lattice.shape[0]
    return 3


def cell_volume_bohr(system_or_lattice) -> float:
    """Cell volume (bohr^3) for 3D; area (bohr^2) for 2D; length (bohr) for 1D.

    For 1D: returns the length of the single lattice vector.
    For 2D: returns |a × b| (area of the spanning parallelogram).
    For 3D: returns |det(lattice)|.
    """
    dim = cell_dimensionality(system_or_lattice)
    if hasattr(system_or_lattice, "lattice"):
        lattice = np.asarray(system_or_lattice.lattice, dtype=float)
    else:
        lattice = np.asarray(system_or_lattice, dtype=float)

    if dim == 1:
        # lattice is (1, 3) or (3,): length of the single vector
        vec = lattice.reshape(-1)[:3]
        return float(np.linalg.norm(vec))
    elif dim == 2:
        # lattice is (2, 3): area = |a × b|
        a = lattice[0, :3]
        b = lattice[1, :3]
        return float(np.linalg.norm(np.cross(a, b)))
    else:
        # dim == 3: lattice is (3, 3)
        return float(abs(np.linalg.det(lattice)))


def cell_length_scale_inv_bohr(system_or_lattice) -> float:
    """Inverse length scale ``(1 / measure)^{1/dim}`` for the BIPOLE
    prototype classifier.

    The classifier's use of this scale is implementation-specific and is
    not derived by Pisani-Dovesi-Roetti (1988) or Saunders (1992).

    For 3D: (1/V)^{1/3}
    For 2D: (1/A)^{1/2}
    For 1D: (1/L)
    """
    measure = cell_volume_bohr(system_or_lattice)
    dim = cell_dimensionality(system_or_lattice)
    if measure <= 0:
        raise ValueError(f"cell measure must be positive; got {measure}")
    if dim == 1:
        return 1.0 / measure
    elif dim == 2:
        return float(measure) ** (-0.5)
    else:
        return float(measure) ** (-1.0 / 3.0)


def lattice_to_cartesian_rotation(
    system, rotation_cart: np.ndarray
) -> np.ndarray:
    """Convert a 3×3 Cartesian rotation to lattice-basis integer matrix.

    ``R_lat = L^{-1} · R_cart · L`` where L is the lattice matrix.
    The result is rounded to the nearest integer (exact for symmorphic
    point-group operations).

    Works for 1D, 2D, and 3D by padding the lattice to 3×3 with
    identity on the transverse dimensions.
    """
    dim = cell_dimensionality(system)
    lattice = np.asarray(system.lattice, dtype=float)
    R = np.asarray(rotation_cart, dtype=float).reshape(3, 3)

    # Pad lattice to 3×3.
    if dim < 3:
        L3 = np.eye(3, dtype=float)
        L3[:dim, :3] = lattice.reshape(dim, 3)[:, :3]
    else:
        L3 = lattice.reshape(3, 3)

    L3_inv = np.linalg.inv(L3)
    R_lat = np.round(L3_inv @ R @ L3).astype(int)
    return R_lat
