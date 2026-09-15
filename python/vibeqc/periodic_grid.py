"""Periodic atomic quadrature with point-centered image neighborhoods.

For a grid point r, the partition includes every physical atom image B+g
within max(image_radius_bohr, 2*d_nearest) of r. Here d_nearest is the
shortest distance to any physical atom image. The normalized Becke (or
selected Stratmann) weight belongs to the owning home atom A when A is in
that set, and is zero otherwise. The same physical point sees the same
neighborhood under atom-image relabelling or a rigid translation. The
nearest-image term keeps vacuum regions covered for diffuse densities.

Grid points, raw atomic weights and ownership retain their atom-major
layout. Only the energy-quadrature weights carry the periodic partition.
Inner points within R/2 of their owner use a cached image atlas; outer
points use complete centered lattice bounds. The radius must be converged
with radial/angular resolution; it is not a total-energy tolerance.
Radius zero explicitly requests a molecular grid.

``extended_partition_atoms`` retains the diagnostic home-centered image
list API; the periodic grid evaluates point-centered neighborhoods.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ._vibeqc_core import (
    Grid,
    GridOptions,
    PeriodicSystem,
    build_periodic_point_grid as _cpp_build_periodic_point_grid,
)


__all__ = [
    "build_periodic_becke_grid",
    "extended_partition_atoms",
]


def _extended_partition_atom_data(
    system: PeriodicSystem,
    image_radius_bohr: float = 10.0,
) -> Tuple[List[np.ndarray], List[int]]:
    """Return paired home/image positions and atomic numbers.

    Both lists always begin with the home-cell atoms in the
    same order as ``system.unit_cell``; image atoms follow in
    enumeration order over (i, j, k) lattice shifts.

    Image atom inclusion criterion: a copy of home-cell atom A shifted
    by lattice vector ``g = i.a₁ + j.a₂ + k.a₃`` is included iff some
    home-cell atom B satisfies ``‖A_pos + g - B_pos‖ <=
    image_radius_bohr``. This keeps only atoms that can actually
    contribute to the Becke partition near home-cell grid points.
    """
    home_atoms = list(system.unit_cell)
    home_positions = np.array([a.xyz for a in home_atoms], dtype=float)
    home_numbers = [int(a.Z) for a in home_atoms]
    lattice = np.asarray(system.lattice, dtype=float)
    n_home = len(home_atoms)

    # If an image of A is within the cutoff of a home atom B, its lattice
    # shift t obeys ||t|| <= image_radius_bohr + ||A - B||.  For active
    # lattice matrix L (3 x dim), ||L n|| >= sigma_min(L) ||n||_2, so every
    # coefficient is bounded by (radius + max home-atom separation) /
    # sigma_min.  Unlike a bound based on the shortest column, this remains
    # complete when large integer multiples of skew vectors nearly cancel.
    # Inactive lattice columns are bookkeeping directions, never images.
    dim = int(system.dim)
    if dim < 1 or dim > 3:
        raise ValueError(
            "extended_partition_atoms: periodic dimension must be in 1..3; "
            f"got {dim}."
        )
    active_lattice = lattice[:, :dim]
    singular_values = np.linalg.svd(active_lattice, compute_uv=False)
    sigma_min = float(singular_values[-1])
    if not np.isfinite(sigma_min) or sigma_min <= 0.0:
        raise ValueError(
            "extended_partition_atoms: degenerate lattice "
            f"(active sigma_min = {sigma_min}); cannot build image atoms."
        )
    max_home_separation = 0.0
    for atom_idx in range(n_home):
        separations = np.linalg.norm(
            home_positions - home_positions[atom_idx], axis=1
        )
        max_home_separation = max(
            max_home_separation, float(np.max(separations, initial=0.0))
        )
    coefficient_radius = (
        image_radius_bohr + max_home_separation
    ) / sigma_min
    if not np.isfinite(coefficient_radius):
        raise ValueError(
            "extended_partition_atoms: image coefficient bound is not finite."
        )
    L_max = max(0, int(np.ceil(coefficient_radius)))
    axis_bounds = [L_max if axis < dim else 0 for axis in range(3)]

    out_positions: List[np.ndarray] = [pos.copy() for pos in home_positions]
    out_numbers: List[int] = list(home_numbers)
    # Image atoms -- start after the home cell.
    for i in range(-axis_bounds[0], axis_bounds[0] + 1):
        for j in range(-axis_bounds[1], axis_bounds[1] + 1):
            for k in range(-axis_bounds[2], axis_bounds[2] + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                shift = (
                    i * lattice[:, 0]
                    + j * lattice[:, 1]
                    + k * lattice[:, 2]
                )
                # Translate every home-cell atom; keep only those within
                # image_radius_bohr of any home-cell atom.
                for atom_idx in range(n_home):
                    new_pos = home_positions[atom_idx] + shift
                    # Distance to nearest home-cell atom.
                    diffs = home_positions - new_pos
                    d_min = float(np.min(np.linalg.norm(diffs, axis=1)))
                    if d_min <= image_radius_bohr:
                        out_positions.append(new_pos)
                        out_numbers.append(home_numbers[atom_idx])

    return out_positions, out_numbers


def extended_partition_atoms(
    system: PeriodicSystem,
    image_radius_bohr: float = 10.0,
) -> List[np.ndarray]:
    """Return home + image atom positions within the requested radius.

    The return type remains the historical position-only list. This diagnostic
    home-centered image list does not define the point neighborhoods used by
    :func:`build_periodic_becke_grid`.
    """
    positions, _ = _extended_partition_atom_data(system, image_radius_bohr)
    return positions


def build_periodic_becke_grid(
    system: PeriodicSystem,
    *,
    grid_options: Optional[GridOptions] = None,
    image_radius_bohr: float = 10.0,
) -> Grid:
    """Build a DFT integration grid with periodic Becke partition.

    Parameters
    ----------
    system
        :class:`PeriodicSystem` whose home-cell atoms will own all
        grid points.
    grid_options
        :class:`GridOptions` with the standard radial / angular /
        smoothing controls. Uses defaults when ``None``.
    image_radius_bohr
        Minimum physical atom-image neighborhood radius around each point
        (default 10 bohr). The radius expands to twice the nearest-image
        distance where needed to retain coverage through vacuum. Converge
        it for the chosen density and cell; it is not an atomic radial
        cutoff or a total-energy tolerance.
        Zero explicitly selects the molecular partition.

    Returns
    -------
    :class:`Grid` whose ``points``, ``weights``, and
    ``atom_of_point`` index into the home-cell atoms (same order as
    ``system.unit_cell``). Ready to feed into
    :func:`vibeqc.build_xc_periodic` for periodic DFT integration.

    Notes
    -----
    In a large vacuum cell, converged integrals of a localized density
    approach the molecular result. Raw atomic weights and grid points
    retain the molecular layout for every positive radius as well.
    """
    if grid_options is None:
        grid_options = GridOptions()
    if not np.isfinite(image_radius_bohr) or image_radius_bohr < 0:
        raise ValueError(
            f"build_periodic_becke_grid: image_radius_bohr must be finite and >= 0; "
            f"got {image_radius_bohr}"
        )
    return _cpp_build_periodic_point_grid(system, float(image_radius_bohr), grid_options)
