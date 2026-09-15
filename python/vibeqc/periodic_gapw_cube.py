"""Gaussian-cube writer for the periodic GPW route.

The molecular cube writer at :mod:`vibeqc.cube` builds an
axis-aligned bounding box around a :class:`Molecule` with a
Becke-style padding heuristic. The periodic GPW route already
has a natural grid -- the :class:`PlaneWaveGrid` the J build and
V_xc projection live on -- and a natural box -- the
:class:`PeriodicSystem` lattice. This module emits a Gaussian
cube whose voxel layout is *exactly* that grid: origin at the
lattice origin, three voxel vectors equal to the per-axis
lattice step ``A[:, i] / n_i``, atoms placed in their unit-cell
positions in bohr.

The data block reuses the GPW collocator
(:func:`vibeqc.periodic_gapw_j.collocate_density_on_grid`) so
the ``(nx, ny, nz)`` array that lands in the cube file is the
same one the SCF saw -- no interpolation, no resampling.

Layout (Gaussian-98 cube spec, see :mod:`vibeqc.cube`):

    line 1  : title comment
    line 2  : property comment
    line 3  : N_atoms  x_origin  y_origin  z_origin
    line 4  : N_x      vx_x      vx_y      vx_z   (voxel along a₁)
    line 5  : N_y      vy_x      vy_y      vy_z   (voxel along a₂)
    line 6  : N_z      vz_x      vz_y      vz_z   (voxel along a₃)
    line 7+ : Z  charge  x  y  z   (one line per atom in the unit cell)
    then    : data, scanned x outer, z inner, 6 floats per line

All coordinates and voxel vectors are in **bohr** -- that's the
cube spec and what the GPW route already speaks internally.

Public API
----------

* :func:`write_cube_density_periodic` -- total electron density
  r(r) on the GPW grid, written from a converged
  :class:`GpwScfResult` (Γ-only) or
  :class:`GpwMultiKScfResult` (multi-k) plus the basis the
  density was built on plus the
  :class:`PeriodicSystem` it lives in.

Only 3D systems are supported (``system.dim == 3``); 1D / 2D
periodic dimensions don't have a well-defined cube box and
raise a :class:`ValueError`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from .periodic_gapw_j import (
    GpwMultiKScfResult,
    GpwScfResult,
    collocate_density_on_grid,
)


__all__ = ["write_cube_density_periodic"]


def _format_atoms_block(system) -> str:
    """One line per unit-cell atom in the cube `Z charge x y z` form.

    Positions are taken from ``system.unit_cell`` (already in bohr,
    matching the lattice convention the GPW grid uses) and emitted
    unwrapped. Viewers that read cube files re-wrap atoms by the
    voxel vectors themselves, so we don't pre-fold here.
    """
    lines = []
    for atom in system.unit_cell:
        x, y, z = atom.xyz
        lines.append(
            f"{int(atom.Z):5d} {float(atom.Z):12.6f} "
            f"{float(x):12.6f} {float(y):12.6f} {float(z):12.6f}"
        )
    return "\n".join(lines)


def _write_cube_header(
    out,
    *,
    title: str,
    comment: str,
    system,
    grid,
) -> None:
    """Write the 6-line header + atom block.

    The cube origin is the lattice origin (s = 0 in fractional
    coords, i.e. the (0, 0, 0) voxel of the PlaneWaveGrid). The
    three voxel vectors are the per-axis lattice steps
    ``A[:, i] / n_i`` -- preserving any non-orthogonality of the
    cell (the cube spec carries full x/y/z components per voxel
    line, viewers like VMD honour them).
    """
    A = np.asarray(grid.lattice_bohr, dtype=float)
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    out.write(f"{title}\n")
    out.write(f"{comment}\n")

    n_atoms = len(system.unit_cell)
    # Origin at the lattice origin -- the PlaneWaveGrid samples
    # s = (i/nx, j/ny, k/nz) so voxel (0, 0, 0) lives at r = 0.
    out.write(
        f"{n_atoms:5d}     0.000000     0.000000     0.000000\n"
    )

    # Voxel along a₁: A[:, 0] / nx. Same for a₂, a₃.
    v1 = A[:, 0] / nx
    v2 = A[:, 1] / ny
    v3 = A[:, 2] / nz
    out.write(
        f"{nx:5d} {v1[0]:12.6f} {v1[1]:12.6f} {v1[2]:12.6f}\n"
    )
    out.write(
        f"{ny:5d} {v2[0]:12.6f} {v2[1]:12.6f} {v2[2]:12.6f}\n"
    )
    out.write(
        f"{nz:5d} {v3[0]:12.6f} {v3[1]:12.6f} {v3[2]:12.6f}\n"
    )

    out.write(_format_atoms_block(system))
    out.write("\n")


def _write_cube_data(out, data: np.ndarray) -> None:
    """Write a ``(nx, ny, nz)`` array in cube order -- x outer, z
    inner, 6 floats per line in scientific notation."""
    flat = np.ascontiguousarray(data).reshape(-1)
    for i in range(0, flat.size, 6):
        chunk = flat[i:i + 6]
        out.write(" ".join(f"{x:13.5e}" for x in chunk) + "\n")


def write_cube_density_periodic(
    path: Union[str, Path],
    result: Union[GpwScfResult, GpwMultiKScfResult],
    basis,
    system,
    *,
    title: str = "vibe-qc periodic electron density (GPW)",
    comment: str = "rho(r) in e/bohr^3 on the GPW grid",
) -> Path:
    """Write the converged periodic electron density to a Gaussian
    cube file.

    The voxel grid is *exactly* the :class:`PlaneWaveGrid` carried
    on ``result`` -- the same grid the GPW SCF used for the
    Hartree-J build and (on the RKS path) the V_xc evaluation.
    No resampling: the file's ``(nx, ny, nz)`` data block is
    ``collocate_density_on_grid(basis, result.density, result.grid)``
    in cube ordering.

    Parameters
    ----------
    path
        Destination cube file path.
    result
        Converged :class:`GpwScfResult` (Γ-only RHF / RKS) or
        :class:`GpwMultiKScfResult` (multi-k RKS). The total
        AO density and the GPW grid are taken from this object.
    basis
        :class:`vibeqc.BasisSet` the density was derived from.
        AO ordering must match -- same object the SCF used.
    system
        :class:`vibeqc.PeriodicSystem` the SCF ran on. Provides
        the unit-cell atoms for the cube atom block. Must be 3D
        (``system.dim == 3``); lower dimensionalities don't have
        a well-defined cube box.

    Returns
    -------
    pathlib.Path
        The path the cube file was written to.

    Raises
    ------
    ValueError
        If ``system.dim != 3``.
    """
    if int(getattr(system, "dim", 0)) != 3:
        raise ValueError(
            f"write_cube_density_periodic: only 3D periodic systems "
            f"are supported (got dim={getattr(system, 'dim', None)!r}); "
            f"the cube format requires three voxel vectors."
        )

    grid = result.grid
    D = np.asarray(result.density, dtype=float)
    rho = collocate_density_on_grid(basis, D, grid)

    p = Path(path)
    with p.open("w") as out:
        _write_cube_header(
            out,
            title=title,
            comment=comment,
            system=system,
            grid=grid,
        )
        _write_cube_data(out, rho)
    return p
