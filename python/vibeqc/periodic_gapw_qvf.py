"""QVF (Quantum Visualization Format) density emission for the GAPW route.

The molecular path uses :func:`vibeqc.output.formats.qvf.qvf_density_data`
which evaluates the converged AO density on a bounding-box uniform grid
built around the molecule.  The periodic GDF / BIPOLE path reuses the
same writer indirectly.  The GPW route already has its own native FFT
grid (the :class:`PlaneWaveGrid` carried on the SCF result), so for the
GPW path the natural sampling is *that* grid -- there is no need to build
a fresh bounding-box grid, and using the SCF grid means the density we
visualise is bit-for-bit the one the SCF iterated on.

This module exposes :func:`qvf_density_data_periodic` which collocates
the converged density onto ``result.grid`` and packages it into the
``{label: (data, origin, span)}`` dict shape that
:func:`vibeqc.output.formats.qvf.write_qvf` expects under its
``volume_data=`` kwarg.

The result is one ``volume.density`` section keyed by ``label`` (default
``"Electron density"``) -- the same kind string the molecular path emits,
so vibe-view treats it identically.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np

from .periodic_gapw_j import (
    GpwMultiKScfResult,
    GpwScfResult,
    collocate_density_on_grid,
)

__all__ = ["qvf_density_data_periodic"]


def qvf_density_data_periodic(
    result: Union[GpwScfResult, GpwMultiKScfResult],
    basis: Any,
    system: Any,
    *,
    label: str = "Electron density",
) -> dict[str, tuple]:
    """Evaluate the converged GAPW density on ``result.grid`` and
    return a dict suitable for ``write_qvf(..., volume_data=...)``.

    Parameters
    ----------
    result
        Converged GPW SCF result -- either single-k
        (:class:`GpwScfResult`) or multi-k
        (:class:`GpwMultiKScfResult`).  ``result.density`` is the AO
        density matrix that gets collocated; ``result.grid`` is the
        :class:`PlaneWaveGrid` used during the SCF.
    basis
        :class:`vibeqc.BasisSet` used in the SCF.  AO ordering must
        match ``result.density``.
    system
        :class:`PeriodicSystem`.  Only ``system.dim`` is consulted --
        the QVF volume grid is taken from ``result.grid``, which
        already carries the lattice.
    label
        Human-readable label for the density section.  Defaults to
        ``"Electron density"`` to match the molecular emitter.

    Returns
    -------
    dict
        ``{label: (data_3d, origin_3, span_3x3)}`` -- pass straight as
        ``volume_data=`` to :func:`vibeqc.output.formats.qvf.write_qvf`.

        * ``data_3d`` -- ``(nx, ny, nz)`` float32 electron density on
          the SCF FFT grid (``e / bohr^3``).
        * ``origin_3`` -- ``(3,)`` float64, in bohr.  The GPW grid is
          fractional-coordinate native and the first sample sits at
          fractional ``(0, 0, 0)``, i.e. Cartesian origin = ``[0, 0, 0]``
          in the cell's home corner.
        * ``span_3x3`` -- ``(3, 3)`` float64 voxel vectors in bohr.
          Row ``i`` is the per-voxel step along lattice axis ``i``:
          ``lattice[:, i] / n_i``.

    Raises
    ------
    ValueError
        If ``system.dim != 3``.  1D / 2D periodic emission is not
        defined for the QVF v1 ``volume.density`` section, which
        assumes a 3D voxel grid.
    """
    dim = getattr(system, "dim", None)
    if dim != 3:
        raise ValueError(
            "qvf_density_data_periodic requires a fully 3D periodic "
            f"system; got system.dim={dim!r}. QVF volume.density is "
            "defined for 3D grids only."
        )

    grid = result.grid
    D = np.asarray(result.density, dtype=float)
    rho = collocate_density_on_grid(basis, D, grid)
    data = np.asarray(rho, dtype=np.float32)

    # GPW grid is fractional-native, first sample at fractional origin.
    origin = np.zeros(3, dtype=np.float64)

    # Per-voxel step vectors.  ``lattice_bohr`` has lattice vectors as
    # *columns* (a_i = lattice[:, i]); the per-voxel step along axis i
    # is a_i / n_i.  Stacked as rows of a (3, 3) matrix to match the
    # ``voxel_vectors`` shape consumed by the QVF writer's
    # ``_grid_descriptor`` helper (row i = voxel_vector_i).
    lattice = np.asarray(grid.lattice_bohr, dtype=np.float64)
    ns = np.array([grid.nx, grid.ny, grid.nz], dtype=np.float64)
    span = (lattice / ns).T.astype(np.float64)

    return {label: (data, origin, span)}
