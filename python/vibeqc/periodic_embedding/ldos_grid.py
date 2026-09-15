"""LDOS grid computation and persistence for embedded surfaces.

Computes the local density of states on a real-space grid over region I
and saves it as a portable archive for 3D visualization.

The LDOS at grid point r and energy E is:

    rho(r, E) = -(1/pi) Im sum_{mu,nu} chi_mu(r) G_{mu,nu}(E+ieta) chi_nu(r)

where G is the region-I Green function and chi are the AO basis functions
evaluated on the grid.

Output format: a ``{stem}.eldos.npz`` NumPy archive with keys:
- ``ldos``: float32 array (n_energies, n_x, n_y, n_z)
- ``energies``: the real energies (Ha)
- ``extent``: (x_min, x_max, y_min, y_max, z_min, z_max) in bohr
- ``origin``: (x0, y0, z0) grid origin
- ``spacing``: (dx, dy, dz) grid spacing
- ``eta``: broadening parameter (Ha)
- ``metadata``: dict with contour, k-mesh, and embedding info
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .._vibeqc_core import BasisSet, evaluate_ao
from .region import RegionPartition
from .scf2step import compute_region_i_gf_at_kz


def compute_ldos_grid(
    system,
    basis: BasisSet,
    region: RegionPartition,
    g_fn,  # callable: complex z -> G_II(z) as (n_i, n_i)
    energies: NDArray[np.float64],
    *,
    margin: float = 3.0,
    spacing: float = 0.3,
    eta: float = 0.05,
    lat_opts=None,
) -> dict:
    """Compute LDOS on a uniform grid over region-I atoms.

    Parameters
    ----------
    system
        Periodic system for AO evaluation.
    basis
        Orbital basis set.
    region
        Region partition defining the AO subspace.
    g_fn
        Callable ``z -> G_II(z)`` returning the (n_i, n_i) GF.
    energies
        Real energies at which to evaluate the LDOS (Ha).
    margin
        Extra padding around the atom bounding box (bohr).
    spacing
        Grid spacing in bohr.
    eta
        Imaginary broadening (Ha).
    lat_opts
        Unused; kept for signature compatibility.

    Returns
    -------
    dict
        With keys ``ldos`` (float32), ``energies``, ``extent``, ``origin``,
        ``spacing``, ``eta``.
    """
    # Determine grid extent from region-I atom positions.
    i_atoms = region.i_atoms
    positions = np.array([system.unit_cell[a].xyz for a in i_atoms])
    xyz_min = positions.min(axis=0) - margin
    xyz_max = positions.max(axis=0) + margin

    nx = max(2, int(np.ceil((xyz_max[0] - xyz_min[0]) / spacing)))
    ny = max(2, int(np.ceil((xyz_max[1] - xyz_min[1]) / spacing)))
    nz = max(2, int(np.ceil((xyz_max[2] - xyz_min[2]) / spacing)))

    xs = np.linspace(xyz_min[0], xyz_max[0], nx)
    ys = np.linspace(xyz_min[1], xyz_max[1], ny)
    zs = np.linspace(xyz_min[2], xyz_max[2], nz)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_points = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    # Evaluate AOs on the grid (full basis, then restrict).
    chi_full = np.asarray(evaluate_ao(basis, grid_points), dtype=float)
    chi = chi_full[:, region.i_ao]  # (n_points, n_i)

    n_energies = len(energies)
    ldos = np.zeros((n_energies, nx, ny, nz), dtype=np.float32)
    inv_pi = -1.0 / np.pi

    for ie, e_real in enumerate(energies):
        z = float(e_real) + 1j * float(eta)
        G = np.asarray(g_fn(z), dtype=np.complex128)
        # rho_g = -(1/pi) Im sum_{mu,nu} chi_{g,mu} G_{mu,nu} chi_{g,nu}
        rho_g = inv_pi * np.imag(np.einsum("gi,ij,gj->g", chi, G, chi))
        ldos[ie] = rho_g.reshape(nx, ny, nz)

    return {
        "ldos": ldos,
        "energies": np.asarray(energies, dtype=np.float32),
        "extent": (
            float(xyz_min[0]),
            float(xyz_max[0]),
            float(xyz_min[1]),
            float(xyz_max[1]),
            float(xyz_min[2]),
            float(xyz_max[2]),
        ),
        "origin": (float(xyz_min[0]), float(xyz_min[1]), float(xyz_min[2])),
        "grid_shape": (nx, ny, nz),
        "spacing": (float(spacing), float(spacing), float(spacing)),
        "eta": float(eta),
    }


def save_embedded_ldos(
    stem: str,
    ldos_result: dict,
    metadata: Optional[dict] = None,
) -> Path:
    """Save an LDOS grid as ``{stem}.eldos.npz``.

    Parameters
    ----------
    stem
        Output file stem (without extension).
    ldos_result
        Dict from :func:`compute_ldos_grid`.
    metadata
        Optional dict of metadata (contour info, k-mesh, version, etc.).

    Returns
    -------
    Path
        The written file path.
    """
    out = Path(stem + ".eldos.npz")
    save_dict = dict(ldos_result)
    if metadata:
        save_dict["metadata"] = json.dumps(metadata, default=str)
    np.savez_compressed(out, **save_dict)
    return out
