"""Scatter/gather between a per-k eigenvalue list and the full regular BZ grid.

The Gilat-Raubenheimer kernel (:mod:`vibeqc.bz_integration.gilat`) operates on
the full regular mesh as a dense `(N1, N2, N3, nband)` array. These helpers map
a driver's per-k eigenvalue list onto that grid (and gather per-k occupations
back off it) using each k-point's **fractional coordinate** -- grid index
``(round(frac*mesh)) mod mesh`` -- which is independent of however the k-points
happen to be ordered or stored.

A **full (symmetry-unreduced) regular mesh** is required: exactly one k-point
per grid cell. For an IBZ-reduced mesh, expand to the full BZ first (e.g. via
``vibeqc.pbc_bipole._expand_ibz_kmesh_for_ewald_j``); band energies are
symmetry-invariant, ``eps_n(R k) = eps_n(k)``, so the expansion just replicates
the IBZ eigenvalues over each star. (That expansion is the driver-wiring step;
these helpers deliberately stay pure-array so they are engine-agnostic and need
no k-mesh object.)
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .gilat import gilat_raubenheimer_occupations

# A k-point is "on the mesh" if frac*mesh is integer to this tolerance.
_GRID_TOL = 1e-6


def kpoint_grid_indices(
    kpoints_frac: np.ndarray,
    mesh: Sequence[int],
) -> np.ndarray:
    """Integer grid indices ``(N, 3)`` for fractional k-points on a regular mesh.

    ``index = round(frac * mesh) mod mesh`` -- robust to the Γ-centred ``[0,1)``
    and symmetric ``[-1/2, 1/2)`` conventions alike. Raises if any k-point does
    not sit on the regular mesh of the given shape.
    """
    kf = np.asarray(kpoints_frac, dtype=float)
    m = np.asarray(mesh, dtype=int).reshape(3)
    if kf.ndim != 2 or kf.shape[1] != 3:
        raise ValueError(
            f"kpoints_frac must have shape (N, 3); got {kf.shape}"
        )
    if np.any(m < 1):
        raise ValueError(f"mesh sizes must be >= 1; got {m.tolist()}")
    scaled = kf * m
    idx = np.rint(scaled).astype(np.int64)
    if np.any(np.abs(scaled - idx) > _GRID_TOL):
        raise ValueError(
            "k-points are not on a regular mesh of shape "
            f"{m.tolist()} (a fractional coordinate is not an integer "
            "multiple of 1/N). Band-path or off-grid meshes are not "
            "supported by the Gilat-Raubenheimer grid."
        )
    return idx % m


def fractional_kpoints(
    reciprocal_lattice: np.ndarray,
    kpoints_cart: np.ndarray,
) -> np.ndarray:
    """Fractional k-coordinates ``(N, 3)`` from Cartesian ones.

    vibe-qc convention: the reciprocal-lattice matrix ``B`` carries the b_i
    vectors as *columns*, so ``k_cart = B @ k_frac`` and
    ``k_frac = pinv(B) @ k_cart``. Mirrors ``KPoints._from_bloch_kmesh`` (the
    pseudo-inverse tolerates low-dimensional cells whose B has zero columns
    along vacuum axes). Use this to recover fractional coordinates from a native
    ``BlochKMesh`` (which exposes only Cartesian ``kpoints``).
    """
    B = np.asarray(reciprocal_lattice, dtype=float).reshape(3, 3)
    cart = np.asarray(kpoints_cart, dtype=float).reshape(-1, 3)
    return (np.linalg.pinv(B) @ cart.T).T


def eigenvalues_to_full_grid(
    kpoints_frac: np.ndarray,
    mesh: Sequence[int],
    eps_per_k: Sequence[np.ndarray],
) -> np.ndarray:
    """Scatter a per-k eigenvalue list onto the full regular grid.

    Returns an array of shape ``(N1, N2, N3, nband)``. Requires a full mesh
    (``len(eps_per_k) == prod(mesh)``) covering every cell exactly once; raises
    otherwise (the signal to expand an IBZ-reduced mesh first).
    """
    m = np.asarray(mesh, dtype=int).reshape(3)
    n_cells = int(np.prod(m))
    eps_list = [np.asarray(np.real(e), dtype=float).reshape(-1) for e in eps_per_k]
    n_k = len(eps_list)
    if n_k == 0:
        raise ValueError("eigenvalues_to_full_grid: no k-points")
    if n_k != n_cells:
        raise ValueError(
            f"eigenvalues_to_full_grid: expected a full mesh of {n_cells} "
            f"k-points (prod{tuple(m.tolist())}); got {n_k}. Expand an "
            "IBZ-reduced mesh to the full BZ before calling."
        )
    nband = eps_list[0].shape[0]
    if any(e.shape[0] != nband for e in eps_list):
        raise ValueError(
            "eigenvalues_to_full_grid: all k-points must carry the same "
            "number of bands"
        )
    idx = kpoint_grid_indices(kpoints_frac, mesh)
    if idx.shape[0] != n_k:
        raise ValueError(
            "eigenvalues_to_full_grid: kpoints_frac and eps_per_k lengths "
            f"differ ({idx.shape[0]} vs {n_k})"
        )

    grid = np.full((int(m[0]), int(m[1]), int(m[2]), nband), np.nan, dtype=float)
    grid[idx[:, 0], idx[:, 1], idx[:, 2], :] = np.stack(eps_list, axis=0)
    # n_k == n_cells, so any leftover NaN means two k-points hit one cell and
    # another cell was missed -- i.e. not a clean bijective full mesh.
    if np.any(np.isnan(grid)):
        raise ValueError(
            "eigenvalues_to_full_grid: k-points do not cover every grid cell "
            "exactly once (duplicate or missing cell)."
        )
    return grid


def grid_to_kpoints(
    grid: np.ndarray,
    kpoints_frac: np.ndarray,
    mesh: Sequence[int],
) -> np.ndarray:
    """Gather a grid quantity back to per-k order (inverse of the scatter).

    ``grid`` has shape ``(N1, N2, N3, ...)``; returns ``(N, ...)`` in the order
    of ``kpoints_frac``.
    """
    g = np.asarray(grid)
    idx = kpoint_grid_indices(kpoints_frac, mesh)
    return g[idx[:, 0], idx[:, 1], idx[:, 2], ...]


def expand_ibz_eigenvalues(
    system,
    ibz_kmesh,
    eps_ibz_per_k: Sequence[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Expand IBZ-reduced per-k eigenvalues to the full regular BZ mesh.

    Band energies are symmetry-invariant (``eps_n(R k) = eps_n(k)``), so each
    full-BZ point takes the eigenvalues of its irreducible representative. The
    k-mesh's ``ir_mapping`` gives, for each point of the *unreduced* Monkhorst-
    Pack mesh, the index of its IBZ representative -- indexed in the same order
    as ``monkhorst_pack(..., use_symmetry=False)``, which is regenerated here to
    recover the full-BZ Cartesian k-points (convention-free: we never assume a
    raster order, we read the C++ mesh back).

    Returns ``(full_kpoints_cart, eps_full_per_k)`` ready for
    :func:`fractional_kpoints` + the GR kernel. If the mesh is already full
    (empty ``ir_mapping``) the inputs are returned unchanged.
    """
    ir = np.asarray(getattr(ibz_kmesh, "ir_mapping", []), dtype=np.int64).reshape(-1)
    eps_ibz = [np.asarray(np.real(e), dtype=float).reshape(-1) for e in eps_ibz_per_k]
    if ir.size == 0:
        return np.asarray(ibz_kmesh.kpoints, dtype=float).reshape(-1, 3), eps_ibz

    from vibeqc import monkhorst_pack  # lazy: vibeqc is loaded by call time

    mesh = [int(x) for x in ibz_kmesh.mesh]
    full = monkhorst_pack(system, mesh, use_symmetry=False)
    full_cart = np.asarray(full.kpoints, dtype=float).reshape(-1, 3)
    if full_cart.shape[0] != ir.size:
        raise ValueError(
            "expand_ibz_eigenvalues: ir_mapping length "
            f"({ir.size}) does not match the full mesh size "
            f"({full_cart.shape[0]})"
        )
    if int(ir.max(initial=-1)) >= len(eps_ibz):
        raise ValueError(
            "expand_ibz_eigenvalues: ir_mapping indexes beyond the supplied "
            f"IBZ eigenvalue list ({len(eps_ibz)} entries)"
        )
    eps_full = [eps_ibz[int(ir[r])] for r in range(ir.size)]
    return full_cart, eps_full


def gilat_occupations_on_kmesh(
    kpoints_frac: np.ndarray,
    mesh: Sequence[int],
    eps_per_k: Sequence[np.ndarray],
    n_electrons_per_cell: float,
    spin_degeneracy: float = 2.0,
) -> Tuple[List[np.ndarray], float]:
    """Gilat-Raubenheimer occupations + Fermi level for a full-mesh k-point list.

    Convenience wrapper for the driver path: scatter ``eps_per_k`` onto the full
    grid, run the GR kernel, and gather the occupations back into per-k order.

    Returns ``(occ_per_k, e_fermi)`` where ``occ_per_k`` is a list of per-band
    occupation arrays (in ``[0, spin_degeneracy]``) aligned with
    ``kpoints_frac``. The mesh weights are uniform ``1/prod(mesh)``.
    """
    eps_grid = eigenvalues_to_full_grid(kpoints_frac, mesh, eps_per_k)
    occ_grid, e_fermi = gilat_raubenheimer_occupations(
        eps_grid, n_electrons_per_cell, spin_degeneracy
    )
    occ_arr = grid_to_kpoints(occ_grid, kpoints_frac, mesh)  # (N, nband)
    occ_per_k = [np.asarray(occ_arr[i], dtype=float) for i in range(occ_arr.shape[0])]
    return occ_per_k, float(e_fermi)


def gilat_occupations_for_kmesh(
    system,
    kmesh,
    eps_per_k: Sequence[np.ndarray],
    n_electrons_per_cell: float,
    spin_degeneracy: float = 2.0,
) -> Tuple[List[np.ndarray], float]:
    """Gilat-Raubenheimer occupations for a driver's k-mesh (full *or* IBZ).

    The driver-facing entry point: handles a full Monkhorst-Pack mesh and a
    symmetry-reduced (IBZ) mesh alike, returning occupations aligned with
    ``kmesh``'s own k-point order (so the SCF density build, which uses the
    k-mesh weights, consumes them directly).

    * Full mesh -> GR straight on those k-points.
    * IBZ mesh  -> expand eigenvalues to the full BZ (``eps_n(R k)=eps_n(k)``),
      run GR on the full grid, then gather the occupations back onto the IBZ
      k-points. Because occupations are constant over a symmetry star and the
      IBZ weights carry the star multiplicity, the IBZ-weighted density equals
      the full-mesh integral.

    Returns ``(occ_per_k, e_fermi)`` with ``occ_per_k`` in ``kmesh`` order.
    """
    recip = np.asarray(system.reciprocal_lattice(), dtype=float)
    mesh = tuple(int(x) for x in kmesh.mesh)
    ir = np.asarray(getattr(kmesh, "ir_mapping", []), dtype=np.int64).reshape(-1)

    if ir.size == 0:
        # Full (unreduced) mesh.
        frac = fractional_kpoints(recip, np.asarray(kmesh.kpoints, dtype=float))
        return gilat_occupations_on_kmesh(
            frac, mesh, eps_per_k, n_electrons_per_cell, spin_degeneracy
        )

    # IBZ-reduced: expand -> GR on the full grid -> gather back to IBZ points.
    full_cart, eps_full = expand_ibz_eigenvalues(system, kmesh, eps_per_k)
    frac_full = fractional_kpoints(recip, full_cart)
    occ_full, e_fermi = gilat_occupations_on_kmesh(
        frac_full, mesh, eps_full, n_electrons_per_cell, spin_degeneracy
    )
    occ_grid = eigenvalues_to_full_grid(frac_full, mesh, occ_full)
    frac_ibz = fractional_kpoints(recip, np.asarray(kmesh.kpoints, dtype=float))
    occ_ibz = grid_to_kpoints(occ_grid, frac_ibz, mesh)  # (n_ibz, nband)
    occ_per_k = [np.asarray(occ_ibz[i], dtype=float) for i in range(occ_ibz.shape[0])]
    return occ_per_k, float(e_fermi)
