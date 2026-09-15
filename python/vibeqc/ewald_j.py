"""Long-range Hartree-matrix builder via FFT Poisson convolution.

This is the ``J_LR`` half of the Ewald-decomposed periodic Hartree
operator. Given a density matrix D and a basis, the builder:

1. Samples the real-space density r(r) = S_{muν} D_{muν} chi_mu(r) chi_ν(r)
   on a uniform grid inside a supplied unit cell.
2. Solves the periodic Poisson equation with the erf-screened Coulomb
   kernel K̃(G) = (4pi / G^2) . exp(-G^2 / (4w^2)) via
   :func:`vibeqc.solve_poisson_erf_screened`.
3. Integrates V_LR(r) against every AO pair:
       J_LR_{muν} = ∫ chi_mu(r) V_LR(r) chi_ν(r) dr
                 ≈ dV . S_g chi_mu(r_g) V_LR(r_g) chi_ν(r_g).

The companion **short-range** J (erfc-screened) is already provided by
:func:`vibeqc.build_jk_gamma_molecular_limit` (Phase 12e-c-2). Summing
J_SR(w) + J_LR(w) recovers the full Hartree matrix for any w.

Current scope (Phase 12e-c-3b)
------------------------------

* All full-rank 3D crystal lattices. Grid points are built in
  fractional coordinates and the C++ FFT Poisson solver applies the
  full reciprocal-lattice metric for ``G.G``.
* The lattice size must be large enough that the periodic images of
  the density don't overlap the original -- in practice a >= 6 bohr of
  vacuum between the molecule and the box boundary on every side. For
  molecular-limit work, wrap the molecule in a cubic box of
  ``a ≈ max_extent + 12 bohr`` and center it.
* Grid resolution: 0.25-0.35 bohr spacing is typically sufficient for
  split-valence bases (6-31G*, def2-SVP). Tight-core bases like
  cc-pVTZ benefit from 0.15-0.20 bohr. The ``auto_grid`` helper picks
  a reasonable default.

Numerics
--------

The builder uses numpy's ``einsum`` for the density contraction and
matrix multiplication for the integration step. For large basis sets
where the grid is ≫ 10⁶ points, this becomes memory-heavy
(``n_points x n_bf`` doubles for chi); the evaluator already chunks the
grid internally so peak memory is ``chunk_size x n_bf``.

G=0 gauge (important)
---------------------

:func:`solve_poisson_erf_screened` pins the reciprocal-space G=0
component of V to zero (otherwise a charged cell gives a divergent
Hartree energy). The consequence is that the returned ``J_LR`` differs
from the full isolated-molecule ``J_full`` (computed via molecular
ERIs) by a scalar-times-overlap correction:

    J_LR(w->inf) = J_full - c(w, Q, V_cell) . S

where S is the AO overlap matrix, Q = tr(D . S) is the electronic
charge, and c is a constant with |c| ∝ 1 / V_cell (Makov-Payne scaling
-- larger boxes shrink the offset). This is a well-known property of
FFT-based Poisson solvers and is the reason periodic DFT codes treat
isolated molecules with care.

**The constant is known in closed form** (2026-08-03,
GRID-BACKEND-CONVERGES-WRONG). The erf kernel's transform is

    FT[erf(w r)/r](G) = 4 pi / G^2  -  pi / w^2  +  O(G^2)

so pinning the WHOLE G=0 component to zero is right for the divergent
4 pi / G^2 (cancelled by the neutralising background) but also discards
the FINITE -pi / w^2. Hence

    c(w, Q, V_cell) = pi Q / (w^2 V_cell)

and the energy a composition misses is  pi Q^2 / (2 w^2 V_cell). That
was NOT a purely academic offset: the composed ``J_SR + J_LR`` grid
backend of :func:`vibeqc.build_j_ewald_3d` never added it back, so it
converged 14.5 mHa away from its own ``analytic_ft`` reference on
H2/STO-3G in a 12-bohr box at w = 0.5 -- flat in the grid spacing, and
scaling as w^-2.

Pass ``restore_g0_finite_part=True`` when composing ``J_SR + J_LR``.
The default keeps the pinned gauge, because ``J_LR`` alone is a
legitimate object in it (and only in it does ``J_LR -> 0`` as
``w -> 0``).

When pairing ``J_LR`` with the matching short-range ``J_SR`` (the
erfc-screened J from :func:`build_jk_gamma_molecular_limit`), the
opposite c.S shifts on the two halves cancel at infinite box size; at
finite box size the residual is the standard Makov-Payne correction.
Phase 12e-c-4 wires up that composition explicitly.

**Consequence for callers**: ``J_LR`` alone is **not** a drop-in
replacement for a molecular J matrix. Use it only in the pair
``J_SR(w) + J_LR(w)`` inside an Ewald-decomposed SCF, or add the
compensating ``c . S`` term by hand.
"""

from __future__ import annotations

import os
import weakref
from collections import OrderedDict
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    Atom,
    BasisSet,
    Molecule,
    PeriodicSystem,
    evaluate_ao,
    solve_poisson_erf_screened,
)


__all__ = [
    "build_j_long_range",
    "auto_grid",
    "evaluate_ao_periodic",
    "get_shifted_basis",
    "clear_shifted_basis_cache",
]


def _shifted_basis_molecule(system: PeriodicSystem, shifted_atoms) -> Molecule:
    """Return a Molecule for shifted AO-basis construction."""
    unit = system.unit_cell_molecule()
    return Molecule(
        shifted_atoms,
        int(getattr(unit, "charge", system.charge)),
        int(getattr(unit, "multiplicity", system.multiplicity)),
    )


# Module-level LRU cache of shifted BasisSet objects keyed by
# (id(basis), id(system), integer lattice index). Reused across
# evaluate_ao_periodic and periodic_density's per-cell shift loop.
#
# This cache used to be an unbounded dict. Long-lived Python processes
# that evaluate many distinct periodic systems could retain every shifted
# image BasisSet until process exit. Keep the hot same-system reuse, but
# bound total retained entries and, when the pybind11 wrappers support weak
# references, drop all entries for an owner pair as soon as that pair dies.
def _shifted_basis_cache_max_entries() -> int:
    raw = os.environ.get("VIBEQC_SHIFTED_BASIS_CACHE_MAX_ENTRIES", "2048")
    try:
        value = int(raw)
    except ValueError:
        return 2048
    return max(0, value)


_SHIFTED_BASIS_CACHE_MAX_ENTRIES = _shifted_basis_cache_max_entries()
_SHIFTED_BASIS_CACHE: OrderedDict[
    tuple[int, int, tuple[int, int, int]], BasisSet
] = OrderedDict()
_SHIFTED_BASIS_OWNER_FINALIZERS: dict[
    tuple[int, int], list[weakref.finalize]
] = {}


def clear_shifted_basis_cache() -> None:
    """Drop all cached shifted BasisSet objects.

    Useful in long-running processes that build many distinct
    ``(basis, system)`` pairs and would otherwise grow the cache
    without bound. Tests that intentionally measure cold-cache cost
    should call this first.
    """
    _SHIFTED_BASIS_CACHE.clear()
    for finalizers in _SHIFTED_BASIS_OWNER_FINALIZERS.values():
        for finalizer in finalizers:
            finalizer.detach()
    _SHIFTED_BASIS_OWNER_FINALIZERS.clear()


def _drop_shifted_basis_owner(owner_key: tuple[int, int]) -> None:
    """Drop all shifted bases derived from one source ``(basis, system)``."""
    for key in list(_SHIFTED_BASIS_CACHE):
        if key[:2] == owner_key:
            _SHIFTED_BASIS_CACHE.pop(key, None)
    for finalizer in _SHIFTED_BASIS_OWNER_FINALIZERS.pop(owner_key, []):
        finalizer.detach()


def _register_shifted_basis_owner(
    basis: BasisSet,
    system: PeriodicSystem,
    owner_key: tuple[int, int],
) -> None:
    """Attach best-effort finalizers for a shifted-basis cache owner pair."""
    if owner_key in _SHIFTED_BASIS_OWNER_FINALIZERS:
        return

    finalizers: list[weakref.finalize] = []
    for obj in (basis, system):
        try:
            finalizers.append(weakref.finalize(obj, _drop_shifted_basis_owner, owner_key))
        except TypeError:
            # Some pybind11 classes are not weak-referenceable unless the
            # binding enables it. The LRU cap still bounds process lifetime
            # growth on those builds.
            pass
    if finalizers:
        _SHIFTED_BASIS_OWNER_FINALIZERS[owner_key] = finalizers


def _cache_shifted_basis(
    key: tuple[int, int, tuple[int, int, int]],
    basis: BasisSet,
) -> BasisSet:
    if _SHIFTED_BASIS_CACHE_MAX_ENTRIES <= 0:
        return basis
    _SHIFTED_BASIS_CACHE[key] = basis
    _SHIFTED_BASIS_CACHE.move_to_end(key)
    while len(_SHIFTED_BASIS_CACHE) > _SHIFTED_BASIS_CACHE_MAX_ENTRIES:
        _SHIFTED_BASIS_CACHE.popitem(last=False)
    return basis


def get_shifted_basis(
    basis: BasisSet,
    system: PeriodicSystem,
    index: Sequence[int],
) -> BasisSet:
    """Return a BasisSet whose atoms are shifted by integer lattice index.

    For ``index = (ix, iy, iz)``, the returned basis sits at
    ``r_a + ix . a1 + iy . a2 + iz . a3`` for each home-cell atom.
    Results are cached per ``(id(basis), id(system), index)`` so the
    same shift is built at most once for the lifetime of the source
    objects.
    """
    key = (id(basis), id(system), tuple(int(i) for i in index))
    cached = _SHIFTED_BASIS_CACHE.pop(key, None)
    if cached is not None:
        _SHIFTED_BASIS_CACHE[key] = cached
        return cached
    if key[2] == (0, 0, 0):
        return basis
    _register_shifted_basis_owner(basis, system, key[:2])
    lat = np.asarray(system.lattice, dtype=float)
    dr = lat @ np.asarray(key[2], dtype=float)
    shifted_atoms = [
        Atom(int(a.Z),
             [float(a.xyz[0] + dr[0]),
              float(a.xyz[1] + dr[1]),
              float(a.xyz[2] + dr[2])])
        for a in system.unit_cell
    ]
    mol_g = _shifted_basis_molecule(system, shifted_atoms)
    shifted = BasisSet(mol_g, basis.name)
    return _cache_shifted_basis(key, shifted)


def evaluate_ao_periodic(
    basis: BasisSet,
    system: PeriodicSystem,
    points: np.ndarray,
    *,
    image_radius: int = 1,
) -> np.ndarray:
    """Evaluate the AO basis on ``points`` summed over periodic images.

    Returns ``chi_periodic[g, mu] = S_R chi_mu(r_g - R)`` with ``R``
    ranging over the lattice vectors with each fractional component
    in ``[-image_radius, +image_radius]``. For ``image_radius = 0``
    this is identical to :func:`evaluate_ao(basis, points)`; for
    ``image_radius >= 1`` the AO Gaussian's tail across the box
    boundary is captured, so the sampled density integrates to the
    correct electron count regardless of where the molecule sits in
    the cell (fixes the v0.6.x translation-invariance break).

    For STO-3G in an L = 30 bohr box, ``image_radius = 1`` (3^3 = 27
    cells) is sufficient. Diffuse basis sets (cc-pVDZ etc.) in tighter
    boxes may benefit from ``image_radius = 2``; verify by running
    :file:`examples/debug/prototype_periodic_ao.py` on the system.

    Parameters
    ----------
    basis
        AO basis for the home cell.
    system
        Periodic system; provides the lattice vectors and the home-cell
        atom positions used to construct shifted bases.
    points
        ``(n_points, 3)`` cartesian grid points in bohr.
    image_radius
        Half-width of the image cube along each axis. ``1`` -> 27
        cells; ``2`` -> 125 cells.

    Returns
    -------
    np.ndarray of shape ``(n_points, nbasis)``.
    """
    import itertools

    if image_radius == 0:
        return evaluate_ao(basis, points)

    chi_p = evaluate_ao(basis, points)  # home cell, image (0,0,0)

    for ix, iy, iz in itertools.product(
        range(-image_radius, image_radius + 1), repeat=3
    ):
        if ix == 0 and iy == 0 and iz == 0:
            continue
        basis_g = get_shifted_basis(basis, system, (ix, iy, iz))
        chi_p += evaluate_ao(basis_g, points)
    return chi_p


GridShape = Tuple[int, int, int]


def auto_grid(
    lattice: np.ndarray,
    spacing_bohr: float = 0.3,
) -> GridShape:
    """Pick a uniform grid shape for a given cell with approximately
    ``spacing_bohr`` per voxel along each lattice vector.

    Rounds up to the nearest even integer on each axis -- FFTW prefers
    even lengths (in particular, powers of 2 and small-factor
    composites). Use ``auto_grid(lat, 0.3)`` for general DFT-style
    work, 0.15-0.20 for tight valence bases.
    """
    lat = np.asarray(lattice, dtype=float)
    lengths = np.linalg.norm(lat, axis=0)
    def _round_even(x: float) -> int:
        n = int(np.ceil(x))
        return n if n % 2 == 0 else n + 1
    return (
        _round_even(lengths[0] / spacing_bohr),
        _round_even(lengths[1] / spacing_bohr),
        _round_even(lengths[2] / spacing_bohr),
    )


def _uniform_grid_points(
    lattice: np.ndarray,
    grid_shape: GridShape,
    origin: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Return ``(points, dV)`` for a uniform grid on a periodic cell.

    Voxels are at fractional coordinates ``i / n`` for i in ``0 ... n-1``
    (no end-point duplication), then mapped to Cartesian coordinates
    by ``r = origin + A f`` where the lattice columns are ``a_i``.
    """
    nx, ny, nz = grid_shape
    fx = np.arange(nx, dtype=float) / float(nx)
    fy = np.arange(ny, dtype=float) / float(ny)
    fz = np.arange(nz, dtype=float) / float(nz)
    FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing="ij")
    frac = np.column_stack([FX.ravel(), FY.ravel(), FZ.ravel()])
    points = np.asarray(origin, dtype=float).reshape(1, 3) + frac @ lattice.T
    cell_volume = abs(np.linalg.det(lattice))
    dV = cell_volume / (nx * ny * nz)
    return points, dV


def build_j_long_range(
    basis: BasisSet,
    D: np.ndarray,
    lattice: np.ndarray,
    omega: float,
    grid_shape: Optional[Union[GridShape, int]] = None,
    *,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    chunk_size: int = 200_000,
    system: Optional[PeriodicSystem] = None,
    ao_image_radius: int = 1,
    restore_g0_finite_part: bool = False,
) -> np.ndarray:
    """Long-range Hartree matrix ``J_LR`` via FFT-Poisson convolution.

    Parameters
    ----------
    basis
        :class:`vibeqc.BasisSet` defining the AOs.
    D
        Density matrix of shape ``(n_bf, n_bf)``. For UHF / UKS pass
        ``D_a + D_b`` (the total density); the Coulomb operator doesn't
        see spin. The routine does NOT symmetrize D -- the caller is
        responsible for providing a proper density.
    lattice
        ``(3, 3)`` cell matrix (columns are a₁, a₂, a₃) in bohr.
    omega
        Ewald splitting parameter (1/bohr). w -> 0 gives J_LR -> 0
        (no long-range in the split); w -> inf gives J_LR -> J_full.
    grid_shape
        Explicit ``(nx, ny, nz)`` grid shape. If omitted (or given as
        an int), :func:`auto_grid` picks one targeting
        ``spacing_bohr`` per voxel.
    origin
        Lower corner of the grid in bohr (default origin: 0,0,0).
        Use this to center the molecule inside the box -- typically
        ``origin = -box_size / 2`` if the molecule sits at the origin.
    spacing_bohr
        Target grid spacing if ``grid_shape`` is auto-picked.
    chunk_size
        Number of grid points per AO-evaluation chunk. Controls peak
        memory of the ``(chunk, n_bf)`` AO matrix.

    Returns
    -------
    J_LR : np.ndarray
        ``(n_bf, n_bf)`` symmetric matrix, Hartree. Zero-mean gauge
        (the G=0 component of V_LR is dropped -- matches the
        periodic-cell convention the short-range builder uses).

    Notes
    -----
    For Γ-only molecular-limit work, wrap the molecule in a cubic box
    large enough that images don't overlap the density tail, call
    :func:`auto_grid` for a reasonable default resolution, and verify
    grid convergence by halving the spacing and re-checking the result.
    """
    lat = np.asarray(lattice, dtype=float)
    D = np.asarray(D, dtype=float)
    if lat.shape != (3, 3):
        raise ValueError(
            f"build_j_long_range: lattice must be 3x3, got {lat.shape}"
        )
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(
            f"build_j_long_range: D must be a square matrix, got {D.shape}"
        )

    if grid_shape is None:
        grid_shape = auto_grid(lat, spacing_bohr)
    elif isinstance(grid_shape, int):
        grid_shape = (grid_shape, grid_shape, grid_shape)
    grid_shape = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))

    if origin is None:
        origin_arr = np.zeros(3)
    else:
        origin_arr = np.asarray(origin, dtype=float).reshape(3)

    points, dV = _uniform_grid_points(lat, grid_shape, origin_arr)
    n_points = points.shape[0]
    nbf = D.shape[0]

    # AO evaluator: when ``system`` is supplied (the v0.7-correct path),
    # AOs are summed over periodic images so chi(r) is properly periodic
    # and the FFT-Poisson density integrates to the right electron count
    # regardless of where the molecule sits in the cell. When ``system``
    # is omitted we fall back to bare ``evaluate_ao`` -- the legacy
    # (translation-broken) v0.6.x behavior, kept for backward
    # compatibility with callers that haven't been updated yet.
    if system is not None and ao_image_radius > 0:
        def _eval_chi(block: np.ndarray) -> np.ndarray:
            return evaluate_ao_periodic(
                basis, system, block, image_radius=ao_image_radius,
            )
    else:
        def _eval_chi(block: np.ndarray) -> np.ndarray:
            return evaluate_ao(basis, block)

    # Build rho(r) and integrate V_LR against AOs in one pass per
    # chunk. We never materialise a full (n_points, n_bf) AO matrix
    # for large grids -- only one chunk at a time. To keep the density
    # accumulation in sync with the integration, we do the density in
    # a first chunked pass, Poisson-solve, then do the integration in
    # a second chunked pass.
    rho_flat = np.empty(n_points, dtype=float)
    for i in range(0, n_points, chunk_size):
        block = points[i:i + chunk_size]
        chi = _eval_chi(block)                   # (m, nbf)
        # rho(r_g) = chi(r_g) D chi(r_g)ᵀ  (diagonal of chiDchiᵀ)
        rho_flat[i:i + chunk_size] = np.einsum(
            "gi,ij,gj->g", chi, D, chi, optimize=True
        )
    rho_3d = rho_flat.reshape(grid_shape)

    V_lr_3d = solve_poisson_erf_screened(rho_3d, lat, float(omega))
    V_lr_flat = V_lr_3d.ravel()

    # ``restore_g0_finite_part`` -- see the "G=0 gauge" section of the
    # module docstring. The erf kernel's transform is
    #
    #   FT[erf(w r)/r](G) = 4 pi / G^2  -  pi / w^2  +  O(G^2)
    #
    # and ``solve_poisson_erf_screened`` pins the WHOLE G=0 component to
    # zero. That is right for the 4 pi / G^2 divergence (cancelled by the
    # neutralising background) but it also discards the FINITE -pi / w^2.
    # The dropped piece is a constant potential, hence exactly the
    # ``c(w, Q, V_cell) . S`` offset the docstring documents:
    #
    #   v0 = -(pi / w^2) * Q / V_cell,   dJ = v0 * S,   dE = v0 * Q / 2
    #
    # Callers that want J_LR ALONE keep the pinned gauge (the default):
    # in it J_LR -> 0 as w -> 0, which several consumers rely on. Callers
    # composing J_SR + J_LR = J_full need the term restored, because the
    # identity fails by +pi Q^2 / (2 w^2 V_cell) without it -- measured as
    # a 14.5 mHa disagreement between the ``grid`` and ``analytic_ft``
    # EWALD_3D backends on H2/STO-3G/12-bohr at w = 0.5, flat in the grid
    # spacing and scaling as w^-2 (GRID-BACKEND-CONVERGES-WRONG).
    if restore_g0_finite_part and float(omega) > 0.0:
        cell_volume_lr = abs(np.linalg.det(lat))
        q_cell = float(np.sum(rho_3d)) * dV
        V_lr_flat = V_lr_flat - (
            (np.pi / float(omega) ** 2) * (q_cell / cell_volume_lr)
        )

    # J_LR_muν = dV . S_g chi_mu(r_g) V_LR(r_g) chi_ν(r_g)
    #        = dV . chiᵀ . diag(V_LR) . chi
    # Chunked accumulation so we never allocate (N, nbf) at once.
    J = np.zeros((nbf, nbf), dtype=float)
    for i in range(0, n_points, chunk_size):
        block = points[i:i + chunk_size]
        chi = _eval_chi(block)                   # (m, nbf)
        weighted = chi * V_lr_flat[i:i + chunk_size, None]   # (m, nbf)
        J += chi.T @ weighted
    J *= dV

    # Numerical symmetrisation -- the underlying grid + AO evaluation
    # preserves symmetry to machine precision, but floating-point
    # accumulation order can leave ~1e-15 asymmetry on the off-diagonal.
    # A single (J + J.T) / 2 cleans it up.
    return 0.5 * (J + J.T)
