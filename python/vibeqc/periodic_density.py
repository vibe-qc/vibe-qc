"""Phase 12e-c-4c-i: periodic density on a uniform real-space grid.

For a Γ-point periodic system, the real-space electron density is a
lattice sum over AO pairs:

    r(r)  =  S_g  S_{muν}  D(g)_{muν} . chi_mu(r) . chi_ν(r - R_g)

where ``D(g)`` is the real-space density matrix at cell shift ``g``
(indexed by lattice cells, same convention as
``LatticeMatrixSet``). For a converged Γ-only SCF this integrates
to ``N_electrons`` per unit cell.

The existing :func:`vibeqc.build_j_long_range` evaluates the density
assuming only the ``g = 0`` term is non-zero -- correct in the
molecular-limit (vacuum-padded molecule in a big box) but wrong for
bulk crystals with cell-to-cell density overlap. This module adds
the proper lattice-summed density grid + a periodic-aware
long-range J builder that composes with the short-range erfc J the
same way the molecular-limit pipeline does.

The periodic J_LR builder returns either:

1. A single ``(n_bf, n_bf)`` matrix (Γ-only J) -- for single-k SCF.
2. A full ``LatticeMatrixSet`` of J(g) blocks -- for multi-k SCF via
   Bloch-summed J(k) = S_g e^{i k.R_g} J(g).

The Γ-only form is what the Phase 12e-c-4b driver consumes;
multi-k form feeds into the 12e-c-4c-iii driver.

Scope (this commit)
-------------------

- Periodic density grid construction: multi-cell sum with caching
  of shifted-basis AO evaluations per cell.
- Γ-only J builder (returns one matrix).
- Multi-g J builder (returns a LatticeMatrixSet).
- Tests: molecular-limit equivalence with build_j_long_range, density
  normalisation to N_electrons, round-trip on a bulk H2 chain.
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    Atom,
    BasisSet,
    LatticeCell,
    LatticeMatrixSet,
    PeriodicSystem,
    evaluate_ao,
    solve_poisson_erf_screened,
)
from .ewald_j import (
    _shifted_basis_molecule,
    auto_grid,
    evaluate_ao_periodic,
    get_shifted_basis,
)

__all__ = [
    "build_j_long_range_periodic",
    "evaluate_periodic_density_on_grid",
    "evaluate_weighted_k_density_on_grid",
]


# ---------------------------------------------------------------------------
# Shifted-basis helpers
# ---------------------------------------------------------------------------

def _shifted_basis(
    basis: BasisSet,
    system: PeriodicSystem,
    dr_cart: np.ndarray,
) -> BasisSet:
    """Build a BasisSet whose atoms are shifted by ``dr_cart`` (bohr).

    Mirrors the C++ ``shifted_basis`` helper in periodic_xc.cpp. When
    ``dr_cart`` is integer-aligned with the lattice vectors the result
    is looked up from the module-level cache in
    :mod:`vibeqc.ewald_j` (same cache shared with
    :func:`evaluate_ao_periodic`'s image sum, so a given lattice shift
    is built at most once for the lifetime of the source basis +
    system); off-lattice shifts fall back to a fresh build.
    """
    lat = np.asarray(system.lattice, dtype=float)
    dr = np.asarray(dr_cart, dtype=float).reshape(3)
    try:
        frac = np.linalg.solve(lat, dr)
    except np.linalg.LinAlgError:
        frac = None
    if frac is not None:
        frac_int = np.rint(frac)
        if np.allclose(frac, frac_int, atol=1e-9):
            return get_shifted_basis(
                basis, system, (int(frac_int[0]), int(frac_int[1]), int(frac_int[2]))
            )
    shifted_atoms: List[Atom] = []
    for a in system.unit_cell:
        x, y, z = a.xyz
        shifted_atoms.append(Atom(int(a.Z),
                                   [float(x + dr[0]),
                                    float(y + dr[1]),
                                    float(z + dr[2])]))
    mol = _shifted_basis_molecule(system, shifted_atoms)
    return BasisSet(mol, basis.name)


def _uniform_grid_points(
    lattice: np.ndarray,
    grid_shape: Tuple[int, int, int],
    origin: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Return ``(points, dV)`` for a uniform fractional grid.

    Same convention as :mod:`vibeqc.ewald_j`: grid indices are
    fractional coordinates and Cartesian points are ``origin + A f``
    with lattice vectors stored in the columns of ``A``.
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


# ---------------------------------------------------------------------------
# Periodic density on a uniform grid
# ---------------------------------------------------------------------------

def evaluate_periodic_density_on_grid(
    basis: BasisSet,
    system: PeriodicSystem,
    D_real: LatticeMatrixSet,
    *,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    chunk_size: int = 200_000,
    ao_image_radius: int = 1,
    _return_workspace: bool = False,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Evaluate ``r(r) = S_g S_{muν} D(g)_{muν} chi_mu(r) chi_ν(r - R_g)`` on a
    uniform real-space grid spanning the periodic unit cell.

    Parameters
    ----------
    basis
        AO basis for the unit cell.
    system
        :class:`PeriodicSystem`.
    D_real
        :class:`LatticeMatrixSet` holding D(g) blocks indexed by
        lattice cells, as returned by
        :func:`vibeqc.real_space_density_from_kpoints` or built from a
        converged SCF.
    grid_shape, origin, spacing_bohr, chunk_size
        Grid controls, same convention as
        :func:`vibeqc.build_j_long_range`.

    Returns
    -------
    rho
        3D numpy array of shape ``grid_shape`` holding r(r) on the
        uniform grid (electrons per bohr^3).
    grid_shape
        The resolved grid shape (useful when ``grid_shape`` was
        ``None`` and auto-picked).
    """
    lat = np.asarray(system.lattice, dtype=float)

    if grid_shape is None:
        grid_shape_t = auto_grid(lat, spacing_bohr)
    elif isinstance(grid_shape, int):
        grid_shape_t = (grid_shape, grid_shape, grid_shape)
    else:
        grid_shape_t = tuple(int(x) for x in grid_shape)

    origin_arr = (np.asarray(origin, dtype=float).reshape(3)
                   if origin is not None else np.zeros(3))

    points, _ = _uniform_grid_points(lat, grid_shape_t, origin_arr)
    n_points = points.shape[0]

    # The periodic density carries exactly ONE lattice sum, and both AO
    # factors share it:
    #
    #   rho(r) = S_R S_g S_muν D(g)_{muν} chi_mu(r - R) chi_ν(r - R - R_g)
    #
    # equivalently rho(r) = S_R rho_0(r - R) with rho_0 built from plain,
    # NON-image-summed AOs. That single sum is what makes rho periodic;
    # the g sum is the separate density-matrix sum.
    #
    # Until 2026-08-02 this routine instead image-summed the two factors
    # INDEPENDENTLY (chi_ref and chi_g each via evaluate_ao_periodic),
    # i.e. S_R chi_mu(r-R_g-R) times S_R' chi_ν(r-R'), on the stated
    # rationale that "the chi_g side ALSO needs periodic image-summing so
    # the density is symmetric in the (mu, ν) indices". Symmetry does not
    # require two independent sums, and letting R and R' run freely
    # double-counts: the grid integral then GREW with ao_image_radius
    # instead of converging. Measured on LiH rocksalt (primitive fcc,
    # STO-3G, Gamma GDF density, 40^3) against the exact
    # S_g tr(D(g) S(g)) = 2.7407 -- shipped 0.2134 / 3.6161 / 3.9915 at
    # radius 0 / 1 / 2 (already 46 % over and still climbing) versus
    # 0.2134 / 2.7320 / 2.7407 for the tied sum below. User-facing .xsf /
    # .cube artifacts inherited the error on every periodic route.
    # Filed as PERIODIC-DENSITY-GRID-DOUBLE-COUNTS-IMAGES.
    #
    # Cost is unchanged: the old form evaluated two image-summed AO
    # tables per g block (each internally (2r+1)^3 cells), the new one
    # evaluates (2r+1)^3 shared-R translations of two plain tables.
    lattice_vectors = np.asarray(lat, dtype=float)
    radius = int(ao_image_radius)
    if radius < 0 or radius != ao_image_radius:
        raise ValueError(
            "periodic density AO image radius must be a nonnegative integer"
        )
    dim = max(0, min(3, int(getattr(system, "dim", 3))))
    image_ranges = [
        range(-radius, radius + 1) if axis < dim else (0,)
        for axis in range(3)
    ]
    translations = [
        (i * lattice_vectors[:, 0]
         + j * lattice_vectors[:, 1]
         + k * lattice_vectors[:, 2])
        for i in image_ranges[0]
        for j in image_ranges[1]
        for k in image_ranges[2]
    ]

    def _plain_ao(basis_obj) -> np.ndarray:
        """AO table with NO image summing (the image sum is the R loop)."""
        out = np.empty((n_points, basis.nbasis), dtype=float)
        for i in range(0, n_points, chunk_size):
            block = points[i:i + chunk_size]
            out[i:i + chunk_size, :] = evaluate_ao_periodic(
                basis_obj, system, block, image_radius=0,
            )
        return out

    density_cells = list(D_real.cells)
    density_blocks = list(D_real.blocks)
    if len(density_cells) != len(density_blocks):
        raise ValueError(
            "periodic density has misaligned lattice cells and matrix blocks"
        )

    live_cells: list[
        tuple[LatticeCell, np.ndarray, tuple[int, int, int]]
    ] = []
    seen_cells: set[tuple[int, int, int]] = set()
    expected_shape = (int(basis.nbasis), int(basis.nbasis))
    for cell, block in zip(density_cells, density_blocks):
        try:
            index_values = np.asarray(cell.index, dtype=float)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "periodic density contains a cell with an invalid index"
            ) from exc
        if index_values.shape != (3,) or not np.isfinite(index_values).all():
            raise ValueError(
                "periodic density lattice-cell indices must be finite "
                "three-vectors"
            )
        key = tuple(int(value) for value in index_values)
        if not np.array_equal(index_values, np.asarray(key, dtype=float)):
            raise ValueError(
                f"periodic density contains a non-integral lattice cell {key}"
            )
        if key in seen_cells:
            raise ValueError(f"periodic density contains duplicate lattice cell {key}")
        seen_cells.add(key)

        try:
            r_cart = np.asarray(cell.r_cart, dtype=float)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"periodic density cell {key} has an invalid Cartesian translation"
            ) from exc
        expected_r_cart = lattice_vectors @ np.asarray(key, dtype=float)
        if (
            r_cart.shape != (3,)
            or not np.isfinite(r_cart).all()
            or not np.allclose(
                r_cart,
                expected_r_cart,
                rtol=1.0e-12,
                atol=1.0e-10,
            )
        ):
            raise ValueError(
                f"periodic density cell {key} has a Cartesian translation "
                "inconsistent with the column-vector lattice"
            )

        matrix = np.asarray(block)
        if np.iscomplexobj(matrix):
            raise ValueError(
                f"periodic density block g={key} is complex; a real-space "
                "density grid requires time-reversal-consistent real blocks"
            )
        matrix = np.ascontiguousarray(matrix, dtype=float)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"periodic density block g={key} has shape {matrix.shape}, "
                f"expected {expected_shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"periodic density block g={key} contains non-finite values"
            )
        # Only mathematically exact zero blocks may be skipped.  Tolerance-
        # based screening changes the converged density by discarding small
        # translated-cell contributions.
        if np.any(matrix != 0.0):
            live_cells.append((cell, matrix, key))

    rho_flat = np.zeros(n_points, dtype=float)
    for R in translations:
        # chi_ν(r - R): the home-cell AO label, translated by R.
        chi_ref = _plain_ao(_shifted_basis(basis, system, R))
        for cell, Dg, key in live_cells:
            shift = R + np.asarray(cell.r_cart, dtype=float)
            if key == (0, 0, 0):
                chi_g = chi_ref
            else:
                # chi_mu(r - R - R_g): shell mu on atom a evaluated at r
                # equals the AO of a basis whose atom a sits at
                # r_a + R + R_g.
                chi_g = _plain_ao(_shifted_basis(basis, system, shift))
            # rho(r) += S_muν D(g)_{muν} chi_mu(r-R) chi_ν(r-R-R_g).
            # Order matters for true multi-k densities: D(g) need not be
            # symmetric. This convention integrates to
            # S_g D(g):S(g), matching D(g)=S_k w_k exp(-ik.R_g)D(k)
            # and S(k)=S_g exp(+ik.R_g)S(g).
            rho_flat += np.einsum(
                "gi,ij,gj->g", chi_ref, Dg, chi_g, optimize=True,
            )

    if _return_workspace:
        # The workspace's ``chi_ref`` is the IMAGE-SUMMED home-cell AO
        # table, which is what ``build_j_long_range_periodic`` consumes
        # for its matrix elements. The density loop above no longer
        # forms that object (its image sum is the shared-R loop), so
        # build it explicitly here rather than handing back whichever
        # translation the loop happened to end on.
        #
        # NOTE for whoever owns the long-range J: the same tied-sum
        # question applies to ``J(g)_{muν} = dV S_r chi_mu(r-R_g) V(r)
        # chi_ν(r)``. With V periodic, the all-space integral folds to
        # ``int_cell V(r) S_R chi_mu(r-R-R_g) chi_ν(r-R)`` -- again ONE
        # shared lattice sum, not two independent ones. That path is
        # left byte-for-byte unchanged here because it is a different
        # quantity (a Fock element, not a density) on a route this
        # change has not validated; it is recorded in the bug entry as
        # a follow-up rather than silently altered.
        chi_ref_periodic = np.empty((n_points, basis.nbasis), dtype=float)
        for i in range(0, n_points, chunk_size):
            block = points[i:i + chunk_size]
            chi_ref_periodic[i:i + chunk_size, :] = evaluate_ao_periodic(
                basis, system, block, image_radius=ao_image_radius,
            )
        return (
            rho_flat.reshape(grid_shape_t),
            grid_shape_t,
            {
                "points": points,
                "chi_ref": chi_ref_periodic,
                "origin": origin_arr,
            },
        )
    return rho_flat.reshape(grid_shape_t), grid_shape_t


def evaluate_weighted_k_density_on_grid(
    basis: BasisSet,
    system: PeriodicSystem,
    weighted_density_per_k: Sequence[np.ndarray],
    kpoints_cart: np.ndarray,
    *,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    chunk_size: int = 200_000,
    ao_image_radius: int = 1,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Evaluate an exact weighted Bloch density on a primitive-cell grid.

    ``weighted_density_per_k[k]`` is ``w_k P(k)`` recovered from the
    *returned* real-space SCF density, not rebuilt from orbitals or
    occupations.  With the C++ ``+i k.T`` Bloch-AO convention this evaluates

    ``rho(r) = sum_k Re[chi_k(r) @ (w_k P(k)) @ chi_k(r).conj()]``.

    The translation cube is symmetric only along periodic axes.  Callers
    that recover the matrices from a finite BIPOLE lattice set must first
    certify its Born-von-Karman periodic representation by refolding onto
    every returned cell; this evaluator deliberately does not infer that
    scientific guarantee from the matrices alone.
    """
    lattice = np.asarray(system.lattice, dtype=float)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError(
            "weighted-k periodic density requires a finite 3x3 "
            "column-vector lattice"
        )
    if grid_shape is None:
        grid_shape_t = auto_grid(lattice, float(spacing_bohr))
    elif isinstance(grid_shape, int):
        grid_shape_t = (int(grid_shape),) * 3
    else:
        if len(grid_shape) != 3:
            raise ValueError("periodic density grid shape must have three axes")
        grid_shape_t = tuple(int(value) for value in grid_shape)
    if min(grid_shape_t) < 1:
        raise ValueError("periodic density grid shape must be positive")
    if int(chunk_size) < 1:
        raise ValueError("periodic density chunk_size must be positive")
    radius = int(ao_image_radius)
    if radius < 0 or radius != ao_image_radius:
        raise ValueError(
            "periodic density AO image radius must be a nonnegative integer"
        )

    kpts = np.asarray(kpoints_cart, dtype=float)
    if kpts.ndim != 2 or kpts.shape[1] != 3 or not np.isfinite(kpts).all():
        raise ValueError(
            "weighted-k periodic density requires finite (n_k, 3) k-points"
        )
    matrices = [
        np.asarray(matrix, dtype=np.complex128)
        for matrix in weighted_density_per_k
    ]
    if not matrices or len(matrices) != kpts.shape[0]:
        raise ValueError(
            "weighted-k periodic density requires one matrix per k-point; "
            f"got {len(matrices)} matrices and {kpts.shape[0]} k-points"
        )
    expected_shape = (int(basis.nbasis), int(basis.nbasis))
    for index, matrix in enumerate(matrices):
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValueError(
                f"weighted density matrix k={index} must be finite with shape "
                f"{expected_shape}; got {matrix.shape}"
            )
        scale = max(1.0, float(np.max(np.abs(matrix))))
        hermitian_error = float(np.max(np.abs(matrix - matrix.conj().T)))
        if hermitian_error > 5.0e-11 * scale:
            raise ValueError(
                f"weighted density matrix k={index} is not Hermitian; "
                f"max residual {hermitian_error:.3e}"
            )

    dim = max(0, min(3, int(getattr(system, "dim", 3))))
    ranges = [
        range(-radius, radius + 1) if axis < dim else (0,)
        for axis in range(3)
    ]

    integer_translations = np.asarray(list(itertools.product(*ranges)), dtype=float)
    translations = np.ascontiguousarray(integer_translations @ lattice.T, dtype=float)
    origin_arr = (
        np.asarray(origin, dtype=float).reshape(3)
        if origin is not None
        else np.zeros(3, dtype=float)
    )
    if not np.isfinite(origin_arr).all():
        raise ValueError("periodic density grid origin must be finite")

    nx, ny, nz = grid_shape_t
    n_points = nx * ny * nz
    # Cache all chi_k for one bounded point batch so each translated plain-AO
    # table is evaluated once, then phase-accumulated across k.  Calling the
    # C++ Bloch helper independently for each k would repeat the expensive AO
    # shell loop n_k times.  The 64 MiB cap bounds the complex workspace.
    workspace_bytes = 64 * 1024 * 1024
    bytes_per_point = max(1, len(matrices) * expected_shape[0] * 16)
    effective_chunk_size = min(
        int(chunk_size),
        max(1, workspace_bytes // bytes_per_point),
    )
    phases = np.exp(1.0j * (kpts @ translations.T))
    rho_flat = np.empty(n_points, dtype=float)
    for start in range(0, n_points, effective_chunk_size):
        stop = min(start + effective_chunk_size, n_points)
        linear = np.arange(start, stop, dtype=np.int64)
        ix = linear // (ny * nz)
        iy = (linear // nz) % ny
        iz = linear % nz
        frac = np.column_stack(
            (
                ix.astype(float) / float(nx),
                iy.astype(float) / float(ny),
                iz.astype(float) / float(nz),
            )
        )
        points = np.ascontiguousarray(origin_arr + frac @ lattice.T, dtype=float)
        chi_per_k = np.zeros(
            (len(matrices), stop - start, expected_shape[0]),
            dtype=np.complex128,
        )
        for translation_index, translation in enumerate(translations):
            plain_ao = np.asarray(
                evaluate_ao(basis, points - translation),
                dtype=float,
            )
            if plain_ao.shape != (stop - start, expected_shape[0]):
                raise ValueError(
                    "AO evaluator returned shape "
                    f"{plain_ao.shape}, expected "
                    f"{(stop - start, expected_shape[0])}"
                )
            for k_index in range(len(matrices)):
                chi_per_k[k_index] += (
                    phases[k_index, translation_index] * plain_ao
                )
        rho_chunk = np.zeros(stop - start, dtype=float)
        for k_index, matrix in enumerate(matrices):
            chi_k = chi_per_k[k_index]
            contribution = np.einsum(
                "pi,ij,pj->p",
                chi_k,
                matrix,
                chi_k.conj(),
                optimize=True,
            )
            real_scale = max(1.0, float(np.max(np.abs(contribution.real))))
            imaginary_error = float(np.max(np.abs(contribution.imag)))
            if imaginary_error > 5.0e-10 * real_scale:
                raise ValueError(
                    f"weighted density contraction k={k_index} has non-real "
                    f"residue {imaginary_error:.3e}"
                )
            rho_chunk += contribution.real
        rho_flat[start:stop] = rho_chunk

    return rho_flat.reshape(grid_shape_t), grid_shape_t


# ---------------------------------------------------------------------------
# Periodic long-range J builder
# ---------------------------------------------------------------------------

def build_j_long_range_periodic(
    basis: BasisSet,
    system: PeriodicSystem,
    D_real: LatticeMatrixSet,
    omega: float,
    *,
    grid_shape: Optional[Union[Tuple[int, int, int], int]] = None,
    origin: Optional[Sequence[float]] = None,
    spacing_bohr: float = 0.3,
    chunk_size: int = 200_000,
    output_cells: Optional[Sequence[int]] = None,
    ao_image_radius: int = 1,
    restore_g0_finite_part: bool = False,
) -> Union[np.ndarray, List[np.ndarray]]:
    """Long-range Hartree J for a periodic system, built from the
    multi-cell density via FFT Poisson convolution.

    Replaces the single-cell density assumption in
    :func:`vibeqc.build_j_long_range` with the full lattice sum
    ``r(r) = S_g S_{muν} D(g)_{muν} chi_mu(r) chi_ν(r - R_g)``, so bulk
    crystals with cell-to-cell density overlap work correctly.

    Parameters
    ----------
    basis, system
        AO basis and periodic system.
    D_real
        Real-space density matrix as a
        :class:`LatticeMatrixSet`.
    omega
        Ewald splitting parameter (1 / bohr). w-invariance of the
        composed J_SR + J_LR = J_full identity holds to ~0.3 % at
        typical FFT grid resolutions.
    grid_shape, origin, spacing_bohr, chunk_size
        Grid controls; see :func:`evaluate_periodic_density_on_grid`.
    output_cells
        If ``None`` (default), return a single ``(n_bf, n_bf)`` Γ-only
        J matrix: ``J(0)_{muν} = ∫ chi_mu(r) V_LR(r) chi_ν(r) dr``. Pass a
        sequence of lattice-cell indices (into ``D_real.cells``) to
        return the corresponding J(g) blocks as a list. Passing the
        full cell list recovers the J_LR needed for multi-k Bloch
        summation.

    Returns
    -------
    np.ndarray (n_bf, n_bf) if ``output_cells`` is None, else a list
    of (n_bf, n_bf) ndarrays in ``output_cells`` order.

    Notes
    -----
    G=0 gauge: V_LR(G=0) = 0 as in :mod:`vibeqc.ewald_j`. For a
    charged cell the Makov-Payne constant is the leading finite-box
    correction; see Phase 12e-c-4a for the derivation. For neutral
    crystals the Madelung offset cancels against matching nuclear
    contributions -- 12e-c-4c-iv wires that through the full SCF
    energy.
    """
    lat = np.asarray(system.lattice, dtype=float)

    rho_grid, grid_shape_t, _workspace = evaluate_periodic_density_on_grid(
        basis, system, D_real,
        grid_shape=grid_shape,
        origin=origin,
        spacing_bohr=spacing_bohr,
        chunk_size=chunk_size,
        ao_image_radius=ao_image_radius,
        _return_workspace=True,
    )

    V_lr_3d = solve_poisson_erf_screened(rho_grid, lat, float(omega))
    V_lr_flat = V_lr_3d.ravel()

    points = _workspace["points"]
    n_points = points.shape[0]
    cell_volume = abs(np.linalg.det(lat))
    dV = cell_volume / (grid_shape_t[0] * grid_shape_t[1] * grid_shape_t[2])

    # G=0 finite part of the erf kernel -- opt-in, exactly as in
    # :func:`vibeqc.build_j_long_range`; see its "G=0 gauge" notes.
    # Default keeps the pinned gauge (J_LR alone). Composition sites that
    # assert J_SR + J_LR = J_full must pass True, or the identity fails
    # by +pi Q^2 / (2 w^2 V_cell) (GRID-BACKEND-CONVERGES-WRONG).
    if restore_g0_finite_part and float(omega) > 0.0:
        q_cell = float(np.sum(rho_grid)) * dV
        V_lr_flat = V_lr_flat - (
            (np.pi / float(omega) ** 2) * (q_cell / cell_volume)
        )

    # ONE lattice sum, shared by both AO factors -- the same correction
    # the density builder above took in `2ef4c79fc`
    # (PERIODIC-DENSITY-GRID-DOUBLE-COUNTS-IMAGES).
    #
    #   J(g)_{muν} = int_all chi_mu(r - R_g) V(r) chi_ν(r) dr
    #              = dV S_R S_r V(r) chi_mu(r - R - R_g) chi_ν(r - R)
    #
    # with V lattice-periodic, so the all-space integral folds to the
    # cell with ONE sum over R.
    #
    # Until 2026-08-03 this routine image-summed the two factors
    # INDEPENDENTLY -- `chi_ref` (the workspace's image-summed table) for
    # ν and a separately image-summed `chi_g` for mu. Substituting
    # r -> r + R with V periodic shows why that is wrong:
    #
    #   S_{R,R'} S_r chi_mu(r - (R_g + R' - R)) V(r) chi_ν(r)
    #
    # so only the R' = R terms belong in block g; every other term
    # deposits a pair of separation R_g + R' - R into block g. The error
    # is CONTAMINATION BETWEEN LATTICE BLOCKS, not a scale factor, so a
    # single-block spot check can look correct while the set is wrong.
    # Routed by the GAPW chat, tracked as the long-range-J follow-up in
    # `agentic-loop/bug-claims.md`.
    lattice_vectors = np.asarray(lat, dtype=float)
    translations = [
        (i * lattice_vectors[:, 0]
         + j * lattice_vectors[:, 1]
         + k * lattice_vectors[:, 2])
        for i in range(-ao_image_radius, ao_image_radius + 1)
        for j in range(-ao_image_radius, ao_image_radius + 1)
        for k in range(-ao_image_radius, ao_image_radius + 1)
    ]

    def _plain_ao(basis_obj) -> np.ndarray:
        """AO table with NO image summing (the image sum is the R loop)."""
        out_ao = np.empty((n_points, basis.nbasis), dtype=float)
        for i in range(0, n_points, chunk_size):
            block = points[i:i + chunk_size]
            out_ao[i:i + chunk_size, :] = evaluate_ao_periodic(
                basis_obj, system, block, image_radius=0,
            )
        return out_ao

    if output_cells is None:
        # Γ-only J(0)_{muν} = dV S_R S_r chi_mu(r-R) V(r) chi_ν(r-R):
        # the same tied sum at R_g = 0.
        J = np.zeros((basis.nbasis, basis.nbasis), dtype=float)
        for R in translations:
            chi_R = _plain_ao(_shifted_basis(basis, system, R))
            J += dV * (chi_R.T @ (chi_R * V_lr_flat[:, None]))
        return 0.5 * (J + J.T)

    # Multi-cell J: accumulate every requested block inside the shared R
    # loop, so each translation's AO table is evaluated once and reused
    # across the g blocks that need it.
    out: List[np.ndarray] = [
        np.zeros((basis.nbasis, basis.nbasis), dtype=float)
        for _ in output_cells
    ]
    for R in translations:
        chi_ref_R = _plain_ao(_shifted_basis(basis, system, R))
        weighted_R = chi_ref_R * V_lr_flat[:, None]
        for slot, cell_idx in enumerate(output_cells):
            cell = D_real.cells[cell_idx]
            shift = R + np.asarray(cell.r_cart, dtype=float)
            if np.allclose(shift, R):
                chi_g = chi_ref_R
            else:
                # chi_mu(r - R - R_g): shell mu on atom a evaluated at r
                # equals the AO of a basis whose atom a sits at
                # r_a + R + R_g.
                chi_g = _plain_ao(_shifted_basis(basis, system, shift))
            out[slot] += dV * (chi_g.T @ weighted_R)
    return out
