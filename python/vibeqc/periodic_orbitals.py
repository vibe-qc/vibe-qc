"""Phase V3: periodic Bloch-orbital evaluation + volumetric writers.

Builds on the C++ Bloch-summed AO kernel ``evaluate_bloch_ao`` to give
users a one-call path from a multi-k SCF result to an XSF / cube
volumetric file that VESTA, XCrySDen, VMD, Avogadro and PyMOL read
directly.

The crystalline orbital is

    psi_{n,k}(r) = S_T  e^{i k.T}  S_mu  C_mu(k) . chi_mu(r - T)
               = S_mu  C_mu(k) . chi_mu^k(r)

so a single Bloch-summed AO matrix chi_mu^k(r) -- built once per (k, grid)
pair via the C++ kernel -- can be contracted against any number of MO
columns at the same k.

Public API
----------

* :func:`make_primitive_cell_grid` -- build a uniform fractional-coordinate
  sample of one primitive unit cell, excluding the duplicate boundary
  voxel that XSF expects to be implicit. Returns the Cartesian grid
  points, the resolved shape, the origin, and the span vectors.
* :func:`evaluate_bloch_orbital` -- one-line contraction chi^k @ C(:, n).
* :func:`write_xsf_mo` -- primitive-cell XSF for a single Bloch orbital.
* :func:`write_xsf_density` -- primitive-cell XSF for the SCF density;
  thin wrapper over :func:`evaluate_periodic_density_on_grid`.
* :func:`write_cube_mo_periodic` -- supercell cube file for viewers that
  cannot read XSF (cube has no lattice-vector concept, so we replicate
  the primitive cell N₁xN₂xN₃ times into an axis-aligned box).

Mode selection for the volumetric scalar
----------------------------------------

The Bloch orbital is in general complex. Every writer takes a
``component`` keyword:

* ``"real"`` (default for orbitals) -- Re psi. Keeps lobes / nodes;
  carries an arbitrary global phase (eigenvector gauge). At
  time-reversal-invariant momenta (Γ, X, ...) psi is real up to a global
  sign and Re psi is unambiguous up to that flip.
* ``"imag"`` -- Im psi. Useful as a gauge-paired view at general k.
* ``"abs"`` -- |psi|. Gauge-invariant magnitude.
* ``"density"`` -- |psi|^2. Gauge-invariant probability density. No nodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    Atom,
    BasisSet,
    LatticeMatrixSet,
    Molecule,
    PeriodicSystem,
    direct_lattice_cells,
    evaluate_bloch_ao,
)
from .cube import CubeGrid, _write_cube_data, _write_cube_header
from .periodic_density import evaluate_periodic_density_on_grid
from .xsf import write_xsf_volume


__all__ = [
    "PrimitiveCellGrid",
    "make_primitive_cell_grid",
    "evaluate_bloch_orbital",
    "write_xsf_mo",
    "write_xsf_density",
    "write_cube_mo_periodic",
]


# Default real-space lattice cutoff for the Bloch sum (bohr). Matches
# ``LatticeSumOptions.cutoff_bohr`` so the AO truncation here is consistent
# with the lattice sums the SCF used to build C(k). Far enough out that AO
# overlap of any reasonable Gaussian basis with the central cell is below
# 1e-12 -- i.e. the truncation error never shows up in plots.
_DEFAULT_LATTICE_CUTOFF_BOHR = 15.0


# ---------------------------------------------------------------------------
# Component-reduction helper
# ---------------------------------------------------------------------------

def _reduce_component(psi: np.ndarray, component: str) -> np.ndarray:
    """Project a complex orbital array down to the real scalar that
    actually gets written to the file."""
    c = component.lower()
    if c == "real":
        return psi.real
    if c == "imag":
        return psi.imag
    if c == "abs":
        return np.abs(psi)
    if c == "density":
        return (psi.real * psi.real + psi.imag * psi.imag)
    raise ValueError(
        f"component must be 'real' | 'imag' | 'abs' | 'density', got {component!r}"
    )


# ---------------------------------------------------------------------------
# Primitive-cell grid factory
# ---------------------------------------------------------------------------

class PrimitiveCellGrid:
    """A uniform sample of one primitive cell at integer fractional
    coordinates ``(i / n_a, j / n_b, k / n_c)`` for ``i in [0, n_a)`` etc.

    The duplicate periodic image at fractional coordinate 1 is *not*
    sampled -- XSF / VESTA expect that boundary point to be implicit.

    Attributes
    ----------
    origin : (3,) ndarray
        Cartesian origin of voxel ``(0, 0, 0)`` in bohr.
    span : (3, 3) ndarray
        Rows are the three spanning vectors of the grid in bohr. For a
        full primitive cell these are the lattice columns transposed.
    shape : (n_a, n_b, n_c)
        Number of voxels along each lattice direction.
    points : (n_a . n_b . n_c, 3) ndarray
        Cartesian voxel centers in bohr. Linearised in C order
        ``(i, j, k) -> (i . n_b + j) . n_c + k`` so reshape ``(n_a, n_b, n_c)``
        recovers the lattice-fractional layout.
    """

    __slots__ = ("origin", "span", "shape", "points")

    def __init__(
        self,
        origin: np.ndarray,
        span: np.ndarray,
        shape: Tuple[int, int, int],
        points: np.ndarray,
    ) -> None:
        self.origin = origin
        self.span = span
        self.shape = shape
        self.points = points


def make_primitive_cell_grid(
    system: PeriodicSystem,
    *,
    spacing_bohr: float = 0.2,
    grid_shape: Optional[Union[int, Tuple[int, int, int]]] = None,
    origin: Optional[Sequence[float]] = None,
) -> PrimitiveCellGrid:
    """Build a primitive-cell grid suitable for direct hand-off to
    :func:`write_xsf_volume`.

    The grid spans one primitive cell along each lattice vector
    (including non-orthogonal ones), with voxels at fractional
    coordinates ``(i / n_a, j / n_b, k / n_c)``. The duplicate
    boundary voxel at fractional coordinate 1 is intentionally
    omitted: XSF treats voxel ``n_a`` as the periodic image of voxel
    ``0`` and would interpret an explicit duplicate as a discontinuity.

    For ``dim < 3`` the vacuum-axis lattice columns are sampled exactly
    the same way -- i.e. one voxel per Cartesian extent of that vacuum
    column. Vacuum columns are typically ``>= 30 bohr`` so spacing
    determines the count.

    Parameters
    ----------
    system
        The periodic system; ``system.lattice`` columns are the spanning
        vectors.
    spacing_bohr
        Target voxel size along each lattice vector. Auto-rounded to
        the nearest non-zero integer per axis from ``ceil(|a_i| / spacing)``.
    grid_shape
        Override the auto count. Either an ``int`` (cubic ``n x n x n``)
        or a triple ``(n_a, n_b, n_c)``. Wins over ``spacing_bohr``.
    origin
        Cartesian origin in bohr. Default ``(0, 0, 0)``.
    """
    L = np.asarray(system.lattice, dtype=float)
    if L.shape != (3, 3):
        raise ValueError(f"system.lattice must be (3, 3), got {L.shape}")
    a = L[:, 0]
    b = L[:, 1]
    c = L[:, 2]

    if grid_shape is None:
        # ceil(|a_i| / spacing); guarantee at least one voxel per axis.
        na = max(1, int(np.ceil(np.linalg.norm(a) / spacing_bohr)))
        nb = max(1, int(np.ceil(np.linalg.norm(b) / spacing_bohr)))
        nc = max(1, int(np.ceil(np.linalg.norm(c) / spacing_bohr)))
        shape = (na, nb, nc)
    elif isinstance(grid_shape, int):
        shape = (int(grid_shape), int(grid_shape), int(grid_shape))
    else:
        shape = tuple(int(x) for x in grid_shape)
    na, nb, nc = shape

    if origin is None:
        origin_arr = np.zeros(3)
    else:
        origin_arr = np.asarray(origin, dtype=float).reshape(3)

    # Cartesian voxel centers. Indexing="ij" gives I.shape == (na, nb, nc)
    # with I[i, j, k] = i, etc. ravel() then walks i fastest? No --
    # numpy default ravel is C order (last axis fastest), so k runs
    # fastest. Either traversal works as long as we reshape consistently
    # downstream.
    fa = np.arange(na, dtype=float) / na
    fb = np.arange(nb, dtype=float) / nb
    fc = np.arange(nc, dtype=float) / nc
    Fa, Fb, Fc = np.meshgrid(fa, fb, fc, indexing="ij")
    pts = (origin_arr[None, :]
           + Fa.ravel()[:, None] * a[None, :]
           + Fb.ravel()[:, None] * b[None, :]
           + Fc.ravel()[:, None] * c[None, :])

    return PrimitiveCellGrid(
        origin=origin_arr,
        span=L.T.copy(),
        shape=shape,
        points=pts,
    )


# ---------------------------------------------------------------------------
# Lattice-translation helper
# ---------------------------------------------------------------------------

def _lattice_translations(
    system: PeriodicSystem,
    cutoff_bohr: float,
) -> np.ndarray:
    """Stack ``r_cart`` of every cell within ``cutoff_bohr`` into an
    ``(n_T, 3)`` float64 array."""
    cells = direct_lattice_cells(system, float(cutoff_bohr))
    return np.array([np.asarray(c.r_cart, dtype=float) for c in cells],
                    dtype=float).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Python wrapper: psi_{n,k}(r) on a flat point list
# ---------------------------------------------------------------------------

def evaluate_bloch_orbital(
    basis: BasisSet,
    system: PeriodicSystem,
    points: np.ndarray,
    C_at_k: np.ndarray,
    k_cart: np.ndarray,
    band_index: int,
    *,
    lattice_cutoff_bohr: float = _DEFAULT_LATTICE_CUTOFF_BOHR,
    lattice_translations: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluate the Bloch crystalline orbital psi_{n,k}(r) at a flat array
    of Cartesian grid points.

    Parameters
    ----------
    basis
        Unit-cell AO basis.
    system
        :class:`PeriodicSystem` -- used to enumerate the lattice
        translations T entering the Bloch sum (unless
        ``lattice_translations`` is given explicitly).
    points
        Cartesian grid points in bohr, shape ``(n_pts, 3)``.
    C_at_k
        Complex multi-k MO coefficient matrix at this k, shape
        ``(n_basis, n_bands)``. Columns are MOs in the AO basis;
        ``BandDiag.coefficients`` from :func:`vibeqc.diagonalize_bloch`
        and the per-k entries of ``PeriodicRHFMultiKEwaldResult.mo_coeffs``
        match this convention.
    k_cart
        Cartesian k-vector in bohr⁻¹, shape ``(3,)``.
    band_index
        Zero-based column of ``C_at_k`` to project onto.
    lattice_cutoff_bohr
        Real-space lattice-sum cutoff for the Bloch translation
        enumeration. Default matches
        :class:`vibeqc.LatticeSumOptions.cutoff_bohr`.
    lattice_translations
        Override: pass a precomputed ``(n_T, 3)`` array of Cartesian
        lattice shifts. Reuse this across many (orbital, k) pairs
        sharing the same grid + system to amortise the
        ``direct_lattice_cells`` call.

    Returns
    -------
    psi : np.ndarray, complex, shape (n_pts,)
        psi_{n,k}(r) evaluated at each grid point.
    """
    C = np.asarray(C_at_k)
    if C.ndim != 2:
        raise ValueError(
            f"C_at_k must be 2D (n_basis, n_bands), got shape {C.shape}"
        )
    if not (0 <= band_index < C.shape[1]):
        raise IndexError(
            f"band_index {band_index} out of range for C_at_k with "
            f"shape {C.shape}"
        )

    pts = np.ascontiguousarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (n_pts, 3), got shape {pts.shape}")

    if lattice_translations is None:
        Ts = _lattice_translations(system, lattice_cutoff_bohr)
    else:
        Ts = np.ascontiguousarray(lattice_translations, dtype=float)
        if Ts.ndim != 2 or Ts.shape[1] != 3:
            raise ValueError(
                "lattice_translations must be (n_T, 3), got shape "
                f"{Ts.shape}"
            )

    k = np.asarray(k_cart, dtype=float).reshape(3)
    chi_k = evaluate_bloch_ao(basis, pts, k, Ts)        # (n_pts, n_bf) complex
    return chi_k @ C[:, band_index].astype(complex)     # (n_pts,) complex


# ---------------------------------------------------------------------------
# Periodic XSF: single Bloch orbital
# ---------------------------------------------------------------------------

def write_xsf_mo(
    path: Union[str, Path],
    system: PeriodicSystem,
    basis: BasisSet,
    C_at_k: np.ndarray,
    k_cart: np.ndarray,
    band_index: int,
    *,
    grid_shape: Optional[Union[int, Tuple[int, int, int]]] = None,
    spacing_bohr: float = 0.2,
    lattice_cutoff_bohr: float = _DEFAULT_LATTICE_CUTOFF_BOHR,
    component: str = "real",
    name: Optional[str] = None,
) -> Path:
    """Write a periodic XSF file with a single Bloch orbital sampled on a
    primitive-cell grid.

    Suitable for VESTA / XCrySDen. The grid spans one primitive cell
    along each lattice vector and excludes the duplicate boundary voxel
    (XSF treats it as the implicit periodic image).

    Parameters
    ----------
    component
        See module docstring. Default ``"real"`` writes Re psi; for general
        k consider ``"density"`` (|psi|^2, gauge-invariant).
    """
    grid = make_primitive_cell_grid(
        system, spacing_bohr=spacing_bohr, grid_shape=grid_shape,
    )
    psi = evaluate_bloch_orbital(
        basis, system, grid.points, C_at_k, k_cart, band_index,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
    )
    scalar = _reduce_component(psi, component).reshape(grid.shape)

    tag = name or f"bloch_orbital_{band_index}_{component}"
    return write_xsf_volume(
        path, system, data=scalar, name=tag,
        origin=grid.origin, span=grid.span,
    )


# ---------------------------------------------------------------------------
# Periodic XSF: SCF density
# ---------------------------------------------------------------------------

def write_xsf_density(
    path: Union[str, Path],
    system: PeriodicSystem,
    basis: BasisSet,
    D_real: LatticeMatrixSet,
    *,
    grid_shape: Optional[Union[int, Tuple[int, int, int]]] = None,
    spacing_bohr: float = 0.2,
    name: str = "density",
) -> Path:
    """Convenience wrapper: evaluate the periodic SCF density on a
    primitive-cell grid and write a periodic XSF file.

    Wraps :func:`vibeqc.evaluate_periodic_density_on_grid` and hands
    the result to :func:`write_xsf_volume`.

    Parameters
    ----------
    D_real
        Real-space density matrix as a :class:`LatticeMatrixSet`,
        e.g. from :func:`vibeqc.real_space_density_from_kpoints`.
    grid_shape, spacing_bohr
        Grid controls; ``grid_shape`` overrides ``spacing_bohr``.
    """
    if grid_shape is None:
        # Match make_primitive_cell_grid's auto-shape rule so the XSF
        # voxel count is consistent across density / orbital writers.
        L = np.asarray(system.lattice, dtype=float)
        shape = tuple(
            max(1, int(np.ceil(np.linalg.norm(L[:, i]) / spacing_bohr)))
            for i in range(3)
        )
    elif isinstance(grid_shape, int):
        shape = (int(grid_shape), int(grid_shape), int(grid_shape))
    else:
        shape = tuple(int(x) for x in grid_shape)

    rho, _ = evaluate_periodic_density_on_grid(
        basis, system, D_real,
        grid_shape=shape,
        spacing_bohr=spacing_bohr,
    )
    return write_xsf_volume(path, system, data=rho, name=name)


# ---------------------------------------------------------------------------
# Cube file (supercell view)
# ---------------------------------------------------------------------------

def _replicate_atoms(
    system: PeriodicSystem,
    n_replica: Tuple[int, int, int],
) -> Tuple[List[Atom], np.ndarray, np.ndarray]:
    """Replicate the primitive-cell atoms across an N₁xN₂xN₃ supercell
    centered on the origin. Returns (atoms, supercell_origin, supercell_span)
    in bohr; the supercell box is axis-aligned (cube format requirement
    for vibe-qc's writer)."""
    n1, n2, n3 = n_replica
    L = np.asarray(system.lattice, dtype=float)
    if not np.allclose(L - np.diag(np.diag(L)), 0.0):
        raise ValueError(
            "write_cube_mo_periodic requires an orthorhombic cell -- "
            "the cube format we emit is axis-aligned. Use write_xsf_mo "
            "for arbitrary lattices."
        )
    a = L[0, 0]
    b = L[1, 1]
    c = L[2, 2]
    # Center the supercell on the origin: atoms span -n_i/2 .. +n_i/2 cells.
    i0 = -(n1 // 2)
    j0 = -(n2 // 2)
    k0 = -(n3 // 2)

    atoms: List[Atom] = []
    for i in range(i0, i0 + n1):
        for j in range(j0, j0 + n2):
            for k in range(k0, k0 + n3):
                shift = np.array([i * a, j * b, k * c])
                for at in system.unit_cell:
                    x, y, z = at.xyz
                    atoms.append(Atom(int(at.Z),
                                      [float(x + shift[0]),
                                       float(y + shift[1]),
                                       float(z + shift[2])]))
    sc_origin = np.array([i0 * a, j0 * b, k0 * c])
    sc_span = np.array([n1 * a, n2 * b, n3 * c])
    return atoms, sc_origin, sc_span


def write_cube_mo_periodic(
    path: Union[str, Path],
    system: PeriodicSystem,
    basis: BasisSet,
    C_at_k: np.ndarray,
    k_cart: np.ndarray,
    band_index: int,
    *,
    n_replica: Tuple[int, int, int] = (3, 3, 3),
    spacing_bohr: float = 0.2,
    lattice_cutoff_bohr: float = _DEFAULT_LATTICE_CUTOFF_BOHR,
    component: str = "real",
    title: Optional[str] = None,
) -> Path:
    """Write a Bloch orbital as a Gaussian cube file over an
    N₁xN₂xN₃ axis-aligned supercell centered on the origin.

    Cube has no native lattice-vector concept, so this is the standard
    "supercell cube" workaround for VMD / Avogadro / PyMOL users -- the
    file is a regular molecular cube whose atom block has been
    replicated across the visible cells. For the proper periodic
    representation use :func:`write_xsf_mo` (XSF / VESTA / XCrySDen).

    Restricted to orthorhombic cells: the cube voxel block we emit is
    axis-aligned. Non-orthogonal cells raise ``ValueError``.

    Parameters
    ----------
    n_replica
        ``(n₁, n₂, n₃)`` -- how many primitive cells to tile along each
        Cartesian axis. The supercell is centered on the origin (running
        from ``-⌊n_i/2⌋.a_i`` to ``+⌈n_i/2⌉.a_i``).
    spacing_bohr
        Target voxel size in bohr.
    component
        See module docstring.
    """
    atoms, sc_origin, sc_span = _replicate_atoms(system, n_replica)
    a, b, c = float(sc_span[0]), float(sc_span[1]), float(sc_span[2])

    nx = max(2, int(np.ceil(a / spacing_bohr)))
    ny = max(2, int(np.ceil(b / spacing_bohr)))
    nz = max(2, int(np.ceil(c / spacing_bohr)))
    spacing_xyz = np.array([a / nx, b / ny, c / nz])
    grid = CubeGrid(origin=sc_origin, spacing=spacing_xyz, shape=(nx, ny, nz))

    psi = evaluate_bloch_orbital(
        basis, system, grid.points(), C_at_k, k_cart, band_index,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
    )
    scalar = _reduce_component(psi, component).reshape(grid.shape)

    # Cube needs a Molecule for the atom block; the replicated atoms live
    # in a transient Molecule that's only used as a write-time formatter.
    mol = Molecule(atoms, system.charge, system.multiplicity)

    p = Path(path)
    with p.open("w") as out:
        _write_cube_header(
            out,
            title=title or f"vibeqc Bloch orbital band {band_index} ({component})",
            comment=(f"k_cart=({k_cart[0]:.6f},{k_cart[1]:.6f},{k_cart[2]:.6f}) "
                     f"bohr^-1; replica={n_replica}"),
            mol=mol, grid=grid,
        )
        _write_cube_data(out, scalar)
    return p
