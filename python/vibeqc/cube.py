"""Gaussian-cube volumetric data writer.

The Gaussian cube format is the de-facto interchange format for
molecular volumetric data -- densities, molecular orbitals,
electrostatic potentials. Most viewers (VMD, Avogadro, PyMOL,
ChimeraX, ...) read it directly.

Layout (the "Gaussian 98" convention):

    line 1  : title
    line 2  : second comment / property tag
    line 3  : N_atoms  x_origin  y_origin  z_origin   [N_val]
    line 4  : N_x      vx_x      vx_y      vx_z       (voxel along x)
    line 5  : N_y      vy_x      vy_y      vy_z
    line 6  : N_z      vz_x      vz_y      vz_z
    line 7+ : Z  charge  x  y  z   for each atom
    then    : data, scanned x_outer, z_inner; 6 floats / line

A *positive* N_atoms means scalar volumetric data and ``N_val`` is
omitted; a *negative* N_atoms signals "multiple values per voxel"
(used here for stacked MOs).

All coordinates and voxel vectors are in **bohr** -- that's the cube
spec.

Public API
----------

* :func:`write_cube_density` -- total electron density on a uniform
  grid (right-hand rule axis-aligned bounding box around the molecule).
* :func:`write_cube_mo` -- one Kohn-Sham / Hartree-Fock MO.
* :func:`write_cube_mos` -- a stack of MOs in a single multi-value cube
  (consumed by VMD / Avogadro as a multi-frame volume).

A grid helper :func:`make_uniform_grid` wraps the typical "wrap a box
around the molecule with N_x x N_y x N_z voxels and ``padding`` bohr
of breathing room" pattern. Pass it a custom ``origin`` /
``spacing`` if you need a finer / coarser / off-center grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import BasisSet, Molecule, evaluate_ao

__all__ = [
    "CubeGrid",
    "make_uniform_grid",
    "make_uniform_grid_for_periodic",
    "evaluate_ao_on_grid",
    "evaluate_single_primitive_on_grid",
    "write_cube_density",
    "write_cube_mo",
    "write_cube_mos",
]


# ---------------------------------------------------------------------------
# Grid container + builder
# ---------------------------------------------------------------------------


@dataclass
class CubeGrid:
    """Uniform axis-aligned grid in bohr.

    ``origin`` is the (x, y, z) of voxel (0, 0, 0). ``spacing`` is the
    diagonal of three voxel widths along each Cartesian axis.
    Non-orthogonal cube grids are supported by the spec but uncommon in
    QC viewers; we keep things axis-aligned.
    """

    origin: np.ndarray  # (3,)  bohr
    spacing: np.ndarray  # (3,)  bohr (positive)
    shape: Tuple[int, int, int]  # (n_x, n_y, n_z)

    @property
    def n_points(self) -> int:
        return int(np.prod(self.shape))

    def points(self) -> np.ndarray:
        """All voxel centers as a ``(n_points, 3)`` array, scanned with
        x as the outer index and z as the inner index -- the order the
        cube format expects."""
        nx, ny, nz = self.shape
        ix = np.arange(nx) * self.spacing[0] + self.origin[0]
        iy = np.arange(ny) * self.spacing[1] + self.origin[1]
        iz = np.arange(nz) * self.spacing[2] + self.origin[2]
        # Broadcast in (x, y, z) order -- outer x, inner z.
        X, Y, Z = np.meshgrid(ix, iy, iz, indexing="ij")
        return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def make_uniform_grid(
    mol: Molecule,
    *,
    spacing: float = 0.2,
    padding: float = 4.0,
) -> CubeGrid:
    """Wrap an axis-aligned box around the molecule with ``padding``
    bohr of headroom and a cubic voxel of ``spacing`` bohr.

    ``spacing = 0.2`` and ``padding = 4`` gives roughly 100^3 ≈ 10⁶
    voxels for a small molecule -- enough to render densities and MOs
    smoothly and small enough to write in a second.
    """
    coords = np.asarray([list(a.xyz) for a in mol.atoms], dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 3)
    if coords.size == 0:
        raise ValueError("make_uniform_grid: molecule has no atoms")
    lo = coords.min(axis=0) - padding
    hi = coords.max(axis=0) + padding
    extent = hi - lo
    n = np.ceil(extent / spacing).astype(int) + 1
    return CubeGrid(
        origin=lo,
        spacing=np.array([spacing, spacing, spacing]),
        shape=(int(n[0]), int(n[1]), int(n[2])),
    )


def make_uniform_grid_for_periodic(
    atoms_or_coords,
    *,
    spacing: float = 0.15,
    padding: float = 8.0,
    lattice_vectors: np.ndarray | None = None,
) -> CubeGrid:
    """Build a uniform grid around a set of atoms or coordinates,
    optionally aligned to the periodic unit cell.

    When ``lattice_vectors`` is provided, the grid origin is snapped
    to the minimum corner of the cell (shifted outward by ``padding``)
    and the extent covers the cell diagonals.

    Parameters
    ----------
    atoms_or_coords
        Either a sequence of objects with ``.xyz`` attributes (atoms),
        or an ``(n, 3)`` array of Cartesian positions in bohr.
    spacing
        Voxel spacing in bohr (default 0.15 -- finer than the
        molecular default because individual primitives can be
        spatially compact).
    padding
        Extra headroom in bohr beyond the bounding box (default 8.0).
    lattice_vectors
        Optional (3, 3) row-vectors in bohr.  When provided, the grid
        spans the cell extended by ``padding`` on both sides.

    Returns
    -------
    CubeGrid
    """
    # Accept either a list of atom-like objects or a raw (n,3) array.
    if hasattr(atoms_or_coords, "shape"):
        coords = np.asarray(atoms_or_coords, dtype=float)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 3)
    else:
        coords = np.asarray(
            [list(getattr(a, "xyz", a)) for a in atoms_or_coords],
            dtype=float,
        )
        if coords.ndim == 1:
            coords = coords.reshape(-1, 3)
    if coords.size == 0:
        raise ValueError("make_uniform_grid_for_periodic: no atoms/coordinates")

    if lattice_vectors is not None:
        lv = np.asarray(lattice_vectors, dtype=float)
        # Grid origin: cell corner minus padding.
        lo = np.zeros(3, dtype=float) - padding
        # Grid extent: cell diagonals plus padding on both sides.
        hi = lv.sum(axis=0) + padding
    else:
        lo = coords.min(axis=0) - padding
        hi = coords.max(axis=0) + padding

    extent = hi - lo
    n = np.ceil(extent / spacing).astype(int) + 1
    return CubeGrid(
        origin=lo,
        spacing=np.array([spacing, spacing, spacing]),
        shape=(int(n[0]), int(n[1]), int(n[2])),
    )


# ---------------------------------------------------------------------------
# Volumetric scalar evaluation
# ---------------------------------------------------------------------------


def _density_on_grid(
    D: np.ndarray,
    basis: BasisSet,
    grid: CubeGrid,
    *,
    chunk_size: int = 200_000,
) -> np.ndarray:
    """r(r) = S_{muν} D_{muν} chi_mu(r) chi_ν(r), evaluated voxel-by-voxel in
    chunks so the (n_points, n_basis) AO matrix never exceeds a few
    hundred MB on big grids."""
    pts = grid.points()
    out = np.empty(pts.shape[0], dtype=float)
    for i in range(0, pts.shape[0], chunk_size):
        block = pts[i : i + chunk_size]
        chi = evaluate_ao(basis, block)  # (m, nb)
        # r = sum_{muν} D_{muν} chi_mu chi_ν = S_mu chi_mu (D chiᵀ)_mu
        out[i : i + chunk_size] = np.einsum("mi,ij,mj->m", chi, D, chi)
    return out.reshape(grid.shape)


def _mo_on_grid(
    C_col: np.ndarray,
    basis: BasisSet,
    grid: CubeGrid,
    *,
    chunk_size: int = 200_000,
) -> np.ndarray:
    """phi(r) = S_mu C_mu chi_mu(r) for one MO (vector ``C_col``)."""
    pts = grid.points()
    out = np.empty(pts.shape[0], dtype=float)
    for i in range(0, pts.shape[0], chunk_size):
        block = pts[i : i + chunk_size]
        chi = evaluate_ao(basis, block)
        out[i : i + chunk_size] = chi @ C_col
    return out.reshape(grid.shape)


def evaluate_ao_on_grid(
    basis: BasisSet,
    grid: CubeGrid,
    ao_indices: Sequence[int],
    *,
    chunk_size: int = 100_000,
) -> list[np.ndarray]:
    """Evaluate a list of contracted AOs on a grid.

    Evaluates the full AO matrix chunk-by-chunk and extracts the
    requested columns.  Returns one 3-D ``(nx, ny, nz)`` array per
    ``ao_indices`` entry, in the same order.

    This is the workhorse for :func:`qvf_ao_data`: a single evaluator
    pass serves all requested AOs.
    """
    idx_list = list(ao_indices)
    if not idx_list:
        return []
    pts = grid.points()
    accum = [np.empty(pts.shape[0], dtype=float) for _ in idx_list]
    for i in range(0, pts.shape[0], chunk_size):
        block = pts[i : i + chunk_size]
        chi = evaluate_ao(basis, block)  # (m, nb)
        for j, ao_idx in enumerate(idx_list):
            accum[j][i : i + chunk_size] = chi[:, ao_idx]
    return [a.reshape(grid.shape) for a in accum]


def evaluate_single_primitive_on_grid(
    basis: BasisSet,
    molecule: Molecule,
    shell_idx: int,
    prim_idx: int,
    grid: CubeGrid,
    *,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Evaluate **one primitive Gaussian** on a grid.

    Builds a temporary single-primitive :class:`BasisSet` so the
    C++ evaluator sees only that one function.  This is how we
    inspect individual primitives before contraction.

    Parameters
    ----------
    basis
        The full basis (needed to read shell metadata).
    molecule
        The :class:`Molecule` that owns the atoms.
    shell_idx
        Which contracted shell the primitive belongs to.
    prim_idx
        Which primitive *within* that shell.
    grid
        Evaluation grid in bohr.

    Returns
    -------
    (nx, ny, nz) float64 array.
    """
    shells = basis.shells()
    shell = shells[shell_idx]
    exp = shell.exponents[prim_idx]
    coeff = shell.coefficients[prim_idx]
    # Build a single-primitive BasisSet.  Coefficients are already
    # libint-normalized (they came from BasisSet.shells()), so we
    # pass coefficients_pre_normalized=True to keep them as-is.
    from ._vibeqc_core import BasisSet as _BasisSet
    from ._vibeqc_core import ShellInfo as _ShellInfo

    prim_shell = _ShellInfo()
    prim_shell.atom_index = shell.atom_index
    prim_shell.l = shell.l
    prim_shell.pure = shell.pure
    prim_shell.exponents = [float(exp)]
    prim_shell.coefficients = [float(coeff)]
    prim_shell.origin = list(shell.origin)
    prim_basis = _BasisSet(
        molecule,
        [prim_shell],
        name=f"<primitive s{shell_idx}p{prim_idx}>",
        coefficients_pre_normalized=True,
    )
    pts = grid.points()
    out = np.empty(pts.shape[0], dtype=float)
    for i in range(0, pts.shape[0], chunk_size):
        block = pts[i : i + chunk_size]
        chi = evaluate_ao(prim_basis, block)  # (m, 1)
        out[i : i + chunk_size] = chi[:, 0]
    return out.reshape(grid.shape)


def _ao_ranges(basis: BasisSet) -> list[tuple[int, int]]:
    """Return (ao_start, ao_end) global AO index ranges for each shell.

    Uses the libint ordering: shells in BasisSet.shells() order,
    spherical m = -l ... +l within each shell.
    """
    shells = basis.shells()
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for s in shells:
        n_ao = 2 * s.l + 1 if s.pure else (s.l + 1) * (s.l + 2) // 2
        ranges.append((cursor, cursor + n_ao))
        cursor += n_ao
    return ranges


def _cartesian_component_powers(l: int) -> list[tuple[int, int, int]]:
    """Return the ``(lx, ly, lz)`` powers of a Cartesian shell's components.

    The counterpart of the ``m = -l ... +l`` ordering :func:`_ao_ranges`
    documents for pure shells: libint orders a Cartesian shell's
    ``(l+1)(l+2)/2`` monomials lexicographically with ``lx`` decreasing
    first, then ``ly`` -- a d shell is ``xx, xy, xz, yy, yz, zz``.  Same
    ordering the QVF format spec mandates in Appendix A.2.
    """
    return [
        (lx, ly, l - lx - ly)
        for lx in range(l, -1, -1)
        for ly in range(l - lx, -1, -1)
    ]


# ---------------------------------------------------------------------------
# Cube file writer
# ---------------------------------------------------------------------------


def _format_atoms_block(mol: Molecule) -> str:
    lines = []
    for a in mol.atoms:
        x, y, z = a.xyz
        # The "atomic charge" column is conventionally the nuclear
        # charge as a float -- viewers read Z and ignore the value, but
        # the canonical convention is to put Z there too.
        lines.append(f"{a.Z:5d} {float(a.Z):12.6f} {x:12.6f} {y:12.6f} {z:12.6f}")
    return "\n".join(lines)


def _write_cube_header(
    out,
    *,
    title: str,
    comment: str,
    mol: Molecule,
    grid: CubeGrid,
    n_values: Optional[int] = None,
    extra_header_int: Optional[int] = None,
    extra_header_ints: Optional[Sequence[int]] = None,
) -> None:
    # Neutralise bidi / zero-width / control characters in the two
    # free-text comment lines before they reach the file: a tainted title
    # (e.g. a run_job output basename carrying U+202E RIGHT-TO-LEFT
    # OVERRIDE) must not be written raw into a header that a terminal or
    # editor later renders. See vibeqc.output._text_safety.
    from .output._text_safety import scrub_output_text

    out.write(f"{scrub_output_text(title)}\n")
    out.write(f"{scrub_output_text(comment)}\n")
    n_atoms = len(mol.atoms)
    if n_values is not None and n_values > 1:
        # Multi-value cube: negate atom count, append number of values.
        out.write(
            f"{-n_atoms:5d} {grid.origin[0]:12.6f} {grid.origin[1]:12.6f} "
            f"{grid.origin[2]:12.6f} {n_values:5d}\n"
        )
    else:
        out.write(
            f"{n_atoms:5d} {grid.origin[0]:12.6f} {grid.origin[1]:12.6f} "
            f"{grid.origin[2]:12.6f}\n"
        )
    nx, ny, nz = grid.shape
    out.write(f"{nx:5d} {grid.spacing[0]:12.6f}     0.000000     0.000000\n")
    out.write(f"{ny:5d}     0.000000 {grid.spacing[1]:12.6f}     0.000000\n")
    out.write(f"{nz:5d}     0.000000     0.000000 {grid.spacing[2]:12.6f}\n")
    out.write(_format_atoms_block(mol))
    out.write("\n")
    # For multi-value cubes a line with "<n_values>  <id_0>  <id_1> ..."
    # follows the atoms -- viewers use it to label the volumes.
    if extra_header_ints is not None:
        toks = [str(len(extra_header_ints))] + [str(i) for i in extra_header_ints]
        out.write(" " + "  ".join(toks) + "\n")


def _write_cube_data(out, data: np.ndarray) -> None:
    """Write a (..., n_z) array (or stacked) flat with 6 numbers per
    line. Cube wants z as the inner running index."""
    flat = data.reshape(-1)
    for i in range(0, flat.size, 6):
        chunk = flat[i : i + 6]
        out.write(" ".join(f"{x:13.5e}" for x in chunk) + "\n")


def write_cube_density(
    path: Union[str, Path],
    D: np.ndarray,
    basis: BasisSet,
    mol: Molecule,
    *,
    grid: Optional[CubeGrid] = None,
    spacing: float = 0.2,
    padding: float = 4.0,
    title: str = "vibe-qc electron density",
    comment: str = "rho(r) in e/bohr^3",
) -> Path:
    """Write the total electron density r(r) = <D, chi⊗chi> to a Gaussian
    cube file.

    For UHF / UKS pass ``D = D_alpha + D_beta`` (the total density).
    """
    if grid is None:
        grid = make_uniform_grid(mol, spacing=spacing, padding=padding)
    rho = _density_on_grid(np.asarray(D, dtype=float), basis, grid)

    p = Path(path)
    with p.open("w") as out:
        _write_cube_header(out, title=title, comment=comment, mol=mol, grid=grid)
        _write_cube_data(out, rho)
    return p


def write_cube_mo(
    path: Union[str, Path],
    C: np.ndarray,
    index: int,
    basis: BasisSet,
    mol: Molecule,
    *,
    grid: Optional[CubeGrid] = None,
    spacing: float = 0.2,
    padding: float = 4.0,
    title: Optional[str] = None,
) -> Path:
    """Write a single molecular orbital ``phi_index(r)`` to a cube file.

    ``C`` is the full MO coefficient matrix (rows = AOs, columns = MOs);
    ``index`` is zero-based.
    """
    C = np.asarray(C, dtype=float)
    if not (0 <= index < C.shape[1]):
        raise IndexError(f"MO index {index} out of range for C with shape {C.shape}")
    if grid is None:
        grid = make_uniform_grid(mol, spacing=spacing, padding=padding)
    phi = _mo_on_grid(C[:, index], basis, grid)

    p = Path(path)
    with p.open("w") as out:
        _write_cube_header(
            out,
            title=title or f"vibeqc MO {index}",
            comment="phi(r), units: 1/bohr^(3/2)",
            mol=mol,
            grid=grid,
        )
        _write_cube_data(out, phi)
    return p


def write_cube_mos(
    path: Union[str, Path],
    C: np.ndarray,
    indices: Iterable[int],
    basis: BasisSet,
    mol: Molecule,
    *,
    grid: Optional[CubeGrid] = None,
    spacing: float = 0.2,
    padding: float = 4.0,
    title: str = "vibe-qc MOs",
) -> Path:
    """Write a stack of MOs in a single multi-value cube file.

    The output is a "negative N_atoms" cube -- VMD's *Volumetric data*
    selector lets you switch between MOs in one window.
    """
    C = np.asarray(C, dtype=float)
    idx = list(indices)
    if not idx:
        raise ValueError("write_cube_mos: indices is empty")
    for i in idx:
        if not (0 <= i < C.shape[1]):
            raise IndexError(f"MO index {i} out of range for C with shape {C.shape}")

    if grid is None:
        grid = make_uniform_grid(mol, spacing=spacing, padding=padding)

    # Evaluate AOs once per chunk and contract against every requested
    # MO column at once.
    pts = grid.points()
    n_v = len(idx)
    out_data = np.empty((pts.shape[0], n_v), dtype=float)
    chunk = 200_000
    C_sel = C[:, idx]  # (nbf, n_v)
    for i in range(0, pts.shape[0], chunk):
        block = pts[i : i + chunk]
        chi = evaluate_ao(basis, block)
        out_data[i : i + chunk, :] = chi @ C_sel
    # Cube wants voxel-major then n_v: shape (nx, ny, nz, n_v).
    out_data = out_data.reshape((*grid.shape, n_v))

    p = Path(path)
    with p.open("w") as out:
        _write_cube_header(
            out,
            title=title,
            comment="MO stack: phi_i(r) for indices " + ",".join(map(str, idx)),
            mol=mol,
            grid=grid,
            n_values=n_v,
            extra_header_ints=[i + 1 for i in idx],  # 1-based labels
        )
        _write_cube_data(out, out_data)
    return p
