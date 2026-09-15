"""XSF / BXSF writers for periodic structure and volumetric data.

XCrySDen's XSF format is the de-facto interchange format for periodic
crystal structures and volumetric data -- VESTA, XCrySDen and most
solid-state visualization tools read it directly.

What this module exposes
------------------------

* :func:`write_xsf_structure` -- just the crystal (lattice + atoms).
  A ``CRYSTAL`` / ``SLAB`` / ``POLYMER`` block (per ``system.dim``),
  ``PRIMVEC``, ``PRIMCOORD``. Lattice vectors and atom positions are
  converted from bohr -> ångström because XSF wants ångström.
* :func:`write_xsf_volume` -- crystal + a 3D grid of scalar values
  (electron density, an MO, a potential, anything you've evaluated on
  a uniform grid). Grid is described by ``DATAGRID_3D`` blocks; one or
  several stacked grids per file are supported.
* :func:`write_bxsf` -- *band* XSF: Fermi-surface-style band data on a
  Monkhorst-Pack k-mesh in the BZ (one rectangular block per band).
  XCrySDen reads this directly to render Fermi surfaces.

Convention reminders:

* All atomic coordinates and lattice vectors are written in **ångström**
  (XSF spec); we convert from bohr internally.
* The opening block keyword carries the *dimensionality*. For ``dim < 3``
  the lattice columns ``dim..2`` are non-physical bookkeeping (see
  ``cpp/include/vibeqc/periodic.hpp``), and a reader cannot recover that
  from the lattice matrix alone -- a synthesized ``a3`` is numerically
  indistinguishable from a real vacuum gap. The keyword is the only
  signal, so it must be written truthfully.
* DATAGRID_3D values are written in *general position*: the spec wants
  the volume covered by ``span_a``, ``span_b``, ``span_c`` from a
  given ``origin`` and the data run with the *first* grid index
  varying fastest.
* BXSF energies are conventionally given in **eV**.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from ._vibeqc_core import BasisSet, PeriodicSystem, evaluate_ao
from .bands import BandStructure


__all__ = [
    "write_xsf_structure",
    "write_xsf_volume",
    "write_bxsf",
]


_BOHR_TO_ANGSTROM = 0.529177210903
_HARTREE_TO_EV = 27.211386245988


# ---------------------------------------------------------------------------
# Crystal-only XSF
# ---------------------------------------------------------------------------

def _atoms_block(system: PeriodicSystem) -> str:
    out = []
    for a in system.unit_cell:
        x, y, z = (np.asarray(a.xyz) * _BOHR_TO_ANGSTROM).tolist()
        out.append(f"{a.Z:3d} {x:14.8f} {y:14.8f} {z:14.8f}")
    return "\n".join(out)


def _lattice_block(system: PeriodicSystem) -> str:
    L = np.asarray(system.lattice) * _BOHR_TO_ANGSTROM
    return "\n".join(
        f"{L[0, i]:14.8f} {L[1, i]:14.8f} {L[2, i]:14.8f}" for i in range(3)
    )


# XCrySDen's XSF spec opens a periodic structure with a keyword naming how
# many of the three PRIMVEC rows are physical lattice vectors: POLYMER is
# periodic along x, SLAB in the xy-plane, CRYSTAL in all three. That matches
# vibe-qc's convention of the first `dim` lattice columns being the periodic
# ones, which `_lattice_block` writes as the first `dim` PRIMVEC rows.
_XSF_DIMENSIONALITY_KEYWORD = {1: "POLYMER", 2: "SLAB", 3: "CRYSTAL"}


def _dimensionality_keyword(system: PeriodicSystem) -> str:
    dim = int(system.dim)
    try:
        return _XSF_DIMENSIONALITY_KEYWORD[dim]
    except KeyError:
        raise ValueError(
            f"cannot write XSF for a system with dim={dim}; "
            f"expected one of {sorted(_XSF_DIMENSIONALITY_KEYWORD)}"
        ) from None


def write_xsf_structure(
    path: Union[str, Path],
    system: PeriodicSystem,
) -> Path:
    """Write a *structure-only* XSF file.

    The opening block is ``CRYSTAL``, ``SLAB`` or ``POLYMER`` according to
    ``system.dim``, so a reader knows which PRIMVEC rows are real lattice
    vectors and which are non-physical padding.

    The result opens directly in VESTA / XCrySDen as the periodic
    structure -- useful as a quick visual sanity check on a
    :class:`PeriodicSystem` before running anything expensive.
    """
    p = Path(path)
    n_at = len(system.unit_cell)
    with p.open("w") as out:
        out.write(_dimensionality_keyword(system) + "\n")
        out.write("PRIMVEC\n")
        out.write(_lattice_block(system) + "\n")
        out.write("PRIMCOORD\n")
        out.write(f"{n_at:5d} 1\n")
        out.write(_atoms_block(system) + "\n")
    return p


# ---------------------------------------------------------------------------
# XSF with one or more volumetric grids
# ---------------------------------------------------------------------------

def _datagrid_block(
    name: str,
    data: np.ndarray,           # shape (n1, n2, n3)
    origin: np.ndarray,         # (3,) ångström
    span_a: np.ndarray,         # (3,) ångström
    span_b: np.ndarray,         # (3,) ångström
    span_c: np.ndarray,         # (3,) ångström
) -> str:
    """One DATAGRID_3D_<name> block. XSF wants 6 values per line and the
    *first* grid index (``i``) running fastest, then ``j``, then ``k``."""
    n1, n2, n3 = data.shape
    lines = []
    lines.append(f"BEGIN_DATAGRID_3D_{name}")
    lines.append(f"  {n1:5d} {n2:5d} {n3:5d}")
    lines.append(f"  {origin[0]:14.8f} {origin[1]:14.8f} {origin[2]:14.8f}")
    lines.append(f"  {span_a[0]:14.8f} {span_a[1]:14.8f} {span_a[2]:14.8f}")
    lines.append(f"  {span_b[0]:14.8f} {span_b[1]:14.8f} {span_b[2]:14.8f}")
    lines.append(f"  {span_c[0]:14.8f} {span_c[1]:14.8f} {span_c[2]:14.8f}")
    # Reorder to ascending k -> j -> i, with i fastest. np.transpose to
    # (k, j, i) then ravel gives that traversal.
    flat = np.transpose(data, (2, 1, 0)).ravel()
    for i in range(0, flat.size, 6):
        chunk = flat[i:i + 6]
        lines.append(" ".join(f"{x:13.5e}" for x in chunk))
    lines.append(f"END_DATAGRID_3D_{name}")
    return "\n".join(lines) + "\n"


def write_xsf_volume(
    path: Union[str, Path],
    system: PeriodicSystem,
    *,
    data: np.ndarray,
    name: str = "scalar",
    origin: Optional[np.ndarray] = None,
    span: Optional[np.ndarray] = None,
) -> Path:
    """Write a periodic XSF file with a single 3D scalar grid.

    Parameters
    ----------
    system
        Crystal structure.
    data
        Scalar values on a 3D grid, shape ``(n1, n2, n3)``. By default
        the grid is assumed to span exactly one unit cell with origin
        at the lattice origin (the most common case for densities).
    name
        Tag used in the ``BEGIN_DATAGRID_3D_<name>`` block; pick
        something descriptive like ``"density"`` or ``"mo_homo"``.
    origin
        Bohr coordinates of the (0, 0, 0) voxel. Defaults to (0, 0, 0).
    span
        ``(3, 3)`` matrix; rows are the three spanning vectors of the
        grid in bohr. Defaults to the system lattice vectors.

    Notes
    -----
    XSF expects the grid to be a *fully-periodic* sample -- the value at
    voxel (n1-1, j, k) repeats voxel (0, j, k). When you build the grid
    with :meth:`vibeqc.LatticeCell` etc., remember to *not* include the
    duplicate boundary point.
    """
    if data.ndim != 3:
        raise ValueError(f"data must be 3D, got shape {data.shape}")

    L_bohr = np.asarray(system.lattice)
    if origin is None:
        origin = np.zeros(3)
    if span is None:
        span = L_bohr.T

    origin_a = np.asarray(origin) * _BOHR_TO_ANGSTROM
    span_a = np.asarray(span) * _BOHR_TO_ANGSTROM

    p = Path(path)
    with p.open("w") as out:
        out.write(_dimensionality_keyword(system) + "\n")
        out.write("PRIMVEC\n")
        out.write(_lattice_block(system) + "\n")
        out.write("PRIMCOORD\n")
        out.write(f"{len(system.unit_cell):5d} 1\n")
        out.write(_atoms_block(system) + "\n")
        out.write("\n")
        out.write("BEGIN_BLOCK_DATAGRID_3D\n")
        out.write(f"  {name}\n")
        out.write(_datagrid_block(
            name, data, origin_a, span_a[0], span_a[1], span_a[2],
        ))
        out.write("END_BLOCK_DATAGRID_3D\n")
    return p


# ---------------------------------------------------------------------------
# BXSF -- band data on a regular k-mesh
# ---------------------------------------------------------------------------

def write_bxsf(
    path: Union[str, Path],
    system: PeriodicSystem,
    energies: np.ndarray,            # (n_kx, n_ky, n_kz, n_bands), Hartree
    *,
    e_fermi: float = 0.0,            # Hartree
) -> Path:
    """Write a BXSF file (band XSF) for Fermi-surface visualization in
    XCrySDen.

    ``energies`` is a 4D array: a regular Monkhorst-Pack-style mesh
    sampled in *fractional* reciprocal coordinates over ``[0, 1)`` along
    each axis, with one rectangular block per band. Values in Hartree;
    the file is written in eV (the conventional BXSF unit).
    """
    if energies.ndim != 4:
        raise ValueError(
            f"energies must have shape (nkx, nky, nkz, nbands), got {energies.shape}"
        )
    nkx, nky, nkz, nbands = energies.shape
    e_eV = energies * _HARTREE_TO_EV
    ef_eV = e_fermi * _HARTREE_TO_EV

    # Reciprocal lattice in 1/ångström (BXSF's expected unit).
    L_bohr = np.asarray(system.lattice)
    L_ang = L_bohr * _BOHR_TO_ANGSTROM
    B_ang = 2.0 * np.pi * np.linalg.inv(L_ang).T  # columns are b_i (1/Å)

    p = Path(path)
    with p.open("w") as out:
        out.write("BEGIN_INFO\n")
        out.write(f"  Fermi Energy: {ef_eV:.6f}\n")
        out.write("END_INFO\n\n")
        out.write("BEGIN_BLOCK_BANDGRID_3D\n")
        out.write("  band_energies\n")
        out.write("  BANDGRID_3D_BANDS\n")
        out.write(f"  {nbands:5d}\n")
        out.write(f"  {nkx:5d} {nky:5d} {nkz:5d}\n")
        # Origin is (0, 0, 0); spanning vectors are b_1, b_2, b_3.
        out.write("  0.0 0.0 0.0\n")
        for i in range(3):
            out.write(f"  {B_ang[0, i]:14.8f} {B_ang[1, i]:14.8f} {B_ang[2, i]:14.8f}\n")

        for ib in range(nbands):
            out.write(f"  BAND:  {ib + 1}\n")
            # XCrySDen wants i fastest; transpose to (k, j, i) then flatten.
            block = np.transpose(e_eV[..., ib], (2, 1, 0)).ravel()
            for i in range(0, block.size, 6):
                chunk = block[i:i + 6]
                out.write("  " + " ".join(f"{x:13.5e}" for x in chunk) + "\n")

        out.write("  END_BANDGRID_3D\n")
        out.write("END_BLOCK_BANDGRID_3D\n")
    return p
