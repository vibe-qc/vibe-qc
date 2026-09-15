"""Minimal POSCAR (VASP 5) reader -> vibeqc.Crystal.

Supports the common case:

    comment line
    scale              (single float; negative means volume in Å^3)
    a_x a_y a_z
    b_x b_y b_z
    c_x c_y c_z
    El1 El2 ...          (element symbols; VASP 5)
    n1  n2  ...          (atom counts per species)
    Direct | Cartesian (or 'D' / 'C' / 'S' prefix for selective dynamics)
    x y z              (x S ni lines, fractional if Direct)

Ignored: selective-dynamics flags, velocity blocks after the coordinates,
VASP 4 files with no element line (species names must be provided there --
reject with a clear error if we detect an all-numeric line where the
species line should be).

Positions are converted to fractional coordinates and lattice vectors are
converted to bohr before being handed to Crystal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ._vibeqc_core import Atom, Crystal, PeriodicSystem
from .periodic_symmetrize import (
    SymmetriseResult,
    symmetrise as _symmetrise_system,
)

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

# Subset of the periodic table sufficient for the crystals we test; extend
# lazily rather than pull in a full table just for POSCAR support.
_Z_BY_SYMBOL = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    # Extend as needed; ValueError fires for anything missing.
}


def _element_to_z(sym: str) -> int:
    try:
        return _Z_BY_SYMBOL[sym]
    except KeyError as err:
        raise ValueError(
            f"POSCAR reader: unknown element symbol {sym!r}. "
            f"Extend _Z_BY_SYMBOL in python/vibeqc/poscar.py."
        ) from err


def read_poscar(
    path: str | Path,
    *,
    symmetrise: bool = False,
    symprec: float = 1.0e-4,
    to_primitive: bool = False,
) -> Crystal | SymmetriseResult:
    """Parse a VASP 5 POSCAR/CONTCAR file into a :class:`Crystal`.

    Lattice vectors in the file are interpreted as Ångström (multiplied by
    the scale factor); the returned Crystal stores them in bohr. Atomic
    positions in a Cartesian block are converted to fractional coordinates
    via the inverse lattice.

    Pass ``symmetrise=True`` to run the parsed 3D structure through
    :func:`vibeqc.symmetrise`; in that mode the return value is a
    :class:`vibeqc.SymmetriseResult` containing a calculation-ready
    :class:`PeriodicSystem` plus the space-group/displacement report.
    """
    p = Path(path)
    lines = p.read_text().splitlines()
    if len(lines) < 8:
        raise ValueError(f"{p}: too short to be a POSCAR file")

    # Line 0: comment (ignored).
    # Line 1: scale factor.
    try:
        scale = float(lines[1].split()[0])
    except (IndexError, ValueError) as err:
        raise ValueError(f"{p}: cannot parse scale factor") from err

    # Lines 2-4: lattice vectors in Å.
    lat_A = np.array(
        [list(map(float, lines[i].split()[:3])) for i in (2, 3, 4)],
        dtype=float,
    )  # Rows = lattice vectors in VASP's file convention.

    if scale > 0:
        lat_A *= scale
    else:
        # Negative scale: interpret |scale| as target volume in Å^3.
        current_vol = abs(np.linalg.det(lat_A))
        lat_A *= (abs(scale) / current_vol) ** (1.0 / 3.0)

    # Line 5: element symbols (VASP 5).
    species_line = lines[5].split()
    if not species_line or species_line[0].replace("-", "").isdigit():
        raise ValueError(
            f"{p}: element line missing (VASP 4 POSCAR not supported). "
            f"Add a species line before the atom counts."
        )

    # Line 6: atom counts.
    try:
        counts = [int(x) for x in lines[6].split()]
    except ValueError as err:
        raise ValueError(f"{p}: cannot parse atom counts") from err
    if len(counts) != len(species_line):
        raise ValueError(
            f"{p}: {len(counts)} counts for {len(species_line)} species"
        )

    # Line 7: coordinate mode (possibly preceded by selective-dynamics line).
    idx = 7
    mode_line = lines[idx].strip()
    if mode_line[:1].upper() == "S":
        # Skip selective-dynamics line; coordinate mode is the next line.
        idx += 1
        mode_line = lines[idx].strip()
    mode_char = mode_line[:1].upper()
    if mode_char not in ("D", "C", "K"):  # K: also treated as Cartesian
        raise ValueError(
            f"{p}: unrecognized coordinate mode {mode_line!r}"
        )
    cartesian = mode_char != "D"

    # Coordinates.
    idx += 1
    n_total = sum(counts)
    coord_lines = lines[idx : idx + n_total]
    if len(coord_lines) < n_total:
        raise ValueError(
            f"{p}: expected {n_total} coordinate lines, found {len(coord_lines)}"
        )
    coords = np.array(
        [list(map(float, ln.split()[:3])) for ln in coord_lines],
        dtype=float,
    )  # (N, 3)

    # Lattice matrix columns-as-vectors, in bohr.
    lattice_bohr = (lat_A.T) * ANGSTROM_TO_BOHR  # (3, 3) columns = a, b, c.

    if cartesian:
        # Cartesian block is in Å (scale already applied to lattice, but
        # VASP also applies the scale to Cartesian coords).
        coords_A = coords if scale > 0 else coords  # same regardless of sign
        if scale > 0:
            coords_A = coords * scale
        coords_bohr = coords_A * ANGSTROM_TO_BOHR
        # frac = L^{-1} . r
        frac = np.linalg.solve(lattice_bohr, coords_bohr.T)  # (3, N)
    else:
        frac = coords.T  # (3, N)

    # Species array: repeat each symbol count[i] times, convert to Z.
    species: list[int] = []
    for sym, n in zip(species_line, counts):
        species.extend([_element_to_z(sym)] * n)

    crystal = Crystal(lattice_bohr, frac, species)
    if not symmetrise:
        return crystal
    return _symmetrise_system(
        _crystal_to_periodic_system(crystal),
        symprec=symprec,
        to_primitive=to_primitive,
    )


def _lattice_cart_from_frac(lattice_bohr: np.ndarray,
                            frac: np.ndarray) -> np.ndarray:
    """Helper: Cartesian coords (N, 3) in bohr from fractional (3, N)."""
    return (lattice_bohr @ frac).T


def _crystal_to_periodic_system(crystal: Crystal) -> PeriodicSystem:
    lattice = np.asarray(crystal.lattice, dtype=float, order="F")
    atoms = [
        Atom(
            int(crystal.species[i]),
            (
                lattice
                @ np.asarray(crystal.fractional_coords[:, i], dtype=float)
            ).tolist(),
        )
        for i in range(crystal.n_atoms)
    ]
    return PeriodicSystem(dim=3, lattice=lattice, unit_cell=atoms)


def write_poscar(path: str | Path,
                 crystal: Crystal,
                 comment: str = "generated by vibe-qc") -> None:
    """Write a Crystal to VASP 5 POSCAR (Direct coordinates, Å lattice).

    Elements are grouped by Z in first-occurrence order so the output
    round-trips :func:`read_poscar`.
    """
    bohr_to_A = 0.529177210903
    lattice_A = np.asarray(crystal.lattice) * bohr_to_A  # columns = vectors.
    frac = np.asarray(crystal.fractional_coords)          # (3, N)
    species = list(crystal.species)

    # Group atoms by Z, preserving first-occurrence order.
    seen_order: list[int] = []
    for z in species:
        if z not in seen_order:
            seen_order.append(z)
    inv_Z = {z: sym for sym, z in _Z_BY_SYMBOL.items()}
    syms = [inv_Z[z] for z in seen_order]
    counts = [species.count(z) for z in seen_order]

    # Reorder columns to match species grouping.
    order = [i for z in seen_order for i, zz in enumerate(species) if zz == z]
    frac_sorted = frac[:, order]

    lines = [comment, "1.0"]
    for j in range(3):  # rows of file = lattice vectors = cols of lattice_A
        v = lattice_A[:, j]
        lines.append(f"  {v[0]:.16f}  {v[1]:.16f}  {v[2]:.16f}")
    lines.append("  " + "  ".join(syms))
    lines.append("  " + "  ".join(str(c) for c in counts))
    lines.append("Direct")
    for k in range(frac_sorted.shape[1]):
        r = frac_sorted[:, k]
        lines.append(f"  {r[0]:.16f}  {r[1]:.16f}  {r[2]:.16f}")
    Path(path).write_text("\n".join(lines) + "\n")
