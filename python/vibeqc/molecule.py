"""Pure-Python convenience layer around the native Molecule/Atom types.

The core ``Atom`` and ``Molecule`` classes live in the C++ extension; this
module just adds input-format parsers.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

from ._vibeqc_core import Atom, Molecule, PeriodicSystem

if TYPE_CHECKING:
    from .periodic_symmetrize import SymmetriseResult

# Bohr = Ångström x (1 / a0). CODATA 2018: a0 = 0.529177210903 Å.
ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

# Elements 1-36; extend as needed. Symbols are case-normalized on lookup.
_ATOMIC_NUMBERS = {
    "H":  1, "He": 2,
    "Li": 3, "Be": 4, "B":  5, "C":  6, "N":  7, "O":  8, "F":  9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P":  15, "S":  16, "Cl": 17, "Ar": 18,
    "K":  19, "Ca": 20, "Sc": 21, "Ti": 22, "V":  23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
}


def _atomic_number(symbol: str) -> int:
    key = symbol.strip().capitalize()
    try:
        return _ATOMIC_NUMBERS[key]
    except KeyError as exc:
        raise ValueError(f"unknown element symbol: {symbol!r}") from exc


def from_xyz(
    path: Union[str, Path],
    *,
    charge: int = 0,
    multiplicity: int = 1,
    periodic: bool = False,
    lattice: Union[None, np.ndarray, list[list[float]]] = None,
    dim: int | None = None,
    symmetrise: bool = False,
    symprec: float = 1.0e-4,
    to_primitive: bool = False,
) -> Molecule | PeriodicSystem | SymmetriseResult:
    """Parse an XYZ file (positions in Ångström).

    Parameters
    ----------
    path:
        Path to an ``.xyz`` file. First line: atom count. Second
        line: comment. Following lines: ``<symbol> <x> <y> <z>``.
    charge:
        Net charge of the molecule (electron count = S Z - charge).
    multiplicity:
        Spin multiplicity 2S+1. Must be >= 1 and have the right parity
        against the electron count (validated by the native Molecule).
    periodic:
        When ``True``, return a :class:`PeriodicSystem`. The lattice
        is taken from an Extended-XYZ ``Lattice="..."`` comment tag
        unless ``lattice=`` is supplied explicitly.
    lattice:
        Optional 3x3 lattice matrix in Ångström, with rows as lattice
        vectors (ASE / Extended-XYZ convention). Supplying this also
        selects periodic mode.
    dim:
        Periodic dimensionality for the returned system. Defaults to
        the prefix length implied by ``pbc="..."`` when present, else
        3. Only prefix PBC masks (``T F F``, ``T T F``, ``T T T``)
        are representable by :class:`PeriodicSystem`.
    symmetrise:
        Standardise the parsed periodic structure via
        :func:`vibeqc.symmetrise` and return ``SymmetriseResult``.
        Implies periodic mode.
    symprec, to_primitive:
        Passed through to :func:`vibeqc.symmetrise` when
        ``symmetrise=True``.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    if len(lines) < 2:
        raise ValueError(f"XYZ file {path} is too short")

    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(
            f"XYZ file {path} first line must be atom count, got {lines[0]!r}"
        ) from exc

    comment = lines[1] if len(lines) > 1 else ""
    tags = _extended_xyz_tags(comment)

    body = [ln for ln in lines[2 : 2 + n_atoms] if ln.strip()]
    if len(body) != n_atoms:
        raise ValueError(
            f"XYZ file {path}: expected {n_atoms} atom lines, got {len(body)}"
        )

    atoms: list[Atom] = []
    for idx, line in enumerate(body, start=1):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(
                f"XYZ file {path} line {idx}: need '<sym> <x> <y> <z>', got {line!r}"
            )
        Z = _atomic_number(parts[0])
        xyz = tuple(float(p) * ANGSTROM_TO_BOHR for p in parts[1:4])
        atoms.append(Atom(Z, list(xyz)))

    wants_periodic = periodic or symmetrise or lattice is not None
    if not wants_periodic:
        return Molecule(atoms, charge, multiplicity)

    lattice_bohr = _resolve_xyz_lattice_bohr(path, lattice, tags)
    dim_value = _resolve_periodic_dim(dim, tags)
    system = PeriodicSystem(
        dim=dim_value,
        lattice=np.asarray(lattice_bohr, dtype=float, order="F"),
        unit_cell=atoms,
        charge=charge,
        multiplicity=multiplicity,
    )
    if not symmetrise:
        return system

    from .periodic_symmetrize import symmetrise as _symmetrise

    return _symmetrise(
        system,
        symprec=symprec,
        to_primitive=to_primitive,
    )


def _extended_xyz_tags(comment: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    try:
        tokens = shlex.split(comment)
    except ValueError:
        tokens = comment.split()
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        tags[key] = value
    return tags


def _resolve_xyz_lattice_bohr(
    path: Path,
    lattice: Union[None, np.ndarray, list[list[float]]],
    tags: dict[str, str],
) -> np.ndarray:
    if lattice is not None:
        lat_rows = np.asarray(lattice, dtype=float)
    else:
        raw = tags.get("Lattice")
        if raw is None:
            raise ValueError(
                f"XYZ file {path}: periodic=True / symmetrise=True "
                "requires an Extended-XYZ Lattice=\"...\" tag or "
                "an explicit lattice= matrix"
            )
        values = [float(x) for x in raw.split()]
        if len(values) != 9:
            raise ValueError(
                f"XYZ file {path}: Lattice tag must contain 9 numbers, "
                f"got {len(values)}"
            )
        lat_rows = np.asarray(values, dtype=float).reshape(3, 3)
    if lat_rows.shape != (3, 3):
        raise ValueError(
            f"XYZ file {path}: lattice must be a 3x3 matrix in Angstrom, "
            f"got shape {lat_rows.shape}"
        )
    # Extended XYZ stores rows as lattice vectors in Å. vibe-qc stores
    # columns as lattice vectors in bohr.
    return lat_rows.T * ANGSTROM_TO_BOHR


def _resolve_periodic_dim(dim: int | None, tags: dict[str, str]) -> int:
    if dim is not None:
        dim_value = int(dim)
    else:
        pbc = tags.get("pbc")
        dim_value = 3 if pbc is None else _dim_from_pbc_tag(pbc)
    if dim_value < 1 or dim_value > 3:
        raise ValueError(f"PeriodicSystem dim must be 1, 2, or 3; got {dim_value}")
    return dim_value


def _dim_from_pbc_tag(raw: str) -> int:
    flags = raw.replace(",", " ").split()
    if len(flags) != 3:
        raise ValueError(f"Extended-XYZ pbc tag must have three flags, got {raw!r}")
    truth = [flag.strip().upper() in {"T", "TRUE", "1", "Y", "YES"} for flag in flags]
    if truth == [True, False, False]:
        return 1
    if truth == [True, True, False]:
        return 2
    if truth == [True, True, True]:
        return 3
    raise ValueError(
        "Extended-XYZ pbc tag must be prefix-periodic for vibe-qc "
        f"(T F F, T T F, or T T T); got {raw!r}"
    )
