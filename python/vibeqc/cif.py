"""Minimal CIF reader for periodic structures.

The reader targets the crystallographic data users commonly download
from materials databases: cell lengths/angles, an atom-site loop with
fractional coordinates, and optional CIF-style symmetry operations such
as ``x, y, z`` / ``-x+1/2, y+1/2, z``. Coordinates are expanded to the
full unit cell before constructing :class:`vibeqc.Crystal`.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import math
import re
import shlex
from typing import NamedTuple

import numpy as np

from ._vibeqc_core import Crystal
from .molecule import _atomic_number
from .periodic_symmetrize import (
    SymmetriseResult,
    symmetrise as _symmetrise_system,
)
from .poscar import ANGSTROM_TO_BOHR, _crystal_to_periodic_system

_VALUE_UNCERTAINTY_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)(?:\(\d+\))?$"
)
_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")


class _CifLoop(NamedTuple):
    headers: list[str]
    rows: list[dict[str, str]]


def read_cif(
    path: str | Path,
    *,
    symmetrise: bool = False,
    symprec: float = 1.0e-4,
    to_primitive: bool = False,
) -> Crystal | SymmetriseResult:
    """Parse a CIF file into a :class:`Crystal`.

    The returned crystal stores lattice vectors in bohr as columns and
    fractional coordinates as one atom per column. When ``symmetrise``
    is true, the parsed crystal is converted to a 3D
    :class:`PeriodicSystem`, passed through :func:`vibeqc.symmetrise`,
    and returned as a :class:`SymmetriseResult`.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    tags, loops = _parse_cif_blocks(text)
    lattice_bohr = _lattice_from_cif_tags(tags, p)
    species, frac = _atom_sites_from_cif(loops, p)
    rotations = _symmetry_ops_from_cif(loops)
    species, frac = _expand_sites_by_symmetry(species, frac, rotations)

    crystal = Crystal(lattice_bohr, frac, species)
    if not symmetrise:
        return crystal
    return _symmetrise_system(
        _crystal_to_periodic_system(crystal),
        symprec=symprec,
        to_primitive=to_primitive,
    )


def _parse_cif_blocks(text: str) -> tuple[dict[str, str], list[_CifLoop]]:
    tags: dict[str, str] = {}
    loops: list[_CifLoop] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        lower = raw.lower()
        if not raw or raw.startswith("#") or lower.startswith("data_"):
            i += 1
            continue
        if lower == "loop_":
            loop, i = _parse_loop(lines, i + 1)
            if loop.headers:
                loops.append(loop)
            continue
        if raw.startswith("_"):
            tokens = _split_cif_line(raw)
            if len(tokens) >= 2:
                tags[tokens[0].lower()] = tokens[1]
            i += 1
            continue
        if raw.startswith(";"):
            raise ValueError("CIF reader: semicolon multiline values are not supported")
        i += 1
    return tags, loops


def _parse_loop(lines: list[str], start: int) -> tuple[_CifLoop, int]:
    headers: list[str] = []
    i = start
    while i < len(lines):
        raw = lines[i].strip()
        if not raw or raw.startswith("#"):
            i += 1
            continue
        if raw.startswith("_"):
            header = _split_cif_line(raw)[0].lower()
            headers.append(header)
            i += 1
            continue
        break

    values: list[str] = []
    while i < len(lines):
        raw = lines[i].strip()
        lower = raw.lower()
        if not raw or raw.startswith("#"):
            i += 1
            continue
        if lower == "loop_" or lower.startswith("data_") or raw.startswith("_"):
            break
        if raw.startswith(";"):
            raise ValueError("CIF reader: semicolon multiline values are not supported")
        values.extend(_split_cif_line(raw))
        i += 1

    if not headers:
        return _CifLoop([], []), i
    if len(values) % len(headers) != 0:
        raise ValueError(
            "CIF reader: loop value count is not a multiple of the header count"
        )
    rows: list[dict[str, str]] = []
    for offset in range(0, len(values), len(headers)):
        rows.append(dict(zip(headers, values[offset : offset + len(headers)])))
    return _CifLoop(headers, rows), i


def _split_cif_line(line: str) -> list[str]:
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    return list(lexer)


def _lattice_from_cif_tags(tags: dict[str, str], path: Path) -> np.ndarray:
    required = {
        "_cell_length_a": None,
        "_cell_length_b": None,
        "_cell_length_c": None,
        "_cell_angle_alpha": None,
        "_cell_angle_beta": None,
        "_cell_angle_gamma": None,
    }
    missing = [key for key in required if key not in tags]
    if missing:
        raise ValueError(f"{path}: missing CIF cell tags: {', '.join(missing)}")

    a = _parse_cif_float(tags["_cell_length_a"])
    b = _parse_cif_float(tags["_cell_length_b"])
    c = _parse_cif_float(tags["_cell_length_c"])
    alpha = math.radians(_parse_cif_float(tags["_cell_angle_alpha"]))
    beta = math.radians(_parse_cif_float(tags["_cell_angle_beta"]))
    gamma = math.radians(_parse_cif_float(tags["_cell_angle_gamma"]))

    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1.0e-12:
        raise ValueError(f"{path}: invalid CIF cell angle gamma={math.degrees(gamma)}")

    a_vec = np.array([a, 0.0, 0.0], dtype=float)
    b_vec = np.array([b * math.cos(gamma), b * sin_gamma, 0.0], dtype=float)
    c_x = c * math.cos(beta)
    c_y = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    c_z_sq = c * c - c_x * c_x - c_y * c_y
    if c_z_sq < -1.0e-10:
        raise ValueError(f"{path}: CIF cell parameters do not form a real lattice")
    c_vec = np.array([c_x, c_y, math.sqrt(max(0.0, c_z_sq))], dtype=float)
    return np.column_stack([a_vec, b_vec, c_vec]) * ANGSTROM_TO_BOHR


def _atom_sites_from_cif(
    loops: list[_CifLoop],
    path: Path,
) -> tuple[list[int], np.ndarray]:
    loop = _find_loop(loops, "_atom_site_fract_x")
    if loop is None:
        raise ValueError(f"{path}: no _atom_site fractional-coordinate loop found")

    species: list[int] = []
    frac_cols: list[list[float]] = []
    for row in loop.rows:
        symbol_value = row.get("_atom_site_type_symbol") or row.get(
            "_atom_site_label"
        )
        if symbol_value is None:
            raise ValueError(f"{path}: atom-site loop lacks type symbol / label")
        try:
            fx = _parse_cif_float(row["_atom_site_fract_x"])
            fy = _parse_cif_float(row["_atom_site_fract_y"])
            fz = _parse_cif_float(row["_atom_site_fract_z"])
        except KeyError as exc:
            raise ValueError(
                f"{path}: atom-site loop lacks fractional coordinates"
            ) from exc
        species.append(_atomic_number(_normalise_element_symbol(symbol_value)))
        frac_cols.append([fx, fy, fz])

    if not species:
        raise ValueError(f"{path}: atom-site loop is empty")
    return species, np.asarray(frac_cols, dtype=float).T


def _symmetry_ops_from_cif(
    loops: list[_CifLoop],
) -> list[tuple[np.ndarray, np.ndarray]]:
    for tag in (
        "_space_group_symop_operation_xyz",
        "_symmetry_equiv_pos_as_xyz",
    ):
        loop = _find_loop(loops, tag)
        if loop is not None:
            return [_parse_symop(row[tag]) for row in loop.rows]
    return [(np.eye(3, dtype=int), np.zeros(3, dtype=float))]


def _find_loop(loops: list[_CifLoop], required_header: str) -> _CifLoop | None:
    required_header = required_header.lower()
    for loop in loops:
        if required_header in loop.headers:
            return loop
    return None


def _expand_sites_by_symmetry(
    species: list[int],
    frac: np.ndarray,
    ops: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[int], np.ndarray]:
    expanded_species: list[int] = []
    expanded_frac: list[np.ndarray] = []
    for site_idx, z in enumerate(species):
        coord = np.asarray(frac[:, site_idx], dtype=float)
        for rotation, translation in ops:
            candidate = (rotation @ coord + translation) % 1.0
            if not _has_fractional_site(
                expanded_species, expanded_frac, z, candidate
            ):
                expanded_species.append(z)
                expanded_frac.append(candidate)
    if not expanded_frac:
        return [], np.empty((3, 0), dtype=float)
    return expanded_species, np.asarray(expanded_frac, dtype=float).T


def _has_fractional_site(
    species: list[int],
    coords: list[np.ndarray],
    z: int,
    candidate: np.ndarray,
    tol: float = 1.0e-6,
) -> bool:
    for existing_z, existing in zip(species, coords):
        if existing_z != z:
            continue
        delta = candidate - existing
        delta -= np.round(delta)
        if float(np.linalg.norm(delta)) < tol:
            return True
    return False


def _parse_symop(raw: str) -> tuple[np.ndarray, np.ndarray]:
    parts = [part.strip().lower().replace(" ", "") for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"CIF reader: symmetry operation must have 3 axes, got {raw!r}"
        )
    rotation = np.zeros((3, 3), dtype=int)
    translation = np.zeros(3, dtype=float)
    for axis, expr in enumerate(parts):
        coeffs, offset = _parse_symop_expr(expr)
        rotation[axis, :] = coeffs
        translation[axis] = offset
    return rotation, translation


def _parse_symop_expr(expr: str) -> tuple[np.ndarray, float]:
    coeffs = np.zeros(3, dtype=int)
    offset = Fraction(0, 1)
    normalised = expr.replace("-", "+-")
    if normalised.startswith("+"):
        normalised = normalised[1:]
    for term in normalised.split("+"):
        if not term:
            continue
        if term in {"x", "y", "z", "-x", "-y", "-z"}:
            sign = -1 if term.startswith("-") else 1
            var = term[-1]
            coeffs[{"x": 0, "y": 1, "z": 2}[var]] += sign
        elif term.endswith(("x", "y", "z")):
            sign = -1 if term.startswith("-") else 1
            magnitude = term[:-1].lstrip("+-")
            if magnitude not in {"", "1"}:
                raise ValueError(
                    f"CIF reader: unsupported symmetry coefficient in {expr!r}"
                )
            coeffs[{"x": 0, "y": 1, "z": 2}[term[-1]]] += sign
        else:
            offset += Fraction(term)
    return coeffs, float(offset)


def _normalise_element_symbol(value: str) -> str:
    match = _ELEMENT_RE.search(value)
    if match is None:
        raise ValueError(f"CIF reader: cannot infer element symbol from {value!r}")
    return match.group(1)


def _parse_cif_float(value: str) -> float:
    if value in {".", "?"}:
        raise ValueError("CIF reader: missing numeric value")
    match = _VALUE_UNCERTAINTY_RE.match(value)
    if match is not None:
        return float(match.group(1))
    return float(value)


__all__ = ["read_cif"]
