"""Import ORCA ``%basis`` block format into the canonical model.

ORCA's ``%basis`` directive uses ``NewGTO`` / ``end`` blocks to define
per-element basis sets.  This importer parses that syntax and converts it
into :class:`BasisSetData`.
"""

from __future__ import annotations

import re
from pathlib import Path

from ._element_data import VALID_SYMBOLS
from .model import (
    BasisRole,
    BasisSetData,
    ElementBasis,
    HarmonicType,
    Primitive,
    Provenance,
    Shell,
)

__all__ = ["from_orca", "from_orca_file"]

_RE_NEWGTO = re.compile(r"^\s*NewGTO\s+([A-Z][a-z]?)\s*$", re.IGNORECASE)
_RE_SHELL_LINE = re.compile(r"^\s*([S|P|D|F|G|H|I])\s+(\d+)\s*$")
_RE_END = re.compile(r"^\s*end\s*$", re.IGNORECASE)

_AM_LABEL: dict[str, list[int]] = {
    "S": [0],
    "P": [1],
    "D": [2],
    "F": [3],
    "G": [4],
    "H": [5],
    "I": [6],
}


def from_orca(
    text: str,
    *,
    name: str | None = None,
    role: str = "orbital",
    family: str | None = None,
    harmonic_type: str = "spherical",
) -> BasisSetData:
    """Parse an ORCA ``%basis`` block string into :class:`BasisSetData`.

    Parameters
    ----------
    text:
        ORCA ``%basis ... end`` block text (the ``%basis`` and final
        ``end`` lines are optional -- parsing starts at the first
        ``NewGTO``).
    name:
        Override basis name.  Defaults to ``"<orca-import>"``.
    role:
        Basis role (orbital, auxiliary, fitting, ecp).
    family:
        Basis family hint.
    harmonic_type:
        Default harmonic convention.

    Returns
    -------
    BasisSetData
    """
    lines = text.splitlines()
    elements: dict[str, ElementBasis] = {}
    current_symbol: str | None = None
    current_shells: list[Shell] = []
    current_label: str | None = None
    current_n_prim: int = 0
    current_prims_read: int = 0
    current_primitives: list[Primitive] = []
    in_basis = True  # tolerant: if no %basis wrapper, parse everything
    saw_basis_keyword = False

    for raw in lines:
        line = raw.strip()
        # Skip comments and empty lines.
        if not line or line.startswith("#") or line.startswith("!"):
            continue

        # Enter basis block.
        if line.lower().startswith("%basis"):
            saw_basis_keyword = True
            in_basis = True
            continue

        # End line: closes current element.  If we saw %basis, the
        # *first* end closes the element and the *second* end closes
        # the basis block.
        # End line: closes current element.
        if _RE_END.match(line):
            if saw_basis_keyword and not in_basis:
                continue
            # Flush pending shell before finalising element.
            if current_label is not None and current_primitives:
                am = _AM_LABEL.get(current_label)
                if am:
                    current_shells.append(
                        Shell(
                            angular_momentum=am,
                            harmonic_type=HarmonicType(harmonic_type),
                            primitives=current_primitives,
                        )
                    )
                current_label = None
                current_primitives = []
            # Finalise current element.
            if current_symbol is not None and current_shells:
                elements[current_symbol] = ElementBasis(
                    element=current_symbol, shells=current_shells
                )
            current_symbol = None
            current_shells = []
            continue

        # New element.
        m_new = _RE_NEWGTO.match(line)
        if m_new:
            # Flush pending shell of previous element.
            if current_label is not None and current_primitives:
                am = _AM_LABEL.get(current_label)
                if am:
                    current_shells.append(
                        Shell(
                            angular_momentum=am,
                            harmonic_type=HarmonicType(harmonic_type),
                            primitives=current_primitives,
                        )
                    )
                current_label = None
                current_primitives = []
            # Finalise previous element.
            if current_symbol is not None and current_shells:
                elements[current_symbol] = ElementBasis(
                    element=current_symbol, shells=current_shells
                )
            current_symbol = m_new.group(1)
            if current_symbol not in VALID_SYMBOLS:
                raise ValueError(
                    f"Unknown element symbol {current_symbol!r} in ORCA NewGTO"
                )
            current_shells = []
            current_label = None
            current_primitives = []
            current_prims_read = 0
            continue

        # Shell header.
        m_shell = _RE_SHELL_LINE.match(line)
        if m_shell:
            # Finalise previous shell if any.
            if current_label is not None and current_primitives:
                am = _AM_LABEL.get(current_label)
                if am is None:
                    raise ValueError(f"Unknown shell label {current_label!r}")
                current_shells.append(
                    Shell(
                        angular_momentum=am,
                        harmonic_type=HarmonicType(harmonic_type),
                        primitives=current_primitives,
                    )
                )
            current_label = m_shell.group(1).upper()
            current_n_prim = int(m_shell.group(2))
            current_primitives = []
            current_prims_read = 0
            continue

        # Primitive line:  "0  13.01  0.019685"
        parts = line.split()
        if len(parts) >= 3:
            try:
                # parts[0] is the index (skip), parts[1] is exponent, parts[2] is coefficient
                exponent = float(parts[1])
                coeff = float(parts[2])
            except (ValueError, IndexError):
                continue
            current_primitives.append(Primitive(exponent=exponent, coefficient=coeff))
            current_prims_read += 1

    # Finalise last element.
    if current_label is not None and current_primitives:
        am = _AM_LABEL.get(current_label)
        if am is not None:
            current_shells.append(
                Shell(
                    angular_momentum=am,
                    harmonic_type=HarmonicType(harmonic_type),
                    primitives=current_primitives,
                )
            )
    if current_symbol is not None and current_shells:
        elements[current_symbol] = ElementBasis(
            element=current_symbol, shells=current_shells
        )

    if not elements:
        raise ValueError("No element blocks found in ORCA basis input")

    return BasisSetData(
        schema_version="1.0.0",
        name=name or "<orca-import>",
        description=f"Imported from ORCA: {name or '<orca-import>'}",
        role=BasisRole(role),
        basis_family=family,
        elements=elements,
        provenance=Provenance(origin="file"),
    )


def from_orca_file(
    path: str | Path,
    *,
    name: str | None = None,
    role: str = "orbital",
    family: str | None = None,
    harmonic_type: str = "spherical",
) -> BasisSetData:
    """Read an ORCA basis file and parse it."""
    text = Path(path).read_text(encoding="utf-8")
    return from_orca(
        text, name=name, role=role, family=family, harmonic_type=harmonic_type
    )
