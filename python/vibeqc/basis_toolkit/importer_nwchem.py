"""Import NWChem ``basis`` block format into the canonical model.

NWChem's basis directive uses element-tagged shell blocks::

    basis "name" spherical
      H    S
        <exponent>    <coefficient>
      O    S
        ...
    end
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

__all__ = ["from_nwchem", "from_nwchem_file"]

_RE_BASIS_HEADER = re.compile(
    r'^\s*basis\s+(?:"([^"]*)"|(\S+))\s*(spherical|cartesian)?', re.IGNORECASE
)
_RE_ELEM_SHELL = re.compile(r"^\s*([A-Z][a-z]?)\s+([S|P|D|F|G|H|I])\s*$")
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


def from_nwchem(
    text: str,
    *,
    name: str | None = None,
    role: str = "orbital",
    family: str | None = None,
) -> BasisSetData:
    """Parse an NWChem basis block string into :class:`BasisSetData`.

    Parameters
    ----------
    text:
        NWChem ``basis ... end`` block text.
    name:
        Override basis name.  Extracted from the header if not provided.
    role:
        Basis role.
    family:
        Basis family hint.

    Returns
    -------
    BasisSetData
    """
    lines = text.splitlines()
    elements: dict[str, ElementBasis] = {}
    current_symbol: str | None = None
    current_label: str | None = None
    current_primitives: list[Primitive] = []
    current_shells: list[Shell] = []  # per-element
    in_block = False
    harmonic_type = "spherical"

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue

        # Basis header.
        m_hdr = _RE_BASIS_HEADER.match(line)
        if m_hdr and not in_block:
            in_block = True
            if name is None:
                name = m_hdr.group(1) or m_hdr.group(2)
            ht = m_hdr.group(3)
            if ht:
                harmonic_type = ht.lower()
            continue

        if not in_block:
            continue

        # End of block.
        if _RE_END.match(line):
            in_block = False
            if current_symbol is not None:
                if current_label and current_primitives:
                    am = _AM_LABEL.get(current_label)
                    if am:
                        current_shells.append(
                            Shell(
                                angular_momentum=am,
                                harmonic_type=HarmonicType(harmonic_type),
                                primitives=current_primitives,
                            )
                        )
                if current_shells:
                    elements[current_symbol] = ElementBasis(
                        element=current_symbol, shells=current_shells
                    )
            current_symbol = None
            current_shells = []
            current_label = None
            current_primitives = []
            continue

        # Element + shell label line: "H    S"
        m_es = _RE_ELEM_SHELL.match(line)
        if m_es:
            # Finalise previous shell.
            if current_label and current_primitives:
                am = _AM_LABEL.get(current_label)
                if am:
                    current_shells.append(
                        Shell(
                            angular_momentum=am,
                            harmonic_type=HarmonicType(harmonic_type),
                            primitives=current_primitives,
                        )
                    )

            sym = m_es.group(1)
            lbl = m_es.group(2).upper()

            if sym != current_symbol:
                # Finalise previous element.
                if current_symbol and current_shells:
                    elements[current_symbol] = ElementBasis(
                        element=current_symbol, shells=current_shells
                    )
                if sym not in VALID_SYMBOLS:
                    raise ValueError(f"Unknown element symbol {sym!r} in NWChem basis")
                current_symbol = sym
                current_shells = []

            current_label = lbl
            current_primitives = []
            continue

        # Primitive line: "<exponent>  <coefficient>"
        parts = line.split()
        if len(parts) >= 2:
            try:
                exponent = float(parts[0])
                coeff = float(parts[1])
            except ValueError:
                continue
            current_primitives.append(Primitive(exponent=exponent, coefficient=coeff))

    # Finalise any remaining data.
    if current_label and current_primitives:
        am = _AM_LABEL.get(current_label)
        if am:
            current_shells.append(
                Shell(
                    angular_momentum=am,
                    harmonic_type=HarmonicType(harmonic_type),
                    primitives=current_primitives,
                )
            )
    if current_symbol and current_shells:
        elements[current_symbol] = ElementBasis(
            element=current_symbol, shells=current_shells
        )

    if not elements:
        raise ValueError("No element blocks found in NWChem basis input")

    return BasisSetData(
        schema_version="1.0.0",
        name=name or "<nwchem-import>",
        description=f"Imported from NWChem: {name or '<nwchem-import>'}",
        role=BasisRole(role),
        basis_family=family,
        elements=elements,
        provenance=Provenance(origin="file"),
    )


def from_nwchem_file(
    path: str | Path,
    *,
    name: str | None = None,
    role: str = "orbital",
    family: str | None = None,
) -> BasisSetData:
    """Read an NWChem basis file and parse it."""
    text = Path(path).read_text(encoding="utf-8")
    return from_nwchem(text, name=name, role=role, family=family)
