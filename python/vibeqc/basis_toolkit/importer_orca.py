"""Import an ORCA ``%basis`` block, shells and ECPs, into the canonical model.

Reads the forms the ORCA 6.1 manual documents (sections 2.7.5 and 2.7.6)
and :mod:`.exporter_orca` writes::

    %basis
      NewGTO Ag            # or the atomic number, NewGTO 47
        S  3
          1   12.567714    0.0151
          ...
      end
      NewECP Ag
        N_core 28
        lmax g
        s 3
          1   12.567714   255.054771   2
          ...
      end
    end

Shell labels may be letters (``s p d f g h i k``, any case) or the integer
angular momentum. A shell with a ``SCALE x`` other than 1 is refused: the
manual does not say how ORCA applies it. ECP rows are
``index exponent coefficient power`` with ``power`` the Gaussian-format n of
``r^(n-2)``; the ``lmax`` channel is the local one.
"""

from __future__ import annotations

import re
from pathlib import Path

from ._element_data import ATOMIC_NUMBER, SYMBOL, VALID_SYMBOLS
from .model import (
    SCHEMA_VERSION,
    BasisRole,
    BasisSetData,
    ECPEntry,
    ECPPotential,
    ECPTerm,
    ElementBasis,
    HarmonicType,
    Primitive,
    Provenance,
    Shell,
)

__all__ = ["from_orca", "from_orca_file"]

_ORCA_LETTERS = "spdfghik"
_RE_BLOCK = re.compile(r"^(NewGTO|NewECP)\s+(\S+)\s*$", re.IGNORECASE)
_RE_END = re.compile(r"^end\s*$", re.IGNORECASE)


def _float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _l_of(label: str) -> int:
    text = label.strip()
    if text.isdigit():
        return int(text)
    letter = text.lower()
    if len(letter) != 1 or letter not in _ORCA_LETTERS:
        raise ValueError(f"unknown ORCA shell label {label!r}")
    return _ORCA_LETTERS.index(letter)


def _symbol(token: str) -> str:
    """``"Ag"``, ``"AG"`` or ``"47"`` -> ``"Ag"``."""
    text = token.strip().strip('"')
    if text.isdigit():
        z = int(text)
        if z not in SYMBOL:
            raise ValueError(f"unknown atomic number {z} in ORCA input")
        return SYMBOL[z]
    sym = text[:1].upper() + text[1:].lower()
    if sym not in VALID_SYMBOLS:
        raise ValueError(f"unknown element symbol {token!r} in ORCA input")
    return sym


def _clean(raw: str) -> str:
    return raw.split("#", 1)[0].strip()


def from_orca(
    text: str,
    *,
    name: str | None = None,
    role: str = "orbital",
    family: str | None = None,
    harmonic_type: str = "spherical",
) -> BasisSetData:
    """Parse ORCA ``NewGTO`` / ``NewECP`` blocks into :class:`BasisSetData`.

    The ``%basis`` wrapper and its final ``end`` are optional. A ``NewGTO``
    that names a built-in basis (a quoted string) cannot be resolved here
    and is refused.
    """
    lines = [_clean(raw) for raw in text.splitlines()]
    elements: dict[str, ElementBasis] = {}
    ecps: dict[str, ECPEntry] = {}
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        i += 1
        if not line or line.lower().startswith("%basis") or _RE_END.match(line):
            continue
        m = _RE_BLOCK.match(line)
        if not m:
            continue
        kind, sym = m.group(1).lower(), _symbol(m.group(2))
        if kind == "newgto":
            shells: list[Shell] = []
            while i < n and not _RE_END.match(lines[i]):
                head = lines[i].split()
                i += 1
                if not head:
                    continue
                if head[0].startswith('"'):
                    raise ValueError(f"{sym}: NewGTO names a built-in basis {head[0]}; only explicit shells can be imported")
                l = _l_of(head[0])
                nprim = int(head[1])
                if len(head) >= 4 and head[2].lower() == "scale" and _float(head[3]) != 1.0:
                    # The ORCA 6.1 manual mentions SCALE without saying whether
                    # it multiplies the exponents by X or by X**2 (CRYSTAL's
                    # rule); importing would mean guessing.
                    raise ValueError(
                        f"{sym}: NewGTO shell with SCALE {head[3]}; ORCA does not document "
                        "the scaling rule, so write the scaled exponents out instead"
                    )
                prims: list[Primitive] = []
                for _ in range(nprim):
                    while i < n and not lines[i]:
                        i += 1
                    row = lines[i].split()
                    i += 1
                    prims.append(Primitive(exponent=_float(row[1]), coefficient=_float(row[2])))
                shells.append(Shell(angular_momentum=[l], harmonic_type=HarmonicType(harmonic_type), primitives=prims))
            i += 1  # the block's end
            if shells:
                elements[sym] = ElementBasis(element=sym, shells=shells)
        else:
            n_core = None
            lmax = None
            channels: dict[int, list[ECPTerm]] = {}
            while i < n and not _RE_END.match(lines[i]):
                head = lines[i].split()
                i += 1
                if not head:
                    continue
                key = head[0].lower()
                if key == "n_core":
                    n_core = int(head[1])
                    continue
                if key == "lmax":
                    lmax = _l_of(head[1])
                    continue
                l = _l_of(head[0])
                nterm = int(head[1])
                terms: list[ECPTerm] = []
                for _ in range(nterm):
                    while i < n and not lines[i]:
                        i += 1
                    row = lines[i].split()
                    i += 1
                    terms.append(ECPTerm(coefficient=_float(row[2]), exponent=_float(row[1]), power=int(row[3])))
                channels[l] = terms
            i += 1
            if n_core is None or lmax is None:
                raise ValueError(f"{sym}: NewECP needs N_core and lmax")
            if lmax not in channels:
                channels[lmax] = [ECPTerm(coefficient=0.0, exponent=1.0, power=2)]
            potentials = [ECPPotential(angular_momentum=lmax, terms=channels[lmax])] + [
                ECPPotential(angular_momentum=l, terms=channels[l]) for l in sorted(channels) if l != lmax
            ]
            ecps[sym] = ECPEntry(element=sym, n_core_electrons=n_core, angular_momentum_max=lmax, potentials=potentials)

    if not elements and not ecps:
        raise ValueError("No NewGTO or NewECP blocks found in ORCA basis input")
    for sym, element in elements.items():
        element.ecp_id = sym if sym in ecps else None
    order = lambda d: dict(sorted(d.items(), key=lambda kv: ATOMIC_NUMBER.get(kv[0], 999)))
    label = name or "<orca-import>"
    return BasisSetData(
        schema_version=SCHEMA_VERSION,
        name=label,
        description=f"Imported from ORCA: {label}",
        role=BasisRole(role),
        basis_family=family,
        elements=order(elements),
        ecps=order(ecps) or None,
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
    return from_orca(text, name=name, role=role, family=family, harmonic_type=harmonic_type)
