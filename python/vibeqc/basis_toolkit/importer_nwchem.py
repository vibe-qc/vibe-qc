"""Import NWChem ``basis`` and ``ECP`` blocks into the canonical model.

Reads what :mod:`.exporter_nwchem` writes and NWChem itself accepts::

    basis "name" spherical
      H    S
        <exponent>    <coefficient>
      O    S
        ...
    end
    ECP
      Ag nelec 28
      Ag ul
        <power>  <exponent>  <coefficient>
      Ag S
        ...
    END

Shell labels follow NWChem: ``S P D F G H I K L M ...`` (no ``J``),
any case, and ``SP`` (or ``L``-free ``SP``) rows carry two coefficients.
ECP ``power`` is the Gaussian-format n of ``r^(n-2)``; ``ul`` is the local
channel, whose angular momentum NWChem does not state: it is taken as one
above the highest projector, which leaves the operator unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

from ._element_data import ATOMIC_NUMBER, VALID_SYMBOLS
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

__all__ = ["from_nwchem", "from_nwchem_file"]

_LETTERS = "SPDFGHIKLMNORTU"
_RE_BASIS_HEADER = re.compile(r'^basis\s+(?:"([^"]*)"|(\S+))?\s*(spherical|cartesian)?', re.IGNORECASE)
_RE_ECP_HEADER = re.compile(r"^ecp(\s|$)", re.IGNORECASE)
_RE_END = re.compile(r"^end\s*$", re.IGNORECASE)


def _float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _symbol(token: str) -> str:
    sym = token[:1].upper() + token[1:].lower()
    if sym not in VALID_SYMBOLS:
        raise ValueError(f"Unknown element symbol {token!r} in NWChem input")
    return sym


def _l_of(label: str) -> list[int]:
    text = label.upper()
    if text == "SP":
        return [0, 1]
    if len(text) != 1 or text not in _LETTERS:
        raise ValueError(f"unknown NWChem shell label {label!r}")
    return [_LETTERS.index(text)]


def _clean(raw: str) -> str:
    return raw.split("#", 1)[0].strip()


def from_nwchem(
    text: str,
    *,
    name: str | None = None,
    role: str = "orbital",
    family: str | None = None,
) -> BasisSetData:
    """Parse NWChem ``basis`` and ``ECP`` blocks into :class:`BasisSetData`."""
    lines = [_clean(raw) for raw in text.splitlines()]
    elements: dict[str, ElementBasis] = {}
    ecp_raw: dict[str, dict] = {}
    harmonic_type = "spherical"
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        i += 1
        if not line:
            continue
        m_basis = _RE_BASIS_HEADER.match(line)
        if m_basis and not _RE_ECP_HEADER.match(line):
            if name is None:
                name = m_basis.group(1) or m_basis.group(2)
            if m_basis.group(3):
                harmonic_type = m_basis.group(3).lower()
            current: Shell | None = None
            while i < n and not _RE_END.match(lines[i]):
                parts = lines[i].split()
                i += 1
                if not parts:
                    continue
                if parts[0][0].isalpha():
                    sym, am = _symbol(parts[0]), _l_of(parts[1])
                    current = Shell(angular_momentum=am, harmonic_type=HarmonicType(harmonic_type), primitives=[])
                    elements.setdefault(sym, ElementBasis(element=sym, shells=[])).shells.append(current)
                    continue
                if current is None:
                    raise ValueError(f"NWChem primitive before a shell label: {lines[i - 1]!r}")
                values = [_float(t) for t in parts]
                if current.angular_momentum == [0, 1]:
                    # Blocked like every other importer: s rows first, then p.
                    k = len(current.primitives) // 2
                    current.primitives.insert(k, Primitive(exponent=values[0], coefficient=values[1]))
                    current.primitives.append(Primitive(exponent=values[0], coefficient=values[2]))
                else:
                    current.primitives.append(Primitive(exponent=values[0], coefficient=values[1]))
            i += 1
            continue
        if _RE_ECP_HEADER.match(line):
            channel: list[ECPTerm] | None = None
            while i < n and not _RE_END.match(lines[i]):
                parts = lines[i].split()
                i += 1
                if not parts:
                    continue
                if parts[0][0].isalpha():
                    sym = _symbol(parts[0])
                    entry = ecp_raw.setdefault(sym, {"n_core": None, "channels": {}})
                    key = parts[1].lower()
                    if key == "nelec":
                        entry["n_core"] = int(parts[2])
                        channel = None
                        continue
                    l = "ul" if key == "ul" else _l_of(parts[1])[0]
                    channel = entry["channels"].setdefault(l, [])
                    continue
                if channel is None:
                    raise ValueError(f"NWChem ECP row before a channel label: {lines[i - 1]!r}")
                channel.append(ECPTerm(coefficient=_float(parts[2]), exponent=_float(parts[1]), power=int(parts[0])))
            i += 1
            continue

    ecps: dict[str, ECPEntry] = {}
    for sym, entry in ecp_raw.items():
        if entry["n_core"] is None:
            raise ValueError(f"{sym}: NWChem ECP block without nelec")
        projectors = {l: t for l, t in entry["channels"].items() if l != "ul"}
        lmax = max(projectors) + 1 if projectors else 0
        local = entry["channels"].get("ul") or [ECPTerm(coefficient=0.0, exponent=1.0, power=2)]
        potentials = [ECPPotential(angular_momentum=lmax, terms=local)] + [
            ECPPotential(angular_momentum=l, terms=projectors[l]) for l in sorted(projectors)
        ]
        ecps[sym] = ECPEntry(element=sym, n_core_electrons=entry["n_core"], angular_momentum_max=lmax, potentials=potentials)

    if not elements and not ecps:
        raise ValueError("No element blocks found in NWChem basis input")
    for sym, element in elements.items():
        element.ecp_id = sym if sym in ecps else None
    order = lambda d: dict(sorted(d.items(), key=lambda kv: ATOMIC_NUMBER.get(kv[0], 999)))
    label = name or "<nwchem-import>"
    return BasisSetData(
        schema_version=SCHEMA_VERSION,
        name=label,
        description=f"Imported from NWChem: {label}",
        role=BasisRole(role),
        basis_family=family,
        elements=order(elements),
        ecps=order(ecps) or None,
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
