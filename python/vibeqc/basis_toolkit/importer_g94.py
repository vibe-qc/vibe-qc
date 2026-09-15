"""Import Gaussian94 (.g94) text format into the canonical model.

G94 is the de facto interchange format for Gaussian-type basis sets.
Every major QC code parses it, but with slight variations in whitespace
tolerance, comment handling, and ECP linkage conventions.  This importer
follows the libint-compatible convention: ``!`` comment lines, ``****``
element separators, ``SP`` label for shared-exponent s+p shells, and
two-coefficient lines for SP primitives.

The importer is a *reverse* of :mod:`.exporter_g94` -- the two are
designed to round-trip losslessly within the 7-field G94 limitation
(metadata beyond name/role/family is not present in G94 text).
"""

from __future__ import annotations

import re
from pathlib import Path

from ._element_data import ATOMIC_NUMBER, VALID_SYMBOLS
from .model import (
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

__all__ = ["from_g94", "from_g94_file"]

# Element-header line:  "H     0",  "Fe     0", "RB-ECP     4     28"
# G94 ECP elements use "SYM-ECP  nshell  ncore" instead of "SYM  0".
_RE_ELEM_HEADER = re.compile(r"^([A-Z][A-Za-z]?)\s+0\s*$")
_RE_ECP_HEADER = re.compile(r"^([A-Z][A-Za-z]?)-ECP\s+(\d+)\s+(\d+)\s*$")

# ECP potential labels in G94.
# ECP channel labels. BSE / Gaussian write the local channel as
# "<L> potential" and the projected channels as "<l>-<L> potential", where
# L is the letter of the local (lmax) channel: "f potential" / "s-f
# potential" for lmax=3, "h potential" / "s-h potential" for the lmax=5
# Stuttgart sets (Cs onward in dhf-*). NWChem writes "ul" for the local
# channel. Match the first letter; that is the channel's angular momentum.
_RE_ECP_LABEL = re.compile(
    r"^([spdfghik])(?:-([spdfghik]|ul))?\s+potential$", re.IGNORECASE
)
_ECP_L_OF_LETTER: dict[str, int] = {
    letter: l for l, letter in enumerate("spdfghik")
}


def _ecp_label_l(label: str) -> int | None:
    m = _RE_ECP_LABEL.match(label.strip())
    if m is None:
        return None
    return _ECP_L_OF_LETTER[m.group(1).lower()]


# Map non-standard uppercase element symbols to canonical form.
# Auto-normalise: any all-uppercase two-letter symbol -> Title case.
def _normalise_symbol(raw: str) -> str:
    """Normalise an element symbol to proper case.

    "rb" -> "Rb", "RB" -> "Rb", "h" -> "H", "SI" -> "Si".
    """
    if len(raw) == 1:
        return raw.upper()
    if len(raw) == 2:
        return raw[0].upper() + raw[1].lower()
    return raw


# Shell-header line:  "S   3   1.00",  "SP   3   1.00",  "D   5   1.00"
_RE_SHELL_HEADER = re.compile(
    r"^(?P<label>S|P|D|F|G|H|I|J|K|L|M|SP)\s+(?P<n_prim>\d+)\s+(?P<scale>[\d.]+)"
)

# Angular-momentum label -> [L] or [L1, L2] for SP.
_LABEL_TO_AM: dict[str, list[int]] = {
    "S": [0],
    "P": [1],
    "D": [2],
    "F": [3],
    "G": [4],
    "H": [5],
    "I": [6],
    "J": [7],
    "K": [8],
    "L": [9],
    "M": [10],
    "SP": [0, 1],
}


def _parse_float(token: str) -> float:
    """Parse a float token, handling Fortran D-format (e.g. ``0.123D+02``)."""
    return float(token.replace("D", "E").replace("d", "e"))


def _strip_comment(line: str) -> str:
    idx = line.find("!")
    if idx >= 0:
        return line[:idx]
    return line


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def from_g94(
    text: str,
    *,
    name: str | None = None,
    role: str = "orbital",
    family: str | None = None,
    harmonic_type: str = "spherical",
) -> BasisSetData:
    """Parse a G94 text string into :class:`BasisSetData`.

    Parameters
    ----------
    text:
        Complete G94 text (may contain comment lines starting with ``!``).
    name:
        Override basis-set name.  If ``None``, extracted from the first
        ``! Basis: ...`` comment line, or defaults to ``"<g94-import>"``.
    role:
        Basis-set role (``"orbital"``, ``"auxiliary"``, ``"fitting"``,
        ``"ecp"``).  Default ``"orbital"`` -- G94 carries no role flag.
    family:
        Basis family hint.  Extracted from ``! Role: ...`` if not provided.
    harmonic_type:
        Default harmonic convention (``"spherical"`` or ``"cartesian"``).
        G94 does not encode spherical vs Cartesian, so the caller must
        supply this.  Default ``"spherical"``.

    Returns
    -------
    BasisSetData
    """
    # -- Parse header comments -----------------------------------------------
    lines = text.splitlines()
    parsed_name = name
    parsed_family = family
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("! Basis:"):
            if parsed_name is None:
                parsed_name = stripped[len("! Basis:") :].strip()
        elif stripped.startswith("! Role:"):
            role_part = stripped[len("! Role:") :].strip().split()
            if role_part:
                parsed_role = role_part[0].lower()
                if parsed_role in ("orbital", "auxiliary", "fitting", "ecp"):
                    role = parsed_role
                if len(role_part) > 1 and parsed_family is None:
                    parsed_family = role_part[1]

    if parsed_name is None:
        parsed_name = "<g94-import>"

    # -- Split into per-element blocks ---------------------------------------
    elements: dict[str, ElementBasis] = {}
    ecps: dict[str, ECPEntry] = {}
    current_symbol: str | None = None
    current_shells: list[Shell] = []

    i = 0
    while i < len(lines):
        raw = lines[i]
        clean = _strip_comment(raw).strip()
        i += 1

        # A blank line is not a terminator: BSE and libint files carry blank
        # lines inside a block (def2-TZVP-C has one right after the Ga..Br
        # headers, which used to drop those five elements from the QVF
        # record). Only "****" ends an element block.
        if not clean:
            continue

        # End of previous element block.
        if clean == "****":
            if current_symbol is not None and current_shells:
                elements[current_symbol] = ElementBasis(
                    element=current_symbol, shells=current_shells
                )
            current_symbol = None
            current_shells = []
            continue

        # Element header:  "H     0" or "RB-ECP     4     28"
        m_elem = _RE_ELEM_HEADER.match(clean)
        m_ecp = _RE_ECP_HEADER.match(clean) if not m_elem else None
        if m_elem or m_ecp:
            raw_sym = (m_elem or m_ecp).group(1)
            current_symbol = _normalise_symbol(raw_sym)
            if current_symbol not in VALID_SYMBOLS:
                raise ValueError(
                    f"Unknown element symbol {current_symbol!r} in G94 header"
                )
            current_shells = []
            if m_ecp:
                # Parse ECP data block.
                n_core = int(m_ecp.group(3))
                potentials: list[ECPPotential] = []
                current_l: int | None = None
                current_terms: list[ECPTerm] = []

                while i < len(lines):
                    raw_ecp = lines[i]
                    clean_ecp = _strip_comment(raw_ecp).strip()

                    # Check for end of ECP block.
                    if (
                        not clean_ecp
                        or clean_ecp == "****"
                        or _RE_ELEM_HEADER.match(clean_ecp)
                        or _RE_ECP_HEADER.match(clean_ecp)
                        or _RE_SHELL_HEADER.match(clean_ecp)
                    ):
                        break

                    # ECP channel label: "S-G potential", etc. (case-insensitive).
                    l_val = _ecp_label_l(clean_ecp)
                    if l_val is not None:
                        if current_l is not None and current_terms:
                            potentials.append(
                                ECPPotential(
                                    angular_momentum=current_l, terms=current_terms
                                )
                            )
                        current_l = l_val
                        current_terms = []
                        i += 1
                        continue

                    # Try as count line or primitive line.
                    i += 1
                    parts = clean_ecp.split()
                    if len(parts) == 1:
                        # Count line: number of terms for this channel.
                        continue
                    elif len(parts) >= 3:
                        # Primitive line: power exponent coefficient.
                        try:
                            power = int(float(parts[0]))
                            exponent = _parse_float(parts[1])
                            coeff = _parse_float(parts[2])
                            current_terms.append(
                                ECPTerm(
                                    coefficient=coeff,
                                    exponent=exponent,
                                    power=power,
                                )
                            )
                        except (ValueError, IndexError):
                            pass

                # Finalise last channel.
                if current_l is not None and current_terms:
                    potentials.append(
                        ECPPotential(angular_momentum=current_l, terms=current_terms)
                    )

                if potentials:
                    ecps[current_symbol] = ECPEntry(
                        element=current_symbol,
                        n_core_electrons=n_core,
                        angular_momentum_max=max(
                            p.angular_momentum for p in potentials
                        ),
                        potentials=potentials,
                    )
            continue

        # Shell header:  "S   3   1.00"
        m_shell = _RE_SHELL_HEADER.match(clean)
        if m_shell:
            label = m_shell.group("label")
            n_prim = int(m_shell.group("n_prim"))
            am = _LABEL_TO_AM.get(label)
            if am is None:
                raise ValueError(f"Unknown shell label {label!r} at line {i}: {raw!r}")

            # Read primitive lines.
            primitives: list[Primitive] = []
            if am == [0, 1]:
                # SP shell: each line has exponent, s-coeff, p-coeff. The model
                # stores SP primitives "blocked": all N s-contraction primitives
                # first, then all N p-contraction primitives (exponents shared/
                # repeated) -- matching exporter_g94/orca/nwchem and the crystal/
                # bse importers. A previous version appended them interleaved
                # (s0,p0,s1,p1,...); the blocked exporters then mis-read that as
                # s-block=(s0,p0), p-block=(s1,p1), corrupting every SP basis
                # (e.g. STO-3G/6-31G) on round-trip.
                sp_exps: list[float] = []
                sp_s: list[float] = []
                sp_p: list[float] = []
                for _ in range(n_prim):
                    if i >= len(lines):
                        raise ValueError(
                            f"Unexpected end of file reading SP primitives "
                            f"for {current_symbol}"
                        )
                    parts = _strip_comment(lines[i]).split()
                    i += 1
                    if len(parts) < 3:
                        raise ValueError(
                            f"SP primitive line needs 3 values at line {i}: {lines[i - 1]!r}"
                        )
                    sp_exps.append(_parse_float(parts[0]))
                    sp_s.append(_parse_float(parts[1]))
                    sp_p.append(_parse_float(parts[2]))
                for exp, s_coeff in zip(sp_exps, sp_s):
                    primitives.append(Primitive(exponent=exp, coefficient=s_coeff))
                for exp, p_coeff in zip(sp_exps, sp_p):
                    primitives.append(Primitive(exponent=exp, coefficient=p_coeff))
            else:
                # Pure shell: each line has exponent, coefficient.
                for _ in range(n_prim):
                    if i >= len(lines):
                        raise ValueError(
                            f"Unexpected end of file reading primitives "
                            f"for {current_symbol} shell {label}"
                        )
                    parts = _strip_comment(lines[i]).split()
                    i += 1
                    if len(parts) < 2:
                        raise ValueError(
                            f"Primitive line needs 2 values at line {i}: {lines[i - 1]!r}"
                        )
                    exp = _parse_float(parts[0])
                    coeff = _parse_float(parts[1])
                    primitives.append(Primitive(exponent=exp, coefficient=coeff))

            current_shells.append(
                Shell(
                    angular_momentum=am,
                    harmonic_type=HarmonicType(harmonic_type),
                    primitives=primitives,
                )
            )
            continue

        # Skip blank/comment-only lines.
        if clean:
            raise ValueError(f"Unrecognised G94 line {i}: {raw!r}")

    # -- Finalise last element -----------------------------------------------
    if current_symbol is not None and current_shells:
        elements[current_symbol] = ElementBasis(
            element=current_symbol, shells=current_shells
        )

    if not elements and not ecps:
        raise ValueError("No element blocks found in G94 text")

    # An element that carries an ECP block in the same text is paired with
    # it; the link is what lets a consumer tell a valence-only block from an
    # all-electron one without re-deriving it from the core count.
    for symbol, element in elements.items():
        if symbol in ecps and element.ecp_id is None:
            element.ecp_id = symbol

    return BasisSetData(
        schema_version="1.0.0",
        name=parsed_name,
        description=f"Imported from G94: {parsed_name}",
        role=BasisRole(role),
        basis_family=parsed_family,
        elements=elements,
        ecps=ecps if ecps else None,
        provenance=Provenance(origin="file"),
    )


def from_g94_file(
    path: str | Path,
    *,
    name: str | None = None,
    role: str = "orbital",
    family: str | None = None,
    harmonic_type: str = "spherical",
) -> BasisSetData:
    """Read a ``.g94`` file and parse it into :class:`BasisSetData`.

    Parameters
    ----------
    path:
        Path to the ``.g94`` file.
    name, role, family, harmonic_type:
        Passed through to :func:`from_g94`.
    """
    text = Path(path).read_text(encoding="utf-8")
    return from_g94(
        text, name=name, role=role, family=family, harmonic_type=harmonic_type
    )
