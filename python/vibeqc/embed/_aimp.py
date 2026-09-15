"""AIMP (Ab Initio Model Potential) parser for embedded-cluster CAS (MR7).

Reads the OpenMolcas EMB-AIMP and CG-AIMP/NR-AIMP/NP-AIMP library files
and provides per-element, per-crystal AIMP parameters for the frontier-ion
exchange + orthogonality shell between the QM cluster and the point-charge
array.

**Licensing**: The AIMP data files are part of OpenMolcas (LGPL 2.1)
and are NOT bundled with vibe-qc (MPL 2.0).  Users provide the path to
their local OpenMolcas installation; this module reads and parses them
at runtime.  See ``HANDOVER_PERIODIC_MULTIREF.md`` § 8.

**Format**: Each AIMP entry is identified by a key like

    /Mg.EMB-AIMP.Pascual.10s4p.1s1p.ECP.MgO.

and contains:

* A valence basis set (Gaussian primitives with contraction
  coefficients).
* An ECP-style local Coulomb+exchange potential (``M1`` block).
* Projector operators (``PROJOP``) enforcing orthogonality between the
  QM electrons and the embedded ion's core orbitals.
* A spectral representation operator (``SREP``) holding the core
  orbital basis.

Reference: Z. Barandiaran, L. Seijo, J. Chem. Phys. 89, 5739 (1988),
doi:10.1063/1.455549.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class AIMPEntry:
    """Parsed AIMP entry for one element in one crystal environment.

    Attributes
    ----------
    key : str
        Full AIMP key, e.g. ``/Mg.EMB-AIMP.Pascual.10s4p.1s1p.ECP.MgO.``.
    element : str
        Element symbol, e.g. ``"Mg"``.
    type : str
        ``"EMB-AIMP"``, ``"CG-AIMP"``, ``"NR-AIMP"``, or ``"NP-AIMP"``.
    author : str
        Author label, e.g. ``"Pascual"``.
    basis_label : str
        Valence basis label, e.g. ``"10s4p"``.
    ecp_label : str
        ECP projector label, e.g. ``"1s1p"``.
    crystal : str
        Crystal context, e.g. ``"MgO"``.
    citations : list[str]
        Reference lines from the file header.
    description : str
        Free-text description line (e.g. SCEI provenance).

    # Valence basis
    valence_s : list[tuple[float, float]] | None
        s-type primitives: ``[(exponent, coefficient), ...]``.
    valence_p : list[tuple[float, float]] | None
        p-type primitives.
    valence_d : list[tuple[float, float]] | None
        d-type primitives.

    # Local potential (M1 block: Coulomb + exchange)
    m1_n_terms : int
        Number of Gaussian terms in M1.
    m1_exponents : list[float]
        Exponents for the M1 local potential.
    m1_coefficients : list[float]
        Coefficients for the M1 local potential.

    # Projector operators
    projop_n_shells : int
        Number of projector shells.
    projop_shells : list[dict]
        Each shell: ``{"n": int, "l": int, "ng": int, "exponents": [...],
        "orb_coeffs": [[...], ...]}`` — the orbital coefficient matrix,
        with rows = primitive functions × columns = ``ng`` core orbitals.

    # Core repulsion
    corerep : float
        Core repulsion coefficient.
    """

    key: str = ""
    element: str = ""
    type: str = ""
    author: str = ""
    basis_label: str = ""
    ecp_label: str = ""
    crystal: str = ""
    citations: list[str] = field(default_factory=list)
    description: str = ""

    valence_s: list[tuple[float, float]] | None = None
    valence_p: list[tuple[float, float]] | None = None
    valence_d: list[tuple[float, float]] | None = None

    m1_n_terms: int = 0
    m1_exponents: list[float] = field(default_factory=list)
    m1_coefficients: list[float] = field(default_factory=list)

    projop_n_shells: int = 0
    projop_shells: list[dict] = field(default_factory=list)

    corerep: float = 0.0

    def has_valence(self) -> bool:
        """True if this AIMP carries an explicit valence basis."""
        return any(
            x is not None and len(x) > 0
            for x in (self.valence_s, self.valence_p, self.valence_d)
        )


# --------------------------------------------------------------------------- #
# Key parsing
# --------------------------------------------------------------------------- #

_AIMP_KEY_RE = re.compile(
    r"^/([A-Z][a-z]?)\.([A-Z]+-AIMP)\.([A-Za-z-]+)\.([0-9a-z]+)\.([0-9a-z]+)\.(?:ECP\.)?(.+)\.$"
)


def _parse_aimp_key(key: str) -> dict[str, str]:
    """Parse an AIMP key into its components.

    Example: ``/Mg.EMB-AIMP.Pascual.10s4p.1s1p.ECP.MgO.`` →
    ``{"element": "Mg", "type": "EMB-AIMP", "author": "Pascual",
    "basis_label": "10s4p", "ecp_label": "1s1p", "crystal": "MgO"}``.
    """
    m = _AIMP_KEY_RE.match(key.strip())
    if not m:
        raise ValueError(f"cannot parse AIMP key: {key!r}")
    return dict(
        element=m.group(1),
        type=m.group(2),
        author=m.group(3),
        basis_label=m.group(4),
        ecp_label=m.group(5),
        crystal=m.group(6),
    )


# --------------------------------------------------------------------------- #
# File parser
# --------------------------------------------------------------------------- #


def _parse_basis_block(
    lines: list[str], start: int
) -> tuple[list[tuple[float, float]], int]:
    """Parse a Gaussian basis block::

        n_prim  n_cont
            exp1
            exp2
            ...           (n_prim exponents)
         coeff11 coeff12 ... coeff1N   (n_cont rows, each n_prim coeffs)
         coeff21 coeff22 ... coeff2N
         ...

    Returns ``([(exp, coeff), ...], next_line_index)``
    where each exponent is paired with each contraction coefficient.
    """
    i = start
    # Skip leading blank / comment lines.
    while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith("*")):
        i += 1
    if i >= len(lines):
        return [], i
    header = lines[i].split()
    n_prim = int(header[0])
    n_cont = int(header[1]) if len(header) > 1 else 1
    i += 1
    # n_prim == 0: bare entry, no basis functions at all.
    if n_prim == 0:
        return [], i
    exponents: list[float] = []
    for _ in range(n_prim):
        while i < len(lines) and (
            not lines[i].strip() or lines[i].strip().startswith("*")
        ):
            i += 1
        if i >= len(lines):
            break
        exponents.append(float(lines[i].strip()))
        i += 1
    # Read n_prim coefficient lines, each with n_cont values.
    coeff_matrix: list[list[float]] = []
    for _ in range(n_prim):
        while i < len(lines) and (
            not lines[i].strip() or lines[i].strip().startswith("*")
        ):
            i += 1
        if i >= len(lines):
            break
        coeff_matrix.append(list(map(float, lines[i].split())))
        i += 1
    # Pair exponent k with contraction j.
    result: list[tuple[float, float]] = []
    for i_e, e in enumerate(exponents):
        if i_e < len(coeff_matrix):
            for c in coeff_matrix[i_e]:
                result.append((e, c))
    return result, i


def _parse_m1_block(
    lines: list[str], start: int
) -> tuple[int, list[float], list[float], int]:
    """Parse the M1 local potential block::

        M1
         n_terms
           exp1  exp2  ...  expN   (may span multiple lines)
           coeff1 coeff2 ... coeffN  (may span multiple lines)

    Returns ``(n_terms, exponents, coefficients, next_line_index)``.
    """
    i = start
    assert lines[i].strip() == "M1"
    i += 1
    n_terms = int(lines[i].strip())
    i += 1
    # Accumulate exponents across lines until we have n_terms values.
    exponents: list[float] = []
    while len(exponents) < n_terms and i < len(lines):
        parts = lines[i].split()
        if parts:
            exponents.extend(float(x) for x in parts)
        i += 1
    # Accumulate coefficients similarly.
    coefficients: list[float] = []
    while len(coefficients) < n_terms and i < len(lines):
        parts = lines[i].split()
        if parts:
            coefficients.extend(float(x) for x in parts)
        i += 1
    return n_terms, exponents[:n_terms], coefficients[:n_terms], i


def _parse_projop_block(lines: list[str], start: int) -> tuple[list[dict], int]:
    """Parse the PROJOP (projector operator) block.

    Format::

        PROJOP
         n_shells
           n_prim  n_cont    (for each shell: primitives × contractions)
           srep1 srep2 ...   (spectral representation coeffs, n_cont values)
             exp1 ... expN   (n_prim exponents)
           c11 c12 ...        (n_prim rows, n_cont values each)
           c21 c22 ...
           ...

    Returns ``([shell_dict, ...], next_line_index)``.
    """
    i = start
    assert lines[i].strip() == "PROJOP"
    i += 1
    n_shells = int(lines[i].strip())
    i += 1
    shells: list[dict] = []
    for _ in range(n_shells):
        # n_prim  n_cont
        header = lines[i].split()
        n_prim = int(header[0])
        n_cont = int(header[1])
        i += 1
        # Spectral representation operator coefficients (n_cont floats).
        srep = list(map(float, lines[i].split()))
        i += 1
        # Exponents (n_prim lines).
        exponents = []
        for _ in range(n_prim):
            exponents.append(float(lines[i].strip()))
            i += 1
        # Orbital coefficient matrix: n_prim rows, each with n_cont values.
        # Each row corresponds to one primitive function.
        orb_coeffs = []
        for _ in range(n_prim):
            orb_coeffs.append(list(map(float, lines[i].split())))
            i += 1
        shells.append(
            dict(
                n_prim=n_prim,
                n_cont=n_cont,
                srep_coeffs=srep,
                exponents=exponents,
                orb_coeffs=orb_coeffs,
            )
        )
    return shells, i


def _parse_one_aimp_entry(lines: list[str], i: int, entry: AIMPEntry) -> int:
    """Parse one AIMP entry starting at line *i* into *entry*.
    Returns the next line index after the entry."""
    # Collect citation lines (before the basis / ECP block).
    while i < len(lines):
        nl = lines[i].strip()
        if nl.startswith("*") or nl.startswith("#"):
            i += 1
            continue
        if not nl:
            i += 1
            continue
        # Stop at known block headers.
        if nl.startswith("/") or nl[0].isdigit() or nl in ("M1", "PROJOP"):
            break
        # This is a citation or description line.
        if (
            nl.startswith("From ")
            or nl.startswith("J.")
            or nl.startswith("Z.")
            or nl.startswith("M.")
        ):
            entry.citations.append(nl)
        else:
            entry.description = nl
        i += 1

    # Parse valence basis (if present).
    if i < len(lines) and not lines[i].strip().startswith("*") and lines[i].strip():
        charge_line = lines[i].strip()
        try:
            float(charge_line.split()[0])
            i += 1
        except (ValueError, IndexError):
            pass

    # s-type
    if i < len(lines) and "s-type" in lines[i].lower():
        i += 1
        entry.valence_s, i = _parse_basis_block(lines, i)
    # p-type
    if i < len(lines) and "p-type" in lines[i].lower():
        i += 1
        entry.valence_p, i = _parse_basis_block(lines, i)
    # d-type
    if i < len(lines) and "d-type" in lines[i].lower():
        i += 1
        entry.valence_d, i = _parse_basis_block(lines, i)

    # Skip separator comment block.
    while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith("*")):
        i += 1

    # Parse M1 block (local potential).
    if i < len(lines) and lines[i].strip() == "M1":
        (
            entry.m1_n_terms,
            entry.m1_exponents,
            entry.m1_coefficients,
            i,
        ) = _parse_m1_block(lines, i)

    # Skip M2.
    if i < len(lines) and lines[i].strip() == "M2":
        i += 1
        if i < len(lines):
            i += 1  # skip the M2 value line (usually "0")

    # Parse COREREP.
    if i < len(lines) and lines[i].strip() == "COREREP":
        i += 1
        entry.corerep = float(lines[i].strip())
        i += 1

    # Parse PROJOP (projector operators).
    if i < len(lines) and lines[i].strip() == "PROJOP":
        entry.projop_shells, i = _parse_projop_block(lines, i)

    # Skip trailing data (Spectral Representation Operator, etc.)
    while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("/"):
        i += 1

    return i


def parse_aimp_file(path: str) -> list[AIMPEntry]:
    """Parse one or more AIMP entries from an OpenMolcas AIMP library file.

    Reads files like ``EMB-AIMP``, ``CG-AIMP``, ``NR-AIMP``, or
    ``NP-AIMP`` from an OpenMolcas ``basis_library/`` directory.

    Parameters
    ----------
    path : str
        Path to the AIMP library file.

    Returns
    -------
    list[AIMPEntry]
        Parsed AIMP entries in file order.
    """
    with open(path) as fh:
        raw = fh.read()

    lines = raw.splitlines()
    entries: list[AIMPEntry] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip blank lines and comment/header lines.
        if not line or line.startswith("*") or line.startswith("#"):
            i += 1
            continue

        # Check for AIMP key pattern.
        if line.startswith("/") and line.endswith(".") and "AIMP" in line:
            key = line
            try:
                parsed = _parse_aimp_key(key)
            except ValueError:
                i += 1
                continue
            entry = AIMPEntry(key=key, **parsed)
            entry_start = i
            i += 1

            try:
                i = _parse_one_aimp_entry(lines, i, entry)
            except (ValueError, IndexError) as exc:
                # Skip entries we can't fully parse; the EMB-AIMP file
                # has many format variants accumulated over 30 years.
                import sys

                print(
                    f"[aimp] skipping {key!r} at line {entry_start + 1}: {exc}",
                    file=sys.stderr,
                )
                # Advance past this entry's key line and try the next.
                i = entry_start + 1
                while i < len(lines):
                    nl = lines[i].strip()
                    if nl.startswith("/") and nl.endswith(".") and "AIMP" in nl:
                        break
                    i += 1
                continue

            entries.append(entry)
        else:
            i += 1

    return entries


# --------------------------------------------------------------------------- #
# Convenience lookup
# --------------------------------------------------------------------------- #


def find_aimp(
    entries: list[AIMPEntry],
    element: str,
    crystal: str | None = None,
    *,
    with_valence: bool | None = None,
    author: str | None = None,
) -> list[AIMPEntry]:
    """Filter AIMP entries by element, crystal, and other criteria.

    Parameters
    ----------
    entries : list[AIMPEntry]
        Parsed entries from ``parse_aimp_file``.
    element : str
        Element symbol, e.g. ``"Mg"``.
    crystal : str, optional
        Crystal context, e.g. ``"MgO"``.
    with_valence : bool, optional
        ``True`` → only entries carrying a valence basis.
        ``False`` → only bare (ECP-only) entries.
        ``None`` → don't filter on valence.
    author : str, optional
        Author label, e.g. ``"Pascual"``.

    Returns
    -------
    list[AIMPEntry]
    """
    result = []
    for e in entries:
        if e.element != element:
            continue
        if crystal is not None and e.crystal != crystal:
            continue
        if with_valence is not None and e.has_valence() != with_valence:
            continue
        if author is not None and e.author != author:
            continue
        result.append(e)
    return result


def resolve_aimp_library(
    openmolcas_path: str | None = None,
) -> str:
    """Resolve the path to the EMB-AIMP library file.

    Checks (in order):
    1. ``$OPENMOLCAS_BASIS_LIBRARY`` environment variable.
    2. ``<openmolcas_path>/basis_library/EMB-AIMP``.
    3. ``~/gitlab/OpenMolcas/build/basis_library/EMB-AIMP`` (vibe-qc
       developer convention).
    4. ``~/gitlab/OpenMolcas/basis_library/EMB-AIMP``.

    Returns the path to the EMB-AIMP file.

    Raises ``FileNotFoundError`` if no library is found.
    """
    candidates: list[str] = []

    env = os.environ.get("OPENMOLCAS_BASIS_LIBRARY")
    if env:
        p = os.path.join(env, "EMB-AIMP")
        if os.path.isfile(p):
            return p
        candidates.append(p)

    if openmolcas_path:
        p = os.path.join(openmolcas_path, "basis_library", "EMB-AIMP")
        if os.path.isfile(p):
            return p
        candidates.append(p)

    for base in (
        os.path.expanduser("~/gitlab/OpenMolcas/build"),
        os.path.expanduser("~/gitlab/OpenMolcas"),
    ):
        p = os.path.join(base, "basis_library", "EMB-AIMP")
        if os.path.isfile(p):
            return p
        candidates.append(p)

    raise FileNotFoundError(
        "Cannot find OpenMolcas EMB-AIMP library.  Set "
        "$OPENMOLCAS_BASIS_LIBRARY or pass openmolcas_path.  Searched:\n  "
        + "\n  ".join(candidates)
    )
