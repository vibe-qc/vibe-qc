"""Import Basis Set Exchange (BSE) JSON format into the canonical model.

The BSE JSON format is the format returned by the Basis Set Exchange API
(e.g. ``https://www.basissetexchange.org/api/basis/cc-pVDZ/format/json``).
This module converts it (and the related QCSchema-style structured JSON) into
the typed :class:`BasisSetData` model.

Key mapping differences from BSE -> QVF-Basis:

* BSE ``elements`` keys are atomic-number strings (``"1"``, ``"6"``) --
  converted to element symbols (``"H"``, ``"C"``).
* BSE exponents and coefficients are stored as **strings** -- converted to
  ``float``.
* BSE ``function_type`` must be ``"gto"``; anything else raises ``ValueError``.
* ``harmonic_type`` defaults to ``"spherical"`` (BSE rarely includes it).
* ``references[].reference_key`` -> QVF ``key``, ``reference_description`` ->
  ``description``, ``reference_doi`` -> ``doi``.
* ``version`` -> ``provenance.source_version``.
* ``provenance.origin`` = ``"bse"``, ``schema_version`` = ``"1.0.0"``.

QCSchema-style input (detected by ``center_data`` instead of ``elements``)
is also accepted and converted to the element-based layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from ._element_data import SYMBOL as _ATOMIC_NUMBER_TO_SYMBOL
from .model import (
    BasisRole,
    BasisSetData,
    ElementBasis,
    HarmonicType,
    Primitive,
    Provenance,
    Reference,
    Shell,
)

__all__ = ["from_bse_json", "from_bse_file"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def from_bse_json(
    data: dict,
    *,
    harmonic_type: str = "spherical",
) -> BasisSetData:
    """Parse a BSE (or QCSchema-style) JSON dict into :class:`BasisSetData`.

    Parameters
    ----------
    data : dict
        A dict in BSE JSON format (with ``elements`` as top-level key) or
        QCSchema-style format (with ``center_data`` as top-level key).
    harmonic_type : str
        Harmonic convention to assign to every shell (``"spherical"`` or
        ``"cartesian"``).  Default is ``"spherical"``.

    Returns
    -------
    BasisSetData
        The fully-typed canonical representation.

    Raises
    ------
    ValueError
        If a shell's ``function_type`` is not ``"gto"``, if an element
        key cannot be mapped to a symbol, or if the dict structure is
        otherwise unrecognisable.
    """
    # -- Detect QCSchema-style input -----------------------------------------
    if "center_data" in data and "elements" not in data:
        data = _convert_qcschema(data)

    # -- BSE-specific top-level fields ---------------------------------------
    name: str = data.get("name", "unknown")
    description: str = data.get("description", "")
    role_str: str = data.get("basis_set_role", data.get("role", "orbital"))
    basis_family: str | None = data.get("basis_set_family") or data.get("basis_family")
    version: str | None = data.get("version")

    # -- Elements ------------------------------------------------------------
    raw_elements: dict = data.get("elements", {})
    if not isinstance(raw_elements, dict):
        raise ValueError(
            f"'elements' must be a dict, got {type(raw_elements).__name__}"
        )

    elements: Dict[str, ElementBasis] = {}
    for atom_key, atom_data in raw_elements.items():
        symbol = _atom_key_to_symbol(atom_key)
        element_shells = _parse_shells(atom_data, harmonic_type=harmonic_type)
        ecp_id = atom_data.get("ecp_id") if isinstance(atom_data, dict) else None
        elements[symbol] = ElementBasis(
            element=symbol,
            shells=element_shells,
            ecp_id=ecp_id,
        )

    # -- References ----------------------------------------------------------
    refs: List[Reference] = []
    for r in data.get("references", []):
        refs.append(
            Reference(
                key=r.get("reference_key", r.get("key", "")),
                description=r.get("reference_description", r.get("description", "")),
                doi=r.get("reference_doi", r.get("doi")),
            )
        )

    # -- Provenance ----------------------------------------------------------
    provenance = Provenance(
        origin="bse",
        source_version=version,
    )

    return BasisSetData(
        schema_version="1.0.0",
        name=name,
        description=description,
        role=BasisRole(role_str),
        basis_family=basis_family or None,
        elements=elements,
        references=refs,
        provenance=provenance,
    )


def from_bse_file(
    path: str | Path,
    *,
    harmonic_type: str = "spherical",
) -> BasisSetData:
    """Read a BSE JSON file from *path* and parse into :class:`BasisSetData`.

    Parameters
    ----------
    path : str or Path
        Path to a BSE-format ``.json`` file.
    harmonic_type : str
        Harmonic convention for every shell (``"spherical"`` or
        ``"cartesian"``).

    Returns
    -------
    BasisSetData

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    json.JSONDecodeError
        If the file contains invalid JSON.
    ValueError
        If the content is not a valid BSE basis-set dict.
    """
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object at top level, got {type(data).__name__}"
        )
    return from_bse_json(data, harmonic_type=harmonic_type)


# ---------------------------------------------------------------------------
# QCSchema -> BSE-like conversion
# ---------------------------------------------------------------------------


def _convert_qcschema(data: dict) -> dict:
    """Convert a QCSchema-style dict (``center_data``) to BSE-style (``elements``).

    QCSchema layout:
        ``center_data``: dict keyed by center index (str), each value has
        ``electron_shells`` (or ``atomic_shells``) and an
        ``element_symbol`` field.

    The returned dict is suitable for consumption by :func:`from_bse_json`.
    """
    center_data: dict = data.get("center_data", {})
    bse_elements: Dict[str, dict] = {}

    for _key, center in center_data.items():
        if not isinstance(center, dict):
            continue
        symbol = center.get("element_symbol") or center.get("element", "")
        if not symbol:
            continue
        shells = center.get("electron_shells") or center.get("atomic_shells", [])

        # QCSchema shells already have float exponents/coefficients, but we
        # convert to BSE-style string arrays so the common path in _parse_shells
        # can handle them transparently.
        converted_shells: list[dict] = []
        for sh in shells:
            if not isinstance(sh, dict):
                continue
            # QCSchema shell: angular_momentum, harmonic_type, exponents (float),
            # coefficients (list[list[float]])
            exponents = sh.get("exponents", [])
            coefficients = sh.get("coefficients", [[]])

            # Convert floats -> strings for uniform downstream handling.
            exponents_str = [str(e) for e in exponents]
            coefficients_str = [[str(c) for c in row] for row in coefficients]
            converted_shells.append(
                {
                    "angular_momentum": [
                        int(am) for am in sh.get("angular_momentum", [0])
                    ],
                    "function_type": "gto",
                    "harmonic_type": sh.get("harmonic_type", "spherical"),
                    "exponents": exponents_str,
                    "coefficients": coefficients_str,
                }
            )

        bse_elements[symbol] = {"electron_shells": converted_shells}

    return {
        "name": data.get("name", "unknown"),
        "description": data.get("description", ""),
        "basis_set_role": data.get("basis_set_role", data.get("role", "orbital")),
        "basis_set_family": data.get("basis_set_family") or data.get("basis_family"),
        "version": data.get("version"),
        "elements": bse_elements,
        "references": data.get("references", []),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atom_key_to_symbol(atom_key: str) -> str:
    """Map a BSE element key (atomic-number string, ``"1"``, ``"6"``) to a symbol.

    If *atom_key* is already a recognised element symbol (e.g. ``"H"``),
    it is returned unchanged.
    """
    from ._element_data import VALID_SYMBOLS

    if atom_key in VALID_SYMBOLS:
        return atom_key
    try:
        z = int(atom_key)
    except (TypeError, ValueError):
        raise ValueError(
            f"Cannot interpret element key {atom_key!r} as an atomic number or symbol."
        )
    symbol = _ATOMIC_NUMBER_TO_SYMBOL.get(z)
    if symbol is None:
        raise ValueError(f"No element symbol for atomic number {z} (key {atom_key!r}).")
    return symbol


def _parse_shells(
    atom_data: dict,
    *,
    harmonic_type: str,
) -> List[Shell]:
    """Parse ``electron_shells`` or ``atomic_shells`` from a BSE element block.

    Returns a list of :class:`Shell` objects.
    """
    raw_shells: list[dict] = atom_data.get(
        "electron_shells", atom_data.get("atomic_shells", [])
    )
    shells: List[Shell] = []

    for sh in raw_shells:
        ft = sh.get("function_type", "gto")
        if ft != "gto":
            raise ValueError(
                f"Unsupported function_type {ft!r}; only 'gto' is supported."
            )

        angular_momentum: List[int] = [int(am) for am in sh["angular_momentum"]]
        ht = sh.get("harmonic_type", harmonic_type)

        exponents: List[float] = [float(e) for e in sh["exponents"]]
        coefficients: List[List[float]] = [
            [float(c) for c in row] for row in sh["coefficients"]
        ]

        n_prim = len(exponents)
        _validate_coefficient_shape(sh, n_prim, len(coefficients), angular_momentum)

        # Build primitives.  For a pure shell (len(angular_momentum)==1) there
        # is one coefficient row; for an SP shell (len==2) there are two rows.
        # The model stores SP shells in interleaved order: first N primitives
        # carry the s coefficients, next N carry the p coefficients.
        primitives: List[Primitive] = []
        for coeff_row in coefficients:
            for exp, coeff in zip(exponents, coeff_row):
                primitives.append(Primitive(exponent=exp, coefficient=coeff))

        shells.append(
            Shell(
                angular_momentum=angular_momentum,
                harmonic_type=HarmonicType(ht),
                primitives=primitives,
            )
        )

    return shells


def _validate_coefficient_shape(
    shell: dict,
    n_prim: int,
    n_coeff_rows: int,
    angular_momentum: List[int],
) -> None:
    """Raise ``ValueError`` if coefficient rows don't match the shell shape."""
    expected_rows = len(angular_momentum)  # 1 for pure, 2 for SP
    if n_coeff_rows != expected_rows:
        raise ValueError(
            f"Shell has angular_momentum={angular_momentum} (expecting "
            f"{expected_rows} coefficient row(s)) but got {n_coeff_rows} "
            f"coefficient rows.  Shell data: {shell}"
        )
    for i, row in enumerate(shell.get("coefficients", [])):
        if len(row) != n_prim:
            raise ValueError(
                f"Coefficient row {i} has {len(row)} entries but "
                f"{n_prim} exponents -- must match.  Shell data: {shell}"
            )
