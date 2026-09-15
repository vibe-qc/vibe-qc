"""HTML parsers for NIST CCCBDB pages.

Single entry point for v1: ``parse_exp2x_html(html)`` consumes the
output of ``GET https://cccbdb.nist.gov/exp2x.asp?casno=<7-digit-CAS>``
-- the per-molecule "Experimental Data" page -- and returns a
property dictionary suitable for :class:`ExperimentalReference`.

CCCBDB ``exp2x.asp`` pages are stable, structured HTML; ~95% of
table-cell ordering has been the same since at least Release 18
(2019). We re-find tables by their first-row column headers rather
than by index, so a future re-ordering of tables on the page
doesn't silently miscategorise data.

Units returned by the parser match CCCBDB's tabulated units
(kJ/mol for thermochem, eV for IE, cm⁻¹ for vibrationals,
Debye for dipoles, atomic units for polarizability). The caller
(``ExperimentalReference`` builder) carries them across into the
schema fields without further conversion.

Tested against ``tests/data/cccbdb/h2o_exp2.html`` -- see
``tests/test_fetch_references_cccbdb.py``.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from ._optional_deps import require_bs4

if TYPE_CHECKING:
    # ``BeautifulSoup`` / ``Tag`` are used only in annotations (strings
    # under ``from __future__ import annotations``); bs4 itself is
    # lazy-imported via ``require_bs4()`` at the parse callsite so
    # importing this module does not require the optional [fetch] extra.
    from bs4 import BeautifulSoup, Tag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(s: str) -> Optional[float]:
    """Parse a CCCBDB numeric cell.

    CCCBDB sometimes uses comma thousands separators ("1,234.5") and
    occasionally non-breaking spaces / unicode minus signs. Handle
    those, return None on un-parseable cells (empty / "--" / "n/a").
    """
    if s is None:
        return None
    s = s.strip().replace(",", "").replace(" ", " ").replace("-", "-")
    if not s or s in ("--", "-", "n/a", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _row_cells(tr: Tag) -> list[str]:
    return [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]


def _strip_navigation(soup: BeautifulSoup) -> None:
    """Remove the site-wide navigation table (every CCCBDB page ships
    with one ~4-row ``table.noborder`` that contains no chemistry data
    but inflates table counts). Modifies ``soup`` in place."""
    for nav in soup.find_all("table", class_="noborder"):
        nav.decompose()


def _find_table_by_header(
    soup: BeautifulSoup, *, contains: list[str], min_rows: int = 2,
) -> Optional[Tag]:
    """Find the first ``<table>`` whose first row contains all the
    given header strings (case-insensitive substring match).

    Resilient to CCCBDB's column-reordering -- better than indexing
    tables by position.
    """
    needles = [s.lower() for s in contains]
    for t in soup.find_all("table"):
        rows = t.find_all("tr")
        if len(rows) < min_rows:
            continue
        header_text = rows[0].get_text(" | ", strip=True).lower()
        if all(n in header_text for n in needles):
            return t
    return None


# ---------------------------------------------------------------------------
# Per-table parsers
# ---------------------------------------------------------------------------

def _parse_thermochem_table(soup: BeautifulSoup) -> dict[str, Optional[float]]:
    """Table[4] in the H2O page -- "Property | Value | Uncertainty | units ...".

    Returns a dict with these keys when present:
      enthalpy_of_formation_298_kj_per_mol
      enthalpy_of_formation_298_uncertainty_kj_per_mol
      enthalpy_of_formation_0_kj_per_mol
      enthalpy_of_formation_0_uncertainty_kj_per_mol
      entropy_298_j_per_mol_per_k
      heat_capacity_298_j_per_mol_per_k
    """
    out: dict[str, Optional[float]] = {}
    table = _find_table_by_header(
        soup, contains=["property", "value", "uncertainty", "units"],
    )
    if table is None:
        return out
    for tr in table.find_all("tr")[1:]:
        cells = _row_cells(tr)
        if len(cells) < 4:
            continue
        prop, value, unc, units = cells[0], cells[1], cells[2], cells[3]
        v = _safe_float(value)
        u = _safe_float(unc)
        prop_lower = prop.lower().replace(" ", "")
        if prop_lower.startswith("hfg(298"):
            out["enthalpy_of_formation_298_kj_per_mol"] = v
            out["enthalpy_of_formation_298_uncertainty_kj_per_mol"] = u
        elif prop_lower.startswith("hfg(0"):
            out["enthalpy_of_formation_0_kj_per_mol"] = v
            out["enthalpy_of_formation_0_uncertainty_kj_per_mol"] = u
        elif "entropy" in prop_lower:
            out["entropy_298_j_per_mol_per_k"] = v
        elif "heatcapacity" in prop_lower:
            out["heat_capacity_298_j_per_mol_per_k"] = v
    return out


def _parse_vibrational_table(
    soup: BeautifulSoup,
) -> dict[str, tuple[float, ...]]:
    """Vibrational modes -- table[5] in the H2O page.

    Header: "Mode Number | Symmetry | Frequency | Intensity | ...".
    Sub-header second row: "Fundamental (cm-1) | Harmonic (cm-1) |
    Reference | (km mol-1)" -- interleaves columns weirdly. We pull
    fundamental + harmonic + IR intensity by column position
    after skipping the two-row header.
    """
    out: dict[str, tuple[float, ...]] = {
        "vibrational_fundamentals_cm_inv": (),
        "vibrational_harmonics_cm_inv": (),
        "ir_intensities_km_per_mol": (),
    }
    table = _find_table_by_header(
        soup,
        contains=["mode", "symmetry", "frequency", "intensity"],
    )
    if table is None:
        return out
    rows = table.find_all("tr")
    if len(rows) < 3:
        return out
    # Two-row header (CCCBDB convention); data starts at row[2].
    fundamentals: list[float] = []
    harmonics: list[float] = []
    ir_intensities: list[float] = []
    for tr in rows[2:]:
        cells = _row_cells(tr)
        # Expected layout (10+ cells):
        # [mode, sym, fundamental, harmonic, ref-name, intensity, ref, ...]
        if len(cells) < 4:
            continue
        # cells[0]=mode_number, cells[1]=symmetry,
        # cells[2]=fundamental, cells[3]=harmonic, cells[5]=intensity.
        f_fund = _safe_float(cells[2]) if len(cells) > 2 else None
        f_harm = _safe_float(cells[3]) if len(cells) > 3 else None
        i_int = _safe_float(cells[5]) if len(cells) > 5 else None
        if f_fund is not None:
            fundamentals.append(f_fund)
        if f_harm is not None:
            harmonics.append(f_harm)
        if i_int is not None:
            ir_intensities.append(i_int)
    out["vibrational_fundamentals_cm_inv"] = tuple(sorted(fundamentals))
    out["vibrational_harmonics_cm_inv"] = tuple(sorted(harmonics))
    # IR intensities follow the *fundamental* mode order on the page.
    # We don't sort them -- caller can reorder by zipping with fundamentals
    # if needed.
    out["ir_intensities_km_per_mol"] = tuple(ir_intensities)
    return out


def _parse_ionization_energy_table(
    soup: BeautifulSoup,
) -> dict[str, Optional[float]]:
    """Table["Ionization Energy | I.E. unc. | vertical I.E. | ..."]."""
    out: dict[str, Optional[float]] = {}
    table = _find_table_by_header(
        soup, contains=["ionization energy", "vertical i.e."],
    )
    if table is None:
        return out
    rows = table.find_all("tr")
    if len(rows) < 2:
        return out
    cells = _row_cells(rows[1])
    if len(cells) >= 2:
        out["ionization_energy_ev"] = _safe_float(cells[0])
        out["ionization_energy_uncertainty_ev"] = _safe_float(cells[1])
    return out


def _parse_proton_affinity_table(
    soup: BeautifulSoup,
) -> dict[str, Optional[float]]:
    """Table["Proton Affinity | unc. | Product | ..."]."""
    out: dict[str, Optional[float]] = {}
    table = _find_table_by_header(
        soup, contains=["proton affinity", "product"],
    )
    if table is None:
        return out
    rows = table.find_all("tr")
    if len(rows) < 2:
        return out
    cells = _row_cells(rows[1])
    if cells:
        out["proton_affinity_kj_per_mol"] = _safe_float(cells[0])
    return out


def _parse_dipole_table(soup: BeautifulSoup) -> dict[str, Optional[float]]:
    """Total dipole moment from the "Dipole (Debye)" table.

    CCCBDB lays this out as a wide table with State / Config / x / y /
    z / total columns under a "Dipole (Debye)" parent header. We look
    for "dipole (debye)" in the header row, then pick the "total"
    column from the data row.
    """
    out: dict[str, Optional[float]] = {}
    table = _find_table_by_header(
        soup, contains=["dipole (debye)"],
    )
    if table is None:
        return out
    rows = table.find_all("tr")
    if len(rows) < 3:
        return out
    # Sub-header (row 1) declares column meanings: x | y | z | total.
    sub_header = _row_cells(rows[1])
    try:
        total_col_in_sub = sub_header.index("total")
    except ValueError:
        return out
    # Data rows start after the two-row header. CCCBDB sometimes inserts
    # an empty <tr> after the sub-header (seen in the H2O page); skip it.
    for tr in rows[2:]:
        cells = _row_cells(tr)
        if not cells:
            continue
        # Heuristic: the first numeric cell at-or-after total_col_in_sub
        # is the total dipole. (Leading State/Config text columns are
        # non-numeric and get skipped naturally.)
        if len(cells) > total_col_in_sub:
            for c in cells[total_col_in_sub:]:
                v = _safe_float(c)
                if v is not None:
                    out["dipole_moment_debye"] = v
                    return out
    return out


def _parse_polarizability_table(
    soup: BeautifulSoup,
) -> dict[str, Optional[float]]:
    """Table["alpha | unc. | Reference"] -- isotropic polarizability."""
    out: dict[str, Optional[float]] = {}
    table = _find_table_by_header(
        soup, contains=["alpha", "unc.", "reference"],
    )
    if table is None:
        return out
    rows = table.find_all("tr")
    if len(rows) < 2:
        return out
    cells = _row_cells(rows[1])
    if cells:
        out["polarizability_au"] = _safe_float(cells[0])
    return out


def _parse_cartesian_geometry_table(
    soup: BeautifulSoup,
) -> tuple[tuple[tuple[str, int, float, float, float], ...], dict]:
    """Table["Atom | x (Å) | y (Å) | z (Å)"] -- explicit Cartesians.

    Returns a tuple of ``(label, z, x, y, z)`` per atom. Atom labels in
    CCCBDB carry a trailing position index (``"O1"``, ``"H2"``, ...);
    this function returns the raw label so downstream consumers can
    preserve the per-page atom numbering, alongside the inferred
    atomic number.
    """
    table = _find_table_by_header(
        soup, contains=["atom", "x (å)", "y (å)", "z (å)"], min_rows=2,
    )
    cart: list[tuple[str, int, float, float, float]] = []
    diag: dict = {}
    if table is None:
        return tuple(cart), diag
    for tr in table.find_all("tr")[1:]:
        cells = _row_cells(tr)
        if len(cells) < 4:
            continue
        label = cells[0]
        x = _safe_float(cells[1])
        y = _safe_float(cells[2])
        z = _safe_float(cells[3])
        if x is None or y is None or z is None:
            continue
        # Strip trailing digits to get the element symbol -- "O1" -> "O",
        # "Cl4" -> "Cl". Two-letter elements (Cl/Br) work because the
        # symbol is at the start; digits are stripped from the end.
        sym = "".join(ch for ch in label if not ch.isdigit())
        z_atomic = _ATOMIC_NUMBER_BY_SYMBOL.get(sym)
        if z_atomic is None:
            # Don't drop the atom -- record it with z=0 so the caller
            # can decide how to react. CCCBDB occasionally uses unusual
            # labels (e.g. dummy atoms in linear molecules).
            z_atomic = 0
        cart.append((label, z_atomic, x, y, z))
    return tuple(cart), diag


# Local element-symbol -> Z table. Kept independent of
# client_optimade._atomic_number to avoid a cross-package import here
# (parsers.py is meant to be lightweight, no fetch-client deps).
_ATOMIC_NUMBER_BY_SYMBOL: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27,
    "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36,
}


def _parse_geometry_tables(
    soup: BeautifulSoup,
) -> dict[str, tuple]:
    """Bond lengths + angles from the "Description | Value | unc. | ..."
    and "atom1 | atom2 | atom3 | angle" tables. Plus Cartesian
    coordinates from the "Atom | x | y | z" table.
    """
    out: dict[str, tuple] = {
        "bond_lengths_ang": (),
        "bond_angles_deg": (),
        "cartesian_geometry_ang": (),
    }
    cart, _ = _parse_cartesian_geometry_table(soup)
    out["cartesian_geometry_ang"] = cart

    # -- Bond lengths via the explicit distance matrix --
    # (Easier than parsing the variable-width "Description | Value | ..."
    # table; the distance matrix has a clean atom_label -> length
    # mapping per Cartesian-aware row.)
    distance = _find_table_by_header(soup, contains=["o1", "h2"])
    if distance is None:
        # Fall back: find any 4-column table whose first row is 4 short
        # atom labels (e.g. "O1 | H2 | H3"). Best-effort.
        pass

    # -- Geometric "Description | Value | unc." table --
    desc_t = _find_table_by_header(
        soup, contains=["description", "value", "unc.", "connectivity"],
    )
    bond_lengths = []
    bond_angles = []
    if desc_t is not None:
        # First two rows are headers (CCCBDB-style). Each data row
        # starts with a label like "rOH" (length) or "aHOH" (angle).
        rows = desc_t.find_all("tr")
        for tr in rows[2:]:
            cells = _row_cells(tr)
            if len(cells) < 3:
                continue
            label = cells[0]
            value = _safe_float(cells[1])
            if value is None:
                continue
            if label.startswith("r"):
                # Length, e.g. "rOH" with value in Å.
                # Atom symbols extracted from label after stripping "r".
                pair = _atom_symbols_from_label(label[1:])
                if pair:
                    bond_lengths.append((pair[0], pair[1], value))
            elif label.startswith("a"):
                # Angle, e.g. "aHOH" -- symbols (1, 2, 3).
                triple = _atom_symbols_from_label(label[1:], n=3)
                if triple:
                    bond_angles.append((triple[0], triple[1], triple[2], value))
    out["bond_lengths_ang"] = tuple(bond_lengths)
    out["bond_angles_deg"] = tuple(bond_angles)
    return out


def _atom_symbols_from_label(label: str, n: int = 2) -> Optional[tuple[str, ...]]:
    """Split a CCCBDB internal-coord label like 'OH' / 'HOH' / 'COH' into
    n element symbols. Returns None if can't parse cleanly.

    CCCBDB labels concat atomic symbols without separators. For mixed
    upper/lower-case symbols (e.g. 'Cl', 'Br') the parser walks the
    label char-by-char and groups into 1- or 2-char element symbols.
    """
    syms: list[str] = []
    i = 0
    L = len(label)
    while i < L:
        if i + 1 < L and label[i + 1].islower():
            syms.append(label[i:i + 2])
            i += 2
        else:
            syms.append(label[i])
            i += 1
    if len(syms) == n:
        return tuple(syms)
    return None


# ---------------------------------------------------------------------------
# Public top-level parser
# ---------------------------------------------------------------------------

def parse_exp2x_html(html: str) -> dict:
    """Extract every property we know how to from a CCCBDB ``exp2x``
    page. Missing fields stay absent from the returned dict.

    Returned keys:
      formula                                 (from h1, e.g. "H2O")
      name                                    (from h1, e.g. "Water")
      enthalpy_of_formation_298_kj_per_mol
      enthalpy_of_formation_298_uncertainty_kj_per_mol
      enthalpy_of_formation_0_kj_per_mol
      enthalpy_of_formation_0_uncertainty_kj_per_mol
      entropy_298_j_per_mol_per_k
      heat_capacity_298_j_per_mol_per_k
      vibrational_fundamentals_cm_inv         (sorted ascending)
      vibrational_harmonics_cm_inv            (sorted ascending)
      ir_intensities_km_per_mol               (page order)
      ionization_energy_ev
      ionization_energy_uncertainty_ev
      proton_affinity_kj_per_mol
      dipole_moment_debye
      polarizability_au
      bond_lengths_ang                        (tuple of (sym1, sym2, length))
      bond_angles_deg                         (tuple of (sym1, sym2, sym3, angle))
    """
    BeautifulSoup = require_bs4()
    soup = BeautifulSoup(html, "lxml")

    # Detect server-error / 404 pages early -- CCCBDB returns 200 with an
    # empty navigation skeleton when the casno doesn't match a real entry.
    title = soup.title.string if soup.title else ""
    if any(s in (title or "") for s in ("500 -", "404 -", "Server Error")):
        raise CCCBDBNoData(f"CCCBDB returned an error page: {title!r}")

    # Identity from H1: "Experimental data for H2O(Water)" -- note the
    # subscripts inside <sub> tags. We use get_text() without a separator
    # to keep "H<sub>2</sub>O" -> "H2O" verbatim. ``get_text(" ")`` would
    # turn it into "H 2 O" and break formula matching.
    h1 = soup.find("h1")
    formula = ""
    name = ""
    if h1:
        h1_text = h1.get_text().strip()
        m = re.search(
            r"Experimental data for\s+([A-Za-z0-9+\-]+)\s*\(([^)]+)\)",
            h1_text,
        )
        if m:
            formula = m.group(1).strip()
            name = m.group(2).strip()

    if not formula:
        # Page exists but doesn't carry the expected master-table layout.
        raise CCCBDBNoData(
            "CCCBDB exp2x page lacks the 'Experimental data for X(Y)' "
            "header; likely the casno is wrong or the species has no "
            "experimental data sheet."
        )

    _strip_navigation(soup)

    out: dict = {"formula": formula, "name": name}
    out.update(_parse_thermochem_table(soup))
    out.update(_parse_vibrational_table(soup))
    out.update(_parse_ionization_energy_table(soup))
    out.update(_parse_proton_affinity_table(soup))
    out.update(_parse_dipole_table(soup))
    out.update(_parse_polarizability_table(soup))
    out.update(_parse_geometry_tables(soup))
    return out


class CCCBDBNoData(LookupError):
    """Raised when CCCBDB serves an error page or a malformed entry.

    Caller (``client_cccbdb.fetch_cccbdb``) should map this to a clean
    ``MoleculeNotFound``-style exit with the offending CAS in the
    message.
    """
