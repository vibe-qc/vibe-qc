"""AIMP-to-ECP bridge: convert parsed AIMP entries into libecpint-compatible
XML so they can be fed to ``compute_ecp_matrix``.

The AIMP local potential (M1 block) is a sum of Gaussians in ECP
convention: ``V(r) = sum_i c_i * r^{n_i-2} * exp(-x_i * r^2)`` with
``n_i = 2``.  This maps directly to libecpint's ``<nxc n="2" x="..."
c="..."/>`` format.

Licensing: the generated XML is created at runtime from user-provided
AIMP data (read via ``parse_aimp_file``).  No AIMP data is bundled
with vibe-qc (MPL 2.0 vs LGPL 2.1 — see CLAUDE.md § 1).
"""

from __future__ import annotations

import os
import tempfile
from xml.etree import ElementTree as ET

from ._aimp import AIMPEntry


def _ncore_from_entry(entry: AIMPEntry, Z: int) -> int:
    """Determine ncore (number of core electrons) for an AIMP entry.

    For bare (0s.0s) entries, the entire atom is a frozen core:
    ncore = Z.  For valence entries, ncore = Z - n_valence, where
    n_valence is estimated from the basis label.
    """
    if not entry.has_valence():
        return Z
    # Estimate valence electrons from basis label like "10s4p" or "6s3p".
    # Each s holds 2, each p holds 6 electrons nominally.
    n_val = 0
    label = entry.basis_label
    import re

    s_match = re.search(r"(\d+)s", label)
    if s_match:
        n_val += int(s_match.group(1)) * 2
    p_match = re.search(r"(\d+)p", label)
    if p_match:
        n_val += int(p_match.group(1)) * 6
    d_match = re.search(r"(\d+)d", label)
    if d_match:
        n_val += int(d_match.group(1)) * 10
    # Clamp: ncore can't exceed Z or be negative.
    return max(0, min(Z, Z - n_val))


def _aimp_entry_to_xml_element(entry: AIMPEntry, Z: int) -> ET.Element:
    """Convert a single AIMP entry to a libecpint XML element.

    Uses the M1 local potential block.  The PROJOP projector data is
    not yet mapped (future work for full orthogonality constraints).
    """
    ncore = _ncore_from_entry(entry, Z)
    # AIMP M1 is a local-only (no l-dependence) potential.
    # We treat it as a single l=0 shell with maxl=0.
    elem = ET.Element(entry.element.lower(), ncore=str(ncore), maxl="0")
    shell = ET.SubElement(elem, "Shell", lval="0", nexp=str(entry.m1_n_terms))
    for x, c in zip(entry.m1_exponents, entry.m1_coefficients):
        # AIMP convention: A(AIMP) = -Zeff * A(ECP).
        # Convert to standard ECP convention by negating.
        # libecpint expects leading space in n attribute (e.g. n=" 2").
        ET.SubElement(shell, "nxc", n=" 2", x=f"{x:.10f}", c=f"{-c:.10f}")
    return elem


def build_aimp_ecp_xml(
    entries: list[AIMPEntry],
    element_Z_map: dict[str, int],
) -> str:
    """Build a complete libecpint-compatible XML string from AIMP entries.

    Parameters
    ----------
    entries : list[AIMPEntry]
        Parsed AIMP entries (one per element, typically bare 0s.0s).
    element_Z_map : dict[str, int]
        Map element symbol → atomic number, e.g. ``{"Mg": 12, "O": 8}``.

    Returns
    -------
    str
        Complete XML ready to write to a ``.xml`` file.
    """
    root = ET.Element("root", name="aimp")
    for entry in entries:
        Z = element_Z_map.get(entry.element)
        if Z is None:
            continue
        if entry.m1_n_terms == 0:
            continue  # skip entries without local potential
        child = _aimp_entry_to_xml_element(entry, Z)
        root.append(child)
    return ET.tostring(root, encoding="unicode")


def write_aimp_ecp_library(
    entries: list[AIMPEntry],
    element_Z_map: dict[str, int],
    *,
    work_dir: str | None = None,
) -> tuple[str, str]:
    """Write AIMP ECP data as a temporary libecpint XML library.

    Returns ``(library_name, share_dir)`` suitable for passing to
    ``compute_ecp_matrix(library_name=..., share_dir=...)``.

    Parameters
    ----------
    entries : list[AIMPEntry]
        Parsed AIMP entries.
    element_Z_map : dict[str, int]
        Element → Z map.
    work_dir : str, optional
        Directory for the temporary library.  Defaults to a system
        temp directory.

    Returns
    -------
    (library_name, share_dir)
        ``library_name`` is the name without ``.xml`` extension;
        ``share_dir`` is the parent directory containing the XML file.
    """
    xml_content = build_aimp_ecp_xml(entries, element_Z_map)
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="vibeqc_aimp_")
    xml_subdir = os.path.join(work_dir, "xml")
    os.makedirs(xml_subdir, exist_ok=True)
    lib_name = "aimp_ecp"
    xml_path = os.path.join(xml_subdir, f"{lib_name}.xml")
    with open(xml_path, "w") as f:
        f.write(xml_content)
    return lib_name, work_dir


# --------------------------------------------------------------------------- #
# Convenience: resolve AIMP entries for a crystal.
# --------------------------------------------------------------------------- #

# Standard element symbol → Z lookup.
_ELEMENT_Z: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
}
