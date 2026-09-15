"""Surface / slab naming and adsorbed-molecule recognition.

Handles periodic 2D systems (slabs) and structures containing a
surface with adsorbed molecules. Detects disconnected components
of the molecular graph to separate the slab from adsorbates.

Supports:
- Slab material identification (stoichiometric + mineral names)
- Miller index detection from lattice vectors
- Adsorbate molecule naming (delegates to organic/inorganic engines)
- Combined names: "benzene on Al2O3(0001)"

References
----------
- IUPAC Red Book 2005, IR-11 (solids nomenclature)
- Strukturbericht designations for common surface structures
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from ._types import Confidence, NamedResult, NamingSource
from .molecular_graph import ATOMIC_NUMBER_TO_SYMBOL, MolecularGraph

# ── Mineral / trivial surface names ────────────────────────────────────────

# Known surface compounds with common names.
# Key: (canonical Hill formula or stoichiometric ratio)
_SURFACE_MATERIAL_NAMES: dict[str, str] = {
    "Al2O3": "alumina",
    "SiO2": "silica",
    "TiO2": "titania",
    "MgO": "magnesia",
    "CaO": "lime",
    "ZnO": "zinc oxide",
    "Fe2O3": "iron oxide",
    "Fe3O4": "magnetite",
    "Cu2O": "cuprous oxide",
    "CeO2": "ceria",
    "ZrO2": "zirconia",
    "HfO2": "hafnia",
    "NiO": "nickel oxide",
    "CoO": "cobalt oxide",
    "MnO": "manganese oxide",
    "MnO2": "manganese dioxide",
    "V2O5": "vanadium pentoxide",
    "Cr2O3": "chromia",
    "MoS2": "molybdenum disulfide",
    "WS2": "tungsten disulfide",
    "BN": "boron nitride",
    "SiC": "silicon carbide",
    "GaN": "gallium nitride",
    "AlN": "aluminium nitride",
    "InP": "indium phosphide",
    "GaAs": "gallium arsenide",
    "CdTe": "cadmium telluride",
    "InN": "indium nitride",
    "InAs": "indium arsenide",
    "GaP": "gallium phosphide",
    "CdS": "cadmium sulfide",
    "CdSe": "cadmium selenide",
    "PbS": "lead sulfide",
    "PbSe": "lead selenide",
    "PbTe": "lead telluride",
    "C": "diamond",
    "Si": "silicon",
    "Ge": "germanium",
    "Cu": "copper",
    "Fe": "iron",
    "W": "tungsten",
    "Au": "gold",
    "Pt": "platinum",
    "Ni": "nickel",
    "Ag": "silver",
    "Al": "aluminium",
    "SrTiO3": "strontium titanate",
    "BaTiO3": "barium titanate",
    "CaTiO3": "calcium titanate",
    "LiNbO3": "lithium niobate",
    "YBa2Cu3O7": "YBCO",
    "La2CuO4": "lanthanum cuprate",
    "NaCl": "rock salt",
    "KCl": "potassium chloride",
    "LiF": "lithium fluoride",
    "CaF2": "fluorite",
    "MgF2": "magnesium fluoride",
}


# ── Miller index naming ────────────────────────────────────────────────────

# Simulated reference lattice parameters for common materials
# (a, c in angstrom). Used only when lattice vectors are provided.
_REFERENCE_LATTICES: dict[str, tuple[float, float, str]] = {
    # formula -> (a, c, crystal_system)
    "Al2O3": (4.76, 12.99, "hexagonal"),
    "SiO2": (4.91, 5.40, "hexagonal"),  # α-quartz
    "TiO2": (3.78, 9.51, "tetragonal"),  # anatase
    "MgO": (4.21, None, "cubic"),
    "CaO": (4.81, None, "cubic"),
    "ZnO": (3.25, 5.21, "hexagonal"),
    "SrTiO3": (3.91, None, "cubic"),
    "MoS2": (3.16, 12.30, "hexagonal"),
    "GaN": (3.19, 5.19, "hexagonal"),
}


def _detect_miller_index(
    lattice_vectors: list[tuple[float, float, float]],
    formula: str,
) -> Optional[str]:
    """Detect the likely Miller index of a slab from lattice geometry.

    When lattice vectors are available, compares the in-plane lattice
    constants against reference values to guess the surface orientation.
    Returns e.g. ``"(0001)"`` or ``"(001)"``.
    """
    if (
        not lattice_vectors
        or len(lattice_vectors) < 3
        or formula not in _REFERENCE_LATTICES
    ):
        return None

    ref_a, ref_c, system = _REFERENCE_LATTICES[formula]

    # Measure in-plane lattice vectors (first two)
    a1 = math.sqrt(
        lattice_vectors[0][0] ** 2
        + lattice_vectors[0][1] ** 2
        + lattice_vectors[0][2] ** 2
    )
    a2 = math.sqrt(
        lattice_vectors[1][0] ** 2
        + lattice_vectors[1][1] ** 2
        + lattice_vectors[1][2] ** 2
    )
    avg_a = (a1 + a2) / 2
    a3 = math.sqrt(
        lattice_vectors[2][0] ** 2
        + lattice_vectors[2][1] ** 2
        + lattice_vectors[2][2] ** 2
    )

    # Check if this looks like a basal-plane slab (a matches reference, c is very large = vacuum)
    vector_ratio = a3 / max(avg_a, 0.01)

    if system in ("hexagonal", "trigonal"):
        # For hexagonal: (0001) is the basal plane
        if abs(avg_a - ref_a) / ref_a < 0.15:
            if vector_ratio > 2.0:
                return "(0001)"  # vacuum in c-direction → basal plane
            else:
                return "(0001)"  # likely basal plane anyway
        else:
            # Orthogonal plane — check ratio
            if ref_c and abs(avg_a - ref_c) / ref_c < 0.20:
                return "(1\u03041\u03040)"  # (1-100) family
            return None

    elif system == "cubic":
        # Cubic: (001) or (111) are common
        ratio_a_ref = avg_a / max(ref_a, 0.01)
        if 0.85 < ratio_a_ref < 1.15:
            return "(001)"
        if 1.38 < ratio_a_ref < 1.55 or 0.65 < ratio_a_ref < 0.75:
            return "(111)"
        return "(001)"  # default

    elif system == "tetragonal":
        if abs(avg_a - ref_a) / ref_a < 0.15:
            return "(001)" if vector_ratio > 2.0 else "(100)"
        return None

    return None


# ── Slab detection ─────────────────────────────────────────────────────────


@dataclass
class SlabSystem:
    """A surface/slab with optional adsorbate molecules.

    Attributes
    ----------
    slab_atoms : list[int]
        Atom indices belonging to the slab.
    adsorbate_components : list[list[int]]
        Each sublist is a disconnected molecular fragment resting on
        or hovering above the surface.
    slab_material : str
        Common name for the surface material (e.g. "alumina").
    slab_formula : str
        Stoichiometric formula of the surface alone.
    miller_index : str or None
        Detected Miller index, e.g. ``"(0001)"``.
    is_slab : bool
        Whether this structure was detected as a 2D periodic slab.
    """

    slab_atoms: list[int]
    adsorbate_components: list[list[int]] = field(default_factory=list)
    slab_material: str = ""
    slab_formula: str = ""
    miller_index: Optional[str] = None
    is_slab: bool = False


def detect_slab(
    graph: MolecularGraph,
    lattice_vectors: Optional[list[tuple[float, float, float]]] = None,
) -> SlabSystem:
    """Detect whether a molecular graph represents a slab/surface system.

    Uses connected-components analysis to separate the surface from any
    adsorbed or gas-phase molecules.

    Parameters
    ----------
    graph :
        MolecularGraph built from coordinates.
    lattice_vectors :
        Optional lattice vectors [(ax, ay, az), (bx, by, bz), (cx, cy, cz)]
        in Angstrom. When provided, Miller index detection and enhanced
        slab confidence are enabled.

    Returns
    -------
    SlabSystem with slab/adsorbate partitioning and material identification.
    """
    components = graph.connected_components()

    if len(components) == 1:
        # Single component — could be a clean slab or a molecule
        comp = components[0]
        n_atoms = len(comp)
        metals = sum(1 for i in comp if _is_metal(graph.atoms[i].z))
        non_metals = sum(
            1 for i in comp if not _is_metal(graph.atoms[i].z) and graph.atoms[i].z > 2
        )

        if lattice_vectors and metals > 0 and n_atoms >= 8:
            # Looks like a bare slab
            formula = _slab_formula(graph, comp)
            material = _identify_material(formula)
            miller = _detect_miller_index(lattice_vectors, formula)
            return SlabSystem(
                slab_atoms=comp,
                slab_material=material,
                slab_formula=formula,
                miller_index=miller,
                is_slab=True,
            )
        else:
            return SlabSystem(slab_atoms=[], is_slab=False)

    # Multiple components: largest is likely the slab, rest are adsorbates
    largest = components[0]
    adsorbates = components[1:]

    # Heuristic: slab if largest component has >50 atoms OR has metals
    n_largest = len(largest)
    metals_in_largest = sum(1 for i in largest if _is_metal(graph.atoms[i].z))
    is_slab = (n_largest >= 100) or (metals_in_largest > 0 and n_largest >= 30)

    if not is_slab:
        return SlabSystem(slab_atoms=[], is_slab=False)

    formula = _slab_formula(graph, largest)
    material = _identify_material(formula)
    miller = _detect_miller_index(lattice_vectors, formula) if lattice_vectors else None

    return SlabSystem(
        slab_atoms=largest,
        adsorbate_components=adsorbates,
        slab_material=material,
        slab_formula=formula,
        miller_index=miller,
        is_slab=True,
    )


def name_slab_system(
    graph: MolecularGraph,
    lattice_vectors: Optional[list[tuple[float, float, float]]] = None,
    *,
    fast_mode: bool = True,
) -> NamedResult:
    """Generate a human-readable name for a slab + adsorbate system.

    Returns names like:
    - "alumina(0001) slab"
    - "benzene on alumina(0001)"
    - "water on TiO2(001)"
    - "CO on MgO(001)"

    Parameters
    ----------
    graph :
        MolecularGraph built from coordinates.
    lattice_vectors :
        Optional lattice vectors (Angstrom). Improves Miller index detection.
    fast_mode :
        When True, uses formula-based naming for adsorbates (fast).
        When False, runs the full naming pipeline per adsorbate.

    Returns
    -------
    NamedResult with the combined name.
    """
    slab = detect_slab(graph, lattice_vectors)

    if not slab.is_slab:
        return NamedResult(
            name="",
            source=NamingSource.INORGANIC_COMPOSITIONAL,
            confidence=Confidence.LOW,
            is_trivial=False,
        )

    surface_part = slab.slab_material
    if slab.miller_index:
        surface_part += slab.miller_index

    if not slab.adsorbate_components:
        # Bare slab
        return NamedResult(
            name=f"{surface_part} slab",
            source=NamingSource.INORGANIC_COMPOSITIONAL,
            confidence=Confidence.HIGH,
            formula=slab.slab_formula,
            is_trivial=False,
        )

    # Name each adsorbate
    adsorbate_names: list[str] = []
    for indices in slab.adsorbate_components:
        sub = graph.subgraph(indices)
        if sub.n_atoms == 0:
            continue
        # Try formula-based naming first
        mol_name = sub.formula
        if mol_name in (
            "H2O",
            "CO",
            "CO2",
            "NH3",
            "CH4",
            "H2",
            "O2",
            "N2",
            "NO",
            "NO2",
        ):
            # Common small molecules — use common names
            mol_name = {
                "H2O": "water",
                "CO": "CO",
                "CO2": "CO2",
                "NH3": "ammonia",
                "CH4": "methane",
                "H2": "H2",
                "O2": "O2",
                "N2": "N2",
                "NO": "NO",
                "NO2": "NO2",
            }.get(mol_name, mol_name)
        elif not fast_mode:
            # Run the full naming pipeline
            from .iupac_name import _name_from_graph

            result = _name_from_graph(sub, prefer_trivial=True, with_external=False)
            mol_name = result.name

        adsorbate_names.append(mol_name)

    if not adsorbate_names:
        return NamedResult(
            name=f"{surface_part} slab",
            source=NamingSource.INORGANIC_COMPOSITIONAL,
            confidence=Confidence.MEDIUM,
            formula=slab.slab_formula,
            is_trivial=False,
        )

    # Deduplicate and count
    name_counts: dict[str, int] = {}
    for n in adsorbate_names:
        name_counts[n] = name_counts.get(n, 0) + 1

    parts: list[str] = []
    for name, count in sorted(name_counts.items()):
        if count == 1:
            parts.append(name)
        else:
            parts.append(f"{count} × {name}")

    adsorbate_str = " + ".join(parts)

    return NamedResult(
        name=f"{adsorbate_str} on {surface_part}",
        source=NamingSource.INORGANIC_COMPOSITIONAL,
        confidence=Confidence.HIGH,
        formula=slab.slab_formula,
        is_trivial=False,
    )


# ── Helpers ─────────────────────────────────────────────────────────────────


def _is_metal(z: int) -> bool:
    """Heuristic: is this atomic number a metal?"""
    metal_ranges = [
        (3, 4),  # Li, Be
        (11, 13),  # Na, Mg, Al
        (19, 31),  # K-Ga
        (37, 50),  # Rb-Sn
        (55, 83),  # Cs-Bi
    ]
    # Explicitly non-metal
    non_metals = {
        1,
        2,
        5,
        6,
        7,
        8,
        9,
        10,
        14,
        15,
        16,
        17,
        18,
        33,
        34,
        35,
        36,
        52,
        53,
        54,
    }
    return (z not in non_metals) and any(lo <= z <= hi for (lo, hi) in metal_ranges)


def _slab_formula(graph: MolecularGraph, indices: list[int]) -> str:
    """Hill formula for the slab component."""
    counts: dict[str, int] = {}
    for i in indices:
        s = graph.atoms[i].symbol
        counts[s] = counts.get(s, 0) + 1

    parts: list[str] = []
    if "C" in counts:
        c = counts.pop("C")
        parts.append(f"C{c}" if c > 1 else "C")
    if "H" in counts:
        h = counts.pop("H")
        parts.append(f"H{h}" if h > 1 else "H")
    for sym in sorted(counts):
        cnt = counts[sym]
        parts.append(f"{sym}{cnt}" if cnt > 1 else sym)
    return "".join(parts)


def _identify_material(formula: str) -> str:
    """Return a human-readable name for a surface material."""
    # Try reduced formula first (e.g. Mg16O16 -> MgO)
    reduced = _reduce_formula(formula)
    if reduced in _SURFACE_MATERIAL_NAMES:
        return _SURFACE_MATERIAL_NAMES[reduced]
    if formula in _SURFACE_MATERIAL_NAMES:
        return _SURFACE_MATERIAL_NAMES[formula]

    # Try compositional naming fallback
    from .iupac_name import _compositional_name_from_formula
def _gcd(a: int, b: int) -> int:
    """Greatest common divisor."""
    while b:
        a, b = b, a % b
    return abs(a)


def _reduce_formula(formula: str) -> str:
    """Reduce a formula to its simplest integer ratio via GCD.
    e.g., Mg16O16 -> MgO, Al24O36 -> Al2O3."""
    counts = {}
    for sym, cnt in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
        counts[sym] = counts.get(sym, 0) + (int(cnt) if cnt else 1)
    if not counts:
        return formula
    g = math.gcd(*counts.values())
    if g <= 1:
        return formula
    parts = []
    if 'C' in counts:
        parts.append(f"C{counts['C']//g}" if counts['C']//g > 1 else 'C')
        del counts['C']
    for sym in sorted(counts):
        c = counts[sym] // g
        parts.append(f"{sym}{c}" if c > 1 else sym)
    return ''.join(parts)

    return _compositional_name_from_formula(formula)
