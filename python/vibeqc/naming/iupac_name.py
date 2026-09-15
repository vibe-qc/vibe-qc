"""IUPAC molecular naming from structure graphs.

Implements systematic naming rules from:
- IUPAC Red Book 2005 (inorganic nomenclature)
- IUPAC Blue Book 2013 (organic nomenclature)
- Favre & Powell, *Nomenclature of Organic Chemistry*, RSC 2014

Strategy
--------
1. Check against a built-in dictionary of common trivial names (water,
   methane, benzene, …) — these are universally accepted IUPAC-preferred
   retained names.
2. For inorganic / binary compounds, use compositional nomenclature
   with multiplying prefixes (Red Book IR-4, IR-5).
3. For coordination complexes, use additive nomenclature (Red Book
   IR-7, IR-9).
4. For organic molecules, build the systematic substitutive name:
   - identify the principal chain / ring
   - number substituents
   - assemble the name per Blue Book P-1–P-7
5. For complex cases, fall back to optional external backends (RDKit,
   PubChem, CACTUS).

Returns NamedResult with name, source, and confidence rather than bare
strings — enabling callers to decide whether to trust or display the
result alongside its provenance.

References
----------
- IUPAC Red Book 2005: *Nomenclature of Inorganic Chemistry*
- IUPAC Blue Book 2013: *Nomenclature of Organic Chemistry*
- Cordero et al., *Dalton Trans.* 2008, 2832 (covalent radii)
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

from .molecular_graph import (
    ATOMIC_NUMBER_TO_SYMBOL,
    MolecularGraph,
    build_graph,
    from_molecule,
)

# NOTE: ``from .organic import ...`` is deliberately NOT done here at module
# top. ``organic`` imports ``Confidence`` back from this module (see its
# bottom-of-file import), so importing it before ``Confidence`` is defined
# below deadlocks the circular dependency. The organic symbols are imported
# after ``Confidence`` is defined — see the ``from .organic import`` line
# further down.

# ── Result type ──────────────────────────────────────────────────────────────


class NamingSource(str, Enum):
    """Provenance of a naming result."""

    TRIVIAL_IUPAC = "trivial_iupac"
    ORGANIC_SYSTEMATIC = "organic_systematic"
    INORGANIC_COMPOSITIONAL = "inorganic_compositional"
    COORDINATION = "coordination_additive"
    EXTERNAL_RDKIT = "external_rdkit"
    EXTERNAL_CACTUS = "external_cactus"
    EXTERNAL_PUBCHEM = "external_pubchem"


class Confidence(str, Enum):
    """Confidence level for a naming result."""

    HIGH = "high"  # trivial name or confident systematic
    MEDIUM = "medium"  # systematic with some uncertainty
    LOW = "low"  # external backend or low-confidence guess


@dataclass(frozen=True)
class NamedResult:
    """Naming result with provenance and confidence.

    Attributes
    ----------
    name :
        The IUPAC name (or compositional name).
    source :
        Where the name came from.
    confidence :
        Confidence in the accuracy of this name.
    formula :
        Hill-system formula for reference.
    is_trivial :
        Whether this is a retained IUPAC trivial name.
    """

    name: str
    source: NamingSource
    confidence: Confidence
    formula: str = ""
    is_trivial: bool = False

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class SolidCompositionResult(NamedResult):
    """Parallel composition result type for the standalone solid engine."""

    display_formula: str = ""
    display_formula_order: str = "hill"
    name_kind: str = "formula_label"


@dataclass(frozen=True)
class SolidResult(SolidCompositionResult):
    """Parallel result type for the standalone solid naming engine."""

    reduced_formula: str = ""
    composition: tuple[tuple[str, float], ...] = ()
    pbc: tuple[bool, bool, bool] = (True, True, True)
    dimensionality: int = 3
    space_group: str | None = None
    space_group_number: int | None = None
    crystal_system: str | None = None
    pearson_symbol: str | None = None
    phase_descriptor: str | None = None
    phase: str | None = None
    miller_indices: tuple[int, ...] | None = None
    notes: tuple[str, ...] = ()

    def _replace(self, **kwargs):
        return replace(self, **kwargs)


# ── Trivial / retained IUPAC names (from Blue Book Table P-1 + common QC set) ─

_TRIVIAL_NAMES: dict[str, str] = {
    "H2": "dihydrogen",
    "N2": "dinitrogen",
    "O2": "dioxygen",
    "F2": "difluorine",
    "Cl2": "dichlorine",
    "Br2": "dibromine",
    "I2": "diiodine",
    "H2O": "water",
    "H2O2": "hydrogen peroxide",
    "NH3": "ammonia",
    "CH4": "methane",
    "C2H2": "acetylene",
    "C2H4": "ethylene",
    "C2H6": "ethane",
    "C3H6": "cyclopropane",
    "C3H8": "propane",
    "C4H6": "1,3-butadiene",
    "C4H8": "cyclobutane",
    "C4H10": "butane",
    "C5H12": "pentane",
    "C6H6": "benzene",
    "C6H12": "cyclohexane",
    "C6H14": "hexane",
    "C7H8": "toluene",
    "C8H8": "cubane",
    "C10H8": "naphthalene",
    "C14H10": "phenanthrene",
    "CO": "carbon monoxide",
    "CO2": "carbon dioxide",
    "NO": "nitrogen monoxide",
    "NO2": "nitrogen dioxide",
    "N2O": "dinitrogen monoxide",
    "SO2": "sulfur dioxide",
    "SO3": "sulfur trioxide",
    "CH2O": "formaldehyde",
    "CH4O": "methanol",
    "C2H4O": "acetaldehyde",
    "C2H6O": "ethanol",
    "C3H6O": "acetone",
    "CH2O2": "formic acid",
    "C2H4O2": "acetic acid",
    "CH3N": "methylamine",
    "C3H9N": "trimethylamine",
    "CH5N": "methylamine",
    "HF": "hydrogen fluoride",
    "HCl": "hydrogen chloride",
    "HBr": "hydrogen bromide",
    "HI": "hydrogen iodide",
    "HCN": "hydrogen cyanide",
    "H2S": "hydrogen sulfide",
    "PH3": "phosphine",
    "SiH4": "silane",
    "B2H6": "diborane",
    "H3O+": "hydronium",
    "NH4+": "ammonium",
    "OH-": "hydroxide",
    "CN-": "cyanide",
    "C20H12": "perylene",
    "C20H14N4": "porphine",
    "C60": "buckminsterfullerene",
}

# Structural disambiguation (formula + rings + n_atoms → name)
_STRUCTURAL_NAMES: dict[tuple[str, int, int], str] = {
    ("C2H2", 0, 4): "acetylene",
    ("C2H4", 0, 6): "ethylene",
    ("C4H6", 0, 10): "1,3-butadiene",
    ("C6H6", 1, 12): "benzene",
    ("C3H6", 1, 9): "cyclopropane",
    ("C4H8", 1, 12): "cyclobutane",
    ("C5H12", 0, 17): "pentane",
    ("C6H12", 1, 18): "cyclohexane",
}

# ── Greek multiplying prefixes (Red Book Table IR-4.1) ──────────────────────

_GREEK_PREFIX: dict[int, str] = {
    1: "mono",
    2: "di",
    3: "tri",
    4: "tetra",
    5: "penta",
    6: "hexa",
    7: "hepta",
    8: "octa",
    9: "nona",
    10: "deca",
    11: "undeca",
    12: "dodeca",
}

# ── Alkane roots (Blue Book P-42) ────────────────────────────────────────────

_ALKANE_ROOT: dict[int, str] = {
    1: "meth",
    2: "eth",
    3: "prop",
    4: "but",
    5: "pent",
    6: "hex",
    7: "hept",
    8: "oct",
    9: "non",
    10: "dec",
    11: "undec",
    12: "dodec",
    13: "tridec",
    14: "tetradec",
    15: "pentadec",
    16: "hexadec",
    17: "heptadec",
    18: "octadec",
    19: "nonadec",
    20: "icos",
}

# ── Main public API ─────────────────────────────────────────────────────────


def name_molecule(molecule, *, prefer_trivial: bool = True) -> str:
    """Return the IUPAC name for a vibe-qc Molecule.

    Parameters
    ----------
    molecule :
        A ``vibeqc.Molecule`` or a list of ``Atom`` objects.
    prefer_trivial :
        When ``True`` (default), return the IUPAC-preferred trivial
        name (e.g. "water", "benzene") when available.

    Returns
    -------
    IUPAC name as a string.
    """
    result = name_molecule_detailed(molecule, prefer_trivial=prefer_trivial)
    return result.name


def name_molecule_detailed(
    molecule,
    *,
    prefer_trivial: bool = True,
    with_external: bool = False,
) -> NamedResult:
    """Return the IUPAC name for a vibe-qc Molecule with provenance.

    Parameters
    ----------
    molecule :
        A ``vibeqc.Molecule`` or a list of ``Atom`` objects.
    prefer_trivial :
        Use trivial/retained names when available.
    with_external :
        If True, try external backends (RDKit, PubChem, CACTUS)
        after the built-in engine.

    Returns
    -------
    NamedResult with name, source, and confidence.
    """
    graph = from_molecule(molecule)
    return _name_from_graph(
        graph,
        prefer_trivial=prefer_trivial,
        with_external=with_external,
    )


def name_from_atoms(
    atoms: list[tuple[int, float, float, float]], *, prefer_trivial: bool = True
) -> str:
    """Return the IUPAC name for atoms given as (Z, x, y, z) in A."""
    result = name_from_atoms_detailed(atoms, prefer_trivial=prefer_trivial)
    return result.name


def name_from_atoms_detailed(
    atoms: list[tuple[int, float, float, float]],
    *,
    prefer_trivial: bool = True,
    with_external: bool = False,
) -> NamedResult:
    """Return the IUPAC name for atoms with provenance.

    Parameters
    ----------
    atoms :
        List of (Z, x_ang, y_ang, z_ang) tuples.
    prefer_trivial :
        Use trivial/retained names when available.
    with_external :
        Try external backends after built-in.

    Returns
    -------
    NamedResult.
    """
    graph = _build_atoms_graph(atoms)
    return _name_from_graph(
        graph,
        prefer_trivial=prefer_trivial,
        with_external=with_external,
    )


def name_from_graph(graph: MolecularGraph, *, prefer_trivial: bool = True) -> str:
    """Return the IUPAC name for a :class:`MolecularGraph`.

    This is the main naming dispatch.
    """
    result = name_from_graph_detailed(graph, prefer_trivial=prefer_trivial)
    return result.name


def name_from_graph_detailed(
    graph: MolecularGraph,
    *,
    prefer_trivial: bool = True,
    with_external: bool = False,
) -> NamedResult:
    """Return the IUPAC name for a :class:`MolecularGraph` with provenance."""
    return _name_from_graph(
        graph,
        prefer_trivial=prefer_trivial,
        with_external=with_external,
    )


def name_from_atoms_with_lattice(
    atoms: list[tuple[int, float, float, float]],
    lattice_vectors: Optional[list[tuple[float, float, float]]] = None,
    *,
    prefer_trivial: bool = True,
    pbc: tuple[bool, bool, bool] | None = None,
    **solid_options,
) -> NamedResult:
    """Name a solid using composition and explicit boundary conditions.

    Surface detection lives in the standalone package. Use its periodic
    dispatcher and preserve this module's result and enum types.
    """
    if lattice_vectors is None and pbc is None and not solid_options:
        return name_from_atoms_detailed(atoms, prefer_trivial=prefer_trivial)

    from vibeqc_naming import name_from_atoms_with_lattice as standalone_name

    result = standalone_name(
        atoms, lattice_vectors, prefer_trivial=prefer_trivial, pbc=pbc, **solid_options
    )
    return _adapt_solid_result(result)


def _adapt_solid_result(result):
    values = {key: getattr(result, key) for key in result.__dataclass_fields__}
    values.update(source=NamingSource(result.source.value), confidence=Confidence(result.confidence.value))
    result_type = (SolidResult if "reduced_formula" in values else
                   SolidCompositionResult if "name_kind" in values else NamedResult)
    return result_type(**values)


def name_solid_from_atoms(atoms, lattice_vectors, **options) -> SolidResult:
    """Name a solid using the shared engine, preserving this module's types."""
    from vibeqc_naming.solids import name_solid_from_atoms as standalone_name

    return _adapt_solid_result(standalone_name(atoms, lattice_vectors, **options))


def name_solid_from_formula(formula, *, prefer_trivial=False) -> SolidCompositionResult:
    """Name a composition without guessing its solid phase."""
    from vibeqc_naming.solids import name_solid_from_formula as standalone_name

    return _adapt_solid_result(standalone_name(formula, prefer_trivial=prefer_trivial))


def name_periodic_system(graph, lattice_vectors=None, *, prefer_trivial=True, pbc=None, **options):
    """Name a periodic composition without invoking molecular bond perception."""
    if lattice_vectors is None and pbc is None and not options:
        return _name_from_graph(graph, prefer_trivial=prefer_trivial)
    return name_from_atoms_with_lattice(
        [(a.z, a.x, a.y, a.z_coord) for a in graph.atoms], lattice_vectors,
        prefer_trivial=prefer_trivial, pbc=pbc, **options,
    )


def formula_to_name(formula: str) -> str:
    """Quick name lookup from Hill formula alone (no geometry needed)."""
    result = formula_to_name_detailed(formula)
    return result.name


def formula_to_name_detailed(formula: str) -> NamedResult:
    """Quick name lookup with provenance."""
    if formula in _TRIVIAL_NAMES:
        return NamedResult(
            name=_TRIVIAL_NAMES[formula],
            source=NamingSource.TRIVIAL_IUPAC,
            confidence=Confidence.HIGH,
            formula=formula,
            is_trivial=True,
        )
    result = _compositional_name_from_formula(formula)
    return NamedResult(
        name=result,
        source=NamingSource.INORGANIC_COMPOSITIONAL,
        confidence=Confidence.LOW,
        formula=formula,
        is_trivial=False,
    )


# ── Internal dispatch ────────────────────────────────────────────────────────


def _name_from_graph(
    graph: MolecularGraph,
    *,
    prefer_trivial: bool = True,
    with_external: bool = False,
) -> NamedResult:
    """Full naming pipeline for a MolecularGraph."""
    formula = graph.formula

    # ── 1. Trivial name lookup ────────────────────────────────────────────
    if prefer_trivial:
        if formula in _TRIVIAL_NAMES:
            return NamedResult(
                name=_TRIVIAL_NAMES[formula],
                source=NamingSource.TRIVIAL_IUPAC,
                confidence=Confidence.HIGH,
                formula=formula,
                is_trivial=True,
            )
        key = (formula, len(graph.rings), graph.n_atoms)
        if key in _STRUCTURAL_NAMES:
            return NamedResult(
                name=_STRUCTURAL_NAMES[key],
                source=NamingSource.TRIVIAL_IUPAC,
                confidence=Confidence.HIGH,
                formula=formula,
                is_trivial=True,
            )

    # ── 2. Coordination complex detection (Red Book IR-7, IR-9) ───────────
    if _looks_like_coordination_complex(graph):
        name = _coordination_name(graph)
        if name:
            return NamedResult(
                name=name,
                source=NamingSource.COORDINATION,
                confidence=Confidence.MEDIUM,
                formula=formula,
                is_trivial=False,
            )

    # ── 3. Organic systematic naming (Blue Book P-1–P-7) ──────────────────
    if graph.is_organic:
        organic_result = _organic_name(graph)
        if organic_result:
            conf = (
                Confidence.HIGH
                if isinstance(organic_result, str)
                else organic_result.confidence
            )
            source = NamingSource.ORGANIC_SYSTEMATIC
            if isinstance(conf, float):
                conf = (
                    Confidence.HIGH
                    if conf >= 0.7
                    else (Confidence.MEDIUM if conf >= 0.4 else Confidence.LOW)
                )
            return NamedResult(
                name=organic_result.name
                if isinstance(organic_result, OrganicResult)
                else organic_result,
                source=source,
                confidence=conf if isinstance(conf, Confidence) else conf,
                formula=formula,
                is_trivial=False,
            )

    # ── 4. Inorganic compositional naming (Red Book IR-4, IR-5) ───────────
    inorg_name = _inorganic_name(graph)
    if inorg_name:
        return NamedResult(
            name=inorg_name,
            source=NamingSource.INORGANIC_COMPOSITIONAL,
            confidence=Confidence.MEDIUM,
            formula=formula,
            is_trivial=False,
        )

    # ── 5. External backends (optional) ────────────────────────────────────
    if with_external:
        external_result = _try_external(graph)
        if external_result:
            name, source_tag = external_result
            return NamedResult(
                name=name,
                source=_external_source_tag(source_tag),
                confidence=Confidence.MEDIUM,
                formula=formula,
                is_trivial=False,
            )

    # ── 6. Ultimate fallback: compositional from formula ───────────────────
    comp = _compositional_name_from_formula(formula)
    return NamedResult(
        name=comp if comp else formula,
        source=NamingSource.INORGANIC_COMPOSITIONAL,
        confidence=Confidence.LOW,
        formula=formula,
        is_trivial=False,
    )


# ── Import organic symbols for the dispatcher above ─────────────────────────
# Deferred to here (after ``Confidence`` is defined at module top) so that
# ``organic``'s reciprocal ``from .iupac_name import Confidence`` resolves.

from .organic import (  # noqa: E402 — deferred to avoid circular import
    OrganicResult,
    name_organic,
)

# ── Inorganic compositional naming (Red Book IR-4, IR-5) ─────────────────────

_EN_ORDER = [
    "Rn",
    "Xe",
    "Kr",
    "Ar",
    "Ne",
    "He",
    "F",
    "O",
    "Cl",
    "N",
    "Br",
    "I",
    "S",
    "Se",
    "Te",
    "P",
    "As",
    "Sb",
    "C",
    "Si",
    "B",
    "Ge",
    "H",
    "Te",
    "Po",
    "At",
    "Hg",
    "Ag",
    "Au",
    "Cu",
    "Bi",
    "Pb",
    "Sn",
    "Tl",
    "Cd",
    "Zn",
    "Co",
    "Ni",
    "Fe",
    "Cr",
    "Mn",
    "V",
    "Ti",
    "Sc",
    "Al",
    "Mg",
    "Be",
    "Ca",
    "Sr",
    "Ba",
    "Li",
    "Na",
    "K",
    "Rb",
    "Cs",
]


def _electronegativity_rank(symbol: str) -> int:
    """Return a compositional-ordering rank: *lower* = more electropositive.

    ``_EN_ORDER`` is listed most-electronegative first, so electropositive
    elements (metals, H) sit near its end with a large index. Returning
    ``len - index`` maps those to a small rank, and sorting ascending then
    places the electropositive element first — the order Red Book IR-4.4.2.2
    requires (electropositive constituent named first, electronegative one
    taking the ``-ide`` suffix last). Unknown symbols rank 0, i.e. sort as
    the most electropositive.
    """
    try:
        return len(_EN_ORDER) - _EN_ORDER.index(symbol)
    except ValueError:
        return 0


_IDE_SUFFIXES: dict[str, str] = {
    "O": "oxide",
    "S": "sulfide",
    "N": "nitride",
    "C": "carbide",
    "F": "fluoride",
    "Cl": "chloride",
    "Br": "bromide",
    "I": "iodide",
    "H": "hydride",
    "P": "phosphide",
    "Se": "selenide",
    "Si": "silicide",
    "B": "boride",
    "As": "arsenide",
    "Te": "telluride",
}


def _ide_suffix(symbol: str) -> str:
    """Map element symbol to its -ide suffix form (binary compound)."""
    return _IDE_SUFFIXES.get(symbol, symbol.lower())


# Full element names for compositional naming (Red Book IR-4). A single-count
# constituent is named in full ("carbon dioxide", not "c dioxide").
_ELEMENT_NAMES: dict[str, str] = {
    "H": "hydrogen", "He": "helium", "Li": "lithium", "Be": "beryllium",
    "B": "boron", "C": "carbon", "N": "nitrogen", "O": "oxygen",
    "F": "fluorine", "Ne": "neon", "Na": "sodium", "Mg": "magnesium",
    "Al": "aluminium", "Si": "silicon", "P": "phosphorus", "S": "sulfur",
    "Cl": "chlorine", "Ar": "argon", "K": "potassium", "Ca": "calcium",
    "Sc": "scandium", "Ti": "titanium", "V": "vanadium", "Cr": "chromium",
    "Mn": "manganese", "Fe": "iron", "Co": "cobalt", "Ni": "nickel",
    "Cu": "copper", "Zn": "zinc", "Ga": "gallium", "Ge": "germanium",
    "As": "arsenic", "Se": "selenium", "Br": "bromine", "Kr": "krypton",
    "Rb": "rubidium", "Sr": "strontium", "Y": "yttrium", "Zr": "zirconium",
    "Nb": "niobium", "Mo": "molybdenum", "Tc": "technetium", "Ru": "ruthenium",
    "Rh": "rhodium", "Pd": "palladium", "Ag": "silver", "Cd": "cadmium",
    "In": "indium", "Sn": "tin", "Sb": "antimony", "Te": "tellurium",
    "I": "iodine", "Xe": "xenon", "Cs": "caesium", "Ba": "barium",
    "Hf": "hafnium", "Ta": "tantalum", "W": "tungsten", "Re": "rhenium",
    "Os": "osmium", "Ir": "iridium", "Pt": "platinum", "Au": "gold",
    "Hg": "mercury", "Tl": "thallium", "Pb": "lead", "Bi": "bismuth",
    "Po": "polonium", "At": "astatine", "Rn": "radon",
}


def _element_name(symbol: str) -> str:
    """Return the full English element name for a chemical symbol."""
    return _ELEMENT_NAMES.get(symbol, symbol.lower())


def _compositional_name_from_formula(formula: str) -> str:
    """Build a stoichiometric name from formula alone."""
    counts: dict[str, int] = {}
    for sym, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if sym:
            counts[sym] = counts.get(sym, 0) + (int(count) if count else 1)
    return _build_compositional_name(counts)


def _build_compositional_name(counts: dict[str, int]) -> str:
    """Compositional name from element count dict.

    Per Red Book IR-4.4.2.2 the electropositive constituent is named
    first and the electronegative one takes the ``-ide`` suffix last
    (e.g. "dihydrogen oxide", "carbon dioxide", "sodium chloride").
    """
    if not counts:
        return ""

    if len(counts) == 1:
        sym, count = next(iter(counts.items()))
        if count == 1:
            return _element_name(sym)
        prefix = _GREEK_PREFIX.get(count, str(count))
        return f"{prefix}{sym.lower()}"

    # Sort electropositive first (ascending rank; see _electronegativity_rank).
    sorted_elems = sorted(
        counts.items(),
        key=lambda x: _electronegativity_rank(x[0]),
    )

    parts: list[str] = []
    for i, (sym, count) in enumerate(sorted_elems):
        prefix = _GREEK_PREFIX.get(count, str(count))
        # The last (most electronegative) element takes the -ide suffix
        # (Red Book IR-4.3); every other element keeps its full name.
        if i == len(sorted_elems) - 1 and sym in _IDE_SUFFIXES:
            name = _IDE_SUFFIXES[sym]
        else:
            name = _element_name(sym)
        parts.append(f"{prefix}{name}" if count > 1 else f"{name}")

    return " ".join(parts)


def ide_map_symbol(sym: str) -> str:
    """Return the base element name for -ide construction."""
    base_names = {
        "O": "ox",
        "S": "sulf",
        "N": "nitr",
        "C": "carb",
        "F": "fluor",
        "Cl": "chlor",
        "Br": "brom",
        "I": "iod",
        "H": "hyd",
        "P": "phosph",
        "Se": "selen",
        "Si": "silic",
        "B": "bor",
        "As": "arsen",
        "Te": "tellur",
    }
    return base_names.get(sym, sym.lower())


def _inorganic_name(graph: MolecularGraph) -> Optional[str]:
    """Systematic inorganic name from molecular graph."""
    counts: dict[str, int] = {}
    for a in graph.atoms:
        counts[a.symbol] = counts.get(a.symbol, 0) + 1
    if len(counts) <= 1:
        return _build_compositional_name(counts) or None
    # Only name multi-element compounds
    result = _build_compositional_name(counts)
    return result if result else None


# ── Coordination complex naming (Red Book IR-7, IR-9) ───────────────────────


def _looks_like_coordination_complex(graph: MolecularGraph) -> bool:
    """Detect whether this graph resembles a coordination complex.

    Heuristic: contains at least one transition metal (Sc–Zn, Y–Cd,
    Hf–Hg) + at least one potential ligand atom (N, O, S, P, halogen).
    """
    metals = (
        {z for z in range(21, 31) if z != 26}
        | {
            z
            for z in range(39, 49)
            if z != 47  # Y-Cd excl Ag (keep Ag)
        }
        | {26, 47}
    )  # Fe, Ag specifically
    metals |= {z for z in range(57, 81) if not (72 <= z <= 80)}  # Lanthanides+Ac
    metals |= {72, 73, 74, 75, 76, 77, 78, 79, 80}  # Hf-Hg

    ligand_atoms = {7, 8, 16, 15, *range(9, 19)}  # N, O, S, P, halogens
    symbols = {a.symbol for a in graph.atoms}
    has_metal = any(a.z in metals for a in graph.atoms)
    has_ligand = any(
        a.z in ligand_atoms or a.symbol in {"Cl", "Br", "I", "F"} for a in graph.atoms
    )
    return has_metal and has_ligand


# Ligand name tables (Red Book Table IR-7.2, IR-9.2)
_LIGAND_NAMES: dict[str, str] = {
    # Neutral ligands
    "H2O": "aqua",
    "NH3": "ammine",
    "CO": "carbonyl",
    "NO": "nitrosyl",
    "PPh3": "triphenylphosphine",
    "en": "ethylenediamine",
    "bipy": "2,2'-bipyridine",
    "phen": "1,10-phenanthroline",
    # Anionic ligands
    "Cl": "chloro",
    "Br": "bromo",
    "I": "iodo",
    "F": "fluoro",
    "OH": "hydroxo",
    "OH2": "aqua",  # explicit water as ligand
    "O": "oxo",
    "SH": "sulfanyl",
    "CN": "cyano",
    "SCN": "thiocyanato",
    "NCS": "isothiocyanato",
    "NO2": "nitro",
    "ONO": "nitrito",
    "SO4": "sulfato",
    "NO3": "nitrato",
}

# Central atom naming: metal name vs metalate (for anionic complexes)
_METAL_NAMES: dict[int, str] = {
    21: "scandium",
    22: "titanium",
    23: "vanadium",
    24: "chromium",
    25: "manganese",
    26: "iron",
    27: "cobalt",
    28: "nickel",
    29: "copper",
    30: "zinc",
    47: "silver",
    39: "yttrium",
    40: "zirconium",
    41: "niobium",
    42: "molybdenum",
    43: "technetium",
    44: "ruthenium",
    45: "rhodium",
    46: "palladium",
    48: "cadmium",
    78: "platinum",
    79: "gold",
    80: "mercury",
}

_METALATE_NAMES: dict[int, str] = {
    26: "ferrate",
    27: "cobaltate",
    28: "nickelate",
    29: "cuprate",
    30: "zincate",
    47: "argenate",
    78: "platinate",
    79: "aurate",
    80: "mercurate",
}


def _coordination_name(graph: MolecularGraph) -> Optional[str]:
    """Build additive nomenclature name for a coordination complex (IR-7).

    Format: [ligand names alphabetically] + central metal(oxidation state)

    For anionic complexes: ligands + metalate(ox.state)
    """
    symbols = {a.symbol for a in graph.atoms}
    counts: dict[str, int] = {}
    for a in graph.atoms:
        counts[a.symbol] = counts.get(a.symbol, 0) + 1

    # Identify the central metal (the transition metal with most ligand connections)
    metals_present = [z for z in range(21, 31)]
    central_metal_z = None
    max_ligands = -1
    for a in graph.atoms:
        if (
            a.z in metals_present
            or a.z == 47
            or a.z in (39, 40, 41, 42, 43, 44, 45, 46, 48)
        ):
            ligand_count = sum(
                1
                for nb_idx in graph.atom_neighbours(a.index)
                if ATOMIC_NUMBER_TO_SYMBOL.get(graph.atoms[nb_idx].z, "")
                in {"Cl", "Br", "I", "F", "O", "N", "S"}
            )
            if ligand_count > max_ligands:
                max_ligands = ligand_count
                central_metal_z = a.z

    if central_metal_z is None or max_ligands == 0:
        return None

    # Determine oxidation state (simplified: based on ligand charges)
    ox_state = _estimate_oxidation_state(graph, central_metal_z)

    # Get metal name (metalate for anionic complexes)
    is_anionic = counts.get("charge", 0) < 0 if "charge" in counts else False
    # Heuristic: more electronegative ligands suggest anionic complex
    has_only_halogens = len({a.symbol for a in graph.atoms if a.z != 1}) <= 1

    metal_name = _METAL_NAMES.get(
        central_metal_z, ATOMIC_NUMBER_TO_SYMBOL.get(central_metal_z, "M").lower()
    )
    if is_anionic or has_only_halogens:
        metal_name = _METALATE_NAMES.get(central_metal_z, f"{metal_name}ate")

    # Build ligand names
    ligands: list[tuple[str, int]] = []
    for sym, count in counts.items():
        if sym == ATOMIC_NUMBER_TO_SYMBOL.get(central_metal_z, ""):
            continue
        if sym == "H":
            continue
        ligand_name = _LIGAND_NAMES.get(sym) or sym.lower() + "o"  # generic suffix
        if count > 1:
            ligands.append((ligand_name, count))
        else:
            ligands.append((ligand_name, 1))

    if not ligands:
        return None

    # Sort ligands alphabetically (IUPAC IR-7.2)
    ligands.sort(key=lambda x: x[0])

    # Assemble name
    parts: list[str] = []
    for name, count in ligands:
        if count > 1:
            # Use multiplicative prefixes (bis, tris, tetrakis for complex names)
            simple_prefixes = {"mono": "", "di": "di", "tri": "tri", "tetra": "tetra"}
            prefix = _GREEK_PREFIX.get(count, str(count))
            if name.startswith(("a", "e", "i", "o", "u")) or (
                "," in name or "'" in name
            ):
                parts.append(f"bis({name})")
            else:
                parts.append(f"{prefix}{name}")
        else:
            parts.append(name)

    # Add oxidation state in Stock notation
    ox_str = f"({_roman_numeral(ox_state)})" if ox_state else ""
    name = " ".join(parts) + f" {metal_name}" + ox_str

    return name


def _estimate_oxidation_state(graph: MolecularGraph, metal_z: int) -> Optional[int]:
    """Estimate the oxidation state of the central metal.

    Simplified heuristic: each anionic ligand contributes its standard charge.
    """
    total_ligand_charge = 0
    for a in graph.atoms:
        if a.z != metal_z and graph.atom_neighbours(a.index):
            # Check if this atom is bonded to the metal
            bonded_to_metal = any(
                (b.i == a.index and graph.atoms[b.j].z == metal_z)
                or (b.j == a.index and graph.atoms[b.i].z == metal_z)
                for b in graph.bonds
            )
            if bonded_to_metal:
                # Assign standard ligand charges
                if a.symbol in ("Cl", "Br", "I", "F"):
                    total_ligand_charge += -1
                elif a.symbol == "O":
                    total_ligand_charge += -2
                elif a.symbol == "N":
                    pass  # Neutral ligand (ammine, etc.)
                elif a.symbol in ("OH",):
                    total_ligand_charge += -1

    # Balance: metal oxidation state + ligand charges = overall charge
    return -total_ligand_charge if total_ligand_charge != 0 else None


def _roman_numeral(n: int) -> str:
    """Convert integer to Roman numeral string."""
    if n < 0:
        return f"-{_roman_numeral(-n)}"
    vals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    result = []
    for val, sym in vals:
        while n >= val:
            result.append(sym)
            n -= val
    return "".join(result)


# ── Organic naming dispatch (integrated with organic.py) ─────────────────────


def _carbon_count(graph: MolecularGraph) -> int:
    return sum(1 for a in graph.atoms if a.z == 6)


def _hydrogen_count(graph: MolecularGraph) -> int:
    return sum(1 for a in graph.atoms if a.z == 1)


def _is_alkane(graph: MolecularGraph) -> bool:
    n_c = _carbon_count(graph)
    n_h = _hydrogen_count(graph)
    if n_c == 0:
        return False
    if len({a.symbol for a in graph.atoms}) != 2:
        return False
    if "C" not in {a.symbol for a in graph.atoms}:
        return False
    if n_h == 2 * n_c + 2:
        return True
    if n_h == 2 * n_c and len(graph.rings) == 1:
        return True
    return False


def _is_alkene_alkyne(graph: MolecularGraph) -> Optional[str]:
    n_c = _carbon_count(graph)
    n_h = _hydrogen_count(graph)
    if n_c == 0:
        return None
    syms = {a.symbol for a in graph.atoms}
    if syms != {"C", "H"}:
        return None
    n_double = sum(1 for b in graph.bonds if b.order >= 2)
    n_triple = sum(1 for b in graph.bonds if b.order >= 3)
    if n_h == 2 * n_c and n_double == 1:
        return "alkene"
    if n_h == 2 * n_c - 2 and n_triple == 1:
        return "alkyne"
    if n_h == 2 * n_c - 4 and n_double == 2:
        return "alkadiene"
    return None


def _is_aromatic_hydrocarbon(graph: MolecularGraph) -> bool:
    if not graph.is_organic:
        return False
    syms = {a.symbol for a in graph.atoms}
    return syms == {"C", "H"} and len(graph.rings) >= 1


def _organic_name(graph: MolecularGraph) -> Optional[str]:
    """Build systematic organic name. Returns None if not applicable."""
    n_c = _carbon_count(graph)
    if n_c == 0:
        return None

    # Simple alkane/alkene/aromatic checks (fast path)
    if _is_alkane(graph):
        if len(graph.rings) == 0:
            root = _ALKANE_ROOT.get(n_c, f"C{n_c}")
            return f"{root}ane"
        elif len(graph.rings) == 1:
            root = _ALKANE_ROOT.get(n_c, f"C{n_c}")
            return f"cyclo{root}ane"

    alkene_class = _is_alkene_alkyne(graph)
    if alkene_class and len(graph.rings) == 0:
        root = _ALKANE_ROOT.get(n_c, f"C{n_c}")
        suffix_map = {"alkene": "ene", "alkyne": "yne", "alkadiene": "adiene"}
        return f"{root}{suffix_map[alkene_class]}"

    if _is_aromatic_hydrocarbon(graph):
        formula = graph.formula
        key = (formula, len(graph.rings), graph.n_atoms)
        if key in _STRUCTURAL_NAMES:
            return _STRUCTURAL_NAMES[key]
        if formula in _TRIVIAL_NAMES:
            return _TRIVIAL_NAMES[formula]

    # Delegate to full organic engine (Blue Book P-1–P-7)
    org_result = name_organic(graph)
    if org_result:
        return org_result

    return None


# ── Helper ───────────────────────────────────────────────────────────────────


def _build_atoms_graph(atoms: list[tuple[int, float, float, float]]) -> MolecularGraph:
    """Build graph from (Z, x, y, z) in Angstrom."""
    from .molecular_graph import from_atoms_list

    return from_atoms_list(atoms)


def _try_external(graph: MolecularGraph) -> Optional[tuple[str, str]]:
    """Try external naming backends. Returns (name, source_tag) or None."""
    try:
        from .external import name_from_atoms_external

        result = name_from_atoms_external(
            [(a.z, a.x, a.y, a.z_coord) for a in graph.atoms]
        )
        return result
    except ImportError:
        return None


def _external_source_tag(tag: str) -> NamingSource:
    """Map source tag to NamingSource enum."""
    mapping = {
        "rdkit": NamingSource.EXTERNAL_RDKIT,
        "cactus": NamingSource.EXTERNAL_CACTUS,
        "pubchem": NamingSource.EXTERNAL_PUBCHEM,
    }
    return mapping.get(tag, NamingSource.INORGANIC_COMPOSITIONAL)


# ── vibe-view QVF integration ───────────────────────────────────────────────


def name_from_qvf(qvf_path: str, **solid_options) -> NamedResult:
    """Get the IUPAC name for the structure inside a QVF archive.

    Parameters
    ----------
    qvf_path :
        Path to a .qvf file.

    Returns
    -------
    NamedResult with the IUPAC name, source, and confidence.
    """
    # vibe-qc reads its own format. This used to prefer vibe-view's
    # QVFReader, which made the core depend on the viewer for a job the
    # producer can do itself; the local fallback it named
    # (vibeqc.output.formats.qvf.QVFReader) never existed, so the path was
    # dead whenever vibe-view was absent. Reading the structure section is
    # a zip member plus a JSON parse against a digest we wrote ourselves.
    from vibeqc_naming._qvf_read import QVFReadError, QVFReader

    try:
        reader = QVFReader(qvf_path)
    except QVFReadError as exc:
        return NamedResult(
            name=f"(QVF read error: {exc})",
            source=NamingSource.INORGANIC_COMPOSITIONAL,
            confidence=Confidence.LOW,
            formula="",
            is_trivial=False,
        )
    try:
        structure = reader.read_structure()
    except Exception as exc:  # noqa: BLE001
        reader.close()
        return NamedResult(
            name=f"(QVF read error: {exc})",
            source=NamingSource.INORGANIC_COMPOSITIONAL,
            confidence=Confidence.LOW,
            formula="",
            is_trivial=False,
        )

    reader.close()

    # Convert StructureData to atom tuples for naming.
    atoms = [
        (a.atomic_number, a.position[0], a.position[1], a.position[2])
        for a in structure.atoms
    ]
    lattice = structure.lattice_vectors
    if any(structure.pbc):
        if lattice is None or lattice.size == 0:
            raise ValueError("periodic QVF structure has no lattice vectors")
        if "pbc" in solid_options or "occupancies" in solid_options:
            raise ValueError("QVF boundary conditions and occupancies come from the archive")
        solid_options.setdefault("prefer_trivial", False)
        return name_from_atoms_with_lattice(
            atoms, lattice, pbc=structure.pbc,
            occupancies=[a.occupancy for a in structure.atoms], **solid_options,
        )
    if solid_options:
        raise ValueError("solid options require a periodic QVF structure")
    result = name_from_atoms_detailed(atoms)

    # Override formula from the QVF data if we have it
    counts: dict[str, int] = {}
    for a in structure.atoms:
        counts[a.symbol] = counts.get(a.symbol, 0) + 1
    parts: list[str] = []
    if "C" in counts:
        c_count = counts.pop("C")
        parts.append(f"C{c_count}" if c_count > 1 else "C")
    if "H" in counts:
        h_count = counts.pop("H")
        parts.append(f"H{h_count}" if h_count > 1 else "H")
    for sym, count in sorted(counts.items()):
        parts.append(f"{sym}{count}" if count > 1 else sym)

    if parts:
        result = replace(result, formula="".join(parts))

    return result
