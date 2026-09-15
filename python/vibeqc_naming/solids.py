"""Composition-first solid naming (IUPAC Red Book 2005, IR-11).

Composition cannot identify a mineral, polymorph, oxidation state, or Miller
plane. Structural claims are recorded separately. No vibeqc core is needed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction
import math
import re
from typing import Sequence

import numpy as np

from ._types import Confidence, NamedResult, NamingSource
from ._solid_elements import ELEMENTS
from ._solid_rules import ELEMENT_RANK, IDE_NAMES, multiplicative_prefix


_ELEMENT_NAMES = {symbol: name for symbol, name in ELEMENTS.values()}
# Composition names only: no mineral or polymorph assignments.
_COMMON_NAMES = {
    "H2O": "water", "Al2O3": "alumina", "O2Si": "silica", "O2Ti": "titania",
    "MgO": "magnesia", "CeO2": "ceria", "O2Zr": "zirconia", "HfO2": "hafnia",
}
_TERNARY_NAMES = {
    "BaO3Ti": "barium titanate", "O3SrTi": "strontium titanate",
    "CaO3Ti": "calcium titanate", "LiNbO3": "lithium niobate",
}
_TERNARY_FORMULAS = {
    "BaO3Ti": "BaTiO3", "O3SrTi": "SrTiO3", "CaO3Ti": "CaTiO3", "LiNbO3": "LiNbO3",
}


def _formula(counts, symbols=None) -> str:
    hill = symbols is None
    symbols = sorted(counts) if hill else list(symbols)
    if hill and "C" in symbols:
        symbols.remove("C")
        if "H" in symbols:
            symbols.remove("H")
            symbols.insert(0, "H")
        symbols.insert(0, "C")
    def suffix(amount):
        if amount == 1:
            return ""
        if amount == int(amount):
            return str(int(amount))
        return f"{float(amount):.12g}"

    return "".join(symbol + suffix(counts[symbol]) for symbol in symbols if counts[symbol] > 0)


def _display_formula(counts):
    """Use binary conventional order; unknown multinary labels retain Hill order."""
    if len(counts) <= 2 and all(s in ELEMENT_RANK for s in counts):
        return _formula(counts, sorted(counts, key=ELEMENT_RANK.__getitem__)), "compositional"
    hill = _formula(counts)
    if hill in _TERNARY_FORMULAS:
        return _TERNARY_FORMULAS[hill], "compositional"
    return hill, "hill"


def _reduced_counts(counts):
    denominator = math.lcm(*(value.denominator for value in counts.values()))
    integers = {symbol: int(value * denominator) for symbol, value in counts.items()}
    divisor = math.gcd(*integers.values())
    return {symbol: value // divisor for symbol, value in integers.items()}


def _composition_name(counts, prefer_trivial):
    formula = _formula(counts)
    if len(counts) == 1:
        return _ELEMENT_NAMES[next(iter(counts))], False, "systematic"
    if prefer_trivial and formula in _COMMON_NAMES:
        return _COMMON_NAMES[formula], True, "common"
    if formula in _TERNARY_NAMES:
        return _TERNARY_NAMES[formula], False, "systematic"
    if len(counts) == 2 and all(s in ELEMENT_RANK for s in counts):
        elements = sorted(counts, key=ELEMENT_RANK.__getitem__)
        parts = []
        for index, symbol in enumerate(elements):
            name = IDE_NAMES[symbol] if index else _ELEMENT_NAMES[symbol]
            prefix = multiplicative_prefix(counts[symbol]) if counts[symbol] > 1 else ""
            if prefix is None:
                break
            parts.append(prefix + name)
        else:
            return " ".join(parts), False, "systematic"
    # Multinary compositions need constituent identities beyond element counts.
    # A formula label makes no unsupported claim about bonding or phase purity.
    return f"{_display_formula(counts)[0]} solid", False, "formula_label"


@dataclass(frozen=True, repr=False)
class SolidCompositionResult(NamedResult):
    """Composition name or explicit label, without an assertion of structure.

    ``formula`` retains the Hill-order database convention. ``display_formula``
    uses chemical order where supported, identified by ``display_formula_order``.
    ``name_kind`` is systematic, common, formula_label, or descriptive.
    """

    display_formula: str = ""
    display_formula_order: str = "hill"
    name_kind: str = "formula_label"


def name_solid_from_formula(formula: str, *, prefer_trivial: bool = False) -> SolidCompositionResult:
    """Name a composition from a plain elemental formula (e.g. Fe2O3).

    Supports positive decimal counts for average compositions. Parentheses,
    charges and symbolic variables require resolved atom counts instead.
    No structural or phase identification is attempted.
    """
    pattern = r"([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?)?"
    if not isinstance(formula, str) or not re.fullmatch(f"(?:{pattern})+", formula):
        raise ValueError("formula must be a nonempty sequence of elements and positive counts")
    counts = Counter()
    for symbol, count in re.findall(pattern, formula):
        amount = Fraction(count or "1")
        if symbol not in _ELEMENT_NAMES or amount <= 0:
            raise ValueError("formula contains an unknown element or nonpositive count")
        counts[symbol] += amount
    reduced = _reduced_counts(counts)
    name, trivial, kind = _composition_name(reduced, prefer_trivial)
    display, order = _display_formula(reduced)
    if any(value.denominator != 1 for value in counts.values()):
        # Decimal coefficients carry the caller's reference composition.
        display, order = _display_formula(counts)
        name, trivial, kind = f"{display} solid", False, "formula_label"
    return SolidCompositionResult(
        name=name, source=NamingSource.INORGANIC_COMPOSITIONAL,
        confidence=Confidence.MEDIUM, formula=_formula(reduced), is_trivial=trivial,
        display_formula=display, display_formula_order=order, name_kind=kind,
    )


@dataclass(frozen=True, repr=False)
class SolidResult(SolidCompositionResult):
    """Solid name plus composition and the evidence for structural labels.

    ``formula`` is the occupied unit-cell formula in Hill order;
    ``reduced_formula`` is its simplest integer ratio, also in Hill order.
    ``composition`` retains the occupancy-weighted counts before reduction.
    ``dimensionality`` describes the boundary conditions, not bond topology.
    """

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

    def _replace(self, **kwargs) -> SolidResult:
        return replace(self, **kwargs)


def _periodic_flags(pbc):
    if pbc is None or len(pbc) != 3 or any(not isinstance(flag, (bool, np.bool_)) for flag in pbc):
        raise ValueError("pbc must contain three booleans")
    return tuple(bool(flag) for flag in pbc)


def _validate(atoms, lattice_vectors, pbc, occupancies):
    lattice = np.asarray(lattice_vectors, dtype=float)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("lattice_vectors must be a finite 3 by 3 matrix in angstrom")
    if np.linalg.matrix_rank(lattice) != 3:
        raise ValueError("lattice_vectors must be linearly independent")
    pbc = _periodic_flags(pbc)
    if not any(pbc):
        raise ValueError("a solid requires at least one periodic axis; use molecular naming")
    array = np.asarray(atoms, dtype=float)
    if array.ndim != 2 or array.shape[1] != 4 or len(array) == 0:
        raise ValueError("atoms must be a nonempty sequence of (Z, x, y, z) in angstrom")
    if not np.isfinite(array).all():
        raise ValueError("atoms must contain finite atomic numbers and coordinates")
    numbers = array[:, 0]
    if np.any(numbers != np.floor(numbers)) or np.any((numbers < 1) | (numbers > 118)):
        raise ValueError("atomic numbers must be integers from 1 through 118")
    weights = np.ones(len(array)) if occupancies is None else np.asarray(occupancies, dtype=float)
    if weights.shape != (len(array),) or not np.isfinite(weights).all():
        raise ValueError("occupancies must contain one finite value per atom")
    if np.any((weights < 0) | (weights > 1)) or not np.any(weights > 0):
        raise ValueError("occupancies must be in [0, 1], with at least one occupied site")
    fractional = array[:, 1:] @ np.linalg.inv(lattice)
    fractional[:, pbc] %= 1.0
    # Alternative species at the same site may share its occupancy, but their
    # total must not exceed one. Include periodic images in the same check.
    sites = Counter()
    for position, weight in zip(fractional, weights):
        position = np.round(position, 10)
        position[list(pbc)] %= 1.0
        sites[tuple(position)] += float(weight)
    if any(weight > 1 + 1e-9 for weight in sites.values()):
        raise ValueError("occupancies at a shared site must sum to at most one")
    return array, lattice, pbc, weights


def _symmetry(atoms, lattice, symprec):
    """Optional 3D crystallographic analysis, never a mineral assignment."""
    try:
        import spglib
    except ImportError:
        return {}, "Symmetry unavailable: install spglib to enable crystallographic analysis."
    fractional = (atoms[:, 1:] @ np.linalg.inv(lattice)) % 1.0
    try:
        dataset = spglib.get_symmetry_dataset(
            (lattice, fractional, atoms[:, 0].astype(int)), symprec=symprec
        )
    except Exception as exc:
        return {}, f"Symmetry analysis failed: {type(exc).__name__}."
    if dataset is None:
        return {}, "Symmetry analysis found no dataset at the requested tolerance."
    # The attribute interface is supported by spglib >=2.5; older versions
    # return a dict. Keep this optional integration compatible with >=2.0.
    number = int(dataset.number if hasattr(dataset, "number") else dataset["number"])
    symbol = dataset.international if hasattr(dataset, "international") else dataset["international"]
    system = next(system for upper, system in (
        (2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
        (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"), (230, "cubic"),
    ) if number <= upper)
    standard_types = dataset.std_types if hasattr(dataset, "std_types") else dataset["std_types"]
    system_letter = {"triclinic": "a", "monoclinic": "m", "orthorhombic": "o",
                     "tetragonal": "t", "trigonal": "h", "hexagonal": "h", "cubic": "c"}[system]
    centering = "S" if symbol[0] in "ABC" else symbol[0]
    # spglib's default R setting is hexagonal. Count its conventional cell,
    # not the primitive rhombohedron (Red Book example carbon(hR6)).
    pearson = f"{system_letter}{centering}{len(standard_types)}"
    return {
        "space_group": symbol, "space_group_number": number, "crystal_system": system,
        "pearson_symbol": pearson,
    }, f"Space group determined by spglib at symprec={symprec:g} angstrom."


def _name_adsorbates(atoms, lattice, pbc, weights, groups, prefer_trivial, phase, miller, formula_units):
    from .iupac_name import name_from_atoms_detailed

    if sum(pbc) != 2:
        raise ValueError("adsorbate_groups requires a 2D substrate")
    used = set()
    names = []
    confidence = Confidence.MEDIUM
    for group in groups:
        indices = list(group)
        if not indices or any(
            isinstance(i, bool) or not isinstance(i, (int, np.integer))
            or i < 0 or i >= len(atoms) for i in indices
        ):
            raise ValueError("adsorbate groups must contain valid atom indices")
        if len(set(indices)) != len(indices) or used.intersection(indices):
            raise ValueError("adsorbate groups must be disjoint and contain no duplicate indices")
        if any(weights[i] != 1 for i in indices):
            raise ValueError("adsorbate atoms must be fully occupied")
        used.update(indices)
        molecule = [(int(atoms[i, 0]), *atoms[i, 1:]) for i in indices]
        result = name_from_atoms_detailed(molecule, prefer_trivial=prefer_trivial)
        names.append(result.name)
        if result.confidence == Confidence.LOW:
            confidence = Confidence.LOW
    remaining = [i for i in range(len(atoms)) if i not in used]
    if not names or not remaining:
        raise ValueError("supply at least one adsorbate group and leave an occupied substrate")
    substrate = name_solid_from_atoms(
        atoms[remaining], lattice, pbc=pbc, occupancies=weights[remaining],
        prefer_trivial=prefer_trivial, phase=phase, miller_indices=miller,
        formula_units=formula_units,
    )
    multiplicities = Counter(names)
    adsorbates = " + ".join(
        name if count == 1 else f"{count} × {name}"
        for name, count in sorted(multiplicities.items())
    )
    return f"{adsorbates} on {substrate.name.removesuffix(' slab')}", confidence


def name_solid_from_atoms(
    atoms: Sequence[tuple[int, float, float, float]],
    lattice_vectors: Sequence[Sequence[float]],
    *,
    pbc: Sequence[bool] = (True, True, True),
    occupancies: Sequence[float] | None = None,
    formula_units: float | None = None,
    prefer_trivial: bool = False,
    analyze_symmetry: bool = False,
    symprec: float = 1e-5,
    phase: str | None = None,
    miller_indices: Sequence[int] | None = None,
    adsorbate_groups: Sequence[Sequence[int]] | None = None,
) -> SolidResult:
    """Name a solid from composition, boundary conditions and optional metadata.

    Coordinates and lattice rows are in angstrom. Explicit PBC determines
    dimensionality; omitting it means 3D, including cells containing vacuum.
    Fractional occupancies describe the average composition. Unknown compounds
    receive a formula label rather than a speculative chemical or mineral name.
    For partial occupancy, ``formula_units`` sets the divisor for display counts.
    By default the GCD of unweighted occupied-species site counts is used; this
    preserves the reference site multiplicities under supercell repetition.

    ``phase`` and ``miller_indices`` are caller-supplied, not inferred.
    ``adsorbate_groups`` supplies disjoint atom-index groups on a 2D substrate;
    the remaining atoms define that substrate. Automatic partitioning of a
    periodic graph is intentionally not used to guess adsorption.

    ``analyze_symmetry`` optionally uses spglib for fully occupied 3D systems.
    Missing/failed symmetry analysis leaves composition naming available and
    records the limitation in ``notes``. No external network calls are made.
    """
    array, lattice, pbc, weights = _validate(atoms, lattice_vectors, pbc, occupancies)
    if not math.isfinite(symprec) or symprec <= 0:
        raise ValueError("symprec must be a positive finite tolerance in angstrom")
    if formula_units is not None and (
        isinstance(formula_units, bool) or not math.isfinite(formula_units) or formula_units <= 0
    ):
        raise ValueError("formula_units must be positive and finite")
    if phase is not None and (not isinstance(phase, str) or not phase.strip()):
        raise ValueError("phase must be a nonempty label")
    dimensions = sum(pbc)
    miller = None
    if miller_indices is not None:
        miller = tuple(miller_indices)
        if (dimensions != 2 or len(miller) not in (3, 4) or not any(miller)
                or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in miller)):
            raise ValueError("Miller indices require a slab and three or four nonzero-as-a-group integers")
        if len(miller) == 4 and sum(miller[:3]) != 0:
            raise ValueError("four-index Miller-Bravais notation requires h + k + i = 0")
        miller = tuple(int(i) for i in miller)

    counts = Counter()
    site_counts = Counter()
    for atom, weight in zip(array, weights):
        if weight > 0:
            symbol = ELEMENTS[int(atom[0])][0]
            counts[symbol] += Fraction(str(float(weight)))
            site_counts[symbol] += 1
    reduced = _reduced_counts(counts)
    name, trivial, kind = _composition_name(reduced, prefer_trivial)
    display, order = _display_formula(reduced)
    notes = []
    partial = any(0 < weight < 1 for weight in weights)
    if partial:
        divisor = (Fraction(str(formula_units)) if formula_units is not None
                   else math.gcd(*site_counts.values()))
        normalized = {symbol: amount / divisor for symbol, amount in counts.items()}
        display, order = _display_formula(normalized)
        name, trivial, kind = f"{display} solid", False, "formula_label"
        notes.append("Partial occupancies supplied; the name describes average composition, not site ordering.")
        basis = "caller-supplied formula units" if formula_units is not None else "GCD of occupied-species site counts"
        notes.append(f"Display counts divided by {float(divisor):g} ({basis}).")
    if phase is not None:
        phase = phase.strip()
        name += f" ({phase})"
        kind = "descriptive"
        notes.append("Phase label supplied by caller; it was not inferred or verified.")
    if miller is not None:
        name += " (" + " ".join(str(i) for i in miller) + ")"
        notes.append("Miller indices supplied by caller in the parent crystal basis.")
    if dimensions < 3:
        if name.endswith(" solid"):
            name = name[:-6]
        name += " slab" if dimensions == 2 else " chain"
        kind = "descriptive"

    confidence = Confidence.MEDIUM
    if adsorbate_groups is not None:
        name, confidence = _name_adsorbates(
            array, lattice, pbc, weights, adsorbate_groups, prefer_trivial, phase, miller, formula_units
        )
        trivial = False
        kind = "descriptive"
        notes.append("Substrate and adsorbate partition supplied by caller.")

    symmetry = {}
    if analyze_symmetry:
        if dimensions != 3 or partial:
            notes.append("3D space-group analysis skipped for partial occupancy or reduced periodicity.")
        else:
            symmetry, note = _symmetry(array[weights > 0], lattice, symprec)
            notes.append(note)

    return SolidResult(
        name=name, source=NamingSource.INORGANIC_COMPOSITIONAL,
        confidence=confidence, is_trivial=trivial,
        formula=_formula(counts), reduced_formula=_formula(reduced),
        display_formula=display, display_formula_order=order, name_kind=kind,
        phase_descriptor=(f"{display}({symmetry['pearson_symbol']})"
                          if symmetry else None),
        composition=tuple((s, float(counts[s])) for s in sorted(counts)),
        pbc=pbc, dimensionality=dimensions, phase=phase, miller_indices=miller,
        notes=tuple(notes), **symmetry,
    )
