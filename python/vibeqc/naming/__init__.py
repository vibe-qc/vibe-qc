"""Molecular IUPAC naming from 3D structure — shim over vibeqc_naming.

This is a thin re-export of the standalone ``vibeqc_naming`` package.
Because ``vibeqc/__init__.py`` unconditionally imports the compiled
``_vibeqc_core`` C++ extension, ``vibeqc.naming`` is only importable
when the full vibe-qc package has been built.

For standalone use without the C++ extension, import directly from
``vibeqc_naming`` instead::

    from vibeqc_naming import name_from_atoms, name_molecule

Usage (when vibe-qc is built)::

    from vibeqc.naming import name_molecule, name_from_atoms

    # From a vibe-qc Molecule (C++ extension object)
    name = name_molecule(molecule)

    # From raw atom data: list of (Z, x_ang, y_ang, z_ang)
    name = name_from_atoms([(8, 0.0, 0.0, 0.117), (1, 0.0, 0.757, -0.469), ...])

    # Get provenance and confidence
    from vibeqc.naming import name_molecule_detailed
    result = name_molecule_detailed(molecule)
    print(result.name, result.source, result.confidence)

    # Quick formula-only lookup (no geometry needed)
    from vibeqc.naming import formula_to_name
    name = formula_to_name("H2O")  # -> "water"

    # From vibe-view QVF archive
    from vibeqc.naming import name_from_qvf
    result = name_from_qvf("calculation.qvf")

References
----------
- IUPAC Red Book 2005: *Nomenclature of Inorganic Chemistry*
- IUPAC Blue Book 2013: *Nomenclature of Organic Chemistry*
- Cordero et al., *Dalton Trans.* 2008, 2832 (covalent radii)
"""

from __future__ import annotations

# Re-export everything from the standalone vibeqc_naming package.
# This shim exists so that ``vibeqc.naming`` remains the canonical
# import path when vibe-qc is fully built, while ``vibeqc_naming``
# is the import path for standalone use (e.g. CI, development, or
# vibe-view without the C++ extension).
from vibeqc_naming import *  # noqa: F403
from vibeqc_naming import (
    known_names,
    structure_from_name,
    name_ring_system,
    structure_from_name,
    known_names,
    Confidence,
    MolecularGraph,
    NamedResult,
    SolidResult,
    SolidCompositionResult,
    NamingSource,
    OrganicResult,
    SlabSystem,
    name_ring_system,
    structure_from_name,
    known_names,
    build_graph,
    detect_slab,
    formula_to_name,
    formula_to_name_detailed,
    from_atoms_list,
    from_molecule,
    name_from_atoms,
    name_from_atoms_detailed,
    name_from_atoms_with_lattice,
    name_from_formula_external,
    name_from_graph,
    name_from_graph_detailed,
    name_from_qvf,
    name_from_rdkit,
    name_molecule,
    name_molecule_detailed,
    name_periodic_system,
    name_slab_system,
    name_solid_from_atoms,
    name_solid_from_formula,
)

__all__ = [
    # Result types
    "NamedResult",
    "SolidResult",
    "SolidCompositionResult",
    "NamingSource",
    "Confidence",
    "OrganicResult",
    # Public API — string names
    "name_molecule",
    "name_from_atoms",
    "name_from_graph",
    "formula_to_name",
    "name_from_qvf",
    "name_solid_from_atoms",
    "name_solid_from_formula",
    "name_from_atoms_with_lattice",
    "name_periodic_system",
    # Public API — detailed (with provenance)
    "name_molecule_detailed",
    "name_from_atoms_detailed",
    "name_from_graph_detailed",
    "formula_to_name_detailed",
    # Graph construction
    "from_molecule",
    "from_atoms_list",
    "build_graph",
    "MolecularGraph",
    # External backends (optional)
    "name_from_rdkit",
    "name_from_formula_external",
]
