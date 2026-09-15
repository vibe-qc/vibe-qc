"""Molecular IUPAC naming from 3D structure — standalone package.

This is the standalone vibeqc_naming package that works without the
vibeqc C++ extension. Because vibeqc/__init__.py unconditionally imports
the compiled _vibeqc_core extension, the naming module cannot be imported
as ``vibeqc.naming`` unless the full vibe-qc package is built.

Usage::

    from vibeqc_naming import name_from_atoms, name_molecule

    # From raw atom data: list of (Z, x_ang, y_ang, z_ang)
    name = name_from_atoms([(8, 0.0, 0.0, 0.117), (1, 0.0, 0.757, -0.469), ...])

    # Get provenance and confidence
    from vibeqc_naming import name_from_atoms_detailed
    result = name_from_atoms_detailed(atoms)
    print(result.name, result.source, result.confidence)

    # Quick formula-only lookup (no geometry needed)
    from vibeqc_naming import formula_to_name
    name = formula_to_name("H2O")  # -> "water"

    # From vibe-view QVF archive
    from vibeqc_naming import name_from_qvf
    result = name_from_qvf("calculation.qvf")

    # When vibe-qc IS built, the shim at vibeqc.naming re-exports this package:
    from vibeqc.naming import name_from_atoms  # same API, same result

References
----------
- IUPAC Red Book 2005: *Nomenclature of Inorganic Chemistry*
- IUPAC Blue Book 2013: *Nomenclature of Organic Chemistry*
- Cordero et al., *Dalton Trans.* 2008, 2832 (covalent radii)
"""

from __future__ import annotations

from ._types import (
    Confidence,
    NamedResult,
    NamingSource,
    OrganicResult,
)
from .external import name_from_formula_external, name_from_rdkit
from .iupac_name import (
    formula_to_name,
    formula_to_name_detailed,
    name_from_atoms,
    name_from_atoms_detailed,
    name_from_atoms_with_lattice,
    name_from_graph,
    name_from_graph_detailed,
    name_from_qvf,
    name_molecule,
    name_molecule_detailed,
    name_periodic_system,
)
from .molecular_graph import (
    MolecularGraph,
    build_graph,
    from_atoms_list,
    from_molecule,
)
from .heterocycle import name_ring_system
from .structure_db import structure_from_name, known_names
from .smiles import graph_to_smiles, smiles_from_atoms, smiles_from_name
from .surface import (
    SlabSystem,
    detect_slab,
    name_slab_system,
)
from .solids import SolidCompositionResult, SolidResult, name_solid_from_atoms, name_solid_from_formula

__all__ = [
    # Result types
    "NamedResult",
    "SolidResult",
    "SolidCompositionResult",
    "NamingSource",
    "Confidence",
    "OrganicResult",
    "name_ring_system",
    "structure_from_name",
    "known_names",
    "graph_to_smiles",
    "smiles_from_atoms",
    "smiles_from_name",
    # Surface/slab types
    "SlabSystem",
    "detect_slab",
    "name_slab_system",
    # Public API — string names
    "name_molecule",
    "name_from_atoms",
    "name_from_atoms_with_lattice",
    "name_periodic_system",
    "name_solid_from_atoms",
    "name_solid_from_formula",
    "name_from_graph",
    "formula_to_name",
    "name_from_qvf",
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
