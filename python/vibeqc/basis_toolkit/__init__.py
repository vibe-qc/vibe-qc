"""vibe-qc basis-set toolkit.

A production-quality prototype for modern basis-set storage, validation,
conversion, and export.  Supports the Basis Set Exchange ecosystem and
the new QVF-Basis standalone container format.

Quick start
-----------

.. code-block:: python

    from vibeqc.basis_toolkit import (
        BasisSetData, BasisRole, HarmonicType,
        from_bse_json, from_bse_file, to_g94, to_orca, to_nwchem,
        to_qvf_json, from_qvf_json, save_qvf, load_qvf,
        validate_basis_file, loss_report_g94,
    )

    # Import from BSE JSON
    bse = from_bse_file("cc-pvdz.json")
    print(bse.name, bse.basis_family, bse.elements.keys())

    # Export to legacy formats
    g94_text = to_g94(bse)
    orca_text = to_orca(bse)
    nwchem_text = to_nwchem(bse)

    # Save as QVF-Basis (text mode)
    save_qvf_json(bse, "cc-pvdz.qvf.json")

    # Save as QVF-Basis (packaged mode)
    save_qvf(bse, "cc-pvdz.qvf")

    # Round-trip
    bse2 = load_qvf("cc-pvdz.qvf")
    assert bse.name == bse2.name

    # Validate
    errors = validate_basis_file("cc-pvdz.qvf.json")
    assert errors == []
"""

from __future__ import annotations

from .bridge import from_libint_basis, to_libint_basis
from .exporter_g94 import loss_report_g94, to_g94, to_g94_file
from .exporter_nwchem import loss_report_nwchem, to_nwchem, to_nwchem_file
from .exporter_orca import loss_report_orca, to_orca, to_orca_file
from .importer_bse import from_bse_file, from_bse_json
from .importer_crystal import from_crystal_atom, from_crystal_file
from .importer_g94 import from_g94, from_g94_file
from .importer_nwchem import from_nwchem, from_nwchem_file
from .importer_orca import from_orca, from_orca_file
from .library_writer import (
    BASIS_LIBRARY_DIR,
    library_formats,
    read_from_library,
    write_to_library,
)
from .model import (
    BasisRole,
    BasisSetData,
    ECPEntry,
    ECPPotential,
    ECPTerm,
    ElementBasis,
    HarmonicType,
    LossReport,
    Primitive,
    Provenance,
    Reference,
    Shell,
)
from .qvf_basis import (
    from_qvf_bytes,
    from_qvf_json,
    load_any,
    load_qvf,
    qvf_manifest,
    save_any,
    save_qvf,
    save_qvf_json,
    to_qvf_bytes,
    to_qvf_json,
)
from .validator import validate_basis_dict, validate_basis_file, validate_basis_json

__all__ = [
    # Canonical model
    "BasisSetData",
    "BasisRole",
    "ElementBasis",
    "Shell",
    "Primitive",
    "ECPEntry",
    "ECPPotential",
    "ECPTerm",
    "HarmonicType",
    "Provenance",
    "Reference",
    "LossReport",
    # BSE importer
    "from_bse_json",
    "from_bse_file",
    # G94 importer
    "from_g94",
    "from_g94_file",
    # ORCA importer
    "from_orca",
    "from_orca_file",
    # NWChem importer
    "from_nwchem",
    "from_nwchem_file",
    # CRYSTAL importer
    "from_crystal_atom",
    "from_crystal_file",
    # Library writer
    "write_to_library",
    "read_from_library",
    "BASIS_LIBRARY_DIR",
    "library_formats",
    # Direct C++ BasisSet bridge
    "to_libint_basis",
    "from_libint_basis",
    # Legacy exporters
    "to_g94",
    "to_g94_file",
    "loss_report_g94",
    "to_orca",
    "to_orca_file",
    "loss_report_orca",
    "to_nwchem",
    "to_nwchem_file",
    "loss_report_nwchem",
    # QVF-Basis serializer
    "to_qvf_json",
    "from_qvf_json",
    "save_qvf_json",
    "to_qvf_bytes",
    "save_qvf",
    "from_qvf_bytes",
    "load_qvf",
    "qvf_manifest",
    "load_any",
    "save_any",
    # Validator
    "validate_basis_dict",
    "validate_basis_json",
    "validate_basis_file",
]
