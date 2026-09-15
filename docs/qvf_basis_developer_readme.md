# QVF-Basis -- Developer README

> Production-quality prototype for modern basis-set storage, validation,
> conversion, and export. Part of the vibe-qc quantum chemistry project.

## Overview

The `vibeqc.basis_toolkit` package provides:

| Capability | Module |
|---|---|
| Canonical typed data model | `model.py` |
| JSON Schema (v1.0) | `schemas/qvf_basis_v1.schema.json` |
| Schema validation | `validator.py` |
| BSE JSON importer | `importer_bse.py` |
| G94 exporter | `exporter_g94.py` |
| ORCA exporter | `exporter_orca.py` |
| NWChem exporter | `exporter_nwchem.py` |
| QVF-Basis serializer (JSON + ZIP) | `qvf_basis.py` |
| CLI tool | `cli.py` |
| Tests (14) | `tests/test_basis_toolkit.py` |
| Sample files (3) | `samples/` |

## Quick start

```python
from vibeqc.basis_toolkit import (
    from_bse_file, to_g94, to_orca, to_nwchem,
    save_qvf_json, save_qvf, load_qvf,
    validate_basis_file,
)

# Import from BSE JSON
basis = from_bse_file("samples/sto-3g_h2o.bse.json")

# Export to legacy formats
with open("sto-3g.g94", "w") as f:
    f.write(to_g94(basis))

# Save as QVF-Basis
save_qvf_json(basis, "sto-3g.qvf.json")   # plain JSON
save_qvf(basis, "sto-3g.qvf")             # packaged ZIP

# Round-trip
basis2 = load_qvf("sto-3g.qvf")
assert basis.name == basis2.name

# Validate
errors = validate_basis_file("sto-3g.qvf.json")
assert errors == []
```

## CLI

```bash
# Validate
python -m vibeqc.basis_toolkit.cli validate sto-3g.qvf.json

# Convert
python -m vibeqc.basis_toolkit.cli convert input.bse.json -o output.g94 -f g94
python -m vibeqc.basis_toolkit.cli convert input.bse.json -o output.qvf.json

# Show loss report
python -m vibeqc.basis_toolkit.cli loss -f g94 input.bse.json

# Inspect
python -m vibeqc.basis_toolkit.cli inspect input.qvf.json
```

## Architecture

### Canonical model (`model.py`)

All data flows through `BasisSetData`. Importers produce it; exporters
consume it; the QVF serializer round-trips it. The model is:

```
BasisSetData
├── schema_version, name, description, role, basis_family
├── elements: dict[str, ElementBasis]
│   └── ElementBasis
│       ├── element (symbol)
│       ├── shells: list[Shell]
│       │   └── Shell
│       │       ├── angular_momentum (list[int], e.g. [0], [1], [0,1])
│       │       ├── harmonic_type (spherical | cartesian)
│       │       └── primitives: list[Primitive]
│       │           └── Primitive (exponent, coefficient)
│       └── ecp_id: Optional[str]
├── ecps: Optional[dict[str, ECPEntry]]
├── references: list[Reference]
└── provenance: Optional[Provenance]
```

### Data flow

```
BSE JSON ──→ importer_bse.py ──→ BasisSetData ──→ exporter_g94.py ──→ .g94
                                      │            exporter_orca.py ─→ .orca
                                      │            exporter_nwchem.py → .nwchem
                                      │
                                      └──→ qvf_basis.py ──→ .qvf.json
                                           qvf_basis.py ──→ .qvf (ZIP)
```

### QVF-Basis format

Two modes, one model:

| Mode | Extension | Structure |
|---|---|---|
| **Text** | `.qvf.json` | Single JSON file -- the serialized `BasisSetData` |
| **Packaged** | `.qvf` | ZIP archive with `manifest.json` + `basis/basis.qvf.json` |

The `.qvf.json` is the canonical single-file representation. The `.qvf`
wraps it in a QVF container with manifest, checksums, and optional
reference payloads.

### Schema

JSON Schema lives at `schemas/qvf_basis_v1.schema.json`. It validates:
- Required fields (`schema_version`, `name`, `role`, `elements`)
- Element symbols (all 118)
- Shell constraints (angular_momentum length 1-2, L ≤ 12)
- Primitive constraints (exponent > 0)
- ECP structure
- Reference keys

### Lossiness

Legacy text formats (G94, ORCA, NWChem) lose 7 metadata fields.
The `LossReport` API tracks exactly what was dropped. See
`docs/qvf_basis_conversion_matrix.md` for the full fidelity matrix.

## Design decisions

See `docs/design_qvf_basis.md` for the full architecture review,
format comparison, and design rationale.

Key decisions:
1. **BSE JSON is the compatibility baseline**, not G94.
2. **G94 is an export target only** -- never the canonical storage format.
3. **QVF-Basis v1 is basis-set-only** -- narrow scope ensures deep quality.
4. **Dual mode** (`.qvf.json` + `.qvf`) serves both Git workflows and
   curated distributions.
5. **Structured metadata** (role, references, provenance, versioning) is
   first-class, not an afterthought.

## Running tests

```bash
cd /path/to/vibeqc
.venv/bin/python -m pytest python/vibeqc/basis_toolkit/tests/ -v
```

## Adding a new format

1. **Importer**: Create `importer_<format>.py`. Parse the external format
   into `BasisSetData.from_dict()`.
2. **Exporter**: Create `exporter_<format>.py`. Accept `BasisSetData` and
   produce text/bytes.
3. **Loss report**: Implement `loss_report_<format>(data) -> LossReport`.
4. **Sample**: Add a sample file in BSE JSON format to `samples/`.
5. **Test**: Add round-trip and format-specific tests to
   `tests/test_basis_toolkit.py`.
6. **CLI**: Register the format in `cli.py`'s `cmd_convert`.

## Future directions

- **QVF-Matrix** profile for wavefunction/binary data
- **BSE API fetcher** for on-demand basis retrieval
- **Direct `BasisSetData` → libint `BasisSet` constructor** to bypass G94
- **G94 importer** for reverse conversion from legacy formats

## License

MPL 2.0 -- see the vibe-qc root `LICENSE` file.
