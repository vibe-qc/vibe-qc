# QVF-Basis -- Architecture Review & Design Decision

> Canonical basis-set representation for vibe-qc.
> Date: 2026-06-23

## 1. The Problem: G94 as Canonical Format

vibe-qc currently stores all basis sets as Gaussian94 (`.g94`) text files.
This is the format libint's native parser consumes, and it works, but it
is a poor canonical representation.

### 1.1 Concrete shortcomings of G94

| Shortcoming | Detail |
|---|---|
| **No structured metadata** | Name, references, provenance, roles, ECP linkage, and revision history are either absent or embedded in free-text comments that no parser consumes. |
| **No machine-readable element map** | A `.g94` file is a concatenation of per-element blocks with `****` separators. There is no index, so finding element Z=79 requires a full scan. |
| **Ambiguous contraction model** | Segmented vs. general contractions are implicit. An SP shell is a convention, not a typed field. |
| **No explicit spherical/Cartesian flag** | The harmonic type is inferred from shell labels (`S`, `P`, `SP`, `D`, `F`, `G`), but `D` could mean 5d or 6d depending on the consuming program's defaults. |
| **No ECP integration** | ECPs live in separate files with ad-hoc formats. There is no standard way to say "this orbital basis uses that ECP." |
| **No auxiliary/fit role tagging** | A `.g94` for a RI-J auxiliary basis looks identical to an orbital basis. The consumer must know the role from context or naming convention. |
| **Lossy for general contractions** | G94's per-shell line format forces every primitive to be repeated for every contracted function sharing it. A general contraction with 20 primitives and 15 contracted functions duplicates 300 numbers. |
| **No checksums or integrity** | A corrupted `.g94` is silently parsed with wrong numbers. |
| **Poor diffability for segmented sets** | A one-line change to a contraction coefficient appears as one line in a diff, but the lack of structure means no tool can say "the p exponent for oxygen changed from X to Y." |
| **No versioning** | There is no schema version, so a parser cannot know whether it understands the format. |
| **Free-text comment convention** | The `!` comment lines have no standard structure. Some files use them for references, some for basis-set family names, some leave them blank. |

### 1.2 The real cost

Every program that reads `.g94` has its own parser with its own quirks.
libint, PySCF, ORCA, and NWChem all parse `.g94` slightly differently
(whitespace tolerance, comment handling, ECP linkage). This is not a
standard -- it is a collection of compatible-enough parsers.

## 2. Survey of Alternatives

### 2.1 Gaussian94 / G94

**Role**: De facto interchange format. Every QC code reads it.
**Verdict**: Export target only. Not a canonical format.

### 2.2 ORCA-style input

ORCA's `%basis` block is an input format, not a storage format. It
mixes basis-set data with program-specific keywords (`NewGTO`, `STO`,
`ECP`). No independent schema exists. Same problems as G94 with extra
vendor lock-in.

**Verdict**: Export target only.

### 2.3 CRYSTAL-style input

Per-element plain-text files with integer-coded shell types. Dense
and unambiguous, but CRYSTAL-specific (LAT codes 0-5, the 200+Z ECP
convention). No metadata model.

**Verdict**: Import source (vibe-qc already parses it via
`basis_crystal.py`). Not a canonical format.

### 2.4 NWChem-style input

NWChem's `basis` block uses a labeled-segment approach:

```
C    S
    6665.0000000     0.0006920
    1000.0000000     0.0053290
C    P
     28.8700000     0.0288300
      ...
```

Slightly more structured than G94 (explicit element label per block)
but otherwise equivalent. Same lossiness.

**Verdict**: Export target only.

### 2.5 Basis Set Exchange (BSE) exports

BSE exports multiple formats (Gaussian, ORCA, NWChem, Molpro, Turbomole,
JSON, …) from a single curated database. The **BSE JSON format** is
the most interesting:

- Structured JSON with explicit element map
- Tagged shell angular momentum as integers
- Explicit `"function_type"` field (gto, sto, numerical)
- Per-element ECP linkage via `"ecp_electrons"`
- Metadata block with name, description, version, references
- Family/role annotations

**This is the best interoperability anchor available today.** BSE is
the closest thing to a universal basis-set registry, and its JSON
export is a well-structured machine-readable format.

### 2.6 QCSchema BasisSet

The MolSSI QCSchema project defines a JSON Schema for basis sets as
part of its broader QC data model. It is strongly typed, extensible,
and designed for programmatic consumption. Its `BasisSet` schema
includes:

- `center_data`: per-atom basis with element-agnostic shell lists
- `atomic_shell`: angular momentum, harmonic type, exponents, coefficients
- `ecp_potentials`: explicit ECP data blocks
- `name`, `description`, `schema_version`

QCSchema is more opinionated than BSE JSON and has a formal JSON
Schema. It is the strongest structured-schema influence available.

### 2.7 XML / HDF5 / other binary formats

- **XML (CML, NMX)**: Heavy, verbose, no ecosystem adoption in QC basis sets.
- **HDF5**: Excellent for large grids and wavefunctions but poor Git
  diffability and overkill for basis-set data (which is ~KB per element).
- **MessagePack / CBOR**: Good binary JSON alternatives but no QC tooling.

**Verdict**: Not suitable as primary basis-set format. HDF5 may be
considered for QVF-Matrix (future wavefunction profile), not QVF-Basis.

## 3. The Verdict

### 3.1 Is there a universal standard today?

**No.** There is no single file format that every QC code reads for
basis sets. G94 is a de facto interchange format by accident, not
design. BSE is the closest thing to a universal registry, and QCSchema
is the strongest structured-schema candidate.

### 3.2 Is BSE the best interoperability anchor?

**Yes.** BSE is the curated source of record for basis sets. Its JSON
export provides structured data with explicit metadata, element maps,
and ECP linkage. vibe-qc should treat BSE-compatible structured JSON
as the canonical import format.

### 3.3 Recommendation for vibe-qc

vibe-qc should adopt a **canonical structured JSON model** aligned
conceptually with QCSchema BasisSet concepts, with BSE JSON as the
primary import source. G94 and other legacy text formats are export
targets only.

The canonical format should be QVF-Basis -- a profile of the QVF
container family (see § 4).

## 4. QVF-Basis Design

vibe-qc already has QVF v1 -- a ZIP-based container format for
visualization data (`.qvf`). The QVF architecture provides:

- `manifest.json` with `qvf_version`, `source`, `sections[]`, `extensions`
- Per-section `kind` taxonomy, `members` with paths, formats, and sha256
- Random access via ZIP member paths
- Formal JSON Schema for validation
- Extensions mechanism for vendor data
- Compact binary `.dat` payloads for large grids

QVF-Basis is a **profile** of the QVF family: same container
architecture, same manifest conventions, but with basis-set-specific
canonical section kinds.

### 4.1 QVF dual mode

| Mode | Extension | Structure | Use case |
|---|---|---|---|
| **Text** | `.qvf.json` | Single plain JSON file | Git-friendly storage, CI diffing, small sets |
| **Packaged** | `.qvf` | ZIP container with manifest | Curated distributions, ECP + basis bundles, large libraries |

The text mode is the canonical single-file representation: one JSON
file containing the full basis-set data model. The packaged mode wraps
it in a ZIP with a manifest, checksums, and optional attachments
(references, ECP data files, provenance records).

A `.qvf.json` file can always be promoted to a `.qvf` archive by
adding a manifest. A `.qvf` archive's core basis payload is a valid
`.qvf.json` file.

### 4.2 QVF v1 scope: basis-set-only

**Yes -- QVF v1 should be basis-set-focused.** This is the right
boundary because:

1. **Narrow scope, deep quality.** Basis sets are a well-defined
   domain with clear semantics. Getting the model right for basis
   sets doesn't require solving wavefunction representation.
2. **Immediate utility.** Every vibe-qc calculation consumes a basis
   set. A clean canonical format is immediately useful.
3. **Avoids QVF v1 overlap.** The existing QVF v1 container already carries
   orbitals, densities, and wavefunction data for visualization. A
   basis-set profile doesn't compete with that.
4. **Testable independently.** Round-trip fidelity tests need only
   basis-set data, not full QC outputs.

Future QVF profiles (QVF-Matrix, QVF-ECP, QVF-Grid) can extend the
family without breaking the basis-set profile.

### 4.3 Section kinds (canonical)

| Kind | Description |
|---|---|
| `basis` | The core basis-set data: per-element shells, exponents, coefficients, angular momenta, harmonic type |
| `basis.metadata` | Name, family, description, version, references, basis-set role |
| `ecp` | Effective core potential data per element |
| `ecp.metadata` | ECP name, n_core_electrons, reference |
| `provenance` | Origin (BSE, manual, program-generated), retrieval date, DOI, checksum of source |
| `references` | Structured citation data (DOI, BibTeX, description) |
| `auxiliary` | Auxiliary/fitting basis data (RI-J, RI-K, CABS) linked to parent orbital basis |

### 4.4 Data model (canonical)

```python
@dataclass
class Primitive:
    """One primitive Gaussian."""
    exponent: float
    coefficient: float  # contraction coefficient, G94 convention (unnormalized)

@dataclass
class Shell:
    """One contracted shell."""
    angular_momentum: list[int]  # e.g. [0] for S, [1] for P, [0,1] for SP
    harmonic_type: Literal["spherical", "cartesian"]
    primitives: list[Primitive]

@dataclass
class ElementBasis:
    """Basis for one element."""
    element: str  # symbol or Z
    shells: list[Shell]
    ecp_id: Optional[str] = None

@dataclass
class ECPEntry:
    """ECP data for one element."""
    element: str
    n_core_electrons: int
    angular_momentum_max: int
    potentials: list[ECPPotential]  # per-L potential terms

@dataclass
class BasisSetData:
    """Top-level basis set object."""
    schema_version: str  # "1.0"
    name: str
    description: str
    role: Literal["orbital", "auxiliary", "fitting", "ecp"]
    basis_family: Optional[str]
    elements: dict[str, ElementBasis]  # keyed by element symbol
    ecps: Optional[dict[str, ECPEntry]]
    references: list[Reference]
    provenance: Optional[Provenance]
```

### 4.5 JSON Schema

A standalone JSON Schema file `basis_toolkit/schemas/qvf_basis_v1.schema.json`
validates the text-mode `.qvf.json` format. The schema enforces:

- Required fields: `schema_version`, `name`, `role`, `elements`
- Per-element symmetry: element keys are valid symbols, shells are
  non-empty, primitives are non-empty arrays
- Type constraints: exponents > 0, harmonic_type enum, role enum
- Optional ECP linkage

### 4.6 Compatibility with BSE and QCSchema

The QVF-Basis model is designed to accept BSE JSON as its primary
import source. A mapping layer translates BSE fields:

| BSE JSON field | QVF-Basis field |
|---|---|
| `name` | `name` |
| `description` | `description` |
| `basis_family` | `basis_family` |
| `basis_set_role` | `role` |
| `elements[X].element_number` | `elements` key |
| `elements[X].electron_shells[].angular_momentum` | `shells[].angular_momentum` |
| `elements[X].electron_shells[].exponents` | `shells[].primitives[].exponent` |
| `elements[X].electron_shells[].coefficients` | `shells[].primitives[].coefficient` |
| `elements[X].electron_shells[].function_type` | validated as `gto` |
| `elements[X].ecp_electrons` | linked `ecps[key]` |
| `references` | `references` |
| `version` | `provenance.basis_version` |

The QCSchema `BasisSet` model maps similarly but uses a center-based
rather than element-based organization. The QVF-Basis importer accepts
both and normalizes to the element-based model.

### 4.7 Lossiness annotations

When exporting to legacy text formats, some information is lost:

| Information | G94 | ORCA | NWChem |
|---|---|---|---|
| Basis name | Lost (filename only) | Lost | Lost |
| Basis family | Lost | Lost | Lost |
| Role (orbital/aux/fit) | Lost | Lost | Lost |
| References (DOI, BibTeX) | Lost | Lost | Lost |
| Provenance / version | Lost | Lost | Lost |
| Spherical/cartesian flag | Implicit | Implicit | Implicit |
| General contractions | Expanded (lossy) | Expanded (lossy) | Expanded (lossy) |
| SP shell distinction | Preserved (G94 convention) | Preserved | Preserved |
| ECP linkage | Separate file | Inline keyword | Separate block |

The exporter API includes a `loss_report()` method that returns which
fields were dropped.

### 4.8 Versioning

QVF-Basis uses strict semantic versioning for the schema:

- **Major** (1.x): breaking changes to required fields or data model
- **Minor** (x.1): new optional fields, new section kinds
- **Patch** (x.x.1): documentation, validation rule clarifications

The `schema_version` field in the data file declares what version of
the schema the file was written against. Consumers check this and
either accept (same major) or reject (different major).

### 4.9 MIME type

Proposed: `application/x-qvf+basis+json` for text mode,
`application/x-qvf+basis+zip` for packaged mode.

### 4.10 Deterministic ordering

For reproducible archives, the packaged `.qvf` uses:
- Element keys sorted by atomic number ascending
- Shells in order of appearance in the source
- Primitives sorted by exponent descending (standard convention)
- ZIP member paths in lexicographic order
- Manifest written last to ensure zip64 EOCD record is deterministic
  (QVF v1 convention)
