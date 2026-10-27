# QVF governance

QVF (the *Quantum Visualization Format*) is an **open standard** for
quantum-chemistry visualization and analysis data. This document describes how
the format is stewarded, versioned, and evolved, so that adopters can rely on it
as a stable, multi-vendor interchange format rather than a single code's captive
output.

## Origin and stewardship

QVF was **created by the [vibe-qc](https://vibe-qc.com) project** and is
**stewarded by it**. That origin is a matter of record, not something the format
hides: vibe-qc designed the container model, the section-kind vocabulary, and the
reference implementations, and it maintains the specification and the JSON
Schema.

Stewardship here means the vibe-qc maintainer is the current custodian of the
spec and registry — merging changes, cutting versions, and arbitrating the
vendor→canonical promotion process. It does **not** mean the format is
vibe-qc-specific:

- The **specification and reference implementations are Apache-2.0** (this
  toolkit), so anyone can implement a producer or consumer,
  including in a proprietary code, with no obligation back to vibe-qc.
- The format is **producer-neutral by design**: `source.program` names any code,
  and no canonical section kind is vibe-qc-specific. vibe-qc's own
  non-standard data lives under its `x_vibeqc.*` vendor namespace, exactly like
  any other adopter's.
- Governance decisions are made **in the open** (spec + schema + registry are
  public and versioned), and the process below is the same for everyone.

As independent adopters ship QVF, stewardship is intended to broaden — see
"Evolving the governance" below.

### QVF in the vibe-qc family

Within the family of programs the steward maintains — vibe-qc (the
calculations), vibe-view (the viewer), vibe-sketch (the structure and reaction
sketcher), vibe-queue (the job queue) and this toolkit — **QVF is the standard
format**. Each program reads and writes it natively, and the programs exchange
data through it: what one saves, the others open, each for what it does.
vibe-view shows what vibe-sketch and vibe-qc save, and vibe-sketch draws the
molecules of a vibe-view or vibe-qc archive. Other formats (XYZ, PDB, CIF,
SDF, cube, Molden, KET, program inputs and outputs) remain welcome as imports
and exports beside it, never as the format the programs hand each other.

The family has no privilege in the format: its programs follow the same
specification and consumer rules (spec § 8) as every other producer and
consumer, put their own data in their vendor namespaces (`x_vibeqc`,
`x_vibesketch`), and bring a question the format does not answer to the
specification instead of agreeing on it privately. A gap one program meets in
another's archive is a bug of the program or a gap of the specification, and
is fixed where it belongs.

**This repository is the single source of the format** (decided 2026-10-15,
[decision](#2026-10-15--one-source-for-the-format-what-the-family-already-writes-and-reads-11)).
The specification, the JSON Schema, the registry and the reference
validators change only here. Each program of the family pins
one qvf release tag, in one place. Any copy it vendors, such as the schema or
spec text, is byte-identical to that tag's file, and the program's CI checks
it against a clone of the tag. No program keeps a derived or edited schema;
that includes the withdrawn `qvf_version` 2, which consumers read by spec
§ 3.1 rather than with a v2 schema of their own. A field a program needs is
proposed here first and released before the program writes it.

## What is under governance

1. The **specification** (`spec/qvf-format-spec.md`) and the machine-readable
   **JSON Schema** (`spec/qvf_manifest.schema.json`), which is the single source
   of truth for the manifest contract.
2. The **registry** (`registry.json`) of canonical section kinds and registered
   vendor namespaces.
3. The **versioning policy** and the **vendor→canonical promotion process**.

## Versioning policy

- `qvf_version` is an integer in every manifest. **v1 = `1`.** It changes only on
  a **breaking** change (a removed kind, a required-member change, an
  incompatible unit change). A consumer that only understands `qvf_version = 1`
  must refuse a file with a higher version.
- The format grows **additively within a major version**: new canonical kinds and
  new *optional* members may be added without bumping `qvf_version`. Consumers
  detect capability by supported-kind checking and the `extensions` block, not by
  a minor-version number.
- The schema `$id` carries the major version
  (`https://vibe-qc.org/spec/qvf/1/manifest.schema.json`).
- Changes to the format are recorded in the toolkit's [`CHANGELOG.md`](CHANGELOG.md)
  and reflected in `registry.json`.

## Version history and decisions

Governance decisions that affect the on-disk contract are recorded here, in the
open. Recording a mistake is part of the process working, not evidence against it.

### 2026-10-16 — Static XSF conversion (WP-48)

The Unreleased Python converter maps static XSF primitive geometry and scalar
fields into existing structure/volume members. XSF grid coordinates convert
from angstrom to canonical bohr; scalar values are reordered without
resampling. Structural periodicity comes from XSF, while field periodicity
requires a caller assertion checked against the cell. Inclusive XSF boundary
samples are retained. Periodic grids with omitted endpoints are refused on
export instead of synthesizing values. Unsupported records fail closed and
conversion losses are reported. This toolkit API addition does not change
format version 1, schema, registry, validators or finding codes.

### 2026-10-16 — Toolkit 0.5.0 compatibility

Toolkit 0.5.0 packages the additive C++/C/Python APIs, optional Fortran
bindings, CLI and converters recorded below. Format version 1, schema identity,
canonical kinds and validator finding codes remain unchanged. The advisory
`x_ascent` namespace is included. The distribution remains `qvf-writer` with
legacy imports available; NumPy array APIs use the explicit `numpy` extra.
Consumers continue to pin a release and vendor exact files from that tag.
This release does not authorize media-type registration or schema hosting changes.

### 2026-10-15 — Optional Fortran binding (WP-45)

The Unreleased toolkit adds an optional Fortran layer over existing ABI 1.
Generated raw interfaces preserve the C signatures and ownership rules;
convenience writers copy column-major inputs into the existing canonical
row-major layouts without changing units or scientific meaning. This is a
source/API addition. Format version 1, schema, registry, validators, finding
codes and generated contract tables are unchanged. Compiler/platform release
coverage remains separate from the focused development checks.

### 2026-10-15 — Native library adoption (WP-49)

The Unreleased toolkit adds pkg-config metadata and explicit CMake test/example
options. Embedded builds default to the library alone; standalone development
retains its existing checks. Installation metadata propagates existing language
and shared-library requirements and supports relocation with relative install
paths. This is a packaging change: ABI 1, format version 1, validators and
generated contract tables are unchanged. Package registry publication remains
separate from this source change.

### 2026-10-15 — Initial text converters (WP-48)

The Python toolkit gains explicit cube and XYZ/extXYZ conversions using
existing canonical structure, trajectory and scalar volume members. This is
an API addition, not an on-disk contract change. Structure/trajectory lengths
are angstrom; cube atoms convert from bohr while grid vectors remain bohr.
Periodic cube imports require an explicit user choice and use the existing
endpoint span rule; grid values are not resampled. No physical quantity,
charge, multiplicity, job or scientific result is inferred. Unsupported
input properties are refused; known conversion omissions are reported.
Exports select an unambiguous section and preserve the original archive.
The broader converter package remains in progress; these APIs are Unreleased.

### 2026-10-15 — ASCENT dataset namespace (#13)

Register `x_ascent` for ASCENT's non-critical `x_ascent.dataset` sections.
Registration is advisory under spec § 7.3 and records an existing producer;
it does not promote a kind or change the schema, writers or finding codes.
The root `extensions` declaration carries version `1`. Ordered `records`
batches and `metadata` retain ASCENT's task/reference semantics, original
units and provenance. Canonical `structure` sections carry geometries and
molecular identity; task spin labels and dataset-wide properties are not
reinterpreted as molecular multiplicities or an archive SCF energy.
Generic consumers report the private section as unsupported and can still
open its structures. The private payload remains the producer's contract.

### 2026-10-15 — Complete additive C ABI (WP-44)

The C binding exposes the established writer, reader, validator and editor
facades through opaque handles. ABI 1 preserves every existing C signature;
new constructors return status and an output handle. Fixed-layout tensor
inputs carry explicit dtype, shape, byte order and byte length. Nested
payloads remain JSON strings, and generated named entries cover every
canonical registry kind. Periodic structures use the existing row-vector
lattice and per-axis `pbc` contract. No format field or finding code changes.

Errors use fixed handle-owned storage; constructor/path errors use thread-local
storage. C++ exceptions are caught at the boundary. Borrowed strings, arrays,
callbacks and separately owned editor writer handles have documented lifetimes.
Shared builds export the API with ABI version 1; the toolkit version continues
to follow `VERSION`. Incompatible layouts require new types/functions or a
future ABI version. This completes the binding direction already specified in
WP-44, with no new maintainer-owned format decision.

### 2026-10-15 — Typed C++ writing and editing (WP-42)

The additive `Writer` facade shares the existing `QvfWriter` streaming
backend; the compatibility API remains supported. Canonical kind classes
and nested payload records derive from the normative schema and registry.
They use the Reader's `from_section` protocol, preserve open payload fields,
and distinguish absent fields from explicit null. `TensorView` borrows
contiguous producer storage and converts endian/dtype in bounded chunks;
typed reads return owning tensors. Schema checking at write time complements
the complete archive validator rather than asserting cross-section semantics.

Editor append/removal follows the established update lifecycle. Atomic
replacement is an explicit option: work in a private sibling directory and
rename only a completed archive, preserving the original on abort or rename
failure. Repack streams live members to remove superseded entries. These are
library API additions; the on-disk format, schema, registry and finding-code
contract are unchanged. Native verification covers every canonical kind's
typed round trip, callback/memory member equality, conversion boundaries,
update rollback and Python validation of edited/repacked output.

### 2026-10-15 — C++ Reader and validation facades (WP-43)

The additive `qvf::Reader` API uses the existing lossless manifest model,
`Tensor` / `DType` and ZIP container. Payload access is lazy; streaming
callbacks receive borrowed little-endian chunks and must discard partial
results on a final integrity failure. CRC-32 remains mandatory, including
when the caller disables SHA-256 verification. `get<K>` delegates to the
kind's `from_section` decoder; canonical kind classes belong to WP-42.

Validation levels are cumulative. Container and Manifest stages avoid
reading payload bytes that belong to later stages. Full validation reuses
the existing validator and remains the default for both APIs. Reports use
the same finding codes and limits. The old `QvfReader` signatures remain
supported. This library API addition changes no on-disk contract, schema,
registry entry, or `qvf_version`.

### 2026-10-15 — Command-line archive interoperability (WP-47)

The Python distribution now exposes `qvf` inspection, validation, extraction,
comparison, repack, scrub, conformance and schema commands. Its validation
levels share the established reference checks and follow the C++ cumulative
stage boundaries; the default full report and existing finding codes remain
unchanged. The compatibility validator command and imports are retained.

This is a toolkit interface change, with no schema, registry or format-version
change. Output commands require new destinations. Scrub is an explicit,
reported transformation that retains the input, removes the deprecated hostname
and author contact fields, applies only caller-selected run-record patterns,
and reports skipped content. It preserves job specifications, rewrites member
hashes, drops orphan data and records a new archive UUID with derivation. Its
scope does not change publication-scanner policy or claim complete privacy
certification. NumPy and JSON Schema dependencies remain optional as before.

### 2026-10-15 — Python package and archive facades (WP-46)

The Python implementation now ships the `qvf` package in the existing
`qvf-writer` distribution, as the maintainer decided. The package API adds
streaming writes, lazy typed reads with a standard-library buffer path,
append editing, optional atomic commits and repacking. The top-level
`qvf_writer` and `qvf_reader` imports preserve their implementation module
identity and command-line behavior while emitting `DeprecationWarning`.
NumPy is an explicit optional extra for numeric convenience operations;
JSON Schema validation remains the `validate` extra. Schema and generated
kind tables are package data/source derived from the same normative inputs.

This is an additive toolkit API change. It introduces no archive fields,
finding codes, version rule or schema identity. Editors preserve job
specifications verbatim; binary sinks reject incomplete or oversized members
before a new manifest is committed. All existing semantic validation is
shared with the compatibility reader, avoiding a second set of format rules.

### 2026-10-15 — Distribution, schema identity and media-type preparation

The maintainer decided to keep the Python distribution name `qvf-writer`
for WP-46 and to preserve the top-level `qvf_writer` and `qvf_reader`
imports. The schema `$id` stays
`https://vibe-qc.org/spec/qvf/1/manifest.schema.json` pending hosting;
this decision does not claim that the URL is served. An IANA registration
[draft for `application/vnd.vibe-qc.qvf+zip`](docs/iana-media-type-registration.md)
is prepared for maintainer review. Submission is not authorized and the media type remains unregistered.

### 2026-10-15 — Unrestricted natural transition orbitals (#12)

The maintainer adopted the two-section proposal for an unrestricted state's
NTOs: one `wavefunction.gto` hole section and one particle section, each with
`spin: "unrestricted"`, `alpha` / `beta` metadata and corresponding
coefficient members. Rows pair within each spin, with equal counts and
equal nonnegative, descending squared singular-value weights. Alpha and
beta may have different counts; an empty spin channel is valid. The state's
`nto_ref` and the unique particle lookup are unchanged. No NTO convention
for restricted-open references is decided here.

Martin, *J. Chem. Phys.* **118**, 4775 (2003),
[doi:10.1063/1.1558471](https://doi.org/10.1063/1.1558471), equations 4–9,
describes the corresponding-orbital transformation and separately diagonal
spin components. For a normalized state, the weights of **both spins
together** sum to one. As Martin notes, full TDDFT/RPA de-excitation terms
can change that sum; the existing normalized-state qualification remains.
There is no normalization flag and validators do not demand a unit sum.

This is a clarification within v1 of the previously unspecified unrestricted
NTO case, using existing fields, not a new kind or version. The schema now
requires the canonical NTO state fields and transition-weight semantics
when `nto_role` is present. Deprecated metadata without `nto_role` retains
its v0.3.0 acceptance and is exempt from the new channel/pair checks; no
role is inferred from section order. Writers always emit canonical fields. Writers check channel row counts, finite nonnegative ordered
weights before mutation; validators resolve forward state/partner
references and check pair consistency. Missing or ambiguous partners and invalid
state references use `E_REF_UNRESOLVED`; spin, row and weight inconsistencies
use `E_MEMBER_MALFORMED`, at the NTO section's `members/mo_metadata`.
Restricted and deprecated legacy NTO shapes keep their existing acceptance.
All three implementations share positive and negative corpus cases, including
unequal spin counts, empty beta, a normalized combined sum, non-unit TDDFT
weights and malformed pairs. No registry kind is promoted.

### 2026-10-15 — One source for the format; what the family already writes and reads (#11)

**Background.** By 2026-10-14 the programs of the vibe-qc family held four
schema copies that disagreed: vibe-qc vendored the schema of v0.1.0-docs.1 and
a `qvf_version` 2 schema derived from it, and vibe-view vendored the schema of
v0.2.0 and a frozen v2 schema. vibe-qc wrote four kinds of data that the 0.2.0
validator rejects: run against 2,000 archives written by vibe-qc 0.15–0.17,
`validate_qvf` failed 463 of them (23 %), for a `job.spec` `functional` of
`null` (336 archives), a `job.spec` `basis` of `""` (143), the `mo_metadata`
keys `excitation_index` and `excitation_energy_ev` of natural transition
orbitals (116), and `iao_charges` and `localization_reference_basis` of
localized orbitals (60); an archive may have several. vibe-view meanwhile read
three `viewer_defaults` hints the specification did not record (#6), and the
format could not say which structure a standalone `bonds` section belongs to
(#4). qvf's own text also disagreed about version 2: spec § 3.1 and the three
validators read a v2 archive as the current v1, while `spec/legacy/README.md`
and `registry.json` called the frozen v2 schema "a strict superset of v1",
which holds only against v0.1.0. Toolkit 0.3.0 merges the drift into the
format, and the family takes the format from here only. Every change is
additive under v1.

1. **qvf is the single source** (see "QVF in the vibe-qc family" above). The
   specification, schema, registry and reference validators change only in
   this repository. A family program (vibe-qc, vibe-view, vibe-sketch,
   vibe-queue) pins one qvf release tag in one place; any copy it vendors is
   byte-identical to that tag's file and checked in its CI against a clone of
   the tag; no program keeps a derived or edited schema, the v2 rule
   included. A field a program needs is proposed here first and released
   before the program writes it.
2. **`localization_reference_basis`** (spec § 5.3), a new optional
   `MoMetadataPayload` field: the non-empty name of the minimal reference
   basis of free-atom AOs the intrinsic atomic orbitals were built from. IAOs
   are constructed from the occupied orbitals and such a minimal basis, and
   the IAOs, IAO charges and IBOs depend on the choice (G. Knizia, *J. Chem.
   Theory Comput.* **2013**, 9, 4834, which compares MINAO with Huzinaga's
   MINI and ANO-RCC functions); a producer names it beside
   `localization_method`, as vibe-qc does with `"mini"`.
3. **Three deprecated `mo_metadata` keys** (spec § 5.3), valid so that
   existing archives keep validating, annotated `"deprecated": true` in the
   schema as `provenance.hostname` and the legacy unit spellings were:
   `iao_charges` (canonical carrier: the `atom_properties` member
   `iao_charge`), `excitation_index` (canonical: `state_index` with
   `state_ref`) and `excitation_energy_ev` (canonical: the
   `excitation_energy` of that state in the `excited_states` section).
   Producers SHOULD NOT write them. `MoMetadataPayload` stays strict for every
   other key.
4. **`null` for a `job.spec` `functional` or `basis`** (spec § 5.9): the
   method takes none (Hartree–Fock has no functional, a semiempirical or
   tight-binding method no basis set), distinct from an absent key, which
   leaves the choice to the engine like an absent `method`. The empty string
   is a deprecated spelling of `null` (vibe-qc 0.15.136–0.17.15 wrote
   `basis: ""`); the schema's `anyOf` marks that branch `deprecated`.
5. **The `viewer_defaults` hints `energy_window`, `camera` and `crossfade`**
   (spec § 3.4, #6), settled as conservatively as `render` and
   `transfer_function` were (#5, 2026-10-07): recorded in the specification
   with the shapes and rules of vibe-view's parser, and not typed by the
   schema, so that a malformed hint never fails schema conformance and no
   consumer refuses a file over one. `energy_window` is a per-section key of
   `bands` and `dos.*` sections: `[min, max]` in eV relative to the Fermi
   level, `max > min`; consumers MAY accept vibe-view's object forms. `camera`
   and `crossfade` sit at the top level, in the key space of section ids, so
   both are **reserved as section ids** beside `auto_open` and `bookmarks`
   (`E_RESERVED_ID`), and the schema lists them as `viewer_defaults` members
   without constraining their values. `camera` has the shape of
   `$defs/Camera`, the camera of a bookmark, which is what vibe-view applies
   (it also tolerates a camera with less); `crossfade` is `{a, b, blend}`
   with `blend` from 0 (`a` alone) to 1 (`b` alone), default 0.5, and a
   consumer reads a missing or malformed `blend` as 0.5. Validators report no
   finding for any of the three hints.
6. **`structure_ref` on a standalone `bonds` section** (spec § 5.1, #4), the
   generic section field of § 4.1 given its meaning for `bonds`: present, the
   indices address the atoms of the `structure` section it names; absent, the
   bonds belong to the archive's first structure (§ 4.1, 2026-10-14).
   Producers that write several structures SHOULD set it. Validators resolve
   it like every section-level `structure_ref` (`E_REF_UNRESOLVED`) and check
   each pair's `i` and `j` against the atom count of the named structure
   (`E_MEMBER_MALFORMED`). A section without `structure_ref`, and a
   structure's own `bonds` member, are not counted, so no archive that
   validated before fails now unless it already named a structure its bonds
   do not fit. The Python and C++ writers' `add_bonds` take `structure_ref`.
7. **`x_vibesketch` in `registry.json`** describes what vibe-sketch writes
   since it moved its picture to the canonical `preview` kind: the JSON
   members `document`, `species` and `sdf`, and no vendor `preview` or
   `thumbnail`.
8. **One rule for the withdrawn `qvf_version` 2.** Spec § 3.1 stands: a v2
   archive is validated against the current v1 schema and reported as
   `W_VERSION_WITHDRAWN`. `spec/legacy/README.md` and the registry's
   `version_decisions` now say what the frozen file is: qvf 0.1.0's v1 schema
   with `qvf_version` 1 or 2, kept so that its `$id` resolves, used by no
   validator and no superset of the current v1 schema. Its bytes are
   unchanged.

**Warnings.** A deprecated key or spelling produces no finding, as
`provenance.hostname` and the legacy unit spellings do not: all three
validators accept it silently, and the schema's `deprecated` annotation and
the specification carry the advice to producers.

**Rationale for no bump.** Every change adds an optional field, accepts a
value that was rejected, records a hint, or reserves two ids. No member, kind,
default or unit changes. The verdicts that change: the 463 archives above, and
any archive with `null` or deprecated spellings, become valid; an archive
with a section whose id is `camera` or `crossfade`, or with a standalone
`bonds` section whose `structure_ref` names a structure its indices do not
fit, becomes invalid, as the new meaning requires. Neither occurs among the
2,000 archives.

**Compatibility.** Producers: additive; vibe-qc moves to the canonical
carriers (`iao_charge`, `state_index`, `null`) and keeps
`localization_reference_basis`. Consumers: read `null` as "none", `""` as
`null`, prefer the canonical carrier of a deprecated key, and apply
`bonds.structure_ref`. The family re-vendors the 0.3.0 schema and drops its
own v2 schemas. Validator authors: Python, C++ and JavaScript implement the
rules with the codes and locations of Appendix B. Conformance: five archives
join the corpus (`wavefunction_localized.qvf`, `job_set_methods.qvf`,
`deprecated_spellings.qvf`, `bonds_structure_ref.qvf`, `viewer_hints.qvf`)
and six the negative corpus.

### 2026-10-14 — Section ids carry no meaning; the first structure

**Decision.** A section id names a section and nothing more; `structure` is a
conventional id, not a role, and a consumer **MUST** find sections by kind and
reference (spec § 4.1). When a consumer shows one structure, or needs one for
an operation that names none, it uses the archive's **first structure**: the
first `structure` section that `viewer_defaults.auto_open` lists, else the
first `structure` section in `sections` order; it **SHOULD** offer the others.
Spec § 8.2 adds the consumer rule. **No `qvf_version` bump**: a
clarification.

**Background.** The specification never said which of several structures a
consumer shows, and a consumer took the section named `structure`. A
vibe-sketch document names its species `structure_<species id>`, as the
conformance archives `multi_structure.qvf`, `reaction_scheme.qvf` and
`job_set.qvf` name theirs, so that consumer showed nothing. Choosing by kind
and order needs nothing new from producers, and `auto_open` already existed for
a producer that wants another structure first.

**Compatibility.** Every archive keeps its meaning. A producer that relied on
the id `structure` being shown first keeps that behaviour when that section
comes first or is listed in `auto_open`. Consumers that looked a structure up
by id must change.

### 2026-10-14 — Sketch documents: chemical identity, `reaction.scheme`, `preview`, job sets

**Background.** vibe-sketch, the structure and reaction sketcher of the
vibe-qc family, saves its documents as QVF. A document holds several species,
the reactions that relate them, the calculations to run on them and a picture;
the drawing itself is Ketcher's KET. Until now the format had a place for
none of it except the 3-D structures, so the sketcher wrote everything into a
vendor section, `x_vibesketch.document`, and a reader that does not know
vibe-sketch saw a heap of unrelated molecules. An earlier question, whether a
*result* archive may carry the sketch that produced it as an attachment, is
superseded: the sketch is now the document, and results join it. The
decisions below give the portable parts of a sketch canonical carriers. Each
is additive under v1.

1. **`x_vibesketch` is registered; there is no canonical kind for 2-D
   drawings.** `registry.json` lists the namespace (owner vibe-sketch). An
   editable drawing has no producer-neutral representation (KET, CDXML and
   molfile 2-D coordinates are each one program's model) and one producer,
   so a canonical kind would freeze one program's format. The drawing stays
   in the vendor section; what it means goes into the canonical carriers of
   decisions 2, 3 and 5, so that every reader gets the chemistry and the
   picture and only the editing program needs the drawing. Spec § 9 records
   the gap.
2. **Chemical identity on `structure` sections** (spec § 5.1). Optional
   section fields `charge`, `multiplicity`, `name`, `smiles`, `inchi` and
   `inchikey`; the generic `label` is the species' compound label (`"1"`,
   `"3a"`). They sit on the section object, beside the biomolecule metadata,
   so a program lists the molecules of an archive, or searches archives by
   InChIKey, from the manifest alone. Absent means not stated, never
   neutral; a single-structure archive may still state charge and
   multiplicity in `provenance`, as before, and then the two **MUST** agree
   (`E_ROOT_INCONSISTENT`). The job's own charge stays the `job.spec`'s,
   authoritative for the calculation and allowed to differ (a vertical
   ionization). Validators check that a stated charge and multiplicity are
   possible for the atoms — a non-negative electron count, and unpaired
   electrons no more than it and of its parity — with the new error
   `E_CHARGE_MULTIPLICITY`. They do not compare line notations with atoms,
   which needs a cheminformatics toolkit; the spec says the identifiers
   describe the atoms and imply no atom order. Only the new fields are
   checked, so no existing archive changes verdict (a `job.spec` with an
   impossible default multiplicity, for example, stays unchecked).
3. **`reaction.scheme`, a canonical kind** (spec § 5.14), added directly by
   maintainer decision (Change process: a clear ecosystem need and a
   reference consumer). One strict JSON payload: `steps` in reading order,
   each with an `id`, `reactants` and `products` (at least one each),
   optional `reagents` (species drawn at the arrow), `arrow` (`forward`, the
   default, `equilibrium`, `retrosynthetic`, `resonance`), `conditions` (the
   text at the arrow, one entry per line), `yield_percent` (0–100) and
   `yield_text` (as printed). A participant is
   `{structure_ref, coefficient?}` naming a `structure` section; the
   coefficient makes the step a balanced equation, which reaction energies
   will need. **Roles are chemical, not positional**: `reactants` are
   consumed in the forward direction whatever the arrow, so the target of a
   retrosynthetic step is among its `products` and a producer that records
   the drawing by arrow side swaps the two for such a step; a quantity of a
   step is then always products minus reactants. The step key `energies` is
   **reserved** for reaction energies, to be specified when a producer
   computes them; until then it is a schema error and producers use `x_`
   keys. Conditions stay text: structured temperature, time or solvent
   fields are additive later. A species without a 3-D structure cannot be a
   participant in v1. Validators resolve every `structure_ref`
   (`E_REF_UNRESOLVED`) and reject duplicate step ids and a structure
   repeated within a role (`E_MEMBER_MALFORMED`).
4. **Job sets** (spec § 5.8, § 5.9). An archive **MAY** carry several
   `job.spec` sections; each is an independent job. `structure_ref` on
   `job.spec` already existed twice — the generic section field of § 4.1 and
   the payload field of § 5.9 — and a job without one in an archive of
   several structures was already ambiguous and refused; the section-level
   field is now the preferred carrier, because a consumer reads it from the
   manifest, and the reference writer writes both. What was missing is the
   link from a run to its job: `run.record` gains `job_ref` (resolved to a
   `job.spec` by validators) and `run_status` (`running`, `converged`,
   `failed`). A job is pending until a run record names it, result sections
   carry `run_ref` and `structure_ref`, and `provenance.run_status`
   summarises the set (`pending`, `running`, then `converged` or `failed`).
   Jobs do not depend on each other; a sequence within one job stays its
   `tasks` list. A single-job archive reads exactly as before.
5. **`preview`, a canonical kind, and the member field `media_type`** (spec
   § 4.2, § 5.15), added directly by maintainer decision. Every member spec
   may carry `media_type`, the IANA media type of its bytes; it describes and
   never changes how a member is read. A `preview` section has a required
   raster `thumbnail` (`image/png` or `image/jpeg`) and an optional `image`
   (`image/svg+xml`, PNG or JPEG), both opaque members whose `media_type` is
   required, so a file manager reads the type from the manifest and never
   sniffs. The archive's picture is the thumbnail of the first `preview`
   section; its longest side should be 256–1024 px and the member at most
   1 MiB. SVG must be self-contained and inert (no external resources, no
   scripts), and consumers must render it without running scripts or
   fetching anything. A fixed entry name such as an office document's
   `Thumbnails/thumbnail.png` was rejected: QVF names every member from the
   manifest (§ 2), and a thumbnail extractor reads a stored manifest as
   cheaply as a fixed entry. Vendor sections may now carry opaque members too,
   which is how vibe-sketch can store a PNG without pretending it is a
   `uint8` array. Validators check the bytes against the media type
   (`E_MEMBER_MALFORMED`, `E_UTF8`).
6. **One media type and one macOS UTI** (spec § 2.4). `.qvf` is
   `application/vnd.vibe-qc.qvf+zip` — the IANA vendor tree, named for the
   steward, with the registered `+zip` suffix — and
   `com.vibe-qc.qvf` on macOS, conforming to `public.zip-archive` and
   `public.content`. `registry.json` carries the values under `file_type`, and
   `spec/file-type/` the declarations every application copies unchanged:
   the UTI dictionary for `Info.plist` and the freedesktop.org
   shared-mime-info XML. One application of the family exports the UTI and
   the others import the identical declaration. The media type is not yet
   registered with IANA; registration is a follow-up for the steward and
   changes nothing in the files.
7. **Reference writer API** (informative; no format change). Python
   `add_*` methods all take `label=`; `add_structure` takes the identity
   fields; `add_reaction_scheme`, `add_preview` and `opaque_members=` are new.
   `remove_section` removes a section, and removing and adding under the same
   id replaces it; `open_for_update` takes an archive's path or bytes, and a
   member whose default path the archive already holds gets a fresh one,
   since an update never rewrites an entry (§ 2.3). `repack=True` on `write`
   and `to_bytes` drops superseded entries, which a program that saves the
   same document repeatedly needs.
8. **Three inconsistencies.** (a) The `Extensions` schema description said
   keys come "without x_" while the pattern required the prefix; it was
   corrected on 2026-10-07 (keys carry `x_`, as spec § 7.4 and the corpus
   do), after the 0.1.0 release that vibe-sketch pins. (b) The schema called
   itself "(v1.1)" while `qvf_version` is `const 1`; the label was removed in
   the same revision, since v1 has no minor version. (c) The validators
   rejected `qvf_version: 2` with `E_VERSION` while spec § 3.1 and
   `registry.json` (`readable_versions: [1, 2]`) told consumers to accept it;
   the conformance README recorded this as a decision pending. **Decided:** a
   validator reads `2` as `1` for every check, the JSON Schema check
   included, and reports the new warning `W_VERSION_WITHDRAWN`, so such an
   archive is valid and the warning still records that its producer wrote a
   version producers must not write. A validator answers whether an archive
   can be read correctly, and this one can. The schema keeps
   `qvf_version: const 1`, the producer contract; `E_VERSION` remains for
   every other value; a tool that updates such an archive writes `1`. The
   legacy corpus expects the warning.

**Rationale for no bump.** Every change adds an optional field, a canonical
kind or a warning, or relaxes a recommendation (`SHOULD carry exactly one
job.spec`). No existing member, kind, default or unit changes. Every archive
that validated before validates unchanged; the one archive whose verdict
changes, the legacy `qvf_version: 2` archive, goes from invalid to valid with
a warning.

**Compatibility.** Producers: additive. vibe-sketch can keep its vendor
section and add the canonical carriers: identity on each `structure_<species
id>` section, one `reaction.scheme`, one `preview` (its PNG and SVG move from
the vendor `preview` member), and job sets with `job_ref` when it writes
calculations; the one-species archives it writes for vibe-view keep charge
and multiplicity in `provenance` and may add them to the structure, equal.
Consumers that do not know the new kinds list them as unsupported, as for
any kind; a consumer that reads charge and multiplicity should prefer the
structure's and fall back to `provenance` for a single-structure archive.
Validator authors: the rules above; Python, C++ and JavaScript implement them
with the codes of Appendix B. Conformance: `reaction_scheme.qvf` and
`job_set.qvf` join the corpus, ten archives the negative corpus.

### 2026-10-08 — `wavefunction.bloch`: k-resolved Bloch orbitals, promoted provisionally

**Decision.** A canonical kind **`wavefunction.bloch`** (spec § 5.3.1)
carries the k-resolved crystalline orbitals of a periodic system over an
atom-centred Gaussian basis. Its members: `basis`, the `wavefunction.gto`
payload (`$defs/BasisPayload`, unchanged); `kpoints`, strict
`$defs/BlochKpointsPayload` (`kpoints` in fractional reciprocal-lattice
coordinates, required; optional `weights` summing to 1 within 1e-6, `labels`,
`mesh`, `shift`, `time_reversal`, `units`); `mo_metadata`, strict
`$defs/BlochMoMetadataPayload` (`spin` `restricted` or `unrestricted`,
`n_spin`, `n_mo`, `n_ao`, `complex` (always `true`) and `k_point_convention`
(`fractional`) required; `orbital_kind`, `n_mo_per_k`, `fermi_energy` and
`units` optional); `mo_coefficients`, float64 `[n_k, n_spin, n_mo, n_ao, 2]`
with the last axis (real, imaginary); `energies`, float64 `[n_k, n_spin,
n_mo]` in hartree; `occupations`, float64 `[n_k, n_spin, n_mo]`, electrons per
orbital at each k-point. The spec states the Bloch sum
φ_μk(r) = N^(−1/2) Σ_T e^{+ik·T} χ_μ(r − T), with k fractional
(k·T = 2π k·n), the phase e^{+ik·T} without the atomic position, and the AO
order and primitive normalization of Appendix A. The three reference
validators check that the member shapes agree with each other and with
`mo_metadata` and `basis.n_ao`, that the weights sum to 1 and the `labels`
and `n_mo_per_k` have one entry per k-point, and that the section's structure
(the section-level `structure_ref`, else `basis.structure_ref`) is a
`structure` section with at least one periodic axis, reporting
`E_MEMBER_MALFORMED` and, for an unresolved `basis.structure_ref`,
`E_REF_UNRESOLVED`; no finding code is added. When an archive also carries
`bands` at the same k-points, the band energies **SHOULD** agree with
`energies`; nothing checks that. The conformance corpus gains `bloch.qvf`, the
negative corpus `bloch_weights_sum.qvf`,
`bloch_coefficients_trailing_axis.qvf`, `bloch_occupations_shape.qvf` and
`bloch_structure_molecular.qvf`. **No `qvf_version` bump.**

**Promoted ahead of the threshold, and provisional.** The kind is the shape of
the vendor section `x_vibeqc.bloch_wavefunction`, which one producer
(vibe-qc) writes. The promotion rules ask for two independent producers; the
maintainer promoted it before a second one ships (format review P15, decision
5), as `spectra.epr` was, so that periodic Gaussian codes (CP2K, CRYSTAL, a
PySCF-PBC adapter) have a canonical target instead of each inventing a vendor
layout. Its `registry.json` `promotions` entry says
`"from": "x_vibeqc.bloch_wavefunction"` and `"status": "provisional"`. A
provisional kind is canonical in every respect: producers may emit it,
validators check it, consumers may rely on it, and any revision it needs
before it becomes final is additive, as for every canonical kind. It becomes
final, and the status is removed, when a second independent producer ships
it. The other open promotion criterion is a reference consumer: vibe-view
renders `wavefunction.gto` orbitals at Γ with a lattice sum and refuses other
k-points because the older metadata had no declared k convention, which this
kind now declares.

**Vendor mapping.** The vendor section's restricted coefficient payload
(`schema_version` 1.0) maps onto the kind with the same numbers and the same
conventions, so the producer can switch kinds without a data change:

| `x_vibeqc.bloch_wavefunction` | `wavefunction.bloch` |
|---|---|
| member `basis` (`structure_ref`, `pure`, `n_ao`, `shells`; coefficients on `N_i`-normalized primitives) | member `basis`, unchanged |
| binary member `kpoints` float64 `[n_k, 3]` and `blocks[k].k_point`, with `k_point_units: "fractional_reciprocal"` | `kpoints.kpoints`, with `mo_metadata.k_point_convention: "fractional"` |
| `blocks[k].k_weight` | `kpoints.weights[k]` |
| `blocks[k].k_index`, `n_kpoints` | the position along axis 0, and its length |
| per-k members `mo_coefficients_k<k>`, float64 `[n_mo_k, n_ao, 2]`, MO-major, with `coefficient_encoding: "complex_split_last_axis"` and `coefficient_components: ["real", "imag"]` | `mo_coefficients[k, 0]`, with `mo_metadata.complex: true` |
| per-k members `mo_energies_k<k>`, float64 `[n_mo_k]`, hartree | `energies[k, 0]` |
| per-k members `occupations_k<k>`, float64 `[n_mo_k]` | `occupations[k, 0]` |
| `blocks[k].n_mo`, which may differ between k-points | `n_mo`, the largest of them, and `mo_metadata.n_mo_per_k[k]`; the rows beyond it are zero padding |
| `n_ao`, `blocks[k].n_ao` | `mo_metadata.n_ao` |
| `spin: "restricted"`, `orbital_kind` | `spin`, `n_spin: 1`, `orbital_kind` |
| `schema_version`, `restart_kind` | not needed; a producer may keep them as `x_` keys of `mo_metadata` |

The vendor writer's Bloch phase is the e^{+ik·T} of the spec, its k-points are
fractional (wrapped to [−1/2, 1/2)), and it divides engine coefficients by
`N_i` as for `wavefunction.gto`. New in the canonical kind: a second spin
channel, `labels`, `mesh`, `shift`, `time_reversal` and `fermi_energy`. Not
carried: the vendor's spin-density restart payload (`schema_version` 1.1,
`restart_kind: "spin_density_bloch_kpoints"`, members
`density_alpha_k<k>` and `density_beta_k<k>`, float64 `[n_ao, n_ao, 2]`),
which holds AO density matrices and no orbitals; it stays in the vendor
namespace until k-resolved density matrices get a canonical form. Two choices
go beyond the sketch of P15: `n_mo_per_k`, without which the vendor's
per-k-point orbital counts (canonical orthogonalisation removes a different
number of near-linear-dependent combinations at each k) could not be carried
without loss, and the k-point axis ahead of the spin axis, which keeps the
vendor's per-k blocks contiguous and differs from the leading spin axis of
§ 5.6 on purpose.

**Rationale for no bump.** A new canonical kind with new payload definitions
is the additive case of "Change process": no existing member, kind, default
or unit changes, and every archive that validated before validates unchanged.

**Compatibility.** Producers: purely additive. The reference writers gain
Python `add_wavefunction_bloch` (which shares the shell normalization of
`add_wavefunction_gto` through a helper), C++ `WavefunctionBloch` and
`add_wavefunction_bloch`, and the C ABI `qvf_add_wavefunction_bloch`; the C
ABI still writes molecular structures only, so a C caller cannot yet write
the periodic structure the kind needs. Consumers that do not support the kind
treat it per the unsupported-kind rules. A consumer that evaluates the
orbitals **MUST** use the stated Bloch sum and **MUST NOT** assume a gauge.
Validator authors: the rules above, with existing codes.
### 2026-10-08 — Property kinds: `properties.electric`, `properties.elastic`, `properties.dielectric`

**Decision.** Three canonical kinds, added directly (spec § 5.13, "Property
kinds"), each with one required JSON member `properties` whose payload is
**strict** (only the listed keys and `x_` keys, in every nested object).
(1) **`properties.electric`** (`$defs/ElectricPropertiesPayload`): `origin` (a
conventional name or a 3-vector in bohr, as `dipole_moment.origin`), `dipole`
(`vector`, `total`, `units`; **debye**), `quadrupole` (`matrix`, `traceless`
and `convention` `buckingham` or `cartesian`, all required; **debye·Å**),
`octupole` (`tensor`, 27 numbers in row-major `[3, 3, 3]`, optional
`convention`; **debye·Å²**), `polarizability` (`matrix`, `frequency_ev`,
`isotropic`; **atomic units**), `hyperpolarizability` (`tensor`,
`frequencies_ev`; **atomic units**) and `units`; at least one of the five
blocks is required. The root `dipole_moment` stays the canonical place for a
bare dipole. When both it and an electric `dipole` are present they **MUST**
agree within 1e-6 debye (`vector` with `vector_debye`, `total` with
`total_debye`); a validator reports a disagreement as the existing code
**`E_ROOT_INCONSISTENT`**, located at the section's `properties` member.
(2) **`properties.elastic`** (`$defs/ElasticPropertiesPayload`): `stiffness`
(6 × 6, Voigt order xx yy zz yz xz xy, **GPa**, required), `compliance`
(6 × 6, **GPa⁻¹**), `bulk_modulus_gpa`, `shear_modulus_gpa`,
`youngs_modulus_gpa`, `poisson_ratio`, `average` (`voigt`, `reuss`, `hill`)
and `units`. Both matrices **MUST** be symmetric: entries `[I][J]` and `[J][I]`
differ by at most 1e-6 times the largest absolute entry. (3)
**`properties.dielectric`** (`$defs/DielectricPropertiesPayload`):
`electronic` (3 × 3, required), `ionic`, `static` (3 × 3, dimensionless),
`born_charges` (one 3 × 3 per atom, **e**), `piezoelectric` (3 × 6, **C/m²**)
and `units`. When `electronic`, `ionic` and `static` are all present, `static`
**MUST** equal `electronic` + `ionic` entry by entry within 1e-6, and
`born_charges` **MUST** have one tensor per atom of the section's structure
(the section-level `structure_ref`, else the archive's only `structure`) when
that resolves. The elastic and dielectric violations are
**`E_MEMBER_MALFORMED`**. Spec § 4.3 gains unit rows for the fields, § 6 and
Appendix B gain the rules, and the schema gains the three section branches,
the three payload definitions and the shape helpers `Matrix3x3`, `Matrix3x6`,
`Matrix6x6` and `Tensor3x3x3`. The conformance corpus gains
`properties_molecular.qvf` and `properties_periodic.qvf`, the negative corpus
`elastic_not_symmetric.qvf`, `dielectric_static_mismatch.qvf`,
`dielectric_born_atom_count.qvf` and `electric_dipole_inconsistent.qvf`. **No
`qvf_version` bump.**

**Background.** Format review P14 named `properties.elastic` and
`properties.dielectric` as demand-driven kinds and P17 `properties.electric`,
so that a periodic-systems code had no schema-valid place for its stiffness
matrix, dielectric response or Born charges, and no code could carry a
quadrupole or a polarizability beside the dipole of the root block.

**Rationale.** All three kinds are added directly by maintainer decision under
the change process ("clear ecosystem need exists and a reference consumer will
render it"), as `forces` and `hessian` were, and are noted in `registry.json`.
A new kind with a payload that is strict from the start is the additive case:
nothing that was valid becomes invalid, because no valid archive carried these
kinds before. Strictness can be loosened later without a break; the reverse
could not.

**Units that the vocabulary cannot name.** The `units` object of each payload
declares only the fields whose unit has a name in § 4.3 (the dipole, `origin`,
the stiffness and moduli, Poisson's ratio, the dielectric tensors and the Born
charges), and a declaration only restates the fixed unit. The quadrupole
(debye·Å), octupole (debye·Å²), polarizabilities (atomic units), compliance
(GPa⁻¹) and piezoelectric tensor (C/m²) are in compound units that the
vocabulary does not name (spec § 9, item 7), so they are fixed by § 5.13 alone
and cannot be declared. This follows the earlier record on the unit
vocabulary.

**Readings to confirm.** (a) `E_ROOT_INCONSISTENT`, defined for two root
blocks, now also covers a section that restates a root block; the code is not
renamed and Appendix B says so. (b) The tolerances differ by quantity: matrix
symmetry is relative to the largest entry (so a zero beside a 1e-9 entry is
not an error), while `static` against `electronic + ionic` and the dipole are
absolute (1e-6, a dimensionless number and debye). (c) P17 names a
quadrupole convention and a traceless flag but not their definitions; § 5.13
defines `buckingham` as ½ Σ q (3 r_a r_b − r² δ_ab), which is always
traceless, and `cartesian` as Σ q r_a r_b, with its trace removed when
`traceless` is true. The octupole gets the same `convention` pair, an
addition to P17's sketch, with `cartesian` as its default, and its unit
(debye·Å²) is the one P17 left open. (d) `average` is optional and says "not
stated" when absent; a producer SHOULD give it whenever it writes a modulus.
(e) The piezoelectric tensor is the relaxed-ion stress tensor. (f) Two
consistency rules are not checked: `polarizability.isotropic` against
one third of the trace, and `dipole.total` against the length of
`dipole.vector`. (g) The section is numbered § 5.13 so that § 5.12 stays free
for work in parallel; a later renumbering is a documentation change.

**Compatibility.** Producers: purely additive. The reference writers gain
Python `add_electric_properties(properties)`, `add_elastic_properties(stiffness,
compliance=, moduli=, average=, units=)` and `add_dielectric_properties(
electronic, ionic=, static=, born_charges=, piezoelectric=, units=)`, C++
`add_electric_properties`, `add_elastic_properties` and
`add_dielectric_properties` (each takes the payload as JSON and checks it), and
the C ABI `qvf_add_electric_properties_json`,
`qvf_add_elastic_properties_json` and `qvf_add_dielectric_properties_json`.
Consumers that do not support the kinds treat them per the unsupported-kind
rules. Validator authors: add the three kinds, the three payload definitions
and the rules of § 6; no code is added.
### 2026-10-08 — `volume.vector`, `volume.complex`, `volume.series` and `volume.laplacian`; periodic grid semantics

**Decision.** Two additions under v1. (1) The volume grid payload
(`$defs/VolumeGridPayload`, spec § 5.2) gains three optional fields:
**`periodic`** (three booleans, absent `[false, false, false]`),
**`includes_endpoint`** (boolean, absent `true`) and **`structure_ref`**. Along
an axis `a` that `periodic` marks, the grid samples exactly one period of the
cell of its structure (the section-level `structure_ref`, else the grid's,
else the archive's only `structure`): with `includes_endpoint` true, the
cube-file convention, the last plane repeats the first and
`voxel_vectors[a] × (shape[a] − 1)` is lattice vector `a`; with it false, the
CHGCAR convention, `voxel_vectors[a] × shape[a]` is. A consumer that tiles or
wraps a periodic grid **MUST** drop the duplicated plane when
`includes_endpoint` is true. Spec § 5.2 is rewritten around a precise
statement of the point positions and a one-axis, four-point worked example of
both conventions. All three reference validators check, with
`E_MEMBER_MALFORMED` at the `grid` member, that a periodic axis is periodic in
the structure (`pbc[a]`) and that its span equals the lattice row converted
to bohr within 10⁻⁶ relative, and report a grid `structure_ref` that names no
`structure` section as `E_REF_UNRESOLVED`. (2) Four canonical kinds:
**`volume.vector`** (`data` float32/64 `[nx, ny, nz, 3]`, Cartesian
components, `$defs/Volume4DVectorBinary`), **`volume.complex`**
(`[nx, ny, nz, 2]`, last axis `[re, im]`, `$defs/Volume4DComplexBinary`),
**`volume.series`** (`[n, nx, ny, nz]` with a required strict `axis` member,
`$defs/VolumeSeriesAxisPayload`: `kind` `time` / `frame` / `state` / `custom`,
`values` one per slab, `units`, `labels`, `trajectory_ref`) and
**`volume.laplacian`** (∇²ρ, `[nx, ny, nz]`, in e/bohr⁵, a unit § 4.3 now
fixes in prose because the vocabulary has no name for it). Validators check
the rank and trailing axis of each new kind and a series axis against its
slabs (`E_MEMBER_MALFORMED`), and resolve `axis.trajectory_ref` to a
`trajectory` or `reaction.path` section (`E_REF_UNRESOLVED`); a series axis
`units` that is not a unit name is `W_UNIT_UNKNOWN`. The `frame_volumes` +
`volume_grid` members of `reaction.path` are documented (§ 5.5) as the older
form of a `volume.series` with a `frame` axis naming the path; producers
**SHOULD** prefer `volume.series`, and both stay valid. The conformance corpus
gains `volume_periodic.qvf`, `volume_vector.qvf`, `volume_complex.qvf`,
`volume_series.qvf` and `volume_laplacian.qvf`; the negative corpus
`volume_periodic_span.qvf`, `volume_endpoint_count.qvf`,
`volume_series_axis_length.qvf` and `volume_vector_trailing_axis.qvf`. **No
`qvf_version` bump.**

**Background: a silent interoperability bug.** Format review P11. A grid that
samples a periodic cell is written in one of two conventions that agree on
every point position and disagree on what a period is. Gaussian cube files
include the endpoint: n points with the last one the image of the first, so
n − 1 steps per period. VASP CHGCAR files do not: n distinct points per
period. Both are in wide use, and nothing in the numbers of a grid says which
one it is, because the voxel vectors give the positions either way. A
consumer that guesses wrong renders without an error message: it tiles a
cube-style grid with a doubled plane at every cell boundary and a period one
step too long, or drops a real plane of a CHGCAR-style grid; a converter that
reads one convention and writes the other shifts every point. QVF v1 said
nothing about it. `includes_endpoint` makes the convention explicit, and the
span check, which needs nothing but the grid and the structure's cell, turns
a mislabelled grid from a silently wrong picture into a validation error.
The new kinds give a canonical home to fields that producers wrote as
`volume.generic` or not at all: density gradients and other vector fields,
complex orbitals and Bloch functions, and fields along a trajectory, in time
or by state, of which `reaction.path` `frame_volumes` covered only the first.

**Rationale.** The four kinds are added directly by maintainer decision
under the change process ("clear ecosystem need exists and a reference
consumer will render it"), as `run.record`, `job.spec`, `forces` and
`hessian` were, and are noted in `registry.json`. New optional payload
fields, new canonical kinds and validator rules that only constrain the new
fields are the additive case: every archive that was valid stays valid,
because no grid declared `periodic` before this record and every rule of
spec § 5.2 is conditional on it or on a new kind. `includes_endpoint`
defaults to `true`, the cube-file convention, as P11 specifies; it matters
only on a periodic axis, and no grid declared one before this record, so the
default changes the meaning of no existing archive.

**Readings to confirm.** The brief (P11) leaves three details open, settled
here and open to revision by the maintainer before the next release. First,
the span check is applied to every grid payload, the `volume_grid` of a
`reaction.path` included, whenever the grid declares a periodic axis and its
structure resolves; a variable-cell path whose volumes cannot share one cell
should therefore not declare `periodic`. Second, a series `trajectory_ref` is
resolved whatever the axis `kind`, not only for `frame` axes, because an
unresolvable reference is an error wherever QVF allows one; and `labels` is
held to one entry per slab like `values`. Third, the span is compared as a
vector (the norm of the difference against 10⁻⁶ of the lattice vector's
length), not as a length, so a step of the right length in the wrong
direction is an error.

**Compatibility.** Producers: purely additive. Writers gain Python
`add_volume(..., periodic=, includes_endpoint=, structure_ref=, axis=)`,
which takes 4-D data for `volume.vector`, `volume.complex` (or a complex 3-D
array) and `volume.series` and checks the rank per kind; C++ `Grid::periodic`,
`Grid::includes_endpoint` and `Grid::structure_ref`, `add_volume` for the new
kinds and `add_volume_series(grid, data, axis, ...)`; and the C ABI's first
generic volume entry points, `qvf_add_volume(w, kind, grid_json, dtype, rank,
shape, data, label)` and `qvf_add_volume_series`. The writers do not check
the span; the validators do. Consumers that do not support a new kind treat
it per the unsupported-kind rules; a consumer that tiles periodic grids
**MUST** honour `includes_endpoint`. Validator authors: no new finding code;
the new rules report `E_MEMBER_MALFORMED`, `E_REF_UNRESOLVED` and
`W_UNIT_UNKNOWN`, and the lattice conversion is the factor
`f = 1.8897261246` of spec § 4.3.
### 2026-10-08 — Wavefunction extensions: ECPs, two-component layout, overlap and density matrices; normalization checkable

**Decision.** `wavefunction.gto` (spec § 5.3) gains, under v1: (1) a full
specification of the **effective core potentials** whose shape the payload
definitions reserved. `basis.ecp[]` replaces `n_core_electrons` electrons of
one atom; each radial function of `projectors` is
`U(r) = Σ_i c_i r^(n_i − 2) exp(−α_i r²)` (r in bohr, U in hartree, the
conventional semi-local form, `powers` holding the `n_i`), and the new
optional `local` flag (default `false`) marks the local part, which acts on
every angular momentum and carries the highest `l` of its centre; every other
radial function multiplies the projector onto its `l`, holding
`U_l − U_L` as the Gaussian, Molpro and NWChem tabulations do. (2)
`basis.n_electrons` is the number of electrons the wavefunction describes,
`Σ Z − Σ n_core_electrons − charge`; it **MUST** equal the sum of the
occupations over both spin channels (within 1e-6, unless
`occupation_semantics` is `transition_weight`) and `provenance.n_electrons`
when both are present. (3) The **spin layouts**: `restricted_open` uses the
single-set shape; `generalized` uses it too, with the AO axis of
`mo_coefficients` doubled in the spinor layout `[α AOs…, β AOs…]`, and
requires `complex: true`; `complex: true` means that every coefficient member
has a trailing `[re, im]` axis of 2, and its absence that none has.
`symmetry_labels` has one entry per MO. (4) Optional binary members
**`overlap`** (the AO overlap matrix of the AOs as `basis` defines them) and
**`density_matrix`**, or **`density_matrix_alpha`** + **`density_matrix_beta`**
(one-particle density matrices in the AO basis), all float64
`[n_ao, n_ao]` (`$defs/WavefunctionAoMatrix`); the density members follow the
coefficient members. (5) **Normalization becomes checkable**: when `overlap`
is present, a validator recomputes from the `basis` payload the self-overlap of each
AO (Appendix A.1, *Verifying with `overlap`*: `S_axial` of its shell for a
spherical component, `S_axial · (2l_x−1)!!(2l_y−1)!!(2l_z−1)!! / (2l−1)!!` for
a Cartesian one) and compares it with the stored diagonal at a relative
tolerance of 1e-6; a disagreement is the new finding code
**`E_NORMALIZATION`**. Producers **SHOULD** write `overlap`. The other rules
are reported as `E_MEMBER_MALFORMED`, and the `provenance` disagreement as
`E_ROOT_INCONSISTENT`. All three reference validators (Python, C++,
JavaScript) implement every check. The conformance corpus gains
`wavefunction_ecp.qvf` and `wavefunction_overlap.qvf`, the negative corpus
`wavefunction_normalization.qvf`, `wavefunction_n_electrons.qvf` and
`wavefunction_complex_rank2.qvf`. **No `qvf_version` bump.**

**Background.** Format review P09. The schema allowed `[n_mo, n_ao, 2]`
coefficients and the payload definitions named `ecp`, `n_electrons`, the
grown `spin` enum and `complex`, but the record of 2026-10-07 left the radial
convention of `powers`, the layout of a `generalized` wavefunction and the
electron count unspecified, so an ECP archive could not be evaluated and its
electron count was ambiguous. Independently, the primitive normalization of
Appendix A.1 was the format's most error-prone contract and could not be
checked from an archive: the reference producer's own readers once applied a
per-component Cartesian correction that nothing caught (record of
2026-08-05, spec § 9). The overlap matrix, which every producer's integral
code already has, turns that contract into a number a validator compares.

**Rationale for no bump.** The additions are optional members and an optional
payload key; the reserved fields keep their shapes. Two rules can fail an
archive that validated before: a `generalized` `mo_metadata` without
`complex: true` (the schema now requires it, `E_PAYLOAD_SCHEMA`) and a
coefficient member whose rank contradicts `complex`. Neither layout was
specified before, so no conforming archive could rely on them, and no
reference writer or corpus archive emits either; the same holds for a
`symmetry_labels` of the wrong length and an `n_electrons` that contradicts
the occupations. This is the additive case of "Change process".

**Readings to confirm.** `n_ao` for the shape checks is counted from the
shells, falling back to the declared `n_ao` only when the shells cannot be
read; the electron-count tolerance is absolute (1e-6 electrons) and the
normalization tolerance relative (1e-6). The normalization check reads the
diagonal only, so it cannot see a permutation of `m` within a spherical shell;
an off-diagonal check would need the real-solid-harmonic phase convention,
which Appendix A.2 does not yet fix. `generalized` requires `complex: true` as
P09 proposed, so a real two-component wavefunction is written with a zero
imaginary part. The density members are real; for complex orbitals they hold
the real part of the Hermitian density matrix, all the electron density
needs. All are open to revision by the maintainer before the next release.

**Compatibility.** Producers: purely additive. Python `add_wavefunction_gto`
gains `spin`, `ecp`, `n_electrons`, `symmetry_labels`, `overlap`,
`density_matrix`, `density_matrix_alpha`, `density_matrix_beta` and
`complex_coefficients`, and now refuses coefficient matrices whose AO extent
is not the basis's; C++ `WavefunctionGTO` gains the matching fields and
`QvfWriter` `set_wavefunction_overlap` and `set_wavefunction_ecp`; the C ABI
gains `qvf_add_wavefunction_gto_unrestricted`,
`qvf_set_wavefunction_overlap` and `qvf_set_wavefunction_ecp_json`.
Consumers: a consumer that evaluates ECP integrals may now do so; a consumer
that counts electrons uses `Z − n_core_electrons` for an ECP atom. Validator
authors: `E_NORMALIZATION` is a new code; codes are added, never renamed.
### 2026-10-08 — `excited_states` and `volume.transition_density` canonical kinds; natural transition orbitals; spectra name states

**Decision.** Four additions under v1. (1) A canonical kind
**`excited_states`** (spec § 5.12): one JSON member `states`
(`$defs/ExcitedStatesPayload`, strict) holding the states of one excited-state
calculation in ascending order of a unique `index` (integer ≥ 1). Each state
(`$defs/ExcitedState`) requires `index` and `excitation_energy`, a
`{value, units}` object that producers **SHOULD** write in `ev`, and may carry
`total_energy` (**SHOULD** be `hartree`), `multiplicity`, `symmetry`,
`oscillator_strength` (dimensionless), `rotatory_strength_cgs` (10⁻⁴⁰ esu²·cm²),
`transition_dipole` and `transition_magnetic_dipole` (atomic units),
`contributions` (`{from_mo, to_mo, spin?, weight}`, rows of the reference
wavefunction's coefficient member), `nto_ref` and `density_ref`; an optional
strict `meta` carries `method`, `reference_state`, `n_states`, `tda`,
`spin_flip` and a `units` declaration. The section may carry the field
`wavefunction_ref`, the `wavefunction.gto` of the reference state. (2) A
canonical kind **`volume.transition_density`** (spec § 5.2): the transition
density of one excitation in e/bohr³ with the members of `volume.density` and
the optional section fields `excited_states_ref` and `state_index` (which
requires `excited_states_ref`). (3) **Natural transition orbitals** (spec
§ 5.3) are two `wavefunction.gto` sections per state, hole and particle, with
`orbital_kind: "nto"`, `occupation_semantics: "transition_weight"`, the
`excited_states` id in `state_ref`, and two new optional `mo_metadata` fields,
`state_index` and `nto_role` (`"hole"` or `"particle"`); row k of the two
sections is the k-th pair and `occupations` are the pair weights. A state's
`nto_ref` names the hole section. (4) A one-dimensional spectrum payload may
carry **`states_ref`**, the `excited_states` section whose `index` values its
peaks' `assignment.state_index` are. Validators report repeated or
descending indices, an `n_states` that differs from the number of states and
a contribution whose `from_mo` or `to_mo` is not a row of the reference
wavefunction (read from the manifest `shape`) as `E_MEMBER_MALFORMED`, and a
`wavefunction_ref`, `nto_ref`, `density_ref`, `states_ref` or
`excited_states_ref` that names no section of the right kind, or a
`state_index` that names no state, as `E_REF_UNRESOLVED`; `meta.units` joins
the unit-name rule (`W_UNIT_UNKNOWN`). The conformance corpus gains
`tddft.qvf`, the negative corpus `excited_states_index_duplicate.qvf`,
`excited_states_nto_ref_wrong_kind.qvf` and
`excited_states_to_mo_out_of_range.qvf`. **No `qvf_version` bump.**

**Background.** Format review P08, and the NTO and transition-density parts
of P09 and P11: an archive could carry a UV-Vis spectrum but not the states it
is made of, so a viewer could neither list the transitions nor draw the
orbitals and densities that explain them. `MoMetadataPayload` already had
`orbital_kind: "nto"` and a `state_ref` whose meaning was left open, and
`SpectrumPayload` an `assignment.state_index` numbered "as the producer
numbers its states"; this record gives both a target.

**Rationale.** Both kinds are added directly by maintainer decision under the
change process ("clear ecosystem need exists and a reference consumer will
render it"), as `forces` and `hessian` were, and are noted in `registry.json`.
New kinds, new optional fields of `mo_metadata`, of a spectrum payload and of
a section, and new rules about them are the additive case: no member or unit
changes and nothing that was valid becomes invalid, except a typed-open
spectrum payload that already carried a key named `states_ref` naming no
`excited_states` section, which no known producer wrote. Finding codes are
reused, none is added.

**Choices made here, open to revision by the maintainer before the next
release.** *Hole and particle NTOs are two sections*, not two halves of one
coefficient matrix, so each is an ordinary `wavefunction.gto` that existing
consumers render; the price is that `nto_ref`, a single string, names the hole
section and the particle section is found through its `mo_metadata`.
*`occupations` of an NTO section are the pair weights* λₖ, the eigenvalues of
T Tᵀ (the squares of the singular values of the transition density matrix),
which sum to 1 for a normalized state and are what programs print as NTO
occupations. *The transition density's integer field is `state_index`*, not
`state_ref`: every other `*_ref` in QVF is a string section id, and
`mo_metadata.state_ref` is the `excited_states` id, so the integer takes the
name the peaks and `mo_metadata` already use. *Units with no unit name*: the
vocabulary of spec § 4.3 has no name for the 10⁻⁴⁰ cgs rotatory strength nor
for atomic units of the dipoles, so those fields have fixed units, carried in
the field name for the rotatory strength, and are not declared in
`meta.units`.

**Compatibility.** Producers: purely additive. The reference writers gain
Python `add_excited_states(states, *, meta=, wavefunction_ref=, section_id=)`,
`add_volume("volume.transition_density", ..., excited_states_ref=,
state_index=)` and `add_wavefunction_gto(..., state_ref=, state_index=,
nto_role=)`; C++ `add_excited_states(states, meta, wavefunction_ref,
section_id)`, `add_volume_transition_density` and the `WavefunctionGTO` fields
`state_ref`, `state_index` and `nto_role`; and the C ABI
`qvf_add_excited_states_json`. `add_spectrum` refuses a `states_ref` that
names no `excited_states` section already added. Consumers that do not
support the kinds treat them per the unsupported-kind rules. Validator
authors: the new rules use the existing codes `E_MEMBER_MALFORMED` and
`E_REF_UNRESOLVED`.
### 2026-10-08 — Typed `provenance` fields; typed `job.spec` fields; `structure_ref` resolved

**Decision.** Two additions under v1, both optional. (1) The root
`provenance` block (spec § 3.2) stays open and gains seven typed fields:
`software` (an array of `$defs/ProvenanceSoftware`, strict: `name` required,
`version`, `url`, and `x_` vendor keys), `git_commit` (7 to 64 hexadecimal
characters, the producer's own revision), `n_threads` and `n_mpi_ranks`
(integers ≥ 1), `n_gpus` (an integer ≥ 0), and `started_utc` and
`finished_utc` (ISO 8601 UTC, the pattern of the `metadata` time stamps).
`wall_seconds` keeps its name; no `walltime_s` is added to `provenance`.
Spec § 3.2 replaces the prose list of recognised fields by a table with type,
unit and meaning for every field, and states that `provenance.n_electrons`,
when present, **MUST** agree with `basis.n_electrons` of a `wavefunction.gto`
section. (2) `$defs/JobSpecPayload` (spec § 5.9) gains `structure_ref` (the
`structure` section to run), `dispersion` and `relativistic` (strings) and
four objects, each a strict `$defs` entry with `x_` vendor keys allowed:
`solvation` (`model` required, `solvent`, `epsilon` > 0), `scf`
(`convergence` > 0, `max_iterations` ≥ 1, `guess`, `damping`), `geometry`
(`convergence` > 0, `max_steps` ≥ 1, `coordinates` of `cartesian`,
`internal` or `redundant`) and `resources` (`n_cores` ≥ 1, `memory_mb` ≥ 1,
`walltime_s` > 0). The payload root stays open and engine keywords stay under
`options`. `resources` are hints, not requirements; what a run used is
recorded in `provenance`. Validators resolve the payload's `structure_ref` to
a section of kind `structure` and report any other value as the existing
`E_REF_UNRESOLVED` (no new finding code); all three reference validators do.
The conformance corpus gains `job_spec_full.qvf` and `provenance_full.qvf`,
and the negative corpus `job_spec_structure_ref_wrong_kind.qvf`,
`provenance_git_commit_not_hex.qvf` and `provenance_started_utc_not_utc.qvf`.
The frozen `job_spec.qvf` is untouched. **No `qvf_version` bump.**

**Background.** Format review P18. Provenance said which method produced an
energy but not which code, which revision, on how many threads or ranks, or
when, so two archives of the same calculation could not be told apart or
reproduced; and a `job.spec` could request a method and a basis but not the
things a runner needs next (a dispersion correction, solvation, SCF and
optimisation settings, resources), which then went untyped into `options`
where no two codes agreed on the key names. The `hostname` part of P18 was
settled earlier (2026-10-07).

**Rationale for no bump.** Every addition is an optional field of an open
object (`provenance`, the `job.spec` payload root) or a new strict object
that nothing carried before, so every archive that validated before still
validates; the only archives newly rejected are those that already carried
one of the new key names with a value outside the new type, and no reference
writer and no corpus archive did. Nothing becomes required, no unit or default
changes, and no kind is added: the additive case of "Change process".

**Readings to confirm.** The brief fixes types and ranges but not meaning, so
the spec states these readings, which the maintainer may revise before the
next release. `scf.convergence` is the convergence threshold of the SCF
energy change in hartree, and `geometry.convergence` the threshold of the
largest Cartesian gradient component in hartree/bohr; an engine that
converges on another measure maps them to its nearest criterion. Without a
unit a bare number would not be portable. `n_threads` counts threads per
process, so the total is `n_mpi_ranks` × `n_threads`.
`provenance.n_electrons` is defined as the number of electrons the
calculation treats explicitly, the same quantity as `basis.n_electrons`
(valence electrons only when an ECP replaces the core), which makes the
agreement rule hold for ECP calculations too. A `job.spec` without
`structure_ref` runs on the archive's only `structure`; with several and none
named the job is ambiguous and a runner **MUST** refuse it. A section-level
`structure_ref` on the `job.spec` section takes precedence over the payload's,
as for the `basis` payload (spec § 4.1). The typed blocks are strict rather
than open so that an engine keyword cannot hide in them. The `provenance` of a
`pending` archive, when it has any of the new fields, describes the step that
prepared the job. `git_commit` accepts either case of hexadecimal digit, as
git prints lowercase but does not require it of a producer's copy.

**Compatibility.** Producers: additive, except that an archive which already
carries one of the new key names with a value outside its type, such as a
`software` string or `n_threads: 0`, no longer validates. Python `set_provenance(**fields)`
and C++ `set_provenance(Json)` (and so `qvf_set_provenance_json`) check the
typed fields and refuse a wrong value before anything changes, where they
accepted anything before; the C++ call now also rejects a provenance that is
not a JSON object, and parses a raw fragment even with `allow_hostname`.
Python `add_job_spec` gains `structure_ref`, `solvation`, `dispersion`,
`relativistic`, `scf`, `geometry` and `resources`; C++ `add_job_spec(Json)`
checks the same fields; the C ABI gains `qvf_add_job_spec_json(w, spec_json)`
for the whole payload and `qvf_add_job_spec` is unchanged. A writer accepts a
`structure_ref` to a section added later and refuses one that already names a
section of another kind. Consumers that ignore the new fields are unaffected.
Validator authors: a `job.spec` payload's `structure_ref` is checked with
`E_REF_UNRESOLVED` at the `spec` member, and the typed fields are checked by
the JSON Schema (`E_SCHEMA` for `provenance`, `E_PAYLOAD_SCHEMA` for the
payload), so a validator without a schema engine does not see them; Appendix B
lists the new case.

### 2026-10-08 — Member `size`; updating an archive in place; the last `manifest.json` wins; root `byte_order`

**Decision.** Three container rules under v1. (1) Every member spec
(`$defs/JsonMember`, `BinaryMember`, `OpaqueBinaryMember`,
`CitationsReferences`) gains an optional **`size`** (`$defs/MemberSize`, an
integer ≥ 0): the uncompressed byte count of the stored member, which **MUST**
equal the uncompressed size the ZIP central directory records for its entry
(spec § 4.2). Producers **SHOULD** write it, and both reference writers now
write it for every member. Validators compare it with the central directory
before reading the member and report a disagreement with the existing code
**`E_BINARY_SIZE`**, whose Appendix B meaning is widened to cover it; a `size`
that is not a non-negative integer is `E_MEMBER_MALFORMED`. (2) A new spec
**§ 2.3, "Updating an archive in place"**, fixes how an archive is appended
to: an update **MAY** append entries, **MUST** write a new `manifest.json` as
the last entry, and **MUST NOT** reuse the name of an existing entry other
than `manifest.json`. The previous manifest and every entry the new manifest
no longer references are superseded orphans, which consumers **MUST** ignore
and validators report as **`W_ORPHAN_MEMBER`**. A consumer **MUST** read the
`manifest.json` whose central-directory record comes last. Repacking removes
the orphans; `metadata.modified_utc` **SHOULD** be set; `run.record.sequence`
orders the run records of successive updates; `provenance.run_status` moves
per § 3.2. (3) An optional root **`byte_order`** with `const: "little"` (spec
§ 2.2, § 3.1) declares what every v1 archive already is; any other value
fails the JSON Schema (`E_SCHEMA`), and consumers **SHOULD** refuse it. The
reference writers emit `"byte_order": "little"`. The conformance corpus gains
`updated_in_place.qvf` (two `manifest.json` entries, one orphaned member,
`modified_utc`; it validates with two warnings) and `size_declared.qvf`, the
negative corpus `member_size_mismatch.qvf` and `byte_order_big.qvf`. **No
`qvf_version` bump.**

**Background.** Format review P19 and P20. Spec § 2.1 already called orphans
"the expected by-product of updating an archive in place" (2026-10-07
record), but nothing said how an update is made, which manifest a consumer
reads afterwards, or what becomes of the previous one, and the `job.spec`
lifecycle (§ 5.9) and live checkpoints rely on exactly that. Appending is the
standard way to change a ZIP archive without rewriting it: the new entries go
where the old central directory began, and a new central directory lists
everything. The manifest is the only entry an update has to replace under the
same name, because consumers find it by name. Member sizes were available only
from the ZIP central directory, so a consumer that holds just the manifest
could not plan a read, and nothing cross-checked the manifest against the
container. The byte order was stated only in prose.

**The `manifest.json` duplicate exception clarifies the 2026-10-07 rule.**
That record made duplicate entry names an error because readers disagree on
which copy wins. For `manifest.json` this record removes the disagreement by
fixing the winner: the last central-directory record, which is also what
Python's `zipfile` and the C++ and JavaScript reference readers already
resolve a duplicated name to, and which a reader that walks the local headers
in order also meets last, because an update writes its manifest last.
Validators therefore report `E_ZIP_DUPLICATE_ENTRY` for any duplicated name
except `manifest.json`, and `W_ORPHAN_MEMBER` for each superseded copy of it.
Every other duplicate stays an error; an update writes a changed member under
a new name instead.

**Rationale for no bump.** All three are the additive case of "Change
process": `size` and `byte_order` are optional fields that no archive
carried, and § 2.3 specifies a use of the container the format already
tolerated (orphans have been valid since the 2026-10-07 record). The only
archives whose verdict changes are those with several `manifest.json`
entries, which were rejected before and are now valid with warnings; none of
the reference writers produced one, and every frozen corpus archive validates
unchanged. Moving the byte order into the dtype names (`<f8`), which would
describe each member, changes the dtype vocabulary every consumer validates
and stays a v2 candidate (format review V2-01).

**Compatibility.** Producers: write `size` when you can (the reference
writers do), write `byte_order` as `"little"` or not at all, and follow § 2.3
when you append; repack to drop the orphans before publishing an archive if
you prefer it without them. Consumers: read the last `manifest.json`, ignore
entries the manifest does not reference, and, if you rely on `size`, check it
against the central directory first. The Python, C++ and JavaScript readers
now open an archive with superseded manifests (they used to refuse it as
ambiguous), refuse a `byte_order` other than `"little"` and a member whose
`size` disagrees with its entry. Validator authors: no new code; the meanings
of `E_ZIP_DUPLICATE_ENTRY`, `W_ORPHAN_MEMBER`, `E_BINARY_SIZE` and
`E_MEMBER_MALFORMED` are widened in Appendix B. A conformance runner must
accept a positive archive that validates with warnings.

### 2026-10-08 — Trajectory extensions: a cell in the structure convention, velocities, gradient, times, binary energies, float32 MD coordinates

**Decision.** A `trajectory` section gains seven optional binary members,
each float64 and in the frame order of `coords` (spec § 5.5): **`lattice`**,
`[3, 3]` (one cell) or `[n_frames, 3, 3]` (one cell per frame), in the
**structure convention**, rows in Å; **`velocities`**, `[n_frames, n_atoms, 3]`
in Å/fs; **`gradient`**, `[n_frames, n_atoms, 3]`, the energy gradient dE/dR in
hartree/bohr, so the forces are its negative, as for the `forces` kind;
**`times`**, `[n_frames]` in fs; **`energies`**, `[n_frames]` in hartree, the
binary form of `metadata.energies`; **`cell_volumes`**, `[n_frames]` in Å³; and
**`temperatures`**, `[n_frames]` in kelvin. A `reaction.path` gains the four
shared members `velocities`, `gradient`, `times` and `energies` and keeps its
own `lattice` (columns, bohr). `coords` **MAY** be float32 when
`metadata.precision` is `"float32"`: the schema's `TrajectoryCoords` dtype
widens from `float64` to `float64 | float32`, and `precision`, which already
had to agree with the dtype, now has a consequence. The metadata gains
`ensemble` (`nve`, `nvt`, `npt`, `nph`, `custom`) and `timestep_fs` (frame `k`
is at `k × timestep_fs` when `times` is absent); `kind` keeps its enum, with
`scan` covering relaxed scans; and `pbc` becomes **REQUIRED** when a trajectory
carries `lattice`, a `true` entry requiring `lattice`, mirroring § 5.1. The
fixed units of the new members are encoded in the metadata `units`
definitions (`times` and `timestep_fs` `fs`; for a trajectory `lattice`
`angstrom`, `cell_volumes` `a3`, `temperatures` `kelvin`). Validators (Python,
C++, JavaScript) check, with `E_MEMBER_MALFORMED`: the extents of every
per-frame member against `coords`, float32 `coords` without `precision`
`"float32"` (or a `precision` naming the other dtype), binary `energies` that
differ from `metadata.energies` by more than 1e-10 hartree, and a trajectory
`lattice` without `pbc` or a periodic `pbc` without `lattice`. The writers gain
Python `add_trajectory(..., lattice=, velocities=, gradient=, times=,
energies=, cell_volumes=, temperatures=, precision=, units=)` and
`add_reaction_path(..., velocities=, gradient=, times=, energies=)`, C++
`TrajectoryExtras` with the `add_trajectory` and `add_reaction_path`
overloads that take it and the `Lattice` / `std::vector<Lattice>` trajectory
overloads, and the C ABI `qvf_add_trajectory`, the generic trajectory entry
it lacked. The conformance corpus gains `md_periodic.qvf`, and the negative
corpus `trajectory_lattice_without_pbc.qvf`,
`trajectory_float32_without_precision.qvf`,
`trajectory_velocities_frame_count.qvf` and
`trajectory_energies_disagree.qvf`. **No `qvf_version` bump.**

**Background.** Format review P12 and D06: `trajectory` carried coordinates
and a JSON list of energies, so a periodic molecular-dynamics run had nowhere
to put its cell, velocities, forces or time axis, and long runs paid for
float64 positions and JSON energies they did not need. `reaction.path` already
had a cell, but in a convention of its own.

**`trajectory.lattice` adopts the structure convention.** Two conventions for
a periodic cell existed (spec § 4.3, "Lattice conventions"):
`structure.lattice_vectors` (rows, Å) and `reaction.path.lattice` (columns,
bohr). Units are frozen within v1, so the second cannot change, but a new
carrier may choose: `trajectory.lattice` takes the structure convention, so
that the frame-0 cell of a trajectory equals the `lattice_vectors` of its
structure element for element, and new carriers converge on one layout as
D06 proposed. § 4.3 now states the three carriers side by side and is the one
place that states both conventions; a validator reads each lattice in its own
convention and never compares or converts one against another. Unifying
`reaction.path.lattice` remains a v2 ledger item (V2-05).

**The gradient, not the forces.** P12 sketches a per-frame `forces` member in
Eh/bohr. Codes disagree on the sign of what they call forces, and the
`forces` kind (2026-10-08 record above) already settled the question by
storing the gradient. The per-frame member follows it and is named
`gradient`, for what it stores, so that the sign cannot be misread.

**Rationale for no bump.** Every member is optional, the new metadata fields
are optional, and widening the dtype of `coords` accepts more. The two rules
that can turn a previously accepted archive into a rejected one restate what
the format already meant: `precision` was already required to agree with the
dtype of `coords`, and a trajectory had no cell before this record, so no
archive carried `lattice` without `pbc`. No reference writer and no frozen
corpus archive is affected: all of them validate unchanged.

**Compatibility.** Producers: purely additive; a producer that writes float32
coordinates must set `metadata.precision`, which the reference writers do.
The Python `add_reaction_path` and C++ `add_reaction_path` keep float64
coordinates and refuse a metadata `precision` of `"float32"`; the reference
writers check every rule above before writing. Consumers: read
`trajectory.lattice` as rows in Å (never as `reaction.path.lattice`), negate
`gradient` to draw forces, and take the time of frame `k` from `times`, else
`k × timestep_fs`; a consumer that does not understand a member ignores it.
Schema users: the trajectory and reaction-path metadata definitions now share
`$defs/TrajectoryMetadataCommon`, so that each can fix the unit of its own
`lattice`; `TrajectoryMetadataPayload` keeps its name and meaning. Validator
authors: no new finding code; `E_MEMBER_MALFORMED` covers the new rules
(Appendix B).

**Two readings to confirm.** The brief left the code of the `pbc` rule open;
it is reported as `E_MEMBER_MALFORMED` at the `metadata` member, like the
other trajectory rules, rather than as `E_STRUCTURE_PERIODICITY`, whose name
and rule (§ 5.1) belong to the `structure` payload. Second, the precision rule
applies to `reaction.path` too, because the two kinds share `TrajectoryCoords`
and the metadata `precision` field. Both readings are open to revision by the
maintainer before the next release.

### 2026-10-08 — A unit vocabulary; typed `units` declarations; legacy spellings accepted

**Decision.** QVF names units with a controlled vocabulary of 25 **unit
names** (spec § 4.3, "Unit names"): `angstrom`, `bohr`, `hartree`, `ev`,
`kcal_per_mol`, `kj_per_mol`, `cm-1`, `nm`, `debye`, `mhz`, `hz`, `ppm`,
`km_per_mol`, `a3`, `bohr3`, `gpa`, `fs`, `ps`, `kelvin`, `atm`, `e`, `amu`,
`radian`, `degree` and `dimensionless`, each with its quantity and its
definition (CODATA 2018 where a measured constant enters). The schema carries
the list as `$defs/UnitName`, the single source of truth that the spec table,
the Python `qvf_units` module, the C++ header `cpp/include/qvf/units.hpp`
and the JavaScript reader repeat and are tested against. Every payload
`units` object (the
volume grid, which gains one, the trajectory, reaction-path and scan
metadata, `kpath`, DOS `meta`, Fermi `mesh`, `qpath`, phonon `meta` and
QTAIM), the 1-D spectrum's `x_units`, `y_units` and `broadening.units`, and
the `units` of a `{value, units}` object are typed as unit names. A
declaration of a unit that § 4.3 fixes may only restate it, which the
payload definitions encode as `const` values (a grid `units.origin` is
`bohr`, a trajectory `units.coords` is `angstrom`). Spec § 4.3, "Declared
units", states the semantics: producers MAY declare units, a declaration is
not a conversion, consumers MUST honour a declared unit, a contradiction of a
fixed unit is an error, and an unknown unit string is the new warning
`W_UNIT_UNKNOWN`, reported by all three reference validators. The conformance
corpus gains `units_declared.qvf`, the negative corpus `unit_unknown.qvf` and
`unit_contradicts_fixed.qvf`. **No `qvf_version` bump.**

**Background.** `ValueWithUnits.units` was free text, so `"Eh"`, `"hartree"`
and `"Hartree"` were all valid and all different to a consumer
(`docs/format_review.md` D12), and the optional `units` objects added by the
payload definitions said in their descriptions that their vocabulary was not
yet fixed (P02). The list names units the format already uses; it changes no
unit of § 4.3, so a payload that declares nothing means what it meant.

**Legacy spellings.** The reference writers wrote `"Eh"` and `"eV"` in
`provenance.scf_energy` and `provenance.fermi_energy`, and frozen corpus
archives carry them. `ValueWithUnits.units` therefore accepts one of two
disjoint enums, `oneOf` `UnitName` and `$defs/LegacyUnitName` (`Eh`, `eV`,
`Hartree`, annotated `deprecated`), and only there: consumers MUST read a
legacy spelling as `hartree` or `ev`, producers SHOULD write the unit name,
and the reference writers now do. `hartree` itself is a unit name, not a
legacy spelling, so the two enums stay disjoint as `oneOf` requires. The
older free-text keys (`unit` of a scan axis, `reaction_coordinate_unit`,
`energy_unit`, …) stay free text, because frozen archives carry spellings
such as `"rad"` there.

**Unknown units: a warning that the schema makes an error.** The unit-name
rule is a semantic check every validator runs, with or without a JSON Schema
engine, and on its own it is a warning, so a validator without one (the C++
and JavaScript references) accepts such an archive with `W_UNIT_UNKNOWN`. The
same strings are typed by the schema, so a validator that runs it also
reports `E_SCHEMA` or `E_PAYLOAD_SCHEMA` and rejects the archive; the negative
corpus records both, and the runners hold the schema-only code only against
a validator that ran the schema, as for the other payload archives.

**Rationale for no bump.** Typing a field that was free text, and naming the
units the format already uses, is additive in the sense of "Change process":
no member, kind or default unit changes, nothing becomes required, and every
archive the reference writers emitted before and every frozen corpus archive
validates unchanged.

**Compatibility.** Producers: a payload `units` value outside the vocabulary,
which the schema used to accept, now fails validation with
`E_PAYLOAD_SCHEMA` (in `provenance`, `E_SCHEMA`); no reference writer and no
corpus archive wrote one. The vocabulary has no name yet for compound units
such as e·bohr⁻³, Å⁻¹ or states·eV⁻¹, so fields in those units (QTAIM `rho`,
for example) cannot be declared until a later additive revision adds names
(spec § 9). Writers: Python `add_volume`, `add_trajectory` and
`add_scan_surface` gain `units=`, and `add_topology_qtaim(units=)` and
`add_spectrum(x_units=, y_units=)` now check their names; C++ gains
`Grid::units` and `Spectrum::x_units` / `y_units`, and checks a `units`
object in JSON metadata; all of them refuse an unknown name or a
contradicted fixed unit and convert nothing. `set_provenance(scf_energy_eh=,
fermi_energy_ev=)` writes `hartree` and `ev`. Consumers: read declared units,
and map a legacy spelling to its unit name. Validator authors: add
`W_UNIT_UNKNOWN` (Appendix B); codes are added, never renamed.

### 2026-10-08 — Root `energies` block; `forces` and `hessian` canonical kinds

**Decision.** Three additions under v1. (1) An optional root block
**`energies`** (spec § 3.6, `$defs/Energies`, strict): `total` (required, the
energy of the archive's primary result), `method`, `components` (twelve named
contributions such as `nuclear_repulsion`, `exchange_correlation`,
`dispersion` and `zpve`, plus `x_` keys) and `by_method` (the energy by
further methods), every energy a `{value, units}` object that producers
**SHOULD** write in `hartree`. Where the manifest states one energy twice the
statements **MUST** agree within 1e-8 hartree: an energy of `energies` stated
for the method `provenance.method` names agrees with `provenance.scf_energy`,
and `components.zpve` with `thermochemistry.zpve_eh`. Validators report a
disagreement with the new finding code **`E_ROOT_INCONSISTENT`** ("two root
blocks state different values for the same quantity"). (2) A canonical kind
**`forces`** (spec § 5.10): `gradient`, float64 `[n_atoms, 3]` in hartree/bohr,
which is the energy **gradient**, so the forces are its negative; an optional
`stress`, float64 `[3, 3]` in hartree/bohr³ for periodic systems; an optional
strict `meta` (`$defs/ForcesMetaPayload`). (3) A canonical kind **`hessian`**
(spec § 5.11): `matrix`, the symmetric Cartesian Hessian, float64 `[3N, 3N]`
in hartree/bohr² with index 3·atom + xyz, and an optional strict `meta`
(`$defs/HessianMetaPayload`: `mass_weighted`, `projected`, `atom_masses_amu`,
which a mass-weighted matrix requires). `vibrations` gains the optional
section field `hessian_ref`. Validators check the gradient, stress and matrix
shapes, and the atom count against the section's structure (the section-level
`structure_ref`, else `meta.structure_ref`, else the archive's only
`structure`) with `E_MEMBER_MALFORMED`, and resolve `meta.structure_ref` and
`hessian_ref` with `E_REF_UNRESOLVED`. The conformance corpus gains
`forces_hessian.qvf`, and the negative corpus `forces_atom_count.qvf`,
`hessian_not_square.qvf` and `energies_inconsistent.qvf`. **No
`qvf_version` bump.**

**Background.** Format review P05, P06 and P07: an archive could carry a
wavefunction and a spectrum but not the energy of the calculation beyond
`provenance.scf_energy`, nor the gradient or the Hessian, which every
optimisation and frequency calculation produces and which a viewer draws as
force arrows or turns into normal modes. Gradient sign conventions differ
between codes; fixing the stored quantity as the gradient, with the force its
negative, removes the ambiguity rather than leaving it to a flag.

**Rationale.** Both kinds are added directly by maintainer decision under the
change process ("clear ecosystem need exists and a reference consumer will
render it"), as `run.record` and `job.spec` were, and are noted in
`registry.json`. A new root block, new canonical kinds, a new optional section
field and a new finding code are all the additive case: nothing that was valid
becomes invalid, except an archive that carries the new `energies` block
inconsistently, which no archive did before this record.

**Two readings to confirm.** P05 compares `provenance.scf_energy` with "the
`by_method` entry of `provenance.method`", which presumes that
`provenance.method` names the method of `scf_energy`. Spec § 3.6 states that
reading, so for a calculation beyond SCF `provenance.method` names the SCF
reference and the final method is named by the new optional
`energies.method`. That field, not in P05's sketch, is what makes "`total`
is the SCF energy" checkable: `total` is compared only when `energies.method`
is the provenance method. Second, when neither the section nor `meta` names a
structure, a `forces` or `hessian` section belongs to the archive's only
`structure` section; with several structures and no reference, no atom count
is checked. Both readings are open to revision by the maintainer before the
next release.

**Compatibility.** Producers: purely additive. The reference writers gain
Python `set_energies`, `add_forces`, `add_hessian` and `add_vibrations(...,
hessian_ref=)`, C++ `set_energies`, `add_forces`, `add_hessian` and a trailing
`hessian_ref` argument of `add_vibrations`, and the C ABI
`qvf_set_energies_json`, `qvf_add_forces` and `qvf_add_hessian`. Consumers
that do not support the kinds treat them per the unsupported-kind rules; a
consumer drawing forces **MUST** negate the stored gradient. Validator
authors: `E_ROOT_INCONSISTENT` is a new code; the comparison converts
`hartree` (and `Eh`, `Ha`), `ev`, `kcal_per_mol` and `kj_per_mol` to hartree
and skips other units. The forces and Hessian `units` objects may only
restate `hartree` and `bohr`.

### 2026-10-07 — `atom_properties`: more scalar roles, vector roles, and custom roles with a `meta` description

**Decision.** `atom_properties` keeps one role list, held by
`$defs/SectionAtomProperties` and mirrored verbatim by the spec's § 5 table,
a new role table in § 5.7, both reference writers and the C ABI. The list
grows from five roles to 24. Seventeen are new scalar roles, float64
`[n_atoms]`: the charges `cm5_charge`, `chelpg_charge`, `resp_charge`,
`esp_charge`, `nbo_charge`, `mbis_charge`, `bader_charge` and `ddec_charge`,
and `valence`, `bonded_valence`, `fukui_plus`, `fukui_minus`, `fukui_zero`,
`dual_descriptor`, `atomic_energy_eh`, `atomic_volume_bohr3` and
`nmr_shielding_iso_ppm`. Two are vector roles, float64 `[n_atoms, 3]`
(`$defs/AtomVectorBinary`): `magnetic_moment` and `force_eh_bohr`. The § 5.7
table gives each role's quantity, unit and convention: a `*_charge` is a net
charge in `e`, positive for electron loss; `spin_population` is an electron
count, α minus β; the Fukui functions are dimensionless condensed
differences of atomic populations and `dual_descriptor` is `fukui_plus` −
`fukui_minus`; `magnetic_moment` is in μB, a collinear producer putting the
moment on the quantisation axis; `force_eh_bohr` is the energy gradient in
hartree/bohr, not the force (force = −gradient), and the `forces` kind is
authoritative when both are present. A role named `custom.<name>` (`<name>`
is `[A-Za-z0-9_]+`) is also valid, scalar or vector, and only together with a
JSON member `meta` (`$defs/AtomPropertiesMetaPayload`, strict) that maps each
custom role to `{units, description, kind}`, `kind` being `scalar` or
`vector`. The schema requires `meta` whenever a custom role is present and a
role besides `meta`; validators report a `custom.*` member without an entry
in `meta` as `E_MEMBER_MALFORMED`, the existing code, in Python, C++ and
JavaScript, with or without a JSON Schema engine. Python
`add_atom_properties` takes every role by keyword plus `custom=` and `meta=`.
C++ gains `add_atom_vector_property` and `set_atom_properties_meta`, and its
map overload and `add_atom_property` take the enlarged scalar list and scalar
custom roles; the C ABI gains `qvf_add_atom_vector_property` and
`qvf_set_atom_properties_meta_json`, and `qvf_add_atom_property` takes the
enlarged scalar list and scalar custom roles. The conformance corpus gains
`atom_properties_extended.qvf` and four negative archives. **No
`qvf_version` bump.**

**Background.** Five roles covered three charge schemes and the spin
population, so a producer with CM5, CHELPG, RESP or Bader charges, reactivity
indices, NMR shieldings or magnetic moments had no schema-valid place for
them, and no per-atom vector existed at all (`docs/format_review.md` P10).
Each addition would have needed a revision of the format. The `custom.<name>`
escape lets a producer ship a quantity at once, and the required `meta` entry
keeps it self-describing: a consumer that does not know the role can still
show its `description` and `units`.

**Rationale for no bump.** New optional members are the additive case in
"Change process". The five roles keep their names, shapes and meaning, and
every archive that was valid stays valid: a member the closed `members`
object did not allow could not appear in a conforming archive before, so no
conforming archive changes meaning. A consumer that does not know a new role
ignores it, as it does any member it does not understand.

**Compatibility.** Producers: nothing changes for a valid archive; a writer
that adds a role the list lacks uses `custom.<name>` and a `meta` entry.
Validator authors: a `custom.*` member without an object entry named like it
in `meta` is now an error, and so is a section with a custom role and no
`meta` member (a schema error as well); a vector role of another shape, and a
role outside the list that is not `custom.<name>`, were schema errors already
and stay so. The writers check what they write: both reject roles of
different lengths, vector roles that are not `[n_atoms, 3]`, and custom roles
without a well-formed entry, before anything is written; the Python writer
derives `kind` from the array and rejects one that disagrees. Two points are
left open for the maintainers. First, the spec requires `kind` to agree with
the member's `shape`, but no validator checks it. Second, the role
`force_eh_bohr` carries a gradient although its name says force; the name is
the one P10 fixed, and § 5.7 states the sign convention instead.

### 2026-10-07 — Root `metadata` block; section `structure_ref`, `run_ref` and `group`

**Decision.** The manifest gains two optional additions. First, a root
object **`metadata`** (`$defs/Metadata`, new spec **§ 3.5**) that describes
the archive itself: `title`, `description`, `created_utc`, `modified_utc`,
`uuid`, `authors` (`$defs/MetadataAuthor`: a required `name`, and `orcid`,
`affiliation`, `email`), `license` (an SPDX identifier recommended), `doi`,
`keywords`, `derived_from` (uuids or DOIs) and `tags`. The block and the
author object are strict: another key is an error unless it begins with `x_`.
The two time stamps must be ISO 8601 UTC ending in `Z`, and `uuid` an RFC 4122
UUID in lowercase 8-4-4-4-12 form; the schema checks both (`E_SCHEMA`).
`email` and `affiliation` are personal data, which producers SHOULD NOT emit
by default. Second, every section object may carry **`structure_ref`** (the
`structure` section it describes), **`run_ref`** (the `run.record` of the run
that produced it) and **`group`** (a free display label) (spec § 4.1). Both
reference validators report a `structure_ref` that does not name a
`structure` section, or a `run_ref` that does not name a `run.record`, as
`E_REF_UNRESOLVED`, and now also report a `reaction.waypoints`
`trajectory_ref` that names a section of another kind, which spec § 6 already
forbade but no validator checked. No finding code is added. The kind-specific
references (`operand_a` / `operand_b`, `trajectory_ref`, and the
`structure_ref` of a `wavefunction.gto` `basis` payload) stay. The conformance
corpus gains `multi_structure.qvf`, and the negative corpus
`structure_ref_wrong_kind.qvf`, `run_ref_unresolved.qvf`,
`trajectory_ref_wrong_kind.qvf` and `metadata_uuid_invalid.qvf`. **No
`qvf_version` bump.**

**Background.** Several structures in one archive were legal but ambiguous
(format review P04): only the `basis` payload of a wavefunction named its
structure, so a consumer guessed from section order which geometry a volume
or spectrum belonged to, and a multi-step archive could not say which run
produced which result. An archive also had no title, licence, authorship or
identity of its own (P03), which publication, citation and provenance
tracking (`derived_from`) need.

**Precedence rule.** When a section carries a `structure_ref` and one of its
payloads carries one too, **the section's takes precedence**: a consumer reads
the payload's atom indices against the structure the section names (spec
§ 4.1, § 8.2). The section object is in the manifest, so the rule needs no
member to be opened, and it is one rule for every kind, including payload
references that later revisions may add. Producers SHOULD keep the two equal;
a disagreement is not a validation finding.

**Automatic metadata in the reference writers.** The Python writer
(`QvfWriter(..., auto_metadata=True)`), the C++ writer (the public field
`QvfWriter::auto_metadata`, default `true`) and therefore the C ABI add
`created_utc` and a random version-4 `uuid` to every archive unless the caller
set them, as spec § 3.5 recommends; both values are made once per writer. The
manifests the writers produce therefore differ from run to run. The positive
conformance generator turns the default off, so the frozen archives keep their
manifests, and `multi_structure.qvf` sets fixed values explicitly; no test
compares regenerated archives byte for byte.

**Rationale for no bump.** An optional root member and optional section fields
are the additive case in "Change process". No valid archive carried a root
`metadata` key, because the root object is closed. The section object was
open, so an archive could already carry a `structure_ref`, `run_ref` or
`group` key with a meaning of its own; none of the reference writers, the
conformance corpus or the reference producer does. A `trajectory_ref` that
names a non-trajectory section was already invalid by § 6.

**Compatibility.** Producers: nothing changes for an archive that does not
use the new fields. Consumers MAY ignore `metadata` and `group`, and MUST
apply the precedence rule. A consumer that validates manifests against a
vendored copy of the schema must update that copy: the root object is closed,
so an older copy rejects a `metadata` block with `E_SCHEMA`, and archives from
the updated reference writers carry one by default. A producer that must stay
readable by such a consumer in the meantime can turn the automatic metadata
off. Writers: Python `set_metadata(**fields)` checks every field and
`annotate_section(section_id, structure_ref=, run_ref=, group=, label=)`
refuses a reference to a missing section or one of the wrong kind; C++
`set_metadata(Json)`, `set_section_field(id, key, value)` and
`annotate_section(id, structure_ref, run_ref, group)`; C ABI
`qvf_set_metadata_json` and `qvf_annotate_section`. No `add_*` signature
changed.

### 2026-10-07 — Payload schemas for the spectra and analysis members; provisional QTAIM position unit

**Decision.** The JSON payloads of the spectra and analysis kinds are
specified by schema definitions, and validators check each member against the
definition for its section kind and role: `$defs/SpectrumPayload` (the
`spectrum` member of `spectra.ir`, `.raman`, `.uvvis`, `.ecd`, `.vcd` and
`.generic`), `$defs/NmrSpectrumPayload` and `$defs/EprSpectrumPayload`
(`spectra.nmr`, `spectra.epr`), `$defs/BondsPayload` (the `bonds` section and
the optional `bonds` member of `structure`), `$defs/BondOrdersPayload`,
`$defs/ScfHistoryPayload` (`scf_history.iterations`), `$defs/QtaimPayload`
(`topology.qtaim.critical_points`) and `$defs/SymmetryPayload`
(`structure.symmetry.data`). Each has a strictness level, defined in spec
§ 5.4: **strict** for `bonds`, `bond_orders` and `scf_history` (no unlisted
keys except `x_`-prefixed extensions), **typed-open** for the 1-D spectrum,
and **permissive** for NMR, EPR, QTAIM and symmetry. Spec § 5.4 and § 5.7 now
describe each payload in a table. The 1-D spectrum gains optional `x_units`,
`y_units`, an `x` alias of `frequencies`, `peaks`, `broadening`,
`temperature_k`, and the `raman` and `uvvis` blocks; `frequencies` keeps its
name and carries the x axis in the kind's § 4.3 unit unless `x_units`
declares another, and `frequencies`, `intensities` and `x` must have one
length. `topology.qtaim` gains an optional `units` object: positions are in
the unit `units.position` declares, and in **ångström when it is absent**. The
conformance corpus gains `spectra_all.qvf`, `nmr.qvf`, `qtaim.qvf` and
`symmetry.qvf`, and the negative corpus three payload archives. **No
`qvf_version` bump.**

**Background.** Only the `structure` and `job.spec` payloads had definitions;
every other JSON member payload was prose (format review D07), so two
producers could diverge without a validator noticing. The x values of a
spectrum are called `frequencies` even where they are energies (D18), and
QTAIM positions had no declared unit (D05). The typed fields are the ones the
reference writers, the conformance corpus, and the reference producer and
consumer already use. Where the reference producer writes a field under a
second name, that name is typed as well: `atom_pair` on a QTAIM point and
`atoms` / `path` on a bond path, and `space_group_number` /
`space_group_symbol` beside `spacegroup_number` / `international` in a
symmetry payload, where both spellings must agree when both are present.
Choosing one spelling is left to the revision that tightens the permissive
payloads.

**QTAIM position unit: provisional.** The unit is maintainer decision 2 in
`docs/format_review.md` and has not been taken. Until it is, the default is
ångström: the unit of the `structure` whose atoms the points overlay, and the
unit the reference producer already writes. Changing a default unit later
would be an incompatible unit change under the versioning policy above, so
producers **SHOULD** declare `units.position` now; an archive that does means
the same thing whichever default the decision fixes.

**Rationale for no bump.** Typing content that was already allowed, and
adding optional fields, is the additive case in "Change process". Nothing
becomes required that the documented shapes did not already carry, and every
archive the reference writers emit and every corpus archive validates
unchanged.

**Compatibility.** Producers: archives that follow the documented shapes are
unaffected. Payloads that were malformed but accepted now fail validation
with `E_PAYLOAD_SCHEMA`: a QTAIM point whose `type` is not `bcp`, `rcp`, `ccp`
or `ncp`, a spectrum whose arrays differ in length, a non-integer atom index,
an unlisted key in a strict payload. Extension keys in a strict payload need
the `x_` prefix. Consumers **MUST NOT** read spectrum values in the kind's
default unit when `x_units` declares another, and **MUST** read QTAIM
positions in a declared `units.position`. Validator authors: the equal-length
rule of a 1-D spectrum cannot be expressed in JSON Schema and is checked
directly; the reference Python validator does so with or without a schema
engine. The C++ reference validator has no schema engine and, as before,
reports `W_SCHEMA_SKIPPED` instead of `E_PAYLOAD_SCHEMA`; the conformance
runners therefore require the code only from a validator that ran the schema
check. The Python writer's `add_spectrum` refuses arrays of different lengths
and accepts `x_units`, `y_units`, `peaks` and `broadening`;
`add_topology_qtaim` accepts `units`.

### 2026-10-07 — Payload definitions for the geometry and wavefunction members

**Decision.** The schema gains payload definitions for the JSON members that
carry geometry and wavefunction data, and spec § 6 now requires every JSON
member whose section kind and role have a payload definition to conform to
it. The new definitions are `VolumeGridPayload` (the `grid` member of the
eight `volume.*` kinds and of `basis.ao`, and the `volume_grid` member of
`reaction.path`), `BasisPayload` and `MoMetadataPayload` (`wavefunction.gto`),
`TrajectoryMetadataPayload`, `ReactionPathMetadataPayload`,
`ReactionWaypointsPayload`, `ScanSurfaceMetadataPayload` and
`VibrationsMetadataPayload`, with the item definitions they use. Spec § 5.2,
§ 5.3 and § 5.5 describe each payload in a table, and § 5 names two
strictness levels. **Strict** payloads (`grid`, `basis`, `mo_metadata`,
`waypoints`) accept only their listed keys plus keys beginning with `x_`;
**typed-open** payloads (the trajectory, reaction, scan and vibration
`metadata`) type their listed keys and accept any other. The Python reference
validator reports a violation as `E_PAYLOAD_SCHEMA`; the C++ validator has no
JSON Schema engine and is unchanged. **No `qvf_version` bump.**

**Background.** Only the `structure` and `job.spec` payloads had definitions.
Every other JSON member was described in prose, so two producers could
disagree on a key's name, type or unit without any validator noticing. That
was the largest gap between "documented" and "checkable" in v1
(`docs/format_review.md` D07, proposal P01).

**What was accepted, and why.** Every payload the reference writers emit
conforms unchanged, and so does every existing corpus archive. Keys that producers and the vibe-view consumer already use are
accepted as optional typed fields rather than by loosening a definition: the
`atoms` object list beside `symbols`, flat scan-axis labels
(`axis_a_label`, `coordinate_a_label`, `coordinate_a_unit`, …), and in
`mo_metadata` the `occupation_semantics`, `coefficient_encoding` /
`coefficient_components`, per-channel `symmetry_labels`, and the
localized-orbital descriptors. `mo_metadata.n_mo` is optional because a
producer in use omits it. Fields that later proposals need are named now,
with their shapes: `basis.ecp` and `basis.n_electrons`, the grown `spin` and
`orbital_kind` enums, `complex`, `symmetry_labels` and `state_ref` (P09), and
trajectory `kind`, `pbc` and `precision` (P12). The radial convention of the
ECP `powers`, the coefficient layout of a `generalized` wavefunction, and the
target of `state_ref` are not yet specified. New per-mode vibration fields
carry fixed units: km·mol⁻¹ for `ir_intensities`, Å⁴·amu⁻¹ for
`raman_activities`, amu for `reduced_masses` and mdyn·Å⁻¹ for
`force_constants`. An imaginary mode is written as a negative frequency.

**Rationale for no bump.** Adding definitions for members that already exist
is additive: no member, kind or unit changes, and nothing becomes required
that the spec did not already describe as carried. The reference writers'
output and every archive in the corpus are valid before and after.

**Compatibility.** Validator authors: the payloads were previously
unconstrained, so a *malformed* value that used to pass validation now fails
it. Examples are a grid `shape` with two entries, a negative basis exponent,
an unrestricted `mo_metadata` without `alpha` and `beta`, a waypoint `kind`
outside the five listed values, and an unknown key in a strict payload. The
equal length of a shell's `exponents` and `coefficients` is required but is a
semantic rule the schema cannot check. Producers: a strict payload may carry
private additions only under keys beginning with `x_`. A single-set
`mo_metadata` must carry `energies` and `occupations`, which the spec always
listed as carried; the reference writers write zeros when no orbital energies
are known. Consumers: nothing changes; payloads that validated before mean
what they meant.

### 2026-10-07 — Phonon optional members, one `atom_properties` role list, DOS section fields named

**Decision.** Three places where the schema, the spec and the reference
writers disagreed now agree, additively. `phonon_bands` gains an **optional**
binary member `eigenvectors`, float64 `[n_q, n_modes, n_atoms, 3]` when real or
`[n_q, n_modes, n_atoms, 3, 2]` when complex (last axis `[re, im]`, as for
complex `wavefunction.gto` coefficients), and `phonon_dos` an **optional**
binary member `projected`, float64 `[n_channels, n_points]`, whose rows a
`channels` array in the `meta` member labels (spec § 5.6). `atom_properties`
has one role list, `mulliken_charge`, `loewdin_charge`, `hirshfeld_charge`,
`iao_charge` and `spin_population`, held by `$defs/SectionAtomProperties` and
mirrored verbatim by the spec's § 5 table and § 5.7, both reference writers
and the C ABI. The fields `dos.total` and `dos.projected` carry on the section
object, `n_spin`, `smearing`, `smearing_type`, `fermi_energy_ev`,
`n_electrons` and, on `dos.projected`, `channels`, are named as optional typed
properties of `$defs/SectionDOSTotal` and `$defs/SectionDOSProjected` and
documented in § 5.6. The conformance corpus gains `phonons.qvf` and
`atom_properties_all.qvf`. **No `qvf_version` bump.**

**Background.** `add_phonon_bands(eigenvectors=…)` and
`add_phonon_dos(projected=…)` in both reference writers wrote members that the
closed `members` objects of the two phonon kinds did not allow, so every
archive that used them failed schema validation (`docs/format_review.md` D03).
The lists of `atom_properties` roles disagreed: three in § 5.7, four in the
§ 5 table, five in the schema, and three in the writers and the C ABI (D02). Both
writers emitted the DOS fields, but they rode the open section object unnamed,
so their types were not part of the contract (D08).

**Rationale for no bump.** New optional members and newly named optional
fields are the additive case in "Change process". The two phonon members could
not appear in a schema-valid archive before, so no conforming archive changes
meaning, and `hirshfeld_charge` and `iao_charge` were already in the schema;
the spec and the writers now say what the schema already accepted.

**The section object stays open.** Naming the DOS fields does not close it: an
unnamed field on a `dos.*` section still validates. Whether each `Section`
branch becomes closed once its peer fields are named is decision 3 in
`docs/format_review.md` and remains open.

**Compatibility.** Producers: nothing changes for an archive that was valid.
Validator authors: the DOS fields were unconstrained, so a *malformed* value
that used to pass now fails, such as `n_spin: 3`, a negative `smearing`, or a
`channels` item whose `spin` is not 0 or 1. Neither reference writer emitted
one. The writers now check what they write: both reject phonon eigenvectors or
a projected phonon DOS of the wrong rank, trailing axes, dtype or extent, and
`atom_properties` roles of different lengths, before anything is written.
Python `add_atom_properties` gains `hirshfeld_charge=` and `iao_charge=`; the
C++ writer gains an overload taking a map from role to values and
`add_atom_property`; the C ABI gains `qvf_add_atom_property`. v1 does not fix
the normalization or mass weighting of phonon eigenvectors; the member stores
them as the producer computed them.

### 2026-10-07 — Payload definitions for the periodic kinds and the run-record `files` index; spin axis on projections

**Decision.** The schema defines the payloads of eight JSON members, and a
validator with a JSON Schema engine checks each member against its definition
(`E_PAYLOAD_SCHEMA`): `bands.kpath` (`$defs/KpathPayload`, with
`$defs/KpathSegment`), `fermi_surface.mesh` (`$defs/FermiMeshPayload`),
`phonon_bands.qpath` (`$defs/QpathPayload`), `equation_of_state.fit`
(`$defs/EosFitPayload`, whose `model` is one of `birch_murnaghan`,
`murnaghan`, `vinet`, `poirier_tarantola`, `polynomial`) and `run.record.files`
(`$defs/RunRecordFilesPayload`) are **strict**: only the listed fields plus
vendor keys beginning `x_`. The `meta` of `dos.coop` / `dos.cohp`
(`$defs/DosMetaPayload`) and of `phonon_dos` (`$defs/PhononMetaPayload`) are
**typed-open**: the listed fields are typed and further keys are allowed.
Projection channels share `$defs/DosChannel` (`atom`, `l`, `m`, `spin`, and a
required `label`; open). Spin-resolved projections carry spin as a leading
axis: `bands.projections` may be `[n_spin, n_k, n_bands, n_channels]` and
`dos.projected.projections` `[n_spin, n_channels, n_points]`; the existing
lower-rank shapes stay legal and mean `n_spin = 1`. `bands` gains an optional
`occupations` member, float64 `[n_spin, n_k, n_bands]`. Both reference
validators report a `bands` section whose `projections` or `occupations`
disagree in shape with its `eigenvalues` as `E_MEMBER_MALFORMED`. Spec § 5.6
is rewritten around per-member payload tables with a *Spin axis* rule, and
§ 5.8 gains a table for the `files` index. **No `qvf_version` bump.**

**Background.** These payloads were prose only (D07 in the format review), so
two producers could name the same field differently and no validator would
notice. The reference consumer already read keys the format never named:
`kpath.labels`, `segments`, `channels` and `title`, the Fermi mesh's
`lattice_vectors` and `band_indices`, the COOP/COHP `pair_labels`, and the
`files` entries' `description` and `truncated`. Band and DOS projections had
no spin axis (D04): eigenvalues were `[n_spin, n_k, n_bands]`, but fat-band
weights could not say which spin they belonged to, and producers already wrote
`[n_spin, n_channels, n_points]` DOS projections that the format did not
describe.

**Rationale for no bump.** Every change is additive: a new optional member,
new optional fields, definitions for members that already existed, and new
legal shapes beside the old ones. The definitions were written against what
the reference writers, the reference producer and the conformance corpus
emit, so all of those archives validate with their bytes unchanged. Where
producers write one quantity under two names (`fermi` and `fermi_energy_ev`
in `kpath`, `shape` and `nk1`/`nk2`/`nk3` in the Fermi mesh), both names are
recorded with the rule that they agree, rather than one being made invalid.
No field becomes required except `model` in an EOS `fit` and `label` in a
channel object, which every known producer writes.

**Compatibility.** Producers: nothing changes for an archive that already
matches the tables; new archives **SHOULD** use the spin axis whenever
`eigenvalues` have two spin channels. Consumers: read a lower-rank projection
as `n_spin = 1`, and a string in `kpath.channels` as a channel with that
`label`. Validator authors: the eight members were unconstrained, so a
malformed payload that used to pass now fails, as in the 2026-07-26 decision
below — the realistic cases are an EOS `model` outside the five forms, a key
outside a strict table, and a `files` entry with a field other than
`filename`, `description`, `media_type` and `truncated`. `bands.projections`
must now be float64, as the reference writers always wrote it. A validator
without a JSON Schema engine (the C++ reference validator) checks the shapes
but not the payloads, so a negative-corpus expectation of `E_SCHEMA` or
`E_PAYLOAD_SCHEMA` binds only a validator that runs the schema. The reference
writers (Python and C++ `add_bands`, `add_dos_projected`) accept both ranks
and an `occupations` argument and refuse shapes that disagree; the C ABI is
unchanged. The channel object is open, so the `atom_index` and `symbol` keys
some producers write on `dos.projected` channels stay valid; whether
`atom_index` or `atom` is the name to standardise is left to the record that
types those channels.

### 2026-10-09 — Constant arrays and ZIP integrity

**Decision.** Remove compression ratio from the default validity limits in
§ 6.1. Keep the absolute per-entry, total, JSON-size and JSON-depth caps, and
reject streams that do not inflate to their recorded size or match their ZIP
CRC-32. Python, C++ and JavaScript agree. The reference APIs retain an explicit
positive `max_ratio` as a stricter application policy; its default is now `0`
(disabled), and `E_COMPRESSION_RATIO` keeps that meaning for callers opting in.
The negative corpus's `compression_ratio` case tests only that opt-in policy;
`entry_size_understated`, `member_too_large`, `total_too_large`, and the two CRC
cases retain the mandatory malformed-input and resource-limit coverage.

**Correction to the 2026-10-07 rationale.** Its assertion that no reference-writer
archive is affected was incorrect: a valid 64³ float64 zero spin-density grid
compresses beyond 1000:1 and was rejected. Raising the ratio threshold does not
bound decompression better than the absolute and declared-size limits. The
C++ container reader already checks CRC-32 for both STORE and DEFLATE members;
new corpus controls cover members without an optional SHA-256. No
`qvf_version` bump: valid archives become readable while existing corrupt-input
rejection remains enforced.

### 2026-10-07 — Validation hardened for untrusted archives; stable finding codes

**Decision.** Spec § 2.1 and § 6 now state the rules a validator needs to read
untrusted input safely, and both reference validators enforce them. These are
**errors**: duplicate ZIP entry names; member paths (manifest `path` values
and ZIP entry names) containing a backslash or NUL, beginning with `/` or a
drive letter, or with a `..` segment; archives over the resource caps
(per-member and total uncompressed size, compression ratio, JSON size, and
JSON nesting depth); and a section whose `id` is `auto_open` or `bookmarks`,
the two reserved `viewer_defaults` keys. ZIP entries that no member references
("orphans") are **warnings**. Every finding carries a stable code listed in the
new informative **Appendix B** (`E_…` errors, `W_…` warnings), shared by the
Python and C++ reference validators, and the conformance suite gains a
negative corpus (`conformance/negative/`) with one must-fail archive per rule.
**No `qvf_version` bump.**

**Background.** The reference validators parsed untrusted archives without
checking for duplicate entry names (ZIP readers disagree on which copy wins),
Windows-style or absolute paths that escape an extraction directory, JSON
nested deeply enough to exhaust a recursive parser, or archives that inflate
far beyond their size on disk. `viewer_defaults` keys its per-section hints by
section id in the same object that holds `auto_open` and `bookmarks`, so a
section with either id was ambiguous. Findings were free-text strings that
tools could only match by substring, and the two reference validators did not
share a vocabulary.

**Rationale for no bump.** These rules clarify and harden validation
semantics; they do not change the format. The only archives newly rejected are
malformed ones: an archive with duplicate entries or an escaping path never
had a single well-defined reading, a reserved-id section was already
ambiguous, and the reference caps sit far above what producers write (the
8 GiB per-member cap was already the § 2.1 recommendation). Orphans stay
valid, so an archive updated in place keeps validating. The schema is
unchanged, and no archive the reference writers produce is affected; the
writers now also refuse the reserved ids and the newly forbidden paths.

**Resource caps are SHOULD values.** A validator reports an exceeded cap as an
error, and **SHOULD** use the reference values: 8 GiB per entry, 64 GiB in
total, a ratio of 1000 (STORE entries and entries under 1 MiB exempt),
256 MiB per JSON document, and nesting depth 64. Producers **SHOULD** stay
within them so their archives validate everywhere.

**Compatibility.** Producers: nothing changes for a conforming archive.
Consumers **SHOULD** refuse duplicate entry names and apply the caps before
decompressing or parsing. Validator authors **SHOULD** emit the Appendix B
codes; a code is never renamed or reused. The Python `validate_qvf()` keeps its
`errors` / `warnings` string lists (each now prefixed `CODE: `) and adds a
`findings` list; the C++ reader keeps `validate()` and adds
`validate_findings()`. `qvf_version: 2` is still reported as an error
(`E_VERSION`): producers **MUST NOT** emit it, while consumers may still open
such archives per the 2026-07-10 decision below.

### 2026-10-07 — `viewer_defaults` hints `render` and `transfer_function` recorded (spec only)

**Decision.** Spec **§ 3.4** describes `viewer_defaults`: the members
`auto_open` and `bookmarks`, which are reserved and may not be used as section
ids, and the per-section hint object with its keys. It records two keys a
consumer already read but the format never named: **`render`**
(`"isosurface"` or `"volume"`) and **`transfer_function`** (a `range`, `color`
and `opacity` nodes, optional `preset` and `colormap`), with the rules a
consumer cleans a function by. It
states the consumer rule for every hint: a consumer MAY ignore a hint, MUST
ignore one it does not understand, and MUST NOT refuse a file because a hint
is malformed. The change is **spec only**; `$defs/ViewerHint` is unchanged.
**No `qvf_version` bump.**

**Background.** vibe-view's volume mode reads both keys, and its manual
documented them "ahead of the specification", so a producer had only a
consumer's manual to go by and a second consumer could not learn the keys from
the format. The recorded shapes and rules are those of vibe-view's shipped
parser. Where the proposal's prose differed, the parser was followed: a
`transfer_function` whose `colormap` is not `"custom"`, including one with no
`colormap`, takes its palette from the section's `colormap` hint.

**Rationale for spec only.** `ViewerHint` is open
(`additionalProperties: true`), so both keys already validate and no archive
changes. Typing them in the schema would turn a malformed hint into a schema
error, and a schema error is fatal in the Python reference validator and in a
consumer that validates the manifest before opening, as vibe-view does. That is
the opposite of the consumer rule, and stricter than the parser the keys come
from, which forgives a capitalised `render` and drops a bad node rather than
the function. An unchanged schema also means no copy of it has to be
re-vendored downstream.

**Rationale for no bump.** Recording optional keys of an open object is the
additive case in "Change process"; nothing becomes required and no existing
archive changes meaning.

**Compatibility.** Purely additive. The four schema-typed keys (`isovalue`,
`colormap`, `opacity`, `replication`) keep their types, so a malformed value of
one of them still fails schema conformance and remains a producer error; under
the consumer rule a consumer nonetheless opens such a file and ignores the
hint. The hint object is defined independently of where it is keyed, so its
keys keep their meaning if a later major version moves per-section hints under
a sub-object of their own. Three further keys the same consumer reads, a
per-section `energy_window` and a top-level `camera` and `crossfade`, remain
unrecorded and are left to a follow-up change (recorded on 2026-10-15).

### 2026-10-07 — `provenance.hostname` deprecated; sharing and sanitisation guidance

**Decision.** The recognized provenance field `hostname` is **deprecated**
(spec § 3.2): producers SHOULD NOT emit it, consumers MUST ignore it for any
purpose other than display and MUST NOT rely on it, and it may be removed in a
future major version. The schema keeps the property, now annotated
`"deprecated": true`, and the `provenance` object stays open. The reference
writers drop a `hostname` key by default — Python `set_provenance(...,
allow_hostname=True)` and C++ `set_provenance(prov, /*allow_hostname=*/true)`
opt back in; the C ABI has no opt-in. Spec § 5.8 gains a "Sharing and
sanitisation" paragraph: the verbatim members of a `run.record` commonly carry
absolute paths, user names, and host names, sanitising them before an archive
is shared is the producer's or user's responsibility, and producers SHOULD
offer a way to suppress or redact such content. **No `qvf_version` bump.**

**Rationale.** QVF archives are meant to be shared, and a recognized
`hostname` field invited producers to write private operational detail into
every file, against the format's purpose and this repository's own
public-safety policy. The `run.record` guidance replaces a one-sentence
reminder with an explicit allocation of responsibility and names the intended
toolkit mechanism, a `scrub` operation with a reviewable report, which is
planned separately and changes nothing in the format.

**Compatibility.** Deprecating an optional field is additive: nothing becomes
invalid. Archives that already carry `hostname` still validate, and the field
remains listed so they stay interpretable. Producer-visible changes: the
reference writers now omit the key unless asked, and because
`qvf_set_provenance_json` must parse its string to find the key, malformed JSON
passed to it is now reported as an error instead of being written verbatim.

### 2026-08-05 — `basis.ao` Cartesian components named by `cartesian_powers`

**Decision.** `ao_metadata` gains an **optional** `cartesian_powers`
(`[l_x, l_y, l_z]`), present if and only if the AO is a component of a
Cartesian (`pure: false`) shell. `angular_momentum` keeps its shape and stays
REQUIRED; its second element is a magnetic quantum number **only** when
`cartesian_powers` is absent. Specified in spec **Appendix A.3**.
**No `qvf_version` bump.**

**Background.** `basis.ao` names one component of one shell, and the two
shell kinds of Appendix A.2 need different names. Only the spherical one was
specified. The reference producer derived `m` as *in-shell index* − `l`,
which is the A.2 spherical ordering and is simply undefined for a Cartesian
shell: it has `(l+1)(l+2)/2` components, not `2l+1`, so the subtraction ran
past `+l` and a Cartesian d shell was labelled `m = −2 … +3` across its six
components. Every value was schema-valid, so nothing caught it.

**Rationale for no bump.** A new optional member is the additive case in
"Change process" above. `angular_momentum`'s **required-ness and shape are
unchanged**, so no required-member change occurs and the compatibility
commitment below is not engaged. Element 0 (`l`) was and remains meaningful
for both shell kinds, so a v1 consumer reading only `l` is unaffected. No
conforming archive changes: `cartesian_powers` can appear only where the old
contract had no correct value to emit.

**Rationale for a separate member over widening `angular_momentum`.**
`[l, l_x, l_y, l_z]` would have overloaded one field with two meanings
distinguished by length, so an un-updated consumer reading element 1 as `m`
would keep getting a plausible-looking number (`[2,0,0]` → `m = 2`). A
distinct key makes the Cartesian case *detectable* rather than
indistinguishable, which is the property the old contract lacked.

**Compatibility.** Purely additive for producers; a spherical-only producer
emits exactly the bytes it did before. Consumers **MUST NOT** read
`angular_momentum[1]` as `m` when `cartesian_powers` is present. Validator
authors: the member is optional, so no existing archive is invalidated.

### 2026-07-26 — biomolecule metadata schema'd on `structure` (no v3)

**Decision.** The optional biomolecule fields on the `structure` section object
— `chains`, `residues`, `secondary_structure`, and the new `b_factors` — are
**schema'd** as optional properties of `$defs/SectionStructure`, with
`$defs/BiomoleculeResidue` and `$defs/BiomoleculeSecondaryStructure` giving
their item shapes, and `$defs/StructureAtom` gaining an optional per-atom
`b_factor`. Spec § 5.1 gains a normative precedence rule: a consumer that can
also derive residues / chains / secondary structure itself MUST prefer the
producer-supplied values, and per-atom `b_factor` wins over the section-level
`b_factors` array. Added **under v1** — every field is optional, which is the
additive case, so there is **no `qvf_version` bump**.

**Rationale.** vibe-view's cross-repo Ask 1 requested a
a separate `qvf_version: 3` manifest schema for these fields (and for the streaming
`run_status` / `checkpoint` / `partial` fields, which landed additively on
2026-07-02 and 2026-07-25). That request predates the versioning policy above
and the 2026-07-10 withdrawal of `qvf_version: 2` below, which settled exactly
this question: an optional member must not bump the major version, because the
specification requires a v1-only consumer to refuse a higher version, so a bump
would break conforming third-party consumers for a field they were free to
ignore. The three earlier fields already shipped unschema'd, riding the open
Section object; naming them in the schema is what turns "the writer happens to
emit this" into a contract two implementations can be held to, and it is what
Ask 1 actually needed.

**Compatibility.** Purely additive for producers. Note for validator authors:
the fields were previously unconstrained, so a *malformed* value that used to
pass validation now fails it — a `secondary_structure` range with a `type`
outside `helix`/`sheet`/`coil` being the realistic case. No known producer
emitted one; the reference producer now drops such a range rather than writing
it.

### 2026-07-25 — `job.spec` canonical kind + `run_status` lifecycle (executable archives)

**Decision.** A new canonical kind **`job.spec`** (spec § 5.9) carries a
declarative specification of the calculation an archive requests — `job_type`,
`method`, `basis`, `functional`, `charge`, `multiplicity`, `kpoints`, `tasks`,
and open engine-specific `options` — as a JSON member schema'd by
`$defs/JobSpecPayload`. In the same decision, `provenance.run_status` is
formalized in spec § 3.2 with the value set `pending | running | converged |
failed`, adding `pending` for archives that describe a job not yet run. Both
are added **under v1** — a new canonical kind and a newly recognized optional
provenance field are the additive case, so there is **no `qvf_version` bump**.

**Rationale.** With `run.record` (2026-07-24) an archive records what *ran*;
without a request counterpart it cannot describe what *should run*, so a
calculation's lifecycle could not live in one file. `job.spec` completes the
container model: structure + job.spec + `run_status: "pending"` in, results +
run.record + `run_status: "converged"/"failed"` out — the same archive updated
in place. The payload is deliberately declarative data, never code: a runner
reconstructs the job from typed fields, and the spec explicitly forbids
executing `run.record.input` as a side effect of opening or running an
archive, keeping "open a file" and "run untrusted code" apart. Added directly
per the change process ("clear ecosystem need exists and a reference consumer
will render it") as a maintainer decision, noted in `registry.json`.

**Compatibility.** Purely additive. Consumers that do not support the kind
treat it per the standard unsupported-kind rules. `run_status` values
`running` / `converged` / `failed` were already emitted by the reference
producer's live checkpointer under the open `provenance` object; `pending` is
new, and the schema now names the recognized value set.

### 2026-07-24 — `run.record` canonical kind added (self-contained archives)

**Decision.** A new canonical kind **`run.record`** (spec § 5.8) carries the
verbatim input of a program invocation, its full log/output, and auxiliary
attachments, plus a required `program` section field naming the code that ran.
Added **under v1** — a new canonical kind is the additive case, so there is
**no `qvf_version` bump**.

**Rationale.** A QVF archive already carries a calculation's results; without
the input and log it is not a self-contained record of the calculation. The
kind is producer-neutral by construction: `program` is independent of the root
`source.program`, so a converter (e.g. an ORCA-output packer) can record
faithfully which code the input belongs to. Added directly per the change
process ("clear ecosystem need exists and a reference consumer will render
it") as a maintainer decision, noted in `registry.json`.

**Compatibility.** Purely additive. Consumers that do not support the kind
treat it per the standard unsupported-kind rules; the opaque-bytes member
shape (no `dtype`/`shape`) follows the existing `citations.references`
precedent, so generic member handling is unaffected.

### 2026-07-10 — `qvf_version: 2` withdrawn (periodic `reaction.path`)

**Decision.** The `qvf_version: 2` bump is **withdrawn**. `reaction.path` gains an
optional `lattice` member under **v1**; a periodic reaction path is detected by
that member's *presence*, not by a version number.

**Background.** vibe-qc v0.10.0–v0.15.x stamped `qvf_version: 2` on any archive
containing a periodic `reaction.path`. The entire schema-visible difference from
v1 was one **optional** member (`lattice`); a companion `dim` field lives inside
a `JsonMember` payload and is not schema-visible at all.

**Rationale.** Under the versioning policy above, an optional member is the
*additive* case and must not bump `qvf_version`. The bump also created a real
interoperability break: the specification requires a v1-only consumer to refuse
any archive with a higher major version, so conforming third-party consumers were
obliged to reject archives the reference producer routinely wrote. Finally, a
major version is the format's scarcest signal — `qvf_version: 2` must mean "a v1
consumer genuinely cannot read this" — and spending the format's first bump on a
backward-compatible optional field would have permanently muddied that meaning.

**Compatibility.** Archives stamped `qvf_version: 2` exist in the wild (shipped
from v0.10.0 onward). They are v1-compatible. Therefore:

- Producers **MUST NOT** emit `qvf_version: 2`.
- Consumers **SHOULD** accept `qvf_version: 2` and treat it as `1`, for at least
  one minor line. This is a deprecation, not a deletion.
- [`spec/legacy/qvf_manifest_v2.schema.json`](spec/legacy/qvf_manifest_v2.schema.json)
  is retained as a frozen generated artifact so anything resolving its `$id`
  continues to validate (it was then a strict superset of v1; it is not of
  the v1 schema as it has grown since, and no validator uses it: see the
  2026-10-15 decision).

### 2026-07-10 — `structure` payload schema'd; `pbc` required when periodic

**Decision.** The `structure` member's JSON payload is now specified, as
`$defs/StructurePayload`. `pbc` is **REQUIRED** whenever `lattice_vectors` is
non-null, and `dimensionality` is optional and derived, with the invariant
`dimensionality == sum(pbc)`. **No `qvf_version` bump.**

**Background.** `pbc`, `lattice_vectors`, and `dimensionality` were prose-only,
and the prose made `dimensionality` a `MAY` on the *section object* while the
reference producer wrote it into the *payload*. A consumer could not assert that
a structure was a 2-D slab, only infer it — and inferring wrong renders a slab
inside a vacuum box.

**Rationale for no bump.** Absent `pbc` had no defined meaning; a periodic
archive that omitted it was already unrenderable. Every known producer emits it,
and every archive in the conformance corpus validates unchanged. This is the
*clarification* case in "Change process" above, not a required-member change.

**The periodic axes are deliberately not required to be the leading ones.**
`pbc = [true, false, true]` — a slab whose normal lies along *y* — is
conforming. Ordering the periodic axes first is an internal convention of one
producer (vibe-qc's periodic-system code); making it normative would have forced other
producers to permute their lattice, against the producer neutrality asserted at
the top of this document. It survives in the spec as an informative note
describing the legacy `dimensionality`-without-`pbc` fallback.

**Compatibility.** `dimensionality` moved from the section object to the
payload. Archives carrying it in the old position remain fully interpretable,
because `pbc` — which every such archive also carries — is what a consumer
reads. A consumer **MUST NOT** read `dimensionality` off the section object.

## The registry

`registry.json` is the machine-readable list of:

- **Canonical kinds** — the section kinds defined in the spec + schema. A drift
  test keeps this list in lock-step with the schema's `Section` branches, so the
  registry can never claim a kind the schema doesn't define (or vice versa).
- **Reserved names** — planning names not yet portable contracts.
- **Vendor namespaces** — registered `x_<vendor>.*` prefixes, their owner, and
  purpose. Registration is advisory (any code may use its own `x_<vendor>.*`
  without asking), but registering avoids collisions and signals intent to
  standardize.
- **Promotions** — vendor kinds that have become canonical.

## Change process

Anyone may propose a change by opening an issue or merge request against the
repository:

- **Register a vendor namespace** — add a row to `registry.json`
  (`vendor_namespaces`). Lightweight; no design review needed, it just records
  who is using which prefix.
- **Extend a canonical kind** (a new *optional* member) — a spec + schema change,
  reviewed for backward compatibility. No `qvf_version` bump.
- **Add a new canonical kind** — via the promotion process below, or directly
  when a clear ecosystem need exists and a reference consumer will render it.
- **Report a spec issue / ambiguity** — an issue; clarifications land in the spec
  without a version bump.

The maintainer reviews and merges. Decisions that affect the contract are
recorded in [`CHANGELOG.md`](CHANGELOG.md) and the registry.

## Vendor → canonical promotion

The path from an experimental `x_<vendor>.*` section to a standard kind
(spec § 5.7 / § 7.5):

1. The kind is used in production by **at least two independent producers** for
   at least one release cycle.
2. Its member contract is **stable** across those producers.
3. A spec + schema + validator change adds the canonical kind, and **at least one
   reference consumer** renders it.

This threshold is what makes a promotion *earned* by real interoperability rather
than declared. (The maintainer may promote ahead of the two-producer threshold
when a canonical target is needed to seed adoption — as was done for
`spectra.epr` and `wavefunction.bloch` — but such promotions are noted as
maintainer decisions in `registry.json`. A kind promoted before a second
producer ships it carries `"status": "provisional"` in its `promotions` entry
until one does; it is canonical in every respect, and any revision it needs
meanwhile is additive.)

## Compatibility commitments

- No canonical kind is **removed** or has its **required members changed** without
  a major `qvf_version` bump.
- A consumer may always **safely ignore** vendor (`x_<vendor>.*`) sections it does
  not understand, unless they are flagged `critical: true` (in which case a
  consumer that cannot support them must refuse to open the file, by design).
- Units and numeric conventions (spec § 3) are stable within a major version.

## Getting involved / becoming a supported producer

1. Emit valid QVF for your code's outputs and **self-certify** against the
   [conformance suite](conformance/README.md).
2. Register your vendor namespace in `registry.json` if you ship non-standard
   data.
3. Reach out — once your code emits conforming archives, it can be listed as a
   supported producer and its native format considered for direct
   [vibe-view](https://vibe-qc.com) support. See
   [`docs/integration_guide.md`](docs/integration_guide.md).

## Evolving the governance

The current model is maintainer-led because there is, so far, one reference
producer/consumer ecosystem. It is explicitly designed to broaden as the format
is adopted:

- As **independent producers** ship QVF, they earn a say in the promotion process
  (a promotion requires their real-world use, per the threshold above).
- If and when adoption warrants it, the spec + registry can move to a
  **vendor-neutral hosting home** with a small steering group of adopters. The
  standard is open regardless of where it is hosted — the Apache-2.0 license, the
  public schema, and this governance document travel with it — and vibe-qc's
  origin and stewardship remain acknowledged.

That is a decision for the maintainer and the adopter community to make together,
not a prerequisite for adopting QVF today.
