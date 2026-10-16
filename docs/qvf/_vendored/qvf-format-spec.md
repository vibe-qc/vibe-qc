# QVF format specification (v1)

> **Status:** Normative. This document specifies the QVF v1 container format
> and its producer/consumer contract. The machine-readable contract is the
> JSON Schema `qvf_manifest.schema.json` (shipped alongside this file); where
> prose and schema disagree, **the schema wins**. This is a self-contained copy of the
> specification maintained by the vibe-qc project, distributed under Apache-2.0
> so that any code can implement a conforming producer or consumer.

QVF (`.qvf`, "Quantum Visualization Format") is a ZIP-based container for
quantum-chemistry visualization and analysis data. It keeps one calculation's
structure, scalar fields, spectra, bands, trajectories, wavefunctions,
provenance, and viewer hints in a single random-access file, instead of
scattering them across Cube, XYZ, Molden, XSF, log, and program-specific
sidecar files. It is also a document format for the steps before a
calculation: the species of a reaction scheme with their chemical identity
(§ 5.1), the scheme that relates them (§ 5.14), the jobs to run on them
(§ 5.9) and a picture of the whole (§ 5.15) travel in one file that the
calculation's results later join.

The format is intended for the whole quantum-chemistry ecosystem. A QVF
**producer** may be any quantum-chemistry code. A QVF **consumer** may be a
structure viewer, a band plotter, a spectrum tool, a validator, a notebook
script, or a GPU viewer.

QVF v1 is deliberately **not** a restart format. It does not encode integral
caches, SCF internal state, engine-internal grids, or every possible
wavefunction representation. The optional `wavefunction.gto` section is a
visualization-oriented, Molden-like molecular-orbital carrier, not a universal
restart record.

The keywords **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** are used as in RFC 2119.

---

## 1. Design goals

1. **One shareable artifact.** A calculation's visualization data travels as one
   `.qvf` file.
2. **Random access.** A consumer reads only the members it needs.
3. **Typed payloads.** Every member declares format, dtype, shape, and a sha256
   checksum where applicable.
4. **Stable common vocabulary.** Common data types use canonical section kinds.
5. **Partial support.** A consumer can support a subset of kinds and still open
   the file. Unsupported vendor sections are reported, not misinterpreted.
6. **Producer neutrality.** `source.program` can name any code; core kinds are
   not producer-specific.

---

## 2. Archive model

A QVF file is a **ZIP archive**. The only mandatory member is `manifest.json` at
the archive root. Every other member is named *from the manifest* — member paths
are part of the manifest contract, not hard-coded by the container.

A typical archive:

```text
example.qvf
├── manifest.json          (REQUIRED — the only member read by name)
├── structure/
│   └── structure.json
├── wavefunction/
│   ├── basis.json
│   ├── mo_metadata.json
│   └── mo_coefficients.dat
├── volumes/
│   ├── density.dat
│   └── density_grid.json
├── spectra/
│   └── ir.json
└── citations/
    └── references.bib
```

Path layout is **not semantic**. Consumers **MUST** follow the `path` fields in
`manifest.json` and **MUST NOT** assume ZIP entry order or directory names.
Producers **SHOULD** use stable, readable directory names for debuggability.

### 2.1 Byte-level container requirements

* The archive **MUST** be a valid ZIP (PKZIP APPNOTE, deflate/store methods).
* `manifest.json` **MUST** be present at the archive root (path exactly
  `manifest.json`, no leading directory). In an archive updated in place,
  which holds more than one entry of that name, the manifest is the entry
  whose central-directory record comes last (§ 2.3).
* Producers **SHOULD** store `manifest.json` **uncompressed** (STORE method) so
  a consumer can read the table of contents cheaply; other members **SHOULD** be
  DEFLATE-compressed. Neither is mandatory — a consumer decompresses via the ZIP
  method recorded in the central directory regardless.
* Producers **SHOULD** write `manifest.json` last (or ensure its recorded
  checksums match the already-written member bytes). Because each member's
  sha256 is embedded in the manifest, all member bytes must be finalized before
  the manifest is serialized.
* Member paths — every `path` in the manifest and every ZIP entry name —
  **MUST** be relative and use `/` as the only separator: a path **MUST NOT**
  contain a backslash (`\`) or a NUL character, **MUST NOT** begin with `/` or
  with a drive-letter prefix (`^[A-Za-z]:`), and **MUST NOT** contain any `..`
  path segment. (Schema: `ZipPath` expresses the leading-`/`, NUL, and `..`
  rules; validators enforce all of them, § 6.)
* Each ZIP entry name **MUST** occur only once in the archive, with one
  exception: an archive updated in place holds its superseded `manifest.json`
  entries until it is repacked (§ 2.3). The ZIP format permits duplicate names
  and readers disagree on which entry wins, so an archive that duplicates any
  other name is ambiguous; a consumer **SHOULD** refuse to open it.
* Every ZIP entry other than the current `manifest.json` and directory entries
  (names ending in `/`) **SHOULD** be referenced by a member `path`. An
  unreferenced ("orphan") entry, a superseded `manifest.json` included, does
  not make the archive invalid — it is the expected by-product of updating an
  archive in place (§ 2.3) — and consumers never read it, because members are
  reached only through the manifest (§ 2). Validators report it as a warning
  (§ 6).
* A consumer **SHOULD** enforce resource caps before decompressing or parsing
  anything — per-member and total uncompressed size, compression ratio, and
  JSON size and nesting depth — to guard against zip-bomb and
  parser-exhaustion inputs. The sizes are recorded in the ZIP central
  directory, so they can be checked without decompressing; a reader **MUST
  NOT** inflate an entry past the uncompressed size recorded for it. § 6.1
  lists the reference values; the reference tools cap each member at 8 GiB
  (1 gigavoxel of float64).

### 2.2 Endianness

All binary members are stored in **little-endian** byte order. QVF v1 dtype
names (§ 4.2) do not carry a byte-order mark. Producers on big-endian
platforms **MUST** byte-swap to little-endian before writing.

The optional root field **`byte_order`** (§ 3.1) states this explicitly. Its
only valid value in v1 is `"little"`, and it documents what every v1 archive
already is: an archive without the field is little-endian too, and a consumer
**MUST NOT** read its absence as an unknown byte order. Any other value is
invalid — the JSON Schema rejects it, so a validator reports `E_SCHEMA` (§ 6)
— and a consumer **SHOULD** refuse an archive that carries one rather than
guess how to read its binary members. Producers **MAY** write the field; the
reference writers write it in every archive.

The tag belongs in the dtype names themselves (`<f8` rather than `float64`),
where it would describe each binary member. That changes the dtype vocabulary
every consumer validates, so it cannot happen within v1; it is recorded as a
candidate for the next major version (V2-01 in `docs/format_review.md`), and
`byte_order` is the v1 declaration until then.

### 2.3 Updating an archive in place

A producer may change an existing archive without rewriting it: a runner that
executes a `pending` `job.spec` archive and writes its results back (§ 5.9), a
program that saves live checkpoints of a running calculation (§ 3.2), a tool
that adds an analysis to a finished archive. The ZIP format allows this by
appending. This section fixes how a QVF archive is appended to, so that every
consumer reads an updated archive the same way.

An update:

1. **MAY append** new ZIP entries. The producer leaves the bytes of the
   existing entries where they are, writes the new entries after the last of
   them (where the old central directory began), and then writes a new
   central directory that lists every entry of the archive, old and new, in
   the order of their data.
2. **MUST write a new `manifest.json` as the last entry** of the archive,
   after every member it references; like any manifest it **SHOULD** be
   stored uncompressed (§ 2.1). The new manifest is complete: it lists every section of the
   updated archive, and its member specs reference the entries the update
   appended as well as the earlier entries it keeps, whose bytes, `sha256`
   and `size` (§ 4.2) are unchanged.
3. **MUST NOT** give a new entry the name of an existing entry, other than
   `manifest.json`. A member whose content changes is written under a new
   path, and the new manifest references that path.

The previous `manifest.json`, and every entry the new manifest no longer
references (a replaced member, the members of a removed section), are
**superseded**. They stay in the archive as orphans (§ 2.1): a consumer
**MUST** ignore them, and a validator reports each one as the warning
`W_ORPHAN_MEMBER`. They still count towards the total-size cap of § 6.1.

A consumer that reads `manifest.json` by name **MUST** use the entry whose
central-directory record comes last when the archive holds more than one
entry of that name. This is the one exception to the rule that entry names are
unique (§ 2.1): a validator reports `E_ZIP_DUPLICATE_ENTRY` for any duplicated
name except `manifest.json`, and for a duplicated `manifest.json` it reports
`W_ORPHAN_MEMBER` for each superseded copy. Because an update writes its
manifest last, a reader that walks the local headers in data order instead of
reading the central directory also meets the current manifest last.

**Repacking** writes the archive anew from its current manifest: every entry
the manifest references, then the manifest, and nothing else. It removes every
orphan and every superseded manifest and changes no member's bytes, so the
`sha256` and `size` values of the manifest stay valid. The reference toolkit
provides it as the `repack` operation (informative).

The updated manifest records what changed:

* `metadata.modified_utc` (§ 3.5) **SHOULD** be set to the time of the update;
  `metadata.created_utc` and `metadata.uuid` **MAY** be kept, because the
  archive is the same archive, changed.
* `provenance.run_status` (§ 3.2) follows the calculation's lifecycle: a
  `pending` archive becomes `running` while a runner writes checkpoints, and
  `converged` or `failed` when it settles.
* Each program run an update records is a `run.record` section (§ 5.8). When
  an archive carries several, their `sequence` fields order them: an update
  adds its run record with the next `sequence` and keeps the earlier ones
  unchanged.
* A `job.spec` section is kept verbatim (§ 5.9), so a settled archive still
  records what was asked.

Appending overwrites the old central directory, so an update interrupted
before it writes the new one leaves an archive that cannot be opened. A
producer that must not lose the previous state writes the updated archive to
a copy and renames it over the original (informative).

### 2.4 File identification

A QVF archive is identified to operating systems, file managers and web
servers by the names below. Every program that reads or writes QVF files
**SHOULD** declare them exactly as given, so that the desktop associates
`.qvf` files with one type whichever program installed it last.

| Identifier | Value |
|---|---|
| File name extension | `.qvf` (lowercase when written; matched case-insensitively) |
| Media type | `application/vnd.vibe-qc.qvf+zip` |
| macOS Uniform Type Identifier | `com.vibe-qc.qvf`, conforming to `public.zip-archive` and `public.content` |
| Type description | `QVF quantum-chemistry archive` |

The media type is in the IANA vendor tree with the registered `+zip`
structured-syntax suffix (RFC 6838, RFC 6839): a generic ZIP tool may open the
archive, and a program that knows the type reads it as QVF. It is the type a
web server sends for a `.qvf` file and the one a desktop entry or file
association names. A QVF archive has no magic number of its own beyond the ZIP
signature, and the manifest is written last (§ 2.1), so content sniffing cannot
tell it from other ZIP files; programs identify it by the extension, or by
finding `manifest.json` with a `qvf_version`.

`registry.json` carries the same values under `file_type`, and
`spec/file-type/` carries the declarations ready to copy (informative):

* `qvf-uti.plist` — the macOS type declaration, the dictionary that goes into
  an application's `UTImportedTypeDeclarations` (or
  `UTExportedTypeDeclarations`) array in `Info.plist`, with the type's
  description, conformance, the extension and the media type. One program of
  a family of applications exports the type and the others import it; the
  declarations are identical, so macOS resolves the type the same way
  whichever of them is installed.
* `qvf-mime.xml` — the freedesktop.org shared-mime-info definition, a
  subclass of `application/zip` matched by the glob `*.qvf`, for Linux
  desktops (installed into `$XDG_DATA_HOME/mime/packages/` or
  `/usr/share/mime/packages/` and activated with `update-mime-database`).

On Windows, the file association's `Content Type` value is the media type.

A picture of the archive, for a file manager's thumbnail or a quick-look
preview, is the `thumbnail` member of its first `preview` section (§ 5.15).

---

## 3. The manifest

### 3.1 Root object

```jsonc
{
  "qvf_version": 1,
  "schema_uri": "https://vibe-qc.org/spec/qvf/1/manifest.schema.json",
  "byte_order": "little",
  "source": {
    "program": "my-code",
    "version": "1.2.3",
    "calculation": "h2o_rhf_sto3g"
  },
  "sections": [ /* … */ ]
}
```

| Root field | Required | Meaning |
|---|:--:|---|
| `qvf_version` | **yes** | Integer format version. v1 = `1`. |
| `source` | **yes** | `{program, version, calculation}`, all strings; `program` and `version` non-empty. |
| `sections` | **yes** | Array of section objects (§ 4). |
| `schema_uri` | no | Canonical URI of the JSON Schema. Informational. |
| `byte_order` | no | Byte order of every binary member. The only valid value is `"little"`, which every v1 archive is whether it declares it or not (§ 2.2). |
| `provenance` | no | Calculation-level metadata (§ 3.2). |
| `metadata` | no | Descriptive metadata of the archive itself: title, authors, licence, identifiers, time stamps (§ 3.5). |
| `viewer_defaults` | no | Viewer hints: `auto_open`, camera `bookmarks`, the initial `camera`, a `crossfade` and per-section hint objects (§ 3.4). The keys `auto_open`, `bookmarks`, `camera` and `crossfade` are reserved, so no section may use one as its `id` (§ 4.1). |
| `thermochemistry` | no | Root thermochemistry block (§ 3.3). |
| `dipole_moment` | no | Root dipole block (§ 3.3). |
| `constraints` | no | Root geometry-constraint block (§ 3.3). |
| `energies` | no | Root energies block: the energy of the primary result, its components, and energies by method (§ 3.6). |
| `extensions` | no | Extension governance block (§ 7). |

Producers **MUST** emit `qvf_version: 1`. A consumer that only understands v1
**MUST** refuse to open a file whose `qvf_version` is an unrecognized *major*
version — with the one historical exception below.

> **Historical note — archives stamped `qvf_version: 2`.**
> vibe-qc **v0.10.0 – v0.15.x** stamped `qvf_version: 2` on archives containing
> a *periodic* `reaction.path`. That bump was a mistake: the only difference from
> v1 is a single **optional** member (`lattice` on `reaction.path`, § 5.5), which
> is the additive case that the versioning policy says must **not** bump the
> version. The bump was **withdrawn** by a governance ruling on 2026-07-10 (see
> `GOVERNANCE.md` § Version history).
>
> Such archives are v1-compatible. Consumers **SHOULD** accept `qvf_version: 2`
> and treat it as `1`. Producers **MUST NOT** emit it. A periodic reaction path
> is detected by the **presence of the `lattice` member**, never by the manifest
> version.
>
> A validator treats such an archive as `qvf_version: 1` for every check,
> the JSON Schema check included, and reports the warning
> `W_VERSION_WITHDRAWN` (§ 6): the archive is valid, and the warning records
> that its producer wrote a version producers must not write. A tool that
> updates such an archive in place (§ 2.3) writes `qvf_version: 1` and the v1
> `schema_uri` into its new manifest. Every other value of `qvf_version` is the
> error `E_VERSION`.

### 3.2 `provenance`

Optional and open (`additionalProperties: true`): a key that is not in the
table below is allowed, and a consumer ignores one it does not know. A
producer emits only what it knows. The recognised fields are typed by the
schema's root `provenance` property:

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `method` | string | — | Method of the calculation, e.g. `"rhf"`. Beyond SCF it names the SCF method whose energy `scf_energy` is (§ 3.6). |
| `functional` | string | — | Exchange–correlation functional of a DFT method. |
| `basis` | string | — | Basis-set name. |
| `charge` | integer | e | Net charge. |
| `multiplicity` | integer | — | Spin multiplicity 2S+1. |
| `n_electrons` | integer ≥ 0 | — | Number of electrons the calculation treats explicitly, the same quantity as `basis.n_electrons` (§ 5.3): the total electron count minus the core electrons an ECP replaces. |
| `n_scf_iterations` | integer ≥ 0 | — | Number of SCF iterations. |
| `scf_converged` | boolean | — | Whether the SCF converged. |
| `scf_energy` | `{value, units}` | `units` is a unit name (§ 4.3), conventionally `hartree` | Energy of the SCF method that `method` names. |
| `fermi_energy` | `{value, units}` | `units` is a unit name, conventionally `ev` | Fermi energy. |
| `wall_seconds` | number ≥ 0 | s | Wall-clock time of the calculation. |
| `dimensionality` | integer, 0–3 | — | Echo of the structure's periodic-axis count (§ 5.1): informational here, and never a substitute for reading `pbc`. |
| `run_status` | enum | — | The archive's place in a calculation's lifecycle; see below. |
| `checkpoint` | object | — | Informative bookkeeping of a `running` snapshot, with a monotonic `seq`; see below. The schema does not type it. |
| `software` | array of software objects | — | The programs and libraries that produced the results; see below. |
| `git_commit` | string, 7–64 hexadecimal characters | — | The producer's own source revision: a git commit hash, abbreviated or full (40 characters for SHA-1, 64 for SHA-256). Git prints lowercase. |
| `n_threads` | integer ≥ 1 | — | Threads the calculation ran with, per process. |
| `n_mpi_ranks` | integer ≥ 1 | — | MPI ranks (processes); 1 for a serial run. |
| `n_gpus` | integer ≥ 0 | — | GPUs the calculation used; 0 for none. |
| `started_utc` | string | UTC | When the calculation started, in the form below. |
| `finished_utc` | string | UTC | When the calculation finished, in the same form. |
| `hostname` | string | — | **Deprecated**; see below. |

`scf_energy` and `fermi_energy` carry the unit names of § 4.3; the legacy
spellings `Eh`, `eV` and `Hartree` are accepted there but deprecated.
`started_utc` and `finished_utc` take the form of the time stamps of the root
`metadata` block (§ 3.5): an ISO 8601 date and time in UTC,
`YYYY-MM-DDThh:mm:ssZ`, optionally with a fractional second, the trailing `Z`
required. A validator checks the form (§ 6), not the truth or the order of the
two. `n_threads`, `n_mpi_ranks` and `n_gpus` record what the run used; the
resources a `job.spec` asks for are hints (§ 5.9), and the two need not agree.
A `pending` archive has not run, so these fields and the time stamps, when it
carries them, describe the step that prepared it.

**Agreement with the sections.** `provenance.n_electrons`, when present,
**MUST** agree with `basis.n_electrons` of a `wavefunction.gto` section (§ 5.3)
that states one.

**`software`** is an array of objects, one per program or library, each
`$defs/ProvenanceSoftware` (strict: only the fields below and vendor keys
beginning `x_`). The root `source` names the QVF producer (§ 3.1) and
`run.record.program` the code that ran (§ 5.8); `software` lists every
component that shaped the numbers, such as an integral library or an
exchange–correlation library.

| Field | Type | Required | Meaning |
|---|---|:--:|---|
| `name` | string, non-empty | **yes** | Name of the program or library, e.g. `"integral-lib"`. |
| `version` | string | no | Version, as the component reports it. |
| `url` | string | no | Where the component is published, e.g. its home page. A public address only: machine names and internal addresses are private operational detail, like `hostname` below. |

```json
"provenance": {
  "method": "rks", "functional": "pbe",
  "software": [{"name": "my-code", "version": "1.2.3"},
               {"name": "integral-lib", "version": "2.9.0",
                "url": "https://example.org/integral-lib"}],
  "git_commit": "9f2b6c1d4e7a",
  "n_threads": 8, "n_mpi_ranks": 4, "n_gpus": 0,
  "started_utc": "2026-10-07T12:00:00Z", "finished_utc": "2026-10-07T12:07:31Z",
  "wall_seconds": 451.0
}
```

One provenance field has normative lifecycle semantics:

* **`run_status`** — one of `"pending"`, `"running"`, `"converged"`,
  `"failed"`; the archive's place in a calculation's lifecycle. `pending`
  describes a job that has not yet run — typically a container carrying a
  `structure` (§ 5.1) plus a `job.spec` (§ 5.9) and no result sections.
  `running` marks a live mid-run checkpoint snapshot (such snapshots may also
  carry an informative `checkpoint` bookkeeping object with a monotonic
  `seq`). `converged` and `failed` mark a settled terminal archive; a
  terminal archive whose producer embeds its own input/log also carries a
  `run.record` (§ 5.8). An absent `run_status` means "not stated" — a
  consumer **MUST NOT** infer a status, and in particular MUST NOT treat
  absence as `converged`. An archive moves from one status to the next by
  being updated in place (§ 2.3) or written anew. An archive that carries
  several `job.spec` sections (a job set, § 5.9) states the status of the set:
  `pending` until one of its jobs starts, `running` from then until every job
  has settled, then `converged` when every job converged and `failed` when at
  least one failed; the status of each job is the `run_status` of the
  `run.record` that names it (§ 5.8).

One provenance field is **deprecated**:

* **`hostname`** — the name of the machine the calculation ran on. Producers
  **SHOULD NOT** emit it. Consumers **MUST** ignore it for any purpose other
  than display and **MUST NOT** rely on it. It may be removed in a future
  major version. Archives are meant to be shared, and machine names are
  private operational detail rather than part of a calculation's result. The
  field stays listed so that archives which already carry it remain
  interpretable; its presence never makes an archive invalid. See also
  "Sharing and sanitisation" in § 5.8.

### 3.3 Root metadata blocks

These describe the calculation as a whole (not binary payloads), so they live at
the manifest root rather than as sections.

* **`thermochemistry`** — `zpve_eh`, `enthalpy_eh`, `entropy_cal_mol_k`,
  `gibbs_free_energy_eh` (Hartree / cal·mol⁻¹·K⁻¹), `temperature_k`,
  `pressure_atm`. All optional.
* **`dipole_moment`** — `total_debye`, `vector_debye` (3-vector, Debye),
  `origin`: either a conventional name (`"center_of_mass"`,
  `"center_of_nuclear_charge"`, `"origin"`) or an explicit 3-vector in **bohr**.
  This block stays the canonical place for a bare dipole. A `properties.electric`
  section (§ 5.13) may restate it beside the higher multipoles, and the two
  statements **MUST** then agree.
* **`constraints`** — `frozen_atoms` (0-based indices), `distance_constraints`
  (each `{atoms: [i, j], target_angstrom}`).
* **`energies`** — the energies of the calculation, specified in § 3.6.

### 3.4 `viewer_defaults`

An optional root object carrying hints for consumers that render the archive:
which sections to show first, camera bookmarks and the initial camera, a
cross-fade between two volumes, and how to draw each section.
Hints are suggestions, not data. A consumer **MAY** ignore any hint, **MUST**
ignore a hint it does not understand, and **MUST NOT** refuse to open a file
because a hint is malformed: it ignores that hint and opens the file as if the
hint were absent. A consumer that validates the manifest against the schema
before opening therefore does not refuse a file over a finding in
`viewer_defaults`. Producers **SHOULD** emit hints in the shapes and the
lowercase spellings given here.

| Member | Value |
|---|---|
| `auto_open` | Array of section ids to show first. The first `structure` section it lists is the archive's first structure (§ 4.1). |
| `bookmarks` | Ordered camera bookmarks (`$defs/Bookmark`): a `name`, an optional `description` and a `camera` (`$defs/Camera`) with `position`, `focal_point` and `view_up` in Å and exactly one of `view_angle` (perspective, degrees) or `parallel_scale` (orthographic). The list order is the playback order. |
| `camera` | The camera a consumer starts with, an object in the shape of a bookmark's `camera` (`$defs/Camera`); below. |
| `crossfade` | Two volume sections to cross-fade between and where the blend starts, `{a, b, blend}`; below. |
| any other key | A section id; the value is that section's hint object (§ 3.4.1). A key that names no section is ignored. |

`auto_open`, `bookmarks`, `camera` and `crossfade` are reserved: a section
**MUST NOT** use one as its `id`, because its hint object could not be told
apart from the reserved member. A validator reports a section that does
(§ 6).

**`camera`.** The view a consumer opens the archive with: `position`,
`focal_point` and `view_up`, Cartesian 3-vectors in Å, and exactly one of
`view_angle` (perspective, in degrees, between 0 and 180 exclusive) and
`parallel_scale` (orthographic, in Å, positive), the shape of
`$defs/Camera`. A consumer that cannot use the camera, because it is not
such an object or because the consumer draws no 3-D scene, ignores it. A
consumer **MAY** open an archive without a `camera` hint at the camera of its
first bookmark.

**`crossfade`.** An object that asks a consumer to show two volume sections
(`volume.*`, § 5.2) blended into each other, for example two orbitals or two
densities on one grid:

| Key | Required | Type | Meaning |
|---|:--:|---|---|
| `a` | yes | string | `id` of the first volume section. |
| `b` | yes | string | `id` of the second volume section. |
| `blend` | no | number, 0 to 1 | Where the blend starts: `0` shows `a` alone, `1` shows `b` alone. Default `0.5`. |

A consumer ignores a `crossfade` that is not an object or lacks `a` or `b`,
and one whose `a` or `b` does not name a section it can draw. It reads a
`blend` that is absent, not a finite number, or outside 0 to 1 as `0.5` and
still applies the pair.

`camera`, `crossfade` and the per-section key `energy_window` (§ 3.4.1) are
specified here only, like `render` and `transfer_function`: the schema
reserves `camera` and `crossfade` but does not check their values, a
validator **MAY** report a value that does not match this section as a
warning, and it **MUST NOT** fail an archive because of one.

#### 3.4.1 Section hint object

A section hint object (`$defs/ViewerHint`) suggests how one section is drawn.
It is defined independently of where it is stored: v1 keys it by section id
directly inside `viewer_defaults`, and the keys below keep their meaning
wherever a later version places the object. Every key is optional. The object
is open, so a consumer ignores keys it does not know.

| Key | Type | Meaning |
|---|---|---|
| `isovalue` | number | Contour level, in the section's data units (§ 4.3). |
| `colormap` | string | Name of the colour map to draw with. Names are not standardised; a consumer uses its default for a name it does not know. |
| `opacity` | number, 0 to 1 | Opacity of the drawn surface; in volume mode, the peak of the consumer's default opacity ramp. |
| `replication` | three integers ≥ 1 | Cells to draw along each lattice vector of a periodic structure. |
| `render` | string | How a scalar-field section is first drawn: `"isosurface"` (the default) or `"volume"`. |
| `transfer_function` | object | The function a scalar-field section is drawn with as a volume. |
| `energy_window` | `[min, max]` | The energy range a `bands` or `dos.*` section is first plotted over, in eV relative to the Fermi level. |

A scalar-field section is one a consumer draws from a 3-D grid: `volume.*` and
`basis.ao`. The schema types the first four keys, so a malformed value of one
of them fails schema conformance (§ 6). `render`, `transfer_function` and
`energy_window` are specified here only: a validator **MAY** report one that
does not match this section as a warning, and **MUST NOT** fail an archive
because of it.

**`render`.** `"isosurface"` contours the field at `isovalue`; `"volume"` draws
the field itself through a transfer function. A consumer **MAY** accept other
capitalisations and surrounding whitespace, and treats any value it does not
recognise as `"isosurface"`. A consumer that cannot draw the section as a
volume (it has no volume renderer, does not draw that kind as one, or the
field has no finite samples) ignores `render` and draws the section as it
otherwise would. In volume mode `isovalue` is unused. Without a
`transfer_function` the consumer builds its own default function for the kind,
with `colormap` as the palette and `opacity` as the peak of its opacity ramp;
with one, the function's opacity nodes decide the opacity and `opacity` is not
applied.

**`transfer_function`.** A JSON object, used whenever the section is drawn as a
volume, whether `render` or the user chose that mode:

| Key | Required | Shape | Meaning |
|---|:--:|---|---|
| `range` | yes | `[low, high]` | Two finite numbers, `high > low`, in the section's data units (§ 4.3). The window the function is defined over; samples outside it take the value at the nearer end. |
| `color` | yes | array of `[x, r, g, b]` | Colour nodes, at least one: `x` in data units, `r`, `g`, `b` from 0 to 1. |
| `opacity` | yes | array of `[x, a]` | Opacity nodes, at least one: `x` in data units, `a` from 0 to 1. |
| `preset` | no | string | Name of the opacity ramp the nodes were derived from, for a consumer's editor. Names are not standardised; the nodes, not the name, define the function. |
| `colormap` | no | string | `"custom"`: the `color` nodes are the palette. Absent or any other value: the palette follows the section's `colormap` hint (or the consumer's default), and the value names the colour map the nodes were sampled from; a consumer that cannot resolve that palette **MAY** draw the nodes as given. |

Between nodes, colour and opacity vary linearly with `x`; beyond the outermost
node they keep its value, and a node position outside `range` clamps to the
nearer end. A consumer drops a node that is not an array of the right length,
has a non-finite member, or has a colour or opacity outside 0 to 1, and sorts
the remaining nodes by `x`. It ignores the whole function, drawing with its
default as if none were given, when `range` is not two finite numbers with
`high > low` or when no colour node or no opacity node remains. Other keys are
ignored. A producer that wants its colour nodes drawn as given sets `colormap`
to `"custom"`.

A spin density that opens as a volume, with a diverging palette and both lobes
visible (`spin` is the section's `id`):

```json
"viewer_defaults": {
  "spin": {
    "render": "volume",
    "transfer_function": {
      "range": [-0.02, 0.02],
      "color": [[-0.02, 0.23, 0.30, 0.75], [0.0, 0.95, 0.95, 0.95], [0.02, 0.71, 0.02, 0.15]],
      "opacity": [[-0.02, 0.5], [-0.002, 0.0], [0.002, 0.0], [0.02, 0.5]],
      "colormap": "custom"
    }
  }
}
```

**`energy_window`.** The energy range a consumer first plots a `bands`
section or a `dos.*` section (§ 5.6) over, so that a producer can open an
all-electron spectrum, whose core states lie hundreds of eV below the Fermi
level, on its valence region. The value is `[min, max]`: two finite numbers,
`max > min`, in eV relative to the Fermi level (the band energies minus
`kpath.fermi`, and the DOS energy grid as stored, which is referenced to it).
A consumer **MAY** also accept the objects `{"min": …, "max": …}` and
`{"emin": …, "emax": …}`, which a producer **SHOULD NOT** write. It ignores
any other value, a non-finite number or a window with `max ≤ min`, and plots
as if no window were given. A consumer that draws a `bands` and a `dos.*`
section on one energy axis may apply the window of either. The user's own
choice of range overrides it, like any hint.

A periodic archive that opens at a saved view, with its band structure and
DOS on the valence region and two densities half cross-faded (`bands`,
`dos`, `rho_a` and `rho_b` are section ids):

```json
"viewer_defaults": {
  "camera": {"position": [0.0, -12.0, 4.0], "focal_point": [0.0, 0.0, 0.0],
             "view_up": [0.0, 0.0, 1.0], "view_angle": 30.0},
  "crossfade": {"a": "rho_a", "b": "rho_b", "blend": 0.5},
  "bands": {"energy_window": [-15.0, 10.0]},
  "dos": {"energy_window": [-15.0, 10.0]}
}
```

### 3.5 Root `metadata`

An optional root object (`$defs/Metadata`) that describes the archive itself,
so that it can be cited, found, and traced: what it is, who made it, under
which licence it may be used, and which identifiers name it and its sources.
The blocks of § 3.2 and § 3.3 describe the calculation; this one describes the
archive. Every field is optional. None of them changes how the rest of the
archive is read, and a consumer **MAY** ignore the block. The block is
**strict**: a key not listed below is an error unless it begins with `x_`,
which a producer may use for its own additions.

```json
"metadata": {
  "title": "Water: RHF/STO-3G density and orbitals",
  "created_utc": "2026-10-07T12:00:00Z",
  "uuid": "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
  "authors": [{"name": "A. Researcher", "orcid": "0000-0002-1825-0097"}],
  "license": "CC-BY-4.0",
  "keywords": ["water", "electron density"]
}
```

| Field | Type | Meaning |
|---|---|---|
| `title` | string | Short human-readable title. |
| `description` | string | Free-text description of the contents. |
| `created_utc` | string | When the archive was first written: an ISO 8601 date and time in UTC, `YYYY-MM-DDThh:mm:ssZ`, optionally with a fractional second (`…:00.25Z`). The trailing `Z` is required. |
| `modified_utc` | string | When the archive was last changed, in the same form. Absent means not changed since `created_utc`. |
| `uuid` | string | Identifier of this archive: an RFC 4122 UUID in lowercase hexadecimal 8-4-4-4-12 form, with a version digit from 1 to 8 and the RFC 4122 variant (the first digit of the fourth group is `8`, `9`, `a`, or `b`). |
| `authors` | array of author objects | People or groups responsible for the contents, in citation order. |
| `license` | string | Licence of the contents. An [SPDX](https://spdx.org/licenses/) identifier or expression (e.g. `"CC-BY-4.0"`) is **RECOMMENDED**. |
| `doi` | string | DOI under which the archive is published, without a resolver prefix (e.g. `"10.5281/zenodo.1234567"`). |
| `keywords` | array of strings | Subject keywords for search and cataloguing. |
| `derived_from` | array of strings | Archives or publications this archive was derived from, each a `uuid` or a DOI. |
| `tags` | array of strings | Free labels for a user's own organisation of archives; unlike `keywords`, not meant for publication. |

An author object (`$defs/MetadataAuthor`, strict like the block):

| Field | Required | Meaning |
|---|:--:|---|
| `name` | **yes** | Name as it should be cited. |
| `orcid` | no | ORCID iD, e.g. `"0000-0002-1825-0097"`. |
| `affiliation` | no | Institution. |
| `email` | no | Contact address. |

`email` and `affiliation` are personal data. Archives are meant to be shared,
so a producer **SHOULD NOT** emit either by default, but only when the person
concerned asks for it; a tool that sanitises an archive for sharing treats
them like the personal content of a `run.record` ("Sharing and sanitisation",
§ 5.8).

**Identity and time.** A producer **SHOULD** set `created_utc` and `uuid` in
every archive it writes, generating a random (version 4) `uuid` for each one;
the reference writers do so by default and let the caller turn it off, for
example to keep test vectors reproducible. An archive that its producer updates
in place (§ 2.3), such as a `job.spec` archive that moves from `pending` to
`converged` (§ 3.2), **MAY** keep its `uuid` and then **SHOULD** set
`modified_utc`. A tool that writes a new archive from an existing one (a
converter, a sanitiser, a subset) **SHOULD** give the new archive its own
`uuid` and list the source's `uuid` or DOI in `derived_from`. A validator checks
the form of the two time stamps and of `uuid` (§ 6), not their truth.

### 3.6 Root `energies`

An optional root object (`$defs/Energies`) stating the energies of the
calculation: the energy of the archive's primary result, a decomposition of
it into components, and the energies of the same system by further methods,
for example the SCF reference and the correlated energy of an MP2
calculation. Every energy is a `{value, units}` object
(`$defs/ValueWithUnits`), and producers **SHOULD** write `hartree`. The object
is strict: it carries only the fields below and vendor keys beginning `x_`.

| Field | Type | Required | Meaning |
|---|---|:--:|---|
| `total` | `{value, units}` | **yes** | Energy of the archive's primary result: the calculation its result sections (`wavefunction.gto`, `forces`, `hessian`, …) belong to. |
| `method` | string | no | Method of the primary result, e.g. `"rhf"`, `"rks"` or `"mp2"`. |
| `components` | object (`$defs/EnergyComponents`) | no | Named contributions to `total`; table below. |
| `by_method` | array of objects (`$defs/EnergyByMethod`) | no | The energy of the same system by each further method: `method` (string, required), `total` (`{value, units}`, required) and `components` (as above, optional). Strict, like the block itself. |
| `units_note` | string | no | Free-text note on the units, for people; it declares nothing. |

`components` (`$defs/EnergyComponents`, strict) names each contribution
by one of the keys below; every key is optional and its value is a
`{value, units}` object. A key beginning `x_` names a further,
producer-defined component and is a `{value, units}` object as well.

| Component | Meaning |
|---|---|
| `nuclear_repulsion` | Nuclear–nuclear repulsion energy. |
| `electronic` | Electronic energy: everything but `nuclear_repulsion`. |
| `one_electron` | One-electron energy: kinetic energy plus electron–nuclear attraction. |
| `two_electron` | Two-electron energy: Coulomb plus exchange (Hartree–Fock), or Coulomb plus exchange–correlation (Kohn–Sham). |
| `exchange_correlation` | Exchange–correlation energy of a density functional. |
| `exchange` | Exchange energy: Hartree–Fock exchange, or the exchange part of `exchange_correlation`. |
| `correlation` | Correlation energy: a correlated energy minus its reference energy, or the correlation part of `exchange_correlation`. |
| `dispersion` | Dispersion correction, e.g. DFT-D3. |
| `solvation` | Contribution of a solvent model. |
| `zpve` | Zero-point vibrational energy. |
| `thermal` | Thermal correction to the energy at `thermochemistry.temperature_k` (§ 3.3), excluding `zpve`. |
| `embedding` | Interaction with an embedding environment, e.g. QM/MM or an embedding potential. |

Components may overlap (`exchange_correlation` contains `exchange` and
`correlation`) and need not be exhaustive, so a consumer **MUST NOT** add
them up to obtain `total`.

**Consistency with the other root blocks.** Where the manifest states one
energy twice, the two statements **MUST** agree within 1e-8 hartree:

* `provenance.scf_energy` (§ 3.2) is the energy of the method that
  `provenance.method` names. Every energy that `energies` states for that
  same method — each `by_method` entry whose `method` is that name, and
  `total` when `energies.method` is that name — **MUST** agree with it.
  Method names are compared case-insensitively in ASCII, so `"RHF"` and
  `"rhf"` are one method. For a calculation beyond SCF, `provenance.method`
  accordingly names the SCF method whose energy `scf_energy` is, and
  `energies.method` the method of the final result. An energy of `energies`
  whose method is not named, or is another method, is not compared.
* `components.zpve` **MUST** agree with `thermochemistry.zpve_eh` (§ 3.3).

A validator converts both values to hartree before comparing them
(1 hartree = 27.211386245988 eV = 627.5094740631 kcal/mol =
2625.4996394799 kJ/mol, CODATA 2018), reports a disagreement as
`E_ROOT_INCONSISTENT` (§ 6), and skips a comparison when it does not
recognise a unit. The unit strings it recognises, case-insensitively, are
`hartree` (and its older spellings `Eh` and `Ha`), `ev`, `kcal_per_mol` and
`kj_per_mol`.

An MP2 calculation on an RHF reference:

```json
"provenance": {"method": "rhf", "scf_energy": {"value": -74.9629, "units": "hartree"}},
"energies": {
  "total": {"value": -75.0124, "units": "hartree"},
  "method": "mp2",
  "components": {"correlation": {"value": -0.0495, "units": "hartree"}},
  "by_method": [
    {"method": "rhf", "total": {"value": -74.9629, "units": "hartree"}},
    {"method": "mp2", "total": {"value": -75.0124, "units": "hartree"}}
  ]
}
```

The `rhf` entry restates `provenance.scf_energy` and must agree with it. The
MP2 `total` and the `mp2` entry are not compared with it, because their
method is not the one `provenance.method` names.

---

## 4. Sections and members

### 4.1 Section object

```jsonc
{
  "id": "vol_dens_0",
  "kind": "volume.density",
  "label": "Electron density",
  "members": { /* role -> member spec */ }
}
```

| Field | Required | Meaning |
|---|:--:|---|
| `id` | **yes** | Unique within the archive. `^[A-Za-z0-9_.-]+$`. **MUST NOT** be `auto_open`, `bookmarks`, `camera` or `crossfade` (reserved `viewer_defaults` keys, § 3.1, § 3.4). |
| `kind` | **yes** | Canonical kind (§ 5) or `x_<vendor>.*`. |
| `members` | **yes** | Map from member *role* to a JSON or binary member spec. |
| `label` | no | Human-readable display label. |
| `component` | no | Component hint (e.g. for `volume.orbital`). |
| `critical` | no | If `true`, a consumer that does not support `kind` **MUST** refuse to open the archive. Default `false` (§ 7.4). |
| `schema_uri` | no | URI of a JSON Schema for a vendor section's members. |
| `structure_ref` | no | Id of the `structure` section (§ 5.1) this section describes. |
| `run_ref` | no | Id of the `run.record` section (§ 5.8) of the program run that produced this section. |
| `group` | no | Free-text label that groups related sections for display, e.g. `"step 2: frequencies"`. |

Some kinds add kind-specific peer fields on the section object (e.g.
`volume.difference` carries `operand_a`/`operand_b`; `reaction.waypoints`
carries `trajectory_ref`). These are documented per kind in § 5.

**Section references and grouping.** Any section, of any kind including a
vendor kind, **MAY** carry `structure_ref`, `run_ref`, and `group`. They make
an archive that holds several structures or several steps of a calculation
unambiguous: a consumer learns which geometry a volume, wavefunction, or
spectrum belongs to and which program run produced it, instead of guessing
from section order or ids. `structure_ref` **MUST** name a section of kind
`structure`, and `run_ref` a section of kind `run.record`, in the same archive;
a validator reports one that does not (§ 6). A section without them makes no
statement, and a consumer keeps its usual behaviour, for example using the
archive's only `structure`, or the first structure of an archive with several
(below). `group` carries no meaning beyond grouping:
sections with equal `group` values belong together, and a consumer **MAY**
present them together, for instance as one node of a tree.

**Section ids carry no meaning.** An id names a section so that other
sections and `viewer_defaults` can refer to it; nothing else may be read
into it. `structure` is a conventional id that the reference writers use by
default, not a role: an archive may hold structures with any ids, such as
`structure_m1` and `structure_m2` for the species of a scheme, and none named
`structure`. A consumer **MUST** find sections by their `kind` and by the
references of § 4.1 and § 5, and **MUST NOT** depend on a conventional id.

**The first structure.** A consumer that shows one structure at a time, or
needs one structure for an operation that names none, uses the **first
structure**: the first `structure` section that `viewer_defaults.auto_open`
lists (§ 3.4), else the first section of kind `structure` in `sections`
order. It **SHOULD** also offer the archive's other structures, for example
by showing a structure when its section is selected. A producer that wants a
particular structure shown first lists it in `auto_open`. The rule chooses
what to show and changes no reference: a section without `structure_ref` in
an archive with several structures still makes no statement about which one
it describes. A standalone `bonds` section is the one exception: its atom
indices need a structure, and without `structure_ref` they address the first
structure (§ 5.1).

The kind-specific references stay as they are: `operand_a` and `operand_b` of
`volume.difference` (§ 5.2), `trajectory_ref` of `reaction.waypoints` (§ 5.5),
and the `structure_ref` inside the `basis` payload of `wavefunction.gto`
(§ 5.3) and `wavefunction.bloch` (§ 5.3.1). When a section carries a `structure_ref` and one of its payloads
carries one too, **the section's takes precedence**: a consumer reads the
payload's atom indices, such as the shell `center` indices of a `basis`,
against the atoms of the structure that the section names. Producers
**SHOULD** keep the two equal.

### 4.2 Member specs

**JSON member:**

```jsonc
{ "path": "structure/structure.json", "format": "json", "sha256": "<64 hex>",
  "size": 1834 }
```

**Binary member:**

```jsonc
{
  "path": "volumes/density.dat",
  "format": "binary",
  "dtype": "float32",
  "shape": [80, 80, 80],
  "sha256": "<64 hex>",
  "size": 2048000
}
```

* `format` is `"json"` or `"binary"`.
* `sha256` is the lowercase hex sha256 of the **exact bytes stored for that
  member** (the uncompressed payload). Consumers **MUST** verify it before use.
* `size` (optional, on every member spec: JSON, binary and opaque) is the
  **uncompressed byte count** of the stored member, an integer ≥ 0. It
  **MUST** equal the uncompressed size that the ZIP central directory records
  for the member's entry, so for a binary member with a `dtype` it is
  `itemsize(dtype) × ∏(shape)`. Producers **SHOULD** write it; the reference
  writers write it for every member. A validator checks it against the
  central directory before reading the member, and reports a disagreement as
  `E_BINARY_SIZE` (§ 6). A consumer that relies on `size` — to allocate a
  buffer before inflating, or to size-gate rendering (§ 5.8) — **MUST** first
  check that it equals the size the central directory records, and treat a
  member whose `size` disagrees as invalid. Archives written before `size`
  existed omit it and remain valid.
* JSON members **MUST** parse as UTF-8 JSON.
* Binary members are **raw, C-contiguous, little-endian** arrays. `dtype` is one
  of: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`,
  `float32`, `float64`. The stored byte length **MUST** equal
  `itemsize(dtype) × ∏(shape)`.
* A small set of **opaque binary members** carry no `dtype`/`shape`:
  `citations.references` (UTF-8 BibTeX bytes), the `run.record` members
  `input` / `log` (UTF-8 text) and `attachment.*` (arbitrary bytes, § 5.8),
  the `preview` members `thumbnail` and `image` (§ 5.15), and any member of a
  vendor section (§ 7.3) that its producer writes without them. Consumers
  skip the byte-length check for these and decode per the owning kind's
  contract.
* `media_type` (optional, on every member spec) is the IANA media type of the
  member's bytes, such as `image/png`, `text/plain` or `application/json`: a
  type and a subtype, without parameters (RFC 6838). It describes the bytes
  and changes nothing in how they are read: a JSON member is UTF-8 JSON and a
  binary member with a `dtype` is a little-endian array whatever its
  `media_type` says. A kind may require it, as `preview` does (§ 5.15);
  elsewhere it is a hint for programs that hand a member to another program,
  most usefully on opaque members.

### 4.3 Units and numeric conventions

Producers convert to these units at write time. Consumers **MUST** treat on-disk
units as the contract (they may convert for display).

| Field | Unit / convention |
|---|---|
| `structure.atoms[].position` | Ångström |
| `structure.lattice_vectors` | Ångström, row vectors |
| `trajectory.lattice` | Ångström, row vectors (the structure convention); see *Lattice conventions* below |
| `reaction.path.lattice` | bohr, column matrix (columns = a, b, c); see *Lattice conventions* below |
| `volume.*.grid.origin` | bohr |
| `volume.*.grid.voxel_vectors` | bohr per grid step |
| `volume.density.data` | e / bohr³ (electron density) |
| `volume.orbital.data` | orbital amplitude on a bohr grid |
| `volume.potential.data` | hartree / e (atomic units of potential) |
| `volume.rdg.data` | dimensionless, s(r) = \|∇ρ\| / (2(3π²)^⅓ ρ^⅔) |
| `volume.transition_density.data` | e / bohr³ (the transition density of one excitation, § 5.2) |
| `volume.laplacian.data` | e / bohr⁵ (∇²ρ); the vocabulary has no unit name for it, so `grid.units.data` is not declared |
| `volume.vector.data`, `volume.complex.data`, `volume.series.data` | the unit of the sampled field, declared as `grid.units.data` when the vocabulary names it; vector components are Cartesian, in the frame of `origin` |
| `volume.series` `axis.values` | `axis.units`; dimensionless when absent |
| periodic `volume.*.grid` axis `a` | `voxel_vectors[a] × (shape[a] − 1)` (`includes_endpoint` true, the default) or `voxel_vectors[a] × shape[a]` (false) equals lattice vector `a` of the grid's structure, in bohr (§ 5.2) |
| `trajectory.coords`, `reaction.path.coords` | Ångström |
| `trajectory.metadata.energies`, `trajectory.energies`, `reaction.path.energies`, `reaction.path.waypoints[].energy_eh` | Hartree |
| `trajectory.velocities`, `reaction.path.velocities` | Å / fs |
| `trajectory.gradient`, `reaction.path.gradient` | hartree / bohr; the energy gradient, so the forces are its negative (§ 5.5) |
| `trajectory.times`, `reaction.path.times`, `metadata.timestep_fs` | fs |
| `trajectory.cell_volumes` | Å³ |
| `trajectory.temperatures` | kelvin |
| `bands.eigenvalues`, `bands.kpath.fermi`, `fermi_energy_ev` | eV |
| `spectra.ir.frequencies` / `intensities` | cm⁻¹ / km·mol⁻¹ |
| `spectra.uvvis.frequencies` / `energies_ev` | eV |
| `vibrations.metadata.frequencies` | cm⁻¹ |
| `wavefunction.gto.basis.shells[].exponents` | bohr⁻² |
| `wavefunction.gto.mo_metadata.energies` | Hartree |
| `wavefunction.gto.basis.ecp[].projectors[]` | `exponents` bohr⁻²; `coefficients` hartree · bohr^(2 − n) for the radial power n, so U(r) is in hartree (§ 5.3) |
| `wavefunction.gto.overlap` | dimensionless; AOs as Appendix A.1 defines them |
| `wavefunction.gto.density_matrix`, `density_matrix_alpha`, `density_matrix_beta` | electrons: tr(P · S) is the electron count (§ 5.3) |
| `wavefunction.bloch.energies` | hartree (§ 5.3.1) |
| `wavefunction.bloch.kpoints.kpoints` | fractional reciprocal-lattice coordinates, k = k₁b₁ + k₂b₂ + k₃b₃ with a_i · b_j = 2π δ_ij (§ 5.3.1) |
| `dos.*.energies` | eV (referenced to E_F at 0.0) |
| `fermi_surface.energies` | eV (signed: E(k) − E_F) |
| `fermi_surface.mesh.lattice_vectors` / `reciprocal_lattice` | Ångström, row vectors / Å⁻¹, row vectors, 2π included |
| `phonon_bands.frequencies`, `phonon_dos.frequencies` | cm⁻¹ |
| `equation_of_state.volumes` / `energies` | Å³ / eV |
| `equation_of_state.fit.V0` / `E0` / `B0` | Å³ / eV / GPa |
| `forces.gradient` | hartree / bohr; the energy gradient ∂E/∂R, so the forces are its negative (§ 5.10) |
| `forces.stress` | hartree / bohr³ |
| `hessian.matrix` | hartree / bohr²; hartree / (bohr² · amu) when `meta.mass_weighted` is true (§ 5.11) |
| `excited_states` `excitation_energy` / `total_energy` | `{value, units}` objects; producers SHOULD write `ev` / `hartree` (§ 5.12) |
| `excited_states` `oscillator_strength` | dimensionless |
| `excited_states` `rotatory_strength_cgs` | 10⁻⁴⁰ esu²·cm² (the "10⁻⁴⁰ cgs" unit), signed |
| `excited_states` `transition_dipole` / `transition_magnetic_dipole` | atomic units: e·bohr / e·ħ/mₑ (= 2 μ_B) |
| root `energies` | `{value, units}` objects; producers SHOULD write `hartree` (§ 3.6) |
| `properties.electric.dipole` | debye (§ 5.13) |
| `properties.electric.quadrupole` / `.octupole` | debye · Å / debye · Å²; `origin`, when a vector, in bohr (§ 5.13) |
| `properties.electric.polarizability` / `.hyperpolarizability` | atomic units, e² a₀² / E_h and e³ a₀³ / E_h² (§ 5.13) |
| `properties.elastic.stiffness`, moduli | GPa; `compliance` in GPa⁻¹; `poisson_ratio` dimensionless (§ 5.13) |
| `properties.dielectric.electronic` / `.ionic` / `.static` | dimensionless (relative permittivity) (§ 5.13) |
| `properties.dielectric.born_charges` / `.piezoelectric` | e / C · m⁻² (§ 5.13) |

When adding fields, prefer unit-bearing names (`energy_eh`, `energy_ev`,
`frequency_cm`) or a `{value, units}` object.

#### Unit names

A unit is named by one of the **unit names** below, a controlled vocabulary.
The JSON Schema carries the same list in the same order as `$defs/UnitName`,
and the two are kept identical. Names are case-sensitive. The list names
units; it changes none: every unit of the table above keeps its meaning.

| Unit name | Quantity | Definition or note |
|---|---|---|
| `angstrom` | length | 1 Å = 10⁻¹⁰ m |
| `bohr` | length | a₀ = 0.529177210903 Å (CODATA 2018) |
| `hartree` | energy | E_h = 27.211386245988 eV (CODATA 2018) |
| `ev` | energy | 1 eV = 1.602176634 × 10⁻¹⁹ J (exact) |
| `kcal_per_mol` | molar energy | thermochemical kcal·mol⁻¹, 4.184 kJ·mol⁻¹ (exact); E_h N_A = 627.5094740631 kcal·mol⁻¹ |
| `kj_per_mol` | molar energy | kJ·mol⁻¹; E_h N_A = 2625.4996394799 kJ·mol⁻¹ (CODATA 2018) |
| `cm-1` | wavenumber | 1 cm⁻¹ = 100 m⁻¹; as an energy E = h c ν̃, with E_h / (h c) = 219474.6313632 cm⁻¹ (CODATA 2018) |
| `nm` | length (wavelength) | 1 nm = 10⁻⁹ m; as a photon energy E = h c / λ, with h c = 1239.841984332 eV·nm (exact) |
| `debye` | electric dipole moment | 1 D = 10⁻²¹ / c C·m ≈ 3.335641 × 10⁻³⁰ C·m |
| `mhz` | frequency | 1 MHz = 10⁶ s⁻¹ |
| `hz` | frequency | 1 Hz = 1 s⁻¹ |
| `ppm` | dimensionless ratio | 10⁻⁶; chemical shifts and shieldings |
| `km_per_mol` | IR intensity | km·mol⁻¹, the integrated molar absorption coefficient |
| `a3` | volume | 1 Å³ = 10⁻³⁰ m³ |
| `bohr3` | volume | a₀³ |
| `gpa` | pressure | 1 GPa = 10⁹ Pa |
| `fs` | time | 1 fs = 10⁻¹⁵ s |
| `ps` | time | 1 ps = 10⁻¹² s |
| `kelvin` | temperature | the SI kelvin |
| `atm` | pressure | standard atmosphere, 101 325 Pa (exact) |
| `e` | electric charge | elementary charge, 1.602176634 × 10⁻¹⁹ C (exact) |
| `amu` | mass | unified atomic mass unit (dalton), 1.66053906660 × 10⁻²⁷ kg (CODATA 2018) |
| `radian` | plane angle | the SI radian |
| `degree` | plane angle | π / 180 rad |
| `dimensionless` | — | a pure number, such as a normalized intensity |

**Legacy spellings.** Archives written before this vocabulary carry three
other spellings in the `units` of a `{value, units}` object (`scf_energy`,
`fermi_energy`, § 3.2). They remain valid there, and only there, and are
**deprecated**: producers **SHOULD** write the unit name, and consumers
**MUST** read a legacy spelling as the unit name it stands for. The schema
lists them as `$defs/LegacyUnitName`.

| Legacy spelling | Unit name |
|---|---|
| `Eh` | `hartree` |
| `eV` | `ev` |
| `Hartree` | `hartree` |

#### Declared units

A payload can say which unit its values are in. The `units` object of the
volume grid (§ 5.2), of the trajectory, reaction-path and scan metadata
(§ 5.5), of the `kpath`, DOS `meta`, Fermi-surface `mesh`, `qpath` and phonon
`meta` payloads (§ 5.6), of the QTAIM payload (§ 5.7) and the `meta.units`
object of the `excited_states` payload (§ 5.12) maps a key to a unit name.
The key is the name of a field of the payload (for `excited_states`, of a
state), or the role of a binary member of the same section, such as `coords`
or `eigenvalues`. A
one-dimensional spectrum declares the units of its axes in `x_units` and
`y_units`, and the unit of its line width in `broadening.units` (§ 5.4). The
`axis` of a `volume.series` declares the unit of its `values` in `units`, a
single unit name (§ 5.2).

* Producers **MAY** declare units. A declaration is not a conversion: the
  values are stored in the declared unit.
* Consumers **MUST** honour a declared unit: they read the values in it, and
  convert from it when they need another.
* Where the table above fixes the unit of a field or member, a declaration
  only restates it, and a declared unit that contradicts it is an **error**:
  a grid `units.origin` other than `bohr`, a trajectory `units.coords`
  other than `angstrom`, or a trajectory `units.lattice` other than
  `angstrom` (a reaction path's is `bohr`). The payload definitions encode
  these fixed units, so
  a validator that runs the JSON Schema check reports the contradiction as
  `E_PAYLOAD_SCHEMA` (§ 6).
* Where a payload leaves the unit open (a spectrum's x axis, QTAIM
  positions, the scan axes), a declaration chooses it, within the quantity
  the field holds: a position takes a length.
* A declared unit is a unit name. So is the `units` of a `{value, units}`
  object, except that it may also be a legacy spelling. A validator reports
  any other string in these places as the warning `W_UNIT_UNKNOWN` (§ 6).
  The schema types these strings as well, so a validator that runs the JSON
  Schema check also reports `E_SCHEMA` or `E_PAYLOAD_SCHEMA` for an unknown
  unit, and the archive does not validate there.

The older unit-bearing keys (`unit` of a scan axis, `reaction_coordinate_unit`,
`energy_unit`, `coordinate_a_unit` and the like) stay free text in v1, because
archives already carry spellings outside the vocabulary, such as `"rad"`.
Producers **SHOULD** use unit names in them as well. When a scan's
`units.energies` and `energy_unit`, or a `units` entry and an axis `unit`,
are both present they **MUST** name the same unit.

#### Lattice conventions

QVF v1 carries the same physical object, a periodic cell, in three members
that follow **two conventions**, which differ in **layout and in unit**.
`structure.lattice_vectors` and `trajectory.lattice` use the *structure
convention* (rows, Å); `reaction.path.lattice` uses the *reaction-path
convention* (columns, bohr). Both conventions are normative and frozen for the
whole of v1; neither will change within it. This subsection is the one place
that states both, and the member descriptions in § 5.1 and § 5.5 refer to it.

| | `structure.lattice_vectors` | `trajectory.lattice` | `reaction.path.lattice` |
|---|---|---|---|
| Convention | structure | structure | reaction path |
| Where | JSON payload of a `structure` section (§ 5.1) | binary member `lattice` of a `trajectory` section (§ 5.5) | binary member `lattice` of a `reaction.path` section (§ 5.5) |
| Shape | `[3, 3]` | `[3, 3]` (fixed cell) or `[n_frames, 3, 3]` (one cell per frame) | `[3, 3]` (fixed cell) or `[n_frames, 3, 3]` (one cell per frame) |
| Layout | **rows**: row `i` is the lattice vector of axis `i`, paired with `pbc[i]` | **rows**: row `i` of each `[3, 3]` matrix is the lattice vector of axis `i`, paired with `metadata.pbc[i]` | **columns**: column `j` of each `[3, 3]` matrix is lattice vector `j` (a, b, c) |
| Unit | Ångström | Ångström | bohr |
| Storage | JSON numbers | float64, little-endian, C order (element `[i][j]` at offset `3i + j`) | float64, little-endian, C order (element `[i][j]` at offset `3i + j`) |
| Periodicity is signalled by | `pbc` | `metadata.pbc`, **REQUIRED** with the member | presence of the member |

A `[3, 3]` slice of `trajectory.lattice` is laid out and scaled exactly as
`structure.lattice_vectors`: the frame-0 cell of a trajectory equals the
`lattice_vectors` of its structure, element for element, with no conversion.
The two conventions differ deliberately: `reaction.path.lattice` predates the
structure-convention carriers and stays columns in bohr for the whole of v1,
and `trajectory.lattice`, added later, adopts the structure convention so that
new carriers converge on one layout.

Write `R` for a matrix in the structure convention (`structure.lattice_vectors`
or one `[3, 3]` slice of `trajectory.lattice`; Å, rows) and `L` for one
`[3, 3]` slice of `reaction.path.lattice` (bohr, columns), and
`f = 1.8897261246` bohr per Å (the reciprocal of 1 bohr = 0.529177210903 Å,
CODATA 2018, rounded to ten decimals). Then

```text
L[i][j] = f · R[j][i]        i.e.  L = f · Rᵀ     (structure → reaction.path)
R[i][j] = L[j][i] / f        i.e.  R = Lᵀ / f     (reaction.path → structure)
```

so the columns of `L` are the rows of `R` scaled by `f`. For a per-frame
`[n_frames, 3, 3]` member, apply the conversion to each frame's slice.

**Worked example.** A hexagonal cell with a = 2.46 Å and a 15.875 Å normal (the
2-D graphene slab of the conformance corpus). As `structure.lattice_vectors`
(Å, rows):

```jsonc
[[2.46, 0.0,    0.0   ],     // a
 [1.23, 2.1304, 0.0   ],     // b
 [0.0,  0.0,    15.875]]     // c
```

and as the equivalent `reaction.path.lattice` (bohr, columns, 4 decimals):

```text
[[ 4.6487,  2.3244,  0.0000 ],      column 0 = a = (4.6487, 0.0000,  0.0000)
 [ 0.0000,  4.0259,  0.0000 ],      column 1 = b = (2.3244, 4.0259,  0.0000)
 [ 0.0000,  0.0000, 29.9994 ]]      column 2 = c = (0.0000, 0.0000, 29.9994)
```

The second matrix is the transpose of the first times `f`: row `a` of the Å
matrix, `(2.46, 0, 0)`, becomes column 0 of the bohr matrix, `(4.6487, 0, 0)ᵀ`,
and row `b`, `(1.23, 2.1304, 0)`, becomes column 1, `(2.3244, 4.0259, 0)ᵀ`.
Reading the bohr matrix by rows would give a = (4.6487, 2.3244, 0) and
b = (0, 4.0259, 0), which is a different cell. For an axis-aligned cubic or
orthorhombic cell the matrix is diagonal and the transpose is invisible,
leaving only the unit factor, so a diagonal test case does not show that the
two conventions agree.

A consumer **MUST NOT** assume the convention of one member for another: it
**MUST** read `structure.lattice_vectors` and `trajectory.lattice` as rows in Å
and `reaction.path.lattice` as columns in bohr, and convert explicitly when it
combines a reaction path with either of the others (for example to overlay a
path on the cell of its structure). A producer that writes several **MUST**
write each in its own convention. A validator reads each member in its own
convention and never mixes them: it checks the shape of each lattice against
the frames of its own section (§ 6), and never compares one carrier with
another.

*Note (informative).* New carriers of a cell adopt the structure convention.
Unifying `reaction.path.lattice` with it is a v2 ledger item (V2-05).

---

## 5. Canonical section kinds

Every kind below is a `oneOf` branch in the schema. A section's `kind` selects
its required member set. The table lists each kind and its required members;
subsections give the payload shapes that matter for interoperability.

| Kind | Required members (roles) |
|---|---|
| `structure` | `structure` (JSON, `$defs/StructurePayload`); optional `bonds` (JSON); optional section fields for biomolecules and for the chemical identity of the species (§ 5.1) |
| `bonds` | `bonds` (JSON) |
| `bond_orders` | `bond_orders` (JSON) |
| `volume.density` / `.orbital` / `.spin` / `.elf` / `.generic` / `.potential` / `.rdg` / `.laplacian` | `grid` (JSON, `$defs/VolumeGridPayload`) + `data` (3-D float32/64 binary) |
| `volume.vector` | `grid` (JSON, `$defs/VolumeGridPayload`) + `data` (float32/64 [nx, ny, nz, 3], `$defs/Volume4DVectorBinary`) |
| `volume.complex` | `grid` (JSON, `$defs/VolumeGridPayload`) + `data` (float32/64 [nx, ny, nz, 2], last axis [re, im], `$defs/Volume4DComplexBinary`) |
| `volume.series` | `grid` (JSON, `$defs/VolumeGridPayload`) + `data` (float32/64 [n, nx, ny, nz]) + `axis` (JSON, `$defs/VolumeSeriesAxisPayload`) |
| `volume.difference` | `grid` (JSON, `$defs/VolumeGridPayload`) + `data`; optional section fields `operand_a`, `operand_b`, `description` |
| `volume.transition_density` | `grid` (JSON, `$defs/VolumeGridPayload`) + `data`; optional section fields `excited_states_ref`, `state_index` |
| `wavefunction.gto` | `basis` (JSON, `$defs/BasisPayload`) + `mo_metadata` (JSON, `$defs/MoMetadataPayload`) + `mo_coefficients` **or** (`mo_coefficients_alpha` + `mo_coefficients_beta`) (float64 binary); optional `overlap` and `density_matrix` **or** (`density_matrix_alpha` + `density_matrix_beta`) (float64 [n_ao, n_ao], `$defs/WavefunctionAoMatrix`) |
| `wavefunction.bloch` (provisional) | `basis` (JSON, `$defs/BasisPayload`) + `kpoints` (JSON, `$defs/BlochKpointsPayload`) + `mo_metadata` (JSON, `$defs/BlochMoMetadataPayload`) + `mo_coefficients` (float64 [n_k, n_spin, n_mo, n_ao, 2]) + `energies` + `occupations` (float64 [n_k, n_spin, n_mo]) |
| `basis.ao` | `grid` (JSON, `$defs/VolumeGridPayload`) + `data`; section field `ao_metadata` |
| `atom_properties` | ≥1 of the scalar roles `mulliken_charge`, `loewdin_charge`, `hirshfeld_charge`, `iao_charge`, `spin_population`, `cm5_charge`, `chelpg_charge`, `resp_charge`, `esp_charge`, `nbo_charge`, `mbis_charge`, `bader_charge`, `ddec_charge`, `valence`, `bonded_valence`, `fukui_plus`, `fukui_minus`, `fukui_zero`, `dual_descriptor`, `atomic_energy_eh`, `atomic_volume_bohr3`, `nmr_shielding_iso_ppm` (float64 [n_atoms] binary, `$defs/AtomChargeBinary`) and the vector roles `magnetic_moment`, `force_eh_bohr` (float64 [n_atoms, 3] binary, `$defs/AtomVectorBinary`); optional `custom.<name>` roles (either shape), each described by the `meta` member (JSON, `$defs/AtomPropertiesMetaPayload`) |
| `trajectory` | `metadata` (JSON, `$defs/TrajectoryMetadataPayload`) + `coords` ([n_frames, n_atoms, 3]; float64, or float32 with `metadata.precision`); optional `lattice` (float64 [3, 3] or [n_frames, 3, 3], rows, Å), `velocities`, `gradient` (float64 [n_frames, n_atoms, 3]), `times`, `energies`, `cell_volumes`, `temperatures` (float64 [n_frames]) (§ 5.5) |
| `reaction.path` | `metadata` (JSON, `$defs/ReactionPathMetadataPayload`) + `coords`; optional `lattice` (periodic marker; columns, bohr), `velocities`, `gradient`, `times`, `energies` (as for `trajectory`), `frame_volumes` (4-D) + `volume_grid` (JSON, `$defs/VolumeGridPayload`) |
| `reaction.waypoints` | `waypoints` (JSON, `$defs/ReactionWaypointsPayload`); section field `trajectory_ref` (REQUIRED) |
| `reaction.scheme` | `scheme` (JSON, `$defs/ReactionSchemePayload`) (§ 5.14) |
| `scan.surface` | `metadata` (JSON, `$defs/ScanSurfaceMetadataPayload`) + `axis_a` + `axis_b` + `energies`; optional `geometries` |
| `vibrations` | `metadata` (JSON, `$defs/VibrationsMetadataPayload`) + `displacements` (float64 [n_modes, n_atoms, 3]); optional section field `hessian_ref` |
| `spectra.ir` / `.raman` / `.uvvis` / `.ecd` / `.vcd` / `.nmr` / `.epr` / `.generic` | `spectrum` (JSON) |
| `bands` | `kpath` (JSON, `$defs/KpathPayload`) + `eigenvalues` (float64 [n_spin, n_k, n_bands]); optional `projections`, `occupations` |
| `dos.total` | `energies` + `dos` (binary); optional section fields (§ 5.6) |
| `dos.projected` | `energies` + `projections` (binary); optional section fields, including `channels` (§ 5.6) |
| `dos.coop` / `dos.cohp` | `energies` + `projections` + `integrated` + `meta` (JSON, `$defs/DosMetaPayload`) |
| `structure.symmetry` | `data` (JSON) |
| `scf_history` | `iterations` (JSON) |
| `citations` | `references` (binary BibTeX) |
| `run.record` | ≥1 of `input`, `log` (opaque UTF-8 binary); optional `files` (JSON, `$defs/RunRecordFilesPayload`) + `attachment.*` (opaque binary); section field `program` (REQUIRED); optional section fields `job_ref`, `run_status` (§ 5.8) |
| `job.spec` | `spec` (JSON, `$defs/JobSpecPayload`) |
| `fermi_surface` | `mesh` (JSON, `$defs/FermiMeshPayload`) + `energies` (4-D float32/64) |
| `phonon_bands` | `qpath` (JSON, `$defs/QpathPayload`) + `frequencies` (binary); optional `eigenvectors` (float64 [n_q, n_modes, n_atoms, 3] or [n_q, n_modes, n_atoms, 3, 2]) |
| `phonon_dos` | `meta` (JSON, `$defs/PhononMetaPayload`) + `frequencies` + `dos` (binary); optional `projected` (float64 [n_channels, n_points]) |
| `equation_of_state` | `volumes` + `energies` (binary) + `fit` (JSON, `$defs/EosFitPayload`) |
| `topology.qtaim` | `critical_points` (JSON) |
| `forces` | `gradient` (float64 [n_atoms, 3]); optional `stress` (float64 [3, 3]) + `meta` (JSON, `$defs/ForcesMetaPayload`) |
| `hessian` | `matrix` (float64 [3N, 3N]); optional `meta` (JSON, `$defs/HessianMetaPayload`) |
| `excited_states` | `states` (JSON, `$defs/ExcitedStatesPayload`); optional section field `wavefunction_ref` |
| `properties.electric` | `properties` (JSON, `$defs/ElectricPropertiesPayload`) |
| `properties.elastic` | `properties` (JSON, `$defs/ElasticPropertiesPayload`) |
| `properties.dielectric` | `properties` (JSON, `$defs/DielectricPropertiesPayload`) |
| `preview` | `thumbnail` (opaque binary, `image/png` or `image/jpeg`); optional `image` (opaque binary, `image/svg+xml`, `image/png` or `image/jpeg`); both with `media_type` (§ 5.15) |
| `x_<vendor>.*` | any mix of valid JSON/binary members |

**Payload definitions.** A JSON member whose section kind and role have a
payload definition, the `$defs` entry of the JSON Schema named in the table
above and in the subsections below, **MUST** conform to it, and a validator
checks that it does (§ 6). A payload definition describes an archive member,
not manifest content, so nothing in the manifest refers to it. Each payload
table below names the strictness level of its definition:

* **strict** — the payload carries only the keys its table lists. Any other
  key is an error, except a key that begins with `x_`: a producer may use such
  keys for its own additions, and a consumer that does not understand one
  ignores it.
* **typed-open** — the keys its table lists are typed, and any other key is
  accepted; a consumer that does not understand one ignores it.

### 5.1 `structure` and `bonds`

`structure.json` — schema'd as `$defs/StructurePayload`:

```jsonc
{
  "atoms": [
    {"symbol": "O", "position": [0.0, 0.0, 0.1173], "atomic_number": 8},
    {"symbol": "H", "position": [0.0, 0.7572, -0.4692], "atomic_number": 1}
  ],
  "pbc": [false, false, false],
  "lattice_vectors": null
}
```

Each atom carries `symbol`, `position` (Cartesian Å, never fractional), and
`atomic_number`; producers may add further per-atom fields.

#### Periodicity

`pbc` is an array of three booleans and is the **normative** carrier of
periodicity: axis `i` is periodic **iff** `pbc[i]`. It is **REQUIRED** whenever
`lattice_vectors` is a non-null array. When absent the structure is a molecule,
exactly as if `[false, false, false]` had been written.

All eight patterns are legal. The periodic axes need **NOT** be the leading
ones: a slab whose surface normal lies along *y* is `[true, false, true]`, and a
producer **MUST NOT** be required to permute its lattice to satisfy a consumer.

`lattice_vectors` is a `[3,3]` array of **row** vectors in Å — row `i` is the
lattice vector of axis `i`, paired with `pbc[i]`. It is **REQUIRED** (non-null)
when any `pbc[i]` is true, and **SHOULD** be `null` or absent otherwise.
`trajectory.lattice` (§ 5.5) uses the same layout and unit; the other lattice
carrier in v1, `reaction.path.lattice`, uses a different layout and unit; see
*Lattice conventions* in § 4.3.

For every axis `i` where `pbc[i]` is false, **row `i` is non-physical
bookkeeping.** Producers keep the matrix full-rank `3×3` so that integral codes
and symmetry finders always see a 3-D lattice, but such a row spans no
periodicity and encodes no geometry. A consumer **MUST NOT** draw it as a cell
edge, replicate the structure along it, or read it as a vacuum extent. This is
not a hypothetical: a synthesized 30-bohr normal and a real 30-bohr vacuum gap
are numerically identical in `lattice_vectors`, so **`pbc` is the only signal
that tells them apart.** Any heuristic of the form *"is this row long and
orthogonal? then it is vacuum"* silently renders 2-D slabs inside empty boxes.

`dimensionality` (0–3) is **OPTIONAL** and **derived**, with the invariant

```text
dimensionality == the number of true entries in pbc
```

A producer that emits it **MUST** satisfy that invariant; a validator **MUST**
reject a payload that violates it. It is a convenience scalar and carries no
axis identity: a consumer **MUST** take per-axis periodicity from `pbc`, and
**MUST NOT** infer *which* axes are periodic from `dimensionality`.

`dimensionality` lives in the **payload**, alongside the `pbc` it is derived
from — not on the section object. An earlier revision of this section placed it
on the section object; a consumer **MUST NOT** read it from there. Archives that
carry it in the old position remain fully interpretable, because `pbc` is what a
consumer reads.

> **Note (informative).** Some producers — vibe-qc among them — order their
> lattice so the periodic axes come first, emitting `pbc` as `[true] * dim +
> [false] * (3 - dim)`. A consumer reading a pre-conformance archive that has
> `dimensionality` but no `pbc` may assume that ordering as a fallback. This is
> a legacy accommodation, not part of the contract, and it is unnecessary for
> any archive conforming to this section, where `pbc` is always present when it
> matters.

#### Connectivity

Explicit connectivity is a separate `bonds` section
(`{"pairs": [{"i": 0, "j": 1, "order": 1.0}, …]}`), or the optional `bonds`
member of the `structure` section itself. If neither is present, a viewer may
infer bonds from covalent radii. The payload, of a `bonds` section and of a
`structure` section's optional `bonds` member alike, is `$defs/BondsPayload`
(§ 5.7).

The atom indices of a `bonds` member of a `structure` section address that
structure's atoms. A standalone `bonds` section names its structure with the
generic section field `structure_ref` (§ 4.1):

* **Present:** the bonds belong to the `structure` section it names, and
  their indices address its atoms. A validator reports a `structure_ref` that
  does not name a `structure` section (`E_REF_UNRESOLVED`), and an atom index
  `i` or `j` that is not below the atom count of that structure
  (`E_MEMBER_MALFORMED`, § 6).
* **Absent:** the bonds belong to the archive's first structure (§ 4.1), in
  an archive with one structure that structure, as before this field was
  specified.

A producer that writes several `structure` sections **SHOULD** set
`structure_ref` on every standalone `bonds` section, or, simpler, write each
structure's bonds as that section's own `bonds` member.

```json
{"id": "bonds_product", "kind": "bonds", "structure_ref": "structure_product",
 "members": {"bonds": {"path": "bonds_product/connectivity.json", "format": "json",
                       "sha256": "…"}}}
```

#### Biomolecule metadata

The `structure` **section object** (peer to `id`/`kind`/`members`, *not* inside
the payload) **MAY** carry optional biomolecule metadata for cartoon/ribbon
rendering. All four fields are **OPTIONAL** and independent; a `structure`
carrying none of them is a valid plain structure, and a consumer that does not
understand them **MUST** ignore them.

| Field | Type | Schema |
|---|---|---|
| `chains` | array of chain-id strings, in presentation order | — |
| `residues` | array of `{name, seq, chain, atom_indices}` | `$defs/BiomoleculeResidue` |
| `secondary_structure` | array of `{type, chain, start_seq, end_seq}`, `type` in `helix`/`sheet`/`coil`, ranges inclusive | `$defs/BiomoleculeSecondaryStructure` |
| `b_factors` | array of number, one per atom, parallel to the payload's `atoms` order | — |

`atom_indices` are **0-based** into the `structure` member payload's `atoms`
array. `residues` are listed in **chain order** — the order a ribbon is drawn
in — not sorted by `seq`: insertion codes and non-monotonic numbering are both
legal, and sorting would silently reorder them. `seq` is **not** a global key:
PDB residue numbers are four columns wide and wrap at 9999, so a solvated
system reuses them.

**Precedence.** Many consumers can derive residues, chains, and secondary
structure themselves — from the per-atom fields below, and from CA geometry.
Where a producer supplies these fields, a consumer **MUST** prefer the supplied
values over its own inference. A producer knows what geometry cannot reveal: an
α- from a π- from a 3-10 helix, a β-bridge from a sheet.

A payload's atoms **MAY** additionally carry the per-atom biomolecular fields
`atom_name` (PDB cols 13-16, e.g. `"CA"`), `residue_name` (cols 18-20),
`residue_seq` (cols 23-26), `chain_id` (col 22) and `b_factor` (cols 61-66).
Where both carriers are present, the per-atom `b_factor` **wins** over the
section-level `b_factors` array: it cannot desynchronize from its own atom. A
consumer **MUST** ignore a `b_factors` array whose length differs from the atom
count rather than applying a partial mapping.

These fields were added under **v1**: an optional member is the additive case
and does not bump `qvf_version` (see `GOVERNANCE.md`).

#### Chemical identity

The `structure` **section object** **MAY** also say which chemical species the
structure is: its net charge and spin multiplicity, a name, and line notations.
Like the biomolecule metadata they are peers of `id` and `members`, so a
consumer reads them from the manifest without opening a member, and a program
that lists the molecules of an archive, or searches archives for one, needs
nothing else. All are **OPTIONAL** and independent.

| Field | Type | Meaning |
|---|---|---|
| `charge` | integer | Net charge of the species, in units of the elementary charge. |
| `multiplicity` | integer ≥ 1 | Spin multiplicity 2S+1 of the species. |
| `name` | string, non-empty | Name of the species, e.g. `"ethanol"` or `"Grubbs II"`. |
| `smiles` | string, no whitespace | SMILES of the species. |
| `inchi` | string beginning `InChI=` | InChI of the species, standard (`InChI=1S/…`) or not. |
| `inchikey` | string, `AAAAAAAAAAAAAA-BBBBBBBBBB-C` | InChIKey of the species: 14 and 10 uppercase letters and one more, joined by hyphens. |

The generic section field `label` (§ 4.1) is the short label a scheme or a
viewer shows for the species, such as its compound number (`"1"`, `"3a"`);
`name` is its chemical name.

The fields describe the species whose atoms the `structure` payload holds:
all of them, hydrogens included, with the stated charge. The line notations
name the species as a whole and imply no atom order, so a consumer
**MUST NOT** map a SMILES atom to a payload atom by position. A producer
writes only identifiers that it derived from these atoms or knows to describe
them; when it writes several they name the same species. The validators
cannot compare a SMILES or an InChI with the atoms and do not try.

A field that is absent states nothing: in particular a structure without
`charge` is not declared neutral. A consumer that needs the charge or the
multiplicity of such a structure takes it from `provenance` (§ 3.2) when the
archive has exactly one `structure` section, and otherwise from its own
default. The charge and multiplicity of a *calculation* are those of its
`job.spec` (§ 5.9), which is authoritative for the job and may differ on
purpose, for a vertical ionization for example.

Two rules relate the fields to the rest of the archive, and a validator checks
both (§ 6):

* **Electron count.** When `charge` is present the species has
  N = Σ Z − `charge` electrons, Σ Z being the sum of the payload's
  `atomic_number` values: N **MUST NOT** be negative, and when `multiplicity`
  is present too, the unpaired electrons, `multiplicity` − 1, **MUST NOT**
  exceed N and **MUST** have the parity of N. A violation is
  `E_CHARGE_MULTIPLICITY`.
* **Agreement with `provenance`.** In an archive with exactly one `structure`
  section, its `charge` and `multiplicity` **MUST** equal `provenance.charge`
  and `provenance.multiplicity` where both are present: there they state the
  same thing twice. A disagreement is `E_ROOT_INCONSISTENT`.

A methane molecule labelled as compound 2 of a scheme:

```json
{
  "id": "structure_methane", "kind": "structure", "label": "2",
  "name": "methane", "charge": 0, "multiplicity": 1,
  "smiles": "C", "inchi": "InChI=1S/CH4/h1H4",
  "inchikey": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
  "members": {"structure": {"path": "structure_methane/structure.json",
                            "format": "json", "sha256": "…"}}
}
```

These fields were added under **v1**, like the biomolecule metadata. An
archive that holds several molecules, such as the species of a reaction
scheme (§ 5.14), gives each its own `structure` section and identity, and
every reader then knows which molecule is which and what its charge and spin
are.

### 5.2 `volume.*`

A volume section samples a field on a regular grid. Every volume kind has a
JSON `grid` member (the payload below) and a binary `data` member, float32 by
default and float64 by opt-in. The kind fixes what `data` holds and its shape;
the grid always describes the three **spatial** axes of `data`.

| Kind | `data` shape | Holds | Unit of `data` |
|---|---|---|---|
| `volume.density` | `[nx, ny, nz]` | electron density | e / bohr³ |
| `volume.orbital` | `[nx, ny, nz]` | orbital amplitude | amplitude on a bohr grid |
| `volume.spin` | `[nx, ny, nz]` | spin density | — |
| `volume.elf` | `[nx, ny, nz]` | electron localization function | — |
| `volume.difference` | `[nx, ny, nz]` | difference of two fields (below) | that of the operands |
| `volume.potential` | `[nx, ny, nz]` | electrostatic potential | hartree / e |
| `volume.rdg` | `[nx, ny, nz]` | reduced density gradient | dimensionless |
| `volume.generic` | `[nx, ny, nz]` | any other scalar field, named by the section `label` | — |
| `volume.laplacian` | `[nx, ny, nz]` | Laplacian of the electron density, ∇²ρ | e / bohr⁵ |
| `volume.vector` | `[nx, ny, nz, 3]` | a Cartesian vector field (below) | — |
| `volume.complex` | `[nx, ny, nz, 2]` | a complex scalar field (below) | — |
| `volume.series` | `[n, nx, ny, nz]` | `n` scalar fields on one grid, with an `axis` member (below) | — |

The units are those of § 4.3. A dash means that § 4.3 fixes none: the unit is
that of the field, and a producer declares it as `grid.units.data` when the
vocabulary of § 4.3 names it.

The grid `shape` is the `[nx, ny, nz]` of this table: the shape of a 3-D
`data` member, the leading three extents of a `volume.vector` or
`volume.complex` member, and the trailing three of a `volume.series` member.

#### The grid payload

```jsonc
// the CHGCAR-style grid of the conformance archive volume_periodic.qvf
// (a hexagonal cell; voxel vectors shown to 4 decimals)
{
  "origin": [0.0, 0.0, 0.0],
  "voxel_vectors": [[0.7748, 0.0, 0.0], [-0.3874, 0.6710, 0.0], [0.0, 0.0, 0.7913]],
  "shape": [6, 6, 8],
  "periodic": [true, true, true],
  "includes_endpoint": false,
  "structure_ref": "structure"
}
```

The grid payload is schema'd as `$defs/VolumeGridPayload` (strict). The same
definition governs the `grid` member of `basis.ao` (§ 5.7) and the
`volume_grid` member of `reaction.path` (§ 5.5).

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `origin` | 3 numbers | **yes** | Cartesian position of grid point `(0, 0, 0)`. | bohr |
| `voxel_vectors` | 3 × 3 numbers | **yes** | The step vectors `v_0`, `v_1`, `v_2`: row `a` is the displacement between neighbouring points along data axis `a`. Non-orthogonal grids use non-zero off-diagonal components. | bohr per step |
| `shape` | 3 integers ≥ 1 | **yes** | Number of points `[nx, ny, nz]` along the three spatial data axes; **MUST** equal the spatial extents of the `data` member (table above; for `volume_grid`, the trailing three dimensions of `frame_volumes`). | — |
| `units` | object | no | Declared units (§ 4.3, *Declared units*), keyed by a field of this payload or a member role of the section; `data` declares the unit of the values. `origin` and `voxel_vectors` are `bohr` when declared. | — |
| `periodic` | 3 booleans | no | `periodic[a]` true: the field repeats with lattice vector `a` of the grid's structure along data axis `a`. Absent means `[false, false, false]`, a finite box. | — |
| `includes_endpoint` | boolean | no | Whether the points along a periodic axis include the endpoint of the period (below). Absent means `true`. It has no effect on an axis that is not periodic. | — |
| `structure_ref` | string | no | `id` of the `structure` section whose cell the grid samples. The section-level `structure_ref` (§ 4.1) takes precedence when both are present. | — |

**Point positions.** For data index `(i, j, k)`, with `0 ≤ i < shape[0]`,
`0 ≤ j < shape[1]` and `0 ≤ k < shape[2]`, the point position is

```text
r(i, j, k) = origin + i·v_0 + j·v_1 + k·v_2
```

This is **point-centred** data: the grid has exactly `shape` points, at these
positions, whatever the other fields say. For the 4-D kinds the extra axis is
not a spatial axis: the point of `data[i][j][k][c]` (vector, complex) and of
`data[s][i][j][k]` (series) is `r(i, j, k)`.

**Periodic grids.** The grid's structure is the `structure` section the
section-level `structure_ref` names, else the one the grid's `structure_ref`
names; when neither is given, it is the archive's `structure` section if the
archive has exactly one. Along an axis `a` with `periodic[a]` true, the field
is periodic with the lattice vector `L_a` of that structure, which is row `a`
of its `lattice_vectors` converted to bohr (`L_a = f · R[a]`, § 4.3, *Lattice
conventions*), and the grid samples exactly one period. `includes_endpoint`
says how:

* `includes_endpoint: true` (the default; the convention of cube files): the
  points span the closed period. The last plane is the image of the first,
  so `shape[a] − 1` steps span one period:
  `v_a · (shape[a] − 1) = L_a`, and plane `shape[a] − 1` repeats plane `0`.
  The grid holds `shape[a] − 1` distinct planes per period.
* `includes_endpoint: false` (the convention of VASP CHGCAR files): the points
  span the period exclusive of its endpoint, so `shape[a]` steps span one
  period: `v_a · shape[a] = L_a`. Index `shape[a]` would be plane `0` of the
  next cell; every plane is distinct.

A consumer that tiles a periodic grid along axis `a` (replicates it to draw a
supercell), or that wraps an index or interpolates across the cell boundary,
**MUST** drop the duplicated plane when `includes_endpoint` is true: the period
in index space is `shape[a] − 1`, and plane `shape[a] − 1` is not repeated
before plane `0` of the next cell. When `includes_endpoint` is false the period
is `shape[a]` and no plane is dropped. A producer **MUST** write the
`includes_endpoint` value that matches its points, and **SHOULD** write
`structure_ref` when the archive has more than one structure.

*Worked example.* One axis of a cell with `L = 4.0` bohr, sampled by four
points in each convention:

```text
includes_endpoint: false  (CHGCAR style)   v = L / 4 = 1.0 bohr
  index         0      1      2      3   | 4 = index 0 of the next cell
  position     0.0    1.0    2.0    3.0  | 4.0
  distinct planes per period: 4
  tiled:       f0 f1 f2 f3 | f0 f1 f2 f3 | f0 …

includes_endpoint: true   (cube style)     v = L / 3 = 1.3333 bohr
  index         0       1       2       3
  position     0.0   1.3333  2.6667   4.0     index 3 is index 0 shifted by L
  distinct planes per period: 3
  tiled:       f0 f1 f2 | f0 f1 f2 | f0 …     plane 3 dropped
```

A consumer that tiles the cube-style grid without dropping plane 3 draws
`f0 f1 f2 f3 f0 …` with `f3 = f0`: a doubled plane at every cell boundary,
and a period of four steps where the cell has three, so the field drifts by
one step per cell. A consumer that drops the last plane of the CHGCAR-style
grid loses a quarter of the period. Producers of both conventions exist, and
the numbers alone do not say which one an archive holds; `includes_endpoint`
does. The conformance archive `volume_periodic.qvf` holds one periodic
density in both conventions.

A validator checks, for every grid payload whose `periodic[a]` is true and
whose structure resolves, that the structure's `pbc[a]` is true and that
`v_a · (shape[a] − 1)` (when `includes_endpoint` is true) or `v_a · shape[a]`
(when it is false) equals `L_a` within a relative tolerance of 10⁻⁶,
`|span − L_a| ≤ 10⁻⁶ |L_a|`, comparing the vectors component by component
rather than their lengths (§ 6). An axis that `periodic` does not mark is not
checked, so a grid may sample a box that is finite along an axis the structure
is periodic in, such as the vacuum direction of a slab. A grid `structure_ref`
that names no `structure` section is an unresolved reference.

`volume.difference` may add `operand_a` and `operand_b` (section ids); if one is
present both **MUST** be. Sign convention: `data = operand_a − operand_b`.

`volume.transition_density` holds the transition density of one electronic
excitation, ρ₀ₙ(**r**): the one-particle transition density between the
reference state 0 and the excited state n, summed over spin, in e / bohr³.
It integrates to zero, and its first moment ∫ **r** ρ₀ₙ(**r**) d³r is the
transition dipole up to its sign. Its overall sign is arbitrary, as is the
phase of the excited state. The density of the excited state minus that of
the reference state is a difference density and is written as
`volume.difference`. Two optional section fields (peers of `members`) name the
state:

| Field | Type | Meaning |
|---|---|---|
| `excited_states_ref` | string | `id` of the `excited_states` section (§ 5.12) the state belongs to. |
| `state_index` | integer ≥ 1 | The `index` of the state in that section. Requires `excited_states_ref`. |

A validator checks that `excited_states_ref` names an `excited_states` section
and that `state_index` is the `index` of one of its states (§ 6).

#### `volume.laplacian`

`data[i][j][k]` is ∇²ρ at `r(i, j, k)` in e / bohr⁵: negative where the
density concentrates charge and positive where it depletes it, the map
QTAIM reads valence-shell charge concentrations from. The vocabulary of § 4.3
has no unit name for e / bohr⁵, so a producer does not declare
`grid.units.data` for it (§ 9).

#### `volume.vector` and `volume.complex`

A `volume.vector` section holds a Cartesian vector at every grid point, such
as the gradient of a density, an electric field or a current density:
`data[i][j][k][c]` is component `c` (`0` = x, `1` = y, `2` = z) of the vector
at `r(i, j, k)`, in the Cartesian frame of `origin`. The member is
`$defs/Volume4DVectorBinary`, whose trailing extent is 3. Its unit is that of
the field; a producer declares it as `grid.units.data` when the vocabulary
of § 4.3 names it. The section field `component` is not used.

A `volume.complex` section holds a complex scalar field, such as a complex
molecular orbital or a Bloch function at one k-point:
`data[i][j][k][0]` is the real part and `data[i][j][k][1]` the imaginary part
at `r(i, j, k)`, the `[re, im]` layout of complex phonon eigenvectors (§ 5.6).
The member is `$defs/Volume4DComplexBinary`, whose trailing extent is 2.

#### `volume.series`

A `volume.series` section holds `n` scalar fields on one grid, such as the
density along a trajectory, a time-dependent density or the densities of
several electronic states: `data[s]`, the slab `s`, is the field at
the axis value `axis.values[s]`. The members are `grid`, `data`
(`$defs/Volume4DBinary`, `[n, nx, ny, nz]`) and `axis`, a JSON member that says
what indexes the slabs.

Payload `axis` (`$defs/VolumeSeriesAxisPayload`, strict):

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `kind` | `"time"`, `"frame"`, `"state"` or `"custom"` | **yes** | What indexes the slabs: a time; a frame of the trajectory or reaction path that `trajectory_ref` names; an electronic state; anything else, which `labels` describe. | — |
| `values` | array of numbers | **yes** | The axis value of each slab, in slab order: `n` entries. For `kind: "frame"` they are 0-based frame indices into the `coords` of the section `trajectory_ref` names, which need not be consecutive (a series may sample every other frame), and the schema requires non-negative integers. | `units` |
| `units` | unit name | no | Unit of `values` (§ 4.3). Absent means the values are pure numbers (`dimensionless`), as frame and state indices are; a `time` axis **SHOULD** declare its unit (`fs` or `ps`). | — |
| `labels` | array of strings | no | Display label of each slab, `n` entries, e.g. `"S1"`. | — |
| `trajectory_ref` | string | no | `id` of the `trajectory` or `reaction.path` section whose frames the slabs belong to. | — |

A validator checks that `data` has rank 4, that `values` and, when present,
`labels` have one entry per slab (`n`, the leading extent of `data`), and
that `trajectory_ref` names a `trajectory` or `reaction.path` section (§ 6).

`volume.series` is the general form of the per-frame volumes that a
`reaction.path` carries as `frame_volumes` and `volume_grid` (§ 5.5); a
producer **SHOULD** write new per-frame volumes as a `volume.series`.

### 5.3 `wavefunction.gto`

Carries an atom-centered Gaussian basis and MO coefficient matrices for
molecular wavefunctions and the **Γ-point (k = 0)** crystalline orbitals of
periodic systems. Crystalline orbitals at other k-points, or at several, are
carried by `wavefunction.bloch` (§ 5.3.1), which shares the `basis` payload
defined here. See **Appendix A** for the AO-ordering and
primitive-normalization contract — this is the single most error-prone part of
implementing a producer, and getting it wrong silently corrupts every rendered
orbital. A producer that also writes the AO `overlap` member makes the contract
checkable: validators compare its diagonal with Appendix A.1.

```jsonc
{
  "kind": "wavefunction.gto",
  "members": {
    "basis":       {"path": "wavefunction/basis.json", "format": "json", "sha256": "…"},
    "mo_metadata": {"path": "wavefunction/mo_metadata.json", "format": "json", "sha256": "…"},
    "mo_coefficients": {
      "path": "wavefunction/mo_coefficients.dat", "format": "binary",
      "dtype": "float64", "shape": [41, 41], "sha256": "…"
    }
  }
}
```

`mo_coefficients` shape is `[n_mo, n_ao]` (rows are MOs). Unrestricted
wavefunctions replace it with `mo_coefficients_alpha` + `mo_coefficients_beta`.
Complex coefficients add a trailing axis of length 2, and a two-component
wavefunction doubles the AO axis (*Spin and complex coefficients* below).

| Role | Format | Required | Meaning |
|---|---|:--:|---|
| `basis` | JSON, `$defs/BasisPayload` | **yes** | The basis set (below). |
| `mo_metadata` | JSON, `$defs/MoMetadataPayload` | **yes** | Orbital energies, occupations and labels (below). |
| `mo_coefficients` | float64 `[n_mo, n_ao]`, or `[n_mo, n_ao, 2]` when complex | **yes**, single set | The MO coefficients, one row per MO, columns in the AO order of `basis`. |
| `mo_coefficients_alpha`, `mo_coefficients_beta` | as `mo_coefficients` | **yes**, `unrestricted` | The α and the β coefficients, replacing `mo_coefficients`. |
| `overlap` | float64 `[n_ao, n_ao]` | no, **SHOULD** | The AO overlap matrix (*AO overlap and density matrices* below). |
| `density_matrix` | float64 `[n_ao, n_ao]` | no; only with `mo_coefficients` | The spin-summed one-particle density matrix. |
| `density_matrix_alpha`, `density_matrix_beta` | float64 `[n_ao, n_ao]` | no; both or neither, only with the α and β coefficients | The α and the β one-particle density matrices. |

`n_ao` is the number of AOs of the `basis` payload: the sum over its shells of
`2l+1` (spherical) or `(l+1)(l+2)/2` (Cartesian).

**`basis` payload** — schema'd as `$defs/BasisPayload` (strict). The AO order
is the concatenation of the shells in array order, each in the in-shell order
of Appendix A.2.

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `structure_ref` | string | **yes** | Id of the `structure` section whose atoms the shell `center` indices refer to. A `structure_ref` on the section object takes precedence over this one (§ 4.1). | — |
| `shells` | array of shells (below) | **yes** | The contracted shells, in AO order. | — |
| `pure` | boolean | no | Angular type of every shell without its own `pure`: `true` = spherical harmonics, `false` = Cartesian. Absent means `true`. | — |
| `n_ao` | integer ≥ 0 | no | Number of AOs: the sum over the shells of `2l+1` (spherical) or `(l+1)(l+2)/2` (Cartesian); equals the AO dimension of the coefficient members (half of it for `generalized`). | — |
| `ecp` | array of ECP entries (below) | no | Effective core potentials, at most one per atom. | — |
| `n_electrons` | integer ≥ 0 | no | Number of electrons the wavefunction describes: `Σ Z − Σ n_core_electrons − charge`, the sum of the atomic numbers of the structure, minus the core electrons the `ecp` entries replace, minus the total charge. | electrons |

Each shell (`$defs/BasisShell`, strict):

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `center` | integer ≥ 0 | **yes** | 0-based index into the atoms of the `structure_ref` section. | — |
| `l` | integer, 0–7 | **yes** | Shell angular momentum. | — |
| `exponents` | ≥ 1 numbers, each > 0 | **yes** | Primitive exponents. | bohr⁻² |
| `coefficients` | ≥ 1 numbers | **yes** | Contraction coefficients on `N_i`-normalized primitives (Appendix A.1), one per exponent. | — |
| `pure` | boolean | no | Overrides the payload-level `pure` for this shell. | — |

`exponents` and `coefficients` **MUST** have the same length. JSON Schema
cannot express that rule, so the schema does not check it.

`n_electrons`, when present, **MUST** equal the sum of the `mo_metadata`
occupations over both spin channels, within 1e-6, unless
`occupation_semantics` is `transition_weight`; and when root
`provenance.n_electrons` (§ 3.2) is present as well, the two **MUST** be equal.

#### Effective core potentials

An `ecp` entry (`$defs/BasisEcpCentre`, strict) replaces the
`n_core_electrons` innermost electrons of one atom by a potential. The
wavefunction describes the other `Z − n_core_electrons` electrons of that
atom, which is why `n_electrons` subtracts the core counts. A consumer that
computes a property from the wavefunction, such as the electrostatic
potential, a population or an electron count, gives an ECP atom the effective
nuclear charge `Z − n_core_electrons`.

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `center` | integer ≥ 0 | **yes** | The atom, a 0-based index as for a shell's `center`. No two entries name the same atom. | — |
| `n_core_electrons` | integer ≥ 0 | **yes** | Number of electrons the potential replaces on this atom. | electrons |
| `label` | string | no | Name of the potential, e.g. `"def2-ECP"`. | — |
| `projectors` | array of radial functions (below) | no | The local part and the semi-local channels. Absent, the potential is named by `label` and its core count only, and cannot be evaluated. | — |

Each radial function (`$defs/BasisEcpProjector`, strict) is a sum of terms in
the conventional semi-local form,

```text
U(r) = Σ_i c_i · r^(n_i − 2) · exp(−α_i r²)        r in bohr, U in hartree
```

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `l` | integer ≥ 0 | **yes** | Angular momentum of a semi-local channel; for the local part, its conventional label `L`. | — |
| `exponents` | ≥ 1 numbers, each > 0 | **yes** | The `α_i`. | bohr⁻² |
| `coefficients` | ≥ 1 numbers | **yes** | The `c_i`. | hartree · bohr^(2 − n_i) |
| `powers` | ≥ 1 integers | **yes** | The radial powers `n_i`: `n_i = 2` is a plain Gaussian, `n_i = 0` carries `r⁻²`. | — |
| `local` | boolean | no | `true`: this function is the local part. Absent means `false`. | — |

The three arrays are parallel and **MUST** have the same length. The potential
of the centre is

```text
V(r) = U_L(r) + Σ_channels Σ_{m=−l..l} |l m⟩ U_l(r) ⟨l m|
```

where `U_L` is the radial function with `local: true` (zero when there is
none), which acts on every angular momentum, and each semi-local channel
multiplies the projector `|l m⟩⟨l m|` onto angular momentum `l` about the
centre. A centre has at most one local part, and its `l` is the highest `l`
of the centre: the local part of a potential with s, p and d channels is
labelled `l = 3`. As in the tabulations of Gaussian, Molpro and NWChem, a
semi-local channel of a centre with a local part holds the difference
`U_l(r) − U_L(r)`: QVF stores the function that multiplies the projector, so a
producer copies such a table without converting it.

**`mo_metadata` payload** — schema'd as `$defs/MoMetadataPayload` (strict).
`spin` selects one of two shapes. `"unrestricted"` carries its orbitals in an
`alpha` and a `beta` object, matching `mo_coefficients_alpha` and
`mo_coefficients_beta`, and no top-level `n_mo`, `energies`, `occupations` or
`symmetry_labels`. Every other `spin` value carries one set of orbitals at the
top level, matching `mo_coefficients`, and no `alpha` or `beta`. Per-MO arrays
are in the row order of the coefficient member.

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `spin` | `"restricted"`, `"unrestricted"`, `"restricted_open"` or `"generalized"` | **yes** | `restricted`: one set of spatial orbitals shared by both spins. `unrestricted`: separate α and β sets. `restricted_open`: one set of spatial orbitals, singly or doubly occupied. `generalized`: one set of two-component spin orbitals that mix α and β. See *Spin and complex coefficients* below. | — |
| `orbital_kind` | `"canonical"`, `"natural"`, `"localized"`, `"nto"`, `"nbo"`, `"iao"`, `"ibo"`, `"boys"`, `"pipek_mezey"` or `"custom"` | no | What the orbitals are: eigenfunctions of the converged Fock or Kohn–Sham operator (`canonical`, the meaning when absent), natural orbitals, localized orbitals of an unnamed method, natural transition orbitals, natural bond orbitals, intrinsic atomic or bond orbitals, Foster–Boys or Pipek–Mezey localized orbitals, or anything else (`custom`). | — |
| `n_ao` | integer ≥ 0 | no | Number of AOs; equals the `basis` payload's `n_ao`. | — |
| `n_mo` | integer ≥ 0 | no; single set only | Number of MOs; equals the row count of `mo_coefficients`. | — |
| `energies` | array of numbers | **yes**, single set only | Orbital energy of each MO. | Hartree |
| `occupations` | array of numbers | **yes**, single set only | Occupation of each MO: electrons per orbital, or a transition weight (see `occupation_semantics`). | electrons |
| `symmetry_labels` | array of strings | no; single set only | Irreducible-representation label of each MO, one per entry of `energies`. | — |
| `alpha`, `beta` | spin channels (below) | **yes**, `unrestricted` only | The α and the β orbitals. | — |
| `occupation_semantics` | `"electron_occupation"` or `"transition_weight"` | no | Meaning of `occupations`: electrons per orbital (the meaning when absent) or the weights of natural transition orbitals. | — |
| `k_point` | 3 numbers | no | Crystal momentum of the orbitals, in fractional coordinates of the reciprocal lattice; absent or `[0, 0, 0]` means Γ. | — |
| `complex` | boolean | no; **yes** for `generalized`, with the value `true` | `true`: every coefficient member is complex, float64 `[n_mo, n_ao, 2]` with the last axis (real, imaginary). Absent or `false`: real `[n_mo, n_ao]`. | — |
| `coefficient_encoding` | `"real"` or `"complex_split_last_axis"` | no | Alternative spelling of `complex` (`false` and `true`); when both are present they **MUST** agree. | — |
| `coefficient_components` | `["real", "imag"]` | no | Order of the last axis of complex coefficients. | — |
| `state_ref` | string | no | Reference to the electronic state the orbitals describe, for state-specific orbitals. For natural transition orbitals it is the `id` of the `excited_states` section (§ 5.12) of the state. | — |
| `state_index` | integer ≥ 1 | no | The `index` of the state the orbitals describe, in the `excited_states` section that `state_ref` names. | — |
| `nto_role` | `"hole"` or `"particle"` | no | Which half of a state's natural transition orbitals the section holds (below). | — |
| `localization_method` | string | no | Name of the method that localized the orbitals. | — |
| `localization_reference_basis` | string, non-empty | no | Name of the minimal reference basis of free-atom AOs from which the producer constructed intrinsic atomic orbitals (IAOs), for IAO-based localization and IAO charges, e.g. `"minao"`. See below. | — |
| `atom_populations` | `[n_mo, n_atoms]` numbers | no | Fraction of each localized orbital on each atom; each row sums to 1. | — |
| `centroids_bohr` | `[n_mo, 3]` numbers | no | Centroid of each localized orbital. | bohr |
| `n_centres` | array of integers ≥ 0 | no | Number of atoms each localized orbital is centred on. | — |

A spin channel (`$defs/MoSpinChannel`, strict) carries `energies` (Hartree)
and `occupations` (electrons, 0 to 1), both required, and optionally `n_mo`
(the row count of its coefficient member) and `symmetry_labels` (one per
entry of its `energies`). What `state_ref` refers to for orbitals other
than natural transition orbitals is not yet specified.

**The IAO reference basis.** Intrinsic atomic orbitals (IAOs) are built from
the occupied orbitals of the calculation and a *minimal* basis of free-atom
AOs, and intrinsic bond orbitals (IBOs) by localizing the occupied orbitals on
them (G. Knizia, *J. Chem. Theory Comput.* **2013**, 9, 4834–4843,
doi:10.1021/ct400687b). The construction takes the minimal basis as an input:
the IAOs, the IAO charges and the IBOs change slightly with it (that paper
uses free-atom AOs from cc-pVTZ, MINAO, and compares them with Huzinaga's
MINI and with ANO-RCC functions). A producer that localizes on IAOs
**SHOULD** therefore name that basis in `localization_reference_basis`, by the
name its program uses (`"minao"`, `"mini"`, …), beside `localization_method`.

**Deprecated fields.** Three keys that producers wrote before the canonical
carriers existed remain valid, so that archives which carry them keep
validating. The schema annotates them `"deprecated": true`; producers
**SHOULD NOT** write them, consumers **SHOULD** prefer the canonical carrier
when both are present, and they may be removed in a future major version.

| Field | Type | Meaning | Canonical carrier |
|---|---|---|---|
| `iao_charges` | array of numbers, one per atom | IAO partial charges, in e. | The `iao_charge` member of an `atom_properties` section (§ 5.7). |
| `excitation_index` | integer ≥ 0 | The excited state natural transition orbitals describe, in the producer's numbering. | `state_index`, the state's `index` in the `excited_states` section that `state_ref` names (§ 5.12). |
| `excitation_energy_ev` | number | That state's excitation energy, in eV. | The state's `excitation_energy` in that `excited_states` section. |

**Natural transition orbitals.** The natural transition orbitals (NTOs) of an
excited state (§ 5.12) are written as **two** `wavefunction.gto` sections with
the basis of the reference wavefunction: one holds the *hole* orbitals, which
span the occupied space of the reference, and one the *particle* orbitals,
which span its virtual space. Both `mo_metadata` payloads carry
`orbital_kind: "nto"`, `occupation_semantics: "transition_weight"`, the same
`state_ref` (the `id` of the `excited_states` section) and `state_index` (the
`index` of the state there), and `nto_role`: `"hole"` in one section,
`"particle"` in the other. Row k of the hole section and row k of the particle
section are the k-th hole–particle pair, so the two sections have the same
number of rows, in order of decreasing weight, and the same `occupations`:
the weight λₖ of each pair. λₖ is the k-th eigenvalue of T Tᵀ for the
transition density matrix T in the orbital basis, the square of its k-th
singular value, and the weights of a normalized state sum to 1. `energies`,
which the payload requires, has no meaning for NTOs: producers **SHOULD**
write zeros, and consumers ignore it. The state's `nto_ref` names the hole
section; a consumer finds the particle section as the `wavefunction.gto`
section whose `mo_metadata` has the same `state_ref` and `state_index` and
`nto_role: "particle"`.

For an **unrestricted** reference, there are still **two sections per state**,
not four. Both payloads use `spin: "unrestricted"`, with `alpha` and `beta`
objects and `mo_coefficients_alpha` / `mo_coefficients_beta` members. Each
spin channel has its own corresponding-orbital transformation and its own
hole–particle pairs. Row k of the hole's α channel pairs with row k of the
particle's α channel, and likewise for β. Hole and particle **MUST** have
the same row count and the same `occupations` in each spin channel. The α
and β row counts may differ, including an empty channel. Each channel's
`occupations` **MUST** be nonnegative, in non-increasing order, with one
weight per coefficient row; its `energies` and optional `n_mo` have that
same length/count. The weights are squared singular values λₖ = σₖ², not
singular values. For a normalized state, **both spins together** sum to 1;
neither spin is separately normalized. The hole and particle each carry
the same weights, so this sum counts each pair once. `nto_ref` still names
the hole section and the particle lookup above is unchanged and unique.
The state identified by `state_ref` and `state_index` **MUST** exist.

For compatibility, deprecated NTO metadata without `nto_role` remains
accepted as in toolkit v0.3.0 and is exempt from the new unrestricted
channel/pair checks. Consumers MUST NOT infer a missing role from section
order. New writers use the complete canonical fields described above.

The normalization statement is conditional: full TDDFT/RPA transition
densities can have a non-unit weight sum because of de-excitation terms.
QVF does not label a transition density as normalized, and validators do
not impose a unit sum. See R. L. Martin, *J. Chem. Phys.* **118**, 4775
(2003), [doi:10.1063/1.1558471](https://doi.org/10.1063/1.1558471),
equations 4–9 and the spin-unrestricted extension. This clarification does
not specify NTOs of restricted-open references.

#### Spin and complex coefficients

* **`restricted`** and **`restricted_open`** use the single-set shape: one
  `mo_coefficients` member and top-level `energies` and `occupations`.
  Occupations run from 0 to 2 electrons per spatial orbital. A
  `restricted_open` wavefunction has singly occupied orbitals (fractional
  occupations are allowed, for natural or ensemble orbitals); by the usual
  high-spin convention a singly occupied orbital carries an α electron.
* **`unrestricted`** uses the two-set shape: `mo_coefficients_alpha` and
  `mo_coefficients_beta`, described by the `alpha` and `beta` spin channels,
  with occupations from 0 to 1.
* **`generalized`** is a two-component (spinor) wavefunction in the
  single-set shape. Each MO is a spin orbital
  `ψ_i = Σ_μ (C[i, μ] χ_μ α + C[i, n_ao + μ] χ_μ β)`: the AO axis of
  `mo_coefficients` has **2 n_ao** entries in the spinor layout
  **`[α AOs…, β AOs…]`**, the α components on the `n_ao` AOs of `basis` in
  their Appendix A.2 order, then the β components on the same AOs in the same
  order. `n_ao` in `basis` and `mo_metadata` stays the number of spatial AOs.
  Occupations run from 0 to 1. A `generalized` wavefunction **MUST** carry
  `complex: true`; a real one is written with a zero imaginary part.
* **`complex: true`** means that every coefficient member has a trailing axis
  of length 2, `[re, im]`: `[n_mo, n_ao, 2]`, or `[n_mo, 2 n_ao, 2]` for
  `generalized`. With `complex` absent or `false` every coefficient member is
  real and has no such axis. `coefficient_encoding:
  "complex_split_last_axis"` is the same statement as `complex: true` (and
  `"real"` as `false`) and is read the same way when `complex` is absent.

#### AO overlap and density matrices

The optional members `overlap`, `density_matrix`, `density_matrix_alpha` and
`density_matrix_beta` are float64 `[n_ao, n_ao]` matrices whose rows and
columns follow the AO order of `basis` (Appendix A.2), real in every case,
including a complex or `generalized` wavefunction.

* **`overlap`** is the AO overlap matrix `S[μ, ν] = ⟨χ_μ | χ_ν⟩` of the
  contracted AOs exactly as `basis` defines them: primitives normalized with
  the `N_i` of Appendix A.1 and contracted with the stored coefficients.
  Producers **SHOULD** write it. Its diagonal lets any consumer check that the
  stored coefficients and AO order mean what the producer's own integral code
  meant (Appendix A.1, *Verifying with `overlap`*), and validators do: a
  diagonal that disagrees with the basis is the error `E_NORMALIZATION`.
* **`density_matrix`** is the spin-summed (total) one-particle density matrix
  `P` in the AO basis of a single-set wavefunction, so that the electron
  density is `ρ(r) = Σ_μν P[μ, ν] χ_μ(r) χ_ν(r)` and `tr(P S)` is the number
  of electrons it describes. For orbitals that define it, such as an SCF
  wavefunction, `P[μ, ν] = Σ_i n_i C[i, μ] C[i, ν]` with the occupations
  `n_i`; a correlated method may store its own density, which its orbitals do
  not reproduce. For complex coefficients `P` is the real part of the
  Hermitian density matrix, all that `ρ` needs because the AOs are real; for
  `generalized` it is the spin-summed spatial density,
  `P[μ, ν] = Re Σ_i n_i (C̄[i, μ] C[i, ν] + C̄[i, n_ao + μ] C[i, n_ao + ν])`.
* **`density_matrix_alpha`** and **`density_matrix_beta`** are the α and the β
  density matrices of an unrestricted wavefunction; the total density is
  their sum and the spin density their difference.

The density members follow the coefficient members: `density_matrix` appears
only with `mo_coefficients`, and `density_matrix_alpha` and
`density_matrix_beta` appear only together and only with
`mo_coefficients_alpha` and `mo_coefficients_beta`.

#### Checks

Besides the payload definitions, validators check (§ 6): the rank of every
coefficient member against `complex`; the length of every `symmetry_labels`
against its `energies`; the `[n_ao, n_ao]` shape of `overlap` and of the
density members, and their pairing with the coefficient members; `n_electrons`
against the occupations (all `E_MEMBER_MALFORMED`) and against
`provenance.n_electrons` (`E_ROOT_INCONSISTENT`); and the diagonal of
`overlap` against Appendix A.1 (`E_NORMALIZATION`).

#### 5.3.1 `wavefunction.bloch` — k-resolved Bloch orbitals, provisional

> **Provisional.** This kind was promoted from the vendor section
> `x_vibeqc.bloch_wavefunction` by maintainer decision before a second
> independent producer shipped it (`GOVERNANCE.md`, "Vendor → canonical
> promotion"); its `registry.json` `promotions` entry carries
> `"status": "provisional"`. It is canonical in every other respect: the
> schema and the validators check it, and the conformance corpus exercises
> it. It becomes final when a second producer ships it, and any revision
> needed before then is additive (§ 7.2).

Carries the k-resolved crystalline orbitals of a periodic system: complex MO
coefficients at every k-point of a set, for one or two spin channels, over the
atom-centred Gaussian basis of the cell, with their energies and occupations.
A producer whose orbitals live at Γ alone may keep using `wavefunction.gto`
(§ 5.3); a producer with orbitals at several k-points, or at one k-point other
than Γ, writes this kind.

```jsonc
{
  "id": "wf_bloch",
  "kind": "wavefunction.bloch",
  "members": {
    "basis":       {"path": "wf_bloch/basis.json", "format": "json", "sha256": "…"},
    "kpoints":     {"path": "wf_bloch/kpoints.json", "format": "json", "sha256": "…"},
    "mo_metadata": {"path": "wf_bloch/mo_metadata.json", "format": "json", "sha256": "…"},
    "mo_coefficients": {"path": "wf_bloch/mo_coefficients.bin", "format": "binary",
                        "dtype": "float64", "shape": [2, 2, 2, 2, 2], "sha256": "…"},
    "energies":    {"path": "wf_bloch/energies.bin", "format": "binary",
                    "dtype": "float64", "shape": [2, 2, 2], "sha256": "…"},
    "occupations": {"path": "wf_bloch/occupations.bin", "format": "binary",
                    "dtype": "float64", "shape": [2, 2, 2], "sha256": "…"}
  }
}
```

**Bloch sum.** The `basis` payload is the one of § 5.3, with the AO order and
the primitive normalization of Appendix A; its shell `center` indices refer to
the atoms of one cell. For each AO χ_μ the Bloch function at crystal momentum
k is

```text
φ_μk(r) = N^(−1/2) · Σ_T e^{+i k·T} · χ_μ(r − T)
```

where T runs over the lattice translations T = n₁a₁ + n₂a₂ + n₃a₃ along the
periodic axes (n_i = 0 for an axis whose `pbc` entry is false) and N is the
number of cells in the Born–von Kármán supercell. k is given in fractional
coordinates of the reciprocal lattice, k = k₁b₁ + k₂b₂ + k₃b₃ with
a_i · b_j = 2π δ_ij, so k·T = 2π (k₁n₁ + k₂n₂ + k₃n₃). The phase is
e^{+ik·T}, and the AO's position inside the cell is not part of it. Orbital m
of spin channel s at k-point k is

```text
ψ_msk(r) = Σ_μ C_ksmμ · φ_μk(r),   C_ksmμ = c[k, s, m, μ, 0] + i · c[k, s, m, μ, 1]
```

with c the `mo_coefficients` array. In this convention the overlap of two
Bloch functions is S_μν(k) = Σ_T e^{ik·T} ⟨χ_μ(r) | χ_ν(r − T)⟩, and the
orbitals of a converged calculation are orthonormal in that metric, C†S(k)C
= 1. The electron density of the cell is
ρ(r) = Σ_k w_k Σ_s Σ_m n_ksm |ψ_msk(r)|², with the `weights` w of the
`kpoints` payload and the `occupations` n. An orbital multiplied by a phase
e^{iθ} is the same state; consumers **MUST NOT** assume a particular gauge.

| Member | Format | Required | Shape and meaning | Unit |
|---|---|:--:|---|---|
| `basis` | JSON, `$defs/BasisPayload` (strict) | **yes** | The contracted shells of the cell, as for `wavefunction.gto` (§ 5.3). Its `structure_ref` names the structure unless the section carries its own. | — |
| `kpoints` | JSON, `$defs/BlochKpointsPayload` (strict) | **yes** | The k-points, in the order of axis 0 of the binary members; table below. | — |
| `mo_metadata` | JSON, `$defs/BlochMoMetadataPayload` (strict) | **yes** | The orbital set; table below. | — |
| `mo_coefficients` | float64 | **yes** | `[n_k, n_spin, n_mo, n_ao, 2]`: the coefficient of Bloch function μ in orbital m of spin channel s at k-point k, with the last axis (real, imaginary), as for complex `wavefunction.gto` coefficients. | — |
| `energies` | float64 | **yes** | `[n_k, n_spin, n_mo]` orbital energies. | hartree |
| `occupations` | float64 | **yes** | `[n_k, n_spin, n_mo]` electrons in each orbital at each k-point, not weighted by the k-point weights: 0 to 2 when `n_spin` is 1, 0 to 1 when it is 2. | electrons |

Spin channel 0 is α and 1 is β. Unlike the arrays of § 5.6, whose spin axis
leads, the orbital members keep the k-point axis first, so that the orbitals
of one k-point are contiguous.

Payload `kpoints` (`$defs/BlochKpointsPayload`, strict; `kpoints` is
required):

| Field | Type | Meaning | Unit |
|---|---|---|---|
| `kpoints` | ≥ 1 arrays of 3 numbers | Every k-point, in fractional reciprocal-lattice coordinates (see above). n_k is their number. | — |
| `weights` | array of numbers ≥ 0 | Integration weight of each k-point, one per k-point, summing to 1 within 1e-6. Absent when the k-points are not an integration set, such as points along a path. | — |
| `labels` | array of strings | One label per k-point, such as a high-symmetry name; `""` marks an unlabelled point. | — |
| `mesh` | 3 integers ≥ 1 | Dimensions of the uniform mesh the k-points were taken from, when they were: the k-points are its points, or the irreducible subset of them that the weights account for. | — |
| `shift` | 3 numbers | Fractional reciprocal-lattice position of mesh point (0, 0, 0); absent means a Γ-centred mesh. | — |
| `time_reversal` | boolean | `true` when the set was reduced with time-reversal symmetry: the orbitals at −k are the complex conjugates of those at k, and the weight of k includes that of −k. Absent means `false`. | — |
| `units` | object | Optional declaration (§ 4.3, *Declared units*) restating that `kpoints`, `weights` and `shift` are `dimensionless`. | — |

Payload `mo_metadata` (`$defs/BlochMoMetadataPayload`, strict). Per-orbital
energies and occupations are the binary members above, not fields of this
payload.

| Field | Type | Required | Meaning |
|---|---|:--:|---|
| `spin` | `"restricted"` or `"unrestricted"` | **yes** | One set of spatial orbitals for both spins (`n_spin` 1), or separate α and β sets (`n_spin` 2). |
| `n_spin` | 1 or 2 | **yes** | Axis 1 of the binary members; 1 for `restricted`, 2 for `unrestricted`. |
| `n_mo` | integer ≥ 0 | **yes** | Axis 2 of the binary members: the orbitals at each k-point and spin. |
| `n_ao` | integer ≥ 0 | **yes** | Axis 3 of `mo_coefficients`; equals the `basis` payload's `n_ao`. |
| `complex` | `true` | **yes** | The coefficients are complex; always `true`. |
| `k_point_convention` | `"fractional"` | **yes** | `kpoints` are fractional reciprocal-lattice coordinates; the only value in v1. |
| `orbital_kind` | as in § 5.3 | no | What the orbitals are; absent means `canonical`. |
| `n_mo_per_k` | array of integers ≥ 0 | no | The number of orbitals that are defined at each k-point, one entry per k-point, each at most `n_mo`, for a producer whose orbital count differs between k-points (canonical orthogonalisation removing a different number of near-linear-dependent combinations at each k). Orbitals `n_mo_per_k[k]` to `n_mo − 1` at k-point k are padding with zero coefficients and zero occupations, and consumers ignore them. Absent means `n_mo` everywhere. |
| `fermi_energy` | `{value, units}` | no | Fermi level, on the absolute scale of `energies`. |
| `units` | object | no | Optional declaration restating that `energies` is `hartree`. |

**Structure.** The section's structure is the one its section-level
`structure_ref` names, else the one `basis.structure_ref` names (§ 4.1). It
**MUST** be a `structure` section with at least one periodic axis (§ 5.1).

**Rules a validator checks** (§ 6): `mo_coefficients` is
`[n_k, n_spin, n_mo, n_ao, 2]` with `n_spin` 1 or 2; `energies` and
`occupations` have its `[n_k, n_spin, n_mo]`; the `kpoints` payload lists n_k
k-points and, when present, n_k `labels` and n_k `weights` that sum to 1
within 1e-6; `mo_metadata` states the `n_spin` that `spin` implies and the
`n_spin`, `n_mo` and `n_ao` of the binary members, and `n_mo_per_k` has n_k
entries of at most `n_mo`; `basis.n_ao` equals `n_ao`. A disagreement, or a
structure with no periodic axis, is `E_MEMBER_MALFORMED`; a
`basis.structure_ref` that does not name a `structure` section (when the
section names none itself) is `E_REF_UNRESOLVED`.

**With `bands`.** When an archive carries both a `wavefunction.bloch` section
and a `bands` section (§ 5.6) whose k-points include some of the Bloch
section's, the band energies at those k-points **SHOULD** agree with
`energies` once converted from hartree to eV (mind the axis orders,
`[n_spin, n_k, n_bands]` against `[n_k, n_spin, n_mo]`). Validators do not
check this.

### 5.4 Spectra

Every `spectra.*` section carries exactly one JSON member, `spectrum`. The six
one-dimensional kinds share one payload, `$defs/SpectrumPayload`;
`spectra.nmr` and `spectra.epr` carry object-shaped parameter sets,
`$defs/NmrSpectrumPayload` and `$defs/EprSpectrumPayload`.

Each payload definition in § 5 has one of three strictness levels:

* **strict** — no keys other than the listed ones, at the root and in every
  listed object, except extension keys whose names begin with `x_`;
* **typed-open** — the listed fields are typed; the root stays open for forward
  growth;
* **permissive** — the listed fields are typed and every object stays open;
  a later revision may tighten the definition once producer conventions
  settle.

A consumer ignores keys it does not know in an open object.

#### One-dimensional spectra

`spectra.ir`, `spectra.raman`, `spectra.uvvis`, `spectra.ecd`, `spectra.vcd`
and `spectra.generic`. Payload: `$defs/SpectrumPayload`, **typed-open**.

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `frequencies` | number[] | **yes** | The x value of every point or line. | the kind's § 4.3 unit, or `x_units` |
| `intensities` | number[] | **yes** | The y value of each `frequencies` entry, in the same order. | the kind's § 4.3 unit, or `y_units` |
| `x` | number[] | no | Alias of `frequencies` under a quantity-neutral name: the same values in the same order. | as `frequencies` |
| `x_units` | unit name (§ 4.3) | no | Unit of `frequencies` and `x`. | — |
| `y_units` | unit name (§ 4.3) | no | Unit of `intensities`. | — |
| `peaks` | object[] | no | Annotated peaks: `x` (number, required), `y` (number), `width` (number ≥ 0, full width at half maximum), `label` (string), `assignment` (`{mode_index?, state_index?}`, integers ≥ 0). | `x`, `width` as `frequencies`; `y` as `intensities` |
| `broadening` | object | no | The line shape the producer applied, or recommends: `type` (required: `none`, `gaussian` or `lorentzian`), `fwhm` (number > 0), `units` (unit name). | `fwhm` in `units`, else in the x unit |
| `temperature_k` | number > 0 | no | Temperature the spectrum refers to. | K |
| `raman` | object | no | `laser_nm` (number > 0): excitation wavelength; `depolarization_ratios` (number[]): one per line, in `frequencies` order. | nm; dimensionless |
| `uvvis` | object | no | `x_axis` (`energy_ev` or `wavelength_nm`): the x quantity the producer recommends for display. It does not change the stored values. | — |
| `states_ref` | string | no | `id` of the `excited_states` section (§ 5.12) whose states the peaks' `assignment.state_index` values name. | — |

**The x axis.** `frequencies` carries the x axis in the unit § 4.3 gives the
kind — cm⁻¹ for `spectra.ir`, eV for `spectra.uvvis` — unless `x_units`
declares another unit. The name is historical and is kept for every kind,
including those whose x values are energies (UV-Vis, ECD); `x` is an optional
quantity-neutral alias, and a rename is left to a future major version. For a
kind § 4.3 gives no unit (`spectra.raman`, `spectra.ecd`, `spectra.vcd`,
`spectra.generic`) producers **SHOULD** declare `x_units` and `y_units`;
consumers conventionally assume cm⁻¹ for Raman and VCD and eV for ECD. A
declared unit is a declaration, not a conversion: the values are stored in it,
and a consumer **MUST NOT** read them in the kind's default unit. `x_units`,
`y_units` and `broadening.units` are unit names of § 4.3 (*Declared units*).

**One length.** `frequencies`, `intensities` and, when present, `x` **MUST**
have the same length. JSON Schema cannot express this rule, so a validator
checks it separately (§ 6).

`peaks[].assignment.mode_index` is a 0-based index into the modes of the
archive's `vibrations` section (§ 5.5); `state_index` identifies the
electronic state a transition reaches. When the payload carries `states_ref`,
every `state_index` is the `index` of a state of the `excited_states` section
that `states_ref` names (§ 5.12), and a validator checks both (§ 6); without
`states_ref`, states are numbered as the producer numbers them.
The `raman` and `uvvis` blocks are meaningful only in the kind they name.

#### `spectra.nmr`

Payload: `$defs/NmrSpectrumPayload`, **permissive**. Atom indices (`atom_index`,
`i`, `j`) are 0-based indices into the `atoms` of the archive's structure.

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `isotope` | string | no | Observed nucleus, e.g. `"1H"`, `"13C"`. | — |
| `reference` | string | no | Shift reference, e.g. `"TMS"`. | — |
| `solvent` | string | no | Solvent or solvation model, when one was used. | — |
| `chemical_shifts` | object[] | no | One entry per nucleus: `atom_index` (integer ≥ 0, required), `symbol`, `isotropic_shift_ppm` (required), `anisotropy_ppm`, `asymmetry`. | ppm; `asymmetry` dimensionless |
| `shielding_tensors` | object[] | no | One entry per nucleus: `atom_index` (integer ≥ 0), `tensor` (3 × 3), `isotropic_ppm`. | ppm |
| `j_couplings` | object[] | no | One entry per atom pair: `i`, `j` (integers ≥ 0, required), `value_hz` (required), `isotope_i`, `isotope_j`. | Hz |

#### `spectra.epr`

Payload: `$defs/EprSpectrumPayload`, **permissive**. Units are fixed: g-values
are dimensionless; hyperfine and zero-field-splitting parameters are in MHz.

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `g_tensor` | object | no | `matrix` (3 × 3), `isotropic` (one third of the trace), `principal` (three principal values). | dimensionless |
| `hyperfine` | object[] | no | One entry per magnetic nucleus: `atom_index` (integer ≥ 0, required), `symbol`, `isotope`, `a_iso_mhz`, `a_tensor_mhz` (3 × 3), `a_principal_mhz` (three values). | MHz |
| `zero_field_splitting` | object | no | `d_mhz` (axial D), `e_mhz` (rhombic E), `tensor_mhz` (3 × 3). | MHz |

A 3 × 3 tensor of either payload is written as three rows in the Cartesian
frame of the structure.

### 5.5 Trajectories, reactions, vibrations

`trajectory`: JSON `metadata` (atom identities, per-frame metadata) + `coords`
`[n_frames, n_atoms, 3]` in Å, plus optional per-frame members (the cell,
velocities, gradients, times, energies, cell volumes, temperatures; below).
`reaction.path` uses the same layout and adds waypoint records
(`{frame_index, label, kind, energy_eh?}`, `kind` ∈
reactant/transition_state/intermediate/product/point). `reaction.waypoints`
carries only annotations and references an existing `trajectory` section via the
section-level `trajectory_ref`. `vibrations`: JSON `metadata` (frequencies in
cm⁻¹) + `displacements` float64 `[n_modes, n_atoms, 3]`. `scan.surface`: JSON
`metadata` + the 1-D coordinate values `axis_a` and `axis_b` + `energies`
`[nA, nB]`, with optional relaxed `geometries`.

**`trajectory` members.** Every binary member is little-endian and C-ordered
(§ 4.2), in the frame order and atom order of `coords`. The per-frame members
are optional and independent: an optimization may carry only `energies`, a
molecular-dynamics run all of them.

| Role | dtype | Shape | Required | Meaning | Unit |
|---|---|---|:--:|---|---|
| `metadata` | JSON | — | **yes** | Atom identities and per-frame metadata (`$defs/TrajectoryMetadataPayload`, below). | — |
| `coords` | float64, or float32 | `[n_frames, n_atoms, 3]` | **yes** | Cartesian position of every atom in every frame. float32 only when `metadata.precision` is `"float32"`. | Å |
| `lattice` | float64 | `[3, 3]` or `[n_frames, 3, 3]` | no | The cell, in the **structure convention**: rows, row `i` the lattice vector of axis `i`, paired with `metadata.pbc[i]`. `[3, 3]` is one cell for every frame; `[n_frames, 3, 3]` one cell per frame (variable-cell, NPT). | Å |
| `velocities` | float64 | `[n_frames, n_atoms, 3]` | no | Velocity of every atom in every frame. | Å / fs |
| `gradient` | float64 | `[n_frames, n_atoms, 3]` | no | The energy **gradient** dE/dR of every atom in every frame, as the `gradient` of a `forces` section (§ 5.10): the forces are its **negative**. The role is named for what it stores, not `forces`, so that the sign cannot be misread. | hartree / bohr |
| `times` | float64 | `[n_frames]` | no | Time of each frame. | fs |
| `energies` | float64 | `[n_frames]` | no | Energy of each frame: the binary form of `metadata.energies`, preferred for long runs. | hartree |
| `cell_volumes` | float64 | `[n_frames]` | no | Volume of the cell of each frame. | Å³ |
| `temperatures` | float64 | `[n_frames]` | no | Instantaneous temperature of each frame. | kelvin |

The schema types these members as `$defs/TrajectoryCoords`,
`$defs/TrajectoryLattice`, `$defs/TrajectoryFrameVectors` (`velocities`,
`gradient`) and `$defs/TrajectoryFrameScalars` (the `[n_frames]` members).
The rules that relate the members to each other are semantic, and a validator
checks them (§ 6):

* **Extents.** `n_frames` and `n_atoms` are the first two extents of
  `coords`. Every per-frame member has `n_frames` as its leading extent, and
  `velocities` and `gradient` have `n_atoms` as their second; a `[n_frames, 3,
  3]` `lattice` has one cell per frame.
* **Precision.** `coords` **MAY** be float32 when `metadata.precision` is
  `"float32"`, which halves the size of a long run; `precision`, when present,
  **MUST** name the dtype of `coords`, and float32 `coords` without it are an
  error. Every other member is float64.
* **Periodicity.** A `trajectory` that carries `lattice` **MUST** carry
  `metadata.pbc`, and a `true` entry of `metadata.pbc` **MUST** be accompanied
  by `lattice`, mirroring the `structure` contract of § 5.1: `pbc` is the
  normative carrier of periodicity, and a row of an aperiodic axis is
  bookkeeping, not a cell edge. A trajectory without `lattice` is molecular.
* **Energies.** When both the binary `energies` and `metadata.energies` are
  present they **MUST** agree within 1e-10 hartree, frame by frame. Producers
  **SHOULD** prefer the binary member for long runs and **MAY** then omit the
  JSON array.
* **Times.** When `times` is absent and `metadata.timestep_fs` is present,
  frame `k` (0-based) is at `k × timestep_fs`. When `times` is present it is
  the time of each frame, whatever `timestep_fs` says.

`trajectory.lattice` is **not** the convention of `reaction.path.lattice`
(columns, bohr); *Lattice conventions* in § 4.3 states both. A validator reads
each in its own convention and never mixes them.

**`reaction.path` members.** A `reaction.path` has the `metadata` and `coords`
of a trajectory, with the same rules, and may carry the per-frame members
`velocities`, `gradient`, `times` and `energies` of the table above, with the
same shapes, units and rules. It carries neither `cell_volumes` nor
`temperatures`, and its cell is its own `lattice` member, below. The
`frame_volumes` and `volume_grid` members carry per-frame volumetric data
(`$defs/Volume4DBinary`, `$defs/VolumeGridPayload`).

**Periodic reaction paths.** `reaction.path` may carry an optional `lattice`
binary member: float64 with shape `[3, 3]` (fixed cell — the common case) or
`[n_frames, 3, 3]` (variable cell), columns = **a, b, c** in **bohr**. Its
**presence is what marks the path periodic** — a `reaction.path` without
`lattice` is unambiguously molecular and renderers must draw no cell. This is
capability detection by member presence, per § 7.1; it does **not** involve a
`qvf_version` bump.
This is **not** the layout or unit of `structure.lattice_vectors` or of
`trajectory.lattice` (rows, Å); see *Lattice conventions* in § 4.3 for the
side-by-side comparison and the conversion.

**`trajectory` `metadata` payload** — schema'd as
`$defs/TrajectoryMetadataPayload` (typed-open), which extends
`$defs/TrajectoryMetadataCommon`, the keys it shares with the `reaction.path`
metadata. It names the atoms, in the atom order of `coords`, through at least
one of `symbols`, `atomic_numbers` and `atoms`; when more than one is present
they **MUST** describe the same atoms. Per-frame arrays are in frame order.

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `symbols` | array of strings | one of three | Element symbol of each atom. | — |
| `atomic_numbers` | array of integers ≥ 0 | one of three | Atomic number of each atom. | — |
| `atoms` | array of `{symbol, atomic_number}` (`$defs/TrajectoryAtom`) | one of three | Atom identities as objects. | — |
| `energies` | array of numbers | no | Energy of each frame; equal to the binary `energies` member within 1e-10 hartree when both are present. | Hartree |
| `n_frames` | integer ≥ 0 | no | Number of frames; equals the leading dimension of `coords`. | — |
| `kind` | `"optimization"`, `"irc"`, `"md"`, `"neb"`, `"scan"` or `"custom"` | no | What produced the frames: `md` is molecular dynamics; `scan` covers relaxed and rigid coordinate scans. | — |
| `pbc` | 3 booleans | for a `trajectory` with `lattice` | Per-axis periodicity of the frames, as in the `structure` payload (§ 5.1); absent means molecular. **REQUIRED** when the section carries `lattice`, and a `true` entry requires `lattice`. | — |
| `ensemble` | `"nve"`, `"nvt"`, `"npt"`, `"nph"` or `"custom"` | no | Thermodynamic ensemble of a molecular-dynamics run. | — |
| `timestep_fs` | number > 0 | no | Time between consecutive frames; frame `k` is at `k × timestep_fs` when the section has no `times` member. | fs |
| `precision` | `"float64"` or `"float32"` | no | Precision of `coords`; absent means `float64`. It **MUST** agree with the member's `dtype`; float32 `coords` require `"float32"`. | — |
| `units` | object | no | Declared units (§ 4.3, *Declared units*), keyed by a field of this payload or a member role of the section. The units of this section remain the contract, so a declaration restates them: `energies` `hartree`, `coords` `angstrom`, `times` and `timestep_fs` `fs`, and for a `trajectory` `lattice` `angstrom`, `cell_volumes` `a3` and `temperatures` `kelvin` (for `reaction.path`, `lattice` `bohr`). `velocities` (Å/fs) and `gradient` (hartree/bohr) have no unit name in the vocabulary and are not declared. | — |

**`reaction.path` `metadata` payload** — schema'd as
`$defs/ReactionPathMetadataPayload` (typed-open), which also extends
`$defs/TrajectoryMetadataCommon`: every field of the `trajectory` table above,
with the same rules except the `pbc` requirement, which belongs to a
trajectory's `lattice` (a reaction path is periodic when it carries its own
`lattice` member), plus:

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `waypoints` | array of waypoint records (below) | no | Annotated frames of the path. | — |
| `reaction_coordinate` | array of numbers | no | Value of the reaction coordinate at each frame: the abscissa of an energy profile. | `reaction_coordinate_unit` |
| `reaction_coordinate_label` | string | no | Display label of the reaction coordinate. | — |
| `reaction_coordinate_unit` | string | no | Unit of `reaction_coordinate`, e.g. `"bohr"` or `"rad"`. | — |
| `dim` | integer, 0–3 | no | Number of periodic dimensions, shared by every frame of a periodic path. | — |
| `dim_per_frame` | array of integers, 0–3 | no | Number of periodic dimensions of each frame, for a path whose frames differ; used instead of `dim`. | — |
| `volume_frame_index` | array of integers ≥ 0 | no | For each slab of `frame_volumes`, in order, the 0-based path frame it belongs to, so that volumes can be emitted for a subset of the frames. One entry per slab. | — |
| `volume_label` | string | no | Display label of the per-frame volumes. | — |
| `volume_isovalue` | number | no | Suggested isovalue for the per-frame volumes. | that of the volume data |

The `volume_grid` member that accompanies `frame_volumes` is a grid payload
(§ 5.2, `$defs/VolumeGridPayload`).

**Per-frame volumes and `volume.series`.** The `frame_volumes` and
`volume_grid` members of a `reaction.path` are the older form of a
`volume.series` section (§ 5.2) whose `axis` has `kind: "frame"`, `values`
equal to the path's `volume_frame_index` (or `0, 1, …, n − 1` when it is
absent) and `trajectory_ref` naming the path; `volume_label` corresponds to
the section `label` of the series, and `volume_isovalue` to the `isovalue`
of a section hint (§ 3.4.1). The two forms carry
the same data, `[n, nx, ny, nz]` slabs on one grid. Producers **SHOULD** write
per-frame volumes as a `volume.series` section, which any trajectory or
reaction path can name; both forms remain valid in v1 and neither is
deprecated, so a consumer that reads per-frame volumes **SHOULD** read both.

**Waypoint record** — `$defs/ReactionWaypoint` (strict), used by `reaction.path`
metadata and by `reaction.waypoints`:

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `frame_index` | integer ≥ 0 | **yes** | 0-based index of the annotated frame: of the section's own `coords` for `reaction.path`, of the `coords` of the trajectory named by `trajectory_ref` for `reaction.waypoints`. | — |
| `label` | string | **yes** | Display label, e.g. `"TS1"`. | — |
| `kind` | `"reactant"`, `"transition_state"`, `"intermediate"`, `"product"` or `"point"` | **yes** | Role of the frame on the path; `point` marks any other annotated frame. | — |
| `energy_eh` | number | no | Energy of the frame. | Hartree |

**`reaction.waypoints` `waypoints` payload** — schema'd as
`$defs/ReactionWaypointsPayload` (strict):

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `waypoints` | array of waypoint records | **yes** | Annotated frames of the referenced trajectory. | — |
| `reaction_coordinate` | array of numbers | no | Value of the reaction coordinate at each frame of that trajectory, in frame order. | — |

**`scan.surface` `metadata` payload** — schema'd as
`$defs/ScanSurfaceMetadataPayload` (typed-open). The axis objects `axis_a` and
`axis_b` are the preferred form. The flat label and unit keys are recognised
alternatives that producers already write; when both forms are present, the
objects take precedence.

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `axis_a`, `axis_b` | axis objects (below) | no | The first driven coordinate (along the rows of `energies`) and the second (along its columns). | — |
| `axis_a_label`, `axis_b_label` | string | no | Flat alternatives to `axis_a.label` and `axis_b.label`. | — |
| `coordinate_a_label`, `coordinate_b_label` | string | no | Flat alternatives to `axis_a.label` and `axis_b.label`. | — |
| `coordinate_a_unit`, `coordinate_b_unit` | string | no | Flat alternatives to `axis_a.unit` and `axis_b.unit`. | — |
| `atoms` | array of `{symbol, atomic_number}` | no | Atom identities, in the atom order of `geometries`. | — |
| `energy_unit` | string | no | Unit of the `energies` member; absent means Hartree. | — |
| `units` | object | no | Declared units, as for `trajectory`; the usual keys are the member roles `axis_a`, `axis_b`, `energies` and `geometries`, none of which has a fixed unit. | — |

An axis object (`$defs/ScanSurfaceAxis`, typed-open) carries `label` (string,
required), `unit` (string: the unit of the values in the matching `axis_a` or
`axis_b` binary member, e.g. `"bohr"` or `"rad"`) and `atoms` (the 0-based
indices of the atoms that define the coordinate: two for a distance, three for
an angle, four for a dihedral).

**`vibrations` `metadata` payload** — schema'd as
`$defs/VibrationsMetadataPayload` (typed-open). Per-mode arrays hold one entry
per mode, in the mode order of `displacements`. An **imaginary mode** is
written as a negative frequency whose magnitude is the imaginary wavenumber:
a mode at 512i cm⁻¹ is written `-512.0`. `imaginary`, when present, **MUST**
agree with the sign of `frequencies`.

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `frequencies` | array of numbers | **yes** | Harmonic wavenumber of each mode; negative for an imaginary mode. | cm⁻¹ |
| `ir_intensities` | array of numbers | no | IR intensity of each mode. | km·mol⁻¹ |
| `raman_activities` | array of numbers | no | Raman scattering activity of each mode. | Å⁴·amu⁻¹ |
| `reduced_masses` | array of numbers | no | Reduced mass of each mode. | amu |
| `force_constants` | array of numbers | no | Force constant of each mode. | mdyn·Å⁻¹ |
| `symmetry_labels` | array of strings | no | Irreducible-representation label of each mode. | — |
| `imaginary` | array of booleans | no | `true` for an imaginary mode. | — |
| `n_modes` | integer ≥ 0 | no | Number of modes; equals the leading dimension of `displacements`. | — |
| `symbols` | array of strings | no | Element symbol of each atom, in the atom order of `displacements`. | — |
| `atoms` | array of atoms as in the `structure` payload (`$defs/StructureAtom`) | no | Reference geometry: one atom per atom of `displacements`, in the same order, with Cartesian positions. | Å |

**`vibrations` `hessian_ref`.** A `vibrations` section **MAY** carry the
section field `hessian_ref` (a peer of `members`), the `id` of the `hessian`
section (§ 5.11) its modes were derived from. A validator checks that it
names a `hessian` section (§ 6).

### 5.6 Bands, DOS, and periodic kinds

Each kind below lists its members; each JSON member names the schema
definition its payload conforms to (`$defs/…`), and a validator checks it
against that definition. A payload is **strict** when it allows only the
fields of its table plus vendor keys beginning `x_`, and **typed-open** when the
fields of its table are typed and further keys are allowed. Units are those of
§ 4.3; the *Unit* columns restate them. All indices are 0-based.

#### Spin axis

An array that resolves spin carries it as its **leading** axis, of length
`n_spin` (1 or 2; index 0 is α, 1 is β), as `bands.eigenvalues`
`[n_spin, n_k, n_bands]` always has. Several members also have an older,
lower-rank form without that axis:

| Member | With the spin axis | Lower-rank form (means `n_spin = 1`) |
|---|---|---|
| `bands.projections` | `[n_spin, n_k, n_bands, n_channels]` | `[n_k, n_bands, n_channels]` |
| `bands.occupations` | `[n_spin, n_k, n_bands]` | — |
| `dos.total.dos` | `[n_spin, n_points]` | `[n_points]` |
| `dos.projected.projections` | `[n_spin, n_channels, n_points]` | `[n_channels, n_points]` |
| `dos.coop` / `dos.cohp` `projections` | `[n_spin, n_pairs, n_points]` | `[n_pairs, n_points]` |
| `dos.coop` / `dos.cohp` `integrated` | `[n_spin, n_pairs]` | `[n_pairs]` |

Both forms are legal, and a consumer reads the lower-rank form as `n_spin = 1`.
In a `bands` section a rank-4 `projections` and an `occupations` **MUST** have
the `[n_spin, n_k, n_bands]` extents of `eigenvalues`, and a rank-3
`projections` its `[n_k, n_bands]`; a validator reports a disagreement as
`E_MEMBER_MALFORMED` (§ 6). A producer whose `eigenvalues` have two spin
channels **SHOULD** write `projections` with the spin axis; a rank-3
`projections` beside two spin channels carries one set of weights, which a
consumer **MAY** apply to both. A DOS section with a spin axis also states
`n_spin` (a section field of `dos.total` and `dos.projected`, a field of the
`meta` payload of `dos.coop` and `dos.cohp`), and it **MUST** equal the length
of that axis. For two-component (spinor) bands `n_spin` is 1.

#### `bands`

| Member | Format | Required | Shape and meaning | Unit |
|---|---|:--:|---|---|
| `kpath` | JSON, `$defs/KpathPayload` (strict) | **yes** | The path, the Fermi level and the fat-band channels; table below. | — |
| `eigenvalues` | float64 | **yes** | `[n_spin, n_k, n_bands]` band energies on the absolute scale (not shifted by the Fermi level). | eV |
| `projections` | float64, `$defs/BandsProjections` | no | Fat-band weights, `[n_spin, n_k, n_bands, n_channels]` or `[n_k, n_bands, n_channels]`; slice `c` of the last axis belongs to `kpath.channels[c]`. | — |
| `occupations` | float64, `$defs/BandsOccupations` | no | `[n_spin, n_k, n_bands]`: electrons in each state, not weighted by k-point weights (0 to 2 when `n_spin` is 1 and the bands are not spinors, otherwise 0 to 1). | electrons |

Payload `kpath` (`$defs/KpathPayload`, strict; no field is required):

| Field | Type | Meaning | Unit |
|---|---|---|---|
| `kpoints` | array of `[3]` numbers | Every k-point of the path in the order of `eigenvalues` axis 1, in fractional reciprocal-lattice coordinates. | — |
| `labels` | array of strings | One label per k-point, parallel to `kpoints`; `""` marks an unlabelled point. | — |
| `segments` | array of segment objects | The straight segments of the path, in path order; table below. | — |
| `fermi` | number | Fermi level, on the scale of `eigenvalues`. | eV |
| `fermi_energy_ev` | number | The same Fermi level under the name the DOS kinds use; equal to `fermi` when both are present. | eV |
| `spinor` | boolean | `true` for two-component (spin–orbit) bands, each band one spinor state. Default `false`. | — |
| `n_spin` | integer, 1 or 2 | Equal to `eigenvalues` axis 0. | — |
| `n_kpoints` | integer ≥ 1 | Equal to `eigenvalues` axis 1. | — |
| `n_bands` | integer ≥ 1 | Equal to `eigenvalues` axis 2. | — |
| `channels` | array of channel objects or strings | One per last-axis slice of `projections`; table below. A string is shorthand for a channel object with only that `label`. | — |
| `title` | string | Display title. | — |
| `units` | object | Optional declaration, field name or member role → unit name, restating the units of this table (§ 4.3, *Declared units*): `fermi`, `fermi_energy_ev` and `eigenvalues` are `ev`. | — |
| `kind`, `version`, `reciprocal_space` | `"bands"`, string, boolean | Informational keys that existing producers write; a consumer needs none of them. | — |

Segment object (`$defs/KpathSegment`, strict; also used by `qpath`):

| Field | Type | Meaning |
|---|---|---|
| `start` | integer ≥ 0 | Index of the segment's first point along the path axis. Absent: the previous segment's `end`, or 0 for the first segment. |
| `end` | integer ≥ 0 | Index of its last point, inclusive. Absent: `start + n_points − 1`. |
| `n_points` | integer ≥ 1 | Number of points on the segment, both ends included. |
| `label_start`, `label_end` | string | High-symmetry labels of the two end points; may be empty. |
| `k_start`, `k_end` | `[3]` numbers | Fractional reciprocal-lattice coordinates of the two end points. |

Consecutive segments therefore share their boundary point, unless the later
one gives an explicit `start`; that is how a path with a jump (`X|U`) is
written.

Channel object (`$defs/DosChannel`, typed-open; also used by the DOS and
phonon payloads below):

| Field | Type | Meaning |
|---|---|---|
| `label` | string, **required** | Display label, e.g. `"Fe-3d"`. |
| `atom` | integer ≥ 0 | Atom index into the `structure` section. Absent: summed over atoms. |
| `l` | integer ≥ 0 | Shell angular momentum. Absent: summed over shells. |
| `m` | integer | Component of a spherical shell, −l to l, as in Appendix A.2. Absent: summed over the shell. |
| `spin` | 0 or 1 | For a channel that is itself spin-resolved, which arises only in a projection without a spin axis. |

Further keys, such as an element `symbol`, are allowed.

#### `dos.total` and `dos.projected`

| Member | Format | Required | Shape and meaning | Unit |
|---|---|:--:|---|---|
| `energies` | float64 | **yes** | `[n_points]` energy grid, referenced to E_F at 0. | eV |
| `dos` (`dos.total`) | float64 | **yes** | `[n_points]` or `[n_spin, n_points]`. | states / eV |
| `projections` (`dos.projected`) | `$defs/DOSProjectedProjections` | **yes** | `[n_spin, n_channels, n_points]` or `[n_channels, n_points]`; row `c` belongs to entry `c` of the section's `channels` list. | states / eV |

#### `dos.coop` and `dos.cohp`

Crystal orbital overlap (COOP) and Hamilton (COHP) populations of atom pairs.

| Member | Format | Required | Shape and meaning | Unit |
|---|---|:--:|---|---|
| `energies` | float64 | **yes** | `[n_points]` energy grid, referenced to E_F at 0. | eV |
| `projections` | float64 | **yes** | `[n_pairs, n_points]` or `[n_spin, n_pairs, n_points]`: the population curve of each pair. | — |
| `integrated` | float64 | **yes** | `[n_pairs]` or `[n_spin, n_pairs]`: each curve integrated up to E_F. | — |
| `meta` | JSON, `$defs/DosMetaPayload` (typed-open) | **yes** | What the pair axis means; table below. | — |

Payload `meta` (`$defs/DosMetaPayload`, typed-open):

| Field | Type | Meaning | Unit |
|---|---|---|---|
| `method` | string | `"coop"` or `"cohp"`, conventionally matching the kind. | — |
| `n_spin` | integer, 1 or 2 | Number of spin channels: the length of the spin axis, or 1 without one. | — |
| `fermi_energy_ev` | number | Absolute Fermi level that `energies` are referenced to. | eV |
| `smearing` | number ≥ 0 | Broadening width. | eV |
| `smearing_type` | string | Broadening function, e.g. `"gaussian"`. | — |
| `pairs` | array | One entry per pair: `[i, j]`, or an object `{i, j, label?, symbol_i?, symbol_j?, distance_ang?}`; `i` and `j` are atom indices. | Å (`distance_ang`) |
| `pair_labels` | array of strings | Display labels, one per pair. | — |
| `channels` | array of channel objects | One per pair, naming its two atoms as `i` and `j`. | — |
| `units` | object | Optional declaration, field name or member role → unit name (§ 4.3, *Declared units*): `fermi_energy_ev` and `energies` are `ev`. | — |

#### `fermi_surface`

| Member | Format | Required | Shape and meaning | Unit |
|---|---|:--:|---|---|
| `mesh` | JSON, `$defs/FermiMeshPayload` (strict) | **yes** | The k-point mesh and the cell; table below. | — |
| `energies` | float32 or float64 | **yes** | `[nk1, nk2, nk3, n_bands]`: E(k) − E_F of each band at each grid point. | eV |

Grid point `(i1, i2, i3)` of a uniform mesh lies at the fractional
reciprocal-lattice coordinates `(i1/nk1 + shift[0], i2/nk2 + shift[1],
i3/nk3 + shift[2])`, modulo the reciprocal lattice; without a `shift`, index 0
on every axis is Γ. A producer **SHOULD** write `lattice_vectors` or
`reciprocal_lattice`: without either, a consumer can place the surface only in
fractional coordinates. When both are present they **MUST** agree.

Payload `mesh` (`$defs/FermiMeshPayload`, strict; no field is required):

| Field | Type | Meaning | Unit |
|---|---|---|---|
| `shape` | `[3]` integers ≥ 1 | `[nk1, nk2, nk3]`, the first three axes of `energies`. | — |
| `nk1`, `nk2`, `nk3` | integers ≥ 1 | The same three numbers as separate keys, the form existing producers write; equal to `shape` when both are present. | — |
| `shift` | `[3]` numbers | Position of grid point `(0, 0, 0)`, in fractional reciprocal-lattice coordinates. Default `[0, 0, 0]` (Γ-centred). | — |
| `symmetry_expanded` | boolean | `true` when the full mesh was filled by applying symmetry operations to an irreducible set of computed k-points. | — |
| `kpoints` | array of `[3]` numbers | For a mesh that is not uniform: the fractional coordinates of every grid point, entry `(i1·nk2 + i2)·nk3 + i3` for point `(i1, i2, i3)`. Replaces the uniform formula. | — |
| `fermi_energy_ev` | number | Absolute Fermi level that `energies` are referenced to. | eV |
| `n_spin` | integer, 1 or 2 | Number of spin channels of the underlying calculation. | — |
| `band_indices` | array of integers ≥ 0 | Index, in the producer's full band list, of each band along the last axis of `energies`. | — |
| `lattice_vectors` | `[3, 3]` numbers | Real-space cell, rows, in the convention of `structure.lattice_vectors` (§ 4.3). | Å |
| `reciprocal_lattice` | `[3, 3]` numbers | b1, b2, b3 as rows, including the 2π factor: a_i · b_j = 2π δ_ij. | Å⁻¹ |
| `units` | object | Optional declaration, field name or member role → unit name (§ 4.3, *Declared units*): `fermi_energy_ev` and `energies` are `ev`, `lattice_vectors` is `angstrom`. | — |

#### `phonon_bands`

| Member | Format | Required | Shape and meaning | Unit |
|---|---|:--:|---|---|
| `qpath` | JSON, `$defs/QpathPayload` (strict) | **yes** | The q-point path; table below. | — |
| `frequencies` | float64 | **yes** | `[n_q, n_modes]` phonon frequencies along the path. | cm⁻¹ |

Payload `qpath` (`$defs/QpathPayload`, strict; no field is required) is laid
out like `kpath`: `kpoints`, `labels`, `segments` (segment objects counting
q-points along `frequencies` axis 0), `channels` (channel objects), `title`
and `units` mean what they mean there, without the electronic fields `fermi`,
`fermi_energy_ev`, `spinor` and `n_spin`; a declared `units.frequencies` is
`cm-1`. It adds:

| Field | Type | Meaning |
|---|---|---|
| `n_kpoints` | integer ≥ 1 | Number of q-points along the path, equal to `frequencies` axis 0 (the name is shared with `kpath`). |
| `n_atoms` | integer ≥ 1 | Number of atoms in the cell. |
| `n_modes` | integer ≥ 1 | Number of phonon branches, 3 · `n_atoms`; equal to `frequencies` axis 1. |
| `has_eigenvectors` | boolean | Whether the section carries mode eigenvectors. |

#### `phonon_dos`

| Member | Format | Required | Shape and meaning | Unit |
|---|---|:--:|---|---|
| `meta` | JSON, `$defs/PhononMetaPayload` (typed-open) | **yes** | Counts, broadening and channels; table below. An empty object is valid. | — |
| `frequencies` | float64 | **yes** | `[n_points]` frequency grid. | cm⁻¹ |
| `dos` | float64 | **yes** | `[n_points]` phonon density of states. | states / cm⁻¹ |

Payload `meta` (`$defs/PhononMetaPayload`, typed-open):

| Field | Type | Meaning | Unit |
|---|---|---|---|
| `n_atoms` | integer ≥ 1 | Number of atoms in the cell. | — |
| `n_modes` | integer ≥ 1 | Number of phonon branches, 3 · `n_atoms`. | — |
| `smearing` | number ≥ 0 | Broadening width. | cm⁻¹ |
| `smearing_type` | string | Broadening function, e.g. `"gaussian"`. | — |
| `channels` | array of channel objects | Channels of a projected phonon DOS, one per projection row. | — |
| `units` | object | Optional declaration, field name or member role → unit name (§ 4.3, *Declared units*): `frequencies` is `cm-1`. | — |

#### `equation_of_state`

| Member | Format | Required | Shape and meaning | Unit |
|---|---|:--:|---|---|
| `volumes` | float64 | **yes** | `[n_points]` cell volumes. | Å³ |
| `energies` | float64 | **yes** | `[n_points]` total energies, parallel to `volumes`. | eV |
| `fit` | JSON, `$defs/EosFitPayload` (strict) | **yes** | The fitted model; table below. | — |

Payload `fit` (`$defs/EosFitPayload`, strict):

| Field | Type | Meaning | Unit |
|---|---|---|---|
| `model` | string, **required** | `birch_murnaghan` (third-order Birch–Murnaghan), `murnaghan`, `vinet`, `poirier_tarantola` (third-order Poirier–Tarantola), or `polynomial` (a polynomial E(V), from which the parameters below are derived). | — |
| `V0` | number | Equilibrium volume. | Å³ |
| `E0` | number | Energy at `V0`. | eV |
| `B0` | number | Bulk modulus at `V0`. | GPa |
| `B0_prime` | number | Pressure derivative of the bulk modulus at `V0`. | dimensionless |
| `residual_rms` | number ≥ 0 | Root-mean-square residual of the fit over the points. | eV |
| `n_points` | integer ≥ 0 | Number of points the fit used. | — |
| `pressures_gpa` | array of numbers | Pressure of the fitted model at each volume, parallel to `volumes`. | GPa |
| `energy_unit`, `volume_unit`, `pressure_unit` | string | Informational restatements of eV, Å³ and GPa. | — |

Model-specific values, such as polynomial coefficients, go under `x_` keys.

**DOS section fields.** `dos.total` and `dos.projected` carry their metadata
as optional fields on the section object (§ 4.1), not as members:

| Field | Type | Meaning |
|---|---|---|
| `n_spin` | integer, `1` or `2` | Number of spin channels: `1` spin-unpolarized, `2` spin-polarized. |
| `smearing` | number ≥ 0, eV | Width of the broadening applied to the DOS. |
| `smearing_type` | string | Name of the broadening function, e.g. `"gaussian"`; free text in v1. |
| `fermi_energy_ev` | number, eV | Fermi energy. |
| `n_electrons` | number ≥ 0 | Number of electrons; may be fractional. |
| `channels` | array of objects (`dos.projected` only) | One label object per row of `projections`, in row order. |

A `channels` item may carry `atom` (integer ≥ 0, the 0-based index of the
projected atom in the `structure` section), `l` (integer ≥ 0), `m` (integer),
`spin` (`0` = α/up, `1` = β/down), and `label` (a display string such as
`"O-2p"`). Every item field is optional. An item may carry further fields;
a consumer ignores those it does not understand.

**Phonon eigenvectors and projected phonon DOS.** `phonon_bands` may carry an
optional binary member `eigenvectors`, float64 with shape
`[n_q, n_modes, n_atoms, 3]` when the eigenvectors are real or
`[n_q, n_modes, n_atoms, 3, 2]` when they are complex. The last axis of the
complex form holds `[re, im]`, as for complex `wavefunction.gto`
coefficients. The first two axes index the same q-points and modes as
`frequencies`, and the axis of length 3 holds the Cartesian x, y, z
components. The member stores the eigenvectors as the producer computed
them; v1 does not fix their normalization or mass weighting.

`phonon_dos` may carry an optional binary member `projected`, float64
`[n_channels, n_points]`: row c is the phonon DOS projected on channel c,
sampled on the `n_points` values of `frequencies`. The `meta` member then
**SHOULD** carry a `channels` array with one object per row, in row order.
An item has the shape of a `dos.projected` channel, typically
`{"atom": 0, "label": "Mg"}`, with every field optional.

### 5.7 Analysis kinds

`atom_properties`: per-atom scalar and vector results, one binary member per
role; the role list, units and conventions are in the `atom_properties`
subsection below. `bond_orders`: JSON
`{method, pairs: [{i, j, order, distance_ang?, symbol_i?, symbol_j?}]}`.

`citations`: one binary `references` member of UTF-8 BibTeX bytes. `basis.ao`:
a per-AO grid + `ao_metadata` (quantum numbers + provenance) — see
**Appendix A.3** for how a spherical and a Cartesian component are each named.

The JSON payloads of `bonds` (§ 5.1), `bond_orders`, `scf_history`,
`topology.qtaim` and `structure.symmetry` are specified by the `$defs` entries
named below, at the strictness levels defined in § 5.4. Atom indices are
0-based indices into the `atoms` of the archive's structure; those of `bonds`
address the structure § 5.1 assigns (the structure that holds a `bonds`
member, or the one a `bonds` section's `structure_ref` names, else the first
structure).

#### `bonds`

Payload of a `bonds` section's `bonds` member and of a `structure` section's
optional `bonds` member: `$defs/BondsPayload`, **strict**.

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `pairs` | object[] | **yes** | One entry per bond. | — |
| `pairs[].i`, `pairs[].j` | integer ≥ 0 | **yes** | The two atoms. | — |
| `pairs[].order` | number ≥ 0 | no | Bond order; absent means 1. | — |

#### `bond_orders`

Payload of the `bond_orders` member: `$defs/BondOrdersPayload`, **strict**.

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `method` | string | **yes** | Bond-order definition. Lowercase names are recommended: `"mayer"`, `"wiberg"`, `"nbo"`, … | — |
| `pairs` | object[] | **yes** | One entry per atom pair. | — |
| `pairs[].i`, `pairs[].j` | integer ≥ 0 | **yes** | The two atoms. | — |
| `pairs[].order` | number | **yes** | Bond order under `method`; some definitions yield small negative values. | — |
| `pairs[].distance_ang` | number ≥ 0 | no | Interatomic distance. | Å |
| `pairs[].symbol_i`, `pairs[].symbol_j` | string | no | Element symbols of the two atoms. | — |

#### `scf_history`

Payload of the `iterations` member: `$defs/ScfHistoryPayload`, **strict**.

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `iterations` | object[] | **yes** | One record per SCF iteration, in order. | — |
| `iterations[].iter` | integer ≥ 0 | **yes** | Iteration number, as the producer counts them. | — |
| `iterations[].energy_eh` | number | **yes** | Total energy after the iteration. | Hartree |
| `iterations[].delta_e` | number | no | Energy change from the previous iteration. | Hartree |
| `iterations[].diis_error` | number | no | DIIS error norm. | atomic units |
| `iterations[].rms_density_change` | number | no | Root-mean-square change of the density matrix from the previous iteration. | atomic units |
| `converged` | boolean | no | Whether the SCF met its convergence criteria; absent means not stated. | — |

#### `topology.qtaim`

Payload of the `critical_points` member: `$defs/QtaimPayload`, **permissive**
at the root, with typed points.

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `points` | object[] | **yes** | Critical points of the electron density. | — |
| `points[].type` | string | **yes** | `bcp` (3,−1), `rcp` (3,+1), `ccp` (3,+3) or `ncp` (3,−3). | — |
| `points[].position` | 3 numbers | **yes** | Cartesian position. | `units.position`, default Å |
| `points[].rho` | number | no | Electron density at the point. | not fixed |
| `points[].laplacian` | number | no | Laplacian of the electron density at the point. | not fixed |
| `points[].ellipticity` | number | no | Ellipticity; meaningful for a `bcp`. | dimensionless |
| `points[].atom_pair` | 2 integers ≥ 0 | no | For a `bcp`: the two atoms it connects. | — |
| `points[].atoms` | integer[] ≥ 0 | no | Atoms the point belongs to, e.g. the nucleus of an `ncp` or the ring atoms of an `rcp`. | — |
| `bond_paths` | object[] | no | Gradient paths joining two nuclei through a `bcp`. | — |
| `bond_paths[].atoms` | 2 integers ≥ 0 | no | The two atoms the path joins. | — |
| `bond_paths[].path` | array of 3-vectors | no | Vertices of the path as a polyline. | `units.position`, default Å |
| `units` | object | no | `position` (unit name of a length): unit of every position. Any other key names the unit of the point field of the same name. Values are unit names (§ 4.3). | — |

**Position unit.** Every position — a point's `position` and every vertex of
a `bond_paths[].path` — is Cartesian, in the unit `units.position` declares.
When `units` or `units.position` is absent, the unit is **ångström**
(`"angstrom"`). This default was chosen to match the `structure` section
(§ 5.1, Å) whose atoms the points overlay. It is **provisional**: it is
subject to maintainer decision 2 ("QTAIM position unit") in
`docs/format_review.md`, and a later revision of this section will record
that decision. Producers **SHOULD** therefore declare `units.position`
explicitly, and a consumer **MUST** read positions in a declared unit. `rho`
and `laplacian` have no fixed unit in v1, and the unit names of § 4.3 do not
yet include a unit of electron density or of its Laplacian, so a producer
cannot declare them; a consumer that does not know the producer's
convention presents them as written.

#### `structure.symmetry`

Payload of the `data` member: `$defs/SymmetryPayload`, **permissive**.

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `spacegroup_number` | integer 1–230 | no | Space-group number of the International Tables for Crystallography. | — |
| `international` | string | no | Short international (Hermann–Mauguin) space-group symbol, e.g. `"Fm-3m"`. | — |
| `hall` | string | no | Hall symbol of the space group. | — |
| `point_group` | string | no | Point-group symbol; Schoenflies notation recommended, e.g. `"C2v"`. | — |
| `hermann_mauguin` | string | no | The point group in Hermann–Mauguin notation, e.g. `"mm2"`. | — |
| `operations` | object[] | no | Symmetry operations x′ = R x + t: `rotation` (3 × 3, required), `translation` (three values). | fractional coordinates of the structure's lattice when periodic; Cartesian for a molecule, t zero or absent |
| `wyckoffs` | string[] | no | Wyckoff letter of each atom, in the order of the structure's `atoms`. | — |
| `equivalent_atoms` | integer[] ≥ 0 | no | For each atom, the index of the representative atom of its symmetry-equivalence class. | — |
| `symprec` | number > 0 | no | Distance tolerance of the symmetry search. | Å |
| `space_group_number`, `space_group_symbol` | integer 1–230, string | no | The same quantities as `spacegroup_number` and `international`, under the names some producers write. When both spellings are present they **MUST** agree. | — |

#### `atom_properties`

An `atom_properties` section holds per-atom results as binary members, one per
**role**. Row `i` of every member belongs to atom `i` of the structure the
section describes, and all roles of one section have the same `n_atoms`. A
section carries one or more roles. A **scalar** role is float64 `[n_atoms]`
(`$defs/AtomChargeBinary`); a **vector** role is float64 `[n_atoms, 3]`
(`$defs/AtomVectorBinary`), whose last axis holds the Cartesian components
`x, y, z` in the frame of the structure. The table is the one role list: the
schema, the § 5 table, this table and both reference writers carry the same
names in the same order.

| Role | Shape | Quantity | Unit | Convention |
|---|---|---|---|---|
| `mulliken_charge` | `[n_atoms]` | Mulliken net atomic charge | `e` | Net charge (see below). |
| `loewdin_charge` | `[n_atoms]` | Löwdin net atomic charge | `e` | Net charge. |
| `hirshfeld_charge` | `[n_atoms]` | Hirshfeld (stockholder) net atomic charge | `e` | Net charge. |
| `iao_charge` | `[n_atoms]` | Net atomic charge from the intrinsic atomic orbital (IAO) partition | `e` | Net charge. |
| `spin_population` | `[n_atoms]` | Spin population, α minus β electrons on the atom | electrons | Positive for an excess of α electrons. |
| `cm5_charge` | `[n_atoms]` | Charge Model 5 (CM5) net atomic charge | `e` | Net charge. |
| `chelpg_charge` | `[n_atoms]` | Net atomic charge fitted to the molecular electrostatic potential on the CHELPG grid | `e` | Net charge. |
| `resp_charge` | `[n_atoms]` | Net atomic charge from the restrained electrostatic potential (RESP) fit | `e` | Net charge. |
| `esp_charge` | `[n_atoms]` | Net atomic charge from an unrestrained fit to the electrostatic potential | `e` | Net charge. |
| `nbo_charge` | `[n_atoms]` | Natural population analysis (NPA) net atomic charge of the NBO analysis | `e` | Net charge. |
| `mbis_charge` | `[n_atoms]` | Minimal basis iterative stockholder (MBIS) net atomic charge | `e` | Net charge. |
| `bader_charge` | `[n_atoms]` | Net atomic charge from integrating the electron density over the QTAIM (Bader) basin | `e` | Net charge. |
| `ddec_charge` | `[n_atoms]` | Density-derived electrostatic and chemical (DDEC) net atomic charge | `e` | Net charge. |
| `valence` | `[n_atoms]` | Total valence of the atom | dimensionless | Under the producer's bond-order scheme; `valence − bonded_valence` is the free valence. |
| `bonded_valence` | `[n_atoms]` | Bonded valence: the sum of the bond orders between the atom and every other atom | dimensionless | Under the producer's bond-order scheme. |
| `fukui_plus` | `[n_atoms]` | Condensed Fukui function f⁺ (nucleophilic attack) | dimensionless | f⁺ = p(N+1) − p(N) (see below). |
| `fukui_minus` | `[n_atoms]` | Condensed Fukui function f⁻ (electrophilic attack) | dimensionless | f⁻ = p(N) − p(N−1). |
| `fukui_zero` | `[n_atoms]` | Condensed Fukui function f⁰ (radical attack) | dimensionless | f⁰ = (f⁺ + f⁻) / 2. |
| `dual_descriptor` | `[n_atoms]` | Condensed dual descriptor Δf | dimensionless | Δf = `fukui_plus` − `fukui_minus`. |
| `atomic_energy_eh` | `[n_atoms]` | Energy assigned to the atom by the producer's energy partition | `hartree` | The partition scheme is the producer's. |
| `atomic_volume_bohr3` | `[n_atoms]` | Volume assigned to the atom by the producer's partition | `bohr3` | The partition scheme is the producer's. |
| `nmr_shielding_iso_ppm` | `[n_atoms]` | Isotropic NMR shielding σ of the atom's nucleus | `ppm` | A shielding, not a chemical shift: no reference compound is subtracted. |
| `magnetic_moment` | `[n_atoms, 3]` | Atomic magnetic moment, Cartesian components | μB (Bohr magneton) | A collinear producer puts the moment on the quantisation axis (see below). |
| `force_eh_bohr` | `[n_atoms, 3]` | Energy gradient on the atom, ∂E/∂R, Cartesian components | `hartree` / `bohr` | A gradient, not a force: force = −gradient (see below). |

**Charges.** Every `*_charge` role is a net atomic charge in units of the
elementary charge `e`: the atom's nuclear charge (its effective core charge
where an effective core potential replaces core electrons) minus its electron
population, so a positive value means electron loss. The role name is the
partition scheme. The schemes give different numbers for the same density,
and a consumer **MUST NOT** treat two charge roles as interchangeable.
`spin_population` counts electrons, α minus β, and is positive for an excess
of α electrons.

**Fukui functions.** The condensed Fukui functions are differences of atomic
electron populations. With p_A(M) the electron population of atom A in the
system of M electrons and N the electron count of the system, f⁺_A =
p_A(N+1) − p_A(N), f⁻_A = p_A(N) − p_A(N−1), f⁰_A = (f⁺_A + f⁻_A) / 2, and the
dual descriptor Δf_A = f⁺_A − f⁻_A. All are dimensionless. A producer that
writes `dual_descriptor` together with `fukui_plus` and `fukui_minus` **SHOULD**
write values that satisfy the identity to numerical precision. The population
scheme behind the populations is the producer's and is not named by the role.

**`magnetic_moment`.** The three components are the Cartesian components of
the atomic moment in μB, in the frame of the structure. A collinear
calculation has no spatial spin direction: its producer puts the signed
moment on the quantisation axis, `[0, 0, m]` for the usual z axis, with the
sign convention of `spin_population` (positive for an excess of α electrons).
A non-collinear producer writes the full vector.

**`force_eh_bohr`.** The row of atom `i` is the gradient ∂E/∂R_i of the
energy with respect to the position of atom `i`, in hartree/bohr, the sign
convention of the `gradient` member of the `forces` kind (force = −gradient),
despite the role's name. The role lets a section show per-atom gradients next
to charges; it is a display alias. When a `forces` section for the same
structure is present and its gradient differs from this role, the `forces`
section is authoritative.

**Custom roles.** A producer may add a quantity the list does not provide as
a role named `custom.<name>`, where `<name>` is one or more characters of
`A–Z`, `a–z`, `0–9` and `_`. A custom role is a scalar (float64 `[n_atoms]`) or
a vector (float64 `[n_atoms, 3]`) member, like the listed roles. A section that
has a `custom.*` member **MUST** also have a JSON member `meta`
(`$defs/AtomPropertiesMetaPayload`, **strict**) with one entry per custom role,
keyed by the role name. A validator reports `E_MEMBER_MALFORMED` for a
`custom.*` member that `meta` does not describe, including when the section
has no `meta` member (§ 6). Entries for roles the section does not carry are
ignored. A consumer that does not know a custom role presents its
`description` and `units`, or ignores it. Use a listed role when one fits, and
a vendor kind (§ 7.3) for data that is not per-atom.

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `custom.<name>` | object | **yes**, one per custom role | Description of the role of that name. | — |
| `custom.<name>.units` | string | **yes** | Unit of the values, spelled as in § 4.3 where it gives a spelling (for example `e`, `bohr3`, `hartree`, `ppm`); `dimensionless` for a pure number. | — |
| `custom.<name>.description` | string | **yes** | What the quantity is, with any sign convention. | — |
| `custom.<name>.kind` | `scalar` or `vector` | **yes** | `scalar` for a float64 `[n_atoms]` member, `vector` for a float64 `[n_atoms, 3]` member. It **MUST** agree with the member's `shape`. | — |

```jsonc
{
  "custom.born_charge_mean": {
    "units": "e",
    "description": "Trace of the Born effective-charge tensor divided by three.",
    "kind": "scalar"
  }
}
```

### 5.8 `run.record` — self-contained run records

A `run.record` section carries the verbatim artefacts of **one program
invocation**: the input handed to a quantum-chemistry code, the full log/output
it produced, and any auxiliary files. With it, a QVF archive is a
*self-contained record of a calculation* — input, results, and log travel in
one file.

Section fields (peers of `members`, like `volume.difference`'s operands):

| Field | Required | Meaning |
|---|:--:|---|
| `program` | **yes** | The code the input/log belong to (`"vibe-qc"`, `"orca"`, …). Lowercase identifiers recommended. |
| `program_version` | no | Version string of that code. |
| `command` | no | Launch command line, as one string. |
| `exit_status` | no | Integer process exit status. |
| `started_utc` / `finished_utc` | no | ISO 8601 UTC timestamps. |
| `sequence` | no | Ordering index (≥ 0) when an archive carries multiple run records (multi-step workflows: opt → freq → …). |
| `job_ref` | no | Id of the `job.spec` section (§ 5.9) this run executed. A validator checks that it names a `job.spec` section (§ 6). |
| `run_status` | no | Outcome of this run: `"running"`, `"converged"` or `"failed"`, with the meanings of the root `provenance.run_status` (§ 3.2). |

`job_ref` and `run_status` make the runs of a job set (§ 5.9) unambiguous: a
job is pending until a `run.record` names it, and that run's `run_status` is
the job's. In an archive with one `job.spec`, a run record without `job_ref`
executed that job, as before.

`program` is deliberately distinct from the root `source.program` (§ 3.1): the
root identifies the **QVF producer**, which may be a converter or packing tool;
`run.record.program` identifies the code that **actually ran**. For a code
writing its own archives the two coincide.

Members — at least one of `input` / `log` is REQUIRED:

* **`input`** — the verbatim main input, opaque binary bytes that **MUST**
  decode as UTF-8 text. Producers converting from another encoding transcode at
  write time.
* **`log`** — the full log/output of the program, opaque UTF-8 binary bytes.
  A producer that finalizes the archive before its own process ends **SHOULD**
  embed the log as complete as it can be at write time and note any truncation
  in the `files` index (`"truncated": true` on the `log` entry); a packing tool
  working from finished files embeds them whole.
* **`files`** — optional JSON index describing the stored members, schema'd
  as `$defs/RunRecordFilesPayload`: a map from member role to an entry that
  records the member's original file. Every key **MUST** name a member that
  exists in the section (`input`, `log`, or an `attachment.<suffix>`), so the
  index has no vendor keys of its own. An entry is strict: only the fields
  below and vendor keys beginning `x_`.

  | Field | Type | Meaning |
  |---|---|---|
  | `filename` | string | Original file name, without directories. |
  | `description` | string | What the file is, for display. |
  | `media_type` | string | IANA media type of the bytes, e.g. `text/plain` or `chemical/x-xyz`. |
  | `truncated` | boolean | `true` when the stored bytes are not the complete file, e.g. a log embedded before the run ended. |

  No field of an entry is required; producers **SHOULD** give `filename`
  whenever the member was read from a file.
* **`attachment.<suffix>`** — optional auxiliary files (structure includes,
  restart data, …), opaque binary bytes with **no** encoding requirement.
  `<suffix>` matches `[A-Za-z0-9_.-]+`. Describe each in the `files` index.

Producers **SHOULD NOT** duplicate large binary result files (wavefunctions,
volumetric grids) as attachments when a canonical section kind exists for the
data — attachments are for run artefacts, not a bypass of § 5.

A consumer renders `input` and `log` as plain text. Logs can reach tens of MB;
consumers **SHOULD** size-gate rendering (the uncompressed size is available
from the ZIP central directory without reading the member, and from the
member's `size`, § 4.2) and offer the full text on demand.

#### Sharing and sanitisation

A `run.record` embeds its artefacts verbatim, and archives are meant to be
shared. Inputs and logs commonly contain absolute paths, user names, host
names, and scratch-directory locations; `attachment.*` members may contain the
same (in any encoding), and so may the `command` field and the original
filenames recorded in the `files` index. The format does not redact anything:
sanitising such content before an archive is shared is the responsibility of
the producer or of the user. Producers **SHOULD** offer a way to suppress or
redact it — for example, an option to omit the run record or individual
members, or to rewrite paths at write time. The intended mechanism in the
reference toolkit is a `scrub` operation that rewrites such content in an
existing archive and reports each change for review (informative: the
operation is not yet part of the toolkit, and it changes nothing in the
format). Root `provenance.hostname` is deprecated for the same reason (§ 3.2).

### 5.9 `job.spec` — declarative job specification

A `job.spec` section describes the calculation an archive *requests* —
before, and independent of, any results it carries. Together with a
`structure` section (§ 5.1) it makes a QVF archive an executable input: a
runner opens the container, reconstructs the job from the declarative
payload, executes it, and writes the results — and a `run.record` (§ 5.8) —
back into the same container. An archive whose `provenance.run_status`
(§ 3.2) is `"pending"` carries at least one `job.spec` section and the
`structure` sections they run on.

Members:

* **`spec`** — REQUIRED JSON member, schema'd as `$defs/JobSpecPayload`.

The payload:

| Field | Required | Meaning |
|---|:--:|---|
| `job_type` | **yes** | `"molecular"` or `"periodic"` — which class of engine runs the job. A periodic job's geometry carries `pbc`/`lattice_vectors` in the `structure` section; `job_type` selects the runner, it does not restate periodicity. |
| `method` | no | Method identifier (e.g. `"rhf"`, `"rks"`, `"ccsd(t)"`, `"gfn2"`). Producer-neutral lowercase string recommended. Absent means the engine's default. |
| `basis` | no | Basis-set name (e.g. `"def2-svp"`), or `null`: the method takes no basis set (a semiempirical or tight-binding method such as GFN2-xTB). Absent means the engine's default. |
| `functional` | no | Exchange-correlation functional name, for DFT methods, or `null`: the method takes no functional (Hartree–Fock, a correlated wavefunction method, a semiempirical method). Absent means the engine's default. |
| `charge` | no | Net charge, integer. Absent means `0`. Authoritative for the job — a consumer never infers charge from the structure. |
| `multiplicity` | no | Spin multiplicity 2S+1, integer ≥ 1. Absent means `1`. |
| `kpoints` | no | `[n1, n2, n3]` Monkhorst–Pack mesh (integers ≥ 1). Periodic jobs only. |
| `tasks` | no | Ordered list of task identifiers, e.g. `["single_point"]`, `["optimize", "hessian"]`. Absent means a single point. |
| `structure_ref` | no | String: id of the `structure` section (§ 5.1) the job runs on. A validator resolves it to a section of kind `structure` (§ 6). Absent means the archive's only `structure` section; with several `structure` sections and no `structure_ref` the job is ambiguous, and a runner **MUST** refuse it rather than guess. A `structure_ref` on the `job.spec` section object takes precedence over this one (§ 4.1); keep the two equal. |
| `solvation` | no | Object: implicit solvation, below. Absent means none. |
| `dispersion` | no | String: the dispersion correction by its conventional lowercase name, e.g. `"d3bj"`, `"d4"`. Absent means none beyond what the method implies. |
| `relativistic` | no | String: the scalar-relativistic treatment by its conventional lowercase name, e.g. `"zora"`, `"x2c"`, `"dkh2"`. Absent means none. |
| `scf` | no | Object: SCF settings, below. Absent means the engine's defaults. |
| `geometry` | no | Object: geometry-optimisation settings, below. Absent means the engine's defaults. |
| `resources` | no | Object: the resources the job would like, below. Hints, not requirements. |
| `options` | no | Object of additional engine-specific keywords (JSON-safe values). A runner that does not recognize an option **MUST** refuse the job rather than silently drop the option. |

**`null` and absent differ.** `null` states that the method takes no such
input, and a runner **MUST NOT** supply one; an absent key leaves the choice
to the engine, as for `method`. The empty string `""` is a **deprecated**
spelling of `null`, which earlier producers wrote for a semiempirical job's
`basis`: it remains valid, a consumer reads it as `null`, and producers
**SHOULD NOT** write it.

The payload root is open (`additionalProperties: true`) for forward growth,
but producers **SHOULD** put engine-specific keywords under `options`,
keeping the typed top level portable across codes. The four objects below are
strict, each a `$defs` entry of the schema: they carry only the fields listed
and vendor keys beginning `x_`, so an engine keyword has no place in them and
goes under `options`. Every field of them is optional, and an absent field
means the engine's default.

`solvation` (`$defs/JobSpecSolvation`):

| Field | Type | Required | Meaning |
|---|---|:--:|---|
| `model` | string, non-empty | **yes** | The solvation model by its conventional lowercase name, e.g. `"pcm"`, `"cpcm"`, `"smd"`, `"cosmo"`. |
| `solvent` | string, non-empty | no | The solvent by name, e.g. `"water"`. |
| `epsilon` | number > 0 | no | Static dielectric constant, dimensionless. When given it is the constant the model uses, and `solvent` then only labels it. |

`scf` (`$defs/JobSpecScf`):

| Field | Type | Meaning | Unit |
|---|---|---|---|
| `convergence` | number > 0 | Convergence threshold of the SCF energy change. An engine that converges on another measure maps it to its nearest criterion. | hartree |
| `max_iterations` | integer ≥ 1 | Maximum number of SCF iterations. | — |
| `guess` | string, non-empty | The initial guess by its conventional lowercase name, e.g. `"core"`, `"sad"`, `"huckel"`. | — |
| `damping` | number | Damping factor of the density or Fock update. | — |

`geometry` (`$defs/JobSpecGeometry`):

| Field | Type | Meaning | Unit |
|---|---|---|---|
| `convergence` | number > 0 | Convergence threshold of the optimisation: the largest Cartesian gradient component. An engine with several criteria applies it to its gradient criterion and keeps its defaults for the others. | hartree/bohr |
| `max_steps` | integer ≥ 1 | Maximum number of optimisation steps. | — |
| `coordinates` | `"cartesian"`, `"internal"` or `"redundant"` | The coordinate system the optimiser steps in. | — |

`resources` (`$defs/JobSpecResources`) are **hints, not requirements**: a
runner uses what it has and may use less, and a run that used other resources
has not failed to honour the spec. What a run actually used is recorded in `provenance` (`n_threads`, `n_mpi_ranks`,
`n_gpus`, `wall_seconds`; § 3.2).

| Field | Type | Meaning | Unit |
|---|---|---|---|
| `n_cores` | integer ≥ 1 | Number of CPU cores to use. | — |
| `memory_mb` | integer ≥ 1 | Memory to use. | megabytes |
| `walltime_s` | number > 0 | Wall-clock time to allow. | s |

A periodic job on a given structure:

```json
{
  "job_type": "periodic",
  "method": "rks", "functional": "pbe", "basis": "pob-tzvp-rev2",
  "structure_ref": "structure",
  "kpoints": [4, 4, 1],
  "tasks": ["optimize"],
  "dispersion": "d3bj",
  "scf": {"convergence": 1e-8, "max_iterations": 100, "guess": "sad"},
  "geometry": {"convergence": 4.5e-4, "max_steps": 50, "coordinates": "redundant"},
  "resources": {"n_cores": 16, "memory_mb": 32000, "walltime_s": 86400}
}
```

A `job.spec` is declarative data, never code. Executing it involves no
evaluation of an embedded program — deliberately distinct from
`run.record.input`, which may be an arbitrary script and is a *record* of
what ran, not a request. A runner **MUST NOT** execute `run.record.input` as
a side effect of opening or running an archive.

Lifecycle: a runner that executes a pending archive updates the same
container (§ 2.3) — result sections are added, a `run.record` is embedded, and
`provenance.run_status` moves to `"converged"` or `"failed"`. The runner also
records what ran, with which resources and when, in `provenance` (`software`,
`git_commit`, `n_threads`, `n_mpi_ranks`, `n_gpus`, `started_utc`,
`finished_utc`; § 3.2). The `job.spec` section is preserved verbatim through
this update, so a settled archive still records what was asked. Multi-step
workflows append further `run.record` sections ordered by their `sequence`
field.

#### Job sets

An archive **MAY** carry several `job.spec` sections, a **job set**: for
example one calculation for each species of a reaction scheme (§ 5.14), or the
same species at several levels of theory. Each `job.spec` section is one job,
independent of the others.

* **Each job names its structure.** In an archive with more than one
  `structure` section, a job without a `structure_ref` is ambiguous, and a
  runner **MUST** refuse it, as above. The section-level `structure_ref`
  (§ 4.1) is the preferred carrier, because a consumer reads it from the
  manifest without opening the payload; a `structure_ref` inside the `spec`
  payload remains valid, the section's takes precedence, and producers
  **SHOULD** keep the two equal.
* **Runs name their job.** A runner executes any subset of the jobs, in any
  order or in parallel. For each job it runs it adds a `run.record` whose
  `job_ref` names the job's section and whose `run_status` gives the job's
  outcome (§ 5.8), and the result sections of that run **SHOULD** carry
  `run_ref` naming the run record and `structure_ref` naming the structure
  (§ 4.1), so that a reader attributes every result to its job. A job that no
  `run.record` names has not run.
* **The set has one status.** `provenance.run_status` summarises the set:
  `pending`, `running`, then `converged` or `failed` (§ 3.2).
* **One refusal does not stop the set.** A runner that refuses or fails one
  job still runs the others it was asked to run, and records the failed one
  with `run_status: "failed"`.

Jobs of a set do not depend on each other. A sequence within one job, such as
an optimisation followed by a frequency calculation on the optimised
geometry, is that job's `tasks` list.

A pending job set with two species, each job naming its structure on the
section object:

```json
"provenance": {"run_status": "pending"},
"sections": [
  {"id": "structure_s1", "kind": "structure", "label": "1", "charge": 0,
   "multiplicity": 1, "members": {"structure": {"path": "…", "format": "json", "sha256": "…"}}},
  {"id": "structure_s2", "kind": "structure", "label": "2", "charge": 0,
   "multiplicity": 2, "members": {"structure": {"path": "…", "format": "json", "sha256": "…"}}},
  {"id": "job_s1_opt", "kind": "job.spec", "structure_ref": "structure_s1",
   "members": {"spec": {"path": "…", "format": "json", "sha256": "…"}}},
  {"id": "job_s2_opt", "kind": "job.spec", "structure_ref": "structure_s2",
   "members": {"spec": {"path": "…", "format": "json", "sha256": "…"}}}
]
```

### 5.10 `forces` — nuclear gradient and stress

A `forces` section carries the first derivatives of the energy at one
geometry: the derivatives with respect to the nuclear positions and, for a
periodic system, the stress. The stored quantity is the **energy gradient**,
not the force: `gradient[a][c]` is ∂E/∂R, the derivative of the energy with
respect to Cartesian coordinate c (0 = x, 1 = y, 2 = z) of atom a. The force
on atom a is the negative of row a, F = −∇E. A consumer that draws forces
**MUST** negate the stored values.

The rows follow the atoms of one `structure` section (§ 5.1), in the order of
its `atoms`, and the Cartesian axes are that structure's. The structure is
the one the section-level `structure_ref` names when the section carries one
(§ 4.1), else the one `meta.structure_ref` names; when neither is given, it
is the archive's `structure` section if the archive has exactly one.
Producers **SHOULD** name the structure whenever the archive has more than
one.

| Member | Format | Required | Shape and meaning | Unit |
|---|---|:--:|---|---|
| `gradient` | float64 | **yes** | `[n_atoms, 3]` energy gradient ∂E/∂R; row a belongs to atom a of the structure. | hartree / bohr |
| `stress` | float64 | no | `[3, 3]` stress tensor of a periodic system, σ_ij = (1/V) ∂E/∂ε_ij for the strain ε and the cell volume V, in the Cartesian axes of the structure. It is symmetric, and the pressure is p = −(σ_xx + σ_yy + σ_zz)/3, so a compressed cell has a negative trace. | hartree / bohr³ |
| `meta` | JSON, `$defs/ForcesMetaPayload` (strict) | no | Table below. | — |

Payload `meta` (`$defs/ForcesMetaPayload`, strict; no field is required):

| Field | Type | Meaning |
|---|---|---|
| `structure_ref` | string | `id` of the `structure` section the rows belong to. The section-level `structure_ref` takes precedence when both are present. |
| `method` | string | Method of the energy that was differentiated, e.g. `"rks"`. |
| `convention` | `"gradient"` | What `gradient` holds. `"gradient"`, the energy gradient, is the only value in v1; absent means `"gradient"`. |
| `units` | object (`$defs/DerivativeUnits`) | Declared units: `energy` (`"hartree"`) and `length` (`"bohr"`). v1 fixes both, so a declaration restates them and cannot name another unit. |

A validator checks that `gradient` is `[n_atoms, 3]` and `stress` `[3, 3]`;
when it can resolve the structure, it checks that `n_atoms` is that
structure's atom count, and a `meta.structure_ref` that names no `structure`
section is an unresolved reference (§ 6).

### 5.11 `hessian` — Cartesian second derivatives

A `hessian` section carries the matrix of second derivatives of the energy
with respect to the nuclear Cartesian coordinates at one geometry, from which
harmonic frequencies and normal modes follow; a `vibrations` section names
the Hessian its modes come from through `hessian_ref` (§ 5.5).

`matrix[i][j]` is ∂²E/∂x_i∂x_j, where index i = 3a + c belongs to Cartesian
coordinate c (0 = x, 1 = y, 2 = z) of atom a: the order is x₀, y₀, z₀, x₁,
y₁, z₁, … for N atoms. The atoms and axes are those of the section's
structure, resolved as for `forces` (§ 5.10). The matrix is symmetric; a
producer whose Hessian is not exactly symmetric, as a finite-difference one
is, **SHOULD** write (H + Hᵀ)/2.

| Member | Format | Required | Shape and meaning | Unit |
|---|---|:--:|---|---|
| `matrix` | float64 | **yes** | `[3N, 3N]`, as above. | hartree / bohr²; hartree / (bohr² · amu) when `meta.mass_weighted` is true |
| `meta` | JSON, `$defs/HessianMetaPayload` (strict) | no | Table below. | — |

Payload `meta` (`$defs/HessianMetaPayload`, strict; no field is required, but
a mass-weighted matrix requires `atom_masses_amu`):

| Field | Type | Meaning | Unit |
|---|---|---|---|
| `structure_ref` | string | `id` of the `structure` section whose atoms index the matrix, as for `forces`. | — |
| `mass_weighted` | boolean | `true`: the matrix is mass-weighted, element (i, j) divided by √(m_a m_b) for the atoms a and b of i and j, with the masses of `atom_masses_amu`, which is then **required**. Absent means `false`. | — |
| `projected` | boolean | `true`: the rigid-body motions have been projected out of the matrix (translations and, for a non-periodic system, rotations), so their eigenvalues are zero. Absent means `false`: the plain second derivatives. | — |
| `atom_masses_amu` | array of numbers > 0 | Mass of each atom in atom order, N entries: the masses of the mass weighting, or the masses frequencies derived from the matrix are to use. | amu (unified atomic mass unit) |
| `units` | object (`$defs/DerivativeUnits`) | Declared units, as for `forces`: `energy` (`"hartree"`) and `length` (`"bohr"`). | — |

A validator checks that `matrix` is square with an extent divisible by 3,
that `atom_masses_amu` has N entries, and, when it can resolve the structure,
that N is that structure's atom count (§ 6).

### 5.12 `excited_states` — excited states and transition properties

An `excited_states` section carries the electronically excited states of one
calculation, from TD-DFT, CIS, ADC, EOM-CC or any other method that yields a
table of states: their energies, transition properties and dominant orbital
excitations. Every state is a vertical excitation from one **reference
state**, usually the ground state that the archive's `wavefunction.gto` and
`provenance` describe. The section has one member:

| Member | Format | Required | Meaning |
|---|---|:--:|---|
| `states` | JSON, `$defs/ExcitedStatesPayload` (strict) | **yes** | The states and a description of the calculation (tables below). |

Besides the generic fields of § 4.1 (its `structure_ref` names the structure
whose axes the transition dipoles use), the section may carry one field of its
own, a peer of `members`:

| Section field | Type | Meaning |
|---|---|---|
| `wavefunction_ref` | string | `id` of the `wavefunction.gto` section of the reference state. The `from_mo` and `to_mo` of every contribution are rows of its coefficient member. |

Payload `states` (`$defs/ExcitedStatesPayload`, strict):

| Field | Type | Required | Meaning |
|---|---|:--:|---|
| `states` | array of states (below) | **yes** | The states, in ascending order of `index`. |
| `meta` | object (below) | no | The calculation the states come from. |

Each state (`$defs/ExcitedState`, strict):

| Field | Type | Required | Meaning | Unit |
|---|---|:--:|---|---|
| `index` | integer ≥ 1 | **yes** | Number of the state, unique within the payload, counted from 1 in the producer's order (normally by increasing energy); the reference state has no entry. Spectrum peaks (§ 5.4), natural transition orbitals (§ 5.3) and transition densities (§ 5.2) name a state by this number. | — |
| `excitation_energy` | `{value, units}` | **yes** | Vertical excitation energy: the energy of the state minus that of the reference state. | producers **SHOULD** write `ev` |
| `total_energy` | `{value, units}` | no | Total energy of the state. | producers **SHOULD** write `hartree` |
| `multiplicity` | integer ≥ 1 | no | Spin multiplicity 2S + 1. | — |
| `symmetry` | string | no | Irreducible representation of the state, e.g. `"B1"`. | — |
| `oscillator_strength` | number ≥ 0 | no | Oscillator strength f of the transition from the reference state. | dimensionless |
| `rotatory_strength_cgs` | number | no | Rotatory strength R of the transition, signed. | 10⁻⁴⁰ esu²·cm² |
| `transition_dipole` | 3 numbers | no | Electric transition dipole moment ⟨0\|μ̂\|n⟩. | e·bohr (atomic units) |
| `transition_magnetic_dipole` | 3 numbers | no | Magnetic transition dipole moment ⟨0\|m̂\|n⟩. | e·ħ/mₑ (atomic units, = 2 μ_B) |
| `contributions` | array (below) | no | The dominant orbital excitations of the state. | — |
| `nto_ref` | string | no | `id` of the `wavefunction.gto` section of the state's hole natural transition orbitals (§ 5.3, *Natural transition orbitals*). | — |
| `density_ref` | string | no | `id` of a `volume.transition_density` section holding the transition density of the state (§ 5.2), or of a `volume.difference` section holding its difference density, the density of the state minus that of the reference state. | — |

The two dipoles are Cartesian vectors in the axes of the section's structure.
The sign of a transition dipole is arbitrary, as is the phase of the state;
f and R are not. In the length gauge f = (2/3) ΔE |μ|², with the excitation
energy ΔE in hartree and the transition dipole μ in atomic units. The
rotatory strength unit is the one quantum-chemistry programs print as
"10⁻⁴⁰ cgs"; the vocabulary of § 4.3 has no name for it, nor for the atomic
units of the dipoles, so this table fixes the units of these three fields and
`meta.units` does not declare them.

Each contribution (`$defs/ExcitedStateContribution`, strict):

| Field | Type | Required | Meaning |
|---|---|:--:|---|
| `from_mo` | integer ≥ 0 | **yes** | The orbital the electron leaves: a 0-based row of the coefficient member of the reference wavefunction. |
| `to_mo` | integer ≥ 0 | **yes** | The orbital the electron enters, indexed the same way. |
| `spin` | `"alpha"` or `"beta"` | no | For an unrestricted reference, the spin channel: `alpha` rows index `mo_coefficients_alpha`, `beta` rows `mo_coefficients_beta`. Absent for a reference with one set of orbitals. |
| `weight` | number | **yes** | Weight of the excitation in the state: the squared amplitude X² (CIS, TDA) or X² − Y² (full TD-DFT, RPA), normalized so that the weights of all excitations of the state sum to 1. A list of the dominant ones sums to a little less. |

Without `wavefunction_ref` the orbital rows refer to the reference calculation
as the producer numbers its orbitals, which a consumer cannot check; producers
**SHOULD** set `wavefunction_ref` whenever they write contributions, and give
`spin` for every contribution of an unrestricted reference. A contribution
without `spin` in an unrestricted reference indexes a row that both
coefficient members have.

Payload `meta` (`$defs/ExcitedStatesMeta`, strict; no field is required):

| Field | Type | Meaning |
|---|---|---|
| `method` | string | The excited-state method, e.g. `"tddft"`, `"cis"`, `"adc2"`, `"eom-ccsd"`. |
| `reference_state` | string | Label of the reference state, e.g. `"S0"`. |
| `n_states` | integer ≥ 0 | Number of states; when present it **MUST** equal the length of `states`. |
| `tda` | boolean | `true`: the Tamm–Dancoff approximation was used. |
| `spin_flip` | boolean | `true`: the states come from a spin-flip calculation. |
| `units` | object | Declared units (§ 4.3), keyed by a field of the states, each value a unit name: `excitation_energy` and `total_energy` restate the `units` of those objects of every state, and `oscillator_strength` is `"dimensionless"`. The dipoles and `rotatory_strength_cgs` are not declared (above). |

**Linked sections.** The table of states is the hub of an excited-state
calculation:

* `wavefunction_ref` names the reference `wavefunction.gto` (§ 5.3).
* A state's `nto_ref` names the `wavefunction.gto` section of its hole NTOs;
  that section and its particle partner carry this section's `id` as
  `state_ref` and the state's `index` as `state_index` (§ 5.3).
* A state's `density_ref` names its `volume.transition_density`, which
  carries this section's `id` as `excited_states_ref` and the state's `index`
  as `state_index` (§ 5.2), or a `volume.difference`.
* A one-dimensional spectrum names this section with `states_ref`, and its
  peaks name states by `assignment.state_index` (§ 5.4). A stick UV-Vis
  spectrum derived from the table has one line per state at its excitation
  energy, with the oscillator strength as intensity (`y_units`
  `"dimensionless"`); an ECD stick spectrum uses the rotatory strengths.

A validator checks that the `index` values are unique and ascending, that
`meta.n_states`, when present, equals the number of states, that
`wavefunction_ref` and every `nto_ref` name `wavefunction.gto` sections and
every `density_ref` a `volume.transition_density` or `volume.difference`
section, and, when `wavefunction_ref` names a `wavefunction.gto` section,
that every `from_mo` and `to_mo` is a row of its coefficient member (§ 6).

### 5.13 Property kinds

Three kinds carry properties of the system as a single JSON payload each:
`properties.electric` (electric multipole moments and polarizabilities),
`properties.elastic` (the elastic constants of a periodic system) and
`properties.dielectric` (the dielectric tensors, Born charges and
piezoelectric tensor of a periodic system). A section of one of these kinds has
exactly one member, **`properties`**, a JSON member whose payload is **strict**
(§ 5): the keys of its table, and keys that begin with `x_`, are allowed, and
the same holds for every object inside it. No field is required unless its
table says so; a field that is absent was not computed.

Conventions common to the three kinds:

* **Axes.** Cartesian components `x, y, z` are those of the `structure` the
  section describes: the one the section-level `structure_ref` names when the
  section carries one (§ 4.1), else the archive's `structure` section if it has
  exactly one. Producers **SHOULD** name the structure whenever the archive has
  more than one.
* **Matrices** are nested arrays of rows: entry `[i][j]` is row `i`, column
  `j`, with the Cartesian order `x, y, z` for a 3 × 3 matrix. In **Voigt
  notation** the six indices are, in this order, `xx, yy, zz, yz, xz, xy`.
* **Units are fixed**; the tables give them (§ 4.3). The optional `units`
  object of each payload maps a field name to a unit name and may only restate
  the unit that is fixed, as for the other payloads (§ 4.3, *Declared units*).
  It can name only the fields whose unit is in the vocabulary: a compound unit
  such as debye · Å or GPa⁻¹ has no name there (§ 9), so those fields are fixed
  by their table alone.
* **One section, one state.** A section holds the properties of one geometry
  and, for an electric field response, one frequency. A producer with several
  (a static and a dynamic polarizability, say) writes several sections with
  distinct ids and **MAY** tie them together with `group` (§ 4.1).

#### `properties.electric`

Payload `properties` (`$defs/ElectricPropertiesPayload`, strict; at least one
of `dipole`, `quadrupole`, `octupole`, `polarizability` and
`hyperpolarizability` is required):

| Field | Type | Meaning | Unit |
|---|---|---|---|
| `origin` | string or `[3]` numbers | Reference point of the multipole moments, with the meaning of `dipole_moment.origin` (§ 3.3): a conventional name (`"center_of_mass"`, `"center_of_nuclear_charge"`, `"origin"`) or an explicit 3-vector. | bohr, for a vector |
| `dipole` | object | `vector` (`[3]` numbers, **required**): the dipole components; `total` (number ≥ 0): its magnitude; `units`: `"debye"`. | debye |
| `quadrupole` | object | `matrix` (3 × 3, symmetric), `traceless` (boolean) and `convention` (`"buckingham"` or `"cartesian"`), **all required**; defined below. | debye · Å |
| `octupole` | object | `tensor` (27 numbers, row-major `[3, 3, 3]`, **required**): element `(a, b, c)` is at index `9a + 3b + c`; `convention` (`"buckingham"` or `"cartesian"`; absent means `"cartesian"`). | debye · Å² |
| `polarizability` | object | `matrix` (3 × 3, **required**): α_ab; `frequency_ev` (number ≥ 0; `0`, the default, is the static polarizability); `isotropic` (number): ⅓ tr α. | atomic units, e² a₀² / E_h; `frequency_ev` in eV |
| `hyperpolarizability` | object | `tensor` (27 numbers, row-major `[3, 3, 3]`, **required**): β_abc; `frequencies_ev` (`[2]` numbers, default `[0, 0]`): the photon energies ω₁, ω₂ of β(−(ω₁ + ω₂); ω₁, ω₂). | atomic units, e³ a₀³ / E_h²; `frequencies_ev` in eV |
| `units` | object | Declared units: `dipole` is `debye`, `origin` is `bohr`. | — |

With the charges q_i of the nuclei and of the electrons (the electron's is −e)
at the positions r_i, taken relative to `origin`, the multipole moments are:

* the dipole μ_a = Σ q_i r_a;
* the **Buckingham** quadrupole Θ_ab = ½ Σ q_i (3 r_a r_b − r² δ_ab), which is
  traceless, and the **Cartesian** quadrupole Q_ab = Σ q_i r_a r_b, whose trace
  is Σ q_i r². `traceless` says whether the stored matrix has zero trace: it is
  always `true` for `"buckingham"`, and for `"cartesian"` it says that the
  trace has been removed, Q_ab − ⅓ δ_ab Σ_c Q_cc;
* the **Buckingham** octupole Ω_abc = ½ Σ q_i (5 r_a r_b r_c − r² (r_a δ_bc +
  r_b δ_ac + r_c δ_ab)) and the **Cartesian** octupole Σ q_i r_a r_b r_c.

For a charged system the moments depend on `origin`; for a neutral one the
dipole does not, but the quadrupole does as long as the dipole is non-zero.
The polarizability and hyperpolarizability are the responses of the dipole to a
uniform electric field F: μ_a(F) = μ_a + α_ab F_b + ½ β_abc F_b F_c + …, with
the dynamic versions at the stated frequencies; α is symmetric for a static
field.

**Consistency with the root dipole.** The root block `dipole_moment` (§ 3.3)
stays the canonical place for a bare dipole. When both it and a
`properties.electric` `dipole` are present, the two state one dipole: `vector`
**MUST** equal `vector_debye`, and `total` **MUST** equal `total_debye`, each
within 1e-6 debye, for every field both state. A producer that restates the
dipole **SHOULD** also give the same `origin`; the check compares the dipole
values only. A validator reports a disagreement as `E_ROOT_INCONSISTENT`
(§ 6), located at the section's `properties` member.

#### `properties.elastic`

Payload `properties` (`$defs/ElasticPropertiesPayload`, strict; `stiffness` is
required):

| Field | Type | Meaning | Unit |
|---|---|---|---|
| `stiffness` | 6 × 6 numbers, **required** | The stiffness matrix C_IJ = ∂σ_I/∂ε_J in Voigt notation, with the engineering shear strains 2ε_yz, 2ε_xz, 2ε_xy. Symmetric. | GPa |
| `compliance` | 6 × 6 numbers | The compliance matrix S, the inverse of `stiffness`, in the same notation. Symmetric. | GPa⁻¹ |
| `bulk_modulus_gpa` | number | Bulk modulus K. | GPa |
| `shear_modulus_gpa` | number | Shear modulus G. | GPa |
| `youngs_modulus_gpa` | number | Young's modulus E. | GPa |
| `poisson_ratio` | number | Poisson's ratio ν. | — |
| `average` | `"voigt"`, `"reuss"` or `"hill"` | The averaging scheme the four isotropic numbers come from: Voigt (uniform strain, from the stiffness), Reuss (uniform stress, from the compliance) or Hill (the arithmetic mean of the two). Absent means that the scheme is not stated; a producer **SHOULD** state it whenever it writes a modulus, because the schemes differ for an anisotropic crystal. | — |
| `units` | object | Declared units: `stiffness` and the three moduli are `gpa`, `poisson_ratio` is `dimensionless`. | — |

A validator checks that `stiffness` and, when present, `compliance` are
**symmetric**: entries `[I][J]` and `[J][I]` differ by at most 1e-6 times the
largest absolute entry of the matrix. A violation is `E_MEMBER_MALFORMED`
(§ 6). The payload carries no positive-definiteness rule: the stiffness of a
mechanically unstable structure is legitimate data.

#### `properties.dielectric`

Payload `properties` (`$defs/DielectricPropertiesPayload`, strict;
`electronic` is required):

| Field | Type | Meaning | Unit |
|---|---|---|---|
| `electronic` | 3 × 3 numbers, **required** | The electronic dielectric tensor ε∞, the high-frequency response with the ions held fixed. | — (relative permittivity) |
| `ionic` | 3 × 3 numbers | The ionic (lattice) contribution to the static dielectric tensor. | — |
| `static` | 3 × 3 numbers | The static dielectric tensor ε₀. | — |
| `born_charges` | array of 3 × 3 numbers | The Born effective charge tensor Z\* of each atom, one per atom in the atom order of the structure: Z\*[i][j] is the polarization along `i` per unit displacement of the atom along `j` (equivalently the force on the atom along `j` per unit field along `i`), in units of the elementary charge. | e |
| `piezoelectric` | 3 × 6 numbers | The piezoelectric stress tensor e_iα = ∂P_i/∂ε_α, row `i` the polarization component and column α the Voigt strain index, at zero field with the ions relaxed. | C / m² |
| `units` | object | Declared units: `electronic`, `ionic` and `static` are `dimensionless`, `born_charges` is `e`. | — |

Two rules need the payload as a whole, so a validator checks them (§ 6):

* when `electronic`, `ionic` and `static` are all present, `static` **MUST**
  equal `electronic` + `ionic`, entry by entry, within 1e-6;
* when the section's structure resolves (the section-level `structure_ref`,
  else the archive's only `structure`), `born_charges` **MUST** have exactly
  one tensor per atom of it.

A violation of either is `E_MEMBER_MALFORMED`.

### 5.14 `reaction.scheme` — reaction schemes

A `reaction.scheme` section relates `structure` sections (§ 5.1) as the
reactants, reagents and products of one or more reaction steps, the way a
scheme in a paper does: which species react, what is drawn at the arrow, the
kind of arrow, the conditions written beside it and the yield. It carries the
chemistry of a drawn scheme, not its drawing; the species are the archive's
`structure` sections, with their chemical identity (§ 5.1, *Chemical
identity*), so every reader knows the molecules and their roles without the
program that drew them.

Members:

* **`scheme`** — REQUIRED JSON member, schema'd as
  `$defs/ReactionSchemePayload` (strict).

The section has no kind-specific fields; its generic `label` (§ 4.1) titles
the scheme. An archive **MAY** carry several schemes.

Payload `scheme` (`$defs/ReactionSchemePayload`, strict):

| Field | Type | Required | Meaning |
|---|---|:--:|---|
| `steps` | array of step objects, at least one | **yes** | The steps of the scheme, in reading order. Steps that share a species form a sequence: the product of one step is a reactant of the next. |

A step (`$defs/ReactionStep`, strict):

| Field | Type | Required | Meaning |
|---|---|:--:|---|
| `id` | string, `^[A-Za-z0-9_.-]+$` | **yes** | Identifier of the step, unique within the payload. |
| `label` | string | no | Display label of the step, e.g. `"a"` or `"step 2"`. |
| `reactants` | array of participants, at least one | **yes** | The species the step consumes. |
| `products` | array of participants, at least one | **yes** | The species the step forms. |
| `reagents` | array of participants | no | Species drawn at the arrow rather than on either side: reagents, catalysts, solvents drawn as structures. |
| `arrow` | `"forward"`, `"equilibrium"`, `"retrosynthetic"` or `"resonance"` | no | The kind of arrow; absent means `"forward"`. |
| `conditions` | array of strings | no | The text written at the arrow, one entry per line, in reading order, e.g. `["THF, −78 °C", "2 h"]`. |
| `yield_percent` | number, 0 to 100 | no | Yield of the step's products, in percent. |
| `yield_text` | string | no | The yield as printed, e.g. `"87%"`, `">95%"` or `"quant."`; with `yield_percent` it is the display form of that number. |

A participant (`$defs/ReactionParticipant`, strict):

| Field | Type | Required | Meaning |
|---|---|:--:|---|
| `structure_ref` | string | **yes** | Id of the `structure` section of the species. |
| `coefficient` | number > 0 | no | Stoichiometric coefficient of the species in the step; absent means 1. |

Every object of the payload may carry vendor keys beginning `x_`.

**Roles are chemical, the arrow is how the step is drawn.** `reactants` are
consumed and `products` formed in the step's forward direction whatever the
arrow, so a quantity of the step such as a reaction energy is always products
minus reactants:

* `forward` — reactants → products.
* `equilibrium` — reactants ⇌ products; the forward direction is the one
  written.
* `retrosynthetic` — products ⇒ reactants: the target, the step's
  `products`, sits at the tail of the arrow and the precursors, its
  `reactants`, at the head. A producer whose drawing records the species by
  their place at the arrow swaps them when it writes a retrosynthetic step.
* `resonance` — reactants ↔ products: two resonance structures of one
  species, at the tail and the head. No reaction takes place; producers
  **SHOULD NOT** write `reagents`, `conditions` or a yield for it.

A species that appears in several steps is one `structure` section named by
each of them. A species without a three-dimensional structure cannot be a
participant, because every participant names a `structure` section; a
producer writes such a reagent into `conditions` instead.

A validator checks (§ 6), reporting `E_REF_UNRESOLVED` for the first rule and
`E_MEMBER_MALFORMED` for the others:

* every `structure_ref` names a `structure` section;
* the step `id` values are unique within the payload;
* within one role of one step, no `structure_ref` occurs twice (a
  coefficient says how many).

**Reaction energies (reserved).** A later revision will add the energies of
each step, computed from energies of its participants and their coefficients;
the key `energies` of a step is reserved for it, and the strict payload rejects
it until then. A producer that has such numbers now writes them under an `x_`
key.

A two-step synthesis, the second step an equilibrium with a reagent drawn at
the arrow (the structures are sections of the same archive):

```json
{
  "steps": [
    {"id": "r1", "label": "a",
     "reactants": [{"structure_ref": "structure_s1"},
                   {"structure_ref": "structure_s2", "coefficient": 2}],
     "products": [{"structure_ref": "structure_s3"}],
     "conditions": ["THF, −78 °C", "2 h"],
     "yield_percent": 87, "yield_text": "87%"},
    {"id": "r2", "label": "b", "arrow": "equilibrium",
     "reactants": [{"structure_ref": "structure_s3"}],
     "reagents": [{"structure_ref": "structure_s4"}],
     "products": [{"structure_ref": "structure_s5"}]}
  ]
}
```

### 5.15 `preview` — a picture of the archive

A `preview` section carries a picture of the archive's contents, so that a
file manager, a quick-look panel or a catalogue can show the archive without
understanding any other section.

| Role | Format | Required | Meaning |
|---|---|:--:|---|
| `thumbnail` | opaque binary with `media_type` `image/png` or `image/jpeg` | **yes** | A raster image, the picture a file manager shows. |
| `image` | opaque binary with `media_type` `image/svg+xml`, `image/png` or `image/jpeg` | no | A larger or vector rendering of the same picture, for a preview pane. |

Both members are opaque binary members (§ 4.2) and **MUST** carry their
`media_type` (`$defs/PreviewThumbnail`, `$defs/PreviewImage`): a reader learns
the image type from the manifest and does not sniff the bytes.

* **Which picture.** The picture of the archive is the `thumbnail` of the
  first `preview` section in `sections` order. Further `preview` sections
  **MAY** show parts of the archive and say which with `structure_ref` or
  `group` (§ 4.1).
* **Size.** The longest side of a thumbnail **SHOULD** be between 256 and
  1024 pixels, and the member **SHOULD** be at most 1 MiB, so that a file
  manager reads it cheaply. An already compressed image gains nothing from
  DEFLATE, so a producer **MAY** store both members uncompressed.
* **The bytes are what the type says.** A PNG member begins with the PNG
  signature (bytes `89 50 4E 47 0D 0A 1A 0A`), a JPEG member with `FF D8 FF`,
  and an SVG member is a UTF-8 SVG document.
* **SVG is inert.** An SVG member **MUST** be self-contained: it references no
  external resource (no remote `href`, image, font or style sheet) and
  contains no script or event-handler attribute. A consumer that renders it
  **MUST NOT** run scripts or fetch external resources, whatever the document
  contains.

The picture describes the archive and is not data: a consumer **MUST NOT**
read chemistry from it, and a producer that updates an archive (§ 2.3) so that
the picture no longer matches **SHOULD** replace or remove the `preview`
section.

A validator checks that each member's bytes begin with the signature of its
raster `media_type`, and that an `image/svg+xml` member decodes as UTF-8 and
contains an `<svg` element (§ 6). The `media_type` values themselves are
checked by the JSON Schema.

---

## 6. Validation semantics

A validator treats every archive as untrusted input. A conforming validator
performs the checks below and reports each problem as a **finding** with a
severity: an **error** makes the archive invalid; a **warning** does not.
Every check is an error except the orphan rule, the unit-name rule and the
withdrawn version `2`, which are warnings. The JSON Schema check is structural; the other checks are
semantic rules that a validator performs whether or not it can run a JSON
Schema engine.

Container — checked from the ZIP central directory before any member is
decompressed:

* The archive is a readable ZIP, and every entry the validator reads
  decompresses to exactly the uncompressed size recorded for it and matches
  its ZIP CRC-32.
* ZIP entry names are unique (§ 2.1), except `manifest.json`: any other
  duplicated name is an error. Of several `manifest.json` entries the one
  whose central-directory record comes last is the manifest, and each earlier
  copy is superseded and reported as the warning `W_ORPHAN_MEMBER` (§ 2.3).
* Every ZIP entry name follows the path rules of § 2.1.
* No entry exceeds the per-member uncompressed-size cap; the uncompressed
  sizes of all entries together do not exceed the total cap (§ 6.1).
  A validator **MUST NOT** decompress an entry that exceeds a cap, nor any
  member payload once the total cap is exceeded.
* **Warning:** every entry other than the current `manifest.json` and
  directory entries is referenced by a member `path` (orphans, § 2.1, § 2.3).

Manifest:

* `manifest.json` exists (the last entry of that name, § 2.3), is within the
  JSON size and depth caps, and parses as UTF-8 JSON to an object; it
  conforms to the JSON Schema. Among other things the schema check rejects a
  root `byte_order` other than `"little"` (§ 2.2).
* `qvf_version` is `1`, or the withdrawn `2`, which a validator reads as `1`
  for every check, the JSON Schema check included, and reports as the warning
  `W_VERSION_WITHDRAWN` (§ 3.1); `source` carries `program`, `version`, and
  `calculation`, with `program` and `version` non-empty.
* `sections` is an array of objects, each with a string `id`, a string `kind`,
  and a `members` object.
* Section `id` values are unique, and none is `auto_open`, `bookmarks`,
  `camera` or `crossfade` (reserved `viewer_defaults` keys, § 3.1, § 3.4).
* Every `kind` is canonical (§ 5) or a vendor kind `x_<vendor>.*` (§ 7.3).
* Extension governance (§ 7): a `critical: true` section whose kind is neither
  canonical nor a declared vendor namespace is an error.
* Root blocks agree (§ 3.6): every energy of `energies` stated for the method
  that `provenance.method` names agrees with `provenance.scf_energy`, and
  `energies.components.zpve` with `thermochemistry.zpve_eh`, within 1e-8
  hartree. In an archive with exactly one `structure` section, its `charge`
  and `multiplicity` equal `provenance.charge` and `provenance.multiplicity`
  where both are present (§ 5.1).

Members and kind-specific rules:

* Every member spec is an object with a string `path` that follows § 2.1.
* Every declared member `path` exists in the archive.
* Every member `size`, when present, is a non-negative integer equal to the
  uncompressed size the ZIP central directory records for the member's entry
  (§ 4.2). The check needs no decompression and is made before the member is
  read.
* Every declared member `sha256` matches the stored bytes.
* Every JSON member is within the JSON size and depth caps and parses as
  UTF-8 JSON.
* Every binary member that declares a `dtype` uses a § 4.2 dtype name, and
  its byte length equals `itemsize × ∏(shape)`.
* Every JSON member whose section kind and role have a payload definition
  (§ 5) conforms to that `$defs` entry of the JSON Schema. This is part of the
  JSON Schema check.
* A `structure` payload's periodicity contract (§ 5.1) holds: `pbc` is
  present whenever `lattice_vectors` is non-null; `lattice_vectors` is non-null
  whenever any `pbc[i]` is true; and `dimensionality`, when present, equals the
  number of true entries in `pbc`.
* A `structure` section's `charge` and `multiplicity` are possible for the
  atoms of its payload (§ 5.1, *Chemical identity*): with `charge`, the
  electron count N = Σ `atomic_number` − `charge` is not negative, and with
  `multiplicity` too, `multiplicity` − 1 is at most N and has the parity of N.
* `volume.difference.operand_a`/`operand_b` resolve when present.
* `reaction.waypoints.trajectory_ref` resolves to a `trajectory` section.
* A section's `structure_ref`, when present, resolves to a `structure`
  section, and its `run_ref` to a `run.record` section (§ 4.1).
* A standalone `bonds` section whose `structure_ref` names a `structure`
  section: every pair's `i` and `j` is an atom of that structure, from 0 to
  its atom count − 1 (§ 5.1). A `bonds` section without `structure_ref`, and
  the `bonds` member of a `structure` section, are not checked against an atom
  count.
* `bands`: a rank-4 `projections` and an `occupations` have the
  `[n_spin, n_k, n_bands]` extents of `eigenvalues`, and a rank-3
  `projections` has its `[n_k, n_bands]` (§ 5.6).
* `trajectory` and `reaction.path` (§ 5.5): `coords` are float32 only when
  `metadata.precision` is `"float32"`, and `precision`, when present, names
  the dtype of `coords`. Against a `coords` of shape `[n_frames, n_atoms, 3]`,
  `velocities` and `gradient` are `[n_frames, n_atoms, 3]`, `times`,
  `energies` and, for a trajectory, `cell_volumes` and `temperatures` are
  `[n_frames]`, and `lattice` is `[3, 3]` or `[n_frames, 3, 3]`. The binary
  `energies` agree with `metadata.energies` within 1e-10 hartree. A
  trajectory that carries `lattice` has `metadata.pbc`, and one whose
  `metadata.pbc` has a `true` entry carries `lattice`. Each `lattice` is read
  in its own convention (§ 4.3); a validator never compares or converts one
  carrier against another.
* `forces` and `hessian` (§ 5.10, § 5.11): `gradient` is `[n_atoms, 3]`,
  `stress` is `[3, 3]`, `matrix` is square with an extent 3N divisible by 3,
  and `hessian` `meta.atom_masses_amu`, when present, has N entries. When the
  section's structure resolves (the section-level `structure_ref`, else
  `meta.structure_ref`, else the archive's only `structure` section),
  `n_atoms` and N equal its number of atoms. A `meta.structure_ref` names a
  `structure` section.
* `vibrations.hessian_ref`, when present, names a `hessian` section (§ 5.5).
* `excited_states` (§ 5.12): the `index` values of `states` are unique and
  ascending, and `meta.n_states`, when present, equals the number of states.
  The section's `wavefunction_ref` and every state's `nto_ref` name
  `wavefunction.gto` sections, and every `density_ref` a
  `volume.transition_density` or `volume.difference` section. When
  `wavefunction_ref` names a `wavefunction.gto` section, every contribution's
  `from_mo` and `to_mo` is a row of its coefficient member: of
  `mo_coefficients`, else of `mo_coefficients_alpha` or
  `mo_coefficients_beta` as `spin` selects, and of both when `spin` is
  absent. The rows are read from the manifest `shape` of the member.
* Unrestricted NTOs (`wavefunction.gto`, § 5.3): each alpha/beta metadata
  channel has one nonnegative, non-increasing weight and one energy per
  coefficient row; optional `n_mo` agrees. The state reference resolves, and
  exactly one opposite-role section exists for the same state, with the
  same spin layout and equal row counts and weights per spin. Findings use
  `E_MEMBER_MALFORMED` for channel/pair inconsistencies and
  `E_REF_UNRESOLVED` for missing/ambiguous partners or state references,
  located at the affected section's `members/mo_metadata`. A unit weight
  sum is not imposed because normalization is conditional.
* A one-dimensional spectrum's `states_ref`, when present, names an
  `excited_states` section, and every `peaks[].assignment.state_index` is the
  `index` of one of its states (§ 5.4).
* A `volume.transition_density` section's `excited_states_ref`, when present,
  names an `excited_states` section, and its `state_index` is the `index` of
  one of its states (§ 5.2).
* `job.spec`: the `structure_ref` of the `spec` payload, when present, names a
  `structure` section (§ 5.9). The payload's `solvation`, `scf`, `geometry` and
  `resources` objects are checked by the JSON Schema check like the rest of
  the payload. A `run.record` `job_ref`, when present, names a `job.spec`
  section (§ 5.8).
* `reaction.scheme` (§ 5.14): every participant's `structure_ref` names a
  `structure` section; the step `id` values are unique within the payload;
  and no `structure_ref` occurs twice within one role of one step.
* `preview` (§ 5.15): a member whose `media_type` is `image/png` begins with
  the PNG signature and one whose `media_type` is `image/jpeg` with
  `FF D8 FF`; an `image/svg+xml` member decodes as UTF-8 and contains `<svg`.
* `wavefunction.gto` (§ 5.3): every coefficient member has rank 3 with a last
  extent of 2 when `mo_metadata` says `complex: true` (or
  `coefficient_encoding: "complex_split_last_axis"`), and rank 2 otherwise;
  every `symmetry_labels` array, at the top level or in a spin channel, has
  one entry per entry of its `energies`; `overlap` and the density members are
  `[n_ao, n_ao]`, with n_ao counted from the `basis` shells (its `n_ao` when
  the shells cannot be read); `density_matrix` appears only with
  `mo_coefficients`, and `density_matrix_alpha` and `density_matrix_beta` only
  together and only with the α and β coefficients; and `basis.n_electrons`,
  when present, equals the sum of the occupations over both spin channels
  within 1e-6, unless `occupation_semantics` is `transition_weight`.
* A `basis.n_electrons` and a root `provenance.n_electrons` that are both
  present are equal (§ 3.2, § 5.3).
* When a `wavefunction.gto` section carries `overlap`, its diagonal agrees
  with the self-overlaps that Appendix A.1 gives for the `basis` payload, to a
  relative tolerance of 1e-6 (Appendix A.1, *Verifying with `overlap`*).
* Volumes (§ 5.2): the `data` of `volume.laplacian` is `[nx, ny, nz]`, of
  `volume.vector` `[nx, ny, nz, 3]`, of `volume.complex` `[nx, ny, nz, 2]`
  and of `volume.series` `[n, nx, ny, nz]`; a `volume.series` `axis` has one
  `values` entry, and one `labels` entry when it has labels, per slab, and
  its `trajectory_ref`, when present, names a `trajectory` or `reaction.path`
  section.
* Grids (§ 5.2): a grid payload's `structure_ref`, when present, names a
  `structure` section. When `periodic[a]` is true and the grid's structure
  resolves (the section-level `structure_ref`, else the grid's
  `structure_ref`, else the archive's only `structure` section), that
  structure's `pbc[a]` is true, and `voxel_vectors[a] × (shape[a] − 1)` when
  `includes_endpoint` is true or absent, or `voxel_vectors[a] × shape[a]`
  when it is false, equals lattice row `a` of the structure converted to bohr
  within 10⁻⁶ relative.
* The property kinds (§ 5.13): a `properties.elastic` `stiffness` and
  `compliance` are symmetric to within 1e-6 of their largest entry; in a
  `properties.dielectric` payload, `static` equals `electronic` + `ionic`
  within 1e-6 when all three are present, and `born_charges` has one tensor per
  atom of the section's structure when that resolves (the section-level
  `structure_ref`, else the archive's only `structure` section); each
  violation is `E_MEMBER_MALFORMED`. A `properties.electric` `dipole` agrees
  with the root `dipole_moment` within 1e-6 debye (`vector` against
  `vector_debye`, `total` against `total_debye`), or the finding is
  `E_ROOT_INCONSISTENT`. The payloads `properties.electric`,
  `properties.elastic` and `properties.dielectric` conform to the `$defs`
  definitions of § 5.13.
* `wavefunction.bloch` (§ 5.3.1): `mo_coefficients` is
  `[n_k, n_spin, n_mo, n_ao, 2]` with `n_spin` 1 or 2, and `energies` and
  `occupations` have its `[n_k, n_spin, n_mo]`; the `kpoints` payload lists
  n_k k-points and, when present, n_k `labels` and n_k `weights` that sum to
  1 within 1e-6; `mo_metadata` `n_spin` (1 for `restricted`, 2 for
  `unrestricted`), `n_mo` and `n_ao`, and `basis.n_ao`, agree with those
  axes, and `mo_metadata.n_mo_per_k` has n_k entries of at most `n_mo`. The
  section's structure (the section-level `structure_ref`, else
  `basis.structure_ref`) is a `structure` section with at least one periodic
  axis.
* The JSON members `bands.kpath`, `dos.coop.meta`, `dos.cohp.meta`,
  `fermi_surface.mesh`, `phonon_bands.qpath`, `phonon_dos.meta`,
  `equation_of_state.fit` and `run.record.files` conform to the `$defs`
  payload definitions their tables name (§ 5.6, § 5.8).
* `run.record`: `input` and `log` members decode as UTF-8; every key of the
  `files` index names a member that exists in the section.
* `atom_properties`: every `custom.<name>` member has an entry named
  `custom.<name>`, a JSON object, in the section's `meta` member (§ 5.7); a
  missing `meta` member, or one without that entry, is an error for that
  member. The `meta` payload itself is checked against
  `$defs/AtomPropertiesMetaPayload` like any payload definition.
* A one-dimensional spectrum's `frequencies`, `intensities` and, when
  present, `x` have the same length — a part of `SpectrumPayload` that JSON
  Schema cannot express (§ 5.4).
* **Warning:** every declared unit is a unit name of § 4.3 (*Declared
  units*): each value of the `units` object of a JSON member of a canonical
  kind, and of the `meta.units` object of an `excited_states` payload; a
  one-dimensional spectrum's `x_units`, `y_units` and
  `broadening.units`; the `units` of a `volume.series` `axis`; and the
  `units` of every `{value, units}` object directly under `provenance`, which
  may also be a legacy spelling. This
  rule needs no JSON Schema engine. The schema types the same strings, so a
  validator that runs it also reports the schema finding.

### 6.1 Resource limits

A validator reports an error when an archive exceeds one of the caps below.
Validators **SHOULD** use the reference values, which are the defaults of the
reference validators; producers **SHOULD** stay within them so that their
archives validate everywhere.

| Quantity | Reference value |
|---|---|
| Uncompressed size of one ZIP entry | 8 GiB |
| Total uncompressed size of all ZIP entries | 64 GiB |
| Size of `manifest.json` or of one JSON member | 256 MiB |
| Nesting depth of JSON arrays and objects | 64 — a top-level `[]` or `{}` has depth 1, a scalar depth 0 |

Compression ratio is not a format validity limit: valid constant arrays can
compress by more than 1000:1. The size caps above and the requirement never to
inflate past the declared uncompressed size bound decompression independently
of compression ratio. Readers **MUST** reject a mismatch between actual and
declared size or a CRC-32 mismatch as `E_ZIP_UNREADABLE`.

The reference APIs retain `max_ratio` as an optional, stricter application
policy for compatibility, with `0` disabling it by default. A positive value
applies to compressed entries of at least `ratio_exempt_bytes` (default 1 MiB);
STORE entries remain exempt. `E_COMPRESSION_RATIO` identifies rejection by this
explicit policy, not a malformed QVF archive.

### 6.2 Finding codes

Conforming validators **SHOULD** report each finding with the stable code
listed in Appendix B ("Validation finding codes"), so that tools, bindings,
and test suites match findings by code rather than by message text. A finding
carries exactly one code: `E_…` codes are errors, `W_…` codes are warnings.
The conformance suite's negative corpus (`conformance/negative/`) holds a
must-fail archive for each rule, naming the code it must produce.

---

## 7. Extension and versioning model

### 7.1 `qvf_version`

Integer. v1 = `1`. A value > `1` signals breaking changes; a v1-only consumer
**MUST** refuse such files, except the withdrawn `2`, which it **SHOULD** read
as `1` (§ 3.1). Minor, backward-compatible growth (new optional
kinds, new optional members) does **not** bump the integer — capability is
detected via supported-kind checking and the `extensions` block.

### 7.2 Canonical kinds

Canonical kinds appear in the schema and § 5. They may grow in minor revisions;
no canonical kind may be removed or have its required members changed without a
major version bump.

### 7.3 Vendor namespace `x_<vendor>.*`

Any kind matching `x_<vendor>.<specific>` is a vendor extension. `<vendor>`
**SHOULD** be a DNS-like identifier (`x_orca`, `x_pyscf`, `x_vasp`). Vendor
members must be valid JSON, binary or opaque binary member specs (§ 4.2);
an opaque member, such as an image, **SHOULD** carry its `media_type`.
Registering the namespace in `registry.json` (`vendor_namespaces`) records
who uses it and is advisory (`GOVERNANCE.md`). A consumer **MUST** list vendor
sections as present-but-unsupported unless the extension is `critical: true`.
Unknown *non-vendor* kinds are not valid QVF.

### 7.4 Root `extensions` block and per-section `critical`

The manifest root may declare vendor extensions:

```jsonc
{ "extensions": { "x_orca": { "version": "1.0", "schema_uri": "…", "critical": false } } }
```

Each section may carry `critical` (default `false`). If `critical: true` and the
consumer does not support the kind, the consumer **MUST** refuse to open the
file rather than silently drop the section. Use `critical: true` only when the
section changes correct interpretation of the rest of the file (e.g. an ECP that
alters the effective electron count).

### 7.5 Promoting a vendor kind to canonical

1. The kind is used in production by ≥ 2 independent producers for ≥ 1 minor
   cycle.
2. A spec PR adds it to § 5, the schema `oneOf` branches, and the validator.
3. Its member contract is stable across the two producers.
4. At least one reference consumer renders it.

The maintainer may promote a kind before the first criterion is met, as a
maintainer decision recorded in `GOVERNANCE.md` and `registry.json`;
`spectra.epr` and `wavefunction.bloch` (§ 5.3.1) were promoted that way. A
kind promoted before a second independent producer ships it carries
`"status": "provisional"` in its `registry.json` `promotions` entry. A
provisional kind is canonical in every respect (§ 7.2): producers may emit it,
consumers may rely on it, and any revision it needs is additive. It becomes
final, and loses the status, when a second producer ships it.

---

## 8. Conformance

### 8.1 Producer conformance

A conforming producer **MUST**:

* write a valid ZIP with `manifest.json` at the root;
* set `qvf_version` to `1`;
* include `source.program`, `source.version`, `source.calculation`;
* use unique section ids matching `^[A-Za-z0-9_.-]+$`, other than the reserved
  `auto_open`, `bookmarks`, `camera` and `crossfade` (§ 3.4);
* use canonical kinds or `x_<vendor>.*`;
* include only member paths that exist in the ZIP and follow § 2.1 (relative,
  `/`-separated, no `..` segments), and write each ZIP entry name once —
  except `manifest.json` when updating an archive in place, whose new
  manifest is the last entry (§ 2.3);
* compute sha256 over the exact stored bytes of each member, and write a
  member's `size`, when it writes one, as the uncompressed byte count of the
  member's entry (§ 4.2);
* write `byte_order`, when it writes it, as `"little"` (§ 2.2);
* write numeric fields in the units of § 4.3, and name every declared unit
  with a unit name of § 4.3;
* write binary payloads matching the declared dtype and shape, little-endian;
* ensure JSON members parse as UTF-8 JSON, and that every JSON member with a
  payload definition (§ 5) conforms to it;
* write `pbc` in every `structure` payload that carries `lattice_vectors`, and
  keep `dimensionality` — if emitted at all — equal to `sum(pbc)` (§ 5.1);
* point every `structure_ref`, `run_ref` and `trajectory_ref` on a section at
  a section of the kind it names (§ 4.1, § 5.5), and likewise every reference
  of an excited-state calculation: `wavefunction_ref`, `nto_ref`,
  `density_ref`, `states_ref` and `excited_states_ref` (§ 5.12), every
  participant of a `reaction.scheme` (§ 5.14) and every `run.record`
  `job_ref` (§ 5.8);
* write a structure's `charge` and `multiplicity` only as values possible for
  its atoms, and equal to those of `provenance` in an archive with one
  `structure` section (§ 5.1);
* write the atom indices of a standalone `bonds` section that carries a
  `structure_ref` as atoms of the structure it names (§ 5.1);
* give every `preview` member its `media_type`, and write an SVG `image` that
  references no external resource and contains no script (§ 5.15);
* declare each `x_<vendor>` namespace in root `extensions` when any section in
  that namespace is `critical: true`;
* set `critical: true` only when consumer support is truly required.

A conforming producer **SHOULD** include `provenance` when known; include
`citations` when results depend on citable methods/bases/functionals; prefer a
specific `volume.*` kind over `volume.generic`; use a vendor namespace for
experimental payloads; set `structure_ref` on every standalone `bonds` section
of an archive with several structures (§ 5.1); stay within the reference
resource limits of § 6.1; and run a validator on its own output in tests.

### 8.2 Consumer conformance

A conforming consumer **MUST**:

* read `manifest.json` by name, not by ZIP entry order, and use the entry
  whose central-directory record comes last when there are several (§ 2.3);
* ignore every ZIP entry the manifest does not reference: orphans and
  superseded manifests (§ 2.1, § 2.3);
* sanity-check `qvf_version`, `source`, and `sections`;
* verify sha256 before using a member's bytes;
* interpret only kinds it supports;
* report unsupported vendor sections without corrupting or misinterpreting them;
* respect declared dtype, shape, and units, reading a legacy unit spelling
  as the unit name it stands for (§ 4.3);
* take per-axis periodicity from `pbc`, and never draw, replicate along, or
  measure an extent from a lattice row whose axis is non-periodic (§ 5.1);
* read a payload's atom indices against the structure named by the section's
  `structure_ref` when the section has one, in preference to a `structure_ref`
  inside the payload (§ 4.1), and the bonds of a standalone `bonds` section
  without one against the first structure (§ 5.1);
* find sections by kind and reference, never by a conventional id such as
  `structure`, and open an archive with several structures, none of them
  named `structure`, showing its first structure (§ 4.1);
* open a file whose `viewer_defaults` carries a malformed or unknown hint,
  ignoring that hint (§ 3.4);
* render an SVG `preview` member, if at all, without running scripts or
  fetching external resources (§ 5.15);
* honor the per-section `critical` flag and the root `extensions` `critical`
  flags, refusing to open when an unsupported extension is critical.

A conforming consumer **SHOULD** read an archive stamped `qvf_version: 2` as
`1` (§ 3.1), refuse an archive with duplicate ZIP entry
names other than `manifest.json`, refuse an archive whose `byte_order` is not
`"little"` (§ 2.2), and apply the § 6.1 resource limits before decompressing or parsing a
member (§ 2.1).

A consumer **MAY** require specific kinds for a given operation (a band plotter
may refuse a file without `bands`; a structure viewer may open it and ignore all
scalar fields).

---

## 9. Known open items (informative)

These are acknowledged limitations of v1, not defects:

1. **Endianness tag.** dtype names do not carry byte order. v1 archives are
   little-endian and may say so once, with the root `byte_order` (§ 2.2);
   a per-dtype prefix such as `<f8` changes the dtype vocabulary and is a
   candidate for the next major version.
2. **k-resolved Bloch wavefunctions.** `wavefunction.gto` covers molecular
   systems and the Γ-point; k-resolved orbitals over an atom-centred
   Gaussian basis are carried by `wavefunction.bloch` (§ 5.3.1), which is
   provisional until a second producer ships it. Not yet covered: k-resolved
   density matrices (the spin-density restart payload of
   `x_vibeqc.bloch_wavefunction` stays in that vendor section), two-component
   (spinor) Bloch orbitals, and Bloch orbitals over other bases such as plane
   waves or numeric atomic orbitals; producers may use `x_<vendor>.*` for
   those meanwhile.
3. **NMR and EPR schema tightening.** The `spectra.nmr` and `spectra.epr`
   payloads are typed but permissive in v1 (§ 5.4): their known fields are
   checked and every object stays open until producer conventions settle.
4. **Fermi-surface meshes.** v1 assumes a Γ-centered uniform Monkhorst-Pack
   grid.
5. **Cartesian `wavefunction.gto` shells have little mileage.** The
   convention is fully specified (Appendix A.1: one `N_i` per shell, from the
   total `l`), but `pure: false` has far less cross-consumer mileage than
   `pure: true`, and the per-component normalization trap described there has
   already been hit in practice — by the reference producer's own two
   readers, which is how it was found. A producer can now make its archives
   checkable: with the `overlap` member (§ 5.3) every reference validator
   compares the stored diagonal with Appendix A.1 and reports a `d_xy`
   self-overlap of 1 rather than 1/3 as `E_NORMALIZATION`. The check covers
   producers only; a consumer that evaluates AOs still has to test its own
   evaluator on a Cartesian d shell (the corpus archives
   `wavefunction_cartesian.qvf` and `wavefunction_overlap.qvf` carry the
   numbers), because a spherical-only test will not catch the trap.
6. **QTAIM position unit.** The ångström default for `topology.qtaim`
   positions (§ 5.7) is provisional pending a maintainer decision; producers
   should declare `units.position`.
7. **Unit names of compound quantities.** The unit vocabulary of § 4.3
   names the units that QVF payloads declare today. It has no name yet for
   compound units such as an electron density (e·bohr⁻³), its Laplacian
   (e·bohr⁻⁵, `volume.laplacian`), a reciprocal length (Å⁻¹) or a density
   of states (states·eV⁻¹), so such fields cannot be declared; their units are those of § 4.3 and the payload tables. A
   later revision may add names, which is an additive change.
8. **Two-dimensional drawings.** v1 has no canonical kind for an editable
   drawing of molecules or of a scheme: the drawing formats in use (Ketcher's
   KET, ChemDraw's CDXML, the 2-D coordinates of an MDL molfile) are
   program-specific, and one producer writes drawings into QVF today. Such a
   drawing travels in its producer's vendor section, as vibe-sketch's
   `x_vibesketch.document` does. What a drawing *means* has canonical
   carriers: the identity of each species on its `structure` section (§ 5.1),
   the scheme as `reaction.scheme` (§ 5.14) and the picture as `preview`
   (§ 5.15).
9. **Reaction energies.** `reaction.scheme` (§ 5.14) reserves the step key
   `energies` for the energies of a step, to be specified when a producer
   computes them from the energies of the participants.

---

## 10. Governance and stewardship

QVF is an **open standard**. It was created by and is stewarded by the
[vibe-qc](https://vibe-qc.com) project — that origin is explicit — but the format
is **producer-neutral by design** (§ 1) and its specification and reference
implementations are Apache-2.0, so any code may implement a conforming producer
or consumer, including a proprietary one, with no obligation back to vibe-qc.
vibe-qc's own non-standard data lives under the `x_vibeqc.*` vendor namespace,
exactly like any other adopter's.

The governance model, versioning policy, change process, and the
vendor→canonical promotion process are described in `GOVERNANCE.md`; the
machine-readable list of canonical kinds and registered vendor namespaces is in
`registry.json` (kept in lock-step with the schema). Stewardship is intended to
broaden as independent producers adopt the format.

---

## Appendix A. AO ordering, primitive normalization, and component naming

A `wavefunction.gto` or `wavefunction.bloch` section is only correct if the
MO coefficients, the AO ordering, and the primitive normalization all agree with **A.1–A.2** of this
appendix. This is the part most producers get subtly wrong. **A.3** applies to
`basis.ao`, which names a single AO rather than ordering a whole matrix, and
depends on the same in-shell ordering.

### A.1 Contraction and the normalization convention

Each contracted basis function is reconstructed as

```text
χ(r) = Σ_i c_i · φ_i(r)
```

where each **primitive** `φ_i` is normalized as:

```text
φ_i(r) = N_i · (x−A_x)^{l_x} (y−A_y)^{l_y} (z−A_z)^{l_z} · exp(−α_i r²)

N_i = (2 α_i / π)^{3/4} · (4 α_i)^{l/2} / √((2l − 1)!!),   l = l_x + l_y + l_z
```

The coefficients `c_i` written into `basis.json` **MUST** apply to primitives
normalized exactly this way.

> **`N_i` depends only on `l`, not on `(l_x, l_y, l_z)`.** This matters for
> Cartesian shells, and it is deliberate: `N_i` is the normalization of the
> *axial* component `(l, 0, 0)` and is applied uniformly to the whole shell.
> It is what libint and libcint store, so a producer divides by one factor per
> primitive and is done.
>
> The consequence is that for `pure: false` shells the primitives are **not**
> individually unit-normalized. Only the axial components (`x^l`, `y^l`, `z^l`)
> have unit norm; a mixed component such as `d_xy` does not. Concretely, for a
> single Cartesian d shell the overlap diagonal in this convention is
> `⟨xx|xx⟩ = 1` but `⟨xy|xy⟩ = 1/3`. A genuinely unit-normalized Cartesian
> Gaussian would divide by `√((2l_x−1)!!(2l_y−1)!!(2l_z−1)!!)` instead of
> `√((2l−1)!!)`, which for `d_xy` differs by a factor of `√3`.
>
> **Consumers MUST NOT apply a per-component correction on top of `N_i`.**
> Doing so scales every mixed Cartesian AO by that factor and the resulting
> density is wrong by its square — a factor of 3 for `d_xy`. This is a real,
> observed failure mode, not a hypothetical one. For `pure: true` shells the
> question does not arise: the solid-harmonic transform absorbs it and the
> two conventions coincide, which is why the discrepancy can sit unnoticed in
> a spherical-only pipeline.

> **This is not a hypothetical warning.** Both of vibe-qc's AO evaluators
> shipped that per-component correction — its own `evaluate_ao` and the
> vibe-view reader — while its writer divided by `N_i` exactly as specified.
> Writer and readers therefore disagreed by that factor on `pure: false`
> shells, and nothing failed loudly, because spherical shells (everything a
> named basis produces) are unaffected. Both readers were corrected on
> 2026-08-05. If you are implementing a consumer, this is the single easiest
> thing here to get wrong, and a spherical-only test will not catch it.

> **The trap.** Integral libraries such as **libint** and **libcint** (used by
> PySCF, and by many other codes including patterns ORCA developers will
> recognize from GBW-style storage) store contraction coefficients **already
> pre-multiplied by `N_i`** — i.e. the stored coefficient is `c_i · N_i`, so it
> multiplies an *un-normalized* primitive. If your engine stores coefficients
> that way, you **MUST divide each stored coefficient by `N_i`** before writing
> it into `basis.json`. Writing the engine-native (pre-scaled) coefficients
> produces orbitals that look plausible but are quantitatively wrong, especially
> for high-`l` and tightly contracted shells.

Reference implementation of `N_i` (Python; the C++ toolkit has the identical
routine as `qvf::primitive_norm`):

```python
import math
def primitive_norm(alpha: float, l: int) -> float:
    radial  = (2.0 * alpha / math.pi) ** 0.75
    angular = (4.0 * alpha) ** (l / 2.0)
    df = 1.0
    for k in range(1, 2 * l, 2):   # (2l-1)!!
        df *= k
    return radial * angular / math.sqrt(df)
```

Both reference writers expose a flag: pass your coefficients as-is if they are
*already* on `N_i`-normalized primitives, or ask the writer to divide by `N_i`
if they are libint/libcint-style. When in doubt, verify against a known orbital: the
HOMO of H₂O/STO-3G evaluated on a grid should integrate to ~1 electron per spin.
Better still, write the `overlap` member and let a validator check it, as
described next.

#### Verifying with `overlap`

The self-overlap of every AO follows from `basis.json` alone. For a shell of
angular momentum `l` with exponents `α_i` and coefficients `c_i`, the
self-overlap of its axial component `(l, 0, 0)` is

```text
S_axial = Σ_i Σ_j c_i c_j · ( 2 √(α_i α_j) / (α_i + α_j) )^(l + 3/2)
```

which is `Σ_ij c_i c_j N_i N_j ⟨x^l e^(−α_i r²) | x^l e^(−α_j r²)⟩` with the
`N_i` above folded in. The diagonal of the AO overlap matrix, at the AO index
of each component of the shell (A.2), is then:

* **spherical shell:** `S_axial` for every `m`, because a real solid harmonic
  built on `N_i`-normalized primitives is normalized like the axial
  component;
* **Cartesian shell:** `S_axial · (2l_x−1)!! (2l_y−1)!! (2l_z−1)!! / (2l−1)!!`
  for the component `(l_x, l_y, l_z)`: `S_axial` for an axial component,
  `S_axial / 3` for `d_xy`, `S_axial / 5` for `f_xxy`, `S_axial / 15` for
  `f_xyz`.

A contraction normalized in the usual way has `S_axial = 1`, so the diagonal
is 1 for every spherical AO and every axial Cartesian AO, and 1/3 for `d_xy`.

A `wavefunction.gto` section may carry the overlap matrix its producer
computed with its own integral code (the `overlap` member, § 5.3), and
producers **SHOULD** write it. A validator then computes the values above from
`basis.json` and compares each with the stored diagonal at that AO's index,
to a relative tolerance of 1e-6. A disagreement is the error
**`E_NORMALIZATION`** (Appendix B): the coefficients in `basis.json` are not
on `N_i`-normalized primitives (the libint/libcint trap above), a
per-component correction has been applied to a Cartesian shell (a `d_xy`
diagonal of 1 rather than 1/3), or the AO order disagrees with A.2. The check
sees the diagonal only, so it cannot tell apart the components of a spherical
shell, whose diagonal entries are all equal: a permutation of `m` within a
spherical shell passes it.

This is the most valuable interoperability check the format has. The
normalization trap produces orbitals that look plausible and are wrong; with
`overlap` in the archive it fails validation at write time instead.

### A.2 AO ordering within a shell

AO order is part of the contract and **MUST** match the `pure` flag:

* **Spherical** (`pure: true`): functions are ordered `m = −l, −l+1, …, +l`.
* **Cartesian** (`pure: false`): libint lexicographic ordering of `(l_x, l_y,
  l_z)` with `l_x + l_y + l_z = l`, decreasing `l_x` first, then `l_y`.

The row/column index of an AO in `mo_coefficients` is its position in the
concatenation of all shells (shell order = `shells` array order), each shell
contributing `2l+1` (spherical) or `(l+1)(l+2)/2` (Cartesian) functions.

If your engine uses a different in-shell order (common differences: `p` as
`x,y,z` vs `m = −1,0,+1`; `d` as `xx,yy,zz,xy,xz,yz` vs `m = −2…+2`), you
**MUST** permute the coefficient rows into QVF order before writing.

### A.3 Naming a single component (`basis.ao`)

A `basis.ao` section carries **one** component of one shell, so its
`ao_metadata` has to name *which* — and the two shell kinds above are named
differently:

* **Spherical** (`pure: true`): `angular_momentum` is `[l, m]` with
  `m ∈ [−l, +l]`, the harmonic's own index under the A.2 ordering.
* **Cartesian** (`pure: false`): there is **no** magnetic quantum number.
  The component is named by `cartesian_powers`, the `[l_x, l_y, l_z]` of the
  monomial, so a d shell's six components are `[2,0,0] … [0,0,2]` in the A.2
  order. `angular_momentum[1]` is then `0` by convention and carries no
  meaning.

`cartesian_powers` is present **if and only if** the component is Cartesian,
and `l_x + l_y + l_z` **MUST** equal `angular_momentum[0]`. A consumer
therefore reads `l` from `angular_momentum[0]` for either kind, but **MUST
NOT** read `angular_momentum[1]` as an `m` when `cartesian_powers` is
present — `[2, 0]` on a Cartesian d component does not mean `d_z²`.

> **The trap.** Deriving `m` as *component index* − `l` is correct for a
> spherical shell and wrong for a Cartesian one, where it runs past `+l`: a
> Cartesian d shell comes out as `m = −2 … +3` over its six components. A
> producer that emits `basis.ao` for Cartesian shells **MUST** branch on the
> shell's `pure` flag.

`cartesian_powers` is an optional member added under v1 — the additive case
per `GOVERNANCE.md`, so there is no `qvf_version` bump, and a spherical-only
producer is unaffected.

---

## Appendix B. Validation finding codes (informative)

Stable identifiers for the findings of § 6. A code is never renamed or
reused; a new check gets a new code. Messages are free text for people; tools
match on the code. The Python reference validator (`python/qvf_reader.py`),
the C++ reference validator (`qvf::QvfReader::validate_findings`) and the
JavaScript validator (`@qvf/core`) emit these codes. All three run the JSON
Schema check: Python through `jsonschema`, C++ and JavaScript through the
checker generated from the schema (`scripts/gen_kinds.py`). A validator that
runs without the check (`jsonschema` missing, or the check disabled) reports
`W_SCHEMA_SKIPPED` instead of `E_SCHEMA` / `E_PAYLOAD_SCHEMA`.

A finding is `{severity, code, where, message}`. `where` locates it:
`archive` (the ZIP as a whole), `entry:<name>` (one ZIP entry),
`manifest.json` (the manifest document as a whole), or a JSON Pointer
(RFC 6901) into `manifest.json`, such as `/sections/2/members/data`. A
finding about a member's payload points at the member spec that declares it.

| Code | Severity | Rule | Meaning |
|---|---|---|---|
| `E_ZIP_UNREADABLE` | error | § 2.1 | The archive is not a readable ZIP, or an entry cannot be decompressed (corrupt data, CRC mismatch, unsupported method, data that does not inflate to its recorded size). |
| `E_ZIP_DUPLICATE_ENTRY` | error | § 2.1, § 2.3 | Two or more ZIP entries share a name other than `manifest.json`. (Several `manifest.json` entries are an archive updated in place: the last is the manifest and each earlier one is a `W_ORPHAN_MEMBER`.) |
| `E_MEMBER_PATH` | error | § 2.1 | A ZIP entry name or a member `path` contains a backslash or a NUL, begins with `/` or a drive letter, or has a `..` segment. |
| `E_MEMBER_TOO_LARGE` | error | § 6.1 | An entry's uncompressed size exceeds the per-member cap; it is not decompressed. |
| `E_TOTAL_TOO_LARGE` | error | § 6.1 | The entries' total uncompressed size exceeds the total cap; no member payload is decompressed. |
| `E_COMPRESSION_RATIO` | error | § 6.1 | An explicitly enabled application ratio policy rejects a compressed entry; it is not decompressed. Disabled by default; not a format validity rule. |
| `W_ORPHAN_MEMBER` | warning | § 2.1, § 2.3 | A ZIP entry, other than the current `manifest.json` or a directory entry, that no member `path` references, or a superseded `manifest.json`: an earlier entry of that name in an archive updated in place (`where` is `entry:manifest.json`). |
| `E_MANIFEST_MISSING` | error | § 2.1 | There is no `manifest.json` at the archive root. |
| `E_MANIFEST_JSON` | error | § 3 | `manifest.json` is not UTF-8 JSON, or not a JSON object. |
| `E_JSON_TOO_LARGE` | error | § 6.1 | `manifest.json` or a JSON member exceeds the JSON size cap. |
| `E_JSON_TOO_DEEP` | error | § 6.1 | `manifest.json` or a JSON member nests arrays and objects deeper than the depth cap. |
| `E_SCHEMA` | error | § 6 | `manifest.json` violates the JSON Schema (one finding per schema error), for example with a root `byte_order` other than `"little"` (§ 2.2). |
| `W_SCHEMA_SKIPPED` | warning | § 6 | The validator could not run the JSON Schema check; the semantic checks still ran. |
| `E_VERSION` | error | § 3.1 | `qvf_version` is neither `1` nor the withdrawn `2`. |
| `W_VERSION_WITHDRAWN` | warning | § 3.1 | `qvf_version` is the withdrawn `2`. The validator checks the archive as version `1`; producers **MUST NOT** write `2`, and consumers **SHOULD** read it as `1`. |
| `E_SOURCE` | error | § 3.1 | `source` is not an object, lacks `program`, `version`, or `calculation`, or has an empty `program` or `version`. |
| `E_SECTION_MALFORMED` | error | § 4.1 | `sections` is not an array, or a section is not an object, has no string `id` or `kind`, or has no `members` object. |
| `E_SECTION_ID_DUPLICATE` | error | § 4.1 | Two sections share an `id`. |
| `E_RESERVED_ID` | error | § 3.1, § 3.4 | A section `id` is `auto_open`, `bookmarks`, `camera` or `crossfade`. |
| `E_KIND_UNKNOWN` | error | § 7.3 | A `kind` is neither canonical nor `x_<vendor>.*`. |
| `E_CRITICAL_UNDECLARED` | error | § 7.4 | A `critical: true` vendor section's namespace is not declared in root `extensions`. |
| `E_MEMBER_MALFORMED` | error | § 4.2, § 5.1, § 5.2, § 5.3, § 5.3.1, § 5.5, § 5.6, § 5.7, § 5.10, § 5.11, § 5.12, § 5.13, § 5.14, § 5.15 | A member spec is not an object or has no string `path`; a binary member's `shape` is not a list of non-negative integers; a pair of a standalone `bonds` section names an atom index `i` or `j` that is not an atom of the structure its `structure_ref` names; a `wavefunction.gto` coefficient member's rank disagrees with `complex`, a `symmetry_labels` array differs in length from its `energies`, an `overlap` or density member is not `[n_ao, n_ao]` or does not pair with the coefficient members, or `basis.n_electrons` differs from the sum of the occupations; a per-frame member of a `trajectory` or `reaction.path` disagrees in shape with its `coords`, its `coords` are float32 without `metadata.precision` `"float32"` (or `precision` names the other dtype), its binary `energies` disagree with `metadata.energies`, or a trajectory carries `lattice` without `metadata.pbc` or a `true` `pbc` entry without `lattice`; the `projections` or `occupations` of a `bands` section disagree in shape with its `eigenvalues`; a `custom.<name>` member of an `atom_properties` section has no entry in the section's `meta` member; a `forces` `gradient` or `stress`, a `hessian` `matrix` or its `meta.atom_masses_amu` has the wrong shape or disagrees with the atom count of the section's structure; the `states` of an `excited_states` section repeat an `index` or list it out of order, disagree in number with `meta.n_states`, or have a contribution whose `from_mo` or `to_mo` is not a row of the reference wavefunction; the `data` of a `volume.laplacian`, `volume.vector`, `volume.complex` or `volume.series` section has the wrong rank or trailing axis, or a `volume.series` `axis` has not one `values` (or `labels`) entry per slab; a periodic grid axis is aperiodic in the grid's structure or does not span its lattice vector; a `properties.elastic` `stiffness` or `compliance` is not symmetric; a `properties.dielectric` `static` is not `electronic` + `ionic`, or its `born_charges` do not number the atoms of the section's structure; the members of a `wavefunction.bloch` section disagree in shape, its k-point weights do not sum to 1, or its structure has no periodic axis; two steps of a `reaction.scheme` share an `id`, or a `structure_ref` occurs twice within one role of a step; a `preview` member does not begin with the signature of its raster `media_type`, or an `image/svg+xml` member has no `<svg` element; or a member's `size` is not a non-negative integer. |
| `E_MEMBER_MISSING` | error | § 4.2 | A declared member `path` is not in the archive. |
| `E_SHA256_MISMATCH` | error | § 4.2 | A member's bytes do not match its `sha256`. |
| `E_JSON_INVALID` | error | § 4.2 | A JSON member is not UTF-8 JSON. |
| `E_DTYPE` | error | § 4.2 | A binary member's `dtype` is not a § 4.2 dtype name. |
| `E_BINARY_SIZE` | error | § 4.2 | A binary member's byte length differs from `itemsize × ∏(shape)`, or a member's declared `size` (JSON, binary or opaque) differs from the uncompressed size the ZIP central directory records for its entry. |
| `E_UTF8` | error | § 5.8, § 5.15 | A `run.record` `input` or `log` member, or a `preview` member of `media_type` `image/svg+xml`, does not decode as UTF-8. |
| `E_PAYLOAD_SCHEMA` | error | § 6 | A JSON member violates the payload definition (§ 5) of its section kind and role (one finding per schema error), or a one-dimensional spectrum's arrays differ in length (§ 5.4). |
| `E_STRUCTURE_PERIODICITY` | error | § 5.1 | A `structure` payload breaks the periodicity contract, or is not a JSON object. |
| `E_CHARGE_MULTIPLICITY` | error | § 5.1 | A `structure` section's `charge` and `multiplicity` are impossible for the atoms of its payload: the electron count Σ `atomic_number` − `charge` is negative, or `multiplicity` − 1 exceeds it or differs from it in parity. |
| `E_REF_UNRESOLVED` | error | § 4.1, § 5.2, § 5.3.1, § 5.4, § 5.5, § 5.8, § 5.9, § 5.10, § 5.11, § 5.12, § 5.14 | A `volume.difference` operand does not name a section in the archive; a `reaction.waypoints` `trajectory_ref` does not name a `trajectory` section; a section's `structure_ref` or `run_ref` does not name a section of kind `structure` or `run.record` respectively; a `job.spec` payload's `structure_ref` or a `forces` or `hessian` `meta.structure_ref` does not name a `structure` section; a `wavefunction.bloch` `basis.structure_ref`, in a section without a section-level `structure_ref`, does not name a `structure` section; a `vibrations` `hessian_ref` does not name a `hessian` section; an `excited_states` `wavefunction_ref` or `nto_ref` does not name a `wavefunction.gto` section, or a `density_ref` a `volume.transition_density` or `volume.difference` section; a grid `structure_ref` does not name a `structure` section; a `volume.series` axis `trajectory_ref` does not name a `trajectory` or `reaction.path` section; or a spectrum's `states_ref` or a `volume.transition_density` `excited_states_ref` does not name an `excited_states` section, or a `state_index` it qualifies names no state of that section; a `reaction.scheme` participant's `structure_ref` does not name a `structure` section; or a `run.record` `job_ref` does not name a `job.spec` section. |
| `E_FILES_INDEX` | error | § 5.8 | A `run.record` `files` index names a role with no member in the section. |
| `W_UNIT_UNKNOWN` | warning | § 4.3 | A declared unit is not a unit name: a value of a payload's `units` object (of an `excited_states` payload's `meta.units`), a one-dimensional spectrum's `x_units`, `y_units` or `broadening.units`, the `units` of a `volume.series` `axis`, or the `units` of a `{value, units}` object in `provenance` that is not a legacy spelling either. A validator with a JSON Schema engine also reports `E_SCHEMA` or `E_PAYLOAD_SCHEMA` for it. |
| `E_ROOT_INCONSISTENT` | error | § 3.6, § 5.1, § 5.3, § 5.13 | Two statements of the same quantity differ: an energy of `energies` stated for the method `provenance.method` names differs from `provenance.scf_energy`, or `energies.components.zpve` from `thermochemistry.zpve_eh`, by more than 1e-8 hartree; a `wavefunction.gto` `basis.n_electrons` differs from `provenance.n_electrons`; the `dipole` of a `properties.electric` section differs from the root `dipole_moment` by more than 1e-6 debye; or, in an archive with one `structure` section, its `charge` or `multiplicity` differs from `provenance.charge` or `provenance.multiplicity`. |
| `E_NORMALIZATION` | error | § 5.3, Appendix A.1 | The diagonal of a `wavefunction.gto` `overlap` member disagrees with the self-overlaps Appendix A.1 gives for its `basis` payload: `basis.json` coefficients are not on `N_i`-normalized primitives, or the AO order disagrees with Appendix A.2. |
