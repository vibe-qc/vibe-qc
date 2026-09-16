<!-- VENDORED from the qvf repository, spec/qvf-format-spec.md at source-80432442 (80432442f318bb679e1afcef55bea166994ea28c).
     DO NOT EDIT HERE. Regenerate from the exact upstream source commit.
     This copy lets vibe-qc build its docs without a QVF checkout. -->

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
sidecar files.

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
  `manifest.json`, no leading directory).
* Producers **SHOULD** store `manifest.json` **uncompressed** (STORE method) so
  a consumer can read the table of contents cheaply; other members **SHOULD** be
  DEFLATE-compressed. Neither is mandatory — a consumer decompresses via the ZIP
  method recorded in the central directory regardless.
* Producers **SHOULD** write `manifest.json` last (or ensure its recorded
  checksums match the already-written member bytes). Because each member's
  sha256 is embedded in the manifest, all member bytes must be finalized before
  the manifest is serialized.
* Member paths **MUST** use forward slashes, **MUST NOT** begin with `/`, and
  **MUST NOT** contain any `..` path segment. (Schema: `ZipPath`.)
* A consumer **SHOULD** enforce a per-member uncompressed-size cap to guard
  against zip-bomb inputs. The vibe-qc reference tools cap at 8 GiB
  (1 gigavoxel of float64) per member.

### 2.2 Endianness

All binary members are stored in **little-endian** byte order. QVF v1 dtype
names (§ 4.2) do not carry an explicit byte-order mark; little-endian is
assumed. Producers on big-endian platforms **MUST** byte-swap to little-endian
before writing. (An explicit byte-order tag is a candidate for a future minor
revision; see § 9.)

---

## 3. The manifest

### 3.1 Root object

```jsonc
{
  "qvf_version": 1,
  "schema_uri": "https://vibe-qc.org/spec/qvf/1/manifest.schema.json",
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
| `provenance` | no | Calculation-level metadata (§ 3.2). |
| `viewer_defaults` | no | Viewer hints: camera bookmarks, per-section render defaults, `auto_open`. |
| `thermochemistry` | no | Root thermochemistry block (§ 3.3). |
| `dipole_moment` | no | Root dipole block (§ 3.3). |
| `constraints` | no | Root geometry-constraint block (§ 3.3). |
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

### 3.2 `provenance`

Optional, open (`additionalProperties: true`). Recognized fields include
`method`, `functional`, `basis` (strings); `charge`, `multiplicity`,
`n_electrons`, `n_scf_iterations` (integers); `scf_converged` (bool);
`scf_energy`, `fermi_energy` (each a `{value, units}` object); `wall_seconds`
(number); `hostname` (string); `dimensionality` (0–3, echoing the structure's
periodic-axis count per § 5.1 — informational here, and never a substitute for
reading `pbc`). A producer emits only what it knows.

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
  absence as `converged`.

### 3.3 Root metadata blocks

These describe the calculation as a whole (not binary payloads), so they live at
the manifest root rather than as sections.

* **`thermochemistry`** — `zpve_eh`, `enthalpy_eh`, `entropy_cal_mol_k`,
  `gibbs_free_energy_eh` (Hartree / cal·mol⁻¹·K⁻¹), `temperature_k`,
  `pressure_atm`. All optional.
* **`dipole_moment`** — `total_debye`, `vector_debye` (3-vector, Debye),
  `origin`: either a conventional name (`"center_of_mass"`,
  `"center_of_nuclear_charge"`, `"origin"`) or an explicit 3-vector in **bohr**.
* **`constraints`** — `frozen_atoms` (0-based indices), `distance_constraints`
  (each `{atoms: [i, j], target_angstrom}`).

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
| `id` | **yes** | Unique within the archive. `^[A-Za-z0-9_.-]+$`. |
| `kind` | **yes** | Canonical kind (§ 5) or `x_<vendor>.*`. |
| `members` | **yes** | Map from member *role* to a JSON or binary member spec. |
| `label` | no | Human-readable display label. |
| `component` | no | Component hint (e.g. for `volume.orbital`). |
| `critical` | no | If `true`, a consumer that does not support `kind` **MUST** refuse to open the archive. Default `false` (§ 7.4). |
| `schema_uri` | no | URI of a JSON Schema for a vendor section's members. |

Some kinds add kind-specific peer fields on the section object (e.g.
`volume.difference` carries `operand_a`/`operand_b`; `reaction.waypoints`
carries `trajectory_ref`). These are documented per kind in § 5.

### 4.2 Member specs

**JSON member:**

```jsonc
{ "path": "structure/structure.json", "format": "json", "sha256": "<64 hex>" }
```

**Binary member:**

```jsonc
{
  "path": "volumes/density.dat",
  "format": "binary",
  "dtype": "float32",
  "shape": [80, 80, 80],
  "sha256": "<64 hex>"
}
```

* `format` is `"json"` or `"binary"`.
* `sha256` is the lowercase hex sha256 of the **exact bytes stored for that
  member** (the uncompressed payload). Consumers **MUST** verify it before use.
* JSON members **MUST** parse as UTF-8 JSON.
* Binary members are **raw, C-contiguous, little-endian** arrays. `dtype` is one
  of: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`,
  `float32`, `float64`. The stored byte length **MUST** equal
  `itemsize(dtype) × ∏(shape)`.
* A small set of **opaque binary members** carry no `dtype`/`shape`:
  `citations.references` (UTF-8 BibTeX bytes) and the `run.record` members
  `input` / `log` (UTF-8 text) and `attachment.*` (arbitrary bytes, § 5.8).
  Consumers skip the byte-length check for these and decode per the owning
  kind's contract.

### 4.3 Units and numeric conventions

Producers convert to these units at write time. Consumers **MUST** treat on-disk
units as the contract (they may convert for display).

| Field | Unit / convention |
|---|---|
| `structure.atoms[].position` | Ångström |
| `structure.lattice_vectors` | Ångström, row vectors |
| `volume.*.grid.origin` | bohr |
| `volume.*.grid.voxel_vectors` | bohr per grid step |
| `volume.density.data` | e / bohr³ (electron density) |
| `volume.orbital.data` | orbital amplitude on a bohr grid |
| `volume.potential.data` | hartree / e (atomic units of potential) |
| `volume.rdg.data` | dimensionless, s(r) = \|∇ρ\| / (2(3π²)^⅓ ρ^⅔) |
| `trajectory.coords`, `reaction.path.coords` | Ångström |
| `trajectory.metadata.energies`, `reaction.path.waypoints[].energy_eh` | Hartree |
| `bands.eigenvalues`, `bands.kpath.fermi`, `fermi_energy_ev` | eV |
| `spectra.ir.frequencies` / `intensities` | cm⁻¹ / km·mol⁻¹ |
| `spectra.uvvis.frequencies` / `energies_ev` | eV |
| `vibrations.metadata.frequencies` | cm⁻¹ |
| `wavefunction.gto.basis.shells[].exponents` | bohr⁻² |
| `wavefunction.gto.mo_metadata.energies` | Hartree |
| `dos.*.energies` | eV (referenced to E_F at 0.0) |
| `fermi_surface.energies` | eV (signed: E(k) − E_F) |
| `phonon_bands.frequencies`, `phonon_dos.frequencies` | cm⁻¹ |
| `equation_of_state.volumes` / `energies` | Å³ / eV |
| `equation_of_state.fit.B0` | GPa |

When adding fields, prefer unit-bearing names (`energy_eh`, `energy_ev`,
`frequency_cm`) or a `{value, units}` object.

---

## 5. Canonical section kinds

Every kind below is a `oneOf` branch in the schema. A section's `kind` selects
its required member set. The table lists each kind and its required members;
subsections give the payload shapes that matter for interoperability.

| Kind | Required members (roles) |
|---|---|
| `structure` | `structure` (JSON, `$defs/StructurePayload`); optional `bonds` (JSON) |
| `bonds` | `bonds` (JSON) |
| `bond_orders` | `bond_orders` (JSON) |
| `volume.density` / `.orbital` / `.spin` / `.elf` / `.generic` / `.potential` / `.rdg` | `grid` (JSON) + `data` (3-D float32/64 binary) |
| `volume.difference` | `grid` + `data`; optional section fields `operand_a`, `operand_b`, `description` |
| `wavefunction.gto` | `basis` + `mo_metadata` (JSON) + `mo_coefficients` **or** (`mo_coefficients_alpha` + `mo_coefficients_beta`) (float64 binary) |
| `basis.ao` | `grid` + `data`; section field `ao_metadata` |
| `atom_properties` | ≥1 of `mulliken_charge`, `loewdin_charge`, `hirshfeld_charge`, `spin_population` (float64 [n_atoms] binary) |
| `trajectory` | `metadata` (JSON) + `coords` (float64 [n_frames, n_atoms, 3]) |
| `reaction.path` | `metadata` + `coords`; optional `lattice` (periodic marker), `frame_volumes` (4-D) + `volume_grid` |
| `reaction.waypoints` | `waypoints` (JSON); section field `trajectory_ref` (REQUIRED) |
| `scan.surface` | `metadata` + `axis_a` + `axis_b` + `energies`; optional `geometries` |
| `vibrations` | `metadata` (JSON) + `displacements` (float64 [n_modes, n_atoms, 3]) |
| `spectra.ir` / `.raman` / `.uvvis` / `.ecd` / `.vcd` / `.nmr` / `.epr` / `.generic` | `spectrum` (JSON) |
| `bands` | `kpath` (JSON) + `eigenvalues` (float64 [n_spin, n_k, n_bands]); optional `projections` |
| `dos.total` | `energies` + `dos` (binary) |
| `dos.projected` | `energies` + `projections` (binary) |
| `dos.coop` / `dos.cohp` | `energies` + `projections` + `integrated` + `meta` |
| `structure.symmetry` | `data` (JSON) |
| `scf_history` | `iterations` (JSON) |
| `citations` | `references` (binary BibTeX) |
| `run.record` | ≥1 of `input`, `log` (opaque UTF-8 binary); optional `files` (JSON) + `attachment.*` (opaque binary); section field `program` (REQUIRED) |
| `job.spec` | `spec` (JSON, `$defs/JobSpecPayload`) |
| `fermi_surface` | `mesh` (JSON) + `energies` (4-D float32/64) |
| `phonon_bands` | `qpath` (JSON) + `frequencies` (binary) |
| `phonon_dos` | `meta` (JSON) + `frequencies` + `dos` (binary) |
| `equation_of_state` | `volumes` + `energies` (binary) + `fit` (JSON) |
| `topology.qtaim` | `critical_points` (JSON) |
| `x_<vendor>.*` | any mix of valid JSON/binary members |

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
(`{"pairs": [{"i": 0, "j": 1, "order": 1.0}, …]}`). If absent, a viewer may infer
bonds from covalent radii.

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

### 5.2 `volume.*`

All volume kinds share the same members: a JSON `grid` and a 3-D `data` binary
(float32 default, float64 opt-in). Grid JSON:

```jsonc
{
  "origin": [-8.0, -8.0, -8.0],
  "voxel_vectors": [[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
  "shape": [80, 80, 80]
}
```

`voxel_vectors` are three per-voxel step vectors in bohr. For data index
`(i, j, k)` the point position is `origin + i·v_i + j·v_j + k·v_k`. This is
**point-centered** data: the rendered grid has exactly `shape` points.
Non-orthogonal grids use non-zero off-diagonal components.

`volume.difference` may add `operand_a` and `operand_b` (section ids); if one is
present both **MUST** be. Sign convention: `data = operand_a − operand_b`.

### 5.3 `wavefunction.gto`

Carries an atom-centered Gaussian basis and MO coefficient matrices for
molecular wavefunctions and the **Γ-point (k = 0)** crystalline orbitals of
periodic systems. See **Appendix A** for the AO-ordering and
primitive-normalization contract — this is the single most error-prone part of
implementing a producer, and getting it wrong silently corrupts every rendered
orbital.

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
`basis.json` conventions: `structure_ref` names the `structure` section;
`shells[].center` is a 0-based atom index; `l` is the shell angular momentum;
`exponents` are in bohr⁻²; `pure: true` = spherical harmonics, `false` =
Cartesian. `mo_metadata` carries `spin`
(`"restricted"`/`"unrestricted"`), `orbital_kind`
(`"canonical"`/`"natural"`/`"localized"`), `energies` (Hartree), `occupations`
(electron count per MO), and optionally `k_point` (fractional; absent or
`[0,0,0]` = Γ).

### 5.4 Spectra

`spectra.ir/raman/uvvis/ecd/vcd/generic` each carry one JSON `spectrum` member
of the form `{"frequencies": [...], "intensities": [...]}`; units depend on kind
(§ 4.3). `spectra.nmr` is different — its `spectrum` is an object with
`isotope`, `reference`, `solvent`, `chemical_shifts` (each
`{atom_index, symbol, isotropic_shift_ppm}`), `shielding_tensors`, and
`j_couplings`. The NMR payload is intentionally loose in v1.

`spectra.epr` follows the same object-shaped pattern for electron paramagnetic
resonance. Its `spectrum` is an object carrying the `g_tensor` (with `matrix` /
`isotropic` / `principal`), a `hyperfine` list (per-nucleus `a_iso_mhz` /
`a_tensor_mhz`), and a `zero_field_splitting` (`d_mhz` / `e_mhz`). g-values are
dimensionless; hyperfine and zero-field-splitting parameters are in MHz. Like
NMR, the payload shape is intentionally loose in v1.

### 5.5 Trajectories, reactions, vibrations

`trajectory`: JSON `metadata` (atom identities + optional energies) + `coords`
float64 `[n_frames, n_atoms, 3]` in Å. `reaction.path` uses the same layout and
adds waypoint records (`{frame_index, label, kind, energy_eh?}`, `kind` ∈
reactant/transition_state/intermediate/product/point). `reaction.waypoints`
carries only annotations and references an existing `trajectory` section via the
section-level `trajectory_ref`. `vibrations`: JSON `metadata` (frequencies in
cm⁻¹) + `displacements` float64 `[n_modes, n_atoms, 3]`.

**Periodic reaction paths.** `reaction.path` may carry an optional `lattice`
binary member: float64 with shape `[3, 3]` (fixed cell — the common case) or
`[n_frames, 3, 3]` (variable cell), columns = **a, b, c** in **bohr**. Its
**presence is what marks the path periodic** — a `reaction.path` without
`lattice` is unambiguously molecular and renderers must draw no cell. This is
capability detection by member presence, per § 7.1; it does **not** involve a
`qvf_version` bump.

### 5.6 Bands, DOS, and periodic kinds

`bands`: JSON `kpath` + `eigenvalues` float64 `[n_spin, n_k, n_bands]` in eV;
optional `projections` `[n_k, n_bands, n_channels]` for fat bands (with a
`channels` array in `kpath`). `dos.total`: `energies` + `dos`. `dos.projected`:
`energies` + `projections` `[n_channels, n_points]` (+ `meta.channels`).
`fermi_surface`: JSON `mesh` + `energies` `[nk1, nk2, nk3, n_bands]`.
`phonon_bands`: JSON `qpath` + `frequencies` `[n_q, n_modes]` (cm⁻¹).
`phonon_dos`: `meta` + `frequencies` + `dos`. `equation_of_state`: `volumes` +
`energies` + JSON `fit` (`model` ∈ birch_murnaghan/murnaghan/vinet, `V0`, `E0`,
`B0`, `B0_prime`, …).

### 5.7 Analysis kinds

`atom_properties`: one or more of `mulliken_charge`, `loewdin_charge`,
`spin_population`, each float64 `[n_atoms]`. `bond_orders`: JSON
`{method, pairs: [{i, j, order, distance_ang?, symbol_i?, symbol_j?}]}`.
`topology.qtaim`: JSON `{points: [{type, position, rho, laplacian, …}],
bond_paths?: […]}` with `type` ∈ bcp/rcp/ccp/ncp. `structure.symmetry`: JSON
`data` (spglib-style summary). `scf_history`: JSON
`{iterations: [{iter, energy_eh, delta_e, diis_error}]}`. `citations`: one
binary `references` member of UTF-8 BibTeX bytes. `basis.ao`: a per-AO grid +
`ao_metadata` (quantum numbers + provenance) — see **Appendix A.3** for how a
spherical and a Cartesian component are each named.

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
* **`files`** — optional JSON index describing the stored members: a map from
  member role to `{filename, description?, media_type?, truncated?}` recording
  original filenames and context. Every key **MUST** name a member that exists
  in the section.
* **`attachment.<suffix>`** — optional auxiliary files (structure includes,
  restart data, …), opaque binary bytes with **no** encoding requirement.
  `<suffix>` matches `[A-Za-z0-9_.-]+`. Describe each in the `files` index.

Producers **SHOULD NOT** duplicate large binary result files (wavefunctions,
volumetric grids) as attachments when a canonical section kind exists for the
data — attachments are for run artefacts, not a bypass of § 5. Producers should
also remember that inputs and logs may embed local absolute paths or hostnames;
sanitizing those is the producer's (or user's) responsibility before an archive
is shared.

A consumer renders `input` and `log` as plain text. Logs can reach tens of MB;
consumers **SHOULD** size-gate rendering (the uncompressed size is available
from the ZIP central directory without reading the member) and offer the full
text on demand.

### 5.9 `job.spec` — declarative job specification

A `job.spec` section describes the calculation an archive *requests* —
before, and independent of, any results it carries. Together with a
`structure` section (§ 5.1) it makes a QVF archive an executable input: a
runner opens the container, reconstructs the job from the declarative
payload, executes it, and writes the results — and a `run.record` (§ 5.8) —
back into the same container. An archive whose `provenance.run_status`
(§ 3.2) is `"pending"` **SHOULD** carry exactly one `job.spec` section and a
`structure` section naming the input geometry.

Members:

* **`spec`** — REQUIRED JSON member, schema'd as `$defs/JobSpecPayload`.

The payload:

| Field | Required | Meaning |
|---|:--:|---|
| `job_type` | **yes** | `"molecular"` or `"periodic"` — which class of engine runs the job. A periodic job's geometry carries `pbc`/`lattice_vectors` in the `structure` section; `job_type` selects the runner, it does not restate periodicity. |
| `method` | no | Method identifier (e.g. `"rhf"`, `"rks"`, `"ccsd(t)"`, `"gfn2"`). Producer-neutral lowercase string recommended. Absent means the engine's default. |
| `basis` | no | Basis-set name (e.g. `"def2-svp"`). |
| `functional` | no | Exchange-correlation functional name, for DFT methods. |
| `charge` | no | Net charge, integer. Absent means `0`. Authoritative for the job — a consumer never infers charge from the structure. |
| `multiplicity` | no | Spin multiplicity 2S+1, integer ≥ 1. Absent means `1`. |
| `kpoints` | no | `[n1, n2, n3]` Monkhorst–Pack mesh (integers ≥ 1). Periodic jobs only. |
| `tasks` | no | Ordered list of task identifiers, e.g. `["single_point"]`, `["optimize", "hessian"]`. Absent means a single point. |
| `options` | no | Object of additional engine-specific keywords (JSON-safe values). A runner that does not recognize an option **MUST** refuse the job rather than silently drop the option. |

The payload root is open (`additionalProperties: true`) for forward growth,
but producers **SHOULD** put engine-specific keywords under `options`,
keeping the typed top level portable across codes.

A `job.spec` is declarative data, never code. Executing it involves no
evaluation of an embedded program — deliberately distinct from
`run.record.input`, which may be an arbitrary script and is a *record* of
what ran, not a request. A runner **MUST NOT** execute `run.record.input` as
a side effect of opening or running an archive.

Lifecycle: a runner that executes a pending archive updates the same
container — result sections are added, a `run.record` is embedded, and
`provenance.run_status` moves to `"converged"` or `"failed"`. The `job.spec`
section is preserved verbatim through this update, so a settled archive
still records what was asked. Multi-step workflows append further
`run.record` sections ordered by their `sequence` field.

---

## 6. Validation semantics

A conforming validator checks:

* ZIP readability and a per-member uncompressed-size cap.
* `manifest.json` exists and parses; conforms to the JSON Schema.
* Section `id` values are unique.
* Every declared member `path` exists in the archive.
* Every declared member `sha256` matches the stored bytes.
* Every JSON member parses as UTF-8 JSON.
* Every binary member's byte length equals `itemsize × ∏(shape)`.
* A `structure` section's `structure` member conforms to
  `$defs/StructurePayload`, and its periodicity contract (§ 5.1) holds: `pbc` is
  present whenever `lattice_vectors` is non-null; `lattice_vectors` is non-null
  whenever any `pbc[i]` is true; and `dimensionality`, when present, equals the
  number of true entries in `pbc`.
* `volume.difference.operand_a`/`operand_b` resolve when present.
* `reaction.waypoints.trajectory_ref` resolves to a `trajectory` section.
* `run.record`: `input` and `log` members decode as UTF-8; every key of the
  `files` index names a member that exists in the section.
* A `job.spec` section's `spec` member conforms to `$defs/JobSpecPayload`.
* Extension governance (§ 7): a `critical: true` section whose kind is neither
  canonical nor a declared vendor namespace is an error.

---

## 7. Extension and versioning model

### 7.1 `qvf_version`

Integer. v1 = `1`. A value > `1` signals breaking changes; a v1-only consumer
**MUST** refuse such files. Minor, backward-compatible growth (new optional
kinds, new optional members) does **not** bump the integer — capability is
detected via supported-kind checking and the `extensions` block.

### 7.2 Canonical kinds

Canonical kinds appear in the schema and § 5. They may grow in minor revisions;
no canonical kind may be removed or have its required members changed without a
major version bump.

### 7.3 Vendor namespace `x_<vendor>.*`

Any kind matching `x_<vendor>.<specific>` is a vendor extension. `<vendor>`
**SHOULD** be a DNS-like identifier (`x_orca`, `x_pyscf`, `x_vasp`). Vendor
members must be valid JSON/binary member specs. A consumer **MUST** list vendor
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

---

## 8. Conformance

### 8.1 Producer conformance

A conforming producer **MUST**:

* write a valid ZIP with `manifest.json` at the root;
* set `qvf_version` to `1`;
* include `source.program`, `source.version`, `source.calculation`;
* use unique section ids matching `^[A-Za-z0-9_.-]+$`;
* use canonical kinds or `x_<vendor>.*`;
* include only member paths that exist in the ZIP, with no `..` segments;
* compute sha256 over the exact stored bytes of each member;
* write numeric fields in the units of § 4.3;
* write binary payloads matching the declared dtype and shape, little-endian;
* ensure JSON members parse as UTF-8 JSON;
* write `pbc` in every `structure` payload that carries `lattice_vectors`, and
  keep `dimensionality` — if emitted at all — equal to `sum(pbc)` (§ 5.1);
* declare each `x_<vendor>` namespace in root `extensions` when any section in
  that namespace is `critical: true`;
* set `critical: true` only when consumer support is truly required.

A conforming producer **SHOULD** include `provenance` when known; include
`citations` when results depend on citable methods/bases/functionals; prefer a
specific `volume.*` kind over `volume.generic`; use a vendor namespace for
experimental payloads; and run a validator on its own output in tests.

### 8.2 Consumer conformance

A conforming consumer **MUST**:

* read `manifest.json` by name, not by ZIP entry order;
* sanity-check `qvf_version`, `source`, and `sections`;
* verify sha256 before using a member's bytes;
* interpret only kinds it supports;
* report unsupported vendor sections without corrupting or misinterpreting them;
* respect declared dtype, shape, and units;
* take per-axis periodicity from `pbc`, and never draw, replicate along, or
  measure an extent from a lattice row whose axis is non-periodic (§ 5.1);
* honor the per-section `critical` flag and the root `extensions` `critical`
  flags, refusing to open when an unsupported extension is critical.

A consumer **MAY** require specific kinds for a given operation (a band plotter
may refuse a file without `bands`; a structure viewer may open it and ignore all
scalar fields).

---

## 9. Known open items (informative)

These are acknowledged limitations of v1, not defects:

1. **Endianness tag.** dtype names do not carry byte order; v1 assumes
   little-endian. A future minor revision may add an explicit prefix.
2. **k-resolved Bloch wavefunctions.** `wavefunction.gto` covers molecular
   systems and the Γ-point only. A canonical cross-consumer k-resolved section
   is future work; producers may use `x_<vendor>.*` meanwhile.
3. **NMR schema tightening.** `spectra.nmr` payload shape is intentionally loose
   in v1.
4. **Fermi-surface meshes.** v1 assumes a Γ-centered uniform Monkhorst-Pack
   grid.
5. **Cartesian `wavefunction.gto` shells are under-exercised.** The convention
   is fully specified (Appendix A.1: one `N_i` per shell, from the total `l`),
   but `pure: false` has far less cross-consumer mileage than `pure: true`,
   and the per-component normalization trap described there has already been
   hit in practice — by the reference producer's own two readers, which is
   how it was found. Producers emitting Cartesian shells, and consumers
   reading them, should validate against Appendix A.1 explicitly rather than
   assuming a spherical-only pipeline generalizes: a d shell whose `d_xy`
   self-overlap comes out at 1 rather than 1/3 has the bug.

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

A `wavefunction.gto` section is only correct if the MO coefficients, the AO
ordering, and the primitive normalization all agree with **A.1–A.2** of this
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
