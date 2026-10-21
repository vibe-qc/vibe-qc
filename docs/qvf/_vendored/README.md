# QVF conformance suite

A language-agnostic way to **prove** a QVF implementation is correct — for both
consumers (can you read QVF right?) and producers (do you write conforming QVF?).
This is what turns *"two implementations that happen to agree"* into *"you can
certify against the standard."*

## What's here

- **`corpus/`** — a set of frozen reference archives (`*.qvf`) paired with
  `*.expected.json` files. Each archive exercises a part of the format
  (structure, the 2-D slab periodicity contract, wavefunction with the
  normalization contract, spectra, volumes, bands with and without a spin
  axis, the DOS family, Fermi surface, equation of state, phonons with
  eigenvectors and a projected DOS, analysis with every `atom_properties`
  role, trajectories, a periodic molecular-dynamics run, reaction paths,
  relaxed scans, vibrations, vendor extensions, root metadata, declared
  units, the typed provenance fields, a job request with every optional
  field, a sketch document with a reaction scheme and a preview, and a job
  set: `provenance_full.qvf`, `job_spec_full.qvf`, `reaction_scheme.qvf`,
  `job_set.qvf`; and what producers and viewers of the vibe-qc family
  already write or read: localized orbitals with their IAO reference basis,
  `job.spec` functionals and bases that are `null`, the deprecated spellings,
  `bonds` sections that name their structure, and the viewer hints
  `camera`, `crossfade` and `energy_window`, in
  [Family data](#family-data)). Every canonical kind of
  `registry.json` appears
  in at least one of them ([Coverage](#coverage)), and a set of edge cases —
  unrestricted and Cartesian wavefunctions, a 1-D wire, a variable-cell
  reaction path, ZIP64 and data-descriptor containers — has an archive each
  ([Edge cases](#edge-cases)). The archives are **fixed test vectors**: they
  define the format, not any one writer, so they are committed and only change
  when the corpus is intentionally extended.

  `structure_slab_2d.qvf` is the one to read first if you consume periodic data.
  Its third lattice row is a 15.875 Å synthesized normal — the same length as a
  typical vacuum gap — and `pbc[2]` is the only thing that says it is not a cell
  edge. A consumer that infers periodicity from lattice geometry passes every
  other archive in this corpus and fails that one.
- **`negative/`** — the must-fail corpus: one small hand-built archive per
  validation rule of spec § 6 (duplicate ZIP entries, illegal paths, resource
  caps, reserved section ids, orphans, bad sha256, …), each paired with an
  `*.expected.json` naming the finding code it must produce (spec Appendix B).
  See [The negative corpus](#the-negative-corpus) below.
- **`legacy/`** — archives the format no longer produces but that exist in the
  wild: today one archive stamped `qvf_version: 2`. They use the negative
  format; the one archive is valid with a warning. See
  [The legacy corpus](#the-legacy-corpus) below.
- **`run_conformance.py`** — the reference runner. For each archive it validates
  it (schema + sha256 + semantic checks), confirms the section kinds, and decodes
  the members named by each check, comparing to the expected value. It then runs
  the negative corpus when `negative/` sits beside the corpus directory, and the
  legacy corpus when `legacy/` does.
- **`../cpp/examples/run_conformance.cpp`** — the same runner for the C++ reader
  (CTest `qvf_conformance`); see [The C++ runner](#the-c-runner). The JavaScript
  reader has its own, `../js/scripts/conformance.mjs`.
- **`generate_corpus.py`** — how the corpus was built, using the reference Python
  writer. Run it only to (re)generate or extend the corpus: into a scratch
  directory, copying only the new files, so that no frozen vector is rewritten.
  It writes `legacy/` next to the corpus directory it is given.
- **`negative/generate_negative.py`** — how the negative corpus was built, with
  `zipfile` and raw bytes (the reference writer refuses to produce these
  archives). Run it only to (re)generate or extend the negative corpus.

## Certify a consumer

Point the runner at the corpus:

```sh
python run_conformance.py            # exits non-zero on any failure
```

A conforming consumer must, for every archive: open it, verify every member's
sha256, decode the members, and reproduce the values in `expected.json`.
Every positive archive validates without errors; warnings are allowed and do
not fail it. `updated_in_place.qvf` is one that has them: it reports
`W_ORPHAN_MEMBER` twice by design (see [Edge cases](#edge-cases)). The
`expected.json` **check format** is deliberately simple and language-agnostic, so
a C++/Rust/Julia consumer can implement the same loop:

```jsonc
{
  "archive": "wavefunction_rhf.qvf",
  "kinds": ["structure", "wavefunction.gto"],
  "checks": [
    {"manifest_pointer": ["source", "program"], "equals": "qvf-conformance"},
    {"section": "wf", "member": "basis",
     "json_pointer": ["shells", 0, "coefficients", 0], "equals": 0.318, "tol": 1e-9},
    {"section": "wf", "member": "mo_coefficients",
     "binary_index": [0, 0], "equals": 1.0, "tol": 1e-12}
  ]
}
```

- `manifest_pointer` — walk `manifest.json` by the given keys/indices.
- `section` + `member` + `json_pointer` — resolve the member's path *through the
  manifest*, parse the JSON, then walk it.
- `section` + `member` + `binary_index` — read the binary member as a
  little-endian array of its declared dtype/shape and index into it.
- `section` + `member` + `decode: "utf8"` — read an *opaque* binary member
  (no dtype/shape: `citations.references`, `run.record` `input`/`log`) and
  decode the raw bytes as UTF-8 text. Compare with `equals` (whole text) or
  `contains` (substring must occur).
- `tol` — compare as floats within tolerance; otherwise compare exactly.
  Without `tol`, arrays and objects compare by value.

## Coverage

**Every kind in `registry.json` `canonical_kinds` appears in the `kinds` list
of at least one archive in `corpus/`.** `python/tests/test_corpus_coverage.py`
enforces it, reading the kinds from the registry at test time: a kind added to
the registry fails the test until an archive exercises it, and it also checks
each `kinds` list against the archive's manifest, so a typo cannot count as
coverage. A consumer that passes the corpus has therefore been shown every
kind the format defines.

An archive that exists for one kind is named for it. The kinds that were
without an archive when the rule was introduced:

| Archive | Kind | What its checks touch |
|---|---|---|
| `bonds.qvf` | `bonds` | the standalone connectivity section: a double bond, a single bond, and a pair that omits `order` (which means a single bond) |
| `volume_orbital.qvf` | `volume.orbital` | a signed amplitude on an anisotropic `5x4x6` float32 grid, so an axis-order mistake changes every value |
| `volume_spin.qvf` | `volume.spin` | signed spin density stored as **float64** (the opt-in dtype) |
| `volume_elf.qvf` | `volume.elf` | a dimensionless field in `[0, 1]` |
| `volume_potential.qvf` | `volume.potential` | hartree / e, positive and negative |
| `volume_rdg.qvf` | `volume.rdg` | dimensionless, non-negative |
| `volume_generic.qvf` | `volume.generic` | the escape hatch, identified by its section `label` |
| `volume_difference.qvf` | `volume.difference` | `operand_a` and `operand_b` naming two `volume.density` sections of the archive; `data = a − b` |
| `basis_ao.qvf` | `basis.ao` | Cartesian components named by `cartesian_powers` with `angular_momentum` `[2, 0]` (Appendix A.3); the data are `N · x^lx y^ly z^lz · exp(−α r²)` with the axial `N` of Appendix A.1 |
| `dos_coop.qvf` | `dos.coop` | the lower-rank shapes (`[n_pairs, n_points]`, `[n_pairs]`, no spin axis), pairs named by `meta` |
| `citations.qvf` | `citations` | an opaque binary member that decodes as UTF-8, including a non-ASCII author name |

Kinds added later bring their archive with them:

| Archive | Kind | What its checks touch |
|---|---|---|
| `volume_laplacian.qvf` | `volume.laplacian` | ∇²ρ in e/bohr⁵, negative near the nuclei and positive farther out; no `units.data` declared |
| `volume_vector.qvf` | `volume.vector` | a density gradient, float32 `[4, 5, 6, 3]`: the grid `shape` is the leading three extents, the last axis holds x, y, z |
| `volume_complex.qvf` | `volume.complex` | the hydrogen 2p orbital with m = +1, float64 `[6, 5, 4, 2]`, last axis `[re, im]` |
| `volume_series.qvf` | `volume.series` | three densities at frames 0, 2 and 4 of a five-frame trajectory: `axis` of kind `frame` whose `values` are frame indices, not slab indices, with `labels` and `trajectory_ref` |
| `reaction_scheme.qvf` | `reaction.scheme`, `preview` | a sketch saved as QVF: eight species with their chemical identity on the `structure` section (compound label, name, charge, multiplicity, SMILES, InChI, InChIKey; a radical and two ions), a three-step scheme (a forward step with a reagent, conditions and a yield; an equilibrium; a retrosynthetic step whose target is its product, one coefficient 0.5), a PNG thumbnail and an SVG image with their `media_type`, and a vendor section whose drawing is an opaque member with a `media_type` |

The property kinds of spec § 5.13 have two archives of their own, each built
from a model so that every number is derived rather than typed in:

| Archive | Kinds | What its checks touch |
|---|---|---|
| `properties_molecular.qvf` | `properties.electric` | water as three point charges: the root `dipole_moment` and the section's `dipole` state one dipole (they must agree within 1e-6 debye), a traceless symmetric Buckingham quadrupole about the centre of mass, and a diagonal static polarizability |
| `properties_periodic.qvf` | `properties.elastic`, `properties.dielectric` | a cubic CsCl-type cell: the cubic Voigt stiffness (C11, C12, C44), its compliance and the Voigt moduli; `static` equal to `electronic` + `ionic`, one Born tensor per atom (+1.1 e and −1.1 e) and a zero piezoelectric tensor |

The negative corpus holds one archive per rule of that section:
`elastic_not_symmetric.qvf`, `dielectric_static_mismatch.qvf` and
`dielectric_born_atom_count.qvf` (`E_MEMBER_MALFORMED`) and
`electric_dipole_inconsistent.qvf` (`E_ROOT_INCONSISTENT`). Every validator,
with or without a JSON Schema engine, rejects all four.

## Edge cases

Archives that exist because a consumer can pass every other archive and fail
one of these. Each also has `expected.json` checks, and
`python/tests/test_corpus_edge_cases.py` verifies the property independently of
the generator, by parsing the archive itself.

| Archive | What it proves |
|---|---|
| `wavefunction_uhf.qvf` | An **unrestricted** `wavefunction.gto` (a lithium atom, 2 α and 1 β electron): `mo_metadata` carries `alpha` and `beta` objects and no top-level `energies` or `occupations`, and the coefficients are two distinct members, `mo_coefficients_alpha` and `mo_coefficients_beta`. A reader that handles only `mo_coefficients` fails it. |
| `wavefunction_cartesian.qvf` | The **Appendix A.1 normalization contract in numbers.** A `pure: false` d shell. The vendor section `x_conformance.overlap_diagonal` stores the self-overlap of each AO, computed from the stored exponents and coefficients with the format's `N_i` (one factor per primitive from the total `l`): **1 for the axial components and 1/3 for `d_xy`, `d_xz`, `d_yz`.** A consumer recomputes them from `basis.json` and compares; one that gets 1 for `d_xy` has applied the per-component correction the format forbids. The test integrates the stored basis numerically and agrees to 1e-9. |
| `wavefunction_overlap.qvf` | **The normalization contract, checked by every validator.** CO with spherical s, p and d shells on C and a spherical s and a Cartesian d shell on O, carrying the AO `overlap` member (spec § 5.3). Its diagonal is 1 for every spherical AO and the axial Cartesian ones, 1/3 for `d_xy`, `d_xz`, `d_yz`, and `S_axial` for the O s shell, whose contraction is deliberately not normalized; `expected.json` records all 16. The validators recompute the diagonal from `basis.json` (Appendix A.1, *Verifying with `overlap`*) and accept the archive; a validator that assumes normalized contractions, or applies the Cartesian factor to spherical shells, rejects it. The test integrates the stored basis numerically and reproduces the whole matrix. |
| `wavefunction_ecp.qvf` | **An effective core potential and the electron count.** NaH with a 10-electron ECP on Na (a local part and s and p semi-local channels with radial powers), so `basis.n_electrons` = 11 + 1 − 10 = 2, which the occupations and `provenance.n_electrons` restate; also `symmetry_labels`, an `overlap` and a `density_matrix`. A consumer that counts electrons from the atomic numbers alone gets 12. |
| `structure_wire_1d.qvf` | A **1-D periodic wire**: `pbc` `[false, true, false]`, so the periodic axis is not the leading one, `dimensionality` 1, and lattice rows 0 and 2 are synthesized normals of different lengths, not cell edges. |
| `reaction_path_variable_cell.qvf` | A **variable-cell** periodic `reaction.path`: `lattice` float64 `[n_frames, 3, 3]`, columns a, b, c in bohr, a different non-symmetric cell in every frame, so reading columns as rows (or Å as bohr) gives different numbers. |
| `md_periodic.qvf` | A **periodic NPT molecular-dynamics** `trajectory`: float32 `coords` with `metadata.precision`, a per-frame `lattice` `[n_frames, 3, 3]` in the **structure convention** (rows, Å; frame 0 equals the structure's `lattice_vectors`), so a consumer that reads it like `reaction.path.lattice` gets a different cell; `velocities`, the energy `gradient`, `times`, binary `energies` restating `metadata.energies`, `cell_volumes` (the determinant of each cell) and `temperatures`. The frames move at constant velocity, so `coords`, `velocities` and `times` agree. |
| `zip64.qvf` | **ZIP64 structures on small members.** Every ZIP64 structure is present: a ZIP64 extra field in each local header and each central-directory record (sizes and local-header offset saturated to `0xFFFFFFFF`), the ZIP64 end record and locator, and a saturated end record. A reader without ZIP64 support cannot open it. Python's `zipfile` writes the ZIP64 extra field into the local header only when it is asked to force ZIP64, and leaves the central directory and end record 32-bit until a size really passes the limit, so the generator serializes this archive itself. |
| `volume_periodic.qvf` | **The endpoint rule of a periodic grid** (spec § 5.2). One periodic density on a hexagonal cell (non-orthogonal a and b), sampled CHGCAR style (`includes_endpoint: false`, 6×6×8, `voxel_vectors[a] × shape[a]` is lattice row `a` in bohr) and cube style (`includes_endpoint: true`, 7×7×9 with the same voxel vectors, so the last plane on every axis repeats plane 0). A consumer that tiles the cube-style grid without dropping that plane draws a doubled plane at every cell boundary; the checks pin the duplicated plane to plane 0. |
| `bloch.qvf` | **`wavefunction.bloch` and its Bloch-sum convention** (spec § 5.3.1, provisional). A dimerised hydrogen chain with two spin channels at Γ and X: coefficients `[n_k, n_spin, n_mo, n_ao, 2]` that are real at Γ and carry a gauge phase at X, so a reader that drops the imaginary axis, or swaps the k and spin axes, gets different numbers. The orbitals satisfy C†S(k)C = 1 for the overlap built with the phase e^{+ik·T} from the archive's own basis and cell, which `python/tests/test_wavefunction_bloch.py` checks, and the `bands` section repeats the orbital energies in eV. |
| `data_descriptor.qvf` | **Streamed members.** Every local header carries the data-descriptor flag (bit 3) with zero CRC and zero sizes, a descriptor follows each member's data, and only the central directory knows the sizes; a reader must use the central directory. All members are DEFLATEd, `manifest.json` included. |
| `job_spec_full.qvf` | A **pending job with every optional `job.spec` field.** Two structures, a bulk cell and a slab, so the geometry is only known through `structure_ref` (it names the slab, the second one); `solvation`, `dispersion`, `relativistic`, `scf`, `geometry` and `resources` carry every field, with `resources` as hints. The `provenance` of this `pending` archive describes the step that prepared it: `software[]` (one item with a `url`), `git_commit`, `n_threads`, `n_mpi_ranks` and the two UTC time stamps (spec § 3.2, § 5.9). |
| `provenance_full.qvf` | **A settled archive that uses every typed provenance field:** `software[]` (one item with a `url`, one with a vendor `x_` key), a 40-character `git_commit`, `n_threads`, `n_mpi_ranks`, `n_gpus` and `started_utc` / `finished_utc` (one with a fractional second) that agree with `wall_seconds` and with the `run.record`. The `job.spec` that asked for the run is preserved, and its `n_cores` hint differs from what the run used. |
| `updated_in_place.qvf` | **An archive updated in place** (spec § 2.3). A pending `job.spec` archive, written with its manifest last, to which a runner appended the log, a rewritten `run.record` `files` index under a new name, Mulliken charges and a new `manifest.json` with Python's `zipfile` in append mode. It holds **two `manifest.json` entries**: the last one is the manifest (`run_status` `converged`, `modified_utc` set), the first is superseded. The superseded manifest and the old `files.json` are orphans, so the validators report `W_ORPHAN_MEMBER` twice and no error. A reader that takes the first `manifest.json`, or refuses the duplicated name, fails it. |
| `job_set.qvf` | **A job set halfway through** (spec § 5.9). Two structures (water, and the hydroxyl radical with multiplicity 2), a `job.spec` for each naming its structure on the section object and in the payload; the water job has run, its `run.record` names it with `job_ref` and says `run_status: "converged"`, and the Mulliken charges it produced carry `run_ref` and `structure_ref`. No run record names the hydroxyl job, so it is pending, and `provenance.run_status` of the set is `running`. |
| `size_declared.qvf` | **`size` on every kind of member spec** (spec § 4.2) — JSON, binary, vendor binary, the opaque `citations.references` and the `run.record` `input`, `log` and `attachment` — and the root `byte_order` `"little"` (spec § 2.2). The checks compare the sizes of the members whose bytes do not depend on the producer; a JSON member's size depends on how its producer formats JSON. |

### Family data

Archives for the toolkit 0.3.0 decisions (`GOVERNANCE.md`, 2026-10-15): data
that programs of the vibe-qc family already wrote or read before the format
named it. `python/tests/test_family_drift.py` checks each property by parsing
the archive itself, and that the archive validates with no finding at all.

| Archive | What it proves |
|---|---|
| `wavefunction_localized.qvf` | **Localized orbitals and their IAO reference basis** (spec § 5.3). Two H₂ molecules 3 Å apart with one intrinsic bond orbital each (`orbital_kind` `ibo`): `mo_metadata` carries `localization_method`, `localization_reference_basis` (`minao`, the minimal basis the IAOs were built from), `atom_populations`, `centroids_bohr` and `n_centres`, and the IAO charges are the canonical `iao_charge` member of an `atom_properties` section. |
| `job_set_methods.qvf` | **`null` is not absent** (spec § 5.9). A pending job set on water: Hartree–Fock with `functional: null`, GFN2-xTB with `functional` and `basis` `null`, and PBE0/def2-SVP. A runner reads `null` as "the method takes none", never as "use your default". |
| `deprecated_spellings.qvf` | **The deprecated spellings are valid** (spec § 5.3, § 5.9). Localized orbitals whose `mo_metadata` carries `iao_charges`; the hole and electron natural transition orbitals of a state with `excitation_index` and `excitation_energy_ev`; a semiempirical `job.spec` with `basis: ""`. No validator reports anything, not even a warning. |
| `bonds_structure_ref.qvf` | **Bonds name their structure** (spec § 5.1). Water, then H₂, and two standalone `bonds` sections: `bonds_h2` names H₂ with `structure_ref`; `bonds_water` names none and belongs to the first structure, water, whose atom 2 it joins. A consumer that applies every `bonds` section to the first structure draws an O–H bond into H₂. |
| `viewer_hints.qvf` | **The hints `camera`, `crossfade` and `energy_window`** (spec § 3.4). An all-electron silicon cell whose bands and DOS hold the Si 1s states 1835 eV below the Fermi level, with an `energy_window` of `[-15, 10]` eV on both, a `camera` hint in the shape of a bookmark's camera, and a `crossfade` between two densities starting at `blend` 0.25. |

`zip64.qvf` and `data_descriptor.qvf` hold the same calculation (water:
structure with bonds, `atom_properties`, an IR spectrum) with the same member
bytes as an ordinary archive; only the container differs. The properties of
`updated_in_place.qvf` and `size_declared.qvf` are verified independently by
`python/tests/test_container_rules.py`.

## The negative corpus

A validator is only as good as what it rejects. Every archive in `negative/`
is a minimal valid archive with exactly one defect, and must be rejected with
the stable finding code (spec Appendix B) its `expected.json` names:

```jsonc
{
  "archive": "compression_ratio.qvf",
  "description": "A 64 KiB run of zeros deflated to well under 1 KiB ...",
  "limits": {"max_ratio": 100, "ratio_exempt_bytes": 4096},   // optional
  "expect": [{"code": "E_COMPRESSION_RATIO", "severity": "error"}]
}
```

- `expect` — the findings the validator **must** report, by `code` and
  `severity`. It may report more (a validator that runs the JSON Schema also
  reports `E_SCHEMA` for several of these archives), but never fewer. An
  expectation of a code that only the JSON Schema check reports
  (`E_SCHEMA`, `E_PAYLOAD_SCHEMA`) binds only a validator that runs the
  schema; a validator that does not (it reports `W_SCHEMA_SKIPPED`) is held to
  the other expectations alone, so `eos_model_unknown.qvf`, whose only defect
  is a payload the schema rejects, is valid to it. The Python validator runs
  the schema through `jsonschema` when it is installed; the C++ and
  JavaScript validators always run it, through a checker generated from the
  schema (`scripts/gen_kinds.py`), unless it is disabled.
- An archive **passes** when every expected finding is present and the
  verdict matches: invalid (`ok` false) when any expected finding is an error;
  valid (`ok` true) when all of them are warnings — `orphan_member.qvf` is the
  one warning-only archive, because orphans do not invalidate an archive.
  `member_size_mismatch.qvf` declares a member `size` that disagrees with its
  ZIP entry (`E_BINARY_SIZE`); `byte_order_big.qvf` declares `byte_order`
  `"big"`, which only the JSON Schema rejects (`E_SCHEMA`), so a validator
  without a schema engine accepts it. A duplicated entry name other than
  `manifest.json` is `zip_duplicate_entry.qvf`; a duplicated `manifest.json`
  is an archive updated in place and lives in the positive corpus
  (`updated_in_place.qvf`).
  `unit_unknown.qvf` expects the warning `W_UNIT_UNKNOWN` and the schema-only
  error `E_PAYLOAD_SCHEMA`: a validator without a JSON Schema engine is held to
  the warning alone, so the archive is valid to it.
- `job_spec_structure_ref_wrong_kind.qvf` expects `E_REF_UNRESOLVED` from every
  validator (a `job.spec` payload's `structure_ref` names a `bonds` section),
  and so does `run_record_job_ref_wrong_kind.qvf` (a `run.record` `job_ref`
  names a `structure` section);
  `provenance_git_commit_not_hex.qvf` and `provenance_started_utc_not_utc.qvf`
  expect the schema-only `E_SCHEMA` (a `git_commit` with a non-hexadecimal
  character, a `started_utc` with a `+02:00` offset instead of `Z`), so a
  validator without a JSON Schema engine accepts them, like
  `metadata_uuid_invalid.qvf`.
- `limits` — optional overrides of the validator's resource caps (spec § 6),
  keyed like the reference `Limits` (`max_member_bytes`, `max_total_bytes`,
  `max_ratio`, `ratio_exempt_bytes`, `max_json_bytes`, `max_json_depth`). The
  cap archives use deliberately lowered limits so that they stay tiny: under
  the default limits they are valid, which proves that the cap, not the
  archive, is what fails. A runner passes the overrides to its validator.

`run_conformance.py` runs it after the positive corpus; `--negative DIR`
points it elsewhere and `--no-negative` skips it (and `legacy/`, below):

```sh
python run_conformance.py                              # Python validator
ctest --test-dir <build> -R qvf_negative_corpus        # C++ validator
```

The reference validators report, for every archive, exactly the codes in
`expect` apart from the JSON Schema codes `E_SCHEMA`, `E_PAYLOAD_SCHEMA` and
`W_SCHEMA_SKIPPED` (the schema check also reports `E_SCHEMA` for several
archives that name another code), and every expected schema code is present,
so they agree code for code; `python/tests/test_schema_checker_parity.py`
also requires the C++ and JavaScript validators to report the same full code
set as the Python one, the schema findings at the same locations. A
third-party validator certifies the same way: run its validator over
`negative/` and check `expect`.

The sketch-document rules have their own archives:
`structure_charge_impossible.qvf` (water with multiplicity 2,
`E_CHARGE_MULTIPLICITY`), `structure_charge_provenance.qvf` (the only
structure disagrees with `provenance`, `E_ROOT_INCONSISTENT`),
`reaction_scheme_ref_wrong_kind.qvf` (`E_REF_UNRESOLVED`),
`reaction_scheme_step_id_duplicate.qvf` and
`reaction_scheme_participant_repeated.qvf` (`E_MEMBER_MALFORMED`),
`preview_signature_mismatch.qvf` (JPEG bytes declared `image/png`,
`E_MEMBER_MALFORMED`) and `preview_svg_not_utf8.qvf` (`E_UTF8`), which every
validator rejects, and the schema-only `reaction_scheme_energies_reserved.qvf`
(a step carrying the reserved key `energies`, `E_PAYLOAD_SCHEMA`) and
`preview_media_type_missing.qvf` (`E_SCHEMA`).

The family-data rules (spec § 3.4, § 5.1, § 5.3) have their own archives:
`bonds_structure_ref_unresolved.qvf` and `bonds_structure_ref_wrong_kind.qvf`
(a standalone `bonds` section whose `structure_ref` names nothing, or a
`job.spec`, `E_REF_UNRESOLVED`), `bonds_atom_out_of_range.qvf` (a pair that
joins atom 2 of the two-atom structure its `structure_ref` names, although the
first structure has that atom, `E_MEMBER_MALFORMED`), `reserved_id_camera.qvf`
and `reserved_id_crossfade.qvf` (`E_RESERVED_ID`), which every validator
rejects, and the schema-only `payload_localization_reference_basis_empty.qvf`
(`E_PAYLOAD_SCHEMA`).

The wavefunction rules of spec § 5.3 have three archives:
`wavefunction_normalization.qvf` (an `overlap` whose `d_xy` diagonal is 1, three
times the Appendix A.1 value, `E_NORMALIZATION`), `wavefunction_n_electrons.qvf`
(`basis.n_electrons` disagreeing with the occupations) and
`wavefunction_complex_rank2.qvf` (`complex: true` with rank-2 coefficients),
both `E_MEMBER_MALFORMED`. All three validators implement the checks.

Archives that break only a member payload's `$defs` definition (the
`payload_*.qvf` archives, `spectrum_length_mismatch.qvf`,
`qtaim_point_type.qvf`, `scf_history_unknown_key.qvf`,
`unit_contradicts_fixed.qvf`; spec § 5) expect nothing but
`E_PAYLOAD_SCHEMA`. A validator that does not run the JSON Schema reports
`W_SCHEMA_SKIPPED` instead and cannot produce that code (the equal-length rule
of `spectrum_length_mismatch.qvf` is a semantic check that reports it anyway),
so every runner (Python, C++, JavaScript) drops the schema-only expectations
for it and judges the verdict on the other codes; the archive must then
validate with no other error at all. The C++ and JavaScript runners show this
behaviour with `--no-schema-check` (CTest `qvf_conformance_no_schema`).

## The legacy corpus

`legacy/` holds archives that producers no longer write but that exist in the
wild. It uses the negative format (`expect` codes, optional `limits`) and every
runner treats it like `negative/`: `run_conformance.py` picks up a `legacy/`
beside the corpus directory (`--legacy DIR` points it elsewhere, `--no-legacy`
skips it) and reports it on its own line, `legacy conformance: N/M archives
judged as expected`. An archive whose expectations are all warnings must
validate (`ok` true) with them.

Today it holds `reaction_path_v2.qvf`: a periodic `reaction.path` stamped
`qvf_version: 2`, as vibe-qc v0.10 – v0.15 wrote it before the 2026-07-10
decision withdrew that version (`GOVERNANCE.md`). Apart from the version and
the v2 `schema_uri` it is a valid v1 archive (a test restamps it as version 1
and validates it clean).

Its expectation is the warning `W_VERSION_WITHDRAWN` (decided 2026-10-14,
`GOVERNANCE.md`): spec § 3.1 and `registry.json` (`readable_versions: [1, 2]`)
say consumers **SHOULD** read `qvf_version: 2` as 1, so the validators check
the archive as version 1, the JSON Schema check included, and accept it with
the warning, which records that producers must not write that version. Any
other version is still `E_VERSION` (`negative/version_unsupported.qvf`).

## Certify a producer

Emit the same logical calculations with **your** writer, then run the checks
against your archives. In practice: reproduce the inputs in `generate_corpus.py`
with your code, write the archives, and run `run_conformance.py` over *your*
output using the committed `expected.json` files. If your archives decode to the
expected values, your writer is conforming. (A conforming producer need not
reproduce the reference bytes — only the decoded contract.)

## The C++ runner

`cpp/examples/run_conformance.cpp` implements the whole check format above for
the C++ reader: `manifest_pointer`; `section` + `member` with `json_pointer`,
`binary_index` (decoded from the member's little-endian dtype) or `decode:
"utf8"` with `equals` or `contains`; `tol`; and the negative format (`expect`
codes, `limits`), with the rules of the C++ negative-corpus test: the
validator runs the generated JSON Schema check, so every expectation binds;
`--no-schema-check` disables the check and drops the schema-only
expectations. It prints the same lines as the Python runner and exits
non-zero on any failure:

```sh
run_conformance CORPUS_DIR [--negative DIR] [--legacy DIR] [--no-schema-check]

ctest --test-dir <build> -R qvf_conformance            # all three corpora
./<build>/run_conformance conformance/corpus --negative conformance/negative \
    --legacy conformance/legacy                          # with the report
```

It exits 77 (a skipped CTest) when `CORPUS_DIR` is absent, as in a standalone
copy of `cpp/`.

## In CI

The toolkit's own test suite runs the reference runner over the committed
corpus **and** over a freshly regenerated corpus (proving the reference writer
stays a conforming producer), and runs the negative and legacy corpora through
both the Python validator and the C++ validator (CTest `qvf_negative_corpus`).
The C++ jobs run the C++ runner over all three corpora (CTest `qvf_conformance`,
and the binary itself after `ctest`), and the sanitizer jobs read every
positive archive with `read_qvf`; the JavaScript job runs its runner over them
too. Every reader opens `zip64.qvf` and `data_descriptor.qvf`. See
`python/tests/test_conformance.py`, `test_corpus_coverage.py` and
`test_corpus_edge_cases.py`.

The `volume_spin_zero.qvf` positive case contains a reference-writer 64³
float64 zero spin-density grid: a ratio over 1000 is valid. `compression_ratio`
is an opt-in application-policy test (`max_ratio` is disabled by default).
`entry_size_understated` and the absolute size-cap negatives guard bounded
decompression; `crc_mismatch_stored` and `crc_mismatch_deflated` reject corrupt
checksums even without an optional member SHA-256.
