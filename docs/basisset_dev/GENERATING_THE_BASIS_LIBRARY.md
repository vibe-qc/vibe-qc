# Generating and reviewing the bundled basis library

The bundled basis library is a store of QVF-Basis records. Each
`python/vibeqc/basis_library/qvf/<name>.qvf.json` is the source of truth
for one basis set: shells, effective core potentials, references and
provenance in one schema-validated file. `python/vibeqc/basis_library/basis/`
is **rendered** from those records: the `.g94` orbital files libint reads
and the `.ecp` sidecars libecpint reads. The rendering is committed so the
wheel ships it and `LIBINT_DATA_PATH` can point at it, and it is pinned
byte for byte to the records by `tests/test_basis_qvf_parity.py`. Nothing
edits the text by hand.

This page documents the generation pipeline and the review checklist that
every change touching `basis_library/` should pass.

## Pipeline at a glance

```
python/vibeqc/basis_library/qvf/<name>.qvf.json        (records, the source of truth)
                       │
                       ▼
     vibeqc.basis_toolkit.library.render_library       (scripts/render_basis_library.py,
                       │                                 vibe-qc-basis render)
                       ▼
python/vibeqc/basis_library/basis/<name>.g94 + .ecp    (committed rendering, pinned to the records)
build/basis_library/basis/<name>.g94 + .ecp            (runtime overlay with a content receipt,
                                                        scripts/setup_basis_library.sh)
```

### The record

A record carries, per element, the shells (`angular_momentum`,
`harmonic_type`, primitives in the G94 coefficient convention) and, where
the set is valence-only, the ECP (`n_core_electrons`,
`angular_momentum_max`, one channel per `l`, each term with a
Gaussian-format `power`; see #207 for the convention). Its `references`
list the originating publications with DOIs, and its `provenance` block
(schema 1.1.0) says what the record is:

| field | meaning |
|---|---|
| `kind` | `vendor` (bundled from a distribution such as libint's bundle or the Bredow archive, possibly completed with reviewed BSE blocks), `bse` (a whole Basis Set Exchange record), `hand-written`, `vibe-basis` |
| `upstream` | one entry per source, with version, URL, identifier and which elements it supplied |
| `retrieval_date` | when the data was retrieved or generated |
| `content_sha256` | SHA-256 of the canonical JSON of `{elements, ecps}`; renaming or re-describing a record does not change it |
| `citations` | citation keys (`database.toml` entries) per element range; the route table in `database.toml` must equal their union |

### The rendering

`vibeqc.basis_toolkit.exporter_g94` writes every number with `repr`, the
shortest decimal that parses back to the identical double, so a rendering
loses no precision (#291). The header of each file is derived from the
record: the content hash, the provenance line, one `Originating
publication` line per reference (with its DOI), the element coverage and
the record's description. An ECP record renders its potentials to the
sidecar only, local channel first, projectors ascending; libint never sees
an `-ECP` block.

### Tools

* **`scripts/render_basis_library.py`**: standard library only, loads the
  toolkit by path, so it works before the native core is built.
  `--target DIR` renders; `--check` compares the committed rendering with
  the records and exits 1 on any difference, naming the files.
* **`scripts/setup_basis_library.sh`**: renders the runtime overlay under
  `build/basis_library/` and writes the content receipt the runtime checks
  before using it (#409). It refuses when the committed rendering is out of
  step with the records, because the receipt would otherwise make every
  import refuse the overlay it just built.
* **`vibe-qc-basis`** (`vibeqc.basis_toolkit.cli`): `convert` turns BSE
  JSON, Gaussian94, ORCA, NWChem or CRYSTAL input into a record; `render`
  and `check` are the two commands above; `validate` checks records against
  the schema; `inspect` prints a summary.
* **`scripts/basisset_dev/pob_basis_verify.py`**: compares the per-element
  CRYSTAL sources under `sources/` with the pob records and their
  renderings.

## Adding or changing a bundled basis

1. Produce the record with `vibe-qc-basis convert <input> -o
   python/vibeqc/basis_library/qvf/<name>.qvf.json`. `<name>` is the
   lower-case identifier users pass to `BasisSet`; the record's `name`
   must equal the file stem.
2. Fill `references` (DOIs) and `provenance` (`kind`, `upstream`,
   `retrieval_date`, `citations`). A BSE import brings references along;
   the rest is yours to state. The parity test refuses a record without a
   kind, an upstream or a date.
3. Render the committed copy:
   `./scripts/setup_basis_library.sh --target-dir python/vibeqc/basis_library`.
4. Route the citation keys in `python/vibeqc/output/citations/database.toml`
   (`[routes.basis_sets]` must list exactly the record's citation keys) and
   record the component in `docs/license.md`.
5. If the set is valence-only anywhere, add its family or threshold to
   `basis_library/registry.toml`, and extend `EXPECTED_ECP_BASES` in
   `tests/test_basis_ecp_sidecars.py`.
6. Run `python scripts/render_basis_library.py --check`, then the basis
   lanes: `tests/test_basis_qvf_parity.py`, `tests/test_basis_integrity.py`,
   `tests/test_basis_ecp_sidecars.py`, `tests/test_basis_citation_coverage.py`,
   `tests/test_basis_registry.py`.

## Review checklist

* `python scripts/render_basis_library.py --check` passes.
* The record names its publications and provenance; the rendered header
  names the same DOIs.
* The citation route table equals the record's citation keys.
* No `.g94` or `.ecp` changed without its record changing, and vice versa.
* `scripts/basisset_dev/diff_basis_library.py` summarises the data diff in a
  form a reviewer can scan.
* No private machine path or private term in any record or rendering
  (`tests/test_basis_no_maintainer_paths.py`).

## History

Until 2026-10 the pipeline ran the other way: libint's bundle plus a
`custom/` directory of hand-placed `.g94` files were copied into
`basis/`, a splitter moved ECP blocks into sidecars, and a converter
produced the QVF records from the text. The converter had stopped being
run: fifteen records carried SP shells in the interleaved layout of an
earlier importer, and 254 were named `<g94-import>`. The records were
regenerated from the shipped text (numerically identical, every value),
seeded with the provenance above by
`scripts/basisset_dev/seed_qvf_records_1_1.py`, and made authoritative
(#406, #407, umbrella #295). `custom/`, the splitter and the converter are
gone.
