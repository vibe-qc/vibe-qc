<!-- VENDORED from the qvf repository, conformance/README.md at v0.1.0-docs.1 (3b4d5548fcd10d75ead8c7ca07cd86ade4cb7586).
     DO NOT EDIT HERE. Regenerate from the tagged upstream source.
     This copy lets vibe-qc build its docs without a QVF checkout. -->

# QVF conformance suite

A language-agnostic way to **prove** a QVF implementation is correct — for both
consumers (can you read QVF right?) and producers (do you write conforming QVF?).
This is what turns *"two implementations that happen to agree"* into *"you can
certify against the standard."*

## What's here

- **`corpus/`** — a set of frozen reference archives (`*.qvf`) paired with
  `*.expected.json` files. Each archive exercises a part of the format
  (structure, the 2-D slab periodicity contract, wavefunction with the
  normalization contract, spectra, volumes, bands, analysis, vendor extensions,
  root metadata). The archives are **fixed test vectors**: they define the
  format, not any one writer, so they are committed and only change when the
  corpus is intentionally extended.

  `structure_slab_2d.qvf` is the one to read first if you consume periodic data.
  Its third lattice row is a 15.875 Å synthesized normal — the same length as a
  typical vacuum gap — and `pbc[2]` is the only thing that says it is not a cell
  edge. A consumer that infers periodicity from lattice geometry passes every
  other archive in this corpus and fails that one.
- **`run_conformance.py`** — the reference runner. For each archive it validates
  it (schema + sha256 + semantic checks), confirms the section kinds, and decodes
  the members named by each check, comparing to the expected value.
- **`generate_corpus.py`** — how the corpus was built, using the reference Python
  writer. Run it only to (re)generate or extend the corpus.

## Certify a consumer

Point the runner at the corpus:

```sh
python run_conformance.py            # exits non-zero on any failure
```

A conforming consumer must, for every archive: open it, verify every member's
sha256, decode the members, and reproduce the values in `expected.json`. The
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

## Certify a producer

Emit the same logical calculations with **your** writer, then run the checks
against your archives. In practice: reproduce the inputs in `generate_corpus.py`
with your code, write the archives, and run `run_conformance.py` over *your*
output using the committed `expected.json` files. If your archives decode to the
expected values, your writer is conforming. (A conforming producer need not
reproduce the reference bytes — only the decoded contract.)

## In CI

The toolkit's own test suite runs this runner over the committed corpus **and**
over a freshly regenerated corpus (proving the reference writer stays a
conforming producer). See `python/tests/test_conformance.py`.
