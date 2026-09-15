# Adopting QVF: how to implement a writer in your own code

```{note}
**QVF track, part 3 of 3.**
[The QVF file format, end to end](qvf_file_format.md) covers what an archive
holds and how vibe-qc writes one;
[Running a calculation as a QVF container](qvf_job_containers.md) covers the
container round trip. This page is the producer's view from *outside* vibe-qc.
The whole section is indexed at [QVF and vibe-view](../visualization.md).
```

This tutorial is for developers of *other* quantum-chemistry codes who want to
emit QVF (`.qvf`) files: the format vibe-qc and vibe-view use to carry a
calculation's structure, wavefunction, spectra, and properties in one
self-describing container. By the end you will understand the container model,
know how to write a conforming archive with the standalone toolkit, and know how
to extend the format for data it doesn't yet cover.

Nothing here depends on vibe-qc. The toolkit is a self-contained, Apache-2.0
distribution you can vendor into any code, including a proprietary one.

```{note}
Clone the [QVF repository](https://github.com/vibe-qc/qvf) independently.
It owns the specification, JSON Schema, registry, Python and C++ reference
implementations, validator, and worked examples. The following commands use
the pinned `v0.1.0-docs.1` reference snapshot; it is not installed into vibe-qc.
```

```sh
git clone --branch v0.1.0-docs.1 https://github.com/vibe-qc/qvf.git
cd qvf
```

Install the example's Python dependencies (NumPy and jsonschema) in a dedicated
environment. Reference-toolkit use is optional for another producer: you can
implement the contract yourself, as vibe-qc does, and prove agreement through
the corpus. The toolkit, viewer, and engine do not require one another at
runtime to implement the format.

Get distributions from the [QVF releases](https://github.com/vibe-qc/qvf/releases).
The qvf-writer tarball in the core documentation downloads is a legacy
monorepo artifact awaiting companion release migration. See the
[specification](../qvf/spec.md), [library guide](../qvf/library_guide.md), and
[ORCA integration guide](../qvf/orca_integration.md).

## 1. Why a new format, and what QVF is

A single calculation today scatters its results across Cube files, XYZ, Molden,
XSF, a log file, and program-specific sidecars. QVF replaces that with **one ZIP
archive** whose members are typed, checksummed, and described by a manifest.

The design rests on six ideas: one shareable artifact; random access (a consumer
reads only the members it needs); typed payloads (declared dtype/shape + sha256);
a stable vocabulary of *section kinds*; partial support (a consumer can
understand a subset and still open the file); and producer neutrality (the format
is not vibe-qc-specific).

## 2. The container model

A `.qvf` file is a ZIP archive. The only member read by name is
`manifest.json` at the root; every other member is located through a `path`
recorded in the manifest. A minimal manifest:

```json
{
  "qvf_version": 1,
  "source": {"program": "my-code", "version": "1.0", "calculation": "h2o"},
  "sections": [
    {
      "id": "structure",
      "kind": "structure",
      "members": {
        "structure": {
          "path": "structure/structure.json",
          "format": "json",
          "sha256": "…64 hex…"
        }
      }
    }
  ]
}
```

The pieces:

- **`source`** (required) names your program, its version, and the calculation.
- **`sections`** is the list of data blocks. Each has a unique `id`, a `kind`
  from the registry (or an `x_<vendor>.*` extension), and `members` mapping a
  *role* to a member spec.
- A **member spec** is either JSON (`{path, format:"json", sha256}`) or binary
  (`{path, format:"binary", dtype, shape, sha256}`). Binary members are raw,
  C-contiguous, little-endian arrays; the byte length must equal
  `itemsize × ∏(shape)`.

Two rules make QVF robust: the **sha256** of every member is stored in the
manifest, so a consumer verifies integrity before use; and unknown `x_<vendor>.*`
sections are *reported*, never misinterpreted, so producers can extend the format
without breaking existing readers.

The manifest is not just a convention: it has a machine-readable JSON Schema,
`qvf_manifest.schema.json`, shipped in the toolkit next to the Python writer. If
you emit manifests by hand rather than through a reference writer, validate
against that schema in your own test suite. It is the same schema the toolkit's
`validate_qvf()` uses, so passing it locally means passing everywhere.

### The sha256 contract, from both sides

The checksum is the part adopters most often treat as decorative. It is not.

**As a producer**, the `sha256` you record must be the hex digest of the member's
bytes exactly as they land in the archive, computed *after* you have serialized
the payload and *before* compression. Never copy a digest forward from an
earlier revision of the data.

**As a consumer**, verify before you use, not after. A member that fails its
digest is a corrupt archive, not a warning: refuse the section rather than
rendering half-truths from it. The reference reader defaults to this, and
`read_member(member, verify=False)` exists only for the case where you have
already verified the whole file and are re-reading in a hot loop.

The [QVF consumer reference](../consumer_qvf_reference.md) spells the reading
side out in full.

### The section-kind vocabulary

A section's `kind` is what tells a consumer how to interpret it, so kinds come
from a fixed vocabulary rather than being free text. The normative catalog, every
kind with its required members, dtypes, shapes and units, is section 5 of the
[format specification](../qvf/spec.md); it is worth reading once end to end
before you decide what your code should emit.

Two practical notes:

* The catalog is grouped by family (`volume.*`, `spectra.*`, `dos.*`,
  `reaction.*`, `phonon_*`, and the singletons `structure`, `bands`,
  `trajectory`, `vibrations`, `wavefunction.gto`, `atom_properties`,
  `bond_orders`, `scf_history`, `citations`, `run.record`, `job.spec`,
  `topology.qtaim`, and friends). Emitting a family member you do not fully
  populate is worse than not emitting it: consumers are entitled to trust the
  declared members.
* What the *reference consumer* renders is a separate question from what the
  format defines. vibe-view's set is the `SUPPORTED_KINDS` frozenset in
  [`src/vibeview/kinds.py` in the separate vibe-view repository](https://github.com/vibe-qc/vibe-view/blob/main/src/vibeview/kinds.py), and the same module carries a
  `DEFERRED_KINDS` set for kinds that are written and schema-valid but have no
  renderer yet. A kind in neither set is reported as unsupported rather than
  silently dropped, which is why a partially understood archive still opens.

## 3. Writing your first archive

Grab the toolkit (link above) and write an archive with the Python reference
writer: this is the "executable spec":

```python
from qvf_writer import QvfWriter

w = QvfWriter(program="my-code", version="1.0", calculation="h2o/rhf/sto-3g")

w.add_structure([
    {"symbol": "O", "position": [0.0, 0.0, 0.1173], "atomic_number": 8},
    {"symbol": "H", "position": [0.0, 0.7572, -0.4692], "atomic_number": 1},
    {"symbol": "H", "position": [0.0, -0.7572, -0.4692], "atomic_number": 1},
])
w.add_spectrum("spectra.ir", frequencies=[1595, 3657, 3756],
               intensities=[67, 5, 42])
w.set_provenance(method="RHF", basis="STO-3G", scf_energy_eh=-74.963,
                 scf_converged=True)
w.write("h2o.qvf")
```

The C++ library mirrors this method-for-method (`qvf::QvfWriter`), builds with
`cmake -S cpp -B build`, and has **zero external dependencies**: it carries its
own ZIP, CRC-32, SHA-256, and JSON emitter, so it drops into an existing build
without new link requirements.

### A complete minimal producer

The snippet above is deliberately tiny. The toolkit also ships a full worked
producer at `python/examples/write_h2o.py`: one fictional H2O / RHF / STO-3G
calculation emitting the sections a real code would emit, namely `structure`,
`bonds`, `wavefunction.gto` (the GBW-equivalent basis plus MO coefficients, with
the normalization flag set), `spectra.ir`, `atom_properties`, `scf_history`,
provenance and `citations`.

```sh
cd <qvf-checkout>/python/examples
python write_h2o.py out.qvf      # write it
python ../qvf_reader.py out.qvf  # validate and dump it
```

Read it as the shape of a real integration: model your data onto kinds, call one
`add_*` per section, set provenance once, write. The numbers are illustrative
rather than a real SCF, so you can substitute your own without untangling any
chemistry.

## 4. Publishing a wavefunction (the one thing to get right)

The `wavefunction.gto` kind carries an atom-centered Gaussian basis plus MO
coefficients, the same content as a Molden file or an ORCA GBW. It is the most
error-prone section to implement, for one reason: **primitive normalization**.

QVF requires contraction coefficients on *`N_i`-normalized* primitives,
where `N_i = (2α_i/π)^{3/4}·(4α_i)^{l/2}/√((2l−1)!!)` is the spec's
Appendix-A.1 axial norm: one factor per shell from the total `l` (unit
normalization for spherical shells, deliberately *not* per-component for
Cartesian ones). But
integral libraries (libint, libcint, and the storage conventions many codes
inherit) keep coefficients pre-multiplied by `N_i`. If you write those
numbers verbatim, every orbital renders quantitatively wrong.

The reference writers divide it out for you: just declare which convention your
coefficients follow:

```python
w.add_wavefunction_gto(
    shells,                              # {center, l, exponents, coefficients}
    mo_coefficients=C,                   # [n_mo, n_ao], rows are MOs
    energies=eps, occupations=occ,
    coeffs_are_libint_normalized=True,   # divide by N_i on the way out
)
```

The in-shell AO order is also fixed (spherical `m = −l … +l`; Cartesian in
libint order). If your code orders AOs differently for a given `l`, permute the
coefficient rows before writing. Both points are spelled out in Appendix A of
the [specification](../qvf/spec.md).

## 5. Extending QVF for data it doesn't cover

Sooner or later you have data with no canonical kind: a bespoke fragment
analysis, an experimental property your code computes and nobody else does. Don't
force it into an ill-fitting slot; use the **vendor namespace**. A kind
`x_<vendor>.<name>` is yours to define, and a declaration in the root
`extensions` block records its contract:

```python
w.set_extensions({"x_myorg": {"version": "1.0", "critical": False}})
w.add_vendor_section(
    "x_myorg.fragment_charges",
    json_members={"fragments": [{"atoms": [0, 1, 2], "charge": -0.31}]},
    critical=False,   # viewers that don't understand it still open the file
)
```

Keep `critical: false` unless the section changes how the rest of the file must
be interpreted. When two independent codes emit the same vendor kind with a
stable shape, it becomes a candidate for promotion to a canonical kind through
the lightweight registry process (spec § 7.5); that is exactly how EPR became
the canonical `spectra.epr` kind after starting life as a vendor extension.

Three rules keep the namespace usable:

* **Pick one vendor token and keep it.** `x_myorg` is yours; every extension
  section your code emits lives under it. Changing the token later strands every
  archive you already wrote.
* **Declare it in `extensions`.** The declaration is what turns an unknown kind
  from "corrupt-looking" into "understood to be someone else's data". A consumer
  reads the `critical` flag from there to decide whether it can proceed.
* **Do not shadow a canonical kind.** If QVF has a kind for your data, use it,
  even if it means dropping a field or two. A vendor section that duplicates
  `spectra.ir` under a private name is invisible to every other consumer for no
  gain.

## 6. Validate everything you write

The toolkit ships a validator. Run it on every archive, and wire it into your CI:

```sh
python python/qvf_reader.py h2o.qvf     # non-zero exit on any problem
```

It checks sha256 integrity, binary byte sizing, id uniqueness, cross-reference
resolution, and extension governance, and, when `jsonschema` is installed, full
conformance against the bundled `qvf_manifest.schema.json`. A clean report means
your file opens in vibe-view and any other QVF consumer.

## 7. Certify against the conformance corpus

A valid archive and a *conforming implementation* are different claims. The
validator proves the file you just wrote is well-formed. The **conformance
suite** proves your code implements the format, which is the claim you want
before telling users your program emits QVF.

The suite lives at `conformance/` in the independent QVF checkout and is documented in full on the
[conformance page](../qvf/conformance.md). It is a frozen corpus of reference
archives, each paired with an `expected.json` naming the values a correct
implementation must decode from it, plus a language-agnostic runner:

```sh
cd <qvf-checkout>/conformance
python run_conformance.py        # non-zero exit on any failure
```

The archives are **fixed test vectors**: they define the format rather than any
one writer, so they change only when the corpus is deliberately extended.

**To certify a consumer**, run the corpus through your reader and reproduce
every expected value. Start with `structure_slab_2d.qvf`, the archive that
catches the most common periodic bug: its third lattice row is a synthesized
normal the same length as a typical vacuum gap, and `pbc[2]` is the only thing
that says it is not a cell edge. A reader that infers periodicity from lattice
geometry passes every other archive and fails that one.

**To certify a producer**, emit the same logical calculations with your own
writer (`generate_corpus.py` shows the inputs) and run the committed
`expected.json` checks against *your* archives. Byte-identical output is not
required; decoding to the same contract is.

Wiring the runner into your CI is the cheap version of this: it is what keeps a
conforming producer conforming across refactors.

## 8. Where to go next

- The full **[specification](../qvf/spec.md)**: every section kind, units, and
  the conformance checklists.
- The **[library guide](../qvf/library_guide.md)**: build/link/API reference for
  both writers.
- The **[ORCA integration guide](../qvf/orca_integration.md)**: a complete
  worked mapping (GBW, spectra, EPR, properties) for a production code.
- The **[integration guide](../qvf/integration_guide.md)**: the two integration
  models, the data-mapping table, and how to become an officially supported
  producer.
- The **[conformance suite](../qvf/conformance.md)**: the compliance target for
  both halves of an implementation.
- The **[governance model](../qvf/governance.md)**: how kinds are registered and
  promoted, and how the spec is versioned.
- Open your archive in **vibe-view** to see it rendered, or read the
  [consumer reference](../consumer_qvf_reference.md) to write your own reader.

That is the whole loop: model your data onto section kinds, write with the
toolkit (mind the normalization flag), extend via the vendor namespace when you
must, validate, and certify against the corpus. A conforming producer is a few
dozen lines on top of the library.
