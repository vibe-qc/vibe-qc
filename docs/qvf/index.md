# The QVF format toolkit

**QVF** (`.qvf`, *Quantum Visualization Format*) is a ZIP-based container that
keeps one calculation's structure, wavefunction, scalar fields, spectra, bands,
trajectories, provenance, and viewer hints in a single random-access file. It is
the format vibe-qc writes and [vibe-view](../user_guide/vibe_view.md) reads, but
it is designed for the **whole** quantum-chemistry ecosystem, not just vibe-qc.

This section is for developers of *other* codes who want to **emit QVF** from
their own program. It hosts the normative specification, the reference writer
libraries, and integration guidance.

```{note}
The source of truth is the independent, Apache-2.0
[QVF repository](https://github.com/vibe-qc/qvf): specification, JSON
schema, registry, conformance corpus, and reference implementations. GitLab
requires authorized access. The [public GitHub snapshot](https://github.com/vibe-qc/qvf)
can be cloned anonymously with `git clone https://github.com/vibe-qc/qvf.git`.

vibe-qc implements QVF independently and is validated against the published
specification and corpus. It does not import or link the reference toolkit.
The viewer and reference toolkit are independent implementations too; their
agreement through the corpus is the interoperability check.
```

The six included reference pages are vendored from the immutable QVF source
commit `80432442f318bb679e1afcef55bea166994ea28c`. Each header records the
original source path and commit. Do not edit `docs/qvf/_vendored/`; change QVF
upstream, then deliberately re-vendor. This documentation update changes no
format, reference implementation or interoperability pin: `QVF_TAG` remains
`v0.1.0-docs.1` in `.gitlab-ci.yml`. The snapshot builds without a sibling
QVF checkout.

For reference-toolkit distributions, use the
[QVF release page](https://github.com/vibe-qc/qvf/releases).
The legacy pre-split writer archive is preserved in the maintainer's external
private archive and is no longer distributed from this source tree.

## What's here

- **[Format specification](spec.md)**: the normative QVF v1 spec: archive model,
  manifest, the full section-kind catalog, units, extension governance, and the
  producer/consumer conformance checklists. Ships with the machine-readable
  `qvf_manifest.schema.json`.
- **[Integration guide](integration_guide.md)**: **start here if you write
  another QC code.** The code-agnostic how-to: the two integration models, the
  data-mapping table, the wavefunction contract, and how to become an officially
  supported producer.
- **[Library guide](library_guide.md)**: build, link, and call the Python and
  C++ reference writers/readers; the section-kind cheat sheet.
- **[ORCA integration guide](orca_integration.md)**: a complete worked mapping
  from a production code's data (GBW-style wavefunction, spectra, EPR, scalar
  properties) onto QVF sections.
- **[Conformance suite](conformance.md)**: the compliance target. A frozen corpus
  of reference archives paired with expected-value files, plus a
  language-agnostic runner, so you can *prove* your reader or writer is correct
  rather than that it happens to agree with one other implementation.
- **[Governance](governance.md)**: how QVF is stewarded, versioned, and evolved:
  an open standard created and stewarded by vibe-qc, producer-neutral by design,
  with a public registry and a vendor→canonical promotion process.
- **Tutorial:** [Adopting QVF: how to implement a writer in your own code](../tutorial/qvf_adapt_and_writer.md).

## Why adopt QVF

- **One artifact.** A calculation travels as one file instead of a scatter of
  Cube / XYZ / Molden / XSF / log / sidecar files.
- **Typed and checksummed.** Every member declares its dtype, shape, and a
  sha256; consumers verify integrity before use.
- **Partial support is safe.** A consumer understands the kinds it knows and
  reports the rest; unknown vendor sections are never misinterpreted.
- **Extensible.** The `x_<vendor>.*` namespace lets you ship data QVF doesn't yet
  standardize, with a clear path to promotion.

## Reference consumer

Anything the toolkit writes opens in **vibe-view**, the reference GPU viewer, and
validates against vibe-qc's `validate_qvf()`. To write your own reader, start
from the [consumer reference](../consumer_qvf_reference.md).

```{toctree}
:maxdepth: 1
:hidden:

spec
integration_guide
library_guide
orca_integration
conformance
governance
```
