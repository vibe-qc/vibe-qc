<!-- VENDORED from the qvf repository, GOVERNANCE.md at v0.1.0-docs.1 (3b4d5548fcd10d75ead8c7ca07cd86ade4cb7586).
     DO NOT EDIT HERE. Regenerate from the tagged upstream source.
     This copy lets vibe-qc build its docs without a QVF checkout. -->

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
  `qvf-writer/` toolkit), so anyone can implement a producer or consumer,
  including in a proprietary code, with no obligation back to vibe-qc.
- The format is **producer-neutral by design**: `source.program` names any code,
  and no canonical section kind is vibe-qc-specific. vibe-qc's own
  non-standard data lives under its `x_vibeqc.*` vendor namespace, exactly like
  any other adopter's.
- Governance decisions are made **in the open** (spec + schema + registry are
  public and versioned), and the process below is the same for everyone.

As independent adopters ship QVF, stewardship is intended to broaden — see
"Evolving the governance" below.

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
- Changes to the format are recorded in the toolkit's `CHANGELOG` and reflected
  in `registry.json`.

## Version history and decisions

Governance decisions that affect the on-disk contract are recorded here, in the
open. Recording a mistake is part of the process working, not evidence against it.

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
`qvf_manifest_v3.schema.json` for these fields (and for the streaming
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
- `qvf_manifest_v2.schema.json` is retained as a frozen generated artifact so
  anything resolving its `$id` continues to validate (it is a strict superset of
  v1).

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
producer (vibe-qc's `periodic.hpp`); making it normative would have forced other
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
recorded in the CHANGELOG and the registry.

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
`spectra.epr` — but such promotions are noted as maintainer decisions in
`registry.json`.)

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
