<!-- VENDORED from the qvf repository, docs/integration_guide.md at v0.1.0-docs.1 (3b4d5548fcd10d75ead8c7ca07cd86ade4cb7586).
     DO NOT EDIT HERE. Regenerate from the tagged upstream source.
     This copy lets vibe-qc build its docs without a QVF checkout. -->

# Integrating QVF into your quantum-chemistry code

QVF is **producer-neutral**: nothing about the format is vibe-qc-specific, and
any quantum-chemistry code can emit `.qvf` files. This guide is the
code-agnostic how-to — the patterns, the data mapping, and the one contract you
must get right — so that a user of your code can open its results in **vibe-view** (the reference
viewer) or any other QVF consumer.

For the exact byte-level contract, read the QVF specification
(`spec/qvf-format-spec.md`). For build/link/API details of the reference writers,
see the [library guide](library_guide.md). For a complete worked example on a
production code, see the [ORCA integration guide](orca_integration.md).

## The whole idea in one paragraph

A `.qvf` file is a ZIP archive with a mandatory `manifest.json` plus typed
JSON/binary members, one sha256 per member. Each member is a *section* of a
canonical kind (`structure`, `wavefunction.gto`, `spectra.ir`, `bands`, …) or a
vendor extension (`x_<vendor>.*`). A consumer reads only the sections it
understands and reports the rest — so you can emit a subset today and grow it
later without breaking anyone. Producing conforming QVF is a few dozen lines on
top of a reference writer.

## Two integration models — pick either

**A. In-process.** Link the zero-dependency C++ library (`qvf::QvfWriter`) into
your code, or `import` the Python writer, and emit the archive at the end of a
run straight from your in-memory data structures. Best when you own the code and
want the richest output (grids, wavefunctions) without re-parsing.

**B. Post-processor.** A small standalone tool reads your code's *existing*
output (log file, binary wavefunction file, native JSON) and calls a writer.
Zero changes to your core; the fastest way to prototype and to support a code you
don't build yourself.

Both produce identical, interoperable archives. Many teams start with a
post-processor and later move the hot paths in-process.

## Step 1 — decide what to emit

Map your code's outputs onto canonical section kinds. This table is by *data
category*, not by code — find your data on the left, emit the kind on the right:

| Your data | QVF target | Notes |
|---|---|---|
| Geometry (Cartesians) | `structure` | positions in Å |
| Unit cell / PBC | `structure` (`lattice_vectors`, `pbc`) | row vectors, Å; `pbc` required, `dimensionality == sum(pbc)` |
| Basis + MO coefficients | `wavefunction.gto` | **see Step 2** |
| Mulliken / Löwdin / spin populations | `atom_properties` | float64 `[n_atoms]` |
| Bond orders (Mayer / Wiberg) | `bond_orders` | quantitative per-pair table |
| Normal modes | `vibrations` | frequencies cm⁻¹ |
| IR / Raman / UV-Vis / ECD / VCD | `spectra.*` | frequency + intensity |
| NMR / EPR | `spectra.nmr` / `spectra.epr` | object-shaped payload |
| SCF convergence trace | `scf_history` | per-iteration energy / error |
| Method / basis / functional / energy / convergence | root `provenance` | |
| Dipole, thermochemistry, constraints | root metadata | |
| Density / orbital / ESP / RDG grids | `volume.*` | bohr grid |
| Optimization / IRC / MD frames | `trajectory`, `reaction.path` | Å |
| Bands / DOS / phonons / EOS / Fermi surface | `bands`, `dos.*`, `phonon_*`, `equation_of_state`, `fermi_surface` | periodic |
| Level-of-theory references | `citations` | BibTeX bytes |
| Verbatim input + full log/output | `run.record` | UTF-8 text; `program` field names your code |
| Declarative job request (pending container) | `job.spec` | data, never code; pair with `provenance.run_status = "pending"` |
| Anything with no canonical kind | `x_<vendor>.*` | **see Step 4** |

**Start small.** The highest-value first cut for most codes is `structure` +
`wavefunction.gto` + whatever spectra you compute + root `provenance`. Add kinds
incrementally; partial support is a first-class feature of the format, not a
limitation.

## Step 2 — the wavefunction contract (the one thing to get right)

`wavefunction.gto` carries an atom-centered Gaussian basis + MO coefficients (the
Molden / GBW equivalent). It is the single most error-prone section, for one
reason: **primitive normalization**.

QVF requires contraction coefficients that apply to **`N_i`-normalized**
primitives — `N_i` is the *axial* norm of spec Appendix A.1, one factor per
shell derived from the total `l`:

```
χ(r) = Σ_i c_i · φ_i(r),   φ_i = N_i · (monomial) · exp(−α_i r²),
N_i = (2 α_i / π)^{3/4} · (4 α_i)^{l/2} / √((2l−1)!!),   l = l_x + l_y + l_z
```

For spherical shells and the axial Cartesian components this coincides with
unit normalization; for mixed Cartesian components (`pure: false` shells) it
deliberately does not — re-read spec Appendix A.1 before emitting those.

Integral libraries — **libint** and **libcint** (the latter is what PySCF uses),
and the storage conventions many codes inherit — keep coefficients
**pre-multiplied by `N_i`**. If you write those numbers verbatim, every rendered
orbital is quantitatively wrong (worst for tight and high-`l` shells) while still
*looking* plausible.

Both reference writers do the divide for you — you just declare which convention
your coefficients follow:

```python
w.add_wavefunction_gto(shells, mo_coefficients=C, energies=eps, occupations=occ,
                       coeffs_are_libint_normalized=True)   # divide by N_i on write
```

Also fix the **AO ordering**: spherical shells run `m = −l … +l`; Cartesian shells
use libint lexicographic order. If your code orders AOs differently for a given
`l`, permute the coefficient rows before writing. Both points are spelled out in
Appendix A of the specification.

**Verify it.** Evaluate the HOMO of H₂O/STO-3G on a grid and integrate |ψ|² — it
must come out ≈ 1 per spin. If it doesn't, your normalization or AO order is off.

## Step 3 — write, then validate

Use a reference writer (Python or C++), or reimplement the producer contract
directly from the spec (§ "Producer conformance"). Either way, **validate every
archive you produce**, and wire it into your CI:

```sh
python qvf_reader.py your_output.qvf     # non-zero exit on any problem
```

The validator checks sha256 integrity, binary byte-sizing, id uniqueness,
cross-reference resolution, and extension governance — and, with `jsonschema`
installed, full conformance against the bundled `qvf_manifest.schema.json`. A
clean report means your file opens in vibe-view and any other conforming
consumer.

## Periodic systems

- `wavefunction.gto` is supported **at Γ (k = 0) only** — the real Γ-point
  crystalline orbitals over the cell's GTO basis. Full k-resolved Bloch
  wavefunctions are out of the v1 visualization contract; emit a vendor section
  if you need them for a restart.
- Use `bands` (k-path eigenvalues), `dos.total` / `dos.projected`,
  `fermi_surface`, `phonon_bands` / `phonon_dos`, and `equation_of_state` for the
  usual periodic analyses. Densities and potentials go through `volume.*` grids.

## Step 4 — extending QVF for code-specific data

When you have data with no canonical kind, don't force it into an ill-fitting
slot. Use the **vendor namespace**: a kind `x_<yourcode>.<name>` is yours to
define, declared in the root `extensions` block:

```python
w.set_extensions({"x_yourcode": {"version": "1.0", "critical": False}})
w.add_vendor_section("x_yourcode.my_analysis", json_members={"...": ...},
                     critical=False)   # false: consumers may skip it and still open the file
```

Keep `critical: false` unless the section changes how the rest of the file must
be interpreted. When two independent codes emit the same vendor kind with a
stable shape, it becomes a candidate for **promotion to a canonical kind** via
the lightweight registry process (spec § 5.7) — that is exactly how EPR became
the canonical `spectra.epr` kind. If a payload you need is common enough to
belong in the standard, that path is open.

## Becoming an officially supported producer

Interop is a two-way street, and we actively want more producers:

1. **Emit valid QVF** for your common outputs and validate it in your CI.
2. **Self-certify** against the [conformance suite](../conformance/README.md) — a
   language-agnostic set of frozen reference archives + expected decoded values +
   a runner. Certifying a consumer is running the checks over the corpus;
   certifying a producer is emitting the same logical calculations with your code
   and checking they decode to the expected values. This is how "conforming
   producer" becomes something you can *prove*, not just assert.
3. **Get listed.** Once your code emits conforming archives, vibe-view can
   officially support your output — reach out and we'll add your code to the
   supported-producer list and, if useful, write a worked integration guide for
   it (like the [ORCA one](orca_integration.md)).
3. **Promote what you need.** If you rely on a vendor kind, or a canonical kind
   is missing a field, tell us — the format grows with its producers.

The reverse direction is also on the table: if your code has a **native output
format** you'd like vibe-view to read directly (in addition to QVF), that's a
conversation worth having too.

## Calling the writer from other languages

- **Python**: `import qvf_writer` — works anywhere Python does. The toolkit's
  `python/` directory is a pip-installable package (`pip install .`, or
  `pip install "qvf-writer[validate]"` for full JSON-Schema validation), which
  installs the writer, the reader/validator, and a `qvf-validate` CLI.
- **C++**: link the static library; `#include "qvf/qvf.hpp"`. Zero external
  dependencies, C++17.
- **C / Fortran / others**: use the bundled **C ABI** (`qvf/qvf_c.h`) — a stable
  `extern "C"` surface over one opaque handle (`qvf_create`, `qvf_add_structure`,
  …, `qvf_write`, `qvf_destroy`), with nested payloads passed as JSON strings.
  Fortran calls it via `ISO_C_BINDING` `bind(C)` interfaces. See the library
  guide's "C and Fortran" section and `cpp/examples/write_h2o_c.c`.

## Worked examples

- [ORCA integration guide](orca_integration.md) — GBW → wavefunction (with the
  normalization caveat), spectra parsing, EPR, scalar properties.
- The reference writers themselves: `python/examples/write_h2o.py` and
  `cpp/examples/write_h2o.cpp` in this toolkit are complete, runnable producers
  you can copy from.

### A note for PySCF-style Python codes

Because PySCF is pure Python and uses libcint, the Python reference writer drops
straight in. The `structure`, `spectra.*`, `atom_properties`, and `provenance`
sections are unambiguous; for `wavefunction.gto`, build the shells from the
molecule's basis and pass `coeffs_are_libint_normalized=True` (libcint stores
coefficients on un-normalized primitives, so the writer's `N_i` divide applies),
then verify with the |ψ|² integral above. This is the shape of a
few-dozen-line adapter that turns any PySCF result into a `.qvf` a user can open
in vibe-view.
