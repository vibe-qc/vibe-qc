<!-- VENDORED from the qvf repository, docs/orca_integration.md at source-80432442 (80432442f318bb679e1afcef55bea166994ea28c).
     DO NOT EDIT HERE. Regenerate from the exact upstream source commit.
     This copy lets vibe-qc build its docs without a QVF checkout. -->

# Integrating a QVF writer into ORCA

This is the concrete worked example behind the code-agnostic
[integration guide](integration_guide.md) — read that first for the general
patterns; this page maps them onto ORCA specifically and calls out the one
subtlety that will bite you if you skip it: **GTO primitive normalization**. The
approach applies to any code with a GBW-style wavefunction file and text/JSON
property output.

The goal: after an ORCA job runs, produce a single `result.qvf` that carries the
geometry, the wavefunction (basis + MOs), spectra, and scalar properties, so the
user can open one file in vibe-view (or any QVF consumer) instead of juggling
`.gbw`, `.out`, `.property.txt`, and `.hess`.

## Where the QVF writer sits

Two integration points, either works:

1. **In-process (C++).** Link `qvf::QvfWriter` (see
   [`library_guide.md`](library_guide.md)) into ORCA and emit the archive at the
   end of a run, reading straight from ORCA's in-memory structures. Zero external
   dependencies, so it does not complicate the ORCA build.
2. **Post-processor (Python or C++).** A small tool that reads the `.gbw` and the
   property/`.json` outputs ORCA already writes, and calls the writer. This keeps
   ORCA's core untouched and is the quickest way to prototype.

ORCA already emits a rudimentary JSON property dump and the binary GBW; the value
QVF adds is a *typed, checksummed, self-describing* container that consumers read
without parsing `.out` text.

## The mapping at a glance

| ORCA data | QVF target | Notes |
|---|---|---|
| Geometry (Cartesians) | `structure` | positions in Å |
| GBW: atoms + basis + MO coefficients | `structure` + `wavefunction.gto` | see normalization below |
| Mulliken / Löwdin population | `atom_properties` | float64 `[n_atoms]` |
| Mayer bond orders | `bond_orders` | `{method:"mayer", pairs:[…]}` |
| IR spectrum (from `.hess` / output) | `spectra.ir` | cm⁻¹ / km·mol⁻¹ |
| Raman activities | `spectra.raman` | |
| UV/Vis, CD (TD-DFT) | `spectra.uvvis`, `spectra.ecd` | eV x-axis |
| NMR shieldings / couplings | `spectra.nmr` | object payload |
| EPR g/A/D tensors | `spectra.epr` | object payload; see EPR below |
| SCF iteration energies | `scf_history` | |
| Dipole moment | root `dipole_moment` | Debye |
| Thermochemistry (freq job) | root `thermochemistry` | Hartree |
| Method / basis / functional / energy | root `provenance` | |
| Level-of-theory references | `citations` | BibTeX bytes |
| The `.inp` + `.out` themselves | `run.record` | `program: "orca"`; makes the archive self-contained |

## GBW → `structure` + `wavefunction.gto`

The GBW file holds the geometry, the atom-centered Gaussian basis, and the MO
coefficient matrix — exactly what `wavefunction.gto` carries. Emit the geometry
as a `structure` section and reference it from the wavefunction's
`structure_ref`.

Build each shell as `{center, l, exponents[], coefficients[], pure}` and pass the
MO coefficient matrix as `[n_mo, n_ao]` (rows are MOs). For a spin-unrestricted
run, provide separate α and β matrices.

### The normalization trap (read this)

QVF requires contraction coefficients that apply to **`N_i`-normalized**
primitives — `N_i` is the *axial* norm of spec Appendix A.1, one factor per
shell derived from the total `l`:

```
χ(r) = Σ_i c_i · φ_i(r),   φ_i = N_i · (monomial) · exp(−α_i r²),
N_i = (2 α_i / π)^{3/4} · (4 α_i)^{l/2} / √((2l−1)!!),   l = l_x + l_y + l_z
```

For spherical shells and the axial Cartesian components this coincides with
unit normalization; for mixed Cartesian components it deliberately does not
(see the Cartesian caveat below).

Integral engines — libint, libcint, and the storage conventions many codes
(including GBW-style formats) inherit — keep coefficients **pre-multiplied by
`N_i`**, i.e. the stored number is `c_i · N_i`, multiplying an *un-normalized*
primitive. If you write those numbers verbatim, every rendered orbital comes out
quantitatively wrong (worst for tight and high-`l` shells) while still *looking*
plausible.

Both reference writers handle this for you — set the flag and pass your
engine-native coefficients:

```cpp
WavefunctionGTO wf;
wf.coeffs_are_libint_normalized = true;   // divide by N_i on the way out
wf.shells = { /* center, l, exponents, engine-native coefficients, pure */ };
wf.mo_coefficients = { n_mo, n_ao, coeff_data };  // row-major, rows are MOs
w.add_wavefunction_gto(wf);
```

If ORCA's in-memory coefficients are already on `N_i`-normalized primitives,
leave the flag `false`. When unsure, verify: evaluate the HOMO of H₂O/STO-3G on
a grid and integrate |ψ|² — it must come out ≈ 1 per spin.

If you emit Cartesian (`pure: false`) shells, re-read spec Appendix A.1 first:
`N_i` is one factor per shell derived from the total `l`, so mixed components
such as `d_xy` are deliberately *not* individually unit-normalized. Verify on a
Cartesian d shell specifically — a spherical-only check will not catch a
per-component normalization error.

### AO ordering

QVF's in-shell AO order is fixed (spec Appendix A.2): spherical shells are
ordered `m = −l … +l`; Cartesian shells use libint lexicographic order. If
ORCA's internal ordering differs for a given `l` (e.g. a different `p` or `d`
sequence), permute the coefficient **rows** into QVF order before writing.
Getting this wrong rotates orbitals within a shell.

## Spectra

ORCA writes spectra into the output and dedicated files (`.hess` for
vibrational/IR/Raman; TD-DFT blocks for UV/Vis and CD). Parse the frequency /
intensity columns and emit the matching `spectra.*` kind:

```cpp
Spectrum ir;
ir.frequencies = /* cm^-1 */;
ir.intensities = /* km/mol */;
w.add_spectrum("spectra.ir", ir);
```

`spectra.nmr` is object-shaped (isotope, reference, per-nucleus shifts and
optional tensors) — build it as a `qvf::Json` and use the JSON overload of
`add_spectrum`.

## EPR — the canonical `spectra.epr` kind

EPR is a **canonical** kind (`spectra.epr`), paralleling `spectra.nmr`: a single
object-shaped `spectrum` member carrying the g-tensor, per-nucleus hyperfine (A)
tensors, and the zero-field-splitting (D) tensor. vibe-view renders it in a
dedicated panel. Emit it like any other object-shaped spectrum:

```cpp
Json epr = Json::object();
Json g = Json::object();
g.set("principal", Json::from_doubles({gx, gy, gz})).set("isotropic", g_iso);
epr.set("g_tensor", g);
epr.set("hyperfine", /* array of {atom_index, symbol, isotope, a_iso_mhz, a_tensor_mhz} */);
Json zfs = Json::object();
zfs.set("d_mhz", d_value).set("e_mhz", e_value);
epr.set("zero_field_splitting", zfs);
w.add_spectrum("spectra.epr", epr);           // Python: w.add_spectrum("spectra.epr", payload=epr)
```

Units follow EPR convention: g-values dimensionless, hyperfine and
zero-field-splitting parameters in MHz. The payload shape is intentionally loose
in v1 — emit whichever subset you computed.

**When to still reach for the vendor namespace.** `spectra.epr` covers the common
g/A/D parameters. For genuinely non-standard data that has *no* canonical kind
(a bespoke analysis, an experimental property), use `x_<vendor>.*` and declare it
in the root `extensions` block:

```cpp
Json ext = Json::object();
Json oe = Json::object();
oe.set("version", "1.0").set("critical", false);   // false: viewers may skip it
ext.set("x_orca", oe);
w.set_extensions(ext);
Json members = Json::object();
members.set("data", /* your payload */);
w.add_vendor_section("x_orca.my_property", members, /*critical=*/false);
```

Set `critical: false` unless the section changes how the rest of the file must be
interpreted. When a vendor kind stabilizes across two independent producers it
can be promoted to canonical via the registry process (spec § 7.5) — which is
exactly how `spectra.epr` itself was standardized.

## Scalar properties and provenance

```cpp
// atom_properties (Mulliken + Löwdin)
w.add_atom_properties(/*mulliken=*/qM, /*loewdin=*/qL);

// root dipole (Debye) and thermochemistry (Hartree)
Json dip = Json::object();
dip.set("total_debye", mu).set("vector_debye", Json::from_doubles({mx, my, mz}))
   .set("origin", "center_of_mass");
w.set_dipole_moment(dip);

// provenance: method / basis / functional / energy / convergence
Json prov = Json::object();
prov.set("method", "DFT").set("functional", "B3LYP").set("basis", "def2-TZVP")
    .set("scf_converged", true);
Json e = Json::object(); e.set("value", E_total).set("units", "Eh");
prov.set("scf_energy", e);
w.set_provenance(prov);
```

## Validate before you ship it

Run the bundled validator on every archive your integration produces, in your CI:

```sh
python python/qvf_reader.py result.qvf
```

It verifies sha256 integrity, binary sizing, id uniqueness, reference
resolution, and (with `jsonschema` installed) full schema conformance. A green
run means any QVF consumer — including vibe-view — will read your file.

## Checklist for an ORCA → QVF integration

- [ ] `structure` from the geometry (Å).
- [ ] `wavefunction.gto` from the GBW, with the normalization flag set correctly
      and AO order permuted into QVF order.
- [ ] `spectra.*` for each spectrum the job produced.
- [ ] `atom_properties`, `bond_orders`, `scf_history` as available.
- [ ] Root `provenance`, `dipole_moment`, `thermochemistry`.
- [ ] `citations` (BibTeX for the level of theory).
- [ ] `run.record` with the verbatim `.inp` + `.out` (`program: "orca"`,
      `program_version` from the output header).
- [ ] EPR (if any) as canonical `spectra.epr` (g-tensor / hyperfine / ZFS).
- [ ] Validate with `qvf_reader.py` in CI.
