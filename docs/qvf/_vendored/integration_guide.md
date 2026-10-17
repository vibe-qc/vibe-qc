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
| Basis + Bloch orbitals at several k-points (periodic) | `wavefunction.bloch` (provisional) | complex coefficients `[n_k, n_spin, n_mo, n_ao, 2]`, energies and occupations `[n_k, n_spin, n_mo]`, fractional k-points and weights; the basis and its normalization as in Step 2 |
| Atomic charges (Mulliken, Löwdin, Hirshfeld, IAO, CM5, CHELPG, RESP, ESP, NBO, MBIS, Bader, DDEC), spin populations, valences, Fukui indices, atomic energies, volumes, NMR shieldings, magnetic moments, per-atom gradients | `atom_properties` | one member per role: float64 `[n_atoms]`, or `[n_atoms, 3]` for the vector roles; roles outside the list are `custom.<name>` and need a `meta` member (spec § 5.7) |
| Bond orders (Mayer / Wiberg) | `bond_orders` | quantitative per-pair table |
| Normal modes | `vibrations` | frequencies cm⁻¹; `hessian_ref` names the Hessian |
| Nuclear gradient / forces, stress | `forces` | store the gradient dE/dR in hartree/bohr (forces = −gradient); stress hartree/bohr³ |
| Cartesian Hessian | `hessian` | `[3N, 3N]`, hartree/bohr², index 3·atom + xyz |
| Excited states (TD-DFT, CIS, ADC, EOM-CC): energies, oscillator / rotatory strengths, transition dipoles, orbital contributions | `excited_states` | one JSON table; excitation energies in eV, dipoles in atomic units; `wavefunction_ref` names the ground-state orbitals the contributions index |
| Natural transition orbitals | `wavefunction.gto` (two per state) | `orbital_kind: "nto"`, `nto_role` `hole` / `particle`, `state_ref` + `state_index`; occupations are the pair weights |
| Transition density on a grid | `volume.transition_density` | e/bohr³, bohr grid; `excited_states_ref` + `state_index` name the state |
| Dipole (restated), quadrupole, octupole, polarizability, hyperpolarizability | `properties.electric` | one JSON member `properties`; dipole debye, quadrupole debye·Å (state `convention` and `traceless`), polarizabilities atomic units; a restated dipole must equal the root `dipole_moment` to 1e-6 debye (spec § 5.13) |
| Elastic constants | `properties.elastic` | symmetric 6 × 6 `stiffness`, Voigt order xx yy zz yz xz xy, GPa; optional `compliance` (1/GPa) and moduli with their `average` (voigt, reuss or hill) |
| Dielectric tensors, Born charges, piezoelectric tensor | `properties.dielectric` | `electronic` (3 × 3, required), `ionic`, `static` = electronic + ionic; `born_charges` one 3 × 3 per atom in e; `piezoelectric` 3 × 6 in C/m² |
| IR / Raman / UV-Vis / ECD / VCD | `spectra.*` | frequency + intensity |
| NMR / EPR | `spectra.nmr` / `spectra.epr` | object-shaped payload |
| SCF convergence trace | `scf_history` | per-iteration energy / error |
| Method / basis / functional / energy / convergence | root `provenance` | also what produced it: `software[]` (`{name, version?, url?}`), `git_commit`, `n_threads` / `n_mpi_ranks` / `n_gpus`, `started_utc` / `finished_utc` (UTC, `…Z`); typed in spec § 3.2, checked by the writers |
| Total energy, components, energies by method | root `energies` | `{value, units}` in hartree; must agree with `provenance.scf_energy` |
| Dipole, thermochemistry, constraints | root metadata | |
| Density / orbital / ESP / RDG grids | `volume.*` | bohr grid |
| Laplacian of the density | `volume.laplacian` | e/bohr⁵ |
| Vector fields (density gradient, current density, electric field) | `volume.vector` | `[nx, ny, nz, 3]`, Cartesian components |
| Complex orbitals / Bloch functions on a grid | `volume.complex` | `[nx, ny, nz, 2]`, last axis `[re, im]` |
| Densities per frame, per time step or per state | `volume.series` | `[n, nx, ny, nz]` + `axis` (`time` / `frame` / `state` / `custom`); prefer it over `reaction.path` `frame_volumes` |
| Periodic grids (CHGCAR, periodic cube) | `volume.*` with `grid.periodic` | declare `includes_endpoint` (cube style `true`, CHGCAR style `false`); the grid must span the cell |
| Optimization / IRC frames | `trajectory`, `reaction.path` | coords in Å; per-frame `energies` (hartree) as a binary member for long runs |
| Molecular dynamics (NVE / NVT / NPT / NPH) | `trajectory` (`metadata.kind` `"md"`, `ensemble`, `timestep_fs`) | coords Å (float32 allowed with `metadata.precision`); per-frame cell `lattice` in **rows, Å** (the structure convention, not `reaction.path`'s columns in bohr) with `metadata.pbc`; `velocities` Å/fs; `gradient` = dE/dR in hartree/bohr (forces = −gradient); `times` fs; `energies` hartree; `cell_volumes` Å³; `temperatures` K |
| Bands / DOS / phonons / EOS / Fermi surface | `bands`, `dos.*`, `phonon_*`, `equation_of_state`, `fermi_surface` | periodic |
| Level-of-theory references | `citations` | BibTeX bytes |
| Verbatim input + full log/output | `run.record` | UTF-8 text; `program` field names your code |
| Declarative job request (pending container) | `job.spec` | data, never code; pair with `provenance.run_status = "pending"`; typed `structure_ref`, `solvation`, `dispersion`, `relativistic`, `scf`, `geometry`, `resources` (hints), engine keywords under `options` (spec § 5.9) |
| Anything with no canonical kind | `x_<vendor>.*` | **see Step 4** |

**Start small.** The highest-value first cut for most codes is `structure` +
`wavefunction.gto` + whatever spectra you compute + root `provenance`. Add kinds
incrementally; partial support is a first-class feature of the format, not a
limitation.

**Before an archive is shared**, keep machine names out of root `provenance`
(`hostname` is deprecated, spec § 3.2, and the reference writers drop it by
default) and remember that a `run.record`'s verbatim input, log, and
attachments typically carry absolute paths, user names, and host names — see
"Sharing and sanitisation" in spec § 5.8.

**A pending container and its provenance.** To hand a job to a runner, write the
input `structure`, a `job.spec` and `provenance.run_status = "pending"`; the
spec's `structure_ref` names the geometry to run, which matters as soon as the
archive holds more than one `structure`. A `functional` or `basis` of `null`
says the method takes none (Hartree–Fock has no functional, GFN2-xTB no basis
set), which is not the same as leaving the key out, the engine's default
(§ 5.9). `resources` (`n_cores`, `memory_mb`,
`walltime_s`) are hints, not requirements, and engine keywords belong under
`options`. When the job has run, record in `provenance` what actually
produced the result: `software` (your code and the libraries that shaped the
numbers), `git_commit` (your own revision, 7 to 64 hex characters),
`n_threads`, `n_mpi_ranks`, `n_gpus` and the UTC `started_utc` /
`finished_utc`. Keep machine names out of `software[].url`.

```python
w.add_job_spec(job_type="periodic", method="rks", functional="pbe",
               basis="pob-tzvp-rev2", kpoints=[4, 4, 1], tasks=["optimize"],
               structure_ref="slab", dispersion="d3bj",
               scf={"convergence": 1e-8, "max_iterations": 100},
               resources={"n_cores": 16, "walltime_s": 86400})
w.set_provenance(run_status="converged", git_commit="9f2b6c1d4e7a", n_threads=8,
                 software=[{"name": "my-code", "version": "1.2.3"}],
                 started_utc="2026-10-07T12:00:00Z",
                 finished_utc="2026-10-07T12:07:31Z")
```

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

**Emit the overlap matrix, and let the validator verify it.** Pass the AO overlap
matrix your integral code computed for the basis you wrote (the AOs as written,
after any `N_i` divide) as the optional `overlap` member:

```python
w.add_wavefunction_gto(shells, mo_coefficients=C, energies=eps, occupations=occ,
                       coeffs_are_libint_normalized=True,
                       overlap=S)                          # [n_ao, n_ao]
```

Producers **SHOULD** write it. Every reference validator then recomputes the
self-overlap of each AO from `basis.json` (spec Appendix A.1, *Verifying with
`overlap`*) and compares it with the diagonal of `S`; a wrong normalization
flag, an AO order that is not Appendix A.2, or a Cartesian `d_xy` normalized on
its own (diagonal 1 instead of 1/3) fails with `E_NORMALIZATION`. It is the one
check that catches the normalization trap at write time rather than in a
rendered orbital. A further sanity check: evaluate the HOMO of H₂O/STO-3G on a
grid and integrate |ψ|² — it must come out ≈ 1 per spin.

**Effective core potentials and the electron count.** If your basis replaces
core electrons by an ECP, write each potential as a `basis.ecp` entry
(`center`, `n_core_electrons`, `label`, and the radial terms
`c r^(n−2) exp(−α r²)` of its local part and semi-local channels; spec § 5.3)
and state `n_electrons`, the electrons the wavefunction describes (`Σ Z − Σ
n_core_electrons − charge`). The occupations must sum to it, and
`provenance.n_electrons`, when you write it, must agree; validators check both.
Optional `density_matrix` (or `density_matrix_alpha` + `density_matrix_beta`)
members carry the one-particle density matrix, and `symmetry_labels` the
irreducible representation of each MO.

## Step 3 — write, then validate

Use a reference writer (Python or C++), or reimplement the producer contract
directly from the spec (§ "Producer conformance"). Either way, **validate every
archive you produce**, and wire it into your CI:

```sh
python qvf_reader.py your_output.qvf     # non-zero exit on any problem
```

The validator checks sha256 integrity, binary byte-sizing, id uniqueness,
cross-reference resolution, extension governance and the wavefunction
normalization against an `overlap` member — and, with `jsonschema`
installed, full conformance against the bundled `qvf_manifest.schema.json`. A
clean report means your file opens in vibe-view and any other conforming
consumer.

## Periodic systems

- `wavefunction.gto` carries molecular orbitals and the **Γ-point (k = 0)**
  crystalline orbitals over the cell's GTO basis. Orbitals at several k-points
  (or one k-point other than Γ) go into **`wavefunction.bloch`** (spec
  § 5.3.1): the same `basis` payload, a `kpoints` payload (fractional
  reciprocal-lattice coordinates, integration weights summing to 1), complex
  `mo_coefficients` `[n_k, n_spin, n_mo, n_ao, 2]` and per-k `energies`
  (hartree) and `occupations`. The Bloch sum is
  φ_μk(r) = N^(−1/2) Σ_T e^{+ik·T} χ_μ(r − T); if your code uses the opposite
  phase sign, or puts the atom position into the phase, convert the
  coefficients before writing. The kind is **provisional**: promoted ahead of
  the two-producer threshold, it becomes final when a second producer ships
  it, so a periodic Gaussian code that adopts it is exactly what it needs.
  k-resolved density matrices, spinor Bloch orbitals and non-Gaussian bases
  have no canonical kind yet; use a vendor section for those.
- Use `bands` (k-path eigenvalues), `dos.total` / `dos.projected`,
  `fermi_surface`, `phonon_bands` / `phonon_dos`, and `equation_of_state` for the
  usual periodic analyses. Densities and potentials go through `volume.*` grids.
- A grid that samples the cell declares `periodic` and `includes_endpoint`
  (spec § 5.2). Cube files repeat the first plane as the last
  (`includes_endpoint: true`, the default: `voxel_vectors[a] × (shape[a] − 1)`
  is lattice vector `a`); CHGCAR files do not (`includes_endpoint: false`:
  `voxel_vectors[a] × shape[a]` is). Write what your points are; the
  validator checks the span against the structure's cell, and consumers drop
  the duplicated plane when they tile a cube-style grid.
- The cell is stored in two conventions (`structure.lattice_vectors`: rows, Å;
  `reaction.path.lattice`: columns, bohr), so convert deliberately; see "Lattice
  conventions" in spec § 4.3 and the `Lattice` helper in the writer libraries.

## Several structures, several steps, and archive metadata

An archive may hold more than one `structure`, for example an ensemble, or the
start and end geometries of a workflow. Say which structure each result belongs
to with a section-level `structure_ref`, and which program run produced it with
`run_ref` naming a `run.record`; a free `group` label (`"step 2: frequencies"`)
lets a viewer show related sections together (spec § 4.1). The validator
rejects a reference that names a section of the wrong kind. A section's
`structure_ref` takes precedence over the one inside a `wavefunction.gto`
`basis` payload, so keep the two equal. The writers set these on an already
added section, so no `add_*` call changes:

```python
w.annotate_section(rho, structure_ref=h2o, run_ref=run, group="water")
```

The optional root `metadata` block describes the archive itself: `title`,
`authors`, `license` (an SPDX identifier such as `"CC-BY-4.0"`), `doi`,
`keywords`, `derived_from`, and the `created_utc` / `uuid` pair that identifies
it (spec § 3.5). The reference writers add `created_utc` and a random `uuid` by
default; pass `auto_metadata=False` (C++: `w.auto_metadata = false`) when you
need a manifest that is the same on every run, as for test fixtures. Leave authors' `email` and
`affiliation` out unless the person asks for them: archives get shared.

## Suggesting how the file opens

The optional root `viewer_defaults` object carries hints for viewers (spec
§ 3.4): `auto_open` lists the sections to show first, `bookmarks` holds camera
bookmarks, `camera` the camera to open with (a bookmark's camera), `crossfade`
two volume sections to blend (`{"a", "b", "blend"}`), and every other key is a
section id whose value suggests how that section is drawn: `isovalue`,
`colormap`, `opacity`, `replication`, for a `volume.*` or `basis.ao` section
`render` and `transfer_function`, and for a `bands` or `dos.*` section
`energy_window`, the `[min, max]` range in eV relative to the Fermi level it is
first plotted over. Hints are
suggestions. A consumer may ignore any of them and never refuses a file over a
malformed one, so a wrong hint costs you the look you wanted, not the file.

A spin density that opens as a volume, with a diverging palette and both lobes
visible:

```python
spin = w.add_volume("volume.spin", spin_density, origin, voxel_vectors,
                    section_id="spin")
w.set_viewer_defaults({
    spin: {
        "render": "volume",                  # "isosurface" is the default
        "transfer_function": {
            "range": [-0.02, 0.02],          # in the section's data units
            "color": [[-0.02, 0.23, 0.30, 0.75], [0.0, 0.95, 0.95, 0.95],
                      [0.02, 0.71, 0.02, 0.15]],                 # [x, r, g, b]
            "opacity": [[-0.02, 0.5], [-0.002, 0.0],
                        [0.002, 0.0], [0.02, 0.5]],              # [x, a]
            "colormap": "custom",            # draw these colour nodes as given
        },
    },
})
```

Without `"colormap": "custom"` the palette follows the section's `colormap`
hint. Never give a section the id `auto_open`, `bookmarks`, `camera` or
`crossfade`.

An all-electron band structure whose core states would squeeze the valence
region into a few pixels opens on the valence region, at a saved camera:

```python
bands = w.add_bands(kpath, eigenvalues)        # eigenvalues in eV
w.set_viewer_defaults({
    "camera": {"position": [0.0, -12.0, 4.0], "focal_point": [0.0, 0.0, 0.0],
               "view_up": [0.0, 0.0, 1.0], "view_angle": 30.0},
    bands: {"energy_window": [-15.0, 10.0]},   # eV relative to the Fermi level
})
```

## Several molecules, a reaction scheme and a picture

An archive that holds several species, such as the reactants and products of
a reaction, says which molecule each `structure` is: its `charge`,
`multiplicity`, `name`, `smiles`, `inchi` and `inchikey` go on the section
object, and its `label` is the compound number (spec § 5.1). A
`reaction.scheme` section relates the structures as the steps of a scheme
(§ 5.14), and a `preview` section carries a thumbnail that file managers show
(§ 5.15). Several `job.spec` sections, each naming its structure, ask for one
calculation per species, and each `run.record` names the job it ran with
`job_ref` (§ 5.9). Connectivity belongs to one structure: write it as the
structure's own `bonds` member, or give a standalone `bonds` section the
`structure_ref` of its structure (`add_bonds(pairs, structure_ref=...)`);
without one, a `bonds` section belongs to the archive's first structure
(§ 5.1). The [library guide](library_guide.md#sketch-documents-species-schemes-pictures-and-job-sets)
shows the calls.

## Declaring the file type

A program that opens or saves `.qvf` files declares the type the same way as
every other (spec § 2.4): the extension `.qvf`, the media type
`application/vnd.vibe-qc.qvf+zip`, and on macOS the Uniform Type Identifier
`com.vibe-qc.qvf`. Copy the declarations in [`spec/file-type/`](../spec/file-type/)
unchanged: `qvf-uti.plist` holds the dictionary for `Info.plist`
(`UTImportedTypeDeclarations`, or `UTExportedTypeDeclarations` in the one
application that exports it) and `qvf-mime.xml` the shared-mime-info
definition for Linux desktops.

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
