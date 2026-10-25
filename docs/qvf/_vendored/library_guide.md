# QVF library guide

This guide covers the reference C++17 and Python libraries for writing,
reading, validating and updating QVF archives. New integrations should use
`qvf::Writer`, `qvf::Reader` and `qvf::Editor` in C++, or `Writer`, `Reader`
and `Editor` from the Python `qvf` package. The original `QvfWriter` and
`QvfReader` APIs remain supported. Both implementations share the producer
contract in `spec/qvf-format-spec.md`; choose the language of your host code.
The [command-line guide](cli.md) covers inspection, validation and exchange
without embedding a library.

## Contents

- [Which writer do I use?](#which-writer-do-i-use)
- [The C++ library](#the-c-library)
  - [Building](#building)
  - [Linking into your project](#linking-into-your-project)
  - [The `QvfWriter` API](#the-qvfwriter-api)
  - [Building JSON payloads](#building-json-payloads)
  - [Binary tensors](#binary-tensors)
- [Command line](cli.md)
- [Python package API](#python-package-api)
- [Compatibility Python writer](#compatibility-python-writer)
  - [Trajectories and MD](#trajectories-and-md)
  - [Sketch documents: species, schemes, pictures and job sets](#sketch-documents-species-schemes-pictures-and-job-sets)
- [Lattices](#lattices)
- [Volumes and periodic grids](#volumes-and-periodic-grids)
- [Units](#units)
- [Validating your output](#validating-your-output)
  - [Generated tables and the schema checker](#generated-tables-and-the-schema-checker)
- [Reading QVF in C++](#reading-qvf-in-c)
- [Streaming and updating archives](#streaming-and-updating-archives)
- [JavaScript reader](#javascript-reader)
- [Section-kind cheat sheet](#section-kind-cheat-sheet)

## Which writer do I use?

| | C++ library | Python writer |
|---|---|---|
| File | `cpp/include/qvf/qvf.hpp` + `cpp/src/` | `python/qvf/` (compatibility: `qvf_writer`) |
| Dependencies | C++17 standard library only | Python stdlib; NumPy optional |
| Best for | linking into a C/C++ code (ORCA, Turbomole, …) | scripts, post-processing, prototyping |
| Compression | ZIP DEFLATE (per-member, STORE if smaller) | ZIP DEFLATE |

Both produce byte-for-byte-identical *binary* payloads (raw little-endian IEEE
arrays) and semantically identical manifests; they interoperate with any QVF
consumer, including vibe-view.

## The C++ library

### Building

```sh
cmake -S cpp -B build          # no find_package calls; zero external deps
cmake --build build
ctest --test-dir build         # runs the known-answer + smoke tests
./build/write_h2o out.qvf      # the worked example
```

Requires only CMake ≥ 3.16 and a C++17 compiler. There is nothing to install and
no library to find — the ZIP writer, CRC-32, SHA-256, and JSON emitter are all
self-contained in the toolkit.

For a library-only installation (toolkit 0.5.0):

```sh
cmake -S cpp -B build -DQVF_BUILD_TESTS=OFF -DQVF_BUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build
cmake --install build
```

`QVF_BUILD_TESTS` and `QVF_BUILD_EXAMPLES` default to `ON` for a standalone
build and `OFF` when another project includes QVF. Either can be set explicitly.
Tests include their conformance runner even if examples are disabled.
`QVF_BUILD_SHARED=ON` builds the shared library with ABI version 1; static
linking remains the default. These options do not change the archive format.

### Fortran (toolkit 0.5.0)

Set `QVF_FORTRAN=ON` to build `qvf::fortran`. The module provides checked
structure, scalar-volume, trajectory and restricted-GTO writers, plus raw
interfaces for the complete C ABI. See the [Fortran guide](fortran.md) for
installation, ownership, array ordering and the runnable example.

### Linking into your project

The library is small enough to vendor directly. Two options:

**A. `add_subdirectory`.** Drop the `qvf-writer/cpp` tree into your source and:

```cmake
add_subdirectory(third_party/qvf-writer/cpp)
target_link_libraries(your_target PRIVATE qvf::qvf)
```

**B. Add the sources directly.** Compile `cpp/src/qvf.cpp`,
`cpp/src/qvf_container.cpp`, `cpp/src/qvf_reader.cpp`,
`cpp/src/qvf_reader_facade.cpp`, `cpp/src/qvf_validation.cpp`,
`cpp/src/qvf_tensor_view.cpp`, `cpp/src/qvf_writer.cpp` and
`cpp/src/qvf_editor.cpp`; add
`cpp/src/qvf_c.cpp` and `cpp/src/qvf_c_facade.cpp` if you use the C ABI.
Add `cpp/include` and `cpp/src`
to the include path. CMake maintains this source list for you.

Then:

```cpp
#include "qvf/qvf.hpp"
```

**C. Single-header drop-in.** For the lowest-friction vendoring, grab the one
amalgamated file `cpp/qvf_single.hpp` (STB-style). In exactly **one** translation
unit define `QVF_IMPLEMENTATION` before including it; elsewhere just include it:

```cpp
// in one .cpp:
#define QVF_IMPLEMENTATION
#include "qvf_single.hpp"

// anywhere else:
#include "qvf_single.hpp"
```

No CMake, no separate compile step, no include-path juggling — one file, C++17.
The header is generated from the `cpp/` sources by `scripts/amalgamate.py` (a
drift test keeps the committed copy current); see `cpp/examples/write_single_header.cpp`.

**D. `find_package` / `FetchContent`.** The library installs a CMake package, so
after `cmake --install` (or a system package) adopters can:

```cmake
find_package(qvf CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE qvf::qvf)
```

The CMake target propagates the C++17 requirement and, for shared builds,
the C API import definition. Set `CMAKE_PREFIX_PATH` to the installation
prefix when the package is not in a standard search location.

Or pull the independently versioned QVF repository without an install step.
Its CMake project is at `cpp/` directly under the repository root. The
source repository URL is a build setting. Set `QVF_SOURCE_REPOSITORY` to
the repository you can access and configure Git credentials if needed. The
public publication URL is `https://github.com/vibe-qc/qvf.git`; it becomes
usable after the source snapshot is approved. Pin a reviewed revision explicitly:

```cmake
include(FetchContent)
FetchContent_Declare(qvf
  GIT_REPOSITORY "${QVF_SOURCE_REPOSITORY}"
  GIT_TAG v0.1.0
  SOURCE_SUBDIR cpp)
FetchContent_MakeAvailable(qvf)
target_link_libraries(your_target PRIVATE qvf::qvf)
```

**E. pkg-config (toolkit 0.5.0).** Installation also provides `qvf.pc`:

```sh
export PKG_CONFIG_PATH="$PWD/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
c++ -std=c++17 main.cpp $(pkg-config --cflags --libs qvf) -o example
```

Use the configured library directory (for example `lib64`) in place of `lib`
when needed. Relative include/library install paths remain relocatable.
Explicit absolute install paths remain fixed. For a C program, compile with
the C driver and link with the C++ driver so a static build's C++ runtime is
included. `pkg-config --cflags qvf` works with both C and C++ compilers;
C++ callers select C++17 themselves. Shared builds may also need the library
directory in the platform's runtime search path.

### The `QvfWriter` API

Every `add_*` method returns the section `id` it created (useful for
`volume.difference` operands and `reaction.waypoints` references). Construct a
writer, add sections and root metadata in any order, then `write()`.
Section-creation calls check duplicate, reserved and invalid IDs before
writing payload members; a rejected ID does not leave orphan ZIP entries.
The `add_atom_property` helpers can still add a new role to an existing
`atom_properties` section.

`WavefunctionGTO::localization_reference_basis` is an optional non-empty
reference-basis name. `add_job_spec` preserves explicit JSON null for
`basis` / `functional`, distinct from omitting the field. The Python writer
uses `localization_reference_basis=`, and `add_job_spec(basis=None,
functional=None)` writes null; leaving either keyword out omits it.


```cpp
#include "qvf/qvf.hpp"
using namespace qvf;

QvfWriter w({"my-code", "1.2.3", "job_name"});   // Source{program, version, calc}

// structure (positions in Ångström)
std::string sid = w.add_structure({
    {"O", 8, {{0.0, 0.0, 0.117}}},
    {"H", 1, {{0.0, 0.757, -0.469}}},
    {"H", 1, {{0.0, -0.757, -0.469}}},
});

// a spectrum
Spectrum ir; ir.frequencies = {1595, 3657, 3756}; ir.intensities = {67, 5, 42};
w.add_spectrum("spectra.ir", ir);

// provenance at the manifest root
Json prov = Json::object();
prov.set("method", "RHF").set("basis", "STO-3G").set("scf_converged", true);
w.set_provenance(prov);

w.write("result.qvf");
```

Method families (see the header for exact signatures):

- **Structure:** `add_structure`, `add_bonds`, `add_bond_orders`.
- **Volumes:** `add_volume(kind, grid, data, …)` for
  `volume.density/orbital/spin/elf/generic/potential/rdg/laplacian` (3-D
  data), `volume.vector` (`[nx, ny, nz, 3]`) and `volume.complex`
  (`[nx, ny, nz, 2]`, last axis `[re, im]`); `add_volume_series(grid, data,
  axis, …)` for `volume.series` (`[n, nx, ny, nz]` plus its `axis` JSON);
  `add_volume_difference`; `add_basis_ao`. `Grid` carries the optional
  `periodic`, `includes_endpoint` and `structure_ref` of a periodic grid (see
  [Volumes and periodic grids](#volumes-and-periodic-grids)).
- **Wavefunction:** `add_wavefunction_gto(WavefunctionGTO)` (restricted,
  unrestricted, `spin = "restricted_open"` or `"generalized"`; see the
  normalization note below). Optional fields (spec § 5.3): `ecp` (a JSON
  array of effective core potentials), `n_electrons` (the written
  occupations must sum to it), `symmetry_labels` (or
  `symmetry_labels_alpha` / `_beta`), `complex_coefficients` (each
  coefficient `Matrix` then holds `rows * cols * 2` interleaved re/im
  doubles; a `generalized` wavefunction needs it and `2 n_ao` columns), and
  the `[n_ao, n_ao]` matrices `overlap`, `density_matrix` or
  `density_matrix_alpha` + `density_matrix_beta`. `set_wavefunction_overlap`
  and `set_wavefunction_ecp` add the overlap and the ECPs to a section that
  is already written.
  `add_wavefunction_bloch(WavefunctionBloch)` writes the k-resolved Bloch
  orbitals of a periodic system (`wavefunction.bloch`, provisional, spec
  § 5.3.1): the same shells and normalization flag, k-points in fractional
  reciprocal-lattice coordinates with optional weights, row-major
  coefficients `[n_k, n_spin, n_mo, n_ao, 2]` (last axis real, imaginary),
  energies (hartree) and occupations `[n_k, n_spin, n_mo]`. Sizes that
  disagree, or weights that do not sum to 1, throw.
- **Spectra:** `add_spectrum(kind, Spectrum)` for 1-D spectra, or
  `add_spectrum(kind, Json)` for `spectra.nmr` and other object payloads.
- **Bands / DOS:** `add_bands(kpath, eigenvalues, projections?,
  occupations?)` — eigenvalues `[n_spin, n_k, n_bands]`; fat-band projections
  `[n_spin, n_k, n_bands, n_channels]`, or the legacy `[n_k, n_bands,
  n_channels]`, which means `n_spin = 1`; occupations `[n_spin, n_k, n_bands]`.
  `add_dos_total`; `add_dos_projected` with projections
  `[n_spin, n_channels, n_points]` or the legacy `[n_channels, n_points]`;
  `add_dos_coop_cohp`. Shapes that disagree with spec § 5.6 throw.
- **Trajectories / reactions / scans / vibrations:** `add_trajectory`,
  `add_reaction_path`, `add_reaction_waypoints`, `add_scan_surface`,
  `add_vibrations` (an optional last argument `hessian_ref` names the
  `hessian` section the modes come from). `add_trajectory(metadata, coords,
  extras)` takes a `TrajectoryExtras` of optional `const Tensor*` per-frame
  members (`lattice` in rows and Å, `velocities`, `gradient`, `times`,
  `energies`, `cell_volumes`, `temperatures`), and overloads take the cell as
  one `Lattice` or a `std::vector<Lattice>` (one per frame); float32 `coords`
  write `"precision": "float32"` into the metadata. `add_reaction_path(...,
  lattice, extras)` takes the shared `velocities`, `gradient`, `times` and
  `energies` (see [Trajectories and MD](#trajectories-and-md)).
- **Energy derivatives:** `add_forces(gradient, stress*, meta)` stores the
  energy **gradient** dE/dR, float64 `[n_atoms, 3]` in hartree/bohr (the
  forces are its negative), with an optional `[3, 3]` stress tensor in
  hartree/bohr³ for periodic systems; `add_hessian(matrix, meta)` stores the
  `[3N, 3N]` Cartesian Hessian in hartree/bohr² (spec § 5.10, § 5.11). Wrong
  shapes throw before anything is written.
- **Excited states:** `add_excited_states(states, meta, wavefunction_ref)`
  writes the `excited_states` table (spec § 5.12) from a JSON array of states
  in ascending `index` order; a number `excitation_energy` is written in eV
  and a number `total_energy` in hartree. `nto_ref` / `density_ref` must name
  sections already added, and with `wavefunction_ref` every contribution's
  `from_mo` / `to_mo` must be an orbital row of that wavefunction.
  Natural transition orbitals are two `WavefunctionGTO` sections with
  `orbital_kind = "nto"`, `nto_role = "hole"` / `"particle"`, `state_ref` and
  `state_index`; `add_volume_transition_density(grid, data,
  excited_states_ref, state_index)` writes the transition density. A 1-D
  spectrum names the table with a `states_ref` in its payload.
- **Property kinds:** `add_electric_properties(json)` (dipole in debye,
  quadrupole in debye·Å with its `convention` and `traceless` flag, octupole,
  polarizability and hyperpolarizability in atomic units),
  `add_elastic_properties(json)` (the symmetric 6 × 6 `stiffness` in Voigt
  order and GPa, optional `compliance`, moduli and their `average`) and
  `add_dielectric_properties(json)` (`electronic`, `ionic` and `static`
  tensors, `born_charges` one 3 × 3 per atom in e, `piezoelectric` in C/m²)
  write the `properties.electric`, `properties.elastic` and
  `properties.dielectric` kinds of spec § 5.13. Each takes the payload as JSON,
  checks its shapes, required fields, enumerations, symmetry and the
  static = electronic + ionic rule, and throws before anything is written.
  A dipole that the root `dipole_moment` also states must agree with it to
  1e-6 debye, and the Born tensors must number the atoms; the validators check
  both.
- **Analysis / periodic:** `add_atom_properties`, `add_scf_history`,
  `add_citations`, `add_structure_symmetry`, `add_fermi_surface`,
  `add_phonon_bands`, `add_phonon_dos`, `add_equation_of_state`,
  `add_topology_qtaim`. Besides the Mulliken / Löwdin / spin-population
  vectors, `add_atom_properties` takes a `std::map` from role name to values
  for any scalar role (see the cheat sheet), and `add_atom_property` appends
  one scalar role to the section. The vector roles (`magnetic_moment`,
  `force_eh_bohr`, `[n_atoms, 3]`) go through `add_atom_vector_property`. A
  `custom.<name>` role needs its description first: call
  `set_atom_properties_meta` with `{"custom.<name>": {units, description,
  kind}}`, and the writer emits the section's `meta` member.
- **Run record:** `add_run_record` — the verbatim program input and/or full
  log (UTF-8) plus attachments; `program` names the code that ran, making the
  archive a self-contained record of the calculation.
- **Job spec:** `add_job_spec` — the declarative specification of the
  calculation the archive *requests* (JobSpecPayload JSON: `job_type`
  required, plus method/basis/functional/charge/multiplicity/kpoints/tasks/
  options, and the typed `structure_ref`, `solvation`, `dispersion`,
  `relativistic`, `scf`, `geometry` and `resources`; the last are hints, not
  requirements). The typed fields are checked and the call throws on a wrong
  one. Pair with `set_provenance({... "run_status": "pending" ...})`
  for a container describing a job that has not yet run.
- **Root metadata:** `set_provenance`, `set_thermochemistry`,
  `set_dipole_moment`, `set_constraints`, `set_energies` (the root
  `energies` block, spec § 3.6), `set_extensions`,
  `set_viewer_defaults` (viewer hints, spec § 3.4; example under
  [Building JSON payloads](#building-json-payloads)). `set_provenance` drops a
  `hostname` key, which is deprecated (spec § 3.2) because archives are meant to
  be shared; call `set_provenance(prov, /*allow_hostname=*/true)` to keep it.
  It also checks the typed fields `software` (`{name, version?, url?}` items),
  `git_commit` (7 to 64 hex characters), `n_threads`, `n_mpi_ranks`, `n_gpus`
  and the UTC `started_utc` / `finished_utc`, and throws on a wrong one.
- **Archive metadata:** `set_metadata(Json)` sets the root `metadata` block
  (spec § 3.5: title, authors, licence, DOI, keywords, …). By default
  `build_manifest()` also adds `created_utc` and a random version-4 `uuid`
  when the block lacks them, made once per writer; set the public field
  `w.auto_metadata = false` for a manifest that is the same on every run.
- **Section references:** `add_bonds(pairs, section_id, structure_ref)`
  names the structure whose atoms a standalone `bonds` section addresses
  (spec § 5.1; without it the bonds belong to the first structure).
  `annotate_section(id, structure_ref, run_ref,
  group)` marks an already added section with the `structure` it describes,
  the `run.record` that produced it and a display group (spec § 4.1; an empty
  string leaves a field alone); `set_section_field(id, key, value)` sets any
  other section field. A reference to a missing section or one of the wrong
  kind throws.
- **Escape hatch:** `add_vendor_section("x_<vendor>.*", members, critical?)`;
  an overload takes `std::vector<OpaqueMember>` (`{role, bytes,
  media_type}`) for members stored without dtype and shape, and
  `add_opaque_member(section_id, role, bytes, media_type)` adds one to an
  existing vendor section (spec § 4.2, § 7.3).
- **Sketch documents** (spec § 5.1, § 5.9, § 5.14, § 5.15):
  `set_structure_identity(id, Json)` gives a `structure` section its
  `charge`, `multiplicity`, `name`, `smiles`, `inchi` and `inchikey` (a null
  value removes one; a charge and multiplicity impossible for the atoms
  throw); `add_reaction_scheme(steps, label, section_id)` takes the steps as
  a JSON array, each participant a structure id or `{structure_ref,
  coefficient}`; `add_preview(thumbnail, image, label, section_id,
  thumbnail_media_type, image_media_type)` takes PNG or JPEG bytes and an
  optional SVG, PNG or JPEG image, sniffs and checks the media types and
  stores both members uncompressed; `RunRecord::job_ref` and
  `RunRecord::run_status` name the job a run executed and its outcome, and
  `add_job_spec` writes a `structure_ref` on the section object as well as
  in the payload. Labels of other sections go through
  `set_section_field(id, "label", …)`.

#### The wavefunction normalization flag

`WavefunctionGTO::coeffs_are_libint_normalized` is the one flag you must get
right (see spec Appendix A). If your engine stores contraction coefficients
already multiplied by the primitive norm `N_i` (libint / libcint / GBW-style
storage), set it to `true` and the writer divides them out so they apply to
`N_i`-normalized primitives, as QVF requires — `N_i` is spec Appendix A.1's
*axial* norm, one factor per shell from the total `l`, not a per-component
unit norm. If your coefficients already apply to `N_i`-normalized primitives,
leave it `false`.

Then write the AO overlap matrix (`WavefunctionGTO::overlap`, computed by
your integral code for the AOs as written): every validator compares its
diagonal with the self-overlaps spec Appendix A.1 gives for the stored basis,
so a wrong flag, a wrong AO order or a per-component Cartesian normalization
fails validation with `E_NORMALIZATION` instead of producing plausible but
wrong orbitals.

### Building JSON payloads

`qvf::Json` is a tiny ordered JSON value used for flexible members (`spectra.nmr`
payloads, EOS `fit`, DOS metadata, viewer defaults, vendor sections):

```cpp
Json nmr = Json::object();
Json shifts = Json::array();
Json cs = Json::object();
cs.set("atom_index", 1).set("symbol", "H").set("isotropic_shift_ppm", 4.6);
shifts.push_back(cs);
nmr.set("isotope", "1H").set("reference", "TMS").set("chemical_shifts", shifts);
w.add_spectrum("spectra.nmr", nmr);
```

Helpers: `Json::from_doubles(vec)`, `Json::from_ints(vec)`,
`Json::from_strings(vec)`, `Json::array()`, `Json::object()`, `.set(key, value)`,
`.push_back(value)`.

Viewer hints are built the same way. `set_viewer_defaults` takes the whole
object; each key other than the reserved `auto_open`, `bookmarks`, `camera`
and `crossfade` is a section id. This
one opens a spin density as a volume with its own transfer function: nodes are
`[x, r, g, b]` and `[x, a]` with `x` in the section's data units, and
`"colormap": "custom"` draws the colour nodes as given:

```cpp
std::string spin = w.add_volume("volume.spin", grid, spin_density, "", "", "spin");

Json color = Json::array();
color.push_back(Json::from_doubles({-0.02, 0.23, 0.30, 0.75}))
     .push_back(Json::from_doubles({0.0, 0.95, 0.95, 0.95}))
     .push_back(Json::from_doubles({0.02, 0.71, 0.02, 0.15}));
Json opacity = Json::array();
opacity.push_back(Json::from_doubles({-0.02, 0.5}))
       .push_back(Json::from_doubles({-0.002, 0.0}))
       .push_back(Json::from_doubles({0.002, 0.0}))
       .push_back(Json::from_doubles({0.02, 0.5}));
Json tf = Json::object();
tf.set("range", Json::from_doubles({-0.02, 0.02}))
  .set("color", color).set("opacity", opacity).set("colormap", "custom");
Json hint = Json::object();
hint.set("render", "volume").set("transfer_function", tf);
Json defaults = Json::object();
defaults.set(spin, hint);
w.set_viewer_defaults(defaults);
```

### Binary tensors

Binary members (grids, coordinates, eigenvalues) are `qvf::Tensor` objects built
by the `tensor_*` helpers, which serialize **little-endian** regardless of host:

```cpp
Tensor rho = tensor_f32(values, {nx, ny, nz});     // float32 3-D grid
Tensor coords = tensor_f64(flat_xyz, {n_frames, n_atoms, 3});
```

`tensor_f32`, `tensor_f64`, `tensor_i32`, `tensor_i64` cover the common cases;
`shape` must multiply to the number of supplied values.

## Python package API

Install `qvf-writer` to use the `qvf` package. The distribution keeps its
existing name. The reader, editor and raw-buffer writer use only the standard
library; install `qvf-writer[numpy]` for numeric writer conveniences and NumPy
array views, and `qvf-writer[validate]` for full JSON Schema validation.
The schema is installed as package data. A schema update must refresh both
`python/qvf_manifest.schema.json` and `python/qvf/schema/qvf_manifest.schema.json`
from `spec/qvf_manifest.schema.json`; packaging checks require identical bytes. `qvf_writer` and `qvf_reader` remain
available through compatibility modules and emit `DeprecationWarning`;
`QvfWriter`, `QvfReader` and `qvf-validate` keep their existing behavior.

`Writer` writes each member immediately and writes `manifest.json` last. A
binary member sink accepts C-contiguous buffer objects and tracks its byte
count and SHA-256 while writing. The declared shape must match exactly. The
input order defaults to little endian; pass `byte_order="big"` or `"native"`
when needed. Conversion handles elements split across chunks. `force_zip64`
is available for small synthetic ZIP64 archives; normal large members select
ZIP64 automatically. A non-seekable output uses ZIP data descriptors.

```python
import struct
from qvf import Writer, Reader, Section

with Writer("density.qvf", program="example", version="1.0") as writer:
    writer.add_section(Section(id="rho", kind="volume.density", members={}))
    writer.add_json("rho", "grid", {
        "origin": [0, 0, 0], "voxel_vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "shape": [1, 1, 2],
    })
    with writer.begin_binary("rho", "data", "float64", [1, 1, 2]) as member:
        member.write(struct.pack("<d", 0.5))
        member.write(struct.pack("<d", 0.25))

with Reader("density.qvf") as reader:
    print(reader.manifest.source.program)  # typed, lossless Manifest
    print(reader.tensor("rho", "data").tolist())
    raw = reader.tensor("rho", "data").data  # read-only byte memoryview
    assert reader.validate()["ok"]
```

`Writer.add_*` supports the established convenience methods for every kind.
These methods write each completed member immediately; array conversion can
allocate one complete member. Use `begin_binary` to bound memory by one slab.
`add_binary` writes one raw buffer, `add_opaque` writes untyped bytes, and
`add(Section)` adds a typed manifest section. `add(reader.get(section_id))`
copies a section, streaming its tensors and verifying their checksums.

`Reader` reads only the manifest at open and checks ZIP entry names and
resource limits before decompression. `sections_of(kind)` and `section(id)`
return `Section` dataclasses; their members are `MemberSpec` dataclasses.
`json(section_or_id, role)` reads one JSON payload; `tensor` returns an owning
`Tensor` with `dtype`, `shape`, byte `data`, standard-library `tolist()` and
optional `numpy()`. `tensor_into` streams little-endian byte chunks to a
callback or binary file. Hash and CRC verification finish at the end of that
stream; discard any consumed result when it raises. A `Limits` instance can
lower member, total, JSON depth/size and compression-ratio caps.

`get(id)` returns a lazy `SectionView` for every canonical or vendor kind.
Structure, volume and GTO views add named accessors (`atoms`, `lattice`,
`grid`, `data`, `basis`, `mo_metadata`, `coefficients(spin=...)`). Keep the
reader open while using those views. `raw_manifest` exposes the original
mapping, while `manifest` and `typed_manifest` expose the lossless model.
`qvf.model.Lattice` and `GridSpec` provide standard-library geometry helpers;
`qvf.units` exposes the schema-derived vocabulary and explicit conversions.

```python
from qvf import Editor, validate

with Editor("density.qvf", atomic=True) as editor:
    editor.set_run_status("running")
    editor.writer.add_citations("@misc{example, title={Example}}")

report = validate.run("density.qvf")
assert report["ok"]
Editor.repack("density.qvf", "density-compact.qvf")
```

`Editor.writer` accepts all streaming writer operations. `remove_section`
removes the manifest entry and leaves its bytes as orphans; reusing its id
writes fresh member paths. Job specifications remain verbatim during updates.
`set_run_status` sets the root lifecycle status or a named `run.record` via
`run_id`. A successful update stamps `metadata.modified_utc`. The default
appends in place; `atomic=True` stages a copy beside the target and replaces
it on commit. Context exit commits on success and aborts on an exception.
`Editor.repack` removes orphan members and superseded manifests, verifies
member hashes, and replaces its output atomically. Its optional `compression`
selects `zipfile.ZIP_STORED` or `ZIP_DEFLATED`; omission preserves the methods.

`validate.run`, `validate_qvf` and `Reader.validate` use the existing reference
validator and return the same finding codes, locations and report shape.
Full validation explicitly reads the payloads; opening a Reader does not.

## Compatibility Python writer

Same shape, no build step:

```python
import numpy as np
from qvf_writer import QvfWriter

w = QvfWriter(program="my-code", version="1.2.3", calculation="job")
w.add_structure([{"symbol": "O", "position": [0, 0, 0.117], "atomic_number": 8}])
w.add_density(np.abs(rho), origin=[-8, -8, -8],
              voxel_vectors=[[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]])
w.add_wavefunction_gto(shells, mo_coefficients=C, energies=eps, occupations=occ,
                       coeffs_are_libint_normalized=True)
w.write("result.qvf")
```

The Python API mirrors the C++ one method-for-method; payloads are plain lists /
dicts / NumPy arrays. See `python/examples/write_h2o.py`. Like the C++ writer,
`set_provenance(**fields)` silently drops the deprecated `hostname` field
unless you pass the keyword-only `allow_hostname=True`. It checks the typed
fields and raises `ValueError` on a wrong one (`software=[{"name": ...,
"version": ...}]`, `git_commit`, `n_threads`, `n_mpi_ranks`, `n_gpus`,
`started_utc` / `finished_utc` as `...Z` strings or aware `datetime`s; `None`
removes a key), and `add_job_spec(..., structure_ref=, solvation=,
dispersion=, relativistic=, scf=, geometry=, resources=)` does the same for the
job request.

Energies and their derivatives:

```python
w.set_provenance(method="rhf", scf_energy={"value": -74.9659, "units": "hartree"})
w.set_energies(-74.9659, method="rhf",                     # floats are hartree
               components={"nuclear_repulsion": 9.1895, "electronic": -84.1554},
               by_method=[{"method": "rhf", "total": -74.9659}])
w.add_forces(gradient, meta={"structure_ref": "structure"})   # dE/dR, not the forces
w.add_hessian(hessian, meta={"atom_masses_amu": [15.999, 1.008, 1.008]})
w.add_vibrations(metadata, displacements, hessian_ref="hessian")
```

Excited states, their natural transition orbitals and a transition density
(spec § 5.12); add the sections a state names before the table:

```python
for role, C_nto in (("hole", C_hole), ("particle", C_particle)):
    w.add_wavefunction_gto(shells, mo_coefficients=C_nto, occupations=weights,
                           orbital_kind="nto", state_ref="excited_states",
                           state_index=1, nto_role=role, section_id=f"nto_s1_{role}")
w.add_volume("volume.transition_density", rho_01, origin, voxels,
             excited_states_ref="excited_states", state_index=1, section_id="tden_s1")
w.add_excited_states([{"index": 1, "excitation_energy": 7.40,     # floats are eV
                       "oscillator_strength": 0.052,
                       "contributions": [{"from_mo": 4, "to_mo": 5, "weight": 0.99}],
                       "nto_ref": "nto_s1_hole", "density_ref": "tden_s1"}],
                     meta={"method": "tddft", "n_states": 1}, wavefunction_ref="wf")
w.add_spectrum("spectra.uvvis", payload={"frequencies": [7.40], "intensities": [0.052],
               "peaks": [{"x": 7.40, "assignment": {"state_index": 1}}],
               "states_ref": "excited_states"})
```

An energy that `energies` states for the method `provenance.method` names
must equal `provenance.scf_energy` within 1e-8 hartree; the validator reports
a disagreement as `E_ROOT_INCONSISTENT`.

### Trajectories and MD

A `trajectory` carries the frames of an optimization, an IRC, a scan or a
molecular-dynamics run (spec § 5.5). Beyond `coords`, every per-frame member
is optional and in frame order; each must have the frame count of `coords`
(and the vectors its atom count), or the writer raises before writing:

```python
cells = [Lattice.from_rows_angstrom(rows_k) for rows_k in box_rows]   # Å, rows
w.add_trajectory(
    {"symbols": symbols, "kind": "md", "ensemble": "npt", "timestep_fs": 0.5,
     "pbc": [True, True, True]},
    coords,                       # [n_frames, n_atoms, 3], Å
    lattice=cells,                # [n_frames, 3, 3] rows in Å (one Lattice: [3, 3])
    velocities=v,                 # [n_frames, n_atoms, 3], Å/fs
    gradient=dEdR,                # [n_frames, n_atoms, 3], hartree/bohr: NOT the forces
    times=t_fs, energies=e_eh,    # [n_frames], fs and hartree
    cell_volumes=vol_a3, temperatures=t_k,
    precision="float32")          # coords stored as float32, metadata.precision set
```

The trajectory `lattice` is in the **structure convention** (rows, Å), like
`add_structure(lattice_vectors=...)` and unlike `add_reaction_path(lattice=...)`
(columns, bohr); a plain array is written as given and must already be rows in
Å. With a `lattice`, the metadata must carry `pbc`, and a periodic `pbc` needs a
`lattice`. The member holding per-frame gradients is called `gradient` because
it stores dE/dR; negate it to draw forces. When the metadata also lists
`energies`, the binary `energies` must equal it within 1e-10 hartree; for long
runs prefer the binary member and leave the JSON list out. `add_reaction_path`
takes the shared `velocities`, `gradient`, `times` and `energies` the same way.
The validators check the same rules (`E_MEMBER_MALFORMED`).

Electric, elastic and dielectric properties (spec § 5.13):

```python
w.set_dipole_moment(total_debye=1.85, vector_debye=[0, 0, -1.85])
w.add_electric_properties({
    "origin": "center_of_mass",
    "dipole": {"vector": [0, 0, -1.85], "total": 1.85},          # debye, as the root block
    "quadrupole": {"matrix": theta, "traceless": True,           # debye·Å
                   "convention": "buckingham"},
    "polarizability": {"matrix": alpha, "frequency_ev": 0.0}})   # atomic units
w.add_elastic_properties(C, compliance=S, average="hill",        # C: 6 x 6, GPa, Voigt
                         moduli={"bulk_modulus_gpa": 76.0, "shear_modulus_gpa": 26.0})
w.add_dielectric_properties(eps_inf, ionic=eps_ion, static=eps_inf + eps_ion,
                            born_charges=Z)                      # Z: [n_atoms, 3, 3], in e
```

The writers check shapes, symmetry and `static == electronic + ionic` and raise
`ValueError` before writing. A restated dipole that differs from the root
`dipole_moment` by more than 1e-6 debye is `E_ROOT_INCONSISTENT` in the
validator (the order of the two calls does not matter), and Born charges that
do not number the atoms of the section's structure are `E_MEMBER_MALFORMED`.

Bloch orbitals of a periodic system at several k-points (`wavefunction.bloch`,
spec § 5.3.1, provisional) take the shells of `add_wavefunction_gto` and
complex arrays:

```python
w.add_structure(atoms, pbc=[True, True, True], lattice_vectors=cell)
w.add_wavefunction_bloch(shells, kpoints_frac,      # [n_k, 3], fractional
                         C_k,                       # complex [n_k, n_spin, n_mo, n_ao]
                         eps_k, occ_k,              # [n_k, n_spin, n_mo], hartree / electrons
                         weights=w_k, spin="unrestricted",
                         coeffs_are_libint_normalized=True)
```

The coefficients follow the Bloch sum φ_μk(r) = N^(−1/2) Σ_T e^{+ik·T}
χ_μ(r − T); `structure_ref` must name a periodic structure, which the
validator checks.

Archive metadata and section references work the same way. `set_metadata(
title=..., authors=[{"name": ...}], license="CC-BY-4.0", keywords=[...])`
fills the root `metadata` block and checks each field (time stamps may be
timezone-aware `datetime` objects, `uuid` a `uuid.UUID`); the writer adds
`created_utc` and a random `uuid` unless you set them or construct it with
`QvfWriter(..., auto_metadata=False)`. `annotate_section(section_id,
structure_ref=..., run_ref=..., group=..., label=...)` sets the generic
references of an added section (spec § 4.1) and raises `ValueError` for a
reference to a missing section or one of the wrong kind.

### Sketch documents: species, schemes, pictures and job sets

A program that prepares calculations, such as a structure and reaction
sketcher, keeps its document as QVF: one `structure` section per species with
its chemical identity (spec § 5.1), a `reaction.scheme` that relates them
(§ 5.14), a `preview` that file managers show (§ 5.15) and, when it asks for
calculations, one `job.spec` per job (§ 5.9). Every `add_*` method takes
`label=`, the section's display label; for a species it is the compound
number.

```python
from qvf_writer import QvfWriter

w = QvfWriter(program="my-sketcher", version="1.0", calculation="")
w.add_preview(png_bytes, image=svg_text)          # PNG or JPEG; SVG, PNG or JPEG
w.add_structure(methane_atoms, charge=0, multiplicity=1, name="methane",
                smiles="C", inchi="InChI=1S/CH4/h1H4",
                inchikey="VNWKTOKETHGBQD-UHFFFAOYSA-N", label="1",
                section_id="structure_s1")
w.add_structure(cl2_atoms, charge=0, multiplicity=1, name="chlorine", label="2",
                section_id="structure_s2")
w.add_structure(ch3cl_atoms, label="3", section_id="structure_s3")
w.add_structure(hcl_atoms, label="4", section_id="structure_s4")
w.add_reaction_scheme([
    {"id": "r1", "reactants": ["structure_s1", "structure_s2"],
     "products": ["structure_s3", {"structure_ref": "structure_s4", "coefficient": 1}],
     "arrow": "forward", "conditions": ["hν", "25 °C"],
     "yield_percent": 61.5, "yield_text": "61.5%"},
], label="Radical chlorination")
w.set_extensions({"x_mysketcher": {"version": "1"}})
w.add_vendor_section("x_mysketcher.document", json_members={"document": doc},
                     opaque_members={"drawing": (svg_bytes, "image/svg+xml")})
w.write("scheme.qvf")
```

`add_structure` refuses a charge and multiplicity impossible for the atoms;
`add_reaction_scheme` refuses a participant that is not an added `structure`
section, a repeated step id or a structure listed twice in one role.
Participants are chemical roles: the target of a `retrosynthetic` step is
among its `products`. `add_preview` sniffs the image types from the bytes
(pass `thumbnail_media_type=` / `image_media_type=` to state them; they are
checked) and stores both members uncompressed; an SVG must reference no
external resource and contain no script.

A job set: one `job.spec` per job, each naming its structure, and one
`run.record` per run naming its job.

```python
w.set_provenance(run_status="pending")
w.add_job_spec(job_type="molecular", method="rks", functional="pbe0",
               basis="def2-svp", structure_ref="structure_s1", section_id="job_s1")
w.add_job_spec(job_type="molecular", method="rks", functional="pbe0",
               basis="def2-svp", structure_ref="structure_s3", section_id="job_s3")
# … a runner later adds, for each job it ran:
w.add_run_record(program="my-engine", log_text=log, job_ref="job_s1",
                 run_status="converged", section_id="run_s1")
```

`add_job_spec` writes `structure_ref` on the section object, where a reader
finds it in the manifest, and in the payload.

Bonds belong to one structure. `add_structure(..., bonds=...)` writes them as
the structure's own member; a standalone `bonds` section names its structure
with `add_bonds(pairs, structure_ref="structure_s3")`, written on the section
object, and without it belongs to the archive's first structure (spec § 5.1).
The reference must name an added `structure` section; validators check the
indices against that structure's atoms.

## Lattices

QVF v1 stores a periodic cell in two conventions: `structure`
`lattice_vectors` and the `trajectory` `lattice` member are **row** vectors in
**Å**, while the `reaction.path` `lattice` is a matrix whose **columns** are a,
b, c in **bohr** (spec § 4.3, "Lattice conventions"). Both are frozen in v1 and
must not be mixed up. The `Lattice` helper holds the cell once and converts to
whichever the member needs, in both languages:

```python
from qvf_writer import Lattice, QvfWriter

cell = Lattice.from_rows_angstrom([[2.46, 0, 0], [1.23, 2.1304, 0], [0, 0, 15.875]])
cell.columns_bohr()          # ≈ [[4.6487, 2.3244, 0], [0, 4.0259, 0], [0, 0, 29.9994]]
cell.volume_angstrom3()      # ≈ 83.1974
w.add_structure(atoms, pbc=[True, True, False], lattice_vectors=cell)   # rows, Å
w.add_trajectory(meta_with_pbc, coords, lattice=cell)                   # rows, Å
w.add_reaction_path(meta, coords, lattice=cell)                         # columns, bohr
# one Lattice per frame for a variable cell: lattice=[cell0, cell1, ...]
# from an engine that holds columns in bohr: Lattice.from_columns_bohr(matrix)
```

```cpp
#include "qvf/qvf.hpp"
using namespace qvf;

Lattice cell = Lattice::from_rows_angstrom({{{{2.46, 0, 0}},
                                             {{1.23, 2.1304, 0}},
                                             {{0, 0, 15.875}}}});
w.add_structure(atoms, {{true, true, false}}, cell);   // lattice_vectors: rows, Å
w.add_trajectory(meta_with_pbc, coords, cell);         // lattice: rows, Å
w.add_reaction_path(meta, coords, &cell);              // lattice: columns, bohr
// cell.columns_bohr(), cell.rows_angstrom(), cell.volume_angstrom3(),
// cell.volume_bohr3(); Lattice::from_columns_bohr(m) for the bohr convention.
```

Plain arrays still work as before: a nested list for `lattice_vectors`, an
array (Python) or `TrajectoryExtras::lattice` (C++) for the trajectory
`lattice`, and a `Tensor*` (C++) or array (Python) for the reaction-path
`lattice` are written as given, so they must already be rows in Å, rows in Å
and columns in bohr respectively. The C++ trajectory overloads take one
`Lattice` (a fixed cell) or a `std::vector<Lattice>` (one per frame); the
reaction-path overload takes one `Lattice` for a fixed cell, and a per-frame
`[n_frames, 3, 3]` path in C++ passes a `Tensor` built from each cell's
`columns_bohr()`.

## Volumes and periodic grids

`add_volume` writes every volume kind of spec § 5.2. Scalar kinds take a 3-D
array; `volume.vector` takes `[nx, ny, nz, 3]` (Cartesian x, y, z on the last
axis), `volume.complex` takes `[nx, ny, nz, 2]` (`[re, im]`; the Python writer
also accepts a complex 3-D array and splits it), and `volume.series` takes
`[n, nx, ny, nz]` with an `axis` that says what indexes the slabs. The grid
`shape` is always the three spatial extents. Wrong ranks and trailing axes
are refused before anything is written.

```python
w.add_volume("volume.vector", grad_rho, origin, vox)            # [nx, ny, nz, 3]
w.add_volume("volume.complex", psi, origin, vox, dtype="float64")  # complex array
w.add_volume("volume.laplacian", lap, origin, vox)              # e/bohr^5
w.add_volume("volume.series", rho_frames, origin, vox,          # [n, nx, ny, nz]
             axis={"kind": "frame", "values": [0, 2, 4],        # frame indices
                   "labels": ["R", "TS", "P"], "trajectory_ref": "trajectory"})
```

A grid that samples a periodic cell says so with `periodic` (per axis),
`includes_endpoint` and `structure_ref`. The rule to get right is the endpoint:
a **cube-style** grid (`includes_endpoint=True`, the default) repeats the first
plane as the last, so `voxel_vectors[a] * (shape[a] - 1)` is lattice vector `a`;
a **CHGCAR-style** grid (`includes_endpoint=False`) does not, so
`voxel_vectors[a] * shape[a]` is. Write the value that matches your points.
Validators check the span against the structure's cell (`E_MEMBER_MALFORMED`
when it differs by more than 1e-6 relative), and a consumer that tiles a
cube-style grid drops the duplicated plane.

```python
cell = Lattice.from_rows_angstrom(rows)
w.add_structure(atoms, pbc=[True, True, True], lattice_vectors=cell)
n = rho.shape                                     # CHGCAR: n planes per period
vox = [[x * BOHR_PER_ANGSTROM / n[a] for x in rows[a]] for a in range(3)]
w.add_density(rho, [0, 0, 0], vox, periodic=[True, True, True],
              includes_endpoint=False, structure_ref="structure")
```

```cpp
Grid g;
g.voxel_vectors = vox;                            // bohr per step
g.periodic = std::array<bool, 3>{{true, true, true}};
g.includes_endpoint = false;                      // CHGCAR style
g.structure_ref = "structure";
w.add_volume("volume.density", g, rho);
Json axis = Json::object();
axis.set("kind", "time").set("values", Json::from_doubles({0.0, 0.5})).set("units", "fs");
w.add_volume_series(Grid{}, rho_t, axis, "rho(t)");   // rho_t: [2, nx, ny, nz]
```

The `frame_volumes` + `volume_grid` members of a `reaction.path` are the older
form of a `volume.series` with a `frame` axis naming the path; they stay
valid, but new code should write a `volume.series`.

## Units

Every value is stored in the unit spec § 4.3 gives it, and the writers never
convert. Where a payload may say which unit it uses, it names that unit with a
**unit name** from the vocabulary of spec § 4.3 (`angstrom`, `bohr`,
`hartree`, `ev`, `cm-1`, `nm`, …; the schema's `$defs/UnitName`). Both writers
check every name they write and refuse an unknown one, and a declaration of a
unit the spec fixes (a grid origin in anything but `bohr`) is refused too.
`set_provenance(scf_energy_eh=…, fermi_energy_ev=…)` writes `hartree` and `ev`;
the older spellings `Eh`, `eV` and `Hartree` remain readable but are
deprecated.

```python
w.add_density(rho, origin, vox, units={"origin": "bohr", "voxel_vectors": "bohr"})
w.add_trajectory(meta, coords, units={"energies": "hartree", "coords": "angstrom"})
w.add_scan_surface(meta, r, theta, e, units={"axis_a": "bohr", "axis_b": "degree"})
w.add_spectrum("spectra.uvvis", frequencies=lam, intensities=f, x_units="nm")
w.add_topology_qtaim(points, units={"position": "bohr"})

from qvf_units import UNITS, canonical, convert
convert(1.0, "hartree", "kcal_per_mol")   # 627.5094740631 (CODATA 2018)
convert(300.0, "nm", "ev")                # photon energy of a 300 nm line
canonical("Eh")                           # "hartree"
```

In C++, `Grid::units`, `Spectrum::x_units` / `y_units` and a `units` object in
the JSON metadata play the same role; `qvf/units.hpp` (included by
`qvf/qvf.hpp`) has `enum class Unit`, `unit_name()`, `parse_unit()`,
`canonical_unit()` and the same `convert()`. The validators warn about an
unknown unit name with `W_UNIT_UNKNOWN`; a validator that runs the JSON Schema
also rejects it.


## C and Fortran — the C ABI

For codes that can't call C++ directly (C, Fortran via `ISO_C_BINDING`, Rust,
Julia, …), the library ships a stable **C ABI** in
[`cpp/include/qvf/qvf_c.h`](../cpp/include/qvf/qvf_c.h). All state lives in one
opaque handle per writer, reader, validation report or editor. The complete
[ABI 1 guide](c_api.md) describes streaming, periodic structures, every canonical
kind, lazy reads, validation, editing and shared-library use. Existing writer
signatures remain available. Functions return `0` on success and non-zero on
error (`qvf_last_error()` gives the message). Nested payloads (provenance,
NMR/EPR spectra, vendor sections) are passed as **JSON strings** the caller
builds, so no JSON parser crosses the boundary. `qvf_set_provenance_json`
parses its string inside the library and drops a top-level `hostname` key
(there is no opt-in through the C ABI); malformed JSON there is an error.

```c
#include "qvf/qvf_c.h"

qvf_writer* w = qvf_create("my-code", "1.0", "h2o/rhf");
const char* sym[3] = {"O", "H", "H"};
int Z[3] = {8, 1, 1};
double pos[9] = {0,0,0.117, 0,0.757,-0.469, 0,-0.757,-0.469};
qvf_add_structure(w, sym, Z, pos, 3);

double freq[3] = {1595, 3657, 3756}, inten[3] = {67, 5, 42};
qvf_add_spectrum_xy(w, "spectra.ir", freq, inten, 3);
qvf_set_provenance_json(w, "{\"method\":\"RHF\",\"basis\":\"STO-3G\"}");
qvf_write(w, "h2o.qvf");
qvf_destroy(w);
```

`qvf_add_job_spec_json(w, spec_json)` writes a `job.spec` from the whole
payload as a JSON string (the typed `qvf_add_job_spec` has no argument for the
newer fields such as `structure_ref`, `scf` or `resources`); the string is
parsed and its typed fields checked. `qvf_set_provenance_json` checks the typed
provenance fields too.

`qvf_add_volume(w, kind, grid_json, dtype, rank, shape, data, label)` writes
any volume kind but `volume.series` from a JSON grid string (`origin`,
`voxel_vectors`, and optionally `units`, `periodic`, `includes_endpoint`,
`structure_ref`) and a `float` (`"float32"`) or `double` (`"float64"`) array of
the given `int64_t` shape; `qvf_add_volume_series(w, grid_json, axis_json,
dtype, shape4, data, label)` writes a `volume.series` with its `axis`.

`qvf_add_wavefunction_bloch(w, n_shells, …, n_k, kpoints_frac,
weights_or_null, n_spin, n_mo, n_ao, coeffs_re_im, energies, occupations)`
takes the shell arrays of `qvf_add_wavefunction_gto` and row-major arrays
(coefficients `[n_k, n_spin, n_mo, n_ao, 2]`, energies and occupations
`[n_k, n_spin, n_mo]`). Its basis names the structure section `structure`,
which must be periodic; `qvf_add_structure` writes molecular structures only,
so a C caller cannot yet write that structure through the C ABI.

`qvf_add_forces(w, n_atoms, gradient, stress_or_null)` and
`qvf_add_hessian(w, n_atoms, matrix)` take row-major double arrays (the
gradient dE/dR in hartree/bohr, the `[3N, 3N]` Hessian in hartree/bohr²), and
`qvf_set_energies_json` writes the root `energies` block from a JSON string.
`qvf_add_excited_states_json(w, states_json, meta_json, wavefunction_ref)`
writes an `excited_states` section from a JSON array string (`meta_json` and
`wavefunction_ref` may be `NULL`).

`qvf_add_trajectory(w, metadata_json, n_frames, n_atoms, coords,
lattice_rows_or_null, lattice_per_frame, velocities_or_null, gradient_or_null,
times_or_null, energies_or_null)` writes a `trajectory` section from row-major
double arrays: `coords`, `velocities` (Å/fs) and `gradient` (dE/dR,
hartree/bohr) are `n_frames * n_atoms * 3` doubles, `times` (fs) and
`energies` (hartree) `n_frames`, and the lattice is in the structure
convention, rows in Å: 9 doubles when `lattice_per_frame` is 0, `9 * n_frames`
otherwise (the metadata then needs `"pbc"`). `"precision": "float32"` in the
metadata stores the coordinates as float32.

`qvf_add_electric_properties_json`, `qvf_add_elastic_properties_json` and
`qvf_add_dielectric_properties_json` add the three property kinds of spec
§ 5.13 from a JSON object string; a payload with a wrong shape, an unknown key,
a missing required field or a violated symmetry or sum rule fails with a
non-zero return and writes nothing.

`qvf_set_metadata_json(w, "{\"title\":\"...\",\"license\":\"CC-BY-4.0\"}")`
sets the root `metadata` block (the automatic `created_utc` and `uuid` are
added when it lacks them), and `qvf_annotate_section(w, id, structure_ref,
run_ref, group)` sets a section's references, with `NULL` leaving a field
unset.

`qvf_add_wavefunction_gto` writes a restricted wavefunction and
`qvf_add_wavefunction_gto_unrestricted` an unrestricted one (separate α and β
coefficient matrices, energies and occupations); both create the section
`"wf"`. `qvf_set_wavefunction_overlap(w, "wf", n_ao, S)` adds the AO overlap
matrix, which validators check against spec Appendix A.1, and
`qvf_set_wavefunction_ecp_json(w, "wf", "[{\"center\": 0,
\"n_core_electrons\": 28, ...}]")` the effective core potentials.

`qvf_add_atom_properties` writes Mulliken, Löwdin and spin populations in one
call; `qvf_add_atom_property(w, "hirshfeld_charge", q, n_atoms)` adds any one
scalar `atom_properties` role, creating the section on the first call, and
`qvf_add_atom_vector_property(w, "magnetic_moment", m, n_atoms)` adds a vector
role (`m` is `n_atoms * 3` doubles, row-major). A role named
`custom.<name>` needs `qvf_set_atom_properties_meta_json(w, "{\"custom.<name>\":
{\"units\": \"e\", \"description\": \"...\", \"kind\": \"scalar\"}}")` first.

`cpp/examples/write_h2o_c.c` is a complete runnable C example (structure,
wavefunction, spectra, EPR, provenance, citations). The C ABI is part of the
same `qvf` library target — link it exactly as from C++, but with the C++ driver
(`set_target_properties(<tgt> PROPERTIES LINKER_LANGUAGE CXX)` in CMake, or link
`-lstdc++` / `-lc++`) so the C++ runtime is pulled in. **Fortran** callers
declare the same entry points with `bind(C)` interfaces and pass
`c_char`/`c_double` arrays plus null-terminated strings.

## Validating your output

Always validate what you write:

```sh
python python/qvf_reader.py result.qvf          # exits non-zero on any error
python python/qvf_reader.py --json result.qvf   # the findings as JSON
```

The validator checks the archive against the spec § 6 semantics — container
rules (unique ZIP entry names, safe member paths, size and compression-ratio
caps checked before anything is decompressed, orphan entries), manifest and
section well-formedness, id uniqueness and the reserved ids `auto_open`,
`bookmarks`, `camera` and `crossfade`, declared member sizes, sha256
integrity, JSON size and nesting
depth, byte sizing,
reference resolution, extension governance, the `wavefunction.gto` rules of
spec § 5.3 (including the Appendix A.1 diagonal of an `overlap` member,
`E_NORMALIZATION`) — and, when `jsonschema` is installed, the full JSON
Schema. With `jsonschema`, the payloads of JSON
members such as volume grids, the wavefunction `basis` and `mo_metadata`, and
trajectory, reaction, scan and vibration metadata are also checked against
their payload definitions (spec § 5), and a malformed one is reported as
`E_PAYLOAD_SCHEMA`. Each finding carries a stable code from spec
Appendix B, printed as `CODE: message`:

```text
QVF validation: result.qvf
  sections: 3
  E_SHA256_MISMATCH: member 'volumes/density.dat': sha256 mismatch
  W_ORPHAN_MEMBER: ZIP entry 'old/structure.json' is not referenced by any manifest member (orphan; consumers ignore it)
  FAILED
```

In your own test suite, call `qvf_reader.validate_qvf(path)` and assert
`report["ok"]`. Match problems by code, not by message text:

```python
from qvf_reader import validate_qvf, Limits

report = validate_qvf("result.qvf")
# {"ok", "errors", "warnings", "n_sections", "findings"}; each finding is
# {"severity": "error"|"warning", "code": "E_…"|"W_…", "where": …, "message": …}
assert not any(f["code"] == "E_SHA256_MISMATCH" for f in report["findings"])

# The resource caps are parameters (defaults: the spec § 6.1 reference values).
report = validate_qvf("huge.qvf", limits=Limits(max_member_bytes=1 << 30))
```

`errors` and `warnings` remain lists of strings (`"CODE: message"`) for
existing callers. `where` is `archive`, `entry:<zip name>`, `manifest.json`,
or a JSON Pointer into the manifest such as `/sections/2/members/data`.
Warnings — `W_ORPHAN_MEMBER`, and `W_SCHEMA_SKIPPED` when `jsonschema` is
missing — leave `ok` true. The C++ and JavaScript validators need no schema
engine: they check the schema with a checker generated from it (see
[Generated tables and the schema checker](#generated-tables-and-the-schema-checker)).

**Member sizes.** Both writers record each member's uncompressed byte count
as `size` in its member spec (spec § 4.2), and every manifest they write
declares `"byte_order": "little"` (spec § 2.2). The validators compare `size`
with the size the ZIP central directory records for the entry before reading
the member; a disagreement is `E_BINARY_SIZE`, the code that also reports a
binary member whose byte length does not match its dtype and shape. `size` is
optional, so archives written before it existed validate unchanged. If you
post-process an archive, for example recompress it, keep the bytes of every
member identical; then `sha256` and `size` stay correct.

**Orphans and updated archives.** `W_ORPHAN_MEMBER` names a ZIP entry that
the manifest does not reference. In an archive you wrote in one go it usually
points at a stray file. In an archive updated in place it is expected: an
update appends new entries and a new `manifest.json` last, and the previous
manifest and any replaced members stay behind (spec § 2.3). Such an archive
holds several `manifest.json` entries; the validators read the last one and
report each earlier one as `W_ORPHAN_MEMBER` with `where` `entry:manifest.json`,
not as the error `E_ZIP_DUPLICATE_ENTRY`, which still applies to every other
duplicated name. Repacking writes the archive anew without the orphans. How to
append to and repack an archive with the toolkit is described in "Streaming
and updating archives".

`conformance/negative/` holds one must-fail archive per rule; it shows what
each code looks like in practice and is what a third-party validator certifies
against (see `conformance/README.md`).

### Generated tables and the schema checker

The JSON Schema `spec/qvf_manifest.schema.json` is the single source of truth,
and the tables every implementation needs are generated from it rather than
copied by hand. `python3 scripts/gen_kinds.py` reads the schema,
`registry.json` (canonical kind order, readable versions) and the finding-code
table of spec Appendix B, and writes, deterministically:

- `cpp/include/qvf/kinds_generated.hpp` — `namespace qvf::gen`: one
  `KindInfo` per canonical kind with a `RoleInfo` per member role (required,
  `Format::Json` / `Binary` / `Opaque`, allowed dtypes, shape rank bounds,
  payload definition; a pattern role such as `custom.<name>` carries its
  regular expression), `kind_info()`, `kCanonicalKinds`, `payload_def(kind,
  role)`, the `Unit` enum with `unit_name()` / `parse_unit()` (re-exported by
  `qvf/units.hpp`), the finding codes and the schema `$id`;
- `python/qvf/_kinds.py` — the same tables (`KINDS`, `CANONICAL_KINDS`,
  `PAYLOAD_DEFS`, `UNITS`, `FIXED_UNITS`, `FINDING_CODES`, …), imported by
  `qvf.validate` and `qvf.units` (`qvf_kinds` remains a compatibility import);
- `cpp/src/qvf_schema_table.hpp` and `js/src/schemaTable.ts` — the schema
  compiled into a compact instruction table: one node per schema object, one
  op per keyword, pooled strings, numbers and literals, and every `pattern`
  compiled to a small Pike-VM program.

A small interpreter of that table, `cpp/src/qvf_schema_check.hpp` and
`js/src/schemaCheck.ts`, is the C++ and JavaScript validators' JSON Schema
check: no schema engine ships, yet both report `E_SCHEMA` for `manifest.json`
and `E_PAYLOAD_SCHEMA` for every JSON member with a payload definition, at the
same location (`where`, and the payload's own JSON pointer in the message) as
the Python validator, which uses `jsonschema`. The checker implements the JSON
Schema 2020-12 subset the schema uses (`type`, `const`, `enum`, `required`,
`properties`, `patternProperties`, `additionalProperties`,
`dependentRequired`, `minProperties` / `maxProperties`, `prefixItems`,
`items`, `contains` / `minContains` / `maxContains`, `minItems` / `maxItems`,
`minimum` / `maximum` / `exclusiveMinimum` / `exclusiveMaximum`, `minLength`
/ `maxLength` in code points, `pattern`, `allOf` / `anyOf` / `oneOf` / `not`,
`if` / `then` / `else`, `$ref` to `$defs`, boolean schemas; also
`uniqueItems`, `multipleOf`, `propertyNames`, `dependentSchemas`), with
`format` an annotation as in the Python reference. Patterns follow ECMA-262:
`$` matches only at the end of a string, where Python's `re` also matches
before a final newline. `python/tests/test_schema_checker_parity.py` holds
both checkers to `jsonschema` on all three corpora and on thousands of mutated
documents.

After a schema change, run `python3 scripts/gen_kinds.py` (then
`python3 scripts/amalgamate.py`); `--check` fails when a generated file is
stale, and CI runs it. The generator refuses a keyword outside the subset, a
pattern outside the supported regular-expression subset, a `*Payload`
definition that no `(kind, role)` names, and a `registry.json` that disagrees
with the schema's kinds, so a schema change can never leave the generated
checkers silently behind. Wiring a new payload definition to the role that
carries it is the one manual step: add a row to `PAYLOAD_DEFS` in
`scripts/gen_kinds.py` (payload definitions describe archive members, so the
manifest never `$ref`s them).

## Reading QVF in C++

`qvf/reader.hpp` provides the lazy `qvf::Reader`. Opening reads the central
directory and `manifest.json`; payload members remain unread until requested.
The file form keeps the archive open, and `from_bytes(std::vector<uint8_t>)`
takes ownership of an in-memory archive. Both support STORE and DEFLATE with
no external dependencies. The single-header amalgamation includes this API.

```cpp
#include "qvf/reader.hpp"
using namespace qvf;

// Validate without opening a Reader: an unreadable ZIP becomes a finding.
auto report = validate::run_file("result.qvf");
for (const auto& f : report.findings)
    std::cerr << f.code << ": " << f.message << " [" << f.where << "]\n";
if (!report.ok) return 1;  // warnings leave the archive valid

auto reader = Reader::open("result.qvf");
for (const Section* section : reader.sections_of("structure")) {
    Json structure = reader.json(*section, "structure");
    std::cout << structure.at("atoms").at(0).at("symbol").as_string();
}
if (const Section* wf = reader.section("wf")) {
    Tensor coefficients = reader.tensor(*wf, "mo_coefficients");
    // coefficients.dtype is a DType, shape is int64_t extents, bytes are
    // little-endian in C order. Decode bytes for the requested scalar type.
}
```

`manifest()` returns the lossless `Manifest` / `Section` / `MemberSpec` model
from `qvf/model.hpp`; `manifest_json()` returns the original parsed JSON.
Optional typed fields use `std::optional`. Unknown peer fields, vendor keys,
and known fields with unexpected types remain in each record's `extra`
object. `section(id)` returns null if absent. Section pointers stay valid
across a Reader move and until the owning Reader is destroyed.

`json(section, role)` loads one JSON member; `tensor(section, role)` returns
an owning `Tensor`; `bytes(section, role)` handles opaque logs, images, or
vendor members. `tensor_into(section, role, sink)` streams chunks of at most
1 MiB through a `std::function<void(const uint8_t*, size_t)>` callback without
allocating a complete tensor. The pointer is borrowed for that callback;
a chunk may split an element. CRC-32 and SHA-256 are checked while reading,
and a final integrity failure can occur after chunks have reached the sink.
Discard partial results if the call throws. Callback exceptions propagate.
All four accessors accept `verify=false` to skip SHA-256 only; ZIP CRC-32
checks remain mandatory.

`get<K>(id)` invokes `K::from_section(const Reader&, const Section&)`; the
kind type performs its own kind check and typed decoding. Custom kinds can
implement that protocol. `qvf/kinds/all.hpp` supplies concrete classes for all
canonical kinds, with typed payload fields and owning tensor results.

### Validation stages and limits

`validate::run_file(path)`, `run_bytes(bytes)`, and `run(open_zip)` return a
`Report` containing `ok`, `findings`, and `n_sections`. `reader.validate()`
validates its held archive. These default to complete validation and reuse
the existing finding codes, locations, schema checker, and resource limits.
An optional `validate::Level` stops work after the requested cumulative stage:

| Level | Work added |
|---|---|
| `Container` | Directory, entry paths, duplicate names, declared resource caps and manifest presence; no entry payloads read |
| `Manifest` | Parse manifest, schema, section/member declarations, orphan entries; no other member bytes read |
| `Payloads` | Member integrity, JSON decoding, binary dtype/shape and payload schema |
| `Semantics` | Kind-specific consistency, including dimensions across declared sections |
| `References` | Unresolved section-reference findings |
| `All` | Complete validation, equivalent to `References` |

A successful lower stage does not certify later stages. `n_sections` is the
number of entries in the parsed sections array, or zero at `Container` and
when no sections array could be read. Pass `ValidationOptions{false}` as the
options argument to skip the generated schema checker; manifest-level and
later validation then report `W_SCHEMA_SKIPPED`.

`Limits` controls `max_member_bytes`, `max_total_bytes`, `max_ratio`,
`ratio_exempt_bytes`, `max_json_bytes`, and `max_json_depth`, using the
spec § 6.1 defaults. Ratio enforcement is opt-in (`max_ratio = 0` disables
it). Reader opening checks manifest limits, and each requested member's
limits, declared size, tensor shape overflow and dtype are checked before
allocation or callback delivery. Duplicate entry names are refused except
for superseded `manifest.json` entries, whose last entry is read (spec § 2.3).

The existing `qvf/qvf_reader.hpp` API remains available: `QvfReader`,
`BinaryMember`, `read_json_member`, `read_binary_member`, `typed_manifest`,
`validate_findings`, and the static `validate_file` / `validate_bytes` retain
their signatures and full-validation defaults. The Python counterpart of
the typed manifest remains `qvf_model` (`QvfReader.typed_manifest`).

The `qvf-inspect` executable (`examples/read_qvf.cpp`) uses the new Reader
and report APIs, prints findings as `CODE: message [where]`, and exits
non-zero on errors. The `read_qvf` executable name is retained for compatibility.

## Streaming and updating archives

Both writers can update an existing archive in place, and the C++ writer can
also stream a new one, so a large grid never has to sit in memory twice. An
update follows spec § 2.3: new members are appended, a new `manifest.json` is
written last (STORED), no existing entry name is reused, and
`metadata.modified_utc` is set to the time of the update (unless you set one).
The previous manifest stays in the archive as a superseded entry; readers use
the last `manifest.json`, and validators report each superseded copy, like any
other orphan, as `W_ORPHAN_MEMBER`. `repack` writes the archive anew from its
current manifest, without orphans or superseded manifests.

### C++

```cpp
#include "qvf/qvf.hpp"
using namespace qvf;

// A streaming writer: the file is created now and every member is written to
// it as soon as it is added. write() puts manifest.json last and finishes it.
QvfWriter w({"my-code", "2.1", "density"}, "density.qvf");
w.add_structure(atoms);
Json meta = Json::object();
meta.set("note", "1 gigavoxel");
Json members = Json::object();
members.set("meta", meta);
w.add_vendor_section("x_mycode.grid", members, false, "", "grid");

// Stream a binary member into an existing section, slab by slab. sha256, CRC
// and size are computed on the fly; close() adds the member spec.
MemberSink s = w.begin_binary("grid", "data", DType::Float64, {nx, ny, nz});
for (const std::vector<double>& slab : slabs)        // little-endian bytes
    s.write(slab.data(), slab.size() * sizeof(double));
s.close();                       // throws unless 8 * nx * ny * nz bytes arrived
w.write();

// An in-place update: the manifest is loaded, add_* append, write() appends
// the new members and a new manifest.json to the same file.
QvfWriter u = QvfWriter::open_for_update("result.qvf");
u.add_spectrum("spectra.ir", ir);
u.remove_section("x_mycode_doc");              // replace a section: remove it, then
u.add_vendor_section("x_mycode.doc", doc, false, "", "x_mycode_doc");  // add it anew
std::vector<uint8_t> preview = u.to_bytes();   // the updated archive; file untouched
u.write();

repack("result.qvf", "result-packed.qvf");     // no orphans, no superseded manifests
```

- A buffered writer (`QvfWriter(source)`) accepts `begin_binary()` too; the
  bytes are then kept until `to_bytes()` / `write(path)`, which both produce
  the same archive (`write` streams it to the file).
- Only one `MemberSink` is open at a time; close it before adding anything
  else. A sink destroyed unclosed abandons its member, which never appears in
  the manifest or the central directory.
- `open_for_update` keeps root keys it does not model, sets
  `auto_metadata = false` (`created_utc` and `uuid` stay those of the
  archive), and never rewrites an existing entry: a new member whose default
  path the archive already holds gets a fresh one, `<stem>.<n><suffix>`
  (adding a custom role to an existing `atom_properties` section writes
  `atom_properties/meta.2.json`). `remove_section(id)` drops a section from
  the manifest and returns it: a buffered writer discards its pending files,
  a streaming writer drops the entries it wrote from the central directory,
  and entries the archive held before stay as orphans until `repack()`. An
  archive stamped with the withdrawn `qvf_version: 2` is written back as
  version 1. Until `write()` returns, the file being updated has no central
  directory once members stream into it; copy it first if an interruption
  must not cost the archive (spec § 2.3).
- The Python writer has three conveniences the C++ one lacks: `label=` on
  every `add_*` method, `open_for_update` from bytes, and `repack=True` on
  `write` and `to_bytes`.
- `force_zip64 = true` writes every ZIP64 structure even for small members;
  without it ZIP64 is used exactly when a size, an offset or the entry count
  passes the 32-bit limits. A 32-bit archive is byte-for-byte what earlier
  versions wrote for the same entries.

The ZIP layer underneath is public as `qvf/container.hpp`:
`container::ZipWriter` over a `FileSink`, `MemorySink` or `CallbackSink`
(non-seekable, so streamed entries carry data descriptors),
`ZipWriter::append(path)`, and `container::ZipReader`, whose `read_into()`
streams one entry in chunks under the `Limits` caps (member and total size,
and an optional positive compression-ratio policy):

```cpp
#include "qvf/container.hpp"

container::ZipReader zip = container::ZipReader::open("big.qvf");
zip.read_into("volumes/density.dat",
              [&](const uint8_t* p, size_t n) { consume(p, n); },
              container::Limits{});
```

`QvfReader::open(path)` reads from the file on demand as well, so an archive
need not fit in memory to be read or validated.

### Python

```python
from qvf_writer import QvfWriter, repack

w = QvfWriter.open_for_update("result.qvf")
w.add_spectrum("spectra.ir", frequencies=[1595.0, 3657.0], intensities=[67.0, 3.0])
w.write()                                  # appends via zipfile mode "a"

repack("result.qvf", "result.qvf")         # in place: drops orphans and old manifests

data = QvfWriter("my-code", "1.0").to_bytes(force_zip64=True)
```

To **replace** a section, remove it and add the new one under the same id.
The update never rewrites an entry (spec § 2.3), so a new member whose default
path the archive already holds gets a fresh one (`doc/doc.2.json`), and the
old members stay behind as superseded entries until the archive is repacked.
A program that saves the same document again and again passes `repack=True`.
`open_for_update` also takes an archive's bytes:

```python
u = QvfWriter.open_for_update(archive_bytes)       # or a path
u.remove_section("vibesketch_document")
u.add_vendor_section("x_vibesketch.document", json_members={"document": doc},
                     section_id="vibesketch_document")
u.add_structure(atoms, charge=-1, multiplicity=1, label="5", section_id="structure_s5")
new_bytes = u.to_bytes(repack=True)                # or u.write("doc.qvf", repack=True)
```

`remove_section` drops the members of a section added in the same writer,
and returns the removed section; it does not touch other sections, so a
reference to a removed section dangles until one with the same id is added
again. An archive stamped with the withdrawn `qvf_version: 2` opens for update
and is written as version 1.

`to_bytes()` and `write()` take `force_zip64`, which puts a ZIP64 extra field
in every local header; `zipfile` adds ZIP64 central-directory fields and end
records only when a size, an offset or the entry count needs them. The Python
writer buffers members in memory; it has no streaming entry point.

## JavaScript reader

`js/` holds `@qvf/core`, a reader and validator for `.qvf` archives in pure
TypeScript that runs in Node and in browsers (one runtime dependency, `fflate`;
SHA-256 through WebCrypto). It opens an archive from a `Uint8Array`, exposes the
typed manifest, decodes JSON members and binary members into typed arrays with
their shape (`Float32Array`, `Float64Array`, ... `BigInt64Array`), and validates
with the same finding codes as the Python and C++ validators (spec Appendix B),
the JSON Schema findings included (through the generated checker; pass
`schemaCheck: false` to skip it); it has no writer, and it is certified by
running the conformance corpus. See [`js/README.md`](../js/README.md) for a Node example, a
browser example (`js/examples/browser.html`) and the test commands.

## Section-kind cheat sheet

| To publish… | Call | Kind |
|---|---|---|
| Geometry | `add_structure` | `structure` |
| A species' charge, multiplicity, name, SMILES, InChI, InChIKey, compound label | `add_structure(..., charge=, multiplicity=, name=, smiles=, inchi=, inchikey=, label=)` | `structure` |
| Reaction scheme (reactants, reagents, products, arrow, conditions, yield) | `add_reaction_scheme` | `reaction.scheme` |
| A picture of the archive for file managers | `add_preview` | `preview` |
| Basis + MOs (GBW) | `add_wavefunction_gto` | `wavefunction.gto` |
| Bloch orbitals at several k-points (periodic) | `add_wavefunction_bloch` | `wavefunction.bloch` (provisional) |
| IR / Raman / UV-Vis / ECD / VCD | `add_spectrum` | `spectra.*` |
| NMR shifts/tensors | `add_spectrum(kind, Json)` | `spectra.nmr` |
| Per-atom charges, populations, valences, Fukui indices, energies, volumes, shieldings, magnetic moments and gradients (`mulliken_charge`, `loewdin_charge`, `hirshfeld_charge`, `iao_charge`, `spin_population`, `cm5_charge`, `chelpg_charge`, `resp_charge`, `esp_charge`, `nbo_charge`, `mbis_charge`, `bader_charge`, `ddec_charge`, `valence`, `bonded_valence`, `fukui_plus`, `fukui_minus`, `fukui_zero`, `dual_descriptor`, `atomic_energy_eh`, `atomic_volume_bohr3`, `nmr_shielding_iso_ppm`, `magnetic_moment`, `force_eh_bohr`), or a `custom.<name>` role with a `meta` description | `add_atom_properties` | `atom_properties` |
| Mayer / Wiberg bond orders | `add_bond_orders` | `bond_orders` |
| Electron density grid | `add_volume("volume.density", …)` | `volume.density` |
| Electrostatic potential | `add_volume("volume.potential", …)` | `volume.potential` |
| Laplacian of the density | `add_volume("volume.laplacian", …)` | `volume.laplacian` |
| Vector field (density gradient, current, field) | `add_volume("volume.vector", …)` | `volume.vector` |
| Complex orbital or Bloch function on a grid | `add_volume("volume.complex", …)` | `volume.complex` |
| Densities along a trajectory, in time or by state | `add_volume("volume.series", …, axis=…)` (C++ `add_volume_series`) | `volume.series` |
| Periodic (CHGCAR- or cube-style) grid | `add_volume(…, periodic=…, includes_endpoint=…)` (C++ `Grid::periodic`) | `volume.*` |
| Optimization / IRC frames | `add_trajectory` | `trajectory` |
| MD run: per-frame cell (rows, Å), velocities, gradients, times, energies, cell volumes, temperatures; float32 coords | `add_trajectory(..., lattice=, velocities=, gradient=, times=, energies=, cell_volumes=, temperatures=, precision=)` | `trajectory` |
| Normal modes | `add_vibrations` | `vibrations` |
| SCF convergence trace | `add_scf_history` | `scf_history` |
| References | `add_citations` | `citations` |
| Input + log of the run | `add_run_record` | `run.record` |
| Declarative job request (not yet run); several make a job set | `add_job_spec` | `job.spec` |
| The run of one job of a set | `add_run_record(..., job_ref=, run_status=)` | `run.record` |
| Nuclear gradient (forces are its negative), stress | `add_forces` | `forces` |
| Cartesian Hessian | `add_hessian` | `hessian` |
| Excited states (energies, strengths, transition dipoles, contributions) | `add_excited_states` | `excited_states` |
| Natural transition orbitals (hole + particle) | `add_wavefunction_gto(..., orbital_kind="nto", nto_role=, state_ref=, state_index=)` | `wavefunction.gto` |
| Transition density grid | `add_volume("volume.transition_density", …)` (C++ `add_volume_transition_density`) | `volume.transition_density` |
| Dipole, quadrupole, octupole, polarizability, hyperpolarizability | `add_electric_properties` | `properties.electric` |
| Elastic constants (Voigt stiffness, moduli) | `add_elastic_properties` | `properties.elastic` |
| Dielectric tensors, Born charges, piezoelectric tensor | `add_dielectric_properties` | `properties.dielectric` |
| Total energy, its components, energies by method | `set_energies` | root `energies` |
| Band structure | `add_bands` | `bands` |
| DOS / PDOS | `add_dos_total` / `add_dos_projected` | `dos.*` |
| Anything not yet standard | `add_vendor_section` | `x_<vendor>.*` |

For the ORCA-specific mapping (GBW → wavefunction, spectra parsing, EPR), see
[`orca_integration.md`](orca_integration.md).


### Streaming C++ Writer, typed kinds and Editor

Include `qvf/writer.hpp`, `qvf/editor.hpp` and `qvf/kinds/all.hpp` for the new
facades. `Writer` uses the same streaming backend as `QvfWriter`; all existing
`QvfWriter` signatures remain available. A writer takes a `Source` and a path,
`std::unique_ptr<container::Sink>`, or no destination for an in-memory archive.
A `CallbackSink` supports pipes and other non-seekable destinations. Members
are written as added; `finish()` writes `manifest.json` last and is idempotent.
For the default memory writer, `to_bytes()` finishes and transfers the bytes
once. A finished writer rejects further additions.

```cpp
#include "qvf/writer.hpp"
#include "qvf/kinds/all.hpp"

qvf::Writer writer({"example", "1.0", "molecule"}, "molecule.qvf");
qvf::kinds::Structure molecule;
qvf::kinds::StructureAtom atom;
atom.symbol = "He";
atom.atomic_number = 2;
atom.position = {0.0, 0.0, 0.0};
molecule.structure.atoms.push_back(atom);
molecule.fields.charge = 0;
molecule.fields.multiplicity = 1;
writer.add(molecule);
writer.finish();

auto reader = qvf::Reader::open("molecule.qvf");
auto decoded = reader.get<qvf::kinds::Structure>("structure");
```

The generated kind classes expose named member roles: for example,
`VolumeDensity.grid` / `.data`, `WavefunctionGTO.basis` / `.mo_metadata` /
`.mo_coefficients_alpha`, and `Trajectory.metadata` / `.coords`. JSON payloads
have concrete scalar, array and nested-record fields. Optional fields remain
absent unless set; an optional nullable field uses nested `std::optional` to
preserve absence versus explicit null. For example, `job.spec.basis.emplace(
std::nullopt)` writes null. Kind-specific section fields live in `.fields`;
common section fields such as `id`, `label` and `structure_ref` are direct
members. Unknown payload fields are retained in `.extra`. Canonical kind and
role definitions, including these records, are regenerated with
`scripts/gen_kinds.py`; the shared schema checker validates typed writes.

A binary role uses `TensorData`, which accepts an owning `Tensor` or a
borrowed `TensorView`. The view describes a contiguous pointer, dtype, shape
and `ByteOrder::Native`, `Little` or `Big`. Keep borrowed storage alive until
`add()` returns. The writer converts byte order and, when `output_dtype` is
set, dtype in chunks no larger than 64 KiB. Integer range errors throw;
floating narrowing uses the host IEEE conversion. `Reader::get<K>()` returns
owning tensors, so moving the decoded object or destroying the reader does
not invalidate its data. `kinds::copy_section(reader, section, writer)` copies
one canonical typed section; copy referenced sibling sections as needed.

`add_section(Section)`, `add_json`, `add_opaque` and `begin_binary(id, role,
dtype, shape)` support custom producers. A binary sink accepts little-endian
bytes through `write(pointer, count)` and must be closed before another
member is added. An abandoned or short member is omitted. Canonical `add()`
checks IDs, member declarations, JSON schemas and tensor sizes before writes;
a later conversion failure removes that new section and its new directory
entries. It does not certify semantic relationships or references across the
archive: use the complete validator after writing. Discard a destination if
its underlying I/O callback fails.

`Editor::open(path)` uses the established in-place append lifecycle. Access
its `writer()` or use `append(kind)`, `remove_section(id)` and
`set_run_status(RunStatus::Converged, run_id)`, then `commit()`. An omitted run
ID is accepted only when there is exactly one `run.record`. Removed members
remain superseded until repacked. In-place streaming can leave an incomplete
archive if interrupted.

For optional atomic replacement, use `Editor::open(path, true)`. It works in
an exclusively reserved sibling directory and leaves the original unchanged
until `commit()` renames the completed archive over it. Abandoning the editor
cleans up only that temporary copy. POSIX guarantees atomic replacement on
the same filesystem; a platform that cannot replace the destination fails
without deleting the original. `Editor::repack(in, out, Method::Deflate)`
streams only referenced members, removes superseded manifests and orphans,
and replaces the output after completing the new archive. `Method::Store`
selects uncompressed members. Repacking preserves payload bytes and their
hashes; it does not repair invalid content.

### Local Conan 2 adoption (toolkit 0.6.0)

The [local recipe](../packaging/conan/README.md) exports exact committed C/C++
source and tests installed consumers for static/shared packages. It retains
`qvf::qvf`, `qvf.pc` and the Apache notices. This path does not upload to a
registry or package compiler-specific Fortran modules.
