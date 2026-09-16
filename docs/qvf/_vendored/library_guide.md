<!-- VENDORED from the qvf repository, docs/library_guide.md at source-80432442 (80432442f318bb679e1afcef55bea166994ea28c).
     DO NOT EDIT HERE. Regenerate from the exact upstream source commit.
     This copy lets vibe-qc build its docs without a QVF checkout. -->

# QVF writer library guide

This guide covers building, linking, and calling the two reference writers in
this toolkit: the **C++17 library** (`qvf::QvfWriter`) and the **Python**
writer (`qvf_writer.QvfWriter`). Both implement the same producer contract
described in the QVF specification (`spec/qvf-format-spec.md`); pick whichever
matches your host code.

## Contents

- [Which writer do I use?](#which-writer-do-i-use)
- [The C++ library](#the-c-library)
  - [Building](#building)
  - [Linking into your project](#linking-into-your-project)
  - [The `QvfWriter` API](#the-qvfwriter-api)
  - [Building JSON payloads](#building-json-payloads)
  - [Binary tensors](#binary-tensors)
- [The Python writer](#the-python-writer)
- [Validating your output](#validating-your-output)
- [Section-kind cheat sheet](#section-kind-cheat-sheet)

## Which writer do I use?

| | C++ library | Python writer |
|---|---|---|
| File | `cpp/include/qvf/qvf.hpp` + `cpp/src/` | `python/qvf_writer.py` |
| Dependencies | C++17 standard library only | Python stdlib + NumPy |
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

### Linking into your project

The library is small enough to vendor directly. Two options:

**A. `add_subdirectory`.** Drop the `qvf-writer/cpp` tree into your source and:

```cmake
add_subdirectory(third_party/qvf-writer/cpp)
target_link_libraries(your_target PRIVATE qvf::qvf)
```

**B. Add the sources directly.** Compile `cpp/src/qvf.cpp` into your build and
add `cpp/include` and `cpp/src` to the include path. That is the entire library
— one translation unit plus header-only helpers.

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

or pull the independently versioned QVF repository without an install step.
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

### The `QvfWriter` API

Every `add_*` method returns the section `id` it created (useful for
`volume.difference` operands and `reaction.waypoints` references). Construct a
writer, add sections and root metadata in any order, then `write()`:

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
  `volume.density/orbital/spin/elf/generic/potential/rdg`;
  `add_volume_difference`; `add_basis_ao`.
- **Wavefunction:** `add_wavefunction_gto(WavefunctionGTO)` (restricted +
  unrestricted; see the normalization note below).
- **Spectra:** `add_spectrum(kind, Spectrum)` for 1-D spectra, or
  `add_spectrum(kind, Json)` for `spectra.nmr` and other object payloads.
- **Bands / DOS:** `add_bands`, `add_dos_total`, `add_dos_projected`,
  `add_dos_coop_cohp`.
- **Trajectories / reactions / scans / vibrations:** `add_trajectory`,
  `add_reaction_path`, `add_reaction_waypoints`, `add_scan_surface`,
  `add_vibrations`.
- **Analysis / periodic:** `add_atom_properties`, `add_scf_history`,
  `add_citations`, `add_structure_symmetry`, `add_fermi_surface`,
  `add_phonon_bands`, `add_phonon_dos`, `add_equation_of_state`,
  `add_topology_qtaim`.
- **Run record:** `add_run_record` — the verbatim program input and/or full
  log (UTF-8) plus attachments; `program` names the code that ran, making the
  archive a self-contained record of the calculation.
- **Job spec:** `add_job_spec` — the declarative specification of the
  calculation the archive *requests* (JobSpecPayload JSON: `job_type`
  required, plus method/basis/functional/charge/multiplicity/kpoints/tasks/
  options). Pair with `set_provenance({... "run_status": "pending" ...})`
  for a container describing a job that has not yet run.
- **Root metadata:** `set_provenance`, `set_thermochemistry`,
  `set_dipole_moment`, `set_constraints`, `set_extensions`,
  `set_viewer_defaults`.
- **Escape hatch:** `add_vendor_section("x_<vendor>.*", members, critical?)`.

#### The wavefunction normalization flag

`WavefunctionGTO::coeffs_are_libint_normalized` is the one flag you must get
right (see spec Appendix A). If your engine stores contraction coefficients
already multiplied by the primitive norm `N_i` (libint / libcint / GBW-style
storage), set it to `true` and the writer divides them out so they apply to
`N_i`-normalized primitives, as QVF requires — `N_i` is spec Appendix A.1's
*axial* norm, one factor per shell from the total `l`, not a per-component
unit norm. If your coefficients already apply to `N_i`-normalized primitives,
leave it `false`.

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

### Binary tensors

Binary members (grids, coordinates, eigenvalues) are `qvf::Tensor` objects built
by the `tensor_*` helpers, which serialize **little-endian** regardless of host:

```cpp
Tensor rho = tensor_f32(values, {nx, ny, nz});     // float32 3-D grid
Tensor coords = tensor_f64(flat_xyz, {n_frames, n_atoms, 3});
```

`tensor_f32`, `tensor_f64`, `tensor_i32`, `tensor_i64` cover the common cases;
`shape` must multiply to the number of supplied values.

## The Python writer

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
dicts / NumPy arrays. See `python/examples/write_h2o.py`.

## C and Fortran — the C ABI

For codes that can't call C++ directly (C, Fortran via `ISO_C_BINDING`, Rust,
Julia, …), the library ships a stable **C ABI** in
[`cpp/include/qvf/qvf_c.h`](../cpp/include/qvf/qvf_c.h). All state lives in one
opaque `qvf_writer` handle; functions return `0` on success and non-zero on
error (`qvf_last_error()` gives the message). Nested payloads (provenance,
NMR/EPR spectra, vendor sections) are passed as **JSON strings** the caller
builds, so no JSON parser crosses the boundary.

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
python python/qvf_reader.py result.qvf     # exits non-zero on any error
```

The validator checks the archive against the spec § 6 semantics (sha256
integrity, byte sizing, id uniqueness, reference resolution, extension
governance) and — when `jsonschema` is installed — the full JSON Schema. In your
own test suite, call `qvf_reader.validate_qvf(path)` and assert `report["ok"]`.

## Reading QVF in C++

The toolkit also ships a **reference reader** (`qvf/qvf_reader.hpp`) that
completes the round-trip — it opens a `.qvf`, parses `manifest.json`, verifies
each member's sha256, and decodes JSON and binary members. Like the writer it is
zero-dependency: the ZIP reader, INFLATE decompressor, JSON parser, and SHA-256
are all in the toolkit, and it reads archives from *any* producer (it handles
stored, fixed-, and dynamic-Huffman DEFLATE, so PySCF/zlib/miniz output is fine).

```cpp
#include "qvf/qvf_reader.hpp"
using namespace qvf;

QvfReader r = QvfReader::open("result.qvf");
std::vector<std::string> errors;
if (!r.validate(errors)) { /* report errors */ }

// Decode a JSON member (resolved through the manifest by section id + role):
Json st = r.read_json_member("structure", "structure");
std::string sym = st.at("atoms").at(0).at("symbol").as_string();

// Decode a binary member (little-endian, with dtype + shape):
BinaryMember mo = r.read_binary_member("wf", "mo_coefficients");
std::vector<double> coeffs = mo.as_doubles();   // mo.dtype == "float64"
```

`QvfReader` verifies sha256 on every read by default (pass `verify=false` to
skip). The parsed `Json` supports `at(key)` / `at(index)` / `as_string()` /
`as_double()` / `as_int()` / `as_bool()` / `contains()` / `items()` /
`elements()`. The reader is included in the single-header amalgamation too.

## Section-kind cheat sheet

| To publish… | Call | Kind |
|---|---|---|
| Geometry | `add_structure` | `structure` |
| Basis + MOs (GBW) | `add_wavefunction_gto` | `wavefunction.gto` |
| IR / Raman / UV-Vis / ECD / VCD | `add_spectrum` | `spectra.*` |
| NMR shifts/tensors | `add_spectrum(kind, Json)` | `spectra.nmr` |
| Mulliken / Löwdin charges | `add_atom_properties` | `atom_properties` |
| Mayer / Wiberg bond orders | `add_bond_orders` | `bond_orders` |
| Electron density grid | `add_volume("volume.density", …)` | `volume.density` |
| Electrostatic potential | `add_volume("volume.potential", …)` | `volume.potential` |
| Optimization / IRC frames | `add_trajectory` | `trajectory` |
| Normal modes | `add_vibrations` | `vibrations` |
| SCF convergence trace | `add_scf_history` | `scf_history` |
| References | `add_citations` | `citations` |
| Input + log of the run | `add_run_record` | `run.record` |
| Declarative job request (not yet run) | `add_job_spec` | `job.spec` |
| Band structure | `add_bands` | `bands` |
| DOS / PDOS | `add_dos_total` / `add_dos_projected` | `dos.*` |
| Anything not yet standard | `add_vendor_section` | `x_<vendor>.*` |

For the ORCA-specific mapping (GBW → wavefunction, spectra parsing, EPR), see
[`orca_integration.md`](orca_integration.md).
