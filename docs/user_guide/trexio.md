# TREXIO wavefunction files

[TREXIO](https://trex-coe.github.io/trexio/) is an open wavefunction format
with HDF5 and portable text backends. vibe-qc uses the official Python API
as a lazy optional dependency:

```sh
pip install 'vibe-qc[trexio]'
```

**QVF stays the default.** TREXIO is opt-in and adds a wavefunction exchange
artifact alongside the ordinary result archive. Start with the
[molecular exchange and restart tutorial](../tutorial/trexio_exchange.md),
then try [CI and periodic restarts](../tutorial/trexio_correlated_periodic.md).
Downloadable scripts are included in both tutorials and collected under
`examples/trexio/` in the source checkout.

## Export a calculation

```python
from vibeqc import Molecule, run_job

mol = Molecule.from_xyz("water.xyz")
run_job(mol, basis="def2-svp", method="rhf", output="water", trexio=True)
# water.trexio.h5, in addition to the ordinary job outputs
```

`trexio=True` adds a planned, manifest-recorded wavefunction file.
`trexio_backend="text"` writes `water.trexio`, a directory containing
TREXIO's text files. A string or path passed as `trexio=` selects an explicit
target. `False`, the default, writes no TREXIO file. Requested exports fail
if the route cannot supply a Gaussian wavefunction. Successful job exports
include the TREXIO paper in the citation output.

Periodic Gaussian calculations accept the same options:

```python
run_periodic_job(system, basis, method="RHF", output="crystal",
                 kpoints=[2, 2, 2], trexio=True)
```

All computed k-point and spin blocks are retained, including complex
coefficients and fractional occupations. The file carries the lattice,
reciprocal lattice, reduced k points, weights and the MO-to-k-point mapping.
A real Gamma wavefunction remains a periodic wavefunction in the file.
For periodic geometry optimizations, this sidecar describes the initial
SCF calculation, like the other periodic wavefunction sidecars; optimized
geometries are written separately.

The standalone exporter offers additional controls:

```python
from vibeqc import write_trexio

write_trexio("water.h5", mol, basis, result,
             write_mo_integrals=True, write_eri=False)
write_trexio("crystal.h5", None, basis, periodic_result, system=system,
             kpoints=kmesh.kpoints, weights=kmesh.weights)
```

`kpoints` on this Python API are Cartesian inverse-bohr coordinates;
`weights` must sum to one. They may be omitted when the result retains its
sampling. Molecular ECP parameters come from the applied SCF result. For a
custom result, pass the options used to compute it as `ecp_source=options`.
Missing ECP parameters are an error, including when the density reveals a
valence-only calculation.

Replacement files are staged beside the destination and installed after the
TREXIO handle closes successfully. A failed write preserves the previous
file. Text directories are checked for TREXIO metadata and recognized files
before replacement. `overwrite=False` refuses an existing target.

## Read and seed a calculation

```python
from vibeqc import read_trexio, run_rhf, InitialGuess

data = read_trexio("water.h5")
mol = data.molecule()
basis = data.basis_set()
options = data.ecp_options()  # RHFOptions; also reconstructs an ECP if present
options.initial_guess = InitialGuess.READ
result = run_rhf(mol, basis, options, read_from="water.h5")
```

`run_job(..., initial_guess="read", read_from="water.h5")` and the molecular
SCF drivers use the existing density projection and population checks.
READ dispatch recognizes file paths ending in `.h5`, `.hdf5` or `.trexio`;
text-backend directories are recognized regardless of their name.
An ECP target calculation must have its ECP options populated, as above;
loading an initial guess does not silently change the target Hamiltonian.
`data.ecp_options(existing_options)` also accepts UHF, RKS and UKS options.
For native periodic drivers, pass their options object; the ECP centers are
then populated through `ecp_home_centers`.

Periodic READ routes accept complete TREXIO k-point/spin wavefunctions.
For example, `run_periodic_job(..., initial_guess="read", read_from=path)`
accepts an HDF5 file or a text-backend directory, including multi-k jobs.
They verify the source basis, lattice, k points and quadrature weights through
the same restart machinery as QVF and in-memory results. A different k-point
ordering is mapped explicitly; an incompatible mesh is rejected.
RDM coherence between separate spin or k-point blocks is preserved in the
general data API and refused by the native block-density restart.
`data.periodic_system()` reconstructs the cell. Files from vibe-qc retain
the periodic dimension in their description; for foreign low-dimensional
files, supply `dim=1` or `dim=2` explicitly.

Convenience views include:

- `data.mo_blocks()`: native-order coefficient columns, energies,
  occupations, spin and k-point index.
- `data.density_matrices()`: an AO density per orbital block, using the
  stored one-particle RDM when present, otherwise the occupations.
- `data.to_libint_order(matrix)`: native AO ordering and normalization for
  an AO operator matrix.
- `data.fields`: every present dataset in the file's original conventions.

Native basis reconstruction requires real, spherical Gaussian primitives
with zero radial power. Cartesian, Slater, numerical and plane-wave data,
complex basis primitives and other format features remain accessible through
`fields` and can be written back without constructing a native basis.
Missing optional MO energies and occupations appear as NaN in convenience
views; no artificial values are written into the file.

## Coverage

| Group | Calculation exporter | General data API |
|---|---|---|
| `metadata`, `nucleus`, `electron`, `state` | Provenance, geometry, effective charges, electron counts and total energy | Every installed-library field, including state labels and links |
| `basis`, `ao`, `mo` | Spherical Gaussian basis, explicit normalizations, real/complex orbitals, energies, occupations, spin and k-point indices | All supported basis types and fields |
| `ecp` | Applied XML-library or inline scalar ECP, including zero-core potentials | All ECP fields |
| `cell`, `pbc` | Periodic cells, reciprocal vectors, all sampled k points and weights | All cell and PBC fields |
| `ao_1e_int` | Molecular overlap, kinetic, effective nuclear attraction, ECP and core Hamiltonian; available overlap/core matrices at Gamma | All real/imaginary operator fields |
| `mo_1e_int` | Molecular operators with `write_mo_integrals=True` | All fields |
| `ao_2e_int` | Molecular AO ERIs with `write_eri=True`, using quartic memory | Sparse ERIs and Cholesky factors |
| `mo_2e_int`, `amplitude` | Supplied through `extra_fields` | Sparse integrals and excitation amplitudes |
| `rdm` | Molecular one-particle density, including separate spin blocks for open-shell results | All RDMs and decompositions |
| `determinant`, `csf`, `grid`, `jastrow`, `qmc` | CASCI/CASSCF/FCI determinants; other data through `extra_fields` | All installed-library fields, including buffered arrays |

The automatic exporter covers SCF and molecular CASCI/CASSCF/FCI results.
Other correlated wavefunctions, coupled-cluster amplitudes and higher RDMs
can be supplied in their MO basis through the general API. An SCF-backed
post-SCF job does not automatically export its correlation amplitudes.
Multi-k AO integrals have no k-index dimension in TREXIO and are not
replaced by molecular or selected-k matrices.
Periodic density reconstruction uses the stored occupations and MOs.

## CI and CASSCF wavefunctions

`run_job(..., method="casci" | "casscf" | "fci", trexio=True)` writes the
actual determinant expansion, common spatial MO basis and spin-resolved
one-particle RDMs. Frozen doubly occupied core orbitals are included in every
determinant. CASSCF uses the optimized orbitals to which its CI coefficients
refer. For multiple roots, the job export selects root zero, with that root's
energy, rather than attaching the state-averaged energy to a single state.

For a low-level CASCI or CASSCF result:

```python
write_trexio("ci.h5", mol, basis, scf_result,
             ci_result=cas_result, n_core=0, root=0)
```

When `ci_result` is a CASSCF result, its cumulative rotation is applied to
`scf_result.mo_coeffs`. Other determinant solvers can provide a result-like
object with `determinants`, `ci_coeffs`, `n_active_orb` and `e_total` in the
same common spatial basis. `root` selects a stored CI root. Orbital energies
are omitted for these correlated orbital sets.

## General data API and conversion

```python
import numpy as np
from vibeqc import read_trexio_fields, write_trexio_fields, TrexioSparse

fields = read_trexio_fields("source.h5")
write_trexio_fields("portable.trexio", fields, backend="text")

# Fields use TREXIO API names and TREXIO units/index order.
fields["rdm_2e"] = TrexioSparse(
    indices=np.array([[0, 1, 0, 1]], dtype=np.int32),
    values=np.array([0.25]),
)
write_trexio_fields("with-rdm.h5", fields)
```

The API discovers fields from the installed official library rather than
bundling a separate schema. All its scalar, dense, sparse, buffered and
determinant-bitfield storage types are supported. It accepts partial files
and preserves group data across backend conversion. `fields=[...]` on the
reader selects a subset. `chunk_size=65536` controls sparse and buffered
I/O; returned arrays still occupy memory proportional to the data read.
Sparse tensors retain their stored indices without implicit symmetry
expansion. Omit empty sparse or buffered datasets.

`data.write(path, backend=...)` serializes `data.fields`. Edit that mapping
to change serialized data; the named convenience attributes are views for
native conversion. TREXIO's file-owned package version and unsafe flag are
regenerated, and its automatically generated determinant count is checked.
Each state remains a separate file; state links are preserved as strings,
and linked files are not copied or followed automatically.
UTF-8 metadata is supported, and HDF5 string reads grow their buffers to
avoid the Python wrapper's fixed-length truncation. TREXIO 2.6's text parser
limits lines to 1023 bytes and cannot preserve leading whitespace or
multiline strings. Such text writes fail before replacing the destination;
use HDF5 for those strings. String arrays cannot contain TREXIO's newline
delimiter or empty entries. Large MO label arrays use a correctly sized
buffer around the library's C API, avoiding the Python binding's fixed
4096-byte array buffer.

## Conventions and verification

The implementation follows the
[TREXIO specification](https://trex-coe.github.io/trexio/trex.html) and
Posenitskiy et al., *J. Chem. Phys.* **158**, 174801 (2023),
[doi:10.1063/5.0148161](https://doi.org/10.1063/5.0148161).

Coordinates use bohr and energies use Hartree. Effective nuclear charges are
stored for ECP nuclei; adding `ecp.z_core` recovers the atomic number even
without an element label. TREXIO's ECP radial power is libecpint's input
power minus two. Spherical AO order is `0,+1,-1,+2,-2,...`, with explicit
primitive, shell and AO factors. MO coefficients are rows in TREXIO and
columns in the native API. The reader handles interleaved shell maps and
folds AO normalization factors into native MO coefficients. ERIs use
physicists' ordering in TREXIO.

The tests reconstruct overlaps, densities and ECP Hamiltonians from exported
parameters, contract exported ERIs to recover an HF energy, exercise READ
restarts and both backend storage types, and verify failed replacements.
`examples/regression/runner_trexio_pyscf.py` independently rebuilds the
all-electron and ECP HF energies, and CI energies, from the stored
wavefunction with PySCF in a separate process. Set `VIBEQC_TREXIO_PYTHON` to
an interpreter containing TREXIO and PySCF to enable those checks in
`tests/test_output_trexio.py` and `tests/test_output_trexio_extended.py`.
These independent checks can detect a consistent AO ordering or
normalization error that a writer/reader round trip would cancel.
Without that interpreter they skip, and the skip
reason names the gate that did not run. Set
`VIBEQC_REQUIRE_TREXIO_REFERENCE=1` as well to make the missing interpreter a
failure instead of a skip; a release gate that is meant to run this check
should set it so the check cannot pass by omission.
The ECP reference check allows a microhartree for the different native and
reference ECP quadratures; the native Hamiltonian round trip is checked
separately to `2e-11` hartree.

## Comparison with the paper and specification

The [TREXIO paper, sections III and IV](https://arxiv.org/html/2302.14793v2)
defines the interchange model and compares storage backends. It does not
provide molecular energy targets for the tutorials here. Its benchmark
writes 100 million determinants: HDF5 reaches 10.4 million determinants/s
(406 MB/s), and text reaches 1.1 million/s (69 MB/s), on the authors' SSD.
Those are published TREXIO-library measurements, not vibe-qc performance
results. The small examples here test correctness and do not reproduce that
throughput experiment.

### Direct comparison with published normalization values

The [specification's basis-set example](https://trex-coe.github.io/trexio/trex.html)
lists primitive normalization factors for an H2 basis. The three values below
exercise its S, P and D shells. Comparing `libint_primitive_norm` with these
published numbers gives:

| Angular momentum | Exponent | Published primitive factor | Absolute difference in the checked build |
|---|---:|---:|---:|
| S, 0 | 33.87 | 10.006253235944540 | 0 |
| P, 1 | 1.407 | 2.1842769845268308 | 4.44e-16 |
| D, 2 | 1.057 | 1.8135965626177861 | 2.22e-16 |

The regression test uses a relative tolerance of `1e-13`. Reproduce it in a
development environment with the test and TREXIO extras installed:

```sh
python -m pytest tests/test_output_trexio.py \
    -k 'prim_factor_reproduces or spherical_ao_permutation' -q
```

These checks pin normalization and ordering. They alone do not validate
contracted functions, spin blocks, ECPs or a complete exported wavefunction.

### Independent reconstruction of the tutorial wavefunctions

The following results were obtained with the runnable tutorial inputs,
TREXIO 2.6.1 and PySCF 2.14.0. HDF5 and text gave the same differences at the
precision shown. Energies include nuclear repulsion. Geometry, basis and
active-space definitions are in the scripts, and no geometry optimization
is performed.

| Tutorial case | Stored energy / Ha | Absolute PySCF reconstruction difference / Ha | Acceptance tolerance / Ha |
|---|---:|---:|---:|
| Water RHF/def2-SVP | -75.958889023533 | 1.99e-13 | 1e-8 |
| OH UHF/def2-SVP | -75.325142658826 | 4.26e-14 | 1e-8 |
| NaH RHF/LANL2DZ ECP | -0.707676592127 | 9.66e-9 | 1e-6 |
| LiH CASCI(2,2)/STO-3G, one core orbital | -7.862504477514 | 9.77e-15 | 1e-8 |
| LiH CASSCF(2,2)/STO-3G, one core orbital | -7.881214343110 | 1.24e-14 | 1e-8 |
| LiH FCI/STO-3G | -7.882504329372 | 1.24e-14 | 1e-8 |

These are measured software-interoperability results, not literature
reference energies or uncertainty estimates for the chemical models.
PySCF rebuilds the integrals from the exported parameters and evaluates the
stored HF or CI state. It does not independently optimize the orbitals.
The looser ECP tolerance accounts for differences between integral libraries;
it must not be used to excuse an electron-count or core-occupation mismatch.

For a release check that requires the independent reference, use an
interpreter containing PySCF and TREXIO and make missing-reference skips fail:

```sh
export VIBEQC_TREXIO_PYTHON="$(command -v python)"
export VIBEQC_REQUIRE_TREXIO_REFERENCE=1
python -m pytest tests/test_output_trexio.py tests/test_output_trexio_extended.py -q
```

The periodic tutorial checks all sampled blocks, weighted populations and
a public-runner READ restart. The molecular reference checker cannot
validate periodic files. Also, the general API exposes the fields available
in the installed library; the evolving online specification can describe
newer fields than the version used for these measurements. Format coverage
does not imply that vibe-qc computes every represented method, or that a
particular downstream program accepts every file.
