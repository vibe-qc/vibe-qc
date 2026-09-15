# TREXIO wavefunction files

[TREXIO](https://github.com/TREX-CoE/trexio) is the TREX Centre of
Excellence's open wavefunction container (Posenitskiy *et al.*, *J. Chem.
Phys.* **158**, 174801 (2023), [doi:10.1063/5.0148161](https://doi.org/10.1063/5.0148161)).
One file carries the nuclei, the Gaussian basis with its normalization
conventions spelled out, the molecular orbitals, and optionally integrals
and density matrices, in a layout that Quantum Package, CHAMP, QMC=Chem,
TurboRVB, QMCkl, and (through `trexio-tools`) PySCF and ORCA all read
without a per-program converter. vibe-qc writes TREXIO files from any
molecular SCF and reads them back into its own objects.

## Install

The TREXIO Python API is an optional extra (BSD-3-Clause, never bundled;
see [Licensing](../license.md)):

```sh
pip install 'vibe-qc[trexio]'
```

Without it, `run_job(trexio=True)`, `write_trexio` and `read_trexio` raise
an `ImportError` that names the extra. Nothing else in vibe-qc depends on
it.

## Write a file from a job

```python
from vibeqc import Molecule, run_job

mol = Molecule.from_xyz("h2o.xyz")
run_job(mol, basis="def2-svp", method="rhf", output="h2o", trexio=True)
# -> h2o.out, h2o.molden, ..., h2o.trexio.h5
```

- `trexio=True` writes `{output}.trexio.h5` with the HDF5 back end.
- `trexio_backend="text"` writes `{output}.trexio` instead: a *directory*
  holding one `<group>.txt` per group, TREXIO's portable text back end.
- `trexio="path/to/file.h5"` writes to that path.

The artefact is declared in the output plan, recorded in the `.system`
manifest (bytes and SHA-256 for the HDF5 file), announced in the `.out`,
and cites the TREXIO paper in `.bibtex` / `.references` only when the file
was actually written. Like `write_molden_file=True`, a TREXIO request is a
guarantee: a route that exposes no Gaussian AO wavefunction (the
semiempirical and MLIP routes) refuses *before* the calculation starts.

## Write and read a file directly

```python
from vibeqc import BasisSet, Molecule, run_rhf, write_trexio, read_trexio

mol = Molecule.from_xyz("h2o.xyz")
basis = BasisSet(mol, "def2-svp")
result = run_rhf(mol, basis)

write_trexio("h2o.h5", mol, basis, result)             # HDF5
write_trexio("h2o.trexio", mol, basis, result, backend="text")

data = read_trexio("h2o.h5")
mol2 = data.molecule()          # Molecule (bohr, charge, multiplicity)
basis2 = data.basis_set(mol2)   # BasisSet rebuilt from the file's basis group
for block in data.mo_blocks():  # libint AO order, columns are MOs
    print(block.spin, block.coefficients.shape, block.occupations.sum())
print(data.energy, data.metadata["code"])
```

`read_trexio` returns a `TrexioData` whose AO-indexed arrays are kept in
the file's own AO order; `mo_blocks()`, `basis_set()` and
`to_libint_order()` do the mapping back to vibe-qc's order. Restricted
open-shell and unrestricted files come back as one block per spin.

## What is written

| Group | Content |
|---|---|
| `metadata` | `code = ["vibe-qc <version>"]`, `description` (job label, version, git revision), TREXIO's own `package_version`. No author is written unless you pass `author=`. |
| `nucleus` | `num`, `charge`, `coord` (**bohr**), `label`, `repulsion` |
| `electron` | `num`, `up_num`, `dn_num` |
| `basis` | `type = "Gaussian"`, `shell_num`, `prim_num`, `nucleus_index`, `shell_ang_mom`, `shell_factor`, `r_power`, `shell_index`, `exponent`, `coefficient`, `prim_factor` |
| `ao` | `cartesian = 0`, `num`, `shell`, `normalization` |
| `mo` | `type` (`RHF`, `UKS`, ...), `num`, `coefficient`, `energy`, `occupation`, `spin`, `class` |
| `ao_1e_int` | `overlap`, `kinetic`, `potential_n_e`, `core_hamiltonian` (pass `write_integrals=False` to omit) |
| `state` | `num = 1`, `id = 0`, `current_label`, `energy` (the SCF total energy) |

Supported results: molecular RHF, UHF, RKS, UKS, and restricted open-shell
(ROHF / ROKS) results of `run_job` or the `run_*` drivers.

## What is not written

- **`ecp`.** A run that used an effective core potential is *refused*
  (`ValueError`), both when the runner knows it and, as a safety net, when
  the density integrates to fewer electrons than the molecule has. A file
  with full nuclear charges next to valence-only orbitals would be an
  incomplete Hamiltonian that no reader could detect.
- **`rdm`.** The one-particle density of a single determinant is
  `diag(occupation)` in the MO basis and follows from `mo.occupation`;
  correlated density matrices are a later increment.
- **`ao_2e_int`, `mo_1e_int`, `mo_2e_int`**, **`cell` / `pbc`** (this writer
  is molecular; Bloch orbitals are refused), **`determinant` / `csf`**.
- **Cartesian shells.** vibe-qc evaluates spherical-harmonic shells only.

## Conventions

These are the conventions the file relies on, with the section of the
TREXIO specification (`trex.org`, v2.6.1) each comes from. The TREXIO paper
itself is on the companion literature library's wishlist, so the
implementation follows the library's own specification and is pinned by
the checks in the next section.

- **Units.** "All data are stored in atomic units": coordinates in bohr
  (vibe-qc's internal unit), energies in Hartree.
- **Radial functions** (basis group). TREXIO stores
  `R_s(r) = N_s r^{n_s} sum_k f_ks a_ks exp(-gamma_ks r^2)` with `r_power`
  `n_s = 0` for Gaussians. vibe-qc writes `prim_factor` `f_ks = N(alpha, l)
  = (2 alpha/pi)^(3/4) (4 alpha)^(l/2) / sqrt((2l-1)!!)`, the norm of the
  axial Cartesian primitive `x^l exp(-alpha r^2)`, which reproduces the
  specification's worked H2 example digit for digit; `coefficient`
  `a_ks = c_eff / f_ks` where `c_eff` is libint's stored contraction
  coefficient; and `shell_factor = 1`. The product `f_ks a_ks` is therefore
  exactly the coefficient vibe-qc evaluated with, and `a_ks` coincides with
  PySCF's `bas_ctr_coeff` (measured to 1e-8 on H2O/def2-SVP), so vibe-qc and
  `trexio-tools`' PySCF converter write the same basis group.
- **Angular functions and normalization** (ao group). Every AO is a real
  regular solid harmonic `S_l^m = sqrt(4 pi/(2l+1)) r^l Y_l^m`; all `2l+1`
  members of a shell share one norm and are unit-normalized, so
  `ao.normalization = 1` throughout and `ao.cartesian = 0`.
- **AO order** (ao group). TREXIO orders real solid harmonics
  `m = 0, +1, -1, +2, -2, ...` (`p` is `pz, px, py`); libint orders them
  `m = -l, ..., +l` (`p` is `py, pz, px`). The permutation is applied to
  the MO rows and to both indices of every `ao_1e_int` matrix.
- **MO layout** (mo group). `mo.coefficient` is `[mo.num, ao.num]`
  row-major, one MO per row. Unrestricted and restricted open-shell results
  are written as spin-orbitals: the alpha block (`mo.spin = 0`) followed by
  the beta block (`mo.spin = 1`), occupations 1/0, so `mo.num` is twice the
  number of spatial orbitals; a closed-shell restricted result is written
  once with occupations 2/0. `mo.class` is `Inactive` for an occupied and
  `Virtual` for an empty orbital.
- **One-electron integrals.** `potential_n_e` is `<p|V_ne|q>` with
  `V_ne = -Z_A/|r - R_A|`; `core_hamiltonian = kinetic + potential_n_e`.

## How the conventions are verified

- **In-process round trip** (`tests/test_output_trexio.py`): H2O/def2-SVP
  RHF and the OH radical UHF are written on both back ends and read back;
  nuclei, basis parameters, MO coefficients, energies, occupations and the
  one-electron matrices agree to 1e-12, and `C^T S C = 1` holds with the
  overlap rebuilt from the *file's* basis group.
- **Out-of-process PySCF check** (`examples/regression/runner_trexio_pyscf.py`,
  CLAUDE.md § 10): in a separate interpreter with `trexio` and `pyscf`
  installed, a PySCF `Mole` is rebuilt from the file's `nucleus` and `basis`
  groups alone; the stored MOs must be orthonormal in PySCF's overlap, the
  file's overlap must match PySCF's, and the SCF energy rebuilt from the
  stored MOs and occupations with PySCF's own integrals must agree with the
  file's `state.energy` to 1e-8 Ha. A wrong permutation, phase or
  normalization factor moves that energy by millihartree. Run it with
  `VIBEQC_TREXIO_PYTHON=/path/to/python pytest tests/test_output_trexio.py`;
  the test skips when that interpreter is not configured.

## Citation

Writing or reading a TREXIO file cites Posenitskiy *et al.* 2023 in the
job's `.out` references block, `.bibtex` and `.references`
(`routes.libraries.trexio` in the citation database), alongside the basis
set and method citations of the run.
