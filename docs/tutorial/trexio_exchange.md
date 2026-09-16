# Exchange and restart a molecular wavefunction with TREXIO

This tutorial runs water at RHF/def2-SVP, exports its wavefunction, converts
between HDF5 and text storage, and uses the imported density to start another
SCF calculation. Success means that the electron count, orbital overlap and
energy survive the exchange. These are file-interchange checks; they do not
establish basis-set convergence or accuracy against experiment.

**QVF remains the default output.** TREXIO is an optional additional file
for exchanging wavefunctions with other programs. Enabling it leaves QVF
enabled. The [TREXIO reference](../user_guide/trexio.md) lists the supported
data and the limits of native reconstruction.

## Install and run

Use the Python environment that already runs vibe-qc:

```sh
python -m pip install 'vibe-qc[trexio]'
```

From a source checkout with a working development environment, install the
extra with `python -m pip install -e '.[trexio]'` instead. All commands below
assume that environment is active and the current directory is the source
checkout. `OMP_NUM_THREADS=1` is enough for these small examples.

```sh
export OMP_NUM_THREADS=1
python examples/trexio/molecular_restart.py --case water \
    --output-dir ~/vibeqc-runs/trexio/water-hdf5
python examples/trexio/molecular_restart.py --case water --backend text \
    --output-dir ~/vibeqc-runs/trexio/water-text
```

The complete input is also available as
{download}`molecular_restart.py <../../examples/trexio/molecular_restart.py>`.
The `--output-dir` argument is required so calculation files have an explicit
destination outside the source tree. Re-running a calculation replaces its
outputs; use a new directory when preserving an earlier result.

| Output | Purpose |
|---|---|
| `water.qvf` | Default vibe-qc result archive |
| `water.trexio.h5` | HDF5 wavefunction, when `--backend hdf5` is selected |
| `water.trexio/` | Complete text-backend directory, when `--backend text` is selected |
| `water.out`, `water.system`, `water.references` | Calculation report, planned output manifest, and citations |

Keep all files inside the text directory together when copying it. A single
group file is not a complete TREXIO wavefunction.

## Understand the calculation and export

The input places O at `(0, 0, 0)` and H at `(0, +/-1.43, -0.98)` bohr.
It is a fixed-geometry, neutral, closed-shell calculation with ten electrons.
The geometry is specified in the script, so an external XYZ file is not
needed. The only extra runner options needed for export are `trexio=True`
and, for text storage, `trexio_backend="text"`:

```{literalinclude} ../../examples/trexio/molecular_restart.py
:language: python
:start-after: "# BEGIN export"
:end-before: "# END export"
:dedent: 4
```

These extracts belong to the complete script above. `molecule`, `basis_name`,
`method`, `stem` and `args` have already been set there.
Optional orbital localization is disabled to keep the example focused on
the exported canonical wavefunction.

For this input, the total energy is approximately **-75.958889023533 Ha**.
The script prints `PASS water`, an electron count near 10, an orbital
orthogonality error below `1e-9`, and a READ energy difference below `1e-8`
Ha. Last digits depend on the build and numerical thresholds. The assertions
check consistency of this run, rather than relying on the printed decimal
energy as a universal reference.

## Inspect orbitals and the density

`read_trexio` detects the storage backend. `data.basis_set()` reconstructs
the basis from its stored parameters. `data.mo_blocks()` returns native AO
order and coefficient **columns**, so no manual permutation or transpose is
needed when combining it with `compute_overlap`.

For molecular orbitals, check
$C^\dagger S C = I$ and $\mathrm{Tr}(DS)=N$.
`data.density_matrices()` uses a stored one-particle RDM when available and
otherwise constructs the density from the orbital occupations. The raw
`data.fields` mapping remains in TREXIO conventions; do not mix those raw
arrays with native matrices without conversion.

```{literalinclude} ../../examples/trexio/molecular_restart.py
:language: python
:start-after: "# BEGIN inspect"
:end-before: "# END inspect"
:dedent: 4
```

The conditional checks are also used by the OH and NaH variants below.
An error here is grounds to inspect the file, basis conventions and electron
counts before passing the wavefunction to a downstream calculation.

## Restart from the file

```{literalinclude} ../../examples/trexio/molecular_restart.py
:language: python
:start-after: "# BEGIN restart"
:end-before: "# END restart"
:dedent: 4
```

READ supplies an initial density to a new SCF calculation. It does not resume
the earlier iteration history. This low-level restart returns an SCF result;
it does not write a second job report. For ordinary all-electron job output,
the equivalent runner route is
`run_job(molecule, basis="def2-svp", method="rhf", initial_guess="read", read_from=path, output="water-read")`.

For an ECP file, `data.ecp_options()` reconstructs the stored potential as
well. READ alone must not change the Hamiltonian of a target calculation.
Reusing only its orbital coefficients with an all-electron target would be
a different calculation.

## Try an open shell and an effective core potential

```sh
python examples/trexio/molecular_restart.py --case oh \
    --output-dir ~/vibeqc-runs/trexio/oh
python examples/trexio/molecular_restart.py --case nah \
    --output-dir ~/vibeqc-runs/trexio/nah
```

| Case | Model | What to inspect |
|---|---|---|
| `oh` | UHF/def2-SVP, O-H = 1.83 bohr, doublet | Separate alpha/beta MO blocks; density traces 5 and 4 |
| `nah` | RHF/LANL2DZ, Na-H = 3.5 bohr, singlet | Ten Na core electrons replaced; two explicit electrons; restored atomic numbers 11 and 1 |

Add `--backend text` to either command to exercise the text directory route.
The NaH example asserts the ECP core count and reconstructs the potential
before restarting. It does not infer the ECP merely from a basis-set label.

## Convert storage without recomputing

The converter uses the general field API, so conversion does not require
that vibe-qc can build a native basis from the file:

```sh
python examples/trexio/convert_backend.py \
    ~/vibeqc-runs/trexio/water-hdf5/water.trexio.h5 \
    ~/vibeqc-runs/trexio/water-converted.trexio --backend text
python examples/trexio/convert_backend.py \
    ~/vibeqc-runs/trexio/water-converted.trexio \
    ~/vibeqc-runs/trexio/water-converted.h5 --backend hdf5
```

Download {download}`convert_backend.py <../../examples/trexio/convert_backend.py>`.
It refuses an existing destination. Conversion preserves supported fields
and sparse indices, but regenerates library-owned metadata. It does not
follow links to other state files or convert wavefunction conventions into
a different chemical model. Text-backend string restrictions still apply;
see [general data and conversion](../user_guide/trexio.md#general-data-api-and-conversion).

## Validate with another integral implementation

A native round trip can hide an error shared by the writer and reader.
The optional reference runner reconstructs the molecule and basis from the
file and evaluates the stored wavefunction with PySCF's own integrals:

```sh
python -m pip install pyscf trexio
python examples/regression/runner_trexio_pyscf.py \
    ~/vibeqc-runs/trexio/water-hdf5/water.trexio.h5
python examples/regression/runner_trexio_pyscf.py \
    ~/vibeqc-runs/trexio/oh/oh.trexio.h5
python examples/regression/runner_trexio_pyscf.py \
    ~/vibeqc-runs/trexio/nah/nah.trexio.h5 --tol-energy 1e-6
```

The final line starts with `VIBEQC-TREXIO-PYSCF-RESULT:` and should contain
`"verdict": "pass"`; the process exits with status zero. An optional separate
environment containing PySCF and TREXIO can run this script too. PySCF is a
reference dependency, not part of the vibe-qc runtime calculation.

The NaH comparison allows `1e-6` Ha because the two libraries use different
ECP quadratures. The all-electron energy tolerance is `1e-8` Ha. These are
implementation comparisons, not published molecular reference energies.
See [the literature comparison](../user_guide/trexio.md#comparison-with-the-paper-and-specification)
for the published conventions, normalization values, and scope of the evidence.

## Common problems and next steps

| Symptom | Action |
|---|---|
| Import error naming the TREXIO extra | Install `[trexio]` into the interpreter running the calculation |
| HDF5 backend unavailable | Install a TREXIO build with HDF5, or select `--backend text` |
| Native reconstruction rejects the basis | Use the raw field API for transport; native reconstruction supports real spherical Gaussian primitives with zero radial power |
| Electron-count or READ mismatch | Check charge, multiplicity, ECP options and the target basis before rerunning |
| Conversion refuses a text string | Use HDF5, or deliberately edit the metadata to satisfy the documented text limits |

Continue with [CI and periodic interchange](trexio_correlated_periodic.md).
For visual inspection, consult the independently released
[vibe-view manual](https://vibe-qc.com/vibe-view/docs/). A viewer's supported
subset can be smaller than the data that vibe-qc transports; in particular,
file export alone does not establish periodic or correlated visualization.
