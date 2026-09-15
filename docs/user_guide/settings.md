# Settings and defaults

vibe-qc exposes settings at three levels. Use the narrowest level that
expresses the decision:

| Setting type | Examples | Where to inspect it |
|---|---|---|
| Job argument | method, basis, output stem, optimization, requested artifacts | `help(vibeqc.run_job)` or the [API reference](../api/index.md) |
| Method option object | SCF thresholds, guess, direct/conventional mode, DFT grid | `RHFOptions`, `RKSOptions`, periodic option classes |
| Process environment | OpenMP threads, output level, performance and structured logs | [Running](../running.md), [Logging](logging.md), and the no-argument settings dump |

Defaults belong to the installed version. Do not copy a default table from an
old tutorial into a new calculation and assume it is still current. Inspect
the live objects, record intentional overrides, and retain the `.system`
manifest written by the production runner.

## Inspect all registered defaults

`vibeqc.print_settings()` prints the option classes registered with the
settings helper, followed by its runtime-environment table:

```python
import vibeqc

vibeqc.print_settings()
```

This is useful when exploring an installation, but it is deliberately broad.
For a calculation, inspect only the option object you will pass.

## Show only intentional changes

A fresh option object is the executable default. Modify it, then pass it to
`print_settings`; fields that differ from a fresh object receive a `*` marker:

```python
import vibeqc as vq

opts = vq.RHFOptions()
opts.scf_mode = vq.SCFMode.DIRECT
opts.max_iter = 160
opts.conv_tol_energy = 1e-9

vq.print_settings(opts)
```

Representative output:

```text
vibe-qc settings
(values prefixed with '*' have been modified from the default)

======================================================================
RHFOptions
======================================================================
      attribute                    current                   default
  ----------------------------------------------------------------------
    * conv_tol_energy              1e-09                     1e-08
    * max_iter                     160                       100
    * scf_mode                     SCFMode.DIRECT            SCFMode.AUTO
```

The precise fields and defaults in your output are authoritative for the
installed build; the abbreviated block above only illustrates the markers.

Use the object with the matching high-level argument:

```python
import vibeqc as vq

mol = vq.Molecule.from_xyz("water.xyz")
opts = vq.RHFOptions()
opts.scf_mode = vq.SCFMode.DIRECT
result = vq.run_job(
    mol,
    basis="cc-pvqz",
    method="rhf",
    rhf_options=opts,
    output="output-water-rhf-qz",
)
```

RHF, UHF, RKS, and UKS have separate option classes because their valid
controls differ. Periodic routes also combine general SCF settings with
route-specific options. Do not pass an RHF object to a KS or periodic driver
merely because two field names happen to match.

## Capture settings in a custom record

`vibeqc.format_settings(...)` returns the same representation as text instead
of printing it:

```python
import vibeqc as vq

opts = vq.RHFOptions()
opts.scf_mode = vq.SCFMode.DIRECT
settings_text = vq.format_settings(opts)
```

This is useful for a notebook display or a custom operator log. Production
calculations should still go through `run_job` or `run_periodic_job`: those
drivers place the effective settings, build identity, completion state, and
artifact outcomes in the normal output family.

The optional solver selector can be included in the all-defaults dump:

```python
import vibeqc as vq

vq.print_settings(solver="davidson")
```

That records the selected eigensolver with the registered default tables; it
does not change the solver by itself. A per-object dump and the solver summary
are separate views in the current API.

## Find the setting for a task

| Goal | Start with |
|---|---|
| Change molecular SCF convergence behavior | [SCF convergence](scf_convergence.md) |
| Control conventional versus direct Fock builds | [SCF modes](scf_modes.md) |
| Plan or override memory | [Memory budget](memory.md) |
| Change a molecular DFT grid | [Functionals](functionals.md) and [Molecular DFT](../tutorial/molecular_dft.md) |
| Configure a periodic Coulomb route | [Periodic methods](periodic_methods.md) |
| Converge k points or occupations | [k-points](k_points.md) and [Smearing](smearing.md) |
| Change durable log detail | [Logging](logging.md) |
| Request output artifacts | [Output files](output_files.md) |

## A reproducible override workflow

1. Start from a fresh option object on a pinned vibe-qc version.
2. Change one field for a stated physical, numerical, or resource reason.
3. Print or format the object and review every `*` line.
4. Run through the high-level driver so the normal manifest is written.
5. Compare the result with the unchanged baseline.
6. Keep the input, `.out`, `.system`, and citations together.

An override that improves convergence or speed is not automatically more
accurate. Check the observable, convergence diagnostics, and provenance as
described in [Planning a calculation](../tutorial/planning_a_calculation.md).
