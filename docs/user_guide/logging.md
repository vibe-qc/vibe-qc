# The output logger (`vibeqc.output`)

vibe-qc has **one** output surface. Every user-facing byte a calculation
emits -- the banner, the SCF trace, energies, properties, timings -- is
formatted in Python by `vibeqc.output` and written through a single
channel. Nothing in the codebase calls `print()` for `.out` content, opens
the `.out` file directly, or hand-formats a number. This page explains how
that works, both for users (log levels, where output goes) and for
contributors (how a module emits its own output).

## Where output goes

A run writes a family of sibling files next to the output stem (see
[Output files](output_files.md) for the full list). The human-readable
one is `{stem}.out`. It is produced by an **output channel**: the runner
opens one at job start, and every module that wants to say something
writes into it.

```python
from vibeqc.output import write, flush

write("  Total energy:  -76.0261408 Ha\n")
flush()
```

`write()` targets the channel the runner installed. Outside a run (a
notebook, a unit test, a library import) no channel is active and
`write()` is a silent no-op, so any writer stays callable anywhere. If you
need to catch a "forgot to install a channel" bug, set
`VIBEQC_STRICT_OUTPUT=1` and the no-op becomes a loud error instead.

For a titled block, use `section()` rather than hand-drawing an underline
rule, so widths stay consistent:

```python
from vibeqc.output import section, write

with section("Energy components"):     # title + a rule sized to it
    write("  Nuclear repulsion   9.1671 Ha\n")
```

`section(title, *, width=None, level=...)` sizes the rule to the title by
default; pass `width=N` to fix it. `section_header(title, ...)` returns the
same header string for code that builds a string rather than writing live.

## Log levels

Every message carries a **level**, and each channel has a **threshold**. A
message appears only when its level is at or below the threshold. The four
levels, from always-shown to diagnostic-only:

| Level | Shown when the channel is... | Use for |
|---|---|---|
| `QUIET` | always (even `quiet`) | essential results: final energy, converged/not-converged, fatal errors |
| `STANDARD` | `standard` and above (the default) | the normal `.out` content |
| `VERBOSE` | `verbose` and above | opt-in detail: per-shell basis dumps, full orbital tables, per-stage timings |
| `DEBUG` | `debug` only | developer diagnostics: intermediate norms, gauge checks |

An untagged `write(text)` is `STANDARD`. So a channel left at its default
`standard` threshold shows exactly the `QUIET` and `STANDARD` messages --
the same content the `.out` has always had. Raising the threshold to
`verbose` reveals the `VERBOSE` extras on top; lowering it to `quiet`
strips everything but the essentials.

Concretely, a `VIBEQC_OUTPUT_LEVEL=quiet` run of an SCF keeps the
`converged in N iterations; E = ... Ha` verdict line (it is `QUIET`) and any
[`warn()`](#warnings-and-notes) output, and drops the per-iteration table
above it (that is `STANDARD`). The default `.out` is unchanged -- the verdict
sits welded to its table exactly as before.

If an engine exposes only a final iteration count, as current basis-free
semiempirical engines do, there is no per-iteration table at any level. The
same `QUIET` verdict remains, with the real count and final energy.

Correlated-solver diagnostics follow the same channel. Setting the solver's
`verbose` option for CASSCF, selected-CI, DMRG, or v2RDM writes those messages
to the runner's `.out`; it never prints around the channel. A direct library
caller that wants interactive console diagnostics can install stdout
explicitly:

```python
from vibeqc.output import OutputChannel

with OutputChannel.to_stdout():
    result = solver.solve(hamiltonian)
```

### Setting the level

As a user, set the default with the environment variable (a name or
`0`-`3`):

```sh
VIBEQC_OUTPUT_LEVEL=verbose   python run.py     # standard + verbose
VIBEQC_OUTPUT_LEVEL=quiet     python run.py     # essentials only
VIBEQC_OUTPUT_LEVEL=debug     python run.py     # everything
```

In code, `output_level(...)` sets the default that newly-opened channels
adopt, and `channel.set_level(...)` changes a channel already open:

```python
from vibeqc.output import output_level, Level

output_level("verbose")          # or Level.VERBOSE, or 2
output_level()                   # read it back -> Level.VERBOSE
```

### Tagging a message

A module decides which band its output belongs to by passing `level=`:

```python
from vibeqc.output import write, Level

write("  Total energy:  -76.0261408 Ha\n")                 # STANDARD (default)
write("  Converged in 12 iterations\n", Level.QUIET)       # always shown
write(basis_shell_dump, Level.VERBOSE)                     # only in verbose+
write(f"  ||[F,DS]|| = {res:.3e}\n", Level.DEBUG)          # only in debug
```

That is the whole contract for making output level-aware: tag the write,
and the channel filters it. There is no separate logger object to obtain
and no per-module configuration.

### Warnings and notes

A warning is not a raw `write()`. Use `warn()`, which emits a
`  WARNING: {message}` line **at `QUIET`** -- so it survives even
`VIBEQC_OUTPUT_LEVEL=quiet` -- and simultaneously fires a machine-readable
`warning` event on the structured log:

```python
from vibeqc.output import warn, note

warn("CIF write failed; skipping structure export", role="cif")
note("dispersion grid fell back to the default (no user grid set)")
```

`note(message, ...)` is the non-problem counterpart: a `  Note: {message}`
line at `STANDARD` plus a `note` event, for something a user wants to know
(a fallback taken, a default filled in) that is not an error. Use `warn()`
for anything that should reach a user who asked for a quiet run; use
`note()` for advisory asides that a quiet run can drop.

For a guard that may be evaluated repeatedly inside one job, pass a stable
`once_key` rather than keeping suppression state in the calling module:

```python
warn(
    "lattice fold is under-converged; increase cutoff_bohr",
    once_key="bipole.fold_truncation.rhf.warning",
    drift=0.03,
)
```

The first call for that key emits both the human line and structured event;
later calls in the same active output-channel or structured-log scope emit
neither. Fresh scopes start empty. Use separate keys for note and warning
severities so a later warning cannot be hidden by an earlier note.

## Formatting numbers

Precision, field width, and units are **not** hand-written into f-strings.
A quantity is a `Quantity`, and a `FormatPolicy` decides how it renders --
which is what keeps the same physical value from being printed at five
widths under four labels. Tables compute their own column widths.

```python
from vibeqc.output import (
    Column,
    CriteriaTable,
    HeaderlessBlock,
    HeaderlessColumn,
    OutputDocument,
    Quantity,
    Table,
    write,
)

doc = OutputDocument()
doc.section("Energy components", unit_for="energy")
doc.scalar("Nuclear repulsion", Quantity(9.1671, "energy"))
doc.scalar("Total energy", Quantity(-76.0261408, "energy"))
write(doc.render())

t = Table([Column("elem"), Column("Mulliken"), Column("Löwdin")])
t.add_row("O", "-0.42", "-0.42")
t.add_row("H", "0.21", "0.21")
t.footer("sum", "0.00", "0.00")   # a total row, set apart by a rule
write(t.render() + "\n")

components = HeaderlessBlock(
    "Energy components",
    [HeaderlessColumn("<", min_width=24), HeaderlessColumn(">")],
    unit_for="energy",
)
components.add_row("Nuclear repulsion", Quantity(9.1671, "energy"))
components.add_row("Total energy", Quantity(-76.0261408, "energy"))
components.divider()
write(components.render() + "\n")

criteria = CriteriaTable()
criteria.add(
    "gmax",
    Quantity(3.27e-4, "gradient"),
    Quantity(4.50e-4, "gradient"),
    True,
)
write(criteria.render() + "\n")
```

Use `Table` when columns have semantic headers. Use `HeaderlessBlock` for a
titled label/value list or matrix whose content, rather than its title, sets
the rule width. A row annotation can sit to the right without extending that
rule, which is useful for ragged HOMO/LUMO markers. Molecular SCF and
standalone double-hybrid energy-component blocks both use this primitive, so
their energy values and unit-bearing titles follow the same active policy.
The molecular and periodic timing summaries use it as well; their iteration
counts are annotations, while a `time:ms` or `time:min` policy moves both the
values and the title away from seconds. Molecular and periodic atom summaries
also size their rules from the coordinate rows rather than a guessed constant.
Periodic basis and primitive-cell metadata use the same layout, while smearing
rows additionally route energy, temperature, and entropy through their
quantity kinds so converted values keep truthful row labels. Periodic band
extrema and crystal-orbital summaries also use the primitive: k-point and
HOCO/LUCO annotations remain outside the rule, the band values retain their
deliberate Ha/eV pair, and the crystal-orbital title and values move together
when the active energy unit changes. The Gaussian-basis and full-k
semiempirical periodic optimization routes likewise share one final-geometry
block whose energy and rule follow the rendered content.
`CriteriaTable` is the value-versus-threshold variant: it keeps both sides in
the same semantic quantity kind, converts them together, and owns the ASCII
`pass`/`fail` verdict. Its `render_inline()` form fits the same diagnostic into
one cell of a streaming progress table.

For a single value inside an existing f-string, the `render_*` helpers route
one number through the active policy while keeping that site's own field
width -- byte-identical at the default unit, converting when the unit is
moved: `render_energy` / `render_energy_labeled` (energy, with/without the
unit token), `render_duration` (seconds), `render_frequency` (cm⁻¹),
`render_temperature` (K). Reach for these when migrating a legacy
`f"{e:16.10f} Ha"` rather than restructuring the whole block into a `Table`.

Quantity kinds today: `energy`, `energy_delta`, `gradient`, `length`,
`temperature`, `duration` (wall-clock seconds; `time:ms`/`time:min`),
`frequency` (vibrational cm⁻¹; `wavenumber:THz`/`wavenumber:meV`), and
`dimensionless`. Units belong to a physical *dimension* and
precision to a *kind*, so `DEFAULT_POLICY.with_unit("energy", "eV")` moves
total energies and SCF energy changes together -- a table can never mix
Hartree and eV across its columns. To render a whole document in eV at 6
decimals, pass a policy to `render()` rather than editing the module that
produced the numbers.

## For contributors: the rules

The original policy is preserved in
[Archived `CLAUDE.md` § 16](https://vibe-qc.com/docs/)
and the archived `AGENTS.md` rule 11. The contributor rules are:

* **Emit through the channel, never `print()`.** `print()` for
  user-facing output is only acceptable in the console-script entry
  points (`_cli.py` and friends), whose job is stdout.
* **Never hand-format a number.** Use `Quantity` + the document
  primitives. A total energy is `Quantity(value, "energy")`, not
  `f"{e:16.10f} Ha"`.
* **The C++ core emits nothing.** It fills result structs and returns
  numbers across pybind11; Python renders them. This is enforced by
  `tests/test_cpp_emits_no_user_output.py`.
* **Do not change the shape of `vibeqc.output` yourself.** The channel,
  the document primitives, the writer/manifest, and `progress.py` are
  owned by the IO maintainer. Coordinate a new quantity kind, table variant,
  or lifecycle hook through the [current core issue tracker](https://github.com/vibe-qc/vibe-qc/issues).
  The [archived output-logger handover](https://vibe-qc.com/docs/)
  preserves design history; it no longer accepts new work.

### The `.out` format is frozen

`tests/test_out_format_snapshot.py` locks the `.out` format against
committed golden files (molecular RHF/RKS/UHF, Hessian, TD-DFT, double
hybrid, periodic RHF, and periodic geometry optimization).
It freezes the *scaffold* -- labels, widths, alignment, separators -- not
the numbers, so a legitimate energy change passes but a format change
trips. If you deliberately change the format, regenerate with
`VIBEQC_REGEN_GOLDEN=1 pytest tests/test_out_format_snapshot.py` and review
the diff before committing.

## Live progress vs the `.out` record

Two surfaces, on purpose:

* The **`.out` file** is the complete batch record, written through the
  channel. It is columnar, with headers.
* The **live stdout stream** (`vibeqc.progress.ProgressLogger`, the
  `progress=` / `verbose=` knobs on the runners) is for watching a run
  happen. Its per-iteration lines are self-describing (`iter 2  E = ...
  Ha  dE = ...  [0.3s]`) so each stands alone in a `tail -f` where a
  columnar header has scrolled off. These are different requirements, so
  the two formats differ by design; they are not expected to be identical.

The same distinction applies to direct experimental helpers. GAPW orbital
transformation iterations and opt-in density/Fock symmetry diagnostics use
`ProgressLogger`, so their normal direct-call stdout remains live and their
quiet modes remain silent. They do not call `print()` and they are not routed
to ambient `write()`, which would discard them when no calculation channel is
installed.

Standalone molecular workflows use the live surface too. Geometry optimizers,
relaxed scans, dimer saddle searches, and MECI optimization preserve their
opt-in stdout progress through `ProgressLogger`; molecular optimizers and
scans can instead receive a caller-owned logger when a notebook or application
wants that progress in another stream. Their persistent calculation results
and sidecars remain owned by the ambient channel and `OutputWriter`.

## See also

* [Output files](output_files.md) -- the full sibling-file family.
* [`docs/design_output_module.md`](https://github.com/vibe-qc/vibe-qc/blob/main/docs/design_output_module.md)
  -- the output module's design contract.
