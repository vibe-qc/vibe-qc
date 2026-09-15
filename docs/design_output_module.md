---
orphan: true
---

# Design, `vibeqc.output` (unified output, logging, and citation surface)

Queue paths in this design refer to the independent [vibe-queue repository](https://github.com/vibe-qc/vibe-queue).

**Status (2026-05-18, `claude/mystifying-mcnulty-38ab6e`)**:
**implemented and landed on `main`.** This document remains the
design contract; the five decisions locked at the bottom of this
section all held through implementation. The phased roadmap below
(O1-O7) plus the pre-v1.0 follow-ups (R1-R7, D1-D5, D7a) have
shipped, see the [§ Implementation status](#implementation-status)
section at the end of the doc for the per-phase commit map. The
runner.py dispatch-overhaul shipped as D7a (the `OutputWriter`
coordinator now owns the `.system` manifest lifecycle); the
remaining conversion of individual writer calls to `dispatch_role` (D7b) was
deferred at that point. Its molecular and periodic milestones shipped on
2026-07-15 and 2026-07-16 respectively; the headerless document primitive
followed on 2026-07-16. Rationale and current status are in the
Implementation-status section.

**Goal**: consolidate the currently diffuse output, logging, and
citation surface of vibe-qc into a single coordinator package
(`vibeqc.output`) that

- produces a *declarative output plan* at job start, before any
  compute, so the `vq` queue knows what files to monitor and copy back
  and the user can run `--vibeqc-dry-run` to inspect expected
  artefacts;
- carries every existing artefact (`.out`, `.system`, `.molden`,
  `.scf.jsonl`, `.perf`, `.traj`, `.dump`) through a single dispatch
  layer with zero behaviour change on the first cut;
- adds new always-on artefacts users expect from a QC program
  (final-geometry `.xyz`, citation files `.bibtex` + `.references`)
  and the periodic / volumetric formats (`.cif`, `POSCAR`, `.xsf`,
  `.cube`) in subsequent phases;
- emits a citation block on every run, assembled from a single
  TOML-backed reference database, covering vibe-qc itself plus every
  basis / functional / method / linked library actually exercised by
  the job.

**Non-goal (in this design)**: the high-blast-radius full
coordinator rewrite scheduled before v1.0. This doc covers the
*thin-layer* phase landing in v0.8.x onwards. The public API
(`OutputPlan`, `vibeqc.output.citations.assemble`, the file-format
identities) is being designed to survive the v1.0 rewrite intact, 
internals under `output/_adapters/` are explicitly transitional.

## Locked decisions

1. **Module name**: `vibeqc.output`. Submodule `vibeqc.output.citations`
   for the reference database + writers. ([§ Module shape](#module-shape))
2. **Refactor depth, Phase O1**: thin adapter layer over existing
   writers. `runner.py` keeps its current call sites; `OutputPlan` +
   manifest-status hooks are added; no writer module is relocated.
   ([§ Phased roadmap](#phased-roadmap))
3. **Citation database location + enforcement**: single source of
   truth at `python/vibeqc/output/citations/database.toml`. CI test
   fails if any method / functional / basis-set / library used by
   vibe-qc lacks a route. Dev chats updating a feature own the
   matching DB update; clause added to `AGENTS.md`.
   ([§ Citation database](#citation-database))
4. **Manifest layout**: the existing `{stem}.system` TOML grows two
   new sections, `[plan]` (declared at job start) and `[outputs]`
   (filled in as artefacts land). No new sibling file. The
   fixed-shape rule from
   [`docs/user_guide/output_files.md`](user_guide/output_files.md)
   § "Sample manifest" still holds, sections never disappear, keys
   never get renamed. ([§ .system schema extension](#system-schema-extension))
5. **basissetdev-conditional citations**: `database.toml` on `main`
   covers only bundled assets. The 87 BSE-fetched basis sets get
   their own sibling `database_basissetdev.toml` that lives on the
   `basissetdev` branch and is loaded conditionally. Matches the
   CLAUDE.md § 4 rule that basissetdev does not merge into main for
   v0.8.x. ([§ basissetdev integration](#basissetdev-integration))

## Context: where output content lives today

[`python/vibeqc/runner.py:297`](../python/vibeqc/runner.py) is the
single user-facing entry point that emits the full output family. It
dispatches to six separate writer modules with a mix of always-on /
opt-in flags:

| Sibling file | Writer | Trigger |
|---|---|---|
| `{out}.out` | [`scf_log.format_scf_trace`](../python/vibeqc/scf_log.py) | always |
| `{out}.molden` | [`io.molden.write_molden`](../python/vibeqc/io/molden.py) | `write_molden_file=True` |
| `{out}.system` | [`system_info.write_system_manifest`](../python/vibeqc/system_info.py) | always |
| `{out}.scf.jsonl` | [`structured_log.StructuredLog`](../python/vibeqc/structured_log.py) | `structured_log=True` or `VIBEQC_STRUCTURED_LOG` |
| `{out}.perf` | [`perf.PerfTracker`](../python/vibeqc/perf.py) | `perf_log=True` or `VIBEQC_PERFLOG` |
| `{out}.traj` | [`io.trajectory`](../python/vibeqc/io/trajectory.py) (ASE) | `optimize=True` |
| `{out}.dump[.*.npy]` | [`crash_dump.dump_on_failure`](../python/vibeqc/crash_dump.py) | exception or non-converged SCF |

Verbosity gates: `verbose=` / `VIBEQC_VERBOSE`, `progress=` /
`VIBEQC_LIVE_LOGGING`, `use_logging=`. Live-progress emission is
funnelled through
[`progress.ProgressLogger`](../python/vibeqc/progress.py).

What is **missing**:

- **No plan emitted before compute starts.** A `vq` daemon watching a
  job's workspace can't tell *"these are the files I should expect"*
  until they land, and a crashed job that wrote nothing is
  indistinguishable from a successful job that finished in 80 ms.
- **No central registry of "actually written files".** vq currently
  tar-streams the entire workspace
  ([`vibe-queue/src/vq/fetch.py:170`](https://github.com/vibe-qc/vibe-queue/blob/main/src/vq/fetch.py));
  there is no concept of "fetch only the vibeqc-declared outputs".
- **No citation surface at runtime.** Users reading
  `output-h2o.out` see the SCF table but no list of papers to cite.
  Cross-reference to
  [`docs/citing.md`](citing.md) is manual. [`CITATION.cff`](../CITATION.cff),
  [`docs/user_guide/functionals.md` § Citations](user_guide/functionals.md),
  and the `.g94` basis-set headers all carry their own human-readable
  copies, three sources of drift.
- **`libxc`'s `xc_func_info_get_references` is unwrapped.** vibe-qc
  links libxc but does not query the per-functional reference list it
  exposes.
- **Periodic + crystal formats unevenly covered.** [`poscar.py`](../python/vibeqc/poscar.py),
  [`xsf.py`](../python/vibeqc/xsf.py), [`cube.py`](../python/vibeqc/cube.py),
  and [`bands.py`](../python/vibeqc/bands.py) exist but are not wired
  into `run_job` and have no per-job artefact convention.

## Module shape

```
python/vibeqc/output/
  __init__.py              # public API: OutputPlan, PlannedFile, OutputWriter,
                           #             register_writer, assemble_citations
  plan.py                  # OutputPlan + PlannedFile dataclasses
  writer.py                # OutputWriter -- owns stem, dispatches to adapters
  manifest.py              # .system [plan] + [outputs] section emission/update
  formats/
    __init__.py
    xyz.py                 # {stem}.xyz writer (P1)
    cube.py                # {stem}.cube wiring around existing cube.py (P2)
    crystal.py             # {stem}.cif / POSCAR / .xsf wiring (P2)
    population.py          # {stem}.population.{txt,json} (P2)
    fchk.py                # placeholder until P3
  citations/
    __init__.py            # public API: assemble(), to_bibtex(), to_plain()
    registry.py            # CitationRegistry + route walk
    database.toml          # the single source of truth
    bibtex.py              # → {stem}.bibtex
    plain.py               # → {stem}.references
    libxc_bridge.py        # wraps xc_func_info_get_references()
  _adapters/               # TRANSITIONAL -- to be inlined in v1.0 rewrite
    __init__.py
    molden_adapter.py      # wraps io.molden.write_molden in OutputWriter shape
    scf_log_adapter.py     # wraps scf_log.format_scf_trace
    system_info_adapter.py # wraps system_info.write_system_manifest
    structured_adapter.py  # wraps structured_log.StructuredLog
    perf_adapter.py        # wraps perf.PerfTracker
    crash_dump_adapter.py  # wraps crash_dump.dump_on_failure
    trajectory_adapter.py  # wraps io.trajectory
```

Existing writer modules stay in place during Phase O1. The
`_adapters/` shim package provides an `OutputWriter`-compatible
wrapper for each, so `runner.py` ends up calling a single dispatcher
instead of six unrelated functions. The adapter shims are explicitly
flagged transitional in their docstrings, they get inlined into
their respective `formats/*.py` cousins during the pre-v1.0 rewrite.

## `OutputPlan` and `PlannedFile`

Single source of truth for "what does this job produce". Frozen
dataclasses, hashable, JSON-serialisable:

```python
# python/vibeqc/output/plan.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

OutputRole = Literal[
    "log",          # .out
    "manifest",     # .system
    "orbitals",     # .molden, .fchk
    "geometry",     # .xyz, .cif, POSCAR, .xsf
    "density",      # .cube (density)
    "orbital_vol",  # .cube (per-orbital)
    "population",   # .population.{txt,json}
    "citations",    # .bibtex, .references
    "trajectory",   # .traj
    "perf",         # .perf
    "structured",   # .scf.jsonl
    "crash",        # .dump + .dump.*.npy
    "checkpoint",   # .h5 (P4)
    "bands",        # .bands.dat, .bands.gnuplot
]

@dataclass(frozen=True)
class PlannedFile:
    role: OutputRole
    path: Path
    format: str          # "text" | "toml" | "json" | "ndjson" | "molden" |
                         # "xyz" | "cif" | "cube" | "ase-traj" | "bibtex" | ...
    always: bool         # True = guaranteed; False = conditional
    description: str     # one-line, surfaced by `vq submit --dry-run`
    # Filled in during/after the run by writers:
    written: bool = False
    bytes: int | None = None
    sha256: str | None = None
    checksum_status: str = "pending"
    wall_time_s: float | None = None

@dataclass(frozen=True)
class OutputPlan:
    stem: Path
    job_kind: Literal["molecular_scf", "periodic_scf", "opt",
                      "hessian", "post_scf"]
    method: str          # "RHF", "RKS", "UHF", "UKS", ...
    basis: str           # name as given to run_job
    functional: str | None
    files: tuple[PlannedFile, ...]
    options_digest: str  # short hex hash of the resolved SCF options
                         # (lets vq detect "did this run change?")
```

`OutputPlan.from_run_job_kwargs(...)` is the factory used by
`runner.py`; it consumes the same kwargs the user passed (`output`,
`write_molden_file`, `optimize`, `perf_log`, `structured_log`,
`citations`, …) plus method/basis/functional and emits the frozen
plan. The plan is serialised into the `.system` manifest's `[plan]`
section *before any compute starts*.

The Molden and population sidecar keywords on the public runners are
independently tri-state. `None` is capability-aware auto mode: Gaussian-basis
SCF routes declare and emit both formats. Molecular SCC-DFTB and GFN2-xTB also
declare population files from their method-native net atomic Mulliken charges;
the other sections carry structured unsupported markers because the Gaussian
AO formulas do not apply to their minimal Slater-orbital models. Their Molden
row stays inapplicable because the current Molden writer is GTO-specific.
Other basis-free semiempirical, MLIP, and solver-only routes omit unsupported
rows. `False` is an explicit opt-out. `True` is a guarantee; if the selected
route cannot supply the contract required by that individual writer, the
runner raises before calculation and before creating a misleading plan.

The conditional files (`crash`, `trajectory`) are emitted with
`always=False` and a `description` so vq can communicate to the user
*"may produce: output-h2o.dump (only on SCF failure)"*.

## `.system` schema extension

The existing `.system` shape
([`docs/user_guide/output_files.md`](user_guide/output_files.md))
stays a superset, `[vibeqc] [host] [cpu] [memory] [python]
[libraries] [validation] [run]` are unchanged. Two new sections land:

```toml
# --- existing sections unchanged ---
[vibeqc]
version = "0.15.0"
codename = "Neese's Cheetah"
git_sha = "0123456789abcdef0123456789abcdef01234567"
# ...

# --- NEW: declared at job start, before any compute ---
[plan]
stem            = "output-h2o"
job_kind        = "molecular_scf"
method          = "RHF"
basis           = "6-31g*"
functional      = ""                       # "" not nil for fixed-shape rule
options_digest  = "f3a2c1..."              # short hex
status          = "running"                # → "complete" | "crashed"
declared_at_iso = "2026-05-18T10:42:00Z"

[[plan.files]]
role        = "log"
path        = "output-h2o.out"
format      = "text"
always      = true
description = "Human-readable SCF log (banner, iter table, orbital block)."

[[plan.files]]
role        = "manifest"
path        = "output-h2o.system"
format      = "toml"
always      = true
description = "Runtime manifest (this file) -- hardware, libs, plan, outputs."

[[plan.files]]
role        = "orbitals"
path        = "output-h2o.molden"
format      = "molden"
always      = true
description = "Molecular orbitals -- coefficients, energies, occupations."

[[plan.files]]
role        = "geometry"
path        = "output-h2o.xyz"
format      = "xyz"
always      = true
description = "Final geometry in Angstrom."

[[plan.files]]
role        = "citations"
path        = "output-h2o.bibtex"
format      = "bibtex"
always      = true
description = "BibTeX entries for every method/basis/library cited."

[[plan.files]]
role        = "citations"
path        = "output-h2o.references"
format      = "text"
always      = true
description = "Plain-text reference list (Chicago-ish formatting)."

[[plan.files]]
role        = "crash"
path        = "output-h2o.dump"
format      = "toml"
always      = false
description = "Post-mortem snapshot -- only written on SCF failure."

# --- NEW: filled in as artefacts land; status flips at job end ---
[outputs]
finished_at_iso = ""                        # filled at job end
status          = "running"                 # → "complete" | "crashed"
# Per-file rows are appended as each writer reports completion. Same
# path as the matching [[plan.files]] row, plus the runtime fields:

[[outputs.files]]
path         = "output-h2o.out"
written      = true
bytes        = 4231
sha256       = "ab12cd34..."
checksum_status = "sha256"
wall_time_s  = 0.082
error        = ""

[[outputs.files]]
path         = "output-h2o.system"
written      = true
bytes        = 0
sha256       = ""
checksum_status = "self-excluded"
wall_time_s  = 0.001
error        = ""

# Optional artefact that failed (population dump, citations, QVF, …):
[[outputs.files]]
path         = "output-h2o.population.txt"
written      = false
bytes        = 0
sha256       = ""
checksum_status = "failed"
wall_time_s  = 0.0
error        = "OSError: [Errno 28] No space left on device"
# ... etc
```

**The checksum contract.** ``checksum_status`` is fixed-shape on every output
row: ``pending`` before a writer runs, ``sha256`` when ``sha256`` contains the
digest of the landed bytes, ``unavailable`` when a reported file could not be
hashed, ``failed`` when its writer failed, and ``self-excluded`` for the
``.system`` row. The last state is deliberate. A manifest cannot contain the
SHA-256 or size of its own final bytes because inserting either value changes
those bytes. The lifecycle therefore records that the manifest landed but
leaves ``bytes = 0`` and ``sha256 = ""`` as sentinels. Consumers must not hash
an intermediate manifest and present that digest as final; legacy self rows
with a non-empty digest are stale by construction.

**The ``error`` column.** Every ``[[outputs.files]]`` row carries a
fixed-shape ``error`` cell. For successful writes it is the empty
string; for a writer that raised on a *declared* artefact the runner
records the exception type + message via
:func:`vibeqc.output._errors.warn_output_failure` (see
[the non-fatal failures section](#output-writer-failures)). This
keeps the fixed-shape rule intact for downstream parsers and lets
``vq fetch`` distinguish "writer never tried" (``written=false``,
``error=""``) from "writer tried and failed" (``written=false``,
``error="<exc-type>: <msg>"``).

**Atomicity rule**: `[plan]` is declared once at job start and retained
unchanged. `[outputs]` is rewritten in place after every file lands, the whole
`.system` file is rewritten atomically (write-to-tmp + rename) to
avoid a half-written manifest if the job is killed mid-update. The rewrite
target is fixed at construction: a relative stem is anchored to the directory
the job started in, so a later `chdir` cannot move a rewrite elsewhere
(vibe-qc#127). The rendered `path` fields keep the declared spelling.

**vq watch pattern** (Phase O4): the daemon polls `{stem}.system`,
parses `[outputs].status`. `"running"` means alive; `"complete"`
means ready to fetch; absence of either after the job-end timestamp
crosses a threshold means crashed. The `[[plan.files]]` rows tell vq
which paths to fetch.

`"complete"` is a completeness assertion: every `[[plan.files]]` row with
`always=true` has a written outcome. `ManifestUpdater.finish()` verifies this
invariant after marking the self-excluded manifest row. If a guaranteed row is
still pending or failed, it records `checksum_status="failed"`, preserves any
writer error, writes `status="crashed"`, and raises
`IncompleteOutputError`. An `always=false` conditional row may remain pending
on a successful run.

## Output writer failures

An artefact writer failure is caught locally so remaining writers, diagnostics,
and manifest bookkeeping can finish. It is surfaced immediately as a warning
and failed manifest row. A polished
`vq fetch` workflow built on top of half-broken deliverables is
indistinguishable from a fully-correct one until the user opens the
file. So every formerly silent ``except Exception: pass`` block in
the output path now routes through
:func:`vibeqc.output._errors.warn_output_failure`.

The catch is not permission to claim success. If the failed writer owns an
`always=true` plan row, finalization is hard: `finish()` records a crashed
manifest and raises `IncompleteOutputError`. Only failures belonging solely to
conditional `always=false` rows remain soft through finalization.

### Categories

Each non-fatal catch declares a category so the user sees what kind
of failure happened and downstream tooling can filter:

| Category                 | Meaning                                                                                                      | Default disposition                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------ | -------------------------------------- |
| `optional_artifact`      | An artefact writer (QVF, cube, population, citations, .xyz, POSCAR, CIF, …) raised; no file produced.        | warn + manifest row; hard at finish when `always=true` |
| `manifest_recording`     | A file landed on disk but the manifest bookkeeping (stat, hash, TOML rewrite, status flip) raised.           | soft, warn + manifest row when possible |
| `cleanup`                | Post-job cleanup / finalization raised.                                                                      | soft, warn                            |
| `compatibility_fallback` | A best-effort compatibility shim (ASE trajectory frame energy probe, QVF sub-section probe, …) raised.       | soft, warn                            |
| `unrecoverable`          | Something that should have been fatal got caught by a broad handler, treated as a bug.                      | warn at ERROR + ``logger.error`` with traceback |

### Failure surface, what is soft vs hard

| Failure path                                                     | Where caught                          | Category                 | Visibility                                                                                  |
| ---------------------------------------------------------------- | ------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------- |
| ASE trajectory frame energy probe                                | `runner.run_job`                      | `compatibility_fallback` | UserWarning                                                                                 |
| Trajectory read for QVF (post-optimization)                      | `runner.run_job`                      | `optional_artifact`      | UserWarning                                                                                 |
| Manifest `crash` status flip after an SCF exception              | `runner.run_job`                      | `manifest_recording`     | UserWarning (SCF exception re-raises regardless)                                            |
| Population dump (.population.txt / .json)                        | `runner` / `periodic_runner`          | `optional_artifact`      | UserWarning + `[[outputs.files]].error`                                                     |
| Density cube                                                     | `runner.run_job`                      | `optional_artifact`      | UserWarning + `[[outputs.files]].error`                                                     |
| MO cube (per-orbital + label resolution)                         | `runner.run_job`                      | `optional_artifact`      | UserWarning + `[[outputs.files]].error`                                                     |
| Citation block (`.bibtex` + `.references`)                       | `runner` / `periodic_runner`          | `optional_artifact`      | UserWarning + `[[outputs.files]].error`                                                     |
| `.xyz` / extended-XYZ                                            | `runner` / `periodic_runner`          | `optional_artifact`      | UserWarning + `[[outputs.files]].error` (molecular); UserWarning only (periodic, pre-manifest) |
| POSCAR                                                           | `periodic_runner`                     | `optional_artifact`      | UserWarning                                                                                 |
| CIF                                                              | `periodic_runner`                     | `optional_artifact`      | UserWarning                                                                                 |
| XSF structure                                                    | `periodic_runner`                     | `optional_artifact`      | UserWarning                                                                                 |
| Density grid evaluation + XSF density                            | `periodic_runner`                     | `optional_artifact`      | UserWarning                                                                                 |
| Molden                                                           | `periodic_runner`                     | `optional_artifact`      | UserWarning                                                                                 |
| Structured-log `properties` event probe                          | `runner.run_job`                      | `compatibility_fallback` | UserWarning                                                                                 |
| QVF sub-section prep (bibtex / population / density / MO data)   | `runner.run_job`                      | `compatibility_fallback` | UserWarning                                                                                 |
| QVF write                                                        | `runner` / `periodic_runner`          | `optional_artifact`      | UserWarning + `[[outputs.files]].error`                                                     |
| QVF manifest record                                              | `runner` / `periodic_runner`          | `manifest_recording`     | UserWarning + `[[outputs.files]].error`                                                     |
| Final manifest record sweep (`record(_pf.path)` / `mark_written`)| `runner` / `periodic_runner` wrap-up  | `manifest_recording`     | UserWarning + `[[outputs.files]].error`                                                     |
| Optimised geometry sibling writers (POSCAR / extended-XYZ)       | `periodic_runner`                     | `optional_artifact`      | UserWarning                                                                                 |
| Hessian / IR-intensity block                                     | `runner` / `periodic_runner`          | (soft, `.out` only)      | Already wrote a `WARNING:` line to the `.out`; not currently routed through `_errors` (out of scope of the silent-swallow audit). |

### Always-hard failures

These deliberately stay hard, they end the job:

* **SCF kernel exception.** `runner.run_job` writes ``SCF FAILED`` to
  the `.out`, emits a `scf_failed` structured-log event, writes a
  crash-dump if `crash_dump=True`, flips the manifest status to
  ``"crashed"`` via :meth:`ManifestUpdater.crash`, and re-raises.
* **Memory pre-flight failure.** ``check_memory`` raises before any
  compute when the requested job would exceed available RAM and
  ``memory_override=False``.
* **`OutputWriter.context()` exception.** Re-raises after marking
  the manifest crashed.
* **Pre-flight plan construction.** `OutputPlan.from_run_job_kwargs`
  raises on invalid combinations (a broken plan would mislead `vq`).
* **Guaranteed artefact finalization.** `ManifestUpdater.finish()` raises
  `IncompleteOutputError` and writes `status="crashed"` if any `always=true`
  plan row is unwritten.

### Implementation contract

* **The helper.** :func:`vibeqc.output._errors.warn_output_failure`
  builds a structured :class:`OutputFailureRecord`, emits a
  ``UserWarning`` with the category-tagged message, logs at
  ``WARNING`` (or ``ERROR`` for ``unrecoverable``), and, if given
  a :class:`ManifestUpdater`, calls
  :meth:`ManifestUpdater.mark_failed` to record ``written=false`` +
  the error string on the matching ``[[outputs.files]]`` row.
  :func:`vibeqc.output._errors.warn_writer_failure` is a thin
  convenience wrapper that accepts an :class:`OutputWriter`.
* **The manifest API.** :meth:`ManifestUpdater.mark_failed` is the
  public surface; callers must not poke ``manifest._outcomes``
  directly.
* **Categories are stable.** They appear in user-facing warnings;
  treat them as a public taxonomy.
* **The runner's job is to convert.** A writer should be free to
  raise whatever exception is natural for its failure mode, the
  runner catches at the right granularity and converts to a record.
  Writers themselves do *not* call ``warn_output_failure`` directly
  (they don't have the manifest handle anyway).
* **Tests.** ``tests/test_output_errors.py`` covers the unit-level
  contract for every category plus molecular population and periodic
  extended-XYZ failures that complete the chemistry but refuse complete
  artefact finalization. ``tests/test_output_manifest.py`` pins the guaranteed
  and conditional-row finalization rules.

## vq integration contract

Three hooks, all on the vibe-qc side. The vq-side work (Phase O4) is
small: a single new field on `JobSpec` and a dry-run pre-flight.

1. **Pre-flight dry-run** (Phase O3):
   `python input-h2o.py --vibeqc-dry-run`. `run_job` short-circuits
   after `OutputPlan` is built, it writes `{stem}.system` with the
   `[plan]` section populated and `[outputs].status = "dry_run"`,
   prints a one-line summary to stdout, and exits 0. No SCF runs.
2. **At job start (real run)**: `run_job` writes the `.system`
   manifest with `[plan]` + `[outputs].status = "running"` before
   any compute. vq's daemon sees the file appear in the workspace.
3. **As artefacts land**: each writer reports back to the
   `OutputWriter` coordinator (Phase O1 plumbing); coordinator
   atomically rewrites `.system` with updated `[[outputs.files]]`
   rows. On final return / exception, `[outputs].status` flips to
   `"complete"` or `"crashed"` and `[outputs].finished_at_iso` is
   stamped.

vq side (Phase O4, separate PR in the vibe-queue repository):

```python
# vibe-queue/src/vq/spec.py -- additions
class JobSpec:
    ...
    expected_outputs: list[str] = []   # populated from .system [[plan.files]]
    output_stem: str | None = None     # for the vq UI
    last_output_status: str | None = None  # "running" | "complete" | "crashed"
```

`vq submit` runs `--vibeqc-dry-run` when the input is a Python script
that imports `vibeqc.run_job`. The parsed plan populates
`expected_outputs`. `vq list` shows `outputs: 5/8` from the
`[outputs]` section. `vq fetch --outputs-only` tars only the planned
files, not the whole workspace.

## Citation database

### File location and format

Single TOML file at
`python/vibeqc/output/citations/database.toml`. Hand-maintained.
Sphinx renders `docs/citing.md` and `docs/user_guide/functionals.md`
§ Citations *from this file* via a new `vibeqc-cite-block` directive
(P3). Editing the rendered docs by hand is disallowed; CI checks the
generated content matches.

Schema sketch (real entries cover v0.8.0-on-main coverage in full, 
roughly 40-60 references):

```toml
# python/vibeqc/output/citations/database.toml

# ---------------------------------------------------------------- #
# REFERENCE ENTRIES
# ---------------------------------------------------------------- #
# Each entry has a stable key (used by routes and bibtex_key).

[entries.vibeqc_software]
kind        = "software"
bibtex_key  = "peintinger_vibeqc"
authors     = ["Peintinger, Michael F."]
title       = "vibe-qc: a quantum-chemistry code for molecules and solids"
year        = "{{VIBEQC_YEAR}}"            # rendered from banner at write time
version     = "{{VIBEQC_VERSION}}"
license     = "MPL-2.0"
url         = "https://vibe-qc.com/"
notes       = """Cite this for every vibe-qc calculation. A peer-reviewed
publication is forthcoming; this software citation is the canonical reference
until then. Pulled from CITATION.cff at write time."""

[entries.pob_tzvp_2013]
kind        = "article"
bibtex_key  = "peintinger_pob_tzvp_2013"
authors     = ["Peintinger, Michael F.", "Vilela Oliveira, Daniel", "Bredow, Thomas"]
title       = "Consistent Gaussian basis sets of triple-zeta valence with polarization quality for solid-state calculations"
journal     = "Journal of Computational Chemistry"
volume      = 34
issue       = 6
pages       = "451--459"
year        = 2013
doi         = "10.1002/jcc.23153"

[entries.vilela_oliveira_rev2_2019]
kind        = "article"
bibtex_key  = "vilela_oliveira_pob_rev2_2019"
authors     = ["Vilela Oliveira, Daniel", "Laun, Joachim", "Peintinger, Michael F.", "Bredow, Thomas"]
title       = "BSSE-correction scheme for consistent Gaussian basis sets of double- and triple-zeta valence with polarization quality for solid-state calculations"
journal     = "Journal of Computational Chemistry"
volume      = 40
issue       = 27
pages       = "2364--2376"
year        = 2019
doi         = "10.1002/jcc.26013"

[entries.libxc_2018]
kind        = "article"
bibtex_key  = "lehtola_libxc_2018"
authors     = ["Lehtola, Susi", "Steigemann, Conrad", "Oliveira, Micael J. T.", "Marques, Miguel A. L."]
title       = "Recent developments in libxc -- A comprehensive library of functionals for density functional theory"
journal     = "SoftwareX"
volume      = 7
pages       = "1--5"
year        = 2018
doi         = "10.1016/j.softx.2017.11.002"

[entries.pbe_1996]
kind        = "article"
bibtex_key  = "perdew_pbe_1996"
authors     = ["Perdew, John P.", "Burke, Kieron", "Ernzerhof, Matthias"]
title       = "Generalized Gradient Approximation Made Simple"
journal     = "Physical Review Letters"
volume      = 77
issue       = 18
pages       = "3865--3868"
year        = 1996
doi         = "10.1103/PhysRevLett.77.3865"

# ... and so on for every entry currently surfacing in
# docs/citing.md and docs/user_guide/functionals.md § Citations.

# ---------------------------------------------------------------- #
# ROUTING RULES
# ---------------------------------------------------------------- #
# Maps from "what the user requested" to "which entries fire".
# Always-blocks fire unconditionally for the category; per-key blocks
# fire when the matching string is seen.

[routes.software]
always = ["vibeqc_software"]

[routes.integrals]
always = ["libint_valeev"]

[routes.basis_sets]
"pob-tzvp"           = ["pob_tzvp_2013"]
"pob-dzvp-rev2"      = ["pob_tzvp_2013", "vilela_oliveira_rev2_2019"]
"pob-tzvp-rev2"      = ["pob_tzvp_2013", "vilela_oliveira_rev2_2019"]
"6-31g*"             = ["pople_6_31g_1972", "hariharan_pople_1973"]
"cc-pvdz"            = ["dunning_1989"]
# ... full coverage of every bundled .g94 in basis_library/

[routes.functionals]
# libxc itself is cited for DFT names not marked as external below.
_libxc_always        = ["libxc_2018"]
"pbe"                = ["pbe_1996"]
"pbe0"               = ["adamo_barone_1999"]
"b3lyp"              = ["becke_1993", "stephens_1994", "vosko_1980"]
"pw91"               = ["perdew_pw91_1992"]
"skala-1.1"          = ["luise_skala_2025", "poschel_skala_cp2k_2026", "sun_pyscf_2020"]
# ... full coverage of every functional in functionals.md.

[routes.external_functionals]
# Empty marker rows suppress only the automatic libxc paper. The ordinary
# functional and atomic-grid routes still fire.
"skala-1.1"          = []

[routes.methods]
"d3bj"               = ["grimme_2010", "grimme_2011"]
"d4"                 = ["caldeweyher_2019"]
"diis"               = ["pulay_1980", "pulay_1982"]
"ediis"              = ["kudin_2002"]

[routes.libraries]
# Per-linked-library citations that fire whenever vibe-qc links them.
# spglib only fires when periodic; libecpint only when ECPs are used.
"spglib"             = ["togo_shinohara_tanaka_2024"]
"libecpint"          = ["shaw_gilbert_libecpint"]
```

### Runtime assembly

`vibeqc.output.citations.assemble(plan: OutputPlan) -> list[Entry]`
walks the routes table and returns an ordered, deduplicated list:

1. `routes.software.always`, vibe-qc itself, first.
2. `routes.integrals.always`, libint.
3. `routes.basis_sets[plan.basis.lower()]`.
4. `routes.functionals._libxc_always` unless the normalized functional is in
   `routes.external_functionals` or was registered at runtime through
   `vibeqc.define_external_functional`; then
   `routes.functionals[plan.functional.lower()]`. A runtime-only provider name
   still reports a missing functional route until its defining citation is
   added to a database.
5. Method-specific (DIIS / EDIIS / dispersion / spglib for periodic / libecpint when ECPs used).

A miss in `routes.basis_sets` or `routes.functionals` raises in
strict mode (CI test fails) and warns in user mode (job still
proceeds; `.references` carries a `# WARNING: no reference for ...`
line so the gap is visible).

### Writers

- `{stem}.bibtex`, BibTeX entries, one per cited reference, in
  citation order. Plain ASCII, suitable for `\bibliography{}` use.
- `{stem}.references`, plain-text Chicago-ish numbered list (the
  same content that gets appended to `{stem}.out`).
- `.out` "## References" section, same content, embedded so the
  text log is self-contained.

### Dev-chat contract (AGENTS.md clause to add)

Drafted clause:

> **Citation database ownership**: Any merge that adds, removes, or
> renames a method, functional, basis set, ECP, dispersion model, or
> third-party linked library to vibe-qc MUST update
> `python/vibeqc/output/citations/database.toml` in the same merge:
>
> - add or remove the `[entries.<key>]` block,
> - update the matching `[routes.<category>]` row,
> - add a check in `tests/test_citations.py` that the feature triggers
>   the expected references.
>
> CI fails if `vibeqc.list_methods()` / `vibeqc.list_functionals()` /
> the bundled `.g94` inventory contains an entry not present in
> `[routes.*]`. `docs/citing.md` and `docs/user_guide/functionals.md`
> § Citations are auto-rendered from `database.toml` via the
> `vibeqc-cite-block` Sphinx directive, do not hand-edit those
> sections for new methods. See
> [`docs/design_output_module.md`](design_output_module.md) for the
> full schema.

### basissetdev integration

Per CLAUDE.md § 4, `basissetdev` does not merge into `main` for v0.8.x.
The 87 BSE-fetched basis sets accordingly live in a sibling DB on the
basissetdev branch:

- `main`: `python/vibeqc/output/citations/database.toml` only.
- `basissetdev`: `database.toml` + `database_basissetdev.toml`. The
  registry loads both files when present, with basissetdev entries
  layered on top.

When basissetdev eventually merges (post-v0.8.x, paper-aligned), the
two files merge into one, the schema is identical and the only
difference is which branch carries the entries. No format change
needed at merge time.

## Phased roadmap

Each phase is a coherent, testable, mergeable chunk. PRs land on
`main` (or on a feature branch and squash-merge, maintainer's call
when scope justifies it).

### Phase O1, Foundation (this chat, next PR)

- Create `python/vibeqc/output/` package skeleton.
- `OutputPlan` + `PlannedFile` dataclasses; `OutputPlan.from_run_job_kwargs()`.
- `OutputWriter` coordinator with `_adapters/` shims for every existing writer.
- `.system` schema extension: `[plan]` written at job start;
  `[outputs]` updated as files land; status flip at end.
- `runner.py` switches its writer dispatch to go through `OutputWriter`,
  but every artefact still has bit-identical content to the
  pre-Phase-O1 output.
- Tests: round-trip plan, manifest status flips through running →
  complete → crashed, every existing test in `tests/` still passes
  unchanged.
- **No new artefacts in this phase.**

### Phase O2, Citation database + writers

- `python/vibeqc/output/citations/database.toml` populated with
  v0.8.0-on-main coverage (software, libint, libxc, pob-*,
  B3LYP/PBE/PBE0/PW91, D3BJ, spglib, libecpint, every bundled `.g94`).
- `assemble()` + `to_bibtex()` + `to_plain()` implemented.
- `{stem}.bibtex` and `{stem}.references` emitted by default.
- "## References" block appended to `{stem}.out`.
- `tests/test_citations.py` smoke-tests every functional / basis /
  method path exercised by the existing test suite.
- AGENTS.md clause added.
- CHANGELOG entry in `[Unreleased]`.

### Phase O3, `.xyz` + dry-run + CLI

- `{stem}.xyz` (always-on for molecular jobs; periodic emits a
  natural-cell `.xyz` plus the P5 crystal formats).
- `--vibeqc-dry-run` CLI flag honoured by `run_job`.
- `vibeqc-cite <stem>` console-script entry point that re-reads the
  `.system` manifest + database and reprints citations for already-
  run jobs.

### Phase O4, vq integration (separate PR in the vibe-queue repository)

- `JobSpec.expected_outputs` + `JobSpec.output_stem` +
  `JobSpec.last_output_status`.
- `vq submit` pre-flight: when input is a `.py` containing `run_job`,
  call `--vibeqc-dry-run` and parse the `[plan]` section.
- `vq list` surfaces outputs progress (`5/8 outputs written`).
- `vq fetch --outputs-only` tars only `[[plan.files]]` paths.
- `vq watch` (new verb, optional in this PR): tails the live
  `{stem}.out` and prints status transitions from the `.system`
  manifest.

### Phase O5, Periodic + crystal formats

- Wire `OutputPlan` into [`periodic_runner.py`](../python/vibeqc/periodic_runner.py).
- `{stem}.cif` / `POSCAR` / `{stem}.xsf` always for periodic jobs;
  consolidate existing scattered writers under `output/formats/crystal.py`.
- Periodic-job plan tests.

### Phase O6, Volumetric + population

- `.cube` for orbitals/density on demand (`run_job(..., write_cube=...)`).
- `{stem}.population.{txt,json}` clean separation from `.out` for
  Mulliken / Löwdin / Mayer / dipole. The corresponding block stays
  in `.out` for human reading; the structured file is for downstream
  parsing.

### Phase O7, Documentation sweep

- `docs/user_guide/output_files.md` updated phase-by-phase (CLAUDE.md
  § 5 lightweight cadence, done same-session as each phase).
- `docs/citing.md` and `docs/user_guide/functionals.md` § Citations
  switched to render from `database.toml` via the Sphinx directive.

### Out of scope for the thin-layer phase

- `.fchk` (Gaussian-compat formatted checkpoint), P3 of the eventual
  v1.0 sprint.
- HDF5 full-state checkpoint (`{stem}.h5`), same.
- `.bands.dat` / band-structure plot script, relevant once multi-k
  bands are wired through `run_job`.
- The relocation of `io/molden.py`, `system_info.py`,
  `structured_log.py`, `perf.py`, `crash_dump.py` into
  `output/formats/`, the v1.0 rewrite.

## Pre-v1.0 full refactor (recorded for continuity)

The pre-v1.0 sprint will:

- Inline the `_adapters/` shims into their `formats/*.py` cousins.
- Move `scf_log.py`, `system_info.py`, `structured_log.py`, `perf.py`,
  `crash_dump.py`, `io/molden.py`, `io/trajectory.py` under
  `vibeqc/output/`.
- Re-point every import across the test suite. High blast radius;
  must land in a quiet sprint.
- Public API (`OutputPlan`, `vibeqc.output.citations`, file-format
  identities, the `.system` schema) survives unchanged. Tests that
  hit the public surface keep working; tests that import the old
  module paths are flagged for rewrite as part of that sprint.

## Compatibility surface

The thin-layer phase is *additive* on the user-facing surface:

- New `[plan]` + `[outputs]` sections in `.system`, old parsers that
  only read `[vibeqc] [host] [cpu] [memory] [python] [libraries]
  [validation] [run]` keep working (TOML readers ignore unknown
  sections).
- Two new default-on artefacts (`{stem}.bibtex`, `{stem}.references`,
  `{stem}.xyz`), opt-out via `citations=False` / `write_xyz=False`.
- No existing artefact changes shape or path.
- No existing kwarg's default flips. The new kwargs default to
  values that preserve old behaviour wherever ambiguity exists.
- `VIBEQC_NO_CITATIONS=1` env-var kill switch for batch contexts.

CHANGELOG `[Unreleased]` carries one entry per phase as it lands.

## Implementation status

All of the O1-O7 phased roadmap plus the pre-v1.0 follow-ups have
landed on `main` (2026-05-18). Per-phase commit map:

| Phase | What landed | Status |
|---|---|---|
| O1 | `OutputPlan` / `PlannedFile` / `ManifestUpdater` / `OutputWriter`; `.system` `[plan]`+`[outputs]` schema; AGENTS.md rule 8 | shipped |
| O2 | citation database (`database.toml`) + `registry` + `.bibtex` / `.references` writers | shipped |
| O3 | `{stem}.xyz` writer; `dry_run` / `VIBEQC_DRY_RUN`; `vibeqc-cite` console script | shipped |
| O4 | vq integration, `JobSpec.expected_outputs`, `vq submit --vibeqc-preflight` | shipped (vq v0.6.14) |
| O5 | periodic `run_periodic_job` wiring; extended-XYZ + POSCAR + XSF; periodic citations | shipped |
| O6 | `population.{txt,json}`; opt-in volumetric `.cube` | shipped |
| O7 | `output_files.md` refresh; Sphinx `vibeqc-cite` directive | shipped |
| R1-R7 | writer modules relocated under `output/formats/` with backward-compat shims | shipped |
| D1 | role-driven `Dispatcher` + `default_dispatcher()` + `OutputWriter.dispatch_role` | shipped |
| D2 | CIF writer + dispatcher registration | shipped |
| D3 | `vibeqc-outputs` manifest-inspection CLI | shipped |
| D4 | citation `method=` routing fix; FCI + composite-3c routes | shipped |
| D5 | citation routes for the v0.9.0 meta-GGA + RSH functionals | shipped |
| D7a | `runner.py` dispatch-overhaul, `OutputWriter` owns the `.system` manifest lifecycle | shipped |
| D7b.1 | molecular single-shot artefacts routed through `dispatch_role` | shipped |
| D7b.2 | periodic `OutputWriter` lifecycle + single-shot artefacts routed through `dispatch_role` | shipped |
| H1 | `HeaderlessBlock` primitive + orbital/energy/system/timing migrations | shipped |

**Deviation from the locked decisions**: decision 2 ("Phase O1 thin
adapter layer; no writer module is relocated") held *for Phase O1*,
but the writer relocation (R1-R7) was then carried out as the
pre-v1.0 follow-up the *Non-goal* paragraph anticipated. The
`output/_adapters/` directory sketched in [§ Module shape](#module-shape)
was not needed, the relocation used in-place backward-compat shims
at the old module paths instead, which preserves import identity
more cleanly than an adapter indirection. The actual layout under
`output/formats/` is the relocated writers themselves.

**The runner.py dispatch-overhaul, what shipped (D7a) and what
didn't (D7b)**: D7a is the load-bearing change. `run_job` no longer
writes `{output}.system` with a bare end-of-job
`write_system_manifest` call, an `OutputWriter`, constructed
before the SCF, owns the manifest for the whole run: `[plan]` +
`[outputs].status="running"` written up front, `crash()` /
`finish()` at the exit paths, an end-of-job record-sweep filling
`[[outputs.files]]`. That removes the genuine "ad-hoc dispatch"
(an uncoordinated manifest write) and gives `vq` a live status
signal. The periodic D7b milestone subsequently replaced its bare
`ManifestUpdater` with the same job-scoped lifecycle.

D7b, converting each individual `write_molden(...)` /
`write_xyz(...)` / `write_population(...)` / cube / citations call
site to `OutputWriter.dispatch_role(...)`, was assessed and
**deliberately not pursued in the thin-layer phase**. Those call
sites are not "ad-hoc dispatch"; they are explicit, well-
instrumented writer calls (each wrapped in its own `plog.stage` +
`PerfScope` and emitting a per-writer `.out` line). Routing them
through `dispatch_role` as it stands today would: (a) lose that
per-writer perf / progress instrumentation unless the dispatcher
API is widened to thread a perf/progress context; (b) re-run the
population property computation twice, because `write_population`
is one writer that emits two files and the per-`PlannedFile`
dispatch model invokes it once per row; (c) need a new
`("orbital_vol", "cube")` adapter for per-MO cubes. All churn on
the most-contended file in the tree for no gain over D7a's
record-sweep, which already produces a correct `[outputs]`
section. The full `dispatch_role` conversion belongs to the
v1.0 coordinator rewrite, where the writers themselves are
restructured into adapters and the one-writer-many-files case
(population) gets a dispatch model that fits.

**D7b resumed (2026-07-15).** The molecular milestone widened the additive
dispatcher API without changing its fail-soft default: runners can opt into
strict error propagation inside their established warning handlers, target a
specific format/path, and dispatch a runtime-only path. A private cache shared
by all rows in one role dispatch lets the population adapter compute and write
its `.txt` and `.json` pair once while recording both plan rows. `runner.py`
now routes molden, population, density/per-MO cubes, citations, XYZ, NTO
molden, and QVF through those adapters with the existing progress and timing
scopes intact. Deterministic files were byte-identical in the representative
RHF before/after audit; QVF scientific members were byte-identical and only
the provenance wall time moved.

The periodic milestone constructs its plan and `OutputWriter` before compute,
updates late-bound execution provenance on that same coordinator, and lets a
public-boundary lifecycle wrapper finish or crash it after runtime geometry
optimization. All single-shot periodic writers, including the basis-free
semiempirical branch, method-specific BIPOLE/AICCM2026DEV-B population
summaries, and runtime `.opt.*` geometry, dispatch by `(role, format)`; the
post-hoc filesystem sweep is gone. An immediate direct-writer control matched
every deterministic periodic artefact it produced byte-for-byte. `.out`
timing/banner fields, QVF wall-time provenance, and the manifest lifecycle are
intentionally dynamic.

## Addendum (2026-07-10): the `.out` gets a channel

The "Context" section above records that `runner.py` assembles the `.out`
inline. The reason was structural: the file handle `f` lived in one
`with` block, so only code inside that block could contribute a line.
Anything else had to be threaded the handle, which is why the IUPAC
naming call sat in the SCF runner despite having nothing to do with SCF.

`vibeqc.output.channel` removes the handle. A module imports `write` and
`flush` and emits; the runner installs an `OutputChannel` once and
decides where the text lands (file, stdout, tee, or null). Outside an
active channel `write()` is a no-op, so a writer stays callable from a
test or a notebook, and importing the module opens nothing.

This does **not** make the `.out` a dispatcher role. The `.out` is a
stream assembled from hundreds of partial writes over the lifetime of a
job, not a file produced by a single writer call, so the
`("log", "text")` exclusion in `dispatch.py` still stands and now points
at the channel instead of promising a future adapter.

Three concerns, three modules, deliberately separate:

| Module | Decides |
|---|---|
| `output/channel.py` | *where* text goes |
| `output/document.py` | *how* a number renders (precision, units, tables) |
| `output/writer.py` | *which files* a job produces, plus the manifest |

The migration was plumbing only: normalized `.out` captures from
`run_job` (RHF, RKS) and `run_periodic_job` were byte-identical before
and after.

## Addendum (2026-07-12): the document layer and channel mature

The 2026-07-10 addendum installed the three-way split (channel = *where*,
document = *how*, writer = *which files*). The document and channel sides
then filled in. Each addition below defaulted to rendering byte-identically
to the pre-existing format, so adopting it never moved the `.out` on its own
-- the change is opt-in via an env var or a moved knob -- and the whole `.out`
scaffold is now frozen by golden snapshots so it cannot drift silently.
The 2026-07-16 `HeaderlessBlock` addition deliberately resized the three
original guessed-rule families; its reviewed golden changes were limited to
those rules. Two 2026-07-18 follow-ups moved the standalone double-hybrid
energy components and both runners' timing summaries onto the same primitive.
Their reviewed golden changes shorten only the affected guessed rule lines.

**Document layer -- one place decides how a number renders.**

* **`FormatPolicy`, end-to-end.** A `Quantity(value, kind)` carries a
  physical *kind*; a process-wide `active_policy()` (a `FormatPolicy`)
  decides its precision and display unit. `VIBEQC_OUTPUT_UNITS=eV`
  (or `energy:eV,time:ms,wavenumber:THz`) reconfigures units in one place
  instead of per-f-string. Kinds today: `energy`, `energy_delta`,
  `gradient`, `length`, `temperature`, `duration` (seconds; `time:ms`/`min`),
  `frequency` (cm⁻¹; `wavenumber:THz`/`meV`), `dimensionless`.
* **Byte-neutral per-site converters.** `render_energy`,
  `render_energy_labeled`, `render_duration`, `render_frequency`,
  `render_temperature` each keep a call site's own field width while routing
  the value (and, for `_labeled`, the unit token) through the policy -- so the
  ~80 scattered `f"{e:16.10f} Ha"` energy strings, the timings block, and the
  vibrational/thermochemistry numbers convert without a normalisation pass.
  Every SCF-energy block, the timings block, and the frequency/temperature
  fields are now policy-driven; the intentional dual-`Ha`/`eV` displays use
  explicit-unit `render_energy` calls (orbital table, VBM/CBM/gap).
* **`Table` with self-sized rules + `footer()`.** Every *headered* data table
  -- SCF trace, frequency, TD-DFT excited states, atomic charges / bond
  orders / dipole -- renders through `Table`, whose columns
  and rule size to content (retiring the `"-" * 52` vs `"-" * 56` guesses).
  `Table.footer(*cells)` adds a total row set apart by a rule (charge sums,
  energy totals).
* **`HeaderlessBlock` with annotations and footers.** Titled label/value and
  matrix displays size their rules from content without inventing semantic
  column headers. Optional dividers and footers cover component totals and
  gap summaries; a right-hand annotation is excluded from rule sizing so
  ragged HOMO/LUMO/SOMO markers remain annotations rather than a boxed column.
  Molecular SCF and standalone double-hybrid energy components, molecular and
  periodic timing summaries, orbital energies, the molecular atom summary,
  and periodic Periodicity/Lattice/Atoms, basis, primitive-cell reduction, and
  smearing, band-extrema, and crystal-orbital summaries now share this
  primitive, as do the Gaussian-basis and full-k semiempirical periodic
  optimization verdicts. Timing iteration counts, k-point locations, and
  frontier markers use the annotation field, so they do not inflate the
  content rule.
  Mixed-kind smearing rows use semantic energy, temperature, and dimensionless
  quantities while retaining their row-specific unit labels; the band block
  preserves its deliberate explicit Ha/eV pair and the crystal-orbital block
  follows the active energy policy.
* **`CriteriaTable` for threshold diagnostics.** Each row retains a semantic
  measured `Quantity`, an optional same-kind threshold, and a Boolean verdict.
  The full form renders criterion/value/threshold/unit/status columns; the
  compact form renders the same record inside a streaming progress-table cell.
  Geometry optimization uses the compact form, so the measurement and
  threshold change units together and `pass`/`fail` wording is single-sourced.

**Channel -- verbosity, warnings, and the human+machine funnel.**

* **Verbosity levels.** Every `write(text, level=...)` carries a `Level`
  (`QUIET` / `STANDARD` / `VERBOSE` / `DEBUG`); the channel filters by
  `VIBEQC_OUTPUT_LEVEL`. `STANDARD` default keeps the `.out` unchanged. The
  SCF convergence verdict is tagged `QUIET` so it survives `--quiet`.
* **`warn()` / `note()`.** First-class warning surface: `warn(msg, **fields)`
  emits `  WARNING: {msg}` at `QUIET` and fires a structured `warning` event;
  `note()` is the `STANDARD` non-problem counterpart. The runners' hand-written
  `WARNING:` lines route through it.
* **`record(event, human, **fields)`.** Emits the human `.out` line *and* the
  machine `StructuredLog` event in one call, so the two surfaces cannot
  drift; adopted at the SCF error/retry sites. Its optional `once_key`, also
  exposed by `warn()` / `note()`, stores first-occurrence state on the active
  channel and structured log. Repeated guards therefore suppress the human
  line and machine event together, without feature modules owning logger
  lifecycle state. Keys are scoped to those active job surfaces, and note /
  warning severities use distinct keys so escalation remains visible.

**Format is frozen, and escaping it is guarded.** `tests/test_out_format_snapshot.py`
freezes the digit-canonicalised `.out` scaffold for RHF / RKS / UHF /
double-hybrid / Hessian / TD-DFT (molecular) and `periodic_rhf` /
`periodic_geomopt`; `tests/test_out_writers_no_print.py` pins that the `.out`
writers never regress to `print()`.

## See also

- [`docs/user_guide/output_files.md`](user_guide/output_files.md), 
  user-facing reference for the output file family.
- [`docs/citing.md`](citing.md), citation guidance; the per-feature
  reference blocks render from `database.toml` via the Phase-O7b
  `vibeqc-cite` Sphinx directive.
- [`docs/announcement_output_module_2026_05_18.md`](announcement_output_module_2026_05_18.md)
  - the dev-chat memo (citation-DB ownership, the new CLIs).
- [`docs/release_process.md`](release_process.md), release cadence,
  documentation cadence (CLAUDE.md § 5).
- [`CLAUDE.md` § 4](CLAUDE.md), basissetdev branch policy.
- [`CLAUDE.md` § 10](CLAUDE.md), external-codes / library-only
  policy; citation routing must respect the library/program line.
- [`AGENTS.md`](AGENTS.md), dev-chat manifest where the
  citation-DB ownership clause lands.
