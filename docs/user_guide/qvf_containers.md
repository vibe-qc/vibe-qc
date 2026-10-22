---
myst:
  html_meta:
    "description": "QVF job containers: one .qvf file describes a calculation, runs it locally or through vq, and comes back as a complete self-contained record - structure, results, log, references, and system manifest."
    "og:title": "QVF job containers: one file, whole lifecycle"
    "og:description": "Build a pending .qvf from a structure + job spec, run it with vibeqc run or ship it through vq, and open the settled archive in vibe-view - no sidecar files needed."
---

# QVF job containers: one file, whole lifecycle

A QVF archive can *describe* a calculation before it has run. Such a
**job container** carries the structure, a declarative `job.spec`
section (method, basis, functional, charge, multiplicity, k-mesh,
tasks, engine options), and `provenance.run_status = "pending"`.
Running it updates the **same file in place**: results sections are
added, a complete `run.record` (the executed spec, the full log, and
the `.system` / `.perf` / structured-event sidecars) is embedded, and
the status settles to `converged` or `failed`. The one file is then a
complete, portable record of the calculation - no sidecar needed to
understand it.

The `job.spec` payload is declarative data, never code: running a
container never executes an embedded script (QVF spec § 5.9).

A single job may name its input geometry with `structure_ref` on the section
or in its payload; the section reference takes precedence. The input geometry,
drawing members, metadata and extension declarations remain in the archive.
Results receive their own structure and section references, and the result
geometry is the default view. The original `job.spec` section and payload stay
unchanged, including when execution is forced again.

A legacy job without either reference must have exactly one structure. It keeps
that structure's id for the latest result and stores the submitted geometry as
`attachment.input_structure_*` members of the run record. It is not silently
converted into an ambiguous job with several structures. Prior canonical result
sections are replaced in this legacy mode; their old bytes and run history remain
in the archive, without associating old properties with the new geometry.
Colliding result paths and understood references are relocated; conflicting
extension declarations or
opaque vendor references that cannot be relocated safely cause an atomic
rollback to the submitted archive.

`basis` and `functional` follow QVF spec § 5.9. A name is a nonempty string.
`null` states that the method takes no such input: semiempirical,
tight-binding (xTB, DFTB) and MACE jobs record `"basis": null` and
`"functional": null`, and Hartree-Fock and correlated wavefunction jobs record
`"functional": null`. An absent key leaves the choice to the engine. The
runners and `write_pending_qvf` write these spellings and never the empty
string. Reading a container, `vibeqc run` treats `null`, an absent key and the
deprecated `""` that earlier versions wrote alike: it supplies no basis set or
functional of its own. A container stamped with the withdrawn `qvf_version: 2`
is read as version 1 with a `W_VERSION_WITHDRAWN` warning and is rewritten as
version 1 when it is updated in place; any other version is refused. Explicit
job specs passed to the low-level writer are preserved verbatim, except that
an empty `basis` or `functional` is written as `null`.

## Build a pending container

```python
import vibeqc as vq

mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])], 0, 1)
vq.write_pending_qvf(
    mol, "job",
    method="rhf", basis="sto-3g",
    tasks=["single_point"],            # or ["optimize", "hessian"]
    options={"perf_log": True},        # supported JSON engine options
)
# -> job.qvf, run_status "pending"
```

Periodic systems work the same way - pass a `PeriodicSystem` (its
`charge`/`multiplicity` are recorded) plus `kpoints=[n1, n2, n3]`.
`options` uses the target runner's keyword names. Values must have a supported
JSON representation; keyword recognition alone does not make arbitrary Python
runtime objects portable. Unknown keys and unsupported representations are
refused at run time rather than silently dropped.

### Structured engine options

Supply option-class arguments as JSON objects, for example:

```python
options={"rhf_options": {
    "use_diis": True, "damping": 0.0,
    "max_iter": 120, "conv_tol_energy": 1e-8,
}}
```

The engine reconstructs its options object before calling the runner. Omitted
fields retain that class's defaults; explicit zero, false and null values are
not replaced by defaults. Supported records are `rhf_options`, `uhf_options`,
`rks_options`, `uks_options`, `rohf_options`, `roks_options`, `mp2_options`,
`ump2_options`, `ccsd_options`, `cisd_options`, `cc3_options`, `ccsdt_options`,
`selected_ci_options`, `dmrg_options`, `v2rdm_options`,
`transcorrelated_options`, `casci_options`, `caspt2_options`, `nevpt2_options`,
`casscf_options`, `dlpno_options`, `dlpno_ccsd_options`, `mlip_options`,
`ccm_options`, `thermo_options`, periodic `smearing`, and
`opentrustregion_options` when that constructor is available in the installed
runtime.

`dlpno_ccsd_options` is the `LocalCCSDOptions` of the local DLPNO-CCSD and
DLPNO-CCSD(T) solver that `run_job` uses (the O(N^6) correctness pilot stays a
Python-API option). A container states its local triples algorithm with
`"triples_mode"`: `"local"` (DLPNO-(T0)), `"t1"` (the default) or `"exact"`.
Its enumerated text fields accept only the values the molecular route
implements, so a misspelling is refused at its path rather than read as some
other choice: `triples_mode`, `localise` (`"boys"` or `"none"`),
`residual_domain`, `domain_criterion` and `pno_norm` (canonical names). Because
`run_job` refuses the top-level `dlpno_thresholds` beside an options object,
`dlpno_options` and `dlpno_ccsd_options` records may name their preset in a
reserved `"dlpno_thresholds"` key, the JSON form of
`options_from_dlpno_thresholds`: the preset is applied first and the record's
own fields after it, so an explicit threshold labels the result `custom`.

```python
options={"dlpno_ccsd_options": {"dlpno_thresholds": "tight",
                                "triples_mode": "t1"},
         "frozen_core": "chemical"}
```

The run record's `system` attachment then carries `dlpno_triples_mode` and
`dlpno_triples_algorithm`, the algorithm that ran.

Native enum fields use their public member names as strings (case insensitive),
for example `"initial_guess": "SAD"` or `"scf_accelerator": "DIIS"`.
Nested option objects such as `"grid": {"n_radial": 75}` and
`"davidson": {"max_iter": 50}` are reconstructed recursively. Typed numeric
arrays, including level-shift schedules, are validated element by element.
Python option records retain their constructor validation, including smearing
aliases and CCSD frozen-core explicitness. Boolean values are not numbers here;
`"max_iter": true`, unknown fields and non-finite numbers are errors that name
the full option path.

This is an engine-specific encoding inside QVF's existing open `options`
object, not a new QVF schema version. It does not deserialize arbitrary classes.
Native matrices, nonempty native object arrays (such as explicit ECP objects),
callbacks, CASSCF `pt2` objects, and structured `band_structure`,
`dft_plus_u`, `read_from`, `dispersion`, `citype`, `triples` or
`dlpno_thresholds` objects currently receive an explicit refusal.
Use the Python runner API for those runtime objects. Scalar shorthand forms
remain available where the runner supports them. Existing mapping APIs such as
`geom_opt_options` and `solvent`, and fragment atom-index lists, keep their
normal consumer semantics (`fragments` must contain lists of integer atom
indices; `dft_plus_u=[]` remains a no-site request). Local Python `run_container(..., **overrides)`
values remain ordinary runner objects, outside the serialized spec.

An option-conversion refusal occurs before SCF and settles an attempted
in-place execution as `failed`, preserving the original `job.spec` bytes.

## Run it locally

```bash
vibeqc run job.qvf
```

(`python -m vibeqc run job.qvf` is equivalent.) The runner
reconstructs the job from the spec, streams the ordinary `.out` and
sidecar files to the output stem (default: the container's own stem),
and finishes by atomically rewriting `job.qvf` itself: during
execution the archive reads `running`; afterwards it is `converged` or
`failed` with a complete, sequenced `run.record`. A run that fails
before producing output settles with an *empty* embedded log - that is
a complete record of that invocation, not a defect.

A settled container is a record; re-running it must be explicit:

```bash
vibeqc run job.qvf --force
```

appends another sequenced `run.record`, preserving the earlier runs as
history. `--output STEM` redirects the sidecar files without moving
the container.

## Ship it through vq

The container is the whole payload and the updated container is the
whole result. With a managed runtime (fleet hosts):

```sh
vq submit HOST job.qvf --program vibeqc-dev
vq fetch HOST JOBID --name job.qvf -o results/
```

The daemon resolves the managed runtime, snapshots its full Git SHA,
runs `python -m vibeqc._cli run job.qvf`, and rejects any runtime drift
before chemistry starts. `vq fetch --name` retrieves just the settled
container. `vq status --json` reports the container's `qvf_lifecycle`
(status, sequence, terminal completeness). Routine submissions do not
need a version or SHA argument. Add `--expected-sha FULL_40_HEX_SHA`
only when reproducing a historical run and you want an extra assertion.

On a host without a managed program entry, the directory form works
with any interpreter:

```sh
vq submit HOST -d jobdir/ -- vibeqc run job.qvf
```

See [Queue](queue.md) and [`docs/agent_interaction.md` in vibe-queue](https://github.com/vibe-qc/vibe-queue/blob/main/docs/agent_interaction.md) for
queue mechanics.

## Submit it from vibe-view

Open any structure, choose **Submit to vq cluster**, and leave
**Submit as QVF container** enabled. The Job Manager writes a pending
container from the structure on screen plus the method and basis fields,
submits that one file, and opens the settled artifact with its Job Spec and
Run Record panels when it returns.

Writing a pending archive is a producer action. If `vibeqc` is not installed
next to vibe-view, the switch is disabled and the viewer keeps using its
generated-script submission path.

## Open it in vibe-view

[vibe-view](vibe_view.md) renders the whole lifecycle: a *pending*
container opens with an amber "pending" chip and a Job Spec panel
stating the job has not yet run; with auto-reload enabled the chip
follows the file through `running` to its terminal state. A settled
container shows the results plus a **Run Record** panel per run - the
executed spec, the complete log, the rendered system manifest,
performance log, and structured event table - labeled
"Run #k of N - latest" when the container has been re-run. A terminal
container whose latest record is missing its input or log is flagged
with an explicit warning rather than silently reading as done.

The same archive can also be inspected without a display server:

```sh
vibe-view show job.qvf --info   # lifecycle and archive summary
vibe-view show job.qvf --plain  # one terminal-rendered structure frame
vibe-view tui job.qvf           # interactive terminal viewer
```

Use `vibe-view open job.qvf` for the graphical viewer. `show` is useful
in logs and CI; `tui` is useful over SSH without X forwarding.

## What lives where

| Content | In the container | Sidecar file |
|---|---|---|
| Structure + `job.spec` (the request) | always, preserved verbatim | none |
| Results (energies, wavefunction, …) | canonical sections | assorted (`.molden`, `.xyz`, …) |
| Full log | `run.record` `log` member | `.out` |
| References | `citations` section | `.references` / `.bibtex` |
| System manifest | `run.record` `attachment.system` | `.system` |
| Performance / structured logs | `attachment.perf` / `attachment.structured` (when requested) | `.perf` / `.scf.jsonl` |

Sidecar files still land at the output stem - the container makes them
redundant, it does not suppress them. Individual writers are switched
off with the usual runner keywords (`write_xyz_file=False`, …), which a
container can carry in its `options`.

## Learn by running one

The tutorial [Running a calculation as a QVF container](../tutorial/qvf_job_containers.md)
walks through a pending molecular container, local execution, inspection,
terminal and graphical viewing, a deliberate re-run, a periodic container,
and the first-class vq submit/fetch path.

Runnable companions live under the checkout path
`examples/qvf_containers/`:

- `build_h2_job.py` creates a molecular H2/STO-3G container.
- `build_periodic_he_job.py` creates a tiny periodic He/SCC-DFTB container.
- `inspect_job.py` validates either file and prints its spec, status,
  section kinds, and run history.

Format details: [QVF design](../design_qvf_format.md) and the QVF
specification ([published QVF spec](../qvf/spec.md), § 3.2 and § 5.9).


### Explicit molecular symmetry source

Experimental molecular RHF/UHF single points can request the source-qualified
fixed-orbit path through the `symmetry_reduction` option. It carries explicit
operations, screening tolerance, omission cap, budget and execution policy.
The [shared symmetry reference](../developer_shared_symmetry.md#job-and-qvf-source-requests)
describes the strict record and its constructor helper. Every execution
qualifies a fresh integral producer; a saved audit or source hash cannot
replace that check. Keep convergence/guess controls in the existing method
option record. Correlation, derivatives, fitting and periodic calculations
require their own source admission.


With hierarchical screening, `executor="ordered"` uses native omission bounds
to qualify the fixed mask without integral audit reads. It evaluates all
retained ordered tiles during SCF. `executor="auto"` with `expected_builds`
1 or 2 also takes this path and skips unprofitable calibration; larger
estimates keep measured routing and the retained-source audit. This is an
experimental molecular execution choice, not a new symmetry assumption
about the electronic state or a guaranteed speedup. Version 2 source hashes
separate the actual integral mask from qualification and cost evidence.
