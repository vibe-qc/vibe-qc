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

## Build a pending container

```python
import vibeqc as vq

mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])], 0, 1)
vq.write_pending_qvf(
    mol, "job",
    method="rhf", basis="sto-3g",
    tasks=["single_point"],            # or ["optimize", "hessian"]
    options={"perf_log": True},        # any run_job/run_periodic_job kwarg
)
# -> job.qvf, run_status "pending"
```

Periodic systems work the same way - pass a `PeriodicSystem` (its
`charge`/`multiplicity` are recorded) plus `kpoints=[n1, n2, n3]`.
`options` takes any keyword the target runner understands; unknown
keys are refused at run time rather than silently dropped.

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
