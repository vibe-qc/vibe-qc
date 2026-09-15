# vq Job Manager

Source commands on this page run from the **separate vibe-view checkout**.
Clone [mpei/vibe-view](https://github.com/vibe-qc/vibe-view) and follow
[viewer setup](../tutorial/vibe_view_getting_started.md). Its `.venv`, `scripts/`, and `electron/` belong to that repository;
vibe-qc does not contain or install the viewer.

vibe-view is a cockpit for the [vibe-queue](queue.md) job runner, not
just a file viewer. The **Job Manager** panel lists the local vq queue, tracks
job state, and fetches finished results straight into the 3D viewer.

## Opening it

Click the **server** icon (`mdi-server-network`) in the app bar, or run any
viewer session and open it from there. The panel slides in from the right and
stays available while you work.

## What it shows

Each row is one vq job:

- a **state chip**, colored by state: `pending` (grey), `running` (blue),
  `suspended` (amber), `completed` (green), `failed` / `killed` / `interrupted`
  (red). Unknown states fall back to grey rather than erroring.
- the **job name** (or short id if unnamed) and the short job id;
- **elapsed time**, since start for a running job, start to finish for a
  terminal one;
- an **open** action that fetches the job and loads its `.qvf` files.

## Queue overview

The strip at the top of the panel summarizes the local queue at a glance: a
green/red dot for daemon health, the host and `vq` version, the host capacity
(max CPUs / max concurrent jobs), and the current load (running / pending CPUs).
It refreshes with the job list.

## Live monitor

By default the panel shows completed jobs (the ones you can open). Flip **Live
monitor** on to include every state (queued, running, suspended) and to
**auto-refresh every 5 seconds** so you can watch a job progress from
submission to results without touching anything. Use the refresh icon to poll
on demand.

## Per-job status + log tail

The magnifier action on a job row expands a detail card (pinned above the list)
showing the job's live state and the tail of its stdout/stderr. With Live
monitor on, that detail advances in step with the queue, handy for watching an
SCF converge. Click the magnifier again to collapse it.

## Submitting a job

Build or parameterize a structure, then submit it straight to the queue: the
viewer generates the vibe-qc input, submits it via `vq` with an auto-derived
job name (`vibeview-<file>-<method>`) and a `vibe-view` tag, and opens the Job
Manager (live monitor on) so the new job appears immediately. If `vq` isn't
installed but vibe-qc is, the job runs locally in the launch directory instead.

### QVF containers (one file out, the same file back)

Leave **Submit as QVF container** on and the payload is a single *pending*
`.qvf` instead of a generated script: the structure on screen plus a
`job.spec` (method, basis, functional, charge, multiplicity), stamped
`run_status = "pending"`. `vq` recognizes the suffix, resolves the pinned
runtime, and runs `vibeqc run job.qvf`, which settles that same archive in
place: results, the full log, citations and the `.system` manifest all
land inside it and the status becomes `converged` or `failed`. The job is
tagged `qvf-container`, and the Job Manager's open action loads the settled
archive with its **Job Spec** and **Run Record** panels intact.

Why you would want it: the request travels with the numbers (so a result is
reproducible from the file alone), one file moves each way instead of a
directory, and nothing has to be reassembled from sidecars afterwards.

The switch is **disabled when `vibeqc` is not installed** next to
vibe-view. Writing a pending archive is a *producer* action and vibe-view is
a consumer (it does not depend on the producer), so in that case the
viewer falls back to submitting its generated script and says so in the
status line. Streaming live checkpoints stays a script-mode option; a
container settles in place instead of emitting a separate checkpoint file.

Full walkthrough, including the Python and command-line routes:
[Running a calculation as a QVF container](../tutorial/qvf_job_containers.md).

## Fetching results

The **Fetch results into** field sets the directory finished jobs are pulled
to (default `vq-fetched/`, relative to where you launched the viewer). Clicking
a completed job's open action runs `vq fetch` under the hood and opens every
`.qvf` it produced, with no manual copy step.

## Requirements

The panel talks to the local queue through the `vq` Python package. If
vibe-queue isn't installed in the environment vibe-view runs from, the panel
shows a friendly notice instead of failing. Job submission and results all go
through `vq` per the shared-host protocol: vibe-view never writes to the
managed git checkouts on compute-reference/compute-small.

## Roadmap

The Job Manager is milestone **A1** of the Avogadro-parity roadmap
(`docs/ROADMAP_AVOGADRO_PARITY.md` in the separate vibe-view checkout). Coming
next: submit a built structure straight from the editor (A2), a live
status/log-tail monitor (A3), and, riding the landed QVF streaming
checkpoints, **live result streaming** that hot-reloads the scene as a job
converges (A4).

## See also

- [Live reload](vibe_view_live_reload.md), the viewer-side half of streaming a
  running job.
- [The `vq` queue](queue.md), the queue itself: submission, tags, fetching.
- [Running a calculation as a QVF container](../tutorial/qvf_job_containers.md),
  the container round trip end to end.
