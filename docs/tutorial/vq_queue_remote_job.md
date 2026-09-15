---
myst:
  html_meta:
    "description": "A complete vq submit-and-fetch cycle for one periodic vibe-qc calculation: dry-run pre-flight, submission, monitoring, and result readback."
    "og:title": "Submitting a vibe-qc job to a remote machine with vq"
    "og:description": "Queue an MgO PBE0 calculation onto a compute host over SSH, watch it through the job lifecycle, and fetch the declared artefacts back."
---

# Submitting a job to a remote machine with `vq`

Supercells, dense k-meshes, transition-metal clusters and big-basis hybrid
DFT can exceed the resources available on a laptop. `vq` is the toolset's job queue:
a daemon on the compute machine that accepts work from your laptop over SSH
and hands the outputs back when each job finishes.

This tutorial walks through a remote submit-and-fetch cycle. First run a
**manual dry-run preflight** to inspect the planned outputs, then submit the
calculation. Remote submission does not forward `--vibeqc-preflight`.

If you have used SLURM or PBS, the verbs map cleanly: `sbatch` becomes
`vq submit`, `squeue` becomes `vq queue`, `scancel` becomes `vq kill`. The
queue also understands QVF payloads and can fetch a named result file;
see the [QVF submission guide](../user_guide/queue.md#submitting-a-qvf-container).

```{important}
This tutorial assumes vq is **already installed on both your laptop and a
remote compute machine**, with `default_host` pointed at the remote. vq is a
separate project from vibe-qc, with
[its own documentation](https://vibe-qc.com/vibe-queue/docs/); the setup is
covered in
[Running vibe-qc through the vq queue](../user_guide/queue.md#install-it-once-on-both-sides).
It is a one-time step.
```

## The system

A small but representative job: MgO in the rocksalt structure, at PBE0 /
pob-TZVP, Gamma-only, through the native GDF driver. The cell below is the
two-atom FCC primitive cell. Its Gamma-only sampling
is for this workflow example, not a converged bulk benchmark.

Working directory on your laptop:

```text
~/vibeqc-runs/mgo-rocksalt/
    input-mgo-pbe0.py
```

`input-mgo-pbe0.py`:

```python
import numpy as np
import vibeqc as vq

# MgO rocksalt with an illustrative lattice constant a = 4.21 A.
a = 4.21 * 1.8897259886    # 7.957 bohr

sysp = vq.PeriodicSystem(
    dim=3,
    lattice=(a / 2) * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
    unit_cell=[
        vq.Atom(12, [0.0, 0.0, 0.0]),                # Mg
        vq.Atom(8,  [a/2, a/2, a/2]),                # O
    ],
)

vq.run_periodic_job(
    sysp,
    basis=vq.BasisSet(sysp.unit_cell_molecule(), "pob-tzvp"),
    method="RKS",
    functional="PBE0",
    jk_method="gdf",
    kpoints=[1, 1, 1],        # Gamma-only
    output="output-mgo-pbe0",
)
```

This is an ordinary vibe-qc input. The same file runs unchanged under
`python input-mgo-pbe0.py` locally or through `vq submit`; there is no
vq-specific markup in it.

(step-1-dry-run-pre-flight-locally)=

## Step 1, dry-run pre-flight

Before queueing, check what the job will write:

```sh
cd ~/vibeqc-runs/mgo-rocksalt/
VIBEQC_DRY_RUN=1 python input-mgo-pbe0.py
```

This short-circuits the runner after the method resolves but before any
compute. It writes a one-shot `.system` manifest with
`[outputs].status = "dry_run"` and prints the declared artefacts.

The printed plan lists the outputs enabled by the runner and its profile.
Use that actual plan rather than assuming every possible sidecar will be
written. The dry run does not test SCF convergence or the remote environment.

(step-2-submit-to-the-remote-queue)=

## Step 2, submit to the queue

```sh
vq submit compute input-mgo-pbe0.py
```

vq uploads the input into a per-job workspace, queues it under the host's
resource policy, and returns a job id. In the commands below, replace
`c0ff50a06462` with that returned id.

The script is **not running yet**. It is queued, and the daemon dispatches it
when the host's CPU, memory and concurrency budgets all allow. That gap is
the point of a queue, and `vq status` will tell you which budget you are
waiting on.

```{tip}
Declare what the job needs at submit time, so it is scheduled against real
numbers rather than an assumed footprint:

    vq submit compute --cpus 4 --mem-mb 8000 --time-limit 00:30:00 \
        input-mgo-pbe0.py

On Linux the caps are enforced with cgroup v2, so one greedy job cannot take
the box down.
```

```{note}
`--vibeqc-preflight` is a local-submit option and is not forwarded over SSH.
The manual dry run in step 1 remains useful, but this remote job is submitted
without automatically populated `expected_outputs`. Array and chain submits
also disable automatic preflight.
```

## Step 3, monitor

```sh
# Snapshot of the queue.
vq queue
# JOBID         STATE     ELAPSED   NAME              SCRIPT
# c0ff50a06462  running   00:00:08  input-mgo-pbe0    input-mgo-pbe0.py

# One job in detail, with the tails of stdout and stderr.
vq status c0ff50a06462

# Block until the job reaches a terminal state. Ctrl-C exits the wait;
# the job keeps running.
vq wait c0ff50a06462
```

`vq queue` is the equivalent of `squeue`, and `vq status` is closer to
`scontrol show job`. Both read on demand; there is no poll loop between
calls.

The states you will see:

```text
pending  ->  running  ->  completed          the happy path
                       ->  failed             non-zero exit
                       ->  time_exceeded      wall-time enforcement
                       ->  oom_killed         the watchdog reclaimed it
                       ->  starved            the watchdog saw no CPU progress
                       ->  killed             you called vq kill
                       ->  interrupted        the job vanished unexplained
                       ->  aborted_by_queue   the queue ended it
```

`suspended` is the non-terminal state `vq pause` produces; `vq resume` puts
the job back to `running`.

```{warning}
Terminal is not the same as successful. `vq wait` returning tells you the job
finished, not that the calculation worked. Check the state.
```

Separately from the job state, the `.system` manifest's `[outputs].status`
field tracks the **output side**: `"running"` while the job is alive, then
`"complete"` or `"crashed"`. That is what lets vq distinguish "the SCF
crashed and wrote a dump" from "the daemon died and the job was orphaned".

## Step 4, fetch the outputs

Once the job is `completed`:

```sh
vq fetch c0ff50a06462 -o ./outputs/
```

The workspace streams back over SSH. With `--job-name` at submit time the
destination is `./outputs/<jobname>-<jobid>/`; otherwise `./outputs/<jobid>/`:

```text
outputs/c0ff50a06462/
    input-mgo-pbe0.py             # the script you submitted
    output-mgo-pbe0.out           # SCF log
    output-mgo-pbe0.qvf           # structured result archive
    output-mgo-pbe0.system        # manifest: plan, outputs status, hardware
    output-mgo-pbe0.xyz           # geometry (extended XYZ)
    output-mgo-pbe0.xsf           # XCrySDen structure
    output-mgo-pbe0.bibtex        # citations
    output-mgo-pbe0.references
    stdout.log                    # vq-captured stdout
    stderr.log                    # vq-captured stderr
```

The listing illustrates the file family; the exact sidecars depend on output
settings and successful completion. The `.bibtex` and `.references` files are assembled automatically, as
[citations](../user_guide/citations.md) describes. Drop the BibTeX file into
your manuscript and cite from it.

(step-5-read-the-result-on-the-laptop)=

## Step 5, read the result

Check the fetched log for convergence and read its reported energy. No
reference energy or iteration count is asserted for this workflow example:

```sh
grep -E "Total energy|converged" outputs/c0ff50a06462/output-mgo-pbe0.out
```

Cross-check the manifest to know what produced the number. This is the part
worth keeping when you return to a result months later:

```sh
python -c '
import tomllib, sys
with open(sys.argv[1], "rb") as f:
    m = tomllib.load(f)
print("CPU   :", m["cpu"]["model"])
print("OMP   :", m["cpu"]["omp_threads_used"])
print("RAM   :", m["memory"]["total_gb"], "GB")
print("vibeqc:", m["vibeqc"]["version"], m["vibeqc"]["git_sha"])
' outputs/c0ff50a06462/output-mgo-pbe0.system
```

If you need that guarantee at dispatch time rather than after the fact,
submit with `--program` and `--expected-sha`, which makes the job refuse to
run if the checkout moved while it sat in the queue. See
[choosing which vibe-qc runs the job](../user_guide/queue.md#choosing-which-vibe-qc-runs-the-job).

(step-6-make-headless-qvf-screenshots)=

## Step 6, headless QVF screenshots

For documentation artefacts the usual pattern is two queue jobs: run the
calculation with the managed vibe-qc program, then submit a small capture
payload with the managed vibe-view program. The capture payload works from a
fetched or staged QVF file and writes PNGs into the job workspace:

```text
qvf-capture/
    result.qvf
    capture.sh
```

`capture.sh`:

```sh
set -euo pipefail

export PYVISTA_OFF_SCREEN="${PYVISTA_OFF_SCREEN:-True}"

"${VQ_PROGRAM_BIN}/vibe-view" capture-selftest
"${VQ_PROGRAM_BIN}/vibe-view" capture result.qvf -o structure.png
```

To render a density or orbital, first list the file's sections with
`vibe-view info result.qvf`, then pass an existing volume section ID with
`capture --section ID`. QVF section IDs and available fields depend on what
the calculation wrote; a default job need not contain density or MO volumes.

Submit it against the managed program:

```sh
JOBID=$(vq submit compute -d qvf-capture/ \
    --program vibeview-dev \
    --mem-mb 8000 \
    --time-limit 00:20:00 \
    --job-name qvf-capture \
    -- bash capture.sh)

vq wait compute "$JOBID"
vq fetch compute "$JOBID" -o calculations/archive/vq_fetch/
```

On hosts whose healthcheck needs Xvfb, wrap the command with `xvfb-run -a`.

Program-managed jobs should use `VQ_PROGRAM_BIN`, `VQ_PROGRAM_PYTHON` and
`VQ_PROGRAM_GIT_DIR` rather than hard-coded checkout paths. If a healthcheck
fails because the managed program is stale, ask the queue operator to refresh
it with `vq admin update`. Shared managed checkouts are not edited by hand.

## Common operations

### Re-running the same job

```sh
vq resubmit c0ff50a06462
```

Clones the job into a fresh workspace with the same inputs and returns a new
job id. Useful when the original hit something transient, a flaky mount or an
OOM caused by a co-tenant, and you want to try again without rebuilding the
workspace.

(killing-a-runaway-job)=

### Stopping a job

```sh
vq kill c0ff50a06462
```

SIGTERM, escalating to SIGKILL after the grace period. The job ends in
`killed`, and `[outputs].status` becomes `"crashed"`, because the SCF did not
finish. For the difference between killing, aborting and letting a job yield,
see [aborting a job](https://vibe-qc.com/vibe-queue/docs/abort.html).

(cleaning-up-old-workspaces)=

### Reclaiming disk

```sh
# Dry run: what would be archived, without touching anything.
vq cleanup --archive --older-than 14d

# Do it.
vq cleanup --archive --older-than 14d -x
```

Cleanup verbs are dry-run by default; `-x` is what executes them. Archiving
tars the workspace and keeps the job spec, so the job stays in the record
with an `archived_at` stamp. `--delete` removes the spec too. There is also a
daemon-side sweep, described in
[automatic cleanup](https://vibe-qc.com/vibe-queue/docs/auto-cleanup.html).

(submitting-an-entire-directory)=

### Submitting a whole directory

For multi-file inputs, a geometry file plus a script that reads it, or a
sweep over several functionals:

```sh
vq submit -d ./my_sweep_dir/ -- python run.py
```

`-d` names the directory to copy across, `--` ends vq's own options, and the
rest is the literal command to run inside the workspace.

For a genuine sweep, `--array N` submits N parallel siblings from one source
and `--chain N` runs N in strict sequence, each waiting for the previous to
succeed.

(pausing-the-queue)=

### Holding the queue back

```sh
vq pause                 # stop dispatching; running jobs continue
vq resume                # dispatch again
vq drain --max-jobs 1    # partial drain: cap concurrency without pausing
```

Useful when you want the box for an interactive session and would rather vq
did not fill the CPU underneath you. To make one *running* job step aside
instead, `vq throttle` lowers its CPU weight; see
[throttling](https://vibe-qc.com/vibe-queue/docs/throttle.html).

(why-vibeqc-preflight-matters)=

## What the manual preflight proves

It checks that the local input reaches the runner and inspects its planned
outputs without paying for the SCF. It does not prove that the remote host
has the same interpreter, basis inventory, or source revision. Use the
program registry and `--expected-sha` when that identity matters.

For the local-submit automatic preflight contract and its limitations, see
{ref}`output-aware submission <vq-core-preflight>`.

(what-s-still-local-only)=

## What vq does not do for you

* **GPU resource claims.** CPU and memory caps are honoured; there is no GPU
  claim machinery.
* **Interactive sessions.** vq is batch-shaped by design. Use
  [Jupyter Lab](../user_guide/jupyter.md) for interactive work.
* **Unregistered clusters.** vq ships PBS and SLURM backends for hosts
  registered in its configuration, including durable submission, monitoring
  and artefact fetch. For an unregistered cluster, an interactive allocation,
  or a workflow outside vq's declared-resource model, use the site's own
  scheduler tools. See
  [scheduler runtime deployment](https://vibe-qc.com/vibe-queue/docs/scheduler_runtime_deployment.html).

(resources)=

## Cost of this example

Runtime and peak memory depend on the native libraries, grid and integral
settings, thread count, and host. The sample resource limits above are a
scheduling request, not measured requirements. Inspect the job's log and
manifest, and adjust the next request from that evidence.

(references)=

## Next

* [Running vibe-qc through the vq queue](../user_guide/queue.md), the
  reference for the vibe-qc side: program registry, branch routing, QVF
  submission, and the CRYSTAL and ORCA wrappers.
* [Auto-citations, from `.out` to manuscript bibliography](auto_citations.md).
* [Cross-validating against ORCA, Psi4 and PySCF](cross_validation.md), which
  runs the same input through several codes over vq.
* [vq's own documentation](https://vibe-qc.com/vibe-queue/docs/) for
  everything the queue does that is not about vibe-qc.
