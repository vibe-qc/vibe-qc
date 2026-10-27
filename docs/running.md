---
myst:
  html_meta:
    "description": "Run vibe-qc locally, in the background, through vq on a remote host, or in a single-node scheduler job, with explicit environments, resources, outputs, and verification."
    "og:title": "Running vibe-qc calculations"
---

# Running vibe-qc

Once [installation](installation.md) is complete, a calculation is a Python
input run with the interpreter from that installation. This page covers the
operational part: where to run, which interpreter to use, how to allocate CPU
and memory, how to monitor a long job, and which files prove that it finished.

For scientific choices such as method, basis, grid, and convergence studies,
use [Planning a calculation](tutorial/planning_a_calculation.md). For naming,
provenance, and archiving conventions, use [Good practices](good_practices.md).

## Choose how to run

| Situation | Recommended route | Why |
|---|---|---|
| One small calculation | invoke the checkout's `.venv/bin/python` directly | explicit and easy to reproduce |
| Interactive exploration | activate the venv or use its Jupyter kernel | convenient for repeated commands |
| Long job on one workstation | `tmux` plus the explicit interpreter | survives terminal and SSH disconnects |
| Batch work on a remote compute host | [vq](user_guide/queue.md) | records payload, resources, state, and fetched results |
| Managed HPC cluster | scheduler script with one OpenMP task | lets Slurm/PBS own CPU, RAM, time, and logs |

vibe-qc production drivers use single-node shared-memory parallelism. The
optional MPI layer provides lower-level collectives and experimental grid
work, not a general distributed HF, DFT, or post-HF driver.

## Use the correct Python

vibe-qc is installed into the checkout-local virtual environment. The safest
invocation names that interpreter explicitly:

```sh
<vibe-qc-checkout>/.venv/bin/python input-water.py
```

A bare `python3 input-water.py` uses whichever Python appears first on
`PATH`. If that interpreter does not contain vibe-qc, it fails with:

```text
ModuleNotFoundError: No module named 'vibeqc'
```

Verify both the interpreter and imported package before a production run:

```sh
<vibe-qc-checkout>/.venv/bin/python -c \
  "import sys, vibeqc; print(sys.executable); print(vibeqc.__file__)"
```

Both paths should belong to the intended checkout. This matters when several
release, development, or worktree installations exist on one machine.

### Activation is optional

Activation changes `PATH` for the current shell only:

```sh
source <vibe-qc-checkout>/.venv/bin/activate
python input-water.py
deactivate
```

It is convenient for an interactive session, but it does not propagate to a
new terminal, tmux pane, scheduler job, or Jupyter kernel. Scripts and batch
files are clearer when they use the absolute interpreter path.

## Keep calculations outside the source checkout

Copy a canonical example into a project run directory and execute the copy:

```sh
mkdir -p ~/vibeqc-runs/water-rhf/molecular
cp <vibe-qc-checkout>/examples/h2o.xyz \
   ~/vibeqc-runs/water-rhf/
cp <vibe-qc-checkout>/examples/molecular/input-h2o-rhf.py \
   ~/vibeqc-runs/water-rhf/molecular/
cd ~/vibeqc-runs/water-rhf/molecular
<vibe-qc-checkout>/.venv/bin/python input-h2o-rhf.py
```

This keeps generated files out of git and preserves the exact input beside
its results. This example loads `h2o.xyz` from the parent directory, so the
copy preserves that two-level layout. The examples use an output stem derived
from their own location, so copying the script also redirects its output
family.

The `output=` argument controls the stem and directory. For example,
`output="output-water"` creates siblings such as `output-water.out` and
`output-water.system`; `output="runs/dz/output-water"` writes below `runs/dz/`
when that directory exists. Do not assume every result follows the shell's
current directory if the script supplies an explicit path.

## Understand a completed run

The high-level `run_job` and `run_periodic_job` drivers write a coordinated
output family. Common members are:

| File | Purpose |
|---|---|
| `.out` | human-readable method, settings, convergence, energy, and references |
| `.system` | machine-readable versions, settings, status, and artifact manifest |
| `.qvf` | default typed visualization archive for vibe-view; disable explicitly when unwanted |
| `.molden`, `.xyz`, `.POSCAR`, `.xsf` | orbitals or structures, depending on the job |
| `.references`, `.bibtex` | citations selected by the route actually used |
| `.perf`, `.scf.jsonl` | optional performance and structured event records |

Check three separate outcomes before using a result:

| Outcome | Evidence | What it does not establish |
|---|---|---|
| Process finished | exit status and stderr | convergence or a correct physical model |
| Files were delivered | `[outputs].status`, per-file outcomes and expected artifacts in `.system` | scientific success; `complete` describes delivery |
| Driver accepted the result | `[outputs].result_status` and `[verdict]`, including `usable` and stability | agreement with a reference or convergence of the basis and sampling |

A `dry_run` manifest describes preflight only. An archive can exist even when
an optional component failed; inspect its `component_errors` before using that
component. Older manifests can lack verdict fields, which means unknown, not
success. The full contract and a small inspection example are in
[Input scripts and output files](user_guide/output_files.md#inspect-a-manifest-before-analysis).

For optimized geometries or correlated methods, also inspect the requested
stage's convergence and diagnostics. A converged SCF reference alone does not
prove completion of every subsequent stage.

## Monitor a foreground run

`run_job` enables live progress by default while the durable calculation
record goes to `{stem}.out`. To preserve terminal messages and Python errors
as a separate operator log, use `tee`:

```sh
# Bash or zsh: preserve failure from Python even when tee succeeds.
set -o pipefail
<vibe-qc-checkout>/.venv/bin/python input-water.py 2>&1 | tee console.log
run_status=$?
printf 'Run pipeline exit status: %s\n' "$run_status"
```

Capture `$?` immediately after the pipeline. Without `pipefail`, it normally
reports only whether `tee` succeeded, so an exception in Python can appear
successful to a wrapper. A nonzero pipeline status may also indicate a log-write
failure; inspect stderr. In a batch script, return this status with
`exit "$run_status"` before the scheduler treats the script as successful.
If using `set -e`, put the pipeline in an `if` statement when you need to
capture its status and perform cleanup after a failure.

The two files have different roles:

- `{stem}.out` is the calculation record formatted by vibe-qc;
- `console.log` is the shell-level transcript, including stderr and any
  messages emitted before the output channel opens.

Use `VIBEQC_OUTPUT_LEVEL=verbose` for more durable calculation detail, or
`VIBEQC_OUTPUT_LEVEL=quiet` for the essential verdict and warnings. Pass
`progress=False` or set `VIBEQC_LIVE_LOGGING=0` when a scheduler or parent
application should receive no live progress stream. See
[Logging and output levels](user_guide/logging.md).

For timing and memory analysis, request a performance sidecar:

```python
from vibeqc import Molecule, run_job

mol = Molecule.from_xyz("water.xyz")
run_job(
    mol,
    basis="cc-pvqz",
    method="rhf",
    output="output-water-qz",
    perf_log=True,
)
```

The `.perf` file records component wall and CPU time plus RSS snapshots. A
whole-process peak from `/usr/bin/time` complements those snapshots. The
[direct-SCF tutorial](tutorial/direct_scf_memory_tradeoff.md) explains how to
compare CPU time and memory without mixing hosts or thread counts.

## Control threads and memory

Set the OpenMP thread count explicitly for every non-trivial run:

```sh
OMP_NUM_THREADS=4 \
  <vibe-qc-checkout>/.venv/bin/python input-water.py
```

More threads are not automatically faster. Small calculations often saturate
after a few cores, while memory bandwidth and integral screening determine
larger cases. Benchmark the representative workload and record CPU time, wall
time, thread count, and host together. Scaling examples are in
[Parallel execution](tutorial/parallel_execution.md).

The high-level runner performs a memory preflight. Compare its complete
estimate with the memory allocated to the process, container, cgroup, or
scheduler job. For molecular SCF, also decide whether the dense
$8N_\mathrm{bf}^4$ conventional integral tensor fits or direct SCF is the
better resource policy. See [Memory budget](user_guide/memory.md).

## Run interactively

Install notebook tools into the same venv that contains vibe-qc:

```sh
<vibe-qc-checkout>/.venv/bin/python -m pip install ipykernel jupyterlab
<vibe-qc-checkout>/.venv/bin/python -m ipykernel install \
  --user --name vibeqc --display-name "Python (vibe-qc)"
```

Select that named kernel in Jupyter and verify `vibeqc.__file__` before doing
work. The [Jupyter guide](user_guide/jupyter.md) covers kernels, widgets, and
visualization in more detail.

## Leave a workstation job running

For an SSH session or a job longer than the terminal should own, use tmux:

```sh
tmux new -s water-qz
set -o pipefail
OMP_NUM_THREADS=8 \
  <vibe-qc-checkout>/.venv/bin/python input-water-qz.py 2>&1 | tee console.log
# Press Ctrl-B, then D, to detach.
tmux attach -t water-qz
```

`nohup` is acceptable for a simple unattended process, but tmux makes the
session, command, and live output easier to inspect. For a series of remote
jobs, use vq instead of maintaining a collection of PIDs and log files.

## Submit through vq

vq is the shipped SSH-backed queue for a laptop plus one or more remote
single-node hosts. It copies each payload into a clean job workspace, selects
the configured vibe-qc interpreter, enforces CPU, memory, and wall-time caps,
captures logs, and fetches outputs back to the client.

After completing the two-sided setup in
[The vq calculation queue](user_guide/queue.md),
the core workflow is:

```sh
vq submit input-water-qz.py \
  --cpus 8 --mem-mb 16000 --wall-time-seconds 7200
vq list
vq wait <job-id>
vq logs <job-id>
vq fetch <job-id> ./fetched-water-qz/
```

Use a directory payload when the input depends on geometry files, custom
bases, or helper modules. The [remote-job tutorial](tutorial/vq_queue_remote_job.md)
walks through submission, monitoring, and result inspection end to end.

## Submit to Slurm or another HPC scheduler

On a managed cluster, let the scheduler own resources and logs. A minimal
single-node Slurm script is:

```sh
#!/bin/bash
#SBATCH --job-name=water-qz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${SLURM_SUBMIT_DIR}"
srun <vibe-qc-checkout>/.venv/bin/python input-water-qz.py
```

A Slurm-only task step with a numeric normal `SLURM_STEP_ID`,
`SLURM_PROCID=0` and `SLURM_STEP_NUM_TASKS=1` runs serially without the
optional `mpi4py` dependency. If the selected launcher plugin adds MPI rank
metadata (PMI, PMIx, Open MPI or MVAPICH), or a wrapper sets
`VIBEQC_MPI_REQUIRED=1`, MPI remains required even for one task. Multi-task
steps also require MPI. Incomplete, malformed or conflicting launcher rank
metadata is rejected rather than treated as serial.

Submit with `sbatch run.slurm`. PBS, LSF, and SGE use the same principles:
one task, an explicit CPU count, an explicit memory request, a wall-time cap,
the intended venv interpreter, and a project or scratch run directory. Do not
run production calculations or write output inside the source checkout.

## Verify an installation or a development change

An end user normally needs the banner and one small calculation:

```sh
<vibe-qc-checkout>/.venv/bin/python -c \
  "from vibeqc import print_banner; print_banner()"
mkdir -p ~/vibeqc-runs/install-check
cp <vibe-qc-checkout>/examples/quickstart.py ~/vibeqc-runs/install-check/
cd ~/vibeqc-runs/install-check
<vibe-qc-checkout>/.venv/bin/python quickstart.py
```

Contributors should use the smallest check that exercises the changed
behavior. Documentation prose needs a text/link check; code changes normally
start with a focused test:

```sh
.venv/bin/python -m pytest tests/test_<area>.py -q
```

Use `scripts/test_gate/run_full_suite.py --list-lanes` to select a broader
lane when the change warrants one. Rebuild native code only when the check
needs a changed native implementation, and verify the loaded module identity.
A passing source check is separate from scientific acceptance; see the
[testing policy](testing.md).

See [Developer test lanes](developer_test_lanes.md) and
[Contributing](contributing.md) for impact selection and release confidence
profiles.

The QVF writer-to-viewer test skips when vibe-view is not installed in the
active environment. To test interoperability in one environment, first clone
the independent viewer repository next to the core checkout, then run from
vibe-qc:

```sh
.venv/bin/python -m pip install -e '../vibe-view[viewer]'
```

The old uv local-source mapping is gone. Normal operation can use separate
environments with QVF files as the interface; see
[Install both](getting_started.md#install-both).

## Quick diagnosis

| Symptom | First check | Continue with |
|---|---|---|
| `ModuleNotFoundError: No module named 'vibeqc'` | print `sys.executable` and `vibeqc.__file__` with the intended venv | [Installation](installation.md) |
| native library cannot be loaded | run the banner and `scripts/doctor.sh` | [Installation troubleshooting](installation.md#common-issues) |
| basis entry is missing | confirm spelling, element coverage, and basis family | [Basis sets](user_guide/basis_sets.md) |
| memory preflight aborts | inspect the largest estimate category before overriding | [Memory budget](user_guide/memory.md) |
| molecular SCF stalls | check state, geometry, basis, and initial guess before adding aids | [SCF convergence](user_guide/scf_convergence.md) |
| periodic SCF oscillates or has an impossible energy | preserve the input and treat it as a possible implementation bug | [Troubleshooting](troubleshooting.md#scf-didnt-converge) |
| expected output is absent | inspect `.system`, `.out`, stderr, and the output stem | [Output files](user_guide/output_files.md) |

## Where to go next

- [Quickstart](quickstart.md) for the first molecular and periodic inputs.
- [Tutorial learning paths](tutorial/index.md#choose-a-learning-path) for a
  method-oriented course.
- [Examples catalog](https://github.com/vibe-qc/vibe-qc/blob/main/examples/README.md)
  for complete inputs organized by goal.
- [Troubleshooting](troubleshooting.md) for symptom-based recovery and known
  implementation limitations.
