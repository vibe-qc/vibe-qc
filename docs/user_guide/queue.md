---
myst:
  html_meta:
    "description": "Queue vibe-qc calculations onto another machine with vq. Point vq at your vibeqc-dev and vibeqc-release venvs, submit QVF containers, run CRYSTAL and ORCA through the shipped wrappers, and fetch results back."
    "og:title": "vibe-qc, running calculations through the vq queue"
    "og:description": "The vibe-qc side of vq: program registry, branch routing, --vibeqc-preflight, QVF submission, external-program wrappers, and result readback. vq's own reference lives at vibe-qc.com/vibe-queue/docs/."
    "og:image": "https://vibe-qc.com/docs/_static/logo/vibe-queue-social.png"
    "og:image:width": "1200"
    "og:image:height": "630"
    "og:image:alt": "vq: Queue the work. Keep the results."
    "twitter:card": "summary_large_image"
    "twitter:image": "https://vibe-qc.com/docs/_static/logo/vibe-queue-social.png"
    "twitter:image:alt": "vq: Queue the work. Keep the results."
---

(the-vq-calculation-queue)=

# Running vibe-qc through the vq queue

```{figure} ../_static/logo/vibe-queue-social.svg
:alt: vq: Queue the work. Keep the results.
:width: 1200px
:class: product-art

Open the [vq manual](https://vibe-qc.com/vibe-queue/docs/) for the independently maintained queue. This page covers submitting vibe-qc calculations.
```

`vq` is the toolset's cross-machine job queue. It takes a command, runs it on
the machine you point it at (a workstation over SSH, or a cluster behind PBS or
SLURM), and tracks it until you fetch the results back.

This page covers **the vibe-qc side of vq**: how to point it at your vibe-qc
environments, how to submit a calculation, and how to get the outputs home.
It does not document vq itself.

```{admonition} vq is a separate project, with its own documentation
:class: important

vq used to live inside the vibe-qc repository, in `vibe-queue/`. It does not
any more: it is its own project, with its own release line, and its own
documentation site.

**Everything general about vq lives at
<https://vibe-qc.com/vibe-queue/docs/>**: installing it, running the daemon,
resource caps, the job lifecycle, the web dashboard, multi-user deployment,
and the machine-readable agent contract. Anything on this page that is not
specific to vibe-qc is a summary with a link, not a second copy.
```

(when-to-use-vq)=
(when-not-to-use-vq)=

## When to reach for the queue

* **The laptop runs out of cores or RAM.** Supercells, dense k-meshes,
  transition-metal clusters and big-basis hybrid DFT want a bigger box. Queue
  those; keep the laptop for development.
* **You want a record of what you ran.** Every submission is stored as a
  durable job spec: the command, the environment, the resource caps, the
  terminal state, and which vibe-qc checkout produced the number.
* **You are running many jobs.** A sweep that takes hours is exactly where
  losing track becomes expensive.
* **You want the box to survive a greedy job.** On Linux, vq caps CPU and
  memory with cgroup v2 rather than with a heuristic.

Not worth it for a three-second molecule on the laptop, and not the right
shape for interactive work: for notebooks, use the
[Jupyter Lab integration](jupyter.md) instead.

(installation)=

(local-laptop)=
(remote-compute-box)=

## Install it once, on both sides

vq is installed from **its own repository**, not from your vibe-qc checkout:

```sh
git clone https://github.com/vibe-qc/vibe-queue.git
cd vibe-queue
./scripts/install.sh
.venv/bin/vq --version
```

You need it on the laptop you submit from and on the machine that runs the
work. It needs Python 3.12 or newer, and it installs nothing on the remote
host but itself: the entire transport is SSH, so if you can `ssh` to a
machine, vq can queue work on it.

For how the four toolset installs relate, including what each uninstall
preserves, see
[Install and maintain the vibe toolset](../toolset_lifecycle.md). For daemon
supervision, profiles and updates, see
[vq's operator documentation](https://vibe-qc.com/vibe-queue/docs/operator/index.html).

## Point vq at your vibe-qc environments

This is the part that is specific to vibe-qc, and it is worth getting right
once.

(configuration)=
(multi-host)=

### The laptop: hosts and branch routing

vq reads `~/.config/vq/config.toml` on the **laptop**. A minimal working
config that can reach one compute machine and choose between a development
and a release vibe-qc:

```toml
# ~/.config/vq/config.toml

# Used whenever you omit the host from a vq command.
default_host = "compute"

[hosts.compute]
# An SSH alias from ~/.ssh/config, or a literal user@host. vq does not
# manage hostnames, ports or keys; if `ssh compute` works, vq works.
ssh = "compute"

# Absolute path to vq on the remote. The remote shell's default PATH
# usually does not include the venv vq was installed into.
remote_vq = "/home/USER/vibe-queue/.venv/bin/vq"

# Default interpreter for single-file submits: a venv with vibe-qc in it.
remote_python = "/home/USER/vibeqc-dev/.venv/bin/python"

# Optional: let `--branch` pick a vibe-qc clone by name instead of
# hard-coding an interpreter path at every submit.
[hosts.compute.branches]
main    = "/home/USER/vibeqc-dev/.venv/bin/python"
release = "/home/USER/vibeqc-release/.venv/bin/python"

[hosts.compute.branch_aliases]
dev     = "main"
latest  = "release"
```

Add more machines by adding more `[hosts.<name>]` blocks, then route with
`--host`, or with the host as the first positional argument.

The annotated template with every available key is
[`docs/config.toml.example`](https://github.com/vibe-qc/vibe-queue/blob/main/docs/config.toml.example)
in the vq repository.

### The remote: the program registry

`[programs.<name>]` describes an actual installed thing: a venv with a git
checkout behind it, or an external binary. It is what `vq programs` lists, and
what `--program` selects.

```{important}
Program entries live in the config **on the host that actually has the venvs
and binaries**, not on your laptop. When you run `vq programs` from the
laptop, vq delegates over SSH to the remote so the availability checks happen
where the files are.
```

```toml
# ~/.config/vq/config.toml on the compute machine

[programs.vibeqc-dev]
kind = "venv"
python = "/home/USER/vibeqc-dev/.venv/bin/python"
git_dir = "/home/USER/vibeqc-dev"
branch = "main"
import_check = "vibeqc"
description = "vibe-qc development (main branch)"
# Which [hosts.X.branches] names this environment serves, so that
# `vq admin update vibeqc-dev` pauses only the jobs routed through it
# rather than the whole queue.
provides_branches = ["main", "dev"]

[programs.vibeqc-release]
kind = "venv"
python = "/home/USER/vibeqc-release/.venv/bin/python"
git_dir = "/home/USER/vibeqc-release"
branch = "release"
import_check = "vibeqc"
description = "vibe-qc release (latest tag)"
provides_branches = ["release", "latest"]
```

A job submitted with `--program vibeqc-dev` gets `VQ_PROGRAM_BIN`,
`VQ_PROGRAM_PYTHON` and `VQ_PROGRAM_GIT_DIR` in its environment, so the
payload can call `"$VQ_PROGRAM_BIN/vibeqc"` without hard-coding this host's
paths. Prefer that to a literal checkout path in every script.

(submission-forms)=

(your-first-job)=
(single-file-most-common)=
(directory-submit-sweeps-multi-file-inputs)=
(pre-packed-tarball)=
(resource-caps)=

## Your first calculation

An ordinary vibe-qc input. Nothing in the file is vq-specific, and the same
script runs unchanged under `python water.py` on your laptop:

```python
# water.py
import vibeqc as vqc

mol = vqc.Molecule([
    vqc.Atom(8, [0.0,  0.00,  0.00]),
    vqc.Atom(1, [0.0,  1.43, -0.98]),
    vqc.Atom(1, [0.0, -1.43, -0.98]),
])
result = vqc.run_job(mol, basis="6-31g*", method="RHF", output="water")
print(f"E(SCF) = {result.energy:.6f} Ha")
```

Queue it, watch it, fetch it:

```sh
vq submit compute water.py          # prints a job id
vq queue                            # what is happening right now
vq status <jobid>                   # one job, with the tail of its output
vq wait <jobid>                     # block until it reaches a terminal state
vq fetch <jobid> -o results/        # bring the workspace back
```

Declaring what the job needs gets it scheduled honestly rather than
conservatively, because an undeclared job is charged an assumed footprint:

```sh
vq submit compute --cpus 8 --mem-mb 16000 --time-limit 04:00:00 water.py
```

```{note}
`vq wait` returning does **not** mean the calculation succeeded. It means the
job reached a terminal state, and `failed`, `killed`, `oom_killed`,
`time_exceeded` and `aborted_by_queue` are all terminal. Check the state.
```

(multi-venv-branch-routing-v0-5-6)=

## Choosing which vibe-qc runs the job

Three mechanisms, from loosest to strictest.

```sh
# By interpreter path.
vq submit compute --python /home/USER/vibeqc-dev/.venv/bin/python water.py

# By branch name, resolved through [hosts.compute.branches].
# Single-file submits only, and mutually exclusive with --python.
vq submit compute --branch release water.py

# By registered program, which also stamps a provenance identity.
vq submit compute water.py --program vibeqc-release
```

For a number you intend to publish, pin the source revision as well:

```sh
vq submit compute water.py --program vibeqc-release \
    --expected-sha 0123456789abcdef0123456789abcdef01234567
```

`--expected-sha` requires the checkout to be at that revision when the job is
queued, then snapshots the host-validated revision so **dispatch fails if the
checkout moves before the job actually runs**. Without it, a queued job can
cross a runtime rollout and silently run against a different vibe-qc than the
one you submitted against.

## Submitting a QVF container

A `.qvf` positional input is a first-class payload, not a Python script:

```sh
vq submit compute job.qvf --program vibeqc-dev
```

vq resolves the named program and invokes the installed vibe-qc CLI as
`python -m vibeqc._cli run job.qvf`. It never falls through to
`python job.qvf`, and it never executes `run.record.input`. Do not pass
`--python` or `--branch` for a QVF submit; the program selects the runtime.

A settled container is refused by default, because running another sequence
into it is usually a mistake. Pass `--qvf-force` only when you deliberately
want another sequenced `run.record`.

Fetch just the updated container and leave the queue's logs and metadata on
the server:

```sh
vq fetch compute <jobid> --name job.qvf -o results/
```

That publishes `results/job.qvf` atomically. Retrying is idempotent only when
the existing file is byte-identical: vq will not overwrite a different result.

(vq-core-preflight)=

## Output-aware submission with `--vibeqc-preflight`

```sh
vq submit localhost --vibeqc-preflight water.py
```

The flag runs your script once with `VIBEQC_DRY_RUN=1` before queueing,
harvests the resulting `.system` manifest, and records the declared artefacts
on the job spec. The queue then knows what the job will write before it writes
it, so fetch can be selective and progress can be reported per declared
artefact instead of by tarring the whole workspace.

It costs a second or two at submit time, and it executes your Python file, so
it is opt-in. Three limits worth knowing:

* It is meaningful only for scripts that go through `vibeqc.run_job`. Other
  scripts are a no-op, and a preflight failure is non-fatal.
* It applies to **local submits**, selected explicitly with `localhost` above.
  It is not forwarded to a remote submit. For a remote job, run the dry run
  explicitly first, then submit normally.
* It is disabled inside `--array` and `--chain`, which run one source many
  times.

You can always run the same dry run by hand, which is a cheap way to see the
file family a calculation will produce without paying for the SCF:

```sh
VIBEQC_DRY_RUN=1 python water.py
```

(external-program-workflows-crystal--orca--pyscf)=

(external-program-workflows-crystal-orca-pyscf)=

## CRYSTAL, ORCA and other external programs

vibe-qc treats other quantum-chemistry codes as external programs. vq runs
them out of process, and ships wrappers for the two whose calling conventions
its `--`-form argument pass-through cannot express. vq has no shell, so it
cannot interpret `<` or `>` itself; the wrappers own the redirection.

Both live in the vq repository under
[`contrib/`](https://github.com/vibe-qc/vibe-queue/tree/main/contrib).

(crystal14-pcrystal-properties14)=

### CRYSTAL

Serial `crystal` reads from stdin and writes to stdout; parallel `Pcrystal`
instead reads a file literally named `INPUT` in the working directory. The
wrapper hides both conventions and restores the workspace on exit.

```sh
# Parallel CRYSTAL14, 14 MPI ranks by default.
vq submit compute -d ./calc --cpus 14 \
    -- bash /home/USER/vibe-queue/contrib/run-crystal.sh INPUT.d12

# Serial, no mpirun startup cost.
vq submit compute -d ./calc --cpus 1 \
    -- bash /home/USER/vibe-queue/contrib/run-crystal.sh --serial INPUT.d12

# A specific rank count.
vq submit compute -d ./calc --cpus 8 \
    -- bash /home/USER/vibe-queue/contrib/run-crystal.sh --np 8 INPUT.d12

# PROPERTIES post-processing rather than an SCF.
vq submit compute -d ./prop --cpus 14 \
    -- bash /home/USER/vibe-queue/contrib/run-crystal.sh --properties prop.d3
```

`--demo` selects the CRYSTAL23 demo binaries. It is serial only, so combining
it with `--np` is rejected, and it is capped at ten atoms per primitive cell.

(orca-6-1)=

### ORCA

ORCA spawns its own MPI ranks from the `%pal nprocs N end` block inside the
input file, so **do not wrap it in `mpirun`**. It also has to be invoked by
absolute path, because it locates its sibling executables relative to
`argv[0]`; the wrapper resolves that for you.

```sh
vq submit compute -d ./orca_run --cpus 8 \
    -- bash /home/USER/vibe-queue/contrib/run-orca.sh input.inp
```

Keep `--cpus` and the input's `%pal` block in agreement: the first is what the
queue reserves and enforces, the second is what ORCA actually spawns.

On success the wrapper deletes the bulky regenerable scratch and keeps the
canonical artefacts (`.out`, `.gbw`, `.property.txt`, `.engrad`, `.xyz`,
`.densities`, `.hess`). A failed run keeps everything, which is what you want
when you are working out why it failed. Pass `--keep-scratch` to keep it all
regardless.

(pyscf-as-a-comparison-parity-reference)=

### Everything else

Register any other binary as a program and submit against it:

```toml
[programs.xtb]
kind = "binary"
binary = "/home/USER/xtb-dist/bin/xtb"
description = "xtb 6.7.1 GFN-xTB / GFN-FF semiempirical"
```

PySCF needs nothing special: it is installed in the vibe-qc test extra, so a
PySCF script submits exactly like a vibe-qc one.

(monitoring-management)=
(fetching-outputs)=

## Getting results back

```sh
vq fetch <jobid> -o results/            # the workspace
vq fetch <jobid> --name job.qvf -o results/   # one named artifact
vq fetch --workdir <jobid>              # the per-job scratch directory
vq fetch-all compute -o results/        # every terminal job on a host
```

A fetched directory is what you would have had if you had run the job locally,
including the `.bibtex` and `.references` files that
[citations](citations.md) explains how to use, plus vq's own `stdout.log` and
`stderr.log`.

The `.system` manifest records what produced the number, which is the part
worth keeping when you come back to a result months later:

```sh
python -c '
import tomllib, sys
with open(sys.argv[1], "rb") as f:
    m = tomllib.load(f)
print("CPU   :", m["cpu"]["model"])
print("OMP   :", m["cpu"]["omp_threads_used"])
print("RAM   :", m["memory"]["total_gb"], "GB")
print("vibeqc:", m["vibeqc"]["version"], m["vibeqc"]["git_sha"])
' results/<jobid>/water.system
```

(architecture)=
(updating-repairing-and-removing-vq)=
(orphan-exit-code-recovery-v0-5-9)=
(web-dashboard)=
(operator-controls-pause-resume-throttle-drain)=
(workspace-cleanup-v0-5-10)=
(daemon-admin)=
(what-happens-at-host-reboot)=
(refreshing-the-remote-vibe-qc-venv-after-a-release-v0-5-20)=
(watching-the-daemon)=
(concurrency)=
(troubleshooting)=
(version-history-recent)=
(roadmap-vq-s-own)=

## Where the rest of vq is documented

Everything below is vq's own documentation, not vibe-qc's:

| You want to | Read |
| --- | --- |
| Submit, watch, fetch, arrays, chains, retries | [Running jobs](https://vibe-qc.com/vibe-queue/docs/user/index.html) |
| Install and supervise the daemon, caps, fleets | [Running a host](https://vibe-qc.com/vibe-queue/docs/operator/index.html) |
| Drive vq from a program, with stable JSON | [The agent contract](https://vibe-qc.com/vibe-queue/docs/agent/index.html) |
| Know what a vq version means for a fleet | [Version compatibility](https://vibe-qc.com/vibe-queue/docs/version_compatibility.html) |
| Run on a PBS or SLURM cluster | [Scheduler runtime deployment](https://vibe-qc.com/vibe-queue/docs/scheduler_runtime_deployment.html) |
| Use the read-only web dashboard | [The web dashboard](https://vibe-qc.com/vibe-queue/docs/web.html) |

## See also

* [Submitting a job to a remote machine with vq](../tutorial/vq_queue_remote_job.md),
  a complete worked cycle for one periodic calculation.
* [Install and maintain the vibe toolset](../toolset_lifecycle.md).
* [Jupyter Lab integration](jupyter.md), for interactive work, which is the
  shape vq is deliberately bad at.
* [vibe-queue on GitLab](https://github.com/vibe-qc/vibe-queue).
