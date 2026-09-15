# Round-trip QVF job containers

These examples create `.qvf` files that describe calculations before they
run. The same file moves through the complete lifecycle:

```text
pending QVF -> vibeqc run -> converged/failed QVF -> view or fetch
```

The container preserves the declarative `job.spec`, gains canonical result
sections, and embeds the executed spec, full log, citations, and system
manifest. Sidecars are still written for ordinary command-line use, but the
settled QVF is sufficient on its own.

Run the molecular example from the repository root:

```sh
.venv/bin/python examples/qvf_containers/build_h2_job.py
.venv/bin/python examples/qvf_containers/inspect_job.py \
    examples/qvf_containers/runs/h2-rhf.qvf
.venv/bin/vibeqc run examples/qvf_containers/runs/h2-rhf.qvf
.venv/bin/python examples/qvf_containers/inspect_job.py \
    examples/qvf_containers/runs/h2-rhf.qvf
```

The periodic example is the same workflow with a helium atom in an
eight-bohr cubic cell and the basis-free SCC-DFTB route:

```sh
.venv/bin/python examples/qvf_containers/build_periodic_he_job.py
.venv/bin/vibeqc run examples/qvf_containers/runs/he-periodic-scc.qvf
```

For viewing, install the separate [vibe-view repository](https://github.com/vibe-qc/vibe-view)
using [viewer setup](../../docs/tutorial/vibe_view_getting_started.md). Put its
`vibe-view` executable on PATH, or substitute its absolute path below. The
core and viewer can use different virtual environments: only the `.qvf` file
passes between them. The QVF reference toolkit is not required.

View either archive without a graphical display:

```sh
vibe-view show examples/qvf_containers/runs/h2-rhf.qvf --info
vibe-view show examples/qvf_containers/runs/h2-rhf.qvf --plain
```

Or use the graphical and interactive-terminal viewers:

```sh
vibe-view open examples/qvf_containers/runs/h2-rhf.qvf
vibe-view tui examples/qvf_containers/runs/h2-rhf.qvf
```

Generated `.qvf`, `.out`, `.system`, and other run artifacts are ignored by
Git. Remove `examples/qvf_containers/runs/` whenever you want a clean slate.

The complete walkthrough is
[Running a calculation as a QVF container](../../docs/tutorial/qvf_job_containers.md).
