---
myst:
  html_meta:
    "description": "Install and maintain vibe-qc, the separate vibe-view and vibe-queue projects, and the retained vibe-basis toolkit."
    "og:title": "Install and maintain the vibe toolset"
---

# Install and maintain the vibe toolset

Since 2026-09-08, the engine, viewer, scheduler, and QVF format have separate
repositories. Clone only what you need. Each installed tool has its own
virtual environment and can be updated independently. **vibe-basis remains
inside vibe-qc** at `vibe-basis/`.

## At a glance

The engine row selects development `main`; its flag-free installer selects
the published `release` branch. Companion installers follow their own
repository policies. Run a command from the **working directory** named in its row. The viewer
and queue rows refer to independent checkouts, never to core subdirectories.

| Project | Working directory | Install | Verify |
| --- | --- | --- | --- |
| vibe-qc | vibe-qc checkout | `./scripts/install.sh --dev` | `.venv/bin/python -c "from vibeqc import print_banner; print_banner()"` |
| vibe-view | vibe-view checkout | `./scripts/install.sh` | `.venv/bin/vibe-view doctor` |
| vibe-queue (`vq`) | vibe-queue checkout | `./scripts/install.sh` | `.venv/bin/vq --version` |
| vibe-basis | vibe-qc checkout | `./vibe-basis/scripts/install.sh` | `vibe-basis/.venv/bin/vb --version` |
| QVF | no install required by vibe-qc | [Published specification and corpus](qvf/index.md) | independent format conformance |

vibe-qc and vibe-basis versions in this tree are {{release}} and
{{vibebasis_version}}. Consult [vibe-view releases](https://github.com/vibe-qc/vibe-view/releases)
and [vibe-queue releases](https://github.com/vibe-qc/vibe-queue/releases)
for companion versions. Every repository tags independently; a core tag does
not release the viewer, queue, or format toolkit.

For release-specific companion details, use the published
[vibe-view installation and lifecycle guide](https://vibe-qc.com/vibe-view/docs/installation.html)
and [vq daemon lifecycle guide](https://vibe-qc.com/vibe-queue/docs/lifecycle.html).
The [vq operator guide](https://vibe-qc.com/vibe-queue/docs/operator/index.html)
separates host setup and service supervision from ordinary job submission.

Only vibe-qc builds native chemistry libraries. The other tools can operate
without the engine. QVF's Apache-2.0 repository owns the specification, JSON
schema, registry, conformance corpus, and reference implementations.
vibe-qc implements the format independently and is validated against that
contract. It does not link the reference toolkit. Agreement between three
independent implementations through the corpus is the interoperability check.

## Before you start

Each split repository has public GitHub source snapshots that can be cloned
anonymously. [Repositories and downloads](installation.md#repositories-and-downloads)
lists the source locations and published refs. Maintainer upstream access is
configured separately outside product checkouts.
A possible layout is:

```text
workspace/
  vibe-qc/                engine checkout
    .venv/                engine environment
    vibe-basis/           retained toolkit
      .venv/              basis-toolkit environment
  vibe-view/              independent viewer checkout
    .venv/
  vibe-queue/             independent queue checkout
    .venv/
  qvf/                    optional specification/reference checkout
```

These adjacent locations are a convenience, not a runtime requirement. QVF
files can be copied between machines. Run installers as your regular user;
only operating-system package setup may need `sudo`.

## Install each component

### vibe-qc

Install the [native prerequisites](installation.md), then:

```sh
git clone https://github.com/vibe-qc/vibe-qc.git
cd vibe-qc
./scripts/install.sh --dev --extras test --venv .venv \
    --python /opt/homebrew/bin/python3.14
source .venv/bin/activate
python -c "from vibeqc import print_banner; print_banner()"
```

This is the tested macOS command. On Linux, select your installed supported
Python instead. The repository published the `v0.16.0` tag and the `release`
branch on 2026-09-08, so the flag-free installer -- which targets `release`
-- is appropriate; `--dev` selects `main`. Do not assume historical monorepo
tags exist in the new repository: its history starts at the split.

### vibe-view

From the directory in which you want the independent viewer checkout:

```sh
git clone https://github.com/vibe-qc/vibe-view.git
cd vibe-view
./scripts/install.sh
source .venv/bin/activate
vibe-view doctor
vibe-view demo --open
```

The `modes` profile provides browser, desktop-server, and TUI support.
Use `--extras core`, `--extras viewer`, or `--extras tui` for a smaller
installation. Add `--with-electron` to fetch the desktop runtime during
installation. See [viewer setup](tutorial/vibe_view_getting_started.md) for
platform requirements and the detailed lifecycle.

### vibe-queue

From the directory in which you want the independent queue checkout:

```sh
git clone https://github.com/vibe-qc/vibe-queue.git
cd vibe-queue
./scripts/install.sh
.venv/bin/vq --version
```

The distribution, Python import package, and CLI are still `vq`. The default
profile installs the CLI and daemon; `--extras web` adds the dashboard.
Install on both client and execution hosts as described in the
[queue guide](user_guide/queue.md). Python 3.12 or newer is required.

### vibe-basis

From the **vibe-qc** checkout:

```sh
./vibe-basis/scripts/install.sh
vibe-basis/.venv/bin/vb --version
```

The default profile supports basis optimization against external programs.
Use [basis toolkit](user_guide/basis_toolkit.md) for integration details.
The former co-located `--with-vq` recipe is not a queue installation route
after the split; install the separate queue first.

## Common maintenance workflows

Run lifecycle commands from the checkout that owns the installed tool:

| Tool | Update | Repair current source | Preview removal |
| --- | --- | --- | --- |
| vibe-qc | `./scripts/update.sh --dev` | `./scripts/reinstall.sh` | `./scripts/uninstall.sh --dry-run` |
| vibe-view | `./scripts/update.sh` | `./scripts/reinstall.sh` | `./scripts/uninstall.sh --dry-run` |
| inactive source vq | `./scripts/update.sh --skip-git` | `./scripts/reinstall.sh` | `./scripts/uninstall.sh --dry-run` |
| vibe-basis (from core root) | `./vibe-basis/scripts/update.sh` | `./vibe-basis/scripts/reinstall.sh` | `./vibe-basis/scripts/uninstall.sh --dry-run` |

For a **serving vq daemon**, use `vq self-update` or `vq admin update`
according to the queue's own release-report workflow. Direct lifecycle
scripts refuse serving environments. Queue state, leases, workspaces, and
job history must survive an update or uninstall.

Quit source-backed viewer windows before running
`./scripts/update-desktop.sh` in the viewer checkout. Packaged desktop apps
follow the viewer project's update-feed instructions. A core update changes
neither companion repository nor its installed environment.

Run `--help` and `--dry-run` to inspect supported profiles, paths, ownership,
and replacement options before a repair. Keep environments checkout-specific;
an editable install records its source path. Do not copy a `.venv` from the
archived monorepo. A failed ownership check is a reason to inspect the target,
not to remove its marker. Replacements retain a rollback environment until
verification succeeds.

## Release artifacts and format documentation

The wheel and qvf-writer archive under `docs/_static/downloads/` are legacy
artifacts from other projects. They are not evidence of a current companion
release. Their owners must publish validated artifacts on their own release
pages before installation instructions can advertise them as current.

The six QVF pages that include `docs/qvf/_vendored/` are pinned copies from
the [QVF repository](https://github.com/vibe-qc/qvf). Do not edit the
vendored files. Make a format change upstream, re-vendor deliberately with
tag/commit provenance, and review the `QVF_TAG` bump in `.gitlab-ci.yml`.

## Related guides

- [Installation](installation.md): native prerequisites and detailed engine setup.
- [Updating vibe-qc](updating.md): migration, native rebuilds, and recovery.
- [Viewer setup](tutorial/vibe_view_getting_started.md): viewer profiles and desktop.
- [Queue guide](user_guide/queue.md): services, configuration, and operations.
- [Release process](release_process.md): independent tags and local validation.
