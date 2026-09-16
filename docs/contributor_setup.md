# Contributor setup, fresh-clone bootstrap and rebuild discipline

End-user installation lives at [`installation.md`](installation.md);
the clone-and-bootstrap path there also applies to contributors.
This page covers the contributor-specific bits: per-clone venvs,
the editable-install rebuild loop, and the silently-stale-`.so`
failure mode the binding-sanity test now catches.

(two-command-bootstrap)=
## Clone and bootstrap

After installing the system prerequisites listed in
[`installation.md` § Requirements](installation.md#requirements) for
your platform (Homebrew on macOS; `apt` / `dnf` / `pacman` package
groups on Linux), clone the source and bootstrap the native/Python environment:

Clone the public source over HTTPS; no account or deploy key is required:

```sh
git clone https://github.com/vibe-qc/vibe-qc.git
```

Authorized maintainers use their separately configured upstream remote.
Private access recipes belong in the external operations documentation.

Then enter the new checkout and install:

```sh
cd vibe-qc
./scripts/install.sh --dev --extras test --venv .venv \
    --python /opt/homebrew/bin/python3.14
```

The [repository directory](installation.md#repositories-and-downloads) lists
public GitHub source snapshots for all four projects; they can be cloned
anonymously. Canonical maintainer development access is configured separately. Public snapshots have
separate commit history, so keep contributions based on the selected host and
project. See [CONTRIBUTING.md](https://github.com/vibe-qc/vibe-qc/blob/main/CONTRIBUTING.md)
for the GitLab merge-request path for GitHub contributions.

`install.sh` drives, in order: preflight (verify build prerequisites
on `$PATH`) → vendored native deps (libint, libxc, spglib, FFTW3,
libecpint, basis library, idempotent, each per-dep script
short-circuits when its `third_party/<dep>/install/` tree is
present) → `.venv` creation → editable `pip install -e .[<extras>]`
→ banner. Total wall-clock on a fresh box: ~15-40 min for native
deps, ~30 s for the editable install.

The command above is the tested macOS development bootstrap. On another
platform, select your installed Python with `--python`.

The `release` branch advances only from a tagged core release;
`--dev` follows `main`, which is where contributions land. For a pinned
install, choose a tag actually present in this repository; pre-split branches
and historical tags were not transferred. Viewer and queue contributors clone their respective
repositories separately. `vibe-basis/` remains here.

If you already have the prereqs installed and only need a different
branch / venv / extras combination, `install.sh --help` enumerates
the flags.

## Per-clone venvs

Every independent checkout needs its own `.venv` at the checkout
root, and a linked `git worktree` is not a substitute for a checkout when
you build or measure. The editable
`pip install -e .` pins the C++ extension to that checkout's own
`third_party/<dep>/install/` trees and to its `cpp/` source, 
sharing a venv across checkouts loads the wrong native libraries.

Run `install.sh` once per fresh clone; it lands the venv at
`<checkout-root>/.venv` by default. To keep multiple branches in
one tree with side-by-side venvs, use `--venv .venv-<name>`.

## Rebuild after upstream binding changes

The editable install rebuilds the C++ extension only when you ask
it to. So when upstream lands a commit that adds a new pybind11
binding, e.g. `39bf8629` adding `compute_nuclear_with_charges` to
both `cpp/src/bindings.cpp` and the `from ._vibeqc_core import (...)`
block in `python/vibeqc/__init__.py`, your locally-built
`_vibeqc_core.so` is silently stale until you rebuild. The next
`import vibeqc` then fails partway through the module body:

```
ImportError: cannot import name 'compute_nuclear_with_charges'
from 'vibeqc._vibeqc_core'
```

The recovery is one command:

```sh
.venv/bin/pip install -e . --no-build-isolation
```

`--no-build-isolation` keeps `pip` from spinning up a fresh build
environment, which would re-download every build dep and discard
the incremental CMake cache. Without it the rebuild takes minutes
instead of seconds.

[`tests/test_binding_sanity.py`](../tests/test_binding_sanity.py)
catches this mechanically, it walks every `from ._vibeqc_core
import X` site in the package and asserts each symbol resolves
against the loaded `.so`. If it fails, the assertion message tells
you exactly which rebuild to run and which files referenced the
missing symbol. **Run this test first** when `import vibeqc` starts
failing after a `git pull`:

```sh
.venv/bin/python -m pytest tests/test_binding_sanity.py
```

## Libint capabilities required during configuration

Libint builds with the same version can contain different generated kernels.
The selected `Libint2::cxx` target must expose ordinary one-body integrals
(`LIBINT2_SUPPORT_ONEBODY=1`) and their geometric derivatives through at
least order 1 (`LIBINT2_DERIV_ONEBODY_ORDER>=1`). CMake compiles and links a
small witness against that target's headers and library before building
the core. Missing capability macros or first-derivative overlap, kinetic
or electrostatic symbols cause a configuration error with compiler/linker
details and the selected `Libint2_DIR`. The witness is not executed.

If configuration rejects a system installation, run
`scripts/build_libint.sh` and set `Libint2_DIR` to the CMake package directory
in its `third_party/libint/install/` tree when reconfiguring. A cached
`Libint2_DIR` can keep selecting an older system dependency despite a newly
created local install. The capability check runs on every configure, including
after replacement at the same path. Setting `VIBEQC_REQUIRE_VENDORED=OFF`
permits system discovery but does not bypass the capability requirement.
The probe log is `cpp/CMakeFiles/vibeqc-libint-onebody.log` beneath the build
directory.

The standard build recipe retains its existing order-2 and property-derivative
settings. This minimum check for ordinary first derivatives does not certify
Hessians, property derivatives, every angular momentum or numerical accuracy.
Continue to verify the built core and the library actually loaded at runtime
after a rebuild; configuration cannot protect against a later library swap.

## Native sources from a local mirror, and clones that cannot hang

`scripts/build_<dep>.sh` fetches libint, libxc, spglib, OpenBLAS and
libecpint's sources with `git clone`, pinned to a commit SHA that the
build verifies. Three environment variables control that step; all of
them are read by `scripts/_verify_source.sh` and none changes what is
built.

`VIBEQC_GIT_REFERENCE_DIR` names one or more directories (colon
separated) that may already hold the pinned source. Point it at another
checkout's `third_party/` directory, or at a directory of mirrors named
after the upstream repositories (`libint`, `libint.git`, `libint/src` and
`libint-src` are all recognised). A local repository is used only when it
already contains the pinned commit, and the result still passes the same
HEAD check as a network clone, so a mirror can never substitute a
different commit. This turns a GitHub outage, or a transfer that keeps
resetting, into a non-event on a machine that has built vibe-qc before:

```sh
VIBEQC_GIT_REFERENCE_DIR=/path/to/other-checkout/third_party ./scripts/update.sh
```

`VIBEQC_GIT_CLONE_ATTEMPTS` (default 3) bounds how often a network clone
is retried, and `VIBEQC_GIT_CLONE_TIMEOUT` (default 1800, seconds per
attempt) bounds how long one attempt may run. A clone past its deadline
receives SIGTERM and, ten seconds later, SIGKILL for its whole process
tree; a stalled HTTP transfer is cut earlier by git's own low-speed
limit (1 KiB/s over 60 s unless `GIT_HTTP_LOW_SPEED_LIMIT` and
`GIT_HTTP_LOW_SPEED_TIME` are set). The failure message names the
override to reach for. Ctrl-C on the build script still stops the clone.

## When the editable install fails on `pybind11`

If `pip install -e .` errors out at the CMake configure step with

```
Could not find a package configuration file provided by "pybind11"
```

your venv is missing the pybind11 Python package (which ships the
`pybind11Config.cmake` file CMake reads via
`pybind11::get_cmake_dir()`). On a fresh `python3 -m venv`-created
environment this can happen if `pip install -e .` skips the
build-system isolation that would normally pull pybind11 from
`pyproject.toml`'s `build-system.requires`. Fix:

```sh
.venv/bin/pip install pybind11
.venv/bin/pip install -e . --no-build-isolation
```

`install.sh` handles this automatically because it installs the
test extras explicitly before the editable install. Manual paths
(see [`installation.md` § Manual bootstrap](installation.md#advanced-manual-bootstrap))
should always run the editable install **without**
`--no-build-isolation` the first time so pip pulls the build
prereqs; only switch to `--no-build-isolation` for the
incremental-rebuild loop above.

## Pre-commit hook

[CONTRIBUTING: pre-commit hook](contributing.md#pre-commit-hook-one-time-setup)
covers activation. The hook enforces the current
[personal-information policy](contributing.md#personal-information): author
home paths, private LAN IPs and the employer string are blocked at commit
time. The [prose policy](contributing.md#documentation-prose) also rejects
em and en dashes outside its documented exemptions. Never bypass hooks with
`--no-verify`; follow the
[branch and checkout policy](contributing.md#branch-and-checkout-policy).

For website and companion documentation work, follow the
[site publishing and CSS export contract](site_publishing.md).

## See also

* [`installation.md`](installation.md), full per-platform recipe,
  build prerequisites, BLAS-backend selection, verifying the
  install with the banner.
* [`../CONTRIBUTING.md`](contributing.md), where to report
  what, MR / patch flow, pre-commit hook activation.
* [Current contributor policies](contributing.md#branch-and-checkout-policy),
  branch and checkout rules, privacy hygiene and documentation prose.
* [`release_process.md`](release_process.md), branch model in
  depth, release procedure, Patch-candidate trailer.
