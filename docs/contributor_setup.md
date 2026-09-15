# Contributor setup, fresh-clone bootstrap and rebuild discipline

End-user installation lives at [`installation.md`](installation.md);
the two-command path there is also the right path for contributors.
This page covers the contributor-specific bits: per-clone venvs,
the editable-install rebuild loop, and the silently-stale-`.so`
failure mode the binding-sanity test now catches.

## Two-command bootstrap

After installing the system prerequisites listed in
[`installation.md` § Requirements](installation.md#requirements) for
your platform (Homebrew on macOS; `apt` / `dnf` / `pacman` package
groups on Linux), a fresh clone reaches a working `import vibeqc`
in two commands:

```sh
git clone https://github.com/vibe-qc/vibe-qc.git && cd vibe-qc
./scripts/install.sh --dev --extras test --venv .venv \
    --python /opt/homebrew/bin/python3.14
```

See [installation](installation.md#request-repository-access) for the publication
access boundary and keep each project in its own checkout.

`install.sh` drives, in order: preflight (verify build prerequisites
on `$PATH`) → vendored native deps (libint, libxc, spglib, FFTW3,
libecpint, basis library, idempotent, each per-dep script
short-circuits when its `third_party/<dep>/install/` tree is
present) → `.venv` creation → editable `pip install -e .[<extras>]`
→ banner. Total wall-clock on a fresh box: ~15-40 min for native
deps, ~30 s for the editable install.

The command above is the tested macOS development bootstrap. On another
platform, select your installed Python with `--python`.

The flag-free installer selects the newest stable tag advertised by origin;
`--dev` follows the public `main` snapshot stream. For a pinned
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
