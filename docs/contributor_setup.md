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

## Rebuild Libxc after its recipe changes

The vendored Libxc recipe patches the PBE and r2SCAN correlation kernels so
they evaluate 1 ± zeta without cancellation at full spin polarization
(`scripts/patch_libxc_spin_fractions.py`). The editable install never re-runs
the dependency scripts, so a clone whose `third_party/libxc/install/` predates
the patch keeps the old kernels after a `git pull` and a core rebuild. The test
session then refuses with `STALE LIBXC` and exit status 97, the same refusal
as a stale core. Rebuild Libxc with:

```sh
scripts/build_libxc.sh
```

It rebuilds an unpatched install and admits a patched one unchanged. The core
loads Libxc at run time, so it needs no rebuild afterwards. A Libxc outside
the checkout, such as a system installation, is not checked.

## Libint capabilities required during configuration

The vendored Libint recipe applies header fixes for the Boys-table endpoint
and for scratch allocation in the short-range Coulomb evaluator. The latter
keeps the evaluator's object layout and arithmetic, uses local scratch through
order 31, and retains dynamic storage at higher orders. Run the normal dependency
update and rebuild vibe-qc after a recipe change: updating a header does not
change an already compiled native extension. System Libint installations are
not patched by this recipe.

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

## Vendored headers come first, and a stale libecpint is refused

The compiler searches the vendored `third_party/<dep>/install` include
directories before every other dependency's. Imported packages otherwise
contribute include directories in link order, and on macOS libint's installed
package links Boost, whose include directory is the whole Homebrew prefix. A
Homebrew libecpint, libxc, spglib or FFTW there used to shadow the vendored
header of the same name (#461): with Homebrew's libecpint installed, the
build stopped in `cpp/src/ecp.cpp` with "no member named
'shell_pair_negligible'". Nothing needs to be uninstalled.

A header can still resolve to another copy through a `CPATH` entry or an
`-I` in `CXXFLAGS` or `CMAKE_CXX_FLAGS`, which the compiler searches before
every system directory unless it repeats one. The build then fails with a
message that names the header: `cpp/src/xc.cpp` and `cpp/src/basis.cpp`
compare libxc's and libint's version macros with the packages CMake selected,
and `cpp/src/ecp.cpp` checks for the vendored libecpint's patched API, since
that copy keeps upstream's version number. Remove the directory from those
variables and rebuild. spglib and FFTW have no version macro and rely on the
search order alone.

`scripts/build_libecpint.sh` records the version of its libecpint patch in
`third_party/libecpint/install/.vibeqc-radial-quadrature`. The editable
rebuild above does not rerun that script, and a patch that changes only
libecpint's sources would otherwise be linked from an older install without
notice (#460). Configuration therefore compares the record with the script's
`RADIAL_PATCH_VERSION`. A mismatch is an error under
`VIBEQC_REQUIRE_VENDORED` and a warning otherwise. Run
`scripts/build_libecpint.sh`, which rebuilds only libecpint in a few minutes,
then the editable rebuild again.

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

## Queue cooperation during builds

The install and update scripts probe the local `vq` daemon before heavy
compilation. When it is available, they pause only local running jobs with a
`vibeqc-build-<pid>` tag and resume only that tag at exit. Existing operator
pauses remain in place. Nested dependency builds share the outer build's pause;
the outer build owns its cleanup. An empty queue is a successful no-op.

Set `VIBEQC_VQ_COOPERATE=0` to skip this coordination, or `VQ_BIN` to choose the
queue executable. A missing local daemon is reported as a skip. Probe failures,
command failures and incomplete/error summaries produce a warning; check the
local jobs carrying the build tag if resume reports a problem.

### FFTW, libxc, spglib and OpenBLAS cache identity

FFTW, libxc, spglib and OpenBLAS cache reuse check that the version entry point
comes from the requested library, using its resolved path and SHA-256 digest. An unloadable or substituted
library stops setup and ordinary updates before a stamp refresh or automatic
rebuild. Inspect the cache before explicitly selecting a rebuild with
`scripts/update_native_deps.sh --dep fftw`, `--dep libxc`, `--dep spglib`, `--dep openblas`, or
`--all`. Selecting one dependency does not authorize replacing another invalid cache. The dry-run forms only
preview that selection.

The reported exported version is distinct from the recipe source pin. Upstream
FFTW 3.3.10's CMake build exports 3.3.9; the string alone does not identify the
source release or build options. Cache admission reports those as not attested.
Existing qualified recipe stamps remain recipe records; legacy unknown source
provenance is not upgraded into a source/build attestation.

For libxc, the loaded image owning `xc_version` is checked before the version
function is called. Its reported version is also an observation, not source or
build-option attestation. Other dependency gates retain their existing checks.

For spglib, the image owning `spg_get_version` is bound before calling it.
Its exported version is kept separate from the source recipe pin; successful
artifact admission does not establish source/build-option provenance.

OpenBLAS also requires `lapacke.h` and binds the BLAS, LAPACK and LAPACKE
exports to the requested image before calling `openblas_get_config`. Header-only
or unloadable installs cannot satisfy cache admission. The current consumer uses
LP64 indices, so an exported `USE64BITINT` configuration is refused. Runtime
version and configuration (including threading) are reported separately from the
recipe pin; they do not attest the source release or build options. Optional
OpenBLAS remains opt-in, and an explicit `REBUILD=1` builder request or
`update_native_deps.sh --dep openblas` can replace a refused cache.


Libint cache reuse requires the C shared library that `Libint2::cxx` links
through its header-only C++ interface. The admission probe binds
`libint2_static_init`, `libint2_cleanup_eri` and the storage address of
`libint2_build_eri` to that requested image without calling initialization or
integral kernels. A config/marker-only, static-only, ambiguous or unloadable
cache stops before installed-header patches. The optional compiled C++ library
is not a substitute for the required C image. Runtime version and generated
max-AM are not attested; marker values remain recorded settings. Proven valid
caches retain header patches and max-AM rebuild decisions. Fetch-only staging
bypasses the installed runtime, and `update_native_deps.sh --dep libint` remains
the explicit recovery route for a refused cache.

## Optional native build receipts

Fresh FFTW, libxc and spglib CMake builds can call an external provenance
collector. Set `VIBEQC_NATIVE_BUILD_COLLECTOR` to its absolute executable path
and `VIBEQC_NATIVE_RECEIPT_DIR` to an existing absolute directory outside the
product checkout and install tree. Each build uses `<dependency>.json` there;
choose a fresh directory to preserve earlier receipts. No collector is bundled.
With both variables unset, the existing build commands are unchanged. Cache hits
and `VIBEQC_FETCH_ONLY=1` do not call the collector or create historical receipts.

The collector is invoked once, after configuration and before build/install:

```text
COLLECTOR record --source NAME=SOURCE --recipe BUILDER --recipe HELPER ...
  --cmake-build BUILD --install INSTALL
  --artifact-manifest BUILD/install_manifest.txt
  --cwd PRODUCT --output RECEIPT_DIRECTORY/NAME.json
  -- cmake --build BUILD --target install
```

Arguments are passed directly, without shell evaluation. Configuration enables
`CMAKE_EXPORT_COMPILE_COMMANDS=ON`. The recipe arguments include the builder,
collector adapter, build-pressure helper, source verification helper and build
lock helper. The install manifest produced by the actual install identifies the
artifacts, including platform-specific names and `lib` or `lib64` locations.
The external collector must observe the configured build, preserve source,
compiler and option identities, and verify the resulting install. Its nonzero
exit stops the builder without retry: the build may already have run.

These receipts do not themselves prove which image a process loaded or whether
its data files work from a copied prefix. Those checks remain separate. This
hook does not cover libint's generated-library build, OpenBLAS Make, or the
libecpint multistage build. Existing installs without a qualified build-time
receipt retain unknown source and build provenance.


## Generated basis overlay integrity

Setup requires Python >=3.11. A nonempty `PYTHON_BIN` explicitly selects the
interpreter; otherwise setup tries a usable project `.venv/bin/python`, then
`$VIRTUAL_ENV/bin/python`, then `python3` on `PATH`. An incompatible explicit
override fails without falling back. Interpreter refusal occurs before any
output is changed. To select a supported interpreter, use
`PYTHON_BIN=/path/to/python bash scripts/setup_basis_library.sh`.

After changing or copying a generated `build/basis_library` directory, run
`bash scripts/setup_basis_library.sh`. Setup records the actual content of every
committed orbital basis and ECP sidecar in a readable receipt. Imports check all
recorded file identities and content digests without writing to the installation.
If shared-filesystem clients assign different device numbers to the same files,
the guard re-reads their content and verifies the recorded SHA256 values; all
other identity fields must still match. This read-only fallback does not reseal
the receipt. Its cost scales with the catalog size on each importing process;
the ordinary same-client path uses the prepared receipt without rehashing files.
Missing, corrupt, unreadable or changed receipts refuse overlay selection;
ordinary imports do not regenerate them. Explicit `LIBINT_DATA_PATH` overrides
retain their existing behavior and remain the caller's responsibility.

Library callers receive `BasisOverlayIntegrityError`, a `RuntimeError` subclass.
The `vibeqc` and `vibe-qc` commands, and their supported `python -m` entry points,
use exit status 78 for this integrity refusal, before native integral evaluation.
This is a filesystem freshness check for ordinary writes and replacements, not
a security boundary against forged receipts or privileged metadata changes.
Filesystems without usable POSIX change times refuse cached selection.
