# Installation

vibe-qc is a Python package with a C++ core. The C++ core links against
[libint](https://github.com/evaleev/libint) (Gaussian integrals),
[libxc](https://libxc.gitlab.io/) (XC functionals),
[spglib](https://spglib.readthedocs.io/) (crystal symmetry),
[FFTW3](https://www.fftw.org/) (Ewald long-range Hartree),
[libecpint](https://github.com/robashaw/libecpint) (effective-core
potentials), plus a pinned snapshot of pugixml + libcerf inside libecpint's
tree, and an **optimised BLAS + LAPACK** (Apple Accelerate on macOS,
OpenBLAS / MKL on Linux) that Eigen delegates dense linear algebra to, 
see [BLAS backend](#blas-backend) below.

## Choose the installation you need

| Goal | Recommended route | Native compiler build? |
|---|---|---:|
| Current tagged release | clone, then `./scripts/install.sh` (release branch or newest stable tag) | yes |
| Develop or test current `main` | `./scripts/install.sh --dev` | yes |
| Reproduce a published calculation | `./scripts/install.sh --branch vX.Y.Z` | yes |
| Read QVF files only | [install vibe-view alone](tutorial/vibe_view_getting_started.md) | no |
| Calculate and view on one machine | install vibe-qc, then follow [Install both](getting_started.md#install-both) | vibe-qc only |

If this is your first visit, [Start here](getting_started.md) gives the
shortest end-to-end route, including verification and a first calculation.
This page is the detailed engine-installation reference: platform packages,
manual build steps, BLAS selection, clusters, updates, and failure recovery.

```{note}
**vibe-qc is not on PyPI yet.** There is no `pip install vibe-qc`. The
supported install path is: clone the repo, install your platform's build
prerequisites, then run `./scripts/install.sh --dev`. It builds or verifies the
vendored libraries, creates the managed virtualenv, installs the package, and
checks the live extension. The advanced manual recipe remains below for custom
provisioning.
```

## Request repository access

All four GitHub repositories are public and can be cloned without an account
or key. Maintainer access to canonical development and deployment systems is
configured separately, using private operations documentation and secret stores.
Do not add those access settings to a product checkout.

## Clone the source

Clone the public source over HTTPS; no account or deploy key is required:

```sh
git clone https://github.com/vibe-qc/vibe-qc.git
```

Authorized maintainers use their separately configured upstream remote.
Private access recipes belong in the external operations documentation.

Then enter the new checkout and install:

```sh
cd vibe-qc
./scripts/install.sh --dev
```

`--dev` selects `main` on either host. For a stable version, choose an origin
that actually carries the desired release ref; the GitLab and mirror refs can
differ. The platform prerequisites and full installer behavior are below.

## Repositories and downloads

Clone each product into its own directory. The Python import remains
`vibeqc`, but the core repository and checkout are named **vibe-qc**.
The viewer and queue are separate projects; **vibe-basis stays in vibe-qc**.

| Project | Checkout | Public source and tags |
| --- | --- | --- |
| vibe-qc engine | `vibe-qc/` | [Source](https://github.com/vibe-qc/vibe-qc), [tags](https://github.com/vibe-qc/vibe-qc/tags) |
| vibe-view | `vibe-view/` | [Source](https://github.com/vibe-qc/vibe-view), [tags](https://github.com/vibe-qc/vibe-view/tags) |
| vibe-queue (`vq`) | `vibe-queue/` | [Source](https://github.com/vibe-qc/vibe-queue), [tags](https://github.com/vibe-qc/vibe-queue/tags) |
| QVF reference | `qvf/` | [Source](https://github.com/vibe-qc/qvf), [tags](https://github.com/vibe-qc/qvf/tags) |

All four GitHub mirrors provide public source snapshots and can be cloned
anonymously. GitLab remains the development source of truth. Public snapshots
have their own commit history and can lag GitLab; check that the desired ref
exists on the host you use. Record the host, ref and resolved commit when
reproducing a calculation; a GitLab SHA does not identify the corresponding
GitHub snapshot. See [GitHub source publication](github_publication.md) for
the publication policy.

For source downloads, choose a tag in the owning project's **tags** page.
Packaged artifacts, when published, belong to that project's release page:
[vibe-qc](https://github.com/vibe-qc/vibe-qc/releases),
[vibe-view](https://github.com/vibe-qc/vibe-view/releases),
[vibe-queue](https://github.com/vibe-qc/vibe-queue/releases), or
[QVF](https://github.com/vibe-qc/qvf/releases).
A tag does not imply that a wheel, desktop installer or container is available.
The managed installers and updaters require a Git checkout; unpacking a source
archive is not equivalent to cloning. See the [manual build](#manual-setup)
for archive-based provisioning.

Historical companion build artifacts are retained only in the private archive.
Use the owning companion repository for current release downloads. QVF is a specification and reference implementation, not a runtime
or installation dependency of vibe-qc.

## What a successful install does

After [cloning and entering the checkout](#clone-the-source), the recommended
command performs the full bootstrap:

```sh
./scripts/install.sh --dev
```

It performs these steps in order:

1. selects the requested release, branch, or tag;
2. checks the compiler, CMake, Ninja, headers, OpenMP, and BLAS;
3. fetches and verifies the pinned native dependency sources;
4. installs native libraries below `third_party/` in this checkout;
5. creates a checkout-local `.venv`;
6. builds the editable Python/C++ package;
7. imports vibe-qc and prints the linked-library banner.

The first clean build normally takes 15 to 40 minutes and uses several
gigabytes of build space. Re-running the installer is much faster when the
native dependency stamps are current.

After it finishes, verify the exact interpreter and package path:

```sh
.venv/bin/python -c "from vibeqc import print_banner; print_banner()"
.venv/bin/python -c "import vibeqc; print(vibeqc.__file__)"
```

The second command should resolve inside the checkout you just installed.
If it points into another worktree, recreate the environment here instead of
sharing that other checkout's `.venv`.

```{seealso}
Once installation is done, [good practices](good_practices.md)
covers the working conventions for actually using vibe-qc --
where to put your calculations (hint: not inside the source tree),
naming, reproducibility, performance hygiene, and what to try when
SCF diverges. Worth a five-minute read before your first
production run.
```

**The five pinned core libraries are vendored.**
`scripts/setup_native_deps.sh` fetches libint, libxc, spglib, FFTW, and
libecpint from upstream, configures them the way vibe-qc needs, builds them,
and installs them into `third_party/<name>/install/`. Eigen, Boost, GMP, and
the BLAS implementation are supplied by the platform or an opt-in route, so
record the banner when exact build provenance matters.

Each fetched tarball / git tag is verified against a pinned SHA-256 /
commit SHA before use (see the per-script comment blocks in
`scripts/build_*.sh` for the resolved values and the re-resolve recipe),
so a moved upstream tag or a network-level tamper fails loudly rather
than silently changing what vibe-qc links against.

If a `third_party/<dep>/install/` is missing, the top-level CMake
configure silently falls back to system discovery for that one
dependency. That is convenient for development, but it weakens the
vendored-ABI guarantee. CI and release builds that need the guarantee
should set `VIBEQC_REQUIRE_VENDORED` (either
`-DVIBEQC_REQUIRE_VENDORED=ON` at configure time or the matching env
var), which turns the fallback into a hard error for the five always-
vendored deps (libint, libxc, spglib, fftw, libecpint). OpenBLAS is
opt-in (`WITH_OPENBLAS=1`) and stays excluded from the strict check.

```{admonition} Only want the viewer?
:class: tip

None of the build requirements below apply if all you need is
**[vibe-view](visualization.md)**, the viewer for `.qvf` archives. It is a
wheel-based Python install: no local C++ compiler, no CMake, no vendored native
dependencies, and it does not need vibe-qc installed at all. One checkout
installer or one hosted-wheel install is enough; vibeview is not on PyPI yet.

Follow
[Getting started with vibe-view alone](tutorial/vibe_view_getting_started.md)
instead, which covers Linux and macOS.
```

## Choose the component you need

The projects share a file format and can be adopted separately:

| Project | What it provides | Source and setup |
| --- | --- | --- |
| vibe-qc | Molecular and periodic calculations | this repository; root `scripts/install.sh --dev` |
| vibe-view | QVF, structure, browser, desktop, and terminal viewing | [separate repository](https://github.com/vibe-qc/vibe-view); [viewer setup](tutorial/vibe_view_getting_started.md) |
| vibe-queue | Reusable scheduling; CLI and import package still `vq` | [separate repository](https://github.com/vibe-qc/vibe-queue); [queue setup](user_guide/queue.md) |
| vibe-basis | External-code basis optimization | stays in this repository at `vibe-basis/` |
| QVF | Specification, schema, registry, corpus, reference implementations | [format repository](https://github.com/vibe-qc/qvf), Apache-2.0; no runtime dependency |

Each source installer operates in its own checkout and creates its own
virtual environment. Only vibe-qc builds the native chemistry libraries.
vibe-qc implements QVF independently and is validated against the published
specification and conformance corpus; it does not link the QVF reference
toolkit. See [Install and maintain the toolset](toolset_lifecycle.md) for
checkout locations and lifecycle commands.

The flag-free installer targets `release`, the stable core snapshot advanced
from a tag on `main`; `--dev` selects ongoing development on `main`.
Companion installation requirements are maintained in the
[vibe-view manual](https://vibe-qc.com/vibe-view/docs/installation.html) and
[vq lifecycle guide](https://vibe-qc.com/vibe-queue/docs/lifecycle.html#source-install-lifecycle).

## Requirements

System tools the build needs:

- **Python ≥ 3.11** (3.11-3.14 all work; 3.14 is what the macOS
  install recipe below uses). On macOS, Apple's stock
  `/usr/bin/python3` is currently 3.9 and is **too old**, install
  Homebrew's `python@3.14` and make sure it wins on `PATH` (see the
  [macOS section](#macos-homebrew) below).
- **C++17 compiler**, AppleClang 13+, GCC 10+, Clang 13+
- **CMake ≥ 3.20** and **Ninja**
- **git**, **make**, **curl**, **tar**, **pkg-config**
- **OpenMP**, comes with GCC / Clang on Linux; on macOS it's
  `libomp` from Homebrew (AppleClang ships without OpenMP support
  and several vendored deps hard-require it).
- **An optimised BLAS + LAPACK**, Apple Accelerate ships with
  macOS; on Linux install OpenBLAS or MKL (system package or
  vendor it via `WITH_OPENBLAS=1`, see [BLAS](#blas-backend)).
  Reference netlib BLAS is functional but slow and not recommended.

Plus three small system libraries that **libint's code generator** uses
during its own build (we deliberately don't vendor these, GMP has
arch-specific assembly, and Boost / Eigen are header-only and on every
distro):

- **Eigen** (3.4+ or 5.0+), header-only. **Both major versions
  work**; libint 2.13.1's bundled CMake module handles either
  header layout. As of mid-2026, Homebrew's `eigen` formula and
  Arch's `eigen` package are both 5.0.x, that's the version vibe-qc
  builds against on macbook + compute-reference today. Debian/Ubuntu still
  ship `libeigen3-dev` 3.4.x, which also works fine.
- **Boost**, header-only
- **GMP** with C++ bindings (gmpxx)

`scripts/setup_native_deps.sh` runs a **preflight check** that verifies
every one of the above is present before it starts building anything,
and prints the exact per-distro install command for whatever is
missing. If you've never installed vibe-qc on a given machine before,
just run the script, it will tell you what to `brew install` /
`pacman -S` / `apt install` first.

Python runtime dependencies such as NumPy, SciPy, spglib, seekpath,
jsonschema, joblib, and ASE come from `pyproject.toml` through pip.
`scikit-build-core` and pybind11 are isolated build requirements. pytest,
PySCF, and the other validation tools belong to the optional `test` and `dev`
extras; a calculation-only installation does not need them.

## One-shot bootstrap

After installing the system tools below, the easiest path on every
platform is:

```sh
git clone https://github.com/vibe-qc/vibe-qc.git
cd vibe-qc
./scripts/install.sh --dev                  # preflight → native deps → venv → pip install → banner
```

`install.sh` drives the full bootstrap and prints the banner so you
can confirm the C++ extension loaded. It accepts the same branch /
extras / venv flags as `update.sh`:

```sh
./scripts/install.sh --dev                        # bleeding-edge main
./scripts/install.sh --current                    # keep the current Git ref
./scripts/install.sh --dev --extras dev                 # tests + dispersion + py-spy
./scripts/install.sh --dev --extras basisopt             # add co-located vibe-basis
./scripts/install.sh --dev --extras mpi                 # add mpi4py for MPI-enabled runs
./scripts/install.sh --dev --extras dispersion,mpi      # add dispersion and MPI support
./scripts/install.sh --dev --python python3.13          # force a specific interpreter
./scripts/install.sh --dev --with-openblas              # vendor OpenBLAS
./scripts/install.sh --dev --libint-max-am 5_4_3        # faster libint, no g-function Hessians
./scripts/install.sh --dev --force                      # atomically replace an existing .venv
```

`install.sh --help` prints the full surface. Internally it coordinates the
same native setup and editable package build shown in the manual recipe, then
adds cross-platform checkout and target locks, safe-path and ownership checks,
transactional replacement, profile handling, and live-extension verification.
With no branch flag, or with `--release`, it targets the `release` branch.
When origin has no release branch, it selects origin's newest stable `vX.Y.Z`
tag and leaves the checkout at a detached HEAD. Local-only tags and prereleases
are excluded. Use `--branch vX.Y.Z` to reproduce a specific published release;
explicit `--branch` requests never fall back to a different ref.
Use `--current` when you
explicitly want to install an already-checked-out development snapshot.

Existing environments have a complete lifecycle:

```sh
./scripts/update.sh --dev                 # update source, native deps, and package
./scripts/reinstall.sh              # rebuild this checkout's venv with rollback
./scripts/uninstall.sh              # remove only this checkout's owned venv
```

`--force`, `--recreate-venv`, and `reinstall.sh` keep the prior venv in a
same-parent backup until the replacement installs and verifies. Unsafe paths,
symlinks, the checkout itself, directories without `pyvenv.cfg`, and venvs
owned by another checkout are refused rather than recursively deleted. For a
pre-marker installation from this checkout, add `--adopt-legacy` (and use
`--python` to name an external interpreter when the default is unsuitable).

```{note}
**Worktree-private venvs.** If you keep multiple checkouts (one per
branch you actively develop on), run `install.sh` inside each
checkout so each gets its own `.venv` pinned to that checkout's
`third_party/` trees. Never share a venv across checkouts -- the
editable pip install pins the C++ extension to one specific
`third_party/` tree, and a venv from a different checkout will
load the wrong native libraries.
```

(manual-bootstrap)=
### Advanced manual bootstrap

(manual-setup)=

If you'd rather invoke the steps by hand (e.g. on a stripped-down
machine, or to integrate with your own provisioning):

```sh
git clone https://github.com/vibe-qc/vibe-qc.git
cd vibe-qc
./scripts/setup_native_deps.sh       # preflight → 15-40 min build, idempotent
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Contributors and parity-test environments should request the development
tools explicitly:

```sh
.venv/bin/python -m pip install -e '.[dev]'
```

This manual environment does not receive the checkout ownership marker written
by `install.sh`. Ordinary in-place pip work remains possible, but a later
managed replacement or uninstall requires the explicit, exact-PEP-610
`--adopt-legacy` proof. Prefer `install.sh` unless custom provisioning needs
the individual steps.

`setup_native_deps.sh` first runs the preflight check (a few seconds)
that verifies every build prerequisite is present. If anything is
missing it bails with a copy-pasteable per-distro install command; if
everything is present it proceeds straight into the vendored builds.

`git clone` selects `main`; the flag-free installer switches to the published
`release` branch. The manual commands above keep the checked-out revision.
For reproducibility, record the actual Git SHA and banner; select only tags
that exist in this repository. Old monorepo tags remain in the archive.

`setup_native_deps.sh` calls (in order):

1. `build_libint.sh`, libint 2.13.1
2. `build_libxc.sh`, libxc 7.0.0
3. `build_spglib.sh`, spglib 2.7.0
4. `build_fftw.sh`, FFTW3 3.3.10
5. `build_libecpint.sh`, libecpint + pugixml + libcerf
6. `setup_basis_library.sh`, populate the gitignored
   `build/basis_library/basis/` overlay from libint's standard set plus
   our custom additions

After the build, `setup_native_deps.sh` probes which BLAS the
system has and prints a tip if it would benefit from an upgrade.
See [BLAS backend](#blas-backend) below.

Each per-dep script is idempotent. The orchestrator compares its provenance
stamp before those existence checks, so a changed version, recipe, or missing
library rebuilds only the affected dependency. Legacy install trees without a
stamp remain explicitly uncertified until a full native rebuild; they are not
relabeled as current merely because an idempotent build skipped them.
Locking-only lifecycle changes are excluded from the build-relevant recipe
fingerprint and therefore do not rebuild unchanged native libraries. Existing
three-field stamps migrate to the newer fingerprint during a normal update
without recompilation when their recipe identity matches directly or an exact,
reviewed historical-to-current fingerprint pair. Unknown fingerprints and
real native-recipe changes still rebuild the affected dependency.

### BLAS backend

vibe-qc's C++ core uses [Eigen](https://eigen.tuxfamily.org/) for
dense linear algebra. At build time, CMake links Eigen against
whatever optimised BLAS+LAPACK is available so dense matrix
products, eigendecompositions, and Cholesky factorisations
delegate to it (`EIGEN_USE_BLAS` / `EIGEN_USE_LAPACKE`). Without
an optimised BLAS, Eigen's generic-C++ kernels run instead, 
correct, but several × slower at SCF size. The print-banner's
`linked:` line carries the chosen backend (`blas Accelerate`,
`blas OpenBLAS`, etc.) so a persisted SCF log records which BLAS
produced the calculation.

Backend selection happens automatically:

* **macOS**: Apple Accelerate (system framework, no install needed).
* **Linux**: system OpenBLAS / MKL / netlib BLAS via CMake's
  `FindBLAS` auto-detect.
* **All platforms** (opt-in): a vendored OpenBLAS built into
  `third_party/openblas/install/`, see below.

To install an optimised BLAS via your system package manager
(simpler, lighter):

* Arch / Manjaro:  `sudo pacman -S blas-openblas`
* Debian / Ubuntu: `sudo apt install libopenblas-dev liblapacke-dev`
* Fedora / RHEL:   `sudo dnf install openblas-devel lapack-devel`
* macOS: nothing, Accelerate already provides what we need.

To vendor OpenBLAS into the checkout (no sudo needed, useful for
HPC / locked-down workstations / CI reproducibility, needs a
Fortran compiler):

```sh
WITH_OPENBLAS=1 ./scripts/setup_native_deps.sh
```

This invokes `scripts/build_openblas.sh` which clones OpenBLAS
0.3.33 from upstream, builds it with `DYNAMIC_ARCH=1`
(runtime CPU detection), `USE_LAPACK=1 USE_LAPACKE=1`
(bundles LAPACK + LAPACKE into `libopenblas.so`), and installs
into `third_party/openblas/install/`. The next `pip install -e .`
picks up the vendored install automatically (CMake's
`find_package(BLAS BLA_VENDOR=OpenBLAS)` resolves it via
`CMAKE_PREFIX_PATH`), with the banner reading
`linked: ... · blas OpenBLAS +LAPACKE`.

To disable BLAS linkage entirely and force Eigen's generic
kernels (mainly for debugging numerical issues), pass
`-DVIBEQC_USE_BLAS=OFF` to the vibe-qc CMake configure.

See [user_guide/blas](user_guide/blas.md) for the full surface:
how to read the linkage off the banner, the threading model,
when the vendored path is worth it (and when it isn't), and a
candid note on which perf problems BLAS linkage does *not*
solve.

## Per-platform: install the system tools

These are the commands the preflight check in
`./scripts/setup_native_deps.sh` will print if anything is missing on
your box. You can run them up-front or wait for the preflight to point
out what to install, either path lands at the same place.

### macOS (Homebrew)

```sh
brew install cmake ninja pkg-config libomp boost eigen gmp git python@3.14
# Optional but recommended for the OpenBLAS escape hatch:
brew install gcc                                   # provides gfortran
```

No separate BLAS package, **Apple Accelerate** ships with macOS and
is what Eigen links against. The preflight check will confirm
Accelerate is reachable and tell you if the framework is somehow
missing (extremely unusual; happens on stripped-down corporate
images).

```{note}
**Homebrew's `eigen` formula is currently Eigen 5.0.1** -- that's
expected and fine. libint 2.13.1 (vendored under
``third_party/libint/``) ships a CMake module that handles both the
Eigen 3 and Eigen 5 header layouts, so vibe-qc builds cleanly
against the brew install with no path-mangling on your end. If
``brew info eigen`` shows you 5.x and you've seen guidance elsewhere
warning to pin Eigen 3, ignore it -- that guidance is stale.
```

If [Homebrew](https://brew.sh) isn't installed yet, install it first
(`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- the preflight check will detect its absence and refuse to start.

> **Python on macOS.** Apple's stock `/usr/bin/python3` is currently
> Python 3.9 (the Xcode Command Line Tools stub) and does **not**
> meet vibe-qc's ≥3.11 minimum. Homebrew's `python@3.14` is the
> recommended runtime, the `brew install` line above pulls it in.
>
> After installing, make sure `python3` on your `$PATH` resolves to
> the new install rather than Apple's stub. The canonical Homebrew
> shell setup does this, verify and persist it once per shell:
>
> ```sh
> eval "$(/opt/homebrew/bin/brew shellenv)"
> echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
> ```
>
> Then check:
>
> ```sh
> python3 --version       # expect 3.14.x (or whichever brew installed)
> which python3           # expect /opt/homebrew/bin/python3
> ```
>
> If `python3 --version` still shows 3.9.x, Homebrew's bin isn't
> winning on `PATH`, force the symlink:
>
> ```sh
> brew link --overwrite --force python@3.14
> ```
>
> The preflight in `setup_native_deps.sh` catches the stale-Apple-
> python case explicitly so you'll get a clear error rather than a
> cryptic CMake or pybind11 failure later.

> **Apple Silicon note.** Homebrew lives at `/opt/homebrew` on ARM Macs.
> vibe-qc's CMake configuration runs `brew --prefix` and adds that to
> `CMAKE_PREFIX_PATH` automatically, so no manual path setup is needed.

> **`libomp` is not optional on macOS.** AppleClang ships without
> OpenMP support, and several vendored deps (spglib in particular)
> hard-require it. ``brew install libomp`` puts the headers + library
> at ``$(brew --prefix libomp)``; vibe-qc's `setup_native_deps.sh`
> auto-detects this on `main` (commit `973c8f3` and later). On older
> release branches that pre-date the auto-detect, also export the
> hint env var **before** running `setup_native_deps.sh` /
> `scripts/update.sh`:
>
> ```sh
> export OpenMP_ROOT=$(brew --prefix libomp)
> ```
>
> Add it to ``~/.zshrc`` if you want it to persist across sessions.
> Without it, CMake's ``FindOpenMP`` errors out with ``Could NOT find
> OpenMP_C (missing: OpenMP_C_FLAGS OpenMP_C_LIB_NAMES)`` and the
> spglib configure stops there.

### Linux, Arch / Manjaro

```sh
sudo pacman -S base-devel cmake ninja pkg-config git curl \
    gmp eigen boost python blas-openblas
# Optional but recommended for the OpenBLAS escape hatch:
sudo pacman -S gcc-fortran
```

`base-devel` is the meta-group that carries gcc/g++/make/binutils.
`python` on Arch is 3.13+ as of mid-2026 and bundles `pip` and the
`venv` module out of the box, no separate `python-pip` /
`python-virtualenv` needed. `curl` is used by ``setup_native_deps.sh``
to fetch libint / libxc / spglib / FFTW / libecpint tarballs.

`blas-openblas` is the OpenBLAS+LAPACK package; without an optimised
BLAS, Eigen's generic-C++ kernels run at SCF size and you lose a
multiple-× of performance. The default Arch BLAS slot ships as
reference netlib BLAS, which is functional but doesn't beat
Eigen-generic, `blas-openblas` is the right pick.

```{note}
**Arch's `eigen` package is currently version 5.0.1.** This works
fine -- libint 2.13.1 (which we vendor in ``third_party/libint/``)
shipped a CMake module that handles both the Eigen 3 and Eigen 5
header layouts. Arch keeps Eigen 5's headers under
``/usr/include/eigen3/`` for backward compatibility with dependent
packages, so no path-mangling is needed on your end.

If you previously installed the AUR `eigen3` (3.4.x) package as a
workaround, you can safely switch back to the standard `eigen` --
``yay -S eigen`` and accept the conflict prompt.
```

If you'd rather not install anything system-wide, an
[AUR](https://aur.archlinux.org/) helper like ``yay`` or ``paru`` works
the same way:

```sh
yay -S base-devel cmake ninja pkg-config git curl \
    gmp eigen boost python blas-openblas
```

If you need a specific Python version that differs from Arch's
rolling-release default, [pyenv](https://github.com/pyenv/pyenv)
(``yay -S pyenv``) builds an isolated interpreter under ``~/.pyenv``
without touching the system Python.

### Linux, Debian / Ubuntu

```sh
sudo apt update
sudo apt install \
    build-essential cmake ninja-build pkg-config git curl \
    libeigen3-dev libboost-dev libgmp-dev libgmpxx4ldbl \
    libopenblas-dev liblapacke-dev \
    python3 python3-dev python3-venv
# Optional but recommended for the OpenBLAS escape hatch:
sudo apt install gfortran
```

`libopenblas-dev` brings the BLAS+LAPACK Eigen wants;
`liblapacke-dev` adds the C interface so Eigen's dense solvers
(`LLT`, `SelfAdjointEigenSolver`, …) also delegate.

### Linux, Fedora / RHEL

```sh
sudo dnf install \
    @development-tools cmake ninja-build pkgconfig git curl \
    eigen3-devel boost-devel gmp-devel gmp-c++ \
    openblas-devel lapack-devel \
    python3 python3-devel
# Optional but recommended for the OpenBLAS escape hatch:
sudo dnf install gcc-gfortran
```

## What each step does

| Step | Purpose |
|---|---|
| OS-package install | C++ compiler, CMake/Ninja, plus the headers libint's code generator needs at build time (Boost, Eigen, GMP) and the BLAS+LAPACK Eigen delegates dense linear algebra to. |
| `setup_native_deps.sh` | Fetches and builds libint, libxc, spglib, FFTW3, libecpint into `third_party/<dep>/install/`. Each dep short-circuits silently if already built. |
| `setup_basis_library.sh` | Builds the packaged basis inventory under `python/vibeqc/basis_library/basis/`: 255 bundled Gaussian `.g94` runtime files, combining the libint-inherited standard files with the custom and BSE-derived overlay files. The bundled basis library ships **inside** the Python package so a stock `pip install` gets every basis set without the user setting `LIBINT_DATA_PATH`. |
| `python3 -m venv .venv` | Isolates vibe-qc from your system Python. |
| `pip install -e '.[test]'` | Drives CMake via scikit-build-core, picks up the vendored libs from `third_party/`, compiles the pybind11 module, installs in editable mode plus test deps (pyscf, ase, dftd3). |

## Verifying

```sh
.venv/bin/python -c "import vibeqc; vibeqc.print_banner()"
```

Should print a labeled box with vibe-qc's version, the build's git
provenance, and every linked native library:

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Release v0.15.x "Neese's Cheetah"  --  Quantum chemistry for molecules and solids                                           ║
║ © Michael F. Peintinger · MPL 2.0  ·  https://vibe-qc.com                                                                 ║
║ linked: libint 2.13.1 · libxc 7.0.0 · spglib 2.7.0 · libecpint 1.0.7 (vendored, MAX_L=5) · fftw3 3.3.10 · blas Accelerate ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

When the optional ``[dispersion]`` extra is installed (``pip install
-e '.[dispersion]'``), the banner gains a second linkage line
listing the ``dftd3`` and ``dftd4`` PyPI-wheel versions whose
bundled binary libraries vibe-qc loads at runtime:

```
║ linked: libint 2.13.1 · libxc 7.0.0 · spglib 2.7.0 · libecpint 1.0.7 (vendored, MAX_L=5) · fftw3 3.3.10 · blas Accelerate ║
║ dispersion: dftd3 1.4.0 · dftd4 4.2.0                                                                                     ║
```

If you cloned from `main` instead of `release`, the first line
reads `dev X.Y.Z.devN (main @ <sha>)` (with a ``dirty`` flag if the
working tree has uncommitted changes). The exact
``libint`` / ``libxc`` / ``spglib`` / ``libecpint`` / ``fftw3``
versions match what ``setup_native_deps.sh`` vendored
(``libecpint`` annotates its bundled angular-momentum cutoff
``MAX_L=5`` so a build can be matched to a ``MAX_L`` mismatch at a
glance). The trailing ``blas …`` field records which BLAS backend
Eigen got linked against -- `Accelerate` on macOS, `OpenBLAS
+LAPACKE` for an optimised Linux install, etc. See
[BLAS](#blas-backend) for the full label table. The banner
exercises the C++ extension under the hood (each linked library
reports its own version through the pybind11 module), so seeing it
is proof both that the Python side imports and that the native
side loaded.

If even that crashes, fall back to the minimal smoke test that
isolates the C++ extension load from anything else:

```sh
.venv/bin/python -c "import vibeqc; print(vibeqc.hello())"
# vibeqc core alive
```

End users should run the banner check and one small calculation. Contributors
should select the affected test lane:

```sh
.venv/bin/python scripts/test_gate/run_full_suite.py \
  --wt "$PWD" --list-lanes
.venv/bin/python scripts/test_gate/run_full_suite.py \
  --wt "$PWD" --py .venv/bin/python --lane <affected-lane>
```

See [Running calculations](running.md#verify-an-installation-or-a-development-change)
and [Developer test lanes](developer_test_lanes.md). The full inventory is a
release-confidence tool, not the default installation smoke test.

## Optional Microsoft SKALA-1.1 functional

Microsoft SKALA-1.1 uses an optional PyTorch runtime on the CPU. It is not part
of a normal vibe-qc install. Build vibe-qc first, then install a compatible
PyTorch build and the extra in the same environment. On platforms where the
normal package index does not supply the CPU-only wheel you want, follow
[PyTorch's CPU installation selector](https://pytorch.org/get-started/locally/)
first; for example, use its documented CPU wheel index rather than assuming
that ordinary package resolution selects a CPU-only build.

```sh
.venv/bin/pip install -e '.[skala]'
```

The supported SKALA runtime is Python 3.11 through 3.13 with PyTorch 2.12 or
2.13. The base vibe-qc package still supports Python 3.14, but the `[skala]`
extra intentionally installs no Torch dependency there. PyTorch 2.13 now has
a Python 3.14 wheel, but the combined vibe-qc native core, Torch, and OpenMP
runtime has not passed the SKALA acceptance oracle. Use a Python 3.13 or
earlier environment for SKALA. The Microsoft `skala` Python package and PySCF
are not installed or imported.

```{warning}
On macOS, the vibe-qc native core and current PyTorch wheels can initialize
different `libomp` copies and abort in one process. The SKALA route does not
yet install a verified workaround. Treat in-process SKALA evaluation on macOS
as unsupported and use a compatible Linux CPU environment. Importing vibe-qc
and `dry_run=True` remain safe because they do not import Torch.
```

Importing vibe-qc does not load PyTorch or contact the network. On the first
SKALA evaluation, vibe-qc downloads the official CPU checkpoint from the
immutable `microsoft/skala-1.1` revision
`99b5ed87e5f69d9216e1f9e30148b922eaea1241`. The expected file is
`skala-1.1-rev1.fun`, with SHA-256:

```text
7f3e8622e1eb520ccd88a55464c3e359ac4d7e5ccbd1fb77a26afa1e1c20a5cd
```

The cache lives below `$XDG_CACHE_HOME/vibeqc/skala`, or
`~/.cache/vibeqc/skala` when `XDG_CACHE_HOME` is unset. To prefetch it for a
connected cluster login node:

```sh
.venv/bin/python -c \
  "from vibeqc.skala import ensure_skala_model; print(ensure_skala_model())"
```

For an offline compute node, ask vibe-qc for the exact destination with
`vibeqc.skala.model_cache_path()`, place the official file there, and copy the
whole revision directory to the node. Every load hashes the bytes before
TorchScript deserialization. A missing or corrupt entry is repaired atomically
when the network is available; an offline or failed repair stops before
deserialization. Do not substitute an untrusted `.fun` file: TorchScript
deserialization can execute code.

See the dedicated [Microsoft SKALA-1.1 guide](user_guide/skala.md) for complete
molecular and experimental periodic examples, supported routes, the separate
D3(BJ) setting, dry-run provenance, and current derivative limitations.

## Optional terminal viewer (MolTUI)

Vibe-qc writes [Molden](https://www.theochem.ru.nl/molden/) files
by default for any calculation that calls ``run_job(...)``. Those
files render in any of [Avogadro], [Jmol], [VMD], [PyMOL],
[Molden], or [ChimeraX] -- but if you'd rather **inspect orbitals
inline in the terminal** (handy on a remote SSH session, or when
you're iterating fast), [MolTUI](https://github.com/kszenes/moltui)
is the right tool. Pure Python; renders geometries, orbitals, and
normal modes via Unicode block characters.

[Avogadro]: https://two.avogadro.cc/
[Jmol]: https://jmol.sourceforge.net/
[VMD]: https://www.ks.uiuc.edu/Research/vmd/
[PyMOL]: https://pymol.org/
[Molden]: https://www.theochem.ru.nl/molden/
[ChimeraX]: https://www.cgl.ucsf.edu/chimerax/

Two ways to install it alongside vibe-qc:

**1. As a vibe-qc extra** -- single command, captures the dependency
in your install record:

```sh
.venv/bin/pip install '.[viewer]'           # from the repo checkout
```

(`pip install 'vibe-qc[viewer]'` will become the right command once
vibe-qc lands on PyPI -- not yet.)

**2. Interactive script** -- run any time after the venv is set up;
re-runnable, asks before installing each tool:

```sh
./scripts/install_optional_tools.sh
# → "Install moltui? [Y/n]"
```

Pass ``--yes`` for non-interactive automation, or a tool
name to install just that one (``./scripts/install_optional_tools.sh
moltui``). Use ``--upgrade`` to refresh an already-installed tool,
``--venv PATH`` or ``--python BIN`` to choose the target explicitly, and
``--dry-run`` to preview the pip command. For example:

```sh
./scripts/install_optional_tools.sh --venv .venv --upgrade --yes moltui
```

After install:

```sh
.venv/bin/python water.py                       # produces water.molden
.venv/bin/moltui water.molden                   # render in the terminal
```

MolTUI also supports ``.cube`` (vibe-qc's grid output), ``.fchk``,
``.xyz``, ``.gbw`` (ORCA), and ``.hess`` (ORCA normal modes), so the
same install lets you view files from external codes too.

### If your output is a `.qvf`

vibe-view renders QVF archives in the terminal itself, no extra tool
required -- and because a QVF is one self-describing archive, the same
command reaches the band structure, spectra, SCF trail, charges and
citation bundle, not just the geometry:

```sh
vibe-view show water.qvf        # one frame, then exit (base install)
vibe-view tui  water.qvf        # interactive; install with --extras tui
```

For a dedicated terminal install, run
`./scripts/install.sh --extras tui` from the separate vibe-view checkout root. For an
existing viewer environment, run
`./scripts/update.sh --skip-git --extras tui`. A bare public-index
`pip install 'vibeview[tui]'` does not work because vibeview is not on PyPI.

Like MolTUI it needs no GUI and no display server. See
[reading a `.qvf` in the terminal, over SSH](tutorial/vibe_view_terminal.md).
Pick MolTUI for loose files or output from other codes; pick terminal mode
when you have a `.qvf`.

## Optional basis-set optimization driver (vibe-basis)

`vibe-basis` drives **external** SCF programs (currently CRYSTAL23) under
an optimization loop to fit new basis sets. It is the modern successor
to the CRYSTAL09 + MINUIT2 pipeline behind pob-TZVP, and is co-located at
`vibe-basis/` in this checkout and versions independently of vibe-qc
(see [`vibe-basis/VERSIONING.md`](https://github.com/vibe-qc/vibe-qc/blob/main/vibe-basis/VERSIONING.md)).

You need it only if you are *fitting* basis sets. Using an existing
basis set needs nothing here.

Use its standalone installer. It owns `vibe-basis/.venv`, so it never
collides with vibe-qc's compiled root environment:

```sh
./vibe-basis/scripts/install.sh                  # standard profile (SciPy)
./vibe-basis/scripts/install.sh --extras all     # every optimizer backend
```

Profiles are `core`, `standard`, `optimizers`, `all`, and `test`. The
standard profile is the recommended starting point. Install the queue from
its [own repository](user_guide/queue.md) when queue integration is needed.
The old `--with-vq` co-located-source recipe does not apply after the split.

The same standalone environment has update, reinstall, and uninstall entry
points. Uninstall retains source and user calculation data:

```sh
./vibe-basis/scripts/update.sh
./vibe-basis/scripts/reinstall.sh
./vibe-basis/scripts/uninstall.sh
```

The root lifecycle also accepts `--extras basisopt` when you intentionally want
vibe-basis in the compiled vibe-qc environment. It resolves the co-located
package explicitly because pip does not read `[tool.uv.sources]`. The dedicated
standalone environment above remains the recommended onboarding path.

Installing it is also what makes `vibeqc.basis_optimization`'s CRYSTAL
recipe modules importable. `calculators` and
`recipes.crystal_stage1..3` / `crystal_objective` / `production` import
`vibe_basis` directly. Without the extra they raise
`ModuleNotFoundError`; the rest of `vibeqc.basis_optimization` (the
parametrisation, gradients, BDIIS driver and LD diagnostics) works
without it.

After install:

```sh
vibe-basis/.venv/bin/vb --version
vibe-basis/.venv/bin/vb parse crystal my-run.out    # energy + convergence flags
```

## Common issues

**"Preflight: the following build prerequisites are missing"**
`setup_native_deps.sh` aborts upfront if a build tool / header /
library it needs isn't on the box. The message names every missing
item and ends with the exact per-distro install command -- copy it,
run it, re-run `./scripts/setup_native_deps.sh`. The check runs at
the very top of the script so you never end up halfway through a
libint build before discovering `pkg-config` was missing.

Set `VIBEQC_SKIP_PREFLIGHT=1` only if you know the message is a
false positive (e.g. a custom-prefix install the heuristic doesn't
find); the underlying CMake errors are much less actionable.

**"python ≥3.11 on PATH (found python3 = 3.9 -- likely Apple's stub)"**
You're on macOS and `/usr/bin/python3` wins over Homebrew's install.
Fix:

```sh
brew install python@3.14                            # if not already
eval "$(/opt/homebrew/bin/brew shellenv)"            # current shell
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc  # persist
brew link --overwrite --force python@3.14           # if still not first
python3 --version                                    # confirm 3.14.x
```

The preflight detects this case explicitly because the Apple stub
silently breaks the venv-and-pip step later with a much less obvious
message.

**`./scripts/build_libint.sh: line N: brew: command not found`**
You're on Linux but the script tried the macOS path. This was fixed
in 0.1.0+; pull `main` and re-run. The script now sniffs `uname -s`
and uses the right toolchain per platform.

**"Could not find Libint2 / Libxc / Spglib / FFTW3"**
CMake didn't find one of the vendored installs. The most common cause
is that `setup_native_deps.sh` was interrupted partway through.
Re-running it is safe -- finished deps short-circuit, only the
unfinished one rebuilds.

If you genuinely want to point at a system install instead of the
vendored one, simply don't run that dep's build script. CMake silently
falls through to system discovery if `third_party/<dep>/install/` is
absent.

**"error: externally-managed-environment" on macOS**
You're trying to `pip install` against Homebrew's system Python.
PEP 668 blocks this; always use a virtualenv:
```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
```

**libint build is slow**
The libint code generator produces a large amount of source for
`max_am=5` plus derivatives through 2nd order (several thousand files).
The generated-kernel count grows roughly as `max_am^4`, and since
2026-08-06 the 2nd-derivative stratum is generated at `max_am=4`
(g functions) rather than 3, which triples that stratum. The compile
phase uses Ninja which parallelizes; expect 1.5-2 h on a modern
8-core machine, longer on Mac minis / small VMs. This is a one-time
cost per checkout: re-running the install scripts skips a libint
install that is already present.

Since 2026-08-13 the codegen + compile chatter goes to
`third_party/libint/build.log` rather than the terminal; the terminal
shows one heartbeat line per minute (elapsed time, last Ninja step,
disk/memory headroom), and if the build fails, the last 200 log lines
are replayed so the error is always visible. Follow the full output
live with `tail -f third_party/libint/build.log`, or set
`VIBEQC_BUILD_VERBOSE=1` to stream everything into the terminal as
before. The log stays still during the single-threaded codegen step
(20-30 min): that is Ninja buffering one long step, not a hang; the
heartbeat keeps counting.

That build is a choice rather than a fixed cost. `--libint-max-am` sets it:

```bash
./scripts/install.sh --dev --libint-max-am 5_4_3
```

The three numbers are the max angular momentum at derivative orders 0,
1 and 2, each in the range 1..6:

| Spec | Build | Energies | Gradients | Hessians |
|---|---|---|---|---|
| `5_4_3` | ~30 min | h (L=5) | g (L=4) | f (L=3) |
| `5_4_4` (default) | ~1.5-2 h | h | g | g |
| `6_5_4` | longer | i (L=6) | h | g |
| `6_6_6` | longest | i | i | i |

Above a stratum's limit libint throws rather than returning a wrong
number, so **the spec only decides which basis sets each derivative
order accepts: no computed value changes.** Dropping to `5_4_3` costs
you analytic Hessians on a g-function basis (def2-TZVPP heavy atoms,
cc-pVQZ, def2-QZVP) and nothing else; energies, gradients, geometry
optimization and every periodic path are bit-identical. Going up to
`6_5_4` buys i-function energies for cc-pV6Z-class work.

6 is the ceiling: vibe-qc's periodic AO-pair Fourier transform throws
above `cart_to_sph_data.hpp`'s `kMaxL = 6`, so a higher libint would
build integrals the periodic stack cannot consume. The error message
says so, and says what raising it would take.

The same flag exists on `update.sh` and `update_native_deps.sh`, and
the setting is recorded in the built tree, so you can change your mind
in either direction later without knowing any force flags:

```bash
./scripts/update.sh --dev --libint-max-am 5_4_4
```

That rebuilds libint, and only libint. `./scripts/doctor.sh` reports
which spec the current install was built with.

**On Linux** the install scripts automatically cap parallelism to
prevent the OOM-kill pattern that hung compute-reference on 2026-05-16. The
cap is `min(nproc, max(2, mem_mb // 15000), 8)` -- 15 GB/worker
budget, hard ceiling of 8 even on monster boxes. Concretely:

| Host (Linux)              | nproc | RAM    | parallelism |
|---------------------------|------:|-------:|------------:|
| compute-reference (32 th, 125 GB)   |    32 | 125 GB |           8 |
| compute-small    (16 th,  62 GB)   |    16 |  62 GB |           4 |
| small VM (4 th, 16 GB)    |     4 |  16 GB |           2 |

The same scripts also self-re-exec under `nice -n 19 ionice -c 3`
on Linux so the build runs at idle CPU + IO priority -- the box
stays responsive while compiling. The mechanism is in
`scripts/_safe_build_env.sh` (sourced by every entry-point script
that triggers heavy compilation).

To override either side-effect:

```sh
# Pin parallelism to a specific value (skips the formula).
CMAKE_BUILD_PARALLEL_LEVEL=12 ./scripts/setup_native_deps.sh

# Skip the nice/ionice re-exec (use full CPU + IO priority).
VIBEQC_BUILD_NICED=1 ./scripts/setup_native_deps.sh

# Silence the cap-announcement banner.
VIBEQC_BUILD_ENV_QUIET=1 ./scripts/setup_native_deps.sh
```

**On macOS** the cap doesn't engage (no `/proc/meminfo`, no
`ionice`) -- set `CMAKE_BUILD_PARALLEL_LEVEL` manually if your dev
box is RAM-constrained.

**Python 3.14 + some dependency refuses to build**
Most scientific packages have wheels for 3.11-3.13; 3.14 is bleeding
edge and occasionally missing binary wheels (numpy, pyscf, scipy
typically catch up within a few weeks). Stick to 3.12 or 3.13 if
you hit this.

**Clean rebuild**
If everything goes sideways, the safe button is `update.sh --clean`:
```sh
./scripts/update.sh --dev --clean
```
which removes the pinned native source/build/install caches and Python build
caches, then failure-atomically replaces `.venv/` after rebuilding from
source. Do not delete `python/vibeqc/basis_library/basis/`: that is tracked
runtime data, not a generated cache. For a reinstall that preserves the native
builds, use:

```sh
./scripts/reinstall.sh --skip-native-deps
```

**"Is my install OK?" -- `doctor.sh`**
Run `./scripts/doctor.sh` for a read-only health report:
working-tree state, which vendored `third_party/<dep>/install/`
trees exist, build-stamp drift against the current `build_*.sh`
files, venv health (broken python / stale pip shebang / pyvenv.cfg
mismatch), and the banner. No builds, no installs -- safe to run
at any time, including during an in-flight calculation.

## Updating

```sh
./scripts/update.sh --dev       # → bleeding-edge main (X.Y.devN banner)
```

The wrapper handles `git pull` + stamp-aware `setup_native_deps.sh` +
`pip install -e '.[test]'` + the new-banner print
in one shot -- and refuses to run if the working tree is dirty
(stashing-without-asking is worse than failing loud). It also refuses before
changing Git when no usable venv exists. `--recreate-venv` preserves the
existing base interpreter by default; select a different one explicitly with
`--recreate-venv --python python3.13`.

The dependency scripts compare version, build-recipe, and artifact stamps and
automatically rebuild an affected library when those inputs drift.
`--rebuild-native-deps` is the repair and audit option that forces clean
rebuilds of all vendored trees; it is not required for an ordinary pinned
version update:

```sh
./scripts/update.sh --dev --rebuild-native-deps
```

If you keep **two checkouts side-by-side** (one for production runs,
one for dev), use `update.sh` (or `--release`) inside the release
tree and `update.sh --dev` inside the dev tree. The full option set
+ the manual-equivalent recipe (for debugging an update that didn't
take) are in [`updating.md`](updating.md).

## Picking a build to test against

vibe-qc's runtime banner records which git revision produced any
output (see [`release_process.md`](release_process.md)). When you
want to run calculations against a specific build for testing --
comparing against `release`, validating a topic-branch fix, etc.
-- `update.sh --ref` takes any branch or tag:

```sh
./scripts/update.sh --ref v0.9.0                 # pin to a tag
./scripts/update.sh --ref feature/some-topic     # try a branch
```

The banner on the next run will then read e.g. `Release v0.9.0` or
`dev 0.9.0 (feature/some-topic @ abc1234)`, which is what you'll see
prepended to every persisted SCF log.
