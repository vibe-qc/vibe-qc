---
myst:
  html_meta:
    "description": "BLAS + LAPACK backend in vibe-qc, Apple Accelerate on macOS, OpenBLAS / MKL / netlib on Linux, vendored OpenBLAS via WITH_OPENBLAS=1. How to check what's linked, threading model, and when it does (and does not) move SCF wall time."
    "og:title": "vibe-qc, BLAS + LAPACK backend"
    "og:description": "EIGEN_USE_BLAS / EIGEN_USE_LAPACKE delegates dense linear algebra to the linked BLAS. macOS uses Accelerate by default. Linux auto-detects system BLAS; the vendored OpenBLAS path closes the gap on no-sudo / HPC / CI-reproducibility setups. Banner reads the chosen backend."
---

# BLAS + LAPACK backend

vibe-qc's C++ core uses [Eigen](https://eigen.tuxfamily.org/) for
dense linear algebra. When a BLAS+LAPACK library is linked, Eigen
delegates its matrix products, eigendecompositions, Cholesky
factorisations, and dense LU / QR / SVD to that library
([`EIGEN_USE_BLAS`](https://eigen.tuxfamily.org/dox/TopicUsingBlasLapack.html)
catches dense `*`, `.noalias()` products and triangular solves;
`EIGEN_USE_LAPACKE` catches `Eigen::LLT`, `Eigen::LDLT`,
`Eigen::SelfAdjointEigenSolver`, `Eigen::FullPivLU`, etc.). The
delegation is set at compile time; no source-code change is
required to take advantage of it.

This page describes:

* which BLAS gets linked on which platform, and how to check;
* how vibe-qc's OpenMP layer interacts with BLAS-internal
  threading;
* when the vendored OpenBLAS path is worth it (and when the
  system install is enough);
* a candid note on what the linkage does *not* fix, most
  notably the parent perf-optimisation lever for RIJCOSX.

## What's linked: read it off the banner

`vibeqc.print_banner()` prints a `linked:` line that names the
BLAS backend in plain English:

```
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║ dev 0.7.5 "Löwdin's Compass" (main @ 713cae9)  —  Quantum chemistry for molecules and …  ║
║ © Michael F. Peintinger · MPL 2.0  ·  https://vibe-qc.com                                ║
║ linked: libint 2.13.1 · libxc 7.0.0 · spglib 2.7.0 · blas Accelerate                     ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
```

The `blas` field is one of:

| Banner label             | What it means                                                                |
|--------------------------|------------------------------------------------------------------------------|
| `blas Accelerate`        | Apple Accelerate framework (macOS default)                                   |
| `blas OpenBLAS`          | System or vendored OpenBLAS, no LAPACKE                                      |
| `blas OpenBLAS +LAPACKE` | OpenBLAS *plus* the LAPACK C interface (Eigen's dense solvers also delegate) |
| `blas MKL`               | Intel MKL                                                                    |
| `blas netlib BLAS`       | Reference netlib BLAS (functional but slow, see below)                      |
| `blas none`              | `VIBEQC_USE_BLAS=OFF` at build, Eigen-generic kernels                        |
| `blas unknown`           | C extension didn't load or `blas_info()` not in this build                   |

Programmatically:

```python
>>> import vibeqc
>>> from vibeqc.banner import library_versions
>>> library_versions()["blas"]
'OpenBLAS +LAPACKE'

>>> from vibeqc._vibeqc_core import blas_info
>>> blas_info()
{'libraries': '/usr/lib/libopenblas.so',
 'blas_enabled': True,
 'lapacke_enabled': True}
```

The raw `libraries` string is the literal `BLAS_LIBRARIES` value
CMake's `FindBLAS` returned at configure time, useful when
debugging which `.so` actually got resolved.

## Backend selection per platform

### macOS, Apple Accelerate (default)

Accelerate ships with the OS and is highly tuned for Apple
Silicon and recent Intel Macs. CMake picks it via
`find_package(BLAS BLA_VENDOR=Apple)`. Nothing to install,
nothing to configure.

`otool -L` on the compiled `_vibeqc_core*.so` confirms:

```
$ otool -L .../vibeqc/_vibeqc_core.cpython-*-darwin.so | grep Accelerate
  /System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate
```

`EIGEN_USE_LAPACKE` is **not** set on macOS, Accelerate's
LAPACKE bridge has version-specific quirks vs Eigen's
expectations, and Accelerate's BLAS catches the SCF matrix
products that matter. To force OpenBLAS on macOS anyway (mostly
for reproducible cross-platform builds), build the vendored
OpenBLAS (see below) and pass `-DVIBEQC_BLAS_VENDOR=OpenBLAS` to
CMake.

### Linux, system OpenBLAS / MKL / netlib auto-detect

CMake's `FindBLAS` scans well-known vendors in order. With no
override, the first one found wins. If you have admin access,
the cleanest path is to install OpenBLAS via the package manager:

```sh
# Arch / Manjaro:
sudo pacman -S blas-openblas      # replaces reference BLAS system-wide

# Debian / Ubuntu:
sudo apt install libopenblas-dev liblapacke-dev

# Fedora / RHEL:
sudo dnf install openblas-devel lapack-devel
```

Then `pip install -e .` will pick up the system OpenBLAS
automatically, no rebuild flag needed.

If only `liblapack` / `libblas` (netlib reference) is installed,
CMake will still link, but the banner reads `blas netlib BLAS`, 
a perf trap (see "When BLAS linkage does and does not matter"
below). `scripts/setup_native_deps.sh` probes for this case and
prints a setup-time recommendation.

### Linux, vendored OpenBLAS (no-sudo / HPC / CI)

For HPC users without root, locked-down workstations, or CI
boxes that want a byte-identical BLAS across builds, an opt-in
vendored OpenBLAS lives at `third_party/openblas/install/`:

```sh
WITH_OPENBLAS=1 ./scripts/setup_native_deps.sh
# ...or directly:
./scripts/build_openblas.sh
```

Requirements:

* A Fortran compiler (`gfortran`), netlib LAPACK's Fortran
  sources are bundled into the resulting `libopenblas.so` to
  give `EIGEN_USE_LAPACKE`-grade dense solvers. Install hints
  per-distro are printed by `build_openblas.sh` if it can't find
  one.
* About 5-10 minutes of build time on first run.

Once `third_party/openblas/install/` exists, CMake auto-prepends
it to `CMAKE_PREFIX_PATH` and pins `BLA_VENDOR=OpenBLAS`, so the
vendored library wins deterministically over any system BLAS.
The next `pip install -e .` rebuilds the C extension against
`libopenblas.so` (with RPATH baked, so `import vibeqc` finds it
without `LD_LIBRARY_PATH` fiddling) and the banner switches to
`blas OpenBLAS +LAPACKE`.

Build flags used (see `scripts/build_openblas.sh` for the full
list and rationale):

* `DYNAMIC_ARCH=1`, runtime CPU dispatch. The vendored binary
  picks the right kernel on Haswell / Skylake / Zen / Apple
  Silicon / … at load time, so the build host's SIMD support
  doesn't constrain where the binary can run.
* `USE_LAPACK=1 USE_LAPACKE=1`, single `libopenblas.so` carries
  BLAS, LAPACK, and the LAPACKE C interface (so both Eigen
  delegation macros activate).
* `USE_THREAD=1 USE_OPENMP=0 NUM_THREADS=128 NO_AFFINITY=1`, 
  pthreads-internal threading, capped, no CPU pinning. Combined
  with `OPENBLAS_NUM_THREADS=1` (set by default in
  `python/vibeqc/__init__.py`) this gives a sane BLAS-serial,
  OpenMP-parallel posture, see the next section.

## Threading: BLAS-serial, app-parallel

vibe-qc parallelises its own inner loops with OpenMP, controlled
by `OMP_NUM_THREADS`. If the linked BLAS *also* threads
internally, OpenBLAS-pthreads, MKL, you get N×K threads
contending for cores: measurably worse than either layer alone.

`python/vibeqc/__init__.py` pins BLAS-internal threading to 1 by
default before the C extension loads:

```python
def _pin_blas_threads() -> None:
    for key in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")
```

`os.environ.setdefault` only fills a key that isn't already set,
so an explicit `OPENBLAS_NUM_THREADS=4` in your shell wins. If
you specifically want BLAS-internal threading (e.g., for a
single-threaded outer loop where BLAS parallelism is the only
parallelism available), export it before `import vibeqc`.

The default, BLAS serial, OpenMP parallel via `OMP_NUM_THREADS`
- is the standard posture for mixed OpenMP-and-BLAS codes (ORCA,
NWChem, PySCF, Psi4 all do the same).

## When BLAS linkage does, and does not, matter

vibe-qc inherits its perf model from Eigen + the vendored
numerical libraries (libint, libxc, libecpint) and its own
hand-rolled kernels. EIGEN_USE_BLAS+LAPACKE matters where Eigen
ops are the bottleneck, and **not** where they aren't.

**Where it helps:** density-fitting SCF (`density_fit=True`) at
larger systems, periodic-SCF Cholesky factorisations, MP2
amplitude updates, hessian assembly. Anything dominated by
dense matrix products on Fock-sized matrices (≳ 500 BFs)
benefits.

**Where it's near-flat:** RIJCOSX-heavy HF / hybrid DFT at small
systems. The cost there is in the COSX grid loop, AO integral
assembly, and DF 3-center kernel evaluation, paths that don't
go through Eigen's dense kernels, so `EIGEN_USE_BLAS` doesn't
touch them. Measured on butanethiol (C₄H₁₀S) HF / def2-TZVP /
RIJCOSX, the wall time difference between no-BLAS, system
netlib BLAS, and vendored OpenBLAS+LAPACKE is within 7% (~298 s
across all three).

If your run is RIJCOSX-bound and you need it faster, the
inner-loop work is in `cpp/src/cosx.cpp` (Q-junction caching,
shell-pair Schwarz screening on the K build, A_g block
sparsity) rather than the BLAS backend. Track that work in the
perf chat on `claude/perf-vs-orca`.

## Disabling BLAS (debugging)

For numerical regressions where you suspect Eigen / BLAS
roundoff differences (rare, should produce ulp-level
discrepancies at most, easily within SCF tolerances), pass
`-DVIBEQC_USE_BLAS=OFF` to the CMake configure to force
Eigen's generic kernels:

```sh
pip install --no-build-isolation -e . \
    -Ccmake.define.VIBEQC_USE_BLAS=OFF
```

The banner will then read `blas none`. Run side-by-side against
a default build to isolate which kernel the discrepancy lives
in.

## Citation

* OpenBLAS, [github.com/OpenMathLib/OpenBLAS](https://github.com/OpenMathLib/OpenBLAS),
  BSD 3-Clause; the LAPACK + LAPACKE sources bundled inside the
  vendored `libopenblas.so` are Reference-LAPACK
  ([github.com/Reference-LAPACK/lapack](https://github.com/Reference-LAPACK/lapack)),
  modified BSD.
* Apple Accelerate is system-bundled on macOS; no separate
  citation, the [Accelerate framework reference](https://developer.apple.com/documentation/accelerate)
  is the entry point.
* Full license inventory:
  [docs/license.md](../license.md).
