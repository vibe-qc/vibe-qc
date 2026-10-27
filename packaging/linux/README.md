# Standalone command-line packages

The website distributes vibe-qc and vibe-queue as Linux x86-64 archives, alongside
Apple Silicon DMGs built by each project's `packaging/macos/build.py`. Linux
packages run on Debian 12 or newer compatible systems. Extract the archive and
run `vibe-qc/bin/vibe-qc --help` or `vibe-queue/bin/vq --help`. No system Python,
compiler, root access or package installation is required.

## Release inputs

Build only from an immutable stable release tag. Export the tag, record its full
commit and source SHA-256, and build its wheel in a bounded native Linux builder.
Never build a release package from an editable development checkout. Use a
separate output directory and preserve the reusable compiler/dependency caches.

The Python runtime is the hash-verified python-build-standalone Linux x86-64
install archive. Preserve its matching full archive's `PYTHON.json` and license
files. Resolve the release's pinned Python dependencies to Linux wheels; retain
all wheel filenames and SHA-256 hashes in `linux-runtime-lock.json`. Build the
application wheel with its release's build backend and record that hash too.

For QC, use the release's native dependency cache identity and recipes, portable
CPU flags, and `VIBEQC_RELOCATABLE_RUNTIME=ON`. Repair the native wheel's dynamic
library closure before assembly. Include required non-glibc libraries and remove
build-host RPATHs. Preserve all native-library source, patches, build options and
licenses, including FFTW, Libint and compiler/BLAS runtimes. The assembler
collects non-glibc system dependencies and their distribution notices.

For bounded-memory Clang builds, pass
`-Ccmake.define.CMAKE_PROJECT_TOP_LEVEL_INCLUDES=/path/to/low-memory-bindings.cmake`
to `pip wheel`. This compiles only the large Python binding translation unit at
`-O1` without LTO and releases the compiler AST before code generation; the
calculation core retains its normal release optimization.
Keep this hook and its hash with the release inputs. QC's application
source is MPL-2.0; bundled dependencies retain their respective terms.

`build.py` consumes these verified inputs without network access:

```sh
python3 build.py --lock linux-runtime-lock.json --inputs runtime-inputs \
  --source release-source.tar.gz --wheel application.whl \
  --notices notices --output packages
```

Use `--libraries native-libraries` when additional qualified libraries must be
copied into the interpreter's library directory. The lock contains `product`,
`version`, `sourceCommit`, `sourceSha256`, `wheelSha256`, `python`, `wheels`,
`inputs` (filename/SHA-256 pairs), `commands`, and user-facing `instructions`.
Include the exact builder, lock and corresponding release source in each package.

## Qualification and publication

Move the resulting package into a directory containing spaces, use an empty
home and a poisoned Python environment, and check every bundled command. Queue
qualification starts a loopback-only web console and checks its HTML and OpenAPI
routes, then stops it. QC qualification checks native HF and LDA reference
energies, basis/ECP resolution, RDKit, D3/D4 and FFTW. Confirm that these operations
do not modify package resources. Repeat in a minimal Linux container without
system Python or compiler to detect accidental build-environment dependencies.

Publish only after both native platforms pass, using
`website/scripts/publish-desktop-release.py`. It preserves immutable versioned
files and advances the current pointer atomically. QC and Queue use
`download-linux.tar.gz`; Sketch and View retain `download-linux.AppImage`.
Verify every public artifact and source archive against the manifest over HTTPS.
Rebuild the website to embed the four current versions into its initial HTML.
Do not cut releases or trigger CI as part of download maintenance.

## Optional MACE package

The `runtime-mace.lock.json` profile builds `vibe-qc-mace-VERSION-x86_64.tar.gz`
with a distinct `vibe-qc-mace/` extraction directory. It includes one Python
3.14 interpreter, the native QC wheel for that interpreter, CPU-only PyTorch,
MACE and their full dependency graph. Standard download profiles are unchanged.
Do not remove setuptools when it appears in the runtime wheel lock: PyTorch
requires it. The MACE builder checks dependencies again after pruning build
tools. It preserves wheel helper executables and repairs their relative library
paths when their required libraries are in the adjacent `lib/` directory.

Keep the locked python-hostlist source archive with the notices. On hosts without system CA certificates, the optional runtime uses its
already bundled certifi store for HTTPS. Explicit SSL certificate settings and
existing system trust stores are preserved. No foundation
model weights belong in the package. Run `packaging/smoke_mace.py` with packaged
Python and a fresh HOME/XDG cache, then rerun with `--offline` and the same cache.
Run the normal native QC archive checks as well. Publish the qualified pair
with `--product vibe-qc --profile mace`; its independent manifest lives under
`/updates/vibe-qc/variants/mace/current/` and cannot replace standard downloads.

### Standalone naming input

For QC revisions requiring `vibeqc-naming`, build the pure-Python wheel from
the selected source's `vibeqc-naming/` directory and include its filename in
`wheels` and its SHA256 in `inputs` in the release lock. The existing offline
installer installs it with the QC wheel and `pip check` refuses an absent or
incompatible dependency. Preserve this wheel and source with the release
inputs; do not relabel an older bundled-naming QC wheel. Publication of a
standalone wheel/source and a resolver route remains a separate release step.
