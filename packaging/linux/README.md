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
