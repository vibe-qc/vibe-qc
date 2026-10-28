# Prospective native build receipts

`scripts/record_native_build.py` is an opt-in public observer for a **fresh**
out-of-source CMake/Ninja build and install. It does not reconstruct the history
of cached binaries, enable runtime-slot reuse, or qualify a library by itself.
Existing build commands remain unchanged unless the observer is selected.

For the already-supported unpatched FFTW recipe, select the executable observer
and a fresh existing receipt directory outside the repository/build/install:

```sh
export VIBEQC_NATIVE_BUILD_COLLECTOR="$PWD/scripts/record_native_build.py"
export VIBEQC_NATIVE_RECEIPT_DIR=/absolute/external/empty-receipts
export VIBEQC_NATIVE_SOURCE_ARCHIVE=/absolute/fftw-3.3.10.tar.gz
bash scripts/build_fftw.sh
```

Use the source archive matching the recipe's exact SHA-256 pin. A cache hit stays
a cache hit and produces no fresh-build receipt. `VIBEQC_NATIVE_SEED_CONTEXT=1`
remains a separate opt-in to the conservative Darwin/Apple-clang context resolver
and FFTW adapter. Its existing compiler, SDK, ABI, dependency, source and
final-prefix donor-read-denied probes still apply; this observer bypasses none.
No extra build is needed just to produce a receipt: select it when a fresh build
is already required.

## Record contract

The observer accepts `record --spec SPEC.json --output RECEIPT.json -- COMMAND…`.
The spec has `sources` (name to source directory), `authenticated_sources` (same
names, each with `archive`, `sha256`, and archive `prefix`), `recipes` (file paths),
`cmake_builds` (configured directories), `install`, `artifact_manifest`, `cwd`,
and explicit `data_sources`. Each data entry maps an install-relative destination
to `{ "source": "source-name", "path": "source-relative-file" }`.
FFTW uses its authenticated `api/fftw3.h` as `include/fftw3.h`.
An empty data map explicitly observes no runtime data; it must not be interpreted
as qualification of a library's data requirements.

The `vibe.native-build-receipt/1` output retains pre-build observations and command,
producer identity, return code, and the complete post-install inventory. Only
`status=recorded` with integer `returncode=0` denotes a successful observation.
Failed commands and changed inputs leave a failed receipt; no automatic retry or
unobserved fallback occurs. Source archives must exactly match extracted files
before and after the command. Patches, source-tree generated files, archive links,
and compiler wrappers requiring transitive provenance are not qualified by this
contract. Output files must match the install manifest exactly. Selected installed
data must match its authenticated source bytes and mode.

The receipt explicitly fingerprints six categories in `inputs.semantic_inputs`:

| Category | Observed input |
|---|---|
| `sources_patches` | Authenticated archives and complete extracted inventories; unpatched only |
| `recipe` | Explicit recipe file bytes |
| `toolchain` | Configured compiler executable paths and bytes |
| `options` | CMake cache, compiler commands, Ninja/install graphs, selected build environment |
| `abi` | Observed host system, architecture, OS/libc and byte order |
| `data` | Explicit authenticated source data mapped to installed destinations |

These hashes are **observations, not universal compatibility keys**. They retain
absolute producing paths and are deliberately conservative. The candidate adapter
must independently resolve its context, normalize only justified relocation
roots, attest transitive toolchain/dependency inputs, compare semantics and run
native/data probes in the final prefix with donor reads denied. Merely comparing
compiler versions, six hashes or output hashes is insufficient.

The generic observer can record supplied unpatched library specs; automatic
recipe integration and runtime-slot admission are currently limited to FFTW.
There is no supported libint/OpenBLAS/libecpint/libxc/spglib seed adapter yet.
Their recipe, patch, ABI, data and isolated native-probe contracts still need
implementation. No live dependency reuse is established by protocol tests.

Receipts can contain local paths and environment observations. Keep them outside
product sources, installers and distributable source archives; do not publish
them automatically.

Focused protocol controls (no dependency build):
`python3 scripts/tests/test_record_native_build.py`.
