# Self-contained macOS package

The Apple Silicon package includes Python 3.14, the native QC engine, basis
and ECP data, NumPy, SciPy, ASE, symmetry support, RDKit naming, structure
fetching and reference D3/D4 libraries. It needs no Python, pip or Homebrew
installation. This first native profile requires macOS 27 or newer. A lower
minimum requires rebuilding and qualifying the entire native library stack.

Drag `vibe-qc.app` to Applications, then open it to start a Terminal session
with all QC commands available. Type `vibe-qc --help`. From another terminal:

```sh
"/Applications/vibe-qc.app/Contents/MacOS/vibe-qc" --help
```

Calculation inputs and results remain in the directories you choose, outside
the app. The Python executable is at
`Contents/Resources/python/bin/python3`; use `-I -B` for an isolated scripting
session. Launchers remove Python and loader environment overrides, always
use this interpreter and support paths containing spaces. Do not pip-install
into the application; replace the complete app when updating.

The included profile is the core plus `naming`, `fetch` and `dispersion`
extras. Optional MACE/SKALA model stacks, model downloads, the extended BSE
catalogue, TREXIO and development/reference programs are outside this profile.
These are not dependencies of the included core workflows. View, Sketch and
Queue are separate applications. Fetching remote structures requires network
access; ordinary local calculations do not.

## Build

Build on an Apple Silicon Mac with Apple's command-line developer tools,
Python 3.14, CMake, Ninja and the native dependencies prepared by the repository's normal
build instructions. Preserve the normal native cache and build only once per
host. In a clean checkout at the intended source revision:

```sh
python3.14 -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/qc-wheels -Ccmake.define.VIBEQC_RELOCATABLE_RUNTIME=ON .
python3 packaging/macos/build.py --app-wheel /tmp/qc-wheels/vibe_qc-VERSION-cp314-cp314-macosx_27_0_arm64.whl
python3 packaging/macos/build.py --verify dist/macos/vibe-qc.app
python3 packaging/macos/smoke.py dist/macos/vibe-qc.app
```

Substitute the actual wheel filename. The build environment needs the
`pyproject.toml` build requirements; it is never shipped. `--source-revision`
defaults to HEAD and must be the revision used to build that wheel. The
builder checks the installed version, records both source and wheel hashes,
and includes the corresponding tracked source archive. The wheel itself
does not attest which source revision produced its native extension.

`runtime.lock.json` pins the standalone interpreter, all binary wheels,
upstream source supplements and notices. No dependency resolution occurs at
package time. Downloads are verified, then installation runs offline.
`--cache` reuses verified inputs; `--native-prefix` selects the prepared
`third_party` directory. The packaging step recursively copies all non-system
native dependencies, rewrites their load paths and removes build-machine
search paths, including unused Homebrew search directories retained in some
upstream wheels. It builds a cached FFTW copy from the locked source with
portable diagnostic paths, preserving the development dependency installation.
It rejects missing libraries, unexpected dependency locations,
wrong architectures and minimum OS mismatches. Basis data comes from the
wheel, not from a source checkout or a generated build overlay.

Outputs in `dist/macos` are an app, DMG, ZIP and checksum file. These previews
are ad-hoc signed. Developer ID signing, notarization, supported-OS acceptance
and Store qualification are separate release steps. The package is a local
preview, not an approved Mac App Store submission.

## Licenses and source

The application sources retain MPL-2.0. The directly linked FFTW library is
GPL-2.0-or-later; Libint and ASE have LGPL obligations. Charging for a download
does not remove recipients' license rights. A direct package containing FFTW
needs the applicable GPL-compatible combined-distribution terms and complete
corresponding source. This is a separate decision from Store distribution;
do not treat this preview as clearance for Apple's Store terms.

`Contents/Resources/notices` contains the application source archive, the
exact package-building scripts and lock, Python notices, upstream native
source supplements and native build configuration markers.
QC's archived dependency recipes contain the local Libint and Libecpint
patches. Library and wheel notices are preserved. `native-libraries.json`
records input and relocated binary hashes. `runtime.json` records the source
revision, wheel and lock hashes and Python package inventory.

Before public redistribution, reconcile these supplements with the actual
native build records and all linked components, including their exact build
options, modifications and source-delivery obligations. A reused native cache
and a matching version number alone are not a source-provenance attestation.
See [the project license guide](../../docs/license.md).

Git builds remain free. Official website packages may require a supporter
membership; downloaded versions keep working without a subscription check.
This builder does not implement a payment service or publish downloads.

## Optional MACE package

Select `--lock packaging/macos/runtime-mace.lock.json` for a separate
`vibe-qc-mace.app`, ZIP and DMG. The default lock and standard downloads stay
unchanged. This profile uses Python 3.14 and adds MACE, PyTorch and the full
locked dependency graph to the standard scientific runtime. It retains
`setuptools` because PyTorch requires it. Build tools are removed only when
absent from the runtime lock, then the final dependency graph is checked.

The one source-only dependency, python-hostlist, is built offline using locked
build wheels. Its upstream source archive ships with the notices. Model
weights are never included: the selected model downloads to the user's cache
on first use. Run `packaging/smoke_mace.py` with the packaged interpreter and
an empty HOME/XDG cache, then repeat with `--offline` using that same cache.
Also run the normal `smoke.py` qualification on the extracted DMG.

### Standalone naming input

Main revisions that require `vibeqc-naming` must supply `--naming-wheel`
built from that same revision's `vibeqc-naming/` directory. The builder
checks its independent version, pure-Python wheel tag, and every package
file against the selected Git tree, then records its SHA256 in runtime.json.
A missing, changed, wrong-version or duplicate locked naming input refuses.
This input does not modify historical release locks or public artifacts.
The existing source supplement contains the co-located distribution source.
