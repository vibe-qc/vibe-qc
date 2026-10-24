# Distribution builds

vibe-qc publishes independently versioned source releases and public standalone
macOS and Linux downloads. The [downloads page](https://vibe-qc.com/download/#vibe-qc)
shows the current version, packages, matching source and SHA-256 checksums.
Downloads do not require a supporter membership or runtime activation.
Windows Store editions remain planned and are not currently available.

Covered application source remains MPL-2.0. Bundled dependencies retain their
own licenses, notices and corresponding source where required. Building from
published source does not require a purchase.

## Package profiles

| Platform | Initial architecture | Release format |
|---|---|---|
| Linux | x86-64 | portable tar.gz with Python, the native engine, required libraries and basis/ECP data |
| macOS | arm64 | Public DMG containing a self-contained Terminal application |
| Windows | x86-64 | Planned Microsoft Store package; not currently downloadable |

Linux CLI archives work without a desktop, FUSE or a system Python installation.
They target x86-64 systems compatible with Debian 12 or newer. Extract the archive
and run `bin/vibe-qc --help`. On macOS, open the DMG, copy the app to
Applications, then launch it to open Terminal with the bundled commands ready.
The Apple Silicon DMG is ad-hoc signed and is not Apple-notarized.

The standalone package includes Python, the native calculation engine, basis/ECP data, naming, structure
fetching and D3/D4. Optional machine-learning models and developer/reference
tools remain separate. macOS requires version 27 or newer.

Each current release is published as a qualified macOS/Linux pair. Versioned
packages are immutable. Download a replacement package when a newer release is
listed; source and dependency notices remain publicly accessible.

## Build and acceptance contract

Use a native runner for each supported platform. Pin the product source revision,
Python runtime, dependencies, frontend lockfile and native toolchain. Build the
same candidate that passed the existing release CI; do not substitute a later
main revision or an old wheel bearing the same version. Reuse dependency caches
only when their platform, architecture, ABI and input hashes match.

Build recipes and non-secret locks must be included with the product source so
community builds do not require the publisher's infrastructure or credentials.
Credentials, private operations scripts, test records and deployment configuration
stay outside product packages and source snapshots.

Linux ships a relocatable runtime archive. Native Windows engine/toolchain support must be established before Windows packaging. Store editions need a useful calculation interface and a resolved engine licensing/sandbox design. The current macOS Terminal launcher is a preview.

Run installed-package acceptance on the oldest advertised OS in a clean account
without Python, Node, pip or development tools. Load bundled basis/ECP data and run a representative engine calculation with no source tree or external Python. Numerical-method changes retain their normal scientific validation requirements.
Also test paths with spaces/non-ASCII, offline local operation, immutable install
contents, backend shutdown, update recovery and uninstall without deleting user
documents. Inventory licenses and native dependencies from the actual payload.

The native macOS and Linux downloads are qualified separately. Their
[matching source supplement](https://vibe-qc.com/updates/vibe-qc/current/source.tar.gz)
contains the application source and packaging recipes. Passing package checks
does not establish trusted signing, notarization or Store acceptance.

## Promotion

Release in this order: exact candidate source and CI, native build, installed
acceptance, channel packaging/signing, final checksums and immutable staging,
then channel publication and delivered-package verification. Give each package
revision an increasing build number; never overwrite previously released bytes.
Keep Store updates separate from website updates. A failure in one product or
platform need not hold a qualified release on another.

Mac App Store submissions require Apple identities, provisioning, appropriate
sandbox entitlements for the app and embedded helpers, and a Store-compatible
application. Electron applications use the MAS variant. A notarized DMG is a
different distribution route. See [Electron's submission guide](https://www.electronjs.org/docs/latest/tutorial/mac-app-store-submission-guide).

Microsoft Store submissions use the assigned Partner Center identity, declared
capabilities and an MSIX containing the full runtime. Qualify installed behavior
and run the Windows App Certification Kit. Microsoft signs Store-distributed
packages; local sideload testing uses its own development certificate. See
[Microsoft's signing guide](https://learn.microsoft.com/en-us/windows/msix/package/sign-msix-package-guide).

Public website downloads and source supplements do not use supporter entitlement
checks. Channel accounts, secrets, release evidence and provisioning belong to
the publisher's external operating process, not this repository.
