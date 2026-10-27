/**
 * Canonical release data for the website download page.
 *
 * Retain legacy candidate metadata for provenance; it is not a current offer.
 * The public page renders only explicitly available package references. A public
 * desktop link requires an explicit available state and a valid HTTPS URL. An
 * OCI image additionally requires its architecture-qualified registry tag,
 * final registry digest, and registry provenance marker. This makes activation
 * a small, reviewable data change and prevents local Docker image IDs from
 * being presented as published registry digests.
 */
export const distributionMatrix = Object.freeze({
  lastReviewed: '2026-10-22',
  oci: Object.freeze({
    id: 'cpu-suite',
    statusLabel: 'Independent qualification pending',
    includes: Object.freeze([
      'vibe-qc',
      'vibe-view',
      'vq',
      'vibe-basis (vb)',
      'Offline documentation',
      'Full tutorial course',
      'Examples',
    ]),
    discoveryCommands: Object.freeze([
      'docs',
      'tutorials',
      'examples',
    ]),
    copyCommands: Object.freeze([
      'docs copy /calculations/vibe-qc-docs',
      'examples copy /calculations/vibe-qc-examples',
    ]),
    images: Object.freeze([
      Object.freeze({
        id: 'cpu-suite-amd64',
        label: 'Linux AMD64',
        architecture: 'AMD64 · x86-64',
        ociPlatform: 'linux/amd64',
        expectedTagSuffix: '-amd64',
        status: 'preview',
        statusLabel: 'Native qualification pending',
        imageReference: null,
        imageReferencePlaceholder:
          '<registry>/<namespace>/vibe-qc-suite:<version>-amd64',
        registryDigest: null,
        registryDigestSource: null,
        gates: Object.freeze([
          'Pass the native Linux AMD64 same-host calculation-performance gate for this exact payload.',
          'Record each component repository and exact tag or commit; publish its corresponding source and retained licenses.',
          'Publish the architecture-qualified :<version>-amd64 registry coordinate.',
          'Record the final digest returned by the registry, not a local Docker image ID.',
        ]),
      }),
      Object.freeze({
        id: 'cpu-suite-arm64',
        label: 'Linux ARM64',
        architecture: 'ARM64 · ARMv8-A',
        ociPlatform: 'linux/arm64/v8',
        expectedTagSuffix: '-arm64',
        status: 'preview',
        statusLabel: 'Native qualification pending',
        imageReference: null,
        imageReferencePlaceholder:
          '<registry>/<namespace>/vibe-qc-suite:<version>-arm64',
        registryDigest: null,
        registryDigestSource: null,
        gates: Object.freeze([
          'Pass the native Linux ARM64 same-host calculation-performance gate for this exact payload.',
          'Record each component repository and exact tag or commit; publish its corresponding source and retained licenses.',
          'Publish the architecture-qualified :<version>-arm64 registry coordinate.',
          'Record the final digest returned by the registry, not a local Docker image ID.',
        ]),
      }),
    ]),
    docs: Object.freeze([
      Object.freeze({
        label: 'Installation guide',
        href: '/docs/installation.html',
      }),
      Object.freeze({
        label: 'Queue guide',
        href: '/vibe-queue/docs/',
      }),
      Object.freeze({
        label: 'License inventory',
        href: '/docs/license.html',
      }),
    ]),
  }),
  desktop: Object.freeze([
    Object.freeze({
  "id": "vibe-view-linux-x64",
  "product": "vibe-view 2.20.19",
  "platform": "Linux",
  "architecture": "x86-64 · AppImage",
  "filename": "vibe-view-2.20.19.AppImage",
  "sha256": "c7382b4c968364a2c969b1c62afaddedb6da5267cf3c15d0bd4849d51e766b81",
  "status": "superseded",
  "statusLabel": "Verified Linux desktop build",
  "downloadUrl": "https://vibe-qc.com/downloads/desktop/vibe-view/2.20.19-linux.2/vibe-view-2.20.19.AppImage",
  "sourceUrl": "https://vibe-qc.com/downloads/desktop/vibe-view/2.20.19-linux.2/vibe-view-linux-source.tar.gz",
  "checksumsUrl": "https://vibe-qc.com/downloads/desktop/vibe-view/2.20.19-linux.2/SHA256SUMS",
  "sizeLabel": "524 MiB",
  "qualification": "Built and tested on Debian 12 x86-64 with an X11 virtual display. Installed package tests passed offline; Python and application libraries are bundled. Requires standard Linux desktop and graphics libraries."
}),
    Object.freeze({
  "id": "vibe-sketch-linux-x64",
  "product": "vibe-sketch 0.3.0",
  "platform": "Linux",
  "architecture": "x86-64 \u00b7 AppImage",
  "filename": "vibe-sketch-0.3.0.AppImage",
  "sha256": "3916fe7e45b20c9934dc6205a6d93a560f3ec08f8608a907d22a53eafe9983d9",
  "status": "superseded",
  "statusLabel": "Public Linux download",
  "downloadUrl": "https://vibe-qc.com/updates/vibe-sketch/linux/0.3.0-linux.1/vibe-sketch-0.3.0.AppImage",
  "sourceUrl": "https://vibe-qc.com/updates/vibe-sketch/linux/0.3.0-linux.1/vibe-sketch-linux-source.tar.gz",
  "checksumsUrl": "https://vibe-qc.com/updates/vibe-sketch/linux/0.3.0-linux.1/SHA256SUMS",
  "sizeLabel": "430 MiB",
  "qualification": "Tested on Debian 12 x86-64 with an X11 virtual display. Includes Python and application libraries; requires standard Linux desktop and graphics libraries."
}),
    Object.freeze({
      id: 'vibe-view-macos-arm64',
      product: 'vibe-view 2.9.0',
      platform: 'macOS',
      architecture: 'Apple Silicon · ARM64',
      filename: 'vibe-view-2.9.0-macos-arm64-standalone.dmg',
      sha256: 'cec23045f7f522568ba5364b9c08f4eb3e2698ccca11e0e87cb746a3101e4051',
      status: 'preview',
      statusLabel: 'Historical unsigned candidate',
      downloadUrl: null,
      qualification:
        'Built and functionally verified on Apple Silicon, but unsigned and not notarized.',
      activation:
        'Retain this historical record. Publish a separate current package only after its product and channel acceptance.',
    }),
    Object.freeze({
      id: 'vibe-view-ubuntu-amd64',
      product: 'vibe-view 2.9.0',
      platform: 'Ubuntu 22.04',
      architecture: 'AMD64 · x86-64',
      filename: 'vibe-view-2.9.0-ubuntu-amd64-standalone.deb',
      sha256: '4d5147cd16c812b79da2522a6c8a1ae1d4a22e966d6e05b47162b048507f36b5',
      status: 'preview',
      statusLabel: 'Historical emulated candidate',
      downloadUrl: null,
      qualification:
        'Historical build verified under emulation; it did not establish native Ubuntu 22.04 AMD64 acceptance.',
      activation:
        'Retain this historical record. Current Linux desktop planning targets an independently qualified AppImage.',
    }),
  ]),
  desktopDocs: Object.freeze([
    Object.freeze({
      label: 'vibe-view source releases (GitHub)',
      href: 'https://github.com/vibe-qc/vibe-view/releases',
    }),
    Object.freeze({
      label: 'vibe-view desktop guide',
      href: '/vibe-view/docs/desktop.html',
    }),
    Object.freeze({
      label: 'Source installation',
      href: '/vibe-view/docs/installation.html',
    }),
  ]),
});

const HTTPS_URL = /^https:\/\/[^\s]+$/;
const OCI_REFERENCE =
  /^[a-z0-9.-]+(?::[0-9]+)?\/[a-z0-9._/-]+:[A-Za-z0-9._-]+$/;
const REGISTRY_DIGEST = /^sha256:[a-f0-9]{64}$/;

export function publicDownloadUrl(artifact) {
  if (
    artifact.status === 'available' &&
    typeof artifact.downloadUrl === 'string' &&
    HTTPS_URL.test(artifact.downloadUrl)
  ) {
    return artifact.downloadUrl;
  }
  return null;
}

export function publicOciImage(image) {
  if (
    image.status !== 'available' ||
    typeof image.imageReference !== 'string' ||
    !OCI_REFERENCE.test(image.imageReference) ||
    !image.imageReference.endsWith(image.expectedTagSuffix) ||
    typeof image.registryDigest !== 'string' ||
    !REGISTRY_DIGEST.test(image.registryDigest) ||
    image.registryDigestSource !== 'registry'
  ) {
    return null;
  }

  return Object.freeze({
    reference: image.imageReference,
    registryDigest: image.registryDigest,
    pinnedReference: `${image.imageReference}@${image.registryDigest}`,
  });
}
