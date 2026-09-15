import assert from 'node:assert/strict';
import test from 'node:test';

import {
  distributionMatrix,
  publicDownloadUrl,
  publicOciImage,
} from '../src/data/distributions.mjs';

const SHA256 = /^[a-f0-9]{64}$/;

test('candidate artifacts retain the supplied filenames and checksums', () => {
  assert.deepEqual(
    distributionMatrix.desktop.map(({ filename, sha256 }) => ({ filename, sha256 })),
    [
      {
        filename: 'vibe-view-2.9.0-macos-arm64-standalone.dmg',
        sha256: 'cec23045f7f522568ba5364b9c08f4eb3e2698ccca11e0e87cb746a3101e4051',
      },
      {
        filename: 'vibe-view-2.9.0-ubuntu-amd64-standalone.deb',
        sha256: '4d5147cd16c812b79da2522a6c8a1ae1d4a22e966d6e05b47162b048507f36b5',
      },
    ],
  );
  for (const artifact of distributionMatrix.desktop) {
    assert.match(artifact.sha256, SHA256);
  }
});

test('the OCI contract uses two explicit architecture-qualified images', () => {
  assert.deepEqual(
    distributionMatrix.oci.images.map(
      ({ ociPlatform, imageReferencePlaceholder }) => ({
        ociPlatform,
        imageReferencePlaceholder,
      }),
    ),
    [
      {
        ociPlatform: 'linux/amd64',
        imageReferencePlaceholder:
          '<registry>/<namespace>/vibe-qc-suite:<version>-amd64',
      },
      {
        ociPlatform: 'linux/arm64/v8',
        imageReferencePlaceholder:
          '<registry>/<namespace>/vibe-qc-suite:<version>-arm64',
      },
    ],
  );
  assert.deepEqual(distributionMatrix.oci.includes, [
    'vibe-qc',
    'vibe-view',
    'vq',
    'vibe-basis (vb)',
    'Offline documentation',
    'Full tutorial course',
    'Examples',
  ]);
});

test('the learning bundle exposes discovery and safe copy workflows', () => {
  assert.deepEqual(distributionMatrix.oci.discoveryCommands, [
    'docs',
    'tutorials',
    'examples',
  ]);
  assert.deepEqual(distributionMatrix.oci.copyCommands, [
    'docs copy /calculations/vibe-qc-docs',
    'examples copy /calculations/vibe-qc-examples',
  ]);
});

test('unqualified candidates cannot produce public links or coordinates', () => {
  for (const image of distributionMatrix.oci.images) {
    assert.equal(publicOciImage(image), null);
  }
  for (const artifact of distributionMatrix.desktop) {
    assert.equal(publicDownloadUrl(artifact), null);
  }
});

test('activation requires an explicit available state and a safe reference', () => {
  const artifact = distributionMatrix.desktop[0];
  assert.equal(
    publicDownloadUrl({
      ...artifact,
      status: 'available',
      downloadUrl: 'http://downloads.example/vibe-view.dmg',
    }),
    null,
  );
  assert.equal(
    publicDownloadUrl({
      ...artifact,
      status: 'available',
      downloadUrl: 'https://downloads.example/vibe-view.dmg',
    }),
    'https://downloads.example/vibe-view.dmg',
  );

  const oci = distributionMatrix.oci.images[0];
  assert.equal(
    publicOciImage({
      ...oci,
      status: 'available',
      imageReference: 'registry.example/vibe-qc/vibe-qc-suite:2.0.0-amd64',
      registryDigest: null,
      registryDigestSource: null,
    }),
    null,
  );
  assert.equal(
    publicOciImage({
      ...oci,
      status: 'available',
      imageReference: 'registry.example/vibe-qc/vibe-qc-suite:2.0.0-arm64',
      registryDigest: 'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      registryDigestSource: 'registry',
    }),
    null,
  );

  assert.equal(
    publicOciImage({
      ...oci,
      status: 'available',
      imageReference: 'registry.example/vibe-qc/vibe-qc-suite:2.0.0-amd64',
      registryDigest: 'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      registryDigestSource: 'local-image-id',
    }),
    null,
  );

  assert.deepEqual(
    publicOciImage({
      ...oci,
      status: 'available',
      imageReference: 'registry.example/vibe-qc/vibe-qc-suite:2.0.0-amd64',
      registryDigest: 'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      registryDigestSource: 'registry',
    }),
    {
      reference: 'registry.example/vibe-qc/vibe-qc-suite:2.0.0-amd64',
      registryDigest: 'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      pinnedReference:
        'registry.example/vibe-qc/vibe-qc-suite:2.0.0-amd64@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    },
  );
});
