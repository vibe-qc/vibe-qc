import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { link } from './base.mjs';

const downloadHtml = await readFile(
  new URL('../dist/download/index.html', import.meta.url),
  'utf8',
);
const homeHtml = await readFile(new URL('../dist/index.html', import.meta.url), 'utf8');

test('the built site exposes one canonical download destination', () => {
  assert.match(homeHtml, new RegExp(`href="${link('download')}"`));
  assert.match(downloadHtml, /official distribution matrix/);
});

test('the built preview contains no live candidate artifact link', () => {
  assert.doesNotMatch(
    downloadHtml,
    /href="[^"]*(?:standalone\.(?:dmg|deb)|vibe-qc-suite)/,
  );
  assert.equal(downloadHtml.match(/Historical candidate/g)?.length, 2);
  assert.match(downloadHtml, /href="https:\/\/github\.com\/vibe-qc\/vibe-view\/releases"/);
  assert.equal(
    downloadHtml.match(/Placeholder only; no qualified registry image exists yet\./g)
      ?.length,
    2,
  );
});

test('the built preview retains qualification metadata', () => {
  assert.match(downloadHtml, /linux\/amd64/);
  assert.match(downloadHtml, /linux\/arm64\/v8/);
  assert.match(downloadHtml, /vibe-qc-suite:&lt;version&gt;-amd64/);
  assert.match(downloadHtml, /vibe-qc-suite:&lt;version&gt;-arm64/);
  assert.match(
    downloadHtml,
    /cec23045f7f522568ba5364b9c08f4eb3e2698ccca11e0e87cb746a3101e4051/,
  );
  assert.match(
    downloadHtml,
    /4d5147cd16c812b79da2522a6c8a1ae1d4a22e966d6e05b47162b048507f36b5/,
  );
  assert.match(downloadHtml, /unsigned and not notarized/);
  assert.match(downloadHtml, /native Ubuntu 22\.04 AMD64 release build/);
});

test('both images include the offline learning and copy workflow', () => {
  assert.match(downloadHtml, /Offline documentation/);
  assert.match(downloadHtml, /Full tutorial course/);
  assert.match(downloadHtml, /docker run --rm &quot;\$IMAGE&quot; docs/);
  assert.match(downloadHtml, /docker run --rm &quot;\$IMAGE&quot; tutorials/);
  assert.match(downloadHtml, /docker run --rm &quot;\$IMAGE&quot; examples/);
  assert.match(downloadHtml, /docs copy \/calculations\/vibe-qc-docs/);
  assert.match(downloadHtml, /examples copy \/calculations\/vibe-qc-examples/);
});

test('the page states independent per-image release gates', () => {
  assert.match(downloadHtml, /there is no combined manifest tag/i);
  assert.equal(
    downloadHtml.match(/Local Docker image IDs are not release digests\./g)?.length,
    2,
  );
  assert.match(downloadHtml, /native Linux AMD64 same-host/);
  assert.match(downloadHtml, /native Linux ARM64 same-host/);
  assert.match(downloadHtml, /final digest returned by the registry/);
});
