import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { link } from './base.mjs';
import { products } from '../src/data/products.mjs';

const downloadHtml = await readFile(new URL('../dist/download/index.html', import.meta.url), 'utf8');
const homeHtml = await readFile(new URL('../dist/index.html', import.meta.url), 'utf8');

test('the built site exposes one canonical download destination', () => {
  assert.match(homeHtml, new RegExp(`href="${link('download')}"`));
  assert.match(downloadHtml, /official distribution matrix/);
});

test('inactive candidates and planned channels never become download links', () => {
  assert.doesNotMatch(downloadHtml, /href="[^"]*(?:standalone\.(?:dmg|deb)|vibe-qc-suite|apps\.apple\.com|apps\.microsoft\.com)/);
  assert.doesNotMatch(downloadHtml, /Download coming soon|Candidate SHA-256|vibe-view-2\.9\.0-/);
  assert.match(downloadHtml, /Download vibe-view 2\.20\.19 for Linux/);
  assert.match(downloadHtml, /Download vibe-sketch 0\.1\.4 for Linux/);
  assert.match(downloadHtml, /Corresponding source/);
});

test('each product retains its actual installation and licensing route', () => {
  for (const product of products) {
    assert.ok(downloadHtml.includes(`href="${product.installGuide}"`));
    assert.ok(downloadHtml.includes(`href="${product.licenseGuide}"`));
    if (product.repository) assert.ok(downloadHtml.includes(`href="${product.repository}/releases"`));
  }
  assert.match(downloadHtml, /Linux desktop &amp; corresponding source/);
  assert.doesNotMatch(downloadHtml, /href="https:\/\/github\.com\/vibe-qc\/vibe-sketch/);
});

test('binary distribution plans preserve the source license and availability boundary', () => {
  assert.match(downloadHtml, /purchase links are not yet available/);
  assert.match(downloadHtml, /Public desktop downloads/);
  assert.match(downloadHtml, /Paid Mac App Store/);
  assert.match(downloadHtml, /Paid Microsoft Store/);
  assert.match(downloadHtml, /Building published source does not require a purchase or membership/);
  assert.match(downloadHtml, /QVF remains Apache-2\.0/);
});

test('implemented desktop runtimes are distinct from signed public installers', () => {
  assert.match(downloadHtml, /bundles Python, PyVista, VTK, Trame and RDKit/);
  assert.match(downloadHtml, /bundles Python, RDKit, QVF and the editor/);
  assert.match(downloadHtml, /Linux AppImage also bundles/);
  assert.match(downloadHtml, /do not establish trusted signing, notarization or Store acceptance/);
});
