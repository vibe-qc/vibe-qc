import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { link } from './base.mjs';
import releases from '../src/data/published-releases.json' with {type:'json'};
import { products } from '../src/data/products.mjs';

const downloadHtml = await readFile(new URL('../dist/download/index.html', import.meta.url), 'utf8');
const homeHtml = await readFile(new URL('../dist/index.html', import.meta.url), 'utf8');

test('the built site exposes one canonical download destination', () => {
  assert.match(homeHtml, new RegExp(`href="${link('download')}"`));
  assert.match(downloadHtml, /Software \/ current releases/);
});

test('inactive candidates and planned channels never become download links', () => {
  assert.doesNotMatch(downloadHtml, /href="[^"]*(?:standalone\.(?:dmg|deb)|vibe-qc-suite|apps\.apple\.com|apps\.microsoft\.com)/);
  assert.doesNotMatch(downloadHtml, /Download coming soon|Candidate SHA-256|vibe-view-2\.9\.0-/);
  assert.doesNotMatch(downloadHtml, /href="[^"]*downloads\/desktop\//);
  for (const product of ['vibe-sketch', 'vibe-queue', 'vibe-qc', 'vibe-view']) {
    const release = releases[product];
    for (const artifact of Object.values(release.manifest.artifacts))
      assert.ok(downloadHtml.includes(`href="${artifact.url}"`));
    assert.ok(downloadHtml.includes(`/updates/${product}/releases/${release.version}/SHA256SUMS`));
  }
  assert.match(downloadHtml, /Source releases on GitHub/);
});

test('each product retains its actual installation and licensing route', () => {
  for (const product of products) {
    assert.ok(downloadHtml.includes(`href="${product.installGuide}"`));
    assert.ok(downloadHtml.includes(`href="${product.licenseGuide}"`));
    if (product.repository) assert.ok(downloadHtml.includes(`href="${product.repository}/releases"`));
  }
  assert.match(downloadHtml, /Standalone downloads/);
  assert.doesNotMatch(downloadHtml, /href="https:\/\/github\.com\/vibe-qc\/vibe-sketch/);
});

test('binary distribution plans preserve the source license and availability boundary', () => {
  assert.match(downloadHtml, /purchase links are not yet available/);
  assert.match(downloadHtml, /Public website downloads/);
  assert.match(downloadHtml, /not Apple-notarized/);
  assert.match(downloadHtml, /Paid Microsoft Store/);
  assert.match(downloadHtml, /Building published source does not require a purchase or membership/);
  assert.match(downloadHtml, /QVF remains Apache-2\.0/);
});

test('implemented desktop runtimes are distinct from signed public installers', () => {
  assert.match(downloadHtml, /bundles Python, PyVista, VTK, Trame and RDKit/);
  assert.match(downloadHtml, /bundles Python, RDKit, QVF and the editor/);
  assert.match(downloadHtml, /vibe-queue includes Python, its CLI, daemon and web console/);
  assert.match(downloadHtml, /do not establish trusted signing, notarization or Store acceptance/);
});

test('desktop versions and workflow order are present in the initial HTML', async () => {
  const releases = JSON.parse(await readFile(new URL('../src/data/published-releases.json',import.meta.url),'utf8'));
  for(const product of ['vibe-sketch','vibe-queue','vibe-qc','vibe-view']) {
    assert.ok(downloadHtml.includes(`v${releases[product].version}`));
    assert.ok(downloadHtml.includes(`id="${product}"`));
  }
  const order = ['vibe-sketch','vibe-queue','vibe-qc','vibe-view'];
  for(let i=1;i<order.length;i++) assert.ok(downloadHtml.indexOf(`id="${order[i-1]}"`) < downloadHtml.indexOf(`id="${order[i]}"`));
  assert.doesNotMatch(downloadHtml, /Version details load from the release publisher/);
});

test('native and CLI packages describe their actual platform requirements', () => {
  assert.match(downloadHtml, /vibe-qc requires macOS 27/);
  assert.match(downloadHtml, /standalone archives/);
  assert.match(downloadHtml, /glibc 2\.36/);
  assert.match(downloadHtml, /bin\/vq|commands in bin\//);
  assert.match(downloadHtml, /Optional package: vibe-qc \+ MACE/);
});

test('the core download card uses the published core manual', () => {
  assert.doesNotMatch(downloadHtml, /href="\/vibe-qc\/docs\/"/);
  assert.match(downloadHtml, /href="\/docs\/"/);
});

test('MACE remains a separate optional download with first-use weights', async () => {
 const release=JSON.parse(await readFile(new URL('../src/data/published-mace-release.json',import.meta.url),'utf8'));
 for(const artifact of Object.values(release.artifacts)) assert.ok(downloadHtml.includes(`href="${artifact.url}"`));
 assert.match(downloadHtml,/Model weights download separately on first use/);
 assert.match(downloadHtml,/CPU-only PyTorch/);
 assert.ok(downloadHtml.includes(releases['vibe-qc'].manifest.artifacts.linux.url));
});
