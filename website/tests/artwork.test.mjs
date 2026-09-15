import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import sharp from 'sharp';
import test from 'node:test';

const asset = name => new URL(`../public/logo/${name}`, import.meta.url);
for (const kind of ['vibe-qc', 'vibe-view', 'vibe-queue']) {
  test(`${kind} wordmarks and icons are self-contained, renderable SVGs`, async () => {
    for (const suffix of ['wordmark-light', 'wordmark-dark', 'favicon']) {
      const svg = await readFile(asset(`${kind}-${suffix}.svg`), 'utf8');
      assert.match(svg, /<title/);
      assert.match(svg, /<desc/);
      assert.doesNotMatch(svg, /<text\b|<script\b|<image\b|<foreignObject\b|(?:href|src)\s*=/i);
      const metadata = await sharp(Buffer.from(svg)).metadata();
      assert.equal(metadata.height, 48);
      assert.ok(metadata.width >= 48);
      await sharp(Buffer.from(svg)).png().toBuffer();
    }
  });
}
/** ProductWordmark.astro hard-codes each wordmark's intrinsic width so the
 * browser reserves the right box before the SVG loads. That number is only
 * correct as long as it matches what generate-brand.py actually emitted, and
 * the width changes whenever a product's displayed name changes: renaming the
 * queue wordmark from `vq` to `vibe-queue` moved it from 115 to 294. Nothing
 * connected the two, so this compares them.
 */
test('the component reserves each wordmark its real intrinsic width', async () => {
  const component = await readFile(
    new URL('../src/components/ProductWordmark.astro', import.meta.url),
    'utf8',
  );
  const declared = Object.fromEntries(
    [...component.matchAll(/'([a-z-]+)':\s*(\d+)/g)].map(([, k, v]) => [k, Number(v)]),
  );
  assert.deepEqual(Object.keys(declared).sort(), ['vibe-qc', 'vibe-queue', 'vibe-view']);
  for (const [kind, width] of Object.entries(declared)) {
    for (const theme of ['light', 'dark']) {
      const svg = await readFile(asset(`${kind}-wordmark-${theme}.svg`), 'utf8');
      const intrinsic = Number(/<svg[^>]*\bwidth="(\d+)"/.exec(svg)[1]);
      assert.equal(intrinsic, width, `${kind}-${theme}: SVG is ${intrinsic}, component says ${width}`);
    }
  }
});

for (const name of ['vibe-qc-social', 'vibe-qc-product-social', 'vibe-view-social', 'vibe-queue-social']) {
  test(`${name} has an outlined SVG and a 1200x630 PNG`, async () => {
    const svg = await readFile(asset(`${name}.svg`), 'utf8');
    assert.doesNotMatch(svg, /<text\b|<script\b|<image\b|<foreignObject\b|(?:href|src)\s*=/i);
    for (const extension of ['svg', 'png']) {
      const { width, height } = await sharp(await readFile(asset(`${name}.${extension}`))).metadata();
      assert.deepEqual([width, height], [1200, 630]);
    }
  });
}
test('the Apple touch icon is an opaque 180px PNG', async () => {
  const meta = await sharp(await readFile(asset('vibe-qc-apple-touch-icon.png'))).metadata();
  assert.equal(meta.format, 'png');
  assert.deepEqual([meta.width, meta.height], [180, 180]);
  assert.equal(meta.hasAlpha, false);
});
test('docs artwork matches the website masters for each page', async () => {
  for (const name of ['vibe-qc-wordmark-light.svg', 'vibe-qc-wordmark-dark.svg', 'vibe-qc-favicon.svg']) {
    assert.deepEqual(await readFile(asset(name)), await readFile(new URL(`../../docs/_static/logo/${name}`, import.meta.url)));
  }
  for (const [source, target] of [
    ['vibe-qc-product-social', 'vibe-qc-social'],
    ['vibe-qc-social', 'vibe-qc-family-social'],
    ['vibe-view-social', 'vibe-view-social'],
    ['vibe-queue-social', 'vibe-queue-social'],
  ]) {
    for (const extension of ['svg', 'png']) {
      assert.deepEqual(await readFile(asset(`${source}.${extension}`)),
        await readFile(new URL(`../../docs/_static/logo/${target}.${extension}`, import.meta.url)));
    }
  }
});
