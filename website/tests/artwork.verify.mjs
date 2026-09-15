import assert from 'node:assert/strict';
import { access, readFile } from 'node:fs/promises';
import test from 'node:test';

const base = process.env.PREVIEW_BASE || '/';
const dist = new URL('../dist/', import.meta.url);
const pages = [
  ['index.html', 'vibe-qc-social.png'],
  ['products/vibe-qc/index.html', 'vibe-qc-product-social.png', 'vibe-qc'],
  ['products/vibe-view/index.html', 'vibe-view-social.png', 'vibe-view'],
  ['products/vibe-queue/index.html', 'vibe-queue-social.png', 'vibe-queue'],
];

for (const [path, card, product] of pages) {
  test(`${path} publishes the correct accessible artwork and icons`, async () => {
    const html = await readFile(new URL(path, dist), 'utf8');
    const image = `https://vibe-qc.com${base}logo/${card}`;
    assert.ok(html.includes(`property="og:image" content="${image}"`));
    assert.ok(html.includes(`name="twitter:image" content="${image}"`));
    assert.match(html, /property="og:image:alt" content="[^"]+"/);
    assert.match(html, /name="twitter:image:alt" content="[^"]+"/);
    assert.match(html, /property="og:image:type" content="image\/png"/);
    assert.match(html, /property="og:image:width" content="1200"/);
    assert.match(html, /property="og:image:height" content="630"/);
    await access(new URL(`logo/${card}`, dist));
    for (const [rel, asset] of [
      ['icon', 'vibe-qc-favicon.svg'],
      ['icon', 'vibe-qc-favicon.png'],
      ['apple-touch-icon', 'vibe-qc-apple-touch-icon.png'],
    ]) {
      assert.ok(html.includes(`rel="${rel}" href="${base}logo/${asset}"`));
      await access(new URL(`logo/${asset}`, dist));
    }
    if (product) {
      for (const theme of ['light', 'dark']) {
        const asset = `${product}-wordmark-${theme}.svg`;
        assert.ok(html.includes(`src="${base}logo/${asset}"`));
        await access(new URL(`logo/${asset}`, dist));
      }
    }
  });
}
