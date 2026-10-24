import assert from 'node:assert/strict';
import { readFile, access } from 'node:fs/promises';
import test from 'node:test';
import { products } from '../src/data/products.mjs';
import { base } from './base.mjs';

const read = path => readFile(new URL(`../dist/${path}`, import.meta.url), 'utf8');
const QVF_REPOSITORY = 'https://github.com/vibe-qc/qvf';

for (const product of products) {
  test(`${product.name} has an independent product page and owner links`, async () => {
    const html = await read(`products/${product.id}/index.html`);
    assert.ok(html.includes(product.headline));
    if (product.docs) assert.ok(html.includes(`href="${product.docs}"`));
    if (product.repository) assert.ok(html.includes(`href="${product.repository}/releases"`));
    if (product.installGuide) assert.ok(html.includes(`href="${product.installGuide}"`));
    for (const [label, href] of product.guides) {
      assert.ok(html.includes(`href="${href}"`), `${product.name} lost ${label}`);
      if (base !== '/') assert.ok(!html.includes(`href="${base}${href.slice(1)}"`));
    }
    assert.match(html, /id="install"/);
    assert.doesNotMatch(html, /<iframe/);
    for (const other of products) {
      assert.ok(html.includes(`href="${base}products/${other.id}/"`));
    }
    if (base !== '/') {
      assert.match(html, /name="robots" content="noindex"/);
      assert.ok(html.includes(`https://vibe-qc.com${base}products/${product.id}/`));
      if (product.docs) assert.ok(!html.includes(`href="${base}${product.docs.slice(1)}"`));
    }
  });
}

test(`the homepage and onboarding expose all ${products.length} product pages`, async () => {
  for (const path of ['index.html', 'get-started/index.html']) {
    const html = await read(path);
    for (const product of products) {
      assert.ok(html.includes(`href="${base}products/${product.id}/"`));
    }
  }
});

/** QVF is a format specification, not an installable product, so it belongs in
 * the repository and format tiers and never in `products`. The membership rule
 * and the reasoning live in `src/data/products.mjs`. This guards both
 * directions: a stray format entry and a dropped format link both fail here.
 */
test('QVF stays a format and a repository, never a product', async () => {
  assert.ok(!products.some(product => /qvf/i.test(product.id)));
  await assert.rejects(access(new URL('../dist/products/qvf/index.html', import.meta.url)));

  // Onboarding lists it among the repositories; each product page links the format.
  const pages = ['get-started/index.html', ...products.map(p => `products/${p.id}/index.html`)];
  for (const path of pages) {
    assert.ok((await read(path)).includes(QVF_REPOSITORY), `${path} lost its QVF link`);
  }
  assert.ok((await read('index.html')).includes('QVF'));
});

test('technical site output never occupies the companion publisher subtrees', async () => {
  for (const name of ['vibe-view', 'vibe-queue', 'vibe-sketch']) {
    await assert.rejects(access(new URL(`../dist/${name}/index.html`, import.meta.url)));
    await assert.rejects(access(new URL(`../dist/${name}/docs/index.html`, import.meta.url)));
  }
});

test('the top navigation links directly to every published manual', async () => {
  const pages = ['index.html', 'download/index.html', 'get-started/index.html',
    'sponsor/index.html', 'contact/index.html', '404.html',
    ...products.map(product => `products/${product.id}/index.html`)];
  for (const path of pages) {
    const html = await read(path);
    const nav = html.match(/<nav\b[^>]*id="site-links"[^>]*>([\s\S]*?)<\/nav>/)?.[1];
    assert.ok(nav, `${path} is missing its top navigation`);
    for (const product of products) {
      const label = `${product.id === 'vibe-queue' ? 'vq' : product.name} docs`;
      if (product.docs) assert.ok(nav.includes(`href="${product.docs}"`), `${path} lost ${label}`);
      if (product.docs) assert.ok(nav.includes(label), `${path} lost the ${label} label`);
      assert.ok(nav.includes(`href="${base}products/${product.id}/"`));
      if (base !== '/' && product.docs) {
        assert.ok(!nav.includes(`href="${base}${product.docs.slice(1)}"`));
      }
    }
  }
});

test('the public sketch manual and Linux desktop downloads are linked', async () => {
  const html = await read('products/vibe-sketch/index.html');
  assert.match(html, /Testing version/);
  assert.match(html, /MPL-2\.0/);
  assert.doesNotMatch(html, /vibe-sketch is (?:currently )?proprietary/);
  assert.match(html, /macOS &amp; Linux downloads and source/);
  assert.ok(html.includes(`${base}download/#published-packages`));
  for (const path of ['index.html', 'get-started/index.html', 'download/index.html', 'documentation/index.html', 'products/vibe-sketch/index.html']) {
    const content = await read(path);
    assert.doesNotMatch(content, /href="(?:null|undefined)|href="https:\/\/github.com\/vibe-qc\/vibe-sketch/);
  }
  for (const path of ['documentation/index.html', 'products/vibe-sketch/index.html']) {
    const content = (await read(path)).match(/<main\b[\s\S]*?<\/main>/)?.[0];
    assert.ok(content?.includes('href="/vibe-sketch/docs/"'), `${path} lost its manual link`);
    assert.ok(content?.includes('href="/vibe-sketch/docs/installation.html"'));
    assert.ok(content?.includes('href="/vibe-sketch/docs/quickstart.html"'));
    assert.ok(content?.includes('href="/vibe-sketch/docs/inputs.html"'));
    assert.ok(content?.includes('href="/vibe-sketch/docs/python_api.html"'));
    assert.doesNotMatch(content, /public manual is not published|Public documentation and source downloads are not available/i);
  }
});
