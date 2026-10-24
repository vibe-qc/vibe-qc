import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { base, link } from './base.mjs';
import { products } from '../src/data/products.mjs';
const homeHtml = await readFile(new URL('../dist/index.html', import.meta.url), 'utf8');

test('the technical homepage exposes four tools and the documentation routes', () => {
  assert.equal(products.length, 4);
  for (const p of products) assert.ok(homeHtml.includes(`href="${base}products/${p.id}/"`));
  for (const page of ['documentation/', 'workflows/', 'methods/', 'get-started', 'download', 'sponsor']) {
    assert.ok(homeHtml.includes(`href="${link(page)}"`), page);
  }
  assert.match(homeHtml, /href="https:\/\/vibyverse.com"/);
  assert.match(homeHtml, /Illustrative schematic/);
  assert.doesNotMatch(homeHtml, /machine precision|THREE INDEPENDENT|three products/i);
});

test('desktop downloads are visible and unqualified artifacts are not linked', () => {
  assert.match(homeHtml, /Downloads and installation/);
  assert.doesNotMatch(homeHtml, /href="[^"]*(?:vibe-qc-suite|standalone\.(?:dmg|deb))/);
});

test('queued work is sketched, submitted, calculated and analyzed in that order', async () => {
  const figure = homeHtml.match(/<div class="figure-steps">([\s\S]*?)<\/div>/)?.[1];
  assert.ok(figure);
  const targets = [...figure.matchAll(/href="[^"]*\/products\/([^/]+)\//g)].map(match => match[1]);
  assert.deepEqual(targets, ['vibe-sketch', 'vibe-queue', 'vibe-qc', 'vibe-view']);
  const workflow = await readFile(new URL('../dist/workflows/index.html', import.meta.url), 'utf8');
  const sections = [...workflow.matchAll(/class="workflow-row" id="([^"]+)"/g)].map(match => match[1]);
  assert.deepEqual(sections, ['prepare', 'schedule', 'calculate', 'explore']);
  assert.match(workflow, /For a direct local run, skip submission/);
});

test('published versions are visible without JavaScript in workflow order', async () => {
  const releases = JSON.parse(await readFile(new URL('../src/data/published-releases.json', import.meta.url),'utf8'));
  for (const [id, release] of Object.entries(releases)) {
    assert.ok(homeHtml.includes(`v${release.version}`),id);
    const product = await readFile(new URL(`../dist/products/${id}/index.html`,import.meta.url),'utf8');
    assert.ok(product.includes(`v${release.version}`),id);
  }
  assert.match(homeHtml,/own version line/);
  assert.doesNotMatch(homeHtml,/not a new release announcement|planned binary channels/);
});
