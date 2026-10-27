import assert from 'node:assert/strict';
import { access, readFile } from 'node:fs/promises';
import test from 'node:test';
import { allMethods, methodAreas, STATUS } from '../src/data/methods.mjs';
import { base, link } from './base.mjs';

const html = await readFile(new URL('../dist/methods/index.html', import.meta.url), 'utf8');
const docsRoot = new URL('../../docs/', import.meta.url);

const escapeHtml = (value) => value
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

test('every method row carries one of the four documented labels', () => {
  const labels = Object.keys(STATUS);
  for (const method of allMethods) {
    assert.ok(labels.includes(method.status), `${method.name}: unknown status ${method.status}`);
    assert.ok(method.note.length > 20, `${method.name}: needs a one-sentence caveat`);
  }
  // Each label must be explained on the page, not only used.
  for (const key of labels) {
    assert.match(html, new RegExp(`<span class="status ${key}[^"]*"[^>]*>${STATUS[key].label}</span>`));
    assert.ok(html.includes(escapeHtml(STATUS[key].meaning)), `legend lost the meaning of ${key}`);
  }
});

test('the rendered page shows every family with its label and note', () => {
  for (const method of allMethods) {
    assert.ok(html.includes(escapeHtml(method.name)), `missing row: ${method.name}`);
    assert.ok(html.includes(escapeHtml(method.note)), `missing note: ${method.name}`);
  }
  const rendered = [...html.matchAll(/<li class="method-row[^"]*" data-status="([a-z]+)"/g)].map((m) => m[1]);
  assert.equal(rendered.length, allMethods.length);
  for (const key of Object.keys(STATUS)) {
    assert.equal(rendered.filter((s) => s === key).length, allMethods.filter((m) => m.status === key).length);
  }
});

test('experimental and gated families are labelled as such, never as production', () => {
  // The manual keeps these lines experimental; the site must not upgrade them.
  const mustNotBeProduction = [/AICCM/, /SECCM/, /GAPW/, /GFN2-xTB/, /PM6/, /OM1/, /SKALA/, /Surface embedding/, /MPI grid overlay/, /Periodic RIJCOSX/, /^DMRG$/];
  for (const pattern of mustNotBeProduction) {
    const rows = allMethods.filter((m) => pattern.test(m.name));
    assert.ok(rows.length > 0, `no row matches ${pattern}`);
    for (const row of rows) {
      assert.notEqual(row.status, 'production', `${row.name} must not be labelled production`);
    }
  }
  assert.equal(allMethods.find((m) => /PM7/.test(m.name)).status, 'gated');
});

test('every docs link resolves to a page in the manual and is never base-prefixed', async () => {
  for (const method of allMethods) {
    assert.match(method.docs, /^\/docs\//, `${method.name}: docs link must be absolute under /docs/`);
    assert.ok(html.includes(`href="${method.docs}"`), `${method.name}: link not rendered`);
    if (base !== '/') assert.ok(!html.includes(`href="${base}${method.docs.slice(1)}"`));
    const page = method.docs.replace(/^\/docs\//, '').replace(/#.*$/, '').replace(/\.html$/, '.md');
    await access(new URL(page, docsRoot));
  }
});

test('the page links the manual anchors that explain the vocabulary and the learning paths', () => {
  assert.ok(html.includes('href="/docs/features.html"'));
  assert.ok(html.includes('href="/docs/experimental/catalog.html"'));
  assert.ok(html.includes('href="/docs/tutorial/index.html#choose-a-learning-path"'));
  assert.ok(html.includes('href="/docs/validation.html"'));
  assert.match(html, new RegExp(`href="${link('get-started')}"`));
  for (const area of methodAreas) {
    assert.match(html, new RegExp(`id="${area.id}"`));
  }
  if (base !== '/') assert.match(html, /name="robots" content="noindex"/);
});
