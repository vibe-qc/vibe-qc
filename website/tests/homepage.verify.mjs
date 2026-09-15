import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { link } from './base.mjs';

const homeHtml = await readFile(new URL('../dist/index.html', import.meta.url), 'utf8');

test('the homepage describes the gated distribution without exposing artifacts', () => {
  assert.match(homeHtml, /Distribution preview/);
  assert.match(homeHtml, /0 of 4 qualified/);
  assert.match(homeHtml, /Qualification pending/);
  assert.match(homeHtml, /Candidates gated/);
  assert.doesNotMatch(homeHtml, /vibe-qc-suite:&lt;version&gt;/);
  assert.doesNotMatch(homeHtml, /standalone\.(?:dmg|deb)/);
});

test('the homepage distinguishes both container and desktop platforms', () => {
  assert.match(homeHtml, /Architecture-qualified Linux images/);
  assert.match(homeHtml, /linux\/amd64/);
  assert.match(homeHtml, /linux\/arm64\/v8/);
  assert.match(homeHtml, /vibe-view for desktop/);
  assert.match(homeHtml, /pre-split candidates are historical records/);
  assert.match(homeHtml, /Apple Silicon · ARM64/);
  assert.match(homeHtml, /Ubuntu 22\.04/);
});

test('distribution calls to action converge on the canonical download page', () => {
  assert.match(
    homeHtml,
    new RegExp(`href="${link('download')}"[^>]*>\\s*Distribution preview`),
  );
  assert.match(homeHtml, /Review container gates/);
  assert.match(homeHtml, /Review desktop candidates/);
  assert.doesNotMatch(homeHtml, /href="[^"]*(?:vibe-qc-suite|standalone\.(?:dmg|deb))/);
});

test('the homepage previews every sponsor tier and permanent supporter roll', () => {
  for (const tier of ['Platinum', 'Gold', 'Silver']) {
    assert.match(homeHtml, new RegExp(`>${tier}<`));
  }
  for (const amount of [250, 100, 25]) {
    assert.match(homeHtml, new RegExp(`>\\$${amount}<`));
  }
  assert.match(homeHtml, /permanent opt-in roll/i);
  assert.match(homeHtml, new RegExp(`href="${link('sponsor')}"`));
  assert.match(homeHtml, /href="https:\/\/ko-fi\.com\/mpeintinger"/);
  assert.doesNotMatch(homeHtml, /href="https:\/\/github\.com\/sponsors\/mpeintinger"/);
});

test('the component band distinguishes local versions from companion releases', async () => {
  const { components } = await import('../src/data/components.mjs');
  assert.match(homeHtml, /Engine and companion projects/i);
  const renderedVersions = [...homeHtml.matchAll(/<span class="cmp-version"[^>]*>([^<]+)<\/span>/g)].map(match => match[1]);
  for (const component of components) {
    if (component.version !== null) {
      assert.ok(renderedVersions.includes(component.version));
    } else {
      assert.ok(homeHtml.includes(`href="${component.releases}"`));
    }
    assert.match(homeHtml, new RegExp(component.role));
  }
  // The independence claim is the reason the list exists; keep it visible.
  assert.match(homeHtml, /own version line/i);
});
