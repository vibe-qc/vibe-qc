import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import {
  components,
  readProjectVersion,
  siblingComponents,
  vibeQcVersion,
} from '../src/data/components.mjs';

const REPO_ROOT = new URL('../../', import.meta.url);
const SEMVER = /^\d+\.\d+\.\d+/;

test('every component has either a local version or its own release link', () => {
  assert.deepEqual(
    components.map(({ id }) => id),
    ['vibe-qc', 'vibe-view', 'vq', 'vibe-basis'],
  );
  for (const component of components) {
    if (component.version !== null) assert.match(component.version, SEMVER);
    else assert.match(component.releases, /^https:\/\/github\.com\/vibe-qc\/vibe-(view|queue)\/releases$/);
    assert.ok(component.role.length > 0, `${component.id} role`);
  }
});

test('versions match each component pyproject, not a copied constant', () => {
  const sources = {
    'vibe-qc': 'pyproject.toml',
    'vibe-basis': 'vibe-basis/pyproject.toml',
  };
  for (const component of components.filter(({ version }) => version !== null)) {
    const path = fileURLToPath(new URL(sources[component.id], REPO_ROOT));
    const expected = readProjectVersion(readFileSync(path, 'utf8'));
    assert.equal(component.version, expected, `${component.id} tracks its pyproject`);
  }
});

test('the component version lines are independent of each other', () => {
  // The whole point of listing all four: vibe-qc's 0.15.x does not imply
  // vibe-view's or vq's number. Guards against a future refactor that
  // sources them all from one constant.
  const distinct = new Set(components.map(({ version }) => version));
  assert.ok(distinct.size > 1, 'components must not collapse to one version');
});

test('the engine version and siblings partition the component list', () => {
  assert.equal(vibeQcVersion, components[0].version);
  assert.deepEqual(
    siblingComponents.map(({ id }) => id),
    ['vibe-view', 'vq', 'vibe-basis'],
  );
});

test('readProjectVersion is scoped to the [project] table', () => {
  const toml = [
    '[build-system]',
    'requires = ["hatchling"]',
    'version = "9.9.9"',
    '',
    '[project]',
    'name = "example"',
    '# a comment between name and version, as vibe-basis has',
    'version = "1.2.3"',
    '',
    '[tool.other]',
    'version = "0.0.0"',
  ].join('\n');
  assert.equal(readProjectVersion(toml), '1.2.3');
});

test('readProjectVersion reports absence rather than guessing', () => {
  assert.equal(readProjectVersion('[project]\nname = "example"\n'), null);
  assert.equal(readProjectVersion('[tool.x]\nversion = "1.0.0"\n'), null);
});
