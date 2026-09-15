/** Versions come only from packages in this checkout. Separately released
 * companions link to their own release pages; no sibling checkout is needed.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = new URL('../../../', import.meta.url);

/**
 * Read `version` from the `[project]` table of a pyproject.toml.
 *
 * Deliberately table-scoped rather than a whole-file search: several
 * components carry a `version` key in other tables (build-system pins,
 * tool config), and vibe-basis places comment lines between `name` and
 * `version`. Anchoring to the `[project]` table and stopping at the next
 * top-level table keeps those from being picked up.
 */
export function readProjectVersion(tomlText) {
  const lines = tomlText.split(/\r?\n/);
  let inProject = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('#')) continue;

    // A new top-level table starts here.
    if (/^\[[^[]/.test(trimmed)) {
      if (inProject) break; // left [project] without finding a version
      inProject = trimmed === '[project]';
      continue;
    }

    if (!inProject) continue;

    const match = /^version\s*=\s*["']([^"']+)["']/.exec(trimmed);
    if (match) return match[1];
  }

  return null;
}

function versionOf(relativePath) {
  const path = fileURLToPath(new URL(relativePath, REPO_ROOT));
  const version = readProjectVersion(readFileSync(path, 'utf8'));
  if (version === null) {
    throw new Error(`no [project] version found in ${relativePath}`);
  }
  return version;
}

/**
 * The four installable components, in the order the docs present them
 * (`docs/toolset_lifecycle.md` "At a glance"): the chemistry engine first,
 * then the tools built around it.
 */
export const components = Object.freeze([
  Object.freeze({
    id: 'vibe-qc',
    name: 'vibe-qc',
    distribution: 'vibe-qc',
    role: 'quantum chemistry engine',
    version: versionOf('pyproject.toml'),
  }),
  Object.freeze({
    id: 'vibe-view',
    name: 'vibe-view',
    distribution: 'vibeview',
    role: 'visualization and QVF viewer',
    version: null,
    releases: 'https://github.com/vibe-qc/vibe-view/releases',
  }),
  Object.freeze({
    id: 'vq',
    name: 'vq',
    distribution: 'vq',
    role: 'cross-machine job queue',
    version: null,
    releases: 'https://github.com/vibe-qc/vibe-queue/releases',
  }),
  Object.freeze({
    id: 'vibe-basis',
    name: 'vibe-basis',
    distribution: 'vibe-basis',
    role: 'basis-set toolkit',
    version: versionOf('vibe-basis/pyproject.toml'),
  }),
]);

/** The engine version, still the headline number for the release band. */
export const vibeQcVersion = components[0].version;

/** Everything except vibe-qc, for pages that headline the engine separately. */
export const siblingComponents = Object.freeze(
  components.filter((component) => component.id !== 'vibe-qc'),
);
