/** Split product repository directory. Check the selected repository
 * for published source tags and packages; no sibling checkout is needed.
 */
export const repositories = Object.freeze([
  { id: 'vibe-qc', name: 'vibe-qc',
    role: 'The calculation engine, examples and documentation. vibe-basis stays inside this checkout.' },
  { id: 'vibe-view', name: 'vibe-view',
    role: 'The independent viewer. Install and run it from its own checkout and environment.' },
  { id: 'vibe-queue', name: 'vibe-queue (vq)',
    role: 'The independent scheduler. Its command and Python import package remain vq.' },
  { id: 'qvf', name: 'QVF',
    role: 'The format specification, schema, corpus and optional reference toolkit. No runtime dependency for the engine or viewer.' },
].map(repository => Object.freeze({
  ...repository,
  gitlab: `https://github.com/vibe-qc/${repository.id}`,
  github: `https://github.com/vibe-qc/${repository.id}`,
})));
