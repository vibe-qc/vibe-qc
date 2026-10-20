/** Verified split repositories. GitHub snapshots are public as of 2026-09-15;
 * refs and release artifacts must be checked on the host used for installation.
 * No sibling checkout or network request is needed to build this directory.
 */
export const repositories = Object.freeze([
  { id: 'vibe-qc', name: 'vibe-qc',
    role: 'The calculation engine, examples and documentation. vibe-basis stays inside this checkout.' },
  { id: 'vibe-view', name: 'vibe-view',
    role: 'The independent viewer. Install and run it from its own checkout and environment.' },
  { id: 'vibe-queue', name: 'vibe-queue (vq)',
    role: 'The independent scheduler. Its command and Python import remain vq; its distribution is vibe-queue.' },
  { id: 'vibe-sketch', name: 'vibe-sketch', private: true,
    role: 'The structure and reaction sketcher. Testing version; MPL-2.0, with access arranged through the maintainer.' },
  { id: 'qvf', name: 'QVF',
    role: 'The format specification, schema, corpus and optional reference toolkit. The engine and basic viewer implement the format independently; strict viewer validation and sketch document exchange use optional toolkit integrations.' },
].map(repository => Object.freeze({
  ...repository,
  github: repository.private ? null : `https://github.com/vibe-qc/${repository.id}`,
})));
