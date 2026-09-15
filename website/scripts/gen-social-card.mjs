// Raster companions for the outlined SVG masters. Run from any directory:
// node website/scripts/gen-social-card.mjs
// SVG generation: python website/scripts/generate-brand.py
import sharp from 'sharp';
import { copyFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const assets = new URL('../public/logo/', import.meta.url);
const path = name => fileURLToPath(new URL(name, assets));
for (const stem of ['vibe-qc-social', 'vibe-qc-product-social', 'vibe-view-social', 'vibe-queue-social']) {
  await sharp(path(`${stem}.svg`), { density: 144 }).resize(1200, 630).png().toFile(path(`${stem}.png`));
}
for (const kind of ['vibe-qc', 'vibe-view', 'vibe-queue']) {
  await sharp(path(`${kind}-favicon.svg`)).resize(32, 32).png().toFile(path(`${kind}-favicon.png`));
}
await sharp(path('vibe-qc-favicon.svg')).resize(180, 180).flatten({ background: '#0F766E' }).png().toFile(path('vibe-qc-apple-touch-icon.png'));
// Core Sphinx receives the same core identity, with its own engine social card.
const docs = new URL('../../docs/_static/logo/', import.meta.url);
for (const name of ['vibe-qc-wordmark-light.svg', 'vibe-qc-wordmark-dark.svg', 'vibe-qc-favicon.svg']) {
  await copyFile(path(name), new URL(name, docs));
}
for (const extension of ['svg', 'png']) {
  await copyFile(path(`vibe-qc-product-social.${extension}`), new URL(`vibe-qc-social.${extension}`, docs));
}
for (const extension of ['svg', 'png']) {
  await copyFile(path(`vibe-qc-social.${extension}`), new URL(`vibe-qc-family-social.${extension}`, docs));
  for (const product of ['vibe-view', 'vibe-queue']) {
    await copyFile(path(`${product}-social.${extension}`), new URL(`${product}-social.${extension}`, docs));
  }
}
console.log('Exported social PNGs, favicons, Apple touch icon and core docs assets.');
