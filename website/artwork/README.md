# Website brand artwork

The source of truth is `website/scripts/generate-brand.py`. Its outlined SVG
masters live in `website/public/logo/`; do not edit generated assets by hand.
The established wave crossing a unit-cell frame remains the core mark.
The viewer's lens and the queue's connected jobs share its frame and line weight.
The queue's public wordmark reads **vq**; its repository remains vibe-queue.

## Assets

| Asset | Formats | Use |
| --- | --- | --- |
| `*-wordmark-light` / `*-wordmark-dark` | SVG, 48-unit height, tight width | Light/dark backgrounds respectively |
| `*-favicon` | SVG + 32 px PNG | Browser icon or compact product mark |
| `vibe-qc-apple-touch-icon` | Opaque 180 px PNG | Apple home-screen icon |
| `vibe-qc-social` | 1200 x 630 SVG + PNG | Three-product marketing homepage |
| `vibe-qc-product-social` | 1200 x 630 SVG + PNG | Core product page |
| `vibe-view-social` | 1200 x 630 SVG + PNG | Viewer product page |
| `vibe-queue-social` | 1200 x 630 SVG + PNG | Queue product page |

Use PNGs in social metadata for broad preview-client support. The SVG masters
have titles/descriptions, no scripts, no external resources and no live fonts.
Artwork motifs are illustrative, not physical structure or workflow diagrams.

Open [the preview gallery](preview.html) through a local HTTP server to review
the light/dark wordmarks and all four social cards together.

## Rebuild

From the repository root, with the website's npm dependencies installed:

```sh
python3 -m venv /tmp/vibe-artwork-venv
/tmp/vibe-artwork-venv/bin/pip install -r website/artwork/requirements.txt
/tmp/vibe-artwork-venv/bin/python website/scripts/generate-brand.py
node website/scripts/gen-social-card.mjs
npm --prefix website run verify
```

Generation is offline and deterministic with the pinned font, fontTools and
sharp. `npm run artwork:svg` and `npm run artwork:raster` are equivalent entry
points from website/, when Python's fontTools is already available.
The Manrope source and OFL notice are in [fonts/](fonts/README.md); only its
outlined documents are published. Typography uses weight 650 for wordmarks
and 700 for headlines. Its license does not change the core code license.

The raster export step also updates the core assets in `docs/_static/logo/`.
The Sphinx social card uses the engine-specific artwork, while the marketing
homepage uses the family card. The docs homepage displays a separate copy
named `vibe-qc-family-social.svg`; the viewer and queue integration guides use
their product SVGs, with PNGs for their social metadata. All of these copies
are refreshed by the same raster/export command.

Docs assets deploy with the next core release;
website assets deploy from main. Companion owners can copy the relevant
assets deliberately and record the source commit in their own handovers;
there is no runtime or build dependency between repositories.

Preserve aspect ratios, keep clear space of at least half the icon width,
and do not recolor the wordmarks ad hoc. Teal is the family/core accent,
violet identifies the viewer, and amber identifies the queue on this website.
