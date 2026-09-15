# `docs/_static/`

Static assets copied into the built HTML site. On the public core docs site,
they are served under `/docs/_static/`.

## Brand artwork

The source of truth is `website/scripts/generate-brand.py`. Its outlined SVG
masters live in `website/public/logo/`; `website/scripts/gen-social-card.mjs`
exports PNGs and copies the docs assets here. See `website/artwork/README.md`
for the pinned font license, regeneration commands and companion adoption.
Do not hand-edit these generated copies.

| File in `logo/` | Use |
| --- | --- |
| `vibe-qc-favicon.svg` | Teal browser icon, configured in `conf.py` |
| `vibe-qc-wordmark-light.svg` / `vibe-qc-wordmark-dark.svg` | Furo sidebar, selected by theme |
| `vibe-qc-social.svg` / `.png` | Core engine artwork; the PNG is the docs homepage social preview |
| `vibe-qc-family-social.svg` / `.png` | Three-product artwork displayed on the docs homepage |
| `vibe-view-social.svg` / `.png` | Viewer integration guide image and social preview |
| `vibe-queue-social.svg` / `.png` | Queue integration guide image and social preview |

All social artwork uses a 1200 x 630 canvas. SVG lettering is outlined, so no
browser font download is required. These are illustrative brand images,
not scientific result figures. Release codename artwork remains separate
under `images-codenames/` on its established 1672 x 941 canvas.

## Theme

`custom.css` contains the Furo overrides and image presentation rules,
activated by `html_css_files` in `conf.py`. It is a deliberate copy export
for companion docs; follow the receipt process in `docs/site_publishing.md`.
Core documentation changes are published with the next approved core release.
