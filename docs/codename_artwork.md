# Upcoming vibe-qc release artwork

This is the vibe-qc gallery. vibe-view and vibe-queue own separate release
galleries and contact sheets; see the [gallery policy](site_publishing.md#codename-galleries).

Artwork prepared for the five concrete upcoming releases in the [roadmap](roadmap.md),
from v0.18.0 through v1.0.0. All five images are AI-generated 1672 x 941 PNGs.
These are planning assets: the codenames remain candidates or provisional
proposals, and do not register new runtime codenames or mark a release shipped.

```{admonition} The version pairings below shifted at the v0.17.0 cut
:class: warning

v0.17.0 shipped symmetry and periodic foundations rather than the native BvK
route that held the number, so the milestone ladder moved up one (see the
v0.17.0 section of the [roadmap](roadmap.md)).
Tew's Tern was assigned to v0.17.0 and has moved to the released gallery; its
artwork is a lattice/paired-orbital treatment of the route whose groundwork
that release shipped.

The five proposals below still carry the version numbers they were generated
against. Each name was chosen for a *milestone* -- Berkelbach for periodic CC,
Runge for TDDFT, Stone for response properties, Amdahl for scaling -- and
those milestones each moved up one, so the numbers printed here are one behind
their milestones. Re-pairing them, and naming v0.18.0 (native BvK periodic
local correlation), are maintainer decisions and are deliberately not made
here.
```

Runge's Raccoon and Stone's Sloth use candidates already listed in the roadmap.
The other three pairings are provisional artwork proposals. The broad post-v1.0
and v2.x research tracks have no remaining concrete release sequence assigned;
this collection covers the numbered upcoming milestones rather than assigning
new release dates or versions to those tracks.

Each image uses one established series treatment: light studio with teal and
violet molecular motifs, or dark cinematic with blue and amber particle networks.
The scientific motifs are artistic metaphors for the planned capabilities.
The [generation manifest](_static/images-codenames/upcoming/prompts.json)
records each final prompt, version, candidate status, and generator.

## v0.18.0 "Berkelbach's Badger"

Periodic coupled cluster. **Provisional artwork proposal.**

```{figure} _static/images-codenames/upcoming/18-vibe-qc-v0.18.0-berkelbachs-badger.png
:alt: AI-generated Berkelbach's Badger proposed codename artwork for vibe-qc v0.18.0
:width: 100%

v0.18.0 "Berkelbach's Badger" - planned artwork.
```

[Download the PNG](_static/images-codenames/upcoming/18-vibe-qc-v0.18.0-berkelbachs-badger.png).

## v0.19.0 "Runge's Raccoon"

Multireference and excited-state extensions. **Roadmap candidate.**

```{figure} _static/images-codenames/upcoming/19-vibe-qc-v0.19.0-runges-raccoon.png
:alt: AI-generated Runge's Raccoon proposed codename artwork for vibe-qc v0.19.0
:width: 100%

v0.19.0 "Runge's Raccoon" - planned artwork.
```

[Download the PNG](_static/images-codenames/upcoming/19-vibe-qc-v0.19.0-runges-raccoon.png).

## v0.20.0 "Stone's Sloth"

Properties suite. **Roadmap candidate.**

```{figure} _static/images-codenames/upcoming/20-vibe-qc-v0.20.0-stones-sloth.png
:alt: AI-generated Stone's Sloth proposed codename artwork for vibe-qc v0.20.0
:width: 100%

v0.20.0 "Stone's Sloth" - planned artwork.
```

[Download the PNG](_static/images-codenames/upcoming/20-vibe-qc-v0.20.0-stones-sloth.png).

## v0.21.0 "Amdahl's Axolotl"

Production MPI and large-system scaling. **Provisional artwork proposal.**

```{figure} _static/images-codenames/upcoming/21-vibe-qc-v0.21.0-amdahls-axolotl.png
:alt: AI-generated Amdahl's Axolotl proposed codename artwork for vibe-qc v0.21.0
:width: 100%

v0.21.0 "Amdahl's Axolotl" - planned artwork.
```

[Download the PNG](_static/images-codenames/upcoming/21-vibe-qc-v0.21.0-amdahls-axolotl.png).

## v1.0.0 "Schrödinger's Llama"

Molecular and periodic feature completeness. **Provisional artwork proposal.**

```{figure} _static/images-codenames/upcoming/22-vibe-qc-v1.0.0-schrodingers-llama.png
:alt: AI-generated Schrödinger's Llama proposed codename artwork for vibe-qc v1.0.0
:width: 100%

v1.0.0 "Schrödinger's Llama" - planned artwork.
```

[Download the PNG](_static/images-codenames/upcoming/22-vibe-qc-v1.0.0-schrodingers-llama.png).

## House style

The series has a fixed specification. It is recorded here because it is
permanent: it applies to every future codename image, not to any one release.
It was previously carried only in a per-release image brief in the release
drop-box, which is deleted once a cut completes.

**Canvas.** 1672 x 941 px (16:9), PNG, roughly 2 MB. Every image in the series
uses exactly this.

**Two treatments. Pick one; do not blend.**

*A. Light studio* (v0.14 Bartlett's Goose, v0.15 Neese's Cheetah, v0.16
Pople's Puffin, v0.17 Tew's Tern). Near-white or pale-blue seamless
background, soft even studio light. The photoreal animal is on a glossy white
surface carrying a ball-and-stick molecular lattice: silver-white atoms, thin
grey bonds, translucent teal orbital lobes, occasional violet triangular
motifs. Part of the animal's natural patterning dissolves into a hexagonal
lattice or orbital texture in teal and violet. Gentle reflections. Calm,
clinical, product-shot feel.

*B. Dark cinematic* (v0.9 Knowles's Kingfisher). Deep navy-black background.
Photoreal animal mid-motion, trailing edge dissolving into glowing particle
networks. Electric blue and warm amber accents, wireframe orbital lobes, faint
lattice cubes and flowing dotted field lines. Dramatic, high contrast.

**Common to both.**

- The animal is unmistakably real and anatomically correct, never stylised or
  cartoon. Species detail matters: a puffin's bill is drab and wrong out of
  breeding season, and getting it wrong defeats the image.
- No text, no lettering, no logos, no watermark anywhere in the frame.
- Leave the left third relatively empty. The documentation page crops and
  overlays there.
- The scientific motif is an artistic metaphor for the release's capability,
  never a literal diagram.

**Negative prompt, as a starting point.** text, watermark, logo, cartoon,
illustration, stylised, low detail, winter or off-season plumage, cluttered
background.

**Once an image exists.**

1. Save it as
   `docs/_static/images-codenames/NN-vibe-qc-vX.Y.Z-codename-slug.png`,
   continuing the existing numbering. Prepared but unassigned artwork lives in
   the `upcoming/` subdirectory until its release is cut.
2. Record the final prompt, version, candidate status and generator in
   [`prompts.json`](_static/images-codenames/upcoming/prompts.json).
3. Reference it from the homepage release admonition in `docs/index.md`, with
   alt text in the established form: "AI-generated <Codename> codename artwork
   for vibe-qc vX.Y".
4. Add a tile to `vibe-qc-codenames-contact-sheet.png`. Its geometry, measured
   from the sheet itself so it does not have to be re-derived: canvas
   1908 x 1022 on background `#171C20`; tiles 360 x 203, five per row, column
   origins x = 18 + 378i, row origins y = 18 + 251j; captions left-aligned to
   the tile in Arial 18 at `#E6ECF2`, with their ink top 15 px below the tile
   (rows 233, 484, 735, 986). Resize the source PNG to 360 x 203 and paste;
   regenerating the whole sheet is not necessary and risks disturbing the
   existing tiles.
5. Run `docs-build` before pushing. `tests/test_docs_download_links.py` guards
   `_static/downloads/`, not this directory, so no test needs updating.
