# Website sources and shared appearance

The Astro marketing site lives in `website/`; the manual lives in `docs/`.
Build and review them locally using their contributor instructions. Publication
accounts, SSH configuration, host keys and deployment runbooks are maintained
in a private operations repository. The supported GitLab jobs retain their
existing destination limits and release gates.

The core manual follows qualified releases. Website publication follows its
existing protected-main gate. Companion products publish independently. See
[release process](release_process.md) for the contributor-facing contract.

## Shared Furo CSS export

The source of truth is **`docs/_static/custom.css` in vibe-qc**. It is a
small Furo override file, currently providing codename image styling and
leaving Furo's stock palette intact. The Astro marketing stylesheet is a
separate surface; copying this file does not reproduce the marketing header.

Companions deliberately vendor this one file into their own
`docs/_static/custom.css`. Each companion owns its copy and chooses when to
update it. Its normal build uses that committed copy, with no network fetch
and no runtime or build dependency on a theme package.

When adopting or updating the export:

1. Select an immutable core tag or full commit SHA and obtain the file from
   that revision. Copy it byte for byte, preserving its license and comments.
2. Commit a receipt alongside it, such as `docs/_static/custom.css.source.json`,
   recording the source repository, path, full commit SHA, optional tag,
   SHA-256 of the copied bytes, and synchronization date. Do not put a mutable
   `main` URL alone in the receipt. The core repository's MPL-2.0 license
   applies to this file.
3. Configure Furo with `html_static_path = ['_static']` and
   `html_css_files = ['custom.css']`. Put product-specific additions in a
   separate stylesheet loaded afterwards, keeping the shared copy unchanged.
4. Review the upstream diff since the previous receipt and build the
   companion docs locally. Check light and dark themes, narrow layouts,
   navigation, and a codename image. Commit the copy and receipt together,
   then publish through that product's release process.

Core CSS changes must be noted in the CHANGELOG and contributor notes,
with the source revision and a short description of what companions should
review. Companion owners compare that revision with their receipt; updates
are deliberate release work, not automatic drift. A core release does not
silently change either companion's already published appearance.

A possible later refinement is a versioned artifact containing the CSS and
header/footer partials at a stable public URL, with checksums and immutable
version URLs. Companion builds could fetch a deliberately pinned version.
This is a proposal only: no endpoint, package, or build fetch is implemented.

## Codename galleries

**Galleries are per product.** Each repository owns its release images,
gallery page, and contact sheet, and publishes them under its own docs path.
The vibe-qc homepage and contact sheet contain only vibe-qc releases.
vibe-view and vibe-queue start their own series and numbering; their version
numbers and release dates need not match the engine's.

All products use the 1672 x 941 PNG canvas. Filenames include the product,
version, and codename; alt text names the product and version and identifies
AI-generated artwork. Keep generation provenance with the owning assets.
Release owners approve a codename before promoting its artwork to the
released gallery. Update that product's contact sheet at the same time.

The six [upcoming vibe-qc images](codename_artwork.md), v0.17.0 through v1.0.0,
remain candidate or provisional planning assets. They do not announce shipped
releases. A future marketing family gallery may link the three product
galleries, with explicit product labels; it should not merge their version
sequences or duplicate their image collections.
