# vibe-qc website

Marketing / landing site for [vibe-qc](https://vibe-qc.com), built with
[Astro](https://astro.build). Static output — no runtime, no database.

One marketing site presents three independent products. Sphinx documentation
is published separately: core under `/docs/`, viewer under `/vibe-view/docs/`,
and queue under `/vibe-queue/docs/`. Each repository publishes only its own
subtree. Astro links to those sites and never imports their content.

Marketing product pages live under `/products/<product>/`. Do not put them
under `/vibe-view/` or `/vibe-queue/`: those subtrees are excluded from the
root deploy and reserved for their independent publishers.

## Repository ownership

This website remains in [vibe-qc](https://github.com/vibe-qc/vibe-qc),
alongside the engine, examples, docs, and retained `vibe-basis/` toolkit.
[vibe-view](https://github.com/vibe-qc/vibe-view),
[vibe-queue](https://github.com/vibe-qc/vibe-queue) (`vq`), and
[qvf](https://github.com/vibe-qc/qvf) are separate repositories.
The QVF specification is a shared format contract, not a runtime dependency.

Website builds read package versions only from the core and vibe-basis
`pyproject.toml` files here. Companion versions and artifacts belong to their
own releases. Do not read files from assumed companion subdirectories or
assume a core tag selects matching companion tags. The get-started page
shows this repository map and links to the separate installation guides.

## Develop

```sh
cd website
npm install
npm run dev          # http://localhost:4321
```

## Build

```sh
npm run build        # → website/dist/  (static files)
npm run preview      # serve the built site locally
npm run verify       # release-data tests + production build
```

`dist/` is what gets rsynced to the web root in CI (alongside the Sphinx
`/docs/` deploy). Both are plain static file trees.

## Deploy

The marketing site is **decoupled from version releases** — copy and fixes ship
from `main` on their own. The CI jobs (see `../.gitlab-ci.yml`):

- **`website-deploy`** → the production root (`example-web-account@vibe-qc.com:/web/`). Runs
  **automatically on every `main` push that changes `website/`**, and is also
  available as a manual job on `main` to force a redeploy. `needs: []` runs it
  immediately, gated only by its own `npm run build` — independent of the C++
  test suite. `--delete-after` excludes `/docs/`, `/updates/`, `/preview/`,
  `/vibe-view/`, `/vibe-queue/`, and
  the server-managed cPanel/Apache files (incl. `.htaccess`).
- **`website-staging`** → `/web/preview/` (**https://vibe-qc.com/preview/**,
  `noindex`). Same `main` + `website/`-changed trigger, built with
  `PREVIEW_BASE=/preview/`. A parallel preview of the same main push for reviewing the built site.

The **docs** are versioned, so they deploy differently:

- **`docs-deploy`** → `/web/docs/`. Runs **only on `release`**, which the
  release owner fast-forwards from a tag on `main`. A docs edit on `main`
  waits for the next release; manual main pipelines may build docs but cannot
  publish them over the released site.

See [site publishing and shared appearance](../docs/site_publishing.md) for
companion ownership, protected refs, CSS synchronization, and gallery policy.

All jobs `--delete-after` within their own subtree only, and share the SSH setup
via the `.deploy_ssh` `!reference` template. To reproduce a staging build
locally: `PREVIEW_BASE=/preview/ npm run build`.

**One-time server-side step (already applied 2026-07-17):** legacy root doc URLs
(`/quickstart.html`, `/_static/…`) are 301-redirected to `/docs/` by Apache
rewrite rules in `../deploy/htaccess-redirects.conf`. CI deploys deliberately
exclude `.htaccess`, so those rules live in `/web/.htaccess` on the server and
persist across deploys. **Re-apply only if the URL structure changes again.**

## Structure

```
src/
  data/products.mjs         # product ownership, copy, docs and source links
  pages/products/[product].astro # three independently addressable product pages
  layouts/Base.astro         # <html> shell, <head> meta, OG tags
  components/OrbitalField.astro  # animated MO iso-contour hero canvas
  data/distributions.mjs     # download readiness + release coordinates
  data/sponsors.mjs          # tier prices + public recognition records
  pages/index.astro          # the landing page (edit content here)
  pages/download.astro       # fail-closed distribution matrix
  pages/sponsor.astro        # sponsorship plans, public wall, priorities
  pages/contact.astro        # contact form (posts to public/contact.php)
  pages/contact/sent.astro   # post-submit confirmation page
  styles/global.css          # design tokens + all component styles
public/
  contact.php                # the only server-side file on the site
  sponsors/                  # opt-in local sponsor logos, when present
  favicon.svg                # placeholder — swap for the official mark
```

Content is edited as files in git — add a page by dropping a new
`src/pages/<name>.astro`; it becomes `/<name>/`.

## Activating distribution downloads

`src/data/distributions.mjs` is the single source of truth for the download
page. Its viewer 2.9.0 filenames and checksums are retained pre-split candidate
records. They are not current viewer releases. Newly validated desktop packages
must come from the [vibe-view release page](https://github.com/vibe-qc/vibe-view/releases).
Candidate filenames and checksums can be recorded without exposing an active link. The rendering helpers fail closed unless an entry has both an
explicit `status: 'available'` and a valid public location.

The OCI contract is two explicit images, not a combined multi-platform
manifest:

- `vibe-qc-suite:<version>-amd64` for Linux AMD64;
- `vibe-qc-suite:<version>-arm64` for Linux ARM64.

Planned images combine independently versioned vibe-qc, vibe-view, vq, and
the retained vibe-basis (`vb`) toolkit with offline documentation, tutorials,
and examples. Qualification must record each repository's exact tag or commit
and the corresponding licenses; there is no shared monorepo release tree.
The container entrypoint exposes `docs`, `tutorials`, and `examples` discovery,
plus `docs copy /calculations/vibe-qc-docs` and
`examples copy /calculations/vibe-qc-examples` for writable copies.

To activate an OCI image:

1. Complete that architecture's native Linux same-host calculation-performance
   gate for the exact payload.
2. Record each component repository and exact tag or commit, then publish
   its corresponding source and retained licenses.
3. Publish the explicit architecture-qualified tag.
4. Record both the real registry coordinate and the final digest returned by
   the registry in the matching `oci.images` entry. Set
   `registryDigestSource` to `registry` and `status` to `available`.

Both entries are activated independently. Never use a local Docker image ID as
`registryDigest`; the helper deliberately requires registry provenance.

To activate a desktop package:

1. Finish its release gate. The macOS DMG must be signed and notarized; the
   Ubuntu DEB must be rebuilt and verified on native Ubuntu 22.04 AMD64.
2. Upload the final artifact to its real HTTPS release location.
3. Replace the candidate checksum if the final file differs, set
   `downloadUrl`, change `status` to `available`, and update its status and
   qualification copy to describe the final release.
4. Run `npm run verify` and inspect the generated `dist/download/index.html`
   before allowing the normal website deployment job to publish the page.

Never add a guessed URL, registry coordinate, or digest. A preview entry with a
null location remains visible but non-downloadable by design.

## Maintaining sponsor recognition

`src/data/sponsors.mjs` is the single source of truth for tier prices, payment
readiness, and the public sponsor wall. Platinum, Gold, and Silver are monthly
plans designed to use one GitHub Sponsors destination. Supporter recognition
covers any contribution and remains on the permanent roll after a recurring
plan ends.

Monthly checkout currently fails closed because the configured GitHub Sponsors
profile is not public. To activate it:

1. Publish matching $250 Platinum, $100 Gold, and $25 Silver monthly tiers in
   GitHub Sponsors.
2. Verify the sponsor profile and each tier from a signed-out browser.
3. Set `monthlyCheckout.status` to `available` in `src/data/sponsors.mjs`.
4. Run `npm run verify`; the tier buttons and homepage route will then use the
   one configured checkout URL.

Recognition is opt-in. Add a person or organization only after receiving the
public display name they want used. Active monthly records belong under the
matching `activeSponsors` tier; all opted-in contributors also belong in
`supporters`. When a plan ends, remove its active-tier record but retain its
supporter entry.

Logo display is optional. Store an authorized PNG, SVG, or WebP file under
`public/sponsors/` and use a relative `sponsors/<file>` path in the record.
External logo URLs are rejected so the public page does not introduce
third-party tracking. Website links must use HTTPS. Run `npm run verify` before
publishing a recognition update.

## Contact form

`/contact/` posts to `public/contact.php`, which forwards the message to
**mpei@vibe-qc.com**. That PHP file is the *only* server-side code on the site —
everything else is static. Astro copies `public/` verbatim into `dist/`, so the
existing `website-deploy` rsync ships it with no extra build or CI step, and no
third-party form service ever sees a visitor's message.

Three files, which have to agree:

| File | Role |
|---|---|
| `src/pages/contact.astro` | the form; renders handler errors from `?error=<code>` |
| `src/pages/contact/sent.astro` | the success page the handler redirects to |
| `public/contact.php` | validation, spam gates, `mail()` |

The happy path needs no JavaScript: the form is a plain POST and the handler
answers with a 303 redirect to a real pre-rendered page. JS only stamps the
fill-time field and renders the error banner. `tests/contact.verify.mjs` pins
the contract, including that the topic allowlist is byte-identical on both
sides — the handler rejects anything not in *its* copy, so drift fails closed.

### One-time host verification

**Done 2026-08-06** — the host runs PHP-FPM (PHP 8.5) with SuEXEC, the
handler executes, and a real submission was delivered and replied to. Redo
this only if the hosting changes. The handler assumes things about the web
host that this repo cannot test:

1. **PHP is enabled for the document root.** If it is not, Apache serves
   `contact.php` as plain text and every submission silently fails. Check
   first — `curl -sI https://vibe-qc.com/contact.php` should return a 303
   redirect, not `Content-Type: text/plain`.
2. **`mail()` can hand off to the host MTA.** A failed handoff is logged via
   `error_log` and returns the visitor to `/contact/?error=send`.
3. **`FROM_ADDRESS` (`website@vibe-qc.com`) exists as a real mailbox** on the
   sending domain. It is deliberately *not* the visitor's address: putting a
   visitor's address in `From` breaks SPF/DKIM alignment and gets the mail
   filtered. The visitor goes in `Reply-To`, so replying still reaches them.
4. **Send one real test message** and confirm it arrives (check spam), that
   reply-to works, and that a reload of the success page does not resend.

Until step 1 is confirmed, the sidebar's plain `mailto:` link is the working
route — it stays visible on the page by design, so a broken handler never
leaves a visitor without a way to make contact.

### Spam handling

Deliberately modest, because the volume does not justify a captcha or a
third-party service: a hidden honeypot field, a fill-time heuristic (only when
JS ran), a per-IP rate limit of 5/hour in a temp-dir counter file, and hard
length caps. Honeypot and fill-time trips answer with the ordinary success page
so an automated submitter learns nothing. Every field that reaches a mail header
is stripped of CR/LF first — that is what stops the form becoming an open relay,
and it is pinned by a test.

To change the destination address, edit `TO_ADDRESS` in `public/contact.php`;
`tests/contact.verify.mjs` asserts it, so the test moves with it.

## Design

Visual identity is the molecular-orbital **phase pair** (blue positive lobe /
red negative lobe). Tokens live at the top of `src/styles/global.css`; the
palette is theme-aware (light/dark via `prefers-color-scheme` + a
`data-theme` override hook). Type pairs a heavy system grotesque with a
monospace label system.
