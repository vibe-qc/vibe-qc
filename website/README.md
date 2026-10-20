# vibe-qc website

Technical website for [vibe-qc](https://vibe-qc.com), built with
[Astro](https://astro.build). Static output — no runtime, no database.

One technical site presents four independent tools: vibe-qc, vibe-sketch,
vibe-view and vibe-queue. The separate vibyverse.com site owns the broader
marketing presentation. Sphinx documentation
is published separately: core under `/docs/`, viewer under `/vibe-view/docs/`,
queue under `/vibe-queue/docs/`, and sketcher under `/vibe-sketch/docs/`. Each repository publishes only its own
subtree. Astro links to those sites and never imports their content.

Technical product pages live under `/products/<product>/`. Do not put them
under `/vibe-view/`, `/vibe-queue/` or `/vibe-sketch/`: those subtrees are excluded from the
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


## Technical content and product availability

The homepage, documentation directory, workflows and product pages prioritize
installation, supported interfaces, method scope and reproducible results.
Keep broader marketing on vibyverse.com. The shared footer links there without
copying its content or establishing a build dependency.

vibe-sketch is a testing version licensed under MPL-2.0. Its manual is public
at `/vibe-sketch/docs/`; source and release URLs remain unset. The product page
links to the manual, local setup instructions for authorized users and the
existing contact form for software access. Do not infer a public source URL from
the product name, claim a desktop package is released, or extend MPL-2.0 to QVF. QVF remains a format project, not a fifth application.

The queued workflow is sketch, submit, calculate, then analyze. Direct local
calculations can skip submission. The homepage, workflow guide and shared
artwork use that sequence.

The methods catalog supports text and status filters. Without JavaScript all
methods remain readable. Mobile navigation, keyboard focus, Escape handling,
reduced motion and the skip-to-content target are part of the shared shell.
Existing download qualification and contact-handler checks remain in force.

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

## Publish

The technical site is decoupled from engine releases. Website publication uses checked source on
`main` and the configured website publication workflow. Manuals can include
explicitly identified source updates newer than released packages. Each publisher owns a separate URL subtree:

| Public path | Content owner | Source |
| --- | --- | --- |
| `/` | vibe-qc | `website/` on `main` |
| `/preview/` | vibe-qc | `website/` on `main`, built with `PREVIEW_BASE=/preview/` |
| `/docs/` | vibe-qc | separately checked and published `docs/` snapshot |
| `/vibe-view/docs/` | vibe-view | its own documentation and release process |
| `/vibe-queue/docs/` | vibe-queue | its own documentation and release process |
| `/vibe-sketch/docs/` | vibe-sketch | its own documentation from committed source |

Ordinary development pushes skip CI under the repository policy. Website-only
publication can use the configured operator workflow after local verification;
do not launch full engine CI for a site edit. The product `.gitlab-ci.yml` contains portable validation jobs and imports the
configured external deployment definition. Credentials, accounts, SSH setup,
job dependencies and destination commands are maintained in private operations
documentation. Do not copy those settings into this checkout or infer the
current deployment job implementation from an old example here.

The root publisher must preserve the independent `/docs/`, `/updates/`,
`/preview/`, `/vibe-view/`, `/vibe-queue/` and `/vibe-sketch/` trees, along with server-managed
configuration and verification files. Companion publishers must restrict
deletion to their own subtree. Source clones or tags do not grant deployment
access.

Before proposing a site change, verify both URL layouts from `website/`:

```sh
npm run verify
PREVIEW_BASE=/preview/ npm run verify
```

The preview is `noindex`; companion and core manual links remain rooted at
their published documentation paths, outside `/preview/`. A manual change on
`main` does not update `/docs/` until the manual is separately built and published.

See [site publishing and shared appearance](../docs/site_publishing.md) for
CSS synchronization and per-product gallery policy. The portable legacy URL
rules are in `deploy/htaccess-redirects.conf` at the repository root; applying
or changing server configuration is an operator task.

## Structure

```
src/
  data/products.mjs         # product ownership, copy, docs and source links
  pages/products/[product].astro # four independently addressable product pages
  layouts/Base.astro         # <html> shell, <head> meta, OG tags
  components/WorkflowFigure.astro # accessible illustrative workflow schematic
  data/distributions.mjs     # download readiness + release coordinates
  data/sponsors.mjs          # tier prices + public recognition records
  pages/index.astro          # technical home and tool directory
  pages/documentation.astro  # manuals, APIs and reference links
  pages/workflows.astro      # task-oriented routes through the tools
  pages/download.astro       # fail-closed distribution matrix
  pages/sponsor.astro        # sponsorship plans, public wall, priorities
  pages/contact.astro        # contact form (posts to public/contact.php)
  pages/contact/sent.astro   # post-submit confirmation page
  styles/global.css          # design tokens + all component styles
public/
  contact.php                # the only server-side file on the site
  sponsors/                  # opt-in local sponsor logos, when present
  logo/                      # generated identities and social previews
```

Content is edited as files in git — add a page by dropping a new
`src/pages/<name>.astro`; it becomes `/<name>/`.

## Maintaining installation and download claims

The download page leads with each product's source installation and manual.
It distinguishes implemented desktop tooling, released source and published
binary artifacts. The current macOS viewer and sketcher package profiles
include their Python runtime; this does not imply trusted signing, Store
acceptance or a public download. Sketch source access remains separately
arranged despite its MPL-2.0 license.

The product distribution contracts describe planned Linux supporter downloads,
paid Mac App Store packages and paid Microsoft Store packages. These channels
must stay labelled planned until actual qualified packages are published.
Building published source requires no membership. Required source and notices
remain accessible without an active membership, and downloaded binaries do not
need runtime activation after membership ends. QVF remains Apache-2.0.

`src/data/distributions.mjs` retains the historical viewer 2.9.0 candidates and
legacy OCI plan for provenance. They are not current offers. The download page
renders package links only when `publicDownloadUrl` or `publicOciImage` accepts
an explicitly available entry. Never add guessed URLs, package versions or
registry digests. A registry image needs its architecture-qualified coordinate,
registry-issued digest and provenance; a desktop package needs its actual
HTTPS download URL after product/channel acceptance. Membership-controlled
packages must use their authorized download service, not a public-storage URL.

Verify actual public release listings and public manual destinations when
updating availability. A source tag is not a binary download. Recheck both
production and preview builds before publishing.

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

## Repository and mirror links

`src/data/repositories.mjs` is the four-project source directory used by the
onboarding, download and product pages. It names GitLab sources and GitHub
mirrors under `github.com/vibe-qc/`. All four GitHub repositories provide public
source snapshots, verified anonymously on 2026-09-15; GitLab requires authorized
access. Public snapshots have separate commit history and can lag GitLab. Check
refs on the selected host and preserve port 26 in GitLab SSH recipes. See
`docs/github_publication.md` for publication policy. Do not infer a package download from
a source tag or copy a companion artifact into this site. QVF is a format
reference, and vibe-basis stays in the core checkout.
