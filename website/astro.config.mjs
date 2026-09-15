// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

// Staging builds set PREVIEW_BASE=/preview/ so the site can be served at
// https://vibe-qc.com/preview/ (a rolling preview of `main`). Production
// builds leave it unset → served at the domain root. Astro's `base`
// prefixes all bundled asset + page URLs; the favicon link and the
// noindex flag in Base.astro key off import.meta.env.BASE_URL to match.
const base = process.env.PREVIEW_BASE || '/';

// vibe-qc marketing site.
//
// This site owns the domain ROOT (https://vibe-qc.com/). The Sphinx
// documentation is a *separate* static build that is deployed under
// /docs/ on the same host — the two are stitched together by the web
// server, not by Astro. Keep it that way: never import or vendor the
// docs build here.
export default defineConfig({
  site: 'https://vibe-qc.com',
  base,
  // The marketing site lives at the domain root in production (base '/').
  // The docs live at /docs/ and are built by Sphinx (see docs/conf.py
  // html_baseurl). Staging overrides base to '/preview/' (see above).
  //
  // Auto-generate the marketing sitemap on build (→ /sitemap-index.xml).
  // The docs ship their own sitemap under /docs/; the root robots.txt
  // (public/robots.txt) lists both.
  integrations: [sitemap()],
  build: {
    // Emit foo/index.html so pretty URLs work behind a plain static
    // file server (matches how the Sphinx output is served).
    format: 'directory',
  },
});
