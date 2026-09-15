/** The base prefix the build under test was produced with.
 *
 * astro.config.mjs reads PREVIEW_BASE, so one source tree builds twice: the
 * production root (`/`) that website-deploy rsyncs to vibe-qc.com, and the
 * `/preview/` subtree that website-staging publishes. Every internal link
 * moves with the base, so an assertion that hard-codes `href="/sponsor"`
 * passes against one build and fails against the other. Seven of them did,
 * silently, for as long as the suite ran nowhere in CI.
 *
 * Import `base` or `link()` instead of writing the leading slash by hand.
 *
 * Links that deliberately do NOT move: the separately published doc subtrees
 * (/docs/, /vibe-view/docs/, /vibe-queue/docs/) belong to other publishers and
 * stay absolute in both builds, which is why products.verify.mjs asserts that
 * they are never base-doubled.
 */
export const base = process.env.PREVIEW_BASE || '/';

const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** A base-prefixed internal path, escaped for embedding in a RegExp. */
export const link = (path) => escapeRegExp(`${base}${path}`);
