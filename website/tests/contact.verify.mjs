import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { link } from './base.mjs';

// Post-build checks for the contact form. The form page and the success
// page are static Astro output; the handler is plain PHP copied verbatim
// out of public/. All three have to agree, and none of it is exercised by
// a build error, so it is pinned here.

const read = (p) => readFile(new URL(p, import.meta.url), 'utf8');

const formHtml = await read('../dist/contact/index.html');
const sentHtml = await read('../dist/contact/sent/index.html');
const handler = await read('../dist/contact.php');
const homeHtml = await read('../dist/index.html');

test('the handler is deployed alongside the static pages', () => {
  // public/ is copied verbatim into dist/, and dist/ is what rsyncs to the
  // web root — so being in dist/ is what makes the endpoint reachable.
  assert.match(handler, /^<\?php/);
  assert.match(
    formHtml,
    new RegExp(`<form[^>]+method="post"[^>]+action="${link('contact.php')}"`),
  );
});

test('the form collects every field the handler requires', () => {
  for (const name of ['name', 'email', 'topic', 'message']) {
    assert.match(formHtml, new RegExp(`name="${name}"[^>]*`), `missing field: ${name}`);
  }
  assert.match(formHtml, /<input[^>]+type="email"[^>]+name="email"[^>]+required/);
  assert.match(formHtml, /<textarea[^>]+name="message"[^>]+required/);
});

test('mail is forwarded to the maintainer address', () => {
  assert.match(handler, /const TO_ADDRESS = 'mpei@vibe-qc\.com';/);
});

test('the visitor address is used for Reply-To, never for From', () => {
  // Putting a visitor's address in From breaks SPF/DKIM alignment and gets
  // the mail filtered. From must stay on the sending domain.
  assert.match(handler, /'From: ' \. display_name\(FROM_NAME\) \. ' <' \. FROM_ADDRESS \. '>'/);
  assert.match(handler, /'Reply-To: ' \. display_name\(\$name\) \. ' <' \. \$email \. '>'/);
  assert.doesNotMatch(handler, /'From: [^']*' \. \$email/);
});

test('every header-bound field is stripped of CR/LF before composing', () => {
  // Header injection is the one way a contact form becomes an open relay.
  assert.match(handler, /function header_safe\(string \$value\): string/);
  assert.match(handler, /str_replace\(\["\\r", "\\n", "\\0"\], '', \$value\)/);
  for (const field of ['name', 'email']) {
    assert.match(
      handler,
      new RegExp(`\\$${field}\\s*=\\s*header_safe\\(field\\('${field}'\\)\\)`),
      `${field} reaches a header unsanitised`,
    );
  }
});

test('the topic allowlist is identical in the page and the handler', () => {
  // The page's <option> values are visitor-controlled on submit; the handler
  // maps them through its own allowlist to build the Subject. If the two
  // drift, a legitimate submission fails closed with error=topic.
  const pageTopics = [...formHtml.matchAll(/<option value="([^"]+)"/g)].map((m) => m[1]);
  const phpBlock = handler.match(/const TOPICS = \[(.*?)\];/s);
  assert.ok(phpBlock, 'TOPICS allowlist not found in handler');
  const phpTopics = [...phpBlock[1].matchAll(/'([^']+)'\s*=>/g)].map((m) => m[1]);

  assert.deepEqual(pageTopics, phpTopics);
  assert.ok(pageTopics.length > 0);
});

test('the honeypot is hidden from people and not advertised in the markup', () => {
  assert.match(formHtml, /<div class="hp" aria-hidden="true"/);
  assert.match(formHtml, /name="website"[^>]*tabindex="-1"/);
  // The source explains the trick with a build-time comment; shipping an
  // HTML comment would tell a scraper which field to skip.
  assert.doesNotMatch(formHtml, /Honeypot/i);
});

test('a direct mail route stays visible on the form if the handler is unavailable', () => {
  // PHP could be disabled on the host, or mail() could fail. Neither should
  // leave a visitor with no way to make contact.
  assert.match(formHtml, /href="mailto:mpei@vibe-qc\.com"/);
});

test('the confirmation page does not republish the maintainer address', () => {
  // Someone who has already submitted the form does not need it, and a
  // confirmation page is a cheap, predictable URL for an address harvester
  // to fetch. The form exists so the address need not be on every page.
  assert.doesNotMatch(sentHtml, /mpei@vibe-qc\.com/);
});

test('the success page is a real pre-rendered page, so no-JS submits land somewhere', () => {
  assert.match(sentHtml, /Message sent/);
  // The handler redirects here; the path it builds must exist as built output.
  assert.match(handler, /\$sentUrl\s*=\s*\$basePath \. '\/contact\/sent\/';/);
  assert.match(handler, /\$formUrl\s*=\s*\$basePath \. '\/contact\/';/);
});

test('the handler answers with 303 so a reload does not resend the message', () => {
  assert.match(handler, /header\('Location: ' \. \$url, true, 303\);/);
});

test('redirect targets are derived from the script URL, not hardcoded to the root', () => {
  // The staging build is served under /preview/ (PREVIEW_BASE in
  // astro.config.mjs); a hardcoded '/contact/' would escape it.
  assert.match(handler, /dirname\(\(string\) \(\$_SERVER\['SCRIPT_NAME'\]/);
  assert.doesNotMatch(handler, /Location: \/contact/);
});

test('the contact page is reachable from the site chrome', () => {
  assert.match(homeHtml, new RegExp(`href="${link('contact')}"`));
});
