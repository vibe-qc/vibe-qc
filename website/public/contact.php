<?php
/**
 * vibe-qc.com — contact-form mail handler.
 * =====================================================================
 * WHY THIS FILE EXISTS
 *
 * The marketing site (../src/) is a static Astro build with no runtime
 * (see ../README.md). A contact form still needs *something* server-side
 * to accept a POST and send mail, so this one small handler lives in
 * `public/` — Astro copies `public/` verbatim into `dist/`, and the
 * existing `website-deploy` rsync ships `dist/` to the web root. No new
 * build step, no new dependency, no third-party form service: a
 * visitor's message never leaves the vibe-qc host.
 *
 * CONTRACT WITH THE STATIC PAGES
 *
 *   /contact/       — the form (src/pages/contact.astro), posts here
 *   /contact/sent/  — the success page (src/pages/contact/sent.astro)
 *
 * Both are real pre-rendered pages, so the happy path works with
 * JavaScript disabled: this handler answers every POST with a 303
 * redirect to one of them. Failures come back as
 * `/contact/?error=<code>`; the form page renders the matching banner
 * and always shows the plain mailto: address as a fallback, so a visitor
 * is never left without a way to reach the maintainer.
 *
 * DEPLOYMENT REQUIREMENTS  (verify once, per ../README.md § Contact form)
 *
 *   - PHP must be enabled for the document root. If it is not, Apache
 *     serves this file as text and the form silently breaks — check
 *     before linking the page anywhere.
 *   - mail() must be able to hand off to the host MTA.
 *   - FROM_ADDRESS below must be a real mailbox on the sending domain so
 *     SPF/DKIM align. The visitor's address goes in Reply-To, never in
 *     From — putting it in From is what gets mail SPF-rejected.
 *
 * Deliberately dependency-free plain PHP: no composer, no framework.
 * =====================================================================
 */

declare(strict_types=1);

// --------------------------------------------------------------------
// Configuration
// --------------------------------------------------------------------

/** Where contact-form mail is forwarded. */
const TO_ADDRESS = 'mpei@vibe-qc.com';

/**
 * Envelope sender. Must be a mailbox on the sending domain — SPF/DKIM
 * are published for vibe-qc.com, so mail claiming to be *from* a
 * visitor's own domain would fail alignment and land in spam.
 */
const FROM_ADDRESS = 'website@vibe-qc.com';
const FROM_NAME    = 'vibe-qc website';

/** Subject prefix, so inbox filters can key off it. */
const SUBJECT_PREFIX = '[vibe-qc contact]';

/**
 * Topics offered by the form. Server-side allowlist: the submitted value
 * must be one of these keys, so nothing visitor-controlled ever reaches
 * the Subject header from this field.
 */
const TOPICS = [
    'access'      => 'Repository access request',
    'bug'         => 'Bug report',
    'question'    => 'Question about using vibe-qc',
    'sponsorship' => 'Sponsorship',
    'other'       => 'Something else',
];

/** Field length caps, enforced before anything is composed. */
const MAX_NAME    = 120;
const MAX_EMAIL   = 254;   // RFC 5321 maximum path length
const MAX_MESSAGE = 8000;

/** Per-IP submission budget. */
const RATE_LIMIT_MAX    = 5;
const RATE_LIMIT_WINDOW = 3600;  // seconds

/**
 * Minimum seconds between the form rendering and its submission. The
 * form page stamps a hidden field with Date.now() on load, so this only
 * applies when JavaScript ran — a no-JS submission sends no stamp and
 * skips the check rather than being rejected. A heuristic against
 * instant-submit bots, not a security control.
 */
const MIN_FILL_SECONDS = 3;

// --------------------------------------------------------------------
// Redirect targets
//
// Derived from this script's own URL rather than hardcoded, so the
// staging build served under /preview/ (PREVIEW_BASE, see
// ../astro.config.mjs) redirects within /preview/ too.
// --------------------------------------------------------------------

$scriptDir = str_replace('\\', '/', dirname((string) ($_SERVER['SCRIPT_NAME'] ?? '/contact.php')));
$basePath  = rtrim($scriptDir, '/');           // '' at the root, '/preview' on staging
$formUrl   = $basePath . '/contact/';
$sentUrl   = $basePath . '/contact/sent/';

/**
 * Answer with a 303 redirect and stop. 303 (not 302) so the browser
 * re-issues the follow-up as GET — that is what stops a reload from
 * re-submitting the message.
 *
 * No `never` return type: this file targets PHP 7.4+ so it keeps working
 * if the host's PHP version is older than the current default.
 */
function redirect_to(string $url)
{
    header('Location: ' . $url, true, 303);
    header('Cache-Control: no-store');
    exit;
}

/** Bounce back to the form with an error code the page knows how to render. */
function fail(string $formUrl, string $code)
{
    redirect_to($formUrl . '?error=' . rawurlencode($code) . '#contact-form');
}

// --------------------------------------------------------------------
// Only POST is meaningful here
// --------------------------------------------------------------------

if (($_SERVER['REQUEST_METHOD'] ?? 'GET') !== 'POST') {
    redirect_to($formUrl);
}

// --------------------------------------------------------------------
// Header-injection defence
//
// Any value interpolated into a mail header must not carry CR, LF or a
// NUL byte, or a crafted field could append headers of its own and turn
// this into an open relay. Applied to every header-bound field below;
// the message body is a body, so it keeps its newlines.
// --------------------------------------------------------------------

function header_safe(string $value): string
{
    return trim(str_replace(["\r", "\n", "\0"], '', $value));
}

/** Read a POST field as a trimmed UTF-8 string. */
function field(string $key): string
{
    $raw = $_POST[$key] ?? '';
    if (!is_string($raw)) {
        return '';
    }
    // Drop invalid UTF-8 rather than letting it corrupt the encoded headers.
    // iconv is not guaranteed on a shared host, so fall back to the raw
    // value when the extension is absent.
    if (function_exists('iconv')) {
        $clean = @iconv('UTF-8', 'UTF-8//IGNORE', $raw);
        if ($clean !== false) {
            $raw = $clean;
        }
    }
    return trim($raw);
}

/** mb_strlen when mbstring is available, byte length otherwise. */
function text_length(string $value): int
{
    return function_exists('mb_strlen') ? mb_strlen($value, 'UTF-8') : strlen($value);
}

/**
 * RFC 2047 encoded-word, for non-ASCII display names and subjects.
 * Plain ASCII is passed through so ordinary mail stays readable in raw
 * form.
 */
function encode_header(string $value): string
{
    if (preg_match('/^[\x20-\x7E]*$/', $value) === 1) {
        return $value;
    }
    return '=?UTF-8?B?' . base64_encode($value) . '?=';
}

/**
 * A display name for the `Name <addr>` form of an address header.
 *
 * RFC 5322 "specials" have to be quoted or the header re-parses into
 * something else: an ordinary name like `Smith, John` would otherwise be
 * read as *two* addresses. An encoded-word is returned as-is — those must
 * not be wrapped in quotes.
 */
function display_name(string $name): string
{
    $encoded = encode_header($name);
    if ($encoded !== $name) {
        return $encoded;
    }

    if (preg_match('/[()<>@,;:\\\\".\[\]]/', $name) === 1) {
        return '"' . str_replace(['\\', '"'], ['\\\\', '\\"'], $name) . '"';
    }

    return $name;
}

// --------------------------------------------------------------------
// Spam gates
// --------------------------------------------------------------------

// 1. Honeypot. A field hidden from humans and left empty by them; bots
//    that fill every input trip it. Answer with the ordinary success
//    page so an automated submitter learns nothing, and send no mail.
if (field('website') !== '') {
    redirect_to($sentUrl);
}

// 2. Fill time (only when the browser ran the stamping script).
$stamp = field('ts');
if ($stamp !== '' && ctype_digit($stamp)) {
    $elapsed = time() - (int) ((int) $stamp / 1000);
    if ($elapsed >= 0 && $elapsed < MIN_FILL_SECONDS) {
        redirect_to($sentUrl);
    }
}

// 3. Per-IP rate limit. A tiny counter file per client under the system
//    temp dir — no database, and it self-heals if the file is lost.
//
//    Checking and recording are deliberately separate calls: only a
//    *sent* message is counted, at the very end. Counting attempts
//    instead would lock a visitor out for an hour because they mistyped
//    their address a few times, and what actually needs limiting is
//    outbound mail volume — a rejected submission sends nothing.

/** Path of this client's counter file, or null when tracking is unavailable. */
function rate_limit_path(string $ip): ?string
{
    if ($ip === '') {
        return null;
    }

    $dir = sys_get_temp_dir() . '/vibeqc-contact-rl';
    if (!is_dir($dir) && !@mkdir($dir, 0700, true) && !is_dir($dir)) {
        return null;  // Cannot track — fail open rather than block real mail.
    }

    return $dir . '/' . hash('sha256', $ip);
}

/** Timestamps of this client's sends that are still inside the window. */
function rate_limit_stamps(?string $path): array
{
    if ($path === null || !is_readable($path)) {
        return [];
    }

    $decoded = json_decode((string) @file_get_contents($path), true);
    if (!is_array($decoded)) {
        return [];
    }

    $now = time();

    return array_values(array_filter(
        $decoded,
        static function ($t) use ($now) {
            return is_int($t) && ($now - $t) < RATE_LIMIT_WINDOW;
        }
    ));
}

/** True when this client has already sent its allowance for the window. */
function rate_limit_exceeded(?string $path): bool
{
    return $path !== null && count(rate_limit_stamps($path)) >= RATE_LIMIT_MAX;
}

/** Record one successfully sent message against this client. */
function rate_limit_record(?string $path)
{
    if ($path === null) {
        return;
    }

    $stamps   = rate_limit_stamps($path);
    $stamps[] = time();
    @file_put_contents($path, json_encode($stamps), LOCK_EX);
}

$clientIp  = header_safe((string) ($_SERVER['REMOTE_ADDR'] ?? ''));
$rateFile  = rate_limit_path($clientIp);
if (rate_limit_exceeded($rateFile)) {
    fail($formUrl, 'rate');
}

// --------------------------------------------------------------------
// Validation
// --------------------------------------------------------------------

$name    = header_safe(field('name'));
$email   = header_safe(field('email'));
$topic   = field('topic');
$message = field('message');

if ($name === '' || $message === '') {
    fail($formUrl, 'missing');
}

if (text_length($name) > MAX_NAME
    || text_length($email) > MAX_EMAIL
    || text_length($message) > MAX_MESSAGE
) {
    fail($formUrl, 'toolong');
}

if (filter_var($email, FILTER_VALIDATE_EMAIL) === false) {
    fail($formUrl, 'email');
}

// Unknown topic keys are rejected outright rather than coerced, so the
// Subject header only ever carries a string from the allowlist above.
if (!array_key_exists($topic, TOPICS)) {
    fail($formUrl, 'topic');
}
$topicLabel = TOPICS[$topic];

// --------------------------------------------------------------------
// Compose
// --------------------------------------------------------------------

$subject = encode_header(SUBJECT_PREFIX . ' ' . $topicLabel);

$body = implode("\n", [
    'A message was submitted through the vibe-qc.com contact form.',
    '',
    'From:    ' . $name . ' <' . $email . '>',
    'Topic:   ' . $topicLabel,
    'Sent:    ' . gmdate('Y-m-d H:i:s') . ' UTC',
    'Client:  ' . ($clientIp === '' ? 'unknown' : $clientIp),
    'Agent:   ' . header_safe(substr((string) ($_SERVER['HTTP_USER_AGENT'] ?? ''), 0, 200)),
    '',
    str_repeat('-', 68),
    '',
    $message,
    '',
]);

$headers = [
    'From: ' . display_name(FROM_NAME) . ' <' . FROM_ADDRESS . '>',
    // Reply-To carries the visitor, so hitting reply in a mail client
    // reaches them — while From stays SPF/DKIM-aligned to this domain.
    'Reply-To: ' . display_name($name) . ' <' . $email . '>',
    'MIME-Version: 1.0',
    'Content-Type: text/plain; charset=UTF-8',
    'Content-Transfer-Encoding: 8bit',
    'X-Mailer: vibe-qc-website',
    'Auto-Submitted: auto-generated',
];

// -f sets the envelope sender so bounces return to a real mailbox.
$sent = @mail(
    TO_ADDRESS,
    $subject,
    $body,
    implode("\r\n", $headers),
    '-f' . FROM_ADDRESS
);

if ($sent === false) {
    error_log('vibe-qc contact form: mail() handoff failed');
    fail($formUrl, 'send');
}

// Counted only now that a message actually went out — see the note above
// rate_limit_path().
rate_limit_record($rateFile);

redirect_to($sentUrl);
