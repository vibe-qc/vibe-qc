import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { link } from './base.mjs';

const sponsorHtml = await readFile(
  new URL('../dist/sponsor/index.html', import.meta.url),
  'utf8',
);
const homeHtml = await readFile(new URL('../dist/index.html', import.meta.url), 'utf8');
const downloadHtml = await readFile(
  new URL('../dist/download/index.html', import.meta.url),
  'utf8',
);

test('Sponsor navigation uses a real destination from every site surface', () => {
  assert.match(homeHtml, new RegExp(`href="${link('sponsor')}"`));
  assert.match(downloadHtml, new RegExp(`href="${link('sponsor')}"`));
  assert.doesNotMatch(homeHtml, new RegExp(`href="${link('#sponsor')}"`));
  assert.doesNotMatch(downloadHtml, new RegExp(`href="${link('#sponsor')}"`));
});

test('the Sponsor page contains substantive funding information', () => {
  assert.match(sponsorHtml, /What support funds/i);
  assert.match(sponsorHtml, /Compute and verification/);
  assert.match(sponsorHtml, /Public infrastructure/);
  assert.match(sponsorHtml, /Reference access/);
  assert.match(sponsorHtml, /NIST Crystal Data SRD 3/);
  assert.match(sponsorHtml, /\$200/);
});

test('the established one-time route is present and unready checkout is gated', () => {
  assert.match(sponsorHtml, /href="https:\/\/ko-fi\.com\/mpeintinger"/);
  assert.doesNotMatch(sponsorHtml, /href="https:\/\/github\.com\/sponsors\/mpeintinger"/);
  assert.doesNotMatch(homeHtml, /href="https:\/\/github\.com\/sponsors\/mpeintinger"/);
  assert.match(sponsorHtml, /Monthly checkout setup pending/);
  assert.match(homeHtml, new RegExp(`href="${link('sponsor')}"`));
});

test('monthly sponsor tiers and their prices are rendered', () => {
  assert.match(sponsorHtml, />Platinum</);
  assert.match(sponsorHtml, />Gold</);
  assert.match(sponsorHtml, />Silver</);
  assert.match(sponsorHtml, />\$250</);
  assert.match(sponsorHtml, />\$100</);
  assert.match(sponsorHtml, />\$25</);
  assert.match(sponsorHtml, /USD \/ month/);
});

test('logo recognition is optional and supporter recognition is permanent', () => {
  assert.match(sponsorHtml, /Your logo, if desired/i);
  assert.match(sponsorHtml, /Recognition is always optional/i);
  assert.match(sponsorHtml, /Permanent, opt-in roll/i);
  assert.match(sponsorHtml, /choosing a public display name/i);
});

test('empty recognition data renders honest open states', () => {
  assert.match(sponsorHtml, /Open for the first Platinum sponsor/);
  assert.match(sponsorHtml, /Open for the first Gold sponsor/);
  assert.match(sponsorHtml, /Open for the first Silver sponsor/);
  assert.doesNotMatch(sponsorHtml, /tracking-logo/);
});

test('sponsorship does not imply scientific or roadmap control', () => {
  assert.match(sponsorHtml, /does not buy the result/i);
  assert.match(sponsorHtml, /do not control the roadmap/i);
  assert.match(sponsorHtml, /no sponsor-only features/i);
});
