import assert from 'node:assert/strict';
import test from 'node:test';

import {
  publicMonthlyCheckoutUrl,
  publicSponsorLogo,
  publicSponsorWebsite,
  sponsorshipProgram,
} from '../src/data/sponsors.mjs';

test('recurring sponsorship tiers use the published monthly plan', () => {
  assert.deepEqual(
    sponsorshipProgram.tiers.map(({ name, monthlyUsd, billing }) => ({
      name,
      monthlyUsd,
      billing,
    })),
    [
      { name: 'Platinum', monthlyUsd: 250, billing: 'monthly' },
      { name: 'Gold', monthlyUsd: 100, billing: 'monthly' },
      { name: 'Silver', monthlyUsd: 25, billing: 'monthly' },
    ],
  );
  assert.equal(sponsorshipProgram.monthlyCheckout.status, 'setup-required');
  assert.equal(publicMonthlyCheckoutUrl(sponsorshipProgram), null);
});

test('supporter recognition is permanent, contribution-wide, and opt-in', () => {
  assert.equal(sponsorshipProgram.supporter.permanence, 'Permanent');
  assert.match(sponsorshipProgram.supporter.recognition, /ever contributed/i);
  assert.match(sponsorshipProgram.supporter.recognition, /opt into/i);
  assert.equal(
    sponsorshipProgram.oneTimeContributionUrl,
    'https://ko-fi.com/mpeintinger',
  );
});

test('the public wall begins empty rather than inventing sponsors', () => {
  for (const tier of sponsorshipProgram.tiers) {
    assert.deepEqual(sponsorshipProgram.activeSponsors[tier.id], []);
  }
  assert.deepEqual(sponsorshipProgram.supporters, []);
});

test('monthly checkout requires an explicit available state and safe URL', () => {
  assert.equal(
    publicMonthlyCheckoutUrl({
      ...sponsorshipProgram,
      monthlyCheckout: {
        status: 'available',
        url: 'http://github.com/sponsors/example',
      },
    }),
    null,
  );
  assert.equal(
    publicMonthlyCheckoutUrl({
      ...sponsorshipProgram,
      monthlyCheckout: {
        status: 'available',
        url: 'https://github.com/sponsors/example',
      },
    }),
    'https://github.com/sponsors/example',
  );
});

test('public recognition accepts only safe links and local logo assets', () => {
  assert.equal(
    publicSponsorWebsite({ websiteUrl: 'https://example.org/research' }),
    'https://example.org/research',
  );
  assert.equal(
    publicSponsorWebsite({ websiteUrl: 'http://example.org/research' }),
    null,
  );
  assert.equal(
    publicSponsorLogo({ logoSrc: 'sponsors/example.svg' }),
    'sponsors/example.svg',
  );
  assert.equal(
    publicSponsorLogo({ logoSrc: 'https://example.org/tracking-logo.svg' }),
    null,
  );
});
