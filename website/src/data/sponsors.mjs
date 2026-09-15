/**
 * Canonical sponsorship plan and public recognition data.
 *
 * Monthly tiers are designed to use one GitHub Sponsors destination once its
 * matching tiers are public and verified. Ko-fi is the established one-time
 * route. Public recognition is opt-in: add only a confirmed public display
 * name, and never infer one from private payment information.
 *
 * Active monthly sponsors belong in `activeSponsors`. When a recurring
 * sponsorship ends, remove that active record but retain the contributor in
 * `supporters` when they opted into the permanent supporter list.
 */
export const sponsorshipProgram = Object.freeze({
  lastReviewed: '2026-07-26',
  monthlyCheckout: Object.freeze({
    status: 'setup-required',
    statusLabel: 'Monthly checkout setup pending',
    url: 'https://github.com/sponsors/mpeintinger',
  }),
  oneTimeContributionUrl: 'https://ko-fi.com/mpeintinger',
  recognitionContact: 'mpei@vibe-qc.com',
  tiers: Object.freeze([
    Object.freeze({
      id: 'platinum',
      name: 'Platinum',
      monthlyUsd: 250,
      billing: 'monthly',
      placement: 'Premier',
      recognition:
        'The largest logo or public name placement at the top of the active sponsor wall, with one website link.',
    }),
    Object.freeze({
      id: 'gold',
      name: 'Gold',
      monthlyUsd: 100,
      billing: 'monthly',
      placement: 'Featured',
      recognition:
        'A featured logo or public name on the active sponsor wall, with one website link.',
    }),
    Object.freeze({
      id: 'silver',
      name: 'Silver',
      monthlyUsd: 25,
      billing: 'monthly',
      placement: 'Standard',
      recognition:
        'A compact logo or public name on the active sponsor wall, with one website link.',
    }),
  ]),
  supporter: Object.freeze({
    name: 'Supporter',
    priceLabel: 'Any contribution',
    permanence: 'Permanent',
    recognition:
      'Anyone who has ever contributed can opt into the permanent supporter list, including one-time and former monthly contributors.',
  }),
  activeSponsors: Object.freeze({
    platinum: Object.freeze([]),
    gold: Object.freeze([]),
    silver: Object.freeze([]),
  }),
  supporters: Object.freeze([]),
});

const HTTPS_URL = /^https:\/\/[^\s]+$/;
const LOCAL_LOGO = /^sponsors\/[A-Za-z0-9][A-Za-z0-9._-]*\.(?:png|svg|webp)$/;

export function publicMonthlyCheckoutUrl(program) {
  if (
    program.monthlyCheckout.status === 'available' &&
    typeof program.monthlyCheckout.url === 'string' &&
    HTTPS_URL.test(program.monthlyCheckout.url)
  ) {
    return program.monthlyCheckout.url;
  }
  return null;
}

export function publicSponsorWebsite(sponsor) {
  if (
    typeof sponsor.websiteUrl === 'string' &&
    HTTPS_URL.test(sponsor.websiteUrl)
  ) {
    return sponsor.websiteUrl;
  }
  return null;
}

export function publicSponsorLogo(sponsor) {
  if (typeof sponsor.logoSrc === 'string' && LOCAL_LOGO.test(sponsor.logoSrc)) {
    return sponsor.logoSrc;
  }
  return null;
}
