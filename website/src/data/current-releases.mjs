import { validateDesktopRelease } from './desktop-releases.mjs';

// One request per product per page, shared by badges and download cards.
const pending = new Map();
export function currentRelease(product, request = fetch) {
  if (!pending.has(product)) {
    pending.set(product, request(`/updates/${product}/current/release.json`, {
      cache: 'no-store', signal: AbortSignal.timeout(15000),
    }).then(response => {
      if (!response.ok) throw new Error('Unavailable release');
      return response.json();
    }).then(data => validateDesktopRelease(data, product)));
  }
  return pending.get(product);
}

export async function refreshReleaseBadges(root = document, load = currentRelease) {
  await Promise.all([...root.querySelectorAll('[data-release-product]')].map(async badge => {
    try {
      const release = await load(badge.dataset.releaseProduct);
      badge.querySelector('[data-release-version]').textContent = `v${release.version}`;
    } catch {
      // Keep the verified, server-rendered version useful when offline.
      badge.querySelector('[data-release-channel]').textContent = 'Release snapshot';
    }
  }));
}
