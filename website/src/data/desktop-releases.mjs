export const desktopProducts = ['vibe-sketch', 'vibe-queue', 'vibe-qc', 'vibe-view'];

export function validateDesktopRelease(data, product, profile = null) {
  if ((profile !== null && (profile !== 'mace' || product !== 'vibe-qc')) ||
      (data?.profile ?? null) !== profile) throw new Error('Invalid desktop profile');
  const directory = `/updates/${product}${profile ? `/variants/${profile}` : ''}`;
  if (data?.product !== product || !/^\d+\.\d+\.\d+$/.test(data.version) ||
      !/^[a-f0-9]{40}$/.test(data.sourceCommit)) throw new Error('Invalid desktop release');
  for (const kind of ['macos', 'linux', 'source']) {
    const item = data.artifacts?.[kind];
    if (!item || !/^[a-f0-9]{64}$/.test(item.sha256) || !Number.isSafeInteger(item.size) || item.size <= 0 ||
        !/^[a-zA-Z0-9._-]+$/.test(item.filename) ||
        item.url !== `${directory}/releases/${data.version}/${item.filename}`)
      throw new Error('Invalid desktop artifact');
  }
  return data;
}
