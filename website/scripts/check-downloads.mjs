/** Refuse a website build that advertises an unavailable release or package. */
import { pathToFileURL } from 'node:url';
import { repositories } from '../src/data/repositories.mjs';
import { desktopProducts, validateDesktopRelease } from '../src/data/desktop-releases.mjs';
import { distributionMatrix, publicDownloadUrl } from '../src/data/distributions.mjs';

export async function checkDownloads(repos, artifacts, request = fetch) {
  async function get(url, method = 'HEAD') {
    const response = await request(url, { method, redirect: 'follow', signal: AbortSignal.timeout(30000) });
    if (!response.ok) throw new Error(`Download unavailable (${response.status}): ${url}`);
    return response;
  }
  for (const repo of repos.filter(repo => repo.github)) {
    // /latest returns 404 when a repository has tags but no published releases.
    await get(`${repo.github}/releases/latest`);
  }
  for (const artifact of artifacts.filter(artifact => publicDownloadUrl(artifact))) {
    if (!/^[a-f0-9]{64}$/.test(artifact.sha256) || !artifact.sourceUrl || !artifact.checksumsUrl) {
      throw new Error(`Missing release source or checksum metadata: ${artifact.id}`);
    }
    const sums = await (await get(artifact.checksumsUrl, 'GET')).text();
    const entries = new Map(sums.trim().split(/\r?\n/).map(line => {
      const match = /^([a-f0-9]{64})\s+\*?(.+)$/.exec(line);
      if (!match) throw new Error(`Invalid checksum manifest: ${artifact.id}`);
      return [match[2], match[1]];
    }));
    const packageName = decodeURIComponent(new URL(artifact.downloadUrl).pathname.split('/').pop());
    const sourceName = decodeURIComponent(new URL(artifact.sourceUrl).pathname.split('/').pop());
    if (entries.get(packageName) !== artifact.sha256 || !entries.has(sourceName)) {
      throw new Error(`Published checksum manifest does not match release: ${artifact.id}`);
    }
    for (const url of [artifact.downloadUrl, artifact.sourceUrl]) {
      const response = await get(url);
      if (/text\/html/i.test(response.headers.get('content-type') || '') || Number(response.headers.get('content-length')) <= 0) {
        throw new Error(`Download is not a nonempty package: ${url}`);
      }
    }
  }
}

export async function checkCurrentDesktops(request = fetch) {
  for (const product of desktopProducts) {
    const base = `https://vibe-qc.com/updates/${product}/current/`;
    const response = await request(base + 'release.json', {cache:'no-store', signal:AbortSignal.timeout(30000)});
    if (!response.ok) throw new Error(`Desktop release unavailable: ${product}`);
    const data = validateDesktopRelease(await response.json(), product);
    const sourceUrl = 'https://vibe-qc.com' + data.artifacts.source.url;
    await checkDownloads([], ['macos','linux'].map(kind => ({
      id:product + '-' + kind, status:'available', sha256:data.artifacts[kind].sha256,
      downloadUrl:'https://vibe-qc.com' + data.artifacts[kind].url,
      sourceUrl, checksumsUrl:`https://vibe-qc.com/updates/${product}/releases/${data.version}/SHA256SUMS`,
    })), request);
  }
}

export async function checkMaceDownloads(request = fetch) {
  const base = 'https://vibe-qc.com/updates/vibe-qc/variants/mace/';
  const response = await request(base + 'current/release.json', {cache:'no-store', signal:AbortSignal.timeout(30000)});
  if (!response.ok) throw new Error('Optional MACE release unavailable');
  const data = validateDesktopRelease(await response.json(), 'vibe-qc', 'mace');
  await checkDownloads([], ['macos','linux'].map(kind => ({
    id:'vibe-qc-mace-' + kind, status:'available', sha256:data.artifacts[kind].sha256,
    downloadUrl:'https://vibe-qc.com' + data.artifacts[kind].url,
    sourceUrl:'https://vibe-qc.com' + data.artifacts.source.url,
    checksumsUrl:base + `releases/${data.version}/SHA256SUMS`,
  })), request);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await checkDownloads(repositories, distributionMatrix.desktop);
  await checkCurrentDesktops();
  await checkMaceDownloads();
  console.log('Public release and download availability verified.');
}
