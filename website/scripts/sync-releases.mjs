import {writeFile} from 'node:fs/promises';
import {validateDesktopRelease} from '../src/data/desktop-releases.mjs';
const order = ['vibe-sketch','vibe-queue','vibe-qc','vibe-view'];
const rows = await Promise.all(order.map(async product => {
  const response = await fetch(`https://vibe-qc.com/updates/${product}/current/release.json`, {signal:AbortSignal.timeout(30000)});
  if (!response.ok) throw new Error(`Cannot verify ${product}: ${response.status}`);
  const data = validateDesktopRelease(await response.json(), product);
  return [product,{version:data.version,channel:'Standalone release',url:`/download/#${product}`,manifest:data}];
}));
await writeFile(new URL('../src/data/published-releases.json',import.meta.url),JSON.stringify(Object.fromEntries(rows),null,2)+'\n');
console.log('Verified release versions embedded for all four programs.');

const maceResponse = await fetch('https://vibe-qc.com/updates/vibe-qc/variants/mace/current/release.json', {signal:AbortSignal.timeout(30000)});
if (!maceResponse.ok) throw new Error(`Cannot verify optional MACE profile: ${maceResponse.status}`);
const maceRelease = validateDesktopRelease(await maceResponse.json(), 'vibe-qc', 'mace');
await writeFile(new URL('../src/data/published-mace-release.json',import.meta.url),JSON.stringify(maceRelease,null,2)+'\n');
console.log('Verified optional MACE release embedded.');
