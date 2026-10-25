import assert from 'node:assert/strict';
import test from 'node:test';
import { checkDownloads } from '../scripts/check-downloads.mjs';
const sha = 'a'.repeat(64);
const artifact = {id:'test',status:'available',sha256:sha,downloadUrl:'https://example.com/app.AppImage',sourceUrl:'https://example.com/source.tar.gz',checksumsUrl:'https://example.com/SHA256SUMS'};
const ok = async url => new Response(url.endsWith('SHA256SUMS') ? `${sha}  app.AppImage\n${sha}  source.tar.gz\n` : 'package', {headers:{'content-length':'7','content-type':'application/octet-stream'}});
test('accepts reachable packages with matching source and checksums', async () => {
  await checkDownloads([{github:'https://github.com/example/app'}], [artifact], ok);
});
test('refuses missing releases instead of accepting tags as releases', async () => {
  await assert.rejects(checkDownloads([{github:'https://github.com/example/app'}], [], async () => new Response('',{status:404})), /unavailable/);
});
test('refuses missing package and mismatched checksums', async () => {
  await assert.rejects(checkDownloads([], [artifact], async url => url.endsWith('AppImage') ? new Response('',{status:404}) : ok(url)), /unavailable/);
  await assert.rejects(checkDownloads([], [{...artifact,sha256:'b'.repeat(64)}], ok), /does not match/);
});
test('refuses a successful HTML error page and ignores unavailable entries', async () => {
  await assert.rejects(checkDownloads([], [artifact], async url => url.endsWith('AppImage') ? new Response('error',{headers:{'content-type':'text/html','content-length':'5'}}) : ok(url)), /not a nonempty package/);
  await checkDownloads([], [{...artifact,status:'unavailable'}], () => {throw new Error('Inactive link requested');});
});
