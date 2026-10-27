import assert from 'node:assert/strict';
import test from 'node:test';
import { checkDownloads, checkMaceDownloads } from '../scripts/check-downloads.mjs';
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

test('optional MACE downloads require their own complete published pair', async () => {
 const prefix='/updates/vibe-qc/variants/mace/releases/1.0.0/';
 const manifest={product:'vibe-qc',profile:'mace',version:'1.0.0',sourceCommit:'a'.repeat(40),artifacts:Object.fromEntries(['macos','linux','source'].map(kind=>[kind,{filename:kind,url:prefix+kind,sha256:sha,size:7}]))};
 const request=async url=>url.endsWith('release.json') ? new Response(JSON.stringify(manifest)) : url.endsWith('SHA256SUMS') ? new Response(['macos','linux','source'].map(kind=>`${sha}  ${kind}\n`).join('')) : new Response('package',{headers:{'content-length':'7','content-type':'application/octet-stream'}});
 await checkMaceDownloads(request);
 await assert.rejects(checkMaceDownloads(async url=>url.endsWith('/linux') ? new Response('',{status:404}) : request(url)),/unavailable/);
 manifest.profile=undefined;
 await assert.rejects(checkMaceDownloads(request),/Invalid desktop profile/);
});
