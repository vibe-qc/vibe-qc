import test from 'node:test';
import assert from 'node:assert/strict';
import { currentRelease, refreshReleaseBadges } from '../src/data/current-releases.mjs';
import releases from '../src/data/published-releases.json' with {type:'json'};

test('all repeated badges advance to the published version and retain snapshots on failure', async () => {
  const make = product => ({dataset:{releaseProduct:product}, fields:{
    '[data-release-version]':{textContent:'v0.0.1'},
    '[data-release-channel]':{textContent:'Standalone release'},
  },querySelector(key){return this.fields[key];}});
  const badges = [make('vibe-qc'),make('vibe-qc'),make('vibe-view')];
  await refreshReleaseBadges({querySelectorAll:()=>badges}, async product => {
    if(product==='vibe-view') throw new Error('offline');
    return {version:'0.18.0'};
  });
  assert.deepEqual(badges.map(b=>b.fields['[data-release-version]'].textContent),['v0.18.0','v0.18.0','v0.0.1']);
  assert.equal(badges[2].fields['[data-release-channel]'].textContent,'Release snapshot');
});

test('badges and cards share one uncached, validated request per product',async()=>{
  let calls=0;
  const request=async(url,options)=>{
    calls++;
    assert.equal(url,'/updates/vibe-qc/current/release.json');
    assert.equal(options.cache,'no-store');
    return {ok:true,json:async()=>releases['vibe-qc'].manifest};
  };
  const [a,b]=await Promise.all([currentRelease('vibe-qc',request),currentRelease('vibe-qc',request)]);
  assert.equal(a,b);assert.equal(calls,1);
  await assert.rejects(currentRelease('vibe-view',async()=>({ok:true,json:async()=>({version:'99.0.0'})})),/Invalid desktop release/);
});
