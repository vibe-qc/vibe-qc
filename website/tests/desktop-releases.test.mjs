import test from 'node:test';
import assert from 'node:assert/strict';
import {validateDesktopRelease} from '../src/data/desktop-releases.mjs';
const fixture = () => ({product:'vibe-view',version:'2.20.20',sourceCommit:'a'.repeat(40),artifacts:Object.fromEntries(['macos','linux','source'].map(kind=>[kind,{filename:kind,url:`/updates/vibe-view/releases/2.20.20/${kind}`,sha256:'b'.repeat(64),size:100}]))});
test('accepts complete matching releases and rejects incomplete or foreign links',()=>{
 assert.equal(validateDesktopRelease(fixture(),'vibe-view').version,'2.20.20');
 const missing=fixture();delete missing.artifacts.linux;assert.throws(()=>validateDesktopRelease(missing,'vibe-view'));
 const foreign=fixture();foreign.artifacts.macos.url='https://example.org/app.dmg';assert.throws(()=>validateDesktopRelease(foreign,'vibe-view'));
 assert.throws(()=>validateDesktopRelease(fixture(),'vibe-sketch'));
});
