#!/usr/bin/env python3
"""Publish an already qualified macOS/Linux pair with one atomic current pointer.

Run against a staged or mounted updates root; deployment credentials stay external.
Build and test both installers from the named immutable release before calling this.
"""
import argparse
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _publish(root, product, version, commit, macos, linux, source):
    if product not in ('vibe-view', 'vibe-sketch', 'vibe-qc', 'vibe-queue') or not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('Expected a supported product and stable release version')
    if not re.fullmatch(r'[a-f0-9]{40}', commit):
        raise ValueError('Expected the immutable release source commit')
    linux_suffix = 'tar.gz' if product in ('vibe-qc', 'vibe-queue') else 'AppImage'
    inputs = [('macos', macos, f'{product}-{version}-arm64.dmg'),
              ('linux', linux, f'{product}-{version}-x86_64.{linux_suffix}'),
              ('source', source, f'{product}-{version}-source.tar.gz')]
    for _, path, _ in inputs:
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError('Every release requires regular, nonempty macOS, Linux and source files')
    directory = root / product
    releases = directory / 'releases'
    releases.mkdir(parents=True, exist_ok=True)
    destination = releases / version
    manifest = {'product': product, 'version': version, 'sourceCommit': commit, 'artifacts': {}}
    for kind, path, name in inputs:
        manifest['artifacts'][kind] = {'filename': name, 'sha256': digest(path), 'size': path.stat().st_size,
            'url': f'/updates/{product}/releases/{version}/{name}'}
    if destination.exists():
        if json.loads((destination / 'release.json').read_text()) != manifest:
            raise ValueError('Refusing to replace an immutable release with different bytes')
        for row in manifest['artifacts'].values():
            if digest(destination / row['filename']) != row['sha256']:
                raise ValueError('Previously published release bytes have changed')
    else:
        with tempfile.TemporaryDirectory(prefix='.publish-', dir=releases) as temp:
            stage = Path(temp) / version
            stage.mkdir()
            for kind, path, name in inputs:
                shutil.copyfile(path, stage / name)
                (stage / {'macos':'download-macos.dmg', 'linux':f'download-linux.{linux_suffix}', 'source':'source.tar.gz'}[kind]).symlink_to(name)
            (stage / 'SHA256SUMS').write_text(''.join(f"{row['sha256']}  {row['filename']}\n" for row in manifest['artifacts'].values()))
            (stage / 'release.json').write_text(json.dumps(manifest, indent=2) + '\n')
            stage.rename(destination)
    current = directory / 'current'
    if current.exists():
        previous = json.loads((current / 'release.json').read_text())
        if tuple(map(int, previous['version'].split('.'))) > tuple(map(int, version.split('.'))):
            raise ValueError('Refusing to roll the current release backwards')
    pending = directory / '.current-next'
    try:
        pending.symlink_to(f'releases/{version}')
        os.replace(pending, current)
    finally:
        pending.unlink(missing_ok=True)
    return manifest


def publish(root, product, version, commit, macos, linux, source):
    if product not in ('vibe-view', 'vibe-sketch', 'vibe-qc', 'vibe-queue'):
        raise ValueError('Unsupported product')
    directory = root / product
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _publish(root, product, version, commit, macos, linux, source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates-root', type=Path, required=True)
    parser.add_argument('--product', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--source-commit', required=True)
    for field in ('macos', 'linux', 'source'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.updates_root, args.product, args.version, args.source_commit,
                             args.macos, args.linux, args.source), indent=2))
