#!/usr/bin/env python3
"""Observe an opt-in fresh build; never qualify an existing install retrospectively."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tarfile
import time

SCHEMA = 'vibe.native-build-receipt/1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def regular(path):
    path = Path(path)
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_nlink != 1:
        raise ValueError('regular unaliased file required: ' + str(path))
    return {'sha256': digest(path), 'mode': stat.S_IMODE(path.stat().st_mode)}


def inventory(root):
    root = Path(root).resolve(strict=True)
    result = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        if any((Path(directory) / n).is_symlink() for n in dirs):
            raise ValueError('directory aliases unsupported')
        for name in sorted(names):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                target = os.readlink(path)
                if '/' in target or target in ('', '.', '..') or not path.resolve().is_file():
                    raise ValueError('only sibling regular-file aliases supported')
                if not path.resolve().is_relative_to(root):
                    raise ValueError('alias escapes install')
                result[relative] = {'symlink': target}
            else:
                result[relative] = regular(path)
    return result


def archive_inventory(path, prefix, expected):
    if regular(path)['sha256'] != expected or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('source archive digest mismatch')
    result, seen = {}, set()
    with tarfile.open(path, 'r:*') as stream:
        for member in stream:
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] != prefix:
                raise ValueError('archive member outside source prefix')
            relative = Path(*name.parts[1:]).as_posix()
            if relative in seen or not (member.isfile() or member.isdir()):
                raise ValueError('duplicate or non-regular archive member')
            seen.add(relative)
            if member.isdir():
                continue
            if relative == '.':
                raise ValueError('source prefix is not a directory')
            h = hashlib.sha256()
            with stream.extractfile(member) as source:
                for block in iter(lambda: source.read(65536), b''):
                    h.update(block)
            result[relative] = {'sha256': h.hexdigest(), 'mode': member.mode & 0o7777}
    if not result or digest(path) != expected:
        raise ValueError('empty or changed source archive')
    return result


def cache(path):
    values = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r'([^/#][^:]*):([^=]+)=(.*)', line)
        if match:
            values[match[1]] = {'type': match[2], 'value': match[3]}
    if not values:
        raise ValueError('configured CMake cache required')
    return values


def snapshot(spec):
    sources, authentication = {}, {}
    if not spec['sources'] or set(spec['sources']) != set(spec['authenticated_sources']):
        raise ValueError('every source needs an authenticated archive')
    for name, directory in spec['sources'].items():
        auth = spec['authenticated_sources'][name]
        observed = inventory(directory)
        if observed != archive_inventory(auth['archive'], auth['prefix'], auth['sha256']):
            raise ValueError('source differs from unpatched authenticated archive: ' + name)
        sources[name] = {'path': str(Path(directory).resolve()), 'files': observed}
        authentication[name] = dict(auth)
    recipes = {str(Path(p).resolve()): regular(p)['sha256'] for p in spec['recipes']}
    if not recipes:
        raise ValueError('recipe files required')
    builds = []
    for directory in spec['cmake_builds']:
        build = Path(directory).resolve(strict=True)
        values = cache(build / 'CMakeCache.txt')
        commands = json.loads((build / 'compile_commands.json').read_text())
        if not isinstance(commands, list) or not commands:
            raise ValueError('configured compiler commands required')
        for command in commands:
            translation_unit = (Path(command['directory']) / command['file']).resolve(strict=True)
            if not any(translation_unit.is_relative_to(Path(item['path'])) for item in sources.values()):
                raise ValueError('compiler input outside authenticated source')
        compilers = []
        for key in sorted(values):
            if re.fullmatch('CMAKE_(?:C|CXX|Fortran)_COMPILER', key):
                path = Path(values[key]['value']).resolve(strict=True)
                compilers.append({'path': str(path), 'sha256': regular(path)['sha256']})
        if not compilers:
            raise ValueError('configured compiler identity required')
        graphs = {name: regular(build / name)['sha256'] for name in
                  ('build.ninja', 'CMakeFiles/rules.ninja', 'cmake_install.cmake')}
        builds.append({'build_directory': str(build), 'cache': values,
                       'compile_commands': commands, 'compilers': compilers, 'graphs': graphs})
    if not builds:
        raise ValueError('configured build required')
    data = {}
    for destination, item in spec['data_sources'].items():
        relative = Path(destination)
        source_path = Path(item['path'])
        if (relative.is_absolute() or '..' in relative.parts or relative == Path('.')
                or source_path.is_absolute() or '..' in source_path.parts):
            raise ValueError('data paths must be contained relative paths')
        data[destination] = sources[item['source']]['files'][source_path.as_posix()]
        if 'sha256' not in data[destination]:
            raise ValueError('data source must be regular')
    # Environment values are private build observations. Receipts belong outside
    # product trees and are never automatically included in a source/package.
    environment = {k: v for k, v in os.environ.items() if k in (
        'PATH', 'CC', 'CXX', 'FC', 'CFLAGS', 'CXXFLAGS', 'FFLAGS', 'CPPFLAGS', 'LDFLAGS',
        'SDKROOT', 'MACOSX_DEPLOYMENT_TARGET', 'CMAKE_TOOLCHAIN_FILE', 'CMAKE_PREFIX_PATH')}
    abi = {'system': platform.system(), 'machine': platform.machine(),
           'release': platform.release(), 'libc': list(platform.libc_ver()),
           'byteorder': sys.byteorder}
    semantic = {
        'sources_patches': identity({'archives': authentication, 'sources': sources, 'patches': []}),
        'recipe': identity(recipes),
        'toolchain': identity([b['compilers'] for b in builds]),
        'options': identity({'cmake': builds, 'environment': environment}),
        'abi': identity(abi), 'data': identity(data)}
    return {'sources': sources, 'source_authentication': {'status': 'archive_matched',
            'sources': authentication}, 'recipes': recipes, 'cmake': builds,
            'data': data, 'abi': abi, 'semantic_inputs': semantic}


def record(spec, output, command):
    output = Path(output)
    install = Path(spec['install'])
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ValueError('fresh absolute receipt destination required')
    parent = output.parent.resolve(strict=True)
    boundaries = [Path(spec['cwd']).resolve(), install.resolve()]
    boundaries.extend(Path(p).resolve() for p in spec['sources'].values())
    boundaries.extend(Path(p).resolve() for p in spec['cmake_builds'])
    if any(parent == root or root in parent.parents for root in boundaries):
        raise ValueError('receipt must be outside source, build and install trees')
    if not install.is_absolute() or install.is_symlink() or (install.exists() and any(install.iterdir())):
        raise ValueError('fresh empty absolute install required; cache hits are not builds')
    if not command:
        raise ValueError('build command required')
    before = snapshot(spec)
    observer = {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)}
    # Reserve the output before running anything; a failed build is retained and
    # cannot fall through into an unobserved retry under the same receipt name.
    with output.open('x') as stream:
        started = time.time()
        receipt = {'schema': SCHEMA, 'status': 'started', 'inputs': before, 'observer': observer,
                   'command': command, 'started_at_unix': started, 'outputs': {}}
        json.dump(receipt, stream, sort_keys=True); stream.flush(); os.fsync(stream.fileno())
        try:
            completed = subprocess.run(command, cwd=spec['cwd'], check=False)
            receipt['returncode'] = completed.returncode
            receipt['finished_at_unix'] = time.time()
            if completed.returncode:
                raise ValueError('build command failed; receipt retained')
            if observer['sha256'] != digest(__file__):
                raise ValueError('receipt producer changed during build')
            if before != snapshot(spec):
                raise ValueError('source, recipe, compiler, configuration or data changed during build')
            outputs = inventory(install)
            if not outputs:
                raise ValueError('build produced no installed files')
            manifest = Path(spec['artifact_manifest'])
            installed = {str((install / p).absolute()) for p in outputs}
            declared = set(manifest.read_text().splitlines())
            if installed != declared:
                raise ValueError('install manifest does not cover exactly the observed install')
            for path, expected in before['data'].items():
                if outputs.get(path) != expected:
                    raise ValueError('installed data differs from authenticated source: ' + path)
            receipt.update(status='recorded', outputs=outputs)
        except BaseException as error:
            receipt.update(status='failed', error=type(error).__name__ + ': ' + str(error))
            raise
        finally:
            stream.seek(0); json.dump(receipt, stream, sort_keys=True, allow_nan=False)
            stream.truncate(); stream.flush(); os.fsync(stream.fileno())
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['record'])
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    arguments = sys.argv[1:]
    separator = arguments.index('--') if '--' in arguments else len(arguments)
    args = parser.parse_args(arguments[:separator])
    command = arguments[separator + 1:]
    try:
        record(json.loads(Path(args.spec).read_text()), args.output, command)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, tarfile.TarError) as error:
        print('native receipt: ' + str(error), file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
