#!/usr/bin/env python3
"""Prospective, conservative FFTW build semantics; no retrospective qualification."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time

RECEIPT = '.vq-native-fftw.json'
PREFIX = 'third_party/fftw/install'
RECIPES = ('build_fftw.sh', '_native_collector.sh', '_safe_build_env.sh',
           '_verify_source.sh', '_build_lock.sh', 'native_seed_context.py')
ENVIRONMENT = ('PATH', 'CC', 'CXX', 'FC', 'CFLAGS', 'CXXFLAGS', 'FFLAGS', 'CPPFLAGS',
               'LDFLAGS', 'CMAKE_TOOLCHAIN_FILE', 'SDKROOT', 'MACOSX_DEPLOYMENT_TARGET',
               'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH',
               'CMAKE_PREFIX_PATH', 'CMAKE_LIBRARY_PATH', 'CMAKE_INCLUDE_PATH',
               'CMAKE_FRAMEWORK_PATH', 'CMAKE_APPBUNDLE_PATH', 'PKG_CONFIG_PATH',
               'PKG_CONFIG_LIBDIR', 'CMAKE_GENERATOR', 'ARCHFLAGS', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH',
               'DYLD_FALLBACK_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def files(root):
    """Complete inventory; only sibling file aliases are supported by seeding."""
    root = Path(root).resolve(strict=True)
    result = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        directory = Path(directory)
        if any((directory / name).is_symlink() for name in dirs):
            raise ValueError('directory aliases are unsupported')
        for name in sorted(names):
            path = directory / name
            key = path.relative_to(root).as_posix()
            if path.is_symlink():
                target = os.readlink(path)
                if '/' in target or target in ('', '.', '..') or not path.resolve().is_file():
                    raise ValueError('only sibling regular-file aliases are supported')
                if not path.resolve().is_relative_to(root):
                    raise ValueError('alias escapes inventory')
                result[key] = {'symlink': target}
            else:
                info = path.stat()
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError('unsupported file type or hardlink')
                result[key] = {'sha256': digest(path), 'mode': stat.S_IMODE(info.st_mode)}
    return result


def tree_digest(root):
    """Fingerprint toolchain data, including contained directory aliases."""
    root = Path(root).resolve(strict=True)
    entries = {}
    deadline = time.monotonic() + 90
    total_bytes = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        directory = Path(directory)
        for name in sorted(dirs + names):
            path = directory / name
            key = path.relative_to(root).as_posix()
            if time.monotonic() > deadline or len(entries) > 100000 or total_bytes > 8 * 1024**3:
                raise ValueError('toolchain fingerprint resource bound exceeded')
            if path.is_symlink():
                if not path.resolve(strict=True).is_relative_to(root):
                    raise ValueError('toolchain alias escapes fingerprint root')
                entries[key] = {'symlink': os.readlink(path)}
            elif path.is_file():
                total_bytes += path.stat().st_size
                if total_bytes > 8 * 1024**3:
                    raise ValueError('toolchain fingerprint byte bound exceeded')
                entries[key] = {'sha256': digest(path), 'mode': stat.S_IMODE(path.stat().st_mode)}
    if not entries:
        raise ValueError('empty toolchain tree')
    return identity(entries)


def cache(build):
    result = {}
    for line in (Path(build) / 'CMakeCache.txt').read_text().splitlines():
        match = re.fullmatch(r'([^/#][^:]*):([^=]+)=(.*)', line)
        if match:
            result[match[1]] = {'type': match[2], 'value': match[3]}
    if not result:
        raise ValueError('configured CMake cache missing')
    return result


def output(argv):
    return subprocess.check_output(argv, text=True, timeout=30).strip()


def context(build, root):
    """Resolve the actual host and tools, never an operator-asserted ABI hash."""
    if sys.platform != 'darwin':
        raise ValueError('only the Darwin Apple-clang/system-threads context is supported')
    if any(os.environ.get(name) for name in ENVIRONMENT if name != 'PATH'):
        raise ValueError('compiler, flags, include, SDK or toolchain overrides unsupported')
    hidden = [name for name, value in os.environ.items() if value and (
        name.startswith(('CLANG_', 'CCC_', 'NIX_', 'COMPILER_PATH', 'GCC_EXEC_PREFIX'))
        or (name.startswith('CMAKE_') and name != 'CMAKE_BUILD_PARALLEL_LEVEL')
        or name == 'DEVELOPER_DIR')]
    if hidden or (Path.home() / '.config/clang').exists():
        raise ValueError('external compiler or discovery configuration unsupported')
    values = cache(build)
    for name, entry in values.items():
        if ('COMPILER_LAUNCHER' in name or name == 'CMAKE_TOOLCHAIN_FILE') and entry['value']:
            raise ValueError('compiler launchers and toolchain files require transitive attestation')
        if name.startswith('OpenMP_') and ('LIBRARY' in name or 'FLAGS' in name):
            if entry['value'] and not entry['value'].endswith('-NOTFOUND'):
                raise ValueError('external OpenMP runtime is outside the supported context')
    tools = {}
    for name in ('CMAKE_C_COMPILER', 'CMAKE_COMMAND', 'CMAKE_MAKE_PROGRAM'):
        path = Path(values[name]['value']).resolve(strict=True)
        tools[name] = {'path': str(path), 'sha256': digest(path)}
    compiler = Path(tools['CMAKE_C_COMPILER']['path'])
    if 'Apple clang' not in output([str(compiler), '--version']):
        raise ValueError('only unwrapped Apple clang is supported')
    selected = Path(output(['/usr/bin/xcrun', '--find', 'clang'])).resolve(strict=True)
    if compiler != selected:
        raise ValueError('configured compiler differs from selected developer toolchain')
    sdk = Path(output(['/usr/bin/xcrun', '--show-sdk-path'])).resolve(strict=True)
    toolchain = compiler.parent.parent
    tools['sdk'] = {'path': str(sdk), 'tree_sha256': tree_digest(sdk)}
    tools['toolchain'] = {'path': str(toolchain), 'tree_sha256': tree_digest(toolchain)}
    root = Path(root).resolve(strict=True)
    recipes = {name: digest(root / 'scripts' / name) for name in RECIPES}
    return {'schema': 'vibeqc.native-context/1', 'tools': tools, 'recipes': recipes,
            'environment': {name: os.environ[name] for name in ENVIRONMENT if name in os.environ},
            'abi': {'system': platform.system(), 'machine': platform.machine(),
                    'release': platform.release(), 'version': platform.version(),
                    'mac_version': platform.mac_ver()[0], 'byteorder': sys.byteorder}}


def recipe_pin(root):
    script = (Path(root) / 'scripts/build_fftw.sh').read_text()
    def one(name, pattern):
        matches = re.findall(r'^' + name + r'="(' + pattern + r')"$', script, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError('ambiguous or missing FFTW source pin')
        return matches[0]
    return one('FFTW_VERSION', r'[0-9]+(?:\.[0-9]+)+'), one('FFTW_SHA256', r'[0-9a-f]{64}')


def normalized(value, replacements):
    """Normalize only the three known relocation roots, never arbitrary flags."""
    if isinstance(value, str):
        for old, new in sorted(replacements.items(), key=lambda item: -len(item[0])):
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [normalized(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: normalized(item, replacements) for key, item in value.items()}
    return value


def semantics(root, build, source, install, source_hash, execution):
    root, build, source, install = map(Path, (root, build, source, install))
    version, pinned = recipe_pin(root)
    if source_hash != pinned:
        raise ValueError('source archive differs from candidate recipe')
    values = cache(build)
    commands = json.loads((build / 'compile_commands.json').read_text())
    if not isinstance(commands, list) or not commands:
        raise ValueError('configured compiler commands missing')
    for command in commands:
        resolved = (Path(command['directory']) / command['file']).resolve()
        if not resolved.is_relative_to(source.resolve()):
            raise ValueError('compiler input outside authenticated source')
    # These cache entries only identify the producing CMake directory. Retain
    # every other discovered option, including compiler/platform feature tests.
    values.pop('CMAKE_CACHEFILE_DIR', None)
    replacements = {str(build): '<BUILD>', str(source): '<SOURCE>', str(install): '<INSTALL>'}
    recipes = {name: digest(root / 'scripts' / name) for name in RECIPES}
    if recipes != execution['recipes']:
        raise ValueError('recipe files changed during semantic resolution')
    return {'sources': identity({'archive': pinned, 'prefix': 'fftw-' + version}),
            'patches': identity([]), 'recipe': identity(recipes),
            'toolchain': identity(execution['tools']),
            'options': identity({'cache': normalized(values, replacements),
                                 'commands': normalized(commands, replacements),
                                 'graphs': {name: normalized((build / name).read_text(), replacements) for name in
                                            ('build.ninja', 'CMakeFiles/rules.ninja', 'cmake_install.cmake')},
                                 'environment': execution['environment']}),
            'abi': identity(execution['abi'])}


def seal(root, build, external_receipt, before_file):
    """Run only after the observed fresh install, before slot activation."""
    root, build = Path(root).resolve(), Path(build).resolve()
    before = json.loads(Path(before_file).read_text())
    after = context(build, root)
    if before != after:
        raise ValueError('build execution context changed during compilation')
    receipt = json.loads(Path(external_receipt).read_text())
    if (receipt.get('schema') != 'vibe.native-build-receipt/1'
            or receipt.get('status') != 'recorded' or type(receipt.get('returncode')) is not int
            or receipt['returncode'] != 0):
        raise ValueError('successful prospective collector receipt required')
    inputs = receipt['inputs']
    auth = inputs['source_authentication']
    if auth['status'] != 'archive_matched' or set(auth['sources']) != {'fftw'}:
        raise ValueError('unpatched authenticated FFTW source required')
    version, pinned = recipe_pin(root)
    if auth['sources']['fftw'].get('prefix') != 'fftw-' + version or auth['sources']['fftw'].get('sha256') != pinned:
        raise ValueError('collector archive differs from recipe pin')
    if set(inputs['sources']) != {'fftw'} or len(inputs['cmake']) != 1:
        raise ValueError('only one fresh FFTW configure/build is supported')
    source = root / 'third_party/fftw/src'
    install = root / PREFIX
    recorded = inputs['cmake'][0]
    if Path(recorded['build_directory']).resolve() != build:
        raise ValueError('collector build directory mismatch')
    if recorded['compile_commands'] != json.loads((build / 'compile_commands.json').read_text()):
        raise ValueError('collector compiler commands differ')
    compiler = after['tools']['CMAKE_C_COMPILER']
    if len(recorded['compilers']) != 1 or any(recorded['compilers'][0].get(key) != compiler[key] for key in ('path', 'sha256')):
        raise ValueError('collector compiler differs from captured context')
    for name in RECIPES:
        if name == 'native_seed_context.py':
            continue  # The before/after public context directly binds itself.
        if inputs['recipes'].get(str(root / 'scripts' / name)) != after['recipes'][name]:
            raise ValueError('collector recipe identity differs')
    if recorded['cache'] != cache(build):
        raise ValueError('collector configured options differ from current cache')
    if inputs['sources']['fftw']['files'] != files(source):
        raise ValueError('source bytes differ from authenticated collector input')
    outputs = files(install)
    if not outputs or outputs != receipt['outputs']:
        raise ValueError('complete install differs from fresh collector output')
    semantics_value = semantics(root, build, source, install, auth['sources']['fftw']['sha256'], after)
    result = {'schema': 'vibeqc.native-fftw-seed/1', 'collector': receipt,
              'context': after, 'inputs': semantics_value, 'installs': {PREFIX: outputs}}
    destination = root / RECEIPT
    with destination.open('x') as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
    return result


def main():
    try:
        if len(sys.argv) == 5 and sys.argv[1] == 'before':
            value = context(Path(sys.argv[2]), Path(sys.argv[3]))
            with Path(sys.argv[4]).open('x') as stream:
                json.dump(value, stream, sort_keys=True)
        elif len(sys.argv) == 6 and sys.argv[1] == 'seal':
            seal(*sys.argv[2:])
        else:
            raise ValueError('use before BUILD ROOT OUTPUT or seal ROOT BUILD RECEIPT BEFORE')
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print('native context: ' + str(error), file=sys.stderr)
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
