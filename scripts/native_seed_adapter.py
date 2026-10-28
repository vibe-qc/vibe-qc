#!/usr/bin/env python3
"""Opt-in Queue adapter for prospectively qualified, unpatched FFTW installs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_seed_context as native


def tracked(root, sha, names):
    for name in names:
        path = root / name
        committed = subprocess.check_output(['git', '-C', str(root), 'show', sha + ':' + name], timeout=30)
        if path.is_symlink() or path.read_bytes() != committed:
            raise ValueError('candidate helper differs from tracked commit: ' + name)


def extract(archive, destination, prefix):
    """No links, devices, duplicate members or escaping paths in source inputs."""
    with tarfile.open(archive, 'r:*') as stream:
        seen = set()
        for member in stream.getmembers():
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != prefix:
                raise ValueError('source archive member outside pinned extraction prefix')
            relative = Path(*path.parts[1:])
            if relative == Path('.'):
                if member.isdir():
                    continue
                raise ValueError('archive prefix is not a directory')
            if relative in seen or not (member.isdir() or member.isfile()):
                raise ValueError('duplicate or non-regular source archive member')
            seen.add(relative)
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, target.open('xb') as output:
                    while block := source.read(65536):
                        output.write(block)
                target.chmod(member.mode & 0o777)


def configuration(root, source, build, install):
    """Parse the actual supported recipe invocation, not a second flag table."""
    script = (root / 'scripts/build_fftw.sh').read_text()
    starts = [part for part in script.split('\n\n') if part.startswith('cmake -S ')]
    if len(starts) != 1:
        raise ValueError('unsupported CMake recipe shape')
    command = starts[0].replace('\\\n', ' ')
    for name, value in {'SRC_DIR': source, 'BUILD_DIR': build, 'INSTALL_DIR': install}.items():
        command = command.replace('$' + name, str(value))
    if '$' in command or '`' in command or ';' in command:
        raise ValueError('unresolved or executable configure expression')
    argv = shlex.split(command)
    if argv[:2] != ['cmake', '-S'] or '--build' in argv or '--install' in argv:
        raise ValueError('only the recipe configure invocation is supported')
    return argv + ['-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']


def request_paths(request):
    if request.get('schema') != 'vq.native-seed-request/1' or request.get('phase') not in ('plan', 'verify'):
        raise ValueError('unsupported Queue request')
    roots = [Path(request[role]['source']) for role in ('donor', 'candidate')]
    if any(not root.is_absolute() or root.resolve(strict=True) != root for root in roots):
        raise ValueError('canonical existing source roots required')
    donor, candidate = roots
    if donor == candidate or donor in candidate.parents or candidate in donor.parents:
        raise ValueError('donor and candidate roots must be disjoint')
    names = ['scripts/' + name for name in (*native.RECIPES, 'native_seed_adapter.py',
                                          'probe_fftw_prefix.py', 'attest_native_image.py')]
    tracked(candidate, request['candidate']['sha'], names)
    return donor, candidate


def envelope(donor):
    path = donor / native.RECEIPT
    if path.is_symlink():
        raise ValueError('receipt aliases unsupported')
    record = json.loads(path.read_text())
    if record.get('schema') != 'vibeqc.native-fftw-seed/1':
        raise ValueError('fresh public FFTW context envelope missing')
    collector = record['collector']
    if (collector.get('schema') != 'vibe.native-build-receipt/1' or collector.get('status') != 'recorded'
            or type(collector.get('returncode')) is not int or collector['returncode'] != 0):
        raise ValueError('successful fresh collector receipt required')
    auth = collector['inputs']['source_authentication']
    if auth['status'] != 'archive_matched' or set(auth['sources']) != {'fftw'}:
        raise ValueError('authenticated unpatched FFTW archive required')
    if set(record['installs']) != {native.PREFIX}:
        raise ValueError('only the FFTW install is qualified')
    if native.files(donor / native.PREFIX) != record['installs'][native.PREFIX]:
        raise ValueError('donor install differs from complete manifest')
    if collector['outputs'] != record['installs'][native.PREFIX]:
        raise ValueError('envelope differs from observed fresh collector outputs')
    if set(record['inputs']) != {'sources', 'patches', 'recipe', 'toolchain', 'options', 'abi'}:
        raise ValueError('incomplete semantic categories')
    return record


def plan(request):
    donor, candidate = request_paths(request)
    record = envelope(donor)
    version, expected = native.recipe_pin(candidate)
    authentication = record['collector']['inputs']['source_authentication']['sources']['fftw']
    if authentication['sha256'] != expected or authentication['prefix'] != 'fftw-' + version:
        raise ValueError('donor source pin differs from candidate')
    archive = Path(os.environ.get('VIBEQC_NATIVE_SOURCE_ARCHIVE', ''))
    if not archive.is_absolute() or not archive.is_file() or archive.is_symlink() or native.digest(archive) != expected:
        raise ValueError('the recipe-pinned source archive must be explicitly supplied')
    install = candidate / native.PREFIX
    if install.exists() or install.is_symlink():
        raise ValueError('candidate install must not exist before seed planning')
    with tempfile.TemporaryDirectory(prefix='vibeqc-fftw-configure-') as temporary:
        temporary = Path(temporary).resolve()
        source, build = temporary / 'src', temporary / 'build'
        source.mkdir()
        extract(archive, source, 'fftw-' + version)
        source_files = native.files(source)
        # CMake performs compiler identification, but never an FFTW target
        # build/install. This occurs only inside an already authorized slot.
        result = subprocess.run(configuration(candidate, source, build, install), cwd=temporary,
                                capture_output=True, text=True, timeout=120, check=False)
        if result.returncode:
            raise ValueError('candidate configure failed: ' + result.stderr[-2000:])
        if source_files != native.files(source) or native.digest(archive) != expected:
            raise ValueError('source or archive changed during candidate configure')
        execution = native.context(build, candidate)
        if execution != record['context']:
            raise ValueError('candidate ABI, tools or environment differ from fresh build')
        inputs = native.semantics(candidate, build, source, install, expected, execution)
    if inputs != record['inputs']:
        differences = sorted(key for key in inputs if inputs[key] != record['inputs'][key])
        raise ValueError('candidate semantic inputs differ: ' + ', '.join(differences))
    return {'schema': 'vq.native-seed-plan/1', 'decision': 'qualified', 'receipt': native.RECEIPT,
            'receipt_sha256': native.digest(donor / native.RECEIPT),
            'donor_inputs': record['inputs'], 'candidate_inputs': inputs, 'installs': record['installs']}


def verify(request):
    donor, candidate = request_paths(request)
    record = envelope(donor)
    supplied = request['plan']
    if (supplied['receipt'] != native.RECEIPT or supplied['receipt_sha256'] != native.digest(donor / native.RECEIPT)
            or supplied['installs'] != record['installs'] or supplied['donor_inputs'] != record['inputs']
            or supplied['candidate_inputs'] != record['inputs']):
        raise ValueError('verification does not match the authenticated seed plan')
    install = candidate / native.PREFIX
    if native.files(install) != record['installs'][native.PREFIX]:
        raise ValueError('candidate install differs from complete manifest')
    # Resolve aliases to regular manifest entries so Queue checks the same bytes.
    def library(pattern):
        matches = {path.resolve() for path in install.glob(pattern)}
        if len(matches) != 1:
            raise ValueError('expected one candidate library for ' + pattern)
        path = matches.pop()
        relative = path.relative_to(install).as_posix()
        if 'sha256' not in record['installs'][native.PREFIX][relative]:
            raise ValueError('observed library is not a regular manifested file')
        return relative
    probe_request = {'candidate': str(install), 'donor': str(donor),
                     'library': library('lib/libfftw3.*.dylib'),
                     'threads_library': library('lib/libfftw3_threads.*.dylib')}
    response = subprocess.run([sys.executable, '-B', '-I', str(candidate / 'scripts/probe_fftw_prefix.py')],
                              input=json.dumps(probe_request), text=True, capture_output=True,
                              timeout=45, check=False)
    if response.returncode:
        raise ValueError('isolated final-prefix probe failed: ' + response.stderr[-2000:])
    observed = json.loads(response.stdout)
    if observed.get('decision') != 'observed' or observed.get('isolation') != 'denied-donor-prefix' or not observed.get('threaded'):
        raise ValueError('serial and threaded donor-denied observations required')
    if native.files(install) != record['installs'][native.PREFIX]:
        raise ValueError('candidate install changed during verification')
    return {'schema': 'vq.native-seed-verification/1', 'decision': 'qualified',
            'isolation': 'denied-donor-prefix', 'operation': 'FFTW serial and configured-threaded transforms',
            'result': {'serial': observed['result'], 'threaded': observed['threaded']},
            'libraries': observed['libraries'], 'data': observed['data']}


def main():
    try:
        request = json.load(sys.stdin)
        result = plan(request) if request.get('phase') == 'plan' else verify(request)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, tarfile.TarError) as error:
        print(json.dumps({'decision': 'hold', 'reason': str(error)}))
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
