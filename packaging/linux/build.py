#!/usr/bin/env python3
"""Assemble a relocatable CLI runtime from an immutable source archive and locked inputs.

Run natively on Linux x86-64. Inputs are prepared in the release builder; no
network resolution happens here. The application wheel must be built from the
supplied source commit. Native QC wheels must first have their dependency closure
repaired. Keep this recipe and the input manifest with the source supplement.
"""
import argparse
import hashlib
import json
import os
import platform
import re
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(command, **kwargs):
    subprocess.run([str(x) for x in command], check=True, **kwargs)


def build(args):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('Build natively on Linux x86-64')
    lock = json.loads(args.lock.read_text())
    if lock['product'] not in ('vibe-qc', 'vibe-queue') or not re.fullmatch(r'[a-f0-9]{40}', lock['sourceCommit']):
        raise ValueError('Expected a supported application and immutable release commit')
    for name in [lock['python'], *lock['wheels'], *lock['commands']]:
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Unsafe input filename')
    for item in lock['inputs']:
        if digest(args.inputs / item['filename']) != item['sha256']:
            raise ValueError('Input hash mismatch: ' + item['filename'])
    if digest(args.source) != lock['sourceSha256'] or digest(args.wheel) != lock['wheelSha256']:
        raise ValueError('Application source or wheel hash mismatch')
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cli-package-', dir=args.output) as temp:
        root = Path(temp) / lock['product']
        root.mkdir()
        with tarfile.open(args.inputs / lock['python']) as archive:
            archive.extractall(root, filter='data')
        python = root / 'python/bin/python3'
        env = {k:v for k,v in os.environ.items() if not k.startswith(('PYTHON', 'VQ_', 'VIBEQC_', 'LIBINT_'))}
        env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
        pip = [python, '-I', '-B', '-m', 'pip', '--isolated', '--disable-pip-version-check']
        run([*pip, 'install', '--no-index', '--no-deps', '--no-compile', args.wheel,
             *[args.inputs / name for name in lock['wheels']]], env=env)
        run([*pip, 'check'], env=env)
        run([*pip, 'uninstall', '-y', 'pip', 'setuptools', 'wheel'], env=env)
        for path in (root / 'python/bin').iterdir():
            if not path.name.startswith('python'):
                path.unlink()
        for path in root.rglob('__pycache__'):
            shutil.rmtree(path)
        for pattern in ('*.pyc', '*.a', 'direct_url.json'):
            for path in root.rglob(pattern):
                path.unlink()
        for name in ('include', 'share', 'lib/pkgconfig'):
            shutil.rmtree(root / 'python' / name, ignore_errors=True)
        # Native libraries supplied by the qualified build, with matching notices.
        if args.libraries:
            shutil.copytree(args.libraries, root / 'python/lib', dirs_exist_ok=True)
        notices = root / 'notices'
        shutil.copytree(args.notices, notices)
        shutil.copyfile(args.source, notices / (lock['product'] + '-source.tar.gz'))
        shutil.copyfile(args.lock, notices / 'linux-runtime-lock.json')
        shutil.copyfile(__file__, notices / 'build-linux.py')
        from native import collect
        collect(root)
        shutil.copyfile(Path(__file__).with_name('native.py'), notices / 'native.py')
        # Notices must remain readable after an administrator installs the archive.
        for path in notices.rglob('*'):
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() else 0o644)
        (root / 'runtime.json').write_text(json.dumps(lock, indent=2) + '\n')
        (root / 'run.py').write_text('''import importlib.metadata as m, sys
name = sys.argv.pop(1)
sys.argv[0] = name
entry = next(e for e in m.entry_points(group="console_scripts") if e.name == name)
raise SystemExit(entry.load()())
''')
        (root / 'bin').mkdir()
        for command in lock['commands']:
            launcher = root / 'bin' / command
            launcher.write_text('''#!/bin/sh
HERE=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$HERE/python/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$HERE/python/bin/python3" -I -B "$HERE/run.py" "''' + command + '''" "$@"
''')
            launcher.chmod(0o755)
        (root / 'README.txt').write_text(lock['instructions'] + '\n')
        final = args.output / lock['product']
        if final.exists():
            raise ValueError('Refusing to overwrite existing package directory')
        root.rename(final)
        archive = args.output / f"{lock['product']}-{lock['version']}-x86_64.tar.gz"
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(final, arcname=lock['product'])
        print(archive)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('lock', 'inputs', 'source', 'wheel', 'notices', 'output'):
        p.add_argument('--' + field, type=Path, required=True)
    p.add_argument('--libraries', type=Path)
    build(p.parse_args())
