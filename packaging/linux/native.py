"""Bundle non-glibc system runtime dependencies and distribution notices."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

BASE_LIBRARIES = re.compile(r'^(ld-linux.*|lib(c|m|mvec|pthread|dl|rt|util|resolv|anl)\.so.*)$')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def is_elf(path):
    if path.is_symlink() or not path.is_file():
        return False
    with path.open('rb') as stream:
        return stream.read(4) == b'\x7fELF'


def collect(root):
    root = root.resolve()
    destination = root / 'python/lib'
    notices = root / 'notices/system-libraries'
    notices.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    records = []
    relocations = []
    for path in list(root.rglob('*')):
        if not is_elf(path):
            continue
        # Wheel-private libraries inherit their extension's loader search path.
        # Inspect each library with its sibling directory available as well.
        env['LD_LIBRARY_PATH'] = os.pathsep.join((str(path.parent), str(destination)))
        result = subprocess.run(['ldd', str(path)], env=env, capture_output=True, text=True)
        missing = re.findall(r'^\s*(\S+) => not found', result.stdout, re.M)
        sibling_lib = path.parent.parent / 'lib'
        # Some wheels ship bin/ helpers whose libraries live in adjacent lib/.
        # Preserve these executables (including shared-memory helpers), and make
        # the loader relationship explicit instead of relying on a build host.
        if missing and path.parent.name == 'bin' and all(
            (sibling_lib / name).is_file() for name in missing
        ):
            before = digest(path)
            subprocess.run(['patchelf', '--add-rpath', '$ORIGIN/../lib', str(path)], check=True)
            relocations.append({'file': str(path.relative_to(root)),
                                'beforeSha256': before, 'sha256': digest(path),
                                'addedRunpath': '$ORIGIN/../lib'})
            result = subprocess.run(['ldd', str(path)], env=env, capture_output=True, text=True)
        if 'not found' in result.stdout:
            raise RuntimeError(f'{path}: {result.stdout}')
        for name, filename in re.findall(r'^\s*(\S+) => (/\S+)', result.stdout, re.M):
            dependency = Path(filename).resolve()
            if dependency.is_relative_to(root) or BASE_LIBRARIES.match(name):
                continue
            target = destination / name
            if target.exists():
                if digest(target) != digest(dependency):
                    raise ValueError('Conflicting library: ' + name)
                continue
            shutil.copyfile(dependency, target)
            subprocess.run(['patchelf', '--set-rpath', '$ORIGIN', str(target)], check=True)
            query = subprocess.run(['dpkg-query', '-S', str(dependency)], capture_output=True, text=True)
            if query.returncode and str(dependency).startswith('/usr/lib/'):
                query = subprocess.run(['dpkg-query', '-S', str(dependency)[4:]], capture_output=True, text=True)
            query.check_returncode()
            owner = query.stdout.split(': ', 1)[0]
            package = owner.split(':')[0]
            copyright_file = Path('/usr/share/doc') / package / 'copyright'
            shutil.copyfile(copyright_file, notices / (package + '-copyright'))
            records.append({'file': name, 'sha256': digest(target), 'package': owner})
    (root / 'notices/system-libraries.json').write_text(json.dumps(records, indent=2) + '\n')
    (root / 'notices/native-relocations.json').write_text(json.dumps(relocations, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    collect(Path(sys.argv[1]))
