"""Probe controls without compiling dependencies or qualifying historical data."""
import errno
import importlib.util
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_fftw_prefix.py"
spec = importlib.util.spec_from_file_location("fftw_prefix_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_constants_are_read_and_unknown_or_duplicate_rejected():
    header = b"#define FFTW_FORWARD (-1)\n#define FFTW_ESTIMATE (1U << 6)\n"
    assert probe.header_constants(header) == {"forward": -1, "estimate": 64}
    for bad in (header.replace(b"(-1)", b"(+1)"), header + header, b""):
        with pytest.raises(ValueError):
            probe.header_constants(bad)


def test_missing_or_readable_donor_never_counts_as_denial(tmp_path):
    with pytest.raises(ValueError, match="remains readable"):
        probe.require_denial(tmp_path)
    with pytest.raises(ValueError, match="not demonstrated"):
        probe.require_denial(tmp_path / "missing")
    for denied in (errno.EACCES, errno.EPERM):
        with patch.object(probe.os, "scandir", side_effect=OSError(denied, "denied")):
            probe.require_denial(tmp_path)


def test_paths_cannot_escape_or_alias(tmp_path):
    root = tmp_path.resolve()
    (root / "real").write_bytes(b"data")
    (root / "link").symlink_to(root / "real")
    for relative in ("../real", str(root / "real"), "link"):
        with pytest.raises(ValueError):
            probe.contained_file(root, relative)


def test_unsupported_isolation_is_hold():
    with patch.object(sys, "platform", "linux"):
        with pytest.raises(ValueError, match="isolation unsupported"):
            probe.run_probe({})


def test_child_requires_denial_before_native_call(tmp_path):
    with patch.object(probe, "native_operation") as operation:
        with pytest.raises(ValueError, match="remains readable"):
            probe.child_probe({"donor": str(tmp_path)})
        operation.assert_not_called()


def test_child_refuses_changed_candidate_bytes(tmp_path):
    root = tmp_path.resolve()
    (root / "include").mkdir()
    header = root / "include/fftw3.h"
    header.write_text("#define FFTW_FORWARD (-1)\n#define FFTW_ESTIMATE (1U << 6)\n")
    (root / "library").write_bytes(b"fixture only; never loaded")
    def mutation(*args):
        header.write_text("changed")
        return []
    with patch.object(probe, "require_denial"), patch.object(probe, "native_operation", side_effect=mutation):
        with pytest.raises(ValueError, match="bytes changed"):
            probe.child_probe({"donor": "unused", "candidate": str(root), "library": "library"})


def test_isolation_backend_really_denies_reads(tmp_path):
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        pytest.skip("Darwin sandbox backend only")
    donor = tmp_path.resolve() / "donor"
    donor.mkdir()
    secret = donor / "data"
    secret.write_text("must not be read")
    profile = '(version 1)(allow default)(deny file-read* (subpath (param "DONOR")))'
    code = '''import errno, pathlib, sys
try:
    pathlib.Path(sys.argv[1]).read_bytes()
except OSError as error:
    assert error.errno in (errno.EACCES, errno.EPERM), error
else:
    raise AssertionError("donor read succeeded")
'''
    result = probe.subprocess.run(
        ["/usr/bin/sandbox-exec", "-D", "DONOR=" + str(donor), "-p", profile,
         sys.executable, "-I", "-c", code, str(secret)],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert secret.read_text() == "must not be read"


def test_parent_rejects_overlapping_prefixes(tmp_path):
    with patch.object(sys, "platform", "darwin"), patch.object(Path, "is_file", return_value=True):
        with pytest.raises(ValueError, match="disjoint"):
            probe.run_probe({"candidate": str(tmp_path.resolve()),
                             "donor": str(tmp_path.resolve()), "library": "fake"})


@pytest.mark.parametrize("substituted", ["fftw_plan_dft_1d", "fftw_execute", "fftw_destroy_plan",
                                        "fftw_init_threads", "fftw_plan_with_nthreads", "fftw_cleanup_threads"])
def test_threaded_symbol_substitution_is_refused(substituted, tmp_path):
    core, threads = tmp_path / "core", tmp_path / "threads"
    library = MagicMock()
    names = ["fftw_plan_dft_1d", "fftw_execute", "fftw_destroy_plan",
             "fftw_init_threads", "fftw_plan_with_nthreads", "fftw_cleanup_threads"]
    paths = {id(getattr(library, name)): (tmp_path / "wrong" if name == substituted
             else core if index < 3 else threads) for index, name in enumerate(names)}
    with patch.object(probe.ctypes, "CDLL", return_value=library), \
         patch.object(probe.ctypes, "cast", side_effect=lambda fn, _: SimpleNamespace(value=id(fn))), \
         patch.object(probe, "loaded_symbol_path", side_effect=lambda address: paths[address]):
        with pytest.raises(ValueError, match="outside candidate library: " + substituted):
            probe.native_operation(core, {"forward": -1, "estimate": 64}, threads)
    library.fftw_init_threads.assert_not_called()
    library.fftw_plan_dft_1d.assert_not_called()


def test_threaded_initialization_failure_prevents_plan(tmp_path):
    library = MagicMock()
    library.fftw_init_threads.return_value = 0
    with patch.object(probe.ctypes, "CDLL", return_value=library), \
         patch.object(probe, "bound_function", side_effect=lambda lib, name, path: getattr(lib, name)):
        with pytest.raises(ValueError, match="initialization failed"):
            probe.native_operation(tmp_path / "core", {"forward": -1, "estimate": 64}, tmp_path / "threads")
    library.fftw_plan_dft_1d.assert_not_called()


# Source-only adapter controls. No CMake configure, dependency build or native
# library invocation is performed by these fixtures.
import io
import json
import tarfile

sys.path.insert(0, str(SCRIPT.parent))
import native_seed_adapter as adapter
import native_seed_context as context


def _configured_fixture(tmp_path):
    root = tmp_path / 'product'
    source, build, install = root / 'source', root / 'build', root / 'install'
    for path in (root / 'scripts', source, build / 'CMakeFiles'):
        path.mkdir(parents=True, exist_ok=True)
    for name in context.RECIPES:
        (root / 'scripts' / name).write_text('fixture\n')
    (root / 'scripts/build_fftw.sh').write_text('FFTW_VERSION="3.3.10"\nFFTW_SHA256="' + 'a' * 64 + '"\n')
    (source / 'a.c').write_text('source fixture')
    (build / 'CMakeCache.txt').write_text('OPTION:BOOL=ON\nCMAKE_CACHEFILE_DIR:INTERNAL=' + str(build) + '\n')
    (build / 'compile_commands.json').write_text(json.dumps([
        {'directory': str(build), 'file': str(source / 'a.c'),
         'command': 'cc -O2 -c ' + str(source / 'a.c') + ' -o a.o'}]))
    for name in ('build.ninja', 'CMakeFiles/rules.ninja', 'cmake_install.cmake'):
        (build / name).write_text('root ' + str(source) + ' ' + str(install) + '\n')
    execution = {'tools': {'compiler': {'sha256': 'c' * 64}}, 'abi': {'machine': 'test'},
                 'environment': {}, 'recipes': {name: context.digest(root / 'scripts' / name)
                                                for name in context.RECIPES}}
    return root, build, source, install, execution


def test_semantics_resolve_effective_inputs_and_relocation(tmp_path):
    root, build, source, install, execution = _configured_fixture(tmp_path / 'one')
    original = context.semantics(root, build, source, install, 'a' * 64, execution)
    other = _configured_fixture(tmp_path / 'two')
    assert original == context.semantics(*other[:4], 'a' * 64, other[4])
    (build / 'CMakeFiles/rules.ninja').write_text('changed link flags')
    changed = context.semantics(root, build, source, install, 'a' * 64, execution)
    assert changed['options'] != original['options']
    execution['abi']['machine'] = 'different'
    assert context.semantics(root, build, source, install, 'a' * 64, execution)['abi'] != original['abi']
    execution['tools']['compiler']['sha256'] = 'd' * 64
    assert context.semantics(root, build, source, install, 'a' * 64, execution)['toolchain'] != original['toolchain']
    with pytest.raises(ValueError, match='source archive'):
        context.semantics(root, build, source, install, 'b' * 64, execution)
    (root / 'scripts/_build_lock.sh').write_text('new recipe')
    with pytest.raises(ValueError, match='recipe files changed'):
        context.semantics(root, build, source, install, 'a' * 64, execution)


def test_manifest_allows_internal_file_alias_but_not_escape(tmp_path):
    (tmp_path / 'real').write_bytes(b'fixture')
    (tmp_path / 'alias').symlink_to('real')
    assert context.files(tmp_path)['alias'] == {'symlink': 'real'}
    (tmp_path / 'alias').unlink()
    (tmp_path / 'alias').symlink_to('../elsewhere')
    with pytest.raises(ValueError, match='sibling'):
        context.files(tmp_path)


@pytest.mark.parametrize('name,kind', [('fftw-3.3.10/../bad', 'file'), ('/outside', 'file'),
                                      ('fftw-3.3.10/link', 'link')])
def test_archive_extraction_rejects_unsafe_members(tmp_path, name, kind):
    archive = tmp_path / 'source.tar'
    with tarfile.open(archive, 'w') as stream:
        member = tarfile.TarInfo(name)
        if kind == 'link':
            member.type, member.linkname = tarfile.SYMTYPE, 'outside'
            stream.addfile(member)
        else:
            member.size = 1
            stream.addfile(member, io.BytesIO(b'x'))
    destination = tmp_path / 'source'
    destination.mkdir()
    with pytest.raises(ValueError):
        adapter.extract(archive, destination, 'fftw-3.3.10')


def test_configure_invocation_comes_from_recipe_and_refuses_shell(tmp_path):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    recipe = scripts / 'build_fftw.sh'
    recipe.write_text('cmake -S "$SRC_DIR" -B "$BUILD_DIR" -GNinja \\\n -DFLAG=ON -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"\n\n')
    argv = adapter.configuration(tmp_path, Path('/src'), Path('/build'), Path('/install'))
    assert argv == ['cmake', '-S', '/src', '-B', '/build', '-GNinja', '-DFLAG=ON',
                    '-DCMAKE_INSTALL_PREFIX=/install', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
    recipe.write_text(recipe.read_text().replace('-DFLAG=ON', '-DFLAG=$(unexpected)'))
    with pytest.raises(ValueError, match='unresolved'):
        adapter.configuration(tmp_path, Path('/src'), Path('/build'), Path('/install'))


def test_context_refuses_unsupported_platform_without_running_tools(tmp_path):
    with patch.object(sys, 'platform', 'linux'), patch.object(context, 'output') as run:
        with pytest.raises(ValueError, match='Darwin'):
            context.context(tmp_path, tmp_path)
        run.assert_not_called()


def test_verify_refuses_altered_plan_before_probe(tmp_path):
    donor, candidate = tmp_path / 'donor', tmp_path / 'candidate'
    donor.mkdir(); candidate.mkdir()
    (donor / context.RECEIPT).write_text('receipt fixture')
    record = {'inputs': {'x': 'old'}, 'installs': {}}
    request = {'plan': {'receipt': context.RECEIPT, 'receipt_sha256': 'wrong'}}
    with patch.object(adapter, 'request_paths', return_value=(donor, candidate)), \
         patch.object(adapter, 'envelope', return_value=record), \
         patch.object(adapter.subprocess, 'run') as run:
        with pytest.raises(ValueError, match='authenticated seed plan'):
            adapter.verify(request)
        run.assert_not_called()


def test_plan_derives_inputs_then_refuses_each_semantic_mismatch(tmp_path, monkeypatch):
    donor, candidate = tmp_path / 'donor', tmp_path / 'candidate'
    donor.mkdir(); candidate.mkdir()
    (donor / context.RECEIPT).write_text('protected fixture')
    archive = tmp_path / 'archive'
    archive.write_bytes(b'fixture')
    monkeypatch.setenv('VIBEQC_NATIVE_SOURCE_ARCHIVE', str(archive))
    archive_hash = context.digest(archive)
    inputs = {key: str(index) * 64 for index, key in enumerate(
        ('sources', 'patches', 'recipe', 'toolchain', 'options', 'abi'))}
    record = {'inputs': inputs, 'context': {'fixture': True}, 'installs': {},
              'collector': {'inputs': {'source_authentication': {'sources': {
                  'fftw': {'sha256': archive_hash, 'prefix': 'fftw-3.3.10'}}}}}}
    with patch.object(adapter, 'request_paths', return_value=(donor, candidate)), \
         patch.object(adapter, 'envelope', return_value=record), \
         patch.object(context, 'recipe_pin', return_value=('3.3.10', archive_hash)), \
         patch.object(adapter, 'extract'), patch.object(context, 'files', return_value={}), \
         patch.object(adapter, 'configuration', return_value=['configure-fixture']), \
         patch.object(adapter.subprocess, 'run', return_value=type('Result', (), {'returncode': 0})()) as run, \
         patch.object(context, 'context', return_value=record['context']), \
         patch.object(context, 'semantics', return_value=inputs) as resolve:
        assert adapter.plan({})['candidate_inputs'] == inputs
        assert run.call_args.args[0] == ['configure-fixture']
        for key in inputs:
            resolve.return_value = dict(inputs, **{key: 'f' * 64})
            with pytest.raises(ValueError, match=key):
                adapter.plan({})


def test_seal_checks_collector_source_options_compiler_and_outputs(tmp_path):
    root, build, source, install, execution = _configured_fixture(tmp_path)
    expected_source = root / 'third_party/fftw/src'
    expected_source.parent.mkdir(parents=True)
    source.rename(expected_source)
    # Match the fixed production layout and configured compiler identity.
    commands = [{'directory': str(build), 'file': str(expected_source / 'a.c'), 'command': 'cc -c a.c'}]
    (build / 'compile_commands.json').write_text(json.dumps(commands))
    install = root / context.PREFIX
    install.mkdir()
    (install / 'library').write_bytes(b'fixture')
    compiler = {'path': '/fixture/cc', 'sha256': 'c' * 64}
    execution['tools']['CMAKE_C_COMPILER'] = compiler
    receipt = {'schema': 'vibe.native-build-receipt/1', 'status': 'recorded', 'returncode': 0,
               'inputs': {'source_authentication': {'status': 'archive_matched', 'sources': {
                   'fftw': {'sha256': 'a' * 64, 'prefix': 'fftw-3.3.10'}}},
                          'sources': {'fftw': {'files': context.files(expected_source)}},
                          'recipes': {str(root / 'scripts' / name): execution['recipes'][name]
                                      for name in context.RECIPES},
                          'cmake': [{'build_directory': str(build), 'cache': context.cache(build),
                                     'compile_commands': commands, 'compilers': [compiler]}]},
               'outputs': context.files(install)}
    external, before = tmp_path / 'receipt.json', tmp_path / 'before.json'
    external.write_text(json.dumps(receipt)); before.write_text(json.dumps(execution))
    with patch.object(context, 'context', return_value=execution):
        result = context.seal(root, build, external, before)
        assert result['installs'][context.PREFIX] == receipt['outputs']
        (root / context.RECEIPT).unlink()
        receipt['outputs']['library']['sha256'] = 'f' * 64
        external.write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match='install differs'):
            context.seal(root, build, external, before)
        receipt['status'] = 'unqualified'
        external.write_text(json.dumps(receipt))
        with pytest.raises(ValueError, match='successful prospective'):
            context.seal(root, build, external, before)


def test_context_failure_stops_hook_before_collector(tmp_path, monkeypatch):
    import hashlib
    import os
    archive = tmp_path / 'archive'
    archive.write_bytes(b'fixture archive')
    scripts, product, receipts = tmp_path / 'scripts', tmp_path / 'product', tmp_path / 'receipts'
    for path in (scripts, product, receipts):
        path.mkdir()
    (scripts / 'native_seed_context.py').write_text('raise SystemExit(7)\n')
    collector = tmp_path / 'collector'
    witness = tmp_path / 'called'
    collector.write_text('#!/bin/sh\ntouch "' + str(witness) + '"\n')
    collector.chmod(0o755)
    env = dict(os.environ, VIBEQC_NATIVE_BUILD_COLLECTOR=str(collector),
               VIBEQC_NATIVE_RECEIPT_DIR=str(receipts), VIBEQC_NATIVE_SOURCE_ARCHIVE=str(archive),
               VIBEQC_NATIVE_SEED_CONTEXT='1', REPO_ROOT=str(product), SCRIPT_DIR=str(scripts),
               SRC_DIR=str(product / 'src'), BUILD_DIR=str(product / 'build'),
               INSTALL_DIR=str(product / 'install'), FFTW_VERSION='3.3.10',
               FFTW_SHA256=hashlib.sha256(archive.read_bytes()).hexdigest())
    result = probe.subprocess.run(['bash', '-c',
        'source "$1"; vibeqc_enable_native_collector fftw || exit $?; cmake --build "$BUILD_DIR" --target install',
        'hook-test', str(SCRIPT.parent / '_native_collector.sh')],
        env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 7, result.stderr
    assert not witness.exists()


@pytest.mark.parametrize('variable', ['CFLAGS', 'CLANG_CONFIG_FILE_USER_DIR', 'CCC_OVERRIDE_OPTIONS'])
def test_context_refuses_unattested_environment_before_tool_execution(tmp_path, variable):
    import os
    with patch.object(sys, 'platform', 'darwin'), patch.dict(os.environ, {variable: 'external'}, clear=True), \
         patch.object(context, 'output') as run:
        with pytest.raises(ValueError, match='unsupported'):
            context.context(tmp_path, tmp_path)
        run.assert_not_called()
