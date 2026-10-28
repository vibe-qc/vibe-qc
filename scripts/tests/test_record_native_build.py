"""Prospective collector protocol controls; no dependency build or reuse claim."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'record_native_build.py'
spec = importlib.util.spec_from_file_location('record_native_build', SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.product = self.root / 'product'; self.product.mkdir()
        self.source = self.product / 'source'; self.source.mkdir()
        (self.source / 'header.h').write_text('fixture data\n')
        self.archive = self.root / 'source.tar'
        with tarfile.open(self.archive, 'w') as stream:
            stream.add(self.source, arcname='dependency-1')
        self.build = self.product / 'build'; self.build.mkdir()
        (self.build / 'CMakeFiles').mkdir()
        self.compiler = self.root / 'compiler'; self.compiler.write_text('fixture compiler identity')
        self.recipe = self.product / 'recipe'; self.recipe.write_text('fixture recipe')
        (self.build / 'CMakeCache.txt').write_text('CMAKE_C_COMPILER:FILEPATH=' + str(self.compiler) + '\n')
        (self.build / 'compile_commands.json').write_text(json.dumps([{'directory':str(self.source), 'file':'header.h'}]))
        for name in ('build.ninja', 'CMakeFiles/rules.ninja', 'cmake_install.cmake'):
            (self.build / name).write_text('fixture graph')
        self.install = self.product / 'install'
        self.receipt = self.root / 'receipt.json'
        self.manifest = self.build / 'install_manifest.txt'
        self.spec = {'sources': {'fixture': str(self.source)},
            'authenticated_sources': {'fixture': {'archive':str(self.archive), 'prefix':'dependency-1',
                                                   'sha256':collector.digest(self.archive)}},
            'recipes':[str(self.recipe)], 'cmake_builds':[str(self.build)],
            'data_sources':{'include/header.h':{'source':'fixture','path':'header.h'}},
            'install':str(self.install), 'cwd':str(self.product), 'artifact_manifest':str(self.manifest)}
        code = ('from pathlib import Path; import shutil; '
                'p=Path(' + repr(str(self.install)) + '); (p/"include").mkdir(parents=True); '
                'shutil.copy2(' + repr(str(self.source/'header.h')) + ',p/"include/header.h"); '
                'Path(' + repr(str(self.manifest)) + ').write_text(str(p/"include/header.h")+"\\n")')
        self.command = [sys.executable, '-c', code]

    def test_complete_prospective_receipt_and_no_reuse(self):
        result = collector.record(self.spec, self.receipt, self.command)
        self.assertEqual(result['status'], 'recorded')
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(set(result['inputs']['semantic_inputs']),
                         {'sources_patches','recipe','toolchain','options','abi','data'})
        self.assertEqual(result['outputs']['include/header.h'], result['inputs']['data']['include/header.h'])
        with self.assertRaisesRegex(ValueError, 'fresh absolute receipt'):
            collector.record(self.spec, self.receipt, self.command)
        with self.assertRaisesRegex(ValueError, 'fresh empty'):
            collector.record(self.spec, self.root/'another.json', self.command)

    def test_cli_runs_exact_command(self):
        path = self.root/'spec.json'; path.write_text(json.dumps(self.spec))
        result = subprocess.run([sys.executable,str(SCRIPT),'record','--spec',str(path),
            '--output',str(self.receipt),'--',*self.command],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(self.receipt.read_text())['command'],self.command)

    def test_source_mismatch_refuses_before_command(self):
        (self.source/'header.h').write_text('tamper')
        with self.assertRaisesRegex(ValueError,'source differs'):
            collector.record(self.spec,self.receipt,self.command)
        self.assertFalse(self.install.exists()); self.assertFalse(self.receipt.exists())

    def test_failed_build_is_retained_never_recorded(self):
        with self.assertRaisesRegex(ValueError,'build command failed'):
            collector.record(self.spec,self.receipt,[sys.executable,'-c','raise SystemExit(17)'])
        data=json.loads(self.receipt.read_text())
        self.assertEqual(data['status'],'failed'); self.assertEqual(data['returncode'],17)

    def test_changed_recipe_compiler_and_data_are_refused(self):
        for target in [self.recipe, self.compiler, self.source/'header.h']:
            with self.subTest(target=target):
                original=target.read_text()
                command=[sys.executable,'-c','from pathlib import Path; Path('+repr(str(target))+').write_text("changed")']
                with self.assertRaises(ValueError):collector.record(self.spec,self.receipt,command)
                self.assertEqual(json.loads(self.receipt.read_text())['status'],'failed')
                target.write_text(original); self.receipt.unlink()

    def test_receipt_cannot_enter_product_tree(self):
        with self.assertRaisesRegex(ValueError,'outside source'):
            collector.record(self.spec,self.product/'receipt.json',self.command)
        self.assertFalse(self.install.exists())

    def test_archive_links_and_escape_are_refused(self):
        for name,kind in [('dependency-1/link',tarfile.SYMTYPE),('../escape',tarfile.REGTYPE)]:
            with self.subTest(name=name):
                with tarfile.open(self.archive,'w') as stream:
                    item=tarfile.TarInfo(name); item.type=kind; item.linkname='header.h';stream.addfile(item)
                with self.assertRaises(ValueError):
                    collector.archive_inventory(self.archive,'dependency-1',collector.digest(self.archive))

    def test_missing_manifest_refused(self):
        command=self.command.copy(); command[-1]+='; Path('+repr(str(self.manifest))+').write_text("")'
        with self.assertRaisesRegex(ValueError,'manifest'):
            collector.record(self.spec,self.receipt,command)
        self.assertEqual(json.loads(self.receipt.read_text())['status'],'failed')

    def test_wrong_installed_data_refused(self):
        command=self.command.copy()
        command[-1]+='; (p/"include/header.h").write_text("wrong data")'
        with self.assertRaisesRegex(ValueError,'installed data differs'):
            collector.record(self.spec,self.receipt,command)
        self.assertEqual(json.loads(self.receipt.read_text())['status'],'failed')

    def test_compile_input_outside_archive_refused(self):
        (self.build/'compile_commands.json').write_text(json.dumps([
            {'directory':str(self.root),'file':'compiler'}]))
        with self.assertRaisesRegex(ValueError,'compiler input outside'):
            collector.record(self.spec,self.receipt,self.command)
        self.assertFalse(self.install.exists())

    def test_data_destination_escape_refused(self):
        self.spec['data_sources']={'../escape':{'source':'fixture','path':'header.h'}}
        with self.assertRaisesRegex(ValueError,'contained relative'):
            collector.record(self.spec,self.receipt,self.command)
        self.assertFalse(self.install.exists())


if __name__ == '__main__':
    unittest.main()
