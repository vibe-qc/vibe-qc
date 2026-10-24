"""The CLI archive format must preserve paired, immutable publication."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('publisher', Path(__file__).parents[1] / 'scripts/publish-desktop-release.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Publication(unittest.TestCase):
    def test_cli_pair_and_immutable_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = [root / name for name in ('mac', 'linux', 'source')]
            for path in files:
                path.write_bytes(path.name.encode())
            for product in ('vibe-qc', 'vibe-queue'):
                result = module.publish(root / 'updates', product, '1.2.3', 'a' * 40, *files)
                current = root / 'updates' / product / 'current'
                self.assertTrue(current.is_symlink())
                self.assertEqual((current / 'download-linux.tar.gz').read_bytes(), b'linux')
                self.assertTrue(result['artifacts']['linux']['filename'].endswith('-x86_64.tar.gz'))
                self.assertEqual(module.publish(root / 'updates', product, '1.2.3', 'a' * 40, *files), result)
                files[1].write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    module.publish(root / 'updates', product, '1.2.3', 'a' * 40, *files)
                self.assertEqual((current / 'download-linux.tar.gz').read_bytes(), b'linux')
                files[1].write_bytes(b'linux')


if __name__ == '__main__':
    unittest.main()
