import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('publisher', Path(__file__).parents[1] / 'scripts/publish-desktop-release.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)

class PublicationTests(unittest.TestCase):
    def test_pair_and_source_are_atomic_and_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = []
            for name in ('mac', 'linux', 'source'):
                path = root / name
                path.write_bytes(name.encode())
                inputs.append(path)
            args = (root / 'updates', 'vibe-view', '2.20.20', 'a' * 40, *inputs)
            manifest = publisher.publish(*args)
            current = root / 'updates/vibe-view/current'
            self.assertTrue((current / 'download-macos.dmg').is_file())
            self.assertTrue((current / 'download-linux.AppImage').is_file())
            self.assertEqual(manifest, publisher.publish(*args))
            inputs[0].write_bytes(b'changed')
            with self.assertRaises(ValueError): publisher.publish(*args)
            self.assertEqual((current / 'download-macos.dmg').read_bytes(), b'mac')
            inputs[1].unlink()
            with self.assertRaises(ValueError): publisher.publish(*args)
            self.assertEqual((current / 'download-linux.AppImage').read_bytes(), b'linux')
    def test_rejects_traversal(self):
        with self.assertRaises(ValueError):
            publisher.publish(Path('.'), '../escape', '1.0.0', 'a'*40, Path('x'), Path('y'), Path('z'))

if __name__ == '__main__': unittest.main()
