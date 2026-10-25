from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.ci import native_cache


class NativeCacheTests(unittest.TestCase):
    def test_lifecycle_edits_reuse_cache_but_build_changes_cannot(self):
        """Exercise the real stamp normalizer and stale-key refusal, without builds."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            for name in [*(f"build_{dep}.sh" for dep in native_cache.DEPENDENCIES),
                         "_native_stamp.sh", "_libint_max_am.sh", "_verify_source.sh"]:
                shutil.copy2(native_cache.ROOT / "scripts" / name, root / "scripts" / name)
            shutil.copy2(native_cache.ROOT / ".gitlab-ci.yml", root / ".gitlab-ci.yml")
            key = native_cache.verify(root)
            with patch.object(native_cache, "recipe_signature", return_value=native_cache.LEGACY_SIGNATURE):
                self.assertEqual(native_cache.cache_key(root), native_cache.LEGACY_KEY)
            recipe = root / "scripts/build_libint.sh"
            original = recipe.read_text()
            recipe.write_text(original + "\n# Optional diagnostics, no build change.\n"
                              "# vibeqc-recipe-hash: lifecycle-only-begin\n"
                              "echo observed\n"
                              "# vibeqc-recipe-hash: lifecycle-only-end\n")
            self.assertEqual(native_cache.verify(root), key)
            # The old GitLab files key would invalidate even this exact no-op.
            recipe.write_text(original)
            for dep in native_cache.DEPENDENCIES:
                with self.subTest(dependency=dep):
                    path = root / f"scripts/build_{dep}.sh"
                    saved = path.read_text()
                    path.write_text(saved + "\ncmake -DNEW_BUILD_OPTION=ON .\n")
                    self.assertNotEqual(native_cache.cache_key(root), key)
                    with self.assertRaisesRegex(ValueError, "recipes changed"):
                        native_cache.verify(root)
                    path.write_text(saved)
            with patch.dict(os.environ, {"VIBEQC_LIBINT_MAX_AM": "6_5_4"}):
                with self.assertRaisesRegex(ValueError, "cache profile"):
                    native_cache.verify_runtime()
            config = (root / ".gitlab-ci.yml").read_text()
            self.assertIn("python scripts/ci/native_cache.py --check --runtime", config)
            self.assertIn("third_party/.build-stamp", config)
            self.assertIn("export CCACHE_DISABLE=1", config)
            self.assertIn('SKBUILD_BUILD_DIR="$CI_PROJECT_DIR/build-clean/$CI_PIPELINE_ID"', config)


if __name__ == "__main__":
    unittest.main()
