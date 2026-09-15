"""Hermetic Git release selection; run directly without the numerical core."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "scripts" / "_setup_helpers.sh"


class ReleaseRefSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.origin = self.root / "origin"
        self.clone = self.root / "clone"
        self.env = dict(os.environ)
        for key in tuple(self.env):
            if key.startswith("GIT_"):
                self.env.pop(key)
        self.env.update({
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_AUTHOR_NAME": "Test Author", "GIT_COMMITTER_NAME": "Test Author",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        })
        self.git(self.root, "init", "--quiet", "--initial-branch=main", str(self.origin))
        self.initial = self.commit("initial")

    def git(self, cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, env=self.env, check=True,
                              text=True, capture_output=True).stdout.strip()

    def commit(self, text):
        (self.origin / "source.txt").write_text(text + "\n")
        self.git(self.origin, "add", "source.txt")
        self.git(self.origin, "commit", "--quiet", "-m", text)
        return self.git(self.origin, "rev-parse", "HEAD")

    def make_clone(self, *args):
        self.git(self.root, "clone", "--quiet", "--no-local", *args,
                 str(self.origin), str(self.clone))

    def select(self, branch="release", fallback=True):
        return subprocess.run(
            ["bash", "-c", 'set -euo pipefail; . "$1"; vibeqc_checkout_ref "$2" "$3"',
             "release-test", str(HELPERS), branch, "1" if fallback else "0"],
            cwd=self.clone, env=self.env, text=True, capture_output=True)

    def assert_at(self, expected, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git(self.clone, "rev-parse", "HEAD"), expected)

    def test_latest_stable_origin_tag_ignores_prerelease_local_tag_and_branch_shadow(self):
        self.git(self.origin, "tag", "v0.17.9")
        expected = self.commit("stable")
        self.git(self.origin, "tag", "-a", "v0.17.10", "-m", "stable release")
        newest = self.commit("development")
        for tag in ("v0.18.0-rc.1", "v99.0.0.dev1", "v01.0.0"):
            self.git(self.origin, "tag", tag)
        self.make_clone()
        self.git(self.clone, "tag", "v999.0.0")
        self.git(self.clone, "branch", "v0.17.10", newest)
        self.assert_at(expected, self.select())
        self.assertEqual(self.git(self.clone, "rev-parse", "--abbrev-ref", "HEAD"), "HEAD")

    def test_remote_release_branch_takes_precedence_over_newer_tag(self):
        self.git(self.origin, "branch", "release", self.initial)
        self.commit("newer tag")
        self.git(self.origin, "tag", "v9.0.0")
        self.make_clone()
        self.assert_at(self.initial, self.select())
        self.assertEqual(self.git(self.clone, "branch", "--show-current"), "release")

    def test_single_branch_clone_discovers_origin_release_branch(self):
        self.git(self.origin, "branch", "release", self.initial)
        self.commit("main work")
        self.git(self.origin, "tag", "v9.0.0")
        self.make_clone("--single-branch", "--branch", "main")
        self.assertNotIn("origin/release", self.git(self.clone, "branch", "--remotes"))
        self.assert_at(self.initial, self.select())
        self.assertEqual(self.git(self.clone, "branch", "--show-current"), "release")

    def test_existing_local_release_branch_fast_forwards(self):
        self.git(self.origin, "branch", "release", self.initial)
        self.make_clone("--branch", "release")
        self.git(self.origin, "checkout", "--quiet", "release")
        expected = self.commit("next release")
        self.assert_at(expected, self.select())

    def test_explicit_missing_release_ref_does_not_fall_back(self):
        self.git(self.origin, "tag", "v0.17.2")
        self.make_clone()
        result = self.select(fallback=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a known branch", result.stderr)
        self.assertEqual(self.git(self.clone, "branch", "--show-current"), "main")

    def test_explicit_older_tag_remains_exact(self):
        self.git(self.origin, "tag", "v0.17.1")
        self.commit("new release")
        self.git(self.origin, "tag", "v0.17.2")
        self.make_clone()
        self.assert_at(self.initial, self.select("v0.17.1", fallback=False))

    def test_only_prereleases_fails_without_switching_to_main(self):
        self.git(self.origin, "tag", "v0.18.0-rc.1")
        self.make_clone()
        result = self.select()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("neither a release branch nor a stable", result.stderr)
        self.assertEqual(self.git(self.clone, "branch", "--show-current"), "main")

    def test_unavailable_origin_does_not_use_cached_release_tag(self):
        self.git(self.origin, "tag", "v0.17.2")
        self.make_clone()
        self.git(self.clone, "remote", "set-url", "origin", str(self.root / "missing"))
        result = self.select()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.git(self.clone, "branch", "--show-current"), "main")

    def test_moved_origin_tag_is_rejected_without_replacing_cached_tag(self):
        self.git(self.origin, "tag", "v0.17.2")
        self.make_clone()
        self.commit("different release")
        self.git(self.origin, "tag", "--force", "v0.17.2")
        result = self.select()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.git(self.clone, "rev-parse", "refs/tags/v0.17.2"), self.initial)
        self.assertEqual(self.git(self.clone, "branch", "--show-current"), "main")


if __name__ == "__main__":
    unittest.main()
