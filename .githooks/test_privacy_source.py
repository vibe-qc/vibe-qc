"""Regression coverage for committed-tree preflight and exact public reviews."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest
from unittest.mock import patch

import source_privacy as guard


class SourcePrivacyTests(unittest.TestCase):
    def test_review_is_bound_to_path_value_and_line(self):
        text = "Upstream contact: " + "upstream" + "@" + "public-fixture.org"
        finding = list(guard.findings("LICENSE", text))[0]
        review = json.dumps({"reviews": [{"key": finding["key"], "reason": "Reviewed public upstream attribution"}]}).encode()
        self.assertFalse(guard.scan({"LICENSE": text.encode(), guard.REVIEW_FILE: review})[0])
        for path, content in (("AGENTS.md", text), ("LICENSE", text + " changed"), ("LICENSE", text.replace("upstream@", "another@"))):
            with self.subTest(path=path, content=content):
                self.assertTrue(guard.scan({path: content.encode(), guard.REVIEW_FILE: review})[0])

    def test_empty_review_reason_does_not_approve_finding(self):
        text = "operator" + "@" + "private-fixture.org"
        finding = list(guard.findings("note.md", text))[0]
        review = json.dumps({"reviews": [{"key": finding["key"], "reason": ""}]}).encode()
        self.assertTrue(guard.scan({"note.md": text.encode(), guard.REVIEW_FILE: review})[0])

    def test_python_attributes_are_not_hosts_but_strings_are(self):
        self.assertFalse(list(guard.findings("code.py", "value = threading.local()")))
        host = "node" + ".local"
        self.assertTrue(list(guard.findings("code.py", "host = " + repr(host))))
        self.assertTrue(list(guard.findings("code.py", "# " + host)))

    def test_private_filename_is_checked_for_binary_content(self):
        path = "operator" + "@" + "private-fixture.org" + "/image.png"
        self.assertTrue(guard.scan({path: b"\0opaque asset"})[0])

    def test_second_private_term_and_unicode_casefold(self):
        self.assertTrue(list(guard.findings("note.md", "Straße secret-fixture", ("secret-fixture",))))
        self.assertTrue(list(guard.findings("note.md", "FIRST SECOND", ("first", "second"))))

    def test_committed_bytes_win_over_unstaged_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            address = "operator" + "@" + "private-fixture.org"
            (root / "AGENTS.md").write_text(address)
            subprocess.run(["git", "-C", str(root), "add", "AGENTS.md"], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "Test fixture"], check=True)
            (root / "AGENTS.md").write_text("portable repair")
            self.assertTrue(guard.scan(guard.source_files(root, tree="HEAD"))[0])
            with patch.dict(os.environ, {"VIBE_PRIVACY_REQUIRE_TERMS": "1"}), patch.object(guard, "load_terms", return_value=()), patch.object(guard.Path, "cwd", return_value=root), patch.object(sys, "argv", ["source_privacy.py"]):
                self.assertEqual(guard.main(), 1)


if __name__ == "__main__":
    unittest.main()
